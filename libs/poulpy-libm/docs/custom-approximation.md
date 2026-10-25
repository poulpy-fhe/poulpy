# Approximating a custom function

`poulpy-libm` intentionally provides named approximations of libm functions.
Custom function fitting belongs to `poulpy-ckks` and is not part of the stable `poulpy-libm` API.
Use a named Poulpy-libm plan whenever the required function is already available.
Use the workflow below only for an application-specific function.

## Select the domain

Choose the smallest domain that contains every encrypted input expected by the circuit.
A narrower domain usually needs a lower polynomial degree for the same target error.
The approximation does not enforce its domain during encrypted evaluation.

For a continuous interval `[a, b]`, use `degree_for_precision` from `poulpy_ckks::approximation`.
Use `minimax` instead when the polynomial degree is fixed in advance.

```rust
degree_for_precision(
    function,
    a,
    b,
    parity,
    target_bits,
    max_degree,
    strategy,
)
```

Select `Parity::Even` or `Parity::Odd` only when the domain is symmetric about zero and the function has the matching parity.
Use `Parity::Full` otherwise.

## Fit disjoint intervals

Some functions are only needed on separated input ranges.
A common example is a function with an excluded neighborhood around a pole or discontinuity.
Use `degree_for_precision_multi_interval` from `poulpy_ckks::approximation` for this case.

```rust
degree_for_precision_multi_interval(
    function,
    &intervals,
    parity,
    target_bits,
    max_degree,
    strategy,
)
```

The intervals must be nonempty, ordered, and pairwise disjoint.
Even and odd fits require the complete interval union to be symmetric about zero.
The polynomial is normalized over the convex hull of the intervals.
Excluded gaps are not included in the fitting error objective.
Encrypted inputs must not enter those gaps.

Use `minimax_multi_interval` when the degree is fixed in advance.
The corresponding `*_with` functions accept explicit `RemezOptions` for advanced fitting control.

## Prepare the polynomial

The degree selection functions return a `DegreeChoice`.
Its `minimax.poly` field contains the host polynomial.
Prepare that polynomial for CKKS evaluation with `PolynomialApproximation::from_polynomial`.

```rust
PolynomialApproximation::from_polynomial(
    &choice.minimax.poly,
    base2k,
    coefficient_layout,
    strategy,
    &host_module,
)
```

Choose the coefficient layout with the same care as a Poulpy-libm plan.
Its scale contributes to plaintext multiplication cost and final numerical error.

## Transfer and evaluate

Transfer the prepared plaintexts to the selected backend with `map_plaintexts` and the application's normal Poulpy host-to-backend transfer path.
Do not copy the deterministic key generation or encryption helpers from the Poulpy-libm walkthroughs into a production application.

Size scratch memory with `ckks_approximation_tmp_bytes` using the actual input layout, output layout, tensor key layout, and prepared approximation.
Evaluate with `ckks_eval_approximation`.

```rust
module.ckks_eval_approximation(
    &mut output,
    &input,
    &plan,
    &tensor_key,
    &mut scratch,
)
```

Check `plan.consumed_bits(input_log_delta)` before evaluation.
The input budget must be strictly larger than that value.
Use `plan.depth()` when composing the approximation with other circuits.

## Validate the result

The fitted error is a numerical estimate and is not a certified bound.
Measure the approximation error over every fitted interval with representative application inputs.
Then validate encryption, evaluation, and decryption with the production CKKS parameters.

Custom approximation does not select secure parameters and does not bootstrap automatically.
Account for the complete circuit and any bootstrap stages when choosing the ciphertext modulus.
