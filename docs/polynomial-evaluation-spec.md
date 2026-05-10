# Scheme-Agnostic Polynomial Evaluation Spec

## Scope

This document extracts the intrinsic polynomial-evaluation design from
Pro7ech's Lattigo fork and restates it as a scheme-agnostic implementation
specification.

The goal is not to copy Go APIs. The goal is to identify the algorithmic
contract needed to re-implement the same ideas in any FHE library, including
libraries with different ciphertext, plaintext, RNS, or metadata abstractions.

Source baseline:

- Repository: `https://github.com/Pro7ech/lattigo`
- Commit inspected: `cf329e68cfab1d4c3dac5a423598a42aa63b5105`
- Main source files:
  - `he/polynomial_evaluator.go`
  - `he/polynomial.go`
  - `he/power_basis.go`
  - `he/polynomial_encoded.go`
  - `he/polynomial_evaluator_sim.go`
  - `he/hefloat/polynomial_evaluator_sim.go`
  - `he/heint/polynomial_evaluator_sim.go`
  - `he/linear_transformation.go`
  - `he/linear_transformation_evaluator.go`
  - `ring/ring.go`
  - `ring/rns_ring.go`

## Executive Summary

The evaluator is built around three separable ideas:

- A `PowerBasis` caches encrypted powers of the input, indexed by exponent.
- A Paterson-Stockmeyer decomposition turns a high-degree polynomial into
  baby polynomials and combines them through giant steps.
- A host-side constructor decomposes `P(X)` into encoded baby polynomials
  `P_i'(X)` stored in a `BSGSPolynomial<C>` helper.

The evaluator is scheme agnostic because it does not assume CKKS or BFV/BGV
internals. It assumes only a small set of homomorphic operations and a direct
coefficient-encoding policy.

## Required Abstract Capabilities

An implementation needs an evaluator that supports these operations:

- `add(ciphertext, ciphertext_or_plain, out)`
- `sub(ciphertext, ciphertext_or_plain, out)`
- `mul(ciphertext, ciphertext_or_plain, out)`
- `mul_new(ciphertext, ciphertext_or_plain) -> ciphertext`
- `mul_then_add(ciphertext, coefficient_or_plain, accumulator)`
- `new_ciphertext(template_or_metadata) -> ciphertext`
- access to ciphertext metadata and slot count

For `poulpy-ckks`, multiplication already includes the tensor/key-switching
behavior needed by the evaluator, so the polynomial layer does not expose
standalone post-multiplication hooks.

## Data Model

### Polynomial

A plaintext polynomial stores:

- coefficients in one basis
- basis tag, usually monomial or Chebyshev
- maximum original degree
- parity metadata: odd, even, or general
- optional coefficient-encoding metadata chosen by the caller or high-level API

The polynomial basis matters:

- In monomial basis, `X^a * X^b = X^(a+b)`.
- In Chebyshev basis, `T_a(x) * T_b(x) = (T_(a+b)(x) + T_|a-b|(x)) / 2`.

The power-basis generator therefore uses different recurrence formulas for
monomial and Chebyshev powers.

### BSGSPolynomial Helper

For Poulpy, the evaluator does not need a separate collection or slot-mapping
abstraction for polynomials.

Instead, the host-side constructor takes a single plaintext polynomial `P(X)`
and constructs a `BSGSPolynomial<C>` helper. Internally this helper stores the
already-encoded baby polynomials:

```text
BSGSPolynomial<C> {
  basis
  degree
  baby_steps: Vec<C>
}
```

where, in `poulpy-ckks`, `C` is bounded like:

```rust
C: GLWEToBackendRef<BE> + CKKSInfos + GLWEInfos
```

In practice `C` is a `CKKSPlaintext`, or a backend/device-owned equivalent
uploaded from the host representation.

The helper exposes indexed access:

```text
bsgs.get(i) -> &baby_steps[i]
```

Each `baby_steps[i]` encodes one baby polynomial `P_i'(X)`. The evaluator does
not inspect the original coefficient vector of `P(X)`. It only retrieves
encoded baby polynomials by index.

### Power Basis

A power basis is a cache:

```text
PowerBasis {
  basis: Monomial | Chebyshev | ...
  value: map exponent -> ciphertext encrypting basis_element_exponent(input)
}
```

It is initialized with exponent `1`:

```text
value[1] = clone(input_ciphertext)
```

The cache must never silently replace a populated power with a different
ciphertext.

## Power Generation

### SplitDegree

To generate `X^n`, split `n` into `a + b = n`, then compute the power from
the two smaller powers.

The Lattigo rule is:

```text
if n is a power of two:
  a = n / 2
  b = n / 2
else:
  k = floor(log2(n - 1))
  a = 2^k - 1
  b = n + 1 - 2^k
```

This rule has two purposes:

- For powers of two, it gives optimal multiplicative depth.
- For other degrees, it favors odd factors, which is beneficial for Chebyshev
  evaluation and error behavior.

### Recursive Generation

Pseudocode:

```text
gen_power(n):
  if value[n] exists:
    return

  (a, b) = split_degree(n)
  gen_power(a)
  gen_power(b)

  value[n] = mul_new(value[a], value[b])

  if basis == Chebyshev:
    c = abs(a - b)
    value[n] = 2 * value[n]
    if c == 0:
      value[n] = value[n] - 1
    else:
      gen_power(c)
      value[n] = value[n] - value[c]
```

Important semantics:

- Multiplication owns any tensoring, key-switching, normalization, or metadata
  update required by the target library.
- Chebyshev generation applies the identity
  `T_(a+b) = 2*T_a*T_b - T_|a-b|`.
- `T_0` is not stored as a ciphertext; it is treated as constant `1`.

### Power Set Needed by Polynomial Evaluation

Let:

```text
log_degree = bit_length(polynomial_degree)
log_split = optimal_split(log_degree)
base = 2^log_split
```

Here `bit_length(d) = floor(log2(d)) + 1`, equivalently
`ceil(log2(d + 1))`. This distinction matters at power-of-two degrees.

Populate the power basis with:

- all recursively required powers up to `2^(log_degree - 1)`
- intermediate powers `base - 1` down to `3`
- only powers matching polynomial parity if the polynomial is purely odd or
  purely even

The largest call recursively generates lower powers. Calling powers in
descending order improves cache reuse and avoids recomputation.

A conservative implementation may also materialize `2^log_degree`, but the
real ciphertext cache only needs the powers that are actually referenced by the
decomposition and baby-step evaluation.

## Optimal Split

The split uses a power-of-two baby-step base:

```text
log_split = floor(log_degree / 2)
a = 2^log_split + 2^(log_degree - log_split) + log_degree - log_split - 3
b = 2^(log_split + 1) + 2^(log_degree - log_split - 1) + log_degree - log_split - 4
if a > b:
  log_split += 1
```

The output base is:

```text
base = 2^log_split
```

This heuristic balances the number of baby powers and giant-step combinations.
It is not tied to a scheme. Its input is the bit length defined above, not a
floating-point logarithm rounded independently.

## Paterson-Stockmeyer Decomposition

### Mathematical Shape

A polynomial is recursively represented as:

```text
P(X) = Q(X) * B(X) + R(X)
```

where `B(X)` is a power-basis element such as `X^n` or `T_n(X)`, and `R` has
degree less than the split power.

For monomial basis:

```text
P(X) = X^n * Q(X) + R(X)
```

For Chebyshev basis, factorization must account for product identities. When
extracting coefficients above degree `n`, the quotient terms are doubled and
mirrored terms are subtracted from the remainder. This preserves the equality
under Chebyshev multiplication.

### Decomposition Shape

For the intended implementation, decomposition does not recursively simulate
future metadata changes. It only computes the baby-step shape and the powers
that must exist in the power basis.

For monomial polynomials, the decomposition can be represented as fixed-width
chunks of size `base = 2^log_split`:

```text
decompose_monomial(coeffs, base):
  babies = []
  for start in 0, base, 2*base, ...:
    chunk = coeffs[start : start + base]
    pad chunk with zeros to length base if desired
    babies.push(chunk)

  return babies ordered from high degree to low degree
```

Each baby polynomial is evaluated as:

```text
baby_j(X) = sum_i chunk_j[i] * X^i
```

and giant-step processing combines those baby outputs with powers
`X^base`, `X^(2*base)`, etc. The exact storage order can differ as long as the
giant-step combiner receives baby steps in ascending degree order.

For Chebyshev polynomials, decomposition must still respect Chebyshev product
identities. That affects coefficient factorization, but it does not require a
metadata simulation pass.

## Evaluation Pipeline

The complete evaluator follows this sequence:

1. Construct or receive a `BSGSPolynomial<C>` helper containing encoded baby
   polynomials `P_i'(X)`.
2. Convert the encrypted input into a `PowerBasis`, or reuse the provided
   `PowerBasis`.
3. Check that the input metadata/budget can support the requested polynomial
   depth.
4. Populate missing ciphertext powers.
5. Evaluate each encoded baby polynomial from the power basis.
6. Process baby results through giant-step combinations.

High-level pseudocode:

```text
construct_bsgs_polynomial(P, coefficient_metadata) -> BSGSPolynomial<C>:
  shape = paterson_stockmeyer_shape(P.degree)
  baby_steps = encode_baby_polynomials(P, shape, coefficient_metadata)
  return BSGSPolynomial { basis: P.basis, degree: P.degree, baby_steps }

evaluate_polynomial(evaluator, input, bsgs, coefficient_metadata):
  pb = input is PowerBasis ? input : new_power_basis(input, bsgs.basis)

  required_depth = bsgs.depth()
  if not evaluator.has_enough_budget(pb.value[1], required_depth):
    error

  pb.populate_for_bsgs(evaluator, bsgs)

  return evaluator.evaluate_paterson_stockmeyer(bsgs, pb)
```

## Baby-Step Evaluation

A baby step evaluates one encoded baby polynomial from the
`BSGSPolynomial<C>` helper:

```text
P_i'(X) = sum_k c_{i,k} * basis_power_k(X)
```

using the already-populated power basis.

Pseudocode:

```text
evaluate_baby_step(i, evaluator, bsgs, pb):
  encoded_baby = bsgs.get(i)
  value = evaluate_encoded_baby_from_power_basis(
    evaluator,
    encoded_baby,
    pb
  )

  return BabyStep {
    degree: bsgs.baby_degree(i),
    value
  }
```

The Lattigo ordering stores baby steps in reverse while iterating, so the
subsequent giant-step merge sees them in ascending degree order.

## Evaluating an Encoded Baby Polynomial From a Power Basis

The evaluator computes:

```text
sum_k c_k * X_k
```

where `X_k` is the cached encrypted basis element.

Algorithm:

1. Determine the highest relevant coefficient degree.
2. If the original polynomial is even or odd, skip irrelevant parity terms.
3. Inspect required cached powers to choose the maximum output ciphertext
   degree.
4. Allocate an output ciphertext from a deterministic accumulator template.
5. Add the constant term if the polynomial has even terms.
6. Iterate coefficients from high to low and call `mul_then_add`.

Pseudocode:

```text
evaluate_encoded_baby_from_power_basis(eval, encoded_baby, pb):
  X = pb.value
  even = encoded_baby.is_even()
  odd = encoded_baby.is_odd()

  max_coeff = encoded_baby.degree()
  if even and not odd and (max_coeff + 1) is odd:
    max_coeff = max(0, max_coeff - 1)

  out_degree = max(X[k].degree for k in 1..max_coeff if X[k] exists)
  if max_coeff < 1:
    out_degree = 0

  res = new_ciphertext(accumulator_template(encoded_baby, X, out_degree))
  copy_metadata(res, X[1])

  if even:
    add(res, encoded_baby.coeff(0), res)
  for k from max_coeff down to 1:
    if coefficient_parity_is_needed(k, even, odd):
      mul_then_add(X[k], encoded_baby.coeff(k), res)

  return res
```

`mul_then_add` is responsible for applying the scheme's normal metadata and
capacity rules. Coefficients are not adjusted differently according to their
future position in the Paterson-Stockmeyer tree.

## Encoded Baby Polynomials

The host constructor encodes each baby polynomial into an object `C`.

Construction requires:

- a coefficient metadata/precision policy
- the plaintext shape

For each baby polynomial `P_i'(X)`:

```text
baby_steps[i] = encode(P_i'(X), coefficient_metadata)
```

Every coefficient in the baby polynomial can use the same metadata unless the
caller deliberately chooses a more specialized encoding policy. There is no
recursive coefficient-compensation step.

Evaluation then retrieves the encoded baby polynomial by index:

```text
encoded_baby = bsgs.get(i)
evaluate_encoded_baby_from_power_basis(encoded_baby, power_basis)
```

Then the resulting baby ciphertexts are processed by the same giant-step
combiner.

## Giant-Step Combination

After all baby steps are evaluated, repeatedly combine adjacent baby results
until one remains.

Each combination has the form:

```text
combined = even + odd * X^deg
```

where `deg` is the next power of two above the baby degree:

```text
deg = 1 << bit_length(baby_degree)
```

In the Lattigo code this is computed as:

```text
deg = 1 << bit_length(baby_degree)
```

### Scheduling

For each pass over the baby list:

```text
for i in 0..len(babies)-1:
  if i == last:
    giant_steps[i] = 2
  else if babies[i].degree == babies[i+1].degree:
    giant_steps[i] = 1
    skip next index
```

Meaning:

- `0`: no action this pass
- `1`: combine this baby with the next baby
- `2`: last unpaired baby; promote its degree to match the previous degree

After the pass, discard entries consumed by combinations.

### Combining Two Baby Steps

Pseudocode:

```text
evaluate_giant_step(i, giant_steps, babies, eval, pb):
  if giant_steps[i] == 2:
    babies[i].degree = babies[i - 1].degree
    return

  if giant_steps[i] == 1:
    even = babies[i]
    odd = babies[i + 1]
    deg = 1 << bit_length(even.degree)
    evaluate_monomial(even.value, odd.value, pb.value[deg], eval)
    odd.degree = 2 * deg - 1
    babies[i] = nil
```

The result is written into the odd baby. The even baby is discarded.

### Monomial Combine

`evaluate_monomial(a, b, xpow)` computes:

```text
b = a + b * xpow
```

Pseudocode:

```text
evaluate_monomial(a, b, xpow, eval):
  mul(b, xpow, b)
  add(b, a, b)
```

The multiplication and addition kernels own metadata alignment, capacity
tracking, tensor/key-switching, and normalization. The polynomial evaluator does
not adjust coefficients to force future giant-step metadata to match.

## Metadata and Capacity Policy

The polynomial evaluator should not contain a recursive metadata planner.

Instead:

- coefficient plaintexts are encoded with a direct metadata/precision policy
- ciphertext-ciphertext multiplication applies the scheme's usual metadata
  update
- ciphertext-plaintext multiplication applies the scheme's usual metadata
  update
- addition aligns operands according to the scheme's usual addition rule
- final output fitting is handled by the destination/capacity rule of the
  target library

This is especially natural for implementations whose ciphertext modulus is a
power-of-two torus width, because coefficient encoding does not require the
RNS coefficient compensation used by Lattigo.

## Coefficient Setting in RNS Rings

Polynomial evaluation ultimately depends on encoding coefficient values into
plaintext/ring polynomials. The intrinsic coefficient-setting rule is simple:

For a single modulus `q`:

```text
poly[j] = coeffs[j] mod q
```

For an RNS basis with moduli `q_0, ..., q_L`:

```text
for each modulus q_i:
  for each coefficient j:
    poly_i[j] = coeffs[j] mod q_i
```

Negative coefficients must be represented by their canonical residue modulo
each `q_i`. Coefficients beyond the provided slice are left unchanged unless
the caller zeroes the polynomial first; a robust API should either require a
full-length slice or explicitly zero untouched coefficients.

This coefficient-setting layer is intentionally below polynomial evaluation.
It is used by encoders and plaintext builders. The evaluator receives already-
encoded baby polynomials through `BSGSPolynomial<C>`.

## Reimplementation Checklist

A complete implementation should provide:

- A polynomial type with basis, degree, parity, and optional coefficient-
  encoding metadata.
- A `BSGSPolynomial<C>` helper storing encoded baby polynomials in `Vec<C>`.
- A power-basis cache keyed by exponent.
- `split_degree` and recursive power generation for monomial and Chebyshev
  bases.
- An `optimal_split` heuristic and Paterson-Stockmeyer decomposition shape.
- Baby-step evaluation from a power basis.
- Giant-step processing that relies on normal multiplication/addition metadata
  rules.
- Host-side construction and optional device upload of encoded baby-step
  plaintexts.
- RNS coefficient setting that maps big integers to residues per modulus.
- Tests comparing encrypted evaluation against cleartext polynomial evaluation
  for monomial, Chebyshev, odd, even, and constant polynomials.

## Common Pitfalls

- Do not compute powers only linearly as `X^2, X^3, ...`; the recursive split
  is part of the depth and precision strategy.
- Do not reintroduce recursive coefficient compensation unless the target
  modulus model requires it.
- Do not hard-code CKKS metadata rules in the generic PS decomposition. Keep
  metadata effects in the arithmetic kernels.
- Do not treat Chebyshev powers as monomials. The recurrence needs the
  subtraction term.
- Do not make the evaluator recover the original coefficient vector from
  `P(X)`. It should consume the encoded BSGS helper.
- Do not ignore parity metadata; it avoids unnecessary powers and
  multiplications.
- Do not ignore destination capacity. Final output fitting still belongs in
  the target library's normal destination/capacity path.

## Minimal Correctness Tests

For any target FHE library, test these cases first:

1. Constant polynomial.
2. Linear polynomial.
3. Dense monomial polynomial of degree 7 or 15.
4. Dense polynomial with degree just above a power-of-two split boundary.
5. Pure odd polynomial.
6. Pure even polynomial.
7. Chebyshev polynomial requiring `T_(a+b) = 2*T_a*T_b - T_|a-b|`.
8. Evaluation from a precomputed power basis reused across two polynomials.
9. Power-basis reuse across multiple polynomial evaluations.
10. Host construction plus device upload of `BSGSPolynomial<C>`.

These tests should compare decrypted outputs to cleartext evaluation using the
same basis and input-domain change of basis.
