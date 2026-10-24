# 🐙 poulpy-libm

`poulpy-libm` adds familiar mathematical functions to encrypted CKKS values handled by [`poulpy-ckks`](https://github.com/poulpy-fhe/poulpy).
It builds function-specific plans and circuits on Poulpy's reusable approximation machinery, while leaving fitting, degree selection, interval-mapped polynomial evaluation, encryption, keys, ciphertexts, and arithmetic to Poulpy.

## API

- Arithmetic: `fma`, `scalbn`, and `ldexp`.
- Exponential and logarithmic: `exp`, `exp2`, `exp10`, `expm1`, `log`, `log2`, `log10`, and `log1p`.
- Powers and roots: `pow`, `powi`, `sqrt`, `cbrt`, `hypot`, reciprocal, and `rsqrt`.
- Trigonometric: `sin`, `cos`, `sincos`, `tan`, `asin`, `acos`, `atan`, and `atan2`.
- Hyperbolic and error functions: `sinh`, `cosh`, `tanh`, `asinh`, `acosh`, `atanh`, `erf`, and `erfc`.
- Special functions: `tgamma`, `lgamma`, `j0`, `j1`, `jn`, `y0`, `y1`, and `yn`.
- Comparison: `fabs`, `fdim`, `copysign`, `fmax`, `fmin`, predicates, selection, and clamp.
- Packed operations: sum, `fmax` and `fmin` reductions, and `softmax`.

Function names follow `libm` when the usual meaning can be implemented sensibly with approximate encrypted arithmetic.
Operations that depend on exact IEEE representations or discontinuous rounding are outside the current scope.

## Use

Most nonlinear functions use a plan that stores a polynomial approximation for a chosen input interval and precision.
Create the plan before encryption, transfer its prepared values to the selected Poulpy backend, allocate the reported temporary memory, and evaluate the function on a ciphertext.
The plan can be reused for any ciphertext that uses compatible parameters and stays inside the same input interval.
Approximation targets, coefficient scales, reduction steps, and iterative refinements can be tuned per plan to trade accuracy for modulus consumption.

`poulpy-libm` does not select CKKS security parameters or insert bootstrapping automatically.
Applications must choose parameters for their security target, reserve enough modulus for each planned circuit, and bootstrap explicitly when required.

Common imports are available through `poulpy_libm::prelude`.
The [API guide](docs/api.md) lists supported intervals and circuit costs.
Custom functions outside the libm scope are covered separately in the [custom approximation guide](docs/custom-approximation.md), which uses Poulpy's fitting API directly.

## Examples

The runnable walkthroughs explain plan creation, encryption, evaluation, decryption, and result verification.
The inference and signal examples show how several operations compose into application-level circuits.

```sh
cargo run --release --features ref --example exp_walkthrough
cargo run --release --features ref --example trig_walkthrough
cargo run --release --features ref --example inference_walkthrough
cargo run --release --features ref --example signal_walkthrough
```

## Test

```sh
RUSTFLAGS='-C target-cpu=native' cargo test --release --features all-backends
```

The backend-generic suite checks numerical precision and exact budget consumption on the reference, AVX, AVX-512, and IFMA backends.

## Benchmark

The same `exp`, `log`, and `pow` workloads are available for every backend.
Run the target matching the benchmark host.

```sh
RUSTFLAGS='-C target-cpu=native' cargo bench --features ref --bench functions_ref
RUSTFLAGS='-C target-cpu=native' cargo bench --features avx --bench functions_avx
RUSTFLAGS='-C target-cpu=native' cargo bench --features avx512 --bench functions_avx512
RUSTFLAGS='-C target-cpu=native' cargo bench --features ifma --bench functions_ifma
```
