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
They live in `poulpy-cpu-portable` and run on its NTT backend.

```sh
cargo run --release -p poulpy-cpu-portable --features enable-ckks --example libm_exp
cargo run --release -p poulpy-cpu-portable --features enable-ckks --example libm_trig
cargo run --release -p poulpy-cpu-portable --features enable-ckks --example libm_inference
cargo run --release -p poulpy-cpu-portable --features enable-ckks --example libm_signal
```

## Test

The backend-generic suite in `test_suite`, compiled with `test-utils`, checks numerical precision and exact budget consumption.
Backend crates register it with `libm_backend_test_suite!`, as the portable, AVX2, AVX-512, IFMA and NEON backends of the workspace do with `enable-ckks`.
CI runs it on the portable, native AVX2 and NEON backends. The AVX-512 jobs filter it out, since their results are byte-identical to the other backends'.

```sh
cargo test -p poulpy-cpu-portable --lib --profile ci --features enable-ckks -- libm_tests
```

## Benchmark

The `exp`, `log`, and `pow` workloads live in `poulpy-bench` (`poulpy_bench::libm::suites`), whose full, standard and light tiers the tier binaries of each backend crate register on its NTT backend.

```sh
cargo bench -p poulpy-cpu-portable --bench light --features enable-ckks,enable-bin-fhe -- libm
RUSTFLAGS='-C target-cpu=native' cargo bench -p poulpy-cpu-avx --bench light --features enable-avx,enable-ckks,enable-bin-fhe -- libm
RUSTFLAGS='-C target-cpu=native' cargo bench -p poulpy-cpu-avx512 --bench light --features enable-avx512f,enable-ckks,enable-bin-fhe -- libm
cargo bench -p poulpy-cpu-arm --bench light --features enable-neon,enable-ckks,enable-bin-fhe -- libm
```
