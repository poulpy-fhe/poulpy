# Deterministic CKKS encoding

CKKS preparation and encoding use one numerical contract for `f64` and `Quad` (IEEE binary128).
Selecting a CPU backend, operating system, or the `libquadmath` feature must not change encoded plaintext bytes.
A new backend must implement this contract at the four `CKKSEncodingOps` boundaries.

## Why this approach

Reproducibility requires fixing the transform's arithmetic graph, the setup constants, and the float/integer conversion together.
A shared FFT alone still leaves platform-dependent twiddles and bootstrap polynomial coefficients.
Extra precision, rounding tolerances, or fewer quantization bits reduce observed mismatches but cannot guarantee equality when a coefficient approaches a rounding boundary.
An exact or adaptively correctly rounded transform would introduce substantially more machinery and variable cost.
The chosen contract instead specifies one reproducible floating-point algorithm and allows backends to accelerate that algorithm.

This change sits above oracle separation because the independent encoding implementation belongs in `poulpy-cpu-oracle`.
The oracle's ring transform retains its independent decomposition; production ring arithmetic does not acquire CKKS rounding constraints.

## Numerical rules

- Basic arithmetic rounds to nearest, ties to even, at the selected scalar precision.
  Subnormals are retained.
  The floating-point environment must use this rounding mode with flushing of subnormals disabled.
- Each FFT multiply, add, and subtract is a separate rounding operation.
  Fusing a multiply and an add, reassociating expressions, or changing the butterfly decomposition changes the result and is not permitted.
- Setup uses `CKKSFloat` methods instead of platform transcendentals.
  Binary64 uses pinned Rust `libm` with architecture dispatch disabled even when another dependency enables its `arch` feature.
  Binary128 uses the existing pure Rust Astro-float implementation at 192 working bits, followed by round-to-even conversion to 113 significand bits.
  This specifies a reproducible approximation; it does not claim correctly rounded transcendental functions for every argument.
- Integer powers use the specified exponentiation-by-squaring loop.
  The general `Quad` math API retains its platform/`libquadmath` routing; CKKS preparation explicitly selects the canonical methods.
- Quantization rounds the exact dyadic value `x * 2^log_delta` to an integer, ties away from zero.
  It rejects NaN, infinity, and signed integer overflow.
  It does not materialize a floating-point scale or use saturating casts.
- Dequantization rounds `integer * 2^-log_delta` once, ties to even.
  This includes subnormal results and avoids double rounding at underflow.

The codec operates directly on IEEE sign, exponent, and significand bits.
For example, the smallest positive binary128 subnormal at scale `2^16494` encodes as `1`, although that scale cannot itself be represented in binary128.

## Transform

The slot permutation and normalization are unchanged.
For `m` complex slots:

1. Scatter both planar halves using the bit-reversed indices of powers of the CKKS Galois generator.
2. Apply the unnormalized inverse negacyclic transform.
3. Multiply every coefficient by the exact power of two `1/m`.

Decoding applies the forward transform followed by the inverse permutation.
The canonical butterfly graph is the existing portable REIM graph, including its radix-4 decomposition, small transforms, and recursion above `m=2048`.
Its twiddles use the same dyadic phase expressions and rounded `2*PI` as the portable tables, evaluated with canonical sine and cosine.
Quarter-turn butterflies use the specified coordinate/sign operations, rather than fresh trigonometric evaluation at a shifted angle.

`poulpy-cpu-oracle/src/ckks_encoding_fft.rs` independently constructs a sequence of butterflies for this graph.
It neither imports production FFT kernels nor production twiddle tables.
Forward execution walks the sequence; inverse execution walks it backwards.
This is separate from the oracle's independent ring FFT, whose different decomposition remains useful for checking integer polynomial arithmetic.

Production portable encoding reuses the existing kernels with canonical tables.
AVX2 and NEON encoding use this implementation.
AVX-512 encoding uses the same SIMD kernels with their FMA policy disabled at compile time.
Ring FFTs continue to use their original tables and fused kernels.

A GPU implementation may use different memory layouts and scheduling, but must preserve the butterfly dependencies and rounding operations.
Uploading canonical setup constants is also valid.
Matching approximate decoded values or matching only the final integer rounding on a small corpus is insufficient.

## Preparation and compatibility

DFT roots and factor scaling, EvalMod interpolation nodes and amplitudes, Han–Ki degree selection, PaCo/SHIP phases, and approximation setup use the canonical math methods.
Generic polynomial interpolation accepts a cosine callback so that CKKS can choose its numerical policy without teaching Core about CKKS.

The contract assumes identical input scalar bits, parameters, ciphertexts, and keys.
A user-supplied approximation callback must itself produce identical values.
Equality is between backends at the same precision; `f64` and binary128 are not expected to produce identical encodings to each other.
Custom scalar implementations must now implement `CKKSFloat`; implementations are provided for `f64` and `Quad`.

These bytes can differ from earlier versions.
Rebuild cached bootstrap parameters when adopting the canonical encoding.
Future changes to the numerical algorithms must explicitly review this compatibility boundary and the frozen fixtures; dependency upgrades must not silently redefine it.

## Verification

- Independent oracle/portable differential tests cover every power-of-two degree through 65,536, both transform directions and both precisions.
  Inputs include zeros, signed zeros, impulses, cancellation, subnormals, large finite values, and deterministic dense vectors.
- AVX-512 runs the same differential corpus against the independent oracle.
- Backend-generic fixtures cover all four encoding primitives, sparse plaintext storage, scales 30/60/110, nontrivial DFT scaling, and all five EvalMod variants.
  Fixture hashes use canonical little-endian active data.
- Standard bootstrap fixtures check ciphertext words at input, C2S real and imaginary outputs, EvalMod real and imaginary outputs, and final output.
  Existing accuracy and manual/orchestrated-circuit checks remain active.
- Codec tests cover half-integers, signed limits, non-finite values, the normal/subnormal transition, and double-rounding counterexamples.
  A binary128 arithmetic reference checks 50,000 binary64 quantizations.
- Native CI is configured to run the same fixtures on Linux x86-64, Intel macOS, Apple Silicon, and Linux AArch64.
  The portable lane additionally enables `libquadmath` to check that the feature cannot change CKKS output.

Local validation ran the workspace and x86 backend suites, including AVX-512, and the `libquadmath` fixture check.
Intel macOS, Apple Silicon, and Linux AArch64 passed cross-compilation checks; their native fixture execution remains a CI check.

Useful local commands:

```sh
cargo test -p poulpy-ckks --profile ci numerics::tests
cargo test -p poulpy-cpu-portable --profile ci --features enable-ckks --test ckks_determinism
cargo test -p poulpy-cpu-portable --profile ci --features enable-ckks encoding_determinism
cargo test -p poulpy-cpu-portable --profile ci --features enable-ckks bootstrapping_standard_e2e
RUSTFLAGS='-C target-feature=+avx2,+fma,+avx512f,+avx512ifma,+avx512vl' \
  cargo test -p poulpy-cpu-avx512 --profile ci --features enable-ifma,enable-ckks encoding_fft_matches_oracle
```

## Performance

The comparison uses the oracle-separation parent (`42e3d25a`), release builds with the pinned toolchain, and `FFT64Avx512` on a Ryzen 9950X pinned to one CPU.
Both versions consume identical saved scalar bytes for dense inputs and bootstrap DFT diagonals.
The table gives microseconds per encoding with cached plans and reused buffers, taking the median of three process medians with nine timing samples per process.
Samples contain 100 encodings for binary64 and 10 for binary128; the before/after process order alternates.

| Scalar | Degree | Input | Before (µs) | After (µs) |
| --- | ---: | --- | ---: | ---: |
| `f64` | 2,048 | Dense | 30.03 | 29.96 |
| `f64` | 2,048 | DFT diagonals | 29.40 | 28.57 |
| `f64` | 65,536 | Dense | 1,008.95 | 1,019.51 |
| `f64` | 65,536 | DFT diagonals | 1,002.10 | 944.00 |
| `Quad` | 2,048 | Dense | 748.60 | 675.56 |
| `Quad` | 2,048 | DFT diagonals | 710.11 | 648.23 |
| `Quad` | 65,536 | Dense | 36,092.36 | 33,464.83 |
| `Quad` | 65,536 | DFT diagonals | 31,948.62 | 29,589.11 |

The unfused AVX-512 encoding path is about 4–5% faster than the canonical scalar fallback on these workloads and passes the full differential corpus.
Binary128 benefits from replacing floating-point scaling and conversion with the exact bit codec.
These measurements cover encoding, not total bootstrap latency.

Canonical binary128 transcendental evaluation increases cold setup cost.
The first encoding, including creation of the module's geometric plan family, takes about 37 ms instead of 2.2 ms at degree 2,048, and about 1.2 s instead of 80 ms at degree 65,536.
Subsequent calls reuse those plans.
These cold timings are separate from the cached-plan table; no claim is made about performance on ARM, macOS, or GPUs.
