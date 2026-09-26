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
- Roots of unity are correctly rounded; see [Roots of unity](#roots-of-unity).
- Other setup constants use `CKKSFloat` methods instead of platform transcendentals.
  Binary64 uses pinned Rust `libm` with architecture dispatch disabled even when another dependency enables its `arch` feature.
  Binary128 uses the existing pure Rust Astro-float implementation at 192 working bits, followed by round-to-even conversion to 113 significand bits.
  This specifies a reproducible approximation; it does not claim correctly rounded transcendental functions for every argument.
- Integer powers use the specified exponentiation-by-squaring loop.
  The general `Quad` math API retains its platform/`libquadmath` routing; CKKS preparation explicitly selects the canonical methods.
- Quantization rounds the exact dyadic value `x * 2^log_delta` to an integer, ties away from zero.
  It rejects NaN, infinity, and signed integer overflow.
  The general codec operates on the IEEE bits without materializing a floating-point scale.
  The binary64-to-`i64` fast path uses an exactly representable power of two for scales up to 1,023, rounds once, and checks the signed range before conversion.
  Scaling up cannot lose significand bits unless it overflows, which is rejected; larger scales use the general codec.
- Dequantization rounds `integer * 2^-log_delta` once, ties to even.
  This includes subnormal results and avoids double rounding at underflow.

The general codec operates directly on IEEE sign, exponent, and significand bits.
For example, the smallest positive binary128 subnormal at scale `2^16494` encodes as `1`, although that scale cannot itself be represented in binary128.

## Transform

The slot permutation and normalization are unchanged.
For `m` complex slots:

1. Scatter both planar halves using the bit-reversed indices of powers of the CKKS Galois generator.
2. Apply the unnormalized inverse negacyclic transform.
3. Multiply every coefficient by the exact power of two `1/m`.

Decoding applies the forward transform followed by the inverse permutation.
The canonical butterfly graph is the existing portable REIM graph, including its radix-4 decomposition, small transforms, and recursion above `m=2048`.
The twiddle at turn `t`, an exact multiple of `1/(4m)`, is the correctly rounded root of unity `(cos 2πt, sin 2πt)`.
Quarter-turn butterflies use the specified coordinate/sign operations, rather than fresh trigonometric evaluation at a shifted angle.

`poulpy-cpu-oracle/src/ckks_encoding_fft.rs` independently constructs a sequence of butterflies for this graph.
It imports neither production FFT kernels nor production twiddle tables, and derives its roots of unity independently.
Forward execution walks the sequence; inverse execution walks it backwards.
This is separate from the oracle's independent ring FFT, whose different decomposition remains useful for checking integer polynomial arithmetic.

Production portable encoding reuses the existing kernels with canonical tables.
NEON encoding uses this implementation.
AVX2 and AVX-512 encoding use their existing SIMD kernels with their FMA policy disabled at compile time.
Ring FFTs continue to use their original tables and fused kernels.
CPU encoding plans initialize each transform dimension on first use; initialized tables remain owned by the module.

A GPU implementation may use different memory layouts and scheduling, but must preserve the butterfly dependencies and rounding operations.
Uploading canonical setup constants is also valid.
Matching approximate decoded values or matching only the final integer rounding on a small corpus is insufficient.

## Roots of unity

`CKKSFloat::ckks_root_of_unity(k, log_order)` returns `(cos, sin)` of `2πk / 2^log_order`, each rounded to nearest, ties to even, from the exact value.
This definition does not depend on any math library.

`poulpy-ckks/src/numerics/cos_quadrant_f64.bin` and `cos_quadrant_f128.bin` hold the little-endian bits of `cos(2πi / 2^17)` for `i` in `0..=2^15` (256 KiB and 512 KiB).
Every other root of order at most `2^17` is a table entry up to an exact sign change or coordinate swap: `cos` is even, `cos(π - θ) = -cos θ`, and `sin θ = cos(θ - π/2)`.
Correct rounding commutes with these symmetries, so the derived coordinates are themselves correctly rounded.
Cardinal points are exact, with positive zeros.

Larger orders are first reduced to lowest terms.
Roots that remain beyond the tables are evaluated with Astro-float: `π`, `cos`, and `sin` at a working precision of at least 320 bits, using the sine of the complement beyond `π/4`.
The result is rounded only when the decision is stable across an error window of `2^8` units in the last working place; otherwise the precision doubles.
The same generator writes the tables:

```sh
cargo +nightly-2026-05-14 test -p poulpy-ckks --release numerics::roots::tests::regenerate_quadrant_tables -- --ignored
```

A custom `CKKSFloat` declares `SIGNIFICAND_BITS` and obtains generated roots for every order.

`poulpy-cpu-oracle/src/ckks_roots.rs` derives the same values without Astro-float.
It computes `π` with Machin's formula and the cosine with its Taylor series on fixed-point `dashu-int` integers with 448 fractional bits, then asserts that a `2^16`-unit truncation-error window cannot change the rounding.

Encoding twiddles and DFT matrix roots use these roots.
PaCo and SHIP phases take runtime residues modulo large powers of two and use the canonical sine and cosine.

## Preparation and compatibility

DFT roots use the correctly rounded roots of unity.
DFT factor scaling, EvalMod interpolation nodes and amplitudes, Han–Ki degree selection, PaCo/SHIP phases, and approximation setup use the canonical math methods.
Generic polynomial interpolation accepts a cosine callback so that CKKS can choose its numerical policy without teaching Core about CKKS.

The contract assumes identical input scalar bits, parameters, ciphertexts, and keys.
Scalar-bit equality requires finite inputs and finite FFT intermediates.
NaN payload propagation is unspecified; plaintext encoding rejects non-finite coefficients and signed integer overflow.
A user-supplied approximation callback must itself produce identical values.
Equality is between backends at the same precision; `f64` and binary128 are not expected to produce identical encodings to each other.
Custom scalar implementations must now implement `CKKSFloat`; implementations are provided for `f64` and `Quad`.

These bytes can differ from earlier versions.
Rebuild cached bootstrap parameters when adopting the canonical encoding.
Future changes to the numerical algorithms must explicitly review this compatibility boundary and the frozen fixtures; dependency upgrades must not silently redefine it.

## Verification

- Independent oracle/portable differential tests cover every power-of-two degree through 65,536, both transform directions and both precisions.
  Inputs include zeros, signed zeros, impulses, cancellation, subnormals, large finite values, and deterministic dense vectors.
- AVX2 and AVX-512 run the same differential corpus against the independent oracle.
- Backend-generic fixtures cover all four encoding primitives, sparse plaintext storage, scales 30/60/110, nontrivial DFT scaling, and all five EvalMod variants.
  Fixture hashes use canonical little-endian active data.
- Standard bootstrap fixtures check ciphertext words at input, C2S real and imaginary outputs, EvalMod real and imaginary outputs, and final output.
  The imaginary EvalMod checkpoint is captured after evaluation has written its output.
  Existing accuracy and manual/orchestrated-circuit checks remain active.
- Both quadrant tables match the generator entry for entry.
  The oracle's independent derivation matches every production root of order `2^17` in both precisions, and sampled roots of order `2^20`.
  Symmetry tests check conjugation, quarter turns, and lowest-terms reduction bit for bit.
- Codec tests cover half-integers, signed limits, non-finite values, the normal/subnormal transition, and double-rounding counterexamples.
  A binary128 arithmetic reference checks 70,000 combinations of binary64 inputs and seven scales through 1,074, checking both integer widths and both scalar implementations.
- Native CI is configured to run the same fixtures on Linux x86-64, Intel macOS, Apple Silicon, and Linux AArch64.
  The portable lane additionally enables `libquadmath` to check that the feature cannot change CKKS output.

Local validation ran the workspace and x86 backend suites, including AVX2 and AVX-512.
Linux AArch64 encoding, setup, and complete bootstrap fixtures also ran locally under QEMU with musl, covering portable and NEON backends with both scalar precisions.
The same environment passed the independent FFT corpus through degree 65,536 and the root-table checks.
Intel macOS and Apple Silicon passed cross-compilation checks; Intel macOS additionally passed actual AVX2 assembly code generation.
QEMU Linux execution checks ARM arithmetic and byte parity, while native macOS CI checks the Apple OS and runtime combinations.

Useful local commands:

```sh
cargo +nightly-2026-05-14 test -p poulpy-ckks --profile ci numerics::
cargo +nightly-2026-05-14 test -p poulpy-cpu-portable --profile ci --features enable-ckks --test ckks_determinism
cargo +nightly-2026-05-14 test -p poulpy-cpu-portable --profile ci --features enable-ckks encoding_determinism
cargo +nightly-2026-05-14 test -p poulpy-cpu-portable --profile ci --features enable-ckks bootstrapping_standard_e2e
RUSTFLAGS='-C target-feature=+avx2,+fma,+avx512f,+avx512ifma,+avx512vl' \
  cargo +nightly-2026-05-14 test -p poulpy-cpu-avx512 --profile ci --features enable-ifma,enable-ckks encoding_fft_matches_oracle
```

A musl run needs Clang, LLVM tools, the Rust target, and QEMU user-mode emulation:

```sh
rustup target add --toolchain nightly-2026-05-14 aarch64-unknown-linux-musl
CC_aarch64_unknown_linux_musl=clang \
AR_aarch64_unknown_linux_musl=llvm-ar \
CFLAGS_aarch64_unknown_linux_musl=-ffreestanding \
CARGO_TARGET_AARCH64_UNKNOWN_LINUX_MUSL_LINKER=rust-lld \
CARGO_TARGET_AARCH64_UNKNOWN_LINUX_MUSL_RUNNER=qemu-aarch64-static \
  cargo +nightly-2026-05-14 test --target aarch64-unknown-linux-musl \
    -p poulpy-cpu-arm --profile ci --features enable-neon,enable-ckks encoding_determinism
```

Use the same environment with `-p poulpy-cpu-portable --test ckks_determinism` for the independent FFT corpus, or with the `bootstrapping_standard_e2e` filter for ciphertext fixtures.

## Performance

The baseline is `42e3d25a`, the version after oracle separation and immediately before deterministic encoding was introduced.
Each production backend is compared with that same backend at the baseline revision, using release builds with the pinned toolchain and a Ryzen 9950X pinned to one CPU.
The reported overhead therefore measures the cost added by deterministic encoding to that production backend.
AVX2-only builds enable `avx2,fma`; AVX-512 builds additionally enable `avx512f,avx512ifma,avx512vl`.
Both versions consume identical saved scalar bytes for dense inputs and bootstrap DFT diagonals, resetting the input before every encoding.
Timings exclude module construction and reuse initialized plans and buffers.
Each result is the median of three process medians, with nine timing samples per process and alternating before/after process order.
Samples contain 100 encodings for binary64 and 10 for binary128.
Emulation, compilation, and other tests are paused during measurement.

The original performance check missed the `i64` conversion path because scale 58 with a ten-bit budget uses `i128` storage.
At scale 30, the general bit codec and the AVX2 scalar transform fallback caused approximately 70% and 56% dense-encoding regressions on AVX2 and AVX-512 respectively.
The checked narrow conversion and restored unfused AVX2 SIMD path remove most of that overhead.
The remaining measured low-scale overhead is approximately 9% on AVX2 and 3% on AVX-512 for dense inputs.

At degree 65,536, the final cached-plan results are:

| Backend | Scalar | Scale bits | Input | Original (µs) | Fixed (µs) | Change |
| --- | --- | ---: | --- | ---: | ---: | ---: |
| AVX2 | `f64` | 30 | Dense | 217.05 | 237.46 | +9.4% |
| AVX2 | `f64` | 30 | DFT diagonals | 222.83 | 240.00 | +7.7% |
| AVX-512 | `f64` | 30 | Dense | 201.10 | 206.88 | +2.9% |
| AVX-512 | `f64` | 30 | DFT diagonals | 208.44 | 210.11 | +0.8% |
| AVX2 | `f64` | 58 | Dense | 987.01 | 1,007.53 | +2.1% |
| AVX2 | `f64` | 58 | DFT diagonals | 973.04 | 940.13 | -3.4% |
| AVX-512 | `f64` | 58 | Dense | 987.02 | 998.61 | +1.2% |
| AVX-512 | `f64` | 58 | DFT diagonals | 962.66 | 932.01 | -3.2% |
| AVX-512 | `Quad` | 110 | Dense | 32,766.73 | 30,536.61 | -6.8% |
| AVX-512 | `Quad` | 110 | DFT diagonals | 29,245.57 | 27,174.85 | -7.1% |

The AVX2 unfused path is about 11–12% faster than its canonical scalar fallback at scale 30 and passes the full independent differential corpus.
The AVX2 ring FFT timings at degree 65,536 remain within 1% of the original; the fused base-16 assembly is byte-identical.

### Cold setup

Encoding table construction reads the root tables.
The following first-encoding timings include encoding plan construction for `FFT64Avx512` at scale 58, as Criterion medians on one pinned CPU.
The comparison column evaluates the same twiddles with canonical sine and cosine instead of the tables.

| Scalar | Degree | Canonical trigonometry | Root tables |
| --- | ---: | ---: | ---: |
| `f64` | 2,048 | 41.0 µs | 47.7 µs |
| `f64` | 65,536 | 1.47 ms | 1.68 ms |
| `Quad` | 2,048 | 10.0 ms | 0.70 ms |
| `Quad` | 65,536 | 331 ms | 34.3 ms |

Binary128 setup is dominated by the Astro-float evaluations that the tables replace.
For binary64, the symmetry reduction and table reads cost about 0.2 ms more than pinned `libm` at degree 65,536.
Later calls reuse the initialized dimension; requesting another dimension constructs that dimension once.
Cached-plan encoding runs the same kernels with the same table layout.
These measurements do not claim native ARM, macOS, GPU, or total bootstrap performance.

### Reproducing the workloads

Each CPU backend has an `encoding` Criterion benchmark covering dense inputs, actual bootstrap DFT diagonals, coefficient-only conversion, and first encoding.
The cached workloads cover both scalar precisions, degrees 2,048 and 65,536, and scales 30, 58, and 110.
The first-encoding benchmark uses scale 58.
For example:

```sh
RUSTFLAGS='-C target-feature=+avx2,+fma' \
  cargo +nightly-2026-05-14 bench -p poulpy-cpu-avx \
    --features enable-avx,enable-ckks --bench encoding -- 'f64/slots_dense/n65536/delta30'
RUSTFLAGS='-C target-feature=+avx2,+fma,+avx512f,+avx512ifma,+avx512vl' \
  cargo +nightly-2026-05-14 bench -p poulpy-cpu-avx512 \
    --features enable-ifma,enable-ckks --bench encoding -- 'f128/first_encoding'
```

The existing general CKKS encoding benchmark also resets its input on every iteration.
Repeatedly encoding the mutated buffer changes the workload and previously hid conversion costs as values approached zero.
