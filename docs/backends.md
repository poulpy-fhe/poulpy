# Backends

Poulpy decouples scheme code from polynomial arithmetic through the hardware abstraction layer (`poulpy-hal`).
A *backend* performs the low-level primitives behind that layer, such as the Fourier transform and matrix-vector products.
Every backend is a small marker type that you pass as the generic parameter of `Module<B>`.
User code stays generic over `B` and the backend is chosen at compile time.

There are two main arithmetic families: `FFT` and `NTT` backends, each with their subfamilies.
They differ in the format of the DFT used to make polynomial multiplication in `Z[X]/(X^N + 1)` efficient: `FFT` backends use a complex floating-point DFT, while `NTT` backends use a number-theoretic transform over integers.
All backends are interchangeable behind the HAL, so the same scheme code runs on any of them.

## Currently available subfamilies

Poulpy currently ships one FFT subfamily, `FFT64`, and two NTT subfamilies, `NTT4x30` and `NTT3x42`.

### FFT64

`FFT64` uses a 64-bit floating-point FFT for polynomial multiplication.
Coefficients live in the DFT domain as `f64` and in the large-integer domain as `i64`.
The transform is approximate because it relies on IEEE 754 rounding.
It is the preferred choice at small ring dimensions, and the subfamily the examples and the gate-level FHE crate use by default.

### NTT4x30

`NTT4x30` uses an exact integer NTT for polynomial multiplication.
It uses the Chinese Remainder Theorem over four roughly 30-bit primes (`Primes30`), giving a modulus `Q` near `2^120`.
Coefficients live in the NTT domain as four `u64` lanes and reconstruct to `i128` in the large-integer domain.
The arithmetic is exact, with no floating-point error, which makes it the better choice at larger ring dimensions.

### NTT3x42

`NTT3x42` is also an exact integer NTT.
It uses three roughly 42-bit primes (`Primes42`) chosen for AVX-512 IFMA52 hardware, giving a modulus `Q` near `2^126`.
It reconstructs to `i128` like `NTT4x30` but reaches slightly higher precision per transform.
It exists only as an IFMA-accelerated backend, because it relies on IFMA multiply-add to keep its matrix-vector products within 64 bits.

## Available backend types

| Subfamily | Reference | AVX2 / FMA | AVX-512 | NEON |
|-----------|-----------|------------|---------|------|
| FFT64  | `FFT64Portable` | `FFT64Avx`, `FFT64AvxRayon` | `FFT64Avx512`, `FFT64Avx512Rayon` | `FFT64Neon`, `FFT64NeonRayon` |
| NTT4x30 | `NTT4x30Portable` | `NTT4x30Avx`, `NTT4x30AvxRayon` | `NTT4x30Avx512`, `NTT4x30Avx512Rayon` | `NTT4x30Neon`, `NTT4x30NeonRayon` |
| NTT3x42 | none | none | `NTT3x42Ifma`, `NTT3x42IfmaRayon` | none |

The `*Ref` types live in `poulpy-cpu-portable` and are portable across every CPU.
They prioritize correctness and validation, not performance; use an accelerated backend for performance-sensitive workloads.
The `*Avx` types live in `poulpy-cpu-avx`.
The `*Avx512` and `NTT3x42Ifma` types live in `poulpy-cpu-avx512`.
The `*Neon` types live in `poulpy-cpu-arm` and target AArch64 (Apple Silicon, Neoverse).
The `*Rayon` types use the same arithmetic subfamily and storage formats as their serial counterparts but schedule supported operations over the active Rayon thread pool.

| Backend | Crate | Feature | Required target features |
|---------|-------|---------|--------------------------|
| `FFT64Portable` | `poulpy-cpu-portable` | none | none |
| `FFT64Avx` | `poulpy-cpu-avx` | `enable-avx` | `+avx2,+fma` |
| `FFT64AvxRayon` | `poulpy-cpu-avx` | `enable-rayon` | `+avx2,+fma` |
| `FFT64Avx512` | `poulpy-cpu-avx512` | `enable-avx512f` | `+avx512f` |
| `FFT64Avx512Rayon` | `poulpy-cpu-avx512` | `enable-rayon` | `+avx512f` |
| `FFT64Neon` | `poulpy-cpu-arm` | `enable-neon` | none |
| `FFT64NeonRayon` | `poulpy-cpu-arm` | `enable-rayon` | none |
| `NTT4x30Portable` | `poulpy-cpu-portable` | none | none |
| `NTT4x30Avx` | `poulpy-cpu-avx` | `enable-avx` | `+avx2,+fma` |
| `NTT4x30AvxRayon` | `poulpy-cpu-avx` | `enable-rayon` | `+avx2,+fma` |
| `NTT4x30Avx512` | `poulpy-cpu-avx512` | `enable-avx512f` | `+avx512f` |
| `NTT4x30Avx512Rayon` | `poulpy-cpu-avx512` | `enable-rayon` | `+avx512f` |
| `NTT4x30Neon` | `poulpy-cpu-arm` | `enable-neon` | none |
| `NTT4x30NeonRayon` | `poulpy-cpu-arm` | `enable-rayon` | none |
| `NTT3x42Ifma` | `poulpy-cpu-avx512` | `enable-ifma` | `+avx512f,+avx512ifma,+avx512vl` |
| `NTT3x42IfmaRayon` | `poulpy-cpu-avx512` | `enable-ifma`, `enable-rayon` | `+avx512f,+avx512ifma,+avx512vl` |

The AVX and AVX-512 backends check the required CPU features at runtime in `Module::new` and panic if they are missing.
They also require the matching `target-feature` flags at compile time.
The `*Neon` backends need no `target-feature` flags and build only on `aarch64`.
In `poulpy-cpu-avx`, `enable-rayon` implies `enable-avx`.
In `poulpy-cpu-avx512`, `enable-rayon` implies `enable-avx512f`, but it must be combined with `enable-ifma` to expose `NTT3x42IfmaRayon`.

## How to use a backend

A backend is selected by naming its type when you build the `Module`.

```rust
use poulpy_hal::layouts::Module;
use poulpy_cpu_portable::FFT64Portable;

let module: Module<FFT64Portable> = Module::new(1 << 10);
```

Switching subfamily, acceleration, or scheduling is a one-line change.

```rust
use poulpy_cpu_portable::NTT4x30Portable;

let module = Module::<NTT4x30Portable>::new(1 << 10);
```

The common pattern in the examples picks the fastest available backend with `cfg`.

```rust
#[cfg(all(feature = "enable-avx", target_arch = "x86_64"))]
use poulpy_cpu_avx::FFT64Avx as BackendImpl;
#[cfg(not(all(feature = "enable-avx", target_arch = "x86_64")))]
use poulpy_cpu_portable::FFT64Portable as BackendImpl;

let module = Module::<BackendImpl>::new(n as u64);
```

## Choosing a subfamily

Select `base2k` with the runtime query `Module::<BE>::max_base2k(n, products, failure_bits, squaring)`, which delegates to `MaxBase2k` without constructing a module.
`products` counts the polynomial products accumulated into one output; `failure_bits` requests an estimated whole-polynomial failure probability of at most `2^(-failure_bits)`.
Set `squaring = false` for independent products and `true` if any term is a square, counting each square once.
For a sum of 32 products, `Module::<NTT4x30Portable>::max_base2k(1 << 16, 32, 128, false)` returns `Some(54)`; the same query gives `Some(19)` for `FFT64Portable` and `Some(57)` for `NTT3x42Ifma`.
For a single square, `Module::<NTT4x30Portable>::max_base2k(1 << 16, 1, 128, true)` returns `Some(55)`; the same query gives `Some(19)` for `FFT64Portable` and `Some(58)` for `NTT3x42Ifma`.
The estimates cover independent accumulated products or squares of centered-uniform inputs, with Gaussian tails and a stochastic roundoff model for FFT64. They are not guarantees for arbitrary inputs.
See [Failure estimates for base-2^K arithmetic](base2k-failure-probability.md) for the models and an example.
`Some(0)` means no positive radix fits, and `None` means the backend has no applicable model.
The radix is capped at the coefficient word width minus two bits: 62 for `i64`, or 30 for a future `i32` backend.
A larger `base2k` represents the same precision in fewer limbs, but also reserve coefficient-word headroom for additions before normalization and respect the circuit's noise budget.

Use `FFT64` for gate-level and TFHE-style work, especially at small ring dimensions: there the limb count is already low, so the wider NTT limbs cannot pay for their extra transforms.

For the leveled schemes, use `NTT3x42` where the CPU has AVX-512-IFMA — it is the fastest leveled backend by a clear margin.
Without IFMA there is no general winner between `NTT4x30` and `FFT64`: the first is faster on coefficient-domain work, where the limb count decides, and the second on key-switch-dominated work, where the size of the prepared key decides.
Which one wins therefore depends on the mix of operations in your circuit, so test both.
See [Performance](performance.md) for the reasoning and for the diagnostics that answer it on your hardware.

Within a chosen subfamily, prefer the most accelerated backend your CPU and build flags allow.
Choose a `*Rayon` variant when one operation should use several CPU cores, especially for large dimensions or batches.
Choose its serial counterpart when the application already parallelizes independent operations or when the workload is too small to repay scheduling overhead.
Rayon variants fall back to serial execution when the active pool has one thread or the work is below their internal parallelization threshold.

## CKKS encoding

CKKS encoding and decoding return the same bytes on every backend and platform, at each scalar precision it implements (`f32`, `f64` or `Quad`), in both rings.
The plaintexts, the decoded slots and the float coefficients of the slot transforms all match, and so do the setup constants of bootstrapping.

A backend implementing `CKKSEncodingImpl` keeps this by following one arithmetic definition, whatever its layout, vectorization or scheduling:

- The slot transforms are the radix-2 negacyclic FFT of `fft_portable_fused` and `ifft_portable_fused` in `poulpy-cpu-portable`, with the same butterflies on the same pairs.
- Each butterfly computes `b * w` with one multiply-add per component, as the portable kernels do: the real part is `fma(br, wr, -(bi * wi))` and the imaginary part `fma(bi, wr, br * wi)`, the product in parentheses rounded once. The butterflies by `i * w` and the inverse butterflies follow the portable kernels in the same way, including which sum is computed negated.
- No other operation is fused or reassociated. On a GPU this means compiling with contraction disabled, such as `--fmad=false`, or writing every operation with the explicitly rounded intrinsics `__fma_rn`, `__fmul_rn` and `__fadd_rn`.
- The twiddles are the correctly rounded roots of unity of `CKKSFloat::ckks_root_of_unity`, as built by `EncodingFFTTable`.
- Scalars convert to plaintext integers with `CKKSFloat::ckks_quantize` and back with `CKKSFloat::ckks_dequantize`.
- Arithmetic runs in the default floating-point environment: round to nearest even, with subnormals kept.

Two tests check the contract.
`test_negacyclic_fft_bit_exact` in the HAL test suite compares a transform with `EncodingFFTTable` byte for byte, and the oracle's independent `EncodingFft` checks the portable table the same way.
The `encoding_determinism` tests of the CKKS backend suites hash the encodings and setup constants and compare them with `poulpy-ckks/src/test_suite/determinism.txt`, so every backend that runs the suites is held to the same bytes.
After an intended change to the encoding or the setup math, `POULPY_UPDATE_FIXTURES=1` records new hashes, and every backend must then pass without the variable.
