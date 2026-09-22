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
| FFT64  | `FFT64Ref` | `FFT64Avx`, `FFT64AvxRayon` | `FFT64Avx512`, `FFT64Avx512Rayon` | `FFT64Neon`, `FFT64NeonRayon` |
| NTT4x30 | `NTT4x30Ref` | `NTT4x30Avx`, `NTT4x30AvxRayon` | `NTT4x30Avx512`, `NTT4x30Avx512Rayon` | `NTT4x30Neon`, `NTT4x30NeonRayon` |
| NTT3x42 | none | none | `NTT3x42Ifma`, `NTT3x42IfmaRayon` | none |

The `*Ref` types live in `poulpy-cpu-ref` and are portable across every CPU.
They prioritize correctness and validation, not performance; use an accelerated backend for performance-sensitive workloads.
The `*Avx` types live in `poulpy-cpu-avx`.
The `*Avx512` and `NTT3x42Ifma` types live in `poulpy-cpu-avx512`.
The `*Neon` types live in `poulpy-cpu-arm` and target AArch64 (Apple Silicon, Neoverse).
The `*Rayon` types use the same arithmetic subfamily and storage formats as their serial counterparts but schedule supported operations over the active Rayon thread pool.

| Backend | Crate | Feature | Required target features |
|---------|-------|---------|--------------------------|
| `FFT64Ref` | `poulpy-cpu-ref` | none | none |
| `FFT64Avx` | `poulpy-cpu-avx` | `enable-avx` | `+avx2,+fma` |
| `FFT64AvxRayon` | `poulpy-cpu-avx` | `enable-rayon` | `+avx2,+fma` |
| `FFT64Avx512` | `poulpy-cpu-avx512` | `enable-avx512f` | `+avx512f` |
| `FFT64Avx512Rayon` | `poulpy-cpu-avx512` | `enable-rayon` | `+avx512f` |
| `FFT64Neon` | `poulpy-cpu-arm` | `enable-neon` | none |
| `FFT64NeonRayon` | `poulpy-cpu-arm` | `enable-rayon` | none |
| `NTT4x30Ref` | `poulpy-cpu-ref` | none | none |
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
use poulpy_cpu_ref::FFT64Ref;

let module: Module<FFT64Ref> = Module::new(1 << 10);
```

Switching subfamily, acceleration, or scheduling is a one-line change.

```rust
use poulpy_cpu_ref::NTT4x30Ref;

let module = Module::<NTT4x30Ref>::new(1 << 10);
```

The common pattern in the examples picks the fastest available backend with `cfg`.

```rust
#[cfg(all(feature = "enable-avx", target_arch = "x86_64"))]
use poulpy_cpu_avx::FFT64Avx as BackendImpl;
#[cfg(not(all(feature = "enable-avx", target_arch = "x86_64")))]
use poulpy_cpu_ref::FFT64Ref as BackendImpl;

let module = Module::<BackendImpl>::new(n as u64);
```

## Choosing a subfamily

Choose the limb size `base2k` from the number of accumulated products and an explicit failure target. The runtime query `Module::<BE>::max_base2k(n, products, failure_bits)` delegates to the backend's `BackendMaxBase2k` implementation:

```rust
use poulpy_cpu_ref::{FFT64Ref, NTT4x30Ref};
use poulpy_hal::layouts::Module;

// 32 accumulated products, with a 2^-128 target for the entire polynomial.
let base2k: Option<usize> = Module::<NTT4x30Ref>::max_base2k(1 << 16, 32, 128);
assert_eq!(base2k, Some(54));
let fft_base2k: Option<usize> = Module::<FFT64Ref>::max_base2k(1 << 16, 32, 128);
assert_eq!(fft_base2k, Some(19));
```

The query uses independent centered-uniform input coefficients, a conservative Gaussian tail envelope, and a union bound over all coefficients of one output polynomial. For `n = 2^16`, 32 products, and 128 failure bits, it selects 19 for `FFT64`, 54 for `NTT4x30`, and 57 for `NTT3x42`. Increasing the accumulation count or tightening the failure target can lower the selected radix.
For NTT backends, `PrimeSet::LOG_Q_PRODUCT` supplies `log2(Q)` from the actual CRT modulus, approximately 119.8861552574811 for `NTT4x30` and 125.99998314565484 for `NTT3x42`.

The result is the largest radix up to 62 satisfying the Gaussian envelope; `Some(0)` means no positive radix fits. For `m` output polynomials, add `ceil(log2(m))` to the requested failure bits.
FFT64 backends implement the trait using the shared stochastic roundoff model covering transforms, complex products, and sequential accumulation before one inverse FFT. It assumes approximately uncorrelated roundoff and twiddle-error contributions; these are model estimates, not certified far-tail bounds. An implementation returns `None` when it has no applicable model for the workload, as with the storage-only `HostBytesBackend`.
Backends implement `BackendMaxBase2k` with an ordinary trait method; the query needs no module instance. They can reuse `poulpy_hal::layouts::{max_base2k_ntt, max_base2k_fft64}` or supply a model for their own arithmetic. The module validates the common input constraints before forwarding the query. Storage delegation through `impl_backend_from!` does not choose a model; wrappers that preserve the arithmetic explicitly forward `BackendMaxBase2k` as well.
See [Failure estimates](base2k-failure-probability.md) for the formula, assumptions, and comparison with worst-case bounds.
This query does not restrict coefficient-only operations whose contracts permit wider radices, such as uniform sampling up to 62 bits.
A larger `base2k` represents the same precision in fewer limbs. Validate the input distribution, accumulation count, numerical-error model, and circuit noise budget for the operations being used.

The returned maximum is not necessarily the practical radix: also reserve coefficient-word headroom for additions and subtractions outside the DFT domain, including repeated accumulations before normalization. For `i64` words, a radix of 62 leaves little headroom for these chains even if the DFT model permits it. Choose a smaller radix when needed to keep every intermediate in range; see [coefficient-domain addition headroom](base2k-failure-probability.md#reserve-headroom-for-coefficient-domain-additions).

Use `FFT64` for gate-level and TFHE-style work, especially at small ring dimensions: there the limb count is already low, so the wider NTT limbs cannot pay for their extra transforms.

For the leveled schemes, use `NTT3x42` where the CPU has AVX-512-IFMA — it is the fastest leveled backend by a clear margin.
Without IFMA there is no general winner between `NTT4x30` and `FFT64`: the first is faster on coefficient-domain work, where the limb count decides, and the second on key-switch-dominated work, where the size of the prepared key decides.
Which one wins therefore depends on the mix of operations in your circuit, so test both.
See [Performance](performance.md) for the reasoning and for the diagnostics that answer it on your hardware.

Within a chosen subfamily, prefer the most accelerated backend your CPU and build flags allow.
Choose a `*Rayon` variant when one operation should use several CPU cores, especially for large dimensions or batches.
Choose its serial counterpart when the application already parallelizes independent operations or when the workload is too small to repay scheduling overhead.
Rayon variants fall back to serial execution when the active pool has one thread or the work is below their internal parallelization threshold.
