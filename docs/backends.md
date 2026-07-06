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
| FFT64  | `FFT64Ref` | `FFT64Avx` | `FFT64Avx512` | `FFT64Neon` |
| NTT4x30 | `NTT4x30Ref` | `NTT4x30Avx` | `NTT4x30Avx512` | `NTT4x30Neon` |
| NTT3x42 | none | none | `NTT3x42Ifma` | none |

The `*Ref` types live in `poulpy-cpu-ref` and are portable across every CPU.
The `*Avx` types live in `poulpy-cpu-avx`.
The `*Avx512` and `NTT3x42Ifma` types live in `poulpy-cpu-avx512`.
The `*Neon` types live in `poulpy-cpu-arm` and target AArch64 (Apple Silicon, Neoverse).

| Backend | Crate | Feature | Required target features |
|---------|-------|---------|--------------------------|
| `FFT64Ref` | `poulpy-cpu-ref` | none | none |
| `FFT64Avx` | `poulpy-cpu-avx` | `enable-avx` | `+avx2,+fma` |
| `FFT64Avx512` | `poulpy-cpu-avx512` | `enable-avx512f` | `+avx512f` |
| `FFT64Neon` | `poulpy-cpu-arm` | `enable-neon` | none |
| `NTT4x30Ref` | `poulpy-cpu-ref` | none | none |
| `NTT4x30Avx` | `poulpy-cpu-avx` | `enable-avx` | `+avx2,+fma` |
| `NTT4x30Avx512` | `poulpy-cpu-avx512` | `enable-avx512f` | `+avx512f` |
| `NTT4x30Neon` | `poulpy-cpu-arm` | `enable-neon` | none |
| `NTT3x42Ifma` | `poulpy-cpu-avx512` | `enable-ifma` | `+avx512f,+avx512ifma,+avx512vl` |

The AVX and AVX-512 backends check the required CPU features at runtime in `Module::new` and panic if they are missing.
They also require the matching `target-feature` flags at compile time.
The `*Neon` backends need no `target-feature` flags and build only on `aarch64`.

## How to use a backend

A backend is selected by naming its type when you build the `Module`.

```rust
use poulpy_hal::layouts::Module;
use poulpy_cpu_ref::FFT64Ref;

let module: Module<FFT64Ref> = Module::new(1 << 10);
```

Switching subfamily or acceleration is a one-line change.

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

The backend fixes the maximum limb size `base2k` you can use.
`FFT64` allows up to 19 bits per limb, while `NTT4x30` allows up to 52.
A larger `base2k` represents the same precision in fewer limbs, at the cost of more expensive elementary operations.
This tradeoff usually pays off for leveled schemes such as BFV, BGV, and CKKS, but not for TFHE.
So use `FFT64` for gate-level and TFHE-style work, especially at small ring dimensions.
Use `NTT4x30` or `NTT3x42` for the leveled schemes, where the larger limbs cut the limb count.
Within a chosen subfamily, prefer the most accelerated backend your CPU and build flags allow.
