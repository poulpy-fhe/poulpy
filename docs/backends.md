# Backends

Poulpy decouples scheme code from polynomial arithmetic through the hardware abstraction layer (`poulpy-hal`).
A *backend* performs the low-level primitives behind that layer, such as the Fourier transform and matrix-vector products.
Every backend is a small marker type that you pass as the generic parameter of `Module<B>`.
User code stays generic over `B` and the backend is chosen at compile time.

There are three arithmetic families: `FFT64`, `NTT120`, and `NTT126`.
They differ in the transform each uses to make polynomial multiplication in `Z[X]/(X^N + 1)` efficient.
They are interchangeable behind the HAL, so the same scheme code runs on any of them.

## The three families

### FFT64

`FFT64` uses a 64-bit floating-point FFT for polynomial multiplication.
Coefficients live in the DFT domain as `f64` and in the large-integer domain as `i64`.
The transform is approximate because it relies on IEEE 754 rounding.
It is the preferred choice at small ring dimensions, and the family the examples and the gate-level FHE crate use by default.

### NTT120

`NTT120` uses an exact integer NTT for polynomial multiplication.
It uses the Chinese Remainder Theorem over four roughly 30-bit primes (`Primes30`), giving a modulus `Q` near `2^120`.
Coefficients live in the NTT domain as four `u64` lanes and reconstruct to `i128` in the large-integer domain.
The arithmetic is exact, with no floating-point error, which makes it the better choice at larger ring dimensions.

### NTT126

`NTT126` is also an exact integer NTT.
It uses three roughly 42-bit primes (`Primes42`) chosen for AVX-512 IFMA52 hardware, giving a modulus `Q` near `2^126`.
It reconstructs to `i128` like `NTT120` but reaches slightly higher precision per transform.
It exists only as an IFMA-accelerated backend, because it relies on IFMA multiply-add to keep its matrix-vector products within 64 bits.

## Available backend types

| Family | Reference | AVX2 / FMA | AVX-512 |
|--------|-----------|------------|---------|
| FFT64  | `FFT64Ref` | `FFT64Avx` | `FFT64Avx512` |
| NTT120 | `NTT120Ref` | `NTT120Avx` | `NTT120Avx512` |
| NTT126 | none | none | `NTT126Ifma` |

The `*Ref` types live in `poulpy-cpu-ref` and are portable across every CPU.
The `*Avx` types live in `poulpy-cpu-avx`.
The `*Avx512` and `NTT126Ifma` types live in `poulpy-cpu-avx512`.

| Backend | Crate | Feature | Required target features |
|---------|-------|---------|--------------------------|
| `FFT64Ref` | `poulpy-cpu-ref` | none | none |
| `FFT64Avx` | `poulpy-cpu-avx` | `enable-avx` | `+avx2,+fma` |
| `FFT64Avx512` | `poulpy-cpu-avx512` | `enable-avx512f` | `+avx512f` |
| `NTT120Ref` | `poulpy-cpu-ref` | none | none |
| `NTT120Avx` | `poulpy-cpu-avx` | `enable-avx` | `+avx2,+fma` |
| `NTT120Avx512` | `poulpy-cpu-avx512` | `enable-avx512f` | `+avx512f` |
| `NTT126Ifma` | `poulpy-cpu-avx512` | `enable-ifma` | `+avx512f,+avx512ifma,+avx512vl,+bmi2,+adx` |

Accelerated backends check the required CPU features at runtime in `Module::new` and panic if they are missing.
They also require the matching `target-feature` flags at compile time.

## How to use a backend

A backend is selected by naming its type when you build the `Module`.

```rust
use poulpy_hal::layouts::Module;
use poulpy_cpu_ref::FFT64Ref;

let module: Module<FFT64Ref> = Module::new(1 << 10);
```

Switching family or acceleration is a one-line change.

```rust
use poulpy_cpu_ref::NTT120Ref;

let module = Module::<NTT120Ref>::new(1 << 10);
```

The common pattern in the examples picks the fastest available backend with `cfg`.

```rust
#[cfg(all(feature = "enable-avx", target_arch = "x86_64"))]
use poulpy_cpu_avx::FFT64Avx as BackendImpl;
#[cfg(not(all(feature = "enable-avx", target_arch = "x86_64")))]
use poulpy_cpu_ref::FFT64Ref as BackendImpl;

let module = Module::<BackendImpl>::new(n as u64);
```

## Choosing a family

The backend fixes the maximum limb size `base2k` you can use.
`FFT64` allows up to 19 bits per limb, while `NTT120` allows up to 52.
A larger `base2k` represents the same precision in fewer limbs, at the cost of more expensive elementary operations.
This tradeoff usually pays off for leveled schemes such as BFV, BGV, and CKKS, but not for TFHE.
So use `FFT64` for gate-level and TFHE-style work, especially at small ring dimensions.
Use `NTT120` or `NTT126` for the leveled schemes, where the larger limbs cut the limb count.
Within a chosen family, prefer the most accelerated backend your CPU and build flags allow.
