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
Coefficients live in the NTT domain as four `u32` residues, one per prime, and reconstruct to `i128` in the large-integer domain.
The arithmetic is exact, with no floating-point error, which makes it the better choice at larger ring dimensions.

### NTT3x42

`NTT3x42` is also an exact integer NTT.
It uses three roughly 42-bit primes (`Primes42`) chosen for AVX-512 IFMA52 hardware, giving a modulus `Q` near `2^126`.
It reconstructs to `i128` like `NTT4x30` but reaches slightly higher precision per transform.
It exists only as an IFMA-accelerated backend, because it relies on IFMA multiply-add to keep its matrix-vector products within 64 bits.

## Available backend types

| Subfamily | Reference | AVX2 / FMA | AVX-512 | NEON |
|-----------|-----------|------------|---------|------|
| FFT64  | `FFT64Portable`, `FFT64PortableRayon` | `FFT64Avx`, `FFT64AvxRayon` | `FFT64Avx512`, `FFT64Avx512Rayon` | `FFT64Neon`, `FFT64NeonRayon` |
| NTT4x30 | `NTT4x30Portable`, `NTT4x30PortableRayon` | `NTT4x30Avx`, `NTT4x30AvxRayon` | `NTT4x30Avx512`, `NTT4x30Avx512Rayon` | `NTT4x30Neon`, `NTT4x30NeonRayon` |
| NTT3x42 | none | none | `NTT3x42Ifma`, `NTT3x42IfmaRayon` | none |

The `*Ref` types live in `poulpy-cpu-portable` and are portable across every CPU.
They prioritize correctness and validation, not performance; use an accelerated backend for performance-sensitive workloads.
The `*Avx` types live in `poulpy-cpu-avx`.
The `*Avx512` and `NTT3x42Ifma` types live in `poulpy-cpu-avx512`.
The `*Neon` types live in `poulpy-cpu-arm` and target AArch64 (Apple Silicon, Neoverse).
The Rayon variants of the portable backends live in `poulpy-cpu-rayon`, since that crate depends on `poulpy-cpu-portable` and not the other way round.
The `*Rayon` types use the same arithmetic subfamily and storage formats as their serial counterparts but schedule supported operations over the active Rayon thread pool.

| Backend | Crate | Feature | Required target features |
|---------|-------|---------|--------------------------|
| `FFT64Portable` | `poulpy-cpu-portable` | none | none |
| `FFT64PortableRayon` | `poulpy-cpu-rayon` | none | none |
| `FFT64Avx` | `poulpy-cpu-avx` | `enable-avx` | `+avx2,+fma` |
| `FFT64AvxRayon` | `poulpy-cpu-avx` | `enable-rayon` | `+avx2,+fma` |
| `FFT64Avx512` | `poulpy-cpu-avx512` | `enable-avx512f` | `+avx512f` |
| `FFT64Avx512Rayon` | `poulpy-cpu-avx512` | `enable-rayon` | `+avx512f` |
| `FFT64Neon` | `poulpy-cpu-arm` | `enable-neon` | none |
| `FFT64NeonRayon` | `poulpy-cpu-arm` | `enable-rayon` | none |
| `NTT4x30Portable` | `poulpy-cpu-portable` | none | none |
| `NTT4x30PortableRayon` | `poulpy-cpu-rayon` | none | none |
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
In `poulpy-cpu-rayon`, the `enable-core`, `enable-ckks`, `enable-bin-fhe` and `enable-mhe` features wire the scheme layers into the portable Rayon backends, as the features of the same names do in `poulpy-cpu-portable`.
In `poulpy-cpu-avx`, `enable-rayon` implies `enable-avx`.
In `poulpy-cpu-avx512`, `enable-rayon` implies `enable-avx512f`, but it must be combined with `enable-ifma` to expose `NTT3x42IfmaRayon`.

## GPU backends

`poulpy-gpu-portable` implements the `NTT4x30` subfamily on GPUs through [CubeCL](https://github.com/tracel-ai/cubecl).
It uses the same four primes as the CPU `NTT4x30` backends, so its results are bit-identical to `NTT4x30Portable`; only the storage differs (four `u32` residues per DFT-domain coefficient, buffers resident on the device).
The implementation is written once, and each CubeCL runtime has its own backend type, so that several runtimes can run side by side in one process.

| Backend | CubeCL runtime | Hardware | Feature |
|---------|----------------|----------|---------|
| `NTT4x30Cuda` | CUDA | NVIDIA | `enable-cuda` |
| `NTT4x30Hip` | HIP | AMD | `enable-hip` |
| `NTT4x30Wgpu` | wgpu | Metal on Apple, Vulkan, other wgpu targets | `enable-metal`, `enable-vulkan`, `enable-wgpu` |

Each type is generic over the ring like the CPU types (`NTT4x30Wgpu<ConjugateInvariant>`), and `enable-core` and `enable-ckks` add the scheme layers.
A backend type owns one device for the life of the process: the first device of its runtime, or the one chosen with `GpuRuntime::select_device` before the first module or buffer of that runtime is created.

Buffers live on the device, so data moves explicitly: generate keys and encrypt with a host backend, move the layouts with `TransferInto`, prepare and evaluate on the GPU backend, and move results back to decrypt.
Operations are queued; `Module::synchronize` returns once they have completed, and a transfer to the host waits for what it reads.
Sampling, CKKS encoding and decoding, the uniform fill of a coefficient column and the normalizations at radices above 62 bits are staged through the host, and everything else runs on the device.
CKKS encoding uses the portable transform there, which keeps it byte-identical to every other backend.

A sequence of calls that is repeated on the same buffers (a bootstrap, say) can be recorded once and replayed as one CubeCL graph:

```rust
use std::cell::RefCell;

use poulpy_gpu_portable::{Replay, Wgpu};

// The closure borrows the buffers at each call, so the caller can reach them between replays.
let (ct_in, ct_out) = (RefCell::new(ct_in), RefCell::new(ct_out));
let mut replay = Replay::<Wgpu>::record(|| {
    bootstrap(&module, &mut ct_out.borrow_mut(), &ct_in.borrow(), &ctx, &keys, &mut scratch.borrow())
});
// Write new inputs into `ct_in.borrow_mut()`, then:
replay.replay();
// and read the result from `ct_out.borrow()`.
```

The closure must bind the same buffers and issue the same launches at every call, and must not read from the device, write host data to it or synchronize; `Replay` documents the contract, and `Replay::check` tests a closure against its recording.
A recording keeps the temporaries of the closure reserved on the device until it is dropped, and while a closure is recorded or a graph is queued, other threads wait for the device.

Limits of the GPU backends:

- Normalization runs on the device for radix widths from 1 to 62 bits. The wider radices the HAL admits for a 128-bit source or an inverse transform go through the host.
- A window whose degree is not a power-of-two fraction of its operand's is rejected with a panic.

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
For a sum of 32 products, `Module::<NTT4x30Portable>::max_base2k(1 << 16, 32, 128, false)` returns 54; the same query gives 19 for `FFT64Portable` and 57 for `NTT3x42Ifma`.
For a single square, `Module::<NTT4x30Portable>::max_base2k(1 << 16, 1, 128, true)` returns 55; the same query gives 19 for `FFT64Portable` and 58 for `NTT3x42Ifma`.
On the conjugate-invariant ring the same two queries give 53 and 52 for `NTT4x30CIPortable`, 18 and 17 for `FFT64CIPortable`, and 56 and 55 for `NTT3x42CIIfma`: its constant coefficient has twice the variance, and a square's has a positive mean.
The estimates cover independent accumulated products or squares of centered-uniform inputs, with Gaussian tails and a stochastic roundoff model for FFT64. They are not guarantees for arbitrary inputs.
See [Failure estimates for base-2^K arithmetic](base2k-failure-probability.md) for the models and an example.
0 means no positive radix fits; a backend without a model does not implement `MaxBase2k`.
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
The `encoding_determinism` tests of the CKKS backend suites hash the encodings and setup constants and compare them with `schemes/poulpy-ckks/src/test_suite/determinism.txt`, so every backend that runs the suites is held to the same bytes.
After an intended change to the encoding or the setup math, `POULPY_UPDATE_FIXTURES=1` records new hashes, and every backend must then pass without the variable.
