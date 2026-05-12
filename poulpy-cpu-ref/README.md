# 🐙 Poulpy-CPU-REF

**Poulpy-CPU-REF** is the **reference (portable) CPU backend for Poulpy**.

It implements the Poulpy HAL extension traits without requiring SIMD or specialized CPU instructions, making it suitable for:

- all CPU architectures (`x86_64`, `aarch64`, `arm`, `riscv64`, …)
- development machines and CI runners
- environments without AVX or other advanced SIMD support

This backend integrates transparently with:

- `poulpy-hal`
- `poulpy-core`
- `poulpy-ckks`
- `poulpy-bin-fhe`

---

## When is this backend used?

The FFT64 and NTT120 reference HAL backends are always available and require
**no compilation flags and no CPU features**.

It is automatically selected when:

- the project does not request an optimized backend, or
- the target CPU does not support the requested SIMD backend (e.g., AVX), or
- portability and reproducibility are more important than raw performance.

No additional configuration is required to use it.

Higher-level backend wiring is feature-gated:

- `enable-core` wires the reference backends into `poulpy-core` defaults.
- `enable-ckks` wires the reference backends into `poulpy-ckks` defaults and
  also enables core support.

Useful test commands:

```sh
# HAL/reference backend tests
cargo test -p poulpy-cpu-ref

# Core conformance tests on FFT64Ref and NTT120Ref
cargo test -p poulpy-cpu-ref --features enable-core

# CKKS conformance tests on FFT64Ref and NTT120Ref
cargo test -p poulpy-cpu-ref --features enable-ckks
```

---

## 🧪 Basic Usage

This crate exposes two backends:

```rust
use poulpy_cpu_ref::{FFT64Ref, NTT120Ref};
use poulpy_hal::{api::ModuleNew, layouts::Module};

let log_n: usize = 10;

// f64 FFT backend
let module: Module<FFT64Ref> = Module::<FFT64Ref>::new(1 << log_n);

// Q120 NTT backend (CRT over four ~30-bit primes)
let module: Module<NTT120Ref> = Module::<NTT120Ref>::new(1 << log_n);
```

Both work on **all supported platforms and architectures**.

---

## Performance Notes

`poulpy-cpu-ref` prioritizes:

* portability
* correctness
* ease of debugging

For maximum performance on x86_64 CPUs with AVX2 + FMA support, consider enabling the optional optimized backend:

```
poulpy-cpu-avx (feature: enable-avx)
```

For x86_64 CPUs with AVX-512 support, consider the AVX-512 backend:

```
poulpy-cpu-avx512 (features: enable-avx512f, enable-ifma)
```

Benchmarks and applications can freely switch between backends without changing source code — backend selection can be handled with feature flags, for example

```rust
#[cfg(all(feature = "enable-avx", target_arch = "x86_64", target_feature = "avx2", target_feature = "fma"))]
use poulpy_cpu_avx::FFT64Avx as BackendImpl;

#[cfg(not(all(feature = "enable-avx", target_arch = "x86_64", target_feature = "avx2", target_feature = "fma")))]
use poulpy_cpu_ref::FFT64Ref as BackendImpl;
```

The same pattern applies to NTT120 backends (`NTT120Ref` / `NTT120Avx`).

---

## 🤝 Contributors

To implement your own backend (SIMD or accelerator):

1. Define a backend struct and implement the `Backend` trait from `poulpy-hal`.
2. For each HAL operation family, either call the blanket default or implement the OEP trait directly with a custom dispatch.
3. For each `poulpy-core` operation family, either call the corresponding `impl_*_defaults_full!` macro to inherit the portable implementation, or implement the OEP trait directly to override it.
4. Optionally, do the same for `poulpy-ckks` behind a backend-owned `enable-ckks` feature using the `impl_ckks_*_defaults!` macros or direct OEP trait implementations.

At every layer the macro and the direct implementation are mutually exclusive per operation family: the macro opts the backend into the portable `default` path, while a direct OEP impl replaces it entirely. There is no requirement to use the macros — a backend that needs full control can implement every OEP trait by hand.

Your backend will automatically integrate with:

* `poulpy-hal`
* `poulpy-core`
* `poulpy-ckks`
* `poulpy-bin-fhe`

No modifications to those crates are necessary — the HAL provides the extension points. Only the operations that need a faster implementation require explicit overrides; everything else is inherited from the `default` layer for free.

---

For questions or guidance, feel free to open an issue or discussion in the repository.
