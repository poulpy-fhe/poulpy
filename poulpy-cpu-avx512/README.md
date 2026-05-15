# 🐙 Poulpy-CPU-AVX512

**Poulpy-CPU-AVX512** is a Rust crate that provides **AVX-512 accelerated CPU backends for Poulpy**.

It exposes three backends, gated behind two layered Cargo features:

- **`FFT64Avx512`** — f64 complex-FFT backend, gated on `enable-avx512f`. Combines AVX-512F REIM butterflies with AVX2+FMA REIM4 vector-matrix kernels (the AVX2+FMA form has shorter dependency chains and benches at parity with or ahead of the AVX-512F variant across ring sizes 2^10..2^16). Requires AVX-512F + AVX2 + FMA at runtime; AVX2/FMA are implied by AVX-512F on real hardware but verified explicitly at module creation.
- **`NTT120Avx512`** — Q120 NTT backend (CRT over four ~30-bit primes), gated on `enable-avx512f` (requires AVX-512F only). Targets AVX-512F-capable CPUs without IFMA (Skylake-X, Cascade Lake, KNL, Zen 4 SKUs without IFMA).
- **`NTT126Ifma`** — Q126 NTT backend (CRT over three ~42-bit primes), gated on `enable-ifma`. The post-iNTT 3-prime CRT-to-i128 reconstruction is a hand-written assembly kernel that fuses IFMA Garner reduction with a BMI2/ADX scalar carry chain. Requires AVX-512F + AVX-512-IFMA + AVX-512VL + BMI2 + ADX.

`enable-ifma` implies `enable-avx512f`, so enabling IFMA builds all three backends.

This crate implements the Poulpy HAL extension traits and can be used by:

- [`poulpy-hal`](https://github.com/poulpy-fhe/poulpy/tree/main/poulpy-hal)
- [`poulpy-core`](https://github.com/poulpy-fhe/poulpy/tree/main/poulpy-core)
- [`poulpy-ckks`](https://github.com/poulpy-fhe/poulpy/tree/main/poulpy-ckks) (backend wiring opt-in via `enable-ckks`)
- [`poulpy-bin-fhe`](https://github.com/poulpy-fhe/poulpy/tree/main/poulpy-bin-fhe)

## 🚩 Safety and Requirements

To avoid illegal hardware instructions (SIGILL) on unsupported CPUs, the backends are **opt-in** and **only build when explicitly requested**.

| Feature | CPU target features required |
|---------|------------------------------|
| `enable-avx512f` (builds `FFT64Avx512` and `NTT120Avx512`) | `AVX512F` (AVX2 + FMA implied by AVX-512F; checked at runtime by `FFT64Avx512`) |
| `enable-ifma` (additionally builds `NTT126Ifma`) | `AVX512F` + `AVX512IFMA` + `AVX512VL` + `BMI2` + `ADX` |

If a feature is enabled but the target does not provide the required capabilities, the build **fails immediately with a clear error message**, rather than generating invalid binaries.

When neither feature is enabled, this crate compiles as an empty shell. That keeps the workspace portable on hosts such as macOS ARM, but code that imports AVX-512 backend types must enable the matching feature.

## ⚙️ Building

For the AVX-512F-only `FFT64Avx512` and `NTT120Avx512` backends:

```bash
RUSTFLAGS="-C target-feature=+avx512f" \
cargo build --features enable-avx512f
```

For all three backends (AVX-512F + IFMA + BMI2 + ADX):

```bash
RUSTFLAGS="-C target-feature=+avx512f,+avx512ifma,+avx512vl,+bmi2,+adx" \
cargo build --features enable-ifma
```

On a host that natively supports the required instructions, `target-cpu=native` also works:

```bash
RUSTFLAGS="-C target-cpu=native" \
cargo build --features enable-ifma
```

### Running an example

```bash
RUSTFLAGS="-C target-feature=+avx512f,+avx512ifma,+avx512vl,+bmi2,+adx" \
cargo run --example <name> --features enable-ifma
```

### Running benchmarks

```bash
RUSTFLAGS="-C target-feature=+avx512f,+avx512ifma,+avx512vl,+bmi2,+adx" \
cargo bench --features enable-ifma
```

### Running Tests

```bash
RUSTFLAGS="-C target-feature=+avx512f" \
cargo test -p poulpy-cpu-avx512 --features enable-avx512f
```

To include CKKS backend wiring and IFMA in the AVX-512 test build:

```bash
RUSTFLAGS="-C target-feature=+avx512f,+avx512ifma,+avx512vl,+bmi2,+adx" \
cargo test -p poulpy-cpu-avx512 --features enable-ifma,enable-ckks
```

## Basic Usage

```rust
use poulpy_cpu_avx512::{FFT64Avx512, NTT120Avx512};
use poulpy_hal::{api::ModuleNew, layouts::Module};

let log_n: usize = 10;

// f64 FFT backend (AVX-512F)
let module: Module<FFT64Avx512> = Module::<FFT64Avx512>::new(1 << log_n);

// Q120 NTT backend (AVX-512F, CRT over four ~30-bit primes)
let module: Module<NTT120Avx512> = Module::<NTT120Avx512>::new(1 << log_n);
```

With `enable-ifma`, `NTT126Ifma` is also available:

```rust
use poulpy_cpu_avx512::NTT126Ifma;
use poulpy_hal::{api::ModuleNew, layouts::Module};

let log_n: usize = 10;
// Q126 NTT backend (AVX-512-IFMA, CRT over three ~42-bit primes)
let module: Module<NTT126Ifma> = Module::<NTT126Ifma>::new(1 << log_n);
```

Each backend is usable transparently anywhere Poulpy expects a backend type (`poulpy-hal`, `poulpy-core`, `poulpy-ckks`, `poulpy-bin-fhe`).

## 🤝 Contributors

To implement your own Poulpy backend (SIMD or accelerator):

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

No modifications to those crates are required — the HAL provides the extension points. Only operations that need a faster implementation require explicit overrides; everything else is inherited from the `default` layer for free.

---

For questions or guidance, feel free to open an issue or discussion in the repository.
