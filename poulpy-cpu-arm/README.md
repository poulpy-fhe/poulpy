# 🐙 Poulpy-CPU-ARM

**Poulpy-CPU-ARM** is a Rust crate that provides a **NEON / ASIMD accelerated CPU backend for Poulpy**, targeting AArch64 (Apple Silicon, Neoverse, …).

This backend implements the Poulpy HAL extension traits and can be used by:

- [`poulpy-hal`](https://github.com/poulpy-fhe/poulpy/tree/main/poulpy-hal)
- [`poulpy-core`](https://github.com/poulpy-fhe/poulpy/tree/main/poulpy-core)
- [`poulpy-ckks`](https://github.com/poulpy-fhe/poulpy/tree/main/poulpy-ckks) (backend wiring opt-in via `enable-ckks`)

## 🚩 Safety and Requirements

To avoid illegal hardware instructions on non-AArch64 CPUs, this backend is **opt-in** and **only builds when explicitly requested**.

| Requirement | Status |
|------------|--------|
| Cargo feature flag | `--features enable-neon` **must be enabled** |
| CPU architecture | `aarch64` |
| CPU target features | NEON / ASIMD (part of the AArch64 architectural baseline) |

If `enable-neon` is enabled but the target is not `aarch64`, the build **fails immediately with a clear error message** (`compile_error!` in `lib.rs`), rather than generating invalid binaries.

When `enable-neon` is **not** enabled, this crate is simply skipped and Poulpy automatically falls back to the portable `poulpy-cpu-ref` backend. This ensures that Poulpy's workspace remains portable on x86 hosts and CI runners.

## ⚙️ Building with the NEON backend enabled

NEON / ASIMD is part of the AArch64 baseline, so no `RUSTFLAGS` target-feature flag is required:

```bash
cargo build --features enable-neon
```

### Running an example

```bash
cargo run --example <name> --features enable-neon
```

### Running benchmarks

```bash
cargo bench --features enable-neon
```

### Running tests

```bash
cargo test -p poulpy-cpu-arm --features enable-neon
```

To include CKKS and Rayon backend wiring in the NEON test build:

```bash
cargo test -p poulpy-cpu-arm --features enable-neon,enable-rayon,enable-ckks
```

## Basic Usage

This crate exposes two NEON-accelerated backends and Rayon-scheduled variants of both:

```rust
use poulpy_cpu_arm::{FFT64Neon, FFT64NeonRayon, NTT4x30Neon, NTT4x30NeonRayon};
use poulpy_hal::{api::ModuleNew, layouts::Module};

let log_n: usize = 10;

// f64 FFT backend (NEON)
let module: Module<FFT64Neon> = Module::<FFT64Neon>::new(1 << log_n);

// Q120 NTT backend (NEON, CRT over four ~30-bit primes)
let module: Module<NTT4x30Neon> = Module::<NTT4x30Neon>::new(1 << log_n);

// Rayon variants use the same NEON kernels with backend-owned scheduling
let module: Module<FFT64NeonRayon> = Module::<FFT64NeonRayon>::new(1 << log_n);
let module: Module<NTT4x30NeonRayon> = Module::<NTT4x30NeonRayon>::new(1 << log_n);
```

The serial backends require `enable-neon`; the Rayon variants additionally require `enable-rayon`. All four are usable anywhere Poulpy expects a backend type in the HAL/core/CKKS/bin-FHE layers.

## Numerical contract

- Integer / modular operations (`Znx*`, `I128BigOps`, `Ntt*`, `NttDFTExecute`) are bit-exact against `poulpy-cpu-ref`.
- FFT-domain operations (`ReimArith`, `Reim4*`, `I64Ops`, `ReimFFTExecute`) match the reference within ULP tolerance — NEON kernels use FMA where the scalar reference does not.

See `poulpy-hal/docs/backend_safety_contract.md` for the full backend contract.

## 🤝 Contributors

To implement your own Poulpy backend (SIMD or accelerator):

1. Define a backend struct and implement the `Backend` trait from `poulpy-hal`.
2. For each HAL operation family, either call the blanket default or implement the OEP trait directly with a custom dispatch.
3. For each `poulpy-core` operation family, either call the corresponding `impl_*_defaults_full!` macro to inherit the portable implementation, or implement the OEP trait directly to override it.
4. Optionally, do the same for `poulpy-ckks` behind a backend-owned `enable-ckks` feature using the `impl_ckks_*_defaults!` macros or direct OEP trait implementations.

At every layer the macro and the direct implementation are mutually exclusive per operation family: the macro opts the backend into the portable `default` path, while a direct OEP impl replaces it entirely. There is no requirement to use the macros — a backend that needs full control can implement every OEP trait by hand.

Your backend will automatically integrate with the backend-generic layers:

* `poulpy-hal`
* `poulpy-core`
* `poulpy-ckks`

No modifications to those crates are required — the HAL provides the extension points. Only operations that need a faster implementation require explicit overrides; everything else is inherited from the `default` layer for free.

---

For questions or guidance, feel free to open an issue or discussion in the repository.
