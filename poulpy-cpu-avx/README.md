# 🐙 Poulpy-CPU-AVX

**Poulpy-CPU-AVX** is a Rust crate that provides an **AVX2 + FMA accelerated CPU backend for Poulpy**.

This backend implements the Poulpy HAL extension traits and can be used by:

- [`poulpy-hal`](https://github.com/poulpy-fhe/poulpy/tree/main/poulpy-hal)
- [`poulpy-core`](https://github.com/poulpy-fhe/poulpy/tree/main/poulpy-core)
- [`poulpy-ckks`](https://github.com/poulpy-fhe/poulpy/tree/main/poulpy-ckks) (backend wiring opt-in via `enable-ckks`)
- [`poulpy-bin-fhe`](https://github.com/poulpy-fhe/poulpy/tree/main/poulpy-bin-fhe)

## 🚩 Safety and Requirements

To avoid illegal hardware instructions (SIGILL) on unsupported CPUs, this backend is **opt-in** and **only builds when explicitly requested**.

| Requirement | Status |
|------------|--------|
| Cargo feature flag | `--features enable-avx` **must be enabled** |
| CPU architecture | `x86_64` |
| CPU target features | `AVX2` + `FMA` |

If `enable-avx` is enabled but the target does not provide these capabilities, the build **fails immediately with a clear error message**, rather than generating invalid binaries.

When `enable-avx` is **not** enabled, this crate is simply skipped and Poulpy automatically falls back to the portable `poulpy-cpu-ref` backend. This ensures that Poulpy's workspace remains portable (e.g. for macOS ARM).

## ⚙️ Building with the AVX backend enabled

Because the compiler must generate AVX2 + FMA instructions, both the Cargo feature and CPU target flags must be specified:

```bash
RUSTFLAGS="-C target-feature=+avx2,+fma" \
cargo build --features enable-avx
```

### Running an example

```bash
RUSTFLAGS="-C target-feature=+avx2,+fma" \
cargo run --example <name> --features enable-avx
```

### Running benchmarks

```bash
RUSTFLAGS="-C target-feature=+avx2,+fma" \
cargo bench --features enable-avx
```

### Running Tests

```bash
RUSTFLAGS="-C target-feature=+avx2,+fma" \
cargo test -p poulpy-cpu-avx --features enable-avx
```

To include CKKS backend wiring in the AVX test build:

```bash
RUSTFLAGS="-C target-feature=+avx2,+fma" \
cargo test -p poulpy-cpu-avx --features enable-avx,enable-ckks
```

## Basic Usage

This crate exposes two AVX2-accelerated backends:

```rust
use poulpy_cpu_avx::{FFT64Avx, NTT120Avx};
use poulpy_hal::{api::ModuleNew, layouts::Module};

let log_n: usize = 10;

// f64 FFT backend (AVX2 + FMA)
let module: Module<FFT64Avx> = Module::<FFT64Avx>::new(1 << log_n);

// Q120 NTT backend (AVX2, CRT over four ~30-bit primes)
let module: Module<NTT120Avx> = Module::<NTT120Avx>::new(1 << log_n);
```

Once compiled with `enable-avx`, both backends are usable transparently anywhere Poulpy expects a backend type (`poulpy-hal`, `poulpy-core`, `poulpy-ckks`, `poulpy-bin-fhe`).

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
