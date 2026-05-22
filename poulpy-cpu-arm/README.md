# 🐙 Poulpy-CPU-ARM

**Poulpy-CPU-ARM** is a Rust crate that provides a **NEON / ASIMD accelerated CPU backend for Poulpy**, targeting AArch64 (Apple Silicon, Neoverse, …).

It exposes two backends, gated behind one Cargo feature:

- **`FFT64Neon`** — f64 complex-FFT backend. Combines hand-written NEON REIM butterflies and Reim4 vector-matrix kernels with the portable `poulpy-cpu-ref` arithmetic for non-hot paths.
- **`NTT120Neon`** — Q120 NTT backend (CRT over four ~30-bit primes). NEON-accelerated NTT/INTT, mat-vec, and conversion kernels; falls back to `poulpy-cpu-ref` for operations where the autovectorized scalar code matches or beats NEON.

Both backends are gated on `enable-neon` and are built only on `target_arch = "aarch64"`.

This backend implements the Poulpy HAL extension traits and can be used by:

- [`poulpy-hal`](https://github.com/poulpy-fhe/poulpy/tree/main/poulpy-hal)
- [`poulpy-core`](https://github.com/poulpy-fhe/poulpy/tree/main/poulpy-core)
- [`poulpy-ckks`](https://github.com/poulpy-fhe/poulpy/tree/main/poulpy-ckks) (backend wiring opt-in via `enable-ckks`)
- [`poulpy-bin-fhe`](https://github.com/poulpy-fhe/poulpy/tree/main/poulpy-bin-fhe) through the crate's explicit `enable-neon` integration feature

## 🚩 Safety and Requirements

To avoid illegal hardware instructions on non-AArch64 CPUs, this backend is **opt-in** and **only builds when explicitly requested**.

| Requirement | Status |
|------------|--------|
| Cargo feature flag | `--features enable-neon` **must be enabled** |
| CPU architecture | `aarch64` (NEON / ASIMD is part of the architectural baseline) |
| Runtime CPU detection | not required — NEON is mandatory on AArch64 |

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

For instance, the bundled performance / sanity bench:

```bash
cargo run --release -p poulpy-cpu-arm \
    --example bench_neon_vs_ref --features enable-neon
```

### Running benchmarks

```bash
cargo bench --features enable-neon
```

### Running tests

```bash
cargo test -p poulpy-cpu-arm --features enable-neon
```

To include CKKS backend wiring in the test build:

```bash
cargo test -p poulpy-cpu-arm --features enable-neon,enable-ckks
```

### Cross-compiling from x86 with qemu

The workspace `.cargo/config.toml` configures the AArch64 cross-targets to use `rust-lld` for the musl target and `aarch64-linux-gnu-gcc` for the GNU target, with `qemu-aarch64-static` as the runner:

```bash
# one-time setup
rustup target add aarch64-unknown-linux-musl
sudo apt install -y qemu-user-static    # or: sudo pacman -S qemu-user-static

# test the NEON backend under emulation
cargo test -p poulpy-cpu-arm --features enable-neon,enable-ckks \
    --target aarch64-unknown-linux-musl
```

qemu emulation overhead dominates the wall-clock time **and distorts SIMD-vs-scalar ratios** (it inflates wins on some kernels and flips the sign of others). Treat qemu runs as a correctness gate only; never use them for performance decisions.

## Basic Usage

```rust
use poulpy_cpu_arm::{FFT64Neon, NTT120Neon};
use poulpy_hal::{api::ModuleNew, layouts::Module};

let log_n: usize = 10;

// f64 FFT backend (NEON)
let module: Module<FFT64Neon> = Module::<FFT64Neon>::new(1 << log_n);

// Q120 NTT backend (NEON, CRT over four ~30-bit primes)
let module: Module<NTT120Neon> = Module::<NTT120Neon>::new(1 << log_n);
```

Once compiled with `enable-neon`, both backends are usable anywhere Poulpy expects a backend type in the HAL/core/CKKS layers. `poulpy-bin-fhe` has a separate `enable-neon` feature for its current crate-local integration.

## Numerical contract

- Integer / modular operations (`Znx*`, `I128BigOps`, `Ntt*`, `NttDFTExecute`) are bit-exact against `poulpy-cpu-ref`.
- FFT-domain operations (`ReimArith`, `Reim4*`, `I64Ops`, `ReimFFTExecute`) match the reference within ULP tolerance — NEON kernels use FMA where the scalar reference does not, so individual rounding bits may differ.

See `poulpy-hal/docs/backend_safety_contract.md` for the full backend contract.

## Performance work

The NEON backend is **not yet at parity** with `poulpy-cpu-avx` on equivalent kernels. The optimization roadmap, current per-kernel gaps, file-level pointers, and the priority order for closing them are tracked in [`docs/NEON_OPTIMIZATION_HANDOFF.md`](docs/NEON_OPTIMIZATION_HANDOFF.md). Start there before touching kernel code.

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

No modifications to those crates are required — the HAL provides the extension points. Scheme crates that still carry crate-specific backend glue, such as parts of `poulpy-bin-fhe` in v0.6.0, may need follow-up integration work. Only operations that need a faster implementation require explicit overrides; everything else is inherited from the `default` layer for free.

---

For questions or guidance, feel free to open an issue or discussion in the repository.
