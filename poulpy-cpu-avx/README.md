# 🐙 Poulpy-CPU-AVX

**Poulpy-CPU-AVX** is a Rust crate that provides an **AVX2 + FMA accelerated CPU backend for Poulpy**.

This backend implements the Poulpy HAL extension traits and can be used by:

- [`poulpy-hal`](https://github.com/poulpy-fhe/poulpy/tree/main/poulpy-hal)
- [`poulpy-core`](https://github.com/poulpy-fhe/poulpy/tree/main/poulpy-core)
- [`poulpy-ckks`](https://github.com/poulpy-fhe/poulpy/tree/main/poulpy-ckks) (backend wiring opt-in via `enable-ckks`)
- [`poulpy-bin-fhe`](https://github.com/poulpy-fhe/poulpy/tree/main/poulpy-bin-fhe) (backend-agnostic; its tests are instantiated here with `bin_fhe_backend_test_suite!`)

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

To include CKKS and Rayon backend wiring in the AVX test build:

```bash
RUSTFLAGS="-C target-feature=+avx2,+fma" \
cargo test -p poulpy-cpu-avx --features enable-avx,enable-rayon,enable-ckks
```

## Basic Usage

This crate exposes two AVX2-accelerated backends and Rayon-scheduled variants of both:

```rust
use poulpy_cpu_avx::{FFT64Avx, FFT64AvxRayon, NTT4x30Avx, NTT4x30AvxRayon};
use poulpy_hal::{api::ModuleNew, layouts::Module};

let log_n: usize = 10;

// f64 FFT backend (AVX2 + FMA)
let module: Module<FFT64Avx> = Module::<FFT64Avx>::new(1 << log_n);

// Q120 NTT backend (AVX2, CRT over four ~30-bit primes)
let module: Module<NTT4x30Avx> = Module::<NTT4x30Avx>::new(1 << log_n);

// Rayon variants use the same AVX2 kernels with backend-owned scheduling
let module: Module<FFT64AvxRayon> = Module::<FFT64AvxRayon>::new(1 << log_n);
let module: Module<NTT4x30AvxRayon> = Module::<NTT4x30AvxRayon>::new(1 << log_n);
```

The serial backends require `enable-avx`; the Rayon variants additionally require `enable-rayon`. All four are usable anywhere Poulpy expects a backend type in the HAL/core/CKKS/bin-FHE layers.

## 🤝 Contributors

To implement your own Poulpy backend (SIMD or accelerator):

1. Define a backend struct and implement the `Backend` trait from `poulpy-hal`.
2. Implement each required HAL OEP method and inherit or override its derived defaults.
3. Implement the core `*Impl` traits, or use family macros to select reference algorithms and derived defaults.
4. Optionally, do the same for `poulpy-ckks` behind a backend-owned `enable-ckks` feature using the `impl_ckks_*_reference!` macros or direct OEP trait implementations.

Use either a family macro or a handwritten implementation of the same core
`*Impl` trait. Reference helpers remain callable for methods you forward, while
derived defaults reuse selected backend operations. Validate the resulting
backend with the shared conformance tests; the [core backend guide](../poulpy-core/docs/core-contracts.md)
describes the contracts and test setup.

Your backend will automatically integrate with the backend-generic layers:

* `poulpy-hal`
* `poulpy-core`
* `poulpy-ckks`

No modifications to those crates are required — the HAL provides the extension points. Only operations that need a faster implementation require explicit overrides, and each override is validated by the parity test against an attested backend (attestation is transitive back to `reference`: the portable backend runs it directly, and your backend may test against the portable backend or against any backend already attested), correct only when that test passes; everything else runs the `reference` layer, which is the implementation.

---

For questions or guidance, feel free to open an issue or discussion in the repository.

## Binary-FHE integration

`enable-bin-fhe` selects the binary-FHE reference circuits and registers the
complete paired and same-backend lifecycle suites. Backend opt-in and test
registration share one declaration in `src/bin_fhe_impl.rs`. Custom operations
implement the binary-FHE `*Impl` contracts, including their scratch queries.
Enable `enable-avx` as well to select this accelerated backend; `enable-rayon` adds its parallel variants.

```sh
RUSTFLAGS="-C target-feature=+avx2,+fma" cargo test -p poulpy-cpu-avx --features enable-avx,enable-rayon,enable-bin-fhe bin_fhe_parity
```
