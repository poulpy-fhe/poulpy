# 🐙 Poulpy-CPU-ARM

NEON / ASIMD CPU backend for Poulpy, targeting AArch64.

Two backends, both gated on the `enable-neon` Cargo feature and built only on `target_arch = "aarch64"`:

- **`FFT64Neon`** — f64 complex-FFT backend (NEON REIM butterflies + Reim4 vector-matrix kernels).
- **`NTT120Neon`** — Q120 NTT backend (CRT over four ~30-bit primes); NEON-accelerated NTT/INTT, mat-vec, conversions, and VMP.

Implements the Poulpy HAL extension traits and is consumable by `poulpy-hal`, `poulpy-core`, `poulpy-ckks` (opt-in via `enable-ckks`), and `poulpy-bin-fhe` (via its `enable-neon` feature).

## Safety and Requirements

NEON / ASIMD is part of the AArch64 architectural baseline; no runtime feature detection is required. With `enable-neon` on a non-aarch64 target, the build fails immediately via `compile_error!`. Without `enable-neon`, the crate is a no-op shell and callers fall back to `poulpy-cpu-ref`.

## Building

```bash
cargo build --features enable-neon
cargo test  -p poulpy-cpu-arm --features enable-neon
cargo test  -p poulpy-cpu-arm --features enable-neon,enable-ckks
cargo bench --features enable-neon
```

### Cross-compiling from x86 with qemu

The workspace `.cargo/config.toml` wires `aarch64-unknown-linux-{gnu,musl}` to `qemu-aarch64-static`:

```bash
rustup target add aarch64-unknown-linux-musl
sudo pacman -S qemu-user-static   # or: sudo apt install -y qemu-user-static

cargo test -p poulpy-cpu-arm --features enable-neon,enable-ckks \
    --target aarch64-unknown-linux-musl
```

qemu distorts SIMD-vs-scalar ratios. Use it as a correctness gate only, never for performance decisions.

## Usage

```rust
use poulpy_cpu_arm::{FFT64Neon, NTT120Neon};
use poulpy_hal::{api::ModuleNew, layouts::Module};

let log_n: usize = 10;
let module: Module<FFT64Neon>  = Module::<FFT64Neon>::new(1 << log_n);
let module: Module<NTT120Neon> = Module::<NTT120Neon>::new(1 << log_n);
```

## Numerical contract

- Integer / modular operations (`Znx*`, `I128BigOps`, `Ntt*`, `NttDFTExecute`) are bit-exact against `poulpy-cpu-ref`.
- FFT-domain operations (`ReimArith`, `Reim4*`, `I64Ops`, `ReimFFTExecute`) match the reference within ULP tolerance — NEON kernels use FMA where the scalar reference does not.

See `poulpy-hal/docs/backend_safety_contract.md` for the full backend contract.

## Contributors

To implement your own Poulpy backend (SIMD or accelerator):

1. Define a backend struct and implement the `Backend` trait from `poulpy-hal`.
2. For each HAL operation family, either call the blanket default or implement the OEP trait directly.
3. For each `poulpy-core` family, either call the matching `impl_*_defaults_full!` macro or implement the OEP trait directly.
4. Optionally, do the same for `poulpy-ckks` behind a backend-owned `enable-ckks` feature.

At every layer the macro and the direct impl are mutually exclusive per operation family: the macro inherits the portable `default` path; a direct OEP impl replaces it entirely.
