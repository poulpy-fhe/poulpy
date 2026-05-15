# poulpy-cpu-arm

NEON-accelerated CPU backend for the [Poulpy](https://github.com/poulpy-fhe/poulpy)
lattice-cryptography library. Targets AArch64 (Apple Silicon and ARMv8-A
servers); SVE/SVE2 are out of scope for now.

## Backend types

| Type           | `ScalarPrep`   | `ScalarBig` | Domain                                  |
|----------------|----------------|-------------|-----------------------------------------|
| `FFT64Neon`    | `f64`          | `i64`       | f64 FFT (twiddle-factor tables)         |
| `NTT120Neon`   | `Q120bScalar`  | `i128`      | Q120 NTT, CRT over four ~30-bit primes  |

Both implement the same `poulpy_hal::oep` extension points as the AVX backends
and share the HAL contract with `poulpy-cpu-ref`.

## Status

All HAL extension points are wired with NEON kernels.

| Family                                    | NEON | Wired |
|-------------------------------------------|:----:|:-----:|
| `Znx*` (i64 add/sub/negate)               | ✅   | ✅    |
| `I128BigOps` (i128 add/sub/negate, ±i64)  | ✅   | ✅    |
| `I128NormalizeOps` (i128 carry-prop)      | ✅   | ✅    |
| `Ntt*` lazy-modular q120b add/sub/negate  | ✅   | ✅    |
| `NttFromZnx64`, `NttToZnx128`, `NttCFromB`| ✅   | ✅    |
| `NttPack*`, `NttPairwisePack*`            | ✅   | ✅    |
| `NttMulBbb`, `NttMulBbc`, `NttMulBbc*X2`  | ✅   | ✅    |
| `NttDFTExecute` (forward/inverse NTT)     | ✅   | ✅    |
| `vec_znx_idft_apply_consume`              | ✅   | ✅    |
| `ReimArith` (pointwise REIM)              | ✅   | ✅    |
| `Reim4BlkMatVec` (4-block mat-vec)        | ✅   | ✅    |
| `Reim4Convolution` (4 primitives)         | ✅   | ✅    |
| `I64Ops` (i64 block move + by-const conv) | ✅   | ✅    |
| `ReimFFTExecute` (FFT/IFFT butterflies)   | ✅   | ✅    |

The crate ships ~5400 lines of NEON code across `src/neon/`. Total LOC
including the wired trait impls and tests is comparable to the AVX backend.

## Feature flag

```toml
poulpy-cpu-arm = { workspace = true, features = ["enable-neon"] }
```

Without `enable-neon`, the crate compiles to an empty shell on any architecture,
so the workspace remains buildable on x86 and other targets. With `enable-neon`,
the crate requires `target_arch = "aarch64"`; a clear `compile_error!` fires
on any other target.

NEON/ASIMD is part of the AArch64 architectural baseline, so no runtime CPU
feature detection is performed.

## Build & test

### Native AArch64

```bash
cargo build -p poulpy-cpu-arm --features enable-neon
cargo test  -p poulpy-cpu-arm --features enable-neon
```

### Cross-build from x86 with rust-lld + qemu

The workspace `.cargo/config.toml` configures the AArch64 cross-targets to
use `rust-lld` (bundled with rustup) for the musl target and
`aarch64-linux-gnu-gcc` for the GNU target, with `qemu-aarch64-static` as
the runner. To execute the NEON test suite under emulation:

```bash
# one-time install (Arch Linux)
sudo pacman -S qemu-user-static aarch64-linux-gnu-gcc
rustup target add aarch64-unknown-linux-musl

# run the tests under qemu
cargo test -p poulpy-cpu-arm --features enable-neon \
    --target aarch64-unknown-linux-musl
```

The musl target only needs `qemu-aarch64-static` (no cross-cc) because
`rust-lld` links the static binary itself. The GNU target additionally
requires the C cross-toolchain.

### x86 portability

```bash
cargo build -p poulpy-cpu-arm   # empty shell, no NEON code
cargo check --workspace         # default features
```

## Benchmarks

`poulpy-bench` includes ARM backend selection behind the `enable-neon`
feature.

### Apple Silicon / native AArch64

No `--target` flag is needed — the host triple (`aarch64-apple-darwin` on
a Mac, `aarch64-unknown-linux-gnu` on Linux servers) already matches.

```bash
# all backends and benches
cargo bench -p poulpy-bench --features enable-neon

# single bench, filtered to one backend
cargo bench -p poulpy-bench --features enable-neon --bench fft       -- fft_neon
cargo bench -p poulpy-bench --features enable-neon --bench ckks_mul  -- ntt120-neon
cargo bench -p poulpy-bench --features enable-neon --bench ckks_mul  -- fft64-neon
```

The benchmark dispatchers (`for_each_backend!`, `for_each_fft_backend!`,
`for_each_ntt_backend!` in `poulpy-bench/src/lib.rs`) emit one bench per
backend in tier order: `ref → avx → neon → gpu`. Most benches do not need
to mention any specific backend; they pick up `FFT64Neon` / `NTT120Neon`
automatically when `enable-neon` is on. The standalone `fft.rs` bench is
the exception — it has hand-written per-backend entry points
(`bench_fft_ref` / `bench_fft_avx` / `bench_fft_neon` and the `ifft_*`
counterparts).

### Cross-compiling benches from x86 + qemu

The bench crate transitively pulls a small C build (`alloca`), so cross-
compiling it from x86 needs an AArch64 C cross-toolchain. `clang` works
out of the box:

```bash
CC_aarch64_unknown_linux_musl=clang \
CFLAGS_aarch64_unknown_linux_musl="--target=aarch64-linux-musl" \
cargo bench -p poulpy-bench --features enable-neon \
    --target aarch64-unknown-linux-musl --bench fft -- fft_neon
```

qemu cycle counts are not representative of native AArch64 performance;
treat such runs as smoke tests only. The `poulpy-cpu-arm` crate itself
has no `cc-rs` dependency and tests fine under qemu without a cross-cc.

## Example

```rust
use poulpy_cpu_arm::{FFT64Neon, NTT120Neon};
use poulpy_hal::{api::ModuleNew, layouts::Module};

let m_fft:  Module<FFT64Neon>   = Module::<FFT64Neon>::new(1 << 12);
let m_ntt:  Module<NTT120Neon>  = Module::<NTT120Neon>::new(1 << 12);
// Use m_fft / m_ntt with poulpy_hal / poulpy_core APIs.
```

## Numerical contract

- Integer and modular operations (`Znx*`, `I128BigOps`, `Ntt*`,
  `NttDFTExecute`) are bit-exact against `poulpy-cpu-ref`.
- FFT-domain (`ReimArith`, `Reim4*`, `I64Ops`) operations match the
  reference within ULP tolerance — the NEON kernels use FMA where the
  scalar reference does not, so individual rounding bits may differ but
  the magnitude stays within the documented FFT tolerance.

See `poulpy-hal/docs/backend_safety_contract.md` for the full backend
contract.

## Future work

- **`reim4_extract_1blk` & related step parameter**: the AVX backend uses
  a `step = m >> 2` __m256i stride; the NEON port currently uses the
  doubled f64-unit stride directly. Re-verify on hardware that the stride
  matches the AVX layout exactly.
- **Hand-written assembly leaves**: the AVX backend has `.s` files for
  `fft16` / `ifft16`. The handoff is briefed in
  `docs/poulpy-cpu-arm-fft16-asm-handoff.md` and is gated on a real
  AArch64 bench delta — defer until benchmarks on Apple Silicon prove the
  intrinsic version is the bottleneck.
- **SVE/SVE2**: intentionally not mixed into the NEON code path. Future
  SVE support will land as separate `FFT64Sve` / `NTT120Sve` backend
  types.

## File map

```
poulpy-cpu-arm/
├── Cargo.toml
├── README.md
└── src/
    ├── lib.rs                       # crate gate, CoreImpl
    ├── hal_impl.rs                  # HAL macro orchestration
    ├── hal_impl/                    # 15 macro modules wiring HAL methods
    ├── fft64/                       # FFT64Neon backend type + Backend impl
    │   ├── mod.rs
    │   ├── module.rs                # Backend handle (FFT/IFFT tables)
    │   ├── reim.rs                  # ReimArith / Reim4* / I64Ops impls
    │   ├── znx.rs                   # Znx* impls (NEON kernels via aliases)
    │   └── tests.rs                 # cross_backend_test_suite!
    ├── ntt120/                      # NTT120Neon backend type + Backend impl
    │   ├── mod.rs
    │   ├── module.rs                # Backend handle (NTT/iNTT tables)
    │   ├── prim.rs                  # Ntt* impls (NEON kernels)
    │   ├── vec_znx_big.rs           # I128BigOps / I128NormalizeOps impls
    │   ├── znx.rs                   # Znx* impls
    │   └── tests.rs                 # cross_backend_test_suite!
    └── neon/                        # NEON kernel modules (~5400 LOC)
        ├── mod.rs
        ├── q120.rs                  # Q120 split-register helpers (shared)
        ├── znx.rs                   # i64 add/sub/negate
        ├── normalize.rs             # i128 carry-propagation (nfc_*)
        ├── vec_znx_big.rs           # i128 paired-register helpers (vi128_*)
        ├── ntt120_arithmetic.rs     # q120b lazy-modular helpers
        ├── ntt120_convert.rs        # i64↔q120b↔i128, packs, consume
        ├── ntt120_mat_vec.rs        # bbb / bbc matrix-vector products
        ├── ntt120_ntt.rs            # forward / inverse NTT butterflies
        ├── reim_arith.rs            # pointwise REIM + i64↔f64 conversions
        ├── reim4_arith.rs           # Reim4 block move + mat-vec
        ├── reim4_conv.rs            # Reim4 convolution kernels
        └── conv_i64.rs              # I64Ops kernels
```
