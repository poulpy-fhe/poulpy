//! Low-level NEON kernels for the `poulpy-cpu-arm` backend.
//!
//! Every kernel here is wired into the HAL trait impls in [`crate::fft64`] and
//! [`crate::ntt120`] under `#[cfg(target_arch = "aarch64")]`. The submodule
//! structure mirrors the AArch64 NEON shape rather than a direct AVX2
//! translation: loads are contiguous, registers are 128-bit (`int64x2_t` /
//! `uint64x2_t` / `uint32x4_t`), and tails fall back to the portable reference
//! functions.
//!
//! Each kernel is bit-exact against its `poulpy-cpu-ref` oracle for integer /
//! modular operations. `cargo test -p poulpy-cpu-arm --features enable-neon`
//! on AArch64 (or under `qemu-aarch64-static`) runs the unit tests in each
//! submodule alongside the cross-backend HAL suites.

pub(crate) mod conv_i64;
pub(crate) mod fft;
pub(crate) mod normalize;
pub(crate) mod ntt120_arithmetic;
pub(crate) mod ntt120_convert;
pub(crate) mod ntt120_mat_vec;
pub(crate) mod ntt120_ntt;
pub(crate) mod q120;
pub(crate) mod reim4_arith;
pub(crate) mod reim4_conv;
pub(crate) mod reim_arith;
pub(crate) mod vec_znx_big;
pub(crate) mod znx;
