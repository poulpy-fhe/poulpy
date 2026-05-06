//! AVX512-IFMA accelerated NTT CPU backend for the Poulpy lattice cryptography library.
//!
//! This module provides [`NTT126Ifma`], an AVX512-IFMA accelerated backend implementation for
//! [`poulpy_hal`] that uses IFMA NTT arithmetic (CRT over three ~42-bit primes). The
//! scalar reference for these kernels lives in the [`reference`] submodule.
//!
//! # Current acceleration status
//!
//! | Domain | Status |
//! |-|-|
//! | Coefficient-domain (`Znx*`) | AVX-512F (reuses `crate::znx_avx512`) |
//! | NTT forward/inverse | AVX512-IFMA (`kernels` module) |
//! | mat_vec BBC product (SVP/VMP hot path) | AVX512-IFMA (`mat_vec_ifma` module) |
//! | VecZnxBig add/sub/negate | shared `i128` helpers wired through the HAL implementation |
//! | VecZnxBig normalization | shared `i128` normalization helpers wired through the HAL implementation |
//!
//! # Scalar types
//!
//! - `ScalarPrep = Q120bScalar` — shared 4-lane prep scalar (three active residues plus padding).
//! - `ScalarBig  = i128` — CRT-reconstructed large coefficients.

pub(crate) mod bbc_meta;
pub(crate) mod convolution;
pub(crate) mod kernels;
pub(crate) mod mat_vec_ifma;
pub(crate) mod module;
mod prim;
pub(crate) mod primes;
pub(crate) mod reference;
pub(crate) mod svp;
pub(crate) mod tables;
pub(crate) mod traits;
pub(crate) mod types;
pub(crate) mod vec_znx_dft;
pub(crate) mod vmp;
mod znx;

#[cfg(test)]
mod tests;

/// AVX512-IFMA accelerated NTT CPU backend for Poulpy HAL.
///
/// `NTT126Ifma` is a zero-sized marker type that selects the AVX512-IFMA accelerated NTT backend
/// when used as the type parameter `B` in [`poulpy_hal::layouts::Module<B>`](poulpy_hal::layouts::Module)
/// and related HAL types. It implements all open extension point (OEP) traits from
/// `poulpy_hal::oep`.
///
/// # Backend characteristics
///
/// - **ScalarPrep**: `Q120bScalar` — shared 4-lane prep scalar with three CRT residues plus one padding lane.
/// - **ScalarBig**: `i128` — large-coefficient ring elements use 128-bit signed integers.
/// - **Prime set**: `Primes42` (three ~42-bit primes, Q ≈ 2^126).
///
/// # CPU feature requirements
///
/// **Runtime check**: [`Module::new()`](poulpy_hal::api::ModuleNew::new) verifies that
/// the CPU supports AVX512-F, AVX512-IFMA, AVX512-VL, BMI2, and ADX. If a required
/// feature is missing, the constructor panics.
///
/// # Thread safety
///
/// `NTT126Ifma` is `Send + Sync` (derived from being a zero-sized, field-less struct).
#[derive(Debug, Clone, Copy)]
pub struct NTT126Ifma;

use poulpy_cpu_ref::reference::ntt120::{I128BigOps, I128NormalizeOps, vec_znx_big::AssignOp};

use crate::vec_znx_big_avx512::{
    nfc_final_step_add_assign_avx512, nfc_final_step_add_assign_scalar, nfc_final_step_assign_avx512,
    nfc_final_step_assign_scalar, nfc_final_step_sub_assign_avx512, nfc_final_step_sub_assign_scalar,
    nfc_middle_step_add_assign_avx512, nfc_middle_step_add_assign_scalar, nfc_middle_step_assign_avx512,
    nfc_middle_step_assign_scalar, nfc_middle_step_avx512, nfc_middle_step_scalar, nfc_middle_step_sub_assign_avx512,
    nfc_middle_step_sub_assign_scalar, vi128_add_assign_avx512, vi128_add_avx512, vi128_add_small_assign_avx512,
    vi128_add_small_avx512, vi128_from_small_avx512, vi128_neg_from_small_avx512, vi128_negate_assign_avx512,
    vi128_negate_avx512, vi128_sub_assign_avx512, vi128_sub_avx512, vi128_sub_negate_assign_avx512, vi128_sub_small_a_avx512,
    vi128_sub_small_assign_avx512, vi128_sub_small_b_avx512, vi128_sub_small_negate_assign_avx512,
};

impl I128BigOps for NTT126Ifma {
    #[inline(always)]
    fn i128_add(res: &mut [i128], a: &[i128], b: &[i128]) {
        unsafe { vi128_add_avx512(res.len(), res, a, b) }
    }
    #[inline(always)]
    fn i128_add_assign(res: &mut [i128], a: &[i128]) {
        unsafe { vi128_add_assign_avx512(res.len(), res, a) }
    }
    #[inline(always)]
    fn i128_add_small(res: &mut [i128], a: &[i128], b: &[i64]) {
        unsafe { vi128_add_small_avx512(res.len(), res, a, b) }
    }
    #[inline(always)]
    fn i128_add_small_assign(res: &mut [i128], a: &[i64]) {
        unsafe { vi128_add_small_assign_avx512(res.len(), res, a) }
    }
    #[inline(always)]
    fn i128_sub(res: &mut [i128], a: &[i128], b: &[i128]) {
        unsafe { vi128_sub_avx512(res.len(), res, a, b) }
    }
    #[inline(always)]
    fn i128_sub_assign(res: &mut [i128], a: &[i128]) {
        unsafe { vi128_sub_assign_avx512(res.len(), res, a) }
    }
    #[inline(always)]
    fn i128_sub_negate_assign(res: &mut [i128], a: &[i128]) {
        unsafe { vi128_sub_negate_assign_avx512(res.len(), res, a) }
    }
    #[inline(always)]
    fn i128_sub_small_a(res: &mut [i128], a: &[i64], b: &[i128]) {
        unsafe { vi128_sub_small_a_avx512(res.len(), res, a, b) }
    }
    #[inline(always)]
    fn i128_sub_small_b(res: &mut [i128], a: &[i128], b: &[i64]) {
        unsafe { vi128_sub_small_b_avx512(res.len(), res, a, b) }
    }
    #[inline(always)]
    fn i128_sub_small_assign(res: &mut [i128], a: &[i64]) {
        unsafe { vi128_sub_small_assign_avx512(res.len(), res, a) }
    }
    #[inline(always)]
    fn i128_sub_small_negate_assign(res: &mut [i128], a: &[i64]) {
        unsafe { vi128_sub_small_negate_assign_avx512(res.len(), res, a) }
    }
    #[inline(always)]
    fn i128_negate(res: &mut [i128], a: &[i128]) {
        unsafe { vi128_negate_avx512(res.len(), res, a) }
    }
    #[inline(always)]
    fn i128_negate_assign(res: &mut [i128]) {
        unsafe { vi128_negate_assign_avx512(res.len(), res) }
    }
    #[inline(always)]
    fn i128_neg_from_small(res: &mut [i128], a: &[i64]) {
        unsafe { vi128_neg_from_small_avx512(res.len(), res, a) }
    }
    #[inline(always)]
    fn i128_from_small(res: &mut [i128], a: &[i64]) {
        unsafe { vi128_from_small_avx512(res.len(), res, a) }
    }
}

impl I128NormalizeOps for NTT126Ifma {
    #[inline(always)]
    fn nfc_middle_step(base2k: usize, lsh: usize, res: &mut [i64], a: &[i128], carry: &mut [i128]) {
        if base2k <= 64 && res.len() >= 8 {
            unsafe { nfc_middle_step_avx512(base2k as u32, lsh as u32, res.len(), res, a, carry) }
        } else {
            nfc_middle_step_scalar(base2k, lsh, res, a, carry);
        }
    }
    #[inline(always)]
    fn nfc_middle_step_assign(base2k: usize, lsh: usize, res: &mut [i64], carry: &mut [i128]) {
        if base2k <= 64 && res.len() >= 8 {
            unsafe { nfc_middle_step_assign_avx512(base2k as u32, lsh as u32, res.len(), res, carry) }
        } else {
            nfc_middle_step_assign_scalar(base2k, lsh, res, carry);
        }
    }
    #[inline(always)]
    fn nfc_middle_step_into<O: AssignOp>(base2k: usize, lsh: usize, res: &mut [i64], a: &[i128], carry: &mut [i128]) {
        if base2k <= 64 && res.len() >= 8 {
            if O::SUB {
                unsafe { nfc_middle_step_sub_assign_avx512(base2k as u32, lsh as u32, res.len(), res, a, carry) }
            } else {
                unsafe { nfc_middle_step_add_assign_avx512(base2k as u32, lsh as u32, res.len(), res, a, carry) }
            }
        } else if O::SUB {
            nfc_middle_step_sub_assign_scalar(base2k, lsh, res, a, carry);
        } else {
            nfc_middle_step_add_assign_scalar(base2k, lsh, res, a, carry);
        }
    }
    #[inline(always)]
    fn nfc_final_step_assign(base2k: usize, lsh: usize, res: &mut [i64], carry: &mut [i128]) {
        if base2k <= 64 && res.len() >= 8 {
            unsafe { nfc_final_step_assign_avx512(base2k as u32, lsh as u32, res.len(), res, carry) }
        } else {
            nfc_final_step_assign_scalar(base2k, lsh, res, carry);
        }
    }
    #[inline(always)]
    fn nfc_final_step_into<O: AssignOp>(base2k: usize, lsh: usize, res: &mut [i64], carry: &mut [i128]) {
        if base2k <= 64 && res.len() >= 8 {
            if O::SUB {
                unsafe { nfc_final_step_sub_assign_avx512(base2k as u32, lsh as u32, res.len(), res, carry) }
            } else {
                unsafe { nfc_final_step_add_assign_avx512(base2k as u32, lsh as u32, res.len(), res, carry) }
            }
        } else if O::SUB {
            nfc_final_step_sub_assign_scalar(base2k, lsh, res, carry);
        } else {
            nfc_final_step_add_assign_scalar(base2k, lsh, res, carry);
        }
    }
}
