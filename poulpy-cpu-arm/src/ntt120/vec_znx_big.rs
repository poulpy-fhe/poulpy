//! Large-coefficient (i128) ring element vector support for [`NTT120Neon`](super::NTT120Neon).
//!
//! Phase 3: on AArch64, both `I128BigOps` and `I128NormalizeOps` are wired
//! to the NEON kernels in `crate::neon`. On other targets the impl bodies
//! are empty so the trait defaults apply (the negative-gate path on x86
//! still compiles cleanly while `compile_error!` aborts the build at the
//! root of `lib.rs`).

use super::NTT120Neon;
#[cfg(target_arch = "aarch64")]
use poulpy_cpu_ref::reference::ntt120::vec_znx_big::AssignOp;
use poulpy_cpu_ref::reference::ntt120::{I128BigOps, I128NormalizeOps};

#[cfg(target_arch = "aarch64")]
use crate::neon::normalize::{
    nfc_final_step_assign_neon, nfc_final_step_into_neon, nfc_middle_step_assign_neon, nfc_middle_step_into_neon,
    nfc_middle_step_neon,
};
#[cfg(target_arch = "aarch64")]
use crate::neon::vec_znx_big::{vi128_from_small_neon, vi128_neg_from_small_neon};

// I128 add/sub/negate family: trait defaults (autovec scalar) are used. The
// hand-NEON paired-register kernels did not beat scalar because aarch64 has
// no native 128-bit add and no widening i64×i64. The widening i64→i128
// (`i128_from_small`, `i128_neg_from_small`) keeps a NEON impl since the
// sign-extension via `vshrq_n_s64::<63>` is a real single-instruction win.
#[cfg(target_arch = "aarch64")]
impl I128BigOps for NTT120Neon {
    #[inline(always)]
    fn i128_neg_from_small(res: &mut [i128], a: &[i64]) {
        vi128_neg_from_small_neon(res.len(), res, a);
    }
    #[inline(always)]
    fn i128_from_small(res: &mut [i128], a: &[i64]) {
        vi128_from_small_neon(res.len(), res, a);
    }
}

#[cfg(not(target_arch = "aarch64"))]
impl I128BigOps for NTT120Neon {}

impl poulpy_cpu_ref::hal_defaults::ScalarBigHadamardProduct for NTT120Neon {
    #[inline(always)]
    fn scalar_big_hadamard_product(res: &mut [i128], a: &[i64], b: &[i64]) {
        <Self as I128BigOps>::i128_hadamard_product_i64(res, a, b)
    }
}

#[cfg(target_arch = "aarch64")]
impl I128NormalizeOps for NTT120Neon {
    #[inline(always)]
    fn nfc_middle_step(base2k: usize, lsh: usize, res: &mut [i64], a: &[i128], carry: &mut [i128]) {
        nfc_middle_step_neon(base2k, lsh, res, a, carry);
    }
    #[inline(always)]
    fn nfc_middle_step_into<O: AssignOp>(base2k: usize, lsh: usize, res: &mut [i64], a: &[i128], carry: &mut [i128]) {
        nfc_middle_step_into_neon::<O>(base2k, lsh, res, a, carry);
    }
    #[inline(always)]
    fn nfc_middle_step_assign(base2k: usize, lsh: usize, res: &mut [i64], carry: &mut [i128]) {
        nfc_middle_step_assign_neon(base2k, lsh, res, carry);
    }
    #[inline(always)]
    fn nfc_final_step_assign(base2k: usize, lsh: usize, res: &mut [i64], carry: &mut [i128]) {
        nfc_final_step_assign_neon(base2k, lsh, res, carry);
    }
    #[inline(always)]
    fn nfc_final_step_into<O: AssignOp>(base2k: usize, lsh: usize, res: &mut [i64], carry: &mut [i128]) {
        nfc_final_step_into_neon::<O>(base2k, lsh, res, carry);
    }
}

#[cfg(not(target_arch = "aarch64"))]
impl I128NormalizeOps for NTT120Neon {}
