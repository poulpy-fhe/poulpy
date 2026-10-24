//! Polynomial approximations: a derived family over the interval map and the selected polynomial evaluation.
use crate::CKKSResult as Result;
use poulpy_core::layouts::{
    BSGSMeta, GGLWEInfos, GLWEToBackendMut, GLWEToBackendRef, GetTensorKey, IntPolyInfos, LWEInfos, SetBSGSMeta,
};
use poulpy_hal::layouts::{Backend, Module, ScratchArena};

use crate::{
    CKKSCtBounds, CKKSInfos, SetCKKSInfos,
    api::CKKSAllOpsTmpBytes,
    layouts::PolynomialApproximation,
    oep::{
        CKKSAddImpl, CKKSAffineImpl, CKKSCopyImpl, CKKSPolynomialEvaluationImpl, CKKSPow2Impl,
        derived::approximation::{ckks_approximation_tmp_bytes_derived, ckks_eval_approximation_derived},
    },
};

/// Backend contract of [`CKKSApproximationOps`](crate::api::CKKSApproximationOps).
///
/// Every method defaults to the interval map through the selected power-of-two,
/// copy, constant addition or affine operations, then the selected real
/// polynomial evaluation; an override reproduces it bit for bit within its own
/// query.
///
/// # Safety
///
/// Implementations uphold `poulpy-hal/docs/backend_safety_contract.md` for every
/// operand and scratch region they touch.
pub unsafe trait CKKSApproximationImpl:
    Backend + CKKSAddImpl + CKKSAffineImpl + CKKSCopyImpl + CKKSPow2Impl + CKKSPolynomialEvaluationImpl
{
    /// Scratch of [`Self::ckks_eval_approximation`]: the shared `ckks_all_ops_tmp_bytes` bound at
    /// both layouts, plus the mapped input and its map when the approximation has one.
    fn ckks_approximation_tmp_bytes<R, I, T, P>(
        module: &Module<Self>,
        res: &R,
        input: &I,
        tsk: &T,
        approximation: &PolynomialApproximation<P>,
    ) -> usize
    where
        Module<Self>: CKKSAllOpsTmpBytes<Self>,
        R: CKKSCtBounds,
        I: CKKSCtBounds,
        T: GGLWEInfos,
        P: CKKSInfos + LWEInfos,
    {
        ckks_approximation_tmp_bytes_derived(module, res, input, tsk, approximation)
    }

    /// Checks the budget, maps `input` into a scratch temporary (`ckks_mul_pow2_into`
    /// or `ckks_copy` and `ckks_div_pow2_assign` then `ckks_add_pt_const_assign`, or
    /// `ckks_affine_pt_const_into`) when the approximation carries a map, then
    /// `ckks_eval_poly_real_const_coeffs`.
    fn ckks_eval_approximation<R, I, P, H>(
        module: &Module<Self>,
        res: &mut R,
        input: &I,
        approximation: &PolynomialApproximation<P>,
        tsk: &H,
        scratch: &mut ScratchArena<'_, Self>,
    ) -> Result<()>
    where
        R: GLWEToBackendMut<Self> + CKKSCtBounds + SetCKKSInfos + SetBSGSMeta,
        I: GLWEToBackendRef<Self> + CKKSCtBounds,
        P: GLWEToBackendRef<Self> + CKKSCtBounds + BSGSMeta + IntPolyInfos,
        H: GetTensorKey<Self>,
    {
        ckks_eval_approximation_derived(module, res, input, approximation, tsk, scratch)
    }
}

/// Selects the derived defaults of `CKKSApproximationImpl` for `$be`.
#[macro_export]
macro_rules! impl_ckks_approximation_derived_full {
    ($be:ty) => {
        unsafe impl $crate::oep::CKKSApproximationImpl for $be {}
    };
}

pub use crate::impl_ckks_approximation_derived_full;
