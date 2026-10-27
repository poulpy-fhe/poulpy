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
/// Implementations reproduce the bodies their methods document bit for bit, report for each
/// `_tmp_bytes` query a bound that covers the body, and refuse as the reference does. The memory,
/// layout, synchrony and aliasing rules of `poulpy-hal/docs/backend_safety_contract.md` apply.
pub unsafe trait CKKSApproximationImpl:
    Backend + CKKSAddImpl + CKKSAffineImpl + CKKSCopyImpl + CKKSPow2Impl + CKKSPolynomialEvaluationImpl
{
    /// Default body: `ckks_approximation_tmp_bytes_derived`, the larger of the selected `ckks_all_ops_tmp_bytes` at
    /// `res` and at `input`, both against the first baby-step coefficient layout and `tsk`. An approximation with an
    /// affine map adds the aligned bytes of an `input`-shaped temporary and, unless it has a power-of-two scale, folds in the
    /// larger of that bound and the selected `ckks_affine_pt_const_tmp_bytes`. Scratch bound of
    /// [`Self::ckks_eval_approximation`]; may report less than the reference, never less than the body takes.
    fn ckks_approximation_tmp_bytes<R, I, T, P>(
        module: &Module<Self>,
        res: &R,
        input: &I,
        approximation: &PolynomialApproximation<P>,
        tsk: &T,
    ) -> usize
    where
        Module<Self>: CKKSAllOpsTmpBytes<Self>,
        R: CKKSCtBounds,
        I: CKKSCtBounds,
        T: GGLWEInfos,
        P: CKKSInfos + LWEInfos,
    {
        ckks_approximation_tmp_bytes_derived(module, res, input, approximation, tsk)
    }

    /// Default body: `ckks_eval_approximation_derived`. A `log_budget` below the bits the approximation consumes at the
    /// input's `log_delta` is `InsufficientHomomorphicCapacity`, then, with an affine map, an arena below the
    /// temporary plus its map step's selected queries a refusal. An approximation with an affine map first fills an
    /// `input`-shaped scratch
    /// temporary: with a positive power-of-two scale `ckks_mul_pow2_into`, with a non-positive one `ckks_copy` (slot
    /// kind joined with real) then `ckks_div_pow2_assign`, either then `ckks_add_pt_const_assign` of the affine
    /// constant at coefficient 0; without a power-of-two scale `ckks_affine_pt_const_into` with offset 0 and scale 1.
    /// Then `ckks_eval_poly_real_const_coeffs` of the temporary, or of `input` itself when there is no map.
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
