use crate::CKKSResult as Result;
use poulpy_core::layouts::GetTensorKey;
use poulpy_core::layouts::IntPolyInfos;
use poulpy_core::layouts::{BSGSMeta, GLWEToBackendMut, GLWEToBackendRef, SetBSGSMeta};
use poulpy_hal::layouts::Normalized;
use poulpy_hal::layouts::{Backend, Module, ScratchArena};

use crate::{
    CKKSCtBounds, SetCKKSInfos,
    api::{CKKSAddOps, CKKSCopyOps, CKKSMulOps, CKKSPolynomialEvaluationOps, CKKSSubOps},
    default::eval_mod::CKKSEvalModOpsDefault,
    layouts::{CKKSCiphertextOwned, CKKSModuleAlloc, eval_mod::EvalMod},
};

/// Backend override hook for [`CKKSEvalModOps`](crate::api::CKKSEvalModOps).
///
/// The blanket impl below forwards to the backend-generic reference
/// [`ckks_eval_mod_default`](crate::default::eval_mod::CKKSEvalModOpsDefault::ckks_eval_mod_default);
/// a backend may instead provide a specialized `ckks_eval_mod` (e.g. a fused or
/// accelerated pipeline) by implementing this trait directly. The public
/// [`CKKSEvalModOps`](crate::api::CKKSEvalModOps) impl dispatches through it.
///
/// # Safety
///
/// Implementations must satisfy the contracts of all trait methods, including
/// any HAL-level invariants (alignment, layout, scratch sizing) implied by the
/// associated method signatures.
pub unsafe trait CKKSEvalModImpl<BE: Backend>: Backend {
    /// See [`CKKSEvalModOps::ckks_eval_mod`](crate::api::CKKSEvalModOps::ckks_eval_mod).
    fn ckks_eval_mod_impl<R, C, P, F, H>(
        module: &Module<BE>,
        res: &mut R,
        ct: &C,
        params: &EvalMod<F, P>,
        tsk: &H,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        R: GLWEToBackendMut<BE, State = Normalized>
            + GLWEToBackendRef<BE, State = Normalized>
            + CKKSCtBounds
            + SetCKKSInfos
            + SetBSGSMeta,
        C: GLWEToBackendRef<BE, State = Normalized> + CKKSCtBounds,
        P: GLWEToBackendRef<BE, State = Normalized> + IntPolyInfos + CKKSCtBounds + BSGSMeta,
        H: GetTensorKey<BE>;
}

unsafe impl<BE: Backend> CKKSEvalModImpl<BE> for BE
where
    Module<BE>: CKKSPolynomialEvaluationOps<BE>
        + CKKSAddOps<BE>
        + CKKSSubOps<BE>
        + CKKSMulOps<BE>
        + CKKSCopyOps<BE>
        + CKKSModuleAlloc<BE>
        + CKKSEvalModOpsDefault<BE>,
    CKKSCiphertextOwned<BE>:
        GLWEToBackendMut<BE, State = Normalized> + GLWEToBackendRef<BE, State = Normalized> + CKKSCtBounds + SetCKKSInfos,
{
    fn ckks_eval_mod_impl<R, C, P, F, H>(
        module: &Module<BE>,
        res: &mut R,
        ct: &C,
        params: &EvalMod<F, P>,
        tsk: &H,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        R: GLWEToBackendMut<BE, State = Normalized>
            + GLWEToBackendRef<BE, State = Normalized>
            + CKKSCtBounds
            + SetCKKSInfos
            + SetBSGSMeta,
        C: GLWEToBackendRef<BE, State = Normalized> + CKKSCtBounds,
        P: GLWEToBackendRef<BE, State = Normalized> + IntPolyInfos + CKKSCtBounds + BSGSMeta,
        H: GetTensorKey<BE>,
    {
        module.ckks_eval_mod_default(res, ct, params, tsk, scratch)
    }
}
