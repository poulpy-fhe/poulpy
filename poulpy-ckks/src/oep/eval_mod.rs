use crate::CKKSResult as Result;
use poulpy_core::layouts::GetTensorKey;
use poulpy_core::layouts::IntPolyInfos;
use poulpy_core::layouts::{BSGSMeta, GLWEToBackendMut, GLWEToBackendRef, SetBSGSMeta};
use poulpy_hal::layouts::{Backend, Module, ScratchArena};

use crate::{
    CKKSCtBounds, SetCKKSInfos,
    api::{CKKSAddOps, CKKSCopyOps, CKKSMulOps, CKKSPolynomialEvaluationOps, CKKSSubOps},
    layouts::{CKKSCiphertextOwned, CKKSModuleAlloc, eval_mod::EvalMod},
    reference::eval_mod::CKKSEvalModOpsReference,
};

/// Backend override hook for [`CKKSEvalModOps`](crate::api::CKKSEvalModOps).
///
/// The blanket impl below forwards to the backend-generic reference
/// [`ckks_eval_mod_reference`](crate::reference::eval_mod::CKKSEvalModOpsReference::ckks_eval_mod_reference);
/// a backend may instead provide a specialized `ckks_eval_mod` (e.g. a fused or
/// accelerated pipeline) by implementing this trait directly. The public
/// [`CKKSEvalModOps`](crate::api::CKKSEvalModOps) impl dispatches through it.
///
/// # Safety
///
/// Implementations must satisfy the contracts of all trait methods, including
/// any HAL-level invariants (alignment, layout, scratch sizing) implied by the
/// associated method signatures.
pub unsafe trait CKKSEvalModImpl: Backend {
    /// See [`CKKSEvalModOps::ckks_eval_mod`](crate::api::CKKSEvalModOps::ckks_eval_mod).
    fn ckks_eval_mod_impl<R, C, P, F, H>(
        module: &Module<Self>,
        res: &mut R,
        ct: &C,
        params: &EvalMod<F, P>,
        tsk: &H,
        scratch: &mut ScratchArena<'_, Self>,
    ) -> Result<()>
    where
        R: GLWEToBackendMut<Self> + GLWEToBackendRef<Self> + CKKSCtBounds + SetCKKSInfos + SetBSGSMeta,
        C: GLWEToBackendRef<Self> + CKKSCtBounds,
        P: GLWEToBackendRef<Self> + IntPolyInfos + CKKSCtBounds + BSGSMeta,
        H: GetTensorKey<Self>;
}

unsafe impl<BE: Backend> CKKSEvalModImpl for BE
where
    Module<BE>: CKKSPolynomialEvaluationOps<BE>
        + CKKSAddOps<BE>
        + CKKSSubOps<BE>
        + CKKSMulOps<BE>
        + CKKSCopyOps<BE>
        + CKKSModuleAlloc<BE>
        + CKKSEvalModOpsReference<BE>,
    CKKSCiphertextOwned<BE>: GLWEToBackendMut<BE> + GLWEToBackendRef<BE> + CKKSCtBounds + SetCKKSInfos,
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
        R: GLWEToBackendMut<BE> + GLWEToBackendRef<BE> + CKKSCtBounds + SetCKKSInfos + SetBSGSMeta,
        C: GLWEToBackendRef<BE> + CKKSCtBounds,
        P: GLWEToBackendRef<BE> + IntPolyInfos + CKKSCtBounds + BSGSMeta,
        H: GetTensorKey<BE>,
    {
        module.ckks_eval_mod_reference(res, ct, params, tsk, scratch)
    }
}
