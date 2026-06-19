use anyhow::Result;
use poulpy_core::layouts::{
    BSGSMeta, GGLWEInfos, GLWETensorKeyPrepared, GLWEToBackendMut, GLWEToBackendRef, SetBSGSMeta,
    prepared::GLWETensorKeyPreparedToBackendRef,
};
use poulpy_hal::layouts::{Backend, Module, ScratchArena};

use crate::{
    CKKSCtBounds, SetCKKSInfos,
    api::{CKKSAddOps, CKKSCopyOps, CKKSMulOps, CKKSSubOps, PolynomialEvaluation},
    default::eval_mod::CKKSEvalModOpsDefault,
    layouts::{CKKSCiphertext, CKKSModuleAlloc, eval_mod::EvalMod},
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
    fn ckks_eval_mod<R, C, P, F>(
        module: &Module<BE>,
        res: &mut R,
        ct: &C,
        params: &EvalMod<F, P>,
        tsk: &GLWETensorKeyPrepared<BE::OwnedBuf, BE>,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        R: GLWEToBackendMut<BE> + GLWEToBackendRef<BE> + CKKSCtBounds + SetCKKSInfos + SetBSGSMeta,
        C: GLWEToBackendRef<BE> + CKKSCtBounds,
        P: GLWEToBackendRef<BE> + CKKSCtBounds + BSGSMeta;
}

unsafe impl<BE: Backend> CKKSEvalModImpl<BE> for BE
where
    Module<BE>: PolynomialEvaluation<BE>
        + CKKSAddOps<BE>
        + CKKSSubOps<BE>
        + CKKSMulOps<BE>
        + CKKSCopyOps<BE>
        + CKKSModuleAlloc<BE>
        + CKKSEvalModOpsDefault<BE>,
    CKKSCiphertext<BE::OwnedBuf>: GLWEToBackendMut<BE> + GLWEToBackendRef<BE> + CKKSCtBounds + SetCKKSInfos,
    GLWETensorKeyPrepared<BE::OwnedBuf, BE>: GGLWEInfos + GLWETensorKeyPreparedToBackendRef<BE>,
{
    fn ckks_eval_mod<R, C, P, F>(
        module: &Module<BE>,
        res: &mut R,
        ct: &C,
        params: &EvalMod<F, P>,
        tsk: &GLWETensorKeyPrepared<BE::OwnedBuf, BE>,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        R: GLWEToBackendMut<BE> + GLWEToBackendRef<BE> + CKKSCtBounds + SetCKKSInfos + SetBSGSMeta,
        C: GLWEToBackendRef<BE> + CKKSCtBounds,
        P: GLWEToBackendRef<BE> + CKKSCtBounds + BSGSMeta,
    {
        module.ckks_eval_mod_default(res, ct, params, tsk, scratch)
    }
}
