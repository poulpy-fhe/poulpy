use anyhow::Result;
use poulpy_core::layouts::{
    GGLWEInfos, GLWEToBackendMut, GLWEToBackendRef, LWEInfos, prepared::GLWETensorKeyPreparedToBackendRef,
};
use poulpy_core::{GLWENormalize, GLWEZero, ScratchArenaTakeCore};
use poulpy_hal::api::ScratchAvailable;
use poulpy_hal::layouts::{Backend, Module, ScratchArena};

use crate::{
    CKKSCtBounds, SetCKKSInfos,
    api::{CKKSAddOps, CKKSAffineOps, CKKSCopyOps, CKKSMulAddOps, CKKSMulOps, CKKSSubOps},
    default::mod1::{CKKSMod1OpsDefault, Mod1Parameters},
    default::polynomial_evaluation::PolynomialEvaluationDefault,
    layouts::{CKKSCiphertext, CKKSModuleAlloc, CKKSPlaintext},
};

/// # Safety
///
/// Implementations must satisfy the contracts of all trait methods, including
/// any HAL-level invariants (alignment, layout, scratch sizing) implied by the
/// associated method signatures.
pub unsafe trait CKKSMod1Impl<BE: Backend>: Backend {
    fn ckks_eval_mod1<R, C, P, T>(
        module: &Module<BE>,
        res: &mut R,
        ct: &C,
        params: &Mod1Parameters<P>,
        tsk: &T,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        R: GLWEToBackendMut<BE> + CKKSCtBounds + SetCKKSInfos,
        C: GLWEToBackendRef<BE> + CKKSCtBounds,
        P: GLWEToBackendRef<BE> + CKKSCtBounds,
        T: GGLWEInfos + GLWETensorKeyPreparedToBackendRef<BE>;
}

unsafe impl<BE: Backend> CKKSMod1Impl<BE> for BE
where
    Module<BE>: PolynomialEvaluationDefault<BE>
        + CKKSAddOps<BE>
        + CKKSSubOps<BE>
        + CKKSMulOps<BE>
        + CKKSMulAddOps<BE>
        + CKKSCopyOps<BE>
        + CKKSAffineOps<BE>
        + CKKSModuleAlloc<BE>
        + GLWENormalize<BE>
        + GLWEZero<BE>
        + CKKSMod1OpsDefault<BE>,
    CKKSCiphertext<BE::OwnedBuf>: GLWEToBackendMut<BE> + GLWEToBackendRef<BE> + CKKSCtBounds + SetCKKSInfos,
    CKKSPlaintext<BE::OwnedBuf>: GLWEToBackendRef<BE> + LWEInfos,
    for<'a> ScratchArena<'a, BE>: ScratchAvailable + ScratchArenaTakeCore<'a, BE>,
{
    fn ckks_eval_mod1<R, C, P, T>(
        module: &Module<BE>,
        res: &mut R,
        ct: &C,
        params: &Mod1Parameters<P>,
        tsk: &T,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        R: GLWEToBackendMut<BE> + CKKSCtBounds + SetCKKSInfos,
        C: GLWEToBackendRef<BE> + CKKSCtBounds,
        P: GLWEToBackendRef<BE> + CKKSCtBounds,
        T: GGLWEInfos + GLWETensorKeyPreparedToBackendRef<BE>,
    {
        module.ckks_eval_mod1_default(res, ct, params, tsk, scratch)
    }
}
