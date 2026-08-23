use crate::CKKSResult as Result;
use poulpy_core::GLWECopy;
use poulpy_core::layouts::IntPolyInfos;
use poulpy_core::layouts::{
    BSGSMeta, GGLWEInfos, GLWETensorKeyPrepared, GLWEToBackendMut, GLWEToBackendRef, SetBSGSMeta,
    prepared::GLWETensorKeyPreparedToBackendRef,
};
use poulpy_hal::api::CnvPVecBytesOf;
use poulpy_hal::layouts::{Backend, Module, ScratchArena};

use crate::{
    CKKSCtBounds, SetCKKSInfos,
    api::{CKKSAddOps, CKKSCopyOps, CKKSMulOps, CKKSPolynomialEvaluationOps, CKKSPow2Ops, CKKSSubOps},
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
    fn ckks_eval_mod_impl<R, C, P, F>(
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
        P: GLWEToBackendRef<BE> + IntPolyInfos + CKKSCtBounds + BSGSMeta;

    /// See [`CKKSEvalModOps::ckks_eval_mod_pair_tmp_bytes`](crate::api::CKKSEvalModOps::ckks_eval_mod_pair_tmp_bytes).
    #[allow(clippy::too_many_arguments)]
    fn ckks_eval_mod_pair_tmp_bytes_impl<R0, R1, C0, C1, P, F, T>(
        module: &Module<BE>,
        res_0: &R0,
        res_1: &R1,
        ct_0: &C0,
        ct_1: &C1,
        params: &EvalMod<F, P>,
        tsk: &T,
    ) -> usize
    where
        R0: CKKSCtBounds,
        R1: CKKSCtBounds,
        C0: CKKSCtBounds,
        C1: CKKSCtBounds,
        P: CKKSCtBounds,
        T: GGLWEInfos;

    /// See [`CKKSEvalModOps::ckks_eval_mod_pair`](crate::api::CKKSEvalModOps::ckks_eval_mod_pair).
    #[allow(clippy::too_many_arguments)]
    fn ckks_eval_mod_pair_impl<R0, R1, C0, C1, P, F>(
        module: &Module<BE>,
        res_0: &mut R0,
        res_1: &mut R1,
        ct_0: &C0,
        ct_1: &C1,
        params: &EvalMod<F, P>,
        tsk: &GLWETensorKeyPrepared<BE::OwnedBuf, BE>,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        R0: GLWEToBackendMut<BE> + GLWEToBackendRef<BE> + CKKSCtBounds + SetCKKSInfos + SetBSGSMeta,
        R1: GLWEToBackendMut<BE> + GLWEToBackendRef<BE> + CKKSCtBounds + SetCKKSInfos + SetBSGSMeta,
        C0: GLWEToBackendRef<BE> + CKKSCtBounds,
        C1: GLWEToBackendRef<BE> + CKKSCtBounds,
        P: GLWEToBackendRef<BE> + IntPolyInfos + CKKSCtBounds + BSGSMeta;
}

unsafe impl<BE: Backend> CKKSEvalModImpl<BE> for BE
where
    Module<BE>: CKKSPolynomialEvaluationOps<BE>
        + CKKSAddOps<BE>
        + CKKSSubOps<BE>
        + CKKSMulOps<BE>
        + CKKSCopyOps<BE>
        + CKKSModuleAlloc<BE>
        + CKKSPow2Ops<BE>
        + GLWECopy<BE>
        + CnvPVecBytesOf
        + CKKSEvalModOpsDefault<BE>,
    CKKSCiphertextOwned<BE>: GLWEToBackendMut<BE> + GLWEToBackendRef<BE> + CKKSCtBounds + SetCKKSInfos,
    GLWETensorKeyPrepared<BE::OwnedBuf, BE>: GGLWEInfos + GLWETensorKeyPreparedToBackendRef<BE>,
{
    fn ckks_eval_mod_impl<R, C, P, F>(
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
        P: GLWEToBackendRef<BE> + IntPolyInfos + CKKSCtBounds + BSGSMeta,
    {
        module.ckks_eval_mod_default(res, ct, params, tsk, scratch)
    }

    fn ckks_eval_mod_pair_tmp_bytes_impl<R0, R1, C0, C1, P, F, T>(
        module: &Module<BE>,
        res_0: &R0,
        res_1: &R1,
        ct_0: &C0,
        ct_1: &C1,
        params: &EvalMod<F, P>,
        tsk: &T,
    ) -> usize
    where
        R0: CKKSCtBounds,
        R1: CKKSCtBounds,
        C0: CKKSCtBounds,
        C1: CKKSCtBounds,
        P: CKKSCtBounds,
        T: GGLWEInfos,
    {
        module.ckks_eval_mod_pair_tmp_bytes_default(res_0, res_1, ct_0, ct_1, params, tsk)
    }

    fn ckks_eval_mod_pair_impl<R0, R1, C0, C1, P, F>(
        module: &Module<BE>,
        res_0: &mut R0,
        res_1: &mut R1,
        ct_0: &C0,
        ct_1: &C1,
        params: &EvalMod<F, P>,
        tsk: &GLWETensorKeyPrepared<BE::OwnedBuf, BE>,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        R0: GLWEToBackendMut<BE> + GLWEToBackendRef<BE> + CKKSCtBounds + SetCKKSInfos + SetBSGSMeta,
        R1: GLWEToBackendMut<BE> + GLWEToBackendRef<BE> + CKKSCtBounds + SetCKKSInfos + SetBSGSMeta,
        C0: GLWEToBackendRef<BE> + CKKSCtBounds,
        C1: GLWEToBackendRef<BE> + CKKSCtBounds,
        P: GLWEToBackendRef<BE> + IntPolyInfos + CKKSCtBounds + BSGSMeta,
    {
        module.ckks_eval_mod_pair_default(res_0, res_1, ct_0, ct_1, params, tsk, scratch)
    }
}

/// Wires the reference EvalMod pipeline (single and paired) into `$be`.
///
/// Emits the marker impl the [`CKKSEvalModImpl`] blanket is keyed on. A backend
/// that wants to own EvalMod omits this and implements [`CKKSEvalModImpl`]
/// directly; the two never overlap because the marker is opt-in.
#[macro_export]
macro_rules! impl_ckks_eval_mod_defaults {
    ($be:ty) => {
        impl $crate::default::eval_mod::CKKSEvalModOpsDefault<$be> for ::poulpy_hal::layouts::Module<$be> {}
    };
}
pub use crate::impl_ckks_eval_mod_defaults;
