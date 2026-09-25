use crate::CKKSResult as Result;
use crate::layouts::EvalModBsgs;
use poulpy_core::layouts::GetTensorKey;
use poulpy_core::layouts::IntPolyInfos;
use poulpy_core::layouts::{BSGSMeta, GGLWEInfos, GLWEToBackendMut, GLWEToBackendRef, SetBSGSMeta};
use poulpy_hal::layouts::{Backend, Module, ScratchArena};

use crate::{CKKSCtBounds, SetCKKSInfos, api::CKKSEvalModOps, layouts::eval_mod::EvalMod, oep::CKKSEvalModImpl};

impl<BE: Backend + CKKSEvalModImpl> CKKSEvalModOps<BE> for Module<BE> {
    fn ckks_eval_mod_tmp_bytes<R, C, P, F, T>(&self, res: &R, ct: &C, params: &EvalMod<F, P>, tsk: &T) -> usize
    where
        R: CKKSCtBounds,
        C: CKKSCtBounds,
        P: CKKSCtBounds,
        T: GGLWEInfos,
    {
        BE::ckks_eval_mod_tmp_bytes_impl(self, res, ct, params, tsk)
    }

    fn ckks_eval_mod<R, C, P, F, H>(
        &self,
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
        crate::ckks_ensure!(
            !matches!(params.f_mod_bsgs, EvalModBsgs::Complex(_))
                || !crate::api::CKKSModuleInfos::ckks_is_conjugate_invariant(self),
            "complex EvalMod requires the standard CKKS ring"
        );
        BE::ckks_eval_mod_impl::<R, C, P, F, H>(self, res, ct, params, tsk, scratch)
    }
}
