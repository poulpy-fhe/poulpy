use crate::CKKSResult as Result;
use poulpy_core::layouts::GetTensorKey;
use poulpy_core::layouts::IntPolyInfos;
use poulpy_core::layouts::{BSGSMeta, GGLWEInfos, GLWEToBackendMut, GLWEToBackendRef, SetBSGSMeta};
use poulpy_hal::layouts::{Backend, Module, ScratchArena};

use crate::{CKKSCtBounds, SetCKKSInfos, api::CKKSEvalModOps, layouts::eval_mod::EvalMod, oep::CKKSEvalModImpl};

impl<BE: Backend + CKKSEvalModImpl> CKKSEvalModOps<BE> for Module<BE> {
    fn ckks_eval_mod_pair_tmp_bytes<R1, R2, C1, C2, P, F, T>(
        &self,
        left_out: &R1,
        right_out: &R2,
        left_in: &C1,
        right_in: &C2,
        params: &EvalMod<F, P>,
        tsk: &T,
    ) -> usize
    where
        R1: CKKSCtBounds,
        R2: CKKSCtBounds,
        C1: CKKSCtBounds,
        C2: CKKSCtBounds,
        P: CKKSCtBounds,
        T: poulpy_core::layouts::GGLWEInfos,
    {
        BE::ckks_eval_mod_pair_tmp_bytes_impl(self, left_out, right_out, left_in, right_in, params, tsk)
    }

    #[allow(clippy::too_many_arguments)]
    fn ckks_eval_mod_pair<R1, R2, C1, C2, P, F, H>(
        &self,
        left_out: &mut R1,
        right_out: &mut R2,
        left_in: &C1,
        right_in: &C2,
        params: &EvalMod<F, P>,
        tsk: &H,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        R1: GLWEToBackendMut<BE> + GLWEToBackendRef<BE> + CKKSCtBounds + SetCKKSInfos + SetBSGSMeta + Send,
        R2: GLWEToBackendMut<BE> + GLWEToBackendRef<BE> + CKKSCtBounds + SetCKKSInfos + SetBSGSMeta + Send,
        C1: GLWEToBackendRef<BE> + CKKSCtBounds + Sync,
        C2: GLWEToBackendRef<BE> + CKKSCtBounds + Sync,
        P: GLWEToBackendRef<BE> + IntPolyInfos + CKKSCtBounds + BSGSMeta + Sync,
        F: Sync,
        H: GetTensorKey<BE> + Sync,
    {
        BE::ckks_eval_mod_pair_impl(self, left_out, right_out, left_in, right_in, params, tsk, scratch)
    }

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
        BE::ckks_eval_mod_impl::<R, C, P, F, H>(self, res, ct, params, tsk, scratch)
    }
}
