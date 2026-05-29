use anyhow::Result;
use poulpy_core::layouts::{GGLWEInfos, GLWEToBackendMut, GLWEToBackendRef, prepared::GLWETensorKeyPreparedToBackendRef};
use poulpy_hal::layouts::{Backend, Module, ScratchArena};

use crate::{CKKSCtBounds, SetCKKSInfos, api::CKKSEvalModOps, default::eval_mod::EvalModParameters, oep::CKKSEvalModImpl};

impl<BE: Backend + CKKSEvalModImpl<BE>> CKKSEvalModOps<BE> for Module<BE> {
    fn ckks_eval_mod_tmp_bytes<R, P, T>(&self, _res: &R, _params: &EvalModParameters<P>, _tsk: &T) -> usize
    where
        R: CKKSCtBounds,
        P: CKKSCtBounds,
        T: GGLWEInfos,
    {
        0
    }

    fn ckks_eval_mod<R, C, P, T>(
        &self,
        res: &mut R,
        ct: &C,
        params: &EvalModParameters<P>,
        tsk: &T,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        R: GLWEToBackendMut<BE> + GLWEToBackendRef<BE> + CKKSCtBounds + SetCKKSInfos,
        C: GLWEToBackendRef<BE> + CKKSCtBounds,
        P: GLWEToBackendRef<BE> + CKKSCtBounds,
        T: GGLWEInfos + GLWETensorKeyPreparedToBackendRef<BE>,
    {
        BE::ckks_eval_mod::<R, C, P, T>(self, res, ct, params, tsk, scratch)
    }
}
