use anyhow::Result;
use poulpy_core::layouts::{GGLWEInfos, GLWEToBackendMut, GLWEToBackendRef, prepared::GLWETensorKeyPreparedToBackendRef};
use poulpy_hal::layouts::{Backend, Module, ScratchArena};

use crate::{CKKSCtBounds, SetCKKSInfos, api::CKKSMod1Ops, default::mod1::Mod1Parameters, oep::CKKSMod1Impl};

impl<BE: Backend + CKKSMod1Impl<BE>> CKKSMod1Ops<BE> for Module<BE> {
    fn ckks_eval_mod1_tmp_bytes<R, P, T>(&self, _res: &R, _params: &Mod1Parameters<P>, _tsk: &T) -> usize
    where
        R: CKKSCtBounds,
        P: CKKSCtBounds,
        T: GGLWEInfos,
    {
        0
    }

    fn ckks_eval_mod1<R, C, P, T>(
        &self,
        res: &mut R,
        ct: &C,
        params: &Mod1Parameters<P>,
        tsk: &T,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        R: GLWEToBackendMut<BE> + GLWEToBackendRef<BE> + CKKSCtBounds + SetCKKSInfos,
        C: GLWEToBackendRef<BE> + CKKSCtBounds,
        P: GLWEToBackendRef<BE> + CKKSCtBounds,
        T: GGLWEInfos + GLWETensorKeyPreparedToBackendRef<BE>,
    {
        BE::ckks_eval_mod1::<R, C, P, T>(self, res, ct, params, tsk, scratch)
    }
}
