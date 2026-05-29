use anyhow::Result;
use poulpy_hal::layouts::{Backend, ScratchArena};

use poulpy_core::layouts::{GGLWEInfos, GLWEToBackendMut, GLWEToBackendRef, prepared::GLWETensorKeyPreparedToBackendRef};

use crate::{CKKSCtBounds, SetCKKSInfos, default::mod1::Mod1Parameters};

pub trait CKKSMod1Ops<BE: Backend> {
    fn ckks_eval_mod1_tmp_bytes<R, P, T>(&self, res: &R, params: &Mod1Parameters<P>, tsk: &T) -> usize
    where
        R: CKKSCtBounds,
        P: CKKSCtBounds,
        T: GGLWEInfos;

    fn ckks_eval_mod1<R, C, P, T>(
        &self,
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
