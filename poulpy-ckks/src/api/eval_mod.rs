use anyhow::Result;
use poulpy_hal::layouts::{Backend, ScratchArena};

use poulpy_core::layouts::{GGLWEInfos, GLWEToBackendMut, GLWEToBackendRef, prepared::GLWETensorKeyPreparedToBackendRef};

use crate::{CKKSCtBounds, SetCKKSInfos, default::eval_mod::EvalModParameters};

pub trait CKKSEvalModOps<BE: Backend> {
    fn ckks_eval_mod_tmp_bytes<R, P, T>(&self, res: &R, params: &EvalModParameters<P>, tsk: &T) -> usize
    where
        R: CKKSCtBounds,
        P: CKKSCtBounds,
        T: GGLWEInfos;

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
        T: GGLWEInfos + GLWETensorKeyPreparedToBackendRef<BE>;
}
