use anyhow::Result;
use poulpy_core::layouts::{GGLWEInfos, GLWEToBackendMut, GLWEToBackendRef, prepared::GLWETensorKeyPreparedToBackendRef};
use poulpy_hal::layouts::{Backend, Module, ScratchArena, VecZnx};

use crate::{
    CKKSCtBounds, SetCKKSInfos,
    api::{CKKSAddOps, CKKSCopyOps, CKKSEvalModOps, CKKSMulOps, CKKSSubOps},
    default::eval_mod::EvalModParameters,
    oep::CKKSEvalModImpl,
};

impl<BE: Backend + CKKSEvalModImpl<BE>> CKKSEvalModOps<BE> for Module<BE>
where
    Module<BE>: CKKSAddOps<BE> + CKKSSubOps<BE> + CKKSMulOps<BE> + CKKSCopyOps<BE>,
{
    fn ckks_eval_mod_tmp_bytes<R, P, T>(&self, res: &R, _params: &EvalModParameters<P>, tsk: &T) -> usize
    where
        R: CKKSCtBounds,
        P: CKKSCtBounds,
        T: GGLWEInfos,
    {
        let compact_ct = VecZnx::bytes_of(res.n().into(), (res.rank() + 1).into(), res.size());
        let bsgs_giant = self.ckks_mul_tmp_bytes(res, tsk).max(self.ckks_add_tmp_bytes()) + 3 * compact_ct;
        let square_scope = self.ckks_square_tmp_bytes(res, tsk) + compact_ct;
        self.ckks_copy_tmp_bytes()
            .max(self.ckks_add_pt_const_tmp_bytes())
            .max(self.ckks_sub_pt_const_tmp_bytes())
            .max(bsgs_giant)
            .max(square_scope)
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
