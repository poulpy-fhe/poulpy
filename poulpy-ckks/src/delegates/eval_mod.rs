use anyhow::Result;
use poulpy_core::layouts::{BSGSMeta, GGLWEInfos, GLWETensorKeyPrepared, GLWEToBackendMut, GLWEToBackendRef, SetBSGSMeta};
use poulpy_hal::api::CnvPVecBytesOf;
use poulpy_hal::layouts::{Backend, HostBytesBackend, Module, ScratchArena, TransferFrom, VecZnx};

use crate::{
    CKKSCtBounds, SetCKKSInfos,
    api::{CKKSAddOps, CKKSCopyOps, CKKSEvalModOps, CKKSMulOps, CKKSSubOps},
    layouts::eval_mod::EvalMod,
    oep::CKKSEvalModImpl,
};

impl<BE: Backend + CKKSEvalModImpl<BE>> CKKSEvalModOps<BE> for Module<BE>
where
    Module<BE>: CKKSAddOps<BE> + CKKSSubOps<BE> + CKKSMulOps<BE> + CKKSCopyOps<BE> + CnvPVecBytesOf,
{
    fn ckks_eval_mod_tmp_bytes<R, P, F, T>(&self, res: &R, _params: &EvalMod<F, P>, tsk: &T) -> usize
    where
        R: CKKSCtBounds,
        P: CKKSCtBounds,
        T: GGLWEInfos,
    {
        let cols: usize = (res.rank() + 1).into();
        let compact_ct = VecZnx::bytes_of(res.n().into(), cols, res.max_size());
        // The giant step hoists the prepared `X^{gsp}` right operand, kept alive
        // across the baby-step pairs that share it.
        let hoisted_right = self.bytes_of_cnv_pvec_right(cols, res.max_size());
        let bsgs_giant = self.ckks_mul_tmp_bytes(res, tsk).max(self.ckks_add_tmp_bytes()) + 3 * compact_ct + hoisted_right;
        let square_scope = self.ckks_square_tmp_bytes(res, tsk) + compact_ct;
        self.ckks_copy_tmp_bytes()
            .max(self.ckks_add_pt_const_tmp_bytes())
            .max(self.ckks_sub_pt_const_tmp_bytes())
            .max(bsgs_giant)
            .max(square_scope)
    }

    fn ckks_eval_mod<R, C, P, F>(
        &self,
        res: &mut R,
        ct: &C,
        params: &EvalMod<F, P>,
        tsk: &GLWETensorKeyPrepared<BE::OwnedBuf, BE>,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        BE: TransferFrom<HostBytesBackend>,
        R: GLWEToBackendMut<BE> + GLWEToBackendRef<BE> + CKKSCtBounds + SetCKKSInfos + SetBSGSMeta,
        C: GLWEToBackendRef<BE> + CKKSCtBounds,
        P: GLWEToBackendRef<BE> + CKKSCtBounds + BSGSMeta,
    {
        BE::ckks_eval_mod::<R, C, P, F>(self, res, ct, params, tsk, scratch)
    }
}
