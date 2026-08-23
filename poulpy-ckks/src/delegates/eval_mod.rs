use crate::CKKSResult as Result;
use poulpy_core::layouts::IntPolyInfos;
use poulpy_core::layouts::{BSGSMeta, GGLWEInfos, GLWETensorKeyPrepared, GLWEToBackendMut, GLWEToBackendRef, SetBSGSMeta};
use poulpy_hal::api::CnvPVecBytesOf;
use poulpy_hal::layouts::{Backend, Module, ScratchArena};

use crate::{
    CKKSCtBounds, SetCKKSInfos,
    api::{CKKSAddOps, CKKSCopyOps, CKKSEvalModOps, CKKSMulOps, CKKSSubOps},
    default::eval_mod::ckks_eval_mod_tmp_bytes_default,
    layouts::eval_mod::EvalMod,
    oep::CKKSEvalModImpl,
};

impl<BE: Backend + CKKSEvalModImpl<BE>> CKKSEvalModOps<BE> for Module<BE>
where
    Module<BE>: CKKSAddOps<BE> + CKKSSubOps<BE> + CKKSMulOps<BE> + CKKSCopyOps<BE> + CnvPVecBytesOf,
{
    fn ckks_eval_mod_tmp_bytes<R, C, P, F, T>(&self, res: &R, ct: &C, params: &EvalMod<F, P>, tsk: &T) -> usize
    where
        R: CKKSCtBounds,
        C: CKKSCtBounds,
        P: CKKSCtBounds,
        T: GGLWEInfos,
    {
        ckks_eval_mod_tmp_bytes_default(self, res, ct, params, tsk)
    }

    fn ckks_eval_mod_pair_tmp_bytes<R0, R1, C0, C1, P, F, T>(
        &self,
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
        BE::ckks_eval_mod_pair_tmp_bytes_impl(self, res_0, res_1, ct_0, ct_1, params, tsk)
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
        R: GLWEToBackendMut<BE> + GLWEToBackendRef<BE> + CKKSCtBounds + SetCKKSInfos + SetBSGSMeta,
        C: GLWEToBackendRef<BE> + CKKSCtBounds,
        P: GLWEToBackendRef<BE> + IntPolyInfos + CKKSCtBounds + BSGSMeta,
    {
        BE::ckks_eval_mod_impl::<R, C, P, F>(self, res, ct, params, tsk, scratch)
    }

    fn ckks_eval_mod_pair<R0, R1, C0, C1, P, F>(
        &self,
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
        BE::ckks_eval_mod_pair_impl::<R0, R1, C0, C1, P, F>(self, res_0, res_1, ct_0, ct_1, params, tsk, scratch)
    }
}
