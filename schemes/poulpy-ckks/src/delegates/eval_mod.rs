use crate::CKKSResult as Result;
use crate::layouts::CKKSMaskedConstants;
use poulpy_core::layouts::GetTensorKey;
use poulpy_core::layouts::IntPolyInfos;
use poulpy_core::layouts::{BSGSMeta, GGLWEInfos, GLWEToBackendMut, GLWEToBackendRef, SetBSGSMeta};
use poulpy_hal::layouts::{Backend, Module, Ring, ScratchArena};

use crate::{CKKSCtBounds, SetCKKSInfos, api::CKKSEvalModOps, layouts::eval_mod::EvalMod, oep::CKKSEvalModImpl};

impl<BE: Backend + CKKSEvalModImpl<Rg>, Rg: Ring> CKKSEvalModOps<BE, Rg> for Module<BE> {
    fn ckks_eval_mod_tmp_bytes<R, C, P, F, T>(&self, res: &R, ct: &C, params: &EvalMod<F, P>, tsk: &T) -> usize
    where
        R: CKKSCtBounds,
        C: CKKSCtBounds,
        P: CKKSCtBounds,
        T: GGLWEInfos,
    {
        BE::ckks_eval_mod_tmp_bytes(self, res, ct, params, tsk)
    }

    fn ckks_eval_mod<R, C, P, F, H>(
        &self,
        res: &mut R,
        ct: &mut C,
        params: &EvalMod<F, P>,
        tsk: &H,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        R: GLWEToBackendMut<BE, Rg> + GLWEToBackendRef<BE, Rg> + CKKSCtBounds + SetCKKSInfos + SetBSGSMeta,
        C: GLWEToBackendMut<BE, Rg> + GLWEToBackendRef<BE, Rg> + CKKSCtBounds,
        P: GLWEToBackendRef<BE, Rg> + IntPolyInfos + CKKSCtBounds + BSGSMeta + CKKSMaskedConstants<BE, Rg>,
        H: GetTensorKey<BE, Rg>,
    {
        BE::ckks_eval_mod::<R, C, P, F, H>(self, res, ct, params, tsk, scratch)
    }
}
