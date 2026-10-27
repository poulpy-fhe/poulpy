use poulpy_core::layouts::GetTensorKey;
use poulpy_core::layouts::{BSGSMeta, GGLWEInfos, GLWEToBackendMut, GLWEToBackendRef, IntPolyInfos, LWEInfos, SetBSGSMeta};
use poulpy_hal::layouts::{Backend, Module, ScratchArena};

use crate::{
    CKKSCtBounds, CKKSInfos, CKKSResult as Result, SetCKKSInfos,
    api::{CKKSAllOpsTmpBytes, CKKSApproximationOps},
    layouts::PolynomialApproximation,
    oep::CKKSApproximationImpl,
};

impl<BE: Backend + CKKSApproximationImpl> CKKSApproximationOps<BE> for Module<BE>
where
    Module<BE>: CKKSAllOpsTmpBytes<BE>,
{
    fn ckks_approximation_tmp_bytes<R, I, T, P>(
        &self,
        res: &R,
        input: &I,
        approximation: &PolynomialApproximation<P>,
        tsk: &T,
    ) -> usize
    where
        R: CKKSCtBounds,
        I: CKKSCtBounds,
        T: GGLWEInfos,
        P: CKKSInfos + LWEInfos,
    {
        BE::ckks_approximation_tmp_bytes(self, res, input, approximation, tsk)
    }

    fn ckks_eval_approximation<R, I, P, H>(
        &self,
        res: &mut R,
        input: &I,
        approximation: &PolynomialApproximation<P>,
        tsk: &H,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        R: GLWEToBackendMut<BE> + CKKSCtBounds + SetCKKSInfos + SetBSGSMeta,
        I: GLWEToBackendRef<BE> + CKKSCtBounds,
        P: GLWEToBackendRef<BE> + CKKSCtBounds + BSGSMeta + IntPolyInfos,
        H: GetTensorKey<BE>,
    {
        BE::ckks_eval_approximation(self, res, input, approximation, tsk, scratch)
    }
}
