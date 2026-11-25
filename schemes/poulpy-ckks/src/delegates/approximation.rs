use poulpy_core::layouts::GetTensorKey;
use poulpy_core::layouts::{BSGSMeta, GGLWEInfos, GLWEToBackendMut, GLWEToBackendRef, IntPolyInfos, LWEInfos, SetBSGSMeta};
use poulpy_hal::layouts::{Backend, Module, Ring, ScratchArena};

use crate::{
    CKKSCtBounds, CKKSInfos, CKKSResult as Result, SetCKKSInfos,
    api::{CKKSAllOpsTmpBytes, CKKSApproximationOps},
    layouts::PolynomialApproximation,
    oep::CKKSApproximationImpl,
};

impl<BE: Backend + CKKSApproximationImpl<Rg>, Rg: Ring> CKKSApproximationOps<BE, Rg> for Module<BE>
where
    Module<BE>: CKKSAllOpsTmpBytes<BE>,
{
    fn ckks_polynomial_approximation_eval_tmp_bytes<R, I, T, P>(
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
        BE::ckks_polynomial_approximation_eval_tmp_bytes(self, res, input, approximation, tsk)
    }

    fn ckks_polynomial_approximation_eval<R, I, P, H>(
        &self,
        res: &mut R,
        input: &I,
        approximation: &PolynomialApproximation<P>,
        tsk: &H,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        R: GLWEToBackendMut<BE, Rg> + CKKSCtBounds + SetCKKSInfos + SetBSGSMeta,
        I: GLWEToBackendRef<BE, Rg> + CKKSCtBounds,
        P: GLWEToBackendRef<BE, Rg> + CKKSCtBounds + BSGSMeta + IntPolyInfos,
        H: GetTensorKey<BE, Rg>,
    {
        BE::ckks_polynomial_approximation_eval(self, res, input, approximation, tsk, scratch)
    }
}
