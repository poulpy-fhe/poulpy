use crate::{CKKSCtBounds, CKKSResult, SetCKKSInfos, layouts::CKKSMaskedConstants, polynomial::EncodedFactoredPolynomial};
use crate::{api::CKKSFactoredPolynomialEvaluationOps, oep::CKKSFactoredPolynomialEvaluationImpl};
use poulpy_core::layouts::{GGLWEInfos, GLWEToBackendMut, GLWEToBackendRef, GetTensorKey, IntPolyInfos};
use poulpy_hal::layouts::{Backend, Module, Ring, ScratchArena};

impl<BE: Backend + CKKSFactoredPolynomialEvaluationImpl<Rg>, Rg: Ring> CKKSFactoredPolynomialEvaluationOps<BE, Rg>
    for Module<BE>
{
    fn ckks_eval_poly_factored_tmp_bytes<R, A, P, T>(
        &self,
        res: &R,
        a: &[&A],
        poly: &EncodedFactoredPolynomial<'_, P, Rg>,
        tsk: &T,
    ) -> usize
    where
        R: CKKSCtBounds,
        A: CKKSCtBounds,
        P: CKKSCtBounds + CKKSMaskedConstants<BE, Rg>,
        T: GGLWEInfos,
    {
        BE::ckks_eval_poly_factored_tmp_bytes(self, res, a, poly, tsk)
    }
    fn ckks_eval_poly_factored<R, A, P, H>(
        &self,
        res: &mut R,
        a: &[&A],
        poly: &EncodedFactoredPolynomial<'_, P, Rg>,
        tsk: &H,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> CKKSResult<()>
    where
        R: GLWEToBackendMut<BE, Rg> + CKKSCtBounds + SetCKKSInfos,
        A: GLWEToBackendRef<BE, Rg> + CKKSCtBounds,
        P: GLWEToBackendRef<BE, Rg> + CKKSCtBounds + IntPolyInfos + CKKSMaskedConstants<BE, Rg>,
        H: GetTensorKey<BE, Rg>,
    {
        BE::ckks_eval_poly_factored(self, res, a, poly, tsk, scratch)
    }
}
