use crate::CKKSResult as Result;
use poulpy_core::layouts::GetTensorKey;
use poulpy_core::layouts::IntPolyInfos;
use poulpy_core::layouts::{BSGSMeta, GLWEInfos, GLWEToBackendMut, GLWEToBackendRef, ModuleCoreAlloc, SetBSGSMeta};
use poulpy_hal::layouts::{Backend, Module, Normalized, ScratchArena};

use crate::{
    CKKSCtBounds, SetCKKSInfos,
    api::{BSGSPolynomialInfos, CKKSPolynomialEvaluationOps, PowerBasisHelper},
    layouts::CKKSCiphertextOwned,
    oep::CKKSPolynomialEvaluationImpl,
    polynomial::ComplexBSGSPolynomial,
};

impl<BE: Backend + CKKSPolynomialEvaluationImpl<BE>> CKKSPolynomialEvaluationOps<BE> for Module<BE>
where
    Module<BE>: ModuleCoreAlloc<OwnedBuf = BE::OwnedBuf, ZnxWord = BE::ZnxWord>,
{
    fn ckks_eval_poly_real_const_coeffs_from_power_basis<R, B, A, G, H>(
        &self,
        res: &mut R,
        poly: &B,
        power_basis: &G,
        tsk: &H,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        R: GLWEToBackendMut<BE, State = Normalized> + CKKSCtBounds + SetCKKSInfos + SetBSGSMeta,
        B: BSGSPolynomialInfos<BE>,
        B::Coeffs: CKKSCtBounds,
        A: GLWEToBackendRef<BE, State = Normalized> + CKKSCtBounds + BSGSMeta,
        G: PowerBasisHelper<BE, A>,
        H: GetTensorKey<BE>,
    {
        BE::ckks_eval_poly_real_const_coeffs_from_power_basis_impl::<R, B, A, G, H>(self, res, poly, power_basis, tsk, scratch)
    }

    fn ckks_eval_poly_complex_const_coeffs_from_power_basis<R, C, A, G, H>(
        &self,
        res: &mut R,
        poly: &ComplexBSGSPolynomial<C>,
        power_basis: &G,
        tsk: &H,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        R: GLWEToBackendMut<BE, State = Normalized> + CKKSCtBounds + SetCKKSInfos + SetBSGSMeta,
        C: GLWEToBackendRef<BE, State = Normalized> + GLWEInfos + BSGSMeta + CKKSCtBounds + IntPolyInfos,
        A: GLWEToBackendRef<BE, State = Normalized> + CKKSCtBounds + BSGSMeta,
        G: PowerBasisHelper<BE, A>,
        H: GetTensorKey<BE>,
    {
        BE::ckks_eval_poly_complex_const_coeffs_from_power_basis_impl::<R, C, A, G, H>(self, res, poly, power_basis, tsk, scratch)
    }

    fn ckks_eval_poly_real_const_coeffs<R, S, B, H>(
        &self,
        dst: &mut R,
        src: &S,
        bsgs: &B,
        tsk: &H,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        R: GLWEToBackendMut<BE, State = Normalized> + CKKSCtBounds + SetCKKSInfos + SetBSGSMeta,
        S: GLWEToBackendRef<BE, State = Normalized> + CKKSCtBounds,
        B: BSGSPolynomialInfos<BE>,
        B::Coeffs: CKKSCtBounds,
        H: GetTensorKey<BE>,
        CKKSCiphertextOwned<BE>:
            GLWEToBackendMut<BE, State = Normalized> + GLWEToBackendRef<BE, State = Normalized> + CKKSCtBounds + SetCKKSInfos,
    {
        BE::ckks_eval_poly_real_const_coeffs_impl::<R, S, B, H>(self, dst, src, bsgs, tsk, scratch)
    }

    fn ckks_eval_poly_complex_const_coeffs<R, S, C, H>(
        &self,
        dst: &mut R,
        src: &S,
        poly: &ComplexBSGSPolynomial<C>,
        tsk: &H,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        R: GLWEToBackendMut<BE, State = Normalized> + CKKSCtBounds + SetCKKSInfos + SetBSGSMeta,
        S: GLWEToBackendRef<BE, State = Normalized> + CKKSCtBounds,
        C: GLWEToBackendRef<BE, State = Normalized> + GLWEInfos + BSGSMeta + CKKSCtBounds + IntPolyInfos,
        H: GetTensorKey<BE>,
        CKKSCiphertextOwned<BE>:
            GLWEToBackendMut<BE, State = Normalized> + GLWEToBackendRef<BE, State = Normalized> + CKKSCtBounds + SetCKKSInfos,
    {
        BE::ckks_eval_poly_complex_const_coeffs_impl::<R, S, C, H>(self, dst, src, poly, tsk, scratch)
    }
}
