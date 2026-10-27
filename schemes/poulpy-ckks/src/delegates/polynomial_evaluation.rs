use crate::CKKSResult as Result;
use poulpy_core::layouts::GetTensorKey;
use poulpy_core::layouts::IntPolyInfos;
use poulpy_core::layouts::{BSGSMeta, GLWEInfos, GLWEToBackendMut, GLWEToBackendRef, ModuleCoreAlloc, SetBSGSMeta};
use poulpy_hal::layouts::{Backend, Module, ScratchArena};

use crate::{
    CKKSCtBounds, SetCKKSInfos,
    api::{BSGSPolynomialInfos, CKKSComplexPolynomialEvaluationOps, CKKSPolynomialEvaluationOps, PowerBasisHelper},
    layouts::CKKSCiphertextOwned,
    oep::{CKKSComplexPolynomialEvaluationImpl, CKKSPolynomialEvaluationImpl},
    polynomial::ComplexBSGSPolynomial,
};

impl<BE: Backend + CKKSPolynomialEvaluationImpl> CKKSPolynomialEvaluationOps<BE> for Module<BE>
where
    Module<BE>: ModuleCoreAlloc<OwnedBuf = BE::OwnedBuf, ZnxWord = BE::ZnxWord>,
{
    fn ckks_eval_poly_real_const_coeffs_from_power_basis<R, B, A, G, H>(
        &self,
        res: &mut R,
        power_basis: &G,
        poly: &B,
        tsk: &H,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        R: GLWEToBackendMut<BE> + CKKSCtBounds + SetCKKSInfos + SetBSGSMeta,
        B: BSGSPolynomialInfos<BE>,
        B::Coeffs: CKKSCtBounds + crate::layouts::CKKSMaskedConstants<BE>,
        A: GLWEToBackendRef<BE> + CKKSCtBounds + BSGSMeta,
        G: PowerBasisHelper<BE, A>,
        H: GetTensorKey<BE>,
    {
        BE::ckks_eval_poly_real_const_coeffs_from_power_basis::<R, B, A, G, H>(self, res, power_basis, poly, tsk, scratch)
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
        R: GLWEToBackendMut<BE> + CKKSCtBounds + SetCKKSInfos + SetBSGSMeta,
        S: GLWEToBackendRef<BE> + CKKSCtBounds,
        B: BSGSPolynomialInfos<BE>,
        B::Coeffs: CKKSCtBounds + crate::layouts::CKKSMaskedConstants<BE>,
        H: GetTensorKey<BE>,
        CKKSCiphertextOwned<BE>: GLWEToBackendMut<BE> + GLWEToBackendRef<BE> + CKKSCtBounds + SetCKKSInfos,
    {
        BE::ckks_eval_poly_real_const_coeffs::<R, S, B, H>(self, dst, src, bsgs, tsk, scratch)
    }
}

impl<BE: Backend + CKKSComplexPolynomialEvaluationImpl> CKKSComplexPolynomialEvaluationOps<BE> for Module<BE>
where
    Module<BE>: ModuleCoreAlloc<OwnedBuf = BE::OwnedBuf, ZnxWord = BE::ZnxWord>,
{
    fn ckks_eval_poly_complex_const_coeffs_from_power_basis<R, C, A, G, H>(
        &self,
        res: &mut R,
        power_basis: &G,
        poly: &ComplexBSGSPolynomial<C>,
        tsk: &H,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        R: GLWEToBackendMut<BE> + CKKSCtBounds + SetCKKSInfos + SetBSGSMeta,
        C: GLWEToBackendRef<BE> + GLWEInfos + BSGSMeta + CKKSCtBounds + IntPolyInfos + crate::layouts::CKKSMaskedConstants<BE>,
        A: GLWEToBackendRef<BE> + CKKSCtBounds + BSGSMeta,
        G: PowerBasisHelper<BE, A>,
        H: GetTensorKey<BE>,
    {
        BE::ckks_eval_poly_complex_const_coeffs_from_power_basis::<R, C, A, G, H>(self, res, power_basis, poly, tsk, scratch)
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
        R: GLWEToBackendMut<BE> + CKKSCtBounds + SetCKKSInfos + SetBSGSMeta,
        S: GLWEToBackendRef<BE> + CKKSCtBounds,
        C: GLWEToBackendRef<BE> + GLWEInfos + BSGSMeta + CKKSCtBounds + IntPolyInfos + crate::layouts::CKKSMaskedConstants<BE>,
        H: GetTensorKey<BE>,
        CKKSCiphertextOwned<BE>: GLWEToBackendMut<BE> + GLWEToBackendRef<BE> + CKKSCtBounds + SetCKKSInfos,
    {
        BE::ckks_eval_poly_complex_const_coeffs::<R, S, C, H>(self, dst, src, poly, tsk, scratch)
    }
}
