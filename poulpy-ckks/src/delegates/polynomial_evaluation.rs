use anyhow::Result;
use poulpy_core::layouts::{
    BSGSMeta, GGLWEInfos, GLWEInfos, GLWEToBackendMut, GLWEToBackendRef, ModuleCoreAlloc, SetBSGSMeta,
    prepared::{GLWETensorKeyPrepared, GLWETensorKeyPreparedToBackendRef},
};
use poulpy_hal::layouts::{Backend, HostBytesBackend, Module, ScratchArena, TransferFrom};

use crate::{
    CKKSCtBounds, SetCKKSInfos,
    api::{BSGSPolynomialInfos, PolynomialEvaluation, PowerBasisHelper},
    layouts::CKKSCiphertext,
    oep::CKKSPolynomialEvaluationImpl,
    polynomial::ComplexBSGSPolynomial,
};

impl<BE: Backend + CKKSPolynomialEvaluationImpl<BE>> PolynomialEvaluation<BE> for Module<BE>
where
    Module<BE>: ModuleCoreAlloc<OwnedBuf = BE::OwnedBuf>,
{
    fn ckks_eval_poly_real_const_coeffs_from_power_basis<R, B, A, G, T>(
        &self,
        res: &mut R,
        poly: &B,
        power_basis: &G,
        tsk: &T,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        R: GLWEToBackendMut<BE> + CKKSCtBounds + SetCKKSInfos + SetBSGSMeta,
        B: BSGSPolynomialInfos<BE>,
        B::Coeffs: CKKSCtBounds,
        A: GLWEToBackendRef<BE> + CKKSCtBounds + BSGSMeta,
        G: PowerBasisHelper<BE, A>,
        T: GGLWEInfos + GLWETensorKeyPreparedToBackendRef<BE>,
    {
        BE::ckks_eval_poly_real_const_coeffs_from_power_basis::<R, B, A, G, T>(self, res, poly, power_basis, tsk, scratch)
    }

    fn ckks_eval_poly_complex_const_coeffs_from_power_basis<R, C, A, G, T>(
        &self,
        res: &mut R,
        poly: &ComplexBSGSPolynomial<C>,
        power_basis: &G,
        tsk: &T,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        R: GLWEToBackendMut<BE> + CKKSCtBounds + SetCKKSInfos + SetBSGSMeta,
        C: GLWEToBackendRef<BE> + GLWEInfos + BSGSMeta + CKKSCtBounds,
        A: GLWEToBackendRef<BE> + CKKSCtBounds + BSGSMeta,
        G: PowerBasisHelper<BE, A>,
        T: GGLWEInfos + GLWETensorKeyPreparedToBackendRef<BE>,
    {
        BE::ckks_eval_poly_complex_const_coeffs_from_power_basis::<R, C, A, G, T>(self, res, poly, power_basis, tsk, scratch)
    }

    fn ckks_eval_poly_real_const_coeffs<R, S, B>(
        &self,
        dst: &mut R,
        src: &S,
        bsgs: &B,
        tsk: &GLWETensorKeyPrepared<BE::OwnedBuf, BE>,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        BE: TransferFrom<HostBytesBackend>,
        R: GLWEToBackendMut<BE> + CKKSCtBounds + SetCKKSInfos + SetBSGSMeta,
        S: GLWEToBackendRef<BE> + CKKSCtBounds,
        B: BSGSPolynomialInfos<BE>,
        B::Coeffs: CKKSCtBounds,
        GLWETensorKeyPrepared<BE::OwnedBuf, BE>: GGLWEInfos + GLWETensorKeyPreparedToBackendRef<BE>,
        CKKSCiphertext<BE::OwnedBuf>: GLWEToBackendMut<BE> + GLWEToBackendRef<BE> + CKKSCtBounds + SetCKKSInfos,
    {
        BE::ckks_eval_poly_real_const_coeffs::<R, S, B>(self, dst, src, bsgs, tsk, scratch)
    }

    fn ckks_eval_poly_complex_const_coeffs<R, S, C>(
        &self,
        dst: &mut R,
        src: &S,
        poly: &ComplexBSGSPolynomial<C>,
        tsk: &GLWETensorKeyPrepared<BE::OwnedBuf, BE>,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        BE: TransferFrom<HostBytesBackend>,
        R: GLWEToBackendMut<BE> + CKKSCtBounds + SetCKKSInfos + SetBSGSMeta,
        S: GLWEToBackendRef<BE> + CKKSCtBounds,
        C: GLWEToBackendRef<BE> + GLWEInfos + BSGSMeta + CKKSCtBounds,
        GLWETensorKeyPrepared<BE::OwnedBuf, BE>: GGLWEInfos + GLWETensorKeyPreparedToBackendRef<BE>,
        CKKSCiphertext<BE::OwnedBuf>: GLWEToBackendMut<BE> + GLWEToBackendRef<BE> + CKKSCtBounds + SetCKKSInfos,
    {
        BE::ckks_eval_poly_complex_const_coeffs::<R, S, C>(self, dst, src, poly, tsk, scratch)
    }
}
