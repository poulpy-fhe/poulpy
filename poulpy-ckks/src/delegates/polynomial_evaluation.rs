use crate::CKKSResult as Result;
use poulpy_core::layouts::GetTensorKey;
use poulpy_core::layouts::IntPolyInfos;
use poulpy_core::layouts::{BSGSMeta, GLWEInfos, GLWEToBackendMut, GLWEToBackendRef, ModuleCoreAlloc, SetBSGSMeta};
use poulpy_hal::layouts::{Backend, Module, ScratchArena};

use crate::{
    CKKSCtBounds, SetCKKSInfos,
    api::{BSGSPolynomialInfos, CKKSPolynomialEvaluationOps, PowerBasisHelper},
    layouts::CKKSCiphertextOwned,
    oep::CKKSPolynomialEvaluationImpl,
    polynomial::ComplexBSGSPolynomial,
};

impl<BE: Backend + CKKSPolynomialEvaluationImpl> CKKSPolynomialEvaluationOps<BE> for Module<BE>
where
    Module<BE>: ModuleCoreAlloc<OwnedBuf = BE::OwnedBuf, ZnxWord = BE::ZnxWord>,
{
    fn ckks_eval_poly_real_const_coeffs_from_power_basis<R, B, A, G, H>(
        &self,
        res: &mut R,
        poly: &B,
        power_basis: &G,
        tsk: &crate::layouts::CKKSKey<H>,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        R: GLWEToBackendMut<BE> + CKKSCtBounds + SetCKKSInfos + SetBSGSMeta,
        B: BSGSPolynomialInfos<BE>,
        B::Coeffs: CKKSCtBounds,
        A: GLWEToBackendRef<BE> + CKKSCtBounds + BSGSMeta,
        G: PowerBasisHelper<BE, A>,
        H: GetTensorKey<BE>,
    {
        check_polynomial_ring::<BE, _>(crate::api::CKKSModuleInfos::ckks_ring(self), poly)?;
        for power in 1..=BSGSPolynomialInfos::<BE>::degree(poly).max(1) {
            if power_basis.has_power(power) {
                crate::api::CKKSModuleInfos::ckks_ring(self)
                    .check_ciphertext("ckks_eval_poly_real_const_coeffs_from_power_basis", power_basis.get(power)?)?;
            }
        }
        crate::api::CKKSModuleInfos::ckks_ring(self)
            .check("ckks_eval_poly_real_const_coeffs_from_power_basis", tsk.key_ring())?;
        crate::api::CKKSModuleInfos::ckks_ring(self)
            .check_ciphertext("ckks_eval_poly_real_const_coeffs_from_power_basis", res)?;
        BE::ckks_eval_poly_real_const_coeffs_from_power_basis_impl::<R, B, A, G, H>(self, res, poly, power_basis, tsk, scratch)
    }

    fn ckks_eval_poly_complex_const_coeffs_from_power_basis<R, C, A, G, H>(
        &self,
        res: &mut R,
        poly: &ComplexBSGSPolynomial<C>,
        power_basis: &G,
        tsk: &crate::layouts::CKKSKey<H>,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        R: GLWEToBackendMut<BE> + CKKSCtBounds + SetCKKSInfos + SetBSGSMeta,
        C: GLWEToBackendRef<BE> + GLWEInfos + BSGSMeta + CKKSCtBounds + IntPolyInfos,
        A: GLWEToBackendRef<BE> + CKKSCtBounds + BSGSMeta,
        G: PowerBasisHelper<BE, A>,
        H: GetTensorKey<BE>,
    {
        crate::ckks_ensure!(
            !crate::api::CKKSModuleInfos::ckks_is_conjugate_invariant(self),
            "complex polynomial evaluation requires the standard CKKS ring"
        );
        check_polynomial_ring::<BE, _>(crate::api::CKKSModuleInfos::ckks_ring(self), &poly.re)?;
        check_polynomial_ring::<BE, _>(crate::api::CKKSModuleInfos::ckks_ring(self), &poly.im)?;
        for power in 1..=BSGSPolynomialInfos::<BE>::degree(&poly.re).max(1) {
            if power_basis.has_power(power) {
                crate::api::CKKSModuleInfos::ckks_ring(self).check_ciphertext(
                    "ckks_eval_poly_complex_const_coeffs_from_power_basis",
                    power_basis.get(power)?,
                )?;
            }
        }
        crate::api::CKKSModuleInfos::ckks_ring(self)
            .check("ckks_eval_poly_complex_const_coeffs_from_power_basis", tsk.key_ring())?;
        crate::api::CKKSModuleInfos::ckks_ring(self)
            .check_ciphertext("ckks_eval_poly_complex_const_coeffs_from_power_basis", res)?;
        BE::ckks_eval_poly_complex_const_coeffs_from_power_basis_impl::<R, C, A, G, H>(self, res, poly, power_basis, tsk, scratch)
    }

    fn ckks_eval_poly_real_const_coeffs<R, S, B, H>(
        &self,
        dst: &mut R,
        src: &S,
        bsgs: &B,
        tsk: &crate::layouts::CKKSKey<H>,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        R: GLWEToBackendMut<BE> + CKKSCtBounds + SetCKKSInfos + SetBSGSMeta,
        S: GLWEToBackendRef<BE> + CKKSCtBounds,
        B: BSGSPolynomialInfos<BE>,
        B::Coeffs: CKKSCtBounds,
        H: GetTensorKey<BE>,
        CKKSCiphertextOwned<BE>: GLWEToBackendMut<BE> + GLWEToBackendRef<BE> + CKKSCtBounds + SetCKKSInfos,
    {
        check_polynomial_ring::<BE, _>(crate::api::CKKSModuleInfos::ckks_ring(self), bsgs)?;
        crate::api::CKKSModuleInfos::ckks_ring(self).check("ckks_eval_poly_real_const_coeffs", tsk.key_ring())?;
        crate::api::CKKSModuleInfos::ckks_ring(self).check_ciphertext("ckks_eval_poly_real_const_coeffs", dst)?;
        crate::api::CKKSModuleInfos::ckks_ring(self).check_ciphertext("ckks_eval_poly_real_const_coeffs", src)?;
        BE::ckks_eval_poly_real_const_coeffs_impl::<R, S, B, H>(self, dst, src, bsgs, tsk, scratch)
    }

    fn ckks_eval_poly_complex_const_coeffs<R, S, C, H>(
        &self,
        dst: &mut R,
        src: &S,
        poly: &ComplexBSGSPolynomial<C>,
        tsk: &crate::layouts::CKKSKey<H>,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        R: GLWEToBackendMut<BE> + CKKSCtBounds + SetCKKSInfos + SetBSGSMeta,
        S: GLWEToBackendRef<BE> + CKKSCtBounds,
        C: GLWEToBackendRef<BE> + GLWEInfos + BSGSMeta + CKKSCtBounds + IntPolyInfos,
        H: GetTensorKey<BE>,
        CKKSCiphertextOwned<BE>: GLWEToBackendMut<BE> + GLWEToBackendRef<BE> + CKKSCtBounds + SetCKKSInfos,
    {
        crate::ckks_ensure!(
            !crate::api::CKKSModuleInfos::ckks_is_conjugate_invariant(self),
            "complex polynomial evaluation requires the standard CKKS ring"
        );
        check_polynomial_ring::<BE, _>(crate::api::CKKSModuleInfos::ckks_ring(self), &poly.re)?;
        check_polynomial_ring::<BE, _>(crate::api::CKKSModuleInfos::ckks_ring(self), &poly.im)?;
        crate::api::CKKSModuleInfos::ckks_ring(self).check("ckks_eval_poly_complex_const_coeffs", tsk.key_ring())?;
        crate::api::CKKSModuleInfos::ckks_ring(self).check_ciphertext("ckks_eval_poly_complex_const_coeffs", dst)?;
        crate::api::CKKSModuleInfos::ckks_ring(self).check_ciphertext("ckks_eval_poly_complex_const_coeffs", src)?;
        BE::ckks_eval_poly_complex_const_coeffs_impl::<R, S, C, H>(self, dst, src, poly, tsk, scratch)
    }
}

pub(crate) fn check_polynomial_ring<BE: Backend, B: BSGSPolynomialInfos<BE>>(
    ring: crate::layouts::CKKSRing,
    poly: &B,
) -> Result<()>
where
    B::Coeffs: crate::CKKSInfos,
{
    for i in 0..poly.baby_steps() {
        ring.check_coefficients("polynomial coefficients", poly.baby_step(i))?;
    }
    Ok(())
}
