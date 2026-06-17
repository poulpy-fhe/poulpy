use anyhow::Result;
use poulpy_hal::layouts::{Backend, Module, ScratchArena};

use poulpy_core::layouts::{
    BSGSMeta, GGLWEInfos, GLWEInfos, GLWEToBackendMut, GLWEToBackendRef, SetBSGSMeta,
    prepared::{GLWETensorKeyPrepared, GLWETensorKeyPreparedToBackendRef},
};

use crate::{
    CKKSCtBounds, CKKSInfos, SetCKKSInfos,
    api::{CKKSAddOps, CKKSCopyOps, CKKSScaleManage},
    checked_log_budget_sub,
    layouts::{CKKSCiphertext, CKKSModuleAlloc},
    polynomial::{AdaptiveBSGS, ComplexBSGSPolynomial},
};

pub use poulpy_core::layouts::{BSGSPolynomialInfos, BabyStep, Basis, Parity, PowerBasisHelper};

/// Adaptive Chebyshev evaluation built from
/// [`crate::polynomial::EncodeBSGS::encode_bsgs_adaptive`].
#[allow(private_bounds)] // CKKSScaleManage is crate-private; this is its sanctioned consumer.
pub fn ckks_eval_poly_real_const_coeffs_adaptive<BE, R, S, P>(
    module: &Module<BE>,
    res: &mut R,
    src: &S,
    adaptive: &AdaptiveBSGS<P>,
    tsk: &GLWETensorKeyPrepared<BE::OwnedBuf, BE>,
    scratch: &mut ScratchArena<'_, BE>,
) -> Result<()>
where
    BE: Backend,
    Module<BE>: PolynomialEvaluation<BE> + CKKSScaleManage<BE> + CKKSAddOps<BE> + CKKSCopyOps<BE> + CKKSModuleAlloc<BE>,
    R: GLWEToBackendMut<BE> + GLWEToBackendRef<BE> + CKKSCtBounds + SetCKKSInfos + SetBSGSMeta,
    S: GLWEToBackendRef<BE> + CKKSCtBounds,
    P: GLWEToBackendRef<BE> + GLWEInfos + BSGSMeta + CKKSCtBounds,
    GLWETensorKeyPrepared<BE::OwnedBuf, BE>: GGLWEInfos + GLWETensorKeyPreparedToBackendRef<BE>,
    CKKSCiphertext<BE::OwnedBuf>: GLWEToBackendMut<BE> + GLWEToBackendRef<BE> + CKKSCtBounds + SetCKKSInfos + SetBSGSMeta,
{
    module.ckks_eval_poly_real_const_coeffs::<R, S, CKKSCiphertext<BE::OwnedBuf>, _>(res, src, &adaptive.low, tsk, scratch)?;

    if adaptive.drop > 0 {
        let mut x_hi = module.ckks_ciphertext_alloc_from_infos(src);
        module.ckks_copy(&mut x_hi, src, scratch)?;
        module.ckks_scale_down_assign(&mut x_hi, adaptive.drop, scratch)?;

        let mut res_hi = module.ckks_ciphertext_alloc_from_infos(&x_hi);
        module.ckks_eval_poly_real_const_coeffs::<_, _, CKKSCiphertext<BE::OwnedBuf>, _>(
            &mut res_hi,
            &x_hi,
            &adaptive.high,
            tsk,
            scratch,
        )?;
        res_hi.set_log_delta(res_hi.log_delta() + adaptive.drop);
        res_hi.set_log_budget(checked_log_budget_sub(
            "adaptive polynomial high-branch compensation",
            res_hi.log_budget(),
            adaptive.drop,
        )?);
        module.ckks_add_assign(res, &res_hi, scratch)?;
    }
    Ok(())
}

pub trait PolynomialEvaluation<BE: Backend> {
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
        T: GGLWEInfos + GLWETensorKeyPreparedToBackendRef<BE>;

    /// Evaluates a complex-coefficient polynomial `Σ_k (a_k + i·b_k)·z^k`,
    /// where `poly.re`/`poly.im` are the matched real/imag BSGS decompositions
    /// (identical baby-step schedule).
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
        T: GGLWEInfos + GLWETensorKeyPreparedToBackendRef<BE>;

    /// Builds the power basis internally then evaluates a real-coefficient
    /// polynomial.
    fn ckks_eval_poly_real_const_coeffs<R, S, C, B>(
        &self,
        dst: &mut R,
        src: &S,
        bsgs: &B,
        tsk: &GLWETensorKeyPrepared<BE::OwnedBuf, BE>,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        R: GLWEToBackendMut<BE> + CKKSCtBounds + SetCKKSInfos + SetBSGSMeta,
        S: GLWEToBackendRef<BE> + CKKSCtBounds,
        C: GLWEToBackendRef<BE> + GLWEInfos + BSGSMeta + CKKSCtBounds,
        B: BSGSPolynomialInfos<BE>,
        B::Coeffs: CKKSCtBounds,
        GLWETensorKeyPrepared<BE::OwnedBuf, BE>: GGLWEInfos + GLWETensorKeyPreparedToBackendRef<BE>,
        CKKSCiphertext<BE::OwnedBuf>: GLWEToBackendMut<BE> + GLWEToBackendRef<BE> + CKKSCtBounds + SetCKKSInfos;

    /// Builds the power basis internally then evaluates a complex-coefficient
    /// polynomial.
    fn ckks_eval_poly_complex_const_coeffs<R, S, C>(
        &self,
        dst: &mut R,
        src: &S,
        poly: &ComplexBSGSPolynomial<C>,
        tsk: &GLWETensorKeyPrepared<BE::OwnedBuf, BE>,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        R: GLWEToBackendMut<BE> + CKKSCtBounds + SetCKKSInfos + SetBSGSMeta,
        S: GLWEToBackendRef<BE> + CKKSCtBounds,
        C: GLWEToBackendRef<BE> + GLWEInfos + BSGSMeta + CKKSCtBounds,
        GLWETensorKeyPrepared<BE::OwnedBuf, BE>: GGLWEInfos + GLWETensorKeyPreparedToBackendRef<BE>,
        CKKSCiphertext<BE::OwnedBuf>: GLWEToBackendMut<BE> + GLWEToBackendRef<BE> + CKKSCtBounds + SetCKKSInfos;
}
