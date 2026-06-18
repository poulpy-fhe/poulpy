use anyhow::Result;
use poulpy_hal::layouts::{Backend, Module, ScratchArena};

use poulpy_core::layouts::{
    BSGSMeta, GGLWEInfos, GLWEInfos, GLWEToBackendMut, GLWEToBackendRef, SetBSGSMeta,
    prepared::{GLWETensorKeyPrepared, GLWETensorKeyPreparedToBackendRef},
};

use crate::{
    CKKSCtBounds, SetCKKSInfos,
    api::{CKKSAddOps, CKKSCopyOps, CKKSMulOps, CKKSPow2Ops, CKKSScaleManage, CKKSSubOps},
    checked_log_budget_sub,
    layouts::{CKKSCiphertext, CKKSModuleAlloc},
    polynomial::{AdaptiveBSGS, ComplexBSGSPolynomial},
    power_basis::{PowerBasis, PowerBasisGen},
};

pub use poulpy_core::layouts::{BSGSPolynomialInfos, BabyStep, Basis, Parity, PowerBasisHelper};

/// Speed / modulus-saving trade-off for adaptive evaluation; both keep the
/// low-order terms at full precision.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum AdaptiveEvalMode {
    /// Seeds the high baby powers from the low branch's full-scale ones (a
    /// shift, no relinearization): ~monolithic speed, saves only the giant
    /// levels (~`giant_depth·drop`).
    #[default]
    Default,
    /// Rebuilds the high baby powers natively at reduced scale: every level
    /// saves `drop` bits (~`depth·drop`, roughly twice as much), but ~1.4×
    /// slower as those powers are computed twice.
    DoubleModulusSaving,
}

/// Adaptive Chebyshev evaluation built from
/// [`crate::polynomial::EncodeBSGS::encode_bsgs_adaptive`].
pub trait CKKSAdaptivePolynomialEvaluation<BE: Backend> {
    fn ckks_eval_poly_real_const_coeffs_adaptive<R, S, P>(
        &self,
        res: &mut R,
        src: &S,
        adaptive: &AdaptiveBSGS<P>,
        mode: AdaptiveEvalMode,
        tsk: &GLWETensorKeyPrepared<BE::OwnedBuf, BE>,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        R: GLWEToBackendMut<BE> + GLWEToBackendRef<BE> + CKKSCtBounds + SetCKKSInfos + SetBSGSMeta,
        S: GLWEToBackendRef<BE> + CKKSCtBounds,
        P: GLWEToBackendRef<BE> + GLWEInfos + BSGSMeta + CKKSCtBounds,
        GLWETensorKeyPrepared<BE::OwnedBuf, BE>: GGLWEInfos + GLWETensorKeyPreparedToBackendRef<BE>,
        CKKSCiphertext<BE::OwnedBuf>: GLWEToBackendMut<BE> + GLWEToBackendRef<BE> + CKKSCtBounds + SetCKKSInfos + SetBSGSMeta;
}

impl<BE: Backend> CKKSAdaptivePolynomialEvaluation<BE> for Module<BE>
where
    Module<BE>: PolynomialEvaluation<BE>
        + CKKSScaleManage<BE>
        + CKKSAddOps<BE>
        + CKKSCopyOps<BE>
        + CKKSMulOps<BE>
        + CKKSPow2Ops<BE>
        + CKKSSubOps<BE>
        + CKKSModuleAlloc<BE>,
{
    fn ckks_eval_poly_real_const_coeffs_adaptive<R, S, P>(
        &self,
        res: &mut R,
        src: &S,
        adaptive: &AdaptiveBSGS<P>,
        mode: AdaptiveEvalMode,
        tsk: &GLWETensorKeyPrepared<BE::OwnedBuf, BE>,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        R: GLWEToBackendMut<BE> + GLWEToBackendRef<BE> + CKKSCtBounds + SetCKKSInfos + SetBSGSMeta,
        S: GLWEToBackendRef<BE> + CKKSCtBounds,
        P: GLWEToBackendRef<BE> + GLWEInfos + BSGSMeta + CKKSCtBounds,
        GLWETensorKeyPrepared<BE::OwnedBuf, BE>: GGLWEInfos + GLWETensorKeyPreparedToBackendRef<BE>,
        CKKSCiphertext<BE::OwnedBuf>: GLWEToBackendMut<BE> + GLWEToBackendRef<BE> + CKKSCtBounds + SetCKKSInfos + SetBSGSMeta,
    {
        let drop = adaptive.drop();

        // Full-scale power basis for the low (full-precision) baby block.
        let mut x_low = self.ckks_ciphertext_alloc_from_infos(src);
        self.ckks_copy(&mut x_low, src, scratch)?;
        let mut low_basis = PowerBasis::new(adaptive.low().basis(), x_low);
        low_basis.populate(
            adaptive.low().degree(),
            adaptive.low().log_split(),
            adaptive.low().parity(),
            self,
            tsk,
            scratch,
        )?;

        // Reduced-scale power basis for the high block. In the default mode the
        // baby powers are seeded by scaling down the low branch's full-scale ones
        // and only the giant powers are rebuilt; otherwise every power is rebuilt
        // natively (see `AdaptiveEvalMode`).
        let base = 1usize << adaptive.high().log_split();
        let mut x_hi = self.ckks_ciphertext_alloc_from_infos(src);
        self.ckks_copy(&mut x_hi, src, scratch)?;
        self.ckks_scale_down_assign(&mut x_hi, drop, scratch)?;
        let mut high_basis = PowerBasis::new(adaptive.high().basis(), x_hi);
        if mode == AdaptiveEvalMode::Default {
            for n in 2..base {
                if !low_basis.contains_power(n) {
                    continue;
                }
                let mut power = self.ckks_ciphertext_alloc_from_infos(low_basis.get(n)?);
                self.ckks_copy(&mut power, low_basis.get(n)?, scratch)?;
                self.ckks_scale_down_assign(&mut power, drop, scratch)?;
                high_basis.insert(n, power)?;
            }
        }
        high_basis.populate(
            adaptive.high().degree(),
            adaptive.high().log_split(),
            adaptive.high().parity(),
            self,
            tsk,
            scratch,
        )?;

        self.ckks_eval_poly_real_const_coeffs_from_power_basis::<R, _, CKKSCiphertext<BE::OwnedBuf>, _, _>(
            res,
            adaptive.high(),
            &high_basis,
            tsk,
            scratch,
        )?;
        res.set_log_delta(res.log_delta() + drop);
        res.set_log_budget(checked_log_budget_sub(
            "adaptive polynomial high-branch compensation",
            res.log_budget(),
            drop,
        )?);

        let mut res_low = self.ckks_ciphertext_alloc_from_infos(src);
        self.ckks_eval_poly_real_const_coeffs_from_power_basis::<_, _, CKKSCiphertext<BE::OwnedBuf>, _, _>(
            &mut res_low,
            adaptive.low(),
            &low_basis,
            tsk,
            scratch,
        )?;
        self.ckks_add_assign(res, &res_low, scratch)?;
        Ok(())
    }
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
