//! Inverse trigonometric functions.

use std::fmt::Debug;

use anyhow::{Result, anyhow, ensure};
use num_traits::{Float, FloatConst, FromPrimitive, ToPrimitive};
use poulpy_core::layouts::{
    BSGSMeta, GGLWEInfos, GGLWEPreparedToBackendRef, GLWE, GLWETensorKeyPrepared, GLWEToBackendMut,
    GLWEToBackendRef, GetGaloisElement, LWEInfos, SetBSGSMeta,
    prepared::{GLWEAutomorphismKeyPreparedToBackendRef, GLWETensorKeyPreparedToBackendRef},
};
use poulpy_hal::layouts::{Backend, HostBytesBackend, Module, ScratchArena};

use poulpy_ckks::{
    CKKSCtBounds, CKKSInfos, CKKSLayout, SetCKKSInfos,
    api::{
        CKKSAddOps, CKKSConjugateOps, CKKSCopyOps, CKKSMulOps, CKKSNegOps,
        CKKSPolynomialEvaluationOps,
    },
    layouts::{
        CKKSCiphertext, CKKSModuleAlloc, CKKSPlaintext, CKKSPlaintextVecHostCodec, CKKSScalar,
        ScratchArenaTakeCKKS,
    },
    polynomial::SplitStrategy,
};

use crate::{
    approximation::{
        ApproximationOptions, CKKSApproximationOps, Parity, parity_for_interval, prepare_function,
    },
    plan::{declare_unary_op, define_unary_plan, impl_unary_op},
    roots::CKKSInverseOps,
    sign::{CKKSSignOps, SignComposite, ckks_sign_into},
};

/// Inverse-trigonometric plan construction options.
pub type InverseTrigOptions = ApproximationOptions;

define_unary_plan!(AtanPlan, "atan");
define_unary_plan!(AsinPlan, "asin");
define_unary_plan!(AcosPlan, "acos");

/// `atan2` plan construction options.
#[derive(Clone, Copy, Debug)]
pub struct Atan2Options {
    /// Requested absolute-error bits.
    pub target_bits: f64,
    /// Largest `atan` degree considered.
    pub max_degree: usize,
    /// Goldschmidt reciprocal iterations.
    pub reciprocal_iters: usize,
    /// Excluded sign gap around each axis.
    pub sign_tau: f64,
    /// Degree of each sign-composite factor.
    pub sign_degree: usize,
    /// Largest sign-composite factor count.
    pub max_sign_factors: usize,
    /// Largest absolute input value.
    pub input_bound: f64,
    /// Poulpy BSGS split strategy.
    pub strategy: SplitStrategy,
}

impl Default for Atan2Options {
    fn default() -> Self {
        Self {
            target_bits: 20.0,
            max_degree: 31,
            reciprocal_iters: 6,
            sign_tau: 0.1,
            sign_degree: 15,
            max_sign_factors: 12,
            input_bound: 1.0,
            strategy: SplitStrategy::MinDepth,
        }
    }
}

/// Prepared full-quadrant `atan2`.
pub struct Atan2Plan<F, P> {
    /// Prepared `atan` stage.
    pub atan: AtanPlan<P>,
    /// Prepared branch-sign stage.
    pub sign: SignComposite<F, P>,
    /// Encoded `pi/2` and the optional normalization factor.
    pub constants: P,
    /// Goldschmidt reciprocal iterations.
    pub reciprocal_iters: usize,
    /// Largest supported `|y/x|`.
    pub ratio_bound: f64,
    /// Excluded sign gap around each axis.
    pub sign_tau: f64,
    /// Largest supported `|x|` and `|y|`.
    pub input_bound: f64,
    /// Public power-of-two normalization exponent.
    pub normalization_steps: usize,
    /// Scale of prepared plaintexts.
    pub coeff_log_delta: usize,
}

impl AtanPlan<CKKSPlaintext<Vec<u8>, i64>> {
    /// Fits and prepares `atan` on `[a, b]`.
    pub fn from_precision<F>(
        a: F,
        b: F,
        base2k: poulpy_core::layouts::Base2K,
        coeff_meta: CKKSLayout,
        options: InverseTrigOptions,
        module: &Module<HostBytesBackend>,
    ) -> Result<Self>
    where
        F: CKKSScalar + Float + FloatConst + FromPrimitive + ToPrimitive + Debug,
        CKKSPlaintext<Vec<u8>, i64>: CKKSPlaintextVecHostCodec<F>,
    {
        let (approximation, approximation_bits) = prepare_function(
            "atan",
            |x: F| x.atan(),
            a,
            b,
            parity_for_interval(a, b, Parity::Odd),
            base2k,
            coeff_meta,
            options,
            module,
        )?;
        Ok(Self {
            approximation,
            approximation_bits,
        })
    }
}

impl AsinPlan<CKKSPlaintext<Vec<u8>, i64>> {
    /// Fits and prepares `asin` on `[a, b] ⊆ [-1, 1]`.
    pub fn from_precision<F>(
        a: F,
        b: F,
        base2k: poulpy_core::layouts::Base2K,
        coeff_meta: CKKSLayout,
        options: InverseTrigOptions,
        module: &Module<HostBytesBackend>,
    ) -> Result<Self>
    where
        F: CKKSScalar + Float + FloatConst + FromPrimitive + ToPrimitive + Debug,
        CKKSPlaintext<Vec<u8>, i64>: CKKSPlaintextVecHostCodec<F>,
    {
        ensure!(
            a >= -F::one() && b <= F::one(),
            "asin: interval must lie in [-1, 1]"
        );
        let (approximation, approximation_bits) = prepare_function(
            "asin",
            |x: F| x.asin(),
            a,
            b,
            parity_for_interval(a, b, Parity::Odd),
            base2k,
            coeff_meta,
            options,
            module,
        )?;
        Ok(Self {
            approximation,
            approximation_bits,
        })
    }
}

impl AcosPlan<CKKSPlaintext<Vec<u8>, i64>> {
    /// Fits and prepares `acos` on `[a, b] ⊆ [-1, 1]`.
    pub fn from_precision<F>(
        a: F,
        b: F,
        base2k: poulpy_core::layouts::Base2K,
        coeff_meta: CKKSLayout,
        options: InverseTrigOptions,
        module: &Module<HostBytesBackend>,
    ) -> Result<Self>
    where
        F: CKKSScalar + Float + FloatConst + FromPrimitive + ToPrimitive + Debug,
        CKKSPlaintext<Vec<u8>, i64>: CKKSPlaintextVecHostCodec<F>,
    {
        ensure!(
            a >= -F::one() && b <= F::one(),
            "acos: interval must lie in [-1, 1]"
        );
        let (approximation, approximation_bits) = prepare_function(
            "acos",
            |x: F| x.acos(),
            a,
            b,
            Parity::Full,
            base2k,
            coeff_meta,
            options,
            module,
        )?;
        Ok(Self {
            approximation,
            approximation_bits,
        })
    }
}

impl<F> Atan2Plan<F, CKKSPlaintext<Vec<u8>, i64>>
where
    F: CKKSScalar + Float + FloatConst + FromPrimitive + ToPrimitive + Debug,
    CKKSPlaintext<Vec<u8>, i64>: CKKSPlaintextVecHostCodec<F>,
{
    /// Prepares `atan(y/x)` and quadrant selection.
    pub fn from_precision(
        ratio_bound: F,
        base2k: poulpy_core::layouts::Base2K,
        coeff_meta: CKKSLayout,
        options: Atan2Options,
        module: &Module<HostBytesBackend>,
    ) -> Result<Self> {
        ensure!(
            ratio_bound.is_finite() && ratio_bound > F::zero(),
            "atan2: ratio_bound must be positive and finite"
        );
        ensure!(
            options.target_bits.is_finite() && options.target_bits > 0.0,
            "atan2: target_bits must be positive and finite"
        );
        ensure!(options.max_degree > 0, "atan2: max_degree must be positive");
        ensure!(
            options.reciprocal_iters > 0,
            "atan2: reciprocal_iters must be positive"
        );
        ensure!(
            options.sign_tau.is_finite() && options.sign_tau > 0.0 && options.sign_tau < 1.0,
            "atan2: sign_tau must lie in (0, 1)"
        );
        ensure!(
            options.sign_degree > 0,
            "atan2: sign_degree must be positive"
        );
        ensure!(
            options.max_sign_factors > 0,
            "atan2: max_sign_factors must be positive"
        );
        ensure!(
            options.input_bound.is_finite() && options.input_bound > 0.0,
            "atan2: input_bound must be positive and finite"
        );
        let atan = AtanPlan::from_precision(
            -ratio_bound,
            ratio_bound,
            base2k,
            coeff_meta,
            InverseTrigOptions {
                target_bits: options.target_bits + 2.0,
                max_degree: options.max_degree,
                strategy: options.strategy,
            },
            module,
        )
        .map_err(|e| anyhow!("atan2: {e}"))?;
        let sign = SignComposite::from_minimax(
            F::from_f64(options.sign_tau).unwrap(),
            options.target_bits + 2.0,
            &[options.sign_degree],
            options.max_sign_factors,
            base2k,
            coeff_meta,
            options.strategy,
            module,
        )
        .map_err(|e| anyhow!("atan2: {e}"))?;
        let normalization_steps = options.input_bound.log2().ceil().max(0.0) as usize;
        let sign_gap = options.sign_tau * 2.0f64.powi(normalization_steps as i32);
        ensure!(
            sign_gap <= options.input_bound,
            "atan2: normalized sign gap exceeds input_bound"
        );
        let normalizer = F::from_f64(2.0f64.powi(-(normalization_steps as i32))).unwrap();
        let mut constants = module.ckks_pt_coeffs_alloc(2, base2k, coeff_meta.k());
        constants.set_meta(coeff_meta.meta());
        constants
            .encode_host_floats(&[F::PI() / (F::one() + F::one()), normalizer])
            .map_err(|e| anyhow!("atan2: constant encoding failed: {e}"))?;
        Ok(Self {
            atan,
            sign,
            constants,
            reciprocal_iters: options.reciprocal_iters,
            ratio_bound: ratio_bound.to_f64().unwrap_or(f64::NAN),
            sign_tau: options.sign_tau,
            input_bound: options.input_bound,
            normalization_steps,
            coeff_log_delta: coeff_meta.log_delta(),
        })
    }
}

impl<F, P> Atan2Plan<F, P> {
    /// Smallest supported absolute input value after accounting for normalization.
    pub fn sign_gap(&self) -> f64 {
        self.sign_tau * 2.0f64.powi(self.normalization_steps as i32)
    }

    /// Consumed modulus bits.
    pub fn consumed_bits(&self, input_log_delta: usize) -> usize {
        let normalization = usize::from(self.normalization_steps > 0) * self.coeff_log_delta;
        normalization
            + self.sign.consumed_bits(input_log_delta)
            + (self.reciprocal_iters + 4) * input_log_delta
            + self.atan.consumed_bits(input_log_delta)
    }

    /// Multiplicative depth.
    pub fn depth(&self) -> usize {
        usize::from(self.normalization_steps > 0)
            + self.sign.depth()
            + self.reciprocal_iters
            + 4
            + self.atan.depth()
    }

    /// Maps prepared plaintexts to another backend.
    pub fn map_plaintexts<Q>(self, mut f: impl FnMut(&P) -> Q) -> Atan2Plan<F, Q> {
        Atan2Plan {
            atan: self.atan.map_plaintexts(&mut f),
            sign: self.sign.map_plaintexts(&mut f),
            constants: f(&self.constants),
            reciprocal_iters: self.reciprocal_iters,
            ratio_bound: self.ratio_bound,
            sign_tau: self.sign_tau,
            input_bound: self.input_bound,
            normalization_steps: self.normalization_steps,
            coeff_log_delta: self.coeff_log_delta,
        }
    }
}

/// Homomorphic inverse trigonometric functions on fixed prepared domains.
pub trait CKKSInverseTrigOps<BE: Backend> {
    declare_unary_op!(ckks_atan_tmp_bytes, ckks_atan, AtanPlan, "atan");
    declare_unary_op!(ckks_asin_tmp_bytes, ckks_asin, AsinPlan, "asin");
    declare_unary_op!(ckks_acos_tmp_bytes, ckks_acos, AcosPlan, "acos");
}

impl<BE: Backend> CKKSInverseTrigOps<BE> for Module<BE>
where
    Module<BE>: CKKSApproximationOps<BE>,
{
    impl_unary_op!(ckks_atan_tmp_bytes, ckks_atan, AtanPlan);
    impl_unary_op!(ckks_asin_tmp_bytes, ckks_asin, AsinPlan);
    impl_unary_op!(ckks_acos_tmp_bytes, ckks_acos, AcosPlan);
}

/// Homomorphic full-quadrant `atan2` away from both axes.
pub trait CKKSAtan2Ops<BE: Backend> {
    /// Scratch bytes for [`Self::ckks_atan2`].
    fn ckks_atan2_tmp_bytes<R, T, A, F, P>(
        &self,
        res: &R,
        tsk: &T,
        conj_key: &A,
        plan: &Atan2Plan<F, P>,
    ) -> usize
    where
        R: CKKSCtBounds,
        T: GGLWEInfos,
        A: GGLWEInfos,
        P: CKKSInfos + LWEInfos;

    /// Computes `atan2(y, x)` when `plan.sign_gap() <= |x|, |y| <=
    /// plan.input_bound` and `|y/x| <= plan.ratio_bound`.
    #[allow(clippy::too_many_arguments)]
    fn ckks_atan2<F, P, K>(
        &self,
        res: &mut CKKSCiphertext<BE::OwnedBuf, BE::ZnxWord>,
        y: &CKKSCiphertext<BE::OwnedBuf, BE::ZnxWord>,
        x: &CKKSCiphertext<BE::OwnedBuf, BE::ZnxWord>,
        plan: &Atan2Plan<F, P>,
        tsk: &GLWETensorKeyPrepared<BE::OwnedBuf, BE>,
        conj_key: &K,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        P: GLWEToBackendRef<BE> + CKKSCtBounds + poulpy_core::layouts::IntPolyInfos + BSGSMeta,
        K: GLWEAutomorphismKeyPreparedToBackendRef<BE>
            + GGLWEPreparedToBackendRef<BE>
            + GetGaloisElement
            + GGLWEInfos;
}

impl<BE: Backend> CKKSAtan2Ops<BE> for Module<BE>
where
    Module<BE>: CKKSAddOps<BE>
        + CKKSConjugateOps<BE>
        + CKKSCopyOps<BE>
        + CKKSApproximationOps<BE>
        + CKKSInverseOps<BE>
        + CKKSInverseTrigOps<BE>
        + CKKSMulOps<BE>
        + CKKSNegOps<BE>
        + CKKSSignOps<BE>
        + CKKSModuleAlloc<BE>
        + CKKSPolynomialEvaluationOps<BE>,
    CKKSCiphertext<BE::OwnedBuf, BE::ZnxWord>:
        GLWEToBackendMut<BE> + GLWEToBackendRef<BE> + CKKSCtBounds + SetCKKSInfos + SetBSGSMeta,
    GLWETensorKeyPrepared<BE::OwnedBuf, BE>: GGLWEInfos + GLWETensorKeyPreparedToBackendRef<BE>,
{
    fn ckks_atan2_tmp_bytes<R, T, A, F, P>(
        &self,
        res: &R,
        tsk: &T,
        conj_key: &A,
        plan: &Atan2Plan<F, P>,
    ) -> usize
    where
        R: CKKSCtBounds,
        T: GGLWEInfos,
        A: GGLWEInfos,
        P: CKKSInfos + LWEInfos,
    {
        let ct = GLWE::<Vec<u8>, BE::ZnxWord>::bytes_of_from_infos(res);
        4 * ct
            + self
                .ckks_sign_tmp_bytes(res, tsk, conj_key, &plan.constants)
                .max(self.ckks_inverse_tmp_bytes(res, tsk))
                .max(self.ckks_atan_tmp_bytes(res, tsk, &plan.atan))
                .max(self.ckks_mul_tmp_bytes(res, res, res, tsk))
                .max(self.ckks_mul_pt_const_tmp_bytes(res, res, &plan.constants))
                .max(self.ckks_add_tmp_bytes())
                .max(self.ckks_copy_tmp_bytes())
                .max(self.ckks_neg_tmp_bytes())
    }

    #[allow(clippy::too_many_arguments)]
    fn ckks_atan2<F, P, K>(
        &self,
        res: &mut CKKSCiphertext<BE::OwnedBuf, BE::ZnxWord>,
        y: &CKKSCiphertext<BE::OwnedBuf, BE::ZnxWord>,
        x: &CKKSCiphertext<BE::OwnedBuf, BE::ZnxWord>,
        plan: &Atan2Plan<F, P>,
        tsk: &GLWETensorKeyPrepared<BE::OwnedBuf, BE>,
        conj_key: &K,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        P: GLWEToBackendRef<BE> + CKKSCtBounds + poulpy_core::layouts::IntPolyInfos + BSGSMeta,
        K: GLWEAutomorphismKeyPreparedToBackendRef<BE>
            + GGLWEPreparedToBackendRef<BE>
            + GetGaloisElement
            + GGLWEInfos,
    {
        scratch.scope(|scratch_local| {
            let (mut sign_x, scratch_local) = scratch_local.take_ckks_ciphertext_like_scratch(x);
            let (mut sign_y, scratch_local) = scratch_local.take_ckks_ciphertext_like_scratch(y);
            let (mut ratio, scratch_local) = scratch_local.take_ckks_ciphertext_like_scratch(x);
            let (mut correction, mut scratch_local) =
                scratch_local.take_ckks_ciphertext_like_scratch(x);
            self.ckks_copy(&mut ratio, x, &mut scratch_local)?;
            self.ckks_copy(&mut correction, y, &mut scratch_local)?;
            if plan.normalization_steps > 0 {
                self.ckks_mul_pt_const_assign(&mut ratio, &plan.constants, 1, &mut scratch_local)?;
                self.ckks_mul_pt_const_assign(
                    &mut correction,
                    &plan.constants,
                    1,
                    &mut scratch_local,
                )?;
            }
            ckks_sign_into(
                self,
                &mut sign_x,
                &ratio,
                &plan.sign,
                tsk,
                conj_key,
                &mut scratch_local,
            )?;
            ckks_sign_into(
                self,
                &mut sign_y,
                &correction,
                &plan.sign,
                tsk,
                conj_key,
                &mut scratch_local,
            )?;
            self.ckks_mul_assign(&mut ratio, &sign_x, tsk, &mut scratch_local)?;
            self.ckks_goldschmidt_division(
                &mut ratio,
                plan.reciprocal_iters,
                tsk,
                &mut scratch_local,
            )?;
            self.ckks_mul_assign(&mut ratio, &sign_x, tsk, &mut scratch_local)?;
            self.ckks_mul_assign(&mut ratio, &correction, tsk, &mut scratch_local)?;
            self.ckks_eval_approximation(
                res,
                &ratio,
                &plan.atan.approximation,
                tsk,
                &mut scratch_local,
            )?;

            // pi/2 * sign(y) * (1 - sign(x)).
            self.ckks_mul_into(&mut correction, &sign_y, &sign_x, tsk, &mut scratch_local)?;
            self.ckks_neg_assign(&mut correction)?;
            self.ckks_add_assign(&mut correction, &sign_y, &mut scratch_local)?;
            self.ckks_mul_pt_const_assign(&mut correction, &plan.constants, 0, &mut scratch_local)?;
            self.ckks_add_assign(res, &correction, &mut scratch_local)
        })?;
        Ok(())
    }
}
