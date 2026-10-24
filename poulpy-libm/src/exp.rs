//! Exponential functions.

use std::fmt::Debug;

use anyhow::{Result, anyhow, ensure};
use num_traits::{Float, FloatConst, FromPrimitive, ToPrimitive};
use poulpy_core::layouts::{
    BSGSMeta, GGLWEInfos, GLWE, GLWETensorKeyPrepared, GLWEToBackendMut, GLWEToBackendRef,
    LWEInfos, SetBSGSMeta, prepared::GLWETensorKeyPreparedToBackendRef,
};
use poulpy_hal::layouts::{Backend, HostBytesBackend, Module, ScratchArena};

use poulpy_ckks::{
    CKKSCtBounds, CKKSInfos, CKKSLayout, SetCKKSInfos,
    api::{CKKSAddOps, CKKSAllOpsTmpBytes, CKKSCopyOps, CKKSMulOps},
    layouts::{
        CKKSCiphertext, CKKSModuleAlloc, CKKSPlaintext, CKKSPlaintextVecHostCodec, CKKSScalar,
        ScratchArenaTakeCKKS,
    },
    polynomial::SplitStrategy,
};

use crate::approximation::{
    CKKSApproximationOps, Parity, PolynomialApproximation, degree_for_precision, error_bits,
};

/// `exp` plan construction options.
#[derive(Clone, Copy, Debug)]
pub struct ExpOptions {
    /// Requested absolute-error bits.
    pub target_bits: f64,
    /// Largest degree considered by the fitter.
    pub max_degree: usize,
    /// Squaring count; `None` selects it from the interval.
    pub reduction_steps: Option<usize>,
    /// Poulpy BSGS split strategy.
    pub strategy: SplitStrategy,
}

impl Default for ExpOptions {
    fn default() -> Self {
        Self {
            target_bits: 20.0,
            max_degree: 31,
            reduction_steps: None,
            strategy: SplitStrategy::MinDepth,
        }
    }
}

/// Prepared `exp` approximation.
pub struct ExpPlan<P> {
    /// Reduced exponential polynomial.
    pub approximation: PolynomialApproximation<P>,
    /// Number of post-evaluation squarings.
    pub reduction_steps: usize,
    /// Absolute-error bits of the reduced polynomial.
    pub approximation_bits: f64,
}

/// Prepared `exp2` approximation.
pub struct Exp2Plan<P> {
    /// Reduced exponential polynomial.
    pub approximation: PolynomialApproximation<P>,
    /// Number of post-evaluation squarings.
    pub reduction_steps: usize,
    /// Absolute-error bits of the reduced polynomial.
    pub approximation_bits: f64,
}

/// Prepared `exp10` approximation.
pub struct Exp10Plan<P> {
    /// Reduced exponential polynomial.
    pub approximation: PolynomialApproximation<P>,
    /// Number of post-evaluation squarings.
    pub reduction_steps: usize,
    /// Absolute-error bits of the reduced polynomial.
    pub approximation_bits: f64,
}

/// Prepared `expm1` approximation.
pub struct Expm1Plan<P> {
    /// Reduced `expm1` polynomial.
    pub approximation: PolynomialApproximation<P>,
    /// Packed constant `2` for the doubling recurrence.
    pub two: P,
    /// Number of doubling steps.
    pub reduction_steps: usize,
    /// Absolute-error bits of the reduced polynomial.
    pub approximation_bits: f64,
}

impl ExpPlan<CKKSPlaintext<Vec<u8>, i64>> {
    /// Fits and prepares `exp` on `[a, b]`.
    #[allow(clippy::too_many_arguments)]
    pub fn from_precision<F>(
        a: F,
        b: F,
        base2k: poulpy_core::layouts::Base2K,
        coeff_meta: CKKSLayout,
        options: ExpOptions,
        module: &Module<HostBytesBackend>,
    ) -> Result<Self>
    where
        F: CKKSScalar + Float + FloatConst + FromPrimitive + ToPrimitive + Debug,
        CKKSPlaintext<Vec<u8>, i64>: CKKSPlaintextVecHostCodec<F>,
    {
        ensure!(
            a.is_finite() && b.is_finite(),
            "exp: interval endpoints must be finite"
        );
        ensure!(b > a, "exp: empty interval [a, b]");
        ensure!(
            options.target_bits.is_finite() && options.target_bits > 0.0,
            "exp: target_bits must be positive and finite"
        );
        ensure!(options.max_degree >= 1, "exp: max_degree must be positive");
        let max_abs = a.abs().max(b.abs()).to_f64().unwrap_or(f64::INFINITY);
        let reduction_steps = options.reduction_steps.unwrap_or_else(|| {
            if max_abs <= 1.0 {
                0
            } else {
                max_abs.log2().ceil() as usize
            }
        });
        ensure!(reduction_steps <= 1023, "exp: too many reduction steps");
        let divisor = F::from_f64(2f64.powi(reduction_steps as i32)).unwrap();
        let upper = b.to_f64().unwrap_or(f64::INFINITY).max(0.0);
        let fit_bits =
            options.target_bits + reduction_steps as f64 + upper * std::f64::consts::LOG2_E + 2.0;
        let choice = degree_for_precision(
            |x: F| (x / divisor).exp(),
            a,
            b,
            Parity::Full,
            fit_bits,
            options.max_degree,
            options.strategy,
        )
        .map_err(|e| anyhow!("exp: {e}"))?;
        let approximation = PolynomialApproximation::from_polynomial(
            &choice.minimax.poly,
            base2k,
            coeff_meta,
            options.strategy,
            module,
        )
        .map_err(|e| anyhow!("exp: {e}"))?;
        Ok(Self {
            approximation,
            reduction_steps,
            approximation_bits: error_bits(choice.minimax.error),
        })
    }
}

impl Exp2Plan<CKKSPlaintext<Vec<u8>, i64>> {
    /// Fits and prepares `exp2` on `[a, b]`.
    pub fn from_precision<F>(
        a: F,
        b: F,
        base2k: poulpy_core::layouts::Base2K,
        coeff_meta: CKKSLayout,
        options: ExpOptions,
        module: &Module<HostBytesBackend>,
    ) -> Result<Self>
    where
        F: CKKSScalar + Float + FloatConst + FromPrimitive + ToPrimitive + Debug,
        CKKSPlaintext<Vec<u8>, i64>: CKKSPlaintextVecHostCodec<F>,
    {
        let (reduction_steps, fit_bits) = exp_reduction(a, b, options, 1.0, "exp2")?;
        let divisor = F::from_f64(2f64.powi(reduction_steps as i32)).unwrap();
        let choice = degree_for_precision(
            |x: F| (x / divisor).exp2(),
            a,
            b,
            Parity::Full,
            fit_bits,
            options.max_degree,
            options.strategy,
        )
        .map_err(|e| anyhow!("exp2: {e}"))?;
        let approximation = PolynomialApproximation::from_polynomial(
            &choice.minimax.poly,
            base2k,
            coeff_meta,
            options.strategy,
            module,
        )
        .map_err(|e| anyhow!("exp2: {e}"))?;
        Ok(Self {
            approximation,
            reduction_steps,
            approximation_bits: error_bits(choice.minimax.error),
        })
    }
}

impl Exp10Plan<CKKSPlaintext<Vec<u8>, i64>> {
    /// Fits and prepares `exp10` on `[a, b]`.
    pub fn from_precision<F>(
        a: F,
        b: F,
        base2k: poulpy_core::layouts::Base2K,
        coeff_meta: CKKSLayout,
        options: ExpOptions,
        module: &Module<HostBytesBackend>,
    ) -> Result<Self>
    where
        F: CKKSScalar + Float + FloatConst + FromPrimitive + ToPrimitive + Debug,
        CKKSPlaintext<Vec<u8>, i64>: CKKSPlaintextVecHostCodec<F>,
    {
        let (reduction_steps, fit_bits) =
            exp_reduction(a, b, options, std::f64::consts::LOG2_10, "exp10")?;
        let divisor = F::from_f64(2f64.powi(reduction_steps as i32)).unwrap();
        let log2_10 = F::from_f64(std::f64::consts::LOG2_10).unwrap();
        let choice = degree_for_precision(
            |x: F| (x * log2_10 / divisor).exp2(),
            a,
            b,
            Parity::Full,
            fit_bits,
            options.max_degree,
            options.strategy,
        )
        .map_err(|e| anyhow!("exp10: {e}"))?;
        let approximation = PolynomialApproximation::from_polynomial(
            &choice.minimax.poly,
            base2k,
            coeff_meta,
            options.strategy,
            module,
        )
        .map_err(|e| anyhow!("exp10: {e}"))?;
        Ok(Self {
            approximation,
            reduction_steps,
            approximation_bits: error_bits(choice.minimax.error),
        })
    }
}

impl Expm1Plan<CKKSPlaintext<Vec<u8>, i64>> {
    /// Fits and prepares `expm1` on `[a, b]`.
    pub fn from_precision<F>(
        a: F,
        b: F,
        base2k: poulpy_core::layouts::Base2K,
        coeff_meta: CKKSLayout,
        options: ExpOptions,
        module: &Module<HostBytesBackend>,
    ) -> Result<Self>
    where
        F: CKKSScalar + Float + FloatConst + FromPrimitive + ToPrimitive + Debug,
        CKKSPlaintext<Vec<u8>, i64>: CKKSPlaintextVecHostCodec<F>,
    {
        let (reduction_steps, fit_bits) =
            exp_reduction(a, b, options, std::f64::consts::LOG2_E, "expm1")?;
        let divisor = F::from_f64(2f64.powi(reduction_steps as i32)).unwrap();
        let choice = degree_for_precision(
            |x: F| (x / divisor).exp_m1(),
            a,
            b,
            Parity::Full,
            fit_bits,
            options.max_degree,
            options.strategy,
        )
        .map_err(|e| anyhow!("expm1: {e}"))?;
        let approximation = PolynomialApproximation::from_polynomial(
            &choice.minimax.poly,
            base2k,
            coeff_meta,
            options.strategy,
            module,
        )
        .map_err(|e| anyhow!("expm1: {e}"))?;
        let mut two = module.ckks_pt_coeffs_alloc(1, base2k, coeff_meta.k());
        two.set_meta(coeff_meta.meta());
        two.encode_host_floats(&[F::from_f64(2.0).unwrap()])
            .map_err(|e| anyhow!("expm1: constant encoding failed: {e}"))?;
        Ok(Self {
            approximation,
            two,
            reduction_steps,
            approximation_bits: error_bits(choice.minimax.error),
        })
    }
}

fn exp_reduction<F>(
    a: F,
    b: F,
    options: ExpOptions,
    growth_bits_per_unit: f64,
    name: &'static str,
) -> Result<(usize, f64)>
where
    F: Float + ToPrimitive,
{
    ensure!(
        a.is_finite() && b.is_finite(),
        "{name}: interval endpoints must be finite"
    );
    ensure!(b > a, "{name}: empty interval [a, b]");
    ensure!(
        options.target_bits.is_finite() && options.target_bits > 0.0,
        "{name}: target_bits must be positive and finite"
    );
    ensure!(
        options.max_degree >= 1,
        "{name}: max_degree must be positive"
    );
    let max_abs = a.abs().max(b.abs()).to_f64().unwrap_or(f64::INFINITY);
    let reduction_steps = options.reduction_steps.unwrap_or_else(|| {
        if max_abs <= 1.0 {
            0
        } else {
            max_abs.log2().ceil() as usize
        }
    });
    ensure!(reduction_steps <= 1023, "{name}: too many reduction steps");
    let upper = b.to_f64().unwrap_or(f64::INFINITY).max(0.0);
    let fit_bits =
        options.target_bits + reduction_steps as f64 + upper * growth_bits_per_unit + 2.0;
    Ok((reduction_steps, fit_bits))
}

macro_rules! impl_reduced_plan {
    ($plan:ident) => {
        impl<P> $plan<P> {
            /// Consumed modulus bits.
            pub fn consumed_bits(&self, input_log_delta: usize) -> usize {
                self.approximation.consumed_bits(input_log_delta)
                    + self.reduction_steps * self.approximation.output_log_delta(input_log_delta)
            }

            /// Multiplicative depth.
            pub fn depth(&self) -> usize {
                self.approximation.depth() + self.reduction_steps
            }

            /// Supported input interval.
            pub fn interval(&self) -> (f64, f64) {
                self.approximation.interval()
            }

            /// Reduced polynomial degree.
            pub fn degree(&self) -> usize {
                self.approximation.degree()
            }

            /// Maps all prepared plaintexts to another backend.
            pub fn map_plaintexts<Q>(self, f: impl FnMut(&P) -> Q) -> $plan<Q> {
                $plan {
                    approximation: self.approximation.map_plaintexts(f),
                    reduction_steps: self.reduction_steps,
                    approximation_bits: self.approximation_bits,
                }
            }
        }
    };
}

impl_reduced_plan!(ExpPlan);
impl_reduced_plan!(Exp2Plan);
impl_reduced_plan!(Exp10Plan);

impl<P> Expm1Plan<P> {
    /// Consumed modulus bits.
    pub fn consumed_bits(&self, input_log_delta: usize) -> usize {
        self.approximation.consumed_bits(input_log_delta)
            + self.reduction_steps * self.approximation.output_log_delta(input_log_delta)
    }

    /// Multiplicative depth.
    pub fn depth(&self) -> usize {
        self.approximation.depth() + self.reduction_steps
    }

    /// Supported input interval.
    pub fn interval(&self) -> (f64, f64) {
        self.approximation.interval()
    }

    /// Reduced polynomial degree.
    pub fn degree(&self) -> usize {
        self.approximation.degree()
    }

    /// Maps all prepared plaintexts to another backend.
    pub fn map_plaintexts<Q>(self, mut f: impl FnMut(&P) -> Q) -> Expm1Plan<Q> {
        let approximation = self.approximation.map_plaintexts(&mut f);
        Expm1Plan {
            approximation,
            two: f(&self.two),
            reduction_steps: self.reduction_steps,
            approximation_bits: self.approximation_bits,
        }
    }
}

/// Homomorphic exponential.
pub trait CKKSExpOps<BE: Backend> {
    /// Scratch bytes for [`Self::ckks_exp`].
    fn ckks_exp_tmp_bytes<R, T, P>(&self, res: &R, tsk: &T, plan: &ExpPlan<P>) -> usize
    where
        R: CKKSCtBounds,
        T: GGLWEInfos,
        P: poulpy_ckks::CKKSInfos + poulpy_core::layouts::LWEInfos;

    /// Evaluates `exp(input)`; slots must lie in `plan.interval()`.
    fn ckks_exp<R, I, P>(
        &self,
        res: &mut R,
        input: &I,
        plan: &ExpPlan<P>,
        tsk: &GLWETensorKeyPrepared<BE::OwnedBuf, BE>,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        R: GLWEToBackendMut<BE> + GLWEToBackendRef<BE> + CKKSCtBounds + SetCKKSInfos + SetBSGSMeta,
        I: GLWEToBackendRef<BE> + CKKSCtBounds,
        P: GLWEToBackendRef<BE> + CKKSCtBounds + poulpy_core::layouts::IntPolyInfos + BSGSMeta;

    /// Scratch bytes for [`Self::ckks_exp2`].
    fn ckks_exp2_tmp_bytes<R, T, P>(&self, res: &R, tsk: &T, plan: &Exp2Plan<P>) -> usize
    where
        R: CKKSCtBounds,
        T: GGLWEInfos,
        P: CKKSInfos + poulpy_core::layouts::LWEInfos;

    /// Evaluates `exp2(input)`; slots must lie in `plan.interval()`.
    fn ckks_exp2<R, I, P>(
        &self,
        res: &mut R,
        input: &I,
        plan: &Exp2Plan<P>,
        tsk: &GLWETensorKeyPrepared<BE::OwnedBuf, BE>,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        R: GLWEToBackendMut<BE> + GLWEToBackendRef<BE> + CKKSCtBounds + SetCKKSInfos + SetBSGSMeta,
        I: GLWEToBackendRef<BE> + CKKSCtBounds,
        P: GLWEToBackendRef<BE> + CKKSCtBounds + poulpy_core::layouts::IntPolyInfos + BSGSMeta;

    /// Scratch bytes for [`Self::ckks_exp10`].
    fn ckks_exp10_tmp_bytes<R, T, P>(&self, res: &R, tsk: &T, plan: &Exp10Plan<P>) -> usize
    where
        R: CKKSCtBounds,
        T: GGLWEInfos,
        P: CKKSInfos + poulpy_core::layouts::LWEInfos;

    /// Evaluates `exp10(input)`; slots must lie in `plan.interval()`.
    fn ckks_exp10<R, I, P>(
        &self,
        res: &mut R,
        input: &I,
        plan: &Exp10Plan<P>,
        tsk: &GLWETensorKeyPrepared<BE::OwnedBuf, BE>,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        R: GLWEToBackendMut<BE> + GLWEToBackendRef<BE> + CKKSCtBounds + SetCKKSInfos + SetBSGSMeta,
        I: GLWEToBackendRef<BE> + CKKSCtBounds,
        P: GLWEToBackendRef<BE> + CKKSCtBounds + poulpy_core::layouts::IntPolyInfos + BSGSMeta;

    /// Scratch bytes for [`Self::ckks_expm1`].
    fn ckks_expm1_tmp_bytes<R, T, P>(&self, res: &R, tsk: &T, plan: &Expm1Plan<P>) -> usize
    where
        R: CKKSCtBounds,
        T: GGLWEInfos,
        P: CKKSInfos + poulpy_core::layouts::LWEInfos;

    /// Evaluates `expm1(input)`; slots must lie in `plan.interval()`.
    fn ckks_expm1<R, I, P>(
        &self,
        res: &mut R,
        input: &I,
        plan: &Expm1Plan<P>,
        tsk: &GLWETensorKeyPrepared<BE::OwnedBuf, BE>,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        R: GLWEToBackendMut<BE> + GLWEToBackendRef<BE> + CKKSCtBounds + SetCKKSInfos + SetBSGSMeta,
        I: GLWEToBackendRef<BE> + CKKSCtBounds,
        P: GLWEToBackendRef<BE> + CKKSCtBounds + poulpy_core::layouts::IntPolyInfos + BSGSMeta;
}

impl<BE: Backend> CKKSExpOps<BE> for Module<BE>
where
    Module<BE>: CKKSAddOps<BE>
        + CKKSAllOpsTmpBytes<BE>
        + CKKSApproximationOps<BE>
        + CKKSCopyOps<BE>
        + CKKSMulOps<BE>,
    CKKSCiphertext<BE::OwnedBuf, BE::ZnxWord>:
        GLWEToBackendMut<BE> + GLWEToBackendRef<BE> + CKKSCtBounds + SetCKKSInfos + SetBSGSMeta,
    GLWETensorKeyPrepared<BE::OwnedBuf, BE>: GGLWEInfos + GLWETensorKeyPreparedToBackendRef<BE>,
{
    fn ckks_exp_tmp_bytes<R, T, P>(&self, res: &R, tsk: &T, plan: &ExpPlan<P>) -> usize
    where
        R: CKKSCtBounds,
        T: GGLWEInfos,
        P: poulpy_ckks::CKKSInfos + poulpy_core::layouts::LWEInfos,
    {
        self.ckks_approximation_tmp_bytes(res, res, tsk, &plan.approximation)
    }

    fn ckks_exp<R, I, P>(
        &self,
        res: &mut R,
        input: &I,
        plan: &ExpPlan<P>,
        tsk: &GLWETensorKeyPrepared<BE::OwnedBuf, BE>,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        R: GLWEToBackendMut<BE> + GLWEToBackendRef<BE> + CKKSCtBounds + SetCKKSInfos + SetBSGSMeta,
        I: GLWEToBackendRef<BE> + CKKSCtBounds,
        P: GLWEToBackendRef<BE> + CKKSCtBounds + poulpy_core::layouts::IntPolyInfos + BSGSMeta,
    {
        self.ckks_eval_approximation(res, input, &plan.approximation, tsk, scratch)?;
        for _ in 0..plan.reduction_steps {
            self.ckks_square_assign(res, tsk, scratch)?;
        }
        Ok(())
    }

    fn ckks_exp2_tmp_bytes<R, T, P>(&self, res: &R, tsk: &T, plan: &Exp2Plan<P>) -> usize
    where
        R: CKKSCtBounds,
        T: GGLWEInfos,
        P: CKKSInfos + poulpy_core::layouts::LWEInfos,
    {
        self.ckks_approximation_tmp_bytes(res, res, tsk, &plan.approximation)
    }

    fn ckks_exp2<R, I, P>(
        &self,
        res: &mut R,
        input: &I,
        plan: &Exp2Plan<P>,
        tsk: &GLWETensorKeyPrepared<BE::OwnedBuf, BE>,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        R: GLWEToBackendMut<BE> + GLWEToBackendRef<BE> + CKKSCtBounds + SetCKKSInfos + SetBSGSMeta,
        I: GLWEToBackendRef<BE> + CKKSCtBounds,
        P: GLWEToBackendRef<BE> + CKKSCtBounds + poulpy_core::layouts::IntPolyInfos + BSGSMeta,
    {
        self.ckks_eval_approximation(res, input, &plan.approximation, tsk, scratch)?;
        for _ in 0..plan.reduction_steps {
            self.ckks_square_assign(res, tsk, scratch)?;
        }
        Ok(())
    }

    fn ckks_exp10_tmp_bytes<R, T, P>(&self, res: &R, tsk: &T, plan: &Exp10Plan<P>) -> usize
    where
        R: CKKSCtBounds,
        T: GGLWEInfos,
        P: CKKSInfos + poulpy_core::layouts::LWEInfos,
    {
        self.ckks_approximation_tmp_bytes(res, res, tsk, &plan.approximation)
    }

    fn ckks_exp10<R, I, P>(
        &self,
        res: &mut R,
        input: &I,
        plan: &Exp10Plan<P>,
        tsk: &GLWETensorKeyPrepared<BE::OwnedBuf, BE>,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        R: GLWEToBackendMut<BE> + GLWEToBackendRef<BE> + CKKSCtBounds + SetCKKSInfos + SetBSGSMeta,
        I: GLWEToBackendRef<BE> + CKKSCtBounds,
        P: GLWEToBackendRef<BE> + CKKSCtBounds + poulpy_core::layouts::IntPolyInfos + BSGSMeta,
    {
        self.ckks_eval_approximation(res, input, &plan.approximation, tsk, scratch)?;
        for _ in 0..plan.reduction_steps {
            self.ckks_square_assign(res, tsk, scratch)?;
        }
        Ok(())
    }

    fn ckks_expm1_tmp_bytes<R, T, P>(&self, res: &R, tsk: &T, plan: &Expm1Plan<P>) -> usize
    where
        R: CKKSCtBounds,
        T: GGLWEInfos,
        P: CKKSInfos + poulpy_core::layouts::LWEInfos,
    {
        let ct = GLWE::<Vec<u8>, BE::ZnxWord>::bytes_of_from_infos(res);
        self.ckks_approximation_tmp_bytes(res, res, tsk, &plan.approximation)
            .max(ct + self.ckks_all_ops_tmp_bytes(res, tsk, &plan.two))
    }

    fn ckks_expm1<R, I, P>(
        &self,
        res: &mut R,
        input: &I,
        plan: &Expm1Plan<P>,
        tsk: &GLWETensorKeyPrepared<BE::OwnedBuf, BE>,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        R: GLWEToBackendMut<BE> + GLWEToBackendRef<BE> + CKKSCtBounds + SetCKKSInfos + SetBSGSMeta,
        I: GLWEToBackendRef<BE> + CKKSCtBounds,
        P: GLWEToBackendRef<BE> + CKKSCtBounds + poulpy_core::layouts::IntPolyInfos + BSGSMeta,
    {
        self.ckks_eval_approximation(res, input, &plan.approximation, tsk, scratch)?;
        for _ in 0..plan.reduction_steps {
            scratch.scope(|scratch_local| {
                let (mut plus_two, mut scratch_local) =
                    scratch_local.take_ckks_ciphertext_like_scratch(&*res);
                self.ckks_copy(&mut plus_two, &*res, &mut scratch_local)?;
                self.ckks_add_pt_const_assign(&mut plus_two, 0, &plan.two, 0, &mut scratch_local)?;
                self.ckks_mul_assign(res, &plus_two, tsk, &mut scratch_local)
            })?;
        }
        Ok(())
    }
}
