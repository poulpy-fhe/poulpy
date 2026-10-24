//! Exponential functions.

use poulpy_hal::{AlignedBuf, layouts::Standard};
use std::fmt::Debug;

use anyhow::{Context, Result, ensure};
use num_traits::{Float, FloatConst, FromPrimitive, ToPrimitive};
use poulpy_core::{
    GLWEBytesOf,
    layouts::{BSGSMeta, GGLWEInfos, GLWEToBackendMut, GLWEToBackendRef, GetTensorKey, IntPolyInfos, LWEInfos},
};
use poulpy_hal::layouts::{Backend, HostBytesBackend, Module, ScratchArena};

use poulpy_ckks::{
    CKKSCtBounds, CKKSInfos, CKKSLayout, SetCKKSInfos,
    api::{CKKSAddOps, CKKSCopyOps, CKKSMulOps},
    layouts::{CKKSModuleAlloc, CKKSPlaintext, CKKSPlaintextVecHostCodec, CKKSScalar, ScratchArenaTakeCKKS},
    polynomial::SplitStrategy,
};

use crate::approximation::{CKKSApproximationOps, Parity, PolynomialApproximation, degree_for_precision, error_bits};

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
    pub(crate) approximation: PolynomialApproximation<P>,
    pub(crate) reduction_steps: usize,
    pub(crate) approximation_bits: f64,
}

/// Prepared `exp2` approximation.
pub struct Exp2Plan<P> {
    pub(crate) approximation: PolynomialApproximation<P>,
    pub(crate) reduction_steps: usize,
    pub(crate) approximation_bits: f64,
}

/// Prepared `exp10` approximation.
pub struct Exp10Plan<P> {
    pub(crate) approximation: PolynomialApproximation<P>,
    pub(crate) reduction_steps: usize,
    pub(crate) approximation_bits: f64,
}

/// Prepared `expm1` approximation.
pub struct Expm1Plan<P> {
    pub(crate) approximation: PolynomialApproximation<P>,
    pub(crate) two: P,
    pub(crate) reduction_steps: usize,
    pub(crate) approximation_bits: f64,
}

impl ExpPlan<CKKSPlaintext<AlignedBuf, i64, Standard>> {
    /// Fits and prepares `exp` on `[a, b]`.
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
        CKKSPlaintext<AlignedBuf, i64, Standard>: CKKSPlaintextVecHostCodec<F>,
    {
        ensure!(a.is_finite() && b.is_finite(), "exp: interval endpoints must be finite");
        ensure!(b > a, "exp: empty interval [a, b]");
        ensure!(
            options.target_bits.is_finite() && options.target_bits > 0.0,
            "exp: target_bits must be positive and finite"
        );
        ensure!(options.max_degree >= 1, "exp: max_degree must be positive");
        let max_abs = a.abs().max(b.abs()).to_f64().unwrap_or(f64::INFINITY);
        let reduction_steps = options
            .reduction_steps
            .unwrap_or_else(|| if max_abs <= 1.0 { 0 } else { max_abs.log2().ceil() as usize });
        ensure!(reduction_steps <= 1023, "exp: too many reduction steps");
        let divisor = F::from_f64(2f64.powi(reduction_steps as i32)).unwrap();
        let upper = b.to_f64().unwrap_or(f64::INFINITY).max(0.0);
        let fit_bits = options.target_bits + reduction_steps as f64 + upper * std::f64::consts::LOG2_E + 2.0;
        let choice = degree_for_precision(
            |x: F| (x / divisor).exp(),
            a,
            b,
            Parity::Full,
            fit_bits,
            options.max_degree,
            options.strategy,
        )
        .context("exp")?;
        let approximation =
            PolynomialApproximation::from_polynomial(&choice.minimax.poly, base2k, coeff_meta, options.strategy, module)
                .context("exp")?;
        Ok(Self {
            approximation,
            reduction_steps,
            approximation_bits: error_bits(choice.minimax.error),
        })
    }
}

impl Exp2Plan<CKKSPlaintext<AlignedBuf, i64, Standard>> {
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
        CKKSPlaintext<AlignedBuf, i64, Standard>: CKKSPlaintextVecHostCodec<F>,
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
        .context("exp2")?;
        let approximation =
            PolynomialApproximation::from_polynomial(&choice.minimax.poly, base2k, coeff_meta, options.strategy, module)
                .context("exp2")?;
        Ok(Self {
            approximation,
            reduction_steps,
            approximation_bits: error_bits(choice.minimax.error),
        })
    }
}

impl Exp10Plan<CKKSPlaintext<AlignedBuf, i64, Standard>> {
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
        CKKSPlaintext<AlignedBuf, i64, Standard>: CKKSPlaintextVecHostCodec<F>,
    {
        let (reduction_steps, fit_bits) = exp_reduction(a, b, options, std::f64::consts::LOG2_10, "exp10")?;
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
        .context("exp10")?;
        let approximation =
            PolynomialApproximation::from_polynomial(&choice.minimax.poly, base2k, coeff_meta, options.strategy, module)
                .context("exp10")?;
        Ok(Self {
            approximation,
            reduction_steps,
            approximation_bits: error_bits(choice.minimax.error),
        })
    }
}

impl Expm1Plan<CKKSPlaintext<AlignedBuf, i64, Standard>> {
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
        CKKSPlaintext<AlignedBuf, i64, Standard>: CKKSPlaintextVecHostCodec<F>,
    {
        let (reduction_steps, fit_bits) = exp_reduction(a, b, options, std::f64::consts::LOG2_E, "expm1")?;
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
        .context("expm1")?;
        let approximation =
            PolynomialApproximation::from_polynomial(&choice.minimax.poly, base2k, coeff_meta, options.strategy, module)
                .context("expm1")?;
        let mut two = module.ckks_pt_coeffs_alloc(1, base2k, coeff_meta.k());
        two.set_meta(coeff_meta.meta());
        two.encode_host_floats(&[F::from_f64(2.0).unwrap()])
            .context("expm1: constant encoding failed")?;
        Ok(Self {
            approximation,
            two,
            reduction_steps,
            approximation_bits: error_bits(choice.minimax.error),
        })
    }
}

fn exp_reduction<F>(a: F, b: F, options: ExpOptions, growth_bits_per_unit: f64, name: &'static str) -> Result<(usize, f64)>
where
    F: Float + ToPrimitive,
{
    ensure!(a.is_finite() && b.is_finite(), "{name}: interval endpoints must be finite");
    ensure!(b > a, "{name}: empty interval [a, b]");
    ensure!(
        options.target_bits.is_finite() && options.target_bits > 0.0,
        "{name}: target_bits must be positive and finite"
    );
    ensure!(options.max_degree >= 1, "{name}: max_degree must be positive");
    let max_abs = a.abs().max(b.abs()).to_f64().unwrap_or(f64::INFINITY);
    let reduction_steps = options
        .reduction_steps
        .unwrap_or_else(|| if max_abs <= 1.0 { 0 } else { max_abs.log2().ceil() as usize });
    ensure!(reduction_steps <= 1023, "{name}: too many reduction steps");
    let upper = b.to_f64().unwrap_or(f64::INFINITY).max(0.0);
    let fit_bits = options.target_bits + reduction_steps as f64 + upper * growth_bits_per_unit + 2.0;
    Ok((reduction_steps, fit_bits))
}

macro_rules! impl_reduced_plan {
    ($plan:ident) => {
        impl<P> $plan<P> {
            /// Reduced polynomial.
            pub fn approximation(&self) -> &PolynomialApproximation<P> {
                &self.approximation
            }

            /// Number of post-evaluation squarings.
            pub fn reduction_steps(&self) -> usize {
                self.reduction_steps
            }

            /// Absolute-error bits of the reduced polynomial.
            pub fn approximation_bits(&self) -> f64 {
                self.approximation_bits
            }

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
    /// Reduced polynomial.
    pub fn approximation(&self) -> &PolynomialApproximation<P> {
        &self.approximation
    }

    /// Number of doubling steps.
    pub fn reduction_steps(&self) -> usize {
        self.reduction_steps
    }

    /// Absolute-error bits of the reduced polynomial.
    pub fn approximation_bits(&self) -> f64 {
        self.approximation_bits
    }

    /// Packed constant `2` of the doubling recurrence.
    pub fn two(&self) -> &P {
        &self.two
    }

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
    fn ckks_exp_tmp_bytes<R, I, T, P>(&self, res: &R, input: &I, tsk: &T, plan: &ExpPlan<P>) -> usize
    where
        R: CKKSCtBounds,
        I: CKKSCtBounds,
        T: GGLWEInfos,
        P: CKKSInfos + LWEInfos;

    /// Evaluates `exp(input)`.
    ///
    /// The slots must lie in `plan.interval()`. `input` needs `plan.consumed_bits(input.log_delta())` bits of budget,
    /// plus the headroom of the result: the remaining budget `b` is guaranteed to hold a slot value `v` with
    /// `|v| < 2^(b - 1)`, and a larger value can wrap without error.
    ///
    /// # Metadata
    ///
    /// The reduced polynomial and its squarings run at the layout of `input`, whose scale and remaining budget the
    /// result keeps, then `ckks_copy` writes it to `res`.
    ///
    /// # Errors
    ///
    /// An error when `input` holds fewer than `plan.consumed_bits(input.log_delta())` bits of budget, and the errors of
    /// the stages and of the final `ckks_copy`. The stages run in scratch, so `res` is untouched on `Err`.
    fn ckks_exp<R, I, P, H>(
        &self,
        res: &mut R,
        input: &I,
        plan: &ExpPlan<P>,
        tsk: &H,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        R: GLWEToBackendMut<BE> + CKKSCtBounds + SetCKKSInfos,
        I: GLWEToBackendRef<BE> + CKKSCtBounds,
        P: GLWEToBackendRef<BE> + CKKSCtBounds + IntPolyInfos + BSGSMeta,
        H: GetTensorKey<BE>;

    /// Scratch bytes for [`Self::ckks_exp2`].
    fn ckks_exp2_tmp_bytes<R, I, T, P>(&self, res: &R, input: &I, tsk: &T, plan: &Exp2Plan<P>) -> usize
    where
        R: CKKSCtBounds,
        I: CKKSCtBounds,
        T: GGLWEInfos,
        P: CKKSInfos + LWEInfos;

    /// Evaluates `exp2(input)`.
    ///
    /// The slots must lie in `plan.interval()`. `input` needs `plan.consumed_bits(input.log_delta())` bits of budget,
    /// plus the headroom of the result: the remaining budget `b` is guaranteed to hold a slot value `v` with
    /// `|v| < 2^(b - 1)`, and a larger value can wrap without error.
    ///
    /// # Metadata
    ///
    /// The reduced polynomial and its squarings run at the layout of `input`, whose scale and remaining budget the
    /// result keeps, then `ckks_copy` writes it to `res`.
    ///
    /// # Errors
    ///
    /// An error when `input` holds fewer than `plan.consumed_bits(input.log_delta())` bits of budget, and the errors of
    /// the stages and of the final `ckks_copy`. The stages run in scratch, so `res` is untouched on `Err`.
    fn ckks_exp2<R, I, P, H>(
        &self,
        res: &mut R,
        input: &I,
        plan: &Exp2Plan<P>,
        tsk: &H,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        R: GLWEToBackendMut<BE> + CKKSCtBounds + SetCKKSInfos,
        I: GLWEToBackendRef<BE> + CKKSCtBounds,
        P: GLWEToBackendRef<BE> + CKKSCtBounds + IntPolyInfos + BSGSMeta,
        H: GetTensorKey<BE>;

    /// Scratch bytes for [`Self::ckks_exp10`].
    fn ckks_exp10_tmp_bytes<R, I, T, P>(&self, res: &R, input: &I, tsk: &T, plan: &Exp10Plan<P>) -> usize
    where
        R: CKKSCtBounds,
        I: CKKSCtBounds,
        T: GGLWEInfos,
        P: CKKSInfos + LWEInfos;

    /// Evaluates `exp10(input)`.
    ///
    /// The slots must lie in `plan.interval()`. `input` needs `plan.consumed_bits(input.log_delta())` bits of budget,
    /// plus the headroom of the result: the remaining budget `b` is guaranteed to hold a slot value `v` with
    /// `|v| < 2^(b - 1)`, and a larger value can wrap without error.
    ///
    /// # Metadata
    ///
    /// The reduced polynomial and its squarings run at the layout of `input`, whose scale and remaining budget the
    /// result keeps, then `ckks_copy` writes it to `res`.
    ///
    /// # Errors
    ///
    /// An error when `input` holds fewer than `plan.consumed_bits(input.log_delta())` bits of budget, and the errors of
    /// the stages and of the final `ckks_copy`. The stages run in scratch, so `res` is untouched on `Err`.
    fn ckks_exp10<R, I, P, H>(
        &self,
        res: &mut R,
        input: &I,
        plan: &Exp10Plan<P>,
        tsk: &H,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        R: GLWEToBackendMut<BE> + CKKSCtBounds + SetCKKSInfos,
        I: GLWEToBackendRef<BE> + CKKSCtBounds,
        P: GLWEToBackendRef<BE> + CKKSCtBounds + IntPolyInfos + BSGSMeta,
        H: GetTensorKey<BE>;

    /// Scratch bytes for [`Self::ckks_expm1`].
    fn ckks_expm1_tmp_bytes<R, I, T, P>(&self, res: &R, input: &I, tsk: &T, plan: &Expm1Plan<P>) -> usize
    where
        R: CKKSCtBounds,
        I: CKKSCtBounds,
        T: GGLWEInfos,
        P: CKKSInfos + LWEInfos;

    /// Evaluates `expm1(input)`.
    ///
    /// The slots must lie in `plan.interval()`. `input` needs `plan.consumed_bits(input.log_delta())` bits of budget,
    /// plus the headroom of the result: the remaining budget `b` is guaranteed to hold a slot value `v` with
    /// `|v| < 2^(b - 1)`, and a larger value can wrap without error.
    ///
    /// # Metadata
    ///
    /// The reduced polynomial and its doublings run at the layout of `input`, whose scale and remaining budget the
    /// result keeps, then `ckks_copy` writes it to `res`.
    ///
    /// # Errors
    ///
    /// An error when `input` holds fewer than `plan.consumed_bits(input.log_delta())` bits of budget, and the errors of
    /// the stages and of the final `ckks_copy`. The stages run in scratch, so `res` is untouched on `Err`.
    fn ckks_expm1<R, I, P, H>(
        &self,
        res: &mut R,
        input: &I,
        plan: &Expm1Plan<P>,
        tsk: &H,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        R: GLWEToBackendMut<BE> + CKKSCtBounds + SetCKKSInfos,
        I: GLWEToBackendRef<BE> + CKKSCtBounds,
        P: GLWEToBackendRef<BE> + CKKSCtBounds + IntPolyInfos + BSGSMeta,
        H: GetTensorKey<BE>;
}

impl<BE: Backend> CKKSExpOps<BE> for Module<BE>
where
    Module<BE>: CKKSAddOps<BE> + CKKSApproximationOps<BE> + CKKSCopyOps<BE> + CKKSMulOps<BE> + GLWEBytesOf<BE>,
{
    fn ckks_exp_tmp_bytes<R, I, T, P>(&self, res: &R, input: &I, tsk: &T, plan: &ExpPlan<P>) -> usize
    where
        R: CKKSCtBounds,
        I: CKKSCtBounds,
        T: GGLWEInfos,
        P: CKKSInfos + LWEInfos,
    {
        reduced_tmp_bytes(self, res, input, tsk, &plan.approximation, false)
    }

    fn ckks_exp<R, I, P, H>(
        &self,
        res: &mut R,
        input: &I,
        plan: &ExpPlan<P>,
        tsk: &H,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        R: GLWEToBackendMut<BE> + CKKSCtBounds + SetCKKSInfos,
        I: GLWEToBackendRef<BE> + CKKSCtBounds,
        P: GLWEToBackendRef<BE> + CKKSCtBounds + IntPolyInfos + BSGSMeta,
        H: GetTensorKey<BE>,
    {
        eval_reduced(
            self,
            "ckks_exp",
            res,
            input,
            &plan.approximation,
            plan.reduction_steps,
            None,
            tsk,
            scratch,
        )
    }

    fn ckks_exp2_tmp_bytes<R, I, T, P>(&self, res: &R, input: &I, tsk: &T, plan: &Exp2Plan<P>) -> usize
    where
        R: CKKSCtBounds,
        I: CKKSCtBounds,
        T: GGLWEInfos,
        P: CKKSInfos + LWEInfos,
    {
        reduced_tmp_bytes(self, res, input, tsk, &plan.approximation, false)
    }

    fn ckks_exp2<R, I, P, H>(
        &self,
        res: &mut R,
        input: &I,
        plan: &Exp2Plan<P>,
        tsk: &H,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        R: GLWEToBackendMut<BE> + CKKSCtBounds + SetCKKSInfos,
        I: GLWEToBackendRef<BE> + CKKSCtBounds,
        P: GLWEToBackendRef<BE> + CKKSCtBounds + IntPolyInfos + BSGSMeta,
        H: GetTensorKey<BE>,
    {
        eval_reduced(
            self,
            "ckks_exp2",
            res,
            input,
            &plan.approximation,
            plan.reduction_steps,
            None,
            tsk,
            scratch,
        )
    }

    fn ckks_exp10_tmp_bytes<R, I, T, P>(&self, res: &R, input: &I, tsk: &T, plan: &Exp10Plan<P>) -> usize
    where
        R: CKKSCtBounds,
        I: CKKSCtBounds,
        T: GGLWEInfos,
        P: CKKSInfos + LWEInfos,
    {
        reduced_tmp_bytes(self, res, input, tsk, &plan.approximation, false)
    }

    fn ckks_exp10<R, I, P, H>(
        &self,
        res: &mut R,
        input: &I,
        plan: &Exp10Plan<P>,
        tsk: &H,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        R: GLWEToBackendMut<BE> + CKKSCtBounds + SetCKKSInfos,
        I: GLWEToBackendRef<BE> + CKKSCtBounds,
        P: GLWEToBackendRef<BE> + CKKSCtBounds + IntPolyInfos + BSGSMeta,
        H: GetTensorKey<BE>,
    {
        eval_reduced(
            self,
            "ckks_exp10",
            res,
            input,
            &plan.approximation,
            plan.reduction_steps,
            None,
            tsk,
            scratch,
        )
    }

    fn ckks_expm1_tmp_bytes<R, I, T, P>(&self, res: &R, input: &I, tsk: &T, plan: &Expm1Plan<P>) -> usize
    where
        R: CKKSCtBounds,
        I: CKKSCtBounds,
        T: GGLWEInfos,
        P: CKKSInfos + LWEInfos,
    {
        reduced_tmp_bytes(self, res, input, tsk, &plan.approximation, true)
    }

    fn ckks_expm1<R, I, P, H>(
        &self,
        res: &mut R,
        input: &I,
        plan: &Expm1Plan<P>,
        tsk: &H,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        R: GLWEToBackendMut<BE> + CKKSCtBounds + SetCKKSInfos,
        I: GLWEToBackendRef<BE> + CKKSCtBounds,
        P: GLWEToBackendRef<BE> + CKKSCtBounds + IntPolyInfos + BSGSMeta,
        H: GetTensorKey<BE>,
    {
        eval_reduced(
            self,
            "ckks_expm1",
            res,
            input,
            &plan.approximation,
            plan.reduction_steps,
            Some(&plan.two),
            tsk,
            scratch,
        )
    }
}

/// Query of [`eval_reduced`]: the stages run on an accumulator with the layout of `input`.
fn reduced_tmp_bytes<BE, R, I, T, P>(
    module: &Module<BE>,
    res: &R,
    input: &I,
    tsk: &T,
    approximation: &PolynomialApproximation<P>,
    doubling: bool,
) -> usize
where
    BE: Backend,
    Module<BE>: CKKSAddOps<BE> + CKKSApproximationOps<BE> + CKKSCopyOps<BE> + CKKSMulOps<BE> + GLWEBytesOf<BE>,
    R: CKKSCtBounds,
    I: CKKSCtBounds,
    T: GGLWEInfos,
    P: CKKSInfos + LWEInfos,
{
    let ct = BE::scratch_aligned(module.glwe_bytes_of_from_infos(input));
    let step = if doubling {
        ct + module
            .ckks_copy_tmp_bytes(input, input)
            .max(module.ckks_add_pt_const_tmp_bytes(input.size()))
            .max(module.ckks_mul_tmp_bytes(input, input, input, tsk))
    } else {
        module.ckks_square_tmp_bytes(input, input, tsk)
    };
    ct + module
        .ckks_approximation_tmp_bytes(input, input, tsk, approximation)
        .max(step)
        .max(module.ckks_copy_tmp_bytes(res, input))
}

/// Evaluates `approximation` on `input`, then `reduction_steps` squarings, or `x * (x + 2)` doublings with `two`, on an
/// accumulator carved like `input`, and copies the result to `res`.
#[allow(clippy::too_many_arguments)]
fn eval_reduced<BE, R, I, P, H>(
    module: &Module<BE>,
    op: &'static str,
    res: &mut R,
    input: &I,
    approximation: &PolynomialApproximation<P>,
    reduction_steps: usize,
    two: Option<&P>,
    tsk: &H,
    scratch: &mut ScratchArena<'_, BE>,
) -> Result<()>
where
    BE: Backend,
    Module<BE>: CKKSAddOps<BE> + CKKSApproximationOps<BE> + CKKSCopyOps<BE> + CKKSMulOps<BE>,
    R: GLWEToBackendMut<BE> + CKKSCtBounds + SetCKKSInfos,
    I: GLWEToBackendRef<BE> + CKKSCtBounds,
    P: GLWEToBackendRef<BE> + CKKSCtBounds + IntPolyInfos + BSGSMeta,
    H: GetTensorKey<BE>,
{
    let log_delta = input.log_delta();
    let required = approximation.consumed_bits(log_delta) + reduction_steps * approximation.output_log_delta(log_delta);
    ensure!(
        input.log_budget() >= required,
        "{op}: log_budget {} < {required} bits required at log_delta {log_delta}",
        input.log_budget()
    );
    scratch.scope(|scratch_local| {
        let (mut acc, mut scratch_local) = scratch_local.take_ckks_ciphertext_like_scratch(input);
        module
            .ckks_eval_approximation(&mut acc, input, approximation, tsk, &mut scratch_local)
            .context(op)?;
        for _ in 0..reduction_steps {
            match two {
                None => module.ckks_square_assign(&mut acc, tsk, &mut scratch_local).context(op)?,
                Some(two) => scratch_local
                    .scope(|step_scratch| {
                        let (mut plus_two, mut step_scratch) = step_scratch.take_ckks_ciphertext_like_scratch(&acc);
                        module.ckks_copy(&mut plus_two, &acc, &mut step_scratch)?;
                        module.ckks_add_pt_const_assign(&mut plus_two, 0, two, 0, &mut step_scratch)?;
                        module.ckks_mul_assign(&mut acc, &plus_two, tsk, &mut step_scratch)
                    })
                    .context(op)?,
            }
        }
        module.ckks_copy(res, &acc, &mut scratch_local).context(op)
    })
}
