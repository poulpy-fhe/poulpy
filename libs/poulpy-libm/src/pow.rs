//! Power functions.

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
    api::{CKKSAddOps, CKKSCopyOps, CKKSMulOps, CKKSSubOps},
    layouts::{CKKSPlaintext, CKKSPlaintextVecHostCodec, CKKSScalar, ScratchArenaTakeCKKSCiphertext},
    polynomial::SplitStrategy,
};

use crate::{
    exp::{CKKSExpOps, ExpOptions, ExpPlan},
    log::{CKKSLogOps, LogOptions, LogPlan},
};

/// `pow` plan construction options.
#[derive(Clone, Copy, Debug)]
pub struct PowOptions {
    /// Requested absolute-error bits.
    pub target_bits: f64,
    /// Largest degree considered per fitted stage.
    pub max_degree: usize,
    /// Explicit `exp` reduction steps, or automatic selection.
    pub reduction_steps: Option<usize>,
    /// Poulpy BSGS split strategy.
    pub strategy: SplitStrategy,
}

impl Default for PowOptions {
    fn default() -> Self {
        Self {
            target_bits: 20.0,
            max_degree: 31,
            reduction_steps: None,
            strategy: SplitStrategy::MinDepth,
        }
    }
}

/// Prepared positive-base `pow`.
pub struct PowPlan<P> {
    pub(crate) log: LogPlan<P>,
    pub(crate) exp: ExpPlan<P>,
    pub(crate) base_interval: (f64, f64),
    pub(crate) exponent_interval: (f64, f64),
}

impl PowPlan<CKKSPlaintext<AlignedBuf, i64, Standard>> {
    /// Prepares `x^y = exp(y log(x))` for positive `x`.
    #[allow(clippy::too_many_arguments)]
    pub fn from_precision<F>(
        base_a: F,
        base_b: F,
        exponent_a: F,
        exponent_b: F,
        base2k: poulpy_core::layouts::Base2K,
        coeff_meta: CKKSLayout,
        options: PowOptions,
        module: &Module<HostBytesBackend>,
    ) -> Result<Self>
    where
        F: CKKSScalar + Float + FloatConst + FromPrimitive + ToPrimitive + Debug,
        CKKSPlaintext<AlignedBuf, i64, Standard>: CKKSPlaintextVecHostCodec<F>,
    {
        ensure!(
            base_a.is_finite() && base_b.is_finite() && exponent_a.is_finite() && exponent_b.is_finite(),
            "pow: interval endpoints must be finite"
        );
        ensure!(base_a > F::zero(), "pow: base interval must be positive");
        ensure!(base_b > base_a, "pow: empty base interval");
        ensure!(exponent_b > exponent_a, "pow: empty exponent interval");
        ensure!(
            options.target_bits.is_finite() && options.target_bits > 0.0,
            "pow: target_bits must be positive and finite"
        );
        ensure!(options.max_degree > 0, "pow: max_degree must be positive");
        let log = LogPlan::from_precision(
            base_a,
            base_b,
            base2k,
            coeff_meta,
            LogOptions {
                target_bits: options.target_bits + 4.0,
                max_degree: options.max_degree,
                strategy: options.strategy,
            },
            module,
        )
        .context("pow")?;
        let log_a = base_a.ln();
        let log_b = base_b.ln();
        let products = [exponent_a * log_a, exponent_a * log_b, exponent_b * log_a, exponent_b * log_b];
        let mut product_a = products[0];
        let mut product_b = products[0];
        for value in products.into_iter().skip(1) {
            product_a = product_a.min(value);
            product_b = product_b.max(value);
        }
        if product_a == product_b {
            product_a = product_a - F::epsilon();
            product_b = product_b + F::epsilon();
        }
        let exp = ExpPlan::from_precision(
            product_a,
            product_b,
            base2k,
            coeff_meta,
            ExpOptions {
                target_bits: options.target_bits + 2.0,
                max_degree: options.max_degree,
                reduction_steps: options.reduction_steps,
                strategy: options.strategy,
            },
            module,
        )
        .context("pow")?;
        Ok(Self {
            log,
            exp,
            base_interval: (base_a.to_f64().unwrap_or(f64::NAN), base_b.to_f64().unwrap_or(f64::NAN)),
            exponent_interval: (
                exponent_a.to_f64().unwrap_or(f64::NAN),
                exponent_b.to_f64().unwrap_or(f64::NAN),
            ),
        })
    }
}

impl<P> PowPlan<P> {
    /// Prepared logarithm stage.
    pub fn log(&self) -> &LogPlan<P> {
        &self.log
    }

    /// Prepared exponential stage.
    pub fn exp(&self) -> &ExpPlan<P> {
        &self.exp
    }

    /// Supported positive base interval.
    pub fn base_interval(&self) -> (f64, f64) {
        self.base_interval
    }

    /// Supported exponent interval.
    pub fn exponent_interval(&self) -> (f64, f64) {
        self.exponent_interval
    }

    /// Consumed modulus bits.
    pub fn consumed_bits(&self, base_log_delta: usize, exponent_log_delta: usize) -> usize {
        let product_log_delta = base_log_delta.min(exponent_log_delta);
        self.log.consumed_bits(base_log_delta)
            + base_log_delta.max(exponent_log_delta)
            + self.exp.consumed_bits(product_log_delta)
    }

    /// Multiplicative depth.
    pub fn depth(&self) -> usize {
        self.log.depth() + 1 + self.exp.depth()
    }

    /// Maps prepared plaintexts to another backend.
    pub fn map_plaintexts<Q>(self, mut f: impl FnMut(&P) -> Q) -> PowPlan<Q> {
        PowPlan {
            log: self.log.map_plaintexts(&mut f),
            exp: self.exp.map_plaintexts(f),
            base_interval: self.base_interval,
            exponent_interval: self.exponent_interval,
        }
    }
}

/// Homomorphic power functions.
pub trait CKKSPowOps<BE: Backend> {
    /// Scratch bytes for [`Self::ckks_powi`].
    fn ckks_powi_tmp_bytes<R, I, T>(&self, res: &R, input: &I, tsk: &T) -> usize
    where
        R: CKKSCtBounds,
        I: CKKSCtBounds,
        T: GGLWEInfos;

    /// Evaluates `input^exponent` for a public non-negative integer exponent, by square and multiply.
    ///
    /// `input` needs `log_delta` bits of budget per multiplication, plus the headroom of the result: the remaining
    /// budget `b` is guaranteed to hold a slot value `v` with `|v| < 2^(b - 1)`.
    ///
    /// # Metadata
    ///
    /// The products run at the layout of `input`, then `ckks_copy` writes the result to `res`.
    ///
    /// # Errors
    ///
    /// An error when `input` holds no more budget than the multiplications consume, and the errors of the products
    /// and of the final `ckks_copy`. The products run in scratch, so `res` is untouched on `Err`.
    fn ckks_powi<R, I, H>(
        &self,
        exponent: u32,
        res: &mut R,
        input: &I,
        tsk: &H,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        R: GLWEToBackendMut<BE> + CKKSCtBounds + SetCKKSInfos,
        I: GLWEToBackendRef<BE> + CKKSCtBounds,
        H: GetTensorKey<BE>;

    /// Scratch bytes for [`Self::ckks_pow`].
    fn ckks_pow_tmp_bytes<R, B, E, P, T>(&self, res: &R, base: &B, exponent: &E, plan: &PowPlan<P>, tsk: &T) -> usize
    where
        R: CKKSCtBounds,
        B: CKKSCtBounds,
        E: CKKSCtBounds,
        P: CKKSInfos + LWEInfos,
        T: GGLWEInfos;

    /// Evaluates `base^exponent = exp(exponent * log(base))`. `base` must lie in the plan's positive domain and the
    /// product in the domain of its `exp` stage.
    ///
    /// The inputs need `plan.consumed_bits(base.log_delta(), exponent.log_delta())` bits of budget, plus the headroom
    /// of the result, as [`CKKSExpOps::ckks_exp`] states.
    ///
    /// # Metadata
    ///
    /// As [`CKKSExpOps::ckks_exp`] on the product, which the logarithm and the multiplication compute at the layout of
    /// `base`.
    ///
    /// # Errors
    ///
    /// The errors of the logarithm, the multiplication and [`CKKSExpOps::ckks_exp`]. The first two run in scratch and
    /// `ckks_exp` leaves `res` untouched on `Err`, so `res` is untouched on `Err`.
    fn ckks_pow<R, B, E, P, H>(
        &self,
        res: &mut R,
        base: &B,
        exponent: &E,
        plan: &PowPlan<P>,
        tsk: &H,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        R: GLWEToBackendMut<BE> + CKKSCtBounds + SetCKKSInfos,
        B: GLWEToBackendRef<BE> + CKKSCtBounds,
        E: GLWEToBackendRef<BE> + CKKSCtBounds,
        P: GLWEToBackendRef<BE> + CKKSCtBounds + IntPolyInfos + BSGSMeta,
        H: GetTensorKey<BE>;
}

impl<BE: Backend> CKKSPowOps<BE> for Module<BE>
where
    Module<BE>:
        CKKSAddOps<BE> + CKKSCopyOps<BE> + CKKSExpOps<BE> + CKKSLogOps<BE> + CKKSMulOps<BE> + CKKSSubOps<BE> + GLWEBytesOf<BE>,
{
    fn ckks_powi_tmp_bytes<R, I, T>(&self, res: &R, input: &I, tsk: &T) -> usize
    where
        R: CKKSCtBounds,
        I: CKKSCtBounds,
        T: GGLWEInfos,
    {
        // The accumulator has the layout of `input`.
        BE::scratch_aligned(self.glwe_bytes_of_from_infos(input))
            + self
                .ckks_sub_tmp_bytes(input.size())
                .max(self.ckks_add_one_tmp_bytes(input.size()))
                .max(self.ckks_copy_tmp_bytes(input, input))
                .max(self.ckks_square_tmp_bytes(input, input, tsk))
                .max(self.ckks_mul_tmp_bytes(input, input, input, tsk))
                .max(self.ckks_copy_tmp_bytes(res, input))
    }

    fn ckks_powi<R, I, H>(&self, exponent: u32, res: &mut R, input: &I, tsk: &H, scratch: &mut ScratchArena<'_, BE>) -> Result<()>
    where
        R: GLWEToBackendMut<BE> + CKKSCtBounds + SetCKKSInfos,
        I: GLWEToBackendRef<BE> + CKKSCtBounds,
        H: GetTensorKey<BE>,
    {
        let multiplications = powi_multiplications(exponent);
        let consumed = multiplications * input.log_delta();
        ensure!(
            input.log_budget() > consumed,
            "ckks_powi: log_budget {} <= {consumed} bits required for exponent {exponent}",
            input.log_budget()
        );

        scratch.scope(|scratch_local| {
            let (mut acc, mut scratch_local) = scratch_local.take_ckks_ciphertext_like_scratch(input);
            if exponent == 0 {
                self.ckks_sub_into(&mut acc, input, input, &mut scratch_local)?;
                self.ckks_add_one_assign(&mut acc, &mut scratch_local)?;
            } else {
                self.ckks_copy(&mut acc, input, &mut scratch_local)?;
                let top = 31 - exponent.leading_zeros();
                for bit in (0..top).rev() {
                    self.ckks_square_assign(&mut acc, tsk, &mut scratch_local)?;
                    if exponent & (1 << bit) != 0 {
                        self.ckks_mul_assign(&mut acc, input, tsk, &mut scratch_local)?;
                    }
                }
            }
            self.ckks_copy(res, &acc, &mut scratch_local)?;
            Ok(())
        })
    }

    fn ckks_pow_tmp_bytes<R, B, E, P, T>(&self, res: &R, base: &B, exponent: &E, plan: &PowPlan<P>, tsk: &T) -> usize
    where
        R: CKKSCtBounds,
        B: CKKSCtBounds,
        E: CKKSCtBounds,
        P: CKKSInfos + LWEInfos,
        T: GGLWEInfos,
    {
        // The logarithm and the product have the layout of `base`.
        let ct = BE::scratch_aligned(self.glwe_bytes_of_from_infos(base));
        ct + self.ckks_log_tmp_bytes(base, base, &plan.log, tsk).max(
            ct + self
                .ckks_mul_tmp_bytes(base, base, exponent, tsk)
                .max(self.ckks_exp_tmp_bytes(res, base, &plan.exp, tsk)),
        )
    }

    fn ckks_pow<R, B, E, P, H>(
        &self,
        res: &mut R,
        base: &B,
        exponent: &E,
        plan: &PowPlan<P>,
        tsk: &H,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        R: GLWEToBackendMut<BE> + CKKSCtBounds + SetCKKSInfos,
        B: GLWEToBackendRef<BE> + CKKSCtBounds,
        E: GLWEToBackendRef<BE> + CKKSCtBounds,
        P: GLWEToBackendRef<BE> + CKKSCtBounds + IntPolyInfos + BSGSMeta,
        H: GetTensorKey<BE>,
    {
        scratch.scope(|scratch_local| {
            let (mut log, mut scratch_local) = scratch_local.take_ckks_ciphertext_like_scratch(base);
            self.ckks_log(&mut log, base, &plan.log, tsk, &mut scratch_local)
                .context("ckks_pow")?;
            scratch_local.scope(|product_scratch| {
                let (mut product, mut product_scratch) = product_scratch.take_ckks_ciphertext_like_scratch(&log);
                self.ckks_mul_into(&mut product, &log, exponent, tsk, &mut product_scratch)
                    .context("ckks_pow")?;
                self.ckks_exp(res, &product, &plan.exp, tsk, &mut product_scratch)
                    .context("ckks_pow")
            })
        })
    }
}

fn powi_multiplications(exponent: u32) -> usize {
    if exponent <= 1 {
        0
    } else {
        (31 - exponent.leading_zeros()) as usize + exponent.count_ones() as usize - 1
    }
}

#[cfg(test)]
mod tests {
    use super::powi_multiplications;

    #[test]
    fn multiplication_count() {
        assert_eq!(powi_multiplications(0), 0);
        assert_eq!(powi_multiplications(1), 0);
        assert_eq!(powi_multiplications(2), 1);
        assert_eq!(powi_multiplications(3), 2);
        assert_eq!(powi_multiplications(8), 3);
        assert_eq!(powi_multiplications(15), 6);
    }
}
