//! Power functions.

use std::fmt::Debug;

use anyhow::{Result, anyhow, ensure};
use num_traits::{Float, FloatConst, FromPrimitive, ToPrimitive};
use poulpy_core::layouts::{
    BSGSMeta, GGLWEInfos, GLWE, GLWETensorKeyPrepared, GLWEToBackendMut, GLWEToBackendRef,
    LWEInfos, SetBSGSMeta, prepared::GLWETensorKeyPreparedToBackendRef,
};
use poulpy_hal::layouts::{Backend, HostBytesBackend, HostStaged, Module, ScratchArena};

use poulpy_ckks::{
    CKKSCtBounds, CKKSInfos, CKKSLayout, SetCKKSInfos,
    api::{CKKSAddOps, CKKSCopyOps, CKKSMulOps, CKKSSubOps},
    layouts::{
        CKKSCiphertext, CKKSPlaintext, CKKSPlaintextVecHostCodec, CKKSScalar, ScratchArenaTakeCKKS,
    },
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
    /// Prepared logarithm stage.
    pub log: LogPlan<P>,
    /// Prepared exponential stage.
    pub exp: ExpPlan<P>,
    /// Supported positive base interval.
    pub base_interval: (f64, f64),
    /// Supported exponent interval.
    pub exponent_interval: (f64, f64),
}

impl PowPlan<CKKSPlaintext<Vec<u8>, i64>> {
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
        CKKSPlaintext<Vec<u8>, i64>: CKKSPlaintextVecHostCodec<F>,
    {
        ensure!(
            base_a.is_finite()
                && base_b.is_finite()
                && exponent_a.is_finite()
                && exponent_b.is_finite(),
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
        .map_err(|e| anyhow!("pow: {e}"))?;
        let log_a = base_a.ln();
        let log_b = base_b.ln();
        let products = [
            exponent_a * log_a,
            exponent_a * log_b,
            exponent_b * log_a,
            exponent_b * log_b,
        ];
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
        .map_err(|e| anyhow!("pow: {e}"))?;
        Ok(Self {
            log,
            exp,
            base_interval: (
                base_a.to_f64().unwrap_or(f64::NAN),
                base_b.to_f64().unwrap_or(f64::NAN),
            ),
            exponent_interval: (
                exponent_a.to_f64().unwrap_or(f64::NAN),
                exponent_b.to_f64().unwrap_or(f64::NAN),
            ),
        })
    }
}

impl<P> PowPlan<P> {
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
    fn ckks_powi_tmp_bytes<R, T>(&self, res: &R, tsk: &T) -> usize
    where
        R: CKKSCtBounds,
        T: GGLWEInfos;

    /// Evaluates `input^exponent` for a public non-negative integer exponent.
    fn ckks_powi(
        &self,
        res: &mut CKKSCiphertext<BE::OwnedBuf, BE::ZnxWord>,
        input: &CKKSCiphertext<BE::OwnedBuf, BE::ZnxWord>,
        exponent: u32,
        tsk: &GLWETensorKeyPrepared<BE::OwnedBuf, BE>,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>;

    /// Scratch bytes for [`Self::ckks_pow`].
    fn ckks_pow_tmp_bytes<R, T, P>(&self, res: &R, tsk: &T, plan: &PowPlan<P>) -> usize
    where
        R: CKKSCtBounds,
        T: GGLWEInfos,
        P: CKKSInfos + LWEInfos;

    /// Evaluates `base^exponent`; `base` must lie in the plan's positive domain.
    fn ckks_pow<P>(
        &self,
        res: &mut CKKSCiphertext<BE::OwnedBuf, BE::ZnxWord>,
        base: &CKKSCiphertext<BE::OwnedBuf, BE::ZnxWord>,
        exponent: &CKKSCiphertext<BE::OwnedBuf, BE::ZnxWord>,
        plan: &PowPlan<P>,
        tsk: &GLWETensorKeyPrepared<BE::OwnedBuf, BE>,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        P: GLWEToBackendRef<BE> + CKKSCtBounds + poulpy_core::layouts::IntPolyInfos + BSGSMeta;
}

impl<BE: Backend + HostStaged> CKKSPowOps<BE> for Module<BE>
where
    Module<BE>: CKKSAddOps<BE>
        + CKKSCopyOps<BE>
        + CKKSExpOps<BE>
        + CKKSLogOps<BE>
        + CKKSMulOps<BE>
        + CKKSSubOps<BE>,
    CKKSCiphertext<BE::OwnedBuf, BE::ZnxWord>:
        GLWEToBackendMut<BE> + GLWEToBackendRef<BE> + CKKSCtBounds + SetCKKSInfos + SetBSGSMeta,
    GLWETensorKeyPrepared<BE::OwnedBuf, BE>: GGLWEInfos + GLWETensorKeyPreparedToBackendRef<BE>,
{
    fn ckks_powi_tmp_bytes<R, T>(&self, res: &R, tsk: &T) -> usize
    where
        R: CKKSCtBounds,
        T: GGLWEInfos,
    {
        self.ckks_mul_tmp_bytes(res, res, res, tsk)
            .max(self.ckks_square_tmp_bytes(res, res, tsk))
            .max(self.ckks_copy_tmp_bytes())
            .max(self.ckks_sub_tmp_bytes())
            .max(self.ckks_add_pt_const_tmp_bytes())
    }

    fn ckks_powi(
        &self,
        res: &mut CKKSCiphertext<BE::OwnedBuf, BE::ZnxWord>,
        input: &CKKSCiphertext<BE::OwnedBuf, BE::ZnxWord>,
        exponent: u32,
        tsk: &GLWETensorKeyPrepared<BE::OwnedBuf, BE>,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()> {
        let multiplications = powi_multiplications(exponent);
        let consumed = multiplications * input.log_delta();
        ensure!(
            input.log_budget() > consumed,
            "ckks_powi: log_budget {} <= {consumed} bits required for exponent {exponent}",
            input.log_budget()
        );

        if exponent == 0 {
            self.ckks_sub_into(res, input, input, scratch)?;
            self.ckks_add_one_assign(res, scratch)?;
            return Ok(());
        }

        self.ckks_copy(res, input, scratch)?;
        let top = 31 - exponent.leading_zeros();
        for bit in (0..top).rev() {
            self.ckks_square_assign(res, tsk, scratch)?;
            if exponent & (1 << bit) != 0 {
                self.ckks_mul_assign(res, input, tsk, scratch)?;
            }
        }
        Ok(())
    }

    fn ckks_pow_tmp_bytes<R, T, P>(&self, res: &R, tsk: &T, plan: &PowPlan<P>) -> usize
    where
        R: CKKSCtBounds,
        T: GGLWEInfos,
        P: CKKSInfos + LWEInfos,
    {
        let ct = GLWE::<Vec<u8>, BE::ZnxWord>::bytes_of_from_infos(res);
        self.ckks_log_tmp_bytes(res, tsk, &plan.log).max(
            ct + self
                .ckks_mul_tmp_bytes(res, res, res, tsk)
                .max(self.ckks_exp_tmp_bytes(res, tsk, &plan.exp)),
        )
    }

    fn ckks_pow<P>(
        &self,
        res: &mut CKKSCiphertext<BE::OwnedBuf, BE::ZnxWord>,
        base: &CKKSCiphertext<BE::OwnedBuf, BE::ZnxWord>,
        exponent: &CKKSCiphertext<BE::OwnedBuf, BE::ZnxWord>,
        plan: &PowPlan<P>,
        tsk: &GLWETensorKeyPrepared<BE::OwnedBuf, BE>,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        P: GLWEToBackendRef<BE> + CKKSCtBounds + poulpy_core::layouts::IntPolyInfos + BSGSMeta,
    {
        self.ckks_log(res, base, &plan.log, tsk, scratch)?;
        scratch.scope(|scratch_local| {
            let (mut product, mut scratch_local) =
                scratch_local.take_ckks_ciphertext_like_scratch(&*res);
            self.ckks_mul_into(&mut product, &*res, exponent, tsk, &mut scratch_local)?;
            self.ckks_exp(res, &product, &plan.exp, tsk, &mut scratch_local)
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
