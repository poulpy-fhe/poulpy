//! Softmax over packed real slots.

use poulpy_ckks::GLWEPlaintextInfos;
use poulpy_hal::{AlignedBuf, layouts::Standard};
use std::fmt::Debug;

use anyhow::{Context, Result, ensure};
use num_traits::{Float, FloatConst, FromPrimitive, ToPrimitive};
use poulpy_core::{
    GLWEBytesOf,
    layouts::{
        BSGSMeta, GGLWEInfos, GLWEToBackendMut, GLWEToBackendRef, GetAutomorphismKey, GetTensorKey, IntPolyInfos, LWEInfos,
    },
};
use poulpy_hal::layouts::{Backend, HostBytesBackend, Module, ScratchArena};

use poulpy_ckks::{
    CKKSCtBounds, CKKSInfos, CKKSLayout, SetCKKSInfos,
    api::{CKKSAddOps, CKKSCopyOps, CKKSMulOps},
    layouts::{CKKSModuleAlloc, CKKSPlaintext, CKKSPlaintextVecHostCodec, CKKSScalar, ScratchArenaTakeCKKS},
    polynomial::SplitStrategy,
};

use crate::{
    exp::{CKKSExpOps, ExpOptions, ExpPlan},
    reduce::CKKSReductionOps,
    roots::CKKSInverseOps,
};

/// Softmax plan construction options.
#[derive(Clone, Copy, Debug)]
pub struct SoftmaxOptions {
    /// Requested absolute-error bits.
    pub target_bits: f64,
    /// Largest exponential approximation degree.
    pub max_degree: usize,
    /// Explicit exponential reduction steps, or automatic selection.
    pub reduction_steps: Option<usize>,
    /// Explicit reciprocal iterations, or automatic selection.
    pub reciprocal_iters: Option<usize>,
    /// Poulpy BSGS split strategy.
    pub strategy: SplitStrategy,
}

impl Default for SoftmaxOptions {
    fn default() -> Self {
        Self {
            target_bits: 20.0,
            max_degree: 31,
            reduction_steps: None,
            reciprocal_iters: None,
            strategy: SplitStrategy::MinDepth,
        }
    }
}

/// Prepared full-slot softmax.
pub struct SoftmaxPlan<P> {
    pub(crate) exp: ExpPlan<P>,
    pub(crate) constants: P,
    pub(crate) reciprocal_iters: usize,
    pub(crate) slots: usize,
    pub(crate) input_interval: (f64, f64),
}

impl SoftmaxPlan<CKKSPlaintext<AlignedBuf, i64, Standard>> {
    /// Prepares softmax for dense inputs in `[a, b]`.
    pub fn from_precision<F>(
        a: F,
        b: F,
        base2k: poulpy_core::layouts::Base2K,
        coeff_meta: CKKSLayout,
        options: SoftmaxOptions,
        module: &Module<HostBytesBackend>,
    ) -> Result<Self>
    where
        F: CKKSScalar + Float + FloatConst + FromPrimitive + ToPrimitive + Debug,
        CKKSPlaintext<AlignedBuf, i64, Standard>: CKKSPlaintextVecHostCodec<F>,
    {
        let a64 = a.to_f64().unwrap_or(f64::NAN);
        let b64 = b.to_f64().unwrap_or(f64::NAN);
        ensure!(
            a64.is_finite() && b64.is_finite(),
            "softmax: interval endpoints must be finite"
        );
        ensure!(b > a, "softmax: empty interval [a, b]");
        ensure!(
            options.target_bits.is_finite() && options.target_bits > 0.0,
            "softmax: target_bits must be positive and finite"
        );
        ensure!(options.max_degree > 0, "softmax: max_degree must be positive");
        ensure!(coeff_meta.log_sparsity() == 0, "softmax: sparse packing is not supported");
        let shifted_a = a - b;
        let exp = ExpPlan::from_precision(
            shifted_a,
            F::zero(),
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
        .context("softmax")?;
        let slots = usize::from(coeff_meta.n()) / 2;
        ensure!(slots.is_power_of_two(), "softmax: slot count must be a power of two");
        let reciprocal_iters = match options.reciprocal_iters {
            Some(iters) => {
                ensure!(iters > 0, "softmax: reciprocal_iters must be positive");
                iters
            }
            None => reciprocal_iters(a64 - b64, options.target_bits + 3.0)?,
        };
        let mut constants = module.ckks_pt_coeffs_alloc(2, base2k, coeff_meta.k());
        constants.set_meta(coeff_meta.meta());
        constants
            .encode_host_floats(&[-b, F::one() / F::from_usize(slots).unwrap()])
            .context("softmax: constant encoding failed")?;
        Ok(Self {
            exp,
            constants,
            reciprocal_iters,
            slots,
            input_interval: (a64, b64),
        })
    }
}

impl<P> SoftmaxPlan<P> {
    /// Prepared exponential stage.
    pub fn exp(&self) -> &ExpPlan<P> {
        &self.exp
    }

    /// Packed `[-upper_bound, 1/slots]`.
    pub fn constants(&self) -> &P {
        &self.constants
    }

    /// Goldschmidt iterations for the normalized denominator.
    pub fn reciprocal_iters(&self) -> usize {
        self.reciprocal_iters
    }

    /// Number of dense slots.
    pub fn slots(&self) -> usize {
        self.slots
    }

    /// Supported input interval.
    pub fn input_interval(&self) -> (f64, f64) {
        self.input_interval
    }

    /// Consumed modulus bits.
    pub fn consumed_bits(&self, input_log_delta: usize) -> usize {
        self.exp.consumed_bits(input_log_delta) + (self.reciprocal_iters + 4) * input_log_delta
    }

    /// Multiplicative depth.
    pub fn depth(&self) -> usize {
        self.exp.depth() + self.reciprocal_iters + 4
    }

    /// Maps prepared plaintexts to another backend.
    pub fn map_plaintexts<Q>(self, mut f: impl FnMut(&P) -> Q) -> SoftmaxPlan<Q> {
        SoftmaxPlan {
            exp: self.exp.map_plaintexts(&mut f),
            constants: f(&self.constants),
            reciprocal_iters: self.reciprocal_iters,
            slots: self.slots,
            input_interval: self.input_interval,
        }
    }
}

fn reciprocal_iters(shifted_a: f64, target_bits: f64) -> Result<usize> {
    let delta = 1.0 - shifted_a.exp();
    ensure!(delta < 1.0, "softmax: interval is too wide for reciprocal initialization");
    if delta <= 0.0 {
        return Ok(1);
    }
    let per_power = -delta.log2();
    let mut iters = 1;
    while per_power * 2f64.powi((iters + 1) as i32) < target_bits {
        iters += 1;
        ensure!(iters <= 20, "softmax: reciprocal needs more than 20 iterations");
    }
    Ok(iters)
}

/// Homomorphic softmax over every dense slot.
pub trait CKKSSoftmaxOps<BE: Backend> {
    /// Scratch bytes for [`Self::ckks_softmax`].
    fn ckks_softmax_tmp_bytes<R, I, T, A, P>(
        &self,
        res: &R,
        input: &I,
        tsk: &T,
        rotation_key: &A,
        plan: &SoftmaxPlan<P>,
    ) -> usize
    where
        R: CKKSCtBounds,
        I: CKKSCtBounds,
        T: GGLWEInfos,
        A: GGLWEInfos,
        P: CKKSInfos + LWEInfos;

    /// Evaluates softmax over every dense slot: `exp(x - b) / sum(exp(x - b))` for the plan's upper bound `b`.
    ///
    /// The slots must lie in `plan.input_interval()`, and `input` needs `plan.consumed_bits(input.log_delta())` bits of
    /// budget. `rotation_keys` answers the Galois elements of
    /// [`CKKSReductionOps::ckks_sum_slots_galois_elements`].
    ///
    /// # Metadata
    ///
    /// The stages run at the layout of `input`, then `ckks_copy` writes the result to `res`.
    ///
    /// # Errors
    ///
    /// An error for a sparse input or an input whose slot count is not the plan's, and the errors of
    /// [`CKKSExpOps::ckks_exp`], [`CKKSReductionOps::ckks_sum_slots`],
    /// [`CKKSInverseOps::ckks_goldschmidt_division`], the products and the final `ckks_copy`. The stages run in scratch,
    /// so `res` is untouched on `Err`.
    fn ckks_softmax<R, I, P, H, A>(
        &self,
        res: &mut R,
        input: &I,
        plan: &SoftmaxPlan<P>,
        tsk: &H,
        rotation_keys: &A,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        R: GLWEToBackendMut<BE> + CKKSCtBounds + SetCKKSInfos,
        I: GLWEToBackendRef<BE> + CKKSCtBounds,
        P: GLWEToBackendRef<BE> + CKKSCtBounds + IntPolyInfos + BSGSMeta,
        H: GetTensorKey<BE>,
        A: GetAutomorphismKey<BE>;
}

impl<BE: Backend> CKKSSoftmaxOps<BE> for Module<BE>
where
    Module<BE>: CKKSAddOps<BE>
        + CKKSCopyOps<BE>
        + CKKSExpOps<BE>
        + CKKSInverseOps<BE>
        + CKKSMulOps<BE>
        + CKKSReductionOps<BE>
        + GLWEBytesOf<BE>,
{
    fn ckks_softmax_tmp_bytes<R, I, T, A, P>(&self, res: &R, input: &I, tsk: &T, rotation_key: &A, plan: &SoftmaxPlan<P>) -> usize
    where
        R: CKKSCtBounds,
        I: CKKSCtBounds,
        T: GGLWEInfos,
        A: GGLWEInfos,
        P: CKKSInfos + LWEInfos,
    {
        // The numerator lives throughout, beside the shifted input, then the denominator.
        let ct = BE::scratch_aligned(self.glwe_bytes_of_from_infos(input));
        let shifted = ct
            + self
                .ckks_add_pt_const_tmp_bytes(input.size())
                .max(self.ckks_exp_tmp_bytes(input, input, tsk, &plan.exp));
        let normalize = ct
            + self
                .ckks_sum_slots_tmp_bytes(input, input, rotation_key)
                .max(self.ckks_mul_pt_const_tmp_bytes(input, input, &plan.constants))
                .max(self.ckks_goldschmidt_division_tmp_bytes(input, tsk))
                .max(self.ckks_mul_tmp_bytes(input, input, input, tsk));
        ct + shifted.max(normalize).max(self.ckks_copy_tmp_bytes(res, input))
    }

    fn ckks_softmax<R, I, P, H, A>(
        &self,
        res: &mut R,
        input: &I,
        plan: &SoftmaxPlan<P>,
        tsk: &H,
        rotation_keys: &A,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        R: GLWEToBackendMut<BE> + CKKSCtBounds + SetCKKSInfos,
        I: GLWEToBackendRef<BE> + CKKSCtBounds,
        P: GLWEToBackendRef<BE> + CKKSCtBounds + IntPolyInfos + BSGSMeta,
        H: GetTensorKey<BE>,
        A: GetAutomorphismKey<BE>,
    {
        ensure!(input.log_sparsity() == 0, "ckks_softmax: sparse packing is not supported");
        ensure!(
            usize::from(input.n()) / 2 == plan.slots,
            "ckks_softmax: plan has {} slots, input has {}",
            plan.slots,
            usize::from(input.n()) / 2
        );
        scratch.scope(|scratch_local| {
            let (mut numerator, mut scratch_local) = scratch_local.take_ckks_ciphertext_like_scratch(input);
            scratch_local.scope(|stage| {
                let (mut shifted, mut stage) = stage.take_ckks_ciphertext_like_scratch(input);
                self.ckks_add_pt_const_into(&mut shifted, input, 0, &plan.constants, 0, &mut stage)?;
                self.ckks_exp(&mut numerator, &shifted, &plan.exp, tsk, &mut stage)
            })?;
            scratch_local.scope(|stage| {
                let (mut denominator, mut stage) = stage.take_ckks_ciphertext_like_scratch(&numerator);
                self.ckks_sum_slots(&mut denominator, &numerator, rotation_keys, &mut stage)?;
                self.ckks_mul_pt_const_assign(&mut denominator, &plan.constants, 1, &mut stage)?;
                self.ckks_goldschmidt_division(&mut denominator, plan.reciprocal_iters, tsk, &mut stage)?;
                self.ckks_mul_assign(&mut numerator, &denominator, tsk, &mut stage)?;
                self.ckks_mul_pt_const_assign(&mut numerator, &plan.constants, 1, &mut stage)?;
                anyhow::Ok(())
            })?;
            self.ckks_copy(res, &numerator, &mut scratch_local)?;
            Ok(())
        })
    }
}
