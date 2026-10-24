//! Softmax over packed real slots.

use std::fmt::Debug;

use anyhow::{Result, anyhow, ensure};
use num_traits::{Float, FloatConst, FromPrimitive, ToPrimitive};
use poulpy_core::layouts::{
    BSGSMeta, GGLWEInfos, GGLWEPreparedToBackendRef, GLWE, GLWEAutomorphismKeyHelper,
    GLWETensorKeyPrepared, GLWEToBackendMut, GLWEToBackendRef, GetGaloisElement, LWEInfos,
    SetBSGSMeta,
    prepared::{GLWEAutomorphismKeyPreparedToBackendRef, GLWETensorKeyPreparedToBackendRef},
};
use poulpy_hal::layouts::{Backend, HostBytesBackend, Module, ScratchArena};

use poulpy_ckks::{
    CKKSCtBounds, CKKSInfos, CKKSLayout, SetCKKSInfos,
    api::{CKKSAddOps, CKKSCopyOps, CKKSMulOps},
    layouts::{
        CKKSCiphertext, CKKSModuleAlloc, CKKSPlaintext, CKKSPlaintextVecHostCodec, CKKSScalar,
        ScratchArenaTakeCKKS,
    },
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
    /// Prepared exponential stage.
    pub exp: ExpPlan<P>,
    /// Packed `[-upper_bound, 1/slots]`.
    pub constants: P,
    /// Goldschmidt iterations for the normalized denominator.
    pub reciprocal_iters: usize,
    /// Number of dense slots.
    pub slots: usize,
    /// Supported input interval.
    pub input_interval: (f64, f64),
}

impl SoftmaxPlan<CKKSPlaintext<Vec<u8>, i64>> {
    /// Prepares softmax for dense inputs in `[a, b]`.
    #[allow(clippy::too_many_arguments)]
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
        CKKSPlaintext<Vec<u8>, i64>: CKKSPlaintextVecHostCodec<F>,
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
        ensure!(
            options.max_degree > 0,
            "softmax: max_degree must be positive"
        );
        ensure!(
            coeff_meta.log_sparsity() == 0,
            "softmax: sparse packing is not supported"
        );
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
        .map_err(|e| anyhow!("softmax: {e}"))?;
        let slots = usize::from(coeff_meta.n()) / 2;
        ensure!(
            slots.is_power_of_two(),
            "softmax: slot count must be a power of two"
        );
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
            .map_err(|e| anyhow!("softmax: constant encoding failed: {e}"))?;
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
    ensure!(
        delta < 1.0,
        "softmax: interval is too wide for reciprocal initialization"
    );
    if delta <= 0.0 {
        return Ok(1);
    }
    let per_power = -delta.log2();
    let mut iters = 1;
    while per_power * 2f64.powi((iters + 1) as i32) < target_bits {
        iters += 1;
        ensure!(
            iters <= 20,
            "softmax: reciprocal needs more than 20 iterations"
        );
    }
    Ok(iters)
}

/// Homomorphic softmax over every dense slot.
pub trait CKKSSoftmaxOps<BE: Backend> {
    /// Scratch bytes for [`Self::ckks_softmax`].
    fn ckks_softmax_tmp_bytes<R, T, A, P>(
        &self,
        res: &R,
        tsk: &T,
        rotation_key: &A,
        plan: &SoftmaxPlan<P>,
    ) -> usize
    where
        R: CKKSCtBounds,
        T: GGLWEInfos,
        A: GGLWEInfos,
        P: CKKSInfos + LWEInfos;

    /// Evaluates softmax over every dense slot.
    fn ckks_softmax<P, H, K>(
        &self,
        res: &mut CKKSCiphertext<BE::OwnedBuf, BE::ZnxWord>,
        input: &CKKSCiphertext<BE::OwnedBuf, BE::ZnxWord>,
        plan: &SoftmaxPlan<P>,
        tsk: &GLWETensorKeyPrepared<BE::OwnedBuf, BE>,
        rotation_keys: &H,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        P: GLWEToBackendRef<BE> + CKKSCtBounds + poulpy_core::layouts::IntPolyInfos + BSGSMeta,
        K: GLWEAutomorphismKeyPreparedToBackendRef<BE>
            + GGLWEPreparedToBackendRef<BE>
            + GetGaloisElement
            + GGLWEInfos,
        H: GLWEAutomorphismKeyHelper<K, BE>;
}

impl<BE: Backend> CKKSSoftmaxOps<BE> for Module<BE>
where
    Module<BE>: CKKSAddOps<BE>
        + CKKSCopyOps<BE>
        + CKKSExpOps<BE>
        + CKKSInverseOps<BE>
        + CKKSMulOps<BE>
        + CKKSModuleAlloc<BE>
        + CKKSReductionOps<BE>,
    CKKSCiphertext<BE::OwnedBuf, BE::ZnxWord>:
        GLWEToBackendMut<BE> + GLWEToBackendRef<BE> + CKKSCtBounds + SetCKKSInfos + SetBSGSMeta,
    GLWETensorKeyPrepared<BE::OwnedBuf, BE>: GGLWEInfos + GLWETensorKeyPreparedToBackendRef<BE>,
{
    fn ckks_softmax_tmp_bytes<R, T, A, P>(
        &self,
        res: &R,
        tsk: &T,
        rotation_key: &A,
        plan: &SoftmaxPlan<P>,
    ) -> usize
    where
        R: CKKSCtBounds,
        T: GGLWEInfos,
        A: GGLWEInfos,
        P: CKKSInfos + LWEInfos,
    {
        let ct = GLWE::<Vec<u8>, BE::ZnxWord>::bytes_of_from_infos(res);
        let shifted = ct
            + self
                .ckks_exp_tmp_bytes(res, tsk, &plan.exp)
                .max(self.ckks_add_pt_const_tmp_bytes());
        let tail = ct
            + self
                .ckks_sum_slots_tmp_bytes(res, rotation_key)
                .max(self.ckks_inverse_tmp_bytes(res, tsk))
                .max(self.ckks_mul_tmp_bytes(res, res, res, tsk))
                .max(self.ckks_mul_pt_const_tmp_bytes(res, res, &plan.constants))
                .max(self.ckks_copy_tmp_bytes());
        shifted.max(tail)
    }

    fn ckks_softmax<P, H, K>(
        &self,
        res: &mut CKKSCiphertext<BE::OwnedBuf, BE::ZnxWord>,
        input: &CKKSCiphertext<BE::OwnedBuf, BE::ZnxWord>,
        plan: &SoftmaxPlan<P>,
        tsk: &GLWETensorKeyPrepared<BE::OwnedBuf, BE>,
        rotation_keys: &H,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        P: GLWEToBackendRef<BE> + CKKSCtBounds + poulpy_core::layouts::IntPolyInfos + BSGSMeta,
        K: GLWEAutomorphismKeyPreparedToBackendRef<BE>
            + GGLWEPreparedToBackendRef<BE>
            + GetGaloisElement
            + GGLWEInfos,
        H: GLWEAutomorphismKeyHelper<K, BE>,
    {
        ensure!(
            input.log_sparsity() == 0,
            "ckks_softmax: sparse packing is not supported"
        );
        ensure!(
            usize::from(input.n()) / 2 == plan.slots,
            "ckks_softmax: plan has {} slots, input has {}",
            plan.slots,
            usize::from(input.n()) / 2
        );
        scratch.scope(|scratch_local| {
            let (mut shifted, mut scratch_local) =
                scratch_local.take_ckks_ciphertext_like_scratch(input);
            self.ckks_add_pt_const_into(
                &mut shifted,
                input,
                0,
                &plan.constants,
                0,
                &mut scratch_local,
            )?;
            self.ckks_exp(res, &shifted, &plan.exp, tsk, &mut scratch_local)
        })?;
        scratch.scope(|scratch_local| {
            let (mut denominator, mut scratch_local) =
                scratch_local.take_ckks_ciphertext_like_scratch(&*res);
            self.ckks_sum_slots(&mut denominator, &*res, rotation_keys, &mut scratch_local)?;
            self.ckks_mul_pt_const_assign(
                &mut denominator,
                &plan.constants,
                1,
                &mut scratch_local,
            )?;
            self.ckks_goldschmidt_division(
                &mut denominator,
                plan.reciprocal_iters,
                tsk,
                &mut scratch_local,
            )?;
            self.ckks_mul_assign(res, &denominator, tsk, &mut scratch_local)?;
            self.ckks_mul_pt_const_assign(res, &plan.constants, 1, &mut scratch_local)
        })?;
        Ok(())
    }
}
