//! Trigonometric functions.

mod inverse;

pub use inverse::{
    AcosPlan, AsinPlan, Atan2Options, Atan2Plan, AtanPlan, CKKSAtan2Ops, CKKSInverseTrigOps,
    InverseTrigOptions,
};

use std::fmt::Debug;

use anyhow::{Result, ensure};
use num_traits::{Float, FloatConst, FromPrimitive, ToPrimitive};
use poulpy_core::layouts::{BSGSMeta, GLWETensorKeyPrepared, GLWEToBackendRef};
use poulpy_hal::layouts::{Backend, HostBytesBackend, Module, ScratchArena};

use poulpy_ckks::{
    CKKSCtBounds, CKKSLayout,
    layouts::{CKKSCiphertext, CKKSPlaintext, CKKSPlaintextVecHostCodec, CKKSScalar},
};

use crate::approximation::{
    ApproximationOptions, CKKSApproximationOps, Parity, parity_for_interval, prepare_function,
};
use crate::plan::{declare_unary_op, define_unary_plan, impl_unary_op};

/// Trigonometric plan construction options.
pub type TrigOptions = ApproximationOptions;

define_unary_plan!(SinPlan, "sin");
define_unary_plan!(CosPlan, "cos");
define_unary_plan!(TanPlan, "tan");

impl SinPlan<CKKSPlaintext<Vec<u8>, i64>> {
    /// Fits and prepares `sin` on `[a, b]` without modular reduction.
    pub fn from_precision<F>(
        a: F,
        b: F,
        base2k: poulpy_core::layouts::Base2K,
        coeff_meta: CKKSLayout,
        options: TrigOptions,
        module: &Module<HostBytesBackend>,
    ) -> Result<Self>
    where
        F: CKKSScalar + Float + FloatConst + FromPrimitive + ToPrimitive + Debug,
        CKKSPlaintext<Vec<u8>, i64>: CKKSPlaintextVecHostCodec<F>,
    {
        let parity = parity_for_interval(a, b, Parity::Odd);
        let (approximation, approximation_bits) = prepare_function(
            "sin",
            |x: F| x.sin(),
            a,
            b,
            parity,
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

impl CosPlan<CKKSPlaintext<Vec<u8>, i64>> {
    /// Fits and prepares `cos` on `[a, b]` without modular reduction.
    pub fn from_precision<F>(
        a: F,
        b: F,
        base2k: poulpy_core::layouts::Base2K,
        coeff_meta: CKKSLayout,
        options: TrigOptions,
        module: &Module<HostBytesBackend>,
    ) -> Result<Self>
    where
        F: CKKSScalar + Float + FloatConst + FromPrimitive + ToPrimitive + Debug,
        CKKSPlaintext<Vec<u8>, i64>: CKKSPlaintextVecHostCodec<F>,
    {
        let parity = parity_for_interval(a, b, Parity::Even);
        let (approximation, approximation_bits) = prepare_function(
            "cos",
            |x: F| x.cos(),
            a,
            b,
            parity,
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

impl TanPlan<CKKSPlaintext<Vec<u8>, i64>> {
    /// Fits and prepares `tan` on `[a, b] ⊂ (-π/2, π/2)`.
    pub fn from_precision<F>(
        a: F,
        b: F,
        base2k: poulpy_core::layouts::Base2K,
        coeff_meta: CKKSLayout,
        options: TrigOptions,
        module: &Module<HostBytesBackend>,
    ) -> Result<Self>
    where
        F: CKKSScalar + Float + FloatConst + FromPrimitive + ToPrimitive + Debug,
        CKKSPlaintext<Vec<u8>, i64>: CKKSPlaintextVecHostCodec<F>,
    {
        let half_pi = F::PI() / (F::one() + F::one());
        ensure!(
            a > -half_pi && b < half_pi,
            "tan: interval must lie in (-pi/2, pi/2)"
        );
        let parity = parity_for_interval(a, b, Parity::Odd);
        let (approximation, approximation_bits) = prepare_function(
            "tan",
            |x: F| x.tan(),
            a,
            b,
            parity,
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

/// Homomorphic trigonometric functions on fixed prepared domains.
pub trait CKKSTrigOps<BE: Backend> {
    declare_unary_op!(ckks_sin_tmp_bytes, ckks_sin, SinPlan, "sin");
    declare_unary_op!(ckks_cos_tmp_bytes, ckks_cos, CosPlan, "cos");

    /// Evaluates `sin(input)` and `cos(input)`.
    #[allow(clippy::too_many_arguments)]
    fn ckks_sincos<P, Q>(
        &self,
        sin: &mut CKKSCiphertext<BE::OwnedBuf, BE::ZnxWord>,
        cos: &mut CKKSCiphertext<BE::OwnedBuf, BE::ZnxWord>,
        input: &CKKSCiphertext<BE::OwnedBuf, BE::ZnxWord>,
        sin_plan: &SinPlan<P>,
        cos_plan: &CosPlan<Q>,
        tsk: &GLWETensorKeyPrepared<BE::OwnedBuf, BE>,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        P: GLWEToBackendRef<BE> + CKKSCtBounds + poulpy_core::layouts::IntPolyInfos + BSGSMeta,
        Q: GLWEToBackendRef<BE> + CKKSCtBounds + poulpy_core::layouts::IntPolyInfos + BSGSMeta;

    declare_unary_op!(ckks_tan_tmp_bytes, ckks_tan, TanPlan, "tan");
}

impl<BE: Backend> CKKSTrigOps<BE> for Module<BE>
where
    Module<BE>: CKKSApproximationOps<BE>,
{
    impl_unary_op!(ckks_sin_tmp_bytes, ckks_sin, SinPlan);
    impl_unary_op!(ckks_cos_tmp_bytes, ckks_cos, CosPlan);

    fn ckks_sincos<P, Q>(
        &self,
        sin: &mut CKKSCiphertext<BE::OwnedBuf, BE::ZnxWord>,
        cos: &mut CKKSCiphertext<BE::OwnedBuf, BE::ZnxWord>,
        input: &CKKSCiphertext<BE::OwnedBuf, BE::ZnxWord>,
        sin_plan: &SinPlan<P>,
        cos_plan: &CosPlan<Q>,
        tsk: &GLWETensorKeyPrepared<BE::OwnedBuf, BE>,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        P: GLWEToBackendRef<BE> + CKKSCtBounds + poulpy_core::layouts::IntPolyInfos + BSGSMeta,
        Q: GLWEToBackendRef<BE> + CKKSCtBounds + poulpy_core::layouts::IntPolyInfos + BSGSMeta,
    {
        self.ckks_sin(sin, input, sin_plan, tsk, scratch)?;
        self.ckks_cos(cos, input, cos_plan, tsk, scratch)
    }

    impl_unary_op!(ckks_tan_tmp_bytes, ckks_tan, TanPlan);
}
