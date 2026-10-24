//! Trigonometric functions.

use poulpy_hal::{AlignedBuf, layouts::Standard};
mod inverse;

pub use inverse::{AcosPlan, AsinPlan, Atan2Options, Atan2Plan, AtanPlan, CKKSAtan2Ops, CKKSInverseTrigOps, InverseTrigOptions};

use std::fmt::Debug;

use anyhow::{Result, ensure};
use num_traits::{Float, FloatConst, FromPrimitive, ToPrimitive};
use poulpy_core::{
    GLWEBytesOf,
    layouts::{BSGSMeta, GGLWEInfos, GLWEToBackendMut, GLWEToBackendRef, GetTensorKey, IntPolyInfos, LWEInfos, SetBSGSMeta},
};
use poulpy_hal::layouts::{Backend, HostBytesBackend, Module, ScratchArena};

use poulpy_ckks::{
    CKKSCtBounds, CKKSInfos, CKKSLayout, SetCKKSInfos,
    api::CKKSCopyOps,
    layouts::{CKKSPlaintext, CKKSPlaintextVecHostCodec, CKKSScalar, ScratchArenaTakeCKKS},
};

use crate::{
    approximation::{ApproximationOptions, CKKSApproximationOps, Parity, parity_for_interval, prepare_function},
    plan::{declare_unary_op, define_unary_plan, impl_unary_op},
};

/// Trigonometric plan construction options.
pub type TrigOptions = ApproximationOptions;

define_unary_plan!(SinPlan, "sin");
define_unary_plan!(CosPlan, "cos");
define_unary_plan!(TanPlan, "tan");

impl SinPlan<CKKSPlaintext<AlignedBuf, i64, Standard>> {
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
        CKKSPlaintext<AlignedBuf, i64, Standard>: CKKSPlaintextVecHostCodec<F>,
    {
        let parity = parity_for_interval(a, b, Parity::Odd);
        let (approximation, approximation_bits) =
            prepare_function("sin", |x: F| x.sin(), a, b, parity, base2k, coeff_meta, options, module)?;
        Ok(Self {
            approximation,
            approximation_bits,
        })
    }
}

impl CosPlan<CKKSPlaintext<AlignedBuf, i64, Standard>> {
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
        CKKSPlaintext<AlignedBuf, i64, Standard>: CKKSPlaintextVecHostCodec<F>,
    {
        let parity = parity_for_interval(a, b, Parity::Even);
        let (approximation, approximation_bits) =
            prepare_function("cos", |x: F| x.cos(), a, b, parity, base2k, coeff_meta, options, module)?;
        Ok(Self {
            approximation,
            approximation_bits,
        })
    }
}

impl TanPlan<CKKSPlaintext<AlignedBuf, i64, Standard>> {
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
        CKKSPlaintext<AlignedBuf, i64, Standard>: CKKSPlaintextVecHostCodec<F>,
    {
        let half_pi = F::PI() / (F::one() + F::one());
        ensure!(a > -half_pi && b < half_pi, "tan: interval must lie in (-pi/2, pi/2)");
        let parity = parity_for_interval(a, b, Parity::Odd);
        let (approximation, approximation_bits) =
            prepare_function("tan", |x: F| x.tan(), a, b, parity, base2k, coeff_meta, options, module)?;
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

    /// Scratch bytes for [`Self::ckks_sincos`].
    fn ckks_sincos_tmp_bytes<S, C, I, T, P, Q>(
        &self,
        sin: &S,
        cos: &C,
        input: &I,
        tsk: &T,
        sin_plan: &SinPlan<P>,
        cos_plan: &CosPlan<Q>,
    ) -> usize
    where
        S: CKKSCtBounds,
        C: CKKSCtBounds,
        I: CKKSCtBounds,
        T: GGLWEInfos,
        P: CKKSInfos + LWEInfos,
        Q: CKKSInfos + LWEInfos;

    /// Evaluates `sin(input)` and `cos(input)`.
    ///
    /// # Metadata
    ///
    /// As [`Self::ckks_sin`] and [`Self::ckks_cos`]. The sine is computed at the layout of `input`, then copied to `sin`.
    ///
    /// # Errors
    ///
    /// The errors of the two evaluations and of the final `ckks_copy`. The sine runs in scratch, so a failure of either
    /// evaluation leaves `sin` untouched, and `cos` keeps the guarantee of [`Self::ckks_cos`].
    #[allow(clippy::too_many_arguments)]
    fn ckks_sincos<S, C, I, P, Q, H>(
        &self,
        sin: &mut S,
        cos: &mut C,
        input: &I,
        sin_plan: &SinPlan<P>,
        cos_plan: &CosPlan<Q>,
        tsk: &H,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        S: GLWEToBackendMut<BE> + CKKSCtBounds + SetCKKSInfos,
        C: GLWEToBackendMut<BE> + CKKSCtBounds + SetCKKSInfos + SetBSGSMeta,
        I: GLWEToBackendRef<BE> + CKKSCtBounds,
        P: GLWEToBackendRef<BE> + CKKSCtBounds + IntPolyInfos + BSGSMeta,
        Q: GLWEToBackendRef<BE> + CKKSCtBounds + IntPolyInfos + BSGSMeta,
        H: GetTensorKey<BE>;

    declare_unary_op!(ckks_tan_tmp_bytes, ckks_tan, TanPlan, "tan");
}

impl<BE: Backend> CKKSTrigOps<BE> for Module<BE>
where
    Module<BE>: CKKSApproximationOps<BE> + CKKSCopyOps<BE> + GLWEBytesOf<BE>,
{
    impl_unary_op!(ckks_sin_tmp_bytes, ckks_sin, SinPlan);
    impl_unary_op!(ckks_cos_tmp_bytes, ckks_cos, CosPlan);

    fn ckks_sincos_tmp_bytes<S, C, I, T, P, Q>(
        &self,
        sin: &S,
        cos: &C,
        input: &I,
        tsk: &T,
        sin_plan: &SinPlan<P>,
        cos_plan: &CosPlan<Q>,
    ) -> usize
    where
        S: CKKSCtBounds,
        C: CKKSCtBounds,
        I: CKKSCtBounds,
        T: GGLWEInfos,
        P: CKKSInfos + LWEInfos,
        Q: CKKSInfos + LWEInfos,
    {
        // The sine has the layout of `input`.
        BE::scratch_aligned(self.glwe_bytes_of_from_infos(input))
            + self
                .ckks_sin_tmp_bytes(input, input, tsk, sin_plan)
                .max(self.ckks_cos_tmp_bytes(cos, input, tsk, cos_plan))
                .max(self.ckks_copy_tmp_bytes(sin, input))
    }

    fn ckks_sincos<S, C, I, P, Q, H>(
        &self,
        sin: &mut S,
        cos: &mut C,
        input: &I,
        sin_plan: &SinPlan<P>,
        cos_plan: &CosPlan<Q>,
        tsk: &H,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        S: GLWEToBackendMut<BE> + CKKSCtBounds + SetCKKSInfos,
        C: GLWEToBackendMut<BE> + CKKSCtBounds + SetCKKSInfos + SetBSGSMeta,
        I: GLWEToBackendRef<BE> + CKKSCtBounds,
        P: GLWEToBackendRef<BE> + CKKSCtBounds + IntPolyInfos + BSGSMeta,
        Q: GLWEToBackendRef<BE> + CKKSCtBounds + IntPolyInfos + BSGSMeta,
        H: GetTensorKey<BE>,
    {
        scratch.scope(|scratch_local| {
            let (mut sine, mut scratch_local) = scratch_local.take_ckks_ciphertext_like_scratch(input);
            self.ckks_sin(&mut sine, input, sin_plan, tsk, &mut scratch_local)?;
            self.ckks_cos(cos, input, cos_plan, tsk, &mut scratch_local)?;
            self.ckks_copy(sin, &sine, &mut scratch_local)?;
            Ok(())
        })
    }

    impl_unary_op!(ckks_tan_tmp_bytes, ckks_tan, TanPlan);
}
