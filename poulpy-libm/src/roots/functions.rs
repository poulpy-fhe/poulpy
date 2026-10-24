//! Root functions.

use std::fmt::Debug;

use anyhow::{Result, ensure};
use num_traits::{Float, FloatConst, FromPrimitive, ToPrimitive};
use poulpy_core::layouts::{
    BSGSMeta, GGLWEInfos, GLWE, GLWETensorKeyPrepared, GLWEToBackendMut, GLWEToBackendRef,
    LWEInfos, SetBSGSMeta, prepared::GLWETensorKeyPreparedToBackendRef,
};
use poulpy_hal::layouts::{Backend, HostBytesBackend, Module, ScratchArena};

use poulpy_ckks::{
    CKKSCtBounds, CKKSInfos, CKKSLayout, SetCKKSInfos,
    api::{CKKSAddOps, CKKSCopyOps, CKKSMulOps},
    layouts::{
        CKKSCiphertext, CKKSModuleAlloc, CKKSPlaintext, CKKSPlaintextVecHostCodec, CKKSScalar,
        ScratchArenaTakeCKKS,
    },
};

use crate::approximation::{
    ApproximationOptions, CKKSApproximationOps, Parity, PolynomialApproximation,
    parity_for_interval, prepare_function,
};
use crate::plan::{declare_unary_op, define_unary_plan, impl_unary_op};

/// Root-function plan construction options.
pub type RootOptions = ApproximationOptions;

define_unary_plan!(CbrtPlan, "cbrt");

/// Prepared `hypot` approximation.
pub struct HypotPlan<P> {
    /// Prepared square-root polynomial.
    pub sqrt: PolynomialApproximation<P>,
    /// Fitted absolute-error bits.
    pub approximation_bits: f64,
    /// Supported `x` interval.
    pub x_interval: (f64, f64),
    /// Supported `y` interval.
    pub y_interval: (f64, f64),
}

impl CbrtPlan<CKKSPlaintext<Vec<u8>, i64>> {
    /// Fits and prepares `cbrt` on `[a, b]`.
    pub fn from_precision<F>(
        a: F,
        b: F,
        base2k: poulpy_core::layouts::Base2K,
        coeff_meta: CKKSLayout,
        options: RootOptions,
        module: &Module<HostBytesBackend>,
    ) -> Result<Self>
    where
        F: CKKSScalar + Float + FloatConst + FromPrimitive + ToPrimitive + Debug,
        CKKSPlaintext<Vec<u8>, i64>: CKKSPlaintextVecHostCodec<F>,
    {
        let parity = parity_for_interval(a, b, Parity::Odd);
        let (approximation, approximation_bits) = prepare_function(
            "cbrt",
            |x: F| x.cbrt(),
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

fn square_range<F: Float>(a: F, b: F) -> (F, F) {
    let aa = a * a;
    let bb = b * b;
    let lo = if a <= F::zero() && b >= F::zero() {
        F::zero()
    } else {
        aa.min(bb)
    };
    (lo, aa.max(bb))
}

impl HypotPlan<CKKSPlaintext<Vec<u8>, i64>> {
    /// Prepares `sqrt(x^2 + y^2)` on the declared intervals.
    #[allow(clippy::too_many_arguments)]
    pub fn from_precision<F>(
        x_a: F,
        x_b: F,
        y_a: F,
        y_b: F,
        base2k: poulpy_core::layouts::Base2K,
        coeff_meta: CKKSLayout,
        options: RootOptions,
        module: &Module<HostBytesBackend>,
    ) -> Result<Self>
    where
        F: CKKSScalar + Float + FloatConst + FromPrimitive + ToPrimitive + Debug,
        CKKSPlaintext<Vec<u8>, i64>: CKKSPlaintextVecHostCodec<F>,
    {
        ensure!(
            x_a.is_finite() && x_b.is_finite(),
            "hypot: x interval endpoints must be finite"
        );
        ensure!(
            y_a.is_finite() && y_b.is_finite(),
            "hypot: y interval endpoints must be finite"
        );
        ensure!(x_b > x_a, "hypot: empty x interval");
        ensure!(y_b > y_a, "hypot: empty y interval");
        let (x2_a, x2_b) = square_range(x_a, x_b);
        let (y2_a, y2_b) = square_range(y_a, y_b);
        let (sqrt, approximation_bits) = prepare_function(
            "hypot",
            |x: F| x.sqrt(),
            x2_a + y2_a,
            x2_b + y2_b,
            Parity::Full,
            base2k,
            coeff_meta,
            options,
            module,
        )?;
        Ok(Self {
            sqrt,
            approximation_bits,
            x_interval: (
                x_a.to_f64().unwrap_or(f64::NAN),
                x_b.to_f64().unwrap_or(f64::NAN),
            ),
            y_interval: (
                y_a.to_f64().unwrap_or(f64::NAN),
                y_b.to_f64().unwrap_or(f64::NAN),
            ),
        })
    }
}

impl<P> HypotPlan<P> {
    /// Consumed modulus bits.
    pub fn consumed_bits(&self, input_log_delta: usize) -> usize {
        input_log_delta + self.sqrt.consumed_bits(input_log_delta)
    }

    /// Multiplicative depth.
    pub fn depth(&self) -> usize {
        1 + self.sqrt.depth()
    }

    /// Square-root polynomial degree.
    pub fn degree(&self) -> usize {
        self.sqrt.degree()
    }

    /// Maps prepared plaintexts to another backend.
    pub fn map_plaintexts<Q>(self, f: impl FnMut(&P) -> Q) -> HypotPlan<Q> {
        HypotPlan {
            sqrt: self.sqrt.map_plaintexts(f),
            approximation_bits: self.approximation_bits,
            x_interval: self.x_interval,
            y_interval: self.y_interval,
        }
    }
}

/// Homomorphic root functions.
pub trait CKKSRootOps<BE: Backend> {
    declare_unary_op!(ckks_cbrt_tmp_bytes, ckks_cbrt, CbrtPlan, "cbrt");

    /// Scratch bytes for [`Self::ckks_hypot`].
    fn ckks_hypot_tmp_bytes<R, T, P>(&self, res: &R, tsk: &T, plan: &HypotPlan<P>) -> usize
    where
        R: CKKSCtBounds,
        T: GGLWEInfos,
        P: CKKSInfos + LWEInfos;

    /// Evaluates `hypot(x, y)`.
    fn ckks_hypot<P>(
        &self,
        res: &mut CKKSCiphertext<BE::OwnedBuf, BE::ZnxWord>,
        x: &CKKSCiphertext<BE::OwnedBuf, BE::ZnxWord>,
        y: &CKKSCiphertext<BE::OwnedBuf, BE::ZnxWord>,
        plan: &HypotPlan<P>,
        tsk: &GLWETensorKeyPrepared<BE::OwnedBuf, BE>,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        P: GLWEToBackendRef<BE> + CKKSCtBounds + poulpy_core::layouts::IntPolyInfos + BSGSMeta;
}

impl<BE: Backend> CKKSRootOps<BE> for Module<BE>
where
    Module<BE>: CKKSAddOps<BE>
        + CKKSApproximationOps<BE>
        + CKKSCopyOps<BE>
        + CKKSMulOps<BE>
        + CKKSModuleAlloc<BE>,
    CKKSCiphertext<BE::OwnedBuf, BE::ZnxWord>:
        GLWEToBackendMut<BE> + GLWEToBackendRef<BE> + CKKSCtBounds + SetCKKSInfos + SetBSGSMeta,
    GLWETensorKeyPrepared<BE::OwnedBuf, BE>: GGLWEInfos + GLWETensorKeyPreparedToBackendRef<BE>,
{
    impl_unary_op!(ckks_cbrt_tmp_bytes, ckks_cbrt, CbrtPlan);

    fn ckks_hypot_tmp_bytes<R, T, P>(&self, res: &R, tsk: &T, plan: &HypotPlan<P>) -> usize
    where
        R: CKKSCtBounds,
        T: GGLWEInfos,
        P: CKKSInfos + LWEInfos,
    {
        let ct = GLWE::<Vec<u8>, BE::ZnxWord>::bytes_of_from_infos(res);
        let square = self.ckks_square_tmp_bytes(res, res, tsk);
        square.max(ct + square.max(self.ckks_add_tmp_bytes())).max(
            ct + self
                .ckks_copy_tmp_bytes()
                .max(self.ckks_approximation_tmp_bytes(res, res, tsk, &plan.sqrt)),
        )
    }

    fn ckks_hypot<P>(
        &self,
        res: &mut CKKSCiphertext<BE::OwnedBuf, BE::ZnxWord>,
        x: &CKKSCiphertext<BE::OwnedBuf, BE::ZnxWord>,
        y: &CKKSCiphertext<BE::OwnedBuf, BE::ZnxWord>,
        plan: &HypotPlan<P>,
        tsk: &GLWETensorKeyPrepared<BE::OwnedBuf, BE>,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        P: GLWEToBackendRef<BE> + CKKSCtBounds + poulpy_core::layouts::IntPolyInfos + BSGSMeta,
    {
        self.ckks_square_into(res, x, tsk, scratch)?;
        scratch.scope(|scratch_local| {
            let (mut y2, mut scratch_local) = scratch_local.take_ckks_ciphertext_like_scratch(y);
            self.ckks_square_into(&mut y2, y, tsk, &mut scratch_local)?;
            self.ckks_add_assign(res, &y2, &mut scratch_local)
        })?;
        scratch.scope(|scratch_local| {
            let (mut sum, mut scratch_local) =
                scratch_local.take_ckks_ciphertext_like_scratch(&*res);
            self.ckks_copy(&mut sum, &*res, &mut scratch_local)?;
            self.ckks_eval_approximation(res, &sum, &plan.sqrt, tsk, &mut scratch_local)
        })?;
        Ok(())
    }
}
