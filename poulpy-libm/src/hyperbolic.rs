//! Hyperbolic functions.

use std::fmt::Debug;

use anyhow::{Result, ensure};
use num_traits::{Float, FloatConst, FromPrimitive, ToPrimitive};
use poulpy_hal::layouts::{Backend, HostBytesBackend, Module};

use poulpy_ckks::{
    CKKSLayout,
    layouts::{CKKSPlaintext, CKKSPlaintextVecHostCodec, CKKSScalar},
};

use crate::approximation::{
    ApproximationOptions, CKKSApproximationOps, Parity, parity_for_interval, prepare_function,
};
use crate::plan::{declare_unary_op, define_unary_plan, impl_unary_op};

/// Hyperbolic plan construction options.
pub type HyperbolicOptions = ApproximationOptions;

define_unary_plan!(SinhPlan, "sinh");
define_unary_plan!(CoshPlan, "cosh");
define_unary_plan!(TanhPlan, "tanh");
define_unary_plan!(AsinhPlan, "asinh");
define_unary_plan!(AcoshPlan, "acosh");
define_unary_plan!(AtanhPlan, "atanh");

macro_rules! impl_host_plan {
    ($plan:ident, $name:literal, $fun:ident, $parity:expr) => {
        impl $plan<CKKSPlaintext<Vec<u8>, i64>> {
            #[doc = concat!("Fits and prepares `", $name, "` on `[a, b]`.")]
            pub fn from_precision<F>(
                a: F,
                b: F,
                base2k: poulpy_core::layouts::Base2K,
                coeff_meta: CKKSLayout,
                options: HyperbolicOptions,
                module: &Module<HostBytesBackend>,
            ) -> Result<Self>
            where
                F: CKKSScalar + Float + FloatConst + FromPrimitive + ToPrimitive + Debug,
                CKKSPlaintext<Vec<u8>, i64>: CKKSPlaintextVecHostCodec<F>,
            {
                let parity = parity_for_interval(a, b, $parity);
                let (approximation, approximation_bits) = prepare_function(
                    $name,
                    |x: F| x.$fun(),
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
    };
}

impl_host_plan!(SinhPlan, "sinh", sinh, Parity::Odd);
impl_host_plan!(CoshPlan, "cosh", cosh, Parity::Even);
impl_host_plan!(TanhPlan, "tanh", tanh, Parity::Odd);
impl_host_plan!(AsinhPlan, "asinh", asinh, Parity::Odd);

impl AcoshPlan<CKKSPlaintext<Vec<u8>, i64>> {
    /// Fits and prepares `acosh` on `[a, b]`, with `a >= 1`.
    pub fn from_precision<F>(
        a: F,
        b: F,
        base2k: poulpy_core::layouts::Base2K,
        coeff_meta: CKKSLayout,
        options: HyperbolicOptions,
        module: &Module<HostBytesBackend>,
    ) -> Result<Self>
    where
        F: CKKSScalar + Float + FloatConst + FromPrimitive + ToPrimitive + Debug,
        CKKSPlaintext<Vec<u8>, i64>: CKKSPlaintextVecHostCodec<F>,
    {
        ensure!(a >= F::one(), "acosh: interval must start at or above 1");
        let (approximation, approximation_bits) = prepare_function(
            "acosh",
            |x: F| x.acosh(),
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

impl AtanhPlan<CKKSPlaintext<Vec<u8>, i64>> {
    /// Fits and prepares `atanh` on `[a, b]`, strictly inside `(-1, 1)`.
    pub fn from_precision<F>(
        a: F,
        b: F,
        base2k: poulpy_core::layouts::Base2K,
        coeff_meta: CKKSLayout,
        options: HyperbolicOptions,
        module: &Module<HostBytesBackend>,
    ) -> Result<Self>
    where
        F: CKKSScalar + Float + FloatConst + FromPrimitive + ToPrimitive + Debug,
        CKKSPlaintext<Vec<u8>, i64>: CKKSPlaintextVecHostCodec<F>,
    {
        ensure!(
            a > -F::one() && b < F::one(),
            "atanh: interval must lie strictly inside (-1, 1)"
        );
        let parity = parity_for_interval(a, b, Parity::Odd);
        let (approximation, approximation_bits) = prepare_function(
            "atanh",
            |x: F| x.atanh(),
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

/// Homomorphic hyperbolic functions on fixed prepared domains.
pub trait CKKSHyperbolicOps<BE: Backend> {
    declare_unary_op!(ckks_sinh_tmp_bytes, ckks_sinh, SinhPlan, "sinh");
    declare_unary_op!(ckks_cosh_tmp_bytes, ckks_cosh, CoshPlan, "cosh");
    declare_unary_op!(ckks_tanh_tmp_bytes, ckks_tanh, TanhPlan, "tanh");
    declare_unary_op!(ckks_asinh_tmp_bytes, ckks_asinh, AsinhPlan, "asinh");
    declare_unary_op!(ckks_acosh_tmp_bytes, ckks_acosh, AcoshPlan, "acosh");
    declare_unary_op!(ckks_atanh_tmp_bytes, ckks_atanh, AtanhPlan, "atanh");
}

impl<BE: Backend> CKKSHyperbolicOps<BE> for Module<BE>
where
    Module<BE>: CKKSApproximationOps<BE>,
{
    impl_unary_op!(ckks_sinh_tmp_bytes, ckks_sinh, SinhPlan);
    impl_unary_op!(ckks_cosh_tmp_bytes, ckks_cosh, CoshPlan);
    impl_unary_op!(ckks_tanh_tmp_bytes, ckks_tanh, TanhPlan);
    impl_unary_op!(ckks_asinh_tmp_bytes, ckks_asinh, AsinhPlan);
    impl_unary_op!(ckks_acosh_tmp_bytes, ckks_acosh, AcoshPlan);
    impl_unary_op!(ckks_atanh_tmp_bytes, ckks_atanh, AtanhPlan);
}
