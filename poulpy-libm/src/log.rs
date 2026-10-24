//! Logarithmic functions.

use std::fmt::Debug;

use anyhow::{Result, ensure};
use num_traits::{Float, FloatConst, FromPrimitive, ToPrimitive};
use poulpy_hal::layouts::{Backend, HostBytesBackend, Module};

use poulpy_ckks::{
    CKKSLayout,
    layouts::{CKKSPlaintext, CKKSPlaintextVecHostCodec, CKKSScalar},
};

use crate::approximation::{ApproximationOptions, CKKSApproximationOps, Parity, prepare_function};
use crate::plan::{declare_unary_op, define_unary_plan, impl_unary_op};

/// Logarithm plan construction options.
pub type LogOptions = ApproximationOptions;

define_unary_plan!(LogPlan, "log");
define_unary_plan!(Log2Plan, "log2");
define_unary_plan!(Log10Plan, "log10");
define_unary_plan!(Log1pPlan, "log1p");

impl LogPlan<CKKSPlaintext<Vec<u8>, i64>> {
    /// Fits and prepares `log` on the positive interval `[a, b]`.
    pub fn from_precision<F>(
        a: F,
        b: F,
        base2k: poulpy_core::layouts::Base2K,
        coeff_meta: CKKSLayout,
        options: LogOptions,
        module: &Module<HostBytesBackend>,
    ) -> Result<Self>
    where
        F: CKKSScalar + Float + FloatConst + FromPrimitive + ToPrimitive + Debug,
        CKKSPlaintext<Vec<u8>, i64>: CKKSPlaintextVecHostCodec<F>,
    {
        ensure!(a > F::zero(), "log: interval must be positive");
        let (approximation, approximation_bits) = prepare_function(
            "log",
            |x: F| x.ln(),
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

impl Log2Plan<CKKSPlaintext<Vec<u8>, i64>> {
    /// Fits and prepares `log2` on the positive interval `[a, b]`.
    pub fn from_precision<F>(
        a: F,
        b: F,
        base2k: poulpy_core::layouts::Base2K,
        coeff_meta: CKKSLayout,
        options: LogOptions,
        module: &Module<HostBytesBackend>,
    ) -> Result<Self>
    where
        F: CKKSScalar + Float + FloatConst + FromPrimitive + ToPrimitive + Debug,
        CKKSPlaintext<Vec<u8>, i64>: CKKSPlaintextVecHostCodec<F>,
    {
        ensure!(a > F::zero(), "log2: interval must be positive");
        let (approximation, approximation_bits) = prepare_function(
            "log2",
            |x: F| x.log2(),
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

impl Log10Plan<CKKSPlaintext<Vec<u8>, i64>> {
    /// Fits and prepares `log10` on the positive interval `[a, b]`.
    pub fn from_precision<F>(
        a: F,
        b: F,
        base2k: poulpy_core::layouts::Base2K,
        coeff_meta: CKKSLayout,
        options: LogOptions,
        module: &Module<HostBytesBackend>,
    ) -> Result<Self>
    where
        F: CKKSScalar + Float + FloatConst + FromPrimitive + ToPrimitive + Debug,
        CKKSPlaintext<Vec<u8>, i64>: CKKSPlaintextVecHostCodec<F>,
    {
        ensure!(a > F::zero(), "log10: interval must be positive");
        let (approximation, approximation_bits) = prepare_function(
            "log10",
            |x: F| x.log10(),
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

impl Log1pPlan<CKKSPlaintext<Vec<u8>, i64>> {
    /// Fits and prepares `log1p` on `[a, b]`, with `a > -1`.
    pub fn from_precision<F>(
        a: F,
        b: F,
        base2k: poulpy_core::layouts::Base2K,
        coeff_meta: CKKSLayout,
        options: LogOptions,
        module: &Module<HostBytesBackend>,
    ) -> Result<Self>
    where
        F: CKKSScalar + Float + FloatConst + FromPrimitive + ToPrimitive + Debug,
        CKKSPlaintext<Vec<u8>, i64>: CKKSPlaintextVecHostCodec<F>,
    {
        ensure!(a > -F::one(), "log1p: interval must be greater than -1");
        let (approximation, approximation_bits) = prepare_function(
            "log1p",
            |x: F| x.ln_1p(),
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

/// Homomorphic logarithms on fixed prepared domains.
pub trait CKKSLogOps<BE: Backend> {
    declare_unary_op!(ckks_log_tmp_bytes, ckks_log, LogPlan, "log");
    declare_unary_op!(ckks_log2_tmp_bytes, ckks_log2, Log2Plan, "log2");
    declare_unary_op!(ckks_log10_tmp_bytes, ckks_log10, Log10Plan, "log10");
    declare_unary_op!(ckks_log1p_tmp_bytes, ckks_log1p, Log1pPlan, "log1p");
}

impl<BE: Backend> CKKSLogOps<BE> for Module<BE>
where
    Module<BE>: CKKSApproximationOps<BE>,
{
    impl_unary_op!(ckks_log_tmp_bytes, ckks_log, LogPlan);
    impl_unary_op!(ckks_log2_tmp_bytes, ckks_log2, Log2Plan);
    impl_unary_op!(ckks_log10_tmp_bytes, ckks_log10, Log10Plan);
    impl_unary_op!(ckks_log1p_tmp_bytes, ckks_log1p, Log1pPlan);
}
