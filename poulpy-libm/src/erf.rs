//! Error functions.

use std::fmt::Debug;

use anyhow::Result;
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

/// Error-function plan construction options.
pub type ErfOptions = ApproximationOptions;

define_unary_plan!(ErfPlan, "erf");
define_unary_plan!(ErfcPlan, "erfc");

impl ErfPlan<CKKSPlaintext<Vec<u8>, i64>> {
    /// Fits and prepares `erf` on `[a, b]`.
    pub fn from_precision<F>(
        a: F,
        b: F,
        base2k: poulpy_core::layouts::Base2K,
        coeff_meta: CKKSLayout,
        options: ErfOptions,
        module: &Module<HostBytesBackend>,
    ) -> Result<Self>
    where
        F: CKKSScalar + Float + FloatConst + FromPrimitive + ToPrimitive + Debug,
        CKKSPlaintext<Vec<u8>, i64>: CKKSPlaintextVecHostCodec<F>,
    {
        let (approximation, approximation_bits) = prepare_function(
            "erf",
            |x: F| F::from_f64(libm::erf(x.to_f64().unwrap())).unwrap(),
            a,
            b,
            parity_for_interval(a, b, Parity::Odd),
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

impl ErfcPlan<CKKSPlaintext<Vec<u8>, i64>> {
    /// Fits and prepares `erfc` on `[a, b]`.
    pub fn from_precision<F>(
        a: F,
        b: F,
        base2k: poulpy_core::layouts::Base2K,
        coeff_meta: CKKSLayout,
        options: ErfOptions,
        module: &Module<HostBytesBackend>,
    ) -> Result<Self>
    where
        F: CKKSScalar + Float + FloatConst + FromPrimitive + ToPrimitive + Debug,
        CKKSPlaintext<Vec<u8>, i64>: CKKSPlaintextVecHostCodec<F>,
    {
        let (approximation, approximation_bits) = prepare_function(
            "erfc",
            |x: F| F::from_f64(libm::erfc(x.to_f64().unwrap())).unwrap(),
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

/// Homomorphic error functions on fixed prepared domains.
pub trait CKKSErfOps<BE: Backend> {
    declare_unary_op!(ckks_erf_tmp_bytes, ckks_erf, ErfPlan, "erf");
    declare_unary_op!(ckks_erfc_tmp_bytes, ckks_erfc, ErfcPlan, "erfc");
}

impl<BE: Backend> CKKSErfOps<BE> for Module<BE>
where
    Module<BE>: CKKSApproximationOps<BE>,
{
    impl_unary_op!(ckks_erf_tmp_bytes, ckks_erf, ErfPlan);
    impl_unary_op!(ckks_erfc_tmp_bytes, ckks_erfc, ErfcPlan);
}
