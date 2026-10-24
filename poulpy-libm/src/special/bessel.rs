//! Bessel functions.

use std::fmt::Debug;

use anyhow::{Result, ensure};
use num_traits::{Float, FloatConst, FromPrimitive, ToPrimitive};
use poulpy_hal::layouts::{Backend, HostBytesBackend, Module};

use poulpy_ckks::{
    CKKSLayout,
    layouts::{CKKSPlaintext, CKKSPlaintextVecHostCodec, CKKSScalar},
};

use crate::{
    approximation::{
        CKKSApproximationOps, Parity, PolynomialApproximation, parity_for_interval,
        prepare_function,
    },
    plan::{declare_unary_op, define_unary_plan, impl_unary_op},
    special::SpecialOptions,
};

define_unary_plan!(J0Plan, "j0");
define_unary_plan!(J1Plan, "j1");
define_unary_plan!(Y0Plan, "y0");
define_unary_plan!(Y1Plan, "y1");

macro_rules! impl_host_plan {
    ($plan:ident, $name:literal, $fun:path, $parity:expr, $positive:expr) => {
        impl $plan<CKKSPlaintext<Vec<u8>, i64>> {
            #[doc = concat!("Fits and prepares `", $name, "` on `[a, b]`.")]
            pub fn from_precision<F>(
                a: F,
                b: F,
                base2k: poulpy_core::layouts::Base2K,
                coeff_meta: CKKSLayout,
                options: SpecialOptions,
                module: &Module<HostBytesBackend>,
            ) -> Result<Self>
            where
                F: CKKSScalar + Float + FloatConst + FromPrimitive + ToPrimitive + Debug,
                CKKSPlaintext<Vec<u8>, i64>: CKKSPlaintextVecHostCodec<F>,
            {
                if $positive {
                    ensure!(a > F::zero(), concat!($name, ": interval must be positive"));
                }
                let parity = parity_for_interval(a, b, $parity);
                let (approximation, approximation_bits) = prepare_function(
                    $name,
                    |x: F| F::from_f64($fun(x.to_f64().unwrap())).unwrap(),
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

impl_host_plan!(J0Plan, "j0", libm::j0, Parity::Even, false);
impl_host_plan!(J1Plan, "j1", libm::j1, Parity::Odd, false);
impl_host_plan!(Y0Plan, "y0", libm::y0, Parity::Full, true);
impl_host_plan!(Y1Plan, "y1", libm::y1, Parity::Full, true);

/// Prepared `jn` approximation for a public integer order.
pub struct JnPlan<P> {
    /// Prepared polynomial.
    pub approximation: PolynomialApproximation<P>,
    /// Fitted absolute-error bits.
    pub approximation_bits: f64,
    /// Public Bessel order.
    pub order: i32,
}

/// Prepared `yn` approximation for a public integer order.
pub struct YnPlan<P> {
    /// Prepared polynomial.
    pub approximation: PolynomialApproximation<P>,
    /// Fitted absolute-error bits.
    pub approximation_bits: f64,
    /// Public Bessel order.
    pub order: i32,
}

impl JnPlan<CKKSPlaintext<Vec<u8>, i64>> {
    /// Fits and prepares `jn(order, x)` on `[a, b]`.
    #[allow(clippy::too_many_arguments)]
    pub fn from_precision<F>(
        order: i32,
        a: F,
        b: F,
        base2k: poulpy_core::layouts::Base2K,
        coeff_meta: CKKSLayout,
        options: SpecialOptions,
        module: &Module<HostBytesBackend>,
    ) -> Result<Self>
    where
        F: CKKSScalar + Float + FloatConst + FromPrimitive + ToPrimitive + Debug,
        CKKSPlaintext<Vec<u8>, i64>: CKKSPlaintextVecHostCodec<F>,
    {
        let target_parity = if order.unsigned_abs().is_multiple_of(2) {
            Parity::Even
        } else {
            Parity::Odd
        };
        let parity = parity_for_interval(a, b, target_parity);
        let (approximation, approximation_bits) = prepare_function(
            "jn",
            |x: F| F::from_f64(libm::jn(order, x.to_f64().unwrap())).unwrap(),
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
            order,
        })
    }
}

impl YnPlan<CKKSPlaintext<Vec<u8>, i64>> {
    /// Fits and prepares `yn(order, x)` on positive `[a, b]`.
    #[allow(clippy::too_many_arguments)]
    pub fn from_precision<F>(
        order: i32,
        a: F,
        b: F,
        base2k: poulpy_core::layouts::Base2K,
        coeff_meta: CKKSLayout,
        options: SpecialOptions,
        module: &Module<HostBytesBackend>,
    ) -> Result<Self>
    where
        F: CKKSScalar + Float + FloatConst + FromPrimitive + ToPrimitive + Debug,
        CKKSPlaintext<Vec<u8>, i64>: CKKSPlaintextVecHostCodec<F>,
    {
        ensure!(a > F::zero(), "yn: interval must be positive");
        let (approximation, approximation_bits) = prepare_function(
            "yn",
            |x: F| F::from_f64(libm::yn(order, x.to_f64().unwrap())).unwrap(),
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
            order,
        })
    }
}

macro_rules! impl_ordered_plan {
    ($plan:ident) => {
        impl<P> $plan<P> {
            /// Consumed modulus bits.
            pub fn consumed_bits(&self, input_log_delta: usize) -> usize {
                self.approximation.consumed_bits(input_log_delta)
            }

            /// Multiplicative depth.
            pub fn depth(&self) -> usize {
                self.approximation.depth()
            }

            /// Supported input interval.
            pub fn interval(&self) -> (f64, f64) {
                self.approximation.interval()
            }

            /// Polynomial degree.
            pub fn degree(&self) -> usize {
                self.approximation.degree()
            }

            /// Maps prepared plaintexts to another backend.
            pub fn map_plaintexts<Q>(self, f: impl FnMut(&P) -> Q) -> $plan<Q> {
                $plan {
                    approximation: self.approximation.map_plaintexts(f),
                    approximation_bits: self.approximation_bits,
                    order: self.order,
                }
            }
        }
    };
}

impl_ordered_plan!(JnPlan);
impl_ordered_plan!(YnPlan);

/// Homomorphic Bessel functions on fixed prepared domains.
pub trait CKKSBesselOps<BE: Backend> {
    declare_unary_op!(ckks_j0_tmp_bytes, ckks_j0, J0Plan, "j0");
    declare_unary_op!(ckks_j1_tmp_bytes, ckks_j1, J1Plan, "j1");
    declare_unary_op!(ckks_jn_tmp_bytes, ckks_jn, JnPlan, "jn");
    declare_unary_op!(ckks_y0_tmp_bytes, ckks_y0, Y0Plan, "y0");
    declare_unary_op!(ckks_y1_tmp_bytes, ckks_y1, Y1Plan, "y1");
    declare_unary_op!(ckks_yn_tmp_bytes, ckks_yn, YnPlan, "yn");
}

impl<BE: Backend> CKKSBesselOps<BE> for Module<BE>
where
    Module<BE>: CKKSApproximationOps<BE>,
{
    impl_unary_op!(ckks_j0_tmp_bytes, ckks_j0, J0Plan);
    impl_unary_op!(ckks_j1_tmp_bytes, ckks_j1, J1Plan);
    impl_unary_op!(ckks_jn_tmp_bytes, ckks_jn, JnPlan);
    impl_unary_op!(ckks_y0_tmp_bytes, ckks_y0, Y0Plan);
    impl_unary_op!(ckks_y1_tmp_bytes, ckks_y1, Y1Plan);
    impl_unary_op!(ckks_yn_tmp_bytes, ckks_yn, YnPlan);
}
