//! Gamma functions.

use std::fmt::Debug;

use anyhow::{Result, ensure};
use num_traits::{Float, FloatConst, FromPrimitive, ToPrimitive};
use poulpy_hal::layouts::{Backend, HostBytesBackend, Module};

use poulpy_ckks::{
    CKKSLayout,
    layouts::{CKKSPlaintext, CKKSPlaintextVecHostCodec, CKKSScalar},
};

use crate::{
    approximation::{CKKSApproximationOps, Parity, prepare_function},
    plan::{declare_unary_op, define_unary_plan, impl_unary_op},
    special::SpecialOptions,
};

define_unary_plan!(TgammaPlan, "tgamma");
define_unary_plan!(LgammaPlan, "lgamma");

macro_rules! impl_host_plan {
    ($plan:ident, $name:literal, $fun:path) => {
        impl $plan<CKKSPlaintext<Vec<u8>, i64>> {
            #[doc = concat!("Fits and prepares `", $name, "` on positive `[a, b]`.")]
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
                ensure!(a > F::zero(), concat!($name, ": interval must be positive"));
                let (approximation, approximation_bits) = prepare_function(
                    $name,
                    |x: F| F::from_f64($fun(x.to_f64().unwrap())).unwrap(),
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
    };
}

impl_host_plan!(TgammaPlan, "tgamma", libm::tgamma);
impl_host_plan!(LgammaPlan, "lgamma", libm::lgamma);

/// Homomorphic gamma functions on positive prepared domains.
pub trait CKKSGammaOps<BE: Backend> {
    declare_unary_op!(ckks_tgamma_tmp_bytes, ckks_tgamma, TgammaPlan, "tgamma");
    declare_unary_op!(ckks_lgamma_tmp_bytes, ckks_lgamma, LgammaPlan, "lgamma");
}

impl<BE: Backend> CKKSGammaOps<BE> for Module<BE>
where
    Module<BE>: CKKSApproximationOps<BE>,
{
    impl_unary_op!(ckks_tgamma_tmp_bytes, ckks_tgamma, TgammaPlan);
    impl_unary_op!(ckks_lgamma_tmp_bytes, ckks_lgamma, LgammaPlan);
}
