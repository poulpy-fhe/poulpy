//! CKKS math functions built as a thin layer over `poulpy-ckks`.
//!
//! The API follows Poulpy: operations on `Module<BE>`, caller-owned scratch,
//! host-built plans, and explicit precision and modulus costs. Names follow
//! `libm` when the semantics transfer to approximate FHE.

#![warn(missing_docs)]

pub mod approximation;
pub mod arithmetic;
pub mod comparison;
pub mod erf;
pub mod exp;
pub mod hyperbolic;
mod iterative;
pub mod log;
mod plan;
pub mod pow;
mod range;
pub mod reduce;
pub mod roots;
mod sign;
pub mod softmax;
pub mod special;
#[doc(hidden)]
#[allow(missing_docs)]
#[cfg(any(test, feature = "test-utils"))]
pub mod test_suite;
pub mod trig;

#[cfg(test)]
mod backend_tests;

/// Common imports for applications using `poulpy-libm`.
pub mod prelude {
    pub use crate::approximation::{
        ApproximationOptions, CKKSApproximationOps, DegreeChoice, Minimax, Parity,
        PolynomialApproximation, RemezOptions, degree_for_precision, error_bits, minimax,
        minimax_with, precision_at_depth,
    };
    pub use crate::arithmetic::CKKSArithmeticOps;
    pub use crate::comparison::{
        CKKSComparisonOps, CKKSPredicateOps, CKKSSignOps, COEFFS_SIGN_X2_CHEBYSHEV,
        COEFFS_SIGN_X4_CHEBYSHEV, DEFAULT_SIGN_COMPOSITE_CHEBYSHEV, SignComposite,
    };
    pub use crate::erf::{CKKSErfOps, ErfOptions, ErfPlan, ErfcPlan};
    pub use crate::exp::{CKKSExpOps, Exp2Plan, Exp10Plan, ExpOptions, ExpPlan, Expm1Plan};
    pub use crate::hyperbolic::{
        AcoshPlan, AsinhPlan, AtanhPlan, CKKSHyperbolicOps, CoshPlan, HyperbolicOptions, SinhPlan,
        TanhPlan,
    };
    pub use crate::log::{CKKSLogOps, Log1pPlan, Log2Plan, Log10Plan, LogOptions, LogPlan};
    pub use crate::pow::{CKKSPowOps, PowOptions, PowPlan};
    pub use crate::reduce::CKKSReductionOps;
    pub use crate::roots::{
        CKKSDivSqrtOps, CKKSInverseDomainOps, CKKSInverseOps, CKKSRootOps, COMPRESSION_L, CbrtPlan,
        HypotPlan, IntervalNorm, RootOptions,
    };
    pub use crate::softmax::{CKKSSoftmaxOps, SoftmaxOptions, SoftmaxPlan};
    pub use crate::special::{
        CKKSBesselOps, CKKSGammaOps, J0Plan, J1Plan, JnPlan, LgammaPlan, SpecialOptions,
        TgammaPlan, Y0Plan, Y1Plan, YnPlan,
    };
    pub use crate::trig::{
        AcosPlan, AsinPlan, Atan2Options, Atan2Plan, AtanPlan, CKKSAtan2Ops, CKKSInverseTrigOps,
        CKKSTrigOps, CosPlan, InverseTrigOptions, SinPlan, TanPlan, TrigOptions,
    };
}
