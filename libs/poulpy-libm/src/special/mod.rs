//! Smooth special functions.

mod bessel;
mod gamma;

pub use bessel::{CKKSBesselOps, J0Plan, J1Plan, JnPlan, Y0Plan, Y1Plan, YnPlan};
pub use gamma::{CKKSGammaOps, LgammaPlan, TgammaPlan};

/// Special-function plan construction options.
pub type SpecialOptions = crate::approximation::ApproximationOptions;
