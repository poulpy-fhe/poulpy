//! Polynomial fitting and prepared evaluation.

use std::fmt::Debug;

use anyhow::{Result, anyhow, ensure};
use num_traits::{Float, FloatConst, FromPrimitive, ToPrimitive};
use poulpy_ckks::{
    CKKSLayout,
    layouts::{CKKSPlaintext, CKKSPlaintextVecHostCodec, CKKSScalar},
    polynomial::SplitStrategy,
};
use poulpy_core::layouts::Base2K;
use poulpy_hal::layouts::{HostBytesBackend, Module};

pub use poulpy_ckks::approximation::{
    CKKSApproximationOps, DegreeChoice, Minimax, Parity, PolynomialApproximation, RemezOptions,
    degree_for_precision, error_bits, minimax, minimax_with, precision_at_depth,
};

/// Common options for fixed-domain function plans.
#[derive(Clone, Copy, Debug)]
pub struct ApproximationOptions {
    /// Requested absolute-error bits.
    pub target_bits: f64,
    /// Largest degree considered by the fitter.
    pub max_degree: usize,
    /// Poulpy BSGS split strategy.
    pub strategy: SplitStrategy,
}

impl Default for ApproximationOptions {
    fn default() -> Self {
        Self {
            target_bits: 20.0,
            max_degree: 31,
            strategy: SplitStrategy::MinDepth,
        }
    }
}

#[allow(clippy::too_many_arguments, clippy::type_complexity)]
pub(crate) fn prepare_function<F, Fun>(
    name: &'static str,
    fun: Fun,
    a: F,
    b: F,
    parity: Parity,
    base2k: Base2K,
    coeff_meta: CKKSLayout,
    options: ApproximationOptions,
    module: &Module<HostBytesBackend>,
) -> Result<(PolynomialApproximation<CKKSPlaintext<Vec<u8>, i64>>, f64)>
where
    F: CKKSScalar + Float + FloatConst + FromPrimitive + ToPrimitive + Debug,
    Fun: Fn(F) -> F + Copy,
    CKKSPlaintext<Vec<u8>, i64>: CKKSPlaintextVecHostCodec<F>,
{
    ensure!(
        a.is_finite() && b.is_finite(),
        "{name}: interval endpoints must be finite"
    );
    ensure!(b > a, "{name}: empty interval [a, b]");
    ensure!(
        options.target_bits.is_finite() && options.target_bits > 0.0,
        "{name}: target_bits must be positive and finite"
    );
    ensure!(
        options.max_degree >= 1,
        "{name}: max_degree must be positive"
    );
    let choice = degree_for_precision(
        fun,
        a,
        b,
        parity,
        options.target_bits,
        options.max_degree,
        options.strategy,
    )
    .map_err(|e| anyhow!("{name}: {e}"))?;
    let approximation = PolynomialApproximation::from_polynomial(
        &choice.minimax.poly,
        base2k,
        coeff_meta,
        options.strategy,
        module,
    )
    .map_err(|e| anyhow!("{name}: {e}"))?;
    Ok((approximation, error_bits(choice.minimax.error)))
}

pub(crate) fn parity_for_interval<F: Float>(a: F, b: F, symmetric: Parity) -> Parity {
    if (a + b).abs() <= F::epsilon() * (b - a).abs() {
        symmetric
    } else {
        Parity::Full
    }
}
