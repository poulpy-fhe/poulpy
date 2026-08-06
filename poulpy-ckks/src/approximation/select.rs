//! Degree and precision selection for minimax polynomials.

use std::fmt::Debug;

use anyhow::{Result, anyhow, bail, ensure};
use num_traits::{Float, FloatConst, FromPrimitive, ToPrimitive};

use poulpy_core::layouts::{SplitStrategy, bsgs_eval_depth};

use super::remez::{Minimax, Parity, minimax};

/// `−log2(error)` bits of precision (`+∞` for a zero error).
pub fn error_bits<F: Float + ToPrimitive>(error: F) -> f64 {
    let e = error.to_f64().unwrap_or(f64::INFINITY);
    if e <= 0.0 { f64::INFINITY } else { -e.log2() }
}

/// A chosen degree with its fitted polynomial and homomorphic evaluation depth.
pub struct DegreeChoice<F> {
    /// Polynomial degree.
    pub degree: usize,
    /// The fitted minimax polynomial and its sup-norm error.
    pub minimax: Minimax<F>,
    /// Multiplicative depth to evaluate it via BSGS at the chosen strategy.
    pub depth: usize,
}

impl<F: Float + ToPrimitive> DegreeChoice<F> {
    /// Achieved precision in bits (`−log2(error)`).
    pub fn bits(&self) -> f64 {
        error_bits(self.minimax.error)
    }
}

fn start_step(parity: Parity) -> (usize, usize) {
    match parity {
        Parity::Full => (0, 1),
        Parity::Even => (0, 2),
        Parity::Odd => (1, 2),
    }
}

/// Smallest degree whose minimax error over `[a, b]` is `≤ 2^{−target_bits}`.
///
/// Steps by 1 (full) or 2 (even/odd). Errors if `max_degree` is reached first.
pub fn degree_for_precision<F, Fun>(
    f: Fun,
    a: F,
    b: F,
    parity: Parity,
    target_bits: f64,
    max_degree: usize,
    strategy: SplitStrategy,
) -> Result<DegreeChoice<F>>
where
    F: Float + FloatConst + FromPrimitive + ToPrimitive + Debug,
    Fun: Fn(F) -> F + Copy,
{
    ensure!(
        target_bits.is_finite() && target_bits > 0.0,
        "degree_for_precision: target_bits must be positive and finite"
    );
    let threshold = F::from_f64(2f64.powf(-target_bits))
        .ok_or_else(|| anyhow!("degree_for_precision: target_bits {target_bits} not representable"))?;
    let (start, step) = start_step(parity);
    ensure!(
        max_degree >= start,
        "degree_for_precision: max_degree {max_degree} is below the first {parity:?} degree {start}"
    );
    let mut degree = start;
    let mut best_bits = f64::NEG_INFINITY;
    while degree <= max_degree {
        let mm = minimax(f, a, b, degree, parity).map_err(|e| anyhow!("degree_for_precision: {e}"))?;
        best_bits = best_bits.max(error_bits(mm.error));
        if mm.error <= threshold {
            let depth = bsgs_eval_depth(degree, strategy);
            return Ok(DegreeChoice {
                degree,
                minimax: mm,
                depth,
            });
        }
        degree += step;
    }
    bail!("degree_for_precision: {target_bits:.1} bits not reached by degree {max_degree} (best {best_bits:.1} bits)");
}

/// Best fit within the degree and BSGS-depth limits.
pub fn precision_at_depth<F, Fun>(
    f: Fun,
    a: F,
    b: F,
    parity: Parity,
    max_depth: usize,
    max_degree: usize,
    strategy: SplitStrategy,
) -> Result<DegreeChoice<F>>
where
    F: Float + FloatConst + FromPrimitive + ToPrimitive + Debug,
    Fun: Fn(F) -> F + Copy,
{
    let (start, step) = start_step(parity);
    ensure!(
        max_degree >= start,
        "precision_at_depth: max_degree {max_degree} is below the first {parity:?} degree {start}"
    );
    let mut best: Option<usize> = None;
    let mut degree = start;
    // bsgs_eval_depth is monotone non-decreasing in degree, so stop at the first overflow.
    while degree <= max_degree {
        if bsgs_eval_depth(degree, strategy) <= max_depth {
            best = Some(degree);
            degree += step;
        } else {
            break;
        }
    }
    let degree = best.ok_or_else(|| anyhow!("precision_at_depth: no degree fits within depth {max_depth}"))?;
    let mm = minimax(f, a, b, degree, parity).map_err(|e| anyhow!("precision_at_depth: {e}"))?;
    let depth = bsgs_eval_depth(degree, strategy);
    Ok(DegreeChoice {
        degree,
        minimax: mm,
        depth,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn error_bits_basic() {
        assert!((error_bits(0.25_f64) - 2.0).abs() < 1e-12);
        assert_eq!(error_bits(0.0_f64), f64::INFINITY);
    }

    #[test]
    fn degree_grows_with_target() {
        let f = |x: f64| x.exp();
        let lo = degree_for_precision(f, -1.0, 1.0, Parity::Full, 15.0, 40, SplitStrategy::MinMult).unwrap();
        let hi = degree_for_precision(f, -1.0, 1.0, Parity::Full, 30.0, 40, SplitStrategy::MinMult).unwrap();
        assert!(lo.bits() >= 15.0, "achieved {} < 15", lo.bits());
        assert!(hi.bits() >= 30.0, "achieved {} < 30", hi.bits());
        assert!(hi.degree >= lo.degree, "higher target needs at least as high a degree");
    }

    #[test]
    fn odd_target_selects_odd_degree() {
        let dc = degree_for_precision(|x: f64| x.tanh(), -1.0, 1.0, Parity::Odd, 12.0, 40, SplitStrategy::MinMult).unwrap();
        assert_eq!(dc.degree % 2, 1, "odd parity must pick an odd degree");
        assert!(dc.bits() >= 12.0);
    }

    #[test]
    fn depth_budget_bounds_degree_and_precision() {
        let f = |x: f64| x.exp();
        let budget = 4usize;
        let dc = precision_at_depth(f, -1.0, 1.0, Parity::Full, budget, 64, SplitStrategy::MinDepth).unwrap();
        assert!(dc.depth <= budget, "chosen depth {} exceeds budget {budget}", dc.depth);
        assert!(dc.bits() > 0.0);
        // A deeper budget admits at least as high a degree (precision may already
        // be saturated at f64 machine precision, so compare degree, not bits).
        let deeper = precision_at_depth(f, -1.0, 1.0, Parity::Full, budget + 2, 64, SplitStrategy::MinDepth).unwrap();
        assert!(deeper.depth <= budget + 2);
        assert!(deeper.degree >= dc.degree, "deeper budget picked lower degree");
    }
}
