//! Degree and precision selection for minimax polynomials.

use std::fmt::Debug;

use anyhow::{Result, anyhow, bail, ensure};
use num_traits::{Float, FloatConst, FromPrimitive, ToPrimitive};

use poulpy_core::layouts::{SplitStrategy, bsgs_eval_depth};

use super::remez::{Minimax, Parity, RemezOptions, minimax_with};

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
        // BSGS plans require degree >= 1.
        Parity::Full => (1, 1),
        Parity::Even => (2, 2),
        Parity::Odd => (1, 2),
    }
}

/// Smallest BSGS-evaluable degree whose minimax error over `[a, b]` is
/// `≤ 2^{−target_bits}`.
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
    degree_for_precision_with(f, a, b, parity, target_bits, max_degree, strategy, RemezOptions::default())
}

/// [`degree_for_precision`] with explicit [`RemezOptions`].
#[allow(clippy::too_many_arguments)]
pub fn degree_for_precision_with<F, Fun>(
    f: Fun,
    a: F,
    b: F,
    parity: Parity,
    target_bits: f64,
    max_degree: usize,
    strategy: SplitStrategy,
    opts: RemezOptions,
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
    let mut finite_fits = 0usize;
    let mut last_error = None;
    while degree <= max_degree {
        let mm = match minimax_with(f, a, b, degree, parity, opts) {
            Ok(mm) => mm,
            Err(error) => {
                last_error = Some(format!("degree {degree}: {error}"));
                let Some(next) = degree.checked_add(step) else {
                    break;
                };
                degree = next;
                continue;
            }
        };
        if !mm.error.is_finite() {
            last_error = Some(format!("degree {degree}: non-finite approximation error"));
        } else {
            finite_fits += 1;
            best_bits = best_bits.max(error_bits(mm.error));
        }
        if mm.error.is_finite() && mm.error <= threshold {
            let depth = bsgs_eval_depth(degree, strategy);
            return Ok(DegreeChoice {
                degree,
                minimax: mm,
                depth,
            });
        }
        let Some(next) = degree.checked_add(step) else {
            break;
        };
        degree = next;
    }
    if finite_fits == 0 {
        bail!(
            "degree_for_precision: no degree produced a finite fit{}",
            last_error.map_or_else(String::new, |error| format!(" ({error})"))
        );
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
    precision_at_depth_with(f, a, b, parity, max_depth, max_degree, strategy, RemezOptions::default())
}

/// [`precision_at_depth`] with explicit [`RemezOptions`].
#[allow(clippy::too_many_arguments)]
pub fn precision_at_depth_with<F, Fun>(
    f: Fun,
    a: F,
    b: F,
    parity: Parity,
    max_depth: usize,
    max_degree: usize,
    strategy: SplitStrategy,
    opts: RemezOptions,
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
    let mut best: Option<DegreeChoice<F>> = None;
    let mut last_error = None;
    let mut degree = start;
    // Finite-precision fit errors need not improve with degree.
    while degree <= max_degree {
        let depth = bsgs_eval_depth(degree, strategy);
        if depth > max_depth {
            break;
        }
        match minimax_with(f, a, b, degree, parity, opts) {
            Ok(mm) if mm.error.is_finite() => {
                let replace = best.as_ref().is_none_or(|current| {
                    mm.error < current.minimax.error
                        || (mm.error == current.minimax.error && mm.converged && !current.minimax.converged)
                });
                if replace {
                    best = Some(DegreeChoice {
                        degree,
                        minimax: mm,
                        depth,
                    });
                }
            }
            Ok(_) => {
                last_error = Some(format!("degree {degree}: non-finite approximation error"));
            }
            Err(error) => {
                last_error = Some(format!("degree {degree}: {error}"));
            }
        }
        let Some(next) = degree.checked_add(step) else {
            break;
        };
        degree = next;
    }
    best.ok_or_else(|| {
        anyhow!(
            "precision_at_depth: no degree produced a finite fit within depth {max_depth}{}",
            last_error.map_or_else(String::new, |error| format!(" ({error})"))
        )
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
        // A deeper budget cannot make the best measured precision worse.
        let deeper = precision_at_depth(f, -1.0, 1.0, Parity::Full, budget + 2, 64, SplitStrategy::MinDepth).unwrap();
        assert!(deeper.depth <= budget + 2);
        assert!(deeper.minimax.error <= dc.minimax.error, "deeper budget picked a worse fit");
    }

    #[test]
    fn depth_selection_skips_unstable_highest_degree() {
        let dc = precision_at_depth(|x: f64| x.exp(), -1.0, 1.0, Parity::Full, 7, 64, SplitStrategy::MinDepth).unwrap();
        assert!(
            dc.minimax.error < 1e-10,
            "selected degree {} with error {}",
            dc.degree,
            dc.minimax.error
        );
    }

    #[test]
    fn selected_constant_is_bsgs_evaluable() {
        let full = degree_for_precision(|_: f64| 1.0, -1.0, 1.0, Parity::Full, 10.0, 4, SplitStrategy::MinDepth).unwrap();
        let even = degree_for_precision(|_: f64| 1.0, -1.0, 1.0, Parity::Even, 10.0, 4, SplitStrategy::MinDepth).unwrap();
        assert_eq!(full.degree, 1);
        assert_eq!(even.degree, 2);
    }
}
