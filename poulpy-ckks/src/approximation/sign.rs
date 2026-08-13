//! Host-side composite minimax generation for `sign`.

use std::fmt::Debug;

use anyhow::{Result, anyhow, bail};
use num_traits::{Float, FloatConst, FromPrimitive};

#[cfg(test)]
use super::remez::eval_cheb;
use super::remez::{RemezOptions, fit_chebyshev_on_intervals, grid_error_bounds};

/// Fits an odd polynomial to `1` on positive `[lo, hi]`.
pub(crate) fn minimax_odd_const1<F>(lo: F, hi: F, degree: usize, opts: RemezOptions) -> Result<(Vec<F>, F, F)>
where
    F: Float + FloatConst + FromPrimitive + Debug,
{
    if degree.is_multiple_of(2) {
        bail!("minimax_odd_const1: degree {degree} must be odd");
    }
    if lo >= hi || lo <= F::zero() {
        bail!("minimax_odd_const1: require 0 < lo < hi");
    }

    let odd_degs: Vec<usize> = (0..).map(|j| 2 * j + 1).take_while(|&d| d <= degree).collect();
    let interval_bits = (hi / lo).log2().ceil().to_usize().unwrap_or(usize::MAX);
    let interval_tol = (lo / hi).to_f64().unwrap_or(opts.rel_tol);
    let fit_opts = RemezOptions {
        grid_mult: opts.grid_mult.max(interval_bits.saturating_mul(64)),
        rel_tol: opts.rel_tol.min(interval_tol),
        ..opts
    };
    let target = |_: F| F::one();
    let domain = [(lo, hi)];
    let fit = fit_chebyshev_on_intervals(&target, &domain, degree, &odd_degs, fit_opts)
        .map_err(|error| anyhow!("minimax_odd_const1: {error}"))?;
    let (undershoot, overshoot) = grid_error_bounds(&target, &fit.coeffs, &domain, fit.grid_len);
    Ok((fit.coeffs, undershoot, overshoot))
}

/// Builds normalized odd factors without an evaluation-error margin.
pub fn sign_composite_coeffs<F>(
    tau: F,
    target_bits: f64,
    degrees: &[usize],
    max_factors: usize,
    opts: RemezOptions,
) -> Result<Vec<Vec<F>>>
where
    F: Float + FloatConst + FromPrimitive + Debug,
{
    sign_composite_coeffs_with_margin(tau, F::zero(), target_bits, degrees, max_factors, opts)
}

/// Builds normalized odd factors, propagating `error_margin` between factors.
pub fn sign_composite_coeffs_with_margin<F>(
    tau: F,
    error_margin: F,
    target_bits: f64,
    degrees: &[usize],
    max_factors: usize,
    opts: RemezOptions,
) -> Result<Vec<Vec<F>>>
where
    F: Float + FloatConst + FromPrimitive + Debug,
{
    if !tau.is_finite() || tau <= F::zero() || tau >= F::one() {
        bail!("sign_composite_coeffs_with_margin: tau must lie in (0, 1)");
    }
    if !error_margin.is_finite() || error_margin < F::zero() {
        bail!("sign_composite_coeffs_with_margin: error_margin must be finite and non-negative");
    }
    if !target_bits.is_finite() || target_bits <= 0.0 {
        bail!("sign_composite_coeffs_with_margin: target_bits must be positive and finite");
    }
    if degrees.is_empty() {
        bail!("sign_composite_coeffs_with_margin: degrees must be non-empty");
    }
    for (i, &degree) in degrees.iter().enumerate() {
        if degree == 0 || degree.is_multiple_of(2) {
            bail!("sign_composite_coeffs_with_margin: degrees[{i}] must be positive and odd, got {degree}");
        }
    }
    if max_factors == 0 {
        bail!("sign_composite_coeffs_with_margin: max_factors must be positive");
    }
    if opts.max_iters == 0 {
        bail!("sign_composite_coeffs_with_margin: max_iters must be positive");
    }
    if !opts.rel_tol.is_finite() || opts.rel_tol <= 0.0 {
        bail!("sign_composite_coeffs_with_margin: rel_tol must be positive and finite");
    }
    if opts.grid_mult == 0 {
        bail!("sign_composite_coeffs_with_margin: grid_mult must be positive");
    }
    let target = F::from_f64(2f64.powf(-target_bits))
        .ok_or_else(|| anyhow!("sign_composite_coeffs_with_margin: target_bits {target_bits} not representable"))?;
    let one = F::one();
    let mut lo = tau;
    let mut hi = one + error_margin;
    if !hi.is_finite() {
        bail!("sign_composite_coeffs_with_margin: error_margin overflows the input interval");
    }
    let mut rows: Vec<Vec<F>> = Vec::new();
    for i in 0..max_factors {
        let deg = degrees[i.min(degrees.len() - 1)];
        let (coeffs, undershoot, overshoot) =
            minimax_odd_const1(lo, hi, deg, opts).map_err(|e| anyhow!("sign_composite_coeffs_with_margin: factor {i}: {e}"))?;
        let error = undershoot.max(overshoot);
        if error <= target {
            rows.push(coeffs);
            return Ok(rows);
        }
        let next_lo = one - undershoot - error_margin;
        let next_hi = one + overshoot + error_margin;
        if next_lo <= F::zero() {
            bail!("sign_composite_coeffs_with_margin: factor {i} (degree {deg}) has a non-positive image bound");
        }
        let mut coeffs = coeffs;
        for coeff in &mut coeffs {
            *coeff = *coeff / next_hi;
        }
        rows.push(coeffs);
        lo = next_lo / next_hi;
        hi = one;
    }
    bail!("sign_composite_coeffs_with_margin: {target_bits} bits not reached in {max_factors} factors");
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Composes the generated rows (Chebyshev, applied `rows[0]` first) at `x`.
    fn compose(rows: &[Vec<f64>], mut x: f64) -> f64 {
        for r in rows {
            x = eval_cheb(r, x);
        }
        x
    }

    fn composite_bits(rows: &[Vec<f64>], tau: f64) -> f64 {
        let mut worst = 0.0f64;
        for k in 0..4000 {
            let x = tau + (1.0 - tau) * (k as f64) / 3999.0;
            worst = worst.max((compose(rows, x) - 1.0).abs());
        }
        -worst.log2()
    }

    #[test]
    fn single_factor_is_odd_and_fits() {
        let (c, undershoot, overshoot) = minimax_odd_const1(0.2_f64, 1.0, 15, RemezOptions::default()).unwrap();
        assert_eq!(c.len(), 16);
        for (k, &ck) in c.iter().enumerate() {
            if k % 2 == 0 {
                assert_eq!(ck, 0.0, "even coeff {k} must be zero");
            }
        }
        let error = undershoot.max(overshoot);
        assert!(error < 0.5, "degree-15 error {error} on [0.2,1] unexpectedly large");
    }

    #[test]
    fn composite_reaches_target() {
        let tau = 0.1;
        let target = 20.0;
        let rows = sign_composite_coeffs::<f64>(tau, target, &[15], 12, RemezOptions::default()).unwrap();
        let bits = composite_bits(&rows, tau);
        assert!(
            bits >= target,
            "composite reached {bits:.1} bits < {target} ({} factors)",
            rows.len()
        );
    }

    #[test]
    fn zero_margin_matches_wrapper() {
        let plain = sign_composite_coeffs(0.1_f64, 15.0, &[15], 8, RemezOptions::default()).unwrap();
        let explicit = sign_composite_coeffs_with_margin(0.1_f64, 0.0, 15.0, &[15], 8, RemezOptions::default()).unwrap();
        assert_eq!(plain, explicit);
    }

    #[test]
    fn more_factors_for_finer_tau() {
        let coarse = sign_composite_coeffs::<f64>(0.2, 15.0, &[15], 12, RemezOptions::default()).unwrap();
        let fine = sign_composite_coeffs::<f64>(0.02, 15.0, &[15], 12, RemezOptions::default()).unwrap();
        assert!(fine.len() >= coarse.len(), "finer tau should need at least as many factors");
    }

    #[test]
    fn resolves_small_gap_with_margin() {
        let rows = sign_composite_coeffs_with_margin::<crate::Quad>(
            crate::Quad::new(2.0f128.powi(-30)),
            crate::Quad::new(2.0f128.powi(-35)),
            20.0,
            &[15, 15, 15, 17, 31, 31, 31, 31],
            8,
            RemezOptions::default(),
        )
        .unwrap();
        assert!(!rows.is_empty());
    }

    #[test]
    fn rejects_negative_error_margin() {
        assert!(sign_composite_coeffs_with_margin(0.1_f64, -1e-6, 10.0, &[15], 2, RemezOptions::default()).is_err());
    }
}
