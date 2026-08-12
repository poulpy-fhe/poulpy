//! Host-side composite minimax generation for `sign`.

use std::fmt::Debug;

use anyhow::{Result, anyhow, bail};
use num_traits::{Float, FloatConst, FromPrimitive};

use super::remez::{RemezOptions, cheb_basis, cheb_lobatto, eval_cheb, parabolic_vertex, select_alternating, solve};

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
    let nb = odd_degs.len();
    let m = nb + 1; // reference points = odd coefficients + leveled error
    let interval_bits = (hi / lo).log2().ceil().to_usize().unwrap_or(usize::MAX);
    let grid_mult = opts.grid_mult.max(interval_bits.saturating_mul(64));
    let grid_len = grid_mult
        .checked_mul(m)
        .ok_or_else(|| anyhow!("minimax_odd_const1: grid size overflow"))?
        .max(256);
    let interval_tol = (lo / hi).to_f64().unwrap_or(opts.rel_tol);
    let rel_tol = F::from_f64(opts.rel_tol.min(interval_tol)).unwrap();
    let two = F::one() + F::one();

    // Map a Chebyshev–Lobatto node y ∈ [−1, 1] to x ∈ [lo, hi], ascending.
    let map = |y: F| lo + (hi - lo) * (y + F::one()) / two;
    let mut refs: Vec<F> = (0..m).map(|i| map(-cheb_lobatto::<F>(i, m))).collect();

    let mut full = vec![F::zero(); degree + 1];
    let mut undershoot = F::zero();
    let mut overshoot = F::zero();
    for _ in 0..opts.max_iters {
        // Σ_j c_j·T_{odd_degs[j]}(x_i) + (−1)^i·E = 1.
        let mut mat: Vec<Vec<F>> = Vec::with_capacity(m);
        let mut rhs: Vec<F> = Vec::with_capacity(m);
        for (i, &xi) in refs.iter().enumerate() {
            let t = cheb_basis::<F>(xi, degree);
            let mut row: Vec<F> = odd_degs.iter().map(|&d| t[d]).collect();
            row.push(if i % 2 == 0 { F::one() } else { -F::one() });
            mat.push(row);
            rhs.push(F::one());
        }
        let sol = match solve(mat, rhs) {
            Some(s) => s,
            None => bail!("minimax_odd_const1: singular reference system"),
        };
        full.iter_mut().for_each(|c| *c = F::zero());
        for (j, &d) in odd_degs.iter().enumerate() {
            full[d] = sol[j];
        }

        // Extrema of e(x) = 1 − p(x) on a dense grid over [lo, hi].
        let xs: Vec<F> = (0..grid_len).map(|k| map(-cheb_lobatto::<F>(k, grid_len))).collect();
        let es: Vec<F> = xs.iter().map(|&x| F::one() - eval_cheb(&full, x)).collect();
        let mut extrema: Vec<(F, F)> = Vec::new();
        extrema.push((xs[0], es[0]));
        for k in 1..grid_len - 1 {
            if es[k].abs() >= es[k - 1].abs() && es[k].abs() >= es[k + 1].abs() {
                extrema.push(parabolic_vertex(xs[k - 1], es[k - 1], xs[k], es[k], xs[k + 1], es[k + 1]));
            }
        }
        extrema.push((xs[grid_len - 1], es[grid_len - 1]));

        undershoot = extrema.iter().map(|&(_, e)| e).fold(F::zero(), F::max);
        overshoot = extrema.iter().map(|&(_, e)| -e).fold(F::zero(), F::max);

        let alt = match select_alternating(extrema, m) {
            Some(a) => a,
            None => break,
        };
        let emax = alt.iter().map(|&(_, e)| e.abs()).fold(F::zero(), F::max);
        let emin = alt.iter().map(|&(_, e)| e.abs()).fold(F::infinity(), F::min);
        refs = alt.into_iter().map(|(x, _)| x).collect();
        if emax > F::zero() && (emax - emin) <= rel_tol * emax {
            break;
        }
    }
    Ok((full, undershoot, overshoot))
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
