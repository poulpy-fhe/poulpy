//! Host-side Remez fitting in the Chebyshev basis.

use std::fmt::Debug;

use anyhow::{Result, ensure};
use num_traits::{Float, FloatConst, FromPrimitive};

use crate::{api::Basis, polynomial::Polynomial};

pub use crate::api::Parity;

/// Remez options.
#[derive(Clone, Copy, Debug)]
pub struct RemezOptions {
    /// Maximum exchange iterations.
    pub max_iters: usize,
    /// Relative equioscillation tolerance.
    pub rel_tol: f64,
    /// Minimum extrema-grid density multiplier.
    pub grid_mult: usize,
}

impl Default for RemezOptions {
    fn default() -> Self {
        Self {
            max_iters: 100,
            rel_tol: 1e-3,
            grid_mult: 16,
        }
    }
}

/// A fitted minimax polynomial and its achieved sup-norm error.
pub struct Minimax<F> {
    /// Polynomial normalized over the convex hull of `intervals`.
    pub poly: Polynomial<F>,
    /// Ordered, pairwise-disjoint intervals on which the fit was performed.
    pub intervals: Vec<(F, F)>,
    /// Estimated sup-norm error over [`intervals`](Self::intervals).
    pub error: F,
    /// Exchange iterations performed.
    pub iters: usize,
    /// Whether the exchange converged within `max_iters`.
    pub converged: bool,
}

/// Best degree-`degree` minimax approximation of `f` on `[a, b]` (default options).
///
/// For `Parity::Even`/`Odd` the interval must be symmetric (`a == −b`); the
/// off-parity coefficients are forced to zero.
pub fn minimax<F, Fun>(f: Fun, a: F, b: F, degree: usize, parity: Parity) -> Result<Minimax<F>>
where
    F: Float + FloatConst + FromPrimitive + Debug,
    Fun: Fn(F) -> F,
{
    minimax_multi_interval_with(f, &[(a, b)], degree, parity, RemezOptions::default())
}

/// [`minimax`] with explicit [`RemezOptions`].
pub fn minimax_with<F, Fun>(f: Fun, a: F, b: F, degree: usize, parity: Parity, opts: RemezOptions) -> Result<Minimax<F>>
where
    F: Float + FloatConst + FromPrimitive + Debug,
    Fun: Fn(F) -> F,
{
    minimax_multi_interval_with(f, &[(a, b)], degree, parity, opts)
}

/// Best degree-`degree` minimax approximation on a union of intervals.
///
/// Intervals must be ordered and disjoint. The polynomial's input map spans
/// their convex hull, but gaps are excluded from the fit. Even and odd fits
/// require a symmetric domain and target.
pub fn minimax_multi_interval<F, Fun>(f: Fun, intervals: &[(F, F)], degree: usize, parity: Parity) -> Result<Minimax<F>>
where
    F: Float + FloatConst + FromPrimitive + Debug,
    Fun: Fn(F) -> F,
{
    minimax_multi_interval_with(f, intervals, degree, parity, RemezOptions::default())
}

/// [`minimax_multi_interval`] with explicit [`RemezOptions`].
pub fn minimax_multi_interval_with<F, Fun>(
    f: Fun,
    intervals: &[(F, F)],
    degree: usize,
    parity: Parity,
    opts: RemezOptions,
) -> Result<Minimax<F>>
where
    F: Float + FloatConst + FromPrimitive + Debug,
    Fun: Fn(F) -> F,
{
    validate_intervals(intervals)?;
    validate_options(opts)?;
    if parity != Parity::Full {
        ensure!(
            intervals_are_symmetric(intervals),
            "minimax: even/odd parity requires intervals symmetric about zero"
        );
    }

    let a = intervals[0].0;
    let b = intervals[intervals.len() - 1].1;
    let two = F::one() + F::one();
    let mid = a / two + b / two;
    let half = (b - a) / two;
    let g = |y: F| f(mid + half * y);
    let mapped: Vec<(F, F)> = intervals
        .iter()
        .map(|&(lo, hi)| ((lo - mid) / half, (hi - mid) / half))
        .collect();
    let fit_domain = match parity {
        Parity::Full => mapped.clone(),
        Parity::Even | Parity::Odd => positive_half(&mapped),
    };
    let degrees: Vec<usize> = (0..=degree)
        .filter(|k| match parity {
            Parity::Full => true,
            Parity::Even => k.is_multiple_of(2),
            Parity::Odd => !k.is_multiple_of(2),
        })
        .collect();
    let mut fit = fit_chebyshev_on_intervals(&g, &fit_domain, degree, &degrees, opts)?;
    fit.error = estimate_sup_error(&g, &fit.coeffs, &mapped, fit.grid_len);

    let poly = Polynomial::new_with_parity(Basis::Chebyshev, fit.coeffs, parity).with_interval(a, b);
    Ok(Minimax {
        poly,
        intervals: intervals.to_vec(),
        error: fit.error,
        iters: fit.iters,
        converged: fit.converged,
    })
}

fn validate_intervals<F: Float + Debug>(intervals: &[(F, F)]) -> Result<()> {
    ensure!(!intervals.is_empty(), "minimax: intervals must be non-empty");
    for (i, &(a, b)) in intervals.iter().enumerate() {
        ensure!(
            a.is_finite() && b.is_finite(),
            "minimax: intervals[{i}] endpoints must be finite"
        );
        ensure!(b > a, "minimax: intervals[{i}] is empty or reversed");
        ensure!((b - a).is_finite(), "minimax: intervals[{i}] width must be finite");
        if i > 0 {
            ensure!(
                intervals[i - 1].1 < a,
                "minimax: intervals must be ordered and pairwise disjoint (intervals[{}] and intervals[{i}])",
                i - 1
            );
        }
    }
    ensure!(
        (intervals[intervals.len() - 1].1 - intervals[0].0).is_finite(),
        "minimax: interval hull width must be finite"
    );
    Ok(())
}

fn validate_options(opts: RemezOptions) -> Result<()> {
    ensure!(opts.max_iters > 0, "minimax: max_iters must be positive");
    ensure!(
        opts.rel_tol.is_finite() && opts.rel_tol > 0.0,
        "minimax: rel_tol must be positive and finite"
    );
    ensure!(opts.grid_mult > 0, "minimax: grid_mult must be positive");
    Ok(())
}

fn intervals_are_symmetric<F: Float + FromPrimitive>(intervals: &[(F, F)]) -> bool {
    let span = intervals[intervals.len() - 1].1 - intervals[0].0;
    let tolerance = F::epsilon() * span * F::from_u8(8).unwrap();
    intervals
        .iter()
        .zip(intervals.iter().rev())
        .all(|(&(a, b), &(c, d))| (a + d).abs() <= tolerance && (b + c).abs() <= tolerance)
}

fn positive_half<F: Float>(intervals: &[(F, F)]) -> Vec<(F, F)> {
    intervals
        .iter()
        .filter_map(|&(a, b)| if b <= F::zero() { None } else { Some((a.max(F::zero()), b)) })
        .collect()
}

pub(crate) struct RemezFit<F> {
    pub(crate) coeffs: Vec<F>,
    pub(crate) error: F,
    pub(crate) iters: usize,
    pub(crate) converged: bool,
    pub(crate) grid_len: usize,
}

pub(crate) fn fit_chebyshev_on_intervals<F, G>(
    g: &G,
    intervals: &[(F, F)],
    degree: usize,
    degrees: &[usize],
    opts: RemezOptions,
) -> Result<RemezFit<F>>
where
    F: Float + FloatConst + FromPrimitive + Debug,
    G: Fn(F) -> F,
{
    ensure!(!intervals.is_empty(), "minimax: fit domain must be non-empty");
    let m = degrees.len() + 1;
    let mut grid_len = opts
        .grid_mult
        .checked_mul(m)
        .ok_or_else(|| anyhow::anyhow!("minimax: grid size overflow"))?
        .max(256);
    let rel_tol = F::from_f64(opts.rel_tol).unwrap();

    if degrees.is_empty() {
        let coeffs = vec![F::zero(); degree + 1];
        return Ok(RemezFit {
            error: estimate_sup_error(g, &coeffs, intervals, grid_len),
            coeffs,
            iters: 0,
            converged: true,
            grid_len,
        });
    }

    let mut refs = initial_references(intervals, m);
    let mut coeffs = vec![F::zero(); degree + 1];
    let mut converged = false;
    let mut iters = 0;

    for it in 0..opts.max_iters {
        iters = it + 1;
        let mut mat: Vec<Vec<F>> = Vec::with_capacity(m);
        let mut rhs: Vec<F> = Vec::with_capacity(m);
        for (i, &yi) in refs.iter().enumerate() {
            let basis = cheb_basis::<F>(yi, degree);
            let mut row: Vec<F> = degrees.iter().map(|&k| basis[k]).collect();
            row.push(if i.is_multiple_of(2) { F::one() } else { -F::one() });
            mat.push(row);
            rhs.push(g(yi));
        }
        let sol = solve(mat, rhs).ok_or_else(|| anyhow::anyhow!("minimax: singular reference system at iteration {it}"))?;
        coeffs.fill(F::zero());
        for (&k, &coefficient) in degrees.iter().zip(&sol) {
            coeffs[k] = coefficient;
        }

        let mut exchange = None;
        for _ in 0..=6 {
            let extrema = find_extrema(g, &coeffs, intervals, grid_len);
            let egrid = extrema.iter().map(|&(_, error)| error.abs()).fold(F::zero(), F::max);
            let value_scale = extrema
                .iter()
                .map(|&(x, _)| g(x).abs().max(eval_cheb(&coeffs, x).abs()))
                .fold(F::one(), F::max);
            let roundoff = F::epsilon() * F::from_usize(1024 * (degree + 1)).unwrap() * value_scale;
            if egrid <= roundoff {
                converged = true;
                break;
            }
            if let Some(alt) = select_alternating(extrema, m) {
                exchange = Some(alt);
                break;
            }
            let Some(next) = grid_len.checked_mul(2) else {
                break;
            };
            grid_len = next;
        }
        if converged {
            break;
        }
        let Some(alt) = exchange else { break };
        let emax = alt.iter().map(|&(_, e)| e.abs()).fold(F::zero(), F::max);
        let emin = alt.iter().map(|&(_, e)| e.abs()).fold(F::infinity(), F::min);
        refs = alt.into_iter().map(|(y, _)| y).collect();
        if emin > F::zero() && (emax - emin) <= rel_tol * emin {
            converged = true;
            break;
        }
    }

    Ok(RemezFit {
        error: estimate_sup_error(g, &coeffs, intervals, grid_len),
        coeffs,
        iters,
        converged,
        grid_len,
    })
}

fn initial_references<F>(intervals: &[(F, F)], m: usize) -> Vec<F>
where
    F: Float + FloatConst + FromPrimitive,
{
    let counts = allocate_references(intervals, m);
    let two = F::one() + F::one();
    intervals
        .iter()
        .zip(counts)
        .flat_map(|(&(a, b), count)| {
            let mid = a / two + b / two;
            let half = (b - a) / two;
            (0..count).map(move |i| {
                let theta = F::PI() * F::from_usize(2 * i + 1).unwrap() / F::from_usize(2 * count).unwrap();
                mid - half * theta.cos()
            })
        })
        .collect()
}

fn allocate_references<F: Float + FromPrimitive>(intervals: &[(F, F)], m: usize) -> Vec<usize> {
    let widths: Vec<F> = intervals.iter().map(|&(a, b)| b - a).collect();
    let mut counts = vec![0; intervals.len()];
    let seeded = m.min(intervals.len());

    for _ in 0..seeded {
        let next = widths
            .iter()
            .enumerate()
            .filter(|&(i, _)| counts[i] == 0)
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .unwrap()
            .0;
        counts[next] = 1;
    }

    for _ in seeded..m {
        let next = widths
            .iter()
            .enumerate()
            .max_by(|&(i, a), &(j, b)| {
                let qa = *a / F::from_usize(counts[i] + 1).unwrap();
                let qb = *b / F::from_usize(counts[j] + 1).unwrap();
                qa.partial_cmp(&qb).unwrap()
            })
            .unwrap()
            .0;
        counts[next] += 1;
    }
    counts
}

pub(crate) fn cheb_lobatto<F: Float + FloatConst + FromPrimitive>(i: usize, m: usize) -> F {
    let pi = F::PI();
    (pi * F::from_usize(i).unwrap() / F::from_usize(m - 1).unwrap()).cos()
}

pub(crate) fn cheb_basis<F: Float>(y: F, n: usize) -> Vec<F> {
    let two = F::one() + F::one();
    let mut t = Vec::with_capacity(n + 1);
    t.push(F::one());
    if n >= 1 {
        t.push(y);
    }
    for k in 2..=n {
        t.push(two * y * t[k - 1] - t[k - 2]);
    }
    t
}

pub(crate) fn eval_cheb<F: Float>(c: &[F], y: F) -> F {
    let two = F::one() + F::one();
    let mut d = F::zero();
    let mut dd = F::zero();
    for k in (1..c.len()).rev() {
        let tmp = d;
        d = two * y * d - dd + c[k];
        dd = tmp;
    }
    y * d - dd + c[0]
}

fn find_extrema<F, G>(g: &G, coeffs: &[F], intervals: &[(F, F)], grid_len: usize) -> Vec<(F, F)>
where
    F: Float + FloatConst + FromPrimitive,
    G: Fn(F) -> F,
{
    let mut out: Vec<(F, F)> = Vec::new();
    let two = F::one() + F::one();
    for &(a, b) in intervals {
        let mid = a / two + b / two;
        let half = (b - a) / two;
        let xs: Vec<F> = (0..grid_len).map(|j| mid - half * cheb_lobatto::<F>(j, grid_len)).collect();
        let es: Vec<F> = xs.iter().map(|&x| g(x) - eval_cheb(coeffs, x)).collect();

        out.push((xs[0], es[0]));
        for j in 1..grid_len - 1 {
            let (left, current, right) = (es[j - 1].abs(), es[j].abs(), es[j + 1].abs());
            if current >= left && current >= right {
                out.push(refine_extremum(g, coeffs, xs[j - 1], xs[j + 1]));
            }
        }
        out.push((xs[grid_len - 1], es[grid_len - 1]));
    }
    out
}

fn refine_extremum<F, G>(g: &G, coeffs: &[F], mut a: F, mut b: F) -> (F, F)
where
    F: Float + FromPrimitive,
    G: Fn(F) -> F,
{
    let two = F::one() + F::one();
    let ratio = (F::from_u8(5).unwrap().sqrt() - F::one()) / two;
    let mut x0 = b - ratio * (b - a);
    let mut x1 = a + ratio * (b - a);
    let mut e0 = g(x0) - eval_cheb(coeffs, x0);
    let mut e1 = g(x1) - eval_cheb(coeffs, x1);

    for _ in 0..256 {
        let scale = F::one() + a.abs().max(b.abs());
        if b - a <= F::epsilon() * F::from_u8(16).unwrap() * scale {
            break;
        }
        if e0.abs() < e1.abs() {
            a = x0;
            x0 = x1;
            e0 = e1;
            x1 = a + ratio * (b - a);
            e1 = g(x1) - eval_cheb(coeffs, x1);
        } else {
            b = x1;
            x1 = x0;
            e1 = e0;
            x0 = b - ratio * (b - a);
            e0 = g(x0) - eval_cheb(coeffs, x0);
        }
    }

    if e0.abs() >= e1.abs() { (x0, e0) } else { (x1, e1) }
}

fn estimate_sup_error<F, G>(g: &G, coeffs: &[F], intervals: &[(F, F)], grid_len: usize) -> F
where
    F: Float + FloatConst + FromPrimitive,
    G: Fn(F) -> F,
{
    find_extrema(g, coeffs, intervals, grid_len)
        .into_iter()
        .map(|(_, e)| e.abs())
        .fold(F::zero(), F::max)
}

pub(crate) fn grid_error_bounds<F, G>(g: &G, coeffs: &[F], intervals: &[(F, F)], grid_len: usize) -> (F, F)
where
    F: Float + FloatConst + FromPrimitive,
    G: Fn(F) -> F,
{
    find_extrema(g, coeffs, intervals, grid_len)
        .into_iter()
        .fold((F::zero(), F::zero()), |(positive, negative), (_, error)| {
            (positive.max(error), negative.max(-error))
        })
}

pub(crate) fn select_alternating<F: Float>(extrema: Vec<(F, F)>, m: usize) -> Option<Vec<(F, F)>> {
    let mut merged: Vec<(F, F)> = Vec::new();
    for (y, e) in extrema {
        if e == F::zero() {
            continue;
        }
        if let Some(&(_, le)) = merged.last()
            && (le >= F::zero()) == (e >= F::zero())
        {
            if e.abs() > le.abs() {
                *merged.last_mut().unwrap() = (y, e);
            }
            continue;
        }
        merged.push((y, e));
    }
    if merged.len() < m {
        return None;
    }
    while merged.len() > m {
        match merged.len() - m {
            1 => {
                if merged[0].1.abs() <= merged[merged.len() - 1].1.abs() {
                    merged.remove(0);
                } else {
                    merged.pop();
                }
            }
            2 => {
                let i = (0..merged.len())
                    .min_by(|&i, &j| {
                        let ei = merged[i].1.abs() + merged[(i + 1) % merged.len()].1.abs();
                        let ej = merged[j].1.abs() + merged[(j + 1) % merged.len()].1.abs();
                        ei.partial_cmp(&ej).unwrap()
                    })
                    .unwrap();
                if i == merged.len() - 1 {
                    merged.pop();
                    merged.remove(0);
                } else {
                    merged.drain(i..i + 2);
                }
            }
            _ => {
                let i = (0..merged.len() - 1)
                    .min_by(|&i, &j| {
                        let ei = merged[i].1.abs() + merged[i + 1].1.abs();
                        let ej = merged[j].1.abs() + merged[j + 1].1.abs();
                        ei.partial_cmp(&ej).unwrap()
                    })
                    .unwrap();
                if i == 0 {
                    merged.remove(0);
                } else if i == merged.len() - 2 {
                    merged.pop();
                } else {
                    merged.drain(i..i + 2);
                }
            }
        }
    }
    Some(merged)
}

#[allow(clippy::needless_range_loop)]
pub(crate) fn solve<F: Float>(mut a: Vec<Vec<F>>, mut b: Vec<F>) -> Option<Vec<F>> {
    let n = b.len();
    for col in 0..n {
        let mut piv = col;
        let mut best = a[col][col].abs();
        for r in col + 1..n {
            if a[r][col].abs() > best {
                best = a[r][col].abs();
                piv = r;
            }
        }
        if best == F::zero() {
            return None;
        }
        a.swap(col, piv);
        b.swap(col, piv);
        let d = a[col][col];
        for r in col + 1..n {
            let factor = a[r][col] / d;
            if factor != F::zero() {
                for c in col..n {
                    let v = a[col][c];
                    a[r][c] = a[r][c] - factor * v;
                }
                b[r] = b[r] - factor * b[col];
            }
        }
    }
    let mut x = vec![F::zero(); n];
    for i in (0..n).rev() {
        let mut s = b[i];
        for c in i + 1..n {
            s = s - a[i][c] * x[c];
        }
        x[i] = s / a[i][i];
    }
    Some(x)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sup_error(poly: &Polynomial<f64>, f: impl Fn(f64) -> f64, n: usize) -> f64 {
        let (a, b) = (poly.a, poly.b);
        (0..=n)
            .map(|i| {
                let x = a + (b - a) * (i as f64) / (n as f64);
                (poly.evaluate_on_interval(x) - f(x)).abs()
            })
            .fold(0.0, f64::max)
    }

    fn sup_error_on_intervals(poly: &Polynomial<f64>, intervals: &[(f64, f64)], f: impl Fn(f64) -> f64, n: usize) -> f64 {
        intervals
            .iter()
            .flat_map(|&(a, b)| {
                let f = &f;
                (0..=n).map(move |i| {
                    let x = a + (b - a) * (i as f64) / (n as f64);
                    (poly.evaluate_on_interval(x) - f(x)).abs()
                })
            })
            .fold(0.0, f64::max)
    }

    #[test]
    fn recovers_low_degree_polynomial() {
        let r = minimax(|x: f64| x * x * x, -1.0, 1.0, 3, Parity::Odd).unwrap();
        assert!(r.error < 1e-9, "recovery error {} too large", r.error);
        assert!((r.poly.coeffs[1] - 0.75).abs() < 1e-9);
        assert!((r.poly.coeffs[3] - 0.25).abs() < 1e-9);
        assert!(r.poly.coeffs[0].abs() < 1e-9 && r.poly.coeffs[2].abs() < 1e-9);
    }

    #[test]
    fn minimax_beats_or_matches_interpolation() {
        let f = |x: f64| x.exp();
        let deg = 8;
        let mm = minimax(f, -1.0, 1.0, deg, Parity::Full).unwrap();
        let interp = Polynomial::chebyshev_interpolate(deg, -1.0, 1.0, f).unwrap();
        let e_mm = sup_error(&mm.poly, f, 4000);
        let e_in = sup_error(&interp, f, 4000);
        assert!(mm.converged, "exp minimax did not converge");
        assert!(e_mm <= e_in * 1.02, "minimax {e_mm:e} worse than interp {e_in:e}");
        assert!(e_mm < 1e-6, "exp deg-8 minimax error {e_mm:e} unexpectedly large");
    }

    #[test]
    fn equioscillates() {
        let f = |x: f64| (2.0 * x).sin();
        let mm = minimax(f, -1.0, 1.0, 7, Parity::Odd).unwrap();
        let e = sup_error(&mm.poly, f, 8000);
        assert!(mm.converged);
        assert!((e - mm.error).abs() <= 1e-3 * e, "reported {} vs measured {e:e}", mm.error);
    }

    #[test]
    fn odd_target_has_zero_even_coeffs() {
        let mm = minimax(|x: f64| x.tanh(), -1.0, 1.0, 9, Parity::Odd).unwrap();
        for (k, &c) in mm.poly.coeffs.iter().enumerate() {
            if k % 2 == 0 {
                assert_eq!(c, 0.0, "even coeff {k} not zeroed");
            }
        }
        assert!(mm.error < 1e-3);
    }

    #[test]
    fn fits_on_shifted_interval() {
        let f = |x: f64| x.exp();
        let mm = minimax(f, 0.0, 2.0, 10, Parity::Full).unwrap();
        assert!(mm.converged);
        assert!(sup_error(&mm.poly, f, 4000) < 1e-6);
    }

    #[test]
    fn fits_a_true_union_without_approximating_the_gap() {
        let intervals = [(-1.0, -0.2), (0.2, 1.0)];
        let sign = |x: f64| if x < 0.0 { -1.0 } else { 1.0 };
        let mm = minimax_multi_interval(sign, &intervals, 15, Parity::Odd).unwrap();
        let measured = sup_error_on_intervals(&mm.poly, &intervals, sign, 8000);

        assert!(mm.converged, "multi-interval sign fit did not converge");
        assert_eq!(mm.intervals, intervals);
        assert_eq!(mm.poly.interval(), (-1.0, 1.0));
        assert!(measured < 0.025, "degree-15 sign error {measured:e} unexpectedly large");
        assert!(
            (measured - mm.error).abs() <= 2e-3 * measured,
            "reported {} vs measured {measured:e}",
            mm.error
        );
    }

    #[test]
    fn multi_interval_validates_domain_invariants() {
        let f = |x: f64| x;
        assert!(minimax_multi_interval(f, &[], 3, Parity::Full).is_err());
        assert!(minimax_multi_interval(f, &[(0.0, 1.0), (0.5, 2.0)], 3, Parity::Full).is_err());
        assert!(minimax_multi_interval(f, &[(1.0, 2.0), (-2.0, -1.0)], 3, Parity::Full).is_err());
        assert!(minimax_multi_interval(f, &[(-1.0, -0.2), (0.3, 1.0)], 3, Parity::Odd).is_err());
        assert!(minimax_multi_interval(f, &[(-f64::MAX, f64::MAX)], 3, Parity::Full).is_err());
    }

    #[test]
    fn multi_interval_hull_map_handles_shifted_domains() {
        let intervals = [(0.0, 0.5), (1.5, 2.0), (3.0, 4.0)];
        let f = |x: f64| x * x * x - 2.0 * x + 1.0;
        let mm = minimax_multi_interval(f, &intervals, 3, Parity::Full).unwrap();
        assert_eq!(mm.poly.interval(), (0.0, 4.0));
        let error = sup_error_on_intervals(&mm.poly, &intervals, f, 1000);
        assert!(error < 1e-11, "shifted-domain recovery error {error:e}");
    }

    #[test]
    fn initial_references_cover_unbalanced_intervals() {
        let intervals = [(-1.0, -0.99), (-0.4, 0.4), (0.999, 1.0)];
        let refs = initial_references(&intervals, 8);
        assert_eq!(refs.len(), 8);
        assert!(intervals.iter().all(|&(a, b)| refs.iter().any(|&x| a < x && x < b)));
    }

    #[test]
    fn exchange_removes_a_weak_interior_pair() {
        let extrema = vec![(0.0, 10.0), (1.0, -1.0), (2.0, 1.0), (3.0, -10.0), (4.0, 10.0), (5.0, -10.0)];
        let selected = select_alternating(extrema, 4).unwrap();
        assert_eq!(selected, vec![(0.0, 10.0), (3.0, -10.0), (4.0, 10.0), (5.0, -10.0)]);
    }

    #[test]
    fn equioscillates_across_unbalanced_intervals() {
        let intervals = [(-1.0, -0.9), (-0.2, -0.1), (0.1, 0.2), (0.9, 1.0)];
        let f = |x: f64| 1.0 / (1.0 + 25.0 * x * x);
        let mm = minimax_multi_interval(f, &intervals, 16, Parity::Even).unwrap();
        assert!(mm.converged, "unbalanced fit did not converge in {} iterations", mm.iters);

        let positive = [(0.1, 0.2), (0.9, 1.0)];
        let extrema = find_extrema(&f, &mm.poly.coeffs, &positive, 4096);
        let selected = select_alternating(extrema, 10).unwrap();
        let emax = selected.iter().map(|(_, e)| e.abs()).fold(0.0, f64::max);
        let emin = selected.iter().map(|(_, e)| e.abs()).fold(f64::INFINITY, f64::min);
        assert!((emax - emin) <= 2e-3 * emin, "error spread [{emin:e}, {emax:e}]");
        assert!(positive.iter().all(|&(a, b)| selected.iter().any(|&(x, _)| a <= x && x <= b)));
    }
}
