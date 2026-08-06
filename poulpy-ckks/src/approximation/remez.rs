//! Host-side Remez fitting in the Chebyshev basis.

use std::fmt::Debug;

use anyhow::{Result, bail, ensure};
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
    /// Extrema grid density multiplier.
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
    /// Chebyshev-basis polynomial over the requested interval `[a, b]`.
    pub poly: Polynomial<F>,
    /// Estimated sup-norm error `max_{x∈[a,b]} |f(x) − poly(x)|`.
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
    minimax_with(f, a, b, degree, parity, RemezOptions::default())
}

/// [`minimax`] with explicit [`RemezOptions`].
pub fn minimax_with<F, Fun>(f: Fun, a: F, b: F, degree: usize, parity: Parity, opts: RemezOptions) -> Result<Minimax<F>>
where
    F: Float + FloatConst + FromPrimitive + Debug,
    Fun: Fn(F) -> F,
{
    ensure!(a.is_finite() && b.is_finite(), "minimax: interval endpoints must be finite");
    ensure!(b > a, "minimax: empty interval [a, b]");
    ensure!(opts.max_iters > 0, "minimax: max_iters must be positive");
    ensure!(
        opts.rel_tol.is_finite() && opts.rel_tol > 0.0,
        "minimax: rel_tol must be positive and finite"
    );
    ensure!(opts.grid_mult > 0, "minimax: grid_mult must be positive");
    if parity != Parity::Full {
        ensure!(
            (a + b).abs() <= F::epsilon() * (b - a),
            "minimax: even/odd parity requires a symmetric interval a == -b"
        );
    }

    let two = F::one() + F::one();
    let mid = (a + b) / two;
    let half = (b - a) / two;
    // Approximate g(y) = f(x(y)) for y ∈ [−1, 1], x = mid + half·y; the returned
    // Chebyshev coefficients are expressed in y, exactly what poulpy evaluates.
    let g = |y: F| f(mid + half * y);

    let n = degree;
    let m = n + 2; // reference points = coefficients (n+1) + leveled error
    let grid_len = opts
        .grid_mult
        .checked_mul(m)
        .ok_or_else(|| anyhow::anyhow!("minimax: grid size overflow"))?
        .max(256);
    let rel_tol = F::from_f64(opts.rel_tol).unwrap();

    // Initial references: Chebyshev–Lobatto nodes, ascending in [−1, 1].
    let mut refs: Vec<F> = (0..m).map(|i| -cheb_lobatto::<F>(i, m)).collect();

    let mut coeffs = vec![F::zero(); n + 1];
    let mut error = F::zero();
    let mut converged = false;
    let mut iters = 0;

    for it in 0..opts.max_iters {
        iters = it + 1;

        // Solve p(y_i) + (−1)^i·E = g(y_i) for the n+1 coefficients and E.
        let mut mat: Vec<Vec<F>> = Vec::with_capacity(m);
        let mut rhs: Vec<F> = Vec::with_capacity(m);
        for (i, &yi) in refs.iter().enumerate() {
            let mut row = cheb_basis::<F>(yi, n);
            row.push(if i % 2 == 0 { F::one() } else { -F::one() });
            mat.push(row);
            rhs.push(g(yi));
        }
        let sol = match solve(mat, rhs) {
            Some(s) => s,
            None => bail!("minimax: singular reference system at iteration {it}"),
        };
        coeffs.copy_from_slice(&sol[..=n]);

        // Extrema of e(y) = g(y) − p(y) on a dense Chebyshev–Lobatto grid.
        let extrema = find_extrema(&g, &coeffs, grid_len);
        let alt = match select_alternating(extrema, m) {
            Some(a) => a,
            None => {
                // Grid too coarse to resolve n+2 alternating extrema: report best so far.
                error = grid_sup_error(&g, &coeffs, grid_len);
                break;
            }
        };

        // Stop on equioscillation: all n+2 alternating extrema equal in magnitude.
        let emax = alt.iter().map(|&(_, e)| e.abs()).fold(F::zero(), F::max);
        let emin = alt.iter().map(|&(_, e)| e.abs()).fold(F::infinity(), F::min);
        error = emax;
        refs = alt.into_iter().map(|(y, _)| y).collect();
        if emax > F::zero() && (emax - emin) <= rel_tol * emax {
            converged = true;
            break;
        }
    }

    if parity != Parity::Full {
        let keep_even = parity == Parity::Even;
        for (k, c) in coeffs.iter_mut().enumerate() {
            if (k % 2 == 0) != keep_even {
                *c = F::zero();
            }
        }
        error = grid_sup_error(&g, &coeffs, grid_len);
    }

    let poly = Polynomial::new_with_parity(Basis::Chebyshev, coeffs, parity).with_interval(a, b);
    Ok(Minimax {
        poly,
        error,
        iters,
        converged,
    })
}

/// `cos(π·i/(m−1))` — the `i`-th Chebyshev–Lobatto node magnitude for `m` nodes.
pub(crate) fn cheb_lobatto<F: Float + FloatConst + FromPrimitive>(i: usize, m: usize) -> F {
    let pi = F::PI();
    (pi * F::from_usize(i).unwrap() / F::from_usize(m - 1).unwrap()).cos()
}

/// `[T_0(y), …, T_n(y)]`.
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

/// Clenshaw evaluation of `Σ c_k T_k(y)`.
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

/// Signed error `e(y) = g(y) − p(y)` at the local extrema of `|e|` on a dense
/// Chebyshev–Lobatto grid, each refined by parabolic interpolation.
fn find_extrema<F, G>(g: &G, coeffs: &[F], grid_len: usize) -> Vec<(F, F)>
where
    F: Float + FloatConst + FromPrimitive,
    G: Fn(F) -> F,
{
    let ys: Vec<F> = (0..grid_len).map(|j| -cheb_lobatto::<F>(j, grid_len)).collect();
    let es: Vec<F> = ys.iter().map(|&y| g(y) - eval_cheb(coeffs, y)).collect();

    let mut out: Vec<(F, F)> = Vec::new();
    out.push((ys[0], es[0]));
    for j in 1..grid_len - 1 {
        let (a, b, c) = (es[j - 1].abs(), es[j].abs(), es[j + 1].abs());
        if b >= a && b >= c {
            let (yr, er) = parabolic_vertex(ys[j - 1], es[j - 1], ys[j], es[j], ys[j + 1], es[j + 1]);
            out.push((yr, er));
        }
    }
    out.push((ys[grid_len - 1], es[grid_len - 1]));
    out
}

/// Sup-norm error estimate over the dense grid.
fn grid_sup_error<F, G>(g: &G, coeffs: &[F], grid_len: usize) -> F
where
    F: Float + FloatConst + FromPrimitive,
    G: Fn(F) -> F,
{
    (0..grid_len)
        .map(|j| {
            let y = -cheb_lobatto::<F>(j, grid_len);
            (g(y) - eval_cheb(coeffs, y)).abs()
        })
        .fold(F::zero(), F::max)
}

/// Vertex `(y*, e*)` of the parabola through the three points; falls back to the
/// middle point if the fit is degenerate or the vertex leaves `[y0, y2]`.
pub(crate) fn parabolic_vertex<F: Float>(y0: F, e0: F, y1: F, e1: F, y2: F, e2: F) -> (F, F) {
    let half = (F::one() + F::one()).recip();
    let d1 = y1 - y0;
    let d2 = y1 - y2;
    let num = d1 * d1 * (e1 - e2) - d2 * d2 * (e1 - e0);
    let den = d1 * (e1 - e2) - d2 * (e1 - e0);
    if den == F::zero() {
        return (y1, e1);
    }
    let ystar = y1 - half * num / den;
    if ystar <= y0 || ystar >= y2 {
        return (y1, e1);
    }
    (ystar, lagrange3(y0, e0, y1, e1, y2, e2, ystar))
}

/// Quadratic Lagrange interpolation of three points at `x`.
pub(crate) fn lagrange3<F: Float>(x0: F, y0: F, x1: F, y1: F, x2: F, y2: F, x: F) -> F {
    let l0 = (x - x1) * (x - x2) / ((x0 - x1) * (x0 - x2));
    let l1 = (x - x0) * (x - x2) / ((x1 - x0) * (x1 - x2));
    let l2 = (x - x0) * (x - x1) / ((x2 - x0) * (x2 - x1));
    y0 * l0 + y1 * l1 + y2 * l2
}

/// Reduces candidate extrema to exactly `m` sign-alternating points (largest
/// magnitude per sign run, trimmed from the ends). Returns `None` if fewer than
/// `m` survive.
pub(crate) fn select_alternating<F: Float>(extrema: Vec<(F, F)>, m: usize) -> Option<Vec<(F, F)>> {
    let mut merged: Vec<(F, F)> = Vec::new();
    for (y, e) in extrema {
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
        if merged.first().unwrap().1.abs() <= merged.last().unwrap().1.abs() {
            merged.remove(0);
        } else {
            merged.pop();
        }
    }
    Some(merged)
}

/// Gaussian elimination with partial pivoting; `None` if singular.
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

    /// Dense sup-norm error of a `Polynomial` against `f` over its interval.
    fn sup_error(poly: &Polynomial<f64>, f: impl Fn(f64) -> f64, n: usize) -> f64 {
        let (a, b) = (poly.a, poly.b);
        (0..=n)
            .map(|i| {
                let x = a + (b - a) * (i as f64) / (n as f64);
                (poly.evaluate_on_interval(x) - f(x)).abs()
            })
            .fold(0.0, f64::max)
    }

    #[test]
    fn recovers_low_degree_polynomial() {
        // f(x) = x^3 is odd; the degree-3 minimax is x^3 itself.
        let r = minimax(|x: f64| x * x * x, -1.0, 1.0, 3, Parity::Odd).unwrap();
        assert!(r.error < 1e-9, "recovery error {} too large", r.error);
        // x^3 = (3·T1 + T3)/4.
        assert!((r.poly.coeffs[1] - 0.75).abs() < 1e-9);
        assert!((r.poly.coeffs[3] - 0.25).abs() < 1e-9);
        assert!(r.poly.coeffs[0].abs() < 1e-9 && r.poly.coeffs[2].abs() < 1e-9);
    }

    #[test]
    fn minimax_beats_or_matches_interpolation() {
        // Minimax is sup-norm optimal, so ≤ Chebyshev interpolation error.
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
        // At the optimum the reported error equals the leveled error, i.e. the
        // extrema equioscillate: check the error is close to the true sup error.
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
        // exp on [0, 2]: interval metadata must be respected by the evaluator.
        let f = |x: f64| x.exp();
        let mm = minimax(f, 0.0, 2.0, 10, Parity::Full).unwrap();
        assert!(mm.converged);
        assert!(sup_error(&mm.poly, f, 4000) < 1e-6);
    }
}
