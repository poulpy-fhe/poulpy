//! Han–Ki Chebyshev approximation of `cos(2π·(x - 1/4)/2^scnum)` on `[-K, K]`,
//! adapted from `lattigo/he/hefloat/cosine/cosine_approx.go`. The solve runs
//! in 256-bit `FBig` since the linear system is poorly conditioned at the
//! supported parameter ranges, and narrows to the target scalar `F` only at
//! the return boundary — mantissa-exactly, via a three-way `f64` decomposition
//! (see [`fbig_to_scalar`]), so `F = Quad` receives full binary128 precision.
//! `cos` and π are implemented locally because `dashu-float` does not provide
//! them.

#![allow(clippy::needless_range_loop)]

use anyhow::{Result, anyhow, ensure};
use dashu_float::{Context, DBig, FBig, round::mode::HalfEven};
use num_traits::{Float, FromPrimitive};

pub const ENCODING_PRECISION: usize = 256;

fn ctx() -> Context<HalfEven> {
    Context::<HalfEven>::new(ENCODING_PRECISION)
}

const PI_STR_256: &str =
    "3.14159265358979323846264338327950288419716939937510582097494459230781640628620899862803482534211706798214808651328";

fn pi_big() -> FBig<HalfEven> {
    let raw: DBig = PI_STR_256.parse().expect("PI_STR_256 is well-formed");
    raw.with_rounding::<HalfEven>()
        .with_base::<2>()
        .value()
        .with_precision(ENCODING_PRECISION)
        .value()
}

fn two_pi_big() -> FBig<HalfEven> {
    let pi = pi_big();
    let ctx = ctx();
    ctx.mul(pi.repr(), FBig::<HalfEven>::from(2).repr()).unwrap().value()
}

fn from_i64(x: i64) -> FBig<HalfEven> {
    FBig::<HalfEven>::from(x).with_precision(ENCODING_PRECISION).value()
}

fn from_f64(x: f64) -> FBig<HalfEven> {
    FBig::<HalfEven>::try_from(x)
        .expect("finite f64")
        .with_precision(ENCODING_PRECISION)
        .value()
}

fn add(a: &FBig<HalfEven>, b: &FBig<HalfEven>) -> FBig<HalfEven> {
    ctx().add(a.repr(), b.repr()).unwrap().value()
}

fn sub(a: &FBig<HalfEven>, b: &FBig<HalfEven>) -> FBig<HalfEven> {
    ctx().sub(a.repr(), b.repr()).unwrap().value()
}

fn mul(a: &FBig<HalfEven>, b: &FBig<HalfEven>) -> FBig<HalfEven> {
    ctx().mul(a.repr(), b.repr()).unwrap().value()
}

fn div(a: &FBig<HalfEven>, b: &FBig<HalfEven>) -> FBig<HalfEven> {
    ctx().div(a.repr(), b.repr()).unwrap().value()
}

fn neg(a: &FBig<HalfEven>) -> FBig<HalfEven> {
    sub(&FBig::<HalfEven>::ZERO, a)
}

fn abs(a: &FBig<HalfEven>) -> FBig<HalfEven> {
    if a < &FBig::<HalfEven>::ZERO { neg(a) } else { a.clone() }
}

fn floor_div_i64(a: &FBig<HalfEven>, b: &FBig<HalfEven>) -> i64 {
    let q = div(a, b);
    let f64q = q.to_f64().value();
    f64q.floor() as i64
}

fn cos_big(x: &FBig<HalfEven>) -> FBig<HalfEven> {
    let pi = pi_big();
    let two_pi = two_pi_big();
    let half = div(&from_i64(1), &from_i64(2));
    let pi_half = mul(&pi, &half);
    let pi_quarter = mul(&pi_half, &half);

    let n = floor_div_i64(x, &two_pi);
    let mut r = sub(x, &mul(&from_i64(n), &two_pi));
    if r < FBig::<HalfEven>::ZERO {
        r = add(&r, &two_pi);
    }
    if r > pi {
        r = sub(&two_pi, &r);
    }
    let mut sign = 1i32;
    if r > pi_half {
        r = sub(&pi_big(), &r);
        sign = -sign;
    }
    let mut use_sin = false;
    if r > pi_quarter {
        r = sub(&pi_half, &r);
        use_sin = true;
    }
    let result = if use_sin { taylor_sin(&r) } else { taylor_cos(&r) };
    if sign < 0 { neg(&result) } else { result }
}

fn taylor_cos(r: &FBig<HalfEven>) -> FBig<HalfEven> {
    let one = from_i64(1);
    let r2 = mul(r, r);
    let mut term = one.clone();
    let mut acc = one.clone();
    let stop = epsilon();
    let mut n: i64 = 0;
    loop {
        n += 2;
        let denom = from_i64(n * (n - 1));
        term = neg(&div(&mul(&term, &r2), &denom));
        acc = add(&acc, &term);
        if abs(&term) < stop {
            break;
        }
        if n > 200 {
            break;
        }
    }
    acc
}

fn taylor_sin(r: &FBig<HalfEven>) -> FBig<HalfEven> {
    let r2 = mul(r, r);
    let mut term = r.clone();
    let mut acc = r.clone();
    let stop = epsilon();
    let mut n: i64 = 1;
    loop {
        n += 2;
        let denom = from_i64(n * (n - 1));
        term = neg(&div(&mul(&term, &r2), &denom));
        acc = add(&acc, &term);
        if abs(&term) < stop {
            break;
        }
        if n > 200 {
            break;
        }
    }
    acc
}

fn epsilon() -> FBig<HalfEven> {
    let mut e = from_i64(1);
    let two = from_i64(2);
    for _ in 0..ENCODING_PRECISION {
        e = div(&e, &two);
    }
    e
}

fn cos2pi_x_minus_quarter_over_r(x: &FBig<HalfEven>, r: &FBig<HalfEven>) -> FBig<HalfEven> {
    let two_pi = two_pi_big();
    let quarter = div(&from_i64(1), &from_i64(4));
    let xq = sub(x, &quarter);
    let inner = div(&xq, r);
    let arg = mul(&two_pi, &inner);
    cos_big(&arg)
}

fn log2(x: f64) -> f64 {
    x.log2()
}

fn max_index(arr: &[f64]) -> usize {
    let mut idx = 0usize;
    let mut best = arr[0];
    for (i, &v) in arr.iter().enumerate().skip(1) {
        if v > best {
            best = v;
            idx = i;
        }
    }
    idx
}

fn gen_degrees(degree: usize, k: usize, dev: f64) -> (Vec<usize>, usize) {
    let degbdd = (degree + 1) as i64;
    let mut totdeg: i64 = (2 * k - 1) as i64;
    let err = 1.0 / dev;
    let mut deg = vec![1usize; k];

    let mut bdd = vec![0f64; k];
    let mut temp = 0f64;
    for i in 1..=(2 * k - 1) as i64 {
        temp -= log2(i as f64);
    }
    let log2_two_pi = (2.0 * std::f64::consts::PI).log2();
    temp += (2.0 * k as f64 - 1.0) * log2_two_pi;
    temp += log2(err);

    for i in 0..k {
        bdd[i] = temp;
        for j in 1..=(k as i64 - 1 - i as i64) {
            bdd[i] += log2(j as f64 + err);
        }
        for j in 1..=(k as i64 - 1 + i as i64) {
            bdd[i] += log2(j as f64 + err);
        }
    }

    let maxiter = 200;
    for _ in 0..maxiter {
        if totdeg >= degbdd {
            break;
        }
        let maxi = max_index(&bdd);

        if maxi != 0 {
            if totdeg + 2 > degbdd {
                break;
            }
            for i in 0..k {
                bdd[i] -= log2((totdeg + 1) as f64);
                bdd[i] -= log2((totdeg + 2) as f64);
                bdd[i] += 2.0 * log2_two_pi;

                if i != maxi {
                    let di = (i as i64 - maxi as i64).unsigned_abs() as f64;
                    bdd[i] += log2(di + err);
                    bdd[i] += log2((i + maxi) as f64 + err);
                } else {
                    bdd[i] += log2(err) - 1.0;
                    bdd[i] += log2(2.0 * i as f64 + err);
                }
            }
            totdeg += 2;
        } else {
            bdd[0] -= log2((totdeg + 1) as f64);
            bdd[0] += log2(err) - 1.0;
            bdd[0] += log2_two_pi;
            for i in 1..k {
                bdd[i] -= log2((totdeg + 1) as f64);
                bdd[i] += log2_two_pi;
                bdd[i] += log2(i as f64 + err);
            }
            totdeg += 1;
        }

        deg[maxi] += 1;
    }

    (deg, totdeg as usize)
}

fn gen_nodes(deg: &[usize], dev: f64, totdeg: usize, k: usize, scnum: usize) -> (Vec<FBig<HalfEven>>, Vec<FBig<HalfEven>>) {
    let scfac = from_f64((1u64 << scnum) as f64);
    let intersize = div(&from_i64(1), &from_f64(dev));
    let pi = pi_big();

    let mut nodes = vec![FBig::<HalfEven>::ZERO; totdeg];
    let mut cnt = 0usize;
    if !deg[0].is_multiple_of(2) {
        cnt += 1;
    }

    for i in (1..k).rev() {
        let twodegi = from_i64((2 * deg[i]) as i64);
        let i_big = from_i64(i as i64);
        for j in 0..deg[i] {
            let two_j = from_i64((2 * j) as i64);
            let arg = div(&mul(&pi, &two_j), &twodegi);
            let c = cos_big(&arg);
            let off = mul(&c, &intersize);
            nodes[cnt] = add(&i_big, &off);
            cnt += 1;
            nodes[cnt] = neg(&nodes[cnt - 1]);
            cnt += 1;
        }
    }

    let twodeg0 = from_i64((2 * deg[0]) as i64);
    for j in 0..(deg[0] / 2) {
        let two_j = from_i64((2 * j) as i64);
        let arg = div(&mul(&pi, &two_j), &twodeg0);
        let c = cos_big(&arg);
        let off = mul(&c, &intersize);
        nodes[cnt] = off;
        cnt += 1;
        nodes[cnt] = neg(&nodes[cnt - 1]);
        cnt += 1;
    }

    let mut y = vec![FBig::<HalfEven>::ZERO; totdeg];
    for i in 0..totdeg {
        y[i] = cos2pi_x_minus_quarter_over_r(&nodes[i], &scfac);
    }

    (nodes, y)
}

fn solve(
    totdeg_in: usize,
    k: usize,
    _scnum: usize,
    nodes: Vec<FBig<HalfEven>>,
    mut y: Vec<FBig<HalfEven>>,
) -> Vec<FBig<HalfEven>> {
    let totdeg = totdeg_in;

    for j in 1..totdeg {
        for i in 0..(totdeg - j) {
            let diff = sub(&y[i + 1], &y[i]);
            let denom = sub(&nodes[i + j], &nodes[i]);
            y[i] = div(&diff, &denom);
        }
    }

    let totdeg_p1 = totdeg + 1;

    // Coefficients are emitted in the standard Chebyshev basis on [-k, k]
    // (variable u = x/k), so callers can evaluate via the standard
    // T_n(v) recurrence on the ciphertext value directly.
    let k_big = from_i64(k as i64);

    let pi = pi_big();
    let mut x = vec![FBig::<HalfEven>::ZERO; totdeg_p1];
    for i in 0..totdeg_p1 {
        let arg = div(&mul(&from_i64(i as i64), &pi), &from_i64((totdeg_p1 - 1) as i64));
        let c = cos_big(&arg);
        x[i] = mul(&k_big, &c);
    }

    let mut p = vec![FBig::<HalfEven>::ZERO; totdeg_p1];
    for i in 0..totdeg_p1 {
        let mut pi_val = y[0].clone();
        for j in 1..(totdeg_p1 - 1) {
            let diff = sub(&x[i], &nodes[j]);
            pi_val = mul(&pi_val, &diff);
            pi_val = add(&pi_val, &y[j]);
        }
        p[i] = pi_val;
    }

    let mut t: Vec<Vec<FBig<HalfEven>>> = (0..totdeg_p1).map(|_| vec![FBig::<HalfEven>::ZERO; totdeg_p1]).collect();
    for i in 0..totdeg_p1 {
        t[i][0] = from_i64(1);
        t[i][1] = div(&x[i], &k_big);
        for j in 2..totdeg_p1 {
            let two_xi_over_k = mul(&from_i64(2), &div(&x[i], &k_big));
            let prev = t[i][j - 1].clone();
            let prev_prev = t[i][j - 2].clone();
            t[i][j] = sub(&mul(&two_xi_over_k, &prev), &prev_prev);
        }
    }

    for i in 0..(totdeg_p1 - 1) {
        let mut maxabs = abs(&t[i][i]);
        let mut maxindex = i;
        for j in (i + 1)..totdeg_p1 {
            let a = abs(&t[j][i]);
            if a > maxabs {
                maxabs = a;
                maxindex = j;
            }
        }
        if i != maxindex {
            for j in i..totdeg_p1 {
                let tmp = t[i][j].clone();
                t[i][j] = t[maxindex][j].clone();
                t[maxindex][j] = tmp;
            }
            p.swap(i, maxindex);
        }

        for j in (i + 1)..totdeg_p1 {
            let factor = t[i][j].clone();
            t[i][j] = div(&factor, &t[i][i]);
        }
        p[i] = div(&p[i], &t[i][i]);
        t[i][i] = from_i64(1);

        for j in (i + 1)..totdeg_p1 {
            let factor = t[j][i].clone();
            let p_i = p[i].clone();
            let tmp = mul(&factor, &p_i);
            p[j] = sub(&p[j], &tmp);
            for l in (i + 1)..totdeg_p1 {
                let t_il = t[i][l].clone();
                let prod = mul(&factor, &t_il);
                t[j][l] = sub(&t[j][l], &prod);
            }
            t[j][i] = FBig::<HalfEven>::ZERO;
        }
    }

    let mut c = vec![FBig::<HalfEven>::ZERO; totdeg_p1];
    c[totdeg_p1 - 1] = p[totdeg_p1 - 1].clone();
    for i in (0..(totdeg_p1 - 1)).rev() {
        let mut ci = p[i].clone();
        for j in (i + 1)..totdeg_p1 {
            let prod = mul(&t[i][j], &c[j]);
            ci = sub(&ci, &prod);
        }
        c[i] = ci;
    }

    c
}

/// Narrows a 256-bit `FBig` to the target scalar `F` **mantissa-exactly** (up
/// to `F`'s own precision): the value is decomposed into three `f64`
/// components (`hi + mid + lo`, each the correctly-rounded remainder of the
/// previous ones — a triple-double covering ~159 significand bits) and
/// recomposed in `F` in descending magnitude, so `F = Quad` (113-bit
/// significand) receives its full precision while `F = f64` collapses to the
/// plain rounding.
fn fbig_to_scalar<F: Float + FromPrimitive>(c: &FBig<HalfEven>) -> F {
    let hi = c.to_f64().value();
    let rem = c - from_f64(hi);
    let mid = rem.to_f64().value();
    let lo = (rem - from_f64(mid)).to_f64().value();
    let f = |x: f64| F::from_f64(x).expect("finite scalar");
    f(hi) + f(mid) + f(lo)
}

/// Number of coefficients [`approximate_cos`] produces for `(k, degree, dev)`
/// (its polynomial has degree `len − 1`). Lets callers size the BSGS depth of the
/// CosHK base polynomial without solving for the coefficients.
pub fn approximate_cos_len(k: usize, degree: usize, dev: f64) -> usize {
    gen_degrees(degree, k, dev).1
}

/// Returns Chebyshev coefficients approximating `cos(2π·(x - 1/4)/2^scnum)` on
/// `[-k, k]` in the standard basis (variable `u = x/k`). The solve runs in
/// 256-bit `FBig`; each coefficient is narrowed mantissa-exactly to the target
/// `F` (see [`fbig_to_scalar`]), so wide scalars such as `Quad` keep their full
/// precision.
pub fn approximate_cos<F: Float + FromPrimitive>(k: usize, degree: usize, dev: f64, scnum: usize) -> Vec<F> {
    let (deg, totdeg) = gen_degrees(degree, k, dev);
    let (nodes, y) = gen_nodes(&deg, dev, totdeg, k, scnum);
    let coeffs = solve(totdeg, k, scnum, nodes, y);
    // solve returns totdeg+1 coefficients; the trailing one is outside the target polynomial degree.
    coeffs.iter().take(totdeg).map(|c| fbig_to_scalar(c)).collect()
}

pub(crate) fn even_cheby_cos_coeffs(k: usize, eps: f64, degree: usize) -> Result<Vec<f64>> {
    ensure!(k > 0, "even Chebyshev interval must be positive");
    ensure!(
        eps.is_finite() && eps > 0.0,
        "even Chebyshev margin must be finite and positive"
    );
    ensure!(degree > 0, "even Chebyshev degree must be positive");

    let beta = std::f64::consts::PI * eps * k as f64;
    ensure!(beta.is_finite(), "even Chebyshev frequency must be finite");
    const TAIL_TERMS: usize = 32;
    let max_order = degree
        .checked_add(2 * TAIL_TERMS)
        .and_then(|order| order.checked_mul(2))
        .ok_or_else(|| anyhow!("even Chebyshev degree overflow"))?;
    let bessel = bessel_j_all(beta, max_order)?;
    let coefficient = |j: usize| {
        if j == 0 {
            bessel[0]
        } else {
            let sign = if j.is_multiple_of(2) { 1.0 } else { -1.0 };
            2.0 * sign * bessel[2 * j]
        }
    };

    let mut coeffs = vec![0.0; 2 * degree + 1];
    let scale = std::f64::consts::TAU.recip();
    for j in 0..=degree {
        coeffs[2 * j] = scale * coefficient(j);
    }
    coeffs[2 * degree] += scale * (1..=TAIL_TERMS).map(|i| coefficient(degree + 2 * i)).sum::<f64>();
    ensure!(
        coeffs.iter().all(|coefficient| coefficient.is_finite()),
        "even Chebyshev coefficients must be finite"
    );
    Ok(coeffs)
}

fn bessel_j_all(x: f64, max_order: usize) -> Result<Vec<f64>> {
    ensure!(x.is_finite() && x > 0.0, "Bessel argument must be finite and positive");
    ensure!(x.ceil() <= usize::MAX as f64, "Bessel argument is too large");
    let start = max_order
        .checked_add(20)
        .and_then(|order| order.checked_add(x.ceil() as usize))
        .ok_or_else(|| anyhow!("Bessel order overflow"))?;
    let mut next = 0.0;
    let mut current = 1e-300;
    let mut values = vec![0.0; max_order + 1];
    for n in (1..=start).rev() {
        let previous = (2.0 * n as f64 / x) * current - next;
        if n - 1 <= max_order {
            values[n - 1] = previous;
        }
        next = current;
        current = previous;
        if current.abs() > 1e250 {
            current *= 1e-250;
            next *= 1e-250;
            values.iter_mut().for_each(|value| *value *= 1e-250);
        }
    }

    let norm = values[0] + 2.0 * values.iter().skip(2).step_by(2).sum::<f64>();
    ensure!(norm.is_finite() && norm != 0.0, "Bessel normalization failed");
    values.iter_mut().for_each(|value| *value /= norm);
    Ok(values)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pi_constant_parses() {
        let p = pi_big();
        let f = p.to_f64().value();
        assert!((f - std::f64::consts::PI).abs() < 1e-15);
    }

    #[test]
    fn cos_matches_f64_at_simple_points() {
        let zero = FBig::<HalfEven>::ZERO;
        assert!((cos_big(&zero).to_f64().value() - 1.0).abs() < 1e-15);

        let pi = pi_big();
        assert!((cos_big(&pi).to_f64().value() + 1.0).abs() < 1e-15);

        let half = div(&from_i64(1), &from_i64(2));
        let pi_half = mul(&pi, &half);
        assert!(cos_big(&pi_half).to_f64().value().abs() < 1e-15);
    }

    #[test]
    fn approximate_cos_returns_finite_coefficients() {
        let coeffs: Vec<f64> = approximate_cos(12, 30, 256.0, 3);
        assert!(coeffs.len() >= 23, "expected totdeg >= 2K-1 = 23, got {}", coeffs.len());
        for c in &coeffs {
            assert!(c.is_finite(), "non-finite coefficient: {c}");
        }
    }

    /// CosHK coefficients must reach the target scalar mantissa-exactly: the
    /// `Quad` coefficients agree with the `f64` ones when rounded to `f64`
    /// (same solve, same rounding boundary), and carry a genuine sub-`f64`
    /// mantissa (a nonzero low-order residue on most coefficients) — the old
    /// `FBig → f64 → F` narrowing zeroed every bit below 53.
    #[test]
    fn approximate_cos_quad_carries_full_mantissa() {
        use crate::Quad;
        use num_traits::ToPrimitive;

        let (k, degree, dev, scnum) = (12usize, 30usize, 256.0, 3usize);
        let c64: Vec<f64> = approximate_cos(k, degree, dev, scnum);
        let cq: Vec<Quad> = approximate_cos(k, degree, dev, scnum);
        assert_eq!(c64.len(), cq.len());

        let mut with_residue = 0usize;
        for (a, b) in c64.iter().zip(&cq) {
            let b64 = b.to_f64().unwrap();
            // Rounding the Quad coefficient to f64 reproduces the f64 path
            // within one ulp (the triple-double recomposition and the direct
            // f64 narrowing may round the tie differently).
            assert!(
                (b64 - a).abs() <= a.abs().max(f64::MIN_POSITIVE) * f64::EPSILON,
                "Quad coefficient diverges from f64 path: {a} vs {b64}"
            );
            let residue = (*b - Quad::from_f64(b64).unwrap()).abs().to_f64().unwrap();
            if residue > 0.0 {
                with_residue += 1;
            }
        }
        // The solve's 256-bit values are irrational-ish; essentially all
        // coefficients must carry bits below the f64 boundary.
        assert!(
            with_residue * 2 > cq.len(),
            "only {with_residue}/{} Quad coefficients carry sub-f64 mantissa",
            cq.len()
        );
    }

    fn clenshaw(coeffs: &[f64], u: f64) -> f64 {
        let mut b2 = 0f64;
        let mut b1 = 0f64;
        for i in (1..coeffs.len()).rev() {
            let t = 2.0 * u * b1 - b2 + coeffs[i];
            b2 = b1;
            b1 = t;
        }
        coeffs[0] + u * b1 - b2
    }

    #[test]
    fn approximate_cos_matches_target_in_standard_basis() {
        let k = 12usize;
        let scnum = 3usize;
        let coeffs: Vec<f64> = approximate_cos(k, 30, 256.0, scnum);

        let sc_fac = (1u64 << scnum) as f64;
        let k_f = k as f64;
        let target = |u: f64| ((2.0 * std::f64::consts::PI) * (k_f * u - 0.25) / sc_fac).cos();

        for i in 1..k {
            let u = i as f64 / k as f64;
            let got = clenshaw(&coeffs, u);
            let want = target(u);
            assert!((got - want).abs() < 1e-6, "u={u:.4}: got={got}, want={want}");
        }
    }

    #[test]
    fn approximate_cos_returns_nonzero_odd_coefficients() {
        let coeffs: Vec<f64> = approximate_cos(12, 30, 256.0, 3);
        let any_odd = coeffs
            .iter()
            .enumerate()
            .any(|(i, c)| !i.is_multiple_of(2) && c.abs() > 1e-12);
        assert!(
            any_odd,
            "cos(2π·(x-1/4)/2^r) is not even; odd Chebyshev coefficients must be nonzero"
        );
    }

    #[test]
    fn approximate_cos_matches_target_at_finer_grid() {
        let k = 12usize;
        let scnum = 3usize;
        let coeffs: Vec<f64> = approximate_cos(k, 30, 256.0, scnum);

        let sc_fac = (1u64 << scnum) as f64;
        let k_f = k as f64;
        let target = |u: f64| ((2.0 * std::f64::consts::PI) * (k_f * u - 0.25) / sc_fac).cos();

        let n = 200usize;
        for i in 0..=n {
            let u = -1.0 + 2.0 * (i as f64) / (n as f64);
            let got = clenshaw(&coeffs, u);
            let want = target(u);
            assert!((got - want).abs() < 1e-2, "u={u:.4}: got={got}, want={want}");
        }
    }

    #[test]
    fn even_cheby_matches_cosine() {
        let (k, eps, degree) = (16, 0.25, 31);
        let coeffs = even_cheby_cos_coeffs(k, eps, degree).unwrap();
        let beta = std::f64::consts::PI * eps * k as f64;
        for i in 0..=128 {
            let t = -1.0 + i as f64 / 64.0;
            let got = clenshaw(&coeffs, t);
            let expected = (beta * t).cos() / std::f64::consts::TAU;
            assert!((got - expected).abs() < 1e-12, "t={t}: got={got}, expected={expected}");
        }
    }
}
