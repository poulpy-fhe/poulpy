//! Ring layer: every ring-dependent operation computes in the standard
//! negacyclic ring of degree `m = R::std_degree(n)`, through the embedding of
//! the backend ring into it.
//!
//! The standard ring embeds as itself. The conjugate-invariant ring `R_n` is
//! the subring of `Z[X]/(X^(2n) + 1)` fixed by `X -> X^-1`, with coordinates
//! `a_0 + sum_{0<j<n} a_j (X^j + X^-j)`. Its image has coefficient `a_j` at
//! `X^j` for `j < n`, so reading a subring element back is a prefix copy.
//!
//! A subring element takes equal values at `z` and `z^-1`. Both transforms
//! place the evaluations at exponents `1 (mod 4)` of a primitive `4n`-th root,
//! one per conjugate pair, in their first `n` slots, and the stored spectrum is
//! that prefix. [`DFTFamily::ci_expand`] rebuilds the full standard spectrum.

use poulpy_hal::layouts::{ConjugateInvariant, Module, Ring, Standard};

use crate::{
    backend::{Oracle, table},
    family::{DFTFamily, Int},
};

/// A backend ring, by the degree of its standard image.
pub trait OracleRing: Ring {
    /// Degree of the standard ring holding the image of a degree-`n` element.
    fn std_degree(n: usize) -> usize;
}

impl OracleRing for Standard {
    fn std_degree(n: usize) -> usize {
        n
    }
}

impl OracleRing for ConjugateInvariant {
    fn std_degree(n: usize) -> usize {
        2 * n
    }
}

/// The standard image of a conjugate-invariant element: `a_0`, then `a_j` at
/// `X^j` and `-a_j` at `X^(2n-j)`, since `X^-j = -X^(2n-j)`.
fn unfold<T: Copy + Default>(a: &[T], neg: impl Fn(T) -> T) -> Vec<T> {
    let n = a.len();
    let mut res = vec![T::default(); 2 * n];
    res[..n].copy_from_slice(a);
    for j in 1..n {
        res[2 * n - j] = neg(a[j]);
    }
    res
}

fn is_standard<R: OracleRing>(n: usize) -> bool {
    R::std_degree(n) == n
}

fn expand<F: DFTFamily>(a: &[F::Dft]) -> Vec<F::Dft> {
    let mut spectrum = vec![F::Dft::default(); 2 * a.len()];
    F::ci_expand(&mut spectrum, a);
    spectrum
}

/// `res = DFT(a)`.
pub fn forward<F: DFTFamily, R: OracleRing>(module: &Module<Oracle<F, R>>, res: &mut [F::Dft], a: &[i64]) {
    let n = a.len();
    if is_standard::<R>(n) {
        return F::forward(table(module, n), res, a);
    }
    let mut spectrum = vec![F::Dft::default(); 2 * n];
    F::forward(table(module, 2 * n), &mut spectrum, &unfold(a, i64::wrapping_neg));
    res.copy_from_slice(&spectrum[..n]);
}

/// `res = IDFT(a)`, the exact integer coefficients.
pub fn inverse<F: DFTFamily, R: OracleRing>(module: &Module<Oracle<F, R>>, res: &mut [F::Big], a: &[F::Dft]) {
    let n = a.len();
    if is_standard::<R>(n) {
        return F::inverse(table(module, n), res, a);
    }
    let mut coeffs = vec![F::Big::default(); 2 * n];
    F::inverse(table(module, 2 * n), &mut coeffs, &expand::<F>(a));
    res.copy_from_slice(&coeffs[..n]);
}

/// Embeds a stored spectrum into a larger degree without returning to coefficients.
pub fn dft_embed<F: DFTFamily, R: OracleRing>(res: &mut [F::Dft], a: &[F::Dft]) {
    if is_standard::<R>(a.len()) {
        return F::dft_embed(res, a);
    }
    let mut spectrum = vec![F::Dft::default(); 2 * res.len()];
    F::dft_embed(&mut spectrum, &expand::<F>(a));
    res.copy_from_slice(&spectrum[..res.len()]);
}

/// `res += a * b` on stored spectra.
pub fn mul_acc<F: DFTFamily, R: OracleRing>(res: &mut [F::Dft], a: &[F::Dft], b: &[F::Dft]) {
    let n = res.len();
    if is_standard::<R>(n) {
        return F::mul_acc(res, a, b);
    }
    let mut spectrum = expand::<F>(res);
    F::mul_acc(&mut spectrum, &expand::<F>(a), &expand::<F>(b));
    res.copy_from_slice(&spectrum[..n]);
}

/// `res *= a` on stored spectra.
pub fn mul_assign<F: DFTFamily, R: OracleRing>(res: &mut [F::Dft], a: &[F::Dft]) {
    let n = res.len();
    if is_standard::<R>(n) {
        return F::mul_assign(res, a);
    }
    let mut spectrum = expand::<F>(res);
    F::mul_assign(&mut spectrum, &expand::<F>(a));
    res.copy_from_slice(&spectrum[..n]);
}

/// `res = DFT(sigma_p(IDFT(a)))` on stored spectra.
pub fn dft_automorphism<F: DFTFamily, R: OracleRing>(p: i64, res: &mut [F::Dft], a: &[F::Dft]) {
    let n = a.len();
    if is_standard::<R>(n) {
        return F::dft_automorphism(p, res, a);
    }
    let mut spectrum = vec![F::Dft::default(); 2 * n];
    F::dft_automorphism(p, &mut spectrum, &expand::<F>(a));
    res.copy_from_slice(&spectrum[..n]);
}

/// `X -> X^p` in `Z[X]/(X^m + 1)`, `m = a.len()`, `p` odd.
fn std_automorphism<T: Int>(p: i64, res: &mut [T], a: &[T]) {
    let m = a.len() as i64;
    assert!(p & 1 == 1, "p must be odd, got {p}");
    for (i, &x) in a.iter().enumerate() {
        let k = (p.rem_euclid(2 * m) * i as i64) % (2 * m);
        if k < m {
            res[k as usize] = x;
        } else {
            res[(k - m) as usize] = x.neg();
        }
    }
}

/// `res = sigma_p(a)` on coefficients.
pub fn automorphism<R: OracleRing, T: Int>(p: i64, res: &mut [T], a: &[T]) {
    let n = a.len();
    if is_standard::<R>(n) {
        return std_automorphism(p, res, a);
    }
    let mut image = vec![T::default(); 2 * n];
    std_automorphism(p, &mut image, &unfold(a, T::neg));
    res.copy_from_slice(&image[..n]);
}

#[cfg(test)]
mod tests {
    use poulpy_hal::layouts::{ConjugateInvariant, Module};

    use super::*;
    use crate::{FFT64CIOracle, NTT4x30CIOracle};

    fn sample(n: usize, seed: i64, bound: i64) -> Vec<i64> {
        (0..n as i64)
            .map(|i| ((i * 7919 + seed * 104729) % (2 * bound + 1)) - bound)
            .collect()
    }

    /// The conjugate-invariant product from its definition: the schoolbook
    /// product of the standard images, read back.
    fn ci_schoolbook(a: &[i64], b: &[i64]) -> Vec<i128> {
        let n = a.len();
        let (ua, ub) = (unfold(a, |x| -x), unfold(b, |x| -x));
        let mut prod = vec![0i128; 2 * n];
        for (i, &x) in ua.iter().enumerate() {
            for (j, &y) in ub.iter().enumerate() {
                let term = i128::from(x) * i128::from(y);
                if i + j < 2 * n {
                    prod[i + j] += term;
                } else {
                    prod[i + j - 2 * n] -= term;
                }
            }
        }
        // The product of two subring elements lies in the subring.
        assert_eq!(prod[n], 0);
        (1..n).for_each(|j| assert_eq!(prod[2 * n - j], -prod[j]));
        prod[..n].to_vec()
    }

    fn check_product<F: DFTFamily>(module: &Module<Oracle<F, ConjugateInvariant>>, bound: i64) {
        for n in [8usize, 16, 64] {
            let (a, b) = (sample(n, 1, bound), sample(n, 2, bound));
            let (mut fa, mut fb) = (vec![F::Dft::default(); n], vec![F::Dft::default(); n]);
            forward(module, &mut fa, &a);
            forward(module, &mut fb, &b);
            let mut prod = vec![F::Dft::default(); n];
            mul_acc::<F, ConjugateInvariant>(&mut prod, &fa, &fb);
            let mut got = vec![F::Big::default(); n];
            inverse(module, &mut got, &prod);
            let got: Vec<i128> = got.into_iter().map(Into::into).collect();
            assert_eq!(got, ci_schoolbook(&a, &b), "n = {n}");
        }
    }

    #[test]
    fn ci_products_match_definition() {
        check_product(&Module::<FFT64CIOracle>::new(64), 1 << 10);
        check_product(&Module::<NTT4x30CIOracle>::new(64), 1 << 20);
    }

    fn check_embedding<F: DFTFamily>(module: &Module<Oracle<F, ConjugateInvariant>>) {
        for n in [1usize, 2, 8, 16, 64] {
            for big in [n, 2 * n, 4 * n] {
                let input = sample(n, 3, 1000);
                let mut small = vec![F::Dft::default(); n];
                forward(module, &mut small, &input);
                let mut embedded = vec![F::Dft::default(); big];
                dft_embed::<F, ConjugateInvariant>(&mut embedded, &small);
                let mut actual = vec![F::Big::default(); big];
                inverse(module, &mut actual, &embedded);
                let actual: Vec<i128> = actual.into_iter().map(Into::into).collect();
                let mut expected = vec![0i128; big];
                for (i, &value) in input.iter().enumerate() {
                    expected[i * (big / n)] = i128::from(value);
                }
                assert_eq!(actual, expected, "embedding {n} -> {big}");
            }
        }
    }

    #[test]
    fn ci_spectrum_embedding_matches_coefficient_embedding() {
        check_embedding(&Module::<FFT64CIOracle>::new(256));
        check_embedding(&Module::<NTT4x30CIOracle>::new(256));
    }

    /// `sigma_p` on the basis: `X^j + X^-j` goes to `X^t + X^-t`, `t = p j mod 4n`,
    /// which is `b_t` for `t < n`, zero for `t = n`, `-b_(2n-t)` up to `2n`, and
    /// symmetric beyond.
    fn ci_automorphism_from_basis(p: i64, a: &[i64]) -> Vec<i64> {
        let n = a.len() as i64;
        let mut res = vec![0i64; a.len()];
        res[0] = a[0];
        for j in 1..n {
            let mut t = (p * j).rem_euclid(4 * n);
            if t > 2 * n {
                t = 4 * n - t;
            }
            if t < n {
                res[t as usize] += a[j as usize];
            } else if t > n {
                res[(2 * n - t) as usize] -= a[j as usize];
            }
        }
        res
    }

    #[test]
    fn ci_automorphism_matches_basis_action() {
        for n in [8usize, 16, 32] {
            let a = sample(n, 5, 1000);
            for p in [1i64, 3, 5, -1, -3, 7, 4 * n as i64 - 1, 2 * n as i64 + 1] {
                let mut got = vec![0i64; n];
                automorphism::<ConjugateInvariant, i64>(p, &mut got, &a);
                assert_eq!(got, ci_automorphism_from_basis(p, &a), "n = {n}, p = {p}");
            }
        }
    }
}
