//! The GBFV plaintext modulus `t(X)` and its inverse.

use anyhow::{Result, ensure};
use dashu_int::{IBig, UBig, ops::Gcd, ops::UnsignedAbs};

/// GBFV plaintext modulus: a non-zero, non-unit element `t(X)` of a power-of-two
/// cyclotomic ring, together with its inverse.
///
/// The caller gives `t` in any subring `Z[X]/(X^n + 1)` that contains it, as its `n`
/// coefficients. [`GBFVPlaintextModulus::new`] stores it in the smallest such subring,
/// of degree [`GBFVPlaintextModulus::n`]: `X^32 - 2` given at degree 512 is stored as
/// `X - 2` at degree 16. Every value stands for its embedding `X -> X^(N/n)` into the
/// encryption ring of degree `N`.
///
/// The inverse is stored as the irreducible fraction
/// `1/t = numerator / characteristic` in `Q[X]/(X^n + 1)`: `numerator` is a ring
/// element of the same subring and `characteristic` a positive integer, with no common
/// factor between `characteristic` and all coefficients of `numerator`. The denominator
/// is then the smallest positive integer in the ideal `(t)`: the characteristic of the
/// plaintext ring `Z[X]/(X^n + 1, t)`, and of `Z[X]/(X^N + 1, t)`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct GBFVPlaintextModulus {
    /// Coefficients of `t` in its smallest subring, lowest degree first.
    ptxt_mod: Vec<i64>,
    /// Coefficients of the numerator of `1/t`, in the same subring.
    numerator: Vec<IBig>,
    /// Denominator of `1/t`.
    characteristic: UBig,
    /// Absolute norm `|N(t)|` of `t` in its smallest subring.
    norm: UBig,
}

impl GBFVPlaintextModulus {
    /// Builds the plaintext modulus from the coefficients of `t` in `Z[X]/(X^n + 1)`,
    /// `n = ptxt_mod.len()`, lowest degree first, and computes its inverse.
    ///
    /// The inverse is computed in the smallest subring holding `t`:
    /// 1. the norm `N(t)`, the product of all automorphisms of `t`;
    /// 2. the product `adj(t)` of all automorphisms of `t` except the identity, so
    ///    that `t * adj(t) = N(t)`;
    /// 3. with `g = gcd(N(t), coefficients of adj(t))`, `characteristic = |N(t)| / g`
    ///    and `numerator = sign(N(t)) * adj(t) / g`.
    ///
    /// Errors when `n` is not a power of two, or when `t` is zero or a unit.
    pub fn new(ptxt_mod: &[i64]) -> Result<Self> {
        ensure!(
            ptxt_mod.len().is_power_of_two(),
            "plaintext modulus: degree must be a power of two"
        );
        ensure!(ptxt_mod.iter().any(|&c| c != 0), "plaintext modulus: must be non-zero");

        // `t` lies in `Z[X^g]`, `g` the largest power of two dividing `n` and every
        // exponent of a non-zero coefficient.
        let step: usize = ptxt_mod
            .iter()
            .enumerate()
            .filter(|&(_, &c)| c != 0)
            .fold(
                ptxt_mod.len(),
                |g, (i, _)| {
                    if i == 0 { g } else { g.min(1 << i.trailing_zeros()) }
                },
            );
        let ptxt_mod: Vec<i64> = ptxt_mod.iter().step_by(step).copied().collect();

        let (norm, adjugate): (IBig, Vec<IBig>) = norm_and_adjugate(&ptxt_mod);

        let norm_abs: UBig = (&norm).unsigned_abs();
        let gcd: UBig = adjugate.iter().fold(norm_abs.clone(), |g, c| g.gcd(c.unsigned_abs()));
        let characteristic: UBig = &norm_abs / &gcd;
        ensure!(characteristic > UBig::ONE, "plaintext modulus: must not be a unit");

        // Divides by `sign(N(t)) * gcd`, so that the characteristic is positive.
        let divisor: IBig = if norm < IBig::ZERO {
            -IBig::from(gcd)
        } else {
            IBig::from(gcd)
        };
        let numerator: Vec<IBig> = adjugate.iter().map(|c| c / &divisor).collect();

        Ok(Self {
            ptxt_mod,
            numerator,
            characteristic,
            norm: norm_abs,
        })
    }

    /// Coefficients of `t` in its smallest subring, lowest degree first.
    pub fn ptxt_mod(&self) -> &[i64] {
        &self.ptxt_mod
    }

    /// Coefficients of the numerator of `1/t`, in the subring of `t`, lowest degree first.
    pub fn numerator(&self) -> &[IBig] {
        &self.numerator
    }

    /// Denominator of `1/t`: the characteristic of the plaintext ring.
    pub fn characteristic(&self) -> &UBig {
        &self.characteristic
    }

    /// Absolute norm `|N(t)|` of `t` from its smallest subring down to `Z`.
    pub(crate) fn norm(&self) -> &UBig {
        &self.norm
    }

    /// Degree `n` of the smallest subring `Z[X]/(X^n + 1)` holding `t`.
    pub fn n(&self) -> usize {
        self.ptxt_mod.len()
    }

    /// Base-2 logarithm of [`Self::n`].
    pub fn log_n(&self) -> usize {
        self.n().trailing_zeros() as usize
    }

    /// Unique representative `r` of `a` modulo `t`: `r = a - t * q` for a ring element
    /// `q`, such that every coefficient of `r / t` lies in `[-1/2, 1/2)`.
    ///
    /// `a` is given in `Z[X]/(X^n + 1)`, `n = a.len()`, and `r` is returned at degree
    /// `max(n, self.n())`, the operand of smaller degree standing for its embedding.
    /// The representative is `r = t * [a * numerator]_characteristic / characteristic`,
    /// with `[x]_m` the centered reduction of each coefficient into `[-m/2, m/2)`. The
    /// division is exact: with `[a * numerator]_m = a * numerator - m * q`,
    /// `t * [a * numerator]_m = m * (a - t * q)`.
    ///
    /// Errors when `n` is not a power of two.
    pub fn flatten(&self, a: &[IBig]) -> Result<Vec<IBig>> {
        ensure!(a.len().is_power_of_two(), "flatten: degree must be a power of two");
        let characteristic: IBig = IBig::from(self.characteristic.clone());
        let reduced: Vec<IBig> = negacyclic_mul(a, &self.numerator)
            .into_iter()
            .map(|c| centered_rem(c, &characteristic))
            .collect();
        let ptxt_mod: Vec<IBig> = self.ptxt_mod.iter().map(|&c| IBig::from(c)).collect();
        Ok(negacyclic_mul(&reduced, &ptxt_mod)
            .into_iter()
            .map(|c| c / &characteristic)
            .collect())
    }
}

/// `x` reduced into `[-m/2, m/2)`, `m > 0`.
fn centered_rem(x: IBig, m: &IBig) -> IBig {
    let mut r: IBig = x % m;
    if r < IBig::ZERO {
        r += m;
    }
    if IBig::from(2u8) * &r >= *m {
        r -= m;
    }
    r
}

/// Norm `N(t)` and adjugate `adj(t)`, the product of the automorphisms of `t` other than
/// the identity, of `t` in `Z[X]/(X^n + 1)`, `n = t.len()`, so that `t * adj(t) = N(t)`.
///
/// Descends the tower of subrings: the relative norm of `t` to the subring of half the
/// degree is `t(X) t(-X) = e(Y)^2 - Y o(Y)^2`, with `Y = X^2` and `t = e(X^2) + X o(X^2)`.
/// Climbing back, `adj(t)(X) = t(-X) adj(t(X) t(-X))(X^2)`. Quadratic in `n`.
fn norm_and_adjugate(t: &[i64]) -> (IBig, Vec<IBig>) {
    let mut level: Vec<IBig> = t.iter().map(|&c| IBig::from(c)).collect();
    // `t_i(-X)` at each level `i` of the descent.
    let mut conjugates: Vec<Vec<IBig>> = Vec::new();
    while level.len() > 1 {
        let half: usize = level.len() / 2;
        let even: Vec<IBig> = level.iter().step_by(2).cloned().collect();
        let odd: Vec<IBig> = level.iter().skip(1).step_by(2).cloned().collect();
        let even_sq: Vec<IBig> = negacyclic_mul(&even, &even);
        let odd_sq: Vec<IBig> = negacyclic_mul(&odd, &odd);
        let next: Vec<IBig> = (0..half)
            .map(|i| {
                // `Y * odd_sq` in `Z[Y]/(Y^half + 1)`.
                let y_odd_sq: IBig = if i == 0 { -&odd_sq[half - 1] } else { odd_sq[i - 1].clone() };
                &even_sq[i] - y_odd_sq
            })
            .collect();
        let conjugate: Vec<IBig> = level
            .into_iter()
            .enumerate()
            .map(|(i, c)| if i % 2 == 1 { -c } else { c })
            .collect();
        conjugates.push(conjugate);
        level = next;
    }
    let norm: IBig = level.pop().unwrap_or(IBig::ZERO);

    let mut adjugate: Vec<IBig> = vec![IBig::ONE];
    for conjugate in conjugates.iter().rev() {
        adjugate = negacyclic_mul(conjugate, &adjugate);
    }
    (norm, adjugate)
}

/// `a * b` in `Z[X]/(X^n + 1)`, `n = max(a.len(), b.len())`, both lengths powers of two:
/// the operand of smaller degree `m` stands for its embedding `X -> X^(n/m)`.
fn negacyclic_mul(a: &[IBig], b: &[IBig]) -> Vec<IBig> {
    let (a, b): (&[IBig], &[IBig]) = if a.len() >= b.len() { (a, b) } else { (b, a) };
    let n: usize = a.len();
    let stride: usize = n / b.len();
    let mut res: Vec<IBig> = vec![IBig::ZERO; n];
    for (j, bj) in b.iter().enumerate() {
        if *bj == IBig::ZERO {
            continue;
        }
        for (i, ai) in a.iter().enumerate() {
            let prod: IBig = ai * bj;
            let e: usize = i + stride * j;
            if e < n {
                res[e] += prod;
            } else {
                res[e - n] -= prod;
            }
        }
    }
    res
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Checks `t * numerator = characteristic` and that the fraction is irreducible.
    fn assert_inverse(t: &GBFVPlaintextModulus) {
        let product: Vec<IBig> = negacyclic_mul(t.numerator(), &ints(t.ptxt_mod()));
        assert_eq!(product[0], IBig::from(t.characteristic().clone()));
        assert!(product[1..].iter().all(|c| *c == IBig::ZERO));
        let gcd: UBig = t
            .numerator()
            .iter()
            .fold(t.characteristic().clone(), |g, c| g.gcd(c.unsigned_abs()));
        assert_eq!(gcd, UBig::ONE);
    }

    fn ints(coeffs: &[i64]) -> Vec<IBig> {
        coeffs.iter().map(|&c| IBig::from(c)).collect()
    }

    #[test]
    fn magma_reference_modulus() {
        // `X^32 - 2` at degree 512 lives in degree 16 as `X - 2`, of norm `2^16 + 1`.
        let mut coeffs: Vec<i64> = vec![0; 512];
        coeffs[0] = -2;
        coeffs[32] = 1;
        let t: GBFVPlaintextModulus = GBFVPlaintextModulus::new(&coeffs).unwrap();
        assert_eq!(t.n(), 16);
        assert_eq!(t.log_n(), 4);
        let mut expected: Vec<i64> = vec![0; 16];
        expected[0] = -2;
        expected[1] = 1;
        assert_eq!(t.ptxt_mod(), expected.as_slice());
        assert_eq!(*t.characteristic(), UBig::from(65537u32));
        assert_inverse(&t);
    }

    #[test]
    fn constant_modulus() {
        let t: GBFVPlaintextModulus = GBFVPlaintextModulus::new(&[7, 0, 0, 0]).unwrap();
        assert_eq!(t.ptxt_mod(), &[7]);
        assert_eq!(*t.characteristic(), UBig::from(7u32));
        assert_eq!(t.numerator(), ints(&[1]).as_slice());

        let t: GBFVPlaintextModulus = GBFVPlaintextModulus::new(&[-7]).unwrap();
        assert_eq!(*t.characteristic(), UBig::from(7u32));
        assert_eq!(t.numerator(), ints(&[-1]).as_slice());
    }

    #[test]
    fn gaussian_integer() {
        // 1/(3 - 2i) = (3 + 2i)/13.
        let t: GBFVPlaintextModulus = GBFVPlaintextModulus::new(&[3, -2]).unwrap();
        assert_eq!(*t.characteristic(), UBig::from(13u32));
        assert_eq!(t.numerator(), ints(&[3, 2]).as_slice());
    }

    #[test]
    fn simplifies_fraction() {
        // N(2 + 2i) = 8 and adj = 2 - 2i, so 1/(2 + 2i) = (1 - i)/4.
        let t: GBFVPlaintextModulus = GBFVPlaintextModulus::new(&[2, 2]).unwrap();
        assert_eq!(*t.characteristic(), UBig::from(4u32));
        assert_eq!(t.numerator(), ints(&[1, -1]).as_slice());
        assert_inverse(&t);
    }

    #[test]
    fn inverse_of_dense_elements() {
        let cases: [&[i64]; 5] = [
            &[1, 1, 0, 0],
            &[2, -1, 1, 0],
            &[3, 1, -4, 1, 5, -9, 2, 6],
            &[6, 0, 4, 0, -2, 0, 8, 0],
            &[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16],
        ];
        for coeffs in cases {
            assert_inverse(&GBFVPlaintextModulus::new(coeffs).unwrap());
        }
    }

    #[test]
    fn rejects_zero() {
        assert!(GBFVPlaintextModulus::new(&[0; 8]).is_err());
    }

    #[test]
    fn rejects_unit() {
        assert!(GBFVPlaintextModulus::new(&[0, 1]).is_err());
        assert!(GBFVPlaintextModulus::new(&[-1]).is_err());
        // 1 - X + X^2 = (1 + X^3) / (1 + X), both of norm 2.
        assert!(GBFVPlaintextModulus::new(&[1, -1, 1, 0]).is_err());
    }

    #[test]
    fn rejects_non_power_of_two_degree() {
        assert!(GBFVPlaintextModulus::new(&[1, 2, 3]).is_err());
        assert!(GBFVPlaintextModulus::new(&[]).is_err());
    }

    /// `X^32 - 2` at degree 512, as in the Magma reference parameters.
    fn magma_modulus() -> GBFVPlaintextModulus {
        let mut coeffs: Vec<i64> = vec![0; 512];
        coeffs[0] = -2;
        coeffs[32] = 1;
        GBFVPlaintextModulus::new(&coeffs).unwrap()
    }

    /// Deterministic coefficients in `[-bound, bound]`.
    fn pseudo_random(len: usize, seed: u64, bound: i64) -> Vec<IBig> {
        let mut state: u64 = seed;
        (0..len)
            .map(|_| {
                state = state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
                IBig::from((state >> 33) as i64 % (2 * bound + 1) - bound)
            })
            .collect()
    }

    /// Magma's `Flatten`, on the unsimplified inverse `adj(t) / N(t)`.
    fn magma_flatten(t: &GBFVPlaintextModulus, a: &[IBig]) -> Vec<IBig> {
        let (norm, adjugate): (IBig, Vec<IBig>) = norm_and_adjugate(t.ptxt_mod());
        let reduced: Vec<IBig> = negacyclic_mul(a, &adjugate)
            .into_iter()
            .map(|c| centered_rem(c, &norm))
            .collect();
        negacyclic_mul(&reduced, &ints(t.ptxt_mod()))
            .into_iter()
            .map(|c| c / &norm)
            .collect()
    }

    #[test]
    fn flatten_of_integer_modulus_is_centered_reduction() {
        let t: GBFVPlaintextModulus = GBFVPlaintextModulus::new(&[7]).unwrap();
        let a: Vec<IBig> = ints(&[10, -4, 3, 7, 0, -10, 13, 4]);
        assert_eq!(t.flatten(&a).unwrap(), ints(&[3, 3, 3, 0, 0, -3, -1, -3]));

        // Ties go to the lower end, as in Magma's `CenteredReduction`.
        let t: GBFVPlaintextModulus = GBFVPlaintextModulus::new(&[4]).unwrap();
        assert_eq!(t.flatten(&ints(&[2, -2, 6, 5])).unwrap(), ints(&[-2, -2, -2, 1]));
    }

    #[test]
    fn flatten_matches_magma() {
        // `2 + 2i` and `6 + 4X^2 - 2X^4 + 8X^6` have a common factor between
        // `adj(t)` and `N(t)`, so Magma's denominator differs from the characteristic.
        let moduli: [&[i64]; 4] = [&[2, 2], &[3, -2], &[6, 0, 4, 0, -2, 0, 8, 0], &[3, 1, -4, 1, 5, -9, 2, 6]];
        for (seed, coeffs) in moduli.into_iter().enumerate() {
            let t: GBFVPlaintextModulus = GBFVPlaintextModulus::new(coeffs).unwrap();
            let a: Vec<IBig> = pseudo_random(16, seed as u64, 1 << 20);
            assert_eq!(t.flatten(&a).unwrap(), magma_flatten(&t, &a));
        }
        let t: GBFVPlaintextModulus = magma_modulus();
        let a: Vec<IBig> = pseudo_random(512, 7, 1 << 40);
        assert_eq!(t.flatten(&a).unwrap(), magma_flatten(&t, &a));
    }

    #[test]
    fn flatten_is_canonical_representative() {
        let t: GBFVPlaintextModulus = magma_modulus();
        let characteristic: IBig = IBig::from(t.characteristic().clone());
        let a: Vec<IBig> = pseudo_random(512, 1, 1 << 40);
        let r: Vec<IBig> = t.flatten(&a).unwrap();

        // `a - r` is a multiple of `t`: `(a - r) * numerator` is divisible by the characteristic.
        let diff: Vec<IBig> = a.iter().zip(&r).map(|(x, y)| x - y).collect();
        assert!(
            negacyclic_mul(&diff, t.numerator())
                .iter()
                .all(|c| c % &characteristic == IBig::ZERO)
        );

        // `r` is its own representative, and so is any other element of `a + (t)`.
        assert_eq!(t.flatten(&r).unwrap(), r);
        let b: Vec<IBig> = pseudo_random(512, 2, 1 << 20);
        let shifted: Vec<IBig> = a
            .iter()
            .zip(negacyclic_mul(&b, &ints(t.ptxt_mod())))
            .map(|(x, y)| x + y)
            .collect();
        assert_eq!(t.flatten(&shifted).unwrap(), r);
    }

    #[test]
    fn flatten_embeds_smaller_operand() {
        // An input of degree 4 under a modulus of degree 16 stands for its embedding.
        let t: GBFVPlaintextModulus = magma_modulus();
        let a: Vec<IBig> = ints(&[123_456, -98_765, 4_321, 77_777]);
        let mut embedded: Vec<IBig> = vec![IBig::ZERO; 16];
        for (i, c) in a.iter().enumerate() {
            embedded[4 * i] = c.clone();
        }
        let r: Vec<IBig> = t.flatten(&a).unwrap();
        assert_eq!(r.len(), 16);
        assert_eq!(r, t.flatten(&embedded).unwrap());
    }

    #[test]
    fn flatten_rejects_non_power_of_two_degree() {
        assert!(magma_modulus().flatten(&ints(&[1, 2, 3])).is_err());
        assert!(magma_modulus().flatten(&[]).is_err());
    }
}
