//! Arithmetic in the Galois ring `GR(2^K, d) = Z_{2^K}[y] / (f(y))`, over
//! which Shamir sharing runs: `f` is irreducible modulo 2, so points with
//! distinct binary coefficients differ by a unit.

use dashu_int::UBig;

/// Low coefficients `f_0..f_{d-1}` of the monic `f`, as a bit mask, for
/// `d = 1..=8`: `y`, `y^2+y+1`, `y^3+y+1`, `y^4+y+1`, `y^5+y^2+1`, `y^6+y+1`,
/// `y^7+y+1`, `y^8+y^4+y^3+y+1`.
const MODULI: [u16; 8] = [0b0, 0b11, 0b11, 0b11, 0b101, 0b11, 0b11, 0b1_1011];

/// The low coefficients of the degree-`d` modulus.
pub(crate) fn modulus_low(d: usize) -> u16 {
    MODULI[d - 1]
}

/// An element of the ring: `d` coefficients modulo `2^K`, lowest power first.
pub(crate) type GrElement = Vec<UBig>;

pub(crate) struct GaloisRing {
    d: usize,
    low: u16,
    k: usize,
    modulus: UBig,
    mask: UBig,
}

impl GaloisRing {
    pub(crate) fn new(k: usize, d: usize) -> Self {
        assert!((1..=8).contains(&d), "invalid Galois ring: degree outside 1..=8");
        let modulus = UBig::ONE << k;
        let mask = &modulus - UBig::ONE;
        Self {
            d,
            low: modulus_low(d),
            k,
            modulus,
            mask,
        }
    }

    fn constant(&self, c: u64) -> GrElement {
        let mut e = vec![UBig::ZERO; self.d];
        e[0] = UBig::from(c) & &self.mask;
        e
    }

    /// The point of party `i`: the element whose coefficients are the bits of `i`.
    pub(crate) fn point(&self, i: u32) -> GrElement {
        assert!(
            i >= 1 && (i as usize) < (1 << self.d),
            "invalid party point: out of the Galois ring's range"
        );
        (0..self.d).map(|b| UBig::from((i >> b) & 1)).collect()
    }

    fn sub(&self, a: &GrElement, b: &GrElement) -> GrElement {
        a.iter().zip(b).map(|(x, y)| (x + &self.modulus - y) & &self.mask).collect()
    }

    pub(crate) fn mul(&self, a: &GrElement, b: &GrElement) -> GrElement {
        let d = self.d;
        let mut prod = vec![UBig::ZERO; 2 * d - 1];
        for (i, x) in a.iter().enumerate() {
            for (j, y) in b.iter().enumerate() {
                prod[i + j] = (&prod[i + j] + x * y) & &self.mask;
            }
        }
        // y^m = -y^(m-d) (f_0 + ... + f_{d-1} y^(d-1)) for m >= d.
        for m in (d..2 * d - 1).rev() {
            let top = std::mem::take(&mut prod[m]);
            for j in (0..d).filter(|j| self.low >> j & 1 == 1) {
                prod[m - d + j] = (&prod[m - d + j] + &self.modulus - &top) & &self.mask;
            }
        }
        prod.truncate(d);
        prod
    }

    fn mul_y(&self, a: &GrElement) -> GrElement {
        if self.d == 1 {
            // In degree 1, y reduces to -f_0 = 0.
            return self.constant(0);
        }
        let mut y = vec![UBig::ZERO; self.d];
        y[1] = UBig::ONE;
        self.mul(a, &y)
    }

    /// The inverse of a unit: its inverse modulo 2, lifted by Newton's iteration.
    pub(crate) fn inv(&self, u: &GrElement) -> GrElement {
        let one = UBig::ONE;
        let mut v = (1u32..1 << self.d)
            .map(|i| self.point(i))
            .find(|v| {
                self.mul(u, v)
                    .iter()
                    .enumerate()
                    .all(|(j, c)| (c & &one) == UBig::from((j == 0) as u8))
            })
            .expect("invalid Galois ring element: not a unit");
        let two = self.constant(2);
        let mut bits = 1;
        while bits < self.k {
            v = self.mul(&v, &self.sub(&two, &self.mul(u, &v)));
            bits *= 2;
        }
        v
    }

    /// The Lagrange coefficient at 0 of party `own` among `actives`.
    pub(crate) fn lagrange(&self, own: u32, actives: &[u32]) -> GrElement {
        let x_own = self.point(own);
        actives.iter().filter(|&&a| a != own).fold(self.constant(1), |acc, &a| {
            let x_a = self.point(a);
            let factor = self.mul(&x_a, &self.inv(&self.sub(&x_a, &x_own)));
            self.mul(&acc, &factor)
        })
    }

    /// `[lambda * y^j]_0` for `j < d`: the weights of the components of a share
    /// in the constant component of its product by `lambda`.
    pub(crate) fn constant_terms(&self, lambda: &GrElement) -> Vec<UBig> {
        let mut e = lambda.clone();
        (0..self.d)
            .map(|_| {
                let c = e[0].clone();
                e = self.mul_y(&e);
                c
            })
            .collect()
    }

    /// The `count` lowest balanced base-`2^base2k` digits of `c`, lowest first.
    pub(crate) fn digits(&self, c: &UBig, base2k: usize, count: usize) -> Vec<i64> {
        let digit_mask = UBig::from((1u64 << base2k) - 1);
        let half = 1i64 << (base2k - 1);
        let mut x = c.clone();
        (0..count)
            .map(|_| {
                let mut digit = u64::try_from(&x & &digit_mask).unwrap() as i64;
                x >>= base2k;
                if digit >= half {
                    digit -= 1 << base2k;
                    x += UBig::ONE;
                }
                digit
            })
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use dashu_int::IBig;

    use super::*;

    fn is_one(gr: &GaloisRing, e: &GrElement) -> bool {
        *e == gr.constant(1)
    }

    #[test]
    fn point_differences_are_units() {
        for d in 1..=8 {
            let gr = GaloisRing::new(73, d);
            let points: Vec<u32> = (1..1 << d).collect();
            for &a in &points {
                assert!(is_one(&gr, &gr.mul(&gr.point(a), &gr.inv(&gr.point(a)))));
                for &b in points.iter().filter(|&&b| b != a).take(8) {
                    let u = gr.sub(&gr.point(a), &gr.point(b));
                    assert!(is_one(&gr, &gr.mul(&u, &gr.inv(&u))));
                }
            }
        }
    }

    #[test]
    fn lagrange_coefficients_sum_to_one() {
        let gr = GaloisRing::new(73, 3);
        for actives in [&[1u32, 2, 3][..], &[2, 4, 5, 7]] {
            let sum = actives.iter().fold(gr.constant(0), |acc, &own| {
                let lambda = gr.lagrange(own, actives);
                acc.iter().zip(&lambda).map(|(x, y)| (x + y) & &gr.mask).collect()
            });
            assert!(is_one(&gr, &sum));
        }
    }

    #[test]
    fn digits_recombine() {
        let gr = GaloisRing::new(73, 3);
        let c = (UBig::ONE << 72) + UBig::from(0xDEAD_BEEF_u64);
        let digits = gr.digits(&c, 12, 7);
        let value = digits.iter().rev().fold(IBig::ZERO, |acc, &x| (acc << 12) + IBig::from(x));
        let modulus = IBig::from(gr.modulus.clone());
        assert_eq!(((value % &modulus) + &modulus) % &modulus, IBig::from(c));
    }
}
