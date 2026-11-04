//! The GBFV batch encoder.

use anyhow::{Result, ensure};
use dashu_int::{UBig, fast_div::ConstDivisor};

use super::GBFVPlaintextModulus;

/// GBFV batch encoder over a prime characteristic `p`.
///
/// Holds one primitive `2n`-th root of unity `w` modulo `p^e`, for the maximum ring
/// degree `n` and the maximum Hensel lifting exponent `e`, that is also a root of the
/// plaintext modulus `t`, embedded in degree `n` as `t(X^(n/n_t))`. Plaintext moduli of
/// characteristic `p^k`, `k <= e`, that vanish at this root, such as the powers of `t`
/// and `p`, encode with it, so their slots line up.
///
/// The root is fixed as a function of `p`, `n` and `t`:
/// - modulo `p`, `w = w_0^k` with `w_0 = g^((p - 1) / 2n)`, `g` the smallest quadratic
///   non-residue modulo `p`, so that `w_0^n = g^((p - 1) / 2) = -1`, and `k` the smallest
///   odd exponent such that `t(w^(n/n_t)) = 0`. The root `theta` of `t` modulo `p` is
///   unique, so `k` is determined modulo `2 n_t` by `(w_0^(n/n_t))^k = theta`, lies in
///   `[1, 2 n_t)` and does not depend on `n`;
/// - modulo `p^e`, `w` is the Hensel lift of that root of `X^n + 1`, computed as
///   `w^(p^(e - 1))`: the unique lift of a root of unity of order dividing `p - 1`.
///
/// The roots of smaller degrees are compatible: the root for degree `n_t <= n' <= n` is
/// `w^(n/n')`, the same root that an encoder of maximum degree `n'` holds.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct GBFVBatchEncoder {
    /// Prime characteristic `p`.
    characteristic: UBig,
    /// Maximum Hensel lifting exponent `e`.
    max_hensel_exponent: usize,
    /// Base-2 logarithm of the maximum ring degree `n`.
    log_n: usize,
    /// Primitive `2n`-th root of unity modulo `p^e`, in `[0, p^e)`.
    root_of_unity: UBig,
}

impl GBFVBatchEncoder {
    /// Builds the batch encoder for the characteristic `p` of `ptxt_mod`, the maximum
    /// Hensel lifting exponent `max_hensel_exponent` and the maximum ring degree
    /// `n = 2^log_n`, and precomputes its root of unity.
    ///
    /// Errors when:
    /// - `max_hensel_exponent` is zero;
    /// - `t` does not live in the ring of degree `n`;
    /// - `p` is not `1` modulo `2n`, or not prime;
    /// - the norm of `t` is not `p`, so that `t` is not a prime of degree one.
    ///
    /// The primality test is deterministic Miller-Rabin over the first 13 prime bases:
    /// a proof below `3.3 * 10^24`, a strong probable-prime test above.
    pub fn new(ptxt_mod: &GBFVPlaintextModulus, max_hensel_exponent: usize, log_n: usize) -> Result<Self> {
        ensure!(
            max_hensel_exponent > 0,
            "batch encoder: hensel lifting exponent must be positive"
        );
        ensure!(log_n < usize::BITS as usize, "batch encoder: ring degree exceeds usize");
        ensure!(
            ptxt_mod.n() <= 1 << log_n,
            "batch encoder: plaintext modulus must live in the ring of degree n"
        );
        let characteristic: UBig = ptxt_mod.characteristic().clone();
        // `p = 1 mod 2^(log_n + 1)` when `p - 1` has more than `log_n` trailing zeros.
        ensure!(
            (&characteristic - UBig::ONE).trailing_zeros().unwrap_or(0) > log_n,
            "batch encoder: characteristic must be 1 modulo 2n"
        );
        ensure!(
            *ptxt_mod.norm() == characteristic,
            "batch encoder: norm of the plaintext modulus must equal its characteristic"
        );
        ensure!(is_prime(&characteristic), "batch encoder: characteristic must be prime");

        let root: UBig = root_of_unity(&characteristic, log_n, ptxt_mod.ptxt_mod())?;
        let modulus: UBig = characteristic.pow(max_hensel_exponent);
        let ring: ConstDivisor = ConstDivisor::new(modulus);
        let root_of_unity: UBig = ring.reduce(root).pow(&characteristic.pow(max_hensel_exponent - 1)).residue();

        Ok(Self {
            characteristic,
            max_hensel_exponent,
            log_n,
            root_of_unity,
        })
    }

    /// Prime characteristic `p`.
    pub fn characteristic(&self) -> &UBig {
        &self.characteristic
    }

    /// Maximum Hensel lifting exponent `e`.
    pub fn max_hensel_exponent(&self) -> usize {
        self.max_hensel_exponent
    }

    /// Base-2 logarithm of the maximum ring degree.
    pub fn log_n(&self) -> usize {
        self.log_n
    }

    /// Maximum ring degree `n`.
    pub fn n(&self) -> usize {
        1 << self.log_n
    }

    /// Primitive `2n`-th root of unity modulo `p^e`, in `[0, p^e)`.
    pub fn root_of_unity(&self) -> &UBig {
        &self.root_of_unity
    }
}

/// `w_0^k mod p`, with `w_0 = g^((p - 1) / 2n)`, `n = 2^log_n`, `g` the smallest
/// quadratic non-residue modulo the prime `p`, and `k` the smallest odd exponent with
/// `t(w_0^(k n/n_t)) = 0 mod p`, `t` given by its `n_t <= n` coefficients and `2n`
/// dividing `p - 1`.
///
/// Each non-residue candidate `a` costs one exponentiation: `a^((p - 1) / 2n)` has
/// `n`-th power `-1` exactly when `a` is a non-residue. Under GRH the smallest
/// non-residue is below `2 ln(p)^2`. The exponent `k` is searched among the odd powers
/// of the primitive `2 n_t`-th root `w_0^(n/n_t)`, by Horner evaluation of `t`: at most
/// `n_t` evaluations of `n_t` steps.
fn root_of_unity(p: &UBig, log_n: usize, t: &[i64]) -> Result<UBig> {
    let ring: ConstDivisor = ConstDivisor::new(p.clone());
    let p_minus_one: UBig = p - UBig::ONE;
    let exponent: UBig = &p_minus_one >> (log_n + 1);
    let n: usize = 1 << log_n;
    let minus_one = ring.reduce(p_minus_one);
    let generator = (2u64..)
        .take_while(|&a| UBig::from(a) < *p)
        .map(|a| ring.reduce(a).pow(&exponent))
        .find(|w| w.pow(&UBig::from(n)) == minus_one)
        .ok_or_else(|| anyhow::anyhow!("batch encoder: no quadratic non-residue found"))?;

    let zero = ring.reduce(0u8);
    let zeta = generator.pow(&UBig::from(n / t.len()));
    let zeta_squared = &zeta * &zeta;
    let mut candidate = zeta;
    for k in (1..2 * t.len()).step_by(2) {
        let value = t.iter().rev().fold(zero.clone(), |acc, &c| acc * &candidate + ring.reduce(c));
        if value == zero {
            return Ok(generator.pow(&UBig::from(k)).residue());
        }
        candidate *= &zeta_squared;
    }
    anyhow::bail!("batch encoder: plaintext modulus has no root of order 2n")
}

/// Deterministic Miller-Rabin over the first 13 prime bases: exact below
/// `3.317 * 10^24`, a strong probable-prime test above.
fn is_prime(n: &UBig) -> bool {
    const BASES: [u8; 13] = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41];
    if *n < UBig::from(2u8) {
        return false;
    }
    for b in BASES {
        let b: UBig = UBig::from(b);
        if *n == b {
            return true;
        }
        if n % &b == UBig::ZERO {
            return false;
        }
    }
    let n_minus_one: UBig = n - UBig::ONE;
    let s: usize = n_minus_one.trailing_zeros().unwrap_or(0);
    let d: UBig = &n_minus_one >> s;
    let ring: ConstDivisor = ConstDivisor::new(n.clone());
    let one = ring.reduce(1u8);
    let minus_one = ring.reduce(n_minus_one);
    BASES.iter().all(|&b| {
        let mut x = ring.reduce(b).pow(&d);
        if x == one || x == minus_one {
            return true;
        }
        (1..s).any(|_| {
            x = &x * &x;
            x == minus_one
        })
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use dashu_int::IBig;

    /// `X^32 - 2` at degree 512, of characteristic `2^16 + 1`.
    fn magma_modulus() -> GBFVPlaintextModulus {
        let mut coeffs: Vec<i64> = vec![0; 512];
        coeffs[0] = -2;
        coeffs[32] = 1;
        GBFVPlaintextModulus::new(&coeffs).unwrap()
    }

    /// `w^k mod m`.
    fn pow_mod(w: &UBig, k: &UBig, m: &UBig) -> UBig {
        ConstDivisor::new(m.clone()).reduce(w.clone()).pow(k).residue()
    }

    /// Checks that the root is a primitive `2n`-th root of unity modulo `p^e`, and that it
    /// is a root of `t(X^(n/n_t))` modulo `p`.
    fn assert_root(encoder: &GBFVBatchEncoder, t: &GBFVPlaintextModulus) {
        let p: &UBig = encoder.characteristic();
        let modulus: UBig = p.pow(encoder.max_hensel_exponent());
        let n: UBig = UBig::from(encoder.n());
        assert!(*encoder.root_of_unity() < modulus);
        assert_eq!(pow_mod(encoder.root_of_unity(), &n, &modulus), &modulus - UBig::ONE);

        let x: UBig = pow_mod(encoder.root_of_unity(), &UBig::from(encoder.n() / t.n()), p);
        let value: IBig = t
            .ptxt_mod()
            .iter()
            .enumerate()
            .map(|(i, &c)| IBig::from(c) * IBig::from(pow_mod(&x, &UBig::from(i), p)))
            .sum();
        assert_eq!(value % IBig::from(p.clone()), IBig::ZERO);
    }

    #[test]
    fn magma_reference_root() {
        let p: UBig = UBig::from(65537u32);
        let t: GBFVPlaintextModulus = magma_modulus();
        let encoder: GBFVBatchEncoder = GBFVBatchEncoder::new(&t, 2, 9).unwrap();
        assert_root(&encoder, &t);

        // 3 is the smallest non-residue modulo `2^16 + 1`, so `w_0 = 3^(2^16 / 1024)`, and
        // `k` is the smallest odd exponent with `w_0^(32 k) = 2`, the root of `X - 2`.
        let generator: UBig = pow_mod(&UBig::from(3u8), &UBig::from(64u8), &p);
        let k: usize = (1..)
            .step_by(2)
            .find(|&k| pow_mod(&generator, &UBig::from(32 * k), &p) == UBig::from(2u8))
            .unwrap();
        let root_mod_p: UBig = pow_mod(&generator, &UBig::from(k), &p);
        assert_eq!(encoder.root_of_unity() % &p, root_mod_p);

        // Same lift as the Newton step of Magma's `FindRoot_ft`:
        // `w - p * ((w^n + 1) / p) * (n w^(n - 1))^(-1) mod p^2`.
        let n: UBig = UBig::from(512u32);
        let p2: UBig = &p * &p;
        let w: UBig = root_mod_p;
        let correction: UBig = (pow_mod(&w, &n, &p2) + UBig::ONE) / &p;
        let derivative: UBig = (&n * pow_mod(&w, &(&n - UBig::ONE), &p)) % &p;
        let inverse: UBig = pow_mod(&derivative, &(&p - UBig::from(2u8)), &p);
        let update: UBig = (correction * inverse) % &p;
        let newton: UBig = (&w + &p2 - (&p * update) % &p2) % &p2;
        assert_eq!(*encoder.root_of_unity(), newton);
    }

    #[test]
    fn roots_of_dense_moduli() {
        // `3 - 2i`, of norm 13, and `-2 - X - X^2 + X^3` in `Z[X]/(X^4 + 1)`, of norm 17.
        let cases: [(&[i64], usize); 2] = [(&[3, -2], 1), (&[-2, -1, -1, 1], 3)];
        for (coeffs, max_log_n) in cases {
            let t: GBFVPlaintextModulus = GBFVPlaintextModulus::new(coeffs).unwrap();
            for log_n in t.log_n()..=max_log_n {
                assert_root(&GBFVBatchEncoder::new(&t, 2, log_n).unwrap(), &t);
            }
        }
    }

    #[test]
    fn roots_are_compatible_across_degrees() {
        let t: GBFVPlaintextModulus = magma_modulus();
        let modulus: UBig = UBig::from(65537u32).pow(3);
        let large: GBFVBatchEncoder = GBFVBatchEncoder::new(&t, 3, 9).unwrap();
        for log_n in t.log_n()..9 {
            let small: GBFVBatchEncoder = GBFVBatchEncoder::new(&t, 3, log_n).unwrap();
            assert_root(&small, &t);
            let ratio: UBig = UBig::from(1u32 << (9 - log_n));
            assert_eq!(pow_mod(large.root_of_unity(), &ratio, &modulus), *small.root_of_unity());
        }
    }

    #[test]
    fn roots_are_compatible_across_exponents() {
        let t: GBFVPlaintextModulus = magma_modulus();
        let high: GBFVBatchEncoder = GBFVBatchEncoder::new(&t, 4, 9).unwrap();
        for e in 1..4 {
            let low: GBFVBatchEncoder = GBFVBatchEncoder::new(&t, e, 9).unwrap();
            let modulus: UBig = UBig::from(65537u32).pow(e);
            assert_eq!(high.root_of_unity() % &modulus, *low.root_of_unity());
        }
    }

    #[test]
    fn integer_modulus() {
        // BFV: `t = 29 * 2^57 + 1`, a prime that is 1 modulo `2^57`.
        let t: GBFVPlaintextModulus = GBFVPlaintextModulus::new(&[4_179_340_454_199_820_289]).unwrap();
        let encoder: GBFVBatchEncoder = GBFVBatchEncoder::new(&t, 3, 16).unwrap();
        assert_root(&encoder, &t);
    }

    #[test]
    fn rejects_invalid_parameters() {
        let t: GBFVPlaintextModulus = magma_modulus();
        // `2^16 + 1` is not 1 modulo `2^18`.
        assert!(GBFVBatchEncoder::new(&t, 2, 17).is_err());
        assert!(GBFVBatchEncoder::new(&t, 2, 1000).is_err());
        assert!(GBFVBatchEncoder::new(&t, 0, 9).is_err());
        // `X - 2` lives in degree 16, above the ring degree 8.
        assert!(GBFVBatchEncoder::new(&t, 2, 3).is_err());
        // `1649 = 17 * 97` is 1 modulo 16 but not prime.
        let composite: GBFVPlaintextModulus = GBFVPlaintextModulus::new(&[1649]).unwrap();
        assert!(GBFVBatchEncoder::new(&composite, 1, 3).is_err());
        // `3 - 2X - 2X^3` is the product of two distinct primes above 17: its
        // characteristic is 17, its norm `17^2`.
        let product: GBFVPlaintextModulus = GBFVPlaintextModulus::new(&[3, -2, 0, -2]).unwrap();
        assert_eq!(*product.characteristic(), UBig::from(17u8));
        assert!(GBFVBatchEncoder::new(&product, 1, 2).is_err());
    }

    #[test]
    fn primality() {
        let primes: [UBig; 6] = [
            UBig::from(2u8),
            UBig::from(65537u32),
            UBig::from((1u64 << 61) - 1),
            UBig::from(4_179_340_454_199_820_289u64),
            UBig::from(18_446_744_073_709_551_557u64),
            (UBig::ONE << 89) - UBig::ONE,
        ];
        let composites: [UBig; 6] = [
            UBig::ZERO,
            UBig::ONE,
            UBig::from(561u32),
            UBig::from(3_215_031_751u64),
            UBig::from(18_446_744_073_709_551_615u64),
            (UBig::ONE << 64) + UBig::ONE,
        ];
        assert!(primes.iter().all(is_prime));
        assert!(composites.iter().all(|c| !is_prime(c)));
    }
}
