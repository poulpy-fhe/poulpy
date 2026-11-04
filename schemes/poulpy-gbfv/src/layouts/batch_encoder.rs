//! The GBFV batch encoder.

use anyhow::{Result, ensure};
use dashu_int::{IBig, UBig, fast_div::ConstDivisor};
use poulpy_hal::layouts::{Backend, Module};

use super::{GBFVNttBuffer, GBFVNttInfos, GBFVNttPlan, GBFVPlaintextModulus, bit_reverse};
use crate::api::GBFVNttOps;

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

    /// Encodes `slots` modulo `p^e` into the polynomial of degree less than
    /// `n' = slots.len() * sparsity` that takes them in the slots of `(Z/p^e)[X]/(X^n' + 1)`.
    ///
    /// The slots of degree `n'` are the values at the roots `w'^(e(pos))` of `X^n' + 1`,
    /// `w' = w^(n/n')`, in the order `pos = 2 i + j -> e(pos) = (-1)^j 5^i mod 2n'`
    /// (`e(0) = 1` for `n' = 1`). Value `k` of `slots` goes to position `k * sparsity` and
    /// the other positions are zero. The plaintext modulus plays no part: the result is
    /// not reduced modulo `t`. Its `n'` coefficients are returned in `[0, p^e)`.
    ///
    /// Errors when `sparsity` is zero, or `n'` is not a power of two at most `n`.
    pub fn embed_in_slots<BE>(&self, module: &Module<BE>, slots: &[IBig], sparsity: usize) -> Result<Vec<UBig>>
    where
        BE: Backend,
        Module<BE>: GBFVNttOps<BE>,
    {
        ensure!(sparsity > 0, "batch encoder: sparsity must be positive");
        let n: usize = slots
            .len()
            .checked_mul(sparsity)
            .ok_or_else(|| anyhow::anyhow!("batch encoder: ring degree overflows usize"))?;
        let (plan, modulus) = self.ntt_plan::<BE>(n)?;
        let indices: Vec<usize> = slot_ntt_indices(plan.log_n());
        let mut values: Vec<UBig> = vec![UBig::ZERO; n];
        for (k, slot) in slots.iter().enumerate() {
            values[indices[k * sparsity]] = reduce(slot, &modulus);
        }
        let mut buffer: GBFVNttBuffer<BE::OwnedBuf> = GBFVNttBuffer::from_host::<BE>(&values, plan.limbs())?;
        module.gbfv_ntt_inverse_assign(&plan, &mut buffer)?;
        Ok(buffer.to_host::<BE>())
    }

    /// Retrieves the slots modulo `p^e` of `poly` in `(Z/p^e)[X]/(X^n' + 1)`,
    /// `n' = poly.len()`: the values at positions `0, sparsity, 2 sparsity, ...` in the
    /// order of [`Self::embed_in_slots`], in `[0, p^e)`. The plaintext modulus plays no
    /// part.
    ///
    /// Errors when `n'` is not a power of two at most `n`, or `sparsity` is zero or does
    /// not divide `n'`.
    pub fn retrieve_slots<BE>(&self, module: &Module<BE>, poly: &[IBig], sparsity: usize) -> Result<Vec<UBig>>
    where
        BE: Backend,
        Module<BE>: GBFVNttOps<BE>,
    {
        ensure!(
            sparsity > 0 && poly.len().is_multiple_of(sparsity),
            "batch encoder: sparsity must divide the ring degree"
        );
        let (plan, modulus) = self.ntt_plan::<BE>(poly.len())?;
        let indices: Vec<usize> = slot_ntt_indices(plan.log_n());
        let values: Vec<UBig> = poly.iter().map(|c| reduce(c, &modulus)).collect();
        let mut buffer: GBFVNttBuffer<BE::OwnedBuf> = GBFVNttBuffer::from_host::<BE>(&values, plan.limbs())?;
        module.gbfv_ntt_forward_assign(&plan, &mut buffer)?;
        let evaluations: Vec<UBig> = buffer.to_host::<BE>();
        Ok((0..poly.len() / sparsity)
            .map(|k| evaluations[indices[k * sparsity]].clone())
            .collect())
    }

    /// NTT plan of degree `n'` modulo `p^e` for the root `w^(n/n')`, and the modulus `p^e`.
    fn ntt_plan<BE: Backend>(&self, n: usize) -> Result<(GBFVNttPlan<BE::OwnedBuf>, UBig)> {
        ensure!(
            n.is_power_of_two() && n <= self.n(),
            "batch encoder: ring degree must be a power of two at most n"
        );
        let modulus: UBig = self.characteristic.pow(self.max_hensel_exponent);
        let root: UBig = ConstDivisor::new(modulus.clone())
            .reduce(self.root_of_unity.clone())
            .pow(&UBig::from(self.n() / n))
            .residue();
        let plan = GBFVNttPlan::new::<BE>(n.trailing_zeros() as usize, &modulus, &root)?;
        Ok((plan, modulus))
    }
}

/// For every slot position `pos < n = 2^log_n`, the index in the output of the forward
/// NTT of degree `n`, which holds the value at `psi^(2 brv(index) + 1)`, of the root
/// `psi^(e(pos))`, `e(pos) = (-1)^j 5^i mod 2n` for `pos = 2 i + j` (`e(0) = 1` for `n = 1`).
fn slot_ntt_indices(log_n: usize) -> Vec<usize> {
    let two_n: usize = 2 << log_n;
    let index = |e: usize| bit_reverse((e - 1) / 2, log_n);
    if log_n == 0 {
        return vec![index(1)];
    }
    let mut indices: Vec<usize> = Vec::with_capacity(1 << log_n);
    let mut power: usize = 1;
    for _ in 0..(1 << log_n) / 2 {
        indices.push(index(power));
        indices.push(index(two_n - power));
        power = power * 5 % two_n;
    }
    indices
}

/// `x mod m` in `[0, m)`.
fn reduce(x: &IBig, m: &UBig) -> UBig {
    let m: IBig = IBig::from(m.clone());
    UBig::try_from(((x % &m) + &m) % &m).expect("a reduced residue is non-negative")
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
    use dashu_int::ops::BitTest;
    use poulpy_hal::layouts::HostBytesBackend;

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
    fn scalar_modulus_keeps_conventional_root() {
        // A scalar vanishes at every root modulo `p`, so `k = 1` and the root is `w_0`
        // itself, whatever the sign of the scalar and the degree it is given at.
        let encoder = |coeffs: &[i64]| GBFVBatchEncoder::new(&GBFVPlaintextModulus::new(coeffs).unwrap(), 2, 9).unwrap();
        let positive: GBFVBatchEncoder = encoder(&[65537]);
        assert_eq!(positive, encoder(&[-65537]));
        assert_eq!(positive, encoder(&[65537, 0, 0, 0]));
        assert_root(&positive, &GBFVPlaintextModulus::new(&[65537]).unwrap());

        // 3 is the smallest non-residue modulo `2^16 + 1`, so `w_0 = 3^(2^16 / 1024)`.
        let p: UBig = UBig::from(65537u32);
        let generator: UBig = pow_mod(&UBig::from(3u8), &UBig::from(64u8), &p);
        assert_eq!(positive.root_of_unity() % &p, generator);
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

    type BE = HostBytesBackend;

    fn module(encoder: &GBFVBatchEncoder) -> Module<BE> {
        Module::<BE>::new(encoder.n().max(8) as u64)
    }

    /// `29 * 2^57 + 1`, encoded with `e = 3`: `p^e` has 186 bits.
    fn wide_encoder() -> GBFVBatchEncoder {
        let t: GBFVPlaintextModulus = GBFVPlaintextModulus::new(&[4_179_340_454_199_820_289]).unwrap();
        GBFVBatchEncoder::new(&t, 3, 6).unwrap()
    }

    fn modulus(encoder: &GBFVBatchEncoder) -> UBig {
        encoder.characteristic().pow(encoder.max_hensel_exponent())
    }

    /// Deterministic signed values of up to `bits + 8` bits.
    fn pseudo_random(len: usize, seed: u64, bits: usize) -> Vec<IBig> {
        let mut state: u64 = seed;
        let mut next = || {
            state = state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            state >> 32
        };
        (0..len)
            .map(|_| {
                let magnitude: UBig = (0..bits.div_ceil(32) + 1).fold(UBig::ZERO, |acc, _| (acc << 32) + UBig::from(next()));
                let value: IBig = IBig::from(magnitude >> (32 - 8));
                if next() % 2 == 0 { value } else { -value }
            })
            .collect()
    }

    fn reduced(values: &[IBig], q: &UBig) -> Vec<UBig> {
        values.iter().map(|v| reduce(v, q)).collect()
    }

    fn signed(values: &[UBig]) -> Vec<IBig> {
        values.iter().map(|v| IBig::from(v.clone())).collect()
    }

    /// `a(X^k)` in `Z[X]/(X^n + 1)`, `k` odd.
    fn automorphism(a: &[UBig], k: usize) -> Vec<IBig> {
        let n: usize = a.len();
        let mut res: Vec<IBig> = vec![IBig::ZERO; n];
        for (i, c) in a.iter().enumerate() {
            let e: usize = i * k % (2 * n);
            if e < n {
                res[e] += IBig::from(c.clone());
            } else {
                res[e - n] -= IBig::from(c.clone());
            }
        }
        res
    }

    /// `k^(-1) mod m` for odd `k` and a power of two `m`.
    fn inverse_mod(k: usize, m: usize) -> usize {
        (1..m).find(|x| x * k % m == 1).unwrap()
    }

    #[test]
    fn slots_round_trip() {
        // Every degree up to the maximum and every sparsity, as Magma's "SIMD batching" test.
        for encoder in [GBFVBatchEncoder::new(&magma_modulus(), 3, 9).unwrap(), wide_encoder()] {
            let module: Module<BE> = module(&encoder);
            let q: UBig = modulus(&encoder);
            for log_n in 0..=encoder.log_n() {
                for log_sparsity in 0..=log_n {
                    let slots: Vec<IBig> = pseudo_random(1 << (log_n - log_sparsity), log_n as u64, q.bit_len());
                    let poly: Vec<UBig> = encoder.embed_in_slots(&module, &slots, 1 << log_sparsity).unwrap();
                    assert_eq!(poly.len(), 1 << log_n);
                    assert!(poly.iter().all(|c| *c < q));
                    let retrieved: Vec<UBig> = encoder.retrieve_slots(&module, &signed(&poly), 1 << log_sparsity).unwrap();
                    assert_eq!(retrieved, reduced(&slots, &q));
                }
            }
        }
    }

    #[test]
    fn embedding_takes_slot_values() {
        // The encoding of degree 16 at sparsity 2 takes slot `k` at the root of position
        // `2k`, `w'^(5^k mod 32)`, and zero at the roots `w'^(-5^i)` of the odd positions.
        let encoder: GBFVBatchEncoder = GBFVBatchEncoder::new(&magma_modulus(), 3, 9).unwrap();
        let q: UBig = modulus(&encoder);
        let slots: Vec<IBig> = pseudo_random(8, 3, q.bit_len());
        let poly: Vec<UBig> = encoder.embed_in_slots(&module(&encoder), &slots, 2).unwrap();
        let root: UBig = pow_mod(encoder.root_of_unity(), &UBig::from(512u32 / 16), &q);
        let evaluate = |e: usize| {
            let x: UBig = pow_mod(&root, &UBig::from(e), &q);
            poly.iter().rev().fold(UBig::ZERO, |acc, c| (acc * &x + c) % &q)
        };
        let mut power: usize = 1;
        for slot in &slots {
            assert_eq!(evaluate(power), reduce(slot, &q));
            assert_eq!(evaluate(32 - power), UBig::ZERO);
            power = power * 5 % 32;
        }
    }

    #[test]
    fn automorphisms_rotate_slots_as_in_magma() {
        let encoder: GBFVBatchEncoder = GBFVBatchEncoder::new(&magma_modulus(), 2, 9).unwrap();
        let module: Module<BE> = module(&encoder);
        let q: UBig = modulus(&encoder);
        for log_n in 2..=9 {
            let (n, conductor): (usize, usize) = (1 << log_n, 2 << log_n);

            // BFV: `X -> X^(-1)` swaps the pairs, `X -> X^(5^(-1))` rotates by one pair.
            let slots: Vec<UBig> = reduced(&pseudo_random(n, log_n as u64, q.bit_len()), &q);
            let poly: Vec<UBig> = encoder.embed_in_slots(&module, &signed(&slots), 1).unwrap();
            let swapped: Vec<UBig> = slots.chunks(2).flat_map(|pair| [pair[1].clone(), pair[0].clone()]).collect();
            let retrieved: Vec<UBig> = encoder
                .retrieve_slots(&module, &automorphism(&poly, conductor - 1), 1)
                .unwrap();
            assert_eq!(retrieved, swapped);
            let mut rotated: Vec<UBig> = slots.clone();
            rotated.rotate_right(2);
            let retrieved: Vec<UBig> = encoder
                .retrieve_slots(&module, &automorphism(&poly, inverse_mod(5, conductor)), 1)
                .unwrap();
            assert_eq!(retrieved, rotated);

            // GBFV at sparsity 4: `X -> X^(25^(-1))` rotates by one slot.
            if log_n >= 3 {
                let slots: Vec<UBig> = reduced(&pseudo_random(n / 4, log_n as u64, q.bit_len()), &q);
                let poly: Vec<UBig> = encoder.embed_in_slots(&module, &signed(&slots), 4).unwrap();
                let mut rotated: Vec<UBig> = slots.clone();
                rotated.rotate_right(1);
                let retrieved: Vec<UBig> = encoder
                    .retrieve_slots(&module, &automorphism(&poly, inverse_mod(25, conductor)), 4)
                    .unwrap();
                assert_eq!(retrieved, rotated);
            }
        }
    }

    #[test]
    fn slots_multiply_pointwise() {
        let encoder: GBFVBatchEncoder = wide_encoder();
        let module: Module<BE> = module(&encoder);
        let q: UBig = modulus(&encoder);
        let (u, v): (Vec<IBig>, Vec<IBig>) = (pseudo_random(16, 1, q.bit_len()), pseudo_random(16, 2, q.bit_len()));
        let a: Vec<UBig> = encoder.embed_in_slots(&module, &u, 2).unwrap();
        let b: Vec<UBig> = encoder.embed_in_slots(&module, &v, 2).unwrap();
        let mut product: Vec<IBig> = vec![IBig::ZERO; 32];
        for (i, x) in a.iter().enumerate() {
            for (j, y) in b.iter().enumerate() {
                let term: IBig = IBig::from(x * y);
                if i + j < 32 {
                    product[i + j] += term;
                } else {
                    product[i + j - 32] -= term;
                }
            }
        }
        let expected: Vec<UBig> = u.iter().zip(&v).map(|(x, y)| reduce(&(x * y), &q)).collect();
        assert_eq!(encoder.retrieve_slots(&module, &product, 2).unwrap(), expected);
    }

    #[test]
    fn slots_at_the_sparsity_of_t_lie_at_roots_of_t() {
        // Magma's GBFV encoding of `t = X^32 - 2` uses sparsity `n_t = 16`: the roots of its
        // slots are roots of `t(X^(n/n_t))` modulo `p`.
        let t: GBFVPlaintextModulus = magma_modulus();
        let encoder: GBFVBatchEncoder = GBFVBatchEncoder::new(&t, 2, 9).unwrap();
        let p: &UBig = encoder.characteristic();
        let indices: Vec<usize> = slot_ntt_indices(9);
        for k in 0..512 / t.n() {
            let e: usize = 2 * bit_reverse(indices[k * t.n()], 9) + 1;
            let x: UBig = pow_mod(encoder.root_of_unity(), &UBig::from(e * 512 / t.n()), p);
            assert_eq!(x, UBig::from(2u8));
        }
    }

    #[test]
    fn encoding_rejects_invalid_shapes() {
        let encoder: GBFVBatchEncoder = GBFVBatchEncoder::new(&magma_modulus(), 2, 9).unwrap();
        let module: Module<BE> = module(&encoder);
        let one: Vec<IBig> = vec![IBig::ONE];
        assert!(encoder.embed_in_slots(&module, &one, 0).is_err());
        assert!(encoder.embed_in_slots(&module, &vec![IBig::ONE; 3], 1).is_err());
        assert!(encoder.embed_in_slots(&module, &vec![IBig::ONE; 512], 2).is_err());
        assert!(encoder.retrieve_slots(&module, &vec![IBig::ONE; 6], 1).is_err());
        assert!(encoder.retrieve_slots(&module, &vec![IBig::ONE; 8], 16).is_err());
        assert!(encoder.retrieve_slots(&module, &vec![IBig::ONE; 1024], 1).is_err());
    }
}
