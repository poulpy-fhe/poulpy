//! Direct negacyclic NTT over `Z/q` for a modulus of any size.
//!
//! No butterflies: the forward transform evaluates the input at each root
//! `psi^(2 bitrev(r) + 1)` by Horner's rule, and the inverse sums the definition of
//! the inverse directly. Both are quadratic in the degree.

use dashu_int::{UBig, fast_div::ConstDivisor, ops::BitTest};
use poulpy_hal::api::{ZqINTT, ZqN, ZqNTT, ZqNew};

use crate::family::bitrev;

/// Evaluation points of the negacyclic NTT of degree `n` modulo `q`, and their inverses.
pub struct ZqOracle {
    q: UBig,
    /// `psi^(2 bitrev(r) + 1)` for `r < n`.
    points: Vec<UBig>,
    /// `psi^(-(2 bitrev(r) + 1))` for `r < n`.
    inv_points: Vec<UBig>,
    /// `n^(-1) mod q`.
    inv_n: UBig,
}

impl ZqNew<UBig> for ZqOracle {
    fn new(n: usize, q: &UBig, psi: &UBig) -> Self {
        assert!(n.is_power_of_two(), "zq ntt: degree must be a positive power of two");
        assert!(*q > UBig::ONE && q.bit(0), "zq ntt: modulus must be odd and greater than one");
        let ring = ConstDivisor::new(q.clone());
        let psi = ring.reduce(psi.clone());
        assert!(
            psi.pow(&UBig::from(n)) == ring.reduce(q - UBig::ONE),
            "zq ntt: root must satisfy root^n = -1"
        );
        let bits: u32 = n.ilog2();
        let exponent = |r: usize| UBig::from(2 * bitrev(r, bits) + 1);
        // `psi^(-e) = psi^(2n - e)`, as `psi` has order `2n`.
        let points: Vec<UBig> = (0..n).map(|r| psi.pow(&exponent(r)).residue()).collect();
        let inv_points: Vec<UBig> = (0..n)
            .map(|r| psi.pow(&(UBig::from(2 * n) - exponent(r))).residue())
            .collect();
        let inv_n: UBig = ring
            .reduce(UBig::from(n))
            .inv()
            .expect("an odd modulus makes a power-of-two degree invertible")
            .residue();
        Self {
            q: q.clone(),
            points,
            inv_points,
            inv_n,
        }
    }
}

impl ZqN for ZqOracle {
    fn n(&self) -> usize {
        self.points.len()
    }
}

impl ZqNTT<UBig> for ZqOracle {
    fn zq_ntt(&self, data: &mut [UBig]) {
        assert!(data.len() == self.n(), "zq ntt: data length must equal the degree");
        let input: Vec<UBig> = data.to_vec();
        for (out, x) in data.iter_mut().zip(&self.points) {
            *out = input.iter().rev().fold(UBig::ZERO, |acc, c| (acc * x + c) % &self.q);
        }
    }
}

impl ZqINTT<UBig> for ZqOracle {
    fn zq_intt(&self, data: &mut [UBig]) {
        assert!(data.len() == self.n(), "zq ntt: data length must equal the degree");
        let input: Vec<UBig> = data.to_vec();
        // `powers[r] = inv_points[r]^t` for the coefficient `t` being computed.
        let mut powers: Vec<UBig> = vec![UBig::ONE; input.len()];
        for out in data.iter_mut() {
            let mut sum: UBig = UBig::ZERO;
            for ((value, power), y) in input.iter().zip(powers.iter_mut()).zip(&self.inv_points) {
                sum = (sum + value * &*power) % &self.q;
                *power = (&*power * y) % &self.q;
            }
            *out = (sum * &self.inv_n) % &self.q;
        }
    }
}
