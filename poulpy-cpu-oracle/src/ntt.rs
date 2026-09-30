//! Scalar negacyclic NTT over four 30-bit primes.
//!
//! Each coefficient is held as four residues. Slot `i` of a degree-`n`
//! transform holds the evaluation at `psi^e` with `psi` a primitive `2n`-th
//! root and `e = 2 bitrev(i) + 1`. Every butterfly reduces canonically, and the
//! inverse reconstructs centered integers by the CRT.

use poulpy_hal::layouts::{CrtWord, LaneElem, PrimeSet};

use crate::family::{Family, bitrev};

/// Four 30-bit primes `2^30 - c 2^19 + 1` with `2^19`-th roots of unity.
pub struct Primes30;

impl PrimeSet for Primes30 {
    type PrimeElem = u32;
    type Lanes<T: LaneElem> = [T; 4];
    const Q: [u32; 4] = [1_056_440_321, 1_053_818_881, 1_051_721_729, 1_049_100_289];
    const OMEGA: [u32; 4] = [195_937_198, 50_863_243, 633_648_745, 87_406_124];
    const LOG_Q: u64 = 30;
    const LOG_Q_PRODUCT: f64 = 119.886_155_257_481_1;
    const MAX_LOG_N: u32 = 18;
}

pub type Residues = CrtWord<Primes30, u64>;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Ntt4x30;

pub struct NttTable {
    n: usize,
    roots: [u64; 4],
    inv_roots: [u64; 4],
    inv_n: [u64; 4],
    crt: [i128; 4],
}

fn q(k: usize) -> u64 {
    u64::from(Primes30::Q[k])
}

fn pow_mod(mut a: u64, mut e: u64, q: u64) -> u64 {
    let mut r = 1;
    while e != 0 {
        if e & 1 != 0 {
            r = r * a % q;
        }
        a = a * a % q;
        e >>= 1;
    }
    r
}

fn inv_mod(a: u64, q: u64) -> u64 {
    pow_mod(a, q - 2, q)
}

/// Forward: twist by `psi^j`, then decimation in frequency.
fn ntt(table: &NttTable, data: &mut [u64]) {
    let n = table.n;
    for k in 0..4 {
        let (q, root) = (q(k), table.roots[k]);
        let mut twist = 1;
        for j in 0..n {
            data[4 * j + k] = data[4 * j + k] % q * twist % q;
            twist = twist * root % q;
        }
        let mut len = n;
        while len >= 2 {
            let step = pow_mod(root, (2 * n / len) as u64, q);
            for start in (0..n).step_by(len) {
                let mut w = 1;
                for j in 0..len / 2 {
                    let (i, h) = (4 * (start + j) + k, 4 * (start + j + len / 2) + k);
                    let (a, b) = (data[i], data[h]);
                    data[i] = (a + b) % q;
                    data[h] = (a + q - b) * w % q;
                    w = w * step % q;
                }
            }
            len /= 2;
        }
    }
}

/// Inverse: decimation in time, then untwist by `psi^-j / n`.
fn intt(table: &NttTable, data: &mut [u64]) {
    let n = table.n;
    for k in 0..4 {
        let (q, root) = (q(k), table.inv_roots[k]);
        let mut len = 2;
        while len <= n {
            let step = pow_mod(root, (2 * n / len) as u64, q);
            for start in (0..n).step_by(len) {
                let mut w = 1;
                for j in 0..len / 2 {
                    let (i, h) = (4 * (start + j) + k, 4 * (start + j + len / 2) + k);
                    let (a, b) = (data[i], data[h] * w % q);
                    data[i] = (a + b) % q;
                    data[h] = (a + q - b) % q;
                    w = w * step % q;
                }
            }
            len *= 2;
        }
        let mut twist = table.inv_n[k];
        for j in 0..n {
            data[4 * j + k] = data[4 * j + k] * twist % q;
            twist = twist * root % q;
        }
    }
}

/// `(Q / q_k)^-1 mod q_k`, `Q` the prime product.
fn crt_inverses() -> [i128; 4] {
    let total: u128 = Primes30::Q.iter().map(|&q| u128::from(q)).product();
    std::array::from_fn(|k| {
        i128::from(inv_mod(
            (total / u128::from(Primes30::Q[k]) % u128::from(Primes30::Q[k])) as u64,
            q(k),
        ))
    })
}

/// The centered integer in `(-Q/2, Q/2]` with residues `x`.
fn crt(inverses: &[i128; 4], x: &[u64; 4]) -> i128 {
    let q: [i128; 4] = std::array::from_fn(|k| i128::from(Primes30::Q[k]));
    let total: i128 = q.iter().product();
    let mut value: i128 = 0;
    for k in 0..4 {
        value += (x[k] as i128 % q[k]) * inverses[k] % q[k] * (total / q[k]);
    }
    value %= total;
    if value > total / 2 { value - total } else { value }
}

fn lanes(a: &[Residues]) -> &[u64] {
    bytemuck::cast_slice(a)
}

fn lanes_mut(a: &mut [Residues]) -> &mut [u64] {
    bytemuck::cast_slice_mut(a)
}

impl Family for Ntt4x30 {
    type Dft = Residues;
    type Big = i128;
    type Table = NttTable;

    fn table(n: usize) -> NttTable {
        assert!(
            n.is_power_of_two() && n <= 1 << Primes30::MAX_LOG_N,
            "unsupported NTT degree {n}"
        );
        let roots: [u64; 4] = std::array::from_fn(|k| {
            let root = pow_mod(
                u64::from(Primes30::OMEGA[k]),
                ((1usize << Primes30::MAX_LOG_N) / n) as u64,
                q(k),
            );
            assert_eq!(pow_mod(root, n as u64, q(k)), q(k) - 1, "not a primitive 2n-th root");
            root
        });
        NttTable {
            n,
            roots,
            inv_roots: std::array::from_fn(|k| inv_mod(roots[k], q(k))),
            inv_n: std::array::from_fn(|k| inv_mod(n as u64, q(k))),
            crt: crt_inverses(),
        }
    }

    fn forward(table: &NttTable, res: &mut [Residues], a: &[i64]) {
        for (r, &x) in res.iter_mut().zip(a) {
            r.0 = std::array::from_fn(|k| x.rem_euclid(q(k) as i64) as u64);
        }
        ntt(table, lanes_mut(res));
    }

    fn inverse(table: &NttTable, res: &mut [i128], a: &[Residues]) {
        let mut values = a.to_vec();
        intt(table, lanes_mut(&mut values));
        for (r, v) in res.iter_mut().zip(&values) {
            *r = crt(&table.crt, &v.0);
        }
    }

    fn dft_add(a: Residues, b: Residues) -> Residues {
        CrtWord(std::array::from_fn(|k| (a.0[k] % q(k) + b.0[k] % q(k)) % q(k)))
    }

    fn dft_neg(a: Residues) -> Residues {
        CrtWord(std::array::from_fn(|k| (q(k) - a.0[k] % q(k)) % q(k)))
    }

    fn mul_acc(res: &mut [Residues], a: &[Residues], b: &[Residues]) {
        let (a, b) = (lanes(a), lanes(b));
        for (i, r) in lanes_mut(res).iter_mut().enumerate() {
            let q = u128::from(q(i % 4));
            *r = ((u128::from(*r) + u128::from(a[i]) * u128::from(b[i])) % q) as u64;
        }
    }

    fn mul_assign(res: &mut [Residues], a: &[Residues]) {
        let a = lanes(a);
        for (i, r) in lanes_mut(res).iter_mut().enumerate() {
            *r = (u128::from(*r) * u128::from(a[i]) % u128::from(q(i % 4))) as u64;
        }
    }

    // sigma_p(a) at psi^e is a at psi^(pe).
    fn dft_automorphism(p: i64, res: &mut [Residues], a: &[Residues]) {
        let n = a.len() as i64;
        let bits = a.len().trailing_zeros();
        assert!(p & 1 == 1, "p must be odd, got {p}");
        for (i, r) in res.iter_mut().enumerate() {
            let e = 2 * bitrev(i, bits) as i64 + 1;
            let s = (p * e).rem_euclid(2 * n);
            *r = a[bitrev(((s - 1) / 2) as usize, bits)];
        }
    }

    // Slot i >= n holds the value at exponent 4n - e_i, the conjugate of a
    // stored slot.
    fn ci_expand(res: &mut [Residues], a: &[Residues]) {
        let (n, m) = (a.len(), 2 * a.len());
        let bits = m.trailing_zeros();
        res[..n].copy_from_slice(a);
        for (i, r) in res.iter_mut().enumerate().skip(n) {
            let e = 2 * bitrev(i, bits) + 1;
            *r = a[bitrev((2 * m - e - 1) / 2, bits)];
        }
    }

    // Products are lane-wise, so they apply to the stored slots as they are.
    fn ci_mul_acc(res: &mut [Residues], a: &[Residues], b: &[Residues]) {
        Self::mul_acc(res, a, b);
    }

    fn ci_mul_assign(res: &mut [Residues], a: &[Residues]) {
        Self::mul_assign(res, a);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn prime_set_is_consistent() {
        Primes30::validate();
        for k in 0..4 {
            let omega = u64::from(Primes30::OMEGA[k]);
            assert_eq!(pow_mod(omega, 1 << Primes30::MAX_LOG_N, q(k)), q(k) - 1);
        }
    }

    #[test]
    fn transform_matches_direct_evaluation() {
        for log_n in 0..=6u32 {
            let n = 1usize << log_n;
            let table = Ntt4x30::table(n);
            let input: Vec<u64> = (0..4 * n).map(|i| u64::MAX - i as u64 * 12345).collect();
            let mut actual = input.clone();
            ntt(&table, &mut actual);
            for k in 0..4 {
                for i in 0..n {
                    let point = pow_mod(table.roots[k], (2 * bitrev(i, log_n) + 1) as u64, q(k));
                    let mut expected = 0;
                    for j in (0..n).rev() {
                        expected = (expected * point + input[4 * j + k] % q(k)) % q(k);
                    }
                    assert_eq!(actual[4 * i + k], expected);
                }
            }
            intt(&table, &mut actual);
            for (i, (x, y)) in actual.iter().zip(&input).enumerate() {
                assert_eq!(*x, y % q(i % 4));
            }
        }
    }

    #[test]
    fn crt_reconstructs_signed_boundary_values() {
        let total: i128 = Primes30::Q.iter().map(|&q| i128::from(q)).product();
        let half = total / 2;
        for x in [-half, -half + 1, i64::MIN as i128, -1, 0, 1, i64::MAX as i128, half - 1, half] {
            let residues: [u64; 4] = std::array::from_fn(|k| x.rem_euclid(i128::from(Primes30::Q[k])) as u64);
            assert_eq!(crt(&crt_inverses(), &residues), x);
        }
    }
}
