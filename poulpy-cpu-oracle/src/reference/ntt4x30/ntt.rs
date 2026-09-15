//! Scalar negacyclic NTT with canonical reduction after every butterfly.

use std::marker::PhantomData;

use super::primes::PrimeSetCrt4;

pub struct NttTable<P: PrimeSetCrt4> {
    pub n: usize,
    roots: [u64; 4],
    _prime: PhantomData<P>,
}

pub struct NttTableInv<P: PrimeSetCrt4> {
    pub n: usize,
    roots: [u64; 4],
    scales: [u64; 4],
    _prime: PhantomData<P>,
}

fn pow_mod(mut a: u64, mut exponent: u64, q: u64) -> u64 {
    let mut result = 1;
    while exponent != 0 {
        if exponent & 1 != 0 {
            result = result * a % q;
        }
        a = a * a % q;
        exponent >>= 1;
    }
    result
}

fn roots<P: PrimeSetCrt4>(n: usize) -> [u64; 4] {
    assert!(n.is_power_of_two() && n <= 65536);
    std::array::from_fn(|k| {
        let q = P::Q[k] as u64;
        let root = pow_mod(P::OMEGA[k] as u64, (65536 / n) as u64, q);
        assert_eq!(pow_mod(root, n as u64, q), q - 1);
        root
    })
}

impl<P: PrimeSetCrt4> NttTable<P> {
    pub fn new(n: usize) -> Self {
        Self {
            n,
            roots: roots::<P>(n),
            _prime: PhantomData,
        }
    }
}

impl<P: PrimeSetCrt4> NttTableInv<P> {
    pub fn new(n: usize) -> Self {
        let roots = roots::<P>(n);
        Self {
            n,
            roots: std::array::from_fn(|k| pow_mod(roots[k], P::Q[k] as u64 - 2, P::Q[k] as u64)),
            scales: std::array::from_fn(|k| pow_mod(n as u64, P::Q[k] as u64 - 2, P::Q[k] as u64)),
            _prime: PhantomData,
        }
    }
}

pub fn ntt_ref<P: PrimeSetCrt4>(table: &NttTable<P>, data: &mut [u64]) {
    let n = table.n;
    assert_eq!(data.len(), 4 * n);
    for k in 0..4 {
        let q = P::Q[k] as u64;
        let root = table.roots[k];
        let mut twist = 1;
        for j in 0..n {
            data[4 * j + k] = (data[4 * j + k] % q) * twist % q;
            twist = twist * root % q;
        }
        let mut len = n;
        while len >= 2 {
            let step = pow_mod(root, (2 * n / len) as u64, q);
            for start in (0..n).step_by(len) {
                let mut w = 1;
                for j in 0..len / 2 {
                    let i = 4 * (start + j) + k;
                    let h = 4 * (start + j + len / 2) + k;
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

pub fn intt_ref<P: PrimeSetCrt4>(table: &NttTableInv<P>, data: &mut [u64]) {
    let n = table.n;
    assert_eq!(data.len(), 4 * n);
    for k in 0..4 {
        let q = P::Q[k] as u64;
        let root = table.roots[k];
        for j in 0..n {
            data[4 * j + k] %= q;
        }
        let mut len = 2;
        while len <= n {
            let step = pow_mod(root, (2 * n / len) as u64, q);
            for start in (0..n).step_by(len) {
                let mut w = 1;
                for j in 0..len / 2 {
                    let i = 4 * (start + j) + k;
                    let h = 4 * (start + j + len / 2) + k;
                    let a = data[i];
                    let b = data[h] * w % q;
                    data[i] = (a + b) % q;
                    data[h] = (a + q - b) % q;
                    w = w * step % q;
                }
            }
            len *= 2;
        }
        let mut twist = table.scales[k];
        for j in 0..n {
            data[4 * j + k] = data[4 * j + k] * twist % q;
            twist = twist * root % q;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::super::primes::{PrimeSet, Primes30};
    use super::*;

    #[test]
    fn transform_matches_polynomial_evaluation() {
        for log_n in 0..=6 {
            let n = 1usize << log_n;
            let table = NttTable::<Primes30>::new(n);
            let input: Vec<u64> = (0..4 * n).map(|i| u64::MAX - i as u64 * 12345).collect();
            let mut actual = input.clone();
            ntt_ref(&table, &mut actual);
            for k in 0..4 {
                let q = Primes30::Q[k] as u64;
                for j in 0..n {
                    let frequency = if log_n == 0 {
                        0
                    } else {
                        j.reverse_bits() >> (usize::BITS - log_n)
                    };
                    let x = pow_mod(table.roots[k], (2 * frequency + 1) as u64, q);
                    let mut expected = 0;
                    for i in (0..n).rev() {
                        expected = (expected * x + input[4 * i + k] % q) % q;
                    }
                    assert_eq!(actual[4 * j + k], expected);
                }
            }
            intt_ref(&NttTableInv::<Primes30>::new(n), &mut actual);
            for i in 0..4 * n {
                assert_eq!(actual[i], input[i] % Primes30::Q[i % 4] as u64);
            }
        }
    }

    #[test]
    fn convolution_matches_schoolbook_negacyclic_product() {
        for log_n in 1..=5 {
            let n = 1usize << log_n;
            let fwd = NttTable::<Primes30>::new(n);
            let inv = NttTableInv::<Primes30>::new(n);
            let input: Vec<u64> = (0..4 * n).map(|i| (i * 31337 + 7) as u64).collect();
            let mut actual = input.clone();
            ntt_ref(&fwd, &mut actual);
            for (i, value) in actual.iter_mut().enumerate() {
                *value = *value * *value % Primes30::Q[i % 4] as u64;
            }
            intt_ref(&inv, &mut actual);
            for k in 0..4 {
                let q = Primes30::Q[k] as i128;
                let mut expected = vec![0i128; n];
                for i in 0..n {
                    for j in 0..n {
                        let product = input[4 * i + k] as i128 * input[4 * j + k] as i128;
                        expected[(i + j) % n] += if i + j < n { product } else { -product };
                    }
                }
                for i in 0..n {
                    assert_eq!(actual[4 * i + k], expected[i].rem_euclid(q) as u64);
                }
            }
        }
    }
}
