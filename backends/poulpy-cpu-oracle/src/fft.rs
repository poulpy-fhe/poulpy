//! Scalar negacyclic FFT, generic over the float precision.
//!
//! A degree-`n` polynomial is transformed as `m = n/2` complex values in reim
//! layout, all real parts then all imaginary parts. Slot `j` holds the
//! evaluation at `w^e` with `w = exp(i pi / n)` and `e = 4 bitrev(j) + 1`,
//! which is the HAL's `omega(m, j)`.

use poulpy_hal::api::{NegacyclicFFT, NegacyclicFFTNew};
use rand_distr::num_traits::{Float, FloatConst};

use crate::family::{DFTFamily, bitrev};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Fft64;

/// Tables of the transform of `m` complex values at precision `F`.
pub struct ComplexFft<F> {
    m: usize,
    forward: Vec<F>,
    inverse: Vec<F>,
}

/// Twists `exp(+-i 2pi j / 4m)` and butterfly roots `exp(+-i 2pi j / m)`.
fn roots<F: Float + FloatConst>(m: usize, inverse: bool) -> Vec<F> {
    let mut roots = vec![F::zero(); 4 * m];
    let turn = if inverse { -F::TAU() } else { F::TAU() };
    for j in 0..m {
        let angle = turn * F::from(j).unwrap() / F::from(m).unwrap();
        let twist = angle / F::from(4).unwrap();
        roots[j] = twist.cos();
        roots[m + j] = twist.sin();
        roots[2 * m + j] = angle.cos();
        roots[3 * m + j] = angle.sin();
    }
    roots
}

/// Forward: twist, then decimation in frequency. Inverse: decimation in time,
/// then untwist, unscaled.
fn transform<F: Float>(m: usize, roots: &[F], data: &mut [F], inverse: bool) {
    assert_eq!(data.len(), 2 * m);
    let (re, im) = data.split_at_mut(m);
    let twist = |re: &mut [F], im: &mut [F]| {
        for j in 0..m {
            let (a, b) = (re[j], im[j]);
            re[j] = a * roots[j] - b * roots[m + j];
            im[j] = a * roots[m + j] + b * roots[j];
        }
    };
    if !inverse {
        twist(re, im);
    }
    let mut len = if inverse { 2 } else { m };
    while len >= 2 && len <= m {
        for start in (0..m).step_by(len) {
            for j in 0..len / 2 {
                let (i, h) = (start + j, start + j + len / 2);
                let (wr, wi) = (roots[2 * m + j * (m / len)], roots[3 * m + j * (m / len)]);
                let (ar, ai, br, bi) = (re[i], im[i], re[h], im[h]);
                if inverse {
                    let (tr, ti) = (br * wr - bi * wi, br * wi + bi * wr);
                    (re[i], im[i], re[h], im[h]) = (ar + tr, ai + ti, ar - tr, ai - ti);
                } else {
                    re[i] = ar + br;
                    im[i] = ai + bi;
                    re[h] = (ar - br) * wr - (ai - bi) * wi;
                    im[h] = (ar - br) * wi + (ai - bi) * wr;
                }
            }
        }
        len = if inverse { len * 2 } else { len / 2 };
    }
    if inverse {
        twist(re, im);
    }
}

impl<F: Float + FloatConst> NegacyclicFFTNew<F> for ComplexFft<F> {
    fn new(m: usize) -> Self {
        Self {
            m,
            forward: roots(m, false),
            inverse: roots(m, true),
        }
    }
}

impl<F: Float> NegacyclicFFT<F> for ComplexFft<F> {
    fn m(&self) -> usize {
        self.m
    }

    fn fft(&self, data: &mut [F]) {
        transform(self.m, &self.forward, data, false);
    }

    fn ifft(&self, data: &mut [F]) {
        transform(self.m, &self.inverse, data, true);
    }
}

impl DFTFamily for Fft64 {
    type Dft = f64;
    type Big = i64;
    type Table = ComplexFft<f64>;

    fn table(n: usize) -> ComplexFft<f64> {
        ComplexFft::new(n / 2)
    }

    fn forward(table: &ComplexFft<f64>, res: &mut [f64], a: &[i64]) {
        for (r, &x) in res.iter_mut().zip(a) {
            *r = x as f64;
        }
        table.fft(res);
    }

    fn inverse(table: &ComplexFft<f64>, res: &mut [i64], a: &[f64]) {
        let mut values = a.to_vec();
        table.ifft(&mut values);
        for (r, v) in res.iter_mut().zip(&values) {
            *r = (v / table.m as f64).round() as i64;
        }
    }

    fn dft_embed(res: &mut [f64], a: &[f64]) {
        assert!(a.len() >= 2 && a.len().is_power_of_two());
        assert!(res.len().is_power_of_two() && res.len() >= a.len());
        let gap = res.len() / a.len();
        // Bit-reversed evaluations repeat consecutively within each reim half.
        for (dst, &value) in res.chunks_exact_mut(gap).zip(a) {
            dst.fill(value);
        }
    }

    fn dft_add(a: f64, b: f64) -> f64 {
        a + b
    }

    fn dft_neg(a: f64) -> f64 {
        -a
    }

    fn mul_acc(res: &mut [f64], a: &[f64], b: &[f64]) {
        let m = res.len() / 2;
        for i in 0..m {
            res[i] += a[i] * b[i] - a[i + m] * b[i + m];
            res[i + m] += a[i] * b[i + m] + a[i + m] * b[i];
        }
    }

    fn mul_assign(res: &mut [f64], a: &[f64]) {
        let m = res.len() / 2;
        for i in 0..m {
            let (re, im) = (res[i], res[i + m]);
            res[i] = re * a[i] - im * a[i + m];
            res[i + m] = re * a[i + m] + im * a[i];
        }
    }

    // sigma_p(a) at w^e is a at w^(pe). For a real polynomial the value at
    // w^-t is the conjugate of the value at w^t, and one of +-pe is 1 mod 4.
    fn dft_automorphism(p: i64, res: &mut [f64], a: &[f64]) {
        let (n, m) = (a.len() as i64, a.len() / 2);
        let bits = m.trailing_zeros();
        assert!(p & 1 == 1, "p must be odd, got {p}");
        for j in 0..m {
            let e = 4 * bitrev(j, bits) as i64 + 1;
            let s = (p * e).rem_euclid(2 * n);
            let (t, sign) = if s % 4 == 1 { (s, 1.0) } else { (2 * n - s, -1.0) };
            let src = bitrev(((t - 1) / 4) as usize, bits);
            res[j] = a[src];
            res[m + j] = sign * a[m + src];
        }
    }

    // The spectrum of a subring element is real: the stored slots are the
    // real parts and the imaginary parts are zero.
    fn ci_expand(res: &mut [f64], a: &[f64]) {
        let n = a.len();
        res[..n].copy_from_slice(a);
        res[n..].fill(0.0);
    }

    // The stored conjugate-invariant slots are real evaluations.
    fn ci_mul_acc(res: &mut [f64], a: &[f64], b: &[f64]) {
        (0..res.len()).for_each(|i| res[i] += a[i] * b[i]);
    }

    fn ci_mul_assign(res: &mut [f64], a: &[f64]) {
        (0..res.len()).for_each(|i| res[i] *= a[i]);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn transform_matches_direct_evaluation() {
        for log_m in 0..=6 {
            let m = 1usize << log_m;
            let input: Vec<f64> = (0..2 * m).map(|i| (i % 17) as f64 - 8.0).collect();
            let mut actual = input.clone();
            transform(m, &roots::<f64>(m, false), &mut actual, false);
            for j in 0..m {
                let angle = std::f64::consts::TAU * (bitrev(j, log_m) as f64 + 0.25) / m as f64;
                let (mut re, mut im) = (0.0, 0.0);
                for i in 0..m {
                    let (s, c) = (angle * i as f64).sin_cos();
                    re += input[i] * c - input[m + i] * s;
                    im += input[i] * s + input[m + i] * c;
                }
                assert!((actual[j] - re).abs() < 1e-9);
                assert!((actual[m + j] - im).abs() < 1e-9);
            }
            transform(m, &roots::<f64>(m, true), &mut actual, true);
            for i in 0..2 * m {
                assert!((actual[i] / m as f64 - input[i]).abs() < 1e-10);
            }
        }
    }

    #[test]
    fn transform_meets_the_hal_contract() {
        for log_m in 0..=8 {
            poulpy_hal::test_suite::reim::test_negacyclic_fft(&ComplexFft::<f64>::new(1 << log_m));
        }
    }
}
