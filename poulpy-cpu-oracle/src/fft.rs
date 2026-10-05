//! Scalar negacyclic FFT over `f64`.
//!
//! A degree-`n` polynomial is transformed as `m = n/2` complex values in reim
//! layout, all real parts then all imaginary parts. Slot `j` holds the
//! evaluation at `w^e` with `w = exp(i pi / n)` and `e = 4 bitrev(j) + 1`.

use crate::family::{DFTFamily, bitrev};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Fft64;

pub struct FftTable {
    m: usize,
    forward: Vec<f64>,
    inverse: Vec<f64>,
}

/// Twists `exp(+-i 2pi j / 4m)` and butterfly roots `exp(+-i 2pi j / m)`.
fn roots(m: usize, inverse: bool) -> Vec<f64> {
    let mut roots = vec![0.0; 4 * m];
    let turn = if inverse {
        -std::f64::consts::TAU
    } else {
        std::f64::consts::TAU
    };
    for j in 0..m {
        let angle = turn * j as f64 / m as f64;
        let twist = angle / 4.0;
        roots[j] = twist.cos();
        roots[m + j] = twist.sin();
        roots[2 * m + j] = angle.cos();
        roots[3 * m + j] = angle.sin();
    }
    roots
}

/// Forward: twist, then decimation in frequency. Inverse: decimation in time,
/// then untwist, unscaled.
fn transform(m: usize, roots: &[f64], data: &mut [f64], inverse: bool) {
    assert_eq!(data.len(), 2 * m);
    let (re, im) = data.split_at_mut(m);
    let twist = |re: &mut [f64], im: &mut [f64]| {
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

impl DFTFamily for Fft64 {
    type Dft = f64;
    type Big = i64;
    type Table = FftTable;

    fn table(n: usize) -> FftTable {
        let m = n / 2;
        FftTable {
            m,
            forward: roots(m, false),
            inverse: roots(m, true),
        }
    }

    fn forward(table: &FftTable, res: &mut [f64], a: &[i64]) {
        for (r, &x) in res.iter_mut().zip(a) {
            *r = x as f64;
        }
        transform(table.m, &table.forward, res, false);
    }

    fn inverse(table: &FftTable, res: &mut [i64], a: &[f64]) {
        let mut values = a.to_vec();
        transform(table.m, &table.inverse, &mut values, true);
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
            transform(m, &roots(m, false), &mut actual, false);
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
            transform(m, &roots(m, true), &mut actual, true);
            for i in 0..2 * m {
                assert!((actual[i] / m as f64 - input[i]).abs() < 1e-10);
            }
        }
    }
}
