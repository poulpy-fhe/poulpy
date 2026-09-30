use std::fmt::Debug;

use rand_distr::num_traits::{Float, FloatConst};

pub(super) fn roots<R: Float + FloatConst>(m: usize, inverse: bool) -> Vec<R> {
    assert!(m.is_power_of_two());
    let mut roots = vec![R::zero(); 4 * m];
    let sign = if inverse { -R::one() } else { R::one() };
    let turn = sign * R::from(2).unwrap() * R::PI();
    for j in 0..m {
        let angle = turn * R::from(j).unwrap() / R::from(m).unwrap();
        let twist = angle / R::from(4).unwrap();
        roots[j] = twist.cos();
        roots[m + j] = twist.sin();
        roots[2 * m + j] = angle.cos();
        roots[3 * m + j] = angle.sin();
    }
    roots
}

pub(super) fn transform<R: Float + FloatConst>(m: usize, roots: &[R], data: &mut [R], inverse: bool) {
    assert!(m.is_power_of_two());
    assert_eq!(data.len(), 2 * m);
    assert_eq!(roots.len(), 4 * m);
    let (re, im) = data.split_at_mut(m);
    if !inverse {
        for j in 0..m {
            let (a, b) = (re[j], im[j]);
            re[j] = a * roots[j] - b * roots[m + j];
            im[j] = a * roots[m + j] + b * roots[j];
        }
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
                    re[i] = ar + tr;
                    im[i] = ai + ti;
                    re[h] = ar - tr;
                    im[h] = ai - ti;
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
        for j in 0..m {
            let (a, b) = (re[j], im[j]);
            re[j] = a * roots[j] - b * roots[m + j];
            im[j] = a * roots[m + j] + b * roots[j];
        }
    }
}

pub fn fft_ref<R: Float + FloatConst + Debug>(m: usize, roots: &[R], data: &mut [R]) {
    transform(m, roots, data, false);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fft_matches_direct_polynomial_evaluation() {
        for log_m in 0..=6 {
            let m = 1usize << log_m;
            let input: Vec<f64> = (0..2 * m).map(|i| (i % 17) as f64 - 8.0).collect();
            let mut actual = input.clone();
            fft_ref(m, &roots(m, false), &mut actual);
            for j in 0..m {
                let index = if log_m == 0 {
                    0
                } else {
                    j.reverse_bits() >> (usize::BITS - log_m)
                };
                let angle = std::f64::consts::TAU * (index as f64 + 0.25) / m as f64;
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
