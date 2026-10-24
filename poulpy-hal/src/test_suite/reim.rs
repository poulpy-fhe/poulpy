//! Direct complex evaluation oracles for raw transform tables.

use rand_distr::num_traits::{Float, FromPrimitive};

use crate::api::NegacyclicFFT;

/// Checks forward root order, the separate real/imaginary halves, and the
/// inverse's unnormalized scale against direct complex sums.
pub fn test_negacyclic_fft<T: NegacyclicFFT<f64>>(table: &T) {
    let m = table.m();
    assert!(m.is_power_of_two());
    let input: Vec<f64> = (0..2 * m).map(|i| ((i * 17 + 3) % 29) as f64 / 16.0 - 0.75).collect();
    let tolerance = 1e-10 * m as f64;
    for inverse in [false, true] {
        let mut have = input.clone();
        if inverse {
            table.ifft(&mut have);
        } else {
            table.fft(&mut have);
        }
        for out in 0..m {
            let mut want_re = 0.0;
            let mut want_im = 0.0;
            for src in 0..m {
                let root_index = if inverse { src } else { out };
                let reversed = if m == 1 {
                    0
                } else {
                    root_index.reverse_bits() >> (usize::BITS - m.ilog2())
                };
                let power = if inverse { -(out as f64) } else { src as f64 };
                let angle = std::f64::consts::TAU * (reversed as f64 + 0.25) * power / m as f64;
                let (sin, cos) = angle.sin_cos();
                want_re += input[src] * cos - input[m + src] * sin;
                want_im += input[src] * sin + input[m + src] * cos;
            }
            assert!(
                (have[out] - want_re).abs() <= tolerance,
                "m={m}, inverse={inverse}, real[{out}]"
            );
            assert!(
                (have[m + out] - want_im).abs() <= tolerance,
                "m={m}, inverse={inverse}, imaginary[{out}]"
            );
        }
    }
    let mut roundtrip = input.clone();
    table.fft(&mut roundtrip);
    table.ifft(&mut roundtrip);
    for (actual, original) in roundtrip.iter().zip(input) {
        assert!((actual - original * m as f64).abs() <= tolerance);
    }
}

/// Checks that two transforms of the same size return the same bits on fixed
/// inputs, forward and inverse: zeros, an impulse, alternating signs, tiny and
/// huge magnitudes and dense dyadic values.
pub fn test_negacyclic_fft_bit_exact<F, A, B>(a: &A, b: &B)
where
    F: Float + FromPrimitive + bytemuck::Pod + std::fmt::Debug,
    A: NegacyclicFFT<F>,
    B: NegacyclicFFT<F>,
{
    let m = a.m();
    assert_eq!(m, b.m(), "transform sizes differ");
    let scalar = |x: f64| F::from_f64(x).expect("representable test value");
    for sample in 0..10u32 {
        let input: Vec<F> = (0..2 * m)
            .map(|i| match sample {
                0 => F::zero(),
                1 => {
                    if i == m / 2 {
                        F::one()
                    } else {
                        F::zero()
                    }
                }
                2 => scalar(if i & 1 == 0 { 1.0 } else { -1.0 }),
                3 => {
                    if i & 1 == 0 {
                        F::zero()
                    } else {
                        -F::zero()
                    }
                }
                4 => F::min_positive_value() * F::epsilon() * scalar((i % 7) as f64),
                5 => F::max_value() / scalar((8 * m) as f64),
                _ => {
                    let bits = (i as u64).wrapping_mul(0x9e37_79b9_7f4a_7c15).rotate_left(17 + sample);
                    scalar((bits as i64 >> 12) as f64) * scalar(1.0 / (1u64 << 51) as f64)
                        + scalar((i as f64 + 1.0) / (1u128 << (60 + sample)) as f64)
                }
            })
            .collect();
        for inverse in [false, true] {
            let (mut have, mut want) = (input.clone(), input.clone());
            if inverse {
                a.ifft(&mut have);
                b.ifft(&mut want);
            } else {
                a.fft(&mut have);
                b.fft(&mut want);
            }
            for (i, (x, y)) in have.iter().zip(&want).enumerate() {
                assert!(
                    bytemuck::bytes_of(x) == bytemuck::bytes_of(y),
                    "m={m} sample={sample} inverse={inverse} scalar {i}: {x:?} != {y:?}"
                );
            }
        }
    }
}
