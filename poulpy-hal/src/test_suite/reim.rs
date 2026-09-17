//! Direct complex evaluation oracles for raw transform tables.

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
