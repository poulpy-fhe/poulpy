//! Rounding boundaries shared by the FFT64 conversion kernels.

use crate::reference::fft64::reim::{ReimArith, reim_to_znx_i64_ref};

pub fn test_reim_to_znx_rounding<BE: ReimArith>() {
    let mut values = vec![0.0, -0.0, f64::from_bits(1), -f64::from_bits(1), f64::MIN_POSITIVE];
    for exponent in -1022..=62 {
        let x = 2.0f64.powi(exponent);
        for value in [
            x.next_down(),
            x,
            x.next_up(),
            (x + 0.5).next_down(),
            x + 0.5,
            (x + 0.5).next_up(),
            x + 1.0,
        ] {
            values.extend([value, -value]);
        }
    }
    values.extend([
        (2.0f64.powi(63)).next_down(),
        -(2.0f64.powi(63)).next_down(),
        -2.0f64.powi(63),
    ]);
    for divisor in [1.0, 2.0, 32768.0] {
        for value in &values {
            // Nine copies exercise every SIMD lane plus a scalar tail.
            let input = vec![value * divisor; 9];
            let mut want = vec![0i64; input.len()];
            reim_to_znx_i64_ref(&mut want, divisor, &input);

            let mut have = vec![0i64; input.len()];
            BE::reim_to_znx(&mut have, divisor, &input);
            assert_eq!(have, want, "out-of-place: value={value}, divisor={divisor}");

            let mut inplace = input;
            BE::reim_to_znx_assign(&mut inplace, divisor);
            let have: Vec<i64> = inplace.iter().map(|x| x.to_bits() as i64).collect();
            assert_eq!(have, want, "in-place: value={value}, divisor={divisor}");
        }
    }
}

#[cfg(test)]
#[test]
fn reim_to_znx_rounding_boundaries() {
    test_reim_to_znx_rounding::<crate::FFT64Ref>();
}
