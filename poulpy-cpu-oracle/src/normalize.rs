use dashu_int::IBig;
use poulpy_hal::layouts::{ZnxView, ZnxViewMut};

fn integer(size: usize, base2k: usize, mut source: impl FnMut(usize) -> i128) -> IBig {
    assert!((1..=127).contains(&base2k));
    let mut value = IBig::ZERO;
    for j in 0..size {
        value = (value << base2k) + IBig::from(source(j));
    }
    value
}

fn digits(
    mut value: IBig,
    source_bits: usize,
    size: usize,
    base2k: usize,
    k: usize,
    offset: i64,
    mut output: impl FnMut(usize, i64),
) {
    assert!((1..=64).contains(&base2k));
    assert!(k <= size * base2k);
    let shift = k as i128 + offset as i128 - source_bits as i128;
    // Bound extreme offsets before constructing an arbitrarily large integer.
    if shift >= k as i128 || -shift > source_bits as i128 + 128 {
        value = IBig::ZERO;
    } else if shift >= 0 {
        value <<= shift as usize;
    } else {
        let drop = (-shift) as usize;
        value = (value + (IBig::ONE << (drop - 1))) >> drop;
    }
    let active = k.div_ceil(base2k);
    let padding = active * base2k - k;
    value <<= padding;
    let radix = IBig::ONE << base2k;
    let half = IBig::ONE << (base2k - 1);
    for j in (0..active).rev() {
        let carry = (&value + &half) >> base2k;
        output(j, i64::try_from(&value - &carry * &radix).unwrap());
        value = carry;
    }
    for j in active..size {
        output(j, 0);
    }
}

pub(crate) fn normalize<R, A>(res: &mut R, base2k: usize, k: usize, offset: i64, col: usize, a: &A, a_base2k: usize, a_col: usize)
where
    R: ZnxViewMut<Scalar = i64>,
    A: ZnxView,
    A::Scalar: Copy + Into<i128>,
{
    assert_eq!(res.n(), a.n());
    let size = res.size();
    for coeff in 0..res.n() {
        let value = integer(a.size(), a_base2k, |j| a.at(a_col, j)[coeff].into());
        digits(value, a.size() * a_base2k, size, base2k, k, offset, |j, x| {
            res.at_mut(col, j)[coeff] = x
        });
    }
}

pub(crate) fn normalize_assign<R: ZnxViewMut<Scalar = i64>>(res: &mut R, base2k: usize, k: usize, offset: i64, col: usize) {
    let size = res.size();
    for coeff in 0..res.n() {
        let value = integer(size, base2k, |j| res.at(col, j)[coeff] as i128);
        digits(value, size * base2k, size, base2k, k, offset, |j, x| {
            res.at_mut(col, j)[coeff] = x
        });
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn normalize_digits(input: &[i128], input_base: usize, base: usize, k: usize, offset: i64) -> Vec<i64> {
        let mut output = vec![99; 3];
        digits(
            integer(input.len(), input_base, |j| input[j]),
            input.len() * input_base,
            output.len(),
            base,
            k,
            offset,
            |j, x| output[j] = x,
        );
        output
    }

    #[test]
    fn rounding_ties_go_toward_positive_infinity() {
        for (input, expected) in [(-3, -1), (-1, 0), (1, 1), (3, 2)] {
            assert_eq!(normalize_digits(&[input], 5, 4, 4, 0), [expected, 0, 0]);
        }
    }

    #[test]
    fn partial_width_and_centered_carry() {
        assert_eq!(normalize_digits(&[3], 4, 4, 3, 0), [4, 0, 0]);
        assert_eq!(normalize_digits(&[-3], 4, 4, 3, 0), [-2, 0, 0]);
        assert_eq!(normalize_digits(&[8, 8], 4, 4, 8, 0), [-7, -8, 0]);
        assert_eq!(normalize_digits(&[15, 15], 4, 4, 8, 0), [0, -1, 0]);
        assert_eq!(normalize_digits(&[15, 15], 4, 4, 0, 0), [0, 0, 0]);
    }

    #[test]
    fn wide_inputs_and_extreme_offsets() {
        assert_eq!(normalize_digits(&[i128::MAX], 64, 64, 64, 0), [-1, 0, 0]);
        assert_eq!(normalize_digits(&[i128::MIN], 64, 64, 64, 0), [0, 0, 0]);
        for offset in [i64::MIN, i64::MAX] {
            for input in [i128::MIN, -1, 1, i128::MAX] {
                assert_eq!(normalize_digits(&[input], 64, 4, 8, offset), [0, 0, 0]);
            }
        }
    }
}
