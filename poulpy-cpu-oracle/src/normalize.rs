use dashu_int::{IBig, ops::BitTest};
use poulpy_hal::layouts::{ZnxView, ZnxViewMut};

fn integer(size: usize, base2k: usize, mut source: impl FnMut(usize) -> i128) -> IBig {
    assert!((1..=127).contains(&base2k));
    let mut value = IBig::ZERO;
    for j in 0..size {
        value = (value << base2k) + IBig::from(source(j));
    }
    value
}

/// `value / 2^drop` rounded to the nearest integer, ties to the even one.
fn round_ties_even(value: IBig, drop: usize) -> IBig {
    let floor = &value >> drop;
    let rem = value - (&floor << drop);
    let half = IBig::ONE << (drop - 1);
    if rem > half || (rem == half && floor.bit(0)) {
        floor + IBig::ONE
    } else {
        floor
    }
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
        value = round_ties_even(value, drop);
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

/// Gadget digit layout, see `poulpy_hal::api::VecZnxGadgetDecompose`: the
/// input rounded half up at `dnum * k_digit`, each digit the centered residue
/// modulo `2^k_digit` of what is left, then split into centered limbs from its
/// bottom up with the top limb taking the rest, on exact integers.
pub(crate) fn gadget_decompose<R, A>(res: &mut R, col: usize, a: &A, a_col: usize, base2k: usize, k_digit: usize)
where
    R: ZnxViewMut<Scalar = i64>,
    A: ZnxView,
    A::Scalar: Copy + Into<i128>,
{
    assert_eq!(res.n(), a.n());
    let r = k_digit.div_ceil(base2k);
    assert!(res.size() > 0 && res.size().is_multiple_of(r));
    let dnum = res.size() / r;
    let source_bits = (a.size() * base2k) as i128;
    let drop = source_bits - (dnum * k_digit) as i128;
    let centered = |value: &IBig, width: usize| -> (IBig, IBig) {
        let carry = (value + (IBig::ONE << (width - 1))) >> width;
        let residue = value - (&carry << width);
        (residue, carry)
    };
    for coeff in 0..res.n() {
        let mut value = integer(a.size(), base2k, |j| a.at(a_col, j)[coeff].into());
        if drop > 0 {
            let drop = drop as usize;
            value = (value + (IBig::ONE << (drop - 1))) >> drop;
        } else {
            value <<= (-drop) as usize;
        }
        for q in (0..dnum).rev() {
            let (mut digit, carry) = centered(&value, k_digit);
            value = carry;
            for t in 0..r - 1 {
                let (limb, rest) = centered(&digit, base2k);
                res.at_mut(col, q * r + r - 1 - t)[coeff] = i64::try_from(&limb).unwrap();
                digit = rest;
            }
            res.at_mut(col, q * r)[coeff] = i64::try_from(&digit).unwrap();
        }
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
    fn rounding_ties_go_to_even() {
        // One bit dropped: round_even(x / 2).
        for (input, expected) in [(-3, -2), (-1, 0), (1, 0), (3, 2), (5, 2), (7, 4)] {
            assert_eq!(normalize_digits(&[input], 5, 4, 4, 0)[0], expected, "input {input}");
        }
        // Two bits dropped: round_even(x / 4); only remainder 2 is a tie.
        for (input, expected) in [(2, 0), (6, 2), (10, 2), (14, 4), (1, 0), (3, 1), (5, 1), (7, 2)] {
            assert_eq!(normalize_digits(&[input], 6, 4, 4, 0)[0], expected, "input {input}");
        }
    }

    #[test]
    fn partial_width_and_centered_carry() {
        assert_eq!(normalize_digits(&[3], 4, 4, 3, 0), [4, 0, 0]);
        assert_eq!(normalize_digits(&[-3], 4, 4, 3, 0), [-4, 0, 0]);
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
