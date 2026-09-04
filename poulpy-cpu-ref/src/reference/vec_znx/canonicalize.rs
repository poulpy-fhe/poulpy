use poulpy_hal::layouts::{Backend, HostDataMut, VecZnxBackendMut, ZnxView, ZnxViewMut};

#[inline]
fn split_digit(base2k: usize, value: i128) -> (i64, i128) {
    let modulus = 1i128 << base2k;
    let half = modulus >> 1;
    let unsigned = value.rem_euclid(modulus);
    let digit = if unsigned >= half { unsigned - modulus } else { unsigned };
    (digit as i64, (value - digit) >> base2k)
}

pub fn vec_znx_canonicalize<BE>(base2k: usize, k: usize, a: &mut VecZnxBackendMut<'_, BE>)
where
    BE: Backend<ZnxWord = i64>,
    for<'a> BE::BufMut<'a>: HostDataMut,
{
    assert!((1..=i64::BITS as usize).contains(&base2k));
    assert!(
        k <= a.size() * base2k,
        "k ({k}) exceeds VecZnx capacity ({})",
        a.size() * base2k
    );

    let active_size = k.div_ceil(base2k);
    for col in 0..a.cols() {
        for limb in active_size..a.size() {
            a.at_mut(col, limb).fill(0);
        }
    }

    let padding = (base2k - k % base2k) % base2k;
    if active_size == 0 || padding == 0 {
        return;
    }

    for col in 0..a.cols() {
        for coeff in 0..a.n() {
            let value = a.at(col, active_size - 1)[coeff];
            let low_digit = split_digit(padding, value as i128).0;
            let (digit, mut carry) = split_digit(base2k, value as i128 - low_digit as i128);
            a.at_mut(col, active_size - 1)[coeff] = digit;

            for limb in (0..active_size - 1).rev() {
                let (digit, next_carry) = split_digit(base2k, a.at(col, limb)[coeff] as i128 + carry);
                a.at_mut(col, limb)[coeff] = digit;
                carry = next_carry;
            }
        }
    }
}
