use crate::kernels::znx::{znx_add_assign_portable, znx_copy_portable};

pub fn znx_mul_power_of_two_portable(mut k: i64, res: &mut [i64], a: &[i64]) {
    {
        assert_eq!(res.len(), a.len());
    }

    if k == 0 {
        znx_copy_portable(res, a);
        return;
    }

    if k > 0 {
        for (y, x) in res.iter_mut().zip(a.iter()) {
            *y = *x << k
        }
        return;
    }

    k = -k;

    for (y, x) in res.iter_mut().zip(a.iter()) {
        let sign_bit: i64 = (x >> 63) & 1;
        let bias: i64 = (1_i64 << (k - 1)) - sign_bit;
        *y = x.wrapping_add(bias) >> k;
    }
}

pub fn znx_mul_power_of_two_assign_portable(mut k: i64, res: &mut [i64]) {
    if k == 0 {
        return;
    }

    if k > 0 {
        for x in res.iter_mut() {
            *x <<= k
        }
        return;
    }

    k = -k;

    for x in res.iter_mut() {
        let sign_bit: i64 = (*x >> 63) & 1;
        let bias: i64 = (1_i64 << (k - 1)) - sign_bit;
        *x = x.wrapping_add(bias) >> k;
    }
}

pub fn znx_mul_add_power_of_two_portable(mut k: i64, res: &mut [i64], a: &[i64]) {
    {
        assert_eq!(res.len(), a.len());
    }

    if k == 0 {
        znx_add_assign_portable(res, a);
        return;
    }

    if k > 0 {
        for (y, x) in res.iter_mut().zip(a.iter()) {
            *y = y.wrapping_add(*x << k)
        }
        return;
    }

    k = -k;

    for (y, x) in res.iter_mut().zip(a.iter()) {
        let sign_bit: i64 = (x >> 63) & 1;
        let bias: i64 = (1_i64 << (k - 1)) - sign_bit;
        *y = y.wrapping_add(x.wrapping_add(bias) >> k);
    }
}
