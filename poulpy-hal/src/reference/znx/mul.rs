use crate::reference::znx::{znx_add_assign_ref, znx_copy_ref};

pub fn znx_mul_power_of_two_ref(mut k: i64, res: &mut [i64], a: &[i64]) {
    #[cfg(debug_assertions)]
    {
        assert_eq!(res.len(), a.len());
    }

    if k == 0 {
        znx_copy_ref(res, a);
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
        *y = (x + bias) >> k;
    }
}

pub fn znx_mul_power_of_two_assign_ref(mut k: i64, res: &mut [i64]) {
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
        *x = (*x + bias) >> k;
    }
}

pub fn znx_mul_add_power_of_two_ref(mut k: i64, res: &mut [i64], a: &[i64]) {
    #[cfg(debug_assertions)]
    {
        assert_eq!(res.len(), a.len());
    }

    if k == 0 {
        znx_add_assign_ref(res, a);
        return;
    }

    if k > 0 {
        for (y, x) in res.iter_mut().zip(a.iter()) {
            *y += *x << k
        }
        return;
    }

    k = -k;

    for (y, x) in res.iter_mut().zip(a.iter()) {
        let sign_bit: i64 = (x >> 63) & 1;
        let bias: i64 = (1_i64 << (k - 1)) - sign_bit;
        *y += (x + bias) >> k;
    }
}
