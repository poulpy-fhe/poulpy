#[inline(always)]
pub fn znx_negate_portable(res: &mut [i64], src: &[i64]) {
    {
        assert_eq!(res.len(), src.len())
    }

    for i in 0..res.len() {
        res[i] = src[i].wrapping_neg()
    }
}

#[inline(always)]
pub fn znx_negate_assign_portable(res: &mut [i64]) {
    for value in res {
        *value = value.wrapping_neg()
    }
}
