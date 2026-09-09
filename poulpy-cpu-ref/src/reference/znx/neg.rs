#[inline(always)]
pub fn znx_negate_ref(res: &mut [i64], src: &[i64]) {
    #[cfg(debug_assertions)]
    {
        assert_eq!(res.len(), src.len())
    }

    for i in 0..res.len() {
        res[i] = -src[i]
    }
}

#[inline(always)]
pub fn znx_negate_assign_ref(res: &mut [i64]) {
    for value in res {
        *value = -*value
    }
}
