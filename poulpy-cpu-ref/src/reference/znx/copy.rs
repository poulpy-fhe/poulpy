#[inline(always)]
pub fn znx_copy_ref(res: &mut [i64], a: &[i64]) {
    #[cfg(debug_assertions)]
    {
        assert_eq!(res.len(), a.len())
    }
    res.copy_from_slice(a);
}
