#[inline(always)]
pub fn znx_copy_portable(res: &mut [i64], a: &[i64]) {
    {
        assert_eq!(res.len(), a.len())
    }
    res.copy_from_slice(a);
}
