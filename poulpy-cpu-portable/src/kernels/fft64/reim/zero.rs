pub fn reim_zero_portable(res: &mut [f64]) {
    res.fill(0.);
}

pub fn reim_copy_portable(res: &mut [f64], a: &[f64]) {
    {
        assert_eq!(res.len(), a.len())
    }
    res.copy_from_slice(a);
}
