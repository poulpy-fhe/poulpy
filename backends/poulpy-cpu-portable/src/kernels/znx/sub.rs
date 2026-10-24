pub fn znx_sub_portable(res: &mut [i64], a: &[i64], b: &[i64]) {
    {
        assert_eq!(res.len(), a.len());
        assert_eq!(res.len(), b.len());
    }

    let n: usize = res.len();
    for i in 0..n {
        res[i] = a[i].wrapping_sub(b[i]);
    }
}

pub fn znx_sub_assign_portable(res: &mut [i64], a: &[i64]) {
    {
        assert_eq!(res.len(), a.len());
    }

    let n: usize = res.len();
    for i in 0..n {
        res[i] = res[i].wrapping_sub(a[i]);
    }
}

pub fn znx_sub_negate_assign_portable(res: &mut [i64], a: &[i64]) {
    {
        assert_eq!(res.len(), a.len());
    }

    let n: usize = res.len();
    for i in 0..n {
        res[i] = a[i].wrapping_sub(res[i]);
    }
}
