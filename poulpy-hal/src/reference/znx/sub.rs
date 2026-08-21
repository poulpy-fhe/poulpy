pub fn znx_sub_ref(res: &mut [i64], a: &[i64], b: &[i64]) {
    #[cfg(debug_assertions)]
    {
        assert_eq!(res.len(), a.len());
        assert_eq!(res.len(), b.len());
    }

    let n: usize = res.len();
    for i in 0..n {
        res[i] = a[i] - b[i];
    }
}

pub fn znx_sub_assign_ref(res: &mut [i64], a: &[i64]) {
    #[cfg(debug_assertions)]
    {
        assert_eq!(res.len(), a.len());
    }

    let n: usize = res.len();
    for i in 0..n {
        res[i] -= a[i];
    }
}

pub fn znx_sub_negate_assign_ref(res: &mut [i64], a: &[i64]) {
    #[cfg(debug_assertions)]
    {
        assert_eq!(res.len(), a.len());
    }

    let n: usize = res.len();
    for i in 0..n {
        res[i] = a[i] - res[i];
    }
}
