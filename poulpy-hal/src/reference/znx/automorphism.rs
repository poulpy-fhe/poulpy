pub fn znx_automorphism_ref(p: i64, res: &mut [i64], a: &[i64]) {
    #[cfg(debug_assertions)]
    {
        assert_eq!(res.len(), a.len());
    }

    let n: usize = res.len();
    let mut k: usize = 0usize;
    let mask: usize = 2 * n - 1;
    let p_2n = (p & mask as i64) as usize;

    res[0] = a[0];
    for ai in a.iter().take(n).skip(1) {
        k = (k + p_2n) & mask;
        if k < n { res[k] = *ai } else { res[k - n] = -*ai }
    }
}
