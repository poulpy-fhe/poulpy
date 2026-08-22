use crate::reference::znx::{ZnxCopy, ZnxNegate};

pub fn znx_rotate<ZNXARI: ZnxNegate + ZnxCopy>(p: i64, res: &mut [i64], src: &[i64]) {
    #[cfg(debug_assertions)]
    {
        assert_eq!(res.len(), src.len());
    }

    let n: usize = res.len();

    let mp_2n: usize = (p & (2 * n as i64 - 1)) as usize; // -p % 2n
    let mp_1n: usize = mp_2n & (n - 1); // -p % n
    let mp_1n_neg: usize = n - mp_1n; //  p % n
    let neg_first: bool = mp_2n < n;

    let (dst1, dst2) = res.split_at_mut(mp_1n);
    let (src1, src2) = src.split_at(mp_1n_neg);

    if neg_first {
        ZNXARI::znx_negate(dst1, src2);
        ZNXARI::znx_copy(dst2, src1);
    } else {
        ZNXARI::znx_copy(dst1, src2);
        ZNXARI::znx_negate(dst2, src1);
    }
}
