use crate::{
    layouts::{
        ArithmeticState, Backend, CoeffFitsIn, HostDataMut, HostDataRef, VecZnxBackendMut, VecZnxBackendRef, ZnxView, ZnxViewMut,
    },
    reference::znx::{ZnxAutomorphismRotate, ZnxZero},
};

/// Fused automorphism + rotation on a single column: computes
/// `res = X^k * auto(p, a)` (see [`znx_automorphism_rotate_ref`]).
///
/// Equivalent to applying [`vec_znx_automorphism`](super::vec_znx_automorphism)
/// with `p` followed by [`vec_znx_rotate`](super::vec_znx_rotate) with `k`, but
/// in a single pass and without an intermediate buffer.
pub fn vec_znx_automorphism_rotate<'r, 'a, BE, S: ArithmeticState>(
    p: i64,
    k: i64,
    res: &mut VecZnxBackendMut<'r, BE, S>,
    res_col: usize,
    a: &VecZnxBackendRef<'a, BE, impl CoeffFitsIn<S>>,
    a_col: usize,
) where
    BE: Backend<ZnxWord = i64> + ZnxAutomorphismRotate + ZnxZero,
    BE::BufMut<'r>: HostDataMut,
    BE::BufRef<'a>: HostDataRef,
{
    #[cfg(debug_assertions)]
    {
        assert_eq!(a.n(), res.n());
    }

    let min_size: usize = res.size().min(a.size());

    for j in 0..min_size {
        BE::znx_automorphism_rotate(p, k, res.at_mut(res_col, j), a.at(a_col, j));
    }

    for j in min_size..res.size() {
        BE::znx_zero(res.at_mut(res_col, j));
    }
}
