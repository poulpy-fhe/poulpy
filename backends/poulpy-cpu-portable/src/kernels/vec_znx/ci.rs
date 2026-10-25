use crate::layouts::{Backend, HostDataMut, HostDataRef, VecZnxBackendMut, VecZnxBackendRef, ZnxView, ZnxViewMut};

/// Embeds the conjugate-invariant `a[a_col]` of degree `N` into `res[res_col]` of degree `2N`:
/// `a_0 + Σ a_i (X^i + X^-i)`, with `X^-i = -X^(2N-i)`.
pub fn vec_znx_ci_embed_portable<'r, 'a, BE>(
    res: &mut VecZnxBackendMut<'r, BE>,
    res_col: usize,
    a: &VecZnxBackendRef<'a, BE>,
    a_col: usize,
) where
    BE: Backend<ZnxWord = i64>,
    BE::BufMut<'r>: HostDataMut,
    BE::BufRef<'a>: HostDataRef,
{
    poulpy_hal::layouts::assert_dense(res, "vec_znx_ci_embed");
    poulpy_hal::layouts::assert_dense(a, "vec_znx_ci_embed");
    let n: usize = a.n();
    assert_eq!(res.n(), 2 * n, "vec_znx_ci_embed: res degree must be twice the degree of a");
    let min_size: usize = a.size().min(res.size());
    for j in 0..min_size {
        let (res, a) = (res.at_mut(res_col, j), a.at(a_col, j));
        res[..n].copy_from_slice(a);
        res[n] = 0;
        for i in 1..n {
            res[2 * n - i] = a[i].wrapping_neg();
        }
    }
    for j in min_size..res.size() {
        res.at_mut(res_col, j).fill(0);
    }
}

/// Writes the relative trace of `a[a_col]` of degree `2N` into `res[res_col]` of
/// degree `N`: the compressed `a(X) + a(X^-1)`.
pub fn vec_znx_ci_trace_portable<'r, 'a, BE>(
    res: &mut VecZnxBackendMut<'r, BE>,
    res_col: usize,
    a: &VecZnxBackendRef<'a, BE>,
    a_col: usize,
) where
    BE: Backend<ZnxWord = i64>,
    BE::BufMut<'r>: HostDataMut,
    BE::BufRef<'a>: HostDataRef,
{
    poulpy_hal::layouts::assert_dense(res, "vec_znx_ci_trace");
    poulpy_hal::layouts::assert_dense(a, "vec_znx_ci_trace");
    let n: usize = res.n();
    assert_eq!(a.n(), 2 * n, "vec_znx_ci_trace: a degree must be twice the degree of res");
    let min_size: usize = a.size().min(res.size());
    for j in 0..min_size {
        let (res, a) = (res.at_mut(res_col, j), a.at(a_col, j));
        res[0] = a[0].wrapping_mul(2);
        for i in 1..n {
            res[i] = a[i].wrapping_sub(a[2 * n - i]);
        }
    }
    for j in min_size..res.size() {
        res.at_mut(res_col, j).fill(0);
    }
}
