use crate::{
    layouts::{
        Backend, FitsIn, HostDataMut, HostDataRef, NormalizationState, VecZnxBackendMut, VecZnxBackendRef, ZnxView, ZnxViewMut,
    },
    reference::znx::{ZnxCopy, ZnxZero},
};

pub fn vec_znx_copy<'r, 'a, BE, S: NormalizationState>(
    res: &mut VecZnxBackendMut<'r, BE, S>,
    res_col: usize,
    a: &VecZnxBackendRef<'a, BE, impl FitsIn<S>>,
    a_col: usize,
) where
    BE: Backend<ZnxWord = i64> + ZnxCopy + ZnxZero,
    BE::BufMut<'r>: HostDataMut,
    BE::BufRef<'a>: HostDataRef,
{
    #[cfg(debug_assertions)]
    {
        assert_eq!(res.n(), a.n())
    }

    let res_size = res.size();
    let a_size = a.size();

    let min_size = res_size.min(a_size);

    for j in 0..min_size {
        BE::znx_copy(res.at_mut(res_col, j), a.at(a_col, j));
    }

    for j in min_size..res_size {
        BE::znx_zero(res.at_mut(res_col, j));
    }
}

pub fn vec_znx_extract_coeff<'r, 'a, BE, S: NormalizationState>(
    res: &mut VecZnxBackendMut<'r, BE, S>,
    res_col: usize,
    a: &VecZnxBackendRef<'a, BE, impl FitsIn<S>>,
    a_col: usize,
    a_coeff: usize,
) where
    BE: Backend<ZnxWord = i64>,
    BE::BufMut<'r>: HostDataMut,
    BE::BufRef<'a>: HostDataRef,
{
    #[cfg(debug_assertions)]
    {
        assert_eq!(
            res.n(),
            1,
            "vec_znx_extract_coeff expects a 1-coeff destination, got {}",
            res.n()
        );
        assert!(a_coeff < a.n(), "a_coeff: {a_coeff} >= a.n(): {}", a.n());
    }

    let min_size = res.size().min(a.size());

    for limb in 0..min_size {
        let dst = res.at_mut(res_col, limb);
        dst.fill(0);
        dst[0] = a.at(a_col, limb)[a_coeff];
    }

    for limb in min_size..res.size() {
        res.at_mut(res_col, limb).fill(0);
    }
}

/// Per-limb square transpose: `res.at(c, j)[k] = a.at(k, j)[c]`.
///
/// Requires `res.n() == a.cols()` and `res.cols() == a.n()`. Limbs beyond
/// `min(res.size(), a.size())` are zero-filled on `res`.
pub fn vec_znx_transpose<'r, 'a, BE, S: NormalizationState>(
    res: &mut VecZnxBackendMut<'r, BE, S>,
    a: &VecZnxBackendRef<'a, BE, impl FitsIn<S>>,
) where
    BE: Backend<ZnxWord = i64>,
    BE::BufMut<'r>: HostDataMut,
    BE::BufRef<'a>: HostDataRef,
{
    assert_eq!(
        res.n(),
        a.cols(),
        "vec_znx_transpose: res.n() ({}) must equal a.cols() ({})",
        res.n(),
        a.cols()
    );
    assert_eq!(
        res.cols(),
        a.n(),
        "vec_znx_transpose: res.cols() ({}) must equal a.n() ({})",
        res.cols(),
        a.n()
    );

    let n_a = a.n();
    let cols_a = a.cols();
    let limb_stride = n_a * cols_a;
    let min_size = res.size().min(a.size());
    let res_size = res.size();

    // Layout (limb-major, column-minor): scalar offset of (col=c, limb=j) is
    // `n * (j * cols + c)`, with `n` consecutive coefficients. Source has
    // (n_a, cols_a) and destination has (n=cols_a, cols=n_a), so both share
    // the same per-limb stride `n_a * cols_a`.
    let src = a.raw();
    let dst = res.raw_mut();

    for limb in 0..min_size {
        let base = limb * limb_stride;
        for c in 0..cols_a {
            let src_off = base + c * n_a;
            for k in 0..n_a {
                // res row k (length cols_a) holds the transposed column.
                dst[base + k * cols_a + c] = src[src_off + k];
            }
        }
    }

    let tail = res_size * limb_stride;
    let head = min_size * limb_stride;
    if head < tail {
        dst[head..tail].fill(0);
    }
}

#[allow(clippy::too_many_arguments)]
pub fn vec_znx_copy_range<'r, 'a, BE, S: NormalizationState>(
    res: &mut VecZnxBackendMut<'r, BE, S>,
    res_col: usize,
    res_limb: usize,
    res_offset: usize,
    a: &VecZnxBackendRef<'a, BE, impl FitsIn<S>>,
    a_col: usize,
    a_limb: usize,
    a_offset: usize,
    len: usize,
) where
    BE: Backend<ZnxWord = i64> + ZnxCopy,
    BE::BufMut<'r>: HostDataMut,
    BE::BufRef<'a>: HostDataRef,
{
    assert!(res_offset + len <= res.n());
    assert!(a_offset + len <= a.n());

    BE::znx_copy(
        &mut res.at_mut(res_col, res_limb)[res_offset..res_offset + len],
        &a.at(a_col, a_limb)[a_offset..a_offset + len],
    );
}
