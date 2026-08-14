use crate::{
    layouts::{
        Backend, HostDataMut, HostDataRef, MatZnxBackendRef, VecZnxDftBackendMut, VecZnxDftBackendRef, VmpPMatBackendMut,
        VmpPMatBackendRef, VmpTMatBackendMut, VmpTMatBackendRef, ZnxView, ZnxViewMut,
    },
    reference::fft64::{
        reim::{ReimArith, ReimFFTExecute, ReimFFTTable},
        reim4::Reim4BlkMatVec,
    },
};

pub fn vmp_prepare_pmat_tmp_bytes(n: usize) -> usize {
    n * size_of::<i64>()
}

pub fn vmp_prepare_pmat<BE>(
    table: &ReimFFTTable<f64>,
    pmat: &mut VmpPMatBackendMut<'_, BE>,
    mat: &MatZnxBackendRef<'_, BE>,
    tmp: &mut [f64],
) where
    BE: Backend<DftWord = f64, ZnxWord = i64> + ReimArith + Reim4BlkMatVec + ReimFFTExecute<ReimFFTTable<f64>, f64> + 'static,
    for<'x> <BE as Backend>::BufMut<'x>: HostDataMut,
    for<'x> <BE as Backend>::BufRef<'x>: HostDataRef,
{
    #[cfg(debug_assertions)]
    {
        assert_eq!(mat.n(), pmat.n());
        assert_eq!(
            pmat.cols_in(),
            mat.cols_in(),
            "pmat.cols_in: {} != mat.cols_in: {}",
            pmat.cols_in(),
            mat.cols_in()
        );
        assert_eq!(
            pmat.rows(),
            mat.rows(),
            "pmat.rows: {} != mat.rows: {}",
            pmat.rows(),
            mat.rows()
        );
        assert_eq!(
            pmat.cols_out(),
            mat.cols_out(),
            "pmat.cols_out: {} != mat.cols_out: {}",
            pmat.cols_out(),
            mat.cols_out()
        );
        assert_eq!(
            pmat.size(),
            mat.size(),
            "pmat.size: {} != mat.size: {}",
            pmat.size(),
            mat.size()
        );
    }

    let nrows: usize = mat.cols_in() * mat.rows();
    let ncols: usize = mat.cols_out() * mat.size();
    vmp_prepare_pmat_core::<BE>(table, pmat.raw_mut(), mat.raw(), nrows, ncols, tmp);
}

pub fn vmp_prepare_tmat_tmp_bytes(n: usize) -> usize {
    vmp_prepare_pmat_tmp_bytes(n)
}

/// Transforms `mat` into the hot-prep [`VmpTMat`](crate::layouts::VmpTMat).
///
/// FFT64 builds both tiers identically, so this shares
/// [`vmp_prepare_pmat_core`] with the packed tier.
pub fn vmp_prepare_tmat<BE>(
    table: &ReimFFTTable<f64>,
    tmat: &mut VmpTMatBackendMut<'_, BE>,
    mat: &MatZnxBackendRef<'_, BE>,
    tmp: &mut [f64],
) where
    BE: Backend<DftWord = f64, ZnxWord = i64> + ReimArith + Reim4BlkMatVec + ReimFFTExecute<ReimFFTTable<f64>, f64> + 'static,
    for<'x> <BE as Backend>::BufMut<'x>: HostDataMut,
    for<'x> <BE as Backend>::BufRef<'x>: HostDataRef,
{
    #[cfg(debug_assertions)]
    {
        assert_eq!(mat.n(), tmat.n());
        assert_eq!(tmat.cols_in(), mat.cols_in());
        assert_eq!(tmat.rows(), mat.rows());
        assert_eq!(tmat.cols_out(), mat.cols_out());
        assert_eq!(tmat.size(), mat.size());
    }

    let nrows: usize = mat.cols_in() * mat.rows();
    let ncols: usize = mat.cols_out() * mat.size();
    vmp_prepare_pmat_core::<BE>(table, tmat.raw_mut(), mat.raw(), nrows, ncols, tmp);
}

pub(crate) fn vmp_prepare_pmat_core<REIM>(
    table: &ReimFFTTable<f64>,
    pmat: &mut [f64],
    mat: &[i64],
    nrows: usize,
    ncols: usize,
    tmp: &mut [f64],
) where
    REIM: ReimArith + Reim4BlkMatVec + ReimFFTExecute<ReimFFTTable<f64>, f64>,
{
    let m: usize = table.m();
    let n: usize = m << 1;

    #[cfg(debug_assertions)]
    {
        assert!(n >= 8);
        assert_eq!(mat.len(), n * nrows * ncols);
        assert_eq!(pmat.len(), n * nrows * ncols);
        assert_eq!(tmp.len(), vmp_prepare_pmat_tmp_bytes(n) / size_of::<i64>())
    }

    let offset: usize = nrows * ncols * 8;

    for row_i in 0..nrows {
        for col_i in 0..ncols {
            let pos: usize = n * (row_i * ncols + col_i);

            REIM::reim_from_znx(tmp, &mat[pos..pos + n]);
            REIM::reim_dft_execute(table, tmp);

            let dst: &mut [f64] = if col_i == (ncols - 1) && !ncols.is_multiple_of(2) {
                &mut pmat[col_i * nrows * 8 + row_i * 8..]
            } else {
                &mut pmat[(col_i / 2) * (nrows * 16) + row_i * 16 + (col_i % 2) * 8..]
            };

            for blk_i in 0..m >> 2 {
                REIM::reim4_extract_1blk_contiguous(m, 1, blk_i, &mut dst[blk_i * offset..], tmp);
            }
        }
    }
}

pub fn vmp_apply_pmat_small_to_dft_tmp_bytes(n: usize, a_size: usize, prows: usize, pcols_in: usize) -> usize {
    let row_max: usize = (a_size).min(prows);
    (16 + (n + 8) * row_max * pcols_in) * size_of::<f64>()
}

pub fn vmp_apply_pmat_dft_to_dft_tmp_bytes(a_size: usize, prows: usize, pcols_in: usize) -> usize {
    let row_max: usize = (a_size).min(prows);
    (16 + 8 * row_max * pcols_in) * size_of::<f64>()
}

/// `res = pmat * a` (or `res += pmat * a` when `OVERWRITE` is false in the core),
/// with the matrix packed cold-prep.
pub fn vmp_apply_pmat_dft_to_dft<const OVERWRITE: bool, BE>(
    res: &mut VecZnxDftBackendMut<'_, BE>,
    pmat: &VmpPMatBackendRef<'_, BE>,
    a: &VecZnxDftBackendRef<'_, BE>,
    limb_offset: usize,
    tmp_bytes: &mut [f64],
) where
    BE: Backend<DftWord = f64, ZnxWord = i64> + ReimArith + Reim4BlkMatVec,
    for<'x> <BE as Backend>::BufMut<'x>: HostDataMut,
    for<'x> <BE as Backend>::BufRef<'x>: HostDataRef,
{
    #[cfg(debug_assertions)]
    {
        assert_eq!(res.n(), pmat.n());
        assert_eq!(a.n(), pmat.n());
        assert_eq!(res.cols(), pmat.cols_out());
        assert_eq!(a.cols(), pmat.cols_in());
    }

    dft_to_dft_inner::<OVERWRITE, BE>(
        res,
        pmat.raw(),
        pmat.rows(),
        pmat.cols_in(),
        pmat.cols_out(),
        pmat.size(),
        a,
        limb_offset,
        tmp_bytes,
    );
}

/// `res = tmat * a`, with the matrix transformed hot-prep.
pub fn vmp_apply_tmat_dft_to_dft<const OVERWRITE: bool, BE>(
    res: &mut VecZnxDftBackendMut<'_, BE>,
    tmat: &VmpTMatBackendRef<'_, BE>,
    a: &VecZnxDftBackendRef<'_, BE>,
    limb_offset: usize,
    tmp_bytes: &mut [f64],
) where
    BE: Backend<DftWord = f64, ZnxWord = i64> + ReimArith + Reim4BlkMatVec,
    for<'x> <BE as Backend>::BufMut<'x>: HostDataMut,
    for<'x> <BE as Backend>::BufRef<'x>: HostDataRef,
{
    #[cfg(debug_assertions)]
    {
        assert_eq!(res.n(), tmat.n());
        assert_eq!(a.n(), tmat.n());
        assert_eq!(res.cols(), tmat.cols_out());
        assert_eq!(a.cols(), tmat.cols_in());
    }

    dft_to_dft_inner::<OVERWRITE, BE>(
        res,
        tmat.raw(),
        tmat.rows(),
        tmat.cols_in(),
        tmat.cols_out(),
        tmat.size(),
        a,
        limb_offset,
        tmp_bytes,
    );
}

/// Shared body of the two `*_dft_to_dft` kernels.
///
/// The matrix arrives as its raw buffer plus shape so both tiers reach the same
/// core; the public kernels keep their concrete container types.
#[allow(clippy::too_many_arguments)]
fn dft_to_dft_inner<const OVERWRITE: bool, BE>(
    res: &mut VecZnxDftBackendMut<'_, BE>,
    mat_raw: &[f64],
    mat_rows: usize,
    mat_cols_in: usize,
    mat_cols_out: usize,
    mat_size: usize,
    a: &VecZnxDftBackendRef<'_, BE>,
    limb_offset: usize,
    tmp_bytes: &mut [f64],
) where
    BE: Backend<DftWord = f64, ZnxWord = i64> + ReimArith + Reim4BlkMatVec,
    for<'x> <BE as Backend>::BufMut<'x>: HostDataMut,
    for<'x> <BE as Backend>::BufRef<'x>: HostDataRef,
{
    let n: usize = res.n();
    let nrows: usize = mat_cols_in * mat_rows;
    let ncols: usize = mat_cols_out * mat_size;

    let a_raw = a.raw();
    let res_raw = res.raw_mut();

    // Split out the hot zero-offset path so LLVM sees a literal `0` instead of
    // the runtime expression `limb_offset * cols_out`. Blind rotation always
    // calls this with `limb_offset == 0`.
    if limb_offset == 0 {
        vmp_apply_pmat_dft_to_dft_core::<OVERWRITE, BE>(n, res_raw, a_raw, mat_raw, 0, nrows, ncols, tmp_bytes)
    } else {
        vmp_apply_pmat_dft_to_dft_core::<OVERWRITE, BE>(
            n,
            res_raw,
            a_raw,
            mat_raw,
            limb_offset * mat_cols_out,
            nrows,
            ncols,
            tmp_bytes,
        )
    }
}

#[allow(clippy::too_many_arguments)]
fn vmp_apply_pmat_dft_to_dft_core<const OVERWRITE: bool, REIM>(
    n: usize,
    res: &mut [f64],
    a: &[f64],
    pmat: &[f64],
    limb_offset: usize,
    nrows: usize,
    ncols: usize,
    tmp_bytes: &mut [f64],
) where
    REIM: ReimArith + Reim4BlkMatVec,
{
    #[cfg(debug_assertions)]
    {
        assert!(n >= 8);
        assert!(n.is_power_of_two());
        assert_eq!(pmat.len(), n * nrows * ncols);
        assert!(res.len() & (n - 1) == 0);
        assert!(a.len() & (n - 1) == 0);
    }

    let a_size: usize = a.len() / n;
    let res_size: usize = res.len() / n;

    let m: usize = n >> 1;

    let (mat2cols_output, extracted_blk) = tmp_bytes.split_at_mut(16);

    let row_max: usize = nrows.min(a_size);
    let col_max: usize = ncols.min(res_size);

    if limb_offset >= col_max {
        if OVERWRITE {
            REIM::reim_zero(res);
        }
        return;
    }

    for blk_i in 0..(m >> 2) {
        let mat_blk_start: &[f64] = &pmat[blk_i * (8 * nrows * ncols)..];

        REIM::reim4_extract_1blk_contiguous(m, row_max, blk_i, extracted_blk, a);

        if limb_offset.is_multiple_of(2) {
            for (col_res, col_pmat) in (0..).step_by(2).zip((limb_offset..col_max - 1).step_by(2)) {
                let col_offset: usize = col_pmat * (8 * nrows);
                REIM::reim4_mat2cols_prod(row_max, mat2cols_output, extracted_blk, &mat_blk_start[col_offset..]);
                REIM::reim4_save_2blks::<OVERWRITE>(m, blk_i, &mut res[col_res * n..], mat2cols_output);
            }
        } else {
            let col_offset: usize = (limb_offset - 1) * (8 * nrows);
            REIM::reim4_mat2cols_2ndcol_prod(row_max, mat2cols_output, extracted_blk, &mat_blk_start[col_offset..]);

            REIM::reim4_save_1blk::<OVERWRITE>(m, blk_i, res, mat2cols_output);

            for (col_res, col_pmat) in (1..).step_by(2).zip((limb_offset + 1..col_max - 1).step_by(2)) {
                let col_offset: usize = col_pmat * (8 * nrows);
                REIM::reim4_mat2cols_prod(row_max, mat2cols_output, extracted_blk, &mat_blk_start[col_offset..]);
                REIM::reim4_save_2blks::<OVERWRITE>(m, blk_i, &mut res[col_res * n..], mat2cols_output);
            }
        }

        if !col_max.is_multiple_of(2) {
            let last_col: usize = col_max - 1;
            let col_offset: usize = last_col * (8 * nrows);

            if last_col >= limb_offset {
                if ncols == col_max {
                    REIM::reim4_mat1col_prod(row_max, mat2cols_output, extracted_blk, &mat_blk_start[col_offset..]);
                } else {
                    REIM::reim4_mat2cols_prod(row_max, mat2cols_output, extracted_blk, &mat_blk_start[col_offset..]);
                }
                REIM::reim4_save_1blk::<OVERWRITE>(m, blk_i, &mut res[(last_col - limb_offset) * n..], mat2cols_output);
            }
        }
    }

    REIM::reim_zero(&mut res[col_max * n..]);
}
