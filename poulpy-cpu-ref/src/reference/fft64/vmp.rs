use crate::{
    cast_mut,
    layouts::{
        Backend, HostDataMut, HostDataRef, MatZnxBackendRef, VecZnxDft, VecZnxDftBackendMut, VecZnxDftBackendRef,
        VecZnxDftToBackendMut, VecZnxToBackendRef, VmpPMatBackendMut, VmpPMatBackendRef, VmpPMatToBackendRef, ZnxView,
        ZnxViewMut,
    },
    reference::fft64::{
        reim::{ReimArith, ReimFFTExecute, ReimFFTTable},
        reim4::Reim4BlkMatVec,
    },
};

pub fn vmp_prepare_tmp_bytes(n: usize) -> usize {
    n * size_of::<i64>()
}

pub fn vmp_prepare<BE>(
    table: &ReimFFTTable<f64>,
    pmat: &mut VmpPMatBackendMut<'_, BE>,
    mat: &MatZnxBackendRef<'_, BE>,
    tmp: &mut [f64],
) where
    BE: Backend<ScalarPrep = f64> + ReimArith + Reim4BlkMatVec + ReimFFTExecute<ReimFFTTable<f64>, f64> + 'static,
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
    vmp_prepare_core::<BE>(table, pmat.raw_mut(), mat.raw(), nrows, ncols, tmp);
}

pub(crate) fn vmp_prepare_core<REIM>(
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
        assert_eq!(tmp.len(), vmp_prepare_tmp_bytes(n) / size_of::<i64>())
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

pub fn vmp_apply_dft_tmp_bytes(n: usize, a_size: usize, prows: usize, pcols_in: usize) -> usize {
    let row_max: usize = (a_size).min(prows);
    (16 + (n + 8) * row_max * pcols_in) * size_of::<f64>()
}

pub fn vmp_apply_dft<R, A, M, BE>(table: &ReimFFTTable<f64>, res: &mut R, a: &A, pmat: &M, tmp_bytes: &mut [f64])
where
    BE: Backend<ScalarPrep = f64> + ReimArith + Reim4BlkMatVec + ReimFFTExecute<ReimFFTTable<f64>, f64> + 'static,
    for<'x> BE::BufMut<'x>: HostDataMut,
    for<'x> BE::BufRef<'x>: HostDataRef,
    R: VecZnxDftToBackendMut<BE>,
    A: VecZnxToBackendRef<BE>,
    M: VmpPMatToBackendRef<BE>,
{
    let a = a.to_backend_ref();
    let pmat = pmat.to_backend_ref();

    let n: usize = a.n();
    let cols: usize = pmat.cols_in();
    let size: usize = a.size().min(pmat.rows());

    #[cfg(debug_assertions)]
    {
        assert!(tmp_bytes.len() >= vmp_apply_dft_tmp_bytes(n, size, pmat.rows(), cols));
        assert!(a.cols() <= cols);
    }

    let (data, tmp_bytes) = tmp_bytes.split_at_mut(BE::bytes_of_vec_znx_dft(n, cols, size));

    let mut a_dft: VecZnxDft<&mut [u8], BE> = VecZnxDft::from_data(cast_mut(data), n, cols, size);

    let offset: usize = cols - a.cols();
    for j in 0..cols {
        if j < offset {
            BE::reim_zero(a_dft.at_mut(j, 0));
        } else {
            BE::reim_from_znx(a_dft.at_mut(j, 0), a.at(offset + j, 0));
            BE::reim_dft_execute(table, a_dft.at_mut(j, 0));
        }
    }

    let mut res_ref = res.to_backend_mut();
    let nrows: usize = pmat.cols_in() * pmat.rows();
    let ncols: usize = pmat.cols_out() * pmat.size();
    vmp_apply_dft_to_dft_core::<true, BE>(n, res_ref.raw_mut(), a_dft.raw(), pmat.raw(), 0, nrows, ncols, tmp_bytes);
}

pub fn vmp_apply_dft_to_dft_tmp_bytes(a_size: usize, prows: usize, pcols_in: usize) -> usize {
    let row_max: usize = (a_size).min(prows);
    (16 + 8 * row_max * pcols_in) * size_of::<f64>()
}

pub fn vmp_zero<BE>(res: &mut VmpPMatBackendMut<'_, BE>)
where
    BE: Backend<ScalarPrep = f64>,
    for<'x> BE::BufMut<'x>: HostDataMut,
{
    res.raw_mut().fill(0.0);
}

pub fn vmp_apply_dft_to_dft<BE>(
    res: &mut VecZnxDftBackendMut<'_, BE>,
    a: &VecZnxDftBackendRef<'_, BE>,
    pmat: &VmpPMatBackendRef<'_, BE>,
    limb_offset: usize,
    tmp_bytes: &mut [f64],
) where
    BE: Backend<ScalarPrep = f64> + ReimArith + Reim4BlkMatVec,
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

    let n: usize = res.n();
    let nrows: usize = pmat.cols_in() * pmat.rows();
    let ncols: usize = pmat.cols_out() * pmat.size();

    let pmat_raw = pmat.raw();
    let a_raw = a.raw();
    let res_raw = res.raw_mut();

    vmp_apply_dft_to_dft_core::<true, BE>(
        n,
        res_raw,
        a_raw,
        pmat_raw,
        limb_offset * pmat.cols_out(),
        nrows,
        ncols,
        tmp_bytes,
    )
}

#[allow(clippy::too_many_arguments)]
fn vmp_apply_dft_to_dft_core<const OVERWRITE: bool, REIM>(
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
