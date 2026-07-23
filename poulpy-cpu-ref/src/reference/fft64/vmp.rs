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

/// Folds discarded fixed-point output limbs of a prepared one-column FFT64 VMP
/// matrix into its last retained limb.
///
/// Keeping this operation beside `vmp_prepare_core` makes the prepared layout
/// an FFT64 backend concern. Scheme crates can request the arithmetic operation
/// through the HAL without depending on paired-column/block offsets.
pub fn vmp_pmat_fold_output_limbs<BE>(res: &mut VmpPMatBackendMut<'_, BE>, src: &VmpPMatBackendRef<'_, BE>, base2k: usize)
where
    BE: Backend<ScalarPrep = f64>,
    for<'x> BE::BufMut<'x>: HostDataMut,
    for<'x> BE::BufRef<'x>: HostDataRef,
{
    assert_eq!(res.n(), src.n(), "folded VMP degree mismatch");
    assert_eq!(res.rows(), src.rows(), "folded VMP row mismatch");
    assert_eq!(res.cols_in(), src.cols_in(), "folded VMP input-column mismatch");
    assert_eq!(res.cols_out(), 1, "folding currently requires one output column");
    assert_eq!(src.cols_out(), 1, "folding currently requires one output column");
    assert!(base2k >= 1, "folding base2k must be positive");

    let src_size = src.size();
    let out_size = res.size();
    assert!(out_size >= 1, "folded VMP must retain at least one limb");
    assert!(out_size <= src_size, "folded VMP cannot grow the limb count");

    let n = src.n();
    assert!(
        n >= 8 && n.is_multiple_of(8),
        "FFT64 prepared folding requires n divisible by 8"
    );
    let nrows = src.rows() * src.cols_in();
    let blocks = n / 8;
    let src_blk_stride = 8 * nrows * src_size;
    let dst_blk_stride = 8 * nrows * out_size;

    let src_raw = src.raw();
    let dst_raw = res.raw_mut();
    assert_eq!(src_raw.len(), blocks * src_blk_stride);
    assert_eq!(dst_raw.len(), blocks * dst_blk_stride);

    // Within-block offset of (limb, row) in the paired-column FFT64 pmat.
    let limb_offset = |limb: usize, row: usize, size: usize| -> usize {
        if limb == size - 1 && !size.is_multiple_of(2) {
            limb * nrows * 8 + row * 8
        } else {
            (limb / 2) * (nrows * 16) + row * 16 + (limb % 2) * 8
        }
    };

    for block in 0..blocks {
        let src_block = block * src_blk_stride;
        let dst_block = block * dst_blk_stride;
        for row in 0..nrows {
            for limb in 0..out_size {
                let src_at = src_block + limb_offset(limb, row, src_size);
                let dst_at = dst_block + limb_offset(limb, row, out_size);
                dst_raw[dst_at..dst_at + 8].copy_from_slice(&src_raw[src_at..src_at + 8]);
            }
            for limb in out_size..src_size {
                let distance = limb - (out_size - 1);
                let scale = (-((distance * base2k) as f64)).exp2();
                let src_at = src_block + limb_offset(limb, row, src_size);
                let dst_at = dst_block + limb_offset(out_size - 1, row, out_size);
                for lane in 0..8 {
                    dst_raw[dst_at + lane] += scale * src_raw[src_at + lane];
                }
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

    // Split out the hot zero-offset path so LLVM sees a literal `0` instead of
    // the runtime expression `limb_offset * pmat.cols_out()`. Blind rotation
    // always calls this with `limb_offset == 0`.
    if limb_offset == 0 {
        vmp_apply_dft_to_dft_core::<true, BE>(n, res_raw, a_raw, pmat_raw, 0, nrows, ncols, tmp_bytes)
    } else {
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
