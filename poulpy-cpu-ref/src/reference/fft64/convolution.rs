use crate::{
    layouts::{
        Backend, CnvPVecLBackendMut, CnvPVecLBackendRef, CnvPVecRBackendMut, CnvPVecRBackendRef, HostDataRef, VecZnxBackendRef,
        VecZnxBigBackendMut, VecZnxDftBackendMut, ZnxInfos, ZnxView, ZnxViewMut, ZnxZero,
    },
    reference::fft64::{
        reim::{ReimArith, ReimFFTExecute, ReimFFTTable},
        reim4::{Reim4BlkMatVec, Reim4Convolution},
        vec_znx_dft::vec_znx_dft_apply,
    },
};

pub fn convolution_prepare_left<BE>(
    table: &ReimFFTTable<f64>,
    res: &mut CnvPVecLBackendMut<'_, BE>,
    a: &VecZnxBackendRef<'_, BE>,
    mask: i64,
    tmp: &mut VecZnxDftBackendMut<'_, BE>,
) where
    BE: Backend<ScalarPrep = f64> + ReimArith + Reim4BlkMatVec + ReimFFTExecute<ReimFFTTable<f64>, f64> + 'static,
    for<'x> BE: Backend<BufRef<'x> = &'x [u8], BufMut<'x> = &'x mut [u8]>,
{
    convolution_prepare(table, res, a, mask, tmp)
}

pub fn convolution_prepare_right<BE>(
    table: &ReimFFTTable<f64>,
    res: &mut CnvPVecRBackendMut<'_, BE>,
    a: &VecZnxBackendRef<'_, BE>,
    mask: i64,
    tmp: &mut VecZnxDftBackendMut<'_, BE>,
) where
    BE: Backend<ScalarPrep = f64> + ReimArith + Reim4BlkMatVec + ReimFFTExecute<ReimFFTTable<f64>, f64> + 'static,
    for<'x> BE: Backend<BufRef<'x> = &'x [u8], BufMut<'x> = &'x mut [u8]>,
{
    convolution_prepare(table, res, a, mask, tmp)
}

fn convolution_prepare<R, BE>(
    table: &ReimFFTTable<f64>,
    res: &mut R,
    a: &VecZnxBackendRef<'_, BE>,
    mask: i64,
    tmp: &mut VecZnxDftBackendMut<'_, BE>,
) where
    BE: Backend<ScalarPrep = f64> + ReimArith + Reim4BlkMatVec + ReimFFTExecute<ReimFFTTable<f64>, f64> + 'static,
    for<'x> BE: Backend<BufRef<'x> = &'x [u8], BufMut<'x> = &'x mut [u8]>,
    R: ZnxInfos + ZnxViewMut<Scalar = BE::ScalarPrep>,
{
    let cols: usize = res.cols();
    assert_eq!(a.cols(), cols, "a.cols():{} != res.cols():{cols}", a.cols());

    let res_size: usize = res.size();
    let min_size: usize = res_size.min(a.size());

    let m: usize = a.n() >> 1;

    let n: usize = table.m() << 1;

    let res_raw: &mut [f64] = res.raw_mut();

    for i in 0..cols {
        // FFT all limbs (unmasked); the last active limb will be overwritten below.
        vec_znx_dft_apply(table, 1, 0, tmp, 0, a, i);

        // Re-compute only the last active limb with the mask applied.
        if min_size > 0 {
            let last = min_size - 1;
            BE::reim_from_znx_masked(tmp.at_mut(0, last), a.at(i, last), mask);
            BE::reim_dft_execute(table, tmp.at_mut(0, last));
        }

        let tmp_raw: &[f64] = tmp.raw();
        let res_col: &mut [f64] = &mut res_raw[i * n * res_size..];

        for blk_i in 0..m / 4 {
            BE::reim4_extract_1blk_contiguous(m, min_size, blk_i, &mut res_col[blk_i * res_size * 8..], tmp_raw);
            BE::reim_zero(&mut res_col[blk_i * res_size * 8 + min_size * 8..(blk_i + 1) * res_size * 8]);
        }
    }
}

pub fn convolution_prepare_self<BE>(
    table: &ReimFFTTable<f64>,
    left: &mut CnvPVecLBackendMut<'_, BE>,
    right: &mut CnvPVecRBackendMut<'_, BE>,
    a: &VecZnxBackendRef<'_, BE>,
    mask: i64,
    tmp: &mut VecZnxDftBackendMut<'_, BE>,
) where
    BE: Backend<ScalarPrep = f64> + ReimArith + Reim4BlkMatVec + ReimFFTExecute<ReimFFTTable<f64>, f64> + 'static,
    for<'x> BE: Backend<BufRef<'x> = &'x [u8], BufMut<'x> = &'x mut [u8]>,
{
    let cols: usize = left.cols();
    assert_eq!(a.cols(), cols, "a.cols():{} != left.cols():{cols}", a.cols());
    assert_eq!(right.cols(), cols, "right.cols():{} != left.cols():{cols}", right.cols());

    let left_size: usize = left.size();
    let right_size: usize = right.size();
    assert_eq!(
        left_size, right_size,
        "left.size():{} != right.size():{right_size}",
        left_size
    );
    let res_size: usize = left_size;
    let min_size: usize = res_size.min(a.size());

    let m: usize = a.n() >> 1;
    let n: usize = table.m() << 1;

    let left_raw: &mut [f64] = left.raw_mut();
    let right_raw: &mut [f64] = right.raw_mut();

    for i in 0..cols {
        // FFT all limbs (unmasked); the last active limb will be overwritten below.
        vec_znx_dft_apply(table, 1, 0, tmp, 0, a, i);

        // Re-compute only the last active limb with the mask applied.
        if min_size > 0 {
            let last = min_size - 1;
            BE::reim_from_znx_masked(tmp.at_mut(0, last), a.at(i, last), mask);
            BE::reim_dft_execute(table, tmp.at_mut(0, last));
        }

        let tmp_raw: &[f64] = tmp.raw();
        let left_col: &mut [f64] = &mut left_raw[i * n * res_size..];

        for blk_i in 0..m / 4 {
            BE::reim4_extract_1blk_contiguous(m, min_size, blk_i, &mut left_col[blk_i * res_size * 8..], tmp_raw);
            BE::reim_zero(&mut left_col[blk_i * res_size * 8 + min_size * 8..(blk_i + 1) * res_size * 8]);
        }

        // Copy from left to right (identical data for FFT64)
        let col_bytes: usize = n * res_size;
        let right_col: &mut [f64] = &mut right_raw[i * n * res_size..];
        right_col[..col_bytes].copy_from_slice(&left_col[..col_bytes]);
    }
}

pub fn convolution_by_const_apply_tmp_bytes(res_size: usize, a_size: usize, b_size: usize) -> usize {
    let min_size: usize = res_size.min(a_size + b_size - 1);
    size_of::<i64>() * (min_size + a_size) * 8
}

pub fn convolution_by_const_apply<BE>(
    cnv_offset: usize,
    res: &mut VecZnxBigBackendMut<'_, BE>,
    res_col: usize,
    a: &VecZnxBackendRef<'_, BE>,
    a_col: usize,
    b: &VecZnxBackendRef<'_, BE>,
    b_col: usize,
    b_coeff: usize,
    tmp: &mut [i64],
) where
    BE: Backend<ScalarBig = i64> + I64Ops + 'static,
    for<'x> BE: Backend<BufRef<'x> = &'x [u8]>,
    for<'x> <BE as Backend>::BufMut<'x>: crate::layouts::HostDataMut,
{
    let n: usize = res.n();
    assert_eq!(a.n(), n);

    let res_size: usize = res.size();
    let a_size: usize = a.size();
    let b_size: usize = b.size();

    let bound: usize = a_size + b_size - 1;
    let min_size: usize = res_size.min(bound);
    let offset: usize = cnv_offset.min(bound);

    let a_sl: usize = n * a.cols();
    let res_sl: usize = n * res.cols();

    let res_raw: &mut [i64] = res.raw_mut();
    let a_raw: &[i64] = a.raw();

    let a_idx: usize = n * a_col;
    let res_idx: usize = n * res_col;

    let (res_blk, a_blk) = tmp[..(min_size + a_size) * 8].split_at_mut(min_size * 8);
    let mut b_digits = vec![0i64; b_size];
    for (j, digit) in b_digits.iter_mut().enumerate() {
        *digit = b.at(b_col, j)[b_coeff];
    }

    for blk_i in 0..n / 8 {
        BE::i64_extract_1blk_contiguous(a_sl, a_idx, a_size, blk_i, a_blk, a_raw);
        BE::i64_convolution_by_const(res_blk, min_size, offset, a_blk, a_size, &b_digits);
        BE::i64_save_1blk_contiguous(res_sl, res_idx, min_size, blk_i, res_raw, res_blk);
    }

    for j in min_size..res_size {
        res.zero_at(res_col, j);
    }
}

pub fn convolution_apply_dft_tmp_bytes(res_size: usize, a_size: usize, b_size: usize) -> usize {
    let min_size: usize = res_size.min(a_size + b_size - 1);
    size_of::<f64>() * 8 * min_size
}

#[allow(clippy::too_many_arguments)]
pub fn convolution_apply_dft<BE>(
    cnv_offset: usize,
    res: &mut VecZnxDftBackendMut<'_, BE>,
    res_col: usize,
    a: &CnvPVecLBackendRef<'_, BE>,
    a_col: usize,
    b: &CnvPVecRBackendRef<'_, BE>,
    b_col: usize,
    tmp: &mut [f64],
) where
    BE: Backend<ScalarPrep = f64> + Reim4BlkMatVec + Reim4Convolution,
    for<'x> <BE as Backend>::BufRef<'x>: HostDataRef,
    for<'x> <BE as Backend>::BufMut<'x>: crate::layouts::HostDataMut,
{
    let n: usize = res.n();
    assert_eq!(a.n(), n);
    assert_eq!(b.n(), n);
    let m: usize = n >> 1;

    let res_size: usize = res.size();
    let a_size: usize = a.size();
    let b_size: usize = b.size();

    let bound: usize = a_size + b_size - 1;
    let min_size: usize = res_size.min(bound);
    let offset: usize = cnv_offset.min(bound);

    let dst: &mut [f64] = res.raw_mut();
    let a_raw: &[f64] = a.raw();
    let b_raw: &[f64] = b.raw();

    let mut a_idx: usize = a_col * n * a_size;
    let mut b_idx: usize = b_col * n * b_size;
    let a_offset: usize = a_size * 8;
    let b_offset: usize = b_size * 8;
    for blk_i in 0..m / 4 {
        BE::reim4_convolution(tmp, min_size, offset, &a_raw[a_idx..], a_size, &b_raw[b_idx..], b_size);
        BE::reim4_save_1blk_contiguous(m, min_size, blk_i, dst, tmp);
        a_idx += a_offset;
        b_idx += b_offset;
    }

    for j in min_size..res_size {
        res.zero_at(res_col, j);
    }
}

pub fn convolution_pairwise_apply_dft_tmp_bytes(res_size: usize, a_size: usize, b_size: usize) -> usize {
    convolution_apply_dft_tmp_bytes(res_size, a_size, b_size) + (a_size + b_size) * size_of::<f64>() * 8
}

#[allow(clippy::too_many_arguments)]
pub fn convolution_pairwise_apply_dft<BE>(
    cnv_offset: usize,
    res: &mut VecZnxDftBackendMut<'_, BE>,
    res_col: usize,
    a: &CnvPVecLBackendRef<'_, BE>,
    b: &CnvPVecRBackendRef<'_, BE>,
    col_i: usize,
    col_j: usize,
    tmp: &mut [f64],
) where
    BE: Backend<ScalarPrep = f64> + ReimArith + Reim4BlkMatVec + Reim4Convolution,
    for<'x> <BE as Backend>::BufRef<'x>: HostDataRef,
    for<'x> <BE as Backend>::BufMut<'x>: crate::layouts::HostDataMut,
{
    if col_i == col_j {
        convolution_apply_dft(cnv_offset, res, res_col, a, col_i, b, col_j, tmp);
        return;
    }

    let n: usize = res.n();
    let m: usize = n >> 1;

    assert_eq!(a.n(), n);
    assert_eq!(b.n(), n);

    let res_size: usize = res.size();
    let a_size: usize = a.size();
    let b_size: usize = b.size();

    assert_eq!(
        tmp.len(),
        convolution_pairwise_apply_dft_tmp_bytes(res_size, a_size, b_size) / size_of::<f64>()
    );

    let bound: usize = a_size + b_size - 1;
    let min_size: usize = res_size.min(bound);
    let offset: usize = cnv_offset.min(bound);

    let res_raw: &mut [f64] = res.raw_mut();
    let a_raw: &[f64] = a.raw();
    let b_raw: &[f64] = b.raw();

    let a_row_size: usize = a_size * 8;
    let b_row_size: usize = b_size * 8;

    let mut a0_idx: usize = col_i * n * a_size;
    let mut a1_idx: usize = col_j * n * a_size;
    let mut b0_idx: usize = col_i * n * b_size;
    let mut b1_idx: usize = col_j * n * b_size;

    let (tmp_a, tmp) = tmp.split_at_mut(a_row_size);
    let (tmp_b, tmp_res) = tmp.split_at_mut(b_row_size);

    for blk_i in 0..m / 4 {
        let a0: &[f64] = &a_raw[a0_idx..];
        let a1: &[f64] = &a_raw[a1_idx..];
        let b0: &[f64] = &b_raw[b0_idx..];
        let b1: &[f64] = &b_raw[b1_idx..];

        BE::reim_add(tmp_a, &a0[..a_row_size], &a1[..a_row_size]);
        BE::reim_add(tmp_b, &b0[..b_row_size], &b1[..b_row_size]);

        BE::reim4_convolution(tmp_res, min_size, offset, tmp_a, a_size, tmp_b, b_size);
        BE::reim4_save_1blk_contiguous(m, min_size, blk_i, res_raw, tmp_res);

        a0_idx += a_row_size;
        a1_idx += a_row_size;
        b0_idx += b_row_size;
        b1_idx += b_row_size;
    }

    for j in min_size..res_size {
        res.zero_at(res_col, j);
    }
}

pub trait I64Ops {
    fn i64_extract_1blk_contiguous(n: usize, offset: usize, rows: usize, blk: usize, dst: &mut [i64], src: &[i64]) {
        i64_extract_1blk_contiguous_ref(n, offset, rows, blk, dst, src)
    }

    fn i64_save_1blk_contiguous(n: usize, offset: usize, rows: usize, blk: usize, dst: &mut [i64], src: &[i64]) {
        i64_save_1blk_contiguous_ref(n, offset, rows, blk, dst, src)
    }

    fn i64_convolution_by_const_1coeff(k: usize, dst: &mut [i64; 8], a: &[i64], a_size: usize, b: &[i64]) {
        i64_convolution_by_const_1coeff_ref(k, dst, a, a_size, b)
    }

    fn i64_convolution_by_const_2coeffs(k: usize, dst: &mut [i64; 16], a: &[i64], a_size: usize, b: &[i64]) {
        i64_convolution_by_const_2coeffs_ref(k, dst, a, a_size, b)
    }

    fn i64_convolution_by_const(dst: &mut [i64], dst_size: usize, offset: usize, a: &[i64], a_size: usize, b: &[i64]) {
        assert!(a_size > 0);

        for k in (0..dst_size - 1).step_by(2) {
            Self::i64_convolution_by_const_2coeffs(k + offset, as_arr_i64_mut(&mut dst[8 * k..]), a, a_size, b);
        }

        if !dst_size.is_multiple_of(2) {
            let k: usize = dst_size - 1;
            Self::i64_convolution_by_const_1coeff(k + offset, as_arr_i64_mut(&mut dst[8 * k..]), a, a_size, b);
        }
    }
}

#[inline(always)]
pub fn i64_extract_1blk_contiguous_ref(n: usize, offset: usize, rows: usize, blk: usize, dst: &mut [i64], src: &[i64]) {
    debug_assert!(blk < (n >> 3));
    debug_assert!(dst.len() >= rows * 8, "dst.len(): {} < rows*8: {}", dst.len(), 8 * rows);

    let offset: usize = offset + (blk << 3);

    // src = 8-values chunks spaced by n, dst = sequential 8-values chunks
    let src_rows = src.chunks_exact(n).take(rows);
    let dst_chunks = dst.chunks_exact_mut(8).take(rows);

    for (dst_chunk, src_row) in dst_chunks.zip(src_rows) {
        dst_chunk.copy_from_slice(&src_row[offset..offset + 8]);
    }
}

#[inline(always)]
pub fn i64_save_1blk_contiguous_ref(n: usize, offset: usize, rows: usize, blk: usize, dst: &mut [i64], src: &[i64]) {
    debug_assert!(blk < (n >> 3));
    debug_assert!(src.len() >= rows * 8);

    let offset: usize = offset + (blk << 3);

    // dst = 4-values chunks spaced by m, src = sequential 4-values chunks
    let dst_rows = dst.chunks_exact_mut(n).take(rows);
    let src_chunks = src.chunks_exact(8).take(rows);

    for (dst_row, src_chunk) in dst_rows.zip(src_chunks) {
        dst_row[offset..offset + 8].copy_from_slice(src_chunk);
    }
}

#[inline(always)]
pub fn i64_convolution_by_const_1coeff_ref(k: usize, dst: &mut [i64; 8], a: &[i64], a_size: usize, b: &[i64]) {
    dst.fill(0);

    let b_size: usize = b.len();

    if k >= a_size + b_size {
        return;
    }
    let j_min: usize = k.saturating_sub(a_size - 1);
    let j_max: usize = (k + 1).min(b_size);

    for j in j_min..j_max {
        let ai: &[i64] = &a[8 * (k - j)..];
        let bi: i64 = b[j];

        dst[0] = dst[0].wrapping_add(ai[0].wrapping_mul(bi));
        dst[1] = dst[1].wrapping_add(ai[1].wrapping_mul(bi));
        dst[2] = dst[2].wrapping_add(ai[2].wrapping_mul(bi));
        dst[3] = dst[3].wrapping_add(ai[3].wrapping_mul(bi));
        dst[4] = dst[4].wrapping_add(ai[4].wrapping_mul(bi));
        dst[5] = dst[5].wrapping_add(ai[5].wrapping_mul(bi));
        dst[6] = dst[6].wrapping_add(ai[6].wrapping_mul(bi));
        dst[7] = dst[7].wrapping_add(ai[7].wrapping_mul(bi));
    }
}

#[allow(dead_code)]
#[inline(always)]
pub(crate) fn as_arr_i64<const SIZE: usize>(x: &[i64]) -> &[i64; SIZE] {
    debug_assert!(x.len() >= SIZE, "x.len():{} < size:{}", x.len(), SIZE);
    unsafe { &*(x.as_ptr() as *const [i64; SIZE]) }
}

#[allow(dead_code)]
#[inline(always)]
pub(crate) fn as_arr_i64_mut<const SIZE: usize>(x: &mut [i64]) -> &mut [i64; SIZE] {
    debug_assert!(x.len() >= SIZE, "x.len():{} < size:{}", x.len(), SIZE);
    unsafe { &mut *(x.as_mut_ptr() as *mut [i64; SIZE]) }
}

#[inline(always)]
pub fn i64_convolution_by_const_2coeffs_ref(k: usize, dst: &mut [i64; 16], a: &[i64], a_size: usize, b: &[i64]) {
    i64_convolution_by_const_1coeff_ref(k, as_arr_i64_mut(&mut dst[..8]), a, a_size, b);
    i64_convolution_by_const_1coeff_ref(k + 1, as_arr_i64_mut(&mut dst[8..]), a, a_size, b);
}
