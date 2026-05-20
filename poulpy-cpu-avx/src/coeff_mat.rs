#![allow(unsafe_op_in_unsafe_fn)]

use poulpy_cpu_ref::hal_defaults::CoeffMatPMatDefault;
use poulpy_hal::layouts::{
    Backend, CoeffMatPMatBackendMut, CoeffMatPMatBackendRef, HostDataMut, HostDataRef, Module, ScratchArena, VecZnxBackendRef,
    VecZnxBigBackendMut, ZnxView, ZnxViewMut,
};

pub(crate) fn coeff_mat_prepare_tmp_bytes<BE>(
    module: &Module<BE>,
    rows: usize,
    cols_in: usize,
    cols_out: usize,
    size: usize,
) -> usize
where
    BE: Backend + CoeffMatPMatDefault<BE>,
    BE::OwnedBuf: HostDataMut,
    for<'x> BE::BufMut<'x>: HostDataMut,
    for<'x> BE::BufRef<'x>: HostDataRef,
{
    <BE as CoeffMatPMatDefault<BE>>::coeff_mat_prepare_tmp_bytes_default(module, rows, cols_in, cols_out, size)
}

pub(crate) fn coeff_mat_prepare<BE>(
    module: &Module<BE>,
    res: &mut CoeffMatPMatBackendMut<'_, BE>,
    matrix: &VecZnxBackendRef<'_, BE>,
    scratch: &mut ScratchArena<'_, BE>,
) where
    BE: Backend + CoeffMatPMatDefault<BE>,
    BE::OwnedBuf: HostDataMut,
    for<'x> BE::BufMut<'x>: HostDataMut,
    for<'x> BE::BufRef<'x>: HostDataRef,
{
    <BE as CoeffMatPMatDefault<BE>>::coeff_mat_prepare_default(module, res, matrix, scratch)
}

pub(crate) fn coeff_mat_apply_big_tmp_bytes(_rows_in: usize, _rows_out: usize) -> usize {
    0
}

#[inline]
fn packed_col_block_offset(nrows: usize, ncols: usize, col: usize, block: usize) -> usize {
    let block_stride = nrows * ncols * 8;
    if col == ncols - 1 && !ncols.is_multiple_of(2) {
        col * nrows * 8 + block * block_stride
    } else {
        (col / 2) * (nrows * 16) + (col % 2) * 8 + block * block_stride
    }
}

#[target_feature(enable = "avx2")]
unsafe fn mullo_i64x4(a: core::arch::x86_64::__m256i, b: core::arch::x86_64::__m256i) -> core::arch::x86_64::__m256i {
    use core::arch::x86_64::{
        _mm256_add_epi64, _mm256_and_si256, _mm256_mul_epu32, _mm256_set1_epi64x, _mm256_slli_epi64, _mm256_srli_epi64,
    };
    let mask = _mm256_set1_epi64x(0xffff_ffff);
    let a_lo = _mm256_and_si256(a, mask);
    let b_lo = _mm256_and_si256(b, mask);
    let a_hi = _mm256_srli_epi64::<32>(a);
    let b_hi = _mm256_srli_epi64::<32>(b);
    let lo = _mm256_mul_epu32(a_lo, b_lo);
    let cross = _mm256_add_epi64(_mm256_mul_epu32(a_lo, b_hi), _mm256_mul_epu32(a_hi, b_lo));
    _mm256_add_epi64(lo, _mm256_slli_epi64::<32>(cross))
}

#[target_feature(enable = "avx2")]
unsafe fn coeff_mat1col_product_i64(rows_in: usize, nrows: usize, ncols: usize, col: usize, a: &[i64], pmat: &[i64]) -> i64 {
    use core::arch::x86_64::{__m256i, _mm256_add_epi64, _mm256_loadu_si256, _mm256_setzero_si256, _mm256_storeu_si256};

    let mut acc = _mm256_setzero_si256();
    let blocks = rows_in / 4;
    for block in 0..blocks {
        let a_v = _mm256_loadu_si256(a.as_ptr().add(4 * block) as *const __m256i);
        let p_off = packed_col_block_offset(nrows, ncols, col, block / 2) + (block % 2) * 4;
        let p_v = _mm256_loadu_si256(pmat.as_ptr().add(p_off) as *const __m256i);
        acc = _mm256_add_epi64(acc, mullo_i64x4(a_v, p_v));
    }

    let mut lanes = [0i64; 4];
    _mm256_storeu_si256(lanes.as_mut_ptr() as *mut __m256i, acc);
    let mut res = lanes.into_iter().fold(0i64, i64::wrapping_add);
    for (coeff, ai) in a.iter().enumerate().take(rows_in).skip(4 * blocks) {
        let p_off = packed_col_block_offset(nrows, ncols, col, coeff / 8) + coeff % 8;
        res = res.wrapping_add(ai.wrapping_mul(pmat[p_off]));
    }
    res
}

#[target_feature(enable = "avx2")]
unsafe fn coeff_mat2cols_product_i64(
    rows_in: usize,
    nrows: usize,
    ncols: usize,
    col: usize,
    a: &[i64],
    pmat: &[i64],
) -> [i64; 2] {
    use core::arch::x86_64::{__m256i, _mm256_add_epi64, _mm256_loadu_si256, _mm256_setzero_si256, _mm256_storeu_si256};

    let mut acc0 = _mm256_setzero_si256();
    let mut acc1 = _mm256_setzero_si256();
    let blocks = rows_in / 4;
    for block in 0..blocks {
        let a_v = _mm256_loadu_si256(a.as_ptr().add(4 * block) as *const __m256i);
        let p_off = packed_col_block_offset(nrows, ncols, col, block / 2) + (block % 2) * 4;
        let p0 = _mm256_loadu_si256(pmat.as_ptr().add(p_off) as *const __m256i);
        let p1 = _mm256_loadu_si256(pmat.as_ptr().add(p_off + 8) as *const __m256i);
        acc0 = _mm256_add_epi64(acc0, mullo_i64x4(a_v, p0));
        acc1 = _mm256_add_epi64(acc1, mullo_i64x4(a_v, p1));
    }

    let mut lanes0 = [0i64; 4];
    let mut lanes1 = [0i64; 4];
    _mm256_storeu_si256(lanes0.as_mut_ptr() as *mut __m256i, acc0);
    _mm256_storeu_si256(lanes1.as_mut_ptr() as *mut __m256i, acc1);
    let mut res = [
        lanes0.into_iter().fold(0i64, i64::wrapping_add),
        lanes1.into_iter().fold(0i64, i64::wrapping_add),
    ];
    for (coeff, ai) in a.iter().enumerate().take(rows_in).skip(4 * blocks) {
        let p_off = packed_col_block_offset(nrows, ncols, col, coeff / 8) + coeff % 8;
        res[0] = res[0].wrapping_add(ai.wrapping_mul(pmat[p_off]));
        res[1] = res[1].wrapping_add(ai.wrapping_mul(pmat[p_off + 8]));
    }
    res
}

#[target_feature(enable = "avx2")]
unsafe fn mul_i64x4_to_i128_bits(
    a: core::arch::x86_64::__m256i,
    b: core::arch::x86_64::__m256i,
    lo_out: &mut [u64; 4],
    hi_out: &mut [u64; 4],
) {
    use core::arch::x86_64::{
        __m256i, _mm256_add_epi64, _mm256_and_si256, _mm256_cmpgt_epi64, _mm256_mul_epu32, _mm256_or_si256, _mm256_set1_epi64x,
        _mm256_setzero_si256, _mm256_slli_epi64, _mm256_srli_epi64, _mm256_storeu_si256, _mm256_sub_epi64,
    };
    let mask = _mm256_set1_epi64x(0xffff_ffff);
    let a_lo = _mm256_and_si256(a, mask);
    let b_lo = _mm256_and_si256(b, mask);
    let a_hi = _mm256_srli_epi64::<32>(a);
    let b_hi = _mm256_srli_epi64::<32>(b);

    let p0 = _mm256_mul_epu32(a_lo, b_lo);
    let p1 = _mm256_mul_epu32(a_lo, b_hi);
    let p2 = _mm256_mul_epu32(a_hi, b_lo);
    let p3 = _mm256_mul_epu32(a_hi, b_hi);

    let t = _mm256_add_epi64(
        _mm256_add_epi64(_mm256_srli_epi64::<32>(p0), _mm256_and_si256(p1, mask)),
        _mm256_and_si256(p2, mask),
    );
    let lo = _mm256_or_si256(_mm256_and_si256(p0, mask), _mm256_slli_epi64::<32>(t));
    let mut hi = _mm256_add_epi64(
        _mm256_add_epi64(p3, _mm256_srli_epi64::<32>(p1)),
        _mm256_add_epi64(_mm256_srli_epi64::<32>(p2), _mm256_srli_epi64::<32>(t)),
    );

    let zero = _mm256_setzero_si256();
    let a_neg = _mm256_cmpgt_epi64(zero, a);
    let b_neg = _mm256_cmpgt_epi64(zero, b);
    hi = _mm256_sub_epi64(hi, _mm256_and_si256(a_neg, b));
    hi = _mm256_sub_epi64(hi, _mm256_and_si256(b_neg, a));

    _mm256_storeu_si256(lo_out.as_mut_ptr() as *mut __m256i, lo);
    _mm256_storeu_si256(hi_out.as_mut_ptr() as *mut __m256i, hi);
}

#[inline]
fn i128_from_bits(lo: u64, hi: u64) -> i128 {
    (((hi as u128) << 64) | lo as u128) as i128
}

#[target_feature(enable = "avx2")]
unsafe fn coeff_mat1col_product_i128(rows_in: usize, nrows: usize, ncols: usize, col: usize, a: &[i64], pmat: &[i64]) -> i128 {
    use core::arch::x86_64::{__m256i, _mm256_loadu_si256};

    let mut acc = 0i128;
    let blocks = rows_in / 4;
    for block in 0..blocks {
        let a_v = _mm256_loadu_si256(a.as_ptr().add(4 * block) as *const __m256i);
        let p_off = packed_col_block_offset(nrows, ncols, col, block / 2) + (block % 2) * 4;
        let p_v = _mm256_loadu_si256(pmat.as_ptr().add(p_off) as *const __m256i);
        let mut lo = [0u64; 4];
        let mut hi = [0u64; 4];
        mul_i64x4_to_i128_bits(a_v, p_v, &mut lo, &mut hi);
        for lane in 0..4 {
            acc = acc.wrapping_add(i128_from_bits(lo[lane], hi[lane]));
        }
    }

    for (coeff, ai) in a.iter().enumerate().take(rows_in).skip(4 * blocks) {
        let p_off = packed_col_block_offset(nrows, ncols, col, coeff / 8) + coeff % 8;
        acc = acc.wrapping_add((*ai as i128).wrapping_mul(pmat[p_off] as i128));
    }
    acc
}

#[target_feature(enable = "avx2")]
unsafe fn coeff_mat2cols_product_i128(
    rows_in: usize,
    nrows: usize,
    ncols: usize,
    col: usize,
    a: &[i64],
    pmat: &[i64],
) -> [i128; 2] {
    use core::arch::x86_64::{__m256i, _mm256_loadu_si256};

    let mut acc = [0i128; 2];
    let blocks = rows_in / 4;
    for block in 0..blocks {
        let a_v = _mm256_loadu_si256(a.as_ptr().add(4 * block) as *const __m256i);
        let p_off = packed_col_block_offset(nrows, ncols, col, block / 2) + (block % 2) * 4;
        for (idx, col_off) in [0usize, 8].into_iter().enumerate() {
            let p_v = _mm256_loadu_si256(pmat.as_ptr().add(p_off + col_off) as *const __m256i);
            let mut lo = [0u64; 4];
            let mut hi = [0u64; 4];
            mul_i64x4_to_i128_bits(a_v, p_v, &mut lo, &mut hi);
            for lane in 0..4 {
                acc[idx] = acc[idx].wrapping_add(i128_from_bits(lo[lane], hi[lane]));
            }
        }
    }
    for (coeff, ai) in a.iter().enumerate().take(rows_in).skip(4 * blocks) {
        let p_off = packed_col_block_offset(nrows, ncols, col, coeff / 8) + coeff % 8;
        let a_value = *ai as i128;
        acc[0] = acc[0].wrapping_add(a_value.wrapping_mul(pmat[p_off] as i128));
        acc[1] = acc[1].wrapping_add(a_value.wrapping_mul(pmat[p_off + 8] as i128));
    }
    acc
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn coeff_mat_apply_big_i64<BE: Backend<ScalarBig = i64>>(
    res: &mut VecZnxBigBackendMut<'_, BE>,
    res_limb: usize,
    pmat: &CoeffMatPMatBackendRef<'_, BE>,
    pmat_limb: usize,
    a: &VecZnxBackendRef<'_, BE>,
    a_col: usize,
    a_limb: usize,
    rows_in: usize,
    rows_out: usize,
) where
    for<'x> BE::BufMut<'x>: HostDataMut,
    for<'x> BE::BufRef<'x>: HostDataRef,
{
    apply_big(
        res,
        res_limb,
        pmat,
        pmat_limb,
        a,
        a_col,
        a_limb,
        rows_in,
        rows_out,
        |rows_in, nrows, ncols, col, a, p| unsafe { coeff_mat1col_product_i64(rows_in, nrows, ncols, col, a, p) },
        |rows_in, nrows, ncols, col, a, p| unsafe { coeff_mat2cols_product_i64(rows_in, nrows, ncols, col, a, p) },
    );
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn coeff_mat_apply_big_i128<BE: Backend<ScalarBig = i128>>(
    res: &mut VecZnxBigBackendMut<'_, BE>,
    res_limb: usize,
    pmat: &CoeffMatPMatBackendRef<'_, BE>,
    pmat_limb: usize,
    a: &VecZnxBackendRef<'_, BE>,
    a_col: usize,
    a_limb: usize,
    rows_in: usize,
    rows_out: usize,
) where
    for<'x> BE::BufMut<'x>: HostDataMut,
    for<'x> BE::BufRef<'x>: HostDataRef,
{
    apply_big(
        res,
        res_limb,
        pmat,
        pmat_limb,
        a,
        a_col,
        a_limb,
        rows_in,
        rows_out,
        |rows_in, nrows, ncols, col, a, p| unsafe { coeff_mat1col_product_i128(rows_in, nrows, ncols, col, a, p) },
        |rows_in, nrows, ncols, col, a, p| unsafe { coeff_mat2cols_product_i128(rows_in, nrows, ncols, col, a, p) },
    );
}

#[allow(clippy::too_many_arguments)]
fn apply_big<BE, F1, F2>(
    res: &mut VecZnxBigBackendMut<'_, BE>,
    res_limb: usize,
    pmat: &CoeffMatPMatBackendRef<'_, BE>,
    pmat_limb: usize,
    a: &VecZnxBackendRef<'_, BE>,
    a_col: usize,
    a_limb: usize,
    rows_in: usize,
    rows_out: usize,
    dot1: F1,
    dot2: F2,
) where
    BE: Backend,
    BE::ScalarBig: Copy + From<i64>,
    for<'x> BE::BufMut<'x>: HostDataMut,
    for<'x> BE::BufRef<'x>: HostDataRef,
    F1: Fn(usize, usize, usize, usize, &[i64], &[i64]) -> BE::ScalarBig,
    F2: Fn(usize, usize, usize, usize, &[i64], &[i64]) -> [BE::ScalarBig; 2],
{
    assert!(rows_in <= a.n(), "coeff_mat_apply_big: rows_in exceeds input degree");
    assert!(rows_in <= pmat.n(), "coeff_mat_apply_big: rows_in exceeds prepared degree");
    assert!(
        rows_out <= pmat.cols_out(),
        "coeff_mat_apply_big: rows_out exceeds prepared columns"
    );
    assert!(rows_out <= res.cols(), "coeff_mat_apply_big: rows_out exceeds result columns");
    assert!(res_limb < res.size(), "coeff_mat_apply_big: result limb out of bounds");
    assert!(pmat_limb < pmat.size(), "coeff_mat_apply_big: prepared limb out of bounds");

    for out_row in 0..rows_out {
        res.at_mut(out_row, res_limb).fill(BE::ScalarBig::from(0));
    }

    let a_coeffs = a.at(a_col, a_limb);
    let pmat_raw = pmat.raw();
    let pmat_col_base = pmat_limb * pmat.cols_out();
    let pmat_nrows = pmat.rows() * pmat.cols_in();
    let pmat_ncols = pmat.cols_out() * pmat.size();
    let mut out_row = 0;

    if rows_out > 0 && !pmat_col_base.is_multiple_of(2) {
        let value = dot1(rows_in, pmat_nrows, pmat_ncols, pmat_col_base, a_coeffs, pmat_raw);
        res.at_mut(0, res_limb)[0] = value;
        out_row = 1;
    }

    while out_row + 1 < rows_out {
        let values = dot2(rows_in, pmat_nrows, pmat_ncols, pmat_col_base + out_row, a_coeffs, pmat_raw);
        res.at_mut(out_row, res_limb)[0] = values[0];
        res.at_mut(out_row + 1, res_limb)[0] = values[1];
        out_row += 2;
    }

    if out_row < rows_out {
        let value = dot1(rows_in, pmat_nrows, pmat_ncols, pmat_col_base + out_row, a_coeffs, pmat_raw);
        res.at_mut(out_row, res_limb)[0] = value;
    }
}
