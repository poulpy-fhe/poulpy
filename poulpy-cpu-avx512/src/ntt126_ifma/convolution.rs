//! Polynomial convolution AVX512 kernels for [`NTT126Ifma`](crate::NTT126Ifma).
//!
//! Mirrors the block-outer pack-then-multiply structure used by the AVX
//! [`NTT120Avx`](poulpy_cpu_avx::NTT120Avx) convolution: for each x2 NTT
//! block, the left and right operand rows are first gathered into contiguous
//! scratch buffers (with the right operand in reversed row order), then each
//! output limb consumes a contiguous window of those buffers via the
//! [`vec_mat1col_product_x2_bbc_ifma`] kernel.
//!
//! Moving the block loop outermost turns the inner j-sum into a straight-line
//! accumulation over adjacent rows, lets the prefetcher stream ahead, and
//! keeps the hot BBC kernel's 4-way unrolling saturated.

use bytemuck::{cast_slice, cast_slice_mut};
use std::mem::size_of;

use crate::ntt126_ifma::{
    bbc_meta::Bbc126IfmaMeta,
    module::handle,
    primes::Primes42,
    tables::Ntt126IfmaTable,
    traits::{Ntt126IfmaCFromB, Ntt126IfmaDFTExecute, Ntt126IfmaFromZnx64},
};
use poulpy_cpu_ref::reference::ntt120::types::Q120bScalar;
use poulpy_hal::layouts::{
    CnvPVecLBackendMut, CnvPVecLBackendRef, CnvPVecRBackendMut, CnvPVecRBackendRef, Module, VecZnxBackendRef,
    VecZnxBigBackendMut, VecZnxDftBackendMut, ZnxView, ZnxViewMut,
};

use super::mat_vec_ifma::vec_mat1col_product_x2_bbc_ifma;

use crate::NTT126Ifma;
use core::arch::x86_64::{__m512i, _mm_sfence, _mm512_add_epi64, _mm512_loadu_si512, _mm512_storeu_si512};

// ─────────────────────────────────────────────────────────────────────────────
// Pack kernels
// ─────────────────────────────────────────────────────────────────────────────
//
// In IFMA layout each NTT coefficient is 4 × u64 (3 active primes + 1 padding
// lane), so one x2-block (two consecutive coefficients) is 8 u64 = one
// `__m512i`. Packing therefore reduces to copying one `__m512i` per row, with
// optional row reversal or pairwise summation.

/// Gather a row range of prep-scalar x2-blocks into a contiguous buffer.
///
/// `a` is a column-start prep-scalar slice with row stride `row_stride` (in `u64`
/// units). For each row, block `blk` (8 u64 values) is copied to `dst`.
/// `dst` must hold at least `8 * row_count` u64.
#[target_feature(enable = "avx512f")]
#[inline]
unsafe fn pack_left_1blk_x2_ifma(dst: &mut [u64], a: &[u64], row_count: usize, row_stride: usize, blk: usize) {
    debug_assert!(dst.len() >= 8 * row_count);
    debug_assert!(a.len() >= row_stride.saturating_mul(row_count.saturating_sub(1)) + 8 * blk + 8);
    unsafe {
        let mut dst_ptr = dst.as_mut_ptr() as *mut __m512i;
        let mut a_ptr = a.as_ptr().add(8 * blk) as *const __m512i;
        for _ in 0..row_count {
            _mm512_storeu_si512(dst_ptr, _mm512_loadu_si512(a_ptr));
            a_ptr = (a_ptr as *const u64).add(row_stride) as *const __m512i;
            dst_ptr = dst_ptr.add(1);
        }
    }
}

/// Gather a row range of prep-scalar x2-blocks in reversed row order.
///
/// Same layout as [`pack_left_1blk_x2_ifma`] but row 0 in `dst` receives the
/// source's last row. This lets each output limb consume a contiguous window
/// `[b_size - j_max ..]` inside the packed buffer.
#[target_feature(enable = "avx512f")]
#[inline]
unsafe fn pack_right_1blk_x2_ifma(dst: &mut [u64], a: &[u64], row_count: usize, row_stride: usize, blk: usize) {
    debug_assert!(dst.len() >= 8 * row_count);
    debug_assert!(a.len() >= row_stride.saturating_mul(row_count.saturating_sub(1)) + 8 * blk + 8);
    unsafe {
        let mut dst_ptr = dst.as_mut_ptr() as *mut __m512i;
        let mut a_ptr = a.as_ptr().add(row_stride * row_count.saturating_sub(1) + 8 * blk) as *const __m512i;
        for _ in 0..row_count {
            _mm512_storeu_si512(dst_ptr, _mm512_loadu_si512(a_ptr));
            a_ptr = (a_ptr as *const u64).sub(row_stride) as *const __m512i;
            dst_ptr = dst_ptr.add(1);
        }
    }
}

/// Pairwise pack: gather and lane-add the x2-blocks of two columns.
///
/// Inputs are in `[0, 2Q)` (left side), so the sum is in `[0, 4Q) < 2^45`,
/// which stays inside the 52-bit VPMADD52 input window.
#[target_feature(enable = "avx512f")]
#[inline]
unsafe fn pairwise_pack_left_1blk_x2_ifma(
    dst: &mut [u64],
    a: &[u64],
    b: &[u64],
    row_count: usize,
    row_stride: usize,
    blk: usize,
) {
    debug_assert!(dst.len() >= 8 * row_count);
    debug_assert!(a.len() >= row_stride.saturating_mul(row_count.saturating_sub(1)) + 8 * blk + 8);
    debug_assert!(b.len() >= row_stride.saturating_mul(row_count.saturating_sub(1)) + 8 * blk + 8);
    unsafe {
        let mut dst_ptr = dst.as_mut_ptr() as *mut __m512i;
        let mut a_ptr = a.as_ptr().add(8 * blk) as *const __m512i;
        let mut b_ptr = b.as_ptr().add(8 * blk) as *const __m512i;
        for _ in 0..row_count {
            _mm512_storeu_si512(
                dst_ptr,
                _mm512_add_epi64(_mm512_loadu_si512(a_ptr), _mm512_loadu_si512(b_ptr)),
            );
            a_ptr = (a_ptr as *const u64).add(row_stride) as *const __m512i;
            b_ptr = (b_ptr as *const u64).add(row_stride) as *const __m512i;
            dst_ptr = dst_ptr.add(1);
        }
    }
}

/// Pairwise pack in reversed row order. Right-side inputs are in `[0, Q)`,
/// so the sum is in `[0, 2Q) < 2^44`, well within the madd52 window.
#[target_feature(enable = "avx512f")]
#[inline]
unsafe fn pairwise_pack_right_1blk_x2_ifma(
    dst: &mut [u64],
    a: &[u64],
    b: &[u64],
    row_count: usize,
    row_stride: usize,
    blk: usize,
) {
    debug_assert!(dst.len() >= 8 * row_count);
    debug_assert!(a.len() >= row_stride.saturating_mul(row_count.saturating_sub(1)) + 8 * blk + 8);
    debug_assert!(b.len() >= row_stride.saturating_mul(row_count.saturating_sub(1)) + 8 * blk + 8);
    unsafe {
        let mut dst_ptr = dst.as_mut_ptr() as *mut __m512i;
        let mut a_ptr = a.as_ptr().add(row_stride * row_count.saturating_sub(1) + 8 * blk) as *const __m512i;
        let mut b_ptr = b.as_ptr().add(row_stride * row_count.saturating_sub(1) + 8 * blk) as *const __m512i;
        for _ in 0..row_count {
            _mm512_storeu_si512(
                dst_ptr,
                _mm512_add_epi64(_mm512_loadu_si512(a_ptr), _mm512_loadu_si512(b_ptr)),
            );
            a_ptr = (a_ptr as *const u64).sub(row_stride) as *const __m512i;
            b_ptr = (b_ptr as *const u64).sub(row_stride) as *const __m512i;
            dst_ptr = dst_ptr.add(1);
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Scratch accounting
// ─────────────────────────────────────────────────────────────────────────────

/// Scratch bytes required by [`cnv_apply_dft_ifma`].
///
/// Stores packed x2-block rows for both operands: 8 u64 per row.
pub(crate) fn cnv_apply_dft_ifma_tmp_bytes(a_size: usize, b_size: usize) -> usize {
    8 * (a_size + b_size) * size_of::<u64>()
}

/// Scratch bytes required by [`cnv_pairwise_apply_dft_ifma`].
///
/// Same requirement as the non-pairwise variant since the pairwise path
/// delegates to it when `col_0 == col_1`.
pub(crate) fn cnv_pairwise_apply_dft_ifma_tmp_bytes(res_size: usize, a_size: usize, b_size: usize) -> usize {
    if a_size == 0 || b_size == 0 || res_size == 0 {
        0
    } else {
        cnv_apply_dft_ifma_tmp_bytes(a_size, b_size)
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// cnv_apply_dft
// ─────────────────────────────────────────────────────────────────────────────

/// DFT-domain bivariate convolution `res[k] = Σ a[j] ⊙ b[k−j]` for the IFMA
/// backend.
///
/// Iterates over x2 NTT blocks outermost: for each block the left and right
/// rows are packed once into contiguous scratch buffers, then each output
/// limb `k` consumes a contiguous `ell`-row window to compute the BBC
/// accumulation via [`vec_mat1col_product_x2_bbc_ifma`].
#[allow(clippy::too_many_arguments)]
#[target_feature(enable = "avx512ifma,avx512vl")]
pub(crate) unsafe fn cnv_apply_dft_ifma(
    res: &mut VecZnxDftBackendMut<'_, NTT126Ifma>,
    cnv_offset: usize,
    res_col: usize,
    a: &CnvPVecLBackendRef<'_, NTT126Ifma>,
    a_col: usize,
    b: &CnvPVecRBackendRef<'_, NTT126Ifma>,
    b_col: usize,
    tmp: &mut [u8],
) {
    let n = res.n();
    let res_size = res.size();
    let a_size = a.size();
    let b_size = b.size();
    if res_size == 0 || a_size == 0 || b_size == 0 {
        for j in 0..res_size {
            cast_slice_mut::<_, u64>(res.at_mut(res_col, j)).fill(0);
        }
        return;
    }

    let bound = a_size + b_size - 1;
    let offset = cnv_offset.min(bound);
    let min_size = res_size.min((bound + 1).saturating_sub(offset));

    let meta = Bbc126IfmaMeta::<Primes42>::new();
    let a_cols = a.cols();
    let b_cols = b.cols();
    let n_blks = n / 2;
    let row_stride_a = 4 * n * a_cols;
    let row_stride_b = 4 * n * b_cols;
    let a_col_offset = 4 * n * a_col;
    let b_col_offset = 4 * n * b_col;
    let a_raw_u64: &[u64] = cast_slice(a.raw());
    let b_raw_u64: &[u64] = cast_slice(b.raw());

    let (prefix, tmp_u64, suffix) = unsafe { tmp.align_to_mut::<u64>() };
    debug_assert!(prefix.is_empty());
    debug_assert!(suffix.is_empty());
    debug_assert!(tmp_u64.len() >= 8 * (a_size + b_size));
    let (a_tmp, b_tmp) = tmp_u64.split_at_mut(8 * a_size);

    for blk in 0..n_blks {
        unsafe {
            pack_left_1blk_x2_ifma(a_tmp, &a_raw_u64[a_col_offset..], a_size, row_stride_a, blk);
            pack_right_1blk_x2_ifma(b_tmp, &b_raw_u64[b_col_offset..], b_size, row_stride_b, blk);
        }

        for k in 0..min_size {
            let k_abs = k + offset;
            let j_min = k_abs.saturating_sub(a_size - 1);
            let j_max = (k_abs + 1).min(b_size);
            let ell = j_max - j_min;
            let a_start = k_abs + 1 - j_max;
            let b_start = b_size - j_max;

            let res_u64: &mut [u64] = cast_slice_mut(res.at_mut(res_col, k));
            unsafe {
                vec_mat1col_product_x2_bbc_ifma::<true>(
                    &meta,
                    ell,
                    &mut res_u64[8 * blk..8 * blk + 8],
                    cast_slice(&a_tmp[8 * a_start..]),
                    cast_slice(&b_tmp[8 * b_start..]),
                );
            }
        }
    }

    for j in min_size..res_size {
        res.at_mut(res_col, j).fill(Q120bScalar([0; 4]));
    }
    // Order the non-temporal stores from the kernel against any subsequent
    // load of `res` (e.g. by the next stage of the FHE pipeline).
    _mm_sfence();
}

// ─────────────────────────────────────────────────────────────────────────────
// cnv_pairwise_apply_dft
// ─────────────────────────────────────────────────────────────────────────────

/// Pairwise DFT-domain convolution:
///
/// ```text
/// res[k] = Σ_{j=j_min..j_max}
///             (a[col_0, k−j] + a[col_1, k−j])
///           ⊙ (b[col_0, j]   + b[col_1, j])
/// ```
///
/// When `col_0 == col_1`, delegates to [`cnv_apply_dft_ifma`].
#[allow(clippy::too_many_arguments)]
#[target_feature(enable = "avx512ifma,avx512vl")]
pub(crate) unsafe fn cnv_pairwise_apply_dft_ifma(
    res: &mut VecZnxDftBackendMut<'_, NTT126Ifma>,
    cnv_offset: usize,
    res_col: usize,
    a: &CnvPVecLBackendRef<'_, NTT126Ifma>,
    b: &CnvPVecRBackendRef<'_, NTT126Ifma>,
    col_0: usize,
    col_1: usize,
    tmp: &mut [u8],
) {
    if col_0 == col_1 {
        unsafe { cnv_apply_dft_ifma(res, cnv_offset, res_col, a, col_0, b, col_1, tmp) };
        return;
    }

    let n = res.n();
    let res_size = res.size();
    let a_size = a.size();
    let b_size = b.size();
    if res_size == 0 || a_size == 0 || b_size == 0 {
        for j in 0..res_size {
            cast_slice_mut::<_, u64>(res.at_mut(res_col, j)).fill(0);
        }
        return;
    }

    let bound = a_size + b_size - 1;
    let offset = cnv_offset.min(bound);
    let min_size = res_size.min((bound + 1).saturating_sub(offset));

    let meta = Bbc126IfmaMeta::<Primes42>::new();
    let a_cols = a.cols();
    let b_cols = b.cols();
    let n_blks = n / 2;
    let row_stride_a = 4 * n * a_cols;
    let row_stride_b = 4 * n * b_cols;
    let a_col_offset_0 = 4 * n * col_0;
    let a_col_offset_1 = 4 * n * col_1;
    let b_col_offset_0 = 4 * n * col_0;
    let b_col_offset_1 = 4 * n * col_1;
    let a_raw_u64: &[u64] = cast_slice(a.raw());
    let b_raw_u64: &[u64] = cast_slice(b.raw());

    let (prefix, tmp_u64, suffix) = unsafe { tmp.align_to_mut::<u64>() };
    debug_assert!(prefix.is_empty());
    debug_assert!(suffix.is_empty());
    debug_assert!(tmp_u64.len() >= 8 * (a_size + b_size));
    let (a_tmp, b_tmp) = tmp_u64.split_at_mut(8 * a_size);

    for blk in 0..n_blks {
        unsafe {
            pairwise_pack_left_1blk_x2_ifma(
                a_tmp,
                &a_raw_u64[a_col_offset_0..],
                &a_raw_u64[a_col_offset_1..],
                a_size,
                row_stride_a,
                blk,
            );
            pairwise_pack_right_1blk_x2_ifma(
                b_tmp,
                &b_raw_u64[b_col_offset_0..],
                &b_raw_u64[b_col_offset_1..],
                b_size,
                row_stride_b,
                blk,
            );
        }

        for k in 0..min_size {
            let k_abs = k + offset;
            let j_min = k_abs.saturating_sub(a_size - 1);
            let j_max = (k_abs + 1).min(b_size);
            let ell = j_max - j_min;
            let a_start = k_abs + 1 - j_max;
            let b_start = b_size - j_max;

            let res_u64: &mut [u64] = cast_slice_mut(res.at_mut(res_col, k));
            unsafe {
                vec_mat1col_product_x2_bbc_ifma::<true>(
                    &meta,
                    ell,
                    &mut res_u64[8 * blk..8 * blk + 8],
                    cast_slice(&a_tmp[8 * a_start..]),
                    cast_slice(&b_tmp[8 * b_start..]),
                );
            }
        }
    }

    for j in min_size..res_size {
        res.at_mut(res_col, j).fill(Q120bScalar([0; 4]));
    }
    _mm_sfence();
}

// ─────────────────────────────────────────────────────────────────────────────
// Convolution prep paths
// ─────────────────────────────────────────────────────────────────────────────
//
// `prepare_left/right/self` materialise NTT-domain operands suitable for
// downstream consumption by [`cnv_apply_dft_ifma`]; `by_const_apply` is the
// scalar-by-VecZnx convolution that lands in `VecZnxBig` (no NTT).
//
// They keep the same algorithmic shape as the NTT120 reference path — the
// SIMD work happens inside `ntt126_ifma_dft_execute` / `ntt126_ifma_from_znx64`
// / `ntt126_ifma_c_from_b`, which the [`NTT126Ifma`] backend implements with
// the IFMA kernels.

pub(crate) fn cnv_prepare_left_tmp_bytes(_n: usize) -> usize {
    0
}

pub(crate) fn cnv_prepare_left(
    module: &Module<NTT126Ifma>,
    res: &mut CnvPVecLBackendMut<'_, NTT126Ifma>,
    a: &VecZnxBackendRef<'_, NTT126Ifma>,
    mask: i64,
    _tmp: &mut [u8],
) {
    let table = &handle(module).table_ntt;
    let cols = res.cols();
    let res_size = res.size();
    let min_size = res_size.min(a.size());

    for col in 0..cols {
        for j in 0..min_size.saturating_sub(1) {
            let res_u64: &mut [u64] = cast_slice_mut(res.at_mut(col, j));
            NTT126Ifma::ntt126_ifma_from_znx64(res_u64, a.at(col, j));
            <NTT126Ifma as Ntt126IfmaDFTExecute<Ntt126IfmaTable<Primes42>>>::ntt126_ifma_dft_execute(table, res_u64);
        }
        if min_size > 0 {
            let last = min_size - 1;
            let res_u64: &mut [u64] = cast_slice_mut(res.at_mut(col, last));
            NTT126Ifma::ntt126_ifma_from_znx64_masked(res_u64, a.at(col, last), mask);
            <NTT126Ifma as Ntt126IfmaDFTExecute<Ntt126IfmaTable<Primes42>>>::ntt126_ifma_dft_execute(table, res_u64);
        }
        for j in min_size..res_size {
            cast_slice_mut::<_, u64>(res.at_mut(col, j)).fill(0);
        }
    }
}

pub(crate) fn cnv_prepare_right_tmp_bytes(n: usize) -> usize {
    4 * n * size_of::<u64>()
}

pub(crate) fn cnv_prepare_right(
    module: &Module<NTT126Ifma>,
    res: &mut CnvPVecRBackendMut<'_, NTT126Ifma>,
    a: &VecZnxBackendRef<'_, NTT126Ifma>,
    mask: i64,
    tmp: &mut [u64],
) {
    let n = res.n();
    let table = &handle(module).table_ntt;
    let cols = res.cols();
    let res_size = res.size();
    let min_size = res_size.min(a.size());

    for col in 0..cols {
        for j in 0..min_size.saturating_sub(1) {
            NTT126Ifma::ntt126_ifma_from_znx64(tmp, a.at(col, j));
            <NTT126Ifma as Ntt126IfmaDFTExecute<Ntt126IfmaTable<Primes42>>>::ntt126_ifma_dft_execute(table, tmp);
            let res_u32: &mut [u32] = cast_slice_mut(res.at_mut(col, j));
            NTT126Ifma::ntt126_ifma_c_from_b(n, res_u32, tmp);
        }
        if min_size > 0 {
            let last = min_size - 1;
            NTT126Ifma::ntt126_ifma_from_znx64_masked(tmp, a.at(col, last), mask);
            <NTT126Ifma as Ntt126IfmaDFTExecute<Ntt126IfmaTable<Primes42>>>::ntt126_ifma_dft_execute(table, tmp);
            let res_u32: &mut [u32] = cast_slice_mut(res.at_mut(col, last));
            NTT126Ifma::ntt126_ifma_c_from_b(n, res_u32, tmp);
        }
        for j in min_size..res_size {
            cast_slice_mut::<_, u64>(res.at_mut(col, j)).fill(0);
        }
    }
}

pub(crate) fn cnv_prepare_self_tmp_bytes(n: usize) -> usize {
    cnv_prepare_left_tmp_bytes(n)
}

pub(crate) fn cnv_prepare_self(
    module: &Module<NTT126Ifma>,
    left: &mut CnvPVecLBackendMut<'_, NTT126Ifma>,
    right: &mut CnvPVecRBackendMut<'_, NTT126Ifma>,
    a: &VecZnxBackendRef<'_, NTT126Ifma>,
    mask: i64,
    _tmp: &mut [u8],
) {
    let table = &handle(module).table_ntt;
    let n = left.n();
    let cols = left.cols();
    let res_size = left.size();
    let min_size = res_size.min(a.size());

    for col in 0..cols {
        for j in 0..min_size.saturating_sub(1) {
            {
                let left_u64: &mut [u64] = cast_slice_mut(left.at_mut(col, j));
                NTT126Ifma::ntt126_ifma_from_znx64(left_u64, a.at(col, j));
                <NTT126Ifma as Ntt126IfmaDFTExecute<Ntt126IfmaTable<Primes42>>>::ntt126_ifma_dft_execute(table, left_u64);
            }
            let left_u64: &[u64] = cast_slice(left.at(col, j));
            let right_u32: &mut [u32] = cast_slice_mut(right.at_mut(col, j));
            NTT126Ifma::ntt126_ifma_c_from_b(n, right_u32, left_u64);
        }
        if min_size > 0 {
            let last = min_size - 1;
            {
                let left_u64: &mut [u64] = cast_slice_mut(left.at_mut(col, last));
                NTT126Ifma::ntt126_ifma_from_znx64_masked(left_u64, a.at(col, last), mask);
                <NTT126Ifma as Ntt126IfmaDFTExecute<Ntt126IfmaTable<Primes42>>>::ntt126_ifma_dft_execute(table, left_u64);
            }
            let left_u64: &[u64] = cast_slice(left.at(col, last));
            let right_u32: &mut [u32] = cast_slice_mut(right.at_mut(col, last));
            NTT126Ifma::ntt126_ifma_c_from_b(n, right_u32, left_u64);
        }
        for j in min_size..res_size {
            cast_slice_mut::<_, u64>(left.at_mut(col, j)).fill(0);
            cast_slice_mut::<_, u64>(right.at_mut(col, j)).fill(0);
        }
    }
}

pub(crate) fn cnv_by_const_apply_tmp_bytes(_res_size: usize, _a_size: usize, _b_size: usize) -> usize {
    0
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn cnv_by_const_apply(
    cnv_offset: usize,
    res: &mut VecZnxBigBackendMut<'_, NTT126Ifma>,
    res_col: usize,
    a: &VecZnxBackendRef<'_, NTT126Ifma>,
    a_col: usize,
    b: &VecZnxBackendRef<'_, NTT126Ifma>,
    b_col: usize,
    b_coeff: usize,
    _tmp: &mut [u8],
) {
    let res_size = res.size();
    let a_size = a.size();
    let b_size = b.size();
    if res_size == 0 || a_size == 0 || b_size == 0 {
        for j in 0..res_size {
            res.at_mut(res_col, j).fill(0i128);
        }
        return;
    }

    let bound = a_size + b_size - 1;
    let min_size = res_size.min(bound);
    let offset = cnv_offset.min(bound);

    if b_size == 1 {
        let b0 = b.at(b_col, 0)[b_coeff] as i128;
        for k in 0..min_size {
            let k_abs = k + offset;
            let res_limb: &mut [i128] = res.at_mut(res_col, k);
            if k_abs < a_size {
                let a_limb = a.at(a_col, k_abs);
                for n_i in 0..res_limb.len() {
                    res_limb[n_i] = (a_limb[n_i] as i128) * b0;
                }
            } else {
                res_limb.fill(0i128);
            }
        }

        for j in min_size..res_size {
            res.at_mut(res_col, j).fill(0i128);
        }
        return;
    }

    for k in 0..min_size {
        let k_abs = k + offset;
        let j_min = k_abs.saturating_sub(a_size - 1);
        let j_max = (k_abs + 1).min(b_size);
        let res_limb: &mut [i128] = res.at_mut(res_col, k);
        for (n_i, r) in res_limb.iter_mut().enumerate() {
            let mut acc: i128 = 0;
            for j in j_min..j_max {
                let b_j = b.at(b_col, j)[b_coeff];
                acc += a.at(a_col, k_abs - j)[n_i] as i128 * b_j as i128;
            }
            *r = acc;
        }
    }

    for j in min_size..res_size {
        res.at_mut(res_col, j).fill(0i128);
    }
}
