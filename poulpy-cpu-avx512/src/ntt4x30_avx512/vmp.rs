//! Vector-matrix product AVX-512F kernels for [`NTT4x30Avx512`](crate::NTT4x30Avx512).
//!
//! Uses a backend-local prime-major prepared-matrix layout so the hot VMP
//! path streams one prime plane at a time and reuses extracted input rows
//! across the output-column loop.

use std::mem::size_of;

use bytemuck::{cast_slice, cast_slice_mut};
use core::arch::x86_64::{
    __m256i, __m512i, _mm_sfence, _mm256_loadu_si256, _mm256_storeu_si256, _mm256_stream_si256, _mm512_add_epi64,
    _mm512_castsi512_si256, _mm512_cvtepi64_epi32, _mm512_cvtepu32_epi64, _mm512_extracti64x4_epi64, _mm512_loadu_si512,
    _mm512_permutex2var_epi64, _mm512_set_epi64,
};

use poulpy_core::oep::gglwe_product_digit_output_size;
use poulpy_cpu_ref::reference::ntt4x30::{
    NttDFTExecute, NttFromZnx64, mat_vec::BbcMeta, primes::Primes30, vec_znx_dft::NttModuleHandle,
};
use poulpy_hal::execution::TaskExecutor;
use poulpy_hal::layouts::{
    DataView, DataViewMut, MatZnxBackendRef, Module, VecZnxDftBackendMut, VecZnxDftBackendRef, VmpPMatBackendMut,
    VmpPMatBackendRef, ZnxView, ZnxViewMut,
};

use super::arithmetic_avx512::{BARRETT_MU, POW32, Q_VEC, bcast_quad, reduce_b_to_canonical_512};
use super::mat_vec_avx512::{vec_mat1col_product_blkpair_bbc_pm_avx512, vec_mat1col_product_blkpair_bbc_pm_x2_avx512};
use super::vec_znx_dft::canonicalize_limb_q120;
use crate::NTT4x30Avx512;

#[derive(Clone, Copy)]
struct SendU32Ptr(*mut u32);

// SAFETY: tasks write distinct block pairs and join before the output is reused.
unsafe impl Send for SendU32Ptr {}
unsafe impl Sync for SendU32Ptr {}

impl SendU32Ptr {
    #[inline(always)]
    fn get(&self) -> *mut u32 {
        self.0
    }
}

/// Scratch space (in bytes) required by the AVX VMP prepare kernel.
pub(crate) fn vmp_prepare_tmp_bytes_avx(n: usize) -> usize {
    4 * n * size_of::<u64>()
}

/// AVX-local VMP prepare into packed prime-pair planes.
///
/// Every u64 stores two canonical u32 residues. The two planes hold prime
/// pairs `(p0, p1)` and `(p2, p3)` for
/// `[blk0.c0, blk0.c1, blk1.c0, blk1.c1]`.
pub(crate) fn vmp_prepare_avx_pm(
    module: &Module<NTT4x30Avx512>,
    res: &mut VmpPMatBackendMut<'_, NTT4x30Avx512>,
    a: &MatZnxBackendRef<'_, NTT4x30Avx512>,
    tmp: &mut [u64],
) {
    let n = res.n();

    debug_assert_eq!(a.n(), n);
    debug_assert_eq!(res.cols_in(), a.cols_in());
    debug_assert_eq!(res.rows(), a.rows());
    debug_assert_eq!(res.cols_out(), a.cols_out());
    debug_assert_eq!(res.size(), a.size());
    debug_assert!(std::mem::size_of_val(tmp) >= vmp_prepare_tmp_bytes_avx(n));
    debug_assert!(n.is_multiple_of(4));

    let nrows = a.cols_in() * a.rows();
    let ncols = a.cols_out() * a.size();
    let n_block_pairs = n / 4;
    let pair_stride = nrows * 4;
    let col_stride = nrows * 8;
    let bp_stride = ncols * nrows * 8;

    let tmp_b = &mut tmp[..4 * n];

    let mat_i64: &[i64] = a.raw();
    let pmat_u64: &mut [u64] = cast_slice_mut(res.data_mut());

    for row_i in 0..nrows {
        for col_i in 0..ncols {
            let pos = n * (row_i * ncols + col_i);

            NTT4x30Avx512::ntt_from_znx64(tmp_b, &mat_i64[pos..pos + n]);
            NTT4x30Avx512::ntt_dft_execute(module.get_ntt_table(), tmp_b);
            unsafe { canonicalize_limb_q120(n, tmp_b) };

            for bp in 0..n_block_pairs {
                let coeff_base = 16 * bp;
                for pair in 0..2usize {
                    let p = 2 * pair;
                    let dst = bp * bp_stride + col_i * col_stride + pair * pair_stride + row_i * 4;
                    for coeff in 0..4 {
                        let coeff_off = coeff_base + 4 * coeff;
                        let r0 = tmp_b[coeff_off + p];
                        let r1 = tmp_b[coeff_off + p + 1];
                        pmat_u64[dst + coeff] = r0 | (r1 << 32);
                    }
                }
            }
        }
    }
}

/// Scratch space (in bytes) required by the AVX VMP apply kernels.
pub(crate) fn vmp_apply_tmp_bytes_avx(a_size: usize, b_rows: usize, b_cols_in: usize) -> usize {
    let row_max = a_size.min(b_rows) * b_cols_in;
    (16 + 16 * row_max) * size_of::<u64>()
}

/// Extract one q120b x2-block from a contiguous q120b matrix.
///
/// Copies one 64-byte block per row using two 256-bit loads and stores.
#[target_feature(enable = "avx512f")]
pub(crate) unsafe fn extract_1blk_from_contiguous_q120b_avx512(
    n: usize,
    row_max: usize,
    blk: usize,
    dst: &mut [u64],
    src: &[u64],
) {
    debug_assert!(n >= 2);
    debug_assert!(n.is_power_of_two());
    debug_assert!(blk < n / 2);
    debug_assert!(src.len() >= row_max * 4 * n);
    debug_assert!(dst.len() >= row_max * 8);

    let src_row_stride = 4 * n;
    let src_blk_off = 8 * blk;

    // Each row copies 8 u64 = one __m512i.
    for row in 0..row_max {
        let src_ptr = unsafe { src.as_ptr().add(row * src_row_stride + src_blk_off) as *const __m512i };
        let dst_ptr = unsafe { dst.as_mut_ptr().add(8 * row) as *mut __m512i };
        unsafe {
            core::arch::x86_64::_mm512_storeu_si512(dst_ptr, _mm512_loadu_si512(src_ptr));
        }
    }
}

/// Extract one q120b block pair into 4 prime-major planes.
///
/// Each plane stores `row_max` rows of 4 u64 with lane order
/// `[blk0.c0, blk0.c1, blk1.c0, blk1.c1]`.
#[target_feature(enable = "avx512f")]
unsafe fn extract_blk_pair_prime_major_avx512(n: usize, row_max: usize, blk_pair: usize, src: &[u32], dst: &mut [u64]) {
    debug_assert!(n.is_multiple_of(4));
    debug_assert!(src.len() >= row_max * 4 * n);
    debug_assert!(dst.len() >= 16 * row_max);

    let plane_stride = 4 * row_max;
    let coeff_base = 16 * blk_pair;

    // Per row, the source contains 4 q120b coefficients (16 u64) laid out as
    // [c0_p0..c0_p3, c1_p0..c1_p3, c2_p0..c2_p3, c3_p0..c3_p3]. We transpose to
    // 4 prime planes of 4 u64 each `[c0_p, c1_p, c2_p, c3_p]`. Each plane is gathered
    // by one `_mm512_permutex2var_epi64` from the two halves of the row, then stored
    // as a 256-bit lane.
    let idx_p0 = _mm512_set_epi64(0, 0, 0, 0, 12, 8, 4, 0);
    let idx_p1 = _mm512_set_epi64(0, 0, 0, 0, 13, 9, 5, 1);
    let idx_p2 = _mm512_set_epi64(0, 0, 0, 0, 14, 10, 6, 2);
    let idx_p3 = _mm512_set_epi64(0, 0, 0, 0, 15, 11, 7, 3);
    for row in 0..row_max {
        let row_base = row * 4 * n + coeff_base;
        let packed = unsafe { _mm512_loadu_si512(src.as_ptr().add(row_base) as *const __m512i) };
        let v0 = _mm512_cvtepu32_epi64(_mm512_extracti64x4_epi64::<0>(packed));
        let v1 = _mm512_cvtepu32_epi64(_mm512_extracti64x4_epi64::<1>(packed));
        for (p, idx) in [idx_p0, idx_p1, idx_p2, idx_p3].into_iter().enumerate() {
            let dst_ptr = unsafe { dst.as_mut_ptr().add(p * plane_stride + row * 4) as *mut __m256i };
            let plane = _mm512_permutex2var_epi64(v0, idx, v1);
            unsafe { _mm256_storeu_si256(dst_ptr, _mm512_castsi512_si256(plane)) };
        }
    }
}

/// Strided-digit variant of [`extract_blk_pair_prime_major_avx512`].
#[allow(clippy::too_many_arguments)]
#[target_feature(enable = "avx512f")]
unsafe fn extract_blk_pair_prime_major_strided_avx512(
    n: usize,
    row_max: usize,
    blk_pair: usize,
    src: &[u32],
    cols: usize,
    limb_base: usize,
    limb_step: usize,
    row_start: usize,
    dst: &mut [u64],
) {
    debug_assert!(n.is_multiple_of(4));
    debug_assert!(dst.len() >= 16 * row_max);

    let plane_stride = 4 * row_max;
    let coeff_base = 16 * blk_pair;
    let idx_p0 = _mm512_set_epi64(0, 0, 0, 0, 12, 8, 4, 0);
    let idx_p1 = _mm512_set_epi64(0, 0, 0, 0, 13, 9, 5, 1);
    let idx_p2 = _mm512_set_epi64(0, 0, 0, 0, 14, 10, 6, 2);
    let idx_p3 = _mm512_set_epi64(0, 0, 0, 0, 15, 11, 7, 3);
    for row in 0..row_max {
        let logical_row = row_start + row;
        let col = logical_row % cols;
        let digit = logical_row / cols;
        let flat = (limb_base + digit * limb_step) * cols + col;
        let row_base = flat * 4 * n + coeff_base;
        let packed = unsafe { _mm512_loadu_si512(src.as_ptr().add(row_base) as *const __m512i) };
        let v0 = _mm512_cvtepu32_epi64(_mm512_extracti64x4_epi64::<0>(packed));
        let v1 = _mm512_cvtepu32_epi64(_mm512_extracti64x4_epi64::<1>(packed));
        for (p, idx) in [idx_p0, idx_p1, idx_p2, idx_p3].into_iter().enumerate() {
            let dst_ptr = unsafe { dst.as_mut_ptr().add(p * plane_stride + row * 4) as *mut __m256i };
            let plane = _mm512_permutex2var_epi64(v0, idx, v1);
            unsafe { _mm256_storeu_si256(dst_ptr, _mm512_castsi512_si256(plane)) };
        }
    }
}

/// Non-temporal write of one x2-block (8 u64) into a q120b vector.
///
/// `VecZnxDft` storage is 64-byte aligned, and every x2-block offset is a
/// multiple of 64 bytes, so both 256-bit stream stores land on aligned cache
/// line halves.
#[target_feature(enable = "avx512f")]
unsafe fn save_blk_overwrite_nt(blk: usize, dst: &mut [u32], src: &[u64]) {
    debug_assert!(src.len() >= 8);
    let off = 8 * blk;
    unsafe {
        let q = bcast_quad(Q_VEC.as_ptr());
        let mu = bcast_quad(BARRETT_MU.as_ptr());
        let pow32 = bcast_quad(POW32.as_ptr());
        let value = reduce_b_to_canonical_512(_mm512_loadu_si512(src.as_ptr() as *const __m512i), q, mu, pow32);
        _mm256_stream_si256(dst.as_mut_ptr().add(off) as *mut __m256i, _mm512_cvtepi64_epi32(value));
    }
}

/// Cached overwrite used when a following gadget digit immediately reads the
/// same block for accumulation.
#[target_feature(enable = "avx512f")]
unsafe fn save_blk_overwrite(blk: usize, dst: &mut [u32], src: &[u64]) {
    debug_assert!(src.len() >= 8);
    let off = 8 * blk;
    unsafe {
        let q = bcast_quad(Q_VEC.as_ptr());
        let mu = bcast_quad(BARRETT_MU.as_ptr());
        let pow32 = bcast_quad(POW32.as_ptr());
        let value = reduce_b_to_canonical_512(_mm512_loadu_si512(src.as_ptr() as *const __m512i), q, mu, pow32);
        _mm256_storeu_si256(dst.as_mut_ptr().add(off) as *mut __m256i, _mm512_cvtepi64_epi32(value));
    }
}

/// Lazy accumulate of one x2-block (8 u64) into a q120b vector.
///
/// Both `dst[8*blk..]` and `src` hold q120b residues in `[0, 2·Q_SHIFTED[k])`
/// (the `accum_to_q120b` invariant). For such inputs `x % Q_SHIFTED[k]` equals a
/// single conditional subtract, so `lazy_reduce_512` (one SIMD compare + masked
/// subtract) reproduces the scalar `%` byte-for-byte. The summed result stays in
/// `[0, 2·Q_SHIFTED[k])`, matching the downstream iNTT/normalize invariant.
#[target_feature(enable = "avx512f")]
unsafe fn save_blk_add(blk: usize, dst: &mut [u32], src: &[u64]) {
    debug_assert!(src.len() >= 8);
    unsafe {
        let q = bcast_quad(Q_VEC.as_ptr());
        let mu = bcast_quad(BARRETT_MU.as_ptr());
        let pow32 = bcast_quad(POW32.as_ptr());
        let dst_ptr = dst.as_mut_ptr().add(8 * blk) as *mut __m256i;
        let dv = _mm512_cvtepu32_epi64(_mm256_loadu_si256(dst_ptr));
        let sv = reduce_b_to_canonical_512(_mm512_loadu_si512(src.as_ptr() as *const __m512i), q, mu, pow32);
        let sum = super::arithmetic_avx512::cond_sub_512(_mm512_add_epi64(dv, sv), q);
        _mm256_storeu_si256(dst_ptr, _mm512_cvtepi64_epi32(sum));
    }
}

#[target_feature(enable = "avx512f")]
#[inline]
unsafe fn save_blkpair_digit(_n: usize, bp: usize, dst_base: *mut u32, src: &[u64], overwrite: bool) {
    unsafe {
        let dst0 = std::slice::from_raw_parts_mut(dst_base.add(16 * bp), 8);
        let dst1 = std::slice::from_raw_parts_mut(dst_base.add(16 * bp + 8), 8);
        if overwrite {
            save_blk_overwrite(0, dst0, &src[0..8]);
            save_blk_overwrite(0, dst1, &src[8..16]);
        } else {
            save_blk_add(0, dst0, &src[0..8]);
            save_blk_add(0, dst1, &src[8..16]);
        }
    }
}

#[allow(clippy::too_many_arguments)]
#[target_feature(enable = "avx512f")]
unsafe fn vmp_apply_core_avx_pm<const OVERWRITE: bool, E: TaskExecutor>(
    n: usize,
    res_u32: &mut [u32],
    a_u32: &[u32],
    pmat_u64: &[u64],
    limb_offset: usize,
    nrows: usize,
    ncols: usize,
    meta: &BbcMeta<Primes30>,
    tmp: &mut [u64],
) {
    debug_assert!(n >= 4);
    debug_assert!(n.is_power_of_two());
    debug_assert!(n.is_multiple_of(4));

    let a_size = a_u32.len() / (4 * n);
    let res_size = res_u32.len() / (4 * n);
    let n_block_pairs = n / 4;

    let row_end = nrows.min(a_size);
    let row_start = a_u32
        .chunks_exact(4 * n)
        .take(row_end)
        .take_while(|row| row.iter().all(|&x| x == 0))
        .count();
    let row_max = row_end - row_start;
    let col_max = ncols.min(res_size + limb_offset);

    if limb_offset >= col_max || row_max == 0 {
        if OVERWRITE {
            res_u32.fill(0);
        }
        return;
    }

    let pair_stride = nrows * 4;
    let col_stride = nrows * 8;
    let bp_stride = ncols * nrows * 8;
    let a_u32 = &a_u32[row_start * 4 * n..];

    if !E::is_parallel() || n_block_pairs < 2 {
        let (blkpair_output, x_pm) = tmp.split_at_mut(16);
        let x_pm = &mut x_pm[..16 * row_max];
        for bp in 0..n_block_pairs {
            unsafe { extract_blk_pair_prime_major_avx512(n, row_max, bp, a_u32, x_pm) };

            for col_pmat in limb_offset..col_max {
                let col_res = col_pmat - limb_offset;
                let y_off = bp * bp_stride + col_pmat * col_stride + row_start * 4;
                unsafe {
                    vec_mat1col_product_blkpair_bbc_pm_avx512(
                        meta,
                        row_max,
                        blkpair_output,
                        x_pm,
                        &pmat_u64[y_off..],
                        pair_stride,
                    )
                };

                let blk0 = 2 * bp;
                let blk1 = blk0 + 1;
                let base = col_res * 4 * n;
                if OVERWRITE {
                    unsafe { save_blk_overwrite_nt(blk0, &mut res_u32[base..], &blkpair_output[0..8]) };
                    unsafe { save_blk_overwrite_nt(blk1, &mut res_u32[base..], &blkpair_output[8..16]) };
                } else {
                    unsafe { save_blk_add(blk0, &mut res_u32[base..], &blkpair_output[0..8]) };
                    unsafe { save_blk_add(blk1, &mut res_u32[base..], &blkpair_output[8..16]) };
                }
            }
        }
    } else {
        let res_ptr = SendU32Ptr(res_u32.as_mut_ptr());
        E::for_each_chunked(n_block_pairs, tmp, 16 + 16 * row_max, |task_tmp, bp| {
            let (blkpair_output, x_pm) = task_tmp.split_at_mut(16);
            unsafe { extract_blk_pair_prime_major_avx512(n, row_max, bp, a_u32, x_pm) };

            for col_pmat in limb_offset..col_max {
                let col_res = col_pmat - limb_offset;
                let y_off = bp * bp_stride + col_pmat * col_stride + row_start * 4;
                unsafe {
                    vec_mat1col_product_blkpair_bbc_pm_avx512(
                        meta,
                        row_max,
                        blkpair_output,
                        x_pm,
                        &pmat_u64[y_off..],
                        pair_stride,
                    );
                    let base = col_res * 4 * n;
                    let dst0 = std::slice::from_raw_parts_mut(res_ptr.get().add(base + 16 * bp), 8);
                    let dst1 = std::slice::from_raw_parts_mut(res_ptr.get().add(base + 16 * bp + 8), 8);
                    if OVERWRITE {
                        save_blk_overwrite(0, dst0, &blkpair_output[0..8]);
                        save_blk_overwrite(0, dst1, &blkpair_output[8..16]);
                    } else {
                        save_blk_add(0, dst0, &blkpair_output[0..8]);
                        save_blk_add(0, dst1, &blkpair_output[8..16]);
                    }
                }
            }
        });
    }

    if OVERWRITE {
        let active_cols = col_max - limb_offset;
        for col in active_cols..res_size {
            res_u32[col * 4 * n..(col + 1) * 4 * n].fill(0);
        }
        if !E::is_parallel() {
            _mm_sfence();
        }
    }
}

pub(crate) fn vmp_apply_dft_to_dft_avx<E: TaskExecutor>(
    module: &Module<NTT4x30Avx512>,
    res: &mut VecZnxDftBackendMut<'_, NTT4x30Avx512>,
    a: &VecZnxDftBackendRef<'_, NTT4x30Avx512>,
    pmat: &VmpPMatBackendRef<'_, NTT4x30Avx512>,
    limb_offset: usize,
    tmp: &mut [u64],
) {
    let n = res.n();
    let nrows = pmat.cols_in() * pmat.rows();
    let ncols = pmat.cols_out() * pmat.size();
    let meta = module.get_bbc_meta();

    let res_u32: &mut [u32] = cast_slice_mut(res.raw_mut());
    let a_u32: &[u32] = cast_slice(a.raw());
    let pmat_u64: &[u64] = cast_slice(pmat.data());

    unsafe {
        vmp_apply_core_avx_pm::<true, E>(
            n,
            res_u32,
            a_u32,
            pmat_u64,
            limb_offset * pmat.cols_out(),
            nrows,
            ncols,
            meta,
            tmp,
        );
    }
}

pub(crate) fn vmp_apply_dft_to_dft_accumulate_avx<E: TaskExecutor>(
    module: &Module<NTT4x30Avx512>,
    res: &mut VecZnxDftBackendMut<'_, NTT4x30Avx512>,
    a: &VecZnxDftBackendRef<'_, NTT4x30Avx512>,
    pmat: &VmpPMatBackendRef<'_, NTT4x30Avx512>,
    limb_offset: usize,
    tmp: &mut [u64],
) {
    let n = res.n();
    let nrows = pmat.cols_in() * pmat.rows();
    let ncols = pmat.cols_out() * pmat.size();
    let meta = module.get_bbc_meta();

    let res_u32: &mut [u32] = cast_slice_mut(res.raw_mut());
    let a_u32: &[u32] = cast_slice(a.raw());
    let pmat_u64: &[u64] = cast_slice(pmat.data());

    unsafe {
        vmp_apply_core_avx_pm::<false, E>(
            n,
            res_u32,
            a_u32,
            pmat_u64,
            limb_offset * pmat.cols_out(),
            nrows,
            ncols,
            meta,
            tmp,
        );
    }
}

pub(crate) fn vmp_apply_digits_strided_tmp_bytes_avx(
    a_cols: usize,
    a_size: usize,
    dsize: usize,
    b_rows: usize,
    b_cols_in: usize,
    workers: usize,
) -> usize {
    let nrows = b_rows * b_cols_in;
    let row_max = (0..dsize)
        .map(|di| (a_cols * ((a_size + di) / dsize).min(b_rows)).min(nrows))
        .max()
        .unwrap_or(0);
    let rhs_count = dsize.clamp(1, 2);
    (4 * dsize + workers * rhs_count * 16 * (row_max + 1)) * size_of::<u64>()
}

/// Applies all gadget digits directly from their interleaved source limbs.
/// The block-pair outer loop keeps the prepared-matrix working set hot and
/// avoids materializing a temporary DFT vector for every digit.
pub(crate) fn vmp_apply_dft_to_dft_digits_strided_avx<E: TaskExecutor>(
    module: &Module<NTT4x30Avx512>,
    res: &mut VecZnxDftBackendMut<'_, NTT4x30Avx512>,
    a: &VecZnxDftBackendRef<'_, NTT4x30Avx512>,
    dsize: usize,
    product_limbs: usize,
    pmat: &VmpPMatBackendRef<'_, NTT4x30Avx512>,
    tmp: &mut [u64],
) {
    vmp_apply_dft_to_dft_digits_strided_avx_inner::<E>(module, res, a, dsize, product_limbs, pmat, None, tmp)
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn vmp_apply_dft_to_dft_digits_strided_avx_known_zero_prefix<E: TaskExecutor>(
    module: &Module<NTT4x30Avx512>,
    res: &mut VecZnxDftBackendMut<'_, NTT4x30Avx512>,
    a: &VecZnxDftBackendRef<'_, NTT4x30Avx512>,
    dsize: usize,
    product_limbs: usize,
    pmat: &VmpPMatBackendRef<'_, NTT4x30Avx512>,
    zero_prefix: usize,
    tmp: &mut [u64],
) {
    assert!(zero_prefix <= a.size());
    vmp_apply_dft_to_dft_digits_strided_avx_inner::<E>(module, res, a, dsize, product_limbs, pmat, Some(zero_prefix), tmp)
}

#[allow(clippy::too_many_arguments)]
fn vmp_apply_dft_to_dft_digits_strided_avx_inner<E: TaskExecutor>(
    module: &Module<NTT4x30Avx512>,
    res: &mut VecZnxDftBackendMut<'_, NTT4x30Avx512>,
    a: &VecZnxDftBackendRef<'_, NTT4x30Avx512>,
    dsize: usize,
    product_limbs: usize,
    pmat: &VmpPMatBackendRef<'_, NTT4x30Avx512>,
    zero_prefix: Option<usize>,
    tmp: &mut [u64],
) {
    let n = res.n();
    let output_size = res.size();
    if dsize == 0 || n < 4 {
        return;
    }

    let n_block_pairs = n / 4;
    let nrows = pmat.cols_in() * pmat.rows();
    let ncols = pmat.cols_out() * pmat.size();
    let cols_out = pmat.cols_out();
    let res_cols = res.cols();
    let a_cols = a.cols();
    let a_size = a.size();
    let dnum = pmat.rows();
    let meta = module.get_bbc_meta();
    let a_u32: &[u32] = cast_slice(a.raw());
    let res_u32: &mut [u32] = cast_slice_mut(res.raw_mut());
    let pmat_u64: &[u64] = cast_slice(pmat.data());

    let pair_stride = nrows * 4;
    let col_stride = nrows * 8;
    let bp_stride = ncols * nrows * 8;

    let (digit_meta, tmp) = tmp.split_at_mut(4 * dsize);
    let (row_maxs, digit_meta) = digit_meta.split_at_mut(dsize);
    let (row_starts, digit_meta) = digit_meta.split_at_mut(dsize);
    let (limb_offsets, col_maxs) = digit_meta.split_at_mut(dsize);
    for di in 0..dsize {
        let digit_limbs = ((a_size + di) / dsize).min(dnum);
        // Match the reference product: full-width overwrite, then narrowed accumulations.
        let active_size = gglwe_product_digit_output_size(output_size, pmat.size(), dsize, di, product_limbs);
        let limb_offset = di * cols_out;
        let row_end = (a_cols * digit_limbs).min(nrows);
        let limb_base = dsize - 1 - di;
        let row_start = zero_prefix.map_or_else(
            || {
                (0..row_end)
                    .take_while(|&row| {
                        let flat = (limb_base + (row / a_cols) * dsize) * a_cols + row % a_cols;
                        a_u32[flat * 4 * n..(flat + 1) * 4 * n].iter().all(|&x| x == 0)
                    })
                    .count()
            },
            |prefix| (a_cols * ((prefix + di) / dsize).min(digit_limbs)).min(row_end),
        );
        row_starts[di] = row_start as u64;
        row_maxs[di] = (row_end - row_start) as u64;
        limb_offsets[di] = limb_offset as u64;
        col_maxs[di] = ncols.min(res_cols * active_size + limb_offset) as u64;
    }

    let res_flat = res_cols * output_size;
    if row_maxs[0] == 0 {
        res_u32.fill(0);
    } else {
        for col in col_maxs[0] as usize..res_flat {
            res_u32[col * 4 * n..(col + 1) * 4 * n].fill(0);
        }
    }

    let res_ptr = SendU32Ptr(res_u32.as_mut_ptr());
    let rhs_count = dsize.clamp(1, 2);
    let x_words = 16 * row_maxs.iter().copied().max().unwrap_or(0) as usize;
    let task_tmp_len = rhs_count * (16 + x_words);
    let process_block_pair = |task_tmp: &mut [u64], bp: usize| {
        let (blkpair_outputs, x_pm) = task_tmp.split_at_mut(16 * rhs_count);
        let (output0, output1) = blkpair_outputs.split_at_mut(16);
        let (x0_pm, x1_pm) = x_pm[..rhs_count * x_words].split_at_mut(x_words);
        let mut di = 0;
        while di < dsize {
            let pair = di + 1 < dsize
                && row_maxs[di] != 0
                && row_starts[di] == row_starts[di + 1]
                && row_maxs[di] == row_maxs[di + 1]
                && limb_offsets[di + 1] < col_maxs[di].min(col_maxs[di + 1]);

            if pair {
                let row_start = row_starts[di] as usize;
                let row_max = row_maxs[di] as usize;
                let x0_pm = &mut x0_pm[..16 * row_max];
                let x1_pm = &mut x1_pm[..16 * row_max];
                unsafe {
                    extract_blk_pair_prime_major_strided_avx512(
                        n,
                        row_max,
                        bp,
                        a_u32,
                        a_cols,
                        dsize - 1 - di,
                        dsize,
                        row_start,
                        x0_pm,
                    );
                    extract_blk_pair_prime_major_strided_avx512(
                        n,
                        row_max,
                        bp,
                        a_u32,
                        a_cols,
                        dsize - 2 - di,
                        dsize,
                        row_start,
                        x1_pm,
                    );
                }

                let limb_offset0 = limb_offsets[di] as usize;
                let limb_offset1 = limb_offsets[di + 1] as usize;
                let col_max0 = col_maxs[di] as usize;
                let col_max1 = col_maxs[di + 1] as usize;
                let prefix_end = limb_offset1.min(col_max0);
                let shared_end = col_max0.min(col_max1);

                for col_pmat in limb_offset0..prefix_end {
                    let y_off = bp * bp_stride + col_pmat * col_stride + row_start * 4;
                    unsafe {
                        vec_mat1col_product_blkpair_bbc_pm_avx512(meta, row_max, output0, x0_pm, &pmat_u64[y_off..], pair_stride);
                        let base = (col_pmat - limb_offset0) * 4 * n;
                        save_blkpair_digit(n, bp, res_ptr.get().add(base), output0, di == 0);
                    }
                }

                for col_pmat in limb_offset1..shared_end {
                    let y_off = bp * bp_stride + col_pmat * col_stride + row_start * 4;
                    unsafe {
                        vec_mat1col_product_blkpair_bbc_pm_x2_avx512(
                            meta,
                            row_max,
                            output0,
                            output1,
                            x0_pm,
                            x1_pm,
                            &pmat_u64[y_off..],
                            pair_stride,
                        );
                        let base0 = (col_pmat - limb_offset0) * 4 * n;
                        let base1 = (col_pmat - limb_offset1) * 4 * n;
                        save_blkpair_digit(n, bp, res_ptr.get().add(base0), output0, di == 0);
                        save_blkpair_digit(n, bp, res_ptr.get().add(base1), output1, false);
                    }
                }

                for col_pmat in shared_end..col_max0 {
                    let y_off = bp * bp_stride + col_pmat * col_stride + row_start * 4;
                    unsafe {
                        vec_mat1col_product_blkpair_bbc_pm_avx512(meta, row_max, output0, x0_pm, &pmat_u64[y_off..], pair_stride);
                        let base = (col_pmat - limb_offset0) * 4 * n;
                        save_blkpair_digit(n, bp, res_ptr.get().add(base), output0, di == 0);
                    }
                }

                for col_pmat in shared_end.max(limb_offset1)..col_max1 {
                    let y_off = bp * bp_stride + col_pmat * col_stride + row_start * 4;
                    unsafe {
                        vec_mat1col_product_blkpair_bbc_pm_avx512(meta, row_max, output1, x1_pm, &pmat_u64[y_off..], pair_stride);
                        let base = (col_pmat - limb_offset1) * 4 * n;
                        save_blkpair_digit(n, bp, res_ptr.get().add(base), output1, false);
                    }
                }

                di += 2;
                continue;
            }
            let limb_offset = limb_offsets[di] as usize;
            let col_max = col_maxs[di] as usize;
            if limb_offset >= col_max {
                di += 1;
                continue;
            }
            let row_max = row_maxs[di] as usize;
            if row_max == 0 {
                di += 1;
                continue;
            }
            let row_start = row_starts[di] as usize;
            let x0_pm = &mut x0_pm[..16 * row_max];
            unsafe {
                extract_blk_pair_prime_major_strided_avx512(
                    n,
                    row_max,
                    bp,
                    a_u32,
                    a_cols,
                    dsize - 1 - di,
                    dsize,
                    row_start,
                    x0_pm,
                );
            }

            for col_pmat in limb_offset..col_max {
                let col_res = col_pmat - limb_offset;
                // `row_start` is an offset within each prime plane.
                let y_off = bp * bp_stride + col_pmat * col_stride + row_start * 4;
                unsafe {
                    vec_mat1col_product_blkpair_bbc_pm_avx512(meta, row_max, output0, x0_pm, &pmat_u64[y_off..], pair_stride);
                    let base = col_res * 4 * n;
                    save_blkpair_digit(n, bp, res_ptr.get().add(base), output0, di == 0);
                }
            }
            di += 1;
        }
    };

    if E::is_parallel() && n_block_pairs > 1 {
        E::for_each_chunked(n_block_pairs, tmp, task_tmp_len, process_block_pair);
    } else {
        for bp in 0..n_block_pairs {
            process_block_pair(tmp, bp);
        }
    }
}

/// Copies rows `first_row + i * row_step` of `a`, truncated to `res.size()`
/// limbs, into rows `i` of `res`, in the prime-major prepared layout.
pub(crate) fn vmp_extract_selected_rows_avx512_pm(
    res: &mut VmpPMatBackendMut<'_, NTT4x30Avx512>,
    a: &VmpPMatBackendRef<'_, NTT4x30Avx512>,
    first_row: usize,
    row_step: usize,
) {
    let n: usize = a.n();

    let cols_in: usize = a.cols_in();
    let (res_rows, res_nrows, res_ncols) = (res.rows(), res.rows() * cols_in, res.cols_out() * res.size());
    let (a_nrows, a_ncols) = (a.rows() * cols_in, a.cols_out() * a.size());
    let n_block_pairs: usize = n / 4;
    // Same strides as the prepare kernel above, per matrix shape.
    let (res_plane, res_bp, res_col) = (res_nrows * 4, res_ncols * res_nrows * 8, res_nrows * 8);
    let (a_plane, a_bp, a_col) = (a_nrows * 4, a_ncols * a_nrows * 8, a_nrows * 8);
    let span: usize = cols_in * 4;

    let src: &[u64] = cast_slice(a.raw());
    let dst: &mut [u64] = cast_slice_mut(res.data_mut());
    // Two planes: each word packs a pair of 32-bit residues.
    for p in 0..2 {
        for bp in 0..n_block_pairs {
            for col in 0..res_ncols {
                let dst_base: usize = p * res_plane + bp * res_bp + col * res_col;
                let src_base: usize = p * a_plane + bp * a_bp + col * a_col;
                for i in 0..res_rows {
                    let (d, s) = (dst_base + i * span, src_base + (first_row + i * row_step) * span);
                    dst[d..d + span].copy_from_slice(&src[s..s + span]);
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::extract_1blk_from_contiguous_q120b_avx512;
    use poulpy_cpu_ref::reference::ntt4x30::mat_vec::extract_1blk_from_contiguous_q120b_ref;

    #[test]
    fn extract_1blk_from_contiguous_q120b_avx2_vs_ref() {
        for &n in &[256usize, 4096, 16384] {
            for &row_max in &[1usize, 3, 7] {
                let src: Vec<u64> = (0..row_max * 4 * n)
                    .map(|i| (0x9e37_79b9_7f4a_7c15u64.wrapping_mul(i as u64 + 1)) ^ ((i as u64) << 17))
                    .collect();

                for &blk in &[0usize, n / 4, n / 2 - 1] {
                    let mut dst_ref = vec![0u64; 8 * row_max];
                    let mut dst_avx = vec![0u64; 8 * row_max];

                    extract_1blk_from_contiguous_q120b_ref(n, row_max, blk, &mut dst_ref, &src);
                    unsafe { extract_1blk_from_contiguous_q120b_avx512(n, row_max, blk, &mut dst_avx, &src) };

                    assert_eq!(dst_avx, dst_ref, "n={n}, row_max={row_max}, blk={blk}");
                }
            }
        }
    }
}
