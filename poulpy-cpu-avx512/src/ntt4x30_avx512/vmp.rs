//! Vector-matrix product AVX-512F kernels for [`NTT4x30Avx512`](crate::NTT4x30Avx512).
//!
//! Uses a backend-local prime-major prepared-matrix layout so the hot VMP
//! path streams one prime plane at a time and reuses extracted input rows
//! across the output-column loop.

use std::mem::size_of;

use bytemuck::{cast_slice, cast_slice_mut};
use core::arch::x86_64::{
    __m256i, __m512i, _mm_sfence, _mm256_loadu_si256, _mm256_storeu_si256, _mm256_stream_si256, _mm512_add_epi64,
    _mm512_castsi512_si256, _mm512_loadu_si512, _mm512_permutex2var_epi64, _mm512_set_epi64, _mm512_set1_epi64,
    _mm512_storeu_si512,
};

use poulpy_cpu_ref::reference::ntt4x30::{
    NttCFromB, NttDFTExecute, NttFromZnx64, mat_vec::BbcMeta, primes::Primes30, vec_znx_dft::NttModuleHandle,
};
use poulpy_hal::layouts::{
    DataViewMut, MatZnxBackendRef, Module, VecZnxDftBackendMut, VecZnxDftBackendRef, VmpPMatBackendMut, VmpPMatBackendRef,
    VmpTMatBackendMut, VmpTMatBackendRef, ZnxView, ZnxViewMut,
};

use super::mat_vec_avx512::vec_mat1col_product_blkpair_bbc_pm_avx512;
use super::prim::{lazy_reduce_512, q_shifted_512};
use crate::NTT4x30Avx512;

/// Scratch space (in bytes) required by the AVX VMP prepare kernel.
pub(crate) fn vmp_prepare_pmat_tmp_bytes_avx(n: usize) -> usize {
    8 * n * size_of::<u64>()
}

/// AVX-local VMP prepare into a prime-major layout interleaved per
/// `(block_pair, output_column)`: `block_pair -> output_column -> prime ->
/// input_row`, every row storing `[blk0.c0, blk0.c1, blk1.c0, blk1.c1]`.
/// Keeping the four prime planes adjacent per chunk makes the apply read the
/// matrix as a single sequential stream.
///
/// Shared body of the two prepare kernels: this backend builds both tiers
/// identically, so they differ only in their destination container, passed
/// here as its raw buffer `out`.
fn prepare_inner(
    module: &Module<NTT4x30Avx512>,
    n: usize,
    out: &mut [u64],
    a: &MatZnxBackendRef<'_, NTT4x30Avx512>,
    tmp: &mut [u64],
) {
    debug_assert!(std::mem::size_of_val(tmp) >= vmp_prepare_pmat_tmp_bytes_avx(n));
    debug_assert!(n.is_multiple_of(4));

    let nrows = a.cols_in() * a.rows();
    let ncols = a.cols_out() * a.size();
    let n_block_pairs = n / 4;
    let plane_stride = nrows * 4;
    let col_stride = nrows * 16;
    let bp_stride = ncols * nrows * 16;

    let (tmp_b, tmp_c_u64) = tmp.split_at_mut(4 * n);
    let tmp_c: &mut [u32] = cast_slice_mut(tmp_c_u64);

    let mat_i64: &[i64] = a.raw();

    for row_i in 0..nrows {
        for col_i in 0..ncols {
            let pos = n * (row_i * ncols + col_i);

            NTT4x30Avx512::ntt_from_znx64(tmp_b, &mat_i64[pos..pos + n]);
            NTT4x30Avx512::ntt_dft_execute(module.get_ntt_table(), tmp_b);
            NTT4x30Avx512::ntt_c_from_b(n, tmp_c, tmp_b);
            let tmp_c_u64: &[u64] = cast_slice(tmp_c);

            for bp in 0..n_block_pairs {
                let coeff_base = 16 * bp;
                for p in 0..4usize {
                    let at = bp * bp_stride + col_i * col_stride + p * plane_stride + row_i * 4;
                    out[at..at + 4].copy_from_slice(&[
                        tmp_c_u64[coeff_base + p],
                        tmp_c_u64[coeff_base + 4 + p],
                        tmp_c_u64[coeff_base + 8 + p],
                        tmp_c_u64[coeff_base + 12 + p],
                    ]);
                }
            }
        }
    }
}

/// Prepares `a` into the packed cold-prep [`VmpPMat`](poulpy_hal::layouts::VmpPMat).
pub(crate) fn vmp_prepare_pmat_avx_pm(
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
    prepare_inner(module, n, cast_slice_mut(res.data_mut()), a, tmp);
}

/// Prepares `a` into the transformed hot-prep [`VmpTMat`](poulpy_hal::layouts::VmpTMat).
pub(crate) fn vmp_prepare_tmat_avx_pm(
    module: &Module<NTT4x30Avx512>,
    res: &mut VmpTMatBackendMut<'_, NTT4x30Avx512>,
    a: &MatZnxBackendRef<'_, NTT4x30Avx512>,
    tmp: &mut [u64],
) {
    let n = res.n();
    debug_assert_eq!(a.n(), n);
    debug_assert_eq!(res.cols_in(), a.cols_in());
    debug_assert_eq!(res.rows(), a.rows());
    debug_assert_eq!(res.cols_out(), a.cols_out());
    debug_assert_eq!(res.size(), a.size());
    prepare_inner(module, n, cast_slice_mut(res.data_mut()), a, tmp);
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
unsafe fn extract_blk_pair_prime_major_avx512(n: usize, row_max: usize, blk_pair: usize, src: &[u64], dst: &mut [u64]) {
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
        let v0: __m512i = unsafe { _mm512_loadu_si512(src.as_ptr().add(row_base) as *const __m512i) };
        let v1: __m512i = unsafe { _mm512_loadu_si512(src.as_ptr().add(row_base + 8) as *const __m512i) };
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
unsafe fn save_blk_overwrite_nt(_n: usize, blk: usize, dst: &mut [u64], src: &[u64]) {
    debug_assert!(src.len() >= 8);
    let off = 8 * blk;
    let dst_ptr = unsafe { dst.as_mut_ptr().add(off) as *mut __m256i };
    let src_ptr = src.as_ptr() as *const __m256i;
    unsafe {
        _mm256_stream_si256(dst_ptr, _mm256_loadu_si256(src_ptr));
        _mm256_stream_si256(dst_ptr.add(1), _mm256_loadu_si256(src_ptr.add(1)));
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
unsafe fn save_blk_add(n: usize, blk: usize, dst: &mut [u64], src: &[u64]) {
    debug_assert!(src.len() >= 8);
    debug_assert!(dst.len() >= 4 * n);
    unsafe {
        let q_s_512 = q_shifted_512();
        let msb_512 = _mm512_set1_epi64(i64::MIN);
        let dst_ptr = dst.as_mut_ptr().add(8 * blk) as *mut __m512i;
        let src_ptr = src.as_ptr() as *const __m512i;
        let dv = lazy_reduce_512(_mm512_loadu_si512(dst_ptr), q_s_512, msb_512);
        let sv = lazy_reduce_512(_mm512_loadu_si512(src_ptr), q_s_512, msb_512);
        _mm512_storeu_si512(dst_ptr, _mm512_add_epi64(dv, sv));
    }
}

#[allow(clippy::too_many_arguments)]
#[target_feature(enable = "avx512f")]
unsafe fn vmp_apply_core_avx_pm<const OVERWRITE: bool>(
    n: usize,
    res_u64: &mut [u64],
    a_u64: &[u64],
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

    let a_size = a_u64.len() / (4 * n);
    let res_size = res_u64.len() / (4 * n);
    let n_block_pairs = n / 4;

    let row_max = nrows.min(a_size);
    let col_max = ncols.min(res_size + limb_offset);

    if limb_offset >= col_max {
        if OVERWRITE {
            res_u64.fill(0);
        }
        return;
    }

    let (blkpair_output, x_pm) = tmp.split_at_mut(16);
    let x_pm = &mut x_pm[..16 * row_max];
    let plane_stride = nrows * 4;
    let col_stride = nrows * 16;
    let bp_stride = ncols * nrows * 16;

    for bp in 0..n_block_pairs {
        unsafe { extract_blk_pair_prime_major_avx512(n, row_max, bp, a_u64, x_pm) };

        for col_pmat in limb_offset..col_max {
            let col_res = col_pmat - limb_offset;
            let y_off = bp * bp_stride + col_pmat * col_stride;

            unsafe {
                vec_mat1col_product_blkpair_bbc_pm_avx512(meta, row_max, blkpair_output, x_pm, &pmat_u64[y_off..], plane_stride)
            };

            let blk0 = 2 * bp;
            let blk1 = blk0 + 1;
            let base = col_res * 4 * n;
            if OVERWRITE {
                unsafe { save_blk_overwrite_nt(n, blk0, &mut res_u64[base..], &blkpair_output[0..8]) };
                unsafe { save_blk_overwrite_nt(n, blk1, &mut res_u64[base..], &blkpair_output[8..16]) };
            } else {
                unsafe { save_blk_add(n, blk0, &mut res_u64[base..], &blkpair_output[0..8]) };
                unsafe { save_blk_add(n, blk1, &mut res_u64[base..], &blkpair_output[8..16]) };
            }
        }
    }

    if OVERWRITE {
        let active_cols = col_max - limb_offset;
        for col in active_cols..res_size {
            res_u64[col * 4 * n..(col + 1) * 4 * n].fill(0);
        }
        _mm_sfence();
    }
}

/// Shared body of the four `*_dft_to_dft{,_accumulate}` kernels.
///
/// The matrix arrives as its raw buffer plus shape, so both tiers and both
/// accumulation modes reach the same core; `OVERWRITE` selects whether `res` is
/// written or accumulated into.
#[allow(clippy::too_many_arguments)]
fn dft_to_dft_inner<const OVERWRITE: bool>(
    module: &Module<NTT4x30Avx512>,
    res: &mut VecZnxDftBackendMut<'_, NTT4x30Avx512>,
    a: &VecZnxDftBackendRef<'_, NTT4x30Avx512>,
    mat: &[u64],
    rows: usize,
    cols_in: usize,
    cols_out: usize,
    size: usize,
    limb_offset: usize,
    tmp: &mut [u64],
) {
    let n = res.n();
    let nrows = cols_in * rows;
    let ncols = cols_out * size;
    let meta = module.get_bbc_meta();

    let res_u64: &mut [u64] = cast_slice_mut(res.raw_mut());
    let a_u64: &[u64] = cast_slice(a.raw());

    unsafe {
        vmp_apply_core_avx_pm::<OVERWRITE>(n, res_u64, a_u64, mat, limb_offset * cols_out, nrows, ncols, meta, tmp);
    }
}

/// `res = pmat * a`, with the matrix cold-prepared.
#[allow(clippy::too_many_arguments)]
pub(crate) fn vmp_apply_pmat_dft_to_dft_avx(
    module: &Module<NTT4x30Avx512>,
    res: &mut VecZnxDftBackendMut<'_, NTT4x30Avx512>,
    a: &VecZnxDftBackendRef<'_, NTT4x30Avx512>,
    pmat: &VmpPMatBackendRef<'_, NTT4x30Avx512>,
    limb_offset: usize,
    tmp: &mut [u64],
) {
    dft_to_dft_inner::<true>(
        module,
        res,
        a,
        cast_slice(pmat.raw()),
        pmat.rows(),
        pmat.cols_in(),
        pmat.cols_out(),
        pmat.size(),
        limb_offset,
        tmp,
    );
}

/// `res = tmat * a`, with the matrix hot-prepared.
#[allow(clippy::too_many_arguments)]
pub(crate) fn vmp_apply_tmat_dft_to_dft_avx(
    module: &Module<NTT4x30Avx512>,
    res: &mut VecZnxDftBackendMut<'_, NTT4x30Avx512>,
    a: &VecZnxDftBackendRef<'_, NTT4x30Avx512>,
    tmat: &VmpTMatBackendRef<'_, NTT4x30Avx512>,
    limb_offset: usize,
    tmp: &mut [u64],
) {
    dft_to_dft_inner::<true>(
        module,
        res,
        a,
        cast_slice(tmat.raw()),
        tmat.rows(),
        tmat.cols_in(),
        tmat.cols_out(),
        tmat.size(),
        limb_offset,
        tmp,
    );
}

/// `res += pmat * a`, with the matrix cold-prepared.
#[allow(clippy::too_many_arguments)]
pub(crate) fn vmp_apply_pmat_dft_to_dft_accumulate_avx(
    module: &Module<NTT4x30Avx512>,
    res: &mut VecZnxDftBackendMut<'_, NTT4x30Avx512>,
    a: &VecZnxDftBackendRef<'_, NTT4x30Avx512>,
    pmat: &VmpPMatBackendRef<'_, NTT4x30Avx512>,
    limb_offset: usize,
    tmp: &mut [u64],
) {
    dft_to_dft_inner::<false>(
        module,
        res,
        a,
        cast_slice(pmat.raw()),
        pmat.rows(),
        pmat.cols_in(),
        pmat.cols_out(),
        pmat.size(),
        limb_offset,
        tmp,
    );
}

/// `res += tmat * a`, with the matrix hot-prepared.
#[allow(clippy::too_many_arguments)]
pub(crate) fn vmp_apply_tmat_dft_to_dft_accumulate_avx(
    module: &Module<NTT4x30Avx512>,
    res: &mut VecZnxDftBackendMut<'_, NTT4x30Avx512>,
    a: &VecZnxDftBackendRef<'_, NTT4x30Avx512>,
    tmat: &VmpTMatBackendRef<'_, NTT4x30Avx512>,
    limb_offset: usize,
    tmp: &mut [u64],
) {
    dft_to_dft_inner::<false>(
        module,
        res,
        a,
        cast_slice(tmat.raw()),
        tmat.rows(),
        tmat.cols_in(),
        tmat.cols_out(),
        tmat.size(),
        limb_offset,
        tmp,
    );
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
