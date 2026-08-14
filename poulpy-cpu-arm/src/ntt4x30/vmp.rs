//! Vector-matrix product NEON kernels for [`NTT4x30Neon`](crate::NTT4x30Neon).
//!
//! Uses a 4-plane prime-major prepared-matrix layout so the apply path streams
//! one prime plane at a time and reuses extracted input rows across output columns.

use std::mem::size_of;

use bytemuck::{cast_slice, cast_slice_mut};

use poulpy_cpu_ref::reference::ntt4x30::{
    NttCFromB, NttDFTExecute, NttFromZnx64, mat_vec::BbcMeta, primes::Primes30, types::Q_SHIFTED, vec_znx_dft::NttModuleHandle,
};
use poulpy_hal::layouts::{
    DataViewMut, MatZnxBackendRef, Module, VecZnxDftBackendMut, VecZnxDftBackendRef, VmpPMatBackendMut, VmpPMatBackendRef,
    VmpTMatBackendMut, VmpTMatBackendRef, ZnxView, ZnxViewMut,
};

use super::super::neon::ntt4x30_mat_vec::vec_mat1col_product_blkpair_bbc_pm_neon;
use crate::NTT4x30Neon;

/// Scratch space (in bytes) required by the NEON VMP prepare kernel.
pub(crate) fn vmp_prepare_pmat_tmp_bytes_neon(n: usize) -> usize {
    8 * n * size_of::<u64>()
}

/// NEON-local VMP prepare into a 4-plane prime-major layout.
/// The prepared matrix uses one plane per CRT prime. Within each plane the
/// layout is `block_pair -> output_column -> input_row`, and every row stores
/// four u64 values in lane order `[blk0.c0, blk0.c1, blk1.c0, blk1.c1]`.
///
/// Shared body of the two prepare kernels: this backend builds both tiers
/// identically, so they differ only in their destination container, passed
/// here as its raw buffer `out`.
fn prepare_inner(
    module: &Module<NTT4x30Neon>,
    n: usize,
    out: &mut [u64],
    a: &MatZnxBackendRef<'_, NTT4x30Neon>,
    tmp: &mut [u64],
) {
    debug_assert!(std::mem::size_of_val(tmp) >= vmp_prepare_pmat_tmp_bytes_neon(n));
    debug_assert!(n.is_multiple_of(4));

    let nrows = a.cols_in() * a.rows();
    let ncols = a.cols_out() * a.size();
    let n_block_pairs = n / 4;
    let plane_stride = n_block_pairs * ncols * nrows * 4;
    let bp_stride = ncols * nrows * 4;
    let col_stride = nrows * 4;

    let (tmp_b, tmp_c_u64) = tmp.split_at_mut(4 * n);
    let tmp_c: &mut [u32] = cast_slice_mut(tmp_c_u64);

    let mat_i64: &[i64] = a.raw();

    for row_i in 0..nrows {
        for col_i in 0..ncols {
            let pos = n * (row_i * ncols + col_i);

            NTT4x30Neon::ntt_from_znx64(tmp_b, &mat_i64[pos..pos + n]);
            NTT4x30Neon::ntt_dft_execute(module.get_ntt_table(), tmp_b);
            NTT4x30Neon::ntt_c_from_b(n, tmp_c, tmp_b);
            let tmp_c_u64: &[u64] = cast_slice(tmp_c);

            for bp in 0..n_block_pairs {
                let coeff_base = 16 * bp;
                for p in 0..4usize {
                    let at = p * plane_stride + bp * bp_stride + col_i * col_stride + row_i * 4;
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
pub(crate) fn vmp_prepare_pmat_neon_pm(
    module: &Module<NTT4x30Neon>,
    res: &mut VmpPMatBackendMut<'_, NTT4x30Neon>,
    a: &MatZnxBackendRef<'_, NTT4x30Neon>,
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
pub(crate) fn vmp_prepare_tmat_neon_pm(
    module: &Module<NTT4x30Neon>,
    res: &mut VmpTMatBackendMut<'_, NTT4x30Neon>,
    a: &MatZnxBackendRef<'_, NTT4x30Neon>,
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

/// Scratch space (in bytes) required by the NEON VMP apply kernels.
pub(crate) fn vmp_apply_tmp_bytes_neon(a_size: usize, b_rows: usize, b_cols_in: usize) -> usize {
    let row_max = a_size.min(b_rows) * b_cols_in;
    (16 + 16 * row_max) * size_of::<u64>()
}

/// Extract one q120b block pair into 4 prime-major planes.
/// Each plane stores `row_max` rows of 4 u64 with lane order
/// `[blk0.c0, blk0.c1, blk1.c0, blk1.c1]`. The packing is scalar — each output
/// u64 comes from a non-contiguous source offset, so SIMD gathers would not
/// help.
#[inline]
fn extract_blk_pair_prime_major_neon(n: usize, row_max: usize, blk_pair: usize, src: &[u64], dst: &mut [u64]) {
    debug_assert!(n.is_multiple_of(4));
    debug_assert!(src.len() >= row_max * 4 * n);
    debug_assert!(dst.len() >= 16 * row_max);

    let plane_stride = 4 * row_max;
    let coeff_base = 16 * blk_pair;

    for row in 0..row_max {
        let row_base = row * 4 * n + coeff_base;
        for p in 0..4usize {
            let off = p * plane_stride + row * 4;
            dst[off] = src[row_base + p];
            dst[off + 1] = src[row_base + 4 + p];
            dst[off + 2] = src[row_base + 8 + p];
            dst[off + 3] = src[row_base + 12 + p];
        }
    }
}

/// Non-temporal store of one x2-block (8 u64) into a q120b vector via two
/// `stnp q, q` pairs. The result is write-once before being read back by the
/// caller's normalization, so NT stores avoid polluting L1/L2 with output lines.
#[inline]
fn save_blk_overwrite(_n: usize, blk: usize, dst: &mut [u64], src: &[u64]) {
    debug_assert!(src.len() >= 8);
    let off = 8 * blk;
    #[cfg(target_arch = "aarch64")]
    unsafe {
        let dst_ptr = dst.as_mut_ptr().add(off);
        let src_ptr = src.as_ptr();
        core::arch::asm!(
            "ldp  {v0:q}, {v1:q}, [{src}]",
            "ldp  {v2:q}, {v3:q}, [{src}, #32]",
            "stnp {v0:q}, {v1:q}, [{dst}]",
            "stnp {v2:q}, {v3:q}, [{dst}, #32]",
            src = in(reg) src_ptr,
            dst = in(reg) dst_ptr,
            v0 = out(vreg) _,
            v1 = out(vreg) _,
            v2 = out(vreg) _,
            v3 = out(vreg) _,
            options(nostack, preserves_flags),
        );
    }
    #[cfg(not(target_arch = "aarch64"))]
    {
        dst[off..off + 8].copy_from_slice(&src[..8]);
    }
}

#[inline(always)]
fn save_blk_add(n: usize, blk: usize, dst: &mut [u64], src: &[u64]) {
    debug_assert!(src.len() >= 8);
    debug_assert!(dst.len() >= 4 * n);
    for i in 0..8 {
        let k = i % 4;
        dst[8 * blk + i] = dst[8 * blk + i] % Q_SHIFTED[k] + src[i] % Q_SHIFTED[k];
    }
}

#[allow(clippy::too_many_arguments)]
fn vmp_apply_core_neon_pm<const OVERWRITE: bool>(
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
    let plane_stride = n_block_pairs * ncols * nrows * 4;
    let bp_stride = ncols * nrows * 4;
    let col_stride = nrows * 4;

    for bp in 0..n_block_pairs {
        extract_blk_pair_prime_major_neon(n, row_max, bp, a_u64, x_pm);

        for col_pmat in limb_offset..col_max {
            let col_res = col_pmat - limb_offset;
            let y_off = bp * bp_stride + col_pmat * col_stride;

            unsafe {
                vec_mat1col_product_blkpair_bbc_pm_neon(meta, row_max, blkpair_output, x_pm, &pmat_u64[y_off..], plane_stride)
            };

            let blk0 = 2 * bp;
            let blk1 = blk0 + 1;
            let base = col_res * 4 * n;
            if OVERWRITE {
                save_blk_overwrite(n, blk0, &mut res_u64[base..], &blkpair_output[0..8]);
                save_blk_overwrite(n, blk1, &mut res_u64[base..], &blkpair_output[8..16]);
            } else {
                save_blk_add(n, blk0, &mut res_u64[base..], &blkpair_output[0..8]);
                save_blk_add(n, blk1, &mut res_u64[base..], &blkpair_output[8..16]);
            }
        }
    }

    if OVERWRITE {
        let active_cols = col_max - limb_offset;
        for col in active_cols..res_size {
            res_u64[col * 4 * n..(col + 1) * 4 * n].fill(0);
        }
    }
}

/// Shared body of the four `*_dft_to_dft{,_accumulate}` kernels.
///
/// The matrix arrives as its raw buffer plus shape, so both tiers and both
/// accumulation modes reach the same core; `OVERWRITE` selects whether `res` is
/// written or accumulated into.
#[allow(clippy::too_many_arguments)]
fn dft_to_dft_inner<const OVERWRITE: bool>(
    module: &Module<NTT4x30Neon>,
    res: &mut VecZnxDftBackendMut<'_, NTT4x30Neon>,
    a: &VecZnxDftBackendRef<'_, NTT4x30Neon>,
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

    vmp_apply_core_neon_pm::<OVERWRITE>(n, res_u64, a_u64, mat, limb_offset * cols_out, nrows, ncols, meta, tmp);
}

/// `res = pmat * a`, with the matrix cold-prepared.
#[allow(clippy::too_many_arguments)]
pub(crate) fn vmp_apply_pmat_dft_to_dft_neon(
    module: &Module<NTT4x30Neon>,
    res: &mut VecZnxDftBackendMut<'_, NTT4x30Neon>,
    a: &VecZnxDftBackendRef<'_, NTT4x30Neon>,
    pmat: &VmpPMatBackendRef<'_, NTT4x30Neon>,
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
pub(crate) fn vmp_apply_tmat_dft_to_dft_neon(
    module: &Module<NTT4x30Neon>,
    res: &mut VecZnxDftBackendMut<'_, NTT4x30Neon>,
    a: &VecZnxDftBackendRef<'_, NTT4x30Neon>,
    tmat: &VmpTMatBackendRef<'_, NTT4x30Neon>,
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
pub(crate) fn vmp_apply_pmat_dft_to_dft_accumulate_neon(
    module: &Module<NTT4x30Neon>,
    res: &mut VecZnxDftBackendMut<'_, NTT4x30Neon>,
    a: &VecZnxDftBackendRef<'_, NTT4x30Neon>,
    pmat: &VmpPMatBackendRef<'_, NTT4x30Neon>,
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
pub(crate) fn vmp_apply_tmat_dft_to_dft_accumulate_neon(
    module: &Module<NTT4x30Neon>,
    res: &mut VecZnxDftBackendMut<'_, NTT4x30Neon>,
    a: &VecZnxDftBackendRef<'_, NTT4x30Neon>,
    tmat: &VmpTMatBackendRef<'_, NTT4x30Neon>,
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
