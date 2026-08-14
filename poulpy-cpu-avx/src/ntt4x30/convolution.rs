//! Fused convolution accumulation AVX2 kernel for [`NTT4x30Avx`](crate::NTT4x30Avx).
//!
//! Computes `res[res_col] = Σ_t a_t ⊛ b_t` in one pass: the lazy q120
//! accumulators stay live in registers across all terms of one output limb,
//! the bbc reduction runs once per output, and the destination column is
//! written exactly once through the staged group flush. The left rows hold the
//! canonical encoding (odd lanes zero), so the `x_hi * y_hi` product path is
//! identically zero and skipped.

use core::arch::x86_64::{
    __m256i, _mm256_add_epi64, _mm256_and_si256, _mm256_loadu_si256, _mm256_mul_epu32, _mm256_set1_epi64x, _mm256_setzero_si256,
    _mm256_srli_epi64, _mm256_storeu_si256,
};
use std::mem::size_of;

use bytemuck::{cast_slice, cast_slice_mut};

use poulpy_cpu_ref::reference::ntt4x30::{
    convolution::cnv_accumulate_schedule, mat_vec::BbcMeta, primes::Primes30, types::Q120bScalar, vec_znx_dft::NttModuleHandle,
};
use poulpy_hal::layouts::{CnvDftAccTermPvec, Module, VecZnxDftBackendMut, ZnxView, ZnxViewMut};

use super::mat_vec_avx::reduce_bbc;
use crate::NTT4x30Avx;

/// Block-group size of the staged flush (matches the reference kernels).
const GROUP: usize = 16;

/// Scratch bytes required by [`cnv_accumulate_pvec_to_dft_avx`]: the group staging.
pub(crate) fn cnv_accumulate_pvec_to_dft_avx_tmp_bytes(res_size: usize) -> usize {
    8 * GROUP * res_size * size_of::<u64>()
}

/// One resolved window: base pointers at block 0 plus the per-block strides.
struct WindowAvx {
    a: *const u32,
    b: *const u32,
    a_stride: usize,
    b_stride: usize,
    len: usize,
}

pub(crate) unsafe fn cnv_accumulate_pvec_to_dft_avx(
    module: &Module<NTT4x30Avx>,
    cnv_offset: usize,
    res: &mut VecZnxDftBackendMut<'_, NTT4x30Avx>,
    res_col: usize,
    terms: &[CnvDftAccTermPvec<'_, NTT4x30Avx>],
    tmp: &mut [u8],
) {
    let n = res.n();
    let res_size = res.size();
    if res_size == 0 {
        return;
    }
    if terms.is_empty() {
        for j in 0..res_size {
            cast_slice_mut::<_, u64>(res.at_mut(res_col, j)).fill(0);
        }
        return;
    }

    let meta: &BbcMeta<Primes30> = module.get_bbc_meta();
    let n_blks = n / 2;

    // Block-major prepared layout: column `col` of a `size`-limb operand spans
    // `8 * n * size` u32 starting at `col * 8 * n * size`; block `blk` holds its
    // `size` rows (16 u32 each) contiguously.
    let term_cols: Vec<(&[u32], &[u32], usize, usize)> = terms
        .iter()
        .map(|t| {
            let a_size = t.a.size();
            let b_size = t.b.size();
            let a_raw: &[Q120bScalar] = t.a.raw();
            let b_raw: &[Q120bScalar] = t.b.raw();
            let a_col: &[u32] = &cast_slice(a_raw)[t.a_col * 8 * n * a_size..(t.a_col + 1) * 8 * n * a_size];
            let b_col: &[u32] = &cast_slice(b_raw)[t.b_col * 8 * n * b_size..(t.b_col + 1) * 8 * n * b_size];
            (a_col, b_col, a_size, b_size)
        })
        .collect();
    let sched = cnv_accumulate_schedule(
        cnv_offset,
        res_size,
        &term_cols.iter().map(|&(_, _, a, b)| (a, b)).collect::<Vec<_>>(),
    );
    let windows: Vec<Vec<WindowAvx>> = sched
        .iter()
        .map(|sched_k| {
            sched_k
                .iter()
                .map(|e| {
                    let (a_col, b_col, a_size, b_size) = term_cols[e.term];
                    WindowAvx {
                        a: unsafe { a_col.as_ptr().add(16 * e.a_row) },
                        b: unsafe { b_col.as_ptr().add(16 * e.b_row) },
                        a_stride: 16 * a_size,
                        b_stride: 16 * b_size,
                        len: e.len,
                    }
                })
                .collect()
        })
        .collect();

    let (prefix, tmp_u64, suffix) = unsafe { tmp.align_to_mut::<u64>() };
    debug_assert!(prefix.is_empty());
    debug_assert!(suffix.is_empty());
    let stage = &mut tmp_u64[..8 * GROUP * res_size];

    unsafe {
        let mask32 = _mm256_set1_epi64x(u32::MAX as i64);
        let mask_h2 = _mm256_set1_epi64x(((1u64 << meta.h) - 1) as i64);
        let s2l_pow_red = _mm256_loadu_si256(meta.s2l_pow_red.as_ptr() as *const __m256i);
        let s2h_pow_red = _mm256_loadu_si256(meta.s2h_pow_red.as_ptr() as *const __m256i);

        for blk in 0..n_blks {
            let grp_pos = blk % GROUP;

            for (k, windows_k) in windows.iter().enumerate() {
                // Per x2 pair half: (low, high) lazy accumulators.
                let mut s0a = _mm256_setzero_si256();
                let mut s1a = _mm256_setzero_si256();
                let mut s0b = _mm256_setzero_si256();
                let mut s1b = _mm256_setzero_si256();

                for w in windows_k {
                    let mut x_ptr = w.a.add(blk * w.a_stride) as *const __m256i;
                    let mut y_ptr = w.b.add(blk * w.b_stride) as *const __m256i;
                    for _ in 0..w.len {
                        let xa = _mm256_loadu_si256(x_ptr);
                        let xb = _mm256_loadu_si256(x_ptr.add(1));
                        let ya = _mm256_loadu_si256(y_ptr);
                        let yb = _mm256_loadu_si256(y_ptr.add(1));

                        // Canonical x: only the x_lo * y_lo product contributes.
                        let pa = _mm256_mul_epu32(xa, ya);
                        let pb = _mm256_mul_epu32(xb, yb);

                        s0a = _mm256_add_epi64(s0a, _mm256_and_si256(pa, mask32));
                        s1a = _mm256_add_epi64(s1a, _mm256_srli_epi64::<32>(pa));
                        s0b = _mm256_add_epi64(s0b, _mm256_and_si256(pb, mask32));
                        s1b = _mm256_add_epi64(s1b, _mm256_srli_epi64::<32>(pb));

                        x_ptr = x_ptr.add(2);
                        y_ptr = y_ptr.add(2);
                    }
                }

                let out = stage.as_mut_ptr().add(8 * (k * GROUP + grp_pos)) as *mut __m256i;
                _mm256_storeu_si256(out, reduce_bbc(s0a, s1a, mask_h2, meta.h, s2l_pow_red, s2h_pow_red));
                _mm256_storeu_si256(out.add(1), reduce_bbc(s0b, s1b, mask_h2, meta.h, s2l_pow_red, s2h_pow_red));
            }

            // Flush the group per limb as one contiguous run.
            let in_group = grp_pos + 1;
            if in_group == GROUP || blk == n_blks - 1 {
                let grp_base = blk + 1 - in_group;
                for k in 0..res_size {
                    let res_u64: &mut [u64] = cast_slice_mut(res.at_mut(res_col, k));
                    res_u64[8 * grp_base..8 * (grp_base + in_group)]
                        .copy_from_slice(&stage[8 * k * GROUP..8 * (k * GROUP + in_group)]);
                }
            }
        }
    }
}
