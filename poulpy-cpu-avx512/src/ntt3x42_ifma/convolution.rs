//! Polynomial convolution AVX512 kernels for [`NTT3x42Ifma`](crate::NTT3x42Ifma).
//!
//! Prepared operands use a block-major layout: for column `col` and x2 NTT
//! block `blk`, all `size` limb rows (8 u64 each) are stored contiguously at
//! `col * (n/2) * size * 8 + blk * size * 8`, with `CnvPVecR` rows in reversed
//! limb order. The apply kernels therefore read both operands sequentially and
//! tile four output limbs per pass over a zero-padded `a` window, reducing
//! each output once with [`reduce_bbc_ifma_simd_512`].

#![allow(dead_code)]

use bytemuck::{cast_slice, cast_slice_mut};
use poulpy_hal::layouts::PrimeSet;
use std::mem::size_of;

use crate::ntt3x42_ifma::{
    kernels::{cond_sub_2q_si512, ntt_avx512},
    module::handle,
    primes::Primes42,
    tables::{harvey_modmul, harvey_quotient},
    traits::{Ntt3x42IfmaAddAssign, Ntt3x42IfmaCFromB, Ntt3x42IfmaFromZnx64},
    types::Q126Scalar,
};
use poulpy_hal::layouts::{
    CnvPVecLBackendMut, CnvPVecLBackendRef, CnvPVecRBackendMut, CnvPVecRBackendRef, Module, VecZnxBackendRef,
    VecZnxBigBackendMut, VecZnxDftBackendMut, ZnxView, ZnxViewMut,
};

use super::mat_vec_ifma::{PrimeConsts512, reduce_bbc_ifma_simd_512, reduce_bbc_single_prime_512};

use crate::NTT3x42Ifma;
use core::arch::x86_64::{
    __m512i, _mm_sfence, _mm512_add_epi64, _mm512_loadu_si512, _mm512_madd52hi_epu64, _mm512_madd52lo_epu64,
    _mm512_setzero_si512, _mm512_storeu_si512, _mm512_stream_si512,
};

// ─────────────────────────────────────────────────────────────────────────────
// Scratch accounting
// ─────────────────────────────────────────────────────────────────────────────

/// Output-tile width of the apply kernels (padded window rows on each side).
const TILE: usize = 4;

/// Block-group size of the accumulate flush.
const CNV_ACC_GROUP: usize = 16;

/// Scratch bytes required by [`cnv_apply_pvec_to_dft_ifma`] and its accumulate variant.
///
/// Stores the padded `a` window plus the accumulate staging group.
pub(crate) fn cnv_apply_pvec_to_dft_ifma_tmp_bytes(res_size: usize, a_size: usize, b_size: usize) -> usize {
    let min_size: usize = res_size.min(a_size + b_size);
    (8 * (a_size + 2 * (TILE - 1)) + 8 * CNV_ACC_GROUP * min_size) * size_of::<u64>()
}

/// Scratch bytes required by [`cnv_pairwise_apply_pvec_to_dft_ifma`]: the apply
/// scratch plus the summed `b` rows.
pub(crate) fn cnv_pairwise_apply_pvec_to_dft_ifma_tmp_bytes(res_size: usize, a_size: usize, b_size: usize) -> usize {
    if a_size == 0 || b_size == 0 || res_size == 0 {
        0
    } else {
        cnv_apply_pvec_to_dft_ifma_tmp_bytes(res_size, a_size, b_size) + 8 * b_size * size_of::<u64>()
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Tiled column kernel
// ─────────────────────────────────────────────────────────────────────────────

/// Convolve one column pair into `res[res_col]`, tiling [`TILE`] output limbs
/// per pass over the zero-padded `a` window.
///
/// - `ACC`: accumulate into `res` (via group-staged `ntt3x42_ifma_add_assign`)
///   instead of overwriting (NT stores).
/// - `PAIRWISE`: operands are the lane-wise sums `a0 + a1` and `b0 + b1`.
#[allow(clippy::too_many_arguments)]
#[target_feature(enable = "avx512ifma,avx512vl")]
unsafe fn conv_columns_ifma<const ACC: bool, const PAIRWISE: bool>(
    cnv_offset: usize,
    res: &mut VecZnxDftBackendMut<'_, NTT3x42Ifma>,
    res_col: usize,
    a0_col: &[u64],
    a1_col: &[u64],
    a_size: usize,
    b0_col: &[u64],
    b1_col: &[u64],
    b_size: usize,
    tmp: &mut [u64],
) {
    let n = res.n();
    let res_size = res.size();

    let bound = a_size + b_size - 1;
    let offset = cnv_offset.min(bound);
    let min_size = res_size.min((bound + 1).saturating_sub(offset));

    let n_blks = n / 2;
    let pad = TILE - 1;
    let win_rows = a_size + 2 * pad;

    let (win, rest) = tmp.split_at_mut(8 * win_rows);
    let (stage, rest) = rest.split_at_mut(if ACC { 8 * CNV_ACC_GROUP * min_size } else { 0 });
    let b_sum: &mut [u64] = &mut rest[..if PAIRWISE { 8 * b_size } else { 0 }];

    unsafe {
        let zero = _mm512_setzero_si512();
        for r in 0..pad {
            _mm512_storeu_si512(win.as_mut_ptr().add(8 * r) as *mut __m512i, zero);
            _mm512_storeu_si512(win.as_mut_ptr().add(8 * (a_size + pad + r)) as *mut __m512i, zero);
        }

        let n_tiles = min_size.div_ceil(TILE);

        for blk in 0..n_blks {
            // Stage this block's a rows (or the pairwise sum) into the padded
            // window; b rows are read in place (summed first when PAIRWISE).
            let a_blk = a0_col.as_ptr().add(blk * 8 * a_size);
            if PAIRWISE {
                let a1_blk = a1_col.as_ptr().add(blk * 8 * a_size);
                for r in 0..a_size {
                    let s = _mm512_add_epi64(
                        _mm512_loadu_si512(a_blk.add(8 * r) as *const __m512i),
                        _mm512_loadu_si512(a1_blk.add(8 * r) as *const __m512i),
                    );
                    _mm512_storeu_si512(win.as_mut_ptr().add(8 * (pad + r)) as *mut __m512i, s);
                }
            } else {
                for r in 0..a_size {
                    let v = _mm512_loadu_si512(a_blk.add(8 * r) as *const __m512i);
                    _mm512_storeu_si512(win.as_mut_ptr().add(8 * (pad + r)) as *mut __m512i, v);
                }
            }
            let b_blk: *const u64 = if PAIRWISE {
                let b0_blk = b0_col.as_ptr().add(blk * 8 * b_size);
                let b1_blk = b1_col.as_ptr().add(blk * 8 * b_size);
                for r in 0..b_size {
                    let s = _mm512_add_epi64(
                        _mm512_loadu_si512(b0_blk.add(8 * r) as *const __m512i),
                        _mm512_loadu_si512(b1_blk.add(8 * r) as *const __m512i),
                    );
                    _mm512_storeu_si512(b_sum.as_mut_ptr().add(8 * r) as *mut __m512i, s);
                }
                b_sum.as_ptr()
            } else {
                b0_col.as_ptr().add(blk * 8 * b_size)
            };

            let grp_pos = blk % CNV_ACC_GROUP;

            for tile in 0..n_tiles {
                let k0 = offset + TILE * tile;
                let j_lo = (k0 + 1).saturating_sub(a_size).min(b_size);
                let j_hi = (k0 + TILE).min(b_size);

                let mut acc_lo0 = _mm512_setzero_si512();
                let mut acc_hi0 = _mm512_setzero_si512();
                let mut acc_lo1 = _mm512_setzero_si512();
                let mut acc_hi1 = _mm512_setzero_si512();
                let mut acc_lo2 = _mm512_setzero_si512();
                let mut acc_hi2 = _mm512_setzero_si512();
                let mut acc_lo3 = _mm512_setzero_si512();
                let mut acc_hi3 = _mm512_setzero_si512();

                if j_lo < j_hi {
                    // b row r holds limb j = b_size-1-r; iterate r ascending.
                    // w_t = padded window row (k0 + t - j) + pad, sliding up one
                    // row per r and reusing three of four registers.
                    let r_start = b_size - j_hi;
                    let r_end = b_size - j_lo;
                    let mut w_ptr = win.as_ptr().add(8 * ((k0 + pad + 1) - j_hi));
                    let mut y_ptr = b_blk.add(8 * r_start);

                    let mut w0 = _mm512_loadu_si512(w_ptr as *const __m512i);
                    let mut w1 = _mm512_loadu_si512(w_ptr.add(8) as *const __m512i);
                    let mut w2 = _mm512_loadu_si512(w_ptr.add(16) as *const __m512i);
                    let mut w3 = _mm512_loadu_si512(w_ptr.add(24) as *const __m512i);

                    let mut r = r_start;
                    loop {
                        let y = _mm512_loadu_si512(y_ptr as *const __m512i);

                        acc_lo0 = _mm512_madd52lo_epu64(acc_lo0, w0, y);
                        acc_hi0 = _mm512_madd52hi_epu64(acc_hi0, w0, y);
                        acc_lo1 = _mm512_madd52lo_epu64(acc_lo1, w1, y);
                        acc_hi1 = _mm512_madd52hi_epu64(acc_hi1, w1, y);
                        acc_lo2 = _mm512_madd52lo_epu64(acc_lo2, w2, y);
                        acc_hi2 = _mm512_madd52hi_epu64(acc_hi2, w2, y);
                        acc_lo3 = _mm512_madd52lo_epu64(acc_lo3, w3, y);
                        acc_hi3 = _mm512_madd52hi_epu64(acc_hi3, w3, y);

                        r += 1;
                        if r == r_end {
                            break;
                        }
                        w0 = w1;
                        w1 = w2;
                        w2 = w3;
                        w_ptr = w_ptr.add(8);
                        w3 = _mm512_loadu_si512(w_ptr.add(24) as *const __m512i);
                        y_ptr = y_ptr.add(8);
                    }
                }

                let k_rel = TILE * tile;
                let mut store = |t: usize, lo: __m512i, hi: __m512i| {
                    let out = reduce_bbc_ifma_simd_512(lo, hi);
                    if ACC {
                        // Limb-major staging keeps each flush run contiguous in res.
                        let dst = stage.as_mut_ptr().add(8 * ((k_rel + t) * CNV_ACC_GROUP + grp_pos));
                        _mm512_storeu_si512(dst as *mut __m512i, out);
                    } else {
                        let res_u64: &mut [u64] = cast_slice_mut(res.at_mut(res_col, k_rel + t));
                        _mm512_stream_si512(res_u64.as_mut_ptr().add(8 * blk) as *mut __m512i, out);
                    }
                };
                store(0, acc_lo0, acc_hi0);
                if k_rel + 1 < min_size {
                    store(1, acc_lo1, acc_hi1);
                }
                if k_rel + 2 < min_size {
                    store(2, acc_lo2, acc_hi2);
                }
                if k_rel + 3 < min_size {
                    store(3, acc_lo3, acc_hi3);
                }
            }

            // Accumulate path: flush the group per limb as one contiguous add.
            if ACC {
                let in_group = grp_pos + 1;
                if in_group == CNV_ACC_GROUP || blk == n_blks - 1 {
                    let grp_base = blk + 1 - in_group;
                    for k in 0..min_size {
                        let res_u64: &mut [u64] = cast_slice_mut(res.at_mut(res_col, k));
                        NTT3x42Ifma::ntt3x42_ifma_add_assign(
                            &mut res_u64[8 * grp_base..8 * (grp_base + in_group)],
                            &stage[8 * k * CNV_ACC_GROUP..8 * (k * CNV_ACC_GROUP + in_group)],
                        );
                    }
                }
            }
        }

        if !ACC {
            for j in min_size..res_size {
                res.at_mut(res_col, j).fill(Q126Scalar([0; 3]));
            }
            // Order the non-temporal stores against any subsequent load of `res`.
            _mm_sfence();
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Entry points
// ─────────────────────────────────────────────────────────────────────────────

fn col_slice(raw: &[Q126Scalar], n: usize, size: usize, col: usize) -> &[u64] {
    let stride = 3 * n * size;
    &cast_slice(raw)[col * stride..(col + 1) * stride]
}

#[inline(always)]
fn col_slice_mut(raw: &mut [Q126Scalar], n: usize, size: usize, col: usize) -> &mut [u64] {
    let stride = 3 * n * size;
    &mut cast_slice_mut(raw)[col * stride..(col + 1) * stride]
}

#[inline(always)]
fn prepared_row_offset(n: usize, size: usize, limb_row: usize, prime: usize, group: usize) -> usize {
    prime * n * size + group * size * 8 + limb_row * 8
}

#[inline(always)]
fn prepared_row(data: &[u64], n: usize, size: usize, limb_row: usize, prime: usize, group: usize) -> &[u64] {
    let off = prepared_row_offset(n, size, limb_row, prime, group);
    &data[off..off + 8]
}

#[target_feature(enable = "avx512f")]
unsafe fn scatter_limb_planar(dst: &mut [u64], src: &[u64], n: usize, size: usize, limb_row: usize) {
    let n_groups = n / 8;
    for prime in 0..3 {
        for group in 0..n_groups {
            let dst_off = prepared_row_offset(n, size, limb_row, prime, group);
            let src_off = prime * n + 8 * group;
            dst[dst_off..dst_off + 8].copy_from_slice(&src[src_off..src_off + 8]);
        }
    }
}

#[target_feature(enable = "avx512f")]
unsafe fn zero_limb_planar(dst: &mut [u64], n: usize, size: usize, limb_row: usize) {
    let n_groups = n / 8;
    for prime in 0..3 {
        for group in 0..n_groups {
            let dst_off = prepared_row_offset(n, size, limb_row, prime, group);
            dst[dst_off..dst_off + 8].fill(0);
        }
    }
}

#[allow(clippy::too_many_arguments)]
#[target_feature(enable = "avx512ifma,avx512vl")]
unsafe fn conv_columns_planar<const ACC: bool, const PAIRWISE: bool>(
    cnv_offset: usize,
    res: &mut VecZnxDftBackendMut<'_, NTT3x42Ifma>,
    res_col: usize,
    a0_col: &[u64],
    a1_col: &[u64],
    a_size: usize,
    b0_col: &[u64],
    b1_col: &[u64],
    b_size: usize,
    tmp: &mut [u64],
) {
    let n = res.n();
    let res_size = res.size();
    let bound = a_size + b_size - 1;
    let offset = cnv_offset.min(bound);
    let min_size = res_size.min((bound + 1).saturating_sub(offset));

    let pad = TILE - 1;
    let win_rows = a_size + 2 * pad;
    let n_groups = n / 8;
    let n_tiles = min_size.div_ceil(TILE);
    let pc = unsafe { [PrimeConsts512::new(0), PrimeConsts512::new(1), PrimeConsts512::new(2)] };
    let (win, rest) = tmp.split_at_mut(8 * win_rows);
    let b_sum = &mut rest[..if PAIRWISE { 8 * b_size } else { 0 }];

    unsafe {
        let zero = _mm512_setzero_si512();
        for r in 0..pad {
            _mm512_storeu_si512(win.as_mut_ptr().add(8 * r) as *mut __m512i, zero);
            _mm512_storeu_si512(win.as_mut_ptr().add(8 * (a_size + pad + r)) as *mut __m512i, zero);
        }

        for (prime, pc_prime) in pc.iter().enumerate() {
            for group in 0..n_groups {
                for r in 0..a_size {
                    let a0 = prepared_row(a0_col, n, a_size, r, prime, group).as_ptr();
                    let v = if PAIRWISE {
                        let a1 = prepared_row(a1_col, n, a_size, r, prime, group).as_ptr();
                        _mm512_add_epi64(
                            _mm512_loadu_si512(a0 as *const __m512i),
                            _mm512_loadu_si512(a1 as *const __m512i),
                        )
                    } else {
                        _mm512_loadu_si512(a0 as *const __m512i)
                    };
                    _mm512_storeu_si512(win.as_mut_ptr().add(8 * (pad + r)) as *mut __m512i, v);
                }

                let b_base: *const u64 = if PAIRWISE {
                    for r in 0..b_size {
                        let b0 = prepared_row(b0_col, n, b_size, r, prime, group).as_ptr();
                        let b1 = prepared_row(b1_col, n, b_size, r, prime, group).as_ptr();
                        let s = _mm512_add_epi64(
                            _mm512_loadu_si512(b0 as *const __m512i),
                            _mm512_loadu_si512(b1 as *const __m512i),
                        );
                        _mm512_storeu_si512(b_sum.as_mut_ptr().add(8 * r) as *mut __m512i, s);
                    }
                    b_sum.as_ptr()
                } else {
                    b0_col.as_ptr().add(prepared_row_offset(n, b_size, 0, prime, group))
                };

                for tile in 0..n_tiles {
                    let k0 = offset + TILE * tile;
                    let j_lo = (k0 + 1).saturating_sub(a_size).min(b_size);
                    let j_hi = (k0 + TILE).min(b_size);

                    let mut acc_lo0 = _mm512_setzero_si512();
                    let mut acc_hi0 = _mm512_setzero_si512();
                    let mut acc_lo1 = _mm512_setzero_si512();
                    let mut acc_hi1 = _mm512_setzero_si512();
                    let mut acc_lo2 = _mm512_setzero_si512();
                    let mut acc_hi2 = _mm512_setzero_si512();
                    let mut acc_lo3 = _mm512_setzero_si512();
                    let mut acc_hi3 = _mm512_setzero_si512();

                    if j_lo < j_hi {
                        let r_start = b_size - j_hi;
                        let r_end = b_size - j_lo;
                        let mut w_ptr = win.as_ptr().add(8 * ((k0 + pad + 1) - j_hi));
                        let mut y_ptr = b_base.add(8 * r_start);

                        let mut w0 = _mm512_loadu_si512(w_ptr as *const __m512i);
                        let mut w1 = _mm512_loadu_si512(w_ptr.add(8) as *const __m512i);
                        let mut w2 = _mm512_loadu_si512(w_ptr.add(16) as *const __m512i);
                        let mut w3 = _mm512_loadu_si512(w_ptr.add(24) as *const __m512i);

                        let mut r = r_start;
                        loop {
                            let y = _mm512_loadu_si512(y_ptr as *const __m512i);

                            acc_lo0 = _mm512_madd52lo_epu64(acc_lo0, w0, y);
                            acc_hi0 = _mm512_madd52hi_epu64(acc_hi0, w0, y);
                            acc_lo1 = _mm512_madd52lo_epu64(acc_lo1, w1, y);
                            acc_hi1 = _mm512_madd52hi_epu64(acc_hi1, w1, y);
                            acc_lo2 = _mm512_madd52lo_epu64(acc_lo2, w2, y);
                            acc_hi2 = _mm512_madd52hi_epu64(acc_hi2, w2, y);
                            acc_lo3 = _mm512_madd52lo_epu64(acc_lo3, w3, y);
                            acc_hi3 = _mm512_madd52hi_epu64(acc_hi3, w3, y);

                            r += 1;
                            if r == r_end {
                                break;
                            }
                            w0 = w1;
                            w1 = w2;
                            w2 = w3;
                            w_ptr = w_ptr.add(8);
                            w3 = _mm512_loadu_si512(w_ptr.add(24) as *const __m512i);
                            y_ptr = y_ptr.add(8);
                        }
                    }

                    let mut store = |t: usize, lo: __m512i, hi: __m512i| {
                        let k_rel = TILE * tile + t;
                        if k_rel >= min_size {
                            return;
                        }
                        let out = reduce_bbc_single_prime_512(
                            lo,
                            hi,
                            pc_prime.q,
                            pc_prime.q2,
                            pc_prime.pow42,
                            pc_prime.pow52,
                            pc_prime.pow52_quot,
                        );
                        let res_u64: &mut [u64] = cast_slice_mut(res.at_mut(res_col, k_rel));
                        let dst = res_u64.as_mut_ptr().add(prime * n + 8 * group) as *mut __m512i;
                        if ACC {
                            let d = _mm512_loadu_si512(dst as *const __m512i);
                            _mm512_storeu_si512(dst, cond_sub_2q_si512(_mm512_add_epi64(d, out), pc_prime.q2));
                        } else {
                            _mm512_stream_si512(dst, out);
                        }
                    };

                    store(0, acc_lo0, acc_hi0);
                    store(1, acc_lo1, acc_hi1);
                    store(2, acc_lo2, acc_hi2);
                    store(3, acc_lo3, acc_hi3);
                }
            }
        }

        if !ACC {
            for j in min_size..res_size {
                cast_slice_mut::<_, u64>(res.at_mut(res_col, j)).fill(0);
            }
            _mm_sfence();
        }
    }
}

#[inline(always)]
fn limb_plane(data: &[u64], n: usize, limb: usize, prime: usize) -> &[u64] {
    let base = limb * 3 * n + prime * n;
    &data[base..base + n]
}

#[inline(always)]
fn limb_plane_mut(data: &mut [u64], n: usize, limb: usize, prime: usize) -> &mut [u64] {
    let base = limb * 3 * n + prime * n;
    &mut data[base..base + n]
}

#[inline(always)]
fn mul_mod_lazy(a: u64, b: u64, prime: usize) -> u64 {
    let q = Primes42::Q[prime];
    let b = b % q;
    harvey_modmul(a, b, harvey_quotient(b, q), q) % q
}

#[allow(clippy::too_many_arguments)]
fn conv_columns_planar_scalar<const ACC: bool, const PAIRWISE: bool>(
    cnv_offset: usize,
    res: &mut VecZnxDftBackendMut<'_, NTT3x42Ifma>,
    res_col: usize,
    a0_col: &[u64],
    a1_col: &[u64],
    a_size: usize,
    b0_col: &[u64],
    b1_col: &[u64],
    b_size: usize,
) {
    let n = res.n();
    let res_size = res.size();
    let bound = a_size + b_size - 1;
    let offset = cnv_offset.min(bound);
    let min_size = res_size.min((bound + 1).saturating_sub(offset));
    let mut tmp = vec![0u64; 3 * n];

    for k_rel in 0..min_size {
        let k_abs = offset + k_rel;
        tmp.fill(0);
        let j_min = k_abs.saturating_sub(a_size - 1);
        let j_max = (k_abs + 1).min(b_size);
        for prime in 0..3 {
            let q = Primes42::Q[prime];
            let out = limb_plane_mut(&mut tmp, n, 0, prime);
            for j in j_min..j_max {
                let a_limb = k_abs - j;
                let a0 = limb_plane(a0_col, n, a_limb, prime);
                let b0 = limb_plane(b0_col, n, j, prime);
                if PAIRWISE {
                    let a1 = limb_plane(a1_col, n, a_limb, prime);
                    let b1 = limb_plane(b1_col, n, j, prime);
                    for i in 0..n {
                        let av = a0[i] + a1[i];
                        let bv = (b0[i] + b1[i]) % q;
                        out[i] = ((out[i] as u128 + mul_mod_lazy(av, bv, prime) as u128) % q as u128) as u64;
                    }
                } else {
                    for i in 0..n {
                        out[i] = ((out[i] as u128 + mul_mod_lazy(a0[i], b0[i], prime) as u128) % q as u128) as u64;
                    }
                }
            }
        }

        let res_u64: &mut [u64] = cast_slice_mut(res.at_mut(res_col, k_rel));
        if ACC {
            NTT3x42Ifma::ntt3x42_ifma_add_assign(res_u64, &tmp);
        } else {
            res_u64.copy_from_slice(&tmp);
        }
    }

    if !ACC {
        for j in min_size..res_size {
            cast_slice_mut::<_, u64>(res.at_mut(res_col, j)).fill(0);
        }
    }
}

/// DFT-domain bivariate convolution `res[k] = Σ a[j] ⊙ b[k−j]`.
#[allow(clippy::too_many_arguments)]
#[target_feature(enable = "avx512ifma,avx512vl")]
pub(crate) unsafe fn cnv_apply_pvec_to_dft_ifma(
    res: &mut VecZnxDftBackendMut<'_, NTT3x42Ifma>,
    cnv_offset: usize,
    res_col: usize,
    a: &CnvPVecLBackendRef<'_, NTT3x42Ifma>,
    a_col: usize,
    b: &CnvPVecRBackendRef<'_, NTT3x42Ifma>,
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

    let (prefix, tmp_u64, suffix) = unsafe { tmp.align_to_mut::<u64>() };
    debug_assert!(prefix.is_empty());
    debug_assert!(suffix.is_empty());

    let a_col_u64 = col_slice(a.raw(), n, a_size, a_col);
    let b_col_u64 = col_slice(b.raw(), n, b_size, b_col);
    unsafe {
        conv_columns_planar::<false, false>(
            cnv_offset, res, res_col, a_col_u64, a_col_u64, a_size, b_col_u64, b_col_u64, b_size, tmp_u64,
        );
    }
}

/// Accumulating variant of [`cnv_apply_pvec_to_dft_ifma`]: `res[k] += Σ a[j] ⊙ b[k−j]`
/// via `ntt3x42_ifma_add_assign` (bit-identical to apply + DFT add).
/// Limbs `>= min_size` are left untouched.
#[allow(clippy::too_many_arguments)]
#[target_feature(enable = "avx512ifma,avx512vl")]
pub(crate) unsafe fn cnv_apply_pvec_to_dft_accumulate_ifma(
    res: &mut VecZnxDftBackendMut<'_, NTT3x42Ifma>,
    cnv_offset: usize,
    res_col: usize,
    a: &CnvPVecLBackendRef<'_, NTT3x42Ifma>,
    a_col: usize,
    b: &CnvPVecRBackendRef<'_, NTT3x42Ifma>,
    b_col: usize,
    tmp: &mut [u8],
) {
    let n = res.n();
    let res_size = res.size();
    let a_size = a.size();
    let b_size = b.size();
    if res_size == 0 || a_size == 0 || b_size == 0 {
        return;
    }

    let (prefix, tmp_u64, suffix) = unsafe { tmp.align_to_mut::<u64>() };
    debug_assert!(prefix.is_empty());
    debug_assert!(suffix.is_empty());

    let a_col_u64 = col_slice(a.raw(), n, a_size, a_col);
    let b_col_u64 = col_slice(b.raw(), n, b_size, b_col);
    unsafe {
        conv_columns_planar::<true, false>(
            cnv_offset, res, res_col, a_col_u64, a_col_u64, a_size, b_col_u64, b_col_u64, b_size, tmp_u64,
        );
    }
}

/// Pairwise DFT-domain convolution:
/// `res[k] = Σ (a[col_0, k−j] + a[col_1, k−j]) ⊙ (b[col_0, j] + b[col_1, j])`.
///
/// When `col_0 == col_1`, delegates to [`cnv_apply_pvec_to_dft_ifma`].
#[allow(clippy::too_many_arguments)]
#[target_feature(enable = "avx512ifma,avx512vl")]
pub(crate) unsafe fn cnv_pairwise_apply_pvec_to_dft_ifma(
    res: &mut VecZnxDftBackendMut<'_, NTT3x42Ifma>,
    cnv_offset: usize,
    res_col: usize,
    a: &CnvPVecLBackendRef<'_, NTT3x42Ifma>,
    b: &CnvPVecRBackendRef<'_, NTT3x42Ifma>,
    col_0: usize,
    col_1: usize,
    tmp: &mut [u8],
) {
    if col_0 == col_1 {
        unsafe { cnv_apply_pvec_to_dft_ifma(res, cnv_offset, res_col, a, col_0, b, col_1, tmp) };
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

    let (prefix, tmp_u64, suffix) = unsafe { tmp.align_to_mut::<u64>() };
    debug_assert!(prefix.is_empty());
    debug_assert!(suffix.is_empty());

    let a0 = col_slice(a.raw(), n, a_size, col_0);
    let a1 = col_slice(a.raw(), n, a_size, col_1);
    let b0 = col_slice(b.raw(), n, b_size, col_0);
    let b1 = col_slice(b.raw(), n, b_size, col_1);
    unsafe { conv_columns_planar::<false, true>(cnv_offset, res, res_col, a0, a1, a_size, b0, b1, b_size, tmp_u64) };
}

// ─────────────────────────────────────────────────────────────────────────────
// Prepare paths
// ─────────────────────────────────────────────────────────────────────────────

/// Scatter one NTT-domain limb (x2 blocks of 8 u64) into block-major rows.
fn scatter_limb(dst: &mut [u64], src: &[u64], size: usize, row: usize, n_blks: usize) {
    for blk in 0..n_blks {
        let off = (blk * size + row) * 8;
        dst[off..off + 8].copy_from_slice(&src[8 * blk..8 * blk + 8]);
    }
}

fn zero_row(dst: &mut [u64], size: usize, row: usize, n_blks: usize) {
    for blk in 0..n_blks {
        let off = (blk * size + row) * 8;
        dst[off..off + 8].fill(0);
    }
}

/// Scratch bytes required by [`cnv_prepare_left_pvec`]: one NTT-domain limb.
pub(crate) fn cnv_prepare_left_pvec_tmp_bytes(n: usize) -> usize {
    3 * n * size_of::<u64>()
}

pub(crate) fn cnv_prepare_left_pvec(
    module: &Module<NTT3x42Ifma>,
    res: &mut CnvPVecLBackendMut<'_, NTT3x42Ifma>,
    a: &VecZnxBackendRef<'_, NTT3x42Ifma>,
    mask: i64,
    tmp: &mut [u8],
) {
    let n = res.n();
    let table = &handle(module).table_ntt;
    let cols = res.cols();
    let res_size = res.size();
    let min_size = res_size.min(a.size());

    let (prefix, tmp_u64, suffix) = unsafe { tmp.align_to_mut::<u64>() };
    debug_assert!(prefix.is_empty());
    debug_assert!(suffix.is_empty());
    let limb = &mut tmp_u64[..3 * n];

    let res_raw = res.raw_mut();
    for col in 0..cols {
        let dst = col_slice_mut(res_raw, n, res_size, col);
        for j in 0..min_size {
            if j + 1 == min_size {
                NTT3x42Ifma::ntt3x42_ifma_from_znx64_masked(limb, a.at(col, j), mask);
            } else {
                NTT3x42Ifma::ntt3x42_ifma_from_znx64(limb, a.at(col, j));
            }
            // Lazy [0, 4q): the left operand feeds only the BBC product (bound 2^44 > 4q).
            unsafe { ntt_avx512::<Primes42>(table, limb, true) };
            unsafe { scatter_limb_planar(dst, limb, n, res_size, j) };
        }
        for j in min_size..res_size {
            unsafe { zero_limb_planar(dst, n, res_size, j) };
        }
    }
}

/// Scratch bytes required by [`cnv_prepare_right_pvec`]: NTT and converted limbs.
pub(crate) fn cnv_prepare_right_pvec_tmp_bytes(n: usize) -> usize {
    6 * n * size_of::<u64>()
}

pub(crate) fn cnv_prepare_right_pvec(
    module: &Module<NTT3x42Ifma>,
    res: &mut CnvPVecRBackendMut<'_, NTT3x42Ifma>,
    a: &VecZnxBackendRef<'_, NTT3x42Ifma>,
    mask: i64,
    tmp: &mut [u64],
) {
    let n = res.n();
    let table = &handle(module).table_ntt;
    let cols = res.cols();
    let res_size = res.size();
    let min_size = res_size.min(a.size());

    let (limb_b, limb_c) = tmp[..6 * n].split_at_mut(3 * n);

    let res_raw = res.raw_mut();
    for col in 0..cols {
        let dst = col_slice_mut(res_raw, n, res_size, col);
        for j in 0..min_size {
            if j + 1 == min_size {
                NTT3x42Ifma::ntt3x42_ifma_from_znx64_masked(limb_b, a.at(col, j), mask);
            } else {
                NTT3x42Ifma::ntt3x42_ifma_from_znx64(limb_b, a.at(col, j));
            }
            // Lazy [0, 4q): c_from_b re-reduces from [0, 4q).
            unsafe { ntt_avx512::<Primes42>(table, limb_b, true) };
            NTT3x42Ifma::ntt3x42_ifma_c_from_b(n, cast_slice_mut(limb_c), limb_b);
            unsafe { scatter_limb_planar(dst, limb_c, n, res_size, res_size - 1 - j) };
        }
        for j in min_size..res_size {
            unsafe { zero_limb_planar(dst, n, res_size, res_size - 1 - j) };
        }
    }
}

/// Scratch bytes required by [`cnv_prepare_self_pvec`]: NTT and converted limbs.
pub(crate) fn cnv_prepare_self_pvec_tmp_bytes(n: usize) -> usize {
    6 * n * size_of::<u64>()
}

pub(crate) fn cnv_prepare_self_pvec(
    module: &Module<NTT3x42Ifma>,
    left: &mut CnvPVecLBackendMut<'_, NTT3x42Ifma>,
    right: &mut CnvPVecRBackendMut<'_, NTT3x42Ifma>,
    a: &VecZnxBackendRef<'_, NTT3x42Ifma>,
    mask: i64,
    tmp: &mut [u8],
) {
    let n = left.n();
    let table = &handle(module).table_ntt;
    let cols = left.cols();
    let res_size = left.size();
    let min_size = res_size.min(a.size());

    let (prefix, tmp_u64, suffix) = unsafe { tmp.align_to_mut::<u64>() };
    debug_assert!(prefix.is_empty());
    debug_assert!(suffix.is_empty());
    let (limb_b, limb_c) = tmp_u64[..6 * n].split_at_mut(3 * n);

    let left_raw = left.raw_mut();
    let right_raw = right.raw_mut();
    for col in 0..cols {
        let dst_l = col_slice_mut(left_raw, n, res_size, col);
        let dst_r = col_slice_mut(right_raw, n, res_size, col);
        for j in 0..min_size {
            if j + 1 == min_size {
                NTT3x42Ifma::ntt3x42_ifma_from_znx64_masked(limb_b, a.at(col, j), mask);
            } else {
                NTT3x42Ifma::ntt3x42_ifma_from_znx64(limb_b, a.at(col, j));
            }
            // Lazy [0, 4q): left operand → BBC (bound 2^44 > 4q); limb_b → c_from_b (re-reduces).
            unsafe { ntt_avx512::<Primes42>(table, limb_b, true) };
            unsafe { scatter_limb_planar(dst_l, limb_b, n, res_size, j) };
            NTT3x42Ifma::ntt3x42_ifma_c_from_b(n, cast_slice_mut(limb_c), limb_b);
            unsafe { scatter_limb_planar(dst_r, limb_c, n, res_size, res_size - 1 - j) };
        }
        for j in min_size..res_size {
            unsafe { zero_limb_planar(dst_l, n, res_size, j) };
            unsafe { zero_limb_planar(dst_r, n, res_size, res_size - 1 - j) };
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// By-const apply (coefficient domain, unchanged by the prepared layout)
// ─────────────────────────────────────────────────────────────────────────────

pub(crate) fn cnv_by_const_apply_tmp_bytes(_res_size: usize, _a_size: usize, _b_size: usize) -> usize {
    0
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn cnv_by_const_apply(
    cnv_offset: usize,
    res: &mut VecZnxBigBackendMut<'_, NTT3x42Ifma>,
    res_col: usize,
    a: &VecZnxBackendRef<'_, NTT3x42Ifma>,
    a_col: usize,
    b: &VecZnxBackendRef<'_, NTT3x42Ifma>,
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
