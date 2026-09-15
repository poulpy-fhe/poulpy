//! Bivariate convolution operations for the NTT4x30 backend family.
//!
//! Prepared operands use a block-major layout: for column `col` and x2 NTT
//! block `blk`, all `size` limb rows (16 u32 each) are stored contiguously at
//! `col * (n/2) * size * 16 + blk * size * 16` in u32 units. `CnvPVecL` rows
//! hold the canonical (`% q`, kernel-ready) u32 encoding produced by
//! [`NttPackLeft1BlkX2`]; `CnvPVecR` rows hold q120c in reversed limb order.
//! The apply kernels read both operands sequentially and tile four output
//! limbs per pass over a zero-padded `a` window via
//! [`NttMulBbc1ColX2::ntt_mul_bbc_tile4_x2`].

use bytemuck::{cast_slice, cast_slice_mut};
use poulpy_hal::execution::TaskExecutor;

use crate::{
    layouts::{
        Backend, CnvPVecLBackendMut, CnvPVecLBackendRef, CnvPVecRBackendMut, CnvPVecRBackendRef, HostDataMut, HostDataRef,
        VecZnxBackendRef, VecZnxBigBackendMut, VecZnxDftBackendMut, ZnxView, ZnxViewMut,
    },
    reference::{
        assert_sparse_degree,
        ntt4x30::{
            NttAddAssign, NttCFromB, NttDFTExecute, NttFromZnx64, NttMulBbc1ColX2, NttPackLeft1BlkX2,
            ntt::NttTable,
            primes::{PrimeSet, Primes30},
            types::Q120bScalar,
            vec_znx_dft::NttModuleHandle,
        },
        sparse_log_gap,
    },
};

// ──────────────────────────────────────────────────────────────────────────────
// Scratch accounting
// ──────────────────────────────────────────────────────────────────────────────

/// Output-tile width of the apply kernels (padded window rows on each side).
const TILE: usize = 4;

/// Block-group size of the accumulate flush.
pub(crate) const CNV_ACC_GROUP: usize = 16;

/// Block-group size of the prepare canonicalize-and-scatter staging.
const PREP_GROUP: usize = 64;

/// Scratch bytes required by [`ntt4x30_cnv_apply_dft`] and its accumulate and
/// pairwise variants: the padded `a` window, the staged `a1` rows of a pairwise
/// product, the staged `b` rows (a sparse `b` is gathered, a pairwise `b` is
/// lazily summed) and its `b1` twin, plus the accumulate staging group.
pub fn ntt4x30_cnv_apply_dft_tmp_bytes(res_size: usize, a_size: usize, b_size: usize) -> usize {
    let min_size: usize = res_size.min(a_size + b_size);
    16 * (2 * a_size + 2 * (TILE - 1) + 2 * b_size) * size_of::<u32>() + 8 * CNV_ACC_GROUP * min_size * size_of::<u64>()
}

/// Scratch bytes required by [`ntt4x30_cnv_pairwise_apply_dft`]: the apply
/// scratch, which already stages both operands' second columns.
pub fn ntt4x30_cnv_pairwise_apply_dft_tmp_bytes(res_size: usize, a_size: usize, b_size: usize) -> usize {
    if a_size == 0 || b_size == 0 || res_size == 0 {
        0
    } else {
        ntt4x30_cnv_apply_dft_tmp_bytes(res_size, a_size, b_size)
    }
}

// ──────────────────────────────────────────────────────────────────────────────
// Tiled column kernel
// ──────────────────────────────────────────────────────────────────────────────

/// Canonical modular sum of one window row: `dst = (a + b) mod q` per active
/// u32 lane (odd lanes are zero in the canonical encoding).
fn canonical_sum_row(dst: &mut [u32], a: &[u32], b: &[u32]) {
    for i in 0..16 {
        let q = Primes30::Q[(i % 8) / 2];
        let mut s = a[i] + b[i];
        if s >= q {
            s -= q;
        }
        dst[i] = s;
    }
}

/// Canonical modular add of one window row in place: `dst = (dst + b) mod q` per
/// active u32 lane (odd lanes are zero in the canonical encoding).
fn canonical_add_row_assign(dst: &mut [u32], b: &[u32]) {
    for i in 0..16 {
        let q = Primes30::Q[(i % 8) / 2];
        let mut s = dst[i] + b[i];
        if s >= q {
            s -= q;
        }
        dst[i] = s;
    }
}

/// Rows `0..size` of degree-`N` x2 block `blk`, gathered from a degree-`n`
/// prepared column `src` (block-major, `size` rows of 16 u32 per block, slot `s`
/// of a row at `[8s, 8s + 8)`), `N = n << log_gap`: degree-`N` slot `i` reads
/// degree-`n` slot `i >> log_gap`, so the two slots of a block read one slot
/// once `log_gap >= 1`. Row order is preserved, which keeps the reversed limb
/// order of a `CnvPVecR` column.
fn gather_sparse_x2_rows(dst: &mut [u32], src: &[u32], size: usize, blk: usize, log_gap: usize) {
    for lane in 0..2 {
        let slot: usize = (2 * blk + lane) >> log_gap;
        let base: usize = (slot >> 1) * size * 16 + 8 * (slot & 1);
        for r in 0..size {
            dst[16 * r + 8 * lane..16 * r + 8 * lane + 8].copy_from_slice(&src[base + 16 * r..base + 16 * r + 8]);
        }
    }
}

/// Row `row` of degree-`N` x2 block `blk` of a prepared column, as its two slot
/// halves: a slice into the column when dense, the two gathered slots in `buf`
/// when sparse.
fn x2_row<'s>(col: &'s [u32], size: usize, blk: usize, row: usize, log_gap: usize, buf: &'s mut [[u32; 8]; 2]) -> &'s [[u32; 8]] {
    if log_gap == 0 {
        return cast_slice(&col[(blk * size + row) * 16..(blk * size + row + 1) * 16]);
    }
    for (lane, dst) in buf.iter_mut().enumerate() {
        let slot: usize = (2 * blk + lane) >> log_gap;
        let s: usize = ((slot >> 1) * size + row) * 16 + 8 * (slot & 1);
        dst.copy_from_slice(&col[s..s + 8]);
    }
    &buf[..]
}

/// Convolve one column pair into `res[res_col]`, tiling [`TILE`] output limbs
/// per pass over the zero-padded `a` window.
///
/// - `ACC`: accumulate into `res` (via group-staged `ntt_add_assign`) instead
///   of overwriting.
/// - `PAIRWISE`: operands are `(a0 + a1) mod q` and the lazy sum `b0 + b1`.
///
/// `a_log_gap` and `b_log_gap` are `log2(N / n)` of a degree-`n` operand (zero
/// when dense): its x2 rows are gathered block by block through
/// [`gather_sparse_x2_rows`].
#[allow(clippy::too_many_arguments)]
unsafe fn ntt4x30_conv_block_group<BE, const ACC: bool, const PAIRWISE: bool>(
    module: &(impl NttModuleHandle + Sync),
    res_addr: usize,
    res_limb_words: usize,
    res_cols: usize,
    res_col: usize,
    min_size: usize,
    offset: usize,
    block_start: usize,
    block_count: usize,
    a0_col: &[u32],
    a1_col: &[u32],
    a_size: usize,
    a_log_gap: usize,
    b0_col: &[u32],
    b1_col: &[u32],
    b_size: usize,
    b_log_gap: usize,
    tmp: &mut [u8],
) where
    BE: Backend<DftWord = Q120bScalar, ZnxWord = i64> + NttAddAssign + NttMulBbc1ColX2,
    for<'x> <BE as Backend>::BufRef<'x>: HostDataRef,
    for<'x> <BE as Backend>::BufMut<'x>: crate::layouts::HostDataMut,
{
    let meta = module.get_bbc_meta();
    let pad = TILE - 1;
    let win_rows = a_size + 2 * pad;

    let (prefix, tmp_u64, suffix) = unsafe { tmp.align_to_mut::<u64>() };
    assert!(prefix.is_empty());
    assert!(suffix.is_empty());
    let (stage, rest) = tmp_u64.split_at_mut(8 * CNV_ACC_GROUP * min_size);
    let rest_u32: &mut [u32] = cast_slice_mut(rest);
    let (win, rest_u32) = rest_u32.split_at_mut(16 * win_rows);
    let (b_stage, rest_u32) = rest_u32.split_at_mut(16 * b_size);
    let (a1_stage, rest_u32) = rest_u32.split_at_mut(16 * a_size);
    let b1_stage: &mut [u32] = &mut rest_u32[..16 * b_size];

    win[..16 * pad].fill(0);
    win[16 * (a_size + pad)..].fill(0);

    let n_tiles = min_size.div_ceil(TILE);
    let mut out = [0u64; 8 * TILE];

    for local_blk in 0..block_count {
        let blk = block_start + local_blk;
        // Stage this block's a rows (or the canonical pairwise sum) into the
        // padded window, gathered when sparse; b rows are read in place when
        // dense and not pairwise, otherwise staged (lazy-summed when PAIRWISE).
        let win_a: &mut [u32] = &mut win[16 * pad..16 * (pad + a_size)];
        if a_log_gap == 0 {
            let a0_blk: &[u32] = &a0_col[blk * 16 * a_size..(blk + 1) * 16 * a_size];
            if PAIRWISE {
                let a1_blk: &[u32] = &a1_col[blk * 16 * a_size..(blk + 1) * 16 * a_size];
                for r in 0..a_size {
                    canonical_sum_row(&mut win_a[16 * r..16 * (r + 1)], &a0_blk[16 * r..], &a1_blk[16 * r..]);
                }
            } else {
                win_a.copy_from_slice(a0_blk);
            }
        } else {
            gather_sparse_x2_rows(win_a, a0_col, a_size, blk, a_log_gap);
            if PAIRWISE {
                gather_sparse_x2_rows(a1_stage, a1_col, a_size, blk, a_log_gap);
                for r in 0..a_size {
                    canonical_add_row_assign(&mut win_a[16 * r..16 * (r + 1)], &a1_stage[16 * r..16 * (r + 1)]);
                }
            }
        }
        let b_blk: &[u32] = if b_log_gap == 0 {
            let b0_blk: &[u32] = &b0_col[blk * 16 * b_size..(blk + 1) * 16 * b_size];
            if PAIRWISE {
                let b1_blk: &[u32] = &b1_col[blk * 16 * b_size..(blk + 1) * 16 * b_size];
                for (d, (x, y)) in b_stage.iter_mut().zip(b0_blk.iter().zip(b1_blk.iter())) {
                    *d = x + y;
                }
                &*b_stage
            } else {
                b0_blk
            }
        } else {
            gather_sparse_x2_rows(b_stage, b0_col, b_size, blk, b_log_gap);
            if PAIRWISE {
                gather_sparse_x2_rows(b1_stage, b1_col, b_size, blk, b_log_gap);
                for (d, x) in b_stage.iter_mut().zip(b1_stage.iter()) {
                    *d += *x;
                }
            }
            &*b_stage
        };

        let grp_pos = local_blk;

        for tile in 0..n_tiles {
            let k0 = offset + TILE * tile;
            let j_lo = (k0 + 1).saturating_sub(a_size).min(b_size);
            let j_hi = (k0 + TILE).min(b_size);
            let len = j_hi.saturating_sub(j_lo);

            // b row r holds limb j = b_size-1-r; output t reads window rows
            // starting at (k0 + pad + 1 - j_hi) + t over `len` rows.
            let win_base = (k0 + pad + 1)
                .saturating_sub(j_hi)
                .min(win_rows.saturating_sub(TILE - 1 + len));
            let r_start = b_size - j_hi;
            BE::ntt_mul_bbc_tile4_x2(meta, len, &mut out, &win[16 * win_base..], &b_blk[16 * r_start..]);

            let k_rel = TILE * tile;
            for t in 0..TILE.min(min_size - k_rel) {
                // Limb-major staging keeps each flush run contiguous in res
                // (direct per-limb stores would alias one L1 set).
                let off = 8 * ((k_rel + t) * CNV_ACC_GROUP + grp_pos);
                for q in 0..8 {
                    stage[off + q] = out[8 * t + q];
                }
            }
        }

        // Flush the group per limb as one contiguous run.
    }

    for k in 0..min_size {
        let dst = unsafe {
            std::slice::from_raw_parts_mut(
                (res_addr as *mut u64).add(res_limb_words * (k * res_cols + res_col) + 8 * block_start),
                8 * block_count,
            )
        };
        let run = &stage[8 * k * CNV_ACC_GROUP..8 * (k * CNV_ACC_GROUP + block_count)];
        if ACC {
            BE::ntt_add_assign(dst, run);
        } else {
            dst.copy_from_slice(run);
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn ntt4x30_conv_columns<BE, const ACC: bool, const PAIRWISE: bool>(
    module: &(impl NttModuleHandle + Sync),
    cnv_offset: usize,
    res: &mut VecZnxDftBackendMut<'_, BE>,
    res_col: usize,
    a0_col: &[u32],
    a1_col: &[u32],
    a_size: usize,
    a_log_gap: usize,
    b0_col: &[u32],
    b1_col: &[u32],
    b_size: usize,
    b_log_gap: usize,
    tmp: &mut [u8],
) where
    BE: Backend<DftWord = Q120bScalar, ZnxWord = i64> + NttAddAssign + NttMulBbc1ColX2,
    for<'x> <BE as Backend>::BufRef<'x>: HostDataRef,
    for<'x> <BE as Backend>::BufMut<'x>: crate::layouts::HostDataMut,
{
    let n = res.n();
    let res_size = res.size();
    let bound = a_size + b_size - 1;
    let offset = cnv_offset.min(bound);
    let min_size = res_size.min((bound + 1).saturating_sub(offset));
    let n_blks = n / 2;
    let group_count = n_blks.div_ceil(CNV_ACC_GROUP);
    let res_limb_words = cast_slice::<Q120bScalar, u64>(res.at(res_col, 0)).len();
    let res_cols = res.cols();
    let res_addr = cast_slice_mut::<Q120bScalar, u64>(res.raw_mut()).as_mut_ptr() as usize;
    let task_tmp_bytes = if PAIRWISE {
        ntt4x30_cnv_pairwise_apply_dft_tmp_bytes(res_size, a_size, b_size)
    } else {
        ntt4x30_cnv_apply_dft_tmp_bytes(res_size, a_size, b_size)
    };

    if BE::TaskExecutor::is_parallel() && group_count > 1 {
        BE::TaskExecutor::for_each_chunked(group_count, tmp, task_tmp_bytes, |local_tmp, group| unsafe {
            let block_start = group * CNV_ACC_GROUP;
            ntt4x30_conv_block_group::<BE, ACC, PAIRWISE>(
                module,
                res_addr,
                res_limb_words,
                res_cols,
                res_col,
                min_size,
                offset,
                block_start,
                CNV_ACC_GROUP.min(n_blks - block_start),
                a0_col,
                a1_col,
                a_size,
                a_log_gap,
                b0_col,
                b1_col,
                b_size,
                b_log_gap,
                local_tmp,
            );
        });
    } else {
        for group in 0..group_count {
            let block_start = group * CNV_ACC_GROUP;
            unsafe {
                ntt4x30_conv_block_group::<BE, ACC, PAIRWISE>(
                    module,
                    res_addr,
                    res_limb_words,
                    res_cols,
                    res_col,
                    min_size,
                    offset,
                    block_start,
                    CNV_ACC_GROUP.min(n_blks - block_start),
                    a0_col,
                    a1_col,
                    a_size,
                    a_log_gap,
                    b0_col,
                    b1_col,
                    b_size,
                    b_log_gap,
                    tmp,
                );
            }
        }
    }

    if !ACC {
        for j in min_size..res_size {
            cast_slice_mut::<_, u64>(res.at_mut(res_col, j)).fill(0);
        }
    }
}

fn col_slice_u32(raw: &[Q120bScalar], n: usize, size: usize, col: usize) -> &[u32] {
    let stride = 8 * n * size;
    &cast_slice(raw)[col * stride..(col + 1) * stride]
}

// ──────────────────────────────────────────────────────────────────────────────
// Apply DFT entry points
// ──────────────────────────────────────────────────────────────────────────────

/// Compute the DFT-domain bivariate convolution `res[k] = Σ a[j] ⊙ b[k−j]`.
///
/// Output limbs `min_size..res.size()` are zeroed.
#[allow(clippy::too_many_arguments)]
pub fn ntt4x30_cnv_apply_dft<BE>(
    module: &(impl NttModuleHandle + Sync),
    cnv_offset: usize,
    res: &mut VecZnxDftBackendMut<'_, BE>,
    res_col: usize,
    a: &CnvPVecLBackendRef<'_, BE>,
    a_col: usize,
    b: &CnvPVecRBackendRef<'_, BE>,
    b_col: usize,
    tmp: &mut [u8],
) where
    BE: Backend<DftWord = Q120bScalar, ZnxWord = i64> + NttAddAssign + NttMulBbc1ColX2,
    for<'x> <BE as Backend>::BufRef<'x>: HostDataRef,
    for<'x> <BE as Backend>::BufMut<'x>: crate::layouts::HostDataMut,
{
    let n = res.n();
    let a_log_gap = sparse_log_gap(n, a.n());
    let b_log_gap = sparse_log_gap(n, b.n());
    let res_size = res.size();
    let a_size = a.size();
    let b_size = b.size();
    if res_size == 0 || a_size == 0 || b_size == 0 {
        for j in 0..res_size {
            cast_slice_mut::<_, u64>(res.at_mut(res_col, j)).fill(0);
        }
        return;
    }

    let a_col_u32 = col_slice_u32(a.raw(), a.n(), a_size, a_col);
    let b_col_u32 = col_slice_u32(b.raw(), b.n(), b_size, b_col);
    ntt4x30_conv_columns::<BE, false, false>(
        module, cnv_offset, res, res_col, a_col_u32, a_col_u32, a_size, a_log_gap, b_col_u32, b_col_u32, b_size, b_log_gap, tmp,
    );
}

/// Accumulating variant of [`ntt4x30_cnv_apply_dft`]: `res[k] += Σ a[j] ⊙ b[k−j]`
/// via the backend `ntt_add_assign` kernel (bit-identical to apply + DFT add).
/// Limbs `>= min_size` are left untouched.
#[allow(clippy::too_many_arguments)]
pub fn ntt4x30_cnv_apply_dft_add<BE>(
    module: &(impl NttModuleHandle + Sync),
    cnv_offset: usize,
    res: &mut VecZnxDftBackendMut<'_, BE>,
    res_col: usize,
    a: &CnvPVecLBackendRef<'_, BE>,
    a_col: usize,
    b: &CnvPVecRBackendRef<'_, BE>,
    b_col: usize,
    tmp: &mut [u8],
) where
    BE: Backend<DftWord = Q120bScalar, ZnxWord = i64> + NttAddAssign + NttMulBbc1ColX2,
    for<'x> <BE as Backend>::BufRef<'x>: HostDataRef,
    for<'x> <BE as Backend>::BufMut<'x>: crate::layouts::HostDataMut,
{
    let n = res.n();
    let a_log_gap = sparse_log_gap(n, a.n());
    let b_log_gap = sparse_log_gap(n, b.n());
    let res_size = res.size();
    let a_size = a.size();
    let b_size = b.size();
    if res_size == 0 || a_size == 0 || b_size == 0 {
        return;
    }

    let a_col_u32 = col_slice_u32(a.raw(), a.n(), a_size, a_col);
    let b_col_u32 = col_slice_u32(b.raw(), b.n(), b_size, b_col);
    ntt4x30_conv_columns::<BE, true, false>(
        module, cnv_offset, res, res_col, a_col_u32, a_col_u32, a_size, a_log_gap, b_col_u32, b_col_u32, b_size, b_log_gap, tmp,
    );
}

/// Scratch bytes required by [`ntt4x30_cnv_apply_dft_sum`]: the group staging.
pub fn ntt4x30_cnv_apply_dft_sum_tmp_bytes(res_size: usize, _a_size: usize, _b_size: usize) -> usize {
    8 * CNV_ACC_GROUP * res_size * size_of::<u64>()
}

/// One window contribution of one term to one output limb: `len` row pairs
/// starting at `a_row` (canonical left rows, ascending) and `b_row` (reversed
/// q120c rows, ascending).
pub struct CnvAccEntry {
    pub term: usize,
    pub a_row: usize,
    pub b_row: usize,
    pub len: usize,
}

/// Builds the per-output-limb window schedule of a fused convolution
/// accumulation. Entry windows are exact (no padding), so kernels read the
/// block-major operand rows in place. Returns `sched[k]` for `k ∈ 0..res_size`.
pub fn cnv_accumulate_schedule(cnv_offset: usize, res_size: usize, term_sizes: &[(usize, usize)]) -> Vec<Vec<CnvAccEntry>> {
    let mut sched: Vec<Vec<CnvAccEntry>> = (0..res_size).map(|_| Vec::new()).collect();
    for (t, &(a_size, b_size)) in term_sizes.iter().enumerate() {
        if a_size == 0 || b_size == 0 {
            continue;
        }
        let bound = a_size + b_size - 1;
        let offset = cnv_offset.min(bound);
        let min_size = res_size.min((bound + 1).saturating_sub(offset));
        for (k, sched_k) in sched.iter_mut().enumerate().take(min_size) {
            let k_abs = k + offset;
            let j_min = k_abs.saturating_sub(a_size - 1);
            let j_max = (k_abs + 1).min(b_size);
            // Iterating j from j_max-1 down to j_min walks both the `a` limb
            // rows (k_abs - j) and the reversed `b` rows (b_size - 1 - j)
            // ascending, so a single contiguous window covers the pair.
            sched_k.push(CnvAccEntry {
                term: t,
                a_row: k_abs + 1 - j_max,
                b_row: b_size - j_max,
                len: j_max - j_min,
            });
        }
    }
    // The q120 bbc reduction is designed for < 10 000 lazily accumulated rows.
    for sched_k in &sched {
        assert!(sched_k.iter().map(|e| e.len).sum::<usize>() < 10_000);
    }
    sched
}

/// Fused convolution accumulation: `res[res_col] = Σ_t a_t ⊛ b_t` (overwriting).
///
/// All terms of one output limb are summed in the lazy q120 accumulators and
/// reduced once, and the destination column is written exactly once through the
/// staged group flush — the result is congruent to, but not bit-identical with,
/// a sequence of [`ntt4x30_cnv_apply_dft_add`] calls.
pub fn ntt4x30_cnv_apply_dft_sum<BE>(
    module: &impl NttModuleHandle,
    cnv_offset: usize,
    res: &mut VecZnxDftBackendMut<'_, BE>,
    res_col: usize,
    terms: &[crate::layouts::CnvDftAccTerm<'_, BE>],
    tmp: &mut [u8],
) where
    BE: Backend<DftWord = Q120bScalar, ZnxWord = i64>,
    for<'x> <BE as Backend>::BufRef<'x>: HostDataRef,
    for<'x> <BE as Backend>::BufMut<'x>: crate::layouts::HostDataMut,
{
    use crate::reference::ntt4x30::mat_vec::{accum_mul_q120_bc, accum_to_q120b};

    let n = res.n();
    let res_size = res.size();

    #[allow(clippy::type_complexity)]
    let term_cols: Vec<(&[u32], &[u32], usize, usize, usize, usize)> = terms
        .iter()
        .map(|t| {
            let a_size = t.a.size();
            let b_size = t.b.size();
            (
                col_slice_u32(t.a.raw(), t.a.n(), a_size, t.a_col),
                col_slice_u32(t.b.raw(), t.b.n(), b_size, t.b_col),
                a_size,
                b_size,
                sparse_log_gap(n, t.a.n()),
                sparse_log_gap(n, t.b.n()),
            )
        })
        .collect();

    if res_size == 0 {
        return;
    }
    if terms.is_empty() {
        for j in 0..res_size {
            cast_slice_mut::<_, u64>(res.at_mut(res_col, j)).fill(0);
        }
        return;
    }

    let meta = module.get_bbc_meta();
    let n_blks = n / 2;

    let sched = cnv_accumulate_schedule(
        cnv_offset,
        res_size,
        &term_cols.iter().map(|&(_, _, a, b, _, _)| (a, b)).collect::<Vec<_>>(),
    );

    let (prefix, tmp_u64, suffix) = unsafe { tmp.align_to_mut::<u64>() };
    assert!(prefix.is_empty());
    assert!(suffix.is_empty());
    let stage = &mut tmp_u64[..8 * CNV_ACC_GROUP * res_size];
    let stage4: &mut [[u64; 4]] = cast_slice_mut(stage);

    // Gather buffers of a sparse term's x2 rows, allocated once (a dense term
    // reads its rows in place and never touches them).
    let (mut xb, mut yb) = ([[0u32; 8]; 2], [[0u32; 8]; 2]);

    for blk in 0..n_blks {
        let grp_pos = blk % CNV_ACC_GROUP;

        for (k, sched_k) in sched.iter().enumerate() {
            let mut s = [[0u64; 8]; 2];
            for e in sched_k {
                let (a_col, b_col, a_size, b_size, a_log_gap, b_log_gap) = term_cols[e.term];
                for i in 0..e.len {
                    let x = x2_row(a_col, a_size, blk, e.a_row + i, a_log_gap, &mut xb);
                    let y = x2_row(b_col, b_size, blk, e.b_row + i, b_log_gap, &mut yb);
                    accum_mul_q120_bc(&mut s[0], &x[0], &y[0]);
                    accum_mul_q120_bc(&mut s[1], &x[1], &y[1]);
                }
            }
            let o = 2 * (k * CNV_ACC_GROUP + grp_pos);
            accum_to_q120b::<Primes30>(&mut stage4[o], &s[0], meta);
            accum_to_q120b::<Primes30>(&mut stage4[o + 1], &s[1], meta);
        }

        // Flush the group per limb as one contiguous run.
        let in_group = grp_pos + 1;
        if in_group == CNV_ACC_GROUP || blk == n_blks - 1 {
            let grp_base = blk + 1 - in_group;
            for k in 0..res_size {
                let res_u64: &mut [u64] = cast_slice_mut(res.at_mut(res_col, k));
                let run: &[u64] = cast_slice(&stage4[2 * k * CNV_ACC_GROUP..2 * (k * CNV_ACC_GROUP + in_group)]);
                res_u64[8 * grp_base..8 * (grp_base + in_group)].copy_from_slice(run);
            }
        }
    }
}

/// Compute the pairwise DFT-domain convolution
/// `res = (a[:,i] + a[:,j]) ⊙ (b[:,i] + b[:,j])`.
///
/// When `col_i == col_j` this delegates to [`ntt4x30_cnv_apply_dft`].
#[allow(clippy::too_many_arguments)]
pub fn ntt4x30_cnv_pairwise_apply_dft<BE>(
    module: &(impl NttModuleHandle + Sync),
    cnv_offset: usize,
    res: &mut VecZnxDftBackendMut<'_, BE>,
    res_col: usize,
    a: &CnvPVecLBackendRef<'_, BE>,
    b: &CnvPVecRBackendRef<'_, BE>,
    col_i: usize,
    col_j: usize,
    tmp: &mut [u8],
) where
    BE: Backend<DftWord = Q120bScalar, ZnxWord = i64> + NttAddAssign + NttMulBbc1ColX2,
    for<'x> <BE as Backend>::BufRef<'x>: HostDataRef,
    for<'x> <BE as Backend>::BufMut<'x>: crate::layouts::HostDataMut,
{
    if col_i == col_j {
        ntt4x30_cnv_apply_dft::<BE>(module, cnv_offset, res, res_col, a, col_i, b, col_j, tmp);
        return;
    }

    let n = res.n();
    let a_log_gap = sparse_log_gap(n, a.n());
    let b_log_gap = sparse_log_gap(n, b.n());
    let res_size = res.size();
    let a_size = a.size();
    let b_size = b.size();
    if res_size == 0 || a_size == 0 || b_size == 0 {
        for j in 0..res_size {
            cast_slice_mut::<_, u64>(res.at_mut(res_col, j)).fill(0);
        }
        return;
    }

    let a0 = col_slice_u32(a.raw(), a.n(), a_size, col_i);
    let a1 = col_slice_u32(a.raw(), a.n(), a_size, col_j);
    let b0 = col_slice_u32(b.raw(), b.n(), b_size, col_i);
    let b1 = col_slice_u32(b.raw(), b.n(), b_size, col_j);
    ntt4x30_conv_columns::<BE, false, true>(
        module, cnv_offset, res, res_col, a0, a1, a_size, a_log_gap, b0, b1, b_size, b_log_gap, tmp,
    );
}

// ──────────────────────────────────────────────────────────────────────────────
// Prepare paths
// ──────────────────────────────────────────────────────────────────────────────

fn zero_row_u32(dst: &mut [u32], size: usize, row: usize, n_blks: usize) {
    for blk in 0..n_blks {
        let off = (blk * size + row) * 16;
        dst[off..off + 16].fill(0);
    }
}

/// Scratch bytes required by [`ntt4x30_cnv_prepare_left`]: NTT and canonical limbs.
pub fn ntt4x30_cnv_prepare_left_tmp_bytes(n: usize) -> usize {
    8 * n * size_of::<u64>()
}

/// Encode a `VecZnx` into a `CnvPVecL` (canonical u32 rows, block-major).
///
/// Limbs of `res` beyond `a.size()` are zeroed.
pub fn ntt4x30_cnv_prepare_left<BE>(
    module: &impl NttModuleHandle,
    res: &mut CnvPVecLBackendMut<'_, BE>,
    a: &VecZnxBackendRef<'_, BE>,
    tmp: &mut [u8],
) where
    BE: Backend<DftWord = Q120bScalar, ZnxWord = i64>
        + NttFromZnx64
        + NttDFTExecute<NttTable<Primes30>>
        + NttPackLeft1BlkX2
        + 'static,
    for<'x> BE: Backend<BufRef<'x> = &'x [u8], BufMut<'x> = &'x mut [u8], ZnxWord = i64>,
{
    poulpy_hal::layouts::assert_dense(a, "ntt4x30_cnv_prepare_left");
    let n = res.n();
    assert_sparse_degree(module.n(), n);
    assert_eq!(a.n(), n, "a.n():{} != res.n():{n}", a.n());
    let table = module.get_ntt_table_for(n);
    let cols = res.cols();
    assert_eq!(a.cols(), cols, "a.cols():{} != res.cols():{cols}", a.cols());
    let res_size = res.size();
    let min_size = res_size.min(a.size());
    let n_blks = n / 2;
    let col_stride = 8 * n * res_size;

    let (prefix, tmp_u64, suffix) = unsafe { tmp.align_to_mut::<u64>() };
    assert!(prefix.is_empty());
    assert!(suffix.is_empty());
    let res_u32: &mut [u32] = cast_slice_mut(res.raw_mut());
    if BE::TaskExecutor::is_parallel() && cols * res_size > 1 {
        let res_addr = res_u32.as_mut_ptr() as usize;
        BE::TaskExecutor::for_each_chunked(cols * res_size, tmp_u64, 8 * n, |task_tmp, task| {
            let col = task / res_size;
            let j = task % res_size;
            let (limb, canon_u64) = task_tmp.split_at_mut(4 * n);
            let canon: &mut [u32] = cast_slice_mut(canon_u64);
            if j < min_size {
                BE::ntt_from_znx64(limb, a.at(col, j));
                BE::ntt_dft_execute(table, limb);
            }
            for g in (0..n_blks).step_by(PREP_GROUP) {
                let gl = PREP_GROUP.min(n_blks - g);
                if j < min_size {
                    BE::ntt_pack_left_1blk_x2(&mut canon[..16 * gl], &limb[8 * g..], gl, 8, 0);
                }
                for i in 0..gl {
                    let off = col * col_stride + ((g + i) * res_size + j) * 16;
                    let dst = unsafe { std::slice::from_raw_parts_mut((res_addr as *mut u32).add(off), 16) };
                    if j < min_size {
                        dst.copy_from_slice(&canon[16 * i..16 * (i + 1)]);
                    } else {
                        dst.fill(0);
                    }
                }
            }
        });
        return;
    }
    let (limb, canon_u64) = tmp_u64[..8 * n].split_at_mut(4 * n);
    let canon: &mut [u32] = cast_slice_mut(canon_u64);
    for col in 0..cols {
        let dst = &mut res_u32[col * col_stride..(col + 1) * col_stride];
        for j in 0..min_size {
            BE::ntt_from_znx64(limb, a.at(col, j));
            BE::ntt_dft_execute(table, limb);
            // Canonicalize and scatter per block group so the staging chunk
            // stays L1-resident.
            for g in (0..n_blks).step_by(PREP_GROUP) {
                let gl = PREP_GROUP.min(n_blks - g);
                BE::ntt_pack_left_1blk_x2(&mut canon[..16 * gl], &limb[8 * g..], gl, 8, 0);
                for (i, chunk) in canon[..16 * gl].chunks_exact(16).enumerate() {
                    let off = ((g + i) * res_size + j) * 16;
                    dst[off..off + 16].copy_from_slice(chunk);
                }
            }
        }
        for j in min_size..res_size {
            zero_row_u32(dst, res_size, j, n_blks);
        }
    }
}

/// Scratch bytes required by [`ntt4x30_cnv_prepare_right`]: NTT and converted limbs.
pub fn ntt4x30_cnv_prepare_right_tmp_bytes(n: usize) -> usize {
    8 * n * size_of::<u64>()
}

/// Encode a `VecZnx` into a `CnvPVecR` (q120c rows, block-major, reversed
/// limb order). Limbs of `res` beyond `a.size()` are zeroed.
pub fn ntt4x30_cnv_prepare_right<BE>(
    module: &impl NttModuleHandle,
    res: &mut CnvPVecRBackendMut<'_, BE>,
    a: &VecZnxBackendRef<'_, BE>,
    tmp: &mut [u64],
) where
    BE: Backend<DftWord = Q120bScalar, ZnxWord = i64> + NttFromZnx64 + NttDFTExecute<NttTable<Primes30>> + NttCFromB + 'static,
    for<'x> BE: Backend<BufRef<'x> = &'x [u8], BufMut<'x> = &'x mut [u8], ZnxWord = i64>,
{
    poulpy_hal::layouts::assert_dense(a, "ntt4x30_cnv_prepare_right");
    let n = res.n();
    assert_sparse_degree(module.n(), n);
    assert_eq!(a.n(), n, "a.n():{} != res.n():{n}", a.n());
    let table = module.get_ntt_table_for(n);
    let cols = res.cols();
    assert_eq!(a.cols(), cols, "a.cols():{} != res.cols():{cols}", a.cols());
    let res_size = res.size();
    let min_size = res_size.min(a.size());
    let n_blks = n / 2;
    let col_stride = 8 * n * res_size;

    let res_u32: &mut [u32] = cast_slice_mut(res.raw_mut());
    if BE::TaskExecutor::is_parallel() && cols * res_size > 1 {
        let res_addr = res_u32.as_mut_ptr() as usize;
        BE::TaskExecutor::for_each_chunked(cols * res_size, tmp, 8 * n, |task_tmp, task| {
            let col = task / res_size;
            let j = task % res_size;
            let (limb_b, limb_c_u64) = task_tmp.split_at_mut(4 * n);
            let limb_c: &mut [u32] = cast_slice_mut(limb_c_u64);
            if j < min_size {
                BE::ntt_from_znx64(limb_b, a.at(col, j));
                BE::ntt_dft_execute(table, limb_b);
                BE::ntt_c_from_b(n, limb_c, limb_b);
            }
            let row = res_size - 1 - j;
            for blk in 0..n_blks {
                let off = col * col_stride + (blk * res_size + row) * 16;
                let dst = unsafe { std::slice::from_raw_parts_mut((res_addr as *mut u32).add(off), 16) };
                if j < min_size {
                    dst.copy_from_slice(&limb_c[16 * blk..16 * (blk + 1)]);
                } else {
                    dst.fill(0);
                }
            }
        });
        return;
    }
    let (limb_b, limb_c_u64) = tmp[..8 * n].split_at_mut(4 * n);
    let limb_c: &mut [u32] = cast_slice_mut(limb_c_u64);
    for col in 0..cols {
        let dst = &mut res_u32[col * col_stride..(col + 1) * col_stride];
        for j in 0..min_size {
            BE::ntt_from_znx64(limb_b, a.at(col, j));
            BE::ntt_dft_execute(table, limb_b);
            BE::ntt_c_from_b(n, limb_c, limb_b);
            // Reversed row order: limb j lands on row size-1-j.
            let row = res_size - 1 - j;
            for blk in 0..n_blks {
                let off = (blk * res_size + row) * 16;
                dst[off..off + 16].copy_from_slice(&limb_c[16 * blk..16 * blk + 16]);
            }
        }
        for j in min_size..res_size {
            zero_row_u32(dst, res_size, res_size - 1 - j, n_blks);
        }
    }
}

/// Scratch bytes required by [`ntt4x30_cnv_prepare_self`]: NTT, canonical and
/// converted limbs.
pub fn ntt4x30_cnv_prepare_self_tmp_bytes(n: usize) -> usize {
    12 * n * size_of::<u64>()
}

/// Encode a `VecZnx` into both `CnvPVecL` and `CnvPVecR` sharing the NTT.
pub fn ntt4x30_cnv_prepare_self<BE>(
    module: &impl NttModuleHandle,
    left: &mut CnvPVecLBackendMut<'_, BE>,
    right: &mut CnvPVecRBackendMut<'_, BE>,
    a: &VecZnxBackendRef<'_, BE>,
    tmp: &mut [u8],
) where
    BE: Backend<DftWord = Q120bScalar, ZnxWord = i64>
        + NttFromZnx64
        + NttDFTExecute<NttTable<Primes30>>
        + NttCFromB
        + NttPackLeft1BlkX2
        + 'static,
    for<'x> BE: Backend<BufRef<'x> = &'x [u8], BufMut<'x> = &'x mut [u8], ZnxWord = i64>,
{
    poulpy_hal::layouts::assert_dense(a, "ntt4x30_cnv_prepare_self");
    let n = left.n();
    assert_sparse_degree(module.n(), n);
    assert_eq!(a.n(), n, "ntt4x30_cnv_prepare_self: a.n():{} != left.n():{n}", a.n());
    assert_eq!(
        right.n(),
        n,
        "ntt4x30_cnv_prepare_self: right.n():{} != left.n():{n}",
        right.n()
    );
    let table = module.get_ntt_table_for(n);
    let cols = left.cols();
    assert_eq!(a.cols(), cols, "a.cols():{} != left.cols():{cols}", a.cols());
    assert_eq!(right.cols(), cols, "right.cols():{} != left.cols():{cols}", right.cols());
    let res_size = left.size();
    assert_eq!(
        right.size(),
        res_size,
        "right.size():{} != left.size():{res_size}",
        right.size()
    );
    let min_size = res_size.min(a.size());
    let n_blks = n / 2;
    let col_stride = 8 * n * res_size;

    let (prefix, tmp_u64, suffix) = unsafe { tmp.align_to_mut::<u64>() };
    assert!(prefix.is_empty());
    assert!(suffix.is_empty());
    let left_u32: &mut [u32] = cast_slice_mut(left.raw_mut());
    let right_u32: &mut [u32] = cast_slice_mut(right.raw_mut());
    if BE::TaskExecutor::is_parallel() && cols * res_size > 1 {
        let left_addr = left_u32.as_mut_ptr() as usize;
        let right_addr = right_u32.as_mut_ptr() as usize;
        BE::TaskExecutor::for_each_chunked(cols * res_size, tmp_u64, 12 * n, |task_tmp, task| {
            let col = task / res_size;
            let j = task % res_size;
            let (limb_b, rest) = task_tmp.split_at_mut(4 * n);
            let (canon_u64, limb_c_u64) = rest.split_at_mut(4 * n);
            let canon: &mut [u32] = cast_slice_mut(canon_u64);
            let limb_c: &mut [u32] = cast_slice_mut(limb_c_u64);
            if j < min_size {
                BE::ntt_from_znx64(limb_b, a.at(col, j));
                BE::ntt_dft_execute(table, limb_b);
                BE::ntt_c_from_b(n, limb_c, limb_b);
            }
            for g in (0..n_blks).step_by(PREP_GROUP) {
                let gl = PREP_GROUP.min(n_blks - g);
                if j < min_size {
                    BE::ntt_pack_left_1blk_x2(&mut canon[..16 * gl], &limb_b[8 * g..], gl, 8, 0);
                }
                for i in 0..gl {
                    let left_off = col * col_stride + ((g + i) * res_size + j) * 16;
                    let left_dst = unsafe { std::slice::from_raw_parts_mut((left_addr as *mut u32).add(left_off), 16) };
                    if j < min_size {
                        left_dst.copy_from_slice(&canon[16 * i..16 * (i + 1)]);
                    } else {
                        left_dst.fill(0);
                    }
                }
            }
            let row = res_size - 1 - j;
            for blk in 0..n_blks {
                let right_off = col * col_stride + (blk * res_size + row) * 16;
                let right_dst = unsafe { std::slice::from_raw_parts_mut((right_addr as *mut u32).add(right_off), 16) };
                if j < min_size {
                    right_dst.copy_from_slice(&limb_c[16 * blk..16 * (blk + 1)]);
                } else {
                    right_dst.fill(0);
                }
            }
        });
        return;
    }
    let (limb_b, rest) = tmp_u64[..12 * n].split_at_mut(4 * n);
    let (canon_u64, limb_c_u64) = rest.split_at_mut(4 * n);
    let canon: &mut [u32] = cast_slice_mut(canon_u64);
    let limb_c: &mut [u32] = cast_slice_mut(limb_c_u64);
    for col in 0..cols {
        let dst_l = &mut left_u32[col * col_stride..(col + 1) * col_stride];
        let dst_r = &mut right_u32[col * col_stride..(col + 1) * col_stride];
        for j in 0..min_size {
            BE::ntt_from_znx64(limb_b, a.at(col, j));
            BE::ntt_dft_execute(table, limb_b);
            for g in (0..n_blks).step_by(PREP_GROUP) {
                let gl = PREP_GROUP.min(n_blks - g);
                BE::ntt_pack_left_1blk_x2(&mut canon[..16 * gl], &limb_b[8 * g..], gl, 8, 0);
                for (i, chunk) in canon[..16 * gl].chunks_exact(16).enumerate() {
                    let off = ((g + i) * res_size + j) * 16;
                    dst_l[off..off + 16].copy_from_slice(chunk);
                }
            }
            BE::ntt_c_from_b(n, limb_c, limb_b);
            let row = res_size - 1 - j;
            for blk in 0..n_blks {
                let off = (blk * res_size + row) * 16;
                dst_r[off..off + 16].copy_from_slice(&limb_c[16 * blk..16 * blk + 16]);
            }
        }
        for j in min_size..res_size {
            zero_row_u32(dst_l, res_size, j, n_blks);
            zero_row_u32(dst_r, res_size, res_size - 1 - j, n_blks);
        }
    }
}

// ──────────────────────────────────────────────────────────────────────────────
// By-const apply  (VecZnx × &[i64] → VecZnxBig, coefficient domain)
// ──────────────────────────────────────────────────────────────────────────────

/// Scratch bytes required by [`ntt4x30_cnv_by_const_apply`].
pub fn ntt4x30_cnv_by_const_apply_tmp_bytes(_res_size: usize, _a_size: usize, _b_size: usize) -> usize {
    0
}

/// Coefficient-domain convolution by a constant: `res[k] = Σ_j a[k_abs − j] · b[j][b_coeff]`.
///
/// Shared by every backend with `BigWord = i128`: each output limb is an
/// `i128` accumulation over the `a` limbs, taken one slice at a time (no
/// per-coefficient accessor calls, no checks inside the coefficient loop).
/// Output limbs `min_size..res.size()` are zeroed by `ntt4x30_cnv_by_const_apply`
/// and left untouched by `ntt4x30_cnv_by_const_apply_add`. `_tmp` is unused.
#[allow(clippy::too_many_arguments)]
pub fn ntt4x30_cnv_by_const_apply<BE, E: TaskExecutor>(
    cnv_offset: usize,
    res: &mut VecZnxBigBackendMut<'_, BE>,
    res_col: usize,
    a: &VecZnxBackendRef<'_, BE>,
    a_col: usize,
    b: &VecZnxBackendRef<'_, BE>,
    b_col: usize,
    b_coeff: usize,
    _tmp: &mut [u8],
) where
    BE: Backend<BigWord = i128, ZnxWord = i64>,
    for<'x> BE::BufRef<'x>: HostDataRef,
    for<'x> BE::BufMut<'x>: HostDataMut,
{
    ntt4x30_cnv_by_const_apply_impl::<BE, E, false>(cnv_offset, res, res_col, a, a_col, b, b_col, b_coeff)
}

#[allow(clippy::too_many_arguments)]
pub fn ntt4x30_cnv_by_const_apply_add<BE, E: TaskExecutor>(
    cnv_offset: usize,
    res: &mut VecZnxBigBackendMut<'_, BE>,
    res_col: usize,
    a: &VecZnxBackendRef<'_, BE>,
    a_col: usize,
    b: &VecZnxBackendRef<'_, BE>,
    b_col: usize,
    b_coeff: usize,
    _tmp: &mut [u8],
) where
    BE: Backend<BigWord = i128, ZnxWord = i64>,
    for<'x> BE::BufRef<'x>: HostDataRef,
    for<'x> BE::BufMut<'x>: HostDataMut,
{
    ntt4x30_cnv_by_const_apply_impl::<BE, E, true>(cnv_offset, res, res_col, a, a_col, b, b_col, b_coeff)
}

#[allow(clippy::too_many_arguments)]
fn ntt4x30_cnv_by_const_apply_impl<BE, E: TaskExecutor, const ADD: bool>(
    cnv_offset: usize,
    res: &mut VecZnxBigBackendMut<'_, BE>,
    res_col: usize,
    a: &VecZnxBackendRef<'_, BE>,
    a_col: usize,
    b: &VecZnxBackendRef<'_, BE>,
    b_col: usize,
    b_coeff: usize,
) where
    BE: Backend<BigWord = i128, ZnxWord = i64>,
    for<'x> BE::BufRef<'x>: HostDataRef,
    for<'x> BE::BufMut<'x>: HostDataMut,
{
    poulpy_hal::layouts::assert_dense(res, "ntt4x30_cnv_by_const_apply_impl");
    poulpy_hal::layouts::assert_dense(a, "ntt4x30_cnv_by_const_apply_impl");
    poulpy_hal::layouts::assert_dense(b, "ntt4x30_cnv_by_const_apply_impl");
    let (res_size, a_size, b_size) = (res.size(), a.size(), b.size());
    let n = res.n();
    let res_cols = res.cols();
    assert!(
        res_col < res_cols,
        "ntt4x30_cnv_by_const_apply_impl: res_col {res_col} >= cols {res_cols}"
    );
    assert!(a.n() == n, "ntt4x30_cnv_by_const_apply_impl: a.n() {} != res.n() {n}", a.n());
    debug_assert!(
        b_coeff < b.n(),
        "ntt4x30_cnv_by_const_apply_impl: b_coeff {b_coeff} >= b.n() {}",
        b.n()
    );
    let res_ptr = crate::reference::SendPtr::new(res.raw_mut().as_mut_ptr());
    let res_limb = |k: usize| unsafe { std::slice::from_raw_parts_mut(res_ptr.get().add(n * (k * res_cols + res_col)), n) };

    if res_size == 0 || a_size == 0 || b_size == 0 {
        if !ADD {
            for k in 0..res_size {
                res_limb(k).fill(0);
            }
        }
        return;
    }

    let bound = a_size + b_size - 1;
    let offset = cnv_offset.min(bound);
    // Every k < min_size has k_abs < bound, hence a non-empty j range.
    let min_size = res_size.min(bound - offset);

    let process = |k: usize| {
        let out = res_limb(k);
        if k >= min_size {
            if !ADD {
                out.fill(0);
            }
            return;
        }
        let k_abs = k + offset;
        let j_min = k_abs.saturating_sub(a_size - 1);
        let j_max = (k_abs + 1).min(b_size);
        for j in j_min..j_max {
            let a_limb: &[i64] = a.at(a_col, k_abs - j);
            let b_j = b.at(b_col, j)[b_coeff] as i128;
            if !ADD && j == j_min {
                for (r, &x) in out.iter_mut().zip(a_limb) {
                    *r = x as i128 * b_j;
                }
            } else {
                for (r, &x) in out.iter_mut().zip(a_limb) {
                    *r = r.wrapping_add(x as i128 * b_j);
                }
            }
        }
    };

    if E::IS_PARALLEL {
        E::for_each(res_size, process);
    } else {
        for k in 0..res_size {
            process(k);
        }
    }
}
