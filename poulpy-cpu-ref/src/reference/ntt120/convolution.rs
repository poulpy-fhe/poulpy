//! Bivariate convolution operations for the NTT120 backend family.
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

use crate::{
    layouts::{
        Backend, CnvPVecLBackendMut, CnvPVecLBackendRef, CnvPVecRBackendMut, CnvPVecRBackendRef, HostDataRef, VecZnxBackendRef,
        VecZnxBigBackendMut, VecZnxDftBackendMut, ZnxView, ZnxViewMut,
    },
    reference::ntt120::{
        NttAddAssign, NttCFromB, NttDFTExecute, NttFromZnx64, NttMulBbc1ColX2, NttPackLeft1BlkX2, NttPackRight1BlkX2,
        ntt::NttTable,
        primes::{PrimeSet, Primes30},
        types::Q120bScalar,
        vec_znx_dft::NttModuleHandle,
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

/// Scratch bytes required by [`ntt120_cnv_apply_dft`] and its accumulate
/// variant: the padded `a` window plus the accumulate staging group.
pub fn ntt120_cnv_apply_dft_tmp_bytes(res_size: usize, a_size: usize, b_size: usize) -> usize {
    let min_size: usize = res_size.min(a_size + b_size);
    16 * (a_size + 2 * (TILE - 1)) * size_of::<u32>() + 8 * CNV_ACC_GROUP * min_size * size_of::<u64>()
}

/// Scratch bytes required by [`ntt120_cnv_pairwise_apply_dft`]: the apply
/// scratch plus the summed `b` rows.
pub fn ntt120_cnv_pairwise_apply_dft_tmp_bytes(res_size: usize, a_size: usize, b_size: usize) -> usize {
    if a_size == 0 || b_size == 0 || res_size == 0 {
        0
    } else {
        ntt120_cnv_apply_dft_tmp_bytes(res_size, a_size, b_size) + 16 * b_size * size_of::<u32>()
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

/// Convolve one column pair into `res[res_col]`, tiling [`TILE`] output limbs
/// per pass over the zero-padded `a` window.
///
/// - `ACC`: accumulate into `res` (via group-staged `ntt_add_assign`) instead
///   of overwriting.
/// - `PAIRWISE`: operands are `(a0 + a1) mod q` and the lazy sum `b0 + b1`.
#[allow(clippy::too_many_arguments)]
fn ntt120_conv_columns<BE, const ACC: bool, const PAIRWISE: bool>(
    module: &impl NttModuleHandle,
    cnv_offset: usize,
    res: &mut VecZnxDftBackendMut<'_, BE>,
    res_col: usize,
    a0_col: &[u32],
    a1_col: &[u32],
    a_size: usize,
    b0_col: &[u32],
    b1_col: &[u32],
    b_size: usize,
    tmp: &mut [u8],
) where
    BE: Backend<ScalarPrep = Q120bScalar> + NttAddAssign + NttMulBbc1ColX2,
    for<'x> <BE as Backend>::BufRef<'x>: HostDataRef,
    for<'x> <BE as Backend>::BufMut<'x>: crate::layouts::HostDataMut,
{
    let n = res.n();
    let res_size = res.size();

    let bound = a_size + b_size - 1;
    let offset = cnv_offset.min(bound);
    let min_size = res_size.min((bound + 1).saturating_sub(offset));

    let meta = module.get_bbc_meta();
    let n_blks = n / 2;
    let pad = TILE - 1;
    let win_rows = a_size + 2 * pad;

    let (prefix, tmp_u64, suffix) = unsafe { tmp.align_to_mut::<u64>() };
    debug_assert!(prefix.is_empty());
    debug_assert!(suffix.is_empty());
    let (stage, rest) = tmp_u64.split_at_mut(8 * CNV_ACC_GROUP * min_size);
    let rest_u32: &mut [u32] = cast_slice_mut(rest);
    let (win, rest_u32) = rest_u32.split_at_mut(16 * win_rows);
    let b_sum: &mut [u32] = &mut rest_u32[..if PAIRWISE { 16 * b_size } else { 0 }];

    win[..16 * pad].fill(0);
    win[16 * (a_size + pad)..].fill(0);

    let n_tiles = min_size.div_ceil(TILE);
    let mut out = [0u64; 8 * TILE];

    for blk in 0..n_blks {
        // Stage this block's a rows (or the canonical pairwise sum) into the
        // padded window; b rows are read in place (lazy-summed when PAIRWISE).
        let a_blk = &a0_col[blk * 16 * a_size..(blk + 1) * 16 * a_size];
        if PAIRWISE {
            let a1_blk = &a1_col[blk * 16 * a_size..(blk + 1) * 16 * a_size];
            for r in 0..a_size {
                canonical_sum_row(
                    &mut win[16 * (pad + r)..16 * (pad + r + 1)],
                    &a_blk[16 * r..],
                    &a1_blk[16 * r..],
                );
            }
        } else {
            win[16 * pad..16 * (pad + a_size)].copy_from_slice(a_blk);
        }
        let b_blk: &[u32] = if PAIRWISE {
            let b0_blk = &b0_col[blk * 16 * b_size..(blk + 1) * 16 * b_size];
            let b1_blk = &b1_col[blk * 16 * b_size..(blk + 1) * 16 * b_size];
            for (d, (x, y)) in b_sum.iter_mut().zip(b0_blk.iter().zip(b1_blk.iter())) {
                *d = x + y;
            }
            b_sum
        } else {
            &b0_col[blk * 16 * b_size..(blk + 1) * 16 * b_size]
        };

        let grp_pos = blk % CNV_ACC_GROUP;

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
        let in_group = grp_pos + 1;
        if in_group == CNV_ACC_GROUP || blk == n_blks - 1 {
            let grp_base = blk + 1 - in_group;
            for k in 0..min_size {
                let res_u64: &mut [u64] = cast_slice_mut(res.at_mut(res_col, k));
                let run = &stage[8 * k * CNV_ACC_GROUP..8 * (k * CNV_ACC_GROUP + in_group)];
                if ACC {
                    BE::ntt_add_assign(&mut res_u64[8 * grp_base..8 * (grp_base + in_group)], run);
                } else {
                    res_u64[8 * grp_base..8 * (grp_base + in_group)].copy_from_slice(run);
                }
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
pub fn ntt120_cnv_apply_dft<BE>(
    module: &impl NttModuleHandle,
    cnv_offset: usize,
    res: &mut VecZnxDftBackendMut<'_, BE>,
    res_col: usize,
    a: &CnvPVecLBackendRef<'_, BE>,
    a_col: usize,
    b: &CnvPVecRBackendRef<'_, BE>,
    b_col: usize,
    tmp: &mut [u8],
) where
    BE: Backend<ScalarPrep = Q120bScalar> + NttAddAssign + NttMulBbc1ColX2,
    for<'x> <BE as Backend>::BufRef<'x>: HostDataRef,
    for<'x> <BE as Backend>::BufMut<'x>: crate::layouts::HostDataMut,
{
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

    let a_col_u32 = col_slice_u32(a.raw(), n, a_size, a_col);
    let b_col_u32 = col_slice_u32(b.raw(), n, b_size, b_col);
    ntt120_conv_columns::<BE, false, false>(
        module, cnv_offset, res, res_col, a_col_u32, a_col_u32, a_size, b_col_u32, b_col_u32, b_size, tmp,
    );
}

/// Accumulating variant of [`ntt120_cnv_apply_dft`]: `res[k] += Σ a[j] ⊙ b[k−j]`
/// via the backend `ntt_add_assign` kernel (bit-identical to apply + DFT add).
/// Limbs `>= min_size` are left untouched.
#[allow(clippy::too_many_arguments)]
pub fn ntt120_cnv_apply_dft_accumulate<BE>(
    module: &impl NttModuleHandle,
    cnv_offset: usize,
    res: &mut VecZnxDftBackendMut<'_, BE>,
    res_col: usize,
    a: &CnvPVecLBackendRef<'_, BE>,
    a_col: usize,
    b: &CnvPVecRBackendRef<'_, BE>,
    b_col: usize,
    tmp: &mut [u8],
) where
    BE: Backend<ScalarPrep = Q120bScalar> + NttAddAssign + NttMulBbc1ColX2,
    for<'x> <BE as Backend>::BufRef<'x>: HostDataRef,
    for<'x> <BE as Backend>::BufMut<'x>: crate::layouts::HostDataMut,
{
    let n = res.n();
    let res_size = res.size();
    let a_size = a.size();
    let b_size = b.size();
    if res_size == 0 || a_size == 0 || b_size == 0 {
        return;
    }

    let a_col_u32 = col_slice_u32(a.raw(), n, a_size, a_col);
    let b_col_u32 = col_slice_u32(b.raw(), n, b_size, b_col);
    ntt120_conv_columns::<BE, true, false>(
        module, cnv_offset, res, res_col, a_col_u32, a_col_u32, a_size, b_col_u32, b_col_u32, b_size, tmp,
    );
}

/// Scratch bytes required by [`ntt120_cnv_accumulate_dft`]: the group staging.
pub fn ntt120_cnv_accumulate_dft_tmp_bytes(res_size: usize, _a_size: usize, _b_size: usize) -> usize {
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
        debug_assert!(sched_k.iter().map(|e| e.len).sum::<usize>() < 10_000);
    }
    sched
}

/// Fused convolution accumulation: `res[res_col] = Σ_t a_t ⊛ b_t` (overwriting).
///
/// All terms of one output limb are summed in the lazy q120 accumulators and
/// reduced once, and the destination column is written exactly once through the
/// staged group flush — the result is congruent to, but not bit-identical with,
/// a sequence of [`ntt120_cnv_apply_dft_accumulate`] calls.
pub fn ntt120_cnv_accumulate_dft<BE>(
    module: &impl NttModuleHandle,
    cnv_offset: usize,
    res: &mut VecZnxDftBackendMut<'_, BE>,
    res_col: usize,
    terms: &[crate::layouts::CnvDftAccTerm<'_, BE>],
    tmp: &mut [u8],
) where
    BE: Backend<ScalarPrep = Q120bScalar>,
    for<'x> <BE as Backend>::BufRef<'x>: HostDataRef,
    for<'x> <BE as Backend>::BufMut<'x>: crate::layouts::HostDataMut,
{
    use crate::reference::ntt120::mat_vec::{accum_mul_q120_bc, accum_to_q120b};

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

    let meta = module.get_bbc_meta();
    let n_blks = n / 2;

    let term_cols: Vec<(&[u32], &[u32], usize, usize)> = terms
        .iter()
        .map(|t| {
            let a_size = t.a.size();
            let b_size = t.b.size();
            (
                col_slice_u32(t.a.raw(), n, a_size, t.a_col),
                col_slice_u32(t.b.raw(), n, b_size, t.b_col),
                a_size,
                b_size,
            )
        })
        .collect();
    let sched = cnv_accumulate_schedule(
        cnv_offset,
        res_size,
        &term_cols.iter().map(|&(_, _, a, b)| (a, b)).collect::<Vec<_>>(),
    );

    let (prefix, tmp_u64, suffix) = unsafe { tmp.align_to_mut::<u64>() };
    debug_assert!(prefix.is_empty());
    debug_assert!(suffix.is_empty());
    let stage = &mut tmp_u64[..8 * CNV_ACC_GROUP * res_size];

    for blk in 0..n_blks {
        let grp_pos = blk % CNV_ACC_GROUP;

        for (k, sched_k) in sched.iter().enumerate() {
            let mut s = [[0u64; 8]; 2];
            for e in sched_k {
                let (a_col, b_col, a_size, b_size) = term_cols[e.term];
                let a_blk = &a_col[blk * 16 * a_size..];
                let b_blk = &b_col[blk * 16 * b_size..];
                for i in 0..e.len {
                    let x = &a_blk[16 * (e.a_row + i)..16 * (e.a_row + i) + 16];
                    let y = &b_blk[16 * (e.b_row + i)..16 * (e.b_row + i) + 16];
                    accum_mul_q120_bc(&mut s[0], x[..8].try_into().unwrap(), y[..8].try_into().unwrap());
                    accum_mul_q120_bc(&mut s[1], x[8..].try_into().unwrap(), y[8..].try_into().unwrap());
                }
            }
            let out = &mut stage[8 * (k * CNV_ACC_GROUP + grp_pos)..];
            accum_to_q120b::<Primes30>((&mut out[..4]).try_into().unwrap(), &s[0], meta);
            accum_to_q120b::<Primes30>((&mut out[4..8]).try_into().unwrap(), &s[1], meta);
        }

        // Flush the group per limb as one contiguous run.
        let in_group = grp_pos + 1;
        if in_group == CNV_ACC_GROUP || blk == n_blks - 1 {
            let grp_base = blk + 1 - in_group;
            for k in 0..res_size {
                let res_u64: &mut [u64] = cast_slice_mut(res.at_mut(res_col, k));
                res_u64[8 * grp_base..8 * (grp_base + in_group)]
                    .copy_from_slice(&stage[8 * k * CNV_ACC_GROUP..8 * (k * CNV_ACC_GROUP + in_group)]);
            }
        }
    }
}

/// Compute the pairwise DFT-domain convolution
/// `res = (a[:,i] + a[:,j]) ⊙ (b[:,i] + b[:,j])`.
///
/// When `col_i == col_j` this delegates to [`ntt120_cnv_apply_dft`].
#[allow(clippy::too_many_arguments)]
pub fn ntt120_cnv_pairwise_apply_dft<BE>(
    module: &impl NttModuleHandle,
    cnv_offset: usize,
    res: &mut VecZnxDftBackendMut<'_, BE>,
    res_col: usize,
    a: &CnvPVecLBackendRef<'_, BE>,
    b: &CnvPVecRBackendRef<'_, BE>,
    col_i: usize,
    col_j: usize,
    tmp: &mut [u8],
) where
    BE: Backend<ScalarPrep = Q120bScalar> + NttAddAssign + NttMulBbc1ColX2,
    for<'x> <BE as Backend>::BufRef<'x>: HostDataRef,
    for<'x> <BE as Backend>::BufMut<'x>: crate::layouts::HostDataMut,
{
    if col_i == col_j {
        ntt120_cnv_apply_dft(module, cnv_offset, res, res_col, a, col_i, b, col_j, tmp);
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

    let a0 = col_slice_u32(a.raw(), n, a_size, col_i);
    let a1 = col_slice_u32(a.raw(), n, a_size, col_j);
    let b0 = col_slice_u32(b.raw(), n, b_size, col_i);
    let b1 = col_slice_u32(b.raw(), n, b_size, col_j);
    ntt120_conv_columns::<BE, false, true>(module, cnv_offset, res, res_col, a0, a1, a_size, b0, b1, b_size, tmp);
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

/// Scratch bytes required by [`ntt120_cnv_prepare_left`]: NTT and canonical limbs.
pub fn ntt120_cnv_prepare_left_tmp_bytes(n: usize) -> usize {
    8 * n * size_of::<u64>()
}

/// Encode a `VecZnx` into a `CnvPVecL` (canonical u32 rows, block-major).
///
/// Limbs of `res` beyond `a.size()` are zeroed.
pub fn ntt120_cnv_prepare_left<BE>(
    module: &impl NttModuleHandle,
    res: &mut CnvPVecLBackendMut<'_, BE>,
    a: &VecZnxBackendRef<'_, BE>,
    mask: i64,
    tmp: &mut [u8],
) where
    BE: Backend<ScalarPrep = Q120bScalar> + NttFromZnx64 + NttDFTExecute<NttTable<Primes30>> + NttPackLeft1BlkX2 + 'static,
    for<'x> BE: Backend<BufRef<'x> = &'x [u8], BufMut<'x> = &'x mut [u8]>,
{
    let n = res.n();
    let table = module.get_ntt_table();
    let cols = res.cols();
    let res_size = res.size();
    let min_size = res_size.min(a.size());
    let n_blks = n / 2;
    let col_stride = 8 * n * res_size;

    let (prefix, tmp_u64, suffix) = unsafe { tmp.align_to_mut::<u64>() };
    debug_assert!(prefix.is_empty());
    debug_assert!(suffix.is_empty());
    let (limb, canon_u64) = tmp_u64[..8 * n].split_at_mut(4 * n);
    let canon: &mut [u32] = cast_slice_mut(canon_u64);

    let res_u32: &mut [u32] = cast_slice_mut(res.raw_mut());
    for col in 0..cols {
        let dst = &mut res_u32[col * col_stride..(col + 1) * col_stride];
        for j in 0..min_size {
            if j + 1 == min_size {
                BE::ntt_from_znx64_masked(limb, a.at(col, j), mask);
            } else {
                BE::ntt_from_znx64(limb, a.at(col, j));
            }
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

/// Scratch bytes required by [`ntt120_cnv_prepare_right`]: NTT and converted limbs.
pub fn ntt120_cnv_prepare_right_tmp_bytes(n: usize) -> usize {
    8 * n * size_of::<u64>()
}

/// Encode a `VecZnx` into a `CnvPVecR` (q120c rows, block-major, reversed
/// limb order). Limbs of `res` beyond `a.size()` are zeroed.
pub fn ntt120_cnv_prepare_right<BE>(
    module: &impl NttModuleHandle,
    res: &mut CnvPVecRBackendMut<'_, BE>,
    a: &VecZnxBackendRef<'_, BE>,
    mask: i64,
    tmp: &mut [u64],
) where
    BE: Backend<ScalarPrep = Q120bScalar> + NttFromZnx64 + NttDFTExecute<NttTable<Primes30>> + NttCFromB + 'static,
    for<'x> BE: Backend<BufRef<'x> = &'x [u8], BufMut<'x> = &'x mut [u8]>,
{
    let n = res.n();
    let table = module.get_ntt_table();
    let cols = res.cols();
    let res_size = res.size();
    let min_size = res_size.min(a.size());
    let n_blks = n / 2;
    let col_stride = 8 * n * res_size;

    let (limb_b, limb_c_u64) = tmp[..8 * n].split_at_mut(4 * n);
    let limb_c: &mut [u32] = cast_slice_mut(limb_c_u64);

    let res_u32: &mut [u32] = cast_slice_mut(res.raw_mut());
    for col in 0..cols {
        let dst = &mut res_u32[col * col_stride..(col + 1) * col_stride];
        for j in 0..min_size {
            if j + 1 == min_size {
                BE::ntt_from_znx64_masked(limb_b, a.at(col, j), mask);
            } else {
                BE::ntt_from_znx64(limb_b, a.at(col, j));
            }
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

/// Scratch bytes required by [`ntt120_cnv_prepare_self`]: NTT, canonical and
/// converted limbs.
pub fn ntt120_cnv_prepare_self_tmp_bytes(n: usize) -> usize {
    12 * n * size_of::<u64>()
}

/// Encode a `VecZnx` into both `CnvPVecL` and `CnvPVecR` sharing the NTT.
pub fn ntt120_cnv_prepare_self<BE>(
    module: &impl NttModuleHandle,
    left: &mut CnvPVecLBackendMut<'_, BE>,
    right: &mut CnvPVecRBackendMut<'_, BE>,
    a: &VecZnxBackendRef<'_, BE>,
    mask: i64,
    tmp: &mut [u8],
) where
    BE: Backend<ScalarPrep = Q120bScalar>
        + NttFromZnx64
        + NttDFTExecute<NttTable<Primes30>>
        + NttCFromB
        + NttPackLeft1BlkX2
        + 'static,
    for<'x> BE: Backend<BufRef<'x> = &'x [u8], BufMut<'x> = &'x mut [u8]>,
{
    let n = left.n();
    let table = module.get_ntt_table();
    let cols = left.cols();
    let res_size = left.size();
    let min_size = res_size.min(a.size());
    let n_blks = n / 2;
    let col_stride = 8 * n * res_size;

    let (prefix, tmp_u64, suffix) = unsafe { tmp.align_to_mut::<u64>() };
    debug_assert!(prefix.is_empty());
    debug_assert!(suffix.is_empty());
    let (limb_b, rest) = tmp_u64[..12 * n].split_at_mut(4 * n);
    let (canon_u64, limb_c_u64) = rest.split_at_mut(4 * n);
    let canon: &mut [u32] = cast_slice_mut(canon_u64);
    let limb_c: &mut [u32] = cast_slice_mut(limb_c_u64);

    let left_u32: &mut [u32] = cast_slice_mut(left.raw_mut());
    let right_u32: &mut [u32] = cast_slice_mut(right.raw_mut());
    for col in 0..cols {
        let dst_l = &mut left_u32[col * col_stride..(col + 1) * col_stride];
        let dst_r = &mut right_u32[col * col_stride..(col + 1) * col_stride];
        for j in 0..min_size {
            if j + 1 == min_size {
                BE::ntt_from_znx64_masked(limb_b, a.at(col, j), mask);
            } else {
                BE::ntt_from_znx64(limb_b, a.at(col, j));
            }
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

/// Scratch bytes required by [`ntt120_cnv_by_const_apply`].
pub fn ntt120_cnv_by_const_apply_tmp_bytes(_res_size: usize, _a_size: usize, _b_size: usize) -> usize {
    0
}

/// Coefficient-domain negacyclic convolution: `res[k] = Σ a[k_abs−j] * b[j]`.
///
/// Each output limb is computed as an `i128` inner product. Output limbs
/// `min_size..res.size()` are zeroed. `_tmp` is unused.
#[allow(clippy::too_many_arguments)]
pub fn ntt120_cnv_by_const_apply<BE>(
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
    BE: Backend<ScalarPrep = Q120bScalar, ScalarBig = i128> + 'static,
    for<'x> BE: Backend<BufRef<'x> = &'x [u8]>,
    for<'x> <BE as Backend>::BufMut<'x>: crate::layouts::HostDataMut,
{
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
    let offset = cnv_offset.min(bound);
    let min_size = res_size.min((bound + 1).saturating_sub(offset));

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

// Lazy path used by glwe_mul_plain: NTT-only prepares, the apply packs each
// block on the fly. Avoids the eager block-major canonicalization that only
// amortizes for the operand-reusing tensor product.

pub fn ntt120_cnv_prepare_left_lazy_tmp_bytes(_n: usize) -> usize {
    0
}

pub fn ntt120_cnv_prepare_left_lazy<BE>(
    module: &impl NttModuleHandle,
    res: &mut CnvPVecLBackendMut<'_, BE>,
    a: &VecZnxBackendRef<'_, BE>,
    mask: i64,
    _tmp: &mut [u8],
) where
    BE: Backend<ScalarPrep = Q120bScalar> + NttFromZnx64 + NttDFTExecute<NttTable<Primes30>> + 'static,
    for<'x> BE: Backend<BufRef<'x> = &'x [u8], BufMut<'x> = &'x mut [u8]>,
{
    let table = module.get_ntt_table();
    let cols = res.cols();
    let res_size = res.size();
    let min_size = res_size.min(a.size());

    for col in 0..cols {
        for j in 0..min_size.saturating_sub(1) {
            let res_u64: &mut [u64] = cast_slice_mut(res.at_mut(col, j));
            BE::ntt_from_znx64(res_u64, a.at(col, j));
            BE::ntt_dft_execute(table, res_u64);
        }
        if min_size > 0 {
            let last = min_size - 1;
            let res_u64: &mut [u64] = cast_slice_mut(res.at_mut(col, last));
            BE::ntt_from_znx64_masked(res_u64, a.at(col, last), mask);
            BE::ntt_dft_execute(table, res_u64);
        }
        for j in min_size..res_size {
            cast_slice_mut::<_, u64>(res.at_mut(col, j)).fill(0);
        }
    }
}

pub fn ntt120_cnv_prepare_right_lazy_tmp_bytes(n: usize) -> usize {
    4 * n * size_of::<u64>()
}

pub fn ntt120_cnv_prepare_right_lazy<BE>(
    module: &impl NttModuleHandle,
    res: &mut CnvPVecRBackendMut<'_, BE>,
    a: &VecZnxBackendRef<'_, BE>,
    mask: i64,
    tmp: &mut [u64],
) where
    BE: Backend<ScalarPrep = Q120bScalar> + NttFromZnx64 + NttDFTExecute<NttTable<Primes30>> + NttCFromB + 'static,
    for<'x> BE: Backend<BufRef<'x> = &'x [u8], BufMut<'x> = &'x mut [u8]>,
{
    let n = res.n();
    let table = module.get_ntt_table();
    let cols = res.cols();
    let res_size = res.size();
    let min_size = res_size.min(a.size());

    for col in 0..cols {
        for j in 0..min_size.saturating_sub(1) {
            BE::ntt_from_znx64(tmp, a.at(col, j));
            BE::ntt_dft_execute(table, tmp);
            let res_u32: &mut [u32] = cast_slice_mut(res.at_mut(col, j));
            BE::ntt_c_from_b(n, res_u32, tmp);
        }
        if min_size > 0 {
            let last = min_size - 1;
            BE::ntt_from_znx64_masked(tmp, a.at(col, last), mask);
            BE::ntt_dft_execute(table, tmp);
            let res_u32: &mut [u32] = cast_slice_mut(res.at_mut(col, last));
            BE::ntt_c_from_b(n, res_u32, tmp);
        }
        for j in min_size..res_size {
            cast_slice_mut::<_, u32>(res.at_mut(col, j)).fill(0);
        }
    }
}

pub fn ntt120_cnv_apply_dft_lazy_tmp_bytes(_res_size: usize, a_size: usize, b_size: usize) -> usize {
    (16 * (a_size + b_size)) * size_of::<u32>()
}

#[allow(clippy::too_many_arguments)]
pub fn ntt120_cnv_apply_dft_lazy<BE>(
    module: &impl NttModuleHandle,
    cnv_offset: usize,
    res: &mut VecZnxDftBackendMut<'_, BE>,
    res_col: usize,
    a: &CnvPVecLBackendRef<'_, BE>,
    a_col: usize,
    b: &CnvPVecRBackendRef<'_, BE>,
    b_col: usize,
    tmp: &mut [u8],
) where
    BE: Backend<ScalarPrep = Q120bScalar> + NttMulBbc1ColX2 + NttPackLeft1BlkX2 + NttPackRight1BlkX2,
    for<'x> <BE as Backend>::BufRef<'x>: HostDataRef,
    for<'x> <BE as Backend>::BufMut<'x>: crate::layouts::HostDataMut,
{
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

    let meta = module.get_bbc_meta();
    let a_cols = a.cols();
    let b_cols = b.cols();
    let n_blks = n / 2;
    let a_row_stride_u64 = 4 * n * a_cols;
    let b_row_stride_u32 = 8 * n * b_cols;
    let a_col_offset_u64 = 4 * n * a_col;
    let b_col_offset_u32 = 8 * n * b_col;
    let a_raw_u64: &[u64] = cast_slice(a.raw());
    let b_raw_u32: &[u32] = cast_slice(b.raw());

    let (prefix, tmp_u32, suffix) = unsafe { tmp.align_to_mut::<u32>() };
    debug_assert!(prefix.is_empty());
    debug_assert!(suffix.is_empty());
    debug_assert!(tmp_u32.len() >= 16 * (a_size + b_size));
    let (a_tmp, b_tmp) = tmp_u32.split_at_mut(16 * a_size);

    for blk in 0..n_blks {
        BE::ntt_pack_left_1blk_x2(a_tmp, &a_raw_u64[a_col_offset_u64..], a_size, a_row_stride_u64, blk);
        BE::ntt_pack_right_1blk_x2(b_tmp, &b_raw_u32[b_col_offset_u32..], b_size, b_row_stride_u32, blk);

        for k in 0..min_size {
            let k_abs = k + offset;
            let j_max = (k_abs + 1).min(b_size);
            let j_min = k_abs.saturating_sub(a_size - 1);
            let ell = j_max - j_min;
            let a_start = k_abs + 1 - j_max;
            let b_start = b_size - j_max;

            let res_u64: &mut [u64] = cast_slice_mut(res.at_mut(res_col, k));
            BE::ntt_mul_bbc_1col_x2(
                meta,
                ell,
                &mut res_u64[8 * blk..8 * blk + 8],
                &a_tmp[16 * a_start..],
                &b_tmp[16 * b_start..],
            );
        }
    }

    for j in min_size..res_size {
        cast_slice_mut::<_, u64>(res.at_mut(res_col, j)).fill(0);
    }
}
