//! Bivariate convolution operations for the NTT120 backend.
//!
//! Implements the five-function convolution pipeline used by
//! `poulpy-cpu-ref`:
//!
//! | Step | Function | Description |
//! |------|----------|-------------|
//! | 1a | [`ntt120_cnv_prepare_left`]  | Encode `VecZnx` → `CnvPVecL` (q120b, NTT domain) |
//! | 1b | [`ntt120_cnv_prepare_right`] | Encode `VecZnx` → `CnvPVecR` (q120c, NTT domain) |
//! | 2  | [`ntt120_cnv_apply_dft`]     | `res[k] = Σ a[j] ⊙ b[k−j]` (bbc product) |
//! | 2p | [`ntt120_cnv_pairwise_apply_dft`] | `res = (a[:,i]+a[:,j]) ⊙ (b[:,i]+b[:,j])` |
//! | 3  | [`ntt120_cnv_by_const_apply`] | Coefficient-domain negacyclic convolution into i128 |
//!
//! # Prepared-format asymmetry
//!
//! The bbc kernel (`accum_mul_q120_bc` in `mat_vec`) expects its
//! left operand in **q120b** (4 × u64 per NTT coefficient) and its right
//! operand in **q120c** (8 × u32: `(r mod Qₖ, r·2³² mod Qₖ)` per prime).
//! `CnvPVecL` stores q120b; `CnvPVecR` stores q120c.  Both are 32 bytes
//! per NTT coefficient — the same as `size_of::<Q120bScalar>()`.
//!
//! # Memory layout (Option A)
//!
//! Both `CnvPVecL` and `CnvPVecR` use the same flat layout as
//! `vec_znx_dft`: for column `col`, limb `j`, and NTT
//! coefficient index `n_i`, the element lives at
//! `(col * size + j) * n + n_i` in [`Q120bScalar`] units.  Access via
//! [`ZnxView::at`] / `at_mut` is identical
//! to `VecZnxDft`.

use bytemuck::{cast_slice, cast_slice_mut};

use crate::{
    layouts::{
        Backend, CnvPVecL, CnvPVecLToMut, CnvPVecLToRef, CnvPVecR, CnvPVecRToMut, CnvPVecRToRef, VecZnx, VecZnxBig,
        VecZnxBigToMut, VecZnxDft, VecZnxDftToMut, VecZnxToRef, ZnxInfos, ZnxView, ZnxViewMut,
    },
    reference::ntt120::{
        NttAddAssign, NttCFromB, NttDFTExecute, NttFromZnx64, NttMulBbc1ColX2, NttMulBbc2ColsX2, NttPackLeft1BlkX2,
        NttPackRight1BlkX2, NttPairwisePackLeft1BlkX2, NttPairwisePackRight1BlkX2, ntt::NttTable, primes::Primes30,
        types::Q120bScalar, vec_znx_dft::NttModuleHandle,
    },
};

// ──────────────────────────────────────────────────────────────────────────────
// Prepare left  (VecZnx → CnvPVecL, q120b)
// ──────────────────────────────────────────────────────────────────────────────

/// Scratch bytes required by [`ntt120_cnv_prepare_left`].
///
/// Returns 0: the function writes the NTT directly into the output buffer.
pub fn ntt120_cnv_prepare_left_tmp_bytes(_n: usize) -> usize {
    0
}

/// Encode a `VecZnx` (i64 coefficients) into a `CnvPVecL` (q120b, NTT domain).
///
/// For each column `col` and each limb `j` of the input `a`:
/// 1. Map i64 coefficients → q120b via `BE::ntt_from_znx64`.
/// 2. Apply the forward NTT in-place via `BE::ntt_dft_execute`.
/// 3. Store the result directly in `res[col, j]` as q120b.
///
/// Limbs of `res` beyond `a.size()` are zeroed.
/// No scratch buffer is needed; `_tmp` is unused.
pub fn ntt120_cnv_prepare_left<R, A, BE>(module: &impl NttModuleHandle, res: &mut R, a: &A, mask: i64, _tmp: &mut [u8])
where
    BE: Backend<ScalarPrep = Q120bScalar> + NttFromZnx64 + NttDFTExecute<NttTable<Primes30>>,
    R: CnvPVecLToMut<BE>,
    A: VecZnxToRef,
{
    let mut res: CnvPVecL<&mut [u8], BE> = res.to_mut();
    let a: VecZnx<&[u8]> = a.to_ref();
    let table = module.get_ntt_table();
    let cols = res.cols();
    let res_size = res.size();
    let min_size = res_size.min(a.size());

    for col in 0..cols {
        // All limbs except the last: unmasked fast path.
        for j in 0..min_size.saturating_sub(1) {
            let res_u64: &mut [u64] = cast_slice_mut(res.at_mut(col, j));
            BE::ntt_from_znx64(res_u64, a.at(col, j));
            BE::ntt_dft_execute(table, res_u64);
        }
        // Last active limb: masked path.
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

// ──────────────────────────────────────────────────────────────────────────────
// Prepare right  (VecZnx → CnvPVecR, q120c)
// ──────────────────────────────────────────────────────────────────────────────

/// Scratch bytes required by [`ntt120_cnv_prepare_right`].
///
/// Returns `4 * n * 8` bytes — one intermediate q120b buffer of `4 * n` u64
/// values used for the NTT before the q120b → q120c conversion.
pub fn ntt120_cnv_prepare_right_tmp_bytes(n: usize) -> usize {
    4 * n * size_of::<u64>()
}

/// Encode a `VecZnx` (i64 coefficients) into a `CnvPVecR` (q120c, NTT domain).
///
/// For each column `col` and each limb `j` of the input `a`:
/// 1. Map i64 coefficients → q120b via [`b_from_znx64_ref`] into `tmp`.
/// 2. Apply the forward NTT in-place via [`ntt_ref`].
/// 3. Convert q120b → q120c via [`c_from_b_ref`] into `res[col, j]`.
///
/// `tmp` must hold at least `ntt120_cnv_prepare_right_tmp_bytes(n) / size_of::<u64>()` elements.
/// Limbs of `res` beyond `a.size()` are zeroed.
pub fn ntt120_cnv_prepare_right<R, A, BE>(module: &impl NttModuleHandle, res: &mut R, a: &A, mask: i64, tmp: &mut [u64])
where
    BE: Backend<ScalarPrep = Q120bScalar> + NttFromZnx64 + NttDFTExecute<NttTable<Primes30>> + NttCFromB,
    R: CnvPVecRToMut<BE>,
    A: VecZnxToRef,
{
    let mut res: CnvPVecR<&mut [u8], BE> = res.to_mut();
    let a: VecZnx<&[u8]> = a.to_ref();
    let n = res.n();
    let table = module.get_ntt_table();
    let cols = res.cols();
    let res_size = res.size();
    let min_size = res_size.min(a.size());

    for col in 0..cols {
        // All limbs except the last: unmasked fast path.
        for j in 0..min_size.saturating_sub(1) {
            BE::ntt_from_znx64(tmp, a.at(col, j));
            BE::ntt_dft_execute(table, tmp);
            let res_u32: &mut [u32] = cast_slice_mut(res.at_mut(col, j));
            BE::ntt_c_from_b(n, res_u32, tmp);
        }
        // Last active limb: masked path.
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

// ──────────────────────────────────────────────────────────────────────────────
// Prepare self  (VecZnx → CnvPVecL + CnvPVecR, shared NTT)
// ──────────────────────────────────────────────────────────────────────────────

/// Scratch bytes required by [`ntt120_cnv_prepare_self`].
///
/// Returns 0: the function writes the NTT into the left buffer first,
/// then derives the right buffer from it.
pub fn ntt120_cnv_prepare_self_tmp_bytes(_n: usize) -> usize {
    0
}

/// Encode a `VecZnx` into both `CnvPVecL` (q120b) and `CnvPVecR` (q120c)
/// sharing the NTT computation.
///
/// For each column and limb:
/// 1. Map i64 → q120b via `BE::ntt_from_znx64` into the left buffer.
/// 2. Apply forward NTT in-place on the left buffer via `BE::ntt_dft_execute`.
/// 3. Convert the NTT-domain q120b (left) → q120c (right) via `BE::ntt_c_from_b`.
///
/// This saves one full `b_from_znx64 + NTT` per (col, limb) compared to
/// calling `prepare_left` + `prepare_right` separately.
pub fn ntt120_cnv_prepare_self<L, R, A, BE>(
    module: &impl NttModuleHandle,
    left: &mut L,
    right: &mut R,
    a: &A,
    mask: i64,
    _tmp: &mut [u8],
) where
    BE: Backend<ScalarPrep = Q120bScalar> + NttFromZnx64 + NttDFTExecute<NttTable<Primes30>> + NttCFromB,
    L: CnvPVecLToMut<BE>,
    R: CnvPVecRToMut<BE>,
    A: VecZnxToRef,
{
    let mut left: CnvPVecL<&mut [u8], BE> = left.to_mut();
    let mut right: CnvPVecR<&mut [u8], BE> = right.to_mut();
    let a: VecZnx<&[u8]> = a.to_ref();
    let table = module.get_ntt_table();
    let n = left.n();
    let cols = left.cols();
    let res_size = left.size();
    let min_size = res_size.min(a.size());

    for col in 0..cols {
        // All limbs except the last: unmasked fast path.
        for j in 0..min_size.saturating_sub(1) {
            {
                let left_u64: &mut [u64] = cast_slice_mut(left.at_mut(col, j));
                BE::ntt_from_znx64(left_u64, a.at(col, j));
                BE::ntt_dft_execute(table, left_u64);
            }
            let left_u64: &[u64] = cast_slice(left.at(col, j));
            let right_u32: &mut [u32] = cast_slice_mut(right.at_mut(col, j));
            BE::ntt_c_from_b(n, right_u32, left_u64);
        }
        // Last active limb: masked path.
        if min_size > 0 {
            let last = min_size - 1;
            {
                let left_u64: &mut [u64] = cast_slice_mut(left.at_mut(col, last));
                BE::ntt_from_znx64_masked(left_u64, a.at(col, last), mask);
                BE::ntt_dft_execute(table, left_u64);
            }
            let left_u64: &[u64] = cast_slice(left.at(col, last));
            let right_u32: &mut [u32] = cast_slice_mut(right.at_mut(col, last));
            BE::ntt_c_from_b(n, right_u32, left_u64);
        }
        for j in min_size..res_size {
            cast_slice_mut::<_, u64>(left.at_mut(col, j)).fill(0);
            cast_slice_mut::<_, u32>(right.at_mut(col, j)).fill(0);
        }
    }
}

// ──────────────────────────────────────────────────────────────────────────────
// Apply DFT  (CnvPVecL × CnvPVecR → VecZnxDft, bbc product)
// ──────────────────────────────────────────────────────────────────────────────

/// Scratch bytes required by [`ntt120_cnv_apply_dft`].
///
/// Stores packed full-row x2 blocks for `a` and `b`.
pub fn ntt120_cnv_apply_dft_tmp_bytes(_res_size: usize, a_size: usize, b_size: usize) -> usize {
    (16 * (a_size + b_size)) * size_of::<u32>()
}

/// Compute the DFT-domain bivariate convolution `res[k] = Σ a[j] ⊙ b[k−j]`.
///
/// For each output limb `k ∈ [0, min_size)` and each x2 NTT block:
///
/// ```text
/// res[res_col, k, blk] = Σ_{j=j_min}^{j_max-1}  bbc_x2( a[a_col, k_abs−j, blk],
///                                                         b[b_col,       j, blk] )
/// ```
///
/// where `k_abs = k + cnv_offset`, `j_min = max(0, k_abs − a.size() + 1)`,
/// `j_max = min(k_abs + 1, b.size())`, and `bbc_x2` denotes the backend
/// x2 q120b × q120c dot-product kernel.
///
/// Output limbs `min_size..res.size()` are zeroed.
#[allow(clippy::too_many_arguments)]
pub fn ntt120_cnv_apply_dft<R, A, B, BE>(
    module: &impl NttModuleHandle,
    cnv_offset: usize,
    res: &mut R,
    res_col: usize,
    a: &A,
    a_col: usize,
    b: &B,
    b_col: usize,
    tmp: &mut [u8],
) where
    BE: Backend<ScalarPrep = Q120bScalar> + NttMulBbc1ColX2 + NttPackLeft1BlkX2 + NttPackRight1BlkX2,
    R: VecZnxDftToMut<BE>,
    A: CnvPVecLToRef<BE>,
    B: CnvPVecRToRef<BE>,
{
    let mut res: VecZnxDft<&mut [u8], BE> = res.to_mut();
    let a: CnvPVecL<&[u8], BE> = a.to_ref();
    let b: CnvPVecR<&[u8], BE> = b.to_ref();

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
            let j_min = k_abs.saturating_sub(a_size - 1);
            let j_max = (k_abs + 1).min(b_size);
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

// ──────────────────────────────────────────────────────────────────────────────
// By-const apply  (VecZnx × &[i64] → VecZnxBig, coefficient domain)
// ──────────────────────────────────────────────────────────────────────────────

/// Scratch bytes required by [`ntt120_cnv_by_const_apply`].
///
/// Returns 0: the function uses `i128` stack accumulators.
pub fn ntt120_cnv_by_const_apply_tmp_bytes(_res_size: usize, _a_size: usize, _b_size: usize) -> usize {
    0
}

/// Coefficient-domain negacyclic convolution: `res[k] = Σ a[k_abs−j] * b[j]`.
///
/// Unlike [`ntt120_cnv_apply_dft`], this function operates entirely in
/// the **coefficient domain** (no NTT).  Each output limb is computed as
/// an `i128` inner product, suitable for accumulation into a
/// [`VecZnxBig`] with `ScalarBig = i128`.
///
/// For each output limb `k ∈ [0, min_size)` and ring coefficient `n_i`:
///
/// ```text
/// res[res_col, k, n_i] = Σ_{j=j_min}^{j_max-1}  a[a_col, k_abs−j, n_i]  ×  b[j]
/// ```
///
/// where `k_abs = k + cnv_offset`.
/// Output limbs `min_size..res.size()` are zeroed.
/// `_tmp` is unused.
#[allow(clippy::too_many_arguments)]
pub fn ntt120_cnv_by_const_apply<R, A, BE>(
    cnv_offset: usize,
    res: &mut R,
    res_col: usize,
    a: &A,
    a_col: usize,
    b: &[i64],
    _tmp: &mut [u8],
) where
    BE: Backend<ScalarPrep = Q120bScalar, ScalarBig = i128>,
    R: VecZnxBigToMut<BE>,
    A: VecZnxToRef,
{
    let mut res: VecZnxBig<&mut [u8], BE> = res.to_mut();
    let a: VecZnx<&[u8]> = a.to_ref();
    let res_size = res.size();
    let a_size = a.size();
    let b_size = b.len();
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
            for (j, &b_j) in b.iter().enumerate().take(j_max).skip(j_min) {
                acc += a.at(a_col, k_abs - j)[n_i] as i128 * b_j as i128;
            }
            *r = acc;
        }
    }

    for j in min_size..res_size {
        res.at_mut(res_col, j).fill(0i128);
    }
}

// ──────────────────────────────────────────────────────────────────────────────
// Pairwise apply DFT
// ──────────────────────────────────────────────────────────────────────────────

/// Scratch bytes required by [`ntt120_cnv_pairwise_apply_dft`].
///
/// Stores one packed x2-block row-set for the summed left operand and one
/// reversed packed x2-block row-set for the summed right operand.
pub fn ntt120_cnv_pairwise_apply_dft_tmp_bytes(res_size: usize, a_size: usize, b_size: usize) -> usize {
    if a_size == 0 || b_size == 0 || res_size == 0 {
        0
    } else {
        (16 * (a_size + b_size) * size_of::<u32>()).max(ntt120_cnv_apply_dft_tmp_bytes(res_size, a_size, b_size))
    }
}

/// Compute the pairwise DFT-domain convolution:
/// `res = (a[:,col_i] + a[:,col_j]) ⊙ (b[:,col_i] + b[:,col_j])`.
///
/// This mirrors the FFT64 reference: both `a` columns are summed first,
/// both `b` columns are summed first, then a single bbc convolution is
/// performed on the sums. This is **not** the same as
/// `a[:,i]⊙b[:,i] + a[:,j]⊙b[:,j]` — cross-terms are present by design.
///
/// When `col_i == col_j` this delegates to [`ntt120_cnv_apply_dft`].
///
/// For each x2 NTT block, the pairwise sums are packed once:
/// ```text
/// a_blk[row] = block_x2(a[col_i, row] + a[col_j, row]) mod Q
/// b_blk[row] = block_x2(b[col_i, b_size-1-row] + b[col_j, b_size-1-row])
/// ```
///
/// Then each output limb consumes contiguous windows from those packed rows:
/// ```text
/// res[res_col, k, blk] =
///     Σ_{row=a_start}^{a_start+ell-1} bbc_x2(a_blk[row], b_blk[b_size-j_max + (row-a_start)])
/// ```
///
/// Output limbs `min_size..res.size()` are zeroed.
#[allow(clippy::too_many_arguments)]
pub fn ntt120_cnv_pairwise_apply_dft<R, A, B, BE>(
    module: &impl NttModuleHandle,
    cnv_offset: usize,
    res: &mut R,
    res_col: usize,
    a: &A,
    b: &B,
    col_i: usize,
    col_j: usize,
    tmp: &mut [u8],
) where
    BE: Backend<ScalarPrep = Q120bScalar>
        + NttAddAssign
        + NttMulBbc1ColX2
        + NttMulBbc2ColsX2
        + NttPackLeft1BlkX2
        + NttPackRight1BlkX2
        + NttPairwisePackLeft1BlkX2
        + NttPairwisePackRight1BlkX2,
    R: VecZnxDftToMut<BE>,
    A: CnvPVecLToRef<BE>,
    B: CnvPVecRToRef<BE>,
{
    if col_i == col_j {
        ntt120_cnv_apply_dft(module, cnv_offset, res, res_col, a, col_i, b, col_j, tmp);
        return;
    }

    let mut res: VecZnxDft<&mut [u8], BE> = res.to_mut();
    let a: CnvPVecL<&[u8], BE> = a.to_ref();
    let b: CnvPVecR<&[u8], BE> = b.to_ref();

    let meta = module.get_bbc_meta();
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

    let a_cols = a.cols();
    let b_cols = b.cols();

    let bound = a_size + b_size - 1;
    let offset = cnv_offset.min(bound);
    let min_size = res_size.min((bound + 1).saturating_sub(offset));
    let n_blks = n / 2;
    let a_row_stride_u64 = 4 * n * a_cols;
    let b_row_stride_u32 = 8 * n * b_cols;
    let a_col_offset_u64_i = 4 * n * col_i;
    let a_col_offset_u64_j = 4 * n * col_j;
    let b_col_offset_u32_i = 8 * n * col_i;
    let b_col_offset_u32_j = 8 * n * col_j;
    let a_raw_u64: &[u64] = cast_slice(a.raw());
    let b_raw_u32: &[u32] = cast_slice(b.raw());

    let (prefix, tmp_u32, suffix) = unsafe { tmp.align_to_mut::<u32>() };
    debug_assert!(prefix.is_empty());
    debug_assert!(suffix.is_empty());
    debug_assert!(tmp_u32.len() >= 16 * (a_size + b_size));
    let (a_tmp, b_tmp) = tmp_u32.split_at_mut(16 * a_size);

    for blk in 0..n_blks {
        BE::ntt_pairwise_pack_left_1blk_x2(
            a_tmp,
            &a_raw_u64[a_col_offset_u64_i..],
            &a_raw_u64[a_col_offset_u64_j..],
            a_size,
            a_row_stride_u64,
            blk,
        );
        BE::ntt_pairwise_pack_right_1blk_x2(
            b_tmp,
            &b_raw_u32[b_col_offset_u32_i..],
            &b_raw_u32[b_col_offset_u32_j..],
            b_size,
            b_row_stride_u32,
            blk,
        );

        for k in 0..min_size {
            let k_abs = k + offset;
            let j_min = k_abs.saturating_sub(a_size - 1);
            let j_max = (k_abs + 1).min(b_size);
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
