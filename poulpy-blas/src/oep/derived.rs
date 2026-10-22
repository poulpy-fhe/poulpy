//! The provided methods of [`BlasMatIntraImpl`], the transpose, the row and column shifts and the axis reductions, each
//! composed over the map hook [`BlasMatIntraImpl::blas_map_blocks_impl`] and poulpy-ckks's copy, addition and rotation;
//! and of [`BlasMatMulImpl`], `blas_matmul` over both hot preparations and the prepared product. Each runs its
//! validator before any write and, on `Ok`, stamps the batch record of `res`, as the references do. They are public so
//! that an override of a provided method can fall back to it; their scratch assertion over the keys a map looks up is
//! shared with the references.

use std::cell::RefCell;

use poulpy_ckks::{
    CKKSCtBounds, CKKSLayout, CKKSResult as Result, SetCKKSInfos,
    api::{
        CKKSAddManyOps, CKKSAddOps, CKKSCopyOps, CKKSInnerSumOps, CKKSModuleInfos, CKKSRotateOps, LinearTransformation,
        LinearTransformationPrepared, LtDiagonalMeta,
    },
    layouts::{CKKSModuleAlloc, ScratchArenaTakeCKKS},
};
use poulpy_core::{
    GLWEBytesOf, GLWENormalize,
    layouts::{
        GGLWEInfos, GGLWELayout, GLWEInfos, GLWEToBackendMut, GLWEToBackendRef, GetAutomorphismKey, GetTensorKey, LWEInfos, Rank,
        TorusPrecision, max_lazy_terms,
        prepared::{GLWEAutomorphismKeyPreparedBackendRef, GLWETensorKeyPreparedBackendRef},
    },
};
use poulpy_hal::layouts::{Backend, Module, PrepareHint, ScratchArena};

use super::{BlasMatIntraImpl, BlasMatMulImpl};
use crate::layouts::{
    BlasContext, BlasContextInfos, BlasGrid, BlasGridInfos, BlasGridMut, BlasMatMulOperandPreparedFactory,
    ciphertext::replicated_at,
    context::{reduction_projection, shift_maps, shift_rotates},
    encoding::{copied_block, mapped_block_into},
    matmul_operand::{left_operand_layout, right_operand_layout},
    validate::{assert_scratch, ensure_rotation, lt_uses, reduction_uses, shift_uses},
    validate_blas_matmul, validate_blas_rotate_cols, validate_blas_rotate_rows, validate_blas_sum_cols, validate_blas_sum_rows,
    validate_blas_transpose,
};

/// The provided `blas_transpose_tmp_bytes_impl`: the hook's scratch for the transpose of every block of `a` into a
/// block of `res`.
#[doc(hidden)]
pub fn blas_transpose_tmp_bytes_derived<BE, R, A, K>(module: &Module<BE>, res: &R, a: &A, ctx: &BlasContext<BE>, atk: &K) -> usize
where
    BE: BlasMatIntraImpl,
    R: BlasGridInfos,
    A: BlasGridInfos,
    K: GGLWEInfos,
{
    let sources = a.rows() * a.cols();
    BE::blas_map_blocks_tmp_bytes_impl(
        module,
        &res.block_layout(),
        &a.block_layout(),
        sources,
        &[ctx.transpose()],
        atk,
    )
}

/// The provided `blas_transpose_impl`, `res_ji = transpose(a_ij)`: one hook call, its sources in the order of `res`'s
/// blocks.
#[doc(hidden)]
pub fn blas_transpose_derived<BE, R, A, K>(
    module: &Module<BE>,
    res: &mut R,
    a: &A,
    ctx: &BlasContext<BE>,
    atk: &K,
    scratch: &mut ScratchArena<'_, BE>,
) -> Result<()>
where
    BE: BlasMatIntraImpl,
    R: BlasGridMut<BE>,
    A: BlasGrid<BE>,
    K: GetAutomorphismKey<BE> + Sync,
{
    let batch = validate_blas_transpose(module, &*res, a, ctx, atk)?;
    let lts = [ctx.transpose()];
    let (src_k, dst_k, rank) = (a.block_layout().k(), res.block_layout().k(), a.block_layout().rank());
    let bytes = |key: &GGLWELayout| blas_transpose_tmp_bytes_derived(module, &*res, a, ctx, key);
    assert_maps_scratch("blas_transpose", module, &lts, atk, src_k, dst_k, rank, bytes, scratch);
    let (rows, cols, blocks) = (a.rows(), a.cols(), a.blocks());
    let src: Vec<_> = (0..cols)
        .flat_map(|j| (0..rows).map(move |i| &blocks[i * cols + j]))
        .collect();
    BE::blas_map_blocks_impl(module, res.blocks_mut(), &src, &lts, atk, scratch)?;
    res.set_mat_slots(batch);
    Ok(())
}

/// The provided `blas_rotate_rows_tmp_bytes_impl`: the scratch of a row shift by any `k`, as [`shift_tmp_bytes`].
#[doc(hidden)]
pub fn blas_rotate_rows_tmp_bytes_derived<BE, R, A, K>(
    module: &Module<BE>,
    res: &R,
    a: &A,
    ctx: &BlasContext<BE>,
    atk: &K,
) -> usize
where
    BE: BlasMatIntraImpl,
    R: BlasGridInfos,
    A: BlasGridInfos,
    K: GGLWEInfos,
{
    shift_tmp_bytes(module, res, a, true, ctx, atk)
}

/// The provided `blas_rotate_cols_tmp_bytes_impl`: the scratch of a column shift by any `k`, as [`shift_tmp_bytes`].
#[doc(hidden)]
pub fn blas_rotate_cols_tmp_bytes_derived<BE, R, A, K>(
    module: &Module<BE>,
    res: &R,
    a: &A,
    ctx: &BlasContext<BE>,
    atk: &K,
) -> usize
where
    BE: BlasMatIntraImpl,
    R: BlasGridInfos,
    A: BlasGridInfos,
    K: GGLWEInfos,
{
    shift_tmp_bytes(module, res, a, false, ctx, atk)
}

/// Scratch of a row (`rows`) or column shift by any `k`: the copy for `k' = 0`, otherwise the rotation where the shift
/// rotates ([`shift_rotates`]), or else for every `k'` the hook with psi^k' (phi^k') along one block, or with both half
/// maps plus the addition along several.
fn shift_tmp_bytes<BE, R, A, K>(module: &Module<BE>, res: &R, a: &A, rows: bool, ctx: &BlasContext<BE>, atk: &K) -> usize
where
    BE: BlasMatIntraImpl,
    R: BlasGridInfos,
    A: BlasGridInfos,
    K: GGLWEInfos,
{
    let extent = if rows { a.rows() } else { a.cols() };
    let rotates = shift_rotates(rows, extent, replicated_at(ctx.slots(), a));
    let (sources, res, a) = (a.rows() * a.cols(), res.block_layout(), a.block_layout());
    let copy = module.ckks_copy_tmp_bytes(&res, &a);
    if rotates {
        return copy
            .max(module.ckks_rotate_tmp_bytes(&res, atk))
            .max(module.ckks_rotate_tmp_bytes(&a, atk));
    }
    (1..1 << ctx.log_d())
        .map(|k1| {
            let lts = shift_maps(&ctx.maps, k1, rows, extent);
            let hook = BE::blas_map_blocks_tmp_bytes_impl(module, &res, &a, sources, &lts, atk);
            if lts.len() == 2 {
                hook.max(module.ckks_add_tmp_bytes(res.size()))
            } else {
                hook
            }
        })
        .fold(copy, usize::max)
}

/// The provided `blas_rotate_rows_impl`, as [`shift`].
#[doc(hidden)]
pub fn blas_rotate_rows_derived<BE, R, A, K>(
    module: &Module<BE>,
    res: &mut R,
    a: &A,
    k: usize,
    ctx: &BlasContext<BE>,
    atk: &K,
    scratch: &mut ScratchArena<'_, BE>,
) -> Result<()>
where
    BE: BlasMatIntraImpl,
    R: BlasGridMut<BE>,
    A: BlasGrid<BE>,
    K: GetAutomorphismKey<BE> + Sync,
{
    shift(module, "blas_rotate_rows", res, a, k, true, ctx, atk, scratch)
}

/// The provided `blas_rotate_cols_impl`, as [`shift`].
#[doc(hidden)]
pub fn blas_rotate_cols_derived<BE, R, A, K>(
    module: &Module<BE>,
    res: &mut R,
    a: &A,
    k: usize,
    ctx: &BlasContext<BE>,
    atk: &K,
    scratch: &mut ScratchArena<'_, BE>,
) -> Result<()>
where
    BE: BlasMatIntraImpl,
    R: BlasGridMut<BE>,
    A: BlasGrid<BE>,
    K: GetAutomorphismKey<BE> + Sync,
{
    shift(module, "blas_rotate_cols", res, a, k, false, ctx, atk, scratch)
}

/// Shifts every matrix by `k mod m d = q d + k'` rows (`rows`) or columns, along the block axis of extent `m`. With
/// `src(t, o)` the block `o` steps past block `t` along that axis: `k' = 0` copies `src(t, q)` into block `t`, then
/// clears its noise tag; where the shift rotates ([`shift_rotates`]), block `t` is `a`'s rotated by `k' d` slots;
/// `m = 1` is otherwise one hook call with psi^k' (phi^k'); otherwise one hook call evaluates both half maps of every
/// block of `a` into a heap grid `h` of `res`'s block layout, and block `t` is
/// `h[first, src(t, q)] + h[second, src(t, q + 1)]`.
#[allow(clippy::too_many_arguments)]
fn shift<BE, R, A, K>(
    module: &Module<BE>,
    op: &'static str,
    res: &mut R,
    a: &A,
    k: usize,
    rows: bool,
    ctx: &BlasContext<BE>,
    atk: &K,
    scratch: &mut ScratchArena<'_, BE>,
) -> Result<()>
where
    BE: BlasMatIntraImpl,
    R: BlasGridMut<BE>,
    A: BlasGrid<BE>,
    K: GetAutomorphismKey<BE> + Sync,
{
    let batch = if rows {
        validate_blas_rotate_rows(module, &*res, a, k, ctx, atk)?
    } else {
        validate_blas_rotate_cols(module, &*res, a, k, ctx, atk)?
    };
    shift_blocks(module, op, res, a, k, rows, ctx, atk, scratch)?;
    res.set_mat_slots(batch);
    Ok(())
}

/// The blocks of [`shift`], on inputs its validator accepted.
#[allow(clippy::too_many_arguments)]
fn shift_blocks<BE, R, A, K>(
    module: &Module<BE>,
    op: &'static str,
    res: &mut R,
    a: &A,
    k: usize,
    rows: bool,
    ctx: &BlasContext<BE>,
    atk: &K,
    scratch: &mut ScratchArena<'_, BE>,
) -> Result<()>
where
    BE: BlasMatIntraImpl,
    R: BlasGridMut<BE>,
    A: BlasGrid<BE>,
    K: GetAutomorphismKey<BE> + Sync,
{
    let d = 1usize << ctx.log_d();
    let (a_rows, a_cols) = (a.rows(), a.cols());
    let extent = if rows { a_rows } else { a_cols };
    // `k mod extent d = q d + k1`, without forming `extent d`.
    let (q, k1) = (k / d % extent, k % d);
    // `res` has `a`'s shape, so block `t` of both sits at row `t / a_cols`, column `t % a_cols`.
    let src = |t: usize, o: usize| {
        let (i, j) = (t / a_cols, t % a_cols);
        if rows {
            (i + o) % a_rows * a_cols + j
        } else {
            i * a_cols + (j + o) % a_cols
        }
    };
    if k1 == 0 {
        for (t, dst) in res.blocks_mut().iter_mut().enumerate() {
            module.ckks_copy(dst, &a.blocks()[src(t, q)], scratch)?;
            dst.set_noise(None);
        }
        return Ok(());
    }
    let bytes = |key: &GGLWELayout| shift_tmp_bytes(module, &*res, a, rows, ctx, key);
    if shift_rotates(rows, extent, replicated_at(ctx.slots(), a)) {
        let used = KeyLayouts::new(atk);
        let _ = shift_uses(&*res, a, k, rows, &ctx.maps, &mut |u| ensure_rotation(op, module, &used, u));
        assert_scratch(op, used.max_bytes(bytes), scratch);
        let rotation = (k1 << ctx.log_d()) as i64;
        for (dst, src) in res.blocks_mut().iter_mut().zip(a.blocks()) {
            module.ckks_rotate_into(dst, src, rotation, atk, scratch)?;
        }
        return Ok(());
    }
    let lts = shift_maps(&ctx.maps, k1, rows, extent);
    let (src_k, dst_k, rank) = (a.block_layout().k(), res.block_layout().k(), a.block_layout().rank());
    assert_maps_scratch(op, module, &lts, atk, src_k, dst_k, rank, bytes, scratch);
    let sources: Vec<_> = a.blocks().iter().collect();
    if lts.len() == 1 {
        // One block along the axis: `q = 0`, so block `t` maps into block `t`.
        return BE::blas_map_blocks_impl(module, res.blocks_mut(), &sources, &lts, atk, scratch);
    }
    // ponytail: one hook call over the whole heap grid, two blocks per block of `a`; an override chunks it.
    let layout = res.block_layout();
    let mut halves: Vec<_> = (0..2 * sources.len())
        .map(|_| module.ckks_ciphertext_alloc_from_infos(&layout))
        .collect();
    BE::blas_map_blocks_impl(module, &mut halves, &sources, &lts, atk, scratch)?;
    for (t, dst) in res.blocks_mut().iter_mut().enumerate() {
        module.ckks_add_into(dst, &halves[2 * src(t, q)], &halves[2 * src(t, q + 1) + 1], scratch)?;
    }
    Ok(())
}

/// The provided `blas_sum_rows_tmp_bytes_impl`, as [`reduction_tmp_bytes`].
#[doc(hidden)]
pub fn blas_sum_rows_tmp_bytes_derived<BE, R, A, K>(module: &Module<BE>, res: &R, a: &A, ctx: &BlasContext<BE>, atk: &K) -> usize
where
    BE: BlasMatIntraImpl,
    R: BlasGridInfos,
    A: BlasGridInfos,
    K: GGLWEInfos,
{
    reduction_tmp_bytes(module, res, a, true, ctx, atk)
}

/// The provided `blas_sum_cols_tmp_bytes_impl`, as [`reduction_tmp_bytes`].
#[doc(hidden)]
pub fn blas_sum_cols_tmp_bytes_derived<BE, R, A, K>(module: &Module<BE>, res: &R, a: &A, ctx: &BlasContext<BE>, atk: &K) -> usize
where
    BE: BlasMatIntraImpl,
    R: BlasGridInfos,
    A: BlasGridInfos,
    K: GGLWEInfos,
{
    reduction_tmp_bytes(module, res, a, false, ctx, atk)
}

/// Scratch of an axis reduction along the rows (`rows`) or the columns: a block `u` of `a`'s layout, beside a line's sum
/// into it, with a second block where [`sum_line`] normalizes flag-clear blocks, then the inner sum of `u` into the
/// line's first block of `res`, or its inner sum in place, the hook with the projection and the replicate on a block of
/// the layout the projection leaves, then the copies of that block.
fn reduction_tmp_bytes<BE, R, A, K>(module: &Module<BE>, res: &R, a: &A, rows: bool, ctx: &BlasContext<BE>, atk: &K) -> usize
where
    BE: BlasMatIntraImpl,
    R: BlasGridInfos,
    A: BlasGridInfos,
    K: GGLWEInfos,
{
    let projection = reduction_projection(&ctx.maps, rows, replicated_at(ctx.slots(), a));
    let (res, a) = (res.block_layout(), a.block_layout());
    let sum = sum_line_tmp_bytes(module, &a, lazy_terms(&a));
    let (first, head) = match projection {
        None => (module.ckks_inner_sum_into_tmp_bytes(&res, &a, atk), copied_block(a, res.k())),
        Some(projection) => {
            let projected = mapped_block_into(a, projection, res.k());
            let first = module
                .ckks_inner_sum_assign_tmp_bytes(&a, atk)
                .max(BE::blas_map_blocks_tmp_bytes_impl(module, &res, &a, 1, &[projection], atk))
                .max(module.ckks_replicate_assign_tmp_bytes(&projected, atk));
            (first, projected)
        }
    };
    BE::scratch_aligned(module.glwe_bytes_of_from_infos(&a)) + sum.max(first).max(module.ckks_copy_tmp_bytes(&res, &head))
}

/// Canonical digits per coefficient a flag-clear block holds at most: every operation leaves a sum of two canonical
/// values in one.
const FLAG_CLEAR_TERMS: usize = 2;

/// The terms a lazy sum of ciphertexts of `layout` holds: poulpy-core's `max_lazy_terms` over their coefficients.
fn lazy_terms(layout: &impl GLWEInfos) -> usize {
    max_lazy_terms(
        layout.base2k(),
        layout.n().as_usize() * (layout.rank().as_usize() + 1) * layout.size(),
    )
}

/// The additions after which a sum of inputs of `weight` terms each normalizes, so that it holds `terms` at most: the
/// first `n + 1` inputs, and a normalized sum plus `n` of them; 0 where two inputs already exceed `terms`.
fn lazy_normalize_after(terms: usize, weight: usize) -> usize {
    (terms / weight).saturating_sub(1)
}

/// Scratch of [`sum_line`] of blocks of `layout` at `terms`, whatever their flags.
pub(crate) fn sum_line_tmp_bytes<BE: BlasMatIntraImpl>(module: &Module<BE>, layout: &CKKSLayout, terms: usize) -> usize {
    let sum = module
        .ckks_copy_tmp_bytes(layout, layout)
        .max(module.ckks_add_many_tmp_bytes(layout.size()));
    if lazy_normalize_after(terms, FLAG_CLEAR_TERMS) == 0 {
        BE::scratch_aligned(module.glwe_bytes_of_from_infos(layout)) + sum
    } else {
        sum
    }
}

/// `u` receives the sum of `blocks`, holding `terms` canonical digits per coefficient at most, a flag-clear block
/// counting [`FLAG_CLEAR_TERMS`]: by `ckks_add_many_into`, or, where two blocks exceed `terms`, block by block, each
/// flag-clear block normalized into a scratch block like it and `u` normalized before each addition.
pub(crate) fn sum_line<BE, U, B>(
    module: &Module<BE>,
    u: &mut U,
    blocks: &[&B],
    terms: usize,
    scratch: &mut ScratchArena<'_, BE>,
) -> Result<()>
where
    BE: BlasMatIntraImpl,
    U: GLWEToBackendMut<BE> + CKKSCtBounds + SetCKKSInfos,
    B: GLWEToBackendRef<BE> + CKKSCtBounds,
{
    let weight = if blocks.iter().all(|block| block.is_canonical()) {
        1
    } else {
        FLAG_CLEAR_TERMS
    };
    let normalize_after = lazy_normalize_after(terms, weight);
    if normalize_after > 0 {
        return module.ckks_add_many_into(u, blocks, normalize_after, scratch);
    }
    module.ckks_copy(u, blocks[0], scratch)?;
    for &block in &blocks[1..] {
        if !u.is_canonical() {
            module.glwe_normalize_assign(u, scratch);
        }
        if block.is_canonical() {
            module.ckks_add_assign(u, block, scratch)?;
        } else {
            let (mut v, mut s) = scratch.borrow().take_ckks_ciphertext_like_scratch(block);
            module.glwe_normalize(&mut v, block, &mut s);
            module.ckks_add_assign(u, &v, &mut s)?;
        }
    }
    Ok(())
}

/// The provided `blas_sum_rows_impl`, as [`reduction`].
#[doc(hidden)]
pub fn blas_sum_rows_derived<BE, R, A, K>(
    module: &Module<BE>,
    res: &mut R,
    a: &A,
    ctx: &BlasContext<BE>,
    atk: &K,
    scratch: &mut ScratchArena<'_, BE>,
) -> Result<()>
where
    BE: BlasMatIntraImpl,
    R: BlasGridMut<BE>,
    A: BlasGrid<BE>,
    K: GetAutomorphismKey<BE> + Sync,
{
    reduction(module, res, a, true, ctx, atk, scratch)
}

/// The provided `blas_sum_cols_impl`, as [`reduction`].
#[doc(hidden)]
pub fn blas_sum_cols_derived<BE, R, A, K>(
    module: &Module<BE>,
    res: &mut R,
    a: &A,
    ctx: &BlasContext<BE>,
    atk: &K,
    scratch: &mut ScratchArena<'_, BE>,
) -> Result<()>
where
    BE: BlasMatIntraImpl,
    R: BlasGridMut<BE>,
    A: BlasGrid<BE>,
    K: GetAutomorphismKey<BE> + Sync,
{
    reduction(module, res, a, false, ctx, atk, scratch)
}

/// Sums every matrix along its rows (`rows`) or its columns, one line of `m` blocks at a time, a block column or a block
/// row: a scratch block `u` of `a`'s layout sums the line's blocks ([`sum_line`]). The line's first block is the inner
/// sum of `u` over `d` batches without a projection, else the projection of that inner sum, then its replicate over
/// `d` batches, and is copied into the others.
fn reduction<BE, R, A, K>(
    module: &Module<BE>,
    res: &mut R,
    a: &A,
    rows: bool,
    ctx: &BlasContext<BE>,
    atk: &K,
    scratch: &mut ScratchArena<'_, BE>,
) -> Result<()>
where
    BE: BlasMatIntraImpl,
    R: BlasGridMut<BE>,
    A: BlasGrid<BE>,
    K: GetAutomorphismKey<BE> + Sync,
{
    let (op, batch) = if rows {
        ("blas_sum_rows", validate_blas_sum_rows(module, &*res, a, ctx, atk)?)
    } else {
        ("blas_sum_cols", validate_blas_sum_cols(module, &*res, a, ctx, atk)?)
    };
    let used = KeyLayouts::new(atk);
    let _ = reduction_uses(&*res, a, rows, &ctx.maps, &mut |u| ensure_rotation(op, module, &used, u));
    let bytes = used.max_bytes(|key| reduction_tmp_bytes(module, &*res, a, rows, ctx, key));
    assert_scratch(op, bytes, scratch);
    let a_cols = a.cols();
    let (lines, extent) = if rows { (a_cols, a.rows()) } else { (a.rows(), a_cols) };
    // Block `t` of a line: row `t` of block column `line`, or column `t` of block row `line`.
    let at = |line: usize, t: usize| if rows { t * a_cols + line } else { line * a_cols + t };
    let (log_d, log_batch) = (a.log_d(), if rows { a.log_d() } else { 0 });
    let projection = reduction_projection(&ctx.maps, rows, replicated_at(ctx.slots(), a));
    let terms = lazy_terms(&a.block_layout());
    let (mut u, mut s) = scratch.borrow().take_ckks_ciphertext_like_scratch(&a.block_layout());
    let mut blocks = Vec::with_capacity(extent);
    for line in 0..lines {
        blocks.clear();
        blocks.extend((0..extent).map(|t| &a.blocks()[at(line, t)]));
        sum_line(module, &mut u, &blocks, terms, &mut s)?;
        // The line's first block has the lowest index of its blocks.
        let first = at(line, 0);
        let (head, tail) = res.blocks_mut().split_at_mut(first + 1);
        let head = &mut head[first];
        match projection {
            None => module.ckks_inner_sum_into(head, &u, log_batch, log_d, false, atk, &mut s)?,
            Some(projection) => {
                module.ckks_inner_sum_assign(&mut u, log_batch, log_d, false, atk, &mut s)?;
                let sources = [&u.to_backend_view_ref()];
                BE::blas_map_blocks_impl(module, std::slice::from_mut(head), &sources, &[projection], atk, &mut s)?;
                module.ckks_replicate_assign(head, log_batch, log_d, atk, &mut s)?;
            }
        }
        head.set_noise(None);
        for t in 1..extent {
            let dst = &mut tail[at(line, t) - first - 1];
            module.ckks_copy(dst, &*head, &mut s)?;
            dst.set_noise(None);
        }
    }
    res.set_mat_slots(batch);
    Ok(())
}

/// The provided `blas_matmul_tmp_bytes_impl`: the larger of both hot preparations' scratch and the prepared product's
/// on the layouts they give.
#[doc(hidden)]
pub fn blas_matmul_tmp_bytes_derived<BE, R, A, B, T, K>(
    module: &Module<BE>,
    res: &R,
    a: &A,
    b: &B,
    ctx: &BlasContext<BE>,
    tsk: &T,
    atk: &K,
) -> usize
where
    BE: BlasMatMulImpl,
    R: BlasGridInfos,
    A: BlasGridInfos,
    B: BlasGridInfos,
    T: GGLWEInfos,
    K: GGLWEInfos,
{
    let hot = PrepareHint::OneShot;
    let (left, right) = (
        left_operand_layout(a, &ctx.maps, hot),
        right_operand_layout(b, &ctx.maps, hot),
    );
    BE::blas_matmul_prepare_left_tmp_bytes_impl(module, a, ctx, atk, hot)
        .max(BE::blas_matmul_prepare_right_tmp_bytes_impl(module, b, ctx, atk, hot))
        .max(BE::blas_matmul_prepared_tmp_bytes_impl(
            module, res, &left, &right, ctx, tsk, atk,
        ))
}

/// The provided `blas_matmul_impl`, `res = a b`: both operands allocated hot and prepared into, then their prepared
/// product under `ctx`.
#[allow(clippy::too_many_arguments)]
#[doc(hidden)]
pub fn blas_matmul_derived<BE, R, A, B, T, K>(
    module: &Module<BE>,
    res: &mut R,
    a: &A,
    b: &B,
    ctx: &BlasContext<BE>,
    tsk: &T,
    atk: &K,
    scratch: &mut ScratchArena<'_, BE>,
) -> Result<()>
where
    BE: BlasMatMulImpl,
    R: BlasGridMut<BE>,
    A: BlasGrid<BE>,
    B: BlasGrid<BE>,
    T: GetTensorKey<BE> + Sync,
    K: GetAutomorphismKey<BE> + Sync,
{
    let batch = validate_blas_matmul(module, &*res, a, b, ctx, tsk, atk)?;
    let hot = PrepareHint::OneShot;
    let mut left = module.blas_matmul_left_alloc_from_infos(&left_operand_layout(a, &ctx.maps, hot));
    BE::blas_matmul_prepare_left_impl(module, &mut left, a, ctx, atk, scratch)?;
    let mut right = module.blas_matmul_right_alloc_from_infos(&right_operand_layout(b, &ctx.maps, hot));
    BE::blas_matmul_prepare_right_impl(module, &mut right, b, ctx, atk, scratch)?;
    BE::blas_matmul_prepared_impl(module, res, &left, &right, ctx, tsk, atk, scratch)?;
    res.set_mat_slots(batch);
    Ok(())
}

/// `keys`, recording the layout of every key looked up through it: an operation sizes its scratch query on the keys its
/// checks look up, so a call that looks up no key does not assert its scratch. An implementation runs on keys its
/// validator accepted, so it repeats the validator's look-ups through it only to record their layouts.
pub(crate) struct KeyLayouts<'a, K> {
    keys: &'a K,
    layouts: RefCell<Vec<GGLWELayout>>,
}

impl<'a, K> KeyLayouts<'a, K> {
    pub(crate) fn new(keys: &'a K) -> Self {
        Self {
            keys,
            layouts: RefCell::default(),
        }
    }

    /// The largest `bytes` over the keys looked up, `0` without any.
    pub(crate) fn max_bytes(&self, bytes: impl Fn(&GGLWELayout) -> usize) -> usize {
        self.layouts.borrow().iter().map(bytes).max().unwrap_or(0)
    }

    fn record<G: GGLWEInfos>(&self, key: &G) {
        let layout = key.gglwe_layout();
        let mut layouts = self.layouts.borrow_mut();
        if !layouts.contains(&layout) {
            layouts.push(layout);
        }
    }
}

impl<BE: Backend, K: GetAutomorphismKey<BE>> GetAutomorphismKey<BE> for KeyLayouts<'_, K> {
    fn lookup_automorphism_key(
        &self,
        p: i64,
        k: TorusPrecision,
    ) -> poulpy_core::Result<GLWEAutomorphismKeyPreparedBackendRef<'_, BE>> {
        let key = self.keys.get_automorphism_key(p, k)?;
        self.record(&key);
        Ok(key)
    }
}

impl<BE: Backend, K: GetTensorKey<BE>> GetTensorKey<BE> for KeyLayouts<'_, K> {
    fn lookup_tensor_key(&self, k: TorusPrecision) -> poulpy_core::Result<GLWETensorKeyPreparedBackendRef<'_, BE>> {
        let key = self.keys.get_tensor_key(k)?;
        self.record(&key);
        Ok(key)
    }
}

/// Panics unless `scratch` holds the largest `bytes` over the layouts of the rotation keys that evaluating `lts` from
/// blocks of rank `rank` and width `src_k` into width `dst_k` looks up in `atk`, as [`assert_scratch`] does; a call
/// that looks up none asserts nothing. The keys passed `op`'s validator, so the look-ups only record their layouts.
#[allow(clippy::too_many_arguments)]
pub(crate) fn assert_maps_scratch<BE: Backend, M: CKKSModuleInfos + ?Sized>(
    op: &'static str,
    module: &M,
    lts: &[&LinearTransformationPrepared<BE>],
    atk: &impl GetAutomorphismKey<BE>,
    src_k: TorusPrecision,
    dst_k: TorusPrecision,
    rank: Rank,
    bytes: impl Fn(&GGLWELayout) -> usize,
    scratch: &ScratchArena<'_, BE>,
) {
    let used = KeyLayouts::new(atk);
    for lt in lts {
        let _ = ensure_lt_keys(op, module, lt, &used, src_k, dst_k, rank);
    }
    assert_scratch(op, used.max_bytes(bytes), scratch);
}

/// Fails as [`ensure_rotation`] unless `keys` holds every key of [`lt_uses`], of that rank.
fn ensure_lt_keys<BE, M, P, K>(
    op: &'static str,
    module: &M,
    lt: &LinearTransformation<P>,
    keys: &K,
    src_k: TorusPrecision,
    dst_k: TorusPrecision,
    rank: Rank,
) -> Result<()>
where
    BE: Backend,
    M: CKKSModuleInfos + ?Sized,
    P: LtDiagonalMeta,
    K: GetAutomorphismKey<BE>,
{
    lt_uses(lt, src_k, dst_k, rank, &mut |u| ensure_rotation(op, module, keys, u))
}
