//! The provided methods of [`BlasMatIntraImpl`], the transpose and the row and column shifts, each composed over the
//! map hook [`BlasMatIntraImpl::blas_map_blocks`] and poulpy-ckks's copy and addition. Each runs its validator before
//! any write and, on `Ok`, stamps the batch record of `res`, as the references do.

use poulpy_ckks::{
    CKKSResult as Result,
    api::{CKKSAddOps, CKKSCopyOps},
    layouts::CKKSModuleAlloc,
};
use poulpy_core::layouts::{GGLWEInfos, GGLWELayout, GLWEInfos, GLWEToBackendMut, GetAutomorphismKey, LWEInfos};
use poulpy_hal::layouts::{Module, ScratchArena};

use crate::{
    layouts::{
        BlasContext, BlasContextInfos, BlasGrid, BlasGridInfos, BlasGridMut, context::shift_maps, validate::assert_maps_scratch,
        validate_blas_rotate_cols, validate_blas_rotate_rows, validate_blas_transpose,
    },
    oep::BlasMatIntraImpl,
};

/// The provided `blas_transpose_tmp_bytes`: the hook's scratch for the transpose of every block of `a` into a
/// block of `res`.
pub(crate) fn blas_transpose_tmp_bytes_derived<BE, R, A, K>(
    module: &Module<BE>,
    res: &R,
    a: &A,
    ctx: &BlasContext<BE>,
    keys: &K,
) -> usize
where
    BE: BlasMatIntraImpl,
    R: BlasGridInfos,
    A: BlasGridInfos,
    K: GGLWEInfos,
{
    let sources = a.rows() * a.cols();
    BE::blas_map_blocks_tmp_bytes(
        module,
        sources,
        &res.block_layout(),
        &a.block_layout(),
        &[ctx.transpose()],
        keys,
    )
}

/// The provided `blas_transpose`, `res_ji = transpose(a_ij)`: one hook call, its sources in the order of `res`'s
/// blocks.
pub(crate) fn blas_transpose_derived<BE, R, A, K>(
    module: &Module<BE>,
    res: &mut R,
    a: &A,
    ctx: &BlasContext<BE>,
    keys: &K,
    scratch: &mut ScratchArena<'_, BE>,
) -> Result<()>
where
    BE: BlasMatIntraImpl,
    R: BlasGridMut<BE>,
    A: BlasGrid<BE>,
    K: GetAutomorphismKey<BE> + Sync,
{
    let batch = validate_blas_transpose(module, &*res, a, ctx, keys)?;
    let lts = [ctx.transpose()];
    let (src_k, dst_k, rank) = (a.block_layout().k(), res.block_layout().k(), a.block_layout().rank());
    let bytes = |key: &GGLWELayout| blas_transpose_tmp_bytes_derived(module, &*res, a, ctx, key);
    assert_maps_scratch("blas_transpose", module, &lts, keys, src_k, dst_k, rank, bytes, scratch);
    let (rows, cols, blocks) = (a.rows(), a.cols(), a.blocks());
    let src: Vec<_> = (0..cols)
        .flat_map(|j| (0..rows).map(move |i| &blocks[i * cols + j]))
        .collect();
    BE::blas_map_blocks(module, res.blocks_mut(), &src, &lts, keys, scratch)?;
    res.set_mat_slots(batch);
    Ok(())
}

/// The provided `blas_rotate_rows_tmp_bytes`: the scratch of a row shift by any `k`, as [`shift_tmp_bytes`].
pub(crate) fn blas_rotate_rows_tmp_bytes_derived<BE, R, A, K>(
    module: &Module<BE>,
    res: &R,
    a: &A,
    ctx: &BlasContext<BE>,
    keys: &K,
) -> usize
where
    BE: BlasMatIntraImpl,
    R: BlasGridInfos,
    A: BlasGridInfos,
    K: GGLWEInfos,
{
    shift_tmp_bytes(module, res, a, true, ctx, keys)
}

/// The provided `blas_rotate_cols_tmp_bytes`: the scratch of a column shift by any `k`, as [`shift_tmp_bytes`].
pub(crate) fn blas_rotate_cols_tmp_bytes_derived<BE, R, A, K>(
    module: &Module<BE>,
    res: &R,
    a: &A,
    ctx: &BlasContext<BE>,
    keys: &K,
) -> usize
where
    BE: BlasMatIntraImpl,
    R: BlasGridInfos,
    A: BlasGridInfos,
    K: GGLWEInfos,
{
    shift_tmp_bytes(module, res, a, false, ctx, keys)
}

/// Scratch of a row (`rows`) or column shift by any `k`: the copy for `k' = 0`, otherwise for every `k'` the hook with
/// psi^k' (phi^k') along one block, or with both half maps plus the addition along several.
fn shift_tmp_bytes<BE, R, A, K>(module: &Module<BE>, res: &R, a: &A, rows: bool, ctx: &BlasContext<BE>, keys: &K) -> usize
where
    BE: BlasMatIntraImpl,
    R: BlasGridInfos,
    A: BlasGridInfos,
    K: GGLWEInfos,
{
    let extent = if rows { a.rows() } else { a.cols() };
    let (sources, res, a) = (a.rows() * a.cols(), res.block_layout(), a.block_layout());
    (1..1 << ctx.log_d())
        .map(|k1| {
            let lts = shift_maps(&ctx.maps, k1, rows, extent);
            let hook = BE::blas_map_blocks_tmp_bytes(module, sources, &res, &a, &lts, keys);
            if lts.len() == 2 {
                hook.max(module.ckks_add_tmp_bytes(res.size()))
            } else {
                hook
            }
        })
        .fold(module.ckks_copy_tmp_bytes(&res, &a), usize::max)
}

/// The provided `blas_rotate_rows`, as [`shift`].
pub(crate) fn blas_rotate_rows_derived<BE, R, A, K>(
    module: &Module<BE>,
    k: usize,
    res: &mut R,
    a: &A,
    ctx: &BlasContext<BE>,
    keys: &K,
    scratch: &mut ScratchArena<'_, BE>,
) -> Result<()>
where
    BE: BlasMatIntraImpl,
    R: BlasGridMut<BE>,
    A: BlasGrid<BE>,
    K: GetAutomorphismKey<BE> + Sync,
{
    shift(module, "blas_rotate_rows", k, res, a, true, ctx, keys, scratch)
}

/// The provided `blas_rotate_cols`, as [`shift`].
pub(crate) fn blas_rotate_cols_derived<BE, R, A, K>(
    module: &Module<BE>,
    k: usize,
    res: &mut R,
    a: &A,
    ctx: &BlasContext<BE>,
    keys: &K,
    scratch: &mut ScratchArena<'_, BE>,
) -> Result<()>
where
    BE: BlasMatIntraImpl,
    R: BlasGridMut<BE>,
    A: BlasGrid<BE>,
    K: GetAutomorphismKey<BE> + Sync,
{
    shift(module, "blas_rotate_cols", k, res, a, false, ctx, keys, scratch)
}

/// Shifts every matrix by `k mod m d = q d + k'` rows (`rows`) or columns, along the block axis of extent `m`. With
/// `src(t, o)` the block `o` steps past block `t` along that axis: `k' = 0` copies `src(t, q)` into block `t`, then
/// clears its noise tag; `m = 1` is one hook call with psi^k' (phi^k'); otherwise one hook call evaluates both half
/// maps of every block of `a` into a heap grid `h` of `res`'s block layout, and block `t` is
/// `h[first, src(t, q)] + h[second, src(t, q + 1)]`.
#[allow(clippy::too_many_arguments)]
fn shift<BE, R, A, K>(
    module: &Module<BE>,
    op: &'static str,
    k: usize,
    res: &mut R,
    a: &A,
    rows: bool,
    ctx: &BlasContext<BE>,
    keys: &K,
    scratch: &mut ScratchArena<'_, BE>,
) -> Result<()>
where
    BE: BlasMatIntraImpl,
    R: BlasGridMut<BE>,
    A: BlasGrid<BE>,
    K: GetAutomorphismKey<BE> + Sync,
{
    let batch = if rows {
        validate_blas_rotate_rows(module, k, &*res, a, ctx, keys)?
    } else {
        validate_blas_rotate_cols(module, k, &*res, a, ctx, keys)?
    };
    shift_blocks(module, op, k, res, a, rows, ctx, keys, scratch)?;
    res.set_mat_slots(batch);
    Ok(())
}

/// The blocks of [`shift`], on inputs its validator accepted.
#[allow(clippy::too_many_arguments)]
fn shift_blocks<BE, R, A, K>(
    module: &Module<BE>,
    op: &'static str,
    k: usize,
    res: &mut R,
    a: &A,
    rows: bool,
    ctx: &BlasContext<BE>,
    keys: &K,
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
    let lts = shift_maps(&ctx.maps, k1, rows, extent);
    let (src_k, dst_k, rank) = (a.block_layout().k(), res.block_layout().k(), a.block_layout().rank());
    let bytes = |key: &GGLWELayout| shift_tmp_bytes(module, &*res, a, rows, ctx, key);
    assert_maps_scratch(op, module, &lts, keys, src_k, dst_k, rank, bytes, scratch);
    let sources: Vec<_> = a.blocks().iter().collect();
    if lts.len() == 1 {
        // One block along the axis: `q = 0`, so block `t` maps into block `t`.
        return BE::blas_map_blocks(module, res.blocks_mut(), &sources, &lts, keys, scratch);
    }
    // ponytail: one hook call over the whole heap grid, two blocks per block of `a`; an override chunks it.
    let layout = res.block_layout();
    let mut halves: Vec<_> = (0..2 * sources.len())
        .map(|_| module.ckks_ciphertext_alloc_from_infos(&layout))
        .collect();
    BE::blas_map_blocks(module, &mut halves, &sources, &lts, keys, scratch)?;
    for (t, dst) in res.blocks_mut().iter_mut().enumerate() {
        module.ckks_add_into(dst, &halves[2 * src(t, q)], &halves[2 * src(t, q + 1) + 1], scratch)?;
    }
    Ok(())
}
