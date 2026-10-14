//! The provided methods of [`BlasMatIntraImpl`], the transpose and the row and column shifts, each composed over the
//! map hook [`BlasMatIntraImpl::blas_map_blocks_impl`] and poulpy-ckks's copy and addition; and of [`BlasMatMulImpl`],
//! `blas_matmul` over both hot preparations and the prepared product.

use poulpy_ckks::{
    CKKSResult as Result,
    api::{CKKSAddOps, CKKSCopyOps},
    layouts::CKKSModuleAlloc,
};
use poulpy_core::layouts::{GGLWEInfos, GGLWELayout, GLWEInfos, GLWEToBackendMut, GetAutomorphismKey, GetTensorKey, LWEInfos};
use poulpy_hal::layouts::{Module, PrepareHint, ScratchArena};

use super::{BlasMatIntraImpl, BlasMatMulImpl};
use crate::{
    layouts::{BlasContext, BlasGrid, BlasGridInfos, BlasGridMut},
    reference::{assert_maps_scratch, blas_matmul_operand_layout, shift_maps},
};

/// The hook's scratch for the transpose of every block of `a` into a block of `res`.
pub(crate) fn blas_transpose_tmp_bytes<BE, R, A, K>(module: &Module<BE>, res: &R, a: &A, ctx: &BlasContext<BE>, atk: &K) -> usize
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

/// `res_ji = transpose(a_ij)`: one hook call, its sources in the order of `res`'s blocks.
pub(crate) fn blas_transpose<BE, R, A, K>(
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
    let lts = [ctx.transpose()];
    let (src_k, dst_k, rank) = (a.block_layout().k(), res.block_layout().k(), a.block_layout().rank());
    let bytes = |key: &GGLWELayout| blas_transpose_tmp_bytes(module, &*res, a, ctx, key);
    assert_maps_scratch("blas_transpose", module, &lts, atk, src_k, dst_k, rank, bytes, scratch);
    let (rows, cols, blocks) = (a.rows(), a.cols(), a.blocks());
    let src: Vec<_> = (0..cols)
        .flat_map(|j| (0..rows).map(move |i| &blocks[i * cols + j]))
        .collect();
    BE::blas_map_blocks_impl(module, res.blocks_mut(), &src, &lts, atk, scratch)
}

/// Scratch of a row (`rows`) or column shift by any `k`: the copy for `k' = 0`, otherwise for every `k'` the hook with
/// psi^k' (phi^k') along one block, or with both half maps plus the addition along several.
pub(crate) fn blas_shift_tmp_bytes<BE, R, A, K>(
    module: &Module<BE>,
    res: &R,
    a: &A,
    rows: bool,
    ctx: &BlasContext<BE>,
    atk: &K,
) -> usize
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
            let lts = shift_maps(ctx, k1, rows, extent);
            let hook = BE::blas_map_blocks_tmp_bytes_impl(module, &res, &a, sources, &lts, atk);
            if lts.len() == 2 {
                hook.max(module.ckks_add_tmp_bytes(res.size()))
            } else {
                hook
            }
        })
        .fold(module.ckks_copy_tmp_bytes(&res, &a), usize::max)
}

/// Shifts every matrix by `k mod m d = q d + k'` rows (`rows`) or columns, along the block axis of extent `m`. With
/// `src(t, o)` the block `o` steps past block `t` along that axis: `k' = 0` copies `src(t, q)` into block `t`, then
/// clears its noise tag; `m = 1` is one hook call with psi^k' (phi^k'); otherwise one hook call evaluates both half maps
/// of every block of `a` into a heap grid `h` of `res`'s block layout, and block `t` is
/// `h[first, src(t, q)] + h[second, src(t, q + 1)]`.
#[allow(clippy::too_many_arguments)]
pub(crate) fn blas_shift<BE, R, A, K>(
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
    let lts = shift_maps(ctx, k1, rows, extent);
    let (src_k, dst_k, rank) = (a.block_layout().k(), res.block_layout().k(), a.block_layout().rank());
    let bytes = |key: &GGLWELayout| blas_shift_tmp_bytes(module, &*res, a, rows, ctx, key);
    assert_maps_scratch(op, module, &lts, atk, src_k, dst_k, rank, bytes, scratch);
    let sources: Vec<_> = a.blocks().iter().collect();
    if lts.len() == 1 {
        // One block along the axis: `q = 0`, so block `t` maps into block `t`.
        return BE::blas_map_blocks_impl(module, res.blocks_mut(), &sources, &lts, atk, scratch);
    }
    // ponytail: the whole 2mn-block heap grid in one hook call; an override chunks it to bound memory.
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

/// The larger of both hot preparations' scratch and the prepared product's on the layouts they give.
pub(crate) fn blas_matmul_tmp_bytes<BE, R, A, B, T, K>(
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
        blas_matmul_operand_layout(a, ctx, hot),
        blas_matmul_operand_layout(b, ctx, hot),
    );
    BE::blas_matmul_prepare_left_tmp_bytes_impl(module, a, ctx, atk, hot)
        .max(BE::blas_matmul_prepare_right_tmp_bytes_impl(module, b, ctx, atk, hot))
        .max(BE::blas_matmul_prepared_tmp_bytes_impl(
            module, res, &left, &right, ctx, tsk, atk,
        ))
}

/// `res = a b`: both operands prepared hot, then their prepared product under `ctx`.
#[allow(clippy::too_many_arguments)]
pub(crate) fn blas_matmul<BE, R, A, B, T, K>(
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
    let left = BE::blas_matmul_prepare_left_impl(module, a, ctx, atk, PrepareHint::OneShot, scratch)?;
    let right = BE::blas_matmul_prepare_right_impl(module, b, ctx, atk, PrepareHint::OneShot, scratch)?;
    BE::blas_matmul_prepared_impl(module, res, &left, &right, ctx, tsk, atk, scratch)
}
