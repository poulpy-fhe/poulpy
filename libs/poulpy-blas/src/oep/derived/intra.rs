//! The provided transpose of [`BlasMatIntraImpl`], composed over the map hook [`BlasMatIntraImpl::blas_map_blocks`]. It
//! runs its validator before any write and, on `Ok`, stamps the batch record of `res`, as the references do.

use poulpy_ckks::CKKSResult as Result;
use poulpy_core::layouts::{GGLWEInfos, GGLWELayout, GLWEInfos, GetAutomorphismKey, GetTensorKey, LWEInfos};
use poulpy_hal::layouts::{Module, ScratchArena};

use crate::{
    layouts::{BlasContext, BlasGrid, BlasGridInfos, BlasGridMut, validate::assert_maps_scratch, validate_blas_transpose},
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
    K: GetAutomorphismKey<BE> + GetTensorKey<BE> + Sync,
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
