use poulpy_ckks::{
    CKKSCtBounds, CKKSResult as Result, SetCKKSInfos,
    api::LinearTransformationPrepared,
    oep::{CKKSAddImpl, CKKSInnerSumImpl, CKKSRotateImpl},
};
use poulpy_core::layouts::{GGLWEInfos, GLWEToBackendMut, GLWEToBackendRef, GetAutomorphismKey};
use poulpy_hal::layouts::{Backend, Module, ScratchArena};

use super::derived;
use crate::layouts::{BlasContext, BlasGrid, BlasGridInfos, BlasGridMut};

/// Extension point of [`crate::api::BlasMatIntraOps`]: the map hook [`Self::blas_map_blocks_impl`] and its query are
/// required, and the transpose, both shifts and both axis reductions are provided over the hook and poulpy-ckks's copy,
/// addition and rotation.
///
/// # Safety
///
/// The memory, layout and aliasing rules of `poulpy-hal/docs/backend_safety_contract.md` apply.
///
/// **The hook.** [`Self::blas_map_blocks_impl`] reproduces
/// [`BlasMatIntraReference::blas_map_blocks_reference`](crate::reference::BlasMatIntraReference::blas_map_blocks_reference)
/// bit for bit, digits and CKKS metadata: block `t * lts.len() + m` of `dst` receives `lts[m]` evaluated from `src[t]`
/// by poulpy-ckks's `ckks_eval_linear_transformation_into`, then the label (`log_sparsity`) of `src[t]`. It is called
/// with at least one map and `src.len() * lts.len()` blocks of `dst`, the blocks of `dst` sharing one layout and those
/// of `src` another, on inputs the calling operation's validator accepted. It looks up the keys of a map at the widths
/// that evaluation does: the baby steps at the width of `src`, the giant steps at `src`'s width less the map's scale
/// (`0` for [`BlasContext::rotate_rows_rep`]), capped at `dst`'s. Within the scratch
/// [`Self::blas_map_blocks_tmp_bytes_impl`] reports for the layouts of `dst` and `src`, at their widths whatever their
/// allocations, `src.len()`, `lts` and every key it looks up, it returns `Ok`, or poulpy-ckks's budget error before any
/// write to `dst`; with less scratch it panics before any write, as the reference does when it looks up a key.
///
/// The references of [`crate::oep::BlasMatMulImpl`] call the hook too: a cold preparation block row after block row,
/// and the products for a hot side, a block row of `a` or a block column of `b` at a time; each time sigma (tau) over
/// every block of the row (column), then every shift of every result, into blocks of the terms' width. So do the JKLS
/// products by public matrices, once per unit for their first map: sigma (tau) over a block row (column) of `a`, into
/// blocks of the layout it leaves.
///
/// **Free:** the order of the blocks; sharing one baby-step cache across the maps of a source, as the reference does;
/// batching key switches; normalizing a source that is read through a DFT. Anything else changes bits.
///
/// **The provided methods** fix the rest of the circuit, so an implementation that overrides one reproduces it:
/// - `blas_transpose`: `res_ji = transpose(a_ij)`;
/// - `blas_rotate_rows` and `blas_rotate_cols`, with `k mod m d = q d + k'` along a block axis of extent `m`:
///   - `k' = 0`: `ckks_copy` of the block `q` steps along, then a cleared noise tag;
///   - `m = 1`, rows, on a grid whose label proves one matrix: `ckks_rotate_into` of each block by `k' d` slots;
///   - `m = 1` otherwise: psi^k' (phi^k') of each block;
///   - otherwise, the block `q` steps along through the first half map and the block `q + 1` steps along through the
///     second, each into a block of `res`'s layout, added by `ckks_add_into` into `res`.
/// - `blas_sum_rows` and `blas_sum_cols`, line by line, a block column (row) of `m` blocks at a time, with batches of
///   `b = d` (`1`) slots: a scratch block `u` of `a`'s layout receives the line's blocks summed in order by
///   `ckks_add_many_into`, normalized after every `t / w - 1` additions, for `t` poulpy-core's `max_lazy_terms` over
///   the coefficients of `u` and `w` the terms a block counts, 1 on a line of canonical blocks and 2 otherwise. Where
///   `t / w < 2`, `u` is instead a `ckks_copy` of the first block, then, for each other block in order, normalized in
///   place by poulpy-core's `glwe_normalize_assign` when flag-clear, plus by `ckks_add_assign` that block, first
///   normalized by poulpy-core's `glwe_normalize` into a scratch block of `a`'s layout when flag-clear. Every inner sum
///   and replicate below is over `d` batches of `b` slots, not normalized. Along the rows of a grid whose label proves
///   one matrix, `ckks_inner_sum_into` of `u` into the line's first block of `res`; otherwise `ckks_inner_sum_assign`
///   of `u`, the projection (`first_row`, `first_col`) of `u` through the hook into that block, then
///   `ckks_replicate_assign` of it. Then `ckks_copy` of that block into the other blocks of the line. Every block of
///   `res` then gets a cleared noise tag.
///
/// Every block of `res` then has width `min(res.k, a.k - p)` for the maps' scale `p`, `a`'s scale and label, `a`'s
/// slot kind joined with `Real`, and no noise tag; it is canonical, except after the two half maps, whose addition
/// leaves it unnormalized, and after a reduction, whose last inner sum or replicate does. A copy (`k' = 0`), a
/// rotation and a reduction without a projection have width `min(res.k, a.k)` and `a`'s metadata, as `ckks_copy`,
/// `ckks_rotate_into` and `ckks_inner_sum_into` give.
///
/// The shift maps and the projections are `Direct`, as the plan records. Each method returns `Ok` within the scratch its
/// `_tmp_bytes_impl` reports, or poulpy-ckks's budget error before any write to `res`, and panics before any write with
/// less scratch when it looks up a key. The provided shifts hold two heap blocks per block of `a`, across both half
/// maps; an override may chunk them.
///
/// A method first runs [`crate::layouts::validate_blas_transpose`], [`crate::layouts::validate_blas_rotate_rows`],
/// [`crate::layouts::validate_blas_rotate_cols`], [`crate::layouts::validate_blas_sum_rows`] or
/// [`crate::layouts::validate_blas_sum_cols`], before any write, returns its errors with `res` and its batch record
/// untouched, and after `Ok` stamps `res` with the batch the validator returns
/// ([`crate::layouts::BlasGridMut::set_mat_slots`]), as the provided methods do. Any other `Err` but the budget error
/// is a backend fault (a validator gap or a device failure), and `res` is then unspecified.
pub unsafe trait BlasMatIntraImpl: Backend + CKKSAddImpl + CKKSRotateImpl + CKKSInnerSumImpl {
    /// Scratch bytes of [`Self::blas_map_blocks_impl`] from `sources` blocks of layout `src` into blocks of layout `dst`
    /// through `lts`, with rotation keys of layout `atk`.
    fn blas_map_blocks_tmp_bytes_impl<D, S, K>(
        module: &Module<Self>,
        dst: &D,
        src: &S,
        sources: usize,
        lts: &[&LinearTransformationPrepared<Self>],
        atk: &K,
    ) -> usize
    where
        D: CKKSCtBounds,
        S: CKKSCtBounds,
        K: GGLWEInfos;

    /// The map hook: `dst[t * lts.len() + m] = lts[m](src[t])`, stamped with the label of `src[t]`, in any order. See the
    /// trait's `# Safety`.
    fn blas_map_blocks_impl<D, S, K>(
        module: &Module<Self>,
        dst: &mut [D],
        src: &[&S],
        lts: &[&LinearTransformationPrepared<Self>],
        atk: &K,
        scratch: &mut ScratchArena<'_, Self>,
    ) -> Result<()>
    where
        D: GLWEToBackendMut<Self> + CKKSCtBounds + SetCKKSInfos + Send,
        S: GLWEToBackendRef<Self> + CKKSCtBounds + Sync,
        K: GetAutomorphismKey<Self> + Sync;

    fn blas_transpose_tmp_bytes_impl<R, A, K>(module: &Module<Self>, res: &R, a: &A, ctx: &BlasContext<Self>, atk: &K) -> usize
    where
        R: BlasGridInfos,
        A: BlasGridInfos,
        K: GGLWEInfos,
    {
        derived::blas_transpose_tmp_bytes_derived(module, res, a, ctx, atk)
    }

    fn blas_transpose_impl<R, A, K>(
        module: &Module<Self>,
        res: &mut R,
        a: &A,
        ctx: &BlasContext<Self>,
        atk: &K,
        scratch: &mut ScratchArena<'_, Self>,
    ) -> Result<()>
    where
        R: BlasGridMut<Self>,
        A: BlasGrid<Self>,
        K: GetAutomorphismKey<Self> + Sync,
    {
        derived::blas_transpose_derived(module, res, a, ctx, atk, scratch)
    }

    fn blas_rotate_rows_tmp_bytes_impl<R, A, K>(module: &Module<Self>, res: &R, a: &A, ctx: &BlasContext<Self>, atk: &K) -> usize
    where
        R: BlasGridInfos,
        A: BlasGridInfos,
        K: GGLWEInfos,
    {
        derived::blas_rotate_rows_tmp_bytes_derived(module, res, a, ctx, atk)
    }

    fn blas_rotate_rows_impl<R, A, K>(
        module: &Module<Self>,
        res: &mut R,
        a: &A,
        k: usize,
        ctx: &BlasContext<Self>,
        atk: &K,
        scratch: &mut ScratchArena<'_, Self>,
    ) -> Result<()>
    where
        R: BlasGridMut<Self>,
        A: BlasGrid<Self>,
        K: GetAutomorphismKey<Self> + Sync,
    {
        derived::blas_rotate_rows_derived(module, res, a, k, ctx, atk, scratch)
    }

    fn blas_rotate_cols_tmp_bytes_impl<R, A, K>(module: &Module<Self>, res: &R, a: &A, ctx: &BlasContext<Self>, atk: &K) -> usize
    where
        R: BlasGridInfos,
        A: BlasGridInfos,
        K: GGLWEInfos,
    {
        derived::blas_rotate_cols_tmp_bytes_derived(module, res, a, ctx, atk)
    }

    fn blas_rotate_cols_impl<R, A, K>(
        module: &Module<Self>,
        res: &mut R,
        a: &A,
        k: usize,
        ctx: &BlasContext<Self>,
        atk: &K,
        scratch: &mut ScratchArena<'_, Self>,
    ) -> Result<()>
    where
        R: BlasGridMut<Self>,
        A: BlasGrid<Self>,
        K: GetAutomorphismKey<Self> + Sync,
    {
        derived::blas_rotate_cols_derived(module, res, a, k, ctx, atk, scratch)
    }

    fn blas_sum_rows_tmp_bytes_impl<R, A, K>(module: &Module<Self>, res: &R, a: &A, ctx: &BlasContext<Self>, atk: &K) -> usize
    where
        R: BlasGridInfos,
        A: BlasGridInfos,
        K: GGLWEInfos,
    {
        derived::blas_sum_rows_tmp_bytes_derived(module, res, a, ctx, atk)
    }

    fn blas_sum_rows_impl<R, A, K>(
        module: &Module<Self>,
        res: &mut R,
        a: &A,
        ctx: &BlasContext<Self>,
        atk: &K,
        scratch: &mut ScratchArena<'_, Self>,
    ) -> Result<()>
    where
        R: BlasGridMut<Self>,
        A: BlasGrid<Self>,
        K: GetAutomorphismKey<Self> + Sync,
    {
        derived::blas_sum_rows_derived(module, res, a, ctx, atk, scratch)
    }

    fn blas_sum_cols_tmp_bytes_impl<R, A, K>(module: &Module<Self>, res: &R, a: &A, ctx: &BlasContext<Self>, atk: &K) -> usize
    where
        R: BlasGridInfos,
        A: BlasGridInfos,
        K: GGLWEInfos,
    {
        derived::blas_sum_cols_tmp_bytes_derived(module, res, a, ctx, atk)
    }

    fn blas_sum_cols_impl<R, A, K>(
        module: &Module<Self>,
        res: &mut R,
        a: &A,
        ctx: &BlasContext<Self>,
        atk: &K,
        scratch: &mut ScratchArena<'_, Self>,
    ) -> Result<()>
    where
        R: BlasGridMut<Self>,
        A: BlasGrid<Self>,
        K: GetAutomorphismKey<Self> + Sync,
    {
        derived::blas_sum_cols_derived(module, res, a, ctx, atk, scratch)
    }
}

/// Selects [`BlasMatIntraReference`](crate::reference::BlasMatIntraReference) for `$be`: the map hook and its query,
/// under the provided transpose, shifts and reductions.
#[macro_export]
macro_rules! impl_blas_intra_reference {
    ($be:ty) => {
        unsafe impl $crate::oep::BlasMatIntraImpl for $be {
            fn blas_map_blocks_tmp_bytes_impl<D, S, K>(
                module: &::poulpy_hal::layouts::Module<Self>,
                dst: &D,
                src: &S,
                sources: usize,
                lts: &[&::poulpy_ckks::api::LinearTransformationPrepared<Self>],
                atk: &K,
            ) -> usize
            where
                D: ::poulpy_ckks::CKKSCtBounds,
                S: ::poulpy_ckks::CKKSCtBounds,
                K: ::poulpy_core::layouts::GGLWEInfos,
            {
                $crate::reference::BlasMatIntraReference::blas_map_blocks_tmp_bytes_reference(module, dst, src, sources, lts, atk)
            }

            fn blas_map_blocks_impl<D, S, K>(
                module: &::poulpy_hal::layouts::Module<Self>,
                dst: &mut [D],
                src: &[&S],
                lts: &[&::poulpy_ckks::api::LinearTransformationPrepared<Self>],
                atk: &K,
                scratch: &mut ::poulpy_hal::layouts::ScratchArena<'_, Self>,
            ) -> ::poulpy_ckks::CKKSResult<()>
            where
                D: ::poulpy_core::layouts::GLWEToBackendMut<Self>
                    + ::poulpy_ckks::CKKSCtBounds
                    + ::poulpy_ckks::SetCKKSInfos
                    + Send,
                S: ::poulpy_core::layouts::GLWEToBackendRef<Self> + ::poulpy_ckks::CKKSCtBounds + Sync,
                K: ::poulpy_core::layouts::GetAutomorphismKey<Self> + Sync,
            {
                $crate::reference::BlasMatIntraReference::blas_map_blocks_reference(module, dst, src, lts, atk, scratch)
            }
        }
    };
}
