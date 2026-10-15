use poulpy_ckks::{
    CKKSCtBounds, CKKSResult as Result, SetCKKSInfos,
    api::LinearTransformationPrepared,
    oep::{CKKSAddImpl, CKKSCopyImpl},
};
use poulpy_core::layouts::{GGLWEInfos, GLWEToBackendMut, GLWEToBackendRef, GetAutomorphismKey};
use poulpy_hal::layouts::{Backend, Module, ScratchArena};

use super::derived;
use crate::layouts::{BlasContext, BlasGrid, BlasGridInfos, BlasGridMut};

/// Extension point of [`crate::api::BlasMatIntraOps`]: the map hook [`Self::blas_map_blocks_impl`] and its query are
/// required, and the transpose and both shifts are provided over the hook, poulpy-ckks's copy and its addition.
///
/// # Safety
///
/// **The hook.** [`Self::blas_map_blocks_impl`] reproduces
/// [`crate::reference::BlasMatIntraReference::blas_map_blocks_reference`] bit for bit, digits and CKKS metadata: block
/// `t * lts.len() + m` of `dst` receives `lts[m]` evaluated from `src[t]` by poulpy-ckks's
/// `ckks_eval_linear_transformation_into`, then the label (`log_sparsity`) of `src[t]`. It is called with at least one
/// map and `src.len() * lts.len()` blocks of `dst`, the blocks of `dst` sharing one layout and those of `src` another,
/// on inputs the calling operation's validator accepted. It looks up the keys of a map at the widths that evaluation
/// does: the baby steps at the width of `src`, the giant steps at `src`'s width less the maps' scale, capped at
/// `dst`'s. Within the scratch [`Self::blas_map_blocks_tmp_bytes_impl`] reports for the layouts of `dst` and `src`, at
/// their widths whatever their allocations, `src.len()`, `lts` and every key it looks up, it returns `Ok`, or
/// poulpy-ckks's budget error before any write to `dst`; with less scratch it panics before any write, as the reference
/// does when it looks up a key.
///
/// The references of [`crate::oep::BlasMatMulImpl`] call the hook too: a cold preparation block row after block row,
/// and the products for a hot side, a block row of `a` or a block column of `b` at a time; each time sigma (tau) over
/// every block of the row (column), then every shift of every result, into blocks of the terms' width.
///
/// **Free:** the order of the blocks; sharing one baby-step cache across the maps of a source, as the reference does;
/// batching key switches; normalizing a source that is read through a DFT. Anything else changes bits.
///
/// **The provided methods** fix the rest of the circuit, so an implementation that overrides one reproduces it:
/// - `blas_transpose`: `res_ji = transpose(a_ij)`;
/// - `blas_rotate_rows` and `blas_rotate_cols`, with `k mod m d = q d + k'` along a block axis of extent `m`:
///   - `k' = 0`: `ckks_copy` of the block `q` steps along, then a cleared noise tag;
///   - `m = 1`: psi^k' (phi^k') of each block;
///   - otherwise, the block `q` steps along through the first half map and the block `q + 1` steps along through the
///     second, each into a block of `res`'s layout, added by `ckks_add_into` into `res`.
///
/// Every block of `res` then has width `min(res.k, a.k - p)` for the maps' scale `p`, `a`'s scale and label, `a`'s
/// slot kind joined with `Real`, and no noise tag; it is canonical, except after the two half maps, whose addition
/// leaves it unnormalized. A copy (`k' = 0`) has width `min(res.k, a.k)` and `a`'s metadata, as `ckks_copy` gives.
///
/// The shift maps are `Direct`, as the plan records. Each method returns `Ok` within the scratch its `_tmp_bytes_impl`
/// reports, or poulpy-ckks's budget error before any write to `res`, and panics before any write with less scratch when
/// it looks up a key. The provided shifts hold `2 m n` heap blocks across both half maps; an override may chunk them.
///
/// The API runs [`crate::reference::validate_blas_transpose`], [`crate::reference::validate_blas_rotate_rows`] or
/// [`crate::reference::validate_blas_rotate_cols`] first, so a method runs only on inputs its validator accepted. Any
/// `Err` but the budget error is a backend fault (a validator gap or a device failure), and `res` is then unspecified.
pub unsafe trait BlasMatIntraImpl: Backend + CKKSCopyImpl + CKKSAddImpl {
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
        derived::blas_transpose_tmp_bytes(module, res, a, ctx, atk)
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
        derived::blas_transpose(module, res, a, ctx, atk, scratch)
    }

    fn blas_rotate_rows_tmp_bytes_impl<R, A, K>(module: &Module<Self>, res: &R, a: &A, ctx: &BlasContext<Self>, atk: &K) -> usize
    where
        R: BlasGridInfos,
        A: BlasGridInfos,
        K: GGLWEInfos,
    {
        derived::blas_shift_tmp_bytes(module, res, a, true, ctx, atk)
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
        derived::blas_shift(module, "blas_rotate_rows", res, a, k, true, ctx, atk, scratch)
    }

    fn blas_rotate_cols_tmp_bytes_impl<R, A, K>(module: &Module<Self>, res: &R, a: &A, ctx: &BlasContext<Self>, atk: &K) -> usize
    where
        R: BlasGridInfos,
        A: BlasGridInfos,
        K: GGLWEInfos,
    {
        derived::blas_shift_tmp_bytes(module, res, a, false, ctx, atk)
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
        derived::blas_shift(module, "blas_rotate_cols", res, a, k, false, ctx, atk, scratch)
    }
}

/// Selects [`crate::reference::BlasMatIntraReference`] for `$be`: the map hook and its query, under the provided
/// transpose and shifts.
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
