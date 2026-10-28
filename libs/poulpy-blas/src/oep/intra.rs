use poulpy_ckks::{CKKSCtBounds, CKKSResult as Result, SetCKKSInfos, api::LinearTransformationPrepared};
use poulpy_core::layouts::{GGLWEInfos, GLWEToBackendMut, GLWEToBackendRef, GetAutomorphismKey};
use poulpy_hal::layouts::{Backend, Module, ScratchArena};

use super::derived::intra::{blas_transpose_derived, blas_transpose_tmp_bytes_derived};
use crate::layouts::{BlasContext, BlasGrid, BlasGridInfos, BlasGridMut};

/// Extension point of [`crate::api::BlasMatIntraOps`]: the map hook [`Self::blas_map_blocks`], both shifts and both
/// axis reductions are required, and the transpose is provided over the hook.
///
/// # Safety
/// Each method reproduces the circuit its doc states bit for bit, digits and CKKS metadata, within the scratch its
/// `_tmp_bytes` reports, and with less scratch panics before any write when it looks up a key, as the reference does.
///
/// A method other than the hook first runs its validator, named on its doc, before any write, and returns its errors
/// with `res` and its batch record untouched. On inputs it accepted, it returns `Ok`, after stamping `res` with the
/// batch the validator returns ([`crate::layouts::BlasGridMut::set_mat_slots`]), or poulpy-ckks's budget error before
/// any write to `res`. Any other `Err` is a backend fault (a validator gap or a device failure), and `res` is then
/// unspecified.
///
/// The memory, layout, synchrony and aliasing rules of `poulpy-hal/docs/backend_safety_contract.md` apply.
pub unsafe trait BlasMatIntraImpl: Backend {
    /// Scratch bound of [`Self::blas_map_blocks`] from `sources` blocks of layout `src` into blocks of layout `dst`
    /// through `lts`, with rotation keys of layout `keys`; may report less than the reference, never less than the body
    /// takes. It reports for the layouts of `dst` and `src` at their widths whatever their allocations, for `sources`,
    /// `lts` and every key the hook looks up.
    fn blas_map_blocks_tmp_bytes<D, S, K>(
        module: &Module<Self>,
        sources: usize,
        dst: &D,
        src: &S,
        lts: &[&LinearTransformationPrepared<Self>],
        keys: &K,
    ) -> usize
    where
        D: CKKSCtBounds,
        S: CKKSCtBounds,
        K: GGLWEInfos;

    /// The map hook. Reproduces
    /// [`blas_map_blocks_reference`](crate::reference::BlasMatIntraReference::blas_map_blocks_reference) bit for bit,
    /// digits and CKKS metadata, within [`Self::blas_map_blocks_tmp_bytes`]: block `t * lts.len() + m` of `dst`
    /// receives `lts[m]` evaluated from `src[t]` by poulpy-ckks's `ckks_eval_linear_transformation_into`, then the
    /// label (`log_sparsity`) of `src[t]`, in any order. It looks up the keys of a map at the widths that evaluation
    /// does: the baby steps at the width of `src`, the giant steps at `src`'s width less the map's scale (`0` for
    /// [`BlasContext::rotate_rows_rep`]), capped at `dst`'s. Within that scratch it returns `Ok`, or poulpy-ckks's
    /// budget error before any write to `dst`; with less scratch it panics before any write, as the reference does when
    /// it looks up a key.
    ///
    /// It is called with at least one map and `src.len() * lts.len()` blocks of `dst`, the blocks of `dst` sharing one
    /// layout and those of `src` another, on inputs the calling operation's validator accepted.
    ///
    /// The references of [`crate::oep::BlasMatMulImpl`] call the hook too: a cold preparation block row after block
    /// row, and the products for a hot side, a block row of `a` or a block column of `b` at a time; each time sigma
    /// (tau) over every block of the row (column), then every shift of every result, into blocks of the terms' width.
    /// So do the JKLS products by public matrices under `Jkls`, once per unit for their first map: sigma (tau) over a
    /// block row (column) of `a`, into blocks of the layout it leaves; and under `JklsStored` and `JklsGalois` the
    /// reference of [`crate::oep::BlasJklsImpl::blas_jkls_pt_unit`], twice per source block: the first map of the block,
    /// then the context's masks of that result.
    ///
    /// **Fixed:** the circuit above; anything not listed as free changes bits.
    ///
    /// **Free:** the order of the blocks; sharing one baby-step cache across the maps of a source, as the reference
    /// does; batching key switches; normalizing a source that is read through a DFT.
    fn blas_map_blocks<D, S, K>(
        module: &Module<Self>,
        dst: &mut [D],
        src: &[&S],
        lts: &[&LinearTransformationPrepared<Self>],
        keys: &K,
        scratch: &mut ScratchArena<'_, Self>,
    ) -> Result<()>
    where
        D: GLWEToBackendMut<Self> + CKKSCtBounds + SetCKKSInfos + Send,
        S: GLWEToBackendRef<Self> + CKKSCtBounds + Sync,
        K: GetAutomorphismKey<Self> + Sync;

    /// Default body: `blas_transpose_tmp_bytes_derived`, [`Self::blas_map_blocks_tmp_bytes`] for the transpose of every
    /// block of `a` into a block of `res`. Scratch bound of [`Self::blas_transpose`]; may report less than the default
    /// body, never less than the body takes.
    fn blas_transpose_tmp_bytes<R, A, K>(module: &Module<Self>, res: &R, a: &A, ctx: &BlasContext<Self>, keys: &K) -> usize
    where
        R: BlasGridInfos,
        A: BlasGridInfos,
        K: GGLWEInfos,
    {
        blas_transpose_tmp_bytes_derived(module, res, a, ctx, keys)
    }

    /// Default body: `blas_transpose_derived`, the validator [`crate::layouts::validate_blas_transpose`], then one
    /// [`Self::blas_map_blocks`] call over the transpose map with its sources in the order of `res`'s blocks, and the
    /// batch stamp: `res_ji = transpose(a_ij)`. The map is `Direct`, as the plan records. Every block of `res` then has
    /// width `min(res.k, a.k - p)` for the map's scale `p`, `a`'s scale and label, `a`'s slot kind joined with `Real`,
    /// and no noise tag; it is canonical.
    ///
    /// **Fixed:** the circuit above; anything not listed as free changes bits.
    ///
    /// **Free:** the order of the blocks the hook evaluates.
    fn blas_transpose<R, A, K>(
        module: &Module<Self>,
        res: &mut R,
        a: &A,
        ctx: &BlasContext<Self>,
        keys: &K,
        scratch: &mut ScratchArena<'_, Self>,
    ) -> Result<()>
    where
        R: BlasGridMut<Self>,
        A: BlasGrid<Self>,
        K: GetAutomorphismKey<Self> + Sync,
    {
        blas_transpose_derived(module, res, a, ctx, keys, scratch)
    }

    /// Scratch bound of [`Self::blas_rotate_rows`] by any `k`; may report less than
    /// [`blas_rotate_rows_tmp_bytes_reference`](crate::reference::BlasMatIntraReference::blas_rotate_rows_tmp_bytes_reference),
    /// never less than the body takes.
    fn blas_rotate_rows_tmp_bytes<R, A, K>(module: &Module<Self>, res: &R, a: &A, ctx: &BlasContext<Self>, keys: &K) -> usize
    where
        R: BlasGridInfos,
        A: BlasGridInfos,
        K: GGLWEInfos;

    /// Reproduces [`blas_rotate_rows_reference`](crate::reference::BlasMatIntraReference::blas_rotate_rows_reference)
    /// bit for bit, within [`Self::blas_rotate_rows_tmp_bytes`]: the validator
    /// [`crate::layouts::validate_blas_rotate_rows`], then a shift of every matrix by `k` rows, `k mod m d = q d + k'`,
    /// along the block axis of extent `m`, and the batch stamp. The shift maps are `Direct`, as the plan records.
    ///
    /// **Fixed:**
    ///
    /// - `k' = 0`: `ckks_copy` of the block `q` steps along, then a cleared noise tag;
    /// - `m = 1` on a grid whose label proves one matrix: `ckks_rotate_into` of each block by `k' d` slots;
    /// - `m = 1` otherwise: psi^k' of each block, one [`Self::blas_map_blocks`] call;
    /// - otherwise, one [`Self::blas_map_blocks`] call evaluates both half maps of every block of `a` into a heap grid
    ///   of two blocks of `res`'s layout per block of `a`, and block `t` of `res` is one `ckks_add_into` of the first
    ///   half map of the block `q` steps along and the second half map of the block `q + 1` steps along.
    ///
    /// Every block of `res` then has width `min(res.k, a.k - p)` for the maps' scale `p`, `a`'s scale and label, `a`'s
    /// slot kind joined with `Real`, and no noise tag; it is canonical, except after the two half maps, whose addition
    /// leaves it unnormalized. A copy (`k' = 0`) and a rotation have width `min(res.k, a.k)` and `a`'s metadata, as
    /// `ckks_copy` and `ckks_rotate_into` give.
    ///
    /// **Free:** chunking the heap blocks, which the reference holds two of per block of `a`, across both half
    /// maps.
    fn blas_rotate_rows<R, A, K>(
        module: &Module<Self>,
        k: usize,
        res: &mut R,
        a: &A,
        ctx: &BlasContext<Self>,
        keys: &K,
        scratch: &mut ScratchArena<'_, Self>,
    ) -> Result<()>
    where
        R: BlasGridMut<Self>,
        A: BlasGrid<Self>,
        K: GetAutomorphismKey<Self> + Sync;

    /// Scratch bound of [`Self::blas_rotate_cols`] by any `k`; may report less than
    /// [`blas_rotate_cols_tmp_bytes_reference`](crate::reference::BlasMatIntraReference::blas_rotate_cols_tmp_bytes_reference),
    /// never less than the body takes.
    fn blas_rotate_cols_tmp_bytes<R, A, K>(module: &Module<Self>, res: &R, a: &A, ctx: &BlasContext<Self>, keys: &K) -> usize
    where
        R: BlasGridInfos,
        A: BlasGridInfos,
        K: GGLWEInfos;

    /// Reproduces [`blas_rotate_cols_reference`](crate::reference::BlasMatIntraReference::blas_rotate_cols_reference)
    /// bit for bit, within [`Self::blas_rotate_cols_tmp_bytes`]: the validator
    /// [`crate::layouts::validate_blas_rotate_cols`], then a shift of every matrix by `k` columns, `k mod m d = q d +
    /// k'`, along the block axis of extent `m`, and the batch stamp. The shift maps are `Direct`, as the plan records.
    ///
    /// **Fixed:**
    ///
    /// - `k' = 0`: `ckks_copy` of the block `q` steps along, then a cleared noise tag;
    /// - `m = 1`: phi^k' of each block, one [`Self::blas_map_blocks`] call;
    /// - otherwise, one [`Self::blas_map_blocks`] call evaluates both half maps of every block of `a` into a heap grid
    ///   of two blocks of `res`'s layout per block of `a`, and block `t` of `res` is one `ckks_add_into` of the first
    ///   half map of the block `q` steps along and the second half map of the block `q + 1` steps along.
    ///
    /// Every block of `res` then has width `min(res.k, a.k - p)` for the maps' scale `p`, `a`'s scale and label, `a`'s
    /// slot kind joined with `Real`, and no noise tag; it is canonical, except after the two half maps, whose addition
    /// leaves it unnormalized. A copy (`k' = 0`) has width `min(res.k, a.k)` and `a`'s metadata, as `ckks_copy` gives.
    ///
    /// **Free:** chunking the heap blocks, which the reference holds two of per block of `a`, across both half
    /// maps.
    fn blas_rotate_cols<R, A, K>(
        module: &Module<Self>,
        k: usize,
        res: &mut R,
        a: &A,
        ctx: &BlasContext<Self>,
        keys: &K,
        scratch: &mut ScratchArena<'_, Self>,
    ) -> Result<()>
    where
        R: BlasGridMut<Self>,
        A: BlasGrid<Self>,
        K: GetAutomorphismKey<Self> + Sync;

    /// Scratch bound of [`Self::blas_sum_rows`]; may report less than
    /// [`blas_sum_rows_tmp_bytes_reference`](crate::reference::BlasMatIntraReference::blas_sum_rows_tmp_bytes_reference),
    /// never less than the body takes.
    fn blas_sum_rows_tmp_bytes<R, A, K>(module: &Module<Self>, res: &R, a: &A, ctx: &BlasContext<Self>, keys: &K) -> usize
    where
        R: BlasGridInfos,
        A: BlasGridInfos,
        K: GGLWEInfos;

    /// Reproduces [`blas_sum_rows_reference`](crate::reference::BlasMatIntraReference::blas_sum_rows_reference) bit for
    /// bit, within [`Self::blas_sum_rows_tmp_bytes`]: the validator [`crate::layouts::validate_blas_sum_rows`], then
    /// line by line, a block column of `m` blocks at a time with batches of `b = d` slots, and the batch stamp.
    ///
    /// **Fixed:** a scratch block `u` of `a`'s layout receives the line's blocks summed in order by `ckks_add_many`,
    /// normalized after every `t / w - 1` additions, for `t` poulpy-core's `max_lazy_terms` over the coefficients of `u`
    /// and `w` the terms a block counts, 1 on a line of canonical blocks and 2 otherwise. Where `t / w < 2`, `u` is
    /// instead a `ckks_copy` of the first block, then, for each other block in order, normalized in place by
    /// poulpy-core's `glwe_normalize_assign` when flag-clear, plus by `ckks_add_assign` that block, first normalized by
    /// poulpy-core's `glwe_normalize` into a scratch block of `a`'s layout when flag-clear. A flag-clear block of `a`
    /// holds at most two canonical digits per coefficient, a caller obligation left unchecked. Every inner sum and
    /// replicate is over `d` batches of `b` slots, with `normalize` clear. On a grid whose label proves one matrix,
    /// `ckks_inner_sum` of `u` into the line's first block of `res`; otherwise `ckks_inner_sum_assign` of `u`, the
    /// projection `first_row` of `u` through [`Self::blas_map_blocks`] into that block, then `ckks_replicate_assign` of
    /// it. Then `ckks_copy` of that block into the other blocks of the line. Every block of `res` then gets a cleared
    /// noise tag, has width `min(res.k, a.k - p)` for the projection's scale `p` (`min(res.k, a.k)` without one) and is
    /// canonical. The projection is `Direct`, as the plan records.
    ///
    /// **Free:** the order of the lines.
    fn blas_sum_rows<R, A, K>(
        module: &Module<Self>,
        res: &mut R,
        a: &A,
        ctx: &BlasContext<Self>,
        keys: &K,
        scratch: &mut ScratchArena<'_, Self>,
    ) -> Result<()>
    where
        R: BlasGridMut<Self>,
        A: BlasGrid<Self>,
        K: GetAutomorphismKey<Self> + Sync;

    /// Scratch bound of [`Self::blas_sum_cols`]; may report less than
    /// [`blas_sum_cols_tmp_bytes_reference`](crate::reference::BlasMatIntraReference::blas_sum_cols_tmp_bytes_reference),
    /// never less than the body takes.
    fn blas_sum_cols_tmp_bytes<R, A, K>(module: &Module<Self>, res: &R, a: &A, ctx: &BlasContext<Self>, keys: &K) -> usize
    where
        R: BlasGridInfos,
        A: BlasGridInfos,
        K: GGLWEInfos;

    /// Reproduces [`blas_sum_cols_reference`](crate::reference::BlasMatIntraReference::blas_sum_cols_reference) bit for
    /// bit, within [`Self::blas_sum_cols_tmp_bytes`]: the validator [`crate::layouts::validate_blas_sum_cols`], then as
    /// [`Self::blas_sum_rows`] along a block row at a time, with batches of `b = 1` slot, the projection `first_col`
    /// and no case without a projection.
    ///
    /// **Fixed** and **Free:** as [`Self::blas_sum_rows`], along the block rows.
    fn blas_sum_cols<R, A, K>(
        module: &Module<Self>,
        res: &mut R,
        a: &A,
        ctx: &BlasContext<Self>,
        keys: &K,
        scratch: &mut ScratchArena<'_, Self>,
    ) -> Result<()>
    where
        R: BlasGridMut<Self>,
        A: BlasGrid<Self>,
        K: GetAutomorphismKey<Self> + Sync;
}

/// Selects [`BlasMatIntraReference`](crate::reference::BlasMatIntraReference) for `$be`: the map hook, the shifts, the
/// axis reductions and their queries, under the provided transpose.
#[macro_export]
macro_rules! impl_blas_mat_intra_reference_full {
    ($be:ty) => {
        unsafe impl $crate::oep::BlasMatIntraImpl for $be {
            fn blas_map_blocks_tmp_bytes<D, S, K>(
                module: &::poulpy_hal::layouts::Module<Self>,
                sources: usize,
                dst: &D,
                src: &S,
                lts: &[&::poulpy_ckks::api::LinearTransformationPrepared<Self>],
                keys: &K,
            ) -> usize
            where
                D: ::poulpy_ckks::CKKSCtBounds,
                S: ::poulpy_ckks::CKKSCtBounds,
                K: ::poulpy_core::layouts::GGLWEInfos,
            {
                $crate::reference::BlasMatIntraReference::blas_map_blocks_tmp_bytes_reference(
                    module, sources, dst, src, lts, keys,
                )
            }

            fn blas_map_blocks<D, S, K>(
                module: &::poulpy_hal::layouts::Module<Self>,
                dst: &mut [D],
                src: &[&S],
                lts: &[&::poulpy_ckks::api::LinearTransformationPrepared<Self>],
                keys: &K,
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
                $crate::reference::BlasMatIntraReference::blas_map_blocks_reference(module, dst, src, lts, keys, scratch)
            }

            fn blas_rotate_rows_tmp_bytes<R, A, K>(
                module: &::poulpy_hal::layouts::Module<Self>,
                res: &R,
                a: &A,
                ctx: &$crate::layouts::BlasContext<Self>,
                keys: &K,
            ) -> usize
            where
                R: $crate::layouts::BlasGridInfos,
                A: $crate::layouts::BlasGridInfos,
                K: ::poulpy_core::layouts::GGLWEInfos,
            {
                $crate::reference::BlasMatIntraReference::blas_rotate_rows_tmp_bytes_reference(module, res, a, ctx, keys)
            }

            fn blas_rotate_rows<R, A, K>(
                module: &::poulpy_hal::layouts::Module<Self>,
                k: usize,
                res: &mut R,
                a: &A,
                ctx: &$crate::layouts::BlasContext<Self>,
                keys: &K,
                scratch: &mut ::poulpy_hal::layouts::ScratchArena<'_, Self>,
            ) -> ::poulpy_ckks::CKKSResult<()>
            where
                R: $crate::layouts::BlasGridMut<Self>,
                A: $crate::layouts::BlasGrid<Self>,
                K: ::poulpy_core::layouts::GetAutomorphismKey<Self> + Sync,
            {
                $crate::reference::BlasMatIntraReference::blas_rotate_rows_reference(module, k, res, a, ctx, keys, scratch)
            }

            fn blas_rotate_cols_tmp_bytes<R, A, K>(
                module: &::poulpy_hal::layouts::Module<Self>,
                res: &R,
                a: &A,
                ctx: &$crate::layouts::BlasContext<Self>,
                keys: &K,
            ) -> usize
            where
                R: $crate::layouts::BlasGridInfos,
                A: $crate::layouts::BlasGridInfos,
                K: ::poulpy_core::layouts::GGLWEInfos,
            {
                $crate::reference::BlasMatIntraReference::blas_rotate_cols_tmp_bytes_reference(module, res, a, ctx, keys)
            }

            fn blas_rotate_cols<R, A, K>(
                module: &::poulpy_hal::layouts::Module<Self>,
                k: usize,
                res: &mut R,
                a: &A,
                ctx: &$crate::layouts::BlasContext<Self>,
                keys: &K,
                scratch: &mut ::poulpy_hal::layouts::ScratchArena<'_, Self>,
            ) -> ::poulpy_ckks::CKKSResult<()>
            where
                R: $crate::layouts::BlasGridMut<Self>,
                A: $crate::layouts::BlasGrid<Self>,
                K: ::poulpy_core::layouts::GetAutomorphismKey<Self> + Sync,
            {
                $crate::reference::BlasMatIntraReference::blas_rotate_cols_reference(module, k, res, a, ctx, keys, scratch)
            }

            fn blas_sum_rows_tmp_bytes<R, A, K>(
                module: &::poulpy_hal::layouts::Module<Self>,
                res: &R,
                a: &A,
                ctx: &$crate::layouts::BlasContext<Self>,
                keys: &K,
            ) -> usize
            where
                R: $crate::layouts::BlasGridInfos,
                A: $crate::layouts::BlasGridInfos,
                K: ::poulpy_core::layouts::GGLWEInfos,
            {
                $crate::reference::BlasMatIntraReference::blas_sum_rows_tmp_bytes_reference(module, res, a, ctx, keys)
            }

            fn blas_sum_rows<R, A, K>(
                module: &::poulpy_hal::layouts::Module<Self>,
                res: &mut R,
                a: &A,
                ctx: &$crate::layouts::BlasContext<Self>,
                keys: &K,
                scratch: &mut ::poulpy_hal::layouts::ScratchArena<'_, Self>,
            ) -> ::poulpy_ckks::CKKSResult<()>
            where
                R: $crate::layouts::BlasGridMut<Self>,
                A: $crate::layouts::BlasGrid<Self>,
                K: ::poulpy_core::layouts::GetAutomorphismKey<Self> + Sync,
            {
                $crate::reference::BlasMatIntraReference::blas_sum_rows_reference(module, res, a, ctx, keys, scratch)
            }

            fn blas_sum_cols_tmp_bytes<R, A, K>(
                module: &::poulpy_hal::layouts::Module<Self>,
                res: &R,
                a: &A,
                ctx: &$crate::layouts::BlasContext<Self>,
                keys: &K,
            ) -> usize
            where
                R: $crate::layouts::BlasGridInfos,
                A: $crate::layouts::BlasGridInfos,
                K: ::poulpy_core::layouts::GGLWEInfos,
            {
                $crate::reference::BlasMatIntraReference::blas_sum_cols_tmp_bytes_reference(module, res, a, ctx, keys)
            }

            fn blas_sum_cols<R, A, K>(
                module: &::poulpy_hal::layouts::Module<Self>,
                res: &mut R,
                a: &A,
                ctx: &$crate::layouts::BlasContext<Self>,
                keys: &K,
                scratch: &mut ::poulpy_hal::layouts::ScratchArena<'_, Self>,
            ) -> ::poulpy_ckks::CKKSResult<()>
            where
                R: $crate::layouts::BlasGridMut<Self>,
                A: $crate::layouts::BlasGrid<Self>,
                K: ::poulpy_core::layouts::GetAutomorphismKey<Self> + Sync,
            {
                $crate::reference::BlasMatIntraReference::blas_sum_cols_reference(module, res, a, ctx, keys, scratch)
            }
        }
    };
}
