use poulpy_ckks::CKKSResult as Result;
use poulpy_core::layouts::{GGLWEInfos, GetAutomorphismKey};
use poulpy_hal::layouts::{Backend, Module, ScratchArena};

use crate::layouts::{BlasGrid, BlasGridInfos, BlasGridMut};

/// Extension point of [`crate::api::BlasMatInterOps`].
///
/// # Safety
/// Every method reproduces its `_reference` counterpart in
/// [`BlasMatInterReference`](crate::reference::BlasMatInterReference) bit for bit, within the scratch the matching
/// `_tmp_bytes` reports. With less, a call that uses a key panics before it writes `res`, as the reference does.
///
/// A method first runs its validator, named on its doc, before any write, and returns its errors with `res` and its
/// batch record untouched. On inputs it accepted, it returns `Ok`, after stamping `res` with the batch the validator
/// returns ([`crate::layouts::BlasGridMut::set_mat_slots`]), or a budget error: that error is the reference's, the one
/// poulpy-ckks raises, returned before any write to `res`. Any other `Err` is a backend fault (a validator gap or a
/// device failure), and `res` is then unspecified.
///
/// The memory, layout, synchrony and aliasing rules of `poulpy-hal/docs/backend_safety_contract.md` apply.
pub unsafe trait BlasMatInterImpl: Backend {
    /// Scratch bound of [`Self::blas_rotate`]; may report less than the reference, never less than the body takes.
    fn blas_rotate_tmp_bytes<R, A, K>(module: &Module<Self>, res: &R, a: &A, keys: &K) -> usize
    where
        R: BlasGridInfos,
        A: BlasGridInfos,
        K: GGLWEInfos;

    /// Reproduces [`blas_rotate_reference`](crate::reference::BlasMatInterReference::blas_rotate_reference) bit for
    /// bit, within [`Self::blas_rotate_tmp_bytes`]; validator: [`crate::layouts::validate_blas_rotate`]. The circuit is
    /// the one the `# Metadata` section of [`BlasMatInterOps::blas_rotate`](crate::api::BlasMatInterOps::blas_rotate)
    /// states, block by block.
    ///
    /// **Fixed:** one `ckks_rotate_into` per block, a `ckks_copy` with a cleared noise tag when the rotation is a
    /// multiple of the slot count.
    ///
    /// **Free:** the order of the blocks.
    fn blas_rotate<R, A, K>(
        module: &Module<Self>,
        r: i64,
        res: &mut R,
        a: &A,
        keys: &K,
        scratch: &mut ScratchArena<'_, Self>,
    ) -> Result<()>
    where
        R: BlasGridMut<Self>,
        A: BlasGrid<Self>,
        K: GetAutomorphismKey<Self> + Sync;

    /// Scratch bound of [`Self::blas_sum`]; may report less than the reference, never less than the body takes.
    fn blas_sum_tmp_bytes<R, A, K>(module: &Module<Self>, res: &R, a: &A, keys: &K) -> usize
    where
        R: BlasGridInfos,
        A: BlasGridInfos,
        K: GGLWEInfos;

    /// Reproduces [`blas_sum_reference`](crate::reference::BlasMatInterReference::blas_sum_reference) bit for bit,
    /// within [`Self::blas_sum_tmp_bytes`]; validator: [`crate::layouts::validate_blas_sum`]. The circuit is the one
    /// the `# Metadata` section of [`BlasMatInterOps::blas_sum`](crate::api::BlasMatInterOps::blas_sum) states, block
    /// by block.
    ///
    /// **Fixed:** a `ckks_copy` with a cleared noise tag, then for each `i < log2(mat_slots)` in turn a rotation by
    /// `2^i d^2` slots into a block of `res`'s layout, added by `ckks_add_assign`.
    ///
    /// **Free:** the order of the blocks.
    fn blas_sum<R, A, K>(module: &Module<Self>, res: &mut R, a: &A, keys: &K, scratch: &mut ScratchArena<'_, Self>) -> Result<()>
    where
        R: BlasGridMut<Self>,
        A: BlasGrid<Self>,
        K: GetAutomorphismKey<Self> + Sync;
}

/// Selects [`BlasMatInterReference`](crate::reference::BlasMatInterReference) for `$be`.
#[macro_export]
macro_rules! impl_blas_mat_inter_reference_full {
    ($be:ty) => {
        unsafe impl $crate::oep::BlasMatInterImpl for $be {
            fn blas_rotate_tmp_bytes<R, A, K>(module: &::poulpy_hal::layouts::Module<Self>, res: &R, a: &A, keys: &K) -> usize
            where
                R: $crate::layouts::BlasGridInfos,
                A: $crate::layouts::BlasGridInfos,
                K: ::poulpy_core::layouts::GGLWEInfos,
            {
                $crate::reference::BlasMatInterReference::blas_rotate_tmp_bytes_reference(module, res, a, keys)
            }

            fn blas_rotate<R, A, K>(
                module: &::poulpy_hal::layouts::Module<Self>,
                r: i64,
                res: &mut R,
                a: &A,
                keys: &K,
                scratch: &mut ::poulpy_hal::layouts::ScratchArena<'_, Self>,
            ) -> ::poulpy_ckks::CKKSResult<()>
            where
                R: $crate::layouts::BlasGridMut<Self>,
                A: $crate::layouts::BlasGrid<Self>,
                K: ::poulpy_core::layouts::GetAutomorphismKey<Self> + Sync,
            {
                $crate::reference::BlasMatInterReference::blas_rotate_reference(module, r, res, a, keys, scratch)
            }

            fn blas_sum_tmp_bytes<R, A, K>(module: &::poulpy_hal::layouts::Module<Self>, res: &R, a: &A, keys: &K) -> usize
            where
                R: $crate::layouts::BlasGridInfos,
                A: $crate::layouts::BlasGridInfos,
                K: ::poulpy_core::layouts::GGLWEInfos,
            {
                $crate::reference::BlasMatInterReference::blas_sum_tmp_bytes_reference(module, res, a, keys)
            }

            fn blas_sum<R, A, K>(
                module: &::poulpy_hal::layouts::Module<Self>,
                res: &mut R,
                a: &A,
                keys: &K,
                scratch: &mut ::poulpy_hal::layouts::ScratchArena<'_, Self>,
            ) -> ::poulpy_ckks::CKKSResult<()>
            where
                R: $crate::layouts::BlasGridMut<Self>,
                A: $crate::layouts::BlasGrid<Self>,
                K: ::poulpy_core::layouts::GetAutomorphismKey<Self> + Sync,
            {
                $crate::reference::BlasMatInterReference::blas_sum_reference(module, res, a, keys, scratch)
            }
        }
    };
}
