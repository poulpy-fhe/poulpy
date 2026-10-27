use crate::CKKSResult as Result;

use poulpy_core::layouts::{Degree, GGLWEInfos, GLWEToBackendMut, GLWEToBackendRef, GetAutomorphismKey};
use poulpy_hal::layouts::{Backend, Module, ScratchArena};

use crate::{CKKSCtBounds, SetCKKSInfos};

/// Backend contract of [`CKKSInnerSumOps`](crate::api::CKKSInnerSumOps).
///
/// # Safety
/// Implementations reproduce the bodies their methods document bit for bit, report for each
/// `_tmp_bytes` query a bound that covers the body, and refuse as the reference does. The memory,
/// layout, synchrony and aliasing rules of `poulpy-hal/docs/backend_safety_contract.md` apply.
pub unsafe trait CKKSInnerSumImpl: Backend {
    /// Returns
    /// [`ckks_inner_sum_galois_elements_reference`](crate::reference::inner_sum::CKKSInnerSumReference::ckks_inner_sum_galois_elements_reference),
    /// panicking on the same windows.
    fn ckks_inner_sum_galois_elements(module: &Module<Self>, n: Degree, log_batch: usize, log_count: usize) -> Vec<i64>;

    /// Returns
    /// [`ckks_replicate_galois_elements_reference`](crate::reference::inner_sum::CKKSInnerSumReference::ckks_replicate_galois_elements_reference),
    /// panicking on the same windows.
    fn ckks_replicate_galois_elements(module: &Module<Self>, n: Degree, log_batch: usize, log_count: usize) -> Vec<i64>;

    /// Scratch bound of the four sums of this trait; may report less than the reference, never less than the body
    /// takes.
    fn ckks_inner_sum_tmp_bytes<R, A, K>(module: &Module<Self>, res: &R, a: &A, key: &K) -> usize
    where
        R: CKKSCtBounds,
        A: CKKSCtBounds,
        K: GGLWEInfos;

    /// Reproduces
    /// [`ckks_inner_sum_reference`](crate::reference::inner_sum::CKKSInnerSumReference::ckks_inner_sum_reference) bit
    /// for bit, within [`Self::ckks_inner_sum_tmp_bytes`]; free in kernels and scratch use.
    #[expect(
        clippy::too_many_arguments,
        reason = "the sum's operands, batch, count and normalization, keys and scratch"
    )]
    fn ckks_inner_sum<R, A, H>(
        module: &Module<Self>,
        log_batch: usize,
        log_count: usize,
        normalize: bool,
        res: &mut R,
        a: &A,
        keys: &H,
        scratch: &mut ScratchArena<'_, Self>,
    ) -> Result<()>
    where
        R: GLWEToBackendMut<Self> + CKKSCtBounds + SetCKKSInfos,
        A: GLWEToBackendRef<Self> + CKKSCtBounds,
        H: GetAutomorphismKey<Self>;

    /// Reproduces
    /// [`ckks_inner_sum_assign_reference`](crate::reference::inner_sum::CKKSInnerSumReference::ckks_inner_sum_assign_reference)
    /// bit for bit, within [`Self::ckks_inner_sum_tmp_bytes`]; free in kernels and scratch use.
    fn ckks_inner_sum_assign<R, H>(
        module: &Module<Self>,
        log_batch: usize,
        log_count: usize,
        normalize: bool,
        res: &mut R,
        keys: &H,
        scratch: &mut ScratchArena<'_, Self>,
    ) -> Result<()>
    where
        R: GLWEToBackendMut<Self> + CKKSCtBounds + SetCKKSInfos,
        H: GetAutomorphismKey<Self>;

    /// Reproduces
    /// [`ckks_replicate_reference`](crate::reference::inner_sum::CKKSInnerSumReference::ckks_replicate_reference) bit
    /// for bit, within [`Self::ckks_inner_sum_tmp_bytes`]; free in kernels and scratch use.
    fn ckks_replicate<R, A, H>(
        module: &Module<Self>,
        log_batch: usize,
        log_count: usize,
        res: &mut R,
        a: &A,
        keys: &H,
        scratch: &mut ScratchArena<'_, Self>,
    ) -> Result<()>
    where
        R: GLWEToBackendMut<Self> + CKKSCtBounds + SetCKKSInfos,
        A: GLWEToBackendRef<Self> + CKKSCtBounds,
        H: GetAutomorphismKey<Self>;

    /// Reproduces
    /// [`ckks_replicate_assign_reference`](crate::reference::inner_sum::CKKSInnerSumReference::ckks_replicate_assign_reference)
    /// bit for bit, within [`Self::ckks_inner_sum_tmp_bytes`]; free in kernels and scratch use.
    fn ckks_replicate_assign<R, H>(
        module: &Module<Self>,
        log_batch: usize,
        log_count: usize,
        res: &mut R,
        keys: &H,
        scratch: &mut ScratchArena<'_, Self>,
    ) -> Result<()>
    where
        R: GLWEToBackendMut<Self> + CKKSCtBounds + SetCKKSInfos,
        H: GetAutomorphismKey<Self>;
}

/// Selects the reference implementation of `CKKSInnerSumImpl` for `$be`.
#[macro_export]
macro_rules! impl_ckks_inner_sum_reference_full {
    ($be:ty) => {
        unsafe impl $crate::oep::CKKSInnerSumImpl for $be {
            fn ckks_inner_sum_galois_elements(
                module: &::poulpy_hal::layouts::Module<Self>,
                n: ::poulpy_core::layouts::Degree,
                log_batch: usize,
                log_count: usize,
            ) -> ::std::vec::Vec<i64> {
                $crate::reference::inner_sum::CKKSInnerSumReference::ckks_inner_sum_galois_elements_reference(
                    module, n, log_batch, log_count,
                )
            }

            fn ckks_replicate_galois_elements(
                module: &::poulpy_hal::layouts::Module<Self>,
                n: ::poulpy_core::layouts::Degree,
                log_batch: usize,
                log_count: usize,
            ) -> ::std::vec::Vec<i64> {
                $crate::reference::inner_sum::CKKSInnerSumReference::ckks_replicate_galois_elements_reference(
                    module, n, log_batch, log_count,
                )
            }

            fn ckks_inner_sum_tmp_bytes<R, A, K>(module: &::poulpy_hal::layouts::Module<Self>, res: &R, a: &A, key: &K) -> usize
            where
                R: $crate::CKKSCtBounds,
                A: $crate::CKKSCtBounds,
                K: ::poulpy_core::layouts::GGLWEInfos,
            {
                $crate::reference::inner_sum::CKKSInnerSumReference::ckks_inner_sum_tmp_bytes_reference(module, res, a, key)
            }

            fn ckks_inner_sum<R, A, H>(
                module: &::poulpy_hal::layouts::Module<Self>,
                log_batch: usize,
                log_count: usize,
                normalize: bool,
                res: &mut R,
                a: &A,
                keys: &H,
                scratch: &mut ::poulpy_hal::layouts::ScratchArena<'_, Self>,
            ) -> $crate::CKKSResult<()>
            where
                R: ::poulpy_core::layouts::GLWEToBackendMut<Self> + $crate::CKKSCtBounds + $crate::SetCKKSInfos,
                A: ::poulpy_core::layouts::GLWEToBackendRef<Self> + $crate::CKKSCtBounds,
                H: ::poulpy_core::layouts::GetAutomorphismKey<Self>,
            {
                $crate::reference::inner_sum::CKKSInnerSumReference::ckks_inner_sum_reference(
                    module, log_batch, log_count, normalize, res, a, keys, scratch,
                )
            }

            fn ckks_inner_sum_assign<R, H>(
                module: &::poulpy_hal::layouts::Module<Self>,
                log_batch: usize,
                log_count: usize,
                normalize: bool,
                res: &mut R,
                keys: &H,
                scratch: &mut ::poulpy_hal::layouts::ScratchArena<'_, Self>,
            ) -> $crate::CKKSResult<()>
            where
                R: ::poulpy_core::layouts::GLWEToBackendMut<Self> + $crate::CKKSCtBounds + $crate::SetCKKSInfos,
                H: ::poulpy_core::layouts::GetAutomorphismKey<Self>,
            {
                $crate::reference::inner_sum::CKKSInnerSumReference::ckks_inner_sum_assign_reference(
                    module, log_batch, log_count, normalize, res, keys, scratch,
                )
            }

            fn ckks_replicate<R, A, H>(
                module: &::poulpy_hal::layouts::Module<Self>,
                log_batch: usize,
                log_count: usize,
                res: &mut R,
                a: &A,
                keys: &H,
                scratch: &mut ::poulpy_hal::layouts::ScratchArena<'_, Self>,
            ) -> $crate::CKKSResult<()>
            where
                R: ::poulpy_core::layouts::GLWEToBackendMut<Self> + $crate::CKKSCtBounds + $crate::SetCKKSInfos,
                A: ::poulpy_core::layouts::GLWEToBackendRef<Self> + $crate::CKKSCtBounds,
                H: ::poulpy_core::layouts::GetAutomorphismKey<Self>,
            {
                $crate::reference::inner_sum::CKKSInnerSumReference::ckks_replicate_reference(
                    module, log_batch, log_count, res, a, keys, scratch,
                )
            }

            fn ckks_replicate_assign<R, H>(
                module: &::poulpy_hal::layouts::Module<Self>,
                log_batch: usize,
                log_count: usize,
                res: &mut R,
                keys: &H,
                scratch: &mut ::poulpy_hal::layouts::ScratchArena<'_, Self>,
            ) -> $crate::CKKSResult<()>
            where
                R: ::poulpy_core::layouts::GLWEToBackendMut<Self> + $crate::CKKSCtBounds + $crate::SetCKKSInfos,
                H: ::poulpy_core::layouts::GetAutomorphismKey<Self>,
            {
                $crate::reference::inner_sum::CKKSInnerSumReference::ckks_replicate_assign_reference(
                    module, log_batch, log_count, res, keys, scratch,
                )
            }
        }
    };
}
pub use crate::impl_ckks_inner_sum_reference_full;
