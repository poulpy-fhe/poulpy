use crate::CKKSResult as Result;
use poulpy_core::layouts::{Degree, GGLWEInfos, GLWELayout, GetAutomorphismKey, prepared::GGLWEPreparedToBackendRef};
use poulpy_hal::layouts::{Backend, Module, Ring, ScratchArena, Standard};

use crate::{
    CKKSCtBounds,
    layouts::{CKKSCiphertextOwned, CKKSFoldKeysLayout, CKKSRingCiphertext},
};

/// Backend override hook for [`CKKSFoldLayoutOps`](crate::api::CKKSFoldLayoutOps).
///
/// # Safety
/// Ciphertext outputs must reproduce the reference noise metadata and provenance
/// checks, including clearing invalidated estimates. Delegates only forward calls.
///
/// Implementations must answer for the fold they register with
/// [`CKKSFoldImpl`]: its key elements, layouts and scratch.
pub unsafe trait CKKSFoldLayoutImpl: Backend<Ring = Standard> {
    fn ckks_fold_layout_impl<C>(module: &Module<Self>, ct_in: &C, degree: Degree, keys: &CKKSFoldKeysLayout) -> GLWELayout
    where
        C: CKKSCtBounds;

    fn ckks_fold_tmp_bytes_impl<C1, C2>(
        module: &Module<Self>,
        ct_out: &C1,
        ct_in: &C2,
        degree: Degree,
        keys: &CKKSFoldKeysLayout,
    ) -> usize
    where
        C1: CKKSCtBounds,
        C2: CKKSCtBounds;
}

/// Backend override hook for [`CKKSFoldOps`](crate::api::CKKSFoldOps) on inputs of ring `R`.
///
/// # Safety
/// Ciphertext outputs must reproduce the reference noise metadata and provenance
/// checks, including clearing invalidated estimates. Delegates only forward calls.
///
/// Implementations must preserve the exact CKKS metadata and ciphertext
/// semantics of the reference fold, honor all key layouts, and stay within the
/// scratch of [`CKKSFoldLayoutImpl::ckks_fold_tmp_bytes_impl`].
pub unsafe trait CKKSFoldImpl<R: Ring>: Backend<Ring = Standard> {
    fn ckks_fold_count_impl(module: &Module<Self>, ins: &[CKKSRingCiphertext<Self, R>], degree: Degree) -> usize;

    fn ckks_unfold_galois_elements_impl<C>(module: &Module<Self>, ct_in: &C) -> Vec<i64>
    where
        C: CKKSCtBounds;

    fn ckks_fold_impl<S>(
        module: &Module<Self>,
        folded: &mut [CKKSCiphertextOwned<Self>],
        ins: &[CKKSRingCiphertext<Self, R>],
        inbound: Option<&S>,
        scratch: &mut ScratchArena<'_, Self>,
    ) -> Result<()>
    where
        S: GGLWEPreparedToBackendRef<Self> + GGLWEInfos;

    fn ckks_unfold_impl<S, H>(
        module: &Module<Self>,
        outs: &mut [CKKSRingCiphertext<Self, R>],
        folded: &mut [CKKSCiphertextOwned<Self>],
        outbound: Option<&S>,
        automorphisms: Option<&H>,
        scratch: &mut ScratchArena<'_, Self>,
    ) -> Result<()>
    where
        S: GGLWEPreparedToBackendRef<Self> + GGLWEInfos,
        H: GetAutomorphismKey<Self>;
}

/// Opts a standard backend into the CKKS reference fold, for every input ring the
/// reference folds.
#[macro_export]
macro_rules! impl_ckks_fold_reference {
    ($be:ty) => {
        unsafe impl $crate::oep::CKKSFoldLayoutImpl for $be {
            fn ckks_fold_layout_impl<C>(
                module: &::poulpy_hal::layouts::Module<$be>,
                ct_in: &C,
                degree: ::poulpy_core::layouts::Degree,
                keys: &$crate::layouts::CKKSFoldKeysLayout,
            ) -> ::poulpy_core::layouts::GLWELayout
            where
                C: $crate::CKKSCtBounds,
            {
                $crate::reference::fold::CKKSFoldLayoutReference::ckks_fold_layout_reference(module, ct_in, degree, keys)
            }

            fn ckks_fold_tmp_bytes_impl<C1, C2>(
                module: &::poulpy_hal::layouts::Module<$be>,
                ct_out: &C1,
                ct_in: &C2,
                degree: ::poulpy_core::layouts::Degree,
                keys: &$crate::layouts::CKKSFoldKeysLayout,
            ) -> usize
            where
                C1: $crate::CKKSCtBounds,
                C2: $crate::CKKSCtBounds,
            {
                $crate::reference::fold::CKKSFoldLayoutReference::ckks_fold_tmp_bytes_reference(
                    module, ct_out, ct_in, degree, keys,
                )
            }
        }

        unsafe impl<R: ::poulpy_hal::layouts::Ring> $crate::oep::CKKSFoldImpl<R> for $be
        where
            ::poulpy_hal::layouts::Module<$be>: $crate::reference::fold::CKKSFoldReference<$be, R>,
        {
            fn ckks_fold_count_impl(
                module: &::poulpy_hal::layouts::Module<$be>,
                ins: &[$crate::layouts::CKKSRingCiphertext<$be, R>],
                degree: ::poulpy_core::layouts::Degree,
            ) -> usize {
                $crate::reference::fold::CKKSFoldReference::ckks_fold_count_reference(module, ins, degree)
            }

            fn ckks_unfold_galois_elements_impl<C>(module: &::poulpy_hal::layouts::Module<$be>, ct_in: &C) -> Vec<i64>
            where
                C: $crate::CKKSCtBounds,
            {
                $crate::reference::fold::CKKSFoldReference::<$be, R>::ckks_unfold_galois_elements_reference(module, ct_in)
            }

            fn ckks_fold_impl<S>(
                module: &::poulpy_hal::layouts::Module<$be>,
                folded: &mut [$crate::layouts::CKKSCiphertextOwned<$be>],
                ins: &[$crate::layouts::CKKSRingCiphertext<$be, R>],
                inbound: Option<&S>,
                scratch: &mut ::poulpy_hal::layouts::ScratchArena<'_, $be>,
            ) -> $crate::CKKSResult<()>
            where
                S: ::poulpy_core::layouts::prepared::GGLWEPreparedToBackendRef<$be> + ::poulpy_core::layouts::GGLWEInfos,
            {
                $crate::reference::fold::CKKSFoldReference::ckks_fold_reference(module, folded, ins, inbound, scratch)
            }

            fn ckks_unfold_impl<S, H>(
                module: &::poulpy_hal::layouts::Module<$be>,
                outs: &mut [$crate::layouts::CKKSRingCiphertext<$be, R>],
                folded: &mut [$crate::layouts::CKKSCiphertextOwned<$be>],
                outbound: Option<&S>,
                automorphisms: Option<&H>,
                scratch: &mut ::poulpy_hal::layouts::ScratchArena<'_, $be>,
            ) -> $crate::CKKSResult<()>
            where
                S: ::poulpy_core::layouts::prepared::GGLWEPreparedToBackendRef<$be> + ::poulpy_core::layouts::GGLWEInfos,
                H: ::poulpy_core::layouts::GetAutomorphismKey<$be>,
            {
                $crate::reference::fold::CKKSFoldReference::ckks_unfold_reference(
                    module,
                    outs,
                    folded,
                    outbound,
                    automorphisms,
                    scratch,
                )
            }
        }
    };
}

pub use crate::impl_ckks_fold_reference;
