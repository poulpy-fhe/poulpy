use crate::CKKSResult as Result;
use poulpy_hal::layouts::{Backend, Module, ScratchArena, Standard};

use crate::{
    CKKSCtBounds, CKKSLayout,
    layouts::{CKKSCiphertextOwned, CKKSFoldKeys},
    reference::fold::CKKSFoldRing,
};

/// Backend override hook for [`CKKSFoldOps`](crate::api::CKKSFoldOps).
/// [`impl_ckks_fold_reference`] wires every method to the reference fold.
///
/// # Safety
///
/// Implementations must preserve the exact CKKS metadata and ciphertext
/// semantics of the reference fold, honor all key layouts, and stay within the
/// scratch they report.
pub unsafe trait CKKSFoldImpl: Backend<Ring = Standard> {
    fn ckks_fold_count_impl<IN>(module: &Module<Self>, input_module: &Module<IN>, ins: &[CKKSCiphertextOwned<IN>]) -> usize
    where
        IN: Backend,
        IN::Ring: CKKSFoldRing<Self, IN>;

    fn ckks_fold_layout_impl<IN, C, K>(module: &Module<Self>, input_module: &Module<IN>, ct_in: &C, keys: &K) -> CKKSLayout
    where
        IN: Backend,
        C: CKKSCtBounds,
        K: CKKSFoldKeys<Self, IN>;

    fn ckks_fold_tmp_bytes_impl<IN, C1, C2, K>(
        module: &Module<Self>,
        input_module: &Module<IN>,
        ct_out: &C1,
        ct_in: &C2,
        keys: &K,
    ) -> usize
    where
        IN: Backend,
        IN::Ring: CKKSFoldRing<Self, IN>,
        C1: CKKSCtBounds,
        C2: CKKSCtBounds,
        K: CKKSFoldKeys<Self, IN>;

    fn ckks_fold_impl<IN, K>(
        module: &Module<Self>,
        input_module: &Module<IN>,
        folded: &mut [CKKSCiphertextOwned<Self>],
        ins: &[CKKSCiphertextOwned<IN>],
        keys: &K,
        scratch: &mut ScratchArena<'_, Self>,
    ) -> Result<()>
    where
        IN: Backend,
        IN::Ring: CKKSFoldRing<Self, IN>,
        K: CKKSFoldKeys<Self, IN>;

    #[allow(clippy::too_many_arguments)]
    fn ckks_unfold_impl<IN, K>(
        module: &Module<Self>,
        input_module: &Module<IN>,
        outs: &mut [CKKSCiphertextOwned<IN>],
        refreshed: &[CKKSCiphertextOwned<Self>],
        ins: &[CKKSCiphertextOwned<IN>],
        keys: &K,
        scratch: &mut ScratchArena<'_, Self>,
    ) -> Result<()>
    where
        IN: Backend,
        IN::Ring: CKKSFoldRing<Self, IN>,
        K: CKKSFoldKeys<Self, IN>;
}

/// Opts a standard backend into the CKKS reference fold.
#[macro_export]
macro_rules! impl_ckks_fold_reference {
    ($be:ty) => {
        unsafe impl $crate::oep::CKKSFoldImpl for $be {
            fn ckks_fold_count_impl<IN>(
                module: &::poulpy_hal::layouts::Module<$be>,
                input_module: &::poulpy_hal::layouts::Module<IN>,
                ins: &[$crate::layouts::CKKSCiphertextOwned<IN>],
            ) -> usize
            where
                IN: ::poulpy_hal::layouts::Backend,
                IN::Ring: $crate::reference::fold::CKKSFoldRing<$be, IN>,
            {
                $crate::reference::fold::ckks_fold_count_reference(module, input_module, ins)
            }

            fn ckks_fold_layout_impl<IN, C, K>(
                module: &::poulpy_hal::layouts::Module<$be>,
                _input_module: &::poulpy_hal::layouts::Module<IN>,
                ct_in: &C,
                keys: &K,
            ) -> $crate::CKKSLayout
            where
                IN: ::poulpy_hal::layouts::Backend,
                C: $crate::CKKSCtBounds,
                K: $crate::layouts::CKKSFoldKeys<$be, IN>,
            {
                $crate::reference::fold::ckks_fold_layout_reference::<$be, IN, C, K>(module, ct_in, keys)
            }

            fn ckks_fold_tmp_bytes_impl<IN, C1, C2, K>(
                module: &::poulpy_hal::layouts::Module<$be>,
                input_module: &::poulpy_hal::layouts::Module<IN>,
                ct_out: &C1,
                ct_in: &C2,
                keys: &K,
            ) -> usize
            where
                IN: ::poulpy_hal::layouts::Backend,
                IN::Ring: $crate::reference::fold::CKKSFoldRing<$be, IN>,
                C1: $crate::CKKSCtBounds,
                C2: $crate::CKKSCtBounds,
                K: $crate::layouts::CKKSFoldKeys<$be, IN>,
            {
                $crate::reference::fold::ckks_fold_tmp_bytes_reference(module, input_module, ct_out, ct_in, keys)
            }

            fn ckks_fold_impl<IN, K>(
                module: &::poulpy_hal::layouts::Module<$be>,
                input_module: &::poulpy_hal::layouts::Module<IN>,
                folded: &mut [$crate::layouts::CKKSCiphertextOwned<$be>],
                ins: &[$crate::layouts::CKKSCiphertextOwned<IN>],
                keys: &K,
                scratch: &mut ::poulpy_hal::layouts::ScratchArena<'_, $be>,
            ) -> $crate::CKKSResult<()>
            where
                IN: ::poulpy_hal::layouts::Backend,
                IN::Ring: $crate::reference::fold::CKKSFoldRing<$be, IN>,
                K: $crate::layouts::CKKSFoldKeys<$be, IN>,
            {
                $crate::reference::fold::ckks_fold_reference(module, input_module, folded, ins, keys, scratch)
            }

            fn ckks_unfold_impl<IN, K>(
                module: &::poulpy_hal::layouts::Module<$be>,
                input_module: &::poulpy_hal::layouts::Module<IN>,
                outs: &mut [$crate::layouts::CKKSCiphertextOwned<IN>],
                refreshed: &[$crate::layouts::CKKSCiphertextOwned<$be>],
                ins: &[$crate::layouts::CKKSCiphertextOwned<IN>],
                keys: &K,
                scratch: &mut ::poulpy_hal::layouts::ScratchArena<'_, $be>,
            ) -> $crate::CKKSResult<()>
            where
                IN: ::poulpy_hal::layouts::Backend,
                IN::Ring: $crate::reference::fold::CKKSFoldRing<$be, IN>,
                K: $crate::layouts::CKKSFoldKeys<$be, IN>,
            {
                $crate::reference::fold::ckks_unfold_reference(module, input_module, outs, refreshed, ins, keys, scratch)
            }
        }
    };
}

pub use crate::impl_ckks_fold_reference;
