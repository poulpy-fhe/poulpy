use crate::CKKSResult as Result;

use poulpy_hal::layouts::{Backend, Module, Ring, ScratchArena, Standard};

use crate::{CKKSCtBounds, GLWEToBackendMut, GLWEToBackendRef, SetCKKSInfos};

/// Backend contract of [`CKKSCopyOps`](crate::api::CKKSCopyOps).
///
/// # Safety
/// Implementations reproduce the bodies their methods document bit for bit, report for each
/// `_tmp_bytes` query a bound that covers the body, and refuse as the reference does. The memory,
/// layout, synchrony and aliasing rules of `poulpy-hal/docs/backend_safety_contract.md` apply.
pub unsafe trait CKKSCopyImpl<Rg: Ring = Standard>: Backend {
    /// Scratch bound of [`Self::ckks_copy`]; may report less than the reference, never less than the body takes.
    fn ckks_copy_tmp_bytes<Dst: CKKSCtBounds, Src: CKKSCtBounds>(module: &Module<Self>, dst: &Dst, src: &Src) -> usize;

    /// Reproduces [`ckks_copy_reference`](crate::reference::copy::CKKSCopyReference::ckks_copy_reference) bit for bit,
    /// within [`Self::ckks_copy_tmp_bytes`]; free in kernels and scratch use.
    fn ckks_copy<Dst, Src>(module: &Module<Self>, dst: &mut Dst, src: &Src, scratch: &mut ScratchArena<'_, Self>) -> Result<()>
    where
        Dst: GLWEToBackendMut<Self, Rg> + CKKSCtBounds + SetCKKSInfos,
        Src: GLWEToBackendRef<Self, Rg> + CKKSCtBounds;
}

/// Selects the reference implementation of `CKKSCopyImpl` for `$be`.
#[macro_export]
macro_rules! impl_ckks_copy_reference_full {
    ($be:ty) => {
        $crate::impl_ckks_copy_reference_full!($be, ::poulpy_hal::layouts::Standard);
        $crate::impl_ckks_copy_reference_full!($be, ::poulpy_hal::layouts::ConjugateInvariant);
    };
    ($be:ty, $ring:ty) => {
        unsafe impl $crate::oep::CKKSCopyImpl<$ring> for $be {
            fn ckks_copy_tmp_bytes<Dst: $crate::CKKSCtBounds, Src: $crate::CKKSCtBounds>(
                module: &::poulpy_hal::layouts::Module<Self>,
                dst: &Dst,
                src: &Src,
            ) -> usize {
                $crate::reference::copy::CKKSCopyReference::<Self, $ring>::ckks_copy_tmp_bytes_reference(module, dst, src)
            }

            fn ckks_copy<Dst, Src>(
                module: &::poulpy_hal::layouts::Module<Self>,
                dst: &mut Dst,
                src: &Src,
                scratch: &mut ::poulpy_hal::layouts::ScratchArena<'_, Self>,
            ) -> $crate::CKKSResult<()>
            where
                Dst: ::poulpy_core::layouts::GLWEToBackendMut<Self, $ring> + $crate::CKKSCtBounds + $crate::SetCKKSInfos,
                Src: ::poulpy_core::layouts::GLWEToBackendRef<Self, $ring> + $crate::CKKSCtBounds,
            {
                $crate::reference::copy::CKKSCopyReference::<Self, $ring>::ckks_copy_reference(module, dst, src, scratch)
            }
        }
    };
}
pub use crate::impl_ckks_copy_reference_full;
