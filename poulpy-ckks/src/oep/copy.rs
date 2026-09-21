use crate::CKKSResult as Result;

use poulpy_hal::layouts::{Backend, Module, ScratchArena};

use crate::{CKKSCtBounds, GLWEToBackendMut, GLWEToBackendRef, SetCKKSInfos};

/// # Safety
///
/// Implementations must satisfy the contracts of all trait methods, including
/// any HAL-level invariants (alignment, layout, scratch sizing) implied by the
/// associated method signatures.
pub unsafe trait CKKSCopyImpl: Backend {
    fn ckks_copy_tmp_bytes_impl<Dst: CKKSCtBounds, Src: CKKSCtBounds>(module: &Module<Self>, dst: &Dst, src: &Src) -> usize;

    fn ckks_copy_impl<Dst, Src>(
        module: &Module<Self>,
        dst: &mut Dst,
        src: &Src,
        scratch: &mut ScratchArena<'_, Self>,
    ) -> Result<()>
    where
        Dst: GLWEToBackendMut<Self> + CKKSCtBounds + SetCKKSInfos,
        Src: GLWEToBackendRef<Self> + CKKSCtBounds;
}

/// Implements this contract with the callable reference algorithms.
#[macro_export]
macro_rules! impl_ckks_copy_reference {
    ($be:ty) => {
        unsafe impl $crate::oep::CKKSCopyImpl for $be {
            fn ckks_copy_tmp_bytes_impl<Dst: $crate::CKKSCtBounds, Src: $crate::CKKSCtBounds>(
                module: &::poulpy_hal::layouts::Module<Self>,
                dst: &Dst,
                src: &Src,
            ) -> usize {
                $crate::reference::copy::CKKSCopyReference::ckks_copy_tmp_bytes_reference(module, dst, src)
            }

            fn ckks_copy_impl<Dst, Src>(
                module: &::poulpy_hal::layouts::Module<Self>,
                dst: &mut Dst,
                src: &Src,
                scratch: &mut ::poulpy_hal::layouts::ScratchArena<'_, Self>,
            ) -> $crate::CKKSResult<()>
            where
                Dst: ::poulpy_core::layouts::GLWEToBackendMut<Self> + $crate::CKKSCtBounds + $crate::SetCKKSInfos,
                Src: ::poulpy_core::layouts::GLWEToBackendRef<Self> + $crate::CKKSCtBounds,
            {
                $crate::reference::copy::CKKSCopyReference::ckks_copy_reference(module, dst, src, scratch)
            }
        }
    };
}
pub use crate::impl_ckks_copy_reference;
