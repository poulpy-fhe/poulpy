use crate::CKKSResult as Result;

use poulpy_hal::layouts::{Backend, Module, ScratchArena};

use crate::{CKKSCtBounds, GLWEToBackendMut, GLWEToBackendRef, SetCKKSInfos};

/// # Safety
///
/// Implementations must satisfy the contracts of all trait methods, including
/// any HAL-level invariants (alignment, layout, scratch sizing) implied by the
/// associated method signatures.
pub unsafe trait CKKSRoundImpl: Backend {
    fn ckks_round_tmp_bytes_impl<Dst, Src>(module: &Module<Self>, dst: &Dst, src: &Src) -> usize
    where
        Dst: CKKSCtBounds,
        Src: CKKSCtBounds;

    fn ckks_round_into_impl<Dst, Src>(
        module: &Module<Self>,
        dst: &mut Dst,
        src: &Src,
        scratch: &mut ScratchArena<'_, Self>,
    ) -> Result<()>
    where
        Dst: GLWEToBackendMut<Self> + CKKSCtBounds + SetCKKSInfos,
        Src: GLWEToBackendRef<Self> + CKKSCtBounds;
}

/// Implements this contract with the host reference algorithm.
#[macro_export]
macro_rules! impl_ckks_round_reference {
    ($be:ty) => {
        unsafe impl $crate::oep::CKKSRoundImpl for $be {
            fn ckks_round_tmp_bytes_impl<Dst, Src>(module: &::poulpy_hal::layouts::Module<Self>, dst: &Dst, src: &Src) -> usize
            where
                Dst: $crate::CKKSCtBounds,
                Src: $crate::CKKSCtBounds,
            {
                $crate::reference::round::CKKSRoundReference::ckks_round_tmp_bytes_reference(module, dst, src)
            }

            fn ckks_round_into_impl<Dst, Src>(
                module: &::poulpy_hal::layouts::Module<Self>,
                dst: &mut Dst,
                src: &Src,
                scratch: &mut ::poulpy_hal::layouts::ScratchArena<'_, Self>,
            ) -> $crate::CKKSResult<()>
            where
                Dst: ::poulpy_core::layouts::GLWEToBackendMut<Self> + $crate::CKKSCtBounds + $crate::SetCKKSInfos,
                Src: ::poulpy_core::layouts::GLWEToBackendRef<Self> + $crate::CKKSCtBounds,
            {
                $crate::reference::round::CKKSRoundReference::ckks_round_into_reference(module, dst, src, scratch)
            }
        }
    };
}
pub use crate::impl_ckks_round_reference;
