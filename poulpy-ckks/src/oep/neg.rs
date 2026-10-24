use crate::CKKSResult as Result;

use poulpy_core::layouts::GLWEInfos;
use poulpy_hal::layouts::{Backend, Module, ScratchArena};

use crate::{CKKSCtBounds, GLWEToBackendMut, GLWEToBackendRef, SetCKKSInfos};

/// # Safety
///
/// Implementations must satisfy the contracts of all trait methods, including
/// any HAL-level invariants (alignment, layout, scratch sizing) implied by the
/// associated method signatures.
pub unsafe trait CKKSNegImpl: Backend {
    fn ckks_neg_tmp_bytes(module: &Module<Self>, res_size: usize) -> usize;

    fn ckks_neg_into<Dst, Src>(
        module: &Module<Self>,
        dst: &mut Dst,
        src: &Src,
        scratch: &mut ScratchArena<'_, Self>,
    ) -> Result<()>
    where
        Dst: GLWEToBackendMut<Self> + CKKSCtBounds + SetCKKSInfos,
        Src: GLWEToBackendRef<Self> + GLWEInfos + CKKSCtBounds;

    fn ckks_neg_assign<Dst>(module: &Module<Self>, dst: &mut Dst) -> Result<()>
    where
        Dst: GLWEToBackendMut<Self> + CKKSCtBounds + SetCKKSInfos;
}

/// Selects the reference implementation of `CKKSNegImpl` for `$be`.
#[macro_export]
macro_rules! impl_ckks_neg_reference_full {
    ($be:ty) => {
        unsafe impl $crate::oep::CKKSNegImpl for $be {
            fn ckks_neg_tmp_bytes(module: &::poulpy_hal::layouts::Module<Self>, res_size: usize) -> usize {
                $crate::reference::neg::CKKSNegReference::ckks_neg_tmp_bytes_reference(module, res_size)
            }

            fn ckks_neg_into<Dst, Src>(
                module: &::poulpy_hal::layouts::Module<Self>,
                dst: &mut Dst,
                src: &Src,
                scratch: &mut ::poulpy_hal::layouts::ScratchArena<'_, Self>,
            ) -> $crate::CKKSResult<()>
            where
                Dst: ::poulpy_core::layouts::GLWEToBackendMut<Self> + $crate::CKKSCtBounds + $crate::SetCKKSInfos,
                Src: ::poulpy_core::layouts::GLWEToBackendRef<Self> + ::poulpy_core::layouts::GLWEInfos + $crate::CKKSCtBounds,
            {
                $crate::reference::neg::CKKSNegReference::ckks_neg_into_reference(module, dst, src, scratch)
            }

            fn ckks_neg_assign<Dst>(module: &::poulpy_hal::layouts::Module<Self>, dst: &mut Dst) -> $crate::CKKSResult<()>
            where
                Dst: ::poulpy_core::layouts::GLWEToBackendMut<Self> + $crate::CKKSCtBounds + $crate::SetCKKSInfos,
            {
                $crate::reference::neg::CKKSNegReference::ckks_neg_assign_reference(module, dst)
            }
        }
    };
}
pub use crate::impl_ckks_neg_reference_full;
