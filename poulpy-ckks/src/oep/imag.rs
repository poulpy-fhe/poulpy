use crate::CKKSResult as Result;

use poulpy_core::layouts::{GLWEInfos, GLWEToBackendMut, GLWEToBackendRef};
use poulpy_hal::layouts::{Backend, Module, ScratchArena};

use crate::{CKKSCtBounds, SetCKKSInfos};

/// # Safety
///
/// Implementations must satisfy the contracts of all trait methods, including
/// any HAL-level invariants (alignment, layout, scratch sizing) implied by the
/// associated method signatures.
pub unsafe trait CKKSImagImpl: Backend {
    fn ckks_mul_i_tmp_bytes_impl(module: &Module<Self>, res_size: usize) -> usize;

    fn ckks_mul_i_into_impl<Dst, Src>(
        module: &Module<Self>,
        dst: &mut Dst,
        src: &Src,
        scratch: &mut ScratchArena<'_, Self>,
    ) -> Result<()>
    where
        Dst: GLWEToBackendMut<Self> + CKKSCtBounds + SetCKKSInfos,
        Src: GLWEToBackendRef<Self> + GLWEInfos + CKKSCtBounds;

    fn ckks_mul_i_assign_impl<Dst>(module: &Module<Self>, dst: &mut Dst, scratch: &mut ScratchArena<'_, Self>) -> Result<()>
    where
        Dst: GLWEToBackendMut<Self> + CKKSCtBounds + SetCKKSInfos;

    fn ckks_div_i_tmp_bytes_impl(module: &Module<Self>, res_size: usize) -> usize;

    fn ckks_div_i_into_impl<Dst, Src>(
        module: &Module<Self>,
        dst: &mut Dst,
        src: &Src,
        scratch: &mut ScratchArena<'_, Self>,
    ) -> Result<()>
    where
        Dst: GLWEToBackendMut<Self> + CKKSCtBounds + SetCKKSInfos,
        Src: GLWEToBackendRef<Self> + GLWEInfos + CKKSCtBounds;

    fn ckks_div_i_assign_impl<Dst>(module: &Module<Self>, dst: &mut Dst, scratch: &mut ScratchArena<'_, Self>) -> Result<()>
    where
        Dst: GLWEToBackendMut<Self> + CKKSCtBounds + SetCKKSInfos;
}

/// Implements this contract with the callable reference algorithms.
#[macro_export]
macro_rules! impl_ckks_imag_reference {
    ($be:ty) => {
        unsafe impl $crate::oep::CKKSImagImpl for $be {
            fn ckks_mul_i_tmp_bytes_impl(module: &::poulpy_hal::layouts::Module<Self>, res_size: usize) -> usize {
                $crate::reference::imag::CKKSImagReference::ckks_mul_i_tmp_bytes_reference(module, res_size)
            }

            fn ckks_mul_i_into_impl<Dst, Src>(
                module: &::poulpy_hal::layouts::Module<Self>,
                dst: &mut Dst,
                src: &Src,
                scratch: &mut ::poulpy_hal::layouts::ScratchArena<'_, Self>,
            ) -> $crate::CKKSResult<()>
            where
                Dst: ::poulpy_core::layouts::GLWEToBackendMut<Self> + $crate::CKKSCtBounds + $crate::SetCKKSInfos,
                Src: ::poulpy_core::layouts::GLWEToBackendRef<Self> + ::poulpy_core::layouts::GLWEInfos + $crate::CKKSCtBounds,
            {
                $crate::reference::imag::CKKSImagReference::ckks_mul_i_into_reference(module, dst, src, scratch)
            }

            fn ckks_mul_i_assign_impl<Dst>(
                module: &::poulpy_hal::layouts::Module<Self>,
                dst: &mut Dst,
                scratch: &mut ::poulpy_hal::layouts::ScratchArena<'_, Self>,
            ) -> $crate::CKKSResult<()>
            where
                Dst: ::poulpy_core::layouts::GLWEToBackendMut<Self> + $crate::CKKSCtBounds + $crate::SetCKKSInfos,
            {
                $crate::reference::imag::CKKSImagReference::ckks_mul_i_assign_reference(module, dst, scratch)
            }

            fn ckks_div_i_tmp_bytes_impl(module: &::poulpy_hal::layouts::Module<Self>, res_size: usize) -> usize {
                $crate::reference::imag::CKKSImagReference::ckks_div_i_tmp_bytes_reference(module, res_size)
            }

            fn ckks_div_i_into_impl<Dst, Src>(
                module: &::poulpy_hal::layouts::Module<Self>,
                dst: &mut Dst,
                src: &Src,
                scratch: &mut ::poulpy_hal::layouts::ScratchArena<'_, Self>,
            ) -> $crate::CKKSResult<()>
            where
                Dst: ::poulpy_core::layouts::GLWEToBackendMut<Self> + $crate::CKKSCtBounds + $crate::SetCKKSInfos,
                Src: ::poulpy_core::layouts::GLWEToBackendRef<Self> + ::poulpy_core::layouts::GLWEInfos + $crate::CKKSCtBounds,
            {
                $crate::reference::imag::CKKSImagReference::ckks_div_i_into_reference(module, dst, src, scratch)
            }

            fn ckks_div_i_assign_impl<Dst>(
                module: &::poulpy_hal::layouts::Module<Self>,
                dst: &mut Dst,
                scratch: &mut ::poulpy_hal::layouts::ScratchArena<'_, Self>,
            ) -> $crate::CKKSResult<()>
            where
                Dst: ::poulpy_core::layouts::GLWEToBackendMut<Self> + $crate::CKKSCtBounds + $crate::SetCKKSInfos,
            {
                $crate::reference::imag::CKKSImagReference::ckks_div_i_assign_reference(module, dst, scratch)
            }
        }
    };
}
pub use crate::impl_ckks_imag_reference;
