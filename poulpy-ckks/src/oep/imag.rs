use crate::CKKSResult as Result;
use crate::reference::imag::CKKSImagReference;

use poulpy_core::{
    GLWECopy, GLWENegate, GLWERotate, GLWEShift,
    layouts::{GLWEInfos, GLWEToBackendMut, GLWEToBackendRef},
};
use poulpy_hal::{
    api::ModuleN,
    layouts::{Backend, Module, ScratchArena},
};

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

unsafe impl<BE: Backend> CKKSImagImpl for BE
where
    BE: poulpy_hal::oep::HalVecZnxImpl,
    Module<BE>:
        crate::reference::imag::CKKSImagReference<BE> + GLWECopy<BE> + GLWENegate<BE> + GLWERotate<BE> + GLWEShift<BE> + ModuleN,
{
    fn ckks_mul_i_tmp_bytes_impl(module: &Module<BE>, res_size: usize) -> usize {
        module.ckks_mul_i_tmp_bytes_reference(res_size)
    }

    fn ckks_mul_i_into_impl<Dst, Src>(
        module: &Module<BE>,
        dst: &mut Dst,
        src: &Src,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        Dst: GLWEToBackendMut<BE> + CKKSCtBounds + SetCKKSInfos,
        Src: GLWEToBackendRef<BE> + GLWEInfos + CKKSCtBounds,
    {
        module.ckks_mul_i_into_reference(dst, src, scratch)
    }

    fn ckks_mul_i_assign_impl<Dst>(module: &Module<BE>, dst: &mut Dst, scratch: &mut ScratchArena<'_, BE>) -> Result<()>
    where
        Dst: GLWEToBackendMut<BE> + CKKSCtBounds + SetCKKSInfos,
    {
        module.ckks_mul_i_assign_reference(dst, scratch)
    }

    fn ckks_div_i_tmp_bytes_impl(module: &Module<BE>, res_size: usize) -> usize {
        module.ckks_div_i_tmp_bytes_reference(res_size)
    }

    fn ckks_div_i_into_impl<Dst, Src>(
        module: &Module<BE>,
        dst: &mut Dst,
        src: &Src,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        Dst: GLWEToBackendMut<BE> + CKKSCtBounds + SetCKKSInfos,
        Src: GLWEToBackendRef<BE> + GLWEInfos + CKKSCtBounds,
    {
        module.ckks_div_i_into_reference(dst, src, scratch)
    }

    fn ckks_div_i_assign_impl<Dst>(module: &Module<BE>, dst: &mut Dst, scratch: &mut ScratchArena<'_, BE>) -> Result<()>
    where
        Dst: GLWEToBackendMut<BE> + CKKSCtBounds + SetCKKSInfos,
    {
        module.ckks_div_i_assign_reference(dst, scratch)
    }
}

#[macro_export]
macro_rules! impl_ckks_imag_reference {
    ($be:ty) => {
        impl $crate::reference::imag::CKKSImagReference<$be> for ::poulpy_hal::layouts::Module<$be> {}
    };
}
pub use crate::impl_ckks_imag_reference;
