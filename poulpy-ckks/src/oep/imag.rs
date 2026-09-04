use crate::CKKSResult as Result;
use crate::default::imag::CKKSImagDefault;
use poulpy_hal::layouts::CoeffNormalized;

use poulpy_core::{
    GLWECopy, GLWENegate, GLWERotate, GLWEShift,
    layouts::{GLWEInfos, GLWEToBackendMut, GLWEToBackendRef},
};
use poulpy_hal::{
    api::ModuleN,
    layouts::{Backend, CoeffFitsIn, Module, ScratchArena},
};

use crate::{CKKSCtBounds, SetCKKSInfos};

/// # Safety
///
/// Implementations must satisfy the contracts of all trait methods, including
/// any HAL-level invariants (alignment, layout, scratch sizing) implied by the
/// associated method signatures.
pub unsafe trait CKKSImagImpl<BE: Backend>: Backend {
    fn ckks_mul_i_tmp_bytes_impl(module: &Module<BE>) -> usize;

    fn ckks_mul_i_into_impl<Dst, Src>(
        module: &Module<BE>,
        dst: &mut Dst,
        src: &Src,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        Dst: GLWEToBackendMut<BE, State = CoeffNormalized> + CKKSCtBounds + SetCKKSInfos,
        Src: GLWEToBackendRef<BE, State = CoeffNormalized> + GLWEInfos + CKKSCtBounds,
        <Src as GLWEToBackendRef<BE>>::State: CoeffFitsIn<<Dst as GLWEToBackendRef<BE>>::State>;

    fn ckks_mul_i_assign_impl<Dst>(module: &Module<BE>, dst: &mut Dst, scratch: &mut ScratchArena<'_, BE>) -> Result<()>
    where
        Dst: GLWEToBackendMut<BE, State = CoeffNormalized> + CKKSCtBounds + SetCKKSInfos;

    fn ckks_div_i_tmp_bytes_impl(module: &Module<BE>) -> usize;

    fn ckks_div_i_into_impl<Dst, Src>(
        module: &Module<BE>,
        dst: &mut Dst,
        src: &Src,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        Dst: GLWEToBackendMut<BE, State = CoeffNormalized> + CKKSCtBounds + SetCKKSInfos,
        Src: GLWEToBackendRef<BE, State = CoeffNormalized> + GLWEInfos + CKKSCtBounds;

    fn ckks_div_i_assign_impl<Dst>(module: &Module<BE>, dst: &mut Dst, scratch: &mut ScratchArena<'_, BE>) -> Result<()>
    where
        Dst: GLWEToBackendMut<BE, State = CoeffNormalized> + CKKSCtBounds + SetCKKSInfos;
}

unsafe impl<BE: Backend> CKKSImagImpl<BE> for BE
where
    BE: poulpy_hal::oep::HalVecZnxImpl<BE>,
    Module<BE>:
        crate::default::imag::CKKSImagDefault<BE> + GLWECopy<BE> + GLWENegate<BE> + GLWERotate<BE> + GLWEShift<BE> + ModuleN,
{
    fn ckks_mul_i_tmp_bytes_impl(module: &Module<BE>) -> usize {
        module.ckks_mul_i_tmp_bytes_default()
    }

    fn ckks_mul_i_into_impl<Dst, Src>(
        module: &Module<BE>,
        dst: &mut Dst,
        src: &Src,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        Dst: GLWEToBackendMut<BE, State = CoeffNormalized> + CKKSCtBounds + SetCKKSInfos,
        Src: GLWEToBackendRef<BE, State = CoeffNormalized> + GLWEInfos + CKKSCtBounds,
        <Src as GLWEToBackendRef<BE>>::State: CoeffFitsIn<<Dst as GLWEToBackendRef<BE>>::State>,
    {
        module.ckks_mul_i_into_default(dst, src, scratch)
    }

    fn ckks_mul_i_assign_impl<Dst>(module: &Module<BE>, dst: &mut Dst, scratch: &mut ScratchArena<'_, BE>) -> Result<()>
    where
        Dst: GLWEToBackendMut<BE, State = CoeffNormalized> + CKKSCtBounds + SetCKKSInfos,
    {
        module.ckks_mul_i_assign_default(dst, scratch)
    }

    fn ckks_div_i_tmp_bytes_impl(module: &Module<BE>) -> usize {
        module.ckks_div_i_tmp_bytes_default()
    }

    fn ckks_div_i_into_impl<Dst, Src>(
        module: &Module<BE>,
        dst: &mut Dst,
        src: &Src,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        Dst: GLWEToBackendMut<BE, State = CoeffNormalized> + CKKSCtBounds + SetCKKSInfos,
        Src: GLWEToBackendRef<BE, State = CoeffNormalized> + GLWEInfos + CKKSCtBounds,
    {
        module.ckks_div_i_into_default(dst, src, scratch)
    }

    fn ckks_div_i_assign_impl<Dst>(module: &Module<BE>, dst: &mut Dst, scratch: &mut ScratchArena<'_, BE>) -> Result<()>
    where
        Dst: GLWEToBackendMut<BE, State = CoeffNormalized> + CKKSCtBounds + SetCKKSInfos,
    {
        module.ckks_div_i_assign_default(dst, scratch)
    }
}

#[macro_export]
macro_rules! impl_ckks_imag_defaults {
    ($be:ty) => {
        impl $crate::default::imag::CKKSImagDefault<$be> for ::poulpy_hal::layouts::Module<$be> {}
    };
}
pub use crate::impl_ckks_imag_defaults;
