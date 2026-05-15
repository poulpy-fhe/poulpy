use crate::default::copy::CKKSCopyDefault;

use anyhow::Result;
use poulpy_core::{GLWECopy, GLWEShift, ScratchArenaTakeCore, layouts::LWEInfos};
use poulpy_hal::layouts::{Backend, Module, ScratchArena};

use crate::{CKKSInfos, GLWEToBackendMut, GLWEToBackendRef, SetCKKSInfos};

/// # Safety
///
/// Implementations must satisfy the contracts of all trait methods, including
/// any HAL-level invariants (alignment, layout, scratch sizing) implied by the
/// associated method signatures.
pub unsafe trait CKKSCopyImpl<BE: Backend>: Backend {
    fn ckks_copy_tmp_bytes(module: &Module<BE>) -> usize;

    fn ckks_copy<Dst, Src>(module: &Module<BE>, dst: &mut Dst, src: &Src, scratch: &mut ScratchArena<'_, BE>) -> Result<()>
    where
        Dst: GLWEToBackendMut<BE> + LWEInfos + CKKSInfos + SetCKKSInfos,
        Src: GLWEToBackendRef<BE> + LWEInfos + CKKSInfos;
}

unsafe impl<BE: Backend> CKKSCopyImpl<BE> for BE
where
    BE: poulpy_hal::oep::HalVecZnxImpl<BE>,
    Module<BE>: crate::default::copy::CKKSCopyDefault<BE> + GLWECopy<BE> + GLWEShift<BE>,
    for<'a> ScratchArena<'a, BE>: ScratchArenaTakeCore<'a, BE>,
{
    fn ckks_copy_tmp_bytes(module: &Module<BE>) -> usize {
        module.ckks_copy_tmp_bytes_default()
    }

    fn ckks_copy<Dst, Src>(module: &Module<BE>, dst: &mut Dst, src: &Src, scratch: &mut ScratchArena<'_, BE>) -> Result<()>
    where
        Dst: GLWEToBackendMut<BE> + LWEInfos + CKKSInfos + SetCKKSInfos,
        Src: GLWEToBackendRef<BE> + LWEInfos + CKKSInfos,
    {
        module.ckks_copy_default(dst, src, scratch)
    }
}

#[macro_export]
macro_rules! impl_ckks_copy_defaults {
    ($be:ty) => {
        impl $crate::default::copy::CKKSCopyDefault<$be> for ::poulpy_hal::layouts::Module<$be> {}
    };
}
pub use crate::impl_ckks_copy_defaults;
