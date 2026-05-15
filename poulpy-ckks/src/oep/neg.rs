use crate::default::neg::CKKSNegDefault;

use anyhow::Result;
use poulpy_core::{
    GLWENegate, GLWEShift, ScratchArenaTakeCore,
    layouts::{GLWEInfos, LWEInfos},
};
use poulpy_hal::layouts::{Backend, Module, ScratchArena};

use crate::{CKKSInfos, GLWEToBackendMut, GLWEToBackendRef, SetCKKSInfos};

/// # Safety
///
/// Implementations must satisfy the contracts of all trait methods, including
/// any HAL-level invariants (alignment, layout, scratch sizing) implied by the
/// associated method signatures.
pub unsafe trait CKKSNegImpl<BE: Backend>: Backend {
    fn ckks_neg_tmp_bytes(module: &Module<BE>) -> usize;

    fn ckks_neg_into<Dst, Src>(module: &Module<BE>, dst: &mut Dst, src: &Src, scratch: &mut ScratchArena<'_, BE>) -> Result<()>
    where
        Dst: GLWEToBackendMut<BE> + LWEInfos + CKKSInfos + SetCKKSInfos,
        Src: GLWEToBackendRef<BE> + GLWEInfos + LWEInfos + CKKSInfos;

    fn ckks_neg_assign<Dst>(module: &Module<BE>, dst: &mut Dst) -> Result<()>
    where
        Dst: GLWEToBackendMut<BE> + LWEInfos + CKKSInfos + SetCKKSInfos;
}

unsafe impl<BE: Backend> CKKSNegImpl<BE> for BE
where
    BE: poulpy_hal::oep::HalVecZnxImpl<BE>,
    Module<BE>: crate::default::neg::CKKSNegDefault<BE> + GLWENegate<BE> + GLWEShift<BE>,
    for<'a> ScratchArena<'a, BE>: ScratchArenaTakeCore<'a, BE>,
{
    fn ckks_neg_tmp_bytes(module: &Module<BE>) -> usize {
        module.ckks_neg_tmp_bytes_default()
    }

    fn ckks_neg_into<Dst, Src>(module: &Module<BE>, dst: &mut Dst, src: &Src, scratch: &mut ScratchArena<'_, BE>) -> Result<()>
    where
        Dst: GLWEToBackendMut<BE> + LWEInfos + CKKSInfos + SetCKKSInfos,
        Src: GLWEToBackendRef<BE> + GLWEInfos + LWEInfos + CKKSInfos,
    {
        module.ckks_neg_into_default(dst, src, scratch)
    }

    fn ckks_neg_assign<Dst>(module: &Module<BE>, dst: &mut Dst) -> Result<()>
    where
        Dst: GLWEToBackendMut<BE> + LWEInfos + CKKSInfos + SetCKKSInfos,
    {
        module.ckks_neg_assign_default(dst)
    }
}

#[macro_export]
macro_rules! impl_ckks_neg_defaults {
    ($be:ty) => {
        impl $crate::default::neg::CKKSNegDefault<$be> for ::poulpy_hal::layouts::Module<$be> {}
    };
}
pub use crate::impl_ckks_neg_defaults;
