use crate::default::rescale::CKKSRescaleOpsDefault;

use anyhow::Result;
use poulpy_core::{
    GLWEShift, ScratchArenaTakeCore,
    layouts::{GLWEInfos, LWEInfos},
};
use poulpy_hal::layouts::{Backend, Module, ScratchArena};

use crate::{CKKSInfos, GLWEToBackendMut, GLWEToBackendRef, SetCKKSInfos};

/// # Safety
///
/// Implementations must satisfy the contracts of all trait methods, including
/// any HAL-level invariants (alignment, layout, scratch sizing) implied by the
/// associated method signatures.
pub unsafe trait CKKSRescaleImpl<BE: Backend>: Backend {
    fn ckks_rescale_tmp_bytes(module: &Module<BE>) -> usize;
    fn ckks_rescale_assign<Dst>(module: &Module<BE>, ct: &mut Dst, k: usize, scratch: &mut ScratchArena<'_, BE>) -> Result<()>
    where
        Dst: GLWEToBackendMut<BE> + LWEInfos + CKKSInfos + SetCKKSInfos;
    fn ckks_rescale_into<Dst, Src>(
        module: &Module<BE>,
        dst: &mut Dst,
        k: usize,
        src: &Src,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        Dst: GLWEToBackendMut<BE> + LWEInfos + CKKSInfos + SetCKKSInfos,
        Src: GLWEToBackendRef<BE> + GLWEInfos + LWEInfos + CKKSInfos;
    fn ckks_align_pair<A, B>(module: &Module<BE>, a: &mut A, b: &mut B, scratch: &mut ScratchArena<'_, BE>) -> Result<()>
    where
        A: GLWEToBackendMut<BE> + LWEInfos + CKKSInfos + SetCKKSInfos,
        B: GLWEToBackendMut<BE> + LWEInfos + CKKSInfos + SetCKKSInfos;
    fn ckks_align_tmp_bytes(module: &Module<BE>) -> usize;
}

#[allow(private_bounds)]
unsafe impl<BE: Backend> CKKSRescaleImpl<BE> for BE
where
    BE: poulpy_hal::oep::HalVecZnxImpl<BE>,
    Module<BE>: crate::default::rescale::CKKSRescaleOpsDefault<BE> + GLWEShift<BE>,
    for<'a> ScratchArena<'a, BE>: ScratchArenaTakeCore<'a, BE>,
{
    fn ckks_rescale_tmp_bytes(module: &Module<BE>) -> usize {
        module.ckks_rescale_tmp_bytes_default()
    }

    fn ckks_rescale_assign<Dst>(module: &Module<BE>, ct: &mut Dst, k: usize, scratch: &mut ScratchArena<'_, BE>) -> Result<()>
    where
        Dst: GLWEToBackendMut<BE> + LWEInfos + CKKSInfos + SetCKKSInfos,
    {
        module.ckks_rescale_assign_default(ct, k, scratch)
    }

    fn ckks_rescale_into<Dst, Src>(
        module: &Module<BE>,
        dst: &mut Dst,
        k: usize,
        src: &Src,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        Dst: GLWEToBackendMut<BE> + LWEInfos + CKKSInfos + SetCKKSInfos,
        Src: GLWEToBackendRef<BE> + GLWEInfos + LWEInfos + CKKSInfos,
    {
        module.ckks_rescale_into_default(dst, k, src, scratch)
    }

    fn ckks_align_pair<A, B>(module: &Module<BE>, a: &mut A, b: &mut B, scratch: &mut ScratchArena<'_, BE>) -> Result<()>
    where
        A: GLWEToBackendMut<BE> + LWEInfos + CKKSInfos + SetCKKSInfos,
        B: GLWEToBackendMut<BE> + LWEInfos + CKKSInfos + SetCKKSInfos,
    {
        module.ckks_align_pair_default(a, b, scratch)
    }

    fn ckks_align_tmp_bytes(module: &Module<BE>) -> usize {
        module.ckks_align_tmp_bytes_default()
    }
}

#[macro_export]
macro_rules! impl_ckks_rescale_defaults {
    ($be:ty) => {
        impl $crate::default::rescale::CKKSRescaleOpsDefault<$be> for ::poulpy_hal::layouts::Module<$be> {}
    };
}
pub use crate::impl_ckks_rescale_defaults;
