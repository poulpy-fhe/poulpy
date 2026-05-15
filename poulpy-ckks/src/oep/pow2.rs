use crate::default::pow2::CKKSPow2Default;

use anyhow::Result;
use poulpy_core::{
    GLWECopy, GLWEShift, ScratchArenaTakeCore,
    layouts::{GLWEInfos, LWEInfos},
};
use poulpy_hal::layouts::{Backend, Module, ScratchArena};

use crate::{CKKSInfos, GLWEToBackendMut, GLWEToBackendRef, SetCKKSInfos};

/// # Safety
///
/// Implementations must satisfy the contracts of all trait methods, including
/// any HAL-level invariants (alignment, layout, scratch sizing) implied by the
/// associated method signatures.
pub unsafe trait CKKSPow2Impl<BE: Backend>: Backend {
    fn ckks_mul_pow2_tmp_bytes(module: &Module<BE>) -> usize;
    fn ckks_mul_pow2_into<Dst, Src>(
        module: &Module<BE>,
        dst: &mut Dst,
        src: &Src,
        bits: usize,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        Dst: GLWEToBackendMut<BE> + LWEInfos + CKKSInfos + SetCKKSInfos,
        Src: GLWEToBackendRef<BE> + GLWEInfos + LWEInfos + CKKSInfos;
    fn ckks_mul_pow2_assign<Dst>(
        module: &Module<BE>,
        dst: &mut Dst,
        bits: usize,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        Dst: GLWEToBackendMut<BE> + LWEInfos + CKKSInfos + SetCKKSInfos;
    fn ckks_div_pow2_tmp_bytes(module: &Module<BE>) -> usize;
    fn ckks_div_pow2_into<Dst, Src>(
        module: &Module<BE>,
        dst: &mut Dst,
        src: &Src,
        bits: usize,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        Dst: GLWEToBackendMut<BE> + LWEInfos + CKKSInfos + SetCKKSInfos,
        Src: GLWEToBackendRef<BE> + GLWEInfos + LWEInfos + CKKSInfos;
    fn ckks_div_pow2_assign<Dst>(module: &Module<BE>, dst: &mut Dst, bits: usize) -> Result<()>
    where
        Dst: GLWEToBackendMut<BE> + LWEInfos + CKKSInfos + SetCKKSInfos;
}

unsafe impl<BE: Backend> CKKSPow2Impl<BE> for BE
where
    BE: poulpy_hal::oep::HalVecZnxImpl<BE>,
    Module<BE>: crate::default::pow2::CKKSPow2Default<BE> + GLWECopy<BE> + GLWEShift<BE>,
    for<'a> ScratchArena<'a, BE>: ScratchArenaTakeCore<'a, BE>,
{
    fn ckks_mul_pow2_tmp_bytes(module: &Module<BE>) -> usize {
        module.ckks_mul_pow2_tmp_bytes_default()
    }

    fn ckks_mul_pow2_into<Dst, Src>(
        module: &Module<BE>,
        dst: &mut Dst,
        src: &Src,
        bits: usize,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        Dst: GLWEToBackendMut<BE> + LWEInfos + CKKSInfos + SetCKKSInfos,
        Src: GLWEToBackendRef<BE> + GLWEInfos + LWEInfos + CKKSInfos,
    {
        module.ckks_mul_pow2_into_default(dst, src, bits, scratch)
    }

    fn ckks_mul_pow2_assign<Dst>(
        module: &Module<BE>,
        dst: &mut Dst,
        bits: usize,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        Dst: GLWEToBackendMut<BE> + LWEInfos + CKKSInfos + SetCKKSInfos,
    {
        module.ckks_mul_pow2_assign_default(dst, bits, scratch)
    }

    fn ckks_div_pow2_tmp_bytes(module: &Module<BE>) -> usize {
        module.ckks_div_pow2_tmp_bytes_default()
    }

    fn ckks_div_pow2_into<Dst, Src>(
        module: &Module<BE>,
        dst: &mut Dst,
        src: &Src,
        bits: usize,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        Dst: GLWEToBackendMut<BE> + LWEInfos + CKKSInfos + SetCKKSInfos,
        Src: GLWEToBackendRef<BE> + GLWEInfos + LWEInfos + CKKSInfos,
    {
        module.ckks_div_pow2_into_default(dst, src, bits, scratch)
    }

    fn ckks_div_pow2_assign<Dst>(module: &Module<BE>, dst: &mut Dst, bits: usize) -> Result<()>
    where
        Dst: GLWEToBackendMut<BE> + LWEInfos + CKKSInfos + SetCKKSInfos,
    {
        module.ckks_div_pow2_assign_default(dst, bits)
    }
}

#[macro_export]
macro_rules! impl_ckks_pow2_defaults {
    ($be:ty) => {
        impl $crate::default::pow2::CKKSPow2Default<$be> for ::poulpy_hal::layouts::Module<$be> {}
    };
}
pub use crate::impl_ckks_pow2_defaults;
