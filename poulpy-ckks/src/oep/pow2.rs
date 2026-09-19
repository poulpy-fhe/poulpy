use crate::CKKSResult as Result;
use crate::reference::pow2::CKKSPow2Reference;

use poulpy_core::{GLWECopy, GLWEShift, layouts::GLWEInfos};
use poulpy_hal::layouts::{Backend, Module, ScratchArena};

use crate::{CKKSCtBounds, GLWEToBackendMut, GLWEToBackendRef, SetCKKSInfos};

/// # Safety
///
/// Implementations must satisfy the contracts of all trait methods, including
/// any HAL-level invariants (alignment, layout, scratch sizing) implied by the
/// associated method signatures.
pub unsafe trait CKKSPow2Impl: Backend {
    fn ckks_mul_pow2_tmp_bytes_impl(module: &Module<Self>, res_size: usize) -> usize;
    fn ckks_mul_pow2_into_impl<Dst, Src>(
        module: &Module<Self>,
        dst: &mut Dst,
        src: &Src,
        bits: usize,
        scratch: &mut ScratchArena<'_, Self>,
    ) -> Result<()>
    where
        Dst: GLWEToBackendMut<Self> + CKKSCtBounds + SetCKKSInfos,
        Src: GLWEToBackendRef<Self> + GLWEInfos + CKKSCtBounds;
    fn ckks_mul_pow2_assign_impl<Dst>(
        module: &Module<Self>,
        dst: &mut Dst,
        bits: usize,
        scratch: &mut ScratchArena<'_, Self>,
    ) -> Result<()>
    where
        Dst: GLWEToBackendMut<Self> + CKKSCtBounds + SetCKKSInfos;
    fn ckks_div_pow2_tmp_bytes_impl(module: &Module<Self>, res_size: usize) -> usize;
    fn ckks_div_pow2_into_impl<Dst, Src>(
        module: &Module<Self>,
        dst: &mut Dst,
        src: &Src,
        bits: usize,
        scratch: &mut ScratchArena<'_, Self>,
    ) -> Result<()>
    where
        Dst: GLWEToBackendMut<Self> + CKKSCtBounds + SetCKKSInfos,
        Src: GLWEToBackendRef<Self> + GLWEInfos + CKKSCtBounds;
    fn ckks_div_pow2_assign_impl<Dst>(module: &Module<Self>, dst: &mut Dst, bits: usize) -> Result<()>
    where
        Dst: GLWEToBackendMut<Self> + CKKSCtBounds + SetCKKSInfos;
}

unsafe impl<BE: Backend> CKKSPow2Impl for BE
where
    BE: poulpy_hal::oep::HalVecZnxImpl,
    Module<BE>: crate::reference::pow2::CKKSPow2Reference<BE> + GLWECopy<BE> + GLWEShift<BE>,
{
    fn ckks_mul_pow2_tmp_bytes_impl(module: &Module<BE>, res_size: usize) -> usize {
        module.ckks_mul_pow2_tmp_bytes_reference(res_size)
    }

    fn ckks_mul_pow2_into_impl<Dst, Src>(
        module: &Module<BE>,
        dst: &mut Dst,
        src: &Src,
        bits: usize,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        Dst: GLWEToBackendMut<BE> + CKKSCtBounds + SetCKKSInfos,
        Src: GLWEToBackendRef<BE> + GLWEInfos + CKKSCtBounds,
    {
        module.ckks_mul_pow2_into_reference(dst, src, bits, scratch)
    }

    fn ckks_mul_pow2_assign_impl<Dst>(
        module: &Module<BE>,
        dst: &mut Dst,
        bits: usize,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        Dst: GLWEToBackendMut<BE> + CKKSCtBounds + SetCKKSInfos,
    {
        module.ckks_mul_pow2_assign_reference(dst, bits, scratch)
    }

    fn ckks_div_pow2_tmp_bytes_impl(module: &Module<BE>, res_size: usize) -> usize {
        module.ckks_div_pow2_tmp_bytes_reference(res_size)
    }

    fn ckks_div_pow2_into_impl<Dst, Src>(
        module: &Module<BE>,
        dst: &mut Dst,
        src: &Src,
        bits: usize,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        Dst: GLWEToBackendMut<BE> + CKKSCtBounds + SetCKKSInfos,
        Src: GLWEToBackendRef<BE> + GLWEInfos + CKKSCtBounds,
    {
        module.ckks_div_pow2_into_reference(dst, src, bits, scratch)
    }

    fn ckks_div_pow2_assign_impl<Dst>(module: &Module<BE>, dst: &mut Dst, bits: usize) -> Result<()>
    where
        Dst: GLWEToBackendMut<BE> + CKKSCtBounds + SetCKKSInfos,
    {
        module.ckks_div_pow2_assign_reference(dst, bits)
    }
}

#[macro_export]
macro_rules! impl_ckks_pow2_reference {
    ($be:ty) => {
        impl $crate::reference::pow2::CKKSPow2Reference<$be> for ::poulpy_hal::layouts::Module<$be> {}
    };
}
pub use crate::impl_ckks_pow2_reference;
