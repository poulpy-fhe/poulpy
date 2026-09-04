use crate::CKKSResult as Result;
use crate::default::pow2::CKKSPow2Default;
use poulpy_hal::layouts::CoeffNormalized;

use poulpy_core::{GLWECopy, GLWEShift, layouts::GLWEInfos};
use poulpy_hal::layouts::{Backend, Module, ScratchArena};

use crate::{CKKSCtBounds, GLWEToBackendMut, GLWEToBackendRef, SetCKKSInfos};

/// # Safety
///
/// Implementations must satisfy the contracts of all trait methods, including
/// any HAL-level invariants (alignment, layout, scratch sizing) implied by the
/// associated method signatures.
pub unsafe trait CKKSPow2Impl<BE: Backend>: Backend {
    fn ckks_mul_pow2_tmp_bytes_impl(module: &Module<BE>) -> usize;
    fn ckks_mul_pow2_into_impl<Dst, Src>(
        module: &Module<BE>,
        dst: &mut Dst,
        src: &Src,
        bits: usize,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        Dst: GLWEToBackendMut<BE, State = CoeffNormalized> + CKKSCtBounds + SetCKKSInfos,
        Src: GLWEToBackendRef<BE, State = CoeffNormalized> + GLWEInfos + CKKSCtBounds;
    fn ckks_mul_pow2_assign_impl<Dst>(
        module: &Module<BE>,
        dst: &mut Dst,
        bits: usize,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        Dst: GLWEToBackendMut<BE, State = CoeffNormalized> + CKKSCtBounds + SetCKKSInfos;
    fn ckks_div_pow2_tmp_bytes_impl(module: &Module<BE>) -> usize;
    fn ckks_div_pow2_into_impl<Dst, Src>(
        module: &Module<BE>,
        dst: &mut Dst,
        src: &Src,
        bits: usize,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        Dst: GLWEToBackendMut<BE, State = CoeffNormalized> + CKKSCtBounds + SetCKKSInfos,
        Src: GLWEToBackendRef<BE, State = CoeffNormalized> + GLWEInfos + CKKSCtBounds;
    fn ckks_div_pow2_assign_impl<Dst>(module: &Module<BE>, dst: &mut Dst, bits: usize) -> Result<()>
    where
        Dst: GLWEToBackendMut<BE, State = CoeffNormalized> + CKKSCtBounds + SetCKKSInfos;
}

unsafe impl<BE: Backend> CKKSPow2Impl<BE> for BE
where
    BE: poulpy_hal::oep::HalVecZnxImpl<BE>,
    Module<BE>: crate::default::pow2::CKKSPow2Default<BE> + GLWECopy<BE> + GLWEShift<BE>,
{
    fn ckks_mul_pow2_tmp_bytes_impl(module: &Module<BE>) -> usize {
        module.ckks_mul_pow2_tmp_bytes_default()
    }

    fn ckks_mul_pow2_into_impl<Dst, Src>(
        module: &Module<BE>,
        dst: &mut Dst,
        src: &Src,
        bits: usize,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        Dst: GLWEToBackendMut<BE, State = CoeffNormalized> + CKKSCtBounds + SetCKKSInfos,
        Src: GLWEToBackendRef<BE, State = CoeffNormalized> + GLWEInfos + CKKSCtBounds,
    {
        module.ckks_mul_pow2_into_default(dst, src, bits, scratch)
    }

    fn ckks_mul_pow2_assign_impl<Dst>(
        module: &Module<BE>,
        dst: &mut Dst,
        bits: usize,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        Dst: GLWEToBackendMut<BE, State = CoeffNormalized> + CKKSCtBounds + SetCKKSInfos,
    {
        module.ckks_mul_pow2_assign_default(dst, bits, scratch)
    }

    fn ckks_div_pow2_tmp_bytes_impl(module: &Module<BE>) -> usize {
        module.ckks_div_pow2_tmp_bytes_default()
    }

    fn ckks_div_pow2_into_impl<Dst, Src>(
        module: &Module<BE>,
        dst: &mut Dst,
        src: &Src,
        bits: usize,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        Dst: GLWEToBackendMut<BE, State = CoeffNormalized> + CKKSCtBounds + SetCKKSInfos,
        Src: GLWEToBackendRef<BE, State = CoeffNormalized> + GLWEInfos + CKKSCtBounds,
    {
        module.ckks_div_pow2_into_default(dst, src, bits, scratch)
    }

    fn ckks_div_pow2_assign_impl<Dst>(module: &Module<BE>, dst: &mut Dst, bits: usize) -> Result<()>
    where
        Dst: GLWEToBackendMut<BE, State = CoeffNormalized> + CKKSCtBounds + SetCKKSInfos,
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
