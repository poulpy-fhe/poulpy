use crate::CKKSResult as Result;

use poulpy_core::layouts::GLWEInfos;
use poulpy_hal::layouts::{Backend, Module, ScratchArena};

use crate::{CKKSCtBounds, GLWEToBackendMut, GLWEToBackendRef, SetCKKSInfos};

/// Backend contract of [`CKKSPow2Ops`](crate::api::CKKSPow2Ops).
///
/// # Safety
/// Implementations reproduce the bodies their methods document bit for bit, report for each
/// `_tmp_bytes` query a bound that covers the body, and refuse as the reference does. The memory,
/// layout, synchrony and aliasing rules of `poulpy-hal/docs/backend_safety_contract.md` apply.
pub unsafe trait CKKSPow2Impl: Backend {
    /// Scratch bound of [`Self::ckks_mul_pow2_into`], [`Self::ckks_mul_pow2_assign`] and [`Self::ckks_double_into`];
    /// may report less than the reference, never less than the body takes.
    fn ckks_mul_pow2_tmp_bytes(module: &Module<Self>, res_size: usize) -> usize;

    /// Reproduces
    /// [`ckks_mul_pow2_into_reference`](crate::reference::pow2::CKKSPow2Reference::ckks_mul_pow2_into_reference) bit
    /// for bit, within [`Self::ckks_mul_pow2_tmp_bytes`]; free in kernels and scratch use.
    fn ckks_mul_pow2_into<Dst, Src>(
        module: &Module<Self>,
        bits: usize,
        dst: &mut Dst,
        src: &Src,
        scratch: &mut ScratchArena<'_, Self>,
    ) -> Result<()>
    where
        Dst: GLWEToBackendMut<Self> + CKKSCtBounds + SetCKKSInfos,
        Src: GLWEToBackendRef<Self> + GLWEInfos + CKKSCtBounds;

    /// Reproduces
    /// [`ckks_mul_pow2_assign_reference`](crate::reference::pow2::CKKSPow2Reference::ckks_mul_pow2_assign_reference)
    /// bit for bit, within [`Self::ckks_mul_pow2_tmp_bytes`]; free in kernels and scratch use.
    fn ckks_mul_pow2_assign<Dst>(
        module: &Module<Self>,
        bits: usize,
        dst: &mut Dst,
        scratch: &mut ScratchArena<'_, Self>,
    ) -> Result<()>
    where
        Dst: GLWEToBackendMut<Self> + CKKSCtBounds + SetCKKSInfos;

    /// Reproduces [`ckks_double_into_reference`](crate::reference::pow2::CKKSPow2Reference::ckks_double_into_reference)
    /// bit for bit, within [`Self::ckks_mul_pow2_tmp_bytes`]; free in kernels and scratch use.
    fn ckks_double_into<Dst, Src>(
        module: &Module<Self>,
        dst: &mut Dst,
        src: &Src,
        scratch: &mut ScratchArena<'_, Self>,
    ) -> Result<()>
    where
        Dst: GLWEToBackendMut<Self> + CKKSCtBounds + SetCKKSInfos,
        Src: GLWEToBackendRef<Self> + GLWEInfos + CKKSCtBounds;

    /// Scratch bound of [`Self::ckks_div_pow2_into`]; may report less than the reference, never less than the body
    /// takes.
    fn ckks_div_pow2_tmp_bytes(module: &Module<Self>, res_size: usize) -> usize;

    /// Reproduces
    /// [`ckks_div_pow2_into_reference`](crate::reference::pow2::CKKSPow2Reference::ckks_div_pow2_into_reference) bit
    /// for bit, within [`Self::ckks_div_pow2_tmp_bytes`]; free in kernels and scratch use.
    fn ckks_div_pow2_into<Dst, Src>(
        module: &Module<Self>,
        bits: usize,
        dst: &mut Dst,
        src: &Src,
        scratch: &mut ScratchArena<'_, Self>,
    ) -> Result<()>
    where
        Dst: GLWEToBackendMut<Self> + CKKSCtBounds + SetCKKSInfos,
        Src: GLWEToBackendRef<Self> + GLWEInfos + CKKSCtBounds;

    /// Reproduces
    /// [`ckks_div_pow2_assign_reference`](crate::reference::pow2::CKKSPow2Reference::ckks_div_pow2_assign_reference)
    /// bit for bit; free in kernels.
    fn ckks_div_pow2_assign<Dst>(module: &Module<Self>, bits: usize, dst: &mut Dst) -> Result<()>
    where
        Dst: GLWEToBackendMut<Self> + CKKSCtBounds + SetCKKSInfos;
}

/// Selects the reference implementation of `CKKSPow2Impl` for `$be`.
#[macro_export]
macro_rules! impl_ckks_pow2_reference_full {
    ($be:ty) => {
        unsafe impl $crate::oep::CKKSPow2Impl for $be {
            fn ckks_mul_pow2_tmp_bytes(module: &::poulpy_hal::layouts::Module<Self>, res_size: usize) -> usize {
                $crate::reference::pow2::CKKSPow2Reference::ckks_mul_pow2_tmp_bytes_reference(module, res_size)
            }

            fn ckks_mul_pow2_into<Dst, Src>(
                module: &::poulpy_hal::layouts::Module<Self>,
                bits: usize,
                dst: &mut Dst,
                src: &Src,
                scratch: &mut ::poulpy_hal::layouts::ScratchArena<'_, Self>,
            ) -> $crate::CKKSResult<()>
            where
                Dst: ::poulpy_core::layouts::GLWEToBackendMut<Self> + $crate::CKKSCtBounds + $crate::SetCKKSInfos,
                Src: ::poulpy_core::layouts::GLWEToBackendRef<Self> + ::poulpy_core::layouts::GLWEInfos + $crate::CKKSCtBounds,
            {
                $crate::reference::pow2::CKKSPow2Reference::ckks_mul_pow2_into_reference(module, bits, dst, src, scratch)
            }

            fn ckks_mul_pow2_assign<Dst>(
                module: &::poulpy_hal::layouts::Module<Self>,
                bits: usize,
                dst: &mut Dst,
                scratch: &mut ::poulpy_hal::layouts::ScratchArena<'_, Self>,
            ) -> $crate::CKKSResult<()>
            where
                Dst: ::poulpy_core::layouts::GLWEToBackendMut<Self> + $crate::CKKSCtBounds + $crate::SetCKKSInfos,
            {
                $crate::reference::pow2::CKKSPow2Reference::ckks_mul_pow2_assign_reference(module, bits, dst, scratch)
            }

            fn ckks_double_into<Dst, Src>(
                module: &::poulpy_hal::layouts::Module<Self>,
                dst: &mut Dst,
                src: &Src,
                scratch: &mut ::poulpy_hal::layouts::ScratchArena<'_, Self>,
            ) -> $crate::CKKSResult<()>
            where
                Dst: ::poulpy_core::layouts::GLWEToBackendMut<Self> + $crate::CKKSCtBounds + $crate::SetCKKSInfos,
                Src: ::poulpy_core::layouts::GLWEToBackendRef<Self> + ::poulpy_core::layouts::GLWEInfos + $crate::CKKSCtBounds,
            {
                $crate::reference::pow2::CKKSPow2Reference::ckks_double_into_reference(module, dst, src, scratch)
            }

            fn ckks_div_pow2_tmp_bytes(module: &::poulpy_hal::layouts::Module<Self>, res_size: usize) -> usize {
                $crate::reference::pow2::CKKSPow2Reference::ckks_div_pow2_tmp_bytes_reference(module, res_size)
            }

            fn ckks_div_pow2_into<Dst, Src>(
                module: &::poulpy_hal::layouts::Module<Self>,
                bits: usize,
                dst: &mut Dst,
                src: &Src,
                scratch: &mut ::poulpy_hal::layouts::ScratchArena<'_, Self>,
            ) -> $crate::CKKSResult<()>
            where
                Dst: ::poulpy_core::layouts::GLWEToBackendMut<Self> + $crate::CKKSCtBounds + $crate::SetCKKSInfos,
                Src: ::poulpy_core::layouts::GLWEToBackendRef<Self> + ::poulpy_core::layouts::GLWEInfos + $crate::CKKSCtBounds,
            {
                $crate::reference::pow2::CKKSPow2Reference::ckks_div_pow2_into_reference(module, bits, dst, src, scratch)
            }

            fn ckks_div_pow2_assign<Dst>(
                module: &::poulpy_hal::layouts::Module<Self>,
                bits: usize,
                dst: &mut Dst,
            ) -> $crate::CKKSResult<()>
            where
                Dst: ::poulpy_core::layouts::GLWEToBackendMut<Self> + $crate::CKKSCtBounds + $crate::SetCKKSInfos,
            {
                $crate::reference::pow2::CKKSPow2Reference::ckks_div_pow2_assign_reference(module, bits, dst)
            }
        }
    };
}
pub use crate::impl_ckks_pow2_reference_full;
