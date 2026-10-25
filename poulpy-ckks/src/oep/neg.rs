use crate::CKKSResult as Result;

use poulpy_core::layouts::GLWEInfos;
use poulpy_hal::layouts::{Backend, Module, ScratchArena};

use crate::{CKKSCtBounds, GLWEToBackendMut, GLWEToBackendRef, SetCKKSInfos};

/// Backend contract of [`CKKSNegOps`](crate::api::CKKSNegOps).
///
/// # Safety
/// Implementations reproduce the bodies their methods document bit for bit, report for each
/// `_tmp_bytes` query a bound that covers the body, and refuse as the reference does. The memory,
/// layout, synchrony and aliasing rules of `poulpy-hal/docs/backend_safety_contract.md` apply.
pub unsafe trait CKKSNegImpl: Backend {
    /// Scratch bound of [`Self::ckks_neg_into`] and [`Self::ckks_neg_assign`]; may report less than the reference,
    /// never less than the body takes.
    fn ckks_neg_tmp_bytes(module: &Module<Self>, res_size: usize) -> usize;

    /// Reproduces [`ckks_neg_into_reference`](crate::reference::neg::CKKSNegReference::ckks_neg_into_reference) bit for
    /// bit, within [`Self::ckks_neg_tmp_bytes`]; free in kernels and scratch use.
    fn ckks_neg_into<Dst, Src>(
        module: &Module<Self>,
        dst: &mut Dst,
        src: &Src,
        scratch: &mut ScratchArena<'_, Self>,
    ) -> Result<()>
    where
        Dst: GLWEToBackendMut<Self> + CKKSCtBounds + SetCKKSInfos,
        Src: GLWEToBackendRef<Self> + GLWEInfos + CKKSCtBounds;

    /// Reproduces [`ckks_neg_assign_reference`](crate::reference::neg::CKKSNegReference::ckks_neg_assign_reference) bit
    /// for bit, within [`Self::ckks_neg_tmp_bytes`]; free in kernels and scratch use.
    fn ckks_neg_assign<Dst>(module: &Module<Self>, dst: &mut Dst) -> Result<()>
    where
        Dst: GLWEToBackendMut<Self> + CKKSCtBounds + SetCKKSInfos;
}

/// Selects the reference implementation of `CKKSNegImpl` for `$be`.
#[macro_export]
macro_rules! impl_ckks_neg_reference_full {
    ($be:ty) => {
        unsafe impl $crate::oep::CKKSNegImpl for $be {
            fn ckks_neg_tmp_bytes(module: &::poulpy_hal::layouts::Module<Self>, res_size: usize) -> usize {
                $crate::reference::neg::CKKSNegReference::ckks_neg_tmp_bytes_reference(module, res_size)
            }

            fn ckks_neg_into<Dst, Src>(
                module: &::poulpy_hal::layouts::Module<Self>,
                dst: &mut Dst,
                src: &Src,
                scratch: &mut ::poulpy_hal::layouts::ScratchArena<'_, Self>,
            ) -> $crate::CKKSResult<()>
            where
                Dst: ::poulpy_core::layouts::GLWEToBackendMut<Self> + $crate::CKKSCtBounds + $crate::SetCKKSInfos,
                Src: ::poulpy_core::layouts::GLWEToBackendRef<Self> + ::poulpy_core::layouts::GLWEInfos + $crate::CKKSCtBounds,
            {
                $crate::reference::neg::CKKSNegReference::ckks_neg_into_reference(module, dst, src, scratch)
            }

            fn ckks_neg_assign<Dst>(module: &::poulpy_hal::layouts::Module<Self>, dst: &mut Dst) -> $crate::CKKSResult<()>
            where
                Dst: ::poulpy_core::layouts::GLWEToBackendMut<Self> + $crate::CKKSCtBounds + $crate::SetCKKSInfos,
            {
                $crate::reference::neg::CKKSNegReference::ckks_neg_assign_reference(module, dst)
            }
        }
    };
}
pub use crate::impl_ckks_neg_reference_full;
