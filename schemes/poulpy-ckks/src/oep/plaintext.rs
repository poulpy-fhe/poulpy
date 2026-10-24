use crate::CKKSResult as Result;
use poulpy_core::layouts::IntPolyInfos;

use poulpy_core::layouts::{GLWEInfos, GLWEToBackendMut, GLWEToBackendRef};
use poulpy_hal::layouts::{Backend, Module, ScratchArena};

use crate::{CKKSInfos, SetCKKSInfos};

/// # Safety
///
/// Implementations must satisfy the contracts of all trait methods, including
/// any HAL-level invariants (alignment, layout, scratch sizing) implied by the
/// associated method signatures.
pub unsafe trait CKKSPlaintextZnxImpl: Backend {
    fn ckks_extract_pt_tmp_bytes_impl(module: &Module<Self>, res_size: usize) -> usize;

    fn ckks_extract_pt_impl<Dst, Src>(
        module: &Module<Self>,
        dst: &mut Dst,
        src: &Src,
        scratch: &mut ScratchArena<'_, Self>,
    ) -> Result<()>
    where
        Dst: GLWEToBackendMut<Self> + CKKSInfos + IntPolyInfos + SetCKKSInfos,
        Src: GLWEToBackendRef<Self> + GLWEInfos + CKKSInfos;
}

/// Implements this contract with the callable reference algorithms.
#[macro_export]
macro_rules! impl_ckks_plaintext_reference {
    ($be:ty) => {
        unsafe impl $crate::oep::CKKSPlaintextZnxImpl for $be {
            fn ckks_extract_pt_tmp_bytes_impl(module: &::poulpy_hal::layouts::Module<Self>, res_size: usize) -> usize {
                $crate::reference::plaintext::CKKSPlaintextReference::ckks_extract_pt_tmp_bytes_reference(module, res_size)
            }

            fn ckks_extract_pt_impl<Dst, Src>(
                module: &::poulpy_hal::layouts::Module<Self>,
                dst: &mut Dst,
                src: &Src,
                scratch: &mut ::poulpy_hal::layouts::ScratchArena<'_, Self>,
            ) -> $crate::CKKSResult<()>
            where
                Dst: ::poulpy_core::layouts::GLWEToBackendMut<Self>
                    + $crate::CKKSInfos
                    + ::poulpy_core::layouts::IntPolyInfos
                    + $crate::SetCKKSInfos,
                Src: ::poulpy_core::layouts::GLWEToBackendRef<Self> + ::poulpy_core::layouts::GLWEInfos + $crate::CKKSInfos,
            {
                $crate::reference::plaintext::CKKSPlaintextReference::ckks_extract_pt_reference(module, dst, src, scratch)
            }
        }
    };
}
pub use crate::impl_ckks_plaintext_reference;
