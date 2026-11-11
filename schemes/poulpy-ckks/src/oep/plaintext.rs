use crate::CKKSResult as Result;
use poulpy_core::layouts::IntPolyInfos;

use poulpy_core::layouts::{GLWEInfos, GLWEToBackendMut, GLWEToBackendRef};
use poulpy_hal::layouts::{Backend, Module, Ring, ScratchArena, Standard};

use crate::{CKKSInfos, SetCKKSInfos};

/// Backend contract of [`CKKSPlaintextVecOps`](crate::api::CKKSPlaintextVecOps).
///
/// # Safety
/// Implementations reproduce the bodies their methods document bit for bit, report for each
/// `_tmp_bytes` query a bound that covers the body, and refuse as the reference does. The memory,
/// layout, synchrony and aliasing rules of `poulpy-hal/docs/backend_safety_contract.md` apply.
pub unsafe trait CKKSPlaintextVecImpl<Rg: Ring = Standard>: Backend {
    /// Scratch bound of [`Self::ckks_extract_pt`]; may report less than the reference, never less than the body takes.
    fn ckks_extract_pt_tmp_bytes(module: &Module<Self>, res_size: usize) -> usize;

    /// Reproduces
    /// [`ckks_extract_pt_reference`](crate::reference::plaintext::CKKSPlaintextVecReference::ckks_extract_pt_reference)
    /// bit for bit, within [`Self::ckks_extract_pt_tmp_bytes`]; free in kernels and scratch use.
    fn ckks_extract_pt<D, S>(module: &Module<Self>, dst: &mut D, src: &S, scratch: &mut ScratchArena<'_, Self>) -> Result<()>
    where
        D: GLWEToBackendMut<Self, Rg> + CKKSInfos + IntPolyInfos + SetCKKSInfos,
        S: GLWEToBackendRef<Self, Rg> + GLWEInfos + CKKSInfos;
}

/// Selects the reference implementation of `CKKSPlaintextVecImpl` for `$be`.
#[macro_export]
macro_rules! impl_ckks_plaintext_vec_reference_full {
    ($be:ty) => {
        $crate::impl_ckks_plaintext_vec_reference_full!($be, ::poulpy_hal::layouts::Standard);
        $crate::impl_ckks_plaintext_vec_reference_full!($be, ::poulpy_hal::layouts::ConjugateInvariant);
    };
    ($be:ty, $ring:ty) => {
        unsafe impl $crate::oep::CKKSPlaintextVecImpl<$ring> for $be {
            fn ckks_extract_pt_tmp_bytes(module: &::poulpy_hal::layouts::Module<Self>, res_size: usize) -> usize {
                $crate::reference::plaintext::CKKSPlaintextVecReference::<Self, $ring>::ckks_extract_pt_tmp_bytes_reference(
                    module, res_size,
                )
            }

            fn ckks_extract_pt<D, S>(
                module: &::poulpy_hal::layouts::Module<Self>,
                dst: &mut D,
                src: &S,
                scratch: &mut ::poulpy_hal::layouts::ScratchArena<'_, Self>,
            ) -> $crate::CKKSResult<()>
            where
                D: ::poulpy_core::layouts::GLWEToBackendMut<Self, $ring>
                    + $crate::CKKSInfos
                    + ::poulpy_core::layouts::IntPolyInfos
                    + $crate::SetCKKSInfos,
                S: ::poulpy_core::layouts::GLWEToBackendRef<Self, $ring> + ::poulpy_core::layouts::GLWEInfos + $crate::CKKSInfos,
            {
                $crate::reference::plaintext::CKKSPlaintextVecReference::<Self, $ring>::ckks_extract_pt_reference(
                    module, dst, src, scratch,
                )
            }
        }
    };
}
pub use crate::impl_ckks_plaintext_vec_reference_full;
