#![allow(clippy::too_many_arguments)]

use crate::CKKSResult as Result;
use poulpy_core::layouts::IntPolyInfos;

use poulpy_core::layouts::{GLWEInfos, GLWESecretPreparedToBackendRef};
use poulpy_hal::{
    layouts::{Backend, Module, Ring, ScratchArena, Standard},
    source::Source,
};

use crate::{CKKSCtBounds, CKKSInfos, GLWEToBackendMut, GLWEToBackendRef, SetCKKSInfos};

/// Backend contract of [`CKKSEncryptOps`](crate::api::CKKSEncryptOps) and
/// [`CKKSDecryptOps`](crate::api::CKKSDecryptOps).
///
/// # Safety
/// Implementations reproduce the bodies their methods document bit for bit, report for each
/// `_tmp_bytes` query a bound that covers the body, and refuse as the reference does. The memory,
/// layout, synchrony and aliasing rules of `poulpy-hal/docs/backend_safety_contract.md` apply.
pub unsafe trait CKKSEncryptionImpl<Rg: Ring = Standard>: Backend {
    /// Scratch bound of [`Self::ckks_encrypt_sk`]; may report less than the reference, never less than the body takes.
    fn ckks_encrypt_sk_tmp_bytes<A>(module: &Module<Self>, ct_infos: &A) -> usize
    where
        A: CKKSCtBounds;

    /// Reproduces
    /// [`ckks_encrypt_sk_reference`](crate::reference::encryption::CKKSEncryptionReference::ckks_encrypt_sk_reference)
    /// bit for bit, within [`Self::ckks_encrypt_sk_tmp_bytes`]; free in kernels and scratch use.
    fn ckks_encrypt_sk<Dct, Dpt, S>(
        module: &Module<Self>,
        ct: &mut Dct,
        pt: &Dpt,
        sk: &S,
        source_xe: &mut Source,
        source_xa: &mut Source,
        scratch: &mut ScratchArena<'_, Self>,
    ) -> Result<()>
    where
        Dpt: GLWEToBackendRef<Self, Rg> + IntPolyInfos + CKKSCtBounds,
        Dct: GLWEToBackendMut<Self, Rg> + CKKSCtBounds + SetCKKSInfos,
        S: GLWESecretPreparedToBackendRef<Self, Rg>;

    /// Scratch bound of [`Self::ckks_decrypt`]; may report less than the reference, never less than the body takes.
    fn ckks_decrypt_tmp_bytes<Pt, Ct>(module: &Module<Self>, pt_infos: &Pt, ct_infos: &Ct) -> usize
    where
        Pt: CKKSInfos,
        Ct: CKKSCtBounds;

    /// Reproduces
    /// [`ckks_decrypt_reference`](crate::reference::encryption::CKKSEncryptionReference::ckks_decrypt_reference) bit
    /// for bit, within [`Self::ckks_decrypt_tmp_bytes`]; free in kernels and scratch use.
    fn ckks_decrypt<Dpt, Dct, S>(
        module: &Module<Self>,
        pt: &mut Dpt,
        ct: &Dct,
        sk: &S,
        scratch: &mut ScratchArena<'_, Self>,
    ) -> Result<()>
    where
        Dpt: GLWEToBackendMut<Self, Rg> + CKKSCtBounds + SetCKKSInfos + IntPolyInfos,
        Dct: GLWEToBackendRef<Self, Rg> + GLWEInfos + CKKSCtBounds,
        S: GLWESecretPreparedToBackendRef<Self, Rg> + GLWEInfos;
}

/// Selects the reference implementation of `CKKSEncryptionImpl` for `$be`.
#[macro_export]
macro_rules! impl_ckks_encryption_reference_full {
    ($be:ty) => {
        $crate::impl_ckks_encryption_reference_full!($be, ::poulpy_hal::layouts::Standard);
        $crate::impl_ckks_encryption_reference_full!($be, ::poulpy_hal::layouts::ConjugateInvariant);
    };
    ($be:ty, $ring:ty) => {
        unsafe impl $crate::oep::CKKSEncryptionImpl<$ring> for $be {
            fn ckks_encrypt_sk_tmp_bytes<A>(module: &::poulpy_hal::layouts::Module<Self>, ct_infos: &A) -> usize
            where
                A: $crate::CKKSCtBounds,
            {
                $crate::reference::encryption::CKKSEncryptionReference::<Self, $ring>::ckks_encrypt_sk_tmp_bytes_reference(
                    module, ct_infos,
                )
            }

            fn ckks_encrypt_sk<Dct, Dpt, S>(
                module: &::poulpy_hal::layouts::Module<Self>,
                ct: &mut Dct,
                pt: &Dpt,
                sk: &S,
                source_xe: &mut ::poulpy_hal::source::Source,
                source_xa: &mut ::poulpy_hal::source::Source,
                scratch: &mut ::poulpy_hal::layouts::ScratchArena<'_, Self>,
            ) -> $crate::CKKSResult<()>
            where
                Dpt: ::poulpy_core::layouts::GLWEToBackendRef<Self, $ring>
                    + $crate::CKKSCtBounds
                    + ::poulpy_core::layouts::IntPolyInfos,
                Dct: ::poulpy_core::layouts::GLWEToBackendMut<Self, $ring> + $crate::CKKSCtBounds + $crate::SetCKKSInfos,
                S: ::poulpy_core::layouts::GLWESecretPreparedToBackendRef<Self, $ring>,
            {
                $crate::reference::encryption::CKKSEncryptionReference::<Self, $ring>::ckks_encrypt_sk_reference(
                    module, ct, pt, sk, source_xe, source_xa, scratch,
                )
            }

            fn ckks_decrypt_tmp_bytes<Pt, Ct>(module: &::poulpy_hal::layouts::Module<Self>, pt_infos: &Pt, ct_infos: &Ct) -> usize
            where
                Pt: $crate::CKKSInfos,
                Ct: $crate::CKKSCtBounds,
            {
                $crate::reference::encryption::CKKSEncryptionReference::<Self, $ring>::ckks_decrypt_tmp_bytes_reference(
                    module, pt_infos, ct_infos,
                )
            }

            fn ckks_decrypt<Dpt, Dct, S>(
                module: &::poulpy_hal::layouts::Module<Self>,
                pt: &mut Dpt,
                ct: &Dct,
                sk: &S,
                scratch: &mut ::poulpy_hal::layouts::ScratchArena<'_, Self>,
            ) -> $crate::CKKSResult<()>
            where
                Dpt: ::poulpy_core::layouts::GLWEToBackendMut<Self, $ring>
                    + $crate::CKKSCtBounds
                    + $crate::SetCKKSInfos
                    + ::poulpy_core::layouts::IntPolyInfos,
                Dct: ::poulpy_core::layouts::GLWEToBackendRef<Self, $ring>
                    + ::poulpy_core::layouts::GLWEInfos
                    + $crate::CKKSCtBounds,
                S: ::poulpy_core::layouts::GLWESecretPreparedToBackendRef<Self, $ring> + ::poulpy_core::layouts::GLWEInfos,
            {
                $crate::reference::encryption::CKKSEncryptionReference::<Self, $ring>::ckks_decrypt_reference(
                    module, pt, ct, sk, scratch,
                )
            }
        }
    };
}
pub use crate::impl_ckks_encryption_reference_full;
