#![allow(clippy::too_many_arguments)]

use crate::CKKSResult as Result;
use poulpy_core::layouts::IntPolyInfos;

use poulpy_core::{
    EncryptionInfos,
    layouts::{GLWEInfos, GLWESecretPreparedToBackendRef},
};
use poulpy_hal::{
    layouts::{Backend, Module, ScratchArena},
    source::Source,
};

use crate::{CKKSCtBounds, CKKSInfos, GLWEToBackendMut, GLWEToBackendRef, SetCKKSInfos};

/// # Safety
///
/// Implementations must satisfy the contracts of all trait methods, including
/// any HAL-level invariants (alignment, layout, scratch sizing) implied by the
/// associated method signatures.
pub unsafe trait CKKSEncryptionImpl: Backend {
    fn ckks_encrypt_sk_tmp_bytes_impl<A>(module: &Module<Self>, ct_infos: &A) -> usize
    where
        A: CKKSCtBounds;

    fn ckks_encrypt_sk_impl<Dct, S, E, Pt>(
        module: &Module<Self>,
        ct: &mut Dct,
        pt: &Pt,
        sk: &crate::layouts::CKKSKey<S>,
        enc_infos: &E,
        source_xe: &mut Source,
        source_xa: &mut Source,
        scratch: &mut ScratchArena<'_, Self>,
    ) -> Result<()>
    where
        E: EncryptionInfos,
        Pt: GLWEToBackendRef<Self> + IntPolyInfos + CKKSCtBounds,
        Dct: GLWEToBackendMut<Self> + CKKSCtBounds + SetCKKSInfos,
        S: GLWESecretPreparedToBackendRef<Self>;

    fn ckks_decrypt_tmp_bytes_impl<Pt, Ct>(module: &Module<Self>, pt_infos: &Pt, ct_infos: &Ct) -> usize
    where
        Pt: CKKSInfos,
        Ct: CKKSCtBounds;

    fn ckks_decrypt_impl<S, Dct, Pt>(
        module: &Module<Self>,
        pt: &mut Pt,
        ct: &Dct,
        sk: &crate::layouts::CKKSKey<S>,
        scratch: &mut ScratchArena<'_, Self>,
    ) -> Result<()>
    where
        Pt: GLWEToBackendMut<Self> + CKKSCtBounds + SetCKKSInfos + IntPolyInfos,
        Dct: GLWEToBackendRef<Self> + GLWEInfos + CKKSCtBounds,
        S: GLWESecretPreparedToBackendRef<Self> + GLWEInfos;
}

/// Implements this contract with the callable reference algorithms.
#[macro_export]
macro_rules! impl_ckks_encryption_reference {
    ($be:ty) => {
        unsafe impl $crate::oep::CKKSEncryptionImpl for $be {
            fn ckks_encrypt_sk_tmp_bytes_impl<A>(module: &::poulpy_hal::layouts::Module<Self>, ct_infos: &A) -> usize
            where
                A: $crate::CKKSCtBounds,
            {
                $crate::reference::encryption::CKKSEncryptionReference::ckks_encrypt_sk_tmp_bytes_reference(module, ct_infos)
            }

            fn ckks_encrypt_sk_impl<Dct, S, E, Pt>(
                module: &::poulpy_hal::layouts::Module<Self>,
                ct: &mut Dct,
                pt: &Pt,
                sk: &$crate::layouts::CKKSKey<S>,
                enc_infos: &E,
                source_xe: &mut ::poulpy_hal::source::Source,
                source_xa: &mut ::poulpy_hal::source::Source,
                scratch: &mut ::poulpy_hal::layouts::ScratchArena<'_, Self>,
            ) -> $crate::CKKSResult<()>
            where
                E: ::poulpy_core::EncryptionInfos,
                Pt: ::poulpy_core::layouts::GLWEToBackendRef<Self> + $crate::CKKSCtBounds + ::poulpy_core::layouts::IntPolyInfos,
                Dct: ::poulpy_core::layouts::GLWEToBackendMut<Self> + $crate::CKKSCtBounds + $crate::SetCKKSInfos,
                S: ::poulpy_core::layouts::GLWESecretPreparedToBackendRef<Self>,
            {
                $crate::reference::encryption::CKKSEncryptionReference::ckks_encrypt_sk_reference(
                    module, ct, pt, sk, enc_infos, source_xe, source_xa, scratch,
                )
            }

            fn ckks_decrypt_tmp_bytes_impl<Pt, Ct>(
                module: &::poulpy_hal::layouts::Module<Self>,
                pt_infos: &Pt,
                ct_infos: &Ct,
            ) -> usize
            where
                Pt: $crate::CKKSInfos,
                Ct: $crate::CKKSCtBounds,
            {
                $crate::reference::encryption::CKKSEncryptionReference::ckks_decrypt_tmp_bytes_reference(
                    module, pt_infos, ct_infos,
                )
            }

            fn ckks_decrypt_impl<S, Dct, Pt>(
                module: &::poulpy_hal::layouts::Module<Self>,
                pt: &mut Pt,
                ct: &Dct,
                sk: &$crate::layouts::CKKSKey<S>,
                scratch: &mut ::poulpy_hal::layouts::ScratchArena<'_, Self>,
            ) -> $crate::CKKSResult<()>
            where
                Pt: ::poulpy_core::layouts::GLWEToBackendMut<Self>
                    + $crate::CKKSCtBounds
                    + $crate::SetCKKSInfos
                    + ::poulpy_core::layouts::IntPolyInfos,
                Dct: ::poulpy_core::layouts::GLWEToBackendRef<Self> + ::poulpy_core::layouts::GLWEInfos + $crate::CKKSCtBounds,
                S: ::poulpy_core::layouts::GLWESecretPreparedToBackendRef<Self> + ::poulpy_core::layouts::GLWEInfos,
            {
                $crate::reference::encryption::CKKSEncryptionReference::ckks_decrypt_reference(module, pt, ct, sk, scratch)
            }
        }
    };
}
pub use crate::impl_ckks_encryption_reference;
