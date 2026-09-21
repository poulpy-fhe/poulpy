#![allow(clippy::too_many_arguments)]

use crate::CKKSResult as Result;
use crate::reference::encryption::CKKSEncryptionReference;
use poulpy_core::layouts::IntPolyInfos;

use poulpy_core::{
    EncryptionInfos, GLWEDecrypt, GLWEEncryptSk,
    layouts::{GLWEInfos, GLWESecretPreparedToBackendRef},
};
use poulpy_hal::{
    api::{VecZnxLsh, VecZnxLshAdd, VecZnxLshTmpBytes, VecZnxRsh, VecZnxRshAdd, VecZnxRshTmpBytes},
    layouts::{Backend, HostBackend, HostDataMut, HostDataRef, Module, ScratchArena},
    oep::{HalSvpImpl, HalVecZnxBigImpl, HalVecZnxDftImpl, HalVecZnxImpl},
    source::Source,
};

use crate::{CKKSCtBounds, GLWEToBackendMut, GLWEToBackendRef, SetCKKSInfos, reference::plaintext::CKKSPlaintextReference};

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
        sk: &S,
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

    fn ckks_decrypt_tmp_bytes_impl<A>(module: &Module<Self>, ct_infos: &A) -> usize
    where
        A: CKKSCtBounds;

    fn ckks_decrypt_impl<S, Dct, Pt>(
        module: &Module<Self>,
        pt: &mut Pt,
        ct: &Dct,
        sk: &S,
        scratch: &mut ScratchArena<'_, Self>,
    ) -> Result<()>
    where
        Pt: GLWEToBackendMut<Self> + CKKSCtBounds + SetCKKSInfos + IntPolyInfos,
        Dct: GLWEToBackendRef<Self> + GLWEInfos + CKKSCtBounds,
        S: GLWESecretPreparedToBackendRef<Self> + GLWEInfos;
}

/// Default encryption/decryption, deliberately restricted to host backends
/// (`HostBackend` + host-visible buffer views): the [`CKKSEncryptionImpl`]
/// trait itself carries no host bounds, and a device backend implements it
/// natively instead of relying on this blanket impl.
unsafe impl<BE: Backend> CKKSEncryptionImpl for BE
where
    BE: HalVecZnxImpl + HalVecZnxBigImpl + HalVecZnxDftImpl + HalSvpImpl + HostBackend,
    Module<BE>: CKKSEncryptionReference<BE>
        + CKKSPlaintextReference<BE>
        + GLWEEncryptSk<BE>
        + GLWEDecrypt<BE>
        + poulpy_core::GLWENormalize<BE>
        + VecZnxLshAdd<BE>
        + VecZnxRshAdd<BE>
        + VecZnxRshTmpBytes
        + VecZnxLsh<BE>
        + VecZnxLshTmpBytes
        + VecZnxRsh<BE>,
    for<'a> BE::BufMut<'a>: HostDataMut,
    for<'a> BE::BufRef<'a>: HostDataRef,
{
    fn ckks_encrypt_sk_tmp_bytes_impl<A>(module: &Module<BE>, ct_infos: &A) -> usize
    where
        A: CKKSCtBounds,
    {
        module.ckks_encrypt_sk_tmp_bytes_reference(ct_infos)
    }

    fn ckks_encrypt_sk_impl<Dct, S, E, Pt>(
        module: &Module<BE>,
        ct: &mut Dct,
        pt: &Pt,
        sk: &S,
        enc_infos: &E,
        source_xe: &mut Source,
        source_xa: &mut Source,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        E: EncryptionInfos,
        Pt: GLWEToBackendRef<BE> + CKKSCtBounds + IntPolyInfos,
        Dct: GLWEToBackendMut<BE> + CKKSCtBounds + SetCKKSInfos,
        S: GLWESecretPreparedToBackendRef<BE>,
    {
        module.ckks_encrypt_sk_reference(ct, pt, sk, enc_infos, source_xe, source_xa, scratch)
    }

    fn ckks_decrypt_tmp_bytes_impl<A>(module: &Module<BE>, ct_infos: &A) -> usize
    where
        A: CKKSCtBounds,
    {
        module.ckks_decrypt_tmp_bytes_reference(ct_infos)
    }

    fn ckks_decrypt_impl<S, Dct, Pt>(
        module: &Module<BE>,
        pt: &mut Pt,
        ct: &Dct,
        sk: &S,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        Pt: GLWEToBackendMut<BE> + CKKSCtBounds + SetCKKSInfos + IntPolyInfos,
        Dct: GLWEToBackendRef<BE> + GLWEInfos + CKKSCtBounds,
        S: GLWESecretPreparedToBackendRef<BE> + GLWEInfos,
    {
        module.ckks_decrypt_reference(pt, ct, sk, scratch)
    }
}

#[macro_export]
macro_rules! impl_ckks_encryption_reference {
    ($be:ty) => {
        impl $crate::reference::encryption::CKKSEncryptionReference<$be> for ::poulpy_hal::layouts::Module<$be> {}
    };
}
pub use crate::impl_ckks_encryption_reference;
