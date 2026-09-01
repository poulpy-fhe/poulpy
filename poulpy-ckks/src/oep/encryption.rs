#![allow(clippy::too_many_arguments)]

use crate::CKKSResult as Result;
use crate::default::encryption::CKKSEncryptionDefault;
use poulpy_core::layouts::IntPolyInfos;

use poulpy_core::{
    EncryptionInfos,
    layouts::{GLWEInfos, GLWESecretPreparedToBackendRef},
    oep::{DecryptionDefault, EncryptionDefault},
};
use poulpy_hal::{
    api::{
        VecZnxLshAddIntoBackend, VecZnxLshBackend, VecZnxLshTmpBytes, VecZnxRshAddIntoBackend, VecZnxRshBackend,
        VecZnxRshTmpBytes,
    },
    layouts::{Backend, HostBackend, HostDataMut, HostDataRef, Module, Normalized, ScratchArena},
    oep::{HalSvpImpl, HalVecZnxBigImpl, HalVecZnxDftImpl, HalVecZnxImpl},
    source::Source,
};

use crate::{CKKSCtBounds, GLWEToBackendMut, GLWEToBackendRef, SetCKKSInfos, default::plaintext::CKKSPlaintextDefault};

/// # Safety
///
/// Implementations must satisfy the contracts of all trait methods, including
/// any HAL-level invariants (alignment, layout, scratch sizing) implied by the
/// associated method signatures.
pub unsafe trait CKKSEncryptionImpl<BE: Backend>: Backend {
    fn ckks_encrypt_sk_tmp_bytes_impl<A>(module: &Module<BE>, ct_infos: &A) -> usize
    where
        A: CKKSCtBounds;

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
        Pt: GLWEToBackendRef<BE, State = Normalized> + IntPolyInfos + CKKSCtBounds,
        Dct: GLWEToBackendMut<BE, State = Normalized> + CKKSCtBounds + SetCKKSInfos,
        S: GLWESecretPreparedToBackendRef<BE>;

    fn ckks_decrypt_tmp_bytes_impl<A>(module: &Module<BE>, ct_infos: &A) -> usize
    where
        A: CKKSCtBounds;

    fn ckks_decrypt_impl<S, Dct, Pt>(
        module: &Module<BE>,
        pt: &mut Pt,
        ct: &Dct,
        sk: &S,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        Pt: GLWEToBackendMut<BE, State = Normalized> + CKKSCtBounds + SetCKKSInfos + IntPolyInfos,
        Dct: GLWEToBackendRef<BE, State = Normalized> + GLWEInfos + CKKSCtBounds,
        S: GLWESecretPreparedToBackendRef<BE> + GLWEInfos;
}

/// Default encryption/decryption, deliberately restricted to host backends
/// (`HostBackend` + host-visible buffer views): the [`CKKSEncryptionImpl`]
/// trait itself carries no host bounds, and a device backend implements it
/// natively instead of relying on this blanket impl.
unsafe impl<BE: Backend> CKKSEncryptionImpl<BE> for BE
where
    BE: HalVecZnxImpl<BE> + HalVecZnxBigImpl<BE> + HalVecZnxDftImpl<BE> + HalSvpImpl<BE> + HostBackend,
    Module<BE>: CKKSEncryptionDefault<BE>
        + CKKSPlaintextDefault<BE>
        + EncryptionDefault<BE>
        + DecryptionDefault<BE>
        + poulpy_core::GLWENormalize<BE>
        + VecZnxLshAddIntoBackend<BE>
        + VecZnxRshAddIntoBackend<BE>
        + VecZnxRshTmpBytes
        + VecZnxLshBackend<BE>
        + VecZnxLshTmpBytes
        + VecZnxRshBackend<BE>,
    for<'a> BE::BufMut<'a>: HostDataMut,
    for<'a> BE::BufRef<'a>: HostDataRef,
{
    fn ckks_encrypt_sk_tmp_bytes_impl<A>(module: &Module<BE>, ct_infos: &A) -> usize
    where
        A: CKKSCtBounds,
    {
        module.ckks_encrypt_sk_tmp_bytes_default(ct_infos)
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
        Pt: GLWEToBackendRef<BE, State = Normalized> + CKKSCtBounds + IntPolyInfos,
        Dct: GLWEToBackendMut<BE, State = Normalized> + CKKSCtBounds + SetCKKSInfos,
        S: GLWESecretPreparedToBackendRef<BE>,
    {
        module.ckks_encrypt_sk_default(ct, pt, sk, enc_infos, source_xe, source_xa, scratch)
    }

    fn ckks_decrypt_tmp_bytes_impl<A>(module: &Module<BE>, ct_infos: &A) -> usize
    where
        A: CKKSCtBounds,
    {
        module.ckks_decrypt_tmp_bytes_default(ct_infos)
    }

    fn ckks_decrypt_impl<S, Dct, Pt>(
        module: &Module<BE>,
        pt: &mut Pt,
        ct: &Dct,
        sk: &S,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        Pt: GLWEToBackendMut<BE, State = Normalized> + CKKSCtBounds + SetCKKSInfos + IntPolyInfos,
        Dct: GLWEToBackendRef<BE, State = Normalized> + GLWEInfos + CKKSCtBounds,
        S: GLWESecretPreparedToBackendRef<BE> + GLWEInfos,
    {
        module.ckks_decrypt_default(pt, ct, sk, scratch)
    }
}

#[macro_export]
macro_rules! impl_ckks_encryption_defaults {
    ($be:ty) => {
        impl $crate::default::encryption::CKKSEncryptionDefault<$be> for ::poulpy_hal::layouts::Module<$be> {}
    };
}
pub use crate::impl_ckks_encryption_defaults;
