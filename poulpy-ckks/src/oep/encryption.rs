#![allow(clippy::too_many_arguments)]

use crate::default::encryption::CKKSEncryptionDefault;

use anyhow::Result;
use poulpy_core::{
    EncryptionInfos, ScratchArenaTakeCore,
    layouts::{GLWEInfos, GLWESecretPreparedToBackendRef, LWEInfos},
    oep::{DecryptionDefault, EncryptionDefault},
};
use poulpy_hal::{
    api::{ScratchAvailable, VecZnxLshBackend, VecZnxLshTmpBytes, VecZnxRshAddIntoBackend, VecZnxRshBackend, VecZnxRshTmpBytes},
    layouts::{Backend, HostBackend, HostDataMut, HostDataRef, Module, ScratchArena},
    oep::{HalSvpImpl, HalVecZnxBigImpl, HalVecZnxDftImpl, HalVecZnxImpl},
    source::Source,
};

use crate::{CKKSInfos, GLWEToBackendMut, GLWEToBackendRef, SetCKKSInfos, default::plaintext::CKKSPlaintextDefault};

/// # Safety
///
/// Implementations must satisfy the contracts of all trait methods, including
/// any HAL-level invariants (alignment, layout, scratch sizing) implied by the
/// associated method signatures.
pub unsafe trait CKKSEncryptionImpl<BE: Backend>: Backend {
    fn ckks_encrypt_sk_tmp_bytes<A>(module: &Module<BE>, ct_infos: &A) -> usize
    where
        A: GLWEInfos + CKKSInfos;

    fn ckks_encrypt_sk<'s, Dct, S, E, Pt>(
        module: &Module<BE>,
        ct: &mut Dct,
        pt: &Pt,
        sk: &S,
        enc_infos: &E,
        source_xa: &mut Source,
        source_xe: &mut Source,
        scratch: &mut ScratchArena<'s, BE>,
    ) -> Result<()>
    where
        E: EncryptionInfos,
        Pt: GLWEToBackendRef<BE> + LWEInfos + CKKSInfos,
        Dct: GLWEToBackendMut<BE> + LWEInfos + CKKSInfos + SetCKKSInfos,
        S: GLWESecretPreparedToBackendRef<BE>,
        BE: 's;

    fn ckks_decrypt_tmp_bytes<A>(module: &Module<BE>, ct_infos: &A) -> usize
    where
        A: GLWEInfos + CKKSInfos;

    fn ckks_decrypt<S, Dct, Pt>(
        module: &Module<BE>,
        pt: &mut Pt,
        ct: &Dct,
        sk: &S,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        Pt: GLWEToBackendMut<BE> + LWEInfos + CKKSInfos + SetCKKSInfos,
        Dct: GLWEToBackendRef<BE> + GLWEInfos + LWEInfos + CKKSInfos,
        S: GLWESecretPreparedToBackendRef<BE> + GLWEInfos;
}

unsafe impl<BE: Backend> CKKSEncryptionImpl<BE> for BE
where
    BE: HalVecZnxImpl<BE> + HalVecZnxBigImpl<BE> + HalVecZnxDftImpl<BE> + HalSvpImpl<BE> + HostBackend,
    Module<BE>: CKKSEncryptionDefault<BE>
        + CKKSPlaintextDefault<BE>
        + EncryptionDefault<BE>
        + DecryptionDefault<BE>
        + VecZnxRshAddIntoBackend<BE>
        + VecZnxRshTmpBytes
        + VecZnxLshBackend<BE>
        + VecZnxLshTmpBytes
        + VecZnxRshBackend<BE>,
    for<'a> BE::BufMut<'a>: HostDataMut,
    for<'a> BE::BufRef<'a>: HostDataRef,
    for<'a> ScratchArena<'a, BE>: ScratchArenaTakeCore<'a, BE> + ScratchAvailable,
{
    fn ckks_encrypt_sk_tmp_bytes<A>(module: &Module<BE>, ct_infos: &A) -> usize
    where
        A: GLWEInfos + CKKSInfos,
    {
        module.ckks_encrypt_sk_tmp_bytes_default(ct_infos)
    }

    fn ckks_encrypt_sk<'s, Dct, S, E, Pt>(
        module: &Module<BE>,
        ct: &mut Dct,
        pt: &Pt,
        sk: &S,
        enc_infos: &E,
        source_xa: &mut Source,
        source_xe: &mut Source,
        scratch: &mut ScratchArena<'s, BE>,
    ) -> Result<()>
    where
        E: EncryptionInfos,
        Pt: GLWEToBackendRef<BE> + LWEInfos + CKKSInfos,
        Dct: GLWEToBackendMut<BE> + LWEInfos + CKKSInfos + SetCKKSInfos,
        S: GLWESecretPreparedToBackendRef<BE>,
        BE: 's,
    {
        module.ckks_encrypt_sk_default(ct, pt, sk, enc_infos, source_xa, source_xe, scratch)
    }

    fn ckks_decrypt_tmp_bytes<A>(module: &Module<BE>, ct_infos: &A) -> usize
    where
        A: GLWEInfos + CKKSInfos,
    {
        module.ckks_decrypt_tmp_bytes_default(ct_infos)
    }

    fn ckks_decrypt<S, Dct, Pt>(
        module: &Module<BE>,
        pt: &mut Pt,
        ct: &Dct,
        sk: &S,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        Pt: GLWEToBackendMut<BE> + LWEInfos + CKKSInfos + SetCKKSInfos,
        Dct: GLWEToBackendRef<BE> + GLWEInfos + LWEInfos + CKKSInfos,
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
