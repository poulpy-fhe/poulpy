use anyhow::Result;
use poulpy_core::{EncryptionInfos, ScratchTakeCore, layouts::GLWEInfos};
use poulpy_hal::{
    api::ScratchAvailable,
    layouts::{Backend, DataMut, DataRef, Scratch},
};

use crate::{
    layouts::{CKKSCiphertext, plaintext::CKKSPlaintextVecZnx},
    oep::CKKSImpl,
};
use poulpy_core::layouts::GLWESecretPreparedToRef;
use poulpy_hal::source::Source;

pub trait CKKSEncrypt<BE: Backend + CKKSImpl<BE>> {
    /// Returns the scratch size, in bytes, required by [`Self::ckks_encrypt_sk`].
    ///
    /// The returned size depends on the ciphertext layout and backend.
    fn ckks_encrypt_sk_tmp_bytes<A>(&self, ct_infos: &A) -> usize
    where
        A: GLWEInfos;

    /// Encrypts a CKKS plaintext vector under a secret key.
    #[allow(clippy::too_many_arguments)]
    fn ckks_encrypt_sk<S, E: EncryptionInfos>(
        &self,
        ct: &mut CKKSCiphertext<impl DataMut>,
        pt: &CKKSPlaintextVecZnx<impl DataRef>,
        sk: &S,
        enc_infos: &E,
        source_xa: &mut Source,
        source_xe: &mut Source,
        scratch: &mut Scratch<BE>,
    ) -> Result<()>
    where
        S: GLWESecretPreparedToRef<BE>,
        Scratch<BE>: ScratchAvailable + ScratchTakeCore<BE>;
}

pub trait CKKSDecrypt<BE: Backend + CKKSImpl<BE>> {
    /// Returns the scratch size, in bytes, required by [`Self::ckks_decrypt`].
    ///
    /// The returned size includes raw GLWE decryption plus plaintext extraction.
    fn ckks_decrypt_tmp_bytes<A>(&self, ct_infos: &A) -> usize
    where
        A: GLWEInfos;

    /// Decrypts a ciphertext into a caller-provided CKKS plaintext layout.
    fn ckks_decrypt<S>(
        &self,
        pt: &mut CKKSPlaintextVecZnx<impl DataMut>,
        ct: &CKKSCiphertext<impl DataRef>,
        sk: &S,
        scratch: &mut Scratch<BE>,
    ) -> Result<()>
    where
        S: GLWESecretPreparedToRef<BE> + GLWEInfos,
        Scratch<BE>: ScratchTakeCore<BE>;
}

// Suppress unused import warnings for trait bounds used only in delegates
#[allow(unused_imports)]
use poulpy_hal::api::{VecZnxLsh, VecZnxLshTmpBytes, VecZnxRsh, VecZnxRshAddInto, VecZnxRshTmpBytes};
#[allow(unused_imports)]
use poulpy_hal::layouts::Module;
