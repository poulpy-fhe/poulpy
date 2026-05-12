use poulpy_core::ScratchArenaTakeCore;
use poulpy_core::{
    EncryptionInfos, GetDistribution,
    layouts::{GGSWInfos, GLWEInfos, GLWESecretPreparedToBackendRef, LWEInfos, LWESecretToBackendRef},
};
use poulpy_hal::{
    layouts::{Backend, ScratchArena},
    source::Source,
};

use crate::blind_rotation::{BlindRotationAlgo, BlindRotationKeyCompressed};

/// Backend-level key-encryption trait for [`BlindRotationKeyCompressed`].
///
/// Equivalent to `BlindRotationKeyEncryptSk` but produces the seed-compressed
/// form of the key.  Instead of a mutable `source_xa`, callers supply a fixed
/// 32-byte `seed_xa`; all per-GGSW mask seeds are derived deterministically
/// from this root seed via `Source::new_seed`.
///
/// # Panics
///
/// Panics if the LWE secret distribution is not a supported binary type.
pub trait BlindRotationKeyCompressedEncryptSk<B: Backend, BRA: BlindRotationAlgo> {
    /// Returns the minimum scratch-space size in bytes required by
    /// [`blind_rotation_key_compressed_encrypt_sk`][Self::blind_rotation_key_compressed_encrypt_sk].
    fn blind_rotation_key_compressed_encrypt_sk_tmp_bytes<A>(&self, infos: &A) -> usize
    where
        A: GGSWInfos;

    /// Encrypts each bit of `sk_lwe` as a compressed GGSW ciphertext under
    /// `sk_glwe`, storing the result in `res`.
    ///
    /// `seed_xa` is the 32-byte root seed from which per-element mask seeds
    /// are derived.  `source_xe` provides randomness for the error components.
    #[allow(clippy::too_many_arguments)]
    fn blind_rotation_key_compressed_encrypt_sk<'s, S0, S1, E>(
        &self,
        res: &mut BlindRotationKeyCompressed<B::OwnedBuf, BRA>,
        sk_glwe: &S0,
        sk_lwe: &S1,
        seed_xa: [u8; 32],
        enc_infos: &E,
        source_xe: &mut Source,
        scratch: &mut ScratchArena<'s, B>,
    ) where
        S0: GLWESecretPreparedToBackendRef<B> + GLWEInfos,
        E: EncryptionInfos,
        S1: LWESecretToBackendRef<B> + LWEInfos + GetDistribution,
        B: 's,
        ScratchArena<'s, B>: ScratchArenaTakeCore<'s, B>;
}
