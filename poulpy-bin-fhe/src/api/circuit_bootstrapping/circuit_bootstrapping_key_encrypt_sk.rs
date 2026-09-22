use crate::{
    blind_rotation::BlindRotationAlgo,
    circuit_bootstrapping::{CircuitBootstrappingEncryptionInfos, CircuitBootstrappingKey, CircuitBootstrappingKeyInfos},
};
use poulpy_core::{
    GetDistribution,
    layouts::{GLWEInfos, GLWESecretToBackendRef, LWEInfos, LWESecretToBackendRef},
};
use poulpy_hal::{
    layouts::{Backend, ScratchArena},
    source::Source,
};

/// Encrypts a complete circuit-bootstrap key using the selected backend.
pub trait CircuitBootstrappingKeyEncryptSk<BRA, BE>
where
    BRA: BlindRotationAlgo,
    BE: Backend,
{
    /// Returns the minimum scratch-space size (in bytes) required by
    /// [`circuit_bootstrapping_key_encrypt_sk`][Self::circuit_bootstrapping_key_encrypt_sk].
    fn circuit_bootstrapping_key_encrypt_sk_tmp_bytes<A>(&self, infos: &A) -> usize
    where
        A: CircuitBootstrappingKeyInfos;

    /// Encrypts all sub-keys of a circuit bootstrapping key bundle.
    ///
    /// The three sub-key components are encrypted in order: ATK, BRK, TSK.
    /// Scratch is reused across sub-key encryptions. The BRK phase also retains
    /// a prepared GLWE secret, included in its advertised peak requirement.
    #[allow(clippy::too_many_arguments)]
    fn circuit_bootstrapping_key_encrypt_sk<S0, S1>(
        &self,
        res: &mut CircuitBootstrappingKey<BE::OwnedBuf, BRA, BE::ZnxWord>,
        sk_lwe: &S0,
        sk_glwe: &S1,
        enc_infos: &CircuitBootstrappingEncryptionInfos,
        source_xe: &mut Source,
        source_xa: &mut Source,
        scratch: &mut ScratchArena<'_, BE>,
    ) where
        S0: LWESecretToBackendRef<BE> + GetDistribution + LWEInfos,
        S1: GLWESecretToBackendRef<BE> + GLWEInfos + GetDistribution;
}
