use crate::{
    blind_rotation::BlindRotationAlgo,
    circuit_bootstrapping::{CircuitBootstrappingKey, CircuitBootstrappingKeyInfos, CircuitBootstrappingKeyPrepared},
};
use poulpy_hal::layouts::{Backend, ScratchArena};

/// Allocates and prepares backend-owned circuit-bootstrap key bundles.
pub trait CircuitBootstrappingKeyPreparedFactory<BRA: BlindRotationAlgo, BE: Backend>: Sized {
    /// Allocates backend-owned storage for a prepared key described by `infos`.
    fn circuit_bootstrapping_key_prepared_alloc_from_infos<A>(
        &self,
        infos: &A,
    ) -> CircuitBootstrappingKeyPrepared<BE::OwnedBuf, BRA, BE>
    where
        A: CircuitBootstrappingKeyInfos;
    /// Returns the scratch-space size, in bytes, required by
    /// [`Self::circuit_bootstrapping_key_prepare`].
    fn circuit_bootstrapping_key_prepare_tmp_bytes<A>(&self, infos: &A) -> usize
    where
        A: CircuitBootstrappingKeyInfos;
    /// Prepares all sub-keys of `other` into the allocated backend-owned `res`.
    ///
    /// Allocate scratch with [`Self::circuit_bootstrapping_key_prepare_tmp_bytes`].
    fn circuit_bootstrapping_key_prepare(
        &self,
        res: &mut CircuitBootstrappingKeyPrepared<BE::OwnedBuf, BRA, BE>,
        other: &CircuitBootstrappingKey<BE::OwnedBuf, BRA, BE::ZnxWord>,
        scratch: &mut ScratchArena<'_, BE>,
    );
}
