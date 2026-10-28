use crate::{
    api::{BlindRotationKeyPreparedFactory, CircuitBootstrappingKeyPreparedFactory},
    bdd_arithmetic::*,
    blind_rotation::BlindRotationAlgo,
};
use poulpy_core::{
    api::{
        GGLWEToGGSWKeyPreparedFactory, GGSWPreparedFactory, GLWEAutomorphismKeyPreparedFactory, GLWESwitchingKeyPreparedFactory,
        GLWEToLWEKeyPreparedFactory,
    },
    layouts::LWEInfos,
};
use poulpy_hal::layouts::*;
/// Backend-level factory for allocating and preparing [`BDDKeyPrepared`] values.
///
/// Allocation and sizing are direct on the module; the preparation and its
/// query dispatch through [`BDDKeyPreparedFactoryImpl`](crate::oep::BDDKeyPreparedFactoryImpl).
pub trait BDDKeyPreparedFactory<BRA: BlindRotationAlgo, BE: Backend> {
    /// Allocates a zero-filled prepared key bundle of `infos`: the prepared
    /// circuit-bootstrapping key, the optional prepared GLWE switching key and
    /// the prepared GLWE-to-LWE key, which [`Self::bdd_key_prepare`] fills.
    ///
    /// # Panics
    ///
    /// When the degree of a component of `infos` differs from the module degree.
    fn bdd_key_prepared_alloc_from_infos<A>(&self, infos: &A) -> BDDKeyPrepared<BE::OwnedBuf, BRA, BE>
    where
        A: BDDKeyInfos,
        Self: CircuitBootstrappingKeyPreparedFactory<BRA, BE>
            + BlindRotationKeyPreparedFactory<BRA, BE>
            + GGSWPreparedFactory<BE>
            + GGLWEToGGSWKeyPreparedFactory<BE>
            + GLWEAutomorphismKeyPreparedFactory<BE>
            + GLWESwitchingKeyPreparedFactory<BE>
            + GLWEToLWEKeyPreparedFactory<BE>,
    {
        let ks_glwe_infos = infos.ks_glwe_infos();
        let ks_lwe_infos = infos.ks_lwe_infos();
        assert!(
            ks_lwe_infos.n() == self.ring_degree() && ks_glwe_infos.is_none_or(|ks_glwe| ks_glwe.n() == self.ring_degree()),
            "bdd_key_prepared_alloc_from_infos: key degree differs from the module degree"
        );
        BDDKeyPrepared {
            cbt: self.circuit_bootstrapping_key_prepared_alloc_from_infos(&infos.cbt_infos()),
            ks_glwe: ks_glwe_infos.map(|ks_glwe| self.glwe_switching_key_prepared_alloc_from_infos(&ks_glwe)),
            ks_lwe: self.glwe_to_lwe_key_prepared_alloc_from_infos(&ks_lwe_infos),
        }
    }

    /// Bytes of the prepared components [`Self::bdd_key_prepared_alloc_from_infos`]
    /// allocates for `infos`, counted as
    /// [`CircuitBootstrappingKeyPreparedFactory::circuit_bootstrapping_key_prepared_bytes_of_from_infos`]
    /// counts the circuit-bootstrapping key.
    ///
    /// # Panics
    ///
    /// As [`Self::bdd_key_prepared_alloc_from_infos`], and when the byte count
    /// overflows `usize`.
    fn bdd_key_prepared_bytes_of_from_infos<A>(&self, infos: &A) -> usize
    where
        A: BDDKeyInfos,
        Self: CircuitBootstrappingKeyPreparedFactory<BRA, BE>
            + BlindRotationKeyPreparedFactory<BRA, BE>
            + GGSWPreparedFactory<BE>
            + GGLWEToGGSWKeyPreparedFactory<BE>
            + GLWEAutomorphismKeyPreparedFactory<BE>
            + GLWESwitchingKeyPreparedFactory<BE>
            + GLWEToLWEKeyPreparedFactory<BE>,
    {
        let ks_glwe_infos = infos.ks_glwe_infos();
        let ks_lwe_infos = infos.ks_lwe_infos();
        assert!(
            ks_lwe_infos.n() == self.ring_degree() && ks_glwe_infos.is_none_or(|ks_glwe| ks_glwe.n() == self.ring_degree()),
            "bdd_key_prepared_bytes_of_from_infos: key degree differs from the module degree"
        );
        self.circuit_bootstrapping_key_prepared_bytes_of_from_infos(&infos.cbt_infos())
            .checked_add(self.glwe_to_lwe_key_prepared_bytes_of_from_infos(&ks_lwe_infos))
            .and_then(|bytes| {
                bytes.checked_add(
                    ks_glwe_infos.map_or(0, |ks_glwe| self.glwe_switching_key_prepared_bytes_of_from_infos(&ks_glwe)),
                )
            })
            .expect("bdd_key_prepared_bytes_of_from_infos: the byte count overflows usize")
    }

    /// Returns the minimum scratch-space size in bytes required by
    /// [`Self::bdd_key_prepare`].
    fn bdd_key_prepare_tmp_bytes<A>(&self, infos: &A) -> usize
    where
        A: BDDKeyInfos;

    /// Prepares every component of `other` into the allocated `res`.
    ///
    /// Allocate scratch with [`Self::bdd_key_prepare_tmp_bytes`].
    fn bdd_key_prepare(
        &self,
        res: &mut BDDKeyPrepared<BE::OwnedBuf, BRA, BE>,
        other: &BDDKey<BE::OwnedBuf, BRA, BE::ZnxWord>,
        scratch: &mut ScratchArena<'_, BE>,
    );
}
