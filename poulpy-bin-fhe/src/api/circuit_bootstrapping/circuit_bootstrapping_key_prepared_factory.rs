use crate::{
    api::BlindRotationKeyPreparedFactory,
    blind_rotation::BlindRotationAlgo,
    circuit_bootstrapping::{
        CircuitBootstrappingKey, CircuitBootstrappingKeyInfos, CircuitBootstrappingKeyPrepared, trace_galois_elements,
    },
};
use poulpy_core::{
    api::{GGLWEToGGSWKeyPreparedFactory, GGSWPreparedFactory, GLWEAutomorphismKeyPreparedFactory},
    layouts::{LWEInfos, SetGaloisElement},
};
use poulpy_hal::layouts::{Backend, ScratchArena};

/// Allocates and prepares backend-owned circuit-bootstrap key bundles.
///
/// Allocation and sizing are direct on the module; the preparation and its
/// query dispatch through
/// [`CircuitBootstrappingKeyPreparedImpl`](crate::oep::CircuitBootstrappingKeyPreparedImpl).
pub trait CircuitBootstrappingKeyPreparedFactory<BRA: BlindRotationAlgo, BE: Backend>: Sized {
    /// Allocates a zero-filled prepared key bundle of `infos`: the prepared
    /// blind-rotation key, the prepared tensor-switching key and one prepared
    /// automorphism key per trace Galois element, which
    /// [`Self::circuit_bootstrapping_key_prepare`] fills.
    ///
    /// # Panics
    ///
    /// When the degree of a component of `infos` differs from the module degree.
    fn circuit_bootstrapping_key_prepared_alloc_from_infos<A>(
        &self,
        infos: &A,
    ) -> CircuitBootstrappingKeyPrepared<BE::OwnedBuf, BRA, BE>
    where
        A: CircuitBootstrappingKeyInfos,
        Self: BlindRotationKeyPreparedFactory<BRA, BE>
            + GGSWPreparedFactory<BE>
            + GGLWEToGGSWKeyPreparedFactory<BE>
            + GLWEAutomorphismKeyPreparedFactory<BE>,
    {
        let atk_infos = infos.atk_infos();
        let tsk_infos = infos.tsk_infos();
        assert!(
            atk_infos.n() == self.ring_degree() && tsk_infos.n() == self.ring_degree(),
            "circuit_bootstrapping_key_prepared_alloc_from_infos: key degree differs from the module degree"
        );
        CircuitBootstrappingKeyPrepared {
            brk: self.blind_rotation_key_prepared_alloc_from_infos(&infos.brk_infos()),
            tsk: self.gglwe_to_ggsw_key_prepared_alloc_from_infos(&tsk_infos),
            atk: trace_galois_elements(atk_infos.log_n(), 2 * atk_infos.n().as_usize() as i64)
                .into_iter()
                .map(|p| {
                    let mut key = self.glwe_automorphism_key_prepared_alloc_from_infos(&atk_infos);
                    key.set_p(p);
                    (p, key)
                })
                .collect(),
        }
    }

    /// Bytes of the prepared components
    /// [`Self::circuit_bootstrapping_key_prepared_alloc_from_infos`] allocates
    /// for `infos`, counted as
    /// [`BlindRotationKeyPreparedFactory::blind_rotation_key_prepared_bytes_of_from_infos`]
    /// counts the blind-rotation key.
    ///
    /// # Panics
    ///
    /// As [`Self::circuit_bootstrapping_key_prepared_alloc_from_infos`], and
    /// when the byte count overflows `usize`.
    fn circuit_bootstrapping_key_prepared_bytes_of_from_infos<A>(&self, infos: &A) -> usize
    where
        A: CircuitBootstrappingKeyInfos,
        Self: BlindRotationKeyPreparedFactory<BRA, BE>
            + GGSWPreparedFactory<BE>
            + GGLWEToGGSWKeyPreparedFactory<BE>
            + GLWEAutomorphismKeyPreparedFactory<BE>,
    {
        let atk_infos = infos.atk_infos();
        let tsk_infos = infos.tsk_infos();
        assert!(
            atk_infos.n() == self.ring_degree() && tsk_infos.n() == self.ring_degree(),
            "circuit_bootstrapping_key_prepared_bytes_of_from_infos: key degree differs from the module degree"
        );
        // The trace takes one automorphism key per level, log_n in all.
        self.glwe_automorphism_key_prepared_bytes_of_from_infos(&atk_infos)
            .checked_mul(atk_infos.log_n())
            .and_then(|atk| atk.checked_add(self.bytes_of_gglwe_to_ggsw_from_infos(&tsk_infos)))
            .and_then(|bytes| bytes.checked_add(self.blind_rotation_key_prepared_bytes_of_from_infos(&infos.brk_infos())))
            .expect("circuit_bootstrapping_key_prepared_bytes_of_from_infos: the byte count overflows usize")
    }

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
