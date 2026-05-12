use itertools::Itertools;
use poulpy_core::layouts::{
    GGLWEInfos, GGLWEToGGSWKeyLayout, GGLWEToGGSWKeyPrepared, GGLWEToGGSWKeyPreparedFactory, GGSWInfos,
    GLWEAutomorphismKeyHelper, GLWEAutomorphismKeyLayout, GLWEAutomorphismKeyPreparedFactory, GLWEInfos,
    GLWETensorKeyPreparedFactory, LWEInfos, prepared::GLWEAutomorphismKeyPrepared,
};
use std::collections::HashMap;

use poulpy_hal::{
    api::ScratchAvailable,
    layouts::{Backend, Data, Module, ScratchArena},
};

use crate::circuit_bootstrapping::trace_galois_elements;
use crate::{
    blind_rotation::{
        BlindRotationAlgo, BlindRotationKeyInfos, BlindRotationKeyLayout, BlindRotationKeyPrepared,
        BlindRotationKeyPreparedFactory,
    },
    circuit_bootstrapping::{CircuitBootstrappingKey, CircuitBootstrappingKeyInfos},
};

impl<BRA: BlindRotationAlgo, BE: Backend<OwnedBuf = Vec<u8>>> CircuitBootstrappingKeyPrepared<BE::OwnedBuf, BRA, BE> {
    pub fn alloc_from_infos<A, M>(module: &M, infos: &A) -> CircuitBootstrappingKeyPrepared<BE::OwnedBuf, BRA, BE>
    where
        A: CircuitBootstrappingKeyInfos,
        M: CircuitBootstrappingKeyPreparedFactory<BRA, BE>,
    {
        module.circuit_bootstrapping_key_prepared_alloc_from_infos(infos)
    }
}

impl<BRA: BlindRotationAlgo, BE: Backend<OwnedBuf = Vec<u8>>> CircuitBootstrappingKeyPrepared<BE::OwnedBuf, BRA, BE> {
    pub fn prepare<'s, M>(
        &mut self,
        module: &M,
        other: &CircuitBootstrappingKey<BE::OwnedBuf, BRA>,
        scratch: &mut ScratchArena<'s, BE>,
    ) where
        M: CircuitBootstrappingKeyPreparedFactory<BRA, BE>,
        for<'a> ScratchArena<'a, BE>: ScratchAvailable,
        BE: 's,
    {
        module.circuit_bootstrapping_key_prepare(self, other, scratch);
    }
}

impl<BE: Backend<OwnedBuf = Vec<u8>>, BRA: BlindRotationAlgo> CircuitBootstrappingKeyPreparedFactory<BRA, BE> for Module<BE>
where
    Self: Sized
        + BlindRotationKeyPreparedFactory<BRA, BE>
        + GLWETensorKeyPreparedFactory<BE>
        + GLWEAutomorphismKeyPreparedFactory<BE>,
    BE::OwnedBuf: AsMut<[u8]> + AsRef<[u8]>,
{
}

/// Backend-level factory for allocating and preparing
/// [`CircuitBootstrappingKeyPrepared`] values.
///
/// Implemented for `Module<BE>` when the backend supports preparation of all
/// three sub-key types.  Default method implementations delegate to the
/// corresponding sub-key factories.
pub trait CircuitBootstrappingKeyPreparedFactory<BRA: BlindRotationAlgo, BE: Backend<OwnedBuf = Vec<u8>>>
where
    Self: Sized
        + BlindRotationKeyPreparedFactory<BRA, BE>
        + GGLWEToGGSWKeyPreparedFactory<BE>
        + GLWEAutomorphismKeyPreparedFactory<BE>,
    BE::OwnedBuf: AsMut<[u8]> + AsRef<[u8]>,
{
    /// Allocates a zero-filled prepared key bundle from a dimension descriptor.
    fn circuit_bootstrapping_key_prepared_alloc_from_infos<A>(
        &self,
        infos: &A,
    ) -> CircuitBootstrappingKeyPrepared<BE::OwnedBuf, BRA, BE>
    where
        A: CircuitBootstrappingKeyInfos,
    {
        let atk_infos: &GLWEAutomorphismKeyLayout = &infos.atk_infos();
        let gal_els: Vec<i64> = trace_galois_elements(atk_infos.log_n(), 2 * atk_infos.n().as_usize() as i64);

        CircuitBootstrappingKeyPrepared {
            brk: BlindRotationKeyPrepared::alloc(self, &infos.brk_infos()),
            tsk: self.gglwe_to_ggsw_key_prepared_alloc_from_infos(&infos.tsk_infos()),
            atk: gal_els
                .iter()
                .map(|&gal_el| {
                    let key = self.glwe_automorphism_key_prepared_alloc_from_infos(atk_infos);
                    (gal_el, key)
                })
                .collect(),
        }
    }

    fn circuit_bootstrapping_key_prepare_tmp_bytes<A>(&self, infos: &A) -> usize
    where
        A: CircuitBootstrappingKeyInfos,
    {
        self.blind_rotation_key_prepare_tmp_bytes(&infos.brk_infos())
            .max(self.gglwe_to_ggsw_key_prepare_tmp_bytes(&infos.tsk_infos()))
            .max(self.glwe_automorphism_key_prepare_tmp_bytes(&infos.atk_infos()))
    }

    fn circuit_bootstrapping_key_prepare<'s>(
        &self,
        res: &mut CircuitBootstrappingKeyPrepared<BE::OwnedBuf, BRA, BE>,
        other: &CircuitBootstrappingKey<BE::OwnedBuf, BRA>,
        scratch: &mut ScratchArena<'s, BE>,
    ) where
        for<'a> ScratchArena<'a, BE>: ScratchAvailable,
        BE: 's,
    {
        // TODO(device): the prepared CBT bundle is still assembled from the
        // host-backed prepared blind-rotation / automorphism / tensor-switching
        // key factories. Keep the public factory generic, but leave the
        // current implementation owned-buffer-only until those sub-factories
        // grow true backend-generic prepared outputs.
        res.brk.prepare(self, &other.brk, scratch);
        self.gglwe_to_ggsw_key_prepare(&mut res.tsk, &other.tsk, scratch);

        let gal_els: Vec<i64> = res.atk.keys().sorted().copied().collect();
        for k in gal_els {
            self.glwe_automorphism_key_prepare(
                res.atk.get_mut(&k).unwrap(),
                other.atk.get(&k).unwrap_or_else(|| {
                    panic!("Galois element {k} is present in the prepared key but missing from the source key")
                }),
                scratch,
            );
        }
    }
}

/// DFT-prepared circuit bootstrapping key bundle, ready for on-line evaluation.
///
/// Contains the three sub-keys in their frequency-domain representations:
///
/// - `brk`: DFT-prepared blind rotation key.
/// - `tsk`: DFT-prepared GGLWE-to-GGSW tensor-switching key.
/// - `atk`: Map of DFT-prepared automorphism keys, keyed by Galois element.
///
/// ## Invariants
///
/// - All sub-keys share the same polynomial degree `n`.
/// - The `atk` map contains exactly the Galois elements generated by
///   `trace_galois_elements` during allocation; querying an absent element
///   panics in [`CircuitBootstrappingKeyPreparedFactory::circuit_bootstrapping_key_prepare`].
///
/// ## Thread Safety
///
/// Shared references (`&CircuitBootstrappingKeyPrepared`) are `Sync` and can
/// be passed to multiple threads simultaneously; evaluation threads require
/// separate scratch arenas.
pub struct CircuitBootstrappingKeyPrepared<D: Data, BRA: BlindRotationAlgo, B: Backend> {
    pub(crate) brk: BlindRotationKeyPrepared<D, BRA, B>,
    pub(crate) tsk: GGLWEToGGSWKeyPrepared<D, B>,
    pub(crate) atk: HashMap<i64, GLWEAutomorphismKeyPrepared<D, B>>,
}

impl<BRA: BlindRotationAlgo, BE: Backend> GLWEAutomorphismKeyHelper<GLWEAutomorphismKeyPrepared<BE::OwnedBuf, BE>, BE>
    for CircuitBootstrappingKeyPrepared<BE::OwnedBuf, BRA, BE>
{
    fn get_automorphism_key(&self, k: i64) -> Option<&GLWEAutomorphismKeyPrepared<BE::OwnedBuf, BE>> {
        self.atk.get_automorphism_key(k)
    }

    fn automorphism_key_infos(&self) -> poulpy_core::layouts::GGLWELayout {
        self.atk.automorphism_key_infos()
    }
}

impl<D: Data, BRA: BlindRotationAlgo, B: Backend> CircuitBootstrappingKeyInfos for CircuitBootstrappingKeyPrepared<D, BRA, B> {
    fn block_size(&self) -> usize {
        self.brk.block_size()
    }

    fn atk_infos(&self) -> GLWEAutomorphismKeyLayout {
        let first_key = self.atk.keys().min().copied().expect("atk is empty");
        let atk = self.atk.get(&first_key).unwrap();
        GLWEAutomorphismKeyLayout {
            n: atk.n(),
            base2k: atk.base2k(),
            k: atk.max_k(),
            dnum: atk.dnum(),
            dsize: atk.dsize(),
            rank: atk.rank(),
        }
    }

    fn brk_infos(&self) -> BlindRotationKeyLayout {
        BlindRotationKeyLayout {
            n_glwe: self.brk.n_glwe(),
            n_lwe: self.brk.n_lwe(),
            base2k: self.brk.base2k(),
            k: self.brk.max_k(),
            dnum: self.brk.dnum(),
            rank: self.brk.rank(),
        }
    }

    fn tsk_infos(&self) -> GGLWEToGGSWKeyLayout {
        GGLWEToGGSWKeyLayout {
            n: self.tsk.n(),
            base2k: self.tsk.base2k(),
            k: self.tsk.max_k(),
            dnum: self.tsk.dnum(),
            dsize: self.tsk.dsize(),
            rank: self.tsk.rank(),
        }
    }
}
