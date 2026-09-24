use poulpy_hal::layouts::{Backend, Data, ScratchArena, SvpPPolOwned};

use std::marker::PhantomData;

use poulpy_core::{
    Distribution,
    layouts::{Base2K, Degree, Dnum, Dsize, GGSWInfos, GLWEInfos, LWEInfos, Rank, prepared::GGSWPrepared},
};

use crate::blind_rotation::{BlindRotationAlgo, BlindRotationKey, BlindRotationKeyInfos};

/// Backend-level factory for allocating and preparing
/// [`BlindRotationKeyPrepared`] values.
///
/// Dispatched through [`crate::oep::BlindRotationKeyPreparedImpl`].  Callers should use the convenience
/// methods on [`BlindRotationKeyPrepared`] rather than calling these directly.
pub use crate::api::BlindRotationKeyPreparedFactory;

impl<BE: Backend, BRA: BlindRotationAlgo> BlindRotationKeyPrepared<BE::OwnedBuf, BRA, BE> {
    pub fn alloc<A, M>(module: &M, infos: &A) -> Self
    where
        A: BlindRotationKeyInfos,
        M: BlindRotationKeyPreparedFactory<BRA, BE>,
    {
        module.blind_rotation_key_prepared_alloc(infos)
    }

    pub fn prepare_tmp_bytes<M, A>(module: &M, infos: &A) -> usize
    where
        A: BlindRotationKeyInfos,
        M: BlindRotationKeyPreparedFactory<BRA, BE>,
    {
        module.blind_rotation_key_prepare_tmp_bytes(infos)
    }
}

impl<BRA: BlindRotationAlgo, BE: Backend> BlindRotationKeyPrepared<BE::OwnedBuf, BRA, BE> {
    /// Populates `self` from the standard key `other`.
    ///
    /// Convenience wrapper around
    /// [`BlindRotationKeyPreparedFactory::prepare_blind_rotation_key`].
    pub fn prepare<M>(
        &mut self,
        module: &M,
        other: &BlindRotationKey<BE::OwnedBuf, BRA, BE::ZnxWord>,
        scratch: &mut ScratchArena<'_, BE>,
    ) where
        M: BlindRotationKeyPreparedFactory<BRA, BE>,
    {
        module.prepare_blind_rotation_key(self, other, scratch);
    }
}

/// DFT-domain prepared blind rotation key, ready for fast on-line evaluation.
///
/// Each GGSW element is stored in the DFT (frequency) domain so that
/// matrix-vector products during blind rotation require no additional
/// forward transforms.  For the `BinaryBlock` distribution the optional
/// `x_pow_a` table pre-computes `X^i` scalar polynomials (also in DFT
/// domain) for `i` in `[0, 2n)`, avoiding re-computation during each
/// iteration of the CMux loop.
///
/// ## Invariants
///
/// - `data.len() == n_lwe`.
/// - `dist` is the LWE secret distribution used during encryption.
/// - `x_pow_a` is `Some` if and only if `dist == BinaryBlock`.
///
/// ## Thread Safety
///
/// `BlindRotationKeyPrepared<&[u8], BRA, BE>` is `Sync` (shared references
/// can be passed to multiple threads simultaneously) because all fields are
/// immutable on the shared path.
#[derive(PartialEq)]
pub struct BlindRotationKeyPrepared<D: Data, BRT: BlindRotationAlgo, B: Backend> {
    pub(crate) data: Vec<GGSWPrepared<D, B>>,
    pub(crate) dist: Distribution,
    pub(crate) x_pow_a: Option<Vec<SvpPPolOwned<B>>>,
    pub(crate) _phantom: PhantomData<BRT>,
}

impl<D: Data, BRT: BlindRotationAlgo, B: Backend> BlindRotationKeyInfos for BlindRotationKeyPrepared<D, BRT, B> {
    fn n_glwe(&self) -> Degree {
        self.n()
    }

    fn n_lwe(&self) -> Degree {
        Degree(self.data.len() as u32)
    }
}

impl<D: Data, BRT: BlindRotationAlgo, B: Backend> LWEInfos for BlindRotationKeyPrepared<D, BRT, B> {
    fn base2k(&self) -> Base2K {
        self.data[0].base2k()
    }

    fn n(&self) -> Degree {
        self.data[0].n()
    }

    fn max_size(&self) -> usize {
        self.data[0].max_size()
    }

    fn k(&self) -> poulpy_core::layouts::TorusPrecision {
        self.data[0].k()
    }
}

impl<D: Data, BRT: BlindRotationAlgo, B: Backend> GLWEInfos for BlindRotationKeyPrepared<D, BRT, B> {
    fn rank(&self) -> Rank {
        self.data[0].rank()
    }
}
impl<D: Data, BRT: BlindRotationAlgo, B: Backend> GGSWInfos for BlindRotationKeyPrepared<D, BRT, B> {
    fn k_aux(&self) -> poulpy_core::layouts::TorusPrecision {
        self.data[0].k_aux()
    }

    fn dsize(&self) -> poulpy_core::layouts::Dsize {
        Dsize(1)
    }

    fn dnum(&self) -> Dnum {
        self.data[0].dnum()
    }
}

impl<D: Data, BRT: BlindRotationAlgo, B: Backend> BlindRotationKeyPrepared<D, BRT, B> {
    pub fn block_size(&self) -> usize {
        match self.dist {
            Distribution::BinaryBlock(value) => value,
            _ => 1,
        }
    }
}

impl<D: Data, BRA: BlindRotationAlgo, BE: Backend> BlindRotationKeyPrepared<D, BRA, BE> {
    /// Constructs a prepared key from backend-owned prepared elements and monomials.
    pub fn from_parts(
        data: Vec<GGSWPrepared<D, BE>>,
        distribution: Distribution,
        monomials: Option<Vec<SvpPPolOwned<BE>>>,
    ) -> Self {
        assert!(!data.is_empty());
        Self {
            data,
            dist: distribution,
            x_pow_a: monomials,
            _phantom: PhantomData,
        }
    }
    /// Prepared key elements.
    pub fn keys(&self) -> &[GGSWPrepared<D, BE>] {
        &self.data
    }
    /// Mutable prepared key elements.
    pub fn keys_mut(&mut self) -> &mut [GGSWPrepared<D, BE>] {
        &mut self.data
    }
    /// Secret distribution carried by the prepared key.
    pub fn distribution(&self) -> Distribution {
        self.dist
    }
    /// Updates distribution metadata after preparation.
    pub fn set_distribution(&mut self, distribution: Distribution) {
        self.dist = distribution;
    }
    /// Prepared monomials used by block execution.
    pub fn monomials(&self) -> Option<&[SvpPPolOwned<BE>]> {
        self.x_pow_a.as_deref()
    }
    /// Replaces the prepared monomial table.
    pub fn set_monomials(&mut self, monomials: Option<Vec<SvpPPolOwned<BE>>>) {
        self.x_pow_a = monomials;
    }
}
