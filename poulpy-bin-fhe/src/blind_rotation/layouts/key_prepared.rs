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
/// Implemented for `Module<BE>` when the backend supports the required
/// DFT-domain preparation operations.  Callers should use the convenience
/// methods on [`BlindRotationKeyPrepared`] rather than calling these directly.
pub trait BlindRotationKeyPreparedFactory<BRA: BlindRotationAlgo, BE: Backend> {
    /// Allocates a zero-filled prepared key from a dimension descriptor.
    fn blind_rotation_key_prepared_alloc<A>(&self, infos: &A) -> BlindRotationKeyPrepared<BE::OwnedBuf, BRA, BE>
    where
        A: BlindRotationKeyInfos;

    /// Returns the minimum scratch-space size in bytes required by
    /// [`prepare_blind_rotation_key`][Self::prepare_blind_rotation_key].
    fn blind_rotation_key_prepare_tmp_bytes<A>(&self, infos: &A) -> usize
    where
        A: BlindRotationKeyInfos;

    /// Transforms the standard key `other` into the DFT-domain prepared form
    /// `res`, ready for use in `BlindRotationExecute::blind_rotation_execute`.
    ///
    /// For the `BinaryBlock` distribution this also pre-computes the
    /// `X^{a_i}` scalar polynomial products used in the batched CMux loop.
    fn prepare_blind_rotation_key(
        &self,
        res: &mut BlindRotationKeyPrepared<BE::OwnedBuf, BRA, BE>,
        other: &BlindRotationKey<BE::OwnedBuf, BRA>,
        scratch: &mut ScratchArena<'_, BE>,
    );
}

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
    pub fn prepare<M>(&mut self, module: &M, other: &BlindRotationKey<BE::OwnedBuf, BRA>, scratch: &mut ScratchArena<'_, BE>)
    where
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
#[derive(PartialEq, Eq)]
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

    fn size(&self) -> usize {
        self.data[0].size()
    }
}

impl<D: Data, BRT: BlindRotationAlgo, B: Backend> GLWEInfos for BlindRotationKeyPrepared<D, BRT, B> {
    fn rank(&self) -> Rank {
        self.data[0].rank()
    }
}
impl<D: Data, BRT: BlindRotationAlgo, B: Backend> GGSWInfos for BlindRotationKeyPrepared<D, BRT, B> {
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
