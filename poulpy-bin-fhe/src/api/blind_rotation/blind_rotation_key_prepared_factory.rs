use crate::blind_rotation::{BlindRotationAlgo, BlindRotationKey, BlindRotationKeyInfos, BlindRotationKeyPrepared};
use poulpy_hal::layouts::{Backend, ScratchArena};

/// Allocates and prepares backend-owned blind-rotation key representations.
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
        other: &BlindRotationKey<BE::OwnedBuf, BRA, BE::ZnxWord>,
        scratch: &mut ScratchArena<'_, BE>,
    );
}
