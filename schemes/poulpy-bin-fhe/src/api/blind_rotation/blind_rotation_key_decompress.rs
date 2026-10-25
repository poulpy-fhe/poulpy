use crate::blind_rotation::{BlindRotationAlgo, BlindRotationKey, BlindRotationKeyCompressed, BlindRotationKeyInfos};
use poulpy_hal::layouts::{Backend, ScratchArena};

/// Decompresses a blind-rotation key into coefficient-domain storage.
pub trait BlindRotationKeyDecompress<BRA: BlindRotationAlgo, BE: Backend> {
    /// Returns the scratch-space size in bytes for [decompression][Self::blind_rotation_key_decompress].
    fn blind_rotation_key_decompress_tmp_bytes<A: BlindRotationKeyInfos>(&self, infos: &A) -> usize;
    /// Restores the masks from `src` into the preallocated coefficient-domain key `res`.
    fn blind_rotation_key_decompress(
        &self,
        res: &mut BlindRotationKey<BE::OwnedBuf, BRA, BE::ZnxWord>,
        src: &BlindRotationKeyCompressed<BE::OwnedBuf, BRA, BE::ZnxWord>,
        scratch: &mut ScratchArena<'_, BE>,
    );
}
