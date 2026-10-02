use crate::blind_rotation::{BlindRotationAlgo, BlindRotationKeyCompressed, BlindRotationKeyInfos};
use poulpy_hal::layouts::Backend;

/// Allocates compressed blind-rotation keys through the selected backend.
///
/// Implemented for `Module<BE>` through
/// [`crate::oep::BlindRotationKeyCompressedFactoryImpl`].
pub trait BlindRotationKeyCompressedFactory<BRA: BlindRotationAlgo, BE: Backend> {
    /// Allocates zero-filled compressed key elements using the module's storage.
    fn blind_rotation_key_compressed_alloc<A>(&self, infos: &A) -> BlindRotationKeyCompressed<BE::OwnedBuf, BRA, BE::ZnxWord>
    where
        A: BlindRotationKeyInfos;
}
