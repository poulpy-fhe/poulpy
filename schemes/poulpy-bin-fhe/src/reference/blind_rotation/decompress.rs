use crate::blind_rotation::{BlindRotationAlgo, BlindRotationKey, BlindRotationKeyCompressed, BlindRotationKeyInfos, CGGI};
use poulpy_core::api::GGSWDecompress;
use poulpy_hal::layouts::{Backend, Module, ScratchArena};

/// Independently callable canonical implementation of
/// [`BlindRotationKeyDecompress::blind_rotation_key_decompress_tmp_bytes`](crate::api::BlindRotationKeyDecompress::blind_rotation_key_decompress_tmp_bytes), selected by
/// `impl_blind_rotation_key_decompress_reference_full!`.
///
/// The core composition needs no scratch.
pub fn blind_rotation_key_decompress_tmp_bytes_reference<BE: Backend, A: BlindRotationKeyInfos>(
    _module: &Module<BE>,
    _infos: &A,
) -> usize {
    0
}

/// Independently callable canonical implementation of
/// [`BlindRotationKeyDecompress::blind_rotation_key_decompress`](crate::api::BlindRotationKeyDecompress::blind_rotation_key_decompress), selected by
/// `impl_blind_rotation_key_decompress_reference_full!`.
///
/// Recreates each GGSW mask through the selected core decompression operation.
pub fn blind_rotation_key_decompress_reference<BE: Backend>(
    module: &Module<BE>,
    res: &mut BlindRotationKey<BE::OwnedBuf, CGGI, BE::ZnxWord>,
    src: &BlindRotationKeyCompressed<BE::OwnedBuf, CGGI, BE::ZnxWord>,
    _scratch: &mut ScratchArena<'_, BE>,
) where
    Module<BE>: GGSWDecompress<BE>,
{
    assert_eq!(res.keys.len(), src.keys.len());
    for (output, input) in res.keys.iter_mut().zip(&src.keys) {
        module.ggsw_decompress(output, input);
    }
    res.dist = src.dist;
}

/// Reference bodies of the [`BlindRotationKeyDecompressImpl`](crate::oep::BlindRotationKeyDecompressImpl) methods for
/// [`CGGI`], which [`impl_blind_rotation_key_decompress_reference_full!`](crate::impl_blind_rotation_key_decompress_reference_full)
/// forwards to.
pub trait BlindRotationKeyDecompressReference<BRA: BlindRotationAlgo, BE: Backend> {
    fn blind_rotation_key_decompress_tmp_bytes_reference<A: BlindRotationKeyInfos>(&self, infos: &A) -> usize;

    fn blind_rotation_key_decompress_reference(
        &self,
        res: &mut BlindRotationKey<BE::OwnedBuf, BRA, BE::ZnxWord>,
        src: &BlindRotationKeyCompressed<BE::OwnedBuf, BRA, BE::ZnxWord>,
        scratch: &mut ScratchArena<'_, BE>,
    );
}

impl<BE: Backend> BlindRotationKeyDecompressReference<CGGI, BE> for Module<BE>
where
    Module<BE>: GGSWDecompress<BE>,
{
    fn blind_rotation_key_decompress_tmp_bytes_reference<A: BlindRotationKeyInfos>(&self, infos: &A) -> usize {
        blind_rotation_key_decompress_tmp_bytes_reference(self, infos)
    }

    fn blind_rotation_key_decompress_reference(
        &self,
        res: &mut BlindRotationKey<BE::OwnedBuf, CGGI, BE::ZnxWord>,
        src: &BlindRotationKeyCompressed<BE::OwnedBuf, CGGI, BE::ZnxWord>,
        scratch: &mut ScratchArena<'_, BE>,
    ) {
        blind_rotation_key_decompress_reference(self, res, src, scratch)
    }
}
