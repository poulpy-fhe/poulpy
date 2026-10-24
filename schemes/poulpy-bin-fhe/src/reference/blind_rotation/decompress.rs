use crate::blind_rotation::{BlindRotationKey, BlindRotationKeyCompressed, BlindRotationKeyInfos, CGGI};
use poulpy_core::layouts::{GGSWDecompress, GLWEDecompress};
use poulpy_hal::layouts::{Backend, Module, ScratchArena};

/// Scratch for coefficient-domain decompression; the core composition needs none.
pub fn blind_rotation_key_decompress_tmp_bytes_ref<BE: Backend, A: BlindRotationKeyInfos>(
    _module: &Module<BE>,
    _infos: &A,
) -> usize {
    0
}

/// Recreates each GGSW mask through the selected core decompression operation.
pub fn blind_rotation_key_decompress_ref<BE: Backend>(
    module: &Module<BE>,
    res: &mut BlindRotationKey<BE::OwnedBuf, CGGI, BE::ZnxWord>,
    src: &BlindRotationKeyCompressed<BE::OwnedBuf, CGGI, BE::ZnxWord>,
    _scratch: &mut ScratchArena<'_, BE>,
) where
    Module<BE>: GGSWDecompress + GLWEDecompress<Backend = BE>,
{
    assert_eq!(res.keys.len(), src.keys.len());
    for (output, input) in res.keys.iter_mut().zip(&src.keys) {
        module.decompress_ggsw(output, input);
    }
    res.dist = src.dist;
}
