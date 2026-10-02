use crate::{
    api::BlindRotationKeyCompressedFactory,
    blind_rotation::{BlindRotationAlgo, BlindRotationKeyCompressed, BlindRotationKeyInfos},
    oep::BlindRotationKeyCompressedFactoryImpl,
};
use poulpy_hal::layouts::Module;

impl<BRA: BlindRotationAlgo, BE: BlindRotationKeyCompressedFactoryImpl<BRA>> BlindRotationKeyCompressedFactory<BRA, BE>
    for Module<BE>
{
    fn blind_rotation_key_compressed_alloc<A>(&self, infos: &A) -> BlindRotationKeyCompressed<BE::OwnedBuf, BRA, BE::ZnxWord>
    where
        A: BlindRotationKeyInfos,
    {
        BE::blind_rotation_key_compressed_alloc(self, infos)
    }
}
