use poulpy_hal::layouts::{Backend, Module};

use crate::{
    GetDistribution, GetDistributionMut,
    api::{
        GGLWEDecompress, GGLWEToGGSWKeyDecompress, GGSWDecompress, GLWEAutomorphismKeyDecompress, GLWEDecompress,
        GLWEPublicKeyDecompress, GLWESwitchingKeyDecompress, GLWETensorKeyDecompress, GLWEToLWESwitchingKeyDecompress,
        LWEDecompress, LWESwitchingKeyDecompress, LWEToGLWEKeyDecompress,
    },
    layouts::{
        GGLWECompressedToBackendRef, GGLWEInfos, GGLWEToBackendMut, GGLWEToGGSWKeyCompressedToBackendRef,
        GGLWEToGGSWKeyToBackendMut, GGSWCompressedToBackendRef, GGSWInfos, GGSWToBackendMut, GLWECompressedToBackendRef,
        GLWEInfos, GLWEPublicKeyCompressedToBackendRef, GLWEPublicKeyToBackendMut, GLWESwitchingKeyDegrees,
        GLWESwitchingKeyDegreesMut, GLWEToBackendMut, GetGaloisElement, LWECompressedToBackendRef, LWEInfos, LWEToBackendMut,
        SetBase2k, SetGLWEPlaintextInfos, SetGaloisElement,
    },
    oep::DecompressionImpl,
};

macro_rules! impl_decompression_delegate {
    ($trait:ty, $($body:item),+ $(,)?) => {
        impl<BE> $trait for Module<BE>
        where
            BE: Backend + DecompressionImpl,
        {
            $($body)+
        }
    };
}

impl_decompression_delegate!(
    GLWEDecompress<BE>,
    fn decompress_glwe<R, O>(&self, res: &mut R, other: &O)
    where
        R: GLWEToBackendMut<BE> + SetBase2k + SetGLWEPlaintextInfos,
        O: GLWECompressedToBackendRef<BE> + GLWEInfos,
    {
        BE::decompress_glwe(self, res, other)
    }
);

impl_decompression_delegate!(
    GGLWEDecompress<BE>,
    fn decompress_gglwe<R, O>(&self, res: &mut R, other: &O)
    where
        R: GGLWEToBackendMut<BE> + GGLWEInfos,
        O: GGLWECompressedToBackendRef<BE> + GGLWEInfos,
    {
        BE::decompress_gglwe(self, res, other)
    }
);

impl_decompression_delegate!(
    GGSWDecompress<BE>,
    fn decompress_ggsw<R, O>(&self, res: &mut R, other: &O)
    where
        R: GGSWToBackendMut<BE> + GGSWInfos,
        O: GGSWCompressedToBackendRef<BE> + GGSWInfos,
    {
        BE::decompress_ggsw(self, res, other)
    }
);

impl_decompression_delegate!(
    GLWEPublicKeyDecompress<BE>,
    fn decompress_glwe_public_key<R, O>(&self, res: &mut R, other: &O)
    where
        R: GLWEPublicKeyToBackendMut<BE> + GetDistributionMut + GLWEInfos,
        O: GLWEPublicKeyCompressedToBackendRef<BE> + GetDistribution + GLWEInfos,
    {
        BE::decompress_glwe_public_key(self, res, other)
    }
);

impl_decompression_delegate!(
    LWEDecompress<BE>,
    fn decompress_lwe<R, O>(&self, res: &mut R, other: &O)
    where
        R: LWEToBackendMut<BE> + LWEInfos + SetBase2k,
        O: LWECompressedToBackendRef<BE>,
    {
        BE::decompress_lwe(self, res, other)
    }
);

impl_decompression_delegate!(
    GLWEAutomorphismKeyDecompress<BE>,
    fn decompress_automorphism_key<R, O>(&self, res: &mut R, other: &O)
    where
        R: GGLWEToBackendMut<BE> + GGLWEInfos + SetGaloisElement,
        O: GGLWECompressedToBackendRef<BE> + GGLWEInfos + GetGaloisElement,
    {
        BE::decompress_automorphism_key(self, res, other)
    }
);

impl_decompression_delegate!(
    GLWESwitchingKeyDecompress<BE>,
    fn decompress_glwe_switching_key<R, O>(&self, res: &mut R, other: &O)
    where
        R: GGLWEToBackendMut<BE> + GGLWEInfos + GLWESwitchingKeyDegreesMut,
        O: GGLWECompressedToBackendRef<BE> + GGLWEInfos + GLWESwitchingKeyDegrees,
    {
        BE::decompress_glwe_switching_key(self, res, other)
    }
);

impl_decompression_delegate!(
    GLWETensorKeyDecompress<BE>,
    fn decompress_tensor_key<R, O>(&self, res: &mut R, other: &O)
    where
        R: GGLWEToBackendMut<BE> + GGLWEInfos,
        O: GGLWECompressedToBackendRef<BE> + GGLWEInfos,
    {
        BE::decompress_tensor_key(self, res, other)
    }
);

impl_decompression_delegate!(
    GLWEToLWESwitchingKeyDecompress<BE>,
    fn decompress_glwe_to_lwe_key<R, O>(&self, res: &mut R, other: &O)
    where
        R: GGLWEToBackendMut<BE> + GGLWEInfos + GLWESwitchingKeyDegreesMut,
        O: GGLWECompressedToBackendRef<BE> + GGLWEInfos + GLWESwitchingKeyDegrees,
    {
        BE::decompress_glwe_to_lwe_key(self, res, other)
    }
);

impl_decompression_delegate!(
    LWESwitchingKeyDecompress<BE>,
    fn decompress_lwe_switching_key<R, O>(&self, res: &mut R, other: &O)
    where
        R: GGLWEToBackendMut<BE> + GGLWEInfos + GLWESwitchingKeyDegreesMut,
        O: GGLWECompressedToBackendRef<BE> + GGLWEInfos + GLWESwitchingKeyDegrees,
    {
        BE::decompress_lwe_switching_key(self, res, other)
    }
);

impl_decompression_delegate!(
    LWEToGLWEKeyDecompress<BE>,
    fn decompress_lwe_to_glwe_key<R, O>(&self, res: &mut R, other: &O)
    where
        R: GGLWEToBackendMut<BE> + GGLWEInfos + GLWESwitchingKeyDegreesMut,
        O: GGLWECompressedToBackendRef<BE> + GGLWEInfos + GLWESwitchingKeyDegrees,
    {
        BE::decompress_lwe_to_glwe_key(self, res, other)
    }
);

impl_decompression_delegate!(
    GGLWEToGGSWKeyDecompress<BE>,
    fn decompress_gglwe_to_ggsw_key<R, O>(&self, res: &mut R, other: &O)
    where
        R: GGLWEToGGSWKeyToBackendMut<BE>,
        O: GGLWEToGGSWKeyCompressedToBackendRef<BE>,
    {
        BE::decompress_gglwe_to_ggsw_key(self, res, other)
    }
);
