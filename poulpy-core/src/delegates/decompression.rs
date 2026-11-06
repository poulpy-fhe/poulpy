use poulpy_hal::layouts::{Backend, Module};

use crate::{
    GetDistribution, GetDistributionMut,
    api::{
        GGLWEDecompress, GGLWEToGGSWKeyDecompress, GGSWDecompress, GLWEAutomorphismKeyDecompress, GLWEDecompress,
        GLWEPublicKeyDecompress, GLWESwitchingKeyDecompress, GLWETensorKeyDecompress, GLWEToLWEKeyDecompress, LWEDecompress,
        LWESwitchingKeyDecompress, LWEToGLWEKeyDecompress,
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
    fn glwe_decompress<R, O>(&self, res: &mut R, other: &O)
    where
        R: GLWEToBackendMut<BE> + SetBase2k + SetGLWEPlaintextInfos,
        O: GLWECompressedToBackendRef<BE> + GLWEInfos,
    {
        BE::glwe_decompress(self, res, other)
    }
);

impl_decompression_delegate!(
    GGLWEDecompress<BE>,
    fn gglwe_decompress<R, O>(&self, res: &mut R, other: &O)
    where
        R: GGLWEToBackendMut<BE> + GGLWEInfos,
        O: GGLWECompressedToBackendRef<BE> + GGLWEInfos,
    {
        BE::gglwe_decompress(self, res, other)
    }
);

impl_decompression_delegate!(
    GGSWDecompress<BE>,
    fn ggsw_decompress<R, O>(&self, res: &mut R, other: &O)
    where
        R: GGSWToBackendMut<BE> + GGSWInfos,
        O: GGSWCompressedToBackendRef<BE> + GGSWInfos,
    {
        BE::ggsw_decompress(self, res, other)
    }
);

impl_decompression_delegate!(
    GLWEPublicKeyDecompress<BE>,
    fn glwe_public_key_decompress<R, O>(&self, res: &mut R, other: &O)
    where
        R: GLWEPublicKeyToBackendMut<BE> + GetDistributionMut + GLWEInfos,
        O: GLWEPublicKeyCompressedToBackendRef<BE> + GetDistribution + GLWEInfos,
    {
        BE::glwe_public_key_decompress(self, res, other)
    }
);

impl_decompression_delegate!(
    LWEDecompress<BE>,
    fn lwe_decompress<R, O>(&self, res: &mut R, other: &O)
    where
        R: LWEToBackendMut<BE> + LWEInfos + SetBase2k,
        O: LWECompressedToBackendRef<BE>,
    {
        BE::lwe_decompress(self, res, other)
    }
);

impl_decompression_delegate!(
    GLWEAutomorphismKeyDecompress<BE>,
    fn glwe_automorphism_key_decompress<R, O>(&self, res: &mut R, other: &O)
    where
        R: GGLWEToBackendMut<BE> + GGLWEInfos + SetGaloisElement,
        O: GGLWECompressedToBackendRef<BE> + GGLWEInfos + GetGaloisElement,
    {
        BE::glwe_automorphism_key_decompress(self, res, other)
    }
);

impl_decompression_delegate!(
    GLWESwitchingKeyDecompress<BE>,
    fn glwe_switching_key_decompress<R, O>(&self, res: &mut R, other: &O)
    where
        R: GGLWEToBackendMut<BE> + GGLWEInfos + GLWESwitchingKeyDegreesMut,
        O: GGLWECompressedToBackendRef<BE> + GGLWEInfos + GLWESwitchingKeyDegrees,
    {
        BE::glwe_switching_key_decompress(self, res, other)
    }
);

impl_decompression_delegate!(
    GLWETensorKeyDecompress<BE>,
    fn glwe_tensor_key_decompress<R, O>(&self, res: &mut R, other: &O)
    where
        R: GGLWEToBackendMut<BE> + GGLWEInfos,
        O: GGLWECompressedToBackendRef<BE> + GGLWEInfos,
    {
        BE::glwe_tensor_key_decompress(self, res, other)
    }
);

impl_decompression_delegate!(
    GLWEToLWEKeyDecompress<BE>,
    fn glwe_to_lwe_key_decompress<R, O>(&self, res: &mut R, other: &O)
    where
        R: GGLWEToBackendMut<BE> + GGLWEInfos + GLWESwitchingKeyDegreesMut,
        O: GGLWECompressedToBackendRef<BE> + GGLWEInfos + GLWESwitchingKeyDegrees,
    {
        BE::glwe_to_lwe_key_decompress(self, res, other)
    }
);

impl_decompression_delegate!(
    LWESwitchingKeyDecompress<BE>,
    fn lwe_switching_key_decompress<R, O>(&self, res: &mut R, other: &O)
    where
        R: GGLWEToBackendMut<BE> + GGLWEInfos + GLWESwitchingKeyDegreesMut,
        O: GGLWECompressedToBackendRef<BE> + GGLWEInfos + GLWESwitchingKeyDegrees,
    {
        BE::lwe_switching_key_decompress(self, res, other)
    }
);

impl_decompression_delegate!(
    LWEToGLWEKeyDecompress<BE>,
    fn lwe_to_glwe_key_decompress<R, O>(&self, res: &mut R, other: &O)
    where
        R: GGLWEToBackendMut<BE> + GGLWEInfos + GLWESwitchingKeyDegreesMut,
        O: GGLWECompressedToBackendRef<BE> + GGLWEInfos + GLWESwitchingKeyDegrees,
    {
        BE::lwe_to_glwe_key_decompress(self, res, other)
    }
);

impl_decompression_delegate!(
    GGLWEToGGSWKeyDecompress<BE>,
    fn gglwe_to_ggsw_key_decompress<R, O>(&self, res: &mut R, other: &O)
    where
        R: GGLWEToGGSWKeyToBackendMut<BE>,
        O: GGLWEToGGSWKeyCompressedToBackendRef<BE>,
    {
        BE::gglwe_to_ggsw_key_decompress(self, res, other)
    }
);
