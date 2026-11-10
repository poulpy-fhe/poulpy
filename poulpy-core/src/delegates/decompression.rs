use poulpy_hal::layouts::{Backend, Module, Ring};

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
    oep::{DecompressionImpl, LWEKeyDecompressionImpl},
};

macro_rules! impl_decompression_delegate {
    (free $trait:ty, $($body:item),+ $(,)?) => {
        impl<BE> $trait for Module<BE>
        where
            BE: Backend + LWEKeyDecompressionImpl,
        {
            $($body)+
        }
    };
    ($trait:ty, $($body:item),+ $(,)?) => {
        impl<BE, Rg: Ring> $trait for Module<BE>
        where
            BE: Backend + DecompressionImpl<Rg>,
        {
            $($body)+
        }
    };
}

impl_decompression_delegate!(
    GLWEDecompress<BE, Rg>,
    fn glwe_decompress<R, O>(&self, res: &mut R, other: &O)
    where
        R: GLWEToBackendMut<BE, Rg> + SetBase2k + SetGLWEPlaintextInfos,
        O: GLWECompressedToBackendRef<BE, Rg> + GLWEInfos,
    {
        BE::glwe_decompress(self, res, other)
    }
);

impl_decompression_delegate!(
    GGLWEDecompress<BE, Rg>,
    fn gglwe_decompress<R, O>(&self, res: &mut R, other: &O)
    where
        R: GGLWEToBackendMut<BE, Rg> + GGLWEInfos,
        O: GGLWECompressedToBackendRef<BE, Rg> + GGLWEInfos,
    {
        BE::gglwe_decompress(self, res, other)
    }
);

impl_decompression_delegate!(
    GGSWDecompress<BE, Rg>,
    fn ggsw_decompress<R, O>(&self, res: &mut R, other: &O)
    where
        R: GGSWToBackendMut<BE, Rg> + GGSWInfos,
        O: GGSWCompressedToBackendRef<BE, Rg> + GGSWInfos,
    {
        BE::ggsw_decompress(self, res, other)
    }
);

impl_decompression_delegate!(
    GLWEPublicKeyDecompress<BE, Rg>,
    fn glwe_public_key_decompress<R, O>(&self, res: &mut R, other: &O)
    where
        R: GLWEPublicKeyToBackendMut<BE, Rg> + GetDistributionMut + GLWEInfos,
        O: GLWEPublicKeyCompressedToBackendRef<BE, Rg> + GetDistribution + GLWEInfos,
    {
        BE::glwe_public_key_decompress(self, res, other)
    }
);

impl_decompression_delegate!(
    LWEDecompress<BE, Rg>,
    fn lwe_decompress<R, O>(&self, res: &mut R, other: &O)
    where
        R: LWEToBackendMut<BE, Rg> + LWEInfos + SetBase2k,
        O: LWECompressedToBackendRef<BE, Rg>,
    {
        BE::lwe_decompress(self, res, other)
    }
);

impl_decompression_delegate!(
    GLWEAutomorphismKeyDecompress<BE, Rg>,
    fn glwe_automorphism_key_decompress<R, O>(&self, res: &mut R, other: &O)
    where
        R: GGLWEToBackendMut<BE, Rg> + GGLWEInfos + SetGaloisElement,
        O: GGLWECompressedToBackendRef<BE, Rg> + GGLWEInfos + GetGaloisElement,
    {
        BE::glwe_automorphism_key_decompress(self, res, other)
    }
);

impl_decompression_delegate!(
    GLWESwitchingKeyDecompress<BE, Rg>,
    fn glwe_switching_key_decompress<R, O>(&self, res: &mut R, other: &O)
    where
        R: GGLWEToBackendMut<BE, Rg> + GGLWEInfos + GLWESwitchingKeyDegreesMut,
        O: GGLWECompressedToBackendRef<BE, Rg> + GGLWEInfos + GLWESwitchingKeyDegrees,
    {
        BE::glwe_switching_key_decompress(self, res, other)
    }
);

impl_decompression_delegate!(
    GLWETensorKeyDecompress<BE, Rg>,
    fn glwe_tensor_key_decompress<R, O>(&self, res: &mut R, other: &O)
    where
        R: GGLWEToBackendMut<BE, Rg> + GGLWEInfos,
        O: GGLWECompressedToBackendRef<BE, Rg> + GGLWEInfos,
    {
        BE::glwe_tensor_key_decompress(self, res, other)
    }
);

impl_decompression_delegate!(
    free GLWEToLWEKeyDecompress<BE>,
    fn glwe_to_lwe_key_decompress<R, O>(&self, res: &mut R, other: &O)
    where
        R: GGLWEToBackendMut<BE> + GGLWEInfos + GLWESwitchingKeyDegreesMut,
        O: GGLWECompressedToBackendRef<BE> + GGLWEInfos + GLWESwitchingKeyDegrees,
    {
        BE::glwe_to_lwe_key_decompress(self, res, other)
    }
);

impl_decompression_delegate!(
    free LWESwitchingKeyDecompress<BE>,
    fn lwe_switching_key_decompress<R, O>(&self, res: &mut R, other: &O)
    where
        R: GGLWEToBackendMut<BE> + GGLWEInfos + GLWESwitchingKeyDegreesMut,
        O: GGLWECompressedToBackendRef<BE> + GGLWEInfos + GLWESwitchingKeyDegrees,
    {
        BE::lwe_switching_key_decompress(self, res, other)
    }
);

impl_decompression_delegate!(
    free LWEToGLWEKeyDecompress<BE>,
    fn lwe_to_glwe_key_decompress<R, O>(&self, res: &mut R, other: &O)
    where
        R: GGLWEToBackendMut<BE> + GGLWEInfos + GLWESwitchingKeyDegreesMut,
        O: GGLWECompressedToBackendRef<BE> + GGLWEInfos + GLWESwitchingKeyDegrees,
    {
        BE::lwe_to_glwe_key_decompress(self, res, other)
    }
);

impl_decompression_delegate!(
    GGLWEToGGSWKeyDecompress<BE, Rg>,
    fn gglwe_to_ggsw_key_decompress<R, O>(&self, res: &mut R, other: &O)
    where
        R: GGLWEToGGSWKeyToBackendMut<BE, Rg>,
        O: GGLWEToGGSWKeyCompressedToBackendRef<BE, Rg>,
    {
        BE::gglwe_to_ggsw_key_decompress(self, res, other)
    }
);
