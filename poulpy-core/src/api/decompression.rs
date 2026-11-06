use poulpy_hal::layouts::Backend;

use crate::{
    GetDistribution, GetDistributionMut,
    layouts::{
        GGLWECompressedToBackendRef, GGLWEInfos, GGLWEToBackendMut, GGLWEToGGSWKeyCompressedToBackendRef,
        GGLWEToGGSWKeyToBackendMut, GGSWCompressedToBackendRef, GGSWInfos, GGSWToBackendMut, GLWECompressedToBackendRef,
        GLWEInfos, GLWEPublicKeyCompressedToBackendRef, GLWEPublicKeyToBackendMut, GLWESwitchingKeyDegrees,
        GLWESwitchingKeyDegreesMut, GLWEToBackendMut, GetGaloisElement, LWECompressedToBackendRef, LWEInfos, LWEToBackendMut,
        SetBase2k, SetGLWEPlaintextInfos, SetGaloisElement,
    },
};

/// Decompresses a [`GLWECompressed`](crate::layouts::GLWECompressed) into a standard [`GLWE`](crate::layouts::GLWE).
///
/// Copies the stored data from the compressed ciphertext and regenerates
/// the mask polynomials from the stored PRNG seed.
pub trait GLWEDecompress<BE: Backend> {
    /// Decompresses `other` into `res` by copying the stored data and regenerating the mask.
    fn glwe_decompress<R, O>(&self, res: &mut R, other: &O)
    where
        R: GLWEToBackendMut<BE> + SetBase2k + SetGLWEPlaintextInfos,
        O: GLWECompressedToBackendRef<BE> + GLWEInfos;
}

/// Decompresses a [`GGLWECompressed`](crate::layouts::GGLWECompressed) into a standard [`GGLWE`](crate::layouts::GGLWE).
///
/// Iterates over every (row, column) entry, decompressing each
/// compressed GLWE row individually via [`GLWEDecompress`].
pub trait GGLWEDecompress<BE: Backend> {
    /// Decompresses `other` into `res`.
    fn gglwe_decompress<R, O>(&self, res: &mut R, other: &O)
    where
        R: GGLWEToBackendMut<BE> + GGLWEInfos,
        O: GGLWECompressedToBackendRef<BE> + GGLWEInfos;
}

/// Decompresses a [`GGSWCompressed`](crate::layouts::GGSWCompressed) into a standard [`GGSW`](crate::layouts::GGSW).
///
/// Iterates over every (row, column) entry, decompressing each
/// compressed GLWE individually via [`GLWEDecompress`].
pub trait GGSWDecompress<BE: Backend> {
    /// Decompresses `other` into `res`.
    fn ggsw_decompress<R, O>(&self, res: &mut R, other: &O)
    where
        R: GGSWToBackendMut<BE> + GGSWInfos,
        O: GGSWCompressedToBackendRef<BE> + GGSWInfos;
}

/// Decompresses a [`GLWEPublicKeyCompressed`](crate::layouts::GLWEPublicKeyCompressed) into a
/// [`GLWEPublicKey`](crate::layouts::GLWEPublicKey) of the same layout.
pub trait GLWEPublicKeyDecompress<BE: Backend> {
    /// Copies each body, regenerates its mask from its seed, and copies the
    /// distribution.
    fn glwe_public_key_decompress<R, O>(&self, res: &mut R, other: &O)
    where
        R: GLWEPublicKeyToBackendMut<BE> + GetDistributionMut + GLWEInfos,
        O: GLWEPublicKeyCompressedToBackendRef<BE> + GetDistribution + GLWEInfos;
}

/// Decompresses an [`LWECompressed`](crate::layouts::LWECompressed) into a standard [`LWE`](crate::layouts::LWE).
pub trait LWEDecompress<BE: Backend> {
    fn lwe_decompress<R, O>(&self, res: &mut R, other: &O)
    where
        R: LWEToBackendMut<BE> + LWEInfos + SetBase2k,
        O: LWECompressedToBackendRef<BE>;
}

/// Decompresses a [`GLWEAutomorphismKeyCompressed`](crate::layouts::GLWEAutomorphismKeyCompressed) into a
/// [`GLWEAutomorphismKey`](crate::layouts::GLWEAutomorphismKey).
pub trait GLWEAutomorphismKeyDecompress<BE: Backend> {
    fn glwe_automorphism_key_decompress<R, O>(&self, res: &mut R, other: &O)
    where
        R: GGLWEToBackendMut<BE> + GGLWEInfos + SetGaloisElement,
        O: GGLWECompressedToBackendRef<BE> + GGLWEInfos + GetGaloisElement;
}

/// Decompresses a [`GLWESwitchingKeyCompressed`](crate::layouts::GLWESwitchingKeyCompressed) into a standard
/// [`GLWESwitchingKey`](crate::layouts::GLWESwitchingKey).
pub trait GLWESwitchingKeyDecompress<BE: Backend> {
    /// Decompresses `other` into `res`, copying degree metadata.
    fn glwe_switching_key_decompress<R, O>(&self, res: &mut R, other: &O)
    where
        R: GGLWEToBackendMut<BE> + GGLWEInfos + GLWESwitchingKeyDegreesMut,
        O: GGLWECompressedToBackendRef<BE> + GGLWEInfos + GLWESwitchingKeyDegrees;
}

/// Decompresses a [`GLWETensorKeyCompressed`](crate::layouts::GLWETensorKeyCompressed) into a
/// [`GLWETensorKey`](crate::layouts::GLWETensorKey).
pub trait GLWETensorKeyDecompress<BE: Backend> {
    fn glwe_tensor_key_decompress<R, O>(&self, res: &mut R, other: &O)
    where
        R: GGLWEToBackendMut<BE> + GGLWEInfos,
        O: GGLWECompressedToBackendRef<BE> + GGLWEInfos;
}

/// Decompresses a [`GLWEToLWEKeyCompressed`](crate::layouts::GLWEToLWEKeyCompressed) into a
/// [`GLWEToLWEKey`](crate::layouts::GLWEToLWEKey).
pub trait GLWEToLWEKeyDecompress<BE: Backend> {
    fn glwe_to_lwe_key_decompress<R, O>(&self, res: &mut R, other: &O)
    where
        R: GGLWEToBackendMut<BE> + GGLWEInfos + GLWESwitchingKeyDegreesMut,
        O: GGLWECompressedToBackendRef<BE> + GGLWEInfos + GLWESwitchingKeyDegrees;
}

/// Decompresses an [`LWESwitchingKeyCompressed`](crate::layouts::LWESwitchingKeyCompressed) into an
/// [`LWESwitchingKey`](crate::layouts::LWESwitchingKey).
pub trait LWESwitchingKeyDecompress<BE: Backend> {
    fn lwe_switching_key_decompress<R, O>(&self, res: &mut R, other: &O)
    where
        R: GGLWEToBackendMut<BE> + GGLWEInfos + GLWESwitchingKeyDegreesMut,
        O: GGLWECompressedToBackendRef<BE> + GGLWEInfos + GLWESwitchingKeyDegrees;
}

/// Decompresses an [`LWEToGLWEKeyCompressed`](crate::layouts::LWEToGLWEKeyCompressed) into an
/// [`LWEToGLWEKey`](crate::layouts::LWEToGLWEKey).
pub trait LWEToGLWEKeyDecompress<BE: Backend> {
    fn lwe_to_glwe_key_decompress<R, O>(&self, res: &mut R, other: &O)
    where
        R: GGLWEToBackendMut<BE> + GGLWEInfos + GLWESwitchingKeyDegreesMut,
        O: GGLWECompressedToBackendRef<BE> + GGLWEInfos + GLWESwitchingKeyDegrees;
}

/// Decompresses a [`GGLWEToGGSWKeyCompressed`](crate::layouts::GGLWEToGGSWKeyCompressed) into a standard
/// [`GGLWEToGGSWKey`](crate::layouts::GGLWEToGGSWKey).
pub trait GGLWEToGGSWKeyDecompress<BE: Backend> {
    /// Decompresses `other` into `res` by decompressing each GGLWE entry.
    fn gglwe_to_ggsw_key_decompress<R, O>(&self, res: &mut R, other: &O)
    where
        R: GGLWEToGGSWKeyToBackendMut<BE>,
        O: GGLWEToGGSWKeyCompressedToBackendRef<BE>;
}
