use poulpy_hal::layouts::{Backend, Module};

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

/// Backend-provided decompression of seed-compressed ciphertexts and keys.
///
/// `decompress_glwe` and `decompress_lwe` are required. The other methods default to compositions of
/// `decompress_glwe`, `decompress_gglwe` and `decompress_glwe_switching_key`. An implementation reproduces
/// the body it replaces bit for bit: the copied bodies, the masks regenerated from the stored seeds,
/// its refusals before any write to `res`, and the metadata it sets on `res`. No method takes scratch;
/// the kernels are free.
///
/// # Safety
/// Implementations read `other` and write `res` only within the regions their layouts describe,
/// and leave `other` unchanged. The memory, layout, synchrony and aliasing rules of
/// `poulpy-hal/docs/backend_safety_contract.md` apply.
pub unsafe trait DecompressionImpl: Backend {
    /// Reproduces
    /// [`decompress_glwe_reference`](crate::reference::decompression::DecompressionReference::decompress_glwe_reference)
    /// bit for bit.
    fn decompress_glwe<R, O>(module: &Module<Self>, res: &mut R, other: &O)
    where
        R: GLWEToBackendMut<Self> + SetBase2k + SetGLWEPlaintextInfos,
        O: GLWECompressedToBackendRef<Self> + GLWEInfos;

    /// Default body: `decompress_gglwe_derived`, `decompress_glwe` on each `(row, column)` entry of the first
    /// `res.dnum()` rows, after checking the digit width, the row offset and the row count, then the source's
    /// noise on `res`.
    fn decompress_gglwe<R, O>(module: &Module<Self>, res: &mut R, other: &O)
    where
        R: GGLWEToBackendMut<Self> + GGLWEInfos,
        O: GGLWECompressedToBackendRef<Self> + GGLWEInfos,
    {
        super::derived::decompression::decompress_gglwe_derived(module, res, other)
    }

    /// Default body: `decompress_ggsw_derived`, `decompress_glwe` on each `(row, column)` entry, after checking
    /// the rank, then the source's noise on `res`.
    fn decompress_ggsw<R, O>(module: &Module<Self>, res: &mut R, other: &O)
    where
        R: GGSWToBackendMut<Self> + GGSWInfos,
        O: GGSWCompressedToBackendRef<Self> + GGSWInfos,
    {
        super::derived::decompression::decompress_ggsw_derived(module, res, other)
    }

    /// Default body: `decompress_glwe_public_key_derived`, `decompress_glwe` on each of the `rank` entries, after
    /// checking the layouts, then the source's noise and distribution on `res`.
    fn decompress_glwe_public_key<R, O>(module: &Module<Self>, res: &mut R, other: &O)
    where
        R: GLWEPublicKeyToBackendMut<Self> + GetDistributionMut + GLWEInfos,
        O: GLWEPublicKeyCompressedToBackendRef<Self> + GetDistribution + GLWEInfos,
    {
        super::derived::decompression::decompress_glwe_public_key_derived(module, res, other)
    }

    /// Reproduces
    /// [`decompress_lwe_reference`](crate::reference::decompression::DecompressionReference::decompress_lwe_reference)
    /// bit for bit.
    fn decompress_lwe<R, O>(module: &Module<Self>, res: &mut R, other: &O)
    where
        R: LWEToBackendMut<Self> + LWEInfos + SetBase2k,
        O: LWECompressedToBackendRef<Self>;

    /// Default body: `decompress_automorphism_key_derived`, `decompress_gglwe`, then the source's Galois element
    /// on `res`.
    fn decompress_automorphism_key<R, O>(module: &Module<Self>, res: &mut R, other: &O)
    where
        R: GGLWEToBackendMut<Self> + GGLWEInfos + SetGaloisElement,
        O: GGLWECompressedToBackendRef<Self> + GGLWEInfos + GetGaloisElement,
    {
        super::derived::decompression::decompress_automorphism_key_derived(module, res, other)
    }

    /// Default body: `decompress_glwe_switching_key_derived`, `decompress_gglwe`, then the source's input and
    /// output degrees on `res`.
    fn decompress_glwe_switching_key<R, O>(module: &Module<Self>, res: &mut R, other: &O)
    where
        R: GGLWEToBackendMut<Self> + GGLWEInfos + GLWESwitchingKeyDegreesMut,
        O: GGLWECompressedToBackendRef<Self> + GGLWEInfos + GLWESwitchingKeyDegrees,
    {
        super::derived::decompression::decompress_glwe_switching_key_derived(module, res, other)
    }

    /// Default body: `decompress_tensor_key_derived`, `decompress_gglwe`.
    fn decompress_tensor_key<R, O>(module: &Module<Self>, res: &mut R, other: &O)
    where
        R: GGLWEToBackendMut<Self> + GGLWEInfos,
        O: GGLWECompressedToBackendRef<Self> + GGLWEInfos,
    {
        super::derived::decompression::decompress_tensor_key_derived(module, res, other)
    }

    /// Default body: `decompress_glwe_to_lwe_key_derived`, `decompress_glwe_switching_key`.
    fn decompress_glwe_to_lwe_key<R, O>(module: &Module<Self>, res: &mut R, other: &O)
    where
        R: GGLWEToBackendMut<Self> + GGLWEInfos + GLWESwitchingKeyDegreesMut,
        O: GGLWECompressedToBackendRef<Self> + GGLWEInfos + GLWESwitchingKeyDegrees,
    {
        super::derived::decompression::decompress_glwe_to_lwe_key_derived(module, res, other)
    }

    /// Default body: `decompress_lwe_switching_key_derived`, `decompress_glwe_switching_key`.
    fn decompress_lwe_switching_key<R, O>(module: &Module<Self>, res: &mut R, other: &O)
    where
        R: GGLWEToBackendMut<Self> + GGLWEInfos + GLWESwitchingKeyDegreesMut,
        O: GGLWECompressedToBackendRef<Self> + GGLWEInfos + GLWESwitchingKeyDegrees,
    {
        super::derived::decompression::decompress_lwe_switching_key_derived(module, res, other)
    }

    /// Default body: `decompress_lwe_to_glwe_key_derived`, `decompress_glwe_switching_key`.
    fn decompress_lwe_to_glwe_key<R, O>(module: &Module<Self>, res: &mut R, other: &O)
    where
        R: GGLWEToBackendMut<Self> + GGLWEInfos + GLWESwitchingKeyDegreesMut,
        O: GGLWECompressedToBackendRef<Self> + GGLWEInfos + GLWESwitchingKeyDegrees,
    {
        super::derived::decompression::decompress_lwe_to_glwe_key_derived(module, res, other)
    }

    /// Default body: `decompress_gglwe_to_ggsw_key_derived`, after checking the key count and each key's digit
    /// width, row offset and row count, [`Self::decompress_gglwe`] on each key, then the source's noise on `res`.
    fn decompress_gglwe_to_ggsw_key<R, O>(module: &Module<Self>, res: &mut R, other: &O)
    where
        R: GGLWEToGGSWKeyToBackendMut<Self>,
        O: GGLWEToGGSWKeyCompressedToBackendRef<Self>,
    {
        super::derived::decompression::decompress_gglwe_to_ggsw_key_derived(module, res, other)
    }
}

/// Selects the reference bodies of the required decompression methods for this backend; the others keep
/// their derived defaults.
#[macro_export]
macro_rules! impl_decompression_reference_full {
    ($be:ty) => {
        unsafe impl $crate::oep::DecompressionImpl for $be {
            fn decompress_glwe<R, O>(module: &::poulpy_hal::layouts::Module<$be>, res: &mut R, other: &O)
            where
                R: $crate::layouts::GLWEToBackendMut<$be> + $crate::layouts::SetBase2k + $crate::layouts::SetGLWEPlaintextInfos,
                O: $crate::layouts::GLWECompressedToBackendRef<$be> + $crate::layouts::GLWEInfos,
            {
                <::poulpy_hal::layouts::Module<$be> as $crate::reference::decompression::DecompressionReference<$be>>::decompress_glwe_reference(module, res, other)
            }

            fn decompress_lwe<R, O>(module: &::poulpy_hal::layouts::Module<$be>, res: &mut R, other: &O)
            where
                R: $crate::layouts::LWEToBackendMut<$be> + $crate::layouts::LWEInfos + $crate::layouts::SetBase2k,
                O: $crate::layouts::LWECompressedToBackendRef<$be>,
            {
                <::poulpy_hal::layouts::Module<$be> as $crate::reference::decompression::DecompressionReference<$be>>::decompress_lwe_reference(module, res, other)
            }
        }
    };
}
