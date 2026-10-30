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
/// `glwe_decompress` and `lwe_decompress` are required. The other methods default to compositions of
/// `glwe_decompress`, `gglwe_decompress` and `glwe_switching_key_decompress`. An implementation reproduces
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
    /// [`glwe_decompress_reference`](crate::reference::decompression::DecompressionReference::glwe_decompress_reference)
    /// bit for bit.
    fn glwe_decompress<R, O>(module: &Module<Self>, res: &mut R, other: &O)
    where
        R: GLWEToBackendMut<Self> + SetBase2k + SetGLWEPlaintextInfos,
        O: GLWECompressedToBackendRef<Self> + GLWEInfos;

    /// Default body: `gglwe_decompress_derived`, `glwe_decompress` on each `(row, column)` entry of the first
    /// `res.dnum()` rows, after checking the digit width, the row offset and the row count, then the source's
    /// noise on `res`.
    fn gglwe_decompress<R, O>(module: &Module<Self>, res: &mut R, other: &O)
    where
        R: GGLWEToBackendMut<Self> + GGLWEInfos,
        O: GGLWECompressedToBackendRef<Self> + GGLWEInfos,
    {
        super::derived::decompression::gglwe_decompress_derived(module, res, other)
    }

    /// Default body: `ggsw_decompress_derived`, `glwe_decompress` on each `(row, column)` entry, after checking
    /// the rank, then the source's noise on `res`.
    fn ggsw_decompress<R, O>(module: &Module<Self>, res: &mut R, other: &O)
    where
        R: GGSWToBackendMut<Self> + GGSWInfos,
        O: GGSWCompressedToBackendRef<Self> + GGSWInfos,
    {
        super::derived::decompression::ggsw_decompress_derived(module, res, other)
    }

    /// Default body: `glwe_public_key_decompress_derived`, `glwe_decompress` on each of the `rank` entries, after
    /// checking the layouts, then the source's noise and distribution on `res`.
    fn glwe_public_key_decompress<R, O>(module: &Module<Self>, res: &mut R, other: &O)
    where
        R: GLWEPublicKeyToBackendMut<Self> + GetDistributionMut + GLWEInfos,
        O: GLWEPublicKeyCompressedToBackendRef<Self> + GetDistribution + GLWEInfos,
    {
        super::derived::decompression::glwe_public_key_decompress_derived(module, res, other)
    }

    /// Reproduces
    /// [`lwe_decompress_reference`](crate::reference::decompression::DecompressionReference::lwe_decompress_reference)
    /// bit for bit.
    fn lwe_decompress<R, O>(module: &Module<Self>, res: &mut R, other: &O)
    where
        R: LWEToBackendMut<Self> + LWEInfos + SetBase2k,
        O: LWECompressedToBackendRef<Self>;

    /// Default body: `glwe_automorphism_key_decompress_derived`, `gglwe_decompress`, then the source's Galois element
    /// on `res`.
    fn glwe_automorphism_key_decompress<R, O>(module: &Module<Self>, res: &mut R, other: &O)
    where
        R: GGLWEToBackendMut<Self> + GGLWEInfos + SetGaloisElement,
        O: GGLWECompressedToBackendRef<Self> + GGLWEInfos + GetGaloisElement,
    {
        super::derived::decompression::glwe_automorphism_key_decompress_derived(module, res, other)
    }

    /// Default body: `glwe_switching_key_decompress_derived`, `gglwe_decompress`, then the source's input and
    /// output degrees on `res`.
    fn glwe_switching_key_decompress<R, O>(module: &Module<Self>, res: &mut R, other: &O)
    where
        R: GGLWEToBackendMut<Self> + GGLWEInfos + GLWESwitchingKeyDegreesMut,
        O: GGLWECompressedToBackendRef<Self> + GGLWEInfos + GLWESwitchingKeyDegrees,
    {
        super::derived::decompression::glwe_switching_key_decompress_derived(module, res, other)
    }

    /// Default body: `glwe_tensor_key_decompress_derived`, `gglwe_decompress`.
    fn glwe_tensor_key_decompress<R, O>(module: &Module<Self>, res: &mut R, other: &O)
    where
        R: GGLWEToBackendMut<Self> + GGLWEInfos,
        O: GGLWECompressedToBackendRef<Self> + GGLWEInfos,
    {
        super::derived::decompression::glwe_tensor_key_decompress_derived(module, res, other)
    }

    /// Default body: `glwe_to_lwe_key_decompress_derived`, `glwe_switching_key_decompress`.
    fn glwe_to_lwe_key_decompress<R, O>(module: &Module<Self>, res: &mut R, other: &O)
    where
        R: GGLWEToBackendMut<Self> + GGLWEInfos + GLWESwitchingKeyDegreesMut,
        O: GGLWECompressedToBackendRef<Self> + GGLWEInfos + GLWESwitchingKeyDegrees,
    {
        super::derived::decompression::glwe_to_lwe_key_decompress_derived(module, res, other)
    }

    /// Default body: `lwe_switching_key_decompress_derived`, `glwe_switching_key_decompress`.
    fn lwe_switching_key_decompress<R, O>(module: &Module<Self>, res: &mut R, other: &O)
    where
        R: GGLWEToBackendMut<Self> + GGLWEInfos + GLWESwitchingKeyDegreesMut,
        O: GGLWECompressedToBackendRef<Self> + GGLWEInfos + GLWESwitchingKeyDegrees,
    {
        super::derived::decompression::lwe_switching_key_decompress_derived(module, res, other)
    }

    /// Default body: `lwe_to_glwe_key_decompress_derived`, `glwe_switching_key_decompress`.
    fn lwe_to_glwe_key_decompress<R, O>(module: &Module<Self>, res: &mut R, other: &O)
    where
        R: GGLWEToBackendMut<Self> + GGLWEInfos + GLWESwitchingKeyDegreesMut,
        O: GGLWECompressedToBackendRef<Self> + GGLWEInfos + GLWESwitchingKeyDegrees,
    {
        super::derived::decompression::lwe_to_glwe_key_decompress_derived(module, res, other)
    }

    /// Default body: `gglwe_to_ggsw_key_decompress_derived`, after checking the key count and each key's digit
    /// width, row offset and row count, [`Self::gglwe_decompress`] on each key, then the source's noise on `res`.
    fn gglwe_to_ggsw_key_decompress<R, O>(module: &Module<Self>, res: &mut R, other: &O)
    where
        R: GGLWEToGGSWKeyToBackendMut<Self>,
        O: GGLWEToGGSWKeyCompressedToBackendRef<Self>,
    {
        super::derived::decompression::gglwe_to_ggsw_key_decompress_derived(module, res, other)
    }
}

/// Selects the reference bodies of the required decompression methods for this backend; the others keep
/// their derived defaults.
#[macro_export]
macro_rules! impl_decompression_reference_full {
    ($be:ty) => {
        unsafe impl $crate::oep::DecompressionImpl for $be {
            fn glwe_decompress<R, O>(module: &::poulpy_hal::layouts::Module<$be>, res: &mut R, other: &O)
            where
                R: $crate::layouts::GLWEToBackendMut<$be> + $crate::layouts::SetBase2k + $crate::layouts::SetGLWEPlaintextInfos,
                O: $crate::layouts::GLWECompressedToBackendRef<$be> + $crate::layouts::GLWEInfos,
            {
                <::poulpy_hal::layouts::Module<$be> as $crate::reference::decompression::DecompressionReference<$be>>::glwe_decompress_reference(module, res, other)
            }

            fn lwe_decompress<R, O>(module: &::poulpy_hal::layouts::Module<$be>, res: &mut R, other: &O)
            where
                R: $crate::layouts::LWEToBackendMut<$be> + $crate::layouts::LWEInfos + $crate::layouts::SetBase2k,
                O: $crate::layouts::LWECompressedToBackendRef<$be>,
            {
                <::poulpy_hal::layouts::Module<$be> as $crate::reference::decompression::DecompressionReference<$be>>::lwe_decompress_reference(module, res, other)
            }
        }
    };
}
