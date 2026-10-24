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
/// Each method reproduces its `DecompressionReference` body bit for bit: the copied bodies, the
/// masks regenerated from the stored seeds, its refusals before any write to `res`, and the
/// metadata it sets on `res`. It takes no scratch; the kernels it uses are free.
///
/// # Safety
/// Implementations read `other` and write `res` only within the regions their layouts describe,
/// and leave `other` unchanged. The memory, layout, synchrony and aliasing rules of
/// `poulpy-hal/docs/backend_safety_contract.md` apply.
pub unsafe trait DecompressionImpl: Backend {
    /// Reproduces [`decompress_glwe_reference`](crate::reference::decompression::DecompressionReference::decompress_glwe_reference).
    fn decompress_glwe<R, O>(module: &Module<Self>, res: &mut R, other: &O)
    where
        R: GLWEToBackendMut<Self> + SetBase2k + SetGLWEPlaintextInfos,
        O: GLWECompressedToBackendRef<Self> + GLWEInfos;

    /// Reproduces [`decompress_gglwe_reference`](crate::reference::decompression::DecompressionReference::decompress_gglwe_reference).
    fn decompress_gglwe<R, O>(module: &Module<Self>, res: &mut R, other: &O)
    where
        R: GGLWEToBackendMut<Self> + GGLWEInfos,
        O: GGLWECompressedToBackendRef<Self> + GGLWEInfos;

    /// Reproduces [`decompress_ggsw_reference`](crate::reference::decompression::DecompressionReference::decompress_ggsw_reference).
    fn decompress_ggsw<R, O>(module: &Module<Self>, res: &mut R, other: &O)
    where
        R: GGSWToBackendMut<Self> + GGSWInfos,
        O: GGSWCompressedToBackendRef<Self> + GGSWInfos;

    /// Reproduces [`decompress_glwe_public_key_reference`](crate::reference::decompression::DecompressionReference::decompress_glwe_public_key_reference).
    fn decompress_glwe_public_key<R, O>(module: &Module<Self>, res: &mut R, other: &O)
    where
        R: GLWEPublicKeyToBackendMut<Self> + GetDistributionMut + GLWEInfos,
        O: GLWEPublicKeyCompressedToBackendRef<Self> + GetDistribution + GLWEInfos;

    /// Reproduces [`decompress_lwe_reference`](crate::reference::decompression::DecompressionReference::decompress_lwe_reference).
    fn decompress_lwe<R, O>(module: &Module<Self>, res: &mut R, other: &O)
    where
        R: LWEToBackendMut<Self> + LWEInfos + SetBase2k,
        O: LWECompressedToBackendRef<Self>;

    /// Reproduces [`decompress_automorphism_key_reference`](crate::reference::decompression::DecompressionReference::decompress_automorphism_key_reference).
    fn decompress_automorphism_key<R, O>(module: &Module<Self>, res: &mut R, other: &O)
    where
        R: GGLWEToBackendMut<Self> + GGLWEInfos + SetGaloisElement,
        O: GGLWECompressedToBackendRef<Self> + GGLWEInfos + GetGaloisElement;

    /// Reproduces [`decompress_glwe_switching_key_reference`](crate::reference::decompression::DecompressionReference::decompress_glwe_switching_key_reference).
    fn decompress_glwe_switching_key<R, O>(module: &Module<Self>, res: &mut R, other: &O)
    where
        R: GGLWEToBackendMut<Self> + GGLWEInfos + GLWESwitchingKeyDegreesMut,
        O: GGLWECompressedToBackendRef<Self> + GGLWEInfos + GLWESwitchingKeyDegrees;

    /// Reproduces [`decompress_tensor_key_reference`](crate::reference::decompression::DecompressionReference::decompress_tensor_key_reference).
    fn decompress_tensor_key<R, O>(module: &Module<Self>, res: &mut R, other: &O)
    where
        R: GGLWEToBackendMut<Self> + GGLWEInfos,
        O: GGLWECompressedToBackendRef<Self> + GGLWEInfos;

    /// Reproduces [`decompress_glwe_to_lwe_key_reference`](crate::reference::decompression::DecompressionReference::decompress_glwe_to_lwe_key_reference).
    fn decompress_glwe_to_lwe_key<R, O>(module: &Module<Self>, res: &mut R, other: &O)
    where
        R: GGLWEToBackendMut<Self> + GGLWEInfos + GLWESwitchingKeyDegreesMut,
        O: GGLWECompressedToBackendRef<Self> + GGLWEInfos + GLWESwitchingKeyDegrees;

    /// Reproduces [`decompress_lwe_switching_key_reference`](crate::reference::decompression::DecompressionReference::decompress_lwe_switching_key_reference).
    fn decompress_lwe_switching_key<R, O>(module: &Module<Self>, res: &mut R, other: &O)
    where
        R: GGLWEToBackendMut<Self> + GGLWEInfos + GLWESwitchingKeyDegreesMut,
        O: GGLWECompressedToBackendRef<Self> + GGLWEInfos + GLWESwitchingKeyDegrees;

    /// Reproduces [`decompress_lwe_to_glwe_key_reference`](crate::reference::decompression::DecompressionReference::decompress_lwe_to_glwe_key_reference).
    fn decompress_lwe_to_glwe_key<R, O>(module: &Module<Self>, res: &mut R, other: &O)
    where
        R: GGLWEToBackendMut<Self> + GGLWEInfos + GLWESwitchingKeyDegreesMut,
        O: GGLWECompressedToBackendRef<Self> + GGLWEInfos + GLWESwitchingKeyDegrees;

    /// Reproduces [`decompress_gglwe_to_ggsw_key_reference`](crate::reference::decompression::DecompressionReference::decompress_gglwe_to_ggsw_key_reference).
    fn decompress_gglwe_to_ggsw_key<R, O>(module: &Module<Self>, res: &mut R, other: &O)
    where
        R: GGLWEToGGSWKeyToBackendMut<Self>,
        O: GGLWEToGGSWKeyCompressedToBackendRef<Self>;
}

/// Selects the reference decompression bodies for this backend.
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

            fn decompress_gglwe<R, O>(module: &::poulpy_hal::layouts::Module<$be>, res: &mut R, other: &O)
            where
                R: $crate::layouts::GGLWEToBackendMut<$be> + $crate::layouts::GGLWEInfos,
                O: $crate::layouts::GGLWECompressedToBackendRef<$be> + $crate::layouts::GGLWEInfos,
            {
                <::poulpy_hal::layouts::Module<$be> as $crate::reference::decompression::DecompressionReference<$be>>::decompress_gglwe_reference(module, res, other)
            }

            fn decompress_ggsw<R, O>(module: &::poulpy_hal::layouts::Module<$be>, res: &mut R, other: &O)
            where
                R: $crate::layouts::GGSWToBackendMut<$be> + $crate::layouts::GGSWInfos,
                O: $crate::layouts::GGSWCompressedToBackendRef<$be> + $crate::layouts::GGSWInfos,
            {
                <::poulpy_hal::layouts::Module<$be> as $crate::reference::decompression::DecompressionReference<$be>>::decompress_ggsw_reference(module, res, other)
            }

            fn decompress_glwe_public_key<R, O>(module: &::poulpy_hal::layouts::Module<$be>, res: &mut R, other: &O)
            where
                R: $crate::layouts::GLWEPublicKeyToBackendMut<$be> + $crate::GetDistributionMut + $crate::layouts::GLWEInfos,
                O: $crate::layouts::GLWEPublicKeyCompressedToBackendRef<$be> + $crate::GetDistribution + $crate::layouts::GLWEInfos,
            {
                <::poulpy_hal::layouts::Module<$be> as $crate::reference::decompression::DecompressionReference<$be>>::decompress_glwe_public_key_reference(module, res, other)
            }

            fn decompress_lwe<R, O>(module: &::poulpy_hal::layouts::Module<$be>, res: &mut R, other: &O)
            where
                R: $crate::layouts::LWEToBackendMut<$be> + $crate::layouts::LWEInfos + $crate::layouts::SetBase2k,
                O: $crate::layouts::LWECompressedToBackendRef<$be>,
            {
                <::poulpy_hal::layouts::Module<$be> as $crate::reference::decompression::DecompressionReference<$be>>::decompress_lwe_reference(module, res, other)
            }

            fn decompress_automorphism_key<R, O>(module: &::poulpy_hal::layouts::Module<$be>, res: &mut R, other: &O)
            where
                R: $crate::layouts::GGLWEToBackendMut<$be> + $crate::layouts::GGLWEInfos + $crate::layouts::SetGaloisElement,
                O: $crate::layouts::GGLWECompressedToBackendRef<$be> + $crate::layouts::GGLWEInfos + $crate::layouts::GetGaloisElement,
            {
                <::poulpy_hal::layouts::Module<$be> as $crate::reference::decompression::DecompressionReference<$be>>::decompress_automorphism_key_reference(module, res, other)
            }

            fn decompress_glwe_switching_key<R, O>(module: &::poulpy_hal::layouts::Module<$be>, res: &mut R, other: &O)
            where
                R: $crate::layouts::GGLWEToBackendMut<$be> + $crate::layouts::GGLWEInfos + $crate::layouts::GLWESwitchingKeyDegreesMut,
                O: $crate::layouts::GGLWECompressedToBackendRef<$be> + $crate::layouts::GGLWEInfos + $crate::layouts::GLWESwitchingKeyDegrees,
            {
                <::poulpy_hal::layouts::Module<$be> as $crate::reference::decompression::DecompressionReference<$be>>::decompress_glwe_switching_key_reference(module, res, other)
            }

            fn decompress_tensor_key<R, O>(module: &::poulpy_hal::layouts::Module<$be>, res: &mut R, other: &O)
            where
                R: $crate::layouts::GGLWEToBackendMut<$be> + $crate::layouts::GGLWEInfos,
                O: $crate::layouts::GGLWECompressedToBackendRef<$be> + $crate::layouts::GGLWEInfos,
            {
                <::poulpy_hal::layouts::Module<$be> as $crate::reference::decompression::DecompressionReference<$be>>::decompress_tensor_key_reference(module, res, other)
            }

            fn decompress_glwe_to_lwe_key<R, O>(module: &::poulpy_hal::layouts::Module<$be>, res: &mut R, other: &O)
            where
                R: $crate::layouts::GGLWEToBackendMut<$be> + $crate::layouts::GGLWEInfos + $crate::layouts::GLWESwitchingKeyDegreesMut,
                O: $crate::layouts::GGLWECompressedToBackendRef<$be> + $crate::layouts::GGLWEInfos + $crate::layouts::GLWESwitchingKeyDegrees,
            {
                <::poulpy_hal::layouts::Module<$be> as $crate::reference::decompression::DecompressionReference<$be>>::decompress_glwe_to_lwe_key_reference(module, res, other)
            }

            fn decompress_lwe_switching_key<R, O>(module: &::poulpy_hal::layouts::Module<$be>, res: &mut R, other: &O)
            where
                R: $crate::layouts::GGLWEToBackendMut<$be> + $crate::layouts::GGLWEInfos + $crate::layouts::GLWESwitchingKeyDegreesMut,
                O: $crate::layouts::GGLWECompressedToBackendRef<$be> + $crate::layouts::GGLWEInfos + $crate::layouts::GLWESwitchingKeyDegrees,
            {
                <::poulpy_hal::layouts::Module<$be> as $crate::reference::decompression::DecompressionReference<$be>>::decompress_lwe_switching_key_reference(module, res, other)
            }

            fn decompress_lwe_to_glwe_key<R, O>(module: &::poulpy_hal::layouts::Module<$be>, res: &mut R, other: &O)
            where
                R: $crate::layouts::GGLWEToBackendMut<$be> + $crate::layouts::GGLWEInfos + $crate::layouts::GLWESwitchingKeyDegreesMut,
                O: $crate::layouts::GGLWECompressedToBackendRef<$be> + $crate::layouts::GGLWEInfos + $crate::layouts::GLWESwitchingKeyDegrees,
            {
                <::poulpy_hal::layouts::Module<$be> as $crate::reference::decompression::DecompressionReference<$be>>::decompress_lwe_to_glwe_key_reference(module, res, other)
            }

            fn decompress_gglwe_to_ggsw_key<R, O>(module: &::poulpy_hal::layouts::Module<$be>, res: &mut R, other: &O)
            where
                R: $crate::layouts::GGLWEToGGSWKeyToBackendMut<$be>,
                O: $crate::layouts::GGLWEToGGSWKeyCompressedToBackendRef<$be>,
            {
                <::poulpy_hal::layouts::Module<$be> as $crate::reference::decompression::DecompressionReference<$be>>::decompress_gglwe_to_ggsw_key_reference(module, res, other)
            }
        }
    };
}
