use poulpy_hal::{
    layouts::{Backend, Data, Module, ScratchArena},
    oep::HalVmpImpl,
};

use crate::{
    GetDistribution, GetDistributionMut,
    layouts::{
        GGLWEInfos, GGLWEPreparedToBackendMut, GGLWEToBackendRef, GGLWEToGGSWKeyPreparedToBackendMut, GGLWEToGGSWKeyToBackendRef,
        GGSWInfos, GGSWPreparedToBackendMut, GGSWToBackendRef, GLWECIKey, GLWECIKeyPrepared, GLWEInfos, GLWEPreparedToBackendMut,
        GLWEPublicKeyPreparedToBackendMut, GLWEPublicKeyToBackendRef, GLWESwitchingKey, GLWESwitchingKeyDegrees,
        GLWESwitchingKeyDegreesMut, GLWESwitchingKeyPrepared, GLWEToBackendRef, GetGaloisElement, SetGaloisElement,
    },
};

/// Backend-provided preparation of stored layouts into their prepared forms.
///
/// The GLWE preparation and its scratch query are required. The GGLWE, GGSW, public-key and
/// GGLWE-to-GGSW-key preparations and their queries default to the selected `vmp_prepare` on each
/// matrix, `ggsw_zero` to the selected `vmp_zero`, and the key-wrapper preparations and their
/// queries to compositions of `gglwe_prepare` and `glwe_switching_key_prepare`. An implementation reproduces the body it replaces bit
/// for bit: its refusals before any write to `res`, the metadata it records on `res`, and a prepared form
/// from which the backend's consumers compute what they compute from the reference's. It chooses only its
/// kernels, its prepared representation and its scratch size, within its own query, which may report less.
///
/// # Safety
/// Implementations read `other` and write `res` only within the regions their layouts describe, leave
/// `other` unchanged, and use no more scratch than the method's query reports. The memory, layout,
/// synchrony and aliasing rules of `poulpy-hal/docs/backend_safety_contract.md` apply.
pub unsafe trait PreparationImpl: Backend + HalVmpImpl {
    /// Scratch bytes of [`Self::glwe_prepare`]: those of
    /// [`glwe_prepare_tmp_bytes_reference`](crate::reference::preparation::PreparationReference::glwe_prepare_tmp_bytes_reference),
    /// or fewer for an override needing fewer.
    fn glwe_prepare_tmp_bytes<A>(module: &Module<Self>, infos: &A) -> usize
    where
        A: GLWEInfos;

    /// Reproduces
    /// [`glwe_prepare_reference`](crate::reference::preparation::PreparationReference::glwe_prepare_reference)
    /// bit for bit within [`Self::glwe_prepare_tmp_bytes`].
    fn glwe_prepare<R, O>(module: &Module<Self>, res: &mut R, other: &O, scratch: &mut ScratchArena<'_, Self>)
    where
        R: GLWEPreparedToBackendMut<Self>,
        O: GLWEToBackendRef<Self> + GLWEInfos;

    /// Default body: `gglwe_prepare_tmp_bytes_derived`, [`HalVmpImpl::vmp_prepare_tmp_bytes`] of the matrix.
    fn gglwe_prepare_tmp_bytes<A>(module: &Module<Self>, infos: &A) -> usize
    where
        A: GGLWEInfos,
    {
        super::derived::preparation::gglwe_prepare_tmp_bytes_derived(module, infos)
    }

    /// Default body: `gglwe_prepare_derived`, after checking the degree, `base2k`, limbs, digit width and row
    /// offset of `res` against `other` and the scratch against [`Self::gglwe_prepare_tmp_bytes`],
    /// [`HalVmpImpl::vmp_prepare`] of the matrix, then the source's noise estimate on `res`.
    fn gglwe_prepare<R, O>(module: &Module<Self>, res: &mut R, other: &O, scratch: &mut ScratchArena<'_, Self>)
    where
        R: GGLWEPreparedToBackendMut<Self>,
        O: GGLWEToBackendRef<Self>,
    {
        super::derived::preparation::gglwe_prepare_derived(module, res, other, scratch)
    }

    /// Default body: `ggsw_prepare_tmp_bytes_derived`, [`HalVmpImpl::vmp_prepare_tmp_bytes`] of the matrix.
    fn ggsw_prepare_tmp_bytes<A>(module: &Module<Self>, infos: &A) -> usize
    where
        A: GGSWInfos,
    {
        super::derived::preparation::ggsw_prepare_tmp_bytes_derived(module, infos)
    }

    /// Default body: `ggsw_prepare_derived`, after checking the degree, `base2k` and digit width of `res` against
    /// `other` and the scratch against [`Self::ggsw_prepare_tmp_bytes`], [`HalVmpImpl::vmp_prepare`] of the matrix,
    /// then the source's noise estimate on `res`.
    fn ggsw_prepare<R, O>(module: &Module<Self>, res: &mut R, other: &O, scratch: &mut ScratchArena<'_, Self>)
    where
        R: GGSWPreparedToBackendMut<Self>,
        O: GGSWToBackendRef<Self>,
    {
        super::derived::preparation::ggsw_prepare_derived(module, res, other, scratch)
    }

    /// Default body: `ggsw_zero_derived`, [`HalVmpImpl::vmp_zero`] on `res`, after clearing its noise
    /// metadata; it takes no scratch.
    fn ggsw_zero<R>(module: &Module<Self>, res: &mut R)
    where
        R: GGSWPreparedToBackendMut<Self>,
    {
        super::derived::preparation::ggsw_zero_derived(module, res)
    }

    /// Default body: `glwe_public_key_prepare_tmp_bytes_derived`, [`HalVmpImpl::vmp_prepare_tmp_bytes`] of the
    /// key's entries.
    fn glwe_public_key_prepare_tmp_bytes<A>(module: &Module<Self>, infos: &A) -> usize
    where
        A: GLWEInfos,
    {
        super::derived::preparation::glwe_public_key_prepare_tmp_bytes_derived(module, infos)
    }

    /// Default body: `glwe_public_key_prepare_derived`, after checking the entry count, degree, `base2k`,
    /// precision and limbs of `res` against `other`, [`HalVmpImpl::vmp_prepare`] of the entries within
    /// [`Self::glwe_public_key_prepare_tmp_bytes`], then the source's noise estimate and distribution on `res`.
    fn glwe_public_key_prepare<R, O>(module: &Module<Self>, res: &mut R, other: &O, scratch: &mut ScratchArena<'_, Self>)
    where
        R: GLWEPublicKeyPreparedToBackendMut<Self> + GetDistributionMut,
        O: GLWEPublicKeyToBackendRef<Self> + GetDistribution,
    {
        super::derived::preparation::glwe_public_key_prepare_derived(module, res, other, scratch)
    }

    /// Default body: `glwe_automorphism_key_prepare_tmp_bytes_derived`, [`Self::gglwe_prepare_tmp_bytes`].
    fn glwe_automorphism_key_prepare_tmp_bytes<A>(module: &Module<Self>, infos: &A) -> usize
    where
        A: GGLWEInfos,
    {
        super::derived::preparation::glwe_automorphism_key_prepare_tmp_bytes_derived(module, infos)
    }

    /// Default body: `glwe_automorphism_key_prepare_derived`, [`Self::gglwe_prepare`] after checking the scratch
    /// against [`Self::glwe_automorphism_key_prepare_tmp_bytes`], then the source's Galois element on `res`.
    fn glwe_automorphism_key_prepare<R, O>(module: &Module<Self>, res: &mut R, other: &O, scratch: &mut ScratchArena<'_, Self>)
    where
        R: GGLWEPreparedToBackendMut<Self> + SetGaloisElement,
        O: GGLWEToBackendRef<Self> + GetGaloisElement,
    {
        super::derived::preparation::glwe_automorphism_key_prepare_derived(module, res, other, scratch)
    }

    /// Default body: `glwe_switching_key_prepare_tmp_bytes_derived`, [`Self::gglwe_prepare_tmp_bytes`].
    fn glwe_switching_key_prepare_tmp_bytes<A>(module: &Module<Self>, infos: &A) -> usize
    where
        A: GGLWEInfos,
    {
        super::derived::preparation::glwe_switching_key_prepare_tmp_bytes_derived(module, infos)
    }

    /// Default body: `glwe_switching_key_prepare_derived`, [`Self::gglwe_prepare`] after checking the scratch
    /// against [`Self::glwe_switching_key_prepare_tmp_bytes`], then the source's input and output degrees on `res`.
    fn glwe_switching_key_prepare<R, O>(module: &Module<Self>, res: &mut R, other: &O, scratch: &mut ScratchArena<'_, Self>)
    where
        R: GGLWEPreparedToBackendMut<Self> + GLWESwitchingKeyDegreesMut,
        O: GGLWEToBackendRef<Self> + GLWESwitchingKeyDegrees,
    {
        super::derived::preparation::glwe_switching_key_prepare_derived(module, res, other, scratch)
    }

    /// Default body: `glwe_tensor_key_prepare_tmp_bytes_derived`, [`Self::gglwe_prepare_tmp_bytes`].
    fn glwe_tensor_key_prepare_tmp_bytes<A>(module: &Module<Self>, infos: &A) -> usize
    where
        A: GGLWEInfos,
    {
        super::derived::preparation::glwe_tensor_key_prepare_tmp_bytes_derived(module, infos)
    }

    /// Default body: `glwe_tensor_key_prepare_derived`, [`Self::gglwe_prepare`] after checking the scratch against
    /// [`Self::glwe_tensor_key_prepare_tmp_bytes`].
    fn glwe_tensor_key_prepare<R, O>(module: &Module<Self>, res: &mut R, other: &O, scratch: &mut ScratchArena<'_, Self>)
    where
        R: GGLWEPreparedToBackendMut<Self>,
        O: GGLWEToBackendRef<Self>,
    {
        super::derived::preparation::glwe_tensor_key_prepare_derived(module, res, other, scratch)
    }

    /// Default body: `glwe_to_lwe_key_prepare_tmp_bytes_derived`, [`Self::glwe_switching_key_prepare_tmp_bytes`].
    fn glwe_to_lwe_key_prepare_tmp_bytes<A>(module: &Module<Self>, infos: &A) -> usize
    where
        A: GGLWEInfos,
    {
        super::derived::preparation::glwe_to_lwe_key_prepare_tmp_bytes_derived(module, infos)
    }

    /// Default body: `glwe_to_lwe_key_prepare_derived`, [`Self::glwe_switching_key_prepare`] after checking the
    /// scratch against [`Self::glwe_to_lwe_key_prepare_tmp_bytes`].
    fn glwe_to_lwe_key_prepare<R, O>(module: &Module<Self>, res: &mut R, other: &O, scratch: &mut ScratchArena<'_, Self>)
    where
        R: GGLWEPreparedToBackendMut<Self> + GLWESwitchingKeyDegreesMut,
        O: GGLWEToBackendRef<Self> + GLWESwitchingKeyDegrees,
    {
        super::derived::preparation::glwe_to_lwe_key_prepare_derived(module, res, other, scratch)
    }

    /// Default body: `lwe_switching_key_prepare_tmp_bytes_derived`, [`Self::glwe_switching_key_prepare_tmp_bytes`].
    fn lwe_switching_key_prepare_tmp_bytes<A>(module: &Module<Self>, infos: &A) -> usize
    where
        A: GGLWEInfos,
    {
        super::derived::preparation::lwe_switching_key_prepare_tmp_bytes_derived(module, infos)
    }

    /// Default body: `lwe_switching_key_prepare_derived`, [`Self::glwe_switching_key_prepare`] after checking the
    /// scratch against [`Self::lwe_switching_key_prepare_tmp_bytes`].
    fn lwe_switching_key_prepare<R, O>(module: &Module<Self>, res: &mut R, other: &O, scratch: &mut ScratchArena<'_, Self>)
    where
        R: GGLWEPreparedToBackendMut<Self> + GLWESwitchingKeyDegreesMut,
        O: GGLWEToBackendRef<Self> + GLWESwitchingKeyDegrees,
    {
        super::derived::preparation::lwe_switching_key_prepare_derived(module, res, other, scratch)
    }

    /// Default body: `lwe_to_glwe_key_prepare_tmp_bytes_derived`, [`Self::glwe_switching_key_prepare_tmp_bytes`].
    fn lwe_to_glwe_key_prepare_tmp_bytes<A>(module: &Module<Self>, infos: &A) -> usize
    where
        A: GGLWEInfos,
    {
        super::derived::preparation::lwe_to_glwe_key_prepare_tmp_bytes_derived(module, infos)
    }

    /// Default body: `lwe_to_glwe_key_prepare_derived`, [`Self::glwe_switching_key_prepare`] after checking the
    /// scratch against [`Self::lwe_to_glwe_key_prepare_tmp_bytes`].
    fn lwe_to_glwe_key_prepare<R, O>(module: &Module<Self>, res: &mut R, other: &O, scratch: &mut ScratchArena<'_, Self>)
    where
        R: GGLWEPreparedToBackendMut<Self> + GLWESwitchingKeyDegreesMut,
        O: GGLWEToBackendRef<Self> + GLWESwitchingKeyDegrees,
    {
        super::derived::preparation::lwe_to_glwe_key_prepare_derived(module, res, other, scratch)
    }

    /// Default body: `gglwe_to_ggsw_key_prepare_tmp_bytes_derived`, [`HalVmpImpl::vmp_prepare_tmp_bytes`] of one
    /// key's matrix.
    fn gglwe_to_ggsw_key_prepare_tmp_bytes<A>(module: &Module<Self>, infos: &A) -> usize
    where
        A: GGLWEInfos,
    {
        super::derived::preparation::gglwe_to_ggsw_key_prepare_tmp_bytes_derived(module, infos)
    }

    /// Default body: `gglwe_to_ggsw_key_prepare_derived`, after checking the key count and the scratch against
    /// [`Self::gglwe_to_ggsw_key_prepare_tmp_bytes`], [`HalVmpImpl::vmp_prepare`] of each key's matrix, then the
    /// source's noise estimate on `res`.
    fn gglwe_to_ggsw_key_prepare<R, O>(module: &Module<Self>, res: &mut R, other: &O, scratch: &mut ScratchArena<'_, Self>)
    where
        R: GGLWEToGGSWKeyPreparedToBackendMut<Self>,
        O: GGLWEToGGSWKeyToBackendRef<Self>,
    {
        super::derived::preparation::gglwe_to_ggsw_key_prepare_derived(module, res, other, scratch)
    }

    /// Default body: `glwe_ci_key_prepare_tmp_bytes_derived`, [`Self::glwe_switching_key_prepare_tmp_bytes`].
    fn glwe_ci_key_prepare_tmp_bytes<A>(module: &Module<Self>, infos: &A) -> usize
    where
        A: GGLWEInfos,
    {
        super::derived::preparation::glwe_ci_key_prepare_tmp_bytes_derived(module, infos)
    }

    /// Default body: `glwe_ci_key_prepare_derived`, [`Self::glwe_switching_key_prepare`] on the inner switching key,
    /// within [`Self::glwe_ci_key_prepare_tmp_bytes`].
    fn glwe_ci_key_prepare<DR, DO, M>(
        module: &Module<Self>,
        res: &mut GLWECIKeyPrepared<DR, Self, M>,
        other: &GLWECIKey<DO, Self::ZnxWord, M>,
        scratch: &mut ScratchArena<'_, Self>,
    ) where
        DR: Data,
        DO: Data,
        GLWESwitchingKeyPrepared<DR, Self>: GGLWEPreparedToBackendMut<Self>,
        GLWESwitchingKey<DO, Self::ZnxWord>: GGLWEToBackendRef<Self>,
    {
        super::derived::preparation::glwe_ci_key_prepare_derived(module, res, other, scratch)
    }
}

/// Selects the reference bodies of the required preparation methods for this backend; the others keep
/// their derived defaults.
#[macro_export]
macro_rules! impl_preparation_reference_full {
    ($be:ty) => {
        unsafe impl $crate::oep::PreparationImpl for $be {
            fn glwe_prepare_tmp_bytes<A>(module: &::poulpy_hal::layouts::Module<$be>, infos: &A) -> usize
            where
                A: $crate::layouts::GLWEInfos,
            {
                <::poulpy_hal::layouts::Module<$be> as $crate::reference::preparation::PreparationReference<$be>>::glwe_prepare_tmp_bytes_reference(module, infos)
            }

            fn glwe_prepare<R, O>(
                module: &::poulpy_hal::layouts::Module<$be>,
                res: &mut R,
                other: &O,
                scratch: &mut ::poulpy_hal::layouts::ScratchArena<'_, $be>,
            ) where
                R: $crate::layouts::GLWEPreparedToBackendMut<$be>,
                O: $crate::layouts::GLWEToBackendRef<$be> + $crate::layouts::GLWEInfos,
            {
                <::poulpy_hal::layouts::Module<$be> as $crate::reference::preparation::PreparationReference<$be>>::glwe_prepare_reference(module, res, other, scratch)
            }

        }
    };
}
