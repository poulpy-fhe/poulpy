use poulpy_hal::{
    layouts::{Backend, Data, Module, ScratchArena, Standard},
    oep::HalVecZnxImpl,
    source::Source,
};

use crate::{
    GetDistribution, GetDistributionMut,
    layouts::{
        GGLWEInfos, GGLWEToBackendMut, GLWECIEmbedKey, GLWECITraceKey, GLWEInfos, GLWESecretToBackendMut, GLWESecretToBackendRef,
        GLWEToBackendMut, GLWEToBackendRef,
    },
    oep::EncryptionImpl,
};

/// Backend-provided maps between conjugate-invariant GLWEs and secrets of degree `N` and
/// standard GLWEs and secrets of degree `2N`, the degree switch between standard GLWEs of
/// degrees `N` and `2N`, and the encryption of the switching keys between a standard secret and
/// an embedded conjugate-invariant one, executed by a standard backend of degree at least `2N`.
///
/// The key encryptions default to [`Self::glwe_secret_ci_embed`] into their scratch followed by the
/// selected [`EncryptionImpl::glwe_switching_key_encrypt_sk`], and the degree switch to the selected
/// [`HalVecZnxImpl::vec_znx_switch_ring`] on each column.
///
/// # Safety
/// Implementations reproduce the bodies their methods document bit for bit, report for each
/// `_tmp_bytes` query a bound that covers the body, and refuse as the reference does.
/// The key encryptions wipe the scratch their query reports before returning.
/// The memory, layout, synchrony and aliasing rules of
/// `poulpy-hal/docs/backend_safety_contract.md` apply.
pub unsafe trait GLWECIConversionImpl: Backend<Ring = Standard> + HalVecZnxImpl + EncryptionImpl {
    /// Reproduces [`glwe_ci_embed_reference`](crate::reference::ci_conversion::GLWECIConversionReference::glwe_ci_embed_reference) bit for bit;
    /// free in kernels.
    fn glwe_ci_embed<R, A>(module: &Module<Self>, res: &mut R, a: &A)
    where
        R: GLWEToBackendMut<Self> + GLWEInfos,
        A: GLWEToBackendRef<Self> + GLWEInfos;

    /// Reproduces
    /// [`glwe_secret_ci_embed_reference`](crate::reference::ci_conversion::GLWECIConversionReference::glwe_secret_ci_embed_reference)
    /// bit for bit; it takes no scratch.
    fn glwe_secret_ci_embed<R, S>(module: &Module<Self>, res: &mut R, src: &S)
    where
        R: GLWESecretToBackendMut<Self> + GetDistributionMut,
        S: GLWESecretToBackendRef<Self>;

    /// Scratch bound of [`Self::glwe_ci_trace`]; may report less than the reference, never less than the body takes.
    fn glwe_ci_trace_tmp_bytes<R, A>(module: &Module<Self>, res_infos: &R, a_infos: &A) -> usize
    where
        R: GLWEInfos,
        A: GLWEInfos;

    /// Reproduces [`glwe_ci_trace_reference`](crate::reference::ci_conversion::GLWECIConversionReference::glwe_ci_trace_reference) bit for bit,
    /// within [`Self::glwe_ci_trace_tmp_bytes`]; free in kernels and scratch use.
    fn glwe_ci_trace<R, A>(module: &Module<Self>, res: &mut R, a: &A, scratch: &mut ScratchArena<'_, Self>)
    where
        R: GLWEToBackendMut<Self> + GLWEInfos,
        A: GLWEToBackendRef<Self> + GLWEInfos;

    /// Default body: `glwe_switch_degree_derived`, [`HalVecZnxImpl::vec_znx_switch_ring`] on each column.
    fn glwe_switch_degree<R, A>(module: &Module<Self>, res: &mut R, a: &A)
    where
        R: GLWEToBackendMut<Self> + GLWEInfos,
        A: GLWEToBackendRef<Self> + GLWEInfos,
    {
        crate::oep::derived::ci_conversion::glwe_switch_degree_derived(module, res, a)
    }

    /// Default body: `glwe_ci_key_encrypt_sk_tmp_bytes_derived`, the embedded secret plus
    /// [`EncryptionImpl::glwe_switching_key_encrypt_sk_tmp_bytes`].
    fn glwe_ci_key_encrypt_sk_tmp_bytes<A>(module: &Module<Self>, infos: &A) -> usize
    where
        A: GGLWEInfos,
    {
        crate::oep::derived::ci_conversion::glwe_ci_key_encrypt_sk_tmp_bytes_derived(module, infos)
    }

    /// Default body: `glwe_ci_embed_key_encrypt_sk_derived`, the operand checks, [`Self::glwe_secret_ci_embed`] of
    /// `sk_ci` into the scratch, then [`EncryptionImpl::glwe_switching_key_encrypt_sk`] from the embedded secret
    /// to `sk`, then the wipe of its own query, `glwe_ci_key_encrypt_sk_tmp_bytes_derived`.
    fn glwe_ci_embed_key_encrypt_sk<D, S1, S2>(
        module: &Module<Self>,
        res: &mut GLWECIEmbedKey<D, Self::ZnxWord>,
        sk_ci: &S1,
        sk: &S2,
        source_xe: &mut Source,
        source_xa: &mut Source,
        scratch: &mut ScratchArena<'_, Self>,
    ) where
        D: Data,
        GLWECIEmbedKey<D, Self::ZnxWord>: GGLWEToBackendMut<Self>,
        S1: GLWESecretToBackendRef<Self>,
        S2: GLWESecretToBackendRef<Self> + GetDistribution + GLWEInfos,
    {
        crate::oep::derived::ci_conversion::glwe_ci_embed_key_encrypt_sk_derived(
            module, res, sk_ci, sk, source_xe, source_xa, scratch,
        )
    }

    /// Default body: `glwe_ci_trace_key_encrypt_sk_derived`, the operand checks, [`Self::glwe_secret_ci_embed`] of
    /// `sk_ci` into the scratch, then [`EncryptionImpl::glwe_switching_key_encrypt_sk`] from `sk` to the embedded
    /// secret, then the wipe of its own query, `glwe_ci_key_encrypt_sk_tmp_bytes_derived`.
    fn glwe_ci_trace_key_encrypt_sk<D, S1, S2>(
        module: &Module<Self>,
        res: &mut GLWECITraceKey<D, Self::ZnxWord>,
        sk_ci: &S1,
        sk: &S2,
        source_xe: &mut Source,
        source_xa: &mut Source,
        scratch: &mut ScratchArena<'_, Self>,
    ) where
        D: Data,
        GLWECITraceKey<D, Self::ZnxWord>: GGLWEToBackendMut<Self>,
        S1: GLWESecretToBackendRef<Self> + GetDistribution,
        S2: GLWESecretToBackendRef<Self> + GetDistribution + GLWEInfos,
    {
        crate::oep::derived::ci_conversion::glwe_ci_trace_key_encrypt_sk_derived(
            module, res, sk_ci, sk, source_xe, source_xa, scratch,
        )
    }
}

/// Selects the portable HAL algorithms for the conjugate-invariant maps.
#[macro_export]
macro_rules! impl_glwe_ci_conversion_reference_full {
    ($be:ty) => {
        unsafe impl $crate::oep::GLWECIConversionImpl for $be {
            fn glwe_ci_embed<R, A>(module: &::poulpy_hal::layouts::Module<$be>, res: &mut R, a: &A)
            where
                R: $crate::layouts::GLWEToBackendMut<$be> + $crate::layouts::GLWEInfos,
                A: $crate::layouts::GLWEToBackendRef<$be> + $crate::layouts::GLWEInfos,
            {
                <::poulpy_hal::layouts::Module<$be> as $crate::reference::ci_conversion::GLWECIConversionReference<$be>>::glwe_ci_embed_reference(module, res, a)
            }

            fn glwe_secret_ci_embed<R, S>(module: &::poulpy_hal::layouts::Module<$be>, res: &mut R, src: &S)
            where
                R: $crate::layouts::GLWESecretToBackendMut<$be> + $crate::GetDistributionMut,
                S: $crate::layouts::GLWESecretToBackendRef<$be>,
            {
                <::poulpy_hal::layouts::Module<$be> as $crate::reference::ci_conversion::GLWECIConversionReference<$be>>::glwe_secret_ci_embed_reference(module, res, src)
            }

            fn glwe_ci_trace_tmp_bytes<R, A>(module: &::poulpy_hal::layouts::Module<$be>, res_infos: &R, a_infos: &A) -> usize
            where
                R: $crate::layouts::GLWEInfos,
                A: $crate::layouts::GLWEInfos,
            {
                <::poulpy_hal::layouts::Module<$be> as $crate::reference::ci_conversion::GLWECIConversionReference<$be>>::glwe_ci_trace_tmp_bytes_reference(module, res_infos, a_infos)
            }

            fn glwe_ci_trace<R, A>(
                module: &::poulpy_hal::layouts::Module<$be>,
                res: &mut R,
                a: &A,
                scratch: &mut ::poulpy_hal::layouts::ScratchArena<'_, $be>,
            ) where
                R: $crate::layouts::GLWEToBackendMut<$be> + $crate::layouts::GLWEInfos,
                A: $crate::layouts::GLWEToBackendRef<$be> + $crate::layouts::GLWEInfos,
            {
                <::poulpy_hal::layouts::Module<$be> as $crate::reference::ci_conversion::GLWECIConversionReference<$be>>::glwe_ci_trace_reference(module, res, a, scratch)
            }
        }
    };
}
