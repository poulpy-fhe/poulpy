use poulpy_hal::{
    layouts::{Backend, Module, ScratchArena},
    oep::HalVecZnxImpl,
    source::Source,
};

use crate::{
    GetDistribution, GetDistributionMut,
    layouts::{
        Degree, GLWEInfos, GLWESecret, GLWESecretPreparedToBackendMut, GLWESecretTensorPreparedToBackendMut,
        GLWESecretTensorToBackendMut, GLWESecretTensorToBackendRef, GLWESecretToBackendMut, GLWESecretToBackendRef, LWESecret,
        LWESecretToBackendMut, LWESecretToBackendRef, Rank,
    },
    oep::SamplingImpl,
};

/// Backend-provided sampling, preparation, tensoring and LWE/GLWE conversion of secrets.
///
/// The ternary and binary fills default to one
/// [`scalar_znx_fill_distribution`](crate::api::ScalarZnxFillDistribution::scalar_znx_fill_distribution)
/// draw per column through the selected sampler, in column order, then the fill's
/// [`Distribution`](crate::Distribution) tag on the secret, and the zero fills to the selected
/// `vec_znx_zero` on each column, then the `ZERO` tag. The secret and tensor preparations, the
/// tensoring, the LWE/GLWE conversions and their queries are required.
/// An implementation reproduces the body it replaces bit for bit: its refusals before any write,
/// its draws and their order, the tag it records, and for a prepared output a form from which
/// the backend's consumers compute what they compute from the reference's. It chooses only its
/// kernels, its prepared representation and its scratch size, within its own query, which may
/// report less.
///
/// # Safety
/// Implementations read their source operands and write their outputs only within the regions
/// their layouts describe, leave the source operands unchanged, use no more scratch than the
/// method's query reports and wipe that scratch before returning. The memory, layout, synchrony
/// and aliasing rules of `poulpy-hal/docs/backend_safety_contract.md` apply.
pub unsafe trait SecretImpl: Backend + SamplingImpl + HalVecZnxImpl {
    /// Default body: `glwe_secret_fill_ternary_prob_derived`, one
    /// [`SamplingImpl::scalar_znx_fill_distribution`] draw per column, then the tag.
    fn glwe_secret_fill_ternary_prob<S>(module: &Module<Self>, sk: &mut S, prob: f64, source: &mut Source)
    where
        S: GLWESecretToBackendMut<Self> + GetDistributionMut + GLWEInfos,
    {
        super::derived::secret::glwe_secret_fill_ternary_prob_derived(module, sk, prob, source)
    }

    /// Default body: `glwe_secret_fill_ternary_hw_derived`, one
    /// [`SamplingImpl::scalar_znx_fill_distribution`] draw per column, then the tag.
    fn glwe_secret_fill_ternary_hw<S>(module: &Module<Self>, sk: &mut S, hw: usize, source: &mut Source)
    where
        S: GLWESecretToBackendMut<Self> + GetDistributionMut + GLWEInfos,
    {
        super::derived::secret::glwe_secret_fill_ternary_hw_derived(module, sk, hw, source)
    }

    /// Default body: `glwe_secret_fill_binary_prob_derived`, one
    /// [`SamplingImpl::scalar_znx_fill_distribution`] draw per column, then the tag.
    fn glwe_secret_fill_binary_prob<S>(module: &Module<Self>, sk: &mut S, prob: f64, source: &mut Source)
    where
        S: GLWESecretToBackendMut<Self> + GetDistributionMut + GLWEInfos,
    {
        super::derived::secret::glwe_secret_fill_binary_prob_derived(module, sk, prob, source)
    }

    /// Default body: `glwe_secret_fill_binary_hw_derived`, one
    /// [`SamplingImpl::scalar_znx_fill_distribution`] draw per column, then the tag.
    fn glwe_secret_fill_binary_hw<S>(module: &Module<Self>, sk: &mut S, hw: usize, source: &mut Source)
    where
        S: GLWESecretToBackendMut<Self> + GetDistributionMut + GLWEInfos,
    {
        super::derived::secret::glwe_secret_fill_binary_hw_derived(module, sk, hw, source)
    }

    /// Default body: `glwe_secret_fill_binary_block_derived`, one
    /// [`SamplingImpl::scalar_znx_fill_distribution`] draw per column, then the tag.
    fn glwe_secret_fill_binary_block<S>(module: &Module<Self>, sk: &mut S, block_size: usize, source: &mut Source)
    where
        S: GLWESecretToBackendMut<Self> + GetDistributionMut + GLWEInfos,
    {
        super::derived::secret::glwe_secret_fill_binary_block_derived(module, sk, block_size, source)
    }

    /// Default body: `glwe_secret_fill_zero_derived`, [`HalVecZnxImpl::vec_znx_zero`] on each column,
    /// then the `ZERO` tag.
    fn glwe_secret_fill_zero<S>(module: &Module<Self>, sk: &mut S)
    where
        S: GLWESecretToBackendMut<Self> + GetDistributionMut + GLWEInfos,
    {
        super::derived::secret::glwe_secret_fill_zero_derived(module, sk)
    }

    /// Default body: `lwe_secret_fill_ternary_prob_derived`, one
    /// [`SamplingImpl::scalar_znx_fill_distribution`] draw, then the tag.
    fn lwe_secret_fill_ternary_prob<S>(module: &Module<Self>, sk: &mut S, prob: f64, source: &mut Source)
    where
        S: LWESecretToBackendMut<Self> + GetDistributionMut,
    {
        super::derived::secret::lwe_secret_fill_ternary_prob_derived(module, sk, prob, source)
    }

    /// Default body: `lwe_secret_fill_ternary_hw_derived`, one
    /// [`SamplingImpl::scalar_znx_fill_distribution`] draw, then the tag.
    fn lwe_secret_fill_ternary_hw<S>(module: &Module<Self>, sk: &mut S, hw: usize, source: &mut Source)
    where
        S: LWESecretToBackendMut<Self> + GetDistributionMut,
    {
        super::derived::secret::lwe_secret_fill_ternary_hw_derived(module, sk, hw, source)
    }

    /// Default body: `lwe_secret_fill_binary_prob_derived`, one
    /// [`SamplingImpl::scalar_znx_fill_distribution`] draw, then the tag.
    fn lwe_secret_fill_binary_prob<S>(module: &Module<Self>, sk: &mut S, prob: f64, source: &mut Source)
    where
        S: LWESecretToBackendMut<Self> + GetDistributionMut,
    {
        super::derived::secret::lwe_secret_fill_binary_prob_derived(module, sk, prob, source)
    }

    /// Default body: `lwe_secret_fill_binary_hw_derived`, one
    /// [`SamplingImpl::scalar_znx_fill_distribution`] draw, then the tag.
    fn lwe_secret_fill_binary_hw<S>(module: &Module<Self>, sk: &mut S, hw: usize, source: &mut Source)
    where
        S: LWESecretToBackendMut<Self> + GetDistributionMut,
    {
        super::derived::secret::lwe_secret_fill_binary_hw_derived(module, sk, hw, source)
    }

    /// Default body: `lwe_secret_fill_binary_block_derived`, one
    /// [`SamplingImpl::scalar_znx_fill_distribution`] draw, then the tag.
    fn lwe_secret_fill_binary_block<S>(module: &Module<Self>, sk: &mut S, block_size: usize, source: &mut Source)
    where
        S: LWESecretToBackendMut<Self> + GetDistributionMut,
    {
        super::derived::secret::lwe_secret_fill_binary_block_derived(module, sk, block_size, source)
    }

    /// Default body: `lwe_secret_fill_zero_derived`, [`HalVecZnxImpl::vec_znx_zero`] on the column,
    /// then the `ZERO` tag.
    fn lwe_secret_fill_zero<S>(module: &Module<Self>, sk: &mut S)
    where
        S: LWESecretToBackendMut<Self> + GetDistributionMut,
    {
        super::derived::secret::lwe_secret_fill_zero_derived(module, sk)
    }

    /// Reproduces
    /// [`glwe_secret_prepare_reference`](crate::reference::secret::SecretReference::glwe_secret_prepare_reference)
    /// bit for bit; it takes no scratch.
    fn glwe_secret_prepare<R, O>(module: &Module<Self>, res: &mut R, other: &O)
    where
        R: GLWESecretPreparedToBackendMut<Self> + GetDistributionMut,
        O: GLWESecretToBackendRef<Self> + GetDistribution;

    /// Reproduces
    /// [`glwe_secret_tensor_prepared_prepare_reference`](crate::reference::secret::SecretReference::glwe_secret_tensor_prepared_prepare_reference)
    /// bit for bit; it takes no scratch.
    fn glwe_secret_tensor_prepared_prepare<R, O>(module: &Module<Self>, res: &mut R, other: &O)
    where
        R: GLWESecretTensorPreparedToBackendMut<Self> + GetDistributionMut,
        O: GLWESecretTensorToBackendRef<Self> + GetDistribution;

    /// Scratch bytes of [`Self::glwe_secret_tensor_prepare`]: those of
    /// [`glwe_secret_tensor_prepare_tmp_bytes_reference`](crate::reference::secret::SecretReference::glwe_secret_tensor_prepare_tmp_bytes_reference),
    /// or fewer for an override needing fewer.
    fn glwe_secret_tensor_prepare_tmp_bytes(module: &Module<Self>, rank: Rank) -> usize;

    /// Reproduces
    /// [`glwe_secret_tensor_prepare_reference`](crate::reference::secret::SecretReference::glwe_secret_tensor_prepare_reference)
    /// bit for bit within [`Self::glwe_secret_tensor_prepare_tmp_bytes`], wiping that budget before returning.
    fn glwe_secret_tensor_prepare<R, O>(module: &Module<Self>, res: &mut R, other: &O, scratch: &mut ScratchArena<'_, Self>)
    where
        R: GLWESecretTensorToBackendMut<Self> + GetDistributionMut + GLWEInfos,
        O: GLWESecretToBackendRef<Self> + GetDistribution + GLWEInfos;

    /// Reproduces
    /// [`glwe_secret_from_lwe_secret_reference`](crate::reference::secret::SecretReference::glwe_secret_from_lwe_secret_reference)
    /// bit for bit; it takes no scratch.
    fn glwe_secret_from_lwe_secret<S>(module: &Module<Self>, src: &S) -> GLWESecret<Self::OwnedBuf, Self::ZnxWord>
    where
        S: LWESecretToBackendRef<Self>;

    /// Scratch bytes of [`Self::lwe_secret_from_glwe_secret`]: those of
    /// [`lwe_secret_from_glwe_secret_tmp_bytes_reference`](crate::reference::secret::SecretReference::lwe_secret_from_glwe_secret_tmp_bytes_reference),
    /// or fewer for an override needing fewer.
    fn lwe_secret_from_glwe_secret_tmp_bytes<A>(module: &Module<Self>, infos: &A) -> usize
    where
        A: GLWEInfos;

    /// Reproduces
    /// [`lwe_secret_from_glwe_secret_reference`](crate::reference::secret::SecretReference::lwe_secret_from_glwe_secret_reference)
    /// bit for bit within [`Self::lwe_secret_from_glwe_secret_tmp_bytes`], wiping that budget before returning.
    fn lwe_secret_from_glwe_secret<S>(
        module: &Module<Self>,
        src: &S,
        lwe_n: Degree,
        scratch: &mut ScratchArena<'_, Self>,
    ) -> LWESecret<Self::OwnedBuf, Self::ZnxWord>
    where
        S: GLWESecretToBackendRef<Self>;
}

/// Selects the reference bodies of the required secret methods for this backend; the fills
/// keep their derived defaults.
#[macro_export]
macro_rules! impl_secret_reference_full {
    ($be:ty) => {
        unsafe impl $crate::oep::SecretImpl for $be {
            fn glwe_secret_prepare<R, O>(module: &::poulpy_hal::layouts::Module<$be>, res: &mut R, other: &O)
            where
                R: $crate::layouts::GLWESecretPreparedToBackendMut<$be> + $crate::GetDistributionMut,
                O: $crate::layouts::GLWESecretToBackendRef<$be> + $crate::GetDistribution,
            {
                <::poulpy_hal::layouts::Module<$be> as $crate::reference::secret::SecretReference<$be>>::glwe_secret_prepare_reference(module, res, other)
            }

            fn glwe_secret_tensor_prepared_prepare<R, O>(module: &::poulpy_hal::layouts::Module<$be>, res: &mut R, other: &O)
            where
                R: $crate::layouts::GLWESecretTensorPreparedToBackendMut<$be> + $crate::GetDistributionMut,
                O: $crate::layouts::GLWESecretTensorToBackendRef<$be> + $crate::GetDistribution,
            {
                <::poulpy_hal::layouts::Module<$be> as $crate::reference::secret::SecretReference<$be>>::glwe_secret_tensor_prepared_prepare_reference(module, res, other)
            }

            fn glwe_secret_tensor_prepare_tmp_bytes(module: &::poulpy_hal::layouts::Module<$be>, rank: $crate::layouts::Rank) -> usize {
                <::poulpy_hal::layouts::Module<$be> as $crate::reference::secret::SecretReference<$be>>::glwe_secret_tensor_prepare_tmp_bytes_reference(module, rank)
            }

            fn glwe_secret_tensor_prepare<R, O>(
                module: &::poulpy_hal::layouts::Module<$be>,
                res: &mut R,
                other: &O,
                scratch: &mut ::poulpy_hal::layouts::ScratchArena<'_, $be>,
            ) where
                R: $crate::layouts::GLWESecretTensorToBackendMut<$be> + $crate::GetDistributionMut + $crate::layouts::GLWEInfos,
                O: $crate::layouts::GLWESecretToBackendRef<$be> + $crate::GetDistribution + $crate::layouts::GLWEInfos,
            {
                <::poulpy_hal::layouts::Module<$be> as $crate::reference::secret::SecretReference<$be>>::glwe_secret_tensor_prepare_reference(module, res, other, scratch)
            }

            fn glwe_secret_from_lwe_secret<S>(
                module: &::poulpy_hal::layouts::Module<$be>,
                src: &S,
            ) -> $crate::layouts::GLWESecret<<$be as ::poulpy_hal::layouts::Backend>::OwnedBuf, <$be as ::poulpy_hal::layouts::Backend>::ZnxWord>
            where
                S: $crate::layouts::LWESecretToBackendRef<$be>,
            {
                <::poulpy_hal::layouts::Module<$be> as $crate::reference::secret::SecretReference<$be>>::glwe_secret_from_lwe_secret_reference(module, src)
            }

            fn lwe_secret_from_glwe_secret_tmp_bytes<A>(module: &::poulpy_hal::layouts::Module<$be>, infos: &A) -> usize
            where
                A: $crate::layouts::GLWEInfos,
            {
                <::poulpy_hal::layouts::Module<$be> as $crate::reference::secret::SecretReference<$be>>::lwe_secret_from_glwe_secret_tmp_bytes_reference(module, infos)
            }

            fn lwe_secret_from_glwe_secret<S>(
                module: &::poulpy_hal::layouts::Module<$be>,
                src: &S,
                lwe_n: $crate::layouts::Degree,
                scratch: &mut ::poulpy_hal::layouts::ScratchArena<'_, $be>,
            ) -> $crate::layouts::LWESecret<<$be as ::poulpy_hal::layouts::Backend>::OwnedBuf, <$be as ::poulpy_hal::layouts::Backend>::ZnxWord>
            where
                S: $crate::layouts::GLWESecretToBackendRef<$be>,
            {
                <::poulpy_hal::layouts::Module<$be> as $crate::reference::secret::SecretReference<$be>>::lwe_secret_from_glwe_secret_reference(module, src, lwe_n, scratch)
            }
        }
    };
}
