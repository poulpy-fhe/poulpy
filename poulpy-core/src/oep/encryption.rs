#![allow(clippy::too_many_arguments)]

use poulpy_hal::{
    layouts::{Backend, Module, ScalarZnxToBackendRef, ScratchArena, ZnxInfos},
    oep::HalVecZnxImpl,
    source::Source,
};

use crate::{
    GetDistribution, GetDistributionMut,
    layouts::{
        GGLWECompressedSeedMut, GGLWECompressedToBackendMut, GGLWEInfos, GGLWEToBackendMut, GGLWEToGGSWKeyCompressedToBackendMut,
        GGLWEToGGSWKeyToBackendMut, GGSWAtViewMut, GGSWCompressedSeedMut, GGSWCompressedToBackendMut, GGSWInfos,
        GGSWToBackendMut, GLWECompressedSeedMut, GLWECompressedToBackendMut, GLWEInfos, GLWEPublicKeyToBackendMut,
        GLWESecretToBackendRef, GLWESwitchingKeyDegreesMut, GLWEToBackendMut, GLWEToBackendRef, LWEInfos,
        LWEPlaintextToBackendRef, LWESecretToBackendRef, LWEToBackendMut, SetGaloisElement, TorusPrecision,
        compressed::{GLWEPublicKeyCompressedSeedMut, GLWEPublicKeyCompressedToBackendMut},
        prepared::{GLWEPublicKeyPreparedToBackendRef, GLWESecretPreparedToBackendRef},
    },
    oep::{GLWENormalizeImpl, SamplingImpl, SecretImpl},
};

/// Backend-provided encryption, key generation and mask sampling.
///
/// Methods that sample draw through [`SamplingImpl`], whose streams are the backend's own, so a body is
/// reproduced given the same sampled values.
///
/// # Safety
/// Implementations reproduce the bodies their methods document bit for bit, report for each
/// `_tmp_bytes` query a bound that covers the body, and refuse as the reference does.
/// Every method taking scratch wipes the scratch its query reports before returning.
/// The memory, layout, synchrony and aliasing rules of
/// `poulpy-hal/docs/backend_safety_contract.md` apply.
pub unsafe trait EncryptionImpl: Backend + SamplingImpl + HalVecZnxImpl + GLWENormalizeImpl + SecretImpl {
    /// Reproduces
    /// [`fill_glwe_mask_from_source_reference`](crate::reference::encryption::GLWEMaskFillReference::fill_glwe_mask_from_source_reference)
    /// bit for bit; free in kernels.
    fn fill_glwe_mask_from_source<R>(module: &Module<Self>, res: &mut R, source_xa: &mut Source)
    where
        R: GLWEToBackendMut<Self>;

    /// Default body: `fill_glwe_mask_from_seed_derived`, [`Self::fill_glwe_mask_from_source`] on a `Source` built
    /// from the seed.
    fn fill_glwe_mask_from_seed<R>(module: &Module<Self>, res: &mut R, seed_xa: [u8; 32])
    where
        R: GLWEToBackendMut<Self>,
    {
        super::derived::encryption::fill_glwe_mask_from_seed_derived(module, res, seed_xa)
    }

    /// Reproduces
    /// [`fill_glwe_from_source_reference`](crate::reference::encryption::GLWEMaskFillReference::fill_glwe_from_source_reference)
    /// bit for bit; free in kernels.
    fn fill_glwe_from_source<R>(module: &Module<Self>, res: &mut R, source: &mut Source)
    where
        R: GLWEToBackendMut<Self>;

    /// Reproduces
    /// [`fill_lwe_mask_from_source_reference`](crate::reference::encryption::LWEFillMaskReference::fill_lwe_mask_from_source_reference)
    /// bit for bit; free in kernels.
    fn fill_lwe_mask_from_source<R>(module: &Module<Self>, base2k: usize, res: &mut R, source_xa: &mut Source)
    where
        R: LWEToBackendMut<Self>;

    /// Default body: `fill_lwe_mask_from_seed_derived`, [`Self::fill_lwe_mask_from_source`] on a `Source` built from
    /// the seed.
    fn fill_lwe_mask_from_seed<R>(module: &Module<Self>, base2k: usize, res: &mut R, seed_xa: [u8; 32])
    where
        R: LWEToBackendMut<Self>,
    {
        super::derived::encryption::fill_lwe_mask_from_seed_derived(module, base2k, res, seed_xa)
    }

    /// Scratch bound of [`Self::lwe_encrypt_sk`]; may report less than the reference, never less than the body takes.
    fn lwe_encrypt_sk_tmp_bytes<A>(module: &Module<Self>, infos: &A) -> usize
    where
        A: LWEInfos;

    /// Reproduces
    /// [`lwe_encrypt_sk_reference`](crate::reference::encryption::LWEEncryptSkReference::lwe_encrypt_sk_reference)
    /// bit for bit, within [`Self::lwe_encrypt_sk_tmp_bytes`]; free in kernels and scratch use before the wipe.
    fn lwe_encrypt_sk<R, P, S>(
        module: &Module<Self>,
        res: &mut R,
        pt: &P,
        sk: &S,
        source_xe: &mut Source,
        source_xa: &mut Source,
        scratch: &mut ScratchArena<'_, Self>,
    ) where
        R: LWEToBackendMut<Self> + LWEInfos,
        P: LWEPlaintextToBackendRef<Self>,
        S: LWESecretToBackendRef<Self>;

    /// Scratch bound of [`Self::glwe_encrypt_sk`], [`Self::glwe_encrypt_zero_sk`] and
    /// [`Self::glwe_encrypt_sk_with_mask`]; may report less than the reference, never less than the body takes.
    fn glwe_encrypt_sk_tmp_bytes<A>(module: &Module<Self>, infos: &A) -> usize
    where
        A: GLWEInfos;

    /// Reproduces
    /// [`glwe_encrypt_sk_reference`](crate::reference::encryption::GLWEEncryptSkReference::glwe_encrypt_sk_reference)
    /// bit for bit under any [`PrepareHint`](poulpy_hal::layouts::PrepareHint), within
    /// [`Self::glwe_encrypt_sk_tmp_bytes`]; free in kernels and scratch use before the wipe.
    fn glwe_encrypt_sk<R, P, S>(
        module: &Module<Self>,
        res: &mut R,
        pt: &P,
        sk: &S,
        source_xe: &mut Source,
        source_xa: &mut Source,
        scratch: &mut ScratchArena<'_, Self>,
    ) where
        R: GLWEToBackendMut<Self>,
        P: GLWEToBackendRef<Self>,
        S: GLWESecretPreparedToBackendRef<Self>;

    /// Reproduces
    /// [`glwe_encrypt_zero_sk_reference`](crate::reference::encryption::GLWEEncryptSkReference::glwe_encrypt_zero_sk_reference)
    /// bit for bit under any [`PrepareHint`](poulpy_hal::layouts::PrepareHint), within
    /// [`Self::glwe_encrypt_sk_tmp_bytes`]; free in kernels and scratch use before the wipe.
    fn glwe_encrypt_zero_sk<R, S>(
        module: &Module<Self>,
        res: &mut R,
        sk: &S,
        source_xe: &mut Source,
        source_xa: &mut Source,
        scratch: &mut ScratchArena<'_, Self>,
    ) where
        R: GLWEToBackendMut<Self>,
        S: GLWESecretPreparedToBackendRef<Self>;

    /// Reproduces
    /// [`glwe_encrypt_sk_with_mask_reference`](crate::reference::encryption::GLWEEncryptSkReference::glwe_encrypt_sk_with_mask_reference)
    /// bit for bit under any [`PrepareHint`](poulpy_hal::layouts::PrepareHint), within
    /// [`Self::glwe_encrypt_sk_tmp_bytes`]; free in kernels and scratch use before the wipe.
    fn glwe_encrypt_sk_with_mask<R, P, S>(
        module: &Module<Self>,
        res: &mut R,
        pt: &P,
        sk: &S,
        source_xe: &mut Source,
        scratch: &mut ScratchArena<'_, Self>,
    ) where
        R: GLWEToBackendMut<Self>,
        P: GLWEToBackendRef<Self>,
        S: GLWESecretPreparedToBackendRef<Self>;

    /// Scratch bound of [`Self::glwe_encrypt_pk`], [`Self::glwe_encrypt_zero_pk`], [`Self::glwe_encrypt_pk_at_col`]
    /// and [`Self::glwe_encrypt_pk_sampled_at`]; may report less than the reference, never less than the body takes.
    fn glwe_encrypt_pk_tmp_bytes<R, K>(module: &Module<Self>, res_infos: &R, pk_infos: &K) -> usize
    where
        R: GLWEInfos,
        K: GLWEInfos;

    /// Default body: `glwe_encrypt_pk_derived`, `glwe_encrypt_pk_at_col_derived` with `pt` added to the body:
    /// [`Self::glwe_encrypt_pk_sampled_at`] at the precision the noise model selects, then its noise estimate on
    /// `res`, within [`Self::glwe_encrypt_pk_tmp_bytes`], under any [`PrepareHint`](poulpy_hal::layouts::PrepareHint)
    /// of its prepared operands.
    fn glwe_encrypt_pk<R, P, K>(
        module: &Module<Self>,
        res: &mut R,
        pt: &P,
        pk: &K,
        source_xu: &mut Source,
        source_xe: &mut Source,
        scratch: &mut ScratchArena<'_, Self>,
    ) where
        R: GLWEToBackendMut<Self> + GLWEInfos,
        P: GLWEToBackendRef<Self> + GLWEInfos,
        K: GLWEPublicKeyPreparedToBackendRef<Self> + GLWEInfos,
    {
        super::derived::encryption::glwe_encrypt_pk_derived(module, res, pt, pk, source_xu, source_xe, scratch)
    }

    /// Default body: `glwe_encrypt_zero_pk_derived`, as [`Self::glwe_encrypt_pk`] with no plaintext, within
    /// [`Self::glwe_encrypt_pk_tmp_bytes`], under any [`PrepareHint`](poulpy_hal::layouts::PrepareHint) of its
    /// prepared operands.
    fn glwe_encrypt_zero_pk<R, K>(
        module: &Module<Self>,
        res: &mut R,
        pk: &K,
        source_xu: &mut Source,
        source_xe: &mut Source,
        scratch: &mut ScratchArena<'_, Self>,
    ) where
        R: GLWEToBackendMut<Self> + GLWEInfos,
        K: GLWEPublicKeyPreparedToBackendRef<Self> + GLWEInfos,
    {
        super::derived::encryption::glwe_encrypt_zero_pk_derived(module, res, pk, source_xu, source_xe, scratch)
    }

    /// Default body: `glwe_encrypt_pk_at_col_derived`, [`Self::glwe_encrypt_pk_sampled_at`] with `pt` added to column
    /// `col`, drawn at the precision the noise model selects, then its noise estimate on `res`, within
    /// [`Self::glwe_encrypt_pk_tmp_bytes`], under any [`PrepareHint`](poulpy_hal::layouts::PrepareHint) of its
    /// prepared operands.
    fn glwe_encrypt_pk_at_col<R, P, K>(
        module: &Module<Self>,
        res: &mut R,
        pt: &P,
        col: usize,
        pk: &K,
        source_xu: &mut Source,
        source_xe: &mut Source,
        scratch: &mut ScratchArena<'_, Self>,
    ) where
        R: GLWEToBackendMut<Self> + GLWEInfos,
        P: GLWEToBackendRef<Self> + GLWEInfos,
        K: GLWEPublicKeyPreparedToBackendRef<Self> + GLWEInfos,
    {
        super::derived::encryption::glwe_encrypt_pk_at_col_derived(
            module,
            res,
            Some((pt, col)),
            pk,
            source_xu,
            source_xe,
            scratch,
        )
    }

    /// Reproduces
    /// [`glwe_encrypt_pk_sampled_at_reference`](crate::reference::encryption::GLWEEncryptPkReference::glwe_encrypt_pk_sampled_at_reference)
    /// bit for bit under any [`PrepareHint`](poulpy_hal::layouts::PrepareHint), within
    /// [`Self::glwe_encrypt_pk_tmp_bytes`]; free in kernels and scratch use before the wipe.
    ///
    /// The caller hands it `pt.0` to add to column `pt.1` (`0` is the body) or `None` for zero, and `k_sample` in
    /// `[res.k(), pk.k()]`, the precision of the fresh errors. Without `body_noise` the body gets no encryption
    /// error: the caller floods it, see [`Self::glwe_encrypt_pk_smudged`]. The product uses the key's leading
    /// `ceil(k_sample / base2k)` limbs and each column is normalized once at `res.k()`. The caller records the output
    /// metadata.
    ///
    /// # Panics
    ///
    /// On invalid keys, plaintexts, columns or `k_sample`, before the output changes.
    fn glwe_encrypt_pk_sampled_at<R, P, K>(
        module: &Module<Self>,
        res: &mut R,
        pt: Option<(&P, usize)>,
        body_noise: bool,
        pk: &K,
        k_sample: TorusPrecision,
        source_xu: &mut Source,
        source_xe: &mut Source,
        scratch: &mut ScratchArena<'_, Self>,
    ) where
        R: GLWEToBackendMut<Self> + GLWEInfos,
        P: GLWEToBackendRef<Self> + GLWEInfos,
        K: GLWEPublicKeyPreparedToBackendRef<Self> + GLWEInfos;

    /// Default body: `glwe_encrypt_pk_smudged_tmp_bytes_derived`, the larger of [`Self::glwe_encrypt_pk_tmp_bytes`]
    /// and `vec_znx_normalize_tmp_bytes`.
    fn glwe_encrypt_pk_smudged_tmp_bytes<R, K>(module: &Module<Self>, res_infos: &R, pk_infos: &K) -> usize
    where
        R: GLWEInfos,
        K: GLWEInfos,
    {
        super::derived::encryption::glwe_encrypt_pk_smudged_tmp_bytes_derived(module, res_infos, pk_infos)
    }

    /// Default body: `glwe_encrypt_pk_smudged_derived`, [`Self::glwe_encrypt_pk_sampled_at`] of `pt` into the body
    /// without its error, then `flood` on the body at the output's precision, drawn from `source_smudge`, normalized
    /// once more, within [`Self::glwe_encrypt_pk_smudged_tmp_bytes`], under any
    /// [`PrepareHint`](poulpy_hal::layouts::PrepareHint) of its prepared operands.
    fn glwe_encrypt_pk_smudged<R, P, K>(
        module: &Module<Self>,
        res: &mut R,
        pt: &P,
        pk: &K,
        flood: crate::Noise,
        source_xu: &mut Source,
        source_xe: &mut Source,
        source_smudge: &mut Source,
        scratch: &mut ScratchArena<'_, Self>,
    ) where
        R: GLWEToBackendMut<Self> + GLWEInfos,
        P: GLWEToBackendRef<Self> + GLWEInfos,
        K: GLWEPublicKeyPreparedToBackendRef<Self> + GLWEInfos,
    {
        super::derived::encryption::glwe_encrypt_pk_smudged_derived(
            module,
            res,
            pt,
            pk,
            flood,
            source_xu,
            source_xe,
            source_smudge,
            scratch,
        )
    }

    /// Default body: `glwe_public_key_generate_tmp_bytes_derived`, the larger of [`Self::glwe_encrypt_sk_tmp_bytes`]
    /// and [`GLWENormalizeImpl::glwe_normalize_tmp_bytes`](crate::oep::GLWENormalizeImpl::glwe_normalize_tmp_bytes).
    fn glwe_public_key_generate_tmp_bytes<A>(module: &Module<Self>, infos: &A) -> usize
    where
        A: GLWEInfos,
    {
        super::derived::encryption::glwe_public_key_generate_tmp_bytes_derived(module, infos)
    }

    /// Default body: `glwe_public_key_generate_derived`, [`Self::glwe_encrypt_zero_sk`] then
    /// [`GLWENormalizeImpl::glwe_normalize_assign`](crate::oep::GLWENormalizeImpl::glwe_normalize_assign) on each of
    /// the `rank` entries, then the secret's noise estimate and distribution on `res`, within
    /// [`Self::glwe_public_key_generate_tmp_bytes`], under any [`PrepareHint`](poulpy_hal::layouts::PrepareHint) of
    /// its prepared operands.
    fn glwe_public_key_generate<R, S>(
        module: &Module<Self>,
        res: &mut R,
        sk: &S,
        source_xe: &mut Source,
        source_xa: &mut Source,
        scratch: &mut ScratchArena<'_, Self>,
    ) where
        R: GLWEPublicKeyToBackendMut<Self> + GetDistributionMut + GLWEInfos,
        S: GLWESecretPreparedToBackendRef<Self> + GetDistribution,
    {
        super::derived::encryption::glwe_public_key_generate_derived(module, res, sk, source_xe, source_xa, scratch)
    }

    /// Default body: `glwe_public_key_compressed_generate_tmp_bytes_derived`,
    /// [`Self::glwe_compressed_encrypt_sk_tmp_bytes`].
    fn glwe_public_key_compressed_generate_tmp_bytes<A>(module: &Module<Self>, infos: &A) -> usize
    where
        A: GLWEInfos,
    {
        super::derived::encryption::glwe_public_key_compressed_generate_tmp_bytes_derived(module, infos)
    }

    /// Default body: `glwe_public_key_compressed_generate_derived`, [`Self::glwe_compressed_encrypt_zero_sk`] on each
    /// of the `rank` entries with its own derived seed, within
    /// [`Self::glwe_public_key_compressed_generate_tmp_bytes`], under any
    /// [`PrepareHint`](poulpy_hal::layouts::PrepareHint) of its prepared operands.
    fn glwe_public_key_compressed_generate<R, S>(
        module: &Module<Self>,
        res: &mut R,
        sk: &S,
        seed: [u8; 32],
        source_xe: &mut Source,
        scratch: &mut ScratchArena<'_, Self>,
    ) where
        R: GLWEPublicKeyCompressedToBackendMut<Self> + GLWEPublicKeyCompressedSeedMut + GetDistributionMut + GLWEInfos,
        S: GLWESecretPreparedToBackendRef<Self> + GetDistribution,
    {
        super::derived::encryption::glwe_public_key_compressed_generate_derived(module, res, sk, seed, source_xe, scratch)
    }

    /// Scratch bound of [`Self::gglwe_encrypt_sk`]; may report less than the reference, never less than the body
    /// takes.
    fn gglwe_encrypt_sk_tmp_bytes<A>(module: &Module<Self>, infos: &A) -> usize
    where
        A: GGLWEInfos;

    /// Reproduces
    /// [`gglwe_encrypt_sk_reference`](crate::reference::encryption::GGLWEEncryptSkReference::gglwe_encrypt_sk_reference)
    /// bit for bit under any [`PrepareHint`](poulpy_hal::layouts::PrepareHint), within
    /// [`Self::gglwe_encrypt_sk_tmp_bytes`]; free in kernels and scratch use before the wipe.
    fn gglwe_encrypt_sk<R, P, S>(
        module: &Module<Self>,
        res: &mut R,
        pt: &P,
        sk: &S,
        source_xe: &mut Source,
        source_xa: &mut Source,
        scratch: &mut ScratchArena<'_, Self>,
    ) where
        R: GGLWEToBackendMut<Self>,
        P: ScalarZnxToBackendRef<Self>,
        S: GLWESecretPreparedToBackendRef<Self>;

    /// Scratch bound of [`Self::ggsw_encrypt_sk`]; may report less than the reference, never less than the body
    /// takes.
    fn ggsw_encrypt_sk_tmp_bytes<A>(module: &Module<Self>, infos: &A) -> usize
    where
        A: GGSWInfos;

    /// Reproduces
    /// [`ggsw_encrypt_sk_reference`](crate::reference::encryption::GGSWEncryptSkReference::ggsw_encrypt_sk_reference)
    /// bit for bit under any [`PrepareHint`](poulpy_hal::layouts::PrepareHint), within
    /// [`Self::ggsw_encrypt_sk_tmp_bytes`]; free in kernels and scratch use before the wipe.
    fn ggsw_encrypt_sk<R, P, S>(
        module: &Module<Self>,
        res: &mut R,
        pt: &P,
        sk: &S,
        source_xe: &mut Source,
        source_xa: &mut Source,
        scratch: &mut ScratchArena<'_, Self>,
    ) where
        R: GGSWToBackendMut<Self> + GGSWInfos + GGSWAtViewMut<Self>,
        P: ScalarZnxToBackendRef<Self> + ZnxInfos,
        S: GLWESecretPreparedToBackendRef<Self> + LWEInfos + GLWEInfos;

    /// Default body: `ggsw_encrypt_pk_tmp_bytes_derived`, a GLWE plaintext temporary plus the larger of
    /// [`Self::glwe_encrypt_pk_tmp_bytes`] and `vec_znx_normalize_tmp_bytes`.
    fn ggsw_encrypt_pk_tmp_bytes<R, K>(module: &Module<Self>, res_infos: &R, pk_infos: &K) -> usize
    where
        R: GGSWInfos,
        K: GLWEInfos,
    {
        super::derived::encryption::ggsw_encrypt_pk_tmp_bytes_derived(module, res_infos, pk_infos)
    }

    /// Default body: `ggsw_encrypt_pk_derived`, for each gadget row, the row's plaintext into a temporary, then
    /// [`Self::glwe_encrypt_pk_sampled_at`] on each column, then the noise estimate on `res`, within
    /// [`Self::ggsw_encrypt_pk_tmp_bytes`], under any [`PrepareHint`](poulpy_hal::layouts::PrepareHint) of its
    /// prepared operands.
    fn ggsw_encrypt_pk<R, P, K>(
        module: &Module<Self>,
        res: &mut R,
        pt: &P,
        pk: &K,
        source_xu: &mut Source,
        source_xe: &mut Source,
        scratch: &mut ScratchArena<'_, Self>,
    ) where
        R: GGSWToBackendMut<Self> + GGSWInfos + GGSWAtViewMut<Self>,
        P: ScalarZnxToBackendRef<Self> + ZnxInfos,
        K: GLWEPublicKeyPreparedToBackendRef<Self> + GLWEInfos,
    {
        super::derived::encryption::ggsw_encrypt_pk_derived(module, res, pt, pk, source_xu, source_xe, scratch)
    }

    /// Scratch bound of [`Self::gglwe_to_ggsw_key_encrypt_sk`]; may report less than the reference, never less than
    /// the body takes.
    fn gglwe_to_ggsw_key_encrypt_sk_tmp_bytes<A>(module: &Module<Self>, infos: &A) -> usize
    where
        A: GGLWEInfos;

    /// Reproduces
    /// [`gglwe_to_ggsw_key_encrypt_sk_reference`](crate::reference::encryption::GGLWEToGGSWKeyEncryptSkReference::gglwe_to_ggsw_key_encrypt_sk_reference)
    /// bit for bit, within [`Self::gglwe_to_ggsw_key_encrypt_sk_tmp_bytes`]; free in kernels and scratch use before the wipe.
    fn gglwe_to_ggsw_key_encrypt_sk<R, S>(
        module: &Module<Self>,
        res: &mut R,
        sk: &S,
        source_xe: &mut Source,
        source_xa: &mut Source,
        scratch: &mut ScratchArena<'_, Self>,
    ) where
        R: GGLWEToGGSWKeyToBackendMut<Self>,
        S: GLWESecretToBackendRef<Self> + GetDistribution + GLWEInfos;

    /// Scratch bound of [`Self::glwe_switching_key_encrypt_sk`]; may report less than the reference, never less than
    /// the body takes.
    fn glwe_switching_key_encrypt_sk_tmp_bytes<A>(module: &Module<Self>, infos: &A) -> usize
    where
        A: GGLWEInfos;

    /// Reproduces
    /// [`glwe_switching_key_encrypt_sk_reference`](crate::reference::encryption::GLWESwitchingKeyEncryptSkReference::glwe_switching_key_encrypt_sk_reference)
    /// bit for bit, within [`Self::glwe_switching_key_encrypt_sk_tmp_bytes`]; free in kernels and scratch use before the wipe.
    fn glwe_switching_key_encrypt_sk<R, S1, S2>(
        module: &Module<Self>,
        res: &mut R,
        sk_in: &S1,
        sk_out: &S2,
        source_xe: &mut Source,
        source_xa: &mut Source,
        scratch: &mut ScratchArena<'_, Self>,
    ) where
        R: GGLWEToBackendMut<Self> + GLWESwitchingKeyDegreesMut + GGLWEInfos,
        S1: GLWESecretToBackendRef<Self> + GLWEInfos,
        S2: GLWESecretToBackendRef<Self> + GetDistribution + GLWEInfos;

    /// Default body: `glwe_tensor_key_encrypt_sk_tmp_bytes_derived`, the prepared secret and its tensor plus the
    /// larger of the tensor preparation and [`Self::gglwe_encrypt_sk_tmp_bytes`].
    fn glwe_tensor_key_encrypt_sk_tmp_bytes<A>(module: &Module<Self>, infos: &A) -> usize
    where
        A: GGLWEInfos,
    {
        super::derived::encryption::glwe_tensor_key_encrypt_sk_tmp_bytes_derived(module, infos)
    }

    /// Default body: `glwe_tensor_key_encrypt_sk_derived`, the secret prepared by
    /// [`SecretImpl::glwe_secret_prepare`](crate::oep::SecretImpl::glwe_secret_prepare) and its tensor by
    /// [`SecretImpl::glwe_secret_tensor_prepare`](crate::oep::SecretImpl::glwe_secret_tensor_prepare), then
    /// [`Self::gglwe_encrypt_sk`] of the tensor under the prepared secret, within
    /// [`Self::glwe_tensor_key_encrypt_sk_tmp_bytes`].
    fn glwe_tensor_key_encrypt_sk<R, S>(
        module: &Module<Self>,
        res: &mut R,
        sk: &S,
        source_xe: &mut Source,
        source_xa: &mut Source,
        scratch: &mut ScratchArena<'_, Self>,
    ) where
        R: GGLWEToBackendMut<Self> + GGLWEInfos,
        S: GLWESecretToBackendRef<Self> + GetDistribution + GLWEInfos,
    {
        super::derived::encryption::glwe_tensor_key_encrypt_sk_derived(module, res, sk, source_xe, source_xa, scratch)
    }

    /// Scratch bound of [`Self::glwe_to_lwe_key_encrypt_sk`]; may report less than the reference, never less than the
    /// body takes.
    fn glwe_to_lwe_key_encrypt_sk_tmp_bytes<A>(module: &Module<Self>, infos: &A) -> usize
    where
        A: GGLWEInfos;

    /// Reproduces
    /// [`glwe_to_lwe_key_encrypt_sk_reference`](crate::reference::encryption::GLWEToLWESwitchingKeyEncryptSkReference::glwe_to_lwe_key_encrypt_sk_reference)
    /// bit for bit, within [`Self::glwe_to_lwe_key_encrypt_sk_tmp_bytes`]; free in kernels and scratch use before the wipe.
    fn glwe_to_lwe_key_encrypt_sk<R, S1, S2>(
        module: &Module<Self>,
        res: &mut R,
        sk_lwe: &S1,
        sk_glwe: &S2,
        source_xe: &mut Source,
        source_xa: &mut Source,
        scratch: &mut ScratchArena<'_, Self>,
    ) where
        S1: LWESecretToBackendRef<Self>,
        S2: GLWESecretToBackendRef<Self>,
        R: GGLWEToBackendMut<Self> + GGLWEInfos;

    /// Scratch bound of [`Self::lwe_switching_key_encrypt_sk`]; may report less than the reference, never less than
    /// the body takes.
    fn lwe_switching_key_encrypt_sk_tmp_bytes<A>(module: &Module<Self>, infos: &A) -> usize
    where
        A: GGLWEInfos;

    /// Reproduces
    /// [`lwe_switching_key_encrypt_sk_reference`](crate::reference::encryption::LWESwitchingKeyEncryptReference::lwe_switching_key_encrypt_sk_reference)
    /// bit for bit, within [`Self::lwe_switching_key_encrypt_sk_tmp_bytes`]; free in kernels and scratch use before the wipe.
    fn lwe_switching_key_encrypt_sk<R, S1, S2>(
        module: &Module<Self>,
        res: &mut R,
        sk_lwe_in: &S1,
        sk_lwe_out: &S2,
        source_xe: &mut Source,
        source_xa: &mut Source,
        scratch: &mut ScratchArena<'_, Self>,
    ) where
        R: GGLWEToBackendMut<Self> + GLWESwitchingKeyDegreesMut + GGLWEInfos,
        S1: LWESecretToBackendRef<Self>,
        S2: LWESecretToBackendRef<Self>;

    /// Scratch bound of [`Self::lwe_to_glwe_key_encrypt_sk`]; may report less than the reference, never less than the
    /// body takes.
    fn lwe_to_glwe_key_encrypt_sk_tmp_bytes<A>(module: &Module<Self>, infos: &A) -> usize
    where
        A: GGLWEInfos;

    /// Reproduces
    /// [`lwe_to_glwe_key_encrypt_sk_reference`](crate::reference::encryption::LWEToGLWESwitchingKeyEncryptSkReference::lwe_to_glwe_key_encrypt_sk_reference)
    /// bit for bit under any [`PrepareHint`](poulpy_hal::layouts::PrepareHint), within
    /// [`Self::lwe_to_glwe_key_encrypt_sk_tmp_bytes`]; free in kernels and scratch use before the wipe.
    fn lwe_to_glwe_key_encrypt_sk<R, S1, S2>(
        module: &Module<Self>,
        res: &mut R,
        sk_lwe: &S1,
        sk_glwe: &S2,
        source_xe: &mut Source,
        source_xa: &mut Source,
        scratch: &mut ScratchArena<'_, Self>,
    ) where
        S1: LWESecretToBackendRef<Self>,
        S2: GLWESecretPreparedToBackendRef<Self>,
        R: GGLWEToBackendMut<Self> + GGLWEInfos;

    /// Scratch bound of [`Self::glwe_automorphism_key_encrypt_sk`]; may report less than the reference, never less
    /// than the body takes.
    fn glwe_automorphism_key_encrypt_sk_tmp_bytes<A>(module: &Module<Self>, infos: &A) -> usize
    where
        A: GGLWEInfos;

    /// Reproduces
    /// [`glwe_automorphism_key_encrypt_sk_reference`](crate::reference::encryption::GLWEAutomorphismKeyEncryptSkReference::glwe_automorphism_key_encrypt_sk_reference)
    /// bit for bit, within [`Self::glwe_automorphism_key_encrypt_sk_tmp_bytes`]; free in kernels and scratch use before the wipe.
    fn glwe_automorphism_key_encrypt_sk<R, S>(
        module: &Module<Self>,
        res: &mut R,
        p: i64,
        sk: &S,
        source_xe: &mut Source,
        source_xa: &mut Source,
        scratch: &mut ScratchArena<'_, Self>,
    ) where
        R: GGLWEToBackendMut<Self> + SetGaloisElement + GGLWEInfos,
        S: GLWESecretToBackendRef<Self> + GLWEInfos;

    /// Scratch bound of [`Self::glwe_compressed_encrypt_sk`] and [`Self::glwe_compressed_encrypt_zero_sk`]; may
    /// report less than the reference, never less than the body takes.
    fn glwe_compressed_encrypt_sk_tmp_bytes<A>(module: &Module<Self>, infos: &A) -> usize
    where
        A: GLWEInfos;

    /// Reproduces
    /// [`glwe_compressed_encrypt_sk_reference`](crate::reference::encryption::GLWECompressedEncryptSkReference::glwe_compressed_encrypt_sk_reference)
    /// bit for bit under any [`PrepareHint`](poulpy_hal::layouts::PrepareHint), within
    /// [`Self::glwe_compressed_encrypt_sk_tmp_bytes`]; free in kernels and scratch use before the wipe.
    fn glwe_compressed_encrypt_sk<R, P, S>(
        module: &Module<Self>,
        res: &mut R,
        pt: &P,
        sk: &S,
        seed_xa: [u8; 32],
        source_xe: &mut Source,
        scratch: &mut ScratchArena<'_, Self>,
    ) where
        R: GLWECompressedToBackendMut<Self> + GLWECompressedSeedMut,
        P: GLWEToBackendRef<Self>,
        S: GLWESecretPreparedToBackendRef<Self>;

    /// Reproduces
    /// [`glwe_compressed_encrypt_zero_sk_reference`](crate::reference::encryption::GLWECompressedEncryptSkReference::glwe_compressed_encrypt_zero_sk_reference)
    /// bit for bit under any [`PrepareHint`](poulpy_hal::layouts::PrepareHint), within
    /// [`Self::glwe_compressed_encrypt_sk_tmp_bytes`]; free in kernels and scratch use before the wipe.
    fn glwe_compressed_encrypt_zero_sk<R, S>(
        module: &Module<Self>,
        res: &mut R,
        sk: &S,
        seed_xa: [u8; 32],
        source_xe: &mut Source,
        scratch: &mut ScratchArena<'_, Self>,
    ) where
        R: GLWECompressedToBackendMut<Self> + GLWECompressedSeedMut,
        S: GLWESecretPreparedToBackendRef<Self>;

    /// Scratch bound of [`Self::gglwe_compressed_encrypt_sk`]; may report less than the reference, never less than
    /// the body takes.
    fn gglwe_compressed_encrypt_sk_tmp_bytes<A>(module: &Module<Self>, infos: &A) -> usize
    where
        A: GGLWEInfos;

    /// Reproduces
    /// [`gglwe_compressed_encrypt_sk_reference`](crate::reference::encryption::GGLWECompressedEncryptSkReference::gglwe_compressed_encrypt_sk_reference)
    /// bit for bit under any [`PrepareHint`](poulpy_hal::layouts::PrepareHint), within
    /// [`Self::gglwe_compressed_encrypt_sk_tmp_bytes`]; free in kernels and scratch use before the wipe.
    fn gglwe_compressed_encrypt_sk<R, P, S>(
        module: &Module<Self>,
        res: &mut R,
        pt: &P,
        sk: &S,
        seed: [u8; 32],
        source_xe: &mut Source,
        scratch: &mut ScratchArena<'_, Self>,
    ) where
        R: GGLWECompressedToBackendMut<Self> + GGLWECompressedSeedMut,
        P: ScalarZnxToBackendRef<Self>,
        S: GLWESecretPreparedToBackendRef<Self>;

    /// Scratch bound of [`Self::ggsw_compressed_encrypt_sk`]; may report less than the reference, never less than the
    /// body takes.
    fn ggsw_compressed_encrypt_sk_tmp_bytes<A>(module: &Module<Self>, infos: &A) -> usize
    where
        A: GGSWInfos;

    /// Reproduces
    /// [`ggsw_compressed_encrypt_sk_reference`](crate::reference::encryption::GGSWCompressedEncryptSkReference::ggsw_compressed_encrypt_sk_reference)
    /// bit for bit under any [`PrepareHint`](poulpy_hal::layouts::PrepareHint), within
    /// [`Self::ggsw_compressed_encrypt_sk_tmp_bytes`]; free in kernels and scratch use before the wipe.
    fn ggsw_compressed_encrypt_sk<R, P, S>(
        module: &Module<Self>,
        res: &mut R,
        pt: &P,
        sk: &S,
        seed_xa: [u8; 32],
        source_xe: &mut Source,
        scratch: &mut ScratchArena<'_, Self>,
    ) where
        R: GGSWCompressedToBackendMut<Self> + GGSWCompressedSeedMut + GGSWInfos,
        P: ScalarZnxToBackendRef<Self>,
        S: GLWESecretPreparedToBackendRef<Self>;

    /// Scratch bound of [`Self::gglwe_to_ggsw_key_compressed_encrypt_sk`]; may report less than the reference, never
    /// less than the body takes.
    fn gglwe_to_ggsw_key_compressed_encrypt_sk_tmp_bytes<A>(module: &Module<Self>, infos: &A) -> usize
    where
        A: GGLWEInfos;

    /// Reproduces
    /// [`gglwe_to_ggsw_key_compressed_encrypt_sk_reference`](crate::reference::encryption::GGLWEToGGSWKeyCompressedEncryptSkReference::gglwe_to_ggsw_key_compressed_encrypt_sk_reference)
    /// bit for bit, within [`Self::gglwe_to_ggsw_key_compressed_encrypt_sk_tmp_bytes`]; free in kernels and scratch
    /// use.
    fn gglwe_to_ggsw_key_compressed_encrypt_sk<R, S>(
        module: &Module<Self>,
        res: &mut R,
        sk: &S,
        seed_xa: [u8; 32],
        source_xe: &mut Source,
        scratch: &mut ScratchArena<'_, Self>,
    ) where
        R: GGLWEToGGSWKeyCompressedToBackendMut<Self> + GGLWEInfos,
        S: GLWESecretToBackendRef<Self> + GetDistribution + GLWEInfos;

    /// Scratch bound of [`Self::glwe_automorphism_key_compressed_encrypt_sk`]; may report less than the reference,
    /// never less than the body takes.
    fn glwe_automorphism_key_compressed_encrypt_sk_tmp_bytes<A>(module: &Module<Self>, infos: &A) -> usize
    where
        A: GGLWEInfos;

    /// Reproduces
    /// [`glwe_automorphism_key_compressed_encrypt_sk_reference`](crate::reference::encryption::GLWEAutomorphismKeyCompressedEncryptSkReference::glwe_automorphism_key_compressed_encrypt_sk_reference)
    /// bit for bit, within [`Self::glwe_automorphism_key_compressed_encrypt_sk_tmp_bytes`]; free in kernels and
    /// scratch use before the wipe.
    fn glwe_automorphism_key_compressed_encrypt_sk<R, S>(
        module: &Module<Self>,
        res: &mut R,
        p: i64,
        sk: &S,
        seed_xa: [u8; 32],
        source_xe: &mut Source,
        scratch: &mut ScratchArena<'_, Self>,
    ) where
        R: GGLWECompressedToBackendMut<Self> + GGLWECompressedSeedMut + SetGaloisElement + GGLWEInfos,
        S: GLWESecretToBackendRef<Self> + GLWEInfos;

    /// Scratch bound of [`Self::glwe_switching_key_compressed_encrypt_sk`]; may report less than the reference, never
    /// less than the body takes.
    fn glwe_switching_key_compressed_encrypt_sk_tmp_bytes<A>(module: &Module<Self>, infos: &A) -> usize
    where
        A: GGLWEInfos;

    /// Reproduces
    /// [`glwe_switching_key_compressed_encrypt_sk_reference`](crate::reference::encryption::GLWESwitchingKeyCompressedEncryptSkReference::glwe_switching_key_compressed_encrypt_sk_reference)
    /// bit for bit, within [`Self::glwe_switching_key_compressed_encrypt_sk_tmp_bytes`]; free in kernels and scratch
    /// use.
    fn glwe_switching_key_compressed_encrypt_sk<R, S1, S2>(
        module: &Module<Self>,
        res: &mut R,
        sk_in: &S1,
        sk_out: &S2,
        seed_xa: [u8; 32],
        source_xe: &mut Source,
        scratch: &mut ScratchArena<'_, Self>,
    ) where
        R: GGLWECompressedToBackendMut<Self> + GGLWECompressedSeedMut + GLWESwitchingKeyDegreesMut + GGLWEInfos,
        S1: GLWESecretToBackendRef<Self> + GLWEInfos,
        S2: GLWESecretToBackendRef<Self> + GetDistribution + GLWEInfos;

    /// Default body: `glwe_tensor_key_compressed_encrypt_sk_tmp_bytes_derived`, the prepared secret and its tensor
    /// plus the larger of the tensor preparation and [`Self::gglwe_compressed_encrypt_sk_tmp_bytes`].
    fn glwe_tensor_key_compressed_encrypt_sk_tmp_bytes<A>(module: &Module<Self>, infos: &A) -> usize
    where
        A: GGLWEInfos,
    {
        super::derived::encryption::glwe_tensor_key_compressed_encrypt_sk_tmp_bytes_derived(module, infos)
    }

    /// Default body: `glwe_tensor_key_compressed_encrypt_sk_derived`, the secret prepared by
    /// [`SecretImpl::glwe_secret_prepare`](crate::oep::SecretImpl::glwe_secret_prepare) and its tensor by
    /// [`SecretImpl::glwe_secret_tensor_prepare`](crate::oep::SecretImpl::glwe_secret_tensor_prepare), then
    /// [`Self::gglwe_compressed_encrypt_sk`] of the tensor under the prepared secret, within
    /// [`Self::glwe_tensor_key_compressed_encrypt_sk_tmp_bytes`].
    fn glwe_tensor_key_compressed_encrypt_sk<R, S>(
        module: &Module<Self>,
        res: &mut R,
        sk: &S,
        seed_xa: [u8; 32],
        source_xe: &mut Source,
        scratch: &mut ScratchArena<'_, Self>,
    ) where
        R: GGLWECompressedToBackendMut<Self> + GGLWEInfos + GGLWECompressedSeedMut,
        S: GLWESecretToBackendRef<Self> + GetDistribution + GLWEInfos,
    {
        super::derived::encryption::glwe_tensor_key_compressed_encrypt_sk_derived(module, res, sk, seed_xa, source_xe, scratch)
    }
}

/// Selects the HAL-based encryption algorithms and the same-layer derived defaults.
///
/// Implement `EncryptionImpl` directly to replace individual operations; unchanged
/// primitive methods may call their corresponding `*Reference` helper.
#[macro_export]
macro_rules! impl_encryption_reference_full {
    ($be:ty) => {
        unsafe impl $crate::oep::EncryptionImpl for $be {
    fn lwe_switching_key_encrypt_sk_tmp_bytes<A>(module: &::poulpy_hal::layouts::Module<$be>, infos: &A) -> usize
    where
        A: $crate::layouts::GGLWEInfos {
            <::poulpy_hal::layouts::Module<$be> as $crate::reference::encryption::LWESwitchingKeyEncryptReference<$be>>::lwe_switching_key_encrypt_sk_tmp_bytes_reference::<A>(module, infos)
        }

    fn lwe_switching_key_encrypt_sk<R, S1, S2>(
        module: &::poulpy_hal::layouts::Module<$be>,
        res: &mut R,
        sk_lwe_in: &S1,
        sk_lwe_out: &S2,
        source_xe: &mut ::poulpy_hal::source::Source,
        source_xa: &mut ::poulpy_hal::source::Source,
        scratch: &mut ::poulpy_hal::layouts::ScratchArena<'_, $be>,
    ) where
        R: $crate::layouts::GGLWEToBackendMut<$be> + $crate::layouts::GLWESwitchingKeyDegreesMut + $crate::layouts::GGLWEInfos,
        S1: $crate::layouts::LWESecretToBackendRef<$be>,
        S2: $crate::layouts::LWESecretToBackendRef<$be> {
            <::poulpy_hal::layouts::Module<$be> as $crate::reference::encryption::LWESwitchingKeyEncryptReference<$be>>::lwe_switching_key_encrypt_sk_reference::<R, S1, S2>(module, res, sk_lwe_in, sk_lwe_out, source_xe, source_xa, scratch)
        }

    fn glwe_to_lwe_key_encrypt_sk_tmp_bytes<A>(module: &::poulpy_hal::layouts::Module<$be>, infos: &A) -> usize
            where
                A: $crate::layouts::GGLWEInfos,
            {
                <::poulpy_hal::layouts::Module<$be> as $crate::reference::encryption::GLWEToLWESwitchingKeyEncryptSkReference<
                    $be,
                >>::glwe_to_lwe_key_encrypt_sk_tmp_bytes_reference::<A>(module, infos)
            }

    fn glwe_to_lwe_key_encrypt_sk<R, S1, S2>(
                module: &::poulpy_hal::layouts::Module<$be>,
                res: &mut R,
                sk_lwe: &S1,
                sk_glwe: &S2,
                source_xe: &mut ::poulpy_hal::source::Source,
                source_xa: &mut ::poulpy_hal::source::Source,
                scratch: &mut ::poulpy_hal::layouts::ScratchArena<'_, $be>,
            ) where
                S1: $crate::layouts::LWESecretToBackendRef<$be>,
                S2: $crate::layouts::GLWESecretToBackendRef<$be>,
                R: $crate::layouts::GGLWEToBackendMut<$be> + $crate::layouts::GGLWEInfos,
            {
                <::poulpy_hal::layouts::Module<$be> as $crate::reference::encryption::GLWEToLWESwitchingKeyEncryptSkReference<
                    $be,
                >>::glwe_to_lwe_key_encrypt_sk_reference::<R, S1, S2>(
                    module, res, sk_lwe, sk_glwe, source_xe, source_xa, scratch,
                )
            }

    fn glwe_switching_key_encrypt_sk_tmp_bytes<A>(module: &::poulpy_hal::layouts::Module<$be>, infos: &A) -> usize
    where
        A: $crate::layouts::GGLWEInfos {
            <::poulpy_hal::layouts::Module<$be> as $crate::reference::encryption::GLWESwitchingKeyEncryptSkReference<$be>>::glwe_switching_key_encrypt_sk_tmp_bytes_reference::<A>(module, infos)
        }

    fn glwe_switching_key_encrypt_sk<R, S1, S2>(
        module: &::poulpy_hal::layouts::Module<$be>,
        res: &mut R,
        sk_in: &S1,
        sk_out: &S2,
        source_xe: &mut ::poulpy_hal::source::Source,
        source_xa: &mut ::poulpy_hal::source::Source,
        scratch: &mut ::poulpy_hal::layouts::ScratchArena<'_, $be>,
    ) where
        R: $crate::layouts::GGLWEToBackendMut<$be> + $crate::layouts::GLWESwitchingKeyDegreesMut + $crate::layouts::GGLWEInfos,
        S1: $crate::layouts::GLWESecretToBackendRef<$be> + $crate::layouts::GLWEInfos,
        S2: $crate::layouts::GLWESecretToBackendRef<$be> + $crate::GetDistribution + $crate::layouts::GLWEInfos {
            <::poulpy_hal::layouts::Module<$be> as $crate::reference::encryption::GLWESwitchingKeyEncryptSkReference<$be>>::glwe_switching_key_encrypt_sk_reference::<R, S1, S2>(module, res, sk_in, sk_out, source_xe, source_xa, scratch)
        }

    fn ggsw_encrypt_sk_tmp_bytes<A>(module: &::poulpy_hal::layouts::Module<$be>, infos: &A) -> usize
    where
        A: $crate::layouts::GGSWInfos {
            <::poulpy_hal::layouts::Module<$be> as $crate::reference::encryption::GGSWEncryptSkReference<$be>>::ggsw_encrypt_sk_tmp_bytes_reference::<A>(module, infos)
        }

    fn ggsw_encrypt_sk<R, P, S>(
        module: &::poulpy_hal::layouts::Module<$be>,
        res: &mut R,
        pt: &P,
        sk: &S,
        source_xe: &mut ::poulpy_hal::source::Source,
        source_xa: &mut ::poulpy_hal::source::Source,
        scratch: &mut ::poulpy_hal::layouts::ScratchArena<'_, $be>,
    ) where
        R: $crate::layouts::GGSWToBackendMut<$be> + $crate::layouts::GGSWInfos + $crate::layouts::GGSWAtViewMut<$be>,
        P: ::poulpy_hal::layouts::ScalarZnxToBackendRef<$be> + ::poulpy_hal::layouts::ZnxInfos,
        S: $crate::layouts::GLWESecretPreparedToBackendRef<$be> + $crate::layouts::LWEInfos + $crate::layouts::GLWEInfos {
            <::poulpy_hal::layouts::Module<$be> as $crate::reference::encryption::GGSWEncryptSkReference<$be>>::ggsw_encrypt_sk_reference::<R, P, S>(module, res, pt, sk, source_xe, source_xa, scratch)
        }

    fn lwe_to_glwe_key_encrypt_sk_tmp_bytes<A>(module: &::poulpy_hal::layouts::Module<$be>, infos: &A) -> usize
            where
                A: $crate::layouts::GGLWEInfos,
            {
                <::poulpy_hal::layouts::Module<$be> as $crate::reference::encryption::LWEToGLWESwitchingKeyEncryptSkReference<
                    $be,
                >>::lwe_to_glwe_key_encrypt_sk_tmp_bytes_reference::<A>(module, infos)
            }

    fn lwe_to_glwe_key_encrypt_sk<R, S1, S2>(
                module: &::poulpy_hal::layouts::Module<$be>,
                res: &mut R,
                sk_lwe: &S1,
                sk_glwe: &S2,
                source_xe: &mut ::poulpy_hal::source::Source,
                source_xa: &mut ::poulpy_hal::source::Source,
                scratch: &mut ::poulpy_hal::layouts::ScratchArena<'_, $be>,
            ) where
                S1: $crate::layouts::LWESecretToBackendRef<$be>,
                S2: $crate::layouts::GLWESecretPreparedToBackendRef<$be>,
                R: $crate::layouts::GGLWEToBackendMut<$be> + $crate::layouts::GGLWEInfos,
            {
                <::poulpy_hal::layouts::Module<$be> as $crate::reference::encryption::LWEToGLWESwitchingKeyEncryptSkReference<
                    $be,
                >>::lwe_to_glwe_key_encrypt_sk_reference::<R, S1, S2>(
                    module, res, sk_lwe, sk_glwe, source_xe, source_xa, scratch,
                )
            }

    fn fill_lwe_mask_from_source<R>(module: &::poulpy_hal::layouts::Module<$be>, base2k: usize, res: &mut R, source_xa: &mut ::poulpy_hal::source::Source)
    where
        R: $crate::layouts::LWEToBackendMut<$be> {
            <::poulpy_hal::layouts::Module<$be> as $crate::reference::encryption::LWEFillMaskReference<$be>>::fill_lwe_mask_from_source_reference::<R>(module, base2k, res, source_xa)
        }

    fn lwe_encrypt_sk_tmp_bytes<A>(module: &::poulpy_hal::layouts::Module<$be>, infos: &A) -> usize
    where
        A: $crate::layouts::LWEInfos {
            <::poulpy_hal::layouts::Module<$be> as $crate::reference::encryption::LWEEncryptSkReference<$be>>::lwe_encrypt_sk_tmp_bytes_reference::<A>(module, infos)
        }

    fn lwe_encrypt_sk<R, P, S>(
        module: &::poulpy_hal::layouts::Module<$be>,
        res: &mut R,
        pt: &P,
        sk: &S,
        source_xe: &mut ::poulpy_hal::source::Source,
        source_xa: &mut ::poulpy_hal::source::Source,
        scratch: &mut ::poulpy_hal::layouts::ScratchArena<'_, $be>,
    ) where
        R: $crate::layouts::LWEToBackendMut<$be> + $crate::layouts::LWEInfos,
        P: $crate::layouts::LWEPlaintextToBackendRef<$be>,
        S: $crate::layouts::LWESecretToBackendRef<$be>,
        {
            <::poulpy_hal::layouts::Module<$be> as $crate::reference::encryption::LWEEncryptSkReference<$be>>::lwe_encrypt_sk_reference::<R, P, S>(module, res, pt, sk, source_xe, source_xa, scratch)
        }

    fn gglwe_encrypt_sk_tmp_bytes<A>(module: &::poulpy_hal::layouts::Module<$be>, infos: &A) -> usize
    where
        A: $crate::layouts::GGLWEInfos {
            <::poulpy_hal::layouts::Module<$be> as $crate::reference::encryption::GGLWEEncryptSkReference<$be>>::gglwe_encrypt_sk_tmp_bytes_reference::<A>(module, infos)
        }

    fn gglwe_encrypt_sk<R, P, S>(
        module: &::poulpy_hal::layouts::Module<$be>,
        res: &mut R,
        pt: &P,
        sk: &S,
        source_xe: &mut ::poulpy_hal::source::Source,
        source_xa: &mut ::poulpy_hal::source::Source,
        scratch: &mut ::poulpy_hal::layouts::ScratchArena<'_, $be>,
    ) where
        R: $crate::layouts::GGLWEToBackendMut<$be>,
        P: ::poulpy_hal::layouts::ScalarZnxToBackendRef<$be>,
        S: $crate::layouts::GLWESecretPreparedToBackendRef<$be> {
            <::poulpy_hal::layouts::Module<$be> as $crate::reference::encryption::GGLWEEncryptSkReference<$be>>::gglwe_encrypt_sk_reference::<R, P, S>(module, res, pt, sk, source_xe, source_xa, scratch)
        }

    fn fill_glwe_mask_from_source<R>(
        module: &::poulpy_hal::layouts::Module<$be>,
        res: &mut R,
        source_xa: &mut ::poulpy_hal::source::Source,
    ) where
        R: $crate::layouts::GLWEToBackendMut<$be> {
            <::poulpy_hal::layouts::Module<$be> as $crate::reference::encryption::GLWEMaskFillReference<$be>>::fill_glwe_mask_from_source_reference::<R>(module, res, source_xa)
        }

    fn fill_glwe_from_source<R>(
        module: &::poulpy_hal::layouts::Module<$be>,
        res: &mut R,
        source: &mut ::poulpy_hal::source::Source,
    ) where
        R: $crate::layouts::GLWEToBackendMut<$be> {
            <::poulpy_hal::layouts::Module<$be> as $crate::reference::encryption::GLWEMaskFillReference<$be>>::fill_glwe_from_source_reference::<R>(module, res, source)
        }

    fn glwe_encrypt_sk_tmp_bytes<A>(module: &::poulpy_hal::layouts::Module<$be>, infos: &A) -> usize
    where
        A: $crate::layouts::GLWEInfos {
            <::poulpy_hal::layouts::Module<$be> as $crate::reference::encryption::GLWEEncryptSkReference<$be>>::glwe_encrypt_sk_tmp_bytes_reference::<A>(module, infos)
        }

    fn glwe_encrypt_sk<R, P, S>(
        module: &::poulpy_hal::layouts::Module<$be>,
        res: &mut R,
        pt: &P,
        sk: &S,
        source_xe: &mut ::poulpy_hal::source::Source,
        source_xa: &mut ::poulpy_hal::source::Source,
        scratch: &mut ::poulpy_hal::layouts::ScratchArena<'_, $be>,
    ) where
        R: $crate::layouts::GLWEToBackendMut<$be>,
        P: $crate::layouts::GLWEToBackendRef<$be>,
        S: $crate::layouts::GLWESecretPreparedToBackendRef<$be> {
            <::poulpy_hal::layouts::Module<$be> as $crate::reference::encryption::GLWEEncryptSkReference<$be>>::glwe_encrypt_sk_reference::<R, P, S>(module, res, pt, sk, source_xe, source_xa, scratch)
        }

    fn glwe_encrypt_zero_sk<R, S>(
        module: &::poulpy_hal::layouts::Module<$be>,
        res: &mut R,
        sk: &S,
        source_xe: &mut ::poulpy_hal::source::Source,
        source_xa: &mut ::poulpy_hal::source::Source,
        scratch: &mut ::poulpy_hal::layouts::ScratchArena<'_, $be>,
    ) where
        R: $crate::layouts::GLWEToBackendMut<$be>,
        S: $crate::layouts::GLWESecretPreparedToBackendRef<$be> {
            <::poulpy_hal::layouts::Module<$be> as $crate::reference::encryption::GLWEEncryptSkReference<$be>>::glwe_encrypt_zero_sk_reference::<R, S>(module, res, sk, source_xe, source_xa, scratch)
        }

    fn glwe_encrypt_sk_with_mask<R, P, S>(
        module: &::poulpy_hal::layouts::Module<$be>,
        res: &mut R,
        pt: &P,
        sk: &S,
        source_xe: &mut ::poulpy_hal::source::Source,
        scratch: &mut ::poulpy_hal::layouts::ScratchArena<'_, $be>,
    ) where
        R: $crate::layouts::GLWEToBackendMut<$be>,
        P: $crate::layouts::GLWEToBackendRef<$be>,
        S: $crate::layouts::GLWESecretPreparedToBackendRef<$be> {
            <::poulpy_hal::layouts::Module<$be> as $crate::reference::encryption::GLWEEncryptSkReference<$be>>::glwe_encrypt_sk_with_mask_reference::<R, P, S>(module, res, pt, sk, source_xe, scratch)
        }

    fn glwe_encrypt_pk_tmp_bytes<R, K>(module: &::poulpy_hal::layouts::Module<$be>, res_infos: &R, pk_infos: &K) -> usize
    where
        R: $crate::layouts::GLWEInfos,
        K: $crate::layouts::GLWEInfos {
            <::poulpy_hal::layouts::Module<$be> as $crate::reference::encryption::GLWEEncryptPkReference<$be>>::glwe_encrypt_pk_tmp_bytes_reference::<R, K>(module, res_infos, pk_infos)
        }

    fn glwe_encrypt_pk_sampled_at<R, P, K>(
        module: &::poulpy_hal::layouts::Module<$be>,
        res: &mut R,
        pt: Option<(&P, usize)>,
        body_noise: bool,
        pk: &K,
        k_sample: $crate::layouts::TorusPrecision,
        source_xu: &mut ::poulpy_hal::source::Source,
        source_xe: &mut ::poulpy_hal::source::Source,
        scratch: &mut ::poulpy_hal::layouts::ScratchArena<'_, $be>,
    ) where
        R: $crate::layouts::GLWEToBackendMut<$be> + $crate::layouts::GLWEInfos,
        P: $crate::layouts::GLWEToBackendRef<$be> + $crate::layouts::GLWEInfos,
        K: $crate::layouts::GLWEPublicKeyPreparedToBackendRef<$be> + $crate::layouts::GLWEInfos {
            <::poulpy_hal::layouts::Module<$be> as $crate::reference::encryption::GLWEEncryptPkReference<$be>>::glwe_encrypt_pk_sampled_at_reference::<R, P, K>(module, res, pt, body_noise, pk, k_sample, source_xu, source_xe, scratch)
        }

    fn gglwe_to_ggsw_key_encrypt_sk_tmp_bytes<A>(module: &::poulpy_hal::layouts::Module<$be>, infos: &A) -> usize
    where
        A: $crate::layouts::GGLWEInfos {
            <::poulpy_hal::layouts::Module<$be> as $crate::reference::encryption::GGLWEToGGSWKeyEncryptSkReference<$be>>::gglwe_to_ggsw_key_encrypt_sk_tmp_bytes_reference::<A>(module, infos)
        }

    fn gglwe_to_ggsw_key_encrypt_sk<R, S>(
        module: &::poulpy_hal::layouts::Module<$be>,
        res: &mut R,
        sk: &S,
        source_xe: &mut ::poulpy_hal::source::Source,
        source_xa: &mut ::poulpy_hal::source::Source,
        scratch: &mut ::poulpy_hal::layouts::ScratchArena<'_, $be>,
    ) where
        R: $crate::layouts::GGLWEToGGSWKeyToBackendMut<$be>,
        S: $crate::layouts::GLWESecretToBackendRef<$be> + $crate::GetDistribution + $crate::layouts::GLWEInfos {
            <::poulpy_hal::layouts::Module<$be> as $crate::reference::encryption::GGLWEToGGSWKeyEncryptSkReference<$be>>::gglwe_to_ggsw_key_encrypt_sk_reference::<R, S>(module, res, sk, source_xe, source_xa, scratch)
        }

    fn glwe_automorphism_key_encrypt_sk_tmp_bytes<A>(module: &::poulpy_hal::layouts::Module<$be>, infos: &A) -> usize
            where
                A: $crate::layouts::GGLWEInfos,
            {
                <::poulpy_hal::layouts::Module<$be> as $crate::reference::encryption::GLWEAutomorphismKeyEncryptSkReference<
                    $be,
                >>::glwe_automorphism_key_encrypt_sk_tmp_bytes_reference::<A>(module, infos)
            }

    fn glwe_automorphism_key_encrypt_sk<R, S>(
                module: &::poulpy_hal::layouts::Module<$be>,
                res: &mut R,
                p: i64,
                sk: &S,
                source_xe: &mut ::poulpy_hal::source::Source,
                source_xa: &mut ::poulpy_hal::source::Source,
                scratch: &mut ::poulpy_hal::layouts::ScratchArena<'_, $be>,
            ) where
                R: $crate::layouts::GGLWEToBackendMut<$be> + $crate::layouts::SetGaloisElement + $crate::layouts::GGLWEInfos,
                S: $crate::layouts::GLWESecretToBackendRef<$be> + $crate::layouts::GLWEInfos,
            {
                <::poulpy_hal::layouts::Module<$be> as $crate::reference::encryption::GLWEAutomorphismKeyEncryptSkReference<
                    $be,
                >>::glwe_automorphism_key_encrypt_sk_reference::<R, S>(
                    module, res, p, sk, source_xe, source_xa, scratch,
                )
            }

    fn glwe_compressed_encrypt_sk_tmp_bytes<A>(module: &::poulpy_hal::layouts::Module<$be>, infos: &A) -> usize
    where
        A: $crate::layouts::GLWEInfos {
            <::poulpy_hal::layouts::Module<$be> as $crate::reference::encryption::GLWECompressedEncryptSkReference<$be>>::glwe_compressed_encrypt_sk_tmp_bytes_reference::<A>(module, infos)
        }

    fn glwe_compressed_encrypt_sk<R, P, S>(
        module: &::poulpy_hal::layouts::Module<$be>,
        res: &mut R,
        pt: &P,
        sk: &S,
        seed_xa: [u8; 32],
        source_xe: &mut ::poulpy_hal::source::Source,
        scratch: &mut ::poulpy_hal::layouts::ScratchArena<'_, $be>,
    ) where
        R: $crate::layouts::GLWECompressedToBackendMut<$be> + $crate::layouts::GLWECompressedSeedMut,
        P: $crate::layouts::GLWEToBackendRef<$be>,
        S: $crate::layouts::GLWESecretPreparedToBackendRef<$be> {
            <::poulpy_hal::layouts::Module<$be> as $crate::reference::encryption::GLWECompressedEncryptSkReference<$be>>::glwe_compressed_encrypt_sk_reference::<R, P, S>(module, res, pt, sk, seed_xa, source_xe, scratch)
        }

    fn glwe_compressed_encrypt_zero_sk<R, S>(
        module: &::poulpy_hal::layouts::Module<$be>,
        res: &mut R,
        sk: &S,
        seed_xa: [u8; 32],
        source_xe: &mut ::poulpy_hal::source::Source,
        scratch: &mut ::poulpy_hal::layouts::ScratchArena<'_, $be>,
    ) where
        R: $crate::layouts::GLWECompressedToBackendMut<$be> + $crate::layouts::GLWECompressedSeedMut,
        S: $crate::layouts::GLWESecretPreparedToBackendRef<$be> {
            <::poulpy_hal::layouts::Module<$be> as $crate::reference::encryption::GLWECompressedEncryptSkReference<$be>>::glwe_compressed_encrypt_zero_sk_reference::<R, S>(module, res, sk, seed_xa, source_xe, scratch)
        }

    fn glwe_switching_key_compressed_encrypt_sk_tmp_bytes<A>(module: &::poulpy_hal::layouts::Module<$be>, infos: &A) -> usize
    where
        A: $crate::layouts::GGLWEInfos {
            <::poulpy_hal::layouts::Module<$be> as $crate::reference::encryption::GLWESwitchingKeyCompressedEncryptSkReference<$be>>::glwe_switching_key_compressed_encrypt_sk_tmp_bytes_reference::<A>(module, infos)
        }

    fn glwe_switching_key_compressed_encrypt_sk<R, S1, S2>(
        module: &::poulpy_hal::layouts::Module<$be>,
        res: &mut R,
        sk_in: &S1,
        sk_out: &S2,
        seed_xa: [u8; 32],
        source_xe: &mut ::poulpy_hal::source::Source,
        scratch: &mut ::poulpy_hal::layouts::ScratchArena<'_, $be>,
    ) where
        R: $crate::layouts::GGLWECompressedToBackendMut<$be> + $crate::layouts::GGLWECompressedSeedMut + $crate::layouts::GLWESwitchingKeyDegreesMut + $crate::layouts::GGLWEInfos,
        S1: $crate::layouts::GLWESecretToBackendRef<$be> + $crate::layouts::GLWEInfos,
        S2: $crate::layouts::GLWESecretToBackendRef<$be> + $crate::GetDistribution + $crate::layouts::GLWEInfos {
            <::poulpy_hal::layouts::Module<$be> as $crate::reference::encryption::GLWESwitchingKeyCompressedEncryptSkReference<$be>>::glwe_switching_key_compressed_encrypt_sk_reference::<R, S1, S2>(module, res, sk_in, sk_out, seed_xa, source_xe, scratch)
        }

    fn ggsw_compressed_encrypt_sk_tmp_bytes<A>(module: &::poulpy_hal::layouts::Module<$be>, infos: &A) -> usize
    where
        A: $crate::layouts::GGSWInfos {
            <::poulpy_hal::layouts::Module<$be> as $crate::reference::encryption::GGSWCompressedEncryptSkReference<$be>>::ggsw_compressed_encrypt_sk_tmp_bytes_reference::<A>(module, infos)
        }

    fn ggsw_compressed_encrypt_sk<R, P, S>(
        module: &::poulpy_hal::layouts::Module<$be>,
        res: &mut R,
        pt: &P,
        sk: &S,
        seed_xa: [u8; 32],
        source_xe: &mut ::poulpy_hal::source::Source,
        scratch: &mut ::poulpy_hal::layouts::ScratchArena<'_, $be>,
    ) where
        R: $crate::layouts::GGSWCompressedToBackendMut<$be> + $crate::layouts::GGSWCompressedSeedMut + $crate::layouts::GGSWInfos,
        P: ::poulpy_hal::layouts::ScalarZnxToBackendRef<$be>,
        S: $crate::layouts::GLWESecretPreparedToBackendRef<$be> {
            <::poulpy_hal::layouts::Module<$be> as $crate::reference::encryption::GGSWCompressedEncryptSkReference<$be>>::ggsw_compressed_encrypt_sk_reference::<R, P, S>(module, res, pt, sk, seed_xa, source_xe, scratch)
        }

    fn gglwe_compressed_encrypt_sk_tmp_bytes<A>(module: &::poulpy_hal::layouts::Module<$be>, infos: &A) -> usize
    where
        A: $crate::layouts::GGLWEInfos {
            <::poulpy_hal::layouts::Module<$be> as $crate::reference::encryption::GGLWECompressedEncryptSkReference<$be>>::gglwe_compressed_encrypt_sk_tmp_bytes_reference::<A>(module, infos)
        }

    fn gglwe_compressed_encrypt_sk<R, P, S>(
        module: &::poulpy_hal::layouts::Module<$be>,
        res: &mut R,
        pt: &P,
        sk: &S,
        seed: [u8; 32],
        source_xe: &mut ::poulpy_hal::source::Source,
        scratch: &mut ::poulpy_hal::layouts::ScratchArena<'_, $be>,
    ) where
        R: $crate::layouts::GGLWECompressedToBackendMut<$be> + $crate::layouts::GGLWECompressedSeedMut,
        P: ::poulpy_hal::layouts::ScalarZnxToBackendRef<$be>,
        S: $crate::layouts::GLWESecretPreparedToBackendRef<$be> {
            <::poulpy_hal::layouts::Module<$be> as $crate::reference::encryption::GGLWECompressedEncryptSkReference<$be>>::gglwe_compressed_encrypt_sk_reference::<R, P, S>(module, res, pt, sk, seed, source_xe, scratch)
        }

    fn gglwe_to_ggsw_key_compressed_encrypt_sk_tmp_bytes<A>(module: &::poulpy_hal::layouts::Module<$be>, infos: &A) -> usize
    where
        A: $crate::layouts::GGLWEInfos {
            <::poulpy_hal::layouts::Module<$be> as $crate::reference::encryption::GGLWEToGGSWKeyCompressedEncryptSkReference<$be>>::gglwe_to_ggsw_key_compressed_encrypt_sk_tmp_bytes_reference::<A>(module, infos)
        }

    fn gglwe_to_ggsw_key_compressed_encrypt_sk<R, S>(
        module: &::poulpy_hal::layouts::Module<$be>,
        res: &mut R,
        sk: &S,
        seed_xa: [u8; 32],
        source_xe: &mut ::poulpy_hal::source::Source,
        scratch: &mut ::poulpy_hal::layouts::ScratchArena<'_, $be>,
    ) where
        R: $crate::layouts::GGLWEToGGSWKeyCompressedToBackendMut<$be> + $crate::layouts::GGLWEInfos,
        S: $crate::layouts::GLWESecretToBackendRef<$be> + $crate::GetDistribution + $crate::layouts::GLWEInfos {
            <::poulpy_hal::layouts::Module<$be> as $crate::reference::encryption::GGLWEToGGSWKeyCompressedEncryptSkReference<$be>>::gglwe_to_ggsw_key_compressed_encrypt_sk_reference::<R, S>(module, res, sk, seed_xa, source_xe, scratch)
        }

    fn glwe_automorphism_key_compressed_encrypt_sk_tmp_bytes<A>(module: &::poulpy_hal::layouts::Module<$be>, infos: &A) -> usize
    where
        A: $crate::layouts::GGLWEInfos {
            <::poulpy_hal::layouts::Module<$be> as $crate::reference::encryption::GLWEAutomorphismKeyCompressedEncryptSkReference<$be>>::glwe_automorphism_key_compressed_encrypt_sk_tmp_bytes_reference::<A>(module, infos)
        }

    fn glwe_automorphism_key_compressed_encrypt_sk<R, S>(
        module: &::poulpy_hal::layouts::Module<$be>,
        res: &mut R,
        p: i64,
        sk: &S,
        seed_xa: [u8; 32],
        source_xe: &mut ::poulpy_hal::source::Source,
        scratch: &mut ::poulpy_hal::layouts::ScratchArena<'_, $be>,
    ) where
        R: $crate::layouts::GGLWECompressedToBackendMut<$be> + $crate::layouts::GGLWECompressedSeedMut + $crate::layouts::SetGaloisElement + $crate::layouts::GGLWEInfos,
        S: $crate::layouts::GLWESecretToBackendRef<$be> + $crate::layouts::GLWEInfos {
            <::poulpy_hal::layouts::Module<$be> as $crate::reference::encryption::GLWEAutomorphismKeyCompressedEncryptSkReference<$be>>::glwe_automorphism_key_compressed_encrypt_sk_reference::<R, S>(module, res, p, sk, seed_xa, source_xe, scratch)
        }
        }
    };
}
