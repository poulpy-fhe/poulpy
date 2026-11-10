#![allow(clippy::too_many_arguments)]

use poulpy_hal::{
    layouts::{Backend, Module, Ring, ScalarZnxToBackendRef, ScratchArena, Standard, ZnxInfos},
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
pub unsafe trait EncryptionImpl<Rg: Ring = Standard>:
    Backend + SamplingImpl<Rg> + HalVecZnxImpl<Rg> + GLWENormalizeImpl<Rg> + SecretImpl<Rg>
{
    /// Reproduces
    /// [`glwe_fill_mask_from_source_reference`](crate::reference::encryption::GLWEFillReference::glwe_fill_mask_from_source_reference)
    /// bit for bit; free in kernels.
    fn glwe_fill_mask_from_source<R>(module: &Module<Self>, res: &mut R, source_xa: &mut Source)
    where
        R: GLWEToBackendMut<Self, Rg>;

    /// Default body: `glwe_fill_mask_from_seed_derived`, [`Self::glwe_fill_mask_from_source`] on a `Source` built
    /// from the seed.
    fn glwe_fill_mask_from_seed<R>(module: &Module<Self>, res: &mut R, seed_xa: [u8; 32])
    where
        R: GLWEToBackendMut<Self, Rg>,
    {
        super::derived::encryption::glwe_fill_mask_from_seed_derived(module, res, seed_xa)
    }

    /// Reproduces
    /// [`glwe_fill_from_source_reference`](crate::reference::encryption::GLWEFillReference::glwe_fill_from_source_reference)
    /// bit for bit; free in kernels.
    fn glwe_fill_from_source<R>(module: &Module<Self>, res: &mut R, source: &mut Source)
    where
        R: GLWEToBackendMut<Self, Rg>;

    /// Reproduces
    /// [`lwe_fill_mask_from_source_reference`](crate::reference::encryption::LWEFillMaskReference::lwe_fill_mask_from_source_reference)
    /// bit for bit; free in kernels.
    fn lwe_fill_mask_from_source<R>(module: &Module<Self>, base2k: usize, res: &mut R, source_xa: &mut Source)
    where
        R: LWEToBackendMut<Self, Rg>;

    /// Default body: `lwe_fill_mask_from_seed_derived`, [`Self::lwe_fill_mask_from_source`] on a `Source` built from
    /// the seed.
    fn lwe_fill_mask_from_seed<R>(module: &Module<Self>, base2k: usize, res: &mut R, seed_xa: [u8; 32])
    where
        R: LWEToBackendMut<Self, Rg>,
    {
        super::derived::encryption::lwe_fill_mask_from_seed_derived(module, base2k, res, seed_xa)
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
        R: LWEToBackendMut<Self, Rg> + LWEInfos,
        P: LWEPlaintextToBackendRef<Self, Rg>,
        S: LWESecretToBackendRef<Self, Rg>;

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
        R: GLWEToBackendMut<Self, Rg>,
        P: GLWEToBackendRef<Self, Rg>,
        S: GLWESecretPreparedToBackendRef<Self, Rg>;

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
        R: GLWEToBackendMut<Self, Rg>,
        S: GLWESecretPreparedToBackendRef<Self, Rg>;

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
        R: GLWEToBackendMut<Self, Rg>,
        P: GLWEToBackendRef<Self, Rg>,
        S: GLWESecretPreparedToBackendRef<Self, Rg>;

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
        R: GLWEToBackendMut<Self, Rg> + GLWEInfos,
        P: GLWEToBackendRef<Self, Rg> + GLWEInfos,
        K: GLWEPublicKeyPreparedToBackendRef<Self, Rg> + GLWEInfos,
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
        R: GLWEToBackendMut<Self, Rg> + GLWEInfos,
        K: GLWEPublicKeyPreparedToBackendRef<Self, Rg> + GLWEInfos,
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
        R: GLWEToBackendMut<Self, Rg> + GLWEInfos,
        P: GLWEToBackendRef<Self, Rg> + GLWEInfos,
        K: GLWEPublicKeyPreparedToBackendRef<Self, Rg> + GLWEInfos,
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
        R: GLWEToBackendMut<Self, Rg> + GLWEInfos,
        P: GLWEToBackendRef<Self, Rg> + GLWEInfos,
        K: GLWEPublicKeyPreparedToBackendRef<Self, Rg> + GLWEInfos;

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
    /// once more, then the wipe of its query, within [`Self::glwe_encrypt_pk_smudged_tmp_bytes`], under any
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
        R: GLWEToBackendMut<Self, Rg> + GLWEInfos,
        P: GLWEToBackendRef<Self, Rg> + GLWEInfos,
        K: GLWEPublicKeyPreparedToBackendRef<Self, Rg> + GLWEInfos,
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
    /// the `rank` entries, then the wipe of its query and the secret's noise estimate and distribution on `res`, within
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
        R: GLWEPublicKeyToBackendMut<Self, Rg> + GetDistributionMut + GLWEInfos,
        S: GLWESecretPreparedToBackendRef<Self, Rg> + GetDistribution,
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
        R: GLWEPublicKeyCompressedToBackendMut<Self, Rg> + GLWEPublicKeyCompressedSeedMut + GetDistributionMut + GLWEInfos,
        S: GLWESecretPreparedToBackendRef<Self, Rg> + GetDistribution,
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
        R: GGLWEToBackendMut<Self, Rg>,
        P: ScalarZnxToBackendRef<Self, Rg>,
        S: GLWESecretPreparedToBackendRef<Self, Rg>;

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
        R: GGSWToBackendMut<Self, Rg> + GGSWInfos + GGSWAtViewMut<Self, Rg>,
        P: ScalarZnxToBackendRef<Self, Rg> + ZnxInfos,
        S: GLWESecretPreparedToBackendRef<Self, Rg> + LWEInfos + GLWEInfos;

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
        R: GGSWToBackendMut<Self, Rg> + GGSWInfos + GGSWAtViewMut<Self, Rg>,
        P: ScalarZnxToBackendRef<Self, Rg> + ZnxInfos,
        K: GLWEPublicKeyPreparedToBackendRef<Self, Rg> + GLWEInfos,
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
        R: GGLWEToGGSWKeyToBackendMut<Self, Rg>,
        S: GLWESecretToBackendRef<Self, Rg> + GetDistribution + GLWEInfos;

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
        R: GGLWEToBackendMut<Self, Rg> + GLWESwitchingKeyDegreesMut + GGLWEInfos,
        S1: GLWESecretToBackendRef<Self, Rg> + GLWEInfos,
        S2: GLWESecretToBackendRef<Self, Rg> + GetDistribution + GLWEInfos;

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
    /// [`SecretImpl::glwe_secret_tensor_from_glwe_secret`](crate::oep::SecretImpl::glwe_secret_tensor_from_glwe_secret), then
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
        R: GGLWEToBackendMut<Self, Rg> + GGLWEInfos,
        S: GLWESecretToBackendRef<Self, Rg> + GetDistribution + GLWEInfos,
    {
        super::derived::encryption::glwe_tensor_key_encrypt_sk_derived(module, res, sk, source_xe, source_xa, scratch)
    }

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
        R: GGLWEToBackendMut<Self, Rg> + SetGaloisElement + GGLWEInfos,
        S: GLWESecretToBackendRef<Self, Rg> + GLWEInfos;

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
        R: GLWECompressedToBackendMut<Self, Rg> + GLWECompressedSeedMut,
        P: GLWEToBackendRef<Self, Rg>,
        S: GLWESecretPreparedToBackendRef<Self, Rg>;

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
        R: GLWECompressedToBackendMut<Self, Rg> + GLWECompressedSeedMut,
        S: GLWESecretPreparedToBackendRef<Self, Rg>;

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
        R: GGLWECompressedToBackendMut<Self, Rg> + GGLWECompressedSeedMut,
        P: ScalarZnxToBackendRef<Self, Rg>,
        S: GLWESecretPreparedToBackendRef<Self, Rg>;

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
        R: GGSWCompressedToBackendMut<Self, Rg> + GGSWCompressedSeedMut + GGSWInfos,
        P: ScalarZnxToBackendRef<Self, Rg>,
        S: GLWESecretPreparedToBackendRef<Self, Rg>;

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
        R: GGLWEToGGSWKeyCompressedToBackendMut<Self, Rg> + GGLWEInfos,
        S: GLWESecretToBackendRef<Self, Rg> + GetDistribution + GLWEInfos;

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
        R: GGLWECompressedToBackendMut<Self, Rg> + GGLWECompressedSeedMut + SetGaloisElement + GGLWEInfos,
        S: GLWESecretToBackendRef<Self, Rg> + GLWEInfos;

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
        R: GGLWECompressedToBackendMut<Self, Rg> + GGLWECompressedSeedMut + GLWESwitchingKeyDegreesMut + GGLWEInfos,
        S1: GLWESecretToBackendRef<Self, Rg> + GLWEInfos,
        S2: GLWESecretToBackendRef<Self, Rg> + GetDistribution + GLWEInfos;

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
    /// [`SecretImpl::glwe_secret_tensor_from_glwe_secret`](crate::oep::SecretImpl::glwe_secret_tensor_from_glwe_secret), then
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
        R: GGLWECompressedToBackendMut<Self, Rg> + GGLWEInfos + GGLWECompressedSeedMut,
        S: GLWESecretToBackendRef<Self, Rg> + GetDistribution + GLWEInfos,
    {
        super::derived::encryption::glwe_tensor_key_compressed_encrypt_sk_derived(module, res, sk, seed_xa, source_xe, scratch)
    }
}

/// Backend-provided encryption of the LWE switching key and of the keys between LWE and GLWE, on the
/// standard ring only.
///
/// # Safety
/// Implementations reproduce the bodies their methods document bit for bit, report for each
/// `_tmp_bytes` query a bound that covers the body, and refuse as the reference does.
/// Every method wipes the scratch its query reports before returning.
/// The memory, layout, synchrony and aliasing rules of
/// `poulpy-hal/docs/backend_safety_contract.md` apply.
pub unsafe trait LWEKeyEncryptionImpl: Backend {
    /// Scratch bound of [`Self::glwe_to_lwe_key_encrypt_sk`]; may report less than the reference, never less than the
    /// body takes.
    fn glwe_to_lwe_key_encrypt_sk_tmp_bytes<A>(module: &Module<Self>, infos: &A) -> usize
    where
        A: GGLWEInfos;

    /// Reproduces
    /// [`glwe_to_lwe_key_encrypt_sk_reference`](crate::reference::encryption::GLWEToLWEKeyEncryptSkReference::glwe_to_lwe_key_encrypt_sk_reference)
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
    /// [`lwe_switching_key_encrypt_sk_reference`](crate::reference::encryption::LWESwitchingKeyEncryptSkReference::lwe_switching_key_encrypt_sk_reference)
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
    /// [`lwe_to_glwe_key_encrypt_sk_reference`](crate::reference::encryption::LWEToGLWEKeyEncryptSkReference::lwe_to_glwe_key_encrypt_sk_reference)
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
}

/// Selects the HAL-based encryption algorithms and the same-layer derived defaults.
///
/// Implement `EncryptionImpl` directly to replace individual operations; unchanged
/// primitive methods may call their corresponding `*Reference` helper.
///
/// `($be)` implements both rings and `($be, $ring)` one ring, as `docs/guides/layering-contract.md` describes.
#[macro_export]
macro_rules! impl_encryption_reference_full {
    ($be:ty) => {
        $crate::impl_encryption_reference_full!($be, ::poulpy_hal::layouts::Standard);
        $crate::impl_encryption_reference_full!($be, ::poulpy_hal::layouts::ConjugateInvariant);
    };
    ($be:ty, $ring:ty) => {
        unsafe impl $crate::oep::EncryptionImpl<$ring> for $be {
    fn glwe_switching_key_encrypt_sk_tmp_bytes<A>(module: &::poulpy_hal::layouts::Module<$be>, infos: &A) -> usize
    where
        A: $crate::layouts::GGLWEInfos {
            <::poulpy_hal::layouts::Module<$be> as $crate::reference::encryption::GLWESwitchingKeyEncryptSkReference<$be, $ring>>::glwe_switching_key_encrypt_sk_tmp_bytes_reference::<A>(module, infos)
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
        R: $crate::layouts::GGLWEToBackendMut<$be, $ring>
            + $crate::layouts::GLWESwitchingKeyDegreesMut
            + $crate::layouts::GGLWEInfos,
        S1: $crate::layouts::GLWESecretToBackendRef<$be, $ring> + $crate::layouts::GLWEInfos,
        S2: $crate::layouts::GLWESecretToBackendRef<$be, $ring> + $crate::GetDistribution + $crate::layouts::GLWEInfos {
            <::poulpy_hal::layouts::Module<$be> as $crate::reference::encryption::GLWESwitchingKeyEncryptSkReference<$be, $ring>>::glwe_switching_key_encrypt_sk_reference::<R, S1, S2>(module, res, sk_in, sk_out, source_xe, source_xa, scratch)
        }

    fn ggsw_encrypt_sk_tmp_bytes<A>(module: &::poulpy_hal::layouts::Module<$be>, infos: &A) -> usize
    where
        A: $crate::layouts::GGSWInfos {
            <::poulpy_hal::layouts::Module<$be> as $crate::reference::encryption::GGSWEncryptSkReference<$be, $ring>>::ggsw_encrypt_sk_tmp_bytes_reference::<A>(module, infos)
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
        R: $crate::layouts::GGSWToBackendMut<$be, $ring>
            + $crate::layouts::GGSWInfos
            + $crate::layouts::GGSWAtViewMut<$be, $ring>,
        P: ::poulpy_hal::layouts::ScalarZnxToBackendRef<$be, $ring> + ::poulpy_hal::layouts::ZnxInfos,
        S: $crate::layouts::GLWESecretPreparedToBackendRef<$be, $ring> + $crate::layouts::LWEInfos + $crate::layouts::GLWEInfos {
            <::poulpy_hal::layouts::Module<$be> as $crate::reference::encryption::GGSWEncryptSkReference<$be, $ring>>::ggsw_encrypt_sk_reference::<R, P, S>(module, res, pt, sk, source_xe, source_xa, scratch)
        }

    fn lwe_fill_mask_from_source<R>(module: &::poulpy_hal::layouts::Module<$be>, base2k: usize, res: &mut R, source_xa: &mut ::poulpy_hal::source::Source)
    where
        R: $crate::layouts::LWEToBackendMut<$be, $ring> {
            <::poulpy_hal::layouts::Module<$be> as $crate::reference::encryption::LWEFillMaskReference<$be, $ring>>::lwe_fill_mask_from_source_reference::<R>(module, base2k, res, source_xa)
        }

    fn lwe_encrypt_sk_tmp_bytes<A>(module: &::poulpy_hal::layouts::Module<$be>, infos: &A) -> usize
    where
        A: $crate::layouts::LWEInfos {
            <::poulpy_hal::layouts::Module<$be> as $crate::reference::encryption::LWEEncryptSkReference<$be, $ring>>::lwe_encrypt_sk_tmp_bytes_reference::<A>(module, infos)
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
        R: $crate::layouts::LWEToBackendMut<$be, $ring> + $crate::layouts::LWEInfos,
        P: $crate::layouts::LWEPlaintextToBackendRef<$be, $ring>,
        S: $crate::layouts::LWESecretToBackendRef<$be, $ring>,
        {
            <::poulpy_hal::layouts::Module<$be> as $crate::reference::encryption::LWEEncryptSkReference<$be, $ring>>::lwe_encrypt_sk_reference::<R, P, S>(module, res, pt, sk, source_xe, source_xa, scratch)
        }

    fn gglwe_encrypt_sk_tmp_bytes<A>(module: &::poulpy_hal::layouts::Module<$be>, infos: &A) -> usize
    where
        A: $crate::layouts::GGLWEInfos {
            <::poulpy_hal::layouts::Module<$be> as $crate::reference::encryption::GGLWEEncryptSkReference<$be, $ring>>::gglwe_encrypt_sk_tmp_bytes_reference::<A>(module, infos)
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
        R: $crate::layouts::GGLWEToBackendMut<$be, $ring>,
        P: ::poulpy_hal::layouts::ScalarZnxToBackendRef<$be, $ring>,
        S: $crate::layouts::GLWESecretPreparedToBackendRef<$be, $ring> {
            <::poulpy_hal::layouts::Module<$be> as $crate::reference::encryption::GGLWEEncryptSkReference<$be, $ring>>::gglwe_encrypt_sk_reference::<R, P, S>(module, res, pt, sk, source_xe, source_xa, scratch)
        }

    fn glwe_fill_mask_from_source<R>(
        module: &::poulpy_hal::layouts::Module<$be>,
        res: &mut R,
        source_xa: &mut ::poulpy_hal::source::Source,
    ) where
        R: $crate::layouts::GLWEToBackendMut<$be, $ring> {
            <::poulpy_hal::layouts::Module<$be> as $crate::reference::encryption::GLWEFillReference<$be, $ring>>::glwe_fill_mask_from_source_reference::<R>(module, res, source_xa)
        }

    fn glwe_fill_from_source<R>(
        module: &::poulpy_hal::layouts::Module<$be>,
        res: &mut R,
        source: &mut ::poulpy_hal::source::Source,
    ) where
        R: $crate::layouts::GLWEToBackendMut<$be, $ring> {
            <::poulpy_hal::layouts::Module<$be> as $crate::reference::encryption::GLWEFillReference<$be, $ring>>::glwe_fill_from_source_reference::<R>(module, res, source)
        }

    fn glwe_encrypt_sk_tmp_bytes<A>(module: &::poulpy_hal::layouts::Module<$be>, infos: &A) -> usize
    where
        A: $crate::layouts::GLWEInfos {
            <::poulpy_hal::layouts::Module<$be> as $crate::reference::encryption::GLWEEncryptSkReference<$be, $ring>>::glwe_encrypt_sk_tmp_bytes_reference::<A>(module, infos)
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
        R: $crate::layouts::GLWEToBackendMut<$be, $ring>,
        P: $crate::layouts::GLWEToBackendRef<$be, $ring>,
        S: $crate::layouts::GLWESecretPreparedToBackendRef<$be, $ring> {
            <::poulpy_hal::layouts::Module<$be> as $crate::reference::encryption::GLWEEncryptSkReference<$be, $ring>>::glwe_encrypt_sk_reference::<R, P, S>(module, res, pt, sk, source_xe, source_xa, scratch)
        }

    fn glwe_encrypt_zero_sk<R, S>(
        module: &::poulpy_hal::layouts::Module<$be>,
        res: &mut R,
        sk: &S,
        source_xe: &mut ::poulpy_hal::source::Source,
        source_xa: &mut ::poulpy_hal::source::Source,
        scratch: &mut ::poulpy_hal::layouts::ScratchArena<'_, $be>,
    ) where
        R: $crate::layouts::GLWEToBackendMut<$be, $ring>,
        S: $crate::layouts::GLWESecretPreparedToBackendRef<$be, $ring> {
            <::poulpy_hal::layouts::Module<$be> as $crate::reference::encryption::GLWEEncryptSkReference<$be, $ring>>::glwe_encrypt_zero_sk_reference::<R, S>(module, res, sk, source_xe, source_xa, scratch)
        }

    fn glwe_encrypt_sk_with_mask<R, P, S>(
        module: &::poulpy_hal::layouts::Module<$be>,
        res: &mut R,
        pt: &P,
        sk: &S,
        source_xe: &mut ::poulpy_hal::source::Source,
        scratch: &mut ::poulpy_hal::layouts::ScratchArena<'_, $be>,
    ) where
        R: $crate::layouts::GLWEToBackendMut<$be, $ring>,
        P: $crate::layouts::GLWEToBackendRef<$be, $ring>,
        S: $crate::layouts::GLWESecretPreparedToBackendRef<$be, $ring> {
            <::poulpy_hal::layouts::Module<$be> as $crate::reference::encryption::GLWEEncryptSkReference<$be, $ring>>::glwe_encrypt_sk_with_mask_reference::<R, P, S>(module, res, pt, sk, source_xe, scratch)
        }

    fn glwe_encrypt_pk_tmp_bytes<R, K>(module: &::poulpy_hal::layouts::Module<$be>, res_infos: &R, pk_infos: &K) -> usize
    where
        R: $crate::layouts::GLWEInfos,
        K: $crate::layouts::GLWEInfos {
            <::poulpy_hal::layouts::Module<$be> as $crate::reference::encryption::GLWEEncryptPkReference<$be, $ring>>::glwe_encrypt_pk_tmp_bytes_reference::<R, K>(module, res_infos, pk_infos)
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
        R: $crate::layouts::GLWEToBackendMut<$be, $ring> + $crate::layouts::GLWEInfos,
        P: $crate::layouts::GLWEToBackendRef<$be, $ring> + $crate::layouts::GLWEInfos,
        K: $crate::layouts::GLWEPublicKeyPreparedToBackendRef<$be, $ring> + $crate::layouts::GLWEInfos {
            <::poulpy_hal::layouts::Module<$be> as $crate::reference::encryption::GLWEEncryptPkReference<$be, $ring>>::glwe_encrypt_pk_sampled_at_reference::<R, P, K>(module, res, pt, body_noise, pk, k_sample, source_xu, source_xe, scratch)
        }

    fn gglwe_to_ggsw_key_encrypt_sk_tmp_bytes<A>(module: &::poulpy_hal::layouts::Module<$be>, infos: &A) -> usize
    where
        A: $crate::layouts::GGLWEInfos {
            <::poulpy_hal::layouts::Module<$be> as $crate::reference::encryption::GGLWEToGGSWKeyEncryptSkReference<$be, $ring>>::gglwe_to_ggsw_key_encrypt_sk_tmp_bytes_reference::<A>(module, infos)
        }

    fn gglwe_to_ggsw_key_encrypt_sk<R, S>(
        module: &::poulpy_hal::layouts::Module<$be>,
        res: &mut R,
        sk: &S,
        source_xe: &mut ::poulpy_hal::source::Source,
        source_xa: &mut ::poulpy_hal::source::Source,
        scratch: &mut ::poulpy_hal::layouts::ScratchArena<'_, $be>,
    ) where
        R: $crate::layouts::GGLWEToGGSWKeyToBackendMut<$be, $ring>,
        S: $crate::layouts::GLWESecretToBackendRef<$be, $ring> + $crate::GetDistribution + $crate::layouts::GLWEInfos {
            <::poulpy_hal::layouts::Module<$be> as $crate::reference::encryption::GGLWEToGGSWKeyEncryptSkReference<$be, $ring>>::gglwe_to_ggsw_key_encrypt_sk_reference::<R, S>(module, res, sk, source_xe, source_xa, scratch)
        }

    fn glwe_automorphism_key_encrypt_sk_tmp_bytes<A>(module: &::poulpy_hal::layouts::Module<$be>, infos: &A) -> usize
            where
                A: $crate::layouts::GGLWEInfos,
            {
                <::poulpy_hal::layouts::Module<$be> as $crate::reference::encryption::GLWEAutomorphismKeyEncryptSkReference<
                    $be,
                $ring, >>::glwe_automorphism_key_encrypt_sk_tmp_bytes_reference::<A>(module, infos)
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
                R: $crate::layouts::GGLWEToBackendMut<$be, $ring>
                    + $crate::layouts::SetGaloisElement
                    + $crate::layouts::GGLWEInfos,
                S: $crate::layouts::GLWESecretToBackendRef<$be, $ring> + $crate::layouts::GLWEInfos,
            {
                <::poulpy_hal::layouts::Module<$be> as $crate::reference::encryption::GLWEAutomorphismKeyEncryptSkReference<
                    $be,
                $ring, >>::glwe_automorphism_key_encrypt_sk_reference::<R, S>(
                    module, res, p, sk, source_xe, source_xa, scratch,
                )
            }

    fn glwe_compressed_encrypt_sk_tmp_bytes<A>(module: &::poulpy_hal::layouts::Module<$be>, infos: &A) -> usize
    where
        A: $crate::layouts::GLWEInfos {
            <::poulpy_hal::layouts::Module<$be> as $crate::reference::encryption::GLWECompressedEncryptSkReference<$be, $ring>>::glwe_compressed_encrypt_sk_tmp_bytes_reference::<A>(module, infos)
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
        R: $crate::layouts::GLWECompressedToBackendMut<$be, $ring> + $crate::layouts::GLWECompressedSeedMut,
        P: $crate::layouts::GLWEToBackendRef<$be, $ring>,
        S: $crate::layouts::GLWESecretPreparedToBackendRef<$be, $ring> {
            <::poulpy_hal::layouts::Module<$be> as $crate::reference::encryption::GLWECompressedEncryptSkReference<$be, $ring>>::glwe_compressed_encrypt_sk_reference::<R, P, S>(module, res, pt, sk, seed_xa, source_xe, scratch)
        }

    fn glwe_compressed_encrypt_zero_sk<R, S>(
        module: &::poulpy_hal::layouts::Module<$be>,
        res: &mut R,
        sk: &S,
        seed_xa: [u8; 32],
        source_xe: &mut ::poulpy_hal::source::Source,
        scratch: &mut ::poulpy_hal::layouts::ScratchArena<'_, $be>,
    ) where
        R: $crate::layouts::GLWECompressedToBackendMut<$be, $ring> + $crate::layouts::GLWECompressedSeedMut,
        S: $crate::layouts::GLWESecretPreparedToBackendRef<$be, $ring> {
            <::poulpy_hal::layouts::Module<$be> as $crate::reference::encryption::GLWECompressedEncryptSkReference<$be, $ring>>::glwe_compressed_encrypt_zero_sk_reference::<R, S>(module, res, sk, seed_xa, source_xe, scratch)
        }

    fn glwe_switching_key_compressed_encrypt_sk_tmp_bytes<A>(module: &::poulpy_hal::layouts::Module<$be>, infos: &A) -> usize
    where
        A: $crate::layouts::GGLWEInfos {
            <::poulpy_hal::layouts::Module<$be> as $crate::reference::encryption::GLWESwitchingKeyCompressedEncryptSkReference<$be, $ring>>::glwe_switching_key_compressed_encrypt_sk_tmp_bytes_reference::<A>(module, infos)
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
        R: $crate::layouts::GGLWECompressedToBackendMut<$be, $ring> + $crate::layouts::GGLWECompressedSeedMut + $crate::layouts::GLWESwitchingKeyDegreesMut + $crate::layouts::GGLWEInfos,
        S1: $crate::layouts::GLWESecretToBackendRef<$be, $ring> + $crate::layouts::GLWEInfos,
        S2: $crate::layouts::GLWESecretToBackendRef<$be, $ring> + $crate::GetDistribution + $crate::layouts::GLWEInfos {
            <::poulpy_hal::layouts::Module<$be> as $crate::reference::encryption::GLWESwitchingKeyCompressedEncryptSkReference<$be, $ring>>::glwe_switching_key_compressed_encrypt_sk_reference::<R, S1, S2>(module, res, sk_in, sk_out, seed_xa, source_xe, scratch)
        }

    fn ggsw_compressed_encrypt_sk_tmp_bytes<A>(module: &::poulpy_hal::layouts::Module<$be>, infos: &A) -> usize
    where
        A: $crate::layouts::GGSWInfos {
            <::poulpy_hal::layouts::Module<$be> as $crate::reference::encryption::GGSWCompressedEncryptSkReference<$be, $ring>>::ggsw_compressed_encrypt_sk_tmp_bytes_reference::<A>(module, infos)
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
        R: $crate::layouts::GGSWCompressedToBackendMut<$be, $ring>
            + $crate::layouts::GGSWCompressedSeedMut
            + $crate::layouts::GGSWInfos,
        P: ::poulpy_hal::layouts::ScalarZnxToBackendRef<$be, $ring>,
        S: $crate::layouts::GLWESecretPreparedToBackendRef<$be, $ring> {
            <::poulpy_hal::layouts::Module<$be> as $crate::reference::encryption::GGSWCompressedEncryptSkReference<$be, $ring>>::ggsw_compressed_encrypt_sk_reference::<R, P, S>(module, res, pt, sk, seed_xa, source_xe, scratch)
        }

    fn gglwe_compressed_encrypt_sk_tmp_bytes<A>(module: &::poulpy_hal::layouts::Module<$be>, infos: &A) -> usize
    where
        A: $crate::layouts::GGLWEInfos {
            <::poulpy_hal::layouts::Module<$be> as $crate::reference::encryption::GGLWECompressedEncryptSkReference<$be, $ring>>::gglwe_compressed_encrypt_sk_tmp_bytes_reference::<A>(module, infos)
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
        R: $crate::layouts::GGLWECompressedToBackendMut<$be, $ring> + $crate::layouts::GGLWECompressedSeedMut,
        P: ::poulpy_hal::layouts::ScalarZnxToBackendRef<$be, $ring>,
        S: $crate::layouts::GLWESecretPreparedToBackendRef<$be, $ring> {
            <::poulpy_hal::layouts::Module<$be> as $crate::reference::encryption::GGLWECompressedEncryptSkReference<$be, $ring>>::gglwe_compressed_encrypt_sk_reference::<R, P, S>(module, res, pt, sk, seed, source_xe, scratch)
        }

    fn gglwe_to_ggsw_key_compressed_encrypt_sk_tmp_bytes<A>(module: &::poulpy_hal::layouts::Module<$be>, infos: &A) -> usize
    where
        A: $crate::layouts::GGLWEInfos {
            <::poulpy_hal::layouts::Module<$be> as $crate::reference::encryption::GGLWEToGGSWKeyCompressedEncryptSkReference<$be, $ring>>::gglwe_to_ggsw_key_compressed_encrypt_sk_tmp_bytes_reference::<A>(module, infos)
        }

    fn gglwe_to_ggsw_key_compressed_encrypt_sk<R, S>(
        module: &::poulpy_hal::layouts::Module<$be>,
        res: &mut R,
        sk: &S,
        seed_xa: [u8; 32],
        source_xe: &mut ::poulpy_hal::source::Source,
        scratch: &mut ::poulpy_hal::layouts::ScratchArena<'_, $be>,
    ) where
        R: $crate::layouts::GGLWEToGGSWKeyCompressedToBackendMut<$be, $ring> + $crate::layouts::GGLWEInfos,
        S: $crate::layouts::GLWESecretToBackendRef<$be, $ring> + $crate::GetDistribution + $crate::layouts::GLWEInfos {
            <::poulpy_hal::layouts::Module<$be> as $crate::reference::encryption::GGLWEToGGSWKeyCompressedEncryptSkReference<$be, $ring>>::gglwe_to_ggsw_key_compressed_encrypt_sk_reference::<R, S>(module, res, sk, seed_xa, source_xe, scratch)
        }

    fn glwe_automorphism_key_compressed_encrypt_sk_tmp_bytes<A>(module: &::poulpy_hal::layouts::Module<$be>, infos: &A) -> usize
    where
        A: $crate::layouts::GGLWEInfos {
            <::poulpy_hal::layouts::Module<$be> as $crate::reference::encryption::GLWEAutomorphismKeyCompressedEncryptSkReference<$be, $ring>>::glwe_automorphism_key_compressed_encrypt_sk_tmp_bytes_reference::<A>(module, infos)
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
        R: $crate::layouts::GGLWECompressedToBackendMut<$be, $ring> + $crate::layouts::GGLWECompressedSeedMut + $crate::layouts::SetGaloisElement + $crate::layouts::GGLWEInfos,
        S: $crate::layouts::GLWESecretToBackendRef<$be, $ring> + $crate::layouts::GLWEInfos {
            <::poulpy_hal::layouts::Module<$be> as $crate::reference::encryption::GLWEAutomorphismKeyCompressedEncryptSkReference<$be, $ring>>::glwe_automorphism_key_compressed_encrypt_sk_reference::<R, S>(module, res, p, sk, seed_xa, source_xe, scratch)
        }
        }
    };
}

/// Selects the HAL-based encryption algorithms of the LWE keys.
#[macro_export]
macro_rules! impl_lwe_key_encryption_reference_full {
    ($be:ty) => {
        unsafe impl $crate::oep::LWEKeyEncryptionImpl for $be {
            fn glwe_to_lwe_key_encrypt_sk_tmp_bytes<A>(module: &::poulpy_hal::layouts::Module<$be>, infos: &A) -> usize
            where
                A: $crate::layouts::GGLWEInfos,
            {
                <::poulpy_hal::layouts::Module<$be> as $crate::reference::encryption::GLWEToLWEKeyEncryptSkReference<$be>>::glwe_to_lwe_key_encrypt_sk_tmp_bytes_reference(module, infos)
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
                <::poulpy_hal::layouts::Module<$be> as $crate::reference::encryption::GLWEToLWEKeyEncryptSkReference<$be>>::glwe_to_lwe_key_encrypt_sk_reference(
                    module, res, sk_lwe, sk_glwe, source_xe, source_xa, scratch,
                )
            }

            fn lwe_switching_key_encrypt_sk_tmp_bytes<A>(module: &::poulpy_hal::layouts::Module<$be>, infos: &A) -> usize
            where
                A: $crate::layouts::GGLWEInfos,
            {
                <::poulpy_hal::layouts::Module<$be> as $crate::reference::encryption::LWESwitchingKeyEncryptSkReference<$be>>::lwe_switching_key_encrypt_sk_tmp_bytes_reference(module, infos)
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
                S2: $crate::layouts::LWESecretToBackendRef<$be>,
            {
                <::poulpy_hal::layouts::Module<$be> as $crate::reference::encryption::LWESwitchingKeyEncryptSkReference<$be>>::lwe_switching_key_encrypt_sk_reference(
                    module, res, sk_lwe_in, sk_lwe_out, source_xe, source_xa, scratch,
                )
            }

            fn lwe_to_glwe_key_encrypt_sk_tmp_bytes<A>(module: &::poulpy_hal::layouts::Module<$be>, infos: &A) -> usize
            where
                A: $crate::layouts::GGLWEInfos,
            {
                <::poulpy_hal::layouts::Module<$be> as $crate::reference::encryption::LWEToGLWEKeyEncryptSkReference<$be>>::lwe_to_glwe_key_encrypt_sk_tmp_bytes_reference(module, infos)
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
                <::poulpy_hal::layouts::Module<$be> as $crate::reference::encryption::LWEToGLWEKeyEncryptSkReference<$be>>::lwe_to_glwe_key_encrypt_sk_reference(
                    module, res, sk_lwe, sk_glwe, source_xe, source_xa, scratch,
                )
            }
        }
    };
}
