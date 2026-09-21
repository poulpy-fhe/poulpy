#![allow(clippy::too_many_arguments)]

use poulpy_hal::{
    layouts::{Backend, Module, ScalarZnxToBackendRef, ScratchArena, ZnxInfos},
    source::Source,
};

use crate::{
    EncryptionInfos, GetDistribution, GetDistributionMut,
    layouts::{
        GGLWECompressedSeedMut, GGLWECompressedToBackendMut, GGLWEInfos, GGLWEToBackendMut, GGLWEToGGSWKeyCompressedToBackendMut,
        GGLWEToGGSWKeyToBackendMut, GGSWAtViewMut, GGSWCompressedSeedMut, GGSWCompressedToBackendMut, GGSWInfos,
        GGSWToBackendMut, GLWECompressedSeedMut, GLWECompressedToBackendMut, GLWEInfos, GLWESecretToBackendRef,
        GLWESwitchingKeyDegreesMut, GLWEToBackendMut, GLWEToBackendRef, LWEInfos, LWEPlaintextToBackendRef,
        LWESecretToBackendRef, LWEToBackendMut, SetGaloisElement,
        prepared::{GLWEPreparedToBackendRef, GLWESecretPreparedToBackendRef},
    },
};

/// # Safety
///
/// Implementations must satisfy the contracts of all trait methods, including
/// any HAL-level invariants (alignment, layout, scratch sizing) implied by the
/// associated method signatures.
pub unsafe trait EncryptionImpl: Backend {
    fn fill_glwe_mask_from_source<R>(
        module: &Module<Self>,
        base2k: usize,
        res: &mut R,
        res_col: usize,
        rank: usize,
        source_xa: &mut Source,
    ) where
        R: GLWEToBackendMut<Self>;

    fn fill_glwe_mask_from_seed<R>(
        module: &Module<Self>,
        base2k: usize,
        res: &mut R,
        res_col: usize,
        rank: usize,
        seed_xa: [u8; 32],
    ) where
        R: GLWEToBackendMut<Self>,
    {
        super::derived::encryption::fill_glwe_mask_from_seed_derived(module, base2k, res, res_col, rank, seed_xa)
    }

    fn fill_lwe_mask_from_source<R>(module: &Module<Self>, base2k: usize, res: &mut R, source_xa: &mut Source)
    where
        R: LWEToBackendMut<Self>;

    fn fill_lwe_mask_from_seed<R>(module: &Module<Self>, base2k: usize, res: &mut R, seed_xa: [u8; 32])
    where
        R: LWEToBackendMut<Self>,
    {
        super::derived::encryption::fill_lwe_mask_from_seed_derived(module, base2k, res, seed_xa)
    }

    fn lwe_encrypt_sk_tmp_bytes<A>(module: &Module<Self>, infos: &A) -> usize
    where
        A: LWEInfos;

    fn lwe_encrypt_sk<R, P, S, E>(
        module: &Module<Self>,
        res: &mut R,
        pt: &P,
        sk: &S,
        enc_infos: &E,
        source_xe: &mut Source,
        source_xa: &mut Source,
        scratch: &mut ScratchArena<'_, Self>,
    ) where
        R: LWEToBackendMut<Self> + LWEInfos,
        P: LWEPlaintextToBackendRef<Self>,
        S: LWESecretToBackendRef<Self>,
        E: EncryptionInfos;

    fn glwe_encrypt_sk_tmp_bytes<A>(module: &Module<Self>, infos: &A) -> usize
    where
        A: GLWEInfos;

    fn glwe_encrypt_sk<R, P, S, E>(
        module: &Module<Self>,
        res: &mut R,
        pt: &P,
        sk: &S,
        enc_infos: &E,
        source_xe: &mut Source,
        source_xa: &mut Source,
        scratch: &mut ScratchArena<'_, Self>,
    ) where
        R: GLWEToBackendMut<Self>,
        P: GLWEToBackendRef<Self>,
        E: EncryptionInfos,
        S: GLWESecretPreparedToBackendRef<Self>;

    fn glwe_encrypt_zero_sk<R, E, S>(
        module: &Module<Self>,
        res: &mut R,
        sk: &S,
        enc_infos: &E,
        source_xe: &mut Source,
        source_xa: &mut Source,
        scratch: &mut ScratchArena<'_, Self>,
    ) where
        R: GLWEToBackendMut<Self>,
        E: EncryptionInfos,
        S: GLWESecretPreparedToBackendRef<Self>;

    fn glwe_encrypt_pk_tmp_bytes<A>(module: &Module<Self>, infos: &A) -> usize
    where
        A: GLWEInfos;

    fn glwe_encrypt_pk<R, P, K, E>(
        module: &Module<Self>,
        res: &mut R,
        pt: &P,
        pk: &K,
        enc_infos: &E,
        source_xu: &mut Source,
        source_xe: &mut Source,
        scratch: &mut ScratchArena<'_, Self>,
    ) where
        R: GLWEToBackendMut<Self> + GLWEInfos,
        P: GLWEToBackendRef<Self> + GLWEInfos,
        E: EncryptionInfos,
        K: GLWEPreparedToBackendRef<Self> + GetDistribution + GLWEInfos;

    fn glwe_encrypt_zero_pk<R, K, E>(
        module: &Module<Self>,
        res: &mut R,
        pk: &K,
        enc_infos: &E,
        source_xu: &mut Source,
        source_xe: &mut Source,
        scratch: &mut ScratchArena<'_, Self>,
    ) where
        R: GLWEToBackendMut<Self> + GLWEInfos,
        E: EncryptionInfos,
        K: GLWEPreparedToBackendRef<Self> + GetDistribution + GLWEInfos;

    fn glwe_public_key_generate<R, S, E>(
        module: &Module<Self>,
        res: &mut R,
        sk: &S,
        enc_infos: &E,
        source_xe: &mut Source,
        source_xa: &mut Source,
    ) where
        R: GLWEToBackendMut<Self> + GetDistributionMut + GLWEInfos,
        E: EncryptionInfos,
        S: GLWESecretPreparedToBackendRef<Self> + GetDistribution,
    {
        super::derived::encryption::glwe_public_key_generate_derived(module, res, sk, enc_infos, source_xe, source_xa)
    }

    fn gglwe_encrypt_sk_tmp_bytes<A>(module: &Module<Self>, infos: &A) -> usize
    where
        A: GGLWEInfos;

    fn gglwe_encrypt_sk<R, P, S, E>(
        module: &Module<Self>,
        res: &mut R,
        pt: &P,
        sk: &S,
        enc_infos: &E,
        source_xe: &mut Source,
        source_xa: &mut Source,
        scratch: &mut ScratchArena<'_, Self>,
    ) where
        R: GGLWEToBackendMut<Self>,
        P: ScalarZnxToBackendRef<Self>,
        E: EncryptionInfos,
        S: GLWESecretPreparedToBackendRef<Self>;

    fn ggsw_encrypt_sk_tmp_bytes<A>(module: &Module<Self>, infos: &A) -> usize
    where
        A: GGSWInfos;

    fn ggsw_encrypt_sk<R, P, S, E>(
        module: &Module<Self>,
        res: &mut R,
        pt: &P,
        sk: &S,
        enc_infos: &E,
        source_xe: &mut Source,
        source_xa: &mut Source,
        scratch: &mut ScratchArena<'_, Self>,
    ) where
        R: GGSWToBackendMut<Self> + GGSWInfos + GGSWAtViewMut<Self>,
        P: ScalarZnxToBackendRef<Self> + ZnxInfos,
        E: EncryptionInfos,
        S: GLWESecretPreparedToBackendRef<Self> + LWEInfos + GLWEInfos;

    fn gglwe_to_ggsw_key_encrypt_sk_tmp_bytes<A>(module: &Module<Self>, infos: &A) -> usize
    where
        A: GGLWEInfos;

    fn gglwe_to_ggsw_key_encrypt_sk<R, S, E>(
        module: &Module<Self>,
        res: &mut R,
        sk: &S,
        enc_infos: &E,
        source_xe: &mut Source,
        source_xa: &mut Source,
        scratch: &mut ScratchArena<'_, Self>,
    ) where
        R: GGLWEToGGSWKeyToBackendMut<Self>,
        E: EncryptionInfos,
        S: GLWESecretToBackendRef<Self> + GetDistribution + GLWEInfos;

    fn glwe_switching_key_encrypt_sk_tmp_bytes<A>(module: &Module<Self>, infos: &A) -> usize
    where
        A: GGLWEInfos;

    fn glwe_switching_key_encrypt_sk<R, S1, S2, E>(
        module: &Module<Self>,
        res: &mut R,
        sk_in: &S1,
        sk_out: &S2,
        enc_infos: &E,
        source_xe: &mut Source,
        source_xa: &mut Source,
        scratch: &mut ScratchArena<'_, Self>,
    ) where
        R: GGLWEToBackendMut<Self> + GLWESwitchingKeyDegreesMut + GGLWEInfos,
        E: EncryptionInfos,
        S1: GLWESecretToBackendRef<Self> + GLWEInfos,
        S2: GLWESecretToBackendRef<Self> + GetDistribution + GLWEInfos;

    fn glwe_tensor_key_encrypt_sk_tmp_bytes<A>(module: &Module<Self>, infos: &A) -> usize
    where
        A: GGLWEInfos,
        Module<Self>: crate::layouts::GLWESecretPreparedFactory<Self> + crate::layouts::GLWESecretTensorFactory<Self>,
    {
        super::derived::encryption::glwe_tensor_key_encrypt_sk_tmp_bytes_derived(module, infos)
    }

    fn glwe_tensor_key_encrypt_sk<R, S, E>(
        module: &Module<Self>,
        res: &mut R,
        sk: &S,
        enc_infos: &E,
        source_xe: &mut Source,
        source_xa: &mut Source,
        scratch: &mut ScratchArena<'_, Self>,
    ) where
        R: GGLWEToBackendMut<Self> + GGLWEInfos,
        E: EncryptionInfos,
        S: GLWESecretToBackendRef<Self> + GetDistribution + GLWEInfos,
        Module<Self>: crate::layouts::GLWESecretPreparedFactory<Self> + crate::layouts::GLWESecretTensorFactory<Self>,
    {
        super::derived::encryption::glwe_tensor_key_encrypt_sk_derived(module, res, sk, enc_infos, source_xe, source_xa, scratch)
    }

    fn glwe_to_lwe_key_encrypt_sk_tmp_bytes<A>(module: &Module<Self>, infos: &A) -> usize
    where
        A: GGLWEInfos;

    fn glwe_to_lwe_key_encrypt_sk<R, S1, S2, E>(
        module: &Module<Self>,
        res: &mut R,
        sk_lwe: &S1,
        sk_glwe: &S2,
        enc_infos: &E,
        source_xe: &mut Source,
        source_xa: &mut Source,
        scratch: &mut ScratchArena<'_, Self>,
    ) where
        S1: LWESecretToBackendRef<Self>,
        S2: GLWESecretToBackendRef<Self>,
        E: EncryptionInfos,
        R: GGLWEToBackendMut<Self> + GGLWEInfos;

    fn lwe_switching_key_encrypt_sk_tmp_bytes<A>(module: &Module<Self>, infos: &A) -> usize
    where
        A: GGLWEInfos;

    fn lwe_switching_key_encrypt_sk<R, S1, S2, E>(
        module: &Module<Self>,
        res: &mut R,
        sk_lwe_in: &S1,
        sk_lwe_out: &S2,
        enc_infos: &E,
        source_xe: &mut Source,
        source_xa: &mut Source,
        scratch: &mut ScratchArena<'_, Self>,
    ) where
        R: GGLWEToBackendMut<Self> + GLWESwitchingKeyDegreesMut + GGLWEInfos,
        E: EncryptionInfos,
        S1: LWESecretToBackendRef<Self>,
        S2: LWESecretToBackendRef<Self>;

    fn lwe_to_glwe_key_encrypt_sk_tmp_bytes<A>(module: &Module<Self>, infos: &A) -> usize
    where
        A: GGLWEInfos;

    fn lwe_to_glwe_key_encrypt_sk<R, S1, S2, E>(
        module: &Module<Self>,
        res: &mut R,
        sk_lwe: &S1,
        sk_glwe: &S2,
        enc_infos: &E,
        source_xe: &mut Source,
        source_xa: &mut Source,
        scratch: &mut ScratchArena<'_, Self>,
    ) where
        S1: LWESecretToBackendRef<Self>,
        S2: GLWESecretPreparedToBackendRef<Self>,
        E: EncryptionInfos,
        R: GGLWEToBackendMut<Self> + GGLWEInfos;

    fn glwe_automorphism_key_encrypt_sk_tmp_bytes<A>(module: &Module<Self>, infos: &A) -> usize
    where
        A: GGLWEInfos;

    fn glwe_automorphism_key_encrypt_sk<R, S, E>(
        module: &Module<Self>,
        res: &mut R,
        p: i64,
        sk: &S,
        enc_infos: &E,
        source_xe: &mut Source,
        source_xa: &mut Source,
        scratch: &mut ScratchArena<'_, Self>,
    ) where
        R: GGLWEToBackendMut<Self> + SetGaloisElement + GGLWEInfos,
        E: EncryptionInfos,
        S: GLWESecretToBackendRef<Self> + GLWEInfos;

    fn glwe_compressed_encrypt_sk_tmp_bytes<A>(module: &Module<Self>, infos: &A) -> usize
    where
        A: GLWEInfos;

    fn glwe_compressed_encrypt_sk<R, P, S, E>(
        module: &Module<Self>,
        res: &mut R,
        pt: &P,
        sk: &S,
        seed_xa: [u8; 32],
        enc_infos: &E,
        source_xe: &mut Source,
        scratch: &mut ScratchArena<'_, Self>,
    ) where
        R: GLWECompressedToBackendMut<Self> + GLWECompressedSeedMut,
        P: GLWEToBackendRef<Self>,
        E: EncryptionInfos,
        S: GLWESecretPreparedToBackendRef<Self>;

    fn gglwe_compressed_encrypt_sk_tmp_bytes<A>(module: &Module<Self>, infos: &A) -> usize
    where
        A: GGLWEInfos;

    fn gglwe_compressed_encrypt_sk<R, P, S, E>(
        module: &Module<Self>,
        res: &mut R,
        pt: &P,
        sk: &S,
        seed: [u8; 32],
        enc_infos: &E,
        source_xe: &mut Source,
        scratch: &mut ScratchArena<'_, Self>,
    ) where
        R: GGLWECompressedToBackendMut<Self> + GGLWECompressedSeedMut,
        P: ScalarZnxToBackendRef<Self>,
        E: EncryptionInfos,
        S: GLWESecretPreparedToBackendRef<Self>;

    fn ggsw_compressed_encrypt_sk_tmp_bytes<A>(module: &Module<Self>, infos: &A) -> usize
    where
        A: GGSWInfos;

    fn ggsw_compressed_encrypt_sk<R, P, S, E>(
        module: &Module<Self>,
        res: &mut R,
        pt: &P,
        sk: &S,
        seed_xa: [u8; 32],
        enc_infos: &E,
        source_xe: &mut Source,
        scratch: &mut ScratchArena<'_, Self>,
    ) where
        R: GGSWCompressedToBackendMut<Self> + GGSWCompressedSeedMut + GGSWInfos,
        P: ScalarZnxToBackendRef<Self>,
        E: EncryptionInfos,
        S: GLWESecretPreparedToBackendRef<Self>;

    fn gglwe_to_ggsw_key_compressed_encrypt_sk_tmp_bytes<A>(module: &Module<Self>, infos: &A) -> usize
    where
        A: GGLWEInfos;

    fn gglwe_to_ggsw_key_compressed_encrypt_sk<R, S, E>(
        module: &Module<Self>,
        res: &mut R,
        sk: &S,
        seed_xa: [u8; 32],
        enc_infos: &E,
        source_xe: &mut Source,
        scratch: &mut ScratchArena<'_, Self>,
    ) where
        R: GGLWEToGGSWKeyCompressedToBackendMut<Self> + GGLWEInfos,
        E: EncryptionInfos,
        S: GLWESecretToBackendRef<Self> + GetDistribution + GLWEInfos;

    fn glwe_automorphism_key_compressed_encrypt_sk_tmp_bytes<A>(module: &Module<Self>, infos: &A) -> usize
    where
        A: GGLWEInfos;

    fn glwe_automorphism_key_compressed_encrypt_sk<R, S, E>(
        module: &Module<Self>,
        res: &mut R,
        p: i64,
        sk: &S,
        seed_xa: [u8; 32],
        enc_infos: &E,
        source_xe: &mut Source,
        scratch: &mut ScratchArena<'_, Self>,
    ) where
        R: GGLWECompressedToBackendMut<Self> + GGLWECompressedSeedMut + SetGaloisElement + GGLWEInfos,
        E: EncryptionInfos,
        S: GLWESecretToBackendRef<Self> + GLWEInfos;

    fn glwe_switching_key_compressed_encrypt_sk_tmp_bytes<A>(module: &Module<Self>, infos: &A) -> usize
    where
        A: GGLWEInfos;

    fn glwe_switching_key_compressed_encrypt_sk<R, S1, S2, E>(
        module: &Module<Self>,
        res: &mut R,
        sk_in: &S1,
        sk_out: &S2,
        seed_xa: [u8; 32],
        enc_infos: &E,
        source_xe: &mut Source,
        scratch: &mut ScratchArena<'_, Self>,
    ) where
        R: GGLWECompressedToBackendMut<Self> + GGLWECompressedSeedMut + GLWESwitchingKeyDegreesMut + GGLWEInfos,
        E: EncryptionInfos,
        S1: GLWESecretToBackendRef<Self> + GLWEInfos,
        S2: GLWESecretToBackendRef<Self> + GetDistribution + GLWEInfos;

    fn glwe_tensor_key_compressed_encrypt_sk_tmp_bytes<A>(module: &Module<Self>, infos: &A) -> usize
    where
        A: GGLWEInfos,
        Module<Self>: crate::layouts::GLWESecretPreparedFactory<Self> + crate::layouts::GLWESecretTensorFactory<Self>,
    {
        super::derived::encryption::glwe_tensor_key_compressed_encrypt_sk_tmp_bytes_derived(module, infos)
    }

    fn glwe_tensor_key_compressed_encrypt_sk<R, S, E>(
        module: &Module<Self>,
        res: &mut R,
        sk: &S,
        seed_xa: [u8; 32],
        enc_infos: &E,
        source_xe: &mut Source,
        scratch: &mut ScratchArena<'_, Self>,
    ) where
        R: GGLWECompressedToBackendMut<Self> + GGLWEInfos + GGLWECompressedSeedMut,
        E: EncryptionInfos,
        S: GLWESecretToBackendRef<Self> + GetDistribution + GLWEInfos,
        Module<Self>: crate::layouts::GLWESecretPreparedFactory<Self> + crate::layouts::GLWESecretTensorFactory<Self>,
    {
        super::derived::encryption::glwe_tensor_key_compressed_encrypt_sk_derived(
            module, res, sk, seed_xa, enc_infos, source_xe, scratch,
        )
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

    fn lwe_switching_key_encrypt_sk<R, S1, S2, E>(
        module: &::poulpy_hal::layouts::Module<$be>,
        res: &mut R,
        sk_lwe_in: &S1,
        sk_lwe_out: &S2,
        enc_infos: &E,
        source_xe: &mut ::poulpy_hal::source::Source,
        source_xa: &mut ::poulpy_hal::source::Source,
        scratch: &mut ::poulpy_hal::layouts::ScratchArena<'_, $be>,
    ) where
        R: $crate::layouts::GGLWEToBackendMut<$be> + $crate::layouts::GLWESwitchingKeyDegreesMut + $crate::layouts::GGLWEInfos,
        E: $crate::api::EncryptionInfos,
        S1: $crate::layouts::LWESecretToBackendRef<$be>,
        S2: $crate::layouts::LWESecretToBackendRef<$be> {
            <::poulpy_hal::layouts::Module<$be> as $crate::reference::encryption::LWESwitchingKeyEncryptReference<$be>>::lwe_switching_key_encrypt_sk_reference::<R, S1, S2, E>(module, res, sk_lwe_in, sk_lwe_out, enc_infos, source_xe, source_xa, scratch)
        }

    fn glwe_to_lwe_key_encrypt_sk_tmp_bytes<A>(module: &::poulpy_hal::layouts::Module<$be>, infos: &A) -> usize
            where
                A: $crate::layouts::GGLWEInfos,
            {
                <::poulpy_hal::layouts::Module<$be> as $crate::reference::encryption::GLWEToLWESwitchingKeyEncryptSkReference<
                    $be,
                >>::glwe_to_lwe_key_encrypt_sk_tmp_bytes_reference::<A>(module, infos)
            }

    fn glwe_to_lwe_key_encrypt_sk<R, S1, S2, E>(
                module: &::poulpy_hal::layouts::Module<$be>,
                res: &mut R,
                sk_lwe: &S1,
                sk_glwe: &S2,
                enc_infos: &E,
                source_xe: &mut ::poulpy_hal::source::Source,
                source_xa: &mut ::poulpy_hal::source::Source,
                scratch: &mut ::poulpy_hal::layouts::ScratchArena<'_, $be>,
            ) where
                S1: $crate::layouts::LWESecretToBackendRef<$be>,
                S2: $crate::layouts::GLWESecretToBackendRef<$be>,
                E: $crate::api::EncryptionInfos,
                R: $crate::layouts::GGLWEToBackendMut<$be> + $crate::layouts::GGLWEInfos,
            {
                <::poulpy_hal::layouts::Module<$be> as $crate::reference::encryption::GLWEToLWESwitchingKeyEncryptSkReference<
                    $be,
                >>::glwe_to_lwe_key_encrypt_sk_reference::<R, S1, S2, E>(
                    module, res, sk_lwe, sk_glwe, enc_infos, source_xe, source_xa, scratch,
                )
            }

    fn glwe_switching_key_encrypt_sk_tmp_bytes<A>(module: &::poulpy_hal::layouts::Module<$be>, infos: &A) -> usize
    where
        A: $crate::layouts::GGLWEInfos {
            <::poulpy_hal::layouts::Module<$be> as $crate::reference::encryption::GLWESwitchingKeyEncryptSkReference<$be>>::glwe_switching_key_encrypt_sk_tmp_bytes_reference::<A>(module, infos)
        }

    fn glwe_switching_key_encrypt_sk<R, S1, S2, E>(
        module: &::poulpy_hal::layouts::Module<$be>,
        res: &mut R,
        sk_in: &S1,
        sk_out: &S2,
        enc_infos: &E,
        source_xe: &mut ::poulpy_hal::source::Source,
        source_xa: &mut ::poulpy_hal::source::Source,
        scratch: &mut ::poulpy_hal::layouts::ScratchArena<'_, $be>,
    ) where
        R: $crate::layouts::GGLWEToBackendMut<$be> + $crate::layouts::GLWESwitchingKeyDegreesMut + $crate::layouts::GGLWEInfos,
        E: $crate::api::EncryptionInfos,
        S1: $crate::layouts::GLWESecretToBackendRef<$be> + $crate::layouts::GLWEInfos,
        S2: $crate::layouts::GLWESecretToBackendRef<$be> + $crate::GetDistribution + $crate::layouts::GLWEInfos {
            <::poulpy_hal::layouts::Module<$be> as $crate::reference::encryption::GLWESwitchingKeyEncryptSkReference<$be>>::glwe_switching_key_encrypt_sk_reference::<R, S1, S2, E>(module, res, sk_in, sk_out, enc_infos, source_xe, source_xa, scratch)
        }

    fn ggsw_encrypt_sk_tmp_bytes<A>(module: &::poulpy_hal::layouts::Module<$be>, infos: &A) -> usize
    where
        A: $crate::layouts::GGSWInfos {
            <::poulpy_hal::layouts::Module<$be> as $crate::reference::encryption::GGSWEncryptSkReference<$be>>::ggsw_encrypt_sk_tmp_bytes_reference::<A>(module, infos)
        }

    fn ggsw_encrypt_sk<R, P, S, E>(
        module: &::poulpy_hal::layouts::Module<$be>,
        res: &mut R,
        pt: &P,
        sk: &S,
        enc_infos: &E,
        source_xe: &mut ::poulpy_hal::source::Source,
        source_xa: &mut ::poulpy_hal::source::Source,
        scratch: &mut ::poulpy_hal::layouts::ScratchArena<'_, $be>,
    ) where
        R: $crate::layouts::GGSWToBackendMut<$be> + $crate::layouts::GGSWInfos + $crate::layouts::GGSWAtViewMut<$be>,
        P: ::poulpy_hal::layouts::ScalarZnxToBackendRef<$be> + ::poulpy_hal::layouts::ZnxInfos,
        E: $crate::api::EncryptionInfos,
        S: $crate::layouts::GLWESecretPreparedToBackendRef<$be> + $crate::layouts::LWEInfos + $crate::layouts::GLWEInfos {
            <::poulpy_hal::layouts::Module<$be> as $crate::reference::encryption::GGSWEncryptSkReference<$be>>::ggsw_encrypt_sk_reference::<R, P, S, E>(module, res, pt, sk, enc_infos, source_xe, source_xa, scratch)
        }

    fn lwe_to_glwe_key_encrypt_sk_tmp_bytes<A>(module: &::poulpy_hal::layouts::Module<$be>, infos: &A) -> usize
            where
                A: $crate::layouts::GGLWEInfos,
            {
                <::poulpy_hal::layouts::Module<$be> as $crate::reference::encryption::LWEToGLWESwitchingKeyEncryptSkReference<
                    $be,
                >>::lwe_to_glwe_key_encrypt_sk_tmp_bytes_reference::<A>(module, infos)
            }

    fn lwe_to_glwe_key_encrypt_sk<R, S1, S2, E>(
                module: &::poulpy_hal::layouts::Module<$be>,
                res: &mut R,
                sk_lwe: &S1,
                sk_glwe: &S2,
                enc_infos: &E,
                source_xe: &mut ::poulpy_hal::source::Source,
                source_xa: &mut ::poulpy_hal::source::Source,
                scratch: &mut ::poulpy_hal::layouts::ScratchArena<'_, $be>,
            ) where
                S1: $crate::layouts::LWESecretToBackendRef<$be>,
                S2: $crate::layouts::GLWESecretPreparedToBackendRef<$be>,
                E: $crate::api::EncryptionInfos,
                R: $crate::layouts::GGLWEToBackendMut<$be> + $crate::layouts::GGLWEInfos,
            {
                <::poulpy_hal::layouts::Module<$be> as $crate::reference::encryption::LWEToGLWESwitchingKeyEncryptSkReference<
                    $be,
                >>::lwe_to_glwe_key_encrypt_sk_reference::<R, S1, S2, E>(
                    module, res, sk_lwe, sk_glwe, enc_infos, source_xe, source_xa, scratch,
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

    fn lwe_encrypt_sk<R, P, S, E>(
        module: &::poulpy_hal::layouts::Module<$be>,
        res: &mut R,
        pt: &P,
        sk: &S,
        enc_infos: &E,
        source_xe: &mut ::poulpy_hal::source::Source,
        source_xa: &mut ::poulpy_hal::source::Source,
        scratch: &mut ::poulpy_hal::layouts::ScratchArena<'_, $be>,
    ) where
        R: $crate::layouts::LWEToBackendMut<$be> + $crate::layouts::LWEInfos,
        P: $crate::layouts::LWEPlaintextToBackendRef<$be>,
        S: $crate::layouts::LWESecretToBackendRef<$be>,
        E: $crate::api::EncryptionInfos {
            <::poulpy_hal::layouts::Module<$be> as $crate::reference::encryption::LWEEncryptSkReference<$be>>::lwe_encrypt_sk_reference::<R, P, S, E>(module, res, pt, sk, enc_infos, source_xe, source_xa, scratch)
        }

    fn gglwe_encrypt_sk_tmp_bytes<A>(module: &::poulpy_hal::layouts::Module<$be>, infos: &A) -> usize
    where
        A: $crate::layouts::GGLWEInfos {
            <::poulpy_hal::layouts::Module<$be> as $crate::reference::encryption::GGLWEEncryptSkReference<$be>>::gglwe_encrypt_sk_tmp_bytes_reference::<A>(module, infos)
        }

    fn gglwe_encrypt_sk<R, P, S, E>(
        module: &::poulpy_hal::layouts::Module<$be>,
        res: &mut R,
        pt: &P,
        sk: &S,
        enc_infos: &E,
        source_xe: &mut ::poulpy_hal::source::Source,
        source_xa: &mut ::poulpy_hal::source::Source,
        scratch: &mut ::poulpy_hal::layouts::ScratchArena<'_, $be>,
    ) where
        R: $crate::layouts::GGLWEToBackendMut<$be>,
        P: ::poulpy_hal::layouts::ScalarZnxToBackendRef<$be>,
        E: $crate::api::EncryptionInfos,
        S: $crate::layouts::GLWESecretPreparedToBackendRef<$be> {
            <::poulpy_hal::layouts::Module<$be> as $crate::reference::encryption::GGLWEEncryptSkReference<$be>>::gglwe_encrypt_sk_reference::<R, P, S, E>(module, res, pt, sk, enc_infos, source_xe, source_xa, scratch)
        }

    fn fill_glwe_mask_from_source<R>(
        module: &::poulpy_hal::layouts::Module<$be>,
        base2k: usize,
        res: &mut R,
        res_col: usize,
        rank: usize,
        source_xa: &mut ::poulpy_hal::source::Source,
    ) where
        R: $crate::layouts::GLWEToBackendMut<$be> {
            <::poulpy_hal::layouts::Module<$be> as $crate::reference::encryption::GLWEMaskFillReference<$be>>::fill_glwe_mask_from_source_reference::<R>(module, base2k, res, res_col, rank, source_xa)
        }

    fn glwe_encrypt_sk_tmp_bytes<A>(module: &::poulpy_hal::layouts::Module<$be>, infos: &A) -> usize
    where
        A: $crate::layouts::GLWEInfos {
            <::poulpy_hal::layouts::Module<$be> as $crate::reference::encryption::GLWEEncryptSkReference<$be>>::glwe_encrypt_sk_tmp_bytes_reference::<A>(module, infos)
        }

    fn glwe_encrypt_sk<R, P, S, E>(
        module: &::poulpy_hal::layouts::Module<$be>,
        res: &mut R,
        pt: &P,
        sk: &S,
        enc_infos: &E,
        source_xe: &mut ::poulpy_hal::source::Source,
        source_xa: &mut ::poulpy_hal::source::Source,
        scratch: &mut ::poulpy_hal::layouts::ScratchArena<'_, $be>,
    ) where
        R: $crate::layouts::GLWEToBackendMut<$be>,
        P: $crate::layouts::GLWEToBackendRef<$be>,
        E: $crate::api::EncryptionInfos,
        S: $crate::layouts::GLWESecretPreparedToBackendRef<$be> {
            <::poulpy_hal::layouts::Module<$be> as $crate::reference::encryption::GLWEEncryptSkReference<$be>>::glwe_encrypt_sk_reference::<R, P, S, E>(module, res, pt, sk, enc_infos, source_xe, source_xa, scratch)
        }

    fn glwe_encrypt_zero_sk<R, E, S>(
        module: &::poulpy_hal::layouts::Module<$be>,
        res: &mut R,
        sk: &S,
        enc_infos: &E,
        source_xe: &mut ::poulpy_hal::source::Source,
        source_xa: &mut ::poulpy_hal::source::Source,
        scratch: &mut ::poulpy_hal::layouts::ScratchArena<'_, $be>,
    ) where
        R: $crate::layouts::GLWEToBackendMut<$be>,
        E: $crate::api::EncryptionInfos,
        S: $crate::layouts::GLWESecretPreparedToBackendRef<$be> {
            <::poulpy_hal::layouts::Module<$be> as $crate::reference::encryption::GLWEEncryptSkReference<$be>>::glwe_encrypt_zero_sk_reference::<R, E, S>(module, res, sk, enc_infos, source_xe, source_xa, scratch)
        }

    fn glwe_encrypt_pk_tmp_bytes<A>(module: &::poulpy_hal::layouts::Module<$be>, infos: &A) -> usize
    where
        A: $crate::layouts::GLWEInfos {
            <::poulpy_hal::layouts::Module<$be> as $crate::reference::encryption::GLWEEncryptPkReference<$be>>::glwe_encrypt_pk_tmp_bytes_reference::<A>(module, infos)
        }

    fn glwe_encrypt_pk<R, P, K, E>(
        module: &::poulpy_hal::layouts::Module<$be>,
        res: &mut R,
        pt: &P,
        pk: &K,
        enc_infos: &E,
        source_xu: &mut ::poulpy_hal::source::Source,
        source_xe: &mut ::poulpy_hal::source::Source,
        scratch: &mut ::poulpy_hal::layouts::ScratchArena<'_, $be>,
    ) where
        R: $crate::layouts::GLWEToBackendMut<$be> + $crate::layouts::GLWEInfos,
        P: $crate::layouts::GLWEToBackendRef<$be> + $crate::layouts::GLWEInfos,
        E: $crate::api::EncryptionInfos,
        K: $crate::layouts::GLWEPreparedToBackendRef<$be> + $crate::GetDistribution + $crate::layouts::GLWEInfos {
            <::poulpy_hal::layouts::Module<$be> as $crate::reference::encryption::GLWEEncryptPkReference<$be>>::glwe_encrypt_pk_reference::<R, P, K, E>(module, res, pt, pk, enc_infos, source_xu, source_xe, scratch)
        }

    fn glwe_encrypt_zero_pk<R, K, E>(
        module: &::poulpy_hal::layouts::Module<$be>,
        res: &mut R,
        pk: &K,
        enc_infos: &E,
        source_xu: &mut ::poulpy_hal::source::Source,
        source_xe: &mut ::poulpy_hal::source::Source,
        scratch: &mut ::poulpy_hal::layouts::ScratchArena<'_, $be>,
    ) where
        R: $crate::layouts::GLWEToBackendMut<$be> + $crate::layouts::GLWEInfos,
        E: $crate::api::EncryptionInfos,
        K: $crate::layouts::GLWEPreparedToBackendRef<$be> + $crate::GetDistribution + $crate::layouts::GLWEInfos {
            <::poulpy_hal::layouts::Module<$be> as $crate::reference::encryption::GLWEEncryptPkReference<$be>>::glwe_encrypt_zero_pk_reference::<R, K, E>(module, res, pk, enc_infos, source_xu, source_xe, scratch)
        }

    fn gglwe_to_ggsw_key_encrypt_sk_tmp_bytes<A>(module: &::poulpy_hal::layouts::Module<$be>, infos: &A) -> usize
    where
        A: $crate::layouts::GGLWEInfos {
            <::poulpy_hal::layouts::Module<$be> as $crate::reference::encryption::GGLWEToGGSWKeyEncryptSkReference<$be>>::gglwe_to_ggsw_key_encrypt_sk_tmp_bytes_reference::<A>(module, infos)
        }

    fn gglwe_to_ggsw_key_encrypt_sk<R, S, E>(
        module: &::poulpy_hal::layouts::Module<$be>,
        res: &mut R,
        sk: &S,
        enc_infos: &E,
        source_xe: &mut ::poulpy_hal::source::Source,
        source_xa: &mut ::poulpy_hal::source::Source,
        scratch: &mut ::poulpy_hal::layouts::ScratchArena<'_, $be>,
    ) where
        R: $crate::layouts::GGLWEToGGSWKeyToBackendMut<$be>,
        E: $crate::api::EncryptionInfos,
        S: $crate::layouts::GLWESecretToBackendRef<$be> + $crate::GetDistribution + $crate::layouts::GLWEInfos {
            <::poulpy_hal::layouts::Module<$be> as $crate::reference::encryption::GGLWEToGGSWKeyEncryptSkReference<$be>>::gglwe_to_ggsw_key_encrypt_sk_reference::<R, S, E>(module, res, sk, enc_infos, source_xe, source_xa, scratch)
        }

    fn glwe_automorphism_key_encrypt_sk_tmp_bytes<A>(module: &::poulpy_hal::layouts::Module<$be>, infos: &A) -> usize
            where
                A: $crate::layouts::GGLWEInfos,
            {
                <::poulpy_hal::layouts::Module<$be> as $crate::reference::encryption::GLWEAutomorphismKeyEncryptSkReference<
                    $be,
                >>::glwe_automorphism_key_encrypt_sk_tmp_bytes_reference::<A>(module, infos)
            }

    fn glwe_automorphism_key_encrypt_sk<R, S, E>(
                module: &::poulpy_hal::layouts::Module<$be>,
                res: &mut R,
                p: i64,
                sk: &S,
                enc_infos: &E,
                source_xe: &mut ::poulpy_hal::source::Source,
                source_xa: &mut ::poulpy_hal::source::Source,
                scratch: &mut ::poulpy_hal::layouts::ScratchArena<'_, $be>,
            ) where
                R: $crate::layouts::GGLWEToBackendMut<$be> + $crate::layouts::SetGaloisElement + $crate::layouts::GGLWEInfos,
                E: $crate::api::EncryptionInfos,
                S: $crate::layouts::GLWESecretToBackendRef<$be> + $crate::layouts::GLWEInfos,
            {
                <::poulpy_hal::layouts::Module<$be> as $crate::reference::encryption::GLWEAutomorphismKeyEncryptSkReference<
                    $be,
                >>::glwe_automorphism_key_encrypt_sk_reference::<R, S, E>(
                    module, res, p, sk, enc_infos, source_xe, source_xa, scratch,
                )
            }

    fn glwe_compressed_encrypt_sk_tmp_bytes<A>(module: &::poulpy_hal::layouts::Module<$be>, infos: &A) -> usize
    where
        A: $crate::layouts::GLWEInfos {
            <::poulpy_hal::layouts::Module<$be> as $crate::reference::encryption::GLWECompressedEncryptSkReference<$be>>::glwe_compressed_encrypt_sk_tmp_bytes_reference::<A>(module, infos)
        }

    fn glwe_compressed_encrypt_sk<R, P, S, E>(
        module: &::poulpy_hal::layouts::Module<$be>,
        res: &mut R,
        pt: &P,
        sk: &S,
        seed_xa: [u8; 32],
        enc_infos: &E,
        source_xe: &mut ::poulpy_hal::source::Source,
        scratch: &mut ::poulpy_hal::layouts::ScratchArena<'_, $be>,
    ) where
        R: $crate::layouts::GLWECompressedToBackendMut<$be> + $crate::layouts::GLWECompressedSeedMut,
        P: $crate::layouts::GLWEToBackendRef<$be>,
        E: $crate::api::EncryptionInfos,
        S: $crate::layouts::GLWESecretPreparedToBackendRef<$be> {
            <::poulpy_hal::layouts::Module<$be> as $crate::reference::encryption::GLWECompressedEncryptSkReference<$be>>::glwe_compressed_encrypt_sk_reference::<R, P, S, E>(module, res, pt, sk, seed_xa, enc_infos, source_xe, scratch)
        }

    fn glwe_switching_key_compressed_encrypt_sk_tmp_bytes<A>(module: &::poulpy_hal::layouts::Module<$be>, infos: &A) -> usize
    where
        A: $crate::layouts::GGLWEInfos {
            <::poulpy_hal::layouts::Module<$be> as $crate::reference::encryption::GLWESwitchingKeyCompressedEncryptSkReference<$be>>::glwe_switching_key_compressed_encrypt_sk_tmp_bytes_reference::<A>(module, infos)
        }

    fn glwe_switching_key_compressed_encrypt_sk<R, S1, S2, E>(
        module: &::poulpy_hal::layouts::Module<$be>,
        res: &mut R,
        sk_in: &S1,
        sk_out: &S2,
        seed_xa: [u8; 32],
        enc_infos: &E,
        source_xe: &mut ::poulpy_hal::source::Source,
        scratch: &mut ::poulpy_hal::layouts::ScratchArena<'_, $be>,
    ) where
        R: $crate::layouts::GGLWECompressedToBackendMut<$be> + $crate::layouts::GGLWECompressedSeedMut + $crate::layouts::GLWESwitchingKeyDegreesMut + $crate::layouts::GGLWEInfos,
        E: $crate::api::EncryptionInfos,
        S1: $crate::layouts::GLWESecretToBackendRef<$be> + $crate::layouts::GLWEInfos,
        S2: $crate::layouts::GLWESecretToBackendRef<$be> + $crate::GetDistribution + $crate::layouts::GLWEInfos {
            <::poulpy_hal::layouts::Module<$be> as $crate::reference::encryption::GLWESwitchingKeyCompressedEncryptSkReference<$be>>::glwe_switching_key_compressed_encrypt_sk_reference::<R, S1, S2, E>(module, res, sk_in, sk_out, seed_xa, enc_infos, source_xe, scratch)
        }

    fn ggsw_compressed_encrypt_sk_tmp_bytes<A>(module: &::poulpy_hal::layouts::Module<$be>, infos: &A) -> usize
    where
        A: $crate::layouts::GGSWInfos {
            <::poulpy_hal::layouts::Module<$be> as $crate::reference::encryption::GGSWCompressedEncryptSkReference<$be>>::ggsw_compressed_encrypt_sk_tmp_bytes_reference::<A>(module, infos)
        }

    fn ggsw_compressed_encrypt_sk<R, P, S, E>(
        module: &::poulpy_hal::layouts::Module<$be>,
        res: &mut R,
        pt: &P,
        sk: &S,
        seed_xa: [u8; 32],
        enc_infos: &E,
        source_xe: &mut ::poulpy_hal::source::Source,
        scratch: &mut ::poulpy_hal::layouts::ScratchArena<'_, $be>,
    ) where
        R: $crate::layouts::GGSWCompressedToBackendMut<$be> + $crate::layouts::GGSWCompressedSeedMut + $crate::layouts::GGSWInfos,
        P: ::poulpy_hal::layouts::ScalarZnxToBackendRef<$be>,
        E: $crate::api::EncryptionInfos,
        S: $crate::layouts::GLWESecretPreparedToBackendRef<$be> {
            <::poulpy_hal::layouts::Module<$be> as $crate::reference::encryption::GGSWCompressedEncryptSkReference<$be>>::ggsw_compressed_encrypt_sk_reference::<R, P, S, E>(module, res, pt, sk, seed_xa, enc_infos, source_xe, scratch)
        }

    fn gglwe_compressed_encrypt_sk_tmp_bytes<A>(module: &::poulpy_hal::layouts::Module<$be>, infos: &A) -> usize
    where
        A: $crate::layouts::GGLWEInfos {
            <::poulpy_hal::layouts::Module<$be> as $crate::reference::encryption::GGLWECompressedEncryptSkReference<$be>>::gglwe_compressed_encrypt_sk_tmp_bytes_reference::<A>(module, infos)
        }

    fn gglwe_compressed_encrypt_sk<R, P, S, E>(
        module: &::poulpy_hal::layouts::Module<$be>,
        res: &mut R,
        pt: &P,
        sk: &S,
        seed: [u8; 32],
        enc_infos: &E,
        source_xe: &mut ::poulpy_hal::source::Source,
        scratch: &mut ::poulpy_hal::layouts::ScratchArena<'_, $be>,
    ) where
        R: $crate::layouts::GGLWECompressedToBackendMut<$be> + $crate::layouts::GGLWECompressedSeedMut,
        P: ::poulpy_hal::layouts::ScalarZnxToBackendRef<$be>,
        E: $crate::api::EncryptionInfos,
        S: $crate::layouts::GLWESecretPreparedToBackendRef<$be> {
            <::poulpy_hal::layouts::Module<$be> as $crate::reference::encryption::GGLWECompressedEncryptSkReference<$be>>::gglwe_compressed_encrypt_sk_reference::<R, P, S, E>(module, res, pt, sk, seed, enc_infos, source_xe, scratch)
        }

    fn gglwe_to_ggsw_key_compressed_encrypt_sk_tmp_bytes<A>(module: &::poulpy_hal::layouts::Module<$be>, infos: &A) -> usize
    where
        A: $crate::layouts::GGLWEInfos {
            <::poulpy_hal::layouts::Module<$be> as $crate::reference::encryption::GGLWEToGGSWKeyCompressedEncryptSkReference<$be>>::gglwe_to_ggsw_key_compressed_encrypt_sk_tmp_bytes_reference::<A>(module, infos)
        }

    fn gglwe_to_ggsw_key_compressed_encrypt_sk<R, S, E>(
        module: &::poulpy_hal::layouts::Module<$be>,
        res: &mut R,
        sk: &S,
        seed_xa: [u8; 32],
        enc_infos: &E,
        source_xe: &mut ::poulpy_hal::source::Source,
        scratch: &mut ::poulpy_hal::layouts::ScratchArena<'_, $be>,
    ) where
        R: $crate::layouts::GGLWEToGGSWKeyCompressedToBackendMut<$be> + $crate::layouts::GGLWEInfos,
        E: $crate::api::EncryptionInfos,
        S: $crate::layouts::GLWESecretToBackendRef<$be> + $crate::GetDistribution + $crate::layouts::GLWEInfos {
            <::poulpy_hal::layouts::Module<$be> as $crate::reference::encryption::GGLWEToGGSWKeyCompressedEncryptSkReference<$be>>::gglwe_to_ggsw_key_compressed_encrypt_sk_reference::<R, S, E>(module, res, sk, seed_xa, enc_infos, source_xe, scratch)
        }

    fn glwe_automorphism_key_compressed_encrypt_sk_tmp_bytes<A>(module: &::poulpy_hal::layouts::Module<$be>, infos: &A) -> usize
    where
        A: $crate::layouts::GGLWEInfos {
            <::poulpy_hal::layouts::Module<$be> as $crate::reference::encryption::GLWEAutomorphismKeyCompressedEncryptSkReference<$be>>::glwe_automorphism_key_compressed_encrypt_sk_tmp_bytes_reference::<A>(module, infos)
        }

    fn glwe_automorphism_key_compressed_encrypt_sk<R, S, E>(
        module: &::poulpy_hal::layouts::Module<$be>,
        res: &mut R,
        p: i64,
        sk: &S,
        seed_xa: [u8; 32],
        enc_infos: &E,
        source_xe: &mut ::poulpy_hal::source::Source,
        scratch: &mut ::poulpy_hal::layouts::ScratchArena<'_, $be>,
    ) where
        R: $crate::layouts::GGLWECompressedToBackendMut<$be> + $crate::layouts::GGLWECompressedSeedMut + $crate::layouts::SetGaloisElement + $crate::layouts::GGLWEInfos,
        E: $crate::api::EncryptionInfos,
        S: $crate::layouts::GLWESecretToBackendRef<$be> + $crate::layouts::GLWEInfos {
            <::poulpy_hal::layouts::Module<$be> as $crate::reference::encryption::GLWEAutomorphismKeyCompressedEncryptSkReference<$be>>::glwe_automorphism_key_compressed_encrypt_sk_reference::<R, S, E>(module, res, p, sk, seed_xa, enc_infos, source_xe, scratch)
        }
        }
    };
}
