#![allow(clippy::too_many_arguments)]

use poulpy_hal::{
    layouts::{Backend, Module, ScalarZnxToBackendRef, ScratchArena, ZnxInfos},
    source::Source,
};

use crate::{
    EncryptionInfos, GetDistribution, GetDistributionMut,
    encryption::{
        GGLWECompressedEncryptSkReference, GGLWEEncryptSkReference, GGLWEToGGSWKeyCompressedEncryptSkReference,
        GGLWEToGGSWKeyEncryptSkReference, GGSWCompressedEncryptSkReference, GGSWEncryptSkReference,
        GLWEAutomorphismKeyCompressedEncryptSkReference, GLWEAutomorphismKeyEncryptSkReference, GLWECompressedEncryptSkReference,
        GLWEEncryptPkReference, GLWEEncryptSkReference, GLWEMaskFillReference, GLWEPublicKeyGenerateReference,
        GLWESwitchingKeyCompressedEncryptSkReference, GLWESwitchingKeyEncryptSkReference,
        GLWETensorKeyCompressedEncryptSkReference, GLWETensorKeyEncryptSkReference, GLWEToLWESwitchingKeyEncryptSkReference,
        LWEEncryptSkReference, LWEFillMaskReference, LWESwitchingKeyEncryptReference, LWEToGLWESwitchingKeyEncryptSkReference,
    },
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
    fn fill_glwe_mask_from_source_reference<R>(
        module: &Module<Self>,
        base2k: usize,
        res: &mut R,
        res_col: usize,
        rank: usize,
        source_xa: &mut Source,
    ) where
        R: GLWEToBackendMut<Self>;

    fn fill_glwe_mask_from_seed_reference<R>(
        module: &Module<Self>,
        base2k: usize,
        res: &mut R,
        res_col: usize,
        rank: usize,
        seed_xa: [u8; 32],
    ) where
        R: GLWEToBackendMut<Self>;

    fn fill_lwe_mask_from_source_reference<R>(module: &Module<Self>, base2k: usize, res: &mut R, source_xa: &mut Source)
    where
        R: LWEToBackendMut<Self>;

    fn fill_lwe_mask_from_seed_reference<R>(module: &Module<Self>, base2k: usize, res: &mut R, seed_xa: [u8; 32])
    where
        R: LWEToBackendMut<Self>;

    fn lwe_encrypt_sk_tmp_bytes_reference<A>(module: &Module<Self>, infos: &A) -> usize
    where
        A: LWEInfos;

    fn lwe_encrypt_sk_reference<R, P, S, E>(
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

    fn glwe_encrypt_sk_tmp_bytes_reference<A>(module: &Module<Self>, infos: &A) -> usize
    where
        A: GLWEInfos;

    fn glwe_encrypt_sk_reference<R, P, S, E>(
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

    fn glwe_encrypt_zero_sk_reference<R, E, S>(
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

    fn glwe_encrypt_pk_tmp_bytes_reference<A>(module: &Module<Self>, infos: &A) -> usize
    where
        A: GLWEInfos;

    fn glwe_encrypt_pk_reference<R, P, K, E>(
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

    fn glwe_public_key_generate_reference<R, S, E>(
        module: &Module<Self>,
        res: &mut R,
        sk: &S,
        enc_infos: &E,
        source_xe: &mut Source,
        source_xa: &mut Source,
    ) where
        R: GLWEToBackendMut<Self> + GetDistributionMut + GLWEInfos,
        E: EncryptionInfos,
        S: GLWESecretPreparedToBackendRef<Self> + GetDistribution;

    fn gglwe_encrypt_sk_tmp_bytes_reference<A>(module: &Module<Self>, infos: &A) -> usize
    where
        A: GGLWEInfos;

    fn gglwe_encrypt_sk_reference<R, P, S, E>(
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

    fn ggsw_encrypt_sk_tmp_bytes_reference<A>(module: &Module<Self>, infos: &A) -> usize
    where
        A: GGSWInfos;

    fn ggsw_encrypt_sk_reference<R, P, S, E>(
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

    fn gglwe_to_ggsw_key_encrypt_sk_tmp_bytes_reference<A>(module: &Module<Self>, infos: &A) -> usize
    where
        A: GGLWEInfos;

    fn gglwe_to_ggsw_key_encrypt_sk_reference<R, S, E>(
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

    fn glwe_switching_key_encrypt_sk_tmp_bytes_reference<A>(module: &Module<Self>, infos: &A) -> usize
    where
        A: GGLWEInfos;

    fn glwe_switching_key_encrypt_sk_reference<R, S1, S2, E>(
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

    fn glwe_tensor_key_encrypt_sk_tmp_bytes_reference<A>(module: &Module<Self>, infos: &A) -> usize
    where
        A: GGLWEInfos;

    fn glwe_tensor_key_encrypt_sk_reference<R, S, E>(
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
        S: GLWESecretToBackendRef<Self> + GetDistribution + GLWEInfos;

    fn glwe_to_lwe_key_encrypt_sk_tmp_bytes_reference<A>(module: &Module<Self>, infos: &A) -> usize
    where
        A: GGLWEInfos;

    fn glwe_to_lwe_key_encrypt_sk_reference<R, S1, S2, E>(
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

    fn lwe_switching_key_encrypt_sk_tmp_bytes_reference<A>(module: &Module<Self>, infos: &A) -> usize
    where
        A: GGLWEInfos;

    fn lwe_switching_key_encrypt_sk_reference<R, S1, S2, E>(
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

    fn lwe_to_glwe_key_encrypt_sk_tmp_bytes_reference<A>(module: &Module<Self>, infos: &A) -> usize
    where
        A: GGLWEInfos;

    fn lwe_to_glwe_key_encrypt_sk_reference<R, S1, S2, E>(
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

    fn glwe_automorphism_key_encrypt_sk_tmp_bytes_reference<A>(module: &Module<Self>, infos: &A) -> usize
    where
        A: GGLWEInfos;

    fn glwe_automorphism_key_encrypt_sk_reference<R, S, E>(
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

    fn glwe_compressed_encrypt_sk_tmp_bytes_reference<A>(module: &Module<Self>, infos: &A) -> usize
    where
        A: GLWEInfos;

    fn glwe_compressed_encrypt_sk_reference<R, P, S, E>(
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

    fn gglwe_compressed_encrypt_sk_tmp_bytes_reference<A>(module: &Module<Self>, infos: &A) -> usize
    where
        A: GGLWEInfos;

    fn gglwe_compressed_encrypt_sk_reference<R, P, S, E>(
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

    fn ggsw_compressed_encrypt_sk_tmp_bytes_reference<A>(module: &Module<Self>, infos: &A) -> usize
    where
        A: GGSWInfos;

    fn ggsw_compressed_encrypt_sk_reference<R, P, S, E>(
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

    fn gglwe_to_ggsw_key_compressed_encrypt_sk_tmp_bytes_reference<A>(module: &Module<Self>, infos: &A) -> usize
    where
        A: GGLWEInfos;

    fn gglwe_to_ggsw_key_compressed_encrypt_sk_reference<R, S, E>(
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

    fn glwe_automorphism_key_compressed_encrypt_sk_tmp_bytes_reference<A>(module: &Module<Self>, infos: &A) -> usize
    where
        A: GGLWEInfos;

    fn glwe_automorphism_key_compressed_encrypt_sk_reference<R, S, E>(
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

    fn glwe_switching_key_compressed_encrypt_sk_tmp_bytes_reference<A>(module: &Module<Self>, infos: &A) -> usize
    where
        A: GGLWEInfos;

    fn glwe_switching_key_compressed_encrypt_sk_reference<R, S1, S2, E>(
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

    fn glwe_tensor_key_compressed_encrypt_sk_tmp_bytes_reference<A>(module: &Module<Self>, infos: &A) -> usize
    where
        A: GGLWEInfos;

    fn glwe_tensor_key_compressed_encrypt_sk_reference<R, S, E>(
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
        S: GLWESecretToBackendRef<Self> + GetDistribution + GLWEInfos;
}

pub trait EncryptionReference<BE: Backend>:
    GLWEMaskFillReference<BE>
    + LWEFillMaskReference<BE>
    + LWEEncryptSkReference<BE>
    + GLWEEncryptSkReference<BE>
    + GLWEEncryptPkReference<BE>
    + GLWEPublicKeyGenerateReference<BE>
    + GGLWEEncryptSkReference<BE>
    + GGSWEncryptSkReference<BE>
    + GGLWEToGGSWKeyEncryptSkReference<BE>
    + GLWESwitchingKeyEncryptSkReference<BE>
    + GLWETensorKeyEncryptSkReference<BE>
    + GLWEToLWESwitchingKeyEncryptSkReference<BE>
    + LWESwitchingKeyEncryptReference<BE>
    + LWEToGLWESwitchingKeyEncryptSkReference<BE>
    + GLWEAutomorphismKeyEncryptSkReference<BE>
    + GLWECompressedEncryptSkReference<BE>
    + GGLWECompressedEncryptSkReference<BE>
    + GGSWCompressedEncryptSkReference<BE>
    + GGLWEToGGSWKeyCompressedEncryptSkReference<BE>
    + GLWEAutomorphismKeyCompressedEncryptSkReference<BE>
    + GLWESwitchingKeyCompressedEncryptSkReference<BE>
    + GLWETensorKeyCompressedEncryptSkReference<BE>
{
}

unsafe impl<BE: Backend> EncryptionImpl for BE
where
    Module<BE>: EncryptionReference<BE>,
{
    fn fill_glwe_mask_from_source_reference<R>(
        module: &Module<BE>,
        base2k: usize,
        res: &mut R,
        res_col: usize,
        rank: usize,
        source_xa: &mut Source,
    ) where
        R: GLWEToBackendMut<BE>,
    {
        module.fill_glwe_mask_from_source_reference(base2k, res, res_col, rank, source_xa)
    }

    fn fill_glwe_mask_from_seed_reference<R>(
        module: &Module<BE>,
        base2k: usize,
        res: &mut R,
        res_col: usize,
        rank: usize,
        seed_xa: [u8; 32],
    ) where
        R: GLWEToBackendMut<BE>,
    {
        module.fill_glwe_mask_from_seed_reference(base2k, res, res_col, rank, seed_xa)
    }

    fn fill_lwe_mask_from_source_reference<R>(module: &Module<BE>, base2k: usize, res: &mut R, source_xa: &mut Source)
    where
        R: LWEToBackendMut<BE>,
    {
        module.fill_lwe_mask_from_source_reference(base2k, res, source_xa)
    }

    fn fill_lwe_mask_from_seed_reference<R>(module: &Module<BE>, base2k: usize, res: &mut R, seed_xa: [u8; 32])
    where
        R: LWEToBackendMut<BE>,
    {
        module.fill_lwe_mask_from_seed_reference(base2k, res, seed_xa)
    }

    fn lwe_encrypt_sk_tmp_bytes_reference<A>(module: &Module<BE>, infos: &A) -> usize
    where
        A: LWEInfos,
    {
        module.lwe_encrypt_sk_tmp_bytes_reference(infos)
    }

    fn lwe_encrypt_sk_reference<R, P, S, E>(
        module: &Module<BE>,
        res: &mut R,
        pt: &P,
        sk: &S,
        enc_infos: &E,
        source_xe: &mut Source,
        source_xa: &mut Source,
        scratch: &mut ScratchArena<'_, BE>,
    ) where
        R: LWEToBackendMut<BE> + LWEInfos,
        P: LWEPlaintextToBackendRef<BE>,
        S: LWESecretToBackendRef<BE>,
        E: EncryptionInfos,
    {
        module.lwe_encrypt_sk_reference(res, pt, sk, enc_infos, source_xe, source_xa, scratch)
    }

    fn glwe_encrypt_sk_tmp_bytes_reference<A>(module: &Module<BE>, infos: &A) -> usize
    where
        A: GLWEInfos,
    {
        module.glwe_encrypt_sk_tmp_bytes_reference(infos)
    }

    fn glwe_encrypt_sk_reference<R, P, S, E>(
        module: &Module<BE>,
        res: &mut R,
        pt: &P,
        sk: &S,
        enc_infos: &E,
        source_xe: &mut Source,
        source_xa: &mut Source,
        scratch: &mut ScratchArena<'_, BE>,
    ) where
        R: GLWEToBackendMut<BE>,
        P: GLWEToBackendRef<BE>,
        E: EncryptionInfos,
        S: GLWESecretPreparedToBackendRef<BE>,
    {
        module.glwe_encrypt_sk_reference(res, pt, sk, enc_infos, source_xe, source_xa, scratch)
    }

    fn glwe_encrypt_zero_sk_reference<R, E, S>(
        module: &Module<BE>,
        res: &mut R,
        sk: &S,
        enc_infos: &E,
        source_xe: &mut Source,
        source_xa: &mut Source,
        scratch: &mut ScratchArena<'_, BE>,
    ) where
        R: GLWEToBackendMut<BE>,
        E: EncryptionInfos,
        S: GLWESecretPreparedToBackendRef<BE>,
    {
        module.glwe_encrypt_zero_sk_reference(res, sk, enc_infos, source_xe, source_xa, scratch)
    }

    fn glwe_encrypt_pk_tmp_bytes_reference<A>(module: &Module<BE>, infos: &A) -> usize
    where
        A: GLWEInfos,
    {
        module.glwe_encrypt_pk_tmp_bytes_reference(infos)
    }

    fn glwe_encrypt_pk_reference<R, P, K, E>(
        module: &Module<BE>,
        res: &mut R,
        pt: &P,
        pk: &K,
        enc_infos: &E,
        source_xu: &mut Source,
        source_xe: &mut Source,
        scratch: &mut ScratchArena<'_, BE>,
    ) where
        R: GLWEToBackendMut<BE> + GLWEInfos,
        P: GLWEToBackendRef<BE> + GLWEInfos,
        E: EncryptionInfos,
        K: GLWEPreparedToBackendRef<BE> + GetDistribution + GLWEInfos,
    {
        module.glwe_encrypt_pk_reference(res, pt, pk, enc_infos, source_xu, source_xe, scratch)
    }

    fn glwe_public_key_generate_reference<R, S, E>(
        module: &Module<BE>,
        res: &mut R,
        sk: &S,
        enc_infos: &E,
        source_xe: &mut Source,
        source_xa: &mut Source,
    ) where
        R: GLWEToBackendMut<BE> + GetDistributionMut + GLWEInfos,
        E: EncryptionInfos,
        S: GLWESecretPreparedToBackendRef<BE> + GetDistribution,
    {
        module.glwe_public_key_generate_reference(res, sk, enc_infos, source_xe, source_xa)
    }

    fn gglwe_encrypt_sk_tmp_bytes_reference<A>(module: &Module<BE>, infos: &A) -> usize
    where
        A: GGLWEInfos,
    {
        module.gglwe_encrypt_sk_tmp_bytes_reference(infos)
    }

    fn gglwe_encrypt_sk_reference<R, P, S, E>(
        module: &Module<BE>,
        res: &mut R,
        pt: &P,
        sk: &S,
        enc_infos: &E,
        source_xe: &mut Source,
        source_xa: &mut Source,
        scratch: &mut ScratchArena<'_, BE>,
    ) where
        R: GGLWEToBackendMut<BE>,
        P: ScalarZnxToBackendRef<BE>,
        E: EncryptionInfos,
        S: GLWESecretPreparedToBackendRef<BE>,
    {
        module.gglwe_encrypt_sk_reference(res, pt, sk, enc_infos, source_xe, source_xa, scratch)
    }

    fn ggsw_encrypt_sk_tmp_bytes_reference<A>(module: &Module<BE>, infos: &A) -> usize
    where
        A: GGSWInfos,
    {
        module.ggsw_encrypt_sk_tmp_bytes_reference(infos)
    }

    fn ggsw_encrypt_sk_reference<R, P, S, E>(
        module: &Module<BE>,
        res: &mut R,
        pt: &P,
        sk: &S,
        enc_infos: &E,
        source_xe: &mut Source,
        source_xa: &mut Source,
        scratch: &mut ScratchArena<'_, BE>,
    ) where
        R: GGSWToBackendMut<BE> + GGSWInfos + GGSWAtViewMut<BE>,
        P: ScalarZnxToBackendRef<BE> + ZnxInfos,
        E: EncryptionInfos,
        S: GLWESecretPreparedToBackendRef<BE> + LWEInfos + GLWEInfos,
    {
        module.ggsw_encrypt_sk_reference(res, pt, sk, enc_infos, source_xe, source_xa, scratch)
    }

    fn gglwe_to_ggsw_key_encrypt_sk_tmp_bytes_reference<A>(module: &Module<BE>, infos: &A) -> usize
    where
        A: GGLWEInfos,
    {
        module.gglwe_to_ggsw_key_encrypt_sk_tmp_bytes_reference(infos)
    }

    fn gglwe_to_ggsw_key_encrypt_sk_reference<R, S, E>(
        module: &Module<BE>,
        res: &mut R,
        sk: &S,
        enc_infos: &E,
        source_xe: &mut Source,
        source_xa: &mut Source,
        scratch: &mut ScratchArena<'_, BE>,
    ) where
        R: GGLWEToGGSWKeyToBackendMut<BE>,
        E: EncryptionInfos,
        S: GLWESecretToBackendRef<BE> + GetDistribution + GLWEInfos,
    {
        module.gglwe_to_ggsw_key_encrypt_sk_reference(res, sk, enc_infos, source_xe, source_xa, scratch)
    }

    fn glwe_switching_key_encrypt_sk_tmp_bytes_reference<A>(module: &Module<BE>, infos: &A) -> usize
    where
        A: GGLWEInfos,
    {
        module.glwe_switching_key_encrypt_sk_tmp_bytes_reference(infos)
    }

    fn glwe_switching_key_encrypt_sk_reference<R, S1, S2, E>(
        module: &Module<BE>,
        res: &mut R,
        sk_in: &S1,
        sk_out: &S2,
        enc_infos: &E,
        source_xe: &mut Source,
        source_xa: &mut Source,
        scratch: &mut ScratchArena<'_, BE>,
    ) where
        R: GGLWEToBackendMut<BE> + GLWESwitchingKeyDegreesMut + GGLWEInfos,
        E: EncryptionInfos,
        S1: GLWESecretToBackendRef<BE> + GLWEInfos,
        S2: GLWESecretToBackendRef<BE> + GetDistribution + GLWEInfos,
    {
        module.glwe_switching_key_encrypt_sk_reference(res, sk_in, sk_out, enc_infos, source_xe, source_xa, scratch)
    }

    fn glwe_tensor_key_encrypt_sk_tmp_bytes_reference<A>(module: &Module<BE>, infos: &A) -> usize
    where
        A: GGLWEInfos,
    {
        module.glwe_tensor_key_encrypt_sk_tmp_bytes_reference(infos)
    }

    fn glwe_tensor_key_encrypt_sk_reference<R, S, E>(
        module: &Module<BE>,
        res: &mut R,
        sk: &S,
        enc_infos: &E,
        source_xe: &mut Source,
        source_xa: &mut Source,
        scratch: &mut ScratchArena<'_, BE>,
    ) where
        R: GGLWEToBackendMut<BE> + GGLWEInfos,
        E: EncryptionInfos,
        S: GLWESecretToBackendRef<BE> + GetDistribution + GLWEInfos,
    {
        module.glwe_tensor_key_encrypt_sk_reference(res, sk, enc_infos, source_xe, source_xa, scratch)
    }

    fn glwe_to_lwe_key_encrypt_sk_tmp_bytes_reference<A>(module: &Module<BE>, infos: &A) -> usize
    where
        A: GGLWEInfos,
    {
        module.glwe_to_lwe_key_encrypt_sk_tmp_bytes_reference(infos)
    }

    fn glwe_to_lwe_key_encrypt_sk_reference<R, S1, S2, E>(
        module: &Module<BE>,
        res: &mut R,
        sk_lwe: &S1,
        sk_glwe: &S2,
        enc_infos: &E,
        source_xe: &mut Source,
        source_xa: &mut Source,
        scratch: &mut ScratchArena<'_, BE>,
    ) where
        S1: LWESecretToBackendRef<BE>,
        S2: GLWESecretToBackendRef<BE>,
        E: EncryptionInfos,
        R: GGLWEToBackendMut<BE> + GGLWEInfos,
    {
        module.glwe_to_lwe_key_encrypt_sk_reference(res, sk_lwe, sk_glwe, enc_infos, source_xe, source_xa, scratch)
    }

    fn lwe_switching_key_encrypt_sk_tmp_bytes_reference<A>(module: &Module<BE>, infos: &A) -> usize
    where
        A: GGLWEInfos,
    {
        module.lwe_switching_key_encrypt_sk_tmp_bytes_reference(infos)
    }

    fn lwe_switching_key_encrypt_sk_reference<R, S1, S2, E>(
        module: &Module<BE>,
        res: &mut R,
        sk_lwe_in: &S1,
        sk_lwe_out: &S2,
        enc_infos: &E,
        source_xe: &mut Source,
        source_xa: &mut Source,
        scratch: &mut ScratchArena<'_, BE>,
    ) where
        R: GGLWEToBackendMut<BE> + GLWESwitchingKeyDegreesMut + GGLWEInfos,
        E: EncryptionInfos,
        S1: LWESecretToBackendRef<BE>,
        S2: LWESecretToBackendRef<BE>,
    {
        module.lwe_switching_key_encrypt_sk_reference(res, sk_lwe_in, sk_lwe_out, enc_infos, source_xe, source_xa, scratch)
    }

    fn lwe_to_glwe_key_encrypt_sk_tmp_bytes_reference<A>(module: &Module<BE>, infos: &A) -> usize
    where
        A: GGLWEInfos,
    {
        module.lwe_to_glwe_key_encrypt_sk_tmp_bytes_reference(infos)
    }

    fn lwe_to_glwe_key_encrypt_sk_reference<R, S1, S2, E>(
        module: &Module<BE>,
        res: &mut R,
        sk_lwe: &S1,
        sk_glwe: &S2,
        enc_infos: &E,
        source_xe: &mut Source,
        source_xa: &mut Source,
        scratch: &mut ScratchArena<'_, BE>,
    ) where
        S1: LWESecretToBackendRef<BE>,
        S2: GLWESecretPreparedToBackendRef<BE>,
        E: EncryptionInfos,
        R: GGLWEToBackendMut<BE> + GGLWEInfos,
    {
        module.lwe_to_glwe_key_encrypt_sk_reference(res, sk_lwe, sk_glwe, enc_infos, source_xe, source_xa, scratch)
    }

    fn glwe_automorphism_key_encrypt_sk_tmp_bytes_reference<A>(module: &Module<BE>, infos: &A) -> usize
    where
        A: GGLWEInfos,
    {
        module.glwe_automorphism_key_encrypt_sk_tmp_bytes_reference(infos)
    }

    fn glwe_automorphism_key_encrypt_sk_reference<R, S, E>(
        module: &Module<BE>,
        res: &mut R,
        p: i64,
        sk: &S,
        enc_infos: &E,
        source_xe: &mut Source,
        source_xa: &mut Source,
        scratch: &mut ScratchArena<'_, BE>,
    ) where
        R: GGLWEToBackendMut<BE> + SetGaloisElement + GGLWEInfos,
        E: EncryptionInfos,
        S: GLWESecretToBackendRef<BE> + GLWEInfos,
    {
        module.glwe_automorphism_key_encrypt_sk_reference(res, p, sk, enc_infos, source_xe, source_xa, scratch)
    }

    fn glwe_compressed_encrypt_sk_tmp_bytes_reference<A>(module: &Module<BE>, infos: &A) -> usize
    where
        A: GLWEInfos,
    {
        module.glwe_compressed_encrypt_sk_tmp_bytes_reference(infos)
    }

    fn glwe_compressed_encrypt_sk_reference<R, P, S, E>(
        module: &Module<BE>,
        res: &mut R,
        pt: &P,
        sk: &S,
        seed_xa: [u8; 32],
        enc_infos: &E,
        source_xe: &mut Source,
        scratch: &mut ScratchArena<'_, BE>,
    ) where
        R: GLWECompressedToBackendMut<BE> + GLWECompressedSeedMut,
        P: GLWEToBackendRef<BE>,
        E: EncryptionInfos,
        S: GLWESecretPreparedToBackendRef<BE>,
    {
        module.glwe_compressed_encrypt_sk_reference(res, pt, sk, seed_xa, enc_infos, source_xe, scratch)
    }

    fn gglwe_compressed_encrypt_sk_tmp_bytes_reference<A>(module: &Module<BE>, infos: &A) -> usize
    where
        A: GGLWEInfos,
    {
        module.gglwe_compressed_encrypt_sk_tmp_bytes_reference(infos)
    }

    fn gglwe_compressed_encrypt_sk_reference<R, P, S, E>(
        module: &Module<BE>,
        res: &mut R,
        pt: &P,
        sk: &S,
        seed: [u8; 32],
        enc_infos: &E,
        source_xe: &mut Source,
        scratch: &mut ScratchArena<'_, BE>,
    ) where
        R: GGLWECompressedToBackendMut<BE> + GGLWECompressedSeedMut,
        P: ScalarZnxToBackendRef<BE>,
        E: EncryptionInfos,
        S: GLWESecretPreparedToBackendRef<BE>,
    {
        module.gglwe_compressed_encrypt_sk_reference(res, pt, sk, seed, enc_infos, source_xe, scratch)
    }

    fn ggsw_compressed_encrypt_sk_tmp_bytes_reference<A>(module: &Module<BE>, infos: &A) -> usize
    where
        A: GGSWInfos,
    {
        module.ggsw_compressed_encrypt_sk_tmp_bytes_reference(infos)
    }

    fn ggsw_compressed_encrypt_sk_reference<R, P, S, E>(
        module: &Module<BE>,
        res: &mut R,
        pt: &P,
        sk: &S,
        seed_xa: [u8; 32],
        enc_infos: &E,
        source_xe: &mut Source,
        scratch: &mut ScratchArena<'_, BE>,
    ) where
        R: GGSWCompressedToBackendMut<BE> + GGSWCompressedSeedMut + GGSWInfos,
        P: ScalarZnxToBackendRef<BE>,
        E: EncryptionInfos,
        S: GLWESecretPreparedToBackendRef<BE>,
    {
        module.ggsw_compressed_encrypt_sk_reference(res, pt, sk, seed_xa, enc_infos, source_xe, scratch)
    }

    fn gglwe_to_ggsw_key_compressed_encrypt_sk_tmp_bytes_reference<A>(module: &Module<BE>, infos: &A) -> usize
    where
        A: GGLWEInfos,
    {
        module.gglwe_to_ggsw_key_compressed_encrypt_sk_tmp_bytes_reference(infos)
    }

    fn gglwe_to_ggsw_key_compressed_encrypt_sk_reference<R, S, E>(
        module: &Module<BE>,
        res: &mut R,
        sk: &S,
        seed_xa: [u8; 32],
        enc_infos: &E,
        source_xe: &mut Source,
        scratch: &mut ScratchArena<'_, BE>,
    ) where
        R: GGLWEToGGSWKeyCompressedToBackendMut<BE> + GGLWEInfos,
        E: EncryptionInfos,
        S: GLWESecretToBackendRef<BE> + GetDistribution + GLWEInfos,
    {
        module.gglwe_to_ggsw_key_compressed_encrypt_sk_reference(res, sk, seed_xa, enc_infos, source_xe, scratch)
    }

    fn glwe_automorphism_key_compressed_encrypt_sk_tmp_bytes_reference<A>(module: &Module<BE>, infos: &A) -> usize
    where
        A: GGLWEInfos,
    {
        module.glwe_automorphism_key_compressed_encrypt_sk_tmp_bytes_reference(infos)
    }

    fn glwe_automorphism_key_compressed_encrypt_sk_reference<R, S, E>(
        module: &Module<BE>,
        res: &mut R,
        p: i64,
        sk: &S,
        seed_xa: [u8; 32],
        enc_infos: &E,
        source_xe: &mut Source,
        scratch: &mut ScratchArena<'_, BE>,
    ) where
        R: GGLWECompressedToBackendMut<BE> + GGLWECompressedSeedMut + SetGaloisElement + GGLWEInfos,
        E: EncryptionInfos,
        S: GLWESecretToBackendRef<BE> + GLWEInfos,
    {
        module.glwe_automorphism_key_compressed_encrypt_sk_reference(res, p, sk, seed_xa, enc_infos, source_xe, scratch)
    }

    fn glwe_switching_key_compressed_encrypt_sk_tmp_bytes_reference<A>(module: &Module<BE>, infos: &A) -> usize
    where
        A: GGLWEInfos,
    {
        module.glwe_switching_key_compressed_encrypt_sk_tmp_bytes_reference(infos)
    }

    fn glwe_switching_key_compressed_encrypt_sk_reference<R, S1, S2, E>(
        module: &Module<BE>,
        res: &mut R,
        sk_in: &S1,
        sk_out: &S2,
        seed_xa: [u8; 32],
        enc_infos: &E,
        source_xe: &mut Source,
        scratch: &mut ScratchArena<'_, BE>,
    ) where
        R: GGLWECompressedToBackendMut<BE> + GGLWECompressedSeedMut + GLWESwitchingKeyDegreesMut + GGLWEInfos,
        E: EncryptionInfos,
        S1: GLWESecretToBackendRef<BE> + GLWEInfos,
        S2: GLWESecretToBackendRef<BE> + GetDistribution + GLWEInfos,
    {
        module.glwe_switching_key_compressed_encrypt_sk_reference(res, sk_in, sk_out, seed_xa, enc_infos, source_xe, scratch)
    }

    fn glwe_tensor_key_compressed_encrypt_sk_tmp_bytes_reference<A>(module: &Module<BE>, infos: &A) -> usize
    where
        A: GGLWEInfos,
    {
        module.glwe_tensor_key_compressed_encrypt_sk_tmp_bytes_reference(infos)
    }

    fn glwe_tensor_key_compressed_encrypt_sk_reference<R, S, E>(
        module: &Module<BE>,
        res: &mut R,
        sk: &S,
        seed_xa: [u8; 32],
        enc_infos: &E,
        source_xe: &mut Source,
        scratch: &mut ScratchArena<'_, BE>,
    ) where
        R: GGLWECompressedToBackendMut<BE> + GGLWEInfos + GGLWECompressedSeedMut,
        E: EncryptionInfos,
        S: GLWESecretToBackendRef<BE> + GetDistribution + GLWEInfos,
    {
        module.glwe_tensor_key_compressed_encrypt_sk_reference(res, sk, seed_xa, enc_infos, source_xe, scratch)
    }
}

/// Marker opt-in for [`EncryptionReference`] on `Module<$be>`.
///
/// Equivalent to writing `impl EncryptionReference<$be> for Module<$be> {}`. The aggregator's
/// supertrait chain auto-derives all 22 encryption sub-defaults from their HAL bounds.
#[macro_export]
macro_rules! impl_encryption_reference_full {
    ($be:ty) => {
        impl $crate::oep::EncryptionReference<$be> for ::poulpy_hal::layouts::Module<$be> {}
    };
}
