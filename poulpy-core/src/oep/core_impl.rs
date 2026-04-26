#![allow(clippy::too_many_arguments)]

use std::collections::HashMap;

use poulpy_hal::{
    layouts::{Backend, DataMut, DataRef, Module, ScalarZnxToRef, Scratch},
    source::Source,
};

use crate::{
    EncryptionInfos, GetDistribution, GetDistributionMut, ScratchTakeCore,
    glwe_packer::GLWEPacker,
    layouts::{
        GGLWECompressedSeedMut, GGLWECompressedToMut, GGLWEInfos, GGLWEPreparedToRef, GGLWEToGGSWKeyCompressedToMut,
        GGLWEToGGSWKeyPreparedToRef, GGLWEToGGSWKeyToMut, GGLWEToMut, GGLWEToRef, GGSWCompressedSeedMut, GGSWCompressedToMut,
        GGSWInfos, GGSWPreparedToRef, GGSWToMut, GGSWToRef, GLWE, GLWEAutomorphismKeyHelper, GLWECompressedSeedMut,
        GLWECompressedToMut, GLWEInfos, GLWEPlaintext, GLWEPlaintextToMut, GLWEPlaintextToRef, GLWEPreparedToRef,
        GLWESecretPrepared, GLWESecretPreparedToRef, GLWESecretTensorPrepared, GLWESecretToRef, GLWESwitchingKeyDegreesMut,
        GLWETensor, GLWETensorKeyPrepared, GLWEToMut, GLWEToRef, GetGaloisElement, LWEInfos, LWEPlaintextToMut,
        LWEPlaintextToRef, LWESecretToRef, LWEToMut, LWEToRef, SetGaloisElement, SetLWEInfos,
    },
};

/// Backend-owned high-level `poulpy-core` extension point.
///
/// `Module<BE>` remains the safe execution object exposed to users. Backends
/// implement this trait on their backend marker type to inherit or override
/// `poulpy-core` algorithms.
///
/// # Safety
/// Implementors must uphold all invariants expected by the core algorithms,
/// including correct buffer sizing, alignment, and aliasing behavior, and must
/// never write out of bounds or return invalid references.
pub unsafe trait CoreImpl<BE: Backend>: Backend {
    // Keyswitching
    fn glwe_keyswitch_tmp_bytes<R, A, K>(module: &Module<BE>, res_infos: &R, a_infos: &A, key_infos: &K) -> usize
    where
        R: GLWEInfos,
        A: GLWEInfos,
        K: GGLWEInfos;

    fn glwe_keyswitch<R, A, K>(module: &Module<BE>, res: &mut R, a: &A, key: &K, scratch: &mut Scratch<BE>)
    where
        R: GLWEToMut + GLWEInfos,
        A: GLWEToRef + GLWEInfos,
        K: GGLWEPreparedToRef<BE> + GGLWEInfos;

    fn glwe_keyswitch_assign<R, K>(module: &Module<BE>, res: &mut R, key: &K, scratch: &mut Scratch<BE>)
    where
        R: GLWEToMut + GLWEInfos,
        K: GGLWEPreparedToRef<BE> + GGLWEInfos;

    fn gglwe_keyswitch_tmp_bytes<R, A, K>(module: &Module<BE>, res_infos: &R, a_infos: &A, key_infos: &K) -> usize
    where
        R: GGLWEInfos,
        A: GGLWEInfos,
        K: GGLWEInfos;

    fn gglwe_keyswitch<R, A, K>(module: &Module<BE>, res: &mut R, a: &A, key: &K, scratch: &mut Scratch<BE>)
    where
        R: GGLWEToMut + GGLWEInfos,
        A: GGLWEToRef + GGLWEInfos,
        K: GGLWEPreparedToRef<BE> + GGLWEInfos,
        Scratch<BE>: ScratchTakeCore<BE>;

    fn gglwe_keyswitch_assign<R, K>(module: &Module<BE>, res: &mut R, key: &K, scratch: &mut Scratch<BE>)
    where
        R: GGLWEToMut,
        K: GGLWEPreparedToRef<BE> + GGLWEInfos,
        Scratch<BE>: ScratchTakeCore<BE>;

    fn ggsw_keyswitch_tmp_bytes<R, A, K, T>(
        module: &Module<BE>,
        res_infos: &R,
        a_infos: &A,
        key_infos: &K,
        tsk_infos: &T,
    ) -> usize
    where
        R: GGSWInfos,
        A: GGSWInfos,
        K: GGLWEInfos,
        T: GGLWEInfos;

    fn ggsw_keyswitch<R, A, K, T>(module: &Module<BE>, res: &mut R, a: &A, key: &K, tsk: &T, scratch: &mut Scratch<BE>)
    where
        R: GGSWToMut,
        A: GGSWToRef,
        K: GGLWEPreparedToRef<BE> + GGLWEInfos,
        T: GGLWEToGGSWKeyPreparedToRef<BE> + GGLWEInfos,
        Scratch<BE>: ScratchTakeCore<BE>;

    fn ggsw_keyswitch_assign<R, K, T>(module: &Module<BE>, res: &mut R, key: &K, tsk: &T, scratch: &mut Scratch<BE>)
    where
        R: GGSWToMut,
        K: GGLWEPreparedToRef<BE> + GGLWEInfos,
        T: GGLWEToGGSWKeyPreparedToRef<BE> + GGLWEInfos,
        Scratch<BE>: ScratchTakeCore<BE>;

    fn lwe_keyswitch_tmp_bytes<R, A, K>(module: &Module<BE>, res_infos: &R, a_infos: &A, key_infos: &K) -> usize
    where
        R: LWEInfos,
        A: LWEInfos,
        K: GGLWEInfos;

    fn lwe_keyswitch<R, A, K>(module: &Module<BE>, res: &mut R, a: &A, ksk: &K, scratch: &mut Scratch<BE>)
    where
        R: LWEToMut,
        A: LWEToRef,
        K: GGLWEPreparedToRef<BE> + GGLWEInfos,
        Scratch<BE>: ScratchTakeCore<BE>;

    // External product
    fn glwe_external_product_tmp_bytes<R, A, G>(module: &Module<BE>, res_infos: &R, a_infos: &A, ggsw_infos: &G) -> usize
    where
        R: GLWEInfos,
        A: GLWEInfos,
        G: GGSWInfos;

    fn glwe_external_product<R, A, G>(module: &Module<BE>, res: &mut R, a: &A, ggsw: &G, scratch: &mut Scratch<BE>)
    where
        R: GLWEToMut + GLWEInfos,
        A: GLWEToRef + GLWEInfos,
        G: GGSWPreparedToRef<BE> + GGSWInfos;

    fn glwe_external_product_assign<R, G>(module: &Module<BE>, res: &mut R, ggsw: &G, scratch: &mut Scratch<BE>)
    where
        R: GLWEToMut + GLWEInfos,
        G: GGSWPreparedToRef<BE> + GGSWInfos;

    fn gglwe_external_product_tmp_bytes<R, A, B>(module: &Module<BE>, res_infos: &R, a_infos: &A, b_infos: &B) -> usize
    where
        R: GGLWEInfos,
        A: GGLWEInfos,
        B: GGSWInfos;

    fn gglwe_external_product<R, A, B>(module: &Module<BE>, res: &mut R, a: &A, b: &B, scratch: &mut Scratch<BE>)
    where
        R: GGLWEToMut + GGLWEInfos,
        A: GGLWEToRef + GGLWEInfos,
        B: GGSWPreparedToRef<BE> + GGSWInfos,
        Scratch<BE>: ScratchTakeCore<BE>;

    fn gglwe_external_product_assign<R, A>(module: &Module<BE>, res: &mut R, a: &A, scratch: &mut Scratch<BE>)
    where
        R: GGLWEToMut,
        A: GGSWPreparedToRef<BE>,
        Scratch<BE>: ScratchTakeCore<BE>;

    fn ggsw_external_product_tmp_bytes<R, A, B>(module: &Module<BE>, res_infos: &R, a_infos: &A, b_infos: &B) -> usize
    where
        R: GGSWInfos,
        A: GGSWInfos,
        B: GGSWInfos;

    fn ggsw_external_product<R, A, B>(module: &Module<BE>, res: &mut R, a: &A, b: &B, scratch: &mut Scratch<BE>)
    where
        R: GGSWToMut,
        A: GGSWToRef,
        B: GGSWPreparedToRef<BE>,
        Scratch<BE>: ScratchTakeCore<BE>;

    fn ggsw_external_product_assign<R, A>(module: &Module<BE>, res: &mut R, a: &A, scratch: &mut Scratch<BE>)
    where
        R: GGSWToMut,
        A: GGSWPreparedToRef<BE>,
        Scratch<BE>: ScratchTakeCore<BE>;

    // Decryption
    fn glwe_decrypt_tmp_bytes<A>(module: &Module<BE>, infos: &A) -> usize
    where
        A: GLWEInfos;

    fn glwe_decrypt<R, P, S>(module: &Module<BE>, res: &R, pt: &mut P, sk: &S, scratch: &mut Scratch<BE>)
    where
        R: GLWEToRef + GLWEInfos,
        P: GLWEPlaintextToMut + GLWEInfos + SetLWEInfos,
        S: GLWESecretPreparedToRef<BE> + GLWEInfos;

    fn lwe_decrypt_tmp_bytes<A>(module: &Module<BE>, infos: &A) -> usize
    where
        A: LWEInfos;

    fn lwe_decrypt<R, P, S>(module: &Module<BE>, res: &R, pt: &mut P, sk: &S, scratch: &mut Scratch<BE>)
    where
        R: LWEToRef,
        P: LWEPlaintextToMut + SetLWEInfos + LWEInfos,
        S: LWESecretToRef,
        Scratch<BE>: ScratchTakeCore<BE>;

    fn glwe_tensor_decrypt_tmp_bytes<A>(module: &Module<BE>, infos: &A) -> usize
    where
        A: GLWEInfos;

    fn glwe_tensor_decrypt<R, P, S0, S1>(
        module: &Module<BE>,
        res: &GLWETensor<R>,
        pt: &mut GLWEPlaintext<P>,
        sk: &GLWESecretPrepared<S0, BE>,
        sk_tensor: &GLWESecretTensorPrepared<S1, BE>,
        scratch: &mut Scratch<BE>,
    ) where
        R: DataRef,
        P: DataMut,
        S0: DataRef,
        S1: DataRef;

    // Conversion
    fn glwe_from_lwe_tmp_bytes<R, A, K>(module: &Module<BE>, glwe_infos: &R, lwe_infos: &A, key_infos: &K) -> usize
    where
        R: GLWEInfos,
        A: LWEInfos,
        K: GGLWEInfos;

    fn glwe_from_lwe<R, A, K>(module: &Module<BE>, res: &mut R, lwe: &A, ksk: &K, scratch: &mut Scratch<BE>)
    where
        R: GLWEToMut,
        A: LWEToRef,
        K: GGLWEPreparedToRef<BE> + GGLWEInfos;

    fn lwe_from_glwe_tmp_bytes<R, A, K>(module: &Module<BE>, lwe_infos: &R, glwe_infos: &A, key_infos: &K) -> usize
    where
        R: LWEInfos,
        A: GLWEInfos,
        K: GGLWEInfos;

    fn lwe_from_glwe<R, A, K>(module: &Module<BE>, res: &mut R, a: &A, a_idx: usize, key: &K, scratch: &mut Scratch<BE>)
    where
        R: LWEToMut,
        A: GLWEToRef,
        K: GGLWEPreparedToRef<BE> + GGLWEInfos,
        Scratch<BE>: ScratchTakeCore<BE>;

    fn ggsw_from_gglwe_tmp_bytes<R, A>(module: &Module<BE>, res_infos: &R, tsk_infos: &A) -> usize
    where
        R: GGSWInfos,
        A: GGLWEInfos;

    fn ggsw_from_gglwe<R, A, T>(module: &Module<BE>, res: &mut R, a: &A, tsk: &T, scratch: &mut Scratch<BE>)
    where
        R: GGSWToMut,
        A: GGLWEToRef,
        T: GGLWEToGGSWKeyPreparedToRef<BE>,
        Scratch<BE>: ScratchTakeCore<BE>;

    fn ggsw_expand_rows_tmp_bytes<R, A>(module: &Module<BE>, res_infos: &R, tsk_infos: &A) -> usize
    where
        R: GGSWInfos,
        A: GGLWEInfos;

    fn ggsw_expand_row<R, T>(module: &Module<BE>, res: &mut R, tsk: &T, scratch: &mut Scratch<BE>)
    where
        R: GGSWToMut,
        T: GGLWEToGGSWKeyPreparedToRef<BE>,
        Scratch<BE>: ScratchTakeCore<BE>;

    // Automorphism
    fn glwe_automorphism_tmp_bytes<R, A, K>(module: &Module<BE>, res_infos: &R, a_infos: &A, key_infos: &K) -> usize
    where
        R: GLWEInfos,
        A: GLWEInfos,
        K: GGLWEInfos;

    fn glwe_automorphism<R, A, K>(module: &Module<BE>, res: &mut R, a: &A, key: &K, scratch: &mut Scratch<BE>)
    where
        R: GLWEToMut + GLWEInfos,
        A: GLWEToRef + GLWEInfos,
        K: GetGaloisElement + GGLWEPreparedToRef<BE> + GGLWEInfos;

    fn glwe_automorphism_assign<R, K>(module: &Module<BE>, res: &mut R, key: &K, scratch: &mut Scratch<BE>)
    where
        R: GLWEToMut + GLWEInfos,
        K: GetGaloisElement + GGLWEPreparedToRef<BE> + GGLWEInfos;

    fn glwe_automorphism_add<R, A, K>(module: &Module<BE>, res: &mut R, a: &A, key: &K, scratch: &mut Scratch<BE>)
    where
        R: GLWEToMut + GLWEInfos,
        A: GLWEToRef + GLWEInfos,
        K: GetGaloisElement + GGLWEPreparedToRef<BE> + GGLWEInfos;

    fn glwe_automorphism_add_assign<R, K>(module: &Module<BE>, res: &mut R, key: &K, scratch: &mut Scratch<BE>)
    where
        R: GLWEToMut + GLWEInfos,
        K: GetGaloisElement + GGLWEPreparedToRef<BE> + GGLWEInfos;

    fn glwe_automorphism_sub<R, A, K>(module: &Module<BE>, res: &mut R, a: &A, key: &K, scratch: &mut Scratch<BE>)
    where
        R: GLWEToMut + GLWEInfos,
        A: GLWEToRef + GLWEInfos,
        K: GetGaloisElement + GGLWEPreparedToRef<BE> + GGLWEInfos;

    fn glwe_automorphism_sub_negate<R, A, K>(module: &Module<BE>, res: &mut R, a: &A, key: &K, scratch: &mut Scratch<BE>)
    where
        R: GLWEToMut + GLWEInfos,
        A: GLWEToRef + GLWEInfos,
        K: GetGaloisElement + GGLWEPreparedToRef<BE> + GGLWEInfos;

    fn glwe_automorphism_sub_assign<R, K>(module: &Module<BE>, res: &mut R, key: &K, scratch: &mut Scratch<BE>)
    where
        R: GLWEToMut + GLWEInfos,
        K: GetGaloisElement + GGLWEPreparedToRef<BE> + GGLWEInfos;

    fn glwe_automorphism_sub_negate_assign<R, K>(module: &Module<BE>, res: &mut R, key: &K, scratch: &mut Scratch<BE>)
    where
        R: GLWEToMut + GLWEInfos,
        K: GetGaloisElement + GGLWEPreparedToRef<BE> + GGLWEInfos;

    fn ggsw_automorphism_tmp_bytes<R, A, K, T>(
        module: &Module<BE>,
        res_infos: &R,
        a_infos: &A,
        key_infos: &K,
        tsk_infos: &T,
    ) -> usize
    where
        R: GGSWInfos,
        A: GGSWInfos,
        K: GGLWEInfos,
        T: GGLWEInfos;

    fn ggsw_automorphism<R, A, K, T>(module: &Module<BE>, res: &mut R, a: &A, key: &K, tsk: &T, scratch: &mut Scratch<BE>)
    where
        R: GGSWToMut + GGSWInfos,
        A: GGSWToRef + GGSWInfos,
        K: GetGaloisElement + GGLWEPreparedToRef<BE> + GGLWEInfos,
        T: GGLWEToGGSWKeyPreparedToRef<BE> + GGLWEInfos,
        Scratch<BE>: ScratchTakeCore<BE>;

    fn ggsw_automorphism_assign<R, K, T>(module: &Module<BE>, res: &mut R, key: &K, tsk: &T, scratch: &mut Scratch<BE>)
    where
        R: GGSWToMut,
        K: GetGaloisElement + GGLWEPreparedToRef<BE> + GGLWEInfos,
        T: GGLWEToGGSWKeyPreparedToRef<BE>,
        Scratch<BE>: ScratchTakeCore<BE>;

    fn glwe_automorphism_key_automorphism_tmp_bytes<R, A, K>(
        module: &Module<BE>,
        res_infos: &R,
        a_infos: &A,
        key_infos: &K,
    ) -> usize
    where
        R: GGLWEInfos,
        A: GGLWEInfos,
        K: GGLWEInfos;

    fn glwe_automorphism_key_automorphism<R, A, K>(module: &Module<BE>, res: &mut R, a: &A, key: &K, scratch: &mut Scratch<BE>)
    where
        R: GGLWEToMut + SetGaloisElement + GGLWEInfos,
        A: GGLWEToRef + GetGaloisElement + GGLWEInfos,
        K: GGLWEPreparedToRef<BE> + GetGaloisElement + GGLWEInfos;

    fn glwe_automorphism_key_automorphism_assign<R, K>(module: &Module<BE>, res: &mut R, key: &K, scratch: &mut Scratch<BE>)
    where
        R: GGLWEToMut + SetGaloisElement + GetGaloisElement + GGLWEInfos,
        K: GGLWEPreparedToRef<BE> + GetGaloisElement + GGLWEInfos;

    // Operations
    fn glwe_mul_const_tmp_bytes<R, A>(module: &Module<BE>, res: &R, a: &A, b_size: usize) -> usize
    where
        R: GLWEInfos,
        A: GLWEInfos;

    fn glwe_mul_const<R, A>(
        module: &Module<BE>,
        cnv_offset: usize,
        res: &mut GLWE<R>,
        a: &GLWE<A>,
        b: &[i64],
        scratch: &mut Scratch<BE>,
    ) where
        R: DataMut,
        A: DataRef;

    fn glwe_mul_const_assign<R>(module: &Module<BE>, cnv_offset: usize, res: &mut GLWE<R>, b: &[i64], scratch: &mut Scratch<BE>)
    where
        R: DataMut;

    fn glwe_mul_plain_tmp_bytes<R, A, B>(module: &Module<BE>, res: &R, a: &A, b: &B) -> usize
    where
        R: GLWEInfos,
        A: GLWEInfos,
        B: GLWEInfos;

    fn glwe_mul_plain<R, A, B>(
        module: &Module<BE>,
        cnv_offset: usize,
        res: &mut GLWE<R>,
        a: &GLWE<A>,
        a_effective_k: usize,
        b: &GLWEPlaintext<B>,
        b_effective_k: usize,
        scratch: &mut Scratch<BE>,
    ) where
        R: DataMut,
        A: DataRef,
        B: DataRef;

    fn glwe_tensor_apply_add_assign<R, A, B>(
        module: &Module<BE>,
        cnv_offset: usize,
        res: &mut GLWETensor<R>,
        a: &GLWE<A>,
        a_effective_k: usize,
        b: &GLWE<B>,
        b_effective_k: usize,
        scratch: &mut Scratch<BE>,
    ) where
        R: DataMut,
        A: DataRef,
        B: DataRef;

    fn glwe_mul_plain_assign<R, A>(
        module: &Module<BE>,
        cnv_offset: usize,
        res: &mut GLWE<R>,
        res_effective_k: usize,
        a: &GLWEPlaintext<A>,
        a_effective_k: usize,
        scratch: &mut Scratch<BE>,
    ) where
        R: DataMut,
        A: DataRef;

    fn glwe_tensor_apply_tmp_bytes<R, A, B>(module: &Module<BE>, res: &R, a: &A, b: &B) -> usize
    where
        R: GLWEInfos,
        A: GLWEInfos,
        B: GLWEInfos;

    fn glwe_tensor_square_apply_tmp_bytes<R, A>(module: &Module<BE>, res: &R, a: &A) -> usize
    where
        R: GLWEInfos,
        A: GLWEInfos;

    fn glwe_tensor_apply<R, A, B>(
        module: &Module<BE>,
        cnv_offset: usize,
        res: &mut GLWETensor<R>,
        a: &GLWE<A>,
        a_effective_k: usize,
        b: &GLWE<B>,
        b_effective_k: usize,
        scratch: &mut Scratch<BE>,
    ) where
        R: DataMut,
        A: DataRef,
        B: DataRef;

    fn glwe_tensor_square_apply<R, A>(
        module: &Module<BE>,
        cnv_offset: usize,
        res: &mut GLWETensor<R>,
        a: &GLWE<A>,
        a_effective_k: usize,
        scratch: &mut Scratch<BE>,
    ) where
        R: DataMut,
        A: DataRef;

    fn glwe_tensor_relinearize<R, A, B>(
        module: &Module<BE>,
        res: &mut GLWE<R>,
        a: &GLWETensor<A>,
        tsk: &GLWETensorKeyPrepared<B, BE>,
        tsk_size: usize,
        scratch: &mut Scratch<BE>,
    ) where
        R: DataMut,
        A: DataRef,
        B: DataRef;

    fn glwe_tensor_relinearize_tmp_bytes<R, A, B>(module: &Module<BE>, res: &R, a: &A, tsk: &B) -> usize
    where
        R: GLWEInfos,
        A: GLWEInfos,
        B: GGLWEInfos;

    fn glwe_rotate_tmp_bytes(module: &Module<BE>) -> usize;

    fn glwe_rotate<R, A>(module: &Module<BE>, k: i64, res: &mut R, a: &A)
    where
        R: GLWEToMut,
        A: GLWEToRef;

    fn glwe_rotate_assign<R>(module: &Module<BE>, k: i64, res: &mut R, scratch: &mut Scratch<BE>)
    where
        R: GLWEToMut,
        Scratch<BE>: ScratchTakeCore<BE>;

    fn ggsw_rotate_tmp_bytes(module: &Module<BE>) -> usize;

    fn ggsw_rotate<R, A>(module: &Module<BE>, k: i64, res: &mut R, a: &A)
    where
        R: GGSWToMut,
        A: GGSWToRef;

    fn ggsw_rotate_assign<R>(module: &Module<BE>, k: i64, res: &mut R, scratch: &mut Scratch<BE>)
    where
        R: GGSWToMut,
        Scratch<BE>: ScratchTakeCore<BE> + poulpy_hal::api::ScratchAvailable;

    fn glwe_mul_xp_minus_one<R, A>(module: &Module<BE>, k: i64, res: &mut R, a: &A)
    where
        R: GLWEToMut,
        A: GLWEToRef;

    fn glwe_mul_xp_minus_one_assign<R>(module: &Module<BE>, k: i64, res: &mut R, scratch: &mut Scratch<BE>)
    where
        R: GLWEToMut;

    fn glwe_shift_tmp_bytes(module: &Module<BE>) -> usize;

    fn glwe_rsh<R>(module: &Module<BE>, k: usize, res: &mut R, scratch: &mut Scratch<BE>)
    where
        R: GLWEToMut,
        Scratch<BE>: ScratchTakeCore<BE>;

    fn glwe_lsh_assign<R>(module: &Module<BE>, res: &mut R, k: usize, scratch: &mut Scratch<BE>)
    where
        R: GLWEToMut,
        Scratch<BE>: ScratchTakeCore<BE>;

    fn glwe_lsh<R, A>(module: &Module<BE>, res: &mut R, a: &A, k: usize, scratch: &mut Scratch<BE>)
    where
        R: GLWEToMut,
        A: GLWEToRef,
        Scratch<BE>: ScratchTakeCore<BE>;

    fn glwe_lsh_add<R, A>(module: &Module<BE>, res: &mut R, a: &A, k: usize, scratch: &mut Scratch<BE>)
    where
        R: GLWEToMut,
        A: GLWEToRef,
        Scratch<BE>: ScratchTakeCore<BE>;

    fn glwe_lsh_sub<R, A>(module: &Module<BE>, res: &mut R, a: &A, k: usize, scratch: &mut Scratch<BE>)
    where
        R: GLWEToMut,
        A: GLWEToRef,
        Scratch<BE>: ScratchTakeCore<BE>;

    fn glwe_normalize_tmp_bytes(module: &Module<BE>) -> usize;

    fn glwe_maybe_cross_normalize_to_ref<'a, A>(
        module: &Module<BE>,
        glwe: &'a A,
        target_base2k: usize,
        tmp_slot: &'a mut Option<GLWE<&'a mut [u8]>>,
        scratch: &'a mut Scratch<BE>,
    ) -> (GLWE<&'a [u8]>, &'a mut Scratch<BE>)
    where
        A: GLWEToRef + GLWEInfos,
        Scratch<BE>: ScratchTakeCore<BE>;

    fn glwe_maybe_cross_normalize_to_mut<'a, A>(
        module: &Module<BE>,
        glwe: &'a mut A,
        target_base2k: usize,
        tmp_slot: &'a mut Option<GLWE<&'a mut [u8]>>,
        scratch: &'a mut Scratch<BE>,
    ) -> (GLWE<&'a mut [u8]>, &'a mut Scratch<BE>)
    where
        A: GLWEToMut + GLWEInfos,
        Scratch<BE>: ScratchTakeCore<BE>;

    fn glwe_normalize<R, A>(module: &Module<BE>, res: &mut R, a: &A, scratch: &mut Scratch<BE>)
    where
        R: GLWEToMut,
        A: GLWEToRef,
        Scratch<BE>: ScratchTakeCore<BE>;

    fn glwe_normalize_assign<R>(module: &Module<BE>, res: &mut R, scratch: &mut Scratch<BE>)
    where
        R: GLWEToMut,
        Scratch<BE>: ScratchTakeCore<BE>;

    fn glwe_trace_galois_elements(module: &Module<BE>) -> Vec<i64>;

    fn glwe_trace_tmp_bytes<R, A, K>(module: &Module<BE>, res_infos: &R, a_infos: &A, key_infos: &K) -> usize
    where
        R: GLWEInfos,
        A: GLWEInfos,
        K: GGLWEInfos;

    fn glwe_trace<R, A, K, H>(module: &Module<BE>, res: &mut R, skip: usize, a: &A, keys: &H, scratch: &mut Scratch<BE>)
    where
        R: GLWEToMut + GLWEInfos,
        A: GLWEToRef + GLWEInfos,
        K: GGLWEPreparedToRef<BE> + GetGaloisElement + GGLWEInfos,
        H: GLWEAutomorphismKeyHelper<K, BE>;

    fn glwe_trace_assign<R, K, H>(module: &Module<BE>, res: &mut R, skip: usize, keys: &H, scratch: &mut Scratch<BE>)
    where
        R: GLWEToMut,
        K: GGLWEPreparedToRef<BE> + GetGaloisElement + GGLWEInfos,
        H: GLWEAutomorphismKeyHelper<K, BE>;

    fn glwe_pack_galois_elements(module: &Module<BE>) -> Vec<i64>;

    fn glwe_pack_tmp_bytes<R, K>(module: &Module<BE>, res: &R, key: &K) -> usize
    where
        R: GLWEInfos,
        K: GGLWEInfos;

    fn glwe_pack<R, A, K, H>(
        module: &Module<BE>,
        res: &mut R,
        a: HashMap<usize, &mut A>,
        log_gap_out: usize,
        keys: &H,
        scratch: &mut Scratch<BE>,
    ) where
        R: GLWEToMut + GLWEInfos,
        A: GLWEToMut + GLWEInfos,
        K: GGLWEPreparedToRef<BE> + GetGaloisElement + GGLWEInfos,
        H: GLWEAutomorphismKeyHelper<K, BE>;

    fn packer_add<A, K, H>(
        module: &Module<BE>,
        packer: &mut GLWEPacker,
        a: Option<&A>,
        i: usize,
        auto_keys: &H,
        scratch: &mut Scratch<BE>,
    ) where
        A: GLWEToRef + GLWEInfos,
        K: GGLWEPreparedToRef<BE> + GetGaloisElement + GGLWEInfos,
        H: GLWEAutomorphismKeyHelper<K, BE>,
        Scratch<BE>: ScratchTakeCore<BE>;

    // Encryption
    fn lwe_encrypt_sk_tmp_bytes<A>(module: &Module<BE>, infos: &A) -> usize
    where
        A: LWEInfos;

    fn lwe_encrypt_sk<R, P, S, E>(
        module: &Module<BE>,
        res: &mut R,
        pt: &P,
        sk: &S,
        enc_infos: &E,
        source_xe: &mut Source,
        source_xa: &mut Source,
        scratch: &mut Scratch<BE>,
    ) where
        R: LWEToMut,
        P: LWEPlaintextToRef,
        S: LWESecretToRef,
        E: EncryptionInfos,
        Scratch<BE>: ScratchTakeCore<BE>;

    fn glwe_encrypt_sk_tmp_bytes<A>(module: &Module<BE>, infos: &A) -> usize
    where
        A: GLWEInfos;

    fn glwe_encrypt_sk<R, P, S, E>(
        module: &Module<BE>,
        res: &mut R,
        pt: &P,
        sk: &S,
        enc_infos: &E,
        source_xe: &mut Source,
        source_xa: &mut Source,
        scratch: &mut Scratch<BE>,
    ) where
        R: GLWEToMut,
        P: GLWEPlaintextToRef,
        E: EncryptionInfos,
        S: GLWESecretPreparedToRef<BE>;

    fn glwe_encrypt_zero_sk<R, E, S>(
        module: &Module<BE>,
        res: &mut R,
        sk: &S,
        enc_infos: &E,
        source_xe: &mut Source,
        source_xa: &mut Source,
        scratch: &mut Scratch<BE>,
    ) where
        R: GLWEToMut,
        E: EncryptionInfos,
        S: GLWESecretPreparedToRef<BE>;

    fn glwe_encrypt_pk_tmp_bytes<A>(module: &Module<BE>, infos: &A) -> usize
    where
        A: GLWEInfos;

    fn glwe_encrypt_pk<R, P, K, E>(
        module: &Module<BE>,
        res: &mut R,
        pt: &P,
        pk: &K,
        enc_infos: &E,
        source_xu: &mut Source,
        source_xe: &mut Source,
        scratch: &mut Scratch<BE>,
    ) where
        R: GLWEToMut + GLWEInfos,
        P: GLWEPlaintextToRef + GLWEInfos,
        E: EncryptionInfos,
        K: GLWEPreparedToRef<BE> + GetDistribution + GLWEInfos;

    fn glwe_encrypt_zero_pk<R, K, E>(
        module: &Module<BE>,
        res: &mut R,
        pk: &K,
        enc_infos: &E,
        source_xu: &mut Source,
        source_xe: &mut Source,
        scratch: &mut Scratch<BE>,
    ) where
        R: GLWEToMut + GLWEInfos,
        E: EncryptionInfos,
        K: GLWEPreparedToRef<BE> + GetDistribution + GLWEInfos;

    fn glwe_public_key_generate<R, S, E>(
        module: &Module<BE>,
        res: &mut R,
        sk: &S,
        enc_infos: &E,
        source_xe: &mut Source,
        source_xa: &mut Source,
    ) where
        R: GLWEToMut + GetDistributionMut + GLWEInfos,
        E: EncryptionInfos,
        S: GLWESecretPreparedToRef<BE> + GetDistribution;

    fn gglwe_encrypt_sk_tmp_bytes<A>(module: &Module<BE>, infos: &A) -> usize
    where
        A: GGLWEInfos;

    fn gglwe_encrypt_sk<R, P, S, E>(
        module: &Module<BE>,
        res: &mut R,
        pt: &P,
        sk: &S,
        enc_infos: &E,
        source_xe: &mut Source,
        source_xa: &mut Source,
        scratch: &mut Scratch<BE>,
    ) where
        R: GGLWEToMut,
        P: ScalarZnxToRef,
        E: EncryptionInfos,
        S: GLWESecretPreparedToRef<BE>;

    fn ggsw_encrypt_sk_tmp_bytes<A>(module: &Module<BE>, infos: &A) -> usize
    where
        A: GGSWInfos;

    fn ggsw_encrypt_sk<R, P, S, E>(
        module: &Module<BE>,
        res: &mut R,
        pt: &P,
        sk: &S,
        enc_infos: &E,
        source_xe: &mut Source,
        source_xa: &mut Source,
        scratch: &mut Scratch<BE>,
    ) where
        R: GGSWToMut,
        P: ScalarZnxToRef,
        E: EncryptionInfos,
        S: GLWESecretPreparedToRef<BE>;

    fn gglwe_to_ggsw_key_encrypt_sk_tmp_bytes<A>(module: &Module<BE>, infos: &A) -> usize
    where
        A: GGLWEInfos;

    fn gglwe_to_ggsw_key_encrypt_sk<R, S, E>(
        module: &Module<BE>,
        res: &mut R,
        sk: &S,
        enc_infos: &E,
        source_xe: &mut Source,
        source_xa: &mut Source,
        scratch: &mut Scratch<BE>,
    ) where
        R: GGLWEToGGSWKeyToMut,
        E: EncryptionInfos,
        S: GLWESecretToRef + GetDistribution + GLWEInfos;

    fn glwe_switching_key_encrypt_sk_tmp_bytes<A>(module: &Module<BE>, infos: &A) -> usize
    where
        A: GGLWEInfos;

    fn glwe_switching_key_encrypt_sk<R, S1, S2, E>(
        module: &Module<BE>,
        res: &mut R,
        sk_in: &S1,
        sk_out: &S2,
        enc_infos: &E,
        source_xe: &mut Source,
        source_xa: &mut Source,
        scratch: &mut Scratch<BE>,
    ) where
        R: GGLWEToMut + GLWESwitchingKeyDegreesMut + GGLWEInfos,
        E: EncryptionInfos,
        S1: GLWESecretToRef,
        S2: GLWESecretToRef;

    fn glwe_switching_key_encrypt_pk_tmp_bytes<A>(module: &Module<BE>, infos: &A) -> usize
    where
        A: GGLWEInfos;

    fn glwe_tensor_key_encrypt_sk_tmp_bytes<A>(module: &Module<BE>, infos: &A) -> usize
    where
        A: GGLWEInfos;

    fn glwe_tensor_key_encrypt_sk<R, S, E>(
        module: &Module<BE>,
        res: &mut R,
        sk: &S,
        enc_infos: &E,
        source_xe: &mut Source,
        source_xa: &mut Source,
        scratch: &mut Scratch<BE>,
    ) where
        R: GGLWEToMut + GGLWEInfos,
        E: EncryptionInfos,
        S: GLWESecretToRef + GetDistribution + GLWEInfos;

    fn glwe_to_lwe_key_encrypt_sk_tmp_bytes<A>(module: &Module<BE>, infos: &A) -> usize
    where
        A: GGLWEInfos;

    fn glwe_to_lwe_key_encrypt_sk<R, S1, S2, E>(
        module: &Module<BE>,
        res: &mut R,
        sk_lwe: &S1,
        sk_glwe: &S2,
        enc_infos: &E,
        source_xe: &mut Source,
        source_xa: &mut Source,
        scratch: &mut Scratch<BE>,
    ) where
        S1: LWESecretToRef,
        S2: GLWESecretToRef,
        E: EncryptionInfos,
        R: GGLWEToMut + GGLWEInfos;

    fn lwe_switching_key_encrypt_sk_tmp_bytes<A>(module: &Module<BE>, infos: &A) -> usize
    where
        A: GGLWEInfos;

    fn lwe_switching_key_encrypt_sk<R, S1, S2, E>(
        module: &Module<BE>,
        res: &mut R,
        sk_lwe_in: &S1,
        sk_lwe_out: &S2,
        enc_infos: &E,
        source_xe: &mut Source,
        source_xa: &mut Source,
        scratch: &mut Scratch<BE>,
    ) where
        R: GGLWEToMut + GLWESwitchingKeyDegreesMut + GGLWEInfos,
        E: EncryptionInfos,
        S1: LWESecretToRef,
        S2: LWESecretToRef;

    fn lwe_to_glwe_key_encrypt_sk_tmp_bytes<A>(module: &Module<BE>, infos: &A) -> usize
    where
        A: GGLWEInfos;

    fn lwe_to_glwe_key_encrypt_sk<R, S1, S2, E>(
        module: &Module<BE>,
        res: &mut R,
        sk_lwe: &S1,
        sk_glwe: &S2,
        enc_infos: &E,
        source_xe: &mut Source,
        source_xa: &mut Source,
        scratch: &mut Scratch<BE>,
    ) where
        S1: LWESecretToRef,
        S2: GLWESecretPreparedToRef<BE>,
        E: EncryptionInfos,
        R: GGLWEToMut + GGLWEInfos;

    fn glwe_automorphism_key_encrypt_sk_tmp_bytes<A>(module: &Module<BE>, infos: &A) -> usize
    where
        A: GGLWEInfos;

    fn glwe_automorphism_key_encrypt_sk<R, S, E>(
        module: &Module<BE>,
        res: &mut R,
        p: i64,
        sk: &S,
        enc_infos: &E,
        source_xe: &mut Source,
        source_xa: &mut Source,
        scratch: &mut Scratch<BE>,
    ) where
        R: GGLWEToMut + SetGaloisElement + GGLWEInfos,
        E: EncryptionInfos,
        S: GLWESecretToRef;

    fn glwe_automorphism_key_encrypt_pk_tmp_bytes<A>(module: &Module<BE>, infos: &A) -> usize
    where
        A: GGLWEInfos;

    fn glwe_compressed_encrypt_sk_tmp_bytes<A>(module: &Module<BE>, infos: &A) -> usize
    where
        A: GLWEInfos;

    fn glwe_compressed_encrypt_sk<R, P, S, E>(
        module: &Module<BE>,
        res: &mut R,
        pt: &P,
        sk: &S,
        seed_xa: [u8; 32],
        enc_infos: &E,
        source_xe: &mut Source,
        scratch: &mut Scratch<BE>,
    ) where
        R: GLWECompressedToMut + GLWECompressedSeedMut,
        P: GLWEPlaintextToRef,
        E: EncryptionInfos,
        S: GLWESecretPreparedToRef<BE>;

    fn gglwe_compressed_encrypt_sk_tmp_bytes<A>(module: &Module<BE>, infos: &A) -> usize
    where
        A: GGLWEInfos;

    fn gglwe_compressed_encrypt_sk<R, P, S, E>(
        module: &Module<BE>,
        res: &mut R,
        pt: &P,
        sk: &S,
        seed: [u8; 32],
        enc_infos: &E,
        source_xe: &mut Source,
        scratch: &mut Scratch<BE>,
    ) where
        R: GGLWECompressedToMut + GGLWECompressedSeedMut,
        P: ScalarZnxToRef,
        E: EncryptionInfos,
        S: GLWESecretPreparedToRef<BE>;

    fn ggsw_compressed_encrypt_sk_tmp_bytes<A>(module: &Module<BE>, infos: &A) -> usize
    where
        A: GGSWInfos;

    fn ggsw_compressed_encrypt_sk<R, P, S, E>(
        module: &Module<BE>,
        res: &mut R,
        pt: &P,
        sk: &S,
        seed_xa: [u8; 32],
        enc_infos: &E,
        source_xe: &mut Source,
        scratch: &mut Scratch<BE>,
    ) where
        R: GGSWCompressedToMut + GGSWCompressedSeedMut + GGSWInfos,
        P: ScalarZnxToRef,
        E: EncryptionInfos,
        S: GLWESecretPreparedToRef<BE>;

    fn gglwe_to_ggsw_key_compressed_encrypt_sk_tmp_bytes<A>(module: &Module<BE>, infos: &A) -> usize
    where
        A: GGLWEInfos;

    fn gglwe_to_ggsw_key_compressed_encrypt_sk<R, S, E>(
        module: &Module<BE>,
        res: &mut R,
        sk: &S,
        seed_xa: [u8; 32],
        enc_infos: &E,
        source_xe: &mut Source,
        scratch: &mut Scratch<BE>,
    ) where
        R: GGLWEToGGSWKeyCompressedToMut + GGLWEInfos,
        E: EncryptionInfos,
        S: GLWESecretToRef + GetDistribution + GLWEInfos;

    fn glwe_automorphism_key_compressed_encrypt_sk_tmp_bytes<A>(module: &Module<BE>, infos: &A) -> usize
    where
        A: GGLWEInfos;

    fn glwe_automorphism_key_compressed_encrypt_sk<R, S, E>(
        module: &Module<BE>,
        res: &mut R,
        p: i64,
        sk: &S,
        seed_xa: [u8; 32],
        enc_infos: &E,
        source_xe: &mut Source,
        scratch: &mut Scratch<BE>,
    ) where
        R: GGLWECompressedToMut + GGLWECompressedSeedMut + SetGaloisElement + GGLWEInfos,
        E: EncryptionInfos,
        S: GLWESecretToRef + GLWEInfos;

    fn glwe_switching_key_compressed_encrypt_sk_tmp_bytes<A>(module: &Module<BE>, infos: &A) -> usize
    where
        A: GGLWEInfos;

    fn glwe_switching_key_compressed_encrypt_sk<R, S1, S2, E>(
        module: &Module<BE>,
        res: &mut R,
        sk_in: &S1,
        sk_out: &S2,
        seed_xa: [u8; 32],
        enc_infos: &E,
        source_xe: &mut Source,
        scratch: &mut Scratch<BE>,
    ) where
        R: GGLWECompressedToMut + GGLWECompressedSeedMut + GLWESwitchingKeyDegreesMut + GGLWEInfos,
        E: EncryptionInfos,
        S1: GLWESecretToRef,
        S2: GLWESecretToRef;

    fn glwe_tensor_key_compressed_encrypt_sk_tmp_bytes<A>(module: &Module<BE>, infos: &A) -> usize
    where
        A: GGLWEInfos;

    fn glwe_tensor_key_compressed_encrypt_sk<R, S, E>(
        module: &Module<BE>,
        res: &mut R,
        sk: &S,
        seed_xa: [u8; 32],
        enc_infos: &E,
        source_xe: &mut Source,
        scratch: &mut Scratch<BE>,
    ) where
        R: GGLWECompressedToMut + GGLWEInfos + GGLWECompressedSeedMut,
        E: EncryptionInfos,
        S: GLWESecretToRef + GetDistribution + GLWEInfos;
}
