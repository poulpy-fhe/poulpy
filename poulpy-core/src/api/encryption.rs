#![allow(clippy::too_many_arguments)]

use poulpy_hal::{
    layouts::{Backend, ScalarZnxToBackendRef, ScratchArena, ZnxInfos},
    source::Source,
};

use crate::{
    GetDistribution, GetDistributionMut,
    layouts::{
        GGLWEInfos, GGLWEToBackendMut, GGLWEToGGSWKeyCompressedToBackendMut, GGLWEToGGSWKeyToBackendMut, GGSWAtViewMut,
        GGSWCompressedSeedMut, GGSWCompressedToBackendMut, GGSWInfos, GGSWToBackendMut, GLWECompressedSeedMut,
        GLWECompressedToBackendMut, GLWEInfos, GLWEPublicKeyToBackendMut, GLWESecretToBackendRef, GLWESwitchingKeyDegreesMut,
        GLWEToBackendMut, GLWEToBackendRef, LWEInfos, LWEPlaintextToBackendRef, LWESecretToBackendRef, LWEToBackendMut,
        SetGaloisElement,
        compressed::{
            GGLWECompressedSeedMut, GGLWECompressedToBackendMut, GLWEPublicKeyCompressedSeedMut,
            GLWEPublicKeyCompressedToBackendMut,
        },
        prepared::{GLWEPublicKeyPreparedToBackendRef, GLWESecretPreparedToBackendRef},
    },
};

pub trait GLWEMaskFill<BE: Backend> {
    /// Fills the mask columns `1..=rank` of `res`, uniform at its radix and `k`, from `source_xa`.
    fn fill_glwe_mask_from_source<R>(&self, res: &mut R, source_xa: &mut Source)
    where
        R: GLWEToBackendMut<BE>;

    /// Fills the mask of `res` as [`Self::fill_glwe_mask_from_source`] does from `Source::new(seed_xa)`.
    fn fill_glwe_mask_from_seed<R>(&self, res: &mut R, seed_xa: [u8; 32])
    where
        R: GLWEToBackendMut<BE>;

    /// Fills every column of `res`, body included, uniform at its radix and `k`, from `source`.
    fn fill_glwe_from_source<R>(&self, res: &mut R, source: &mut Source)
    where
        R: GLWEToBackendMut<BE>;
}

pub trait LWEFillMask<BE: Backend> {
    /// Fill the LWE mask from `source_xa`.
    fn fill_lwe_mask_from_source<R>(&self, base2k: usize, res: &mut R, source_xa: &mut Source)
    where
        R: LWEToBackendMut<BE>;

    /// Fill the LWE mask from a deterministic seed.
    fn fill_lwe_mask_from_seed<R>(&self, base2k: usize, res: &mut R, seed_xa: [u8; 32])
    where
        R: LWEToBackendMut<BE>;
}

pub trait LWEEncryptSk<BE: Backend> {
    fn lwe_encrypt_sk_tmp_bytes<A>(&self, infos: &A) -> usize
    where
        A: LWEInfos;

    fn lwe_encrypt_sk<R, P, S>(
        &self,
        res: &mut R,
        pt: &P,
        sk: &S,
        source_xe: &mut Source,
        source_xa: &mut Source,
        scratch: &mut ScratchArena<'_, BE>,
    ) where
        R: LWEToBackendMut<BE> + LWEInfos,
        P: LWEPlaintextToBackendRef<BE>,
        S: LWESecretToBackendRef<BE>;
}

pub trait GLWEEncryptSk<BE: Backend> {
    fn glwe_encrypt_sk_tmp_bytes<A>(&self, infos: &A) -> usize
    where
        A: GLWEInfos;

    fn glwe_encrypt_sk<R, P, S>(
        &self,
        res: &mut R,
        pt: &P,
        sk: &S,
        source_xe: &mut Source,
        source_xa: &mut Source,
        scratch: &mut ScratchArena<'_, BE>,
    ) where
        R: GLWEToBackendMut<BE>,
        P: GLWEToBackendRef<BE>,
        S: GLWESecretPreparedToBackendRef<BE>;

    fn glwe_encrypt_zero_sk<R, S>(
        &self,
        res: &mut R,
        sk: &S,
        source_xe: &mut Source,
        source_xa: &mut Source,
        scratch: &mut ScratchArena<'_, BE>,
    ) where
        R: GLWEToBackendMut<BE>,
        S: GLWESecretPreparedToBackendRef<BE>;

    /// Encrypts `pt` under `sk` over the mask columns already in `res`, which
    /// are kept: the body becomes `pt - Sum_k mask_k s_k + e`. The masks must be
    /// uniform, and two encryptions over the same masks under the same secret
    /// reveal the difference of their plaintexts. `pt` must be normalized.
    /// Scratch is [`glwe_encrypt_sk_tmp_bytes`](Self::glwe_encrypt_sk_tmp_bytes).
    fn glwe_encrypt_sk_with_mask<R, P, S>(
        &self,
        res: &mut R,
        pt: &P,
        sk: &S,
        source_xe: &mut Source,
        scratch: &mut ScratchArena<'_, BE>,
    ) where
        R: GLWEToBackendMut<BE>,
        P: GLWEToBackendRef<BE>,
        S: GLWESecretPreparedToBackendRef<BE>;
}

/// Public-key encryption under a [`GLWEPublicKeyPrepared`](crate::layouts::GLWEPublicKeyPrepared)
/// of rank `r`: draws `u_1, .., u_r` in order from `source_xu` under the key's
/// distribution and outputs `Sum_l u_l pk_l + (e_0 + m, e_1, .., e_r)`, one
/// fresh error per column drawn in column order from `source_xe`, each column
/// normalized once at the output's `k`.
pub trait GLWEEncryptPk<BE: Backend> {
    /// Scratch required to encrypt into `res_infos` under a public key of layout `pk_infos`.
    fn glwe_encrypt_pk_tmp_bytes<R, K>(&self, res_infos: &R, pk_infos: &K) -> usize
    where
        R: GLWEInfos,
        K: GLWEInfos;

    /// `pt` must be normalized. Panics if `pk` is less precise than `res`.
    fn glwe_encrypt_pk<R, P, K>(
        &self,
        res: &mut R,
        pt: &P,
        pk: &K,
        source_xu: &mut Source,
        source_xe: &mut Source,
        scratch: &mut ScratchArena<'_, BE>,
    ) where
        R: GLWEToBackendMut<BE> + GLWEInfos,
        P: GLWEToBackendRef<BE> + GLWEInfos,
        K: GLWEPublicKeyPreparedToBackendRef<BE> + GLWEInfos;

    /// Same precondition as [`Self::glwe_encrypt_pk`].
    fn glwe_encrypt_zero_pk<R, K>(
        &self,
        res: &mut R,
        pk: &K,
        source_xu: &mut Source,
        source_xe: &mut Source,
        scratch: &mut ScratchArena<'_, BE>,
    ) where
        R: GLWEToBackendMut<BE> + GLWEInfos,
        K: GLWEPublicKeyPreparedToBackendRef<BE> + GLWEInfos;

    /// [`Self::glwe_encrypt_pk`] with `pt` added to column `col` of the output
    /// (`0` is the body), under the same precondition and scratch. Panics if
    /// `col` exceeds the rank.
    fn glwe_encrypt_pk_at_col<R, P, K>(
        &self,
        res: &mut R,
        pt: &P,
        col: usize,
        pk: &K,
        source_xu: &mut Source,
        source_xe: &mut Source,
        scratch: &mut ScratchArena<'_, BE>,
    ) where
        R: GLWEToBackendMut<BE> + GLWEInfos,
        P: GLWEToBackendRef<BE> + GLWEInfos,
        K: GLWEPublicKeyPreparedToBackendRef<BE> + GLWEInfos;
}

pub trait GLWEPublicKeyGenerate<BE: Backend> {
    /// Scratch required to generate a public key with the given output layout.
    fn glwe_public_key_generate_tmp_bytes<A>(&self, infos: &A) -> usize
    where
        A: GLWEInfos;

    /// Generate a public key using caller-owned scratch. The arena may contain
    /// arbitrary bytes and must meet [`Self::glwe_public_key_generate_tmp_bytes`].
    /// The `rank` entries are generated in order, each a normalized encryption of
    /// zero under `sk` (mask from `source_xa`, error from `source_xe`), and the
    /// key takes the secret's distribution.
    fn glwe_public_key_generate<R, S>(
        &self,
        res: &mut R,
        sk: &S,
        source_xe: &mut Source,
        source_xa: &mut Source,
        scratch: &mut ScratchArena<'_, BE>,
    ) where
        R: GLWEPublicKeyToBackendMut<BE> + GetDistributionMut + GLWEInfos,
        S: GLWESecretPreparedToBackendRef<BE> + GetDistribution;
}

pub trait GLWEPublicKeyCompressedGenerate<BE: Backend> {
    /// Scratch required to generate a compressed public key with the given layout.
    fn glwe_public_key_compressed_generate_tmp_bytes<A>(&self, infos: &A) -> usize
    where
        A: GLWEInfos;

    /// Generates the compressed form of [`GLWEPublicKeyGenerate::glwe_public_key_generate`]:
    /// entry `l` is the body of a normalized encryption of zero under `sk`
    /// whose mask is drawn from the `l`-th seed derived from `seed`, and the key
    /// takes the secret's distribution.
    fn glwe_public_key_compressed_generate<R, S>(
        &self,
        res: &mut R,
        sk: &S,
        seed: [u8; 32],
        source_xe: &mut Source,
        scratch: &mut ScratchArena<'_, BE>,
    ) where
        R: GLWEPublicKeyCompressedToBackendMut<BE> + GLWEPublicKeyCompressedSeedMut + GetDistributionMut + GLWEInfos,
        S: GLWESecretPreparedToBackendRef<BE> + GetDistribution;
}

pub trait GGLWEEncryptSk<BE: Backend> {
    fn gglwe_encrypt_sk_tmp_bytes<A>(&self, infos: &A) -> usize
    where
        A: GGLWEInfos;

    fn gglwe_encrypt_sk<R, P, S>(
        &self,
        res: &mut R,
        pt: &P,
        sk: &S,
        source_xe: &mut Source,
        source_xa: &mut Source,
        scratch: &mut ScratchArena<'_, BE>,
    ) where
        R: GGLWEToBackendMut<BE>,
        P: ScalarZnxToBackendRef<BE>,
        S: GLWESecretPreparedToBackendRef<BE>;
}

pub trait GGSWEncryptSk<BE: Backend> {
    fn ggsw_encrypt_sk_tmp_bytes<A>(&self, infos: &A) -> usize
    where
        A: GGSWInfos;

    fn ggsw_encrypt_sk<R, P, S>(
        &self,
        res: &mut R,
        pt: &P,
        sk: &S,
        source_xe: &mut Source,
        source_xa: &mut Source,
        scratch: &mut ScratchArena<'_, BE>,
    ) where
        R: GGSWToBackendMut<BE> + GGSWInfos + GGSWAtViewMut<BE>,
        P: ScalarZnxToBackendRef<BE> + ZnxInfos,
        S: GLWESecretPreparedToBackendRef<BE> + LWEInfos + GLWEInfos;
}

/// [`GLWEEncryptPk`] whose body error is a flood: outputs
/// `Sum_l u_l pk_l + (m + f, e_1, .., e_r)`, where `f` is drawn from `flood` at
/// the output's precision with `source_smudge` and replaces the body's encryption
/// error. `u` and `e_1, .., e_r` are drawn as [`GLWEEncryptPk`] draws them.
/// Sampling precision is selected from inherited, key-truncation and fresh mask errors;
/// the deliberate flood is excluded from that target and added at output `k`.
pub trait GLWEEncryptPkSmudged<BE: Backend> {
    fn glwe_encrypt_pk_smudged_tmp_bytes<R, K>(&self, res_infos: &R, pk_infos: &K) -> usize
    where
        R: GLWEInfos,
        K: GLWEInfos;

    /// `pt` must be normalized. Panics if `pk` is less precise than `res` or if
    /// `flood` does not fit the precision of `res`.
    #[allow(clippy::too_many_arguments)]
    fn glwe_encrypt_pk_smudged<R, P, K>(
        &self,
        res: &mut R,
        pt: &P,
        pk: &K,
        flood: crate::Noise,
        source_xu: &mut Source,
        source_xe: &mut Source,
        source_smudge: &mut Source,
        scratch: &mut ScratchArena<'_, BE>,
    ) where
        R: GLWEToBackendMut<BE> + GLWEInfos,
        P: GLWEToBackendRef<BE> + GLWEInfos,
        K: GLWEPublicKeyPreparedToBackendRef<BE> + GLWEInfos;
}

/// Public-key GGSW encryption: entry `(row, col)` is
/// [`GLWEEncryptPk::glwe_encrypt_pk_at_col`] of `pt` at limb
/// `(dsize - 1) + row * dsize` into column `col`, entries in row then column
/// order.
pub trait GGSWEncryptPk<BE: Backend> {
    /// Scratch required to encrypt into `res_infos` under a public key of layout `pk_infos`.
    fn ggsw_encrypt_pk_tmp_bytes<R, K>(&self, res_infos: &R, pk_infos: &K) -> usize
    where
        R: GGSWInfos,
        K: GLWEInfos;

    /// Panics if `pk` is less precise than `res`.
    fn ggsw_encrypt_pk<R, P, K>(
        &self,
        res: &mut R,
        pt: &P,
        pk: &K,
        source_xu: &mut Source,
        source_xe: &mut Source,
        scratch: &mut ScratchArena<'_, BE>,
    ) where
        R: GGSWInfos + GGSWAtViewMut<BE>,
        P: ScalarZnxToBackendRef<BE> + ZnxInfos,
        K: GLWEPublicKeyPreparedToBackendRef<BE> + GLWEInfos;
}

pub trait GGLWEToGGSWKeyEncryptSk<BE: Backend> {
    fn gglwe_to_ggsw_key_encrypt_sk_tmp_bytes<A>(&self, infos: &A) -> usize
    where
        A: GGLWEInfos;

    fn gglwe_to_ggsw_key_encrypt_sk<R, S>(
        &self,
        res: &mut R,
        sk: &S,
        source_xe: &mut Source,
        source_xa: &mut Source,
        scratch: &mut ScratchArena<'_, BE>,
    ) where
        R: GGLWEToGGSWKeyToBackendMut<BE>,
        S: GLWESecretToBackendRef<BE> + GetDistribution + GLWEInfos;
}

pub trait GLWESwitchingKeyEncryptSk<BE: Backend> {
    fn glwe_switching_key_encrypt_sk_tmp_bytes<A>(&self, infos: &A) -> usize
    where
        A: GGLWEInfos;

    fn glwe_switching_key_encrypt_sk<R, S1, S2>(
        &self,
        res: &mut R,
        sk_in: &S1,
        sk_out: &S2,
        source_xe: &mut Source,
        source_xa: &mut Source,
        scratch: &mut ScratchArena<'_, BE>,
    ) where
        R: GGLWEToBackendMut<BE> + GLWESwitchingKeyDegreesMut + GGLWEInfos,
        S1: GLWESecretToBackendRef<BE> + GLWEInfos,
        S2: GLWESecretToBackendRef<BE> + GetDistribution + GLWEInfos;
}

pub trait GLWETensorKeyEncryptSk<BE: Backend> {
    fn glwe_tensor_key_encrypt_sk_tmp_bytes<A>(&self, infos: &A) -> usize
    where
        A: GGLWEInfos;

    fn glwe_tensor_key_encrypt_sk<R, S>(
        &self,
        res: &mut R,
        sk: &S,
        source_xe: &mut Source,
        source_xa: &mut Source,
        scratch: &mut ScratchArena<'_, BE>,
    ) where
        R: GGLWEToBackendMut<BE> + GGLWEInfos,
        S: GLWESecretToBackendRef<BE> + GetDistribution + GLWEInfos;
}

pub trait GLWEToLWESwitchingKeyEncryptSk<BE: Backend> {
    fn glwe_to_lwe_key_encrypt_sk_tmp_bytes<A>(&self, infos: &A) -> usize
    where
        A: GGLWEInfos;

    fn glwe_to_lwe_key_encrypt_sk<R, S1, S2>(
        &self,
        res: &mut R,
        sk_lwe: &S1,
        sk_glwe: &S2,
        source_xe: &mut Source,
        source_xa: &mut Source,
        scratch: &mut ScratchArena<'_, BE>,
    ) where
        S1: LWESecretToBackendRef<BE>,
        S2: GLWESecretToBackendRef<BE>,
        R: GGLWEToBackendMut<BE> + GGLWEInfos;
}

pub trait LWESwitchingKeyEncrypt<BE: Backend> {
    fn lwe_switching_key_encrypt_sk_tmp_bytes<A>(&self, infos: &A) -> usize
    where
        A: GGLWEInfos;

    fn lwe_switching_key_encrypt_sk<R, S1, S2>(
        &self,
        res: &mut R,
        sk_lwe_in: &S1,
        sk_lwe_out: &S2,
        source_xe: &mut Source,
        source_xa: &mut Source,
        scratch: &mut ScratchArena<'_, BE>,
    ) where
        R: GGLWEToBackendMut<BE> + GLWESwitchingKeyDegreesMut + GGLWEInfos,
        S1: LWESecretToBackendRef<BE>,
        S2: LWESecretToBackendRef<BE>;
}

pub trait LWEToGLWESwitchingKeyEncryptSk<BE: Backend> {
    fn lwe_to_glwe_key_encrypt_sk_tmp_bytes<A>(&self, infos: &A) -> usize
    where
        A: GGLWEInfos;

    fn lwe_to_glwe_key_encrypt_sk<R, S1, S2>(
        &self,
        res: &mut R,
        sk_lwe: &S1,
        sk_glwe: &S2,
        source_xe: &mut Source,
        source_xa: &mut Source,
        scratch: &mut ScratchArena<'_, BE>,
    ) where
        S1: LWESecretToBackendRef<BE>,
        S2: GLWESecretPreparedToBackendRef<BE>,
        R: GGLWEToBackendMut<BE> + GGLWEInfos;
}

pub trait GLWEAutomorphismKeyEncryptSk<BE: Backend> {
    fn glwe_automorphism_key_encrypt_sk_tmp_bytes<A>(&self, infos: &A) -> usize
    where
        A: GGLWEInfos;

    fn glwe_automorphism_key_encrypt_sk<R, S>(
        &self,
        res: &mut R,
        p: i64,
        sk: &S,
        source_xe: &mut Source,
        source_xa: &mut Source,
        scratch: &mut ScratchArena<'_, BE>,
    ) where
        R: GGLWEToBackendMut<BE> + SetGaloisElement + GGLWEInfos,
        S: GLWESecretToBackendRef<BE> + GLWEInfos;
}

pub trait GLWECompressedEncryptSk<BE: Backend> {
    fn glwe_compressed_encrypt_sk_tmp_bytes<A>(&self, infos: &A) -> usize
    where
        A: GLWEInfos;

    fn glwe_compressed_encrypt_sk<R, P, S>(
        &self,
        res: &mut R,
        pt: &P,
        sk: &S,
        seed_xa: [u8; 32],
        source_xe: &mut Source,
        scratch: &mut ScratchArena<'_, BE>,
    ) where
        R: GLWECompressedToBackendMut<BE> + GLWECompressedSeedMut,
        P: GLWEToBackendRef<BE>,
        S: GLWESecretPreparedToBackendRef<BE>;

    /// Encrypts zero: the draws and the output of [`Self::glwe_compressed_encrypt_sk`]
    /// with a zero plaintext, without one. Scratch is
    /// [`Self::glwe_compressed_encrypt_sk_tmp_bytes`].
    fn glwe_compressed_encrypt_zero_sk<R, S>(
        &self,
        res: &mut R,
        sk: &S,
        seed_xa: [u8; 32],
        source_xe: &mut Source,
        scratch: &mut ScratchArena<'_, BE>,
    ) where
        R: GLWECompressedToBackendMut<BE> + GLWECompressedSeedMut,
        S: GLWESecretPreparedToBackendRef<BE>;
}

pub trait GGLWECompressedEncryptSk<BE: Backend> {
    fn gglwe_compressed_encrypt_sk_tmp_bytes<A>(&self, infos: &A) -> usize
    where
        A: GGLWEInfos;

    fn gglwe_compressed_encrypt_sk<R, P, S>(
        &self,
        res: &mut R,
        pt: &P,
        sk: &S,
        seed: [u8; 32],
        source_xe: &mut Source,
        scratch: &mut ScratchArena<'_, BE>,
    ) where
        R: GGLWECompressedToBackendMut<BE> + GGLWECompressedSeedMut,
        P: ScalarZnxToBackendRef<BE>,
        S: GLWESecretPreparedToBackendRef<BE>;
}

pub trait GGSWCompressedEncryptSk<BE: Backend> {
    fn ggsw_compressed_encrypt_sk_tmp_bytes<A>(&self, infos: &A) -> usize
    where
        A: GGSWInfos;

    fn ggsw_compressed_encrypt_sk<R, P, S>(
        &self,
        res: &mut R,
        pt: &P,
        sk: &S,
        seed_xa: [u8; 32],
        source_xe: &mut Source,
        scratch: &mut ScratchArena<'_, BE>,
    ) where
        R: GGSWCompressedToBackendMut<BE> + GGSWCompressedSeedMut + GGSWInfos,
        P: ScalarZnxToBackendRef<BE>,
        S: GLWESecretPreparedToBackendRef<BE>;
}

pub trait GLWESwitchingKeyCompressedEncryptSk<BE: Backend> {
    fn glwe_switching_key_compressed_encrypt_sk_tmp_bytes<A>(&self, infos: &A) -> usize
    where
        A: GGLWEInfos;

    fn glwe_switching_key_compressed_encrypt_sk<R, S1, S2>(
        &self,
        res: &mut R,
        sk_in: &S1,
        sk_out: &S2,
        seed_xa: [u8; 32],
        source_xe: &mut Source,
        scratch: &mut ScratchArena<'_, BE>,
    ) where
        R: GGLWECompressedToBackendMut<BE> + GGLWECompressedSeedMut + GLWESwitchingKeyDegreesMut + GGLWEInfos,
        S1: GLWESecretToBackendRef<BE> + GLWEInfos,
        S2: GLWESecretToBackendRef<BE> + GetDistribution + GLWEInfos;
}

pub trait GLWEAutomorphismKeyCompressedEncryptSk<BE: Backend> {
    fn glwe_automorphism_key_compressed_encrypt_sk_tmp_bytes<A>(&self, infos: &A) -> usize
    where
        A: GGLWEInfos;

    fn glwe_automorphism_key_compressed_encrypt_sk<R, S>(
        &self,
        res: &mut R,
        p: i64,
        sk: &S,
        seed_xa: [u8; 32],
        source_xe: &mut Source,
        scratch: &mut ScratchArena<'_, BE>,
    ) where
        R: GGLWECompressedToBackendMut<BE> + GGLWECompressedSeedMut + SetGaloisElement + GGLWEInfos,
        S: GLWESecretToBackendRef<BE> + GLWEInfos;
}

pub trait GLWETensorKeyCompressedEncryptSk<BE: Backend> {
    fn glwe_tensor_key_compressed_encrypt_sk_tmp_bytes<A>(&self, infos: &A) -> usize
    where
        A: GGLWEInfos;

    fn glwe_tensor_key_compressed_encrypt_sk<R, S>(
        &self,
        res: &mut R,
        sk: &S,
        seed_xa: [u8; 32],
        source_xe: &mut Source,
        scratch: &mut ScratchArena<'_, BE>,
    ) where
        R: GGLWECompressedToBackendMut<BE> + GGLWEInfos + GGLWECompressedSeedMut,
        S: GLWESecretToBackendRef<BE> + GetDistribution + GLWEInfos;
}

pub trait GGLWEToGGSWKeyCompressedEncryptSk<BE: Backend> {
    fn gglwe_to_ggsw_key_compressed_encrypt_sk_tmp_bytes<A>(&self, infos: &A) -> usize
    where
        A: GGLWEInfos;

    fn gglwe_to_ggsw_key_compressed_encrypt_sk<R, S>(
        &self,
        res: &mut R,
        sk: &S,
        seed_xa: [u8; 32],
        source_xe: &mut Source,
        scratch: &mut ScratchArena<'_, BE>,
    ) where
        R: GGLWEToGGSWKeyCompressedToBackendMut<BE> + GGLWEInfos,
        S: GLWESecretToBackendRef<BE> + GetDistribution + GLWEInfos;
}
