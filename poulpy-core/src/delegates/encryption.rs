use poulpy_hal::{
    layouts::{Backend, Module, Ring, ScalarZnxToBackendRef, ScratchArena, ZnxInfos},
    source::Source,
};

use crate::{
    GetDistribution, GetDistributionMut,
    api::{
        GGLWECompressedEncryptSk, GGLWEEncryptSk, GGLWEToGGSWKeyCompressedEncryptSk, GGLWEToGGSWKeyEncryptSk,
        GGSWCompressedEncryptSk, GGSWEncryptPk, GGSWEncryptSk, GLWEAutomorphismKeyCompressedEncryptSk,
        GLWEAutomorphismKeyEncryptSk, GLWECompressedEncryptSk, GLWEEncryptPk, GLWEEncryptPkSmudged, GLWEEncryptSk, GLWEFill,
        GLWEPublicKeyCompressedGenerate, GLWEPublicKeyGenerate, GLWESwitchingKeyCompressedEncryptSk, GLWESwitchingKeyEncryptSk,
        GLWETensorKeyCompressedEncryptSk, GLWETensorKeyEncryptSk, GLWEToLWEKeyEncryptSk, LWEEncryptSk, LWEFillMask,
        LWESwitchingKeyEncryptSk, LWEToGLWEKeyEncryptSk,
    },
    layouts::{
        GGLWECompressedSeedMut, GGLWECompressedToBackendMut, GGLWEInfos, GGLWEToBackendMut, GGLWEToGGSWKeyCompressedToBackendMut,
        GGLWEToGGSWKeyToBackendMut, GGSWAtViewMut, GGSWCompressedSeedMut, GGSWCompressedToBackendMut, GGSWInfos,
        GGSWToBackendMut, GLWECompressedSeedMut, GLWECompressedToBackendMut, GLWEInfos, GLWEPublicKeyToBackendMut,
        GLWESecretToBackendRef, GLWESwitchingKeyDegreesMut, GLWEToBackendMut, GLWEToBackendRef, LWEInfos,
        LWEPlaintextToBackendRef, LWESecretToBackendRef, LWEToBackendMut, SetGaloisElement,
        compressed::{GLWEPublicKeyCompressedSeedMut, GLWEPublicKeyCompressedToBackendMut},
        prepared::{GLWEPublicKeyPreparedToBackendRef, GLWESecretPreparedToBackendRef},
    },
    oep::{EncryptionImpl, LWEKeyEncryptionImpl},
};

macro_rules! impl_encryption_delegate {
    (free $trait:ty, $($body:item),+ $(,)?) => {
        impl<BE> $trait for Module<BE>
        where
            BE: Backend + LWEKeyEncryptionImpl,
        {
            $($body)+
        }
    };
    ($trait:ty, $($body:item),+ $(,)?) => {
        impl<BE, Rg: Ring> $trait for Module<BE>
        where
            BE: Backend + EncryptionImpl<Rg>,
        {
            $($body)+
        }
    };
}

impl_encryption_delegate!(
    GLWEFill<BE, Rg>,
    fn glwe_fill_mask_from_source<R>(&self, res: &mut R, source_xa: &mut Source)
    where
        R: GLWEToBackendMut<BE, Rg>,
    {
        BE::glwe_fill_mask_from_source(self, res, source_xa)
    },
    fn glwe_fill_mask_from_seed<R>(&self, res: &mut R, seed_xa: [u8; 32])
    where
        R: GLWEToBackendMut<BE, Rg>,
    {
        BE::glwe_fill_mask_from_seed(self, res, seed_xa)
    },
    fn glwe_fill_from_source<R>(&self, res: &mut R, source: &mut Source)
    where
        R: GLWEToBackendMut<BE, Rg>,
    {
        BE::glwe_fill_from_source(self, res, source)
    }
);

impl_encryption_delegate!(
    LWEFillMask<BE, Rg>,
    fn lwe_fill_mask_from_source<R>(&self, base2k: usize, res: &mut R, source_xa: &mut Source)
    where
        R: LWEToBackendMut<BE, Rg>,
    {
        BE::lwe_fill_mask_from_source(self, base2k, res, source_xa)
    },
    fn lwe_fill_mask_from_seed<R>(&self, base2k: usize, res: &mut R, seed_xa: [u8; 32])
    where
        R: LWEToBackendMut<BE, Rg>,
    {
        BE::lwe_fill_mask_from_seed(self, base2k, res, seed_xa)
    }
);

impl_encryption_delegate!(
    LWEEncryptSk<BE, Rg>,
    fn lwe_encrypt_sk_tmp_bytes<A>(&self, infos: &A) -> usize
    where
        A: LWEInfos,
    {
        BE::lwe_encrypt_sk_tmp_bytes(self, infos)
    },
    fn lwe_encrypt_sk<R, P, S>(
        &self,
        res: &mut R,
        pt: &P,
        sk: &S,
        source_xe: &mut Source,
        source_xa: &mut Source,
        scratch: &mut ScratchArena<BE>,
    ) where
        R: LWEToBackendMut<BE, Rg> + LWEInfos,
        P: LWEPlaintextToBackendRef<BE, Rg>,
        S: LWESecretToBackendRef<BE, Rg>,
    {
        BE::lwe_encrypt_sk(self, res, pt, sk, source_xe, source_xa, scratch)
    }
);

impl_encryption_delegate!(
    GLWEEncryptSk<BE, Rg>,
    fn glwe_encrypt_sk_tmp_bytes<A>(&self, infos: &A) -> usize
    where
        A: GLWEInfos,
    {
        BE::glwe_encrypt_sk_tmp_bytes(self, infos)
    },
    fn glwe_encrypt_sk<R, P, S>(
        &self,
        res: &mut R,
        pt: &P,
        sk: &S,
        source_xe: &mut Source,
        source_xa: &mut Source,
        scratch: &mut ScratchArena<BE>,
    ) where
        R: GLWEToBackendMut<BE, Rg>,
        P: GLWEToBackendRef<BE, Rg>,
        S: GLWESecretPreparedToBackendRef<BE, Rg>,
    {
        BE::glwe_encrypt_sk(self, res, pt, sk, source_xe, source_xa, scratch)
    },
    fn glwe_encrypt_zero_sk<R, S>(
        &self,
        res: &mut R,
        sk: &S,
        source_xe: &mut Source,
        source_xa: &mut Source,
        scratch: &mut ScratchArena<BE>,
    ) where
        R: GLWEToBackendMut<BE, Rg>,
        S: GLWESecretPreparedToBackendRef<BE, Rg>,
    {
        BE::glwe_encrypt_zero_sk(self, res, sk, source_xe, source_xa, scratch)
    },
    fn glwe_encrypt_sk_with_mask<R, P, S>(
        &self,
        res: &mut R,
        pt: &P,
        sk: &S,
        source_xe: &mut Source,
        scratch: &mut ScratchArena<BE>,
    ) where
        R: GLWEToBackendMut<BE, Rg>,
        P: GLWEToBackendRef<BE, Rg>,
        S: GLWESecretPreparedToBackendRef<BE, Rg>,
    {
        BE::glwe_encrypt_sk_with_mask(self, res, pt, sk, source_xe, scratch)
    }
);

impl_encryption_delegate!(
    GLWEEncryptPk<BE, Rg>,
    fn glwe_encrypt_pk_tmp_bytes<R, K>(&self, res_infos: &R, pk_infos: &K) -> usize
    where
        R: GLWEInfos,
        K: GLWEInfos,
    {
        BE::glwe_encrypt_pk_tmp_bytes(self, res_infos, pk_infos)
    },
    fn glwe_encrypt_pk<R, P, K>(
        &self,
        res: &mut R,
        pt: &P,
        pk: &K,
        source_xu: &mut Source,
        source_xe: &mut Source,
        scratch: &mut ScratchArena<BE>,
    ) where
        R: GLWEToBackendMut<BE, Rg> + GLWEInfos,
        P: GLWEToBackendRef<BE, Rg> + GLWEInfos,
        K: GLWEPublicKeyPreparedToBackendRef<BE, Rg> + GLWEInfos,
    {
        BE::glwe_encrypt_pk(self, res, pt, pk, source_xu, source_xe, scratch)
    },
    fn glwe_encrypt_zero_pk<R, K>(
        &self,
        res: &mut R,
        pk: &K,
        source_xu: &mut Source,
        source_xe: &mut Source,
        scratch: &mut ScratchArena<BE>,
    ) where
        R: GLWEToBackendMut<BE, Rg> + GLWEInfos,
        K: GLWEPublicKeyPreparedToBackendRef<BE, Rg> + GLWEInfos,
    {
        BE::glwe_encrypt_zero_pk(self, res, pk, source_xu, source_xe, scratch)
    },
    fn glwe_encrypt_pk_at_col<R, P, K>(
        &self,
        res: &mut R,
        pt: &P,
        col: usize,
        pk: &K,
        source_xu: &mut Source,
        source_xe: &mut Source,
        scratch: &mut ScratchArena<BE>,
    ) where
        R: GLWEToBackendMut<BE, Rg> + GLWEInfos,
        P: GLWEToBackendRef<BE, Rg> + GLWEInfos,
        K: GLWEPublicKeyPreparedToBackendRef<BE, Rg> + GLWEInfos,
    {
        BE::glwe_encrypt_pk_at_col(self, res, pt, col, pk, source_xu, source_xe, scratch)
    }
);

impl_encryption_delegate!(
    GLWEEncryptPkSmudged<BE, Rg>,
    fn glwe_encrypt_pk_smudged_tmp_bytes<R, K>(&self, res_infos: &R, pk_infos: &K) -> usize
    where
        R: GLWEInfos,
        K: GLWEInfos,
    {
        BE::glwe_encrypt_pk_smudged_tmp_bytes(self, res_infos, pk_infos)
    },
    fn glwe_encrypt_pk_smudged<R, P, K>(
        &self,
        res: &mut R,
        pt: &P,
        pk: &K,
        flood: crate::Noise,
        source_xu: &mut Source,
        source_xe: &mut Source,
        source_smudge: &mut Source,
        scratch: &mut ScratchArena<BE>,
    ) where
        R: GLWEToBackendMut<BE, Rg> + GLWEInfos,
        P: GLWEToBackendRef<BE, Rg> + GLWEInfos,
        K: GLWEPublicKeyPreparedToBackendRef<BE, Rg> + GLWEInfos,
    {
        BE::glwe_encrypt_pk_smudged(self, res, pt, pk, flood, source_xu, source_xe, source_smudge, scratch)
    }
);

impl_encryption_delegate!(
    GLWEPublicKeyGenerate<BE, Rg>,
    fn glwe_public_key_generate_tmp_bytes<A>(&self, infos: &A) -> usize
    where
        A: GLWEInfos,
    {
        BE::glwe_public_key_generate_tmp_bytes(self, infos)
    },
    fn glwe_public_key_generate<R, S>(
        &self,
        res: &mut R,
        sk: &S,
        source_xe: &mut Source,
        source_xa: &mut Source,
        scratch: &mut ScratchArena<'_, BE>,
    ) where
        R: GLWEPublicKeyToBackendMut<BE, Rg> + GetDistributionMut + GLWEInfos,
        S: GLWESecretPreparedToBackendRef<BE, Rg> + GetDistribution,
    {
        BE::glwe_public_key_generate(self, res, sk, source_xe, source_xa, scratch)
    }
);

impl_encryption_delegate!(
    GLWEPublicKeyCompressedGenerate<BE, Rg>,
    fn glwe_public_key_compressed_generate_tmp_bytes<A>(&self, infos: &A) -> usize
    where
        A: GLWEInfos,
    {
        BE::glwe_public_key_compressed_generate_tmp_bytes(self, infos)
    },
    fn glwe_public_key_compressed_generate<R, S>(
        &self,
        res: &mut R,
        sk: &S,
        seed: [u8; 32],
        source_xe: &mut Source,
        scratch: &mut ScratchArena<'_, BE>,
    ) where
        R: GLWEPublicKeyCompressedToBackendMut<BE, Rg> + GLWEPublicKeyCompressedSeedMut + GetDistributionMut + GLWEInfos,
        S: GLWESecretPreparedToBackendRef<BE, Rg> + GetDistribution,
    {
        BE::glwe_public_key_compressed_generate(self, res, sk, seed, source_xe, scratch)
    }
);

impl_encryption_delegate!(
    GGLWEEncryptSk<BE, Rg>,
    fn gglwe_encrypt_sk_tmp_bytes<A>(&self, infos: &A) -> usize
    where
        A: GGLWEInfos,
    {
        BE::gglwe_encrypt_sk_tmp_bytes(self, infos)
    },
    fn gglwe_encrypt_sk<R, P, S>(
        &self,
        res: &mut R,
        pt: &P,
        sk: &S,
        source_xe: &mut Source,
        source_xa: &mut Source,
        scratch: &mut ScratchArena<BE>,
    ) where
        R: GGLWEToBackendMut<BE, Rg>,
        P: ScalarZnxToBackendRef<BE, Rg>,
        S: GLWESecretPreparedToBackendRef<BE, Rg>,
    {
        BE::gglwe_encrypt_sk(self, res, pt, sk, source_xe, source_xa, scratch)
    }
);

impl_encryption_delegate!(
    GGSWEncryptSk<BE, Rg>,
    fn ggsw_encrypt_sk_tmp_bytes<A>(&self, infos: &A) -> usize
    where
        A: GGSWInfos,
    {
        BE::ggsw_encrypt_sk_tmp_bytes(self, infos)
    },
    fn ggsw_encrypt_sk<R, P, S>(
        &self,
        res: &mut R,
        pt: &P,
        sk: &S,
        source_xe: &mut Source,
        source_xa: &mut Source,
        scratch: &mut ScratchArena<BE>,
    ) where
        R: GGSWToBackendMut<BE, Rg> + GGSWInfos + GGSWAtViewMut<BE, Rg>,
        P: ScalarZnxToBackendRef<BE, Rg> + ZnxInfos,
        S: GLWESecretPreparedToBackendRef<BE, Rg> + LWEInfos + GLWEInfos,
    {
        BE::ggsw_encrypt_sk(self, res, pt, sk, source_xe, source_xa, scratch)
    }
);

impl_encryption_delegate!(
    GGSWEncryptPk<BE, Rg>,
    fn ggsw_encrypt_pk_tmp_bytes<R, K>(&self, res_infos: &R, pk_infos: &K) -> usize
    where
        R: GGSWInfos,
        K: GLWEInfos,
    {
        BE::ggsw_encrypt_pk_tmp_bytes(self, res_infos, pk_infos)
    },
    fn ggsw_encrypt_pk<R, P, K>(
        &self,
        res: &mut R,
        pt: &P,
        pk: &K,
        source_xu: &mut Source,
        source_xe: &mut Source,
        scratch: &mut ScratchArena<BE>,
    ) where
        R: GGSWToBackendMut<BE, Rg> + GGSWInfos + GGSWAtViewMut<BE, Rg>,
        P: ScalarZnxToBackendRef<BE, Rg> + ZnxInfos,
        K: GLWEPublicKeyPreparedToBackendRef<BE, Rg> + GLWEInfos,
    {
        BE::ggsw_encrypt_pk(self, res, pt, pk, source_xu, source_xe, scratch)
    }
);

impl_encryption_delegate!(
    GGLWEToGGSWKeyEncryptSk<BE, Rg>,
    fn gglwe_to_ggsw_key_encrypt_sk_tmp_bytes<A>(&self, infos: &A) -> usize
    where
        A: GGLWEInfos,
    {
        BE::gglwe_to_ggsw_key_encrypt_sk_tmp_bytes(self, infos)
    },
    fn gglwe_to_ggsw_key_encrypt_sk<R, S>(
        &self,
        res: &mut R,
        sk: &S,
        source_xe: &mut Source,
        source_xa: &mut Source,
        scratch: &mut ScratchArena<'_, BE>,
    ) where
        R: GGLWEToGGSWKeyToBackendMut<BE, Rg>,
        S: GLWESecretToBackendRef<BE, Rg> + GetDistribution + GLWEInfos,
    {
        BE::gglwe_to_ggsw_key_encrypt_sk(self, res, sk, source_xe, source_xa, scratch)
    }
);

impl_encryption_delegate!(
    GLWESwitchingKeyEncryptSk<BE, Rg>,
    fn glwe_switching_key_encrypt_sk_tmp_bytes<A>(&self, infos: &A) -> usize
    where
        A: GGLWEInfos,
    {
        BE::glwe_switching_key_encrypt_sk_tmp_bytes(self, infos)
    },
    fn glwe_switching_key_encrypt_sk<R, S1, S2>(
        &self,
        res: &mut R,
        sk_in: &S1,
        sk_out: &S2,
        source_xe: &mut Source,
        source_xa: &mut Source,
        scratch: &mut ScratchArena<'_, BE>,
    ) where
        R: GGLWEToBackendMut<BE, Rg> + GLWESwitchingKeyDegreesMut + GGLWEInfos,
        S1: GLWESecretToBackendRef<BE, Rg> + GLWEInfos,
        S2: GLWESecretToBackendRef<BE, Rg> + GetDistribution + GLWEInfos,
    {
        BE::glwe_switching_key_encrypt_sk(self, res, sk_in, sk_out, source_xe, source_xa, scratch)
    }
);

impl_encryption_delegate!(
    GLWETensorKeyEncryptSk<BE, Rg>,
    fn glwe_tensor_key_encrypt_sk_tmp_bytes<A>(&self, infos: &A) -> usize
    where
        A: GGLWEInfos,
    {
        BE::glwe_tensor_key_encrypt_sk_tmp_bytes(self, infos)
    },
    fn glwe_tensor_key_encrypt_sk<R, S>(
        &self,
        res: &mut R,
        sk: &S,
        source_xe: &mut Source,
        source_xa: &mut Source,
        scratch: &mut ScratchArena<'_, BE>,
    ) where
        R: GGLWEToBackendMut<BE, Rg> + GGLWEInfos,
        S: GLWESecretToBackendRef<BE, Rg> + GetDistribution + GLWEInfos,
    {
        BE::glwe_tensor_key_encrypt_sk(self, res, sk, source_xe, source_xa, scratch)
    }
);

impl_encryption_delegate!(
    free GLWEToLWEKeyEncryptSk<BE>,
    fn glwe_to_lwe_key_encrypt_sk_tmp_bytes<A>(&self, infos: &A) -> usize
    where
        A: GGLWEInfos,
    {
        BE::glwe_to_lwe_key_encrypt_sk_tmp_bytes(self, infos)
    },
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
        R: GGLWEToBackendMut<BE> + GGLWEInfos,
    {
        BE::glwe_to_lwe_key_encrypt_sk(self, res, sk_lwe, sk_glwe, source_xe, source_xa, scratch)
    }
);

impl_encryption_delegate!(
    free LWESwitchingKeyEncryptSk<BE>,
    fn lwe_switching_key_encrypt_sk_tmp_bytes<A>(&self, infos: &A) -> usize
    where
        A: GGLWEInfos,
    {
        BE::lwe_switching_key_encrypt_sk_tmp_bytes(self, infos)
    },
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
        S2: LWESecretToBackendRef<BE>,
    {
        BE::lwe_switching_key_encrypt_sk(self, res, sk_lwe_in, sk_lwe_out, source_xe, source_xa, scratch)
    }
);

impl_encryption_delegate!(
    free LWEToGLWEKeyEncryptSk<BE>,
    fn lwe_to_glwe_key_encrypt_sk_tmp_bytes<A>(&self, infos: &A) -> usize
    where
        A: GGLWEInfos,
    {
        BE::lwe_to_glwe_key_encrypt_sk_tmp_bytes(self, infos)
    },
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
        R: GGLWEToBackendMut<BE> + GGLWEInfos,
    {
        BE::lwe_to_glwe_key_encrypt_sk(self, res, sk_lwe, sk_glwe, source_xe, source_xa, scratch)
    }
);

impl_encryption_delegate!(
    GLWEAutomorphismKeyEncryptSk<BE, Rg>,
    fn glwe_automorphism_key_encrypt_sk_tmp_bytes<A>(&self, infos: &A) -> usize
    where
        A: GGLWEInfos,
    {
        BE::glwe_automorphism_key_encrypt_sk_tmp_bytes(self, infos)
    },
    fn glwe_automorphism_key_encrypt_sk<R, S>(
        &self,
        res: &mut R,
        p: i64,
        sk: &S,
        source_xe: &mut Source,
        source_xa: &mut Source,
        scratch: &mut ScratchArena<'_, BE>,
    ) where
        R: GGLWEToBackendMut<BE, Rg> + SetGaloisElement + GGLWEInfos,
        S: GLWESecretToBackendRef<BE, Rg> + GLWEInfos,
    {
        BE::glwe_automorphism_key_encrypt_sk(self, res, p, sk, source_xe, source_xa, scratch)
    }
);

impl_encryption_delegate!(
    GLWECompressedEncryptSk<BE, Rg>,
    fn glwe_compressed_encrypt_sk_tmp_bytes<A>(&self, infos: &A) -> usize
    where
        A: GLWEInfos,
    {
        BE::glwe_compressed_encrypt_sk_tmp_bytes(self, infos)
    },
    fn glwe_compressed_encrypt_sk<R, P, S>(
        &self,
        res: &mut R,
        pt: &P,
        sk: &S,
        seed_xa: [u8; 32],
        source_xe: &mut Source,
        scratch: &mut ScratchArena<BE>,
    ) where
        R: GLWECompressedToBackendMut<BE, Rg> + GLWECompressedSeedMut,
        P: GLWEToBackendRef<BE, Rg>,
        S: GLWESecretPreparedToBackendRef<BE, Rg>,
    {
        BE::glwe_compressed_encrypt_sk(self, res, pt, sk, seed_xa, source_xe, scratch)
    },
    fn glwe_compressed_encrypt_zero_sk<R, S>(
        &self,
        res: &mut R,
        sk: &S,
        seed_xa: [u8; 32],
        source_xe: &mut Source,
        scratch: &mut ScratchArena<BE>,
    ) where
        R: GLWECompressedToBackendMut<BE, Rg> + GLWECompressedSeedMut,
        S: GLWESecretPreparedToBackendRef<BE, Rg>,
    {
        BE::glwe_compressed_encrypt_zero_sk(self, res, sk, seed_xa, source_xe, scratch)
    }
);

impl_encryption_delegate!(
    GGLWECompressedEncryptSk<BE, Rg>,
    fn gglwe_compressed_encrypt_sk_tmp_bytes<A>(&self, infos: &A) -> usize
    where
        A: GGLWEInfos,
    {
        BE::gglwe_compressed_encrypt_sk_tmp_bytes(self, infos)
    },
    fn gglwe_compressed_encrypt_sk<R, P, S>(
        &self,
        res: &mut R,
        pt: &P,
        sk: &S,
        seed: [u8; 32],
        source_xe: &mut Source,
        scratch: &mut ScratchArena<BE>,
    ) where
        R: GGLWECompressedToBackendMut<BE, Rg> + GGLWECompressedSeedMut,
        P: ScalarZnxToBackendRef<BE, Rg>,
        S: GLWESecretPreparedToBackendRef<BE, Rg>,
    {
        BE::gglwe_compressed_encrypt_sk(self, res, pt, sk, seed, source_xe, scratch)
    }
);

impl_encryption_delegate!(
    GGSWCompressedEncryptSk<BE, Rg>,
    fn ggsw_compressed_encrypt_sk_tmp_bytes<A>(&self, infos: &A) -> usize
    where
        A: GGSWInfos,
    {
        BE::ggsw_compressed_encrypt_sk_tmp_bytes(self, infos)
    },
    fn ggsw_compressed_encrypt_sk<R, P, S>(
        &self,
        res: &mut R,
        pt: &P,
        sk: &S,
        seed_xa: [u8; 32],
        source_xe: &mut Source,
        scratch: &mut ScratchArena<BE>,
    ) where
        R: GGSWCompressedToBackendMut<BE, Rg> + GGSWCompressedSeedMut + GGSWInfos,
        P: ScalarZnxToBackendRef<BE, Rg>,
        S: GLWESecretPreparedToBackendRef<BE, Rg>,
    {
        BE::ggsw_compressed_encrypt_sk(self, res, pt, sk, seed_xa, source_xe, scratch)
    }
);

impl_encryption_delegate!(
    GGLWEToGGSWKeyCompressedEncryptSk<BE, Rg>,
    fn gglwe_to_ggsw_key_compressed_encrypt_sk_tmp_bytes<A>(&self, infos: &A) -> usize
    where
        A: GGLWEInfos,
    {
        BE::gglwe_to_ggsw_key_compressed_encrypt_sk_tmp_bytes(self, infos)
    },
    fn gglwe_to_ggsw_key_compressed_encrypt_sk<R, S>(
        &self,
        res: &mut R,
        sk: &S,
        seed_xa: [u8; 32],
        source_xe: &mut Source,
        scratch: &mut ScratchArena<'_, BE>,
    ) where
        R: GGLWEToGGSWKeyCompressedToBackendMut<BE, Rg> + GGLWEInfos,
        S: GLWESecretToBackendRef<BE, Rg> + GetDistribution + GLWEInfos,
    {
        BE::gglwe_to_ggsw_key_compressed_encrypt_sk(self, res, sk, seed_xa, source_xe, scratch)
    }
);

impl_encryption_delegate!(
    GLWEAutomorphismKeyCompressedEncryptSk<BE, Rg>,
    fn glwe_automorphism_key_compressed_encrypt_sk_tmp_bytes<A>(&self, infos: &A) -> usize
    where
        A: GGLWEInfos,
    {
        BE::glwe_automorphism_key_compressed_encrypt_sk_tmp_bytes(self, infos)
    },
    fn glwe_automorphism_key_compressed_encrypt_sk<R, S>(
        &self,
        res: &mut R,
        p: i64,
        sk: &S,
        seed_xa: [u8; 32],
        source_xe: &mut Source,
        scratch: &mut ScratchArena<'_, BE>,
    ) where
        R: GGLWECompressedToBackendMut<BE, Rg> + GGLWECompressedSeedMut + SetGaloisElement + GGLWEInfos,
        S: GLWESecretToBackendRef<BE, Rg> + GLWEInfos,
    {
        BE::glwe_automorphism_key_compressed_encrypt_sk(self, res, p, sk, seed_xa, source_xe, scratch)
    }
);

impl_encryption_delegate!(
    GLWESwitchingKeyCompressedEncryptSk<BE, Rg>,
    fn glwe_switching_key_compressed_encrypt_sk_tmp_bytes<A>(&self, infos: &A) -> usize
    where
        A: GGLWEInfos,
    {
        BE::glwe_switching_key_compressed_encrypt_sk_tmp_bytes(self, infos)
    },
    fn glwe_switching_key_compressed_encrypt_sk<R, S1, S2>(
        &self,
        res: &mut R,
        sk_in: &S1,
        sk_out: &S2,
        seed_xa: [u8; 32],
        source_xe: &mut Source,
        scratch: &mut ScratchArena<'_, BE>,
    ) where
        R: GGLWECompressedToBackendMut<BE, Rg> + GGLWECompressedSeedMut + GLWESwitchingKeyDegreesMut + GGLWEInfos,
        S1: GLWESecretToBackendRef<BE, Rg> + GLWEInfos,
        S2: GLWESecretToBackendRef<BE, Rg> + GetDistribution + GLWEInfos,
    {
        BE::glwe_switching_key_compressed_encrypt_sk(self, res, sk_in, sk_out, seed_xa, source_xe, scratch)
    }
);

impl_encryption_delegate!(
    GLWETensorKeyCompressedEncryptSk<BE, Rg>,
    fn glwe_tensor_key_compressed_encrypt_sk_tmp_bytes<A>(&self, infos: &A) -> usize
    where
        A: GGLWEInfos,
    {
        BE::glwe_tensor_key_compressed_encrypt_sk_tmp_bytes(self, infos)
    },
    fn glwe_tensor_key_compressed_encrypt_sk<R, S>(
        &self,
        res: &mut R,
        sk: &S,
        seed_xa: [u8; 32],
        source_xe: &mut Source,
        scratch: &mut ScratchArena<'_, BE>,
    ) where
        R: GGLWECompressedToBackendMut<BE, Rg> + GGLWEInfos + GGLWECompressedSeedMut,
        S: GLWESecretToBackendRef<BE, Rg> + GetDistribution + GLWEInfos,
    {
        BE::glwe_tensor_key_compressed_encrypt_sk(self, res, sk, seed_xa, source_xe, scratch)
    }
);
