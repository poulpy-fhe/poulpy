use poulpy_hal::{
    layouts::{Backend, Module, ScalarZnxToBackendRef, ScratchArena, ZnxInfos},
    source::Source,
};

use crate::{
    GetDistribution, GetDistributionMut,
    api::{
        GGLWECompressedEncryptSk, GGLWEEncryptSk, GGLWEToGGSWKeyCompressedEncryptSk, GGLWEToGGSWKeyEncryptSk,
        GGSWCompressedEncryptSk, GGSWEncryptPk, GGSWEncryptSk, GLWEAutomorphismKeyCompressedEncryptSk,
        GLWEAutomorphismKeyEncryptSk, GLWECompressedEncryptSk, GLWEEncryptPk, GLWEEncryptPkSmudged, GLWEEncryptSk, GLWEMaskFill,
        GLWEPublicKeyCompressedGenerate, GLWEPublicKeyGenerate, GLWESwitchingKeyCompressedEncryptSk, GLWESwitchingKeyEncryptSk,
        GLWETensorKeyCompressedEncryptSk, GLWETensorKeyEncryptSk, GLWEToLWESwitchingKeyEncryptSk, LWEEncryptSk, LWEFillMask,
        LWESwitchingKeyEncrypt, LWEToGLWESwitchingKeyEncryptSk,
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
    oep::EncryptionImpl,
};

macro_rules! impl_encryption_delegate {
    (tensor $trait:ty, $($body:item),+ $(,)?) => {
        impl<BE> $trait for Module<BE>
        where
            BE: Backend + EncryptionImpl,
            Module<BE>: crate::layouts::GLWESecretPreparedFactory<BE> + crate::layouts::GLWESecretTensorFactory<BE>,
        {
            $($body)+
        }
    };
    (vec_znx $trait:ty, $($body:item),+ $(,)?) => {
        impl<BE> $trait for Module<BE>
        where
            BE: Backend + EncryptionImpl,
            Module<BE>: poulpy_hal::api::VecZnxZero<BE>
                + poulpy_hal::api::VecZnxAddScalarAssign<BE>
                + poulpy_hal::api::VecZnxNormalizeAssign<BE>
                + poulpy_hal::api::VecZnxNormalizeTmpBytes,
        {
            $($body)+
        }
    };
    (smudge $trait:ty, $($body:item),+ $(,)?) => {
        impl<BE> $trait for Module<BE>
        where
            BE: Backend + EncryptionImpl,
            Module<BE>: crate::VecZnxAddNoise<BE>
                + poulpy_hal::api::VecZnxNormalizeAssign<BE>
                + poulpy_hal::api::VecZnxNormalizeTmpBytes,
        {
            $($body)+
        }
    };
    (normalize $trait:ty, $($body:item),+ $(,)?) => {
        impl<BE> $trait for Module<BE>
        where
            BE: Backend + EncryptionImpl,
            Module<BE>: crate::GLWENormalize<BE>,
        {
            $($body)+
        }
    };
    ($trait:ty, $($body:item),+ $(,)?) => {
        impl<BE> $trait for Module<BE>
        where
            BE: Backend + EncryptionImpl,
        {
            $($body)+
        }
    };
}

impl_encryption_delegate!(
    GLWEMaskFill<BE>,
    fn fill_glwe_mask_from_source<R>(&self, res: &mut R, source_xa: &mut Source)
    where
        R: GLWEToBackendMut<BE>,
    {
        BE::fill_glwe_mask_from_source(self, res, source_xa)
    },
    fn fill_glwe_mask_from_seed<R>(&self, res: &mut R, seed_xa: [u8; 32])
    where
        R: GLWEToBackendMut<BE>,
    {
        BE::fill_glwe_mask_from_seed(self, res, seed_xa)
    },
    fn fill_glwe_from_source<R>(&self, res: &mut R, source: &mut Source)
    where
        R: GLWEToBackendMut<BE>,
    {
        BE::fill_glwe_from_source(self, res, source);
        res.set_encryption_metadata(None);
    }
);

impl_encryption_delegate!(
    LWEFillMask<BE>,
    fn fill_lwe_mask_from_source<R>(&self, base2k: usize, res: &mut R, source_xa: &mut Source)
    where
        R: LWEToBackendMut<BE>,
    {
        BE::fill_lwe_mask_from_source(self, base2k, res, source_xa)
    },
    fn fill_lwe_mask_from_seed<R>(&self, base2k: usize, res: &mut R, seed_xa: [u8; 32])
    where
        R: LWEToBackendMut<BE>,
    {
        BE::fill_lwe_mask_from_seed(self, base2k, res, seed_xa)
    }
);

impl_encryption_delegate!(
    LWEEncryptSk<BE>,
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
        R: LWEToBackendMut<BE> + LWEInfos,
        P: LWEPlaintextToBackendRef<BE>,
        S: LWESecretToBackendRef<BE>,
    {
        let metadata = Some(crate::EncryptionMetadata::from_secret(sk.to_backend_ref().dist));
        BE::lwe_encrypt_sk(self, res, pt, sk, source_xe, source_xa, scratch);
        res.set_encryption_metadata(metadata);
    }
);

impl_encryption_delegate!(
    GLWEEncryptSk<BE>,
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
        R: GLWEToBackendMut<BE>,
        P: GLWEToBackendRef<BE>,
        S: GLWESecretPreparedToBackendRef<BE>,
    {
        let metadata = Some(crate::EncryptionMetadata::from_secret(sk.to_backend_ref().dist));
        BE::glwe_encrypt_sk(self, res, pt, sk, source_xe, source_xa, scratch);
        res.set_encryption_metadata(metadata);
    },
    fn glwe_encrypt_zero_sk<R, S>(
        &self,
        res: &mut R,
        sk: &S,
        source_xe: &mut Source,
        source_xa: &mut Source,
        scratch: &mut ScratchArena<BE>,
    ) where
        R: GLWEToBackendMut<BE>,
        S: GLWESecretPreparedToBackendRef<BE>,
    {
        let metadata = Some(crate::EncryptionMetadata::from_secret(sk.to_backend_ref().dist));
        BE::glwe_encrypt_zero_sk(self, res, sk, source_xe, source_xa, scratch);
        res.set_encryption_metadata(metadata);
    },
    fn glwe_encrypt_sk_with_mask<R, P, S>(
        &self,
        res: &mut R,
        pt: &P,
        sk: &S,
        source_xe: &mut Source,
        scratch: &mut ScratchArena<BE>,
    ) where
        R: GLWEToBackendMut<BE>,
        P: GLWEToBackendRef<BE>,
        S: GLWESecretPreparedToBackendRef<BE>,
    {
        let metadata = Some(crate::EncryptionMetadata::from_secret(sk.to_backend_ref().dist));
        BE::glwe_encrypt_sk_with_mask(self, res, pt, sk, source_xe, scratch);
        res.set_encryption_metadata(metadata);
    }
);

impl_encryption_delegate!(
    GLWEEncryptPk<BE>,
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
        R: GLWEToBackendMut<BE> + GLWEInfos,
        P: GLWEToBackendRef<BE> + GLWEInfos,
        K: GLWEPublicKeyPreparedToBackendRef<BE> + GLWEInfos,
    {
        let metadata = pk.encryption_metadata();
        BE::glwe_encrypt_pk(self, res, pt, pk, source_xu, source_xe, scratch);
        res.set_encryption_metadata(metadata);
    },
    fn glwe_encrypt_zero_pk<R, K>(
        &self,
        res: &mut R,
        pk: &K,
        source_xu: &mut Source,
        source_xe: &mut Source,
        scratch: &mut ScratchArena<BE>,
    ) where
        R: GLWEToBackendMut<BE> + GLWEInfos,
        K: GLWEPublicKeyPreparedToBackendRef<BE> + GLWEInfos,
    {
        let metadata = pk.encryption_metadata();
        BE::glwe_encrypt_zero_pk(self, res, pk, source_xu, source_xe, scratch);
        res.set_encryption_metadata(metadata);
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
        R: GLWEToBackendMut<BE> + GLWEInfos,
        P: GLWEToBackendRef<BE> + GLWEInfos,
        K: GLWEPublicKeyPreparedToBackendRef<BE> + GLWEInfos,
    {
        let metadata = pk.encryption_metadata();
        BE::glwe_encrypt_pk_at_col(self, res, Some((pt, col)), true, pk, source_xu, source_xe, scratch);
        res.set_encryption_metadata(metadata);
    }
);

impl_encryption_delegate!(
    smudge GLWEEncryptPkSmudged<BE>,
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
        R: GLWEToBackendMut<BE> + GLWEInfos,
        P: GLWEToBackendRef<BE> + GLWEInfos,
        K: GLWEPublicKeyPreparedToBackendRef<BE> + GLWEInfos,
    {
        let metadata = pk.encryption_metadata();
        BE::glwe_encrypt_pk_smudged(
            self,
            res,
            pt,
            pk,
            flood,
            source_xu,
            source_xe,
            source_smudge,
            scratch,
        );
        res.set_encryption_metadata(metadata);
    }
);

impl_encryption_delegate!(
    normalize GLWEPublicKeyGenerate<BE>,
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
        R: GLWEPublicKeyToBackendMut<BE> + GetDistributionMut + GLWEInfos,
        S: GLWESecretPreparedToBackendRef<BE> + GetDistribution,
    {
        let metadata = Some(crate::EncryptionMetadata::from_secret(sk.to_backend_ref().dist));
        BE::glwe_public_key_generate(self, res, sk, source_xe, source_xa, scratch);
        res.set_encryption_metadata(metadata);
    }
);

impl_encryption_delegate!(
    GLWEPublicKeyCompressedGenerate<BE>,
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
        R: GLWEPublicKeyCompressedToBackendMut<BE> + GLWEPublicKeyCompressedSeedMut + GetDistributionMut + GLWEInfos,
        S: GLWESecretPreparedToBackendRef<BE> + GetDistribution,
    {
        let metadata = Some(crate::EncryptionMetadata::from_secret(sk.to_backend_ref().dist));
        BE::glwe_public_key_compressed_generate(self, res, sk, seed, source_xe, scratch);
        res.set_encryption_metadata(metadata);
    }
);

impl_encryption_delegate!(
    GGLWEEncryptSk<BE>,
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
        R: GGLWEToBackendMut<BE>,
        P: ScalarZnxToBackendRef<BE>,
        S: GLWESecretPreparedToBackendRef<BE>,
    {
        let metadata = Some(crate::EncryptionMetadata::from_secret(sk.to_backend_ref().dist));
        BE::gglwe_encrypt_sk(self, res, pt, sk, source_xe, source_xa, scratch);
        res.set_encryption_metadata(metadata);
    }
);

impl_encryption_delegate!(
    GGSWEncryptSk<BE>,
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
        R: GGSWToBackendMut<BE> + GGSWInfos + GGSWAtViewMut<BE>,
        P: ScalarZnxToBackendRef<BE> + ZnxInfos,
        S: GLWESecretPreparedToBackendRef<BE> + LWEInfos + GLWEInfos,
    {
        let metadata = Some(crate::EncryptionMetadata::from_secret(sk.to_backend_ref().dist));
        BE::ggsw_encrypt_sk(self, res, pt, sk, source_xe, source_xa, scratch);
        res.set_encryption_metadata(metadata);
    }
);

impl_encryption_delegate!(
    vec_znx GGSWEncryptPk<BE>,
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
        R: GGSWToBackendMut<BE> + GGSWInfos + GGSWAtViewMut<BE>,
        P: ScalarZnxToBackendRef<BE> + ZnxInfos,
        K: GLWEPublicKeyPreparedToBackendRef<BE> + GLWEInfos,
    {
        let metadata = pk.encryption_metadata();
        BE::ggsw_encrypt_pk(self, res, pt, pk, source_xu, source_xe, scratch);
        res.set_encryption_metadata(metadata);
    }
);

impl_encryption_delegate!(
    GGLWEToGGSWKeyEncryptSk<BE>,
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
        R: GGLWEToGGSWKeyToBackendMut<BE>,
        S: GLWESecretToBackendRef<BE> + GetDistribution + GLWEInfos,
    {
        let metadata = Some(crate::EncryptionMetadata::from_secret(sk.to_backend_ref().dist));
        BE::gglwe_to_ggsw_key_encrypt_sk(self, res, sk, source_xe, source_xa, scratch);
        res.set_encryption_metadata(metadata);
    }
);

impl_encryption_delegate!(
    GLWESwitchingKeyEncryptSk<BE>,
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
        R: GGLWEToBackendMut<BE> + GLWESwitchingKeyDegreesMut + GGLWEInfos,
        S1: GLWESecretToBackendRef<BE> + GLWEInfos,
        S2: GLWESecretToBackendRef<BE> + GetDistribution + GLWEInfos,
    {
        let metadata = Some(crate::EncryptionMetadata::from_secret(sk_out.to_backend_ref().dist));
        BE::glwe_switching_key_encrypt_sk(self, res, sk_in, sk_out, source_xe, source_xa, scratch);
        res.set_encryption_metadata(metadata);
    }
);

impl_encryption_delegate!(
    tensor GLWETensorKeyEncryptSk<BE>,
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
        R: GGLWEToBackendMut<BE> + GGLWEInfos,
        S: GLWESecretToBackendRef<BE> + GetDistribution + GLWEInfos,
    {
        let metadata = Some(crate::EncryptionMetadata::from_secret(sk.to_backend_ref().dist));
        BE::glwe_tensor_key_encrypt_sk(self, res, sk, source_xe, source_xa, scratch);
        res.set_encryption_metadata(metadata);
    }
);

impl_encryption_delegate!(
    GLWEToLWESwitchingKeyEncryptSk<BE>,
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
        let metadata = Some(crate::EncryptionMetadata::from_secret(sk_lwe.to_backend_ref().dist));
        BE::glwe_to_lwe_key_encrypt_sk(self, res, sk_lwe, sk_glwe, source_xe, source_xa, scratch);
        res.set_encryption_metadata(metadata);
    }
);

impl_encryption_delegate!(
    LWESwitchingKeyEncrypt<BE>,
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
        let metadata = Some(crate::EncryptionMetadata::from_secret(sk_lwe_out.to_backend_ref().dist));
        BE::lwe_switching_key_encrypt_sk(self, res, sk_lwe_in, sk_lwe_out, source_xe, source_xa, scratch);
        res.set_encryption_metadata(metadata);
    }
);

impl_encryption_delegate!(
    LWEToGLWESwitchingKeyEncryptSk<BE>,
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
        let metadata = Some(crate::EncryptionMetadata::from_secret(sk_glwe.to_backend_ref().dist));
        BE::lwe_to_glwe_key_encrypt_sk(self, res, sk_lwe, sk_glwe, source_xe, source_xa, scratch);
        res.set_encryption_metadata(metadata);
    }
);

impl_encryption_delegate!(
    GLWEAutomorphismKeyEncryptSk<BE>,
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
        R: GGLWEToBackendMut<BE> + SetGaloisElement + GGLWEInfos,
        S: GLWESecretToBackendRef<BE> + GLWEInfos,
    {
        let metadata = Some(crate::EncryptionMetadata::from_secret(sk.to_backend_ref().dist));
        BE::glwe_automorphism_key_encrypt_sk(self, res, p, sk, source_xe, source_xa, scratch);
        res.set_encryption_metadata(metadata);
    }
);

impl_encryption_delegate!(
    GLWECompressedEncryptSk<BE>,
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
        R: GLWECompressedToBackendMut<BE> + GLWECompressedSeedMut,
        P: GLWEToBackendRef<BE>,
        S: GLWESecretPreparedToBackendRef<BE>,
    {
        let metadata = Some(crate::EncryptionMetadata::from_secret(sk.to_backend_ref().dist));
        BE::glwe_compressed_encrypt_sk(self, res, pt, sk, seed_xa, source_xe, scratch);
        res.set_encryption_metadata(metadata);
    },
    fn glwe_compressed_encrypt_zero_sk<R, S>(
        &self,
        res: &mut R,
        sk: &S,
        seed_xa: [u8; 32],
        source_xe: &mut Source,
        scratch: &mut ScratchArena<BE>,
    ) where
        R: GLWECompressedToBackendMut<BE> + GLWECompressedSeedMut,
        S: GLWESecretPreparedToBackendRef<BE>,
    {
        let metadata = Some(crate::EncryptionMetadata::from_secret(sk.to_backend_ref().dist));
        BE::glwe_compressed_encrypt_zero_sk(self, res, sk, seed_xa, source_xe, scratch);
        res.set_encryption_metadata(metadata);
    }
);

impl_encryption_delegate!(
    GGLWECompressedEncryptSk<BE>,
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
        R: GGLWECompressedToBackendMut<BE> + GGLWECompressedSeedMut,
        P: ScalarZnxToBackendRef<BE>,
        S: GLWESecretPreparedToBackendRef<BE>,
    {
        let metadata = Some(crate::EncryptionMetadata::from_secret(sk.to_backend_ref().dist));
        BE::gglwe_compressed_encrypt_sk(self, res, pt, sk, seed, source_xe, scratch);
        res.set_encryption_metadata(metadata);
    }
);

impl_encryption_delegate!(
    GGSWCompressedEncryptSk<BE>,
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
        R: GGSWCompressedToBackendMut<BE> + GGSWCompressedSeedMut + GGSWInfos,
        P: ScalarZnxToBackendRef<BE>,
        S: GLWESecretPreparedToBackendRef<BE>,
    {
        let metadata = Some(crate::EncryptionMetadata::from_secret(sk.to_backend_ref().dist));
        BE::ggsw_compressed_encrypt_sk(self, res, pt, sk, seed_xa, source_xe, scratch);
        res.set_encryption_metadata(metadata);
    }
);

impl_encryption_delegate!(
    GGLWEToGGSWKeyCompressedEncryptSk<BE>,
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
        R: GGLWEToGGSWKeyCompressedToBackendMut<BE> + GGLWEInfos,
        S: GLWESecretToBackendRef<BE> + GetDistribution + GLWEInfos,
    {
        let metadata = Some(crate::EncryptionMetadata::from_secret(sk.to_backend_ref().dist));
        BE::gglwe_to_ggsw_key_compressed_encrypt_sk(self, res, sk, seed_xa, source_xe, scratch);
        res.set_encryption_metadata(metadata);
    }
);

impl_encryption_delegate!(
    GLWEAutomorphismKeyCompressedEncryptSk<BE>,
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
        R: GGLWECompressedToBackendMut<BE> + GGLWECompressedSeedMut + SetGaloisElement + GGLWEInfos,
        S: GLWESecretToBackendRef<BE> + GLWEInfos,
    {
        let metadata = Some(crate::EncryptionMetadata::from_secret(sk.to_backend_ref().dist));
        BE::glwe_automorphism_key_compressed_encrypt_sk(self, res, p, sk, seed_xa, source_xe, scratch);
        res.set_encryption_metadata(metadata);
    }
);

impl_encryption_delegate!(
    GLWESwitchingKeyCompressedEncryptSk<BE>,
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
        R: GGLWECompressedToBackendMut<BE> + GGLWECompressedSeedMut + GLWESwitchingKeyDegreesMut + GGLWEInfos,
        S1: GLWESecretToBackendRef<BE> + GLWEInfos,
        S2: GLWESecretToBackendRef<BE> + GetDistribution + GLWEInfos,
    {
        let metadata = Some(crate::EncryptionMetadata::from_secret(sk_out.to_backend_ref().dist));
        BE::glwe_switching_key_compressed_encrypt_sk(self, res, sk_in, sk_out, seed_xa, source_xe, scratch);
        res.set_encryption_metadata(metadata);
    }
);

impl_encryption_delegate!(
    tensor GLWETensorKeyCompressedEncryptSk<BE>,
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
        R: GGLWECompressedToBackendMut<BE> + GGLWEInfos + GGLWECompressedSeedMut,
        S: GLWESecretToBackendRef<BE> + GetDistribution + GLWEInfos,
    {
        let metadata = Some(crate::EncryptionMetadata::from_secret(sk.to_backend_ref().dist));
        BE::glwe_tensor_key_compressed_encrypt_sk(self, res, sk, seed_xa, source_xe, scratch);
        res.set_encryption_metadata(metadata);
    }
);
