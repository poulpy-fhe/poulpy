use poulpy_core::{
    GLWEAdd, GLWEBytesOf, GLWECompressedEncryptSk, GLWECopy, GLWEMaskInnerProduct, GLWENormalize, GLWESub, Noise,
    ScratchArenaTakeCore, VecZnxAddNoise,
    layouts::{GLWEInfos, GLWEMaskToBackendRef, GLWESecretPreparedToBackendRef, GLWEToBackendMut, GLWEToBackendRef, LWEInfos},
};
use poulpy_hal::{
    api::{VecZnxAddAssign, VecZnxFillUniformSource},
    layouts::{Backend, Module, ScratchArena},
    source::Source,
};

use crate::layouts::{GLWEEncToShareShareOwned, GLWEShareToEncShareOwned};

pub trait GLWEEncToShareMHEProtocolReference<BE: Backend> {
    fn mhe_glwe_enc_to_share_share_gen_tmp_bytes_reference<A>(&self, ct_infos: &A) -> usize
    where
        A: GLWEInfos;

    #[allow(clippy::too_many_arguments)]
    fn mhe_glwe_enc_to_share_share_gen_reference<P, C, S>(
        &self,
        public: &mut GLWEEncToShareShareOwned<BE>,
        secret: &mut P,
        mask: &C,
        sk: &S,
        flood: Noise,
        source_xm: &mut Source,
        source_smudge: &mut Source,
        scratch: &mut ScratchArena<'_, BE>,
    ) where
        P: GLWEToBackendMut<BE> + GLWEInfos,
        C: GLWEMaskToBackendRef<BE> + GLWEInfos,
        S: GLWESecretPreparedToBackendRef<BE> + GLWEInfos;

    fn mhe_glwe_enc_to_share_share_aggregate_reference(
        &self,
        res: &mut GLWEEncToShareShareOwned<BE>,
        a: &GLWEEncToShareShareOwned<BE>,
    );

    fn mhe_glwe_enc_to_share_share_finalize_tmp_bytes_reference(&self) -> usize;

    fn mhe_glwe_enc_to_share_share_finalize_reference<P, C>(
        &self,
        secret: &mut P,
        ct: &C,
        public: &GLWEEncToShareShareOwned<BE>,
        scratch: &mut ScratchArena<'_, BE>,
    ) where
        P: GLWEToBackendMut<BE> + GLWEInfos,
        C: GLWEToBackendRef<BE> + GLWEInfos;
}

impl<BE: Backend> GLWEEncToShareMHEProtocolReference<BE> for Module<BE>
where
    Self: GLWEMaskInnerProduct<BE>
        + GLWESub<BE>
        + GLWEAdd<BE>
        + GLWENormalize<BE>
        + GLWEBytesOf<BE>
        + VecZnxFillUniformSource<BE>
        + VecZnxAddNoise<BE>
        + VecZnxAddAssign<BE>,
{
    fn mhe_glwe_enc_to_share_share_gen_tmp_bytes_reference<A>(&self, ct_infos: &A) -> usize
    where
        A: GLWEInfos,
    {
        BE::scratch_aligned(self.glwe_plaintext_bytes_of_from_infos(ct_infos))
            + self
                .glwe_mask_inner_product_tmp_bytes(ct_infos)
                .max(self.glwe_normalize_tmp_bytes())
    }

    fn mhe_glwe_enc_to_share_share_gen_reference<P, C, S>(
        &self,
        public: &mut GLWEEncToShareShareOwned<BE>,
        secret: &mut P,
        mask: &C,
        sk: &S,
        flood: Noise,
        source_xm: &mut Source,
        source_smudge: &mut Source,
        scratch: &mut ScratchArena<'_, BE>,
    ) where
        P: GLWEToBackendMut<BE> + GLWEInfos,
        C: GLWEMaskToBackendRef<BE> + GLWEInfos,
        S: GLWESecretPreparedToBackendRef<BE> + GLWEInfos,
    {
        let public = &mut public.inner;
        let k = mask.k().as_usize();
        assert!(
            mask.n().as_usize() == self.n() && sk.n() == mask.n() && sk.rank() == mask.rank(),
            "invalid share: secret and mask layouts differ"
        );
        assert!(
            public.rank().as_usize() == 0 && secret.rank().as_usize() == 0,
            "invalid share: additive shares must have rank zero"
        );
        assert!(
            secret.n() == mask.n()
                && secret.base2k() == mask.base2k()
                && secret.k() == mask.k()
                && public.n() == mask.n()
                && public.base2k() == mask.base2k()
                && public.k() == mask.k(),
            "invalid share: share and mask layouts differ"
        );
        let base2k = public.base2k().as_usize();
        flood.assert_valid_for(base2k, k);
        let tmp_bytes = self.mhe_glwe_enc_to_share_share_gen_tmp_bytes_reference(mask);
        {
            let (mut pt, mut scratch_1) = scratch.borrow().take_glwe_plaintext_scratch(mask);
            self.glwe_mask_inner_product(&mut pt, mask, sk, &mut scratch_1);
            self.vec_znx_fill_uniform_source(base2k, k, secret.to_backend_mut().data_mut(), 0, source_xm);
            secret.set_canonical(true);
            self.glwe_sub(public, &pt, secret);
            self.glwe_normalize_assign(public, &mut scratch_1);
            self.vec_znx_add_noise(
                base2k,
                k,
                GLWEToBackendMut::<BE>::to_backend_mut(public).data_mut(),
                0,
                flood,
                source_smudge,
            );
            self.glwe_normalize_assign(public, &mut scratch_1);
        }
        scratch.wipe(tmp_bytes);
    }

    fn mhe_glwe_enc_to_share_share_aggregate_reference(
        &self,
        res: &mut GLWEEncToShareShareOwned<BE>,
        a: &GLWEEncToShareShareOwned<BE>,
    ) {
        assert!(res.glwe_layout() == a.glwe_layout(), "invalid aggregation: layouts differ");
        self.glwe_add_assign(&mut res.inner, &a.inner);
    }

    fn mhe_glwe_enc_to_share_share_finalize_tmp_bytes_reference(&self) -> usize {
        self.glwe_normalize_tmp_bytes()
    }

    fn mhe_glwe_enc_to_share_share_finalize_reference<P, C>(
        &self,
        secret: &mut P,
        ct: &C,
        public: &GLWEEncToShareShareOwned<BE>,
        scratch: &mut ScratchArena<'_, BE>,
    ) where
        P: GLWEToBackendMut<BE> + GLWEInfos,
        C: GLWEToBackendRef<BE> + GLWEInfos,
    {
        let public = &public.inner;
        assert!(
            ct.n().as_usize() == self.n()
                && ct.n() == secret.n()
                && ct.base2k() == secret.base2k()
                && ct.k() == secret.k()
                && public.n() == ct.n()
                && public.base2k() == ct.base2k()
                && public.k() == ct.k(),
            "invalid finalization: ciphertext and share layouts differ"
        );
        assert!(
            secret.rank().as_usize() == 0 && public.rank().as_usize() == 0,
            "invalid finalization: additive shares must have rank zero"
        );
        self.glwe_add_assign(secret, public);
        self.vec_znx_add_assign(secret.to_backend_mut().data_mut(), 0, ct.to_backend_ref().data(), 0);
        self.glwe_normalize_assign(secret, scratch);
        scratch.wipe(self.mhe_glwe_enc_to_share_share_finalize_tmp_bytes_reference());
    }
}

pub trait GLWEShareToEncMHEProtocolReference<BE: Backend> {
    fn mhe_glwe_share_to_enc_share_gen_tmp_bytes_reference<A, B>(&self, res_infos: &A, secret_infos: &B) -> usize
    where
        A: GLWEInfos,
        B: GLWEInfos;

    #[allow(clippy::too_many_arguments)]
    fn mhe_glwe_share_to_enc_share_gen_reference<P, S>(
        &self,
        res: &mut GLWEShareToEncShareOwned<BE>,
        secret: &P,
        sk: &S,
        seed: [u8; 32],
        source_xe: &mut Source,
        scratch: &mut ScratchArena<'_, BE>,
    ) where
        P: GLWEToBackendRef<BE> + GLWEInfos,
        S: GLWESecretPreparedToBackendRef<BE> + GLWEInfos;
}

impl<BE: Backend> GLWEShareToEncMHEProtocolReference<BE> for Module<BE>
where
    Self: GLWECopy<BE> + GLWECompressedEncryptSk<BE> + GLWEBytesOf<BE>,
{
    fn mhe_glwe_share_to_enc_share_gen_tmp_bytes_reference<A, B>(&self, res_infos: &A, secret_infos: &B) -> usize
    where
        A: GLWEInfos,
        B: GLWEInfos,
    {
        BE::scratch_aligned(self.glwe_plaintext_bytes_of_from_infos(res_infos))
            + self
                .glwe_copy_tmp_bytes(res_infos, secret_infos)
                .max(self.glwe_compressed_encrypt_sk_tmp_bytes(res_infos))
    }

    fn mhe_glwe_share_to_enc_share_gen_reference<P, S>(
        &self,
        res: &mut GLWEShareToEncShareOwned<BE>,
        secret: &P,
        sk: &S,
        seed: [u8; 32],
        source_xe: &mut Source,
        scratch: &mut ScratchArena<'_, BE>,
    ) where
        P: GLWEToBackendRef<BE> + GLWEInfos,
        S: GLWESecretPreparedToBackendRef<BE> + GLWEInfos,
    {
        let res = &mut res.inner;
        assert!(
            secret.k() <= res.k(),
            "invalid share: secret share more precise than the output"
        );
        assert!(
            res.n().as_usize() == self.n() && secret.n() == res.n() && sk.n() == res.n() && sk.rank() == res.rank(),
            "invalid share: plaintext, secret and output layouts differ"
        );
        assert!(
            secret.rank().as_usize() == 0,
            "invalid share: additive share must have rank zero"
        );
        let infos = res.glwe_layout();
        // Encryption samples directly on the output's grid, down to its low bits.
        let tmp_bytes = self.mhe_glwe_share_to_enc_share_gen_tmp_bytes_reference(&infos, secret);
        {
            let (mut pt, mut scratch_1) = scratch.borrow().take_glwe_plaintext_scratch(&infos);
            self.glwe_copy(&mut pt, secret, &mut scratch_1);
            self.glwe_compressed_encrypt_sk(res, &pt, sk, seed, source_xe, &mut scratch_1);
        }
        scratch.wipe(tmp_bytes);
    }
}
