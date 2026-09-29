use crate::layouts::{GLWEKeyswitchShareOwned, GLWEPublicKeyswitchShareOwned};
use poulpy_core::{
    EncryptionInfos, GLWEAdd, GLWEBytesOf, GLWEDecrypt, GLWEEncryptPk, GLWENormalize, GLWESub, ScratchArenaTakeCore,
    SmudgingInfos, VecZnxAddSmudging,
    layouts::{
        GLWEInfos, GLWEPublicKeyPreparedToBackendRef, GLWESecretPreparedToBackendRef, GLWEToBackendMut, GLWEToBackendRef,
        LWEInfos,
    },
};
use poulpy_hal::{
    api::{VecZnxAddAssign, VecZnxSubAssign},
    layouts::{Backend, Module, ScratchArena},
    source::Source,
};

pub trait GLWEKeyswitchMHEProtocolReference<BE: Backend> {
    fn mhe_glwe_keyswitch_share_gen_tmp_bytes_reference<A>(&self, infos: &A) -> usize
    where
        A: GLWEInfos;

    #[allow(clippy::too_many_arguments)]
    fn mhe_glwe_keyswitch_share_gen_reference<C, S1, S2, E>(
        &self,
        res: &mut GLWEKeyswitchShareOwned<BE>,
        ct: &C,
        sk_in: &S1,
        sk_out: &S2,
        flood: &E,
        source_xe: &mut Source,
        scratch: &mut ScratchArena<'_, BE>,
    ) where
        C: GLWEToBackendRef<BE> + GLWEInfos,
        S1: GLWESecretPreparedToBackendRef<BE> + GLWEInfos,
        S2: GLWESecretPreparedToBackendRef<BE> + GLWEInfos,
        E: SmudgingInfos;

    fn mhe_glwe_keyswitch_share_aggregate_reference(
        &self,
        res: &mut GLWEKeyswitchShareOwned<BE>,
        a: &GLWEKeyswitchShareOwned<BE>,
    );

    fn mhe_glwe_keyswitch_share_finalize_tmp_bytes_reference(&self) -> usize;

    fn mhe_glwe_keyswitch_share_finalize_reference<R, C>(
        &self,
        res: &mut R,
        ct: &C,
        share: &GLWEKeyswitchShareOwned<BE>,
        scratch: &mut ScratchArena<'_, BE>,
    ) where
        R: GLWEToBackendMut<BE> + GLWEInfos,
        C: GLWEToBackendRef<BE> + GLWEInfos;
}

impl<BE: Backend> GLWEKeyswitchMHEProtocolReference<BE> for Module<BE>
where
    Self: GLWEDecrypt<BE> + GLWESub<BE> + GLWEAdd<BE> + GLWENormalize<BE> + GLWEBytesOf<BE> + VecZnxAddSmudging<BE>,
{
    fn mhe_glwe_keyswitch_share_gen_tmp_bytes_reference<A>(&self, infos: &A) -> usize
    where
        A: GLWEInfos,
    {
        assert!(
            infos.n().as_usize() == self.n(),
            "invalid layout: degree differs from the module's"
        );
        2 * BE::scratch_aligned(self.glwe_plaintext_bytes_of_from_infos(infos))
            + self.glwe_decrypt_tmp_bytes(infos).max(self.glwe_normalize_tmp_bytes())
    }

    fn mhe_glwe_keyswitch_share_gen_reference<C, S1, S2, E>(
        &self,
        res: &mut GLWEKeyswitchShareOwned<BE>,
        ct: &C,
        sk_in: &S1,
        sk_out: &S2,
        flood: &E,
        source_xe: &mut Source,
        scratch: &mut ScratchArena<'_, BE>,
    ) where
        C: GLWEToBackendRef<BE> + GLWEInfos,
        S1: GLWESecretPreparedToBackendRef<BE> + GLWEInfos,
        S2: GLWESecretPreparedToBackendRef<BE> + GLWEInfos,
        E: SmudgingInfos,
    {
        let res = &mut res.inner;
        assert!(
            ct.n().as_usize() == self.n(),
            "invalid share: ciphertext degree differs from the module's"
        );
        assert!(
            res.n() == ct.n() && res.base2k() == ct.base2k(),
            "invalid share: share and ciphertext layouts differ"
        );
        assert!(res.rank() == 0, "invalid share: share rank differs from 0");
        assert!(
            sk_in.n() == ct.n(),
            "invalid share: input secret degree differs from the ciphertext's"
        );
        assert!(
            sk_out.n() == ct.n(),
            "invalid share: output secret degree differs from the ciphertext's"
        );
        assert!(
            sk_in.rank() == ct.rank(),
            "invalid share: input secret rank differs from the ciphertext's"
        );
        assert!(
            sk_out.rank() == ct.rank(),
            "invalid share: output secret rank differs from the ciphertext's"
        );
        let flood_noise = crate::reference::assert_flood::<BE, _>(res.base2k().as_usize(), res.k().as_usize(), flood);
        let (mut pt_in, scratch_1) = scratch.borrow().take_glwe_plaintext_scratch(ct);
        let (mut pt_out, mut scratch_2) = scratch_1.take_glwe_plaintext_scratch(ct);
        self.glwe_decrypt(ct, &mut pt_in, sk_in, &mut scratch_2);
        self.glwe_decrypt(ct, &mut pt_out, sk_out, &mut scratch_2);
        self.glwe_sub(res, &pt_in, &pt_out);
        self.glwe_normalize_assign(res, &mut scratch_2);
        let base2k = res.base2k().as_usize();
        self.vec_znx_add_smudging(
            base2k,
            GLWEToBackendMut::<BE>::to_backend_mut(res).data_mut(),
            0,
            flood_noise,
            source_xe,
        );
        self.glwe_normalize_assign(res, &mut scratch_2);
    }

    fn mhe_glwe_keyswitch_share_aggregate_reference(
        &self,
        res: &mut GLWEKeyswitchShareOwned<BE>,
        a: &GLWEKeyswitchShareOwned<BE>,
    ) {
        assert!(res.glwe_layout() == a.glwe_layout(), "invalid aggregation: layouts differ");
        self.glwe_add_assign(&mut res.inner, &a.inner);
    }

    fn mhe_glwe_keyswitch_share_finalize_tmp_bytes_reference(&self) -> usize {
        self.glwe_normalize_tmp_bytes()
    }

    fn mhe_glwe_keyswitch_share_finalize_reference<R, C>(
        &self,
        res: &mut R,
        ct: &C,
        share: &GLWEKeyswitchShareOwned<BE>,
        scratch: &mut ScratchArena<'_, BE>,
    ) where
        R: GLWEToBackendMut<BE> + GLWEInfos,
        C: GLWEToBackendRef<BE> + GLWEInfos,
    {
        let share = &share.inner;
        assert!(
            ct.n().as_usize() == self.n(),
            "invalid finalization: ciphertext degree differs from the module's"
        );
        assert!(
            res.n() == ct.n() && res.base2k() == ct.base2k() && res.rank() == ct.rank(),
            "invalid finalization: ciphertext and output layouts differ"
        );
        assert!(
            share.n() == ct.n() && share.base2k() == ct.base2k() && share.rank() == 0,
            "invalid finalization: share layout differs from the ciphertext body"
        );
        self.glwe_add_into(res, ct, share);
        self.glwe_normalize_assign(res, scratch);
    }
}

pub trait GLWEPublicKeyswitchMHEProtocolReference<BE: Backend> {
    fn mhe_glwe_public_keyswitch_share_gen_tmp_bytes_reference<A, B, P>(
        &self,
        ct_infos: &A,
        res_infos: &B,
        pk_infos: &P,
    ) -> usize
    where
        A: GLWEInfos,
        B: GLWEInfos,
        P: GLWEInfos;

    #[allow(clippy::too_many_arguments)]
    fn mhe_glwe_public_keyswitch_share_gen_reference<C, S, K, E1, E2>(
        &self,
        res: &mut GLWEPublicKeyswitchShareOwned<BE>,
        ct: &C,
        sk_in: &S,
        pk_out: &K,
        flood: &E1,
        enc_infos: &E2,
        source_xu: &mut Source,
        source_xe: &mut Source,
        scratch: &mut ScratchArena<'_, BE>,
    ) where
        C: GLWEToBackendRef<BE> + GLWEInfos,
        S: GLWESecretPreparedToBackendRef<BE> + GLWEInfos,
        K: GLWEPublicKeyPreparedToBackendRef<BE> + GLWEInfos,
        E1: SmudgingInfos,
        E2: EncryptionInfos;

    fn mhe_glwe_public_keyswitch_share_aggregate_reference(
        &self,
        res: &mut GLWEPublicKeyswitchShareOwned<BE>,
        a: &GLWEPublicKeyswitchShareOwned<BE>,
    );

    fn mhe_glwe_public_keyswitch_share_finalize_tmp_bytes_reference(&self) -> usize;

    fn mhe_glwe_public_keyswitch_share_finalize_reference<R, C>(
        &self,
        res: &mut R,
        ct: &C,
        share: &GLWEPublicKeyswitchShareOwned<BE>,
        scratch: &mut ScratchArena<'_, BE>,
    ) where
        R: GLWEToBackendMut<BE> + GLWEInfos,
        C: GLWEToBackendRef<BE> + GLWEInfos;
}

impl<BE: Backend> GLWEPublicKeyswitchMHEProtocolReference<BE> for Module<BE>
where
    Self: GLWEDecrypt<BE>
        + GLWEAdd<BE>
        + GLWEEncryptPk<BE>
        + GLWENormalize<BE>
        + GLWEBytesOf<BE>
        + VecZnxAddSmudging<BE>
        + VecZnxSubAssign<BE>
        + VecZnxAddAssign<BE>,
{
    fn mhe_glwe_public_keyswitch_share_gen_tmp_bytes_reference<A, B, P>(&self, ct_infos: &A, res_infos: &B, pk_infos: &P) -> usize
    where
        A: GLWEInfos,
        B: GLWEInfos,
        P: GLWEInfos,
    {
        assert!(
            ct_infos.n().as_usize() == self.n() && res_infos.n() == ct_infos.n() && pk_infos.n() == ct_infos.n(),
            "invalid layout: degree differs from the module's"
        );
        BE::scratch_aligned(self.glwe_plaintext_bytes_of_from_infos(ct_infos))
            + self
                .glwe_decrypt_tmp_bytes(ct_infos)
                .max(self.glwe_normalize_tmp_bytes())
                .max(self.glwe_encrypt_pk_tmp_bytes(res_infos, pk_infos))
    }

    fn mhe_glwe_public_keyswitch_share_gen_reference<C, S, K, E1, E2>(
        &self,
        res: &mut GLWEPublicKeyswitchShareOwned<BE>,
        ct: &C,
        sk_in: &S,
        pk_out: &K,
        flood: &E1,
        enc_infos: &E2,
        source_xu: &mut Source,
        source_xe: &mut Source,
        scratch: &mut ScratchArena<'_, BE>,
    ) where
        C: GLWEToBackendRef<BE> + GLWEInfos,
        S: GLWESecretPreparedToBackendRef<BE> + GLWEInfos,
        K: GLWEPublicKeyPreparedToBackendRef<BE> + GLWEInfos,
        E1: SmudgingInfos,
        E2: EncryptionInfos,
    {
        let res = &mut res.inner;
        assert!(
            ct.n().as_usize() == self.n(),
            "invalid share: ciphertext degree differs from the module's"
        );
        assert!(
            res.n() == ct.n() && res.base2k() == ct.base2k(),
            "invalid share: share and ciphertext layouts differ"
        );
        assert!(
            sk_in.n() == ct.n(),
            "invalid share: input secret degree differs from the ciphertext's"
        );
        assert!(
            sk_in.rank() == ct.rank(),
            "invalid share: input secret rank differs from the ciphertext's"
        );
        assert!(
            pk_out.n() == res.n() && pk_out.base2k() == res.base2k() && pk_out.rank() == res.rank(),
            "invalid share: public key and share layouts differ"
        );
        assert!(pk_out.k() >= res.k(), "invalid share: public key less precise than the share");
        let flood_noise = crate::reference::assert_flood::<BE, _>(ct.base2k().as_usize(), ct.k().as_usize(), flood);
        let (mut pt, mut scratch_1) = scratch.borrow().take_glwe_plaintext_scratch(ct);
        self.glwe_decrypt(ct, &mut pt, sk_in, &mut scratch_1);
        let base2k = ct.base2k().as_usize();
        self.vec_znx_sub_assign(pt.data_mut(), 0, ct.to_backend_ref().data(), 0);
        self.glwe_normalize_assign(&mut pt, &mut scratch_1);
        self.vec_znx_add_smudging(base2k, pt.data_mut(), 0, flood_noise, source_xe);
        self.glwe_normalize_assign(&mut pt, &mut scratch_1);
        self.glwe_encrypt_pk(res, &pt, pk_out, enc_infos, source_xu, source_xe, &mut scratch_1);
    }

    fn mhe_glwe_public_keyswitch_share_aggregate_reference(
        &self,
        res: &mut GLWEPublicKeyswitchShareOwned<BE>,
        a: &GLWEPublicKeyswitchShareOwned<BE>,
    ) {
        assert!(res.glwe_layout() == a.glwe_layout(), "invalid aggregation: layouts differ");
        self.glwe_add_assign(&mut res.inner, &a.inner);
    }

    fn mhe_glwe_public_keyswitch_share_finalize_tmp_bytes_reference(&self) -> usize {
        self.glwe_normalize_tmp_bytes()
    }

    fn mhe_glwe_public_keyswitch_share_finalize_reference<R, C>(
        &self,
        res: &mut R,
        ct: &C,
        share: &GLWEPublicKeyswitchShareOwned<BE>,
        scratch: &mut ScratchArena<'_, BE>,
    ) where
        R: GLWEToBackendMut<BE> + GLWEInfos,
        C: GLWEToBackendRef<BE> + GLWEInfos,
    {
        let share = &share.inner;
        assert!(
            ct.n().as_usize() == self.n(),
            "invalid finalization: ciphertext degree differs from the module's"
        );
        assert!(
            ct.n() == res.n() && ct.base2k() == res.base2k(),
            "invalid finalization: ciphertext and output layouts differ"
        );
        assert!(
            share.n() == res.n() && share.rank() == res.rank(),
            "invalid finalization: share and output layouts differ"
        );
        self.glwe_normalize(res, share, scratch);
        res.set_canonical(false);
        self.vec_znx_add_assign(res.to_backend_mut().data_mut(), 0, ct.to_backend_ref().data(), 0);
        self.glwe_normalize_assign(res, scratch);
    }
}
