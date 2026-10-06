use crate::layouts::{GLWEPrivateKeyswitchShareOwned, GLWEPublicKeyswitchShareOwned};
use poulpy_core::{
    ComponentNoise, FreshNoiseEstimate, GLWEAdd, GLWEBytesOf, GLWEEncryptPkSmudged, GLWEMaskInnerProduct, GLWENormalize, GLWESub,
    GetDistribution, Noise, ScratchArenaTakeCore, VecZnxAddNoise,
    layouts::{
        GLWEInfos, GLWEMaskToBackendRef, GLWEPublicKeyPreparedToBackendRef, GLWESecretPreparedToBackendRef, GLWEToBackendMut,
        GLWEToBackendRef, LWEInfos,
    },
};
use poulpy_hal::{
    api::VecZnxAddAssign,
    layouts::{Backend, Module, ScratchArena},
    source::Source,
};

pub trait GLWEPrivateKeyswitchMHEProtocolReference<BE: Backend> {
    fn mhe_glwe_private_keyswitch_share_gen_tmp_bytes_reference<A>(&self, infos: &A) -> usize
    where
        A: GLWEInfos;

    #[allow(clippy::too_many_arguments)]
    fn mhe_glwe_private_keyswitch_share_gen_reference<C, S1, S2>(
        &self,
        res: &mut GLWEPrivateKeyswitchShareOwned<BE>,
        mask: &C,
        sk_in: &S1,
        sk_out: &S2,
        flood: Noise,
        source_smudge: &mut Source,
        scratch: &mut ScratchArena<'_, BE>,
    ) where
        C: GLWEMaskToBackendRef<BE> + GLWEInfos,
        S1: GLWESecretPreparedToBackendRef<BE> + GLWEInfos,
        S2: GLWESecretPreparedToBackendRef<BE> + GLWEInfos;

    fn mhe_glwe_private_keyswitch_share_aggregate_reference(
        &self,
        res: &mut GLWEPrivateKeyswitchShareOwned<BE>,
        a: &GLWEPrivateKeyswitchShareOwned<BE>,
    );

    fn mhe_glwe_private_keyswitch_share_finalize_tmp_bytes_reference(&self) -> usize;

    fn mhe_glwe_private_keyswitch_share_finalize_reference<R, C>(
        &self,
        res: &mut R,
        ct: &C,
        share: &GLWEPrivateKeyswitchShareOwned<BE>,
        scratch: &mut ScratchArena<'_, BE>,
    ) where
        R: GLWEToBackendMut<BE> + GLWEInfos,
        C: GLWEToBackendRef<BE> + GLWEInfos;
}

impl<BE: Backend> GLWEPrivateKeyswitchMHEProtocolReference<BE> for Module<BE>
where
    Self: GLWEMaskInnerProduct<BE> + GLWESub<BE> + GLWEAdd<BE> + GLWENormalize<BE> + GLWEBytesOf<BE> + VecZnxAddNoise<BE>,
{
    fn mhe_glwe_private_keyswitch_share_gen_tmp_bytes_reference<A>(&self, infos: &A) -> usize
    where
        A: GLWEInfos,
    {
        assert!(
            infos.n().as_usize() == self.n(),
            "invalid layout: degree differs from the module's"
        );
        2 * BE::scratch_aligned(self.glwe_plaintext_bytes_of_from_infos(infos))
            + self
                .glwe_mask_inner_product_tmp_bytes(infos)
                .max(self.glwe_normalize_tmp_bytes())
    }

    fn mhe_glwe_private_keyswitch_share_gen_reference<C, S1, S2>(
        &self,
        res: &mut GLWEPrivateKeyswitchShareOwned<BE>,
        mask: &C,
        sk_in: &S1,
        sk_out: &S2,
        flood: Noise,
        source_smudge: &mut Source,
        scratch: &mut ScratchArena<'_, BE>,
    ) where
        C: GLWEMaskToBackendRef<BE> + GLWEInfos,
        S1: GLWESecretPreparedToBackendRef<BE> + GLWEInfos,
        S2: GLWESecretPreparedToBackendRef<BE> + GLWEInfos,
    {
        let res = &mut res.inner;
        assert!(
            mask.n().as_usize() == self.n(),
            "invalid share: mask degree differs from the module's"
        );
        assert!(
            res.n() == mask.n() && res.base2k() == mask.base2k(),
            "invalid share: share and mask layouts differ"
        );
        assert!(res.rank() == 0, "invalid share: share rank differs from 0");
        assert!(
            sk_in.n() == mask.n(),
            "invalid share: input secret degree differs from the mask's"
        );
        assert!(
            sk_out.n() == mask.n(),
            "invalid share: output secret degree differs from the mask's"
        );
        assert!(
            sk_in.rank() == mask.rank(),
            "invalid share: input secret rank differs from the mask's"
        );
        assert!(
            sk_out.rank() == mask.rank(),
            "invalid share: output secret rank differs from the mask's"
        );
        let (base2k, k) = (res.base2k().as_usize(), res.k().as_usize());
        flood.assert_valid_for(base2k, k);
        let tmp_bytes = self.mhe_glwe_private_keyswitch_share_gen_tmp_bytes_reference(mask);
        {
            let (mut pt_in, scratch_1) = scratch.borrow().take_glwe_plaintext_scratch(mask);
            let (mut pt_out, mut scratch_2) = scratch_1.take_glwe_plaintext_scratch(mask);
            self.glwe_mask_inner_product(&mut pt_in, mask, sk_in, &mut scratch_2);
            self.glwe_mask_inner_product(&mut pt_out, mask, sk_out, &mut scratch_2);
            self.glwe_sub(res, &pt_in, &pt_out);
            self.glwe_normalize_assign(res, &mut scratch_2);
            self.vec_znx_add_noise(
                base2k,
                k,
                GLWEToBackendMut::<BE>::to_backend_mut(res).data_mut(),
                0,
                flood,
                source_smudge,
            );
            self.glwe_normalize_assign(res, &mut scratch_2);
        }
        // Model the conversion of two cropped inner products followed by
        // normalization as one output-grid ulp when the share is narrower.
        // As for other rounding terms, this is an effective estimate.
        let rounding = if res.k() < mask.k() { 1.0 } else { 0.0 };
        GLWEToBackendMut::<BE>::set_noise(
            res,
            Some(
                ComponentNoise::from_secret_at(*sk_out.to_backend_ref().dist(), res.k(), res.rank().as_usize())
                    .with_body_noise(FreshNoiseEstimate::new(super::flood_variance(flood) + rounding, res.k())),
            ),
        );
        scratch.wipe(tmp_bytes);
    }

    fn mhe_glwe_private_keyswitch_share_aggregate_reference(
        &self,
        res: &mut GLWEPrivateKeyswitchShareOwned<BE>,
        a: &GLWEPrivateKeyswitchShareOwned<BE>,
    ) {
        assert!(res.glwe_layout() == a.glwe_layout(), "invalid aggregation: layouts differ");
        let metadata = super::aggregate_metadata(res.noise(), a.noise());
        self.glwe_add_assign(&mut res.inner, &a.inner);
        GLWEToBackendMut::<BE>::set_noise(res, metadata);
    }

    fn mhe_glwe_private_keyswitch_share_finalize_tmp_bytes_reference(&self) -> usize {
        self.glwe_normalize_tmp_bytes()
    }

    fn mhe_glwe_private_keyswitch_share_finalize_reference<R, C>(
        &self,
        res: &mut R,
        ct: &C,
        share: &GLWEPrivateKeyswitchShareOwned<BE>,
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
        res.set_noise(None);
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
    fn mhe_glwe_public_keyswitch_share_gen_reference<C, S, K>(
        &self,
        res: &mut GLWEPublicKeyswitchShareOwned<BE>,
        mask: &C,
        sk_in: &S,
        pk_out: &K,
        flood: Noise,
        source_xu: &mut Source,
        source_xe: &mut Source,
        source_smudge: &mut Source,
        scratch: &mut ScratchArena<'_, BE>,
    ) where
        C: GLWEMaskToBackendRef<BE> + GLWEInfos,
        S: GLWESecretPreparedToBackendRef<BE> + GLWEInfos,
        K: GLWEPublicKeyPreparedToBackendRef<BE> + GLWEInfos;

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
    Self: GLWEMaskInnerProduct<BE>
        + GLWEAdd<BE>
        + GLWEEncryptPkSmudged<BE>
        + GLWENormalize<BE>
        + GLWEBytesOf<BE>
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
                .glwe_mask_inner_product_tmp_bytes(ct_infos)
                .max(self.glwe_encrypt_pk_smudged_tmp_bytes(res_infos, pk_infos))
    }

    fn mhe_glwe_public_keyswitch_share_gen_reference<C, S, K>(
        &self,
        res: &mut GLWEPublicKeyswitchShareOwned<BE>,
        mask: &C,
        sk_in: &S,
        pk_out: &K,
        flood: Noise,
        source_xu: &mut Source,
        source_xe: &mut Source,
        source_smudge: &mut Source,
        scratch: &mut ScratchArena<'_, BE>,
    ) where
        C: GLWEMaskToBackendRef<BE> + GLWEInfos,
        S: GLWESecretPreparedToBackendRef<BE> + GLWEInfos,
        K: GLWEPublicKeyPreparedToBackendRef<BE> + GLWEInfos,
    {
        let res = &mut res.inner;
        assert!(
            mask.n().as_usize() == self.n(),
            "invalid share: mask degree differs from the module's"
        );
        assert!(
            res.n() == mask.n() && res.base2k() == mask.base2k(),
            "invalid share: share and mask layouts differ"
        );
        assert!(
            sk_in.n() == mask.n(),
            "invalid share: input secret degree differs from the mask's"
        );
        assert!(
            sk_in.rank() == mask.rank(),
            "invalid share: input secret rank differs from the mask's"
        );
        assert!(
            pk_out.n() == res.n() && pk_out.base2k() == res.base2k() && pk_out.rank() == res.rank(),
            "invalid share: public key and share layouts differ"
        );
        assert!(pk_out.k() >= res.k(), "invalid share: public key less precise than the share");
        flood.assert_valid_for(res.base2k().as_usize(), res.k().as_usize());
        super::assert_public_key_distribution::<BE, _>(pk_out);
        let tmp_bytes = self.mhe_glwe_public_keyswitch_share_gen_tmp_bytes_reference(mask, &*res, pk_out);
        {
            let (mut pt, mut scratch_1) = scratch.borrow().take_glwe_plaintext_scratch(mask);
            self.glwe_mask_inner_product(&mut pt, mask, sk_in, &mut scratch_1);
            self.glwe_encrypt_pk_smudged(res, &pt, pk_out, flood, source_xu, source_xe, source_smudge, &mut scratch_1);
        }
        if res.k() < mask.k() && pk_out.k() == res.k() {
            if let Some(metadata) = res.noise() {
                let mut components = metadata.components().to_vec();
                components[0] = FreshNoiseEstimate::new(metadata.body().variance_at(res.k()) + 1.0, res.k());
                GLWEToBackendMut::<BE>::set_noise(res, Some(metadata.with_components(components)));
            }
        }
        scratch.wipe(tmp_bytes);
    }

    fn mhe_glwe_public_keyswitch_share_aggregate_reference(
        &self,
        res: &mut GLWEPublicKeyswitchShareOwned<BE>,
        a: &GLWEPublicKeyswitchShareOwned<BE>,
    ) {
        assert!(res.glwe_layout() == a.glwe_layout(), "invalid aggregation: layouts differ");
        let metadata = super::aggregate_common_key_metadata(res.noise(), a.noise(), res.n().as_usize());
        self.glwe_add_assign(&mut res.inner, &a.inner);
        GLWEToBackendMut::<BE>::set_noise(res, metadata);
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
        res.set_noise(None);
    }
}
