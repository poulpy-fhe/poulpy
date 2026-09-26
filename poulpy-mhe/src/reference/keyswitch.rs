use poulpy_core::{
    EncryptionInfos, GLWEAdd, GLWEBytesOf, GLWEDecrypt, GLWEEncryptPk, GLWENormalize, GLWESub, GetDistribution,
    ScratchArenaTakeCore, VecZnxAddNormal,
    layouts::{
        GLWEInfos, GLWEPreparedToBackendRef, GLWESecretPreparedToBackendRef, GLWEToBackendMut, GLWEToBackendRef, LWEInfos,
    },
};
use poulpy_hal::{
    api::{VecZnxAddAssign, VecZnxSubAssign},
    layouts::{Backend, Module, ScratchArena},
    source::Source,
};

pub trait GLWEKeyswitchShareReference<BE: Backend> {
    fn glwe_keyswitch_share_tmp_bytes_reference<A>(&self, infos: &A) -> usize
    where
        A: GLWEInfos;

    #[allow(clippy::too_many_arguments)]
    fn glwe_keyswitch_share_reference<R, C, S1, S2, E>(
        &self,
        res: &mut R,
        ct: &C,
        sk_in: &S1,
        sk_out: &S2,
        flood: &E,
        source_xe: &mut Source,
        scratch: &mut ScratchArena<'_, BE>,
    ) where
        R: GLWEToBackendMut<BE> + GLWEInfos,
        C: GLWEToBackendRef<BE> + GLWEInfos,
        S1: GLWESecretPreparedToBackendRef<BE> + GLWEInfos,
        S2: GLWESecretPreparedToBackendRef<BE> + GLWEInfos,
        E: EncryptionInfos;

    fn glwe_keyswitch_finalize_tmp_bytes_reference(&self) -> usize;

    fn glwe_keyswitch_finalize_reference<R, C, H>(&self, res: &mut R, ct: &C, share: &H, scratch: &mut ScratchArena<'_, BE>)
    where
        R: GLWEToBackendMut<BE> + GLWEInfos,
        C: GLWEToBackendRef<BE> + GLWEInfos,
        H: GLWEToBackendRef<BE> + GLWEInfos;
}

impl<BE: Backend> GLWEKeyswitchShareReference<BE> for Module<BE>
where
    Self: GLWEDecrypt<BE> + GLWESub<BE> + GLWEAdd<BE> + GLWENormalize<BE> + GLWEBytesOf<BE> + VecZnxAddNormal<BE>,
{
    fn glwe_keyswitch_share_tmp_bytes_reference<A>(&self, infos: &A) -> usize
    where
        A: GLWEInfos,
    {
        2 * BE::scratch_aligned(self.glwe_plaintext_bytes_of_from_infos(infos))
            + self.glwe_decrypt_tmp_bytes(infos).max(self.glwe_normalize_tmp_bytes())
    }

    fn glwe_keyswitch_share_reference<R, C, S1, S2, E>(
        &self,
        res: &mut R,
        ct: &C,
        sk_in: &S1,
        sk_out: &S2,
        flood: &E,
        source_xe: &mut Source,
        scratch: &mut ScratchArena<'_, BE>,
    ) where
        R: GLWEToBackendMut<BE> + GLWEInfos,
        C: GLWEToBackendRef<BE> + GLWEInfos,
        S1: GLWESecretPreparedToBackendRef<BE> + GLWEInfos,
        S2: GLWESecretPreparedToBackendRef<BE> + GLWEInfos,
        E: EncryptionInfos,
    {
        let (mut pt_in, scratch_1) = scratch.borrow().take_glwe_plaintext_scratch(ct);
        let (mut pt_out, mut scratch_2) = scratch_1.take_glwe_plaintext_scratch(ct);
        self.glwe_decrypt(ct, &mut pt_in, sk_in, &mut scratch_2);
        self.glwe_decrypt(ct, &mut pt_out, sk_out, &mut scratch_2);
        self.glwe_sub(res, &pt_in, &pt_out);
        let base2k = res.base2k().as_usize();
        self.vec_znx_add_normal(base2k, res.to_backend_mut().data_mut(), 0, flood.noise_infos(), source_xe);
        self.glwe_normalize_assign(res, &mut scratch_2);
    }

    fn glwe_keyswitch_finalize_tmp_bytes_reference(&self) -> usize {
        self.glwe_normalize_tmp_bytes()
    }

    fn glwe_keyswitch_finalize_reference<R, C, H>(&self, res: &mut R, ct: &C, share: &H, scratch: &mut ScratchArena<'_, BE>)
    where
        R: GLWEToBackendMut<BE> + GLWEInfos,
        C: GLWEToBackendRef<BE> + GLWEInfos,
        H: GLWEToBackendRef<BE> + GLWEInfos,
    {
        self.glwe_add_into(res, ct, share);
        self.glwe_normalize_assign(res, scratch);
    }
}

pub trait GLWEPublicKeyswitchShareReference<BE: Backend> {
    fn glwe_public_keyswitch_share_tmp_bytes_reference<A, B, P>(&self, ct_infos: &A, res_infos: &B, pk_infos: &P) -> usize
    where
        A: GLWEInfos,
        B: GLWEInfos,
        P: GLWEInfos;

    #[allow(clippy::too_many_arguments)]
    fn glwe_public_keyswitch_share_reference<R, C, S, K, E1, E2>(
        &self,
        res: &mut R,
        ct: &C,
        sk_in: &S,
        pk_out: &K,
        flood: &E1,
        enc_infos: &E2,
        source_xu: &mut Source,
        source_xe: &mut Source,
        scratch: &mut ScratchArena<'_, BE>,
    ) where
        R: GLWEToBackendMut<BE> + GLWEInfos,
        C: GLWEToBackendRef<BE> + GLWEInfos,
        S: GLWESecretPreparedToBackendRef<BE> + GLWEInfos,
        K: GLWEPreparedToBackendRef<BE> + GetDistribution + GLWEInfos,
        E1: EncryptionInfos,
        E2: EncryptionInfos;

    fn glwe_public_keyswitch_finalize_tmp_bytes_reference(&self) -> usize;

    fn glwe_public_keyswitch_finalize_reference<R, C, H>(
        &self,
        res: &mut R,
        ct: &C,
        share: &H,
        scratch: &mut ScratchArena<'_, BE>,
    ) where
        R: GLWEToBackendMut<BE> + GLWEInfos,
        C: GLWEToBackendRef<BE> + GLWEInfos,
        H: GLWEToBackendRef<BE> + GLWEInfos;
}

impl<BE: Backend> GLWEPublicKeyswitchShareReference<BE> for Module<BE>
where
    Self: GLWEDecrypt<BE>
        + GLWEEncryptPk<BE>
        + GLWENormalize<BE>
        + GLWEBytesOf<BE>
        + VecZnxAddNormal<BE>
        + VecZnxSubAssign<BE>
        + VecZnxAddAssign<BE>,
{
    fn glwe_public_keyswitch_share_tmp_bytes_reference<A, B, P>(&self, ct_infos: &A, res_infos: &B, pk_infos: &P) -> usize
    where
        A: GLWEInfos,
        B: GLWEInfos,
        P: GLWEInfos,
    {
        BE::scratch_aligned(self.glwe_plaintext_bytes_of_from_infos(ct_infos))
            + self
                .glwe_decrypt_tmp_bytes(ct_infos)
                .max(self.glwe_normalize_tmp_bytes())
                .max(self.glwe_encrypt_pk_tmp_bytes(res_infos))
                .max(self.glwe_encrypt_pk_tmp_bytes(pk_infos))
    }

    fn glwe_public_keyswitch_share_reference<R, C, S, K, E1, E2>(
        &self,
        res: &mut R,
        ct: &C,
        sk_in: &S,
        pk_out: &K,
        flood: &E1,
        enc_infos: &E2,
        source_xu: &mut Source,
        source_xe: &mut Source,
        scratch: &mut ScratchArena<'_, BE>,
    ) where
        R: GLWEToBackendMut<BE> + GLWEInfos,
        C: GLWEToBackendRef<BE> + GLWEInfos,
        S: GLWESecretPreparedToBackendRef<BE> + GLWEInfos,
        K: GLWEPreparedToBackendRef<BE> + GetDistribution + GLWEInfos,
        E1: EncryptionInfos,
        E2: EncryptionInfos,
    {
        assert!(pk_out.k() >= res.k(), "invalid share: public key less precise than the share");
        let (mut pt, mut scratch_1) = scratch.borrow().take_glwe_plaintext_scratch(ct);
        self.glwe_decrypt(ct, &mut pt, sk_in, &mut scratch_1);
        let base2k = ct.base2k().as_usize();
        self.vec_znx_sub_assign(pt.data_mut(), 0, ct.to_backend_ref().data(), 0);
        self.vec_znx_add_normal(base2k, pt.data_mut(), 0, flood.noise_infos(), source_xe);
        self.glwe_normalize_assign(&mut pt, &mut scratch_1);
        self.glwe_encrypt_pk(res, &pt, pk_out, enc_infos, source_xu, source_xe, &mut scratch_1);
    }

    fn glwe_public_keyswitch_finalize_tmp_bytes_reference(&self) -> usize {
        self.glwe_normalize_tmp_bytes()
    }

    fn glwe_public_keyswitch_finalize_reference<R, C, H>(
        &self,
        res: &mut R,
        ct: &C,
        share: &H,
        scratch: &mut ScratchArena<'_, BE>,
    ) where
        R: GLWEToBackendMut<BE> + GLWEInfos,
        C: GLWEToBackendRef<BE> + GLWEInfos,
        H: GLWEToBackendRef<BE> + GLWEInfos,
    {
        assert!(
            ct.n() == res.n() && ct.base2k() == res.base2k(),
            "invalid finalization: ciphertext and output layouts differ"
        );
        self.glwe_normalize(res, share, scratch);
        res.set_canonical(false);
        self.vec_znx_add_assign(res.to_backend_mut().data_mut(), 0, ct.to_backend_ref().data(), 0);
        self.glwe_normalize_assign(res, scratch);
    }
}
