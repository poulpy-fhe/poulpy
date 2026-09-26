use poulpy_core::{
    EncryptionInfos, GLWEAdd, GLWEBytesOf, GLWEDecrypt, GLWENormalize, GLWESub, ScratchArenaTakeCore, VecZnxAddNormal,
    layouts::{GLWEInfos, GLWELayout, GLWESecretPreparedToBackendRef, GLWEToBackendMut, GLWEToBackendRef, LWEInfos, Rank},
};
use poulpy_hal::{
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
        let infos = plaintext_layout(ct);
        let (mut pt_in, scratch_1) = scratch.borrow().take_glwe_plaintext_scratch(&infos);
        let (mut pt_out, mut scratch_2) = scratch_1.take_glwe_plaintext_scratch(&infos);
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

fn plaintext_layout<A: GLWEInfos>(infos: &A) -> GLWELayout {
    GLWELayout {
        n: infos.n(),
        base2k: infos.base2k(),
        k: infos.k(),
        rank: Rank(0),
    }
}
