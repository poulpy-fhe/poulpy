use poulpy_hal::{
    api::ScratchAvailable,
    layouts::{Backend, Module, Scratch, Stats},
};

use crate::{
    GLWENormalize, GLWESub, ScratchTakeCore,
    api::GLWENoise,
    decryption::GLWEDecrypt,
    layouts::{GLWEInfos, GLWEPlaintext, GLWEToRef, LWEInfos, prepared::GLWESecretPreparedToRef},
};

impl<BE: Backend> GLWENoise<BE> for Module<BE>
where
    Module<BE>: GLWEDecrypt<BE> + GLWESub + GLWENormalize<BE>,
    Scratch<BE>: ScratchTakeCore<BE>,
{
    fn glwe_noise_tmp_bytes<A>(&self, infos: &A) -> usize
    where
        A: GLWEInfos,
    {
        let lvl_0: usize = GLWEPlaintext::<Vec<u8>>::bytes_of_from_infos(infos);
        let lvl_1: usize = self.glwe_normalize_tmp_bytes().max(self.glwe_decrypt_tmp_bytes(infos));

        lvl_0 + lvl_1
    }

    fn glwe_noise<R, P, S>(&self, res: &R, pt_want: &P, sk_prepared: &S, scratch: &mut Scratch<BE>) -> Stats
    where
        R: GLWEToRef + GLWEInfos,
        P: GLWEToRef,
        S: GLWESecretPreparedToRef<BE> + GLWEInfos,
    {
        assert!(
            scratch.available() >= self.glwe_noise_tmp_bytes(res),
            "scratch.available(): {} < GLWENoise::glwe_noise_tmp_bytes: {}",
            scratch.available(),
            self.glwe_noise_tmp_bytes(res)
        );

        let (mut pt_have, scratch_1) = scratch.take_glwe_plaintext(res);
        self.glwe_decrypt(res, &mut pt_have, sk_prepared, scratch_1);
        self.glwe_sub_assign(&mut pt_have, pt_want);
        self.glwe_normalize_assign(&mut pt_have, scratch_1);
        pt_have.data.stats(pt_have.base2k().into(), 0)
    }
}
