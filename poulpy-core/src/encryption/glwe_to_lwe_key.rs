use poulpy_hal::{
    api::{ModuleN, ScratchAvailable, VecZnxAutomorphismAssign, VecZnxAutomorphismAssignTmpBytes},
    layouts::{Backend, Module, Scratch, ZnxView, ZnxViewMut, ZnxZero},
    source::Source,
};

pub use crate::api::GLWEToLWESwitchingKeyEncryptSk;
use crate::{
    EncryptionInfos, GGLWEEncryptSk, ScratchTakeCore,
    layouts::{
        GGLWEInfos, GGLWEToMut, GLWESecret, GLWESecretToRef, LWEInfos, LWESecret, LWESecretToRef, Rank,
        prepared::GLWESecretPreparedFactory,
    },
};

#[doc(hidden)]
pub trait GLWEToLWESwitchingKeyEncryptSkDefault<BE: Backend> {
    fn glwe_to_lwe_key_encrypt_sk_tmp_bytes<A>(&self, infos: &A) -> usize
    where
        A: GGLWEInfos;

    fn glwe_to_lwe_key_encrypt_sk<R, S1, S2, E>(
        &self,
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
}

impl<BE: Backend> GLWEToLWESwitchingKeyEncryptSkDefault<BE> for Module<BE>
where
    Self: ModuleN
        + GGLWEEncryptSk<BE>
        + GLWESecretPreparedFactory<BE>
        + VecZnxAutomorphismAssign<BE>
        + VecZnxAutomorphismAssignTmpBytes,
    Scratch<BE>: ScratchTakeCore<BE>,
{
    fn glwe_to_lwe_key_encrypt_sk_tmp_bytes<A>(&self, infos: &A) -> usize
    where
        A: GGLWEInfos,
    {
        assert_eq!(self.n() as u32, infos.n());

        let lvl_0: usize = self.glwe_secret_prepared_bytes_of(infos.rank_in());
        let lvl_1_sk_lwe_as_glwe: usize =
            GLWESecret::bytes_of(self.n().into(), infos.rank_in()) + self.vec_znx_automorphism_assign_tmp_bytes();
        let lvl_1_encrypt: usize = self.gglwe_encrypt_sk_tmp_bytes(infos);

        lvl_0 + lvl_1_sk_lwe_as_glwe.max(lvl_1_encrypt)
    }

    #[allow(clippy::too_many_arguments)]
    fn glwe_to_lwe_key_encrypt_sk<R, S1, S2, E>(
        &self,
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
        R: GGLWEToMut + GGLWEInfos,
    {
        let sk_lwe: &LWESecret<&[u8]> = &sk_lwe.to_ref();
        let sk_glwe: &GLWESecret<&[u8]> = &sk_glwe.to_ref();

        assert!(sk_lwe.n().0 <= self.n() as u32);
        assert!(
            scratch.available() >= self.glwe_to_lwe_key_encrypt_sk_tmp_bytes(res),
            "scratch.available(): {} < GLWEToLWESwitchingKeyEncryptSk::glwe_to_lwe_key_encrypt_sk_tmp_bytes: {}",
            scratch.available(),
            self.glwe_to_lwe_key_encrypt_sk_tmp_bytes(res)
        );

        let (mut sk_lwe_as_glwe_prep, scratch_1) = scratch.take_glwe_secret_prepared(self, Rank(1));

        {
            let (mut sk_lwe_as_glwe, scratch_2) = scratch_1.take_glwe_secret(self.n().into(), sk_lwe_as_glwe_prep.rank());
            sk_lwe_as_glwe.dist = sk_lwe.dist;
            sk_lwe_as_glwe.data.zero();
            sk_lwe_as_glwe.data.at_mut(0, 0)[..sk_lwe.n().into()].copy_from_slice(sk_lwe.data.at(0, 0));
            self.vec_znx_automorphism_assign(-1, &mut sk_lwe_as_glwe.data.as_vec_znx_mut(), 0, scratch_2);
            self.glwe_secret_prepare(&mut sk_lwe_as_glwe_prep, &sk_lwe_as_glwe);
        }

        self.gglwe_encrypt_sk(
            res,
            &sk_glwe.data,
            &sk_lwe_as_glwe_prep,
            enc_infos,
            source_xe,
            source_xa,
            scratch_1,
        );
    }
}
