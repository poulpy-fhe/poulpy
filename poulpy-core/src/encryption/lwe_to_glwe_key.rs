use poulpy_hal::{
    api::{ModuleN, ScratchAvailable, VecZnxAutomorphismAssign, VecZnxAutomorphismAssignTmpBytes},
    layouts::{Backend, Module, Scratch, ZnxView, ZnxViewMut},
    source::Source,
};

pub use crate::api::LWEToGLWESwitchingKeyEncryptSk;
use crate::{
    EncryptionInfos, GGLWEEncryptSk, ScratchTakeCore,
    layouts::{
        GGLWEInfos, GGLWEToMut, GLWESecret, GLWESecretPreparedFactory, GLWESecretPreparedToRef, LWEInfos, LWESecret,
        LWESecretToRef, Rank,
    },
};

#[doc(hidden)]
pub trait LWEToGLWESwitchingKeyEncryptSkDefault<BE: Backend> {
    fn lwe_to_glwe_key_encrypt_sk_tmp_bytes<A>(&self, infos: &A) -> usize
    where
        A: GGLWEInfos;

    fn lwe_to_glwe_key_encrypt_sk<R, S1, S2, E>(
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
        S2: GLWESecretPreparedToRef<BE>,
        E: EncryptionInfos,
        R: GGLWEToMut + GGLWEInfos;
}

impl<BE: Backend> LWEToGLWESwitchingKeyEncryptSkDefault<BE> for Module<BE>
where
    Self: ModuleN
        + GGLWEEncryptSk<BE>
        + VecZnxAutomorphismAssign<BE>
        + GLWESecretPreparedFactory<BE>
        + VecZnxAutomorphismAssignTmpBytes,
    Scratch<BE>: ScratchTakeCore<BE>,
{
    fn lwe_to_glwe_key_encrypt_sk_tmp_bytes<A>(&self, infos: &A) -> usize
    where
        A: GGLWEInfos,
    {
        debug_assert_eq!(
            infos.rank_in(),
            Rank(1),
            "rank_in != 1 is not supported for LWEToGLWEKeyPrepared"
        );
        assert_eq!(self.n() as u32, infos.n());

        let lvl_0: usize = GLWESecret::bytes_of(self.n().into(), infos.rank_in());
        let lvl_1: usize = self
            .gglwe_encrypt_sk_tmp_bytes(infos)
            .max(self.vec_znx_automorphism_assign_tmp_bytes());

        lvl_0 + lvl_1
    }

    #[allow(clippy::too_many_arguments)]
    fn lwe_to_glwe_key_encrypt_sk<R, S1, S2, E>(
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
        S2: GLWESecretPreparedToRef<BE>,
        E: EncryptionInfos,
        R: GGLWEToMut + GGLWEInfos,
    {
        let sk_lwe: &LWESecret<&[u8]> = &sk_lwe.to_ref();

        assert!(sk_lwe.n().0 <= self.n() as u32);
        assert!(
            scratch.available() >= self.lwe_to_glwe_key_encrypt_sk_tmp_bytes(res),
            "scratch.available(): {} < LWEToGLWESwitchingKeyEncryptSk::lwe_to_glwe_key_encrypt_sk_tmp_bytes: {}",
            scratch.available(),
            self.lwe_to_glwe_key_encrypt_sk_tmp_bytes(res)
        );

        let (mut sk_lwe_as_glwe, scratch_1) = scratch.take_glwe_secret(self.n().into(), Rank(1));
        sk_lwe_as_glwe.dist = sk_lwe.dist;

        sk_lwe_as_glwe.data.at_mut(0, 0)[..sk_lwe.n().into()].copy_from_slice(sk_lwe.data.at(0, 0));
        sk_lwe_as_glwe.data.at_mut(0, 0)[sk_lwe.n().into()..].fill(0);
        self.vec_znx_automorphism_assign(-1, &mut sk_lwe_as_glwe.data.as_vec_znx_mut(), 0, scratch_1);

        self.gglwe_encrypt_sk(res, &sk_lwe_as_glwe.data, sk_glwe, enc_infos, source_xe, source_xa, scratch_1);
    }
}
