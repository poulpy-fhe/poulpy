use poulpy_hal::{
    api::ModuleN,
    layouts::{Backend, Module, ScratchArena},
    source::Source,
};

use crate::{
    EncryptionInfos, GGLWEEncryptSk, GetDistribution, ScratchArenaTakeCore,
    layouts::{
        GGLWEInfos, GGLWEToBackendMut, GLWEInfos, GLWESecretTensor, GLWESecretTensorFactory, GLWESecretToBackendRef,
        prepared::GLWESecretPreparedFactory,
    },
};

#[doc(hidden)]
pub trait GLWETensorKeyEncryptSkDefault<BE: Backend> {
    fn glwe_tensor_key_encrypt_sk_tmp_bytes_default<A>(&self, infos: &A) -> usize
    where
        A: GGLWEInfos;

    fn glwe_tensor_key_encrypt_sk_default<R, S, E>(
        &self,
        res: &mut R,
        sk: &S,
        enc_infos: &E,
        source_xe: &mut Source,
        source_xa: &mut Source,
        scratch: &mut ScratchArena<'_, BE>,
    ) where
        R: GGLWEToBackendMut<BE> + GGLWEInfos,
        E: EncryptionInfos,
        S: GLWESecretToBackendRef<BE> + GetDistribution + GLWEInfos;
}

impl<BE: Backend> GLWETensorKeyEncryptSkDefault<BE> for Module<BE>
where
    Self: ModuleN + GGLWEEncryptSk<BE> + GLWESecretPreparedFactory<BE> + GLWESecretTensorFactory<BE>,
    for<'s> ScratchArena<'s, BE>: ScratchArenaTakeCore<'s, BE>,
{
    fn glwe_tensor_key_encrypt_sk_tmp_bytes_default<A>(&self, infos: &A) -> usize
    where
        A: GGLWEInfos,
    {
        assert_eq!(self.n() as u32, infos.n());

        let sk_prepared: usize = self.glwe_secret_prepared_bytes_of(infos.rank_out());
        let sk_tensor: usize = GLWESecretTensor::bytes_of_from_infos(infos);

        let lvl_0: usize = sk_prepared;
        let lvl_1: usize = sk_tensor;
        let lvl_2_prepare: usize = self.glwe_secret_tensor_prepare_tmp_bytes(infos.rank());
        let lvl_2: usize = lvl_2_prepare;
        let lvl_3_encrypt: usize = self.gglwe_encrypt_sk_tmp_bytes(infos);

        lvl_0 + lvl_1 + lvl_2 + lvl_3_encrypt
    }

    fn glwe_tensor_key_encrypt_sk_default<R, S, E>(
        &self,
        res: &mut R,
        sk: &S,
        enc_infos: &E,
        source_xe: &mut Source,
        source_xa: &mut Source,
        scratch: &mut ScratchArena<'_, BE>,
    ) where
        R: GGLWEToBackendMut<BE> + GGLWEInfos,
        E: EncryptionInfos,
        S: GLWESecretToBackendRef<BE> + GetDistribution + GLWEInfos,
    {
        assert_eq!(res.rank_out(), sk.rank());
        assert_eq!(res.n(), sk.n());
        assert!(
            scratch.available() >= self.glwe_tensor_key_encrypt_sk_tmp_bytes_default(res),
            "scratch.available(): {} < GLWETensorKeyEncryptSk::glwe_tensor_key_encrypt_sk_tmp_bytes: {}",
            scratch.available(),
            self.glwe_tensor_key_encrypt_sk_tmp_bytes_default(res)
        );

        let scratch = scratch.borrow();
        let (mut sk_prepared, scratch_1) = scratch.take_glwe_secret_prepared_scratch(self, res.rank());
        let (mut sk_tensor, scratch_2) = scratch_1.take_glwe_secret_tensor_scratch(self.n().into(), res.rank());
        let (mut tensor_scratch, scratch_3) = scratch_2.split_at(self.glwe_secret_tensor_prepare_tmp_bytes(res.rank()));
        self.glwe_secret_prepare(&mut sk_prepared, sk);
        self.glwe_secret_tensor_prepare(&mut sk_tensor, sk, &mut tensor_scratch);

        let (mut enc_scratch, _scratch_4) = scratch_3.split_at(self.gglwe_encrypt_sk_tmp_bytes(res));
        let sk_tensor_data = &mut sk_tensor.data;
        self.gglwe_encrypt_sk(
            res,
            &sk_tensor_data,
            &sk_prepared,
            enc_infos,
            source_xe,
            source_xa,
            &mut enc_scratch,
        );
    }
}
