use crate::{
    CKKSMeta,
    leveled::api::{
        CKKSAddOps, CKKSAllOpsTmpBytes, CKKSConjugateOps, CKKSDecrypt, CKKSEncrypt, CKKSMulOps, CKKSNegOps, CKKSPow2Ops,
        CKKSRescaleOps, CKKSRotateOps, CKKSSubOps,
    },
    oep::CKKSImpl,
};
use poulpy_core::{
    GLWEAutomorphism, GLWEAutomorphismKeyEncryptSk, GLWEMulConst, GLWEMulPlain, GLWERotate, GLWEShift, GLWETensorKeyEncryptSk,
    GLWETensoring,
    layouts::{GGLWEInfos, GLWEAutomorphismKeyPreparedFactory, GLWEInfos, GLWETensorKeyPreparedFactory},
};
use poulpy_hal::{
    api::{ModuleN, VecZnxLsh, VecZnxLshTmpBytes, VecZnxRsh, VecZnxRshAddInto, VecZnxRshSub, VecZnxRshTmpBytes},
    layouts::{Backend, Module},
};

impl<BE: Backend + CKKSImpl<BE>> CKKSAllOpsTmpBytes<BE> for Module<BE>
where
    Self: CKKSEncrypt<BE>
        + CKKSDecrypt<BE>
        + CKKSAddOps<BE>
        + CKKSConjugateOps<BE>
        + CKKSSubOps<BE>
        + CKKSNegOps<BE>
        + CKKSPow2Ops<BE>
        + CKKSRescaleOps<BE>
        + CKKSRotateOps<BE>
        + CKKSMulOps<BE>
        + GLWEAutomorphism<BE>
        + GLWEAutomorphismKeyEncryptSk<BE>
        + GLWEAutomorphismKeyPreparedFactory<BE>
        + ModuleN
        + GLWEShift<BE>
        + GLWEMulPlain<BE>
        + GLWEMulConst<BE>
        + GLWERotate<BE>
        + GLWETensoring<BE>
        + GLWETensorKeyEncryptSk<BE>
        + GLWETensorKeyPreparedFactory<BE>
        + VecZnxLsh<BE>
        + VecZnxLshTmpBytes
        + VecZnxRsh<BE>
        + VecZnxRshAddInto<BE>
        + VecZnxRshSub<BE>
        + VecZnxRshTmpBytes,
{
    fn ckks_all_ops_tmp_bytes<C, T>(&self, ct_infos: &C, tsk_infos: &T, pt_prec: &CKKSMeta) -> usize
    where
        C: GLWEInfos,
        T: GGLWEInfos,
    {
        self.ckks_encrypt_sk_tmp_bytes(ct_infos)
            .max(self.ckks_decrypt_tmp_bytes(ct_infos))
            .max(self.ckks_add_tmp_bytes())
            .max(self.ckks_add_pt_vec_znx_tmp_bytes())
            .max(self.ckks_add_pt_vec_rnx_tmp_bytes(ct_infos, ct_infos, pt_prec))
            .max(self.ckks_add_pt_const_tmp_bytes())
            .max(self.ckks_sub_tmp_bytes())
            .max(self.ckks_sub_pt_vec_znx_tmp_bytes())
            .max(self.ckks_sub_pt_vec_rnx_tmp_bytes(ct_infos, ct_infos, pt_prec))
            .max(self.ckks_sub_pt_const_tmp_bytes())
            .max(self.ckks_neg_tmp_bytes())
            .max(self.ckks_mul_pow2_tmp_bytes())
            .max(self.ckks_div_pow2_tmp_bytes())
            .max(self.ckks_rescale_tmp_bytes())
            .max(self.ckks_align_tmp_bytes())
            .max(self.ckks_mul_tmp_bytes(ct_infos, tsk_infos))
            .max(self.ckks_square_tmp_bytes(ct_infos, tsk_infos))
            .max(self.ckks_mul_pt_vec_znx_tmp_bytes(ct_infos, ct_infos, pt_prec))
            .max(self.ckks_mul_pt_vec_rnx_tmp_bytes(ct_infos, ct_infos, pt_prec))
            .max(self.ckks_mul_pt_const_tmp_bytes(ct_infos, ct_infos, pt_prec))
            .max(self.prepare_tensor_key_tmp_bytes(tsk_infos))
            .max(self.glwe_tensor_key_encrypt_sk_tmp_bytes(tsk_infos))
    }

    fn ckks_all_ops_with_atk_tmp_bytes<C, T, A>(&self, ct_infos: &C, tsk_infos: &T, atk_infos: &A, pt_prec: &CKKSMeta) -> usize
    where
        C: GLWEInfos,
        T: GGLWEInfos,
        A: GGLWEInfos,
    {
        self.ckks_all_ops_tmp_bytes(ct_infos, tsk_infos, pt_prec)
            .max(self.ckks_rotate_tmp_bytes(ct_infos, atk_infos))
            .max(self.ckks_conjugate_tmp_bytes(ct_infos, atk_infos))
            .max(self.glwe_automorphism_key_encrypt_sk_tmp_bytes(atk_infos))
            .max(self.glwe_automorphism_key_prepare_tmp_bytes(atk_infos))
    }
}
