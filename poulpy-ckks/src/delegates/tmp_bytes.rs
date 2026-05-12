use crate::{
    CKKSCtBounds, CKKSInfos,
    leveled::api::{
        CKKSAddOps, CKKSAllOpsTmpBytes, CKKSConjugateOps, CKKSDecrypt, CKKSEncrypt, CKKSImagOps, CKKSMulOps, CKKSNegOps,
        CKKSPow2Ops, CKKSRescaleOps, CKKSRotateOps, CKKSSubOps,
    },
};
use poulpy_core::{
    GLWEAutomorphism, GLWEAutomorphismKeyEncryptSk, GLWEMulConst, GLWEMulPlain, GLWERotate, GLWEShift, GLWETensorKeyEncryptSk,
    GLWETensoring,
    layouts::{GGLWEInfos, GLWEAutomorphismKeyPreparedFactory, GLWETensorKeyPreparedFactory},
};
use poulpy_hal::{
    api::{
        ModuleN, VecZnxLshBackend, VecZnxLshTmpBytes, VecZnxRshAddIntoBackend, VecZnxRshBackend, VecZnxRshSubBackend,
        VecZnxRshTmpBytes,
    },
    layouts::{Backend, Module},
};

impl<BE: Backend> CKKSAllOpsTmpBytes<BE> for Module<BE>
where
    Self: CKKSEncrypt<BE>
        + CKKSDecrypt<BE>
        + CKKSAddOps<BE>
        + CKKSConjugateOps<BE>
        + CKKSSubOps<BE>
        + CKKSNegOps<BE>
        + CKKSPow2Ops<BE>
        + CKKSImagOps<BE>
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
        + VecZnxLshBackend<BE>
        + VecZnxLshTmpBytes
        + VecZnxRshBackend<BE>
        + VecZnxRshAddIntoBackend<BE>
        + VecZnxRshSubBackend<BE>
        + VecZnxRshTmpBytes,
{
    fn ckks_all_ops_tmp_bytes<C, T, P>(&self, ct_infos: &C, tsk_infos: &T, pt_prec: &P) -> usize
    where
        C: CKKSCtBounds,
        T: GGLWEInfos,
        P: CKKSInfos,
    {
        self.ckks_encrypt_sk_tmp_bytes(ct_infos)
            .max(self.ckks_decrypt_tmp_bytes(ct_infos))
            .max(self.ckks_add_tmp_bytes())
            .max(self.ckks_add_pt_vec_tmp_bytes())
            .max(self.ckks_add_pt_const_tmp_bytes())
            .max(self.ckks_sub_tmp_bytes())
            .max(self.ckks_sub_pt_vec_tmp_bytes())
            .max(self.ckks_sub_pt_const_tmp_bytes())
            .max(self.ckks_neg_tmp_bytes())
            .max(self.ckks_mul_pow2_tmp_bytes())
            .max(self.ckks_div_pow2_tmp_bytes())
            .max(self.ckks_mul_i_tmp_bytes())
            .max(self.ckks_div_i_tmp_bytes())
            .max(self.ckks_rescale_tmp_bytes())
            .max(self.ckks_align_tmp_bytes())
            .max(self.ckks_mul_tmp_bytes(ct_infos, tsk_infos))
            .max(self.ckks_square_tmp_bytes(ct_infos, tsk_infos))
            .max(self.ckks_mul_pt_vec_tmp_bytes(ct_infos, ct_infos, pt_prec))
            .max(self.ckks_mul_pt_const_tmp_bytes(ct_infos, ct_infos, pt_prec))
            .max(self.prepare_tensor_key_tmp_bytes(tsk_infos))
            .max(self.glwe_tensor_key_encrypt_sk_tmp_bytes(tsk_infos))
    }

    fn ckks_all_ops_with_atk_tmp_bytes<C, T, A, P>(&self, ct_infos: &C, tsk_infos: &T, atk_infos: &A, pt_prec: &P) -> usize
    where
        C: CKKSCtBounds,
        T: GGLWEInfos,
        A: GGLWEInfos,
        P: CKKSInfos,
    {
        self.ckks_all_ops_tmp_bytes(ct_infos, tsk_infos, pt_prec)
            .max(self.ckks_rotate_tmp_bytes(ct_infos, atk_infos))
            .max(self.ckks_conjugate_tmp_bytes(ct_infos, atk_infos))
            .max(self.glwe_automorphism_key_encrypt_sk_tmp_bytes(atk_infos))
            .max(self.glwe_automorphism_key_prepare_tmp_bytes(atk_infos))
    }
}
