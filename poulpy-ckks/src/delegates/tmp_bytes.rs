use crate::{
    CKKSCtBounds, CKKSInfos,
    api::{
        CKKSAddOps, CKKSAllOpsTmpBytes, CKKSConjugateOps, CKKSDecryptOps, CKKSEncryptOps, CKKSImagOps, CKKSMulAddOps, CKKSMulOps,
        CKKSMulSubOps, CKKSNegOps, CKKSPow2Ops, CKKSRotateOps, CKKSSubOps,
    },
};
use poulpy_core::{
    GLWEAutomorphism, GLWEAutomorphismKeyEncryptSk, GLWELinearTransformations, GLWEMulConst, GLWEMulPlain, GLWERotate, GLWEShift,
    GLWETensorKeyEncryptSk, GLWETensoring,
    layouts::{GGLWEInfos, GLWEAutomorphismKeyPreparedFactory, GLWETensorKeyPreparedFactory},
};
use poulpy_hal::{
    api::{
        CnvPVecBytesOf, ModuleN, VecZnxLshBackend, VecZnxLshTmpBytes, VecZnxRshAddIntoBackend, VecZnxRshBackend,
        VecZnxRshSubBackend, VecZnxRshTmpBytes,
    },
    layouts::{Backend, Module},
};

impl<BE: Backend> CKKSAllOpsTmpBytes<BE> for Module<BE>
where
    Self: CKKSEncryptOps<BE>
        + CKKSDecryptOps<BE>
        + CKKSAddOps<BE>
        + CKKSConjugateOps<BE>
        + CKKSSubOps<BE>
        + CKKSNegOps<BE>
        + CKKSPow2Ops<BE>
        + CKKSImagOps<BE>
        + CKKSRotateOps<BE>
        + CKKSMulOps<BE>
        + CKKSMulAddOps<BE>
        + CKKSMulSubOps<BE>
        + GLWEAutomorphism<BE>
        + GLWEAutomorphismKeyEncryptSk<BE>
        + GLWELinearTransformations<BE>
        + GLWEAutomorphismKeyPreparedFactory<BE>
        + ModuleN
        + GLWEShift<BE>
        + GLWEMulPlain<BE>
        + GLWEMulConst<BE>
        + GLWERotate<BE>
        + GLWETensoring<BE>
        + GLWETensorKeyEncryptSk<BE>
        + GLWETensorKeyPreparedFactory<BE>
        + CnvPVecBytesOf
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
        // The giant step hoists the prepared `X^{gsp}` right operand into a
        // backend-resident (heap) buffer, so it no longer draws on scratch; the
        // per-pair `ct×ct` multiply and `ct+ct` add bound the scratch.
        let polynomial_giant_steps_tmp_bytes = self
            .ckks_mul_tmp_bytes(ct_infos, ct_infos, ct_infos, tsk_infos)
            .max(self.ckks_add_tmp_bytes());

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
            .max(self.ckks_mul_tmp_bytes(ct_infos, ct_infos, ct_infos, tsk_infos))
            .max(self.ckks_mul_add_ct_tmp_bytes(ct_infos, ct_infos, ct_infos, tsk_infos))
            .max(self.ckks_mul_sub_ct_tmp_bytes(ct_infos, ct_infos, ct_infos, tsk_infos))
            .max(self.ckks_square_tmp_bytes(ct_infos, ct_infos, tsk_infos))
            .max(polynomial_giant_steps_tmp_bytes)
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
            .max(self.glwe_eval_linear_transformation_tmp_bytes(ct_infos, ct_infos, ct_infos, atk_infos))
            .max(self.glwe_eval_linear_transformation_unprepared_rhs_tmp_bytes(ct_infos, ct_infos, ct_infos, atk_infos))
            .max(self.glwe_automorphism_key_encrypt_sk_tmp_bytes(atk_infos))
            .max(self.glwe_automorphism_key_prepare_tmp_bytes(atk_infos))
    }
}
