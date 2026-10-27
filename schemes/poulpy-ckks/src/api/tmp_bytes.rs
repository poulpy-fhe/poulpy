use crate::{
    CKKSCtBounds, CKKSInfos,
    api::{
        CKKSAddOps, CKKSCopyOps, CKKSDecryptOps, CKKSEncryptOps, CKKSMulAddOps, CKKSMulOps, CKKSMulSubOps, CKKSNegOps,
        CKKSPow2Ops, CKKSRotateOps, CKKSSubOps,
    },
};
use poulpy_core::{
    GLWEAutomorphism, GLWEAutomorphismKeyEncryptSk, GLWELinearTransformations, GLWEMulConst, GLWEMulPlain, GLWEShift,
    GLWETensorKeyEncryptSk, GLWETensoring,
    api::{GLWEAutomorphismKeyPreparedFactory, GLWETensorKeyPreparedFactory},
    layouts::{GGLWEInfos, LinearTransformationPlan},
};
use poulpy_hal::{
    api::{CnvPVecBytesOf, ModuleN, VecZnxLsh, VecZnxLshTmpBytes, VecZnxRsh, VecZnxRshAdd, VecZnxRshSub, VecZnxRshTmpBytes},
    layouts::{Backend, Module},
};

/// Helpers that return the maximum scratch size needed across broad CKKS
/// operation sets, including the selected copy implementation.
///
/// Pass layouts covering the widest live ciphertext and plaintext operands in
/// the intended pipeline. These are shared bounds for polynomial evaluation.
/// DFT preparation and evaluation require their selected matrix-specific queries,
/// [`crate::api::CKKSDFTOps::ckks_prepare_dft_matrix_tmp_bytes`] and
/// [`crate::api::CKKSDFTOps::ckks_dft_tmp_bytes`], including format wrappers.
/// EvalMod has its own selected query because its working precision depends on the compiled plan.
pub trait CKKSAllOpsTmpBytes<BE: Backend> {
    /// Returns a scratch size large enough for the common CKKS workflow using
    /// ciphertext ops, plaintext ops, encryption/decryption, multiplication,
    /// and tensor-key setup, including one-shot polynomial input copies. The
    /// shift and multiplication terms bound the standard-only multiplication by `i`.
    fn ckks_all_ops_tmp_bytes<C, T, P>(&self, ct_infos: &C, pt_prec: &P, tsk_infos: &T) -> usize
    where
        C: CKKSCtBounds,
        T: GGLWEInfos,
        P: CKKSInfos;

    /// Returns a scratch size large enough for [`Self::ckks_all_ops_tmp_bytes`]
    /// plus automorphism-key setup, rotation, and core prepared or streamed linear transformations
    /// with a key per rotation. Rotation also bounds the standard-only conjugation.
    /// Query DFT preparation and evaluation separately for the selected backend implementation.
    fn ckks_all_ops_with_atk_tmp_bytes<C, T, A, P>(&self, ct_infos: &C, pt_prec: &P, tsk_infos: &T, atk_infos: &A) -> usize
    where
        C: CKKSCtBounds,
        T: GGLWEInfos,
        A: GGLWEInfos,
        P: CKKSInfos;
}

impl<BE: Backend> CKKSAllOpsTmpBytes<BE> for Module<BE>
where
    BE: poulpy_core::oep::GLWEMulConstImpl,
    Self: CKKSEncryptOps<BE>
        + CKKSDecryptOps<BE>
        + CKKSAddOps<BE>
        + CKKSCopyOps<BE>
        + CKKSSubOps<BE>
        + CKKSNegOps<BE>
        + CKKSPow2Ops<BE>
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
        + GLWETensoring<BE>
        + GLWETensorKeyEncryptSk<BE>
        + GLWETensorKeyPreparedFactory<BE>
        + CnvPVecBytesOf
        + VecZnxLsh<BE>
        + VecZnxLshTmpBytes
        + VecZnxRsh<BE>
        + VecZnxRshAdd<BE>
        + VecZnxRshSub<BE>
        + VecZnxRshTmpBytes
        + poulpy_hal::api::Convolution<BE>
        + poulpy_hal::api::VecZnxBigNormalizeTmpBytes
        + poulpy_hal::api::VecZnxIdftApplyTmpBytes
        + poulpy_core::GLWENormalize<BE>
        + crate::api::CKKSRoundOps<BE>,
{
    fn ckks_all_ops_tmp_bytes<C, T, P>(&self, ct_infos: &C, pt_prec: &P, tsk_infos: &T) -> usize
    where
        C: CKKSCtBounds,
        T: GGLWEInfos,
        P: CKKSInfos,
    {
        // The prepared giant-step operand is heap-owned, so the per-pair product and sum bound the giant steps.
        let polynomial_giant_steps_tmp_bytes = self
            .ckks_mul_tmp_bytes(ct_infos, ct_infos, ct_infos, tsk_infos)
            .max(self.ckks_add_tmp_bytes(ct_infos.max_size()));

        self.ckks_encrypt_sk_tmp_bytes(ct_infos)
            .max(self.ckks_decrypt_tmp_bytes(pt_prec, ct_infos))
            .max(self.ckks_copy_tmp_bytes(ct_infos, ct_infos))
            .max(self.ckks_add_tmp_bytes(ct_infos.max_size()))
            .max(self.ckks_add_one_tmp_bytes(ct_infos.max_size()))
            .max(self.ckks_add_pt_vec_tmp_bytes(ct_infos.max_size()))
            .max(self.ckks_add_pt_const_tmp_bytes(ct_infos.max_size()))
            .max(self.ckks_sub_tmp_bytes(ct_infos.max_size()))
            .max(self.ckks_sub_one_tmp_bytes(ct_infos.max_size()))
            .max(self.ckks_sub_pt_vec_tmp_bytes(ct_infos.max_size()))
            .max(self.ckks_sub_pt_const_tmp_bytes(ct_infos.max_size()))
            .max(self.ckks_neg_tmp_bytes(ct_infos.max_size()))
            .max(self.ckks_mul_pow2_tmp_bytes(ct_infos.max_size()))
            .max(self.ckks_div_pow2_tmp_bytes(ct_infos.max_size()))
            .max(self.ckks_mul_tmp_bytes(ct_infos, ct_infos, ct_infos, tsk_infos))
            .max(self.ckks_mul_add_ct_tmp_bytes(ct_infos, ct_infos, ct_infos, tsk_infos))
            .max(self.ckks_mul_sub_ct_tmp_bytes(ct_infos, ct_infos, ct_infos, tsk_infos))
            .max(self.ckks_square_tmp_bytes(ct_infos, ct_infos, tsk_infos))
            .max(polynomial_giant_steps_tmp_bytes)
            .max(
                crate::oep::derived::polynomial_evaluation::eval_baby_linear_combination_tmp_bytes(
                    self,
                    ct_infos.max_size(),
                    pt_prec.k().as_usize().div_ceil(ct_infos.base2k().as_usize()).max(1),
                ),
            )
            .max(if matches!(ct_infos.slot_layout(), crate::SlotLayout::Replicated(_)) {
                crate::reference::polynomial_evaluation::eval_shaped_baby_linear_combination_tmp_bytes(
                    self,
                    ct_infos,
                    pt_prec.k().as_usize().div_ceil(ct_infos.base2k().as_usize()).max(1),
                )
            } else {
                0
            })
            .max(self.ckks_mul_pt_vec_tmp_bytes(ct_infos, ct_infos, pt_prec))
            .max(self.ckks_mul_pt_const_tmp_bytes(ct_infos, ct_infos, pt_prec))
            .max(self.glwe_tensor_key_prepare_tmp_bytes(tsk_infos))
            .max(self.glwe_tensor_key_encrypt_sk_tmp_bytes(tsk_infos))
    }

    fn ckks_all_ops_with_atk_tmp_bytes<C, T, A, P>(&self, ct_infos: &C, pt_prec: &P, tsk_infos: &T, atk_infos: &A) -> usize
    where
        C: CKKSCtBounds,
        T: GGLWEInfos,
        A: GGLWEInfos,
        P: CKKSInfos,
    {
        // The core linear transformations with a key per rotation, which every plan of that kind sizes alike.
        let plan = LinearTransformationPlan::default();
        self.ckks_all_ops_tmp_bytes(ct_infos, pt_prec, tsk_infos)
            .max(self.ckks_rotate_tmp_bytes(ct_infos, atk_infos))
            .max(self.glwe_eval_linear_transformation_tmp_bytes(ct_infos, ct_infos, ct_infos, &plan, atk_infos))
            .max(self.glwe_eval_linear_transformation_unprepared_rhs_tmp_bytes(ct_infos, ct_infos, ct_infos, &plan, atk_infos))
            .max(self.glwe_automorphism_key_encrypt_sk_tmp_bytes(atk_infos))
            .max(self.glwe_automorphism_key_prepare_tmp_bytes(atk_infos))
    }
}
