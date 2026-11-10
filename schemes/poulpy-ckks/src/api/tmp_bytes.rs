use crate::{
    CKKSCtBounds, CKKSInfos,
    api::{
        CKKSAddOps, CKKSCopyOps, CKKSDecryptOps, CKKSEncryptOps, CKKSMulAddOps, CKKSMulOps, CKKSMulSubOps, CKKSNegOps,
        CKKSPow2Ops, CKKSRotateOps, CKKSSubOps,
    },
};
use poulpy_core::{
    GLWEAutomorphism, GLWEAutomorphismKeyEncryptSk, GLWELinearTransformationOps, GLWEMulConst, GLWEMulPlain, GLWENormalize,
    GLWEShift, GLWETensorKeyEncryptSk, GLWETensoring,
    api::{GLWEAutomorphismKeyPreparedFactory, GLWETensorKeyPreparedFactory},
    layouts::{GGLWEInfos, LinearTransformationPlan},
    oep::GLWEMulConstImpl,
};
use poulpy_hal::{
    api::{
        CnvOps, CnvPVecBytesOf, ModuleN, VecZnxBigNormalizeTmpBytes, VecZnxLsh, VecZnxLshTmpBytes, VecZnxRsh, VecZnxRshAdd,
        VecZnxRshSub, VecZnxRshTmpBytes,
    },
    layouts::{Backend, ConjugateInvariant, Module, Ring, Standard},
};

/// Helpers that return the maximum scratch size needed across broad CKKS
/// operation sets, including the selected copy implementation.
///
/// Pass layouts covering the widest live ciphertext and plaintext operands in
/// the intended pipeline. These are shared bounds for polynomial evaluation.
/// DFT preparation and evaluation require their selected matrix-specific queries,
/// [`crate::api::CKKSDFTMatrixPreparedFactory::ckks_dft_matrix_prepare_tmp_bytes`] and
/// [`crate::api::CKKSDFTOps::ckks_dft_tmp_bytes`], including format wrappers.
/// EvalMod has its own selected query because its working precision depends on the compiled plan.
/// Each bound is the larger of its constituent queries on the two rings, so it serves operands of either.
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
    /// plus automorphism-key setup, rotation, and core prepared or streamed linear transformations of plaintext
    /// diagonals with a key per rotation. Rotation also bounds the standard-only conjugation.
    /// Query DFT preparation and evaluation separately for the selected backend implementation.
    fn ckks_all_ops_with_atk_tmp_bytes<C, T, A, P>(&self, ct_infos: &C, pt_prec: &P, tsk_infos: &T, atk_infos: &A) -> usize
    where
        C: CKKSCtBounds,
        T: GGLWEInfos,
        A: GGLWEInfos,
        P: CKKSInfos;
}

/// The queries [`CKKSAllOpsTmpBytes`] takes on ring `Rg`, blanket-implemented for every module providing them.
pub trait CKKSAllOpsTmpBytesAt<BE: Backend, Rg: Ring>:
    CKKSEncryptOps<BE, Rg>
    + CKKSDecryptOps<BE, Rg>
    + CKKSAddOps<BE, Rg>
    + CKKSCopyOps<BE, Rg>
    + CKKSSubOps<BE, Rg>
    + CKKSNegOps<BE, Rg>
    + CKKSPow2Ops<BE, Rg>
    + CKKSRotateOps<BE, Rg>
    + CKKSMulOps<BE, Rg>
    + CKKSMulAddOps<BE, Rg>
    + CKKSMulSubOps<BE, Rg>
    + GLWEAutomorphism<BE, Rg>
    + GLWENormalize<BE, Rg>
    + GLWEAutomorphismKeyEncryptSk<BE, Rg>
    + GLWELinearTransformationOps<BE, Rg>
    + GLWEAutomorphismKeyPreparedFactory<BE, Rg>
    + ModuleN
    + GLWEShift<BE, Rg>
    + GLWEMulPlain<BE, Rg>
    + GLWEMulConst<BE, Rg>
    + GLWETensoring<BE, Rg>
    + GLWETensorKeyEncryptSk<BE, Rg>
    + GLWETensorKeyPreparedFactory<BE, Rg>
    + CnvPVecBytesOf
    + VecZnxLsh<BE, Rg>
    + VecZnxLshTmpBytes
    + VecZnxRsh<BE, Rg>
    + VecZnxRshAdd<BE, Rg>
    + VecZnxRshSub<BE, Rg>
    + VecZnxRshTmpBytes
    + CnvOps<BE, Rg>
    + VecZnxBigNormalizeTmpBytes
{
}

impl<BE: Backend, Rg: Ring, M> CKKSAllOpsTmpBytesAt<BE, Rg> for M where
    M: CKKSEncryptOps<BE, Rg>
        + CKKSDecryptOps<BE, Rg>
        + CKKSAddOps<BE, Rg>
        + CKKSCopyOps<BE, Rg>
        + CKKSSubOps<BE, Rg>
        + CKKSNegOps<BE, Rg>
        + CKKSPow2Ops<BE, Rg>
        + CKKSRotateOps<BE, Rg>
        + CKKSMulOps<BE, Rg>
        + CKKSMulAddOps<BE, Rg>
        + CKKSMulSubOps<BE, Rg>
        + GLWEAutomorphism<BE, Rg>
        + GLWENormalize<BE, Rg>
        + GLWEAutomorphismKeyEncryptSk<BE, Rg>
        + GLWELinearTransformationOps<BE, Rg>
        + GLWEAutomorphismKeyPreparedFactory<BE, Rg>
        + ModuleN
        + GLWEShift<BE, Rg>
        + GLWEMulPlain<BE, Rg>
        + GLWEMulConst<BE, Rg>
        + GLWETensoring<BE, Rg>
        + GLWETensorKeyEncryptSk<BE, Rg>
        + GLWETensorKeyPreparedFactory<BE, Rg>
        + CnvPVecBytesOf
        + VecZnxLsh<BE, Rg>
        + VecZnxLshTmpBytes
        + VecZnxRsh<BE, Rg>
        + VecZnxRshAdd<BE, Rg>
        + VecZnxRshSub<BE, Rg>
        + VecZnxRshTmpBytes
        + CnvOps<BE, Rg>
        + VecZnxBigNormalizeTmpBytes
{
}

fn all_ops_tmp_bytes_at<BE, C, T, P, Rg: Ring>(module: &Module<BE>, ct_infos: &C, pt_prec: &P, tsk_infos: &T) -> usize
where
    BE: GLWEMulConstImpl<Rg>,
    Module<BE>: CKKSAllOpsTmpBytesAt<BE, Rg>,
    C: CKKSCtBounds,
    T: GGLWEInfos,
    P: CKKSInfos,
{
    // The prepared giant-step operand is heap-owned, so the per-pair product and sum bound the giant steps.
    let polynomial_giant_steps_tmp_bytes = module
        .ckks_mul_tmp_bytes(ct_infos, ct_infos, ct_infos, tsk_infos)
        .max(module.ckks_add_tmp_bytes(ct_infos.max_size()));

    module
        .ckks_encrypt_sk_tmp_bytes(ct_infos)
        .max(module.ckks_decrypt_tmp_bytes(pt_prec, ct_infos))
        .max(module.ckks_copy_tmp_bytes(ct_infos, ct_infos))
        .max(module.ckks_add_tmp_bytes(ct_infos.max_size()))
        .max(module.ckks_add_one_tmp_bytes(ct_infos.max_size()))
        .max(module.ckks_add_pt_vec_tmp_bytes(ct_infos.max_size()))
        .max(module.ckks_add_pt_const_tmp_bytes(ct_infos.max_size()))
        .max(module.ckks_sub_tmp_bytes(ct_infos.max_size()))
        .max(module.ckks_sub_one_tmp_bytes(ct_infos.max_size()))
        .max(module.ckks_sub_pt_vec_tmp_bytes(ct_infos.max_size()))
        .max(module.ckks_sub_pt_const_tmp_bytes(ct_infos.max_size()))
        .max(module.ckks_neg_tmp_bytes(ct_infos.max_size()))
        .max(module.ckks_mul_pow2_tmp_bytes(ct_infos.max_size()))
        .max(module.ckks_double_tmp_bytes(ct_infos.max_size()))
        .max(module.ckks_div_pow2_tmp_bytes(ct_infos.max_size()))
        .max(module.ckks_mul_tmp_bytes(ct_infos, ct_infos, ct_infos, tsk_infos))
        .max(module.ckks_mul_add_ct_tmp_bytes(ct_infos, ct_infos, ct_infos, tsk_infos))
        .max(module.ckks_mul_sub_ct_tmp_bytes(ct_infos, ct_infos, ct_infos, tsk_infos))
        .max(module.ckks_square_tmp_bytes(ct_infos, ct_infos, tsk_infos))
        .max(polynomial_giant_steps_tmp_bytes)
        .max(crate::power_basis::ckks_power_product_tmp_bytes(module, ct_infos, tsk_infos))
        .max(
            crate::oep::derived::polynomial_evaluation::eval_baby_linear_combination_tmp_bytes::<BE, Rg>(
                module,
                ct_infos.max_size(),
                pt_prec.k().as_usize().div_ceil(ct_infos.base2k().as_usize()).max(1),
            ),
        )
        .max(module.ckks_mul_pt_vec_tmp_bytes(ct_infos, ct_infos, pt_prec))
        .max(module.ckks_mul_pt_const_tmp_bytes(ct_infos, ct_infos, pt_prec))
        .max(module.glwe_tensor_key_prepare_tmp_bytes(tsk_infos))
        .max(module.glwe_tensor_key_encrypt_sk_tmp_bytes(tsk_infos))
}

fn all_ops_with_atk_tmp_bytes_at<BE, C, T, A, Rg: Ring>(module: &Module<BE>, ct_infos: &C, tsk_infos: &T, atk_infos: &A) -> usize
where
    Module<BE>: CKKSAllOpsTmpBytesAt<BE, Rg>,
    BE: Backend,
    C: CKKSCtBounds,
    T: GGLWEInfos,
    A: GGLWEInfos,
{
    // The core linear transformations with a key per rotation, which every plan of that kind sizes alike.
    let plan = LinearTransformationPlan::default();
    let diagonal = crate::api::ckks_plaintext_diagonal_layout(ct_infos);
    module
        .ckks_rotate_tmp_bytes(ct_infos, atk_infos)
        .max(module.glwe_linear_transformation_eval_tmp_bytes(ct_infos, ct_infos, &diagonal, &plan, tsk_infos, atk_infos))
        .max(
            module.glwe_linear_transformation_eval_unprepared_rhs_tmp_bytes(
                ct_infos, ct_infos, &diagonal, &plan, tsk_infos, atk_infos,
            ),
        )
        .max(module.glwe_automorphism_key_encrypt_sk_tmp_bytes(atk_infos))
        .max(module.glwe_automorphism_key_prepare_tmp_bytes(atk_infos))
}

impl<BE: Backend> CKKSAllOpsTmpBytes<BE> for Module<BE>
where
    BE: GLWEMulConstImpl<Standard> + GLWEMulConstImpl<ConjugateInvariant>,
    Self: CKKSAllOpsTmpBytesAt<BE, Standard> + CKKSAllOpsTmpBytesAt<BE, ConjugateInvariant>,
{
    fn ckks_all_ops_tmp_bytes<C, T, P>(&self, ct_infos: &C, pt_prec: &P, tsk_infos: &T) -> usize
    where
        C: CKKSCtBounds,
        T: GGLWEInfos,
        P: CKKSInfos,
    {
        all_ops_tmp_bytes_at::<BE, C, T, P, Standard>(self, ct_infos, pt_prec, tsk_infos).max(all_ops_tmp_bytes_at::<
            BE,
            C,
            T,
            P,
            ConjugateInvariant,
        >(
            self, ct_infos, pt_prec, tsk_infos,
        ))
    }

    fn ckks_all_ops_with_atk_tmp_bytes<C, T, A, P>(&self, ct_infos: &C, pt_prec: &P, tsk_infos: &T, atk_infos: &A) -> usize
    where
        C: CKKSCtBounds,
        T: GGLWEInfos,
        A: GGLWEInfos,
        P: CKKSInfos,
    {
        self.ckks_all_ops_tmp_bytes(ct_infos, pt_prec, tsk_infos)
            .max(all_ops_with_atk_tmp_bytes_at::<BE, C, T, A, Standard>(
                self, ct_infos, tsk_infos, atk_infos,
            ))
            .max(all_ops_with_atk_tmp_bytes_at::<BE, C, T, A, ConjugateInvariant>(
                self, ct_infos, tsk_infos, atk_infos,
            ))
    }
}
