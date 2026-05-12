use crate::{
    layouts::{Backend, NoiseInfos, ScalarZnxBackendMut, ScalarZnxBackendRef, ScratchArena, VecZnxBackendMut, VecZnxBackendRef},
    source::Source,
};

pub trait VecZnxNormalizeTmpBytes {
    /// Returns the minimum number of bytes necessary for normalization.
    fn vec_znx_normalize_tmp_bytes(&self) -> usize;
}

pub trait VecZnxZeroBackend<B: Backend> {
    fn vec_znx_zero_backend<'r>(&self, res: &mut VecZnxBackendMut<'r, B>, res_col: usize);
}

pub trait VecZnxCopyRangeBackend<B: Backend> {
    #[allow(clippy::too_many_arguments)]
    fn vec_znx_copy_range_backend<'r, 'a>(
        &self,
        res: &mut VecZnxBackendMut<'r, B>,
        res_col: usize,
        res_limb: usize,
        res_offset: usize,
        a: &VecZnxBackendRef<'a, B>,
        a_col: usize,
        a_limb: usize,
        a_offset: usize,
        len: usize,
    );
}

pub trait VecZnxExtractCoeffBackend<B: Backend> {
    fn vec_znx_extract_coeff_backend<'r, 'a>(
        &self,
        res: &mut VecZnxBackendMut<'r, B>,
        res_col: usize,
        a: &VecZnxBackendRef<'a, B>,
        a_col: usize,
        a_coeff: usize,
    );
}

pub trait VecZnxNormalizeCoeffBackend<B: Backend> {
    #[allow(clippy::too_many_arguments)]
    /// Normalizes the selected coefficient of `a` across its limbs into a 1-coeff destination column.
    fn vec_znx_normalize_coeff_backend<'s, 'r, 'a>(
        &self,
        res: &mut VecZnxBackendMut<'r, B>,
        res_base2k: usize,
        res_offset: i64,
        res_col: usize,
        a: &VecZnxBackendRef<'a, B>,
        a_base2k: usize,
        a_col: usize,
        a_coeff: usize,
        scratch: &mut ScratchArena<'s, B>,
    );
}

pub trait VecZnxSubInnerProductAssignBackend<B: Backend> {
    #[allow(clippy::too_many_arguments)]
    fn vec_znx_sub_inner_product_assign_backend<'r, 'a, 'b>(
        &self,
        res: &mut VecZnxBackendMut<'r, B>,
        res_col: usize,
        res_limb: usize,
        res_offset: usize,
        a: &VecZnxBackendRef<'a, B>,
        a_col: usize,
        a_limb: usize,
        a_offset: usize,
        b: &ScalarZnxBackendRef<'b, B>,
        b_col: usize,
        b_offset: usize,
        len: usize,
    );
}

pub trait VecZnxNormalize<B: Backend> {
    #[allow(clippy::too_many_arguments)]
    /// Normalizes the selected column of `a` and stores the result into the selected column of `res`.
    fn vec_znx_normalize<'s, 'r, 'a>(
        &self,
        res: &mut VecZnxBackendMut<'r, B>,
        res_base2k: usize,
        res_offset: i64,
        res_col: usize,
        a: &VecZnxBackendRef<'a, B>,
        a_base2k: usize,
        a_col: usize,
        scratch: &mut ScratchArena<'s, B>,
    );
}

pub trait VecZnxNormalizeAssignBackend<B: Backend> {
    fn vec_znx_normalize_assign_backend<'s, 'r>(
        &self,
        base2k: usize,
        a: &mut VecZnxBackendMut<'r, B>,
        a_col: usize,
        scratch: &mut ScratchArena<'s, B>,
    );
}

pub trait VecZnxNormalizeCoeffAssignBackend<B: Backend> {
    fn vec_znx_normalize_coeff_assign_backend<'s, 'r>(
        &self,
        base2k: usize,
        a: &mut VecZnxBackendMut<'r, B>,
        a_col: usize,
        a_coeff: usize,
        scratch: &mut ScratchArena<'s, B>,
    );
}

pub trait VecZnxAddIntoBackend<B: Backend> {
    /// Adds the selected backend-native column of `a` to the selected backend-native column of `b`.
    fn vec_znx_add_into_backend<'r, 'a>(
        &self,
        res: &mut VecZnxBackendMut<'r, B>,
        res_col: usize,
        a: &VecZnxBackendRef<'a, B>,
        a_col: usize,
        b: &VecZnxBackendRef<'a, B>,
        b_col: usize,
    );
}

pub trait VecZnxAddAssignBackend<B: Backend> {
    fn vec_znx_add_assign_backend<'r, 'a>(
        &self,
        res: &mut VecZnxBackendMut<'r, B>,
        res_col: usize,
        a: &VecZnxBackendRef<'a, B>,
        a_col: usize,
    );
}

pub trait VecZnxAddConstIntoBackend<B: Backend> {
    #[allow(clippy::too_many_arguments)]
    fn vec_znx_add_const_into_backend<'r, 'a>(
        &self,
        res: &mut VecZnxBackendMut<'r, B>,
        res_col: usize,
        a: &VecZnxBackendRef<'a, B>,
        a_col: usize,
        cnst: &VecZnxBackendRef<'a, B>,
        cnst_col: usize,
        cnst_coeff: usize,
        res_limb: usize,
        res_coeff: usize,
    );
}

pub trait VecZnxAddConstAssignBackend<B: Backend> {
    #[allow(clippy::too_many_arguments)]
    fn vec_znx_add_const_assign_backend<'r, 'a>(
        &self,
        res: &mut VecZnxBackendMut<'r, B>,
        res_col: usize,
        cnst: &VecZnxBackendRef<'a, B>,
        cnst_col: usize,
        cnst_coeff: usize,
        res_limb: usize,
        res_coeff: usize,
    );
}

pub trait VecZnxAddScalarIntoBackend<B: Backend> {
    #[allow(clippy::too_many_arguments)]
    fn vec_znx_add_scalar_into_backend<'r, 'a>(
        &self,
        res: &mut VecZnxBackendMut<'r, B>,
        res_col: usize,
        a: &ScalarZnxBackendRef<'a, B>,
        a_col: usize,
        b: &VecZnxBackendRef<'a, B>,
        b_col: usize,
        b_limb: usize,
    );
}

pub trait VecZnxAddScalarAssignBackend<B: Backend> {
    fn vec_znx_add_scalar_assign_backend<'r, 'a>(
        &self,
        res: &mut VecZnxBackendMut<'r, B>,
        res_col: usize,
        res_limb: usize,
        a: &ScalarZnxBackendRef<'a, B>,
        a_col: usize,
    );
}

pub trait VecZnxSubBackend<B: Backend> {
    fn vec_znx_sub_backend<'r, 'a>(
        &self,
        res: &mut VecZnxBackendMut<'r, B>,
        res_col: usize,
        a: &VecZnxBackendRef<'a, B>,
        a_col: usize,
        b: &VecZnxBackendRef<'a, B>,
        b_col: usize,
    );
}

pub trait VecZnxSubAssignBackend<B: Backend> {
    fn vec_znx_sub_assign_backend<'r, 'a>(
        &self,
        res: &mut VecZnxBackendMut<'r, B>,
        res_col: usize,
        a: &VecZnxBackendRef<'a, B>,
        a_col: usize,
    );
}

pub trait VecZnxSubNegateAssignBackend<B: Backend> {
    fn vec_znx_sub_negate_assign_backend<'r, 'a>(
        &self,
        res: &mut VecZnxBackendMut<'r, B>,
        res_col: usize,
        a: &VecZnxBackendRef<'a, B>,
        a_col: usize,
    );
}

pub trait VecZnxSubScalarBackend<B: Backend> {
    #[allow(clippy::too_many_arguments)]
    fn vec_znx_sub_scalar_backend<'r, 'a>(
        &self,
        res: &mut VecZnxBackendMut<'r, B>,
        res_col: usize,
        a: &ScalarZnxBackendRef<'a, B>,
        a_col: usize,
        b: &VecZnxBackendRef<'a, B>,
        b_col: usize,
        b_limb: usize,
    );
}

pub trait VecZnxSubScalarAssignBackend<B: Backend> {
    fn vec_znx_sub_scalar_assign_backend<'r, 'a>(
        &self,
        res: &mut VecZnxBackendMut<'r, B>,
        res_col: usize,
        res_limb: usize,
        a: &ScalarZnxBackendRef<'a, B>,
        a_col: usize,
    );
}

pub trait VecZnxNegateBackend<B: Backend> {
    fn vec_znx_negate_backend(
        &self,
        res: &mut VecZnxBackendMut<'_, B>,
        res_col: usize,
        a: &VecZnxBackendRef<'_, B>,
        a_col: usize,
    );
}

pub trait VecZnxNegateAssignBackend<B: Backend> {
    fn vec_znx_negate_assign_backend(&self, a: &mut VecZnxBackendMut<'_, B>, a_col: usize);
}

/// Returns scratch bytes required for left-shift operations.
pub trait VecZnxLshTmpBytes {
    fn vec_znx_lsh_tmp_bytes(&self) -> usize;
}

pub trait VecZnxLshBackend<B: Backend> {
    /// Left shift by k bits all columns of `a`.
    #[allow(clippy::too_many_arguments)]
    fn vec_znx_lsh_backend<'s, 'r, 'a>(
        &self,
        base2k: usize,
        k: usize,
        res: &mut VecZnxBackendMut<'r, B>,
        res_col: usize,
        a: &VecZnxBackendRef<'a, B>,
        a_col: usize,
        scratch: &mut ScratchArena<'s, B>,
    );
}

pub trait VecZnxLshCoeffBackend<B: Backend> {
    #[allow(clippy::too_many_arguments)]
    fn vec_znx_lsh_coeff_backend<'s, 'r, 'a>(
        &self,
        base2k: usize,
        k: usize,
        res: &mut VecZnxBackendMut<'r, B>,
        res_col: usize,
        a: &VecZnxBackendRef<'a, B>,
        a_col: usize,
        a_coeff: usize,
        scratch: &mut ScratchArena<'s, B>,
    );
}

pub trait VecZnxLshAddIntoBackend<B: Backend> {
    /// Left shift by k bits all columns of `a`.
    #[allow(clippy::too_many_arguments)]
    fn vec_znx_lsh_add_into_backend<'s, 'r, 'a>(
        &self,
        base2k: usize,
        k: usize,
        res: &mut VecZnxBackendMut<'r, B>,
        res_col: usize,
        a: &VecZnxBackendRef<'a, B>,
        a_col: usize,
        scratch: &mut ScratchArena<'s, B>,
    );
}

pub trait VecZnxLshAddCoeffIntoBackend<B: Backend> {
    #[allow(clippy::too_many_arguments)]
    fn vec_znx_lsh_add_coeff_into_backend<'s, 'r, 'a>(
        &self,
        base2k: usize,
        k: usize,
        res: &mut VecZnxBackendMut<'r, B>,
        res_col: usize,
        a: &VecZnxBackendRef<'a, B>,
        a_col: usize,
        a_coeff: usize,
        scratch: &mut ScratchArena<'s, B>,
    );
}

/// Returns scratch bytes required for right-shift operations.
pub trait VecZnxRshTmpBytes {
    fn vec_znx_rsh_tmp_bytes(&self) -> usize;
}

pub trait VecZnxRshBackend<B: Backend> {
    /// Right shift by k bits all columns of `a`.
    #[allow(clippy::too_many_arguments)]
    fn vec_znx_rsh_backend<'s, 'r, 'a>(
        &self,
        base2k: usize,
        k: usize,
        res: &mut VecZnxBackendMut<'r, B>,
        res_col: usize,
        a: &VecZnxBackendRef<'a, B>,
        a_col: usize,
        scratch: &mut ScratchArena<'s, B>,
    );
}

pub trait VecZnxRshCoeffBackend<B: Backend> {
    #[allow(clippy::too_many_arguments)]
    fn vec_znx_rsh_coeff_backend<'s, 'r, 'a>(
        &self,
        base2k: usize,
        k: usize,
        res: &mut VecZnxBackendMut<'r, B>,
        res_col: usize,
        a: &VecZnxBackendRef<'a, B>,
        a_col: usize,
        a_coeff: usize,
        scratch: &mut ScratchArena<'s, B>,
    );
}

pub trait VecZnxRshAddIntoBackend<B: Backend> {
    /// Right shift by k bits all columns of `a`.
    #[allow(clippy::too_many_arguments)]
    fn vec_znx_rsh_add_into_backend<'s, 'r, 'a>(
        &self,
        base2k: usize,
        k: usize,
        res: &mut VecZnxBackendMut<'r, B>,
        res_col: usize,
        a: &VecZnxBackendRef<'a, B>,
        a_col: usize,
        scratch: &mut ScratchArena<'s, B>,
    );
}

pub trait VecZnxRshAddCoeffIntoBackend<B: Backend> {
    #[allow(clippy::too_many_arguments)]
    fn vec_znx_rsh_add_coeff_into_backend<'s, 'r, 'a>(
        &self,
        base2k: usize,
        k: usize,
        res: &mut VecZnxBackendMut<'r, B>,
        res_col: usize,
        a: &VecZnxBackendRef<'a, B>,
        a_col: usize,
        a_coeff: usize,
        res_coeff: usize,
        scratch: &mut ScratchArena<'s, B>,
    );
}

pub trait VecZnxRshSubCoeffIntoBackend<B: Backend> {
    #[allow(clippy::too_many_arguments)]
    fn vec_znx_rsh_sub_coeff_into_backend<'s, 'r, 'a>(
        &self,
        base2k: usize,
        k: usize,
        res: &mut VecZnxBackendMut<'r, B>,
        res_col: usize,
        a: &VecZnxBackendRef<'a, B>,
        a_col: usize,
        a_coeff: usize,
        res_coeff: usize,
        scratch: &mut ScratchArena<'s, B>,
    );
}

pub trait VecZnxLshSubBackend<B: Backend> {
    /// Left shift by k bits and subtract from destination.
    #[allow(clippy::too_many_arguments)]
    fn vec_znx_lsh_sub_backend<'s, 'r, 'a>(
        &self,
        base2k: usize,
        k: usize,
        res: &mut VecZnxBackendMut<'r, B>,
        res_col: usize,
        a: &VecZnxBackendRef<'a, B>,
        a_col: usize,
        scratch: &mut ScratchArena<'s, B>,
    );
}

pub trait VecZnxRshSubBackend<B: Backend> {
    /// Right shift by k bits and subtract from destination.
    #[allow(clippy::too_many_arguments)]
    fn vec_znx_rsh_sub_backend<'s, 'r, 'a>(
        &self,
        base2k: usize,
        k: usize,
        res: &mut VecZnxBackendMut<'r, B>,
        res_col: usize,
        a: &VecZnxBackendRef<'a, B>,
        a_col: usize,
        scratch: &mut ScratchArena<'s, B>,
    );
}

pub trait VecZnxLshAssignBackend<B: Backend> {
    /// Left shift by k bits all columns of `a`.
    fn vec_znx_lsh_assign_backend<'s, 'r>(
        &self,
        base2k: usize,
        k: usize,
        a: &mut VecZnxBackendMut<'r, B>,
        a_col: usize,
        scratch: &mut ScratchArena<'s, B>,
    );
}

pub trait VecZnxRshAssignBackend<B: Backend> {
    /// Right shift by k bits all columns of `a`.
    fn vec_znx_rsh_assign_backend<'s, 'r>(
        &self,
        base2k: usize,
        k: usize,
        a: &mut VecZnxBackendMut<'r, B>,
        a_col: usize,
        scratch: &mut ScratchArena<'s, B>,
    );
}

pub trait VecZnxRotateBackend<B: Backend> {
    /// Multiplies the selected column of `a` by X^k and stores the result in `res_col` of `res`.
    fn vec_znx_rotate_backend<'r, 'a>(
        &self,
        p: i64,
        res: &mut VecZnxBackendMut<'r, B>,
        res_col: usize,
        a: &VecZnxBackendRef<'a, B>,
        a_col: usize,
    );
}

pub trait VecZnxRotateAssignTmpBytes {
    fn vec_znx_rotate_assign_tmp_bytes(&self) -> usize;
}

pub trait VecZnxRotateAssignBackend<B: Backend> {
    /// Multiplies the selected column of `a` by X^k.
    fn vec_znx_rotate_assign_backend<'s, 'r>(
        &self,
        p: i64,
        a: &mut VecZnxBackendMut<'r, B>,
        a_col: usize,
        scratch: &mut ScratchArena<'s, B>,
    );
}

pub trait VecZnxAutomorphismBackend<B: Backend> {
    /// Applies the automorphism X^i -> X^ik on the selected column of `a` and stores the result in `res_col` column of `res`.
    fn vec_znx_automorphism_backend<'r, 'a>(
        &self,
        k: i64,
        res: &mut VecZnxBackendMut<'r, B>,
        res_col: usize,
        a: &VecZnxBackendRef<'a, B>,
        a_col: usize,
    );
}

pub trait VecZnxAutomorphismAssignTmpBytes {
    fn vec_znx_automorphism_assign_tmp_bytes(&self) -> usize;
}

pub trait VecZnxAutomorphismAssign<B: Backend> {
    /// Applies the automorphism X^i -> X^ik on the selected column of `a`.
    fn vec_znx_automorphism_assign<'s, 'r>(
        &self,
        k: i64,
        res: &mut VecZnxBackendMut<'r, B>,
        res_col: usize,
        scratch: &mut ScratchArena<'s, B>,
    );
}

/// Multiplies the selected column by `(X^p - 1)` in `Z[X]/(X^N + 1)`.
pub trait VecZnxMulXpMinusOneBackend<B: Backend> {
    fn vec_znx_mul_xp_minus_one_backend(
        &self,
        p: i64,
        res: &mut VecZnxBackendMut<'_, B>,
        res_col: usize,
        a: &VecZnxBackendRef<'_, B>,
        a_col: usize,
    );
}

pub trait VecZnxMulXpMinusOneAssignTmpBytes {
    fn vec_znx_mul_xp_minus_one_assign_tmp_bytes(&self) -> usize;
}

pub trait VecZnxMulXpMinusOneAssignBackend<B: Backend> {
    fn vec_znx_mul_xp_minus_one_assign_backend<'s>(
        &self,
        p: i64,
        res: &mut VecZnxBackendMut<'_, B>,
        res_col: usize,
        scratch: &mut ScratchArena<'s, B>,
    );
}

pub trait VecZnxSplitRingTmpBytes {
    fn vec_znx_split_ring_tmp_bytes(&self) -> usize;
}

pub trait VecZnxSplitRingBackend<B: Backend> {
    /// Splits the selected columns of `b` into subrings and copies them them into the selected column of `res`.
    ///
    /// # Panics
    ///
    /// This method requires that all [crate::layouts::VecZnx] of b have the same ring degree
    /// and that b.n() * b.len() <= a.n()
    fn vec_znx_split_ring_backend<'s>(
        &self,
        res: &mut [VecZnxBackendMut<'_, B>],
        res_col: usize,
        a: &VecZnxBackendRef<'_, B>,
        a_col: usize,
        scratch: &mut ScratchArena<'s, B>,
    );
}

pub trait VecZnxMergeRingsTmpBytes {
    fn vec_znx_merge_rings_tmp_bytes(&self) -> usize;
}

pub trait VecZnxMergeRingsBackend<B: Backend> {
    /// Merges the subrings of the selected column of `a` into the selected column of `res`.
    ///
    /// # Panics
    ///
    /// This method requires that all [crate::layouts::VecZnx] of a have the same ring degree
    /// and that a.n() * a.len() <= b.n()
    fn vec_znx_merge_rings_backend<'s>(
        &self,
        res: &mut VecZnxBackendMut<'_, B>,
        res_col: usize,
        a: &[VecZnxBackendRef<'_, B>],
        a_col: usize,
        scratch: &mut ScratchArena<'s, B>,
    );
}

/// Switches ring degree between `a` and `res` by truncation or zero-padding.
pub trait VecZnxSwitchRingBackend<B: Backend> {
    fn vec_znx_switch_ring_backend(
        &self,
        res: &mut VecZnxBackendMut<'_, B>,
        res_col: usize,
        a: &VecZnxBackendRef<'_, B>,
        a_col: usize,
    );
}

pub trait VecZnxCopyBackend<B: Backend> {
    fn vec_znx_copy_backend(&self, res: &mut VecZnxBackendMut<'_, B>, res_col: usize, a: &VecZnxBackendRef<'_, B>, a_col: usize);
}

pub trait ScalarZnxFillTernaryHwSourceBackend<B: Backend> {
    fn scalar_znx_fill_ternary_hw_source_backend(
        &self,
        res: &mut ScalarZnxBackendMut<'_, B>,
        res_col: usize,
        hw: usize,
        source: &mut Source,
    );
}

pub trait ScalarZnxFillTernaryHwBackend<B: Backend> {
    fn scalar_znx_fill_ternary_hw_backend(&self, res: &mut ScalarZnxBackendMut<'_, B>, res_col: usize, hw: usize, seed: [u8; 32]);
}

pub trait ScalarZnxFillTernaryProbSourceBackend<B: Backend> {
    fn scalar_znx_fill_ternary_prob_source_backend(
        &self,
        res: &mut ScalarZnxBackendMut<'_, B>,
        res_col: usize,
        prob: f64,
        source: &mut Source,
    );
}

pub trait ScalarZnxFillTernaryProbBackend<B: Backend> {
    fn scalar_znx_fill_ternary_prob_backend(
        &self,
        res: &mut ScalarZnxBackendMut<'_, B>,
        res_col: usize,
        prob: f64,
        seed: [u8; 32],
    );
}

pub trait ScalarZnxFillBinaryHwSourceBackend<B: Backend> {
    fn scalar_znx_fill_binary_hw_source_backend(
        &self,
        res: &mut ScalarZnxBackendMut<'_, B>,
        res_col: usize,
        hw: usize,
        source: &mut Source,
    );
}

pub trait ScalarZnxFillBinaryHwBackend<B: Backend> {
    fn scalar_znx_fill_binary_hw_backend(&self, res: &mut ScalarZnxBackendMut<'_, B>, res_col: usize, hw: usize, seed: [u8; 32]);
}

pub trait ScalarZnxFillBinaryProbSourceBackend<B: Backend> {
    fn scalar_znx_fill_binary_prob_source_backend(
        &self,
        res: &mut ScalarZnxBackendMut<'_, B>,
        res_col: usize,
        prob: f64,
        source: &mut Source,
    );
}

pub trait ScalarZnxFillBinaryProbBackend<B: Backend> {
    fn scalar_znx_fill_binary_prob_backend(
        &self,
        res: &mut ScalarZnxBackendMut<'_, B>,
        res_col: usize,
        prob: f64,
        seed: [u8; 32],
    );
}

pub trait ScalarZnxFillBinaryBlockSourceBackend<B: Backend> {
    fn scalar_znx_fill_binary_block_source_backend(
        &self,
        res: &mut ScalarZnxBackendMut<'_, B>,
        res_col: usize,
        block_size: usize,
        source: &mut Source,
    );
}

pub trait ScalarZnxFillBinaryBlockBackend<B: Backend> {
    fn scalar_znx_fill_binary_block_backend(
        &self,
        res: &mut ScalarZnxBackendMut<'_, B>,
        res_col: usize,
        block_size: usize,
        seed: [u8; 32],
    );
}

pub trait VecZnxFillUniformSourceBackend<B: Backend> {
    /// Fills the first `size` size with uniform values in \[-2^{base2k-1}, 2^{base2k-1}\]
    fn vec_znx_fill_uniform_source_backend(
        &self,
        base2k: usize,
        res: &mut VecZnxBackendMut<'_, B>,
        res_col: usize,
        source: &mut Source,
    );
}

pub trait VecZnxFillUniformBackend<B: Backend> {
    /// Fills the selected backend-native column from a backend-defined uniform sampler seeded by `seed`.
    fn vec_znx_fill_uniform_backend(&self, base2k: usize, res: &mut VecZnxBackendMut<'_, B>, res_col: usize, seed: [u8; 32]);
}

#[allow(clippy::too_many_arguments)]
/// Fills the selected column with a discrete Gaussian noise vector
/// scaled by `2^{-k}` with standard deviation `sigma`, bounded to `[-bound, bound]`.
pub trait VecZnxFillNormalSourceBackend<B: Backend> {
    fn vec_znx_fill_normal_source_backend(
        &self,
        base2k: usize,
        res: &mut VecZnxBackendMut<'_, B>,
        res_col: usize,
        noise_infos: NoiseInfos,
        source_xe: &mut Source,
    );
}

#[allow(clippy::too_many_arguments)]
pub trait VecZnxFillNormalBackend<B: Backend> {
    /// Fills the selected backend-native column from a backend-defined normal sampler seeded by `seed`.
    fn vec_znx_fill_normal_backend(
        &self,
        base2k: usize,
        res: &mut VecZnxBackendMut<'_, B>,
        res_col: usize,
        noise_infos: NoiseInfos,
        seed: [u8; 32],
    );
}

#[allow(clippy::too_many_arguments)]
pub trait VecZnxAddNormalSourceBackend<B: Backend> {
    /// Adds a discrete normal vector scaled by 2^{-k} with the provided standard deviation and bounded to \[-bound, bound\].
    fn vec_znx_add_normal_source_backend(
        &self,
        base2k: usize,
        res: &mut VecZnxBackendMut<'_, B>,
        res_col: usize,
        noise_infos: NoiseInfos,
        source_xe: &mut Source,
    );
}

#[allow(clippy::too_many_arguments)]
pub trait VecZnxAddNormalBackend<B: Backend> {
    /// Adds backend-defined normal noise to the selected backend-native column using `seed`.
    fn vec_znx_add_normal_backend(
        &self,
        base2k: usize,
        res: &mut VecZnxBackendMut<'_, B>,
        res_col: usize,
        noise_infos: NoiseInfos,
        seed: [u8; 32],
    );
}
