use crate::{
    layouts::{Backend, NoiseInfos, ScalarZnxBackendMut, ScalarZnxBackendRef, ScratchArena, VecZnxBackendMut, VecZnxBackendRef},
    source::Source,
};

pub trait VecZnxNormalizeTmpBytes {
    /// Returns the minimum number of bytes necessary for normalization.
    fn vec_znx_normalize_tmp_bytes(&self) -> usize;
}

pub trait VecZnxZeroBackend<B: Backend> {
    fn vec_znx_zero_backend(&self, res: &mut VecZnxBackendMut<'_, B>, res_col: usize);
}

pub trait VecZnxCopyRangeBackend<B: Backend> {
    #[allow(clippy::too_many_arguments)]
    fn vec_znx_copy_range_backend(
        &self,
        res: &mut VecZnxBackendMut<'_, B>,
        res_col: usize,
        res_limb: usize,
        res_offset: usize,
        a: &VecZnxBackendRef<'_, B>,
        a_col: usize,
        a_limb: usize,
        a_offset: usize,
        len: usize,
    );
}

pub trait VecZnxExtractCoeffBackend<B: Backend> {
    fn vec_znx_extract_coeff_backend(
        &self,
        res: &mut VecZnxBackendMut<'_, B>,
        res_col: usize,
        a: &VecZnxBackendRef<'_, B>,
        a_col: usize,
        a_coeff: usize,
    );
}

/// Converts a column to centered digits, rounding once at the destination precision.
///
/// For i64 backends, each input limb coefficient must lie in `[-2^62, 2^62]`,
/// and both radix widths must lie in `1..=62`. These bounds leave room for
/// shifted digits and propagated carries. They are caller preconditions;
/// normalization does not scan the input to validate them.
pub trait VecZnxNormalize<B: Backend> {
    #[allow(clippy::too_many_arguments)]
    /// Normalizes the selected column of `a` at `res_k` bits into `res`.
    fn vec_znx_normalize(
        &self,
        res: &mut VecZnxBackendMut<'_, B>,
        res_base2k: usize,
        res_k: usize,
        res_offset: i64,
        res_col: usize,
        a: &VecZnxBackendRef<'_, B>,
        a_base2k: usize,
        a_col: usize,
        scratch: &mut ScratchArena<'_, B>,
    );
}

/// In-place normalization with the input and radix bounds of [`VecZnxNormalize`].
pub trait VecZnxNormalizeAssignBackend<B: Backend> {
    fn vec_znx_normalize_assign_backend(
        &self,
        base2k: usize,
        k: usize,
        a: &mut VecZnxBackendMut<'_, B>,
        a_col: usize,
        scratch: &mut ScratchArena<'_, B>,
    );
}

pub trait VecZnxAddIntoBackend<B: Backend> {
    /// Adds the selected backend-native column of `a` to the selected backend-native column of `b`.
    fn vec_znx_add_into_backend(
        &self,
        res: &mut VecZnxBackendMut<'_, B>,
        res_col: usize,
        a: &VecZnxBackendRef<'_, B>,
        a_col: usize,
        b: &VecZnxBackendRef<'_, B>,
        b_col: usize,
    );
}

pub trait VecZnxAddAssignBackend<B: Backend> {
    fn vec_znx_add_assign_backend(
        &self,
        res: &mut VecZnxBackendMut<'_, B>,
        res_col: usize,
        a: &VecZnxBackendRef<'_, B>,
        a_col: usize,
    );
}

pub trait VecZnxAddScalarAssignBackend<B: Backend> {
    fn vec_znx_add_scalar_assign_backend(
        &self,
        res: &mut VecZnxBackendMut<'_, B>,
        res_col: usize,
        res_limb: usize,
        a: &ScalarZnxBackendRef<'_, B>,
        a_col: usize,
    );
}

pub trait VecZnxSubBackend<B: Backend> {
    fn vec_znx_sub_backend(
        &self,
        res: &mut VecZnxBackendMut<'_, B>,
        res_col: usize,
        a: &VecZnxBackendRef<'_, B>,
        a_col: usize,
        b: &VecZnxBackendRef<'_, B>,
        b_col: usize,
    );
}

pub trait VecZnxSubAssignBackend<B: Backend> {
    fn vec_znx_sub_assign_backend(
        &self,
        res: &mut VecZnxBackendMut<'_, B>,
        res_col: usize,
        a: &VecZnxBackendRef<'_, B>,
        a_col: usize,
    );
}

pub trait VecZnxSubNegateAssignBackend<B: Backend> {
    fn vec_znx_sub_negate_assign_backend(
        &self,
        res: &mut VecZnxBackendMut<'_, B>,
        res_col: usize,
        a: &VecZnxBackendRef<'_, B>,
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
    fn vec_znx_lsh_backend(
        &self,
        base2k: usize,
        k: usize,
        res: &mut VecZnxBackendMut<'_, B>,
        res_col: usize,
        a: &VecZnxBackendRef<'_, B>,
        a_col: usize,
        scratch: &mut ScratchArena<'_, B>,
    );
}

pub trait VecZnxLshAddIntoBackend<B: Backend> {
    /// Adds `a` left-shifted by `k` bits into `res`: `res += a << k`, column-wise.
    ///
    /// Normalization contract: the shifted operand `a` is normalized on the fly
    /// (its own inter-limb carries are propagated), so the addend is in the
    /// canonical `base2k` digit range. The addition into `res` is **not**
    /// re-normalized: adding a normalized digit onto an already-normalized `res`
    /// limb can leave that limb one bit beyond the `base2k` range. Callers that
    /// require a normalized `res` afterwards must normalize it themselves; this
    /// op alone does not restore the digit contract on `res`.
    #[allow(clippy::too_many_arguments)]
    fn vec_znx_lsh_add_into_backend(
        &self,
        base2k: usize,
        k: usize,
        res: &mut VecZnxBackendMut<'_, B>,
        res_col: usize,
        a: &VecZnxBackendRef<'_, B>,
        a_col: usize,
        scratch: &mut ScratchArena<'_, B>,
    );
}

pub trait VecZnxLshAddCoeffToCoeffBackend<B: Backend> {
    #[allow(clippy::too_many_arguments)]
    fn vec_znx_lsh_add_coeff_to_coeff_backend(
        &self,
        base2k: usize,
        k: usize,
        res: &mut VecZnxBackendMut<'_, B>,
        res_col: usize,
        a: &VecZnxBackendRef<'_, B>,
        a_col: usize,
        a_coeff: usize,
        res_coeff: usize,
        scratch: &mut ScratchArena<'_, B>,
    );
}

pub trait VecZnxLshSubCoeffToCoeffBackend<B: Backend> {
    #[allow(clippy::too_many_arguments)]
    fn vec_znx_lsh_sub_coeff_to_coeff_backend(
        &self,
        base2k: usize,
        k: usize,
        res: &mut VecZnxBackendMut<'_, B>,
        res_col: usize,
        a: &VecZnxBackendRef<'_, B>,
        a_col: usize,
        a_coeff: usize,
        res_coeff: usize,
        scratch: &mut ScratchArena<'_, B>,
    );
}

/// Returns scratch bytes required for right-shift operations.
pub trait VecZnxRshTmpBytes {
    fn vec_znx_rsh_tmp_bytes(&self) -> usize;
}

pub trait VecZnxRshBackend<B: Backend> {
    /// Right shift by k bits all columns of `a`.
    #[allow(clippy::too_many_arguments)]
    fn vec_znx_rsh_backend(
        &self,
        base2k: usize,
        k: usize,
        res: &mut VecZnxBackendMut<'_, B>,
        res_col: usize,
        a: &VecZnxBackendRef<'_, B>,
        a_col: usize,
        scratch: &mut ScratchArena<'_, B>,
    );
}

pub trait VecZnxRshCoeffBackend<B: Backend> {
    #[allow(clippy::too_many_arguments)]
    fn vec_znx_rsh_coeff_backend(
        &self,
        base2k: usize,
        k: usize,
        res: &mut VecZnxBackendMut<'_, B>,
        res_col: usize,
        a: &VecZnxBackendRef<'_, B>,
        a_col: usize,
        a_coeff: usize,
        scratch: &mut ScratchArena<'_, B>,
    );
}

pub trait VecZnxRshAddIntoBackend<B: Backend> {
    /// Adds `a` right-shifted by `k` bits into `res`: `res += a >> k`, column-wise.
    ///
    /// Normalization contract: the shifted operand `a` is normalized on the fly
    /// (its own inter-limb carries are propagated), so the addend is in the
    /// canonical `base2k` digit range. The addition into `res` is **not**
    /// re-normalized: adding a normalized digit onto an already-normalized `res`
    /// limb can leave that limb one bit beyond the `base2k` range. Callers that
    /// require a normalized `res` afterwards must normalize it themselves; this
    /// op alone does not restore the digit contract on `res`.
    #[allow(clippy::too_many_arguments)]
    fn vec_znx_rsh_add_into_backend(
        &self,
        base2k: usize,
        k: usize,
        res: &mut VecZnxBackendMut<'_, B>,
        res_col: usize,
        a: &VecZnxBackendRef<'_, B>,
        a_col: usize,
        scratch: &mut ScratchArena<'_, B>,
    );
}

pub trait VecZnxRshAddCoeffIntoBackend<B: Backend> {
    #[allow(clippy::too_many_arguments)]
    fn vec_znx_rsh_add_coeff_into_backend(
        &self,
        base2k: usize,
        k: usize,
        res: &mut VecZnxBackendMut<'_, B>,
        res_col: usize,
        a: &VecZnxBackendRef<'_, B>,
        a_col: usize,
        a_coeff: usize,
        res_coeff: usize,
        scratch: &mut ScratchArena<'_, B>,
    );
}

pub trait VecZnxRshSubCoeffIntoBackend<B: Backend> {
    #[allow(clippy::too_many_arguments)]
    fn vec_znx_rsh_sub_coeff_into_backend(
        &self,
        base2k: usize,
        k: usize,
        res: &mut VecZnxBackendMut<'_, B>,
        res_col: usize,
        a: &VecZnxBackendRef<'_, B>,
        a_col: usize,
        a_coeff: usize,
        res_coeff: usize,
        scratch: &mut ScratchArena<'_, B>,
    );
}

pub trait VecZnxLshSubBackend<B: Backend> {
    /// Left shift by k bits and subtract from destination.
    #[allow(clippy::too_many_arguments)]
    fn vec_znx_lsh_sub_backend(
        &self,
        base2k: usize,
        k: usize,
        res: &mut VecZnxBackendMut<'_, B>,
        res_col: usize,
        a: &VecZnxBackendRef<'_, B>,
        a_col: usize,
        scratch: &mut ScratchArena<'_, B>,
    );
}

pub trait VecZnxRshSubBackend<B: Backend> {
    /// Right shift by k bits and subtract from destination.
    #[allow(clippy::too_many_arguments)]
    fn vec_znx_rsh_sub_backend(
        &self,
        base2k: usize,
        k: usize,
        res: &mut VecZnxBackendMut<'_, B>,
        res_col: usize,
        a: &VecZnxBackendRef<'_, B>,
        a_col: usize,
        scratch: &mut ScratchArena<'_, B>,
    );
}

pub trait VecZnxLshAssignBackend<B: Backend> {
    /// Left shift by k bits all columns of `a`.
    fn vec_znx_lsh_assign_backend(
        &self,
        base2k: usize,
        k: usize,
        a: &mut VecZnxBackendMut<'_, B>,
        a_col: usize,
        scratch: &mut ScratchArena<'_, B>,
    );
}

pub trait VecZnxRshAssignBackend<B: Backend> {
    /// Right shift by k bits all columns of `a`.
    fn vec_znx_rsh_assign_backend(
        &self,
        base2k: usize,
        k: usize,
        a: &mut VecZnxBackendMut<'_, B>,
        a_col: usize,
        scratch: &mut ScratchArena<'_, B>,
    );
}

pub trait VecZnxRotateBackend<B: Backend> {
    /// Multiplies the selected column of `a` by X^k and stores the result in `res_col` of `res`.
    fn vec_znx_rotate_backend(
        &self,
        p: i64,
        res: &mut VecZnxBackendMut<'_, B>,
        res_col: usize,
        a: &VecZnxBackendRef<'_, B>,
        a_col: usize,
    );
}

pub trait VecZnxRotateAssignTmpBytes {
    fn vec_znx_rotate_assign_tmp_bytes(&self) -> usize;
}

pub trait VecZnxRotateAssignBackend<B: Backend> {
    /// Multiplies the selected column of `a` by X^k.
    fn vec_znx_rotate_assign_backend(
        &self,
        p: i64,
        a: &mut VecZnxBackendMut<'_, B>,
        a_col: usize,
        scratch: &mut ScratchArena<'_, B>,
    );
}

pub trait VecZnxAutomorphismBackend<B: Backend> {
    /// Applies the automorphism X^i -> X^ik on the selected column of `a` and stores the result in `res_col` column of `res`.
    fn vec_znx_automorphism_backend(
        &self,
        k: i64,
        res: &mut VecZnxBackendMut<'_, B>,
        res_col: usize,
        a: &VecZnxBackendRef<'_, B>,
        a_col: usize,
    );
}

pub trait VecZnxAutomorphismAssignTmpBytes {
    fn vec_znx_automorphism_assign_tmp_bytes(&self) -> usize;
}

pub trait VecZnxAutomorphismAssignBackend<B: Backend> {
    /// Applies the automorphism X^i -> X^ik on the selected column of `a`.
    fn vec_znx_automorphism_assign_backend(
        &self,
        k: i64,
        res: &mut VecZnxBackendMut<'_, B>,
        res_col: usize,
        scratch: &mut ScratchArena<'_, B>,
    );
}

pub trait ScalarZnxAutomorphismBackend<B: Backend> {
    /// Applies the automorphism X^i -> X^ik on the selected column of `a` and stores the result in `res_col` column of `res`.
    fn scalar_znx_automorphism_backend(
        &self,
        k: i64,
        res: &mut ScalarZnxBackendMut<'_, B>,
        res_col: usize,
        a: &ScalarZnxBackendRef<'_, B>,
        a_col: usize,
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
    fn vec_znx_mul_xp_minus_one_assign_backend(
        &self,
        p: i64,
        res: &mut VecZnxBackendMut<'_, B>,
        res_col: usize,
        scratch: &mut ScratchArena<'_, B>,
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

pub trait ScalarZnxFillTernaryProbSourceBackend<B: Backend> {
    fn scalar_znx_fill_ternary_prob_source_backend(
        &self,
        res: &mut ScalarZnxBackendMut<'_, B>,
        res_col: usize,
        prob: f64,
        source: &mut Source,
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

pub trait ScalarZnxFillBinaryProbSourceBackend<B: Backend> {
    fn scalar_znx_fill_binary_prob_source_backend(
        &self,
        res: &mut ScalarZnxBackendMut<'_, B>,
        res_col: usize,
        prob: f64,
        source: &mut Source,
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

pub trait VecZnxFillUniformSourceBackend<B: Backend> {
    /// Fills a column with a uniform `k`-bit torus value in base `2^base2k`.
    ///
    /// Unused low bits in the last live limb and limbs above `k` are zeroed.
    fn vec_znx_fill_uniform_source_backend(
        &self,
        base2k: usize,
        k: usize,
        res: &mut VecZnxBackendMut<'_, B>,
        res_col: usize,
        source: &mut Source,
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
