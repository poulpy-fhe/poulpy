#![allow(clippy::too_many_arguments)]

use crate::{
    layouts::{
        Backend, Module, NoiseInfos, ScalarZnxBackendMut, ScalarZnxBackendRef, ScratchArena, VecZnxBackendMut, VecZnxBackendRef,
    },
    source::Source,
};

/// Module construction extension point.
///
/// # Safety
/// Implementations must return a module handle that is valid for the backend
/// and ring degree, and uphold the backend safety contract.
pub unsafe trait HalModuleImpl<BE: Backend>: Backend {
    #[allow(clippy::new_ret_no_self)]
    fn new(n: u64) -> Module<BE>;
}

/// Coefficient-domain `VecZnx` extension point.
///
/// # Safety
/// Implementations must uphold the backend safety contract for layout access,
/// aliasing, scratch usage, and arithmetic correctness.
pub unsafe trait HalVecZnxImpl<BE: Backend>: Backend {
    fn vec_znx_zero_backend<'r>(module: &Module<BE>, res: &mut VecZnxBackendMut<'r, BE>, res_col: usize);

    fn scalar_znx_fill_ternary_hw_backend(
        module: &Module<BE>,
        res: &mut ScalarZnxBackendMut<'_, BE>,
        res_col: usize,
        hw: usize,
        seed: [u8; 32],
    );

    fn scalar_znx_fill_ternary_prob_backend(
        module: &Module<BE>,
        res: &mut ScalarZnxBackendMut<'_, BE>,
        res_col: usize,
        prob: f64,
        seed: [u8; 32],
    );

    fn scalar_znx_fill_binary_hw_backend(
        module: &Module<BE>,
        res: &mut ScalarZnxBackendMut<'_, BE>,
        res_col: usize,
        hw: usize,
        seed: [u8; 32],
    );

    fn scalar_znx_fill_binary_prob_backend(
        module: &Module<BE>,
        res: &mut ScalarZnxBackendMut<'_, BE>,
        res_col: usize,
        prob: f64,
        seed: [u8; 32],
    );

    fn scalar_znx_fill_binary_block_backend(
        module: &Module<BE>,
        res: &mut ScalarZnxBackendMut<'_, BE>,
        res_col: usize,
        block_size: usize,
        seed: [u8; 32],
    );

    #[allow(clippy::too_many_arguments)]
    fn vec_znx_sub_inner_product_assign_backend<'r, 'a, 'b>(
        module: &Module<BE>,
        res: &mut VecZnxBackendMut<'r, BE>,
        res_col: usize,
        res_limb: usize,
        res_offset: usize,
        a: &VecZnxBackendRef<'a, BE>,
        a_col: usize,
        a_limb: usize,
        a_offset: usize,
        b: &ScalarZnxBackendRef<'b, BE>,
        b_col: usize,
        b_offset: usize,
        len: usize,
    );

    fn vec_znx_normalize_tmp_bytes_backend(module: &Module<BE>) -> usize;

    #[allow(clippy::too_many_arguments)]
    fn vec_znx_normalize_backend<'s, 'r, 'a>(
        module: &Module<BE>,
        res: &mut VecZnxBackendMut<'r, BE>,
        res_base2k: usize,
        res_offset: i64,
        res_col: usize,
        a: &VecZnxBackendRef<'a, BE>,
        a_base2k: usize,
        a_col: usize,
        scratch: &mut ScratchArena<'s, BE>,
    );

    fn vec_znx_normalize_assign_backend<'s, 'r>(
        module: &Module<BE>,
        base2k: usize,
        a: &mut VecZnxBackendMut<'r, BE>,
        a_col: usize,
        scratch: &mut ScratchArena<'s, BE>,
    );

    fn vec_znx_normalize_coeff_assign_backend<'s, 'r>(
        module: &Module<BE>,
        base2k: usize,
        a: &mut VecZnxBackendMut<'r, BE>,
        a_col: usize,
        a_coeff: usize,
        scratch: &mut ScratchArena<'s, BE>,
    );

    #[allow(clippy::too_many_arguments)]
    fn vec_znx_normalize_coeff_backend<'s, 'r, 'a>(
        module: &Module<BE>,
        res: &mut VecZnxBackendMut<'r, BE>,
        res_base2k: usize,
        res_offset: i64,
        res_col: usize,
        a: &VecZnxBackendRef<'a, BE>,
        a_base2k: usize,
        a_col: usize,
        a_coeff: usize,
        scratch: &mut ScratchArena<'s, BE>,
    );

    fn vec_znx_add_into_backend<'r, 'a>(
        module: &Module<BE>,
        res: &mut VecZnxBackendMut<'r, BE>,
        res_col: usize,
        a: &VecZnxBackendRef<'a, BE>,
        a_col: usize,
        b: &VecZnxBackendRef<'a, BE>,
        b_col: usize,
    );

    fn vec_znx_add_assign_backend<'r, 'a>(
        module: &Module<BE>,
        res: &mut VecZnxBackendMut<'r, BE>,
        res_col: usize,
        a: &VecZnxBackendRef<'a, BE>,
        a_col: usize,
    );

    #[allow(clippy::too_many_arguments)]
    fn vec_znx_add_const_into_backend<'r, 'a>(
        module: &Module<BE>,
        res: &mut VecZnxBackendMut<'r, BE>,
        res_col: usize,
        a: &VecZnxBackendRef<'a, BE>,
        a_col: usize,
        cnst: &VecZnxBackendRef<'a, BE>,
        cnst_col: usize,
        cnst_coeff: usize,
        res_limb: usize,
        res_coeff: usize,
    );

    fn vec_znx_add_const_assign_backend<'r, 'a>(
        module: &Module<BE>,
        res: &mut VecZnxBackendMut<'r, BE>,
        res_col: usize,
        cnst: &VecZnxBackendRef<'a, BE>,
        cnst_col: usize,
        cnst_coeff: usize,
        res_limb: usize,
        res_coeff: usize,
    );

    #[allow(clippy::too_many_arguments)]
    fn vec_znx_add_scalar_into_backend<'r, 'a>(
        module: &Module<BE>,
        res: &mut VecZnxBackendMut<'r, BE>,
        res_col: usize,
        a: &ScalarZnxBackendRef<'a, BE>,
        a_col: usize,
        b: &VecZnxBackendRef<'a, BE>,
        b_col: usize,
        b_limb: usize,
    );

    fn vec_znx_add_scalar_assign_backend<'r, 'a>(
        module: &Module<BE>,
        res: &mut VecZnxBackendMut<'r, BE>,
        res_col: usize,
        res_limb: usize,
        a: &ScalarZnxBackendRef<'a, BE>,
        a_col: usize,
    );

    fn vec_znx_sub_backend<'r, 'a>(
        module: &Module<BE>,
        res: &mut VecZnxBackendMut<'r, BE>,
        res_col: usize,
        a: &VecZnxBackendRef<'a, BE>,
        a_col: usize,
        b: &VecZnxBackendRef<'a, BE>,
        b_col: usize,
    );

    fn vec_znx_sub_assign_backend<'r, 'a>(
        module: &Module<BE>,
        res: &mut VecZnxBackendMut<'r, BE>,
        res_col: usize,
        a: &VecZnxBackendRef<'a, BE>,
        a_col: usize,
    );

    fn vec_znx_sub_negate_assign_backend<'r, 'a>(
        module: &Module<BE>,
        res: &mut VecZnxBackendMut<'r, BE>,
        res_col: usize,
        a: &VecZnxBackendRef<'a, BE>,
        a_col: usize,
    );

    #[allow(clippy::too_many_arguments)]
    fn vec_znx_sub_scalar_backend<'r, 'a>(
        module: &Module<BE>,
        res: &mut VecZnxBackendMut<'r, BE>,
        res_col: usize,
        a: &ScalarZnxBackendRef<'a, BE>,
        a_col: usize,
        b: &VecZnxBackendRef<'a, BE>,
        b_col: usize,
        b_limb: usize,
    );

    fn vec_znx_sub_scalar_assign_backend<'r, 'a>(
        module: &Module<BE>,
        res: &mut VecZnxBackendMut<'r, BE>,
        res_col: usize,
        res_limb: usize,
        a: &ScalarZnxBackendRef<'a, BE>,
        a_col: usize,
    );

    fn vec_znx_negate_backend(
        module: &Module<BE>,
        res: &mut VecZnxBackendMut<'_, BE>,
        res_col: usize,
        a: &VecZnxBackendRef<'_, BE>,
        a_col: usize,
    );

    fn vec_znx_negate_assign_backend(module: &Module<BE>, a: &mut VecZnxBackendMut<'_, BE>, a_col: usize);

    fn vec_znx_rsh_tmp_bytes_backend(module: &Module<BE>) -> usize;

    fn vec_znx_rsh_backend<'s, 'r, 'a>(
        module: &Module<BE>,
        base2k: usize,
        k: usize,
        res: &mut VecZnxBackendMut<'r, BE>,
        res_col: usize,
        a: &VecZnxBackendRef<'a, BE>,
        a_col: usize,
        scratch: &mut ScratchArena<'s, BE>,
    );

    fn vec_znx_rsh_coeff_backend<'s, 'r, 'a>(
        module: &Module<BE>,
        base2k: usize,
        k: usize,
        res: &mut VecZnxBackendMut<'r, BE>,
        res_col: usize,
        a: &VecZnxBackendRef<'a, BE>,
        a_col: usize,
        a_coeff: usize,
        scratch: &mut ScratchArena<'s, BE>,
    );

    fn vec_znx_rsh_add_into_backend<'s, 'r, 'a>(
        module: &Module<BE>,
        base2k: usize,
        k: usize,
        res: &mut VecZnxBackendMut<'r, BE>,
        res_col: usize,
        a: &VecZnxBackendRef<'a, BE>,
        a_col: usize,
        scratch: &mut ScratchArena<'s, BE>,
    );

    fn vec_znx_rsh_add_coeff_into_backend<'s, 'r, 'a>(
        module: &Module<BE>,
        base2k: usize,
        k: usize,
        res: &mut VecZnxBackendMut<'r, BE>,
        res_col: usize,
        a: &VecZnxBackendRef<'a, BE>,
        a_col: usize,
        a_coeff: usize,
        res_coeff: usize,
        scratch: &mut ScratchArena<'s, BE>,
    );

    fn vec_znx_rsh_sub_coeff_into_backend<'s, 'r, 'a>(
        module: &Module<BE>,
        base2k: usize,
        k: usize,
        res: &mut VecZnxBackendMut<'r, BE>,
        res_col: usize,
        a: &VecZnxBackendRef<'a, BE>,
        a_col: usize,
        a_coeff: usize,
        res_coeff: usize,
        scratch: &mut ScratchArena<'s, BE>,
    );

    fn vec_znx_lsh_tmp_bytes_backend(module: &Module<BE>) -> usize;

    fn vec_znx_lsh_backend<'s, 'r, 'a>(
        module: &Module<BE>,
        base2k: usize,
        k: usize,
        res: &mut VecZnxBackendMut<'r, BE>,
        res_col: usize,
        a: &VecZnxBackendRef<'a, BE>,
        a_col: usize,
        scratch: &mut ScratchArena<'s, BE>,
    );

    fn vec_znx_lsh_coeff_backend<'s, 'r, 'a>(
        module: &Module<BE>,
        base2k: usize,
        k: usize,
        res: &mut VecZnxBackendMut<'r, BE>,
        res_col: usize,
        a: &VecZnxBackendRef<'a, BE>,
        a_col: usize,
        a_coeff: usize,
        scratch: &mut ScratchArena<'s, BE>,
    );

    fn vec_znx_lsh_add_into_backend<'s, 'r, 'a>(
        module: &Module<BE>,
        base2k: usize,
        k: usize,
        res: &mut VecZnxBackendMut<'r, BE>,
        res_col: usize,
        a: &VecZnxBackendRef<'a, BE>,
        a_col: usize,
        scratch: &mut ScratchArena<'s, BE>,
    );

    fn vec_znx_lsh_add_coeff_into_backend<'s, 'r, 'a>(
        module: &Module<BE>,
        base2k: usize,
        k: usize,
        res: &mut VecZnxBackendMut<'r, BE>,
        res_col: usize,
        a: &VecZnxBackendRef<'a, BE>,
        a_col: usize,
        a_coeff: usize,
        scratch: &mut ScratchArena<'s, BE>,
    );

    fn vec_znx_lsh_sub_backend<'s, 'r, 'a>(
        module: &Module<BE>,
        base2k: usize,
        k: usize,
        res: &mut VecZnxBackendMut<'r, BE>,
        res_col: usize,
        a: &VecZnxBackendRef<'a, BE>,
        a_col: usize,
        scratch: &mut ScratchArena<'s, BE>,
    );

    fn vec_znx_rsh_sub_backend<'s, 'r, 'a>(
        module: &Module<BE>,
        base2k: usize,
        k: usize,
        res: &mut VecZnxBackendMut<'r, BE>,
        res_col: usize,
        a: &VecZnxBackendRef<'a, BE>,
        a_col: usize,
        scratch: &mut ScratchArena<'s, BE>,
    );

    fn vec_znx_rsh_assign_backend<'s, 'r>(
        module: &Module<BE>,
        base2k: usize,
        k: usize,
        a: &mut VecZnxBackendMut<'r, BE>,
        a_col: usize,
        scratch: &mut ScratchArena<'s, BE>,
    );

    fn vec_znx_lsh_assign_backend<'s, 'r>(
        module: &Module<BE>,
        base2k: usize,
        k: usize,
        a: &mut VecZnxBackendMut<'r, BE>,
        a_col: usize,
        scratch: &mut ScratchArena<'s, BE>,
    );

    fn vec_znx_rotate_backend<'r, 'a>(
        module: &Module<BE>,
        k: i64,
        res: &mut VecZnxBackendMut<'r, BE>,
        res_col: usize,
        a: &VecZnxBackendRef<'a, BE>,
        a_col: usize,
    );

    fn vec_znx_rotate_assign_tmp_bytes_backend(module: &Module<BE>) -> usize;

    fn vec_znx_rotate_assign_backend<'s, 'r>(
        module: &Module<BE>,
        k: i64,
        a: &mut VecZnxBackendMut<'r, BE>,
        a_col: usize,
        scratch: &mut ScratchArena<'s, BE>,
    );

    fn vec_znx_automorphism_backend<'r, 'a>(
        module: &Module<BE>,
        k: i64,
        res: &mut VecZnxBackendMut<'r, BE>,
        res_col: usize,
        a: &VecZnxBackendRef<'a, BE>,
        a_col: usize,
    );

    fn vec_znx_automorphism_assign_tmp_bytes_backend(module: &Module<BE>) -> usize;

    fn vec_znx_automorphism_assign_backend<'s, 'r>(
        module: &Module<BE>,
        k: i64,
        res: &mut VecZnxBackendMut<'r, BE>,
        res_col: usize,
        scratch: &mut ScratchArena<'s, BE>,
    );

    fn vec_znx_mul_xp_minus_one_backend(
        module: &Module<BE>,
        k: i64,
        res: &mut VecZnxBackendMut<'_, BE>,
        res_col: usize,
        a: &VecZnxBackendRef<'_, BE>,
        a_col: usize,
    );

    fn vec_znx_mul_xp_minus_one_assign_tmp_bytes_backend(module: &Module<BE>) -> usize;

    fn vec_znx_mul_xp_minus_one_assign_backend<'s>(
        module: &Module<BE>,
        k: i64,
        res: &mut VecZnxBackendMut<'_, BE>,
        res_col: usize,
        scratch: &mut ScratchArena<'s, BE>,
    );

    fn vec_znx_split_ring_tmp_bytes_backend(module: &Module<BE>) -> usize;

    fn vec_znx_split_ring_backend<'s>(
        module: &Module<BE>,
        res: &mut [VecZnxBackendMut<'_, BE>],
        res_col: usize,
        a: &VecZnxBackendRef<'_, BE>,
        a_col: usize,
        scratch: &mut ScratchArena<'s, BE>,
    );

    fn vec_znx_merge_rings_tmp_bytes_backend(module: &Module<BE>) -> usize;

    fn vec_znx_merge_rings_backend<'s>(
        module: &Module<BE>,
        res: &mut VecZnxBackendMut<'_, BE>,
        res_col: usize,
        a: &[VecZnxBackendRef<'_, BE>],
        a_col: usize,
        scratch: &mut ScratchArena<'s, BE>,
    );

    fn vec_znx_switch_ring_backend(
        module: &Module<BE>,
        res: &mut VecZnxBackendMut<'_, BE>,
        res_col: usize,
        a: &VecZnxBackendRef<'_, BE>,
        a_col: usize,
    );

    fn vec_znx_copy_backend(
        module: &Module<BE>,
        res: &mut VecZnxBackendMut<'_, BE>,
        res_col: usize,
        a: &VecZnxBackendRef<'_, BE>,
        a_col: usize,
    );

    fn vec_znx_copy_range_backend(
        module: &Module<BE>,
        res: &mut VecZnxBackendMut<'_, BE>,
        res_col: usize,
        res_limb: usize,
        res_offset: usize,
        a: &VecZnxBackendRef<'_, BE>,
        a_col: usize,
        a_limb: usize,
        a_offset: usize,
        len: usize,
    );

    fn vec_znx_extract_coeff_backend(
        module: &Module<BE>,
        res: &mut VecZnxBackendMut<'_, BE>,
        res_col: usize,
        a: &VecZnxBackendRef<'_, BE>,
        a_col: usize,
        a_coeff: usize,
    );

    fn vec_znx_fill_uniform_backend(
        module: &Module<BE>,
        base2k: usize,
        res: &mut VecZnxBackendMut<'_, BE>,
        res_col: usize,
        seed: [u8; 32],
    );

    fn vec_znx_fill_normal_backend(
        module: &Module<BE>,
        res_base2k: usize,
        res: &mut VecZnxBackendMut<'_, BE>,
        res_col: usize,
        noise_infos: NoiseInfos,
        seed: [u8; 32],
    );

    fn vec_znx_add_normal_backend(
        module: &Module<BE>,
        res_base2k: usize,
        res: &mut VecZnxBackendMut<'_, BE>,
        res_col: usize,
        noise_infos: NoiseInfos,
        seed: [u8; 32],
    );
}

/// Big-coefficient `VecZnxBig` extension point.
///
/// # Safety
/// Implementations must uphold the backend safety contract for backend-native
/// accumulator layouts and arithmetic correctness.
pub unsafe trait HalVecZnxBigImpl<BE: Backend>: Backend {
    fn vec_znx_big_from_small_backend(
        res: &mut crate::layouts::VecZnxBigBackendMut<'_, BE>,
        res_col: usize,
        a: &VecZnxBackendRef<'_, BE>,
        a_col: usize,
    );

    fn vec_znx_big_add_normal_backend(
        module: &Module<BE>,
        res_base2k: usize,
        res: &mut crate::layouts::VecZnxBigBackendMut<'_, BE>,
        res_col: usize,
        noise_infos: NoiseInfos,
        seed: [u8; 32],
    );

    fn vec_znx_big_add_normal(
        module: &Module<BE>,
        res_base2k: usize,
        res: &mut crate::layouts::VecZnxBigBackendMut<'_, BE>,
        res_col: usize,
        noise_infos: NoiseInfos,
        source: &mut Source,
    ) {
        Self::vec_znx_big_add_normal_backend(module, res_base2k, res, res_col, noise_infos, source.new_seed());
    }

    fn vec_znx_big_add_into(
        module: &Module<BE>,
        res: &mut crate::layouts::VecZnxBigBackendMut<'_, BE>,
        res_col: usize,
        a: &crate::layouts::VecZnxBigBackendRef<'_, BE>,
        a_col: usize,
        b: &crate::layouts::VecZnxBigBackendRef<'_, BE>,
        b_col: usize,
    );

    fn vec_znx_big_add_assign(
        module: &Module<BE>,
        res: &mut crate::layouts::VecZnxBigBackendMut<'_, BE>,
        res_col: usize,
        a: &crate::layouts::VecZnxBigBackendRef<'_, BE>,
        a_col: usize,
    );

    fn vec_znx_big_add_small_into_backend(
        module: &Module<BE>,
        res: &mut crate::layouts::VecZnxBigBackendMut<'_, BE>,
        res_col: usize,
        a: &crate::layouts::VecZnxBigBackendRef<'_, BE>,
        a_col: usize,
        b: &VecZnxBackendRef<'_, BE>,
        b_col: usize,
    );

    fn vec_znx_big_add_small_assign<'r, 'a>(
        module: &Module<BE>,
        res: &mut crate::layouts::VecZnxBigBackendMut<'r, BE>,
        res_col: usize,
        a: &VecZnxBackendRef<'a, BE>,
        a_col: usize,
    );

    fn vec_znx_big_sub(
        module: &Module<BE>,
        res: &mut crate::layouts::VecZnxBigBackendMut<'_, BE>,
        res_col: usize,
        a: &crate::layouts::VecZnxBigBackendRef<'_, BE>,
        a_col: usize,
        b: &crate::layouts::VecZnxBigBackendRef<'_, BE>,
        b_col: usize,
    );

    fn vec_znx_big_sub_assign(
        module: &Module<BE>,
        res: &mut crate::layouts::VecZnxBigBackendMut<'_, BE>,
        res_col: usize,
        a: &crate::layouts::VecZnxBigBackendRef<'_, BE>,
        a_col: usize,
    );

    fn vec_znx_big_sub_negate_assign(
        module: &Module<BE>,
        res: &mut crate::layouts::VecZnxBigBackendMut<'_, BE>,
        res_col: usize,
        a: &crate::layouts::VecZnxBigBackendRef<'_, BE>,
        a_col: usize,
    );

    fn vec_znx_big_sub_small_a_backend(
        module: &Module<BE>,
        res: &mut crate::layouts::VecZnxBigBackendMut<'_, BE>,
        res_col: usize,
        a: &VecZnxBackendRef<'_, BE>,
        a_col: usize,
        b: &crate::layouts::VecZnxBigBackendRef<'_, BE>,
        b_col: usize,
    );

    fn vec_znx_big_sub_small_assign<'r, 'a>(
        module: &Module<BE>,
        res: &mut crate::layouts::VecZnxBigBackendMut<'r, BE>,
        res_col: usize,
        a: &VecZnxBackendRef<'a, BE>,
        a_col: usize,
    );

    fn vec_znx_big_sub_small_b_backend(
        module: &Module<BE>,
        res: &mut crate::layouts::VecZnxBigBackendMut<'_, BE>,
        res_col: usize,
        a: &crate::layouts::VecZnxBigBackendRef<'_, BE>,
        a_col: usize,
        b: &VecZnxBackendRef<'_, BE>,
        b_col: usize,
    );

    fn vec_znx_big_sub_small_negate_assign<'r, 'a>(
        module: &Module<BE>,
        res: &mut crate::layouts::VecZnxBigBackendMut<'r, BE>,
        res_col: usize,
        a: &VecZnxBackendRef<'a, BE>,
        a_col: usize,
    );

    fn vec_znx_big_negate(
        module: &Module<BE>,
        res: &mut crate::layouts::VecZnxBigBackendMut<'_, BE>,
        res_col: usize,
        a: &crate::layouts::VecZnxBigBackendRef<'_, BE>,
        a_col: usize,
    );

    fn vec_znx_big_negate_assign(module: &Module<BE>, a: &mut crate::layouts::VecZnxBigBackendMut<'_, BE>, a_col: usize);

    fn vec_znx_big_normalize_tmp_bytes(module: &Module<BE>) -> usize;

    #[allow(clippy::too_many_arguments)]
    fn vec_znx_big_normalize<'s, 'r, 'a>(
        module: &Module<BE>,
        res: &mut VecZnxBackendMut<'r, BE>,
        res_base2k: usize,
        res_offset: i64,
        res_col: usize,
        a: &crate::layouts::VecZnxBigBackendRef<'a, BE>,
        a_base2k: usize,
        a_col: usize,
        scratch: &mut ScratchArena<'s, BE>,
    );

    fn vec_znx_big_automorphism(
        module: &Module<BE>,
        k: i64,
        res: &mut crate::layouts::VecZnxBigBackendMut<'_, BE>,
        res_col: usize,
        a: &crate::layouts::VecZnxBigBackendRef<'_, BE>,
        a_col: usize,
    );

    fn vec_znx_big_automorphism_assign_tmp_bytes(module: &Module<BE>) -> usize;

    fn vec_znx_big_automorphism_assign<'s>(
        module: &Module<BE>,
        k: i64,
        a: &mut crate::layouts::VecZnxBigBackendMut<'_, BE>,
        a_col: usize,
        scratch: &mut ScratchArena<'s, BE>,
    );
}

/// Prepared / DFT-domain `VecZnxDft` extension point.
///
/// # Safety
/// Implementations must uphold the backend safety contract for prepared-domain
/// layouts, transforms, and arithmetic correctness.
pub unsafe trait HalVecZnxDftImpl<BE: Backend>: Backend {
    fn vec_znx_dft_apply(
        module: &Module<BE>,
        step: usize,
        offset: usize,
        res: &mut crate::layouts::VecZnxDftBackendMut<'_, BE>,
        res_col: usize,
        a: &crate::layouts::VecZnxBackendRef<'_, BE>,
        a_col: usize,
    );

    fn vec_znx_idft_apply_tmp_bytes(module: &Module<BE>) -> usize;

    fn vec_znx_idft_apply<'s>(
        module: &Module<BE>,
        res: &mut crate::layouts::VecZnxBigBackendMut<'_, BE>,
        res_col: usize,
        a: &crate::layouts::VecZnxDftBackendRef<'_, BE>,
        a_col: usize,
        scratch: &mut ScratchArena<'s, BE>,
    );

    fn vec_znx_idft_apply_tmpa(
        module: &Module<BE>,
        res: &mut crate::layouts::VecZnxBigBackendMut<'_, BE>,
        res_col: usize,
        a: &mut crate::layouts::VecZnxDftBackendMut<'_, BE>,
        a_col: usize,
    );

    fn vec_znx_dft_add_into(
        module: &Module<BE>,
        res: &mut crate::layouts::VecZnxDftBackendMut<'_, BE>,
        res_col: usize,
        a: &crate::layouts::VecZnxDftBackendRef<'_, BE>,
        a_col: usize,
        b: &crate::layouts::VecZnxDftBackendRef<'_, BE>,
        b_col: usize,
    );

    fn vec_znx_dft_add_scaled_assign(
        module: &Module<BE>,
        res: &mut crate::layouts::VecZnxDftBackendMut<'_, BE>,
        res_col: usize,
        a: &crate::layouts::VecZnxDftBackendRef<'_, BE>,
        a_col: usize,
        a_scale: i64,
    );

    fn vec_znx_dft_add_assign(
        module: &Module<BE>,
        res: &mut crate::layouts::VecZnxDftBackendMut<'_, BE>,
        res_col: usize,
        a: &crate::layouts::VecZnxDftBackendRef<'_, BE>,
        a_col: usize,
    );

    fn vec_znx_dft_sub(
        module: &Module<BE>,
        res: &mut crate::layouts::VecZnxDftBackendMut<'_, BE>,
        res_col: usize,
        a: &crate::layouts::VecZnxDftBackendRef<'_, BE>,
        a_col: usize,
        b: &crate::layouts::VecZnxDftBackendRef<'_, BE>,
        b_col: usize,
    );

    fn vec_znx_dft_sub_assign(
        module: &Module<BE>,
        res: &mut crate::layouts::VecZnxDftBackendMut<'_, BE>,
        res_col: usize,
        a: &crate::layouts::VecZnxDftBackendRef<'_, BE>,
        a_col: usize,
    );

    fn vec_znx_dft_sub_negate_assign(
        module: &Module<BE>,
        res: &mut crate::layouts::VecZnxDftBackendMut<'_, BE>,
        res_col: usize,
        a: &crate::layouts::VecZnxDftBackendRef<'_, BE>,
        a_col: usize,
    );

    fn vec_znx_dft_copy(
        module: &Module<BE>,
        step: usize,
        offset: usize,
        res: &mut crate::layouts::VecZnxDftBackendMut<'_, BE>,
        res_col: usize,
        a: &crate::layouts::VecZnxDftBackendRef<'_, BE>,
        a_col: usize,
    );

    fn vec_znx_dft_zero(module: &Module<BE>, res: &mut crate::layouts::VecZnxDftBackendMut<'_, BE>, res_col: usize);
}

/// Scalar-vector product family extension point.
///
/// # Safety
/// Implementations must uphold the backend safety contract for prepared
/// polynomial layouts and arithmetic correctness.
pub unsafe trait HalSvpImpl<BE: Backend>: Backend {
    fn svp_prepare(
        module: &Module<BE>,
        res: &mut crate::layouts::SvpPPolBackendMut<'_, BE>,
        res_col: usize,
        a: &ScalarZnxBackendRef<'_, BE>,
        a_col: usize,
    );

    fn svp_ppol_copy_backend(
        module: &Module<BE>,
        res: &mut crate::layouts::SvpPPolBackendMut<'_, BE>,
        res_col: usize,
        a: &crate::layouts::SvpPPolBackendRef<'_, BE>,
        a_col: usize,
    );

    fn svp_apply_dft(
        module: &Module<BE>,
        res: &mut crate::layouts::VecZnxDftBackendMut<'_, BE>,
        res_col: usize,
        a: &crate::layouts::SvpPPolBackendRef<'_, BE>,
        a_col: usize,
        b: &crate::layouts::VecZnxBackendRef<'_, BE>,
        b_col: usize,
    );

    fn svp_apply_dft_to_dft(
        module: &Module<BE>,
        res: &mut crate::layouts::VecZnxDftBackendMut<'_, BE>,
        res_col: usize,
        a: &crate::layouts::SvpPPolBackendRef<'_, BE>,
        a_col: usize,
        b: &crate::layouts::VecZnxDftBackendRef<'_, BE>,
        b_col: usize,
    );

    fn svp_apply_dft_to_dft_assign(
        module: &Module<BE>,
        res: &mut crate::layouts::VecZnxDftBackendMut<'_, BE>,
        res_col: usize,
        a: &crate::layouts::SvpPPolBackendRef<'_, BE>,
        a_col: usize,
    );
}

/// Vector-matrix product family extension point.
///
/// # Safety
/// Implementations must uphold the backend safety contract for prepared matrix
/// layouts, scratch usage, and arithmetic correctness.
pub unsafe trait HalVmpImpl<BE: Backend>: Backend {
    fn vmp_prepare_tmp_bytes(module: &Module<BE>, rows: usize, cols_in: usize, cols_out: usize, size: usize) -> usize;

    fn vmp_prepare<'s>(
        module: &Module<BE>,
        res: &mut crate::layouts::VmpPMatBackendMut<'_, BE>,
        a: &crate::layouts::MatZnxBackendRef<'_, BE>,
        scratch: &mut ScratchArena<'s, BE>,
    );

    #[allow(clippy::too_many_arguments)]
    fn vmp_apply_dft_tmp_bytes(
        module: &Module<BE>,
        res_size: usize,
        a_size: usize,
        b_rows: usize,
        b_cols_in: usize,
        b_cols_out: usize,
        b_size: usize,
    ) -> usize;

    fn vmp_apply_dft<'s, R>(
        module: &Module<BE>,
        res: &mut R,
        a: &crate::layouts::VecZnxBackendRef<'_, BE>,
        b: &crate::layouts::VmpPMatBackendRef<'_, BE>,
        scratch: &mut ScratchArena<'s, BE>,
    ) where
        R: crate::layouts::VecZnxDftToBackendMut<BE>;

    #[allow(clippy::too_many_arguments)]
    fn vmp_apply_dft_to_dft_tmp_bytes(
        module: &Module<BE>,
        res_size: usize,
        a_size: usize,
        b_rows: usize,
        b_cols_in: usize,
        b_cols_out: usize,
        b_size: usize,
    ) -> usize;

    fn vmp_apply_dft_to_dft<'s, 'r>(
        module: &Module<BE>,
        res: &mut crate::layouts::VecZnxDftBackendMut<'r, BE>,
        a: &crate::layouts::VecZnxDftBackendRef<'_, BE>,
        b: &crate::layouts::VmpPMatBackendRef<'_, BE>,
        limb_offset: usize,
        scratch: &mut ScratchArena<'s, BE>,
    );

    #[allow(clippy::too_many_arguments)]
    fn vmp_apply_dft_to_dft_accumulate_tmp_bytes(
        module: &Module<BE>,
        res_size: usize,
        a_size: usize,
        b_rows: usize,
        b_cols_in: usize,
        b_cols_out: usize,
        b_size: usize,
    ) -> usize;

    fn vmp_apply_dft_to_dft_accumulate<'s, 'r>(
        module: &Module<BE>,
        res: &mut crate::layouts::VecZnxDftBackendMut<'r, BE>,
        a: &crate::layouts::VecZnxDftBackendRef<'_, BE>,
        b: &crate::layouts::VmpPMatBackendRef<'_, BE>,
        limb_offset: usize,
        scratch: &mut ScratchArena<'s, BE>,
    );

    fn vmp_zero(module: &Module<BE>, res: &mut crate::layouts::VmpPMatBackendMut<'_, BE>);
}

/// Convolution family extension point.
///
/// # Safety
/// Implementations must uphold the backend safety contract for prepared
/// convolution layouts, scratch usage, and arithmetic correctness.
pub unsafe trait HalConvolutionImpl<BE: Backend>: Backend {
    fn cnv_prepare_left_tmp_bytes(module: &Module<BE>, res_size: usize, a_size: usize) -> usize;

    fn cnv_prepare_left<'s, 'r>(
        module: &Module<BE>,
        res: &mut crate::layouts::CnvPVecLBackendMut<'r, BE>,
        a: &crate::layouts::VecZnxBackendRef<'_, BE>,
        mask: i64,
        scratch: &mut ScratchArena<'s, BE>,
    );

    fn cnv_prepare_right_tmp_bytes(module: &Module<BE>, res_size: usize, a_size: usize) -> usize;

    fn cnv_prepare_right<'s, 'r>(
        module: &Module<BE>,
        res: &mut crate::layouts::CnvPVecRBackendMut<'r, BE>,
        a: &crate::layouts::VecZnxBackendRef<'_, BE>,
        mask: i64,
        scratch: &mut ScratchArena<'s, BE>,
    );

    fn cnv_apply_dft_tmp_bytes(module: &Module<BE>, cnv_offset: usize, res_size: usize, a_size: usize, b_size: usize) -> usize;

    fn cnv_by_const_apply_tmp_bytes(
        module: &Module<BE>,
        cnv_offset: usize,
        res_size: usize,
        a_size: usize,
        b_size: usize,
    ) -> usize;

    #[allow(clippy::too_many_arguments)]
    fn cnv_by_const_apply<'s>(
        module: &Module<BE>,
        cnv_offset: usize,
        res: &mut crate::layouts::VecZnxBigBackendMut<'_, BE>,
        res_col: usize,
        a: &crate::layouts::VecZnxBackendRef<'_, BE>,
        a_col: usize,
        b: &crate::layouts::VecZnxBackendRef<'_, BE>,
        b_col: usize,
        b_coeff: usize,
        scratch: &mut ScratchArena<'s, BE>,
    );

    #[allow(clippy::too_many_arguments)]
    fn cnv_apply_dft<'s>(
        module: &Module<BE>,
        cnv_offset: usize,
        res: &mut crate::layouts::VecZnxDftBackendMut<'_, BE>,
        res_col: usize,
        a: &crate::layouts::CnvPVecLBackendRef<'_, BE>,
        a_col: usize,
        b: &crate::layouts::CnvPVecRBackendRef<'_, BE>,
        b_col: usize,
        scratch: &mut ScratchArena<'s, BE>,
    );

    fn cnv_pairwise_apply_dft_tmp_bytes(
        module: &Module<BE>,
        cnv_offset: usize,
        res_size: usize,
        a_size: usize,
        b_size: usize,
    ) -> usize;

    #[allow(clippy::too_many_arguments)]
    fn cnv_pairwise_apply_dft<'s>(
        module: &Module<BE>,
        cnv_offset: usize,
        res: &mut crate::layouts::VecZnxDftBackendMut<'_, BE>,
        res_col: usize,
        a: &crate::layouts::CnvPVecLBackendRef<'_, BE>,
        b: &crate::layouts::CnvPVecRBackendRef<'_, BE>,
        i: usize,
        j: usize,
        scratch: &mut ScratchArena<'s, BE>,
    );

    fn cnv_prepare_self_tmp_bytes(module: &Module<BE>, res_size: usize, a_size: usize) -> usize;

    fn cnv_prepare_self<'s, 'l, 'r>(
        module: &Module<BE>,
        left: &mut crate::layouts::CnvPVecLBackendMut<'l, BE>,
        right: &mut crate::layouts::CnvPVecRBackendMut<'r, BE>,
        a: &crate::layouts::VecZnxBackendRef<'_, BE>,
        mask: i64,
        scratch: &mut ScratchArena<'s, BE>,
    );
}
