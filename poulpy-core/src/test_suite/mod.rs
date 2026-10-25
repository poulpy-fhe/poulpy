//! Shared test suites for `poulpy-core`, split by the question each answers.
//!
//! - [`noise`]: does this backend implement the scheme? Encrypt, operate,
//!   decrypt, compare the residual noise against the analytic bound. Judges one
//!   backend against a model, and is host-only because verification reads
//!   coefficients.
//! - [`parity`]: does this backend agree with the selected comparison backend?
//!   Run the same operation on both over identical inputs and compare integer
//!   outputs and metadata. Encryption parity also controls the sampled inputs.
//!   An already validated backend can bootstrap another through these tests.
//!
//! The two are complementary: a bound is a weak oracle (a gadget-product
//! accumulator one limb too narrow passes the key-switch noise sweep), and
//! parity alone cannot tell you the reference is right.

pub mod copy;
pub mod decryption;
pub mod encryption;
pub mod keys;
pub mod noise;
pub mod parity;
pub(crate) mod refusal;
pub mod sampling;
pub mod serialization;

#[macro_export]
macro_rules! core_backend_test_suite {
    (
        mod $modname:ident,
        backend = $backend:ty,
        params = $params:expr $(,)?
    ) => {
        poulpy_hal::backend_test_suite!(
            mod $modname,
            backend = $backend,
            params = $params,
            tests = {
                glwe_encrypt_sk => $crate::test_suite::noise::encryption::test_glwe_encrypt_sk,
                glwe_encrypt_sk_with_mask => $crate::test_suite::noise::encryption::test_glwe_encrypt_sk_with_mask,
                glwe_compressed_encrypt_sk => $crate::test_suite::noise::encryption::test_glwe_compressed_encrypt_sk,
                glwe_encrypt_zero_sk => $crate::test_suite::noise::encryption::test_glwe_encrypt_zero_sk,
                glwe_encrypt_pk => $crate::test_suite::noise::encryption::test_glwe_encrypt_pk,
                glwe_encrypt_pk_rejects_imprecise_key => $crate::test_suite::noise::encryption::test_glwe_encrypt_pk_rejects_imprecise_key,
                glwe_encrypt_pk_replay => $crate::test_suite::noise::encryption::test_glwe_encrypt_pk_replay,
                glwe_encrypt_zero_pk_rejects_zero_ephemeral => $crate::test_suite::noise::encryption::test_glwe_encrypt_zero_pk_rejects_zero_ephemeral,
                glwe_encrypt_zero_pk_rejects_nan_ephemeral => $crate::test_suite::noise::encryption::test_glwe_encrypt_zero_pk_rejects_nan_ephemeral,
                #[should_panic(expected = "invalid public key: rank must be at least 1")]
                glwe_public_key_alloc_rejects_rank_zero => $crate::test_suite::noise::encryption::test_glwe_public_key_alloc_rejects_rank_zero,
                scalar_znx_fill_distribution => $crate::test_suite::sampling::test_scalar_znx_fill_distribution,
                glwe_base2k_conv => $crate::test_suite::noise::test_glwe_base2k_conversion,
                glwe_copy => $crate::test_suite::copy::test_glwe_copy,
                test_glwe_tensoring => $crate::test_suite::noise::glwe_tensor::test_glwe_tensoring,
                glwe_mul_relinearize_sum_rejects_no_pairs =>
                    $crate::test_suite::noise::glwe_tensor::test_glwe_mul_relinearize_sum_rejects_no_pairs,
                glwe_mul_relinearize_sum_rejects_mismatched_layouts =>
                    $crate::test_suite::noise::glwe_tensor::test_glwe_mul_relinearize_sum_rejects_mismatched_layouts,
                glwe_mul_relinearize_sum_rejects_unequal_lengths =>
                    $crate::test_suite::noise::glwe_tensor::test_glwe_mul_relinearize_sum_rejects_unequal_lengths,
                glwe_mul_prepared_relinearize_sum_rejects_no_pairs =>
                    $crate::test_suite::noise::glwe_tensor::test_glwe_mul_prepared_relinearize_sum_rejects_no_pairs,
                glwe_mul_prepared_relinearize_sum_rejects_mismatched_pairs =>
                    $crate::test_suite::noise::glwe_tensor::test_glwe_mul_prepared_relinearize_sum_rejects_mismatched_pairs,
                glwe_mul_prepared_relinearize_sum_rejects_size_of_another_width =>
                    $crate::test_suite::noise::glwe_tensor::test_glwe_mul_prepared_relinearize_sum_rejects_size_of_another_width,
                glwe_mul_prepared_relinearize_sum_rejects_unequal_lengths =>
                    $crate::test_suite::noise::glwe_tensor::test_glwe_mul_prepared_relinearize_sum_rejects_unequal_lengths,
                glwe_tensor_apply_prepared_rejects_mismatched_operands =>
                    $crate::test_suite::noise::glwe_tensor::test_glwe_tensor_apply_prepared_rejects_mismatched_operands,
                glwe_prepare_left_rejects_size_of_another_width =>
                    $crate::test_suite::noise::glwe_tensor::test_glwe_prepare_left_rejects_size_of_another_width,
                test_glwe_tensor_square => $crate::test_suite::noise::glwe_tensor::test_glwe_tensor_square,
                glwe_tensor_relinearize_coarsened => $crate::test_suite::parity::test_glwe_tensor_relinearize_coarsened,
                test_glwe_mul_plain => $crate::test_suite::noise::glwe_tensor::test_glwe_mul_plain,
                test_glwe_mul_plain_plaintext_degree => $crate::test_suite::noise::glwe_tensor::test_glwe_mul_plain_plaintext_degree,
                test_glwe_mul_const => $crate::test_suite::noise::glwe_tensor::test_glwe_mul_const,
                glwe_keyswitch => $crate::test_suite::noise::keyswitch::test_glwe_keyswitch,
                glwe_keyswitch_assign => $crate::test_suite::noise::keyswitch::test_glwe_keyswitch_assign,
                glwe_keyswitch_row_offset => $crate::test_suite::noise::keyswitch::test_glwe_keyswitch_row_offset,
                glwe_keyswitch_dft_add => $crate::test_suite::noise::keyswitch::test_glwe_keyswitch_dft_add,
                glwe_keyswitch_row_offset_refusals => $crate::test_suite::noise::keyswitch::test_glwe_keyswitch_row_offset_refusals,
                glwe_automorphism => $crate::test_suite::noise::automorphism::test_glwe_automorphism,
                glwe_automorphism_assign => $crate::test_suite::noise::automorphism::test_glwe_automorphism_assign,
                glwe_automorphism_coarsened => $crate::test_suite::parity::test_glwe_automorphism_coarsened,
                glwe_tensor_relinearize_cross_radix =>
                    $crate::test_suite::noise::glwe_tensor::test_glwe_tensor_relinearize_cross_radix,
                glwe_external_product => $crate::test_suite::noise::external_product::test_glwe_external_product,
                glwe_external_product_assign => $crate::test_suite::noise::external_product::test_glwe_external_product_assign,
                glwe_keyswitch_k_digit => $crate::test_suite::noise::test_glwe_keyswitch_k_digit,
                glwe_external_product_k_digit => $crate::test_suite::noise::test_glwe_external_product_k_digit,
                glwe_keyswitch_ignores_dirty_scratch =>
                    $crate::test_suite::noise::keyswitch::test_glwe_keyswitch_ignores_dirty_scratch,
                glwe_external_product_ignores_dirty_scratch =>
                    $crate::test_suite::noise::keyswitch::test_glwe_external_product_ignores_dirty_scratch,
                glwe_rotate => $crate::test_suite::noise::test_glwe_rotate,
                glwe_shift_saturates => $crate::test_suite::noise::test_glwe_shift_saturates,
                glwe_shift_exact_scratch => $crate::test_suite::noise::test_glwe_shift_exact_scratch,
                glwe_trace_assign => $crate::test_suite::noise::test_glwe_trace_assign,
                glwe_trace_sum => $crate::test_suite::noise::test_glwe_trace_sum,
                glwe_trace_coarsened => $crate::test_suite::parity::test_glwe_trace_coarsened,
                glwe_hoisted_baby_rotations_match_automorphism =>
                    $crate::test_suite::noise::linear_transformation::test_glwe_hoisted_baby_rotations_match_automorphism,
                glwe_prepare_linear_transformation_rhs_rejects_wide_diagonal =>
                    $crate::test_suite::noise::linear_transformation::test_glwe_prepare_linear_transformation_rhs_rejects_wide_diagonal,
                glwe_linear_transformation_sum => $crate::test_suite::noise::linear_transformation::test_glwe_linear_transformation_sum,
                glwe_linear_transformation_powers_of_two =>
                    $crate::test_suite::noise::linear_transformation::test_glwe_linear_transformation_powers_of_two,
                glwe_linear_transformation_sum_rejects_mismatched_schedule =>
                    $crate::test_suite::noise::linear_transformation::test_glwe_linear_transformation_sum_rejects_mismatched_schedule,
                glwe_linear_transformation_sum_rejects_mismatched_diagonal_width =>
                    $crate::test_suite::noise::linear_transformation::test_glwe_linear_transformation_sum_rejects_mismatched_diagonal_width,
                glwe_linear_transformation_sum_rejects_mismatched_diagonal_scale =>
                    $crate::test_suite::noise::linear_transformation::test_glwe_linear_transformation_sum_rejects_mismatched_diagonal_scale,
                glwe_linear_transformation_sum_rejects_mismatched_source =>
                    $crate::test_suite::noise::linear_transformation::test_glwe_linear_transformation_sum_rejects_mismatched_source,
                glwe_linear_transformation_sum_rejects_no_terms =>
                    $crate::test_suite::noise::linear_transformation::test_glwe_linear_transformation_sum_rejects_no_terms,
                glwe_linear_transformation_sum_rejects_unequal_lengths =>
                    $crate::test_suite::noise::linear_transformation::test_glwe_linear_transformation_sum_rejects_unequal_lengths,
                glwe_linear_transformation_sum_rejects_first_term_without_giant_steps =>
                    $crate::test_suite::noise::linear_transformation::test_glwe_linear_transformation_sum_rejects_first_term_without_giant_steps,
                glwe_linear_transformation_rejects_no_giant_steps =>
                    $crate::test_suite::noise::linear_transformation::test_glwe_linear_transformation_rejects_no_giant_steps,
                glwe_linear_transformation_rejects_empty_giant_step =>
                    $crate::test_suite::noise::linear_transformation::test_glwe_linear_transformation_rejects_empty_giant_step,
                glwe_linear_transformation_sum_rejects_empty_giant_step =>
                    $crate::test_suite::noise::linear_transformation::test_glwe_linear_transformation_sum_rejects_empty_giant_step,
                glwe_linear_transformation_sum_rejects_missing_baby_step =>
                    $crate::test_suite::noise::linear_transformation::test_glwe_linear_transformation_sum_rejects_missing_baby_step,
                glwe_linear_transformation_sum_rejects_empty_cache =>
                    $crate::test_suite::noise::linear_transformation::test_glwe_linear_transformation_sum_rejects_empty_cache,
                glwe_linear_transformation_rejects_uneven_giant_step_width =>
                    $crate::test_suite::noise::linear_transformation::test_glwe_linear_transformation_rejects_uneven_giant_step_width,
                glwe_linear_transformation_sum_rejects_uneven_giant_step_width =>
                    $crate::test_suite::noise::linear_transformation::test_glwe_linear_transformation_sum_rejects_uneven_giant_step_width,
                glwe_packing => $crate::test_suite::noise::test_glwe_packing,
                gglwe_switching_key_encrypt_sk => $crate::test_suite::noise::encryption::test_gglwe_switching_key_encrypt_sk,
                gglwe_switching_key_compressed_encrypt_sk =>
                    $crate::test_suite::noise::encryption::test_gglwe_switching_key_compressed_encrypt_sk,
                gglwe_compressed_encrypt_sk => $crate::test_suite::noise::encryption::test_gglwe_compressed_encrypt_sk,
                gglwe_automorphism_key_encrypt_sk => $crate::test_suite::noise::encryption::test_gglwe_automorphism_key_encrypt_sk,
                gglwe_automorphism_key_compressed_encrypt_sk =>
                    $crate::test_suite::noise::encryption::test_gglwe_automorphism_key_compressed_encrypt_sk,
                gglwe_tensor_key_encrypt_sk => $crate::test_suite::noise::encryption::test_gglwe_tensor_key_encrypt_sk,
                gglwe_tensor_key_compressed_encrypt_sk =>
                    $crate::test_suite::noise::encryption::test_gglwe_tensor_key_compressed_encrypt_sk,
                gglwe_to_ggsw_key_encrypt_sk => $crate::test_suite::noise::encryption::test_gglwe_to_ggsw_key_encrypt_sk,
                gglwe_switching_key_keyswitch => $crate::test_suite::noise::keyswitch::test_gglwe_switching_key_keyswitch,
                gglwe_switching_key_keyswitch_assign => $crate::test_suite::noise::keyswitch::test_gglwe_switching_key_keyswitch_assign,
                gglwe_switching_key_keyswitch_own_layout =>
                    $crate::test_suite::noise::keyswitch::test_gglwe_switching_key_keyswitch_own_layout,
                gglwe_copy => $crate::test_suite::noise::test_gglwe_copy,
                gglwe_switching_key_external_product =>
                    $crate::test_suite::noise::external_product::test_gglwe_switching_key_external_product,
                gglwe_switching_key_external_product_assign =>
                    $crate::test_suite::noise::external_product::test_gglwe_switching_key_external_product_assign,
                gglwe_automorphism_key_automorphism =>
                    $crate::test_suite::noise::automorphism::test_gglwe_automorphism_key_automorphism,
                gglwe_automorphism_key_automorphism_assign =>
                    $crate::test_suite::noise::automorphism::test_gglwe_automorphism_key_automorphism_assign,
                gglwe_automorphism_key_automorphism_own_layout =>
                    $crate::test_suite::noise::automorphism::test_gglwe_automorphism_key_automorphism_own_layout,
                gadget_row_plaintext => $crate::test_suite::encryption::test_gadget_row_plaintext,
                ggsw_encrypt_sk => $crate::test_suite::noise::encryption::test_ggsw_encrypt_sk,
                ggsw_encrypt_pk => $crate::test_suite::noise::encryption::test_ggsw_encrypt_pk,
                ggsw_encrypt_pk_unnormalized_plaintext =>
                    $crate::test_suite::noise::encryption::test_ggsw_encrypt_pk_unnormalized_plaintext,
                ggsw_compressed_encrypt_sk => $crate::test_suite::noise::encryption::test_ggsw_compressed_encrypt_sk,
                ggsw_keyswitch => $crate::test_suite::noise::keyswitch::test_ggsw_keyswitch,
                ggsw_keyswitch_assign => $crate::test_suite::noise::keyswitch::test_ggsw_keyswitch_assign,
                ggsw_keyswitch_own_layout => $crate::test_suite::noise::keyswitch::test_ggsw_keyswitch_own_layout,
                ggsw_copy => $crate::test_suite::noise::test_ggsw_copy,
                ggsw_external_product => $crate::test_suite::noise::external_product::test_ggsw_external_product,
                ggsw_external_product_assign => $crate::test_suite::noise::external_product::test_ggsw_external_product_assign,
                ggsw_automorphism => $crate::test_suite::noise::automorphism::test_ggsw_automorphism,
                ggsw_automorphism_assign => $crate::test_suite::noise::automorphism::test_ggsw_automorphism_assign,
                lwe_keyswitch => $crate::test_suite::noise::keyswitch::test_lwe_keyswitch,
                glwe_to_lwe => $crate::test_suite::noise::test_glwe_to_lwe,
                lwe_to_glwe => $crate::test_suite::noise::test_lwe_to_glwe,
                glwe_expand_lwe => $crate::test_suite::noise::test_glwe_expand_lwe,
                glwe_expand_lwe_matrix_decrypt => $crate::test_suite::noise::test_glwe_expand_lwe_matrix_decrypt,
                glwe_expand_lwe_rejects_incompatible_lwe_layout =>
                    $crate::test_suite::noise::test_glwe_expand_lwe_rejects_incompatible_lwe_layout,
                lwe_read_from_rejects_malformed_shape => $crate::test_suite::noise::test_lwe_read_from_rejects_malformed_shape,
                lwe_secret_from_glwe_secret_flattens_rank_and_preserves_metadata =>
                    $crate::test_suite::noise::test_lwe_secret_from_glwe_secret_flattens_rank_and_preserves_metadata,
            }
        );
    };
}

pub use crate::core_backend_test_suite;

/// Builds precisely the advertised scratch capacity, poisoned before each operation.
/// No other operation or backend's budget can conceal an underestimate.
pub(crate) fn poisoned_scratch<B: poulpy_hal::layouts::Backend>(bytes: usize) -> poulpy_hal::layouts::ScratchOwned<B> {
    poulpy_hal::layouts::ScratchOwned {
        data: B::from_host_bytes(&vec![0xA5; bytes]),
        _phantom: std::marker::PhantomData,
    }
}

/// Runs `op` on exactly `bytes` of poisoned scratch and asserts it left every
/// byte zero, as an operation that handles a secret must.
pub fn assert_wipes_scratch<BE: poulpy_hal::layouts::Backend>(
    bytes: usize,
    op: impl FnOnce(&mut poulpy_hal::layouts::ScratchArena<'_, BE>),
) {
    let mut scratch: poulpy_hal::layouts::ScratchOwned<BE> = poulpy_hal::layouts::ScratchOwned {
        data: BE::from_host_bytes(&vec![0xA5; bytes]),
        _phantom: std::marker::PhantomData,
    };
    op(&mut scratch.arena());
    assert!(
        BE::to_host_bytes(&scratch.data).iter().all(|&b| b == 0),
        "the operation left data in its scratch"
    );
}

/// The left and right hints of prepared pair `i`: they swap from one pair to the next, so that two pairs cover both
/// hints on each side.
pub(crate) fn prepare_hints(i: usize) -> (poulpy_hal::layouts::PrepareHint, poulpy_hal::layouts::PrepareHint) {
    use poulpy_hal::layouts::PrepareHint::{OneShot, Reuse};
    if i.is_multiple_of(2) {
        (OneShot, Reuse)
    } else {
        (Reuse, OneShot)
    }
}
