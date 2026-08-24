//! Shared test suites for `poulpy-core`, split by the question each answers.
//!
//! - [`noise`]: does this backend implement the scheme? Encrypt, operate,
//!   decrypt, compare the residual noise against the analytic bound. Judges one
//!   backend against a model, and is host-only because verification reads
//!   coefficients.
//! - [`parity`]: does this backend agree with a reference backend? Run the same
//!   operation on both over identical inputs and compare the outputs
//!   byte-for-byte. Needs no secrets, encryption or noise model, so a device
//!   backend can run it.
//!
//! The two are complementary: a bound is a weak oracle (a gadget-product
//! accumulator one limb too narrow passes the key-switch noise sweep), and
//! parity alone cannot tell you the reference is right.

pub mod noise;
pub mod parity;

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
                glwe_compressed_encrypt_sk => $crate::test_suite::noise::encryption::test_glwe_compressed_encrypt_sk,
                glwe_encrypt_zero_sk => $crate::test_suite::noise::encryption::test_glwe_encrypt_zero_sk,
                glwe_encrypt_pk => $crate::test_suite::noise::encryption::test_glwe_encrypt_pk,
                glwe_base2k_conv => $crate::test_suite::noise::test_glwe_base2k_conversion,
                test_glwe_tensoring => $crate::test_suite::noise::glwe_tensor::test_glwe_tensoring,
                test_glwe_tensor_square => $crate::test_suite::noise::glwe_tensor::test_glwe_tensor_square,
                test_glwe_tensor_fused_relinearize =>
                    $crate::test_suite::noise::glwe_tensor::test_glwe_tensor_fused_relinearize,
                test_glwe_mul_plain => $crate::test_suite::noise::glwe_tensor::test_glwe_mul_plain,
                test_glwe_mul_const => $crate::test_suite::noise::glwe_tensor::test_glwe_mul_const,
                glwe_keyswitch => $crate::test_suite::noise::keyswitch::test_glwe_keyswitch,
                glwe_keyswitch_assign => $crate::test_suite::noise::keyswitch::test_glwe_keyswitch_assign,
                glwe_automorphism => $crate::test_suite::noise::automorphism::test_glwe_automorphism,
                glwe_automorphism_assign => $crate::test_suite::noise::automorphism::test_glwe_automorphism_assign,
                glwe_external_product => $crate::test_suite::noise::external_product::test_glwe_external_product,
                glwe_external_product_assign => $crate::test_suite::noise::external_product::test_glwe_external_product_assign,
                glwe_keyswitch_ignores_dirty_scratch =>
                    $crate::test_suite::noise::keyswitch::test_glwe_keyswitch_ignores_dirty_scratch,
                glwe_external_product_ignores_dirty_scratch =>
                    $crate::test_suite::noise::keyswitch::test_glwe_external_product_ignores_dirty_scratch,
                glwe_rotate => $crate::test_suite::noise::test_glwe_rotate,
                glwe_trace_assign => $crate::test_suite::noise::test_glwe_trace_assign,
                glwe_hoisted_baby_rotations_match_automorphism =>
                    $crate::test_suite::noise::linear_transformation::test_glwe_hoisted_baby_rotations_match_automorphism,
                glwe_eval_linear_transformation_skips_empty_giant_steps =>
                    $crate::test_suite::noise::linear_transformation::test_glwe_eval_linear_transformation_skips_empty_giant_steps,
                glwe_prepared_giant_prods_match_sequential =>
                    $crate::test_suite::noise::linear_transformation::test_glwe_prepared_giant_prods_match_sequential,
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
                gglwe_switching_key_external_product =>
                    $crate::test_suite::noise::external_product::test_gglwe_switching_key_external_product,
                gglwe_switching_key_external_product_assign =>
                    $crate::test_suite::noise::external_product::test_gglwe_switching_key_external_product_assign,
                gglwe_automorphism_key_automorphism =>
                    $crate::test_suite::noise::automorphism::test_gglwe_automorphism_key_automorphism,
                gglwe_automorphism_key_automorphism_assign =>
                    $crate::test_suite::noise::automorphism::test_gglwe_automorphism_key_automorphism_assign,
                ggsw_encrypt_sk => $crate::test_suite::noise::encryption::test_ggsw_encrypt_sk,
                ggsw_compressed_encrypt_sk => $crate::test_suite::noise::encryption::test_ggsw_compressed_encrypt_sk,
                ggsw_keyswitch => $crate::test_suite::noise::keyswitch::test_ggsw_keyswitch,
                ggsw_keyswitch_assign => $crate::test_suite::noise::keyswitch::test_ggsw_keyswitch_assign,
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
