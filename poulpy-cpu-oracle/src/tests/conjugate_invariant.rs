//! HAL suites on the conjugate-invariant oracles. Monomial operations are
//! absent from that ring and have no suites here.

use poulpy_hal::{backend_test_suite, cross_backend_test_suite};

cross_backend_test_suite! {
    mod vec_znx_ci,
    backend_ref =  crate::FFT64CIOracle,
    backend_test = crate::NTT4x30CIOracle,
    params = TestParams { size: 1<<8, base2k: 12, n: 8 },
    tests = {
        test_vec_znx_zero_matches_wrapper => poulpy_hal::test_suite::vec_znx::test_vec_znx_zero_matches_wrapper,
        test_vec_znx_add_matches_reference => poulpy_hal::test_suite::vec_znx::test_vec_znx_add_matches_reference,
        test_vec_znx_add_assign => poulpy_hal::test_suite::vec_znx::test_vec_znx_add_assign,
        test_vec_znx_add_assign_matches_wrapper => poulpy_hal::test_suite::vec_znx::test_vec_znx_add_assign_matches_wrapper,
        test_vec_znx_add_scalar_assign => poulpy_hal::test_suite::vec_znx::test_vec_znx_add_scalar_assign,
        test_vec_znx_sub => poulpy_hal::test_suite::vec_znx::test_vec_znx_sub,
        test_vec_znx_sub_assign => poulpy_hal::test_suite::vec_znx::test_vec_znx_sub_assign,
        test_vec_znx_sub_negate_assign => poulpy_hal::test_suite::vec_znx::test_vec_znx_sub_negate_assign,
        test_vec_znx_rsh => poulpy_hal::test_suite::vec_znx::test_vec_znx_rsh,
        test_vec_znx_rsh_assign => poulpy_hal::test_suite::vec_znx::test_vec_znx_rsh_assign,
        test_vec_znx_lsh => poulpy_hal::test_suite::vec_znx::test_vec_znx_lsh,
        test_vec_znx_lsh_assign => poulpy_hal::test_suite::vec_znx::test_vec_znx_lsh_assign,
        test_vec_znx_negate => poulpy_hal::test_suite::vec_znx::test_vec_znx_negate,
        test_vec_znx_negate_matches_wrapper => poulpy_hal::test_suite::vec_znx::test_vec_znx_negate_matches_wrapper,
        test_vec_znx_negate_assign => poulpy_hal::test_suite::vec_znx::test_vec_znx_negate_assign,
        test_vec_znx_negate_assign_matches_wrapper => poulpy_hal::test_suite::vec_znx::test_vec_znx_negate_assign_matches_wrapper,
        test_vec_znx_automorphism => poulpy_hal::test_suite::vec_znx::test_vec_znx_automorphism,
        test_vec_znx_automorphism_assign => poulpy_hal::test_suite::vec_znx::test_vec_znx_automorphism_assign,
        test_scalar_znx_automorphism => poulpy_hal::test_suite::vec_znx::test_scalar_znx_automorphism,
        test_vec_znx_normalize => poulpy_hal::test_suite::vec_znx::test_vec_znx_normalize,
        test_vec_znx_normalize_assign => poulpy_hal::test_suite::vec_znx::test_vec_znx_normalize_assign,
        test_vec_znx_switch_ring => poulpy_hal::test_suite::vec_znx::test_vec_znx_switch_ring,
        test_vec_znx_switch_ring_matches_wrapper => poulpy_hal::test_suite::vec_znx::test_vec_znx_switch_ring_matches_wrapper,
        test_vec_znx_copy => poulpy_hal::test_suite::vec_znx::test_vec_znx_copy,
        test_vec_znx_copy_matches_wrapper => poulpy_hal::test_suite::vec_znx::test_vec_znx_copy_matches_wrapper,
    }
}

cross_backend_test_suite! {
    mod svp_ci,
    backend_ref =  crate::FFT64CIOracle,
    backend_test = crate::NTT4x30CIOracle,
    params = TestParams { size: 1<<8, base2k: 12, n: 8 },
    tests = {
        test_svp_apply_dft => poulpy_hal::test_suite::svp::test_svp_apply_dft,
        test_svp_apply_dft_to_dft => poulpy_hal::test_suite::svp::test_svp_apply_dft_to_dft,
        test_svp_apply_dft_to_dft_assign => poulpy_hal::test_suite::svp::test_svp_apply_dft_to_dft_assign,
    }
}

cross_backend_test_suite! {
    mod vec_znx_big_ci,
    backend_ref =  crate::FFT64CIOracle,
    backend_test = crate::NTT4x30CIOracle,
    params = TestParams { size: 1<<8, base2k: 12, n: 8 },
    tests = {
        test_vec_znx_big_add => poulpy_hal::test_suite::vec_znx_big::test_vec_znx_big_add,
        test_vec_znx_big_add_assign => poulpy_hal::test_suite::vec_znx_big::test_vec_znx_big_add_assign,
        test_vec_znx_big_add_small => poulpy_hal::test_suite::vec_znx_big::test_vec_znx_big_add_small,
        test_vec_znx_big_add_small_assign => poulpy_hal::test_suite::vec_znx_big::test_vec_znx_big_add_small_assign,
        test_vec_znx_big_sub => poulpy_hal::test_suite::vec_znx_big::test_vec_znx_big_sub,
        test_vec_znx_big_sub_assign => poulpy_hal::test_suite::vec_znx_big::test_vec_znx_big_sub_assign,
        test_vec_znx_big_automorphism => poulpy_hal::test_suite::vec_znx_big::test_vec_znx_big_automorphism,
        test_vec_znx_big_automorphism_assign => poulpy_hal::test_suite::vec_znx_big::test_vec_znx_big_automorphism_assign,
        test_vec_znx_big_negate => poulpy_hal::test_suite::vec_znx_big::test_vec_znx_big_negate,
        test_vec_znx_big_negate_assign => poulpy_hal::test_suite::vec_znx_big::test_vec_znx_big_negate_assign,
        test_vec_znx_big_normalize => poulpy_hal::test_suite::vec_znx_big::test_vec_znx_big_normalize,
        test_vec_znx_big_sub_negate_assign => poulpy_hal::test_suite::vec_znx_big::test_vec_znx_big_sub_negate_assign,
        test_vec_znx_big_sub_small_a => poulpy_hal::test_suite::vec_znx_big::test_vec_znx_big_sub_small_a,
        test_vec_znx_big_sub_small_a_assign => poulpy_hal::test_suite::vec_znx_big::test_vec_znx_big_sub_small_a_assign,
        test_vec_znx_big_sub_small_b => poulpy_hal::test_suite::vec_znx_big::test_vec_znx_big_sub_small_b,
        test_vec_znx_big_sub_small_b_assign => poulpy_hal::test_suite::vec_znx_big::test_vec_znx_big_sub_small_b_assign,
        test_vec_znx_big_from_small => poulpy_hal::test_suite::vec_znx_big::test_vec_znx_big_from_small,
        test_vec_znx_big_inner_sum => poulpy_hal::test_suite::vec_znx_big::test_vec_znx_big_inner_sum,
        test_vec_znx_big_col_weighted_sum => poulpy_hal::test_suite::vec_znx_big::test_vec_znx_big_col_weighted_sum,
        test_vec_znx_scalar_product => poulpy_hal::test_suite::vec_znx_big::test_vec_znx_scalar_product,
    }
}

cross_backend_test_suite! {
    mod vec_znx_dft_ci,
    backend_ref =  crate::FFT64CIOracle,
    backend_test = crate::NTT4x30CIOracle,
    params = TestParams { size: 1<<8, base2k: 12, n: 8 },
    tests = {
        test_vec_znx_dft_add => poulpy_hal::test_suite::vec_znx_dft::test_vec_znx_dft_add,
        test_vec_znx_dft_add_assign => poulpy_hal::test_suite::vec_znx_dft::test_vec_znx_dft_add_assign,
        test_vec_znx_dft_sub => poulpy_hal::test_suite::vec_znx_dft::test_vec_znx_dft_sub,
        test_vec_znx_dft_sub_assign => poulpy_hal::test_suite::vec_znx_dft::test_vec_znx_dft_sub_assign,
        test_vec_znx_dft_sub_negate_assign => poulpy_hal::test_suite::vec_znx_dft::test_vec_znx_dft_sub_negate_assign,
        test_vec_znx_dft_copy => poulpy_hal::test_suite::vec_znx_dft::test_vec_znx_dft_copy,
        test_vec_znx_idft_apply => poulpy_hal::test_suite::vec_znx_dft::test_vec_znx_idft_apply,
        test_vec_znx_idft_apply_tmpa => poulpy_hal::test_suite::vec_znx_dft::test_vec_znx_idft_apply_tmpa,
        test_vec_znx_dft_apply => poulpy_hal::test_suite::vec_znx_dft::test_vec_znx_dft_apply,
        test_vec_znx_dft_zero => poulpy_hal::test_suite::vec_znx_dft::test_vec_znx_dft_zero,
    }
}

cross_backend_test_suite! {
    mod vec_znx_dft_automorphism_ci,
    backend_ref =  crate::FFT64CIOracle,
    backend_test = crate::NTT4x30CIOracle,
    params = TestParams { size: 1<<8, base2k: 12, n: 8 },
    tests = {
        test_vec_znx_dft_automorphism => poulpy_hal::test_suite::vec_znx_dft::test_vec_znx_dft_automorphism,
        test_vec_znx_dft_automorphism_add => poulpy_hal::test_suite::vec_znx_dft::test_vec_znx_dft_automorphism_add,
        test_vec_znx_idft_normalize_consume => poulpy_hal::test_suite::vec_znx_dft::test_vec_znx_idft_normalize_consume,
    }
}

cross_backend_test_suite! {
    mod vmp_ci,
    backend_ref =  crate::FFT64CIOracle,
    backend_test = crate::NTT4x30CIOracle,
    params = TestParams { size: 1<<8, base2k: 12, n: 8 },
    tests = {
        test_vmp_apply_dft => poulpy_hal::test_suite::vmp::test_vmp_apply_dft,
        test_vmp_apply_dft_to_dft => poulpy_hal::test_suite::vmp::test_vmp_apply_dft_to_dft,
        test_vmp_extract_selected_rows => poulpy_hal::test_suite::vmp::test_vmp_extract_selected_rows,
        test_vmp_apply_dft_to_dft_add => poulpy_hal::test_suite::vmp::test_vmp_apply_dft_to_dft_add,
        test_vmp_zero => poulpy_hal::test_suite::vmp::test_vmp_zero,
        test_word_compat_prepare_hint_sizes => poulpy_hal::test_suite::word_compat::test_word_compat_prepare_hint_sizes,
    }
}

backend_test_suite! {
    mod derived_fft64_ci,
    backend = crate::FFT64CIOracle,
    params = TestParams { size: 1<<8, base2k: 12, n: 8 },
    tests = {
        test_vmp_apply_dft_derived => poulpy_hal::test_suite::derived::test_vmp_apply_dft_derived,
        test_vmp_apply_dft_to_dft_add_derived => poulpy_hal::test_suite::derived::test_vmp_apply_dft_to_dft_add_derived,
        test_vec_znx_lsh_derived => poulpy_hal::test_suite::derived::test_vec_znx_lsh_derived,
        test_vec_znx_rsh_derived => poulpy_hal::test_suite::derived::test_vec_znx_rsh_derived,
        test_vec_znx_lsh_add_derived => poulpy_hal::test_suite::derived::test_vec_znx_lsh_add_derived,
        test_vec_znx_lsh_sub_derived => poulpy_hal::test_suite::derived::test_vec_znx_lsh_sub_derived,
        test_vec_znx_rsh_add_derived => poulpy_hal::test_suite::derived::test_vec_znx_rsh_add_derived,
        test_vec_znx_rsh_sub_derived => poulpy_hal::test_suite::derived::test_vec_znx_rsh_sub_derived,
        test_vec_znx_lsh_assign_derived => poulpy_hal::test_suite::derived::test_vec_znx_lsh_assign_derived,
        test_vec_znx_rsh_assign_derived => poulpy_hal::test_suite::derived::test_vec_znx_rsh_assign_derived,
        test_vec_znx_fill_uniform_source_all_derived => poulpy_hal::test_suite::derived::test_vec_znx_fill_uniform_source_all_derived,
        test_vec_znx_add_scalar_assign_derived => poulpy_hal::test_suite::derived::test_vec_znx_add_scalar_assign_derived,
        test_vec_znx_big_add_small_derived => poulpy_hal::test_suite::derived::test_vec_znx_big_add_small_derived,
        test_vec_znx_big_sub_small_a_derived => poulpy_hal::test_suite::derived::test_vec_znx_big_sub_small_a_derived,
        test_vec_znx_big_sub_small_b_derived => poulpy_hal::test_suite::derived::test_vec_znx_big_sub_small_b_derived,
        test_vec_znx_idft_normalize_consume_derived => poulpy_hal::test_suite::derived::test_vec_znx_idft_normalize_consume_derived,
        test_vec_znx_dft_automorphism_add_with_plan_derived => poulpy_hal::test_suite::derived::test_vec_znx_dft_automorphism_add_with_plan_derived,
        test_vec_znx_dft_automorphism_derived => poulpy_hal::test_suite::derived::test_vec_znx_dft_automorphism_derived,
        test_svp_apply_dft_derived => poulpy_hal::test_suite::derived::test_svp_apply_dft_derived,
        test_cnv_prepare_self_derived => poulpy_hal::test_suite::derived::test_cnv_prepare_self_derived,
        test_cnv_apply_dft_add_derived => poulpy_hal::test_suite::derived::test_cnv_apply_dft_add_derived,
        test_cnv_apply_dft_sum_derived => poulpy_hal::test_suite::derived::test_cnv_apply_dft_sum_derived,
        test_cnv_pairwise_apply_dft_derived => poulpy_hal::test_suite::derived::test_cnv_pairwise_apply_dft_derived,
        test_cnv_by_const_apply_add_derived => poulpy_hal::test_suite::derived::test_cnv_by_const_apply_add_derived,
    }
}

backend_test_suite! {
    mod derived_ntt4x30_ci,
    backend = crate::NTT4x30CIOracle,
    params = TestParams { size: 1<<8, base2k: 12, n: 8 },
    tests = {
        test_vmp_apply_dft_derived => poulpy_hal::test_suite::derived::test_vmp_apply_dft_derived,
        test_vmp_apply_dft_to_dft_add_derived => poulpy_hal::test_suite::derived::test_vmp_apply_dft_to_dft_add_derived,
        test_vec_znx_lsh_derived => poulpy_hal::test_suite::derived::test_vec_znx_lsh_derived,
        test_vec_znx_rsh_derived => poulpy_hal::test_suite::derived::test_vec_znx_rsh_derived,
        test_vec_znx_lsh_add_derived => poulpy_hal::test_suite::derived::test_vec_znx_lsh_add_derived,
        test_vec_znx_lsh_sub_derived => poulpy_hal::test_suite::derived::test_vec_znx_lsh_sub_derived,
        test_vec_znx_rsh_add_derived => poulpy_hal::test_suite::derived::test_vec_znx_rsh_add_derived,
        test_vec_znx_rsh_sub_derived => poulpy_hal::test_suite::derived::test_vec_znx_rsh_sub_derived,
        test_vec_znx_lsh_assign_derived => poulpy_hal::test_suite::derived::test_vec_znx_lsh_assign_derived,
        test_vec_znx_rsh_assign_derived => poulpy_hal::test_suite::derived::test_vec_znx_rsh_assign_derived,
        test_vec_znx_fill_uniform_source_all_derived => poulpy_hal::test_suite::derived::test_vec_znx_fill_uniform_source_all_derived,
        test_vec_znx_add_scalar_assign_derived => poulpy_hal::test_suite::derived::test_vec_znx_add_scalar_assign_derived,
        test_vec_znx_big_add_small_derived => poulpy_hal::test_suite::derived::test_vec_znx_big_add_small_derived,
        test_vec_znx_big_sub_small_a_derived => poulpy_hal::test_suite::derived::test_vec_znx_big_sub_small_a_derived,
        test_vec_znx_big_sub_small_b_derived => poulpy_hal::test_suite::derived::test_vec_znx_big_sub_small_b_derived,
        test_vec_znx_idft_normalize_consume_derived => poulpy_hal::test_suite::derived::test_vec_znx_idft_normalize_consume_derived,
        test_vec_znx_dft_automorphism_add_with_plan_derived => poulpy_hal::test_suite::derived::test_vec_znx_dft_automorphism_add_with_plan_derived,
        test_vec_znx_dft_automorphism_derived => poulpy_hal::test_suite::derived::test_vec_znx_dft_automorphism_derived,
        test_svp_apply_dft_derived => poulpy_hal::test_suite::derived::test_svp_apply_dft_derived,
        test_cnv_prepare_self_derived => poulpy_hal::test_suite::derived::test_cnv_prepare_self_derived,
        test_cnv_apply_dft_add_derived => poulpy_hal::test_suite::derived::test_cnv_apply_dft_add_derived,
        test_cnv_apply_dft_sum_derived => poulpy_hal::test_suite::derived::test_cnv_apply_dft_sum_derived,
        test_cnv_pairwise_apply_dft_derived => poulpy_hal::test_suite::derived::test_cnv_pairwise_apply_dft_derived,
        test_cnv_by_const_apply_add_derived => poulpy_hal::test_suite::derived::test_cnv_by_const_apply_add_derived,
    }
}

backend_test_suite! {
    mod sampling_ci,
    backend = crate::NTT4x30CIOracle,
    params = TestParams { size: 1<<12, base2k: 12, n: 8 },
    tests = {
        test_vec_znx_fill_uniform => poulpy_hal::test_suite::vec_znx::test_vec_znx_fill_uniform,
    }
}

#[cfg(feature = "enable-core")]
backend_test_suite! {
    mod sampling_core_ci,
    backend = crate::NTT4x30CIOracle,
    // the core suite computes at the module degree, no sweep
    params = TestParams { size: 1<<12, base2k: 12, n: 1<<12 },
    tests = {
        test_vec_znx_add_normal => poulpy_core::test_suite::sampling::test_vec_znx_add_normal,
        test_vec_znx_big_add_normal => poulpy_core::test_suite::sampling::test_vec_znx_big_add_normal,
    }
}

#[cfg(feature = "enable-core")]
backend_test_suite! {
    mod sampling_core_fft64_ci,
    backend = crate::FFT64CIOracle,
    // the core suite computes at the module degree, no sweep
    params = TestParams { size: 1<<12, base2k: 17, n: 1<<12 },
    tests = {
        test_vec_znx_add_normal => poulpy_core::test_suite::sampling::test_vec_znx_add_normal,
        test_vec_znx_big_add_normal => poulpy_core::test_suite::sampling::test_vec_znx_big_add_normal,
    }
}

backend_test_suite! {
    mod window_fft64_ci,
    backend = crate::FFT64CIOracle,
    params = TestParams { size: 1 << 8, base2k: 12, n: 8 },
    tests = {
        test_vec_znx_window_ops => poulpy_hal::test_suite::window::test_vec_znx_window_ops,
        test_vec_znx_big_window_ops => poulpy_hal::test_suite::window::test_vec_znx_big_window_ops,
        test_vec_znx_window_normalize_ops => poulpy_hal::test_suite::window::test_vec_znx_window_normalize_ops,
        test_vec_znx_big_window_normalize => poulpy_hal::test_suite::window::test_vec_znx_big_window_normalize,
        test_vec_znx_dft_step_zero_rejected => poulpy_hal::test_suite::vec_znx_dft::test_vec_znx_dft_step_zero_rejected,
        test_convolution_prepare_shape_rejected => poulpy_hal::test_suite::convolution::test_convolution_prepare_shape_rejected,
        test_vmp_apply_dft_to_dft_shape_rejected => poulpy_hal::test_suite::vmp::test_vmp_apply_dft_to_dft_shape_rejected,
        test_vec_znx_sparse_add_sub => poulpy_hal::test_suite::sparse::test_vec_znx_sparse_add_sub,
        test_vec_znx_big_sparse_add_sub => poulpy_hal::test_suite::sparse::test_vec_znx_big_sparse_add_sub,
        test_convolution_sparse => poulpy_hal::test_suite::convolution::test_convolution_sparse,
        test_convolution_by_const_degree_rejected => poulpy_hal::test_suite::convolution::test_convolution_by_const_degree_rejected,
        test_transfer_padded_lengths => poulpy_hal::test_suite::transfer::test_transfer_padded_lengths,
    }
}

backend_test_suite! {
    mod window_ntt4x30_ci,
    backend = crate::NTT4x30CIOracle,
    params = TestParams { size: 1 << 8, base2k: 12, n: 8 },
    tests = {
        test_vec_znx_window_ops => poulpy_hal::test_suite::window::test_vec_znx_window_ops,
        test_vec_znx_big_window_ops => poulpy_hal::test_suite::window::test_vec_znx_big_window_ops,
        test_vec_znx_window_normalize_ops => poulpy_hal::test_suite::window::test_vec_znx_window_normalize_ops,
        test_vec_znx_big_window_normalize => poulpy_hal::test_suite::window::test_vec_znx_big_window_normalize,
        test_vec_znx_dft_step_zero_rejected => poulpy_hal::test_suite::vec_znx_dft::test_vec_znx_dft_step_zero_rejected,
        test_convolution_prepare_shape_rejected => poulpy_hal::test_suite::convolution::test_convolution_prepare_shape_rejected,
        test_vmp_apply_dft_to_dft_shape_rejected => poulpy_hal::test_suite::vmp::test_vmp_apply_dft_to_dft_shape_rejected,
        test_vec_znx_sparse_add_sub => poulpy_hal::test_suite::sparse::test_vec_znx_sparse_add_sub,
        test_vec_znx_big_sparse_add_sub => poulpy_hal::test_suite::sparse::test_vec_znx_big_sparse_add_sub,
        test_convolution_sparse => poulpy_hal::test_suite::convolution::test_convolution_sparse,
        test_convolution_by_const_degree_rejected => poulpy_hal::test_suite::convolution::test_convolution_by_const_degree_rejected,
        test_transfer_padded_lengths => poulpy_hal::test_suite::transfer::test_transfer_padded_lengths,
    }
}

cross_backend_test_suite! {
    mod vec_znx_ci_maps,
    backend_ref =  crate::FFT64Oracle,
    backend_test = crate::NTT4x30Oracle,
    params = TestParams { size: 1<<8, base2k: 12, n: 8 },
    tests = {
        test_vec_znx_ci_embed_trace => poulpy_hal::test_suite::vec_znx::test_vec_znx_ci_embed_trace,
        test_scalar_znx_ci_embed => poulpy_hal::test_suite::vec_znx::test_scalar_znx_ci_embed,
    }
}

// Core parity between the two conjugate-invariant oracles, over the families
// that do not need monomial multiplication.
#[cfg(feature = "enable-core")]
poulpy_core::core_parity_test_suite! {
    mod core_parity_ci,
    backend_ref = crate::NTT4x30CIOracle,
    backend_test = crate::FFT64CIOracle,
    // computes at the module degree, no sweep
    params = TestParams { size: 1<<8, base2k: 12, n: 1<<8 },
    tests = {
        glwe_keyswitch => poulpy_core::test_suite::parity::test_glwe_keyswitch_parity,
        glwe_keyswitch_assign => poulpy_core::test_suite::parity::test_glwe_keyswitch_assign_parity,
        gglwe_keyswitch => poulpy_core::test_suite::parity::test_gglwe_keyswitch_parity,
        glwe_automorphism => poulpy_core::test_suite::parity::test_glwe_automorphism_parity,
        glwe_external_product => poulpy_core::test_suite::parity::test_glwe_external_product_parity,
        glwe_copy_zero => poulpy_core::test_suite::parity::test_glwe_copy_zero_parity,
        glwe_multiplication => poulpy_core::test_suite::parity::test_glwe_multiplication_parity,
        gadget_external_product => poulpy_core::test_suite::parity::test_gadget_external_product_parity,
        gadget_conversion => poulpy_core::test_suite::parity::test_gadget_conversion_parity,
        gglwe_product_digits_strided => poulpy_core::test_suite::parity::test_gglwe_product_digits_strided_parity,
        polynomial_evaluation => poulpy_core::test_suite::parity::test_polynomial_evaluation_parity,
        tensor_relinearize_decrypt => poulpy_core::test_suite::parity::test_tensor_relinearize_decrypt_parity,
        linear_transformation => poulpy_core::test_suite::parity::test_linear_transformation_parity,
        sampling => poulpy_core::test_suite::sampling::test_sampling_contract,
        preparation => poulpy_core::test_suite::parity::test_preparation_contract,
        glwe_add => poulpy_core::test_suite::parity::test_glwe_add_parity,
        glwe_sub => poulpy_core::test_suite::parity::test_glwe_sub_parity,
        glwe_negate => poulpy_core::test_suite::parity::test_glwe_negate_parity,
        glwe_normalize => poulpy_core::test_suite::parity::test_glwe_normalize_parity,
        glwe_tensor => poulpy_core::test_suite::parity::test_glwe_tensor_parity,
    }
}
