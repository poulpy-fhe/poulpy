use poulpy_hal::{
    layouts::Module,
    test_suite::convolution::{
        test_convolution, test_convolution_add, test_convolution_by_const, test_convolution_by_const_add,
        test_convolution_pairwise, test_convolution_sum,
    },
};

use crate::{FFT64Oracle, NTT4x30Oracle};

#[cfg(feature = "enable-ckks")]
mod ckks_tests;
mod conjugate_invariant;
mod derived_scratch;

#[test]
fn test_convolution_by_const_fft64_oracle() {
    let module: Module<FFT64Oracle> = Module::<FFT64Oracle>::new(64);
    test_convolution_by_const(&module, 8, 17);
    test_convolution_by_const(&module, 64, 17);
    test_convolution_by_const_add(&module, 8, 17);
    test_convolution_by_const_add(&module, 64, 17);
}

#[test]
fn test_convolution_fft64_oracle() {
    let module: Module<FFT64Oracle> = Module::<FFT64Oracle>::new(64);
    test_convolution(&module, 8, 17);
    test_convolution(&module, 64, 17);
}

#[test]
fn test_convolution_pairwise_fft64_oracle() {
    let module: Module<FFT64Oracle> = Module::<FFT64Oracle>::new(64);
    test_convolution_pairwise(&module, 8, 17);
    test_convolution_pairwise(&module, 64, 17);
}

#[test]
fn test_convolution_add_fft64_oracle() {
    let module: Module<FFT64Oracle> = Module::<FFT64Oracle>::new(64);
    test_convolution_add(&module, 8, 17);
    test_convolution_add(&module, 64, 17);
}

#[test]
fn test_convolution_sum_fft64_oracle() {
    let module: Module<FFT64Oracle> = Module::<FFT64Oracle>::new(64);
    test_convolution_sum(&module, 8, 17);
    test_convolution_sum(&module, 64, 17);
}

#[test]
fn test_convolution_by_const_ntt4x30_oracle() {
    let module: Module<NTT4x30Oracle> = Module::<NTT4x30Oracle>::new(64);
    test_convolution_by_const(&module, 8, 50);
    test_convolution_by_const(&module, 64, 50);
    test_convolution_by_const_add(&module, 8, 50);
    test_convolution_by_const_add(&module, 64, 50);
}

#[test]
fn test_convolution_ntt4x30_oracle() {
    let module: Module<NTT4x30Oracle> = Module::<NTT4x30Oracle>::new(64);
    test_convolution(&module, 8, 50);
    test_convolution(&module, 64, 50);
}

#[test]
fn test_convolution_pairwise_ntt4x30_oracle() {
    let module: Module<NTT4x30Oracle> = Module::<NTT4x30Oracle>::new(64);
    test_convolution_pairwise(&module, 8, 50);
    test_convolution_pairwise(&module, 64, 50);
}

#[test]
fn test_convolution_add_ntt4x30_oracle() {
    let module: Module<NTT4x30Oracle> = Module::<NTT4x30Oracle>::new(64);
    test_convolution_add(&module, 8, 50);
    test_convolution_add(&module, 64, 50);
}

#[test]
fn test_convolution_sum_ntt4x30_oracle() {
    let module: Module<NTT4x30Oracle> = Module::<NTT4x30Oracle>::new(64);
    test_convolution_sum(&module, 8, 50);
    test_convolution_sum(&module, 64, 50);
}

use poulpy_hal::{backend_test_suite, cross_backend_test_suite};

cross_backend_test_suite! {
    mod vec_znx,
    backend_ref =  crate::FFT64Oracle,
    backend_test = crate::NTT4x30Oracle,
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
        test_vec_znx_rotate => poulpy_hal::test_suite::vec_znx::test_vec_znx_rotate,
        test_vec_znx_rotate_assign => poulpy_hal::test_suite::vec_znx::test_vec_znx_rotate_assign,
        test_vec_znx_automorphism => poulpy_hal::test_suite::vec_znx::test_vec_znx_automorphism,
        test_vec_znx_automorphism_assign => poulpy_hal::test_suite::vec_znx::test_vec_znx_automorphism_assign,
        test_scalar_znx_automorphism => poulpy_hal::test_suite::vec_znx::test_scalar_znx_automorphism,
        test_vec_znx_mul_xp_minus_one => poulpy_hal::test_suite::vec_znx::test_vec_znx_mul_xp_minus_one,
        test_vec_znx_mul_xp_minus_one_assign => poulpy_hal::test_suite::vec_znx::test_vec_znx_mul_xp_minus_one_assign,
        test_vec_znx_normalize => poulpy_hal::test_suite::vec_znx::test_vec_znx_normalize,
        test_vec_znx_normalize_assign => poulpy_hal::test_suite::vec_znx::test_vec_znx_normalize_assign,
        test_vec_znx_switch_ring => poulpy_hal::test_suite::vec_znx::test_vec_znx_switch_ring,
        test_vec_znx_switch_ring_matches_wrapper => poulpy_hal::test_suite::vec_znx::test_vec_znx_switch_ring_matches_wrapper,
        test_vec_znx_copy => poulpy_hal::test_suite::vec_znx::test_vec_znx_copy,
        test_vec_znx_copy_matches_wrapper => poulpy_hal::test_suite::vec_znx::test_vec_znx_copy_matches_wrapper,
    }
}
cross_backend_test_suite! {
    mod svp,
    backend_ref =  crate::FFT64Oracle,
    backend_test = crate::NTT4x30Oracle,
    params = TestParams { size: 1<<8, base2k: 12, n: 8 },
    tests = {
        test_svp_apply_dft => poulpy_hal::test_suite::svp::test_svp_apply_dft,
        test_svp_apply_dft_to_dft => poulpy_hal::test_suite::svp::test_svp_apply_dft_to_dft,
        test_svp_apply_dft_to_dft_assign => poulpy_hal::test_suite::svp::test_svp_apply_dft_to_dft_assign,
    }
}
cross_backend_test_suite! {
    mod vec_znx_big,
    backend_ref =  crate::FFT64Oracle,
    backend_test = crate::NTT4x30Oracle,
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
    mod vec_znx_dft,
    backend_ref =  crate::FFT64Oracle,
    backend_test = crate::NTT4x30Oracle,
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
    mod vec_znx_dft_automorphism,
    backend_ref =  crate::FFT64Oracle,
    backend_test = crate::NTT4x30Oracle,
    params = TestParams { size: 1<<8, base2k: 12, n: 8 },
    tests = {
        test_vec_znx_dft_automorphism => poulpy_hal::test_suite::vec_znx_dft::test_vec_znx_dft_automorphism,
        test_vec_znx_dft_automorphism_add => poulpy_hal::test_suite::vec_znx_dft::test_vec_znx_dft_automorphism_add,
        test_vec_znx_idft_normalize_consume => poulpy_hal::test_suite::vec_znx_dft::test_vec_znx_idft_normalize_consume,
    }
}
cross_backend_test_suite! {
    mod vmp,
    backend_ref =  crate::FFT64Oracle,
    backend_test = crate::NTT4x30Oracle,
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
    mod derived_fft64,
    backend = crate::FFT64Oracle,
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
        test_vec_znx_mul_xp_minus_one_derived => poulpy_hal::test_suite::derived::test_vec_znx_mul_xp_minus_one_derived,
        test_vec_znx_mul_xp_minus_one_assign_derived => poulpy_hal::test_suite::derived::test_vec_znx_mul_xp_minus_one_assign_derived,
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
    mod derived_ntt4x30,
    backend = crate::NTT4x30Oracle,
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
        test_vec_znx_mul_xp_minus_one_derived => poulpy_hal::test_suite::derived::test_vec_znx_mul_xp_minus_one_derived,
        test_vec_znx_mul_xp_minus_one_assign_derived => poulpy_hal::test_suite::derived::test_vec_znx_mul_xp_minus_one_assign_derived,
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
    mod sampling,
    backend = crate::NTT4x30Oracle,
    params = TestParams { size: 1<<12, base2k: 12, n: 8 },
    tests = {
        test_vec_znx_fill_uniform => poulpy_hal::test_suite::vec_znx::test_vec_znx_fill_uniform,
    }
}

// Gated on `enable-core`, like `core_impl`, which implements the noise seam for both oracle backends.
#[cfg(feature = "enable-core")]
backend_test_suite! {
    mod sampling_core,
    backend = crate::NTT4x30Oracle,
    // the core suite computes at the module degree, no sweep
    params = TestParams { size: 1<<12, base2k: 12, n: 1<<12 },
    tests = {
        test_vec_znx_add_normal => poulpy_core::test_suite::sampling::test_vec_znx_add_normal,
        test_vec_znx_big_add_normal => poulpy_core::test_suite::sampling::test_vec_znx_big_add_normal,
    }
}

#[cfg(feature = "enable-core")]
backend_test_suite! {
    mod sampling_core_fft64,
    backend = crate::FFT64Oracle,
    // the core suite computes at the module degree, no sweep
    params = TestParams { size: 1<<12, base2k: 17, n: 1<<12 },
    tests = {
        test_vec_znx_add_normal => poulpy_core::test_suite::sampling::test_vec_znx_add_normal,
        test_vec_znx_big_add_normal => poulpy_core::test_suite::sampling::test_vec_znx_big_add_normal,
    }
}

backend_test_suite! {
    mod window_fft64,
    backend = crate::FFT64Oracle,
    params = TestParams { size: 1 << 8, base2k: 12, n: 8 },
    tests = {
        test_vec_znx_window_ops => poulpy_hal::test_suite::window::test_vec_znx_window_ops,
        test_vec_znx_big_window_ops => poulpy_hal::test_suite::window::test_vec_znx_big_window_ops,
        test_vec_znx_window_normalize_ops => poulpy_hal::test_suite::window::test_vec_znx_window_normalize_ops,
        test_vec_znx_big_window_normalize => poulpy_hal::test_suite::window::test_vec_znx_big_window_normalize,
        test_vec_znx_window_rejected_by_ring_ops => poulpy_hal::test_suite::window::test_vec_znx_window_rejected_by_ring_ops,
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
    mod window_ntt4x30,
    backend = crate::NTT4x30Oracle,
    params = TestParams { size: 1 << 8, base2k: 12, n: 8 },
    tests = {
        test_vec_znx_window_ops => poulpy_hal::test_suite::window::test_vec_znx_window_ops,
        test_vec_znx_big_window_ops => poulpy_hal::test_suite::window::test_vec_znx_big_window_ops,
        test_vec_znx_window_normalize_ops => poulpy_hal::test_suite::window::test_vec_znx_window_normalize_ops,
        test_vec_znx_big_window_normalize => poulpy_hal::test_suite::window::test_vec_znx_big_window_normalize,
        test_vec_znx_window_rejected_by_ring_ops => poulpy_hal::test_suite::window::test_vec_znx_window_rejected_by_ring_ops,
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

#[cfg(feature = "enable-core")]
poulpy_core::core_backend_test_suite!(
    mod fft64,
    backend = crate::FFT64Oracle,
    // computes at the module degree, no sweep
    params = TestParams { size: 1<<8, base2k: 17, n: 1<<8 },
);

#[cfg(feature = "enable-core")]
poulpy_core::core_backend_test_suite!(
    mod ntt4x30,
    backend = crate::NTT4x30Oracle,
    // computes at the module degree, no sweep
    params = TestParams { size: 1<<8, base2k: 52, n: 1<<8 },
);

#[test]
fn test_vec_znx_rsh_assign_multi_limb_matches_rsh() {
    use poulpy_hal::api::{
        ScratchOwnedAlloc, ScratchOwnedBorrow, VecZnxFillUniformSourceAll, VecZnxRsh, VecZnxRshAssign, VecZnxRshTmpBytes,
    };
    use poulpy_hal::layouts::ScratchOwned;
    use poulpy_hal::source::Source;
    use poulpy_hal::test_suite::{download_vec_znx, vec_znx_backend_mut, vec_znx_backend_ref};

    let n = 8usize;
    let module: Module<NTT4x30Oracle> = Module::<NTT4x30Oracle>::new(n as u64);
    let mut scratch: ScratchOwned<NTT4x30Oracle> = ScratchOwned::alloc(module.vec_znx_rsh_tmp_bytes(4));
    let base2k = 52usize;
    let mut source = Source::new([3u8; 32]);

    // shifts spanning >= 2 limbs previously corrupted the in-place variant
    for size in [2usize, 3, 4] {
        for k in [60usize, 90, 105, 116] {
            if k / base2k + 1 > size {
                continue;
            }
            let mut a_be = module.vec_znx_alloc(n, 1, size);
            module.vec_znx_fill_uniform_source_all(base2k, size * base2k, &mut a_be, &mut source);
            let mut want_be = module.vec_znx_alloc(n, 1, size);
            module.vec_znx_rsh(
                base2k,
                k,
                &mut vec_znx_backend_mut::<NTT4x30Oracle>(&mut want_be),
                0,
                &vec_znx_backend_ref::<NTT4x30Oracle>(&a_be),
                0,
                &mut scratch.borrow(),
            );
            let mut got_be = a_be.clone();
            module.vec_znx_rsh_assign(
                base2k,
                k,
                &mut vec_znx_backend_mut::<NTT4x30Oracle>(&mut got_be),
                0,
                &mut scratch.borrow(),
            );
            assert_eq!(
                download_vec_znx::<NTT4x30Oracle>(&got_be),
                download_vec_znx::<NTT4x30Oracle>(&want_be),
                "vec_znx_rsh_assign mismatch for size={size} k={k}"
            );
        }
    }
}

/// Compile-time regression check: container equality is byte equality, so the
/// DFT/big-family containers implement `Eq` even when the logical word is
/// `f64` (a derived `Eq` used to demand `W: Eq` and silently vanish here).
#[allow(dead_code)]
fn assert_f64_word_containers_are_eq() {
    fn requires_eq<T: Eq>() {}
    requires_eq::<poulpy_hal::layouts::VecZnxDftOwned<crate::FFT64Oracle>>();
    requires_eq::<poulpy_hal::layouts::VecZnxBigOwned<crate::FFT64Oracle>>();
    requires_eq::<poulpy_hal::layouts::SvpPPolOwned<crate::FFT64Oracle>>();
    requires_eq::<poulpy_hal::layouts::VmpPMatOwned<crate::FFT64Oracle>>();
}

#[test]
fn test_hal_serialization_fft64_oracle() {
    poulpy_hal::test_suite::serialization::test_serialization(&Module::<FFT64Oracle>::new(1024));
}

#[cfg(feature = "enable-core")]
#[test]
fn test_glwe_public_key_rank1_golden() {
    use poulpy_core::test_suite::noise::encryption::glwe_public_key_rank1_digests;
    // Recorded on the single-key public key; the digest hashes its byte stream, the distribution then entry 0 as a GLWE.
    const FFT64: [u64; 3] = [2646170676813990930, 724828226321361831, 12849013967890643351];
    const NTT4X30: [u64; 3] = [15179721698775570956, 467002870803367667, 5727457524732813243];
    assert_eq!(
        (
            glwe_public_key_rank1_digests(&Module::<FFT64Oracle>::new(256), 17),
            glwe_public_key_rank1_digests(&Module::<NTT4x30Oracle>::new(256), 52),
        ),
        (FFT64, NTT4X30)
    );
}

#[cfg(feature = "enable-core")]
#[test]
fn test_core_serialization_fft64_oracle() {
    poulpy_core::test_suite::serialization::test_serialization(&Module::<FFT64Oracle>::new(64));
}

#[cfg(feature = "enable-core")]
#[test]
fn test_gglwe_product_dft_selected_fft64_oracle() {
    poulpy_core::test_suite::parity::test_gglwe_product_dft_selected(&Module::<FFT64Oracle>::new(64), 12);
}

#[cfg(feature = "enable-core")]
#[test]
fn test_gglwe_product_dft_selected_ntt4x30_oracle() {
    poulpy_core::test_suite::parity::test_gglwe_product_dft_selected(&Module::<NTT4x30Oracle>::new(64), 12);
}

// Cross-family parity: the NTT backend is exact, so at a radix small enough
// for FFT64 products to round exactly the two families must agree
// byte-for-byte. This catches a family-specific limb-window bug that a
// same-family parity suite cannot see.
#[cfg(feature = "enable-core")]
poulpy_core::core_parity_test_suite! {
    mod core_parity_cross_family,
    backend_ref = crate::NTT4x30Oracle,
    backend_test = crate::FFT64Oracle,
    // computes at the module degree, no sweep
    params = TestParams { size: 1<<8, base2k: 12, n: 1<<8 },
    tests = {
        glwe_keyswitch => poulpy_core::test_suite::parity::test_glwe_keyswitch_parity,
        glwe_keyswitch_assign => poulpy_core::test_suite::parity::test_glwe_keyswitch_assign_parity,
        gglwe_keyswitch => poulpy_core::test_suite::parity::test_gglwe_keyswitch_parity,
        glwe_automorphism => poulpy_core::test_suite::parity::test_glwe_automorphism_parity,
        glwe_external_product => poulpy_core::test_suite::parity::test_glwe_external_product_parity,
        glwe_copy_zero => poulpy_core::test_suite::parity::test_glwe_copy_zero_parity,
        glwe_shift => poulpy_core::test_suite::parity::test_glwe_shift_parity,
        glwe_multiplication => poulpy_core::test_suite::parity::test_glwe_multiplication_parity,
        ggsw_rotate => poulpy_core::test_suite::parity::test_ggsw_rotate_parity,
        gadget_external_product => poulpy_core::test_suite::parity::test_gadget_external_product_parity,
        gadget_conversion => poulpy_core::test_suite::parity::test_gadget_conversion_parity,
        lwe_conversion => poulpy_core::test_suite::parity::test_lwe_conversion_parity,
        gglwe_product_digits_strided => poulpy_core::test_suite::parity::test_gglwe_product_digits_strided_parity,
        polynomial_evaluation => poulpy_core::test_suite::parity::test_polynomial_evaluation_parity,
        trace_packing => poulpy_core::test_suite::parity::test_trace_packing_parity,
        tensor_relinearize_decrypt => poulpy_core::test_suite::parity::test_tensor_relinearize_decrypt_parity,
        linear_transformation => poulpy_core::test_suite::parity::test_linear_transformation_parity,
        sampling => poulpy_core::test_suite::sampling::test_sampling_contract,
        preparation => poulpy_core::test_suite::parity::test_preparation_contract,
        glwe_add => poulpy_core::test_suite::parity::test_glwe_add_parity,
        glwe_sub => poulpy_core::test_suite::parity::test_glwe_sub_parity,
        glwe_negate => poulpy_core::test_suite::parity::test_glwe_negate_parity,
        glwe_normalize => poulpy_core::test_suite::parity::test_glwe_normalize_parity,
        glwe_rotate => poulpy_core::test_suite::parity::test_glwe_rotate_parity,
        glwe_tensor => poulpy_core::test_suite::parity::test_glwe_tensor_parity,
    }
}

fn vec_znx_big_normalize_limb_bounds<BE>(module: &Module<BE>)
where
    BE: poulpy_hal::test_suite::TestBackend,
    BE::OwnedBuf: poulpy_hal::layouts::HostDataRef,
    Module<BE>: poulpy_hal::api::VecZnxBigAlloc<BE>
        + poulpy_hal::api::VecZnxBigFromSmall<BE>
        + poulpy_hal::api::VecZnxBigNormalize<BE>
        + poulpy_hal::api::VecZnxBigNormalizeTmpBytes
        + poulpy_hal::api::VecZnxFillUniformSourceAll<BE>
        + poulpy_hal::api::VecZnxAddAssign<BE>
        + poulpy_hal::api::VecZnxSubNegateAssign<BE>,
    poulpy_hal::layouts::ScratchOwned<BE>: poulpy_hal::api::ScratchOwnedAlloc<BE>,
{
    use poulpy_hal::{
        api::{
            ScratchOwnedAlloc, VecZnxAddAssign, VecZnxBigAlloc, VecZnxBigFromSmall, VecZnxBigNormalize,
            VecZnxBigNormalizeTmpBytes, VecZnxFillUniformSourceAll, VecZnxSubNegateAssign,
        },
        layouts::{ScratchOwned, VecZnx, VecZnxBigToBackendMut, VecZnxBigToBackendRef, VecZnxToBackendMut, ZnxView},
        source::Source,
        test_suite::{vec_znx_backend_mut, vec_znx_backend_ref},
    };
    let mut source = Source::new([2u8; 32]);
    let mut scratch: ScratchOwned<BE> = ScratchOwned::alloc(module.vec_znx_big_normalize_tmp_bytes());
    for a_base2k in 1..=51usize {
        for res_base2k in 1..=51usize {
            for offset in [-(a_base2k as i64), -3, -1, 0, 1, 3, a_base2k as i64] {
                for a_size in 1..=3usize {
                    for res_size in 1..=3usize {
                        // 2x - y (x at radix 62, y at radix 1) is uniform on [-2^62, 2^62), the radix-63 digits the fill cannot draw.
                        let mut a = module.vec_znx_alloc(module.n(), 1, a_size);
                        let mut x = module.vec_znx_alloc(module.n(), 1, a_size);
                        module.vec_znx_fill_uniform_source_all(1, a_size, &mut a, &mut source);
                        module.vec_znx_fill_uniform_source_all(62, a_size * 62, &mut x, &mut source);
                        module.vec_znx_sub_negate_assign(
                            &mut vec_znx_backend_mut::<BE>(&mut a),
                            0,
                            &vec_znx_backend_ref::<BE>(&x),
                            0,
                        );
                        module.vec_znx_add_assign(&mut vec_znx_backend_mut::<BE>(&mut a), 0, &vec_znx_backend_ref::<BE>(&x), 0);
                        let mut big = module.vec_znx_big_alloc(module.n(), 1, a_size);
                        module.vec_znx_big_from_small(&mut big.to_backend_mut(), 0, &vec_znx_backend_ref::<BE>(&a), 0);
                        let mut res = module.vec_znx_alloc(module.n(), 1, res_size);
                        module.vec_znx_big_normalize(
                            &mut <VecZnx<BE::OwnedBuf, i64> as VecZnxToBackendMut<BE>>::to_backend_mut(&mut res),
                            res_base2k,
                            res_size * res_base2k,
                            offset,
                            0,
                            &big.to_backend_ref(),
                            a_base2k,
                            0,
                            &mut scratch.arena(),
                        );
                        let bound: i64 = 1 << (res_base2k - 1);
                        for j in 0..res_size {
                            assert!(
                                res.at(0, j).iter().all(|x| (-bound..bound).contains(x)),
                                "a_base2k={a_base2k} res_base2k={res_base2k} offset={offset} a_size={a_size} res_size={res_size} limb={j}"
                            );
                        }
                    }
                }
            }
        }
    }
}

#[test]
fn test_vec_znx_big_normalize_limb_bounds_fft64_oracle() {
    vec_znx_big_normalize_limb_bounds(&Module::<FFT64Oracle>::new(8));
}

#[test]
fn test_vec_znx_big_normalize_limb_bounds_ntt4x30_oracle() {
    vec_znx_big_normalize_limb_bounds(&Module::<NTT4x30Oracle>::new(8));
}

#[cfg(feature = "enable-core")]
poulpy_core::core_parity_test_suite! {
    mod core_parity_same_family,
    backend_ref = crate::FFT64Oracle,
    backend_test = crate::NTT4x30Oracle,
    // computes at the module degree, no sweep
    params = TestParams { size: 1<<8, base2k: 12, n: 1<<8 },
    tests = {
        glwe_keyswitch => poulpy_core::test_suite::parity::test_glwe_keyswitch_parity,
        glwe_keyswitch_assign => poulpy_core::test_suite::parity::test_glwe_keyswitch_assign_parity,
        gglwe_keyswitch => poulpy_core::test_suite::parity::test_gglwe_keyswitch_parity,
        glwe_automorphism => poulpy_core::test_suite::parity::test_glwe_automorphism_parity,
        glwe_external_product => poulpy_core::test_suite::parity::test_glwe_external_product_parity,
        glwe_copy_zero => poulpy_core::test_suite::parity::test_glwe_copy_zero_parity,
        glwe_shift => poulpy_core::test_suite::parity::test_glwe_shift_parity,
        glwe_multiplication => poulpy_core::test_suite::parity::test_glwe_multiplication_parity,
        ggsw_rotate => poulpy_core::test_suite::parity::test_ggsw_rotate_parity,
        gadget_external_product => poulpy_core::test_suite::parity::test_gadget_external_product_parity,
        gadget_conversion => poulpy_core::test_suite::parity::test_gadget_conversion_parity,
        lwe_conversion => poulpy_core::test_suite::parity::test_lwe_conversion_parity,
        gglwe_product_digits_strided => poulpy_core::test_suite::parity::test_gglwe_product_digits_strided_parity,
        polynomial_evaluation => poulpy_core::test_suite::parity::test_polynomial_evaluation_parity,
        trace_packing => poulpy_core::test_suite::parity::test_trace_packing_parity,
        tensor_relinearize_decrypt => poulpy_core::test_suite::parity::test_tensor_relinearize_decrypt_parity,
        linear_transformation => poulpy_core::test_suite::parity::test_linear_transformation_parity,
        sampling => poulpy_core::test_suite::sampling::test_sampling_contract,
        preparation => poulpy_core::test_suite::parity::test_preparation_contract,
        glwe_add => poulpy_core::test_suite::parity::test_glwe_add_parity,
        glwe_sub => poulpy_core::test_suite::parity::test_glwe_sub_parity,
        glwe_negate => poulpy_core::test_suite::parity::test_glwe_negate_parity,
        glwe_normalize => poulpy_core::test_suite::parity::test_glwe_normalize_parity,
        glwe_rotate => poulpy_core::test_suite::parity::test_glwe_rotate_parity,
        glwe_tensor => poulpy_core::test_suite::parity::test_glwe_tensor_parity,
    }
}

#[cfg(feature = "enable-ckks")]
poulpy_ckks::conjugate_invariant_ckks_test_suite!(
    ckks_ci_fft64,
    crate::FFT64CIOracle,
    crate::FFT64Oracle,
    poulpy_ckks::test_suite::BASE19_PARAMS_F64
);

#[cfg(feature = "enable-ckks")]
poulpy_ckks::conjugate_invariant_ckks_test_suite!(
    ckks_ci_ntt4x30,
    crate::NTT4x30CIOracle,
    crate::NTT4x30Oracle,
    poulpy_ckks::test_suite::BASE52_PARAMS_F64
);

#[cfg(feature = "enable-bin-fhe")]
mod bin_fhe_tests {
    poulpy_bin_fhe::bin_fhe_reference_test_suite!(mod fft64, backend = crate::FFT64Oracle);
    poulpy_bin_fhe::bin_fhe_reference_test_suite!(mod ntt4x30, backend = crate::NTT4x30Oracle);
    poulpy_bin_fhe::bin_fhe_parity_test_suite!(mod parity, backend_ref = crate::FFT64Oracle, backend_test = crate::NTT4x30Oracle);
}
