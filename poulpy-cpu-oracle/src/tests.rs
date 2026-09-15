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
mod derived_scratch;

#[test]
fn test_convolution_by_const_fft64_ref() {
    let module: Module<FFT64Oracle> = Module::<FFT64Oracle>::new(8);
    test_convolution_by_const(&module, 17);
    test_convolution_by_const_add(&module, 17);
}

#[test]
fn test_convolution_fft64_ref() {
    let module: Module<FFT64Oracle> = Module::<FFT64Oracle>::new(8);
    test_convolution(&module, 17);
}

#[test]
fn test_convolution_pairwise_fft64_ref() {
    let module: Module<FFT64Oracle> = Module::<FFT64Oracle>::new(8);
    test_convolution_pairwise(&module, 17);
}

#[test]
fn test_convolution_add_fft64_ref() {
    let module: Module<FFT64Oracle> = Module::<FFT64Oracle>::new(8);
    test_convolution_add(&module, 17);
}

#[test]
fn test_convolution_sum_fft64_ref() {
    let module: Module<FFT64Oracle> = Module::<FFT64Oracle>::new(8);
    test_convolution_sum(&module, 17);
}

#[test]
fn test_convolution_by_const_ntt4x30_ref() {
    let module: Module<NTT4x30Oracle> = Module::<NTT4x30Oracle>::new(8);
    test_convolution_by_const(&module, 50);
    test_convolution_by_const_add(&module, 50);
}

#[test]
fn test_convolution_ntt4x30_ref() {
    let module: Module<NTT4x30Oracle> = Module::<NTT4x30Oracle>::new(8);
    test_convolution(&module, 50);
}

#[test]
fn test_convolution_pairwise_ntt4x30_ref() {
    let module: Module<NTT4x30Oracle> = Module::<NTT4x30Oracle>::new(8);
    test_convolution_pairwise(&module, 50);
}

#[test]
fn test_convolution_add_ntt4x30_ref() {
    let module: Module<NTT4x30Oracle> = Module::<NTT4x30Oracle>::new(8);
    test_convolution_add(&module, 50);
}

#[test]
fn test_convolution_sum_ntt4x30_ref() {
    let module: Module<NTT4x30Oracle> = Module::<NTT4x30Oracle>::new(8);
    test_convolution_sum(&module, 50);
}

use poulpy_hal::{backend_test_suite, cross_backend_test_suite};

cross_backend_test_suite! {
    mod vec_znx,
    backend_ref =  crate::FFT64Oracle,
    backend_test = crate::NTT4x30Oracle,
    params = TestParams { size: 1<<8, base2k: 12 },
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
    params = TestParams { size: 1<<8, base2k: 12 },
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
    params = TestParams { size: 1<<8, base2k: 12 },
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
    params = TestParams { size: 1<<8, base2k: 12 },
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
    params = TestParams { size: 1<<8, base2k: 12 },
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
    params = TestParams { size: 1<<8, base2k: 12 },
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
    params = TestParams { size: 1<<8, base2k: 12 },
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
    params = TestParams { size: 1<<8, base2k: 12 },
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
    params = TestParams { size: 1<<12, base2k: 12 },
    tests = {
        test_vec_znx_fill_uniform => poulpy_hal::test_suite::vec_znx::test_vec_znx_fill_uniform,
    }
}

// Gated on `enable-core`, like `core_impl`, which implements the noise seam for both reference backends.
#[cfg(feature = "enable-core")]
backend_test_suite! {
    mod sampling_core,
    backend = crate::NTT4x30Oracle,
    params = TestParams { size: 1<<12, base2k: 12 },
    tests = {
        test_vec_znx_add_normal => poulpy_core::test_suite::sampling::test_vec_znx_add_normal,
        test_vec_znx_big_add_normal => poulpy_core::test_suite::sampling::test_vec_znx_big_add_normal,
    }
}

#[cfg(feature = "enable-core")]
backend_test_suite! {
    mod sampling_core_fft64,
    backend = crate::FFT64Oracle,
    params = TestParams { size: 1<<12, base2k: 17 },
    tests = {
        test_vec_znx_add_normal => poulpy_core::test_suite::sampling::test_vec_znx_add_normal,
        test_vec_znx_big_add_normal => poulpy_core::test_suite::sampling::test_vec_znx_big_add_normal,
    }
}

backend_test_suite! {
    mod window_fft64,
    backend = crate::FFT64Oracle,
    params = TestParams { size: 1 << 8, base2k: 12 },
    tests = {
        test_vec_znx_window_ops => poulpy_hal::test_suite::window::test_vec_znx_window_ops,
        test_vec_znx_big_window_ops => poulpy_hal::test_suite::window::test_vec_znx_big_window_ops,
        test_vec_znx_window_normalize_ops => poulpy_hal::test_suite::window::test_vec_znx_window_normalize_ops,
        test_vec_znx_big_window_normalize => poulpy_hal::test_suite::window::test_vec_znx_big_window_normalize,
        test_vec_znx_window_rejected_by_ring_ops => poulpy_hal::test_suite::window::test_vec_znx_window_rejected_by_ring_ops,
        test_vec_znx_dft_step_zero_rejected => poulpy_hal::test_suite::vec_znx_dft::test_vec_znx_dft_step_zero_rejected,
        test_convolution_prepare_shape_rejected => poulpy_hal::test_suite::convolution::test_convolution_prepare_shape_rejected,
        test_convolution_by_const_degree_rejected => poulpy_hal::test_suite::convolution::test_convolution_by_const_degree_rejected,
    }
}

backend_test_suite! {
    mod window_ntt4x30,
    backend = crate::NTT4x30Oracle,
    params = TestParams { size: 1 << 8, base2k: 12 },
    tests = {
        test_vec_znx_window_ops => poulpy_hal::test_suite::window::test_vec_znx_window_ops,
        test_vec_znx_big_window_ops => poulpy_hal::test_suite::window::test_vec_znx_big_window_ops,
        test_vec_znx_window_normalize_ops => poulpy_hal::test_suite::window::test_vec_znx_window_normalize_ops,
        test_vec_znx_big_window_normalize => poulpy_hal::test_suite::window::test_vec_znx_big_window_normalize,
        test_vec_znx_window_rejected_by_ring_ops => poulpy_hal::test_suite::window::test_vec_znx_window_rejected_by_ring_ops,
        test_vec_znx_dft_step_zero_rejected => poulpy_hal::test_suite::vec_znx_dft::test_vec_znx_dft_step_zero_rejected,
        test_convolution_prepare_shape_rejected => poulpy_hal::test_suite::convolution::test_convolution_prepare_shape_rejected,
        test_convolution_by_const_degree_rejected => poulpy_hal::test_suite::convolution::test_convolution_by_const_degree_rejected,
    }
}

#[cfg(feature = "enable-core")]
poulpy_core::core_backend_test_suite!(
    mod fft64,
    backend = crate::FFT64Oracle,
    params = TestParams { size: 1<<8, base2k: 17 },
);

#[cfg(feature = "enable-core")]
poulpy_core::core_backend_test_suite!(
    mod ntt4x30,
    backend = crate::NTT4x30Oracle,
    params = TestParams { size: 1<<8, base2k: 52 },
);

#[test]
fn test_vec_znx_rsh_assign_multi_limb_matches_rsh() {
    use poulpy_hal::api::{ScratchOwnedAlloc, ScratchOwnedBorrow, VecZnxRsh, VecZnxRshAssign, VecZnxRshTmpBytes};
    use poulpy_hal::layouts::{FillUniform, HostBytesBackend, ScratchOwned, VecZnx};
    use poulpy_hal::source::Source;
    use poulpy_hal::test_suite::{download_vec_znx, upload_vec_znx, vec_znx_backend_mut, vec_znx_backend_ref};

    let n = 8usize;
    let module: Module<NTT4x30Oracle> = Module::<NTT4x30Oracle>::new(n as u64);
    let module_host: Module<HostBytesBackend> = Module::<HostBytesBackend>::new(n as u64);
    let mut scratch: ScratchOwned<NTT4x30Oracle> = ScratchOwned::alloc(module.vec_znx_rsh_tmp_bytes(4));
    let base2k = 52usize;
    let mut source = Source::new([3u8; 32]);

    // shifts spanning >= 2 limbs previously corrupted the in-place variant
    for size in [2usize, 3, 4] {
        for k in [60usize, 90, 105, 116] {
            if k / base2k + 1 > size {
                continue;
            }
            let mut a: VecZnx<Vec<u8>, i64> = module_host.vec_znx_alloc(1, size);
            a.fill_uniform(base2k, &mut source);
            let a_be = upload_vec_znx::<NTT4x30Oracle>(&a);
            let mut want_be = upload_vec_znx::<NTT4x30Oracle>(&module_host.vec_znx_alloc(1, size));
            module.vec_znx_rsh(
                base2k,
                k,
                &mut vec_znx_backend_mut::<NTT4x30Oracle>(&mut want_be),
                0,
                &vec_znx_backend_ref::<NTT4x30Oracle>(&a_be),
                0,
                &mut scratch.borrow(),
            );
            let mut got_be = upload_vec_znx::<NTT4x30Oracle>(&a);
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

#[cfg(feature = "enable-core")]
poulpy_bin_fhe::bin_fhe_backend_test_suite!(mod bin_fhe_fft64, backend = crate::FFT64Oracle);

#[cfg(feature = "enable-core")]
#[test]
fn test_gglwe_product_dft_selected_fft64_ref() {
    poulpy_core::test_suite::parity::test_gglwe_product_dft_selected(&Module::<FFT64Oracle>::new(64), 12);
}

#[cfg(feature = "enable-core")]
#[test]
fn test_gglwe_product_dft_selected_ntt4x30_ref() {
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
    params = TestParams { size: 1<<8, base2k: 12 },
    tests = {
        glwe_keyswitch => poulpy_core::test_suite::parity::test_glwe_keyswitch_parity,
        glwe_keyswitch_assign => poulpy_core::test_suite::parity::test_glwe_keyswitch_assign_parity,
        gglwe_keyswitch => poulpy_core::test_suite::parity::test_gglwe_keyswitch_parity,
        glwe_automorphism => poulpy_core::test_suite::parity::test_glwe_automorphism_parity,
        glwe_external_product => poulpy_core::test_suite::parity::test_glwe_external_product_parity,
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
        + poulpy_hal::api::VecZnxBigNormalizeTmpBytes,
    poulpy_hal::layouts::ScratchOwned<BE>: poulpy_hal::api::ScratchOwnedAlloc<BE>,
{
    use poulpy_hal::{
        api::{ScratchOwnedAlloc, VecZnxBigAlloc, VecZnxBigFromSmall, VecZnxBigNormalize, VecZnxBigNormalizeTmpBytes},
        layouts::{
            FillUniform, HostBytesBackend, ScratchOwned, VecZnx, VecZnxBigToBackendMut, VecZnxBigToBackendRef,
            VecZnxToBackendMut, VecZnxToBackendRef, ZnxView,
        },
        source::Source,
        test_suite::upload_vec_znx,
    };
    let module_host: Module<HostBytesBackend> = Module::<HostBytesBackend>::new(module.n() as u64);
    let mut source = Source::new([2u8; 32]);
    let mut scratch: ScratchOwned<BE> = ScratchOwned::alloc(module.vec_znx_big_normalize_tmp_bytes());
    for a_base2k in 1..=51usize {
        for res_base2k in 1..=51usize {
            for offset in [-(a_base2k as i64), -3, -1, 0, 1, 3, a_base2k as i64] {
                for a_size in 1..=3usize {
                    for res_size in 1..=3usize {
                        let mut a = module_host.vec_znx_alloc(1, a_size);
                        a.fill_uniform(63, &mut source);
                        let uploaded = upload_vec_znx::<BE>(&a);
                        let mut big = module.vec_znx_big_alloc(1, a_size);
                        module.vec_znx_big_from_small(
                            &mut big.to_backend_mut(),
                            0,
                            &<VecZnx<BE::OwnedBuf, i64> as VecZnxToBackendRef<BE>>::to_backend_ref(&uploaded),
                            0,
                        );
                        let mut res = module.vec_znx_alloc(1, res_size);
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
fn test_vec_znx_big_normalize_limb_bounds_fft64_ref() {
    vec_znx_big_normalize_limb_bounds(&Module::<FFT64Oracle>::new(8));
}

#[test]
fn test_vec_znx_big_normalize_limb_bounds_ntt4x30_ref() {
    vec_znx_big_normalize_limb_bounds(&Module::<NTT4x30Oracle>::new(8));
}
