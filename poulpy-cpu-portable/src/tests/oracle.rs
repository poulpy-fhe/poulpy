use poulpy_hal::cross_backend_test_suite;

mod fft64 {
    use super::*;
    cross_backend_test_suite! {
        mod vec_znx,
        backend_ref =  poulpy_cpu_oracle::FFT64Oracle,
        backend_test = crate::FFT64Portable,
        params = TestParams { size: 1<<8, base2k: 12 },
        tests = {
            test_vec_znx_add_matches_reference => poulpy_hal::test_suite::vec_znx::test_vec_znx_add_matches_reference,
            test_vec_znx_add_assign => poulpy_hal::test_suite::vec_znx::test_vec_znx_add_assign,
            test_vec_znx_add_scalar_assign => poulpy_hal::test_suite::vec_znx::test_vec_znx_add_scalar_assign,
            test_vec_znx_sub => poulpy_hal::test_suite::vec_znx::test_vec_znx_sub,
            test_vec_znx_sub_assign => poulpy_hal::test_suite::vec_znx::test_vec_znx_sub_assign,
            test_vec_znx_sub_negate_assign => poulpy_hal::test_suite::vec_znx::test_vec_znx_sub_negate_assign,
            test_vec_znx_rsh => poulpy_hal::test_suite::vec_znx::test_vec_znx_rsh,
            test_vec_znx_rsh_assign => poulpy_hal::test_suite::vec_znx::test_vec_znx_rsh_assign,
            test_vec_znx_lsh => poulpy_hal::test_suite::vec_znx::test_vec_znx_lsh,
            test_vec_znx_lsh_assign => poulpy_hal::test_suite::vec_znx::test_vec_znx_lsh_assign,
            test_vec_znx_negate => poulpy_hal::test_suite::vec_znx::test_vec_znx_negate,
            test_vec_znx_negate_assign => poulpy_hal::test_suite::vec_znx::test_vec_znx_negate_assign,
            test_vec_znx_rotate => poulpy_hal::test_suite::vec_znx::test_vec_znx_rotate,
            test_vec_znx_rotate_assign => poulpy_hal::test_suite::vec_znx::test_vec_znx_rotate_assign,
            test_vec_znx_automorphism => poulpy_hal::test_suite::vec_znx::test_vec_znx_automorphism,
            test_vec_znx_automorphism_assign => poulpy_hal::test_suite::vec_znx::test_vec_znx_automorphism_assign,
            test_vec_znx_mul_xp_minus_one => poulpy_hal::test_suite::vec_znx::test_vec_znx_mul_xp_minus_one,
            test_vec_znx_mul_xp_minus_one_assign => poulpy_hal::test_suite::vec_znx::test_vec_znx_mul_xp_minus_one_assign,
            test_vec_znx_normalize => poulpy_hal::test_suite::vec_znx::test_vec_znx_normalize,
            test_vec_znx_normalize_assign => poulpy_hal::test_suite::vec_znx::test_vec_znx_normalize_assign,
            test_vec_znx_switch_ring => poulpy_hal::test_suite::vec_znx::test_vec_znx_switch_ring,
            test_vec_znx_copy => poulpy_hal::test_suite::vec_znx::test_vec_znx_copy,
        }
    }
    cross_backend_test_suite! {
        mod svp,
        backend_ref =  poulpy_cpu_oracle::FFT64Oracle,
        backend_test = crate::FFT64Portable,
        params = TestParams { size: 1<<8, base2k: 12 },
        tests = {
            test_svp_apply_dft => poulpy_hal::test_suite::svp::test_svp_apply_dft,
            test_svp_apply_dft_to_dft => poulpy_hal::test_suite::svp::test_svp_apply_dft_to_dft,
            test_svp_apply_dft_to_dft_assign => poulpy_hal::test_suite::svp::test_svp_apply_dft_to_dft_assign,
        }
    }
    cross_backend_test_suite! {
        mod vec_znx_big,
        backend_ref =  poulpy_cpu_oracle::FFT64Oracle,
        backend_test = crate::FFT64Portable,
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
        backend_ref =  poulpy_cpu_oracle::FFT64Oracle,
        backend_test = crate::FFT64Portable,
        params = TestParams { size: 1<<8, base2k: 12 },
        tests = {
            test_vec_znx_dft_add => poulpy_hal::test_suite::vec_znx_dft::test_vec_znx_dft_add,
            test_vec_znx_dft_add_assign => poulpy_hal::test_suite::vec_znx_dft::test_vec_znx_dft_add_assign,
            test_vec_znx_dft_sub => poulpy_hal::test_suite::vec_znx_dft::test_vec_znx_dft_sub,
            test_vec_znx_dft_sub_assign => poulpy_hal::test_suite::vec_znx_dft::test_vec_znx_dft_sub_assign,
            test_vec_znx_dft_sub_negate_assign => poulpy_hal::test_suite::vec_znx_dft::test_vec_znx_dft_sub_negate_assign,
            test_vec_znx_idft_apply => poulpy_hal::test_suite::vec_znx_dft::test_vec_znx_idft_apply,
            test_vec_znx_idft_apply_consume => poulpy_hal::test_suite::vec_znx_dft::test_vec_znx_idft_apply_alloc,
            test_vec_znx_idft_apply_tmpa => poulpy_hal::test_suite::vec_znx_dft::test_vec_znx_idft_apply_tmpa,
            test_vec_znx_dft_apply => poulpy_hal::test_suite::vec_znx_dft::test_vec_znx_dft_apply,
            test_vec_znx_dft_zero => poulpy_hal::test_suite::vec_znx_dft::test_vec_znx_dft_zero,
            test_vec_znx_dft_automorphism => poulpy_hal::test_suite::vec_znx_dft::test_vec_znx_dft_automorphism,
            test_vec_znx_dft_automorphism_add => poulpy_hal::test_suite::vec_znx_dft::test_vec_znx_dft_automorphism_add,
            test_vec_znx_idft_normalize_consume => poulpy_hal::test_suite::vec_znx_dft::test_vec_znx_idft_normalize_consume,
        }
    }
    cross_backend_test_suite! {
        mod vmp,
        backend_ref =  poulpy_cpu_oracle::FFT64Oracle,
        backend_test = crate::FFT64Portable,
        params = TestParams { size: 1<<8, base2k: 12 },
        tests = {
            test_vmp_apply_dft => poulpy_hal::test_suite::vmp::test_vmp_apply_dft,
            test_vmp_apply_dft_to_dft => poulpy_hal::test_suite::vmp::test_vmp_apply_dft_to_dft,
            test_vmp_extract_selected_rows => poulpy_hal::test_suite::vmp::test_vmp_extract_selected_rows,
            test_vmp_apply_dft_to_dft_add => poulpy_hal::test_suite::vmp::test_vmp_apply_dft_to_dft_add,
            test_vmp_zero => poulpy_hal::test_suite::vmp::test_vmp_zero,
        }
    }
}
mod ntt4x30 {
    use super::*;
    cross_backend_test_suite! {
        mod vec_znx,
        backend_ref =  poulpy_cpu_oracle::NTT4x30Oracle,
        backend_test = crate::NTT4x30Portable,
        params = TestParams { size: 1<<8, base2k: 50 },
        tests = {
            test_vec_znx_add_matches_reference => poulpy_hal::test_suite::vec_znx::test_vec_znx_add_matches_reference,
            test_vec_znx_add_assign => poulpy_hal::test_suite::vec_znx::test_vec_znx_add_assign,
            test_vec_znx_add_scalar_assign => poulpy_hal::test_suite::vec_znx::test_vec_znx_add_scalar_assign,
            test_vec_znx_sub => poulpy_hal::test_suite::vec_znx::test_vec_znx_sub,
            test_vec_znx_sub_assign => poulpy_hal::test_suite::vec_znx::test_vec_znx_sub_assign,
            test_vec_znx_sub_negate_assign => poulpy_hal::test_suite::vec_znx::test_vec_znx_sub_negate_assign,
            test_vec_znx_rsh => poulpy_hal::test_suite::vec_znx::test_vec_znx_rsh,
            test_vec_znx_rsh_assign => poulpy_hal::test_suite::vec_znx::test_vec_znx_rsh_assign,
            test_vec_znx_lsh => poulpy_hal::test_suite::vec_znx::test_vec_znx_lsh,
            test_vec_znx_lsh_assign => poulpy_hal::test_suite::vec_znx::test_vec_znx_lsh_assign,
            test_vec_znx_negate => poulpy_hal::test_suite::vec_znx::test_vec_znx_negate,
            test_vec_znx_negate_assign => poulpy_hal::test_suite::vec_znx::test_vec_znx_negate_assign,
            test_vec_znx_rotate => poulpy_hal::test_suite::vec_znx::test_vec_znx_rotate,
            test_vec_znx_rotate_assign => poulpy_hal::test_suite::vec_znx::test_vec_znx_rotate_assign,
            test_vec_znx_automorphism => poulpy_hal::test_suite::vec_znx::test_vec_znx_automorphism,
            test_vec_znx_automorphism_assign => poulpy_hal::test_suite::vec_znx::test_vec_znx_automorphism_assign,
            test_vec_znx_mul_xp_minus_one => poulpy_hal::test_suite::vec_znx::test_vec_znx_mul_xp_minus_one,
            test_vec_znx_mul_xp_minus_one_assign => poulpy_hal::test_suite::vec_znx::test_vec_znx_mul_xp_minus_one_assign,
            test_vec_znx_normalize => poulpy_hal::test_suite::vec_znx::test_vec_znx_normalize,
            test_vec_znx_normalize_assign => poulpy_hal::test_suite::vec_znx::test_vec_znx_normalize_assign,
            test_vec_znx_switch_ring => poulpy_hal::test_suite::vec_znx::test_vec_znx_switch_ring,
            test_vec_znx_copy => poulpy_hal::test_suite::vec_znx::test_vec_znx_copy,
        }
    }
    cross_backend_test_suite! {
        mod svp,
        backend_ref =  poulpy_cpu_oracle::NTT4x30Oracle,
        backend_test = crate::NTT4x30Portable,
        params = TestParams { size: 1<<8, base2k: 50 },
        tests = {
            test_svp_apply_dft => poulpy_hal::test_suite::svp::test_svp_apply_dft,
            test_svp_apply_dft_to_dft => poulpy_hal::test_suite::svp::test_svp_apply_dft_to_dft,
            test_svp_apply_dft_to_dft_assign => poulpy_hal::test_suite::svp::test_svp_apply_dft_to_dft_assign,
        }
    }
    cross_backend_test_suite! {
        mod vec_znx_big,
        backend_ref =  poulpy_cpu_oracle::NTT4x30Oracle,
        backend_test = crate::NTT4x30Portable,
        params = TestParams { size: 1<<8, base2k: 50 },
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
        backend_ref =  poulpy_cpu_oracle::NTT4x30Oracle,
        backend_test = crate::NTT4x30Portable,
        params = TestParams { size: 1<<8, base2k: 50 },
        tests = {
            test_vec_znx_dft_add => poulpy_hal::test_suite::vec_znx_dft::test_vec_znx_dft_add,
            test_vec_znx_dft_add_assign => poulpy_hal::test_suite::vec_znx_dft::test_vec_znx_dft_add_assign,
            test_vec_znx_dft_sub => poulpy_hal::test_suite::vec_znx_dft::test_vec_znx_dft_sub,
            test_vec_znx_dft_sub_assign => poulpy_hal::test_suite::vec_znx_dft::test_vec_znx_dft_sub_assign,
            test_vec_znx_dft_sub_negate_assign => poulpy_hal::test_suite::vec_znx_dft::test_vec_znx_dft_sub_negate_assign,
            test_vec_znx_idft_apply => poulpy_hal::test_suite::vec_znx_dft::test_vec_znx_idft_apply,
            test_vec_znx_idft_apply_consume => poulpy_hal::test_suite::vec_znx_dft::test_vec_znx_idft_apply_alloc,
            test_vec_znx_idft_apply_tmpa => poulpy_hal::test_suite::vec_znx_dft::test_vec_znx_idft_apply_tmpa,
            test_vec_znx_dft_apply => poulpy_hal::test_suite::vec_znx_dft::test_vec_znx_dft_apply,
            test_vec_znx_dft_zero => poulpy_hal::test_suite::vec_znx_dft::test_vec_znx_dft_zero,
            test_vec_znx_dft_automorphism => poulpy_hal::test_suite::vec_znx_dft::test_vec_znx_dft_automorphism,
            test_vec_znx_dft_automorphism_add => poulpy_hal::test_suite::vec_znx_dft::test_vec_znx_dft_automorphism_add,
            test_vec_znx_idft_normalize_consume => poulpy_hal::test_suite::vec_znx_dft::test_vec_znx_idft_normalize_consume,
        }
    }
    cross_backend_test_suite! {
        mod vmp,
        backend_ref =  poulpy_cpu_oracle::NTT4x30Oracle,
        backend_test = crate::NTT4x30Portable,
        params = TestParams { size: 1<<8, base2k: 50 },
        tests = {
            test_vmp_apply_dft => poulpy_hal::test_suite::vmp::test_vmp_apply_dft,
            test_vmp_apply_dft_to_dft => poulpy_hal::test_suite::vmp::test_vmp_apply_dft_to_dft,
            test_vmp_extract_selected_rows => poulpy_hal::test_suite::vmp::test_vmp_extract_selected_rows,
            test_vmp_apply_dft_to_dft_add => poulpy_hal::test_suite::vmp::test_vmp_apply_dft_to_dft_add,
            test_vmp_zero => poulpy_hal::test_suite::vmp::test_vmp_zero,
        }
    }
    cross_backend_test_suite! {
        mod ntt_n1024,
        backend_ref =  poulpy_cpu_oracle::NTT4x30Oracle,
        backend_test = crate::NTT4x30Portable,
        params = TestParams { size: 1<<10, base2k: 50 },
        tests = {
            test_vec_znx_idft_apply => poulpy_hal::test_suite::vec_znx_dft::test_vec_znx_idft_apply,
            test_vec_znx_idft_apply_consume => poulpy_hal::test_suite::vec_znx_dft::test_vec_znx_idft_apply_alloc,
            test_svp_apply_dft_to_dft => poulpy_hal::test_suite::svp::test_svp_apply_dft_to_dft,
        }
    }
    cross_backend_test_suite! {
        mod ntt_n8192,
        backend_ref =  poulpy_cpu_oracle::NTT4x30Oracle,
        backend_test = crate::NTT4x30Portable,
        params = TestParams { size: 1<<13, base2k: 50 },
        tests = {
            test_vec_znx_idft_apply => poulpy_hal::test_suite::vec_znx_dft::test_vec_znx_idft_apply,
            test_vec_znx_idft_apply_consume => poulpy_hal::test_suite::vec_znx_dft::test_vec_znx_idft_apply_alloc,
            test_svp_apply_dft_to_dft => poulpy_hal::test_suite::svp::test_svp_apply_dft_to_dft,
        }
    }
    cross_backend_test_suite! {
        mod ntt_n16384,
        backend_ref =  poulpy_cpu_oracle::NTT4x30Oracle,
        backend_test = crate::NTT4x30Portable,
        params = TestParams { size: 1<<14, base2k: 50 },
        tests = {
            test_vec_znx_idft_apply => poulpy_hal::test_suite::vec_znx_dft::test_vec_znx_idft_apply,
            test_vec_znx_idft_apply_consume => poulpy_hal::test_suite::vec_znx_dft::test_vec_znx_idft_apply_alloc,
            test_svp_apply_dft_to_dft => poulpy_hal::test_suite::svp::test_svp_apply_dft_to_dft,
        }
    }
    cross_backend_test_suite! {
        mod ntt_n32768,
        backend_ref =  poulpy_cpu_oracle::NTT4x30Oracle,
        backend_test = crate::NTT4x30Portable,
        params = TestParams { size: 1<<15, base2k: 50 },
        tests = {
            test_vec_znx_idft_apply => poulpy_hal::test_suite::vec_znx_dft::test_vec_znx_idft_apply,
            test_vec_znx_idft_apply_consume => poulpy_hal::test_suite::vec_znx_dft::test_vec_znx_idft_apply_alloc,
            test_svp_apply_dft_to_dft => poulpy_hal::test_suite::svp::test_svp_apply_dft_to_dft,
        }
    }
}
#[cfg(feature = "enable-core")]
poulpy_core::core_parity_test_suite! {
    mod core_parity_fft64,
    backend_ref = poulpy_cpu_oracle::FFT64Oracle,
    backend_test = crate::FFT64Portable,
    params = TestParams { size: 1<<8, base2k: 17 },
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
#[cfg(feature = "enable-core")]
poulpy_core::core_parity_test_suite! {
    mod core_parity_ntt4x30,
    backend_ref = poulpy_cpu_oracle::NTT4x30Oracle,
    backend_test = crate::NTT4x30Portable,
    params = TestParams { size: 1<<8, base2k: 52 },
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
