use poulpy_hal::{backend_test_suite, cross_backend_test_suite};
use poulpy_hal::{
    layouts::Module,
    test_suite::convolution::{
        test_convolution, test_convolution_accumulate, test_convolution_by_const, test_convolution_pairwise,
    },
};

use crate::NTT4x30Avx512;

cross_backend_test_suite! {
    mod vec_znx,
    backend_ref =  poulpy_cpu_ref::NTT4x30Ref,
    backend_test = crate::NTT4x30Avx512,
    params = TestParams { size: 1<<8, base2k: 50 },
    tests = {
        test_vec_znx_add_into => poulpy_hal::test_suite::vec_znx::test_vec_znx_add_into_backend_matches_reference,
        test_vec_znx_add_assign => poulpy_hal::test_suite::vec_znx::test_vec_znx_add_assign,
        test_vec_znx_extract_coeff_backend => poulpy_hal::test_suite::vec_znx::test_vec_znx_extract_coeff_backend,
        test_vec_znx_normalize_coeff_backend => poulpy_hal::test_suite::vec_znx::test_vec_znx_normalize_coeff_backend,
        test_vec_znx_normalize_coeff_assign_backend => poulpy_hal::test_suite::vec_znx::test_vec_znx_normalize_coeff_assign_backend,
        test_vec_znx_lsh_coeff_backend => poulpy_hal::test_suite::vec_znx::test_vec_znx_lsh_coeff_backend,
        test_vec_znx_lsh_add_coeff_into_backend => poulpy_hal::test_suite::vec_znx::test_vec_znx_lsh_add_coeff_into_backend,
        test_vec_znx_lsh_add_coeff_to_coeff_backend => poulpy_hal::test_suite::vec_znx::test_vec_znx_lsh_add_coeff_to_coeff_backend,
        test_vec_znx_lsh_sub_coeff_to_coeff_backend => poulpy_hal::test_suite::vec_znx::test_vec_znx_lsh_sub_coeff_to_coeff_backend,
        test_vec_znx_rsh_coeff_backend => poulpy_hal::test_suite::vec_znx::test_vec_znx_rsh_coeff_backend,
        test_vec_znx_rsh_add_coeff_into_backend => poulpy_hal::test_suite::vec_znx::test_vec_znx_rsh_add_coeff_into_backend,
        test_vec_znx_rsh_sub_coeff_into_backend => poulpy_hal::test_suite::vec_znx::test_vec_znx_rsh_sub_coeff_into_backend,
        test_vec_znx_add_scalar_into => poulpy_hal::test_suite::vec_znx::test_vec_znx_add_scalar_into,
        test_vec_znx_add_scalar_assign => poulpy_hal::test_suite::vec_znx::test_vec_znx_add_scalar_assign,
        test_vec_znx_sub => poulpy_hal::test_suite::vec_znx::test_vec_znx_sub,
        test_vec_znx_sub_assign => poulpy_hal::test_suite::vec_znx::test_vec_znx_sub_assign,
        test_vec_znx_sub_negate_assign => poulpy_hal::test_suite::vec_znx::test_vec_znx_sub_negate_assign,
        test_vec_znx_sub_scalar => poulpy_hal::test_suite::vec_znx::test_vec_znx_sub_scalar,
        test_vec_znx_sub_scalar_assign => poulpy_hal::test_suite::vec_znx::test_vec_znx_sub_scalar_assign,
        test_vec_znx_canonicalize_to_k_assign => poulpy_hal::test_suite::vec_znx::test_vec_znx_canonicalize_to_k_assign,
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
        test_vec_znx_merge_rings => poulpy_hal::test_suite::vec_znx::test_vec_znx_merge_rings,
        test_vec_znx_switch_ring => poulpy_hal::test_suite::vec_znx::test_vec_znx_switch_ring,
        test_vec_znx_split_ring => poulpy_hal::test_suite::vec_znx::test_vec_znx_split_ring,
        test_vec_znx_copy => poulpy_hal::test_suite::vec_znx::test_vec_znx_copy,
    }
}

cross_backend_test_suite! {
    mod svp,
    backend_ref =  poulpy_cpu_ref::NTT4x30Ref,
    backend_test = crate::NTT4x30Avx512,
    params = TestParams { size: 1<<8, base2k: 50 },
    tests = {
        test_svp_apply_dft_to_dft => poulpy_hal::test_suite::svp::test_svp_apply_dft_to_dft,
        test_svp_apply_dft_to_dft_assign => poulpy_hal::test_suite::svp::test_svp_apply_dft_to_dft_assign,
    }
}

cross_backend_test_suite! {
    mod vec_znx_big,
    backend_ref =  poulpy_cpu_ref::NTT4x30Ref,
    backend_test = crate::NTT4x30Avx512,
    params = TestParams { size: 1<<8, base2k: 50 },
    tests = {
        test_vec_znx_big_add_into => poulpy_hal::test_suite::vec_znx_big::test_vec_znx_big_add_into,
        test_vec_znx_big_add_assign => poulpy_hal::test_suite::vec_znx_big::test_vec_znx_big_add_assign,
        test_vec_znx_big_add_small_into => poulpy_hal::test_suite::vec_znx_big::test_vec_znx_big_add_small_into,
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
    }
}

cross_backend_test_suite! {
    mod vec_znx_dft,
    backend_ref =  poulpy_cpu_ref::NTT4x30Ref,
    backend_test = crate::NTT4x30Avx512,
    params = TestParams { size: 1<<8, base2k: 50 },
    tests = {
        test_vec_znx_dft_add_into => poulpy_hal::test_suite::vec_znx_dft::test_vec_znx_dft_add_into,
        test_vec_znx_dft_add_assign => poulpy_hal::test_suite::vec_znx_dft::test_vec_znx_dft_add_assign,
        test_vec_znx_dft_sub => poulpy_hal::test_suite::vec_znx_dft::test_vec_znx_dft_sub,
        test_vec_znx_dft_sub_assign => poulpy_hal::test_suite::vec_znx_dft::test_vec_znx_dft_sub_assign,
        test_vec_znx_dft_sub_negate_assign => poulpy_hal::test_suite::vec_znx_dft::test_vec_znx_dft_sub_negate_assign,
        test_vec_znx_idft_apply => poulpy_hal::test_suite::vec_znx_dft::test_vec_znx_idft_apply,
        test_vec_znx_idft_apply_consume => poulpy_hal::test_suite::vec_znx_dft::test_vec_znx_idft_apply_alloc,
        test_vec_znx_idft_apply_tmpa => poulpy_hal::test_suite::vec_znx_dft::test_vec_znx_idft_apply_tmpa,
        test_vec_znx_dft_automorphism => poulpy_hal::test_suite::vec_znx_dft::test_vec_znx_dft_automorphism,
    }
}

cross_backend_test_suite! {
    mod vmp,
    backend_ref =  poulpy_cpu_ref::NTT4x30Ref,
    backend_test = crate::NTT4x30Avx512,
    params = TestParams { size: 1<<8, base2k: 50 },
    tests = {
        test_vmp_apply_dft => poulpy_hal::test_suite::vmp::test_vmp_apply_dft,
        test_vmp_apply_dft_to_dft => poulpy_hal::test_suite::vmp::test_vmp_apply_dft_to_dft,
        test_vmp_apply_dft_to_dft_accumulate => poulpy_hal::test_suite::vmp::test_vmp_apply_dft_to_dft_accumulate,
    }
}

backend_test_suite! {
    mod sampling,
    backend = crate::NTT4x30Avx512,
    params = TestParams { size: 1<<12, base2k: 50 },
    tests = {
        test_vec_znx_fill_uniform => poulpy_hal::test_suite::vec_znx::test_vec_znx_fill_uniform,
        test_vec_znx_fill_normal => poulpy_hal::test_suite::vec_znx::test_vec_znx_fill_normal,
        test_vec_znx_add_normal => poulpy_hal::test_suite::vec_znx::test_vec_znx_add_normal,
    }
}

backend_test_suite! {
    mod lwe_matrix,
    backend = crate::NTT4x30Avx512,
    params = TestParams { size: 1<<8, base2k: 50 },
    tests = {
        glwe_expand_lwe_matrix_decrypt => poulpy_core::test_suite::noise::test_glwe_expand_lwe_matrix_decrypt,
    }
}

// NTT CHANGE_MODE_N boundary tests.
// CHANGE_MODE_N = 1024: for n <= 1024 the AVX NTT runs fully by-block;
// for n > 1024 it first completes upper levels by-level then switches to
// by-block for the remaining levels. These suites ensure both modes are
// exercised and agree with the reference backend.

// n = 1024: last size that uses by-block only.
cross_backend_test_suite! {
    mod ntt_n1024,
    backend_ref =  poulpy_cpu_ref::NTT4x30Ref,
    backend_test = crate::NTT4x30Avx512,
    params = TestParams { size: 1<<10, base2k: 50 },
    tests = {
        test_vec_znx_idft_apply => poulpy_hal::test_suite::vec_znx_dft::test_vec_znx_idft_apply,
        test_vec_znx_idft_apply_consume => poulpy_hal::test_suite::vec_znx_dft::test_vec_znx_idft_apply_alloc,
        test_svp_apply_dft_to_dft => poulpy_hal::test_suite::svp::test_svp_apply_dft_to_dft,
    }
}

// n = 8192: large size exercising many by-level stages.
cross_backend_test_suite! {
    mod ntt_n8192,
    backend_ref =  poulpy_cpu_ref::NTT4x30Ref,
    backend_test = crate::NTT4x30Avx512,
    params = TestParams { size: 1<<13, base2k: 50 },
    tests = {
        test_vec_znx_idft_apply => poulpy_hal::test_suite::vec_znx_dft::test_vec_znx_idft_apply,
        test_vec_znx_idft_apply_consume => poulpy_hal::test_suite::vec_znx_dft::test_vec_znx_idft_apply_alloc,
        test_svp_apply_dft_to_dft => poulpy_hal::test_suite::svp::test_svp_apply_dft_to_dft,
    }
}

// n = 16384: largest size before the AVX NTT switches to by-level mode only.
cross_backend_test_suite! {
    mod ntt_n16384,
    backend_ref =  poulpy_cpu_ref::NTT4x30Ref,
    backend_test = crate::NTT4x30Avx512,
    params = TestParams { size: 1<<14, base2k: 50 },
    tests = {
        test_vec_znx_idft_apply => poulpy_hal::test_suite::vec_znx_dft::test_vec_znx_idft_apply,
        test_vec_znx_idft_apply_consume => poulpy_hal::test_suite::vec_znx_dft::test_vec_znx_idft_apply_alloc,
        test_svp_apply_dft_to_dft => poulpy_hal::test_suite::svp::test_svp_apply_dft_to_dft,
    }
}

// n = 32768: largest size (still by-level only) included in the test suite.
cross_backend_test_suite! {
    mod ntt_n32768,
    backend_ref =  poulpy_cpu_ref::NTT4x30Ref,
    backend_test = crate::NTT4x30Avx512,
    params = TestParams { size: 1<<15, base2k: 50 },
    tests = {
        test_vec_znx_idft_apply => poulpy_hal::test_suite::vec_znx_dft::test_vec_znx_idft_apply,
        test_vec_znx_idft_apply_consume => poulpy_hal::test_suite::vec_znx_dft::test_vec_znx_idft_apply_alloc,
        test_svp_apply_dft_to_dft => poulpy_hal::test_suite::svp::test_svp_apply_dft_to_dft,
    }
}

#[test]
fn test_convolution_direct() {
    let module = Module::<NTT4x30Avx512>::new(1 << 8);
    test_convolution(&module, 50);
    test_convolution_by_const(&module, 50);
    test_convolution_pairwise(&module, 50);
    test_convolution_accumulate(&module, 50);
}

#[test]
fn test_gglwe_product_digits_strided_bit_identical() {
    poulpy_core::test_suite::parity::test_gglwe_product_digits_strided(&Module::<NTT4x30Avx512>::new(64), 50);
}

cross_backend_test_suite! {
    mod word_compat,
    backend_ref =  poulpy_cpu_ref::NTT4x30Ref,
    backend_test = crate::NTT4x30Avx512,
    params = TestParams { size: 1<<8, base2k: 50 },
    tests = {
        test_word_compat_dft_bytes => poulpy_hal::test_suite::word_compat::test_word_compat_dft_bytes,
        test_word_compat_svp_prepare_bytes => poulpy_hal::test_suite::word_compat::test_word_compat_svp_prepare_bytes,
        test_word_compat_dft_cross_idft => poulpy_hal::test_suite::word_compat::test_word_compat_dft_cross_idft,
    }
}
