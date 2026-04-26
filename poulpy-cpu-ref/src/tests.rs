use poulpy_hal::{
    api::ModuleNew,
    layouts::Module,
    test_suite::convolution::{test_convolution, test_convolution_by_const, test_convolution_pairwise},
};

use crate::{FFT64Ref, NTT120Ref};

#[test]
fn test_convolution_by_const_fft64_ref() {
    let module: Module<FFT64Ref> = Module::<FFT64Ref>::new(8);
    test_convolution_by_const(&module, 17);
}

#[test]
fn test_convolution_fft64_ref() {
    let module: Module<FFT64Ref> = Module::<FFT64Ref>::new(8);
    test_convolution(&module, 17);
}

#[test]
fn test_convolution_pairwise_fft64_ref() {
    let module: Module<FFT64Ref> = Module::<FFT64Ref>::new(8);
    test_convolution_pairwise(&module, 17);
}

#[test]
fn test_convolution_by_const_ntt120_ref() {
    let module: Module<NTT120Ref> = Module::<NTT120Ref>::new(8);
    test_convolution_by_const(&module, 50);
}

#[test]
fn test_convolution_ntt120_ref() {
    let module: Module<NTT120Ref> = Module::<NTT120Ref>::new(8);
    test_convolution(&module, 50);
}

#[test]
fn test_convolution_pairwise_ntt120_ref() {
    let module: Module<NTT120Ref> = Module::<NTT120Ref>::new(8);
    test_convolution_pairwise(&module, 50);
}

use poulpy_hal::{backend_test_suite, cross_backend_test_suite};

cross_backend_test_suite! {
    mod vec_znx,
    backend_ref =  crate::FFT64Ref,
    backend_test = crate::NTT120Ref,
    params = TestParams { size: 1<<8, base2k: 12 },
    tests = {
        test_vec_znx_add_into => poulpy_hal::test_suite::vec_znx::test_vec_znx_add_into,
        test_vec_znx_add_assign => poulpy_hal::test_suite::vec_znx::test_vec_znx_add_assign,
        test_vec_znx_add_scalar_into => poulpy_hal::test_suite::vec_znx::test_vec_znx_add_scalar_into,
        test_vec_znx_add_scalar_assign => poulpy_hal::test_suite::vec_znx::test_vec_znx_add_scalar_assign,
        test_vec_znx_sub => poulpy_hal::test_suite::vec_znx::test_vec_znx_sub,
        test_vec_znx_sub_assign => poulpy_hal::test_suite::vec_znx::test_vec_znx_sub_assign,
        test_vec_znx_sub_negate_assign => poulpy_hal::test_suite::vec_znx::test_vec_znx_sub_negate_assign,
        test_vec_znx_sub_scalar => poulpy_hal::test_suite::vec_znx::test_vec_znx_sub_scalar,
        test_vec_znx_sub_scalar_assign => poulpy_hal::test_suite::vec_znx::test_vec_znx_sub_scalar_assign,
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
        test_vec_znx_split_ring => poulpy_hal::test_suite::vec_znx::test_vec_znx_split_ring,
        test_vec_znx_copy => poulpy_hal::test_suite::vec_znx::test_vec_znx_copy,
    }
}
cross_backend_test_suite! {
    mod svp,
    backend_ref =  crate::FFT64Ref,
    backend_test = crate::NTT120Ref,
    params = TestParams { size: 1<<8, base2k: 12 },
    tests = {
        test_svp_apply_dft_to_dft => poulpy_hal::test_suite::svp::test_svp_apply_dft_to_dft,
        test_svp_apply_dft_to_dft_assign => poulpy_hal::test_suite::svp::test_svp_apply_dft_to_dft_assign,
    }
}
cross_backend_test_suite! {
    mod vec_znx_big,
    backend_ref =  crate::FFT64Ref,
    backend_test = crate::NTT120Ref,
    params = TestParams { size: 1<<8, base2k: 12 },
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
        test_vec_znx_big_normalize_fused => poulpy_hal::test_suite::vec_znx_big::test_vec_znx_big_normalize_fused,
        test_vec_znx_big_sub_negate_assign => poulpy_hal::test_suite::vec_znx_big::test_vec_znx_big_sub_negate_assign,
        test_vec_znx_big_sub_small_a => poulpy_hal::test_suite::vec_znx_big::test_vec_znx_big_sub_small_a,
        test_vec_znx_big_sub_small_a_assign => poulpy_hal::test_suite::vec_znx_big::test_vec_znx_big_sub_small_a_assign,
        test_vec_znx_big_sub_small_b => poulpy_hal::test_suite::vec_znx_big::test_vec_znx_big_sub_small_b,
        test_vec_znx_big_sub_small_b_assign => poulpy_hal::test_suite::vec_znx_big::test_vec_znx_big_sub_small_b_assign,
    }
}
cross_backend_test_suite! {
    mod vec_znx_dft,
    backend_ref =  crate::FFT64Ref,
    backend_test = crate::NTT120Ref,
    params = TestParams { size: 1<<8, base2k: 12 },
    tests = {
        test_vec_znx_dft_add_into => poulpy_hal::test_suite::vec_znx_dft::test_vec_znx_dft_add_into,
        test_vec_znx_dft_add_assign => poulpy_hal::test_suite::vec_znx_dft::test_vec_znx_dft_add_assign,
        test_vec_znx_dft_sub => poulpy_hal::test_suite::vec_znx_dft::test_vec_znx_dft_sub,
        test_vec_znx_dft_sub_assign => poulpy_hal::test_suite::vec_znx_dft::test_vec_znx_dft_sub_assign,
        test_vec_znx_dft_sub_negate_assign => poulpy_hal::test_suite::vec_znx_dft::test_vec_znx_dft_sub_negate_assign,
        test_vec_znx_dft_copy => poulpy_hal::test_suite::vec_znx_dft::test_vec_znx_copy,
        test_vec_znx_idft_apply => poulpy_hal::test_suite::vec_znx_dft::test_vec_znx_idft_apply,
        test_vec_znx_idft_apply_consume => poulpy_hal::test_suite::vec_znx_dft::test_vec_znx_idft_apply_consume,
        test_vec_znx_idft_apply_tmpa => poulpy_hal::test_suite::vec_znx_dft::test_vec_znx_idft_apply_tmpa,
    }
}
cross_backend_test_suite! {
    mod vmp,
    backend_ref =  crate::FFT64Ref,
    backend_test = crate::NTT120Ref,
    params = TestParams { size: 1<<8, base2k: 12 },
    tests = {
        test_vmp_apply_dft_to_dft => poulpy_hal::test_suite::vmp::test_vmp_apply_dft_to_dft,
    }
}

backend_test_suite! {
    mod sampling,
    backend = crate::NTT120Ref,
    params = TestParams { size: 1<<12, base2k: 12 },
    tests = {
        test_vec_znx_fill_uniform => poulpy_hal::test_suite::vec_znx::test_vec_znx_fill_uniform,
        test_vec_znx_fill_normal => poulpy_hal::test_suite::vec_znx::test_vec_znx_fill_normal,
        test_vec_znx_add_normal => poulpy_hal::test_suite::vec_znx::test_vec_znx_add_normal,
    }
}

poulpy_core::core_backend_test_suite!(
    mod fft64,
    backend = crate::FFT64Ref,
    params = TestParams { size: 1<<8, base2k: 17 },
);

poulpy_core::core_backend_test_suite!(
    mod ntt120,
    backend = crate::NTT120Ref,
    params = TestParams { size: 1<<8, base2k: 52 },
);
