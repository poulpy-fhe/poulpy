//! Conformance of the portable Rayon variants against their serial bases.

use poulpy_hal::{
    backend_test_suite, cross_backend_test_suite,
    layouts::Module,
    test_suite::convolution::{
        test_convolution, test_convolution_add, test_convolution_by_const_add, test_convolution_pairwise, test_convolution_sum,
    },
};

use super::FFT64PortableRayon;

// The size crosses the parallel-work floors of the overrides that have them.
cross_backend_test_suite! {
    mod vec_znx_dft_fft64,
    backend_ref = poulpy_cpu_portable::FFT64Portable,
    backend_test = crate::FFT64PortableRayon,
    params = TestParams { size: 1<<14, base2k: 12, n: 8 },
    tests = {
        test_vec_znx_dft_copy => poulpy_hal::test_suite::vec_znx_dft::test_vec_znx_dft_copy,
        test_vec_znx_dft_automorphism_add => poulpy_hal::test_suite::vec_znx_dft::test_vec_znx_dft_automorphism_add,
        test_vec_znx_idft_normalize_consume => poulpy_hal::test_suite::vec_znx_dft::test_vec_znx_idft_normalize_consume,
    }
}

cross_backend_test_suite! {
    mod vmp_fft64,
    backend_ref = poulpy_cpu_portable::FFT64Portable,
    backend_test = crate::FFT64PortableRayon,
    params = TestParams { size: 1<<8, base2k: 12, n: 8 },
    tests = {
        test_vmp_apply_dft => poulpy_hal::test_suite::vmp::test_vmp_apply_dft,
        test_vmp_apply_dft_to_dft => poulpy_hal::test_suite::vmp::test_vmp_apply_dft_to_dft,
        test_vmp_extract_selected_rows => poulpy_hal::test_suite::vmp::test_vmp_extract_selected_rows,
        test_vmp_apply_dft_to_dft_add => poulpy_hal::test_suite::vmp::test_vmp_apply_dft_to_dft_add,
        test_vmp_zero => poulpy_hal::test_suite::vmp::test_vmp_zero,
    }
}

backend_test_suite! {
    mod window_fft64,
    backend = crate::FFT64PortableRayon,
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
    }
}

#[test]
fn test_convolution_fft64() {
    let module = Module::<FFT64PortableRayon>::new(1 << 8);
    test_convolution(&module, module.n(), 12);
    test_convolution_add(&module, module.n(), 12);
    test_convolution_pairwise(&module, module.n(), 12);
    test_convolution_sum(&module, module.n(), 12);
    test_convolution_by_const_add(&module, module.n(), 12);
}

poulpy_cpu_portable::conjugate_invariant_test_suite!(
    ci_fft64,
    crate::FFT64CIPortableRayon,
    crate::FFT64PortableRayon,
    reference = poulpy_cpu_portable::FFT64CIPortable
);

use super::NTT4x30PortableRayon;

// The size crosses the parallel-work floors of the overrides that have them.
cross_backend_test_suite! {
    mod vec_znx_dft_ntt4x30,
    backend_ref = poulpy_cpu_portable::NTT4x30Portable,
    backend_test = crate::NTT4x30PortableRayon,
    params = TestParams { size: 1<<14, base2k: 50, n: 8 },
    tests = {
        test_vec_znx_dft_copy => poulpy_hal::test_suite::vec_znx_dft::test_vec_znx_dft_copy,
        test_vec_znx_dft_add => poulpy_hal::test_suite::vec_znx_dft::test_vec_znx_dft_add,
        test_vec_znx_dft_add_assign => poulpy_hal::test_suite::vec_znx_dft::test_vec_znx_dft_add_assign,
        test_vec_znx_dft_sub => poulpy_hal::test_suite::vec_znx_dft::test_vec_znx_dft_sub,
        test_vec_znx_dft_sub_assign => poulpy_hal::test_suite::vec_znx_dft::test_vec_znx_dft_sub_assign,
        test_vec_znx_dft_sub_negate_assign => poulpy_hal::test_suite::vec_znx_dft::test_vec_znx_dft_sub_negate_assign,
        test_vec_znx_idft_apply => poulpy_hal::test_suite::vec_znx_dft::test_vec_znx_idft_apply,
        test_vec_znx_idft_apply_tmpa => poulpy_hal::test_suite::vec_znx_dft::test_vec_znx_idft_apply_tmpa,
        test_vec_znx_dft_apply => poulpy_hal::test_suite::vec_znx_dft::test_vec_znx_dft_apply,
        test_vec_znx_dft_automorphism_add => poulpy_hal::test_suite::vec_znx_dft::test_vec_znx_dft_automorphism_add,
        test_vec_znx_idft_normalize_consume => poulpy_hal::test_suite::vec_znx_dft::test_vec_znx_idft_normalize_consume,
    }
}

cross_backend_test_suite! {
    mod vmp_ntt4x30,
    backend_ref = poulpy_cpu_portable::NTT4x30Portable,
    backend_test = crate::NTT4x30PortableRayon,
    params = TestParams { size: 1<<8, base2k: 50, n: 8 },
    tests = {
        test_vmp_apply_dft => poulpy_hal::test_suite::vmp::test_vmp_apply_dft,
        test_vmp_apply_dft_to_dft => poulpy_hal::test_suite::vmp::test_vmp_apply_dft_to_dft,
        test_vmp_extract_selected_rows => poulpy_hal::test_suite::vmp::test_vmp_extract_selected_rows,
        test_vmp_apply_dft_to_dft_add => poulpy_hal::test_suite::vmp::test_vmp_apply_dft_to_dft_add,
        test_vmp_zero => poulpy_hal::test_suite::vmp::test_vmp_zero,
    }
}

cross_backend_test_suite! {
    mod svp_ntt4x30,
    backend_ref = poulpy_cpu_portable::NTT4x30Portable,
    backend_test = crate::NTT4x30PortableRayon,
    params = TestParams { size: 1<<8, base2k: 50, n: 8 },
    tests = {
        test_svp_apply_dft => poulpy_hal::test_suite::svp::test_svp_apply_dft,
        test_svp_apply_dft_to_dft => poulpy_hal::test_suite::svp::test_svp_apply_dft_to_dft,
        test_svp_apply_dft_to_dft_assign => poulpy_hal::test_suite::svp::test_svp_apply_dft_to_dft_assign,
    }
}

backend_test_suite! {
    mod window_ntt4x30,
    backend = crate::NTT4x30PortableRayon,
    params = TestParams { size: 1 << 8, base2k: 50, n: 8 },
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
    }
}

#[test]
fn test_convolution_ntt4x30() {
    let module = Module::<NTT4x30PortableRayon>::new(1 << 8);
    for n in [module.n(), module.n() >> 1, module.n() >> 2] {
        test_convolution(&module, n, 50);
        test_convolution_add(&module, n, 50);
        test_convolution_pairwise(&module, n, 50);
        test_convolution_sum(&module, n, 50);
    }
    test_convolution_by_const_add(&module, module.n(), 50);
}

poulpy_cpu_portable::conjugate_invariant_test_suite!(
    ci_ntt4x30,
    crate::NTT4x30CIPortableRayon,
    crate::NTT4x30PortableRayon,
    reference = poulpy_cpu_portable::NTT4x30CIPortable
);

#[cfg(feature = "enable-core")]
mod core {
    use poulpy_hal::layouts::Module;

    poulpy_core::core_parity_test_suite! {
        mod parity_ntt4x30,
        backend_ref = poulpy_cpu_portable::NTT4x30Portable,
        backend_test = crate::NTT4x30PortableRayon,
        // computes at the module degree, no sweep
        params = TestParams { size: 1<<8, base2k: 12, n: 1<<8 },
        tests = {
            glwe_keyswitch => poulpy_core::test_suite::parity::test_glwe_keyswitch_parity,
            glwe_keyswitch_assign => poulpy_core::test_suite::parity::test_glwe_keyswitch_assign_parity,
            gglwe_keyswitch => poulpy_core::test_suite::parity::test_gglwe_keyswitch_parity,
            glwe_automorphism => poulpy_core::test_suite::parity::test_glwe_automorphism_parity,
            glwe_external_product => poulpy_core::test_suite::parity::test_glwe_external_product_parity,
            glwe_multiplication => poulpy_core::test_suite::parity::test_glwe_multiplication_parity,
            gadget_external_product => poulpy_core::test_suite::parity::test_gadget_external_product_parity,
            gglwe_product_digits_strided => poulpy_core::test_suite::parity::test_gglwe_product_digits_strided_parity,
            tensor_relinearize_decrypt => poulpy_core::test_suite::parity::test_tensor_relinearize_decrypt_parity,
            linear_transformation => poulpy_core::test_suite::parity::test_linear_transformation_parity,
            glwe_tensor => poulpy_core::test_suite::parity::test_glwe_tensor_parity,
        }
    }

    poulpy_core::core_parity_test_suite! {
        mod parity_fft64,
        backend_ref = poulpy_cpu_portable::FFT64Portable,
        backend_test = crate::FFT64PortableRayon,
        // computes at the module degree, no sweep
        params = TestParams { size: 1<<8, base2k: 12, n: 1<<8 },
        tests = {
            glwe_keyswitch => poulpy_core::test_suite::parity::test_glwe_keyswitch_parity,
            glwe_automorphism => poulpy_core::test_suite::parity::test_glwe_automorphism_parity,
            glwe_external_product => poulpy_core::test_suite::parity::test_glwe_external_product_parity,
            glwe_multiplication => poulpy_core::test_suite::parity::test_glwe_multiplication_parity,
            gglwe_product_digits_strided => poulpy_core::test_suite::parity::test_gglwe_product_digits_strided_parity,
            glwe_tensor => poulpy_core::test_suite::parity::test_glwe_tensor_parity,
        }
    }

    /// The fused interleaved-digit product against the core reference body, on the same backend.
    #[test]
    fn test_gglwe_product_digits_strided_bit_identical() {
        poulpy_core::test_suite::parity::test_gglwe_product_digits_strided(&Module::<crate::NTT4x30PortableRayon>::new(64), 50);
    }

    poulpy_cpu_portable::conjugate_invariant_core_test_suite!(
        ci_core_fft64,
        crate::FFT64CIPortableRayon,
        crate::FFT64PortableRayon,
        reference = poulpy_cpu_portable::FFT64CIPortable
    );

    poulpy_cpu_portable::conjugate_invariant_core_test_suite!(
        ci_core_ntt4x30,
        crate::NTT4x30CIPortableRayon,
        crate::NTT4x30PortableRayon,
        reference = poulpy_cpu_portable::NTT4x30CIPortable
    );
}

#[cfg(feature = "enable-ckks")]
mod ckks {
    poulpy_ckks::ckks_parity_test_suite! {
        mod parity_ntt4x30_f64,
        backend_ref = poulpy_cpu_portable::NTT4x30Portable,
        backend_test = crate::NTT4x30PortableRayon,
        scalar = f64,
        params = poulpy_ckks::test_suite::CKKSTestParams { n: 64, hw: 48, base2k: 52, ..poulpy_ckks::test_suite::BASE52_PARAMS_F64 },
        tests = {
            arithmetic => poulpy_ckks::test_suite::parity::test_arithmetic_parity,
            imag => poulpy_ckks::test_suite::parity::test_imag_parity,
            multiplication => poulpy_ckks::test_suite::parity::test_multiplication_parity,
            rotate => poulpy_ckks::test_suite::parity::test_rotate_parity,
            conjugate => poulpy_ckks::test_suite::parity::test_conjugate_parity,
            plaintext => poulpy_ckks::test_suite::parity::test_plaintext_parity,
            encoding => poulpy_ckks::test_suite::parity::test_encoding_parity,
            slot_encoding => poulpy_ckks::test_suite::parity::test_slot_encoding_parity,
            paco_encoding => poulpy_ckks::test_suite::parity::test_paco_encoding_parity,
            ship_encoding => poulpy_ckks::test_suite::parity::test_ship_encoding_parity,
            dft => poulpy_ckks::test_suite::parity::test_dft_parity,
            real_polynomial => poulpy_ckks::test_suite::parity::test_real_polynomial_parity,
            polynomial_eval_mod => poulpy_ckks::test_suite::parity::test_polynomial_eval_mod_parity,
            encapsulated_mod_up => poulpy_ckks::test_suite::parity::test_encapsulated_mod_up_parity,
            bootstrapping => poulpy_ckks::test_suite::parity::test_bootstrapping_parity,
            fold => poulpy_ckks::test_suite::parity::test_fold_parity,
        }
    }

    poulpy_ckks::ckks_parity_test_suite! {
        mod parity_fft64_f64,
        backend_ref = poulpy_cpu_portable::FFT64Portable,
        backend_test = crate::FFT64PortableRayon,
        scalar = f64,
        params = poulpy_ckks::test_suite::CKKSTestParams { n: 64, hw: 48, base2k: 12, ..poulpy_ckks::test_suite::BASE19_PARAMS_F64 },
        tests = {
            arithmetic => poulpy_ckks::test_suite::parity::test_arithmetic_parity,
            multiplication => poulpy_ckks::test_suite::parity::test_multiplication_parity,
            rotate => poulpy_ckks::test_suite::parity::test_rotate_parity,
            encapsulated_mod_up => poulpy_ckks::test_suite::parity::test_encapsulated_mod_up_parity,
            bootstrapping => poulpy_ckks::test_suite::parity::test_bootstrapping_parity,
        }
    }

    poulpy_ckks::ckks_encryption_parity_test_suite! {
        mod parity_ntt4x30_encryption,
        backend_ref = poulpy_cpu_portable::NTT4x30Portable,
        backend_test = crate::NTT4x30PortableRayon,
        params = poulpy_ckks::test_suite::CKKSTestParams { n: 64, hw: 48, base2k: 12, ..poulpy_ckks::test_suite::BASE19_PARAMS_F64 },
    }
}
