mod ckks_tests;

poulpy_core::core_parity_test_suite! {
    mod core_parity_fft64,
    backend_ref = poulpy_cpu_ref::FFT64Ref,
    backend_test = crate::FFT64Avx512,
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

poulpy_core::core_parity_test_suite! {
    mod core_parity_ntt4x30,
    backend_ref = poulpy_cpu_ref::NTT4x30Ref,
    backend_test = crate::NTT4x30Avx512,
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

// Exercise the rank-one specialization at degrees above the general suites.
poulpy_core::core_parity_test_suite! {
    mod core_parity_ntt4x30_fused,
    backend_ref = poulpy_cpu_ref::NTT4x30Ref,
    backend_test = crate::NTT4x30Avx512,
    params = TestParams { size: 1<<15, base2k: 52 },
    tests = {
        glwe_tensor => poulpy_core::test_suite::parity::test_glwe_tensor_parity,
    }
}

poulpy_core::core_parity_test_suite! {
    mod core_parity_ntt4x30_fused_n16,
    backend_ref = poulpy_cpu_ref::NTT4x30Ref,
    backend_test = crate::NTT4x30Avx512,
    params = TestParams { size: 1<<16, base2k: 52 },
    tests = {
        glwe_tensor => poulpy_core::test_suite::parity::test_glwe_tensor_parity,
    }
}

#[cfg(feature = "enable-ifma")]
poulpy_core::core_parity_test_suite! {
    mod core_parity_ntt3x42_ifma_fused,
    backend_ref = poulpy_cpu_ref::NTT4x30Ref,
    backend_test = crate::NTT3x42Ifma,
    params = TestParams { size: 1<<15, base2k: 52 },
    tests = {
        glwe_tensor => poulpy_core::test_suite::parity::test_glwe_tensor_parity,
    }
}

#[cfg(feature = "enable-ifma")]
poulpy_core::core_parity_test_suite! {
    mod core_parity_ntt3x42_ifma_fused_n16,
    backend_ref = poulpy_cpu_ref::NTT4x30Ref,
    backend_test = crate::NTT3x42Ifma,
    params = TestParams { size: 1<<16, base2k: 52 },
    tests = {
        glwe_tensor => poulpy_core::test_suite::parity::test_glwe_tensor_parity,
    }
}

#[cfg(feature = "enable-ifma")]
poulpy_core::core_parity_test_suite! {
    mod core_parity_ntt3x42_ifma,
    backend_ref = poulpy_cpu_ref::NTT4x30Ref,
    backend_test = crate::NTT3x42Ifma,
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

#[cfg(all(feature = "enable-avx512f", target_arch = "x86_64", target_feature = "avx512f"))]
poulpy_bin_fhe::bin_fhe_backend_test_suite!(mod bin_fhe_fft64, backend = crate::FFT64Avx512);
