#[cfg(feature = "enable-ckks")]
mod ckks_tests;

#[test]
fn max_base2k() {
    use poulpy_hal::layouts::{Backend, Module};

    // Cover odd and even log2(n), including ceiling rounding.
    const fn limits<B: Backend>() -> [usize; 2] {
        [Module::<B>::max_base2k(8), Module::<B>::max_base2k(1 << 16)]
    }

    assert_eq!(const { limits::<crate::FFT64Neon>() }, [25, 19]);
    assert_eq!(const { limits::<crate::NTT4x30Neon>() }, [59, 52]);
    #[cfg(feature = "enable-rayon")]
    {
        assert_eq!(const { limits::<crate::FFT64NeonRayon>() }, [25, 19]);
        assert_eq!(const { limits::<crate::NTT4x30NeonRayon>() }, [59, 52]);
    }
}

poulpy_core::core_parity_test_suite! {
    mod core_parity_fft64,
    backend_ref = poulpy_cpu_ref::FFT64Ref,
    backend_test = crate::FFT64Neon,
    // computes at the module degree, no sweep
    params = TestParams { size: 1<<8, base2k: 17, n: 1<<8 },
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

poulpy_bin_fhe::bin_fhe_backend_test_suite!(mod bin_fhe_fft64, backend = crate::FFT64Neon);

#[cfg(feature = "enable-rayon")]
poulpy_bin_fhe::bin_fhe_backend_test_suite!(mod bin_fhe_fft64_rayon, backend = crate::FFT64NeonRayon);

#[cfg(feature = "enable-rayon")]
poulpy_bin_fhe::bin_fhe_backend_test_suite!(mod bin_fhe_ntt4x30_rayon, backend = crate::NTT4x30NeonRayon);

#[cfg(feature = "enable-rayon")]
poulpy_core::core_parity_test_suite! {
    mod core_parity_fft64_rayon,
    // The serial backend is validated above; this edge checks the Rayon implementation.
    backend_ref = crate::FFT64Neon,
    backend_test = crate::FFT64NeonRayon,
    // computes at the module degree, no sweep
    params = TestParams { size: 1<<8, base2k: 17, n: 1<<8 },
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

poulpy_core::core_parity_test_suite! {
    mod core_parity_ntt4x30,
    backend_ref = poulpy_cpu_ref::NTT4x30Ref,
    backend_test = crate::NTT4x30Neon,
    // computes at the module degree, no sweep
    params = TestParams { size: 1<<8, base2k: 52, n: 1<<8 },
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

#[cfg(feature = "enable-rayon")]
poulpy_core::core_parity_test_suite! {
    mod core_parity_ntt4x30_rayon,
    // The serial backend is validated above; this edge checks the Rayon implementation.
    backend_ref = crate::NTT4x30Neon,
    backend_test = crate::NTT4x30NeonRayon,
    // computes at the module degree, no sweep
    params = TestParams { size: 1<<8, base2k: 52, n: 1<<8 },
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

#[cfg(feature = "enable-rayon")]
poulpy_bin_fhe::bin_fhe_parity_test_suite!(
    mod bin_fhe_parity_fft64_rayon,
    backend_ref = crate::FFT64Neon,
    backend_test = crate::FFT64NeonRayon,
);

#[cfg(feature = "enable-rayon")]
poulpy_bin_fhe::bin_fhe_parity_test_suite!(
    mod bin_fhe_parity_ntt4x30_rayon,
    backend_ref = crate::NTT4x30Neon,
    backend_test = crate::NTT4x30NeonRayon,
);

/// On-demand thread-count diagnostic; see `docs/performance.md`.
#[cfg(feature = "enable-rayon")]
mod tuning {
    use poulpy_cpu_rayon::tuning::{Mode, ProbeShape, default_thread_sweep, thread_scaling};

    const LOG_N: usize = 15;
    const SIZE: usize = 12;
    /// GLWE rank of the workload; the probes take `rank + 1` columns.
    const RANK: usize = 1;
    const MODE: Mode = Mode::Fast;

    #[test]
    #[ignore = "diagnostic: run on the machine you deploy on"]
    fn thread_scaling_report() {
        let sweep = default_thread_sweep();
        thread_scaling::<crate::FFT64NeonRayon>(ProbeShape::square(1 << LOG_N, SIZE, RANK + 1), &sweep, MODE)
            .print("FFT64NeonRayon");
        thread_scaling::<crate::NTT4x30NeonRayon>(ProbeShape::square(1 << LOG_N, SIZE, RANK + 1), &sweep, MODE)
            .print("NTT4x30NeonRayon");
    }
}
poulpy_core::core_encryption_parity_test_suite!(
    mod core_encryption_fft64neon,
    backend_ref = poulpy_cpu_ref::test_suite::ControlledSamplingFFT64Ref,
    backend_test = crate::FFT64Neon
);
poulpy_core::core_encryption_parity_test_suite!(
    mod core_encryption_ntt4x30neon,
    backend_ref = poulpy_cpu_ref::test_suite::ControlledSamplingFFT64Ref,
    backend_test = crate::NTT4x30Neon
);

#[cfg(feature = "enable-rayon")]
poulpy_core::core_encryption_parity_test_suite!(
    mod core_encryption_fft64neonrayon,
    backend_ref = crate::FFT64Neon,
    backend_test = crate::FFT64NeonRayon
);

#[cfg(feature = "enable-rayon")]
poulpy_core::core_encryption_parity_test_suite!(
    mod core_encryption_ntt4x30neonrayon,
    backend_ref = crate::NTT4x30Neon,
    backend_test = crate::NTT4x30NeonRayon
);
