#[cfg(feature = "enable-ckks")]
mod ckks_tests;

poulpy_core::core_parity_test_suite! {
    mod core_parity_fft64,
    backend_ref = poulpy_cpu_ref::FFT64Ref,
    backend_test = crate::FFT64Avx,
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

#[cfg(feature = "enable-rayon")]
poulpy_core::core_parity_test_suite! {
    mod core_parity_fft64_rayon,
    backend_ref = poulpy_cpu_ref::FFT64Ref,
    backend_test = crate::FFT64AvxRayon,
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
    backend_test = crate::NTT4x30Avx,
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

// Guards the narrowing path: a backend that only serves rank 1 restricts the
// sweep instead of forgoing the suite.
poulpy_core::core_parity_test_suite! {
    mod core_parity_rank1_only,
    backend_ref = poulpy_cpu_ref::FFT64Ref,
    backend_test = crate::FFT64Avx,
    params = TestParams { size: 1<<8, base2k: 17 },
    shapes = poulpy_core::test_suite::parity::ParityShapes {
        ranks: vec![1],
        dsizes: Some(vec![1, 2]),
    },
    tests = {
        glwe_keyswitch => poulpy_core::test_suite::parity::test_glwe_keyswitch_parity,
        glwe_automorphism => poulpy_core::test_suite::parity::test_glwe_automorphism_parity,
    }
}

#[cfg(feature = "enable-avx")]
poulpy_bin_fhe::bin_fhe_backend_test_suite!(mod bin_fhe_fft64, backend = crate::FFT64Avx);

#[cfg(feature = "enable-rayon")]
poulpy_bin_fhe::bin_fhe_backend_test_suite!(mod bin_fhe_fft64_rayon, backend = crate::FFT64AvxRayon);

#[cfg(feature = "enable-rayon")]
poulpy_bin_fhe::bin_fhe_backend_test_suite!(mod bin_fhe_ntt4x30_rayon, backend = crate::NTT4x30AvxRayon);

#[cfg(all(
    feature = "enable-rayon",
    target_arch = "x86_64",
    target_feature = "avx2",
    target_feature = "fma"
))]
poulpy_bin_fhe::bin_fhe_parity_test_suite!(
    mod bin_fhe_parity_fft64_rayon,
    backend_ref = crate::FFT64Avx,
    backend_test = crate::FFT64AvxRayon,
);

#[cfg(all(
    feature = "enable-rayon",
    target_arch = "x86_64",
    target_feature = "avx2",
    target_feature = "fma"
))]
poulpy_bin_fhe::bin_fhe_parity_test_suite!(
    mod bin_fhe_parity_ntt4x30_rayon,
    backend_ref = crate::NTT4x30Avx,
    backend_test = crate::NTT4x30AvxRayon,
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
        thread_scaling::<crate::FFT64AvxRayon>(ProbeShape::square(1 << LOG_N, SIZE, RANK + 1), &sweep, MODE)
            .print("FFT64AvxRayon");
        thread_scaling::<crate::NTT4x30AvxRayon>(ProbeShape::square(1 << LOG_N, SIZE, RANK + 1), &sweep, MODE)
            .print("NTT4x30AvxRayon");
    }
}
