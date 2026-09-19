#[cfg(feature = "enable-ckks")]
mod ckks_tests;

#[test]
fn glwe_copy() {
    use poulpy_core::test_suite::copy::test_glwe_copy;
    use poulpy_hal::{layouts::Module, test_suite::TestParams};

    let fft = TestParams {
        size: 256,
        base2k: 17,
        n: 256,
    };
    let ntt = TestParams {
        size: 256,
        base2k: 52,
        n: 256,
    };
    test_glwe_copy(&fft, &Module::<crate::FFT64Avx512>::new(256));
    test_glwe_copy(&ntt, &Module::<crate::NTT4x30Avx512>::new(256));
    #[cfg(feature = "enable-rayon")]
    {
        test_glwe_copy(&fft, &Module::<crate::FFT64Avx512Rayon>::new(256));
        test_glwe_copy(&ntt, &Module::<crate::NTT4x30Avx512Rayon>::new(256));
    }
    #[cfg(feature = "enable-ifma")]
    {
        let ifma = TestParams {
            size: 256,
            base2k: 42,
            n: 256,
        };
        test_glwe_copy(&ifma, &Module::<crate::NTT3x42Ifma>::new(256));
        #[cfg(feature = "enable-rayon")]
        test_glwe_copy(&ifma, &Module::<crate::NTT3x42IfmaRayon>::new(256));
    }
}

poulpy_core::core_parity_test_suite! {
    mod core_parity_fft64,
    backend_ref = poulpy_cpu_ref::FFT64Ref,
    backend_test = crate::FFT64Avx512,
    // computes at the module degree, no sweep
    params = TestParams { size: 1<<8, base2k: 17, n: 1<<8 },
    tests = {
        glwe_keyswitch => poulpy_core::test_suite::parity::test_glwe_keyswitch_parity,
        glwe_keyswitch_assign => poulpy_core::test_suite::parity::test_glwe_keyswitch_assign_parity,
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
    // computes at the module degree, no sweep
    params = TestParams { size: 1<<8, base2k: 52, n: 1<<8 },
    tests = {
        glwe_keyswitch => poulpy_core::test_suite::parity::test_glwe_keyswitch_parity,
        glwe_keyswitch_assign => poulpy_core::test_suite::parity::test_glwe_keyswitch_assign_parity,
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
    mod core_parity_ntt4x30_rayon,
    backend_ref = poulpy_cpu_ref::NTT4x30Ref,
    backend_test = crate::NTT4x30Avx512Rayon,
    // computes at the module degree, no sweep
    params = TestParams { size: 1<<8, base2k: 52, n: 1<<8 },
    tests = {
        glwe_keyswitch => poulpy_core::test_suite::parity::test_glwe_keyswitch_parity,
        glwe_keyswitch_assign => poulpy_core::test_suite::parity::test_glwe_keyswitch_assign_parity,
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
    params = TestParams { size: 1<<15, base2k: 52, n: 1<<15 },
    tests = {
        glwe_tensor => poulpy_core::test_suite::parity::test_glwe_tensor_parity,
    }
}

#[cfg(feature = "enable-rayon")]
poulpy_core::core_parity_test_suite! {
    mod core_parity_ntt4x30_rayon_fused,
    backend_ref = poulpy_cpu_ref::NTT4x30Ref,
    backend_test = crate::NTT4x30Avx512Rayon,
    params = TestParams { size: 1<<15, base2k: 52, n: 1<<15 },
    tests = {
        glwe_tensor => poulpy_core::test_suite::parity::test_glwe_tensor_parity,
    }
}

poulpy_core::core_parity_test_suite! {
    mod core_parity_ntt4x30_fused_n16,
    backend_ref = poulpy_cpu_ref::NTT4x30Ref,
    backend_test = crate::NTT4x30Avx512,
    params = TestParams { size: 1<<16, base2k: 52, n: 1<<16 },
    tests = {
        glwe_tensor => poulpy_core::test_suite::parity::test_glwe_tensor_parity,
    }
}

#[cfg(feature = "enable-ifma")]
poulpy_core::core_parity_test_suite! {
    mod core_parity_ntt3x42_ifma_fused,
    backend_ref = poulpy_cpu_ref::NTT4x30Ref,
    backend_test = crate::NTT3x42Ifma,
    params = TestParams { size: 1<<15, base2k: 52, n: 1<<15 },
    tests = {
        glwe_tensor => poulpy_core::test_suite::parity::test_glwe_tensor_parity,
    }
}

#[cfg(feature = "enable-ifma")]
poulpy_core::core_parity_test_suite! {
    mod core_parity_ntt3x42_ifma_fused_n16,
    backend_ref = poulpy_cpu_ref::NTT4x30Ref,
    backend_test = crate::NTT3x42Ifma,
    params = TestParams { size: 1<<16, base2k: 52, n: 1<<16 },
    tests = {
        glwe_tensor => poulpy_core::test_suite::parity::test_glwe_tensor_parity,
    }
}

#[cfg(all(feature = "enable-ifma", feature = "enable-rayon"))]
poulpy_core::core_parity_test_suite! {
    mod core_parity_ntt3x42_ifma_rayon_fused,
    backend_ref = poulpy_cpu_ref::NTT4x30Ref,
    backend_test = crate::NTT3x42IfmaRayon,
    params = TestParams { size: 1<<15, base2k: 52, n: 1<<15 },
    tests = {
        glwe_tensor => poulpy_core::test_suite::parity::test_glwe_tensor_parity,
    }
}

#[cfg(feature = "enable-ifma")]
poulpy_core::core_parity_test_suite! {
    mod core_parity_ntt3x42_ifma,
    backend_ref = poulpy_cpu_ref::NTT4x30Ref,
    backend_test = crate::NTT3x42Ifma,
    // computes at the module degree, no sweep
    params = TestParams { size: 1<<8, base2k: 52, n: 1<<8 },
    tests = {
        glwe_keyswitch => poulpy_core::test_suite::parity::test_glwe_keyswitch_parity,
        glwe_keyswitch_assign => poulpy_core::test_suite::parity::test_glwe_keyswitch_assign_parity,
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

#[cfg(all(feature = "enable-ifma", feature = "enable-rayon"))]
poulpy_core::core_parity_test_suite! {
    mod core_parity_ntt3x42_ifma_rayon,
    backend_ref = poulpy_cpu_ref::NTT4x30Ref,
    backend_test = crate::NTT3x42IfmaRayon,
    // computes at the module degree, no sweep
    params = TestParams { size: 1<<8, base2k: 52, n: 1<<8 },
    tests = {
        glwe_keyswitch => poulpy_core::test_suite::parity::test_glwe_keyswitch_parity,
        glwe_keyswitch_assign => poulpy_core::test_suite::parity::test_glwe_keyswitch_assign_parity,
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

#[cfg(all(feature = "enable-rayon", target_arch = "x86_64", target_feature = "avx512f"))]
poulpy_bin_fhe::bin_fhe_backend_test_suite!(mod bin_fhe_fft64_rayon, backend = crate::FFT64Avx512Rayon);

#[cfg(all(feature = "enable-rayon", target_arch = "x86_64", target_feature = "avx512f"))]
poulpy_bin_fhe::bin_fhe_backend_test_suite!(mod bin_fhe_ntt4x30_rayon, backend = crate::NTT4x30Avx512Rayon);

#[cfg(all(feature = "enable-rayon", target_arch = "x86_64", target_feature = "avx512f"))]
poulpy_bin_fhe::bin_fhe_parity_test_suite!(
    mod bin_fhe_parity_fft64_rayon,
    backend_ref = crate::FFT64Avx512,
    backend_test = crate::FFT64Avx512Rayon,
);

#[cfg(all(feature = "enable-rayon", target_arch = "x86_64", target_feature = "avx512f"))]
poulpy_bin_fhe::bin_fhe_parity_test_suite!(
    mod bin_fhe_parity_ntt4x30_rayon,
    backend_ref = crate::NTT4x30Avx512,
    backend_test = crate::NTT4x30Avx512Rayon,
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
        thread_scaling::<crate::FFT64Avx512Rayon>(ProbeShape::square(1 << LOG_N, SIZE, RANK + 1), &sweep, MODE)
            .print("FFT64Avx512Rayon");
        thread_scaling::<crate::NTT4x30Avx512Rayon>(ProbeShape::square(1 << LOG_N, SIZE, RANK + 1), &sweep, MODE)
            .print("NTT4x30Avx512Rayon");
        #[cfg(feature = "enable-ifma")]
        thread_scaling::<crate::NTT3x42IfmaRayon>(ProbeShape::square(1 << LOG_N, SIZE, RANK + 1), &sweep, MODE)
            .print("NTT3x42IfmaRayon");
    }
}
