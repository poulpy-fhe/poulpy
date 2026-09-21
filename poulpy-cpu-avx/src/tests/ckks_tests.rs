use poulpy_ckks::ckks_backend_test_suite;

const ATK_ROTATIONS: &[i64] = &[1, 7];

ckks_backend_test_suite!(
    mod fft64_f64,
    backend = crate::FFT64Avx,
    scalar = f64,
    encoder = crate::FFT64AvxReimTable,
    params = poulpy_ckks::test_suite::BASE19_PARAMS_F64,
    rotations = super::ATK_ROTATIONS,
);

#[cfg(feature = "enable-rayon")]
ckks_backend_test_suite!(
    mod fft64_rayon_f64,
    backend = crate::FFT64AvxRayon,
    scalar = f64,
    encoder = crate::FFT64AvxReimTable,
    params = poulpy_ckks::test_suite::BASE19_PARAMS_F64,
    rotations = super::ATK_ROTATIONS,
);

#[cfg(feature = "enable-rayon")]
ckks_backend_test_suite!(
    mod ntt4x30_rayon_f64,
    backend = crate::NTT4x30AvxRayon,
    scalar = f64,
    encoder = crate::FFT64AvxReimTable,
    params = poulpy_ckks::test_suite::BASE52_PARAMS_F64,
    rotations = super::ATK_ROTATIONS,
);

ckks_backend_test_suite!(
    mod ntt4x30_f64,
    backend = crate::NTT4x30Avx,
    scalar = f64,
    encoder = crate::FFT64AvxReimTable,
    params = poulpy_ckks::test_suite::BASE52_PARAMS_F64,
    rotations = super::ATK_ROTATIONS,
);

/// Full logN16 bootstraps per preset: slow, so opt in with `--ignored`.
mod bootstrapping_presets {
    use poulpy_ckks::test_suite::presets::bootstrapping_presets_meet_precision;

    #[test]
    #[ignore = "runs a full logN16 bootstrap per preset; opt in with --ignored"]
    fn ntt4x30_presets_meet_precision() {
        bootstrapping_presets_meet_precision::<crate::NTT4x30Avx>();
    }

    #[test]
    #[ignore = "runs a full logN16 bootstrap per preset; opt in with --ignored"]
    fn fft64_presets_meet_precision() {
        bootstrapping_presets_meet_precision::<crate::FFT64Avx>();
    }
}

// Paired OEP validation. Serial backends validate against portable implementations;
// Rayon backends validate against their serial counterparts.
poulpy_ckks::ckks_parity_test_suite! {
    mod ckks_parity_fft64avx_f64,
    backend_ref = poulpy_cpu_ref::FFT64Ref,
    backend_test = crate::FFT64Avx,
    scalar = f64,
    params = poulpy_ckks::test_suite::CKKSTestParams { n: 64, hw: 48, base2k: 19, ..poulpy_ckks::test_suite::BASE19_PARAMS_F64 },
    tests = {
        arithmetic => poulpy_ckks::test_suite::parity::test_arithmetic_parity,
        multiplication => poulpy_ckks::test_suite::parity::test_multiplication_parity,
        automorphism => poulpy_ckks::test_suite::parity::test_automorphism_parity,
        plaintext => poulpy_ckks::test_suite::parity::test_plaintext_parity,
        encoding => poulpy_ckks::test_suite::parity::test_encoding_parity,
        paco_encoding => poulpy_ckks::test_suite::parity::test_paco_encoding_parity,
        ship_encoding => poulpy_ckks::test_suite::parity::test_ship_encoding_parity,
        dft => poulpy_ckks::test_suite::parity::test_dft_parity,
        polynomial_eval_mod => poulpy_ckks::test_suite::parity::test_polynomial_eval_mod_parity,
        encapsulated_mod_up => poulpy_ckks::test_suite::parity::test_encapsulated_mod_up_parity,
    }
}

poulpy_ckks::ckks_parity_test_suite! {
    mod ckks_parity_fft64avx_quad,
    backend_ref = poulpy_cpu_ref::FFT64Ref,
    backend_test = crate::FFT64Avx,
    scalar = poulpy_ckks::Quad,
    params = poulpy_ckks::test_suite::CKKSTestParams { n: 64, hw: 48, base2k: 19, ..poulpy_ckks::test_suite::BASE52_PARAMS_QUAD },
    tests = {
        arithmetic => poulpy_ckks::test_suite::parity::test_arithmetic_parity,
        multiplication => poulpy_ckks::test_suite::parity::test_multiplication_parity,
        automorphism => poulpy_ckks::test_suite::parity::test_automorphism_parity,
        plaintext => poulpy_ckks::test_suite::parity::test_plaintext_parity,
        encoding => poulpy_ckks::test_suite::parity::test_encoding_parity,
        paco_encoding => poulpy_ckks::test_suite::parity::test_paco_encoding_parity,
        ship_encoding => poulpy_ckks::test_suite::parity::test_ship_encoding_parity,
        dft => poulpy_ckks::test_suite::parity::test_dft_parity,
        polynomial_eval_mod => poulpy_ckks::test_suite::parity::test_polynomial_eval_mod_parity,
        encapsulated_mod_up => poulpy_ckks::test_suite::parity::test_encapsulated_mod_up_parity,
    }
}

poulpy_ckks::ckks_encryption_parity_test_suite! {
    mod ckks_parity_fft64avx_encryption,
    backend_ref = poulpy_cpu_ref::test_suite::ControlledSamplingFFT64Ref,
    backend_test = crate::FFT64Avx,
    params = poulpy_ckks::test_suite::CKKSTestParams { n: 64, hw: 48, base2k: 12, ..poulpy_ckks::test_suite::BASE19_PARAMS_F64 },
}

poulpy_ckks::ckks_parity_test_suite! {
    mod ckks_parity_ntt4x30avx_f64,
    backend_ref = poulpy_cpu_ref::NTT4x30Ref,
    backend_test = crate::NTT4x30Avx,
    scalar = f64,
    params = poulpy_ckks::test_suite::CKKSTestParams { n: 64, hw: 48, base2k: 52, ..poulpy_ckks::test_suite::BASE52_PARAMS_F64 },
    tests = {
        arithmetic => poulpy_ckks::test_suite::parity::test_arithmetic_parity,
        multiplication => poulpy_ckks::test_suite::parity::test_multiplication_parity,
        automorphism => poulpy_ckks::test_suite::parity::test_automorphism_parity,
        plaintext => poulpy_ckks::test_suite::parity::test_plaintext_parity,
        encoding => poulpy_ckks::test_suite::parity::test_encoding_parity,
        paco_encoding => poulpy_ckks::test_suite::parity::test_paco_encoding_parity,
        ship_encoding => poulpy_ckks::test_suite::parity::test_ship_encoding_parity,
        dft => poulpy_ckks::test_suite::parity::test_dft_parity,
        polynomial_eval_mod => poulpy_ckks::test_suite::parity::test_polynomial_eval_mod_parity,
        encapsulated_mod_up => poulpy_ckks::test_suite::parity::test_encapsulated_mod_up_parity,
    }
}

poulpy_ckks::ckks_parity_test_suite! {
    mod ckks_parity_ntt4x30avx_quad,
    backend_ref = poulpy_cpu_ref::NTT4x30Ref,
    backend_test = crate::NTT4x30Avx,
    scalar = poulpy_ckks::Quad,
    params = poulpy_ckks::test_suite::CKKSTestParams { n: 64, hw: 48, base2k: 52, ..poulpy_ckks::test_suite::BASE52_PARAMS_QUAD },
    tests = {
        arithmetic => poulpy_ckks::test_suite::parity::test_arithmetic_parity,
        multiplication => poulpy_ckks::test_suite::parity::test_multiplication_parity,
        automorphism => poulpy_ckks::test_suite::parity::test_automorphism_parity,
        plaintext => poulpy_ckks::test_suite::parity::test_plaintext_parity,
        encoding => poulpy_ckks::test_suite::parity::test_encoding_parity,
        paco_encoding => poulpy_ckks::test_suite::parity::test_paco_encoding_parity,
        ship_encoding => poulpy_ckks::test_suite::parity::test_ship_encoding_parity,
        dft => poulpy_ckks::test_suite::parity::test_dft_parity,
        polynomial_eval_mod => poulpy_ckks::test_suite::parity::test_polynomial_eval_mod_parity,
        encapsulated_mod_up => poulpy_ckks::test_suite::parity::test_encapsulated_mod_up_parity,
    }
}

poulpy_ckks::ckks_encryption_parity_test_suite! {
    mod ckks_parity_ntt4x30avx_encryption,
    backend_ref = poulpy_cpu_ref::test_suite::ControlledSamplingFFT64Ref,
    backend_test = crate::NTT4x30Avx,
    params = poulpy_ckks::test_suite::CKKSTestParams { n: 64, hw: 48, base2k: 12, ..poulpy_ckks::test_suite::BASE19_PARAMS_F64 },
}

#[cfg(feature = "enable-rayon")]
poulpy_ckks::ckks_parity_test_suite! {
    mod ckks_parity_fft64avxrayon_f64,
    backend_ref = crate::FFT64Avx,
    backend_test = crate::FFT64AvxRayon,
    scalar = f64,
    params = poulpy_ckks::test_suite::CKKSTestParams { n: 64, hw: 48, base2k: 19, ..poulpy_ckks::test_suite::BASE19_PARAMS_F64 },
    tests = {
        arithmetic => poulpy_ckks::test_suite::parity::test_arithmetic_parity,
        multiplication => poulpy_ckks::test_suite::parity::test_multiplication_parity,
        automorphism => poulpy_ckks::test_suite::parity::test_automorphism_parity,
        plaintext => poulpy_ckks::test_suite::parity::test_plaintext_parity,
        encoding => poulpy_ckks::test_suite::parity::test_encoding_parity,
        paco_encoding => poulpy_ckks::test_suite::parity::test_paco_encoding_parity,
        ship_encoding => poulpy_ckks::test_suite::parity::test_ship_encoding_parity,
        dft => poulpy_ckks::test_suite::parity::test_dft_parity,
        polynomial_eval_mod => poulpy_ckks::test_suite::parity::test_polynomial_eval_mod_parity,
        encapsulated_mod_up => poulpy_ckks::test_suite::parity::test_encapsulated_mod_up_parity,
    }
}

#[cfg(feature = "enable-rayon")]
poulpy_ckks::ckks_parity_test_suite! {
    mod ckks_parity_fft64avxrayon_quad,
    backend_ref = crate::FFT64Avx,
    backend_test = crate::FFT64AvxRayon,
    scalar = poulpy_ckks::Quad,
    params = poulpy_ckks::test_suite::CKKSTestParams { n: 64, hw: 48, base2k: 19, ..poulpy_ckks::test_suite::BASE52_PARAMS_QUAD },
    tests = {
        arithmetic => poulpy_ckks::test_suite::parity::test_arithmetic_parity,
        multiplication => poulpy_ckks::test_suite::parity::test_multiplication_parity,
        automorphism => poulpy_ckks::test_suite::parity::test_automorphism_parity,
        plaintext => poulpy_ckks::test_suite::parity::test_plaintext_parity,
        encoding => poulpy_ckks::test_suite::parity::test_encoding_parity,
        paco_encoding => poulpy_ckks::test_suite::parity::test_paco_encoding_parity,
        ship_encoding => poulpy_ckks::test_suite::parity::test_ship_encoding_parity,
        dft => poulpy_ckks::test_suite::parity::test_dft_parity,
        polynomial_eval_mod => poulpy_ckks::test_suite::parity::test_polynomial_eval_mod_parity,
        encapsulated_mod_up => poulpy_ckks::test_suite::parity::test_encapsulated_mod_up_parity,
    }
}

#[cfg(feature = "enable-rayon")]
poulpy_ckks::ckks_encryption_parity_test_suite! {
    mod ckks_parity_fft64avxrayon_encryption,
    backend_ref = crate::FFT64Avx,
    backend_test = crate::FFT64AvxRayon,
    params = poulpy_ckks::test_suite::CKKSTestParams { n: 64, hw: 48, base2k: 12, ..poulpy_ckks::test_suite::BASE19_PARAMS_F64 },
}

#[cfg(feature = "enable-rayon")]
poulpy_ckks::ckks_parity_test_suite! {
    mod ckks_parity_ntt4x30avxrayon_f64,
    backend_ref = crate::NTT4x30Avx,
    backend_test = crate::NTT4x30AvxRayon,
    scalar = f64,
    params = poulpy_ckks::test_suite::CKKSTestParams { n: 64, hw: 48, base2k: 52, ..poulpy_ckks::test_suite::BASE52_PARAMS_F64 },
    tests = {
        arithmetic => poulpy_ckks::test_suite::parity::test_arithmetic_parity,
        multiplication => poulpy_ckks::test_suite::parity::test_multiplication_parity,
        automorphism => poulpy_ckks::test_suite::parity::test_automorphism_parity,
        plaintext => poulpy_ckks::test_suite::parity::test_plaintext_parity,
        encoding => poulpy_ckks::test_suite::parity::test_encoding_parity,
        paco_encoding => poulpy_ckks::test_suite::parity::test_paco_encoding_parity,
        ship_encoding => poulpy_ckks::test_suite::parity::test_ship_encoding_parity,
        dft => poulpy_ckks::test_suite::parity::test_dft_parity,
        polynomial_eval_mod => poulpy_ckks::test_suite::parity::test_polynomial_eval_mod_parity,
        encapsulated_mod_up => poulpy_ckks::test_suite::parity::test_encapsulated_mod_up_parity,
    }
}

#[cfg(feature = "enable-rayon")]
poulpy_ckks::ckks_parity_test_suite! {
    mod ckks_parity_ntt4x30avxrayon_quad,
    backend_ref = crate::NTT4x30Avx,
    backend_test = crate::NTT4x30AvxRayon,
    scalar = poulpy_ckks::Quad,
    params = poulpy_ckks::test_suite::CKKSTestParams { n: 64, hw: 48, base2k: 52, ..poulpy_ckks::test_suite::BASE52_PARAMS_QUAD },
    tests = {
        arithmetic => poulpy_ckks::test_suite::parity::test_arithmetic_parity,
        multiplication => poulpy_ckks::test_suite::parity::test_multiplication_parity,
        automorphism => poulpy_ckks::test_suite::parity::test_automorphism_parity,
        plaintext => poulpy_ckks::test_suite::parity::test_plaintext_parity,
        encoding => poulpy_ckks::test_suite::parity::test_encoding_parity,
        paco_encoding => poulpy_ckks::test_suite::parity::test_paco_encoding_parity,
        ship_encoding => poulpy_ckks::test_suite::parity::test_ship_encoding_parity,
        dft => poulpy_ckks::test_suite::parity::test_dft_parity,
        polynomial_eval_mod => poulpy_ckks::test_suite::parity::test_polynomial_eval_mod_parity,
        encapsulated_mod_up => poulpy_ckks::test_suite::parity::test_encapsulated_mod_up_parity,
    }
}

#[cfg(feature = "enable-rayon")]
poulpy_ckks::ckks_encryption_parity_test_suite! {
    mod ckks_parity_ntt4x30avxrayon_encryption,
    backend_ref = crate::NTT4x30Avx,
    backend_test = crate::NTT4x30AvxRayon,
    params = poulpy_ckks::test_suite::CKKSTestParams { n: 64, hw: 48, base2k: 12, ..poulpy_ckks::test_suite::BASE19_PARAMS_F64 },
}

// Explicit rank-2 contracts. A caller can select another supported rank through params.
poulpy_ckks::ckks_parity_test_suite! {
    mod ckks_parity_fft64avx_rank2,
    backend_ref = poulpy_cpu_ref::FFT64Ref,
    backend_test = crate::FFT64Avx,
    scalar = f64,
    params = poulpy_ckks::test_suite::CKKSTestParams { n: 64, hw: 48, rank: 2, base2k: 19, ..poulpy_ckks::test_suite::BASE19_PARAMS_F64 },
    tests = {
        arithmetic => poulpy_ckks::test_suite::parity::test_arithmetic_parity,
        multiplication => poulpy_ckks::test_suite::parity::test_multiplication_parity,
        automorphism => poulpy_ckks::test_suite::parity::test_automorphism_parity,
    }
}

poulpy_ckks::ckks_encryption_parity_test_suite! {
    mod ckks_parity_fft64avx_encryption_rank2,
    backend_ref = poulpy_cpu_ref::test_suite::ControlledSamplingFFT64Ref,
    backend_test = crate::FFT64Avx,
    params = poulpy_ckks::test_suite::CKKSTestParams { n: 64, hw: 48, rank: 2, base2k: 12, ..poulpy_ckks::test_suite::BASE19_PARAMS_F64 },
}

poulpy_ckks::ckks_parity_test_suite! {
    mod ckks_parity_ntt4x30avx_rank2,
    backend_ref = poulpy_cpu_ref::NTT4x30Ref,
    backend_test = crate::NTT4x30Avx,
    scalar = f64,
    params = poulpy_ckks::test_suite::CKKSTestParams { n: 64, hw: 48, rank: 2, base2k: 52, ..poulpy_ckks::test_suite::BASE19_PARAMS_F64 },
    tests = {
        arithmetic => poulpy_ckks::test_suite::parity::test_arithmetic_parity,
        multiplication => poulpy_ckks::test_suite::parity::test_multiplication_parity,
        automorphism => poulpy_ckks::test_suite::parity::test_automorphism_parity,
    }
}

poulpy_ckks::ckks_encryption_parity_test_suite! {
    mod ckks_parity_ntt4x30avx_encryption_rank2,
    backend_ref = poulpy_cpu_ref::test_suite::ControlledSamplingFFT64Ref,
    backend_test = crate::NTT4x30Avx,
    params = poulpy_ckks::test_suite::CKKSTestParams { n: 64, hw: 48, rank: 2, base2k: 12, ..poulpy_ckks::test_suite::BASE19_PARAMS_F64 },
}

#[cfg(feature = "enable-rayon")]
poulpy_ckks::ckks_parity_test_suite! {
    mod ckks_parity_fft64avxrayon_rank2,
    backend_ref = crate::FFT64Avx,
    backend_test = crate::FFT64AvxRayon,
    scalar = f64,
    params = poulpy_ckks::test_suite::CKKSTestParams { n: 64, hw: 48, rank: 2, base2k: 19, ..poulpy_ckks::test_suite::BASE19_PARAMS_F64 },
    tests = {
        arithmetic => poulpy_ckks::test_suite::parity::test_arithmetic_parity,
        multiplication => poulpy_ckks::test_suite::parity::test_multiplication_parity,
        automorphism => poulpy_ckks::test_suite::parity::test_automorphism_parity,
    }
}

#[cfg(feature = "enable-rayon")]
poulpy_ckks::ckks_encryption_parity_test_suite! {
    mod ckks_parity_fft64avxrayon_encryption_rank2,
    backend_ref = crate::FFT64Avx,
    backend_test = crate::FFT64AvxRayon,
    params = poulpy_ckks::test_suite::CKKSTestParams { n: 64, hw: 48, rank: 2, base2k: 12, ..poulpy_ckks::test_suite::BASE19_PARAMS_F64 },
}

#[cfg(feature = "enable-rayon")]
poulpy_ckks::ckks_parity_test_suite! {
    mod ckks_parity_ntt4x30avxrayon_rank2,
    backend_ref = crate::NTT4x30Avx,
    backend_test = crate::NTT4x30AvxRayon,
    scalar = f64,
    params = poulpy_ckks::test_suite::CKKSTestParams { n: 64, hw: 48, rank: 2, base2k: 52, ..poulpy_ckks::test_suite::BASE19_PARAMS_F64 },
    tests = {
        arithmetic => poulpy_ckks::test_suite::parity::test_arithmetic_parity,
        multiplication => poulpy_ckks::test_suite::parity::test_multiplication_parity,
        automorphism => poulpy_ckks::test_suite::parity::test_automorphism_parity,
    }
}

#[cfg(feature = "enable-rayon")]
poulpy_ckks::ckks_encryption_parity_test_suite! {
    mod ckks_parity_ntt4x30avxrayon_encryption_rank2,
    backend_ref = crate::NTT4x30Avx,
    backend_test = crate::NTT4x30AvxRayon,
    params = poulpy_ckks::test_suite::CKKSTestParams { n: 64, hw: 48, rank: 2, base2k: 12, ..poulpy_ckks::test_suite::BASE19_PARAMS_F64 },
}
