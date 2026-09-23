use poulpy_ckks::ckks_backend_test_suite;

const ATK_ROTATIONS: &[i64] = &[1, 7];

#[cfg(feature = "enable-ifma")]
mod bootstrapping_presets {
    use poulpy_ckks::test_suite::presets::bootstrapping_presets_meet_precision;

    #[test]
    #[ignore = "full logN16 bootstrapping presets"]
    fn ifma() {
        bootstrapping_presets_meet_precision::<crate::NTT3x42Ifma>(52);
    }

    #[cfg(feature = "enable-rayon")]
    #[test]
    #[ignore = "full logN16 bootstrapping presets"]
    fn ifma_rayon() {
        bootstrapping_presets_meet_precision::<crate::NTT3x42IfmaRayon>(52);
    }
}

ckks_backend_test_suite!(
    mod fft64_avx512_f64,
    backend = crate::FFT64Avx512,
    scalar = f64,
    encoder = crate::FFT64Avx512ReimTable,
    params = poulpy_ckks::test_suite::BASE19_PARAMS_F64,
    rotations = super::ATK_ROTATIONS,
);

#[cfg(feature = "enable-rayon")]
ckks_backend_test_suite!(
    mod fft64_avx512_rayon_f64,
    backend = crate::FFT64Avx512Rayon,
    scalar = f64,
    encoder = crate::FFT64Avx512ReimTable,
    params = poulpy_ckks::test_suite::BASE19_PARAMS_F64,
    rotations = super::ATK_ROTATIONS,
);

ckks_backend_test_suite!(
    mod ntt4x30_avx512_f64,
    backend = crate::NTT4x30Avx512,
    scalar = f64,
    encoder = crate::FFT64Avx512ReimTable,
    params = poulpy_ckks::test_suite::BASE52_PARAMS_F64,
    rotations = super::ATK_ROTATIONS,
);

ckks_backend_test_suite!(
    mod ntt4x30_avx512_f128,
    backend = crate::NTT4x30Avx512,
    scalar = poulpy_ckks::Quad,
    encoder = poulpy_cpu_ref::FFT64ReimTable<poulpy_ckks::Quad>,
    params = poulpy_ckks::test_suite::BASE52_PARAMS_QUAD,
    rotations = super::ATK_ROTATIONS,
);

#[cfg(feature = "enable-rayon")]
ckks_backend_test_suite!(
    mod ntt4x30_avx512_rayon_f64,
    backend = crate::NTT4x30Avx512Rayon,
    scalar = f64,
    encoder = crate::FFT64Avx512ReimTable,
    params = poulpy_ckks::test_suite::BASE52_PARAMS_F64,
    rotations = super::ATK_ROTATIONS,
);

#[cfg(feature = "enable-rayon")]
ckks_backend_test_suite!(
    mod ntt4x30_avx512_rayon_f128,
    backend = crate::NTT4x30Avx512Rayon,
    scalar = poulpy_ckks::Quad,
    encoder = poulpy_cpu_ref::FFT64ReimTable<poulpy_ckks::Quad>,
    params = poulpy_ckks::test_suite::BASE52_PARAMS_QUAD,
    rotations = super::ATK_ROTATIONS,
);

#[cfg(feature = "enable-ifma")]
ckks_backend_test_suite!(
    mod ntt3x42_ifma_f64,
    backend = crate::NTT3x42Ifma,
    scalar = f64,
    encoder = crate::FFT64Avx512ReimTable,
    params = poulpy_ckks::test_suite::BASE52_PARAMS_F64,
    rotations = super::ATK_ROTATIONS,
);

#[cfg(feature = "enable-ifma")]
ckks_backend_test_suite!(
    mod ntt3x42_ifma_f128,
    backend = crate::NTT3x42Ifma,
    scalar = poulpy_ckks::Quad,
    encoder = poulpy_cpu_ref::FFT64ReimTable<poulpy_ckks::Quad>,
    params = poulpy_ckks::test_suite::BASE52_PARAMS_QUAD,
    rotations = super::ATK_ROTATIONS,
);

#[cfg(all(feature = "enable-ifma", feature = "enable-rayon"))]
ckks_backend_test_suite!(
    mod ntt3x42_ifma_rayon_f64,
    backend = crate::NTT3x42IfmaRayon,
    scalar = f64,
    encoder = crate::FFT64Avx512ReimTable,
    params = poulpy_ckks::test_suite::BASE52_PARAMS_F64,
    rotations = super::ATK_ROTATIONS,
);

// Paired OEP validation. Serial backends validate against portable implementations;
// Rayon backends validate against their serial counterparts.
poulpy_ckks::ckks_parity_test_suite! {
    mod ckks_parity_fft64avx512_f64,
    backend_ref = poulpy_cpu_ref::FFT64Ref,
    backend_test = crate::FFT64Avx512,
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
    mod ckks_parity_fft64avx512_quad,
    backend_ref = poulpy_cpu_ref::FFT64Ref,
    backend_test = crate::FFT64Avx512,
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
    mod ckks_parity_fft64avx512_encryption,
    backend_ref = poulpy_cpu_ref::test_suite::ControlledSamplingFFT64Ref,
    backend_test = crate::FFT64Avx512,
    params = poulpy_ckks::test_suite::CKKSTestParams { n: 64, hw: 48, base2k: 12, ..poulpy_ckks::test_suite::BASE19_PARAMS_F64 },
}

poulpy_ckks::ckks_parity_test_suite! {
    mod ckks_parity_ntt4x30avx512_f64,
    backend_ref = poulpy_cpu_ref::NTT4x30Ref,
    backend_test = crate::NTT4x30Avx512,
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
    mod ckks_parity_ntt4x30avx512_quad,
    backend_ref = poulpy_cpu_ref::NTT4x30Ref,
    backend_test = crate::NTT4x30Avx512,
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
    mod ckks_parity_ntt4x30avx512_encryption,
    backend_ref = poulpy_cpu_ref::test_suite::ControlledSamplingFFT64Ref,
    backend_test = crate::NTT4x30Avx512,
    params = poulpy_ckks::test_suite::CKKSTestParams { n: 64, hw: 48, base2k: 12, ..poulpy_ckks::test_suite::BASE19_PARAMS_F64 },
}

#[cfg(feature = "enable-ifma")]
poulpy_ckks::ckks_parity_test_suite! {
    mod ckks_parity_ntt3x42ifma_f64,
    backend_ref = poulpy_cpu_ref::NTT4x30Ref,
    backend_test = crate::NTT3x42Ifma,
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

#[cfg(feature = "enable-ifma")]
poulpy_ckks::ckks_parity_test_suite! {
    mod ckks_parity_ntt3x42ifma_quad,
    backend_ref = poulpy_cpu_ref::NTT4x30Ref,
    backend_test = crate::NTT3x42Ifma,
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

#[cfg(feature = "enable-ifma")]
poulpy_ckks::ckks_encryption_parity_test_suite! {
    mod ckks_parity_ntt3x42ifma_encryption,
    backend_ref = poulpy_cpu_ref::test_suite::ControlledSamplingFFT64Ref,
    backend_test = crate::NTT3x42Ifma,
    params = poulpy_ckks::test_suite::CKKSTestParams { n: 64, hw: 48, base2k: 12, ..poulpy_ckks::test_suite::BASE19_PARAMS_F64 },
}

#[cfg(feature = "enable-rayon")]
poulpy_ckks::ckks_parity_test_suite! {
    mod ckks_parity_fft64avx512rayon_f64,
    backend_ref = crate::FFT64Avx512,
    backend_test = crate::FFT64Avx512Rayon,
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
    mod ckks_parity_fft64avx512rayon_quad,
    backend_ref = crate::FFT64Avx512,
    backend_test = crate::FFT64Avx512Rayon,
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
    mod ckks_parity_fft64avx512rayon_encryption,
    backend_ref = crate::FFT64Avx512,
    backend_test = crate::FFT64Avx512Rayon,
    params = poulpy_ckks::test_suite::CKKSTestParams { n: 64, hw: 48, base2k: 12, ..poulpy_ckks::test_suite::BASE19_PARAMS_F64 },
}

#[cfg(feature = "enable-rayon")]
poulpy_ckks::ckks_parity_test_suite! {
    mod ckks_parity_ntt4x30avx512rayon_f64,
    backend_ref = crate::NTT4x30Avx512,
    backend_test = crate::NTT4x30Avx512Rayon,
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
    mod ckks_parity_ntt4x30avx512rayon_quad,
    backend_ref = crate::NTT4x30Avx512,
    backend_test = crate::NTT4x30Avx512Rayon,
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
    mod ckks_parity_ntt4x30avx512rayon_encryption,
    backend_ref = crate::NTT4x30Avx512,
    backend_test = crate::NTT4x30Avx512Rayon,
    params = poulpy_ckks::test_suite::CKKSTestParams { n: 64, hw: 48, base2k: 12, ..poulpy_ckks::test_suite::BASE19_PARAMS_F64 },
}

#[cfg(all(feature = "enable-ifma", feature = "enable-rayon"))]
poulpy_ckks::ckks_parity_test_suite! {
    mod ckks_parity_ntt3x42ifmarayon_f64,
    backend_ref = crate::NTT3x42Ifma,
    backend_test = crate::NTT3x42IfmaRayon,
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

#[cfg(all(feature = "enable-ifma", feature = "enable-rayon"))]
poulpy_ckks::ckks_parity_test_suite! {
    mod ckks_parity_ntt3x42ifmarayon_quad,
    backend_ref = crate::NTT3x42Ifma,
    backend_test = crate::NTT3x42IfmaRayon,
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

#[cfg(all(feature = "enable-ifma", feature = "enable-rayon"))]
poulpy_ckks::ckks_encryption_parity_test_suite! {
    mod ckks_parity_ntt3x42ifmarayon_encryption,
    backend_ref = crate::NTT3x42Ifma,
    backend_test = crate::NTT3x42IfmaRayon,
    params = poulpy_ckks::test_suite::CKKSTestParams { n: 64, hw: 48, base2k: 12, ..poulpy_ckks::test_suite::BASE19_PARAMS_F64 },
}

// Explicit rank-2 contracts. A caller can select another supported rank through params.
poulpy_ckks::ckks_parity_test_suite! {
    mod ckks_parity_fft64avx512_rank2,
    backend_ref = poulpy_cpu_ref::FFT64Ref,
    backend_test = crate::FFT64Avx512,
    scalar = f64,
    params = poulpy_ckks::test_suite::CKKSTestParams { n: 64, hw: 48, rank: 2, base2k: 19, ..poulpy_ckks::test_suite::BASE19_PARAMS_F64 },
    tests = {
        arithmetic => poulpy_ckks::test_suite::parity::test_arithmetic_parity,
        multiplication => poulpy_ckks::test_suite::parity::test_multiplication_parity,
        automorphism => poulpy_ckks::test_suite::parity::test_automorphism_parity,
    }
}

poulpy_ckks::ckks_encryption_parity_test_suite! {
    mod ckks_parity_fft64avx512_encryption_rank2,
    backend_ref = poulpy_cpu_ref::test_suite::ControlledSamplingFFT64Ref,
    backend_test = crate::FFT64Avx512,
    params = poulpy_ckks::test_suite::CKKSTestParams { n: 64, hw: 48, rank: 2, base2k: 12, ..poulpy_ckks::test_suite::BASE19_PARAMS_F64 },
}

poulpy_ckks::ckks_parity_test_suite! {
    mod ckks_parity_ntt4x30avx512_rank2,
    backend_ref = poulpy_cpu_ref::NTT4x30Ref,
    backend_test = crate::NTT4x30Avx512,
    scalar = f64,
    params = poulpy_ckks::test_suite::CKKSTestParams { n: 64, hw: 48, rank: 2, base2k: 52, ..poulpy_ckks::test_suite::BASE19_PARAMS_F64 },
    tests = {
        arithmetic => poulpy_ckks::test_suite::parity::test_arithmetic_parity,
        multiplication => poulpy_ckks::test_suite::parity::test_multiplication_parity,
        automorphism => poulpy_ckks::test_suite::parity::test_automorphism_parity,
    }
}

poulpy_ckks::ckks_encryption_parity_test_suite! {
    mod ckks_parity_ntt4x30avx512_encryption_rank2,
    backend_ref = poulpy_cpu_ref::test_suite::ControlledSamplingFFT64Ref,
    backend_test = crate::NTT4x30Avx512,
    params = poulpy_ckks::test_suite::CKKSTestParams { n: 64, hw: 48, rank: 2, base2k: 12, ..poulpy_ckks::test_suite::BASE19_PARAMS_F64 },
}

#[cfg(feature = "enable-ifma")]
poulpy_ckks::ckks_parity_test_suite! {
    mod ckks_parity_ntt3x42ifma_rank2,
    backend_ref = poulpy_cpu_ref::NTT4x30Ref,
    backend_test = crate::NTT3x42Ifma,
    scalar = f64,
    params = poulpy_ckks::test_suite::CKKSTestParams { n: 64, hw: 48, rank: 2, base2k: 52, ..poulpy_ckks::test_suite::BASE19_PARAMS_F64 },
    tests = {
        arithmetic => poulpy_ckks::test_suite::parity::test_arithmetic_parity,
        multiplication => poulpy_ckks::test_suite::parity::test_multiplication_parity,
        automorphism => poulpy_ckks::test_suite::parity::test_automorphism_parity,
    }
}

#[cfg(feature = "enable-ifma")]
poulpy_ckks::ckks_encryption_parity_test_suite! {
    mod ckks_parity_ntt3x42ifma_encryption_rank2,
    backend_ref = poulpy_cpu_ref::test_suite::ControlledSamplingFFT64Ref,
    backend_test = crate::NTT3x42Ifma,
    params = poulpy_ckks::test_suite::CKKSTestParams { n: 64, hw: 48, rank: 2, base2k: 12, ..poulpy_ckks::test_suite::BASE19_PARAMS_F64 },
}

#[cfg(feature = "enable-rayon")]
poulpy_ckks::ckks_parity_test_suite! {
    mod ckks_parity_fft64avx512rayon_rank2,
    backend_ref = crate::FFT64Avx512,
    backend_test = crate::FFT64Avx512Rayon,
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
    mod ckks_parity_fft64avx512rayon_encryption_rank2,
    backend_ref = crate::FFT64Avx512,
    backend_test = crate::FFT64Avx512Rayon,
    params = poulpy_ckks::test_suite::CKKSTestParams { n: 64, hw: 48, rank: 2, base2k: 12, ..poulpy_ckks::test_suite::BASE19_PARAMS_F64 },
}

#[cfg(feature = "enable-rayon")]
poulpy_ckks::ckks_parity_test_suite! {
    mod ckks_parity_ntt4x30avx512rayon_rank2,
    backend_ref = crate::NTT4x30Avx512,
    backend_test = crate::NTT4x30Avx512Rayon,
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
    mod ckks_parity_ntt4x30avx512rayon_encryption_rank2,
    backend_ref = crate::NTT4x30Avx512,
    backend_test = crate::NTT4x30Avx512Rayon,
    params = poulpy_ckks::test_suite::CKKSTestParams { n: 64, hw: 48, rank: 2, base2k: 12, ..poulpy_ckks::test_suite::BASE19_PARAMS_F64 },
}

#[cfg(all(feature = "enable-ifma", feature = "enable-rayon"))]
poulpy_ckks::ckks_parity_test_suite! {
    mod ckks_parity_ntt3x42ifmarayon_rank2,
    backend_ref = crate::NTT3x42Ifma,
    backend_test = crate::NTT3x42IfmaRayon,
    scalar = f64,
    params = poulpy_ckks::test_suite::CKKSTestParams { n: 64, hw: 48, rank: 2, base2k: 52, ..poulpy_ckks::test_suite::BASE19_PARAMS_F64 },
    tests = {
        arithmetic => poulpy_ckks::test_suite::parity::test_arithmetic_parity,
        multiplication => poulpy_ckks::test_suite::parity::test_multiplication_parity,
        automorphism => poulpy_ckks::test_suite::parity::test_automorphism_parity,
    }
}

#[cfg(all(feature = "enable-ifma", feature = "enable-rayon"))]
poulpy_ckks::ckks_encryption_parity_test_suite! {
    mod ckks_parity_ntt3x42ifmarayon_encryption_rank2,
    backend_ref = crate::NTT3x42Ifma,
    backend_test = crate::NTT3x42IfmaRayon,
    params = poulpy_ckks::test_suite::CKKSTestParams { n: 64, hw: 48, rank: 2, base2k: 12, ..poulpy_ckks::test_suite::BASE19_PARAMS_F64 },
}
