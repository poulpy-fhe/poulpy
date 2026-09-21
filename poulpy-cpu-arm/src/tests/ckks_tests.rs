use poulpy_ckks::ckks_backend_test_suite;

const ATK_ROTATIONS: &[i64] = &[1, 7];

ckks_backend_test_suite!(
    mod fft64_f64,
    backend = crate::FFT64Neon,
    scalar = f64,
    encoder = crate::FFT64NeonReimTable,
    params = poulpy_ckks::test_suite::BASE19_PARAMS_F64,
    rotations = super::ATK_ROTATIONS,
);

#[cfg(feature = "enable-rayon")]
ckks_backend_test_suite!(
    mod fft64_rayon_f64,
    backend = crate::FFT64NeonRayon,
    scalar = f64,
    encoder = crate::FFT64NeonReimTable,
    params = poulpy_ckks::test_suite::BASE19_PARAMS_F64,
    rotations = super::ATK_ROTATIONS,
);

ckks_backend_test_suite!(
    mod ntt4x30_f64,
    backend = crate::NTT4x30Neon,
    scalar = f64,
    encoder = crate::FFT64NeonReimTable,
    params = poulpy_ckks::test_suite::BASE52_PARAMS_F64,
    rotations = super::ATK_ROTATIONS,
);

#[cfg(feature = "enable-rayon")]
ckks_backend_test_suite!(
    mod ntt4x30_rayon_f64,
    backend = crate::NTT4x30NeonRayon,
    scalar = f64,
    encoder = crate::FFT64NeonReimTable,
    params = poulpy_ckks::test_suite::BASE52_PARAMS_F64,
    rotations = super::ATK_ROTATIONS,
);

// binary128 path on aarch64: `Quad` scalar with the reference encoder.
ckks_backend_test_suite!(
    mod ntt4x30_f128,
    backend = crate::NTT4x30Neon,
    scalar = poulpy_ckks::Quad,
    encoder = poulpy_cpu_ref::FFT64ReimTable<poulpy_ckks::Quad>,
    params = poulpy_ckks::test_suite::BASE52_PARAMS_QUAD,
    rotations = super::ATK_ROTATIONS,
);

#[cfg(feature = "enable-rayon")]
ckks_backend_test_suite!(
    mod ntt4x30_rayon_f128,
    backend = crate::NTT4x30NeonRayon,
    scalar = poulpy_ckks::Quad,
    encoder = poulpy_cpu_ref::FFT64ReimTable<poulpy_ckks::Quad>,
    params = poulpy_ckks::test_suite::BASE52_PARAMS_QUAD,
    rotations = super::ATK_ROTATIONS,
);

// Paired OEP validation. Serial backends validate against portable implementations;
// Rayon backends validate against their serial counterparts.
poulpy_ckks::ckks_parity_test_suite! {
    mod ckks_parity_fft64neon_f64,
    backend_ref = poulpy_cpu_ref::FFT64Ref,
    backend_test = crate::FFT64Neon,
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
    mod ckks_parity_fft64neon_quad,
    backend_ref = poulpy_cpu_ref::FFT64Ref,
    backend_test = crate::FFT64Neon,
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
    mod ckks_parity_fft64neon_encryption,
    backend_ref = poulpy_cpu_ref::test_suite::ControlledSamplingFFT64Ref,
    backend_test = crate::FFT64Neon,
    params = poulpy_ckks::test_suite::CKKSTestParams { n: 64, hw: 48, base2k: 12, ..poulpy_ckks::test_suite::BASE19_PARAMS_F64 },
}

poulpy_ckks::ckks_parity_test_suite! {
    mod ckks_parity_ntt4x30neon_f64,
    backend_ref = poulpy_cpu_ref::NTT4x30Ref,
    backend_test = crate::NTT4x30Neon,
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
    mod ckks_parity_ntt4x30neon_quad,
    backend_ref = poulpy_cpu_ref::NTT4x30Ref,
    backend_test = crate::NTT4x30Neon,
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
    mod ckks_parity_ntt4x30neon_encryption,
    backend_ref = poulpy_cpu_ref::test_suite::ControlledSamplingFFT64Ref,
    backend_test = crate::NTT4x30Neon,
    params = poulpy_ckks::test_suite::CKKSTestParams { n: 64, hw: 48, base2k: 12, ..poulpy_ckks::test_suite::BASE19_PARAMS_F64 },
}

#[cfg(feature = "enable-rayon")]
poulpy_ckks::ckks_parity_test_suite! {
    mod ckks_parity_fft64neonrayon_f64,
    backend_ref = crate::FFT64Neon,
    backend_test = crate::FFT64NeonRayon,
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
    mod ckks_parity_fft64neonrayon_quad,
    backend_ref = crate::FFT64Neon,
    backend_test = crate::FFT64NeonRayon,
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
    mod ckks_parity_fft64neonrayon_encryption,
    backend_ref = crate::FFT64Neon,
    backend_test = crate::FFT64NeonRayon,
    params = poulpy_ckks::test_suite::CKKSTestParams { n: 64, hw: 48, base2k: 12, ..poulpy_ckks::test_suite::BASE19_PARAMS_F64 },
}

#[cfg(feature = "enable-rayon")]
poulpy_ckks::ckks_parity_test_suite! {
    mod ckks_parity_ntt4x30neonrayon_f64,
    backend_ref = crate::NTT4x30Neon,
    backend_test = crate::NTT4x30NeonRayon,
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
    mod ckks_parity_ntt4x30neonrayon_quad,
    backend_ref = crate::NTT4x30Neon,
    backend_test = crate::NTT4x30NeonRayon,
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
    mod ckks_parity_ntt4x30neonrayon_encryption,
    backend_ref = crate::NTT4x30Neon,
    backend_test = crate::NTT4x30NeonRayon,
    params = poulpy_ckks::test_suite::CKKSTestParams { n: 64, hw: 48, base2k: 12, ..poulpy_ckks::test_suite::BASE19_PARAMS_F64 },
}

// Explicit rank-2 contracts. A caller can select another supported rank through params.
poulpy_ckks::ckks_parity_test_suite! {
    mod ckks_parity_fft64neon_rank2,
    backend_ref = poulpy_cpu_ref::FFT64Ref,
    backend_test = crate::FFT64Neon,
    scalar = f64,
    params = poulpy_ckks::test_suite::CKKSTestParams { n: 64, hw: 48, rank: 2, base2k: 19, ..poulpy_ckks::test_suite::BASE19_PARAMS_F64 },
    tests = {
        arithmetic => poulpy_ckks::test_suite::parity::test_arithmetic_parity,
        multiplication => poulpy_ckks::test_suite::parity::test_multiplication_parity,
        automorphism => poulpy_ckks::test_suite::parity::test_automorphism_parity,
    }
}

poulpy_ckks::ckks_encryption_parity_test_suite! {
    mod ckks_parity_fft64neon_encryption_rank2,
    backend_ref = poulpy_cpu_ref::test_suite::ControlledSamplingFFT64Ref,
    backend_test = crate::FFT64Neon,
    params = poulpy_ckks::test_suite::CKKSTestParams { n: 64, hw: 48, rank: 2, base2k: 12, ..poulpy_ckks::test_suite::BASE19_PARAMS_F64 },
}

poulpy_ckks::ckks_parity_test_suite! {
    mod ckks_parity_ntt4x30neon_rank2,
    backend_ref = poulpy_cpu_ref::NTT4x30Ref,
    backend_test = crate::NTT4x30Neon,
    scalar = f64,
    params = poulpy_ckks::test_suite::CKKSTestParams { n: 64, hw: 48, rank: 2, base2k: 52, ..poulpy_ckks::test_suite::BASE19_PARAMS_F64 },
    tests = {
        arithmetic => poulpy_ckks::test_suite::parity::test_arithmetic_parity,
        multiplication => poulpy_ckks::test_suite::parity::test_multiplication_parity,
        automorphism => poulpy_ckks::test_suite::parity::test_automorphism_parity,
    }
}

poulpy_ckks::ckks_encryption_parity_test_suite! {
    mod ckks_parity_ntt4x30neon_encryption_rank2,
    backend_ref = poulpy_cpu_ref::test_suite::ControlledSamplingFFT64Ref,
    backend_test = crate::NTT4x30Neon,
    params = poulpy_ckks::test_suite::CKKSTestParams { n: 64, hw: 48, rank: 2, base2k: 12, ..poulpy_ckks::test_suite::BASE19_PARAMS_F64 },
}

#[cfg(feature = "enable-rayon")]
poulpy_ckks::ckks_parity_test_suite! {
    mod ckks_parity_fft64neonrayon_rank2,
    backend_ref = crate::FFT64Neon,
    backend_test = crate::FFT64NeonRayon,
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
    mod ckks_parity_fft64neonrayon_encryption_rank2,
    backend_ref = crate::FFT64Neon,
    backend_test = crate::FFT64NeonRayon,
    params = poulpy_ckks::test_suite::CKKSTestParams { n: 64, hw: 48, rank: 2, base2k: 12, ..poulpy_ckks::test_suite::BASE19_PARAMS_F64 },
}

#[cfg(feature = "enable-rayon")]
poulpy_ckks::ckks_parity_test_suite! {
    mod ckks_parity_ntt4x30neonrayon_rank2,
    backend_ref = crate::NTT4x30Neon,
    backend_test = crate::NTT4x30NeonRayon,
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
    mod ckks_parity_ntt4x30neonrayon_encryption_rank2,
    backend_ref = crate::NTT4x30Neon,
    backend_test = crate::NTT4x30NeonRayon,
    params = poulpy_ckks::test_suite::CKKSTestParams { n: 64, hw: 48, rank: 2, base2k: 12, ..poulpy_ckks::test_suite::BASE19_PARAMS_F64 },
}
