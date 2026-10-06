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
    encoder = poulpy_cpu_portable::FFT64ReimTable<poulpy_ckks::Quad>,
    params = poulpy_ckks::test_suite::BASE52_PARAMS_QUAD,
    rotations = super::ATK_ROTATIONS,
);

#[cfg(feature = "enable-rayon")]
ckks_backend_test_suite!(
    mod ntt4x30_rayon_f128,
    backend = crate::NTT4x30NeonRayon,
    scalar = poulpy_ckks::Quad,
    encoder = poulpy_cpu_portable::FFT64ReimTable<poulpy_ckks::Quad>,
    params = poulpy_ckks::test_suite::BASE52_PARAMS_QUAD,
    rotations = super::ATK_ROTATIONS,
);

// Paired OEP validation. Serial backends validate against portable implementations;
// Rayon backends validate against their serial counterparts.
poulpy_ckks::ckks_parity_test_suite! {
    mod ckks_parity_fft64neon_f64,
    backend_ref = poulpy_cpu_portable::FFT64Portable,
    backend_test = crate::FFT64Neon,
    scalar = f64,
    params = poulpy_ckks::test_suite::CKKSTestParams { n: 64, hw: 48, base2k: 19, ..poulpy_ckks::test_suite::BASE19_PARAMS_F64 },
    tests = {
        arithmetic => poulpy_ckks::test_suite::parity::test_arithmetic_parity,
        imag => poulpy_ckks::test_suite::parity::test_imag_parity,
        multiplication => poulpy_ckks::test_suite::parity::test_multiplication_parity,
        rotate => poulpy_ckks::test_suite::parity::test_rotate_parity,
        conjugate => poulpy_ckks::test_suite::parity::test_conjugate_parity,
        plaintext => poulpy_ckks::test_suite::parity::test_plaintext_parity,
        encoding => poulpy_ckks::test_suite::parity::test_encoding_parity,
        #[ignore = "encoding is not bit-exact across backends yet"]
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
    mod ckks_parity_fft64neon_quad,
    backend_ref = poulpy_cpu_portable::FFT64Portable,
    backend_test = crate::FFT64Neon,
    scalar = poulpy_ckks::Quad,
    params = poulpy_ckks::test_suite::CKKSTestParams { n: 64, hw: 48, base2k: 19, ..poulpy_ckks::test_suite::BASE52_PARAMS_QUAD },
    tests = {
        arithmetic => poulpy_ckks::test_suite::parity::test_arithmetic_parity,
        imag => poulpy_ckks::test_suite::parity::test_imag_parity,
        multiplication => poulpy_ckks::test_suite::parity::test_multiplication_parity,
        rotate => poulpy_ckks::test_suite::parity::test_rotate_parity,
        conjugate => poulpy_ckks::test_suite::parity::test_conjugate_parity,
        plaintext => poulpy_ckks::test_suite::parity::test_plaintext_parity,
        encoding => poulpy_ckks::test_suite::parity::test_encoding_parity,
        #[ignore = "encoding is not bit-exact across backends yet"]
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

poulpy_ckks::ckks_encryption_parity_test_suite! {
    mod ckks_parity_fft64neon_encryption,
    backend_ref = poulpy_cpu_portable::test_suite::ControlledSamplingFFT64Portable,
    backend_test = crate::FFT64Neon,
    params = poulpy_ckks::test_suite::CKKSTestParams { n: 64, hw: 48, base2k: 12, ..poulpy_ckks::test_suite::BASE19_PARAMS_F64 },
}

poulpy_ckks::ckks_parity_test_suite! {
    mod ckks_parity_ntt4x30neon_f64,
    backend_ref = poulpy_cpu_portable::NTT4x30Portable,
    backend_test = crate::NTT4x30Neon,
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
        #[ignore = "encoding is not bit-exact across backends yet"]
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
    mod ckks_parity_ntt4x30neon_quad,
    backend_ref = poulpy_cpu_portable::NTT4x30Portable,
    backend_test = crate::NTT4x30Neon,
    scalar = poulpy_ckks::Quad,
    params = poulpy_ckks::test_suite::CKKSTestParams { n: 64, hw: 48, base2k: 52, ..poulpy_ckks::test_suite::BASE52_PARAMS_QUAD },
    tests = {
        arithmetic => poulpy_ckks::test_suite::parity::test_arithmetic_parity,
        imag => poulpy_ckks::test_suite::parity::test_imag_parity,
        multiplication => poulpy_ckks::test_suite::parity::test_multiplication_parity,
        rotate => poulpy_ckks::test_suite::parity::test_rotate_parity,
        conjugate => poulpy_ckks::test_suite::parity::test_conjugate_parity,
        plaintext => poulpy_ckks::test_suite::parity::test_plaintext_parity,
        encoding => poulpy_ckks::test_suite::parity::test_encoding_parity,
        #[ignore = "encoding is not bit-exact across backends yet"]
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

poulpy_ckks::ckks_encryption_parity_test_suite! {
    mod ckks_parity_ntt4x30neon_encryption,
    backend_ref = poulpy_cpu_portable::test_suite::ControlledSamplingFFT64Portable,
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
        imag => poulpy_ckks::test_suite::parity::test_imag_parity,
        multiplication => poulpy_ckks::test_suite::parity::test_multiplication_parity,
        rotate => poulpy_ckks::test_suite::parity::test_rotate_parity,
        conjugate => poulpy_ckks::test_suite::parity::test_conjugate_parity,
        plaintext => poulpy_ckks::test_suite::parity::test_plaintext_parity,
        encoding => poulpy_ckks::test_suite::parity::test_encoding_parity,
        #[ignore = "encoding is not bit-exact across backends yet"]
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

#[cfg(feature = "enable-rayon")]
poulpy_ckks::ckks_parity_test_suite! {
    mod ckks_parity_fft64neonrayon_quad,
    backend_ref = crate::FFT64Neon,
    backend_test = crate::FFT64NeonRayon,
    scalar = poulpy_ckks::Quad,
    params = poulpy_ckks::test_suite::CKKSTestParams { n: 64, hw: 48, base2k: 19, ..poulpy_ckks::test_suite::BASE52_PARAMS_QUAD },
    tests = {
        arithmetic => poulpy_ckks::test_suite::parity::test_arithmetic_parity,
        imag => poulpy_ckks::test_suite::parity::test_imag_parity,
        multiplication => poulpy_ckks::test_suite::parity::test_multiplication_parity,
        rotate => poulpy_ckks::test_suite::parity::test_rotate_parity,
        conjugate => poulpy_ckks::test_suite::parity::test_conjugate_parity,
        plaintext => poulpy_ckks::test_suite::parity::test_plaintext_parity,
        encoding => poulpy_ckks::test_suite::parity::test_encoding_parity,
        #[ignore = "encoding is not bit-exact across backends yet"]
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
        imag => poulpy_ckks::test_suite::parity::test_imag_parity,
        multiplication => poulpy_ckks::test_suite::parity::test_multiplication_parity,
        rotate => poulpy_ckks::test_suite::parity::test_rotate_parity,
        conjugate => poulpy_ckks::test_suite::parity::test_conjugate_parity,
        plaintext => poulpy_ckks::test_suite::parity::test_plaintext_parity,
        encoding => poulpy_ckks::test_suite::parity::test_encoding_parity,
        #[ignore = "encoding is not bit-exact across backends yet"]
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

#[cfg(feature = "enable-rayon")]
poulpy_ckks::ckks_parity_test_suite! {
    mod ckks_parity_ntt4x30neonrayon_quad,
    backend_ref = crate::NTT4x30Neon,
    backend_test = crate::NTT4x30NeonRayon,
    scalar = poulpy_ckks::Quad,
    params = poulpy_ckks::test_suite::CKKSTestParams { n: 64, hw: 48, base2k: 52, ..poulpy_ckks::test_suite::BASE52_PARAMS_QUAD },
    tests = {
        arithmetic => poulpy_ckks::test_suite::parity::test_arithmetic_parity,
        imag => poulpy_ckks::test_suite::parity::test_imag_parity,
        multiplication => poulpy_ckks::test_suite::parity::test_multiplication_parity,
        rotate => poulpy_ckks::test_suite::parity::test_rotate_parity,
        conjugate => poulpy_ckks::test_suite::parity::test_conjugate_parity,
        plaintext => poulpy_ckks::test_suite::parity::test_plaintext_parity,
        encoding => poulpy_ckks::test_suite::parity::test_encoding_parity,
        #[ignore = "encoding is not bit-exact across backends yet"]
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
    backend_ref = poulpy_cpu_portable::FFT64Portable,
    backend_test = crate::FFT64Neon,
    scalar = f64,
    params = poulpy_ckks::test_suite::CKKSTestParams { n: 64, hw: 48, rank: 2, base2k: 19, ..poulpy_ckks::test_suite::BASE19_PARAMS_F64 },
    tests = {
        arithmetic => poulpy_ckks::test_suite::parity::test_arithmetic_parity,
        imag => poulpy_ckks::test_suite::parity::test_imag_parity,
        multiplication => poulpy_ckks::test_suite::parity::test_multiplication_parity,
        rotate => poulpy_ckks::test_suite::parity::test_rotate_parity,
        conjugate => poulpy_ckks::test_suite::parity::test_conjugate_parity,
    }
}

poulpy_ckks::ckks_encryption_parity_test_suite! {
    mod ckks_parity_fft64neon_encryption_rank2,
    backend_ref = poulpy_cpu_portable::test_suite::ControlledSamplingFFT64Portable,
    backend_test = crate::FFT64Neon,
    params = poulpy_ckks::test_suite::CKKSTestParams { n: 64, hw: 48, rank: 2, base2k: 12, ..poulpy_ckks::test_suite::BASE19_PARAMS_F64 },
}

poulpy_ckks::ckks_parity_test_suite! {
    mod ckks_parity_ntt4x30neon_rank2,
    backend_ref = poulpy_cpu_portable::NTT4x30Portable,
    backend_test = crate::NTT4x30Neon,
    scalar = f64,
    params = poulpy_ckks::test_suite::CKKSTestParams { n: 64, hw: 48, rank: 2, base2k: 52, ..poulpy_ckks::test_suite::BASE19_PARAMS_F64 },
    tests = {
        arithmetic => poulpy_ckks::test_suite::parity::test_arithmetic_parity,
        imag => poulpy_ckks::test_suite::parity::test_imag_parity,
        multiplication => poulpy_ckks::test_suite::parity::test_multiplication_parity,
        rotate => poulpy_ckks::test_suite::parity::test_rotate_parity,
        conjugate => poulpy_ckks::test_suite::parity::test_conjugate_parity,
    }
}

poulpy_ckks::ckks_encryption_parity_test_suite! {
    mod ckks_parity_ntt4x30neon_encryption_rank2,
    backend_ref = poulpy_cpu_portable::test_suite::ControlledSamplingFFT64Portable,
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
        imag => poulpy_ckks::test_suite::parity::test_imag_parity,
        multiplication => poulpy_ckks::test_suite::parity::test_multiplication_parity,
        rotate => poulpy_ckks::test_suite::parity::test_rotate_parity,
        conjugate => poulpy_ckks::test_suite::parity::test_conjugate_parity,
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
        imag => poulpy_ckks::test_suite::parity::test_imag_parity,
        multiplication => poulpy_ckks::test_suite::parity::test_multiplication_parity,
        rotate => poulpy_ckks::test_suite::parity::test_rotate_parity,
        conjugate => poulpy_ckks::test_suite::parity::test_conjugate_parity,
    }
}

#[cfg(feature = "enable-rayon")]
poulpy_ckks::ckks_encryption_parity_test_suite! {
    mod ckks_parity_ntt4x30neonrayon_encryption_rank2,
    backend_ref = crate::NTT4x30Neon,
    backend_test = crate::NTT4x30NeonRayon,
    params = poulpy_ckks::test_suite::CKKSTestParams { n: 64, hw: 48, rank: 2, base2k: 12, ..poulpy_ckks::test_suite::BASE19_PARAMS_F64 },
}

// Conjugate-invariant backends run the ring-generic suites against the same pairs.
poulpy_ckks::ckks_parity_test_suite! {
    mod ckks_parity_ci_fft64neon_f64,
    backend_ref = poulpy_cpu_portable::FFT64CIPortable,
    backend_test = crate::FFT64CINeon,
    scalar = f64,
    params = poulpy_ckks::test_suite::CKKSTestParams { n: 64, hw: 48, base2k: 19, ..poulpy_ckks::test_suite::BASE19_PARAMS_F64 },
    tests = {
        arithmetic => poulpy_ckks::test_suite::parity::test_arithmetic_parity,
        multiplication => poulpy_ckks::test_suite::parity::test_multiplication_parity,
        rotate => poulpy_ckks::test_suite::parity::test_rotate_parity,
        plaintext => poulpy_ckks::test_suite::parity::test_plaintext_parity,
        encoding => poulpy_ckks::test_suite::parity::test_encoding_parity,
        #[ignore = "encoding is not bit-exact across backends yet"]
        slot_encoding => poulpy_ckks::test_suite::parity::test_slot_encoding_parity,
        real_polynomial => poulpy_ckks::test_suite::parity::test_real_polynomial_parity,
    }
}

poulpy_ckks::ckks_parity_test_suite! {
    mod ckks_parity_ci_fft64neon_quad,
    backend_ref = poulpy_cpu_portable::FFT64CIPortable,
    backend_test = crate::FFT64CINeon,
    scalar = poulpy_ckks::Quad,
    params = poulpy_ckks::test_suite::CKKSTestParams { n: 64, hw: 48, base2k: 19, ..poulpy_ckks::test_suite::BASE52_PARAMS_QUAD },
    tests = {
        arithmetic => poulpy_ckks::test_suite::parity::test_arithmetic_parity,
        multiplication => poulpy_ckks::test_suite::parity::test_multiplication_parity,
        rotate => poulpy_ckks::test_suite::parity::test_rotate_parity,
        plaintext => poulpy_ckks::test_suite::parity::test_plaintext_parity,
        encoding => poulpy_ckks::test_suite::parity::test_encoding_parity,
        #[ignore = "encoding is not bit-exact across backends yet"]
        slot_encoding => poulpy_ckks::test_suite::parity::test_slot_encoding_parity,
        real_polynomial => poulpy_ckks::test_suite::parity::test_real_polynomial_parity,
    }
}

poulpy_ckks::ckks_encryption_parity_test_suite! {
    mod ckks_parity_ci_fft64neon_encryption,
    backend_ref = poulpy_cpu_portable::test_suite::ControlledSamplingFFT64CIPortable,
    backend_test = crate::FFT64CINeon,
    params = poulpy_ckks::test_suite::CKKSTestParams { n: 64, hw: 48, base2k: 12, ..poulpy_ckks::test_suite::BASE19_PARAMS_F64 },
}

poulpy_ckks::ckks_parity_test_suite! {
    mod ckks_parity_ci_ntt4x30neon_f64,
    backend_ref = poulpy_cpu_portable::NTT4x30CIPortable,
    backend_test = crate::NTT4x30CINeon,
    scalar = f64,
    params = poulpy_ckks::test_suite::CKKSTestParams { n: 64, hw: 48, base2k: 52, ..poulpy_ckks::test_suite::BASE52_PARAMS_F64 },
    tests = {
        arithmetic => poulpy_ckks::test_suite::parity::test_arithmetic_parity,
        multiplication => poulpy_ckks::test_suite::parity::test_multiplication_parity,
        rotate => poulpy_ckks::test_suite::parity::test_rotate_parity,
        plaintext => poulpy_ckks::test_suite::parity::test_plaintext_parity,
        encoding => poulpy_ckks::test_suite::parity::test_encoding_parity,
        #[ignore = "encoding is not bit-exact across backends yet"]
        slot_encoding => poulpy_ckks::test_suite::parity::test_slot_encoding_parity,
        real_polynomial => poulpy_ckks::test_suite::parity::test_real_polynomial_parity,
    }
}

poulpy_ckks::ckks_parity_test_suite! {
    mod ckks_parity_ci_ntt4x30neon_quad,
    backend_ref = poulpy_cpu_portable::NTT4x30CIPortable,
    backend_test = crate::NTT4x30CINeon,
    scalar = poulpy_ckks::Quad,
    params = poulpy_ckks::test_suite::CKKSTestParams { n: 64, hw: 48, base2k: 52, ..poulpy_ckks::test_suite::BASE52_PARAMS_QUAD },
    tests = {
        arithmetic => poulpy_ckks::test_suite::parity::test_arithmetic_parity,
        multiplication => poulpy_ckks::test_suite::parity::test_multiplication_parity,
        rotate => poulpy_ckks::test_suite::parity::test_rotate_parity,
        plaintext => poulpy_ckks::test_suite::parity::test_plaintext_parity,
        encoding => poulpy_ckks::test_suite::parity::test_encoding_parity,
        #[ignore = "encoding is not bit-exact across backends yet"]
        slot_encoding => poulpy_ckks::test_suite::parity::test_slot_encoding_parity,
        real_polynomial => poulpy_ckks::test_suite::parity::test_real_polynomial_parity,
    }
}

poulpy_ckks::ckks_encryption_parity_test_suite! {
    mod ckks_parity_ci_ntt4x30neon_encryption,
    backend_ref = poulpy_cpu_portable::test_suite::ControlledSamplingFFT64CIPortable,
    backend_test = crate::NTT4x30CINeon,
    params = poulpy_ckks::test_suite::CKKSTestParams { n: 64, hw: 48, base2k: 12, ..poulpy_ckks::test_suite::BASE19_PARAMS_F64 },
}

#[cfg(feature = "enable-rayon")]
poulpy_ckks::ckks_parity_test_suite! {
    mod ckks_parity_ci_fft64neonrayon_f64,
    backend_ref = crate::FFT64CINeon,
    backend_test = crate::FFT64CINeonRayon,
    scalar = f64,
    params = poulpy_ckks::test_suite::CKKSTestParams { n: 64, hw: 48, base2k: 19, ..poulpy_ckks::test_suite::BASE19_PARAMS_F64 },
    tests = {
        arithmetic => poulpy_ckks::test_suite::parity::test_arithmetic_parity,
        multiplication => poulpy_ckks::test_suite::parity::test_multiplication_parity,
        rotate => poulpy_ckks::test_suite::parity::test_rotate_parity,
        plaintext => poulpy_ckks::test_suite::parity::test_plaintext_parity,
        encoding => poulpy_ckks::test_suite::parity::test_encoding_parity,
        #[ignore = "encoding is not bit-exact across backends yet"]
        slot_encoding => poulpy_ckks::test_suite::parity::test_slot_encoding_parity,
        real_polynomial => poulpy_ckks::test_suite::parity::test_real_polynomial_parity,
    }
}

#[cfg(feature = "enable-rayon")]
poulpy_ckks::ckks_parity_test_suite! {
    mod ckks_parity_ci_fft64neonrayon_quad,
    backend_ref = crate::FFT64CINeon,
    backend_test = crate::FFT64CINeonRayon,
    scalar = poulpy_ckks::Quad,
    params = poulpy_ckks::test_suite::CKKSTestParams { n: 64, hw: 48, base2k: 19, ..poulpy_ckks::test_suite::BASE52_PARAMS_QUAD },
    tests = {
        arithmetic => poulpy_ckks::test_suite::parity::test_arithmetic_parity,
        multiplication => poulpy_ckks::test_suite::parity::test_multiplication_parity,
        rotate => poulpy_ckks::test_suite::parity::test_rotate_parity,
        plaintext => poulpy_ckks::test_suite::parity::test_plaintext_parity,
        encoding => poulpy_ckks::test_suite::parity::test_encoding_parity,
        #[ignore = "encoding is not bit-exact across backends yet"]
        slot_encoding => poulpy_ckks::test_suite::parity::test_slot_encoding_parity,
        real_polynomial => poulpy_ckks::test_suite::parity::test_real_polynomial_parity,
    }
}

#[cfg(feature = "enable-rayon")]
poulpy_ckks::ckks_encryption_parity_test_suite! {
    mod ckks_parity_ci_fft64neonrayon_encryption,
    backend_ref = crate::FFT64CINeon,
    backend_test = crate::FFT64CINeonRayon,
    params = poulpy_ckks::test_suite::CKKSTestParams { n: 64, hw: 48, base2k: 12, ..poulpy_ckks::test_suite::BASE19_PARAMS_F64 },
}

#[cfg(feature = "enable-rayon")]
poulpy_ckks::ckks_parity_test_suite! {
    mod ckks_parity_ci_ntt4x30neonrayon_f64,
    backend_ref = crate::NTT4x30CINeon,
    backend_test = crate::NTT4x30CINeonRayon,
    scalar = f64,
    params = poulpy_ckks::test_suite::CKKSTestParams { n: 64, hw: 48, base2k: 52, ..poulpy_ckks::test_suite::BASE52_PARAMS_F64 },
    tests = {
        arithmetic => poulpy_ckks::test_suite::parity::test_arithmetic_parity,
        multiplication => poulpy_ckks::test_suite::parity::test_multiplication_parity,
        rotate => poulpy_ckks::test_suite::parity::test_rotate_parity,
        plaintext => poulpy_ckks::test_suite::parity::test_plaintext_parity,
        encoding => poulpy_ckks::test_suite::parity::test_encoding_parity,
        #[ignore = "encoding is not bit-exact across backends yet"]
        slot_encoding => poulpy_ckks::test_suite::parity::test_slot_encoding_parity,
        real_polynomial => poulpy_ckks::test_suite::parity::test_real_polynomial_parity,
    }
}

#[cfg(feature = "enable-rayon")]
poulpy_ckks::ckks_parity_test_suite! {
    mod ckks_parity_ci_ntt4x30neonrayon_quad,
    backend_ref = crate::NTT4x30CINeon,
    backend_test = crate::NTT4x30CINeonRayon,
    scalar = poulpy_ckks::Quad,
    params = poulpy_ckks::test_suite::CKKSTestParams { n: 64, hw: 48, base2k: 52, ..poulpy_ckks::test_suite::BASE52_PARAMS_QUAD },
    tests = {
        arithmetic => poulpy_ckks::test_suite::parity::test_arithmetic_parity,
        multiplication => poulpy_ckks::test_suite::parity::test_multiplication_parity,
        rotate => poulpy_ckks::test_suite::parity::test_rotate_parity,
        plaintext => poulpy_ckks::test_suite::parity::test_plaintext_parity,
        encoding => poulpy_ckks::test_suite::parity::test_encoding_parity,
        #[ignore = "encoding is not bit-exact across backends yet"]
        slot_encoding => poulpy_ckks::test_suite::parity::test_slot_encoding_parity,
        real_polynomial => poulpy_ckks::test_suite::parity::test_real_polynomial_parity,
    }
}

#[cfg(feature = "enable-rayon")]
poulpy_ckks::ckks_encryption_parity_test_suite! {
    mod ckks_parity_ci_ntt4x30neonrayon_encryption,
    backend_ref = crate::NTT4x30CINeon,
    backend_test = crate::NTT4x30CINeonRayon,
    params = poulpy_ckks::test_suite::CKKSTestParams { n: 64, hw: 48, base2k: 12, ..poulpy_ckks::test_suite::BASE19_PARAMS_F64 },
}

poulpy_ckks::ckks_parity_test_suite! {
    mod ckks_parity_ci_fft64neon_rank2,
    backend_ref = poulpy_cpu_portable::FFT64CIPortable,
    backend_test = crate::FFT64CINeon,
    scalar = f64,
    params = poulpy_ckks::test_suite::CKKSTestParams { n: 64, hw: 48, rank: 2, base2k: 19, ..poulpy_ckks::test_suite::BASE19_PARAMS_F64 },
    tests = {
        arithmetic => poulpy_ckks::test_suite::parity::test_arithmetic_parity,
        multiplication => poulpy_ckks::test_suite::parity::test_multiplication_parity,
        rotate => poulpy_ckks::test_suite::parity::test_rotate_parity,
    }
}

poulpy_ckks::ckks_encryption_parity_test_suite! {
    mod ckks_parity_ci_fft64neon_encryption_rank2,
    backend_ref = poulpy_cpu_portable::test_suite::ControlledSamplingFFT64CIPortable,
    backend_test = crate::FFT64CINeon,
    params = poulpy_ckks::test_suite::CKKSTestParams { n: 64, hw: 48, rank: 2, base2k: 12, ..poulpy_ckks::test_suite::BASE19_PARAMS_F64 },
}

poulpy_ckks::ckks_parity_test_suite! {
    mod ckks_parity_ci_ntt4x30neon_rank2,
    backend_ref = poulpy_cpu_portable::NTT4x30CIPortable,
    backend_test = crate::NTT4x30CINeon,
    scalar = f64,
    params = poulpy_ckks::test_suite::CKKSTestParams { n: 64, hw: 48, rank: 2, base2k: 52, ..poulpy_ckks::test_suite::BASE19_PARAMS_F64 },
    tests = {
        arithmetic => poulpy_ckks::test_suite::parity::test_arithmetic_parity,
        multiplication => poulpy_ckks::test_suite::parity::test_multiplication_parity,
        rotate => poulpy_ckks::test_suite::parity::test_rotate_parity,
    }
}

poulpy_ckks::ckks_encryption_parity_test_suite! {
    mod ckks_parity_ci_ntt4x30neon_encryption_rank2,
    backend_ref = poulpy_cpu_portable::test_suite::ControlledSamplingFFT64CIPortable,
    backend_test = crate::NTT4x30CINeon,
    params = poulpy_ckks::test_suite::CKKSTestParams { n: 64, hw: 48, rank: 2, base2k: 12, ..poulpy_ckks::test_suite::BASE19_PARAMS_F64 },
}

#[cfg(feature = "enable-rayon")]
poulpy_ckks::ckks_parity_test_suite! {
    mod ckks_parity_ci_fft64neonrayon_rank2,
    backend_ref = crate::FFT64CINeon,
    backend_test = crate::FFT64CINeonRayon,
    scalar = f64,
    params = poulpy_ckks::test_suite::CKKSTestParams { n: 64, hw: 48, rank: 2, base2k: 19, ..poulpy_ckks::test_suite::BASE19_PARAMS_F64 },
    tests = {
        arithmetic => poulpy_ckks::test_suite::parity::test_arithmetic_parity,
        multiplication => poulpy_ckks::test_suite::parity::test_multiplication_parity,
        rotate => poulpy_ckks::test_suite::parity::test_rotate_parity,
    }
}

#[cfg(feature = "enable-rayon")]
poulpy_ckks::ckks_encryption_parity_test_suite! {
    mod ckks_parity_ci_fft64neonrayon_encryption_rank2,
    backend_ref = crate::FFT64CINeon,
    backend_test = crate::FFT64CINeonRayon,
    params = poulpy_ckks::test_suite::CKKSTestParams { n: 64, hw: 48, rank: 2, base2k: 12, ..poulpy_ckks::test_suite::BASE19_PARAMS_F64 },
}

#[cfg(feature = "enable-rayon")]
poulpy_ckks::ckks_parity_test_suite! {
    mod ckks_parity_ci_ntt4x30neonrayon_rank2,
    backend_ref = crate::NTT4x30CINeon,
    backend_test = crate::NTT4x30CINeonRayon,
    scalar = f64,
    params = poulpy_ckks::test_suite::CKKSTestParams { n: 64, hw: 48, rank: 2, base2k: 52, ..poulpy_ckks::test_suite::BASE19_PARAMS_F64 },
    tests = {
        arithmetic => poulpy_ckks::test_suite::parity::test_arithmetic_parity,
        multiplication => poulpy_ckks::test_suite::parity::test_multiplication_parity,
        rotate => poulpy_ckks::test_suite::parity::test_rotate_parity,
    }
}

#[cfg(feature = "enable-rayon")]
poulpy_ckks::ckks_encryption_parity_test_suite! {
    mod ckks_parity_ci_ntt4x30neonrayon_encryption_rank2,
    backend_ref = crate::NTT4x30CINeon,
    backend_test = crate::NTT4x30CINeonRayon,
    params = poulpy_ckks::test_suite::CKKSTestParams { n: 64, hw: 48, rank: 2, base2k: 12, ..poulpy_ckks::test_suite::BASE19_PARAMS_F64 },
}
