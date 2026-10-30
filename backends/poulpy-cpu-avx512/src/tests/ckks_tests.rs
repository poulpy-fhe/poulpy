use poulpy_ckks::ckks_determinism_test_suite;

#[cfg(feature = "enable-ifma")]
mod bootstrapping_presets {
    use poulpy_ckks::{
        layouts::EvalModRing,
        test_suite::presets::{BOOTSTRAP_FAILURE_BITS, bootstrapping_presets_meet_precision, preset_for_backend},
    };
    use poulpy_core::layouts::LWEInfos;

    #[test]
    fn ifma_radix() {
        let presets = poulpy_ckks::presets::bootstrapping::all().unwrap().into_iter();
        for preset in presets.filter(|preset| preset.plan().eval_mod_ring() == EvalModRing::Standard) {
            // The 126-bit prime product leaves three bits more than NTT4x30's.
            assert_eq!(
                preset.max_backend_base2k::<crate::NTT3x42Ifma, crate::NTT3x42CIIfma>(BOOTSTRAP_FAILURE_BITS),
                56
            );
            let adapted = preset_for_backend::<crate::NTT3x42Ifma, crate::NTT3x42CIIfma>(&preset).unwrap();
            assert_eq!(adapted.base2k(), 56);
            let keys = adapted.keys_layout().rotation_key_layouts();
            assert!(keys.iter().all(|key| key.k().as_usize() <= adapted.max_dense_modulus()));
            let small_key = &adapted.keys_layout().encapsulation.as_ref().unwrap().dense_to_sparse;
            assert!(small_key.k().as_usize() <= adapted.max_sparse_modulus());
        }
    }

    #[test]
    #[ignore = "full logN16 bootstrapping presets"]
    fn ifma() {
        bootstrapping_presets_meet_precision::<crate::NTT3x42Ifma>();
    }

    #[cfg(feature = "enable-rayon")]
    #[test]
    #[ignore = "full logN16 bootstrapping presets"]
    fn ifma_rayon() {
        bootstrapping_presets_meet_precision::<crate::NTT3x42IfmaRayon>();
    }
}

ckks_determinism_test_suite!(mod ckks_determinism_fft64avx512, backend = crate::FFT64Avx512, f32);

ckks_determinism_test_suite!(mod ckks_determinism_ntt4x30avx512, backend = crate::NTT4x30Avx512, f32);

#[cfg(feature = "enable-ifma")]
ckks_determinism_test_suite!(mod ckks_determinism_ntt3x42ifma, backend = crate::NTT3x42Ifma, f32);

#[cfg(feature = "enable-rayon")]
ckks_determinism_test_suite!(mod ckks_determinism_fft64avx512rayon, backend = crate::FFT64Avx512Rayon, f32);

#[cfg(feature = "enable-rayon")]
ckks_determinism_test_suite!(mod ckks_determinism_ntt4x30avx512rayon, backend = crate::NTT4x30Avx512Rayon, f32);

#[cfg(all(feature = "enable-ifma", feature = "enable-rayon"))]
ckks_determinism_test_suite!(mod ckks_determinism_ntt3x42ifmarayon, backend = crate::NTT3x42IfmaRayon, f32);

// Paired OEP validation. Serial backends follow AVX -> AVX-512 -> IFMA.
// Rayon backends validate against their serial counterparts.
poulpy_ckks::ckks_parity_test_suite! {
    mod ckks_parity_fft64avx512_f64,
    backend_ref = poulpy_cpu_avx::FFT64Avx,
    backend_test = crate::FFT64Avx512,
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
    mod ckks_parity_fft64avx512_quad,
    backend_ref = poulpy_cpu_avx::FFT64Avx,
    backend_test = crate::FFT64Avx512,
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
    mod ckks_parity_fft64avx512_encryption,
    backend_ref = poulpy_cpu_avx::FFT64Avx,
    backend_test = crate::FFT64Avx512,
    params = poulpy_ckks::test_suite::CKKSTestParams { n: 64, hw: 48, base2k: 12, ..poulpy_ckks::test_suite::BASE19_PARAMS_F64 },
}

poulpy_ckks::ckks_parity_test_suite! {
    mod ckks_parity_ntt4x30avx512_f64,
    backend_ref = poulpy_cpu_avx::NTT4x30Avx,
    backend_test = crate::NTT4x30Avx512,
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
    mod ckks_parity_ntt4x30avx512_quad,
    backend_ref = poulpy_cpu_avx::NTT4x30Avx,
    backend_test = crate::NTT4x30Avx512,
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
    mod ckks_parity_ntt4x30avx512_encryption,
    backend_ref = poulpy_cpu_avx::NTT4x30Avx,
    backend_test = crate::NTT4x30Avx512,
    params = poulpy_ckks::test_suite::CKKSTestParams { n: 64, hw: 48, base2k: 12, ..poulpy_ckks::test_suite::BASE19_PARAMS_F64 },
}

#[cfg(feature = "enable-ifma")]
poulpy_ckks::ckks_parity_test_suite! {
    mod ckks_parity_ntt3x42ifma_f64,
    backend_ref = crate::NTT4x30Avx512,
    backend_test = crate::NTT3x42Ifma,
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

#[cfg(feature = "enable-ifma")]
poulpy_ckks::ckks_parity_test_suite! {
    mod ckks_parity_ntt3x42ifma_quad,
    backend_ref = crate::NTT4x30Avx512,
    backend_test = crate::NTT3x42Ifma,
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

#[cfg(feature = "enable-ifma")]
poulpy_ckks::ckks_encryption_parity_test_suite! {
    mod ckks_parity_ntt3x42ifma_encryption,
    backend_ref = crate::NTT4x30Avx512,
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

#[cfg(feature = "enable-rayon")]
poulpy_ckks::ckks_parity_test_suite! {
    mod ckks_parity_fft64avx512rayon_quad,
    backend_ref = crate::FFT64Avx512,
    backend_test = crate::FFT64Avx512Rayon,
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

#[cfg(feature = "enable-rayon")]
poulpy_ckks::ckks_parity_test_suite! {
    mod ckks_parity_ntt4x30avx512rayon_quad,
    backend_ref = crate::NTT4x30Avx512,
    backend_test = crate::NTT4x30Avx512Rayon,
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

#[cfg(all(feature = "enable-ifma", feature = "enable-rayon"))]
poulpy_ckks::ckks_parity_test_suite! {
    mod ckks_parity_ntt3x42ifmarayon_quad,
    backend_ref = crate::NTT3x42Ifma,
    backend_test = crate::NTT3x42IfmaRayon,
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
    backend_ref = poulpy_cpu_avx::FFT64Avx,
    backend_test = crate::FFT64Avx512,
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
    mod ckks_parity_fft64avx512_encryption_rank2,
    backend_ref = poulpy_cpu_avx::FFT64Avx,
    backend_test = crate::FFT64Avx512,
    params = poulpy_ckks::test_suite::CKKSTestParams { n: 64, hw: 48, rank: 2, base2k: 12, ..poulpy_ckks::test_suite::BASE19_PARAMS_F64 },
}

poulpy_ckks::ckks_parity_test_suite! {
    mod ckks_parity_ntt4x30avx512_rank2,
    backend_ref = poulpy_cpu_avx::NTT4x30Avx,
    backend_test = crate::NTT4x30Avx512,
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
    mod ckks_parity_ntt4x30avx512_encryption_rank2,
    backend_ref = poulpy_cpu_avx::NTT4x30Avx,
    backend_test = crate::NTT4x30Avx512,
    params = poulpy_ckks::test_suite::CKKSTestParams { n: 64, hw: 48, rank: 2, base2k: 12, ..poulpy_ckks::test_suite::BASE19_PARAMS_F64 },
}

#[cfg(feature = "enable-ifma")]
poulpy_ckks::ckks_parity_test_suite! {
    mod ckks_parity_ntt3x42ifma_rank2,
    backend_ref = crate::NTT4x30Avx512,
    backend_test = crate::NTT3x42Ifma,
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

#[cfg(feature = "enable-ifma")]
poulpy_ckks::ckks_encryption_parity_test_suite! {
    mod ckks_parity_ntt3x42ifma_encryption_rank2,
    backend_ref = crate::NTT4x30Avx512,
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
        imag => poulpy_ckks::test_suite::parity::test_imag_parity,
        multiplication => poulpy_ckks::test_suite::parity::test_multiplication_parity,
        rotate => poulpy_ckks::test_suite::parity::test_rotate_parity,
        conjugate => poulpy_ckks::test_suite::parity::test_conjugate_parity,
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
        imag => poulpy_ckks::test_suite::parity::test_imag_parity,
        multiplication => poulpy_ckks::test_suite::parity::test_multiplication_parity,
        rotate => poulpy_ckks::test_suite::parity::test_rotate_parity,
        conjugate => poulpy_ckks::test_suite::parity::test_conjugate_parity,
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
        imag => poulpy_ckks::test_suite::parity::test_imag_parity,
        multiplication => poulpy_ckks::test_suite::parity::test_multiplication_parity,
        rotate => poulpy_ckks::test_suite::parity::test_rotate_parity,
        conjugate => poulpy_ckks::test_suite::parity::test_conjugate_parity,
    }
}

#[cfg(all(feature = "enable-ifma", feature = "enable-rayon"))]
poulpy_ckks::ckks_encryption_parity_test_suite! {
    mod ckks_parity_ntt3x42ifmarayon_encryption_rank2,
    backend_ref = crate::NTT3x42Ifma,
    backend_test = crate::NTT3x42IfmaRayon,
    params = poulpy_ckks::test_suite::CKKSTestParams { n: 64, hw: 48, rank: 2, base2k: 12, ..poulpy_ckks::test_suite::BASE19_PARAMS_F64 },
}

// Conjugate-invariant backends run the ring-generic suites against the same pairs.
poulpy_ckks::ckks_parity_test_suite! {
    mod ckks_parity_ci_fft64avx512_f64,
    backend_ref = poulpy_cpu_avx::FFT64CIAvx,
    backend_test = crate::FFT64CIAvx512,
    scalar = f64,
    params = poulpy_ckks::test_suite::CKKSTestParams { n: 64, hw: 48, base2k: 19, ..poulpy_ckks::test_suite::BASE19_PARAMS_F64 },
    tests = {
        arithmetic => poulpy_ckks::test_suite::parity::test_arithmetic_parity,
        multiplication => poulpy_ckks::test_suite::parity::test_multiplication_parity,
        rotate => poulpy_ckks::test_suite::parity::test_rotate_parity,
        plaintext => poulpy_ckks::test_suite::parity::test_plaintext_parity,
        encoding => poulpy_ckks::test_suite::parity::test_encoding_parity,
        slot_encoding => poulpy_ckks::test_suite::parity::test_slot_encoding_parity,
        real_polynomial => poulpy_ckks::test_suite::parity::test_real_polynomial_parity,
    }
}

poulpy_ckks::ckks_parity_test_suite! {
    mod ckks_parity_ci_fft64avx512_quad,
    backend_ref = poulpy_cpu_avx::FFT64CIAvx,
    backend_test = crate::FFT64CIAvx512,
    scalar = poulpy_ckks::Quad,
    params = poulpy_ckks::test_suite::CKKSTestParams { n: 64, hw: 48, base2k: 19, ..poulpy_ckks::test_suite::BASE52_PARAMS_QUAD },
    tests = {
        arithmetic => poulpy_ckks::test_suite::parity::test_arithmetic_parity,
        multiplication => poulpy_ckks::test_suite::parity::test_multiplication_parity,
        rotate => poulpy_ckks::test_suite::parity::test_rotate_parity,
        plaintext => poulpy_ckks::test_suite::parity::test_plaintext_parity,
        encoding => poulpy_ckks::test_suite::parity::test_encoding_parity,
        slot_encoding => poulpy_ckks::test_suite::parity::test_slot_encoding_parity,
        real_polynomial => poulpy_ckks::test_suite::parity::test_real_polynomial_parity,
    }
}

poulpy_ckks::ckks_encryption_parity_test_suite! {
    mod ckks_parity_ci_fft64avx512_encryption,
    backend_ref = poulpy_cpu_avx::FFT64CIAvx,
    backend_test = crate::FFT64CIAvx512,
    params = poulpy_ckks::test_suite::CKKSTestParams { n: 64, hw: 48, base2k: 12, ..poulpy_ckks::test_suite::BASE19_PARAMS_F64 },
}

poulpy_ckks::ckks_parity_test_suite! {
    mod ckks_parity_ci_ntt4x30avx512_f64,
    backend_ref = poulpy_cpu_avx::NTT4x30CIAvx,
    backend_test = crate::NTT4x30CIAvx512,
    scalar = f64,
    params = poulpy_ckks::test_suite::CKKSTestParams { n: 64, hw: 48, base2k: 52, ..poulpy_ckks::test_suite::BASE52_PARAMS_F64 },
    tests = {
        arithmetic => poulpy_ckks::test_suite::parity::test_arithmetic_parity,
        multiplication => poulpy_ckks::test_suite::parity::test_multiplication_parity,
        rotate => poulpy_ckks::test_suite::parity::test_rotate_parity,
        plaintext => poulpy_ckks::test_suite::parity::test_plaintext_parity,
        encoding => poulpy_ckks::test_suite::parity::test_encoding_parity,
        slot_encoding => poulpy_ckks::test_suite::parity::test_slot_encoding_parity,
        real_polynomial => poulpy_ckks::test_suite::parity::test_real_polynomial_parity,
    }
}

poulpy_ckks::ckks_parity_test_suite! {
    mod ckks_parity_ci_ntt4x30avx512_quad,
    backend_ref = poulpy_cpu_avx::NTT4x30CIAvx,
    backend_test = crate::NTT4x30CIAvx512,
    scalar = poulpy_ckks::Quad,
    params = poulpy_ckks::test_suite::CKKSTestParams { n: 64, hw: 48, base2k: 52, ..poulpy_ckks::test_suite::BASE52_PARAMS_QUAD },
    tests = {
        arithmetic => poulpy_ckks::test_suite::parity::test_arithmetic_parity,
        multiplication => poulpy_ckks::test_suite::parity::test_multiplication_parity,
        rotate => poulpy_ckks::test_suite::parity::test_rotate_parity,
        plaintext => poulpy_ckks::test_suite::parity::test_plaintext_parity,
        encoding => poulpy_ckks::test_suite::parity::test_encoding_parity,
        slot_encoding => poulpy_ckks::test_suite::parity::test_slot_encoding_parity,
        real_polynomial => poulpy_ckks::test_suite::parity::test_real_polynomial_parity,
    }
}

poulpy_ckks::ckks_encryption_parity_test_suite! {
    mod ckks_parity_ci_ntt4x30avx512_encryption,
    backend_ref = poulpy_cpu_avx::NTT4x30CIAvx,
    backend_test = crate::NTT4x30CIAvx512,
    params = poulpy_ckks::test_suite::CKKSTestParams { n: 64, hw: 48, base2k: 12, ..poulpy_ckks::test_suite::BASE19_PARAMS_F64 },
}

#[cfg(feature = "enable-ifma")]
poulpy_ckks::ckks_parity_test_suite! {
    mod ckks_parity_ci_ntt3x42ifma_f64,
    backend_ref = crate::NTT4x30CIAvx512,
    backend_test = crate::NTT3x42CIIfma,
    scalar = f64,
    params = poulpy_ckks::test_suite::CKKSTestParams { n: 64, hw: 48, base2k: 52, ..poulpy_ckks::test_suite::BASE52_PARAMS_F64 },
    tests = {
        arithmetic => poulpy_ckks::test_suite::parity::test_arithmetic_parity,
        multiplication => poulpy_ckks::test_suite::parity::test_multiplication_parity,
        rotate => poulpy_ckks::test_suite::parity::test_rotate_parity,
        plaintext => poulpy_ckks::test_suite::parity::test_plaintext_parity,
        encoding => poulpy_ckks::test_suite::parity::test_encoding_parity,
        slot_encoding => poulpy_ckks::test_suite::parity::test_slot_encoding_parity,
        real_polynomial => poulpy_ckks::test_suite::parity::test_real_polynomial_parity,
    }
}

#[cfg(feature = "enable-ifma")]
poulpy_ckks::ckks_parity_test_suite! {
    mod ckks_parity_ci_ntt3x42ifma_quad,
    backend_ref = crate::NTT4x30CIAvx512,
    backend_test = crate::NTT3x42CIIfma,
    scalar = poulpy_ckks::Quad,
    params = poulpy_ckks::test_suite::CKKSTestParams { n: 64, hw: 48, base2k: 52, ..poulpy_ckks::test_suite::BASE52_PARAMS_QUAD },
    tests = {
        arithmetic => poulpy_ckks::test_suite::parity::test_arithmetic_parity,
        multiplication => poulpy_ckks::test_suite::parity::test_multiplication_parity,
        rotate => poulpy_ckks::test_suite::parity::test_rotate_parity,
        plaintext => poulpy_ckks::test_suite::parity::test_plaintext_parity,
        encoding => poulpy_ckks::test_suite::parity::test_encoding_parity,
        slot_encoding => poulpy_ckks::test_suite::parity::test_slot_encoding_parity,
        real_polynomial => poulpy_ckks::test_suite::parity::test_real_polynomial_parity,
    }
}

#[cfg(feature = "enable-ifma")]
poulpy_ckks::ckks_encryption_parity_test_suite! {
    mod ckks_parity_ci_ntt3x42ifma_encryption,
    backend_ref = crate::NTT4x30CIAvx512,
    backend_test = crate::NTT3x42CIIfma,
    params = poulpy_ckks::test_suite::CKKSTestParams { n: 64, hw: 48, base2k: 12, ..poulpy_ckks::test_suite::BASE19_PARAMS_F64 },
}

#[cfg(feature = "enable-rayon")]
poulpy_ckks::ckks_parity_test_suite! {
    mod ckks_parity_ci_fft64avx512rayon_f64,
    backend_ref = crate::FFT64CIAvx512,
    backend_test = crate::FFT64CIAvx512Rayon,
    scalar = f64,
    params = poulpy_ckks::test_suite::CKKSTestParams { n: 64, hw: 48, base2k: 19, ..poulpy_ckks::test_suite::BASE19_PARAMS_F64 },
    tests = {
        arithmetic => poulpy_ckks::test_suite::parity::test_arithmetic_parity,
        multiplication => poulpy_ckks::test_suite::parity::test_multiplication_parity,
        rotate => poulpy_ckks::test_suite::parity::test_rotate_parity,
        plaintext => poulpy_ckks::test_suite::parity::test_plaintext_parity,
        encoding => poulpy_ckks::test_suite::parity::test_encoding_parity,
        slot_encoding => poulpy_ckks::test_suite::parity::test_slot_encoding_parity,
        real_polynomial => poulpy_ckks::test_suite::parity::test_real_polynomial_parity,
    }
}

#[cfg(feature = "enable-rayon")]
poulpy_ckks::ckks_parity_test_suite! {
    mod ckks_parity_ci_fft64avx512rayon_quad,
    backend_ref = crate::FFT64CIAvx512,
    backend_test = crate::FFT64CIAvx512Rayon,
    scalar = poulpy_ckks::Quad,
    params = poulpy_ckks::test_suite::CKKSTestParams { n: 64, hw: 48, base2k: 19, ..poulpy_ckks::test_suite::BASE52_PARAMS_QUAD },
    tests = {
        arithmetic => poulpy_ckks::test_suite::parity::test_arithmetic_parity,
        multiplication => poulpy_ckks::test_suite::parity::test_multiplication_parity,
        rotate => poulpy_ckks::test_suite::parity::test_rotate_parity,
        plaintext => poulpy_ckks::test_suite::parity::test_plaintext_parity,
        encoding => poulpy_ckks::test_suite::parity::test_encoding_parity,
        slot_encoding => poulpy_ckks::test_suite::parity::test_slot_encoding_parity,
        real_polynomial => poulpy_ckks::test_suite::parity::test_real_polynomial_parity,
    }
}

#[cfg(feature = "enable-rayon")]
poulpy_ckks::ckks_encryption_parity_test_suite! {
    mod ckks_parity_ci_fft64avx512rayon_encryption,
    backend_ref = crate::FFT64CIAvx512,
    backend_test = crate::FFT64CIAvx512Rayon,
    params = poulpy_ckks::test_suite::CKKSTestParams { n: 64, hw: 48, base2k: 12, ..poulpy_ckks::test_suite::BASE19_PARAMS_F64 },
}

#[cfg(feature = "enable-rayon")]
poulpy_ckks::ckks_parity_test_suite! {
    mod ckks_parity_ci_ntt4x30avx512rayon_f64,
    backend_ref = crate::NTT4x30CIAvx512,
    backend_test = crate::NTT4x30CIAvx512Rayon,
    scalar = f64,
    params = poulpy_ckks::test_suite::CKKSTestParams { n: 64, hw: 48, base2k: 52, ..poulpy_ckks::test_suite::BASE52_PARAMS_F64 },
    tests = {
        arithmetic => poulpy_ckks::test_suite::parity::test_arithmetic_parity,
        multiplication => poulpy_ckks::test_suite::parity::test_multiplication_parity,
        rotate => poulpy_ckks::test_suite::parity::test_rotate_parity,
        plaintext => poulpy_ckks::test_suite::parity::test_plaintext_parity,
        encoding => poulpy_ckks::test_suite::parity::test_encoding_parity,
        slot_encoding => poulpy_ckks::test_suite::parity::test_slot_encoding_parity,
        real_polynomial => poulpy_ckks::test_suite::parity::test_real_polynomial_parity,
    }
}

#[cfg(feature = "enable-rayon")]
poulpy_ckks::ckks_parity_test_suite! {
    mod ckks_parity_ci_ntt4x30avx512rayon_quad,
    backend_ref = crate::NTT4x30CIAvx512,
    backend_test = crate::NTT4x30CIAvx512Rayon,
    scalar = poulpy_ckks::Quad,
    params = poulpy_ckks::test_suite::CKKSTestParams { n: 64, hw: 48, base2k: 52, ..poulpy_ckks::test_suite::BASE52_PARAMS_QUAD },
    tests = {
        arithmetic => poulpy_ckks::test_suite::parity::test_arithmetic_parity,
        multiplication => poulpy_ckks::test_suite::parity::test_multiplication_parity,
        rotate => poulpy_ckks::test_suite::parity::test_rotate_parity,
        plaintext => poulpy_ckks::test_suite::parity::test_plaintext_parity,
        encoding => poulpy_ckks::test_suite::parity::test_encoding_parity,
        slot_encoding => poulpy_ckks::test_suite::parity::test_slot_encoding_parity,
        real_polynomial => poulpy_ckks::test_suite::parity::test_real_polynomial_parity,
    }
}

#[cfg(feature = "enable-rayon")]
poulpy_ckks::ckks_encryption_parity_test_suite! {
    mod ckks_parity_ci_ntt4x30avx512rayon_encryption,
    backend_ref = crate::NTT4x30CIAvx512,
    backend_test = crate::NTT4x30CIAvx512Rayon,
    params = poulpy_ckks::test_suite::CKKSTestParams { n: 64, hw: 48, base2k: 12, ..poulpy_ckks::test_suite::BASE19_PARAMS_F64 },
}

#[cfg(all(feature = "enable-ifma", feature = "enable-rayon"))]
poulpy_ckks::ckks_parity_test_suite! {
    mod ckks_parity_ci_ntt3x42ifmarayon_f64,
    backend_ref = crate::NTT3x42CIIfma,
    backend_test = crate::NTT3x42CIIfmaRayon,
    scalar = f64,
    params = poulpy_ckks::test_suite::CKKSTestParams { n: 64, hw: 48, base2k: 52, ..poulpy_ckks::test_suite::BASE52_PARAMS_F64 },
    tests = {
        arithmetic => poulpy_ckks::test_suite::parity::test_arithmetic_parity,
        multiplication => poulpy_ckks::test_suite::parity::test_multiplication_parity,
        rotate => poulpy_ckks::test_suite::parity::test_rotate_parity,
        plaintext => poulpy_ckks::test_suite::parity::test_plaintext_parity,
        encoding => poulpy_ckks::test_suite::parity::test_encoding_parity,
        slot_encoding => poulpy_ckks::test_suite::parity::test_slot_encoding_parity,
        real_polynomial => poulpy_ckks::test_suite::parity::test_real_polynomial_parity,
    }
}

#[cfg(all(feature = "enable-ifma", feature = "enable-rayon"))]
poulpy_ckks::ckks_parity_test_suite! {
    mod ckks_parity_ci_ntt3x42ifmarayon_quad,
    backend_ref = crate::NTT3x42CIIfma,
    backend_test = crate::NTT3x42CIIfmaRayon,
    scalar = poulpy_ckks::Quad,
    params = poulpy_ckks::test_suite::CKKSTestParams { n: 64, hw: 48, base2k: 52, ..poulpy_ckks::test_suite::BASE52_PARAMS_QUAD },
    tests = {
        arithmetic => poulpy_ckks::test_suite::parity::test_arithmetic_parity,
        multiplication => poulpy_ckks::test_suite::parity::test_multiplication_parity,
        rotate => poulpy_ckks::test_suite::parity::test_rotate_parity,
        plaintext => poulpy_ckks::test_suite::parity::test_plaintext_parity,
        encoding => poulpy_ckks::test_suite::parity::test_encoding_parity,
        slot_encoding => poulpy_ckks::test_suite::parity::test_slot_encoding_parity,
        real_polynomial => poulpy_ckks::test_suite::parity::test_real_polynomial_parity,
    }
}

#[cfg(all(feature = "enable-ifma", feature = "enable-rayon"))]
poulpy_ckks::ckks_encryption_parity_test_suite! {
    mod ckks_parity_ci_ntt3x42ifmarayon_encryption,
    backend_ref = crate::NTT3x42CIIfma,
    backend_test = crate::NTT3x42CIIfmaRayon,
    params = poulpy_ckks::test_suite::CKKSTestParams { n: 64, hw: 48, base2k: 12, ..poulpy_ckks::test_suite::BASE19_PARAMS_F64 },
}

poulpy_ckks::ckks_parity_test_suite! {
    mod ckks_parity_ci_fft64avx512_rank2,
    backend_ref = poulpy_cpu_avx::FFT64CIAvx,
    backend_test = crate::FFT64CIAvx512,
    scalar = f64,
    params = poulpy_ckks::test_suite::CKKSTestParams { n: 64, hw: 48, rank: 2, base2k: 19, ..poulpy_ckks::test_suite::BASE19_PARAMS_F64 },
    tests = {
        arithmetic => poulpy_ckks::test_suite::parity::test_arithmetic_parity,
        multiplication => poulpy_ckks::test_suite::parity::test_multiplication_parity,
        rotate => poulpy_ckks::test_suite::parity::test_rotate_parity,
    }
}

poulpy_ckks::ckks_encryption_parity_test_suite! {
    mod ckks_parity_ci_fft64avx512_encryption_rank2,
    backend_ref = poulpy_cpu_avx::FFT64CIAvx,
    backend_test = crate::FFT64CIAvx512,
    params = poulpy_ckks::test_suite::CKKSTestParams { n: 64, hw: 48, rank: 2, base2k: 12, ..poulpy_ckks::test_suite::BASE19_PARAMS_F64 },
}

poulpy_ckks::ckks_parity_test_suite! {
    mod ckks_parity_ci_ntt4x30avx512_rank2,
    backend_ref = poulpy_cpu_avx::NTT4x30CIAvx,
    backend_test = crate::NTT4x30CIAvx512,
    scalar = f64,
    params = poulpy_ckks::test_suite::CKKSTestParams { n: 64, hw: 48, rank: 2, base2k: 52, ..poulpy_ckks::test_suite::BASE19_PARAMS_F64 },
    tests = {
        arithmetic => poulpy_ckks::test_suite::parity::test_arithmetic_parity,
        multiplication => poulpy_ckks::test_suite::parity::test_multiplication_parity,
        rotate => poulpy_ckks::test_suite::parity::test_rotate_parity,
    }
}

poulpy_ckks::ckks_encryption_parity_test_suite! {
    mod ckks_parity_ci_ntt4x30avx512_encryption_rank2,
    backend_ref = poulpy_cpu_avx::NTT4x30CIAvx,
    backend_test = crate::NTT4x30CIAvx512,
    params = poulpy_ckks::test_suite::CKKSTestParams { n: 64, hw: 48, rank: 2, base2k: 12, ..poulpy_ckks::test_suite::BASE19_PARAMS_F64 },
}

#[cfg(feature = "enable-ifma")]
poulpy_ckks::ckks_parity_test_suite! {
    mod ckks_parity_ci_ntt3x42ifma_rank2,
    backend_ref = crate::NTT4x30CIAvx512,
    backend_test = crate::NTT3x42CIIfma,
    scalar = f64,
    params = poulpy_ckks::test_suite::CKKSTestParams { n: 64, hw: 48, rank: 2, base2k: 52, ..poulpy_ckks::test_suite::BASE19_PARAMS_F64 },
    tests = {
        arithmetic => poulpy_ckks::test_suite::parity::test_arithmetic_parity,
        multiplication => poulpy_ckks::test_suite::parity::test_multiplication_parity,
        rotate => poulpy_ckks::test_suite::parity::test_rotate_parity,
    }
}

#[cfg(feature = "enable-ifma")]
poulpy_ckks::ckks_encryption_parity_test_suite! {
    mod ckks_parity_ci_ntt3x42ifma_encryption_rank2,
    backend_ref = crate::NTT4x30CIAvx512,
    backend_test = crate::NTT3x42CIIfma,
    params = poulpy_ckks::test_suite::CKKSTestParams { n: 64, hw: 48, rank: 2, base2k: 12, ..poulpy_ckks::test_suite::BASE19_PARAMS_F64 },
}

#[cfg(feature = "enable-rayon")]
poulpy_ckks::ckks_parity_test_suite! {
    mod ckks_parity_ci_fft64avx512rayon_rank2,
    backend_ref = crate::FFT64CIAvx512,
    backend_test = crate::FFT64CIAvx512Rayon,
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
    mod ckks_parity_ci_fft64avx512rayon_encryption_rank2,
    backend_ref = crate::FFT64CIAvx512,
    backend_test = crate::FFT64CIAvx512Rayon,
    params = poulpy_ckks::test_suite::CKKSTestParams { n: 64, hw: 48, rank: 2, base2k: 12, ..poulpy_ckks::test_suite::BASE19_PARAMS_F64 },
}

#[cfg(feature = "enable-rayon")]
poulpy_ckks::ckks_parity_test_suite! {
    mod ckks_parity_ci_ntt4x30avx512rayon_rank2,
    backend_ref = crate::NTT4x30CIAvx512,
    backend_test = crate::NTT4x30CIAvx512Rayon,
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
    mod ckks_parity_ci_ntt4x30avx512rayon_encryption_rank2,
    backend_ref = crate::NTT4x30CIAvx512,
    backend_test = crate::NTT4x30CIAvx512Rayon,
    params = poulpy_ckks::test_suite::CKKSTestParams { n: 64, hw: 48, rank: 2, base2k: 12, ..poulpy_ckks::test_suite::BASE19_PARAMS_F64 },
}

#[cfg(all(feature = "enable-ifma", feature = "enable-rayon"))]
poulpy_ckks::ckks_parity_test_suite! {
    mod ckks_parity_ci_ntt3x42ifmarayon_rank2,
    backend_ref = crate::NTT3x42CIIfma,
    backend_test = crate::NTT3x42CIIfmaRayon,
    scalar = f64,
    params = poulpy_ckks::test_suite::CKKSTestParams { n: 64, hw: 48, rank: 2, base2k: 52, ..poulpy_ckks::test_suite::BASE19_PARAMS_F64 },
    tests = {
        arithmetic => poulpy_ckks::test_suite::parity::test_arithmetic_parity,
        multiplication => poulpy_ckks::test_suite::parity::test_multiplication_parity,
        rotate => poulpy_ckks::test_suite::parity::test_rotate_parity,
    }
}

#[cfg(all(feature = "enable-ifma", feature = "enable-rayon"))]
poulpy_ckks::ckks_encryption_parity_test_suite! {
    mod ckks_parity_ci_ntt3x42ifmarayon_encryption_rank2,
    backend_ref = crate::NTT3x42CIIfma,
    backend_test = crate::NTT3x42CIIfmaRayon,
    params = poulpy_ckks::test_suite::CKKSTestParams { n: 64, hw: 48, rank: 2, base2k: 12, ..poulpy_ckks::test_suite::BASE19_PARAMS_F64 },
}

#[cfg(all(feature = "enable-ifma", feature = "enable-rayon"))]
poulpy_ckks::ckks_parity_test_suite! {
    mod ckks_parity_ifma_prepared_large,
    backend_ref = poulpy_cpu_avx::NTT4x30Avx,
    backend_test = crate::NTT3x42IfmaRayon,
    scalar = f64,
    params = poulpy_ckks::test_suite::CKKSTestParams { n: 1 << 16, base2k: 52, rank: 1, ..poulpy_ckks::test_suite::BASE52_PARAMS_F64 },
    tests = {
        multiplication => poulpy_ckks::test_suite::parity::test_prepared_multiplication_parity,
    }
}
