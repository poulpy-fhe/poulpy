//! CKKS conformance tests for the portable backends.
//!
//! All test logic lives in `poulpy_ckks::test_suite` as backend-generic
//! functions; this file only instantiates that suite for each concrete
//! `(backend, scalar, encoder, params)` combination via `ckks_backend_test_suite!`.
//! There should be no hand-written test here.

use poulpy_ckks::{ckks_backend_rank2_test_suite, ckks_backend_test_suite};

const ATK_ROTATIONS: &[i64] = &[1, 7];

ckks_backend_test_suite!(
    mod fft64_f64,
    backend = crate::FFT64Portable,
    scalar = f64,
    encoder = crate::ckks_encoding::EncodingFFTTable<f64>,
    params = poulpy_ckks::test_suite::BASE19_PARAMS_F64,
    rotations = super::ATK_ROTATIONS,
);

ckks_backend_test_suite!(
    mod ntt4x30_f64,
    backend = crate::NTT4x30Portable,
    scalar = f64,
    encoder = crate::ckks_encoding::EncodingFFTTable<f64>,
    params = poulpy_ckks::test_suite::BASE52_PARAMS_F64,
    rotations = super::ATK_ROTATIONS,
);

ckks_backend_test_suite!(
    mod ntt4x30_f128,
    backend = crate::NTT4x30Portable,
    scalar = poulpy_ckks::Quad,
    encoder = crate::ckks_encoding::EncodingFFTTable<poulpy_ckks::Quad>,
    params = poulpy_ckks::test_suite::BASE52_PARAMS_QUAD,
    rotations = super::ATK_ROTATIONS,
);

// Rank-2 coverage: the rank-generic arithmetic subset (the full suite's
// bootstrapping/EvalMod/DFT/PaCo pipelines are rank-1 by construction).
ckks_backend_rank2_test_suite!(
    mod ntt4x30_f64_rank2,
    backend = crate::NTT4x30Portable,
    scalar = f64,
    encoder = crate::ckks_encoding::EncodingFFTTable<f64>,
    params = poulpy_ckks::test_suite::BASE52_PARAMS_F64_RANK2,
    rotations = super::ATK_ROTATIONS,
);

/// Full logN16 bootstraps per preset: slow, so opt in with `--ignored`.
mod bootstrapping_presets {
    use poulpy_ckks::test_suite::presets::{bootstrapping_preset_keys_roundtrip, bootstrapping_presets_meet_precision};

    #[test]
    #[ignore = "generates a full-size preset key set; opt in with --ignored"]
    fn ntt4x30_preset_keys_roundtrip() {
        bootstrapping_preset_keys_roundtrip::<crate::NTT4x30Portable>(52);
    }

    #[test]
    #[ignore = "runs a full logN16 bootstrap per preset; opt in with --ignored"]
    fn ntt4x30_presets_meet_precision() {
        bootstrapping_presets_meet_precision::<crate::NTT4x30Portable>(52);
    }

    #[test]
    #[ignore = "runs a full logN16 bootstrap per preset; opt in with --ignored"]
    fn fft64_presets_meet_precision() {
        bootstrapping_presets_meet_precision::<crate::FFT64Portable>(19);
    }
}

// Paired OEP validation against the oracle of each family.
poulpy_ckks::ckks_parity_test_suite! {
    mod ckks_parity_fft64portable_f64,
    backend_ref = poulpy_cpu_oracle::FFT64Oracle,
    backend_test = crate::FFT64Portable,
    scalar = f64,
    params = poulpy_ckks::test_suite::CKKSTestParams { n: 64, hw: 48, base2k: 12, ..poulpy_ckks::test_suite::BASE19_PARAMS_F64 },
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
    mod ckks_parity_fft64portable_quad,
    backend_ref = poulpy_cpu_oracle::FFT64Oracle,
    backend_test = crate::FFT64Portable,
    scalar = poulpy_ckks::Quad,
    params = poulpy_ckks::test_suite::CKKSTestParams { n: 64, hw: 48, base2k: 12, ..poulpy_ckks::test_suite::BASE52_PARAMS_QUAD },
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
    mod ckks_parity_fft64portable_encryption,
    backend_ref = poulpy_cpu_oracle::FFT64Oracle,
    backend_test = crate::FFT64Portable,
    reference_factory = poulpy_cpu_oracle::test_suite::controlled_sampling_module,
    params = poulpy_ckks::test_suite::CKKSTestParams { n: 64, hw: 48, base2k: 12, ..poulpy_ckks::test_suite::BASE19_PARAMS_F64 },
}

poulpy_ckks::ckks_parity_test_suite! {
    mod ckks_parity_ntt4x30portable_f64,
    backend_ref = poulpy_cpu_oracle::NTT4x30Oracle,
    backend_test = crate::NTT4x30Portable,
    scalar = f64,
    params = poulpy_ckks::test_suite::CKKSTestParams { n: 64, hw: 48, base2k: 12, ..poulpy_ckks::test_suite::BASE52_PARAMS_F64 },
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
    mod ckks_parity_ntt4x30portable_quad,
    backend_ref = poulpy_cpu_oracle::NTT4x30Oracle,
    backend_test = crate::NTT4x30Portable,
    scalar = poulpy_ckks::Quad,
    params = poulpy_ckks::test_suite::CKKSTestParams { n: 64, hw: 48, base2k: 12, ..poulpy_ckks::test_suite::BASE52_PARAMS_QUAD },
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
    mod ckks_parity_ntt4x30portable_encryption,
    backend_ref = poulpy_cpu_oracle::FFT64Oracle,
    backend_test = crate::NTT4x30Portable,
    reference_factory = poulpy_cpu_oracle::test_suite::controlled_sampling_module,
    params = poulpy_ckks::test_suite::CKKSTestParams { n: 64, hw: 48, base2k: 12, ..poulpy_ckks::test_suite::BASE19_PARAMS_F64 },
}

poulpy_ckks::ckks_parity_test_suite! {
    mod ckks_parity_f32_encoding_fft64,
    backend_ref = poulpy_cpu_oracle::FFT64Oracle,
    backend_test = crate::FFT64Portable,
    scalar = f32,
    params = poulpy_ckks::test_suite::CKKSTestParams { n: 64, hw: 48, base2k: 12, ..poulpy_ckks::test_suite::BASE19_PARAMS_F64 },
    tests = {
        encoding => poulpy_ckks::test_suite::parity::test_encoding_parity,
        slot_encoding => poulpy_ckks::test_suite::parity::test_slot_encoding_parity,
        dft => poulpy_ckks::test_suite::parity::test_dft_parity,
    }
}

poulpy_ckks::ckks_parity_test_suite! {
    mod ckks_parity_f32_encoding_ntt4x30,
    backend_ref = poulpy_cpu_oracle::NTT4x30Oracle,
    backend_test = crate::NTT4x30Portable,
    scalar = f32,
    params = poulpy_ckks::test_suite::CKKSTestParams { n: 64, hw: 48, base2k: 12, ..poulpy_ckks::test_suite::BASE19_PARAMS_F64 },
    tests = {
        encoding => poulpy_ckks::test_suite::parity::test_encoding_parity,
        slot_encoding => poulpy_ckks::test_suite::parity::test_slot_encoding_parity,
        dft => poulpy_ckks::test_suite::parity::test_dft_parity,
    }
}

// Explicit rank-2 contracts. A caller can select another supported rank through params.
poulpy_ckks::ckks_parity_test_suite! {
    mod ckks_parity_fft64portable_rank2,
    backend_ref = poulpy_cpu_oracle::FFT64Oracle,
    backend_test = crate::FFT64Portable,
    scalar = f64,
    params = poulpy_ckks::test_suite::CKKSTestParams { n: 64, hw: 48, rank: 2, base2k: 12, ..poulpy_ckks::test_suite::BASE19_PARAMS_F64 },
    tests = {
        arithmetic => poulpy_ckks::test_suite::parity::test_arithmetic_parity,
        imag => poulpy_ckks::test_suite::parity::test_imag_parity,
        multiplication => poulpy_ckks::test_suite::parity::test_multiplication_parity,
        rotate => poulpy_ckks::test_suite::parity::test_rotate_parity,
        conjugate => poulpy_ckks::test_suite::parity::test_conjugate_parity,
    }
}

poulpy_ckks::ckks_encryption_parity_test_suite! {
    mod ckks_parity_fft64portable_encryption_rank2,
    backend_ref = poulpy_cpu_oracle::FFT64Oracle,
    backend_test = crate::FFT64Portable,
    reference_factory = poulpy_cpu_oracle::test_suite::controlled_sampling_module,
    params = poulpy_ckks::test_suite::CKKSTestParams { n: 64, hw: 48, rank: 2, base2k: 12, ..poulpy_ckks::test_suite::BASE19_PARAMS_F64 },
}

poulpy_ckks::ckks_parity_test_suite! {
    mod ckks_parity_ntt4x30portable_rank2,
    backend_ref = poulpy_cpu_oracle::NTT4x30Oracle,
    backend_test = crate::NTT4x30Portable,
    scalar = f64,
    params = poulpy_ckks::test_suite::CKKSTestParams { n: 64, hw: 48, rank: 2, base2k: 12, ..poulpy_ckks::test_suite::BASE19_PARAMS_F64 },
    tests = {
        arithmetic => poulpy_ckks::test_suite::parity::test_arithmetic_parity,
        imag => poulpy_ckks::test_suite::parity::test_imag_parity,
        multiplication => poulpy_ckks::test_suite::parity::test_multiplication_parity,
        rotate => poulpy_ckks::test_suite::parity::test_rotate_parity,
        conjugate => poulpy_ckks::test_suite::parity::test_conjugate_parity,
    }
}

poulpy_ckks::ckks_encryption_parity_test_suite! {
    mod ckks_parity_ntt4x30portable_encryption_rank2,
    backend_ref = poulpy_cpu_oracle::FFT64Oracle,
    backend_test = crate::NTT4x30Portable,
    reference_factory = poulpy_cpu_oracle::test_suite::controlled_sampling_module,
    params = poulpy_ckks::test_suite::CKKSTestParams { n: 64, hw: 48, rank: 2, base2k: 12, ..poulpy_ckks::test_suite::BASE19_PARAMS_F64 },
}

// Conjugate-invariant backends run the ring-generic suites against the same pairs.
poulpy_ckks::ckks_parity_test_suite! {
    mod ckks_parity_ci_fft64portable_f64,
    backend_ref = poulpy_cpu_oracle::FFT64CIOracle,
    backend_test = crate::FFT64CIPortable,
    scalar = f64,
    params = poulpy_ckks::test_suite::CKKSTestParams { n: 64, hw: 48, base2k: 12, ..poulpy_ckks::test_suite::BASE19_PARAMS_F64 },
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
    mod ckks_parity_ci_fft64portable_quad,
    backend_ref = poulpy_cpu_oracle::FFT64CIOracle,
    backend_test = crate::FFT64CIPortable,
    scalar = poulpy_ckks::Quad,
    params = poulpy_ckks::test_suite::CKKSTestParams { n: 64, hw: 48, base2k: 12, ..poulpy_ckks::test_suite::BASE52_PARAMS_QUAD },
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
    mod ckks_parity_ci_fft64portable_encryption,
    backend_ref = poulpy_cpu_oracle::FFT64CIOracle,
    backend_test = crate::FFT64CIPortable,
    reference_factory = poulpy_cpu_oracle::test_suite::controlled_sampling_module,
    params = poulpy_ckks::test_suite::CKKSTestParams { n: 64, hw: 48, base2k: 12, ..poulpy_ckks::test_suite::BASE19_PARAMS_F64 },
}

poulpy_ckks::ckks_parity_test_suite! {
    mod ckks_parity_ci_ntt4x30portable_f64,
    backend_ref = poulpy_cpu_oracle::NTT4x30CIOracle,
    backend_test = crate::NTT4x30CIPortable,
    scalar = f64,
    params = poulpy_ckks::test_suite::CKKSTestParams { n: 64, hw: 48, base2k: 12, ..poulpy_ckks::test_suite::BASE52_PARAMS_F64 },
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
    mod ckks_parity_ci_ntt4x30portable_quad,
    backend_ref = poulpy_cpu_oracle::NTT4x30CIOracle,
    backend_test = crate::NTT4x30CIPortable,
    scalar = poulpy_ckks::Quad,
    params = poulpy_ckks::test_suite::CKKSTestParams { n: 64, hw: 48, base2k: 12, ..poulpy_ckks::test_suite::BASE52_PARAMS_QUAD },
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
    mod ckks_parity_ci_ntt4x30portable_encryption,
    backend_ref = poulpy_cpu_oracle::FFT64CIOracle,
    backend_test = crate::NTT4x30CIPortable,
    reference_factory = poulpy_cpu_oracle::test_suite::controlled_sampling_module,
    params = poulpy_ckks::test_suite::CKKSTestParams { n: 64, hw: 48, base2k: 12, ..poulpy_ckks::test_suite::BASE19_PARAMS_F64 },
}

poulpy_ckks::ckks_parity_test_suite! {
    mod ckks_parity_ci_f32_encoding_fft64,
    backend_ref = poulpy_cpu_oracle::FFT64CIOracle,
    backend_test = crate::FFT64CIPortable,
    scalar = f32,
    params = poulpy_ckks::test_suite::CKKSTestParams { n: 64, hw: 48, base2k: 12, ..poulpy_ckks::test_suite::BASE19_PARAMS_F64 },
    tests = {
        encoding => poulpy_ckks::test_suite::parity::test_encoding_parity,
        slot_encoding => poulpy_ckks::test_suite::parity::test_slot_encoding_parity,
    }
}

poulpy_ckks::ckks_parity_test_suite! {
    mod ckks_parity_ci_f32_encoding_ntt4x30,
    backend_ref = poulpy_cpu_oracle::NTT4x30CIOracle,
    backend_test = crate::NTT4x30CIPortable,
    scalar = f32,
    params = poulpy_ckks::test_suite::CKKSTestParams { n: 64, hw: 48, base2k: 12, ..poulpy_ckks::test_suite::BASE19_PARAMS_F64 },
    tests = {
        encoding => poulpy_ckks::test_suite::parity::test_encoding_parity,
        slot_encoding => poulpy_ckks::test_suite::parity::test_slot_encoding_parity,
    }
}

poulpy_ckks::ckks_parity_test_suite! {
    mod ckks_parity_ci_fft64portable_rank2,
    backend_ref = poulpy_cpu_oracle::FFT64CIOracle,
    backend_test = crate::FFT64CIPortable,
    scalar = f64,
    params = poulpy_ckks::test_suite::CKKSTestParams { n: 64, hw: 48, rank: 2, base2k: 12, ..poulpy_ckks::test_suite::BASE19_PARAMS_F64 },
    tests = {
        arithmetic => poulpy_ckks::test_suite::parity::test_arithmetic_parity,
        multiplication => poulpy_ckks::test_suite::parity::test_multiplication_parity,
        rotate => poulpy_ckks::test_suite::parity::test_rotate_parity,
    }
}

poulpy_ckks::ckks_encryption_parity_test_suite! {
    mod ckks_parity_ci_fft64portable_encryption_rank2,
    backend_ref = poulpy_cpu_oracle::FFT64CIOracle,
    backend_test = crate::FFT64CIPortable,
    reference_factory = poulpy_cpu_oracle::test_suite::controlled_sampling_module,
    params = poulpy_ckks::test_suite::CKKSTestParams { n: 64, hw: 48, rank: 2, base2k: 12, ..poulpy_ckks::test_suite::BASE19_PARAMS_F64 },
}

poulpy_ckks::ckks_parity_test_suite! {
    mod ckks_parity_ci_ntt4x30portable_rank2,
    backend_ref = poulpy_cpu_oracle::NTT4x30CIOracle,
    backend_test = crate::NTT4x30CIPortable,
    scalar = f64,
    params = poulpy_ckks::test_suite::CKKSTestParams { n: 64, hw: 48, rank: 2, base2k: 12, ..poulpy_ckks::test_suite::BASE19_PARAMS_F64 },
    tests = {
        arithmetic => poulpy_ckks::test_suite::parity::test_arithmetic_parity,
        multiplication => poulpy_ckks::test_suite::parity::test_multiplication_parity,
        rotate => poulpy_ckks::test_suite::parity::test_rotate_parity,
    }
}

poulpy_ckks::ckks_encryption_parity_test_suite! {
    mod ckks_parity_ci_ntt4x30portable_encryption_rank2,
    backend_ref = poulpy_cpu_oracle::FFT64CIOracle,
    backend_test = crate::NTT4x30CIPortable,
    reference_factory = poulpy_cpu_oracle::test_suite::controlled_sampling_module,
    params = poulpy_ckks::test_suite::CKKSTestParams { n: 64, hw: 48, rank: 2, base2k: 12, ..poulpy_ckks::test_suite::BASE19_PARAMS_F64 },
}
