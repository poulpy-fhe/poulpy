//! CKKS conformance tests for the reference backends.
//!
//! All test logic lives in `poulpy_ckks::test_suite` as backend-generic
//! functions; this file only instantiates that suite for each concrete
//! `(backend, scalar, encoder, params)` combination via `ckks_backend_test_suite!`.
//! There should be no hand-written test here.

use poulpy_ckks::{ckks_backend_rank2_test_suite, ckks_backend_test_suite};

const ATK_ROTATIONS: &[i64] = &[1, 7];

ckks_backend_test_suite!(
    mod fft64_f64,
    backend = crate::FFT64Ref,
    scalar = f64,
    encoder = crate::FFT64ReimTable<f64>,
    params = poulpy_ckks::test_suite::BASE19_PARAMS_F64,
    rotations = super::ATK_ROTATIONS,
);

ckks_backend_test_suite!(
    mod ntt4x30_f64,
    backend = crate::NTT4x30Ref,
    scalar = f64,
    encoder = crate::FFT64ReimTable<f64>,
    params = poulpy_ckks::test_suite::BASE52_PARAMS_F64,
    rotations = super::ATK_ROTATIONS,
);

ckks_backend_test_suite!(
    mod ntt4x30_f128,
    backend = crate::NTT4x30Ref,
    scalar = poulpy_ckks::Quad,
    encoder = crate::FFT64ReimTable<poulpy_ckks::Quad>,
    params = poulpy_ckks::test_suite::BASE52_PARAMS_QUAD,
    rotations = super::ATK_ROTATIONS,
);

// Rank-2 coverage: the rank-generic arithmetic subset (the full suite's
// bootstrapping/EvalMod/DFT/PaCo pipelines are rank-1 by construction).
ckks_backend_rank2_test_suite!(
    mod ntt4x30_f64_rank2,
    backend = crate::NTT4x30Ref,
    scalar = f64,
    encoder = crate::FFT64ReimTable<f64>,
    params = poulpy_ckks::test_suite::BASE52_PARAMS_F64_RANK2,
    rotations = super::ATK_ROTATIONS,
);

/// Full logN16 bootstraps per preset: slow, so opt in with `--ignored`.
mod bootstrapping_presets {
    use poulpy_ckks::test_suite::presets::bootstrapping_presets_meet_precision;

    #[test]
    #[ignore = "runs a full logN16 bootstrap per preset; opt in with --ignored"]
    fn ntt4x30_presets_meet_precision() {
        bootstrapping_presets_meet_precision::<crate::NTT4x30Ref>(52);
    }

    #[test]
    #[ignore = "runs a full logN16 bootstrap per preset; opt in with --ignored"]
    fn fft64_presets_meet_precision() {
        bootstrapping_presets_meet_precision::<crate::FFT64Ref>(19);
    }
}

// Paired OEP validation. Serial backends validate against portable implementations;
// Rayon backends validate against their serial counterparts.
poulpy_ckks::ckks_parity_test_suite! {
    mod ckks_parity_fft64ref_f64,
    backend_ref = crate::NTT4x30Ref,
    backend_test = crate::FFT64Ref,
    scalar = f64,
    params = poulpy_ckks::test_suite::CKKSTestParams { n: 64, hw: 48, base2k: 12, ..poulpy_ckks::test_suite::BASE19_PARAMS_F64 },
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
    mod ckks_parity_fft64ref_quad,
    backend_ref = crate::NTT4x30Ref,
    backend_test = crate::FFT64Ref,
    scalar = poulpy_ckks::Quad,
    params = poulpy_ckks::test_suite::CKKSTestParams { n: 64, hw: 48, base2k: 12, ..poulpy_ckks::test_suite::BASE52_PARAMS_QUAD },
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
    mod ckks_parity_fft64ref_encryption,
    backend_ref = crate::test_suite::ControlledSamplingFFT64Ref,
    backend_test = crate::FFT64Ref,
    params = poulpy_ckks::test_suite::CKKSTestParams { n: 64, hw: 48, base2k: 12, ..poulpy_ckks::test_suite::BASE19_PARAMS_F64 },
}

poulpy_ckks::ckks_parity_test_suite! {
    mod ckks_parity_ntt4x30ref_f64,
    backend_ref = crate::FFT64Ref,
    backend_test = crate::NTT4x30Ref,
    scalar = f64,
    params = poulpy_ckks::test_suite::CKKSTestParams { n: 64, hw: 48, base2k: 12, ..poulpy_ckks::test_suite::BASE52_PARAMS_F64 },
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
    mod ckks_parity_ntt4x30ref_quad,
    backend_ref = crate::FFT64Ref,
    backend_test = crate::NTT4x30Ref,
    scalar = poulpy_ckks::Quad,
    params = poulpy_ckks::test_suite::CKKSTestParams { n: 64, hw: 48, base2k: 12, ..poulpy_ckks::test_suite::BASE52_PARAMS_QUAD },
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
    mod ckks_parity_ntt4x30ref_encryption,
    backend_ref = crate::test_suite::ControlledSamplingFFT64Ref,
    backend_test = crate::NTT4x30Ref,
    params = poulpy_ckks::test_suite::CKKSTestParams { n: 64, hw: 48, base2k: 12, ..poulpy_ckks::test_suite::BASE19_PARAMS_F64 },
}

poulpy_ckks::ckks_parity_test_suite! {
    mod ckks_parity_f32_encoding,
    backend_ref = crate::FFT64Ref,
    backend_test = crate::NTT4x30Ref,
    scalar = f32,
    params = poulpy_ckks::test_suite::CKKSTestParams { n: 64, hw: 48, base2k: 12, ..poulpy_ckks::test_suite::BASE19_PARAMS_F64 },
    tests = {
        encoding => poulpy_ckks::test_suite::parity::test_encoding_parity,
        dft => poulpy_ckks::test_suite::parity::test_dft_parity,
    }
}

// Explicit rank-2 contracts. A caller can select another supported rank through params.
poulpy_ckks::ckks_parity_test_suite! {
    mod ckks_parity_fft64ref_rank2,
    backend_ref = crate::NTT4x30Ref,
    backend_test = crate::FFT64Ref,
    scalar = f64,
    params = poulpy_ckks::test_suite::CKKSTestParams { n: 64, hw: 48, rank: 2, base2k: 12, ..poulpy_ckks::test_suite::BASE19_PARAMS_F64 },
    tests = {
        arithmetic => poulpy_ckks::test_suite::parity::test_arithmetic_parity,
        multiplication => poulpy_ckks::test_suite::parity::test_multiplication_parity,
        automorphism => poulpy_ckks::test_suite::parity::test_automorphism_parity,
    }
}

poulpy_ckks::ckks_encryption_parity_test_suite! {
    mod ckks_parity_fft64ref_encryption_rank2,
    backend_ref = crate::test_suite::ControlledSamplingFFT64Ref,
    backend_test = crate::FFT64Ref,
    params = poulpy_ckks::test_suite::CKKSTestParams { n: 64, hw: 48, rank: 2, base2k: 12, ..poulpy_ckks::test_suite::BASE19_PARAMS_F64 },
}

poulpy_ckks::ckks_parity_test_suite! {
    mod ckks_parity_ntt4x30ref_rank2,
    backend_ref = crate::FFT64Ref,
    backend_test = crate::NTT4x30Ref,
    scalar = f64,
    params = poulpy_ckks::test_suite::CKKSTestParams { n: 64, hw: 48, rank: 2, base2k: 12, ..poulpy_ckks::test_suite::BASE19_PARAMS_F64 },
    tests = {
        arithmetic => poulpy_ckks::test_suite::parity::test_arithmetic_parity,
        multiplication => poulpy_ckks::test_suite::parity::test_multiplication_parity,
        automorphism => poulpy_ckks::test_suite::parity::test_automorphism_parity,
    }
}

poulpy_ckks::ckks_encryption_parity_test_suite! {
    mod ckks_parity_ntt4x30ref_encryption_rank2,
    backend_ref = crate::test_suite::ControlledSamplingFFT64Ref,
    backend_test = crate::NTT4x30Ref,
    params = poulpy_ckks::test_suite::CKKSTestParams { n: 64, hw: 48, rank: 2, base2k: 12, ..poulpy_ckks::test_suite::BASE19_PARAMS_F64 },
}
