use poulpy_ckks::ckks_backend_test_suite;

const ATK_ROTATIONS: &[i64] = &[1, 7];

#[test]
fn encode_decode_reim_roundtrip() {
    use crate::FFT64ReimTable;
    use poulpy_ckks::encoding::reim::Encoder;
    use poulpy_ckks::layouts::CKKSModuleAlloc;

    let n = 16usize;
    let m = n / 2;

    let re_in: Vec<f64> = (0..m).map(|i| (i as f64) / (m as f64)).collect();
    let im_in: Vec<f64> = (0..m).map(|i| -((i as f64) / (m as f64))).collect();

    let encoder = Encoder::<FFT64ReimTable<f64>>::new::<f64>(m).unwrap();

    let host_module = poulpy_hal::layouts::Module::<poulpy_hal::layouts::HostBytesBackend>::new(n as u64);
    let mut pt = host_module.ckks_pt_vec_alloc(
        poulpy_core::layouts::Base2K(16),
        poulpy_ckks::CKKSMeta {
            log_sparsity: 0,
            log_delta: 40,
            log_budget: 10,
        },
    );
    encoder.encode_reim(&mut pt, &re_in, &im_in).unwrap();

    let mut re_out = vec![0.0f64; m];
    let mut im_out = vec![0.0f64; m];
    encoder.decode_reim(&pt, &mut re_out, &mut im_out).unwrap();

    let max_err = |a: &[f64], b: &[f64]| a.iter().zip(b).map(|(x, y)| (x - y).abs()).fold(0.0_f64, f64::max);
    let bound = 1e-10;
    let err_re = max_err(&re_in, &re_out);
    let err_im = max_err(&im_in, &im_out);
    assert!(err_re < bound, "re max_err={err_re:.2e} exceeds bound={bound:.2e}");
    assert!(err_im < bound, "im max_err={err_im:.2e} exceeds bound={bound:.2e}");
}

/// Homomorphic DFT round-trip: `CoeffsToSlots` (Encode/IDFT) then
/// `SlotsToCoeffs` (Decode/DFT) recovers the input slot vector (`Standard`
/// format). Validates the full Phase-2 pipeline — factor encoding, the chained
/// prepared linear transformations, the implicit-rescale scale accounting, and
/// key management — against CKKS precision. Custom large-`k` params give the
/// depth (`2·log_slots` multiplies) enough `log_budget`.
#[test]
fn dft_round_trip_standard() {
    use std::collections::HashMap;

    // This test drives the homomorphic DFT through the public `DFTOps` method
    // surface (`module.ckks_*`) rather than the free functions.
    use poulpy_ckks::{
        CKKSMeta,
        api::DFTOps,
        encoding::reim::Encoder,
        layouts::{DFTOutputFormat, DFTPlan, DFTType},
        test_suite::{
            CKKSTestParams,
            helpers::{alloc_scratch, ckks_decrypt_decode, ckks_encrypt, gen_atk, gen_sk_with_raw, test_vector_1},
        },
    };
    use poulpy_core::layouts::Base2K;
    use poulpy_hal::{
        api::ScratchOwnedBorrow,
        layouts::{CyclotomicOrder, Module},
    };

    use crate::{FFT64Ref, FFT64ReimTable};

    // m = 16 complex slots (log_slots = 4), base2k = 19, with enough k for the
    // 2*4 = 8 chained plaintext-multiplies at 30 bits each.
    let base2k = 19usize;
    let log_delta = 30usize;
    let params = CKKSTestParams {
        n: 32,
        base2k,
        k: base2k * 18, // 342 bits
        prec: CKKSMeta {
            log_sparsity: 0,
            log_delta,
            log_budget: 10,
        },
        hw: 16,
        dsize: 1,
    };
    let m = params.n / 2;
    let log_slots = m.trailing_zeros() as usize;

    let module = Module::<FFT64Ref>::new(params.n as u64);
    let host_module = Module::<poulpy_hal::layouts::HostBytesBackend>::new(params.n as u64);
    let encoder = Encoder::<FFT64ReimTable<f64>>::new::<f64>(m).unwrap();

    let (sk_raw, sk) = gen_sk_with_raw(&params, &module, &host_module, [0u8; 32]);
    let mut scratch = alloc_scratch(&params, &module);

    // Encode (IDFT) and Decode (DFT) matrices, Standard format, no merge.
    let factor_meta = CKKSMeta {
        log_sparsity: 0,
        log_delta,
        log_budget: 10,
    };
    let make = |kind| DFTPlan {
        kind,
        factorization_depth: vec![1usize; log_slots],
        factor_giant_steps: vec![2usize; log_slots],
        format: DFTOutputFormat::Standard,
        scaling: None,
        bit_reversed: false,
        factor_log_delta: 0,
    };
    let enc_dft = module.ckks_new_dft_matrix(
        &host_module,
        &encoder,
        Base2K(base2k as u32),
        factor_meta,
        &make(DFTType::Encode),
        &mut scratch.borrow(),
    );
    let dec_dft = module.ckks_new_dft_matrix(
        &host_module,
        &encoder,
        Base2K(base2k as u32),
        factor_meta,
        &make(DFTType::Decode),
        &mut scratch.borrow(),
    );

    // Automorphism keys: union of both matrices' Galois elements.
    let order = module.cyclotomic_order();
    let mut atks = HashMap::new();
    for p in enc_dft
        .galois_elements(order)
        .into_iter()
        .chain(dec_dft.galois_elements(order))
    {
        atks.entry(p)
            .or_insert_with(|| gen_atk(&params, &module, p, &sk_raw, &mut scratch.borrow()));
    }

    // Encrypt a random complex vector and round-trip it.
    let (a_re, a_im) = test_vector_1::<f64>(m);
    let mut ct = ckks_encrypt(
        &params,
        &module,
        &host_module,
        &encoder,
        &sk,
        params.k,
        &a_re,
        &a_im,
        &mut scratch.borrow(),
    );

    module
        .ckks_coeffs_to_slots(&mut ct, &enc_dft, &atks, &mut scratch.borrow())
        .unwrap();
    module
        .ckks_slots_to_coeffs(&mut ct, &dec_dft, &atks, &mut scratch.borrow())
        .unwrap();

    let (got_re, got_im) = ckks_decrypt_decode::<FFT64Ref, f64, _>(&params, &module, &encoder, &ct, &sk, &mut scratch.borrow());

    let max_err = a_re
        .iter()
        .zip(&got_re)
        .chain(a_im.iter().zip(&got_im))
        .map(|(want, got)| (want - got).abs())
        .fold(0.0_f64, f64::max);
    assert!(max_err < 1e-4, "dft round-trip max_err={max_err:.3e} exceeds 1e-4");
}

/// Same Standard `CoeffsToSlots` → `SlotsToCoeffs` round-trip as
/// [`dft_round_trip_standard`], but built with the **streamed** (unprepared-RHS)
/// constructor: the diagonals are materialized per factor at eval time instead
/// of kept resident. Exercises the generic `DFTMatrix<BE, R>` eval path over the
/// streamed `R` and confirms it matches the prepared result to CKKS precision.
#[test]
fn dft_round_trip_standard_streamed() {
    use std::collections::HashMap;

    use poulpy_ckks::{
        CKKSMeta,
        api::DFTOps,
        encoding::reim::Encoder,
        layouts::{DFTOutputFormat, DFTPlan, DFTType},
        test_suite::{
            CKKSTestParams,
            helpers::{alloc_scratch, ckks_decrypt_decode, ckks_encrypt, gen_atk, gen_sk_with_raw, test_vector_1},
        },
    };
    use poulpy_core::layouts::Base2K;
    use poulpy_hal::{
        api::ScratchOwnedBorrow,
        layouts::{CyclotomicOrder, Module},
    };

    use crate::{FFT64Ref, FFT64ReimTable};

    let base2k = 19usize;
    let log_delta = 30usize;
    let params = CKKSTestParams {
        n: 32,
        base2k,
        k: base2k * 18,
        prec: CKKSMeta {
            log_sparsity: 0,
            log_delta,
            log_budget: 10,
        },
        hw: 16,
        dsize: 1,
    };
    let m = params.n / 2;
    let log_slots = m.trailing_zeros() as usize;

    let module = Module::<FFT64Ref>::new(params.n as u64);
    let host_module = Module::<poulpy_hal::layouts::HostBytesBackend>::new(params.n as u64);
    let encoder = Encoder::<FFT64ReimTable<f64>>::new::<f64>(m).unwrap();

    let (sk_raw, sk) = gen_sk_with_raw(&params, &module, &host_module, [0u8; 32]);
    let mut scratch = alloc_scratch(&params, &module);

    let factor_meta = CKKSMeta {
        log_sparsity: 0,
        log_delta,
        log_budget: 10,
    };
    let make = |kind| DFTPlan {
        kind,
        factorization_depth: vec![1usize; log_slots],
        factor_giant_steps: vec![2usize; log_slots],
        format: DFTOutputFormat::Standard,
        scaling: None,
        bit_reversed: false,
        factor_log_delta: 0,
    };
    // Streamed constructors: note no `scratch` argument (nothing is prepared).
    let enc_dft = module.ckks_new_dft_matrix_streamed(
        &host_module,
        &encoder,
        Base2K(base2k as u32),
        factor_meta,
        &make(DFTType::Encode),
        &mut scratch.borrow(),
    );
    let dec_dft = module.ckks_new_dft_matrix_streamed(
        &host_module,
        &encoder,
        Base2K(base2k as u32),
        factor_meta,
        &make(DFTType::Decode),
        &mut scratch.borrow(),
    );

    let order = module.cyclotomic_order();
    let mut atks = HashMap::new();
    for p in enc_dft
        .galois_elements(order)
        .into_iter()
        .chain(dec_dft.galois_elements(order))
    {
        atks.entry(p)
            .or_insert_with(|| gen_atk(&params, &module, p, &sk_raw, &mut scratch.borrow()));
    }

    let (a_re, a_im) = test_vector_1::<f64>(m);
    let mut ct = ckks_encrypt(
        &params,
        &module,
        &host_module,
        &encoder,
        &sk,
        params.k,
        &a_re,
        &a_im,
        &mut scratch.borrow(),
    );

    // Same generic eval entry points accept the streamed matrix.
    module
        .ckks_coeffs_to_slots(&mut ct, &enc_dft, &atks, &mut scratch.borrow())
        .unwrap();
    module
        .ckks_slots_to_coeffs(&mut ct, &dec_dft, &atks, &mut scratch.borrow())
        .unwrap();

    let (got_re, got_im) = ckks_decrypt_decode::<FFT64Ref, f64, _>(&params, &module, &encoder, &ct, &sk, &mut scratch.borrow());

    let max_err = a_re
        .iter()
        .zip(&got_re)
        .chain(a_im.iter().zip(&got_im))
        .map(|(want, got)| (want - got).abs())
        .fold(0.0_f64, f64::max);
    assert!(max_err < 1e-4, "streamed dft round-trip max_err={max_err:.3e} exceeds 1e-4");
}

/// Proper homomorphic `CoeffsToSlots` test: encode the input **coefficient-wise** as
/// `bitReverse(re) || bitReverse(im)`, encrypt, apply CoeffsToSlots (Encode/IDFT),
/// then **slot**-decode and check it recovers `(re, im)`. This is basis-sensitive
/// (the bit-reversed coefficient encoding is part of the semantics) and validates
/// the full homomorphic pipeline against the true coeffs→slots meaning.
#[test]
fn dft_coeffs_to_slots_standard() {
    use std::collections::HashMap;

    use poulpy_ckks::{
        CKKSMeta,
        default::dft::{ckks_coeffs_to_slots_assign, ckks_new_dft_matrix},
        encoding::reim::Encoder,
        layouts::{DFTOutputFormat, DFTPlan, DFTType},
        test_suite::{
            CKKSTestParams,
            helpers::{alloc_scratch, ckks_decrypt_decode, ckks_encrypt_coeffs, gen_atk, gen_sk_with_raw, test_vector_1},
        },
    };
    use poulpy_core::layouts::Base2K;
    use poulpy_hal::{
        api::ScratchOwnedBorrow,
        layouts::{CyclotomicOrder, Module},
    };

    use crate::{FFT64Ref, FFT64ReimTable};

    let base2k = 19usize;
    let log_delta = 30usize;
    let params = CKKSTestParams {
        n: 32,
        base2k,
        k: base2k * 14,
        prec: CKKSMeta {
            log_sparsity: 0,
            log_delta,
            log_budget: 10,
        },
        hw: 16,
        dsize: 1,
    };
    let m = params.n / 2;
    let log_slots = m.trailing_zeros() as usize;

    let module = Module::<FFT64Ref>::new(params.n as u64);
    let host_module = Module::<poulpy_hal::layouts::HostBytesBackend>::new(params.n as u64);
    let encoder = Encoder::<FFT64ReimTable<f64>>::new::<f64>(m).unwrap();

    let (sk_raw, sk) = gen_sk_with_raw(&params, &module, &host_module, [0u8; 32]);
    let mut scratch = alloc_scratch(&params, &module);

    let factor_meta = CKKSMeta {
        log_sparsity: 0,
        log_delta,
        log_budget: 10,
    };
    let enc_dft = ckks_new_dft_matrix(
        &module,
        &host_module,
        &encoder,
        Base2K(base2k as u32),
        factor_meta,
        &DFTPlan {
            kind: DFTType::Encode,
            factorization_depth: vec![1usize; log_slots],
            factor_giant_steps: vec![2usize; log_slots],
            format: DFTOutputFormat::Standard,
            scaling: None,
            bit_reversed: false,
            factor_log_delta: 0,
        },
        &mut scratch.borrow(),
    );

    let order = module.cyclotomic_order();
    let mut atks = HashMap::new();
    for p in enc_dft.galois_elements(order) {
        atks.entry(p)
            .or_insert_with(|| gen_atk(&params, &module, p, &sk_raw, &mut scratch.borrow()));
    }

    // Coefficient-encode bitReverse(re) || bitReverse(im).
    let (re, im) = test_vector_1::<f64>(m);
    let bitrev = |j: usize| ((j as u32).reverse_bits() >> (u32::BITS - log_slots as u32)) as usize;
    let mut coeffs = vec![0.0f64; params.n];
    for j in 0..m {
        coeffs[j] = re[bitrev(j)];
        coeffs[j + m] = im[bitrev(j)];
    }
    let mut ct = ckks_encrypt_coeffs(
        &params,
        &module,
        &host_module,
        &sk,
        params.k,
        &coeffs,
        params.prec,
        &mut scratch.borrow(),
    );

    ckks_coeffs_to_slots_assign(&module, &mut ct, &enc_dft, &atks, &mut scratch.borrow()).unwrap();

    let (got_re, got_im) = ckks_decrypt_decode::<FFT64Ref, f64, _>(&params, &module, &encoder, &ct, &sk, &mut scratch.borrow());
    let max_err = re
        .iter()
        .zip(&got_re)
        .chain(im.iter().zip(&got_im))
        .map(|(want, got)| (want - got).abs())
        .fold(0.0_f64, f64::max);
    assert!(max_err < 1e-4, "coeffs_to_slots max_err={max_err:.3e} exceeds 1e-4");
}

/// `SplitRealAndImag` CoeffsToSlots: the real and imaginary parts come back in two
/// separate real-vector ciphertexts. Coefficient-encode `bitrev(re)||bitrev(im)`,
/// apply the split CoeffsToSlots, and check `ct_real` slots ≈ `re` and `ct_imag`
/// slots ≈ `im` (each with a ~0 imaginary slot component).
#[test]
fn dft_coeffs_to_slots_split() {
    use std::collections::HashMap;

    use poulpy_ckks::{
        CKKSMeta,
        default::dft::{ckks_coeffs_to_slots_split, ckks_new_dft_matrix},
        encoding::reim::Encoder,
        layouts::{DFTOutputFormat, DFTPlan, DFTType},
        test_suite::{
            CKKSTestParams,
            helpers::{
                alloc_ct, alloc_scratch, ckks_decrypt_decode, ckks_encrypt_coeffs, gen_atk, gen_sk_with_raw, test_vector_1,
            },
        },
    };
    use poulpy_core::layouts::Base2K;
    use poulpy_hal::{
        api::ScratchOwnedBorrow,
        layouts::{CyclotomicOrder, Module},
    };

    use crate::{FFT64Ref, FFT64ReimTable};

    let base2k = 19usize;
    let log_delta = 30usize;
    let params = CKKSTestParams {
        n: 32,
        base2k,
        k: base2k * 14,
        prec: CKKSMeta {
            log_sparsity: 0,
            log_delta,
            log_budget: 10,
        },
        hw: 16,
        dsize: 1,
    };
    let m = params.n / 2;
    let log_slots = m.trailing_zeros() as usize;

    let module = Module::<FFT64Ref>::new(params.n as u64);
    let host_module = Module::<poulpy_hal::layouts::HostBytesBackend>::new(params.n as u64);
    let encoder = Encoder::<FFT64ReimTable<f64>>::new::<f64>(m).unwrap();

    let (sk_raw, sk) = gen_sk_with_raw(&params, &module, &host_module, [0u8; 32]);
    let mut scratch = alloc_scratch(&params, &module);

    let factor_meta = CKKSMeta {
        log_sparsity: 0,
        log_delta,
        log_budget: 10,
    };
    let enc_dft = ckks_new_dft_matrix(
        &module,
        &host_module,
        &encoder,
        Base2K(base2k as u32),
        factor_meta,
        &DFTPlan {
            kind: DFTType::Encode,
            factorization_depth: vec![1usize; log_slots],
            factor_giant_steps: vec![2usize; log_slots],
            format: DFTOutputFormat::SplitRealAndImag,
            scaling: None,
            bit_reversed: false,
            factor_log_delta: 0,
        },
        &mut scratch.borrow(),
    );

    let order = module.cyclotomic_order();
    let mut atks = HashMap::new();
    for p in enc_dft.galois_elements(order) {
        atks.entry(p)
            .or_insert_with(|| gen_atk(&params, &module, p, &sk_raw, &mut scratch.borrow()));
    }
    let conj_key = gen_atk(&params, &module, -1, &sk_raw, &mut scratch.borrow());

    let (re, im) = test_vector_1::<f64>(m);
    let bitrev = |j: usize| ((j as u32).reverse_bits() >> (u32::BITS - log_slots as u32)) as usize;
    let mut coeffs = vec![0.0f64; params.n];
    for j in 0..m {
        coeffs[j] = re[bitrev(j)];
        coeffs[j + m] = im[bitrev(j)];
    }
    let ct_in = ckks_encrypt_coeffs(
        &params,
        &module,
        &host_module,
        &sk,
        params.k,
        &coeffs,
        params.prec,
        &mut scratch.borrow(),
    );

    let mut ct_real = alloc_ct(&params, &module, params.k);
    let mut ct_imag = alloc_ct(&params, &module, params.k);
    ckks_coeffs_to_slots_split(
        &module,
        &mut ct_real,
        &mut ct_imag,
        &ct_in,
        &enc_dft,
        &atks,
        &conj_key,
        &mut scratch.borrow(),
    )
    .unwrap();

    let (rr, ri) = ckks_decrypt_decode::<FFT64Ref, f64, _>(&params, &module, &encoder, &ct_real, &sk, &mut scratch.borrow());
    let (ir, ii) = ckks_decrypt_decode::<FFT64Ref, f64, _>(&params, &module, &encoder, &ct_imag, &sk, &mut scratch.borrow());

    let err = |a: &[f64], b: &[f64]| a.iter().zip(b).map(|(x, y)| (x - y).abs()).fold(0.0_f64, f64::max);
    let zero = vec![0.0f64; m];
    let e_real = err(&rr, &re).max(err(&ri, &zero));
    let e_imag = err(&ir, &im).max(err(&ii, &zero));
    assert!(e_real < 1e-4, "split real part err={e_real:.3e}");
    assert!(e_imag < 1e-4, "split imag part err={e_imag:.3e}");
}

/// `SplitRealAndImag` round-trip: `CoeffsToSlots` (split) then `SlotsToCoeffs`
/// (split) recovers the original coefficient vector. Validates
/// `ckks_slots_to_coeffs_split` (the real/imag *combine* + Decode) against the
/// raw coefficients.
#[test]
fn dft_split_round_trip() {
    use std::collections::HashMap;

    use poulpy_ckks::{
        CKKSInfos, CKKSMeta,
        default::dft::{ckks_coeffs_to_slots_split, ckks_new_dft_matrix, ckks_slots_to_coeffs_split},
        encoding::reim::Encoder,
        layouts::{CKKSPlaintextVecHostCodec, DFTOutputFormat, DFTPlan, DFTType},
        test_suite::{
            CKKSTestParams,
            helpers::{
                alloc_ct, alloc_scratch, ckks_decrypt_with_prec, ckks_encrypt_coeffs, gen_atk, gen_sk_with_raw, test_vector_1,
            },
        },
    };
    use poulpy_core::layouts::Base2K;
    use poulpy_hal::{
        api::ScratchOwnedBorrow,
        layouts::{CyclotomicOrder, Module},
    };

    use crate::{FFT64Ref, FFT64ReimTable};

    let base2k = 19usize;
    let log_delta = 30usize;
    let params = CKKSTestParams {
        n: 32,
        base2k,
        k: base2k * 18,
        prec: CKKSMeta {
            log_sparsity: 0,
            log_delta,
            log_budget: 10,
        },
        hw: 16,
        dsize: 1,
    };
    let m = params.n / 2;
    let log_slots = m.trailing_zeros() as usize;

    let module = Module::<FFT64Ref>::new(params.n as u64);
    let host_module = Module::<poulpy_hal::layouts::HostBytesBackend>::new(params.n as u64);
    let encoder = Encoder::<FFT64ReimTable<f64>>::new::<f64>(m).unwrap();

    let (sk_raw, sk) = gen_sk_with_raw(&params, &module, &host_module, [0u8; 32]);
    let mut scratch = alloc_scratch(&params, &module);

    let factor_meta = CKKSMeta {
        log_sparsity: 0,
        log_delta,
        log_budget: 10,
    };
    let make = |kind| DFTPlan {
        kind,
        factorization_depth: vec![1usize; log_slots],
        factor_giant_steps: vec![2usize; log_slots],
        format: DFTOutputFormat::SplitRealAndImag,
        scaling: None,
        bit_reversed: false,
        factor_log_delta: 0,
    };
    let enc_dft = ckks_new_dft_matrix(
        &module,
        &host_module,
        &encoder,
        Base2K(base2k as u32),
        factor_meta,
        &make(DFTType::Encode),
        &mut scratch.borrow(),
    );
    let dec_dft = ckks_new_dft_matrix(
        &module,
        &host_module,
        &encoder,
        Base2K(base2k as u32),
        factor_meta,
        &make(DFTType::Decode),
        &mut scratch.borrow(),
    );

    let order = module.cyclotomic_order();
    let mut atks = HashMap::new();
    for p in enc_dft
        .galois_elements(order)
        .into_iter()
        .chain(dec_dft.galois_elements(order))
    {
        atks.entry(p)
            .or_insert_with(|| gen_atk(&params, &module, p, &sk_raw, &mut scratch.borrow()));
    }
    let conj_key = gen_atk(&params, &module, -1, &sk_raw, &mut scratch.borrow());

    let (re, im) = test_vector_1::<f64>(m);
    let bitrev = |j: usize| ((j as u32).reverse_bits() >> (u32::BITS - log_slots as u32)) as usize;
    let mut coeffs = vec![0.0f64; params.n];
    for j in 0..m {
        coeffs[j] = re[bitrev(j)];
        coeffs[j + m] = im[bitrev(j)];
    }
    let ct_in = ckks_encrypt_coeffs(
        &params,
        &module,
        &host_module,
        &sk,
        params.k,
        &coeffs,
        params.prec,
        &mut scratch.borrow(),
    );

    let mut ct_real = alloc_ct(&params, &module, params.k);
    let mut ct_imag = alloc_ct(&params, &module, params.k);
    ckks_coeffs_to_slots_split(
        &module,
        &mut ct_real,
        &mut ct_imag,
        &ct_in,
        &enc_dft,
        &atks,
        &conj_key,
        &mut scratch.borrow(),
    )
    .unwrap();

    let mut op_out = alloc_ct(&params, &module, params.k);
    ckks_slots_to_coeffs_split(
        &module,
        &mut op_out,
        &ct_real,
        &ct_imag,
        &dec_dft,
        &atks,
        &mut scratch.borrow(),
    )
    .unwrap();

    // Read op_out's raw coefficients and compare to the original input coefficients.
    let prec = CKKSMeta {
        log_sparsity: 0,
        log_delta: op_out.log_delta(),
        log_budget: op_out.log_budget().min(params.prec.log_budget),
    };
    let pt = ckks_decrypt_with_prec(&module, &op_out, &sk, prec, &mut scratch.borrow()).unwrap();
    let mut got = vec![0.0f64; params.n];
    pt.decode_host_floats(&mut got).unwrap();

    let max_err = coeffs.iter().zip(&got).map(|(a, b)| (a - b).abs()).fold(0.0_f64, f64::max);
    assert!(max_err < 1e-4, "split round-trip coeff max_err={max_err:.3e}");
}

/// Basis-sensitive check: the generated **Encode** (IDFT) matrix computes the
/// same slots→coefficients map that the reim [`Encoder`] defines, in poulpy's
/// canonical slot basis (up to the encoder's bit-reversal and the `1/N` scale).
/// The encoder is an *independent* oracle (it does not use the DFT-matrix
/// generator), so unlike the homomorphic `Encode∘Decode=I` round-trip — which
/// holds in any basis — this catches a basis/permutation mismatch.
///
/// Ground truth: the coefficients of the polynomial whose slots are `(re, im)`,
/// obtained by `encode_reim` then reading the raw coefficients, viewed as `m`
/// complex numbers `coeffs[j] + i·coeffs[j+m]`. They must equal the generated
/// Encode matrix applied to `(re, im)`, indexed bit-reversed, up to a single
/// global complex scale.
#[test]
fn dft_encode_matches_encoder_basis() {
    use poulpy_ckks::{
        CKKSMeta,
        default::gen_dft_matrices,
        encoding::reim::Encoder,
        layouts::{CKKSModuleAlloc, CKKSPlaintextVecHostCodec, DFTOutputFormat, DFTPlan, DFTType},
    };
    use poulpy_core::layouts::{Base2K, Evaluate, LinearTransformationStrategy};
    use poulpy_hal::layouts::{HostBytesBackend, Module};

    use crate::FFT64ReimTable;

    let m = 16usize;
    let log_slots = m.trailing_zeros() as usize;
    let n = 2 * m;

    let encoder = Encoder::<FFT64ReimTable<f64>>::new::<f64>(m).unwrap();
    let host = Module::<HostBytesBackend>::new(n as u64);

    let re: Vec<f64> = (0..m).map(|j| (0.3 * (j as f64 + 1.0)).sin()).collect();
    let im: Vec<f64> = (0..m).map(|j| (0.7 * (j as f64 + 2.0)).cos()).collect();

    // Ground-truth coefficients (high precision so quantization is negligible).
    let mut pt = host.ckks_pt_vec_alloc(
        Base2K(55),
        CKKSMeta {
            log_sparsity: 0,
            log_delta: 50,
            log_budget: 0,
        },
    );
    encoder.encode_reim(&mut pt, &re, &im).unwrap();
    let mut coeffs = vec![0.0f64; n];
    pt.decode_host_floats(&mut coeffs).unwrap();
    // m-complex view of the 2m real coefficients.
    let gt: Vec<(f64, f64)> = (0..m).map(|j| (coeffs[j], coeffs[j + m])).collect();

    // The generated Encode (IDFT) matrix, applied in the clear.
    let enc_lit = DFTPlan {
        kind: DFTType::Encode,
        factorization_depth: vec![1usize; log_slots],
        factor_giant_steps: vec![2usize; log_slots],
        format: DFTOutputFormat::Standard,
        scaling: None,
        bit_reversed: false,
        factor_log_delta: 0,
    };
    // Dense full packing for this clear-text basis check: log_n = log_slots + 1.
    let factors = gen_dft_matrices(&enc_lit, log_slots + 1);
    let (mut wre, mut wim) = (re.clone(), im.clone());
    for f in &factors {
        let (r, i) = f.evaluate((wre.as_slice(), wim.as_slice()), LinearTransformationStrategy::Direct);
        wre = r;
        wim = i;
    }
    let w: Vec<(f64, f64)> = wre.iter().zip(&wim).map(|(&r, &i)| (r, i)).collect();

    // The encoder's coefficient order is bit-reversed relative to the generated
    // matrix's natural slot order (poulpy's `slot_map` includes `.reverse_bits()`).
    // Comparing `w[bitrev(j)]` against the ground-truth coefficients must agree up
    // to a single global complex scale (the 1/N convention).
    let bitrev = |j: usize| ((j as u32).reverse_bits() >> (u32::BITS - log_slots as u32)) as usize;
    let cmul = |a: (f64, f64), b: (f64, f64)| (a.0 * b.0 - a.1 * b.1, a.0 * b.1 + a.1 * b.0);
    let k0 = (0..m)
        .max_by(|&a, &b| w[a].0.hypot(w[a].1).total_cmp(&w[b].0.hypot(w[b].1)))
        .unwrap();
    let o = w[bitrev(k0)];
    let den = o.0 * o.0 + o.1 * o.1;
    let s = (
        (gt[k0].0 * o.0 + gt[k0].1 * o.1) / den,
        (gt[k0].1 * o.0 - gt[k0].0 * o.1) / den,
    );
    let scale_mag = gt[k0].0.hypot(gt[k0].1).max(1e-12);
    let max_err = (0..m)
        .map(|j| {
            let ws = cmul(w[bitrev(j)], s);
            (ws.0 - gt[j].0).hypot(ws.1 - gt[j].1) / scale_mag
        })
        .fold(0.0_f64, f64::max);
    assert!(
        max_err < 1e-9,
        "generated Encode matrix does not match the encoder basis (rel err {max_err:.3e}, scale ({:.4},{:.4}))",
        s.0,
        s.1
    );
}

/// Sparse `RepackImagAsReal` CoeffsToSlots: coefficient-encode a sparsely-packed
/// `slots`-vector `bitrev(re)||bitrev(im)`, apply the repack CoeffsToSlots, and
/// decode the result at `2·slots` resolution — the real part must hold `re` in the
/// left `slots` and `im` in the right `slots` (the imag-into-right-half repack).
#[test]
fn dft_coeffs_to_slots_repack_sparse() {
    use std::collections::HashMap;

    use poulpy_ckks::{
        CKKSInfos, CKKSMeta, SetCKKSInfos,
        default::dft::{ckks_coeffs_to_slots_repack, ckks_new_dft_matrix},
        encoding::reim::Encoder,
        layouts::{DFTOutputFormat, DFTPlan, DFTType},
        test_suite::{
            CKKSTestParams,
            helpers::{
                alloc_ct, alloc_scratch, ckks_decrypt_with_prec, ckks_encrypt_coeffs, gen_atk, gen_sk_with_raw, test_vector_1,
            },
        },
    };
    use poulpy_core::layouts::Base2K;
    use poulpy_hal::{
        api::ScratchOwnedBorrow,
        layouts::{CyclotomicOrder, Module},
    };

    use crate::{FFT64Ref, FFT64ReimTable};

    let base2k = 19usize;
    let log_delta = 30usize;
    // n = 64 → log_max_slots = 5; log_slots = 2 (slots = 4) → sparse.
    let params = CKKSTestParams {
        n: 64,
        base2k,
        k: base2k * 14,
        prec: CKKSMeta {
            log_delta,
            log_budget: 10,
            log_sparsity: 3,
        },
        hw: 32,
        dsize: 1,
    };
    let log_slots = 2usize;
    let slots = 1usize << log_slots;

    let module = Module::<FFT64Ref>::new(params.n as u64);
    let host_module = Module::<poulpy_hal::layouts::HostBytesBackend>::new(params.n as u64);
    let encoder = Encoder::<FFT64ReimTable<f64>>::new::<f64>(params.n / 2).unwrap();

    let (sk_raw, sk) = gen_sk_with_raw(&params, &module, &host_module, [0u8; 32]);
    let mut scratch = alloc_scratch(&params, &module);

    let factor_meta = CKKSMeta {
        log_delta,
        log_budget: 10,
        log_sparsity: 0,
    };
    let enc_dft = ckks_new_dft_matrix(
        &module,
        &host_module,
        &encoder,
        Base2K(base2k as u32),
        factor_meta,
        &DFTPlan {
            kind: DFTType::Encode,
            factorization_depth: vec![1usize; log_slots],
            factor_giant_steps: vec![2usize; log_slots],
            format: DFTOutputFormat::RepackImagAsReal,
            scaling: None,
            bit_reversed: false,
            factor_log_delta: 0,
        },
        &mut scratch.borrow(),
    );
    assert!(enc_dft.is_sparse(), "expected sparse path");

    let order = module.cyclotomic_order();
    let mut atks = HashMap::new();
    for p in enc_dft.galois_elements(order) {
        atks.entry(p)
            .or_insert_with(|| gen_atk(&params, &module, p, &sk_raw, &mut scratch.borrow()));
    }
    let conj_key = gen_atk(&params, &module, -1, &sk_raw, &mut scratch.borrow());

    // Coefficient-encode bitrev(re) || bitrev(im) for the `slots`-vector (sparse).
    let (re_full, im_full) = test_vector_1::<f64>(params.n / 2);
    let (re, im) = (&re_full[..slots], &im_full[..slots]);
    // Sparse coefficient layout: bitrev(re) at gap, bitrev(im) at N/2 + gap.
    let bitrev = |j: usize| ((j as u32).reverse_bits() >> (u32::BITS - log_slots as u32)) as usize;
    let gap = params.n / (2 * slots);
    let mut coeffs = vec![0.0f64; params.n];
    for j in 0..slots {
        coeffs[j * gap] = re[bitrev(j)];
        coeffs[params.n / 2 + j * gap] = im[bitrev(j)];
    }
    let mut ct_in = ckks_encrypt_coeffs(
        &params,
        &module,
        &host_module,
        &sk,
        params.k,
        &coeffs,
        params.prec,
        &mut scratch.borrow(),
    );
    ct_in.set_log_sparsity(3);

    let mut ct_out = alloc_ct(&params, &module, params.k);
    ckks_coeffs_to_slots_repack(
        &module,
        &mut ct_out,
        &ct_in,
        &enc_dft,
        &atks,
        &conj_key,
        &mut scratch.borrow(),
    )
    .unwrap();

    // Decode at 2·slots resolution: real part = [re | im], imag part ≈ 0.
    let small = Encoder::<FFT64ReimTable<f64>>::new::<f64>(2 * slots).unwrap();
    let prec = CKKSMeta {
        log_delta: ct_out.log_delta(),
        log_budget: ct_out.log_budget().min(params.prec.log_budget),
        log_sparsity: ct_out.log_sparsity(),
    };
    let pt = ckks_decrypt_with_prec(&module, &ct_out, &sk, prec, &mut scratch.borrow()).unwrap();
    let mut gr = vec![0.0f64; 2 * slots];
    let mut gi = vec![0.0f64; 2 * slots];
    small.decode_reim_sparse(&pt, &mut gr, &mut gi).unwrap();

    let mut max_err = 0.0f64;
    for j in 0..slots {
        max_err = max_err.max((gr[j] - re[j]).abs()).max((gr[slots + j] - im[j]).abs());
    }
    let imag_err = gi.iter().fold(0.0f64, |m, &x| m.max(x.abs()));
    assert!(
        max_err < 1e-3 && imag_err < 1e-3,
        "repack: re|im err={max_err:.3e} imag={imag_err:.3e}; gr={gr:?} gi={gi:?}"
    );
}

/// Sparse `RepackImagAsReal` round-trip: CoeffsToSlots-repack then
/// SlotsToCoeffs-repack recovers the original sparse coefficient vector, and the
/// `log_sparsity` returns to its starting value (down 1 then up 1).
#[test]
fn dft_repack_round_trip_sparse() {
    use std::collections::HashMap;

    use poulpy_ckks::{
        CKKSInfos, CKKSMeta, SetCKKSInfos,
        default::dft::{ckks_coeffs_to_slots_repack, ckks_new_dft_matrix, ckks_slots_to_coeffs_repack},
        encoding::reim::Encoder,
        layouts::{CKKSPlaintextVecHostCodec, DFTOutputFormat, DFTPlan, DFTType},
        test_suite::{
            CKKSTestParams,
            helpers::{
                alloc_ct, alloc_scratch, ckks_decrypt_with_prec, ckks_encrypt_coeffs, gen_atk, gen_sk_with_raw, test_vector_1,
            },
        },
    };
    use poulpy_core::layouts::Base2K;
    use poulpy_hal::{
        api::ScratchOwnedBorrow,
        layouts::{CyclotomicOrder, Module},
    };

    use crate::{FFT64Ref, FFT64ReimTable};

    let base2k = 19usize;
    let log_delta = 30usize;
    let params = CKKSTestParams {
        n: 64,
        base2k,
        k: base2k * 18,
        prec: CKKSMeta {
            log_delta,
            log_budget: 10,
            log_sparsity: 3,
        },
        hw: 32,
        dsize: 1,
    };
    let log_slots = 2usize;
    let slots = 1usize << log_slots;

    let module = Module::<FFT64Ref>::new(params.n as u64);
    let host_module = Module::<poulpy_hal::layouts::HostBytesBackend>::new(params.n as u64);
    let encoder = Encoder::<FFT64ReimTable<f64>>::new::<f64>(params.n / 2).unwrap();

    let (sk_raw, sk) = gen_sk_with_raw(&params, &module, &host_module, [0u8; 32]);
    let mut scratch = alloc_scratch(&params, &module);

    let factor_meta = CKKSMeta {
        log_delta,
        log_budget: 10,
        log_sparsity: 0,
    };
    let mk = |kind| DFTPlan {
        kind,
        factorization_depth: vec![1usize; log_slots],
        factor_giant_steps: vec![2usize; log_slots],
        format: DFTOutputFormat::RepackImagAsReal,
        scaling: None,
        bit_reversed: false,
        factor_log_delta: 0,
    };
    let enc_dft = ckks_new_dft_matrix(
        &module,
        &host_module,
        &encoder,
        Base2K(base2k as u32),
        factor_meta,
        &mk(DFTType::Encode),
        &mut scratch.borrow(),
    );
    let dec_dft = ckks_new_dft_matrix(
        &module,
        &host_module,
        &encoder,
        Base2K(base2k as u32),
        factor_meta,
        &mk(DFTType::Decode),
        &mut scratch.borrow(),
    );

    let order = module.cyclotomic_order();
    let mut atks = HashMap::new();
    for p in enc_dft
        .galois_elements(order)
        .into_iter()
        .chain(dec_dft.galois_elements(order))
    {
        atks.entry(p)
            .or_insert_with(|| gen_atk(&params, &module, p, &sk_raw, &mut scratch.borrow()));
    }
    let conj_key = gen_atk(&params, &module, -1, &sk_raw, &mut scratch.borrow());

    let (re_full, im_full) = test_vector_1::<f64>(params.n / 2);
    let (re, im) = (&re_full[..slots], &im_full[..slots]);
    let bitrev = |j: usize| ((j as u32).reverse_bits() >> (u32::BITS - log_slots as u32)) as usize;
    let gap = params.n / (2 * slots);
    let mut coeffs = vec![0.0f64; params.n];
    for j in 0..slots {
        coeffs[j * gap] = re[bitrev(j)];
        coeffs[params.n / 2 + j * gap] = im[bitrev(j)];
    }
    let mut ct_in = ckks_encrypt_coeffs(
        &params,
        &module,
        &host_module,
        &sk,
        params.k,
        &coeffs,
        params.prec,
        &mut scratch.borrow(),
    );
    ct_in.set_log_sparsity(3);

    let mut ct_mid = alloc_ct(&params, &module, params.k);
    ckks_coeffs_to_slots_repack(
        &module,
        &mut ct_mid,
        &ct_in,
        &enc_dft,
        &atks,
        &conj_key,
        &mut scratch.borrow(),
    )
    .unwrap();
    assert_eq!(ct_mid.log_sparsity(), 2, "slots doubled");

    let mut ct_out = alloc_ct(&params, &module, params.k);
    ckks_slots_to_coeffs_repack(&module, &mut ct_out, &ct_mid, &dec_dft, &atks, &mut scratch.borrow()).unwrap();
    assert_eq!(ct_out.log_sparsity(), 3, "slots halved back");

    // Compare ct_out's raw coefficients to the original input coefficients.
    let prec = CKKSMeta {
        log_delta: ct_out.log_delta(),
        log_budget: ct_out.log_budget().min(params.prec.log_budget),
        log_sparsity: ct_out.log_sparsity(),
    };
    let pt = ckks_decrypt_with_prec(&module, &ct_out, &sk, prec, &mut scratch.borrow()).unwrap();
    let mut got = vec![0.0f64; params.n];
    pt.decode_host_floats(&mut got).unwrap();
    let max_err = coeffs.iter().zip(&got).map(|(a, b)| (a - b).abs()).fold(0.0f64, f64::max);
    assert!(max_err < 1e-3, "repack round-trip coeff max_err={max_err:.3e}");
}

/// The sparse reim codec round-trips: `m` sub-ring slots encode (via the gap
/// `R[Y]→Z[X]` mapping) into a larger degree-`N` plaintext and decode back, with
/// no `N/2` slot expansion. Foundation for sparse-packed homomorphic DFT.
#[test]
fn reim_sparse_codec_roundtrip() {
    use poulpy_ckks::{CKKSMeta, encoding::reim::Encoder, layouts::CKKSModuleAlloc};
    use poulpy_core::layouts::{Base2K, LWEInfos};
    use poulpy_hal::layouts::{HostBytesBackend, Module};

    use crate::FFT64ReimTable;

    let n = 32usize; // ring degree (16 max slots)
    let m_small = 4usize; // sub-ring slots (sparse: 4 < 16)
    let host = Module::<HostBytesBackend>::new(n as u64);
    let encoder = Encoder::<FFT64ReimTable<f64>>::new::<f64>(m_small).unwrap();

    let mut pt = host.ckks_pt_vec_alloc(
        Base2K(50),
        CKKSMeta {
            log_delta: 40,
            log_budget: 8,
            log_sparsity: 0,
        },
    );
    assert_eq!(pt.n().as_usize(), n);

    let re = [1.0_f64, -2.0, 0.5, 3.0];
    let im = [0.5_f64, 2.0, -1.0, 1.0];
    encoder.encode_reim_sparse(&mut pt, &re, &im).unwrap();

    let mut ro = [0.0_f64; 4];
    let mut io = [0.0_f64; 4];
    encoder.decode_reim_sparse(&pt, &mut ro, &mut io).unwrap();

    let err = re
        .iter()
        .zip(&ro)
        .chain(im.iter().zip(&io))
        .map(|(a, b)| (a - b).abs())
        .fold(0.0_f64, f64::max);
    assert!(err < 1e-9, "sparse reim round-trip err={err:.3e}; re={ro:?} im={io:?}");
}

ckks_backend_test_suite!(
    mod fft64_f64,
    backend = crate::FFT64Ref,
    scalar = f64,
    encoder = crate::FFT64ReimTable<f64>,
    params = poulpy_ckks::test_suite::FFT64_PARAMS_F64,
    rotations = super::ATK_ROTATIONS,
);

ckks_backend_test_suite!(
    mod ntt120_f64,
    backend = crate::NTT120Ref,
    scalar = f64,
    encoder = crate::FFT64ReimTable<f64>,
    params = poulpy_ckks::test_suite::NTT120_PARAMS_F64,
    rotations = super::ATK_ROTATIONS,
);

ckks_backend_test_suite!(
    mod ntt120_f128,
    backend = crate::NTT120Ref,
    scalar = f128::f128,
    encoder = crate::FFT64ReimTable<f128::f128>,
    params = poulpy_ckks::test_suite::NTT120_PARAMS_F128,
    rotations = super::ATK_ROTATIONS,
);
