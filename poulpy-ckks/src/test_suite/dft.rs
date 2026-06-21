//! Homomorphic DFT (CoeffsToSlots / SlotsToCoeffs) tests.
//!
//! Backend-generic, **directional** correctness checks (not round trips): each
//! transform is validated against an *independent* plaintext reference, so a
//! basis/permutation/scale error that would cancel in an `Encode∘Decode` round
//! trip is caught.
//!
//! - **CoeffsToSlots** (Encode): a ciphertext whose *coefficients* hold the
//!   plaintext layout of a slot vector `(re, im)` must, after the transform,
//!   decode to those slots `(re, im)`.
//! - **SlotsToCoeffs** (Decode): a ciphertext whose *slots* hold `(re, im)` must,
//!   after the transform, hold the matching coefficient layout.
//!
//! Each test is self-contained and drives the transform through the public
//! [`DFTOps`] method surface. The parameters come from one of two centralized
//! builders, each shared by the CoeffsToSlots and SlotsToCoeffs directions:
//! [`dense_params`] (full slot count, `Standard` / `Split` formats) and
//! [`sparse_params`] (sub-maximal slots, the `RepackImagAsReal` path).

use std::collections::HashMap;

use poulpy_core::layouts::Base2K;
use poulpy_hal::{
    api::{NegacyclicFFT, NegacyclicFFTNew, ScratchOwnedBorrow},
    layouts::{Backend, CyclotomicOrder, HostBytesBackend, HostDataMut, HostDataRef, Module},
};

use poulpy_core::{GLWENoise, layouts::LWEInfos};

use crate::{
    CKKSInfos, CKKSMeta, SetCKKSInfos,
    api::DFTOps,
    encoding::reim::Encoder,
    layouts::{
        CKKSCiphertext, CKKSModuleAlloc, CKKSPlaintext, CKKSPlaintextVecHostCodec, DFTMatrix, DFTOutputFormat, DFTPlan, DFTType,
        Decode, Encode, Repack, Split, Standard,
    },
    test_suite::{
        CKKSTestParams,
        helpers::{
            TestContextBackend, TestContextHostModule, TestContextModule, TestScalar, alloc_ct, alloc_scratch, ckks_encrypt,
            ckks_encrypt_coeffs, ckks_encrypt_pt, gen_atk, gen_sk_with_raw, test_vector_1,
        },
    },
};

/// `log2` of the live complex slot count for the **dense** tests (`Standard` /
/// `SplitRealAndImag`): all `n/2 = 2^DENSE_LOG_SLOTS` slots are live, so the
/// transform factorizes into this many factors over a ring of degree `n = 2·2^…`.
const DENSE_LOG_SLOTS: usize = 8;

/// `log2` of the live complex slot count for the **sparse** repack tests: a
/// sub-maximal slot count (`< log_max_slots`) that drives the `RepackImagAsReal`
/// path. Combined with [`sparse_params`]'s `n = 64` (`log_max_slots = 5`).
const SPARSE_LOG_SLOTS: usize = 2;

/// Dense parameter set (full slot count, `Standard` / `Split` formats), shared by
/// the CoeffsToSlots and SlotsToCoeffs dense tests. Keeps the backend's `base2k` /
/// `log_delta` but picks `n = 2·2^DENSE_LOG_SLOTS` and a `k` sized for the
/// `DENSE_LOG_SLOTS` chained factor plaintext-multiplies plus input scale + headroom.
fn dense_params(params: &CKKSTestParams) -> CKKSTestParams {
    let base2k = params.base2k;
    let log_delta = params.prec.log_delta;
    let k = (log_delta * (DENSE_LOG_SLOTS + 3)).next_multiple_of(base2k);
    CKKSTestParams {
        n: 1 << (DENSE_LOG_SLOTS + 1),
        base2k,
        k,
        prec: CKKSMeta {
            log_sparsity: 0,
            log_delta,
            log_budget: 10,
        },
        hw: params.hw.min(1 << DENSE_LOG_SLOTS),
        dsize: params.dsize,
    }
}

/// Sparse `RepackImagAsReal` parameter set: `n = 64` (`log_max_slots = 5`) with only
/// `2^SPARSE_LOG_SLOTS` slots live, so the transform takes the sparse repack path.
/// Shared by the CoeffsToSlots and SlotsToCoeffs repack tests.
fn sparse_params(params: &CKKSTestParams) -> CKKSTestParams {
    let base2k = params.base2k;
    let log_delta = params.prec.log_delta;
    let k = (log_delta * 7).next_multiple_of(base2k);
    CKKSTestParams {
        n: 64,
        base2k,
        k,
        prec: CKKSMeta {
            log_sparsity: 3,
            log_delta,
            log_budget: 10,
        },
        hw: params.hw.min(32),
        dsize: params.dsize,
    }
}

/// `CKKSMeta` for the factor matrices: the per-factor scale, minimal budget.
fn factor_meta(log_delta: usize) -> CKKSMeta {
    CKKSMeta {
        log_sparsity: 0,
        log_delta,
        log_budget: 10,
    }
}

/// A factorized (I)DFT plan over `log_slots` factors: one FFT layer per factor
/// (no merging), BSGS width 2.
fn plan(log_slots: usize, kind: DFTType, format: DFTOutputFormat, log_delta: usize) -> DFTPlan {
    DFTPlan {
        kind,
        factorization_depth: vec![1usize; log_slots],
        giant_steps: vec![2usize; log_slots],
        format,
        scaling: None,
        bit_reversed: false,
        meta: factor_meta(log_delta),
    }
}

/// Bit-reversal permutation over `DENSE_LOG_SLOTS` bits (poulpy's slot map order).
fn bitrev(j: usize) -> usize {
    ((j as u32).reverse_bits() >> (u32::BITS - DENSE_LOG_SLOTS as u32)) as usize
}

/// Coefficient layout of the slot vector `(re, im)`: `bitrev(re) || bitrev(im)`.
fn coeff_layout<F: TestScalar>(re: &[F], im: &[F], n: usize) -> Vec<F> {
    let m = n / 2;
    let mut coeffs = vec![F::from_f64(0.0).unwrap(); n];
    for j in 0..m {
        coeffs[j] = re[bitrev(j)];
        coeffs[j + m] = im[bitrev(j)];
    }
    coeffs
}

/// Upper bound (in log2 bits) on the acceptable output noise (`std`) for a
/// `log_delta`-bit transform.
///
/// These are *structural* oracle tests: a basis, permutation, or scale error
/// makes the output disagree with the reference at signal level, i.e. noise
/// `log2 ≈ 0`. A correct transform stays far below that (empirically `≤ -2·
/// log_delta` across backends). The bound `-log_delta + 16` sits comfortably
/// between the two — tight enough to flag any structural break, loose enough to
/// be robust to per-backend noise-floor differences. A precise noise-growth
/// regression would need an analytic model (cf. the keyswitch suite); that is
/// not this test's job.
fn noise_bound(log_delta: usize) -> f64 {
    -(log_delta as f64) + 16.0
}

/// Allocates a CKKS plaintext at the same `(base2k, log_delta, log_budget,
/// log_sparsity)` as `ct` — the scale [`GLWENoise`] needs the expected value at.
fn want_plaintext<BE>(module: &Module<BE>, ct: &CKKSCiphertext<BE::OwnedBuf>) -> CKKSPlaintext<BE::OwnedBuf>
where
    BE: TestContextBackend,
    Module<BE>: CKKSModuleAlloc<BE>,
{
    module.ckks_pt_vec_alloc(
        ct.base2k(),
        CKKSMeta {
            log_sparsity: ct.log_sparsity(),
            log_delta: ct.log_delta(),
            log_budget: ct.log_budget(),
        },
    )
}

/// **CoeffsToSlots**, `Standard` format: coefficient-encode the layout of a slot
/// vector `(re, im)`, apply CoeffsToSlots, and check the result decodes to those
/// slots `(re, im)` — the independent reference.
pub fn test_dft_coeffs_to_slots_standard<BE, F, E>(
    params: CKKSTestParams,
    _module: &Module<BE>,
    _host_module: &Module<HostBytesBackend>,
) where
    BE: TestContextBackend,
    Module<BE>: TestContextModule<BE> + DFTOps<BE> + GLWENoise<BE>,
    Module<HostBytesBackend>: TestContextHostModule,
    F: TestScalar,
    E: NegacyclicFFT<F> + NegacyclicFFTNew<F>,
    for<'a> <BE as Backend>::BufRef<'a>: HostDataRef,
    for<'a> <BE as Backend>::BufMut<'a>: HostDataMut,
    CKKSPlaintext<Vec<u8>>: CKKSPlaintextVecHostCodec<f64>,
{
    let params = dense_params(&params);
    let m = params.n / 2;
    let base2k = params.base2k;
    let log_delta = params.prec.log_delta;

    let module = Module::<BE>::new(params.n as u64);
    let host_module = Module::<HostBytesBackend>::new(params.n as u64);
    let encoder = Encoder::<E>::new::<F>(m).unwrap();

    let (sk_raw, sk) = gen_sk_with_raw(&params, &module, &host_module, [0u8; 32]);
    let mut scratch = alloc_scratch(&params, &module);

    let enc_lt: DFTMatrix<BE, Encode, Standard> = module
        .ckks_new_dft_matrix(
            &host_module,
            &encoder,
            Base2K(base2k as u32),
            &plan(DENSE_LOG_SLOTS, DFTType::Encode, DFTOutputFormat::Standard, log_delta),
            &mut scratch.borrow(),
        )
        .unwrap();
    let enc_dft = module.ckks_prepare_dft_matrix(&enc_lt, &mut scratch.borrow());

    let order = module.cyclotomic_order();
    let mut atks = HashMap::new();
    for p in enc_dft.galois_elements(order) {
        atks.entry(p)
            .or_insert_with(|| gen_atk(&params, &module, p, &sk_raw, &mut scratch.borrow()));
    }

    // Input: a ciphertext whose coefficients hold the layout of slots (re, im).
    let (re, im) = test_vector_1::<F>(m);
    let coeffs = coeff_layout(&re, &im, params.n);
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

    module
        .ckks_coeffs_to_slots(&mut ct, &enc_dft, &atks, &mut scratch.borrow())
        .unwrap();

    // Reference: a plaintext holding the expected slots (re, im) at the output
    // scale. Measure the error via GLWE noise (std of the per-coefficient error,
    // computed in arbitrary precision) and bound its log2.
    let mut pt_want = want_plaintext(&module, &ct);
    encoder.encode_reim(&mut pt_want, &re, &im).unwrap();
    let noise = module.glwe_noise(&ct, &pt_want, &sk, &mut scratch.borrow()).std().log2();
    let bound = noise_bound(log_delta);
    assert!(
        noise < bound,
        "coeffs_to_slots (Standard) noise log2={noise:.1} (bound {bound:.1})"
    );
}

/// **SlotsToCoeffs**, `Standard` format: slot-encode `(re, im)`, apply
/// SlotsToCoeffs, and check the result's coefficients hold the matching layout
/// `bitrev(re) || bitrev(im)` — the independent reference (the inverse of
/// [`test_dft_coeffs_to_slots_standard`]).
pub fn test_dft_slots_to_coeffs_standard<BE, F, E>(
    params: CKKSTestParams,
    _module: &Module<BE>,
    _host_module: &Module<HostBytesBackend>,
) where
    BE: TestContextBackend,
    Module<BE>: TestContextModule<BE> + DFTOps<BE> + GLWENoise<BE>,
    Module<HostBytesBackend>: TestContextHostModule,
    F: TestScalar,
    E: NegacyclicFFT<F> + NegacyclicFFTNew<F>,
    for<'a> <BE as Backend>::BufRef<'a>: HostDataRef,
    for<'a> <BE as Backend>::BufMut<'a>: HostDataMut,
    CKKSPlaintext<Vec<u8>>: CKKSPlaintextVecHostCodec<f64>,
{
    let params = dense_params(&params);
    let m = params.n / 2;
    let base2k = params.base2k;
    let log_delta = params.prec.log_delta;

    let module = Module::<BE>::new(params.n as u64);
    let host_module = Module::<HostBytesBackend>::new(params.n as u64);
    let encoder = Encoder::<E>::new::<F>(m).unwrap();

    let (sk_raw, sk) = gen_sk_with_raw(&params, &module, &host_module, [0u8; 32]);
    let mut scratch = alloc_scratch(&params, &module);

    let dec_lt = module
        .ckks_new_dft_matrix(
            &host_module,
            &encoder,
            Base2K(base2k as u32),
            &plan(DENSE_LOG_SLOTS, DFTType::Decode, DFTOutputFormat::Standard, log_delta),
            &mut scratch.borrow(),
        )
        .unwrap();
    let dec_dft = module.ckks_prepare_dft_matrix(&dec_lt, &mut scratch.borrow());

    let order = module.cyclotomic_order();
    let mut atks = HashMap::new();
    for p in dec_dft.galois_elements(order) {
        atks.entry(p)
            .or_insert_with(|| gen_atk(&params, &module, p, &sk_raw, &mut scratch.borrow()));
    }

    // Input: slot-encode (re, im).
    let (re, im) = test_vector_1::<F>(m);
    let mut ct = ckks_encrypt(
        &params,
        &module,
        &host_module,
        &encoder,
        &sk,
        params.k,
        &re,
        &im,
        &mut scratch.borrow(),
    );

    module
        .ckks_slots_to_coeffs(&mut ct, &dec_dft, &atks, &mut scratch.borrow())
        .unwrap();

    // Reference: a plaintext whose coefficients hold bitrev(re) || bitrev(im) at
    // the output scale. Measure the error via GLWE noise and bound its log2.
    let want = coeff_layout(&re, &im, params.n);
    let mut pt_want = want_plaintext(&module, &ct);
    pt_want.encode_host_floats(&want).unwrap();
    let noise = module.glwe_noise(&ct, &pt_want, &sk, &mut scratch.borrow()).std().log2();
    let bound = noise_bound(log_delta);
    assert!(
        noise < bound,
        "slots_to_coeffs (Standard) noise log2={noise:.1} (bound {bound:.1})"
    );
}

/// **CoeffsToSlots**, `SplitRealAndImag` format: the real and imaginary parts come
/// back in two separate real-vector ciphertexts. Coefficient-encode the layout of
/// `(re, im)`, apply the split CoeffsToSlots, and check `ct_real` decodes to
/// `(re, 0)` and `ct_imag` to `(im, 0)`.
pub fn test_dft_coeffs_to_slots_split<BE, F, E>(
    params: CKKSTestParams,
    _module: &Module<BE>,
    _host_module: &Module<HostBytesBackend>,
) where
    BE: TestContextBackend,
    Module<BE>: TestContextModule<BE> + DFTOps<BE> + GLWENoise<BE>,
    Module<HostBytesBackend>: TestContextHostModule,
    F: TestScalar,
    E: NegacyclicFFT<F> + NegacyclicFFTNew<F>,
    for<'a> <BE as Backend>::BufRef<'a>: HostDataRef,
    for<'a> <BE as Backend>::BufMut<'a>: HostDataMut,
    CKKSPlaintext<Vec<u8>>: CKKSPlaintextVecHostCodec<f64>,
{
    let params = dense_params(&params);
    let m = params.n / 2;
    let base2k = params.base2k;
    let log_delta = params.prec.log_delta;

    let module = Module::<BE>::new(params.n as u64);
    let host_module = Module::<HostBytesBackend>::new(params.n as u64);
    let encoder = Encoder::<E>::new::<F>(m).unwrap();

    let (sk_raw, sk) = gen_sk_with_raw(&params, &module, &host_module, [0u8; 32]);
    let mut scratch = alloc_scratch(&params, &module);

    let enc_lt = module
        .ckks_new_dft_matrix(
            &host_module,
            &encoder,
            Base2K(base2k as u32),
            &plan(DENSE_LOG_SLOTS, DFTType::Encode, DFTOutputFormat::SplitRealAndImag, log_delta),
            &mut scratch.borrow(),
        )
        .unwrap();
    let enc_dft = module.ckks_prepare_dft_matrix(&enc_lt, &mut scratch.borrow());

    let order = module.cyclotomic_order();
    let mut atks = HashMap::new();
    for p in enc_dft.galois_elements(order) {
        atks.entry(p)
            .or_insert_with(|| gen_atk(&params, &module, p, &sk_raw, &mut scratch.borrow()));
    }
    let conj_key = gen_atk(&params, &module, -1, &sk_raw, &mut scratch.borrow());

    let (re, im) = test_vector_1::<F>(m);
    let coeffs = coeff_layout(&re, &im, params.n);
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
    module
        .ckks_coeffs_to_slots_split(
            &mut ct_real,
            &mut ct_imag,
            &ct_in,
            &enc_dft,
            &atks,
            &conj_key,
            &mut scratch.borrow(),
        )
        .unwrap();

    // Reference: ct_real holds slots (re, 0), ct_imag holds slots (im, 0). Measure
    // each via GLWE noise and bound the worst log2.
    let zero = vec![F::from_f64(0.0).unwrap(); m];
    let mut pt_real = want_plaintext(&module, &ct_real);
    encoder.encode_reim(&mut pt_real, &re, &zero).unwrap();
    let mut pt_imag = want_plaintext(&module, &ct_imag);
    encoder.encode_reim(&mut pt_imag, &im, &zero).unwrap();
    let noise_real = module.glwe_noise(&ct_real, &pt_real, &sk, &mut scratch.borrow()).std().log2();
    let noise_imag = module.glwe_noise(&ct_imag, &pt_imag, &sk, &mut scratch.borrow()).std().log2();
    let noise = noise_real.max(noise_imag);
    let bound = noise_bound(log_delta);
    assert!(
        noise < bound,
        "coeffs_to_slots (Split) noise log2={noise:.1} (bound {bound:.1})"
    );
}

/// **CoeffsToSlots**, sparse `RepackImagAsReal` format: the imaginary part is
/// repacked into the right half of a single ciphertext. Coefficient-encode the
/// sparse layout of a `slots = 4` vector `(re, im)`, apply the repack
/// CoeffsToSlots, and check the result holds `[re | im]` in the real part (imag
/// ≈ 0) at `2·slots` resolution.
pub fn test_dft_coeffs_to_slots_repack_sparse<BE, F, E>(
    params: CKKSTestParams,
    _module: &Module<BE>,
    _host_module: &Module<HostBytesBackend>,
) where
    BE: TestContextBackend,
    Module<BE>: TestContextModule<BE> + DFTOps<BE> + GLWENoise<BE>,
    Module<HostBytesBackend>: TestContextHostModule,
    F: TestScalar,
    E: NegacyclicFFT<F> + NegacyclicFFTNew<F>,
    for<'a> <BE as Backend>::BufRef<'a>: HostDataRef,
    for<'a> <BE as Backend>::BufMut<'a>: HostDataMut,
    CKKSPlaintext<Vec<u8>>: CKKSPlaintextVecHostCodec<f64>,
{
    let params = sparse_params(&params);
    let base2k = params.base2k;
    let log_delta = params.prec.log_delta;
    let log_slots = SPARSE_LOG_SLOTS;
    let slots = 1usize << log_slots;

    let module = Module::<BE>::new(params.n as u64);
    let host_module = Module::<HostBytesBackend>::new(params.n as u64);
    let encoder = Encoder::<E>::new::<F>(params.n / 2).unwrap();

    let (sk_raw, sk) = gen_sk_with_raw(&params, &module, &host_module, [0u8; 32]);
    let mut scratch = alloc_scratch(&params, &module);

    let enc_lt = module
        .ckks_new_dft_matrix(
            &host_module,
            &encoder,
            Base2K(base2k as u32),
            &plan(log_slots, DFTType::Encode, DFTOutputFormat::RepackImagAsReal, log_delta),
            &mut scratch.borrow(),
        )
        .unwrap();
    let enc_dft = module.ckks_prepare_dft_matrix(&enc_lt, &mut scratch.borrow());
    assert!(enc_dft.is_sparse(), "expected sparse repack path");

    let order = module.cyclotomic_order();
    let mut atks = HashMap::new();
    for p in enc_dft.galois_elements(order) {
        atks.entry(p)
            .or_insert_with(|| gen_atk(&params, &module, p, &sk_raw, &mut scratch.borrow()));
    }
    let conj_key = gen_atk(&params, &module, -1, &sk_raw, &mut scratch.borrow());

    // Sparse coefficient layout: bitrev(re) at gap, bitrev(im) at N/2 + gap.
    let (re_full, im_full) = test_vector_1::<F>(params.n / 2);
    let (re, im) = (&re_full[..slots], &im_full[..slots]);
    let brev = |j: usize| ((j as u32).reverse_bits() >> (u32::BITS - log_slots as u32)) as usize;
    let gap = params.n / (2 * slots);
    let mut coeffs = vec![F::from_f64(0.0).unwrap(); params.n];
    for j in 0..slots {
        coeffs[j * gap] = re[brev(j)];
        coeffs[params.n / 2 + j * gap] = im[brev(j)];
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
    module
        .ckks_coeffs_to_slots_repack(&mut ct_out, &ct_in, &enc_dft, &atks, &conj_key, &mut scratch.borrow())
        .unwrap();

    // Reference at 2·slots resolution: real part = [re | im], imag part = 0.
    // Sparse-encode that expected vector and measure the error via GLWE noise.
    let small = Encoder::<E>::new::<F>(2 * slots).unwrap();
    let mut want_re = vec![F::from_f64(0.0).unwrap(); 2 * slots];
    want_re[..slots].copy_from_slice(re);
    want_re[slots..].copy_from_slice(im);
    let want_im = vec![F::from_f64(0.0).unwrap(); 2 * slots];
    let mut pt_want = want_plaintext(&module, &ct_out);
    small.encode_reim_sparse(&mut pt_want, &want_re, &want_im).unwrap();
    let noise = module.glwe_noise(&ct_out, &pt_want, &sk, &mut scratch.borrow()).std().log2();
    let bound = noise_bound(log_delta);
    assert!(
        noise < bound,
        "coeffs_to_slots (Repack) noise log2={noise:.1} (bound {bound:.1})"
    );
}

/// **SlotsToCoeffs**, `SplitRealAndImag` format: the real and imaginary slot
/// vectors are supplied as two separate real-vector ciphertexts (the engine forms
/// `ct_real + i·ct_imag` then Decodes). Slot-encode `(re, 0)` and `(im, 0)`, apply
/// the split SlotsToCoeffs, and check the result holds the coefficient layout
/// `bitrev(re) || bitrev(im)` — the inverse of [`test_dft_coeffs_to_slots_split`].
pub fn test_dft_slots_to_coeffs_split<BE, F, E>(
    params: CKKSTestParams,
    _module: &Module<BE>,
    _host_module: &Module<HostBytesBackend>,
) where
    BE: TestContextBackend,
    Module<BE>: TestContextModule<BE> + DFTOps<BE> + GLWENoise<BE>,
    Module<HostBytesBackend>: TestContextHostModule,
    F: TestScalar,
    E: NegacyclicFFT<F> + NegacyclicFFTNew<F>,
    for<'a> <BE as Backend>::BufRef<'a>: HostDataRef,
    for<'a> <BE as Backend>::BufMut<'a>: HostDataMut,
    CKKSPlaintext<Vec<u8>>: CKKSPlaintextVecHostCodec<f64>,
{
    let params = dense_params(&params);
    let m = params.n / 2;
    let base2k = params.base2k;
    let log_delta = params.prec.log_delta;

    let module = Module::<BE>::new(params.n as u64);
    let host_module = Module::<HostBytesBackend>::new(params.n as u64);
    let encoder = Encoder::<E>::new::<F>(m).unwrap();

    let (sk_raw, sk) = gen_sk_with_raw(&params, &module, &host_module, [0u8; 32]);
    let mut scratch = alloc_scratch(&params, &module);

    let dec_lt = module
        .ckks_new_dft_matrix(
            &host_module,
            &encoder,
            Base2K(base2k as u32),
            &plan(DENSE_LOG_SLOTS, DFTType::Decode, DFTOutputFormat::SplitRealAndImag, log_delta),
            &mut scratch.borrow(),
        )
        .unwrap();
    let dec_dft = module.ckks_prepare_dft_matrix(&dec_lt, &mut scratch.borrow());

    let order = module.cyclotomic_order();
    let mut atks = HashMap::new();
    for p in dec_dft.galois_elements(order) {
        atks.entry(p)
            .or_insert_with(|| gen_atk(&params, &module, p, &sk_raw, &mut scratch.borrow()));
    }

    // Inputs: ct_real holds slots (re, 0), ct_imag holds slots (im, 0).
    let (re, im) = test_vector_1::<F>(m);
    let zero = vec![F::from_f64(0.0).unwrap(); m];
    let ct_real = ckks_encrypt(
        &params,
        &module,
        &host_module,
        &encoder,
        &sk,
        params.k,
        &re,
        &zero,
        &mut scratch.borrow(),
    );
    let ct_imag = ckks_encrypt(
        &params,
        &module,
        &host_module,
        &encoder,
        &sk,
        params.k,
        &im,
        &zero,
        &mut scratch.borrow(),
    );

    let mut op_out = alloc_ct(&params, &module, params.k);
    module
        .ckks_slots_to_coeffs_split(&mut op_out, &ct_real, &ct_imag, &dec_dft, &atks, &mut scratch.borrow())
        .unwrap();

    // Reference: coefficients bitrev(re) || bitrev(im) at the output scale.
    let want = coeff_layout(&re, &im, params.n);
    let mut pt_want = want_plaintext(&module, &op_out);
    pt_want.encode_host_floats(&want).unwrap();
    let noise = module.glwe_noise(&op_out, &pt_want, &sk, &mut scratch.borrow()).std().log2();
    let bound = noise_bound(log_delta);
    assert!(
        noise < bound,
        "slots_to_coeffs (Split) noise log2={noise:.1} (bound {bound:.1})"
    );
}

/// **SlotsToCoeffs**, sparse `RepackImagAsReal` format: a single ciphertext holding
/// `[re | im]` in its real part at `2·slots` resolution is Decoded back to the
/// sparse coefficient layout — the inverse of
/// [`test_dft_coeffs_to_slots_repack_sparse`]. Sparse-encode the repacked input,
/// apply the repack SlotsToCoeffs, and check the result holds the sparse layout.
pub fn test_dft_slots_to_coeffs_repack_sparse<BE, F, E>(
    params: CKKSTestParams,
    _module: &Module<BE>,
    _host_module: &Module<HostBytesBackend>,
) where
    BE: TestContextBackend,
    Module<BE>: TestContextModule<BE> + DFTOps<BE> + GLWENoise<BE>,
    Module<HostBytesBackend>: TestContextHostModule,
    F: TestScalar,
    E: NegacyclicFFT<F> + NegacyclicFFTNew<F>,
    for<'a> <BE as Backend>::BufRef<'a>: HostDataRef,
    for<'a> <BE as Backend>::BufMut<'a>: HostDataMut,
    CKKSPlaintext<Vec<u8>>: CKKSPlaintextVecHostCodec<f64>,
{
    let params = sparse_params(&params);
    let base2k = params.base2k;
    let log_delta = params.prec.log_delta;
    let log_slots = SPARSE_LOG_SLOTS;
    let slots = 1usize << log_slots;

    let module = Module::<BE>::new(params.n as u64);
    let host_module = Module::<HostBytesBackend>::new(params.n as u64);
    let encoder = Encoder::<E>::new::<F>(params.n / 2).unwrap();

    let (sk_raw, sk) = gen_sk_with_raw(&params, &module, &host_module, [0u8; 32]);
    let mut scratch = alloc_scratch(&params, &module);

    let dec_lt = module
        .ckks_new_dft_matrix(
            &host_module,
            &encoder,
            Base2K(base2k as u32),
            &plan(log_slots, DFTType::Decode, DFTOutputFormat::RepackImagAsReal, log_delta),
            &mut scratch.borrow(),
        )
        .unwrap();
    let dec_dft = module.ckks_prepare_dft_matrix(&dec_lt, &mut scratch.borrow());
    assert!(dec_dft.is_sparse(), "expected sparse repack path");

    let order = module.cyclotomic_order();
    let mut atks = HashMap::new();
    for p in dec_dft.galois_elements(order) {
        atks.entry(p)
            .or_insert_with(|| gen_atk(&params, &module, p, &sk_raw, &mut scratch.borrow()));
    }

    // Input: a single ciphertext holding [re | im] in the real part at 2·slots
    // resolution (imag 0), the repacked-slots form (log_sparsity = 2).
    let (re_full, im_full) = test_vector_1::<F>(params.n / 2);
    let (re, im) = (&re_full[..slots], &im_full[..slots]);
    let small = Encoder::<E>::new::<F>(2 * slots).unwrap();
    let mut want_re = vec![F::from_f64(0.0).unwrap(); 2 * slots];
    want_re[..slots].copy_from_slice(re);
    want_re[slots..].copy_from_slice(im);
    let want_im = vec![F::from_f64(0.0).unwrap(); 2 * slots];
    let mut host_pt = host_module.ckks_pt_vec_alloc(
        Base2K(base2k as u32),
        CKKSMeta {
            log_sparsity: 2,
            log_delta,
            log_budget: 10,
        },
    );
    small.encode_reim_sparse(&mut host_pt, &want_re, &want_im).unwrap();
    let mut ct_in = ckks_encrypt_pt(&params, &module, &sk, params.k, &host_pt, &mut scratch.borrow());
    ct_in.set_log_sparsity(2);

    let mut op_out = alloc_ct(&params, &module, params.k);
    module
        .ckks_slots_to_coeffs_repack(&mut op_out, &ct_in, &dec_dft, &atks, &mut scratch.borrow())
        .unwrap();
    assert_eq!(op_out.log_sparsity(), 3, "repack-decode halves the live slot count");

    // Reference: the sparse coefficient layout bitrev(re) at gap, bitrev(im) at
    // N/2 + gap (the input of the forward repack), at the output scale.
    let brev = |j: usize| ((j as u32).reverse_bits() >> (u32::BITS - log_slots as u32)) as usize;
    let gap = params.n / (2 * slots);
    let mut want = vec![F::from_f64(0.0).unwrap(); params.n];
    for j in 0..slots {
        want[j * gap] = re[brev(j)];
        want[params.n / 2 + j * gap] = im[brev(j)];
    }
    let mut pt_want = want_plaintext(&module, &op_out);
    pt_want.encode_host_floats(&want).unwrap();
    let noise = module.glwe_noise(&op_out, &pt_want, &sk, &mut scratch.borrow()).std().log2();
    let bound = noise_bound(log_delta);
    assert!(
        noise < bound,
        "slots_to_coeffs (Repack) noise log2={noise:.1} (bound {bound:.1})"
    );
}

/// The plan-level [`DFTPlan`] helpers — [`DFTPlan::galois_elements`],
/// [`DFTPlan::diagonal_indexes`], [`DFTPlan::num_diagonals`] — are derived
/// structurally from the factorization schedule, without generating any diagonal.
/// This pins them to the compiled [`DFTMatrix`]: the Galois elements must match
/// exactly (keys are addressed by Galois element), for both directions and the
/// dense and sparse-repack paths.
pub fn test_dft_plan_helpers_match_compiled<BE, F, E>(
    params: CKKSTestParams,
    _module: &Module<BE>,
    _host_module: &Module<HostBytesBackend>,
) where
    BE: TestContextBackend,
    Module<BE>: TestContextModule<BE> + DFTOps<BE>,
    Module<HostBytesBackend>: TestContextHostModule,
    F: TestScalar,
    E: NegacyclicFFT<F> + NegacyclicFFTNew<F>,
    for<'a> <BE as Backend>::BufRef<'a>: HostDataRef,
    for<'a> <BE as Backend>::BufMut<'a>: HostDataMut,
    CKKSPlaintext<Vec<u8>>: CKKSPlaintextVecHostCodec<f64>,
{
    // ---- dense (full slot count): Split, Encode + Decode ----
    {
        let p = dense_params(&params);
        let module = Module::<BE>::new(p.n as u64);
        let host_module = Module::<HostBytesBackend>::new(p.n as u64);
        let encoder = Encoder::<E>::new::<F>(p.n / 2).unwrap();
        let mut scratch = alloc_scratch(&p, &module);
        let log_n = p.n.ilog2() as usize;
        let order = module.cyclotomic_order();
        let base2k = Base2K(p.base2k as u32);
        let ld = p.prec.log_delta;

        let pe = plan(DENSE_LOG_SLOTS, DFTType::Encode, DFTOutputFormat::SplitRealAndImag, ld);
        let me: DFTMatrix<BE, Encode, Split> = module
            .ckks_new_dft_matrix(&host_module, &encoder, base2k, &pe, &mut scratch.borrow())
            .unwrap();
        assert_eq!(pe.galois_elements(log_n, order), me.galois_elements(order), "dense encode galois");
        assert!(!pe.is_sparse_repack(log_n));
        assert_eq!(pe.num_diagonals(log_n).len(), pe.num_factors());

        let pd = plan(DENSE_LOG_SLOTS, DFTType::Decode, DFTOutputFormat::SplitRealAndImag, ld);
        let md: DFTMatrix<BE, Decode, Split> = module
            .ckks_new_dft_matrix(&host_module, &encoder, base2k, &pd, &mut scratch.borrow())
            .unwrap();
        assert_eq!(pd.galois_elements(log_n, order), md.galois_elements(order), "dense decode galois");
    }

    // ---- sparse RepackImagAsReal: Encode + Decode ----
    {
        let p = sparse_params(&params);
        let module = Module::<BE>::new(p.n as u64);
        let host_module = Module::<HostBytesBackend>::new(p.n as u64);
        let encoder = Encoder::<E>::new::<F>(1 << SPARSE_LOG_SLOTS).unwrap();
        let mut scratch = alloc_scratch(&p, &module);
        let log_n = p.n.ilog2() as usize;
        let order = module.cyclotomic_order();
        let base2k = Base2K(p.base2k as u32);
        let ld = p.prec.log_delta;

        let pe = plan(SPARSE_LOG_SLOTS, DFTType::Encode, DFTOutputFormat::RepackImagAsReal, ld);
        assert!(pe.is_sparse_repack(log_n), "expected the sparse repack path");
        let me: DFTMatrix<BE, Encode, Repack> = module
            .ckks_new_dft_matrix(&host_module, &encoder, base2k, &pe, &mut scratch.borrow())
            .unwrap();
        assert_eq!(pe.galois_elements(log_n, order), me.galois_elements(order), "sparse encode galois");

        let pd = plan(SPARSE_LOG_SLOTS, DFTType::Decode, DFTOutputFormat::RepackImagAsReal, ld);
        let md: DFTMatrix<BE, Decode, Repack> = module
            .ckks_new_dft_matrix(&host_module, &encoder, base2k, &pd, &mut scratch.borrow())
            .unwrap();
        assert_eq!(pd.galois_elements(log_n, order), md.galois_elements(order), "sparse decode galois");
    }
}
