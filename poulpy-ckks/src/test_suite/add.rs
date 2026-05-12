//! Addition tests: ct+ct, ct+pt, ct+cst (out-of-place and in-place).
//!
//! # Test inventory
//!
//! ## Operations-layer ct+ct (`GLWE<_, CKKS>::add`)
//!
//! | Function | Path exercised |
//! |----------|----------------|
//! | [`test_add_ct_aligned`] | `a.log_budget() == b.log_budget()`, `offset == 0` → `glwe_add` fast path |
//! | [`test_add_ct_delta_a_lt_b`] | `a.log_budget() < b.log_budget()` → b shifted to align with a |
//! | [`test_add_ct_delta_a_gt_b`] | `a.log_budget() > b.log_budget()` → a shifted to align with b |
//! | [`test_add_ct_smaller_output`] | `offset > 0` (output one limb narrower than inputs) |
//!
//! ## Operations-layer ct+ct (`GLWE<_, CKKS>::add_assign`)
//!
//! | Function | Path exercised |
//! |----------|----------------|
//! | [`test_add_ct_assign_aligned`] | `self.log_budget() == a.log_budget()` |
//! | [`test_add_ct_assign_self_lt`] | `self.log_budget() < a.log_budget()` → a shifted to align with self |
//! | [`test_add_ct_assign_self_gt`] | `self.log_budget() > a.log_budget()` → self shifted to align with a |
//!
//! ## Operations-layer ct + ZNX plaintext (`GLWE<_, CKKS>::add_pt_vec[_assign]`)
//!
//! | Function | Path exercised |
//! |----------|----------------|
//! | [`test_add_pt_vec_assign`] | in-place, `offset == 0` |
//! | [`test_add_pt_vec_into_aligned`] | out-of-place, `offset == 0` |
//! | [`test_add_pt_vec_into_delta_log_delta`] | out-of-place, plaintext encoded at lower `log_delta` |
//! | [`test_add_pt_vec_into_smaller_output`] | out-of-place, `offset > 0` (output one limb narrower) |
//!
//! ## Operations-layer ct + packed plaintext constants (`GLWE<_, CKKS>::add_pt_const[_assign]`)
//!
//! | Function | Path exercised |
//! |----------|----------------|
//! | [`test_add_const_into_aligned`] | out-of-place, aligned packed coeffs |
//! | [`test_add_const_assign`] | in-place, aligned packed coeffs |
//! | [`test_add_const_into_delta_log_delta`] | out-of-place, constant encoded at lower `log_delta` |
//! | [`test_add_const_into_smaller_output`] | out-of-place, smaller output with packed-cst precision |
//! | [`test_add_const_into_real_only`] | out-of-place, real coefficient only |
use crate::{CKKSCompositionError, CKKSInfos, layouts::CKKSModuleAlloc, leveled::api::CKKSAddOps};

use super::helpers::{
    ADD_SUB_CONST, PT_PREC, TestContextBackend, TestContextModule, TestScalar, TestVector, add_sub_const_pt, alloc_ct,
    alloc_scratch, assert_binary_output_meta, assert_ckks_error, assert_ct_meta, assert_decrypt_precision,
    assert_decrypt_precision_at_log_delta, assert_unary_output_meta, ckks_encrypt, ckks_encrypt_with_prec, ckks_pt_cst,
    encode_and_upload_pt, gen_sk, precision_at, quantize, quantized_const, quantized_vector, test_vector_1, test_vector_2,
    want_add, want_add_const,
};
use poulpy_core::layouts::Base2K;
use poulpy_hal::api::ScratchOwnedBorrow;
use poulpy_hal::{
    api::{NegacyclicFFT, NegacyclicFFTNew},
    layouts::{HostBytesBackend, Module},
};

use crate::{encoding::reim::Encoder, test_suite::CKKSTestParams};

const DELTA_LOG_DELTA: usize = 12;

// ─── ct+ct out-of-place (GLWE<_, CKKS>::add) ────────────────────────────────

/// ct+ct out-of-place, aligned (same log_budget(), offset == 0 → glwe_add fast path).
pub fn test_add_ct_aligned<BE, F, E>(params: CKKSTestParams, module: &Module<BE>, host_module: &Module<HostBytesBackend>)
where
    BE: TestContextBackend,
    Module<BE>: TestContextModule<BE>,
    F: TestScalar,
    E: NegacyclicFFT<F> + NegacyclicFFTNew<F>,
{
    let m = params.n / 2;
    let encoder = Encoder::<E>::new(m).unwrap();
    let (re1, im1) = test_vector_1::<F>(m);
    let (re2, im2) = test_vector_2::<F>(m);
    let sk = gen_sk(&params, module, host_module, [0u8; 32]);
    let mut scratch = alloc_scratch(&params, module);

    let ct1 = ckks_encrypt(
        &params,
        module,
        host_module,
        &encoder,
        &sk,
        params.k,
        &re1,
        &im1,
        &mut scratch.borrow(),
    );
    let ct2 = ckks_encrypt(
        &params,
        module,
        host_module,
        &encoder,
        &sk,
        params.k,
        &re2,
        &im2,
        &mut scratch.borrow(),
    );
    let (want_re, want_im) = want_add(&re1, &im1, &re2, &im2);
    let mut ct_res = alloc_ct(&params, module, params.k);
    module.ckks_add_into(&mut ct_res, &ct1, &ct2, &mut scratch.borrow()).unwrap();
    assert_binary_output_meta("add_ct_aligned", &ct_res, &ct1, &ct2);
    assert_decrypt_precision(
        "add_ct_aligned",
        &params,
        module,
        &encoder,
        &ct_res,
        &sk,
        &want_re,
        &want_im,
        &mut scratch.borrow(),
    );
}

/// ct+ct out-of-place, a.log_budget() < b.log_budget() (b is shifted to align with a).
pub fn test_add_ct_delta_a_lt_b<BE, F, E>(params: CKKSTestParams, module: &Module<BE>, host_module: &Module<HostBytesBackend>)
where
    BE: TestContextBackend,
    Module<BE>: TestContextModule<BE>,
    F: TestScalar,
    E: NegacyclicFFT<F> + NegacyclicFFTNew<F>,
{
    let m = params.n / 2;
    let encoder = Encoder::<E>::new(m).unwrap();
    let (re1, im1) = test_vector_1::<F>(m);
    let (re2, im2) = test_vector_2::<F>(m);
    let sk = gen_sk(&params, module, host_module, [0u8; 32]);
    let mut scratch = alloc_scratch(&params, module);

    let ct1 = ckks_encrypt(
        &params,
        module,
        host_module,
        &encoder,
        &sk,
        params.k - params.base2k + 1,
        &re1,
        &im1,
        &mut scratch.borrow(),
    );
    let ct2 = ckks_encrypt(
        &params,
        module,
        host_module,
        &encoder,
        &sk,
        params.k,
        &re2,
        &im2,
        &mut scratch.borrow(),
    );
    let (want_re, want_im) = want_add(&re1, &im1, &re2, &im2);
    let mut ct_res = alloc_ct(&params, module, params.k);
    module.ckks_add_into(&mut ct_res, &ct1, &ct2, &mut scratch.borrow()).unwrap();
    assert_binary_output_meta("add_ct a_lt_b", &ct_res, &ct1, &ct2);
    assert_decrypt_precision(
        "add_ct a_lt_b",
        &params,
        module,
        &encoder,
        &ct_res,
        &sk,
        &want_re,
        &want_im,
        &mut scratch.borrow(),
    );
}

/// ct+ct out-of-place, a.log_budget() > b.log_budget() (a is shifted to align with b).
pub fn test_add_ct_delta_a_gt_b<BE, F, E>(params: CKKSTestParams, module: &Module<BE>, host_module: &Module<HostBytesBackend>)
where
    BE: TestContextBackend,
    Module<BE>: TestContextModule<BE>,
    F: TestScalar,
    E: NegacyclicFFT<F> + NegacyclicFFTNew<F>,
{
    let m = params.n / 2;
    let encoder = Encoder::<E>::new(m).unwrap();
    let (re1, im1) = test_vector_1::<F>(m);
    let (re2, im2) = test_vector_2::<F>(m);
    let sk = gen_sk(&params, module, host_module, [0u8; 32]);
    let mut scratch = alloc_scratch(&params, module);

    let ct1 = ckks_encrypt(
        &params,
        module,
        host_module,
        &encoder,
        &sk,
        params.k,
        &re1,
        &im1,
        &mut scratch.borrow(),
    );
    let ct2 = ckks_encrypt(
        &params,
        module,
        host_module,
        &encoder,
        &sk,
        params.k - params.base2k + 1,
        &re2,
        &im2,
        &mut scratch.borrow(),
    );
    let (want_re, want_im) = want_add(&re1, &im1, &re2, &im2);
    let mut ct_res = alloc_ct(&params, module, params.k);
    module.ckks_add_into(&mut ct_res, &ct1, &ct2, &mut scratch.borrow()).unwrap();
    assert_binary_output_meta("add_ct a_gt_b", &ct_res, &ct1, &ct2);
    assert_decrypt_precision(
        "add_ct a_gt_b",
        &params,
        module,
        &encoder,
        &ct_res,
        &sk,
        &want_re,
        &want_im,
        &mut scratch.borrow(),
    );
}

/// ct+ct out-of-place with aligned homomorphic capacity but different log_delta.
pub fn test_add_ct_delta_log_delta<BE, F, E>(params: CKKSTestParams, module: &Module<BE>, host_module: &Module<HostBytesBackend>)
where
    BE: TestContextBackend,
    Module<BE>: TestContextModule<BE>,
    F: TestScalar,
    E: NegacyclicFFT<F> + NegacyclicFFTNew<F>,
{
    let m = params.n / 2;
    let encoder = Encoder::<E>::new(m).unwrap();
    let sk = gen_sk(&params, module, host_module, [0u8; 32]);
    let mut scratch = alloc_scratch(&params, module);

    let low_log_delta = params.prec.log_delta - DELTA_LOG_DELTA;
    let low_prec = precision_at(&params, low_log_delta);
    let (a_re, a_im) = quantized_vector(host_module, &encoder, &params, TestVector::First, params.prec.log_delta);
    let (b_re, b_im) = quantized_vector(host_module, &encoder, &params, TestVector::Second, low_log_delta);
    let ct1 = ckks_encrypt(
        &params,
        module,
        host_module,
        &encoder,
        &sk,
        params.k,
        &a_re,
        &a_im,
        &mut scratch.borrow(),
    );
    let ct2 = ckks_encrypt_with_prec(
        &params,
        module,
        host_module,
        &encoder,
        &sk,
        params.k - DELTA_LOG_DELTA,
        &b_re,
        &b_im,
        low_prec,
        &mut scratch.borrow(),
    );
    let (want_re, want_im) = want_add(&a_re, &a_im, &b_re, &b_im);
    let mut ct_res = alloc_ct(&params, module, params.k);
    module.ckks_add_into(&mut ct_res, &ct1, &ct2, &mut scratch.borrow()).unwrap();
    assert_binary_output_meta("add_ct delta_log_delta", &ct_res, &ct1, &ct2);
    assert_decrypt_precision_at_log_delta(
        "add_ct delta_log_delta",
        &params,
        module,
        &encoder,
        &ct_res,
        &sk,
        &want_re,
        &want_im,
        low_log_delta,
        &mut scratch.borrow(),
    );
}

/// ct+ct out-of-place, output buffer has smaller max_k than inputs (offset > 0).
pub fn test_add_ct_aligned_smaller_output<BE, F, E>(
    params: CKKSTestParams,
    module: &Module<BE>,
    host_module: &Module<HostBytesBackend>,
) where
    BE: TestContextBackend,
    Module<BE>: TestContextModule<BE>,
    F: TestScalar,
    E: NegacyclicFFT<F> + NegacyclicFFTNew<F>,
{
    let m = params.n / 2;
    let encoder = Encoder::<E>::new(m).unwrap();
    let (re1, im1) = test_vector_1::<F>(m);
    let (re2, im2) = test_vector_2::<F>(m);
    let sk = gen_sk(&params, module, host_module, [0u8; 32]);
    let mut scratch = alloc_scratch(&params, module);

    let ct1 = ckks_encrypt(
        &params,
        module,
        host_module,
        &encoder,
        &sk,
        params.k,
        &re1,
        &im1,
        &mut scratch.borrow(),
    );
    let ct2 = ckks_encrypt(
        &params,
        module,
        host_module,
        &encoder,
        &sk,
        params.k,
        &re2,
        &im2,
        &mut scratch.borrow(),
    );
    let (want_re, want_im) = want_add(&re1, &im1, &re2, &im2);
    let mut ct_res = alloc_ct(&params, module, params.k - params.base2k - 1);
    module.ckks_add_into(&mut ct_res, &ct1, &ct2, &mut scratch.borrow()).unwrap();
    assert_binary_output_meta("add_ct smaller_output", &ct_res, &ct1, &ct2);
    assert_decrypt_precision(
        "add_ct smaller_output",
        &params,
        module,
        &encoder,
        &ct_res,
        &sk,
        &want_re,
        &want_im,
        &mut scratch.borrow(),
    );
}

// ─── ct+ct in-place (GLWE<_, CKKS>::add_assign) ────────────────────────────

/// ct+ct in-place, aligned (same log_budget()).
pub fn test_add_ct_assign_aligned<BE, F, E>(params: CKKSTestParams, module: &Module<BE>, host_module: &Module<HostBytesBackend>)
where
    BE: TestContextBackend,
    Module<BE>: TestContextModule<BE>,
    F: TestScalar,
    E: NegacyclicFFT<F> + NegacyclicFFTNew<F>,
{
    let m = params.n / 2;
    let encoder = Encoder::<E>::new(m).unwrap();
    let (re1, im1) = test_vector_1::<F>(m);
    let (re2, im2) = test_vector_2::<F>(m);
    let sk = gen_sk(&params, module, host_module, [0u8; 32]);
    let mut scratch = alloc_scratch(&params, module);

    let mut ct1 = ckks_encrypt(
        &params,
        module,
        host_module,
        &encoder,
        &sk,
        params.k,
        &re1,
        &im1,
        &mut scratch.borrow(),
    );
    let ct2 = ckks_encrypt(
        &params,
        module,
        host_module,
        &encoder,
        &sk,
        params.k,
        &re2,
        &im2,
        &mut scratch.borrow(),
    );
    let (want_re, want_im) = want_add(&re1, &im1, &re2, &im2);
    let expected_log_budget = ct1.log_budget().min(ct2.log_budget());
    let expected_log_delta = ct1.log_delta().max(ct2.log_delta());
    module.ckks_add_assign(&mut ct1, &ct2, &mut scratch.borrow()).unwrap();
    assert_ct_meta("add_ct_assign_aligned", &ct1, expected_log_delta, expected_log_budget);
    assert_decrypt_precision(
        "add_ct_assign_aligned",
        &params,
        module,
        &encoder,
        &ct1,
        &sk,
        &want_re,
        &want_im,
        &mut scratch.borrow(),
    );
}

/// ct+ct in-place, self.log_budget() < a.log_budget() (a is shifted down to align with self).
pub fn test_add_ct_assign_self_lt<BE, F, E>(params: CKKSTestParams, module: &Module<BE>, host_module: &Module<HostBytesBackend>)
where
    BE: TestContextBackend,
    Module<BE>: TestContextModule<BE>,
    F: TestScalar,
    E: NegacyclicFFT<F> + NegacyclicFFTNew<F>,
{
    let m = params.n / 2;
    let encoder = Encoder::<E>::new(m).unwrap();
    let (re1, im1) = test_vector_1::<F>(m);
    let (re2, im2) = test_vector_2::<F>(m);
    let sk = gen_sk(&params, module, host_module, [0u8; 32]);
    let mut scratch = alloc_scratch(&params, module);

    let mut ct_self = ckks_encrypt(
        &params,
        module,
        host_module,
        &encoder,
        &sk,
        params.k - params.base2k - 1,
        &re1,
        &im1,
        &mut scratch.borrow(),
    );
    let ct_other = ckks_encrypt(
        &params,
        module,
        host_module,
        &encoder,
        &sk,
        params.k,
        &re2,
        &im2,
        &mut scratch.borrow(),
    );
    let (want_re, want_im) = want_add(&re1, &im1, &re2, &im2);
    let expected_log_budget = ct_self.log_budget().min(ct_other.log_budget());
    let expected_log_delta = ct_self.log_delta().max(ct_other.log_delta());
    module
        .ckks_add_assign(&mut ct_self, &ct_other, &mut scratch.borrow())
        .unwrap();
    assert_ct_meta("add_ct_assign self_lt", &ct_self, expected_log_delta, expected_log_budget);
    assert_decrypt_precision(
        "add_ct_assign self_lt",
        &params,
        module,
        &encoder,
        &ct_self,
        &sk,
        &want_re,
        &want_im,
        &mut scratch.borrow(),
    );
}

/// ct+ct in-place, self.log_budget() > a.log_budget() (self is shifted down to align with a).
pub fn test_add_ct_assign_self_gt<BE, F, E>(params: CKKSTestParams, module: &Module<BE>, host_module: &Module<HostBytesBackend>)
where
    BE: TestContextBackend,
    Module<BE>: TestContextModule<BE>,
    F: TestScalar,
    E: NegacyclicFFT<F> + NegacyclicFFTNew<F>,
{
    let m = params.n / 2;
    let encoder = Encoder::<E>::new(m).unwrap();
    let (re1, im1) = test_vector_1::<F>(m);
    let (re2, im2) = test_vector_2::<F>(m);
    let sk = gen_sk(&params, module, host_module, [0u8; 32]);
    let mut scratch = alloc_scratch(&params, module);

    let mut ct_self = ckks_encrypt(
        &params,
        module,
        host_module,
        &encoder,
        &sk,
        params.k,
        &re1,
        &im1,
        &mut scratch.borrow(),
    );
    let ct_other = ckks_encrypt(
        &params,
        module,
        host_module,
        &encoder,
        &sk,
        params.k - params.base2k - 1,
        &re2,
        &im2,
        &mut scratch.borrow(),
    );
    let (want_re, want_im) = want_add(&re1, &im1, &re2, &im2);
    let expected_log_budget = ct_self.log_budget().min(ct_other.log_budget());
    let expected_log_delta = ct_self.log_delta().max(ct_other.log_delta());
    module
        .ckks_add_assign(&mut ct_self, &ct_other, &mut scratch.borrow())
        .unwrap();
    assert_ct_meta("add_ct_assign self_gt", &ct_self, expected_log_delta, expected_log_budget);
    assert_decrypt_precision(
        "add_ct_assign self_gt",
        &params,
        module,
        &encoder,
        &ct_self,
        &sk,
        &want_re,
        &want_im,
        &mut scratch.borrow(),
    );
}

// ─── ct + compact ZNX plaintext (GLWE<_, CKKS>::add_pt_vec[_assign]) ────────

/// ct + ZNX plaintext, in-place.
pub fn test_add_pt_vec_assign<BE, F, E>(params: CKKSTestParams, module: &Module<BE>, host_module: &Module<HostBytesBackend>)
where
    BE: TestContextBackend,
    Module<BE>: TestContextModule<BE>,
    F: TestScalar,
    E: NegacyclicFFT<F> + NegacyclicFFTNew<F>,
{
    let m = params.n / 2;
    let encoder = Encoder::<E>::new(m).unwrap();
    let (re1, im1) = test_vector_1::<F>(m);
    let (re2, im2) = test_vector_2::<F>(m);
    let sk = gen_sk(&params, module, host_module, [0u8; 32]);
    let mut scratch = alloc_scratch(&params, module);

    let mut ct = ckks_encrypt(
        &params,
        module,
        host_module,
        &encoder,
        &sk,
        params.k,
        &re1,
        &im1,
        &mut scratch.borrow(),
    );
    let pt = encode_and_upload_pt(host_module, module, &encoder, params.base2k.into(), params.prec, &re2, &im2);
    let (want_re, want_im) = want_add(
        &re1,
        &im1,
        &quantize(&re2, params.prec.log_delta),
        &quantize(&im2, params.prec.log_delta),
    );
    let expected_log_delta = ct.log_delta();
    let expected_log_budget = ct.log_budget();
    module.ckks_add_pt_vec_assign(&mut ct, &pt, &mut scratch.borrow()).unwrap();
    assert_ct_meta("add_pt_vec_assign", &ct, expected_log_delta, expected_log_budget);
    assert_decrypt_precision(
        "add_pt_vec_assign",
        &params,
        module,
        &encoder,
        &ct,
        &sk,
        &want_re,
        &want_im,
        &mut scratch.borrow(),
    );
}

/// ct + ZNX plaintext, out-of-place.
pub fn test_add_pt_vec_into_aligned<BE, F, E>(params: CKKSTestParams, module: &Module<BE>, host_module: &Module<HostBytesBackend>)
where
    BE: TestContextBackend,
    Module<BE>: TestContextModule<BE>,
    F: TestScalar,
    E: NegacyclicFFT<F> + NegacyclicFFTNew<F>,
{
    let m = params.n / 2;
    let encoder = Encoder::<E>::new(m).unwrap();
    let (re1, im1) = test_vector_1::<F>(m);
    let (re2, im2) = test_vector_2::<F>(m);
    let sk = gen_sk(&params, module, host_module, [0u8; 32]);
    let mut scratch = alloc_scratch(&params, module);

    let ct1 = ckks_encrypt(
        &params,
        module,
        host_module,
        &encoder,
        &sk,
        params.k,
        &re1,
        &im1,
        &mut scratch.borrow(),
    );
    let pt = encode_and_upload_pt(host_module, module, &encoder, params.base2k.into(), params.prec, &re2, &im2);
    let (want_re, want_im) = want_add(
        &re1,
        &im1,
        &quantize(&re2, params.prec.log_delta),
        &quantize(&im2, params.prec.log_delta),
    );
    let mut ct_res = alloc_ct(&params, module, params.k);
    module
        .ckks_add_pt_vec_into(&mut ct_res, &ct1, &pt, &mut scratch.borrow())
        .unwrap();
    assert_unary_output_meta("add_pt_vec", &ct_res, &ct1);
    assert_decrypt_precision(
        "add_pt_vec",
        &params,
        module,
        &encoder,
        &ct_res,
        &sk,
        &want_re,
        &want_im,
        &mut scratch.borrow(),
    );
}

/// ct + ZNX plaintext, out-of-place, plaintext encoded at lower decimal precision.
pub fn test_add_pt_vec_into_delta_log_delta<BE, F, E>(
    params: CKKSTestParams,
    module: &Module<BE>,
    host_module: &Module<HostBytesBackend>,
) where
    BE: TestContextBackend,
    Module<BE>: TestContextModule<BE>,
    F: TestScalar,
    E: NegacyclicFFT<F> + NegacyclicFFTNew<F>,
{
    let m = params.n / 2;
    let encoder = Encoder::<E>::new(m).unwrap();
    let (re2, im2) = test_vector_2::<F>(m);
    let sk = gen_sk(&params, module, host_module, [0u8; 32]);
    let mut scratch = alloc_scratch(&params, module);

    let (a_re, a_im) = quantized_vector(host_module, &encoder, &params, TestVector::First, params.prec.log_delta);
    let ct1 = ckks_encrypt(
        &params,
        module,
        host_module,
        &encoder,
        &sk,
        params.k,
        &a_re,
        &a_im,
        &mut scratch.borrow(),
    );
    let pt = encode_and_upload_pt(host_module, module, &encoder, params.base2k.into(), PT_PREC, &re2, &im2);
    let (want_re, want_im) = want_add(
        &a_re,
        &a_im,
        &quantize(&re2, PT_PREC.log_delta),
        &quantize(&im2, PT_PREC.log_delta),
    );
    let mut ct_res = alloc_ct(&params, module, params.k);
    module
        .ckks_add_pt_vec_into(&mut ct_res, &ct1, &pt, &mut scratch.borrow())
        .unwrap();
    assert_unary_output_meta("add_pt_vec delta_log_delta", &ct_res, &ct1);
    assert_decrypt_precision_at_log_delta(
        "add_pt_vec delta_log_delta",
        &params,
        module,
        &encoder,
        &ct_res,
        &sk,
        &want_re,
        &want_im,
        PT_PREC.log_delta,
        &mut scratch.borrow(),
    );
}

/// ct + complex constant, out-of-place.
pub fn test_add_const_into_aligned<BE, F, E>(params: CKKSTestParams, module: &Module<BE>, host_module: &Module<HostBytesBackend>)
where
    BE: TestContextBackend,
    Module<BE>: TestContextModule<BE>,
    F: TestScalar,
    E: NegacyclicFFT<F> + NegacyclicFFTNew<F>,
{
    let m = params.n / 2;
    let encoder = Encoder::<E>::new(m).unwrap();
    let (re1, im1) = test_vector_1::<F>(m);
    let sk = gen_sk(&params, module, host_module, [0u8; 32]);
    let mut scratch = alloc_scratch(&params, module);

    let ct = ckks_encrypt(
        &params,
        module,
        host_module,
        &encoder,
        &sk,
        params.k,
        &re1,
        &im1,
        &mut scratch.borrow(),
    );
    let (const_re, const_im) = quantized_const::<F>(ADD_SUB_CONST.0, ADD_SUB_CONST.1, PT_PREC.log_delta);
    let (want_re, want_im) = want_add_const(&re1, &im1, const_re, const_im);
    let mut ct_res = alloc_ct(&params, module, params.k);
    let cst = add_sub_const_pt::<BE, F>(host_module, module, params.base2k.into());
    module
        .ckks_add_pt_const_into(&mut ct_res, &ct, 0, &cst, 0, &mut scratch.borrow())
        .unwrap();
    module
        .ckks_add_pt_const_assign(&mut ct_res, m, &cst, 1, &mut scratch.borrow())
        .unwrap();
    assert_unary_output_meta("add_const_into_aligned", &ct_res, &ct);
    assert_decrypt_precision(
        "add_const_into_aligned",
        &params,
        module,
        &encoder,
        &ct_res,
        &sk,
        &want_re,
        &want_im,
        &mut scratch.borrow(),
    );
}

/// ct + complex constant, in-place.
pub fn test_add_const_assign<BE, F, E>(params: CKKSTestParams, module: &Module<BE>, host_module: &Module<HostBytesBackend>)
where
    BE: TestContextBackend,
    Module<BE>: TestContextModule<BE>,
    F: TestScalar,
    E: NegacyclicFFT<F> + NegacyclicFFTNew<F>,
{
    let m = params.n / 2;
    let encoder = Encoder::<E>::new(m).unwrap();
    let (re1, im1) = test_vector_1::<F>(m);
    let sk = gen_sk(&params, module, host_module, [0u8; 32]);
    let mut scratch = alloc_scratch(&params, module);

    let mut ct = ckks_encrypt(
        &params,
        module,
        host_module,
        &encoder,
        &sk,
        params.k,
        &re1,
        &im1,
        &mut scratch.borrow(),
    );
    let (const_re, const_im) = quantized_const::<F>(ADD_SUB_CONST.0, ADD_SUB_CONST.1, PT_PREC.log_delta);
    let (want_re, want_im) = want_add_const(&re1, &im1, const_re, const_im);
    let expected_log_delta = ct.log_delta();
    let expected_log_budget = ct.log_budget();
    let cst = add_sub_const_pt::<BE, F>(host_module, module, params.base2k.into());
    module
        .ckks_add_pt_const_assign(&mut ct, 0, &cst, 0, &mut scratch.borrow())
        .unwrap();
    module
        .ckks_add_pt_const_assign(&mut ct, m, &cst, 1, &mut scratch.borrow())
        .unwrap();
    assert_ct_meta("add_const_assign", &ct, expected_log_delta, expected_log_budget);
    assert_decrypt_precision(
        "add_const_assign",
        &params,
        module,
        &encoder,
        &ct,
        &sk,
        &want_re,
        &want_im,
        &mut scratch.borrow(),
    );
}

/// ct + 1, in-place.
pub fn test_add_one_assign<BE, F, E>(params: CKKSTestParams, module: &Module<BE>, host_module: &Module<HostBytesBackend>)
where
    BE: TestContextBackend,
    Module<BE>: TestContextModule<BE>,
    F: TestScalar,
    E: NegacyclicFFT<F> + NegacyclicFFTNew<F>,
{
    let m = params.n / 2;
    let encoder = Encoder::<E>::new(m).unwrap();
    let (re1, im1) = test_vector_1::<F>(m);
    let sk = gen_sk(&params, module, host_module, [0u8; 32]);
    let mut scratch = alloc_scratch(&params, module);

    let mut ct = ckks_encrypt(
        &params,
        module,
        host_module,
        &encoder,
        &sk,
        params.k,
        &re1,
        &im1,
        &mut scratch.borrow(),
    );
    let want_re: Vec<F> = re1.iter().map(|x| *x + F::one()).collect();
    let want_im = im1.clone();
    let expected_log_delta = ct.log_delta();
    let expected_log_budget = ct.log_budget();
    module.ckks_add_one_assign(&mut ct, &mut scratch.borrow()).unwrap();
    assert_ct_meta("add_one_assign", &ct, expected_log_delta, expected_log_budget);
    assert_decrypt_precision(
        "add_one_assign",
        &params,
        module,
        &encoder,
        &ct,
        &sk,
        &want_re,
        &want_im,
        &mut scratch.borrow(),
    );
}

/// ct + complex constant, out-of-place, constant encoded at lower decimal precision.
pub fn test_add_const_into_delta_log_delta<BE, F, E>(
    params: CKKSTestParams,
    module: &Module<BE>,
    host_module: &Module<HostBytesBackend>,
) where
    BE: TestContextBackend,
    Module<BE>: TestContextModule<BE>,
    F: TestScalar,
    E: NegacyclicFFT<F> + NegacyclicFFTNew<F>,
{
    let m = params.n / 2;
    let encoder = Encoder::<E>::new(m).unwrap();
    let sk = gen_sk(&params, module, host_module, [0u8; 32]);
    let mut scratch = alloc_scratch(&params, module);

    let (a_re, a_im) = quantized_vector(host_module, &encoder, &params, TestVector::First, params.prec.log_delta);
    let (const_re, const_im) = quantized_const::<F>(ADD_SUB_CONST.0, ADD_SUB_CONST.1, PT_PREC.log_delta);
    let ct = ckks_encrypt(
        &params,
        module,
        host_module,
        &encoder,
        &sk,
        params.k,
        &a_re,
        &a_im,
        &mut scratch.borrow(),
    );
    let (want_re, want_im) = want_add_const(&a_re, &a_im, const_re, const_im);
    let mut ct_res = alloc_ct(&params, module, params.k);
    let cst = add_sub_const_pt::<BE, F>(host_module, module, params.base2k.into());
    module
        .ckks_add_pt_const_into(&mut ct_res, &ct, 0, &cst, 0, &mut scratch.borrow())
        .unwrap();
    module
        .ckks_add_pt_const_assign(&mut ct_res, m, &cst, 1, &mut scratch.borrow())
        .unwrap();
    assert_unary_output_meta("add_const_into_delta_log_delta", &ct_res, &ct);
    assert_decrypt_precision_at_log_delta(
        "add_const_into_delta_log_delta",
        &params,
        module,
        &encoder,
        &ct_res,
        &sk,
        &want_re,
        &want_im,
        PT_PREC.log_delta,
        &mut scratch.borrow(),
    );
}

/// ct + complex constant, out-of-place, output buffer has smaller max_k than `a` (offset > 0).
pub fn test_add_const_into_smaller_output<BE, F, E>(
    params: CKKSTestParams,
    module: &Module<BE>,
    host_module: &Module<HostBytesBackend>,
) where
    BE: TestContextBackend,
    Module<BE>: TestContextModule<BE>,
    F: TestScalar,
    E: NegacyclicFFT<F> + NegacyclicFFTNew<F>,
{
    let m = params.n / 2;
    let encoder = Encoder::<E>::new(m).unwrap();
    let (re1, im1) = test_vector_1::<F>(m);
    let sk = gen_sk(&params, module, host_module, [0u8; 32]);
    let mut scratch = alloc_scratch(&params, module);

    let ct = ckks_encrypt(
        &params,
        module,
        host_module,
        &encoder,
        &sk,
        params.k,
        &re1,
        &im1,
        &mut scratch.borrow(),
    );
    let (const_re, const_im) = quantized_const::<F>(ADD_SUB_CONST.0, ADD_SUB_CONST.1, PT_PREC.log_delta);
    let (want_re, want_im) = want_add_const(&re1, &im1, const_re, const_im);
    let mut ct_res = alloc_ct(&params, module, params.k - params.base2k - 1);
    let cst = add_sub_const_pt::<BE, F>(host_module, module, params.base2k.into());
    module
        .ckks_add_pt_const_into(&mut ct_res, &ct, 0, &cst, 0, &mut scratch.borrow())
        .unwrap();
    module
        .ckks_add_pt_const_assign(&mut ct_res, m, &cst, 1, &mut scratch.borrow())
        .unwrap();
    assert_unary_output_meta("add_const_into_smaller_output", &ct_res, &ct);
    assert_decrypt_precision(
        "add_const_into_smaller_output",
        &params,
        module,
        &encoder,
        &ct_res,
        &sk,
        &want_re,
        &want_im,
        &mut scratch.borrow(),
    );
}

/// ct + real constant only, out-of-place.
pub fn test_add_const_into_real_only<BE, F, E>(
    params: CKKSTestParams,
    module: &Module<BE>,
    host_module: &Module<HostBytesBackend>,
) where
    BE: TestContextBackend,
    Module<BE>: TestContextModule<BE>,
    F: TestScalar,
    E: NegacyclicFFT<F> + NegacyclicFFTNew<F>,
{
    let m = params.n / 2;
    let encoder = Encoder::<E>::new(m).unwrap();
    let (re1, im1) = test_vector_1::<F>(m);
    let sk = gen_sk(&params, module, host_module, [0u8; 32]);
    let mut scratch = alloc_scratch(&params, module);

    let ct = ckks_encrypt(
        &params,
        module,
        host_module,
        &encoder,
        &sk,
        params.k,
        &re1,
        &im1,
        &mut scratch.borrow(),
    );
    let const_re_f64 = ADD_SUB_CONST.0;
    let (const_re, const_im) = quantized_const::<F>(const_re_f64, 0.0, PT_PREC.log_delta);
    let (want_re, want_im) = want_add_const(&re1, &im1, const_re, const_im);
    let mut ct_res = alloc_ct(&params, module, params.k);
    let cst = ckks_pt_cst::<BE, F>(host_module, module, params.base2k.into(), PT_PREC, Some(const_re_f64), None);
    module
        .ckks_add_pt_const_into(&mut ct_res, &ct, 0, &cst, 0, &mut scratch.borrow())
        .unwrap();
    assert_unary_output_meta("add_const_into_real_only", &ct_res, &ct);
    assert_decrypt_precision(
        "add_const_into_real_only",
        &params,
        module,
        &encoder,
        &ct_res,
        &sk,
        &want_re,
        &want_im,
        &mut scratch.borrow(),
    );
}

/// ct + ZNX plaintext, out-of-place, output buffer has smaller max_k than `a` (offset > 0).
pub fn test_add_pt_vec_into_smaller_output<BE, F, E>(
    params: CKKSTestParams,
    module: &Module<BE>,
    host_module: &Module<HostBytesBackend>,
) where
    BE: TestContextBackend,
    Module<BE>: TestContextModule<BE>,
    F: TestScalar,
    E: NegacyclicFFT<F> + NegacyclicFFTNew<F>,
{
    let m = params.n / 2;
    let encoder = Encoder::<E>::new(m).unwrap();
    let (re1, im1) = test_vector_1::<F>(m);
    let (re2, im2) = test_vector_2::<F>(m);
    let sk = gen_sk(&params, module, host_module, [0u8; 32]);
    let mut scratch = alloc_scratch(&params, module);

    let ct1 = ckks_encrypt(
        &params,
        module,
        host_module,
        &encoder,
        &sk,
        params.k,
        &re1,
        &im1,
        &mut scratch.borrow(),
    );
    let pt = encode_and_upload_pt(host_module, module, &encoder, params.base2k.into(), params.prec, &re2, &im2);
    let (want_re, want_im) = want_add(
        &re1,
        &im1,
        &quantize(&re2, params.prec.log_delta),
        &quantize(&im2, params.prec.log_delta),
    );
    let mut ct_res = alloc_ct(&params, module, params.k - params.base2k - 1);
    module
        .ckks_add_pt_vec_into(&mut ct_res, &ct1, &pt, &mut scratch.borrow())
        .unwrap();
    assert_unary_output_meta("add_pt_vec smaller_output", &ct_res, &ct1);
    assert_decrypt_precision(
        "add_pt_vec smaller_output",
        &params,
        module,
        &encoder,
        &ct_res,
        &sk,
        &want_re,
        &want_im,
        &mut scratch.borrow(),
    );
}

/// ct + ZNX plaintext must reject mismatched base2k with an explicit error.
pub fn test_add_pt_vec_base2k_mismatch_error<BE, F, E>(
    params: CKKSTestParams,
    module: &Module<BE>,
    host_module: &Module<HostBytesBackend>,
) where
    BE: TestContextBackend,
    Module<BE>: TestContextModule<BE>,
    F: TestScalar,
    E: NegacyclicFFT<F> + NegacyclicFFTNew<F>,
{
    let m = params.n / 2;
    let encoder = Encoder::<E>::new(m).unwrap();
    let (re1, im1) = test_vector_1::<F>(m);
    let sk = gen_sk(&params, module, host_module, [0u8; 32]);
    let mut scratch = alloc_scratch(&params, module);

    let mut ct = ckks_encrypt(
        &params,
        module,
        host_module,
        &encoder,
        &sk,
        params.k,
        &re1,
        &im1,
        &mut scratch.borrow(),
    );
    let mismatched_base2k = Base2K((params.base2k / 2) as u32);
    let pt = host_module.ckks_pt_vec_alloc(mismatched_base2k, PT_PREC);
    let err = module
        .ckks_add_pt_vec_assign(&mut ct, &pt, &mut scratch.borrow())
        .unwrap_err();
    assert_ckks_error(
        "add_pt_vec_base2k_mismatch",
        &err,
        CKKSCompositionError::PlaintextBase2KMismatch {
            op: "ckks_add_pt_vec",
            ct_base2k: params.base2k,
            pt_base2k: mismatched_base2k.as_usize(),
        },
    );
}
