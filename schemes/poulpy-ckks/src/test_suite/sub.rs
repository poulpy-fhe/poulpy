//! Subtraction tests: ct-ct, ct-pt, ct-const (out-of-place and in-place).

use crate::{CKKSInfos, api::CKKSSubOps};
use poulpy_core::layouts::{IntPolyInfos, LWEInfos};

use super::helpers::{
    ADD_SUB_CONST, PT_PREC, TestContextBackend, TestContextModule, TestScalar, TestVector, add_sub_const_pt, alloc_ct,
    alloc_scratch, assert_binary_output_meta, assert_ct_meta, assert_decrypt_precision, assert_decrypt_precision_at_log_delta,
    assert_unary_output_meta, ckks_encrypt, ckks_encrypt_with_prec, ckks_pt_cst, ckks_spec, encode_and_upload_pt, gen_sk,
    precision_at, quantize, quantized_const, quantized_vector, test_vector_1, test_vector_2, want_sub,
};
use super::parity::helpers::with_scratch;
use poulpy_hal::{
    api::{NegacyclicFFT, NegacyclicFFTNew, ScratchOwnedBorrow},
    layouts::{HostBytesBackend, Module, Standard},
};

use crate::{test_suite::CKKSTestParams, test_suite::reference_encoder::ReferenceEncoder};

const DELTA_LOG_DELTA: usize = 12;

pub fn test_sub_ct_aligned<BE, F, E>(params: CKKSTestParams, module: &Module<BE>, host_module: &Module<HostBytesBackend>)
where
    BE: TestContextBackend<Ring = Standard>,
    for<'a> <BE as poulpy_hal::layouts::Backend>::BufRef<'a>: poulpy_hal::layouts::HostDataRef,
    for<'a> <BE as poulpy_hal::layouts::Backend>::BufMut<'a>: poulpy_hal::layouts::HostDataMut,
    Module<BE>: TestContextModule<BE>,
    F: TestScalar,
    E: NegacyclicFFT<F> + NegacyclicFFTNew<F>,
{
    let m = params.n / 2;
    let encoder = ReferenceEncoder::<E>::new(m).unwrap();
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
    let (want_re, want_im) = want_sub(&re1, &im1, &re2, &im2);
    let mut ct_res = alloc_ct(&params, module, params.k);
    with_scratch::<BE, _>(module.ckks_sub_tmp_bytes(ct_res.max_size()), |s| {
        module.ckks_sub_into(&mut ct_res, &ct1, &ct2, s)
    })
    .unwrap();
    assert_binary_output_meta("sub_ct_aligned", &ct_res, params.k, &ct1, &ct2);
    assert_decrypt_precision(
        "sub_ct_aligned",
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

pub fn test_sub_ct_delta_a_lt_b<BE, F, E>(params: CKKSTestParams, module: &Module<BE>, host_module: &Module<HostBytesBackend>)
where
    BE: TestContextBackend<Ring = Standard>,
    for<'a> <BE as poulpy_hal::layouts::Backend>::BufRef<'a>: poulpy_hal::layouts::HostDataRef,
    for<'a> <BE as poulpy_hal::layouts::Backend>::BufMut<'a>: poulpy_hal::layouts::HostDataMut,
    Module<BE>: TestContextModule<BE>,
    F: TestScalar,
    E: NegacyclicFFT<F> + NegacyclicFFTNew<F>,
{
    let m = params.n / 2;
    let encoder = ReferenceEncoder::<E>::new(m).unwrap();
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
    let (want_re, want_im) = want_sub(&re1, &im1, &re2, &im2);
    let mut ct_res = alloc_ct(&params, module, params.k);
    with_scratch::<BE, _>(module.ckks_sub_tmp_bytes(ct_res.max_size()), |s| {
        module.ckks_sub_into(&mut ct_res, &ct1, &ct2, s)
    })
    .unwrap();
    assert_binary_output_meta("sub_ct a_lt_b", &ct_res, params.k, &ct1, &ct2);
    assert_decrypt_precision(
        "sub_ct a_lt_b",
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

pub fn test_sub_ct_delta_a_gt_b<BE, F, E>(params: CKKSTestParams, module: &Module<BE>, host_module: &Module<HostBytesBackend>)
where
    BE: TestContextBackend<Ring = Standard>,
    for<'a> <BE as poulpy_hal::layouts::Backend>::BufRef<'a>: poulpy_hal::layouts::HostDataRef,
    for<'a> <BE as poulpy_hal::layouts::Backend>::BufMut<'a>: poulpy_hal::layouts::HostDataMut,
    Module<BE>: TestContextModule<BE>,
    F: TestScalar,
    E: NegacyclicFFT<F> + NegacyclicFFTNew<F>,
{
    let m = params.n / 2;
    let encoder = ReferenceEncoder::<E>::new(m).unwrap();
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
    let (want_re, want_im) = want_sub(&re1, &im1, &re2, &im2);
    let mut ct_res = alloc_ct(&params, module, params.k);
    with_scratch::<BE, _>(module.ckks_sub_tmp_bytes(ct_res.max_size()), |s| {
        module.ckks_sub_into(&mut ct_res, &ct1, &ct2, s)
    })
    .unwrap();
    assert_binary_output_meta("sub_ct a_gt_b", &ct_res, params.k, &ct1, &ct2);
    assert_decrypt_precision(
        "sub_ct a_gt_b",
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

pub fn test_sub_ct_delta_log_delta<BE, F, E>(params: CKKSTestParams, module: &Module<BE>, host_module: &Module<HostBytesBackend>)
where
    BE: TestContextBackend<Ring = Standard>,
    for<'a> <BE as poulpy_hal::layouts::Backend>::BufRef<'a>: poulpy_hal::layouts::HostDataRef,
    for<'a> <BE as poulpy_hal::layouts::Backend>::BufMut<'a>: poulpy_hal::layouts::HostDataMut,
    Module<BE>: TestContextModule<BE>,
    F: TestScalar,
    E: NegacyclicFFT<F> + NegacyclicFFTNew<F>,
{
    let m = params.n / 2;
    let encoder = ReferenceEncoder::<E>::new(m).unwrap();
    let sk = gen_sk(&params, module, host_module, [0u8; 32]);
    let mut scratch = alloc_scratch(&params, module);

    let low_log_delta = params.prec().log_delta() - DELTA_LOG_DELTA;
    let low_prec = precision_at(&params, low_log_delta);
    let (a_re, a_im) = quantized_vector(host_module, &encoder, &params, TestVector::First, params.prec().log_delta());
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
    let (want_re, want_im) = want_sub(&a_re, &a_im, &b_re, &b_im);
    let mut ct_res = alloc_ct(&params, module, params.k);
    with_scratch::<BE, _>(module.ckks_sub_tmp_bytes(ct_res.max_size()), |s| {
        module.ckks_sub_into(&mut ct_res, &ct1, &ct2, s)
    })
    .unwrap();
    assert_binary_output_meta("sub_ct delta_log_delta", &ct_res, params.k, &ct1, &ct2);
    assert_decrypt_precision_at_log_delta(
        "sub_ct delta_log_delta",
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

pub fn test_sub_ct_smaller_output<BE, F, E>(params: CKKSTestParams, module: &Module<BE>, host_module: &Module<HostBytesBackend>)
where
    BE: TestContextBackend<Ring = Standard>,
    for<'a> <BE as poulpy_hal::layouts::Backend>::BufRef<'a>: poulpy_hal::layouts::HostDataRef,
    for<'a> <BE as poulpy_hal::layouts::Backend>::BufMut<'a>: poulpy_hal::layouts::HostDataMut,
    Module<BE>: TestContextModule<BE>,
    F: TestScalar,
    E: NegacyclicFFT<F> + NegacyclicFFTNew<F>,
{
    let m = params.n / 2;
    let encoder = ReferenceEncoder::<E>::new(m).unwrap();
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
    let (want_re, want_im) = want_sub(&re1, &im1, &re2, &im2);
    let dst_k = params.k - params.base2k - 1;
    let mut ct_res = alloc_ct(&params, module, dst_k);
    with_scratch::<BE, _>(module.ckks_sub_tmp_bytes(ct_res.max_size()), |s| {
        module.ckks_sub_into(&mut ct_res, &ct1, &ct2, s)
    })
    .unwrap();
    assert_binary_output_meta("sub_ct smaller_output", &ct_res, dst_k, &ct1, &ct2);
    assert_decrypt_precision(
        "sub_ct smaller_output",
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

pub fn test_sub_ct_assign_aligned<BE, F, E>(params: CKKSTestParams, module: &Module<BE>, host_module: &Module<HostBytesBackend>)
where
    BE: TestContextBackend<Ring = Standard>,
    for<'a> <BE as poulpy_hal::layouts::Backend>::BufRef<'a>: poulpy_hal::layouts::HostDataRef,
    for<'a> <BE as poulpy_hal::layouts::Backend>::BufMut<'a>: poulpy_hal::layouts::HostDataMut,
    Module<BE>: TestContextModule<BE>,
    F: TestScalar,
    E: NegacyclicFFT<F> + NegacyclicFFTNew<F>,
{
    let m = params.n / 2;
    let encoder = ReferenceEncoder::<E>::new(m).unwrap();
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
    let (want_re, want_im) = want_sub(&re1, &im1, &re2, &im2);
    let expected_log_budget = ct1.log_budget().min(ct2.log_budget());
    let expected_log_delta = ct1.log_delta().max(ct2.log_delta());
    with_scratch::<BE, _>(module.ckks_sub_tmp_bytes(ct1.max_size()), |s| {
        module.ckks_sub_assign(&mut ct1, &ct2, s)
    })
    .unwrap();
    assert_ct_meta("sub_ct_assign_aligned", &ct1, expected_log_delta, expected_log_budget);
    assert_decrypt_precision(
        "sub_ct_assign_aligned",
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

pub fn test_sub_ct_assign_self_lt<BE, F, E>(params: CKKSTestParams, module: &Module<BE>, host_module: &Module<HostBytesBackend>)
where
    BE: TestContextBackend<Ring = Standard>,
    for<'a> <BE as poulpy_hal::layouts::Backend>::BufRef<'a>: poulpy_hal::layouts::HostDataRef,
    for<'a> <BE as poulpy_hal::layouts::Backend>::BufMut<'a>: poulpy_hal::layouts::HostDataMut,
    Module<BE>: TestContextModule<BE>,
    F: TestScalar,
    E: NegacyclicFFT<F> + NegacyclicFFTNew<F>,
{
    let m = params.n / 2;
    let encoder = ReferenceEncoder::<E>::new(m).unwrap();
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
    let (want_re, want_im) = want_sub(&re1, &im1, &re2, &im2);
    let expected_log_budget = ct_self.log_budget().min(ct_other.log_budget());
    let expected_log_delta = ct_self.log_delta().max(ct_other.log_delta());
    with_scratch::<BE, _>(module.ckks_sub_tmp_bytes(ct_self.max_size()), |s| {
        module.ckks_sub_assign(&mut ct_self, &ct_other, s)
    })
    .unwrap();
    assert_ct_meta("sub_ct_assign self_lt", &ct_self, expected_log_delta, expected_log_budget);
    assert_decrypt_precision(
        "sub_ct_assign self_lt",
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

pub fn test_sub_ct_assign_self_gt<BE, F, E>(params: CKKSTestParams, module: &Module<BE>, host_module: &Module<HostBytesBackend>)
where
    BE: TestContextBackend<Ring = Standard>,
    for<'a> <BE as poulpy_hal::layouts::Backend>::BufRef<'a>: poulpy_hal::layouts::HostDataRef,
    for<'a> <BE as poulpy_hal::layouts::Backend>::BufMut<'a>: poulpy_hal::layouts::HostDataMut,
    Module<BE>: TestContextModule<BE>,
    F: TestScalar,
    E: NegacyclicFFT<F> + NegacyclicFFTNew<F>,
{
    let m = params.n / 2;
    let encoder = ReferenceEncoder::<E>::new(m).unwrap();
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
    let (want_re, want_im) = want_sub(&re1, &im1, &re2, &im2);
    let expected_log_budget = ct_self.log_budget().min(ct_other.log_budget());
    let expected_log_delta = ct_self.log_delta().max(ct_other.log_delta());
    with_scratch::<BE, _>(module.ckks_sub_tmp_bytes(ct_self.max_size()), |s| {
        module.ckks_sub_assign(&mut ct_self, &ct_other, s)
    })
    .unwrap();
    assert_ct_meta("sub_ct_assign self_gt", &ct_self, expected_log_delta, expected_log_budget);
    assert_decrypt_precision(
        "sub_ct_assign self_gt",
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

pub fn test_sub_pt_vec_assign<BE, F, E>(params: CKKSTestParams, module: &Module<BE>, host_module: &Module<HostBytesBackend>)
where
    BE: TestContextBackend<Ring = Standard>,
    for<'a> <BE as poulpy_hal::layouts::Backend>::BufRef<'a>: poulpy_hal::layouts::HostDataRef,
    for<'a> <BE as poulpy_hal::layouts::Backend>::BufMut<'a>: poulpy_hal::layouts::HostDataMut,
    Module<BE>: TestContextModule<BE>,
    F: TestScalar,
    E: NegacyclicFFT<F> + NegacyclicFFTNew<F>,
{
    let m = params.n / 2;
    let encoder = ReferenceEncoder::<E>::new(m).unwrap();
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
    let pt = encode_and_upload_pt(host_module, module, &encoder, params.base2k.into(), params.prec(), &re2, &im2);
    let (want_re, want_im) = want_sub(
        &re1,
        &im1,
        &quantize(&re2, params.prec().log_delta()),
        &quantize(&im2, params.prec().log_delta()),
    );
    let expected_log_delta = ct.log_delta();
    let expected_log_budget = ct.log_budget();
    let provenance = ct.noise();
    assert!(provenance.is_some());
    with_scratch::<BE, _>(module.ckks_sub_pt_vec_tmp_bytes(ct.max_size()), |s| {
        module.ckks_sub_pt_vec_assign(&mut ct, &pt, s)
    })
    .unwrap();
    assert_eq!(ct.noise(), None);
    assert_ct_meta("sub_pt_vec_assign", &ct, expected_log_delta, expected_log_budget);
    assert_decrypt_precision(
        "sub_pt_vec_assign",
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

pub fn test_sub_pt_vec<BE, F, E>(params: CKKSTestParams, module: &Module<BE>, host_module: &Module<HostBytesBackend>)
where
    BE: TestContextBackend<Ring = Standard>,
    for<'a> <BE as poulpy_hal::layouts::Backend>::BufRef<'a>: poulpy_hal::layouts::HostDataRef,
    for<'a> <BE as poulpy_hal::layouts::Backend>::BufMut<'a>: poulpy_hal::layouts::HostDataMut,
    Module<BE>: TestContextModule<BE>,
    F: TestScalar,
    E: NegacyclicFFT<F> + NegacyclicFFTNew<F>,
{
    let m = params.n / 2;
    let encoder = ReferenceEncoder::<E>::new(m).unwrap();
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
    let pt = encode_and_upload_pt(host_module, module, &encoder, params.base2k.into(), params.prec(), &re2, &im2);
    let (want_re, want_im) = want_sub(
        &re1,
        &im1,
        &quantize(&re2, params.prec().log_delta()),
        &quantize(&im2, params.prec().log_delta()),
    );
    let mut ct_res = alloc_ct(&params, module, params.k);
    with_scratch::<BE, _>(module.ckks_sub_pt_vec_tmp_bytes(ct_res.max_size()), |s| {
        module.ckks_sub_pt_vec_into(&mut ct_res, &ct1, &pt, s)
    })
    .unwrap();
    assert_unary_output_meta("sub_pt_vec_into", &ct_res, params.k, &ct1);
    assert!(ct1.noise().is_some());
    assert_eq!(ct_res.noise(), None);
    assert_decrypt_precision(
        "sub_pt_vec_into",
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

pub fn test_sub_pt_vec_into_delta_log_delta<BE, F, E>(
    params: CKKSTestParams,
    module: &Module<BE>,
    host_module: &Module<HostBytesBackend>,
) where
    BE: TestContextBackend<Ring = Standard>,
    for<'a> <BE as poulpy_hal::layouts::Backend>::BufRef<'a>: poulpy_hal::layouts::HostDataRef,
    for<'a> <BE as poulpy_hal::layouts::Backend>::BufMut<'a>: poulpy_hal::layouts::HostDataMut,
    Module<BE>: TestContextModule<BE>,
    F: TestScalar,
    E: NegacyclicFFT<F> + NegacyclicFFTNew<F>,
{
    let m = params.n / 2;
    let encoder = ReferenceEncoder::<E>::new(m).unwrap();
    let (re2, im2) = test_vector_2::<F>(m);
    let sk = gen_sk(&params, module, host_module, [0u8; 32]);
    let mut scratch = alloc_scratch(&params, module);

    let (a_re, a_im) = quantized_vector(host_module, &encoder, &params, TestVector::First, params.prec().log_delta());
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
    let (want_re, want_im) = want_sub(
        &a_re,
        &a_im,
        &quantize(&re2, PT_PREC.log_delta()),
        &quantize(&im2, PT_PREC.log_delta()),
    );
    let mut ct_res = alloc_ct(&params, module, params.k);
    with_scratch::<BE, _>(module.ckks_sub_pt_vec_tmp_bytes(ct_res.max_size()), |s| {
        module.ckks_sub_pt_vec_into(&mut ct_res, &ct1, &pt, s)
    })
    .unwrap();
    assert_unary_output_meta("sub_pt_vec_into delta_log_delta", &ct_res, params.k, &ct1);
    assert_decrypt_precision_at_log_delta(
        "sub_pt_vec_into delta_log_delta",
        &params,
        module,
        &encoder,
        &ct_res,
        &sk,
        &want_re,
        &want_im,
        PT_PREC.log_delta(),
        &mut scratch.borrow(),
    );
}

pub fn test_sub_pt_vec_into_smaller_output<BE, F, E>(
    params: CKKSTestParams,
    module: &Module<BE>,
    host_module: &Module<HostBytesBackend>,
) where
    BE: TestContextBackend<Ring = Standard>,
    for<'a> <BE as poulpy_hal::layouts::Backend>::BufRef<'a>: poulpy_hal::layouts::HostDataRef,
    for<'a> <BE as poulpy_hal::layouts::Backend>::BufMut<'a>: poulpy_hal::layouts::HostDataMut,
    Module<BE>: TestContextModule<BE>,
    F: TestScalar,
    E: NegacyclicFFT<F> + NegacyclicFFTNew<F>,
{
    let m = params.n / 2;
    let encoder = ReferenceEncoder::<E>::new(m).unwrap();
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
    let pt = encode_and_upload_pt(host_module, module, &encoder, params.base2k.into(), params.prec(), &re2, &im2);
    let (want_re, want_im) = want_sub(
        &re1,
        &im1,
        &quantize(&re2, params.prec().log_delta()),
        &quantize(&im2, params.prec().log_delta()),
    );
    let dst_k = params.k - params.base2k - 1;
    let mut ct_res = alloc_ct(&params, module, dst_k);
    with_scratch::<BE, _>(module.ckks_sub_pt_vec_tmp_bytes(ct_res.max_size()), |s| {
        module.ckks_sub_pt_vec_into(&mut ct_res, &ct1, &pt, s)
    })
    .unwrap();
    assert_unary_output_meta("sub_pt_vec_into smaller_output", &ct_res, dst_k, &ct1);
    assert_decrypt_precision(
        "sub_pt_vec_into smaller_output",
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

pub fn test_sub_pt_const_into_aligned<BE, F, E>(
    params: CKKSTestParams,
    module: &Module<BE>,
    host_module: &Module<HostBytesBackend>,
) where
    BE: TestContextBackend<Ring = Standard>,
    for<'a> <BE as poulpy_hal::layouts::Backend>::BufRef<'a>: poulpy_hal::layouts::HostDataRef,
    for<'a> <BE as poulpy_hal::layouts::Backend>::BufMut<'a>: poulpy_hal::layouts::HostDataMut,
    Module<BE>: TestContextModule<BE>,
    F: TestScalar,
    E: NegacyclicFFT<F> + NegacyclicFFTNew<F>,
{
    let m = params.n / 2;
    let encoder = ReferenceEncoder::<E>::new(m).unwrap();
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
    let (const_re, const_im) = quantized_const::<F>(ADD_SUB_CONST.0, ADD_SUB_CONST.1, PT_PREC.log_delta());
    let want_re: Vec<F> = re1.iter().map(|x| *x - const_re).collect();
    let want_im: Vec<F> = im1.iter().map(|x| *x - const_im).collect();
    let mut ct_res = alloc_ct(&params, module, params.k);
    let cst = add_sub_const_pt::<BE, F>(host_module, module, params.base2k.into());
    with_scratch::<BE, _>(module.ckks_sub_pt_const_tmp_bytes(ct_res.max_size()), |s| {
        module.ckks_sub_pt_const_into(&mut ct_res, 0, &ct, &cst, 0, s)
    })
    .unwrap();
    with_scratch::<BE, _>(module.ckks_sub_pt_const_tmp_bytes(ct_res.max_size()), |s| {
        module.ckks_sub_pt_const_assign(&mut ct_res, m, &cst, 1, s)
    })
    .unwrap();
    assert_unary_output_meta("sub_pt_const_into_aligned", &ct_res, params.k, &ct);
    assert_decrypt_precision(
        "sub_pt_const_into_aligned",
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

/// ct - complex constant whose `max_k` exceeds the available precision → left-shift alignment path.
pub fn test_sub_const_into_lsh_alignment<BE, F, E>(
    params: CKKSTestParams,
    module: &Module<BE>,
    host_module: &Module<HostBytesBackend>,
) where
    BE: TestContextBackend<Ring = Standard>,
    for<'a> <BE as poulpy_hal::layouts::Backend>::BufRef<'a>: poulpy_hal::layouts::HostDataRef,
    for<'a> <BE as poulpy_hal::layouts::Backend>::BufMut<'a>: poulpy_hal::layouts::HostDataMut,
    Module<BE>: TestContextModule<BE>,
    F: TestScalar,
    E: NegacyclicFFT<F> + NegacyclicFFTNew<F>,
{
    let m = params.n / 2;
    let encoder = ReferenceEncoder::<E>::new(m).unwrap();
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
    let (const_re, const_im) = quantized_const::<F>(ADD_SUB_CONST.0, ADD_SUB_CONST.1, PT_PREC.log_delta());
    let want_re: Vec<F> = re1.iter().map(|x| *x - const_re).collect();
    let want_im: Vec<F> = im1.iter().map(|x| *x - const_im).collect();
    let mut ct_res = alloc_ct(&params, module, params.k);
    // Padding the constant with the full ciphertext budget makes its allocated
    // max_k round up past the available precision, forcing PlaintextShift::Lsh.
    let cst_prec = ckks_spec(params.n, params.base2k, PT_PREC.log_delta(), ct.log_budget());
    let cst = ckks_pt_cst::<BE, F>(
        host_module,
        module,
        params.base2k.into(),
        cst_prec,
        Some(ADD_SUB_CONST.0),
        Some(ADD_SUB_CONST.1),
    );
    assert!(
        cst.encoded_k().as_usize() > ct.log_budget() + cst.log_delta(),
        "test setup no longer triggers the Lsh alignment path (max_k={} <= available={})",
        cst.encoded_k().as_usize(),
        ct.log_budget() + cst.log_delta()
    );
    with_scratch::<BE, _>(module.ckks_sub_pt_const_tmp_bytes(ct_res.max_size()), |s| {
        module.ckks_sub_pt_const_into(&mut ct_res, 0, &ct, &cst, 0, s)
    })
    .unwrap();
    with_scratch::<BE, _>(module.ckks_sub_pt_const_tmp_bytes(ct_res.max_size()), |s| {
        module.ckks_sub_pt_const_assign(&mut ct_res, m, &cst, 1, s)
    })
    .unwrap();
    assert_unary_output_meta("sub_const_into_lsh_alignment", &ct_res, params.k, &ct);
    assert_decrypt_precision(
        "sub_const_into_lsh_alignment",
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

pub fn test_sub_one_assign<BE, F, E>(params: CKKSTestParams, module: &Module<BE>, host_module: &Module<HostBytesBackend>)
where
    BE: TestContextBackend<Ring = Standard>,
    for<'a> <BE as poulpy_hal::layouts::Backend>::BufRef<'a>: poulpy_hal::layouts::HostDataRef,
    for<'a> <BE as poulpy_hal::layouts::Backend>::BufMut<'a>: poulpy_hal::layouts::HostDataMut,
    Module<BE>: TestContextModule<BE>,
    F: TestScalar,
    E: NegacyclicFFT<F> + NegacyclicFFTNew<F>,
{
    let m = params.n / 2;
    let encoder = ReferenceEncoder::<E>::new(m).unwrap();
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
    let want_re: Vec<F> = re1.iter().map(|x| *x - F::one()).collect();
    let want_im = im1.clone();
    let expected_log_delta = ct.log_delta();
    let expected_log_budget = ct.log_budget();
    with_scratch::<BE, _>(module.ckks_sub_one_tmp_bytes(ct.max_size()), |s| {
        module.ckks_sub_one_assign(&mut ct, s)
    })
    .unwrap();
    assert_ct_meta("sub_one_assign", &ct, expected_log_delta, expected_log_budget);
    assert_decrypt_precision(
        "sub_one_assign",
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

/// A narrower destination, which the copy of `a` would write, keeps its content when the plaintext's checks refuse:
/// a plaintext of another radix (`PlaintextBase2KMismatch`) and a coefficient past the plaintext's degree
/// (`PlaintextCoefficientOutOfRange`), on an empty arena and on the query.
pub fn test_sub_pt_vec_into_rejects_plaintext_before_copy<BE, F, E>(
    params: crate::test_suite::CKKSTestParams,
    module: &poulpy_hal::layouts::Module<BE>,
    _host_module: &poulpy_hal::layouts::Module<poulpy_hal::layouts::HostBytesBackend>,
) where
    BE: crate::test_suite::helpers::TestContextBackend<Ring = poulpy_hal::layouts::Standard>,
    poulpy_hal::layouts::Module<BE>: crate::test_suite::helpers::TestContextModule<BE>,
    F: crate::test_suite::helpers::TestScalar,
    E: poulpy_hal::api::NegacyclicFFT<F> + poulpy_hal::api::NegacyclicFFTNew<F>,
{
    use crate::test_suite::{
        helpers::assert_ckks_error,
        parity::{
            helpers::{fixture_ciphertext, fixture_plaintext, snapshot, with_scratch},
            layout,
        },
    };
    use crate::{CKKSCompositionError, CKKSLayout, SlotsKind};
    use poulpy_core::layouts::LWEInfos;
    let b = params.base2k;
    let ct_layout = layout(params, params.rank, 3 * b + 5, b - 1, 0, SlotsKind::Complex);
    let pt_layout = layout(params, 0, b + 3, b - 1, 0, SlotsKind::Complex);
    let other_radix = CKKSLayout {
        glwe_layout: poulpy_core::layouts::GLWELayout {
            base2k: (b - 1).into(),
            ..pt_layout.glwe_layout
        },
        meta: pt_layout.meta,
    };
    use crate::api::CKKSSubOps;
    let a = fixture_ciphertext(module, &ct_layout, 41);
    let pt = fixture_plaintext(module, &other_radix, 43);
    let same_radix = fixture_plaintext(module, &pt_layout, 44);
    let n = same_radix.n().as_usize();
    let mut res = fixture_ciphertext(
        module,
        &layout(params, params.rank, 2 * b + 3, b - 1, 0, SlotsKind::Complex),
        31,
    );
    let radix = CKKSCompositionError::PlaintextBase2KMismatch {
        op: "ckks_sub_pt_vec",
        ct_base2k: b,
        pt_base2k: b - 1,
    };
    for bytes in [0, module.ckks_sub_pt_vec_tmp_bytes(res.max_size())] {
        let before = snapshot::<BE, _>(&res);
        let err = with_scratch::<BE, _>(bytes, |s| module.ckks_sub_pt_vec_into(&mut res, &a, &pt, s)).unwrap_err();
        assert_ckks_error("sub_pt_vec_into: plaintext of another radix", &err, radix.clone());
        assert_eq!(snapshot::<BE, _>(&res), before, "a refused plaintext form wrote dst");
    }
    for bytes in [0, module.ckks_sub_pt_const_tmp_bytes(res.max_size())] {
        let before = snapshot::<BE, _>(&res);
        let err =
            with_scratch::<BE, _>(bytes, |s| module.ckks_sub_pt_const_into(&mut res, 0, &a, &same_radix, n, s)).unwrap_err();
        assert_ckks_error(
            "sub_pt_const_into: coefficient past the plaintext",
            &err,
            CKKSCompositionError::PlaintextCoefficientOutOfRange {
                op: "ckks_sub_pt_const",
                role: "plaintext",
                coeff: n,
                n,
            },
        );
        assert_eq!(snapshot::<BE, _>(&res), before, "a refused plaintext form wrote dst");
    }
}
