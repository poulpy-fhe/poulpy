//! Subtraction tests: ct-ct, ct-pt, ct-const (out-of-place and in-place).

use crate::{CKKSInfos, leveled::api::CKKSSubOps};

use super::helpers::{
    ADD_SUB_CONST, PT_PREC, TestContextBackend, TestContextModule, TestScalar, TestVector, add_sub_const_pt, alloc_ct,
    alloc_scratch, assert_binary_output_meta, assert_ct_meta, assert_decrypt_precision, assert_decrypt_precision_at_log_delta,
    assert_unary_output_meta, ckks_encrypt, ckks_encrypt_with_prec, encode_and_upload_pt, gen_sk, precision_at, quantize,
    quantized_const, quantized_vector, test_vector_1, test_vector_2, want_sub,
};
use poulpy_hal::{
    api::{NegacyclicFFT, NegacyclicFFTNew, ScratchOwnedBorrow},
    layouts::{HostBytesBackend, Module},
};

use crate::{encoding::reim::Encoder, test_suite::CKKSTestParams};

const DELTA_LOG_DELTA: usize = 12;

pub fn test_sub_ct_aligned<BE, F, E>(params: CKKSTestParams, module: &Module<BE>, host_module: &Module<HostBytesBackend>)
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
    let (want_re, want_im) = want_sub(&re1, &im1, &re2, &im2);
    let mut ct_res = alloc_ct(&params, module, params.k);
    module.ckks_sub_into(&mut ct_res, &ct1, &ct2, &mut scratch.borrow()).unwrap();
    assert_binary_output_meta("sub_ct_aligned", &ct_res, &ct1, &ct2);
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
    let (want_re, want_im) = want_sub(&re1, &im1, &re2, &im2);
    let mut ct_res = alloc_ct(&params, module, params.k);
    module.ckks_sub_into(&mut ct_res, &ct1, &ct2, &mut scratch.borrow()).unwrap();
    assert_binary_output_meta("sub_ct a_lt_b", &ct_res, &ct1, &ct2);
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
    let (want_re, want_im) = want_sub(&re1, &im1, &re2, &im2);
    let mut ct_res = alloc_ct(&params, module, params.k);
    module.ckks_sub_into(&mut ct_res, &ct1, &ct2, &mut scratch.borrow()).unwrap();
    assert_binary_output_meta("sub_ct a_gt_b", &ct_res, &ct1, &ct2);
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
    let (want_re, want_im) = want_sub(&a_re, &a_im, &b_re, &b_im);
    let mut ct_res = alloc_ct(&params, module, params.k);
    module.ckks_sub_into(&mut ct_res, &ct1, &ct2, &mut scratch.borrow()).unwrap();
    assert_binary_output_meta("sub_ct delta_log_delta", &ct_res, &ct1, &ct2);
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
    let (want_re, want_im) = want_sub(&re1, &im1, &re2, &im2);
    let mut ct_res = alloc_ct(&params, module, params.k - params.base2k - 1);
    module.ckks_sub_into(&mut ct_res, &ct1, &ct2, &mut scratch.borrow()).unwrap();
    assert_binary_output_meta("sub_ct smaller_output", &ct_res, &ct1, &ct2);
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
    let (want_re, want_im) = want_sub(&re1, &im1, &re2, &im2);
    let expected_log_budget = ct1.log_budget().min(ct2.log_budget());
    let expected_log_delta = ct1.log_delta().max(ct2.log_delta());
    module.ckks_sub_assign(&mut ct1, &ct2, &mut scratch.borrow()).unwrap();
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
    let (want_re, want_im) = want_sub(&re1, &im1, &re2, &im2);
    let expected_log_budget = ct_self.log_budget().min(ct_other.log_budget());
    let expected_log_delta = ct_self.log_delta().max(ct_other.log_delta());
    module
        .ckks_sub_assign(&mut ct_self, &ct_other, &mut scratch.borrow())
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
    let (want_re, want_im) = want_sub(&re1, &im1, &re2, &im2);
    let expected_log_budget = ct_self.log_budget().min(ct_other.log_budget());
    let expected_log_delta = ct_self.log_delta().max(ct_other.log_delta());
    module
        .ckks_sub_assign(&mut ct_self, &ct_other, &mut scratch.borrow())
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
    let (want_re, want_im) = want_sub(
        &re1,
        &im1,
        &quantize(&re2, params.prec.log_delta),
        &quantize(&im2, params.prec.log_delta),
    );
    let expected_log_delta = ct.log_delta();
    let expected_log_budget = ct.log_budget();
    module.ckks_sub_pt_vec_assign(&mut ct, &pt, &mut scratch.borrow()).unwrap();
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
    let (want_re, want_im) = want_sub(
        &re1,
        &im1,
        &quantize(&re2, params.prec.log_delta),
        &quantize(&im2, params.prec.log_delta),
    );
    let mut ct_res = alloc_ct(&params, module, params.k);
    module
        .ckks_sub_pt_vec_into(&mut ct_res, &ct1, &pt, &mut scratch.borrow())
        .unwrap();
    assert_unary_output_meta("sub_pt_vec_into", &ct_res, &ct1);
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
    let (want_re, want_im) = want_sub(
        &a_re,
        &a_im,
        &quantize(&re2, PT_PREC.log_delta),
        &quantize(&im2, PT_PREC.log_delta),
    );
    let mut ct_res = alloc_ct(&params, module, params.k);
    module
        .ckks_sub_pt_vec_into(&mut ct_res, &ct1, &pt, &mut scratch.borrow())
        .unwrap();
    assert_unary_output_meta("sub_pt_vec_into delta_log_delta", &ct_res, &ct1);
    assert_decrypt_precision_at_log_delta(
        "sub_pt_vec_into delta_log_delta",
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

pub fn test_sub_pt_vec_into_smaller_output<BE, F, E>(
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
    let (want_re, want_im) = want_sub(
        &re1,
        &im1,
        &quantize(&re2, params.prec.log_delta),
        &quantize(&im2, params.prec.log_delta),
    );
    let mut ct_res = alloc_ct(&params, module, params.k - params.base2k - 1);
    module
        .ckks_sub_pt_vec_into(&mut ct_res, &ct1, &pt, &mut scratch.borrow())
        .unwrap();
    assert_unary_output_meta("sub_pt_vec_into smaller_output", &ct_res, &ct1);
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
    let want_re: Vec<F> = re1.iter().map(|x| *x - const_re).collect();
    let want_im: Vec<F> = im1.iter().map(|x| *x - const_im).collect();
    let mut ct_res = alloc_ct(&params, module, params.k);
    let cst = add_sub_const_pt::<BE, F>(host_module, module, params.base2k.into());
    module
        .ckks_sub_pt_const_into(&mut ct_res, &ct, 0, &cst, 0, &mut scratch.borrow())
        .unwrap();
    module
        .ckks_sub_pt_const_assign(&mut ct_res, m, &cst, 1, &mut scratch.borrow())
        .unwrap();
    assert_unary_output_meta("sub_pt_const_into_aligned", &ct_res, &ct);
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

pub fn test_sub_one_assign<BE, F, E>(params: CKKSTestParams, module: &Module<BE>, host_module: &Module<HostBytesBackend>)
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
    let want_re: Vec<F> = re1.iter().map(|x| *x - F::one()).collect();
    let want_im = im1.clone();
    let expected_log_delta = ct.log_delta();
    let expected_log_budget = ct.log_budget();
    module.ckks_sub_one_assign(&mut ct, &mut scratch.borrow()).unwrap();
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
