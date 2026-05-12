//! Subtraction tests for the `CKKSSubOpsUnnormalized` API.

use poulpy_hal::{
    api::{NegacyclicFFT, NegacyclicFFTNew, ScratchOwnedBorrow},
    layouts::{HostBytesBackend, Module},
};

use crate::{CKKSInfos, layouts::UnnormalizedCKKSCiphertext, leveled::api::CKKSSubOpsUnnormalized};

use super::helpers::{
    ADD_SUB_CONST, PT_PREC, TestContextBackend, TestContextModule, TestScalar, add_sub_const_pt, alloc_ct, alloc_scratch,
    assert_binary_output_meta, assert_ct_meta, assert_decrypt_precision, assert_unary_output_meta, ckks_encrypt,
    encode_and_upload_pt, gen_sk, quantize, quantized_const, test_vector_1, test_vector_2, want_sub,
};

use crate::{encoding::reim::Encoder, test_suite::CKKSTestParams};

pub fn test_sub_ct_aligned_unsafe<BE, F, E>(params: CKKSTestParams, module: &Module<BE>, host_module: &Module<HostBytesBackend>)
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
    let mut ct_res = UnnormalizedCKKSCiphertext::new(alloc_ct(&params, module, params.k));
    module
        .ckks_sub_into_unnormalized(&mut ct_res, &ct1, &ct2, &mut scratch.borrow())
        .unwrap();
    assert_binary_output_meta("sub_ct_aligned_unsafe", ct_res.as_inner(), &ct1, &ct2);
    let ct_res = ct_res.normalize(module, &mut scratch.borrow());
    assert_decrypt_precision(
        "sub_ct_aligned_unsafe",
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

pub fn test_sub_ct_assign_aligned_unsafe<BE, F, E>(
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

    let ct1_raw = ckks_encrypt(
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
    let expected_log_budget = ct1_raw.log_budget().min(ct2.log_budget());
    let expected_log_delta = ct1_raw.log_delta().max(ct2.log_delta());
    let mut ct1 = UnnormalizedCKKSCiphertext::new(ct1_raw);
    module
        .ckks_sub_assign_unnormalized(&mut ct1, &ct2, &mut scratch.borrow())
        .unwrap();
    assert_ct_meta(
        "sub_ct_assign_aligned_unsafe",
        ct1.as_inner(),
        expected_log_delta,
        expected_log_budget,
    );
    let ct1 = ct1.normalize(module, &mut scratch.borrow());
    assert_decrypt_precision(
        "sub_ct_assign_aligned_unsafe",
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

pub fn test_sub_pt_vec_into_unsafe<BE, F, E>(params: CKKSTestParams, module: &Module<BE>, host_module: &Module<HostBytesBackend>)
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
    let mut ct_res = UnnormalizedCKKSCiphertext::new(alloc_ct(&params, module, params.k));
    module
        .ckks_sub_pt_vec_into_unnormalized(&mut ct_res, &ct1, &pt, &mut scratch.borrow())
        .unwrap();
    assert_unary_output_meta("sub_pt_vec_into_unsafe", ct_res.as_inner(), &ct1);
    let ct_res = ct_res.normalize(module, &mut scratch.borrow());
    assert_decrypt_precision(
        "sub_pt_vec_into_unsafe",
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

pub fn test_sub_pt_const_into_aligned_unsafe<BE, F, E>(
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
    let mut ct_res = UnnormalizedCKKSCiphertext::new(alloc_ct(&params, module, params.k));
    let cst = add_sub_const_pt::<BE, F>(host_module, module, params.base2k.into());
    module
        .ckks_sub_pt_const_into_unnormalized(&mut ct_res, &ct, 0, &cst, 0, &mut scratch.borrow())
        .unwrap();
    module
        .ckks_sub_pt_const_assign_unnormalized(&mut ct_res, m, &cst, 1, &mut scratch.borrow())
        .unwrap();
    assert_unary_output_meta("sub_pt_const_into_aligned_unsafe", ct_res.as_inner(), &ct);
    let ct_res = ct_res.normalize(module, &mut scratch.borrow());
    assert_decrypt_precision(
        "sub_pt_const_into_aligned_unsafe",
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
