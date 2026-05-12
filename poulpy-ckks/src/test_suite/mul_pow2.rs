//! Multiplication and division by a power of two.

use crate::{CKKSCompositionError, CKKSInfos, leveled::api::CKKSPow2Ops};
use poulpy_core::layouts::LWEInfos;

use super::helpers::{
    TestContextBackend, TestContextModule, TestScalar, alloc_ct, alloc_scratch, assert_ckks_error, assert_ct_meta,
    assert_decrypt_precision, assert_decrypt_precision_at_log_delta, assert_unary_output_meta, ckks_encrypt, gen_sk,
    test_vector_1, want_div_pow2, want_mul_pow2,
};
use poulpy_hal::{
    api::{NegacyclicFFT, NegacyclicFFTNew, ScratchOwnedBorrow},
    layouts::{HostBytesBackend, Module},
};

use crate::{encoding::reim::Encoder, test_suite::CKKSTestParams};

const SHIFT_BITS: usize = 7;

pub fn test_mul_pow2_aligned<BE, F, E>(params: CKKSTestParams, module: &Module<BE>, host_module: &Module<HostBytesBackend>)
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
    let (want_re, want_im) = want_mul_pow2(&re1, &im1, SHIFT_BITS);
    let mut ct_res = alloc_ct(&params, module, params.k);
    module
        .ckks_mul_pow2_into(&mut ct_res, &ct, SHIFT_BITS, &mut scratch.borrow())
        .unwrap();
    assert_unary_output_meta("mul_pow2", &ct_res, &ct);
    assert_decrypt_precision_at_log_delta(
        "mul_pow2",
        &params,
        module,
        &encoder,
        &ct_res,
        &sk,
        &want_re,
        &want_im,
        ct.log_delta() - SHIFT_BITS,
        &mut scratch.borrow(),
    );
}

pub fn test_mul_pow2_smaller_output<BE, F, E>(params: CKKSTestParams, module: &Module<BE>, host_module: &Module<HostBytesBackend>)
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
    let (want_re, want_im) = want_mul_pow2(&re1, &im1, SHIFT_BITS);
    let mut ct_res = alloc_ct(&params, module, params.k - params.base2k - 1);
    module
        .ckks_mul_pow2_into(&mut ct_res, &ct, SHIFT_BITS, &mut scratch.borrow())
        .unwrap();
    assert_unary_output_meta("mul_pow2 smaller_output", &ct_res, &ct);
    assert_decrypt_precision_at_log_delta(
        "mul_pow2",
        &params,
        module,
        &encoder,
        &ct_res,
        &sk,
        &want_re,
        &want_im,
        ct.log_delta() - SHIFT_BITS,
        &mut scratch.borrow(),
    );
}

pub fn test_mul_pow2_assign<BE, F, E>(params: CKKSTestParams, module: &Module<BE>, host_module: &Module<HostBytesBackend>)
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
    let (want_re, want_im) = want_mul_pow2(&re1, &im1, SHIFT_BITS);
    let expected_log_delta = ct.log_delta();
    let expected_log_budget = ct.log_budget();
    module
        .ckks_mul_pow2_assign(&mut ct, SHIFT_BITS, &mut scratch.borrow())
        .unwrap();
    assert_ct_meta("mul_pow2_assign", &ct, expected_log_delta, expected_log_budget);
    assert_decrypt_precision_at_log_delta(
        "mul_pow2_assign",
        &params,
        module,
        &encoder,
        &ct,
        &sk,
        &want_re,
        &want_im,
        expected_log_delta - SHIFT_BITS,
        &mut scratch.borrow(),
    );
}

pub fn test_div_pow2_aligned<BE, F, E>(params: CKKSTestParams, module: &Module<BE>, host_module: &Module<HostBytesBackend>)
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
    let (want_re, want_im) = want_div_pow2(&re1, &im1, SHIFT_BITS);
    let mut ct_res = alloc_ct(&params, module, params.k);
    module
        .ckks_div_pow2_into(&mut ct_res, &ct, SHIFT_BITS, &mut scratch.borrow())
        .unwrap();
    assert_ct_meta("div_pow2", &ct_res, ct.log_delta() + SHIFT_BITS, ct.log_budget() - SHIFT_BITS);
    assert_decrypt_precision(
        "div_pow2",
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

pub fn test_div_pow2_smaller_output<BE, F, E>(params: CKKSTestParams, module: &Module<BE>, host_module: &Module<HostBytesBackend>)
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
    let (want_re, want_im) = want_div_pow2(&re1, &im1, SHIFT_BITS);
    let mut ct_res = alloc_ct(&params, module, params.k - params.base2k - 1);
    module
        .ckks_div_pow2_into(&mut ct_res, &ct, SHIFT_BITS, &mut scratch.borrow())
        .unwrap();
    let offset = ct.effective_k().saturating_sub(ct_res.max_k().as_usize());
    assert_ct_meta(
        "div_pow2 smaller_output",
        &ct_res,
        ct.log_delta() + SHIFT_BITS,
        ct.log_budget() - SHIFT_BITS - offset,
    );
    assert_decrypt_precision(
        "div_pow2",
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

pub fn test_div_pow2_assign<BE, F, E>(params: CKKSTestParams, module: &Module<BE>, host_module: &Module<HostBytesBackend>)
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
    let (want_re, want_im) = want_div_pow2(&re1, &im1, SHIFT_BITS);
    let expected_log_delta = ct.log_delta();
    let expected_log_budget = ct.log_budget() - SHIFT_BITS;
    module.ckks_div_pow2_assign(&mut ct, SHIFT_BITS).unwrap();
    assert_ct_meta("div_pow2_assign", &ct, expected_log_delta, expected_log_budget);
    assert_decrypt_precision(
        "div_pow2_assign",
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

pub fn test_div_pow2_assign_explicit_error<BE, F, E>(
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
    let available_log_budget = ct.log_budget();
    let required_bits = available_log_budget + 1;
    let err = module.ckks_div_pow2_assign(&mut ct, required_bits).unwrap_err();
    assert_ckks_error(
        "div_pow2_assign_explicit_error",
        &err,
        CKKSCompositionError::InsufficientHomomorphicCapacity {
            op: "div_pow2_assign",
            available_log_budget,
            required_bits,
        },
    );
}
