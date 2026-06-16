//! Value-preserving re-scaling of a ciphertext's working scale.

use crate::{CKKSCompositionError, CKKSInfos, leveled::api::CKKSRescaleOps};

use super::helpers::{
    TestContextBackend, TestContextModule, TestScalar, alloc_scratch, assert_ckks_error, assert_ct_meta,
    assert_decrypt_precision, assert_decrypt_precision_at_log_delta, ckks_encrypt, gen_sk, test_vector_1,
};
use poulpy_hal::{
    api::{NegacyclicFFT, NegacyclicFFTNew, ScratchOwnedBorrow},
    layouts::{HostBytesBackend, Module},
};

use crate::{encoding::reim::Encoder, test_suite::CKKSTestParams};

const SCALE_BITS: usize = 5;

pub fn test_scale_down_assign<BE, F, E>(params: CKKSTestParams, module: &Module<BE>, host_module: &Module<HostBytesBackend>)
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
    let expected_log_delta = ct.log_delta() - SCALE_BITS;
    let expected_log_budget = ct.log_budget() + SCALE_BITS;
    module
        .ckks_scale_down_assign(&mut ct, SCALE_BITS, &mut scratch.borrow())
        .unwrap();
    assert_ct_meta("scale_down_assign", &ct, expected_log_delta, expected_log_budget);
    // The encrypted value is preserved; only the precision drops to the new log_delta.
    assert_decrypt_precision(
        "scale_down_assign",
        &params,
        module,
        &encoder,
        &ct,
        &sk,
        &re1,
        &im1,
        &mut scratch.borrow(),
    );
}

pub fn test_scale_up_assign<BE, F, E>(params: CKKSTestParams, module: &Module<BE>, host_module: &Module<HostBytesBackend>)
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
    let original_log_delta = ct.log_delta();
    let expected_log_budget = ct.log_budget() - SCALE_BITS;
    module
        .ckks_scale_up_assign(&mut ct, SCALE_BITS, &mut scratch.borrow())
        .unwrap();
    assert_ct_meta("scale_up_assign", &ct, original_log_delta + SCALE_BITS, expected_log_budget);
    // The value is preserved: the shifted-in bits are zero, so the recovered
    // precision is still the original log_delta, not the inflated one.
    assert_decrypt_precision_at_log_delta(
        "scale_up_assign",
        &params,
        module,
        &encoder,
        &ct,
        &sk,
        &re1,
        &im1,
        original_log_delta,
        &mut scratch.borrow(),
    );
}

pub fn test_scale_round_trip<BE, F, E>(params: CKKSTestParams, module: &Module<BE>, host_module: &Module<HostBytesBackend>)
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
    let original_log_delta = ct.log_delta();
    let original_log_budget = ct.log_budget();
    // Scaling up then back down restores both the metadata and the value.
    module
        .ckks_scale_up_assign(&mut ct, SCALE_BITS, &mut scratch.borrow())
        .unwrap();
    module
        .ckks_scale_down_assign(&mut ct, SCALE_BITS, &mut scratch.borrow())
        .unwrap();
    assert_ct_meta("scale_round_trip", &ct, original_log_delta, original_log_budget);
    assert_decrypt_precision(
        "scale_round_trip",
        &params,
        module,
        &encoder,
        &ct,
        &sk,
        &re1,
        &im1,
        &mut scratch.borrow(),
    );
}

pub fn test_scale_up_insufficient_budget_error<BE, F, E>(
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
    let err = module
        .ckks_scale_up_assign(&mut ct, required_bits, &mut scratch.borrow())
        .unwrap_err();
    assert_ckks_error(
        "scale_up_insufficient_budget_error",
        &err,
        CKKSCompositionError::InsufficientHomomorphicCapacity {
            op: "scale_up",
            available_log_budget,
            required_bits,
        },
    );
}

pub fn test_scale_down_insufficient_precision_error<BE, F, E>(
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
    let available_log_delta = ct.log_delta();
    let required_bits = available_log_delta + 1;
    let err = module
        .ckks_scale_down_assign(&mut ct, required_bits, &mut scratch.borrow())
        .unwrap_err();
    assert_ckks_error(
        "scale_down_insufficient_precision_error",
        &err,
        CKKSCompositionError::InsufficientScalePrecision {
            op: "scale_down",
            available_log_delta,
            required_bits,
        },
    );
}
