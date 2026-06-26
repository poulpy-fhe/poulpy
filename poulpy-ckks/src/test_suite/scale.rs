//! Value-preserving re-scaling of a ciphertext's working scale.

use crate::{CKKSCompositionError, CKKSInfos, api::CKKSScaleManage, leveled::api::CKKSMulOps};

use super::helpers::{
    TestContextBackend, TestContextModule, TestScalar, alloc_scratch, assert_ckks_error, assert_ct_meta,
    assert_decrypt_precision_at_log_delta, ckks_encrypt, ckks_encrypt_with_prec, gen_sk,
    gen_sk_with_raw, gen_tsk, test_vector_1,
};
use poulpy_hal::{
    api::{NegacyclicFFT, NegacyclicFFTNew, ScratchOwnedBorrow},
    layouts::{HostBytesBackend, Module},
};

use crate::{encoding::reim::Encoder, test_suite::CKKSTestParams};

const SCALE_BITS: usize = 5;

/// scale_down → multiply → scale_up
pub fn test_scale_down_then_multiply<BE, F, E>(
    params: CKKSTestParams,
    module: &Module<BE>,
    host_module: &Module<HostBytesBackend>,
) where
    BE: TestContextBackend,
    for<'a> <BE as poulpy_hal::layouts::Backend>::BufRef<'a>: poulpy_hal::layouts::HostDataRef,
    for<'a> <BE as poulpy_hal::layouts::Backend>::BufMut<'a>: poulpy_hal::layouts::HostDataMut,
    Module<BE>: TestContextModule<BE>,
    F: TestScalar,
    E: NegacyclicFFT<F> + NegacyclicFFTNew<F>,
{
    let m = params.n / 2;
    let encoder = Encoder::<E>::new(m).unwrap();
    let (re1, im1) = test_vector_1::<F>(m);
    let (sk_raw, sk) = gen_sk_with_raw(&params, module, host_module, [0u8; 32]);
    let mut scratch = alloc_scratch(&params, module);
    let tsk = gen_tsk(&params, module, &sk_raw, &mut scratch.borrow());

    let want_re: Vec<F> = (0..m).map(|j| re1[j] * re1[j] - im1[j] * im1[j]).collect();
    let want_im: Vec<F> = (0..m).map(|j| F::from_f64(2.0).unwrap() * re1[j] * im1[j]).collect();

    let ld = params.prec_meta.log_delta;
    for &bits in &[ld / 4, ld / 2 + 1] {
        assert!(bits < ld, "test setup");
        let mut ct = ckks_encrypt_with_prec::<BE, F, E>(
            &params,
            module,
            host_module,
            &encoder,
            &sk,
            params.k,
            &re1,
            &im1,
            params.prec(),
            &mut scratch.borrow(),
        );
        module.ckks_scale_down_assign(&mut ct, bits, &mut scratch.borrow()).unwrap();
        module.ckks_square_assign(&mut ct, &tsk, &mut scratch.borrow()).unwrap();
        module.ckks_scale_up_assign(&mut ct, bits, &mut scratch.borrow()).unwrap();
        assert_decrypt_precision_at_log_delta(
            &format!("scale_down_then_multiply bits={bits}"),
            &params,
            module,
            &encoder,
            &ct,
            &sk,
            &want_re,
            &want_im,
            ld - bits,
            &mut scratch.borrow(),
        );
    }
}

pub fn test_scale_up_assign<BE, F, E>(params: CKKSTestParams, module: &Module<BE>, host_module: &Module<HostBytesBackend>)
where
    BE: TestContextBackend,
    for<'a> <BE as poulpy_hal::layouts::Backend>::BufRef<'a>: poulpy_hal::layouts::HostDataRef,
    for<'a> <BE as poulpy_hal::layouts::Backend>::BufMut<'a>: poulpy_hal::layouts::HostDataMut,
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
    for<'a> <BE as poulpy_hal::layouts::Backend>::BufRef<'a>: poulpy_hal::layouts::HostDataRef,
    for<'a> <BE as poulpy_hal::layouts::Backend>::BufMut<'a>: poulpy_hal::layouts::HostDataMut,
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
    // scale_down then scale_up restores the metadata and flushes the overflow
    // (the reverse order would leave the scale_down overflow in the head-room).
    module
        .ckks_scale_down_assign(&mut ct, SCALE_BITS, &mut scratch.borrow())
        .unwrap();
    module
        .ckks_scale_up_assign(&mut ct, SCALE_BITS, &mut scratch.borrow())
        .unwrap();
    assert_ct_meta("scale_round_trip", &ct, original_log_delta, original_log_budget);
    // The round trip re-zeros the `SCALE_BITS` low torus bits (right-shift then
    // left-shift), so the recovered precision sits ~1 bit below a fresh
    // encryption on an inexact backend (FFT64); allow that margin while still
    // proving the value survived. Decoding is still at the full `original_log_delta`.
    assert_decrypt_precision_at_log_delta(
        "scale_round_trip",
        &params,
        module,
        &encoder,
        &ct,
        &sk,
        &re1,
        &im1,
        original_log_delta - 1,
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
