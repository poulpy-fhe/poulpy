//! Multiplication and division by a power of two.

use crate::{CKKSCompositionError, CKKSInfos, api::CKKSPow2Ops};
use poulpy_core::layouts::LWEInfos;

use super::helpers::{
    TestContextBackend, TestContextModule, TestScalar, alloc_ct, alloc_scratch, assert_ckks_error, assert_ct_meta,
    assert_decrypt_precision, assert_decrypt_precision_at_log_delta, assert_unary_output_meta, ckks_encrypt, gen_sk,
    test_vector_1, want_div_pow2, want_mul_pow2,
};
use super::parity::helpers::{snapshot, with_scratch};
use poulpy_hal::{
    api::{ReimNew, ScratchOwnedBorrow},
    layouts::{HostBytesBackend, Module, Standard},
};

use crate::{test_suite::CKKSTestParams, test_suite::reference_encoder::ReferenceEncoder};

const SHIFT_BITS: usize = 7;

pub fn test_mul_pow2_aligned<BE, F, E>(params: CKKSTestParams, module: &Module<BE>, host_module: &Module<HostBytesBackend>)
where
    BE: TestContextBackend<Ring = Standard>,
    for<'a> <BE as poulpy_hal::layouts::Backend>::BufRef<'a>: poulpy_hal::layouts::HostDataRef,
    for<'a> <BE as poulpy_hal::layouts::Backend>::BufMut<'a>: poulpy_hal::layouts::HostDataMut,
    Module<BE>: TestContextModule<BE>,
    F: TestScalar,
    E: ReimNew<F>,
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
    let (want_re, want_im) = want_mul_pow2(&re1, &im1, SHIFT_BITS);
    let mut ct_res = alloc_ct(&params, module, params.k);
    with_scratch::<BE, _>(module.ckks_mul_pow2_tmp_bytes(ct_res.max_size()), |s| {
        module.ckks_mul_pow2_into(SHIFT_BITS, &mut ct_res, &ct, s)
    })
    .unwrap();
    assert_unary_output_meta("mul_pow2", &ct_res, params.k, &ct);
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
    BE: TestContextBackend<Ring = Standard>,
    for<'a> <BE as poulpy_hal::layouts::Backend>::BufRef<'a>: poulpy_hal::layouts::HostDataRef,
    for<'a> <BE as poulpy_hal::layouts::Backend>::BufMut<'a>: poulpy_hal::layouts::HostDataMut,
    Module<BE>: TestContextModule<BE>,
    F: TestScalar,
    E: ReimNew<F>,
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
    let (want_re, want_im) = want_mul_pow2(&re1, &im1, SHIFT_BITS);
    let dst_k = params.k - params.base2k - 1;
    let mut ct_res = alloc_ct(&params, module, dst_k);
    with_scratch::<BE, _>(module.ckks_mul_pow2_tmp_bytes(ct_res.max_size()), |s| {
        module.ckks_mul_pow2_into(SHIFT_BITS, &mut ct_res, &ct, s)
    })
    .unwrap();
    assert_unary_output_meta("mul_pow2 smaller_output", &ct_res, dst_k, &ct);
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
    BE: TestContextBackend<Ring = Standard>,
    for<'a> <BE as poulpy_hal::layouts::Backend>::BufRef<'a>: poulpy_hal::layouts::HostDataRef,
    for<'a> <BE as poulpy_hal::layouts::Backend>::BufMut<'a>: poulpy_hal::layouts::HostDataMut,
    Module<BE>: TestContextModule<BE>,
    F: TestScalar,
    E: ReimNew<F>,
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
    let (want_re, want_im) = want_mul_pow2(&re1, &im1, SHIFT_BITS);
    let expected_log_delta = ct.log_delta();
    let expected_log_budget = ct.log_budget();
    with_scratch::<BE, _>(module.ckks_mul_pow2_tmp_bytes(ct.max_size()), |s| {
        module.ckks_mul_pow2_assign(SHIFT_BITS, &mut ct, s)
    })
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
    BE: TestContextBackend<Ring = Standard>,
    for<'a> <BE as poulpy_hal::layouts::Backend>::BufRef<'a>: poulpy_hal::layouts::HostDataRef,
    for<'a> <BE as poulpy_hal::layouts::Backend>::BufMut<'a>: poulpy_hal::layouts::HostDataMut,
    Module<BE>: TestContextModule<BE>,
    F: TestScalar,
    E: ReimNew<F>,
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
    let (want_re, want_im) = want_div_pow2(&re1, &im1, SHIFT_BITS);
    let mut ct_res = alloc_ct(&params, module, params.k);
    with_scratch::<BE, _>(module.ckks_div_pow2_tmp_bytes(ct_res.max_size()), |s| {
        module.ckks_div_pow2_into(SHIFT_BITS, &mut ct_res, &ct, s)
    })
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
    BE: TestContextBackend<Ring = Standard>,
    for<'a> <BE as poulpy_hal::layouts::Backend>::BufRef<'a>: poulpy_hal::layouts::HostDataRef,
    for<'a> <BE as poulpy_hal::layouts::Backend>::BufMut<'a>: poulpy_hal::layouts::HostDataMut,
    Module<BE>: TestContextModule<BE>,
    F: TestScalar,
    E: ReimNew<F>,
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
    let (want_re, want_im) = want_div_pow2(&re1, &im1, SHIFT_BITS);
    let dst_k = params.k - params.base2k - 1;
    let mut ct_res = alloc_ct(&params, module, dst_k);
    with_scratch::<BE, _>(module.ckks_div_pow2_tmp_bytes(ct_res.max_size()), |s| {
        module.ckks_div_pow2_into(SHIFT_BITS, &mut ct_res, &ct, s)
    })
    .unwrap();
    let offset = ct.k().as_usize().saturating_sub(dst_k);
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
    BE: TestContextBackend<Ring = Standard>,
    for<'a> <BE as poulpy_hal::layouts::Backend>::BufRef<'a>: poulpy_hal::layouts::HostDataRef,
    for<'a> <BE as poulpy_hal::layouts::Backend>::BufMut<'a>: poulpy_hal::layouts::HostDataMut,
    Module<BE>: TestContextModule<BE>,
    F: TestScalar,
    E: ReimNew<F>,
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
    let (want_re, want_im) = want_div_pow2(&re1, &im1, SHIFT_BITS);
    // Lossless relabel, mirroring `_into`: the charged bits move under `log_delta`.
    let expected_log_delta = ct.log_delta() + SHIFT_BITS;
    let expected_log_budget = ct.log_budget() - SHIFT_BITS;
    module.ckks_div_pow2_assign(SHIFT_BITS, &mut ct).unwrap();
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

pub fn test_div_pow2_assign_rejects_insufficient_capacity<BE, F, E>(
    params: CKKSTestParams,
    module: &Module<BE>,
    host_module: &Module<HostBytesBackend>,
) where
    BE: TestContextBackend<Ring = Standard>,
    for<'a> <BE as poulpy_hal::layouts::Backend>::BufRef<'a>: poulpy_hal::layouts::HostDataRef,
    for<'a> <BE as poulpy_hal::layouts::Backend>::BufMut<'a>: poulpy_hal::layouts::HostDataMut,
    Module<BE>: TestContextModule<BE>,
    F: TestScalar,
    E: ReimNew<F>,
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
    let available_log_budget = ct.log_budget();
    let required_bits = available_log_budget + 1;
    let before = snapshot::<BE, _>(&ct);
    let err = module.ckks_div_pow2_assign(required_bits, &mut ct).unwrap_err();
    assert_ckks_error(
        "div_pow2_assign_rejects_insufficient_capacity",
        &err,
        CKKSCompositionError::InsufficientHomomorphicCapacity {
            op: "ckks_div_pow2_assign",
            available_log_budget,
            required_bits,
        },
    );
    assert_eq!(snapshot::<BE, _>(&ct), before, "a refused ckks_div_pow2_assign wrote res");
}

/// A destination narrower than the scale of `a` sends the doubling through the aligning shift, whose budget refusal
/// is `InsufficientHomomorphicCapacity` of `ckks_double_into`, before any write, on an empty arena and on the query.
pub fn test_double_into_rejects_insufficient_capacity<BE, F, E>(
    params: crate::test_suite::CKKSTestParams,
    module: &poulpy_hal::layouts::Module<BE>,
    _host_module: &poulpy_hal::layouts::Module<poulpy_hal::layouts::HostBytesBackend>,
) where
    BE: crate::test_suite::helpers::TestContextBackend<Ring = poulpy_hal::layouts::Standard>,
    poulpy_hal::layouts::Module<BE>: crate::test_suite::helpers::TestContextModule<BE>,
    F: crate::test_suite::helpers::TestScalar,
    E: ReimNew<F>,
{
    use crate::test_suite::{
        helpers::assert_ckks_error,
        parity::{
            helpers::{fixture_ciphertext, snapshot, with_scratch},
            layout,
        },
    };
    use crate::{CKKSCompositionError, SlotsKind};
    use poulpy_core::layouts::LWEInfos;
    let b = params.base2k;
    let ct_layout = layout(params, params.rank, 3 * b + 5, b - 1, 0, SlotsKind::Complex);
    use crate::api::CKKSPow2Ops;
    let a = fixture_ciphertext(module, &ct_layout, 41);
    let mut res = fixture_ciphertext(module, &layout(params, params.rank, b - 2, b - 3, 0, SlotsKind::Complex), 31);
    for bytes in [0, module.ckks_mul_pow2_tmp_bytes(res.max_size())] {
        let before = snapshot::<BE, _>(&res);
        let err = with_scratch::<BE, _>(bytes, |s| module.ckks_double_into(&mut res, &a, s)).unwrap_err();
        assert_ckks_error(
            "double_into: destination below the scale",
            &err,
            CKKSCompositionError::InsufficientHomomorphicCapacity {
                op: "ckks_double_into",
                available_log_budget: 2 * b + 6,
                required_bits: 2 * b + 7,
            },
        );
        assert_eq!(snapshot::<BE, _>(&res), before, "a refused doubling wrote dst");
    }
}
