//! Multiplication tests: ct × ct and ct² (square).
//!
//! # Test inventory
//!
//! ## ct × ct multiplication out of place (`GLWE<_, CKKS>::mul`)
//!
//! | Function | Path exercised |
//! |----------|----------------|
//! | [`test_mul_ct_aligned`] | both inputs at same `log_budget()` |
//! | [`test_mul_ct_delta_a_lt_b`] | `a.log_budget() < b.log_budget()` |
//! | [`test_mul_ct_delta_a_gt_b`] | `a.log_budget() > b.log_budget()` |
//! | [`test_mul_ct_larger_budget_smaller_scale`] | the operand with the larger `log_budget()` has the smaller `log_delta()`, both orders, also `_assign`, `_prepared_assign` and the dot product |
//! | [`test_mul_ct_smaller_output`] | output has smaller `max_k()` than inputs |
//! | [`test_mul_ct_smaller_output_exact_scratch`] | same, on a scratch arena of exactly `ckks_mul_tmp_bytes` |
//!
//! ## ct x ct inplace ct-ct (`GLWE<_, CKKS>::mul_assign`)
//!
//! | Function | Path exercised |
//! |----------|----------------|
//! | [`test_mul_ct_assign_aligned`] | `self.log_budget() == a.log_budget()` |
//! | [`test_mul_ct_assign_self_lt`] | `self.log_budget() < a.log_budget()` → a shifted to align with self |
//! | [`test_mul_ct_assign_self_gt`] | `self.log_budget() > a.log_budget()` → self shifted to align with a |
//!
//! ## ct² squaring out of place (`GLWE<_, CKKS>::square`)
//!
//! | Function | Path exercised |
//! |----------|----------------|
//! | [`test_square_aligned`] | square at default precision |
//! | [`test_square_rescaled_input`] | square after a rescale (reduced `log_budget()`) |
//! | [`test_square_smaller_output`] | square into smaller output buffer |
//!
//! ## ct² squaring inplace (`GLWE<_, CKKS>::square_assign`)
//!
//! | Function | Path exercised |
//! |----------|----------------|
//! | [`test_square_assign`] | square at default precision |
//!
//! ## ct x pt out of place (`GLWE<_, CKKS>::mul_pt_vec`)
//!
//! | Function | Path exercised |
//! |----------|----------------|
//! | [`test_mul_pt_vec_into_aligned`] | input and output at same `log_budget()` |
//! | [`test_mul_pt_vec_into_smaller_output`] | output at smaller `log_budget()` |
//! | [`test_mul_pt_scale_above_ct`] | `pt.log_delta() > a.log_delta()`, also `_assign` and the constant forms |
//!
//! ## ct x pt inplace (`GLWE<_, CKKS>::mul_pt_vec_assign`)
//!
//! | Function | Path exercised |
//! |----------|----------------|
//! | [`test_mul_pt_vec_assign`] | - |
use crate::{
    CKKSCompositionError, CKKSInfos,
    api::{CKKSDotProductOps, CKKSMulOps},
    layouts::CKKSModuleAlloc,
};
use poulpy_core::layouts::{GLWEToBackendMut, LWEInfos};

use poulpy_hal::{
    AlignedBuf,
    api::{NegacyclicFFT, NegacyclicFFTNew, ScratchOwnedAlloc, ScratchOwnedBorrow},
    layouts::{DataView, DataViewMut, HostBytesBackend, Module, PrepareHint, ScratchArena, ScratchOwned, Standard},
};

use super::helpers::{
    MUL_CONST, PT_PREC, TestContextBackend, TestContextModule, TestScalar, TestVector, alloc_ct, alloc_scratch,
    assert_ckks_error, assert_decrypt_precision, assert_decrypt_precision_at_log_delta, assert_mul_ct_output_meta,
    assert_mul_pt_output_meta, ckks_encrypt, ckks_encrypt_with_prec, ckks_pt_cst_full, ckks_snapshot, encode_and_upload_pt,
    gen_sk_with_raw, gen_tsk, precision_at, quantize, quantized_const, quantized_vector, want_mul, want_square,
};

use crate::{SetCKKSInfos, test_suite::CKKSTestParams, test_suite::reference_encoder::ReferenceEncoder};

const DELTA_LOG_DELTA: usize = 8;

// ─── ct × ct out-of-place ───────────────────────────────────────────────────

pub fn test_mul_ct_aligned<BE, F, E>(params: CKKSTestParams, module: &Module<BE>, host_module: &Module<HostBytesBackend>)
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
    let (re1, im1) = super::helpers::test_vector_1::<F>(m);
    let (re2, im2) = super::helpers::test_vector_2::<F>(m);
    let (sk_raw, sk) = gen_sk_with_raw(&params, module, host_module, [0u8; 32]);
    let mut scratch = alloc_scratch(&params, module);
    let tsk = gen_tsk(&params, module, &sk_raw, &mut scratch.borrow());

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
    let (want_re, want_im) = want_mul(&re1, &im1, &re2, &im2);
    let mut ct_res = alloc_ct(&params, module, params.k);
    module
        .ckks_mul_into(&mut ct_res, &ct1, &ct2, &tsk, &mut scratch.borrow())
        .unwrap();
    assert_mul_ct_output_meta("mul_ct_aligned", &ct_res, params.k, &ct1, &ct2);
    let metadata = ct1.noise();
    assert!(metadata.is_some());
    assert_eq!(ct_res.noise(), None);
    let prepared = module
        .ckks_prepare_right(&ct2, PrepareHint::OneShot, &mut scratch.borrow())
        .unwrap();
    let mut prepared_result = ct1.clone();
    module
        .ckks_mul_prepared_assign(&mut prepared_result, &prepared, &tsk, &mut scratch.borrow())
        .unwrap();
    assert_eq!(prepared_result.noise(), None);
    assert_eq!(ckks_snapshot(&prepared_result), ckks_snapshot(&ct_res));

    let mut unknown = ct2.clone();
    GLWEToBackendMut::<BE>::set_noise(&mut unknown, None);
    let prepared = module
        .ckks_prepare_right(&unknown, PrepareHint::Reuse, &mut scratch.borrow())
        .unwrap();
    let mut prepared_result = ct1.clone();
    module
        .ckks_mul_prepared_assign(&mut prepared_result, &prepared, &tsk, &mut scratch.borrow())
        .unwrap();
    assert_eq!(prepared_result.noise(), None);
    assert_decrypt_precision(
        "mul_ct_aligned",
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

pub fn test_mul_ct_delta_a_lt_b<BE, F, E>(params: CKKSTestParams, module: &Module<BE>, host_module: &Module<HostBytesBackend>)
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
    let (re1, im1) = super::helpers::test_vector_1::<F>(m);
    let (re2, im2) = super::helpers::test_vector_2::<F>(m);
    let (sk_raw, sk) = gen_sk_with_raw(&params, module, host_module, [0u8; 32]);
    let mut scratch = alloc_scratch(&params, module);
    let tsk = gen_tsk(&params, module, &sk_raw, &mut scratch.borrow());

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
    let (want_re, want_im) = want_mul(&re1, &im1, &re2, &im2);
    let mut ct_res = alloc_ct(&params, module, params.k);
    module
        .ckks_mul_into(&mut ct_res, &ct1, &ct2, &tsk, &mut scratch.borrow())
        .unwrap();
    assert_mul_ct_output_meta("mul_ct a_lt_b", &ct_res, params.k, &ct1, &ct2);
    assert_decrypt_precision(
        "mul_ct a_lt_b",
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

pub fn test_mul_ct_delta_a_gt_b<BE, F, E>(params: CKKSTestParams, module: &Module<BE>, host_module: &Module<HostBytesBackend>)
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
    let (re1, im1) = super::helpers::test_vector_1::<F>(m);
    let (re2, im2) = super::helpers::test_vector_2::<F>(m);
    let (sk_raw, sk) = gen_sk_with_raw(&params, module, host_module, [0u8; 32]);
    let mut scratch = alloc_scratch(&params, module);
    let tsk = gen_tsk(&params, module, &sk_raw, &mut scratch.borrow());

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
    let (want_re, want_im) = want_mul(&re1, &im1, &re2, &im2);
    let mut ct_res = alloc_ct(&params, module, params.k);
    module
        .ckks_mul_into(&mut ct_res, &ct1, &ct2, &tsk, &mut scratch.borrow())
        .unwrap();
    assert_mul_ct_output_meta("mul_ct a_gt_b", &ct_res, params.k, &ct1, &ct2);
    assert_decrypt_precision(
        "mul_ct a_gt_b",
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

pub fn test_mul_ct_delta_log_delta<BE, F, E>(params: CKKSTestParams, module: &Module<BE>, host_module: &Module<HostBytesBackend>)
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
    let (sk_raw, sk) = gen_sk_with_raw(&params, module, host_module, [0u8; 32]);
    let mut scratch = alloc_scratch(&params, module);
    let tsk = gen_tsk(&params, module, &sk_raw, &mut scratch.borrow());

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
    let (want_re, want_im) = want_mul(&a_re, &a_im, &b_re, &b_im);
    let mut ct_res = alloc_ct(&params, module, params.k);
    module
        .ckks_mul_into(&mut ct_res, &ct1, &ct2, &tsk, &mut scratch.borrow())
        .unwrap();
    assert_mul_ct_output_meta("mul_ct delta_log_delta", &ct_res, params.k, &ct1, &ct2);
    assert_decrypt_precision_at_log_delta(
        "mul_ct delta_log_delta",
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

/// Operands at the same `k` where the one with the larger budget has the smaller scale, in both orders,
/// through `ckks_mul_into`, `ckks_mul_assign`, `ckks_mul_prepared_assign` and `ckks_dot_product_ct`.
pub fn test_mul_ct_larger_budget_smaller_scale<BE, F, E>(
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
    let (sk_raw, sk) = gen_sk_with_raw(&params, module, host_module, [0u8; 32]);
    let mut scratch = alloc_scratch(&params, module);
    let tsk = gen_tsk(&params, module, &sk_raw, &mut scratch.borrow());

    let low_log_delta = params.prec().log_delta() - DELTA_LOG_DELTA;
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
        params.k,
        &b_re,
        &b_im,
        precision_at(&params, low_log_delta),
        &mut scratch.borrow(),
    );
    assert_eq!(ct2.log_budget(), ct1.log_budget() + DELTA_LOG_DELTA);
    let (want_re, want_im) = want_mul(&a_re, &a_im, &b_re, &b_im);

    for (order, a, b) in [("a*b", &ct1, &ct2), ("b*a", &ct2, &ct1)] {
        let label = format!("mul_ct larger_budget_smaller_scale {order}");
        let mut ct_res = alloc_ct(&params, module, params.k);
        module.ckks_mul_into(&mut ct_res, a, b, &tsk, &mut scratch.borrow()).unwrap();
        assert_mul_ct_output_meta(&label, &ct_res, params.k, a, b);

        let mut assigned = a.clone();
        module.ckks_mul_assign(&mut assigned, b, &tsk, &mut scratch.borrow()).unwrap();

        let prepared = module
            .ckks_prepare_right(b, PrepareHint::OneShot, &mut scratch.borrow())
            .unwrap();
        let mut prepared_result = a.clone();
        module
            .ckks_mul_prepared_assign(&mut prepared_result, &prepared, &tsk, &mut scratch.borrow())
            .unwrap();

        for (variant, ct) in [("into", &ct_res), ("assign", &assigned), ("prepared", &prepared_result)] {
            assert_decrypt_precision_at_log_delta(
                &format!("{label} {variant}"),
                &params,
                module,
                &encoder,
                ct,
                &sk,
                &want_re,
                &want_im,
                low_log_delta,
                &mut scratch.borrow(),
            );
        }
    }

    // The dot product sums one `ckks_mul_into` per term: `a*b + b*a`.
    let mut dot = alloc_ct(&params, module, params.k);
    module
        .ckks_dot_product_ct(&mut dot, &[&ct1, &ct2], &[&ct2, &ct1], &tsk, &mut scratch.borrow())
        .unwrap();
    let twice = |v: &[F]| v.iter().map(|&x| x + x).collect::<Vec<F>>();
    assert_decrypt_precision_at_log_delta(
        "mul_ct larger_budget_smaller_scale dot_product",
        &params,
        module,
        &encoder,
        &dot,
        &sk,
        &twice(&want_re),
        &twice(&want_im),
        low_log_delta,
        &mut scratch.borrow(),
    );
}

pub fn test_mul_ct_smaller_output<BE, F, E>(params: CKKSTestParams, module: &Module<BE>, host_module: &Module<HostBytesBackend>)
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
    let (re1, im1) = super::helpers::test_vector_1::<F>(m);
    let (re2, im2) = super::helpers::test_vector_2::<F>(m);
    let (sk_raw, sk) = gen_sk_with_raw(&params, module, host_module, [0u8; 32]);
    let mut scratch = alloc_scratch(&params, module);
    let tsk = gen_tsk(&params, module, &sk_raw, &mut scratch.borrow());

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
    let (want_re, want_im) = want_mul(&re1, &im1, &re2, &im2);
    let dst_k = params.k - params.base2k - 1;
    let mut ct_res = alloc_ct(&params, module, dst_k);
    module
        .ckks_mul_into(&mut ct_res, &ct1, &ct2, &tsk, &mut scratch.borrow())
        .unwrap();
    assert_mul_ct_output_meta("mul_ct smaller_output", &ct_res, dst_k, &ct1, &ct2);
    assert_decrypt_precision(
        "mul_ct smaller_output",
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

/// Wide operands into a narrower destination, on a scratch arena of exactly
/// `ckks_mul_tmp_bytes(res, a, b, tsk)`.
///
/// Regression test: the sizing must account for the operands' widths — the op
/// carves its tensor intermediate at `max(a.max_k, b.max_k)`, which exceeds
/// the destination's width here. Sizing from `res` alone under-allocated and
/// blew the scratch-arena assert on this documented-legal call.
pub fn test_mul_ct_smaller_output_exact_scratch<BE, F, E>(
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
    let (re1, im1) = super::helpers::test_vector_1::<F>(m);
    let (re2, im2) = super::helpers::test_vector_2::<F>(m);
    let (sk_raw, sk) = gen_sk_with_raw(&params, module, host_module, [0u8; 32]);
    let mut scratch = alloc_scratch(&params, module);
    let tsk = gen_tsk(&params, module, &sk_raw, &mut scratch.borrow());

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
    let (want_re, want_im) = want_mul(&re1, &im1, &re2, &im2);
    let dst_k = params.k - params.base2k - 1;
    let mut ct_res = alloc_ct(&params, module, dst_k);
    let exact_bytes = module.ckks_mul_tmp_bytes(&ct_res, &ct1, &ct2, &tsk);
    let mut mul_scratch = ScratchOwned::<BE>::alloc(exact_bytes);
    module
        .ckks_mul_into(&mut ct_res, &ct1, &ct2, &tsk, &mut mul_scratch.borrow())
        .unwrap();
    assert_mul_ct_output_meta("mul_ct smaller_output_exact_scratch", &ct_res, dst_k, &ct1, &ct2);
    assert_decrypt_precision(
        "mul_ct smaller_output_exact_scratch",
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

// ─── ct × ct in-place ───────────────────────────────────────────────────────

pub fn test_mul_ct_assign_aligned<BE, F, E>(params: CKKSTestParams, module: &Module<BE>, host_module: &Module<HostBytesBackend>)
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
    let (re1, im1) = super::helpers::test_vector_1::<F>(m);
    let (re2, im2) = super::helpers::test_vector_2::<F>(m);
    let (sk_raw, sk) = gen_sk_with_raw(&params, module, host_module, [0u8; 32]);
    let mut scratch = alloc_scratch(&params, module);
    let tsk = gen_tsk(&params, module, &sk_raw, &mut scratch.borrow());

    let mut ct_res = ckks_encrypt(
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
    let ct1 = ckks_encrypt(
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
    let (want_re, want_im) = want_mul(&re1, &im1, &re2, &im2);
    let ct_res_meta = ckks_snapshot(&ct_res);
    module
        .ckks_mul_assign(&mut ct_res, &ct1, &tsk, &mut scratch.borrow())
        .unwrap();
    assert_mul_ct_output_meta(
        "mul_ct_assign_aligned",
        &ct_res,
        ct_res_meta.k().as_usize(),
        &ct_res_meta,
        &ct1,
    );
    assert_decrypt_precision(
        "mul_ct_assign_aligned",
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

pub fn test_mul_ct_assign_self_lt<BE, F, E>(params: CKKSTestParams, module: &Module<BE>, host_module: &Module<HostBytesBackend>)
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
    let (re1, im1) = super::helpers::test_vector_1::<F>(m);
    let (re2, im2) = super::helpers::test_vector_2::<F>(m);
    let (sk_raw, sk) = gen_sk_with_raw(&params, module, host_module, [0u8; 32]);
    let mut scratch = alloc_scratch(&params, module);
    let tsk = gen_tsk(&params, module, &sk_raw, &mut scratch.borrow());

    let mut ct_res = ckks_encrypt(
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
    let ct1 = ckks_encrypt(
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
    let (want_re, want_im) = want_mul(&re1, &im1, &re2, &im2);
    let ct_res_meta = ckks_snapshot(&ct_res);
    module
        .ckks_mul_assign(&mut ct_res, &ct1, &tsk, &mut scratch.borrow())
        .unwrap();
    assert_mul_ct_output_meta(
        "mul_ct_assign_self_lt",
        &ct_res,
        ct_res_meta.k().as_usize(),
        &ct_res_meta,
        &ct1,
    );
    assert_decrypt_precision(
        "mul_ct_assign_self_lt",
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

pub fn test_mul_ct_assign_self_gt<BE, F, E>(params: CKKSTestParams, module: &Module<BE>, host_module: &Module<HostBytesBackend>)
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
    let (re1, im1) = super::helpers::test_vector_1::<F>(m);
    let (re2, im2) = super::helpers::test_vector_2::<F>(m);
    let (sk_raw, sk) = gen_sk_with_raw(&params, module, host_module, [0u8; 32]);
    let mut scratch = alloc_scratch(&params, module);
    let tsk = gen_tsk(&params, module, &sk_raw, &mut scratch.borrow());

    let mut ct_res = ckks_encrypt(
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
    let ct1 = ckks_encrypt(
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
    let (want_re, want_im) = want_mul(&re1, &im1, &re2, &im2);
    let ct_res_meta = ckks_snapshot(&ct_res);
    module
        .ckks_mul_assign(&mut ct_res, &ct1, &tsk, &mut scratch.borrow())
        .unwrap();
    assert_mul_ct_output_meta(
        "mul_ct_assign_self_gt",
        &ct_res,
        ct_res_meta.k().as_usize(),
        &ct_res_meta,
        &ct1,
    );
    assert_decrypt_precision(
        "mul_ct_assign_self_gt",
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

// ─── ct² squaring out of place ───────────────────────────────────────────────

pub fn test_square_aligned<BE, F, E>(params: CKKSTestParams, module: &Module<BE>, host_module: &Module<HostBytesBackend>)
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
    let (re1, im1) = super::helpers::test_vector_1::<F>(m);
    let (sk_raw, sk) = gen_sk_with_raw(&params, module, host_module, [0u8; 32]);
    let mut scratch = alloc_scratch(&params, module);
    let tsk = gen_tsk(&params, module, &sk_raw, &mut scratch.borrow());

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
    let (want_re, want_im) = want_square(&re1, &im1);
    let mut ct_res = alloc_ct(&params, module, params.k);
    module
        .ckks_square_into(&mut ct_res, &ct, &tsk, &mut scratch.borrow())
        .unwrap();
    assert_mul_ct_output_meta("square_aligned", &ct_res, params.k, &ct, &ct);
    assert_decrypt_precision(
        "square_aligned",
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

pub fn test_square_rescaled_input<BE, F, E>(params: CKKSTestParams, module: &Module<BE>, host_module: &Module<HostBytesBackend>)
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
    let (re1, im1) = super::helpers::test_vector_1::<F>(m);
    let (sk_raw, sk) = gen_sk_with_raw(&params, module, host_module, [0u8; 32]);
    let mut scratch = alloc_scratch(&params, module);
    let tsk = gen_tsk(&params, module, &sk_raw, &mut scratch.borrow());

    let ct = ckks_encrypt(
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
    let (want_re, want_im) = want_square(&re1, &im1);
    let mut ct_res = alloc_ct(&params, module, params.k);
    module
        .ckks_square_into(&mut ct_res, &ct, &tsk, &mut scratch.borrow())
        .unwrap();
    assert_mul_ct_output_meta("square_rescaled_input", &ct_res, params.k, &ct, &ct);
    assert_decrypt_precision(
        "square_rescaled_input",
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

pub fn test_square_smaller_output<BE, F, E>(params: CKKSTestParams, module: &Module<BE>, host_module: &Module<HostBytesBackend>)
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
    let (re1, im1) = super::helpers::test_vector_1::<F>(m);
    let (sk_raw, sk) = gen_sk_with_raw(&params, module, host_module, [0u8; 32]);
    let mut scratch = alloc_scratch(&params, module);
    let tsk = gen_tsk(&params, module, &sk_raw, &mut scratch.borrow());

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
    let (want_re, want_im) = want_square(&re1, &im1);
    let dst_k = params.k - params.base2k - 1;
    let mut ct_res = alloc_ct(&params, module, dst_k);
    module
        .ckks_square_into(&mut ct_res, &ct, &tsk, &mut scratch.borrow())
        .unwrap();
    assert_mul_ct_output_meta("square_smaller_output", &ct_res, dst_k, &ct, &ct);
    assert_decrypt_precision(
        "square_smaller_output",
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

// ─── ct² squaring in-place ───────────────────────────────────────────────────

pub fn test_square_assign<BE, F, E>(params: CKKSTestParams, module: &Module<BE>, host_module: &Module<HostBytesBackend>)
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
    let (re1, im1) = super::helpers::test_vector_1::<F>(m);
    let (sk_raw, sk) = gen_sk_with_raw(&params, module, host_module, [0u8; 32]);
    let mut scratch = alloc_scratch(&params, module);
    let tsk = gen_tsk(&params, module, &sk_raw, &mut scratch.borrow());

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
    let (want_re, want_im) = want_square(&re1, &im1);
    let ct_in_meta = ckks_snapshot(&ct);
    module.ckks_square_assign(&mut ct, &tsk, &mut scratch.borrow()).unwrap();
    assert_mul_ct_output_meta("square_assign", &ct, ct_in_meta.k().as_usize(), &ct_in_meta, &ct_in_meta);
    assert_decrypt_precision(
        "square_assign",
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

// ─── ct × pt out of place ───────────────────────────────────────────────────

pub fn test_mul_pt_vec_into_aligned<BE, F, E>(params: CKKSTestParams, module: &Module<BE>, host_module: &Module<HostBytesBackend>)
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
    let (re1, im1) = super::helpers::test_vector_1::<F>(m);
    let (re2, im2) = super::helpers::test_vector_2::<F>(m);
    let sk = super::helpers::gen_sk(&params, module, host_module, [0u8; 32]);
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
    let pt = encode_and_upload_pt(host_module, module, &encoder, params.base2k.into(), params.prec(), &re2, &im2);
    let (want_re, want_im) = want_mul(
        &re1,
        &im1,
        &quantize(&re2, params.prec().log_delta()),
        &quantize(&im2, params.prec().log_delta()),
    );
    let mut ct_res = alloc_ct(&params, module, params.k);
    module
        .ckks_mul_pt_vec_into(&mut ct_res, &ct, &pt, &mut scratch.borrow())
        .unwrap();
    assert_mul_pt_output_meta("mul_pt_vec_into_aligned", &ct_res, params.k, &ct, &pt);
    assert_decrypt_precision(
        "mul_pt_vec_into_aligned",
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

pub fn test_mul_pt_vec_into_delta_log_delta<BE, F, E>(
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
    let (re2, im2) = super::helpers::test_vector_2::<F>(m);
    let sk = super::helpers::gen_sk(&params, module, host_module, [0u8; 32]);
    let mut scratch = alloc_scratch(&params, module);

    let (a_re, a_im) = quantized_vector(host_module, &encoder, &params, TestVector::First, params.prec().log_delta());
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
    let pt = encode_and_upload_pt(host_module, module, &encoder, params.base2k.into(), PT_PREC, &re2, &im2);
    let (want_re, want_im) = want_mul(
        &a_re,
        &a_im,
        &quantize(&re2, PT_PREC.log_delta()),
        &quantize(&im2, PT_PREC.log_delta()),
    );
    let mut ct_res = alloc_ct(&params, module, params.k);
    module
        .ckks_mul_pt_vec_into(&mut ct_res, &ct, &pt, &mut scratch.borrow())
        .unwrap();
    assert_mul_pt_output_meta("mul_pt_vec_into_delta_log_delta", &ct_res, params.k, &ct, &pt);
    assert_decrypt_precision_at_log_delta(
        "mul_pt_vec_into_delta_log_delta",
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

pub fn test_mul_pt_vec_into_smaller_output<BE, F, E>(
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
    let (re1, im1) = super::helpers::test_vector_1::<F>(m);
    let (re2, im2) = super::helpers::test_vector_2::<F>(m);
    let sk = super::helpers::gen_sk(&params, module, host_module, [0u8; 32]);
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
    let pt = encode_and_upload_pt(host_module, module, &encoder, params.base2k.into(), params.prec(), &re2, &im2);
    let (want_re, want_im) = want_mul(
        &re1,
        &im1,
        &quantize(&re2, params.prec().log_delta()),
        &quantize(&im2, params.prec().log_delta()),
    );
    let dst_k = params.k - params.base2k - 1;
    let mut ct_res = alloc_ct(&params, module, dst_k);
    module
        .ckks_mul_pt_vec_into(&mut ct_res, &ct, &pt, &mut scratch.borrow())
        .unwrap();
    assert_mul_pt_output_meta("mul_pt_vec_into_smaller_output", &ct_res, dst_k, &ct, &pt);
    assert_decrypt_precision(
        "mul_pt_vec_into_smaller_output",
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

// ─── ct × pt in-place ───────────────────────────────────────────────────────

pub fn test_mul_pt_vec_assign<BE, F, E>(params: CKKSTestParams, module: &Module<BE>, host_module: &Module<HostBytesBackend>)
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
    let (re1, im1) = super::helpers::test_vector_1::<F>(m);
    let (re2, im2) = super::helpers::test_vector_2::<F>(m);
    let sk = super::helpers::gen_sk(&params, module, host_module, [0u8; 32]);
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
    let (want_re, want_im) = want_mul(
        &re1,
        &im1,
        &quantize(&re2, params.prec().log_delta()),
        &quantize(&im2, params.prec().log_delta()),
    );
    let ct_meta = ckks_snapshot(&ct);
    module.ckks_mul_pt_vec_assign(&mut ct, &pt, &mut scratch.borrow()).unwrap();
    assert_mul_pt_output_meta("mul_pt_vec_assign", &ct, ct_meta.k().as_usize(), &ct_meta, &pt);
    assert_decrypt_precision(
        "mul_pt_vec_assign",
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

// ─── ct × const ─────────────────────────────────────────────────────────────

pub fn test_mul_pt_const_into_aligned<BE, F, E>(
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
    let (re1, im1) = super::helpers::test_vector_1::<F>(m);
    let sk = super::helpers::gen_sk(&params, module, host_module, [0u8; 32]);
    let mut scratch = alloc_scratch(&params, module);

    let const_re_f64 = MUL_CONST.0;
    let (const_re, const_im) = quantized_const::<F>(const_re_f64, 0.0, PT_PREC.log_delta());
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
    let (want_re, want_im) = super::helpers::want_mul_const(&re1, &im1, const_re, const_im);
    let mut ct_res = alloc_ct(&params, module, params.k);
    let cst = ckks_pt_cst_full::<BE, F>(
        host_module,
        module,
        params.base2k.into(),
        PT_PREC,
        m,
        Some(const_re_f64),
        None,
    );
    module
        .ckks_mul_pt_const_into(&mut ct_res, &ct, &cst, 0, &mut scratch.borrow())
        .unwrap();
    assert_mul_pt_output_meta("mul_const_into_aligned", &ct_res, params.k, &ct, &PT_PREC);
    assert_decrypt_precision(
        "mul_const_into_aligned",
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

pub fn test_mul_pt_const_assign<BE, F, E>(params: CKKSTestParams, module: &Module<BE>, host_module: &Module<HostBytesBackend>)
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
    let (re1, im1) = super::helpers::test_vector_1::<F>(m);
    let sk = super::helpers::gen_sk(&params, module, host_module, [0u8; 32]);
    let mut scratch = alloc_scratch(&params, module);

    let const_re_f64 = MUL_CONST.0;
    let (const_re, const_im) = quantized_const::<F>(const_re_f64, 0.0, PT_PREC.log_delta());
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
    let (want_re, want_im) = super::helpers::want_mul_const(&re1, &im1, const_re, const_im);
    let ct_meta = ckks_snapshot(&ct);
    let cst = ckks_pt_cst_full::<BE, F>(
        host_module,
        module,
        params.base2k.into(),
        PT_PREC,
        m,
        Some(const_re_f64),
        None,
    );
    module
        .ckks_mul_pt_const_assign(&mut ct, &cst, 0, &mut scratch.borrow())
        .unwrap();
    assert_mul_pt_output_meta("mul_const_assign", &ct, ct_meta.k().as_usize(), &ct_meta, &PT_PREC);
    assert_decrypt_precision(
        "mul_const_assign",
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

pub fn test_mul_pt_const_into_delta_log_delta<BE, F, E>(
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
    let sk = super::helpers::gen_sk(&params, module, host_module, [0u8; 32]);
    let mut scratch = alloc_scratch(&params, module);

    let const_re_f64 = MUL_CONST.0;
    let (const_re, const_im) = quantized_const::<F>(const_re_f64, 0.0, PT_PREC.log_delta());
    let (a_re, a_im) = quantized_vector(host_module, &encoder, &params, TestVector::First, params.prec().log_delta());
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
    let (want_re, want_im) = super::helpers::want_mul_const(&a_re, &a_im, const_re, const_im);
    let mut ct_res = alloc_ct(&params, module, params.k);
    let cst = ckks_pt_cst_full::<BE, F>(
        host_module,
        module,
        params.base2k.into(),
        PT_PREC,
        m,
        Some(const_re_f64),
        None,
    );
    module
        .ckks_mul_pt_const_into(&mut ct_res, &ct, &cst, 0, &mut scratch.borrow())
        .unwrap();
    assert_mul_pt_output_meta("mul_const_into_delta_log_delta", &ct_res, params.k, &ct, &PT_PREC);
    assert_decrypt_precision_at_log_delta(
        "mul_const_into_delta_log_delta",
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

/// A plaintext at a larger scale than the ciphertext's (`pt.log_delta() > a.log_delta()`), as a vector
/// and as a constant, through `_into` and `_assign`: the product keeps `a.log_delta()` and spends
/// `pt.log_delta()` bits of budget.
pub fn test_mul_pt_scale_above_ct<BE, F, E>(params: CKKSTestParams, module: &Module<BE>, host_module: &Module<HostBytesBackend>)
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
    let (re1, im1) = super::helpers::test_vector_1::<F>(m);
    let (re2, im2) = super::helpers::test_vector_2::<F>(m);
    let sk = super::helpers::gen_sk(&params, module, host_module, [0u8; 32]);
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
    let pt_prec = precision_at(&params, params.prec().log_delta() + DELTA_LOG_DELTA);
    let pt = encode_and_upload_pt(host_module, module, &encoder, params.base2k.into(), pt_prec, &re2, &im2);
    let cst = ckks_pt_cst_full::<BE, F>(host_module, module, params.base2k.into(), pt_prec, m, Some(MUL_CONST.0), None);
    let want_vec = want_mul(
        &re1,
        &im1,
        &quantize(&re2, pt_prec.log_delta()),
        &quantize(&im2, pt_prec.log_delta()),
    );
    let (const_re, const_im) = quantized_const::<F>(MUL_CONST.0, 0.0, pt_prec.log_delta());
    let want_const = super::helpers::want_mul_const(&re1, &im1, const_re, const_im);

    let mut vec_into = alloc_ct(&params, module, params.k);
    module
        .ckks_mul_pt_vec_into(&mut vec_into, &ct, &pt, &mut scratch.borrow())
        .unwrap();
    let mut vec_assign = ct.clone();
    module
        .ckks_mul_pt_vec_assign(&mut vec_assign, &pt, &mut scratch.borrow())
        .unwrap();
    let mut const_into = alloc_ct(&params, module, params.k);
    module
        .ckks_mul_pt_const_into(&mut const_into, &ct, &cst, 0, &mut scratch.borrow())
        .unwrap();
    let mut const_assign = ct.clone();
    module
        .ckks_mul_pt_const_assign(&mut const_assign, &cst, 0, &mut scratch.borrow())
        .unwrap();

    for (variant, res, b, (want_re, want_im)) in [
        ("vec_into", &vec_into, &pt, &want_vec),
        ("vec_assign", &vec_assign, &pt, &want_vec),
        ("const_into", &const_into, &cst, &want_const),
        ("const_assign", &const_assign, &cst, &want_const),
    ] {
        let label = format!("mul_pt_scale_above_ct {variant}");
        // Every destination asks for `params.k`: `_into` allocates it there, `_assign` writes `ct`, encrypted there.
        assert_mul_pt_output_meta(&label, res, params.k, &ct, b);
        assert_decrypt_precision(
            &label,
            &params,
            module,
            &encoder,
            res,
            &sk,
            want_re,
            want_im,
            &mut scratch.borrow(),
        );
    }
}

// ─── error test ──────────────────────────────────────────────────────────────

/// `ckks_mul_prepared_assign` must reject a prepared operand built under a
/// different layout (here: a different `base2k`) with a typed error, before
/// touching the destination.
pub fn test_mul_prepared_layout_mismatch_error<BE, F, E>(
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
    let (re1, im1) = super::helpers::test_vector_1::<F>(m);
    let (sk_raw, sk) = gen_sk_with_raw(&params, module, host_module, [0u8; 32]);
    let mut scratch = alloc_scratch(&params, module);
    let tsk = gen_tsk(&params, module, &sk_raw, &mut scratch.borrow());

    let mut dst = ckks_encrypt(
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
    // Prepared from an operand with a different limb radix.
    let mismatched_base2k = params.base2k / 2;
    let mut other = module.ckks_ciphertext_alloc(mismatched_base2k.into(), params.k.into());
    other.set_meta(params.prec().meta);
    let prepared = module
        .ckks_prepare_right(&other, PrepareHint::Reuse, &mut scratch.borrow())
        .unwrap();
    let err = module
        .ckks_mul_prepared_assign(&mut dst, &prepared, &tsk, &mut scratch.borrow())
        .unwrap_err();
    assert_ckks_error(
        "mul_prepared_layout_mismatch",
        &err,
        CKKSCompositionError::PreparedOperandLayoutMismatch {
            op: "mul_prepared",
            dst_n: params.n,
            dst_base2k: params.base2k,
            dst_rank: 1,
            prep_n: params.n,
            prep_base2k: mismatched_base2k,
            prep_rank: 1,
        },
    );
}

/// A prepared operand allocated from a ciphertext's layout, under either hint, on either side: its bytes query is its
/// buffer's length, preparing into it gives the allocating preparation's bytes, preparing into it again after poisoning
/// it keeps its buffer and gives them again, and a source of another layout is rejected before any write.
pub fn test_prepare_into<BE, F, E>(params: CKKSTestParams, module: &Module<BE>, host_module: &Module<HostBytesBackend>)
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
    let (re, im) = super::helpers::test_vector_1::<F>(m);
    let (_, sk) = gen_sk_with_raw(&params, module, host_module, [0u8; 32]);
    let mut scratch = alloc_scratch(&params, module);
    let ct = ckks_encrypt(
        &params,
        module,
        host_module,
        &encoder,
        &sk,
        params.k,
        &re,
        &im,
        &mut scratch.borrow(),
    );
    let narrow = params.k - params.base2k;
    let other = ckks_encrypt(
        &params,
        module,
        host_module,
        &encoder,
        &sk,
        narrow,
        &re,
        &im,
        &mut scratch.borrow(),
    );
    for hint in [PrepareHint::OneShot, PrepareHint::Reuse] {
        check_prepare_into(
            &format!("left {hint:?}"),
            "ckks_prepare_left_into",
            module.ckks_prepared_left_bytes_of_from_infos(&ct, hint),
            || module.ckks_prepared_left_alloc_from_infos(&ct, hint),
            |p| p.prep.data(),
            |p| p.prep.data_mut(),
            |a, s| module.ckks_prepare_left(a, hint, s),
            |p, a, s| module.ckks_prepare_left_into(p, a, s),
            (&ct, &other),
            &mut scratch.borrow(),
        );
        check_prepare_into(
            &format!("right {hint:?}"),
            "ckks_prepare_right_into",
            module.ckks_prepared_right_bytes_of_from_infos(&ct, hint),
            || module.ckks_prepared_right_alloc_from_infos(&ct, hint),
            |p| p.prep.data(),
            |p| p.prep.data_mut(),
            |a, s| module.ckks_prepare_right(a, hint, s),
            |p, a, s| module.ckks_prepare_right_into(p, a, s),
            (&ct, &other),
            &mut scratch.borrow(),
        );
    }
}

/// [`test_prepare_into`] on one side: `a` prepared into an allocation, again into it after poisoning it, then `other`,
/// of another layout, rejected by `op`.
#[allow(clippy::too_many_arguments)]
fn check_prepare_into<
    BE: poulpy_hal::layouts::Backend<OwnedBuf = AlignedBuf>,
    P: CKKSInfos + poulpy_core::layouts::GLWEInfos,
    A,
>(
    label: &str,
    op: &str,
    bytes_of: usize,
    alloc: impl Fn() -> P,
    view: fn(&P) -> &AlignedBuf,
    view_mut: fn(&mut P) -> &mut AlignedBuf,
    prepare: impl Fn(&A, &mut ScratchArena<'_, BE>) -> crate::CKKSResult<P>,
    into: impl Fn(&mut P, &A, &mut ScratchArena<'_, BE>) -> crate::CKKSResult<()>,
    (a, other): (&A, &A),
    scratch: &mut ScratchArena<'_, BE>,
) {
    let fresh = prepare(a, scratch).unwrap();
    let mut res = alloc();
    assert_eq!(view(&res).len(), bytes_of, "{label}: bytes_of");
    into(&mut res, a, scratch).unwrap();
    assert_eq!(view(&res)[..], view(&fresh)[..], "{label}: prepare-into differs from prepare");
    let address = view(&res).as_ptr();
    view_mut(&mut res).as_mut().fill(0xA5);
    into(&mut res, a, scratch).unwrap();
    assert_eq!(view(&res).as_ptr(), address, "{label}: the refill reallocated");
    assert_eq!(view(&res)[..], view(&fresh)[..], "{label}: the refill differs from prepare");
    let infos = (res.glwe_layout(), res.meta());
    let err = into(&mut res, other, scratch).unwrap_err();
    assert!(
        matches!(err.composition(), Some(CKKSCompositionError::InvalidOperand { op: got, .. }) if *got == op),
        "{label}: unexpected error: {err}"
    );
    assert_eq!(
        view(&res)[..],
        view(&fresh)[..],
        "{label}: a rejected source changed the operand"
    );
    assert_eq!(
        (res.glwe_layout(), res.meta()),
        infos,
        "{label}: a rejected source changed the operand's infos"
    );
}

pub fn test_mul_ct_explicit_metadata_error<BE, F, E>(
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
    let (re1, im1) = super::helpers::test_vector_1::<F>(m);
    let (re2, im2) = super::helpers::test_vector_2::<F>(m);
    let (sk_raw, sk) = gen_sk_with_raw(&params, module, host_module, [0u8; 32]);
    let mut scratch = alloc_scratch(&params, module);
    let tsk = gen_tsk(&params, module, &sk_raw, &mut scratch.borrow());

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
    let mut ct2 = ckks_encrypt(
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
    ct1.set_log_budget(8);
    ct2.set_log_budget(9);
    let mut ct_res = alloc_ct(&params, module, params.k);
    let err = module
        .ckks_mul_into(&mut ct_res, &ct1, &ct2, &tsk, &mut scratch.borrow())
        .unwrap_err();
    assert_ckks_error(
        "mul_ct_explicit_metadata_error",
        &err,
        CKKSCompositionError::MultiplicationPrecisionUnderflow {
            op: "mul",
            lhs_log_budget: 8,
            rhs_log_budget: 9,
            lhs_log_delta: params.prec().log_delta(),
            rhs_log_delta: params.prec().log_delta(),
        },
    );
}
