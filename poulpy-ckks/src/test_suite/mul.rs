//! Multiplication tests: ct × ct and ct² (square).
//!
//! # Test inventory
//!
//! ## ct × ct multiplication out of place (`GLWE<_, CKKS>::mul`)
//!
//! | Function | Path exercised |
//! |----------|----------------|
//! | [`test_mul_ct_aligned`] | both inputs at same `log_budget()` |
//! | [`test_mul_ct_limb_boundary_precision`] | precision immediately after a radix-limb boundary |
//! | [`test_mul_ct_delta_a_lt_b`] | `a.log_budget() < b.log_budget()` |
//! | [`test_mul_ct_delta_a_gt_b`] | `a.log_budget() > b.log_budget()` |
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
//!
//! ## ct x pt inplace (`GLWE<_, CKKS>::mul_pt_vec_assign`)
//!
//! | Function | Path exercised |
//! |----------|----------------|
//! | [`test_mul_pt_vec_assign`] | - |
use crate::{
    CKKSCompositionError, CKKSInfos,
    api::{CKKSEncryptOps, CKKSMulOps},
    layouts::CKKSModuleAlloc,
};

use poulpy_core::EncryptionLayout;

use poulpy_hal::{
    api::{NegacyclicFFT, NegacyclicFFTNew, ScratchOwnedAlloc, ScratchOwnedBorrow},
    layouts::{HostBytesBackend, Module, ScratchOwned},
    source::Source,
};

use super::helpers::{
    MUL_CONST, PT_PREC, TestContextBackend, TestContextModule, TestScalar, TestVector, alloc_ct, alloc_scratch,
    assert_ckks_error, assert_decrypt_precision, assert_decrypt_precision_at_log_delta, assert_mul_ct_output_meta,
    assert_mul_pt_output_meta, ckks_decrypt_decode, ckks_encrypt, ckks_encrypt_with_prec, ckks_pt_cst_full, ckks_snapshot,
    encode_and_upload_pt, gen_sk_with_raw, gen_tsk, precision_at, quantize, quantized_const, quantized_vector, want_mul,
    want_square,
};

use crate::{SetCKKSInfos, test_suite::CKKSTestParams, test_suite::reference_encoder::ReferenceEncoder};

const DELTA_LOG_DELTA: usize = 8;

// ─── ct × ct out-of-place ───────────────────────────────────────────────────

pub fn test_mul_ct_aligned<BE, F, E>(params: CKKSTestParams, module: &Module<BE>, host_module: &Module<HostBytesBackend>)
where
    BE: TestContextBackend,
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
    assert_mul_ct_output_meta("mul_ct_aligned", &ct_res, &ct1, &ct2);
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

pub fn test_mul_ct_limb_boundary_precision<BE, F, E>(
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
    let aligned_k = (2 * params.prec().log_delta() + 1).div_ceil(params.base2k) * params.base2k;
    assert!(aligned_k < params.k);

    let encoder = ReferenceEncoder::<E>::new(params.n / 2).unwrap();
    let (re1, im1) = super::helpers::test_vector_1::<F>(params.n / 2);
    let (re2, im2) = super::helpers::test_vector_2::<F>(params.n / 2);
    let pt1 = encode_and_upload_pt(host_module, module, &encoder, params.base2k.into(), params.prec(), &re1, &im1);
    let pt2 = encode_and_upload_pt(host_module, module, &encoder, params.base2k.into(), params.prec(), &re2, &im2);
    let (sk_raw, sk) = gen_sk_with_raw(&params, module, host_module, [0x42; 32]);
    let mut scratch = alloc_scratch(&params, module);
    let tsk = gen_tsk(&params, module, &sk_raw, &mut scratch.borrow());

    let mut operation_error = |k: usize| {
        let mut layout = params.glwe_layout().layout;
        layout.k = k.into();
        let encryption = EncryptionLayout::new_from_default_sigma(layout).unwrap();
        let mut ct1 = alloc_ct(&params, module, k);
        module
            .ckks_encrypt_sk(
                &mut ct1,
                &pt1,
                &sk,
                &encryption,
                &mut Source::new([0xe1; 32]),
                &mut Source::new([0xa1; 32]),
                &mut scratch.borrow(),
            )
            .unwrap();
        let mut ct2 = alloc_ct(&params, module, k);
        module
            .ckks_encrypt_sk(
                &mut ct2,
                &pt2,
                &sk,
                &encryption,
                &mut Source::new([0xe2; 32]),
                &mut Source::new([0xa2; 32]),
                &mut scratch.borrow(),
            )
            .unwrap();
        let mut product = alloc_ct(&params, module, k);
        module
            .ckks_mul_into(&mut product, &ct1, &ct2, &tsk, &mut scratch.borrow())
            .unwrap();

        let (ct1_re, ct1_im) = ckks_decrypt_decode::<_, F, _>(&params, module, &encoder, &ct1, &sk, &mut scratch.borrow());
        let (ct2_re, ct2_im) = ckks_decrypt_decode::<_, F, _>(&params, module, &encoder, &ct2, &sk, &mut scratch.borrow());
        let (product_re, product_im) =
            ckks_decrypt_decode::<_, F, _>(&params, module, &encoder, &product, &sk, &mut scratch.borrow());
        let (want_re, want_im) = want_mul(&ct1_re, &ct1_im, &ct2_re, &ct2_im);

        product_re
            .iter()
            .zip(&product_im)
            .zip(want_re.iter().zip(&want_im))
            .map(|((&got_re, &got_im), (&want_re, &want_im))| (got_re - want_re).hypot(got_im - want_im).to_f64().unwrap())
            .fold(0.0, f64::max)
    };

    let aligned_error = operation_error(aligned_k);
    let next_error = operation_error(aligned_k + 1);
    let precision_loss = (next_error / aligned_error).log2();
    assert!(
        precision_loss <= 2.0,
        "increasing k from {aligned_k} to {} lost {precision_loss:.3} bits of multiplication precision \
         (errors {aligned_error:e} -> {next_error:e})",
        aligned_k + 1,
    );
}

pub fn test_mul_ct_delta_a_lt_b<BE, F, E>(params: CKKSTestParams, module: &Module<BE>, host_module: &Module<HostBytesBackend>)
where
    BE: TestContextBackend,
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
    assert_mul_ct_output_meta("mul_ct a_lt_b", &ct_res, &ct1, &ct2);
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
    BE: TestContextBackend,
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
    assert_mul_ct_output_meta("mul_ct a_gt_b", &ct_res, &ct1, &ct2);
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
    BE: TestContextBackend,
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
    assert_mul_ct_output_meta("mul_ct delta_log_delta", &ct_res, &ct1, &ct2);
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

pub fn test_mul_ct_smaller_output<BE, F, E>(params: CKKSTestParams, module: &Module<BE>, host_module: &Module<HostBytesBackend>)
where
    BE: TestContextBackend,
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
    let mut ct_res = alloc_ct(&params, module, params.k - params.base2k - 1);
    module
        .ckks_mul_into(&mut ct_res, &ct1, &ct2, &tsk, &mut scratch.borrow())
        .unwrap();
    assert_mul_ct_output_meta("mul_ct smaller_output", &ct_res, &ct1, &ct2);
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
    BE: TestContextBackend,
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
    let mut ct_res = alloc_ct(&params, module, params.k - params.base2k - 1);
    let exact_bytes = module.ckks_mul_tmp_bytes(&ct_res, &ct1, &ct2, &tsk);
    let mut mul_scratch = ScratchOwned::<BE>::alloc(exact_bytes);
    module
        .ckks_mul_into(&mut ct_res, &ct1, &ct2, &tsk, &mut mul_scratch.borrow())
        .unwrap();
    assert_mul_ct_output_meta("mul_ct smaller_output_exact_scratch", &ct_res, &ct1, &ct2);
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
    BE: TestContextBackend,
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
    assert_mul_ct_output_meta("mul_ct_assign_aligned", &ct_res, &ct_res_meta, &ct1);
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
    BE: TestContextBackend,
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
    assert_mul_ct_output_meta("mul_ct_assign_self_lt", &ct_res, &ct_res_meta, &ct1);
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
    BE: TestContextBackend,
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
    assert_mul_ct_output_meta("mul_ct_assign_self_gt", &ct_res, &ct_res_meta, &ct1);
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
    BE: TestContextBackend,
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
    assert_mul_ct_output_meta("square_aligned", &ct_res, &ct, &ct);
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
    BE: TestContextBackend,
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
    assert_mul_ct_output_meta("square_rescaled_input", &ct_res, &ct, &ct);
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
    BE: TestContextBackend,
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
    let mut ct_res = alloc_ct(&params, module, params.k - params.base2k - 1);
    module
        .ckks_square_into(&mut ct_res, &ct, &tsk, &mut scratch.borrow())
        .unwrap();
    assert_mul_ct_output_meta("square_smaller_output", &ct_res, &ct, &ct);
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
    BE: TestContextBackend,
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
    assert_mul_ct_output_meta("square_assign", &ct, &ct_in_meta, &ct_in_meta);
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
    BE: TestContextBackend,
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
    assert_mul_pt_output_meta("mul_pt_vec_into_aligned", &ct_res, &ct, &pt);
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
    BE: TestContextBackend,
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
    assert_mul_pt_output_meta("mul_pt_vec_into_delta_log_delta", &ct_res, &ct, &pt);
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
    BE: TestContextBackend,
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
    let mut ct_res = alloc_ct(&params, module, params.k - params.base2k - 1);
    module
        .ckks_mul_pt_vec_into(&mut ct_res, &ct, &pt, &mut scratch.borrow())
        .unwrap();
    assert_mul_pt_output_meta("mul_pt_vec_into_smaller_output", &ct_res, &ct, &pt);
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
    BE: TestContextBackend,
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
    assert_mul_pt_output_meta("mul_pt_vec_assign", &ct, &ct_meta, &pt);
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
    BE: TestContextBackend,
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
    assert_mul_pt_output_meta("mul_const_into_aligned", &ct_res, &ct, &PT_PREC);
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
    BE: TestContextBackend,
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
    assert_mul_pt_output_meta("mul_const_assign", &ct, &ct_meta, &PT_PREC);
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
    BE: TestContextBackend,
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
    assert_mul_pt_output_meta("mul_const_into_delta_log_delta", &ct_res, &ct, &PT_PREC);
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

// ─── error test ──────────────────────────────────────────────────────────────

/// `ckks_mul_prepared_assign` must reject a prepared operand built under a
/// different layout (here: a different `base2k`) with a typed error, before
/// touching the destination.
pub fn test_mul_prepared_layout_mismatch_error<BE, F, E>(
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
    let prepared = module.ckks_prepare_right(&other, &mut scratch.borrow()).unwrap();
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

pub fn test_mul_ct_explicit_metadata_error<BE, F, E>(
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
