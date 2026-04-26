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
//! | [`test_mul_ct_smaller_output`] | output has smaller `max_k()` than inputs |
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
//! ## ct x pt_znx out of place (`GLWE<_, CKKS>::mul_pt_vec_znx`)
//!
//! | Function | Path exercised |
//! |----------|----------------|
//! | [`test_mul_pt_vec_znx_into_aligned`] | input and output at same `log_budget()` |
//! | [`test_mul_pt_vec_znx_into_smaller_output`] | output at smaller `log_budget()` |
//!
//! ## ct x pt_znx inplace (`GLWE<_, CKKS>::mul_pt_vec_znx_assign`)
//!
//! | Function | Path exercised |
//! |----------|----------------|
//! | [`test_mul_pt_vec_znx_assign`] | - |
//!
//! ## ct x pt_rnx out of place (`GLWE<_, CKKS>::mul_pt_vec_rnx`)
//!
//! | Function | Path exercised |
//! |----------|----------------|
//! | [`test_mul_pt_vec_znx_into_aligned`] | input and output at same `log_budget()` |
//! | [`test_mul_pt_vec_znx_into_smaller_output`] | output at smaller `log_budget()` |
//!
//! ## ct x pt_rnx inplace (`GLWE<_, CKKS>::mul_pt_vec_rnx_assign`)
//!
//! | Function | Path exercised |
//! |----------|----------------|
//! | [`test_mul_pt_vec_rnx_assign`] | - |
use crate::{
    CKKSCompositionError, CKKSInfos,
    layouts::plaintext::CKKSConstPlaintextConversion,
    leveled::{
        api::CKKSMulOps,
        tests::test_suite::helpers::{
            TestContext, TestMulBackend as Backend, TestScalar, TestVector, assert_ckks_error, assert_mul_ct_output_meta,
            assert_mul_pt_output_meta,
        },
    },
};

use poulpy_hal::api::ScratchOwnedBorrow;

const CONST_RE: f64 = 0.314_159_265_358_979_3;
const CONST_IM: f64 = -0.271_828_182_845_904_5;
const DELTA_LOG_DECIMAL: usize = 8;
// ─── ct × ct out-of-place (GLWE<_, CKKS>::mul) ─────────────────────────────────

/// ct × ct multiplication with both inputs at the same log_budget().
pub fn test_mul_ct_aligned<BE: Backend, F: TestScalar>(ctx: &TestContext<BE, F>) {
    let mut scratch = ctx.alloc_scratch();
    let ct1 = ctx.encrypt(ctx.max_k(), &ctx.re1, &ctx.im1, scratch.borrow());
    let ct2 = ctx.encrypt(ctx.max_k(), &ctx.re2, &ctx.im2, scratch.borrow());
    let (want_re, want_im) = ctx.want_mul();
    let mut ct_res = ctx.alloc_ct(ctx.max_k());
    ctx.module
        .ckks_mul_into(&mut ct_res, &ct1, &ct2, ctx.tsk(), scratch.borrow())
        .unwrap();
    assert_mul_ct_output_meta("mul_ct_aligned", &ct_res, &ct1, &ct2);
    ctx.assert_decrypt_precision("mul_ct_aligned", &ct_res, &want_re, &want_im, scratch.borrow());
}

/// ct × ct, a.log_budget() < b.log_budget() (a rescaled by one limb).
pub fn test_mul_ct_delta_a_lt_b<BE: Backend, F: TestScalar>(ctx: &TestContext<BE, F>) {
    let mut scratch = ctx.alloc_scratch();
    let ct1 = ctx.encrypt(
        ctx.max_k() - ctx.base2k().as_usize() + 1,
        &ctx.re1,
        &ctx.im1,
        scratch.borrow(),
    );
    let ct2 = ctx.encrypt(ctx.max_k(), &ctx.re2, &ctx.im2, scratch.borrow());
    let (want_re, want_im) = ctx.want_mul();
    let mut ct_res = ctx.alloc_ct(ctx.max_k());
    ctx.module
        .ckks_mul_into(&mut ct_res, &ct1, &ct2, ctx.tsk(), scratch.borrow())
        .unwrap();
    assert_mul_ct_output_meta("mul_ct a_lt_b", &ct_res, &ct1, &ct2);
    ctx.assert_decrypt_precision("mul_ct a_lt_b", &ct_res, &want_re, &want_im, scratch.borrow());
}

/// ct × ct, a.log_budget() > b.log_budget() (b rescaled by one limb).
pub fn test_mul_ct_delta_a_gt_b<BE: Backend, F: TestScalar>(ctx: &TestContext<BE, F>) {
    let mut scratch = ctx.alloc_scratch();
    let ct1 = ctx.encrypt(ctx.max_k(), &ctx.re1, &ctx.im1, scratch.borrow());
    let ct2 = ctx.encrypt(
        ctx.max_k() - ctx.base2k().as_usize() + 1,
        &ctx.re2,
        &ctx.im2,
        scratch.borrow(),
    );
    let (want_re, want_im) = ctx.want_mul();
    let mut ct_res = ctx.alloc_ct(ctx.max_k());
    ctx.module
        .ckks_mul_into(&mut ct_res, &ct1, &ct2, ctx.tsk(), scratch.borrow())
        .unwrap();
    assert_mul_ct_output_meta("mul_ct a_gt_b", &ct_res, &ct1, &ct2);
    ctx.assert_decrypt_precision("mul_ct a_gt_b", &ct_res, &want_re, &want_im, scratch.borrow());
}

/// ct × ct with aligned homomorphic capacity but different log_delta.
pub fn test_mul_ct_delta_log_delta<BE: Backend, F: TestScalar>(ctx: &TestContext<BE, F>) {
    let mut scratch = ctx.alloc_scratch();
    let low_log_delta = ctx.meta().log_delta - DELTA_LOG_DECIMAL;
    let low_prec = ctx.precision_at(low_log_delta);
    let (a_re, a_im) = ctx.quantized_vector(TestVector::First, ctx.meta().log_delta);
    let (b_re, b_im) = ctx.quantized_vector(TestVector::Second, low_log_delta);
    let ct1 = ctx.encrypt(ctx.max_k(), &a_re, &a_im, scratch.borrow());
    let ct2 = ctx.encrypt_with_prec(ctx.max_k() - DELTA_LOG_DECIMAL, &b_re, &b_im, low_prec, scratch.borrow());
    let (want_re, want_im) = ctx.want_mul_from(&a_re, &a_im, &b_re, &b_im);
    let mut ct_res = ctx.alloc_ct(ctx.max_k());
    ctx.module
        .ckks_mul_into(&mut ct_res, &ct1, &ct2, ctx.tsk(), scratch.borrow())
        .unwrap();
    assert_mul_ct_output_meta("mul_ct delta_log_delta", &ct_res, &ct1, &ct2);
    ctx.assert_decrypt_precision_at_log_delta(
        "mul_ct delta_log_delta",
        &ct_res,
        &want_re,
        &want_im,
        low_log_delta,
        scratch.borrow(),
    );
}

/// ct × ct, output buffer has smaller max_k than inputs (offset > 0).
pub fn test_mul_ct_smaller_output<BE: Backend, F: TestScalar>(ctx: &TestContext<BE, F>) {
    let mut scratch = ctx.alloc_scratch();
    let ct1 = ctx.encrypt(ctx.max_k(), &ctx.re1, &ctx.im1, scratch.borrow());
    let ct2 = ctx.encrypt(ctx.max_k(), &ctx.re2, &ctx.im2, scratch.borrow());
    let (want_re, want_im) = ctx.want_mul();
    let mut ct_res = ctx.alloc_ct(ctx.max_k() - ctx.base2k().as_usize() - 1);
    ctx.module
        .ckks_mul_into(&mut ct_res, &ct1, &ct2, ctx.tsk(), scratch.borrow())
        .unwrap();
    assert_mul_ct_output_meta("mul_ct smaller_output", &ct_res, &ct1, &ct2);
    ctx.assert_decrypt_precision("mul_ct smaller_output", &ct_res, &want_re, &want_im, scratch.borrow());
}

// ─── ct × ct out-of-place (GLWE<_, CKKS>::mul_assign) ─────────────────────────────────

/// ct × ct multiplication with both inputs at the same log_budget().
pub fn test_mul_ct_assign_aligned<BE: Backend, F: TestScalar>(ctx: &TestContext<BE, F>) {
    let mut scratch = ctx.alloc_scratch();
    let mut ct_res = ctx.encrypt(ctx.max_k(), &ctx.re1, &ctx.im1, scratch.borrow());
    let ct1 = ctx.encrypt(ctx.max_k(), &ctx.re2, &ctx.im2, scratch.borrow());
    let (want_re, want_im) = ctx.want_mul();
    let ct_res_meta = ct_res.meta();
    ctx.module
        .ckks_mul_assign(&mut ct_res, &ct1, ctx.tsk(), scratch.borrow())
        .unwrap();
    assert_mul_ct_output_meta("mul_ct_aligned", &ct_res, &ct_res_meta, &ct1);
    ctx.assert_decrypt_precision("mul_ct_aligned", &ct_res, &want_re, &want_im, scratch.borrow());
}

/// ct-ct in-place, self.log_budget() < a.log_budget() (a is shifted down to align with self).
pub fn test_mul_ct_assign_self_lt<BE: Backend, F: TestScalar>(ctx: &TestContext<BE, F>) {
    let mut scratch = ctx.alloc_scratch();
    let mut ct_res = ctx.encrypt(
        ctx.max_k() - ctx.base2k().as_usize() - 1,
        &ctx.re1,
        &ctx.im1,
        scratch.borrow(),
    );
    let ct1 = ctx.encrypt(ctx.max_k(), &ctx.re2, &ctx.im2, scratch.borrow());
    let (want_re, want_im) = ctx.want_mul();
    let ct_res_meta = ct_res.meta();
    ctx.module
        .ckks_mul_assign(&mut ct_res, &ct1, ctx.tsk(), scratch.borrow())
        .unwrap();
    assert_mul_ct_output_meta("mul_ct_aligned", &ct_res, &ct_res_meta, &ct1);
    ctx.assert_decrypt_precision("mul_ct_aligned", &ct_res, &want_re, &want_im, scratch.borrow());
}

/// ct-ct in-place, self.log_budget() > a.log_budget() (a is shifted down to align with self).
pub fn test_mul_ct_assign_self_gt<BE: Backend, F: TestScalar>(ctx: &TestContext<BE, F>) {
    let mut scratch = ctx.alloc_scratch();
    let mut ct_res = ctx.encrypt(ctx.max_k(), &ctx.re1, &ctx.im1, scratch.borrow());
    let ct1 = ctx.encrypt(
        ctx.max_k() - ctx.base2k().as_usize() - 1,
        &ctx.re2,
        &ctx.im2,
        scratch.borrow(),
    );
    let (want_re, want_im) = ctx.want_mul();
    let ct_res_meta = ct_res.meta();
    ctx.module
        .ckks_mul_assign(&mut ct_res, &ct1, ctx.tsk(), scratch.borrow())
        .unwrap();
    assert_mul_ct_output_meta("mul_ct_aligned", &ct_res, &ct_res_meta, &ct1);
    ctx.assert_decrypt_precision("mul_ct_aligned", &ct_res, &want_re, &want_im, scratch.borrow());
}

// ─── ct² squaring out of place (GLWE<_, CKKS>::square) ───────────────────────────────────────

/// ct² at default precision (same as fresh encryption).
pub fn test_square_aligned<BE: Backend, F: TestScalar>(ctx: &TestContext<BE, F>) {
    let mut scratch = ctx.alloc_scratch();
    let ct = ctx.encrypt(ctx.max_k(), &ctx.re1, &ctx.im1, scratch.borrow());
    let (want_re, want_im) = ctx.want_square();
    let mut ct_res = ctx.alloc_ct(ctx.max_k());
    ctx.module
        .ckks_square_into(&mut ct_res, &ct, ctx.tsk(), scratch.borrow())
        .unwrap();
    assert_mul_ct_output_meta("square_aligned", &ct_res, &ct, &ct);
    ctx.assert_decrypt_precision("square_aligned", &ct_res, &want_re, &want_im, scratch.borrow());
}

/// ct² after the input has already been rescaled by one limb.
pub fn test_square_rescaled_input<BE: Backend, F: TestScalar>(ctx: &TestContext<BE, F>) {
    let mut scratch = ctx.alloc_scratch();
    let ct = ctx.encrypt(
        ctx.max_k() - ctx.base2k().as_usize() + 1,
        &ctx.re1,
        &ctx.im1,
        scratch.borrow(),
    );
    let (want_re, want_im) = ctx.want_square();
    let mut ct_res = ctx.alloc_ct(ctx.max_k());
    ctx.module
        .ckks_square_into(&mut ct_res, &ct, ctx.tsk(), scratch.borrow())
        .unwrap();
    assert_mul_ct_output_meta("square_rescaled_input", &ct_res, &ct, &ct);
    ctx.assert_decrypt_precision("square_rescaled_input", &ct_res, &want_re, &want_im, scratch.borrow());
}

/// ct² into an output buffer with smaller k.
pub fn test_square_smaller_output<BE: Backend, F: TestScalar>(ctx: &TestContext<BE, F>) {
    let mut scratch = ctx.alloc_scratch();
    let ct = ctx.encrypt(ctx.max_k(), &ctx.re1, &ctx.im1, scratch.borrow());
    let (want_re, want_im) = ctx.want_square();
    let mut ct_res = ctx.alloc_ct(ctx.max_k() - ctx.base2k().as_usize() - 1);
    ctx.module
        .ckks_square_into(&mut ct_res, &ct, ctx.tsk(), scratch.borrow())
        .unwrap();
    assert_mul_ct_output_meta("square_smaller_output", &ct_res, &ct, &ct);
    ctx.assert_decrypt_precision(" square_smaller_output", &ct_res, &want_re, &want_im, scratch.borrow());
}

// ─── ct² squaring inplace (GLWE<_, CKKS>::square) ───────────────────────────────────────

/// ct² inplace.
pub fn test_square_assign<BE: Backend, F: TestScalar>(ctx: &TestContext<BE, F>) {
    let mut scratch = ctx.alloc_scratch();
    let mut ct = ctx.encrypt(ctx.max_k(), &ctx.re1, &ctx.im1, scratch.borrow());
    let (want_re, want_im) = ctx.want_square();
    let ct_in_meta = ct.meta();
    ctx.module.ckks_square_assign(&mut ct, ctx.tsk(), scratch.borrow()).unwrap();
    assert_mul_ct_output_meta("square_assign", &ct, &ct_in_meta, &ct_in_meta);
    ctx.assert_decrypt_precision("square_assign", &ct, &want_re, &want_im, scratch.borrow());
}

// ─── ct x pt_znx out of place (`GLWE<_, CKKS>::mul_pt_vec_znx`)) ───────────────────────────

pub fn test_mul_pt_vec_znx_into_aligned<BE: Backend, F: TestScalar>(ctx: &TestContext<BE, F>) {
    let mut scratch = ctx.alloc_scratch();
    let ct = ctx.encrypt(ctx.max_k(), &ctx.re1, &ctx.im1, scratch.borrow());
    let pt = ctx.encode_pt_znx(&ctx.re2, &ctx.im2);
    let (want_re, want_im) = ctx.want_mul();
    let mut ct_res = ctx.alloc_ct(ctx.max_k());
    ctx.module
        .ckks_mul_pt_vec_znx_into(&mut ct_res, &ct, &pt, scratch.borrow())
        .unwrap();
    assert_mul_pt_output_meta("mul_pt_vec_znx_into_aligned", &ct_res, &ct, &pt);
    ctx.assert_decrypt_precision("mul_pt_vec_znx_into_aligned", &ct_res, &want_re, &want_im, scratch.borrow());
}

pub fn test_mul_pt_vec_znx_into_delta_log_delta<BE: Backend, F: TestScalar>(ctx: &TestContext<BE, F>) {
    let mut scratch = ctx.alloc_scratch();
    let low_log_delta = ctx.meta().log_delta - DELTA_LOG_DECIMAL;
    let low_prec = ctx.precision_at(low_log_delta);
    let (a_re, a_im) = ctx.quantized_vector(TestVector::First, ctx.meta().log_delta);
    let (b_re, b_im) = ctx.quantized_vector(TestVector::Second, low_log_delta);
    let ct = ctx.encrypt(ctx.max_k(), &a_re, &a_im, scratch.borrow());
    let pt = ctx.encode_pt_znx_with_prec(&b_re, &b_im, low_prec);
    let (want_re, want_im) = ctx.want_mul_from(&a_re, &a_im, &b_re, &b_im);
    let mut ct_res = ctx.alloc_ct(ctx.max_k());
    ctx.module
        .ckks_mul_pt_vec_znx_into(&mut ct_res, &ct, &pt, scratch.borrow())
        .unwrap();
    assert_mul_pt_output_meta("mul_pt_vec_znx_into_delta_log_delta", &ct_res, &ct, &pt);
    ctx.assert_decrypt_precision_at_log_delta(
        "mul_pt_vec_znx_into_delta_log_delta",
        &ct_res,
        &want_re,
        &want_im,
        low_log_delta,
        scratch.borrow(),
    );
}

pub fn test_mul_pt_vec_znx_into_smaller_output<BE: Backend, F: TestScalar>(ctx: &TestContext<BE, F>) {
    let mut scratch = ctx.alloc_scratch();
    let ct = ctx.encrypt(ctx.max_k(), &ctx.re1, &ctx.im1, scratch.borrow());
    let pt = ctx.encode_pt_znx(&ctx.re2, &ctx.im2);
    let (want_re, want_im) = ctx.want_mul();
    let mut ct_res = ctx.alloc_ct(ctx.max_k() - ctx.base2k().as_usize() - 1);
    ctx.module
        .ckks_mul_pt_vec_znx_into(&mut ct_res, &ct, &pt, scratch.borrow())
        .unwrap();
    assert_mul_pt_output_meta("mul_pt_vec_znx_into_aligned", &ct_res, &ct, &pt);
    ctx.assert_decrypt_precision("mul_pt_vec_znx_into_aligned", &ct_res, &want_re, &want_im, scratch.borrow());
}

// ─── ct x pt_znx inplace (`GLWE<_, CKKS>::mul_pt_vec_znx_assign`)) ───────────────────────────

pub fn test_mul_pt_vec_znx_assign<BE: Backend, F: TestScalar>(ctx: &TestContext<BE, F>) {
    let mut scratch = ctx.alloc_scratch();
    let mut ct = ctx.encrypt(ctx.max_k(), &ctx.re1, &ctx.im1, scratch.borrow());
    let pt = ctx.encode_pt_znx(&ctx.re2, &ctx.im2);
    let (want_re, want_im) = ctx.want_mul();
    let ct_meta = ct.meta();
    ctx.module.ckks_mul_pt_vec_znx_assign(&mut ct, &pt, scratch.borrow()).unwrap();
    assert_mul_pt_output_meta("mul_pt_vec_znx_into_aligned", &ct, &ct_meta, &pt);
    ctx.assert_decrypt_precision("mul_pt_vec_znx_into_aligned", &ct, &want_re, &want_im, scratch.borrow());
}

// ─── ct x pt_rnx out of place (`GLWE<_, CKKS>::mul_pt_vec_rnx`) ───────────────────────────

pub fn test_mul_pt_vec_rnx_into_aligned<BE: Backend, F: TestScalar>(ctx: &TestContext<BE, F>) {
    let mut scratch = ctx.alloc_scratch();
    let ct = ctx.encrypt(ctx.max_k(), &ctx.re1, &ctx.im1, scratch.borrow());
    let pt = ctx.encode_pt_rnx(&ctx.re2, &ctx.im2);
    let (want_re, want_im) = ctx.want_mul();
    let mut ct_res = ctx.alloc_ct(ctx.max_k());
    ctx.module
        .ckks_mul_pt_vec_rnx_into(&mut ct_res, &ct, &pt, ctx.meta(), scratch.borrow())
        .unwrap();
    assert_mul_pt_output_meta("mul_pt_vec_znx_into_aligned", &ct_res, &ct, &ctx.meta());
    ctx.assert_decrypt_precision("mul_pt_vec_znx_into_aligned", &ct_res, &want_re, &want_im, scratch.borrow());
}

pub fn test_mul_pt_vec_rnx_into_delta_log_delta<BE: Backend, F: TestScalar>(ctx: &TestContext<BE, F>) {
    let mut scratch = ctx.alloc_scratch();
    let low_log_delta = ctx.meta().log_delta - DELTA_LOG_DECIMAL;
    let low_prec = ctx.precision_at(low_log_delta);
    let (a_re, a_im) = ctx.quantized_vector(TestVector::First, ctx.meta().log_delta);
    let (b_re, b_im) = ctx.quantized_vector(TestVector::Second, low_log_delta);
    let ct = ctx.encrypt(ctx.max_k(), &a_re, &a_im, scratch.borrow());
    let pt = ctx.encode_pt_rnx(&b_re, &b_im);
    let (want_re, want_im) = ctx.want_mul_from(&a_re, &a_im, &b_re, &b_im);
    let mut ct_res = ctx.alloc_ct(ctx.max_k());
    ctx.module
        .ckks_mul_pt_vec_rnx_into(&mut ct_res, &ct, &pt, low_prec, scratch.borrow())
        .unwrap();
    assert_mul_pt_output_meta("mul_pt_vec_rnx_into_delta_log_delta", &ct_res, &ct, &low_prec);
    ctx.assert_decrypt_precision_at_log_delta(
        "mul_pt_vec_rnx_into_delta_log_delta",
        &ct_res,
        &want_re,
        &want_im,
        low_log_delta,
        scratch.borrow(),
    );
}

pub fn test_mul_pt_vec_rnx_into_smaller_output<BE: Backend, F: TestScalar>(ctx: &TestContext<BE, F>) {
    let mut scratch = ctx.alloc_scratch();
    let ct = ctx.encrypt(ctx.max_k(), &ctx.re1, &ctx.im1, scratch.borrow());
    let pt = ctx.encode_pt_rnx(&ctx.re2, &ctx.im2);
    let (want_re, want_im) = ctx.want_mul();
    let mut ct_res = ctx.alloc_ct(ctx.max_k() - ctx.base2k().as_usize() - 1);
    ctx.module
        .ckks_mul_pt_vec_rnx_into(&mut ct_res, &ct, &pt, ctx.meta(), scratch.borrow())
        .unwrap();
    assert_mul_pt_output_meta("mul_pt_vec_znx_into_aligned", &ct_res, &ct, &ctx.meta());
    ctx.assert_decrypt_precision("mul_pt_vec_znx_into_aligned", &ct_res, &want_re, &want_im, scratch.borrow());
}

// ─── ct x pt_rnx inplace (`GLWE<_, CKKS>::mul_pt_vec_rnx_assign`)) ───────────────────────────

pub fn test_mul_pt_vec_rnx_assign<BE: Backend, F: TestScalar>(ctx: &TestContext<BE, F>) {
    let mut scratch = ctx.alloc_scratch();
    let mut ct = ctx.encrypt(ctx.max_k(), &ctx.re1, &ctx.im1, scratch.borrow());
    let pt = ctx.encode_pt_rnx(&ctx.re2, &ctx.im2);
    let (want_re, want_im) = ctx.want_mul();
    let ct_meta = ct.meta();
    ctx.module
        .ckks_mul_pt_vec_rnx_assign(&mut ct, &pt, ctx.meta(), scratch.borrow())
        .unwrap();
    assert_mul_pt_output_meta("mul_pt_vec_znx_into_aligned", &ct, &ct_meta, &ctx.meta());
    ctx.assert_decrypt_precision("mul_pt_vec_znx_into_aligned", &ct, &want_re, &want_im, scratch.borrow());
}

pub fn test_mul_pt_const_rnx_into_aligned<BE: Backend, F: TestScalar>(ctx: &TestContext<BE, F>) {
    let mut scratch = ctx.alloc_scratch();
    let ct = ctx.encrypt(ctx.max_k(), &ctx.re1, &ctx.im1, scratch.borrow());
    let (const_re, const_im) = ctx.quantized_const(CONST_RE, CONST_IM, ctx.meta().log_delta);
    let (want_re, want_im) = ctx.want_mul_const_from(&ctx.re1, &ctx.im1, const_re, const_im);
    let mut ct_res = ctx.alloc_ct(ctx.max_k());
    let cst = ctx.const_rnx(Some(CONST_RE), Some(CONST_IM));
    ctx.module
        .ckks_mul_pt_const_rnx_into(&mut ct_res, &ct, &cst, ctx.meta(), scratch.borrow())
        .unwrap();
    assert_mul_pt_output_meta("mul_const_into_aligned", &ct_res, &ct, &ctx.meta());
    ctx.assert_decrypt_precision("mul_const_into_aligned", &ct_res, &want_re, &want_im, scratch.borrow());
}

pub fn test_mul_pt_const_assign<BE: Backend, F: TestScalar>(ctx: &TestContext<BE, F>) {
    let mut scratch = ctx.alloc_scratch();
    let mut ct = ctx.encrypt(ctx.max_k(), &ctx.re1, &ctx.im1, scratch.borrow());
    let (const_re, const_im) = ctx.quantized_const(CONST_RE, CONST_IM, ctx.meta().log_delta);
    let (want_re, want_im) = ctx.want_mul_const_from(&ctx.re1, &ctx.im1, const_re, const_im);
    let ct_meta = ct.meta();
    let cst = ctx.const_rnx(Some(CONST_RE), Some(CONST_IM));
    ctx.module
        .ckks_mul_pt_const_rnx_assign(&mut ct, &cst, ctx.meta(), scratch.borrow())
        .unwrap();
    assert_mul_pt_output_meta("mul_const_assign", &ct, &ct_meta, &ctx.meta());
    ctx.assert_decrypt_precision("mul_const_assign", &ct, &want_re, &want_im, scratch.borrow());
}

pub fn test_mul_pt_const_rnx_into_delta_log_delta<BE: Backend, F: TestScalar>(ctx: &TestContext<BE, F>) {
    let mut scratch = ctx.alloc_scratch();
    let low_log_delta = ctx.meta().log_delta - DELTA_LOG_DECIMAL;
    let low_prec = ctx.precision_at(low_log_delta);
    let (a_re, a_im) = ctx.quantized_vector(TestVector::First, ctx.meta().log_delta);
    let (const_re, const_im) = ctx.quantized_const(CONST_RE, CONST_IM, low_log_delta);
    let ct = ctx.encrypt(ctx.max_k(), &a_re, &a_im, scratch.borrow());
    let (want_re, want_im) = ctx.want_mul_const_from(&a_re, &a_im, const_re, const_im);
    let mut ct_res = ctx.alloc_ct(ctx.max_k());
    let cst = ctx.const_rnx(Some(CONST_RE), Some(CONST_IM));
    ctx.module
        .ckks_mul_pt_const_rnx_into(&mut ct_res, &ct, &cst, low_prec, scratch.borrow())
        .unwrap();
    assert_mul_pt_output_meta("mul_const_into_delta_log_delta", &ct_res, &ct, &low_prec);
    ctx.assert_decrypt_precision_at_log_delta(
        "mul_const_into_delta_log_delta",
        &ct_res,
        &want_re,
        &want_im,
        low_log_delta,
        scratch.borrow(),
    );
}

pub fn test_mul_pt_const_rnx_into_real_only<BE: Backend, F: TestScalar>(ctx: &TestContext<BE, F>) {
    let mut scratch = ctx.alloc_scratch();
    let ct = ctx.encrypt(ctx.max_k(), &ctx.re1, &ctx.im1, scratch.borrow());
    let (const_re, const_im) = ctx.quantized_const(CONST_RE, 0.0, ctx.meta().log_delta);
    let (want_re, want_im) = ctx.want_mul_const_from(&ctx.re1, &ctx.im1, const_re, const_im);
    let mut ct_res = ctx.alloc_ct(ctx.max_k());
    let cst = ctx.const_rnx(Some(CONST_RE), None);
    ctx.module
        .ckks_mul_pt_const_rnx_into(&mut ct_res, &ct, &cst, ctx.meta(), scratch.borrow())
        .unwrap();
    assert_mul_pt_output_meta("mul_const_into_real_only", &ct_res, &ct, &ctx.meta());
    ctx.assert_decrypt_precision("mul_const_into_real_only", &ct_res, &want_re, &want_im, scratch.borrow());
}

pub fn test_mul_pt_const_znx_into_aligned<BE: Backend, F: TestScalar>(ctx: &TestContext<BE, F>) {
    let mut scratch = ctx.alloc_scratch();
    let ct = ctx.encrypt(ctx.max_k(), &ctx.re1, &ctx.im1, scratch.borrow());
    let (const_re, const_im) = ctx.quantized_const(CONST_RE, CONST_IM, ctx.meta().log_delta);
    let (want_re, want_im) = ctx.want_mul_const_from(&ctx.re1, &ctx.im1, const_re, const_im);
    let mut ct_res = ctx.alloc_ct(ctx.max_k());
    let cst_rnx = ctx.const_rnx(Some(CONST_RE), Some(CONST_IM));
    let cst_znx = cst_rnx.to_znx(ctx.base2k(), ctx.meta()).unwrap();
    ctx.module
        .ckks_mul_pt_const_znx_into(&mut ct_res, &ct, &cst_znx, scratch.borrow())
        .unwrap();
    assert_mul_pt_output_meta("mul_const_znx_into_aligned", &ct_res, &ct, &ctx.meta());
    ctx.assert_decrypt_precision("mul_const_znx_into_aligned", &ct_res, &want_re, &want_im, scratch.borrow());
}

/// Multiplication with inconsistent metadata must fail explicitly instead of panicking on usize underflow.
pub fn test_mul_ct_explicit_metadata_error<BE: Backend, F: TestScalar>(ctx: &TestContext<BE, F>) {
    let mut scratch = ctx.alloc_scratch();
    let mut ct1 = ctx.encrypt(ctx.max_k(), &ctx.re1, &ctx.im1, scratch.borrow());
    let mut ct2 = ctx.encrypt(ctx.max_k(), &ctx.re2, &ctx.im2, scratch.borrow());
    ct1.meta.log_budget = 8;
    ct2.meta.log_budget = 9;
    let mut ct_res = ctx.alloc_ct(ctx.max_k());
    let err = ctx
        .module
        .ckks_mul_into(&mut ct_res, &ct1, &ct2, ctx.tsk(), scratch.borrow())
        .unwrap_err();
    assert_ckks_error(
        "mul_ct_explicit_metadata_error",
        &err,
        CKKSCompositionError::MultiplicationPrecisionUnderflow {
            op: "mul",
            lhs_log_budget: 8,
            rhs_log_budget: 9,
            lhs_log_delta: ctx.meta().log_delta,
            rhs_log_delta: ctx.meta().log_delta,
        },
    );
}
