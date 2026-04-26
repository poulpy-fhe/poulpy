//! Subtraction tests for the `CKKSSubOpsUnsafe` unsafe API.
//!
//! The safe [`CKKSSubOps`](super::super::sub::CKKSSubOps) path is literally
//! the unnormalized default plus a trailing `glwe_normalize_assign`, so
//! the many path-coverage tests in [`super::sub`] already exercise the
//! shared core for free. These tests only cover what's structurally unique
//! to the unsafe API:
//!
//! - the `unsafe`-trait dispatch reaches the right default helper,
//! - meta (`log_budget`, `log_delta`) is set by the unnormalized op,
//! - a caller-supplied `glwe_normalize_assign` recovers a decryptable
//!   ciphertext equivalent to the safe path.
//!
//! One test is kept per distinct kernel family:
//!
//! | Function | Kernel exercised |
//! |----------|------------------|
//! | [`test_sub_ct_aligned_unsafe`] | ct-ct, `glwe_sub` / shift-sub fast path |
//! | [`test_sub_ct_assign_aligned_unsafe`] | ct-ct inplace, `glwe_sub_assign` |
//! | [`test_sub_pt_vec_znx_into_unsafe`] | ct - ZNX plaintext, `VecZnxRshSub` |
//! | [`test_sub_pt_vec_rnx_into_unsafe`] | ct - RNX plaintext, rnx→znx wrapper |
//! | [`test_sub_pt_const_znx_into_aligned_unsafe`] | ct - ZNX const, raw `data_mut()[..] -= digit` path |

use poulpy_core::GLWENormalize;
use poulpy_hal::api::ScratchOwnedBorrow;

use crate::{CKKSInfos, layouts::plaintext::CKKSConstPlaintextConversion, leveled::api::CKKSSubOpsUnsafe};

use super::helpers::{
    TestContext, TestScalar, TestSubBackend as Backend, assert_binary_output_meta, assert_ct_meta, assert_unary_output_meta,
};

const CONST_RE: f64 = 0.314_159_265_358_979_3;
const CONST_IM: f64 = -0.271_828_182_845_904_5;

pub fn test_sub_ct_aligned_unsafe<BE: Backend, F: TestScalar>(ctx: &TestContext<BE, F>) {
    let mut scratch = ctx.alloc_scratch();
    let ct1 = ctx.encrypt(ctx.max_k(), &ctx.re1, &ctx.im1, scratch.borrow());
    let ct2 = ctx.encrypt(ctx.max_k(), &ctx.re2, &ctx.im2, scratch.borrow());
    let (want_re, want_im) = ctx.want_sub();
    let mut ct_res = ctx.alloc_ct(ctx.max_k());
    unsafe {
        ctx.module
            .ckks_sub_into_unsafe(&mut ct_res, &ct1, &ct2, scratch.borrow())
            .unwrap();
    }
    assert_binary_output_meta("sub_ct_aligned_unsafe", &ct_res, &ct1, &ct2);
    ctx.module.glwe_normalize_assign(&mut ct_res, scratch.borrow());
    ctx.assert_decrypt_precision("sub_ct_aligned_unsafe", &ct_res, &want_re, &want_im, scratch.borrow());
}

pub fn test_sub_ct_assign_aligned_unsafe<BE: Backend, F: TestScalar>(ctx: &TestContext<BE, F>) {
    let mut scratch = ctx.alloc_scratch();
    let mut ct1 = ctx.encrypt(ctx.max_k(), &ctx.re1, &ctx.im1, scratch.borrow());
    let ct2 = ctx.encrypt(ctx.max_k(), &ctx.re2, &ctx.im2, scratch.borrow());
    let (want_re, want_im) = ctx.want_sub();
    let expected_log_budget = ct1.log_budget().min(ct2.log_budget());
    let expected_log_delta = ct1.log_delta().max(ct2.log_delta());
    unsafe {
        ctx.module.ckks_sub_assign_unsafe(&mut ct1, &ct2, scratch.borrow()).unwrap();
    }
    assert_ct_meta("sub_ct_assign_aligned_unsafe", &ct1, expected_log_delta, expected_log_budget);
    ctx.module.glwe_normalize_assign(&mut ct1, scratch.borrow());
    ctx.assert_decrypt_precision("sub_ct_assign_aligned_unsafe", &ct1, &want_re, &want_im, scratch.borrow());
}

pub fn test_sub_pt_vec_znx_into_unsafe<BE: Backend, F: TestScalar>(ctx: &TestContext<BE, F>) {
    let mut scratch = ctx.alloc_scratch();
    let ct1 = ctx.encrypt(ctx.max_k(), &ctx.re1, &ctx.im1, scratch.borrow());
    let pt_znx = ctx.encode_pt_znx(&ctx.re2, &ctx.im2);
    let (want_re, want_im) = ctx.want_sub();
    let mut ct_res = ctx.alloc_ct(ctx.max_k());
    unsafe {
        ctx.module
            .ckks_sub_pt_vec_znx_into_unsafe(&mut ct_res, &ct1, &pt_znx, scratch.borrow())
            .unwrap();
    }
    assert_unary_output_meta("sub_pt_vec_znx_into_unsafe", &ct_res, &ct1);
    ctx.module.glwe_normalize_assign(&mut ct_res, scratch.borrow());
    ctx.assert_decrypt_precision("sub_pt_vec_znx_into_unsafe", &ct_res, &want_re, &want_im, scratch.borrow());
}

pub fn test_sub_pt_vec_rnx_into_unsafe<BE: Backend, F: TestScalar>(ctx: &TestContext<BE, F>) {
    let mut scratch = ctx.alloc_scratch();
    let ct1 = ctx.encrypt(ctx.max_k(), &ctx.re1, &ctx.im1, scratch.borrow());
    let pt_rnx = ctx.encode_pt_rnx(&ctx.re2, &ctx.im2);
    let (want_re, want_im) = ctx.want_sub();
    let mut ct_res = ctx.alloc_ct(ctx.max_k());
    unsafe {
        ctx.module
            .ckks_sub_pt_vec_rnx_into_unsafe(&mut ct_res, &ct1, &pt_rnx, ctx.meta(), scratch.borrow())
            .unwrap();
    }
    assert_unary_output_meta("sub_pt_vec_rnx_into_unsafe", &ct_res, &ct1);
    ctx.module.glwe_normalize_assign(&mut ct_res, scratch.borrow());
    ctx.assert_decrypt_precision("sub_pt_vec_rnx_into_unsafe", &ct_res, &want_re, &want_im, scratch.borrow());
}

pub fn test_sub_pt_const_znx_into_aligned_unsafe<BE: Backend, F: TestScalar>(ctx: &TestContext<BE, F>) {
    let mut scratch = ctx.alloc_scratch();
    let ct = ctx.encrypt(ctx.max_k(), &ctx.re1, &ctx.im1, scratch.borrow());
    let (const_re, const_im) = ctx.quantized_const(CONST_RE, CONST_IM, ctx.meta().log_delta);
    let want_re: Vec<F> = ctx.re1.iter().map(|x| *x - const_re).collect();
    let want_im: Vec<F> = ctx.im1.iter().map(|x| *x - const_im).collect();
    let mut ct_res = ctx.alloc_ct(ctx.max_k());
    let cst_rnx = ctx.const_rnx(Some(CONST_RE), Some(CONST_IM));
    let cst_znx = cst_rnx
        .to_znx_at_k(
            ctx.base2k(),
            ct.log_budget()
                .checked_add(ctx.meta().log_delta)
                .expect("aligned precision overflow"),
            ctx.meta().log_delta,
        )
        .unwrap();
    unsafe {
        ctx.module
            .ckks_sub_pt_const_znx_into_unsafe(&mut ct_res, &ct, &cst_znx, scratch.borrow())
            .unwrap();
    }
    assert_unary_output_meta("sub_pt_const_znx_into_aligned_unsafe", &ct_res, &ct);
    ctx.module.glwe_normalize_assign(&mut ct_res, scratch.borrow());
    ctx.assert_decrypt_precision(
        "sub_pt_const_znx_into_aligned_unsafe",
        &ct_res,
        &want_re,
        &want_im,
        scratch.borrow(),
    );
}
