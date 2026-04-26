//! Negation tests (out-of-place and in-place).
//!
//! # Test inventory
//!
//! ## Operations-layer negation (`GLWE<_, CKKS>::neg`)
//!
//! | Function | Path exercised |
//! |----------|----------------|
//! | [`test_neg_aligned`] | out-of-place negation |
//! | [`test_neg_smaller_output`] | out-of-place negation into a smaller output buffer |
//!
//! ## Operations-layer negation (`GLWE<_, CKKS>::neg_assign`)
//!
//! | Function | Path exercised |
//! |----------|----------------|
//! | [`test_neg_assign`] | in-place negation |

use crate::{CKKSInfos, leveled::api::CKKSNegOps};

use super::helpers::{TestContext, TestNegBackend as Backend, TestScalar, assert_ct_meta, assert_unary_output_meta};
use anyhow::Result;
use poulpy_hal::api::ScratchOwnedBorrow;

// ─── negation out-of-place (GLWE<_, CKKS>::neg) ────────────────────────────

/// Negation out-of-place.
pub fn test_neg_aligned<BE: Backend, F: TestScalar>(ctx: &TestContext<BE, F>) -> Result<()> {
    let mut scratch = ctx.alloc_scratch();
    let ct1 = ctx.encrypt(ctx.max_k(), &ctx.re1, &ctx.im1, scratch.borrow());
    let (want_re, want_im) = ctx.want_neg();
    let mut ct_res = ctx.alloc_ct(ctx.max_k());
    ctx.module.ckks_neg_into(&mut ct_res, &ct1, scratch.borrow())?;
    assert_unary_output_meta("neg", &ct_res, &ct1);
    ctx.assert_decrypt_precision("neg", &ct_res, &want_re, &want_im, scratch.borrow());
    Ok(())
}

/// Negation out-of-place.
pub fn test_neg_smaller_output<BE: Backend, F: TestScalar>(ctx: &TestContext<BE, F>) -> Result<()> {
    let mut scratch = ctx.alloc_scratch();
    let ct1 = ctx.encrypt(ctx.max_k(), &ctx.re1, &ctx.im1, scratch.borrow());
    let (want_re, want_im) = ctx.want_neg();
    let mut ct_res = ctx.alloc_ct(ctx.max_k() - ctx.base2k().as_usize() - 1);
    ctx.module.ckks_neg_into(&mut ct_res, &ct1, scratch.borrow())?;
    assert_unary_output_meta("neg smaller_output", &ct_res, &ct1);
    ctx.assert_decrypt_precision("neg", &ct_res, &want_re, &want_im, scratch.borrow());

    Ok(())
}

// ─── negation in-place (GLWE<_, CKKS>::neg_assign) ────────────────────────

/// Negation in-place.
pub fn test_neg_assign<BE: Backend, F: TestScalar>(ctx: &TestContext<BE, F>) -> Result<()> {
    let mut scratch = ctx.alloc_scratch();
    let mut ct = ctx.encrypt(ctx.max_k(), &ctx.re1, &ctx.im1, scratch.borrow());
    let (want_re, want_im) = ctx.want_neg();
    let expected_log_delta = ct.log_delta();
    let expected_log_budget = ct.log_budget();
    ctx.module.ckks_neg_assign(&mut ct)?;
    assert_ct_meta("neg_assign", &ct, expected_log_delta, expected_log_budget);
    ctx.assert_decrypt_precision("neg_assign", &ct, &want_re, &want_im, scratch.borrow());
    Ok(())
}
