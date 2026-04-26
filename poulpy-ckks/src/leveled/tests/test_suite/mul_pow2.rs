//! Multiplication and division by a power of two.
//!
//! These operations shift the GLWE payload without altering CKKS metadata
//! (`log_delta`, `log_budget`).
//!
//! # Test inventory
//!
//! ## `GLWE<_, CKKS>::mul_pow2` / `mul_pow2_assign`
//!
//! | Function | Path exercised |
//! |----------|----------------|
//! | [`test_mul_pow2_aligned`] | out-of-place, message × 2^bits |
//! | [`test_mul_pow2_smaller_output`] | out-of-place into a smaller output buffer |
//! | [`test_mul_pow2_assign`] | in-place, message × 2^bits |
//!
//! ## `GLWE<_, CKKS>::div_pow2` / `div_pow2_assign`
//!
//! | Function | Path exercised |
//! |----------|----------------|
//! | [`test_div_pow2_aligned`] | out-of-place, message / 2^bits |
//! | [`test_div_pow2_smaller_output`] | out-of-place into a smaller output buffer |
//! | [`test_div_pow2_assign`] | in-place, message / 2^bits |

use crate::{CKKSCompositionError, CKKSInfos, leveled::api::CKKSPow2Ops};

use super::helpers::{
    TestContext, TestPow2Backend as Backend, TestScalar, assert_ckks_error, assert_ct_meta, assert_unary_output_meta,
};
use poulpy_core::layouts::LWEInfos;
use poulpy_hal::api::ScratchOwnedBorrow;

const SHIFT_BITS: usize = 7;

// ─── mul_pow2 (message × 2^bits) ───────────────────────────────────────────────

/// Out-of-place multiplication by 2^bits.
pub fn test_mul_pow2_aligned<BE: Backend, F: TestScalar>(ctx: &TestContext<BE, F>) {
    let mut scratch = ctx.alloc_scratch();
    let ct = ctx.encrypt(ctx.max_k(), &ctx.re1, &ctx.im1, scratch.borrow());
    let (want_re, want_im) = ctx.want_mul_pow2(SHIFT_BITS);
    let mut ct_res = ctx.alloc_ct(ctx.max_k());
    ctx.module
        .ckks_mul_pow2_into(&mut ct_res, &ct, SHIFT_BITS, scratch.borrow())
        .unwrap();
    assert_unary_output_meta("mul_pow2", &ct_res, &ct);
    ctx.assert_decrypt_precision_at_log_delta(
        "mul_pow2",
        &ct_res,
        &want_re,
        &want_im,
        ct.log_delta() - SHIFT_BITS,
        scratch.borrow(),
    );
}

/// Out-of-place multiplication by 2^bits.
pub fn test_mul_pow2_smaller_output<BE: Backend, F: TestScalar>(ctx: &TestContext<BE, F>) {
    let mut scratch = ctx.alloc_scratch();
    let ct = ctx.encrypt(ctx.max_k(), &ctx.re1, &ctx.im1, scratch.borrow());
    let (want_re, want_im) = ctx.want_mul_pow2(SHIFT_BITS);
    let mut ct_res = ctx.alloc_ct(ctx.max_k() - ctx.base2k().as_usize() - 1);
    ctx.module
        .ckks_mul_pow2_into(&mut ct_res, &ct, SHIFT_BITS, scratch.borrow())
        .unwrap();
    assert_unary_output_meta("mul_pow2 smaller_output", &ct_res, &ct);
    ctx.assert_decrypt_precision_at_log_delta(
        "mul_pow2",
        &ct_res,
        &want_re,
        &want_im,
        ct.log_delta() - SHIFT_BITS,
        scratch.borrow(),
    );
}

/// In-place multiplication by 2^bits.
pub fn test_mul_pow2_assign<BE: Backend, F: TestScalar>(ctx: &TestContext<BE, F>) {
    let mut scratch = ctx.alloc_scratch();
    let mut ct = ctx.encrypt(ctx.max_k(), &ctx.re1, &ctx.im1, scratch.borrow());
    let (want_re, want_im) = ctx.want_mul_pow2(SHIFT_BITS);
    let expected_log_delta = ct.log_delta();
    let expected_log_budget = ct.log_budget();
    ctx.module
        .ckks_mul_pow2_assign(&mut ct, SHIFT_BITS, scratch.borrow())
        .unwrap();
    assert_ct_meta("mul_pow2_assign", &ct, expected_log_delta, expected_log_budget);
    ctx.assert_decrypt_precision_at_log_delta(
        "mul_pow2_assign",
        &ct,
        &want_re,
        &want_im,
        expected_log_delta - SHIFT_BITS,
        scratch.borrow(),
    );
}

// ─── div_pow2 (message / 2^bits) ───────────────────────────────────────────────

/// Out-of-place division by 2^bits.
pub fn test_div_pow2_aligned<BE: Backend, F: TestScalar>(ctx: &TestContext<BE, F>) {
    let mut scratch = ctx.alloc_scratch();
    let ct = ctx.encrypt(ctx.max_k(), &ctx.re1, &ctx.im1, scratch.borrow());
    let (want_re, want_im) = ctx.want_div_pow2(SHIFT_BITS);
    let mut ct_res = ctx.alloc_ct(ctx.max_k());
    ctx.module
        .ckks_div_pow2_into(&mut ct_res, &ct, SHIFT_BITS, scratch.borrow())
        .unwrap();
    assert_ct_meta("div_pow2", &ct_res, ct.log_delta(), ct.log_budget() - SHIFT_BITS);
    ctx.assert_decrypt_precision("div_pow2", &ct_res, &want_re, &want_im, scratch.borrow());
}

/// Out-of-place division by 2^bits.
pub fn test_div_pow2_smaller_output<BE: Backend, F: TestScalar>(ctx: &TestContext<BE, F>) {
    let mut scratch = ctx.alloc_scratch();
    let ct = ctx.encrypt(ctx.max_k(), &ctx.re1, &ctx.im1, scratch.borrow());
    let (want_re, want_im) = ctx.want_div_pow2(SHIFT_BITS);
    let mut ct_res = ctx.alloc_ct(ctx.max_k() - ctx.base2k().as_usize() - 1);
    ctx.module
        .ckks_div_pow2_into(&mut ct_res, &ct, SHIFT_BITS, scratch.borrow())
        .unwrap();
    let offset = ct.effective_k().saturating_sub(ct_res.max_k().as_usize());
    assert_ct_meta(
        "div_pow2 smaller_output",
        &ct_res,
        ct.log_delta(),
        ct.log_budget() - SHIFT_BITS - offset,
    );
    ctx.assert_decrypt_precision("div_pow2", &ct_res, &want_re, &want_im, scratch.borrow());
}

/// In-place division by 2^bits.
pub fn test_div_pow2_assign<BE: Backend, F: TestScalar>(ctx: &TestContext<BE, F>) {
    let mut scratch = ctx.alloc_scratch();
    let mut ct = ctx.encrypt(ctx.max_k(), &ctx.re1, &ctx.im1, scratch.borrow());
    let (want_re, want_im) = ctx.want_div_pow2(SHIFT_BITS);
    let expected_log_delta = ct.log_delta();
    let expected_log_budget = ct.log_budget() - SHIFT_BITS;
    ctx.module.ckks_div_pow2_assign(&mut ct, SHIFT_BITS).unwrap();
    assert_ct_meta("div_pow2_assign", &ct, expected_log_delta, expected_log_budget);
    ctx.assert_decrypt_precision("div_pow2_assign", &ct, &want_re, &want_im, scratch.borrow());
}

/// In-place division by too large a power of two must return a clear metadata error.
pub fn test_div_pow2_assign_explicit_error<BE: Backend, F: TestScalar>(ctx: &TestContext<BE, F>) {
    let mut scratch = ctx.alloc_scratch();
    let mut ct = ctx.encrypt(ctx.max_k(), &ctx.re1, &ctx.im1, scratch.borrow());
    let available_log_budget = ct.log_budget();
    let required_bits = available_log_budget + 1;
    let err = ctx.module.ckks_div_pow2_assign(&mut ct, required_bits).unwrap_err();
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
