use crate::{
    CKKSCompositionError, CKKSInfos,
    layouts::{ciphertext::CKKSMaintainOps, plaintext::alloc_pt_znx},
    leveled::api::{CKKSAddOps, CKKSDotProductOps},
};
use poulpy_core::layouts::LWEInfos;
use poulpy_core::layouts::{Base2K, Degree, TorusPrecision};
use poulpy_hal::api::ScratchOwnedBorrow;

use super::helpers::{TestAddBackend as Backend, TestContext, TestScalar, assert_ckks_error};

pub fn test_reallocate_limbs_checked_error<BE: Backend, F: TestScalar>(ctx: &TestContext<BE, F>) {
    let mut scratch = ctx.alloc_scratch();
    let mut ct = ctx.encrypt(ctx.max_k(), &ctx.re1, &ctx.im1, scratch.borrow());
    let requested_limbs = ct.effective_k().div_ceil(ct.base2k().as_usize()).saturating_sub(1);
    let err = ctx
        .module
        .ckks_reallocate_limbs_checked(&mut ct, requested_limbs)
        .unwrap_err();
    assert_ckks_error(
        "reallocate_limbs_checked",
        &err,
        CKKSCompositionError::LimbReallocationShrinksBelowMetadata {
            max_k: ct.max_k().as_usize(),
            log_delta: ct.log_delta(),
            base2k: ct.base2k().as_usize(),
            requested_limbs,
        },
    );
}

pub fn test_compact_limbs_copy<BE: Backend, F: TestScalar>(ctx: &TestContext<BE, F>) {
    let mut scratch = ctx.alloc_scratch();
    let mut ct = ctx.encrypt(ctx.max_k(), &ctx.re1, &ctx.im1, scratch.borrow());
    let oversized_limbs = ct.size() + 1;
    ctx.module.ckks_reallocate_limbs_checked(&mut ct, oversized_limbs).unwrap();

    let compact = ctx.module.ckks_compact_limbs_copy(&ct).unwrap();
    let expected_limbs = ct.effective_k().div_ceil(ct.base2k().as_usize());

    assert_eq!(ct.size(), oversized_limbs, "source ciphertext should remain oversized");
    assert_eq!(compact.size(), expected_limbs, "compacted copy should drop excess limbs");
    assert_eq!(compact.meta(), ct.meta(), "compacted copy should preserve metadata");
    assert_eq!(compact.max_k().as_usize(), expected_limbs * ct.base2k().as_usize());

    ctx.assert_decrypt_precision("compact_limbs_copy", &compact, &ctx.re1, &ctx.im1, scratch.borrow());
}

pub fn test_add_pt_vec_znx_alignment_error<BE: Backend, F: TestScalar>(ctx: &TestContext<BE, F>) {
    let mut scratch = ctx.alloc_scratch();
    let mut ct = ctx.encrypt(ctx.max_k(), &ctx.re1, &ctx.im1, scratch.borrow());
    ct.meta.log_budget = 0;
    let pt_znx = alloc_pt_znx(ctx.degree(), ctx.base2k(), ctx.meta());
    let err = ctx
        .module
        .ckks_add_pt_vec_znx_assign(&mut ct, &pt_znx, scratch.borrow())
        .unwrap_err();
    assert_ckks_error(
        "add_pt_vec_znx_alignment",
        &err,
        CKKSCompositionError::PlaintextAlignmentImpossible {
            op: "ckks_add_pt_vec_znx_into",
            ct_log_budget: 0,
            pt_log_delta: ctx.meta().log_delta,
            pt_max_k: pt_znx.max_k().as_usize(),
        },
    );
}

pub fn test_dot_product_overflow_guard<BE: Backend, F: TestScalar>(ctx: &TestContext<BE, F>) {
    let mut scratch = ctx.alloc_scratch();
    let mut dst = crate::layouts::CKKSCiphertext::alloc(Degree(8), TorusPrecision(64), Base2K(63));
    dst.meta = ctx.meta();
    let a = crate::layouts::CKKSCiphertext::alloc(Degree(8), TorusPrecision(64), Base2K(63));
    let b = crate::layouts::CKKSCiphertext::alloc(Degree(8), TorusPrecision(64), Base2K(63));
    let a_refs = vec![&a, &a];
    let b_refs = vec![&b, &b];
    let err = ctx
        .module
        .ckks_dot_product_ct(&mut dst, &a_refs, &b_refs, ctx.tsk(), scratch.borrow())
        .unwrap_err()
        .to_string();
    assert!(
        err.contains("risks i64 overflow"),
        "dot_product_overflow_guard: unexpected error: {err}"
    );
}
