//! Composition tests: multi-step CKKS evaluation paths that combine primitives.

use super::helpers::{TestCompositionBackend, TestContext, TestScalar, assert_ckks_error};
use crate::{
    CKKSCompositionError, CKKSInfos,
    leveled::api::{CKKSAddOps, CKKSMulOps},
};
use poulpy_hal::{
    api::{ScratchOwnedAlloc, ScratchOwnedBorrow},
    layouts::ScratchOwned,
};

fn constant_rnx<BE: super::helpers::TestBackend, F: TestScalar>(
    ctx: &TestContext<BE, F>,
    c: (f64, f64),
) -> crate::layouts::plaintext::CKKSPlaintextRnx<F> {
    let m = ctx.params.n / 2;
    ctx.encode_pt_rnx(&vec![F::from_f64(c.0).unwrap(); m], &vec![F::from_f64(c.1).unwrap(); m])
}

fn alloc_composition_scratch<BE: TestCompositionBackend, F: TestScalar>(ctx: &TestContext<BE, F>) -> ScratchOwned<BE>
where
    ScratchOwned<BE>: ScratchOwnedAlloc<BE> + ScratchOwnedBorrow<BE>,
{
    let ct_infos = ctx.params.glwe_layout();
    let prec = ctx.meta();
    let mul_pt_vec_rnx = ctx.module.ckks_mul_pt_vec_rnx_tmp_bytes(&ct_infos, &ct_infos, &prec);
    ScratchOwned::<BE>::alloc(ctx.scratch_size.max(mul_pt_vec_rnx))
}

fn poly2_expected<BE: super::helpers::TestBackend, F: TestScalar>(
    ctx: &TestContext<BE, F>,
    c0: (f64, f64),
    c1: (f64, f64),
    c2: (f64, f64),
) -> (Vec<F>, Vec<F>) {
    let m = ctx.module.n() / 2;
    let two = F::from_f64(2.0).unwrap();
    let c0_re = F::from_f64(c0.0).unwrap();
    let c0_im = F::from_f64(c0.1).unwrap();
    let c1_re = F::from_f64(c1.0).unwrap();
    let c1_im = F::from_f64(c1.1).unwrap();
    let c2_re = F::from_f64(c2.0).unwrap();
    let c2_im = F::from_f64(c2.1).unwrap();
    let want_re: Vec<F> = (0..m)
        .map(|j| {
            let x_re = ctx.re1[j];
            let x_im = ctx.im1[j];
            let x2_re = x_re * x_re - x_im * x_im;
            let x2_im = two * x_re * x_im;
            c0_re + c1_re * x_re - c1_im * x_im + c2_re * x2_re - c2_im * x2_im
        })
        .collect();
    let want_im: Vec<F> = (0..m)
        .map(|j| {
            let x_re = ctx.re1[j];
            let x_im = ctx.im1[j];
            let x2_re = x_re * x_re - x_im * x_im;
            let x2_im = two * x_re * x_im;
            c0_im + c1_re * x_im + c1_im * x_re + c2_re * x2_im + c2_im * x2_re
        })
        .collect();
    (want_re, want_im)
}

fn same_offset_expected<BE: super::helpers::TestBackend, F: TestScalar>(
    ctx: &TestContext<BE, F>,
    c1: (f64, f64),
    c2: (f64, f64),
) -> (Vec<F>, Vec<F>) {
    let m = ctx.module.n() / 2;
    let coeff_re = F::from_f64(c1.0 + c2.0).unwrap();
    let coeff_im = F::from_f64(c1.1 + c2.1).unwrap();
    let want_re: Vec<F> = (0..m).map(|j| coeff_re * ctx.re1[j] - coeff_im * ctx.im1[j]).collect();
    let want_im: Vec<F> = (0..m).map(|j| coeff_re * ctx.im1[j] + coeff_im * ctx.re1[j]).collect();
    (want_re, want_im)
}

fn mul_by_y_expected<BE: super::helpers::TestBackend, F: TestScalar>(
    ctx: &TestContext<BE, F>,
    c0: (f64, f64),
    c1: (f64, f64),
    c2: (f64, f64),
) -> (Vec<F>, Vec<F>) {
    let m = ctx.module.n() / 2;
    let (poly_re, poly_im) = poly2_expected(ctx, c0, c1, c2);
    let want_re: Vec<F> = (0..m).map(|j| poly_re[j] * ctx.re2[j] - poly_im[j] * ctx.im2[j]).collect();
    let want_im: Vec<F> = (0..m).map(|j| poly_re[j] * ctx.im2[j] + poly_im[j] * ctx.re2[j]).collect();
    (want_re, want_im)
}

/// Adding two plaintext-scaled copies of the same ciphertext stays accurate.
pub fn test_linear_sum<BE: TestCompositionBackend, F: TestScalar>(ctx: &TestContext<BE, F>) {
    let mut scratch = alloc_composition_scratch(ctx);

    let c1 = (0.625, -0.125);
    let c2 = (-0.375, 0.25);
    let pt1 = constant_rnx(ctx, c1);
    let pt2 = constant_rnx(ctx, c2);
    let (want_re, want_im) = same_offset_expected(ctx, c1, c2);

    let x = ctx.encrypt(ctx.max_k(), &ctx.re1, &ctx.im1, scratch.borrow());
    // `alloc_ct` takes the total stored width `k`. For `x * pt`, the output
    // effective width is `x.effective_k() - log_delta`, which is numerically
    // equal to `x.log_budget()` in these tests.
    let mut term1 = ctx.alloc_ct(x.log_budget());
    let mut term2 = ctx.alloc_ct(x.log_budget());
    ctx.module
        .ckks_mul_pt_vec_rnx_into(&mut term1, &x, &pt1, ctx.meta(), scratch.borrow())
        .unwrap();
    ctx.module
        .ckks_mul_pt_vec_rnx_into(&mut term2, &x, &pt2, ctx.meta(), scratch.borrow())
        .unwrap();

    assert_eq!(
        term1.log_budget(),
        term2.log_budget(),
        "linear branches should remain aligned"
    );
    ctx.module.ckks_add_assign(&mut term1, &term2, scratch.borrow()).unwrap();

    ctx.assert_decrypt_precision("linear_sum", &term1, &want_re, &want_im, scratch.borrow());
}

/// A mixed `c1*x + c2*x^2` composition remains decryptable and accurate.
pub fn test_poly2_sum<BE: TestCompositionBackend, F: TestScalar>(ctx: &TestContext<BE, F>) {
    let mut scratch = alloc_composition_scratch(ctx);

    let c1 = (0.625, -0.125);
    let c2 = (-0.375, 0.25);
    let pt1 = constant_rnx(ctx, c1);
    let pt2 = constant_rnx(ctx, c2);
    let (want_re, want_im) = poly2_expected(ctx, (0.0, 0.0), c1, c2);

    let x = ctx.encrypt(ctx.max_k(), &ctx.re1, &ctx.im1, scratch.borrow());
    // Likewise, `square` consumes one `log_delta` chunk, so the post-square
    // effective width is `x.effective_k() - log_delta == x.log_budget()`.
    let mut x2 = ctx.alloc_ct(x.log_budget());
    ctx.module.ckks_square_into(&mut x2, &x, ctx.tsk(), scratch.borrow()).unwrap();

    // These allocations still mean "result effective_k", not "headroom only".
    // The equality with `log_budget()` is specific to these operation chains.
    let mut term1 = ctx.alloc_ct(x.log_budget());
    let mut term2 = ctx.alloc_ct(x2.log_budget());
    ctx.module
        .ckks_mul_pt_vec_rnx_into(&mut term1, &x, &pt1, ctx.meta(), scratch.borrow())
        .unwrap();
    ctx.module
        .ckks_mul_pt_vec_rnx_into(&mut term2, &x2, &pt2, ctx.meta(), scratch.borrow())
        .unwrap();

    assert!(
        term1.log_budget() > term2.log_budget(),
        "x^2 branch should consume more precision"
    );
    let mut sum = ctx.alloc_ct(term2.effective_k());
    ctx.module.ckks_add_into(&mut sum, &term1, &term2, scratch.borrow()).unwrap();

    ctx.assert_decrypt_precision("poly2_sum", &sum, &want_re, &want_im, scratch.borrow());
}

/// Adding a constant plaintext to `c1*x + c2*x^2` keeps the expected value.
pub fn test_poly2_sum_with_const<BE: TestCompositionBackend, F: TestScalar>(ctx: &TestContext<BE, F>) {
    let mut scratch = alloc_composition_scratch(ctx);

    let c0 = (0.125, -0.0625);
    let c1 = (0.625, -0.125);
    let c2 = (-0.375, 0.25);
    let pt0 = constant_rnx(ctx, c0);
    let pt1 = constant_rnx(ctx, c1);
    let pt2 = constant_rnx(ctx, c2);
    let (want_re, want_im) = poly2_expected(ctx, c0, c1, c2);

    let x = ctx.encrypt(ctx.max_k(), &ctx.re1, &ctx.im1, scratch.borrow());
    // `square` output width matches `x.log_budget()` here only because
    // squaring drops one `log_delta` chunk from `x.effective_k()`.
    let mut x2 = ctx.alloc_ct(x.log_budget());
    ctx.module.ckks_square_into(&mut x2, &x, ctx.tsk(), scratch.borrow()).unwrap();

    // `mul_pt_vec_rnx` again allocates by post-op effective width; using
    // `log_budget()` is a numeric shortcut that holds for this fixture.
    let mut term1 = ctx.alloc_ct(x.log_budget());
    let mut term2 = ctx.alloc_ct(x2.log_budget());
    ctx.module
        .ckks_mul_pt_vec_rnx_into(&mut term1, &x, &pt1, ctx.meta(), scratch.borrow())
        .unwrap();
    ctx.module
        .ckks_mul_pt_vec_rnx_into(&mut term2, &x2, &pt2, ctx.meta(), scratch.borrow())
        .unwrap();
    let mut poly = ctx.alloc_ct(term2.effective_k());
    ctx.module.ckks_add_into(&mut poly, &term1, &term2, scratch.borrow()).unwrap();
    ctx.module
        .ckks_add_pt_vec_rnx_assign(&mut poly, &pt0, ctx.meta(), scratch.borrow())
        .unwrap();

    ctx.assert_decrypt_precision("poly2_sum_with_const", &poly, &want_re, &want_im, scratch.borrow());
}

/// Evaluates `y * (c0 + c1*x + c2*x^2)` with encrypted `x` and `y`.
pub fn test_poly2_mul<BE: TestCompositionBackend, F: TestScalar>(ctx: &TestContext<BE, F>) {
    let mut scratch = alloc_composition_scratch(ctx);

    let c0 = (0.125, -0.0625);
    let c1 = (0.625, -0.125);
    let c2 = (-0.375, 0.25);
    let pt0 = constant_rnx(ctx, c0);
    let pt1 = constant_rnx(ctx, c1);
    let pt2 = constant_rnx(ctx, c2);
    let (want_re, want_im) = mul_by_y_expected(ctx, c0, c1, c2);

    let x = ctx.encrypt(ctx.max_k(), &ctx.re1, &ctx.im1, scratch.borrow());
    let y = ctx.encrypt(ctx.max_k(), &ctx.re2, &ctx.im2, scratch.borrow());
    // Here too, the allocated `k` is the post-square effective width, which
    // happens to equal `x.log_budget()` for CKKS square in this setup.
    let mut x2 = ctx.alloc_ct(x.log_budget());
    ctx.module.ckks_square_into(&mut x2, &x, ctx.tsk(), scratch.borrow()).unwrap();

    // The same caution applies below: `log_budget()` is not semantically the
    // ciphertext width, it just matches the required post-op width here.
    let mut term1 = ctx.alloc_ct(x.log_budget());
    let mut term2 = ctx.alloc_ct(x2.log_budget());
    ctx.module
        .ckks_mul_pt_vec_rnx_into(&mut term1, &x, &pt1, ctx.meta(), scratch.borrow())
        .unwrap();
    ctx.module
        .ckks_mul_pt_vec_rnx_into(&mut term2, &x2, &pt2, ctx.meta(), scratch.borrow())
        .unwrap();
    let mut poly = ctx.alloc_ct(term2.effective_k());
    ctx.module.ckks_add_into(&mut poly, &term1, &term2, scratch.borrow()).unwrap();
    ctx.module
        .ckks_add_pt_vec_rnx_assign(&mut poly, &pt0, ctx.meta(), scratch.borrow())
        .unwrap();

    let mut res = ctx.alloc_ct(ctx.max_k());
    ctx.module
        .ckks_mul_into(&mut res, &y, &poly, ctx.tsk(), scratch.borrow())
        .unwrap();

    ctx.assert_decrypt_precision("poly2_mul", &res, &want_re, &want_im, scratch.borrow());
}

/// Repeated squaring on unit-circle slots should exhaust HE capacity before it blows up numerically.
pub fn test_repeated_square_exhausts_capacity<BE: TestCompositionBackend, F: TestScalar>(ctx: &TestContext<BE, F>) {
    let mut scratch = alloc_composition_scratch(ctx);
    let mut ct = ctx.encrypt(ctx.max_k(), &ctx.re1, &ctx.im1, scratch.borrow());
    let mut squares = 0usize;

    while ct.log_budget() >= ct.log_delta() {
        let prev_log_budget = ct.log_budget();
        let prev_log_delta = ct.log_delta();
        let next_k = ct.effective_k() - ct.log_delta();
        let mut next = ctx.alloc_ct(next_k);
        ctx.module
            .ckks_square_into(&mut next, &ct, ctx.tsk(), scratch.borrow())
            .unwrap();
        assert_eq!(
            next.log_delta(),
            prev_log_delta,
            "square should preserve log_delta across repeated squaring",
        );
        assert_eq!(
            next.log_budget(),
            prev_log_budget - prev_log_delta,
            "square should consume exactly one log_delta chunk of HE capacity",
        );
        ct = next;
        squares += 1;
    }

    assert!(squares > 0, "expected at least one square");
    assert!(
        ct.log_budget() < ct.log_delta(),
        "expected squaring to consume all HE capacity"
    );
    let (got_re, got_im) = ctx.decrypt_decode(&ct, scratch.borrow());
    for (idx, (re, im)) in got_re.iter().zip(got_im.iter()).enumerate() {
        assert!(
            re.is_finite() && im.is_finite(),
            "repeated_square_exhausts_capacity: non-finite slot at index {idx}: ({re:?}, {im:?})"
        );
        let norm = *re * *re + *im * *im;
        assert!(
            norm <= F::from_f64(1.25).unwrap(),
            "repeated_square_exhausts_capacity: slot {idx} escaped unit-circle bound: norm={norm:?}"
        );
    }

    let mut no_capacity = ctx.alloc_ct(ctx.max_k());
    let err = ctx
        .module
        .ckks_square_into(&mut no_capacity, &ct, ctx.tsk(), scratch.borrow())
        .unwrap_err();
    assert_ckks_error(
        "repeated_square_exhausts_capacity",
        &err,
        CKKSCompositionError::MultiplicationPrecisionUnderflow {
            op: "mul",
            lhs_log_budget: ct.log_budget(),
            rhs_log_budget: ct.log_budget(),
            lhs_log_delta: ct.log_delta(),
            rhs_log_delta: ct.log_delta(),
        },
    );
}
