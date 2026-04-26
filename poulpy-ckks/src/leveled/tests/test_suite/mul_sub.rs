//! Tests for `CKKSMulSubOps` — fused `dst -= a · b` variants.
//!
//! # Test inventory
//!
//! | Function | Path exercised |
//! |----------|----------------|
//! | [`test_mul_sub_ct_aligned`] | ct-ct, all operands aligned |
//! | [`test_mul_sub_ct_unaligned_dst`] | ct-ct with `dst` at a lower `log_budget` |
//! | [`test_mul_sub_pt_vec_znx_aligned`] | ZNX plaintext, aligned |
//! | [`test_mul_sub_pt_vec_znx_into_delta_log_delta`] | ZNX plaintext at lower `log_delta` |
//! | [`test_mul_sub_pt_vec_rnx_aligned`] | RNX plaintext, aligned |
//! | [`test_mul_sub_pt_const_znx_into_aligned`] | ZNX constant, aligned |
//! | [`test_mul_sub_pt_const_rnx_aligned`] | RNX constant, aligned |
//! | [`test_mul_sub_pt_const_znx_zero_preserves_dst_meta`] | ZNX zero constant no-op |
//! | [`test_mul_sub_pt_const_rnx_zero_preserves_dst_meta`] | RNX zero constant no-op |

use poulpy_hal::{
    api::{ScratchOwnedAlloc, ScratchOwnedBorrow},
    layouts::ScratchOwned,
};

use crate::{CKKSInfos, layouts::plaintext::CKKSConstPlaintextConversion, leveled::api::CKKSMulSubOps};

use super::helpers::{TestContext, TestMulBackend as Backend, TestScalar, TestVector, assert_ct_meta};

const CONST_RE: f64 = 0.2718281828459045;
const CONST_IM: f64 = -0.1414213562373095;
const DELTA_LOG_DECIMAL: usize = 8;

fn alloc_scratch<BE: Backend, F: TestScalar>(ctx: &TestContext<BE, F>) -> ScratchOwned<BE> {
    let ct_infos = ctx.params.glwe_layout();
    let tsk_infos = ctx.params.tsk_layout();
    let ct_bytes = ctx.module.ckks_mul_sub_ct_tmp_bytes(&ct_infos, &tsk_infos);
    let pt_znx_bytes = ctx
        .module
        .ckks_mul_sub_pt_vec_znx_tmp_bytes(&ct_infos, &ct_infos, &ctx.meta());
    let pt_rnx_bytes = ctx
        .module
        .ckks_mul_sub_pt_vec_rnx_tmp_bytes(&ct_infos, &ct_infos, &ctx.meta());
    let const_bytes = ctx.module.ckks_mul_sub_pt_const_tmp_bytes(&ct_infos, &ct_infos, &ctx.meta());
    let bytes = ct_bytes.max(pt_znx_bytes).max(pt_rnx_bytes).max(const_bytes);
    ScratchOwned::<BE>::alloc(ctx.scratch_size.max(bytes))
}

fn scaled<F: TestScalar>(v: &[F], scale: F) -> Vec<F> {
    v.iter().copied().map(|x| x * scale).collect()
}

fn cmul<F: TestScalar>(a_re: &[F], a_im: &[F], b_re: &[F], b_im: &[F]) -> (Vec<F>, Vec<F>) {
    let m = a_re.len();
    let mut re = Vec::with_capacity(m);
    let mut im = Vec::with_capacity(m);
    for i in 0..m {
        re.push(a_re[i] * b_re[i] - a_im[i] * b_im[i]);
        im.push(a_re[i] * b_im[i] + a_im[i] * b_re[i]);
    }
    (re, im)
}

fn cmul_scalar<F: TestScalar>(a_re: &[F], a_im: &[F], c_re: F, c_im: F) -> (Vec<F>, Vec<F>) {
    let m = a_re.len();
    let mut re = Vec::with_capacity(m);
    let mut im = Vec::with_capacity(m);
    for i in 0..m {
        re.push(a_re[i] * c_re - a_im[i] * c_im);
        im.push(a_re[i] * c_im + a_im[i] * c_re);
    }
    (re, im)
}

pub fn test_mul_sub_ct_aligned<BE: Backend, F: TestScalar>(ctx: &TestContext<BE, F>) {
    let mut scratch = alloc_scratch(ctx);
    let half = F::from_f64(0.5).unwrap();
    let dst_re = scaled(&ctx.re1, half);
    let dst_im = scaled(&ctx.im1, half);
    let a_re = scaled(&ctx.re2, half);
    let a_im = scaled(&ctx.im2, half);
    let b_re = scaled(&ctx.re1, half);
    let b_im = scaled(&ctx.im2, half);

    let (prod_re, prod_im) = cmul(&a_re, &a_im, &b_re, &b_im);
    let want_re: Vec<F> = (0..dst_re.len()).map(|i| dst_re[i] - prod_re[i]).collect();
    let want_im: Vec<F> = (0..dst_im.len()).map(|i| dst_im[i] - prod_im[i]).collect();

    let mut dst = ctx.encrypt(ctx.max_k(), &dst_re, &dst_im, scratch.borrow());
    let a = ctx.encrypt(ctx.max_k(), &a_re, &a_im, scratch.borrow());
    let b = ctx.encrypt(ctx.max_k(), &b_re, &b_im, scratch.borrow());
    ctx.module
        .ckks_mul_sub_ct(&mut dst, &a, &b, ctx.tsk(), scratch.borrow())
        .unwrap();
    ctx.assert_decrypt_precision("mul_sub_ct_aligned", &dst, &want_re, &want_im, scratch.borrow());
}

/// `dst` starts at a lower `log_budget` than the product `a·b`, forcing
/// alignment inside the fused sub.
pub fn test_mul_sub_ct_unaligned_dst<BE: Backend, F: TestScalar>(ctx: &TestContext<BE, F>) {
    let mut scratch = alloc_scratch(ctx);
    let half = F::from_f64(0.5).unwrap();
    let dst_re = scaled(&ctx.re1, half);
    let dst_im = scaled(&ctx.im1, half);
    let a_re = scaled(&ctx.re2, half);
    let a_im = scaled(&ctx.im2, half);
    let b_re = scaled(&ctx.re1, half);
    let b_im = scaled(&ctx.im2, half);

    let (prod_re, prod_im) = cmul(&a_re, &a_im, &b_re, &b_im);
    let want_re: Vec<F> = (0..dst_re.len()).map(|i| dst_re[i] - prod_re[i]).collect();
    let want_im: Vec<F> = (0..dst_im.len()).map(|i| dst_im[i] - prod_im[i]).collect();

    let smaller_k = ctx.max_k() - ctx.base2k().as_usize() + 1;
    let mut dst = ctx.encrypt(smaller_k, &dst_re, &dst_im, scratch.borrow());
    let a = ctx.encrypt(ctx.max_k(), &a_re, &a_im, scratch.borrow());
    let b = ctx.encrypt(ctx.max_k(), &b_re, &b_im, scratch.borrow());
    ctx.module
        .ckks_mul_sub_ct(&mut dst, &a, &b, ctx.tsk(), scratch.borrow())
        .unwrap();
    ctx.assert_decrypt_precision("mul_sub_ct_unaligned_dst", &dst, &want_re, &want_im, scratch.borrow());
}

pub fn test_mul_sub_pt_vec_znx_aligned<BE: Backend, F: TestScalar>(ctx: &TestContext<BE, F>) {
    let mut scratch = alloc_scratch(ctx);
    let half = F::from_f64(0.5).unwrap();
    let dst_re = scaled(&ctx.re1, half);
    let dst_im = scaled(&ctx.im1, half);
    let a_re = scaled(&ctx.re2, half);
    let a_im = scaled(&ctx.im2, half);
    let b_re = scaled(&ctx.re1, half);
    let b_im = scaled(&ctx.im2, half);

    let (prod_re, prod_im) = cmul(&a_re, &a_im, &b_re, &b_im);
    let want_re: Vec<F> = (0..dst_re.len()).map(|i| dst_re[i] - prod_re[i]).collect();
    let want_im: Vec<F> = (0..dst_im.len()).map(|i| dst_im[i] - prod_im[i]).collect();

    let mut dst = ctx.encrypt(ctx.max_k(), &dst_re, &dst_im, scratch.borrow());
    let a = ctx.encrypt(ctx.max_k(), &a_re, &a_im, scratch.borrow());
    let pt = ctx.encode_pt_znx(&b_re, &b_im);
    ctx.module
        .ckks_mul_sub_pt_vec_znx(&mut dst, &a, &pt, scratch.borrow())
        .unwrap();
    ctx.assert_decrypt_precision("mul_sub_pt_vec_znx_aligned", &dst, &want_re, &want_im, scratch.borrow());
}

pub fn test_mul_sub_pt_vec_znx_into_delta_log_delta<BE: Backend, F: TestScalar>(ctx: &TestContext<BE, F>) {
    let mut scratch = alloc_scratch(ctx);
    let half = F::from_f64(0.5).unwrap();
    let low_log_delta = ctx.meta().log_delta - DELTA_LOG_DECIMAL;
    let low_prec = ctx.precision_at(low_log_delta);
    let dst_re = scaled(&ctx.re1, half);
    let dst_im = scaled(&ctx.im1, half);
    let (a_re, a_im) = ctx.quantized_vector(TestVector::First, ctx.meta().log_delta);
    let (b_re, b_im) = ctx.quantized_vector(TestVector::Second, low_log_delta);
    let a_re = scaled(&a_re, half);
    let a_im = scaled(&a_im, half);
    let b_re = scaled(&b_re, half);
    let b_im = scaled(&b_im, half);

    let (prod_re, prod_im) = cmul(&a_re, &a_im, &b_re, &b_im);
    let want_re: Vec<F> = (0..dst_re.len()).map(|i| dst_re[i] - prod_re[i]).collect();
    let want_im: Vec<F> = (0..dst_im.len()).map(|i| dst_im[i] - prod_im[i]).collect();

    let mut dst = ctx.encrypt(ctx.max_k(), &dst_re, &dst_im, scratch.borrow());
    let a = ctx.encrypt(ctx.max_k(), &a_re, &a_im, scratch.borrow());
    let pt = ctx.encode_pt_znx_with_prec(&b_re, &b_im, low_prec);
    ctx.module
        .ckks_mul_sub_pt_vec_znx(&mut dst, &a, &pt, scratch.borrow())
        .unwrap();
    ctx.assert_decrypt_precision_at_log_delta(
        "mul_sub_pt_vec_znx_into_delta_log_delta",
        &dst,
        &want_re,
        &want_im,
        low_log_delta,
        scratch.borrow(),
    );
}

pub fn test_mul_sub_pt_vec_rnx_aligned<BE: Backend, F: TestScalar>(ctx: &TestContext<BE, F>) {
    let mut scratch = alloc_scratch(ctx);
    let half = F::from_f64(0.5).unwrap();
    let dst_re = scaled(&ctx.re1, half);
    let dst_im = scaled(&ctx.im1, half);
    let a_re = scaled(&ctx.re2, half);
    let a_im = scaled(&ctx.im2, half);
    let b_re = scaled(&ctx.re1, half);
    let b_im = scaled(&ctx.im2, half);

    let (prod_re, prod_im) = cmul(&a_re, &a_im, &b_re, &b_im);
    let want_re: Vec<F> = (0..dst_re.len()).map(|i| dst_re[i] - prod_re[i]).collect();
    let want_im: Vec<F> = (0..dst_im.len()).map(|i| dst_im[i] - prod_im[i]).collect();

    let mut dst = ctx.encrypt(ctx.max_k(), &dst_re, &dst_im, scratch.borrow());
    let a = ctx.encrypt(ctx.max_k(), &a_re, &a_im, scratch.borrow());
    let pt = ctx.encode_pt_rnx(&b_re, &b_im);
    ctx.module
        .ckks_mul_sub_pt_vec_rnx(&mut dst, &a, &pt, ctx.meta(), scratch.borrow())
        .unwrap();
    ctx.assert_decrypt_precision("mul_sub_pt_vec_rnx_aligned", &dst, &want_re, &want_im, scratch.borrow());
}

pub fn test_mul_sub_pt_const_znx_into_aligned<BE: Backend, F: TestScalar>(ctx: &TestContext<BE, F>) {
    let mut scratch = alloc_scratch(ctx);
    let half = F::from_f64(0.5).unwrap();
    let dst_re = scaled(&ctx.re1, half);
    let dst_im = scaled(&ctx.im1, half);
    let a_re = scaled(&ctx.re2, half);
    let a_im = scaled(&ctx.im2, half);

    let (c_re, c_im) = ctx.quantized_const(CONST_RE, CONST_IM, ctx.meta().log_delta);
    let (prod_re, prod_im) = cmul_scalar(&a_re, &a_im, c_re, c_im);
    let want_re: Vec<F> = (0..dst_re.len()).map(|i| dst_re[i] - prod_re[i]).collect();
    let want_im: Vec<F> = (0..dst_im.len()).map(|i| dst_im[i] - prod_im[i]).collect();

    let mut dst = ctx.encrypt(ctx.max_k(), &dst_re, &dst_im, scratch.borrow());
    let a = ctx.encrypt(ctx.max_k(), &a_re, &a_im, scratch.borrow());
    let cst_rnx = ctx.const_rnx(Some(CONST_RE), Some(CONST_IM));
    let cst_znx = cst_rnx.to_znx(ctx.base2k(), ctx.meta()).unwrap();
    ctx.module
        .ckks_mul_sub_pt_const_znx(&mut dst, &a, &cst_znx, scratch.borrow())
        .unwrap();
    ctx.assert_decrypt_precision(
        "mul_sub_pt_const_znx_into_aligned",
        &dst,
        &want_re,
        &want_im,
        scratch.borrow(),
    );
}

pub fn test_mul_sub_pt_const_rnx_aligned<BE: Backend, F: TestScalar>(ctx: &TestContext<BE, F>) {
    let mut scratch = alloc_scratch(ctx);
    let half = F::from_f64(0.5).unwrap();
    let dst_re = scaled(&ctx.re1, half);
    let dst_im = scaled(&ctx.im1, half);
    let a_re = scaled(&ctx.re2, half);
    let a_im = scaled(&ctx.im2, half);

    let (c_re, c_im) = ctx.quantized_const(CONST_RE, CONST_IM, ctx.meta().log_delta);
    let (prod_re, prod_im) = cmul_scalar(&a_re, &a_im, c_re, c_im);
    let want_re: Vec<F> = (0..dst_re.len()).map(|i| dst_re[i] - prod_re[i]).collect();
    let want_im: Vec<F> = (0..dst_im.len()).map(|i| dst_im[i] - prod_im[i]).collect();

    let mut dst = ctx.encrypt(ctx.max_k(), &dst_re, &dst_im, scratch.borrow());
    let a = ctx.encrypt(ctx.max_k(), &a_re, &a_im, scratch.borrow());
    let cst = ctx.const_rnx(Some(CONST_RE), Some(CONST_IM));
    ctx.module
        .ckks_mul_sub_pt_const_rnx(&mut dst, &a, &cst, ctx.meta(), scratch.borrow())
        .unwrap();
    ctx.assert_decrypt_precision("mul_sub_pt_const_rnx_aligned", &dst, &want_re, &want_im, scratch.borrow());
}

pub fn test_mul_sub_pt_const_znx_zero_preserves_dst_meta<BE: Backend, F: TestScalar>(ctx: &TestContext<BE, F>) {
    let mut scratch = alloc_scratch(ctx);
    let half = F::from_f64(0.5).unwrap();
    let dst_re = scaled(&ctx.re1, half);
    let dst_im = scaled(&ctx.im1, half);
    let a_re = scaled(&ctx.re2, half);
    let a_im = scaled(&ctx.im2, half);

    let mut dst = ctx.encrypt(ctx.max_k(), &dst_re, &dst_im, scratch.borrow());
    let a = ctx.encrypt(ctx.max_k(), &a_re, &a_im, scratch.borrow());
    let dst_meta = dst.meta();
    let cst_rnx = ctx.const_rnx(None, None);
    let cst_znx = cst_rnx
        .to_znx(ctx.base2k(), ctx.precision_at(ctx.meta().log_delta - DELTA_LOG_DECIMAL))
        .unwrap();
    ctx.module
        .ckks_mul_sub_pt_const_znx(&mut dst, &a, &cst_znx, scratch.borrow())
        .unwrap();

    assert_ct_meta("mul_sub_pt_const_znx_zero", &dst, dst_meta.log_delta, dst_meta.log_budget);
    ctx.assert_decrypt_precision("mul_sub_pt_const_znx_zero", &dst, &dst_re, &dst_im, scratch.borrow());
}

pub fn test_mul_sub_pt_const_rnx_zero_preserves_dst_meta<BE: Backend, F: TestScalar>(ctx: &TestContext<BE, F>) {
    let mut scratch = alloc_scratch(ctx);
    let half = F::from_f64(0.5).unwrap();
    let dst_re = scaled(&ctx.re1, half);
    let dst_im = scaled(&ctx.im1, half);
    let a_re = scaled(&ctx.re2, half);
    let a_im = scaled(&ctx.im2, half);

    let mut dst = ctx.encrypt(ctx.max_k(), &dst_re, &dst_im, scratch.borrow());
    let a = ctx.encrypt(ctx.max_k(), &a_re, &a_im, scratch.borrow());
    let dst_meta = dst.meta();
    let cst = ctx.const_rnx(None, None);
    ctx.module
        .ckks_mul_sub_pt_const_rnx(
            &mut dst,
            &a,
            &cst,
            ctx.precision_at(ctx.meta().log_delta - DELTA_LOG_DECIMAL),
            scratch.borrow(),
        )
        .unwrap();

    assert_ct_meta("mul_sub_pt_const_rnx_zero", &dst, dst_meta.log_delta, dst_meta.log_budget);
    ctx.assert_decrypt_precision("mul_sub_pt_const_rnx_zero", &dst, &dst_re, &dst_im, scratch.borrow());
}
