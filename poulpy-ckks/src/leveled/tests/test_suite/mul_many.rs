//! Tests for `CKKSMulManyOps::ckks_mul_many`.
//!
//! # Test inventory
//!
//! | Function | Path exercised |
//! |----------|----------------|
//! | [`test_mul_many_aligned`] | balanced tree on `n=4` aligned inputs |
//! | [`test_mul_many_single_smaller_output`] | one input into a narrower output |
//! | [`test_mul_many_odd_tree`] | odd `n=5` exercising the carry-up branch |
//! | [`test_mul_many_unaligned_log_budget`] | one input rescaled by one limb |
//! | [`test_mul_many_smaller_output`] | output narrower than would be needed |

use poulpy_hal::{
    api::{ScratchOwnedAlloc, ScratchOwnedBorrow},
    layouts::ScratchOwned,
};

use crate::leveled::api::CKKSMulManyOps;

use super::helpers::{TestContext, TestMulBackend as Backend, TestScalar};

type Factors<F> = (Vec<(Vec<F>, Vec<F>)>, Vec<F>, Vec<F>);

fn scaled_pair<F: TestScalar>(re: &[F], im: &[F], scale: F) -> (Vec<F>, Vec<F>) {
    let re = re.iter().copied().map(|x| x * scale).collect();
    let im = im.iter().copied().map(|x| x * scale).collect();
    (re, im)
}

fn cmul_assign<F: TestScalar>(acc_re: &mut [F], acc_im: &mut [F], re: &[F], im: &[F]) {
    for i in 0..acc_re.len() {
        let a = acc_re[i];
        let b = acc_im[i];
        let c = re[i];
        let d = im[i];
        acc_re[i] = a * c - b * d;
        acc_im[i] = a * d + b * c;
    }
}

/// Builds `n` small-magnitude inputs and the expected elementwise product.
fn build_factors<F: TestScalar>(ctx: &TestContext<impl Backend, F>, n: usize) -> Factors<F> {
    let scale = F::from_f64(0.5).unwrap();
    let (re_a, im_a) = scaled_pair(&ctx.re1, &ctx.im1, scale);
    let (re_b, im_b) = scaled_pair(&ctx.re2, &ctx.im2, scale);
    let mut factors: Vec<(Vec<F>, Vec<F>)> = Vec::with_capacity(n);
    for k in 0..n {
        factors.push(if k % 2 == 0 {
            (re_a.clone(), im_a.clone())
        } else {
            (re_b.clone(), im_b.clone())
        });
    }
    let m: usize = ctx.re1.len();
    let mut want_re: Vec<F> = vec![F::from_f64(1.0).unwrap(); m];
    let mut want_im: Vec<F> = vec![F::zero(); m];
    for (re, im) in factors.iter() {
        cmul_assign(&mut want_re, &mut want_im, re, im);
    }
    (factors, want_re, want_im)
}

fn alloc_scratch<BE: Backend, F: TestScalar>(ctx: &TestContext<BE, F>, n: usize) -> ScratchOwned<BE> {
    let ct_infos = ctx.params.glwe_layout();
    let tsk_infos = ctx.params.tsk_layout();
    let bytes = ctx.module.ckks_mul_many_tmp_bytes(n, &ct_infos, &tsk_infos);
    ScratchOwned::<BE>::alloc(ctx.scratch_size.max(bytes))
}

fn run<BE: Backend, F: TestScalar>(ctx: &TestContext<BE, F>, n: usize, output_k: usize, label: &str) {
    let mut scratch = alloc_scratch(ctx, n);
    let (factors, want_re, want_im) = build_factors(ctx, n);
    let cts: Vec<_> = factors
        .iter()
        .map(|(re, im)| ctx.encrypt(ctx.max_k(), re, im, scratch.borrow()))
        .collect();
    let ct_refs: Vec<&_> = cts.iter().collect();
    let mut ct_res = ctx.alloc_ct(output_k);
    ctx.module
        .ckks_mul_many(&mut ct_res, &ct_refs, ctx.tsk(), scratch.borrow())
        .unwrap();
    ctx.assert_decrypt_precision(label, &ct_res, &want_re, &want_im, scratch.borrow());
}

pub fn test_mul_many_aligned<BE: Backend, F: TestScalar>(ctx: &TestContext<BE, F>) {
    run(ctx, 4, ctx.max_k(), "mul_many_aligned");
}

pub fn test_mul_many_single_smaller_output<BE: Backend, F: TestScalar>(ctx: &TestContext<BE, F>) {
    let mut scratch = alloc_scratch(ctx, 1);
    let ct = ctx.encrypt(ctx.max_k(), &ctx.re1, &ctx.im1, scratch.borrow());
    let ct_refs = vec![&ct];
    let mut ct_res = ctx.alloc_ct(ctx.max_k() - ctx.base2k().as_usize() - 1);
    ctx.module
        .ckks_mul_many(&mut ct_res, &ct_refs, ctx.tsk(), scratch.borrow())
        .unwrap();
    ctx.assert_decrypt_precision(
        "mul_many_single_smaller_output",
        &ct_res,
        &ctx.re1,
        &ctx.im1,
        scratch.borrow(),
    );
}

pub fn test_mul_many_odd_tree<BE: Backend, F: TestScalar>(ctx: &TestContext<BE, F>) {
    run(ctx, 5, ctx.max_k(), "mul_many_odd_tree");
}

pub fn test_mul_many_unaligned_log_budget<BE: Backend, F: TestScalar>(ctx: &TestContext<BE, F>) {
    let n: usize = 4;
    let mut scratch = alloc_scratch(ctx, n);
    let (factors, want_re, want_im) = build_factors(ctx, n);
    let smaller_k = ctx.max_k() - ctx.base2k().as_usize() + 1;
    let cts: Vec<_> = factors
        .iter()
        .enumerate()
        .map(|(i, (re, im))| {
            let k = if i == 2 { smaller_k } else { ctx.max_k() };
            ctx.encrypt(k, re, im, scratch.borrow())
        })
        .collect();
    let ct_refs: Vec<&_> = cts.iter().collect();
    let mut ct_res = ctx.alloc_ct(ctx.max_k());
    ctx.module
        .ckks_mul_many(&mut ct_res, &ct_refs, ctx.tsk(), scratch.borrow())
        .unwrap();
    ctx.assert_decrypt_precision("mul_many unaligned_log_budget", &ct_res, &want_re, &want_im, scratch.borrow());
}

pub fn test_mul_many_smaller_output<BE: Backend, F: TestScalar>(ctx: &TestContext<BE, F>) {
    let n: usize = 4;
    run(ctx, n, ctx.max_k() - ctx.base2k().as_usize() - 1, "mul_many smaller_output");
}
