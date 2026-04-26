//! Tests for `CKKSDotProductOps` — inner-product variants.
//!
//! All tests use `n = 3` terms.
//!
//! # Test inventory
//!
//! | Function | Path exercised |
//! |----------|----------------|
//! | [`test_dot_product_ct_aligned`] | ct · ct, aligned |
//! | [`test_dot_product_ct_unaligned`] | ct · ct, one a-side input rescaled |
//! | [`test_dot_product_ct_unaligned_b`] | ct · ct, one b-side input rescaled |
//! | [`test_dot_product_ct_delta_log_delta`] | ct · ct fallback with non-uniform `log_delta` |
//! | [`test_dot_product_ct_smaller_output`] | ct · ct, output narrower than inputs |
//! | [`test_dot_product_pt_vec_znx_aligned`] | ct · ZNX plaintext |
//! | [`test_dot_product_pt_vec_rnx_aligned`] | ct · RNX plaintext |
//! | [`test_dot_product_const_znx_aligned`] | ct · ZNX constant |
//! | [`test_dot_product_const_rnx_aligned`] | ct · RNX constant |

use poulpy_hal::{
    api::{ScratchOwnedAlloc, ScratchOwnedBorrow},
    layouts::ScratchOwned,
};

use crate::{
    layouts::plaintext::{CKKSConstPlaintextConversion, CKKSPlaintextCstRnx, CKKSPlaintextCstZnx},
    leveled::api::CKKSDotProductOps,
};

use super::helpers::{TestContext, TestMulBackend as Backend, TestScalar, TestVector};

const N: usize = 3;
const DELTA_LOG_DECIMAL: usize = 8;

fn alloc_scratch<BE: Backend, F: TestScalar>(ctx: &TestContext<BE, F>) -> ScratchOwned<BE> {
    let ct_infos = ctx.params.glwe_layout();
    let tsk_infos = ctx.params.tsk_layout();
    let ct_bytes = ctx.module.ckks_dot_product_ct_tmp_bytes(N, &ct_infos, &tsk_infos);
    let pt_znx_bytes = ctx
        .module
        .ckks_dot_product_pt_vec_znx_tmp_bytes(&ct_infos, &ct_infos, &ctx.meta());
    let pt_rnx_bytes = ctx
        .module
        .ckks_dot_product_pt_vec_rnx_tmp_bytes(&ct_infos, &ct_infos, &ctx.meta());
    let const_bytes = ctx
        .module
        .ckks_dot_product_pt_const_tmp_bytes(&ct_infos, &ct_infos, &ctx.meta());
    let bytes = ct_bytes.max(pt_znx_bytes).max(pt_rnx_bytes).max(const_bytes);
    ScratchOwned::<BE>::alloc(ctx.scratch_size.max(bytes))
}

fn scaled<F: TestScalar>(v: &[F], scale: F) -> Vec<F> {
    v.iter().copied().map(|x| x * scale).collect()
}

fn three_vectors<F: TestScalar>(ctx: &TestContext<impl Backend, F>) -> [(Vec<F>, Vec<F>); 3] {
    let s = F::from_f64(0.5).unwrap();
    [
        (scaled(&ctx.re1, s), scaled(&ctx.im1, s)),
        (scaled(&ctx.re2, s), scaled(&ctx.im2, s)),
        (scaled(&ctx.re1, s), scaled(&ctx.im2, s)),
    ]
}

fn cmul_acc<F: TestScalar>(acc_re: &mut [F], acc_im: &mut [F], a_re: &[F], a_im: &[F], b_re: &[F], b_im: &[F]) {
    for i in 0..acc_re.len() {
        let pr = a_re[i] * b_re[i] - a_im[i] * b_im[i];
        let pi = a_re[i] * b_im[i] + a_im[i] * b_re[i];
        acc_re[i] = acc_re[i] + pr;
        acc_im[i] = acc_im[i] + pi;
    }
}

fn cmul_scalar_acc<F: TestScalar>(acc_re: &mut [F], acc_im: &mut [F], a_re: &[F], a_im: &[F], c_re: F, c_im: F) {
    for i in 0..acc_re.len() {
        let pr = a_re[i] * c_re - a_im[i] * c_im;
        let pi = a_re[i] * c_im + a_im[i] * c_re;
        acc_re[i] = acc_re[i] + pr;
        acc_im[i] = acc_im[i] + pi;
    }
}

pub fn test_dot_product_ct_aligned<BE: Backend, F: TestScalar>(ctx: &TestContext<BE, F>) {
    let mut scratch = alloc_scratch(ctx);
    let a_vecs = three_vectors(ctx);
    let b_vecs = three_vectors(ctx);

    let m = ctx.re1.len();
    let mut want_re = vec![F::zero(); m];
    let mut want_im = vec![F::zero(); m];
    for i in 0..N {
        cmul_acc(
            &mut want_re,
            &mut want_im,
            &a_vecs[i].0,
            &a_vecs[i].1,
            &b_vecs[i].0,
            &b_vecs[i].1,
        );
    }

    let a_cts: Vec<_> = a_vecs
        .iter()
        .map(|(re, im)| ctx.encrypt(ctx.max_k(), re, im, scratch.borrow()))
        .collect();
    let b_cts: Vec<_> = b_vecs
        .iter()
        .map(|(re, im)| ctx.encrypt(ctx.max_k(), re, im, scratch.borrow()))
        .collect();
    let a_refs: Vec<&_> = a_cts.iter().collect();
    let b_refs: Vec<&_> = b_cts.iter().collect();

    let mut ct_res = ctx.alloc_ct(ctx.max_k());
    ctx.module
        .ckks_dot_product_ct(&mut ct_res, &a_refs, &b_refs, ctx.tsk(), scratch.borrow())
        .unwrap();
    ctx.assert_decrypt_precision("dot_product_ct_aligned", &ct_res, &want_re, &want_im, scratch.borrow());
}

/// One of the `a`-side ciphertexts is encrypted at a lower `log_budget`.
pub fn test_dot_product_ct_unaligned<BE: Backend, F: TestScalar>(ctx: &TestContext<BE, F>) {
    let mut scratch = alloc_scratch(ctx);
    let a_vecs = three_vectors(ctx);
    let b_vecs = three_vectors(ctx);

    let m = ctx.re1.len();
    let mut want_re = vec![F::zero(); m];
    let mut want_im = vec![F::zero(); m];
    for i in 0..N {
        cmul_acc(
            &mut want_re,
            &mut want_im,
            &a_vecs[i].0,
            &a_vecs[i].1,
            &b_vecs[i].0,
            &b_vecs[i].1,
        );
    }

    let smaller_k = ctx.max_k() - ctx.base2k().as_usize() + 1;
    let a_cts: Vec<_> = a_vecs
        .iter()
        .enumerate()
        .map(|(i, (re, im))| {
            let k = if i == 1 { smaller_k } else { ctx.max_k() };
            ctx.encrypt(k, re, im, scratch.borrow())
        })
        .collect();
    let b_cts: Vec<_> = b_vecs
        .iter()
        .map(|(re, im)| ctx.encrypt(ctx.max_k(), re, im, scratch.borrow()))
        .collect();
    let a_refs: Vec<&_> = a_cts.iter().collect();
    let b_refs: Vec<&_> = b_cts.iter().collect();

    let mut ct_res = ctx.alloc_ct(ctx.max_k());
    ctx.module
        .ckks_dot_product_ct(&mut ct_res, &a_refs, &b_refs, ctx.tsk(), scratch.borrow())
        .unwrap();
    ctx.assert_decrypt_precision("dot_product_ct_unaligned", &ct_res, &want_re, &want_im, scratch.borrow());
}

pub fn test_dot_product_ct_unaligned_b<BE: Backend, F: TestScalar>(ctx: &TestContext<BE, F>) {
    let mut scratch = alloc_scratch(ctx);
    let a_vecs = three_vectors(ctx);
    let b_vecs = three_vectors(ctx);

    let m = ctx.re1.len();
    let mut want_re = vec![F::zero(); m];
    let mut want_im = vec![F::zero(); m];
    for i in 0..N {
        cmul_acc(
            &mut want_re,
            &mut want_im,
            &a_vecs[i].0,
            &a_vecs[i].1,
            &b_vecs[i].0,
            &b_vecs[i].1,
        );
    }

    let smaller_k = ctx.max_k() - ctx.base2k().as_usize() + 1;
    let a_cts: Vec<_> = a_vecs
        .iter()
        .map(|(re, im)| ctx.encrypt(ctx.max_k(), re, im, scratch.borrow()))
        .collect();
    let b_cts: Vec<_> = b_vecs
        .iter()
        .enumerate()
        .map(|(i, (re, im))| {
            let k = if i == 1 { smaller_k } else { ctx.max_k() };
            ctx.encrypt(k, re, im, scratch.borrow())
        })
        .collect();
    let a_refs: Vec<&_> = a_cts.iter().collect();
    let b_refs: Vec<&_> = b_cts.iter().collect();

    let mut ct_res = ctx.alloc_ct(ctx.max_k());
    ctx.module
        .ckks_dot_product_ct(&mut ct_res, &a_refs, &b_refs, ctx.tsk(), scratch.borrow())
        .unwrap();
    ctx.assert_decrypt_precision("dot_product_ct_unaligned_b", &ct_res, &want_re, &want_im, scratch.borrow());
}

pub fn test_dot_product_ct_delta_log_delta<BE: Backend, F: TestScalar>(ctx: &TestContext<BE, F>) {
    let mut scratch = alloc_scratch(ctx);
    let half = F::from_f64(0.5).unwrap();
    let low_log_delta = ctx.meta().log_delta - DELTA_LOG_DECIMAL;
    let low_prec = ctx.precision_at(low_log_delta);
    let (a_hi_re, a_hi_im) = ctx.quantized_vector(TestVector::First, ctx.meta().log_delta);
    let (b_hi_re, b_hi_im) = ctx.quantized_vector(TestVector::Second, ctx.meta().log_delta);
    let (b_lo_re, b_lo_im) = ctx.quantized_vector(TestVector::Second, low_log_delta);
    let a_vecs = [
        (scaled(&a_hi_re, half), scaled(&a_hi_im, half)),
        (scaled(&a_hi_re, half), scaled(&a_hi_im, half)),
        (scaled(&a_hi_re, half), scaled(&a_hi_im, half)),
    ];
    let b_vecs = [
        (scaled(&b_lo_re, half), scaled(&b_lo_im, half)),
        (scaled(&b_hi_re, half), scaled(&b_hi_im, half)),
        (scaled(&b_hi_re, half), scaled(&b_hi_im, half)),
    ];

    let m = ctx.re1.len();
    let mut want_re = vec![F::zero(); m];
    let mut want_im = vec![F::zero(); m];
    for i in 0..N {
        cmul_acc(
            &mut want_re,
            &mut want_im,
            &a_vecs[i].0,
            &a_vecs[i].1,
            &b_vecs[i].0,
            &b_vecs[i].1,
        );
    }

    let a_cts: Vec<_> = a_vecs
        .iter()
        .map(|(re, im)| ctx.encrypt(ctx.max_k(), re, im, scratch.borrow()))
        .collect();
    let mut b_cts: Vec<_> = Vec::with_capacity(N);
    b_cts.push(ctx.encrypt_with_prec(
        ctx.max_k() - DELTA_LOG_DECIMAL,
        &b_vecs[0].0,
        &b_vecs[0].1,
        low_prec,
        scratch.borrow(),
    ));
    for (re, im) in &b_vecs[1..] {
        b_cts.push(ctx.encrypt(ctx.max_k(), re, im, scratch.borrow()));
    }
    let a_refs: Vec<&_> = a_cts.iter().collect();
    let b_refs: Vec<&_> = b_cts.iter().collect();

    let mut ct_res = ctx.alloc_ct(ctx.max_k());
    ctx.module
        .ckks_dot_product_ct(&mut ct_res, &a_refs, &b_refs, ctx.tsk(), scratch.borrow())
        .unwrap();
    ctx.assert_decrypt_precision_at_log_delta(
        "dot_product_ct_delta_log_delta",
        &ct_res,
        &want_re,
        &want_im,
        low_log_delta,
        scratch.borrow(),
    );
}

pub fn test_dot_product_ct_smaller_output<BE: Backend, F: TestScalar>(ctx: &TestContext<BE, F>) {
    let mut scratch = alloc_scratch(ctx);
    let a_vecs = three_vectors(ctx);
    let b_vecs = three_vectors(ctx);

    let m = ctx.re1.len();
    let mut want_re = vec![F::zero(); m];
    let mut want_im = vec![F::zero(); m];
    for i in 0..N {
        cmul_acc(
            &mut want_re,
            &mut want_im,
            &a_vecs[i].0,
            &a_vecs[i].1,
            &b_vecs[i].0,
            &b_vecs[i].1,
        );
    }

    let a_cts: Vec<_> = a_vecs
        .iter()
        .map(|(re, im)| ctx.encrypt(ctx.max_k(), re, im, scratch.borrow()))
        .collect();
    let b_cts: Vec<_> = b_vecs
        .iter()
        .map(|(re, im)| ctx.encrypt(ctx.max_k(), re, im, scratch.borrow()))
        .collect();
    let a_refs: Vec<&_> = a_cts.iter().collect();
    let b_refs: Vec<&_> = b_cts.iter().collect();

    let mut ct_res = ctx.alloc_ct(ctx.max_k() - ctx.base2k().as_usize() - 1);
    ctx.module
        .ckks_dot_product_ct(&mut ct_res, &a_refs, &b_refs, ctx.tsk(), scratch.borrow())
        .unwrap();
    ctx.assert_decrypt_precision("dot_product_ct smaller_output", &ct_res, &want_re, &want_im, scratch.borrow());
}

pub fn test_dot_product_pt_vec_znx_aligned<BE: Backend, F: TestScalar>(ctx: &TestContext<BE, F>) {
    let mut scratch = alloc_scratch(ctx);
    let a_vecs = three_vectors(ctx);
    let b_vecs = three_vectors(ctx);

    let m = ctx.re1.len();
    let mut want_re = vec![F::zero(); m];
    let mut want_im = vec![F::zero(); m];
    for i in 0..N {
        cmul_acc(
            &mut want_re,
            &mut want_im,
            &a_vecs[i].0,
            &a_vecs[i].1,
            &b_vecs[i].0,
            &b_vecs[i].1,
        );
    }

    let a_cts: Vec<_> = a_vecs
        .iter()
        .map(|(re, im)| ctx.encrypt(ctx.max_k(), re, im, scratch.borrow()))
        .collect();
    let pts: Vec<_> = b_vecs.iter().map(|(re, im)| ctx.encode_pt_znx(re, im)).collect();
    let a_refs: Vec<&_> = a_cts.iter().collect();
    let pt_refs: Vec<&_> = pts.iter().collect();

    let mut ct_res = ctx.alloc_ct(ctx.max_k());
    ctx.module
        .ckks_dot_product_pt_vec_znx(&mut ct_res, &a_refs, &pt_refs, scratch.borrow())
        .unwrap();
    ctx.assert_decrypt_precision(
        "dot_product_pt_vec_znx_aligned",
        &ct_res,
        &want_re,
        &want_im,
        scratch.borrow(),
    );
}

pub fn test_dot_product_pt_vec_rnx_aligned<BE: Backend, F: TestScalar>(ctx: &TestContext<BE, F>) {
    let mut scratch = alloc_scratch(ctx);
    let a_vecs = three_vectors(ctx);
    let b_vecs = three_vectors(ctx);

    let m = ctx.re1.len();
    let mut want_re = vec![F::zero(); m];
    let mut want_im = vec![F::zero(); m];
    for i in 0..N {
        cmul_acc(
            &mut want_re,
            &mut want_im,
            &a_vecs[i].0,
            &a_vecs[i].1,
            &b_vecs[i].0,
            &b_vecs[i].1,
        );
    }

    let a_cts: Vec<_> = a_vecs
        .iter()
        .map(|(re, im)| ctx.encrypt(ctx.max_k(), re, im, scratch.borrow()))
        .collect();
    let pts: Vec<_> = b_vecs.iter().map(|(re, im)| ctx.encode_pt_rnx(re, im)).collect();
    let a_refs: Vec<&_> = a_cts.iter().collect();
    let pt_refs: Vec<&_> = pts.iter().collect();

    let mut ct_res = ctx.alloc_ct(ctx.max_k());
    ctx.module
        .ckks_dot_product_pt_vec_rnx(&mut ct_res, &a_refs, &pt_refs, ctx.meta(), scratch.borrow())
        .unwrap();
    ctx.assert_decrypt_precision(
        "dot_product_pt_vec_rnx_aligned",
        &ct_res,
        &want_re,
        &want_im,
        scratch.borrow(),
    );
}

pub fn test_dot_product_const_znx_aligned<BE: Backend, F: TestScalar>(ctx: &TestContext<BE, F>) {
    let mut scratch = alloc_scratch(ctx);
    let a_vecs = three_vectors(ctx);

    let const_pairs: [(f64, f64); 3] = [(0.25, -0.125), (0.125, 0.0625), (-0.3125, 0.25)];
    let quantized: Vec<(F, F)> = const_pairs
        .iter()
        .map(|(r, i)| ctx.quantized_const(*r, *i, ctx.meta().log_delta))
        .collect();

    let m = ctx.re1.len();
    let mut want_re = vec![F::zero(); m];
    let mut want_im = vec![F::zero(); m];
    for i in 0..N {
        cmul_scalar_acc(
            &mut want_re,
            &mut want_im,
            &a_vecs[i].0,
            &a_vecs[i].1,
            quantized[i].0,
            quantized[i].1,
        );
    }

    let a_cts: Vec<_> = a_vecs
        .iter()
        .map(|(re, im)| ctx.encrypt(ctx.max_k(), re, im, scratch.borrow()))
        .collect();
    let cst_rnxs: Vec<CKKSPlaintextCstRnx<F>> = const_pairs.iter().map(|(r, i)| ctx.const_rnx(Some(*r), Some(*i))).collect();
    let cst_znxs: Vec<CKKSPlaintextCstZnx> = cst_rnxs.iter().map(|c| c.to_znx(ctx.base2k(), ctx.meta()).unwrap()).collect();

    let a_refs: Vec<&_> = a_cts.iter().collect();
    let cst_refs: Vec<&_> = cst_znxs.iter().collect();

    let mut ct_res = ctx.alloc_ct(ctx.max_k());
    ctx.module
        .ckks_dot_product_pt_const_znx(&mut ct_res, &a_refs, &cst_refs, scratch.borrow())
        .unwrap();
    ctx.assert_decrypt_precision("dot_product_const_znx_aligned", &ct_res, &want_re, &want_im, scratch.borrow());
}

pub fn test_dot_product_const_rnx_aligned<BE: Backend, F: TestScalar>(ctx: &TestContext<BE, F>) {
    let mut scratch = alloc_scratch(ctx);
    let a_vecs = three_vectors(ctx);

    let const_pairs: [(f64, f64); 3] = [(0.25, -0.125), (0.125, 0.0625), (-0.3125, 0.25)];
    let quantized: Vec<(F, F)> = const_pairs
        .iter()
        .map(|(r, i)| ctx.quantized_const(*r, *i, ctx.meta().log_delta))
        .collect();

    let m = ctx.re1.len();
    let mut want_re = vec![F::zero(); m];
    let mut want_im = vec![F::zero(); m];
    for i in 0..N {
        cmul_scalar_acc(
            &mut want_re,
            &mut want_im,
            &a_vecs[i].0,
            &a_vecs[i].1,
            quantized[i].0,
            quantized[i].1,
        );
    }

    let a_cts: Vec<_> = a_vecs
        .iter()
        .map(|(re, im)| ctx.encrypt(ctx.max_k(), re, im, scratch.borrow()))
        .collect();
    let csts: Vec<CKKSPlaintextCstRnx<F>> = const_pairs.iter().map(|(r, i)| ctx.const_rnx(Some(*r), Some(*i))).collect();

    let a_refs: Vec<&_> = a_cts.iter().collect();
    let cst_refs: Vec<&_> = csts.iter().collect();

    let mut ct_res = ctx.alloc_ct(ctx.max_k());
    ctx.module
        .ckks_dot_product_pt_const_rnx(&mut ct_res, &a_refs, &cst_refs, ctx.meta(), scratch.borrow())
        .unwrap();
    ctx.assert_decrypt_precision("dot_product_const_rnx_aligned", &ct_res, &want_re, &want_im, scratch.borrow());
}
