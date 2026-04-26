//! Tests for `CKKSAddManyOps::ckks_add_many`.
//!
//! # Test inventory
//!
//! | Function | Path exercised |
//! |----------|----------------|
//! | [`test_add_many_aligned`] | all inputs at the same `log_budget` / `log_delta` |
//! | [`test_add_many_single_smaller_output`] | one input into a narrower output |
//! | [`test_add_many_unaligned_log_budget`] | one input rescaled by one limb |
//! | [`test_add_many_delta_log_delta`] | inputs at different `log_delta` |
//! | [`test_add_many_smaller_output`] | output narrower than inputs (`offset > 0`) |

use poulpy_hal::api::ScratchOwnedBorrow;

use crate::{CKKSInfos, leveled::api::CKKSAddManyOps, leveled::tests::test_suite::helpers::assert_ct_meta};

use super::helpers::{TestAddBackend as Backend, TestContext, TestScalar, TestVector};

const DELTA_LOG_DECIMAL: usize = 12;
const N: usize = 5;

type Terms<F> = (Vec<(Vec<F>, Vec<F>)>, Vec<F>, Vec<F>);

/// `n` deterministic plaintext vectors built as small linear combinations of
/// the context's two base vectors. Per-term magnitude is kept to `≤ 1/(2n)`
/// so the sum stays inside the torus fundamental domain.
fn build_terms<F: TestScalar>(ctx: &TestContext<impl Backend, F>, n: usize) -> Terms<F> {
    let m: usize = ctx.re1.len();
    let mut terms: Vec<(Vec<F>, Vec<F>)> = Vec::with_capacity(n);
    let mut want_re: Vec<F> = vec![F::zero(); m];
    let mut want_im: Vec<F> = vec![F::zero(); m];
    let scale = F::from_f64(1.0 / (2.0 * n as f64)).unwrap();
    for k in 0..n {
        let alpha = F::from_f64((k as f64 + 1.0) / (n as f64 + 1.0)).unwrap();
        let beta = F::from_f64(1.0).unwrap() - alpha;
        let re: Vec<F> = (0..m).map(|i| (alpha * ctx.re1[i] + beta * ctx.re2[i]) * scale).collect();
        let im: Vec<F> = (0..m).map(|i| (alpha * ctx.im1[i] + beta * ctx.im2[i]) * scale).collect();
        for i in 0..m {
            want_re[i] = want_re[i] + re[i];
            want_im[i] = want_im[i] + im[i];
        }
        terms.push((re, im));
    }
    (terms, want_re, want_im)
}

/// Aligned ct+ct+...+ct sum at the max encryption size (fast-path accumulator).
pub fn test_add_many_aligned<BE: Backend, F: TestScalar>(ctx: &TestContext<BE, F>) {
    let mut scratch = ctx.alloc_scratch();
    let (terms, want_re, want_im) = build_terms(ctx, N);
    let cts: Vec<_> = terms
        .iter()
        .map(|(re, im)| ctx.encrypt(ctx.max_k(), re, im, scratch.borrow()))
        .collect();
    let ct_refs: Vec<&_> = cts.iter().collect();
    let mut ct_res = ctx.alloc_ct(ctx.max_k());
    ctx.module.ckks_add_many(&mut ct_res, &ct_refs, scratch.borrow()).unwrap();
    let expected_log_delta: usize = cts.iter().map(|c| c.log_delta()).max().unwrap();
    let expected_log_budget: usize = cts.iter().map(|c| c.log_budget()).min().unwrap();
    assert_ct_meta("add_many_aligned", &ct_res, expected_log_delta, expected_log_budget);
    ctx.assert_decrypt_precision("add_many_aligned", &ct_res, &want_re, &want_im, scratch.borrow());
}

pub fn test_add_many_single_smaller_output<BE: Backend, F: TestScalar>(ctx: &TestContext<BE, F>) {
    let mut scratch = ctx.alloc_scratch();
    let ct = ctx.encrypt(ctx.max_k(), &ctx.re1, &ctx.im1, scratch.borrow());
    let ct_refs = vec![&ct];
    let mut ct_res = ctx.alloc_ct(ctx.max_k() - ctx.base2k().as_usize() - 1);
    ctx.module.ckks_add_many(&mut ct_res, &ct_refs, scratch.borrow()).unwrap();
    ctx.assert_decrypt_precision(
        "add_many_single_smaller_output",
        &ct_res,
        &ctx.re1,
        &ctx.im1,
        scratch.borrow(),
    );
}

/// One input encrypted at a lower `log_budget`, forcing the others to be
/// shifted down to align before the sum is performed.
pub fn test_add_many_unaligned_log_budget<BE: Backend, F: TestScalar>(ctx: &TestContext<BE, F>) {
    let mut scratch = ctx.alloc_scratch();
    let (terms, want_re, want_im) = build_terms(ctx, N);
    let smaller_k = ctx.max_k() - ctx.base2k().as_usize() + 1;
    let cts: Vec<_> = terms
        .iter()
        .enumerate()
        .map(|(i, (re, im))| {
            let k = if i == 1 { smaller_k } else { ctx.max_k() };
            ctx.encrypt(k, re, im, scratch.borrow())
        })
        .collect();
    let ct_refs: Vec<&_> = cts.iter().collect();
    let mut ct_res = ctx.alloc_ct(ctx.max_k());
    ctx.module.ckks_add_many(&mut ct_res, &ct_refs, scratch.borrow()).unwrap();
    ctx.assert_decrypt_precision("add_many unaligned_log_budget", &ct_res, &want_re, &want_im, scratch.borrow());
}

/// One input encoded at a lower `log_delta`. The sum's precision is
/// bounded by the least precise summand.
pub fn test_add_many_delta_log_delta<BE: Backend, F: TestScalar>(ctx: &TestContext<BE, F>) {
    let mut scratch = ctx.alloc_scratch();
    let low_log_delta = ctx.meta().log_delta - DELTA_LOG_DECIMAL;
    let low_prec = ctx.precision_at(low_log_delta);
    let (hi_re, hi_im) = ctx.quantized_vector(TestVector::First, ctx.meta().log_delta);
    let (lo_re, lo_im) = ctx.quantized_vector(TestVector::Second, low_log_delta);

    let scale = F::from_f64(1.0 / (2.0 * N as f64)).unwrap();
    let hi_scaled: (Vec<F>, Vec<F>) = (
        hi_re.iter().copied().map(|x| x * scale).collect(),
        hi_im.iter().copied().map(|x| x * scale).collect(),
    );
    let lo_scaled: (Vec<F>, Vec<F>) = (
        lo_re.iter().copied().map(|x| x * scale).collect(),
        lo_im.iter().copied().map(|x| x * scale).collect(),
    );

    // First term at the low precision, rest at the high precision.
    let ct_low = ctx.encrypt_with_prec(
        ctx.max_k() - DELTA_LOG_DECIMAL,
        &lo_scaled.0,
        &lo_scaled.1,
        low_prec,
        scratch.borrow(),
    );
    let cts_hi: Vec<_> = (0..N - 1)
        .map(|_| ctx.encrypt(ctx.max_k(), &hi_scaled.0, &hi_scaled.1, scratch.borrow()))
        .collect();
    let mut ct_refs: Vec<&_> = Vec::with_capacity(N);
    ct_refs.push(&ct_low);
    for c in &cts_hi {
        ct_refs.push(c);
    }

    let m = ctx.re1.len();
    let mut want_re: Vec<F> = vec![F::zero(); m];
    let mut want_im: Vec<F> = vec![F::zero(); m];
    for i in 0..m {
        want_re[i] = lo_scaled.0[i] + hi_scaled.0[i] * F::from_f64((N - 1) as f64).unwrap();
        want_im[i] = lo_scaled.1[i] + hi_scaled.1[i] * F::from_f64((N - 1) as f64).unwrap();
    }

    let mut ct_res = ctx.alloc_ct(ctx.max_k());
    ctx.module.ckks_add_many(&mut ct_res, &ct_refs, scratch.borrow()).unwrap();
    ctx.assert_decrypt_precision_at_log_delta(
        "add_many delta_log_delta",
        &ct_res,
        &want_re,
        &want_im,
        low_log_delta,
        scratch.borrow(),
    );
}

/// Output allocated one limb narrower than the inputs: forces a pre-shift.
pub fn test_add_many_smaller_output<BE: Backend, F: TestScalar>(ctx: &TestContext<BE, F>) {
    let mut scratch = ctx.alloc_scratch();
    let (terms, want_re, want_im) = build_terms(ctx, N);
    let cts: Vec<_> = terms
        .iter()
        .map(|(re, im)| ctx.encrypt(ctx.max_k(), re, im, scratch.borrow()))
        .collect();
    let ct_refs: Vec<&_> = cts.iter().collect();
    let mut ct_res = ctx.alloc_ct(ctx.max_k() - ctx.base2k().as_usize() - 1);
    ctx.module.ckks_add_many(&mut ct_res, &ct_refs, scratch.borrow()).unwrap();
    ctx.assert_decrypt_precision("add_many smaller_output", &ct_res, &want_re, &want_im, scratch.borrow());
}
