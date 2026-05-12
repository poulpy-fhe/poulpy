//! Composition tests: multi-step CKKS evaluation paths that combine primitives.

use super::helpers::{
    PT_PREC, TestContextBackend, TestContextModule, TestScalar, alloc_ct, alloc_scratch, assert_ckks_error,
    assert_decrypt_precision, ckks_decrypt_decode, ckks_encrypt, encode_and_upload_pt, gen_sk, gen_sk_with_raw, gen_tsk,
    quantized_const, test_vector_1, test_vector_2,
};
use crate::{
    CKKSCompositionError, CKKSInfos,
    layouts::CKKSPlaintext,
    leveled::api::{CKKSAddOps, CKKSMulOps},
};
use poulpy_hal::{
    api::{NegacyclicFFT, NegacyclicFFTNew, ScratchOwnedBorrow},
    layouts::{HostBytesBackend, Module},
};

use crate::{encoding::reim::Encoder, test_suite::CKKSTestParams};

fn constant<BE, F, E>(
    host_module: &Module<HostBytesBackend>,
    module: &Module<BE>,
    encoder: &Encoder<E>,
    params: &CKKSTestParams,
    c: (f64, f64),
    m: usize,
) -> CKKSPlaintext<BE::OwnedBuf>
where
    BE: TestContextBackend,
    F: TestScalar,
    E: NegacyclicFFT<F>,
{
    let re = vec![F::from_f64(c.0).unwrap(); m];
    let im = vec![F::from_f64(c.1).unwrap(); m];
    encode_and_upload_pt(host_module, module, encoder, params.base2k.into(), PT_PREC, &re, &im)
}

fn poly2_expected<F: TestScalar>(re1: &[F], im1: &[F], c0: (F, F), c1: (F, F), c2: (F, F)) -> (Vec<F>, Vec<F>) {
    let m = re1.len();
    let two = F::from_f64(2.0).unwrap();
    let (c0_re, c0_im) = c0;
    let (c1_re, c1_im) = c1;
    let (c2_re, c2_im) = c2;
    let want_re: Vec<F> = (0..m)
        .map(|j| {
            let x_re = re1[j];
            let x_im = im1[j];
            let x2_re = x_re * x_re - x_im * x_im;
            let x2_im = two * x_re * x_im;
            c0_re + c1_re * x_re - c1_im * x_im + c2_re * x2_re - c2_im * x2_im
        })
        .collect();
    let want_im: Vec<F> = (0..m)
        .map(|j| {
            let x_re = re1[j];
            let x_im = im1[j];
            let x2_re = x_re * x_re - x_im * x_im;
            let x2_im = two * x_re * x_im;
            c0_im + c1_re * x_im + c1_im * x_re + c2_re * x2_im + c2_im * x2_re
        })
        .collect();
    (want_re, want_im)
}

fn same_offset_expected<F: TestScalar>(re1: &[F], im1: &[F], c1: (F, F), c2: (F, F)) -> (Vec<F>, Vec<F>) {
    let m = re1.len();
    let coeff_re = c1.0 + c2.0;
    let coeff_im = c1.1 + c2.1;
    let want_re: Vec<F> = (0..m).map(|j| coeff_re * re1[j] - coeff_im * im1[j]).collect();
    let want_im: Vec<F> = (0..m).map(|j| coeff_re * im1[j] + coeff_im * re1[j]).collect();
    (want_re, want_im)
}

fn mul_by_y_expected<F: TestScalar>(
    re1: &[F],
    im1: &[F],
    re2: &[F],
    im2: &[F],
    c0: (F, F),
    c1: (F, F),
    c2: (F, F),
) -> (Vec<F>, Vec<F>) {
    let m = re1.len();
    let (poly_re, poly_im) = poly2_expected(re1, im1, c0, c1, c2);
    let want_re: Vec<F> = (0..m).map(|j| poly_re[j] * re2[j] - poly_im[j] * im2[j]).collect();
    let want_im: Vec<F> = (0..m).map(|j| poly_re[j] * im2[j] + poly_im[j] * re2[j]).collect();
    (want_re, want_im)
}

/// Adding two plaintext-scaled copies of the same ciphertext stays accurate.
pub fn test_linear_sum<BE, F, E>(params: CKKSTestParams, module: &Module<BE>, host_module: &Module<HostBytesBackend>)
where
    BE: TestContextBackend,
    Module<BE>: TestContextModule<BE>,
    F: TestScalar,
    E: NegacyclicFFT<F> + NegacyclicFFTNew<F>,
{
    let m = params.n / 2;
    let encoder = Encoder::<E>::new(m).unwrap();
    let (re1, im1) = test_vector_1::<F>(m);
    let sk = gen_sk(&params, module, host_module, [0u8; 32]);
    let mut scratch = alloc_scratch(&params, module);

    let c1 = (0.625, -0.125);
    let c2 = (-0.375, 0.25);
    let c1_q = quantized_const::<F>(c1.0, c1.1, PT_PREC.log_delta);
    let c2_q = quantized_const::<F>(c2.0, c2.1, PT_PREC.log_delta);
    let pt1 = constant::<BE, F, E>(host_module, module, &encoder, &params, c1, m);
    let pt2 = constant::<BE, F, E>(host_module, module, &encoder, &params, c2, m);
    let (want_re, want_im) = same_offset_expected(&re1, &im1, c1_q, c2_q);

    let x = ckks_encrypt(
        &params,
        module,
        host_module,
        &encoder,
        &sk,
        params.k,
        &re1,
        &im1,
        &mut scratch.borrow(),
    );
    let mut term1 = alloc_ct(&params, module, x.log_budget());
    let mut term2 = alloc_ct(&params, module, x.log_budget());
    module
        .ckks_mul_pt_vec_into(&mut term1, &x, &pt1, &mut scratch.borrow())
        .unwrap();
    module
        .ckks_mul_pt_vec_into(&mut term2, &x, &pt2, &mut scratch.borrow())
        .unwrap();

    assert_eq!(
        term1.log_budget(),
        term2.log_budget(),
        "linear branches should remain aligned"
    );
    module.ckks_add_assign(&mut term1, &term2, &mut scratch.borrow()).unwrap();

    assert_decrypt_precision(
        "linear_sum",
        &params,
        module,
        &encoder,
        &term1,
        &sk,
        &want_re,
        &want_im,
        &mut scratch.borrow(),
    );
}

/// A mixed `c1*x + c2*x^2` composition remains decryptable and accurate.
pub fn test_poly2_sum<BE, F, E>(params: CKKSTestParams, module: &Module<BE>, host_module: &Module<HostBytesBackend>)
where
    BE: TestContextBackend,
    Module<BE>: TestContextModule<BE>,
    F: TestScalar,
    E: NegacyclicFFT<F> + NegacyclicFFTNew<F>,
{
    let m = params.n / 2;
    let encoder = Encoder::<E>::new(m).unwrap();
    let (re1, im1) = test_vector_1::<F>(m);
    let (sk_raw, sk) = gen_sk_with_raw(&params, module, host_module, [0u8; 32]);
    let mut scratch = alloc_scratch(&params, module);
    let tsk = gen_tsk(&params, module, &sk_raw, &mut scratch.borrow());

    let c1 = (0.625, -0.125);
    let c2 = (-0.375, 0.25);
    let c1_q = quantized_const::<F>(c1.0, c1.1, PT_PREC.log_delta);
    let c2_q = quantized_const::<F>(c2.0, c2.1, PT_PREC.log_delta);
    let pt1 = constant::<BE, F, E>(host_module, module, &encoder, &params, c1, m);
    let pt2 = constant::<BE, F, E>(host_module, module, &encoder, &params, c2, m);
    let (want_re, want_im) = poly2_expected(&re1, &im1, (F::zero(), F::zero()), c1_q, c2_q);

    let x = ckks_encrypt(
        &params,
        module,
        host_module,
        &encoder,
        &sk,
        params.k,
        &re1,
        &im1,
        &mut scratch.borrow(),
    );
    let mut x2 = alloc_ct(&params, module, x.log_budget());
    module.ckks_square_into(&mut x2, &x, &tsk, &mut scratch.borrow()).unwrap();

    let mut term1 = alloc_ct(&params, module, x.log_budget());
    let mut term2 = alloc_ct(&params, module, x2.log_budget());
    module
        .ckks_mul_pt_vec_into(&mut term1, &x, &pt1, &mut scratch.borrow())
        .unwrap();
    module
        .ckks_mul_pt_vec_into(&mut term2, &x2, &pt2, &mut scratch.borrow())
        .unwrap();

    assert!(
        term1.log_budget() > term2.log_budget(),
        "x^2 branch should consume more precision"
    );
    let mut sum = alloc_ct(&params, module, term2.effective_k());
    module.ckks_add_into(&mut sum, &term1, &term2, &mut scratch.borrow()).unwrap();

    assert_decrypt_precision(
        "poly2_sum",
        &params,
        module,
        &encoder,
        &sum,
        &sk,
        &want_re,
        &want_im,
        &mut scratch.borrow(),
    );
}

/// Adding a constant plaintext to `c1*x + c2*x^2` keeps the expected value.
pub fn test_poly2_sum_with_const<BE, F, E>(params: CKKSTestParams, module: &Module<BE>, host_module: &Module<HostBytesBackend>)
where
    BE: TestContextBackend,
    Module<BE>: TestContextModule<BE>,
    F: TestScalar,
    E: NegacyclicFFT<F> + NegacyclicFFTNew<F>,
{
    let m = params.n / 2;
    let encoder = Encoder::<E>::new(m).unwrap();
    let (re1, im1) = test_vector_1::<F>(m);
    let (sk_raw, sk) = gen_sk_with_raw(&params, module, host_module, [0u8; 32]);
    let mut scratch = alloc_scratch(&params, module);
    let tsk = gen_tsk(&params, module, &sk_raw, &mut scratch.borrow());

    let c0 = (0.125, -0.0625);
    let c1 = (0.625, -0.125);
    let c2 = (-0.375, 0.25);
    let c0_q = quantized_const::<F>(c0.0, c0.1, PT_PREC.log_delta);
    let c1_q = quantized_const::<F>(c1.0, c1.1, PT_PREC.log_delta);
    let c2_q = quantized_const::<F>(c2.0, c2.1, PT_PREC.log_delta);
    let pt0 = constant::<BE, F, E>(host_module, module, &encoder, &params, c0, m);
    let pt1 = constant::<BE, F, E>(host_module, module, &encoder, &params, c1, m);
    let pt2 = constant::<BE, F, E>(host_module, module, &encoder, &params, c2, m);
    let (want_re, want_im) = poly2_expected(&re1, &im1, c0_q, c1_q, c2_q);

    let x = ckks_encrypt(
        &params,
        module,
        host_module,
        &encoder,
        &sk,
        params.k,
        &re1,
        &im1,
        &mut scratch.borrow(),
    );
    let mut x2 = alloc_ct(&params, module, x.log_budget());
    module.ckks_square_into(&mut x2, &x, &tsk, &mut scratch.borrow()).unwrap();

    let mut term1 = alloc_ct(&params, module, x.log_budget());
    let mut term2 = alloc_ct(&params, module, x2.log_budget());
    module
        .ckks_mul_pt_vec_into(&mut term1, &x, &pt1, &mut scratch.borrow())
        .unwrap();
    module
        .ckks_mul_pt_vec_into(&mut term2, &x2, &pt2, &mut scratch.borrow())
        .unwrap();
    let mut poly = alloc_ct(&params, module, term2.effective_k());
    module
        .ckks_add_into(&mut poly, &term1, &term2, &mut scratch.borrow())
        .unwrap();
    module.ckks_add_pt_vec_assign(&mut poly, &pt0, &mut scratch.borrow()).unwrap();

    assert_decrypt_precision(
        "poly2_sum_with_const",
        &params,
        module,
        &encoder,
        &poly,
        &sk,
        &want_re,
        &want_im,
        &mut scratch.borrow(),
    );
}

/// Evaluates `y * (c0 + c1*x + c2*x^2)` with encrypted `x` and `y`.
pub fn test_poly2_mul<BE, F, E>(params: CKKSTestParams, module: &Module<BE>, host_module: &Module<HostBytesBackend>)
where
    BE: TestContextBackend,
    Module<BE>: TestContextModule<BE>,
    F: TestScalar,
    E: NegacyclicFFT<F> + NegacyclicFFTNew<F>,
{
    let m = params.n / 2;
    let encoder = Encoder::<E>::new(m).unwrap();
    let (re1, im1) = test_vector_1::<F>(m);
    let (re2, im2) = test_vector_2::<F>(m);
    let (sk_raw, sk) = gen_sk_with_raw(&params, module, host_module, [0u8; 32]);
    let mut scratch = alloc_scratch(&params, module);
    let tsk = gen_tsk(&params, module, &sk_raw, &mut scratch.borrow());

    let c0 = (0.125, -0.0625);
    let c1 = (0.625, -0.125);
    let c2 = (-0.375, 0.25);
    let c0_q = quantized_const::<F>(c0.0, c0.1, PT_PREC.log_delta);
    let c1_q = quantized_const::<F>(c1.0, c1.1, PT_PREC.log_delta);
    let c2_q = quantized_const::<F>(c2.0, c2.1, PT_PREC.log_delta);
    let pt0 = constant::<BE, F, E>(host_module, module, &encoder, &params, c0, m);
    let pt1 = constant::<BE, F, E>(host_module, module, &encoder, &params, c1, m);
    let pt2 = constant::<BE, F, E>(host_module, module, &encoder, &params, c2, m);
    let (want_re, want_im) = mul_by_y_expected(&re1, &im1, &re2, &im2, c0_q, c1_q, c2_q);

    let x = ckks_encrypt(
        &params,
        module,
        host_module,
        &encoder,
        &sk,
        params.k,
        &re1,
        &im1,
        &mut scratch.borrow(),
    );
    let y = ckks_encrypt(
        &params,
        module,
        host_module,
        &encoder,
        &sk,
        params.k,
        &re2,
        &im2,
        &mut scratch.borrow(),
    );
    let mut x2 = alloc_ct(&params, module, x.log_budget());
    module.ckks_square_into(&mut x2, &x, &tsk, &mut scratch.borrow()).unwrap();

    let mut term1 = alloc_ct(&params, module, x.log_budget());
    let mut term2 = alloc_ct(&params, module, x2.log_budget());
    module
        .ckks_mul_pt_vec_into(&mut term1, &x, &pt1, &mut scratch.borrow())
        .unwrap();
    module
        .ckks_mul_pt_vec_into(&mut term2, &x2, &pt2, &mut scratch.borrow())
        .unwrap();
    let mut poly = alloc_ct(&params, module, term2.effective_k());
    module
        .ckks_add_into(&mut poly, &term1, &term2, &mut scratch.borrow())
        .unwrap();
    module.ckks_add_pt_vec_assign(&mut poly, &pt0, &mut scratch.borrow()).unwrap();

    let mut res = alloc_ct(&params, module, params.k);
    module
        .ckks_mul_into(&mut res, &y, &poly, &tsk, &mut scratch.borrow())
        .unwrap();

    assert_decrypt_precision(
        "poly2_mul",
        &params,
        module,
        &encoder,
        &res,
        &sk,
        &want_re,
        &want_im,
        &mut scratch.borrow(),
    );
}

/// Repeated squaring on unit-circle slots should exhaust HE capacity before it blows up numerically.
pub fn test_repeated_square_exhausts_capacity<BE, F, E>(
    params: CKKSTestParams,
    module: &Module<BE>,
    host_module: &Module<HostBytesBackend>,
) where
    BE: TestContextBackend,
    Module<BE>: TestContextModule<BE>,
    F: TestScalar,
    E: NegacyclicFFT<F> + NegacyclicFFTNew<F>,
{
    let m = params.n / 2;
    let encoder = Encoder::<E>::new(m).unwrap();
    let (re1, im1) = test_vector_1::<F>(m);
    let (sk_raw, sk) = gen_sk_with_raw(&params, module, host_module, [0u8; 32]);
    let mut scratch = alloc_scratch(&params, module);
    let tsk = gen_tsk(&params, module, &sk_raw, &mut scratch.borrow());

    let mut ct = ckks_encrypt(
        &params,
        module,
        host_module,
        &encoder,
        &sk,
        params.k,
        &re1,
        &im1,
        &mut scratch.borrow(),
    );
    let mut squares = 0usize;

    while ct.log_budget() >= ct.log_delta() {
        let prev_log_budget = ct.log_budget();
        let prev_log_delta = ct.log_delta();
        let next_k = ct.effective_k() - ct.log_delta();
        let mut next = alloc_ct(&params, module, next_k);
        module.ckks_square_into(&mut next, &ct, &tsk, &mut scratch.borrow()).unwrap();
        assert_eq!(
            next.log_delta(),
            prev_log_delta,
            "square should preserve log_delta across repeated squaring"
        );
        assert_eq!(
            next.log_budget(),
            prev_log_budget - prev_log_delta,
            "square should consume exactly one log_delta chunk of HE capacity"
        );
        ct = next;
        squares += 1;
    }

    assert!(squares > 0, "expected at least one square");
    assert!(
        ct.log_budget() < ct.log_delta(),
        "expected squaring to consume all HE capacity"
    );
    let (got_re, got_im) = ckks_decrypt_decode::<BE, F, E>(&params, module, &encoder, &ct, &sk, &mut scratch.borrow());
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

    let mut no_capacity = alloc_ct(&params, module, params.k);
    let err = module
        .ckks_square_into(&mut no_capacity, &ct, &tsk, &mut scratch.borrow())
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
