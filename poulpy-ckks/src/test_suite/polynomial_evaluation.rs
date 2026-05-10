use poulpy_core::layouts::LWEInfos;
use poulpy_hal::api::{NegacyclicFFT, ScratchOwnedBorrow};

use crate::{
    layouts::{CKKSCiphertext, CKKSModuleAlloc, CKKSPlaintext, CKKSPlaintextVecHostCodec},
    leveled::api::{CKKSMulOps, PolynomialEvaluation},
    polynomial::{BSGSPolynomial, Basis, Polynomial, PowerBasis, chebyshev_interpolate},
};

use super::helpers::{TestBackend, TestContext, TestPolynomialEvaluationBackend, TestScalar};

fn scale_add<F: TestScalar>(acc: &mut [F], src: &[F], scale: F) {
    for (a, s) in acc.iter_mut().zip(src.iter()) {
        *a = *a + *s * scale;
    }
}

fn pointwise_mul<F: TestScalar>(a: &[F], b: &[F]) -> Vec<F> {
    a.iter().zip(b.iter()).map(|(x, y)| *x * *y).collect()
}

fn pointwise_pow<F: TestScalar>(x: &[F], power: usize) -> Vec<F> {
    let mut acc = vec![F::one(); x.len()];
    for _ in 0..power {
        for (a, x_i) in acc.iter_mut().zip(x.iter()) {
            *a = *a * *x_i;
        }
    }
    acc
}

fn chebyshev_values<F: TestScalar>(x: &[F], degree: usize) -> Vec<Vec<F>> {
    let two = F::one() + F::one();
    let mut values = Vec::with_capacity(degree + 1);
    values.push(vec![F::one(); x.len()]);
    if degree == 0 {
        return values;
    }
    values.push(x.to_vec());
    for i in 2..=degree {
        let next = x
            .iter()
            .zip(values[i - 1].iter())
            .zip(values[i - 2].iter())
            .map(|((&x_i, &t_prev), &t_prev_prev)| two * x_i * t_prev - t_prev_prev)
            .collect();
        values.push(next);
    }
    values
}

fn chebyshev_value<F: TestScalar>(x: F, degree: usize) -> F {
    if degree == 0 {
        return F::one();
    }
    if degree == 1 {
        return x;
    }
    let two = F::one() + F::one();
    let mut t_prev_prev = F::one();
    let mut t_prev = x;
    for _ in 2..=degree {
        let t = two * x * t_prev - t_prev_prev;
        t_prev_prev = t_prev;
        t_prev = t;
    }
    t_prev
}

fn eval_encoded_bsgs_chebyshev<F: TestScalar>(poly: &BSGSPolynomial<CKKSPlaintext<Vec<u8>>>, x: F) -> F {
    #[derive(Clone, Copy)]
    struct Step<F> {
        degree: usize,
        value: F,
    }

    let mut steps = Vec::with_capacity(poly.baby_steps.len());
    for (degree, coeffs_pt) in poly.baby_degrees.iter().copied().zip(poly.baby_steps.iter()) {
        let mut coeffs = vec![F::zero(); coeffs_pt.n().as_usize()];
        coeffs_pt.decode_host_floats(&mut coeffs).unwrap();
        let value = coeffs
            .iter()
            .take(degree + 1)
            .enumerate()
            .fold(F::zero(), |acc, (i, &c)| acc + c * chebyshev_value(x, i));
        steps.push(Step { degree, value });
    }

    while steps.len() > 1 {
        let mut i = 0;
        while i < steps.len() {
            let is_last = i + 1 == steps.len();
            if !is_last && steps[i].degree == steps[i + 1].degree {
                let gsp = (steps[i].degree + 1).next_power_of_two();
                let low = steps.remove(i);
                steps[i].value = steps[i].value * chebyshev_value(x, gsp) + low.value;
                steps[i].degree = 2 * gsp - 1;
            } else if is_last && i > 0 {
                steps[i].degree = steps[i - 1].degree;
            }
            i += 1;
        }
    }

    steps[0].value
}

fn coeff_pt<BE: TestBackend, F: TestScalar, E: NegacyclicFFT<F>>(
    ctx: &TestContext<BE, F, E>,
    coeffs: &[F],
) -> CKKSPlaintext<Vec<u8>> {
    let mut pt = ctx
        .host_module
        .ckks_pt_coeffs_alloc(coeffs.len(), ctx.base2k(), ctx.meta_pt());
    pt.encode_host_floats(coeffs).unwrap();
    pt
}

// ── Power-basis test ─────────────────────────────────────────────────────────

pub fn test_power_basis_populate_degree7<BE: TestPolynomialEvaluationBackend, F: TestScalar, E: NegacyclicFFT<F>>(
    ctx: &TestContext<BE, F, E>,
) {
    let mut scratch = ctx.alloc_scratch();

    let x_re_raw = ctx.re1.clone();
    let x_im_raw = vec![F::zero(); x_re_raw.len()];
    let (x_re, _x_im) = ctx.quantized_slots(&x_re_raw, &x_im_raw, ctx.meta());

    let x_ct = ctx.encrypt(ctx.max_k(), &x_re_raw, &x_im_raw, &mut scratch.borrow());
    let mut power_basis = PowerBasis::new(Basis::Monomial, x_ct);
    power_basis
        .populate(7, &ctx.module, ctx.tsk(), &mut scratch.borrow())
        .expect("populate power basis for degree 7");

    let zero_im = vec![F::zero(); x_re.len()];
    for power in 1..=4 {
        let want_re = pointwise_pow(&x_re, power);
        let ct = power_basis
            .get_stored(power)
            .unwrap_or_else(|| panic!("missing power-basis entry X^{power}"));
        ctx.assert_decrypt_precision(
            &format!("power_basis_x{power}"),
            ct,
            &want_re,
            &zero_im,
            &mut scratch.borrow(),
        );
    }
}

pub fn test_power_basis_populate_chebyshev_degree7<BE: TestPolynomialEvaluationBackend, F: TestScalar, E: NegacyclicFFT<F>>(
    ctx: &TestContext<BE, F, E>,
) {
    let mut scratch = ctx.alloc_scratch();

    let x_re_raw = ctx.re1.clone();
    let x_im_raw = vec![F::zero(); x_re_raw.len()];
    let (x_re, _x_im) = ctx.quantized_slots(&x_re_raw, &x_im_raw, ctx.meta());

    let x_ct = ctx.encrypt(ctx.max_k(), &x_re_raw, &x_im_raw, &mut scratch.borrow());

    let mut power_basis = PowerBasis::new(Basis::Chebyshev, x_ct);
    power_basis
        .populate_chebyshev(7, &ctx.module, ctx.tsk(), &mut scratch.borrow())
        .expect("populate Chebyshev power basis for degree 7");

    let want = chebyshev_values(&x_re, 4);
    let zero_im = vec![F::zero(); x_re.len()];
    for (power, want_re) in want.iter().enumerate().take(5).skip(1) {
        let ct = power_basis
            .get_stored(power)
            .unwrap_or_else(|| panic!("missing Chebyshev power-basis entry T_{power}"));
        ctx.assert_decrypt_precision(
            &format!("power_basis_chebyshev_t{power}"),
            ct,
            want_re,
            &zero_im,
            &mut scratch.borrow(),
        );
    }
}

pub fn test_chebyshev_interpolation_quadratic<BE: TestPolynomialEvaluationBackend, F: TestScalar, E: NegacyclicFFT<F>>(
    _ctx: &TestContext<BE, F, E>,
) {
    let zero = F::zero();
    let one = F::one();
    let two = one + one;
    let poly = chebyshev_interpolate(4, zero, two, |x: F| x * x - two * x + one).expect("Chebyshev interpolation should succeed");

    for i in 0..17 {
        let x = two * F::from_usize(i).unwrap() / F::from_usize(16).unwrap();
        let want = x * x - two * x + one;
        let got = poly.evaluate_on_interval(x, zero, two);
        let err = (got - want).abs();
        assert!(
            err < F::epsilon() * F::from_usize(256).unwrap(),
            "chebyshev interpolation mismatch at {:?}: got {:?}, want {:?}, err {:?}",
            x,
            got,
            want,
            err
        );
    }
}

// ── Cubic polynomial test ─────────────────────────────────────────────────────

pub fn test_eval_poly_const_coeffs_cubic<BE: TestPolynomialEvaluationBackend, F: TestScalar, E: NegacyclicFFT<F>>(
    ctx: &TestContext<BE, F, E>,
) {
    let mut scratch = ctx.alloc_scratch();

    let quarter = F::from_f64(0.25).unwrap();
    let x_re_raw: Vec<F> = ctx.re1.iter().copied().map(|x| x * quarter).collect();
    let x_im_raw = vec![F::zero(); x_re_raw.len()];
    let (x_re, x_im) = ctx.quantized_slots(&x_re_raw, &x_im_raw, ctx.meta());

    let c0 = ctx.quantized_const_pt(0.125, 0.0).0;
    let c1 = ctx.quantized_const_pt(-0.25, 0.0).0;
    let c2 = ctx.quantized_const_pt(0.0625, 0.0).0;
    let c3 = ctx.quantized_const_pt(0.03125, 0.0).0;

    // degree 3, base 2: two baby steps of degree 1, stored lowest-first.
    let low = coeff_pt(ctx, &[c0, c1]);
    let high = coeff_pt(ctx, &[c2, c3]);
    let poly = BSGSPolynomial {
        basis: Basis::Monomial,
        degree: 3,
        base: 2,
        baby_degrees: vec![1, 1],
        baby_steps: vec![low, high],
    };

    let x = ctx.encrypt(ctx.max_k(), &x_re_raw, &x_im_raw, &mut scratch.borrow());
    let mut x2 = ctx.alloc_ct(ctx.max_k());
    ctx.module
        .ckks_square_into(&mut x2, &x, ctx.tsk(), &mut scratch.borrow())
        .unwrap();

    let mut power_basis = PowerBasis::new(Basis::Monomial, x);
    power_basis.insert(2, x2);

    let x2_re = pointwise_mul(&x_re, &x_re);
    let x3_re = pointwise_mul(&x2_re, &x_re);
    let mut want_re = vec![c0; x_re.len()];
    let want_im = vec![F::zero(); x_im.len()];
    scale_add(&mut want_re, &x_re, c1);
    scale_add(&mut want_re, &x2_re, c2);
    scale_add(&mut want_re, &x3_re, c3);

    let mut res = ctx.alloc_ct(ctx.max_k());
    ctx.module
        .ckks_eval_poly_const_coeffs::<_, _, CKKSCiphertext<Vec<u8>>, _, _>(
            &mut res,
            &poly,
            &power_basis,
            ctx.tsk(),
            &mut scratch.borrow(),
        )
        .unwrap();

    ctx.assert_decrypt_precision(
        "eval_poly_const_coeffs_cubic",
        &res,
        &want_re,
        &want_im,
        &mut scratch.borrow(),
    );
}

// ── Degree-7 exponential polynomial test (matches Lattigo's test) ─────────────

/// Evaluates the degree-7 Taylor approximation of e^x on all slots and checks
/// precision.  Mirrors the `PolynomialEvaluator/Evaluate/PolySingle/Exp` test
/// from the Lattigo test suite.
pub fn test_eval_poly_const_coeffs_exp7<BE: TestPolynomialEvaluationBackend, F: TestScalar, E: NegacyclicFFT<F>>(
    ctx: &TestContext<BE, F, E>,
) {
    // e^x Taylor coefficients: 1, 1, 1/2!, 1/3!, …, 1/7!
    let raw_coeffs: [f64; 8] = [
        1.0,
        1.0,
        1.0 / 2.0,
        1.0 / 6.0,
        1.0 / 24.0,
        1.0 / 120.0,
        1.0 / 720.0,
        1.0 / 5040.0,
    ];

    // ctx.re1 values are cosines, so they lie in [-1, 1].
    let x_re_raw = ctx.re1.clone();
    let x_im_raw = vec![F::zero(); x_re_raw.len()];

    // Reference: evaluate the polynomial with the same coefficient precision
    // used by the encoded BSGS helper.
    let poly_ref = Polynomial::new(Basis::Monomial, raw_coeffs.to_vec());
    let encoded_coeffs: Vec<F> = raw_coeffs
        .iter()
        .map(|&c| ctx.quantized_const(c, 0.0, ctx.meta_pt().log_delta).0)
        .collect();
    let want_re: Vec<F> = x_re_raw
        .iter()
        .map(|&x| encoded_coeffs.iter().rev().fold(F::zero(), |acc, &c| acc * x + c))
        .collect();
    let want_im = vec![F::zero(); x_re_raw.len()];

    // Encode the BSGS polynomial (degree 7, 2 baby steps of degree 3 each).
    let bsgs = poly_ref
        .encode_bsgs(&ctx.host_module, ctx.base2k(), ctx.meta_pt())
        .expect("encode_bsgs should succeed for degree-7 monomial polynomial");

    let mut scratch = ctx.alloc_scratch();

    // Encrypt x.
    let x_ct = ctx.encrypt(ctx.max_k(), &x_re_raw, &x_im_raw, &mut scratch.borrow());

    // Build power basis: needs X^1, X^2, X^3, X^4.
    let mut pb = PowerBasis::new(Basis::Monomial, x_ct);
    pb.populate(7, &ctx.module, ctx.tsk(), &mut scratch.borrow())
        .expect("populate power basis for degree 7");

    let mut res = ctx.alloc_ct(ctx.max_k());
    ctx.module
        .ckks_eval_poly_const_coeffs::<_, _, CKKSCiphertext<Vec<u8>>, _, _>(
            &mut res,
            &bsgs,
            &pb,
            ctx.tsk(),
            &mut scratch.borrow(),
        )
        .expect("ckks_eval_poly_const_coeffs should succeed");

    ctx.assert_decrypt_precision("eval_poly_const_coeffs_exp7", &res, &want_re, &want_im, &mut scratch.borrow());
}

pub fn test_eval_poly_const_coeffs_chebyshev_degree31<BE: TestPolynomialEvaluationBackend, F: TestScalar, E: NegacyclicFFT<F>>(
    ctx: &TestContext<BE, F, E>,
) {
    let x_re_raw = ctx.re1.clone();
    let x_im_raw = vec![F::zero(); x_re_raw.len()];
    let input_meta = ctx.precision_at(ctx.meta().log_delta.min(20));
    let (x_re, _x_im) = ctx.quantized_slots(&x_re_raw, &x_im_raw, input_meta);

    let poly = chebyshev_interpolate(31, -F::one(), F::one(), |x: F| x.sin())
        .expect("degree-31 Chebyshev interpolation of sin(x) should succeed");

    let bsgs = poly
        .encode_bsgs(&ctx.host_module, ctx.base2k(), ctx.meta_pt())
        .expect("encode_bsgs should succeed for degree-31 Chebyshev polynomial");
    let want_re: Vec<F> = x_re.iter().map(|&x| eval_encoded_bsgs_chebyshev(&bsgs, x)).collect();
    let want_im = vec![F::zero(); x_re.len()];

    let mut scratch = ctx.alloc_scratch();
    let x_ct = ctx.encrypt_with_prec(ctx.max_k(), &x_re_raw, &x_im_raw, input_meta, &mut scratch.borrow());

    println!("x_ct.meta: {:?}", x_ct.meta);

    let mut pb = PowerBasis::new(Basis::Chebyshev, x_ct);

    pb.populate_chebyshev(31, &ctx.module, ctx.tsk(), &mut scratch.borrow())
        .expect("populate Chebyshev power basis for degree 31");

    let mut res = ctx.alloc_ct(ctx.max_k());
    ctx.module
        .ckks_eval_poly_const_coeffs::<_, _, CKKSCiphertext<Vec<u8>>, _, _>(
            &mut res,
            &bsgs,
            &pb,
            ctx.tsk(),
            &mut scratch.borrow(),
        )
        .expect("ckks_eval_poly_const_coeffs should succeed");

    println!("res.meta: {:?}", res.meta);

    ctx.assert_decrypt_precision(
        "eval_poly_const_coeffs_chebyshev_degree31",
        &res,
        &want_re,
        &want_im,
        &mut scratch.borrow(),
    );
}
