use poulpy_core::{
    layouts::GLWETensorKeyPrepared,
    layouts::{GGLWEInfos, GLWEToBackendMut, GLWEToBackendRef, LWEInfos, prepared::GLWETensorKeyPreparedToBackendRef},
};
use poulpy_hal::{
    api::{NegacyclicFFT, NegacyclicFFTNew, ScratchOwnedBorrow},
    layouts::{HostBytesBackend, Module},
};

use crate::{
    CKKSCtBounds, CKKSMeta, SetCKKSInfos,
    encoding::reim::Encoder,
    layouts::{CKKSCiphertext, CKKSPlaintext, CKKSPlaintextVecHostCodec},
    leveled::api::{CKKSMulOps, PolynomialEvaluation},
    polynomial::{BSGSPolynomial, Basis, EncodeBSGS, Parity, Polynomial, SplitStrategy},
    power_basis::PowerBasis,
    test_suite::CKKSTestParams,
};

use super::helpers::{
    PT_PREC, TestContextBackend, TestContextModule, TestScalar, alloc_ct, alloc_scratch, assert_decrypt_precision, ckks_encrypt,
    ckks_encrypt_with_prec, gen_sk_with_raw, gen_tsk, precision_at, quantized_const, quantized_slots, test_vector_1, upload_pt,
};

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

    let mut steps = Vec::with_capacity(poly.baby_steps().len());
    for coeffs_pt in poly.baby_steps().iter() {
        let degree = coeffs_pt.n().as_usize() - 1;
        let mut coeffs = vec![F::zero(); coeffs_pt.n().as_usize()];
        coeffs_pt.decode_host_floats(&mut coeffs).unwrap();
        let value = coeffs
            .iter()
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

fn upload_bsgs<BE>(
    module: &Module<BE>,
    poly: &BSGSPolynomial<CKKSPlaintext<Vec<u8>>>,
) -> BSGSPolynomial<CKKSPlaintext<BE::OwnedBuf>>
where
    BE: TestContextBackend,
    Module<BE>: TestContextModule<BE>,
{
    poly.map_baby_steps_ref(|pt| upload_pt(module, pt))
}

pub fn test_power_basis_populate_degree7<BE, F, E>(
    params: CKKSTestParams,
    module: &Module<BE>,
    host_module: &Module<HostBytesBackend>,
) where
    BE: TestContextBackend,
    Module<BE>: TestContextModule<BE>,
    CKKSCiphertext<BE::OwnedBuf>: GLWEToBackendMut<BE> + GLWEToBackendRef<BE> + CKKSCtBounds + SetCKKSInfos,
    GLWETensorKeyPrepared<BE::OwnedBuf, BE>: GLWETensorKeyPreparedToBackendRef<BE> + GGLWEInfos,
    F: TestScalar,
    E: NegacyclicFFT<F> + NegacyclicFFTNew<F>,
{
    let m = params.n / 2;
    let encoder = Encoder::<E>::new(m).unwrap();
    let (x_re_raw, _x_im_raw) = test_vector_1::<F>(m);
    let x_im_raw = vec![F::zero(); x_re_raw.len()];
    let (x_re, _) = quantized_slots(host_module, &encoder, params.base2k.into(), params.prec, &x_re_raw, &x_im_raw);

    let (sk_raw, sk) = gen_sk_with_raw(&params, module, host_module, [0u8; 32]);
    let mut scratch = alloc_scratch(&params, module);
    let tsk = gen_tsk(&params, module, &sk_raw, &mut scratch.borrow());

    let x_ct = ckks_encrypt(
        &params,
        module,
        host_module,
        &encoder,
        &sk,
        params.k,
        &x_re_raw,
        &x_im_raw,
        &mut scratch.borrow(),
    );
    let mut power_basis = PowerBasis::new(Basis::Monomial, x_ct);
    power_basis
        .populate(7, 2, Parity::Full, module, &tsk, &mut scratch.borrow())
        .expect("populate power basis for degree 7");

    let zero_im = vec![F::zero(); x_re.len()];
    for power in 1..=4 {
        let want_re = pointwise_pow(&x_re, power);
        let ct = power_basis
            .get_stored(power)
            .unwrap_or_else(|| panic!("missing power-basis entry X^{power}"));
        assert_decrypt_precision(
            &format!("power_basis_x{power}"),
            &params,
            module,
            &encoder,
            ct,
            &sk,
            &want_re,
            &zero_im,
            &mut scratch.borrow(),
        );
    }
}

pub fn test_power_basis_populate_chebyshev_degree7<BE, F, E>(
    params: CKKSTestParams,
    module: &Module<BE>,
    host_module: &Module<HostBytesBackend>,
) where
    BE: TestContextBackend,
    Module<BE>: TestContextModule<BE>,
    CKKSCiphertext<BE::OwnedBuf>: GLWEToBackendMut<BE> + GLWEToBackendRef<BE> + CKKSCtBounds + SetCKKSInfos,
    GLWETensorKeyPrepared<BE::OwnedBuf, BE>: GLWETensorKeyPreparedToBackendRef<BE> + GGLWEInfos,
    F: TestScalar,
    E: NegacyclicFFT<F> + NegacyclicFFTNew<F>,
{
    let m = params.n / 2;
    let encoder = Encoder::<E>::new(m).unwrap();
    let (x_re_raw, _x_im_raw) = test_vector_1::<F>(m);
    let x_im_raw = vec![F::zero(); x_re_raw.len()];
    let (x_re, _) = quantized_slots(host_module, &encoder, params.base2k.into(), params.prec, &x_re_raw, &x_im_raw);

    let (sk_raw, sk) = gen_sk_with_raw(&params, module, host_module, [0u8; 32]);
    let mut scratch = alloc_scratch(&params, module);
    let tsk = gen_tsk(&params, module, &sk_raw, &mut scratch.borrow());

    let x_ct = ckks_encrypt(
        &params,
        module,
        host_module,
        &encoder,
        &sk,
        params.k,
        &x_re_raw,
        &x_im_raw,
        &mut scratch.borrow(),
    );
    let mut power_basis = PowerBasis::new(Basis::Chebyshev, x_ct);
    power_basis
        .populate(7, 2, Parity::Full, module, &tsk, &mut scratch.borrow())
        .expect("populate Chebyshev power basis for degree 7");

    let want = chebyshev_values(&x_re, 4);
    let zero_im = vec![F::zero(); x_re.len()];
    for (power, want_re) in want.iter().enumerate().take(5).skip(1) {
        let ct = power_basis
            .get_stored(power)
            .unwrap_or_else(|| panic!("missing Chebyshev power-basis entry T_{power}"));
        assert_decrypt_precision(
            &format!("power_basis_chebyshev_t{power}"),
            &params,
            module,
            &encoder,
            ct,
            &sk,
            want_re,
            &zero_im,
            &mut scratch.borrow(),
        );
    }
}

pub fn test_chebyshev_interpolation_quadratic<BE, F, E>(
    _params: CKKSTestParams,
    _module: &Module<BE>,
    _host_module: &Module<HostBytesBackend>,
) where
    BE: TestContextBackend,
    F: TestScalar,
    E: NegacyclicFFT<F>,
{
    let zero = F::zero();
    let one = F::one();
    let two = one + one;
    let poly = Polynomial::chebyshev_interpolate(4, zero, two, |x: F| x * x - two * x + one)
        .expect("Chebyshev interpolation should succeed");

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

pub fn test_encode_bsgs_preserves_chebyshev_eval<BE, F, E>(
    params: CKKSTestParams,
    _module: &Module<BE>,
    host_module: &Module<HostBytesBackend>,
) where
    BE: TestContextBackend,
    F: TestScalar,
    E: NegacyclicFFT<F>,
{
    let poly = Polynomial::chebyshev_interpolate(31, -F::one(), F::one(), |x: F| x.sin())
        .expect("degree-31 Chebyshev interpolation of sin(x) should succeed");
    let coeff_meta = CKKSMeta {
        log_delta: 40,
        log_budget: 8,
    };
    let bsgs = poly
        .encode_bsgs(host_module, params.base2k.into(), coeff_meta)
        .expect("encode_bsgs should succeed for degree-31 Chebyshev polynomial");
    let tolerance = (-F::from_usize(coeff_meta.log_delta).unwrap()).exp2() * F::from_usize(1024).unwrap();

    for i in 0..=64 {
        let x = -F::one() + (F::one() + F::one()) * F::from_usize(i).unwrap() / F::from_usize(64).unwrap();
        let got = eval_encoded_bsgs_chebyshev(&bsgs, x);
        let want = poly.evaluate(x);
        let err = (got - want).abs();
        assert!(
            err <= tolerance,
            "encoded BSGS evaluation mismatch at {:?}: got {:?}, want {:?}, err {:?}, tolerance {:?}",
            x,
            got,
            want,
            err,
            tolerance
        );
    }
}

pub fn test_eval_poly_const_coeffs_cubic<BE, F, E>(
    params: CKKSTestParams,
    module: &Module<BE>,
    host_module: &Module<HostBytesBackend>,
) where
    BE: TestContextBackend,
    Module<BE>: TestContextModule<BE> + PolynomialEvaluation<BE>,
    CKKSCiphertext<BE::OwnedBuf>: GLWEToBackendMut<BE> + GLWEToBackendRef<BE> + CKKSCtBounds + SetCKKSInfos,
    CKKSPlaintext<BE::OwnedBuf>: GLWEToBackendRef<BE> + LWEInfos,
    GLWETensorKeyPrepared<BE::OwnedBuf, BE>: GLWETensorKeyPreparedToBackendRef<BE> + GGLWEInfos,
    F: TestScalar,
    E: NegacyclicFFT<F> + NegacyclicFFTNew<F>,
{
    let m = params.n / 2;
    let encoder = Encoder::<E>::new(m).unwrap();

    let quarter = F::from_f64(0.25).unwrap();
    let (re1, _im1) = test_vector_1::<F>(m);
    let x_re_raw: Vec<F> = re1.iter().copied().map(|x| x * quarter).collect();
    let x_im_raw = vec![F::zero(); x_re_raw.len()];
    let (x_re, x_im) = quantized_slots(host_module, &encoder, params.base2k.into(), params.prec, &x_re_raw, &x_im_raw);

    let raw_coeffs = [0.125f64, -0.25, 0.0625, 0.03125];
    let c0 = quantized_const::<F>(raw_coeffs[0], 0.0, PT_PREC.log_delta).0;
    let c1 = quantized_const::<F>(raw_coeffs[1], 0.0, PT_PREC.log_delta).0;
    let c2 = quantized_const::<F>(raw_coeffs[2], 0.0, PT_PREC.log_delta).0;
    let c3 = quantized_const::<F>(raw_coeffs[3], 0.0, PT_PREC.log_delta).0;

    let poly_ref = Polynomial::new(Basis::Monomial, raw_coeffs.to_vec());
    let bsgs_host = poly_ref
        .encode_bsgs(host_module, params.base2k.into(), PT_PREC)
        .expect("encode_bsgs should succeed for cubic monomial polynomial");
    let poly = upload_bsgs(module, &bsgs_host);

    let (sk_raw, sk) = gen_sk_with_raw(&params, module, host_module, [0u8; 32]);
    let mut scratch = alloc_scratch(&params, module);
    let tsk = gen_tsk(&params, module, &sk_raw, &mut scratch.borrow());

    let x = ckks_encrypt(
        &params,
        module,
        host_module,
        &encoder,
        &sk,
        params.k,
        &x_re_raw,
        &x_im_raw,
        &mut scratch.borrow(),
    );
    let mut x2 = alloc_ct(&params, module, params.k);
    module.ckks_square_into(&mut x2, &x, &tsk, &mut scratch.borrow()).unwrap();

    let mut power_basis = PowerBasis::new(Basis::Monomial, x);
    power_basis.insert(2, x2).expect("insert pre-computed X^2");

    let x2_re = pointwise_mul(&x_re, &x_re);
    let x3_re = pointwise_mul(&x2_re, &x_re);
    let mut want_re = vec![c0; x_re.len()];
    let want_im = vec![F::zero(); x_im.len()];
    scale_add(&mut want_re, &x_re, c1);
    scale_add(&mut want_re, &x2_re, c2);
    scale_add(&mut want_re, &x3_re, c3);

    let mut res = alloc_ct(&params, module, params.k);
    module
        .ckks_eval_poly_real_const_coeffs_from_power_basis::<_, _, CKKSCiphertext<BE::OwnedBuf>, _, _>(
            &mut res,
            &poly,
            &power_basis,
            &tsk,
            &mut scratch.borrow(),
        )
        .unwrap();

    assert_decrypt_precision(
        "eval_poly_const_coeffs_cubic",
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

pub fn test_eval_poly_rejects_power_basis_mismatch<BE, F, E>(
    params: CKKSTestParams,
    module: &Module<BE>,
    host_module: &Module<HostBytesBackend>,
) where
    BE: TestContextBackend,
    Module<BE>: TestContextModule<BE> + PolynomialEvaluation<BE>,
    CKKSCiphertext<BE::OwnedBuf>: GLWEToBackendMut<BE> + GLWEToBackendRef<BE> + CKKSCtBounds + SetCKKSInfos,
    CKKSPlaintext<BE::OwnedBuf>: GLWEToBackendRef<BE> + LWEInfos,
    GLWETensorKeyPrepared<BE::OwnedBuf, BE>: GLWETensorKeyPreparedToBackendRef<BE> + GGLWEInfos,
    F: TestScalar,
    E: NegacyclicFFT<F> + NegacyclicFFTNew<F>,
{
    let m = params.n / 2;
    let encoder = Encoder::<E>::new(m).unwrap();
    let (x_re_raw, _x_im_raw) = test_vector_1::<F>(m);
    let x_im_raw = vec![F::zero(); x_re_raw.len()];

    let poly_ref = Polynomial::new(Basis::Monomial, vec![0.125f64, -0.25, 0.0625]);
    let bsgs_host = poly_ref
        .encode_bsgs(host_module, params.base2k.into(), PT_PREC)
        .expect("encode_bsgs should succeed for monomial polynomial");
    let bsgs = upload_bsgs(module, &bsgs_host);

    let (sk_raw, sk) = gen_sk_with_raw(&params, module, host_module, [0u8; 32]);
    let mut scratch = alloc_scratch(&params, module);
    let tsk = gen_tsk(&params, module, &sk_raw, &mut scratch.borrow());
    let x_ct = ckks_encrypt(
        &params,
        module,
        host_module,
        &encoder,
        &sk,
        params.k,
        &x_re_raw,
        &x_im_raw,
        &mut scratch.borrow(),
    );
    let power_basis = PowerBasis::new(Basis::Chebyshev, x_ct);

    let mut res = alloc_ct(&params, module, params.k);
    let err = module
        .ckks_eval_poly_real_const_coeffs_from_power_basis::<_, _, CKKSCiphertext<BE::OwnedBuf>, _, _>(
            &mut res,
            &bsgs,
            &power_basis,
            &tsk,
            &mut scratch.borrow(),
        )
        .expect_err("basis mismatch should be rejected");

    let err = err.to_string();
    assert!(
        err.contains("polynomial basis Monomial does not match power basis Chebyshev"),
        "unexpected basis mismatch error: {err}"
    );
}

pub fn test_eval_poly_const_coeffs_exp7<BE, F, E>(
    params: CKKSTestParams,
    module: &Module<BE>,
    host_module: &Module<HostBytesBackend>,
) where
    BE: TestContextBackend,
    Module<BE>: TestContextModule<BE> + PolynomialEvaluation<BE>,
    CKKSCiphertext<BE::OwnedBuf>: GLWEToBackendMut<BE> + GLWEToBackendRef<BE> + CKKSCtBounds + SetCKKSInfos,
    CKKSPlaintext<BE::OwnedBuf>: GLWEToBackendRef<BE> + LWEInfos,
    GLWETensorKeyPrepared<BE::OwnedBuf, BE>: GLWETensorKeyPreparedToBackendRef<BE> + GGLWEInfos,
    F: TestScalar,
    E: NegacyclicFFT<F> + NegacyclicFFTNew<F>,
{
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

    let m = params.n / 2;
    let encoder = Encoder::<E>::new(m).unwrap();
    let (x_re_raw, _x_im_raw) = test_vector_1::<F>(m);
    let x_im_raw = vec![F::zero(); x_re_raw.len()];

    let encoded_coeffs: Vec<F> = raw_coeffs
        .iter()
        .map(|&c| quantized_const::<F>(c, 0.0, PT_PREC.log_delta).0)
        .collect();
    let want_re: Vec<F> = x_re_raw
        .iter()
        .map(|&x| encoded_coeffs.iter().rev().fold(F::zero(), |acc, &c| acc * x + c))
        .collect();
    let want_im = vec![F::zero(); x_re_raw.len()];

    let poly_ref = Polynomial::new(Basis::Monomial, raw_coeffs.to_vec());
    let bsgs_host = poly_ref
        .encode_bsgs(host_module, params.base2k.into(), PT_PREC)
        .expect("encode_bsgs should succeed for degree-7 monomial polynomial");
    let bsgs = upload_bsgs(module, &bsgs_host);

    let (sk_raw, sk) = gen_sk_with_raw(&params, module, host_module, [0u8; 32]);
    let mut scratch = alloc_scratch(&params, module);
    let tsk = gen_tsk(&params, module, &sk_raw, &mut scratch.borrow());

    let x_ct = ckks_encrypt(
        &params,
        module,
        host_module,
        &encoder,
        &sk,
        params.k,
        &x_re_raw,
        &x_im_raw,
        &mut scratch.borrow(),
    );
    let mut pb = PowerBasis::new(Basis::Monomial, x_ct);
    pb.populate(7, bsgs_host.log_split(), Parity::Full, module, &tsk, &mut scratch.borrow())
        .expect("populate power basis for degree 7");

    let mut res = alloc_ct(&params, module, params.k);
    module
        .ckks_eval_poly_real_const_coeffs_from_power_basis::<_, _, CKKSCiphertext<BE::OwnedBuf>, _, _>(
            &mut res,
            &bsgs,
            &pb,
            &tsk,
            &mut scratch.borrow(),
        )
        .expect("ckks_eval_poly_real_const_coeffs_from_power_basis should succeed");

    assert_decrypt_precision(
        "eval_poly_const_coeffs_exp7",
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

pub fn test_eval_poly_const_coeffs_even_monomial<BE, F, E>(
    params: CKKSTestParams,
    module: &Module<BE>,
    host_module: &Module<HostBytesBackend>,
) where
    BE: TestContextBackend,
    Module<BE>: TestContextModule<BE> + PolynomialEvaluation<BE>,
    CKKSCiphertext<BE::OwnedBuf>: GLWEToBackendMut<BE> + GLWEToBackendRef<BE> + CKKSCtBounds + SetCKKSInfos,
    CKKSPlaintext<BE::OwnedBuf>: GLWEToBackendRef<BE> + LWEInfos,
    GLWETensorKeyPrepared<BE::OwnedBuf, BE>: GLWETensorKeyPreparedToBackendRef<BE> + GGLWEInfos,
    F: TestScalar,
    E: NegacyclicFFT<F> + NegacyclicFFTNew<F>,
{
    // f(x) = 0.5 + 0.25·x² + 0.125·x⁴  (is_even: all odd coefficients zero)
    let raw_coeffs = [0.5f64, 0.0, 0.25, 0.0, 0.125];

    let m = params.n / 2;
    let encoder = Encoder::<E>::new(m).unwrap();
    let quarter = F::from_f64(0.25).unwrap();
    let (re1, _) = test_vector_1::<F>(m);
    let x_re_raw: Vec<F> = re1.iter().map(|&x| x * quarter).collect();
    let x_im_raw = vec![F::zero(); x_re_raw.len()];

    let encoded_coeffs: Vec<F> = raw_coeffs
        .iter()
        .map(|&c| quantized_const::<F>(c, 0.0, PT_PREC.log_delta).0)
        .collect();
    let want_re: Vec<F> = x_re_raw
        .iter()
        .map(|&x| encoded_coeffs.iter().rev().fold(F::zero(), |acc, &c| acc * x + c))
        .collect();
    let want_im = vec![F::zero(); x_re_raw.len()];

    let poly_ref = Polynomial::new(Basis::Monomial, raw_coeffs.to_vec());
    assert_eq!(poly_ref.parity, Parity::Even, "polynomial should be detected as even");
    let bsgs_host = poly_ref
        .encode_bsgs(host_module, params.base2k.into(), PT_PREC)
        .expect("encode_bsgs should succeed");
    assert_eq!(bsgs_host.parity(), Parity::Even, "BSGSPolynomial should carry Even parity");
    let bsgs = upload_bsgs(module, &bsgs_host);

    let (sk_raw, sk) = gen_sk_with_raw(&params, module, host_module, [0u8; 32]);
    let mut scratch = alloc_scratch(&params, module);
    let tsk = gen_tsk(&params, module, &sk_raw, &mut scratch.borrow());

    let x_ct = ckks_encrypt(
        &params,
        module,
        host_module,
        &encoder,
        &sk,
        params.k,
        &x_re_raw,
        &x_im_raw,
        &mut scratch.borrow(),
    );
    let mut pb = PowerBasis::new(Basis::Monomial, x_ct);
    pb.populate(
        4,
        bsgs_host.log_split(),
        bsgs_host.parity(),
        module,
        &tsk,
        &mut scratch.borrow(),
    )
    .expect("populate power basis for degree 4");

    let mut res = alloc_ct(&params, module, params.k);
    module
        .ckks_eval_poly_real_const_coeffs_from_power_basis::<_, _, CKKSCiphertext<BE::OwnedBuf>, _, _>(
            &mut res,
            &bsgs,
            &pb,
            &tsk,
            &mut scratch.borrow(),
        )
        .expect("ckks_eval_poly_real_const_coeffs_from_power_basis should succeed");

    assert_decrypt_precision(
        "eval_poly_const_coeffs_even_monomial",
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

pub fn test_eval_poly_const_coeffs_odd_monomial<BE, F, E>(
    params: CKKSTestParams,
    module: &Module<BE>,
    host_module: &Module<HostBytesBackend>,
) where
    BE: TestContextBackend,
    Module<BE>: TestContextModule<BE> + PolynomialEvaluation<BE>,
    CKKSCiphertext<BE::OwnedBuf>: GLWEToBackendMut<BE> + GLWEToBackendRef<BE> + CKKSCtBounds + SetCKKSInfos,
    CKKSPlaintext<BE::OwnedBuf>: GLWEToBackendRef<BE> + LWEInfos,
    GLWETensorKeyPrepared<BE::OwnedBuf, BE>: GLWETensorKeyPreparedToBackendRef<BE> + GGLWEInfos,
    F: TestScalar,
    E: NegacyclicFFT<F> + NegacyclicFFTNew<F>,
{
    // f(x) = 0.25·x + 0.125·x³ + 0.0625·x⁵  (is_odd: all even coefficients zero)
    let raw_coeffs = [0.0f64, 0.25, 0.0, 0.125, 0.0, 0.0625];

    let m = params.n / 2;
    let encoder = Encoder::<E>::new(m).unwrap();
    let quarter = F::from_f64(0.25).unwrap();
    let (re1, _) = test_vector_1::<F>(m);
    let x_re_raw: Vec<F> = re1.iter().map(|&x| x * quarter).collect();
    let x_im_raw = vec![F::zero(); x_re_raw.len()];

    let encoded_coeffs: Vec<F> = raw_coeffs
        .iter()
        .map(|&c| quantized_const::<F>(c, 0.0, PT_PREC.log_delta).0)
        .collect();
    let want_re: Vec<F> = x_re_raw
        .iter()
        .map(|&x| encoded_coeffs.iter().rev().fold(F::zero(), |acc, &c| acc * x + c))
        .collect();
    let want_im = vec![F::zero(); x_re_raw.len()];

    let poly_ref = Polynomial::new(Basis::Monomial, raw_coeffs.to_vec());
    assert_eq!(poly_ref.parity, Parity::Odd, "polynomial should be detected as odd");
    let bsgs_host = poly_ref
        .encode_bsgs(host_module, params.base2k.into(), PT_PREC)
        .expect("encode_bsgs should succeed");
    assert_eq!(bsgs_host.parity(), Parity::Odd, "BSGSPolynomial should carry Odd parity");
    let bsgs = upload_bsgs(module, &bsgs_host);

    let (sk_raw, sk) = gen_sk_with_raw(&params, module, host_module, [0u8; 32]);
    let mut scratch = alloc_scratch(&params, module);
    let tsk = gen_tsk(&params, module, &sk_raw, &mut scratch.borrow());

    let x_ct = ckks_encrypt(
        &params,
        module,
        host_module,
        &encoder,
        &sk,
        params.k,
        &x_re_raw,
        &x_im_raw,
        &mut scratch.borrow(),
    );
    let mut pb = PowerBasis::new(Basis::Monomial, x_ct);
    pb.populate(
        5,
        bsgs_host.log_split(),
        bsgs_host.parity(),
        module,
        &tsk,
        &mut scratch.borrow(),
    )
    .expect("populate power basis for degree 5");

    let mut res = alloc_ct(&params, module, params.k);
    module
        .ckks_eval_poly_real_const_coeffs_from_power_basis::<_, _, CKKSCiphertext<BE::OwnedBuf>, _, _>(
            &mut res,
            &bsgs,
            &pb,
            &tsk,
            &mut scratch.borrow(),
        )
        .expect("ckks_eval_poly_real_const_coeffs_from_power_basis should succeed");

    assert_decrypt_precision(
        "eval_poly_const_coeffs_odd_monomial",
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

pub fn test_eval_poly_const_coeffs_chebyshev_degree31<BE, F, E>(
    params: CKKSTestParams,
    module: &Module<BE>,
    host_module: &Module<HostBytesBackend>,
) where
    BE: TestContextBackend,
    Module<BE>: TestContextModule<BE> + PolynomialEvaluation<BE>,
    CKKSCiphertext<BE::OwnedBuf>: GLWEToBackendMut<BE> + GLWEToBackendRef<BE> + CKKSCtBounds + SetCKKSInfos,
    CKKSPlaintext<BE::OwnedBuf>: GLWEToBackendRef<BE> + LWEInfos,
    GLWETensorKeyPrepared<BE::OwnedBuf, BE>: GLWETensorKeyPreparedToBackendRef<BE> + GGLWEInfos,
    F: TestScalar,
    E: NegacyclicFFT<F> + NegacyclicFFTNew<F>,
{
    let m = params.n / 2;
    let encoder = Encoder::<E>::new(m).unwrap();
    let (x_re_raw, _x_im_raw) = test_vector_1::<F>(m);
    let x_im_raw = vec![F::zero(); x_re_raw.len()];
    let input_meta = precision_at(&params, params.prec.log_delta.min(20));
    let (x_re, _) = quantized_slots(host_module, &encoder, params.base2k.into(), input_meta, &x_re_raw, &x_im_raw);

    let poly = Polynomial::chebyshev_interpolate(31, -F::one(), F::one(), |x: F| x.sin())
        .expect("degree-31 Chebyshev interpolation of sin(x) should succeed");
    let bsgs_host = poly
        .encode_bsgs(host_module, params.base2k.into(), PT_PREC)
        .expect("encode_bsgs should succeed for degree-31 Chebyshev polynomial");
    let want_re: Vec<F> = x_re.iter().map(|&x| eval_encoded_bsgs_chebyshev(&bsgs_host, x)).collect();
    let want_im = vec![F::zero(); x_re.len()];
    let bsgs = upload_bsgs(module, &bsgs_host);

    let (sk_raw, sk) = gen_sk_with_raw(&params, module, host_module, [0u8; 32]);
    let mut scratch = alloc_scratch(&params, module);
    let tsk = gen_tsk(&params, module, &sk_raw, &mut scratch.borrow());

    let x_ct = ckks_encrypt_with_prec(
        &params,
        module,
        host_module,
        &encoder,
        &sk,
        params.k,
        &x_re_raw,
        &x_im_raw,
        input_meta,
        &mut scratch.borrow(),
    );
    let mut pb = PowerBasis::new(Basis::Chebyshev, x_ct);
    pb.populate(
        31,
        bsgs_host.log_split(),
        bsgs_host.parity(),
        module,
        &tsk,
        &mut scratch.borrow(),
    )
    .expect("populate Chebyshev power basis for degree 31");

    let mut res = alloc_ct(&params, module, params.k);
    module
        .ckks_eval_poly_real_const_coeffs_from_power_basis::<_, _, CKKSCiphertext<BE::OwnedBuf>, _, _>(
            &mut res,
            &bsgs,
            &pb,
            &tsk,
            &mut scratch.borrow(),
        )
        .expect("ckks_eval_poly_real_const_coeffs_from_power_basis should succeed");

    assert_decrypt_precision(
        "eval_poly_const_coeffs_chebyshev_degree31",
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

pub fn test_eval_poly_const_coeffs_chebyshev_degree31_min_mult<BE, F, E>(
    params: CKKSTestParams,
    module: &Module<BE>,
    host_module: &Module<HostBytesBackend>,
) where
    BE: TestContextBackend,
    Module<BE>: TestContextModule<BE> + PolynomialEvaluation<BE>,
    CKKSCiphertext<BE::OwnedBuf>: GLWEToBackendMut<BE> + GLWEToBackendRef<BE> + CKKSCtBounds + SetCKKSInfos,
    CKKSPlaintext<BE::OwnedBuf>: GLWEToBackendRef<BE> + LWEInfos,
    GLWETensorKeyPrepared<BE::OwnedBuf, BE>: GLWETensorKeyPreparedToBackendRef<BE> + GGLWEInfos,
    F: TestScalar,
    E: NegacyclicFFT<F> + NegacyclicFFTNew<F>,
{
    let m = params.n / 2;
    let encoder = Encoder::<E>::new(m).unwrap();
    let (x_re_raw, _x_im_raw) = test_vector_1::<F>(m);
    let x_im_raw = vec![F::zero(); x_re_raw.len()];
    let input_meta = precision_at(&params, params.prec.log_delta.min(20));
    let (x_re, _) = quantized_slots(host_module, &encoder, params.base2k.into(), input_meta, &x_re_raw, &x_im_raw);

    let poly = Polynomial::chebyshev_interpolate(31, -F::one(), F::one(), |x: F| x.sin())
        .expect("degree-31 Chebyshev interpolation of sin(x) should succeed");
    let bsgs_host = poly
        .encode_bsgs_with(host_module, params.base2k.into(), PT_PREC, SplitStrategy::MinMult)
        .expect("encode_bsgs_with MinMult should succeed for degree-31 Chebyshev polynomial");
    let want_re: Vec<F> = x_re.iter().map(|&x| eval_encoded_bsgs_chebyshev(&bsgs_host, x)).collect();
    let want_im = vec![F::zero(); x_re.len()];
    let bsgs = upload_bsgs(module, &bsgs_host);

    let (sk_raw, sk) = gen_sk_with_raw(&params, module, host_module, [0u8; 32]);
    let mut scratch = alloc_scratch(&params, module);
    let tsk = gen_tsk(&params, module, &sk_raw, &mut scratch.borrow());

    let x_ct = ckks_encrypt_with_prec(
        &params,
        module,
        host_module,
        &encoder,
        &sk,
        params.k,
        &x_re_raw,
        &x_im_raw,
        input_meta,
        &mut scratch.borrow(),
    );
    let mut pb = PowerBasis::new(Basis::Chebyshev, x_ct);
    pb.populate(
        31,
        bsgs_host.log_split(),
        bsgs_host.parity(),
        module,
        &tsk,
        &mut scratch.borrow(),
    )
    .expect("populate Chebyshev power basis for degree 31 with MinMult split");

    let mut res = alloc_ct(&params, module, params.k);
    module
        .ckks_eval_poly_real_const_coeffs_from_power_basis::<_, _, CKKSCiphertext<BE::OwnedBuf>, _, _>(
            &mut res,
            &bsgs,
            &pb,
            &tsk,
            &mut scratch.borrow(),
        )
        .expect("ckks_eval_poly_real_const_coeffs_from_power_basis should succeed with MinMult split");

    assert_decrypt_precision(
        "eval_poly_const_coeffs_chebyshev_degree31_min_mult",
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
