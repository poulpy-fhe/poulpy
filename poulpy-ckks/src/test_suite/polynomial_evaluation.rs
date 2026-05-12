use poulpy_core::{
    layouts::GLWETensorKeyPrepared,
    layouts::{GGLWEInfos, GLWEToBackendMut, GLWEToBackendRef, LWEInfos, prepared::GLWETensorKeyPreparedToBackendRef},
};
use poulpy_hal::{
    api::{NegacyclicFFT, NegacyclicFFTNew, ScratchOwnedBorrow},
    layouts::{HostBytesBackend, Module},
};

use crate::{
    CKKSCtBounds, SetCKKSInfos,
    encoding::reim::Encoder,
    layouts::{CKKSCiphertext, CKKSModuleAlloc, CKKSPlaintext, CKKSPlaintextVecHostCodec},
    leveled::api::{CKKSMulOps, PolynomialEvaluation},
    polynomial::{BSGSPolynomial, Basis, Polynomial, PowerBasis, chebyshev_interpolate},
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

fn coeff_pt<BE, F>(
    params: &CKKSTestParams,
    host_module: &Module<HostBytesBackend>,
    module: &Module<BE>,
    coeffs: &[F],
) -> CKKSPlaintext<BE::OwnedBuf>
where
    BE: TestContextBackend,
    Module<BE>: TestContextModule<BE>,
    F: TestScalar,
{
    let mut pt = host_module.ckks_pt_coeffs_alloc(coeffs.len(), params.base2k.into(), PT_PREC);
    pt.encode_host_floats(coeffs).unwrap();
    upload_pt(module, &pt)
}

fn upload_bsgs<BE>(
    module: &Module<BE>,
    poly: &BSGSPolynomial<CKKSPlaintext<Vec<u8>>>,
) -> BSGSPolynomial<CKKSPlaintext<BE::OwnedBuf>>
where
    BE: TestContextBackend,
    Module<BE>: TestContextModule<BE>,
{
    BSGSPolynomial {
        basis: poly.basis,
        degree: poly.degree,
        base: poly.base,
        baby_degrees: poly.baby_degrees.clone(),
        baby_steps: poly.baby_steps.iter().map(|pt| upload_pt(module, pt)).collect(),
    }
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
        .populate(7, module, &tsk, &mut scratch.borrow())
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
        .populate_chebyshev(7, module, &tsk, &mut scratch.borrow())
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

    let c0 = quantized_const::<F>(0.125, 0.0, PT_PREC.log_delta).0;
    let c1 = quantized_const::<F>(-0.25, 0.0, PT_PREC.log_delta).0;
    let c2 = quantized_const::<F>(0.0625, 0.0, PT_PREC.log_delta).0;
    let c3 = quantized_const::<F>(0.03125, 0.0, PT_PREC.log_delta).0;

    let low = coeff_pt(&params, host_module, module, &[c0, c1]);
    let high = coeff_pt(&params, host_module, module, &[c2, c3]);
    let poly = BSGSPolynomial {
        basis: Basis::Monomial,
        degree: 3,
        base: 2,
        baby_degrees: vec![1, 1],
        baby_steps: vec![low, high],
    };

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
    power_basis.insert(2, x2);

    let x2_re = pointwise_mul(&x_re, &x_re);
    let x3_re = pointwise_mul(&x2_re, &x_re);
    let mut want_re = vec![c0; x_re.len()];
    let want_im = vec![F::zero(); x_im.len()];
    scale_add(&mut want_re, &x_re, c1);
    scale_add(&mut want_re, &x2_re, c2);
    scale_add(&mut want_re, &x3_re, c3);

    let mut res = alloc_ct(&params, module, params.k);
    module
        .ckks_eval_poly_const_coeffs::<_, _, CKKSCiphertext<BE::OwnedBuf>, _, _>(
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
    pb.populate(7, module, &tsk, &mut scratch.borrow())
        .expect("populate power basis for degree 7");

    let mut res = alloc_ct(&params, module, params.k);
    module
        .ckks_eval_poly_const_coeffs::<_, _, CKKSCiphertext<BE::OwnedBuf>, _, _>(
            &mut res,
            &bsgs,
            &pb,
            &tsk,
            &mut scratch.borrow(),
        )
        .expect("ckks_eval_poly_const_coeffs should succeed");

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

    let poly = chebyshev_interpolate(31, -F::one(), F::one(), |x: F| x.sin())
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
    pb.populate_chebyshev(31, module, &tsk, &mut scratch.borrow())
        .expect("populate Chebyshev power basis for degree 31");

    let mut res = alloc_ct(&params, module, params.k);
    module
        .ckks_eval_poly_const_coeffs::<_, _, CKKSCiphertext<BE::OwnedBuf>, _, _>(
            &mut res,
            &bsgs,
            &pb,
            &tsk,
            &mut scratch.borrow(),
        )
        .expect("ckks_eval_poly_const_coeffs should succeed");

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
