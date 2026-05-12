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
//! | [`test_dot_product_pt_vec_aligned`] | ct · ZNX plaintext |
//! | [`test_dot_product_const_aligned`] | ct · ZNX constant |

use poulpy_hal::{
    api::{NegacyclicFFT, NegacyclicFFTNew, ScratchOwnedBorrow},
    layouts::{HostBytesBackend, Module},
};

use crate::{layouts::plaintext::CKKSPlaintext, leveled::api::CKKSDotProductOps};

use super::helpers::{
    PT_PREC, TestContextBackend, TestContextModule, TestScalar, TestVector, alloc_ct, alloc_scratch, assert_decrypt_precision,
    assert_decrypt_precision_at_log_delta, ckks_encrypt, ckks_encrypt_with_prec, ckks_pt_cst_full, encode_and_upload_pt,
    gen_sk_with_raw, gen_tsk, precision_at, quantize, quantized_const, quantized_vector, test_vector_1, test_vector_2,
};

use crate::{encoding::reim::Encoder, test_suite::CKKSTestParams};

const N: usize = 3;
const DELTA_LOG_DELTA: usize = 8;

fn scaled<F: TestScalar>(v: &[F], scale: F) -> Vec<F> {
    v.iter().copied().map(|x| x * scale).collect()
}

fn three_vectors<F: TestScalar>(re1: &[F], im1: &[F], re2: &[F], im2: &[F]) -> [(Vec<F>, Vec<F>); 3] {
    let s = F::from_f64(0.5).unwrap();
    [
        (scaled(re1, s), scaled(im1, s)),
        (scaled(re2, s), scaled(im2, s)),
        (scaled(re1, s), scaled(im2, s)),
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

pub fn test_dot_product_ct_aligned<BE, F, E>(params: CKKSTestParams, module: &Module<BE>, host_module: &Module<HostBytesBackend>)
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

    let a_vecs = three_vectors(&re1, &im1, &re2, &im2);
    let b_vecs = three_vectors(&re1, &im1, &re2, &im2);

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
        .map(|(re, im)| {
            ckks_encrypt(
                &params,
                module,
                host_module,
                &encoder,
                &sk,
                params.k,
                re,
                im,
                &mut scratch.borrow(),
            )
        })
        .collect();
    let b_cts: Vec<_> = b_vecs
        .iter()
        .map(|(re, im)| {
            ckks_encrypt(
                &params,
                module,
                host_module,
                &encoder,
                &sk,
                params.k,
                re,
                im,
                &mut scratch.borrow(),
            )
        })
        .collect();
    let a_refs: Vec<&_> = a_cts.iter().collect();
    let b_refs: Vec<&_> = b_cts.iter().collect();

    let mut ct_res = alloc_ct(&params, module, params.k);
    module
        .ckks_dot_product_ct(&mut ct_res, &a_refs, &b_refs, &tsk, &mut scratch.borrow())
        .unwrap();
    assert_decrypt_precision(
        "dot_product_ct_aligned",
        &params,
        module,
        &encoder,
        &ct_res,
        &sk,
        &want_re,
        &want_im,
        &mut scratch.borrow(),
    );
}

pub fn test_dot_product_ct_unaligned<BE, F, E>(
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
    let (re2, im2) = test_vector_2::<F>(m);
    let (sk_raw, sk) = gen_sk_with_raw(&params, module, host_module, [0u8; 32]);
    let mut scratch = alloc_scratch(&params, module);
    let tsk = gen_tsk(&params, module, &sk_raw, &mut scratch.borrow());

    let a_vecs = three_vectors(&re1, &im1, &re2, &im2);
    let b_vecs = three_vectors(&re1, &im1, &re2, &im2);

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

    let smaller_k = params.k - params.base2k + 1;
    let a_cts: Vec<_> = a_vecs
        .iter()
        .enumerate()
        .map(|(i, (re, im))| {
            let k = if i == 1 { smaller_k } else { params.k };
            ckks_encrypt(&params, module, host_module, &encoder, &sk, k, re, im, &mut scratch.borrow())
        })
        .collect();
    let b_cts: Vec<_> = b_vecs
        .iter()
        .map(|(re, im)| {
            ckks_encrypt(
                &params,
                module,
                host_module,
                &encoder,
                &sk,
                params.k,
                re,
                im,
                &mut scratch.borrow(),
            )
        })
        .collect();
    let a_refs: Vec<&_> = a_cts.iter().collect();
    let b_refs: Vec<&_> = b_cts.iter().collect();

    let mut ct_res = alloc_ct(&params, module, params.k);
    module
        .ckks_dot_product_ct(&mut ct_res, &a_refs, &b_refs, &tsk, &mut scratch.borrow())
        .unwrap();
    assert_decrypt_precision(
        "dot_product_ct_unaligned",
        &params,
        module,
        &encoder,
        &ct_res,
        &sk,
        &want_re,
        &want_im,
        &mut scratch.borrow(),
    );
}

pub fn test_dot_product_ct_unaligned_b<BE, F, E>(
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
    let (re2, im2) = test_vector_2::<F>(m);
    let (sk_raw, sk) = gen_sk_with_raw(&params, module, host_module, [0u8; 32]);
    let mut scratch = alloc_scratch(&params, module);
    let tsk = gen_tsk(&params, module, &sk_raw, &mut scratch.borrow());

    let a_vecs = three_vectors(&re1, &im1, &re2, &im2);
    let b_vecs = three_vectors(&re1, &im1, &re2, &im2);

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

    let smaller_k = params.k - params.base2k + 1;
    let a_cts: Vec<_> = a_vecs
        .iter()
        .map(|(re, im)| {
            ckks_encrypt(
                &params,
                module,
                host_module,
                &encoder,
                &sk,
                params.k,
                re,
                im,
                &mut scratch.borrow(),
            )
        })
        .collect();
    let b_cts: Vec<_> = b_vecs
        .iter()
        .enumerate()
        .map(|(i, (re, im))| {
            let k = if i == 1 { smaller_k } else { params.k };
            ckks_encrypt(&params, module, host_module, &encoder, &sk, k, re, im, &mut scratch.borrow())
        })
        .collect();
    let a_refs: Vec<&_> = a_cts.iter().collect();
    let b_refs: Vec<&_> = b_cts.iter().collect();

    let mut ct_res = alloc_ct(&params, module, params.k);
    module
        .ckks_dot_product_ct(&mut ct_res, &a_refs, &b_refs, &tsk, &mut scratch.borrow())
        .unwrap();
    assert_decrypt_precision(
        "dot_product_ct_unaligned_b",
        &params,
        module,
        &encoder,
        &ct_res,
        &sk,
        &want_re,
        &want_im,
        &mut scratch.borrow(),
    );
}

pub fn test_dot_product_ct_delta_log_delta<BE, F, E>(
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
    let (sk_raw, sk) = gen_sk_with_raw(&params, module, host_module, [0u8; 32]);
    let mut scratch = alloc_scratch(&params, module);
    let tsk = gen_tsk(&params, module, &sk_raw, &mut scratch.borrow());

    let half = F::from_f64(0.5).unwrap();
    let low_log_delta = params.prec.log_delta - DELTA_LOG_DELTA;
    let low_prec = precision_at(&params, low_log_delta);
    let (a_hi_re, a_hi_im) = quantized_vector(host_module, &encoder, &params, TestVector::First, params.prec.log_delta);
    let (b_hi_re, b_hi_im) = quantized_vector(host_module, &encoder, &params, TestVector::Second, params.prec.log_delta);
    let (b_lo_re, b_lo_im) = quantized_vector(host_module, &encoder, &params, TestVector::Second, low_log_delta);
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
        .map(|(re, im)| {
            ckks_encrypt(
                &params,
                module,
                host_module,
                &encoder,
                &sk,
                params.k,
                re,
                im,
                &mut scratch.borrow(),
            )
        })
        .collect();
    let mut b_cts: Vec<_> = Vec::with_capacity(N);
    b_cts.push(ckks_encrypt_with_prec(
        &params,
        module,
        host_module,
        &encoder,
        &sk,
        params.k - DELTA_LOG_DELTA,
        &b_vecs[0].0,
        &b_vecs[0].1,
        low_prec,
        &mut scratch.borrow(),
    ));
    for (re, im) in &b_vecs[1..] {
        b_cts.push(ckks_encrypt(
            &params,
            module,
            host_module,
            &encoder,
            &sk,
            params.k,
            re,
            im,
            &mut scratch.borrow(),
        ));
    }
    let a_refs: Vec<&_> = a_cts.iter().collect();
    let b_refs: Vec<&_> = b_cts.iter().collect();

    let mut ct_res = alloc_ct(&params, module, params.k);
    module
        .ckks_dot_product_ct(&mut ct_res, &a_refs, &b_refs, &tsk, &mut scratch.borrow())
        .unwrap();
    assert_decrypt_precision_at_log_delta(
        "dot_product_ct_delta_log_delta",
        &params,
        module,
        &encoder,
        &ct_res,
        &sk,
        &want_re,
        &want_im,
        low_log_delta,
        &mut scratch.borrow(),
    );
}

pub fn test_dot_product_ct_smaller_output<BE, F, E>(
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
    let (re2, im2) = test_vector_2::<F>(m);
    let (sk_raw, sk) = gen_sk_with_raw(&params, module, host_module, [0u8; 32]);
    let mut scratch = alloc_scratch(&params, module);
    let tsk = gen_tsk(&params, module, &sk_raw, &mut scratch.borrow());

    let a_vecs = three_vectors(&re1, &im1, &re2, &im2);
    let b_vecs = three_vectors(&re1, &im1, &re2, &im2);

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
        .map(|(re, im)| {
            ckks_encrypt(
                &params,
                module,
                host_module,
                &encoder,
                &sk,
                params.k,
                re,
                im,
                &mut scratch.borrow(),
            )
        })
        .collect();
    let b_cts: Vec<_> = b_vecs
        .iter()
        .map(|(re, im)| {
            ckks_encrypt(
                &params,
                module,
                host_module,
                &encoder,
                &sk,
                params.k,
                re,
                im,
                &mut scratch.borrow(),
            )
        })
        .collect();
    let a_refs: Vec<&_> = a_cts.iter().collect();
    let b_refs: Vec<&_> = b_cts.iter().collect();

    let mut ct_res = alloc_ct(&params, module, params.k - params.base2k - 1);
    module
        .ckks_dot_product_ct(&mut ct_res, &a_refs, &b_refs, &tsk, &mut scratch.borrow())
        .unwrap();
    assert_decrypt_precision(
        "dot_product_ct smaller_output",
        &params,
        module,
        &encoder,
        &ct_res,
        &sk,
        &want_re,
        &want_im,
        &mut scratch.borrow(),
    );
}

pub fn test_dot_product_pt_vec_aligned<BE, F, E>(
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
    let (re2, im2) = test_vector_2::<F>(m);
    let sk = super::helpers::gen_sk(&params, module, host_module, [0u8; 32]);
    let mut scratch = alloc_scratch(&params, module);

    let a_vecs = three_vectors(&re1, &im1, &re2, &im2);
    let b_vecs = three_vectors(&re1, &im1, &re2, &im2);
    let b_pt_vecs: Vec<_> = b_vecs
        .iter()
        .map(|(re, im)| (quantize(re, params.prec.log_delta), quantize(im, params.prec.log_delta)))
        .collect();

    let mut want_re = vec![F::zero(); m];
    let mut want_im = vec![F::zero(); m];
    for i in 0..N {
        cmul_acc(
            &mut want_re,
            &mut want_im,
            &a_vecs[i].0,
            &a_vecs[i].1,
            &b_pt_vecs[i].0,
            &b_pt_vecs[i].1,
        );
    }

    let a_cts: Vec<_> = a_vecs
        .iter()
        .map(|(re, im)| {
            ckks_encrypt(
                &params,
                module,
                host_module,
                &encoder,
                &sk,
                params.k,
                re,
                im,
                &mut scratch.borrow(),
            )
        })
        .collect();
    let pts: Vec<_> = b_vecs
        .iter()
        .map(|(re, im)| encode_and_upload_pt(host_module, module, &encoder, params.base2k.into(), params.prec, re, im))
        .collect();
    let a_refs: Vec<&_> = a_cts.iter().collect();
    let pt_refs: Vec<&_> = pts.iter().collect();

    let mut ct_res = alloc_ct(&params, module, params.k);
    module
        .ckks_dot_product_pt_vec(&mut ct_res, &a_refs, &pt_refs, &mut scratch.borrow())
        .unwrap();
    assert_decrypt_precision(
        "dot_product_pt_vec_aligned",
        &params,
        module,
        &encoder,
        &ct_res,
        &sk,
        &want_re,
        &want_im,
        &mut scratch.borrow(),
    );
}

pub fn test_dot_product_const_aligned<BE, F, E>(
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
    let (re2, im2) = test_vector_2::<F>(m);
    let sk = super::helpers::gen_sk(&params, module, host_module, [0u8; 32]);
    let mut scratch = alloc_scratch(&params, module);

    let a_vecs = three_vectors(&re1, &im1, &re2, &im2);
    let const_coeffs: [f64; 3] = [0.25, 0.125, -0.3125];
    let quantized: Vec<F> = const_coeffs
        .iter()
        .map(|r| quantized_const::<F>(*r, 0.0, PT_PREC.log_delta).0)
        .collect();

    let mut want_re = vec![F::zero(); m];
    let mut want_im = vec![F::zero(); m];
    for i in 0..N {
        cmul_scalar_acc(
            &mut want_re,
            &mut want_im,
            &a_vecs[i].0,
            &a_vecs[i].1,
            quantized[i],
            F::zero(),
        );
    }

    let a_cts: Vec<_> = a_vecs
        .iter()
        .map(|(re, im)| {
            ckks_encrypt(
                &params,
                module,
                host_module,
                &encoder,
                &sk,
                params.k,
                re,
                im,
                &mut scratch.borrow(),
            )
        })
        .collect();
    let consts: Vec<CKKSPlaintext<_>> = const_coeffs
        .iter()
        .map(|r| ckks_pt_cst_full::<BE, F>(host_module, module, params.base2k.into(), PT_PREC, m, Some(*r), None))
        .collect();
    let pt_coeffs = vec![0usize; N];

    let a_refs: Vec<&_> = a_cts.iter().collect();
    let const_refs: Vec<&_> = consts.iter().collect();

    let mut ct_res = alloc_ct(&params, module, params.k);
    module
        .ckks_dot_product_pt_const(&mut ct_res, &a_refs, &const_refs, &pt_coeffs, &mut scratch.borrow())
        .unwrap();
    assert_decrypt_precision(
        "dot_product_const_aligned",
        &params,
        module,
        &encoder,
        &ct_res,
        &sk,
        &want_re,
        &want_im,
        &mut scratch.borrow(),
    );
}
