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

use poulpy_hal::{
    api::{NegacyclicFFT, NegacyclicFFTNew, ScratchOwnedBorrow},
    layouts::{HostBytesBackend, Module},
};

use crate::{CKKSInfos, leveled::api::CKKSAddManyOps, test_suite::helpers::assert_ct_meta};

use super::helpers::{
    TestContextBackend, TestContextModule, TestScalar, TestVector, alloc_ct, alloc_scratch, assert_decrypt_precision,
    assert_decrypt_precision_at_log_delta, ckks_encrypt, ckks_encrypt_with_prec, gen_sk, precision_at, quantized_vector,
    test_vector_1, test_vector_2,
};

use crate::{encoding::reim::Encoder, test_suite::CKKSTestParams};

const DELTA_LOG_DELTA: usize = 12;
const N: usize = 5;

type Terms<F> = (Vec<(Vec<F>, Vec<F>)>, Vec<F>, Vec<F>);

fn build_terms<F: TestScalar>(re1: &[F], im1: &[F], re2: &[F], im2: &[F], n: usize) -> Terms<F> {
    let m: usize = re1.len();
    let mut terms: Vec<(Vec<F>, Vec<F>)> = Vec::with_capacity(n);
    let mut want_re: Vec<F> = vec![F::zero(); m];
    let mut want_im: Vec<F> = vec![F::zero(); m];
    let scale = F::from_f64(1.0 / (2.0 * n as f64)).unwrap();
    for k in 0..n {
        let alpha = F::from_f64((k as f64 + 1.0) / (n as f64 + 1.0)).unwrap();
        let beta = F::from_f64(1.0).unwrap() - alpha;
        let re: Vec<F> = (0..m).map(|i| (alpha * re1[i] + beta * re2[i]) * scale).collect();
        let im: Vec<F> = (0..m).map(|i| (alpha * im1[i] + beta * im2[i]) * scale).collect();
        for i in 0..m {
            want_re[i] = want_re[i] + re[i];
            want_im[i] = want_im[i] + im[i];
        }
        terms.push((re, im));
    }
    (terms, want_re, want_im)
}

pub fn test_add_many_aligned<BE, F, E>(params: CKKSTestParams, module: &Module<BE>, host_module: &Module<HostBytesBackend>)
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
    let sk = gen_sk(&params, module, host_module, [0u8; 32]);
    let mut scratch = alloc_scratch(&params, module);

    let (terms, want_re, want_im) = build_terms(&re1, &im1, &re2, &im2, N);
    let cts: Vec<_> = terms
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
    let ct_refs: Vec<&_> = cts.iter().collect();
    let mut ct_res = alloc_ct(&params, module, params.k);
    module.ckks_add_many(&mut ct_res, &ct_refs, &mut scratch.borrow()).unwrap();
    let expected_log_delta: usize = cts.iter().map(|c| c.log_delta()).max().unwrap();
    let expected_log_budget: usize = cts.iter().map(|c| c.log_budget()).min().unwrap();
    assert_ct_meta("add_many_aligned", &ct_res, expected_log_delta, expected_log_budget);
    assert_decrypt_precision(
        "add_many_aligned",
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

pub fn test_add_many_single_smaller_output<BE, F, E>(
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
    let sk = gen_sk(&params, module, host_module, [0u8; 32]);
    let mut scratch = alloc_scratch(&params, module);

    let ct = ckks_encrypt(
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
    let ct_refs = vec![&ct];
    let mut ct_res = alloc_ct(&params, module, params.k - params.base2k - 1);
    module.ckks_add_many(&mut ct_res, &ct_refs, &mut scratch.borrow()).unwrap();
    assert_decrypt_precision(
        "add_many_single_smaller_output",
        &params,
        module,
        &encoder,
        &ct_res,
        &sk,
        &re1,
        &im1,
        &mut scratch.borrow(),
    );
}

pub fn test_add_many_unaligned_log_budget<BE, F, E>(
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
    let sk = gen_sk(&params, module, host_module, [0u8; 32]);
    let mut scratch = alloc_scratch(&params, module);

    let (terms, want_re, want_im) = build_terms(&re1, &im1, &re2, &im2, N);
    let smaller_k = params.k - params.base2k + 1;
    let cts: Vec<_> = terms
        .iter()
        .enumerate()
        .map(|(i, (re, im))| {
            let k = if i == 1 { smaller_k } else { params.k };
            ckks_encrypt(&params, module, host_module, &encoder, &sk, k, re, im, &mut scratch.borrow())
        })
        .collect();
    let ct_refs: Vec<&_> = cts.iter().collect();
    let mut ct_res = alloc_ct(&params, module, params.k);
    module.ckks_add_many(&mut ct_res, &ct_refs, &mut scratch.borrow()).unwrap();
    assert_decrypt_precision(
        "add_many unaligned_log_budget",
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

pub fn test_add_many_delta_log_delta<BE, F, E>(
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
    let sk = gen_sk(&params, module, host_module, [0u8; 32]);
    let mut scratch = alloc_scratch(&params, module);

    let low_log_delta = params.prec.log_delta - DELTA_LOG_DELTA;
    let low_prec = precision_at(&params, low_log_delta);
    let (hi_re, hi_im) = quantized_vector(host_module, &encoder, &params, TestVector::First, params.prec.log_delta);
    let (lo_re, lo_im) = quantized_vector(host_module, &encoder, &params, TestVector::Second, low_log_delta);

    let scale = F::from_f64(1.0 / (2.0 * N as f64)).unwrap();
    let hi_scaled: (Vec<F>, Vec<F>) = (
        hi_re.iter().copied().map(|x| x * scale).collect(),
        hi_im.iter().copied().map(|x| x * scale).collect(),
    );
    let lo_scaled: (Vec<F>, Vec<F>) = (
        lo_re.iter().copied().map(|x| x * scale).collect(),
        lo_im.iter().copied().map(|x| x * scale).collect(),
    );

    let ct_low = ckks_encrypt_with_prec(
        &params,
        module,
        host_module,
        &encoder,
        &sk,
        params.k - DELTA_LOG_DELTA,
        &lo_scaled.0,
        &lo_scaled.1,
        low_prec,
        &mut scratch.borrow(),
    );
    let cts_hi: Vec<_> = (0..N - 1)
        .map(|_| {
            ckks_encrypt(
                &params,
                module,
                host_module,
                &encoder,
                &sk,
                params.k,
                &hi_scaled.0,
                &hi_scaled.1,
                &mut scratch.borrow(),
            )
        })
        .collect();
    let mut ct_refs: Vec<&_> = Vec::with_capacity(N);
    ct_refs.push(&ct_low);
    for c in &cts_hi {
        ct_refs.push(c);
    }

    let mut want_re: Vec<F> = vec![F::zero(); m];
    let mut want_im: Vec<F> = vec![F::zero(); m];
    for i in 0..m {
        want_re[i] = lo_scaled.0[i] + hi_scaled.0[i] * F::from_f64((N - 1) as f64).unwrap();
        want_im[i] = lo_scaled.1[i] + hi_scaled.1[i] * F::from_f64((N - 1) as f64).unwrap();
    }

    let mut ct_res = alloc_ct(&params, module, params.k);
    module.ckks_add_many(&mut ct_res, &ct_refs, &mut scratch.borrow()).unwrap();
    assert_decrypt_precision_at_log_delta(
        "add_many delta_log_delta",
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

pub fn test_add_many_smaller_output<BE, F, E>(params: CKKSTestParams, module: &Module<BE>, host_module: &Module<HostBytesBackend>)
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
    let sk = gen_sk(&params, module, host_module, [0u8; 32]);
    let mut scratch = alloc_scratch(&params, module);

    let (terms, want_re, want_im) = build_terms(&re1, &im1, &re2, &im2, N);
    let cts: Vec<_> = terms
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
    let ct_refs: Vec<&_> = cts.iter().collect();
    let mut ct_res = alloc_ct(&params, module, params.k - params.base2k - 1);
    module.ckks_add_many(&mut ct_res, &ct_refs, &mut scratch.borrow()).unwrap();
    assert_decrypt_precision(
        "add_many smaller_output",
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
