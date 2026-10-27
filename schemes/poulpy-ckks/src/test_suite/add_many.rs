//! Tests for `CKKSAddManyOps::ckks_add_many`.
//!
//! # Test inventory
//!
//! | Function | Path exercised |
//! |----------|----------------|
//! | [`test_add_many_aligned`] | all inputs at the same `log_budget` / `log_delta` |
//! | [`test_add_many_single_smaller_output`] | one input, at its width and into a narrower output |
//! | [`test_add_many_unaligned_log_budget`] | one input rescaled by one limb |
//! | [`test_add_many_delta_log_delta`] | inputs at different `log_delta` |
//! | [`test_add_many_smaller_output`] | output narrower than inputs (`offset > 0`) |
//! | [`test_add_many_normalize_after`] | normalization after every `normalize_after` additions, aligned, into a narrower output and at unaligned budgets |
//! | [`test_add_many_rejects_invalid_calls`] | a zero `normalize_after`, an empty slice, a term of another radix or rank at every position and an arena one byte below the query |

use poulpy_core::{
    GLWENormalize,
    layouts::{GLWELayout, LWEInfos, Rank},
};
use poulpy_hal::{
    api::{NegacyclicFFT, NegacyclicFFTNew, ScratchOwnedBorrow},
    layouts::{HostBytesBackend, Module, Standard},
};

use crate::{
    CKKSCompositionError, CKKSError, CKKSInfos, CKKSLayout, CKKSResult,
    api::CKKSAddManyOps,
    layouts::CKKSCiphertextOwned,
    test_suite::{
        helpers::{assert_ckks_error, assert_ct_meta, dirty_ct},
        parity::helpers::{fixture_ciphertext, snapshot, with_scratch},
    },
};

use super::helpers::{
    TestContextBackend, TestContextModule, TestScalar, TestVector, alloc_ct, alloc_scratch, assert_decrypt_precision,
    assert_decrypt_precision_at_log_delta, ckks_encrypt, ckks_encrypt_with_prec, gen_sk, precision_at, quantized_vector,
    test_vector_1, test_vector_2,
};

use crate::{test_suite::CKKSTestParams, test_suite::reference_encoder::ReferenceEncoder};

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

/// `ckks_add_many` on exactly the scratch its query reports for `res` and `a[0]`.
fn add_many<BE: TestContextBackend>(
    module: &Module<BE>,
    res: &mut CKKSCiphertextOwned<BE>,
    a: &[&CKKSCiphertextOwned<BE>],
    normalize_after: usize,
) -> CKKSResult<()>
where
    Module<BE>: CKKSAddManyOps<BE>,
{
    with_scratch::<BE, _>(module.ckks_add_many_tmp_bytes(&*res, a[0]), |s| {
        module.ckks_add_many(normalize_after, res, a, s)
    })
}

pub fn test_add_many_aligned<BE, F, E>(params: CKKSTestParams, module: &Module<BE>, host_module: &Module<HostBytesBackend>)
where
    BE: TestContextBackend<Ring = Standard>,
    for<'a> <BE as poulpy_hal::layouts::Backend>::BufRef<'a>: poulpy_hal::layouts::HostDataRef,
    for<'a> <BE as poulpy_hal::layouts::Backend>::BufMut<'a>: poulpy_hal::layouts::HostDataMut,
    Module<BE>: TestContextModule<BE>,
    F: TestScalar,
    E: NegacyclicFFT<F> + NegacyclicFFTNew<F>,
{
    let m = params.n / 2;
    let encoder = ReferenceEncoder::<E>::new(m).unwrap();
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
    add_many(module, &mut ct_res, &ct_refs, usize::MAX).unwrap();
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
    BE: TestContextBackend<Ring = Standard>,
    for<'a> <BE as poulpy_hal::layouts::Backend>::BufRef<'a>: poulpy_hal::layouts::HostDataRef,
    for<'a> <BE as poulpy_hal::layouts::Backend>::BufMut<'a>: poulpy_hal::layouts::HostDataMut,
    Module<BE>: TestContextModule<BE>,
    F: TestScalar,
    E: NegacyclicFFT<F> + NegacyclicFFTNew<F>,
{
    let m = params.n / 2;
    let encoder = ReferenceEncoder::<E>::new(m).unwrap();
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
    // At the input's width, one input is a lossless copy, still an evaluation.
    let mut ct_same = alloc_ct(&params, module, params.k);
    add_many(module, &mut ct_same, &ct_refs, usize::MAX).unwrap();
    assert!(ct.noise().is_some());
    assert!(ct_same.noise().is_none());
    let mut ct_res = alloc_ct(&params, module, params.k - params.base2k - 1);
    add_many(module, &mut ct_res, &ct_refs, usize::MAX).unwrap();
    assert!(ct_res.noise().is_none());
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
    BE: TestContextBackend<Ring = Standard>,
    for<'a> <BE as poulpy_hal::layouts::Backend>::BufRef<'a>: poulpy_hal::layouts::HostDataRef,
    for<'a> <BE as poulpy_hal::layouts::Backend>::BufMut<'a>: poulpy_hal::layouts::HostDataMut,
    Module<BE>: TestContextModule<BE>,
    F: TestScalar,
    E: NegacyclicFFT<F> + NegacyclicFFTNew<F>,
{
    let m = params.n / 2;
    let encoder = ReferenceEncoder::<E>::new(m).unwrap();
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
    add_many(module, &mut ct_res, &ct_refs, usize::MAX).unwrap();
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
    BE: TestContextBackend<Ring = Standard>,
    for<'a> <BE as poulpy_hal::layouts::Backend>::BufRef<'a>: poulpy_hal::layouts::HostDataRef,
    for<'a> <BE as poulpy_hal::layouts::Backend>::BufMut<'a>: poulpy_hal::layouts::HostDataMut,
    Module<BE>: TestContextModule<BE>,
    F: TestScalar,
    E: NegacyclicFFT<F> + NegacyclicFFTNew<F>,
{
    let m = params.n / 2;
    let encoder = ReferenceEncoder::<E>::new(m).unwrap();
    let sk = gen_sk(&params, module, host_module, [0u8; 32]);
    let mut scratch = alloc_scratch(&params, module);

    let low_log_delta = params.prec().log_delta() - DELTA_LOG_DELTA;
    let low_prec = precision_at(&params, low_log_delta);
    let (hi_re, hi_im) = quantized_vector(host_module, &encoder, &params, TestVector::First, params.prec().log_delta());
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
    add_many(module, &mut ct_res, &ct_refs, usize::MAX).unwrap();
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
    BE: TestContextBackend<Ring = Standard>,
    for<'a> <BE as poulpy_hal::layouts::Backend>::BufRef<'a>: poulpy_hal::layouts::HostDataRef,
    for<'a> <BE as poulpy_hal::layouts::Backend>::BufMut<'a>: poulpy_hal::layouts::HostDataMut,
    Module<BE>: TestContextModule<BE>,
    F: TestScalar,
    E: NegacyclicFFT<F> + NegacyclicFFTNew<F>,
{
    let m = params.n / 2;
    let encoder = ReferenceEncoder::<E>::new(m).unwrap();
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
    add_many(module, &mut ct_res, &ct_refs, usize::MAX).unwrap();
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

/// Eight inputs normalized after every addition, and after every two, normalize to the digits of the sum never
/// normalized, aligned, into a narrower output and with one input a limb narrower, and decrypt to the sum, each within
/// exactly the scratch its query reports; aligned, they hold at most two and three canonical digits per coefficient,
/// where never normalized they exceed three.
pub fn test_add_many_normalize_after<BE, F, E>(
    params: CKKSTestParams,
    module: &Module<BE>,
    host_module: &Module<HostBytesBackend>,
) where
    BE: TestContextBackend<Ring = Standard>,
    for<'a> <BE as poulpy_hal::layouts::Backend>::BufRef<'a>: poulpy_hal::layouts::HostDataRef,
    for<'a> <BE as poulpy_hal::layouts::Backend>::BufMut<'a>: poulpy_hal::layouts::HostDataMut,
    Module<BE>: TestContextModule<BE>,
    F: TestScalar,
    E: NegacyclicFFT<F> + NegacyclicFFTNew<F>,
{
    let m = params.n / 2;
    let encoder = ReferenceEncoder::<E>::new(m).unwrap();
    let (re1, im1) = test_vector_1::<F>(m);
    let (re2, im2) = test_vector_2::<F>(m);
    let sk = gen_sk(&params, module, host_module, [0u8; 32]);
    let mut scratch = alloc_scratch(&params, module);

    let (terms, want_re, want_im) = build_terms(&re1, &im1, &re2, &im2, 8);
    let encrypt = |k: usize, (re, im): &(Vec<F>, Vec<F>), scratch: &mut poulpy_hal::layouts::ScratchOwned<BE>| {
        ckks_encrypt(&params, module, host_module, &encoder, &sk, k, re, im, &mut scratch.borrow())
    };
    let cts: Vec<_> = terms.iter().map(|term| encrypt(params.k, term, &mut scratch)).collect();
    let narrower_term = encrypt(params.k - params.base2k + 1, &terms[1], &mut scratch);
    let aligned: Vec<&_> = cts.iter().collect();
    let mut unaligned = aligned.clone();
    unaligned[1] = &narrower_term;

    let digit = 1i64 << (params.base2k - 1);
    let narrow_k = params.k - params.base2k - 1;
    for (case, a, res_k) in [
        ("aligned", &aligned, params.k),
        ("smaller output", &aligned, narrow_k),
        ("unaligned budgets", &unaligned, params.k),
    ] {
        let mut sum = |normalize_after: usize| {
            let mut ct_res = alloc_ct(&params, module, res_k);
            add_many(module, &mut ct_res, a, normalize_after).unwrap();
            let lazy = snapshot::<BE, _>(&ct_res).digits;
            assert_decrypt_precision(
                &format!("add_many {case} normalize_after={normalize_after}"),
                &params,
                module,
                &encoder,
                &ct_res,
                &sk,
                &want_re,
                &want_im,
                &mut scratch.borrow(),
            );
            module.glwe_normalize_assign(&mut ct_res, &mut scratch.borrow());
            (lazy, snapshot::<BE, _>(&ct_res).digits)
        };
        let (never, never_normalized) = sum(usize::MAX);
        if case == "aligned" {
            assert!(
                never.iter().any(|x| x.abs() > 3 * digit),
                "the sum never normalized stays within three digits, so the bounds below test nothing"
            );
        }
        for (normalize_after, held) in [(1, 2), (2, 3)] {
            let (lazy, normalized) = sum(normalize_after);
            if case == "aligned" {
                assert!(
                    lazy.iter().all(|x| x.abs() <= held * digit),
                    "normalize_after={normalize_after}: a coefficient exceeds {held} digits"
                );
            }
            assert!(
                normalized == never_normalized,
                "{case} normalize_after={normalize_after}: normalizing changed the sum"
            );
        }
    }
}

/// A zero `normalize_after`, checked before an empty slice, an empty slice and a term of another radix or rank than
/// `res`, at every position, are refused with `InvalidOperand` on an empty arena, and one byte below the query, for one input or several, returns `CKKSError::Internal`, each before
/// `res` changes.
pub fn test_add_many_rejects_invalid_calls<BE, F, E>(
    params: CKKSTestParams,
    module: &Module<BE>,
    host_module: &Module<HostBytesBackend>,
) where
    BE: TestContextBackend<Ring = Standard>,
    for<'a> <BE as poulpy_hal::layouts::Backend>::BufRef<'a>: poulpy_hal::layouts::HostDataRef,
    for<'a> <BE as poulpy_hal::layouts::Backend>::BufMut<'a>: poulpy_hal::layouts::HostDataMut,
    Module<BE>: TestContextModule<BE>,
    F: TestScalar,
    E: NegacyclicFFT<F> + NegacyclicFFTNew<F>,
{
    let m = params.n / 2;
    let encoder = ReferenceEncoder::<E>::new(m).unwrap();
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

    let mut res = dirty_ct(module, &ct, params.k);
    let before = snapshot::<BE, _>(&res);
    let refused = |label: &str, result: CKKSResult<()>, res: &CKKSCiphertextOwned<BE>, check: &dyn Fn(&CKKSError)| {
        check(&result.expect_err(label));
        assert!(snapshot::<BE, _>(res) == before, "{label}: wrote the destination");
    };
    let invalid = |reason: &str| {
        let want = CKKSCompositionError::InvalidOperand {
            op: "ckks_add_many",
            reason: reason.into(),
        };
        move |err: &CKKSError| assert_ckks_error("ckks_add_many", err, want.clone())
    };
    let empty: Vec<&CKKSCiphertextOwned<BE>> = Vec::new();
    let several = [&ct, &ct, &ct];
    for (label, a, reason) in [
        ("zero normalize_after", &several[..], "normalize_after must be at least 1"),
        (
            "zero normalize_after and empty",
            &empty[..],
            "normalize_after must be at least 1",
        ),
    ] {
        let result = with_scratch::<BE, _>(0, |s| module.ckks_add_many(0, &mut res, a, s));
        refused(label, result, &res, &invalid(reason));
    }
    let result = with_scratch::<BE, _>(0, |s| module.ckks_add_many(1, &mut res, &empty, s));
    refused("empty", result, &res, &invalid("a must contain at least one ciphertext"));

    let term_like = |base2k: usize, rank: usize, seed: u8| {
        let layout = CKKSLayout {
            glwe_layout: GLWELayout {
                base2k: base2k.into(),
                rank: Rank(rank as u32),
                ..params.glwe_layout()
            },
            meta: ct.meta(),
        };
        fixture_ciphertext(module, &layout, seed)
    };
    let other_radix = term_like(params.base2k - 1, params.rank, 8);
    let other_rank = term_like(params.base2k, params.rank + 1, 9);
    for (label, term, reason) in [
        (
            "other radix",
            &other_radix,
            format!("has radix {} where res has {}", params.base2k - 1, params.base2k),
        ),
        (
            "other rank",
            &other_rank,
            format!("has rank {} where res has {}", params.rank + 1, params.rank),
        ),
    ] {
        for i in 0..3 {
            let mut a = [&ct, &ct, &ct];
            a[i] = term;
            let result = with_scratch::<BE, _>(module.ckks_add_many_tmp_bytes(&res, a[0]), |s| {
                module.ckks_add_many(1, &mut res, &a, s)
            });
            refused(
                &format!("{label} at {i}"),
                result,
                &res,
                &invalid(&format!("term {i} {reason}")),
            );
            let result = with_scratch::<BE, _>(0, |s| module.ckks_add_many(1, &mut res, &a, s));
            refused(
                &format!("{label} at {i}, empty arena"),
                result,
                &res,
                &invalid(&format!("term {i} {reason}")),
            );
        }
    }

    let internal = |err: &CKKSError| {
        assert!(
            matches!(err, CKKSError::Internal(_)) && err.to_string().contains("ckks_add_many: scratch below its query"),
            "unexpected error: {err}"
        )
    };
    for (label, a) in [("one input", &several[..1]), ("three inputs", &several[..])] {
        let bytes = module.ckks_add_many_tmp_bytes(&res, a[0]);
        let result = with_scratch::<BE, _>(bytes - 1, |s| module.ckks_add_many(1, &mut res, a, s));
        refused(&format!("{label} one byte short"), result, &res, &internal);
    }
}
