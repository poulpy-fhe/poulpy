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
//! | [`test_dot_product_ct_lazy`] | ct · ct, one relinearization: bits, metadata, precision, rejections |
//! | [`test_dot_product_ct_long_sum`] | ct · ct, one relinearization past `2^(63 - base2k)` terms |
//! | [`test_dot_product_ct_non_uniform`] | ct · ct, pairs without a shared layout or scale |
//! | [`test_dot_product_pt_vec_aligned`] | ct · ZNX plaintext |
//! | [`test_dot_product_const_aligned`] | ct · ZNX constant |

use poulpy_core::{
    CoreError, GLWEAdd, GLWENormalize, GLWETensoring, ScratchArenaTakeCore,
    layouts::{
        GLWEInfos, GLWELayout, GetTensorKey, LWEInfos, SetGLWEPlaintextInfos, TorusPrecision,
        prepared::GLWETensorKeyPreparedBackendRef,
    },
};
use poulpy_hal::{
    api::{NegacyclicFFT, NegacyclicFFTNew, ScratchOwnedAlloc, ScratchOwnedBorrow},
    layouts::{Backend, HostBytesBackend, Module, ScratchArena, ScratchOwned, Standard},
};

use crate::{
    CKKSCompositionError, CKKSError, CKKSInfos, Scale, SetCKKSInfos, SlotsKind,
    api::{CKKSAddOps, CKKSDotProductOps, CKKSMulOps},
    layouts::{CKKSCiphertextOwned, ScratchArenaTakeCKKS, plaintext::CKKSPlaintext},
    reference::mul::get_mul_ct_params,
};

use super::helpers::{
    PT_PREC, TestContextBackend, TestContextModule, TestScalar, TestVector, alloc_ct, alloc_scratch, assert_ckks_error,
    assert_decrypt_precision, assert_decrypt_precision_at_log_delta, ckks_decrypt_decode, ckks_encrypt, ckks_encrypt_with_prec,
    ckks_pt_cst_full, encode_and_upload_pt, gen_sk_with_raw, gen_tsk, precision_at, precision_stats, quantize, quantized_const,
    quantized_vector, test_vector_1, test_vector_2, want_mul,
};
use super::parity::helpers::{snapshot, with_scratch};

use crate::{test_suite::CKKSTestParams, test_suite::reference_encoder::ReferenceEncoder};

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
    BE: TestContextBackend<Ring = Standard>,
    for<'a> <BE as poulpy_hal::layouts::Backend>::BufRef<'a>: poulpy_hal::layouts::HostDataRef,
    for<'a> <BE as poulpy_hal::layouts::Backend>::BufMut<'a>: poulpy_hal::layouts::HostDataMut,
    Module<BE>: TestContextModule<BE>,
    F: TestScalar,
    E: NegacyclicFFT<F> + NegacyclicFFTNew<F>,
{
    let m = params.n / 2;
    let encoder = ReferenceEncoder::<E>::new(m).unwrap();
    let (sk_raw, sk) = gen_sk_with_raw(&params, module, host_module, [0u8; 32]);
    let mut scratch = alloc_scratch(&params, module);
    let tsk = gen_tsk(&params, module, &sk_raw, &mut scratch.borrow());

    let half = F::from_f64(0.5).unwrap();
    let low_log_delta = params.prec().log_delta() - DELTA_LOG_DELTA;
    let low_prec = precision_at(&params, low_log_delta);
    let (a_hi_re, a_hi_im) = quantized_vector(host_module, &encoder, &params, TestVector::First, params.prec().log_delta());
    let (b_hi_re, b_hi_im) = quantized_vector(host_module, &encoder, &params, TestVector::Second, params.prec().log_delta());
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

/// The single-relinearization circuit from core steps: each pair's tensor product at `ckks_mul_into`'s
/// width `max(a.k, b.k)`, added and normalized one at a time, then `dst` stamped with the ct × ct rule
/// and relinearized once.
fn lazy_reference<BE: TestContextBackend, H: GetTensorKey<BE>>(
    module: &Module<BE>,
    dst: &mut CKKSCiphertextOwned<BE>,
    a: &[&CKKSCiphertextOwned<BE>],
    b: &[&CKKSCiphertextOwned<BE>],
    tsk: &H,
    scratch: &mut ScratchArena<'_, BE>,
) where
    Module<BE>: GLWETensoring<BE> + GLWEAdd<BE> + GLWENormalize<BE>,
{
    let layout = GLWELayout {
        n: dst.n(),
        base2k: dst.base2k(),
        k: a[0].k().max(b[0].k()),
        rank: dst.rank(),
    };
    let (log_budget, log_delta, cnv_offset) = get_mul_ct_params(&*dst, a[0], b[0]).unwrap();
    let scratch = scratch.borrow();
    let (mut acc, scratch) = scratch.take_glwe_tensor_scratch(&layout);
    let (mut term, mut scratch) = scratch.take_glwe_tensor_scratch(&layout);
    module.glwe_tensor_apply(cnv_offset, &mut acc, a[0], b[0], &mut scratch);
    for (ai, bi) in a.iter().zip(b).skip(1) {
        module.glwe_tensor_apply(cnv_offset, &mut term, *ai, *bi, &mut scratch);
        module.glwe_add_assign(&mut acc, &term);
        module.glwe_normalize_assign(&mut acc, &mut scratch);
    }
    dst.set_log_budget(log_budget);
    dst.set_log_delta(log_delta);
    module.glwe_tensor_relinearize(dst, &acc, tsk, &mut scratch);
}

/// The eager path as `ckks_dot_product_ct` ran it before the lazy one: the first product into
/// `dst`, then every other product into one temporary shaped like `dst` and added.
fn eager_reference<BE: TestContextBackend, H: GetTensorKey<BE>>(
    module: &Module<BE>,
    dst: &mut CKKSCiphertextOwned<BE>,
    a: &[&CKKSCiphertextOwned<BE>],
    b: &[&CKKSCiphertextOwned<BE>],
    tsk: &H,
    scratch: &mut ScratchArena<'_, BE>,
) where
    Module<BE>: TestContextModule<BE>,
{
    module.ckks_mul_into(dst, a[0], b[0], tsk, scratch).unwrap();
    scratch.scope(|scratch| {
        let (mut tmp, mut scratch) = scratch.take_ckks_ciphertext_like_scratch(&*dst);
        for (ai, bi) in a.iter().zip(b).skip(1) {
            module.ckks_mul_into(&mut tmp, *ai, *bi, tsk, &mut scratch).unwrap();
            module.ckks_add_assign(dst, &tmp, &mut scratch).unwrap();
        }
    });
}

/// A tensor-key store that holds its key at one width only.
struct TensorKeyAt<'a, K>(&'a K, TorusPrecision);

impl<BE: Backend, K: GetTensorKey<BE>> GetTensorKey<BE> for TensorKeyAt<'_, K> {
    fn get_tensor_key(&self, k: TorusPrecision) -> poulpy_core::Result<GLWETensorKeyPreparedBackendRef<'_, BE>> {
        if k != self.1 {
            return Err(CoreError::GGLWEKeyUse {
                op: "get_tensor_key",
                detail: format!("no tensor key at k={}", k.as_u32()),
            });
        }
        self.0.get_tensor_key(k)
    }
}

/// Runs a dot product expected to fail on `bytes` of poisoned scratch, checks that `dst` is
/// untouched, and returns the error.
fn assert_rejected<BE: TestContextBackend, H: GetTensorKey<BE>>(
    module: &Module<BE>,
    dst: &mut CKKSCiphertextOwned<BE>,
    a: &[&CKKSCiphertextOwned<BE>],
    b: &[&CKKSCiphertextOwned<BE>],
    tsk: &H,
    bytes: usize,
) -> CKKSError
where
    Module<BE>: TestContextModule<BE>,
{
    let before = snapshot::<BE, _>(&*dst);
    let err = with_scratch::<BE, _>(bytes, |scratch| module.ckks_dot_product_ct(dst, a, b, tsk, scratch)).unwrap_err();
    assert_eq!(snapshot::<BE, _>(&*dst), before, "a rejected dot product wrote dst: {err}");
    err
}

/// `-log2` of the mean slot error of `ct` against `want`, over the real and imaginary parts.
#[allow(clippy::too_many_arguments)]
fn log2_precision<BE, F, E>(
    params: &CKKSTestParams,
    module: &Module<BE>,
    encoder: &ReferenceEncoder<E>,
    ct: &CKKSCiphertextOwned<BE>,
    sk: &poulpy_core::layouts::prepared::GLWESecretPrepared<BE::OwnedBuf, BE>,
    want_re: &[F],
    want_im: &[F],
    scratch: &mut ScratchArena<'_, BE>,
) -> f64
where
    BE: TestContextBackend,
    Module<BE>: TestContextModule<BE>,
    F: TestScalar,
    E: NegacyclicFFT<F>,
{
    let (re, im) = ckks_decrypt_decode(params, module, encoder, ct, sk, scratch);
    let got: Vec<F> = re.into_iter().chain(im).collect();
    let want: Vec<F> = want_re.iter().chain(want_im).copied().collect();
    precision_stats(&got, &want, ct.log_delta()).avg_log2_prec
}

/// Pairs that share one layout and scale per side take one relinearization, on the exact poisoned
/// budget, at four widths (all equal, a narrower `dst`, `b` narrower than `a`, `dst` wider than both) and
/// with `b` at a lower scale than `a`.
///
/// The result is a test-local composition of the core steps bit for bit, which also normalizes after
/// every term. Its metadata and noise tag are the eager path's: the smaller scale, the smallest label and
/// the joined slot kind of all six operands. Its precision is at least the eager path's, and it is canonical.
/// The tensor key is looked up at `max(a.k, b.k)` only, as `ckks_mul_into` does, whatever `dst`'s width.
/// Wrong lengths, a missing tensor key, inputs without the budget and a too narrow `dst` are rejected
/// with `dst` untouched.
pub fn test_dot_product_ct_lazy<BE, F, E>(params: CKKSTestParams, module: &Module<BE>, host_module: &Module<HostBytesBackend>)
where
    BE: TestContextBackend<Ring = Standard>,
    for<'a> <BE as poulpy_hal::layouts::Backend>::BufRef<'a>: poulpy_hal::layouts::HostDataRef,
    for<'a> <BE as poulpy_hal::layouts::Backend>::BufMut<'a>: poulpy_hal::layouts::HostDataMut,
    Module<BE>: TestContextModule<BE> + GLWETensoring<BE> + GLWEAdd<BE>,
    F: TestScalar,
    E: NegacyclicFFT<F> + NegacyclicFFTNew<F>,
{
    let m = params.n / 2;
    let encoder = ReferenceEncoder::<E>::new(m).unwrap();
    let (re1, im1) = test_vector_1::<F>(m);
    let (re2, im2) = test_vector_2::<F>(m);
    let (sk_raw, sk) = gen_sk_with_raw(&params, module, host_module, [0u8; 32]);
    let mut scratch = alloc_scratch(&params, module);
    let tsk = gen_tsk(&params, module, &sk_raw, &mut scratch.borrow());

    let vecs = three_vectors(&re1, &im1, &re2, &im2);
    let mut want_re = vec![F::zero(); m];
    let mut want_im = vec![F::zero(); m];
    for (re, im) in &vecs {
        cmul_acc(&mut want_re, &mut want_im, re, im, re, im);
    }

    let (k, base2k, log_delta) = (params.k, params.base2k, params.prec().log_delta());
    // (dst.k, a.k, b.k, b's scale). The lower scale keeps `b`'s budget, as in `test_dot_product_ct_delta_log_delta`.
    let cases = [
        (k, k, k, log_delta),
        (k - base2k - 1, k, k, log_delta),
        (k, k, k - base2k + 1, log_delta),
        (k, k - base2k + 1, k - 2 * base2k + 1, log_delta),
        (k, k, k - DELTA_LOG_DELTA, log_delta - DELTA_LOG_DELTA),
    ];
    for (case, (dst_k, a_k, b_k, b_log_delta)) in cases.into_iter().enumerate() {
        let label = format!("dot_product_ct_lazy dst.k={dst_k} a.k={a_k} b.k={b_k} b.log_delta={b_log_delta}");
        let mut encrypt = |k: usize, log_delta: usize| -> Vec<_> {
            let prec = precision_at(&params, log_delta);
            vecs.iter()
                .map(|(re, im)| {
                    ckks_encrypt_with_prec(
                        &params,
                        module,
                        host_module,
                        &encoder,
                        &sk,
                        k,
                        re,
                        im,
                        prec,
                        &mut scratch.borrow(),
                    )
                })
                .collect()
        };
        let (mut a_cts, mut b_cts) = (encrypt(a_k, log_delta), encrypt(b_k, b_log_delta));
        // Label 2 but 1 on the last `b`, Integer slots but Real on the last `a`: the stamp has to read every
        // pair, and neither the label nor the slot kind is a fresh `dst`'s.
        for ct in a_cts.iter_mut().chain(b_cts.iter_mut()) {
            ct.set_log_sparsity(2);
            ct.set_slots(SlotsKind::Integer);
        }
        a_cts[N - 1].set_slots(SlotsKind::Real);
        b_cts[N - 1].set_log_sparsity(1);
        let a_refs: Vec<&_> = a_cts.iter().collect();
        let b_refs: Vec<&_> = b_cts.iter().collect();

        let mut lazy = alloc_ct(&params, module, dst_k);
        let bytes = module.ckks_dot_product_ct_tmp_bytes(N, &lazy, a_refs[0], b_refs[0], &tsk);
        with_scratch::<BE, _>(bytes, |scratch| {
            module.ckks_dot_product_ct(&mut lazy, &a_refs, &b_refs, &tsk, scratch)
        })
        .unwrap();
        let have = snapshot::<BE, _>(&lazy);

        let mut reference = alloc_ct(&params, module, dst_k);
        lazy_reference(
            module,
            &mut reference,
            &a_refs,
            &b_refs,
            &tsk,
            &mut ScratchOwned::<BE>::alloc(bytes).borrow(),
        );
        let reference = snapshot::<BE, _>(&reference);
        assert_eq!(
            (have.layout.glwe_layout, &have.digits),
            (reference.layout.glwe_layout, &reference.digits),
            "{label}: not the core composition"
        );

        let mut eager = alloc_ct(&params, module, dst_k);
        eager_reference(
            module,
            &mut eager,
            &a_refs,
            &b_refs,
            &tsk,
            &mut ScratchOwned::<BE>::alloc(bytes).borrow(),
        );
        let want = snapshot::<BE, _>(&eager);
        assert_eq!(
            (have.layout, &have.noise),
            (want.layout, &want.noise),
            "{label}: metadata differs from the eager path's"
        );
        assert_eq!(
            (have.layout.meta.log_sparsity, have.layout.meta.slots, have.layout.meta.scale),
            (1, SlotsKind::Real, Scale::Log(log_delta.min(b_log_delta))),
            "{label}: label, slot kind and scale"
        );
        assert!(have.canonical, "{label}: a relinearized sum is canonical");

        assert_decrypt_precision(
            &label,
            &params,
            module,
            &encoder,
            &lazy,
            &sk,
            &want_re,
            &want_im,
            &mut scratch.borrow(),
        );
        let lazy_prec = log2_precision(
            &params,
            module,
            &encoder,
            &lazy,
            &sk,
            &want_re,
            &want_im,
            &mut scratch.borrow(),
        );
        let eager_prec = log2_precision(
            &params,
            module,
            &encoder,
            &eager,
            &sk,
            &want_re,
            &want_im,
            &mut scratch.borrow(),
        );
        // Strict: the lazy sum rounds and relinearizes once, the eager one once per term. Every FFT64 and
        // NTT4x30 suite measured the lazy path 0.37 to 0.84 bits ahead.
        assert!(
            lazy_prec >= eager_prec,
            "{label}: precision {lazy_prec:.2} bits < eager {eager_prec:.2} bits"
        );

        let key_k = TorusPrecision(a_k.max(b_k) as u32);
        let mut keyed = alloc_ct(&params, module, dst_k);
        with_scratch::<BE, _>(bytes, |scratch| {
            module.ckks_dot_product_ct(&mut keyed, &a_refs, &b_refs, &TensorKeyAt(&tsk, key_k), scratch)
        })
        .unwrap();
        assert_eq!(snapshot::<BE, _>(&keyed), have, "{label}: a key held at {key_k:?} only");
        if dst_k > a_k.max(b_k) {
            let at_dst = TensorKeyAt(&tsk, TorusPrecision(dst_k as u32));
            let err = assert_rejected(module, &mut lazy, &a_refs, &b_refs, &at_dst, bytes);
            assert_ckks_error(
                &label,
                &err,
                CKKSCompositionError::MissingRelinearizationKey {
                    op: "ckks_dot_product_ct",
                    k: key_k.as_u32(),
                },
            );
        }

        if case > 0 {
            continue;
        }
        let err = assert_rejected(module, &mut lazy, &[], &[], &tsk, bytes);
        assert!(err.to_string().contains("at least one pair"), "{label}: {err}");
        let err = assert_rejected(module, &mut lazy, &a_refs, &b_refs[..N - 1], &tsk, bytes);
        assert!(err.to_string().contains("length mismatch"), "{label}: {err}");
        // Inputs whose budget is below their scale.
        let thin: Vec<_> = a_cts
            .iter()
            .map(|ct| {
                let mut ct = ct.clone();
                ct.set_log_budget(ct.log_delta() - 1);
                ct
            })
            .collect();
        let thin_refs: Vec<&_> = thin.iter().collect();
        let err = assert_rejected(module, &mut lazy, &thin_refs, &b_refs, &tsk, bytes);
        assert!(
            matches!(
                err.composition(),
                Some(CKKSCompositionError::MultiplicationPrecisionUnderflow { .. })
            ),
            "{label}: {err}"
        );
        // A destination narrower than the product's scale.
        let mut narrow = lazy.clone();
        narrow.set_log_budget(0);
        narrow.set_log_delta(lazy.log_delta() - 1);
        let err = assert_rejected(module, &mut narrow, &a_refs, &b_refs, &tsk, bytes);
        assert!(
            matches!(
                err.composition(),
                Some(CKKSCompositionError::InsufficientHomomorphicCapacity { .. })
            ),
            "{label}: {err}"
        );
    }
}

/// A sum three times longer than the `2^(63 - base2k)` terms an `i64` digit holds, so the lazy path
/// has to normalize on the way. Every pair is the same two ciphertexts, so the digits add up
/// coherently, and so does the noise. The result is the core composition that normalizes after every
/// term, bit for bit, and decrypts to the sum.
///
/// The bound is in reach only at a high radix: at `N = 32`, NTT4x30's `MaxBase2k` is 57, a bound of
/// 64 terms, while FFT64's stops near 23. Suites below radix 52 return.
pub fn test_dot_product_ct_long_sum<BE, F, E>(
    params: CKKSTestParams,
    _module: &Module<BE>,
    _host_module: &Module<HostBytesBackend>,
) where
    BE: TestContextBackend<Ring = Standard>,
    for<'a> <BE as poulpy_hal::layouts::Backend>::BufRef<'a>: poulpy_hal::layouts::HostDataRef,
    for<'a> <BE as poulpy_hal::layouts::Backend>::BufMut<'a>: poulpy_hal::layouts::HostDataMut,
    Module<BE>: TestContextModule<BE> + GLWETensoring<BE> + GLWEAdd<BE>,
    F: TestScalar,
    E: NegacyclicFFT<F> + NegacyclicFFTNew<F>,
{
    if params.base2k < 52 {
        return;
    }
    let params = CKKSTestParams {
        n: 32,
        base2k: 57,
        hw: 16,
        ..params
    };
    let module = &Module::<BE>::new(params.n as u64);
    let host_module = &Module::<HostBytesBackend>::new(params.n as u64);
    let m = params.n / 2;
    let encoder = ReferenceEncoder::<E>::new(m).unwrap();
    let (re1, im1) = test_vector_1::<F>(m);
    let (re2, im2) = test_vector_2::<F>(m);
    let (sk_raw, sk) = gen_sk_with_raw(&params, module, host_module, [0u8; 32]);
    let mut scratch = alloc_scratch(&params, module);
    let tsk = gen_tsk(&params, module, &sk_raw, &mut scratch.borrow());
    let mut encrypt = |re: &[F], im: &[F]| {
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
    };
    let (a, b) = (encrypt(&re1, &im1), encrypt(&re2, &im2));

    let terms = 3 << (63 - params.base2k);
    let (a_refs, b_refs) = (vec![&a; terms], vec![&b; terms]);
    let mut lazy = alloc_ct(&params, module, params.k);
    let bytes = module.ckks_dot_product_ct_tmp_bytes(terms, &lazy, &a, &b, &tsk);
    with_scratch::<BE, _>(bytes, |scratch| {
        module.ckks_dot_product_ct(&mut lazy, &a_refs, &b_refs, &tsk, scratch)
    })
    .unwrap();
    let mut reference = alloc_ct(&params, module, params.k);
    lazy_reference(
        module,
        &mut reference,
        &a_refs,
        &b_refs,
        &tsk,
        &mut ScratchOwned::<BE>::alloc(bytes).borrow(),
    );
    assert_eq!(
        snapshot::<BE, _>(&lazy).digits,
        snapshot::<BE, _>(&reference).digits,
        "dot_product_ct_long_sum: not the core composition"
    );

    let (re, im) = want_mul(&re1, &im1, &re2, &im2);
    let copies = F::from_usize(terms).unwrap();
    // `terms` equal terms repeat their noise as often: `log2(terms)` bits of precision go.
    let log_delta = lazy.log_delta() - terms.next_power_of_two().ilog2() as usize;
    assert_decrypt_precision_at_log_delta(
        "dot_product_ct_long_sum",
        &params,
        module,
        &encoder,
        &lazy,
        &sk,
        &scaled(&re, copies),
        &scaled(&im, copies),
        log_delta,
        &mut scratch.borrow(),
    );
}

/// Pairs without one layout and scale per side keep the eager path: a narrower `a`, a narrower `b`,
/// and an `a` at another scale give the eager composition's bits and metadata.
pub fn test_dot_product_ct_non_uniform<BE, F, E>(
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
    let (sk_raw, sk) = gen_sk_with_raw(&params, module, host_module, [0u8; 32]);
    let mut scratch = alloc_scratch(&params, module);
    let tsk = gen_tsk(&params, module, &sk_raw, &mut scratch.borrow());

    let vecs = three_vectors(&re1, &im1, &re2, &im2);
    let mut encrypt = |(re, im): &(Vec<F>, Vec<F>), k: usize, prec| {
        ckks_encrypt_with_prec(
            &params,
            module,
            host_module,
            &encoder,
            &sk,
            k,
            re,
            im,
            prec,
            &mut scratch.borrow(),
        )
    };
    let full: Vec<_> = vecs.iter().map(|v| encrypt(v, params.k, params.prec())).collect();
    let narrow = encrypt(&vecs[1], params.k - params.base2k + 1, params.prec());
    let low_log_delta = params.prec().log_delta() - DELTA_LOG_DELTA;
    let low_scale = encrypt(&vecs[1], params.k, precision_at(&params, low_log_delta));
    assert_eq!(low_scale.glwe_layout(), full[1].glwe_layout());

    let uniform: Vec<&_> = full.iter().collect();
    let with = |odd| vec![&full[0], odd, &full[2]];
    let cases = [
        ("narrower a", with(&narrow), uniform.clone()),
        ("narrower b", uniform.clone(), with(&narrow)),
        ("a at another scale", with(&low_scale), uniform.clone()),
    ];
    for (label, a, b) in cases {
        let mut have = alloc_ct(&params, module, params.k);
        let bytes = module.ckks_dot_product_ct_tmp_bytes(N, &have, &full[0], &full[0], &tsk);
        with_scratch::<BE, _>(bytes, |scratch| module.ckks_dot_product_ct(&mut have, &a, &b, &tsk, scratch)).unwrap();
        let mut want = alloc_ct(&params, module, params.k);
        eager_reference(
            module,
            &mut want,
            &a,
            &b,
            &tsk,
            &mut ScratchOwned::<BE>::alloc(bytes).borrow(),
        );
        assert_eq!(
            snapshot::<BE, _>(&have),
            snapshot::<BE, _>(&want),
            "dot_product_ct_non_uniform {label}: not the eager path"
        );
    }
}

pub fn test_dot_product_pt_vec_aligned<BE, F, E>(
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
    let sk = super::helpers::gen_sk(&params, module, host_module, [0u8; 32]);
    let mut scratch = alloc_scratch(&params, module);

    let a_vecs = three_vectors(&re1, &im1, &re2, &im2);
    let b_vecs = three_vectors(&re1, &im1, &re2, &im2);
    let b_pt_vecs: Vec<_> = b_vecs
        .iter()
        .map(|(re, im)| {
            (
                quantize(re, params.prec().log_delta()),
                quantize(im, params.prec().log_delta()),
            )
        })
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
        .map(|(re, im)| encode_and_upload_pt(host_module, module, &encoder, params.base2k.into(), params.prec(), re, im))
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
    let sk = super::helpers::gen_sk(&params, module, host_module, [0u8; 32]);
    let mut scratch = alloc_scratch(&params, module);

    let a_vecs = three_vectors(&re1, &im1, &re2, &im2);
    let const_coeffs: [f64; 3] = [0.25, 0.125, -0.3125];
    let quantized: Vec<F> = const_coeffs
        .iter()
        .map(|r| quantized_const::<F>(*r, 0.0, PT_PREC.log_delta()).0)
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
    let consts: Vec<CKKSPlaintext<_, _, _>> = const_coeffs
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
