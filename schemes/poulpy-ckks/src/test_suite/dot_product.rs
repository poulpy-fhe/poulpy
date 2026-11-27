//! Tests for `CKKSDotProductOps` — inner-product variants.
//!
//! All tests use `n = 3` pairs, except the long sum's 32, the prepared products' 1, 2 and 16, the single pairs of
//! [`test_dot_product_pt_aligned`] and the empty and mismatched slices of [`test_dot_product_pt_rejects_invalid_operands`].
//!
//! # Test inventory
//!
//! | Function | Path exercised |
//! |----------|----------------|
//! | [`test_dot_product_aligned`] | ct · ct, aligned |
//! | [`test_dot_product_unaligned`] | ct · ct, one a-side input rescaled |
//! | [`test_dot_product_unaligned_b`] | ct · ct, one b-side input rescaled |
//! | [`test_dot_product_delta_log_delta`] | ct · ct fallback with non-uniform `log_delta` |
//! | [`test_dot_product_smaller_output`] | ct · ct, output narrower than inputs |
//! | [`test_dot_product_lazy`] | ct · ct, one relinearization: zero padding, metadata, precision, rejections |
//! | [`test_dot_product_long_sum`] | ct · ct, one relinearization of 32 distinct pairs |
//! | [`test_dot_product_non_uniform`] | ct · ct, pairs without a shared layout or scale |
//! | [`test_dot_product_prepared`] | prepared ct · prepared ct: the bits of ct · ct, under every pair of hints; rejections |
//! | [`test_dot_product_pt_aligned`] | ct · plaintext, at the module's degree and half of it: the sum one byte short and on its query, one pair as its product by a plaintext and by constants, a constant beside plaintexts at every position, also wider than them |
//! | [`test_dot_product_pt_unaligned`] | ct · plaintext, pairs a limb of budget apart and one at a coarser scale, into a narrower destination off the limb grid |
//! | [`test_dot_product_pt_rejects_invalid_operands`] | ct · plaintext: no pair, two lengths, and another rank, radix, non-dividing plaintext degree, budget short of the plaintext's scale, slot layout and replicated scale at every position |
//! | [`test_dot_product_prepared_pt`] | prepared ct · prepared ZNX plaintexts through Galois elements: one sum, the bits of ct · pt; rejections |
//! | [`test_dot_product_prepared_pt_add`] | the prepared sum added into `res`, rounded once, by plaintexts of a limb and more and of less; rejections |
//! | [`test_dot_product_pt_const_aligned`] | ct · ZNX constant |

use poulpy_core::{
    CoreError, GLWEMulPreparedPlainSum, GLWESub,
    layouts::{
        GLWEInfos, GLWELayout, GLWEPlaintextInfos, GetTensorKey, LWEInfos, Rank, SetGLWEPlaintextInfos, TorusPrecision,
        prepared::{GLWESecretPrepared, GLWETensorKeyPreparedBackendRef},
    },
};
use poulpy_hal::{
    api::{ReimNew, ScratchOwnedBorrow, VecZnxDftAlloc},
    layouts::{
        Backend, HostBytesBackend, HostDataMut, HostDataRef, Module, PrepareHint, Ring, ScratchArena, ScratchOwned, Standard,
        VecZnxDftToBackendMut,
    },
};

use crate::{
    CKKSCompositionError, CKKSError, CKKSInfos, CKKSLayout, CKKSModuleInfos, GLWEPlaintextMeta, Scale, SetCKKSInfos, SlotsKind,
    api::{
        CKKSAddOps, CKKSDotProductOps, CKKSLeftOperandPreparedFactory, CKKSMulAddOps, CKKSMulOps, CKKSRightOperandPreparedFactory,
    },
    error::DIVISOR,
    layouts::{CKKSCiphertextAlloc, CKKSCiphertextOwned, CKKSLeftOperandPrepared, CKKSPlaintextOwned, CKKSRightOperandPrepared},
};

use super::helpers::{
    PT_PREC, TestContextBackend, TestContextModule, TestContextSharedModule, TestScalar, TestVector, alloc_ct, alloc_scratch,
    assert_ckks_error, assert_ckks_internal, assert_decrypt_precision, assert_decrypt_precision_at_log_delta,
    assert_invalid_operand, ckks_encrypt, ckks_encrypt_with_prec, ckks_pt_cst, dirty_ct, encode_and_upload_pt, gen_sk_with_raw,
    gen_tsk, precision_at, quantize, quantized_const, quantized_vector, test_vector_1, test_vector_2, want_mul, want_rotate,
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

pub fn test_dot_product_aligned<BE, F, E>(params: CKKSTestParams, module: &Module<BE>, host_module: &Module<HostBytesBackend>)
where
    BE: TestContextBackend,
    for<'a> <BE as poulpy_hal::layouts::Backend>::BufRef<'a>: poulpy_hal::layouts::HostDataRef,
    for<'a> <BE as poulpy_hal::layouts::Backend>::BufMut<'a>: poulpy_hal::layouts::HostDataMut,
    Module<BE>: TestContextModule<BE>,
    F: TestScalar,
    E: ReimNew<F>,
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
    let bytes = module.ckks_dot_product_tmp_bytes(N, &ct_res, a_refs[0], b_refs[0], &tsk);
    with_scratch::<BE, _>(bytes, |scratch| {
        module.ckks_dot_product(&mut ct_res, &a_refs, &b_refs, &tsk, scratch)
    })
    .unwrap();
    assert_decrypt_precision(
        "dot_product_aligned",
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

pub fn test_dot_product_unaligned<BE, F, E>(params: CKKSTestParams, module: &Module<BE>, host_module: &Module<HostBytesBackend>)
where
    BE: TestContextBackend,
    for<'a> <BE as poulpy_hal::layouts::Backend>::BufRef<'a>: poulpy_hal::layouts::HostDataRef,
    for<'a> <BE as poulpy_hal::layouts::Backend>::BufMut<'a>: poulpy_hal::layouts::HostDataMut,
    Module<BE>: TestContextModule<BE>,
    F: TestScalar,
    E: ReimNew<F>,
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
    let bytes = module.ckks_dot_product_tmp_bytes(N, &ct_res, a_refs[0], b_refs[0], &tsk);
    with_scratch::<BE, _>(bytes, |scratch| {
        module.ckks_dot_product(&mut ct_res, &a_refs, &b_refs, &tsk, scratch)
    })
    .unwrap();
    assert_decrypt_precision(
        "dot_product_unaligned",
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

pub fn test_dot_product_unaligned_b<BE, F, E>(params: CKKSTestParams, module: &Module<BE>, host_module: &Module<HostBytesBackend>)
where
    BE: TestContextBackend,
    for<'a> <BE as poulpy_hal::layouts::Backend>::BufRef<'a>: poulpy_hal::layouts::HostDataRef,
    for<'a> <BE as poulpy_hal::layouts::Backend>::BufMut<'a>: poulpy_hal::layouts::HostDataMut,
    Module<BE>: TestContextModule<BE>,
    F: TestScalar,
    E: ReimNew<F>,
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
    let bytes = module.ckks_dot_product_tmp_bytes(N, &ct_res, a_refs[0], b_refs[0], &tsk);
    with_scratch::<BE, _>(bytes, |scratch| {
        module.ckks_dot_product(&mut ct_res, &a_refs, &b_refs, &tsk, scratch)
    })
    .unwrap();
    assert_decrypt_precision(
        "dot_product_unaligned_b",
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

pub fn test_dot_product_delta_log_delta<BE, F, E>(
    params: CKKSTestParams,
    module: &Module<BE>,
    host_module: &Module<HostBytesBackend>,
) where
    BE: TestContextBackend,
    for<'a> <BE as poulpy_hal::layouts::Backend>::BufRef<'a>: poulpy_hal::layouts::HostDataRef,
    for<'a> <BE as poulpy_hal::layouts::Backend>::BufMut<'a>: poulpy_hal::layouts::HostDataMut,
    Module<BE>: TestContextModule<BE>,
    F: TestScalar,
    E: ReimNew<F>,
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
    let bytes = module.ckks_dot_product_tmp_bytes(N, &ct_res, a_refs[0], b_refs[1], &tsk);
    with_scratch::<BE, _>(bytes, |scratch| {
        module.ckks_dot_product(&mut ct_res, &a_refs, &b_refs, &tsk, scratch)
    })
    .unwrap();
    assert_decrypt_precision_at_log_delta(
        "dot_product_delta_log_delta",
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

pub fn test_dot_product_smaller_output<BE, F, E>(
    params: CKKSTestParams,
    module: &Module<BE>,
    host_module: &Module<HostBytesBackend>,
) where
    BE: TestContextBackend,
    for<'a> <BE as poulpy_hal::layouts::Backend>::BufRef<'a>: poulpy_hal::layouts::HostDataRef,
    for<'a> <BE as poulpy_hal::layouts::Backend>::BufMut<'a>: poulpy_hal::layouts::HostDataMut,
    Module<BE>: TestContextModule<BE>,
    F: TestScalar,
    E: ReimNew<F>,
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
    let bytes = module.ckks_dot_product_tmp_bytes(N, &ct_res, a_refs[0], b_refs[0], &tsk);
    with_scratch::<BE, _>(bytes, |scratch| {
        module.ckks_dot_product(&mut ct_res, &a_refs, &b_refs, &tsk, scratch)
    })
    .unwrap();
    assert_decrypt_precision(
        "dot_product smaller_output",
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

/// The eager path: `ckks_mul_into` of the first pair, then `ckks_mul_add_into` of each other.
fn eager<BE: TestContextBackend, H: GetTensorKey<BE, Rg>, Rg: Ring>(
    module: &Module<BE>,
    dst: &mut CKKSCiphertextOwned<BE, Rg>,
    a: &[&CKKSCiphertextOwned<BE, Rg>],
    b: &[&CKKSCiphertextOwned<BE, Rg>],
    tsk: &H,
    scratch: &mut ScratchArena<'_, BE>,
) where
    Module<BE>: TestContextSharedModule<BE, Rg>,
{
    module.ckks_mul_into(dst, a[0], b[0], tsk, scratch).unwrap();
    for (ai, bi) in a.iter().zip(b).skip(1) {
        module.ckks_mul_add_into(dst, *ai, *bi, tsk, scratch).unwrap();
    }
}

/// A tensor-key store that holds its key at one width only.
pub(crate) struct TensorKeyAt<'a, K>(pub(crate) &'a K, pub(crate) TorusPrecision);

impl<BE: Backend, K: GetTensorKey<BE, Rg>, Rg: Ring> GetTensorKey<BE, Rg> for TensorKeyAt<'_, K> {
    fn lookup_tensor_key(&self, k: TorusPrecision) -> poulpy_core::Result<GLWETensorKeyPreparedBackendRef<'_, BE, Rg>> {
        if k != self.1 {
            return Err(CoreError::GGLWEKeyUse {
                op: "get_tensor_key",
                detail: format!("no tensor key at k={}", k.as_u32()),
            });
        }
        self.0.get_tensor_key(k)
    }
}

/// Runs a dot product expected to fail before its scratch check on an empty arena, checks that `dst` is
/// untouched, and returns the error.
fn assert_rejected<BE: TestContextBackend, H: GetTensorKey<BE, Rg>, Rg: Ring>(
    module: &Module<BE>,
    dst: &mut CKKSCiphertextOwned<BE, Rg>,
    a: &[&CKKSCiphertextOwned<BE, Rg>],
    b: &[&CKKSCiphertextOwned<BE, Rg>],
    tsk: &H,
) -> CKKSError
where
    Module<BE>: TestContextSharedModule<BE, Rg>,
{
    let before = snapshot::<BE, _, Rg>(&*dst);
    let err = with_scratch::<BE, _>(0, |scratch| module.ckks_dot_product(dst, a, b, tsk, scratch)).unwrap_err();
    assert_eq!(
        snapshot::<BE, _, Rg>(&*dst),
        before,
        "a rejected dot product wrote dst: {err}"
    );
    err
}

/// Pairs that share one layout and scale per side take one relinearization, on the exact poisoned
/// budget, at four widths (all equal, a narrower `dst`, `b` narrower than `a`, `dst` wider than both) and
/// with `b` at a lower scale than `a`. The first pair beside two pairs of zeros is `ckks_mul_into` of that
/// pair, bit for bit, metadata and canonical flag included. The metadata and noise tag are the eager
/// path's: the smaller scale, the smallest label and the joined slot kind of all six operands. The result
/// is canonical and decrypts to the precision of a product. The tensor key is looked up at `max(a.k, b.k)`
/// only, whatever `dst`'s width. Wrong lengths, a missing tensor key, inputs without the budget and a too
/// narrow `dst` are rejected with `dst` untouched.
pub fn test_dot_product_lazy<BE, F, E>(params: CKKSTestParams, module: &Module<BE>, host_module: &Module<HostBytesBackend>)
where
    BE: TestContextBackend,
    for<'a> <BE as poulpy_hal::layouts::Backend>::BufRef<'a>: poulpy_hal::layouts::HostDataRef,
    for<'a> <BE as poulpy_hal::layouts::Backend>::BufMut<'a>: poulpy_hal::layouts::HostDataMut,
    Module<BE>: TestContextModule<BE>,
    F: TestScalar,
    E: ReimNew<F>,
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
    // (dst.k, a.k, b.k, b's scale). The lower scale keeps `b`'s budget, as in `test_dot_product_delta_log_delta`.
    let cases = [
        (k, k, k, log_delta),
        (k - base2k - 1, k, k, log_delta),
        (k, k, k - base2k + 1, log_delta),
        (k, k - base2k + 1, k - 2 * base2k + 1, log_delta),
        (k, k, k - DELTA_LOG_DELTA, log_delta - DELTA_LOG_DELTA),
    ];
    for (case, (dst_k, a_k, b_k, b_log_delta)) in cases.into_iter().enumerate() {
        let label = format!("dot_product_lazy dst.k={dst_k} a.k={a_k} b.k={b_k} b.log_delta={b_log_delta}");
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
        let bytes = module.ckks_dot_product_tmp_bytes(N, &lazy, a_refs[0], b_refs[0], &tsk);
        with_scratch::<BE, _>(bytes, |scratch| {
            module.ckks_dot_product(&mut lazy, &a_refs, &b_refs, &tsk, scratch)
        })
        .unwrap();
        let have = snapshot::<BE, _, Standard>(&lazy);

        // Pairs of zeros add nothing to the products, whichever circuit sums them.
        let zero = |ct: &CKKSCiphertextOwned<BE, Standard>| {
            let mut zero = ct.clone();
            module.glwe_sub(&mut zero, ct, ct);
            zero
        };
        let (za, zb) = (zero(a_refs[0]), zero(b_refs[0]));
        let mut padded = alloc_ct(&params, module, dst_k);
        with_scratch::<BE, _>(bytes, |scratch| {
            module.ckks_dot_product(&mut padded, &[a_refs[0], &za, &za], &[b_refs[0], &zb, &zb], &tsk, scratch)
        })
        .unwrap();
        let mut product = alloc_ct(&params, module, dst_k);
        with_scratch::<BE, _>(module.ckks_mul_tmp_bytes(&product, a_refs[0], b_refs[0], &tsk), |s| {
            module.ckks_mul_into(&mut product, a_refs[0], b_refs[0], &tsk, s)
        })
        .unwrap();
        assert_eq!(
            snapshot::<BE, _, Standard>(&padded),
            snapshot::<BE, _, Standard>(&product),
            "{label}: a pair beside pairs of zeros is not its product"
        );

        let mut want = alloc_ct(&params, module, dst_k);
        eager(module, &mut want, &a_refs, &b_refs, &tsk, &mut scratch.borrow());
        let want = snapshot::<BE, _, Standard>(&want);
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
        let key_k = TorusPrecision(a_k.max(b_k) as u32);
        let mut keyed = alloc_ct(&params, module, dst_k);
        with_scratch::<BE, _>(bytes, |scratch| {
            module.ckks_dot_product(&mut keyed, &a_refs, &b_refs, &TensorKeyAt(&tsk, key_k), scratch)
        })
        .unwrap();
        assert_eq!(
            snapshot::<BE, _, Standard>(&keyed),
            have,
            "{label}: a key held at {key_k:?} only"
        );
        if dst_k > a_k.max(b_k) {
            let at_dst = TensorKeyAt(&tsk, TorusPrecision(dst_k as u32));
            let err = assert_rejected(module, &mut lazy, &a_refs, &b_refs, &at_dst);
            assert!(
                matches!(
                    err.composition(),
                    Some(CKKSCompositionError::MissingRelinearizationKey { op: "ckks_dot_product", k, .. })
                        if *k == key_k.as_u32()
                ),
                "{label}: unexpected error: {err}"
            );
        }

        if case > 0 {
            continue;
        }
        let err = assert_rejected(module, &mut lazy, &[], &[], &tsk);
        assert!(err.to_string().contains("at least one pair"), "{label}: {err}");
        let err = assert_rejected(module, &mut lazy, &a_refs, &b_refs[..N - 1], &tsk);
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
        let err = assert_rejected(module, &mut lazy, &thin_refs, &b_refs, &tsk);
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
        let err = assert_rejected(module, &mut narrow, &a_refs, &b_refs, &tsk);
        assert!(
            matches!(
                err.composition(),
                Some(CKKSCompositionError::InsufficientHomomorphicCapacity { .. })
            ),
            "{label}: {err}"
        );
    }
}

/// A long sum, 32 pairs, relinearized once. The pairs are fresh encryptions of the same two messages,
/// so their digits are independent, as `Module::max_base2k`'s model assumes: copies of one pair would
/// add up coherently, outside it, past what an FFT suite's radix holds once the products accumulate in
/// the transform domain. The result decrypts to the sum. It runs on its own ring, `N = 32`, at the
/// suite's radix.
pub fn test_dot_product_long_sum<BE, F, E>(params: CKKSTestParams, _module: &Module<BE>, _host_module: &Module<HostBytesBackend>)
where
    BE: TestContextBackend,
    for<'a> <BE as poulpy_hal::layouts::Backend>::BufRef<'a>: poulpy_hal::layouts::HostDataRef,
    for<'a> <BE as poulpy_hal::layouts::Backend>::BufMut<'a>: poulpy_hal::layouts::HostDataMut,
    Module<BE>: TestContextModule<BE>,
    F: TestScalar,
    E: ReimNew<F>,
{
    let params = CKKSTestParams { n: 32, hw: 16, ..params };
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
    let terms = 32;
    let (a, b): (Vec<_>, Vec<_>) = (0..terms).map(|_| (encrypt(&re1, &im1), encrypt(&re2, &im2))).unzip();
    let (a_refs, b_refs): (Vec<&_>, Vec<&_>) = (a.iter().collect(), b.iter().collect());
    let mut lazy = alloc_ct(&params, module, params.k);
    let bytes = module.ckks_dot_product_tmp_bytes(terms, &lazy, &a[0], &b[0], &tsk);
    with_scratch::<BE, _>(bytes, |scratch| {
        module.ckks_dot_product(&mut lazy, &a_refs, &b_refs, &tsk, scratch)
    })
    .unwrap();
    let (re, im) = want_mul(&re1, &im1, &re2, &im2);
    let copies = F::from_usize(terms).unwrap();
    // `terms` terms add their noise: at most `log2(terms)` bits of precision go.
    let log_delta = lazy.log_delta() - terms.next_power_of_two().ilog2() as usize;
    assert_decrypt_precision_at_log_delta(
        "dot_product_long_sum",
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
pub fn test_dot_product_non_uniform<BE, F, E>(params: CKKSTestParams, module: &Module<BE>, host_module: &Module<HostBytesBackend>)
where
    BE: TestContextBackend,
    for<'a> <BE as poulpy_hal::layouts::Backend>::BufRef<'a>: poulpy_hal::layouts::HostDataRef,
    for<'a> <BE as poulpy_hal::layouts::Backend>::BufMut<'a>: poulpy_hal::layouts::HostDataMut,
    Module<BE>: TestContextModule<BE>,
    F: TestScalar,
    E: ReimNew<F>,
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
        let bytes = module.ckks_dot_product_tmp_bytes(N, &have, &full[0], &full[0], &tsk);
        with_scratch::<BE, _>(bytes, |scratch| module.ckks_dot_product(&mut have, &a, &b, &tsk, scratch)).unwrap();
        let mut want = alloc_ct(&params, module, params.k);
        eager(module, &mut want, &a, &b, &tsk, &mut scratch.borrow());
        assert_eq!(
            snapshot::<BE, _, Standard>(&have),
            snapshot::<BE, _, Standard>(&want),
            "dot_product_non_uniform {label}: not the eager path"
        );
    }
}

/// Runs a prepared dot product expected to fail before its scratch check on an empty arena, checks that `dst` is
/// untouched, and returns the error.
fn assert_prepared_rejected<BE: TestContextBackend, H: GetTensorKey<BE, Rg>, Rg: Ring>(
    module: &Module<BE>,
    dst: &mut CKKSCiphertextOwned<BE, Rg>,
    a: &[&CKKSLeftOperandPrepared<BE, Rg>],
    b: &[&CKKSRightOperandPrepared<BE, Rg>],
    tsk: &H,
) -> CKKSError
where
    Module<BE>: TestContextSharedModule<BE, Rg>,
{
    let before = snapshot::<BE, _, Rg>(&*dst);
    let err = with_scratch::<BE, _>(0, |scratch| module.ckks_dot_product_prepared(dst, a, b, tsk, scratch)).unwrap_err();
    assert_eq!(
        snapshot::<BE, _, Rg>(&*dst),
        before,
        "a rejected prepared dot product wrote dst: {err}"
    );
    err
}

/// Pairs prepared by `ckks_left_operand_prepare_alloc` and `ckks_right_operand_prepare_alloc` give `ckks_dot_product` of the ciphertexts they
/// were prepared from, bit for bit, metadata, noise tag and canonical flag included, for one pair and for 16 fresh
/// pairs, under every pair of hints: the one-pass sum on rank 1, the two-pass circuit on rank 2. Each preparation runs on the exact
/// poisoned budget of its `ckks_<side>_operand_prepare_tmp_bytes`, and each prepared operand reports its source's layout, metadata and
/// noise estimate, `None` from a source without metadata. Each product runs on the exact poisoned budget of its own query,
/// `ckks_dot_product_tmp_bytes` of the ciphertexts and `ckks_dot_product_prepared_tmp_bytes` of the prepared operands. Wrong lengths, operands of another
/// degree, radix or rank, mixed layouts, radix 63, a missing tensor key and inputs without the budget are rejected, in
/// that order, with `dst` untouched.
pub fn test_dot_product_prepared<BE, F, E>(params: CKKSTestParams, module: &Module<BE>, host_module: &Module<HostBytesBackend>)
where
    BE: TestContextBackend,
    for<'a> <BE as poulpy_hal::layouts::Backend>::BufRef<'a>: poulpy_hal::layouts::HostDataRef,
    for<'a> <BE as poulpy_hal::layouts::Backend>::BufMut<'a>: poulpy_hal::layouts::HostDataMut,
    Module<BE>: TestContextModule<BE>,
    F: TestScalar,
    E: ReimNew<F>,
{
    const PAIRS: usize = 16;
    const HINTS: [PrepareHint; 2] = [PrepareHint::OneShot, PrepareHint::Reuse];
    const OP: &str = "ckks_dot_product_prepared";
    let m = params.n / 2;
    let encoder = ReferenceEncoder::<E>::new(m).unwrap();
    let (re1, im1) = test_vector_1::<F>(m);
    let (re2, im2) = test_vector_2::<F>(m);
    let (sk_raw, sk) = gen_sk_with_raw(&params, module, host_module, [0u8; 32]);
    let prepare_left = |ct: &CKKSCiphertextOwned<BE, Standard>, hint| {
        with_scratch::<BE, _>(module.ckks_left_operand_prepare_tmp_bytes(ct), |s| {
            module.ckks_left_operand_prepare_alloc(ct, hint, s)
        })
        .unwrap()
    };
    let prepare_right = |ct: &CKKSCiphertextOwned<BE, Standard>, hint| {
        with_scratch::<BE, _>(module.ckks_right_operand_prepare_tmp_bytes(ct), |s| {
            module.ckks_right_operand_prepare_alloc(ct, hint, s)
        })
        .unwrap()
    };

    let p = &params;
    let mut scratch = alloc_scratch(p, module);
    let tsk = &gen_tsk(p, module, &sk_raw, &mut scratch.borrow());
    let mut encrypt =
        |re: &[F], im: &[F]| ckks_encrypt(p, module, host_module, &encoder, &sk, p.k, re, im, &mut scratch.borrow());
    let (a, b): (Vec<_>, Vec<_>) = (0..PAIRS).map(|_| (encrypt(&re1, &im1), encrypt(&re2, &im2))).unzip();
    // Each operand prepared under each hint.
    let left: Vec<Vec<_>> = HINTS
        .iter()
        .map(|&hint| a.iter().map(|ct| prepare_left(ct, hint)).collect())
        .collect();
    let right: Vec<Vec<_>> = HINTS
        .iter()
        .map(|&hint| b.iter().map(|ct| prepare_right(ct, hint)).collect())
        .collect();
    assert!(a[0].noise().is_some(), "a fresh ciphertext carries a noise estimate");
    for (prepared, ct) in left.iter().flatten().zip(a.iter().cycle()) {
        assert_eq!(
            (prepared.glwe_layout(), prepared.plaintext_meta(), prepared.noise()),
            (ct.glwe_layout(), ct.plaintext_meta(), None)
        );
    }
    for (prepared, ct) in right.iter().flatten().zip(b.iter().cycle()) {
        assert_eq!(
            (prepared.glwe_layout(), prepared.plaintext_meta(), prepared.noise()),
            (ct.glwe_layout(), ct.plaintext_meta(), None)
        );
    }

    for n in [1, PAIRS] {
        let (a, b): (Vec<&_>, Vec<&_>) = (a[..n].iter().collect(), b[..n].iter().collect());
        let mut want = alloc_ct(p, module, p.k);
        let bytes = module.ckks_dot_product_tmp_bytes(n, &want, a[0], b[0], tsk);
        with_scratch::<BE, _>(bytes, |scratch| module.ckks_dot_product(&mut want, &a, &b, tsk, scratch)).unwrap();
        let want = snapshot::<BE, _, Standard>(&want);
        for (ha, hb) in [(0, 0), (1, 1), (0, 1), (1, 0)] {
            let label = format!("dot_product_prepared pairs={n} hints={:?}", (HINTS[ha], HINTS[hb]));
            let lhs: Vec<&_> = left[ha][..n].iter().collect();
            let rhs: Vec<&_> = right[hb][..n].iter().collect();
            let mut have = alloc_ct(p, module, p.k);
            let prepared_bytes = module.ckks_dot_product_prepared_tmp_bytes(n, &have, lhs[0], rhs[0], tsk);
            with_scratch::<BE, _>(prepared_bytes, |scratch| {
                module.ckks_dot_product_prepared(&mut have, &lhs, &rhs, tsk, scratch)
            })
            .unwrap();
            assert_eq!(snapshot::<BE, _, Standard>(&have), want, "{label}: not ckks_dot_product");
        }
    }

    // A source without plaintext metadata gives operands without it.
    let mut bare = a[0].clone();
    bare.set_plaintext_meta(None);
    assert_eq!(
        prepare_left(&bare, PrepareHint::Reuse).plaintext_meta(),
        None,
        "left of a bare source"
    );
    assert_eq!(
        prepare_right(&bare, PrepareHint::Reuse).plaintext_meta(),
        None,
        "right of a bare source"
    );

    // Rejections.
    let lhs: Vec<&_> = left[1].iter().collect();
    let rhs: Vec<&_> = right[1].iter().collect();
    let mut dst = alloc_ct(p, module, p.k);
    let no_key = TensorKeyAt(tsk, TorusPrecision(1));
    /// `side` with `odd` at `at`.
    fn with<'a, T>(side: &[&'a T], at: usize, odd: &'a T) -> Vec<&'a T> {
        let mut side = side.to_vec();
        side[at] = odd;
        side
    }

    let err = assert_prepared_rejected(module, &mut dst, &[], &[], tsk);
    assert!(err.to_string().contains("at least one pair"), "{err}");
    let err = assert_prepared_rejected(module, &mut dst, &lhs, &rhs[..PAIRS - 1], tsk);
    assert!(err.to_string().contains("length mismatch"), "{err}");

    // Operands of another degree, radix or rank, on either side, before the layouts each side shares.
    let foreign_cts: Vec<_> = [
        GLWELayout {
            n: (p.n / 2).into(),
            ..p.glwe_layout()
        },
        GLWELayout {
            base2k: (p.base2k - 1).into(),
            ..p.glwe_layout()
        },
        GLWELayout {
            rank: Rank((p.rank % 2 + 1) as u32),
            ..p.glwe_layout()
        },
    ]
    .iter()
    .map(|layout| {
        let mut ct = module.ckks_ciphertext_alloc_from_glwe_infos(layout);
        ct.set_meta(p.prec().meta);
        ct
    })
    .collect();
    let foreign_left: Vec<_> = foreign_cts.iter().map(|ct| prepare_left(ct, PrepareHint::Reuse)).collect();
    let foreign_right: Vec<_> = foreign_cts.iter().map(|ct| prepare_right(ct, PrepareHint::Reuse)).collect();
    let mismatch = |odd: GLWELayout| CKKSCompositionError::PreparedOperandLayoutMismatch {
        op: OP,
        dst_n: p.n,
        dst_base2k: p.base2k,
        dst_rank: p.rank,
        prep_n: odd.n.as_usize(),
        prep_base2k: odd.base2k.as_usize(),
        prep_rank: odd.rank.as_usize(),
    };
    for (odd_left, odd_right) in foreign_left.iter().zip(&foreign_right) {
        // The lengths come first.
        let err = assert_prepared_rejected(module, &mut dst, &with(&lhs, 0, odd_left), &rhs[1..], tsk);
        assert!(err.to_string().contains("length mismatch"), "{err}");
        let err = assert_prepared_rejected(module, &mut dst, &with(&lhs, PAIRS - 1, odd_left), &rhs, tsk);
        assert_ckks_error("foreign left operand", &err, mismatch(odd_left.glwe_layout()));
        let err = assert_prepared_rejected(module, &mut dst, &lhs, &with(&rhs, 0, odd_right), tsk);
        assert_ckks_error("foreign right operand", &err, mismatch(odd_right.glwe_layout()));
    }

    // A side that does not share one layout and scale: a narrower operand, then one at a lower scale.
    let mut narrow = a[1].clone();
    narrow.set_log_budget(narrow.log_budget() - 1);
    let narrow = prepare_left(&narrow, PrepareHint::Reuse);
    let mut low_scale = b[1].clone();
    low_scale.set_log_delta(low_scale.log_delta() - 1);
    let low_scale = prepare_right(&low_scale, PrepareHint::Reuse);
    let invalid = |err: &CKKSError| matches!(err.composition(), Some(CKKSCompositionError::InvalidOperand { op: OP, .. }));
    for (lhs, rhs) in [
        (with(&lhs, 1, &narrow), rhs.clone()),
        (lhs.clone(), with(&rhs, 1, &low_scale)),
    ] {
        let err = assert_prepared_rejected(module, &mut dst, &lhs, &rhs, &no_key);
        assert!(invalid(&err), "mixed layouts, before the tensor key: {err}");
    }
    // The operands' layouts come first.
    let err = assert_prepared_rejected(
        module,
        &mut dst,
        &with(&lhs, 1, &narrow),
        &with(&rhs, 0, &foreign_right[0]),
        tsk,
    );
    assert_ckks_error("mixed and foreign", &err, mismatch(foreign_right[0].glwe_layout()));

    // Radix 63, past the radix of two lazy terms, before the tensor key.
    let radix_63 = GLWELayout {
        base2k: 63usize.into(),
        ..p.glwe_layout()
    };
    let mut wide = module.ckks_ciphertext_alloc_from_glwe_infos(&radix_63);
    wide.set_meta(p.prec().meta);
    let (wide_left, wide_right) = (
        prepare_left(&wide, PrepareHint::Reuse),
        prepare_right(&wide, PrepareHint::Reuse),
    );
    let mut wide_dst = module.ckks_ciphertext_alloc_from_glwe_infos(&radix_63);
    let err = assert_prepared_rejected(module, &mut wide_dst, &[&wide_left], &[&wide_right], &no_key);
    assert!(invalid(&err) && err.to_string().contains("base2k 63 > 62"), "{err}");

    // No tensor key at `max(a.k, b.k)`.
    let err = assert_prepared_rejected(module, &mut dst, &lhs, &rhs, &no_key);
    assert!(
        matches!(
            err.composition(),
            Some(CKKSCompositionError::MissingRelinearizationKey { op: OP, k, .. }) if *k == p.k as u32
        ),
        "missing tensor key: unexpected error: {err}"
    );

    // Inputs whose budget is below their scale, after the tensor key.
    let thin: Vec<_> = a
        .iter()
        .map(|ct| {
            let mut ct = ct.clone();
            ct.set_log_budget(ct.log_delta() - 1);
            prepare_left(&ct, PrepareHint::Reuse)
        })
        .collect();
    let thin: Vec<&_> = thin.iter().collect();
    let err = assert_prepared_rejected(module, &mut dst, &thin, &rhs, &no_key);
    assert!(
        matches!(
            err.composition(),
            Some(CKKSCompositionError::MissingRelinearizationKey { .. })
        ),
        "{err}"
    );
    let err = assert_prepared_rejected(module, &mut dst, &thin, &rhs, tsk);
    assert!(
        matches!(
            err.composition(),
            Some(CKKSCompositionError::MultiplicationPrecisionUnderflow { .. })
        ),
        "{err}"
    );
}

/// The pairs of the plaintext dot products: [`three_vectors`] encrypted at `params.k` and encoded as plaintexts at
/// `params.prec()`, the plaintexts' quantized values and `Σ a_i ⊙ b_i`.
struct PtFixture<BE: Backend, F, E> {
    encoder: ReferenceEncoder<E>,
    sk: GLWESecretPrepared<BE::OwnedBuf, BE>,
    scratch: ScratchOwned<BE>,
    a_vecs: [(Vec<F>, Vec<F>); 3],
    b_pts: Vec<(Vec<F>, Vec<F>)>,
    want: (Vec<F>, Vec<F>),
    a_cts: Vec<CKKSCiphertextOwned<BE>>,
    pts: Vec<CKKSPlaintextOwned<BE>>,
}

fn pt_fixture<BE, F, E>(
    params: &CKKSTestParams,
    module: &Module<BE>,
    host_module: &Module<HostBytesBackend>,
) -> PtFixture<BE, F, E>
where
    BE: TestContextBackend,
    Module<BE>: TestContextModule<BE>,
    F: TestScalar,
    E: ReimNew<F>,
{
    let m = params.n / 2;
    let encoder = ReferenceEncoder::<E>::new(m).unwrap();
    let (re1, im1) = test_vector_1::<F>(m);
    let (re2, im2) = test_vector_2::<F>(m);
    let sk = super::helpers::gen_sk(params, module, host_module, [0u8; 32]);
    let mut scratch = alloc_scratch(params, module);

    let a_vecs = three_vectors(&re1, &im1, &re2, &im2);
    let b_vecs = three_vectors(&re1, &im1, &re2, &im2);
    let b_pts: Vec<_> = b_vecs
        .iter()
        .map(|(re, im)| {
            (
                quantize(re, params.prec().log_delta()),
                quantize(im, params.prec().log_delta()),
            )
        })
        .collect();

    let mut want = (vec![F::zero(); m], vec![F::zero(); m]);
    for i in 0..N {
        cmul_acc(&mut want.0, &mut want.1, &a_vecs[i].0, &a_vecs[i].1, &b_pts[i].0, &b_pts[i].1);
    }

    let a_cts = a_vecs
        .iter()
        .map(|(re, im)| {
            ckks_encrypt(
                params,
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
    let pts = b_vecs
        .iter()
        .map(|(re, im)| encode_and_upload_pt(host_module, module, &encoder, params.base2k.into(), params.prec(), re, im))
        .collect();
    PtFixture {
        encoder,
        sk,
        scratch,
        a_vecs,
        b_pts,
        want,
        a_cts,
        pts,
    }
}

/// `ckks_dot_product_pt` of the pairs into a dirty ciphertext of `k` bits, on exactly the query of its widest ciphertext
/// and of the largest degree and `k` of its plaintexts.
fn dot<BE>(
    module: &Module<BE>,
    a: &[&CKKSCiphertextOwned<BE>],
    b: &[&CKKSPlaintextOwned<BE>],
    k: usize,
) -> CKKSCiphertextOwned<BE>
where
    BE: TestContextBackend,
    Module<BE>: TestContextModule<BE>,
{
    let mut res = dirty_ct(module, a[0], k);
    let widest_a = a.iter().max_by_key(|ct| ct.size()).unwrap();
    // The largest degree and `k` may come from two plaintexts.
    let widest_b = CKKSLayout {
        glwe_layout: GLWELayout {
            n: b.iter().map(|pt| pt.n()).max().unwrap(),
            base2k: res.base2k(),
            k: b.iter().map(|pt| pt.max_k()).max().unwrap(),
            rank: Rank(0),
        },
        meta: GLWEPlaintextMeta::default(),
    };
    with_scratch::<BE, _>(module.ckks_dot_product_pt_tmp_bytes(&res, *widest_a, &widest_b), |s| {
        module.ckks_dot_product_pt(&mut res, a, b, s)
    })
    .unwrap();
    res
}

/// Asserts the `k` and metadata `ckks_dot_product_pt` stamps on `res` from its pairs and `dst_k`, the destination's
/// `k()` before the call: the smallest `ckks_mul_pt_into` budget into `dst_k` and scale of the pairs, their joined slot
/// kinds, the smallest sparsity of the pairs, each at most its plaintext's, and their layout.
fn assert_dot_product_pt_meta<A: CKKSInfos, B: CKKSInfos>(label: &str, res: &impl CKKSInfos, dst_k: usize, a: &[&A], b: &[&B]) {
    let pairs = || a.iter().zip(b);
    let log_delta = a.iter().map(|a| a.log_delta()).min().unwrap();
    let log_budget = pairs()
        .map(|(a, b)| {
            let log_budget = a.log_budget() - b.log_delta();
            log_budget - (log_budget + a.log_delta()).saturating_sub(dst_k)
        })
        .min()
        .unwrap();
    let want = GLWEPlaintextMeta {
        scale: Scale::Log(log_delta),
        slots: pairs().fold(a[0].slots(), |slots, (a, b)| slots.join(a.slots()).join(b.slots())),
        log_sparsity: pairs()
            .map(|(a, b)| a.log_sparsity().min((a.n().as_usize() / b.n().as_usize()).ilog2() as usize))
            .min()
            .unwrap(),
        slot_layout: a[0].slot_layout(),
    };
    assert_eq!(
        (res.k().as_usize(), res.meta()),
        (log_budget + log_delta, want),
        "{label}: k and metadata from the pairs"
    );
}

/// `ckks_dot_product_pt` at the module's degree and, as a sub-degree leg, at half of it: the sum one byte short and on
/// its query, one pair as its product by a plaintext and by constants, and a constant beside plaintexts at every
/// position, also one wider than them.
pub fn test_dot_product_pt_aligned<BE, F, E>(params: CKKSTestParams, module: &Module<BE>, host_module: &Module<HostBytesBackend>)
where
    BE: TestContextBackend,
    for<'a> BE::BufRef<'a>: HostDataRef,
    for<'a> BE::BufMut<'a>: HostDataMut,
    Module<BE>: TestContextModule<BE>,
    F: TestScalar,
    E: ReimNew<F>,
{
    let half = CKKSTestParams {
        n: params.n / 2,
        hw: params.hw.min(params.n / 4),
        ..params
    };
    let half_host = Module::<HostBytesBackend>::new(half.n as u64);
    for (params, host_module) in [(params, host_module), (half, &half_host)] {
        dot_product_pt_aligned_at::<BE, F, E>(params, module, host_module);
    }
}

fn dot_product_pt_aligned_at<BE, F, E>(params: CKKSTestParams, module: &Module<BE>, host_module: &Module<HostBytesBackend>)
where
    BE: TestContextBackend,
    for<'a> BE::BufRef<'a>: HostDataRef,
    for<'a> BE::BufMut<'a>: HostDataMut,
    Module<BE>: TestContextModule<BE>,
    F: TestScalar,
    E: ReimNew<F>,
{
    let PtFixture {
        encoder,
        sk,
        mut scratch,
        a_vecs,
        b_pts,
        want: (want_re, want_im),
        a_cts,
        pts,
    } = pt_fixture::<BE, F, E>(&params, module, host_module);
    let m = params.n / 2;
    let a_refs: Vec<&_> = a_cts.iter().collect();
    let pt_refs: Vec<&_> = pts.iter().collect();

    let mut ct_res = dirty_ct(module, &a_cts[0], params.k);
    let bytes = module.ckks_dot_product_pt_tmp_bytes(&ct_res, a_refs[0], pt_refs[0]);
    let before = snapshot::<BE, _, Standard>(&ct_res);
    let err = with_scratch::<BE, _>(bytes - 1, |s| module.ckks_dot_product_pt(&mut ct_res, &a_refs, &pt_refs, s)).unwrap_err();
    assert_ckks_internal("one byte short", &err, "ckks_dot_product_pt needs a scratch arena");
    assert_eq!(snapshot::<BE, _, Standard>(&ct_res), before, "a short arena wrote dst");
    with_scratch::<BE, _>(bytes, |s| module.ckks_dot_product_pt(&mut ct_res, &a_refs, &pt_refs, s)).unwrap();
    assert_decrypt_precision(
        "dot_product_pt_aligned",
        &params,
        module,
        &encoder,
        &ct_res,
        &sk,
        &want_re,
        &want_im,
        &mut scratch.borrow(),
    );

    // One pair is the product of the pair, bit for bit, by a plaintext and by a constant of degree 1 or 2, the latter
    // below every backend's floor; the sum of aligned pairs carries the metadata of each product.
    let single = |a: &CKKSCiphertextOwned<BE>, b: &CKKSPlaintextOwned<BE>| {
        let mut res = alloc_ct(&params, module, params.k);
        with_scratch::<BE, _>(module.ckks_mul_pt_tmp_bytes(&res, a, b), |s| {
            module.ckks_mul_pt_into(&mut res, a, b, s)
        })
        .unwrap();
        res
    };
    let real = ckks_pt_cst::<BE, F, Standard>(host_module, module, params.base2k.into(), PT_PREC, Some(0.375), None);
    let complex = ckks_pt_cst::<BE, F, Standard>(host_module, module, params.base2k.into(), PT_PREC, Some(0.375), Some(-0.25));
    for (label, b) in [
        ("plaintext", &pts[0]),
        ("real constant", &real),
        ("complex constant", &complex),
    ] {
        assert_eq!(
            snapshot::<BE, _, Standard>(&dot(module, &[a_refs[0]], &[b], params.k)),
            snapshot::<BE, _, Standard>(&single(a_refs[0], b)),
            "one pair by a {label} is its product"
        );
    }
    let product = single(a_refs[0], pt_refs[0]);
    assert_eq!(ct_res.k(), product.k(), "dot product width");
    assert_eq!(ct_res.meta(), product.meta(), "dot product metadata of aligned pairs");

    // A constant beside plaintexts, at every position, also one wider than the plaintexts, the largest degree and `k`
    // then of two plaintexts: the sum of the products.
    let constant = ckks_pt_cst::<BE, F, Standard>(host_module, module, params.base2k.into(), PT_PREC, Some(0.5), None);
    let wide_prec = CKKSLayout {
        glwe_layout: GLWELayout {
            k: (params.prec().k().as_usize() + params.base2k).into(),
            ..PT_PREC.glwe_layout
        },
        ..PT_PREC
    };
    let wide_constant = ckks_pt_cst::<BE, F, Standard>(host_module, module, params.base2k.into(), wide_prec, Some(0.5), None);
    assert!(wide_constant.max_k() > pts[0].max_k());
    let (halves, zeros) = (vec![F::from_f64(0.5).unwrap(); m], vec![F::zero(); m]);
    for (position, constant) in (0..N).flat_map(|position| [(position, &constant), (position, &wide_constant)]) {
        let mut b = pt_refs.clone();
        b[position] = constant;
        let (mut want_re, mut want_im) = (vec![F::zero(); m], vec![F::zero(); m]);
        for i in 0..N {
            let (b_re, b_im) = if i == position {
                (&halves, &zeros)
            } else {
                (&b_pts[i].0, &b_pts[i].1)
            };
            cmul_acc(&mut want_re, &mut want_im, &a_vecs[i].0, &a_vecs[i].1, b_re, b_im);
        }
        let label = format!("dot_product_pt constant of k {} at {position}", constant.k());
        let res = dot(module, &a_refs, &b, params.k);
        assert_dot_product_pt_meta(&label, &res, params.k, &a_refs, &b);
        assert_decrypt_precision(
            &label,
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
}

/// `ckks_dot_product_pt` of pairs a limb of budget apart, one at a scale `DELTA_LOG_DELTA` bits coarser, into a
/// narrower destination off the limb grid lands at the smallest budget and scale.
pub fn test_dot_product_pt_unaligned<BE, F, E>(
    params: CKKSTestParams,
    module: &Module<BE>,
    host_module: &Module<HostBytesBackend>,
) where
    BE: TestContextBackend,
    for<'a> BE::BufRef<'a>: HostDataRef,
    for<'a> BE::BufMut<'a>: HostDataMut,
    Module<BE>: TestContextModule<BE>,
    F: TestScalar,
    E: ReimNew<F>,
{
    let PtFixture {
        encoder,
        sk,
        mut scratch,
        a_vecs,
        want: (want_re, want_im),
        a_cts,
        pts,
        ..
    } = pt_fixture::<BE, F, E>(&params, module, host_module);
    let short_pair = ckks_encrypt(
        &params,
        module,
        host_module,
        &encoder,
        &sk,
        params.k - params.base2k,
        &a_vecs[1].0,
        &a_vecs[1].1,
        &mut scratch.borrow(),
    );
    let coarse_pair = ckks_encrypt_with_prec(
        &params,
        module,
        host_module,
        &encoder,
        &sk,
        params.k - DELTA_LOG_DELTA,
        &a_vecs[2].0,
        &a_vecs[2].1,
        precision_at(&params, params.prec().log_delta() - DELTA_LOG_DELTA),
        &mut scratch.borrow(),
    );
    let apart = [&a_cts[0], &short_pair, &coarse_pair];
    let pt_refs: Vec<&_> = pts.iter().collect();
    let off_grid = params.k - params.base2k - 3;
    let res = dot(module, &apart, &pt_refs, off_grid);
    assert_dot_product_pt_meta("dot_product_pt unaligned", &res, off_grid, &apart, &pt_refs);
    assert_decrypt_precision(
        "dot_product_pt unaligned",
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

/// `ckks_dot_product_pt` refuses on an empty arena, `dst` unchanged: no pair, pairs of two lengths and, with the
/// offending term at every position, a ciphertext of another rank, a plaintext of another radix or of a degree not
/// dividing the terms', a ciphertext without the budget for its plaintext's scale, a term of another slot layout and, on
/// a replicated layout, a ciphertext of another scale.
pub fn test_dot_product_pt_rejects_invalid_operands<BE, F, E>(
    params: CKKSTestParams,
    module: &Module<BE>,
    host_module: &Module<HostBytesBackend>,
) where
    BE: TestContextBackend,
    Module<BE>: TestContextModule<BE>,
    F: TestScalar,
    E: ReimNew<F>,
{
    let PtFixture {
        encoder,
        b_pts,
        a_cts,
        pts,
        ..
    } = pt_fixture::<BE, F, E>(&params, module, host_module);
    let a_refs: Vec<&_> = a_cts.iter().collect();
    let pt_refs: Vec<&_> = pts.iter().collect();
    const OP: &str = "ckks_dot_product_pt";
    let constant = ckks_pt_cst::<BE, F, Standard>(host_module, module, params.base2k.into(), PT_PREC, Some(0.5), None);
    let replicated = crate::SlotLayout::Replicated(crate::shaping::ShapingUnit::U16_40.layout(1).unwrap());
    let mut other_rank = module.ckks_ciphertext_alloc_with_rank::<Standard>(
        params.base2k.into(),
        params.k.into(),
        poulpy_core::layouts::Rank((params.rank % 2 + 1) as u32),
    );
    other_rank.set_meta(a_cts[0].meta());
    let other_radix = encode_and_upload_pt(
        host_module,
        module,
        &encoder,
        (params.base2k - 1).into(),
        params.prec(),
        &b_pts[0].0,
        &b_pts[0].1,
    );
    let mut res = dirty_ct(module, &a_cts[0], params.k);
    // Terms of half the degree, of which the fixture's plaintexts are no divisor.
    let half = CKKSTestParams {
        n: params.n / 2,
        ..params
    };
    let mut half_term = alloc_ct::<BE, Standard>(&half, module, params.k);
    half_term.set_meta(a_cts[0].meta());
    let mut half_res = dirty_ct(module, &half_term, params.k);
    let refused = |label: &str,
                   dst: &mut CKKSCiphertextOwned<BE>,
                   a: &[&CKKSCiphertextOwned<BE>],
                   b: &[&CKKSPlaintextOwned<BE>],
                   check: &dyn Fn(&CKKSError)| {
        let before = snapshot::<BE, _, Standard>(&*dst);
        let err = with_scratch::<BE, _>(0, |s| module.ckks_dot_product_pt(dst, a, b, s)).unwrap_err();
        check(&err);
        assert_eq!(
            snapshot::<BE, _, Standard>(&*dst),
            before,
            "{label}: a refused dot product wrote dst"
        );
    };
    refused("no pair", &mut res, &[], &[], &|err| {
        assert_invalid_operand("no pair", err, OP, "inputs must contain at least one pair")
    });
    refused("two lengths", &mut res, &a_refs, &pt_refs[..2], &|err| {
        assert_invalid_operand(
            "two lengths",
            err,
            OP,
            "length mismatch between ct vector (3) and weight vector (2)",
        )
    });
    for position in 0..N {
        let swap_b = |pt| {
            let mut b = pt_refs.clone();
            b[position] = pt;
            b
        };
        let swap_a = |ct| {
            let mut a = a_refs.clone();
            a[position] = ct;
            a
        };
        refused(
            &format!("rank at {position}"),
            &mut res,
            &swap_a(&other_rank),
            &pt_refs,
            &|err| {
                assert_invalid_operand(
                    "another rank",
                    err,
                    OP,
                    "a ciphertext of another degree, rank or radix than dst",
                )
            },
        );
        refused(
            &format!("radix at {position}"),
            &mut res,
            &a_refs,
            &swap_b(&other_radix),
            &|err| {
                assert!(
                    matches!(
                        err.composition(),
                        Some(CKKSCompositionError::PlaintextBase2KMismatch { op: OP, .. })
                    ),
                    "a plaintext of another radix: {err}"
                )
            },
        );
        // The other terms by constants, which divide every degree.
        let mut wide = vec![&constant; N];
        wide[position] = &pts[position];
        let degree_mismatch = CKKSCompositionError::PlaintextDegreeMismatch {
            op: OP,
            ct_n: half_term.n().as_usize(),
            pt_n: pts[position].n().as_usize(),
            rule: DIVISOR,
        };
        refused(
            &format!("degree at {position}"),
            &mut half_res,
            &[&half_term; N],
            &wide,
            &|err| assert_ckks_error("a plaintext of a non-dividing degree", err, degree_mismatch.clone()),
        );
        let mut short = a_cts[position].clone();
        short.set_log_budget(pts[position].log_delta() - 1);
        let underflow = CKKSCompositionError::MultiplicationPrecisionUnderflow {
            op: OP,
            lhs_log_budget: short.log_budget(),
            rhs_log_budget: pts[position].log_budget(),
            lhs_log_delta: short.log_delta(),
            rhs_log_delta: pts[position].log_delta(),
        };
        refused(
            &format!("budget at {position}"),
            &mut res,
            &swap_a(&short),
            &pt_refs,
            &|err| assert_ckks_error("a term without the budget for its plaintext", err, underflow.clone()),
        );
        let mut relabelled = a_cts[position].clone();
        relabelled.set_slot_layout(replicated);
        refused(
            &format!("layout at {position}"),
            &mut res,
            &swap_a(&relabelled),
            &pt_refs,
            &|err| {
                assert!(
                    matches!(err.composition(), Some(CKKSCompositionError::InvalidOperand { op: OP, .. })),
                    "a term of another slot layout: {err}"
                )
            },
        );
        // Replicated terms by constants, one of another scale.
        let mut terms = a_cts.to_vec();
        terms.iter_mut().for_each(|t| t.set_slot_layout(replicated));
        let finer = terms[position].log_delta() + 1;
        terms[position].set_log_delta(finer);
        let constants = vec![&constant; N];
        refused(
            &format!("scale at {position}"),
            &mut res,
            &terms.iter().collect::<Vec<_>>(),
            &constants,
            &|err| assert_invalid_operand("replicated scales", err, OP, "a replicated layout adds at one scale"),
        );
    }
}

/// `ckks_dot_product_prepared_pt` of three ciphertexts prepared by `ckks_left_operand_prepare_alloc` by plaintexts prepared by
/// `ckks_right_operand_prepare_alloc`, read through the Galois elements of the rotations `0`, `1` and `-3`, decrypts to
/// `Σ a_t ⊙ Rot_{r_t}(b_t)` at the rule of `ckks_mul_pt_into`; one term through `1` is `ckks_mul_pt_into` of its
/// sources, bit for bit, on exactly its scratch; the DFT forms, the first term then the other two added, finished, are
/// the sum, bit for bit; an even element, slices of two lengths, terms of two scales and an accumulator of another limb
/// count are rejected before `dst` changes.
pub fn test_dot_product_prepared_pt<BE, F, E>(params: CKKSTestParams, module: &Module<BE>, host_module: &Module<HostBytesBackend>)
where
    BE: TestContextBackend,
    for<'a> <BE as poulpy_hal::layouts::Backend>::BufRef<'a>: poulpy_hal::layouts::HostDataRef,
    for<'a> <BE as poulpy_hal::layouts::Backend>::BufMut<'a>: poulpy_hal::layouts::HostDataMut,
    Module<BE>: TestContextModule<BE> + GLWEMulPreparedPlainSum<BE, Standard> + VecZnxDftAlloc<BE>,
    F: TestScalar,
    E: ReimNew<F>,
{
    const OP: &str = "ckks_dot_product_prepared_pt";
    const ROTATIONS: [i64; N] = [0, 1, -3];
    let m = params.n / 2;
    let encoder = ReferenceEncoder::<E>::new(m).unwrap();
    let (re1, im1) = test_vector_1::<F>(m);
    let (re2, im2) = test_vector_2::<F>(m);
    let sk = super::helpers::gen_sk(&params, module, host_module, [0u8; 32]);
    let mut scratch = alloc_scratch(&params, module);
    let log_delta = params.prec().log_delta();
    let prepare_left = |ct: &CKKSCiphertextOwned<BE, Standard>| {
        with_scratch::<BE, _>(module.ckks_left_operand_prepare_tmp_bytes(ct), |s| {
            module.ckks_left_operand_prepare_alloc(ct, PrepareHint::Reuse, s)
        })
        .unwrap()
    };

    let a_vecs = three_vectors(&re1, &im1, &re2, &im2);
    let b_vecs = three_vectors(&re2, &im2, &re1, &im1);
    let mut want_re = vec![F::zero(); m];
    let mut want_im = vec![F::zero(); m];
    for ((a, b), r) in a_vecs.iter().zip(&b_vecs).zip(ROTATIONS) {
        let (b_re, b_im) = want_rotate(&quantize(&b.0, log_delta), &quantize(&b.1, log_delta), r, m);
        cmul_acc(&mut want_re, &mut want_im, &a.0, &a.1, &b_re, &b_im);
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
    let lefts: Vec<_> = a_cts.iter().map(prepare_left).collect();
    let rights: Vec<_> = pts
        .iter()
        .map(|pt| {
            with_scratch::<BE, _>(module.ckks_right_operand_prepare_tmp_bytes(pt), |s| {
                module.ckks_right_operand_prepare_alloc(pt, PrepareHint::Reuse, s)
            })
            .unwrap()
        })
        .collect();
    let galois: Vec<i64> = ROTATIONS.iter().map(|&r| module.ckks_galois_element::<Standard>(r)).collect();
    let (a, b): (Vec<&_>, Vec<&_>) = (lefts.iter().collect(), rights.iter().collect());

    let mut dst = alloc_ct(&params, module, params.k);
    let bytes = module.ckks_dot_product_prepared_pt_tmp_bytes(&dst, a[0], b[0]);
    with_scratch::<BE, _>(bytes, |s| module.ckks_dot_product_prepared_pt(&mut dst, &a, &b, &galois, s)).unwrap();
    assert_eq!(
        (dst.log_delta(), dst.log_budget()),
        (a_cts[0].log_delta(), a_cts[0].log_budget() - pts[0].log_delta()),
        "{OP}: metadata"
    );
    assert_decrypt_precision(
        OP,
        &params,
        module,
        &encoder,
        &dst,
        &sk,
        &want_re,
        &want_im,
        &mut scratch.borrow(),
    );

    let (cols, acc_size) = (dst.rank().as_usize() + 1, a[0].size() + b[0].size());
    let mut acc = module.vec_znx_dft_alloc(params.n, cols, acc_size);
    let mut from_dft = alloc_ct(&params, module, params.k);
    with_scratch::<BE, _>(
        module.ckks_dot_product_prepared_pt_dft_tmp_bytes(&from_dft, a[0], b[0]),
        |s| module.ckks_dot_product_prepared_pt_dft(&mut acc.to_backend_mut(), &from_dft, &a[..1], &b[..1], &galois[..1], s),
    )
    .unwrap();
    with_scratch::<BE, _>(
        module.ckks_dot_product_prepared_pt_dft_add_tmp_bytes(&from_dft, a[0], b[0]),
        |s| module.ckks_dot_product_prepared_pt_dft_add(&mut acc.to_backend_mut(), &from_dft, &a[1..], &b[1..], &galois[1..], s),
    )
    .unwrap();
    with_scratch::<BE, _>(
        module.ckks_dot_product_prepared_pt_from_dft_tmp_bytes(&from_dft, a[0], b[0]),
        |s| module.ckks_dot_product_prepared_pt_from_dft(&mut from_dft, &mut acc.to_backend_mut(), a[0], b[0], s),
    )
    .unwrap();
    assert!(
        snapshot::<BE, _, Standard>(&from_dft) == snapshot::<BE, _, Standard>(&dst),
        "{OP}: the DFT forms are not the summed product"
    );
    let mut short = module.vec_znx_dft_alloc(params.n, cols, acc_size - 1);
    let reason = "acc must hold the degree and rank + 1 columns of res on a.size() + b.size() limbs";
    let err = with_scratch::<BE, _>(0, |s| {
        module.ckks_dot_product_prepared_pt_dft(&mut short.to_backend_mut(), &from_dft, &a, &b, &galois, s)
    })
    .unwrap_err();
    assert_ckks_error(
        "short accumulator",
        &err,
        CKKSCompositionError::InvalidOperand {
            op: "ckks_dot_product_prepared_pt_dft",
            reason: reason.into(),
        },
    );
    let before = snapshot::<BE, _, Standard>(&from_dft);
    let err = with_scratch::<BE, _>(0, |s| {
        module.ckks_dot_product_prepared_pt_from_dft(&mut from_dft, &mut short.to_backend_mut(), a[0], b[0], s)
    })
    .unwrap_err();
    assert_ckks_error(
        "short accumulator",
        &err,
        CKKSCompositionError::InvalidOperand {
            op: "ckks_dot_product_prepared_pt_from_dft",
            reason: reason.into(),
        },
    );
    assert!(
        snapshot::<BE, _, Standard>(&from_dft) == before,
        "{OP}_from_dft: a rejection changed res"
    );

    let mut want = alloc_ct(&params, module, params.k);
    with_scratch::<BE, _>(module.ckks_mul_pt_tmp_bytes(&want, &a_cts[0], &pts[0]), |s| {
        module.ckks_mul_pt_into(&mut want, &a_cts[0], &pts[0], s)
    })
    .unwrap();
    let mut have = alloc_ct(&params, module, params.k);
    with_scratch::<BE, _>(bytes, |s| {
        module.ckks_dot_product_prepared_pt(&mut have, &a[..1], &b[..1], &[1], s)
    })
    .unwrap();
    assert!(
        snapshot::<BE, _, Standard>(&have) == snapshot::<BE, _, Standard>(&want),
        "{OP}: one term through 1 is not ckks_mul_pt_into"
    );

    let before = snapshot::<BE, _, Standard>(&have);
    let even = [galois[0], 2, galois[2]];
    let err = with_scratch::<BE, _>(0, |s| module.ckks_dot_product_prepared_pt(&mut have, &a, &b, &even, s)).unwrap_err();
    assert_ckks_error(
        "even element",
        &err,
        CKKSCompositionError::InvalidOperand {
            op: OP,
            reason: "a Galois element is even".into(),
        },
    );
    let err = with_scratch::<BE, _>(0, |s| module.ckks_dot_product_prepared_pt(&mut have, &a, &b[..2], &galois, s)).unwrap_err();
    assert_ckks_error(
        "slices of two lengths",
        &err,
        CKKSCompositionError::InvalidOperand {
            op: OP,
            reason: format!("length mismatch between ct vector ({}) and weight vector (2)", a.len()),
        },
    );
    let err = with_scratch::<BE, _>(0, |s| module.ckks_dot_product_prepared_pt(&mut have, &a, &b, &galois[..2], s)).unwrap_err();
    assert_ckks_error(
        "two Galois lengths",
        &err,
        CKKSCompositionError::InvalidOperand {
            op: OP,
            reason: "one Galois element per term".into(),
        },
    );
    let mut rescaled = a_cts[1].clone();
    rescaled.set_log_delta(a_cts[1].log_delta() - 1);
    let rescaled = prepare_left(&rescaled);
    let mixed = [a[0], &rescaled, a[2]];
    let err = with_scratch::<BE, _>(0, |s| module.ckks_dot_product_prepared_pt(&mut have, &mixed, &b, &galois, s)).unwrap_err();
    assert_ckks_error(
        "two scales",
        &err,
        CKKSCompositionError::InvalidOperand {
            op: OP,
            reason: "every a[t] must share one layout and scale, and every b[t] one".into(),
        },
    );
    assert!(snapshot::<BE, _, Standard>(&have) == before, "{OP}: a rejection changed dst");
}

/// `res += Σ a[t] · σ_{galois[t]}(b[t])` on `res` at the sum's frame, the scale of the terms and the width the product
/// leaves: one rounding of the total, so it decrypts to `c + Σ a[t] · rot(b[t], r_t)` and keeps `res`'s metadata, as the
/// sum into a temporary added to `res` does, on exactly its query's bytes. It runs one term and two, through rotations
/// 0, 1 and -3, by plaintexts at the suite's precision and by plaintexts narrower than one limb, whose product's offset
/// falls below a limb. A `res` at another scale, or wider than the sum, is refused before any write, and one byte short
/// of its query, `Internal`, `res` unchanged.
pub fn test_dot_product_prepared_pt_add<BE, F, E>(
    params: CKKSTestParams,
    module: &Module<BE>,
    host_module: &Module<HostBytesBackend>,
) where
    BE: TestContextBackend,
    for<'a> <BE as poulpy_hal::layouts::Backend>::BufRef<'a>: poulpy_hal::layouts::HostDataRef,
    for<'a> <BE as poulpy_hal::layouts::Backend>::BufMut<'a>: poulpy_hal::layouts::HostDataMut,
    Module<BE>: TestContextModule<BE> + GLWEMulPreparedPlainSum<BE, Standard>,
    F: TestScalar,
    E: ReimNew<F>,
{
    const OP: &str = "ckks_dot_product_prepared_pt_add";
    let m = params.n / 2;
    let encoder = ReferenceEncoder::<E>::new(m).unwrap();
    let (re1, im1) = test_vector_1::<F>(m);
    let (re2, im2) = test_vector_2::<F>(m);
    let sk = super::helpers::gen_sk(&params, module, host_module, [0u8; 32]);
    let mut scratch = alloc_scratch(&params, module);
    let encrypt = |k: usize, re: &[F], im: &[F], scratch: &mut ScratchArena<'_, BE>| {
        ckks_encrypt(&params, module, host_module, &encoder, &sk, k, re, im, scratch)
    };
    let left = |ct: &CKKSCiphertextOwned<BE, Standard>| {
        with_scratch::<BE, _>(module.ckks_left_operand_prepare_tmp_bytes(ct), |s| {
            module.ckks_left_operand_prepare_alloc(ct, PrepareHint::Reuse, s)
        })
        .unwrap()
    };
    let a = [
        left(&encrypt(params.k, &re1, &im1, &mut scratch.borrow())),
        left(&encrypt(params.k, &re2, &im2, &mut scratch.borrow())),
    ];
    let full = params.prec();
    let narrow_k = params.base2k - 1;
    let narrow = CKKSLayout {
        glwe_layout: GLWELayout {
            k: TorusPrecision(narrow_k as u32),
            ..full.glwe_layout
        },
        meta: GLWEPlaintextMeta {
            scale: Scale::Log(full.log_delta().min(narrow_k - 2)),
            ..full.meta
        },
    };
    for prec in [full, narrow] {
        let p = prec.log_delta();
        let right = |re: &[F], im: &[F]| {
            let pt = encode_and_upload_pt(host_module, module, &encoder, params.base2k.into(), prec, re, im);
            with_scratch::<BE, _>(module.ckks_right_operand_prepare_tmp_bytes(&pt), |s| {
                module.ckks_right_operand_prepare_alloc(&pt, PrepareHint::Reuse, s)
            })
            .unwrap()
        };
        let b = [right(&re2, &im2), right(&re1, &im1)];
        // `c`, the value accumulated into, at the sum's frame: the width `a` leaves through `b`'s scale.
        let k_frame = params.k - p;
        for (terms, r) in [(1, 0), (1, 1), (2, -3)] {
            let label = format!("{OP} p={p} terms={terms} r={r}");
            let rotations = [r, r + 2];
            let galois: Vec<i64> = rotations[..terms]
                .iter()
                .map(|&r| module.ckks_galois_element::<Standard>(r))
                .collect();
            let (mut want_re, mut want_im) = (im1.clone(), re1.clone());
            // Term `t` multiplies `a[t]`'s values by the rotation of `b[t]`'s, the other test vector.
            let values = [((&re1, &im1), (&re2, &im2)), ((&re2, &im2), (&re1, &im1))];
            for (&rotation, ((a_re, a_im), (b_re, b_im))) in rotations[..terms].iter().zip(values) {
                let (b_re, b_im) = want_rotate(&quantize(b_re, p), &quantize(b_im, p), rotation, m);
                cmul_acc(&mut want_re, &mut want_im, a_re, a_im, &b_re, &b_im);
            }
            let a_refs: Vec<&CKKSLeftOperandPrepared<BE, Standard>> = a[..terms].iter().collect();
            let b_refs: Vec<&CKKSRightOperandPrepared<BE, Standard>> = b[..terms].iter().collect();

            let mut res = encrypt(k_frame, &im1, &re1, &mut scratch.borrow());
            let layout = (res.glwe_layout(), res.log_delta(), res.log_budget());
            let bytes = module.ckks_dot_product_prepared_pt_add_tmp_bytes(&res, &a[0], &b[0]);
            with_scratch::<BE, _>(bytes, |s| {
                module.ckks_dot_product_prepared_pt_add(&mut res, &a_refs, &b_refs, &galois, s)
            })
            .unwrap();
            assert_eq!(
                (res.glwe_layout(), res.log_delta(), res.log_budget()),
                layout,
                "{label}: metadata"
            );
            assert!(res.is_canonical() && res.noise().is_none(), "{label}: canonical, untagged");
            assert_decrypt_precision_at_log_delta(
                &label,
                &params,
                module,
                &encoder,
                &res,
                &sk,
                &want_re,
                &want_im,
                p,
                &mut scratch.borrow(),
            );

            // The sum into a temporary, then added: the same value.
            let mut composed = encrypt(k_frame, &im1, &re1, &mut scratch.borrow());
            let mut tmp = alloc_ct(&params, module, k_frame);
            with_scratch::<BE, _>(module.ckks_dot_product_prepared_pt_tmp_bytes(&tmp, &a[0], &b[0]), |s| {
                module.ckks_dot_product_prepared_pt(&mut tmp, &a_refs, &b_refs, &galois, s)
            })
            .unwrap();
            with_scratch::<BE, _>(module.ckks_add_tmp_bytes(composed.size()), |s| {
                module.ckks_add_assign(&mut composed, &tmp, s)
            })
            .unwrap();
            assert_eq!(
                (composed.glwe_layout(), composed.log_delta(), composed.log_budget()),
                layout,
                "{label}: composed metadata"
            );
            assert_decrypt_precision_at_log_delta(
                &format!("{label}, composed"),
                &params,
                module,
                &encoder,
                &composed,
                &sk,
                &want_re,
                &want_im,
                p,
                &mut scratch.borrow(),
            );
        }

        // Refused before any write: `res` at another scale, `res` wider than the sum, and a scratch one byte short.
        let galois = [module.ckks_galois_element::<Standard>(1)];
        let (a_refs, b_refs) = ([&a[0]], [&b[0]]);
        let mut other_scale = encrypt(k_frame, &re1, &im1, &mut scratch.borrow());
        other_scale.set_log_delta(other_scale.log_delta() - 1);
        let mut wider = encrypt(k_frame + params.base2k, &re1, &im1, &mut scratch.borrow());
        for (label, res) in [("another scale", &mut other_scale), ("wider than the sum", &mut wider)] {
            let before = snapshot::<BE, _, Standard>(&*res);
            let bytes = module.ckks_dot_product_prepared_pt_add_tmp_bytes(&*res, &a[0], &b[0]);
            let err = with_scratch::<BE, _>(bytes, |s| {
                module.ckks_dot_product_prepared_pt_add(res, &a_refs, &b_refs, &galois, s)
            })
            .unwrap_err();
            assert_invalid_operand(label, &err, OP, "the product's scale and budget");
            assert!(
                snapshot::<BE, _, Standard>(&*res) == before,
                "{OP} p={p} {label}: res written before the refusal"
            );
        }
        let mut res = encrypt(k_frame, &im1, &re1, &mut scratch.borrow());
        let before = snapshot::<BE, _, Standard>(&res);
        let bytes = module.ckks_dot_product_prepared_pt_add_tmp_bytes(&res, &a[0], &b[0]);
        let err = with_scratch::<BE, _>(bytes - 1, |s| {
            module.ckks_dot_product_prepared_pt_add(&mut res, &a_refs, &b_refs, &galois, s)
        })
        .unwrap_err();
        assert_ckks_internal(
            &format!("{OP} p={p}: one byte short"),
            &err,
            &format!("{OP}: scratch below its query"),
        );
        assert!(
            snapshot::<BE, _, Standard>(&res) == before,
            "{OP} p={p}: a refused call wrote res"
        );
    }
}

pub fn test_dot_product_pt_const_aligned<BE, F, E>(
    params: CKKSTestParams,
    module: &Module<BE>,
    host_module: &Module<HostBytesBackend>,
) where
    BE: TestContextBackend,
    for<'a> <BE as poulpy_hal::layouts::Backend>::BufRef<'a>: poulpy_hal::layouts::HostDataRef,
    for<'a> <BE as poulpy_hal::layouts::Backend>::BufMut<'a>: poulpy_hal::layouts::HostDataMut,
    Module<BE>: TestContextModule<BE>,
    F: TestScalar,
    E: ReimNew<F>,
{
    let PtFixture {
        encoder,
        sk,
        mut scratch,
        a_vecs,
        a_cts,
        ..
    } = pt_fixture::<BE, F, E>(&params, module, host_module);
    let m = params.n / 2;
    let const_coeffs: [f64; 3] = [0.25, 0.125, -0.3125];
    let (mut want_re, mut want_im) = (vec![F::zero(); m], vec![F::zero(); m]);
    for (i, r) in const_coeffs.iter().enumerate() {
        let quantized = quantized_const::<F>(*r, 0.0, PT_PREC.log_delta()).0;
        cmul_scalar_acc(&mut want_re, &mut want_im, &a_vecs[i].0, &a_vecs[i].1, quantized, F::zero());
    }
    let consts: Vec<CKKSPlaintextOwned<BE>> = const_coeffs
        .iter()
        .map(|r| ckks_pt_cst::<BE, F, Standard>(host_module, module, params.base2k.into(), PT_PREC, Some(*r), None))
        .collect();
    let a_refs: Vec<&_> = a_cts.iter().collect();
    let const_refs: Vec<&_> = consts.iter().collect();
    let ct_res = dot(module, &a_refs, &const_refs, params.k);
    assert_dot_product_pt_meta("dot_product_pt_const_aligned", &ct_res, params.k, &a_refs, &const_refs);
    assert_decrypt_precision(
        "dot_product_pt_const_aligned",
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
