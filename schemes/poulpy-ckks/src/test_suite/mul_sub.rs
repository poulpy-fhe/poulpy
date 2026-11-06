//! Tests for `CKKSMulSubOps` — fused `dst -= a · b` variants.
//!
//! # Test inventory
//!
//! | Function | Path exercised |
//! |----------|----------------|
//! | [`test_mul_sub_ct_aligned`] | ct-ct, all operands aligned |
//! | [`test_mul_sub_ct_unaligned_dst`] | ct-ct with `dst` at a lower `log_budget` |
//! | [`test_mul_sub_pt_vec_aligned`] | ZNX plaintext, aligned |
//! | [`test_mul_sub_pt_vec_into_delta_log_delta`] | ZNX plaintext at lower `log_delta` |
//! | [`test_mul_sub_pt_const_into_aligned`] | ZNX constant, aligned |
//! | [`test_mul_sub_pt_const_zero_preserves_dst_meta`] | ZNX zero constant no-op |

use poulpy_hal::{
    api::{ReimNew, ScratchOwnedBorrow},
    layouts::{HostBytesBackend, Module, Standard},
};

use crate::{CKKSInfos, api::CKKSMulSubOps};

use super::helpers::{
    MUL_CONST, PT_PREC, TestContextBackend, TestContextModule, TestScalar, alloc_scratch, assert_ckks_internal, assert_ct_meta,
    assert_decrypt_precision, assert_decrypt_precision_at_log_delta, ckks_encrypt, ckks_pt_cst_full, ckks_snapshot, ckks_spec,
    encode_and_upload_pt, gen_sk, gen_sk_with_raw, gen_tsk, quantize, quantized_const, quantized_slots, test_vector_1,
    test_vector_2,
};
use super::parity::helpers::{snapshot, with_scratch};

use crate::{test_suite::CKKSTestParams, test_suite::reference_encoder::ReferenceEncoder};

fn scaled<F: TestScalar>(v: &[F], scale: F) -> Vec<F> {
    v.iter().copied().map(|x| x * scale).collect()
}

fn cmul<F: TestScalar>(a_re: &[F], a_im: &[F], b_re: &[F], b_im: &[F]) -> (Vec<F>, Vec<F>) {
    let m = a_re.len();
    let mut re = Vec::with_capacity(m);
    let mut im = Vec::with_capacity(m);
    for i in 0..m {
        re.push(a_re[i] * b_re[i] - a_im[i] * b_im[i]);
        im.push(a_re[i] * b_im[i] + a_im[i] * b_re[i]);
    }
    (re, im)
}

fn cmul_scalar<F: TestScalar>(a_re: &[F], a_im: &[F], c_re: F, c_im: F) -> (Vec<F>, Vec<F>) {
    let m = a_re.len();
    let mut re = Vec::with_capacity(m);
    let mut im = Vec::with_capacity(m);
    for i in 0..m {
        re.push(a_re[i] * c_re - a_im[i] * c_im);
        im.push(a_re[i] * c_im + a_im[i] * c_re);
    }
    (re, im)
}

pub fn test_mul_sub_ct_aligned<BE, F, E>(params: CKKSTestParams, module: &Module<BE>, host_module: &Module<HostBytesBackend>)
where
    BE: TestContextBackend<Ring = Standard>,
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

    let half = F::from_f64(0.5).unwrap();
    let dst_re = scaled(&re1, half);
    let dst_im = scaled(&im1, half);
    let a_re = scaled(&re2, half);
    let a_im = scaled(&im2, half);
    let b_re = scaled(&re1, half);
    let b_im = scaled(&im2, half);

    let (prod_re, prod_im) = cmul(&a_re, &a_im, &b_re, &b_im);
    let want_re: Vec<F> = (0..dst_re.len()).map(|i| dst_re[i] - prod_re[i]).collect();
    let want_im: Vec<F> = (0..dst_im.len()).map(|i| dst_im[i] - prod_im[i]).collect();

    let mut dst = ckks_encrypt(
        &params,
        module,
        host_module,
        &encoder,
        &sk,
        params.k,
        &dst_re,
        &dst_im,
        &mut scratch.borrow(),
    );
    let a = ckks_encrypt(
        &params,
        module,
        host_module,
        &encoder,
        &sk,
        params.k,
        &a_re,
        &a_im,
        &mut scratch.borrow(),
    );
    let b = ckks_encrypt(
        &params,
        module,
        host_module,
        &encoder,
        &sk,
        params.k,
        &b_re,
        &b_im,
        &mut scratch.borrow(),
    );
    let bytes = module.ckks_mul_sub_ct_tmp_bytes(&dst, &a, &b, &tsk);
    let before = snapshot::<BE, _>(&dst);
    let err = with_scratch::<BE, _>(bytes - 1, |s| module.ckks_mul_sub_ct_into(&mut dst, &a, &b, &tsk, s)).unwrap_err();
    assert_ckks_internal(
        "ckks_mul_sub_ct_into: one byte short",
        &err,
        "ckks_mul_sub_ct_into: scratch below its query",
    );
    assert_eq!(
        snapshot::<BE, _>(&dst),
        before,
        "ckks_mul_sub_ct_into: a refused call wrote dst"
    );
    with_scratch::<BE, _>(bytes, |s| module.ckks_mul_sub_ct_into(&mut dst, &a, &b, &tsk, s)).unwrap();
    assert_decrypt_precision(
        "mul_sub_ct_aligned",
        &params,
        module,
        &encoder,
        &dst,
        &sk,
        &want_re,
        &want_im,
        &mut scratch.borrow(),
    );
}

pub fn test_mul_sub_ct_unaligned_dst<BE, F, E>(
    params: CKKSTestParams,
    module: &Module<BE>,
    host_module: &Module<HostBytesBackend>,
) where
    BE: TestContextBackend<Ring = Standard>,
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

    let half = F::from_f64(0.5).unwrap();
    let dst_re = scaled(&re1, half);
    let dst_im = scaled(&im1, half);
    let a_re = scaled(&re2, half);
    let a_im = scaled(&im2, half);
    let b_re = scaled(&re1, half);
    let b_im = scaled(&im2, half);

    let (prod_re, prod_im) = cmul(&a_re, &a_im, &b_re, &b_im);
    let want_re: Vec<F> = (0..dst_re.len()).map(|i| dst_re[i] - prod_re[i]).collect();
    let want_im: Vec<F> = (0..dst_im.len()).map(|i| dst_im[i] - prod_im[i]).collect();

    let smaller_k = params.k - params.base2k + 1;
    let mut dst = ckks_encrypt(
        &params,
        module,
        host_module,
        &encoder,
        &sk,
        smaller_k,
        &dst_re,
        &dst_im,
        &mut scratch.borrow(),
    );
    let a = ckks_encrypt(
        &params,
        module,
        host_module,
        &encoder,
        &sk,
        params.k,
        &a_re,
        &a_im,
        &mut scratch.borrow(),
    );
    let b = ckks_encrypt(
        &params,
        module,
        host_module,
        &encoder,
        &sk,
        params.k,
        &b_re,
        &b_im,
        &mut scratch.borrow(),
    );
    with_scratch::<BE, _>(module.ckks_mul_sub_ct_tmp_bytes(&dst, &a, &b, &tsk), |s| {
        module.ckks_mul_sub_ct_into(&mut dst, &a, &b, &tsk, s)
    })
    .unwrap();
    assert_decrypt_precision(
        "mul_sub_ct_unaligned_dst",
        &params,
        module,
        &encoder,
        &dst,
        &sk,
        &want_re,
        &want_im,
        &mut scratch.borrow(),
    );
}

pub fn test_mul_sub_pt_vec_aligned<BE, F, E>(params: CKKSTestParams, module: &Module<BE>, host_module: &Module<HostBytesBackend>)
where
    BE: TestContextBackend<Ring = Standard>,
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
    let sk = super::helpers::gen_sk(&params, module, host_module, [0u8; 32]);
    let mut scratch = alloc_scratch(&params, module);

    let half = F::from_f64(0.5).unwrap();
    let dst_re = scaled(&re1, half);
    let dst_im = scaled(&im1, half);
    let a_re = scaled(&re2, half);
    let a_im = scaled(&im2, half);
    let b_re_raw = scaled(&re1, half);
    let b_im_raw = scaled(&im2, half);
    let (b_re, b_im) = (
        quantize(&b_re_raw, params.prec().log_delta()),
        quantize(&b_im_raw, params.prec().log_delta()),
    );

    let (prod_re, prod_im) = cmul(&a_re, &a_im, &b_re, &b_im);
    let want_re: Vec<F> = (0..dst_re.len()).map(|i| dst_re[i] - prod_re[i]).collect();
    let want_im: Vec<F> = (0..dst_im.len()).map(|i| dst_im[i] - prod_im[i]).collect();

    let mut dst = ckks_encrypt(
        &params,
        module,
        host_module,
        &encoder,
        &sk,
        params.k,
        &dst_re,
        &dst_im,
        &mut scratch.borrow(),
    );
    let a = ckks_encrypt(
        &params,
        module,
        host_module,
        &encoder,
        &sk,
        params.k,
        &a_re,
        &a_im,
        &mut scratch.borrow(),
    );
    let pt = encode_and_upload_pt(
        host_module,
        module,
        &encoder,
        params.base2k.into(),
        params.prec(),
        &b_re_raw,
        &b_im_raw,
    );
    let bytes = module.ckks_mul_sub_pt_vec_tmp_bytes(&dst, &a, &pt);
    let before = snapshot::<BE, _>(&dst);
    let err = with_scratch::<BE, _>(bytes - 1, |s| module.ckks_mul_sub_pt_vec_into(&mut dst, &a, &pt, s)).unwrap_err();
    assert_ckks_internal(
        "ckks_mul_sub_pt_vec_into: one byte short",
        &err,
        "ckks_mul_sub_pt_vec_into: scratch below its query",
    );
    assert_eq!(
        snapshot::<BE, _>(&dst),
        before,
        "ckks_mul_sub_pt_vec_into: a refused call wrote dst"
    );
    with_scratch::<BE, _>(bytes, |s| module.ckks_mul_sub_pt_vec_into(&mut dst, &a, &pt, s)).unwrap();
    assert_decrypt_precision(
        "mul_sub_pt_vec_aligned",
        &params,
        module,
        &encoder,
        &dst,
        &sk,
        &want_re,
        &want_im,
        &mut scratch.borrow(),
    );
}

pub fn test_mul_sub_pt_vec_into_delta_log_delta<BE, F, E>(
    params: CKKSTestParams,
    module: &Module<BE>,
    host_module: &Module<HostBytesBackend>,
) where
    BE: TestContextBackend<Ring = Standard>,
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
    let sk = super::helpers::gen_sk(&params, module, host_module, [0u8; 32]);
    let mut scratch = alloc_scratch(&params, module);

    let half = F::from_f64(0.5).unwrap();
    let dst_re = scaled(&re1, half);
    let dst_im = scaled(&im1, half);
    let a_re = scaled(&re2, half);
    let a_im = scaled(&im2, half);
    let b_re_raw = scaled(&re2, half);
    let b_im_raw = scaled(&im2, half);
    let (b_re, b_im) = quantized_slots(host_module, &encoder, params.base2k.into(), PT_PREC, &b_re_raw, &b_im_raw);

    let (prod_re, prod_im) = cmul(&a_re, &a_im, &b_re, &b_im);
    let want_re: Vec<F> = (0..dst_re.len()).map(|i| dst_re[i] - prod_re[i]).collect();
    let want_im: Vec<F> = (0..dst_im.len()).map(|i| dst_im[i] - prod_im[i]).collect();

    let mut dst = ckks_encrypt(
        &params,
        module,
        host_module,
        &encoder,
        &sk,
        params.k,
        &dst_re,
        &dst_im,
        &mut scratch.borrow(),
    );
    let a = ckks_encrypt(
        &params,
        module,
        host_module,
        &encoder,
        &sk,
        params.k,
        &a_re,
        &a_im,
        &mut scratch.borrow(),
    );
    let pt = encode_and_upload_pt(
        host_module,
        module,
        &encoder,
        params.base2k.into(),
        PT_PREC,
        &b_re_raw,
        &b_im_raw,
    );
    with_scratch::<BE, _>(module.ckks_mul_sub_pt_vec_tmp_bytes(&dst, &a, &pt), |s| {
        module.ckks_mul_sub_pt_vec_into(&mut dst, &a, &pt, s)
    })
    .unwrap();
    assert_decrypt_precision_at_log_delta(
        "mul_sub_pt_vec_into_delta_log_delta",
        &params,
        module,
        &encoder,
        &dst,
        &sk,
        &want_re,
        &want_im,
        PT_PREC.log_delta(),
        &mut scratch.borrow(),
    );
}

pub fn test_mul_sub_pt_const_into_aligned<BE, F, E>(
    params: CKKSTestParams,
    module: &Module<BE>,
    host_module: &Module<HostBytesBackend>,
) where
    BE: TestContextBackend<Ring = Standard>,
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
    let sk = gen_sk(&params, module, host_module, [0u8; 32]);
    let mut scratch = alloc_scratch(&params, module);

    let half = F::from_f64(0.5).unwrap();
    let dst_re = scaled(&re1, half);
    let dst_im = scaled(&im1, half);
    let a_re = scaled(&re2, half);
    let a_im = scaled(&im2, half);

    let c_re_f64 = MUL_CONST.0;
    let (c_re, c_im) = quantized_const::<F>(c_re_f64, 0.0, PT_PREC.log_delta());
    let (prod_re, prod_im) = cmul_scalar(&a_re, &a_im, c_re, c_im);
    let want_re: Vec<F> = (0..dst_re.len()).map(|i| dst_re[i] - prod_re[i]).collect();
    let want_im: Vec<F> = (0..dst_im.len()).map(|i| dst_im[i] - prod_im[i]).collect();

    let mut dst = ckks_encrypt(
        &params,
        module,
        host_module,
        &encoder,
        &sk,
        params.k,
        &dst_re,
        &dst_im,
        &mut scratch.borrow(),
    );
    let a = ckks_encrypt(
        &params,
        module,
        host_module,
        &encoder,
        &sk,
        params.k,
        &a_re,
        &a_im,
        &mut scratch.borrow(),
    );
    let cst = ckks_pt_cst_full::<BE, F>(host_module, module, params.base2k.into(), PT_PREC, m, Some(c_re_f64), None);
    let bytes = module.ckks_mul_sub_pt_const_tmp_bytes(&dst, &a, &cst);
    let before = snapshot::<BE, _>(&dst);
    let err = with_scratch::<BE, _>(bytes - 1, |s| module.ckks_mul_sub_pt_const_into(&mut dst, &a, &cst, 0, s)).unwrap_err();
    assert_ckks_internal(
        "ckks_mul_sub_pt_const_into: one byte short",
        &err,
        "ckks_mul_sub_pt_const_into: scratch below its query",
    );
    assert_eq!(
        snapshot::<BE, _>(&dst),
        before,
        "ckks_mul_sub_pt_const_into: a refused call wrote dst"
    );
    with_scratch::<BE, _>(bytes, |s| module.ckks_mul_sub_pt_const_into(&mut dst, &a, &cst, 0, s)).unwrap();
    assert_decrypt_precision(
        "mul_sub_pt_const_into_aligned",
        &params,
        module,
        &encoder,
        &dst,
        &sk,
        &want_re,
        &want_im,
        &mut scratch.borrow(),
    );
}

pub fn test_mul_sub_pt_const_zero_preserves_dst_meta<BE, F, E>(
    params: CKKSTestParams,
    module: &Module<BE>,
    host_module: &Module<HostBytesBackend>,
) where
    BE: TestContextBackend<Ring = Standard>,
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
    let sk = gen_sk(&params, module, host_module, [0u8; 32]);
    let mut scratch = alloc_scratch(&params, module);

    let half = F::from_f64(0.5).unwrap();
    let dst_re = scaled(&re1, half);
    let dst_im = scaled(&im1, half);
    let a_re = scaled(&re2, half);
    let a_im = scaled(&im2, half);

    let mut dst = ckks_encrypt(
        &params,
        module,
        host_module,
        &encoder,
        &sk,
        params.k,
        &dst_re,
        &dst_im,
        &mut scratch.borrow(),
    );
    let a = ckks_encrypt(
        &params,
        module,
        host_module,
        &encoder,
        &sk,
        params.k,
        &a_re,
        &a_im,
        &mut scratch.borrow(),
    );
    let dst_meta = ckks_snapshot(&dst);
    let zero_prec = ckks_spec(params.n, params.base2k, 0, PT_PREC.log_budget());
    let cst = ckks_pt_cst_full::<BE, F>(host_module, module, params.base2k.into(), zero_prec, m, None, None);
    with_scratch::<BE, _>(module.ckks_mul_sub_pt_const_tmp_bytes(&dst, &a, &cst), |s| {
        module.ckks_mul_sub_pt_const_into(&mut dst, &a, &cst, 0, s)
    })
    .unwrap();

    assert_ct_meta("mul_sub_pt_const_zero", &dst, dst_meta.log_delta(), dst_meta.log_budget());
    assert_decrypt_precision(
        "mul_sub_pt_const_zero",
        &params,
        module,
        &encoder,
        &dst,
        &sk,
        &dst_re,
        &dst_im,
        &mut scratch.borrow(),
    );
}

/// A plaintext of another radix is the product's `PlaintextBase2KMismatch` of the multiply-accumulate form, before its
/// arena check, on an empty arena and on the query, for both plaintext forms.
pub fn test_mul_sub_pt_vec_into_rejects_base2k_mismatch<BE, F, E>(
    params: crate::test_suite::CKKSTestParams,
    module: &poulpy_hal::layouts::Module<BE>,
    _host_module: &poulpy_hal::layouts::Module<poulpy_hal::layouts::HostBytesBackend>,
) where
    BE: crate::test_suite::helpers::TestContextBackend<Ring = poulpy_hal::layouts::Standard>,
    poulpy_hal::layouts::Module<BE>: crate::test_suite::helpers::TestContextModule<BE>,
    F: crate::test_suite::helpers::TestScalar,
    E: ReimNew<F>,
{
    use crate::test_suite::{
        helpers::assert_ckks_error,
        parity::{
            helpers::{fixture_ciphertext, fixture_plaintext, snapshot, with_scratch},
            layout,
        },
    };
    use crate::{CKKSCompositionError, CKKSLayout, SlotsKind};
    let b = params.base2k;
    let ct_layout = layout(params, params.rank, 3 * b + 5, b - 1, 0, SlotsKind::Complex);
    let pt_layout = layout(params, 0, b + 3, b - 1, 0, SlotsKind::Complex);
    let other_radix = CKKSLayout {
        glwe_layout: poulpy_core::layouts::GLWELayout {
            base2k: (b - 1).into(),
            ..pt_layout.glwe_layout
        },
        meta: pt_layout.meta,
    };
    use crate::api::CKKSMulSubOps;
    let a = fixture_ciphertext(module, &ct_layout, 41);
    let pt = fixture_plaintext(module, &other_radix, 43);
    let mut res = fixture_ciphertext(module, &ct_layout, 31);
    let want = |op| CKKSCompositionError::PlaintextBase2KMismatch {
        op,
        ct_base2k: b,
        pt_base2k: b - 1,
    };
    for bytes in [0, module.ckks_mul_sub_pt_vec_tmp_bytes(&res, &a, &pt)] {
        let before = snapshot::<BE, _>(&res);
        let err = with_scratch::<BE, _>(bytes, |s| module.ckks_mul_sub_pt_vec_into(&mut res, &a, &pt, s)).unwrap_err();
        assert_ckks_error(
            "mul_sub_pt_vec_into: plaintext of another radix",
            &err,
            want("ckks_mul_sub_pt_vec_into"),
        );
        assert_eq!(snapshot::<BE, _>(&res), before, "a refused multiply-accumulate wrote dst");
    }
    for bytes in [0, module.ckks_mul_sub_pt_const_tmp_bytes(&res, &a, &pt)] {
        let before = snapshot::<BE, _>(&res);
        let err = with_scratch::<BE, _>(bytes, |s| module.ckks_mul_sub_pt_const_into(&mut res, &a, &pt, 0, s)).unwrap_err();
        assert_ckks_error(
            "mul_sub_pt_const_into: plaintext of another radix",
            &err,
            want("ckks_mul_sub_pt_const_into"),
        );
        assert_eq!(snapshot::<BE, _>(&res), before, "a refused multiply-accumulate wrote dst");
    }
}
