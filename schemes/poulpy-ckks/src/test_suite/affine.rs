//! Tests for `CKKSAffineOps` — scalar `offset + scale * ct`.
//!
//! # Test inventory
//!
//! | Function | Path exercised |
//! |----------|----------------|
//! | [`test_affine_pt_into_const_aligned`] | affine op by constants of degree 1 with aligned output |
//! | [`test_affine_pt_into_const_zero_bias_matches_mul`] | zero offset preserves multiply result metadata |
//! | [`test_affine_pt_assign_const_aligned`] | assign variant of the affine op by constants |
//! | [`test_affine_pt_into_aligned`] | full-vector affine op into a fresh destination |
//! | [`test_affine_pt_assign_aligned`] | full-vector affine assign in-place |

use poulpy_hal::{
    api::{ReimNew, ScratchOwnedBorrow},
    layouts::{HostBytesBackend, Module, Standard},
};

use crate::{CKKSInfos, api::CKKSAffineOps};

use super::helpers::{
    ADD_SUB_CONST, MUL_CONST, PT_PREC, TestContextBackend, TestContextModule, TestScalar, alloc_ct, alloc_scratch,
    assert_ct_meta, assert_decrypt_precision, ckks_encrypt, ckks_pt_cst, ckks_pt_cst_full, gen_sk, quantized_const,
    test_vector_1,
};
use super::parity::helpers::with_scratch;

use crate::{test_suite::CKKSTestParams, test_suite::reference_encoder::ReferenceEncoder};

fn scaled<F: TestScalar>(v: &[F], scale: F) -> Vec<F> {
    v.iter().copied().map(|x| x * scale).collect()
}

pub fn test_affine_pt_into_const_aligned<BE, F, E>(
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
    let (re1, _im1) = test_vector_1::<F>(m);
    let sk = gen_sk(&params, module, host_module, [0u8; 32]);
    let mut scratch = alloc_scratch(&params, module);

    let half = F::from_f64(0.5).unwrap();
    let a_re = scaled(&re1, half);
    let a_im = vec![F::zero(); a_re.len()];

    let offset_f64 = ADD_SUB_CONST.0;
    let scale_f64 = MUL_CONST.0;
    let offset = quantized_const::<F>(offset_f64, 0.0, PT_PREC.log_delta()).0;
    let scale = quantized_const::<F>(scale_f64, 0.0, PT_PREC.log_delta()).0;
    let want_re: Vec<F> = a_re.iter().map(|x| *x * scale + offset).collect();
    let want_im: Vec<F> = vec![F::zero(); a_re.len()];

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
    let [offset_pt, scale_pt] = [offset_f64, scale_f64]
        .map(|value| ckks_pt_cst::<BE, F, Standard>(host_module, module, params.base2k.into(), PT_PREC, Some(value), None));
    let mut dst = alloc_ct(&params, module, params.k);

    with_scratch::<BE, _>(module.ckks_affine_pt_tmp_bytes(&dst, &a, &scale_pt), |s| {
        module.ckks_affine_pt_into(&mut dst, &a, &scale_pt, &offset_pt, s)
    })
    .unwrap();
    assert_decrypt_precision(
        "affine_pt_into_const_aligned",
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

pub fn test_affine_pt_into_const_zero_bias_matches_mul<BE, F, E>(
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
    let (re1, _im1) = test_vector_1::<F>(m);
    let sk = gen_sk(&params, module, host_module, [0u8; 32]);
    let mut scratch = alloc_scratch(&params, module);

    let half = F::from_f64(0.5).unwrap();
    let a_re = scaled(&re1, half);
    let a_im = vec![F::zero(); a_re.len()];

    let scale_f64 = MUL_CONST.0;
    let scale = quantized_const::<F>(scale_f64, 0.0, PT_PREC.log_delta()).0;
    let want_re: Vec<F> = a_re.iter().map(|x| *x * scale).collect();
    let want_im: Vec<F> = vec![F::zero(); a_re.len()];

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
    let [offset_pt, scale_pt] = [0.0, scale_f64]
        .map(|value| ckks_pt_cst::<BE, F, Standard>(host_module, module, params.base2k.into(), PT_PREC, Some(value), None));
    let mut dst = alloc_ct(&params, module, params.k);

    with_scratch::<BE, _>(module.ckks_affine_pt_tmp_bytes(&dst, &a, &scale_pt), |s| {
        module.ckks_affine_pt_into(&mut dst, &a, &scale_pt, &offset_pt, s)
    })
    .unwrap();

    assert_ct_meta(
        "affine_const_zero_bias",
        &dst,
        a.log_delta(),
        a.log_budget() - PT_PREC.log_delta(),
    );
    assert_decrypt_precision(
        "affine_const_zero_bias",
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

pub fn test_affine_pt_assign_const_aligned<BE, F, E>(
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
    let (re1, _im1) = test_vector_1::<F>(m);
    let sk = gen_sk(&params, module, host_module, [0u8; 32]);
    let mut scratch = alloc_scratch(&params, module);

    let half = F::from_f64(0.5).unwrap();
    let a_re = scaled(&re1, half);
    let a_im = vec![F::zero(); a_re.len()];

    let offset_f64 = ADD_SUB_CONST.0;
    let scale_f64 = MUL_CONST.0;
    let offset = quantized_const::<F>(offset_f64, 0.0, PT_PREC.log_delta()).0;
    let scale = quantized_const::<F>(scale_f64, 0.0, PT_PREC.log_delta()).0;
    let want_re: Vec<F> = a_re.iter().map(|x| *x * scale + offset).collect();
    let want_im: Vec<F> = vec![F::zero(); a_re.len()];

    let mut ct = ckks_encrypt(
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
    let [offset_pt, scale_pt] = [offset_f64, scale_f64]
        .map(|value| ckks_pt_cst::<BE, F, Standard>(host_module, module, params.base2k.into(), PT_PREC, Some(value), None));

    with_scratch::<BE, _>(module.ckks_affine_pt_tmp_bytes(&ct, &ct, &scale_pt), |s| {
        module.ckks_affine_pt_assign(&mut ct, &scale_pt, &offset_pt, s)
    })
    .unwrap();
    assert_decrypt_precision(
        "affine_pt_assign_const_aligned",
        &params,
        module,
        &encoder,
        &ct,
        &sk,
        &want_re,
        &want_im,
        &mut scratch.borrow(),
    );
}

pub fn test_affine_pt_into_aligned<BE, F, E>(params: CKKSTestParams, module: &Module<BE>, host_module: &Module<HostBytesBackend>)
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
    let (re1, _im1) = test_vector_1::<F>(m);
    let sk = gen_sk(&params, module, host_module, [0u8; 32]);
    let mut scratch = alloc_scratch(&params, module);

    let half = F::from_f64(0.5).unwrap();
    let a_re = scaled(&re1, half);
    let a_im = vec![F::zero(); a_re.len()];

    let offset_f64 = ADD_SUB_CONST.0;
    let scale_f64 = MUL_CONST.0;
    let offset = quantized_const::<F>(offset_f64, 0.0, PT_PREC.log_delta()).0;
    let scale = quantized_const::<F>(scale_f64, 0.0, PT_PREC.log_delta()).0;
    let want_re: Vec<F> = a_re.iter().map(|x| *x * scale + offset).collect();
    let want_im: Vec<F> = vec![F::zero(); a_re.len()];

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
    let scale_pt =
        ckks_pt_cst_full::<BE, F, Standard>(host_module, module, params.base2k.into(), PT_PREC, m, Some(scale_f64), None);
    let offset_pt =
        ckks_pt_cst_full::<BE, F, Standard>(host_module, module, params.base2k.into(), PT_PREC, m, Some(offset_f64), None);
    let mut dst = alloc_ct(&params, module, params.k);

    with_scratch::<BE, _>(module.ckks_affine_pt_tmp_bytes(&dst, &a, &scale_pt), |s| {
        module.ckks_affine_pt_into(&mut dst, &a, &scale_pt, &offset_pt, s)
    })
    .unwrap();
    assert_decrypt_precision(
        "affine_pt_into_aligned",
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

pub fn test_affine_pt_assign_aligned<BE, F, E>(
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
    let (re1, _im1) = test_vector_1::<F>(m);
    let sk = gen_sk(&params, module, host_module, [0u8; 32]);
    let mut scratch = alloc_scratch(&params, module);

    let half = F::from_f64(0.5).unwrap();
    let a_re = scaled(&re1, half);
    let a_im = vec![F::zero(); a_re.len()];

    let offset_f64 = ADD_SUB_CONST.0;
    let scale_f64 = MUL_CONST.0;
    let offset = quantized_const::<F>(offset_f64, 0.0, PT_PREC.log_delta()).0;
    let scale = quantized_const::<F>(scale_f64, 0.0, PT_PREC.log_delta()).0;
    let want_re: Vec<F> = a_re.iter().map(|x| *x * scale + offset).collect();
    let want_im: Vec<F> = vec![F::zero(); a_re.len()];

    let mut ct = ckks_encrypt(
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
    let scale_pt =
        ckks_pt_cst_full::<BE, F, Standard>(host_module, module, params.base2k.into(), PT_PREC, m, Some(scale_f64), None);
    let offset_pt =
        ckks_pt_cst_full::<BE, F, Standard>(host_module, module, params.base2k.into(), PT_PREC, m, Some(offset_f64), None);

    with_scratch::<BE, _>(module.ckks_affine_pt_tmp_bytes(&ct, &ct, &scale_pt), |s| {
        module.ckks_affine_pt_assign(&mut ct, &scale_pt, &offset_pt, s)
    })
    .unwrap();
    assert_decrypt_precision(
        "affine_pt_assign_aligned",
        &params,
        module,
        &encoder,
        &ct,
        &sk,
        &want_re,
        &want_im,
        &mut scratch.borrow(),
    );
}

/// An offset of another radix is `PlaintextBase2KMismatch` of the affine form, before the product writes `dst`, on an
/// empty arena and on the query, out of place and in place.
pub fn test_affine_pt_into_rejects_offset_base2k_mismatch<BE, F, E>(
    params: crate::test_suite::CKKSTestParams,
    module: &poulpy_hal::layouts::Module<BE>,
    _host_module: &poulpy_hal::layouts::Module<poulpy_hal::layouts::HostBytesBackend>,
) where
    BE: crate::test_suite::helpers::TestContextBackend,
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
    use crate::api::CKKSAffineOps;
    let a = fixture_ciphertext(module, &ct_layout, 41);
    let scale = fixture_plaintext(module, &pt_layout, 43);
    let offset = fixture_plaintext(module, &other_radix, 44);
    let mut res = fixture_ciphertext(module, &ct_layout, 31);
    let query = module.ckks_affine_pt_tmp_bytes(&res, &a, &scale);
    let want = |op| CKKSCompositionError::PlaintextBase2KMismatch {
        op,
        ct_base2k: b,
        pt_base2k: b - 1,
    };
    for bytes in [0, query] {
        let before = snapshot::<BE, _, Standard>(&res);
        let err = with_scratch::<BE, _>(bytes, |s| module.ckks_affine_pt_into(&mut res, &a, &scale, &offset, s)).unwrap_err();
        assert_ckks_error("affine_pt_into: offset of another radix", &err, want("ckks_affine_pt_into"));
        assert_eq!(snapshot::<BE, _, Standard>(&res), before, "a refused affine form wrote dst");
        let err = with_scratch::<BE, _>(bytes, |s| module.ckks_affine_pt_assign(&mut res, &scale, &offset, s)).unwrap_err();
        assert_ckks_error(
            "affine_pt_assign: offset of another radix",
            &err,
            want("ckks_affine_pt_assign"),
        );
        assert_eq!(snapshot::<BE, _, Standard>(&res), before, "a refused affine form wrote dst");
    }
}
