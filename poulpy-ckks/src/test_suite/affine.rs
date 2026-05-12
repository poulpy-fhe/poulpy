//! Tests for `CKKSAffineOps` — scalar `offset + scale * ct`.
//!
//! # Test inventory
//!
//! | Function | Path exercised |
//! |----------|----------------|
//! | [`test_affine_pt_const_into_aligned`] | scalar affine op with aligned output |
//! | [`test_affine_pt_const_zero_bias_matches_mul`] | zero offset preserves multiply result metadata |
//! | [`test_affine_pt_const_assign_aligned`] | assign variant of scalar affine op |
//! | [`test_affine_pt_vec_into_aligned`] | full-vector affine op into a fresh destination |
//! | [`test_affine_pt_vec_assign_aligned`] | full-vector affine assign in-place |

use poulpy_hal::{
    api::{NegacyclicFFT, NegacyclicFFTNew, ScratchOwnedBorrow},
    layouts::{HostBytesBackend, Module},
};

use crate::{
    CKKSInfos, CKKSMeta,
    layouts::{CKKSModuleAlloc, CKKSPlaintext, CKKSPlaintextVecHostCodec},
    leveled::api::CKKSAffineOps,
};

use super::helpers::{
    ADD_SUB_CONST, MUL_CONST, PT_PREC, TestContextBackend, TestContextModule, TestScalar, alloc_ct, alloc_scratch,
    assert_ct_meta, assert_decrypt_precision, ckks_encrypt, ckks_pt_cst_full, gen_sk, quantized_const, test_vector_1,
};

use crate::{encoding::reim::Encoder, test_suite::CKKSTestParams};

fn scaled<F: TestScalar>(v: &[F], scale: F) -> Vec<F> {
    v.iter().copied().map(|x| x * scale).collect()
}

fn encode_affine_const<F: TestScalar>(
    host_module: &Module<HostBytesBackend>,
    base2k: poulpy_core::layouts::Base2K,
    offset: F,
    scale: F,
    prec: CKKSMeta,
) -> CKKSPlaintext<Vec<u8>>
where
    Module<HostBytesBackend>: CKKSModuleAlloc<HostBytesBackend>,
{
    let mut pt = host_module.ckks_pt_coeffs_alloc(2, base2k, prec);
    pt.encode_host_floats(&[offset, scale]).unwrap();
    pt
}

pub fn test_affine_pt_const_into_aligned<BE, F, E>(
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
    let (re1, _im1) = test_vector_1::<F>(m);
    let sk = gen_sk(&params, module, host_module, [0u8; 32]);
    let mut scratch = alloc_scratch(&params, module);

    let half = F::from_f64(0.5).unwrap();
    let a_re = scaled(&re1, half);
    let a_im = vec![F::zero(); a_re.len()];

    let offset_f64 = ADD_SUB_CONST.0;
    let scale_f64 = MUL_CONST.0;
    let offset = quantized_const::<F>(offset_f64, 0.0, PT_PREC.log_delta).0;
    let scale = quantized_const::<F>(scale_f64, 0.0, PT_PREC.log_delta).0;
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
    let affine_const = encode_affine_const::<F>(host_module, params.base2k.into(), offset, scale, PT_PREC);
    let mut dst = alloc_ct(&params, module, params.k);

    module
        .ckks_affine_pt_const_into(&mut dst, &a, &affine_const, 0, 1, &mut scratch.borrow())
        .unwrap();
    assert_decrypt_precision(
        "affine_pt_const_into_aligned",
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

pub fn test_affine_pt_const_zero_bias_matches_mul<BE, F, E>(
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
    let (re1, _im1) = test_vector_1::<F>(m);
    let sk = gen_sk(&params, module, host_module, [0u8; 32]);
    let mut scratch = alloc_scratch(&params, module);

    let half = F::from_f64(0.5).unwrap();
    let a_re = scaled(&re1, half);
    let a_im = vec![F::zero(); a_re.len()];

    let scale_f64 = MUL_CONST.0;
    let scale = quantized_const::<F>(scale_f64, 0.0, PT_PREC.log_delta).0;
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
    let affine_const = encode_affine_const::<F>(host_module, params.base2k.into(), F::zero(), scale, PT_PREC);
    let mut dst = alloc_ct(&params, module, params.k);

    module
        .ckks_affine_pt_const_into(&mut dst, &a, &affine_const, 0, 1, &mut scratch.borrow())
        .unwrap();

    assert_ct_meta(
        "affine_pt_const_zero_bias",
        &dst,
        a.log_delta(),
        a.log_budget() - PT_PREC.log_delta,
    );
    assert_decrypt_precision(
        "affine_pt_const_zero_bias",
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

pub fn test_affine_pt_const_assign_aligned<BE, F, E>(
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
    let (re1, _im1) = test_vector_1::<F>(m);
    let sk = gen_sk(&params, module, host_module, [0u8; 32]);
    let mut scratch = alloc_scratch(&params, module);

    let half = F::from_f64(0.5).unwrap();
    let a_re = scaled(&re1, half);
    let a_im = vec![F::zero(); a_re.len()];

    let offset_f64 = ADD_SUB_CONST.0;
    let scale_f64 = MUL_CONST.0;
    let offset = quantized_const::<F>(offset_f64, 0.0, PT_PREC.log_delta).0;
    let scale = quantized_const::<F>(scale_f64, 0.0, PT_PREC.log_delta).0;
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
    let affine_const = encode_affine_const::<F>(host_module, params.base2k.into(), offset, scale, PT_PREC);

    module
        .ckks_affine_pt_const_assign(&mut ct, &affine_const, 0, 1, &mut scratch.borrow())
        .unwrap();
    assert_decrypt_precision(
        "affine_pt_const_assign_aligned",
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

pub fn test_affine_pt_vec_into_aligned<BE, F, E>(
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
    let (re1, _im1) = test_vector_1::<F>(m);
    let sk = gen_sk(&params, module, host_module, [0u8; 32]);
    let mut scratch = alloc_scratch(&params, module);

    let half = F::from_f64(0.5).unwrap();
    let a_re = scaled(&re1, half);
    let a_im = vec![F::zero(); a_re.len()];

    let offset_f64 = ADD_SUB_CONST.0;
    let scale_f64 = MUL_CONST.0;
    let offset = quantized_const::<F>(offset_f64, 0.0, PT_PREC.log_delta).0;
    let scale = quantized_const::<F>(scale_f64, 0.0, PT_PREC.log_delta).0;
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
    let scale_pt = ckks_pt_cst_full::<BE, F>(host_module, module, params.base2k.into(), PT_PREC, m, Some(scale_f64), None);
    let offset_pt = ckks_pt_cst_full::<BE, F>(host_module, module, params.base2k.into(), PT_PREC, m, Some(offset_f64), None);
    let mut dst = alloc_ct(&params, module, params.k);

    module
        .ckks_affine_pt_vec_into(&mut dst, &a, &scale_pt, &offset_pt, &mut scratch.borrow())
        .unwrap();
    assert_decrypt_precision(
        "affine_pt_vec_into_aligned",
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

pub fn test_affine_pt_vec_assign_aligned<BE, F, E>(
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
    let (re1, _im1) = test_vector_1::<F>(m);
    let sk = gen_sk(&params, module, host_module, [0u8; 32]);
    let mut scratch = alloc_scratch(&params, module);

    let half = F::from_f64(0.5).unwrap();
    let a_re = scaled(&re1, half);
    let a_im = vec![F::zero(); a_re.len()];

    let offset_f64 = ADD_SUB_CONST.0;
    let scale_f64 = MUL_CONST.0;
    let offset = quantized_const::<F>(offset_f64, 0.0, PT_PREC.log_delta).0;
    let scale = quantized_const::<F>(scale_f64, 0.0, PT_PREC.log_delta).0;
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
    let scale_pt = ckks_pt_cst_full::<BE, F>(host_module, module, params.base2k.into(), PT_PREC, m, Some(scale_f64), None);
    let offset_pt = ckks_pt_cst_full::<BE, F>(host_module, module, params.base2k.into(), PT_PREC, m, Some(offset_f64), None);

    module
        .ckks_affine_pt_vec_assign(&mut ct, &scale_pt, &offset_pt, &mut scratch.borrow())
        .unwrap();
    assert_decrypt_precision(
        "affine_pt_vec_assign_aligned",
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
