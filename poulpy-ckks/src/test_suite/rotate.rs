//! Slot rotation tests (out-of-place and in-place).
//!
//! # Test inventory
//!
//! ## Operations-layer rotation (`GLWE<_, CKKS>::rotate`)
//!
//! | Function | Path exercised |
//! |----------|----------------|
//! | [`test_rotate_aligned`] | out-of-place rotation for each requested shift |
//!
//! ## Operations-layer rotation (`GLWE<_, CKKS>::rotate_assign`)
//!
//! | Function | Path exercised |
//! |----------|----------------|
//! | [`test_rotate_assign`] | in-place rotation for each requested shift |

use crate::{CKKSCompositionError, CKKSInfos, leveled::api::CKKSRotateOps};
use std::collections::HashMap;

use super::helpers::{
    TestContextBackend, TestContextModule, TestScalar, alloc_ct, alloc_scratch, assert_ckks_error, assert_ct_meta,
    assert_decrypt_precision, assert_unary_output_meta, ckks_encrypt, gen_atk, gen_sk_with_raw, test_vector_1, want_rotate,
};
use poulpy_core::layouts::GLWEAutomorphismKeyPrepared;
use poulpy_core::{GLWEAutomorphism, GLWEShift, oep::GLWEAutomorphismDefaults};
use poulpy_hal::{
    api::{NegacyclicFFT, NegacyclicFFTNew, ScratchAvailable, ScratchOwnedBorrow},
    layouts::{HostBytesBackend, Module, ScratchArena},
};

use crate::{encoding::reim::Encoder, test_suite::CKKSTestParams};

pub fn test_rotate_aligned<BE, F, E>(
    params: CKKSTestParams,
    module: &Module<BE>,
    host_module: &Module<HostBytesBackend>,
    rotations: &[i64],
) where
    BE: TestContextBackend,
    Module<BE>: TestContextModule<BE> + GLWEAutomorphismDefaults<BE> + GLWEAutomorphism<BE> + GLWEShift<BE> + CKKSRotateOps<BE>,
    F: TestScalar,
    E: NegacyclicFFT<F> + NegacyclicFFTNew<F>,
    for<'a> ScratchArena<'a, BE>: ScratchAvailable,
{
    let m = params.n / 2;
    let encoder = Encoder::<E>::new(m).unwrap();
    let (re1, im1) = test_vector_1::<F>(m);
    let (sk_raw, sk) = gen_sk_with_raw(&params, module, host_module, [0u8; 32]);
    let mut scratch = alloc_scratch(&params, module);

    let mut atks = HashMap::new();
    for &r in rotations {
        let atk = gen_atk(&params, module, r, &sk_raw, &mut scratch.borrow());
        atks.insert(r, atk);
    }

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
    for &r in rotations {
        let (want_re, want_im) = want_rotate(&re1, &im1, r, m);
        let mut ct_res = alloc_ct(&params, module, params.k);
        module
            .ckks_rotate_into(&mut ct_res, &ct, r, &atks, &mut scratch.borrow())
            .unwrap();
        assert_unary_output_meta(&format!("rotate({r})"), &ct_res, &ct);
        assert_decrypt_precision(
            &format!("rotate({r})"),
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
}

pub fn test_rotate_smaller_output<BE, F, E>(
    params: CKKSTestParams,
    module: &Module<BE>,
    host_module: &Module<HostBytesBackend>,
    rotations: &[i64],
) where
    BE: TestContextBackend,
    Module<BE>: TestContextModule<BE> + GLWEAutomorphismDefaults<BE> + GLWEAutomorphism<BE> + GLWEShift<BE> + CKKSRotateOps<BE>,
    F: TestScalar,
    E: NegacyclicFFT<F> + NegacyclicFFTNew<F>,
    for<'a> ScratchArena<'a, BE>: ScratchAvailable,
{
    let m = params.n / 2;
    let encoder = Encoder::<E>::new(m).unwrap();
    let (re1, im1) = test_vector_1::<F>(m);
    let (sk_raw, sk) = gen_sk_with_raw(&params, module, host_module, [0u8; 32]);
    let mut scratch = alloc_scratch(&params, module);

    let mut atks = HashMap::new();
    for &r in rotations {
        let atk = gen_atk(&params, module, r, &sk_raw, &mut scratch.borrow());
        atks.insert(r, atk);
    }

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
    for &r in rotations {
        let (want_re, want_im) = want_rotate(&re1, &im1, r, m);
        let mut ct_res = alloc_ct(&params, module, params.k - params.base2k - 1);
        module
            .ckks_rotate_into(&mut ct_res, &ct, r, &atks, &mut scratch.borrow())
            .unwrap();
        assert_unary_output_meta(&format!("rotate smaller_output({r})"), &ct_res, &ct);
        assert_decrypt_precision(
            &format!("rotate({r})"),
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
}

pub fn test_rotate_assign<BE, F, E>(
    params: CKKSTestParams,
    module: &Module<BE>,
    host_module: &Module<HostBytesBackend>,
    rotations: &[i64],
) where
    BE: TestContextBackend,
    Module<BE>: TestContextModule<BE> + GLWEAutomorphismDefaults<BE> + GLWEAutomorphism<BE> + GLWEShift<BE> + CKKSRotateOps<BE>,
    F: TestScalar,
    E: NegacyclicFFT<F> + NegacyclicFFTNew<F>,
    for<'a> ScratchArena<'a, BE>: ScratchAvailable,
{
    let m = params.n / 2;
    let encoder = Encoder::<E>::new(m).unwrap();
    let (re1, im1) = test_vector_1::<F>(m);
    let (sk_raw, sk) = gen_sk_with_raw(&params, module, host_module, [0u8; 32]);
    let mut scratch = alloc_scratch(&params, module);

    let mut atks = HashMap::new();
    for &r in rotations {
        let atk = gen_atk(&params, module, r, &sk_raw, &mut scratch.borrow());
        atks.insert(r, atk);
    }

    for &r in rotations {
        let (want_re, want_im) = want_rotate(&re1, &im1, r, m);
        let mut ct = ckks_encrypt(
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
        let expected_log_delta = ct.log_delta();
        let expected_log_budget = ct.log_budget();
        module.ckks_rotate_assign(&mut ct, r, &atks, &mut scratch.borrow()).unwrap();
        assert_ct_meta(&format!("rotate_assign({r})"), &ct, expected_log_delta, expected_log_budget);
        assert_decrypt_precision(
            &format!("rotate_assign({r})"),
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
}

pub fn test_rotate_assign_missing_key_error<BE, F, E>(
    params: CKKSTestParams,
    module: &Module<BE>,
    host_module: &Module<HostBytesBackend>,
) where
    BE: TestContextBackend,
    Module<BE>: TestContextModule<BE> + GLWEAutomorphismDefaults<BE> + GLWEAutomorphism<BE> + GLWEShift<BE> + CKKSRotateOps<BE>,
    F: TestScalar,
    E: NegacyclicFFT<F> + NegacyclicFFTNew<F>,
    for<'a> ScratchArena<'a, BE>: ScratchAvailable,
{
    let m = params.n / 2;
    let encoder = Encoder::<E>::new(m).unwrap();
    let (re1, im1) = test_vector_1::<F>(m);
    let sk = super::helpers::gen_sk(&params, module, host_module, [0u8; 32]);
    let mut scratch = alloc_scratch(&params, module);

    let mut ct = ckks_encrypt(
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
    let empty_keys: HashMap<i64, GLWEAutomorphismKeyPrepared<BE::OwnedBuf, BE>> = HashMap::new();
    let err = module
        .ckks_rotate_assign(&mut ct, 1, &empty_keys, &mut scratch.borrow())
        .unwrap_err();
    assert_ckks_error(
        "rotate_assign missing_key",
        &err,
        CKKSCompositionError::MissingAutomorphismKey {
            op: "rotate_assign",
            rotation: 1,
        },
    );
}
