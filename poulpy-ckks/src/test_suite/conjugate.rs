//! Conjugation tests (out-of-place and in-place).
//!
//! # Test inventory
//!
//! ## Operations-layer conjugation (`GLWE<_, CKKS>::conjugate`)
//!
//! | Function | Path exercised |
//! |----------|----------------|
//! | [`test_conjugate_aligned`] | out-of-place conjugation |
//! | [`test_conjugate_smaller_output`] | out-of-place conjugation |
//!
//! ## Operations-layer conjugation (`GLWE<_, CKKS>::conjugate_assign`)
//!
//! | Function | Path exercised |
//! |----------|----------------|
//! | [`test_conjugate_assign`] | in-place conjugation |

use crate::{CKKSInfos, leveled::api::CKKSConjugateOps};

use super::helpers::{
    TestContextBackend, TestContextModule, TestScalar, alloc_ct, alloc_scratch, assert_ct_meta, assert_decrypt_precision,
    assert_unary_output_meta, ckks_encrypt, gen_atk, gen_sk_with_raw, test_vector_1, want_conjugate,
};
use poulpy_core::{GLWEAutomorphism, GLWEShift, oep::GLWEAutomorphismDefaults};
use poulpy_hal::api::{NegacyclicFFT, NegacyclicFFTNew, ScratchAvailable, ScratchOwnedBorrow};
use poulpy_hal::layouts::{HostBytesBackend, Module, ScratchArena};

use crate::{encoding::reim::Encoder, test_suite::CKKSTestParams};

pub fn test_conjugate_aligned<BE, F, E>(params: CKKSTestParams, module: &Module<BE>, host_module: &Module<HostBytesBackend>)
where
    BE: TestContextBackend,
    Module<BE>:
        TestContextModule<BE> + GLWEAutomorphismDefaults<BE> + GLWEAutomorphism<BE> + GLWEShift<BE> + CKKSConjugateOps<BE>,
    F: TestScalar,
    E: NegacyclicFFT<F> + NegacyclicFFTNew<F>,
    for<'a> ScratchArena<'a, BE>: ScratchAvailable,
{
    let m = params.n / 2;
    let encoder = Encoder::<E>::new(m).unwrap();
    let (re1, im1) = test_vector_1::<F>(m);
    let (sk_raw, sk) = gen_sk_with_raw(&params, module, host_module, [0u8; 32]);
    let mut scratch = alloc_scratch(&params, module);
    let conj_key = gen_atk(&params, module, -1, &sk_raw, &mut scratch.borrow());

    let ct1 = ckks_encrypt(
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
    let (want_re, want_im) = want_conjugate(&re1, &im1);
    let mut ct_res = alloc_ct(&params, module, params.k);
    module
        .ckks_conjugate_into(&mut ct_res, &ct1, &conj_key, &mut scratch.borrow())
        .unwrap();
    assert_unary_output_meta("conjugate", &ct_res, &ct1);
    assert_decrypt_precision(
        "conjugate",
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

pub fn test_conjugate_smaller_output<BE, F, E>(
    params: CKKSTestParams,
    module: &Module<BE>,
    host_module: &Module<HostBytesBackend>,
) where
    BE: TestContextBackend,
    Module<BE>:
        TestContextModule<BE> + GLWEAutomorphismDefaults<BE> + GLWEAutomorphism<BE> + GLWEShift<BE> + CKKSConjugateOps<BE>,
    F: TestScalar,
    E: NegacyclicFFT<F> + NegacyclicFFTNew<F>,
    for<'a> ScratchArena<'a, BE>: ScratchAvailable,
{
    let m = params.n / 2;
    let encoder = Encoder::<E>::new(m).unwrap();
    let (re1, im1) = test_vector_1::<F>(m);
    let (sk_raw, sk) = gen_sk_with_raw(&params, module, host_module, [0u8; 32]);
    let mut scratch = alloc_scratch(&params, module);
    let conj_key = gen_atk(&params, module, -1, &sk_raw, &mut scratch.borrow());

    let ct1 = ckks_encrypt(
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
    let (want_re, want_im) = want_conjugate(&re1, &im1);
    let mut ct_res = alloc_ct(&params, module, params.k - params.base2k - 1);
    module
        .ckks_conjugate_into(&mut ct_res, &ct1, &conj_key, &mut scratch.borrow())
        .unwrap();
    assert_unary_output_meta("conjugate smaller_output", &ct_res, &ct1);
    assert_decrypt_precision(
        "conjugate",
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

pub fn test_conjugate_assign<BE, F, E>(params: CKKSTestParams, module: &Module<BE>, host_module: &Module<HostBytesBackend>)
where
    BE: TestContextBackend,
    Module<BE>:
        TestContextModule<BE> + GLWEAutomorphismDefaults<BE> + GLWEAutomorphism<BE> + GLWEShift<BE> + CKKSConjugateOps<BE>,
    F: TestScalar,
    E: NegacyclicFFT<F> + NegacyclicFFTNew<F>,
    for<'a> ScratchArena<'a, BE>: ScratchAvailable,
{
    let m = params.n / 2;
    let encoder = Encoder::<E>::new(m).unwrap();
    let (re1, im1) = test_vector_1::<F>(m);
    let (sk_raw, sk) = gen_sk_with_raw(&params, module, host_module, [0u8; 32]);
    let mut scratch = alloc_scratch(&params, module);
    let conj_key = gen_atk(&params, module, -1, &sk_raw, &mut scratch.borrow());

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
    let (want_re, want_im) = want_conjugate(&re1, &im1);
    let expected_log_delta = ct.log_delta();
    let expected_log_budget = ct.log_budget();
    module
        .ckks_conjugate_assign(&mut ct, &conj_key, &mut scratch.borrow())
        .unwrap();
    assert_ct_meta("conjugate_assign", &ct, expected_log_delta, expected_log_budget);
    assert_decrypt_precision(
        "conjugate_assign",
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
