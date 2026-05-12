//! Negation tests (out-of-place and in-place).
//!
//! # Test inventory
//!
//! ## Operations-layer negation (`GLWE<_, CKKS>::neg`)
//!
//! | Function | Path exercised |
//! |----------|----------------|
//! | [`test_neg_aligned`] | out-of-place negation |
//! | [`test_neg_smaller_output`] | out-of-place negation into a smaller output buffer |
//!
//! ## Operations-layer negation (`GLWE<_, CKKS>::neg_assign`)
//!
//! | Function | Path exercised |
//! |----------|----------------|
//! | [`test_neg_assign`] | in-place negation |

use crate::{CKKSInfos, leveled::api::CKKSNegOps};

use super::helpers::{
    TestContextBackend, TestContextModule, TestScalar, alloc_ct, alloc_scratch, assert_ct_meta, assert_decrypt_precision,
    assert_unary_output_meta, ckks_encrypt, gen_sk, test_vector_1, want_neg,
};
use anyhow::Result;
use poulpy_hal::{
    api::{NegacyclicFFT, NegacyclicFFTNew, ScratchOwnedBorrow},
    layouts::{HostBytesBackend, Module},
};

use crate::{encoding::reim::Encoder, test_suite::CKKSTestParams};

// ─── negation out-of-place (GLWE<_, CKKS>::neg) ────────────────────────────

/// Negation out-of-place.
pub fn test_neg_aligned<BE, F, E>(
    params: CKKSTestParams,
    module: &Module<BE>,
    host_module: &Module<HostBytesBackend>,
) -> Result<()>
where
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
    let (want_re, want_im) = want_neg(&re1, &im1);
    let mut ct_res = alloc_ct(&params, module, params.k);
    module.ckks_neg_into(&mut ct_res, &ct1, &mut scratch.borrow())?;
    assert_unary_output_meta("neg", &ct_res, &ct1);
    assert_decrypt_precision(
        "neg",
        &params,
        module,
        &encoder,
        &ct_res,
        &sk,
        &want_re,
        &want_im,
        &mut scratch.borrow(),
    );
    Ok(())
}

/// Negation out-of-place into a smaller output buffer.
pub fn test_neg_smaller_output<BE, F, E>(
    params: CKKSTestParams,
    module: &Module<BE>,
    host_module: &Module<HostBytesBackend>,
) -> Result<()>
where
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
    let (want_re, want_im) = want_neg(&re1, &im1);
    let mut ct_res = alloc_ct(&params, module, params.k - params.base2k - 1);
    module.ckks_neg_into(&mut ct_res, &ct1, &mut scratch.borrow())?;
    assert_unary_output_meta("neg smaller_output", &ct_res, &ct1);
    assert_decrypt_precision(
        "neg",
        &params,
        module,
        &encoder,
        &ct_res,
        &sk,
        &want_re,
        &want_im,
        &mut scratch.borrow(),
    );
    Ok(())
}

// ─── negation in-place (GLWE<_, CKKS>::neg_assign) ────────────────────────

/// Negation in-place.
pub fn test_neg_assign<BE, F, E>(
    params: CKKSTestParams,
    module: &Module<BE>,
    host_module: &Module<HostBytesBackend>,
) -> Result<()>
where
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
    let (want_re, want_im) = want_neg(&re1, &im1);
    let expected_log_delta = ct.log_delta();
    let expected_log_budget = ct.log_budget();
    module.ckks_neg_assign(&mut ct)?;
    assert_ct_meta("neg_assign", &ct, expected_log_delta, expected_log_budget);
    assert_decrypt_precision(
        "neg_assign",
        &params,
        module,
        &encoder,
        &ct,
        &sk,
        &want_re,
        &want_im,
        &mut scratch.borrow(),
    );
    Ok(())
}
