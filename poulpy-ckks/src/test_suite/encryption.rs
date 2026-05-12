//! Encrypt / decrypt round-trip tests.

use super::helpers::{
    TestContextBackend, TestContextModule, TestScalar, alloc_scratch, assert_ckks_error, assert_ct_meta,
    assert_precision_for_log_delta, ckks_decrypt_decode, ckks_decrypt_with_prec, ckks_encrypt, ckks_encrypt_with_prec, gen_sk,
    quantized_slots, test_vector_1,
};
use crate::{CKKSCompositionError, CKKSInfos, CKKSMeta, layouts::CKKSModuleAlloc, leveled::api::CKKSDecrypt};
use poulpy_hal::{
    api::{NegacyclicFFT, NegacyclicFFTNew, ScratchOwnedBorrow},
    layouts::{HostBytesBackend, Module},
};

use crate::{encoding::reim::Encoder, test_suite::CKKSTestParams};

fn extract_src_prec(params: &CKKSTestParams) -> CKKSMeta {
    if params.base2k == 19 {
        CKKSMeta {
            log_delta: 40,
            log_budget: 17,
        }
    } else {
        CKKSMeta {
            log_delta: 40,
            log_budget: 12,
        }
    }
}

fn assert_decrypt_extract_success<BE, F, E>(
    label: &str,
    params: &CKKSTestParams,
    module: &Module<BE>,
    host_module: &Module<HostBytesBackend>,
    encoder: &Encoder<E>,
    dst_prec: CKKSMeta,
) where
    BE: TestContextBackend,
    Module<BE>: TestContextModule<BE>,
    F: TestScalar,
    E: NegacyclicFFT<F> + NegacyclicFFTNew<F>,
    Module<BE>: CKKSDecrypt<BE>,
{
    let src_prec = extract_src_prec(params);
    let sk = gen_sk(params, module, host_module, [0u8; 32]);
    let mut scratch = alloc_scratch(params, module);
    let m = params.n / 2;

    let (re1, im1) = test_vector_1::<F>(m);
    let ct = ckks_encrypt_with_prec(
        params,
        module,
        host_module,
        encoder,
        &sk,
        src_prec.effective_k(),
        &re1,
        &im1,
        src_prec,
        &mut scratch.borrow(),
    );
    assert_ct_meta(&format!("{label} src"), &ct, src_prec.log_delta, src_prec.log_budget);

    let pt = ckks_decrypt_with_prec(module, &ct, &sk, dst_prec, &mut scratch.borrow()).unwrap();
    assert_eq!(pt.meta, dst_prec, "{label}: decrypt changed destination metadata");

    let mut re_out = vec![F::zero(); m];
    let mut im_out = vec![F::zero(); m];
    encoder.decode_reim(&pt, &mut re_out, &mut im_out).unwrap();

    let (want_prec, assert_log_delta) = if dst_prec.log_delta > src_prec.log_delta {
        (src_prec, src_prec.log_delta)
    } else {
        (dst_prec, dst_prec.log_delta)
    };
    let (want_re, want_im) = quantized_slots(host_module, encoder, params.base2k.into(), want_prec, &re1, &im1);
    assert_precision_for_log_delta(&format!("{label} re"), &re_out, &want_re, assert_log_delta, params.n);
    assert_precision_for_log_delta(&format!("{label} im"), &im_out, &want_im, assert_log_delta, params.n);
}

/// Verifies that encrypt → decrypt → decode recovers the original message.
pub fn test_encrypt_decrypt<BE, F, E>(params: CKKSTestParams, module: &Module<BE>, host_module: &Module<HostBytesBackend>)
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
    assert_ct_meta(
        "encrypt_decrypt",
        &ct,
        params.prec.log_delta,
        params.k - params.prec.log_delta,
    );
    let (re_out, im_out) = ckks_decrypt_decode::<BE, F, E>(&params, module, &encoder, &ct, &sk, &mut scratch.borrow());
    assert_precision_for_log_delta("encrypt_decrypt re", &re_out, &re1, ct.log_delta(), params.n);
    assert_precision_for_log_delta("encrypt_decrypt im", &im_out, &im1, ct.log_delta(), params.n);
}

pub fn test_decrypt_extract_same_meta<BE, F, E>(
    params: CKKSTestParams,
    module: &Module<BE>,
    host_module: &Module<HostBytesBackend>,
) where
    BE: TestContextBackend,
    Module<BE>: TestContextModule<BE>,
    F: TestScalar,
    E: NegacyclicFFT<F> + NegacyclicFFTNew<F>,
    Module<BE>: CKKSDecrypt<BE>,
{
    let m = params.n / 2;
    let encoder = Encoder::<E>::new(m).unwrap();
    assert_decrypt_extract_success(
        "decrypt_extract_same_meta",
        &params,
        module,
        host_module,
        &encoder,
        extract_src_prec(&params),
    );
}

pub fn test_decrypt_extract_truncates_log_budget<BE, F, E>(
    params: CKKSTestParams,
    module: &Module<BE>,
    host_module: &Module<HostBytesBackend>,
) where
    BE: TestContextBackend,
    Module<BE>: TestContextModule<BE>,
    F: TestScalar,
    E: NegacyclicFFT<F> + NegacyclicFFTNew<F>,
    Module<BE>: CKKSDecrypt<BE>,
{
    let m = params.n / 2;
    let encoder = Encoder::<E>::new(m).unwrap();
    let src_prec = extract_src_prec(&params);
    assert_decrypt_extract_success(
        "decrypt_extract_truncates_log_budget",
        &params,
        module,
        host_module,
        &encoder,
        CKKSMeta {
            log_delta: src_prec.log_delta,
            log_budget: 0,
        },
    );
}

pub fn test_decrypt_extract_rsh_for_smaller_log_delta<BE, F, E>(
    params: CKKSTestParams,
    module: &Module<BE>,
    host_module: &Module<HostBytesBackend>,
) where
    BE: TestContextBackend,
    Module<BE>: TestContextModule<BE>,
    F: TestScalar,
    E: NegacyclicFFT<F> + NegacyclicFFTNew<F>,
    Module<BE>: CKKSDecrypt<BE>,
{
    let m = params.n / 2;
    let encoder = Encoder::<E>::new(m).unwrap();
    let src_prec = extract_src_prec(&params);
    assert_decrypt_extract_success(
        "decrypt_extract_rsh",
        &params,
        module,
        host_module,
        &encoder,
        CKKSMeta {
            log_delta: src_prec.log_delta - 8,
            log_budget: src_prec.log_budget,
        },
    );
}

pub fn test_decrypt_extract_lsh_for_larger_log_delta<BE, F, E>(
    params: CKKSTestParams,
    module: &Module<BE>,
    host_module: &Module<HostBytesBackend>,
) where
    BE: TestContextBackend,
    Module<BE>: TestContextModule<BE>,
    F: TestScalar,
    E: NegacyclicFFT<F> + NegacyclicFFTNew<F>,
    Module<BE>: CKKSDecrypt<BE>,
{
    let m = params.n / 2;
    let encoder = Encoder::<E>::new(m).unwrap();
    let src_prec = extract_src_prec(&params);
    assert_decrypt_extract_success(
        "decrypt_extract_lsh",
        &params,
        module,
        host_module,
        &encoder,
        CKKSMeta {
            log_delta: src_prec.log_delta,
            log_budget: src_prec.log_budget - 8,
        },
    );
}

pub fn test_decrypt_extract_output_hom_rem_too_large<BE, F, E>(
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
    let src_prec = extract_src_prec(&params);
    let sk = gen_sk(&params, module, host_module, [0u8; 32]);
    let mut scratch = alloc_scratch(&params, module);

    let (re1, im1) = test_vector_1::<F>(m);
    let ct = ckks_encrypt_with_prec(
        &params,
        module,
        host_module,
        &encoder,
        &sk,
        src_prec.effective_k(),
        &re1,
        &im1,
        src_prec,
        &mut scratch.borrow(),
    );
    let mut pt = module.ckks_pt_vec_alloc(
        params.base2k.into(),
        CKKSMeta {
            log_delta: src_prec.log_delta,
            log_budget: src_prec.log_budget + 1,
        },
    );
    let err = module.ckks_decrypt(&mut pt, &ct, &sk, &mut scratch.borrow()).unwrap_err();
    assert_ckks_error(
        "decrypt_extract_output_hom_rem_too_large",
        &err,
        CKKSCompositionError::PlaintextAlignmentImpossible {
            op: "ckks_extract_pt",
            ct_log_budget: src_prec.log_budget,
            pt_log_delta: src_prec.log_delta,
            pt_k: pt.effective_k(),
        },
    );
}

pub fn test_decrypt_extract_base2k_mismatch_error<BE, F, E>(
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
    let src_prec = extract_src_prec(&params);
    let sk = gen_sk(&params, module, host_module, [0u8; 32]);
    let mut scratch = alloc_scratch(&params, module);

    let (re1, im1) = test_vector_1::<F>(m);
    let ct = ckks_encrypt_with_prec(
        &params,
        module,
        host_module,
        &encoder,
        &sk,
        src_prec.effective_k(),
        &re1,
        &im1,
        src_prec,
        &mut scratch.borrow(),
    );
    let mismatched_base2k = (params.base2k / 2).into();
    let mut pt = host_module.ckks_pt_vec_alloc(mismatched_base2k, src_prec);
    let err = module.ckks_decrypt(&mut pt, &ct, &sk, &mut scratch.borrow()).unwrap_err();
    assert_ckks_error(
        "decrypt_extract_base2k_mismatch",
        &err,
        CKKSCompositionError::PlaintextBase2KMismatch {
            op: "ckks_extract_pt",
            ct_base2k: params.base2k,
            pt_base2k: mismatched_base2k.as_usize(),
        },
    );
}
