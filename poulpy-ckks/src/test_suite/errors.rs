use crate::{
    CKKSCompositionError, CKKSInfos, SetCKKSInfos,
    api::{CKKSAddOps, CKKSDotProductOps},
    layouts::CKKSModuleAlloc,
};
use poulpy_core::layouts::{Base2K, LWEInfos, TorusPrecision};
use poulpy_hal::{
    api::{NegacyclicFFT, NegacyclicFFTNew, ScratchOwnedBorrow},
    layouts::{HostBytesBackend, Module},
};

use super::helpers::{
    TestContextBackend, TestContextModule, TestScalar, alloc_scratch, assert_ckks_error, ckks_encrypt, gen_sk, gen_sk_with_raw,
    gen_tsk, test_vector_1,
};

use crate::{test_suite::CKKSTestParams, test_suite::reference_encoder::ReferenceEncoder};

pub fn test_add_pt_vec_alignment_error<BE, F, E>(
    params: CKKSTestParams,
    module: &Module<BE>,
    host_module: &Module<HostBytesBackend>,
) where
    BE: TestContextBackend,
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
    ct.set_log_budget(0);
    let mut pt = module.ckks_pt_vec_alloc(params.base2k.into(), params.prec().k());
    pt.set_meta(params.prec().meta());
    let err = module
        .ckks_add_pt_vec_assign(&mut ct, &pt, &mut scratch.borrow())
        .unwrap_err();
    assert_ckks_error(
        "add_pt_vec_alignment",
        &err,
        CKKSCompositionError::PlaintextAlignmentImpossible {
            op: "ckks_add_pt_vec",
            ct_log_budget: 0,
            pt_log_delta: params.prec().log_delta(),
            // Alignment is checked against the plaintext's effective (meaningful)
            // precision `log_delta + log_budget`, not its physical `max_k`.
            pt_k: params.prec().log_delta() + params.prec().log_budget(),
        },
    );
}

pub fn test_dot_product_overflow_guard<BE, F, E>(
    params: CKKSTestParams,
    module: &Module<BE>,
    host_module: &Module<HostBytesBackend>,
) where
    BE: TestContextBackend,
    for<'a> <BE as poulpy_hal::layouts::Backend>::BufRef<'a>: poulpy_hal::layouts::HostDataRef,
    for<'a> <BE as poulpy_hal::layouts::Backend>::BufMut<'a>: poulpy_hal::layouts::HostDataMut,
    Module<BE>: TestContextModule<BE>,
    F: TestScalar,
    E: NegacyclicFFT<F> + NegacyclicFFTNew<F>,
{
    let (sk_raw, _sk) = gen_sk_with_raw(&params, module, host_module, [0u8; 32]);
    let mut scratch = alloc_scratch(&params, module);

    let tsk = gen_tsk(&params, module, &sk_raw, &mut scratch.borrow());

    // Allocate ciphertexts with enormous base2k=63 to force the overflow guard.
    // The guard is metadata-only so data content does not matter.
    let mut dst = module.ckks_ciphertext_alloc(Base2K(63), TorusPrecision(64));
    dst.meta = params.prec().meta;
    let a = module.ckks_ciphertext_alloc(Base2K(63), TorusPrecision(64));
    let b = module.ckks_ciphertext_alloc(Base2K(63), TorusPrecision(64));
    let a_refs = vec![&a, &a];
    let b_refs = vec![&b, &b];
    let err = module
        .ckks_dot_product_ct(&mut dst, &a_refs, &b_refs, &tsk, &mut scratch.borrow())
        .unwrap_err()
        .to_string();
    assert!(
        err.contains("risks i64 overflow"),
        "dot_product_overflow_guard: unexpected error: {err}"
    );
}
