use crate::{
    CKKSCompositionError, CKKSInfos,
    layouts::{CKKSModuleAlloc, ciphertext::CKKSMaintainOps},
    leveled::api::{CKKSAddOps, CKKSDotProductOps},
};
use poulpy_core::layouts::{Base2K, LWEInfos, TorusPrecision};
use poulpy_hal::{
    api::{NegacyclicFFT, NegacyclicFFTNew, ScratchOwnedBorrow},
    layouts::{HostBackend, HostBytesBackend, Module},
};

use super::helpers::{
    TestContextBackend, TestContextModule, TestScalar, alloc_scratch, assert_ckks_error, assert_decrypt_precision, ckks_encrypt,
    gen_sk, gen_sk_with_raw, gen_tsk, test_vector_1,
};

use crate::{encoding::reim::Encoder, test_suite::CKKSTestParams};

pub fn test_reallocate_limbs_checked_error<BE, F, E>(
    params: CKKSTestParams,
    module: &Module<BE>,
    host_module: &Module<HostBytesBackend>,
) where
    BE: TestContextBackend + HostBackend<OwnedBuf = Vec<u8>>,
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
    let requested_limbs = ct.effective_k().div_ceil(ct.base2k().as_usize()).saturating_sub(1);
    let err = module.ckks_reallocate_limbs_checked(&mut ct, requested_limbs).unwrap_err();
    assert_ckks_error(
        "reallocate_limbs_checked",
        &err,
        CKKSCompositionError::LimbReallocationShrinksBelowMetadata {
            max_k: ct.max_k().as_usize(),
            log_delta: ct.log_delta(),
            base2k: ct.base2k().as_usize(),
            requested_limbs,
        },
    );
}

pub fn test_compact_limbs_copy<BE, F, E>(params: CKKSTestParams, module: &Module<BE>, host_module: &Module<HostBytesBackend>)
where
    BE: TestContextBackend + HostBackend<OwnedBuf = Vec<u8>>,
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
    let oversized_limbs = ct.size() + 1;
    module.ckks_reallocate_limbs_checked(&mut ct, oversized_limbs).unwrap();

    let compact = module.ckks_compact_limbs_copy(&ct).unwrap();
    let expected_limbs = ct.effective_k().div_ceil(ct.base2k().as_usize());

    assert_eq!(ct.size(), oversized_limbs, "source ciphertext should remain oversized");
    assert_eq!(compact.size(), expected_limbs, "compacted copy should drop excess limbs");
    assert_eq!(compact.meta(), ct.meta(), "compacted copy should preserve metadata");
    assert_eq!(compact.max_k().as_usize(), expected_limbs * ct.base2k().as_usize());

    assert_decrypt_precision(
        "compact_limbs_copy",
        &params,
        module,
        &encoder,
        &compact,
        &sk,
        &re1,
        &im1,
        &mut scratch.borrow(),
    );
}

pub fn test_add_pt_vec_alignment_error<BE, F, E>(
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
    ct.meta.log_budget = 0;
    let pt = module.ckks_pt_vec_alloc(params.base2k.into(), params.prec);
    let err = module
        .ckks_add_pt_vec_assign(&mut ct, &pt, &mut scratch.borrow())
        .unwrap_err();
    assert_ckks_error(
        "add_pt_vec_alignment",
        &err,
        CKKSCompositionError::PlaintextAlignmentImpossible {
            op: "ckks_add_pt_vec",
            ct_log_budget: 0,
            pt_log_delta: params.prec.log_delta,
            pt_k: pt.max_k().as_usize(),
        },
    );
}

pub fn test_dot_product_overflow_guard<BE, F, E>(
    params: CKKSTestParams,
    module: &Module<BE>,
    host_module: &Module<HostBytesBackend>,
) where
    BE: TestContextBackend,
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
    dst.meta = params.prec;
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
