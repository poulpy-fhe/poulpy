//! Value-preserving re-scaling of a ciphertext's working scale.

use crate::{CKKSCompositionError, CKKSInfos, CKKSMeta, api::CKKSScaleManage, leveled::api::CKKSMulOps};

use super::helpers::{
    TestContextBackend, TestContextModule, TestScalar, alloc_scratch, assert_ckks_error, assert_ct_meta,
    assert_decrypt_precision, assert_decrypt_precision_at_log_delta, assert_precision, ckks_decode_pt, ckks_decrypt_decode,
    ckks_decrypt_with_prec, ckks_encrypt, ckks_encrypt_with_prec, gen_sk, gen_sk_with_raw, gen_tsk, test_vector_1,
};
use poulpy_core::layouts::LWEInfos;
use poulpy_hal::{
    api::{NegacyclicFFT, NegacyclicFFTNew, ScratchOwnedBorrow},
    layouts::{HostBytesBackend, Module},
};

use crate::{encoding::reim::Encoder, test_suite::CKKSTestParams};

const SCALE_BITS: usize = 5;

pub fn test_scale_down_assign<BE, F, E>(params: CKKSTestParams, module: &Module<BE>, host_module: &Module<HostBytesBackend>)
where
    BE: TestContextBackend,
    Module<BE>: TestContextModule<BE>,
    F: TestScalar,
    E: NegacyclicFFT<F> + NegacyclicFFTNew<F>,
{
    let m = params.n / 2;
    let base2k = params.base2k;
    let encoder = Encoder::<E>::new(m).unwrap();
    let (re1, im1) = test_vector_1::<F>(m);
    let sk = gen_sk(&params, module, host_module, [0u8; 32]);
    let mut scratch = alloc_scratch(&params, module);

    // Small modulus so the full torus is readable in one i128 decode window (<=127).
    let k_small = (127 / base2k) * base2k;
    let prec_small = CKKSMeta {
        log_delta: 30,
        log_budget: 10,
        log_sparsity: 0,
    };
    let bits = 7usize;

    let mut ct = ckks_encrypt_with_prec::<BE, F, E>(
        &params,
        module,
        host_module,
        &encoder,
        &sk,
        k_small,
        &re1,
        &im1,
        prec_small,
        &mut scratch.borrow(),
    );
    let (ld, max_k) = (ct.log_delta(), ct.max_k().as_usize());
    let expected_log_delta = ld - bits;
    let expected_log_budget = ct.log_budget() + bits;
    module.ckks_scale_down_assign(&mut ct, bits, &mut scratch.borrow()).unwrap();
    assert_ct_meta("scale_down_assign", &ct, expected_log_delta, expected_log_budget);

    // Not value-preserving over the full modulus: a full-range decode shows the
    // overflow left in the top `bits`.
    let pt_full = ckks_decrypt_with_prec(
        module,
        &ct,
        &sk,
        CKKSMeta {
            log_delta: ld - bits,
            log_budget: max_k - (ld - bits),
            log_sparsity: 0,
        },
        &mut scratch.borrow(),
    )
    .unwrap();
    let (re_full, im_full): (Vec<F>, Vec<F>) = ckks_decode_pt(&encoder, m, &pt_full);
    let max_int = re_full
        .iter()
        .chain(im_full.iter())
        .map(|v| v.to_f64().unwrap().round().abs())
        .fold(0.0f64, f64::max);
    assert!(
        max_int > 1.0,
        "scale_down_assign: expected a high-bit overflow over the full modulus, found none"
    );

    // Only the message region is preserved: a window one limb below the top recovers `m`.
    let safe_budget = (max_k - base2k) - (ld - bits);
    let pt_msg = ckks_decrypt_with_prec(
        module,
        &ct,
        &sk,
        CKKSMeta {
            log_delta: ld - bits,
            log_budget: safe_budget,
            log_sparsity: 0,
        },
        &mut scratch.borrow(),
    )
    .unwrap();
    let (re_msg, im_msg): (Vec<F>, Vec<F>) = ckks_decode_pt(&encoder, m, &pt_msg);
    assert_precision("scale_down_assign message preserved (re)", &re_msg, &re1, ld - bits, params.n);
    assert_precision("scale_down_assign message preserved (im)", &im_msg, &im1, ld - bits, params.n);
}

/// A *single* multiply after `scale_down` keeps the message clean, for both
/// `2·bits ≤ log_delta` and `2·bits > log_delta` (no `2·bits` requirement). This
/// only pins the single-multiply case: the overflow sits at the top of the live
/// range and each further multiply rescales it down toward the message, so deeper
/// chains are not covered here.
pub fn test_scale_down_then_multiply<BE, F, E>(
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
    let (sk_raw, sk) = gen_sk_with_raw(&params, module, host_module, [0u8; 32]);
    let mut scratch = alloc_scratch(&params, module);
    let tsk = gen_tsk(&params, module, &sk_raw, &mut scratch.borrow());

    let want_re: Vec<F> = (0..m).map(|j| re1[j] * re1[j] - im1[j] * im1[j]).collect();
    let want_im: Vec<F> = (0..m).map(|j| F::from_f64(2.0).unwrap() * re1[j] * im1[j]).collect();

    let ld = params.prec.log_delta;
    // ld/4: 2·bits ≤ ld ;  ld/2+1: 2·bits > ld .
    for &bits in &[ld / 4, ld / 2 + 1] {
        assert!(bits < ld, "test setup");
        let mut ct = ckks_encrypt_with_prec::<BE, F, E>(
            &params,
            module,
            host_module,
            &encoder,
            &sk,
            params.k,
            &re1,
            &im1,
            params.prec,
            &mut scratch.borrow(),
        );
        module.ckks_scale_down_assign(&mut ct, bits, &mut scratch.borrow()).unwrap();
        module.ckks_square_assign(&mut ct, &tsk, &mut scratch.borrow()).unwrap();
        let (re_sq, im_sq) = ckks_decrypt_decode::<BE, F, E>(&params, module, &encoder, &ct, &sk, &mut scratch.borrow());
        let err = (0..m)
            .map(|j| {
                (re_sq[j] - want_re[j])
                    .to_f64()
                    .unwrap()
                    .abs()
                    .max((im_sq[j] - want_im[j]).to_f64().unwrap().abs())
            })
            .fold(0.0f64, f64::max);
        assert!(
            err < 0.1,
            "bits={bits} (log_delta={ld}): square err {err:.3e} — multiply must not leak the overflow into the message"
        );
    }
}

pub fn test_scale_up_assign<BE, F, E>(params: CKKSTestParams, module: &Module<BE>, host_module: &Module<HostBytesBackend>)
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
    let original_log_delta = ct.log_delta();
    let expected_log_budget = ct.log_budget() - SCALE_BITS;
    module
        .ckks_scale_up_assign(&mut ct, SCALE_BITS, &mut scratch.borrow())
        .unwrap();
    assert_ct_meta("scale_up_assign", &ct, original_log_delta + SCALE_BITS, expected_log_budget);
    // The value is preserved: the shifted-in bits are zero, so the recovered
    // precision is still the original log_delta, not the inflated one.
    assert_decrypt_precision_at_log_delta(
        "scale_up_assign",
        &params,
        module,
        &encoder,
        &ct,
        &sk,
        &re1,
        &im1,
        original_log_delta,
        &mut scratch.borrow(),
    );
}

pub fn test_scale_round_trip<BE, F, E>(params: CKKSTestParams, module: &Module<BE>, host_module: &Module<HostBytesBackend>)
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
    let original_log_delta = ct.log_delta();
    let original_log_budget = ct.log_budget();
    // Scaling up then back down restores both the metadata and the value.
    module
        .ckks_scale_up_assign(&mut ct, SCALE_BITS, &mut scratch.borrow())
        .unwrap();
    module
        .ckks_scale_down_assign(&mut ct, SCALE_BITS, &mut scratch.borrow())
        .unwrap();
    assert_ct_meta("scale_round_trip", &ct, original_log_delta, original_log_budget);
    assert_decrypt_precision(
        "scale_round_trip",
        &params,
        module,
        &encoder,
        &ct,
        &sk,
        &re1,
        &im1,
        &mut scratch.borrow(),
    );
}

pub fn test_scale_up_insufficient_budget_error<BE, F, E>(
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
    let available_log_budget = ct.log_budget();
    let required_bits = available_log_budget + 1;
    let err = module
        .ckks_scale_up_assign(&mut ct, required_bits, &mut scratch.borrow())
        .unwrap_err();
    assert_ckks_error(
        "scale_up_insufficient_budget_error",
        &err,
        CKKSCompositionError::InsufficientHomomorphicCapacity {
            op: "scale_up",
            available_log_budget,
            required_bits,
        },
    );
}

pub fn test_scale_down_insufficient_precision_error<BE, F, E>(
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
    let available_log_delta = ct.log_delta();
    let required_bits = available_log_delta + 1;
    let err = module
        .ckks_scale_down_assign(&mut ct, required_bits, &mut scratch.borrow())
        .unwrap_err();
    assert_ckks_error(
        "scale_down_insufficient_precision_error",
        &err,
        CKKSCompositionError::InsufficientScalePrecision {
            op: "scale_down",
            available_log_delta,
            required_bits,
        },
    );
}
