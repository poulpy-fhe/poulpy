//! Reim encoder encode/decode round-trip test.

use crate::CKKSModuleInfos;
use crate::Scale;
use poulpy_core::layouts::Base2K;
use poulpy_hal::{
    api::{NegacyclicFFT, NegacyclicFFTNew},
    layouts::{HostBytesBackend, Module, Standard},
};

use crate::SlotsKind;
use crate::{
    CKKSInfos, GLWEPlaintextMeta, SetCKKSInfos,
    layouts::CKKSModuleAlloc,
    test_suite::reference_encoder::ReferenceEncoder,
    test_suite::{
        CKKSTestParams,
        helpers::{TestContextBackend, TestScalar, assert_precision_for_log_delta, test_vector_1},
    },
};

/// The reim [`ReferenceEncoder`] is its own inverse: `decode_reim(encode_reim(re, im))`
/// recovers `(re, im)` up to the encoding (`log_delta`) quantization, with no
/// encryption involved. Backend-generic over the scalar `F` and FFT engine `E`.
pub fn test_encode_decode_reim_roundtrip<BE, F, E>(
    params: CKKSTestParams,
    _module: &Module<BE>,
    host_module: &Module<HostBytesBackend>,
) where
    BE: TestContextBackend<Ring = Standard>,
    for<'a> <BE as poulpy_hal::layouts::Backend>::BufRef<'a>: poulpy_hal::layouts::HostDataRef,
    for<'a> <BE as poulpy_hal::layouts::Backend>::BufMut<'a>: poulpy_hal::layouts::HostDataMut,
    F: TestScalar,
    E: NegacyclicFFT<F> + NegacyclicFFTNew<F>,
{
    let m = params.n / 2;
    let log_delta = params.prec().log_delta();
    let encoder = ReferenceEncoder::<E>::new::<F>(m).unwrap();

    let (re_in, im_in) = test_vector_1::<F>(m);

    let mut pt = host_module.ckks_pt_vec_alloc(
        host_module.ckks_max_slots(),
        Base2K(params.base2k as u32),
        poulpy_core::layouts::TorusPrecision((log_delta + 10) as u32),
    );
    pt.set_meta(GLWEPlaintextMeta {
        scale: Scale::Log(log_delta),
        slots: SlotsKind::Complex,
        log_sparsity: 0,
    });
    encoder.encode_reim(&mut pt, &re_in, &im_in).unwrap();

    let mut re_out = vec![F::from_f64(0.0).unwrap(); m];
    let mut im_out = vec![F::from_f64(0.0).unwrap(); m];
    encoder.decode_reim(&pt, &mut re_out, &mut im_out).unwrap();

    assert_precision_for_log_delta("encode_decode_reim re", &re_out, &re_in, log_delta, params.n);
    assert_precision_for_log_delta("encode_decode_reim im", &im_out, &im_in, log_delta, params.n);
}

/// A coefficient buffer of another length than the plaintext's degree is `InvalidOperand` of `ckks_encode_coeffs_into`
/// before any write.
pub fn test_encode_coeffs_into_rejects_coefficient_count<BE, F, E>(
    params: crate::test_suite::CKKSTestParams,
    module: &poulpy_hal::layouts::Module<BE>,
    host_module: &poulpy_hal::layouts::Module<poulpy_hal::layouts::HostBytesBackend>,
) where
    BE: crate::test_suite::helpers::TestContextBackend<Ring = poulpy_hal::layouts::Standard>,
    poulpy_hal::layouts::Module<BE>: crate::test_suite::helpers::TestContextModule<BE> + crate::api::CKKSEncodingOps<BE, F>,
    F: crate::test_suite::helpers::TestScalar,
    E: poulpy_hal::api::NegacyclicFFT<F> + poulpy_hal::api::NegacyclicFFTNew<F>,
{
    use crate::test_suite::{
        helpers::assert_ckks_error,
        parity::{
            helpers::{fixture_plaintext, snapshot},
            layout,
        },
    };
    use crate::{CKKSCompositionError, SlotsKind, api::CKKSEncodingOps, layouts::CKKSEncodingBuffer};
    use poulpy_core::layouts::LWEInfos;
    let _ = host_module;
    let b = params.base2k;
    let mut pt = fixture_plaintext(module, &layout(params, 0, b + 3, b - 1, 0, SlotsKind::Complex), 43);
    let n = pt.n().as_usize();
    let coeffs = CKKSEncodingBuffer::<BE::OwnedBuf, F>::from_host::<BE>(&vec![F::from_f64(0.0).unwrap(); n - 1]);
    let before = snapshot::<BE, _>(&pt);
    let err = module.ckks_encode_coeffs_into(&mut pt, &coeffs).unwrap_err();
    assert_ckks_error(
        "encode_coeffs_into: one coefficient short",
        &err,
        CKKSCompositionError::InvalidOperand {
            op: "ckks_encode_coeffs_into",
            reason: format!("coefficient count {} must equal plaintext degree {n}", n - 1),
        },
    );
    assert_eq!(snapshot::<BE, _>(&pt), before, "a refused encoding wrote the plaintext");
}

/// The host slot adapters refuse real and imaginary slices of two lengths, and a slot count that is not a nonzero
/// power of two, as `InvalidOperand` of the invoked adapter on an empty arena, the plaintext and slices untouched.
pub fn test_encode_reim_into_rejects_slot_counts<BE, F, E>(
    params: crate::test_suite::CKKSTestParams,
    module: &poulpy_hal::layouts::Module<BE>,
    host_module: &poulpy_hal::layouts::Module<poulpy_hal::layouts::HostBytesBackend>,
) where
    BE: crate::test_suite::helpers::TestContextBackend<Ring = poulpy_hal::layouts::Standard>,
    poulpy_hal::layouts::Module<BE>: crate::test_suite::helpers::TestContextModule<BE> + crate::api::CKKSEncodingOps<BE, F>,
    F: crate::test_suite::helpers::TestScalar,
    E: poulpy_hal::api::NegacyclicFFT<F> + poulpy_hal::api::NegacyclicFFTNew<F>,
{
    use crate::test_suite::{
        helpers::assert_ckks_error,
        parity::{
            helpers::{fixture_plaintext, snapshot, with_scratch},
            layout,
        },
    };
    use crate::{CKKSCompositionError, SlotsKind, api::CKKSEncodingHostOps};
    let _ = host_module;
    let b = params.base2k;
    let mut pt = fixture_plaintext(module, &layout(params, 0, b + 3, b - 1, 0, SlotsKind::Complex), 47);
    let value = |i: usize| F::from_f64(i as f64 / 64.0).unwrap();
    for (case, re_len, im_len, reason) in [
        ("two lengths", 4, 2, "real and imaginary slot counts differ".to_string()),
        ("no slot", 0, 0, "slot count must be a non-zero power of two, got 0".into()),
        (
            "three slots",
            3,
            3,
            "slot count must be a non-zero power of two, got 3".into(),
        ),
    ] {
        let (mut re, mut im): (Vec<F>, Vec<F>) = ((0..re_len).map(value).collect(), (0..im_len).map(value).collect());
        let before = (snapshot::<BE, _>(&pt), re.clone(), im.clone());
        let err = with_scratch::<BE, _>(0, |s| module.ckks_encode_reim_into(&mut pt, &re, &im, s)).unwrap_err();
        assert_ckks_error(
            &format!("encode_reim_into: {case}"),
            &err,
            CKKSCompositionError::InvalidOperand {
                op: "ckks_encode_reim_into",
                reason: reason.clone(),
            },
        );
        let err = with_scratch::<BE, _>(0, |s| module.ckks_decode_reim_into(&pt, &mut re, &mut im, s)).unwrap_err();
        assert_ckks_error(
            &format!("decode_reim_into: {case}"),
            &err,
            CKKSCompositionError::InvalidOperand {
                op: "ckks_decode_reim_into",
                reason,
            },
        );
        assert_eq!(
            (snapshot::<BE, _>(&pt), re, im),
            before,
            "{case}: a refused adapter wrote the plaintext or slices"
        );
    }
}
