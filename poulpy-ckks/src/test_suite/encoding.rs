//! Reim encoder encode/decode round-trip test.

use poulpy_core::layouts::Base2K;
use poulpy_hal::{
    api::{NegacyclicFFT, NegacyclicFFTNew},
    layouts::{HostBytesBackend, Module},
};

use crate::{
    CKKSInfos, CKKSMeta, SetCKKSInfos,
    layouts::CKKSModuleAlloc,
    test_suite::reference_encoder::ReferenceEncoder,
    test_suite::{
        CKKSTestParams,
        helpers::{TestContextBackend, TestContextHostModule, TestScalar, assert_precision_for_log_delta, test_vector_1},
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
    BE: TestContextBackend,
    for<'a> <BE as poulpy_hal::layouts::Backend>::BufRef<'a>: poulpy_hal::layouts::HostDataRef,
    for<'a> <BE as poulpy_hal::layouts::Backend>::BufMut<'a>: poulpy_hal::layouts::HostDataMut,
    F: TestScalar,
    E: NegacyclicFFT<F> + NegacyclicFFTNew<F>,
    Module<HostBytesBackend>: TestContextHostModule,
{
    let m = params.n / 2;
    let log_delta = params.prec().log_delta();
    let encoder = ReferenceEncoder::<E>::new::<F>(m).unwrap();

    let (re_in, im_in) = test_vector_1::<F>(m);

    let mut pt = host_module.ckks_pt_vec_alloc(
        Base2K(params.base2k as u32),
        poulpy_core::layouts::TorusPrecision((log_delta + 10) as u32),
    );
    pt.set_meta(CKKSMeta {
        log_sparsity: 0,
        log_delta,
    });
    encoder.encode_reim(&mut pt, &re_in, &im_in).unwrap();

    let mut re_out = vec![F::from_f64(0.0).unwrap(); m];
    let mut im_out = vec![F::from_f64(0.0).unwrap(); m];
    encoder.decode_reim(&pt, &mut re_out, &mut im_out).unwrap();

    assert_precision_for_log_delta("encode_decode_reim re", &re_out, &re_in, log_delta, params.n);
    assert_precision_for_log_delta("encode_decode_reim im", &im_out, &im_in, log_delta, params.n);
}
