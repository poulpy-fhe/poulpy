#[cfg(any(feature = "ref", feature = "avx", feature = "avx512", feature = "ifma"))]
use crate::ckks_libm_backend_test_suite;

#[cfg(feature = "ref")]
ckks_libm_backend_test_suite!(
    mod reference,
    backend = poulpy_cpu_ref::NTT4x30Ref,
    scalar = f64,
    encoder = poulpy_cpu_ref::FFT64ReimTable<f64>,
    params = poulpy_ckks::test_suite::NTT4X30_PARAMS_F64,
);

#[cfg(feature = "ref")]
#[test]
fn precision_tuning_log_n12() {
    use poulpy_ckks::{
        CKKSMeta,
        test_suite::{CKKSTestParams, NTT4X30_PARAMS_F64},
    };
    use poulpy_hal::layouts::{HostBytesBackend, Module};

    let params = CKKSTestParams {
        n: 1 << 12,
        prec_meta: CKKSMeta {
            log_sparsity: 0,
            // The standard bootstrap recipe raises the output scale from the
            // 45-bit message scale by its 11-bit message ratio.
            log_delta: 56,
            slots: Default::default(),
        },
        ..NTT4X30_PARAMS_F64
    };
    let module = Module::<poulpy_cpu_ref::NTT4x30Ref>::new(params.n as u64);
    let host_module = Module::<HostBytesBackend>::new(params.n as u64);

    assert_eq!(params.log_n(), 12);
    crate::test_suite::approximation::test_precision_tuning::<
        poulpy_cpu_ref::NTT4x30Ref,
        f64,
        poulpy_cpu_ref::FFT64ReimTable<f64>,
    >(params, &module, &host_module);
}

#[cfg(feature = "avx")]
ckks_libm_backend_test_suite!(
    mod avx,
    backend = poulpy_cpu_avx::NTT4x30Avx,
    scalar = f64,
    encoder = poulpy_cpu_avx::FFT64AvxReimTable,
    params = poulpy_ckks::test_suite::NTT4X30_PARAMS_F64,
);

#[cfg(feature = "avx512")]
ckks_libm_backend_test_suite!(
    mod avx512,
    backend = poulpy_cpu_avx512::NTT4x30Avx512,
    scalar = f64,
    encoder = poulpy_cpu_avx512::FFT64Avx512ReimTable,
    params = poulpy_ckks::test_suite::NTT4X30_PARAMS_F64,
);

#[cfg(feature = "ifma")]
ckks_libm_backend_test_suite!(
    mod ifma,
    backend = poulpy_cpu_avx512::NTT3x42Ifma,
    scalar = f64,
    encoder = poulpy_cpu_avx512::FFT64Avx512ReimTable,
    params = poulpy_ckks::test_suite::NTT4X30_PARAMS_F64,
);
