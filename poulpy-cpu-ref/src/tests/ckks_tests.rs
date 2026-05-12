use poulpy_ckks::ckks_backend_test_suite;

const ATK_ROTATIONS: &[i64] = &[1, 7];

#[test]
fn encode_decode_reim_roundtrip() {
    use crate::FFT64ReimTable;
    use poulpy_ckks::encoding::reim::Encoder;
    use poulpy_ckks::layouts::CKKSModuleAlloc;

    let n = 16usize;
    let m = n / 2;

    let re_in: Vec<f64> = (0..m).map(|i| (i as f64) / (m as f64)).collect();
    let im_in: Vec<f64> = (0..m).map(|i| -((i as f64) / (m as f64))).collect();

    let encoder = Encoder::<FFT64ReimTable<f64>>::new::<f64>(m).unwrap();

    let host_module = poulpy_hal::layouts::Module::<poulpy_hal::layouts::HostBytesBackend>::new(n as u64);
    let mut pt = host_module.ckks_pt_vec_alloc(
        poulpy_core::layouts::Base2K(16),
        poulpy_ckks::CKKSMeta {
            log_delta: 40,
            log_budget: 10,
        },
    );
    encoder.encode_reim(&mut pt, &re_in, &im_in).unwrap();

    let mut re_out = vec![0.0f64; m];
    let mut im_out = vec![0.0f64; m];
    encoder.decode_reim(&pt, &mut re_out, &mut im_out).unwrap();

    let max_err = |a: &[f64], b: &[f64]| a.iter().zip(b).map(|(x, y)| (x - y).abs()).fold(0.0_f64, f64::max);
    let bound = 1e-10;
    let err_re = max_err(&re_in, &re_out);
    let err_im = max_err(&im_in, &im_out);
    assert!(err_re < bound, "re max_err={err_re:.2e} exceeds bound={bound:.2e}");
    assert!(err_im < bound, "im max_err={err_im:.2e} exceeds bound={bound:.2e}");
}

ckks_backend_test_suite!(
    mod fft64_f64,
    backend = crate::FFT64Ref,
    scalar = f64,
    encoder = crate::FFT64ReimTable<f64>,
    params = poulpy_ckks::test_suite::FFT64_PARAMS_F64,
    rotations = super::ATK_ROTATIONS,
);

ckks_backend_test_suite!(
    mod ntt120_f64,
    backend = crate::NTT120Ref,
    scalar = f64,
    encoder = crate::FFT64ReimTable<f64>,
    params = poulpy_ckks::test_suite::NTT120_PARAMS_F64,
    rotations = super::ATK_ROTATIONS,
);

ckks_backend_test_suite!(
    mod ntt120_f128,
    backend = crate::NTT120Ref,
    scalar = f128::f128,
    encoder = crate::FFT64ReimTable<f128::f128>,
    params = poulpy_ckks::test_suite::NTT120_PARAMS_F128,
    rotations = super::ATK_ROTATIONS,
);
