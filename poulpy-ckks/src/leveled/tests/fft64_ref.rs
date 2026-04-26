use crate::ckks_backend_test_suite;

const ATK_ROTATIONS: &[i64] = &[1, 7];

ckks_backend_test_suite!(
    mod f64_tests,
    backend = poulpy_cpu_ref::FFT64Ref,
    scalar = f64,
    params = crate::leveled::tests::test_suite::FFT64_PARAMS_F64,
    rotations = super::ATK_ROTATIONS,
);
