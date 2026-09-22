#[test]
fn fft64_accumulation_error() {
    poulpy_cpu_ref::test_suite::fft64_error::test_fft64_accumulation_error::<crate::FFT64Avx512>();
}

#[test]
#[ignore = "numerical-error model measurements with exact NTT oracles"]
fn fft64_accumulation_error_measurements() {
    poulpy_cpu_ref::test_suite::fft64_error::measure_fft64_accumulation_error::<crate::FFT64Avx512>();
}
