#![cfg(feature = "enable-ckks")]

use poulpy_ckks::{Quad, test_suite::determinism::assert_transform_matches};
use poulpy_cpu_oracle::ckks_encoding_fft::EncodingFFTTable as Oracle;
use poulpy_cpu_portable::ckks_encoding::EncodingFFTTable as Portable;

#[test]
fn independent_encoding_fft_f64() {
    assert_transform_matches::<f64, Oracle<f64>, Portable<f64>>();
}

#[test]
fn independent_encoding_fft_f128() {
    assert_transform_matches::<Quad, Oracle<Quad>, Portable<Quad>>();
}
