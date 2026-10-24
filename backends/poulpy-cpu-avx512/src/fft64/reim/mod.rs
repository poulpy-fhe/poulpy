// ----------------------------------------------------------------------
// DISCLAIMER
//
// This module contains code adapted from the AVX2 / FMA C kernels of the
// spqlios-arithmetic library
// (https://github.com/tfhe/spqlios-arithmetic), which is licensed
// under the Apache License, Version 2.0.
//
// The 256-bit AVX2 originals were widened to 512-bit AVX-512 and translated
// to Rust intrinsics; algorithmic structure is preserved one-to-one with the
// spqlios sources to keep semantics identical.
//
// Both Poulpy and spqlios-arithmetic are distributed under the terms
// of the Apache License, Version 2.0. See the LICENSE file for details.
//
// ----------------------------------------------------------------------

mod conversion;
mod fft_avx512;
mod fft_vec_avx512;
mod ifft_avx512;

pub(crate) use conversion::*;
pub(crate) use fft_vec_avx512::*;

use poulpy_cpu_portable::kernels::fft64::reim::{ReimFFTExecute, ReimFFTTable, ReimIFFTTable};
use rand_distr::num_traits::{Float, FloatConst};

use crate::fft64::reim::{fft_avx512::fft_avx512, ifft_avx512::ifft_avx512};

#[inline(always)]
pub(crate) fn as_arr<const SIZE: usize, R: Float + FloatConst>(x: &[R]) -> &[R; SIZE] {
    assert!(x.len() >= SIZE);
    unsafe { &*(x.as_ptr() as *const [R; SIZE]) }
}

/// Negacyclic transform for CKKS encoding in `f64`: the AVX-512 kernels on
/// the correctly rounded twiddles of
/// [`EncodingFFTTable`](poulpy_cpu_portable::ckks_encoding::EncodingFFTTable),
/// byte identical to every other CPU backend.
#[cfg(feature = "enable-ckks")]
pub type FFT64Avx512EncodingTable = poulpy_cpu_portable::ckks_encoding::EncodingFFTTable<f64, ReimFFTAvx512, ReimIFFTAvx512>;

/// Former CKKS encoding transform, now [`FFT64Avx512EncodingTable`].
#[cfg(feature = "enable-ckks")]
#[deprecated(note = "use `FFT64Avx512EncodingTable`, which encodes byte identically on every CPU backend")]
pub type FFT64Avx512ReimTable = FFT64Avx512EncodingTable;

pub struct ReimFFTAvx512;

impl ReimFFTExecute<ReimFFTTable<f64>, f64> for ReimFFTAvx512 {
    #[inline(always)]
    fn reim_dft_execute(table: &ReimFFTTable<f64>, data: &mut [f64]) {
        unsafe {
            fft_avx512(table.m(), table.omg(), data);
        }
    }
}

pub struct ReimIFFTAvx512;

impl ReimFFTExecute<ReimIFFTTable<f64>, f64> for ReimIFFTAvx512 {
    #[inline(always)]
    fn reim_dft_execute(table: &ReimIFFTTable<f64>, data: &mut [f64]) {
        unsafe {
            ifft_avx512(table.m(), table.omg(), data);
        }
    }
}

#[cfg(all(test, feature = "enable-ckks"))]
mod encoding_tests {
    use poulpy_cpu_portable::ckks_encoding::EncodingFFTTable;
    use poulpy_hal::{
        api::NegacyclicFFTNew,
        test_suite::reim::{test_negacyclic_fft, test_negacyclic_fft_bit_exact},
    };

    use super::FFT64Avx512EncodingTable;

    #[test]
    fn raw_transform_matches_contract() {
        for m in [16, 32, 64, 128, 256] {
            test_negacyclic_fft(&FFT64Avx512EncodingTable::new(m));
        }
    }

    #[test]
    fn encoding_transform_matches_portable() {
        for log_m in 0..=15 {
            let m = 1 << log_m;
            test_negacyclic_fft_bit_exact::<f64, _, _>(&FFT64Avx512EncodingTable::new(m), &EncodingFFTTable::<f64>::new(m));
        }
    }
}
