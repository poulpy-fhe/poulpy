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
use poulpy_hal::api::{NegacyclicFFT, NegacyclicFFTNew};
use rand_distr::num_traits::{Float, FloatConst};

use crate::fft64::reim::{fft_avx512::fft_avx512, ifft_avx512::ifft_avx512};

#[inline(always)]
pub(crate) fn as_arr<const SIZE: usize, R: Float + FloatConst>(x: &[R]) -> &[R; SIZE] {
    assert!(x.len() >= SIZE);
    unsafe { &*(x.as_ptr() as *const [R; SIZE]) }
}

/// Precomputed twiddle-factor tables for the negacyclic reim FFT and IFFT,
/// dispatching to AVX-512F-accelerated kernels.
///
/// Wraps [`ReimFFTTable`] and [`ReimIFFTTable`] into a single object that
/// implements [`NegacyclicFFT`], suitable for use as the transform provider
/// in the CPU CKKS encoding implementation.
pub struct FFT64Avx512ReimTable {
    fft: ReimFFTTable<f64>,
    ifft: ReimIFFTTable<f64>,
}

impl NegacyclicFFT<f64> for FFT64Avx512ReimTable {
    fn m(&self) -> usize {
        self.fft.m()
    }

    fn fft(&self, data: &mut [f64]) {
        ReimFFTAvx512::reim_dft_execute(&self.fft, data);
    }

    fn ifft(&self, data: &mut [f64]) {
        ReimIFFTAvx512::reim_dft_execute(&self.ifft, data);
    }
}

impl NegacyclicFFTNew<f64> for FFT64Avx512ReimTable {
    fn new(m: usize) -> Self {
        Self {
            fft: ReimFFTTable::new(m),
            ifft: ReimIFFTTable::new(m),
        }
    }
}

/// Negacyclic transform for CKKS encoding in `f64`: the AVX-512 kernels on
/// the correctly rounded twiddles of
/// [`EncodingFFTTable`](poulpy_cpu_portable::ckks_encoding::EncodingFFTTable),
/// byte identical to every other CPU backend.
#[cfg(feature = "enable-ckks")]
pub struct FFT64Avx512EncodingTable(poulpy_cpu_portable::ckks_encoding::EncodingFFTTable<f64>);

#[cfg(feature = "enable-ckks")]
impl NegacyclicFFT<f64> for FFT64Avx512EncodingTable {
    fn m(&self) -> usize {
        self.0.m()
    }

    fn fft(&self, data: &mut [f64]) {
        ReimFFTAvx512::reim_dft_execute(self.0.forward(), data);
    }

    fn ifft(&self, data: &mut [f64]) {
        ReimIFFTAvx512::reim_dft_execute(self.0.inverse(), data);
    }
}

#[cfg(feature = "enable-ckks")]
impl NegacyclicFFTNew<f64> for FFT64Avx512EncodingTable {
    fn new(m: usize) -> Self {
        Self(NegacyclicFFTNew::new(m))
    }
}

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

#[cfg(test)]
mod contract_tests {
    use super::*;
    use poulpy_hal::api::NegacyclicFFTNew;

    #[test]
    fn raw_transform_matches_contract() {
        for m in [16, 32, 64, 128, 256] {
            let table = FFT64Avx512ReimTable::new(m);
            poulpy_hal::test_suite::reim::test_negacyclic_fft(&table);
        }
    }
}

#[cfg(all(test, feature = "enable-ckks"))]
mod encoding_tests {
    use poulpy_cpu_portable::ckks_encoding::EncodingFFTTable;
    use poulpy_hal::{api::NegacyclicFFTNew, test_suite::reim::test_negacyclic_fft_bit_exact};

    use super::FFT64Avx512EncodingTable;

    #[test]
    fn encoding_transform_matches_portable() {
        for log_m in 0..=15 {
            let m = 1 << log_m;
            test_negacyclic_fft_bit_exact::<f64, _, _>(&FFT64Avx512EncodingTable::new(m), &EncodingFFTTable::<f64>::new(m));
        }
    }
}
