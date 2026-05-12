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

use poulpy_cpu_ref::reference::fft64::reim::{ReimFFTExecute, ReimFFTTable, ReimIFFTTable};
use poulpy_hal::api::{NegacyclicFFT, NegacyclicFFTNew};
use rand_distr::num_traits::{Float, FloatConst};

use crate::fft64::reim::{fft_avx512::fft_avx512, ifft_avx512::ifft_avx512};

#[inline(always)]
pub(crate) fn as_arr<const SIZE: usize, R: Float + FloatConst>(x: &[R]) -> &[R; SIZE] {
    debug_assert!(x.len() >= SIZE);
    unsafe { &*(x.as_ptr() as *const [R; SIZE]) }
}

/// Precomputed twiddle-factor tables for the negacyclic reim FFT and IFFT,
/// dispatching to AVX-512F-accelerated kernels.
///
/// Wraps [`ReimFFTTable`] and [`ReimIFFTTable`] into a single object that
/// implements [`NegacyclicFFT`], suitable for use as the transform provider
/// in a CKKS [`poulpy_ckks::encoding::Encoder`].
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
