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

use poulpy_cpu_portable::reference::fft64::reim::{ReimFFTExecute, ReimFFTTable, ReimIFFTTable};
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

#[inline]
#[target_feature(enable = "avx512f")]
fn encoding_mul_add<const FUSED: bool>(
    a: std::arch::x86_64::__m512d,
    b: std::arch::x86_64::__m512d,
    c: std::arch::x86_64::__m512d,
) -> std::arch::x86_64::__m512d {
    use std::arch::x86_64::*;
    if FUSED {
        _mm512_fmadd_pd(a, b, c)
    } else {
        _mm512_add_pd(_mm512_mul_pd(a, b), c)
    }
}

#[inline]
#[target_feature(enable = "avx512f")]
fn encoding_mul_sub<const FUSED: bool>(
    a: std::arch::x86_64::__m512d,
    b: std::arch::x86_64::__m512d,
    c: std::arch::x86_64::__m512d,
) -> std::arch::x86_64::__m512d {
    use std::arch::x86_64::*;
    if FUSED {
        _mm512_fmsub_pd(a, b, c)
    } else {
        _mm512_sub_pd(_mm512_mul_pd(a, b), c)
    }
}

#[cfg(feature = "enable-ckks")]
pub struct EncodingFFTTable(poulpy_cpu_portable::ckks_encoding::EncodingFFTTable<f64>);

#[cfg(feature = "enable-ckks")]
impl NegacyclicFFTNew<f64> for EncodingFFTTable {
    fn new(m: usize) -> Self {
        Self(NegacyclicFFTNew::new(m))
    }
}

#[cfg(feature = "enable-ckks")]
impl NegacyclicFFT<f64> for EncodingFFTTable {
    fn m(&self) -> usize {
        self.0.m()
    }
    fn fft(&self, data: &mut [f64]) {
        unsafe {
            fft_avx512::fft_avx512_with_fma::<false>(self.m(), self.0.fft_twiddles(), data);
        }
    }
    fn ifft(&self, data: &mut [f64]) {
        unsafe {
            ifft_avx512::ifft_avx512_with_fma::<false>(self.m(), self.0.ifft_twiddles(), data);
        }
    }
}

#[cfg(all(test, feature = "enable-ckks"))]
#[test]
fn encoding_fft_matches_oracle() {
    poulpy_ckks::test_suite::determinism::assert_transform_matches::<
        f64,
        poulpy_cpu_oracle::ckks_encoding_fft::EncodingFFTTable<f64>,
        EncodingFFTTable,
    >();
}
