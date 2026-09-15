// ----------------------------------------------------------------------
// DISCLAIMER
//
// This module contains code that has been directly ported from the
// spqlios-arithmetic library
// (https://github.com/tfhe/spqlios-arithmetic), which is licensed
// under the Apache License, Version 2.0.
//
// The porting process from C to Rust was done with minimal changes
// in order to preserve the semantics and performance characteristics
// of the original implementation.
//
// Both Poulpy and spqlios-arithmetic are distributed under the terms
// of the Apache License, Version 2.0. See the LICENSE file for details.
//
// ----------------------------------------------------------------------

#![allow(bad_asm_style)]

mod conversion;
mod fft_avx2_fma;
mod fft_vec_avx2_fma;
mod ifft_avx2_fma;

use std::arch::global_asm;

pub(crate) use conversion::*;
pub(crate) use fft_vec_avx2_fma::*;

use poulpy_cpu_portable::reference::fft64::reim::{ReimFFTExecute, ReimFFTTable, ReimIFFTTable};
use poulpy_hal::api::{NegacyclicFFT, NegacyclicFFTNew};
use rand_distr::num_traits::{Float, FloatConst};

use crate::fft64::reim::{fft_avx2_fma::fft_avx2_fma, ifft_avx2_fma::ifft_avx2_fma};

macro_rules! fft16_kernels {
    ($fft:literal, $ifft:literal, $fused:expr) => {
        #[cfg(target_vendor = "apple")]
        global_asm!(
            ".att_syntax prefix",
            include_str!("mul_add.s"),
            ".text",
            concat!(".globl _", $fft),
            concat!(".private_extern _", $fft),
            ".p2align 4, 0x90",
            concat!("_", $fft, ":"),
            include_str!("fft16_avx2_fma.s"),
            concat!(".globl _", $ifft),
            concat!(".private_extern _", $ifft),
            ".p2align 4, 0x90",
            concat!("_", $ifft, ":"),
            include_str!("ifft16_avx2_fma.s"),
            ".purgem poulpy_madd",
            ".purgem poulpy_msub",
            fused = const $fused as usize,
        );
        #[cfg(not(target_vendor = "apple"))]
        global_asm!(
            ".att_syntax prefix",
            include_str!("mul_add.s"),
            ".text",
            concat!(".globl ", $fft),
            concat!(".hidden ", $fft),
            concat!(".type ", $fft, ",@function"),
            ".p2align 4, 0x90",
            concat!($fft, ":"),
            include_str!("fft16_avx2_fma.s"),
            concat!(".size ", $fft, ", .-", $fft),
            concat!(".globl ", $ifft),
            concat!(".hidden ", $ifft),
            concat!(".type ", $ifft, ",@function"),
            ".p2align 4, 0x90",
            concat!($ifft, ":"),
            include_str!("ifft16_avx2_fma.s"),
            concat!(".size ", $ifft, ", .-", $ifft),
            ".purgem poulpy_madd",
            ".purgem poulpy_msub",
            ".section .note.GNU-stack,\"\",@progbits",
            fused = const $fused as usize,
        );
    };
}

fft16_kernels!("fft16_avx2_fma_asm", "ifft16_avx2_fma_asm", true);
fft16_kernels!("fft16_avx2_encoding_asm", "ifft16_avx2_encoding_asm", false);

#[inline(always)]
pub(crate) fn as_arr<const SIZE: usize, R: Float + FloatConst>(x: &[R]) -> &[R; SIZE] {
    assert!(x.len() >= SIZE);
    unsafe { &*(x.as_ptr() as *const [R; SIZE]) }
}

#[inline(always)]
pub(crate) fn as_arr_mut<const SIZE: usize, R: Float + FloatConst>(x: &mut [R]) -> &mut [R; SIZE] {
    assert!(x.len() >= SIZE);
    unsafe { &mut *(x.as_mut_ptr() as *mut [R; SIZE]) }
}

/// Precomputed twiddle-factor tables for the negacyclic reim FFT and IFFT,
/// dispatching to AVX2/FMA-accelerated kernels.
///
/// Wraps [`ReimFFTTable`] and [`ReimIFFTTable`] for fused ring arithmetic.
/// CKKS uses a separate table with its canonical rounding contract.
pub struct FFT64AvxReimTable {
    fft: ReimFFTTable<f64>,
    ifft: ReimIFFTTable<f64>,
}

impl NegacyclicFFT<f64> for FFT64AvxReimTable {
    fn m(&self) -> usize {
        self.fft.m()
    }

    fn fft(&self, data: &mut [f64]) {
        ReimFFTAvx::reim_dft_execute(&self.fft, data);
    }

    fn ifft(&self, data: &mut [f64]) {
        ReimIFFTAvx::reim_dft_execute(&self.ifft, data);
    }
}

impl NegacyclicFFTNew<f64> for FFT64AvxReimTable {
    fn new(m: usize) -> Self {
        Self {
            fft: ReimFFTTable::new(m),
            ifft: ReimIFFTTable::new(m),
        }
    }
}

pub struct ReimFFTAvx;

impl ReimFFTExecute<ReimFFTTable<f64>, f64> for ReimFFTAvx {
    #[inline(always)]
    fn reim_dft_execute(table: &ReimFFTTable<f64>, data: &mut [f64]) {
        unsafe {
            fft_avx2_fma(table.m(), table.omg(), data);
        }
    }
}

pub struct ReimIFFTAvx;

impl ReimFFTExecute<ReimIFFTTable<f64>, f64> for ReimIFFTAvx {
    #[inline(always)]
    fn reim_dft_execute(table: &ReimIFFTTable<f64>, data: &mut [f64]) {
        unsafe {
            ifft_avx2_fma(table.m(), table.omg(), data);
        }
    }
}

#[inline]
#[target_feature(enable = "avx2,fma")]
fn encoding_mul_add<const FUSED: bool>(
    a: std::arch::x86_64::__m256d,
    b: std::arch::x86_64::__m256d,
    c: std::arch::x86_64::__m256d,
) -> std::arch::x86_64::__m256d {
    use std::arch::x86_64::*;
    if FUSED {
        _mm256_fmadd_pd(a, b, c)
    } else {
        _mm256_add_pd(_mm256_mul_pd(a, b), c)
    }
}

#[inline]
#[target_feature(enable = "avx2,fma")]
fn encoding_mul_sub<const FUSED: bool>(
    a: std::arch::x86_64::__m256d,
    b: std::arch::x86_64::__m256d,
    c: std::arch::x86_64::__m256d,
) -> std::arch::x86_64::__m256d {
    use std::arch::x86_64::*;
    if FUSED {
        _mm256_fmsub_pd(a, b, c)
    } else {
        _mm256_sub_pd(_mm256_mul_pd(a, b), c)
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
            fft_avx2_fma::fft_avx2_with_fma::<false>(self.m(), self.0.fft_twiddles(), data);
        }
    }
    fn ifft(&self, data: &mut [f64]) {
        unsafe {
            ifft_avx2_fma::ifft_avx2_with_fma::<false>(self.m(), self.0.ifft_twiddles(), data);
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
