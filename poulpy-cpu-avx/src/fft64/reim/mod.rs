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

use poulpy_cpu_ref::reference::fft64::reim::{ReimFFTExecute, ReimFFTTable, ReimIFFTTable};
use poulpy_hal::api::{NegacyclicFFT, NegacyclicFFTNew};
use rand_distr::num_traits::{Float, FloatConst};

use crate::fft64::reim::{fft_avx2_fma::fft_avx2_fma, ifft_avx2_fma::ifft_avx2_fma};

// Mach-O: symbols carry a leading underscore, and `.hidden`/`.type`/`.size`/
// `.note.GNU-stack` are ELF-only directives the Mach-O assembler rejects.
#[cfg(target_vendor = "apple")]
global_asm!(
    ".text",
    ".globl  _fft16_avx2_fma_asm",
    ".private_extern _fft16_avx2_fma_asm",
    ".p2align 4, 0x90",
    "_fft16_avx2_fma_asm:",
    ".att_syntax prefix",
    include_str!("fft16_avx2_fma.s"),
    ".text",
    ".globl  _ifft16_avx2_fma_asm",
    ".private_extern _ifft16_avx2_fma_asm",
    ".p2align 4, 0x90",
    "_ifft16_avx2_fma_asm:",
    ".att_syntax prefix",
    include_str!("ifft16_avx2_fma.s"),
);

#[cfg(not(target_vendor = "apple"))]
global_asm!(
    ".text",
    ".globl  fft16_avx2_fma_asm",
    ".hidden fft16_avx2_fma_asm",
    ".p2align 4, 0x90",
    ".type   fft16_avx2_fma_asm,@function",
    "fft16_avx2_fma_asm:",
    ".att_syntax prefix",
    include_str!("fft16_avx2_fma.s"),
    ".size   fft16_avx2_fma_asm, .-fft16_avx2_fma_asm",
    ".text",
    ".globl  ifft16_avx2_fma_asm",
    ".hidden ifft16_avx2_fma_asm",
    ".p2align 4, 0x90",
    ".type   ifft16_avx2_fma_asm,@function",
    "ifft16_avx2_fma_asm:",
    ".att_syntax prefix",
    include_str!("ifft16_avx2_fma.s"),
    ".size   ifft16_avx2_fma_asm, .-ifft16_avx2_fma_asm",
    ".section .note.GNU-stack,\"\",@progbits",
);

#[inline(always)]
pub(crate) fn as_arr<const SIZE: usize, R: Float + FloatConst>(x: &[R]) -> &[R; SIZE] {
    debug_assert!(x.len() >= SIZE);
    unsafe { &*(x.as_ptr() as *const [R; SIZE]) }
}

#[inline(always)]
pub(crate) fn as_arr_mut<const SIZE: usize, R: Float + FloatConst>(x: &mut [R]) -> &mut [R; SIZE] {
    debug_assert!(x.len() >= SIZE);
    unsafe { &mut *(x.as_mut_ptr() as *mut [R; SIZE]) }
}

/// Precomputed twiddle-factor tables for the negacyclic reim FFT and IFFT,
/// dispatching to AVX2/FMA-accelerated kernels.
///
/// Wraps [`ReimFFTTable`] and [`ReimIFFTTable`] into a single object that
/// implements [`NegacyclicFFT`], suitable for use as the transform provider
/// in the CPU CKKS encoding implementation.
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
