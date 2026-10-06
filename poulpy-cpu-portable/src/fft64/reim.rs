//! Real/imaginary interleaved FFT primitives for [`FFT64Portable`](super::FFT64Portable).
//!
//! Implements the `ReimArith`, `Reim4BlkMatVec`, `Reim4Convolution`, and `I64Ops`
//! traits from `crate::kernels::fft64`, covering:
//!
//! - **FFT/IFFT execution**: forward and inverse transforms using precomputed twiddle tables.
//! - **Domain conversion**: `Z[X]/(X^n+1)` integer coefficients to/from `f64` REIM layout.
//! - **Frequency-domain arithmetic**: pointwise add, sub, negate, mul, and fused multiply-add.
//! - **4-block batch operations**: `Reim4` variants that process 4 interleaved coefficient
//!   blocks in a single pass, used internally by convolution and VMP kernels. These include
//!   block extraction/save, matrix-vector products, and convolution-by-constant.
//! - **Integer block operations**: `I64` variants for constant-coefficient convolution
//!   and block save/extract in the integer domain.
//!
//! All implementations use the default `_portable` implementations.

use super::FFT64Portable;
use poulpy_hal::layouts::Ring;

use crate::kernels::fft64::{
    convolution::I64Ops,
    reim::{ReimArith, ReimFFTExecute, ReimFFTTable, ReimIFFTTable, fft_portable, ifft_portable},
    reim4::{Reim4BlkMatVec, Reim4Convolution},
};

impl<R: Ring> ReimFFTExecute<ReimFFTTable<f64>, f64> for FFT64Portable<R> {
    fn reim_dft_execute(table: &ReimFFTTable<f64>, data: &mut [f64]) {
        fft_portable(table.m(), table.omg(), data);
    }
}

impl<R: Ring> ReimFFTExecute<ReimIFFTTable<f64>, f64> for FFT64Portable<R> {
    fn reim_dft_execute(table: &ReimIFFTTable<f64>, data: &mut [f64]) {
        ifft_portable(table.m(), table.omg(), data);
    }
}

impl<R: Ring> ReimArith for FFT64Portable<R> {}

impl<R: Ring> Reim4BlkMatVec for FFT64Portable<R> {}

impl<R: Ring> Reim4Convolution for FFT64Portable<R> {}

impl<R: Ring> I64Ops for FFT64Portable<R> {}

impl crate::kernels::fft64::ring_arith::Fft64RingArith for FFT64Portable<poulpy_hal::layouts::Standard> {
    crate::fft64_ring_arith_standard!();
}

impl crate::kernels::fft64::ring_arith::Fft64RingArith for FFT64Portable<poulpy_hal::layouts::ConjugateInvariant> {
    crate::fft64_ring_arith_ci!();
}
