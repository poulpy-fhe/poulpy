//! Real/imaginary interleaved FFT primitives for [`FFT64Ref`](super::FFT64Ref).
//!
//! Implements the `ReimArith`, `Reim4BlkMatVec`, `Reim4Convolution`, and `I64Ops`
//! traits from `crate::reference::fft64`, covering:
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
//! All implementations use the default `_ref` implementations.

use super::FFT64Ref;

use crate::reference::fft64::{
    convolution::I64Ops,
    reim::{ReimArith, ReimFFTExecute, ReimFFTTable, ReimIFFTTable, fft_ref, ifft_ref},
    reim4::{Reim4BlkMatVec, Reim4Convolution},
};

impl ReimFFTExecute<ReimFFTTable<f64>, f64> for FFT64Ref {
    fn reim_dft_execute(table: &ReimFFTTable<f64>, data: &mut [f64]) {
        fft_ref(table.m(), table.omg(), data);
    }
}

impl ReimFFTExecute<ReimIFFTTable<f64>, f64> for FFT64Ref {
    fn reim_dft_execute(table: &ReimIFFTTable<f64>, data: &mut [f64]) {
        ifft_ref(table.m(), table.omg(), data);
    }
}

impl ReimArith for FFT64Ref {}

impl Reim4BlkMatVec for FFT64Ref {}

impl Reim4Convolution for FFT64Ref {}

impl I64Ops for FFT64Ref {}
