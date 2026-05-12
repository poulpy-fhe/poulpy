//! Real/imaginary interleaved FFT primitives for [`FFT64Ref`](crate::FFT64Ref).
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

use std::fmt::Debug;

use crate::reference::fft64::{
    convolution::I64Ops,
    reim::{ReimArith, ReimFFTExecute, ReimFFTTable, ReimIFFTTable, fft_ref, ifft_ref},
    reim4::{Reim4BlkMatVec, Reim4Convolution},
};
use poulpy_hal::api::{NegacyclicFFT, NegacyclicFFTNew};
use rand_distr::num_traits::{Float, FloatConst};

use super::FFT64Ref;

/// Precomputed twiddle-factor tables for the negacyclic reim FFT and IFFT.
///
/// Wraps [`ReimFFTTable`] and [`ReimIFFTTable`] into a single object that
/// implements [`NegacyclicFFT`], suitable for use as the transform provider
/// in a CKKS [`poulpy_ckks::encoding::Encoder`].
pub struct FFT64ReimTable<F: Float + FloatConst + Debug> {
    fft: ReimFFTTable<F>,
    ifft: ReimIFFTTable<F>,
}

impl<F: Float + FloatConst + Debug> NegacyclicFFT<F> for FFT64ReimTable<F> {
    fn m(&self) -> usize {
        self.fft.m()
    }

    fn fft(&self, data: &mut [F]) {
        self.fft.execute(data);
    }

    fn ifft(&self, data: &mut [F]) {
        self.ifft.execute(data);
    }
}

impl<F: Float + FloatConst + Debug> NegacyclicFFTNew<F> for FFT64ReimTable<F> {
    fn new(m: usize) -> Self {
        Self {
            fft: ReimFFTTable::new(m),
            ifft: ReimIFFTTable::new(m),
        }
    }
}

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
