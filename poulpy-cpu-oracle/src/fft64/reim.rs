use std::fmt::Debug;

use crate::reference::fft64::reim::{ReimArith, ReimFFTExecute, ReimFFTTable, ReimIFFTTable, fft_ref, ifft_ref};
use poulpy_hal::api::{NegacyclicFFT, NegacyclicFFTNew};
use rand_distr::num_traits::{Float, FloatConst};

use super::FFT64Oracle;
use crate::reference::fft64::module::FFT64Plan;

/// Precomputed twiddle-factor tables for the negacyclic reim FFT and IFFT.
///
/// Wraps forward and inverse transform tables into a single object that
/// implements [`NegacyclicFFT`], suitable for use as the transform provider
/// in the CPU CKKS encoding implementation.
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

impl<F: Float + FloatConst + Debug> NegacyclicFFT<F> for FFT64Plan<F> {
    fn m(&self) -> usize {
        self.fft().m()
    }

    fn fft(&self, data: &mut [F]) {
        self.fft().execute(data);
    }

    fn ifft(&self, data: &mut [F]) {
        self.ifft().execute(data);
    }
}

impl ReimFFTExecute<ReimFFTTable<f64>, f64> for FFT64Oracle {
    fn reim_dft_execute(table: &ReimFFTTable<f64>, data: &mut [f64]) {
        fft_ref(table.m(), table.omg(), data);
    }
}

impl ReimFFTExecute<ReimIFFTTable<f64>, f64> for FFT64Oracle {
    fn reim_dft_execute(table: &ReimIFFTTable<f64>, data: &mut [f64]) {
        ifft_ref(table.m(), table.omg(), data);
    }
}

impl ReimArith for FFT64Oracle {}
