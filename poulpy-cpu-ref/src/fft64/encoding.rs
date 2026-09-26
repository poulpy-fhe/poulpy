use crate::reference::fft64::{
    module::FFT64Plan,
    reim::{ReimFFTTable, ReimIFFTTable},
};
use bytemuck::Zeroable;
use poulpy_hal::api::{NegacyclicFFT, NegacyclicFFTNew};
use rand_distr::num_traits::{Float, FloatConst};
use std::fmt::Debug;

/// Precomputed twiddle-factor tables for the negacyclic reim FFT and IFFT.
///
/// Wraps [`ReimFFTTable`] and [`ReimIFFTTable`] into a single object that
/// implements [`NegacyclicFFT`], suitable for use as the transform provider
/// in the CPU CKKS encoding implementation.
pub struct FFT64ReimTable<F: Float + FloatConst + Debug + Zeroable> {
    fft: ReimFFTTable<F>,
    ifft: ReimIFFTTable<F>,
}

impl<F: Float + FloatConst + Debug + Zeroable> NegacyclicFFT<F> for FFT64ReimTable<F> {
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

impl<F: Float + FloatConst + Debug + Zeroable> NegacyclicFFTNew<F> for FFT64ReimTable<F> {
    fn new(m: usize) -> Self {
        Self {
            fft: ReimFFTTable::new(m),
            ifft: ReimIFFTTable::new(m),
        }
    }
}

impl<F: Float + FloatConst + Debug + Zeroable> NegacyclicFFT<F> for FFT64Plan<F> {
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

#[cfg(test)]
mod contract_tests {
    use super::*;

    #[test]
    fn raw_transform_matches_contract() {
        for m in [1, 2, 4, 8, 16, 32, 64] {
            let table = FFT64ReimTable::<f64>::new(m);
            poulpy_hal::test_suite::reim::test_negacyclic_fft(&table);
        }
    }
}
