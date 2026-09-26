use poulpy_cpu_ref::reference::fft64::reim::{ReimFFTExecute, ReimFFTTable, ReimIFFTTable};
#[cfg(not(target_arch = "aarch64"))]
use poulpy_cpu_ref::reference::fft64::reim::{fft_ref, ifft_ref};
use poulpy_hal::api::{NegacyclicFFT, NegacyclicFFTNew};

/// Precomputed twiddle-factor tables for the negacyclic reim FFT and IFFT,
/// dispatching to NEON-accelerated kernels on AArch64 and the portable
/// reference kernels otherwise.
/// Wraps [`ReimFFTTable`] and [`ReimIFFTTable`] into a single object that
/// implements [`NegacyclicFFT`], suitable for use as the transform provider
/// in the CPU CKKS encoding implementation.
pub struct FFT64NeonReimTable {
    fft: ReimFFTTable<f64>,
    ifft: ReimIFFTTable<f64>,
}

impl NegacyclicFFT<f64> for FFT64NeonReimTable {
    fn m(&self) -> usize {
        self.fft.m()
    }

    fn fft(&self, data: &mut [f64]) {
        ReimFFTNeon::reim_dft_execute(&self.fft, data);
    }

    fn ifft(&self, data: &mut [f64]) {
        ReimIFFTNeon::reim_dft_execute(&self.ifft, data);
    }
}

impl NegacyclicFFTNew<f64> for FFT64NeonReimTable {
    fn new(m: usize) -> Self {
        Self {
            fft: ReimFFTTable::new(m),
            ifft: ReimIFFTTable::new(m),
        }
    }
}

pub struct ReimFFTNeon;

impl ReimFFTExecute<ReimFFTTable<f64>, f64> for ReimFFTNeon {
    #[inline(always)]
    fn reim_dft_execute(table: &ReimFFTTable<f64>, data: &mut [f64]) {
        #[cfg(target_arch = "aarch64")]
        {
            crate::neon::fft::fft_neon(table.m(), table.omg(), data);
        }
        #[cfg(not(target_arch = "aarch64"))]
        {
            fft_ref(table.m(), table.omg(), data);
        }
    }
}

pub struct ReimIFFTNeon;

impl ReimFFTExecute<ReimIFFTTable<f64>, f64> for ReimIFFTNeon {
    #[inline(always)]
    fn reim_dft_execute(table: &ReimIFFTTable<f64>, data: &mut [f64]) {
        #[cfg(target_arch = "aarch64")]
        {
            crate::neon::fft::ifft_neon(table.m(), table.omg(), data);
        }
        #[cfg(not(target_arch = "aarch64"))]
        {
            ifft_ref(table.m(), table.omg(), data);
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
            let table = FFT64NeonReimTable::new(m);
            poulpy_hal::test_suite::reim::test_negacyclic_fft(&table);
        }
    }
}
