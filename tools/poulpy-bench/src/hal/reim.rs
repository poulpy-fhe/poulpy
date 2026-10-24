use std::hint::black_box;

use criterion::{Bencher, measurement::Measurement};

use poulpy_hal::api::{NegacyclicFFT, NegacyclicFFTNew};

use crate::hal::params::ReimSweepParams;

fn reim_values(m: usize) -> Vec<f64> {
    let mut values: Vec<f64> = vec![0f64; m << 1];
    let scale: f64 = 1.0f64 / (2 * m) as f64;
    values.iter_mut().enumerate().for_each(|(i, x)| *x = (i + 1) as f64 * scale);
    values
}

pub fn runner_reim_fft<T, M: Measurement>(bencher: &mut Bencher<'_, M>, sweep: &ReimSweepParams)
where
    T: NegacyclicFFT<f64> + NegacyclicFFTNew<f64>,
{
    let table: T = T::new(sweep.m);
    let mut values: Vec<f64> = reim_values(sweep.m);

    bencher.iter(|| {
        table.fft(&mut values);
        black_box(());
    });
}

pub fn runner_reim_ifft<T, M: Measurement>(bencher: &mut Bencher<'_, M>, sweep: &ReimSweepParams)
where
    T: NegacyclicFFT<f64> + NegacyclicFFTNew<f64>,
{
    let table: T = T::new(sweep.m);
    let mut values: Vec<f64> = reim_values(sweep.m);

    bencher.iter(|| {
        table.ifft(&mut values);
        black_box(());
    });
}
