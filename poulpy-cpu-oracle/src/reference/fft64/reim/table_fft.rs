use std::fmt::Debug;

use rand_distr::num_traits::{Float, FloatConst};

use super::{ReimFFTExecute, fft_ref};

pub struct ReimFFTRef;

impl ReimFFTExecute<ReimFFTTable<f64>, f64> for ReimFFTRef {
    fn reim_dft_execute(table: &ReimFFTTable<f64>, data: &mut [f64]) {
        table.execute(data);
    }
}

pub struct ReimFFTTable<R: Float + FloatConst + Debug> {
    m: usize,
    roots: Vec<R>,
}

impl<R: Float + FloatConst + Debug> ReimFFTTable<R> {
    pub fn new(m: usize) -> Self {
        Self {
            m,
            roots: super::fft_ref::roots(m, false),
        }
    }

    pub fn execute(&self, data: &mut [R]) {
        fft_ref(self.m, &self.roots, data);
    }

    pub fn m(&self) -> usize {
        self.m
    }

    pub fn omg(&self) -> &[R] {
        &self.roots
    }
}
