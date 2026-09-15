use std::fmt::Debug;

use rand_distr::num_traits::{Float, FloatConst};

use super::ifft_ref;

pub struct ReimIFFTTable<R: Float + FloatConst + Debug> {
    m: usize,
    roots: Vec<R>,
}

impl<R: Float + FloatConst + Debug> ReimIFFTTable<R> {
    pub fn new(m: usize) -> Self {
        Self {
            m,
            roots: super::fft_ref::roots(m, true),
        }
    }

    pub fn execute(&self, data: &mut [R]) {
        ifft_ref(self.m, &self.roots, data);
    }

    pub fn m(&self) -> usize {
        self.m
    }

    pub fn omg(&self) -> &[R] {
        &self.roots
    }
}
