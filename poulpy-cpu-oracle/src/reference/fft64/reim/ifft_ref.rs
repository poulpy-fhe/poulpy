use std::fmt::Debug;

use rand_distr::num_traits::{Float, FloatConst};

pub fn ifft_ref<R: Float + FloatConst + Debug>(m: usize, roots: &[R], data: &mut [R]) {
    super::fft_ref::transform(m, roots, data, true);
}
