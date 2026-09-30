//! The CKKS encoding transform, as a plain recursive radix-2 negacyclic FFT.
//!
//! A block of `n` values with turn `J` pairs `(j, j + n/2)` through one
//! butterfly with `w = exp(2*pi*i * J/2)`, then transforms its halves with
//! turns `J/2` and `J/2 + 1/2`. The whole transform starts at `J = 1/4` and
//! its output is in bit-reversed order. The inverse visits the same tree in
//! postorder with the conjugate roots and is unscaled.
//!
//! Each butterfly computes `b * w` with one multiply-add per component,
//! rounding the product by `Im(w)`, as the CPU backends do. The roots come
//! from the fixed-point derivation of [`ckks_roots`](crate::ckks_roots), not
//! from the production tables, so this checks the production transforms
//! value for value.

use poulpy_ckks::numerics::CKKSFloat;
use poulpy_hal::api::{NegacyclicFFT, NegacyclicFFTNew};

use crate::ckks_roots::root_of_unity;

/// Encoding transform of `m` complex values at precision `F`.
pub struct EncodingFft<F> {
    m: usize,
    /// `roots[k] = exp(2*pi*i * k / 4m)`.
    roots: Vec<(F, F)>,
}

/// `b * w` with the real part rounded as `fma(br, wr, -(bi * wi))` and the
/// imaginary part as `fma(bi, wr, br * wi)`.
fn mul<F: CKKSFloat>((br, bi): (F, F), (wr, wi): (F, F)) -> (F, F) {
    (br.mul_add(wr, -(bi * wi)), bi.mul_add(wr, br * wi))
}

impl<F: CKKSFloat> EncodingFft<F> {
    /// Forward on the block `[off, off + n)` with turn `j / 4m`.
    fn forward(&self, re: &mut [F], im: &mut [F], off: usize, n: usize, j: usize) {
        if n == 1 {
            return;
        }
        let h = n / 2;
        let w = self.roots[j / 2];
        for i in off..off + h {
            let (dr, di) = mul((re[i + h], im[i + h]), w);
            (re[i], im[i], re[i + h], im[i + h]) = (re[i] + dr, im[i] + di, re[i] - dr, im[i] - di);
        }
        self.forward(re, im, off, h, j / 2);
        self.forward(re, im, off + h, h, j / 2 + 2 * self.m);
    }

    /// Inverse on the block `[off, off + n)` with turn `j / 4m`.
    fn inverse(&self, re: &mut [F], im: &mut [F], off: usize, n: usize, j: usize) {
        if n == 1 {
            return;
        }
        let h = n / 2;
        self.inverse(re, im, off, h, j / 2);
        self.inverse(re, im, off + h, h, j / 2 + 2 * self.m);
        let (wr, wi) = self.roots[j / 2];
        for i in off..off + h {
            let (dr, di) = (re[i] - re[i + h], im[i] - im[i + h]);
            (re[i], im[i]) = (re[i] + re[i + h], im[i] + im[i + h]);
            (re[i + h], im[i + h]) = mul((dr, di), (wr, -wi));
        }
    }
}

impl<F: CKKSFloat> NegacyclicFFTNew<F> for EncodingFft<F> {
    fn new(m: usize) -> Self {
        let log_order = (4 * m).trailing_zeros();
        Self {
            m,
            roots: (0..4 * m as u64).map(|k| root_of_unity(k, log_order)).collect(),
        }
    }
}

impl<F: CKKSFloat> NegacyclicFFT<F> for EncodingFft<F> {
    fn m(&self) -> usize {
        self.m
    }

    fn fft(&self, data: &mut [F]) {
        assert_eq!(data.len(), 2 * self.m);
        let (re, im) = data.split_at_mut(self.m);
        self.forward(re, im, 0, self.m, self.m);
    }

    fn ifft(&self, data: &mut [F]) {
        assert_eq!(data.len(), 2 * self.m);
        let (re, im) = data.split_at_mut(self.m);
        self.inverse(re, im, 0, self.m, self.m);
    }
}

#[cfg(test)]
mod tests {
    use poulpy_hal::{api::NegacyclicFFTNew, test_suite::reim::test_negacyclic_fft};

    use super::EncodingFft;

    #[test]
    fn transform_meets_the_hal_contract() {
        for log_m in 0..=8 {
            test_negacyclic_fft(&EncodingFft::<f64>::new(1 << log_m));
        }
    }
}
