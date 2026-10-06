//! Conjugate invariant ring items of [`FFT64Neon`].

use poulpy_cpu_portable::kernels::{
    fft64::ring_arith::Fft64RingArith,
    znx::{ZnxAutomorphism, conjugate_invariant::znx_automorphism_portable},
};
use poulpy_hal::layouts::ConjugateInvariant;

use super::FFT64Neon;

/// No failure model: a square's constant coefficient has a large positive mean on this ring.
impl poulpy_hal::layouts::MaxBase2k for FFT64Neon<ConjugateInvariant> {
    fn max_base2k(_n: usize, _products: usize, _failure_bits: usize, _squaring: bool) -> Option<usize> {
        None
    }
}

impl ZnxAutomorphism for FFT64Neon<ConjugateInvariant> {
    #[inline(always)]
    fn znx_automorphism(p: i64, res: &mut [i64], a: &[i64]) {
        znx_automorphism_portable(p, res, a)
    }

    #[inline(always)]
    fn znx_automorphism_i128(p: i64, res: &mut [i128], a: &[i128]) {
        znx_automorphism_portable(p, res, a)
    }
}

impl Fft64RingArith for FFT64Neon<ConjugateInvariant> {
    poulpy_cpu_portable::fft64_ring_arith_ci!();
}

#[cfg(all(test, target_arch = "aarch64"))]
mod tests {
    use super::FFT64Neon;

    #[test]
    fn real_arithmetic_parity() {
        poulpy_cpu_portable::test_suite::conjugate_invariant::test_conjugate_invariant_fft_arithmetic::<FFT64Neon>();
    }
}
