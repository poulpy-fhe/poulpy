//! Conjugate invariant ring items of [`FFT64Avx512`].

use poulpy_cpu_ref::reference::{
    fft64::ring_arith::Fft64RingArith,
    znx::{ZnxAutomorphism, conjugate_invariant::znx_automorphism_ref},
};
use poulpy_hal::layouts::ConjugateInvariant;

use super::FFT64Avx512;

/// No failure model: a square's constant coefficient has a large positive mean on this ring.
impl poulpy_hal::layouts::MaxBase2k for FFT64Avx512<ConjugateInvariant> {
    fn max_base2k(_n: usize, _products: usize, _failure_bits: usize, _squaring: bool) -> Option<usize> {
        None
    }
}

impl ZnxAutomorphism for FFT64Avx512<ConjugateInvariant> {
    #[inline(always)]
    fn znx_automorphism(p: i64, res: &mut [i64], a: &[i64]) {
        znx_automorphism_ref(p, res, a)
    }

    #[inline(always)]
    fn znx_automorphism_i128(p: i64, res: &mut [i128], a: &[i128]) {
        znx_automorphism_ref(p, res, a)
    }
}

impl Fft64RingArith for FFT64Avx512<ConjugateInvariant> {
    poulpy_cpu_ref::fft64_ring_arith_ci!();
}

#[cfg(all(test, target_feature = "avx512f"))]
mod tests {
    use super::FFT64Avx512;

    #[test]
    fn real_arithmetic_parity() {
        poulpy_cpu_ref::test_suite::conjugate_invariant::test_conjugate_invariant_fft_arithmetic::<FFT64Avx512>();
    }
}
