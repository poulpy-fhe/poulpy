//! Conjugate invariant ring items of [`FFT64Avx`].

use poulpy_cpu_portable::kernels::{
    fft64::ring_arith::Fft64RingArith,
    znx::{ZnxAutomorphism, conjugate_invariant::znx_automorphism_portable},
};
use poulpy_hal::layouts::ConjugateInvariant;

use super::FFT64Avx;

impl ZnxAutomorphism for FFT64Avx<ConjugateInvariant> {
    #[inline(always)]
    fn znx_automorphism(p: i64, res: &mut [i64], a: &[i64]) {
        znx_automorphism_portable(p, res, a)
    }

    #[inline(always)]
    fn znx_automorphism_i128(p: i64, res: &mut [i128], a: &[i128]) {
        znx_automorphism_portable(p, res, a)
    }
}

impl Fft64RingArith for FFT64Avx<ConjugateInvariant> {
    poulpy_cpu_portable::fft64_ring_arith_ci!();
}
