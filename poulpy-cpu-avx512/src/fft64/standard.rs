//! Standard-ring items of [`FFT64Avx512`].

use poulpy_cpu_portable::reference::{
    fft64::ring_arith::Fft64RingArith,
    znx::{ZnxAutomorphism, ZnxAutomorphismRotate, standard::znx_automorphism_ref},
};

use super::FFT64Avx512;
use crate::znx_avx512::{znx_automorphism_avx512, znx_automorphism_rotate_avx512};

impl poulpy_hal::layouts::MaxBase2k for FFT64Avx512 {
    fn max_base2k(n: usize, products: usize, failure_bits: usize, squaring: bool) -> Option<usize> {
        Some(poulpy_hal::layouts::max_base2k_fft64::<Self>(
            n,
            products,
            failure_bits,
            squaring,
        ))
    }
}

impl ZnxAutomorphism for FFT64Avx512 {
    #[inline(always)]
    fn znx_automorphism(p: i64, res: &mut [i64], a: &[i64]) {
        unsafe {
            znx_automorphism_avx512(p, res, a);
        }
    }

    #[inline(always)]
    fn znx_automorphism_i128(p: i64, res: &mut [i128], a: &[i128]) {
        znx_automorphism_ref(p, res, a)
    }
}

impl ZnxAutomorphismRotate for FFT64Avx512 {
    #[inline(always)]
    fn znx_automorphism_rotate(p: i64, k: i64, res: &mut [i64], a: &[i64]) {
        unsafe {
            znx_automorphism_rotate_avx512(p, k, res, a);
        }
    }
}

impl Fft64RingArith for FFT64Avx512 {
    poulpy_cpu_portable::fft64_ring_arith_standard!();
}
