//! Standard-ring items of [`FFT64Neon`].

use poulpy_cpu_portable::kernels::{
    fft64::ring_arith::Fft64RingArith,
    znx::{ZnxAutomorphism, ZnxAutomorphismRotate, standard::znx_automorphism_portable},
};

use super::FFT64Neon;
#[cfg(target_arch = "aarch64")]
use crate::neon::znx::{znx_automorphism_neon as kn_automorphism, znx_automorphism_rotate_neon as kn_automorphism_rotate};
#[cfg(not(target_arch = "aarch64"))]
use poulpy_cpu_portable::kernels::znx::{
    standard::znx_automorphism_portable as kn_automorphism, znx_automorphism_rotate_portable as kn_automorphism_rotate,
};

impl poulpy_hal::layouts::MaxBase2k for FFT64Neon {
    fn max_base2k(n: usize, products: usize, failure_bits: usize, squaring: bool) -> Option<usize> {
        Some(poulpy_hal::layouts::max_base2k_fft64::<Self>(
            n,
            products,
            failure_bits,
            squaring,
        ))
    }
}

impl ZnxAutomorphism for FFT64Neon {
    #[inline(always)]
    fn znx_automorphism(p: i64, res: &mut [i64], a: &[i64]) {
        kn_automorphism(p, res, a);
    }

    #[inline(always)]
    fn znx_automorphism_i128(p: i64, res: &mut [i128], a: &[i128]) {
        znx_automorphism_portable(p, res, a)
    }
}

impl ZnxAutomorphismRotate for FFT64Neon {
    #[inline(always)]
    fn znx_automorphism_rotate(p: i64, k: i64, res: &mut [i64], a: &[i64]) {
        kn_automorphism_rotate(p, k, res, a);
    }
}

impl Fft64RingArith for FFT64Neon {
    poulpy_cpu_portable::fft64_ring_arith_standard!();
}
