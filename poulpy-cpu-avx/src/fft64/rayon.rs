//! Rayon-scheduled wrapper for the AVX2 FFT64 backend.

use poulpy_hal::layouts::Ring;

use super::FFT64AvxRayon;

mod standard {
    use poulpy_cpu_portable::kernels::{fft64::ring_arith::Fft64RingArith, znx::ZnxAutomorphismRotate};

    use super::FFT64AvxRayon;
    use crate::FFT64Avx;

    poulpy_cpu_rayon::impl_fft64_rayon_backend!(FFT64AvxRayon, FFT64Avx);

    unsafe impl poulpy_hal::oep::HalVecZnxMonomialImpl for FFT64AvxRayon {
        poulpy_cpu_portable::hal_impl_vec_znx_monomial!();
    }

    unsafe impl poulpy_hal::oep::HalVecZnxCIImpl for FFT64AvxRayon {
        poulpy_cpu_portable::hal_impl_vec_znx_ci!();
    }

    impl ZnxAutomorphismRotate for FFT64AvxRayon {
        #[inline(always)]
        fn znx_automorphism_rotate(p: i64, k: i64, res: &mut [i64], a: &[i64]) {
            <FFT64Avx as ZnxAutomorphismRotate>::znx_automorphism_rotate(p, k, res, a)
        }
    }

    impl Fft64RingArith for FFT64AvxRayon {
        poulpy_cpu_portable::fft64_ring_arith_standard!();
    }
}

mod conjugate_invariant {
    use poulpy_cpu_portable::kernels::fft64::ring_arith::Fft64RingArith;
    use poulpy_hal::layouts::ConjugateInvariant;

    use super::FFT64AvxRayon;
    use crate::FFT64Avx;

    poulpy_cpu_rayon::impl_fft64_rayon_backend!(FFT64AvxRayon<ConjugateInvariant>, FFT64Avx<ConjugateInvariant>);

    impl Fft64RingArith for FFT64AvxRayon<ConjugateInvariant> {
        poulpy_cpu_portable::fft64_ring_arith_ci!();
    }
}

impl<R: Ring> poulpy_cpu_rayon::RayonTuning for FFT64AvxRayon<R> {
    const COEFF_MIN_LEN: usize = 1 << 15;
    const COEFF_MIN_TASK: usize = 1 << 13;
    const NORMALIZE_MIN_TASK: usize = 1 << 12;
}

impl<R: Ring> poulpy_hal::execution::ScratchWorkers for FFT64AvxRayon<R> {
    const PREPARE: usize = 8;
    const APPLY: usize = 8;
    const VMP: usize = 8;
    const IDFT: usize = 8;
}
