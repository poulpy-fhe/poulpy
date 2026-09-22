//! Rayon-scheduled wrapper for the AVX2 FFT64 backend.

use crate::FFT64AvxBackend;
#[cfg(feature = "enable-rayon")]
use crate::FFT64AvxRayonBackend;
use poulpy_cpu_ref::ring::CpuRing;

use poulpy_hal::layouts::{Module, VecZnxDftBackendMut, VecZnxDftBackendRef};

fn dft_automorphism<R: CpuRing>(
    _module: &Module<FFT64AvxRayonBackend<R>>,
    plan: &<FFT64AvxBackend<R> as poulpy_hal::oep::HalVecZnxDftImpl>::AutomorphismPlan,
    res: &mut VecZnxDftBackendMut<'_, FFT64AvxRayonBackend<R>>,
    res_col: usize,
    a: &VecZnxDftBackendRef<'_, FFT64AvxRayonBackend<R>>,
    a_col: usize,
) {
    super::fft64_vec_znx_dft_automorphism_avx::<FFT64AvxRayonBackend<R>>(plan, res, res_col, a, a_col);
}

poulpy_cpu_rayon::impl_fft64_rayon_backend!(R, FFT64AvxRayonBackend<R>, FFT64AvxBackend<R>, dft_automorphism);

impl<R: CpuRing> poulpy_cpu_rayon::RayonTuning for FFT64AvxRayonBackend<R> {
    const COEFF_MIN_LEN: usize = 1 << 15;
    const COEFF_MIN_TASK: usize = 1 << 13;
    const NORMALIZE_MIN_TASK: usize = 1 << 12;
}

impl<R: CpuRing> poulpy_hal::execution::ScratchWorkers for FFT64AvxRayonBackend<R> {
    const PREPARE: usize = 8;
    const APPLY: usize = 8;
    const VMP: usize = 8;
    const IDFT: usize = 8;
}
