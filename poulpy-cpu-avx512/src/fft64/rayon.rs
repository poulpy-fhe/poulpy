//! Rayon-scheduled wrapper for the AVX-512 FFT64 backend.

use crate::FFT64Avx512Backend;
#[cfg(feature = "enable-rayon")]
use crate::FFT64Avx512RayonBackend;
use poulpy_cpu_ref::ring::CpuRing;

use poulpy_hal::layouts::{Module, VecZnxDftBackendMut, VecZnxDftBackendRef};

fn dft_automorphism<R: CpuRing>(
    _module: &Module<FFT64Avx512RayonBackend<R>>,
    plan: &<FFT64Avx512Backend<R> as poulpy_hal::oep::HalVecZnxDftImpl>::AutomorphismPlan,
    res: &mut VecZnxDftBackendMut<'_, FFT64Avx512RayonBackend<R>>,
    res_col: usize,
    a: &VecZnxDftBackendRef<'_, FFT64Avx512RayonBackend<R>>,
    a_col: usize,
) {
    super::fft64_vec_znx_dft_automorphism_avx512::<FFT64Avx512RayonBackend<R>>(plan, res, res_col, a, a_col);
}

poulpy_cpu_rayon::impl_fft64_rayon_backend!(R, FFT64Avx512RayonBackend<R>, FFT64Avx512Backend<R>, dft_automorphism);

impl<R: CpuRing> poulpy_cpu_rayon::RayonTuning for FFT64Avx512RayonBackend<R> {
    const COEFF_MIN_LEN: usize = 1 << 15;
    const COEFF_MIN_TASK: usize = 1 << 13;
    const NORMALIZE_MIN_TASK: usize = 1 << 12;
}

impl<R: CpuRing> poulpy_hal::execution::ScratchWorkers for FFT64Avx512RayonBackend<R> {
    const PREPARE: usize = 8;
    const APPLY: usize = 8;
    const VMP: usize = 8;
    const IDFT: usize = 8;
}
