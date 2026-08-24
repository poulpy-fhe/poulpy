//! Rayon-scheduled wrapper for the AVX-512 FFT64 backend.

use poulpy_hal::layouts::{Module, VecZnxDftBackendMut, VecZnxDftBackendRef};

use super::FFT64Avx512Rayon;
use crate::FFT64Avx512;

fn dft_automorphism(
    _module: &Module<FFT64Avx512Rayon>,
    plan: &<FFT64Avx512 as poulpy_hal::oep::HalVecZnxDftImpl<FFT64Avx512>>::AutomorphismPlan,
    res: &mut VecZnxDftBackendMut<'_, FFT64Avx512Rayon>,
    res_col: usize,
    a: &VecZnxDftBackendRef<'_, FFT64Avx512Rayon>,
    a_col: usize,
) {
    super::fft64_vec_znx_dft_automorphism_avx512::<FFT64Avx512Rayon>(plan, res, res_col, a, a_col);
}

poulpy_cpu_rayon::impl_fft64_rayon_backend!(FFT64Avx512Rayon, FFT64Avx512, dft_automorphism);

impl poulpy_cpu_rayon::RayonTuning for FFT64Avx512Rayon {
    const COEFF_MIN_LEN: usize = 1 << 15;
    const COEFF_MIN_TASK: usize = 1 << 13;
    const NORMALIZE_MIN_TASK: usize = 1 << 12;
}

impl poulpy_hal::execution::ScratchWorkers for FFT64Avx512Rayon {
    const PREPARE: usize = 8;
    const APPLY: usize = 8;
    const VMP: usize = 8;
    const IDFT: usize = 8;
}
