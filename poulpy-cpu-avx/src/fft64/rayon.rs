//! Rayon-scheduled wrapper for the AVX2 FFT64 backend.

use poulpy_hal::layouts::{Module, VecZnxDftBackendMut, VecZnxDftBackendRef};

use super::FFT64AvxRayon;
use crate::FFT64Avx;

fn dft_automorphism(
    _module: &Module<FFT64AvxRayon>,
    plan: &<FFT64Avx as poulpy_hal::oep::HalVecZnxDftImpl<FFT64Avx>>::AutomorphismPlan,
    res: &mut VecZnxDftBackendMut<'_, FFT64AvxRayon>,
    res_col: usize,
    a: &VecZnxDftBackendRef<'_, FFT64AvxRayon>,
    a_col: usize,
) {
    super::fft64_vec_znx_dft_automorphism_avx::<FFT64AvxRayon>(plan, res, res_col, a, a_col);
}

poulpy_cpu_rayon::impl_fft64_rayon_backend!(FFT64AvxRayon, FFT64Avx, dft_automorphism);

impl poulpy_cpu_rayon::RayonTuning for FFT64AvxRayon {
    const COEFF_MIN_LEN: usize = 1 << 15;
    const COEFF_MIN_TASK: usize = 1 << 13;
    const NORMALIZE_MIN_TASK: usize = 1 << 12;
}

impl poulpy_hal::execution::ScratchWorkers for FFT64AvxRayon {
    const PREPARE: usize = 8;
    const APPLY: usize = 8;
    const VMP: usize = 8;
    const IDFT: usize = 8;
}
