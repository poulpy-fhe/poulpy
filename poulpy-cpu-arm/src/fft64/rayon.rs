//! Rayon-scheduled wrapper for the NEON FFT64 backend.

use poulpy_hal::layouts::{DataView, DataViewMut, Module, VecZnxDft, VecZnxDftBackendMut, VecZnxDftBackendRef};
use poulpy_hal::oep::HalVecZnxDftImpl;

use super::FFT64NeonRayon;
use crate::FFT64Neon;

fn dft_automorphism(
    module: &Module<FFT64NeonRayon>,
    plan: &<FFT64Neon as HalVecZnxDftImpl<FFT64Neon>>::AutomorphismPlan,
    res: &mut VecZnxDftBackendMut<'_, FFT64NeonRayon>,
    res_col: usize,
    a: &VecZnxDftBackendRef<'_, FFT64NeonRayon>,
    a_col: usize,
) {
    let (n, cols, size) = (res.n(), res.cols(), res.size());
    <FFT64Neon as HalVecZnxDftImpl<FFT64Neon>>::vec_znx_dft_automorphism_with_plan(
        module.reinterpret(),
        plan,
        &mut VecZnxDft::from_data(&mut **res.data_mut(), n, cols, size),
        res_col,
        &VecZnxDft::from_data(&**a.data(), a.n(), a.cols(), a.size()),
        a_col,
    );
}

poulpy_cpu_rayon::impl_fft64_rayon_backend!(FFT64NeonRayon, FFT64Neon, dft_automorphism);

/// Inherited from the x86 measurements; not yet measured on AArch64.
impl poulpy_hal::execution::ScratchWorkers for FFT64NeonRayon {
    const PREPARE: usize = 4;
    const APPLY: usize = 8;
    const VMP: usize = 4;
    const IDFT: usize = 8;
}
