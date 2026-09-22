//! Rayon-scheduled wrapper for the NEON FFT64 backend.

use crate::FFT64NeonBackend;
#[cfg(feature = "enable-rayon")]
use crate::FFT64NeonRayonBackend;
use poulpy_cpu_ref::ring::CpuRing;

use poulpy_hal::layouts::{DataView, DataViewMut, Module, VecZnxDft, VecZnxDftBackendMut, VecZnxDftBackendRef};
use poulpy_hal::oep::HalVecZnxDftImpl;

fn dft_automorphism<R: CpuRing>(
    module: &Module<FFT64NeonRayonBackend<R>>,
    plan: &<FFT64NeonBackend<R> as HalVecZnxDftImpl>::AutomorphismPlan,
    res: &mut VecZnxDftBackendMut<'_, FFT64NeonRayonBackend<R>>,
    res_col: usize,
    a: &VecZnxDftBackendRef<'_, FFT64NeonRayonBackend<R>>,
    a_col: usize,
) {
    let res_shape = res.shape();
    FFT64NeonBackend::<R>::vec_znx_dft_automorphism_with_plan(
        module.reinterpret(),
        plan,
        &mut VecZnxDft::from_shape(&mut **res.data_mut(), res_shape),
        res_col,
        &VecZnxDft::from_shape(&**a.data(), a.shape()),
        a_col,
    );
}

poulpy_cpu_rayon::impl_fft64_rayon_backend!(R, FFT64NeonRayonBackend<R>, FFT64NeonBackend<R>, dft_automorphism);

impl<R: CpuRing> poulpy_cpu_rayon::RayonTuning for FFT64NeonRayonBackend<R> {
    const COEFF_MIN_LEN: usize = 1 << 15;
    const COEFF_MIN_TASK: usize = 1 << 13;
    const NORMALIZE_MIN_TASK: usize = 1 << 12;
}

impl<R: CpuRing> poulpy_hal::execution::ScratchWorkers for FFT64NeonRayonBackend<R> {
    const PREPARE: usize = 8;
    const APPLY: usize = 8;
    const VMP: usize = 8;
    const IDFT: usize = 8;
}
