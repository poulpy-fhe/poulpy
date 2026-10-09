//! Standard benchmark binary: a small, representative cross-section of the
//! HAL, core and CKKS operations, swept over `log_n` in {13, 14, 15}, on
//! every runtime enabled by a feature. See
//! `poulpy_bench::{hal,core,schemes}::suites::standard` for the shared
//! implementation. Every timed closure ends with a device synchronization.

use criterion::{Criterion, criterion_group, criterion_main};

#[allow(unused_macros)]
macro_rules! standard {
    ($c:expr, $backend:ty) => {{
        poulpy_bench::hal::suites::standard::bench_hal_ckks::<$backend>($c);
        poulpy_bench::core::suites::standard::bench_core_ckks::<$backend>($c);
        poulpy_bench::schemes::suites::standard::bench_ckks::<$backend>($c);
    }};
}

#[allow(unused_variables)]
fn runtimes(c: &mut Criterion) {
    #[cfg(feature = "enable-cuda")]
    standard!(c, poulpy_gpu_cubecl::NTT4x30Cuda);
    #[cfg(feature = "enable-hip")]
    standard!(c, poulpy_gpu_cubecl::NTT4x30Hip);
    #[cfg(any(feature = "enable-metal", feature = "enable-vulkan", feature = "enable-wgpu"))]
    standard!(c, poulpy_gpu_cubecl::NTT4x30Wgpu);
}

criterion_group! {
    name = benches;
    config = poulpy_bench::criterion_config();
    targets = runtimes
}
criterion_main!(benches);
