//! Standard benchmark binary: a small, representative cross-section of the
//! HAL, core and CKKS operations, swept over `log_n` in {13, 14, 15}, on the
//! wgpu runtime. See `poulpy_bench::{hal,core,schemes}::suites::standard` for
//! the shared implementation. Every timed closure ends with a device
//! synchronization.

use criterion::{criterion_group, criterion_main};
use poulpy_bench::schemes::suites::standard::bench_ckks;

use poulpy_gpu_cubecl::NTT4x30Wgpu as Ntt;

criterion_group! {
    name = benches;
    config = poulpy_bench::criterion_config();
    targets =
     bench_ckks::<Ntt>
}

criterion_main!(benches);
