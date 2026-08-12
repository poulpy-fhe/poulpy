//! Light HAL benchmark suite: just the DFT/iDFT ops and convolution, for a
//! fast sanity-check run (e.g. in CI) without paying for the full HAL sweep
//! in `standard.rs`. Needs neither `enable-core` nor `enable-ckks` since it
//! only touches HAL-layer ops.

use criterion::{Criterion, criterion_group, criterion_main, measurement::WallTime};

use poulpy_bench::{BenchOp, hal::vec_znx_dft};
use poulpy_bench::params::{default_bench_params_cnv, default_bench_params_hal};

type BE = poulpy_cpu_ref::NTT4x30Ref;

/// Caps the standard sweep down to `log_n` in `[10, 12]` for a fast CI-sized run.
const MAX_N: usize = 1 << 12;

fn hal_light(c: &mut Criterion) {
    use poulpy_bench::{bench_ops, hal::suites::{convolution_ops}};

    let hal_params: Vec<_> = default_bench_params_hal().into_iter().filter(|p| p.n <= MAX_N).collect();
    let cnv_params: Vec<_> = default_bench_params_cnv().into_iter().filter(|p| p.n <= MAX_N).collect();

    bench_ops(&[
        BenchOp { name: "vec_znx_dft_apply", runner: vec_znx_dft::runner_vec_znx_dft_apply::<BE, WallTime> },
        BenchOp { name: "vec_znx_idft_apply", runner: vec_znx_dft::runner_vec_znx_idft_apply::<BE, WallTime> },
    ], hal_params.as_slice(), "NTT4x30Ref", c);

    bench_ops(&convolution_ops::<BE, WallTime>(), cnv_params.as_slice(), "NTT4x30Ref", c);
}

criterion_group! {
    name = benches;
    config = poulpy_bench::criterion_config();
    targets = hal_light
}

criterion_main!(benches);
