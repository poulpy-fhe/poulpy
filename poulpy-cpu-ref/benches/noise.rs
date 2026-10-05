//! Unified noise sampling, including the encryption performance gate at N=2^14.
use std::hint::black_box;

use criterion::{Criterion, Throughput, criterion_group, criterion_main};
use poulpy_core::{Noise, VecZnxAddNoise};
use poulpy_cpu_ref::FFT64Ref;
use poulpy_hal::{
    layouts::{Module, VecZnxToBackendMut},
    source::Source,
};

fn noise(c: &mut Criterion) {
    let n = 1 << 14;
    let module = Module::<FFT64Ref>::new(n as u64);
    let mut destination = module.vec_znx_alloc(n, 1, 10);
    let mut source = Source::new([42; 32]);
    let mut group = c.benchmark_group("FFT64Ref/noise");
    group.throughput(Throughput::Elements(n as u64));
    for (name, noise) in [
        ("encryption", Noise::ENCRYPTION),
        ("gaussian_table_bound_60", Noise::Gaussian { sigma: 10.0, cutoff: 6 }),
        ("gaussian_rejection_bound_66", Noise::Gaussian { sigma: 11.0, cutoff: 6 }),
        (
            "gaussian_2_pow_128",
            Noise::Gaussian {
                sigma: 2f64.powi(128),
                cutoff: 6,
            },
        ),
        ("uniform_192_bits", Noise::Uniform { bits: 192 }),
    ] {
        group.bench_function(name, |b| {
            b.iter(|| {
                module.vec_znx_add_noise(
                    30,
                    300,
                    &mut VecZnxToBackendMut::<FFT64Ref>::to_backend_mut(&mut destination),
                    0,
                    noise,
                    &mut source,
                );
                black_box(&destination);
            })
        });
    }
    group.finish();
}

criterion_group!(benches, noise);
criterion_main!(benches);
