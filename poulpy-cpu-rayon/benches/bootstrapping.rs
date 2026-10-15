//! CKKS bootstrapping presets on the portable NTT backend, serial and Rayon.
use criterion::{criterion_group, criterion_main};
use poulpy_bench::schemes::suites::bench_ckks_bootstrapping;
use poulpy_cpu_portable::NTT4x30Portable;
use poulpy_cpu_rayon::NTT4x30PortableRayon;

criterion_group! {
    name = benches;
    config = poulpy_bench::criterion_config();
    targets = bench_ckks_bootstrapping::<NTT4x30Portable, 52>, bench_ckks_bootstrapping::<NTT4x30PortableRayon, 52>
}
criterion_main!(benches);
