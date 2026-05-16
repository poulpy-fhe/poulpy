//! Standard benchmark binary for library-wide regression tracking.
//!
//! Run with:
//!   cargo bench -p poulpy-bench --bench standard
//!   RUSTFLAGS="-C target-feature=+avx2,+fma" \
//!     cargo bench -p poulpy-bench --bench standard --features enable-avx
//!   RUSTFLAGS="-C target-feature=+avx512f,+avx512ifma,+avx512vl" \
//!     cargo bench -p poulpy-bench --bench standard --features enable-ifma
//!   RUSTFLAGS="-C target-feature=+avx2,+fma,+avx512f,+avx512ifma,+avx512vl" \
//!     cargo bench -p poulpy-bench --bench standard --features "enable-avx,enable-ifma"
//!
//! Save and compare baselines:
//!   cargo bench -p poulpy-bench --bench standard -- --save-baseline v0.4.4
//!   cargo bench -p poulpy-bench --bench standard -- --baseline v0.4.4
//!
//! This binary covers every layer of the stack with a single representative
//! parameter set. It is intentionally separate from the per-subsystem bench
//! binaries, which sweep parameter ranges for detailed profiling.

use criterion::{Criterion, criterion_group, criterion_main};
use poulpy_bin_fhe::blind_rotation::CGGI;
use poulpy_core::layouts::{
    Base2K, Degree, Dnum, Dsize, GGSWLayout, GLWEAutomorphismKeyLayout, GLWELayout, GLWESwitchingKeyLayout, Rank, TorusPrecision,
};

fn p() -> &'static poulpy_bench::params::BenchParams {
    poulpy_bench::params::BenchParams::get()
}

// ── Layer 1: HAL – FFT-domain ────────────────────────────────────────────────

fn std_vec_znx_dft_apply(c: &mut Criterion) {
    poulpy_bench::for_each_backend!(poulpy_bench::bench_suite::hal::vec_znx_dft::bench_vec_znx_dft_apply, &p().hal; c);
}

fn std_vec_znx_idft_apply(c: &mut Criterion) {
    poulpy_bench::for_each_backend!(poulpy_bench::bench_suite::hal::vec_znx_dft::bench_vec_znx_idft_apply, &p().hal; c);
}

fn std_vmp_apply_dft_to_dft(c: &mut Criterion) {
    poulpy_bench::for_each_backend!(poulpy_bench::bench_suite::hal::vmp::bench_vmp_apply_dft_to_dft, &p().vmp; c);
}

fn std_svp_apply_dft_to_dft(c: &mut Criterion) {
    poulpy_bench::for_each_backend!(poulpy_bench::bench_suite::hal::svp::bench_svp_apply_dft_to_dft, &p().hal; c);
}

// ── Layer 1: HAL – coefficient domain (all backends) ────────────────────────

fn std_vec_znx_add_into(c: &mut Criterion) {
    poulpy_bench::for_each_backend!(poulpy_bench::bench_suite::hal::vec_znx::bench_vec_znx_add_into; c);
}

fn std_vec_znx_normalize(c: &mut Criterion) {
    poulpy_bench::for_each_backend!(poulpy_bench::bench_suite::hal::vec_znx::bench_vec_znx_normalize; c);
}

fn std_vec_znx_big_add_into(c: &mut Criterion) {
    poulpy_bench::for_each_backend!(poulpy_bench::bench_suite::hal::vec_znx_big::bench_vec_znx_big_add_into, &p().hal; c);
}

fn std_vec_znx_big_normalize(c: &mut Criterion) {
    poulpy_bench::for_each_backend!(poulpy_bench::bench_suite::hal::vec_znx_big::bench_vec_znx_normalize, &p().hal; c);
}

// ── Layer 2: Core – encryption ───────────────────────────────────────────────

fn std_glwe_encrypt_sk(c: &mut Criterion) {
    let cp = &p().core;
    let infos = GLWELayout {
        n: Degree(cp.n),
        base2k: Base2K(cp.base2k),
        k: TorusPrecision(cp.k),
        rank: Rank(cp.rank),
    };
    poulpy_bench::for_each_backend!(poulpy_bench::bench_suite::core::encryption::bench_glwe_encrypt_sk, &infos; c);
}

fn std_ggsw_encrypt_sk(c: &mut Criterion) {
    let cp = &p().core;
    let infos = GGSWLayout {
        n: Degree(cp.n),
        base2k: Base2K(cp.base2k),
        k: TorusPrecision(cp.k),
        rank: Rank(cp.rank),
        dnum: Dnum(cp.dnum()),
        dsize: Dsize(cp.dsize),
    };
    poulpy_bench::for_each_backend!(poulpy_bench::bench_suite::core::encryption::bench_ggsw_encrypt_sk, &infos; c);
}

// ── Layer 2: Core – ciphertext operations ───────────────────────────────────

fn std_glwe_external_product(c: &mut Criterion) {
    let cp = &p().core;
    let infos = GGSWLayout {
        n: Degree(cp.n),
        base2k: Base2K(cp.base2k),
        k: TorusPrecision(cp.k),
        rank: Rank(cp.rank),
        dnum: Dnum(cp.dnum()),
        dsize: Dsize(cp.dsize),
    };
    poulpy_bench::for_each_backend!(
        poulpy_bench::bench_suite::core::external_product::bench_glwe_external_product_assign,
        &infos;
        c
    );
}

fn std_glwe_automorphism(c: &mut Criterion) {
    let cp = &p().core;
    let glwe_infos = GLWELayout {
        n: Degree(cp.n),
        base2k: Base2K(cp.base2k),
        k: TorusPrecision(cp.k),
        rank: Rank(cp.rank),
    };
    let atk_infos = GLWEAutomorphismKeyLayout {
        n: Degree(cp.n),
        base2k: Base2K(cp.base2k),
        k: TorusPrecision(cp.k),
        rank: Rank(cp.rank),
        dnum: Dnum(cp.dnum()),
        dsize: Dsize(cp.dsize),
    };
    poulpy_bench::for_each_backend!(
        poulpy_bench::bench_suite::core::automorphism::bench_glwe_automorphism,
        &glwe_infos, &atk_infos, 3;
        c
    );
}

fn std_glwe_keyswitch(c: &mut Criterion) {
    let cp = &p().core;
    let glwe = GLWELayout {
        n: Degree(cp.n),
        base2k: Base2K(cp.base2k),
        k: TorusPrecision(cp.k),
        rank: Rank(cp.rank),
    };
    let ksk_infos = GLWESwitchingKeyLayout {
        n: Degree(cp.n),
        base2k: Base2K(cp.base2k),
        k: TorusPrecision(cp.k + cp.base2k * cp.dsize),
        rank_in: Rank(cp.rank),
        rank_out: Rank(cp.rank),
        dnum: Dnum(cp.dnum()),
        dsize: Dsize(cp.dsize),
    };
    poulpy_bench::for_each_backend!(
        poulpy_bench::bench_suite::core::keyswitch::bench_glwe_keyswitch,
        &glwe, &glwe, &ksk_infos;
        c
    );
}

fn std_glwe_decrypt(c: &mut Criterion) {
    let cp = &p().core;
    let infos = GLWELayout {
        n: Degree(cp.n),
        base2k: Base2K(cp.base2k),
        k: TorusPrecision(cp.k),
        rank: Rank(cp.rank),
    };
    poulpy_bench::for_each_backend!(poulpy_bench::bench_suite::core::decryption::bench_glwe_decrypt, &infos; c);
}

// ── Layer 3: Scheme ──────────────────────────────────────────────────────────

fn std_blind_rotate(c: &mut Criterion) {
    poulpy_bench::bench_suite::schemes::blind_rotation::bench_blind_rotate::<poulpy_cpu_ref::FFT64Ref, CGGI>(c, "fft64-ref");
    #[cfg(all(feature = "enable-avx", target_arch = "x86_64"))]
    poulpy_bench::bench_suite::schemes::blind_rotation::bench_blind_rotate::<poulpy_cpu_avx::FFT64Avx, CGGI>(c, "fft64-avx");
    #[cfg(all(feature = "enable-avx512f", target_arch = "x86_64"))]
    poulpy_bench::bench_suite::schemes::blind_rotation::bench_blind_rotate::<poulpy_cpu_avx512::FFT64Avx512, CGGI>(
        c,
        "fft64-avx512",
    );
}

fn std_circuit_bootstrapping(c: &mut Criterion) {
    poulpy_bench::bench_suite::schemes::circuit_bootstrapping::bench_circuit_bootstrapping::<poulpy_cpu_ref::FFT64Ref, CGGI>(
        c,
        "fft64-ref",
    );
    #[cfg(all(feature = "enable-avx", target_arch = "x86_64"))]
    poulpy_bench::bench_suite::schemes::circuit_bootstrapping::bench_circuit_bootstrapping::<poulpy_cpu_avx::FFT64Avx, CGGI>(
        c,
        "fft64-avx",
    );
    #[cfg(all(feature = "enable-avx512f", target_arch = "x86_64"))]
    poulpy_bench::bench_suite::schemes::circuit_bootstrapping::bench_circuit_bootstrapping::<poulpy_cpu_avx512::FFT64Avx512, CGGI>(
        c,
        "fft64-avx512",
    );
}

criterion_group! {
    name = benches;
    config = poulpy_bench::criterion_config();
    targets = // Layer 1 – HAL FFT-domain,
    std_vec_znx_dft_apply,
    std_vec_znx_idft_apply,
    std_vmp_apply_dft_to_dft,
    std_svp_apply_dft_to_dft,
    // Layer 1 – HAL coefficient-domain,
    std_vec_znx_add_into,
    std_vec_znx_normalize,
    std_vec_znx_big_add_into,
    std_vec_znx_big_normalize,
    // Layer 2 – Core,
    std_glwe_encrypt_sk,
    std_ggsw_encrypt_sk,
    std_glwe_external_product,
    std_glwe_automorphism,
    std_glwe_keyswitch,
    std_glwe_decrypt,
    // Layer 3 – Scheme,
    std_blind_rotate,
    std_circuit_bootstrapping
}
criterion_main!(benches);
