use criterion::{Criterion, criterion_group, criterion_main};
use poulpy_core::layouts::{Base2K, Degree, Dnum, Dsize, GLWELayout, GLWESwitchingKeyLayout, Rank, TorusPrecision};

fn bench_glwe_keyswitch(c: &mut Criterion) {
    let p = &poulpy_bench::params::BenchParams::get().core;
    let n = p.n;
    let base2k = p.base2k;
    let k = p.k;
    let dsize = p.dsize;
    let (dnum, k_aux) = poulpy_bench::params::key_dnum_k_aux(k + dsize * base2k, base2k, dsize);

    let glwe_in = GLWELayout {
        n: Degree(n),
        base2k: Base2K(base2k),
        k: TorusPrecision(k),
        rank: Rank(p.rank),
    };
    let glwe_out = GLWELayout {
        n: Degree(n),
        base2k: Base2K(base2k),
        k: TorusPrecision(k),
        rank: Rank(p.rank),
    };
    let gglwe = GLWESwitchingKeyLayout {
        n: Degree(n),
        base2k: Base2K(base2k),
        k_aux: TorusPrecision(k_aux),
        rank_in: Rank(p.rank),
        rank_out: Rank(p.rank),
        dnum: Dnum(dnum),
        dsize: Dsize(dsize),
    };

    poulpy_bench::for_each_backend!(
        poulpy_bench::bench_suite::core::keyswitch::bench_glwe_keyswitch,
        &glwe_in, &glwe_out, &gglwe;
        c
    );
}

criterion_group! {
    name = benches;
    config = poulpy_bench::criterion_config();
    targets = bench_glwe_keyswitch
}
criterion_main!(benches);
