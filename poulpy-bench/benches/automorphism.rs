use criterion::{Criterion, criterion_group, criterion_main};
use poulpy_core::layouts::{Dnum, Dsize, GLWEAutomorphismKeyLayout, GLWELayout};

fn bench_glwe_automorphism(c: &mut Criterion) {
    let p = &poulpy_bench::params::BenchParams::get().core;
    let n = p.n;
    let base2k = p.base2k;
    let k_ct = p.k;
    let (dnum, k_aux) = poulpy_bench::params::key_dnum_k_aux(p.k, base2k, p.dsize);
    let dsize = Dsize(p.dsize);

    let glwe_infos = GLWELayout {
        n: n.into(),
        base2k: base2k.into(),
        k: k_ct.into(),
        rank: p.rank.into(),
    };
    let atk_infos = GLWEAutomorphismKeyLayout {
        n: n.into(),
        base2k: base2k.into(),
        k_aux: k_aux.into(),
        rank: p.rank.into(),
        dnum: Dnum(dnum),
        dsize,
    };

    poulpy_bench::for_each_backend!(
        poulpy_bench::bench_suite::core::automorphism::bench_glwe_automorphism,
        &glwe_infos, &atk_infos, 3;
        c
    );
}

criterion_group! {
    name = benches;
    config = poulpy_bench::criterion_config();
    targets = bench_glwe_automorphism
}
criterion_main!(benches);
