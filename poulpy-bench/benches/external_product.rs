use criterion::{Criterion, criterion_group, criterion_main};
use poulpy_core::layouts::{Dnum, Dsize, GGSWLayout, GLWELayout};

fn bench_glwe_external_product(c: &mut Criterion) {
    let p = &poulpy_bench::params::BenchParams::get().core;
    let n = p.n;
    let base2k = p.base2k;
    let (dnum, k_aux) = poulpy_bench::params::key_dnum_k_aux(p.k, base2k, p.dsize);
    let dsize = Dsize(p.dsize);

    let glwe_infos = GLWELayout {
        n: n.into(),
        base2k: base2k.into(),
        k: p.k.into(),
        rank: p.rank.into(),
    };
    let ggsw_infos = GGSWLayout {
        n: n.into(),
        base2k: base2k.into(),
        k_aux: k_aux.into(),
        rank: p.rank.into(),
        dnum: Dnum(dnum),
        dsize,
    };
    poulpy_bench::for_each_backend!(
        poulpy_bench::bench_suite::core::external_product::bench_glwe_external_product,
        &glwe_infos, &ggsw_infos;
        c
    );
}

fn bench_glwe_external_product_assign(c: &mut Criterion) {
    let p = &poulpy_bench::params::BenchParams::get().core;
    let n = p.n;
    let base2k = p.base2k;
    let (dnum, k_aux) = poulpy_bench::params::key_dnum_k_aux(p.k, base2k, p.dsize);
    let dsize = Dsize(p.dsize);

    let infos = GGSWLayout {
        n: n.into(),
        base2k: base2k.into(),
        k_aux: k_aux.into(),
        rank: p.rank.into(),
        dnum: Dnum(dnum),
        dsize,
    };
    poulpy_bench::for_each_backend!(
        poulpy_bench::bench_suite::core::external_product::bench_glwe_external_product_assign,
        &infos;
        c
    );
}

criterion_group! {
    name = benches;
    config = poulpy_bench::criterion_config();
    targets = bench_glwe_external_product,
    bench_glwe_external_product_assign
}
criterion_main!(benches);
