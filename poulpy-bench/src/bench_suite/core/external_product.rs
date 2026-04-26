use poulpy_core::{
    DEFAULT_BOUND_XE, DEFAULT_SIGMA_XE, GGSWEncryptSk, GLWEEncryptSk, GLWEExternalProduct, ScratchTakeCore,
    layouts::{
        GGSW, GGSWInfos, GLWE, GLWEInfos, GLWESecret, GLWESecretPreparedFactory,
        prepared::{GGSWPrepared, GGSWPreparedFactory, GLWESecretPrepared},
    },
};
use poulpy_hal::{
    api::{ModuleNew, ScratchOwnedAlloc, ScratchOwnedBorrow},
    layouts::{Backend, DeviceBuf, Module, NoiseInfos, ScalarZnx, Scratch, ScratchOwned},
    source::Source,
};
use std::hint::black_box;

use criterion::Criterion;

pub fn bench_glwe_external_product<BE: Backend>(
    glwe_infos: &impl GLWEInfos,
    ggsw_infos: &impl GGSWInfos,
    c: &mut Criterion,
    label: &str,
) where
    Module<BE>: ModuleNew<BE>
        + GLWEExternalProduct<BE>
        + GGSWEncryptSk<BE>
        + GGSWPreparedFactory<BE>
        + GLWEEncryptSk<BE>
        + GLWESecretPreparedFactory<BE>,
    ScratchOwned<BE>: ScratchOwnedAlloc<BE> + ScratchOwnedBorrow<BE>,
    Scratch<BE>: ScratchTakeCore<BE>,
{
    let n: usize = ggsw_infos.n().into();
    let module: Module<BE> = Module::<BE>::new(n as u64);

    let mut source_xs = Source::new([0u8; 32]);
    let mut source_xa = Source::new([1u8; 32]);
    let mut source_xe = Source::new([2u8; 32]);

    let mut sk: GLWESecret<Vec<u8>> = GLWESecret::alloc_from_infos(ggsw_infos);
    sk.fill_ternary_prob(0.5, &mut source_xs);

    let mut sk_prepared: GLWESecretPrepared<DeviceBuf<BE>, BE> = module.glwe_secret_prepared_alloc(ggsw_infos.rank());
    module.glwe_secret_prepare(&mut sk_prepared, &sk);

    let pt = ScalarZnx::alloc(n, 1);
    let mut ct_ggsw: GGSW<Vec<u8>> = GGSW::alloc_from_infos(ggsw_infos);
    let mut ct_glwe_in: GLWE<Vec<u8>> = GLWE::alloc_from_infos(glwe_infos);
    let mut ct_glwe_out: GLWE<Vec<u8>> = GLWE::alloc_from_infos(glwe_infos);

    let mut scratch: ScratchOwned<BE> = ScratchOwned::alloc(
        module.ggsw_encrypt_sk_tmp_bytes(ggsw_infos)
            | module.glwe_encrypt_sk_tmp_bytes(glwe_infos)
            | module.glwe_external_product_tmp_bytes(glwe_infos, glwe_infos, ggsw_infos),
    );

    let ggsw_enc_infos = NoiseInfos::new(ggsw_infos.max_k().as_usize(), DEFAULT_SIGMA_XE, DEFAULT_BOUND_XE).unwrap();
    let glwe_enc_infos = NoiseInfos::new(glwe_infos.max_k().as_usize(), DEFAULT_SIGMA_XE, DEFAULT_BOUND_XE).unwrap();

    module.ggsw_encrypt_sk(
        &mut ct_ggsw,
        &pt,
        &sk_prepared,
        &ggsw_enc_infos,
        &mut source_xe,
        &mut source_xa,
        scratch.borrow(),
    );
    module.glwe_encrypt_zero_sk(
        &mut ct_glwe_in,
        &sk_prepared,
        &glwe_enc_infos,
        &mut source_xe,
        &mut source_xa,
        scratch.borrow(),
    );

    let mut ggsw_prepared: GGSWPrepared<DeviceBuf<BE>, BE> = module.ggsw_prepared_alloc_from_infos(&ct_ggsw);
    module.ggsw_prepare(&mut ggsw_prepared, &ct_ggsw, scratch.borrow());

    let group_name = format!("glwe_external_product::{label}");
    let mut group = c.benchmark_group(group_name);
    group.bench_function(format!("n={n}"), |bench| {
        bench.iter(|| {
            module.glwe_external_product(&mut ct_glwe_out, &ct_glwe_in, &ggsw_prepared, scratch.borrow());
            black_box(());
        })
    });
    group.finish();
}

pub fn bench_glwe_external_product_assign<BE: Backend>(infos: &impl GGSWInfos, c: &mut Criterion, label: &str)
where
    Module<BE>: ModuleNew<BE>
        + GLWEExternalProduct<BE>
        + GGSWEncryptSk<BE>
        + GGSWPreparedFactory<BE>
        + GLWEEncryptSk<BE>
        + GLWESecretPreparedFactory<BE>,
    ScratchOwned<BE>: ScratchOwnedAlloc<BE> + ScratchOwnedBorrow<BE>,
    Scratch<BE>: ScratchTakeCore<BE>,
{
    let n: usize = infos.n().into();
    let module: Module<BE> = Module::<BE>::new(n as u64);

    let mut source_xs = Source::new([0u8; 32]);
    let mut source_xa = Source::new([1u8; 32]);
    let mut source_xe = Source::new([2u8; 32]);

    let mut sk: GLWESecret<Vec<u8>> = GLWESecret::alloc_from_infos(infos);
    sk.fill_ternary_prob(0.5, &mut source_xs);

    let mut sk_prepared: GLWESecretPrepared<DeviceBuf<BE>, BE> = module.glwe_secret_prepared_alloc(infos.rank());
    module.glwe_secret_prepare(&mut sk_prepared, &sk);

    let pt = ScalarZnx::alloc(n, 1);
    let mut ct_ggsw: GGSW<Vec<u8>> = GGSW::alloc_from_infos(infos);
    let mut ct_glwe: GLWE<Vec<u8>> = GLWE::alloc_from_infos(infos);

    let mut scratch: ScratchOwned<BE> = ScratchOwned::alloc(
        module.ggsw_encrypt_sk_tmp_bytes(infos)
            | module.glwe_encrypt_sk_tmp_bytes(infos)
            | module.glwe_external_product_tmp_bytes(infos, infos, infos),
    );

    let enc_infos = NoiseInfos::new(infos.max_k().as_usize(), DEFAULT_SIGMA_XE, DEFAULT_BOUND_XE).unwrap();

    module.ggsw_encrypt_sk(
        &mut ct_ggsw,
        &pt,
        &sk_prepared,
        &enc_infos,
        &mut source_xe,
        &mut source_xa,
        scratch.borrow(),
    );
    module.glwe_encrypt_zero_sk(
        &mut ct_glwe,
        &sk_prepared,
        &enc_infos,
        &mut source_xe,
        &mut source_xa,
        scratch.borrow(),
    );

    let mut ggsw_prepared: GGSWPrepared<DeviceBuf<BE>, BE> = module.ggsw_prepared_alloc_from_infos(&ct_ggsw);
    module.ggsw_prepare(&mut ggsw_prepared, &ct_ggsw, scratch.borrow());

    let group_name = format!("glwe_external_product_assign::{label}");
    let mut group = c.benchmark_group(group_name);
    group.bench_function(format!("n={n}"), |bench| {
        bench.iter(|| {
            module.glwe_external_product_assign(&mut ct_glwe, &ggsw_prepared, scratch.borrow());
            black_box(());
        })
    });
    group.finish();
}
