use poulpy_core::{
    DEFAULT_BOUND_XE, DEFAULT_SIGMA_XE, GGSWEncryptSk, GLWEAutomorphismKeyEncryptSk, GLWEEncryptSk,
    layouts::{
        GGLWEInfos, GGSWInfos, GLWEAutomorphismKey, GLWEInfos, GLWESecret, GLWESecretPreparedFactory, ModuleCoreAlloc,
        prepared::GLWESecretPrepared,
    },
};
use poulpy_hal::{
    api::{ModuleNew, ScratchOwnedAlloc, ScratchOwnedBorrow},
    layouts::{Backend, Module, NoiseInfos, ScratchOwned},
    source::Source,
};
use std::hint::black_box;

use criterion::Criterion;

pub fn bench_glwe_encrypt_sk<BE: Backend<OwnedBuf = Vec<u8>>>(infos: &impl GLWEInfos, c: &mut Criterion, label: &str)
where
    Module<BE>: ModuleNew<BE> + GLWEEncryptSk<BE> + GLWESecretPreparedFactory<BE>,
    ScratchOwned<BE>: ScratchOwnedAlloc<BE> + ScratchOwnedBorrow<BE>,
{
    let n: usize = infos.n().into();
    let module: Module<BE> = Module::<BE>::new(n as u64);

    let mut source_xs = Source::new([0u8; 32]);
    let mut source_xa = Source::new([1u8; 32]);
    let mut source_xe = Source::new([2u8; 32]);

    let mut sk: GLWESecret<Vec<u8>> = module.glwe_secret_alloc_from_infos(infos);
    sk.fill_ternary_prob(0.5, &mut source_xs);

    let mut sk_prepared: GLWESecretPrepared<BE::OwnedBuf, BE> = module.glwe_secret_prepared_alloc(infos.rank());
    module.glwe_secret_prepare(&mut sk_prepared, &sk);

    let mut ct: poulpy_core::layouts::GLWE<Vec<u8>> = module.glwe_alloc_from_infos(infos);
    let mut scratch: ScratchOwned<BE> = ScratchOwned::alloc(module.glwe_encrypt_sk_tmp_bytes(infos));

    let enc_infos = NoiseInfos::new(infos.max_k().as_usize(), DEFAULT_SIGMA_XE, DEFAULT_BOUND_XE).unwrap();

    let group_name = format!("glwe_encrypt_sk::{label}");
    let mut group = c.benchmark_group(group_name);
    group.bench_function(format!("n={n}"), |bench| {
        bench.iter(|| {
            module.glwe_encrypt_zero_sk(
                &mut ct,
                &sk_prepared,
                &enc_infos,
                &mut source_xe,
                &mut source_xa,
                &mut scratch.borrow(),
            );
            black_box(());
        })
    });
    group.finish();
}

pub fn bench_ggsw_encrypt_sk<BE: Backend<OwnedBuf = Vec<u8>>>(infos: &impl GGSWInfos, c: &mut Criterion, label: &str)
where
    Module<BE>: ModuleNew<BE> + GGSWEncryptSk<BE> + GLWESecretPreparedFactory<BE>,
    ScratchOwned<BE>: ScratchOwnedAlloc<BE> + ScratchOwnedBorrow<BE>,
    for<'a> BE::BufMut<'a>: AsRef<[u8]> + AsMut<[u8]> + Sync,
    for<'a> BE::BufRef<'a>: AsRef<[u8]> + Send,
{
    let n: usize = infos.n().into();
    let module: Module<BE> = Module::<BE>::new(n as u64);

    let mut source_xs = Source::new([0u8; 32]);
    let mut source_xa = Source::new([1u8; 32]);
    let mut source_xe = Source::new([2u8; 32]);

    let mut sk: GLWESecret<Vec<u8>> = module.glwe_secret_alloc_from_infos(infos);
    sk.fill_ternary_prob(0.5, &mut source_xs);

    let mut sk_prepared: GLWESecretPrepared<BE::OwnedBuf, BE> = module.glwe_secret_prepared_alloc(infos.rank());
    module.glwe_secret_prepare(&mut sk_prepared, &sk);

    let pt = module.scalar_znx_alloc(1);
    let mut ct = module.ggsw_alloc_from_infos(infos);
    let mut scratch: ScratchOwned<BE> = ScratchOwned::alloc(module.ggsw_encrypt_sk_tmp_bytes(infos));

    let enc_infos = NoiseInfos::new(infos.max_k().as_usize(), DEFAULT_SIGMA_XE, DEFAULT_BOUND_XE).unwrap();

    let group_name = format!("ggsw_encrypt_sk::{label}");
    let mut group = c.benchmark_group(group_name);
    group.bench_function(format!("n={n}"), |bench| {
        bench.iter(|| {
            module.ggsw_encrypt_sk(
                &mut ct,
                &pt,
                &sk_prepared,
                &enc_infos,
                &mut source_xe,
                &mut source_xa,
                &mut scratch.borrow(),
            );
            black_box(());
        })
    });
    group.finish();
}

pub fn bench_glwe_automorphism_key_encrypt_sk<BE: Backend<OwnedBuf = Vec<u8>>>(
    atk_infos: &impl GGLWEInfos,
    p: i64,
    c: &mut Criterion,
    label: &str,
) where
    Module<BE>: ModuleNew<BE> + GLWEAutomorphismKeyEncryptSk<BE>,
    ScratchOwned<BE>: ScratchOwnedAlloc<BE> + ScratchOwnedBorrow<BE>,
{
    let n: usize = atk_infos.n().into();
    let module: Module<BE> = Module::<BE>::new(n as u64);

    let mut source_xs = Source::new([0u8; 32]);
    let mut source_xa = Source::new([1u8; 32]);
    let mut source_xe = Source::new([2u8; 32]);

    let mut sk: GLWESecret<Vec<u8>> = module.glwe_secret_alloc_from_infos(atk_infos);
    sk.fill_ternary_prob(0.5, &mut source_xs);

    let mut atk: GLWEAutomorphismKey<Vec<u8>> = module.glwe_automorphism_key_alloc_from_infos(atk_infos);
    let mut scratch: ScratchOwned<BE> = ScratchOwned::alloc(module.glwe_automorphism_key_encrypt_sk_tmp_bytes(atk_infos));

    let enc_infos = NoiseInfos::new(atk_infos.max_k().as_usize(), DEFAULT_SIGMA_XE, DEFAULT_BOUND_XE).unwrap();

    let group_name = format!("glwe_automorphism_key_encrypt_sk::{label}");
    let mut group = c.benchmark_group(group_name);
    group.bench_function(format!("n={n}"), |bench| {
        bench.iter(|| {
            module.glwe_automorphism_key_encrypt_sk(
                &mut atk,
                p,
                &sk,
                &enc_infos,
                &mut source_xe,
                &mut source_xa,
                &mut scratch.borrow(),
            );
            black_box(());
        })
    });
    group.finish();
}
