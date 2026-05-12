use poulpy_core::{
    DEFAULT_BOUND_XE, DEFAULT_SIGMA_XE, GLWEAutomorphism, GLWEAutomorphismKeyEncryptSk, GLWEEncryptSk,
    layouts::{
        GGLWEInfos, GLWE, GLWEAutomorphismKey, GLWEInfos, GLWESecret, GLWESecretPreparedFactory, LWEInfos, ModuleCoreAlloc,
        prepared::{GLWEAutomorphismKeyPrepared, GLWEAutomorphismKeyPreparedFactory, GLWESecretPrepared},
    },
};
use poulpy_hal::{
    api::{ModuleNew, ScratchOwnedAlloc, ScratchOwnedBorrow},
    layouts::{Backend, Module, NoiseInfos, ScratchOwned},
    source::Source,
};
use std::hint::black_box;

use criterion::Criterion;

/// Benchmarks the GLWE automorphism operation.
///
/// `glwe_infos` describes the input/output GLWE layout.
/// `atk_infos` describes the automorphism key layout.
/// `p` is the Galois element (e.g. 3 for X -> X^3).
pub fn bench_glwe_automorphism<BE: Backend<OwnedBuf = Vec<u8>>>(
    glwe_infos: &impl GLWEInfos,
    atk_infos: &impl GGLWEInfos,
    p: i64,
    c: &mut Criterion,
    label: &str,
) where
    Module<BE>: ModuleNew<BE>
        + GLWEAutomorphism<BE>
        + GLWEAutomorphismKeyEncryptSk<BE>
        + GLWEEncryptSk<BE>
        + GLWESecretPreparedFactory<BE>
        + GLWEAutomorphismKeyPreparedFactory<BE>,
    ScratchOwned<BE>: ScratchOwnedAlloc<BE> + ScratchOwnedBorrow<BE>,
    for<'a> BE::BufMut<'a>: AsRef<[u8]> + AsMut<[u8]> + Sync,
    for<'a> BE::BufRef<'a>: AsRef<[u8]> + Send,
{
    let n: usize = atk_infos.n().into();
    let module: Module<BE> = Module::<BE>::new(n as u64);

    let mut source_xs = Source::new([0u8; 32]);
    let mut source_xa = Source::new([1u8; 32]);
    let mut source_xe = Source::new([2u8; 32]);

    let mut sk: GLWESecret<Vec<u8>> = module.glwe_secret_alloc_from_infos(atk_infos);
    sk.fill_ternary_prob(0.5, &mut source_xs);

    let mut sk_prepared: GLWESecretPrepared<BE::OwnedBuf, BE> = module.glwe_secret_prepared_alloc(atk_infos.rank_out());
    module.glwe_secret_prepare(&mut sk_prepared, &sk);

    let mut atk: GLWEAutomorphismKey<Vec<u8>> = module.glwe_automorphism_key_alloc_from_infos(atk_infos);
    let mut scratch: ScratchOwned<BE> = ScratchOwned::alloc(
        module.glwe_automorphism_key_encrypt_sk_tmp_bytes(atk_infos)
            | module.glwe_encrypt_sk_tmp_bytes(glwe_infos)
            | module.glwe_automorphism_tmp_bytes(glwe_infos, glwe_infos, atk_infos),
    );

    let atk_enc_infos = NoiseInfos::new(atk_infos.max_k().as_usize(), DEFAULT_SIGMA_XE, DEFAULT_BOUND_XE).unwrap();
    module.glwe_automorphism_key_encrypt_sk(
        &mut atk,
        p,
        &sk,
        &atk_enc_infos,
        &mut source_xe,
        &mut source_xa,
        &mut scratch.borrow(),
    );

    let mut atk_prepared: GLWEAutomorphismKeyPrepared<BE::OwnedBuf, BE> =
        module.glwe_automorphism_key_prepared_alloc_from_infos(&atk);
    module.glwe_automorphism_key_prepare(&mut atk_prepared, &atk, &mut scratch.borrow());

    let mut ct_in: GLWE<Vec<u8>> = module.glwe_alloc_from_infos(glwe_infos);
    let mut ct_out: GLWE<Vec<u8>> = module.glwe_alloc_from_infos(glwe_infos);

    let glwe_enc_infos = NoiseInfos::new(glwe_infos.max_k().as_usize(), DEFAULT_SIGMA_XE, DEFAULT_BOUND_XE).unwrap();
    module.glwe_encrypt_zero_sk(
        &mut ct_in,
        &sk_prepared,
        &glwe_enc_infos,
        &mut source_xe,
        &mut source_xa,
        &mut scratch.borrow(),
    );

    let group_name = format!("glwe_automorphism::{label}");
    let mut group = c.benchmark_group(group_name);
    group.bench_function(format!("n={n}"), |bench| {
        bench.iter(|| {
            module.glwe_automorphism(
                &mut ct_out,
                &ct_in,
                &atk_prepared,
                ct_in.size() + atk_prepared.dsize().as_usize(),
                &mut scratch.borrow(),
            );
            black_box(());
        })
    });
    group.finish();
}
