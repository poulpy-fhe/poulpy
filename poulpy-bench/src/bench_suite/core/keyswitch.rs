use poulpy_core::{
    DEFAULT_BOUND_XE, DEFAULT_SIGMA_XE, GLWEEncryptSk, GLWEKeyswitch, GLWESwitchingKeyEncryptSk,
    layouts::{
        GGLWEInfos, GLWE, GLWEInfos, GLWESecret, GLWESecretPreparedFactory, GLWESwitchingKey, GLWESwitchingKeyPrepared,
        GLWESwitchingKeyPreparedFactory, ModuleCoreAlloc, prepared::GLWESecretPrepared,
    },
};
use poulpy_hal::{
    api::{ModuleNew, ScratchOwnedAlloc, ScratchOwnedBorrow},
    layouts::{Backend, Module, NoiseInfos, ScratchOwned},
    source::Source,
};

use std::hint::black_box;

use criterion::Criterion;

pub fn bench_glwe_keyswitch<BE: Backend<OwnedBuf = Vec<u8>>>(
    glwe_in: &impl GLWEInfos,
    glwe_out: &impl GLWEInfos,
    gglwe: &impl GGLWEInfos,
    c: &mut Criterion,
    label: &str,
) where
    Module<BE>: ModuleNew<BE>
        + GLWESwitchingKeyEncryptSk<BE>
        + GLWEEncryptSk<BE>
        + GLWEKeyswitch<BE>
        + GLWESecretPreparedFactory<BE>
        + GLWESwitchingKeyPreparedFactory<BE>,
    ScratchOwned<BE>: ScratchOwnedAlloc<BE> + ScratchOwnedBorrow<BE>,
    for<'a> BE::BufMut<'a>: AsRef<[u8]> + AsMut<[u8]> + Sync,
    for<'a> BE::BufRef<'a>: AsRef<[u8]> + Send,
{
    let n: usize = gglwe.n().into();
    let module: Module<BE> = Module::<BE>::new(n as u64);

    let mut ksk: GLWESwitchingKey<Vec<u8>> = module.glwe_switching_key_alloc_from_infos(gglwe);
    let mut ct_in: GLWE<Vec<u8>> = module.glwe_alloc_from_infos(glwe_in);
    let mut ct_out: GLWE<Vec<u8>> = module.glwe_alloc_from_infos(glwe_out);

    let mut scratch: ScratchOwned<BE> = ScratchOwned::alloc(
        module.glwe_switching_key_encrypt_sk_tmp_bytes(gglwe)
            | module.glwe_encrypt_sk_tmp_bytes(glwe_in)
            | module.glwe_keyswitch_tmp_bytes(glwe_out, glwe_in, gglwe),
    );

    let mut source_xs: Source = Source::new([0u8; 32]);
    let mut source_xe: Source = Source::new([1u8; 32]);
    let mut source_xa: Source = Source::new([2u8; 32]);

    let mut sk_in: GLWESecret<Vec<u8>> = module.glwe_secret_alloc_from_infos(glwe_in);
    sk_in.fill_ternary_prob(0.5, &mut source_xs);

    let mut sk_in_prepared: GLWESecretPrepared<BE::OwnedBuf, BE> = module.glwe_secret_prepared_alloc(glwe_in.rank());
    module.glwe_secret_prepare(&mut sk_in_prepared, &sk_in);

    let mut sk_out: GLWESecret<Vec<u8>> = module.glwe_secret_alloc_from_infos(glwe_out);
    sk_out.fill_ternary_prob(0.5, &mut source_xs);

    let ksk_enc_infos = NoiseInfos::new(gglwe.max_k().as_usize(), DEFAULT_SIGMA_XE, DEFAULT_BOUND_XE).unwrap();
    let glwe_enc_infos = NoiseInfos::new(glwe_in.max_k().as_usize(), DEFAULT_SIGMA_XE, DEFAULT_BOUND_XE).unwrap();

    module.glwe_switching_key_encrypt_sk(
        &mut ksk,
        &sk_in,
        &sk_out,
        &ksk_enc_infos,
        &mut source_xe,
        &mut source_xa,
        &mut scratch.borrow(),
    );

    module.glwe_encrypt_zero_sk(
        &mut ct_in,
        &sk_in_prepared,
        &glwe_enc_infos,
        &mut source_xe,
        &mut source_xa,
        &mut scratch.borrow(),
    );

    let mut ksk_prepared: GLWESwitchingKeyPrepared<BE::OwnedBuf, BE> = module.glwe_switching_key_prepared_alloc_from_infos(&ksk);
    module.glwe_switching_key_prepare(&mut ksk_prepared, &ksk, &mut scratch.borrow());

    let group_name = format!("glwe_keyswitch::{label}");
    let mut group = c.benchmark_group(group_name);
    group.bench_function(format!("n={n}"), |bench| {
        bench.iter(|| {
            module.glwe_keyswitch(&mut ct_out, &ct_in, &ksk_prepared, &mut scratch.borrow());
            black_box(());
        })
    });
    group.finish();
}
