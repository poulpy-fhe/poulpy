use poulpy_core::{
    DEFAULT_BOUND_XE, DEFAULT_SIGMA_XE, GLWEDecrypt, GLWEEncryptSk,
    layouts::{
        Base2K, Degree, GLWE, GLWEInfos, GLWELayout, GLWEPlaintext, GLWESecret, GLWESecretPreparedFactory, GLWESecretSampling,
        LWEInfos, ModuleCoreAlloc, Rank, TorusPrecision, prepared::GLWESecretPrepared,
    },
};
use poulpy_hal::{
    api::{ModuleNew, ScratchOwnedAlloc, ScratchOwnedBorrow},
    layouts::{Backend, HostBackend, Module, NoiseInfos, ScratchOwned},
    source::Source,
};
use std::hint::black_box;

use criterion::{Bencher, measurement::Measurement};

use crate::core::params::CoreParams;

pub fn runner_glwe_decrypt<BE: Backend<OwnedBuf = Vec<u8>, ZnxWord = i64> + HostBackend, M: Measurement>(
    bencher: &mut Bencher<'_, M>,
    cp: &CoreParams,
) where
    Module<BE>: ModuleNew<BE> + GLWEDecrypt<BE> + GLWEEncryptSk<BE> + GLWESecretPreparedFactory<BE> + GLWESecretSampling<BE>,
    ScratchOwned<BE>: ScratchOwnedAlloc<BE> + ScratchOwnedBorrow<BE>,
    for<'a> BE::BufMut<'a>: AsRef<[u8]> + AsMut<[u8]> + Sync,
    for<'a> BE::BufRef<'a>: AsRef<[u8]> + Send,
{
    let infos = GLWELayout {
        n: Degree(cp.n),
        base2k: Base2K(cp.base2k),
        k: TorusPrecision(cp.k),
        rank: Rank(cp.rank),
    };

    let module: Module<BE> = Module::<BE>::new(cp.n as u64);

    let mut source_xs = Source::new([0u8; 32]);
    let mut source_xa = Source::new([1u8; 32]);
    let mut source_xe = Source::new([2u8; 32]);

    let mut sk: GLWESecret<Vec<u8>, i64> = module.glwe_secret_alloc_from_infos(&infos);
    module.glwe_secret_fill_ternary_prob(&mut sk, 0.5, &mut source_xs);

    let mut sk_prepared: GLWESecretPrepared<BE::OwnedBuf, BE> = module.glwe_secret_prepared_alloc(infos.rank());
    module.glwe_secret_prepare(&mut sk_prepared, &sk);

    let mut ct: GLWE<Vec<u8>, i64> = module.glwe_alloc_from_infos(&infos);
    let mut pt: GLWEPlaintext<Vec<u8>, i64> = module.glwe_plaintext_alloc_from_infos(&infos);

    let mut scratch: ScratchOwned<BE> = ScratchOwned::alloc(
        module
            .glwe_encrypt_sk_tmp_bytes(&infos)
            .max(module.glwe_decrypt_tmp_bytes(&infos)),
    );

    let enc_infos = NoiseInfos::new(infos.k().as_usize(), DEFAULT_SIGMA_XE, DEFAULT_BOUND_XE).unwrap();
    module.glwe_encrypt_zero_sk(
        &mut ct,
        &sk_prepared,
        &enc_infos,
        &mut source_xe,
        &mut source_xa,
        &mut scratch.borrow(),
    );

    bencher.iter(|| {
        module.glwe_decrypt(&ct, &mut pt, &sk_prepared, &mut scratch.borrow());
        black_box(());
    });
}
