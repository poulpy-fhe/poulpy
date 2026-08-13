use poulpy_core::{
    DEFAULT_BOUND_XE, DEFAULT_SIGMA_XE, GGSWEncryptSk, GLWEAutomorphismKeyEncryptSk, GLWEEncryptSk,
    layouts::{
        Base2K, Degree, Dnum, Dsize, GGSWLayout, GLWEAutomorphismKey, GLWEAutomorphismKeyLayout, GLWEInfos, GLWELayout,
        GLWESecret, GLWESecretPreparedFactory, GLWESecretSampling, LWEInfos, ModuleCoreAlloc, Rank, TorusPrecision,
        prepared::GLWESecretPrepared,
    },
};
use poulpy_hal::{
    api::{ModuleNew, ScratchOwnedAlloc, ScratchOwnedBorrow},
    layouts::{Backend, Module, NoiseInfos, ScratchOwned},
    source::Source,
};
use std::hint::black_box;

use criterion::{Bencher, measurement::Measurement};

use crate::params::{CoreParams, key_dnum_k_aux};

pub fn runner_glwe_encrypt_sk<BE: Backend<OwnedBuf = Vec<u8>, ZnxWord = i64>, M: Measurement>(
    bencher: &mut Bencher<'_, M>,
    cp: &CoreParams,
) where
    Module<BE>: ModuleNew<BE> + GLWEEncryptSk<BE> + GLWESecretPreparedFactory<BE> + GLWESecretSampling<BE>,
    ScratchOwned<BE>: ScratchOwnedAlloc<BE> + ScratchOwnedBorrow<BE>,
    BE::OwnedBuf: AsRef<[u8]> + AsMut<[u8]>,
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

    let mut ct: poulpy_core::layouts::GLWE<Vec<u8>, i64> = module.glwe_alloc_from_infos(&infos);
    let mut scratch: ScratchOwned<BE> = ScratchOwned::alloc(module.glwe_encrypt_sk_tmp_bytes(&infos));

    let enc_infos = NoiseInfos::new(infos.max_k().as_usize(), DEFAULT_SIGMA_XE, DEFAULT_BOUND_XE).unwrap();

    bencher.iter(|| {
        module.glwe_encrypt_zero_sk(
            &mut ct,
            &sk_prepared,
            &enc_infos,
            &mut source_xe,
            &mut source_xa,
            &mut scratch.borrow(),
        );
        black_box(());
    });
}

pub fn runner_ggsw_encrypt_sk<BE: Backend<OwnedBuf = Vec<u8>, ZnxWord = i64>, M: Measurement>(
    bencher: &mut Bencher<'_, M>,
    cp: &CoreParams,
) where
    Module<BE>: ModuleNew<BE> + GGSWEncryptSk<BE> + GLWESecretPreparedFactory<BE> + GLWESecretSampling<BE>,
    ScratchOwned<BE>: ScratchOwnedAlloc<BE> + ScratchOwnedBorrow<BE>,
    for<'a> BE::BufMut<'a>: AsRef<[u8]> + AsMut<[u8]> + Sync,
    for<'a> BE::BufRef<'a>: AsRef<[u8]> + Send,
{
    let (dnum, k_aux) = key_dnum_k_aux(cp.k, cp.base2k, cp.dsize);
    let infos = GGSWLayout {
        n: Degree(cp.n),
        base2k: Base2K(cp.base2k),
        k_aux: TorusPrecision(k_aux),
        rank: Rank(cp.rank),
        dnum: Dnum(dnum),
        dsize: Dsize(cp.dsize),
    };

    let module: Module<BE> = Module::<BE>::new(cp.n as u64);

    let mut source_xs = Source::new([0u8; 32]);
    let mut source_xa = Source::new([1u8; 32]);
    let mut source_xe = Source::new([2u8; 32]);

    let mut sk: GLWESecret<Vec<u8>, i64> = module.glwe_secret_alloc_from_infos(&infos);
    module.glwe_secret_fill_ternary_prob(&mut sk, 0.5, &mut source_xs);

    let mut sk_prepared: GLWESecretPrepared<BE::OwnedBuf, BE> = module.glwe_secret_prepared_alloc(infos.rank());
    module.glwe_secret_prepare(&mut sk_prepared, &sk);

    let pt = module.scalar_znx_alloc(1);
    let mut ct = module.ggsw_alloc_from_infos(&infos);
    let mut scratch: ScratchOwned<BE> = ScratchOwned::alloc(module.ggsw_encrypt_sk_tmp_bytes(&infos));

    let enc_infos = NoiseInfos::new(infos.max_k().as_usize(), DEFAULT_SIGMA_XE, DEFAULT_BOUND_XE).unwrap();

    bencher.iter(|| {
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
    });
}

pub fn runner_glwe_automorphism_key_encrypt_sk<BE: Backend<OwnedBuf = Vec<u8>, ZnxWord = i64>, M: Measurement>(
    bencher: &mut Bencher<'_, M>,
    cp: &CoreParams,
) where
    Module<BE>: ModuleNew<BE> + GLWEAutomorphismKeyEncryptSk<BE> + GLWESecretSampling<BE>,
    ScratchOwned<BE>: ScratchOwnedAlloc<BE> + ScratchOwnedBorrow<BE>,
{
    const P: i64 = 3;

    let (dnum, k_aux) = key_dnum_k_aux(cp.k, cp.base2k, cp.dsize);
    let atk_infos = GLWEAutomorphismKeyLayout {
        n: Degree(cp.n),
        base2k: Base2K(cp.base2k),
        k_aux: TorusPrecision(k_aux),
        rank: Rank(cp.rank),
        dnum: Dnum(dnum),
        dsize: Dsize(cp.dsize),
    };

    let module: Module<BE> = Module::<BE>::new(cp.n as u64);

    let mut source_xs = Source::new([0u8; 32]);
    let mut source_xa = Source::new([1u8; 32]);
    let mut source_xe = Source::new([2u8; 32]);

    let mut sk: GLWESecret<Vec<u8>, i64> = module.glwe_secret_alloc_from_infos(&atk_infos);
    module.glwe_secret_fill_ternary_prob(&mut sk, 0.5, &mut source_xs);

    let mut atk: GLWEAutomorphismKey<Vec<u8>, i64> = module.glwe_automorphism_key_alloc_from_infos(&atk_infos);
    let mut scratch: ScratchOwned<BE> = ScratchOwned::alloc(module.glwe_automorphism_key_encrypt_sk_tmp_bytes(&atk_infos));

    let enc_infos = NoiseInfos::new(atk_infos.max_k().as_usize(), DEFAULT_SIGMA_XE, DEFAULT_BOUND_XE).unwrap();

    bencher.iter(|| {
        module.glwe_automorphism_key_encrypt_sk(
            &mut atk,
            P,
            &sk,
            &enc_infos,
            &mut source_xe,
            &mut source_xa,
            &mut scratch.borrow(),
        );
        black_box(());
    });
}
