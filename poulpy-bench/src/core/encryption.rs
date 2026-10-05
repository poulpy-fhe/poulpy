use poulpy_core::{
    GGSWEncryptSk, GLWEAutomorphismKeyEncryptSk, GLWEEncryptPk, GLWEEncryptSk, GLWEPublicKeyGenerate,
    layouts::{
        Base2K, Degree, Dnum, Dsize, GGSWLayout, GLWEAutomorphismKey, GLWEAutomorphismKeyLayout, GLWEInfos, GLWELayout,
        GLWEPublicKeyPreparedFactory, GLWESecret, GLWESecretPreparedFactory, GLWESecretSampling, ModuleCoreAlloc, Rank,
        TorusPrecision,
        prepared::{GLWEPublicKeyPrepared, GLWESecretPrepared},
    },
};
use poulpy_hal::{
    api::{ModuleNew, ScratchOwnedAlloc, ScratchOwnedBorrow},
    layouts::{Backend, Module, ScratchOwned},
    source::Source,
};
use std::hint::black_box;

use criterion::{Bencher, measurement::Measurement};

use crate::core::params::{CoreParams, key_dnum_k_aux};

pub fn runner_glwe_encrypt_sk<BE: Backend<ZnxWord = i64>, M: Measurement>(bencher: &mut Bencher<'_, M>, cp: &CoreParams)
where
    Module<BE>: ModuleNew<BE>
        + GLWEEncryptSk<BE>
        + GLWESecretPreparedFactory<BE>
        + GLWESecretSampling<BE>
        + ModuleCoreAlloc<OwnedBuf = BE::OwnedBuf, ZnxWord = i64>,
    ScratchOwned<BE>: ScratchOwnedAlloc<BE> + ScratchOwnedBorrow<BE>,
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

    let mut sk: GLWESecret<BE::OwnedBuf, i64> = module.glwe_secret_alloc_from_infos(&infos);
    module.glwe_secret_fill_ternary_prob(&mut sk, 0.5, &mut source_xs);

    let mut sk_prepared: GLWESecretPrepared<BE::OwnedBuf, BE> = module.glwe_secret_prepared_alloc(infos.rank());
    module.glwe_secret_prepare(&mut sk_prepared, &sk);

    let mut ct: poulpy_core::layouts::GLWE<BE::OwnedBuf, i64> = module.glwe_alloc_from_infos(&infos);
    let mut scratch: ScratchOwned<BE> = ScratchOwned::alloc(module.glwe_encrypt_sk_tmp_bytes(&infos));

    bencher.iter(|| {
        module.glwe_encrypt_zero_sk(&mut ct, &sk_prepared, &mut source_xe, &mut source_xa, &mut scratch.borrow());
        black_box(());
    });
}

pub fn runner_glwe_encrypt_pk<BE: Backend<ZnxWord = i64>, M: Measurement>(bencher: &mut Bencher<'_, M>, cp: &CoreParams)
where
    Module<BE>: ModuleNew<BE>
        + GLWEEncryptPk<BE>
        + GLWEPublicKeyGenerate<BE>
        + GLWEPublicKeyPreparedFactory<BE>
        + GLWESecretPreparedFactory<BE>
        + GLWESecretSampling<BE>
        + ModuleCoreAlloc<OwnedBuf = BE::OwnedBuf, ZnxWord = i64>,
    ScratchOwned<BE>: ScratchOwnedAlloc<BE> + ScratchOwnedBorrow<BE>,
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
    let mut source_xu = Source::new([3u8; 32]);

    let mut sk: GLWESecret<BE::OwnedBuf, i64> = module.glwe_secret_alloc_from_infos(&infos);
    module.glwe_secret_fill_ternary_prob(&mut sk, 0.5, &mut source_xs);

    let mut sk_prepared: GLWESecretPrepared<BE::OwnedBuf, BE> = module.glwe_secret_prepared_alloc(infos.rank());
    module.glwe_secret_prepare(&mut sk_prepared, &sk);

    let mut scratch: ScratchOwned<BE> = ScratchOwned::alloc(
        module
            .glwe_encrypt_pk_tmp_bytes(&infos, &infos)
            .max(module.glwe_public_key_generate_tmp_bytes(&infos))
            .max(module.glwe_public_key_prepare_tmp_bytes(&infos)),
    );

    let mut pk = module.glwe_public_key_alloc_from_infos(&infos);
    module.glwe_public_key_generate(&mut pk, &sk_prepared, &mut source_xe, &mut source_xa, &mut scratch.borrow());

    let mut pk_prepared: GLWEPublicKeyPrepared<BE::OwnedBuf, BE> = module.glwe_public_key_prepared_alloc_from_infos(&infos);
    module.glwe_public_key_prepare(&mut pk_prepared, &pk, &mut scratch.borrow());

    let pt = module.glwe_plaintext_alloc_from_infos(&infos);
    let mut ct = module.glwe_alloc_from_infos(&infos);

    bencher.iter(|| {
        module.glwe_encrypt_pk(
            &mut ct,
            &pt,
            &pk_prepared,
            &mut source_xu,
            &mut source_xe,
            &mut scratch.borrow(),
        );
        black_box(());
    });
}

pub fn runner_ggsw_encrypt_sk<BE: Backend<ZnxWord = i64>, M: Measurement>(bencher: &mut Bencher<'_, M>, cp: &CoreParams)
where
    Module<BE>: ModuleNew<BE>
        + GGSWEncryptSk<BE>
        + GLWESecretPreparedFactory<BE>
        + GLWESecretSampling<BE>
        + ModuleCoreAlloc<OwnedBuf = BE::OwnedBuf, ZnxWord = i64>,
    ScratchOwned<BE>: ScratchOwnedAlloc<BE> + ScratchOwnedBorrow<BE>,
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

    let mut sk: GLWESecret<BE::OwnedBuf, i64> = module.glwe_secret_alloc_from_infos(&infos);
    module.glwe_secret_fill_ternary_prob(&mut sk, 0.5, &mut source_xs);

    let mut sk_prepared: GLWESecretPrepared<BE::OwnedBuf, BE> = module.glwe_secret_prepared_alloc(infos.rank());
    module.glwe_secret_prepare(&mut sk_prepared, &sk);

    let pt = module.scalar_znx_alloc(module.n(), 1);
    let mut ct = module.ggsw_alloc_from_infos(&infos);
    let mut scratch: ScratchOwned<BE> = ScratchOwned::alloc(module.ggsw_encrypt_sk_tmp_bytes(&infos));

    bencher.iter(|| {
        module.ggsw_encrypt_sk(
            &mut ct,
            &pt,
            &sk_prepared,
            &mut source_xe,
            &mut source_xa,
            &mut scratch.borrow(),
        );
        black_box(());
    });
}

pub fn runner_glwe_automorphism_key_encrypt_sk<BE: Backend<ZnxWord = i64>, M: Measurement>(
    bencher: &mut Bencher<'_, M>,
    cp: &CoreParams,
) where
    Module<BE>: ModuleNew<BE>
        + GLWEAutomorphismKeyEncryptSk<BE>
        + GLWESecretSampling<BE>
        + ModuleCoreAlloc<OwnedBuf = BE::OwnedBuf, ZnxWord = i64>,
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

    let mut sk: GLWESecret<BE::OwnedBuf, i64> = module.glwe_secret_alloc_from_infos(&atk_infos);
    module.glwe_secret_fill_ternary_prob(&mut sk, 0.5, &mut source_xs);

    let mut atk: GLWEAutomorphismKey<BE::OwnedBuf, i64> = module.glwe_automorphism_key_alloc_from_infos(&atk_infos);
    let mut scratch: ScratchOwned<BE> = ScratchOwned::alloc(module.glwe_automorphism_key_encrypt_sk_tmp_bytes(&atk_infos));

    bencher.iter(|| {
        module.glwe_automorphism_key_encrypt_sk(&mut atk, P, &sk, &mut source_xe, &mut source_xa, &mut scratch.borrow());
        black_box(());
    });
}
