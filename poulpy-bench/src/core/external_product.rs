use poulpy_core::{
    DEFAULT_BOUND_XE, DEFAULT_SIGMA_XE, GGSWEncryptSk, GLWEEncryptSk, GLWEExternalProduct,
    layouts::{
        Base2K, Degree, Dnum, Dsize, GGSW, GGSWLayout, GLWE, GLWEInfos, GLWELayout, GLWESecret, GLWESecretPreparedFactory, GLWESecretSampling, LWEInfos, ModuleCoreAlloc, Rank, TorusPrecision, prepared::{GGSWPrepared, GGSWPreparedFactory, GLWESecretPrepared}
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

pub fn runner_glwe_external_product<BE: Backend<OwnedBuf = Vec<u8>, ZnxWord = i64>, M: Measurement>(
    bencher: &mut Bencher<'_, M>,
    cp: &CoreParams,
) where
    Module<BE>: ModuleNew<BE>
        + GLWEExternalProduct<BE>
        + GGSWEncryptSk<BE>
        + GGSWPreparedFactory<BE>
        + GLWEEncryptSk<BE>
        + GLWESecretPreparedFactory<BE>
        + GLWESecretSampling<BE>,
    ScratchOwned<BE>: ScratchOwnedAlloc<BE> + ScratchOwnedBorrow<BE>,
    for<'a> BE::BufMut<'a>: AsRef<[u8]> + AsMut<[u8]> + Sync,
    for<'a> BE::BufRef<'a>: AsRef<[u8]> + Send,
{
    let glwe_infos = GLWELayout {
        n: Degree(cp.n),
        base2k: Base2K(cp.base2k),
        k: TorusPrecision(cp.k),
        rank: Rank(cp.rank),
    };
    let (dnum, k_aux) = key_dnum_k_aux(cp.k, cp.base2k, cp.dsize);
    let ggsw_infos = GGSWLayout {
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

    let mut sk: GLWESecret<Vec<u8>, i64> = module.glwe_secret_alloc_from_infos(&ggsw_infos);
    module.glwe_secret_fill_ternary_prob(&mut sk, 0.5, &mut source_xs);

    let mut sk_prepared: GLWESecretPrepared<BE::OwnedBuf, BE> = module.glwe_secret_prepared_alloc(ggsw_infos.rank());
    module.glwe_secret_prepare(&mut sk_prepared, &sk);

    let pt = module.scalar_znx_alloc(1);
    let mut ct_ggsw: GGSW<Vec<u8>, i64> = module.ggsw_alloc_from_infos(&ggsw_infos);
    let mut ct_glwe_in: GLWE<Vec<u8>, i64> = module.glwe_alloc_from_infos(&glwe_infos);
    let mut ct_glwe_out: GLWE<Vec<u8>, i64> = module.glwe_alloc_from_infos(&glwe_infos);

    let mut scratch: ScratchOwned<BE> = ScratchOwned::alloc(
        module.ggsw_encrypt_sk_tmp_bytes(&ggsw_infos)
            | module.glwe_encrypt_sk_tmp_bytes(&glwe_infos)
            | module.glwe_external_product_tmp_bytes(&glwe_infos, &glwe_infos, &ggsw_infos),
    );

    let ggsw_enc_infos = NoiseInfos::new(ggsw_infos.k().as_usize(), DEFAULT_SIGMA_XE, DEFAULT_BOUND_XE).unwrap();
    let glwe_enc_infos = NoiseInfos::new(glwe_infos.k().as_usize(), DEFAULT_SIGMA_XE, DEFAULT_BOUND_XE).unwrap();

    module.ggsw_encrypt_sk(
        &mut ct_ggsw,
        &pt,
        &sk_prepared,
        &ggsw_enc_infos,
        &mut source_xe,
        &mut source_xa,
        &mut scratch.borrow(),
    );
    module.glwe_encrypt_zero_sk(
        &mut ct_glwe_in,
        &sk_prepared,
        &glwe_enc_infos,
        &mut source_xe,
        &mut source_xa,
        &mut scratch.borrow(),
    );
    let mut ggsw_prepared: GGSWPrepared<BE::OwnedBuf, BE> = module.ggsw_prepared_alloc_from_infos(&ct_ggsw);
    module.ggsw_prepare(&mut ggsw_prepared, &ct_ggsw, &mut scratch.borrow());

    bencher.iter(|| {
        module.glwe_external_product(&mut ct_glwe_out, &ct_glwe_in, &ggsw_prepared, &mut scratch.borrow());
        black_box(());
    });
}

pub fn runner_glwe_external_product_assign<BE: Backend<OwnedBuf = Vec<u8>, ZnxWord = i64>, M: Measurement>(
    bencher: &mut Bencher<'_, M>,
    cp: &CoreParams,
) where
    Module<BE>: ModuleNew<BE>
        + GLWEExternalProduct<BE>
        + GGSWEncryptSk<BE>
        + GGSWPreparedFactory<BE>
        + GLWEEncryptSk<BE>
        + GLWESecretPreparedFactory<BE>
        + GLWESecretSampling<BE>,
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
    let mut ct_ggsw: GGSW<Vec<u8>, i64> = module.ggsw_alloc_from_infos(&infos);
    let mut ct_glwe: GLWE<Vec<u8>, i64> = module.glwe_alloc_from_infos(&infos);

    let mut scratch: ScratchOwned<BE> = ScratchOwned::alloc(
        module.ggsw_encrypt_sk_tmp_bytes(&infos)
            | module.glwe_encrypt_sk_tmp_bytes(&infos)
            | module.glwe_external_product_tmp_bytes(&infos, &infos, &infos),
    );

    let enc_infos = NoiseInfos::new(infos.k().as_usize(), DEFAULT_SIGMA_XE, DEFAULT_BOUND_XE).unwrap();

    module.ggsw_encrypt_sk(
        &mut ct_ggsw,
        &pt,
        &sk_prepared,
        &enc_infos,
        &mut source_xe,
        &mut source_xa,
        &mut scratch.borrow(),
    );
    module.glwe_encrypt_zero_sk(
        &mut ct_glwe,
        &sk_prepared,
        &enc_infos,
        &mut source_xe,
        &mut source_xa,
        &mut scratch.borrow(),
    );
    let mut ggsw_prepared: GGSWPrepared<BE::OwnedBuf, BE> = module.ggsw_prepared_alloc_from_infos(&ct_ggsw);
    module.ggsw_prepare(&mut ggsw_prepared, &ct_ggsw, &mut scratch.borrow());

    bencher.iter(|| {
        module.glwe_external_product_assign(&mut ct_glwe, &ggsw_prepared, &mut scratch.borrow());
        black_box(());
    });
}
