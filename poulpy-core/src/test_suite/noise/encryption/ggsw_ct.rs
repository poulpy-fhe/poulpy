use poulpy_hal::AlignedBuf;
use poulpy_hal::{
    api::{ScratchOwnedAlloc, ScratchOwnedBorrow},
    layouts::{Module, ScalarZnx, ScalarZnxToBackendRef, ScratchOwned},
    source::Source,
    test_suite::TestParams,
};

use crate::layouts::GLWESecretSampling;
use crate::{Distribution, ScalarZnxFillDistribution};
use crate::{
    EncryptionLayout, GGSWCompressedEncryptSk, GGSWEncryptPk, GGSWEncryptSk, GGSWNoise, GLWEPublicKeyGenerate,
    encryption::DEFAULT_SIGMA_XE,
    layouts::{
        GGSW, GGSWDecompress, GGSWInfos, GGSWLayout, GLWEInfos, GLWELayout, GLWEPublicKey, GLWEPublicKeyPreparedFactory,
        GLWESecret, GLWESecretPreparedFactory, LWEInfos, ModuleCoreAlloc, ModuleCoreCompressedAlloc,
        compressed::GGSWCompressed,
        prepared::{GLWEPublicKeyPrepared, GLWESecretPrepared},
    },
};
use poulpy_hal::test_suite::scalar_znx_backend_mut;

pub fn test_ggsw_encrypt_sk<BE: crate::test_suite::noise::TestBackend>(params: &TestParams, module: &Module<BE>)
where
    BE::OwnedBuf: poulpy_hal::layouts::HostDataMut,
    for<'a> BE::BufRef<'a>: poulpy_hal::layouts::HostDataRef,
    for<'a> BE::BufMut<'a>: poulpy_hal::layouts::HostDataMut,
    ScratchOwned<BE>: ScratchOwnedAlloc<BE> + ScratchOwnedBorrow<BE>,
    Module<BE>: GGSWEncryptSk<BE> + GLWESecretPreparedFactory<BE> + GGSWNoise<BE>,
{
    let base2k: usize = params.base2k;
    let k: usize = 4 * base2k + 1;
    let dsize: usize = k / base2k;
    for rank in 1_usize..3 {
        for di in 1..dsize + 1 {
            let n: usize = module.n();
            let dnum: usize = (k - di * base2k) / (di * base2k);

            let ggsw_infos = EncryptionLayout::new_from_default_sigma(GGSWLayout {
                n: n.into(),
                base2k: base2k.into(),
                dnum: dnum.into(),
                k_aux: (di * base2k + module.log_n()).into(),
                dsize: di.into(),
                rank: rank.into(),
            })
            .unwrap();

            let mut ct: GGSW<BE::OwnedBuf, BE::ZnxWord> = module.ggsw_alloc_from_infos(&ggsw_infos);

            let mut pt_scalar: ScalarZnx<BE::OwnedBuf, BE::ZnxWord> = module.scalar_znx_alloc(module.n(), 1);

            let mut source_xs: Source = Source::new([0u8; 32]);
            let mut source_xe: Source = Source::new([0u8; 32]);
            let mut source_xa: Source = Source::new([0u8; 32]);

            module.scalar_znx_fill_distribution(
                &mut scalar_znx_backend_mut::<BE>(&mut pt_scalar),
                0,
                Distribution::TernaryFixed(n),
                &mut source_xs,
            );

            let mut scratch: ScratchOwned<BE> = ScratchOwned::alloc(
                (module)
                    .ggsw_encrypt_sk_tmp_bytes(&ggsw_infos)
                    .max(module.ggsw_noise_tmp_bytes(&ggsw_infos)),
            );

            let mut sk: GLWESecret<BE::OwnedBuf, BE::ZnxWord> = module.glwe_secret_alloc_from_infos(&ggsw_infos);
            module.glwe_secret_fill_ternary_prob(&mut sk, 0.5, &mut source_xs);

            let mut sk_prepared: GLWESecretPrepared<BE::OwnedBuf, BE> = module.glwe_secret_prepared_alloc(rank.into());
            module.glwe_secret_prepare(&mut sk_prepared, &sk);

            module.ggsw_encrypt_sk(
                &mut ct,
                &pt_scalar,
                &sk_prepared,
                &ggsw_infos,
                &mut source_xe,
                &mut source_xa,
                &mut scratch.borrow(),
            );

            let noise_f = |_col_i: usize| -(ggsw_infos.k().as_usize() as f64) + DEFAULT_SIGMA_XE.log2() + 0.5;

            for row in 0..ct.dnum().as_usize() {
                for col in 0..ct.rank().as_usize() + 1 {
                    assert!(
                        ct.noise(
                            module,
                            row,
                            col,
                            &<ScalarZnx<AlignedBuf, i64> as ScalarZnxToBackendRef<poulpy_hal::layouts::HostBytesBackend>>::to_backend_ref(
                                &pt_scalar,
                            ),
                            &sk_prepared,
                            &mut scratch.borrow(),
                        )
                        .std()
                        .log2()
                            <= noise_f(col)
                    )
                }
            }
        }
    }
}

pub fn test_ggsw_encrypt_pk<BE: crate::test_suite::noise::TestBackend>(params: &TestParams, module: &Module<BE>)
where
    BE::OwnedBuf: poulpy_hal::layouts::HostDataMut,
    for<'a> BE::BufRef<'a>: poulpy_hal::layouts::HostDataRef,
    for<'a> BE::BufMut<'a>: poulpy_hal::layouts::HostDataMut,
    ScratchOwned<BE>: ScratchOwnedAlloc<BE> + ScratchOwnedBorrow<BE>,
    Module<BE>: GGSWEncryptPk<BE>
        + GLWEPublicKeyGenerate<BE>
        + GLWEPublicKeyPreparedFactory<BE>
        + GLWESecretPreparedFactory<BE>
        + GGSWNoise<BE>,
{
    let base2k: usize = params.base2k;
    let k: usize = 4 * base2k + 1;
    let dsize: usize = k / base2k;
    for rank in 1_usize..3 {
        for di in 1..dsize + 1 {
            let n: usize = module.n();
            let dnum: usize = (k - di * base2k) / (di * base2k);

            let ggsw_infos = EncryptionLayout::new_from_default_sigma(GGSWLayout {
                n: n.into(),
                base2k: base2k.into(),
                dnum: dnum.into(),
                k_aux: (di * base2k + module.log_n()).into(),
                dsize: di.into(),
                rank: rank.into(),
            })
            .unwrap();
            let pk_infos = EncryptionLayout::new_from_default_sigma(GLWELayout {
                n: n.into(),
                base2k: base2k.into(),
                k: ggsw_infos.k(),
                rank: rank.into(),
            })
            .unwrap();

            let mut ct: GGSW<BE::OwnedBuf, BE::ZnxWord> = module.ggsw_alloc_from_infos(&ggsw_infos);
            let mut pt_scalar: ScalarZnx<BE::OwnedBuf, BE::ZnxWord> = module.scalar_znx_alloc(module.n(), 1);

            let mut source_xs: Source = Source::new([0u8; 32]);
            let mut source_xe: Source = Source::new([0u8; 32]);
            let mut source_xa: Source = Source::new([0u8; 32]);
            let mut source_xu: Source = Source::new([0u8; 32]);

            module.scalar_znx_fill_distribution(
                &mut scalar_znx_backend_mut::<BE>(&mut pt_scalar),
                0,
                Distribution::TernaryFixed(n),
                &mut source_xs,
            );

            let mut scratch: ScratchOwned<BE> = ScratchOwned::alloc(
                module
                    .ggsw_noise_tmp_bytes(&ggsw_infos)
                    .max(module.glwe_public_key_generate_tmp_bytes(&pk_infos))
                    .max(module.glwe_public_key_prepare_tmp_bytes(&pk_infos)),
            );
            let mut enc_scratch: ScratchOwned<BE> = ScratchOwned::alloc(module.ggsw_encrypt_pk_tmp_bytes(&ggsw_infos, &pk_infos));

            let mut sk: GLWESecret<BE::OwnedBuf, BE::ZnxWord> = module.glwe_secret_alloc_from_infos(&ggsw_infos);
            module.glwe_secret_fill_ternary_prob(&mut sk, 0.5, &mut source_xs);
            let mut sk_prepared: GLWESecretPrepared<BE::OwnedBuf, BE> = module.glwe_secret_prepared_alloc(rank.into());
            module.glwe_secret_prepare(&mut sk_prepared, &sk);

            let mut pk: GLWEPublicKey<BE::OwnedBuf, BE::ZnxWord> = module.glwe_public_key_alloc_from_infos(&pk_infos);
            module.glwe_public_key_generate(
                &mut pk,
                &sk_prepared,
                &pk_infos,
                &mut source_xe,
                &mut source_xa,
                &mut scratch.borrow(),
            );
            let mut pk_prepared: GLWEPublicKeyPrepared<BE::OwnedBuf, BE> =
                module.glwe_public_key_prepared_alloc_from_infos(&pk_infos);
            module.glwe_public_key_prepare(&mut pk_prepared, &pk, &mut scratch.borrow());

            module.ggsw_encrypt_pk(
                &mut ct,
                &pt_scalar,
                &pk_prepared,
                &ggsw_infos,
                &mut source_xu,
                &mut source_xe,
                &mut enc_scratch.borrow(),
            );

            // Sum_l u_l e_l has rank terms, as Sum_j e_j s_j does.
            let noise_want: f64 = ((2.0 * rank as f64 * n as f64 * 0.5 * DEFAULT_SIGMA_XE * DEFAULT_SIGMA_XE).sqrt()).log2()
                - (ggsw_infos.k().as_usize() as f64)
                + 0.5;
            for row in 0..ct.dnum().as_usize() {
                for col in 0..ct.rank().as_usize() + 1 {
                    let noise_have: f64 = ct
                        .noise(
                            module,
                            row,
                            col,
                            &<ScalarZnx<AlignedBuf, i64> as ScalarZnxToBackendRef<poulpy_hal::layouts::HostBytesBackend>>::to_backend_ref(
                                &pt_scalar,
                            ),
                            &sk_prepared,
                            &mut scratch.borrow(),
                        )
                        .std()
                        .log2();
                    assert!(
                        noise_have <= noise_want,
                        "row {row} col {col}: noise {noise_have} > {noise_want}"
                    );
                }
            }
        }
    }
}

pub fn test_ggsw_compressed_encrypt_sk<BE: crate::test_suite::noise::TestBackend>(params: &TestParams, module: &Module<BE>)
where
    BE::OwnedBuf: poulpy_hal::layouts::HostDataMut,
    for<'a> BE::BufMut<'a>: poulpy_hal::layouts::HostDataMut,
    for<'a> BE::BufRef<'a>: poulpy_hal::layouts::HostDataRef,
    ScratchOwned<BE>: ScratchOwnedAlloc<BE> + ScratchOwnedBorrow<BE>,
    Module<BE>: GGSWCompressedEncryptSk<BE>
        + GLWESecretPreparedFactory<BE>
        + GGSWNoise<BE>
        + GGSWDecompress
        + crate::layouts::compressed::GLWEDecompress<Backend = BE>,
{
    let base2k: usize = params.base2k;
    let k: usize = 4 * base2k + 1;
    let dsize: usize = k / base2k;
    for rank in 1_usize..3 {
        for di in 1..dsize + 1 {
            let n: usize = module.n();
            let dnum: usize = (k - di * base2k) / (di * base2k);

            let ggsw_infos = EncryptionLayout::new_from_default_sigma(GGSWLayout {
                n: n.into(),
                base2k: base2k.into(),
                dnum: dnum.into(),
                k_aux: (di * base2k + module.log_n()).into(),
                dsize: di.into(),
                rank: rank.into(),
            })
            .unwrap();

            let mut ct_compressed: GGSWCompressed<BE::OwnedBuf, BE::ZnxWord> =
                module.ggsw_compressed_alloc_from_infos(&ggsw_infos);

            let mut pt_scalar: ScalarZnx<BE::OwnedBuf, BE::ZnxWord> = module.scalar_znx_alloc(module.n(), 1);

            let mut source_xs: Source = Source::new([0u8; 32]);
            let mut source_xe: Source = Source::new([0u8; 32]);

            module.scalar_znx_fill_distribution(
                &mut scalar_znx_backend_mut::<BE>(&mut pt_scalar),
                0,
                Distribution::TernaryFixed(n),
                &mut source_xs,
            );

            let mut scratch: ScratchOwned<BE> = ScratchOwned::alloc(
                (module)
                    .ggsw_compressed_encrypt_sk_tmp_bytes(&ggsw_infos)
                    .max(module.ggsw_noise_tmp_bytes(&ggsw_infos)),
            );

            let mut sk: GLWESecret<BE::OwnedBuf, BE::ZnxWord> = module.glwe_secret_alloc_from_infos(&ggsw_infos);
            module.glwe_secret_fill_ternary_prob(&mut sk, 0.5, &mut source_xs);

            let mut sk_prepared: GLWESecretPrepared<BE::OwnedBuf, BE> = module.glwe_secret_prepared_alloc(rank.into());
            module.glwe_secret_prepare(&mut sk_prepared, &sk);

            let seed_xa: [u8; 32] = [1u8; 32];

            module.ggsw_compressed_encrypt_sk(
                &mut ct_compressed,
                &pt_scalar,
                &sk_prepared,
                seed_xa,
                &ggsw_infos,
                &mut source_xe,
                &mut scratch.borrow(),
            );

            let noise_f = |_col_i: usize| -(ggsw_infos.k().as_usize() as f64) + DEFAULT_SIGMA_XE.log2() + 0.5;

            let mut ct: GGSW<BE::OwnedBuf, BE::ZnxWord> = module.ggsw_alloc_from_infos(&ggsw_infos);
            module.decompress_ggsw(&mut ct, &ct_compressed);

            for row in 0..ct.dnum().as_usize() {
                for col in 0..ct.rank().as_usize() + 1 {
                    assert!(
                        ct.noise(
                            module,
                            row,
                            col,
                            &<ScalarZnx<AlignedBuf, i64> as ScalarZnxToBackendRef<poulpy_hal::layouts::HostBytesBackend>>::to_backend_ref(
                                &pt_scalar,
                            ),
                            &sk_prepared,
                            &mut scratch.borrow(),
                        )
                        .std()
                        .log2()
                            <= noise_f(col)
                    )
                }
            }
        }
    }
}
