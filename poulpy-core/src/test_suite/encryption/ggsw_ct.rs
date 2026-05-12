use poulpy_hal::{
    api::{ScratchOwnedAlloc, ScratchOwnedBorrow},
    layouts::{Module, ScalarZnx, ScalarZnxToBackendRef, ScratchOwned},
    source::Source,
    test_suite::TestParams,
};

use crate::{
    EncryptionLayout, GGSWCompressedEncryptSk, GGSWEncryptSk, GGSWNoise, ScratchArenaTakeCore,
    encryption::DEFAULT_SIGMA_XE,
    layouts::{
        GGSW, GGSWDecompress, GGSWInfos, GGSWLayout, GLWEInfos, GLWESecret, GLWESecretPreparedFactory, ModuleCoreAlloc,
        ModuleCoreCompressedAlloc, compressed::GGSWCompressed, prepared::GLWESecretPrepared,
    },
};

pub fn test_ggsw_encrypt_sk<BE: crate::test_suite::TestBackend>(params: &TestParams, module: &Module<BE>)
where
    BE::OwnedBuf: poulpy_hal::layouts::HostDataMut,
    for<'a> BE::BufRef<'a>: poulpy_hal::layouts::HostDataRef,
    for<'a> BE::BufMut<'a>: poulpy_hal::layouts::HostDataMut,
    ScratchOwned<BE>: ScratchOwnedAlloc<BE> + ScratchOwnedBorrow<BE>,
    for<'a> poulpy_hal::layouts::ScratchArena<'a, BE>: ScratchArenaTakeCore<'a, BE>,
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
                k: k.into(),
                dnum: dnum.into(),
                dsize: di.into(),
                rank: rank.into(),
            })
            .unwrap();

            let mut ct: GGSW<Vec<u8>> = module.ggsw_alloc_from_infos(&ggsw_infos);

            let mut pt_scalar: ScalarZnx<Vec<u8>> = module.scalar_znx_alloc(1);

            let mut source_xs: Source = Source::new([0u8; 32]);
            let mut source_xe: Source = Source::new([0u8; 32]);
            let mut source_xa: Source = Source::new([0u8; 32]);

            pt_scalar.fill_ternary_hw(0, n, &mut source_xs);

            let mut scratch: ScratchOwned<BE> = ScratchOwned::alloc(
                (module)
                    .ggsw_encrypt_sk_tmp_bytes(&ggsw_infos)
                    .max(module.ggsw_noise_tmp_bytes(&ggsw_infos)),
            );

            let mut sk: GLWESecret<Vec<u8>> = module.glwe_secret_alloc_from_infos(&ggsw_infos);
            sk.fill_ternary_prob(0.5, &mut source_xs);

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

            let noise_f = |_col_i: usize| -(k as f64) + DEFAULT_SIGMA_XE.log2() + 0.5;

            for row in 0..ct.dnum().as_usize() {
                for col in 0..ct.rank().as_usize() + 1 {
                    assert!(
                        ct.noise(
                            module,
                            row,
                            col,
                            &<ScalarZnx<Vec<u8>> as ScalarZnxToBackendRef<poulpy_hal::layouts::HostBytesBackend>>::to_backend_ref(
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

pub fn test_ggsw_compressed_encrypt_sk<BE: crate::test_suite::TestBackend>(params: &TestParams, module: &Module<BE>)
where
    BE::OwnedBuf: poulpy_hal::layouts::HostDataMut,
    for<'a> BE::BufMut<'a>: poulpy_hal::layouts::HostDataMut,
    for<'a> BE::BufRef<'a>: poulpy_hal::layouts::HostDataRef,
    ScratchOwned<BE>: ScratchOwnedAlloc<BE> + ScratchOwnedBorrow<BE>,
    for<'a> poulpy_hal::layouts::ScratchArena<'a, BE>: ScratchArenaTakeCore<'a, BE>,
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
                k: k.into(),
                dnum: dnum.into(),
                dsize: di.into(),
                rank: rank.into(),
            })
            .unwrap();

            let mut ct_compressed: GGSWCompressed<Vec<u8>> = module.ggsw_compressed_alloc_from_infos(&ggsw_infos);

            let mut pt_scalar: ScalarZnx<Vec<u8>> = module.scalar_znx_alloc(1);

            let mut source_xs: Source = Source::new([0u8; 32]);
            let mut source_xe: Source = Source::new([0u8; 32]);

            pt_scalar.fill_ternary_hw(0, n, &mut source_xs);

            let mut scratch: ScratchOwned<BE> = ScratchOwned::alloc(
                (module)
                    .ggsw_compressed_encrypt_sk_tmp_bytes(&ggsw_infos)
                    .max(module.ggsw_noise_tmp_bytes(&ggsw_infos)),
            );

            let mut sk: GLWESecret<Vec<u8>> = module.glwe_secret_alloc_from_infos(&ggsw_infos);
            sk.fill_ternary_prob(0.5, &mut source_xs);

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

            let noise_f = |_col_i: usize| -(k as f64) + DEFAULT_SIGMA_XE.log2() + 0.5;

            let mut ct: GGSW<Vec<u8>> = module.ggsw_alloc_from_infos(&ggsw_infos);
            module.decompress_ggsw(&mut ct, &ct_compressed);

            for row in 0..ct.dnum().as_usize() {
                for col in 0..ct.rank().as_usize() + 1 {
                    assert!(
                        ct.noise(
                            module,
                            row,
                            col,
                            &<ScalarZnx<Vec<u8>> as ScalarZnxToBackendRef<poulpy_hal::layouts::HostBytesBackend>>::to_backend_ref(
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
