use poulpy_hal::{
    api::{ScratchOwnedAlloc, ScratchOwnedBorrow, VecZnxFillUniformSourceBackend},
    layouts::{Module, ScratchOwned},
    source::Source,
    test_suite::TestParams,
};

use crate::{
    EncryptionLayout, GGLWECompressedEncryptSk, GGLWEEncryptSk, GGLWEKeyswitch, GLWESwitchingKeyCompressedEncryptSk,
    GLWESwitchingKeyEncryptSk, ScratchArenaTakeCore,
    decryption::GLWEDecrypt,
    encryption::DEFAULT_SIGMA_XE,
    layouts::{
        GGLWE, GGLWECompressed, GGLWEDecompress, GGLWEInfos, GGLWELayout, GLWESecret, GLWESecretPreparedFactory,
        GLWESwitchingKey, GLWESwitchingKeyCompressed, GLWESwitchingKeyDecompress, ModuleCoreAlloc, ModuleCoreCompressedAlloc,
        prepared::{GGLWEPreparedFactory, GLWESecretPrepared},
    },
    noise::GGLWENoise,
};

pub fn test_gglwe_switching_key_encrypt_sk<BE: crate::test_suite::TestBackend>(params: &TestParams, module: &Module<BE>)
where
    BE::OwnedBuf: poulpy_hal::layouts::HostDataMut,
    for<'a> BE::BufRef<'a>: poulpy_hal::layouts::HostDataRef,
    for<'a> BE::BufMut<'a>: poulpy_hal::layouts::HostDataMut,
    Module<BE>: GGLWEEncryptSk<BE>
        + GGLWEPreparedFactory<BE>
        + GGLWEKeyswitch<BE>
        + GLWEDecrypt<BE>
        + GLWESecretPreparedFactory<BE>
        + GLWESwitchingKeyEncryptSk<BE>
        + VecZnxFillUniformSourceBackend<BE>
        + GGLWENoise<BE>,
    ScratchOwned<BE>: ScratchOwnedAlloc<BE> + ScratchOwnedBorrow<BE>,
    for<'a> poulpy_hal::layouts::ScratchArena<'a, BE>: ScratchArenaTakeCore<'a, BE>,
{
    let n: usize = module.n();
    let base2k: usize = params.base2k;
    let k_ksk: usize = 4 * base2k + 1;
    let dsize: usize = k_ksk / base2k;
    for rank_in in 1_usize..3 {
        for rank_out in 1_usize..3 {
            for di in 1_usize..dsize + 1 {
                let dnum: usize = (k_ksk - di * base2k) / (di * base2k);

                let gglwe_infos = EncryptionLayout::new_from_default_sigma(GGLWELayout {
                    n: n.into(),
                    base2k: base2k.into(),
                    k: k_ksk.into(),
                    dnum: dnum.into(),
                    dsize: di.into(),
                    rank_in: rank_in.into(),
                    rank_out: rank_out.into(),
                })
                .unwrap();

                let mut ksk: GLWESwitchingKey<Vec<u8>> = module.glwe_switching_key_alloc_from_infos(&gglwe_infos);

                let mut source_xs: Source = Source::new([0u8; 32]);
                let mut source_xe: Source = Source::new([0u8; 32]);
                let mut source_xa: Source = Source::new([0u8; 32]);

                let mut scratch: ScratchOwned<BE> = ScratchOwned::alloc(
                    (module)
                        .glwe_switching_key_encrypt_sk_tmp_bytes(&gglwe_infos)
                        .max(module.gglwe_noise_tmp_bytes(&gglwe_infos)),
                );

                let mut sk_in: GLWESecret<Vec<u8>> = module.glwe_secret_alloc(rank_in.into());
                sk_in.fill_ternary_prob(0.5, &mut source_xs);

                let mut sk_out: GLWESecret<Vec<u8>> = module.glwe_secret_alloc(rank_out.into());
                sk_out.fill_ternary_prob(0.5, &mut source_xs);
                let mut sk_out_prepared: GLWESecretPrepared<BE::OwnedBuf, BE> =
                    module.glwe_secret_prepared_alloc(rank_out.into());
                module.glwe_secret_prepare(&mut sk_out_prepared, &sk_out);

                module.glwe_switching_key_encrypt_sk(
                    &mut ksk,
                    &sk_in,
                    &sk_out,
                    &gglwe_infos,
                    &mut source_xe,
                    &mut source_xa,
                    &mut scratch.arena(),
                );

                let max_noise: f64 = DEFAULT_SIGMA_XE.log2() - (k_ksk as f64) + 0.5;

                for row in 0..ksk.dnum().as_usize() {
                    for col in 0..ksk.rank_in().as_usize() {
                        let noise_have = ksk
                            .key
                            .noise(
                                module,
                                row,
                                col,
                                &sk_in.data.to_ref(),
                                &sk_out_prepared,
                                &mut scratch.borrow(),
                            )
                            .std()
                            .log2();

                        assert!(
                            noise_have <= max_noise,
                            "row:{row} col:{col} noise_have:{noise_have} > max_noise:{max_noise}",
                        );
                    }
                }
            }
        }
    }

    let smaller_n = n / 2;
    if smaller_n > 0 {
        let dnum: usize = (k_ksk - base2k) / base2k;
        let gglwe_infos = EncryptionLayout::new_from_default_sigma(GGLWELayout {
            n: n.into(),
            base2k: base2k.into(),
            k: k_ksk.into(),
            dnum: dnum.into(),
            dsize: 1_u32.into(),
            rank_in: 1_u32.into(),
            rank_out: 1_u32.into(),
        })
        .unwrap();

        let mut ksk: GLWESwitchingKey<Vec<u8>> = module.glwe_switching_key_alloc_from_infos(&gglwe_infos);
        let mut source_xs: Source = Source::new([0u8; 32]);
        let mut source_xe: Source = Source::new([0u8; 32]);
        let mut source_xa: Source = Source::new([0u8; 32]);
        let mut scratch: ScratchOwned<BE> = ScratchOwned::alloc(module.glwe_switching_key_encrypt_sk_tmp_bytes(&gglwe_infos));

        let mut sk_in: GLWESecret<Vec<u8>> = module.glwe_secret_alloc(1_u32.into());
        sk_in.fill_ternary_prob(0.5, &mut source_xs);
        let mut sk_out: GLWESecret<Vec<u8>> = GLWESecret::alloc(smaller_n.into(), 1_u32.into());
        sk_out.fill_ternary_prob(0.5, &mut source_xs);

        module.glwe_switching_key_encrypt_sk(
            &mut ksk,
            &sk_in,
            &sk_out,
            &gglwe_infos,
            &mut source_xe,
            &mut source_xa,
            &mut scratch.arena(),
        );
    }
}

pub fn test_gglwe_switching_key_compressed_encrypt_sk<BE: crate::test_suite::TestBackend>(
    params: &TestParams,
    module: &Module<BE>,
) where
    BE::OwnedBuf: poulpy_hal::layouts::HostDataMut,
    for<'a> BE::BufRef<'a>: poulpy_hal::layouts::HostDataRef,
    for<'a> BE::BufMut<'a>: poulpy_hal::layouts::HostDataMut,
    Module<BE>: GGLWEEncryptSk<BE>
        + GGLWEPreparedFactory<BE>
        + GGLWEKeyswitch<BE>
        + GLWEDecrypt<BE>
        + GLWESecretPreparedFactory<BE>
        + GLWESwitchingKeyEncryptSk<BE>
        + GLWESwitchingKeyCompressedEncryptSk<BE>
        + GLWESwitchingKeyDecompress
        + crate::layouts::compressed::GLWEDecompress<Backend = BE>
        + GGLWENoise<BE>
        + VecZnxFillUniformSourceBackend<BE>,
    ScratchOwned<BE>: ScratchOwnedAlloc<BE> + ScratchOwnedBorrow<BE>,
    for<'a> poulpy_hal::layouts::ScratchArena<'a, BE>: ScratchArenaTakeCore<'a, BE>,
{
    let n: usize = module.n();
    let base2k: usize = params.base2k;
    let k_ksk: usize = 4 * base2k + 1;
    let max_dsize: usize = k_ksk / base2k;
    for rank_in in 1_usize..3 {
        for rank_out in 1_usize..3 {
            for dsize in 1_usize..max_dsize {
                let dnum: usize = (k_ksk - dsize * base2k) / (dsize * base2k);

                let gglwe_infos = EncryptionLayout::new_from_default_sigma(GGLWELayout {
                    n: n.into(),
                    base2k: base2k.into(),
                    k: k_ksk.into(),
                    dnum: dnum.into(),
                    dsize: dsize.into(),
                    rank_in: rank_in.into(),
                    rank_out: rank_out.into(),
                })
                .unwrap();

                let mut ksk_compressed: GLWESwitchingKeyCompressed<Vec<u8>> =
                    module.glwe_switching_key_compressed_alloc_from_infos(&gglwe_infos);

                let mut source_xs: Source = Source::new([0u8; 32]);
                let mut source_xe: Source = Source::new([0u8; 32]);

                let mut scratch: ScratchOwned<BE> = ScratchOwned::alloc(
                    (module)
                        .glwe_switching_key_compressed_encrypt_sk_tmp_bytes(&gglwe_infos)
                        .max(module.gglwe_noise_tmp_bytes(&gglwe_infos)),
                );

                let mut sk_in: GLWESecret<Vec<u8>> = module.glwe_secret_alloc(rank_in.into());
                sk_in.fill_ternary_prob(0.5, &mut source_xs);

                let mut sk_out: GLWESecret<Vec<u8>> = module.glwe_secret_alloc(rank_out.into());
                sk_out.fill_ternary_prob(0.5, &mut source_xs);
                let mut sk_out_prepared: GLWESecretPrepared<BE::OwnedBuf, BE> =
                    module.glwe_secret_prepared_alloc(rank_out.into());
                module.glwe_secret_prepare(&mut sk_out_prepared, &sk_out);

                let seed_xa = [1u8; 32];

                module.glwe_switching_key_compressed_encrypt_sk(
                    &mut ksk_compressed,
                    &sk_in,
                    &sk_out,
                    seed_xa,
                    &gglwe_infos,
                    &mut source_xe,
                    &mut crate::test_suite::scratch_host_arena(&mut scratch),
                );

                let mut ksk: GLWESwitchingKey<Vec<u8>> = module.glwe_switching_key_alloc_from_infos(&gglwe_infos);
                module.decompress_glwe_switching_key(&mut ksk, &ksk_compressed);

                let max_noise: f64 = DEFAULT_SIGMA_XE.log2() - (k_ksk as f64) + 0.5;

                for row in 0..ksk.dnum().as_usize() {
                    for col in 0..ksk.rank_in().as_usize() {
                        let noise_have = ksk
                            .key
                            .noise(
                                module,
                                row,
                                col,
                                &sk_in.data.to_ref(),
                                &sk_out_prepared,
                                &mut scratch.borrow(),
                            )
                            .std()
                            .log2();

                        assert!(
                            noise_have <= max_noise,
                            "row:{row} col:{col} noise_have:{noise_have} > max_noise:{max_noise}",
                        );
                    }
                }
            }
        }
    }

    let smaller_n = n / 2;
    if smaller_n > 0 {
        let dnum: usize = (k_ksk - base2k) / base2k;
        let gglwe_infos = EncryptionLayout::new_from_default_sigma(GGLWELayout {
            n: n.into(),
            base2k: base2k.into(),
            k: k_ksk.into(),
            dnum: dnum.into(),
            dsize: 1_u32.into(),
            rank_in: 1_u32.into(),
            rank_out: 1_u32.into(),
        })
        .unwrap();

        let mut ksk_compressed: GLWESwitchingKeyCompressed<Vec<u8>> =
            module.glwe_switching_key_compressed_alloc_from_infos(&gglwe_infos);
        let mut source_xs: Source = Source::new([0u8; 32]);
        let mut source_xe: Source = Source::new([0u8; 32]);
        let seed_xa = [1u8; 32];
        let mut scratch: ScratchOwned<BE> =
            ScratchOwned::alloc(module.glwe_switching_key_compressed_encrypt_sk_tmp_bytes(&gglwe_infos));

        let mut sk_in: GLWESecret<Vec<u8>> = module.glwe_secret_alloc(1_u32.into());
        sk_in.fill_ternary_prob(0.5, &mut source_xs);
        let mut sk_out: GLWESecret<Vec<u8>> = GLWESecret::alloc(smaller_n.into(), 1_u32.into());
        sk_out.fill_ternary_prob(0.5, &mut source_xs);

        module.glwe_switching_key_compressed_encrypt_sk(
            &mut ksk_compressed,
            &sk_in,
            &sk_out,
            seed_xa,
            &gglwe_infos,
            &mut source_xe,
            &mut crate::test_suite::scratch_host_arena(&mut scratch),
        );
    }
}

pub fn test_gglwe_compressed_encrypt_sk<BE: crate::test_suite::TestBackend>(params: &TestParams, module: &Module<BE>)
where
    BE::OwnedBuf: poulpy_hal::layouts::HostDataMut,
    for<'a> BE::BufRef<'a>: poulpy_hal::layouts::HostDataRef,
    for<'a> BE::BufMut<'a>: poulpy_hal::layouts::HostDataMut,
    Module<BE>: GGLWEEncryptSk<BE>
        + GGLWEPreparedFactory<BE>
        + GGLWEKeyswitch<BE>
        + GLWEDecrypt<BE>
        + GLWESecretPreparedFactory<BE>
        + GLWESwitchingKeyEncryptSk<BE>
        + GGLWECompressedEncryptSk<BE>
        + GLWESwitchingKeyDecompress
        + crate::layouts::compressed::GLWEDecompress<Backend = BE>
        + GGLWENoise<BE>
        + VecZnxFillUniformSourceBackend<BE>,
    ScratchOwned<BE>: ScratchOwnedAlloc<BE> + ScratchOwnedBorrow<BE>,
    for<'a> poulpy_hal::layouts::ScratchArena<'a, BE>: ScratchArenaTakeCore<'a, BE>,
{
    let n: usize = module.n();
    let base2k: usize = params.base2k;
    let k_ksk: usize = 4 * base2k + 1;
    let max_dsize: usize = k_ksk / base2k;
    for rank_in in 1_usize..3 {
        for rank_out in 1_usize..3 {
            for dsize in 1_usize..max_dsize + 1 {
                let dnum: usize = (k_ksk - dsize * base2k) / (dsize * base2k);

                let gglwe_infos = EncryptionLayout::new_from_default_sigma(GGLWELayout {
                    n: n.into(),
                    base2k: base2k.into(),
                    k: k_ksk.into(),
                    dnum: dnum.into(),
                    dsize: dsize.into(),
                    rank_in: rank_in.into(),
                    rank_out: rank_out.into(),
                })
                .unwrap();

                let mut ksk_compressed: GGLWECompressed<Vec<u8>> = module.gglwe_compressed_alloc_from_infos(&gglwe_infos);

                let mut source_xs: Source = Source::new([0u8; 32]);
                let mut source_xe: Source = Source::new([0u8; 32]);

                let mut scratch: ScratchOwned<BE> = ScratchOwned::alloc(
                    (module)
                        .gglwe_compressed_encrypt_sk_tmp_bytes(&gglwe_infos)
                        .max(module.gglwe_noise_tmp_bytes(&gglwe_infos)),
                );

                let mut sk_in: GLWESecret<Vec<u8>> = module.glwe_secret_alloc(rank_in.into());
                sk_in.fill_ternary_prob(0.5, &mut source_xs);

                let mut sk_out: GLWESecret<Vec<u8>> = module.glwe_secret_alloc(rank_out.into());
                sk_out.fill_ternary_prob(0.5, &mut source_xs);
                let mut sk_out_prepared: GLWESecretPrepared<BE::OwnedBuf, BE> =
                    module.glwe_secret_prepared_alloc(rank_out.into());
                module.glwe_secret_prepare(&mut sk_out_prepared, &sk_out);

                let seed_xa = [1u8; 32];

                module.gglwe_compressed_encrypt_sk(
                    &mut ksk_compressed,
                    &sk_in.data,
                    &sk_out_prepared,
                    seed_xa,
                    &gglwe_infos,
                    &mut source_xe,
                    &mut scratch.borrow(),
                );

                let mut ksk: GGLWE<Vec<u8>> = module.gglwe_alloc_from_infos(&gglwe_infos);
                module.decompress_gglwe(&mut ksk, &ksk_compressed);

                let max_noise: f64 = DEFAULT_SIGMA_XE.log2() - (k_ksk as f64) + 0.5;

                for row in 0..ksk.dnum().as_usize() {
                    for col in 0..ksk.rank_in().as_usize() {
                        let noise_have = ksk
                            .noise(
                                module,
                                row,
                                col,
                                &sk_in.data.to_ref(),
                                &sk_out_prepared,
                                &mut scratch.borrow(),
                            )
                            .std()
                            .log2();

                        assert!(
                            noise_have <= max_noise,
                            "row:{row} col:{col} noise_have:{noise_have} > max_noise:{max_noise}",
                        );
                    }
                }
            }
        }
    }
}
