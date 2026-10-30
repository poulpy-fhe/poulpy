use poulpy_hal::AlignedBuf;
use poulpy_hal::{
    api::{ScalarZnxAlloc, ScratchOwnedAlloc, ScratchOwnedBorrow, VecZnxFillUniformSource},
    layouts::{DataViewMut, Module, ReaderFrom, ScratchOwned, WriterTo},
    source::Source,
    test_suite::TestParams,
};

use crate::api::GLWESecretSampling;
use crate::test_suite::poisoned_scratch;
use crate::test_suite::refusal::assert_refusal;
use crate::{
    Distribution, GGLWECompressedEncryptSk, GGLWEDecompress, GGLWEEncryptSk, GGLWEKeyswitch, GLWESwitchingKeyCompressedEncryptSk,
    GLWESwitchingKeyDecompress, GLWESwitchingKeyEncryptSk, GetDistributionMut,
    api::{GGLWEPreparedFactory, GLWESecretPreparedFactory},
    decryption::GLWEDecrypt,
    encryption::DEFAULT_SIGMA_XE,
    layouts::{
        GGLWE, GGLWEAlloc, GGLWECompressed, GGLWECompressedAlloc, GGLWEInfos, GGLWELayout, GLWESecret, GLWESecretAlloc,
        GLWESwitchingKey, GLWESwitchingKeyAlloc, GLWESwitchingKeyCompressed, GLWESwitchingKeyCompressedAlloc, LWEInfos,
        prepared::GLWESecretPrepared,
    },
    noise::GGLWENoise,
};

pub fn test_gglwe_switching_key_encrypt_sk<BE: crate::test_suite::noise::TestBackend>(params: &TestParams, module: &Module<BE>)
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
        + VecZnxFillUniformSource<BE>
        + GGLWENoise<BE>,
    ScratchOwned<BE>: ScratchOwnedAlloc<BE> + ScratchOwnedBorrow<BE>,
{
    let n: usize = module.n();
    let base2k: usize = params.base2k;
    let k_ksk: usize = 4 * base2k + 1;
    let dsize: usize = k_ksk / base2k;
    // Offset-free keys, and keys whose rows sit an aligned or unaligned offset lower.
    for k_offset in [0, base2k, base2k + 5] {
        for rank_in in 1_usize..3 {
            for rank_out in 1_usize..3 {
                for di in 1_usize..dsize + 1 {
                    let dnum: usize = (k_ksk - di * base2k) / (di * base2k);

                    let gglwe_infos = GGLWELayout {
                        n: n.into(),
                        base2k: base2k.into(),
                        dnum: dnum.into(),
                        k_aux: (di * base2k + module.log_n()).into(),
                        k_digit: crate::layouts::k_digit_of(base2k, di),
                        k_offset: k_offset.into(),
                        rank_in: rank_in.into(),
                        rank_out: rank_out.into(),
                        stride: 1,
                    };

                    let mut ksk: GLWESwitchingKey<BE::OwnedBuf, BE::ZnxWord> =
                        module.glwe_switching_key_alloc_from_infos(&gglwe_infos);

                    let mut source_xs: Source = Source::new([0u8; 32]);
                    let mut source_xe: Source = Source::new([0u8; 32]);
                    let mut source_xa: Source = Source::new([0u8; 32]);

                    let mut scratch: ScratchOwned<BE> = ScratchOwned::alloc(module.gglwe_noise_tmp_bytes(&gglwe_infos));

                    let mut sk_in: GLWESecret<BE::OwnedBuf, BE::ZnxWord> = module.glwe_secret_alloc(rank_in.into());
                    module.glwe_secret_fill_ternary_prob(&mut sk_in, 0.5, &mut source_xs);

                    let mut sk_out: GLWESecret<BE::OwnedBuf, BE::ZnxWord> = module.glwe_secret_alloc(rank_out.into());
                    module.glwe_secret_fill_ternary_prob(&mut sk_out, 0.5, &mut source_xs);
                    *sk_out.dist_mut() = Distribution::ENCAPSULATED("roundtrip");
                    let mut sk_out_prepared: GLWESecretPrepared<BE::OwnedBuf, BE> =
                        module.glwe_secret_prepared_alloc(rank_out.into());
                    module.glwe_secret_prepare(&mut sk_out_prepared, &sk_out);

                    module.glwe_switching_key_encrypt_sk(
                        &mut ksk,
                        &sk_in,
                        &sk_out,
                        &mut source_xe,
                        &mut source_xa,
                        &mut poisoned_scratch::<BE>(module.glwe_switching_key_encrypt_sk_tmp_bytes(&gglwe_infos)).borrow(),
                    );

                    let mut bytes = Vec::new();
                    ksk.write_to(&mut bytes).unwrap();
                    let mut restored = module.glwe_switching_key_alloc_from_infos(&gglwe_infos);
                    restored.read_from(&mut bytes.as_slice()).unwrap();
                    assert!(restored == ksk);
                    assert_eq!(ksk.noise().unwrap().secret_distribution().base(), Distribution::NONE);

                    let max_noise: f64 = DEFAULT_SIGMA_XE.log2() - (gglwe_infos.k().as_usize() as f64) + 0.5;

                    for row in 0..ksk.dnum().as_usize() {
                        for col in 0..ksk.rank_in().as_usize() {
                            let noise_have = ksk
                                .key
                                .noise_stats(
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

    let smaller_n = n / 2;
    if smaller_n > 0 {
        let dnum: usize = (k_ksk - base2k) / base2k;
        let gglwe_infos = GGLWELayout {
            n: n.into(),
            base2k: base2k.into(),
            dnum: dnum.into(),
            k_aux: (base2k + module.log_n()).into(),
            k_digit: crate::layouts::k_digit_of(base2k, 1_u32),
            k_offset: crate::layouts::TorusPrecision(0),
            rank_in: 1_u32.into(),
            rank_out: 1_u32.into(),
            stride: 1,
        };

        let mut ksk: GLWESwitchingKey<BE::OwnedBuf, BE::ZnxWord> = module.glwe_switching_key_alloc_from_infos(&gglwe_infos);
        let mut source_xs: Source = Source::new([0u8; 32]);
        let mut source_xe: Source = Source::new([0u8; 32]);
        let mut source_xa: Source = Source::new([0u8; 32]);
        let mut sk_in: GLWESecret<BE::OwnedBuf, BE::ZnxWord> = module.glwe_secret_alloc(1_u32.into());
        module.glwe_secret_fill_ternary_prob(&mut sk_in, 0.5, &mut source_xs);
        let mut sk_out: GLWESecret<AlignedBuf, i64> = GLWESecret::alloc(smaller_n.into(), 1_u32.into());
        module.glwe_secret_fill_ternary_prob(&mut sk_out, 0.5, &mut source_xs);

        // The lifted and prepared secrets would be left in the scratch.
        crate::test_suite::assert_wipes_scratch::<BE>(module.glwe_switching_key_encrypt_sk_tmp_bytes(&gglwe_infos), |scratch| {
            module.glwe_switching_key_encrypt_sk(&mut ksk, &sk_in, &sk_out, &mut source_xe, &mut source_xa, scratch)
        });
    }
}

pub fn test_gglwe_switching_key_compressed_encrypt_sk<BE: crate::test_suite::noise::TestBackend>(
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
        + GLWESwitchingKeyDecompress<BE>
        + GGLWENoise<BE>
        + VecZnxFillUniformSource<BE>,
    ScratchOwned<BE>: ScratchOwnedAlloc<BE> + ScratchOwnedBorrow<BE>,
{
    let n: usize = module.n();
    let base2k: usize = params.base2k;
    let k_ksk: usize = 4 * base2k + 1;
    let max_dsize: usize = k_ksk / base2k;
    // Offset-free keys, and keys whose rows sit an aligned or unaligned offset lower.
    for k_offset in [0, base2k, base2k + 5] {
        for rank_in in 1_usize..3 {
            for rank_out in 1_usize..3 {
                for dsize in 1_usize..max_dsize {
                    let dnum: usize = (k_ksk - dsize * base2k) / (dsize * base2k);

                    let gglwe_infos = GGLWELayout {
                        n: n.into(),
                        base2k: base2k.into(),
                        dnum: dnum.into(),
                        k_aux: (dsize * base2k + module.log_n()).into(),
                        k_digit: crate::layouts::k_digit_of(base2k, dsize),
                        k_offset: k_offset.into(),
                        rank_in: rank_in.into(),
                        rank_out: rank_out.into(),
                        stride: 1,
                    };

                    let mut ksk_compressed: GLWESwitchingKeyCompressed<BE::OwnedBuf, BE::ZnxWord> =
                        module.glwe_switching_key_compressed_alloc_from_infos(&gglwe_infos);

                    let mut source_xs: Source = Source::new([0u8; 32]);
                    let mut source_xe: Source = Source::new([0u8; 32]);

                    let mut scratch: ScratchOwned<BE> = ScratchOwned::alloc(module.gglwe_noise_tmp_bytes(&gglwe_infos));

                    let mut sk_in: GLWESecret<BE::OwnedBuf, BE::ZnxWord> = module.glwe_secret_alloc(rank_in.into());
                    module.glwe_secret_fill_ternary_prob(&mut sk_in, 0.5, &mut source_xs);

                    let mut sk_out: GLWESecret<BE::OwnedBuf, BE::ZnxWord> = module.glwe_secret_alloc(rank_out.into());
                    module.glwe_secret_fill_ternary_prob(&mut sk_out, 0.5, &mut source_xs);
                    let mut sk_out_prepared: GLWESecretPrepared<BE::OwnedBuf, BE> =
                        module.glwe_secret_prepared_alloc(rank_out.into());
                    module.glwe_secret_prepare(&mut sk_out_prepared, &sk_out);

                    let seed_xa = [1u8; 32];

                    module.glwe_switching_key_compressed_encrypt_sk(
                        &mut ksk_compressed,
                        &sk_in,
                        &sk_out,
                        seed_xa,
                        &mut source_xe,
                        &mut poisoned_scratch::<BE>(module.glwe_switching_key_compressed_encrypt_sk_tmp_bytes(&gglwe_infos))
                            .borrow(),
                    );

                    let mut bytes = Vec::new();
                    ksk_compressed.write_to(&mut bytes).unwrap();
                    let mut restored = module.glwe_switching_key_compressed_alloc_from_infos(&gglwe_infos);
                    restored.read_from(&mut bytes.as_slice()).unwrap();
                    assert!(restored == ksk_compressed);
                    assert_eq!(ksk_compressed.k_offset(), gglwe_infos.k_offset);

                    let mut ksk: GLWESwitchingKey<BE::OwnedBuf, BE::ZnxWord> =
                        module.glwe_switching_key_alloc_from_infos(&gglwe_infos);
                    module.glwe_switching_key_decompress(&mut ksk, &ksk_compressed);

                    let max_noise: f64 = DEFAULT_SIGMA_XE.log2() - (gglwe_infos.k().as_usize() as f64) + 0.5;

                    for row in 0..ksk.dnum().as_usize() {
                        for col in 0..ksk.rank_in().as_usize() {
                            let noise_have = ksk
                                .key
                                .noise_stats(
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

    let smaller_n = n / 2;
    if smaller_n > 0 {
        let dnum: usize = (k_ksk - base2k) / base2k;
        let gglwe_infos = GGLWELayout {
            n: n.into(),
            base2k: base2k.into(),
            dnum: dnum.into(),
            k_aux: (base2k + module.log_n()).into(),
            k_digit: crate::layouts::k_digit_of(base2k, 1_u32),
            k_offset: crate::layouts::TorusPrecision(0),
            rank_in: 1_u32.into(),
            rank_out: 1_u32.into(),
            stride: 1,
        };

        let mut ksk_compressed: GLWESwitchingKeyCompressed<BE::OwnedBuf, BE::ZnxWord> =
            module.glwe_switching_key_compressed_alloc_from_infos(&gglwe_infos);
        let mut source_xs: Source = Source::new([0u8; 32]);
        let mut source_xe: Source = Source::new([0u8; 32]);
        let seed_xa = [1u8; 32];
        let mut scratch = poisoned_scratch::<BE>(module.glwe_switching_key_compressed_encrypt_sk_tmp_bytes(&gglwe_infos));

        let mut sk_in: GLWESecret<BE::OwnedBuf, BE::ZnxWord> = module.glwe_secret_alloc(1_u32.into());
        module.glwe_secret_fill_ternary_prob(&mut sk_in, 0.5, &mut source_xs);
        let mut sk_out: GLWESecret<AlignedBuf, i64> = GLWESecret::alloc(smaller_n.into(), 1_u32.into());
        module.glwe_secret_fill_ternary_prob(&mut sk_out, 0.5, &mut source_xs);

        module.glwe_switching_key_compressed_encrypt_sk(
            &mut ksk_compressed,
            &sk_in,
            &sk_out,
            seed_xa,
            &mut source_xe,
            &mut crate::test_suite::noise::scratch_host_arena(&mut scratch),
        );
    }
}

pub fn test_gglwe_compressed_encrypt_sk<BE: crate::test_suite::noise::TestBackend>(params: &TestParams, module: &Module<BE>)
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
        + GGLWEDecompress<BE>
        + GGLWENoise<BE>
        + VecZnxFillUniformSource<BE>,
    ScratchOwned<BE>: ScratchOwnedAlloc<BE> + ScratchOwnedBorrow<BE>,
{
    let n: usize = module.n();
    let base2k: usize = params.base2k;
    let k_ksk: usize = 4 * base2k + 1;
    let max_dsize: usize = k_ksk / base2k;
    // Offset-free keys, and keys whose rows sit an aligned or unaligned offset lower.
    for k_offset in [0, base2k, base2k + 5] {
        for rank_in in 1_usize..3 {
            for rank_out in 1_usize..3 {
                for dsize in 1_usize..max_dsize + 1 {
                    let dnum: usize = (k_ksk - dsize * base2k) / (dsize * base2k);

                    let gglwe_infos = GGLWELayout {
                        n: n.into(),
                        base2k: base2k.into(),
                        dnum: dnum.into(),
                        k_aux: (dsize * base2k + module.log_n()).into(),
                        k_digit: crate::layouts::k_digit_of(base2k, dsize),
                        k_offset: k_offset.into(),
                        rank_in: rank_in.into(),
                        rank_out: rank_out.into(),
                        stride: 1,
                    };

                    let mut ksk_compressed: GGLWECompressed<BE::OwnedBuf, BE::ZnxWord> =
                        module.gglwe_compressed_alloc_from_infos(&gglwe_infos);

                    let mut source_xs: Source = Source::new([0u8; 32]);
                    let mut source_xe: Source = Source::new([0u8; 32]);

                    let mut scratch: ScratchOwned<BE> = ScratchOwned::alloc(module.gglwe_noise_tmp_bytes(&gglwe_infos));

                    let mut sk_in: GLWESecret<BE::OwnedBuf, BE::ZnxWord> = module.glwe_secret_alloc(rank_in.into());
                    module.glwe_secret_fill_ternary_prob(&mut sk_in, 0.5, &mut source_xs);

                    let mut sk_out: GLWESecret<BE::OwnedBuf, BE::ZnxWord> = module.glwe_secret_alloc(rank_out.into());
                    module.glwe_secret_fill_ternary_prob(&mut sk_out, 0.5, &mut source_xs);
                    let mut sk_out_prepared: GLWESecretPrepared<BE::OwnedBuf, BE> =
                        module.glwe_secret_prepared_alloc(rank_out.into());
                    module.glwe_secret_prepare(&mut sk_out_prepared, &sk_out);

                    let seed_xa = [1u8; 32];

                    module.gglwe_compressed_encrypt_sk(
                        &mut ksk_compressed,
                        &sk_in.data,
                        &sk_out_prepared,
                        seed_xa,
                        &mut source_xe,
                        &mut poisoned_scratch::<BE>(module.gglwe_compressed_encrypt_sk_tmp_bytes(&gglwe_infos)).borrow(),
                    );

                    let mut ksk: GGLWE<BE::OwnedBuf, BE::ZnxWord> = module.gglwe_alloc_from_infos(&gglwe_infos);
                    module.gglwe_decompress(&mut ksk, &ksk_compressed);

                    let max_noise: f64 = DEFAULT_SIGMA_XE.log2() - (gglwe_infos.k().as_usize() as f64) + 0.5;

                    for row in 0..ksk.dnum().as_usize() {
                        for col in 0..ksk.rank_in().as_usize() {
                            let noise_have = ksk
                                .noise_stats(
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
}

/// Encrypting a GGLWE under a secret that was never sampled is refused before any draw or carve, with the key, both
/// sources and the scratch as they were.
pub fn test_gglwe_encrypt_sk_rejects_unset_secret<BE: crate::test_suite::noise::TestBackend>(
    params: &TestParams,
    module: &Module<BE>,
) where
    BE::OwnedBuf: poulpy_hal::layouts::HostDataMut,
    for<'a> BE::BufRef<'a>: poulpy_hal::layouts::HostDataRef,
    for<'a> BE::BufMut<'a>: poulpy_hal::layouts::HostDataMut,
    Module<BE>: GGLWEEncryptSk<BE> + GLWESecretPreparedFactory<BE> + ScalarZnxAlloc<BE>,
    ScratchOwned<BE>: ScratchOwnedAlloc<BE> + ScratchOwnedBorrow<BE>,
{
    let base2k: usize = params.base2k;
    let infos = GGLWELayout {
        n: module.n().into(),
        base2k: base2k.into(),
        dnum: 2_usize.into(),
        k_digit: base2k.into(),
        k_aux: base2k.into(),
        k_offset: 0_usize.into(),
        rank_in: 1_usize.into(),
        rank_out: 2_usize.into(),
        stride: 1,
    };
    let mut key: GGLWE<BE::OwnedBuf, BE::ZnxWord> = module.gglwe_alloc_from_infos(&infos);
    key.data
        .data_mut()
        .iter_mut()
        .enumerate()
        .for_each(|(i, x)| *x = (i as u8).wrapping_mul(29).wrapping_add(11));
    let pt = module.scalar_znx_alloc(module.n(), 1);
    let sk: GLWESecretPrepared<BE::OwnedBuf, BE> = module.glwe_secret_prepared_alloc(2_usize.into());
    let mut scratch = poisoned_scratch::<BE>(0);
    let arena = BE::to_host_bytes(&scratch.data);
    let (mut source_xe, mut source_xa) = (Source::new([7u8; 32]), Source::new([8u8; 32]));
    assert_refusal(
        "glwe secret distribution is NONE (have you prepared the key?)",
        &mut key,
        |key| key.clone(),
        |key| module.gglwe_encrypt_sk(key, &pt, &sk, &mut source_xe, &mut source_xa, &mut scratch.borrow()),
    );
    assert!(
        BE::to_host_bytes(&scratch.data) == arena,
        "a refused call wrote into its scratch"
    );
    assert_eq!(
        (source_xe.next_u128(), source_xa.next_u128()),
        (Source::new([7u8; 32]).next_u128(), Source::new([8u8; 32]).next_u128()),
        "a refused call drew from its sources"
    );
}
