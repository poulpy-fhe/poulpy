use poulpy_hal::AlignedBuf;
use poulpy_hal::{
    api::{ScalarZnxAlloc, ScratchOwnedAlloc, ScratchOwnedBorrow, VecZnxFillUniformSource},
    layouts::{DataViewMut, Module, ReaderFrom, Ring, ScratchOwned, WriterTo},
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

pub fn test_gglwe_switching_key_encrypt_sk<BE: crate::test_suite::noise::TestBackend<Rg>, Rg: Ring>(
    params: &TestParams,
    module: &Module<BE>,
) where
    BE::OwnedBuf: poulpy_hal::layouts::HostDataMut,
    for<'a> BE::BufRef<'a>: poulpy_hal::layouts::HostDataRef,
    for<'a> BE::BufMut<'a>: poulpy_hal::layouts::HostDataMut,
    Module<BE>: GGLWEEncryptSk<BE, Rg>
        + GGLWEPreparedFactory<BE, Rg>
        + GGLWEKeyswitch<BE, Rg>
        + GLWEDecrypt<BE, Rg>
        + GLWESecretPreparedFactory<BE, Rg>
        + GLWESwitchingKeyEncryptSk<BE, Rg>
        + VecZnxFillUniformSource<BE, Rg>
        + GGLWENoise<BE, Rg>,
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

                    let mut ksk: GLWESwitchingKey<BE::OwnedBuf, BE::ZnxWord, Rg> =
                        module.glwe_switching_key_alloc_from_infos(&gglwe_infos);

                    let mut source_xs: Source = Source::new([0u8; 32]);
                    let mut source_xe: Source = Source::new([0u8; 32]);
                    let mut source_xa: Source = Source::new([0u8; 32]);

                    let mut scratch: ScratchOwned<BE> = ScratchOwned::alloc(module.gglwe_noise_tmp_bytes(&gglwe_infos));

                    let mut sk_in: GLWESecret<BE::OwnedBuf, BE::ZnxWord, Rg> = module.glwe_secret_alloc(rank_in.into());
                    module.glwe_secret_fill_ternary_prob(&mut sk_in, 0.5, &mut source_xs);

                    let mut sk_out: GLWESecret<BE::OwnedBuf, BE::ZnxWord, Rg> = module.glwe_secret_alloc(rank_out.into());
                    module.glwe_secret_fill_ternary_prob(&mut sk_out, 0.5, &mut source_xs);
                    *sk_out.dist_mut() = Distribution::ENCAPSULATED("roundtrip");
                    let mut sk_out_prepared: GLWESecretPrepared<BE::OwnedBuf, BE, Rg> =
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

        let mut ksk: GLWESwitchingKey<BE::OwnedBuf, BE::ZnxWord, Rg> = module.glwe_switching_key_alloc_from_infos(&gglwe_infos);
        let mut source_xs: Source = Source::new([0u8; 32]);
        let mut source_xe: Source = Source::new([0u8; 32]);
        let mut source_xa: Source = Source::new([0u8; 32]);
        let mut sk_in: GLWESecret<BE::OwnedBuf, BE::ZnxWord, Rg> = module.glwe_secret_alloc(1_u32.into());
        module.glwe_secret_fill_ternary_prob(&mut sk_in, 0.5, &mut source_xs);
        let mut sk_out: GLWESecret<AlignedBuf, i64, Rg> = GLWESecret::alloc(smaller_n.into(), 1_u32.into());
        module.glwe_secret_fill_ternary_prob(&mut sk_out, 0.5, &mut source_xs);

        // The lifted and prepared secrets would be left in the scratch.
        crate::test_suite::assert_wipes_scratch::<BE>(module.glwe_switching_key_encrypt_sk_tmp_bytes(&gglwe_infos), |scratch| {
            module.glwe_switching_key_encrypt_sk(&mut ksk, &sk_in, &sk_out, &mut source_xe, &mut source_xa, scratch)
        });
    }
}

pub fn test_gglwe_switching_key_compressed_encrypt_sk<BE: crate::test_suite::noise::TestBackend<Rg>, Rg: Ring>(
    params: &TestParams,
    module: &Module<BE>,
) where
    BE::OwnedBuf: poulpy_hal::layouts::HostDataMut,
    for<'a> BE::BufRef<'a>: poulpy_hal::layouts::HostDataRef,
    for<'a> BE::BufMut<'a>: poulpy_hal::layouts::HostDataMut,
    Module<BE>: GGLWEEncryptSk<BE, Rg>
        + GGLWEPreparedFactory<BE, Rg>
        + GGLWEKeyswitch<BE, Rg>
        + GLWEDecrypt<BE, Rg>
        + GLWESecretPreparedFactory<BE, Rg>
        + GLWESwitchingKeyEncryptSk<BE, Rg>
        + GLWESwitchingKeyCompressedEncryptSk<BE, Rg>
        + GLWESwitchingKeyDecompress<BE, Rg>
        + GGLWENoise<BE, Rg>
        + VecZnxFillUniformSource<BE, Rg>,
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

                    let mut ksk_compressed: GLWESwitchingKeyCompressed<BE::OwnedBuf, BE::ZnxWord, Rg> =
                        module.glwe_switching_key_compressed_alloc_from_infos(&gglwe_infos);

                    let mut source_xs: Source = Source::new([0u8; 32]);
                    let mut source_xe: Source = Source::new([0u8; 32]);

                    let mut scratch: ScratchOwned<BE> = ScratchOwned::alloc(module.gglwe_noise_tmp_bytes(&gglwe_infos));

                    let mut sk_in: GLWESecret<BE::OwnedBuf, BE::ZnxWord, Rg> = module.glwe_secret_alloc(rank_in.into());
                    module.glwe_secret_fill_ternary_prob(&mut sk_in, 0.5, &mut source_xs);

                    let mut sk_out: GLWESecret<BE::OwnedBuf, BE::ZnxWord, Rg> = module.glwe_secret_alloc(rank_out.into());
                    module.glwe_secret_fill_ternary_prob(&mut sk_out, 0.5, &mut source_xs);
                    let mut sk_out_prepared: GLWESecretPrepared<BE::OwnedBuf, BE, Rg> =
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

                    let mut ksk: GLWESwitchingKey<BE::OwnedBuf, BE::ZnxWord, Rg> =
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

        let mut ksk_compressed: GLWESwitchingKeyCompressed<BE::OwnedBuf, BE::ZnxWord, Rg> =
            module.glwe_switching_key_compressed_alloc_from_infos(&gglwe_infos);
        let mut source_xs: Source = Source::new([0u8; 32]);
        let mut source_xe: Source = Source::new([0u8; 32]);
        let seed_xa = [1u8; 32];
        let mut scratch = poisoned_scratch::<BE>(module.glwe_switching_key_compressed_encrypt_sk_tmp_bytes(&gglwe_infos));

        let mut sk_in: GLWESecret<BE::OwnedBuf, BE::ZnxWord, Rg> = module.glwe_secret_alloc(1_u32.into());
        module.glwe_secret_fill_ternary_prob(&mut sk_in, 0.5, &mut source_xs);
        let mut sk_out: GLWESecret<AlignedBuf, i64, Rg> = GLWESecret::alloc(smaller_n.into(), 1_u32.into());
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

pub fn test_gglwe_compressed_encrypt_sk<BE: crate::test_suite::noise::TestBackend<Rg>, Rg: Ring>(
    params: &TestParams,
    module: &Module<BE>,
) where
    BE::OwnedBuf: poulpy_hal::layouts::HostDataMut,
    for<'a> BE::BufRef<'a>: poulpy_hal::layouts::HostDataRef,
    for<'a> BE::BufMut<'a>: poulpy_hal::layouts::HostDataMut,
    Module<BE>: GGLWEEncryptSk<BE, Rg>
        + GGLWEPreparedFactory<BE, Rg>
        + GGLWEKeyswitch<BE, Rg>
        + GLWEDecrypt<BE, Rg>
        + GLWESecretPreparedFactory<BE, Rg>
        + GLWESwitchingKeyEncryptSk<BE, Rg>
        + GGLWECompressedEncryptSk<BE, Rg>
        + GGLWEDecompress<BE, Rg>
        + GGLWENoise<BE, Rg>
        + VecZnxFillUniformSource<BE, Rg>,
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

                    let mut ksk_compressed: GGLWECompressed<BE::OwnedBuf, BE::ZnxWord, Rg> =
                        module.gglwe_compressed_alloc_from_infos(&gglwe_infos);

                    let mut source_xs: Source = Source::new([0u8; 32]);
                    let mut source_xe: Source = Source::new([0u8; 32]);

                    let mut scratch: ScratchOwned<BE> = ScratchOwned::alloc(module.gglwe_noise_tmp_bytes(&gglwe_infos));

                    let mut sk_in: GLWESecret<BE::OwnedBuf, BE::ZnxWord, Rg> = module.glwe_secret_alloc(rank_in.into());
                    module.glwe_secret_fill_ternary_prob(&mut sk_in, 0.5, &mut source_xs);

                    let mut sk_out: GLWESecret<BE::OwnedBuf, BE::ZnxWord, Rg> = module.glwe_secret_alloc(rank_out.into());
                    module.glwe_secret_fill_ternary_prob(&mut sk_out, 0.5, &mut source_xs);
                    let mut sk_out_prepared: GLWESecretPrepared<BE::OwnedBuf, BE, Rg> =
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

                    let mut ksk: GGLWE<BE::OwnedBuf, BE::ZnxWord, Rg> = module.gglwe_alloc_from_infos(&gglwe_infos);
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
pub fn test_gglwe_encrypt_sk_rejects_unset_secret<BE: crate::test_suite::noise::TestBackend<Rg>, Rg: Ring>(
    params: &TestParams,
    module: &Module<BE>,
) where
    BE::OwnedBuf: poulpy_hal::layouts::HostDataMut,
    for<'a> BE::BufRef<'a>: poulpy_hal::layouts::HostDataRef,
    for<'a> BE::BufMut<'a>: poulpy_hal::layouts::HostDataMut,
    Module<BE>: GGLWEEncryptSk<BE, Rg> + GLWESecretPreparedFactory<BE, Rg> + ScalarZnxAlloc<BE>,
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
    let mut key: GGLWE<BE::OwnedBuf, BE::ZnxWord, Rg> = module.gglwe_alloc_from_infos(&infos);
    key.data
        .data_mut()
        .iter_mut()
        .enumerate()
        .for_each(|(i, x)| *x = (i as u8).wrapping_mul(29).wrapping_add(11));
    let pt = module.scalar_znx_alloc(module.n(), 1);
    let sk: GLWESecretPrepared<BE::OwnedBuf, BE, Rg> = module.glwe_secret_prepared_alloc(2_usize.into());
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

/// The keys of a sparse-secret encapsulation against their definition and their model, with and without a gap and a
/// return key: the secret is ternary of the asked Hamming weight on the subring of the gap and tagged encapsulated,
/// the keys are bit for bit the two switching keys encrypted under it from the same sources, on exactly the bytes of
/// the query, and every refusal fires before any draw or write.
pub fn test_glwe_switching_key_encrypt_sk_encapsulation<BE, Rg: Ring>(params: &TestParams, module: &Module<BE>)
where
    BE: crate::test_suite::noise::TestBackend<Rg> + poulpy_hal::layouts::Backend<ZnxWord = i64>,
    BE::OwnedBuf: poulpy_hal::layouts::HostDataMut,
    for<'a> BE::BufRef<'a>: poulpy_hal::layouts::HostDataRef,
    for<'a> BE::BufMut<'a>: poulpy_hal::layouts::HostDataMut,
    Module<BE>: GLWESwitchingKeyEncryptSk<BE, Rg>,
    ScratchOwned<BE>: ScratchOwnedAlloc<BE> + ScratchOwnedBorrow<BE>,
{
    use crate::GetDistribution;
    use poulpy_hal::layouts::DataView;

    let (n, base2k) = (module.n(), params.base2k);
    let hw: usize = 3;
    for rank in 1_usize..3 {
        let layout = GGLWELayout {
            n: n.into(),
            base2k: base2k.into(),
            dnum: 2_usize.into(),
            k_aux: (base2k + module.log_n()).into(),
            k_digit: crate::layouts::k_digit_of(base2k, 1_usize),
            k_offset: 0_usize.into(),
            rank_in: rank.into(),
            rank_out: rank.into(),
            stride: 1,
        };
        let mut sk_dense: GLWESecret<BE::OwnedBuf, BE::ZnxWord, Rg> = module.glwe_secret_alloc(rank.into());
        module.glwe_secret_fill_ternary_prob(&mut sk_dense, 0.5, &mut Source::new([1u8; 32]));
        let coeffs = |sk: &GLWESecret<BE::OwnedBuf, BE::ZnxWord, Rg>| -> Vec<i64> {
            bytemuck::cast_slice(&BE::to_host_bytes(sk.data.data())).to_vec()
        };

        for log_gap in [0_usize, 2] {
            for with_return_key in [true, false] {
                let seeds = || (Source::new([2u8; 32]), Source::new([3u8; 32]), Source::new([4u8; 32]));
                let bytes: usize = module.glwe_switching_key_encrypt_sk_encapsulation_tmp_bytes(
                    &layout,
                    with_return_key.then_some(&layout),
                    log_gap,
                );

                let (mut xs, mut xe, mut xa) = seeds();
                let mut dense_to_sparse = module.glwe_switching_key_alloc_from_infos(&layout);
                let mut sparse_to_dense = with_return_key.then(|| module.glwe_switching_key_alloc_from_infos(&layout));
                let mut sk_sparse: GLWESecret<BE::OwnedBuf, BE::ZnxWord, Rg> = module.glwe_secret_alloc(rank.into());
                module.glwe_switching_key_encrypt_sk_encapsulation(
                    &mut dense_to_sparse,
                    sparse_to_dense.as_mut(),
                    &mut sk_sparse,
                    &sk_dense,
                    log_gap,
                    hw,
                    &mut xs,
                    &mut xe,
                    &mut xa,
                    &mut poisoned_scratch::<BE>(bytes).borrow(),
                );

                // The model of the secret.
                assert_eq!(*sk_sparse.dist(), Distribution::ENCAPSULATED("sparse-encapsulation"));
                let values = coeffs(&sk_sparse);
                for column in values.chunks(n) {
                    assert!(
                        column.iter().all(|c| (-1..=1).contains(c)),
                        "the sparse secret is not ternary"
                    );
                    assert_eq!(
                        column.iter().filter(|&&c| c != 0).count(),
                        hw,
                        "the sparse secret has another weight"
                    );
                    assert!(
                        column.iter().enumerate().all(|(i, &c)| c == 0 || i % (1 << log_gap) == 0),
                        "the sparse secret leaves the subring of the gap"
                    );
                }

                // The definition: the same secret, then the two switching keys, from the same sources.
                let (mut ys, mut ye, mut ya) = seeds();
                let mut expected_sk: GLWESecret<BE::OwnedBuf, BE::ZnxWord, Rg> = module.glwe_secret_alloc(rank.into());
                if log_gap == 0 {
                    module.glwe_secret_fill_ternary_hw(&mut expected_sk, hw, &mut ys);
                } else {
                    let mut subring: GLWESecret<BE::OwnedBuf, BE::ZnxWord, Rg> =
                        module.glwe_secret_alloc_from_infos(&crate::layouts::GLWESecretLayout {
                            n: (n >> log_gap).into(),
                            rank: rank.into(),
                        });
                    module.glwe_secret_fill_ternary_hw(&mut subring, hw, &mut ys);
                    let embedded: Vec<i64> = coeffs(&subring)
                        .chunks(n >> log_gap)
                        .flat_map(|column| {
                            column
                                .iter()
                                .flat_map(|&c| std::iter::once(c).chain(std::iter::repeat_n(0, (1 << log_gap) - 1)))
                        })
                        .collect();
                    assert_eq!(values, embedded, "the sparse secret is not the embedded subring secret");
                    let mut view = crate::layouts::GLWESecretToBackendMut::<BE, Rg>::to_backend_mut(&mut expected_sk);
                    BE::copy_host_to_view(view.data.data_mut(), bytemuck::cast_slice(&embedded));
                }
                *expected_sk.dist_mut() = Distribution::ENCAPSULATED("sparse-encapsulation");
                assert_eq!(values, coeffs(&expected_sk), "the sparse secret differs from its definition");
                let key_bytes: usize = module.glwe_switching_key_encrypt_sk_tmp_bytes(&layout);
                let mut expected = module.glwe_switching_key_alloc_from_infos(&layout);
                module.glwe_switching_key_encrypt_sk(
                    &mut expected,
                    &sk_dense,
                    &expected_sk,
                    &mut ye,
                    &mut ya,
                    &mut poisoned_scratch::<BE>(key_bytes).borrow(),
                );
                assert!(dense_to_sparse == expected, "dense_to_sparse differs from its definition");
                if let Some(sparse_to_dense) = &sparse_to_dense {
                    module.glwe_switching_key_encrypt_sk(
                        &mut expected,
                        &expected_sk,
                        &sk_dense,
                        &mut ye,
                        &mut ya,
                        &mut poisoned_scratch::<BE>(key_bytes).borrow(),
                    );
                    assert!(*sparse_to_dense == expected, "sparse_to_dense differs from its definition");
                }
                assert_eq!(
                    (xs.next_u128(), xe.next_u128(), xa.next_u128()),
                    (ys.next_u128(), ye.next_u128(), ya.next_u128()),
                    "the sources end in another state than their definition's"
                );

                // One byte below the query is refused before any draw or write.
                let (mut xs, mut xe, mut xa) = seeds();
                assert_refusal(
                    "insufficient scratch for GLWE encapsulation key encryption",
                    &mut dense_to_sparse,
                    |key| key.clone(),
                    |key| {
                        module.glwe_switching_key_encrypt_sk_encapsulation(
                            key,
                            sparse_to_dense.as_mut(),
                            &mut sk_sparse,
                            &sk_dense,
                            log_gap,
                            hw,
                            &mut xs,
                            &mut xe,
                            &mut xa,
                            &mut poisoned_scratch::<BE>(bytes - 1).borrow(),
                        )
                    },
                );
                assert_eq!(values, coeffs(&sk_sparse), "a refused call wrote the sparse secret");
                let (mut ys, mut ye, mut ya) = seeds();
                assert_eq!(
                    (xs.next_u128(), xe.next_u128(), xa.next_u128()),
                    (ys.next_u128(), ye.next_u128(), ya.next_u128()),
                    "a refused call drew from its sources"
                );
            }
        }

        // A sparse secret of another rank, and a gap past the ring.
        let mut dense_to_sparse = module.glwe_switching_key_alloc_from_infos(&layout);
        let mut scratch = poisoned_scratch::<BE>(module.glwe_switching_key_encrypt_sk_encapsulation_tmp_bytes(
            &layout,
            None::<&GGLWELayout>,
            0,
        ));
        let mut other_rank: GLWESecret<BE::OwnedBuf, BE::ZnxWord, Rg> = module.glwe_secret_alloc((rank + 1).into());
        let mut same_rank: GLWESecret<BE::OwnedBuf, BE::ZnxWord, Rg> = module.glwe_secret_alloc(rank.into());
        for (message, sk_sparse, log_gap) in [
            (
                "the sparse secret's degree or rank differs from the dense secret's",
                &mut other_rank,
                0,
            ),
            (
                "the subring of the gap has no coefficient",
                &mut same_rank,
                module.log_n() + 1,
            ),
        ] {
            assert_eq!(sk_sparse.n(), sk_dense.n());
            assert_refusal(
                message,
                &mut dense_to_sparse,
                |key| key.clone(),
                |key| {
                    module.glwe_switching_key_encrypt_sk_encapsulation(
                        key,
                        None::<&mut GLWESwitchingKey<BE::OwnedBuf, BE::ZnxWord, Rg>>,
                        sk_sparse,
                        &sk_dense,
                        log_gap,
                        hw,
                        &mut Source::new([2u8; 32]),
                        &mut Source::new([3u8; 32]),
                        &mut Source::new([4u8; 32]),
                        &mut scratch.borrow(),
                    )
                },
            );
        }
    }
}
