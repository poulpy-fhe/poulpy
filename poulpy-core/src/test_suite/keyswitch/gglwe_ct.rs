use poulpy_hal::{
    api::{ScratchAvailable, ScratchOwnedAlloc, ScratchOwnedBorrow},
    layouts::{DeviceBuf, Module, Scratch, ScratchOwned},
    source::Source,
    test_suite::TestParams,
};

use crate::{
    EncryptionLayout, GGLWEKeyswitch, GGLWENoise, GLWESwitchingKeyEncryptSk, ScratchTakeCore,
    encryption::DEFAULT_SIGMA_XE,
    layouts::{
        GGLWEInfos, GLWESecret, GLWESecretPreparedFactory, GLWESwitchingKey, GLWESwitchingKeyLayout,
        GLWESwitchingKeyPreparedFactory, LWEInfos,
        prepared::{GLWESecretPrepared, GLWESwitchingKeyPrepared},
    },
    noise::log2_std_noise_gglwe_product,
    var_noise_gglwe_product_v2,
};

pub fn test_gglwe_switching_key_keyswitch<BE: crate::test_suite::TestBackend>(params: &TestParams, module: &Module<BE>)
where
    Module<BE>: GLWESwitchingKeyEncryptSk<BE>
        + GGLWEKeyswitch<BE>
        + GLWESwitchingKeyPreparedFactory<BE>
        + GLWESecretPreparedFactory<BE>
        + GGLWENoise<BE>,
    ScratchOwned<BE>: ScratchOwnedAlloc<BE> + ScratchOwnedBorrow<BE>,
    Scratch<BE>: ScratchAvailable + ScratchTakeCore<BE>,
{
    let base2k: usize = params.base2k;
    let in_base2k: usize = base2k - 1;
    let key_base2k: usize = base2k;
    let out_base2k: usize = in_base2k; // MUST BE SAME
    let k_in: usize = 4 * in_base2k + 1;
    let max_dsize: usize = k_in.div_ceil(key_base2k);

    for rank_in_s0s1 in 1_usize..2 {
        for rank_out_s0s1 in 1_usize..3 {
            for rank_out_s1s2 in 1_usize..3 {
                for dsize in 1_usize..max_dsize + 1 {
                    let k_ksk: usize = k_in + key_base2k * dsize;
                    let k_out: usize = k_ksk; // Better capture noise.

                    let n: usize = module.n();
                    let dsize_in: usize = 1;
                    let dnum_in: usize = k_in / in_base2k;
                    let dnum_ksk: usize = k_in.div_ceil(key_base2k * dsize);

                    let gglwe_s0s1_infos = EncryptionLayout::new_from_default_sigma(GLWESwitchingKeyLayout {
                        n: n.into(),
                        base2k: in_base2k.into(),
                        k: k_in.into(),
                        dnum: dnum_in.into(),
                        dsize: dsize_in.into(),
                        rank_in: rank_in_s0s1.into(),
                        rank_out: rank_out_s0s1.into(),
                    })
                    .unwrap();

                    let gglwe_s1s2_infos = EncryptionLayout::new_from_default_sigma(GLWESwitchingKeyLayout {
                        n: n.into(),
                        base2k: key_base2k.into(),
                        k: k_ksk.into(),
                        dnum: dnum_ksk.into(),
                        dsize: dsize.into(),
                        rank_in: rank_out_s0s1.into(),
                        rank_out: rank_out_s1s2.into(),
                    })
                    .unwrap();

                    let gglwe_s0s2_infos: GLWESwitchingKeyLayout = GLWESwitchingKeyLayout {
                        n: n.into(),
                        base2k: out_base2k.into(),
                        k: k_out.into(),
                        dnum: dnum_in.into(),
                        dsize: dsize_in.into(),
                        rank_in: rank_in_s0s1.into(),
                        rank_out: rank_out_s1s2.into(),
                    };

                    let mut gglwe_s0s1: GLWESwitchingKey<Vec<u8>> = GLWESwitchingKey::alloc_from_infos(&gglwe_s0s1_infos);
                    let mut gglwe_s1s2: GLWESwitchingKey<Vec<u8>> = GLWESwitchingKey::alloc_from_infos(&gglwe_s1s2_infos);
                    let mut gglwe_s0s2: GLWESwitchingKey<Vec<u8>> = GLWESwitchingKey::alloc_from_infos(&gglwe_s0s2_infos);

                    let mut source_xs: Source = Source::new([0u8; 32]);
                    let mut source_xe: Source = Source::new([0u8; 32]);
                    let mut source_xa: Source = Source::new([0u8; 32]);

                    let mut scratch_enc: ScratchOwned<BE> = ScratchOwned::alloc(
                        (module).glwe_switching_key_encrypt_sk_tmp_bytes(&gglwe_s0s1_infos)
                            | (module).glwe_switching_key_encrypt_sk_tmp_bytes(&gglwe_s1s2_infos)
                            | (module).glwe_switching_key_encrypt_sk_tmp_bytes(&gglwe_s0s2_infos),
                    );
                    let mut scratch_apply: ScratchOwned<BE> = ScratchOwned::alloc(
                        module.gglwe_keyswitch_tmp_bytes(&gglwe_s0s2_infos, &gglwe_s0s1_infos, &gglwe_s1s2_infos)
                            | module.gglwe_noise_tmp_bytes(&gglwe_s0s2_infos),
                    );

                    let mut sk0: GLWESecret<Vec<u8>> = GLWESecret::alloc(n.into(), rank_in_s0s1.into());
                    sk0.fill_ternary_prob(0.5, &mut source_xs);

                    let mut sk1: GLWESecret<Vec<u8>> = GLWESecret::alloc(n.into(), rank_out_s0s1.into());
                    sk1.fill_ternary_prob(0.5, &mut source_xs);

                    let mut sk2: GLWESecret<Vec<u8>> = GLWESecret::alloc(n.into(), rank_out_s1s2.into());
                    sk2.fill_ternary_prob(0.5, &mut source_xs);

                    let mut sk2_prepared: GLWESecretPrepared<DeviceBuf<BE>, BE> =
                        module.glwe_secret_prepared_alloc(rank_out_s1s2.into());
                    module.glwe_secret_prepare(&mut sk2_prepared, &sk2);

                    // gglwe_{s1}(s0) = s0 -> s1
                    module.glwe_switching_key_encrypt_sk(
                        &mut gglwe_s0s1,
                        &sk0,
                        &sk1,
                        &gglwe_s0s1_infos,
                        &mut source_xe,
                        &mut source_xa,
                        scratch_enc.borrow(),
                    );

                    // gglwe_{s2}(s1) -> s1 -> s2
                    module.glwe_switching_key_encrypt_sk(
                        &mut gglwe_s1s2,
                        &sk1,
                        &sk2,
                        &gglwe_s1s2_infos,
                        &mut source_xe,
                        &mut source_xa,
                        scratch_enc.borrow(),
                    );

                    let mut gglwe_s1s2_prepared: GLWESwitchingKeyPrepared<DeviceBuf<BE>, BE> =
                        module.glwe_switching_key_prepared_alloc_from_infos(&gglwe_s1s2);
                    module.glwe_switching_key_prepare(&mut gglwe_s1s2_prepared, &gglwe_s1s2, scratch_apply.borrow());

                    // gglwe_{s1}(s0) (x) gglwe_{s2}(s1) = gglwe_{s2}(s0)
                    module.gglwe_keyswitch(&mut gglwe_s0s2, &gglwe_s0s1, &gglwe_s1s2_prepared, scratch_apply.borrow());

                    let max_noise: f64 = var_noise_gglwe_product_v2(
                        module.n() as f64,
                        k_ksk,
                        dnum_ksk,
                        dsize,
                        key_base2k,
                        0.5,
                        0.5,
                        0f64,
                        DEFAULT_SIGMA_XE * DEFAULT_SIGMA_XE,
                        0f64,
                        rank_out_s0s1 as f64,
                    )
                    .sqrt()
                    .log2()
                        + 0.5;

                    for row in 0..gglwe_s0s2.dnum().as_usize() {
                        for col in 0..gglwe_s0s2.rank_in().as_usize() {
                            let noise_have: f64 = gglwe_s0s2
                                .key
                                .noise(module, row, col, &sk0.data, &sk2_prepared, scratch_apply.borrow())
                                .std()
                                .log2();
                            assert!(noise_have <= max_noise, "noise_have: {noise_have} > max_noise: {max_noise}")
                        }
                    }
                }
            }
        }
    }
}

#[allow(clippy::too_many_arguments)]
pub fn test_gglwe_switching_key_keyswitch_assign<BE: crate::test_suite::TestBackend>(params: &TestParams, module: &Module<BE>)
where
    Module<BE>: GLWESwitchingKeyEncryptSk<BE>
        + GGLWEKeyswitch<BE>
        + GLWESecretPreparedFactory<BE>
        + GGLWENoise<BE>
        + GLWESwitchingKeyPreparedFactory<BE>,
    ScratchOwned<BE>: ScratchOwnedAlloc<BE> + ScratchOwnedBorrow<BE>,
    Scratch<BE>: ScratchAvailable + ScratchTakeCore<BE>,
{
    let base2k: usize = params.base2k;
    let out_base2k: usize = base2k - 1;
    let key_base2k: usize = base2k;
    let k_out: usize = 4 * out_base2k + 1;
    let max_dsize: usize = k_out.div_ceil(key_base2k);

    for rank_in in 1_usize..3 {
        for rank_out in 1_usize..3 {
            for dsize in 1_usize..max_dsize + 1 {
                let k_ksk: usize = k_out + key_base2k * dsize;

                let n: usize = module.n();
                let dsize_in: usize = 1;

                let dnum_in: usize = k_out / out_base2k;
                let dnum_ksk: usize = k_out.div_ceil(key_base2k * dsize);

                let gglwe_s0s1_infos = EncryptionLayout::new_from_default_sigma(GLWESwitchingKeyLayout {
                    n: n.into(),
                    base2k: out_base2k.into(),
                    k: k_out.into(),
                    dnum: dnum_in.into(),
                    dsize: dsize_in.into(),
                    rank_in: rank_in.into(),
                    rank_out: rank_out.into(),
                })
                .unwrap();

                let gglwe_s1s2_infos = EncryptionLayout::new_from_default_sigma(GLWESwitchingKeyLayout {
                    n: n.into(),
                    base2k: key_base2k.into(),
                    k: k_ksk.into(),
                    dnum: dnum_ksk.into(),
                    dsize: dsize.into(),
                    rank_in: rank_out.into(),
                    rank_out: rank_out.into(),
                })
                .unwrap();

                let mut gglwe_s0s1: GLWESwitchingKey<Vec<u8>> = GLWESwitchingKey::alloc_from_infos(&gglwe_s0s1_infos);
                let mut gglwe_s1s2: GLWESwitchingKey<Vec<u8>> = GLWESwitchingKey::alloc_from_infos(&gglwe_s1s2_infos);

                let mut source_xs: Source = Source::new([0u8; 32]);
                let mut source_xe: Source = Source::new([0u8; 32]);
                let mut source_xa: Source = Source::new([0u8; 32]);

                let mut scratch_enc: ScratchOwned<BE> = ScratchOwned::alloc(
                    (module).glwe_switching_key_encrypt_sk_tmp_bytes(&gglwe_s0s1_infos)
                        | (module).glwe_switching_key_encrypt_sk_tmp_bytes(&gglwe_s1s2_infos),
                );
                let mut scratch_apply: ScratchOwned<BE> = ScratchOwned::alloc(
                    module.gglwe_keyswitch_tmp_bytes(&gglwe_s0s1_infos, &gglwe_s0s1_infos, &gglwe_s1s2_infos)
                        | module.gglwe_noise_tmp_bytes(&gglwe_s0s1_infos),
                );

                println!("k_out: {k_out}, k_ksk: {k_ksk}");

                println!(
                    "{} {} {}",
                    gglwe_s0s1_infos.base2k(),
                    gglwe_s0s1_infos.max_k(),
                    gglwe_s0s1_infos.size()
                );
                println!(
                    "{} {} {}",
                    gglwe_s1s2_infos.base2k(),
                    gglwe_s1s2_infos.max_k(),
                    gglwe_s1s2_infos.size()
                );
                println!(
                    "{}",
                    module.gglwe_keyswitch_tmp_bytes(&gglwe_s0s1_infos, &gglwe_s0s1_infos, &gglwe_s1s2_infos)
                );

                let var_xs: f64 = 0.5;

                let mut sk0: GLWESecret<Vec<u8>> = GLWESecret::alloc(n.into(), rank_in.into());
                sk0.fill_ternary_prob(var_xs, &mut source_xs);

                let mut sk1: GLWESecret<Vec<u8>> = GLWESecret::alloc(n.into(), rank_out.into());
                sk1.fill_ternary_prob(var_xs, &mut source_xs);

                let mut sk2: GLWESecret<Vec<u8>> = GLWESecret::alloc(n.into(), rank_out.into());
                sk2.fill_ternary_prob(var_xs, &mut source_xs);

                let mut sk2_prepared: GLWESecretPrepared<DeviceBuf<BE>, BE> = module.glwe_secret_prepared_alloc(rank_out.into());
                module.glwe_secret_prepare(&mut sk2_prepared, &sk2);

                // gglwe_{s1}(s0) = s0 -> s1
                module.glwe_switching_key_encrypt_sk(
                    &mut gglwe_s0s1,
                    &sk0,
                    &sk1,
                    &gglwe_s0s1_infos,
                    &mut source_xe,
                    &mut source_xa,
                    scratch_enc.borrow(),
                );

                // gglwe_{s2}(s1) -> s1 -> s2
                module.glwe_switching_key_encrypt_sk(
                    &mut gglwe_s1s2,
                    &sk1,
                    &sk2,
                    &gglwe_s1s2_infos,
                    &mut source_xe,
                    &mut source_xa,
                    scratch_enc.borrow(),
                );

                let mut gglwe_s1s2_prepared: GLWESwitchingKeyPrepared<DeviceBuf<BE>, BE> =
                    module.glwe_switching_key_prepared_alloc_from_infos(&gglwe_s1s2);
                module.glwe_switching_key_prepare(&mut gglwe_s1s2_prepared, &gglwe_s1s2, scratch_apply.borrow());

                // gglwe_{s1}(s0) (x) gglwe_{s2}(s1) = gglwe_{s2}(s0)
                println!("{} {} {}", gglwe_s0s1.base2k(), gglwe_s0s1.max_k(), gglwe_s0s1.size());
                println!(
                    "{} {} {}",
                    gglwe_s1s2_prepared.base2k(),
                    gglwe_s1s2_prepared.max_k(),
                    gglwe_s1s2_prepared.size()
                );
                module.gglwe_keyswitch_assign(&mut gglwe_s0s1, &gglwe_s1s2_prepared, scratch_apply.borrow());

                let gglwe_s0s2: GLWESwitchingKey<Vec<u8>> = gglwe_s0s1;

                let max_noise: f64 = log2_std_noise_gglwe_product(
                    n as f64,
                    key_base2k * dsize,
                    var_xs,
                    var_xs,
                    0f64,
                    DEFAULT_SIGMA_XE * DEFAULT_SIGMA_XE,
                    0f64,
                    rank_out as f64,
                    k_out,
                    k_ksk,
                ) + 0.5;

                for row in 0..gglwe_s0s2.dnum().as_usize() {
                    for col in 0..gglwe_s0s2.rank_in().as_usize() {
                        let noise_have = gglwe_s0s2
                            .key
                            .noise(module, row, col, &sk0.data, &sk2_prepared, scratch_apply.borrow())
                            .std()
                            .log2();
                        assert!(noise_have <= max_noise, "noise_have: {noise_have} > max_noise: {max_noise}")
                    }
                }
            }
        }
    }
}
