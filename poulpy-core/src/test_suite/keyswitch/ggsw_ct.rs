use poulpy_hal::{
    api::{ScratchAvailable, ScratchOwnedAlloc, ScratchOwnedBorrow},
    layouts::{DeviceBuf, Module, ScalarZnx, Scratch, ScratchOwned},
    source::Source,
    test_suite::TestParams,
};

use crate::{
    EncryptionLayout, GGLWEToGGSWKeyEncryptSk, GGSWEncryptSk, GGSWKeyswitch, GGSWNoise, GLWESwitchingKeyEncryptSk,
    ScratchTakeCore,
    encryption::DEFAULT_SIGMA_XE,
    layouts::{
        GGLWEToGGSWKey, GGLWEToGGSWKeyLayout, GGLWEToGGSWKeyPrepared, GGLWEToGGSWKeyPreparedFactory, GGSW, GGSWInfos, GGSWLayout,
        GLWEInfos, GLWESecret, GLWESecretPreparedFactory, GLWESwitchingKey, GLWESwitchingKeyLayout,
        GLWESwitchingKeyPreparedFactory,
        prepared::{GLWESecretPrepared, GLWESwitchingKeyPrepared},
    },
    noise::noise_ggsw_keyswitch,
};

#[allow(clippy::too_many_arguments)]
pub fn test_ggsw_keyswitch<BE: crate::test_suite::TestBackend>(params: &TestParams, module: &Module<BE>)
where
    Module<BE>: GGSWEncryptSk<BE>
        + GLWESwitchingKeyEncryptSk<BE>
        + GGLWEToGGSWKeyEncryptSk<BE>
        + GGSWKeyswitch<BE>
        + GLWESecretPreparedFactory<BE>
        + GGLWEToGGSWKeyPreparedFactory<BE>
        + GLWESwitchingKeyPreparedFactory<BE>
        + GGSWNoise<BE>,
    ScratchOwned<BE>: ScratchOwnedAlloc<BE> + ScratchOwnedBorrow<BE>,
    Scratch<BE>: ScratchAvailable + ScratchTakeCore<BE>,
{
    let base2k: usize = params.base2k;
    let in_base2k: usize = base2k - 1;
    let key_base2k: usize = base2k;
    let out_base2k: usize = in_base2k; // MUST BE SAME
    let k_in: usize = 4 * in_base2k + 1;
    let max_dsize: usize = k_in.div_ceil(key_base2k);

    for rank in 1_usize..3 {
        for dsize in 1..max_dsize + 1 {
            let k_ksk: usize = k_in + key_base2k * dsize;
            let k_tsk: usize = k_ksk;
            let k_out: usize = k_ksk; // Better capture noise.

            let n: usize = module.n();
            let dnum_in: usize = k_in / in_base2k;
            let dnum_ksk: usize = k_in.div_ceil(key_base2k * dsize);

            let dsize_in: usize = 1;

            let ggsw_in_infos = EncryptionLayout::new_from_default_sigma(GGSWLayout {
                n: n.into(),
                base2k: in_base2k.into(),
                k: k_in.into(),
                dnum: dnum_in.into(),
                dsize: dsize_in.into(),
                rank: rank.into(),
            })
            .unwrap();

            let ggsw_out_infos: GGSWLayout = GGSWLayout {
                n: n.into(),
                base2k: out_base2k.into(),
                k: k_out.into(),
                dnum: dnum_in.into(),
                dsize: dsize_in.into(),
                rank: rank.into(),
            };

            let tsk_infos = EncryptionLayout::new_from_default_sigma(GGLWEToGGSWKeyLayout {
                n: n.into(),
                base2k: key_base2k.into(),
                k: k_tsk.into(),
                dnum: dnum_ksk.into(),
                dsize: dsize.into(),
                rank: rank.into(),
            })
            .unwrap();

            let ksk_apply_infos = EncryptionLayout::new_from_default_sigma(GLWESwitchingKeyLayout {
                n: n.into(),
                base2k: key_base2k.into(),
                k: k_ksk.into(),
                dnum: dnum_ksk.into(),
                dsize: dsize.into(),
                rank_in: rank.into(),
                rank_out: rank.into(),
            })
            .unwrap();

            let mut ggsw_in: GGSW<Vec<u8>> = GGSW::alloc_from_infos(&ggsw_in_infos);
            let mut ggsw_out: GGSW<Vec<u8>> = GGSW::alloc_from_infos(&ggsw_out_infos);
            let mut tsk: GGLWEToGGSWKey<Vec<u8>> = GGLWEToGGSWKey::alloc_from_infos(&tsk_infos);
            let mut ksk: GLWESwitchingKey<Vec<u8>> = GLWESwitchingKey::alloc_from_infos(&ksk_apply_infos);
            let mut pt_scalar: ScalarZnx<Vec<u8>> = ScalarZnx::alloc(n, 1);

            let mut source_xs: Source = Source::new([0u8; 32]);
            let mut source_xe: Source = Source::new([0u8; 32]);
            let mut source_xa: Source = Source::new([0u8; 32]);

            let mut scratch: ScratchOwned<BE> = ScratchOwned::alloc(
                (module).ggsw_encrypt_sk_tmp_bytes(&ggsw_in_infos)
                    | (module).glwe_switching_key_encrypt_sk_tmp_bytes(&ksk_apply_infos)
                    | (module).gglwe_to_ggsw_key_encrypt_sk_tmp_bytes(&tsk_infos)
                    | module.ggsw_keyswitch_tmp_bytes(&ggsw_out_infos, &ggsw_in_infos, &ksk_apply_infos, &tsk_infos),
            );

            let var_xs: f64 = 0.5;

            let mut sk_in: GLWESecret<Vec<u8>> = GLWESecret::alloc(n.into(), rank.into());
            sk_in.fill_ternary_prob(var_xs, &mut source_xs);

            let mut sk_in_prepared: GLWESecretPrepared<DeviceBuf<BE>, BE> = module.glwe_secret_prepared_alloc(rank.into());
            module.glwe_secret_prepare(&mut sk_in_prepared, &sk_in);

            let mut sk_out: GLWESecret<Vec<u8>> = GLWESecret::alloc(n.into(), rank.into());
            sk_out.fill_ternary_prob(var_xs, &mut source_xs);

            let mut sk_out_prepared: GLWESecretPrepared<DeviceBuf<BE>, BE> = module.glwe_secret_prepared_alloc(rank.into());
            module.glwe_secret_prepare(&mut sk_out_prepared, &sk_out);

            module.glwe_switching_key_encrypt_sk(
                &mut ksk,
                &sk_in,
                &sk_out,
                &ksk_apply_infos,
                &mut source_xe,
                &mut source_xa,
                scratch.borrow(),
            );
            module.gglwe_to_ggsw_key_encrypt_sk(
                &mut tsk,
                &sk_out,
                &tsk_infos,
                &mut source_xe,
                &mut source_xa,
                scratch.borrow(),
            );

            pt_scalar.fill_ternary_hw(0, n, &mut source_xs);

            module.ggsw_encrypt_sk(
                &mut ggsw_in,
                &pt_scalar,
                &sk_in_prepared,
                &ggsw_in_infos,
                &mut source_xe,
                &mut source_xa,
                scratch.borrow(),
            );

            let mut ksk_prepared: GLWESwitchingKeyPrepared<DeviceBuf<BE>, BE> =
                module.glwe_switching_key_prepared_alloc_from_infos(&ksk);
            module.glwe_switching_key_prepare(&mut ksk_prepared, &ksk, scratch.borrow());

            let mut tsk_prepared: GGLWEToGGSWKeyPrepared<DeviceBuf<BE>, BE> =
                module.gglwe_to_ggsw_key_prepared_alloc_from_infos(&tsk);
            module.gglwe_to_ggsw_key_prepare(&mut tsk_prepared, &tsk, scratch.borrow());

            module.ggsw_keyswitch(&mut ggsw_out, &ggsw_in, &ksk_prepared, &tsk_prepared, scratch.borrow());

            let max_noise = |col_j: usize| -> f64 {
                noise_ggsw_keyswitch(
                    n as f64,
                    key_base2k * dsize,
                    col_j,
                    var_xs,
                    0f64,
                    DEFAULT_SIGMA_XE * DEFAULT_SIGMA_XE,
                    0f64,
                    rank as f64,
                    k_in,
                    k_ksk,
                    k_tsk,
                ) + 0.5
            };

            for row in 0..ggsw_out.dnum().as_usize() {
                for col in 0..ggsw_out.rank().as_usize() + 1 {
                    let noise = ggsw_out
                        .noise(module, row, col, &pt_scalar, &sk_out_prepared, scratch.borrow())
                        .std()
                        .log2();
                    let max_noise = max_noise(col);
                    assert!(noise <= max_noise, "noise: {noise} > max_noise: {max_noise}")
                }
            }
        }
    }
}

#[allow(clippy::too_many_arguments)]
pub fn test_ggsw_keyswitch_assign<BE: crate::test_suite::TestBackend>(params: &TestParams, module: &Module<BE>)
where
    Module<BE>: GGSWEncryptSk<BE>
        + GLWESwitchingKeyEncryptSk<BE>
        + GGLWEToGGSWKeyEncryptSk<BE>
        + GGSWKeyswitch<BE>
        + GLWESecretPreparedFactory<BE>
        + GGLWEToGGSWKeyPreparedFactory<BE>
        + GLWESwitchingKeyPreparedFactory<BE>
        + GGSWNoise<BE>,
    ScratchOwned<BE>: ScratchOwnedAlloc<BE> + ScratchOwnedBorrow<BE>,
    Scratch<BE>: ScratchAvailable + ScratchTakeCore<BE>,
{
    let base2k: usize = params.base2k;
    let out_base2k: usize = base2k - 1;
    let key_base2k: usize = base2k;
    let k_out: usize = 4 * out_base2k + 1;
    let max_dsize: usize = k_out.div_ceil(key_base2k);

    for rank in 1_usize..3 {
        for dsize in 1..max_dsize + 1 {
            let k_ksk: usize = k_out + key_base2k * dsize;
            let k_tsk: usize = k_ksk;

            let n: usize = module.n();
            let dnum_in: usize = k_out / out_base2k;
            let dnum_ksk: usize = k_out.div_ceil(key_base2k * dsize);
            let dsize_in: usize = 1;

            let ggsw_out_infos = EncryptionLayout::new_from_default_sigma(GGSWLayout {
                n: n.into(),
                base2k: out_base2k.into(),
                k: k_out.into(),
                dnum: dnum_in.into(),
                dsize: dsize_in.into(),
                rank: rank.into(),
            })
            .unwrap();

            let tsk_infos = EncryptionLayout::new_from_default_sigma(GGLWEToGGSWKeyLayout {
                n: n.into(),
                base2k: key_base2k.into(),
                k: k_tsk.into(),
                dnum: dnum_ksk.into(),
                dsize: dsize.into(),
                rank: rank.into(),
            })
            .unwrap();

            let ksk_apply_infos = EncryptionLayout::new_from_default_sigma(GLWESwitchingKeyLayout {
                n: n.into(),
                base2k: key_base2k.into(),
                k: k_ksk.into(),
                dnum: dnum_ksk.into(),
                dsize: dsize.into(),
                rank_in: rank.into(),
                rank_out: rank.into(),
            })
            .unwrap();

            let mut ggsw_out: GGSW<Vec<u8>> = GGSW::alloc_from_infos(&ggsw_out_infos);
            let mut tsk: GGLWEToGGSWKey<Vec<u8>> = GGLWEToGGSWKey::alloc_from_infos(&tsk_infos);
            let mut ksk: GLWESwitchingKey<Vec<u8>> = GLWESwitchingKey::alloc_from_infos(&ksk_apply_infos);
            let mut pt_scalar: ScalarZnx<Vec<u8>> = ScalarZnx::alloc(n, 1);

            let mut source_xs: Source = Source::new([0u8; 32]);
            let mut source_xe: Source = Source::new([0u8; 32]);
            let mut source_xa: Source = Source::new([0u8; 32]);

            let mut scratch: ScratchOwned<BE> = ScratchOwned::alloc(
                (module).ggsw_encrypt_sk_tmp_bytes(&ggsw_out_infos)
                    | (module).glwe_switching_key_encrypt_sk_tmp_bytes(&ksk_apply_infos)
                    | (module).gglwe_to_ggsw_key_encrypt_sk_tmp_bytes(&tsk_infos)
                    | module.ggsw_keyswitch_tmp_bytes(&ggsw_out_infos, &ggsw_out_infos, &ksk_apply_infos, &tsk_infos),
            );

            let var_xs: f64 = 0.5;

            let mut sk_in: GLWESecret<Vec<u8>> = GLWESecret::alloc(n.into(), rank.into());
            sk_in.fill_ternary_prob(var_xs, &mut source_xs);

            let mut sk_in_prepared: GLWESecretPrepared<DeviceBuf<BE>, BE> = module.glwe_secret_prepared_alloc(rank.into());
            module.glwe_secret_prepare(&mut sk_in_prepared, &sk_in);

            let mut sk_out: GLWESecret<Vec<u8>> = GLWESecret::alloc(n.into(), rank.into());
            sk_out.fill_ternary_prob(var_xs, &mut source_xs);

            let mut sk_out_prepared: GLWESecretPrepared<DeviceBuf<BE>, BE> = module.glwe_secret_prepared_alloc(rank.into());
            module.glwe_secret_prepare(&mut sk_out_prepared, &sk_out);

            module.glwe_switching_key_encrypt_sk(
                &mut ksk,
                &sk_in,
                &sk_out,
                &ksk_apply_infos,
                &mut source_xe,
                &mut source_xa,
                scratch.borrow(),
            );
            module.gglwe_to_ggsw_key_encrypt_sk(
                &mut tsk,
                &sk_out,
                &tsk_infos,
                &mut source_xe,
                &mut source_xa,
                scratch.borrow(),
            );

            pt_scalar.fill_ternary_hw(0, n, &mut source_xs);

            module.ggsw_encrypt_sk(
                &mut ggsw_out,
                &pt_scalar,
                &sk_in_prepared,
                &ggsw_out_infos,
                &mut source_xe,
                &mut source_xa,
                scratch.borrow(),
            );

            let mut ksk_prepared: GLWESwitchingKeyPrepared<DeviceBuf<BE>, BE> =
                module.glwe_switching_key_prepared_alloc_from_infos(&ksk);
            module.glwe_switching_key_prepare(&mut ksk_prepared, &ksk, scratch.borrow());

            let mut tsk_prepared: GGLWEToGGSWKeyPrepared<DeviceBuf<BE>, BE> =
                module.gglwe_to_ggsw_key_prepared_alloc_from_infos(&tsk);
            module.gglwe_to_ggsw_key_prepare(&mut tsk_prepared, &tsk, scratch.borrow());

            module.ggsw_keyswitch_assign(&mut ggsw_out, &ksk_prepared, &tsk_prepared, scratch.borrow());

            let max_noise = |col_j: usize| -> f64 {
                noise_ggsw_keyswitch(
                    n as f64,
                    key_base2k * dsize,
                    col_j,
                    var_xs,
                    0f64,
                    DEFAULT_SIGMA_XE * DEFAULT_SIGMA_XE,
                    0f64,
                    rank as f64,
                    k_out,
                    k_ksk,
                    k_tsk,
                ) + 0.5
            };

            for row in 0..ggsw_out.dnum().as_usize() {
                for col in 0..ggsw_out.rank().as_usize() + 1 {
                    let noise = ggsw_out
                        .noise(module, row, col, &pt_scalar, &sk_out_prepared, scratch.borrow())
                        .std()
                        .log2();
                    let max_noise = max_noise(col);
                    assert!(noise <= max_noise, "noise: {noise} > max_noise: {max_noise}")
                }
            }
        }
    }
}
