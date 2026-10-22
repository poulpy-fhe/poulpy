use poulpy_hal::{
    api::{ScratchOwnedAlloc, ScratchOwnedBorrow},
    layouts::{Module, ScratchOwned},
    source::Source,
    test_suite::TestParams,
};

use crate::layouts::GLWESecretSampling;
use crate::layouts::prepared::GGLWEPreparedToBackendRef;
use crate::{
    GGLWEKeyswitch, GGLWENoise, GLWENormalize, GLWESwitchingKeyEncryptSk,
    encryption::DEFAULT_SIGMA_XE,
    layouts::{
        Dnum, GGLWEInfos, GGLWEPreparedFactory, GGLWEViewWithDigits, GLWESecret, GLWESecretPreparedFactory, GLWESwitchingKey,
        GLWESwitchingKeyLayout, GLWESwitchingKeyPreparedFactory, LWEInfos, ModuleCoreAlloc,
        prepared::{GLWESecretPrepared, GLWESwitchingKeyPrepared},
    },
    noise::GGLWENoiseModel,
};

pub fn test_gglwe_switching_key_keyswitch<BE: crate::test_suite::noise::TestBackend>(params: &TestParams, module: &Module<BE>)
where
    BE::OwnedBuf: poulpy_hal::layouts::HostDataMut,
    for<'a> BE::BufRef<'a>: poulpy_hal::layouts::HostDataRef,
    for<'a> BE::BufMut<'a>: poulpy_hal::layouts::HostDataMut,
    Module<BE>: GLWESwitchingKeyEncryptSk<BE>
        + GGLWEKeyswitch<BE>
        + GLWESwitchingKeyPreparedFactory<BE>
        + GLWESecretPreparedFactory<BE>
        + GGLWENoise<BE>,
    ScratchOwned<BE>: ScratchOwnedAlloc<BE> + ScratchOwnedBorrow<BE>,
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
                    let n: usize = module.n();
                    let dsize_in: usize = 1;
                    let dnum_in: usize = k_in / in_base2k;

                    let gglwe_s0s1_infos = GLWESwitchingKeyLayout {
                        n: n.into(),
                        base2k: in_base2k.into(),
                        dnum: dnum_in.into(),
                        k_aux: (dsize_in * in_base2k + module.log_n()).into(),
                        k_digit: crate::layouts::k_digit_of(in_base2k, dsize_in),
                        k_offset: crate::layouts::TorusPrecision(0),
                        rank_in: rank_in_s0s1.into(),
                        rank_out: rank_out_s0s1.into(),
                    };

                    // Decomposes the rows it switches whole, their guard included.
                    let dnum_ksk: usize = gglwe_s0s1_infos.k().as_usize().div_ceil(key_base2k * dsize);

                    let gglwe_s1s2_infos = GLWESwitchingKeyLayout {
                        n: n.into(),
                        base2k: key_base2k.into(),
                        dnum: dnum_ksk.into(),
                        k_aux: (dsize * key_base2k + module.log_n()).into(),
                        k_digit: crate::layouts::k_digit_of(key_base2k, dsize),
                        k_offset: crate::layouts::TorusPrecision(0),
                        rank_in: rank_out_s0s1.into(),
                        rank_out: rank_out_s1s2.into(),
                    };

                    let gglwe_s0s2_infos: GLWESwitchingKeyLayout = GLWESwitchingKeyLayout {
                        n: n.into(),
                        base2k: out_base2k.into(),
                        dnum: dnum_in.into(),
                        k_aux: (dsize_in * out_base2k + module.log_n()).into(),
                        k_digit: crate::layouts::k_digit_of(out_base2k, dsize_in),
                        k_offset: crate::layouts::TorusPrecision(0),
                        rank_in: rank_in_s0s1.into(),
                        rank_out: rank_out_s1s2.into(),
                    };

                    let mut gglwe_s0s1: GLWESwitchingKey<BE::OwnedBuf, BE::ZnxWord> =
                        module.glwe_switching_key_alloc_from_infos(&gglwe_s0s1_infos);
                    let mut gglwe_s1s2: GLWESwitchingKey<BE::OwnedBuf, BE::ZnxWord> =
                        module.glwe_switching_key_alloc_from_infos(&gglwe_s1s2_infos);
                    let mut gglwe_s0s2: GLWESwitchingKey<BE::OwnedBuf, BE::ZnxWord> =
                        module.glwe_switching_key_alloc_from_infos(&gglwe_s0s2_infos);

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

                    let mut sk0: GLWESecret<BE::OwnedBuf, BE::ZnxWord> = module.glwe_secret_alloc(rank_in_s0s1.into());
                    module.glwe_secret_fill_ternary_prob(&mut sk0, 0.5, &mut source_xs);

                    let mut sk1: GLWESecret<BE::OwnedBuf, BE::ZnxWord> = module.glwe_secret_alloc(rank_out_s0s1.into());
                    module.glwe_secret_fill_ternary_prob(&mut sk1, 0.5, &mut source_xs);

                    let mut sk2: GLWESecret<BE::OwnedBuf, BE::ZnxWord> = module.glwe_secret_alloc(rank_out_s1s2.into());
                    module.glwe_secret_fill_ternary_prob(&mut sk2, 0.5, &mut source_xs);

                    let mut sk2_prepared: GLWESecretPrepared<BE::OwnedBuf, BE> =
                        module.glwe_secret_prepared_alloc(rank_out_s1s2.into());
                    module.glwe_secret_prepare(&mut sk2_prepared, &sk2);

                    // gglwe_{s1}(s0) = s0 -> s1
                    module.glwe_switching_key_encrypt_sk(
                        &mut gglwe_s0s1,
                        &sk0,
                        &sk1,
                        &mut source_xe,
                        &mut source_xa,
                        &mut scratch_enc.arena(),
                    );

                    // gglwe_{s2}(s1) -> s1 -> s2
                    module.glwe_switching_key_encrypt_sk(
                        &mut gglwe_s1s2,
                        &sk1,
                        &sk2,
                        &mut source_xe,
                        &mut source_xa,
                        &mut scratch_enc.arena(),
                    );

                    let mut gglwe_s1s2_prepared: GLWESwitchingKeyPrepared<BE::OwnedBuf, BE> =
                        module.glwe_switching_key_prepared_alloc_from_infos(&gglwe_s1s2);
                    module.glwe_switching_key_prepare(&mut gglwe_s1s2_prepared, &gglwe_s1s2, &mut scratch_apply.borrow());

                    // gglwe_{s1}(s0) (x) gglwe_{s2}(s1) = gglwe_{s2}(s0)
                    module.gglwe_keyswitch(
                        &mut gglwe_s0s2,
                        &gglwe_s0s1,
                        &gglwe_s1s2_prepared.to_backend_ref(),
                        &mut scratch_apply.borrow(),
                    );

                    let max_noise: f64 = gglwe_s1s2_infos.log2_std_noise_keyswitch(
                        &gglwe_s0s2_infos,
                        &gglwe_s0s1_infos,
                        0.5,
                        0.5,
                        DEFAULT_SIGMA_XE * DEFAULT_SIGMA_XE,
                        DEFAULT_SIGMA_XE * DEFAULT_SIGMA_XE,
                        0f64,
                    ) + 0.5;

                    for row in 0..gglwe_s0s2.dnum().as_usize() {
                        for col in 0..gglwe_s0s2.rank_in().as_usize() {
                            let noise_have: f64 = gglwe_s0s2
                                .key
                                .noise_stats(
                                    module,
                                    row,
                                    col,
                                    &sk0.data.to_ref(),
                                    &sk2_prepared,
                                    &mut scratch_apply.borrow(),
                                )
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

/// A switching key switches a key of its own layout through a view of the
/// latter's first digits, into a key of that many digits: the rows decrypt
/// within the noise model of the view's width, and match bit for bit those of
/// a switch of the narrowed copy the view reads.
pub fn test_gglwe_switching_key_keyswitch_digits<BE: crate::test_suite::noise::TestBackend>(
    params: &TestParams,
    module: &Module<BE>,
) where
    BE::OwnedBuf: poulpy_hal::layouts::HostDataMut,
    for<'a> BE::BufRef<'a>: poulpy_hal::layouts::HostDataRef,
    for<'a> BE::BufMut<'a>: poulpy_hal::layouts::HostDataMut,
    Module<BE>: GLWESwitchingKeyEncryptSk<BE>
        + GGLWEKeyswitch<BE>
        + GLWESwitchingKeyPreparedFactory<BE>
        + GLWESecretPreparedFactory<BE>
        + GGLWEPreparedFactory<BE>
        + GLWENormalize<BE>
        + GGLWENoise<BE>,
    ScratchOwned<BE>: ScratchOwnedAlloc<BE> + ScratchOwnedBorrow<BE>,
{
    let base2k: usize = params.base2k;
    let n: usize = module.n();
    let dnum: usize = 4;
    for rank in 1_usize..3 {
        let layout = GLWESwitchingKeyLayout {
            n: n.into(),
            base2k: base2k.into(),
            dnum: dnum.into(),
            k_aux: (base2k + module.log_n()).into(),
            k_digit: crate::layouts::k_digit_of(base2k, 1_usize),
            k_offset: crate::layouts::TorusPrecision(0),
            rank_in: rank.into(),
            rank_out: rank.into(),
        };
        let at = |d: usize| GLWESwitchingKeyLayout {
            dnum: d.into(),
            ..layout
        };
        let max_input_k = layout.gglwe_layout().max_input_k();
        let digits: usize = (1..dnum).rev().find(|&d| at(d).k() <= max_input_k).unwrap();
        let view_layout = at(digits);

        let mut scratch: ScratchOwned<BE> = ScratchOwned::alloc(
            module
                .glwe_switching_key_encrypt_sk_tmp_bytes(&layout)
                .max(module.gglwe_keyswitch_tmp_bytes(&view_layout, &view_layout, &layout))
                .max(module.glwe_normalize_tmp_bytes())
                .max(module.gglwe_noise_tmp_bytes(&view_layout)),
        );
        let mut source_xs: Source = Source::new([1u8; 32]);
        let mut source_xe: Source = Source::new([2u8; 32]);
        let mut source_xa: Source = Source::new([3u8; 32]);
        let mut sks: Vec<GLWESecret<BE::OwnedBuf, BE::ZnxWord>> = (0..3).map(|_| module.glwe_secret_alloc(rank.into())).collect();
        for sk in sks.iter_mut() {
            module.glwe_secret_fill_ternary_prob(sk, 0.5, &mut source_xs);
        }
        let mut sk2_prepared: GLWESecretPrepared<BE::OwnedBuf, BE> = module.glwe_secret_prepared_alloc(rank.into());
        module.glwe_secret_prepare(&mut sk2_prepared, &sks[2]);

        let mut s0s1: GLWESwitchingKey<BE::OwnedBuf, BE::ZnxWord> = module.glwe_switching_key_alloc_from_infos(&layout);
        let mut s1s2: GLWESwitchingKey<BE::OwnedBuf, BE::ZnxWord> = module.glwe_switching_key_alloc_from_infos(&layout);
        module.glwe_switching_key_encrypt_sk(
            &mut s0s1,
            &sks[0],
            &sks[1],
            &mut source_xe,
            &mut source_xa,
            &mut scratch.arena(),
        );
        module.glwe_switching_key_encrypt_sk(
            &mut s1s2,
            &sks[1],
            &sks[2],
            &mut source_xe,
            &mut source_xa,
            &mut scratch.arena(),
        );
        let mut s1s2_prepared: GLWESwitchingKeyPrepared<BE::OwnedBuf, BE> =
            module.glwe_switching_key_prepared_alloc_from_infos(&s1s2);
        module.glwe_switching_key_prepare(&mut s1s2_prepared, &s1s2, &mut scratch.borrow());

        let mut s0s2: GLWESwitchingKey<BE::OwnedBuf, BE::ZnxWord> = module.glwe_switching_key_alloc_from_infos(&view_layout);
        module.gglwe_keyswitch(
            &mut s0s2,
            &s0s1.view_with_digits(Dnum(digits as u32)).unwrap(),
            &s1s2_prepared.to_backend_ref(),
            &mut scratch.borrow(),
        );

        let max_noise: f64 = layout.log2_std_noise_keyswitch(
            &view_layout,
            &view_layout,
            0.5,
            0.5,
            DEFAULT_SIGMA_XE * DEFAULT_SIGMA_XE,
            DEFAULT_SIGMA_XE * DEFAULT_SIGMA_XE,
            0f64,
        ) + 0.5;
        for row in 0..digits {
            for col in 0..rank {
                let noise_have: f64 = module
                    .gglwe_noise(&s0s2, row, col, &sks[0].data.to_ref(), &sk2_prepared, &mut scratch.borrow())
                    .std()
                    .log2();
                assert!(
                    noise_have <= max_noise,
                    "rank={rank}: noise_have: {noise_have} > max_noise: {max_noise}"
                );
            }
        }

        let mut s0s1_copy: GLWESwitchingKey<BE::OwnedBuf, BE::ZnxWord> = module.glwe_switching_key_alloc_from_infos(&view_layout);
        crate::test_suite::noise::gglwe_narrowed_copy(
            module,
            &mut s0s1_copy,
            &s0s1.view_with_digits(Dnum(digits as u32)).unwrap(),
            &mut scratch.borrow(),
        );
        let mut s0s2_copy: GLWESwitchingKey<BE::OwnedBuf, BE::ZnxWord> = module.glwe_switching_key_alloc_from_infos(&view_layout);
        module.gglwe_keyswitch(
            &mut s0s2_copy,
            &s0s1_copy,
            &s1s2_prepared.to_backend_ref(),
            &mut scratch.borrow(),
        );
        assert!(
            s0s2.key.data() == s0s2_copy.key.data(),
            "rank={rank}: the view's switch differs from the narrowed copy's"
        );
        crate::test_suite::noise::assert_gglwe_view_refusals(module, &s0s1, digits, &mut scratch.borrow());
    }
}

#[allow(clippy::too_many_arguments)]
pub fn test_gglwe_switching_key_keyswitch_assign<BE: crate::test_suite::noise::TestBackend>(
    params: &TestParams,
    module: &Module<BE>,
) where
    BE::OwnedBuf: poulpy_hal::layouts::HostDataMut,
    for<'a> BE::BufRef<'a>: poulpy_hal::layouts::HostDataRef,
    for<'a> BE::BufMut<'a>: poulpy_hal::layouts::HostDataMut,
    Module<BE>: GLWESwitchingKeyEncryptSk<BE>
        + GGLWEKeyswitch<BE>
        + GLWESecretPreparedFactory<BE>
        + GGLWENoise<BE>
        + GLWESwitchingKeyPreparedFactory<BE>,
    ScratchOwned<BE>: ScratchOwnedAlloc<BE> + ScratchOwnedBorrow<BE>,
{
    let base2k: usize = params.base2k;
    let out_base2k: usize = base2k - 1;
    let key_base2k: usize = base2k;
    let k_out: usize = 4 * out_base2k + 1;
    let max_dsize: usize = k_out.div_ceil(key_base2k);

    for rank_in in 1_usize..3 {
        for rank_out in 1_usize..3 {
            for dsize in 1_usize..max_dsize + 1 {
                let n: usize = module.n();
                let dsize_in: usize = 1;

                let dnum_in: usize = k_out / out_base2k;

                let gglwe_s0s1_infos = GLWESwitchingKeyLayout {
                    n: n.into(),
                    base2k: out_base2k.into(),
                    dnum: dnum_in.into(),
                    k_aux: (dsize_in * out_base2k + module.log_n()).into(),
                    k_digit: crate::layouts::k_digit_of(out_base2k, dsize_in),
                    k_offset: crate::layouts::TorusPrecision(0),
                    rank_in: rank_in.into(),
                    rank_out: rank_out.into(),
                };

                // Decomposes the rows it switches whole, their guard included.
                let dnum_ksk: usize = gglwe_s0s1_infos.k().as_usize().div_ceil(key_base2k * dsize);

                let gglwe_s1s2_infos = GLWESwitchingKeyLayout {
                    n: n.into(),
                    base2k: key_base2k.into(),
                    dnum: dnum_ksk.into(),
                    k_aux: (dsize * key_base2k + module.log_n()).into(),
                    k_digit: crate::layouts::k_digit_of(key_base2k, dsize),
                    k_offset: crate::layouts::TorusPrecision(0),
                    rank_in: rank_out.into(),
                    rank_out: rank_out.into(),
                };

                let mut gglwe_s0s1: GLWESwitchingKey<BE::OwnedBuf, BE::ZnxWord> =
                    module.glwe_switching_key_alloc_from_infos(&gglwe_s0s1_infos);
                let mut gglwe_s1s2: GLWESwitchingKey<BE::OwnedBuf, BE::ZnxWord> =
                    module.glwe_switching_key_alloc_from_infos(&gglwe_s1s2_infos);

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

                println!(
                    "{} {} {}",
                    gglwe_s0s1_infos.base2k(),
                    gglwe_s0s1_infos.k(),
                    gglwe_s0s1_infos.size()
                );
                println!(
                    "{} {} {}",
                    gglwe_s1s2_infos.base2k(),
                    gglwe_s1s2_infos.k(),
                    gglwe_s1s2_infos.size()
                );
                println!(
                    "{}",
                    module.gglwe_keyswitch_tmp_bytes(&gglwe_s0s1_infos, &gglwe_s0s1_infos, &gglwe_s1s2_infos)
                );

                let var_xs: f64 = 0.5;

                let mut sk0: GLWESecret<BE::OwnedBuf, BE::ZnxWord> = module.glwe_secret_alloc(rank_in.into());
                module.glwe_secret_fill_ternary_prob(&mut sk0, var_xs, &mut source_xs);

                let mut sk1: GLWESecret<BE::OwnedBuf, BE::ZnxWord> = module.glwe_secret_alloc(rank_out.into());
                module.glwe_secret_fill_ternary_prob(&mut sk1, var_xs, &mut source_xs);

                let mut sk2: GLWESecret<BE::OwnedBuf, BE::ZnxWord> = module.glwe_secret_alloc(rank_out.into());
                module.glwe_secret_fill_ternary_prob(&mut sk2, var_xs, &mut source_xs);

                let mut sk2_prepared: GLWESecretPrepared<BE::OwnedBuf, BE> = module.glwe_secret_prepared_alloc(rank_out.into());
                module.glwe_secret_prepare(&mut sk2_prepared, &sk2);

                // gglwe_{s1}(s0) = s0 -> s1
                module.glwe_switching_key_encrypt_sk(
                    &mut gglwe_s0s1,
                    &sk0,
                    &sk1,
                    &mut source_xe,
                    &mut source_xa,
                    &mut scratch_enc.arena(),
                );

                // gglwe_{s2}(s1) -> s1 -> s2
                module.glwe_switching_key_encrypt_sk(
                    &mut gglwe_s1s2,
                    &sk1,
                    &sk2,
                    &mut source_xe,
                    &mut source_xa,
                    &mut scratch_enc.arena(),
                );

                let mut gglwe_s1s2_prepared: GLWESwitchingKeyPrepared<BE::OwnedBuf, BE> =
                    module.glwe_switching_key_prepared_alloc_from_infos(&gglwe_s1s2);
                module.glwe_switching_key_prepare(&mut gglwe_s1s2_prepared, &gglwe_s1s2, &mut scratch_apply.borrow());

                // gglwe_{s1}(s0) (x) gglwe_{s2}(s1) = gglwe_{s2}(s0)
                println!("{} {} {}", gglwe_s0s1.base2k(), gglwe_s0s1.k(), gglwe_s0s1.size());
                println!(
                    "{} {} {}",
                    gglwe_s1s2_prepared.base2k(),
                    gglwe_s1s2_prepared.k(),
                    gglwe_s1s2_prepared.size()
                );
                module.gglwe_keyswitch_assign(
                    &mut gglwe_s0s1,
                    &gglwe_s1s2_prepared.to_backend_ref(),
                    &mut scratch_apply.borrow(),
                );

                let gglwe_s0s2: GLWESwitchingKey<BE::OwnedBuf, BE::ZnxWord> = gglwe_s0s1;

                let max_noise: f64 = gglwe_s1s2_infos.log2_std_noise_keyswitch(
                    &gglwe_s0s1_infos,
                    &gglwe_s0s1_infos,
                    var_xs,
                    var_xs,
                    DEFAULT_SIGMA_XE * DEFAULT_SIGMA_XE,
                    DEFAULT_SIGMA_XE * DEFAULT_SIGMA_XE,
                    0f64,
                ) + 0.5;

                for row in 0..gglwe_s0s2.dnum().as_usize() {
                    for col in 0..gglwe_s0s2.rank_in().as_usize() {
                        let noise_have = gglwe_s0s2
                            .key
                            .noise_stats(
                                module,
                                row,
                                col,
                                &sk0.data.to_ref(),
                                &sk2_prepared,
                                &mut scratch_apply.borrow(),
                            )
                            .std()
                            .log2();
                        assert!(noise_have <= max_noise, "noise_have: {noise_have} > max_noise: {max_noise}")
                    }
                }
            }
        }
    }
}
