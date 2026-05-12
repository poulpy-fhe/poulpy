use poulpy_hal::{
    api::{ScratchOwnedAlloc, ScratchOwnedBorrow, VecZnxFillUniformSourceBackend},
    layouts::{Module, ScratchOwned},
    source::Source,
    test_suite::{TestParams, vec_znx_backend_mut},
};

use crate::{
    EncryptionLayout, GLWEEncryptSk, GLWEKeyswitch, GLWENoise, GLWENormalize, GLWESwitchingKeyEncryptSk, ScratchArenaTakeCore,
    encryption::DEFAULT_SIGMA_XE,
    layouts::{
        GLWE, GLWELayout, GLWEPlaintext, GLWESecret, GLWESecretPreparedFactory, GLWESwitchingKey, GLWESwitchingKeyLayout,
        GLWESwitchingKeyPreparedFactory, LWEInfos, ModuleCoreAlloc,
        prepared::{GLWESecretPrepared, GLWESwitchingKeyPrepared},
    },
    var_noise_gglwe_product_v2,
};

#[allow(clippy::too_many_arguments)]
pub fn test_glwe_keyswitch<BE: crate::test_suite::TestBackend>(params: &TestParams, module: &Module<BE>)
where
    BE::OwnedBuf: poulpy_hal::layouts::HostDataMut,
    for<'a> BE::BufRef<'a>: poulpy_hal::layouts::HostDataRef,
    for<'a> BE::BufMut<'a>: poulpy_hal::layouts::HostDataMut,
    Module<BE>: VecZnxFillUniformSourceBackend<BE>
        + GLWESwitchingKeyEncryptSk<BE>
        + GLWEEncryptSk<BE>
        + GLWEKeyswitch<BE>
        + GLWESecretPreparedFactory<BE>
        + GLWESwitchingKeyPreparedFactory<BE>
        + GLWENoise<BE>
        + GLWENormalize<BE>,
    ScratchOwned<BE>: ScratchOwnedAlloc<BE> + ScratchOwnedBorrow<BE>,
    for<'a> poulpy_hal::layouts::ScratchArena<'a, BE>: ScratchArenaTakeCore<'a, BE>,
{
    let base2k: usize = params.base2k;
    let in_base2k: usize = base2k - 1;
    let key_base2k: usize = base2k;
    let out_base2k: usize = base2k - 2;
    let k_in: usize = 4 * in_base2k + 1;
    let max_dsize: usize = k_in.div_ceil(key_base2k);

    for rank_in in 1_usize..3 {
        for rank_out in 1_usize..3 {
            for dsize in 1_usize..max_dsize + 1 {
                let k_ksk: usize = k_in + key_base2k * dsize;
                let k_out: usize = k_ksk; // better capture noise

                let n: usize = module.n();
                let dnum: usize = k_in.div_ceil(key_base2k * dsize);

                let glwe_in_infos = EncryptionLayout::new_from_default_sigma(GLWELayout {
                    n: n.into(),
                    base2k: in_base2k.into(),
                    k: k_in.into(),
                    rank: rank_in.into(),
                })
                .unwrap();

                let glwe_out_infos: GLWELayout = GLWELayout {
                    n: n.into(),
                    base2k: out_base2k.into(),
                    k: k_out.into(),
                    rank: rank_out.into(),
                };

                let ksk_infos = EncryptionLayout::new_from_default_sigma(GLWESwitchingKeyLayout {
                    n: n.into(),
                    base2k: key_base2k.into(),
                    k: k_ksk.into(),
                    dnum: dnum.into(),
                    dsize: dsize.into(),
                    rank_in: rank_in.into(),
                    rank_out: rank_out.into(),
                })
                .unwrap();

                let mut ksk: GLWESwitchingKey<Vec<u8>> = module.glwe_switching_key_alloc_from_infos(&ksk_infos);
                let mut glwe_in: GLWE<Vec<u8>> = module.glwe_alloc_from_infos(&glwe_in_infos);
                let mut glwe_out: GLWE<Vec<u8>> = module.glwe_alloc_from_infos(&glwe_out_infos);
                let mut pt_in: GLWEPlaintext<Vec<u8>> = module.glwe_plaintext_alloc_from_infos(&glwe_in_infos);
                let mut pt_out: GLWEPlaintext<Vec<u8>> = module.glwe_plaintext_alloc_from_infos(&glwe_out_infos);

                let mut source_xs: Source = Source::new([0u8; 32]);
                let mut source_xe: Source = Source::new([0u8; 32]);
                let mut source_xa: Source = Source::new([0u8; 32]);

                module.vec_znx_fill_uniform_source_backend(
                    pt_in.base2k().into(),
                    &mut vec_znx_backend_mut::<BE>(&mut pt_in.data),
                    0,
                    &mut source_xa,
                );

                let mut scratch: ScratchOwned<BE> = ScratchOwned::alloc(
                    (module).glwe_switching_key_encrypt_sk_tmp_bytes(&ksk_infos)
                        | (module).glwe_encrypt_sk_tmp_bytes(&glwe_in_infos)
                        | module.glwe_keyswitch_tmp_bytes(&glwe_out_infos, &glwe_in_infos, &ksk_infos),
                );

                let mut sk_in: GLWESecret<Vec<u8>> = module.glwe_secret_alloc(rank_in.into());
                sk_in.fill_ternary_prob(0.5, &mut source_xs);

                let mut sk_in_prepared: GLWESecretPrepared<BE::OwnedBuf, BE> = module.glwe_secret_prepared_alloc(rank_in.into());
                module.glwe_secret_prepare(&mut sk_in_prepared, &sk_in);

                let mut sk_out: GLWESecret<Vec<u8>> = module.glwe_secret_alloc(rank_out.into());
                sk_out.fill_ternary_prob(0.5, &mut source_xs);

                let mut sk_out_prepared: GLWESecretPrepared<BE::OwnedBuf, BE> =
                    module.glwe_secret_prepared_alloc(rank_out.into());
                module.glwe_secret_prepare(&mut sk_out_prepared, &sk_out);

                module.glwe_switching_key_encrypt_sk(
                    &mut ksk,
                    &sk_in,
                    &sk_out,
                    &ksk_infos,
                    &mut source_xe,
                    &mut source_xa,
                    &mut scratch.arena(),
                );

                module.glwe_encrypt_sk(
                    &mut glwe_in,
                    &pt_in,
                    &sk_in_prepared,
                    &glwe_in_infos,
                    &mut source_xe,
                    &mut source_xa,
                    &mut scratch.borrow(),
                );

                let mut ksk_prepared: GLWESwitchingKeyPrepared<BE::OwnedBuf, BE> =
                    module.glwe_switching_key_prepared_alloc_from_infos(&ksk);
                module.glwe_switching_key_prepare(&mut ksk_prepared, &ksk, &mut scratch.borrow());

                module.glwe_keyswitch(
                    &mut glwe_out,
                    &glwe_in,
                    &ksk_prepared,
                    ksk_prepared.size(),
                    &mut scratch.borrow(),
                );

                let noise_max: f64 = var_noise_gglwe_product_v2(
                    module.n() as f64,
                    k_ksk,
                    dnum,
                    dsize,
                    key_base2k,
                    0.5,
                    0.5,
                    0f64,
                    DEFAULT_SIGMA_XE * DEFAULT_SIGMA_XE,
                    0f64,
                    rank_in as f64,
                )
                .sqrt()
                .log2()
                    + 1.0;

                module.glwe_normalize(&mut pt_out, &pt_in, &mut scratch.borrow());

                let noise_have = module
                    .glwe_noise(&glwe_out, &pt_out, &sk_out_prepared, &mut scratch.borrow())
                    .std()
                    .log2();

                assert!(noise_have <= noise_max, "noise_have: {noise_have} > noise_max: {noise_max}");
            }
        }
    }
}

pub fn test_glwe_keyswitch_assign<BE: crate::test_suite::TestBackend>(params: &TestParams, module: &Module<BE>)
where
    BE::OwnedBuf: poulpy_hal::layouts::HostDataMut,
    for<'a> BE::BufRef<'a>: poulpy_hal::layouts::HostDataRef,
    for<'a> BE::BufMut<'a>: poulpy_hal::layouts::HostDataMut,
    Module<BE>: VecZnxFillUniformSourceBackend<BE>
        + GLWESwitchingKeyEncryptSk<BE>
        + GLWEEncryptSk<BE>
        + GLWEKeyswitch<BE>
        + GLWESecretPreparedFactory<BE>
        + GLWESwitchingKeyPreparedFactory<BE>
        + GLWENoise<BE>,
    ScratchOwned<BE>: ScratchOwnedAlloc<BE> + ScratchOwnedBorrow<BE>,
    for<'a> poulpy_hal::layouts::ScratchArena<'a, BE>: ScratchArenaTakeCore<'a, BE>,
{
    let base2k: usize = params.base2k;
    let out_base2k: usize = base2k - 1;
    let key_base2k: usize = base2k;
    let k_out: usize = 4 * out_base2k + 1;
    let max_dsize: usize = k_out.div_ceil(key_base2k);

    for rank in 1_usize..3 {
        for dsize in 1..max_dsize + 1 {
            let k_ksk: usize = k_out + key_base2k * dsize;

            let n: usize = module.n();
            let dnum: usize = k_out.div_ceil(key_base2k * dsize);
            let glwe_out_infos = EncryptionLayout::new_from_default_sigma(GLWELayout {
                n: n.into(),
                base2k: out_base2k.into(),
                k: k_out.into(),
                rank: rank.into(),
            })
            .unwrap();

            let ksk_infos = EncryptionLayout::new_from_default_sigma(GLWESwitchingKeyLayout {
                n: n.into(),
                base2k: key_base2k.into(),
                k: k_ksk.into(),
                dnum: dnum.into(),
                dsize: dsize.into(),
                rank_in: rank.into(),
                rank_out: rank.into(),
            })
            .unwrap();

            let mut ksk: GLWESwitchingKey<Vec<u8>> = module.glwe_switching_key_alloc_from_infos(&ksk_infos);
            let mut glwe_out: GLWE<Vec<u8>> = module.glwe_alloc_from_infos(&glwe_out_infos);
            let mut pt_want: GLWEPlaintext<Vec<u8>> = module.glwe_plaintext_alloc_from_infos(&glwe_out_infos);

            let mut source_xs: Source = Source::new([0u8; 32]);
            let mut source_xe: Source = Source::new([0u8; 32]);
            let mut source_xa: Source = Source::new([0u8; 32]);

            module.vec_znx_fill_uniform_source_backend(
                pt_want.base2k().into(),
                &mut vec_znx_backend_mut::<BE>(&mut pt_want.data),
                0,
                &mut source_xa,
            );

            let mut scratch: ScratchOwned<BE> = ScratchOwned::alloc(
                (module).glwe_switching_key_encrypt_sk_tmp_bytes(&ksk_infos)
                    | (module).glwe_encrypt_sk_tmp_bytes(&glwe_out_infos)
                    | module.glwe_keyswitch_tmp_bytes(&glwe_out_infos, &glwe_out_infos, &ksk_infos),
            );

            let mut sk_in: GLWESecret<Vec<u8>> = module.glwe_secret_alloc(rank.into());
            sk_in.fill_ternary_prob(0.5, &mut source_xs);

            let mut sk_in_prepared: GLWESecretPrepared<BE::OwnedBuf, BE> = module.glwe_secret_prepared_alloc(rank.into());
            module.glwe_secret_prepare(&mut sk_in_prepared, &sk_in);

            let mut sk_out: GLWESecret<Vec<u8>> = module.glwe_secret_alloc(rank.into());
            sk_out.fill_ternary_prob(0.5, &mut source_xs);

            let mut sk_out_prepared: GLWESecretPrepared<BE::OwnedBuf, BE> = module.glwe_secret_prepared_alloc(rank.into());
            module.glwe_secret_prepare(&mut sk_out_prepared, &sk_out);

            module.glwe_switching_key_encrypt_sk(
                &mut ksk,
                &sk_in,
                &sk_out,
                &ksk_infos,
                &mut source_xe,
                &mut source_xa,
                &mut scratch.arena(),
            );

            module.glwe_encrypt_sk(
                &mut glwe_out,
                &pt_want,
                &sk_in_prepared,
                &glwe_out_infos,
                &mut source_xe,
                &mut source_xa,
                &mut scratch.borrow(),
            );

            let mut ksk_prepared: GLWESwitchingKeyPrepared<BE::OwnedBuf, BE> =
                module.glwe_switching_key_prepared_alloc_from_infos(&ksk);
            module.glwe_switching_key_prepare(&mut ksk_prepared, &ksk, &mut scratch.borrow());

            module.glwe_keyswitch_assign(&mut glwe_out, &ksk_prepared, ksk_prepared.size(), &mut scratch.borrow());

            let noise_max: f64 = var_noise_gglwe_product_v2(
                module.n() as f64,
                k_ksk,
                dnum,
                dsize,
                key_base2k,
                0.5,
                0.5,
                0f64,
                DEFAULT_SIGMA_XE * DEFAULT_SIGMA_XE,
                0f64,
                rank as f64,
            )
            .sqrt()
            .log2()
                + 1.0;

            let noise_have = module
                .glwe_noise(&glwe_out, &pt_want, &sk_out_prepared, &mut scratch.borrow())
                .std()
                .log2();

            assert!(noise_have <= noise_max, "noise_have: {noise_have} > noise_max: {noise_max}");
        }
    }
}
