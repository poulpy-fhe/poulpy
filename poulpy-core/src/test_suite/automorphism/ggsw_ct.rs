use poulpy_hal::{
    api::{ScratchOwnedAlloc, ScratchOwnedBorrow, VecZnxAutomorphismAssign},
    layouts::{HostBytesBackend, Module, ScalarZnx, ScalarZnxToBackendRef, ScratchOwned},
    source::Source,
    test_suite::TestParams,
};

use crate::{
    EncryptionLayout, GGLWEToGGSWKeyEncryptSk, GGSWAutomorphism, GGSWEncryptSk, GGSWNoise, GLWEAutomorphismKeyEncryptSk,
    ScratchArenaTakeCore,
    encryption::DEFAULT_SIGMA_XE,
    layouts::{
        GGLWEToGGSWKey, GGLWEToGGSWKeyLayout, GGLWEToGGSWKeyPreparedFactory, GGSW, GGSWInfos, GGSWLayout, GLWEAutomorphismKey,
        GLWEAutomorphismKeyPreparedFactory, GLWEInfos, GLWESecret, GLWESecretPreparedFactory, LWEInfos, ModuleCoreAlloc,
        prepared::{GGLWEToGGSWKeyPrepared, GLWEAutomorphismKeyPrepared, GLWESecretPrepared},
    },
    noise::noise_ggsw_keyswitch,
    test_suite::{
        download_scalar_znx, upload_gglwe_to_ggsw_key, upload_ggsw, upload_glwe_automorphism_key, upload_glwe_secret,
        upload_scalar_znx,
    },
};

pub fn test_ggsw_automorphism<BE: crate::test_suite::TestBackend>(params: &TestParams, module: &Module<BE>)
where
    BE::OwnedBuf: poulpy_hal::layouts::HostDataMut,
    for<'a> BE::BufRef<'a>: poulpy_hal::layouts::HostDataRef,
    for<'a> BE::BufMut<'a>: poulpy_hal::layouts::HostDataMut,
    Module<BE>: GGSWEncryptSk<BE>
        + GLWEAutomorphismKeyEncryptSk<BE>
        + GLWEAutomorphismKeyPreparedFactory<BE>
        + GGSWAutomorphism<BE>
        + GGLWEToGGSWKeyPreparedFactory<BE>
        + GGLWEToGGSWKeyEncryptSk<BE>
        + GLWESecretPreparedFactory<BE>
        + VecZnxAutomorphismAssign<BE>
        + GGSWNoise<BE>,
    ScratchOwned<BE>: ScratchOwnedAlloc<BE> + ScratchOwnedBorrow<BE>,
    for<'a> poulpy_hal::layouts::ScratchArena<'a, BE>: ScratchArenaTakeCore<'a, BE>,
{
    let base2k: usize = params.base2k;
    let in_base2k: usize = base2k - 1;
    let key_base2k: usize = base2k;
    let out_base2k: usize = in_base2k; // MUST BE SAME
    let k_in: usize = 4 * in_base2k + 1;
    let max_dsize: usize = k_in.div_ceil(key_base2k);

    let p: i64 = -5;
    for rank in 1_usize..3 {
        for dsize in 1..max_dsize + 1 {
            let k_ksk: usize = k_in + key_base2k * dsize;
            let k_tsk: usize = k_ksk;
            let k_out: usize = k_ksk; // Better capture noise.

            let n: usize = module.n();
            let dnum_in: usize = k_in / in_base2k;
            let dnum_ksk: usize = k_in.div_ceil(key_base2k * dsize);

            let dsize_in: usize = 1;

            let ggsw_in_layout = EncryptionLayout::new_from_default_sigma(GGSWLayout {
                n: n.into(),
                base2k: in_base2k.into(),
                k: k_in.into(),
                dnum: dnum_in.into(),
                dsize: dsize_in.into(),
                rank: rank.into(),
            })
            .unwrap();

            let ggsw_out_layout = EncryptionLayout::new_from_default_sigma(GGSWLayout {
                n: n.into(),
                base2k: out_base2k.into(),
                k: k_out.into(),
                dnum: dnum_in.into(),
                dsize: dsize_in.into(),
                rank: rank.into(),
            })
            .unwrap();

            let tsk_layout = EncryptionLayout::new_from_default_sigma(GGLWEToGGSWKeyLayout {
                n: n.into(),
                base2k: key_base2k.into(),
                k: k_tsk.into(),
                dnum: dnum_ksk.into(),
                dsize: dsize.into(),
                rank: rank.into(),
            })
            .unwrap();

            let auto_key_layout = EncryptionLayout::new_from_default_sigma(GGLWEToGGSWKeyLayout {
                n: n.into(),
                base2k: key_base2k.into(),
                k: k_ksk.into(),
                dnum: dnum_ksk.into(),
                dsize: dsize.into(),
                rank: rank.into(),
            })
            .unwrap();

            let ct_in_template: GGSW<Vec<u8>> = module.ggsw_alloc_from_infos(&ggsw_in_layout);
            let ct_out_template: GGSW<Vec<u8>> = module.ggsw_alloc_from_infos(&ggsw_out_layout);
            let tsk_template: GGLWEToGGSWKey<Vec<u8>> = module.gglwe_to_ggsw_key_alloc_from_infos(&tsk_layout);
            let auto_key_template: GLWEAutomorphismKey<Vec<u8>> = module.glwe_automorphism_key_alloc_from_infos(&auto_key_layout);
            let mut pt_scalar: ScalarZnx<Vec<u8>> = module.scalar_znx_alloc(1);

            let mut source_xs: Source = Source::new([0u8; 32]);
            let mut source_xe: Source = Source::new([0u8; 32]);
            let mut source_xa: Source = Source::new([0u8; 32]);

            let mut scratch: ScratchOwned<BE> = ScratchOwned::alloc(
                (module).ggsw_encrypt_sk_tmp_bytes(&ggsw_in_layout)
                    | (module).glwe_automorphism_key_encrypt_sk_tmp_bytes(&auto_key_layout)
                    | (module).gglwe_to_ggsw_key_encrypt_sk_tmp_bytes(&tsk_layout)
                    | module.ggsw_automorphism_tmp_bytes(&ggsw_out_layout, &ggsw_in_layout, &auto_key_layout, &tsk_layout),
            );

            let var_xs: f64 = 0.5;

            let mut sk: GLWESecret<Vec<u8>> = module.glwe_secret_alloc_from_infos(&ct_out_template);
            sk.fill_ternary_prob(var_xs, &mut source_xs);
            let sk_backend = upload_glwe_secret(module, &sk);

            let mut sk_prepared: GLWESecretPrepared<BE::OwnedBuf, BE> = module.glwe_secret_prepared_alloc_from_infos(&sk);
            module.glwe_secret_prepare(&mut sk_prepared, &sk_backend);

            let mut auto_key = upload_glwe_automorphism_key(module, &auto_key_template);
            let mut tsk = upload_gglwe_to_ggsw_key(module, &tsk_template);

            module.glwe_automorphism_key_encrypt_sk(
                &mut auto_key,
                p,
                &sk_backend,
                &auto_key_layout,
                &mut source_xe,
                &mut source_xa,
                &mut crate::test_suite::scratch_host_arena(&mut scratch),
            );
            module.gglwe_to_ggsw_key_encrypt_sk(
                &mut tsk,
                &sk_backend,
                &tsk_layout,
                &mut source_xe,
                &mut source_xa,
                &mut crate::test_suite::scratch_host_arena(&mut scratch),
            );

            pt_scalar.fill_ternary_hw(0, n, &mut source_xs);
            let mut pt_scalar_backend = upload_scalar_znx::<BE>(&pt_scalar);
            let mut ct_in = upload_ggsw(module, &ct_in_template);

            module.ggsw_encrypt_sk(
                &mut ct_in,
                &pt_scalar_backend,
                &sk_prepared,
                &ggsw_in_layout,
                &mut source_xe,
                &mut source_xa,
                &mut scratch.borrow(),
            );

            let mut auto_key_prepared: GLWEAutomorphismKeyPrepared<BE::OwnedBuf, BE> =
                module.glwe_automorphism_key_prepared_alloc_from_infos(&auto_key_layout);
            module.glwe_automorphism_key_prepare(&mut auto_key_prepared, &auto_key, &mut scratch.borrow());

            let mut tsk_prepared: GGLWEToGGSWKeyPrepared<BE::OwnedBuf, BE> =
                module.gglwe_to_ggsw_key_prepared_alloc_from_infos(&tsk);
            module.gglwe_to_ggsw_key_prepare(&mut tsk_prepared, &tsk, &mut scratch.borrow());

            let mut ct_out = upload_ggsw(module, &ct_out_template);
            module.ggsw_automorphism(
                &mut ct_out,
                &ct_in,
                &auto_key_prepared,
                auto_key_prepared.size(),
                &tsk_prepared,
                tsk_prepared.size(),
                &mut scratch.borrow(),
            );

            {
                let mut pt_scalar_backend_as_vec =
                    crate::test_suite::scalar_znx_as_vec_znx_backend_mut::<BE>(&mut pt_scalar_backend);
                module.vec_znx_automorphism_assign(p, &mut pt_scalar_backend_as_vec, 0, &mut scratch.borrow());
            }
            let pt_scalar_noise = download_scalar_znx::<BE>(&pt_scalar_backend);

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

            for row in 0..ct_out_template.dnum().as_usize() {
                for col in 0..ct_out_template.rank().as_usize() + 1 {
                    let noise = ct_out
                        .noise(
                            module,
                            row,
                            col,
                            &<ScalarZnx<Vec<u8>> as ScalarZnxToBackendRef<HostBytesBackend>>::to_backend_ref(&pt_scalar_noise),
                            &sk_prepared,
                            &mut scratch.borrow(),
                        )
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
pub fn test_ggsw_automorphism_assign<BE: crate::test_suite::TestBackend>(params: &TestParams, module: &Module<BE>)
where
    BE::OwnedBuf: poulpy_hal::layouts::HostDataMut,
    for<'a> BE::BufRef<'a>: poulpy_hal::layouts::HostDataRef,
    for<'a> BE::BufMut<'a>: poulpy_hal::layouts::HostDataMut,
    Module<BE>: GGSWEncryptSk<BE>
        + GLWEAutomorphismKeyEncryptSk<BE>
        + GLWEAutomorphismKeyPreparedFactory<BE>
        + GGSWAutomorphism<BE>
        + GGLWEToGGSWKeyPreparedFactory<BE>
        + GGLWEToGGSWKeyEncryptSk<BE>
        + GLWESecretPreparedFactory<BE>
        + VecZnxAutomorphismAssign<BE>
        + GGSWNoise<BE>,
    ScratchOwned<BE>: ScratchOwnedAlloc<BE> + ScratchOwnedBorrow<BE>,
    for<'a> poulpy_hal::layouts::ScratchArena<'a, BE>: ScratchArenaTakeCore<'a, BE>,
{
    let base2k: usize = params.base2k;
    let out_base2k: usize = base2k - 1;
    let key_base2k: usize = base2k;
    let k_out: usize = 4 * out_base2k + 1;
    let max_dsize: usize = k_out.div_ceil(key_base2k);

    let p: i64 = -1;
    for rank in 1_usize..3 {
        for dsize in 1..max_dsize + 1 {
            let k_ksk: usize = k_out + key_base2k * dsize;
            let k_tsk: usize = k_ksk;

            let n: usize = module.n();
            let dnum_in: usize = k_out / out_base2k;
            let dnum_ksk: usize = k_out.div_ceil(key_base2k * dsize);
            let dsize_in: usize = 1;

            let ggsw_out_layout = EncryptionLayout::new_from_default_sigma(GGSWLayout {
                n: n.into(),
                base2k: out_base2k.into(),
                k: k_out.into(),
                dnum: dnum_in.into(),
                dsize: dsize_in.into(),
                rank: rank.into(),
            })
            .unwrap();

            let tsk_layout = EncryptionLayout::new_from_default_sigma(GGLWEToGGSWKeyLayout {
                n: n.into(),
                base2k: key_base2k.into(),
                k: k_tsk.into(),
                dnum: dnum_ksk.into(),
                dsize: dsize.into(),
                rank: rank.into(),
            })
            .unwrap();

            let auto_key_layout = EncryptionLayout::new_from_default_sigma(GGLWEToGGSWKeyLayout {
                n: n.into(),
                base2k: key_base2k.into(),
                k: k_ksk.into(),
                dnum: dnum_ksk.into(),
                dsize: dsize.into(),
                rank: rank.into(),
            })
            .unwrap();

            let ct_template: GGSW<Vec<u8>> = module.ggsw_alloc_from_infos(&ggsw_out_layout);
            let tsk_template: GGLWEToGGSWKey<Vec<u8>> = module.gglwe_to_ggsw_key_alloc_from_infos(&tsk_layout);
            let auto_key_template: GLWEAutomorphismKey<Vec<u8>> = module.glwe_automorphism_key_alloc_from_infos(&auto_key_layout);
            let mut pt_scalar: ScalarZnx<Vec<u8>> = module.scalar_znx_alloc(1);

            let mut source_xs: Source = Source::new([0u8; 32]);
            let mut source_xe: Source = Source::new([0u8; 32]);
            let mut source_xa: Source = Source::new([0u8; 32]);

            let mut scratch: ScratchOwned<BE> = ScratchOwned::alloc(
                (module).ggsw_encrypt_sk_tmp_bytes(&ggsw_out_layout)
                    | (module).glwe_automorphism_key_encrypt_sk_tmp_bytes(&auto_key_layout)
                    | (module).gglwe_to_ggsw_key_encrypt_sk_tmp_bytes(&tsk_layout)
                    | module.ggsw_automorphism_tmp_bytes(&ggsw_out_layout, &ggsw_out_layout, &auto_key_layout, &tsk_layout),
            );

            let var_xs: f64 = 0.5;

            let mut sk: GLWESecret<Vec<u8>> = module.glwe_secret_alloc_from_infos(&ct_template);
            sk.fill_ternary_prob(var_xs, &mut source_xs);
            let sk_backend = upload_glwe_secret(module, &sk);

            let mut sk_prepared: GLWESecretPrepared<BE::OwnedBuf, BE> = module.glwe_secret_prepared_alloc_from_infos(&sk);
            module.glwe_secret_prepare(&mut sk_prepared, &sk_backend);

            let mut auto_key = upload_glwe_automorphism_key(module, &auto_key_template);
            let mut tsk = upload_gglwe_to_ggsw_key(module, &tsk_template);

            module.glwe_automorphism_key_encrypt_sk(
                &mut auto_key,
                p,
                &sk_backend,
                &auto_key_layout,
                &mut source_xe,
                &mut source_xa,
                &mut crate::test_suite::scratch_host_arena(&mut scratch),
            );
            module.gglwe_to_ggsw_key_encrypt_sk(
                &mut tsk,
                &sk_backend,
                &tsk_layout,
                &mut source_xe,
                &mut source_xa,
                &mut crate::test_suite::scratch_host_arena(&mut scratch),
            );

            pt_scalar.fill_ternary_hw(0, n, &mut source_xs);
            let mut pt_scalar_backend = upload_scalar_znx::<BE>(&pt_scalar);
            let mut ct = upload_ggsw(module, &ct_template);

            module.ggsw_encrypt_sk(
                &mut ct,
                &pt_scalar_backend,
                &sk_prepared,
                &ggsw_out_layout,
                &mut source_xe,
                &mut source_xa,
                &mut scratch.borrow(),
            );

            let mut auto_key_prepared: GLWEAutomorphismKeyPrepared<BE::OwnedBuf, BE> =
                module.glwe_automorphism_key_prepared_alloc_from_infos(&auto_key_layout);
            module.glwe_automorphism_key_prepare(&mut auto_key_prepared, &auto_key, &mut scratch.borrow());

            let mut tsk_prepared: GGLWEToGGSWKeyPrepared<BE::OwnedBuf, BE> =
                module.gglwe_to_ggsw_key_prepared_alloc_from_infos(&tsk);
            module.gglwe_to_ggsw_key_prepare(&mut tsk_prepared, &tsk, &mut scratch.borrow());

            module.ggsw_automorphism_assign(
                &mut ct,
                &auto_key_prepared,
                auto_key_prepared.size(),
                &tsk_prepared,
                tsk_prepared.size(),
                &mut scratch.borrow(),
            );

            {
                let mut pt_scalar_backend_as_vec =
                    crate::test_suite::scalar_znx_as_vec_znx_backend_mut::<BE>(&mut pt_scalar_backend);
                module.vec_znx_automorphism_assign(p, &mut pt_scalar_backend_as_vec, 0, &mut scratch.borrow());
            }
            let pt_scalar_noise = download_scalar_znx::<BE>(&pt_scalar_backend);

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
                ) + 4.0
            };

            for row in 0..ct.dnum().as_usize() {
                for col in 0..ct.rank().as_usize() + 1 {
                    let noise_have: f64 = ct
                        .noise(
                            module,
                            row,
                            col,
                            &<ScalarZnx<Vec<u8>> as ScalarZnxToBackendRef<HostBytesBackend>>::to_backend_ref(&pt_scalar_noise),
                            &sk_prepared,
                            &mut scratch.borrow(),
                        )
                        .std()
                        .log2();
                    let noise_max: f64 = max_noise(col);
                    assert!(noise_have <= noise_max, "noise_have:{noise_have} > noise_max:{noise_max}")
                }
            }
        }
    }
}
