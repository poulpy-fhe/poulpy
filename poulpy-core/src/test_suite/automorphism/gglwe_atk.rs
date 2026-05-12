use poulpy_hal::{
    api::{ScratchOwnedAlloc, ScratchOwnedBorrow, VecZnxAutomorphismBackend},
    layouts::{Backend, GaloisElement, Module, ScalarZnx, ScratchOwned},
    source::Source,
    test_suite::TestParams,
};

use crate::{
    DEFAULT_SIGMA_XE, EncryptionLayout, GGLWENoise, GLWEAutomorphismKeyAutomorphism, GLWEAutomorphismKeyEncryptSk,
    ScratchArenaTakeCore,
    layouts::{
        GGLWEInfos, GLWEAutomorphismKey, GLWEAutomorphismKeyLayout, GLWEAutomorphismKeyPreparedFactory, GLWEInfos, GLWESecret,
        GLWESecretPreparedFactory, LWEInfos, ModuleCoreAlloc,
        prepared::{GLWEAutomorphismKeyPrepared, GLWESecretPrepared},
    },
    var_noise_gglwe_product_v2,
};

#[allow(clippy::too_many_arguments)]
pub fn test_gglwe_automorphism_key_automorphism<BE: crate::test_suite::TestBackend + Backend<OwnedBuf = Vec<u8>>>(
    params: &TestParams,
    module: &Module<BE>,
) where
    BE::OwnedBuf: poulpy_hal::layouts::HostDataMut,
    for<'a> BE::BufRef<'a>: poulpy_hal::layouts::HostDataRef,
    for<'a> BE::BufMut<'a>: poulpy_hal::layouts::HostDataMut,
    Module<BE>: GLWEAutomorphismKeyEncryptSk<BE>
        + GLWEAutomorphismKeyPreparedFactory<BE>
        + GLWEAutomorphismKeyAutomorphism<BE>
        + VecZnxAutomorphismBackend<BE>
        + GaloisElement
        + GLWESecretPreparedFactory<BE>
        + GGLWENoise<BE>,
    ScratchOwned<BE>: ScratchOwnedAlloc<BE> + ScratchOwnedBorrow<BE>,
    for<'a> poulpy_hal::layouts::ScratchArena<'a, BE>: ScratchArenaTakeCore<'a, BE>,
{
    let base2k: usize = params.base2k;
    let in_base2k: usize = base2k - 1;
    let key_base2k: usize = base2k;
    let out_base2k: usize = in_base2k; // MUST BE SAME
    let k_in: usize = 4 * in_base2k + 1;
    let max_dsize: usize = k_in.div_ceil(key_base2k);
    let p0: i64 = -1;

    let p1: i64 = -5;
    for rank in 1_usize..3 {
        for dsize in 1..max_dsize + 1 {
            let k_ksk: usize = k_in + key_base2k * dsize;
            let k_out: usize = k_ksk; // Better capture noise.

            let n: usize = module.n();
            let dsize_in: usize = 1;

            let dnum_in: usize = k_in / in_base2k;
            let dnum_ksk: usize = k_in.div_ceil(key_base2k * dsize);

            let auto_key_in_infos = EncryptionLayout::new_from_default_sigma(GLWEAutomorphismKeyLayout {
                n: n.into(),
                base2k: in_base2k.into(),
                k: k_in.into(),
                dnum: dnum_in.into(),
                dsize: dsize_in.into(),
                rank: rank.into(),
            })
            .unwrap();

            let auto_key_out_infos = EncryptionLayout::new_from_default_sigma(GLWEAutomorphismKeyLayout {
                n: n.into(),
                base2k: out_base2k.into(),
                k: k_out.into(),
                dnum: dnum_in.into(),
                dsize: dsize_in.into(),
                rank: rank.into(),
            })
            .unwrap();

            let auto_key_apply_infos = EncryptionLayout::new_from_default_sigma(GLWEAutomorphismKeyLayout {
                n: n.into(),
                base2k: key_base2k.into(),
                k: k_ksk.into(),
                dnum: dnum_ksk.into(),
                dsize: dsize.into(),
                rank: rank.into(),
            })
            .unwrap();

            let mut auto_key_in: GLWEAutomorphismKey<Vec<u8>> = module.glwe_automorphism_key_alloc_from_infos(&auto_key_in_infos);
            let mut auto_key_out: GLWEAutomorphismKey<Vec<u8>> =
                module.glwe_automorphism_key_alloc_from_infos(&auto_key_out_infos);
            let mut auto_key_apply: GLWEAutomorphismKey<Vec<u8>> =
                module.glwe_automorphism_key_alloc_from_infos(&auto_key_apply_infos);

            let mut source_xs: Source = Source::new([0u8; 32]);
            let mut source_xe: Source = Source::new([0u8; 32]);
            let mut source_xa: Source = Source::new([0u8; 32]);

            let mut scratch: ScratchOwned<BE> = ScratchOwned::alloc(
                (module)
                    .glwe_automorphism_key_encrypt_sk_tmp_bytes(&auto_key_in_infos)
                    .max((module).glwe_automorphism_key_encrypt_sk_tmp_bytes(&auto_key_apply_infos))
                    .max(module.glwe_automorphism_key_automorphism_tmp_bytes(
                        &auto_key_out_infos,
                        &auto_key_in_infos,
                        &auto_key_apply_infos,
                    )),
            );

            let mut sk: GLWESecret<Vec<u8>> = module.glwe_secret_alloc_from_infos(&auto_key_in);
            sk.fill_ternary_prob(0.5, &mut source_xs);

            // gglwe_{s1}(s0) = s0 -> s1
            module.glwe_automorphism_key_encrypt_sk(
                &mut auto_key_in,
                p0,
                &sk,
                &auto_key_in_infos,
                &mut source_xe,
                &mut source_xa,
                &mut crate::test_suite::scratch_host_arena(&mut scratch),
            );

            // gglwe_{s2}(s1) -> s1 -> s2
            module.glwe_automorphism_key_encrypt_sk(
                &mut auto_key_apply,
                p1,
                &sk,
                &auto_key_apply_infos,
                &mut source_xe,
                &mut source_xa,
                &mut crate::test_suite::scratch_host_arena(&mut scratch),
            );

            let mut auto_key_apply_prepared: GLWEAutomorphismKeyPrepared<BE::OwnedBuf, BE> =
                module.glwe_automorphism_key_prepared_alloc_from_infos(&auto_key_apply_infos);

            module.glwe_automorphism_key_prepare(&mut auto_key_apply_prepared, &auto_key_apply, &mut scratch.borrow());

            // gglwe_{s1}(s0) (x) gglwe_{s2}(s1) = gglwe_{s2}(s0)
            module.glwe_automorphism_key_automorphism(
                &mut auto_key_out,
                &auto_key_in,
                &auto_key_apply_prepared,
                auto_key_apply_prepared.size(),
                &mut scratch.borrow(),
            );

            let mut sk_auto: GLWESecret<Vec<u8>> = module.glwe_secret_alloc_from_infos(&auto_key_out_infos);
            sk_auto.fill_zero(); // Necessary to avoid panic of unfilled sk
            let sk_backend = ScalarZnx::from_data(BE::from_host_bytes(sk.data.data.as_ref()), sk.data.n(), sk.data.cols());
            let mut sk_auto_backend = ScalarZnx::from_data(
                BE::from_host_bytes(sk_auto.data.data.as_ref()),
                sk_auto.data.n(),
                sk_auto.data.cols(),
            );
            {
                let sk_backend_as_vec = crate::test_suite::scalar_znx_as_vec_znx_backend_ref::<BE>(&sk_backend);
                let mut sk_auto_backend_as_vec = crate::test_suite::scalar_znx_as_vec_znx_backend_mut::<BE>(&mut sk_auto_backend);
                for i in 0..rank {
                    module.vec_znx_automorphism_backend(
                        module.galois_element_inv(p0 * p1),
                        &mut sk_auto_backend_as_vec,
                        i,
                        &sk_backend_as_vec,
                        i,
                    );
                }
            }
            BE::copy_to_host(&sk_auto_backend.data, sk_auto.data.data.as_mut());

            let mut sk_auto_dft: GLWESecretPrepared<BE::OwnedBuf, BE> = module.glwe_secret_prepared_alloc_from_infos(&sk_auto);
            module.glwe_secret_prepare(&mut sk_auto_dft, &sk_auto);

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
                rank as f64,
            )
            .sqrt()
            .log2()
                + 0.5;

            for row in 0..auto_key_out.dnum().as_usize() {
                for col in 0..auto_key_out.rank().as_usize() {
                    let noise_have = auto_key_out
                        .key
                        .noise(module, row, col, &sk.data.to_ref(), &sk_auto_dft, &mut scratch.borrow())
                        .std()
                        .log2();

                    assert!(noise_have <= max_noise, "noise_have: {noise_have} > max_noise: {max_noise}");
                }
            }
        }
    }
}

#[allow(clippy::too_many_arguments)]
pub fn test_gglwe_automorphism_key_automorphism_assign<BE: crate::test_suite::TestBackend + Backend<OwnedBuf = Vec<u8>>>(
    params: &TestParams,
    module: &Module<BE>,
) where
    BE::OwnedBuf: poulpy_hal::layouts::HostDataMut,
    for<'a> BE::BufRef<'a>: poulpy_hal::layouts::HostDataRef,
    for<'a> BE::BufMut<'a>: poulpy_hal::layouts::HostDataMut,
    Module<BE>: GLWEAutomorphismKeyEncryptSk<BE>
        + GLWEAutomorphismKeyPreparedFactory<BE>
        + GLWEAutomorphismKeyAutomorphism<BE>
        + VecZnxAutomorphismBackend<BE>
        + GaloisElement
        + GLWESecretPreparedFactory<BE>
        + GGLWENoise<BE>,
    ScratchOwned<BE>: ScratchOwnedAlloc<BE> + ScratchOwnedBorrow<BE>,
    for<'a> poulpy_hal::layouts::ScratchArena<'a, BE>: ScratchArenaTakeCore<'a, BE>,
{
    let base2k: usize = params.base2k;
    let out_base2k: usize = base2k - 1;
    let key_base2k: usize = base2k;
    let k_out: usize = 4 * out_base2k + 1;
    let max_dsize: usize = k_out.div_ceil(key_base2k);

    let p0: i64 = -1;
    let p1: i64 = -5;
    for rank in 1_usize..3 {
        for dsize in 1..max_dsize + 1 {
            let k_ksk: usize = k_out + key_base2k * dsize;

            let n: usize = module.n();
            let dsize_in: usize = 1;

            let dnum_in: usize = k_out / out_base2k;
            let dnum_ksk: usize = k_out.div_ceil(key_base2k * dsize);

            let auto_key_layout = EncryptionLayout::new_from_default_sigma(GLWEAutomorphismKeyLayout {
                n: n.into(),
                base2k: out_base2k.into(),
                k: k_out.into(),
                dnum: dnum_in.into(),
                dsize: dsize_in.into(),
                rank: rank.into(),
            })
            .unwrap();

            let auto_key_apply_layout = EncryptionLayout::new_from_default_sigma(GLWEAutomorphismKeyLayout {
                n: n.into(),
                base2k: key_base2k.into(),
                k: k_ksk.into(),
                dnum: dnum_ksk.into(),
                dsize: dsize.into(),
                rank: rank.into(),
            })
            .unwrap();

            let mut auto_key: GLWEAutomorphismKey<Vec<u8>> = module.glwe_automorphism_key_alloc_from_infos(&auto_key_layout);
            let mut auto_key_apply: GLWEAutomorphismKey<Vec<u8>> =
                module.glwe_automorphism_key_alloc_from_infos(&auto_key_apply_layout);

            let mut source_xs: Source = Source::new([0u8; 32]);
            let mut source_xe: Source = Source::new([0u8; 32]);
            let mut source_xa: Source = Source::new([0u8; 32]);

            let mut scratch: ScratchOwned<BE> = ScratchOwned::alloc(
                (module).glwe_automorphism_key_encrypt_sk_tmp_bytes(&auto_key)
                    | (module).glwe_automorphism_key_encrypt_sk_tmp_bytes(&auto_key_apply)
                    | module.glwe_automorphism_key_automorphism_tmp_bytes(&auto_key, &auto_key, &auto_key_apply),
            );

            let mut sk: GLWESecret<Vec<u8>> = module.glwe_secret_alloc_from_infos(&auto_key);
            sk.fill_ternary_prob(0.5, &mut source_xs);

            // gglwe_{s1}(s0) = s0 -> s1
            module.glwe_automorphism_key_encrypt_sk(
                &mut auto_key,
                p0,
                &sk,
                &auto_key_layout,
                &mut source_xe,
                &mut source_xa,
                &mut crate::test_suite::scratch_host_arena(&mut scratch),
            );

            // gglwe_{s2}(s1) -> s1 -> s2
            module.glwe_automorphism_key_encrypt_sk(
                &mut auto_key_apply,
                p1,
                &sk,
                &auto_key_apply_layout,
                &mut source_xe,
                &mut source_xa,
                &mut crate::test_suite::scratch_host_arena(&mut scratch),
            );

            let mut auto_key_apply_prepared: GLWEAutomorphismKeyPrepared<BE::OwnedBuf, BE> =
                module.glwe_automorphism_key_prepared_alloc_from_infos(&auto_key_apply_layout);

            module.glwe_automorphism_key_prepare(&mut auto_key_apply_prepared, &auto_key_apply, &mut scratch.borrow());

            // gglwe_{s1}(s0) (x) gglwe_{s2}(s1) = gglwe_{s2}(s0)
            module.glwe_automorphism_key_automorphism_assign(
                &mut auto_key,
                &auto_key_apply_prepared,
                auto_key_apply_prepared.size(),
                &mut scratch.borrow(),
            );

            let mut sk_auto: GLWESecret<Vec<u8>> = module.glwe_secret_alloc_from_infos(&auto_key);
            sk_auto.fill_zero(); // Necessary to avoid panic of unfilled sk

            let sk_backend = ScalarZnx::from_data(BE::from_host_bytes(sk.data.data.as_ref()), sk.data.n(), sk.data.cols());
            let mut sk_auto_backend = ScalarZnx::from_data(
                BE::from_host_bytes(sk_auto.data.data.as_ref()),
                sk_auto.data.n(),
                sk_auto.data.cols(),
            );
            {
                let sk_backend_as_vec = crate::test_suite::scalar_znx_as_vec_znx_backend_ref::<BE>(&sk_backend);
                let mut sk_auto_backend_as_vec = crate::test_suite::scalar_znx_as_vec_znx_backend_mut::<BE>(&mut sk_auto_backend);
                for i in 0..rank {
                    module.vec_znx_automorphism_backend(
                        module.galois_element_inv(p0 * p1),
                        &mut sk_auto_backend_as_vec,
                        i,
                        &sk_backend_as_vec,
                        i,
                    );
                }
            }
            BE::copy_to_host(&sk_auto_backend.data, sk_auto.data.data.as_mut());

            let mut sk_auto_dft: GLWESecretPrepared<BE::OwnedBuf, BE> = module.glwe_secret_prepared_alloc_from_infos(&sk_auto);
            module.glwe_secret_prepare(&mut sk_auto_dft, &sk_auto);

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
                rank as f64,
            )
            .sqrt()
            .log2()
                + 0.5;

            for row in 0..auto_key.dnum().as_usize() {
                for col in 0..auto_key.rank().as_usize() {
                    let noise_have = auto_key
                        .key
                        .noise(module, row, col, &sk.data.to_ref(), &sk_auto_dft, &mut scratch.borrow())
                        .std()
                        .log2();

                    assert!(noise_have <= max_noise, "noise_have: {noise_have} > max_noise: {max_noise}");
                }
            }
        }
    }
}
