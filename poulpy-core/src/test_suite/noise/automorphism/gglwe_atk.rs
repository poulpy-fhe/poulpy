use poulpy_hal::AlignedBuf;
use poulpy_hal::{
    api::{ScratchOwnedAlloc, ScratchOwnedBorrow, VecZnxAutomorphism},
    layouts::{Backend, CyclotomicOrder, GaloisElement, Module, ScalarZnx, ScratchOwned},
    source::Source,
    test_suite::TestParams,
};

use crate::layouts::GLWESecretSampling;
use crate::layouts::LWEInfos;
use crate::layouts::prepared::GLWEAutomorphismKeyPreparedToBackendRef;
use crate::{
    DEFAULT_SIGMA_XE, GGLWENoise, GLWEAutomorphismKeyAutomorphism, GLWEAutomorphismKeyEncryptSk,
    layouts::{
        Dnum, GGLWEInfos, GGLWEViewWithDigits, GLWEAutomorphismKey, GLWEAutomorphismKeyLayout,
        GLWEAutomorphismKeyPreparedFactory, GLWEInfos, GLWESecret, GLWESecretPreparedFactory, ModuleCoreAlloc,
        prepared::{GLWEAutomorphismKeyPrepared, GLWESecretPrepared},
    },
    noise::GGLWENoiseModel,
};

#[allow(clippy::too_many_arguments)]
pub fn test_gglwe_automorphism_key_automorphism<BE: crate::test_suite::noise::TestBackend + Backend<OwnedBuf = AlignedBuf>>(
    params: &TestParams,
    module: &Module<BE>,
) where
    BE::OwnedBuf: poulpy_hal::layouts::HostDataMut,
    for<'a> BE::BufRef<'a>: poulpy_hal::layouts::HostDataRef,
    for<'a> BE::BufMut<'a>: poulpy_hal::layouts::HostDataMut,
    Module<BE>: GLWEAutomorphismKeyEncryptSk<BE>
        + GLWEAutomorphismKeyPreparedFactory<BE>
        + GLWEAutomorphismKeyAutomorphism<BE>
        + VecZnxAutomorphism<BE>
        + GaloisElement
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
    let p0: i64 = -1;

    let p1: i64 = -5;
    for rank in 1_usize..3 {
        for dsize in 1..max_dsize + 1 {
            let n: usize = module.n();
            let dsize_in: usize = 1;

            let dnum_in: usize = k_in / in_base2k;

            let auto_key_in_infos = GLWEAutomorphismKeyLayout {
                n: n.into(),
                base2k: in_base2k.into(),
                dnum: dnum_in.into(),
                k_aux: (dsize_in * in_base2k + module.log_n()).into(),
                k_digit: crate::layouts::k_digit_of(in_base2k, dsize_in),
                rank: rank.into(),
            };

            let auto_key_out_infos = GLWEAutomorphismKeyLayout {
                n: n.into(),
                base2k: out_base2k.into(),
                dnum: dnum_in.into(),
                k_aux: (dsize_in * out_base2k + module.log_n()).into(),
                k_digit: crate::layouts::k_digit_of(out_base2k, dsize_in),
                rank: rank.into(),
            };

            // Decomposes the rows it switches whole, their guard included.
            let dnum_ksk: usize = auto_key_in_infos
                .k()
                .max(auto_key_out_infos.k())
                .as_usize()
                .div_ceil(key_base2k * dsize);

            let auto_key_apply_infos = GLWEAutomorphismKeyLayout {
                n: n.into(),
                base2k: key_base2k.into(),
                dnum: dnum_ksk.into(),
                k_aux: (dsize * key_base2k + module.log_n()).into(),
                k_digit: crate::layouts::k_digit_of(key_base2k, dsize),
                rank: rank.into(),
            };

            let mut auto_key_in: GLWEAutomorphismKey<BE::OwnedBuf, BE::ZnxWord> =
                module.glwe_automorphism_key_alloc_from_infos(&auto_key_in_infos);
            let mut auto_key_out: GLWEAutomorphismKey<BE::OwnedBuf, BE::ZnxWord> =
                module.glwe_automorphism_key_alloc_from_infos(&auto_key_out_infos);
            let mut auto_key_apply: GLWEAutomorphismKey<BE::OwnedBuf, BE::ZnxWord> =
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

            let mut sk: GLWESecret<BE::OwnedBuf, BE::ZnxWord> = module.glwe_secret_alloc_from_infos(&auto_key_in);
            module.glwe_secret_fill_ternary_prob(&mut sk, 0.5, &mut source_xs);

            // gglwe_{s1}(s0) = s0 -> s1
            module.glwe_automorphism_key_encrypt_sk(
                &mut auto_key_in,
                p0,
                &sk,
                &mut source_xe,
                &mut source_xa,
                &mut crate::test_suite::noise::scratch_host_arena(&mut scratch),
            );

            // gglwe_{s2}(s1) -> s1 -> s2
            module.glwe_automorphism_key_encrypt_sk(
                &mut auto_key_apply,
                p1,
                &sk,
                &mut source_xe,
                &mut source_xa,
                &mut crate::test_suite::noise::scratch_host_arena(&mut scratch),
            );

            let mut auto_key_apply_prepared: GLWEAutomorphismKeyPrepared<BE::OwnedBuf, BE> =
                module.glwe_automorphism_key_prepared_alloc_from_infos(&auto_key_apply_infos);

            module.glwe_automorphism_key_prepare(&mut auto_key_apply_prepared, &auto_key_apply, &mut scratch.borrow());

            // gglwe_{s1}(s0) (x) gglwe_{s2}(s1) = gglwe_{s2}(s0)
            module.glwe_automorphism_key_automorphism(
                &mut auto_key_out,
                &auto_key_in,
                &auto_key_apply_prepared.to_backend_ref(),
                &mut scratch.borrow(),
            );

            let mut sk_auto: GLWESecret<BE::OwnedBuf, BE::ZnxWord> = module.glwe_secret_alloc_from_infos(&auto_key_out_infos);
            module.glwe_secret_fill_zero(&mut sk_auto); // Necessary to avoid panic of unfilled sk
            let sk_backend = ScalarZnx::from_data(BE::from_host_bytes(sk.data().data.as_ref()), sk.data().n(), sk.data().cols());
            let mut sk_auto_backend = ScalarZnx::from_data(
                BE::from_host_bytes(sk_auto.data().data.as_ref()),
                sk_auto.data().n(),
                sk_auto.data().cols(),
            );
            {
                let sk_backend_as_vec = crate::test_suite::noise::scalar_znx_as_vec_znx_backend_ref::<BE>(&sk_backend);
                let mut sk_auto_backend_as_vec =
                    crate::test_suite::noise::scalar_znx_as_vec_znx_backend_mut::<BE>(&mut sk_auto_backend);
                for i in 0..rank {
                    module.vec_znx_automorphism(
                        module.galois_element_inv(p0 * p1),
                        &mut sk_auto_backend_as_vec,
                        i,
                        &sk_backend_as_vec,
                        i,
                    );
                }
            }
            BE::copy_to_host(&sk_auto_backend.data, sk_auto.data_mut().data.as_mut());

            let mut sk_auto_dft: GLWESecretPrepared<BE::OwnedBuf, BE> = module.glwe_secret_prepared_alloc_from_infos(&sk_auto);
            module.glwe_secret_prepare(&mut sk_auto_dft, &sk_auto);

            let max_noise: f64 = auto_key_apply_infos.log2_std_noise_keyswitch(
                &auto_key_out_infos,
                &auto_key_in_infos,
                0.5,
                0.5,
                DEFAULT_SIGMA_XE * DEFAULT_SIGMA_XE,
                DEFAULT_SIGMA_XE * DEFAULT_SIGMA_XE,
                0f64,
            ) + 0.5;

            for row in 0..auto_key_out.dnum().as_usize() {
                for col in 0..auto_key_out.rank().as_usize() {
                    let noise_have = auto_key_out
                        .key
                        .noise_stats(module, row, col, &sk.data().to_ref(), &sk_auto_dft, &mut scratch.borrow())
                        .std()
                        .log2();

                    assert!(noise_have <= max_noise, "noise_have: {noise_have} > max_noise: {max_noise}");
                }
            }
        }
    }
}

/// An automorphism key applies to a key of its own layout through a view of
/// the latter's first digits, into a key of that many digits, which takes the
/// composed element: the rows decrypt within the noise model of the view's
/// width.
pub fn test_gglwe_automorphism_key_automorphism_digits<
    BE: crate::test_suite::noise::TestBackend + Backend<OwnedBuf = AlignedBuf>,
>(
    params: &TestParams,
    module: &Module<BE>,
) where
    BE::OwnedBuf: poulpy_hal::layouts::HostDataMut,
    for<'a> BE::BufRef<'a>: poulpy_hal::layouts::HostDataRef,
    for<'a> BE::BufMut<'a>: poulpy_hal::layouts::HostDataMut,
    Module<BE>: GLWEAutomorphismKeyEncryptSk<BE>
        + GLWEAutomorphismKeyPreparedFactory<BE>
        + GLWEAutomorphismKeyAutomorphism<BE>
        + VecZnxAutomorphism<BE>
        + GaloisElement
        + GLWESecretPreparedFactory<BE>
        + GGLWENoise<BE>,
    ScratchOwned<BE>: ScratchOwnedAlloc<BE> + ScratchOwnedBorrow<BE>,
{
    let base2k: usize = params.base2k;
    let n: usize = module.n();
    let dnum: usize = 4;
    let (p0, p1): (i64, i64) = (-1, -5);
    for rank in 1_usize..3 {
        let layout = GLWEAutomorphismKeyLayout {
            n: n.into(),
            base2k: base2k.into(),
            dnum: dnum.into(),
            k_aux: (base2k + module.log_n()).into(),
            k_digit: crate::layouts::k_digit_of(base2k, 1_usize),
            rank: rank.into(),
        };
        let at = |d: usize| GLWEAutomorphismKeyLayout {
            dnum: d.into(),
            ..layout
        };
        let max_input_k = layout.gglwe_layout().max_input_k();
        let digits: usize = (1..dnum).rev().find(|&d| at(d).k() <= max_input_k).unwrap();
        let view_layout = at(digits);

        let mut scratch: ScratchOwned<BE> = ScratchOwned::alloc(
            module
                .glwe_automorphism_key_encrypt_sk_tmp_bytes(&layout)
                .max(module.glwe_automorphism_key_automorphism_tmp_bytes(&view_layout, &view_layout, &layout))
                .max(module.gglwe_noise_tmp_bytes(&view_layout)),
        );
        let mut source_xs: Source = Source::new([4u8; 32]);
        let mut source_xe: Source = Source::new([5u8; 32]);
        let mut source_xa: Source = Source::new([6u8; 32]);
        let mut sk: GLWESecret<BE::OwnedBuf, BE::ZnxWord> = module.glwe_secret_alloc(rank.into());
        module.glwe_secret_fill_ternary_prob(&mut sk, 0.5, &mut source_xs);

        let mut key_in: GLWEAutomorphismKey<BE::OwnedBuf, BE::ZnxWord> = module.glwe_automorphism_key_alloc_from_infos(&layout);
        let mut key_apply: GLWEAutomorphismKey<BE::OwnedBuf, BE::ZnxWord> =
            module.glwe_automorphism_key_alloc_from_infos(&layout);
        for (key, p) in [(&mut key_in, p0), (&mut key_apply, p1)] {
            module.glwe_automorphism_key_encrypt_sk(
                key,
                p,
                &sk,
                &mut source_xe,
                &mut source_xa,
                &mut crate::test_suite::noise::scratch_host_arena(&mut scratch),
            );
        }
        let mut key_apply_prepared: GLWEAutomorphismKeyPrepared<BE::OwnedBuf, BE> =
            module.glwe_automorphism_key_prepared_alloc_from_infos(&layout);
        module.glwe_automorphism_key_prepare(&mut key_apply_prepared, &key_apply, &mut scratch.borrow());

        let mut key_out: GLWEAutomorphismKey<BE::OwnedBuf, BE::ZnxWord> =
            module.glwe_automorphism_key_alloc_from_infos(&view_layout);
        module.glwe_automorphism_key_automorphism(
            &mut key_out,
            &key_in.view_with_digits(Dnum(digits as u32)).unwrap(),
            &key_apply_prepared.to_backend_ref(),
            &mut scratch.borrow(),
        );
        assert_eq!(key_out.p(), (p0 * p1) % module.cyclotomic_order());

        let mut sk_auto: GLWESecret<BE::OwnedBuf, BE::ZnxWord> = module.glwe_secret_alloc(rank.into());
        module.glwe_secret_fill_zero(&mut sk_auto);
        let sk_backend = ScalarZnx::from_data(BE::from_host_bytes(sk.data().data.as_ref()), sk.data().n(), sk.data().cols());
        let mut sk_auto_backend = ScalarZnx::from_data(
            BE::from_host_bytes(sk_auto.data().data.as_ref()),
            sk_auto.data().n(),
            sk_auto.data().cols(),
        );
        {
            let sk_backend_as_vec = crate::test_suite::noise::scalar_znx_as_vec_znx_backend_ref::<BE>(&sk_backend);
            let mut sk_auto_backend_as_vec =
                crate::test_suite::noise::scalar_znx_as_vec_znx_backend_mut::<BE>(&mut sk_auto_backend);
            for i in 0..rank {
                module.vec_znx_automorphism(
                    module.galois_element_inv(p0 * p1),
                    &mut sk_auto_backend_as_vec,
                    i,
                    &sk_backend_as_vec,
                    i,
                );
            }
        }
        BE::copy_to_host(&sk_auto_backend.data, sk_auto.data_mut().data.as_mut());
        let mut sk_auto_prepared: GLWESecretPrepared<BE::OwnedBuf, BE> = module.glwe_secret_prepared_alloc(rank.into());
        module.glwe_secret_prepare(&mut sk_auto_prepared, &sk_auto);

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
                    .gglwe_noise(
                        &key_out,
                        row,
                        col,
                        &sk.data().to_ref(),
                        &sk_auto_prepared,
                        &mut scratch.borrow(),
                    )
                    .std()
                    .log2();
                assert!(
                    noise_have <= max_noise,
                    "rank={rank}: noise_have: {noise_have} > max_noise: {max_noise}"
                );
            }
        }
    }
}

#[allow(clippy::too_many_arguments)]
pub fn test_gglwe_automorphism_key_automorphism_assign<
    BE: crate::test_suite::noise::TestBackend + Backend<OwnedBuf = AlignedBuf>,
>(
    params: &TestParams,
    module: &Module<BE>,
) where
    BE::OwnedBuf: poulpy_hal::layouts::HostDataMut,
    for<'a> BE::BufRef<'a>: poulpy_hal::layouts::HostDataRef,
    for<'a> BE::BufMut<'a>: poulpy_hal::layouts::HostDataMut,
    Module<BE>: GLWEAutomorphismKeyEncryptSk<BE>
        + GLWEAutomorphismKeyPreparedFactory<BE>
        + GLWEAutomorphismKeyAutomorphism<BE>
        + VecZnxAutomorphism<BE>
        + GaloisElement
        + GLWESecretPreparedFactory<BE>
        + GGLWENoise<BE>,
    ScratchOwned<BE>: ScratchOwnedAlloc<BE> + ScratchOwnedBorrow<BE>,
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
            let n: usize = module.n();
            let dsize_in: usize = 1;

            let dnum_in: usize = k_out / out_base2k;

            let auto_key_layout = GLWEAutomorphismKeyLayout {
                n: n.into(),
                base2k: out_base2k.into(),
                dnum: dnum_in.into(),
                k_aux: (dsize_in * out_base2k + module.log_n()).into(),
                k_digit: crate::layouts::k_digit_of(out_base2k, dsize_in),
                rank: rank.into(),
            };

            // Decomposes the rows it switches whole, their guard included.
            let dnum_ksk: usize = auto_key_layout.k().as_usize().div_ceil(key_base2k * dsize);

            let auto_key_apply_layout = GLWEAutomorphismKeyLayout {
                n: n.into(),
                base2k: key_base2k.into(),
                dnum: dnum_ksk.into(),
                k_aux: (dsize * key_base2k + module.log_n()).into(),
                k_digit: crate::layouts::k_digit_of(key_base2k, dsize),
                rank: rank.into(),
            };

            let mut auto_key: GLWEAutomorphismKey<BE::OwnedBuf, BE::ZnxWord> =
                module.glwe_automorphism_key_alloc_from_infos(&auto_key_layout);
            let mut auto_key_apply: GLWEAutomorphismKey<BE::OwnedBuf, BE::ZnxWord> =
                module.glwe_automorphism_key_alloc_from_infos(&auto_key_apply_layout);

            let mut source_xs: Source = Source::new([0u8; 32]);
            let mut source_xe: Source = Source::new([0u8; 32]);
            let mut source_xa: Source = Source::new([0u8; 32]);

            let mut scratch: ScratchOwned<BE> = ScratchOwned::alloc(
                (module).glwe_automorphism_key_encrypt_sk_tmp_bytes(&auto_key)
                    | (module).glwe_automorphism_key_encrypt_sk_tmp_bytes(&auto_key_apply)
                    | module.glwe_automorphism_key_automorphism_tmp_bytes(&auto_key, &auto_key, &auto_key_apply),
            );

            let mut sk: GLWESecret<BE::OwnedBuf, BE::ZnxWord> = module.glwe_secret_alloc_from_infos(&auto_key);
            module.glwe_secret_fill_ternary_prob(&mut sk, 0.5, &mut source_xs);

            // gglwe_{s1}(s0) = s0 -> s1
            module.glwe_automorphism_key_encrypt_sk(
                &mut auto_key,
                p0,
                &sk,
                &mut source_xe,
                &mut source_xa,
                &mut crate::test_suite::noise::scratch_host_arena(&mut scratch),
            );

            // gglwe_{s2}(s1) -> s1 -> s2
            module.glwe_automorphism_key_encrypt_sk(
                &mut auto_key_apply,
                p1,
                &sk,
                &mut source_xe,
                &mut source_xa,
                &mut crate::test_suite::noise::scratch_host_arena(&mut scratch),
            );

            let mut auto_key_apply_prepared: GLWEAutomorphismKeyPrepared<BE::OwnedBuf, BE> =
                module.glwe_automorphism_key_prepared_alloc_from_infos(&auto_key_apply_layout);

            module.glwe_automorphism_key_prepare(&mut auto_key_apply_prepared, &auto_key_apply, &mut scratch.borrow());

            // gglwe_{s1}(s0) (x) gglwe_{s2}(s1) = gglwe_{s2}(s0)
            module.glwe_automorphism_key_automorphism_assign(
                &mut auto_key,
                &auto_key_apply_prepared.to_backend_ref(),
                &mut scratch.borrow(),
            );

            let mut sk_auto: GLWESecret<BE::OwnedBuf, BE::ZnxWord> = module.glwe_secret_alloc_from_infos(&auto_key);
            module.glwe_secret_fill_zero(&mut sk_auto); // Necessary to avoid panic of unfilled sk

            let sk_backend = ScalarZnx::from_data(BE::from_host_bytes(sk.data().data.as_ref()), sk.data().n(), sk.data().cols());
            let mut sk_auto_backend = ScalarZnx::from_data(
                BE::from_host_bytes(sk_auto.data().data.as_ref()),
                sk_auto.data().n(),
                sk_auto.data().cols(),
            );
            {
                let sk_backend_as_vec = crate::test_suite::noise::scalar_znx_as_vec_znx_backend_ref::<BE>(&sk_backend);
                let mut sk_auto_backend_as_vec =
                    crate::test_suite::noise::scalar_znx_as_vec_znx_backend_mut::<BE>(&mut sk_auto_backend);
                for i in 0..rank {
                    module.vec_znx_automorphism(
                        module.galois_element_inv(p0 * p1),
                        &mut sk_auto_backend_as_vec,
                        i,
                        &sk_backend_as_vec,
                        i,
                    );
                }
            }
            BE::copy_to_host(&sk_auto_backend.data, sk_auto.data_mut().data.as_mut());

            let mut sk_auto_dft: GLWESecretPrepared<BE::OwnedBuf, BE> = module.glwe_secret_prepared_alloc_from_infos(&sk_auto);
            module.glwe_secret_prepare(&mut sk_auto_dft, &sk_auto);

            let max_noise: f64 = auto_key_apply_layout.log2_std_noise_keyswitch(
                &auto_key_layout,
                &auto_key_layout,
                0.5,
                0.5,
                DEFAULT_SIGMA_XE * DEFAULT_SIGMA_XE,
                DEFAULT_SIGMA_XE * DEFAULT_SIGMA_XE,
                0f64,
            ) + 0.5;

            for row in 0..auto_key.dnum().as_usize() {
                for col in 0..auto_key.rank().as_usize() {
                    let noise_have = auto_key
                        .key
                        .noise_stats(module, row, col, &sk.data().to_ref(), &sk_auto_dft, &mut scratch.borrow())
                        .std()
                        .log2();

                    assert!(noise_have <= max_noise, "noise_have: {noise_have} > max_noise: {max_noise}");
                }
            }
        }
    }
}
