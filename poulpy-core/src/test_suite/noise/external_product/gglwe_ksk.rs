use poulpy_hal::{
    api::{ScratchOwnedAlloc, ScratchOwnedBorrow, VecZnxRotateAssign},
    layouts::{Module, ScalarZnx, ScratchOwned, ZnxViewMut},
    source::Source,
    test_suite::TestParams,
};

use crate::layouts::GLWESecretSampling;
use crate::layouts::prepared::GGSWPreparedToBackendRef;
use crate::{
    GGLWEExternalProduct, GGLWENoise, GGSWEncryptSk, GLWENormalize, GLWESwitchingKeyEncryptSk,
    encryption::DEFAULT_SIGMA_XE,
    layouts::{
        Dnum, GGLWEInfos, GGLWEViewWithDigits, GGSW, GGSWLayout, GGSWPreparedFactory, GLWESecret, GLWESecretPreparedFactory,
        GLWESwitchingKey, GLWESwitchingKeyLayout, ModuleCoreAlloc,
        prepared::{GGSWPrepared, GLWESecretPrepared},
    },
    noise::GGSWNoiseModel,
};

#[allow(clippy::too_many_arguments)]
pub fn test_gglwe_switching_key_external_product<BE: crate::test_suite::noise::TestBackend>(
    params: &TestParams,
    module: &Module<BE>,
) where
    BE::OwnedBuf: poulpy_hal::layouts::HostDataMut,
    for<'a> BE::BufRef<'a>: poulpy_hal::layouts::HostDataRef,
    for<'a> BE::BufMut<'a>: poulpy_hal::layouts::HostDataMut,
    Module<BE>: GGLWEExternalProduct<BE>
        + GGSWEncryptSk<BE>
        + GLWESwitchingKeyEncryptSk<BE>
        + GLWESecretPreparedFactory<BE>
        + VecZnxRotateAssign<BE>
        + GGSWPreparedFactory<BE>
        + GGLWENoise<BE>,
    ScratchOwned<BE>: ScratchOwnedAlloc<BE> + ScratchOwnedBorrow<BE>,
{
    let base2k: usize = params.base2k;
    let in_base2k: usize = base2k - 1;
    let key_base2k: usize = base2k;
    let out_base2k: usize = in_base2k; // MUST BE SAME
    let k_in: usize = 4 * in_base2k + 1;
    let max_dsize: usize = k_in.div_ceil(key_base2k);

    for rank_in in 1_usize..3 {
        for rank_out in 1_usize..3 {
            for dsize in 1_usize..max_dsize + 1 {
                let n: usize = module.n();
                let dnum_in: usize = k_in / in_base2k;
                let dnum: usize = k_in.div_ceil(key_base2k * dsize);
                let dsize_in: usize = 1;

                let gglwe_in_infos = GLWESwitchingKeyLayout {
                    n: n.into(),
                    base2k: in_base2k.into(),
                    dnum: dnum_in.into(),
                    k_aux: (dsize_in * in_base2k + module.log_n()).into(),
                    k_digit: crate::layouts::k_digit_of(in_base2k, dsize_in),
                    k_offset: crate::layouts::TorusPrecision(0),
                    rank_in: rank_in.into(),
                    rank_out: rank_out.into(),
                };

                let gglwe_out_infos: GLWESwitchingKeyLayout = GLWESwitchingKeyLayout {
                    n: n.into(),
                    base2k: out_base2k.into(),
                    dnum: dnum_in.into(),
                    k_aux: (dsize_in * out_base2k + module.log_n()).into(),
                    k_digit: crate::layouts::k_digit_of(out_base2k, dsize_in),
                    k_offset: crate::layouts::TorusPrecision(0),
                    rank_in: rank_in.into(),
                    rank_out: rank_out.into(),
                };

                let ggsw_infos = GGSWLayout {
                    n: n.into(),
                    base2k: key_base2k.into(),
                    dnum: dnum.into(),
                    k_aux: (dsize * key_base2k + module.log_n()).into(),
                    k_digit: crate::layouts::k_digit_of(key_base2k, dsize),
                    rank: rank_out.into(),
                };

                let mut ct_gglwe_in: GLWESwitchingKey<BE::OwnedBuf, BE::ZnxWord> =
                    module.glwe_switching_key_alloc_from_infos(&gglwe_in_infos);
                let mut ct_gglwe_out: GLWESwitchingKey<BE::OwnedBuf, BE::ZnxWord> =
                    module.glwe_switching_key_alloc_from_infos(&gglwe_out_infos);
                let mut ct_rgsw: GGSW<BE::OwnedBuf, BE::ZnxWord> = module.ggsw_alloc_from_infos(&ggsw_infos);

                let mut pt_rgsw: ScalarZnx<BE::OwnedBuf, BE::ZnxWord> = module.scalar_znx_alloc(module.n(), 1);

                let mut source_xs: Source = Source::new([0u8; 32]);
                let mut source_xe: Source = Source::new([0u8; 32]);
                let mut source_xa: Source = Source::new([0u8; 32]);

                let mut scratch: ScratchOwned<BE> = ScratchOwned::alloc(
                    (module).glwe_switching_key_encrypt_sk_tmp_bytes(&gglwe_in_infos)
                        | module.gglwe_external_product_tmp_bytes(&gglwe_out_infos, &gglwe_in_infos, &ggsw_infos)
                        | (module).ggsw_encrypt_sk_tmp_bytes(&ggsw_infos),
                );

                let r: usize = 1;

                pt_rgsw.raw_mut()[r] = 1; // X^{r}

                let var_xs: f64 = 0.5;

                let mut sk_in: GLWESecret<BE::OwnedBuf, BE::ZnxWord> = module.glwe_secret_alloc(rank_in.into());
                module.glwe_secret_fill_ternary_prob(&mut sk_in, var_xs, &mut source_xs);

                let mut sk_out: GLWESecret<BE::OwnedBuf, BE::ZnxWord> = module.glwe_secret_alloc(rank_out.into());
                module.glwe_secret_fill_ternary_prob(&mut sk_out, var_xs, &mut source_xs);

                let mut sk_out_prepared: GLWESecretPrepared<BE::OwnedBuf, BE> =
                    module.glwe_secret_prepared_alloc(rank_out.into());
                module.glwe_secret_prepare(&mut sk_out_prepared, &sk_out);

                // gglwe_{s1}(s0) = s0 -> s1
                module.glwe_switching_key_encrypt_sk(
                    &mut ct_gglwe_in,
                    &sk_in,
                    &sk_out,
                    &mut source_xe,
                    &mut source_xa,
                    &mut scratch.arena(),
                );

                module.ggsw_encrypt_sk(
                    &mut ct_rgsw,
                    &pt_rgsw,
                    &sk_out_prepared,
                    &mut source_xe,
                    &mut source_xa,
                    &mut scratch.borrow(),
                );

                let mut ct_rgsw_prepared: GGSWPrepared<BE::OwnedBuf, BE> = module.ggsw_prepared_alloc_from_infos(&ct_rgsw);
                module.ggsw_prepare(&mut ct_rgsw_prepared, &ct_rgsw, &mut scratch.borrow());

                // gglwe_(m) (x) RGSW_(X^k) = gglwe_(m * X^k)
                module.gglwe_external_product(
                    &mut ct_gglwe_out,
                    &ct_gglwe_in,
                    &ct_rgsw_prepared.to_backend_ref(),
                    &mut scratch.borrow(),
                );

                {
                    let mut sk_in_as_vec = crate::test_suite::noise::scalar_znx_as_vec_znx_backend_mut::<BE>(&mut sk_in.data);
                    (0..rank_in).for_each(|i| {
                        module.vec_znx_rotate_assign(r as i64, &mut sk_in_as_vec, i, &mut scratch.borrow()); // * X^{r}
                    });
                }

                let var_key_err_body: f64 = DEFAULT_SIGMA_XE * DEFAULT_SIGMA_XE;
                let var_key_err_mask: f64 = 0f64;

                let var_msg: f64 = 1f64 / n as f64; // X^{k}
                let var_a0_err: f64 = DEFAULT_SIGMA_XE * DEFAULT_SIGMA_XE;
                let var_a1_err: f64 = 0f64;

                let max_noise: f64 = ggsw_infos.log2_std_noise_external_product(
                    &gglwe_out_infos,
                    &gglwe_in_infos,
                    var_xs,
                    var_msg,
                    var_a0_err,
                    var_a1_err,
                    var_key_err_body,
                    var_key_err_mask,
                ) + 0.5;

                for row in 0..ct_gglwe_out.dnum().as_usize() {
                    for col in 0..ct_gglwe_out.rank_in().as_usize() {
                        let noise_have: f64 = ct_gglwe_out
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
                        assert!(noise_have <= max_noise, "noise_have:{noise_have} > noise_max:{max_noise}")
                    }
                }
            }
        }
    }
}

#[allow(clippy::too_many_arguments)]
pub fn test_gglwe_switching_key_external_product_assign<BE: crate::test_suite::noise::TestBackend>(
    params: &TestParams,
    module: &Module<BE>,
) where
    BE::OwnedBuf: poulpy_hal::layouts::HostDataMut,
    for<'a> BE::BufRef<'a>: poulpy_hal::layouts::HostDataRef,
    for<'a> BE::BufMut<'a>: poulpy_hal::layouts::HostDataMut,
    Module<BE>: GGLWEExternalProduct<BE>
        + GGSWEncryptSk<BE>
        + GLWESwitchingKeyEncryptSk<BE>
        + GLWESecretPreparedFactory<BE>
        + VecZnxRotateAssign<BE>
        + GGSWPreparedFactory<BE>
        + GGLWENoise<BE>,
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
                let dnum_in: usize = k_out / out_base2k;
                let dnum: usize = k_out.div_ceil(key_base2k * dsize);

                let dsize_in: usize = 1;

                let gglwe_out_infos = GLWESwitchingKeyLayout {
                    n: n.into(),
                    base2k: out_base2k.into(),
                    dnum: dnum_in.into(),
                    k_aux: (dsize_in * out_base2k + module.log_n()).into(),
                    k_digit: crate::layouts::k_digit_of(out_base2k, dsize_in),
                    k_offset: crate::layouts::TorusPrecision(0),
                    rank_in: rank_in.into(),
                    rank_out: rank_out.into(),
                };

                let ggsw_infos = GGSWLayout {
                    n: n.into(),
                    base2k: key_base2k.into(),
                    dnum: dnum.into(),
                    k_aux: (dsize * key_base2k + module.log_n()).into(),
                    k_digit: crate::layouts::k_digit_of(key_base2k, dsize),
                    rank: rank_out.into(),
                };

                let mut ct_gglwe: GLWESwitchingKey<BE::OwnedBuf, BE::ZnxWord> =
                    module.glwe_switching_key_alloc_from_infos(&gglwe_out_infos);
                let mut ct_rgsw: GGSW<BE::OwnedBuf, BE::ZnxWord> = module.ggsw_alloc_from_infos(&ggsw_infos);

                let mut pt_rgsw: ScalarZnx<BE::OwnedBuf, BE::ZnxWord> = module.scalar_znx_alloc(module.n(), 1);

                let mut source_xs: Source = Source::new([0u8; 32]);
                let mut source_xe: Source = Source::new([0u8; 32]);
                let mut source_xa: Source = Source::new([0u8; 32]);

                let mut scratch: ScratchOwned<BE> = ScratchOwned::alloc(
                    (module).glwe_switching_key_encrypt_sk_tmp_bytes(&gglwe_out_infos)
                        | module.gglwe_external_product_tmp_bytes(&gglwe_out_infos, &gglwe_out_infos, &ggsw_infos)
                        | (module).ggsw_encrypt_sk_tmp_bytes(&ggsw_infos),
                );

                let r: usize = 1;

                pt_rgsw.raw_mut()[r] = 1; // X^{r}

                let var_xs: f64 = 0.5;

                let mut sk_in: GLWESecret<BE::OwnedBuf, BE::ZnxWord> = module.glwe_secret_alloc(rank_in.into());
                module.glwe_secret_fill_ternary_prob(&mut sk_in, var_xs, &mut source_xs);

                let mut sk_out: GLWESecret<BE::OwnedBuf, BE::ZnxWord> = module.glwe_secret_alloc(rank_out.into());
                module.glwe_secret_fill_ternary_prob(&mut sk_out, var_xs, &mut source_xs);

                let mut sk_out_prepared: GLWESecretPrepared<BE::OwnedBuf, BE> =
                    module.glwe_secret_prepared_alloc(rank_out.into());
                module.glwe_secret_prepare(&mut sk_out_prepared, &sk_out);

                // gglwe_{s1}(s0) = s0 -> s1
                module.glwe_switching_key_encrypt_sk(
                    &mut ct_gglwe,
                    &sk_in,
                    &sk_out,
                    &mut source_xe,
                    &mut source_xa,
                    &mut scratch.arena(),
                );

                module.ggsw_encrypt_sk(
                    &mut ct_rgsw,
                    &pt_rgsw,
                    &sk_out_prepared,
                    &mut source_xe,
                    &mut source_xa,
                    &mut scratch.borrow(),
                );

                let mut ct_rgsw_prepared: GGSWPrepared<BE::OwnedBuf, BE> = module.ggsw_prepared_alloc_from_infos(&ct_rgsw);
                module.ggsw_prepare(&mut ct_rgsw_prepared, &ct_rgsw, &mut scratch.borrow());

                // gglwe_(m) (x) RGSW_(X^k) = gglwe_(m * X^k)
                module.gglwe_external_product_assign(&mut ct_gglwe, &ct_rgsw_prepared.to_backend_ref(), &mut scratch.borrow());

                {
                    let mut sk_in_as_vec = crate::test_suite::noise::scalar_znx_as_vec_znx_backend_mut::<BE>(&mut sk_in.data);
                    (0..rank_in).for_each(|i| {
                        module.vec_znx_rotate_assign(r as i64, &mut sk_in_as_vec, i, &mut scratch.borrow()); // * X^{r}
                    });
                }

                let var_key_err_body: f64 = DEFAULT_SIGMA_XE * DEFAULT_SIGMA_XE;
                let var_key_err_mask: f64 = 0f64;

                let var_msg: f64 = 1f64 / n as f64; // X^{k}
                let var_a0_err: f64 = DEFAULT_SIGMA_XE * DEFAULT_SIGMA_XE;
                let var_a1_err: f64 = 0f64;

                let max_noise: f64 = ggsw_infos.log2_std_noise_external_product(
                    &gglwe_out_infos,
                    &gglwe_out_infos,
                    var_xs,
                    var_msg,
                    var_a0_err,
                    var_a1_err,
                    var_key_err_body,
                    var_key_err_mask,
                ) + 0.5;

                for row in 0..ct_gglwe.dnum().as_usize() {
                    for col in 0..ct_gglwe.rank_in().as_usize() {
                        let noise_have: f64 = ct_gglwe
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
                        assert!(noise_have <= max_noise, "noise_have:{noise_have} > noise_max:{max_noise}")
                    }
                }
            }
        }
    }
}

/// A GGSW multiplies a view of a key's first digits, into a key of that many
/// digits, bit for bit as it multiplies the narrowed copy the view reads as.
pub fn test_gglwe_switching_key_external_product_digits<BE: crate::test_suite::noise::TestBackend>(
    params: &TestParams,
    module: &Module<BE>,
) where
    BE::OwnedBuf: poulpy_hal::layouts::HostDataMut,
    for<'a> BE::BufRef<'a>: poulpy_hal::layouts::HostDataRef,
    for<'a> BE::BufMut<'a>: poulpy_hal::layouts::HostDataMut,
    Module<BE>: GGLWEExternalProduct<BE>
        + GGSWEncryptSk<BE>
        + GLWESwitchingKeyEncryptSk<BE>
        + GLWESecretPreparedFactory<BE>
        + GGSWPreparedFactory<BE>
        + GLWENormalize<BE>,
    ScratchOwned<BE>: ScratchOwnedAlloc<BE> + ScratchOwnedBorrow<BE>,
{
    let base2k: usize = params.base2k;
    let n: usize = module.n();
    let (dnum, digits): (usize, usize) = (4, 2);
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
        let view_layout = GLWESwitchingKeyLayout {
            dnum: digits.into(),
            ..layout
        };
        let ggsw_layout = GGSWLayout {
            n: n.into(),
            base2k: base2k.into(),
            dnum: dnum.into(),
            k_aux: (base2k + module.log_n()).into(),
            k_digit: crate::layouts::k_digit_of(base2k, 1_usize),
            rank: rank.into(),
        };

        let mut scratch: ScratchOwned<BE> = ScratchOwned::alloc(
            module
                .glwe_switching_key_encrypt_sk_tmp_bytes(&layout)
                .max(module.ggsw_encrypt_sk_tmp_bytes(&ggsw_layout))
                .max(module.gglwe_external_product_tmp_bytes(&view_layout, &view_layout, &ggsw_layout))
                .max(module.glwe_normalize_tmp_bytes()),
        );
        let mut source_xs: Source = Source::new([10u8; 32]);
        let mut source_xe: Source = Source::new([11u8; 32]);
        let mut source_xa: Source = Source::new([12u8; 32]);
        let mut sk_in: GLWESecret<BE::OwnedBuf, BE::ZnxWord> = module.glwe_secret_alloc(rank.into());
        module.glwe_secret_fill_ternary_prob(&mut sk_in, 0.5, &mut source_xs);
        let mut sk_out: GLWESecret<BE::OwnedBuf, BE::ZnxWord> = module.glwe_secret_alloc(rank.into());
        module.glwe_secret_fill_ternary_prob(&mut sk_out, 0.5, &mut source_xs);
        let mut sk_out_prepared: GLWESecretPrepared<BE::OwnedBuf, BE> = module.glwe_secret_prepared_alloc(rank.into());
        module.glwe_secret_prepare(&mut sk_out_prepared, &sk_out);

        let mut key: GLWESwitchingKey<BE::OwnedBuf, BE::ZnxWord> = module.glwe_switching_key_alloc_from_infos(&layout);
        module.glwe_switching_key_encrypt_sk(
            &mut key,
            &sk_in,
            &sk_out,
            &mut source_xe,
            &mut source_xa,
            &mut scratch.arena(),
        );
        let mut pt_ggsw: ScalarZnx<BE::OwnedBuf, BE::ZnxWord> = module.scalar_znx_alloc(n, 1);
        pt_ggsw.raw_mut()[1] = 1;
        let mut ggsw: GGSW<BE::OwnedBuf, BE::ZnxWord> = module.ggsw_alloc_from_infos(&ggsw_layout);
        module.ggsw_encrypt_sk(
            &mut ggsw,
            &pt_ggsw,
            &sk_out_prepared,
            &mut source_xe,
            &mut source_xa,
            &mut scratch.borrow(),
        );
        let mut ggsw_prepared: GGSWPrepared<BE::OwnedBuf, BE> = module.ggsw_prepared_alloc_from_infos(&ggsw);
        module.ggsw_prepare(&mut ggsw_prepared, &ggsw, &mut scratch.borrow());

        let mut res: GLWESwitchingKey<BE::OwnedBuf, BE::ZnxWord> = module.glwe_switching_key_alloc_from_infos(&view_layout);
        module.gglwe_external_product(
            &mut res,
            &key.view_with_digits(Dnum(digits as u32)).unwrap(),
            &ggsw_prepared.to_backend_ref(),
            &mut scratch.borrow(),
        );

        let mut key_copy: GLWESwitchingKey<BE::OwnedBuf, BE::ZnxWord> = module.glwe_switching_key_alloc_from_infos(&view_layout);
        crate::test_suite::noise::gglwe_narrowed_copy(
            module,
            &mut key_copy,
            &key.view_with_digits(Dnum(digits as u32)).unwrap(),
            &mut scratch.borrow(),
        );
        let mut res_copy: GLWESwitchingKey<BE::OwnedBuf, BE::ZnxWord> = module.glwe_switching_key_alloc_from_infos(&view_layout);
        module.gglwe_external_product(
            &mut res_copy,
            &key_copy,
            &ggsw_prepared.to_backend_ref(),
            &mut scratch.borrow(),
        );
        assert!(
            res.key.data() == res_copy.key.data(),
            "rank={rank}: the view's product differs from the narrowed copy's"
        );
    }
}
