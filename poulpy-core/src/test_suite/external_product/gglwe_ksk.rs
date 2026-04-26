use poulpy_hal::{
    api::{ScratchAvailable, ScratchOwnedAlloc, ScratchOwnedBorrow, VecZnxRotateAssign},
    layouts::{DeviceBuf, Module, ScalarZnx, ScalarZnxToMut, Scratch, ScratchOwned, ZnxViewMut},
    source::Source,
    test_suite::TestParams,
};

use crate::{
    EncryptionLayout, GGLWEExternalProduct, GGLWENoise, GGSWEncryptSk, GLWESwitchingKeyEncryptSk, ScratchTakeCore,
    encryption::DEFAULT_SIGMA_XE,
    layouts::{
        GGLWEInfos, GGSW, GGSWLayout, GGSWPreparedFactory, GLWESecret, GLWESecretPreparedFactory, GLWESwitchingKey,
        GLWESwitchingKeyLayout,
        prepared::{GGSWPrepared, GLWESecretPrepared},
    },
    noise::noise_ggsw_product,
};

#[allow(clippy::too_many_arguments)]
pub fn test_gglwe_switching_key_external_product<BE: crate::test_suite::TestBackend>(params: &TestParams, module: &Module<BE>)
where
    Module<BE>: GGLWEExternalProduct<BE>
        + GGSWEncryptSk<BE>
        + GLWESwitchingKeyEncryptSk<BE>
        + GLWESecretPreparedFactory<BE>
        + VecZnxRotateAssign<BE>
        + GGSWPreparedFactory<BE>
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

    for rank_in in 1_usize..3 {
        for rank_out in 1_usize..3 {
            for dsize in 1_usize..max_dsize + 1 {
                let k_ggsw: usize = k_in + key_base2k * dsize;
                let k_out: usize = k_in; // Better capture noise.

                let n: usize = module.n();
                let dnum_in: usize = k_in / in_base2k;
                let dnum: usize = k_in.div_ceil(key_base2k * dsize);
                let dsize_in: usize = 1;

                let gglwe_in_infos = EncryptionLayout::new_from_default_sigma(GLWESwitchingKeyLayout {
                    n: n.into(),
                    base2k: in_base2k.into(),
                    k: k_in.into(),
                    dnum: dnum_in.into(),
                    dsize: dsize_in.into(),
                    rank_in: rank_in.into(),
                    rank_out: rank_out.into(),
                })
                .unwrap();

                let gglwe_out_infos: GLWESwitchingKeyLayout = GLWESwitchingKeyLayout {
                    n: n.into(),
                    base2k: out_base2k.into(),
                    k: k_out.into(),
                    dnum: dnum_in.into(),
                    dsize: dsize_in.into(),
                    rank_in: rank_in.into(),
                    rank_out: rank_out.into(),
                };

                let ggsw_infos = EncryptionLayout::new_from_default_sigma(GGSWLayout {
                    n: n.into(),
                    base2k: key_base2k.into(),
                    k: k_ggsw.into(),
                    dnum: dnum.into(),
                    dsize: dsize.into(),
                    rank: rank_out.into(),
                })
                .unwrap();

                let mut ct_gglwe_in: GLWESwitchingKey<Vec<u8>> = GLWESwitchingKey::alloc_from_infos(&gglwe_in_infos);
                let mut ct_gglwe_out: GLWESwitchingKey<Vec<u8>> = GLWESwitchingKey::alloc_from_infos(&gglwe_out_infos);
                let mut ct_rgsw: GGSW<Vec<u8>> = GGSW::alloc_from_infos(&ggsw_infos);

                let mut pt_rgsw: ScalarZnx<Vec<u8>> = ScalarZnx::alloc(n, 1);

                let mut source_xs: Source = Source::new([0u8; 32]);
                let mut source_xe: Source = Source::new([0u8; 32]);
                let mut source_xa: Source = Source::new([0u8; 32]);

                let mut scratch: ScratchOwned<BE> = ScratchOwned::alloc(
                    (module).glwe_switching_key_encrypt_sk_tmp_bytes(&gglwe_in_infos)
                        | module.gglwe_external_product_tmp_bytes(&gglwe_out_infos, &gglwe_in_infos, &ggsw_infos)
                        | (module).ggsw_encrypt_sk_tmp_bytes(&ggsw_infos),
                );

                let r: usize = 1;

                pt_rgsw.to_mut().raw_mut()[r] = 1; // X^{r}

                let var_xs: f64 = 0.5;

                let mut sk_in: GLWESecret<Vec<u8>> = GLWESecret::alloc(n.into(), rank_in.into());
                sk_in.fill_ternary_prob(var_xs, &mut source_xs);

                let mut sk_out: GLWESecret<Vec<u8>> = GLWESecret::alloc(n.into(), rank_out.into());
                sk_out.fill_ternary_prob(var_xs, &mut source_xs);

                let mut sk_out_prepared: GLWESecretPrepared<DeviceBuf<BE>, BE> =
                    module.glwe_secret_prepared_alloc(rank_out.into());
                module.glwe_secret_prepare(&mut sk_out_prepared, &sk_out);

                // gglwe_{s1}(s0) = s0 -> s1
                module.glwe_switching_key_encrypt_sk(
                    &mut ct_gglwe_in,
                    &sk_in,
                    &sk_out,
                    &gglwe_in_infos,
                    &mut source_xe,
                    &mut source_xa,
                    scratch.borrow(),
                );

                module.ggsw_encrypt_sk(
                    &mut ct_rgsw,
                    &pt_rgsw,
                    &sk_out_prepared,
                    &ggsw_infos,
                    &mut source_xe,
                    &mut source_xa,
                    scratch.borrow(),
                );

                let mut ct_rgsw_prepared: GGSWPrepared<DeviceBuf<BE>, BE> = module.ggsw_prepared_alloc_from_infos(&ct_rgsw);
                module.ggsw_prepare(&mut ct_rgsw_prepared, &ct_rgsw, scratch.borrow());

                // gglwe_(m) (x) RGSW_(X^k) = gglwe_(m * X^k)
                module.gglwe_external_product(&mut ct_gglwe_out, &ct_gglwe_in, &ct_rgsw_prepared, scratch.borrow());

                (0..rank_in).for_each(|i| {
                    module.vec_znx_rotate_assign(r as i64, &mut sk_in.data.as_vec_znx_mut(), i, scratch.borrow()); // * X^{r}
                });

                let var_gct_err_lhs: f64 = DEFAULT_SIGMA_XE * DEFAULT_SIGMA_XE;
                let var_gct_err_rhs: f64 = 0f64;

                let var_msg: f64 = 1f64 / n as f64; // X^{k}
                let var_a0_err: f64 = DEFAULT_SIGMA_XE * DEFAULT_SIGMA_XE;
                let var_a1_err: f64 = 1f64 / 12f64;

                let max_noise: f64 = noise_ggsw_product(
                    n as f64,
                    key_base2k * dsize,
                    var_xs,
                    var_msg,
                    var_a0_err,
                    var_a1_err,
                    var_gct_err_lhs,
                    var_gct_err_rhs,
                    rank_out as f64,
                    k_in,
                    k_ggsw,
                ) + 0.5;

                for row in 0..ct_gglwe_out.dnum().as_usize() {
                    for col in 0..ct_gglwe_out.rank_in().as_usize() {
                        let noise_have: f64 = ct_gglwe_out
                            .key
                            .noise(module, row, col, &sk_in.data, &sk_out_prepared, scratch.borrow())
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
pub fn test_gglwe_switching_key_external_product_assign<BE: crate::test_suite::TestBackend>(
    params: &TestParams,
    module: &Module<BE>,
) where
    Module<BE>: GGLWEExternalProduct<BE>
        + GGSWEncryptSk<BE>
        + GLWESwitchingKeyEncryptSk<BE>
        + GLWESecretPreparedFactory<BE>
        + VecZnxRotateAssign<BE>
        + GGSWPreparedFactory<BE>
        + GGLWENoise<BE>,
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
                let k_ggsw: usize = k_out + key_base2k * dsize;

                let n: usize = module.n();
                let dnum_in: usize = k_out / out_base2k;
                let dnum: usize = k_out.div_ceil(key_base2k * dsize);

                let dsize_in: usize = 1;

                let gglwe_out_infos = EncryptionLayout::new_from_default_sigma(GLWESwitchingKeyLayout {
                    n: n.into(),
                    base2k: out_base2k.into(),
                    k: k_out.into(),
                    dnum: dnum_in.into(),
                    dsize: dsize_in.into(),
                    rank_in: rank_in.into(),
                    rank_out: rank_out.into(),
                })
                .unwrap();

                let ggsw_infos = EncryptionLayout::new_from_default_sigma(GGSWLayout {
                    n: n.into(),
                    base2k: key_base2k.into(),
                    k: k_ggsw.into(),
                    dnum: dnum.into(),
                    dsize: dsize.into(),
                    rank: rank_out.into(),
                })
                .unwrap();

                let mut ct_gglwe: GLWESwitchingKey<Vec<u8>> = GLWESwitchingKey::alloc_from_infos(&gglwe_out_infos);
                let mut ct_rgsw: GGSW<Vec<u8>> = GGSW::alloc_from_infos(&ggsw_infos);

                let mut pt_rgsw: ScalarZnx<Vec<u8>> = ScalarZnx::alloc(n, 1);

                let mut source_xs: Source = Source::new([0u8; 32]);
                let mut source_xe: Source = Source::new([0u8; 32]);
                let mut source_xa: Source = Source::new([0u8; 32]);

                let mut scratch: ScratchOwned<BE> = ScratchOwned::alloc(
                    (module).glwe_switching_key_encrypt_sk_tmp_bytes(&gglwe_out_infos)
                        | module.gglwe_external_product_tmp_bytes(&gglwe_out_infos, &gglwe_out_infos, &ggsw_infos)
                        | (module).ggsw_encrypt_sk_tmp_bytes(&ggsw_infos),
                );

                let r: usize = 1;

                pt_rgsw.to_mut().raw_mut()[r] = 1; // X^{r}

                let var_xs: f64 = 0.5;

                let mut sk_in: GLWESecret<Vec<u8>> = GLWESecret::alloc(n.into(), rank_in.into());
                sk_in.fill_ternary_prob(var_xs, &mut source_xs);

                let mut sk_out: GLWESecret<Vec<u8>> = GLWESecret::alloc(n.into(), rank_out.into());
                sk_out.fill_ternary_prob(var_xs, &mut source_xs);

                let mut sk_out_prepared: GLWESecretPrepared<DeviceBuf<BE>, BE> =
                    module.glwe_secret_prepared_alloc(rank_out.into());
                module.glwe_secret_prepare(&mut sk_out_prepared, &sk_out);

                // gglwe_{s1}(s0) = s0 -> s1
                module.glwe_switching_key_encrypt_sk(
                    &mut ct_gglwe,
                    &sk_in,
                    &sk_out,
                    &gglwe_out_infos,
                    &mut source_xe,
                    &mut source_xa,
                    scratch.borrow(),
                );

                module.ggsw_encrypt_sk(
                    &mut ct_rgsw,
                    &pt_rgsw,
                    &sk_out_prepared,
                    &ggsw_infos,
                    &mut source_xe,
                    &mut source_xa,
                    scratch.borrow(),
                );

                let mut ct_rgsw_prepared: GGSWPrepared<DeviceBuf<BE>, BE> = module.ggsw_prepared_alloc_from_infos(&ct_rgsw);
                module.ggsw_prepare(&mut ct_rgsw_prepared, &ct_rgsw, scratch.borrow());

                // gglwe_(m) (x) RGSW_(X^k) = gglwe_(m * X^k)
                module.gglwe_external_product_assign(&mut ct_gglwe, &ct_rgsw_prepared, scratch.borrow());

                (0..rank_in).for_each(|i| {
                    module.vec_znx_rotate_assign(r as i64, &mut sk_in.data.as_vec_znx_mut(), i, scratch.borrow()); // * X^{r}
                });

                let var_gct_err_lhs: f64 = DEFAULT_SIGMA_XE * DEFAULT_SIGMA_XE;
                let var_gct_err_rhs: f64 = 0f64;

                let var_msg: f64 = 1f64 / n as f64; // X^{k}
                let var_a0_err: f64 = DEFAULT_SIGMA_XE * DEFAULT_SIGMA_XE;
                let var_a1_err: f64 = 1f64 / 12f64;

                let max_noise: f64 = noise_ggsw_product(
                    n as f64,
                    key_base2k * dsize,
                    var_xs,
                    var_msg,
                    var_a0_err,
                    var_a1_err,
                    var_gct_err_lhs,
                    var_gct_err_rhs,
                    rank_out as f64,
                    k_out,
                    k_ggsw,
                ) + 0.5;

                for row in 0..ct_gglwe.dnum().as_usize() {
                    for col in 0..ct_gglwe.rank_in().as_usize() {
                        let noise_have: f64 = ct_gglwe
                            .key
                            .noise(module, row, col, &sk_in.data, &sk_out_prepared, scratch.borrow())
                            .std()
                            .log2();
                        assert!(noise_have <= max_noise, "noise_have:{noise_have} > noise_max:{max_noise}")
                    }
                }
            }
        }
    }
}
