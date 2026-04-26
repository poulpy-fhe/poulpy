use poulpy_hal::{
    api::{ScratchAvailable, ScratchOwnedAlloc, ScratchOwnedBorrow, VecZnxFillUniform, VecZnxRotateAssign},
    layouts::{DeviceBuf, Module, ScalarZnx, Scratch, ScratchOwned, ZnxViewMut},
    source::Source,
    test_suite::TestParams,
};

use crate::{
    EncryptionLayout, GGSWEncryptSk, GLWEEncryptSk, GLWEExternalProduct, GLWENoise, GLWENormalize, ScratchTakeCore,
    encryption::DEFAULT_SIGMA_XE,
    layouts::{
        GGSW, GGSWLayout, GGSWPreparedFactory, GLWE, GLWELayout, GLWEPlaintext, GLWESecret, GLWESecretPreparedFactory,
        prepared::{GGSWPrepared, GLWESecretPrepared},
    },
    noise::noise_ggsw_product,
};

#[allow(clippy::too_many_arguments)]
pub fn test_glwe_external_product<BE: crate::test_suite::TestBackend>(params: &TestParams, module: &Module<BE>)
where
    Module<BE>: GGSWEncryptSk<BE>
        + GGSWPreparedFactory<BE>
        + VecZnxFillUniform
        + GLWEExternalProduct<BE>
        + GLWEEncryptSk<BE>
        + GLWENoise<BE>
        + VecZnxRotateAssign<BE>
        + GLWESecretPreparedFactory<BE>
        + GLWENormalize<BE>,
    ScratchOwned<BE>: ScratchOwnedAlloc<BE> + ScratchOwnedBorrow<BE>,
    Scratch<BE>: ScratchAvailable + ScratchTakeCore<BE>,
{
    let base2k: usize = params.base2k;
    let in_base2k: usize = base2k - 1;
    let key_base2k: usize = base2k;
    let out_base2k: usize = base2k - 2;
    let k_in: usize = 4 * in_base2k + 1;
    let max_dsize: usize = k_in.div_ceil(key_base2k);
    for rank in 1_usize..3 {
        for dsize in 1..max_dsize + 1 {
            let k_ggsw: usize = k_in + key_base2k * dsize;
            let k_out: usize = k_ggsw; // Better capture noise

            let n: usize = module.n();
            let dnum: usize = k_in.div_ceil(k_ggsw * dsize);

            let glwe_in_infos = EncryptionLayout::new_from_default_sigma(GLWELayout {
                n: n.into(),
                base2k: in_base2k.into(),
                k: k_in.into(),
                rank: rank.into(),
            })
            .unwrap();

            let glwe_out_infos: GLWELayout = GLWELayout {
                n: n.into(),
                base2k: out_base2k.into(),
                k: k_out.into(),
                rank: rank.into(),
            };

            let ggsw_apply_infos = EncryptionLayout::new_from_default_sigma(GGSWLayout {
                n: n.into(),
                base2k: key_base2k.into(),
                k: k_ggsw.into(),
                dnum: dnum.into(),
                dsize: dsize.into(),
                rank: rank.into(),
            })
            .unwrap();

            let mut ggsw_apply: GGSW<Vec<u8>> = GGSW::alloc_from_infos(&ggsw_apply_infos);
            let mut glwe_in: GLWE<Vec<u8>> = GLWE::alloc_from_infos(&glwe_in_infos);
            let mut glwe_out: GLWE<Vec<u8>> = GLWE::alloc_from_infos(&glwe_out_infos);
            let mut pt_ggsw: ScalarZnx<Vec<u8>> = ScalarZnx::alloc(n, 1);
            let mut pt_in: GLWEPlaintext<Vec<u8>> = GLWEPlaintext::alloc_from_infos(&glwe_in_infos);
            let mut pt_out: GLWEPlaintext<Vec<u8>> = GLWEPlaintext::alloc_from_infos(&glwe_out_infos);

            let mut source_xs: Source = Source::new([0u8; 32]);
            let mut source_xe: Source = Source::new([0u8; 32]);
            let mut source_xa: Source = Source::new([0u8; 32]);

            // Random input plaintext
            module.vec_znx_fill_uniform(in_base2k, &mut pt_in.data, 0, &mut source_xa);

            pt_in.data.at_mut(0, 0)[1] = 1;

            let k: usize = 1;

            pt_ggsw.raw_mut()[k] = 1; // X^{k}

            let mut scratch: ScratchOwned<BE> = ScratchOwned::alloc(
                (module).ggsw_encrypt_sk_tmp_bytes(&ggsw_apply_infos)
                    | (module).glwe_encrypt_sk_tmp_bytes(&glwe_in_infos)
                    | module.glwe_external_product_tmp_bytes(&glwe_out_infos, &glwe_in_infos, &ggsw_apply_infos),
            );

            let mut sk: GLWESecret<Vec<u8>> = GLWESecret::alloc(n.into(), rank.into());
            sk.fill_ternary_prob(0.5, &mut source_xs);

            let mut sk_prepared: GLWESecretPrepared<DeviceBuf<BE>, BE> = module.glwe_secret_prepared_alloc(rank.into());
            module.glwe_secret_prepare(&mut sk_prepared, &sk);

            module.ggsw_encrypt_sk(
                &mut ggsw_apply,
                &pt_ggsw,
                &sk_prepared,
                &ggsw_apply_infos,
                &mut source_xe,
                &mut source_xa,
                scratch.borrow(),
            );

            module.glwe_encrypt_sk(
                &mut glwe_in,
                &pt_in,
                &sk_prepared,
                &glwe_in_infos,
                &mut source_xe,
                &mut source_xa,
                scratch.borrow(),
            );

            let mut ct_ggsw_prepared: GGSWPrepared<DeviceBuf<BE>, BE> = module.ggsw_prepared_alloc_from_infos(&ggsw_apply);
            module.ggsw_prepare(&mut ct_ggsw_prepared, &ggsw_apply, scratch.borrow());

            module.glwe_external_product(&mut glwe_out, &glwe_in, &ct_ggsw_prepared, scratch.borrow());

            module.vec_znx_rotate_assign(k as i64, &mut pt_in.data, 0, scratch.borrow());

            module.glwe_normalize(&mut pt_out, &pt_in, scratch.borrow());

            let var_gct_err_lhs: f64 = DEFAULT_SIGMA_XE * DEFAULT_SIGMA_XE;
            let var_gct_err_rhs: f64 = 0f64;

            let var_msg: f64 = 1f64 / n as f64; // X^{k}
            let var_a0_err: f64 = DEFAULT_SIGMA_XE * DEFAULT_SIGMA_XE;
            let var_a1_err: f64 = 1f64 / 12f64;

            let max_noise: f64 = noise_ggsw_product(
                n as f64,
                key_base2k * max_dsize,
                0.5,
                var_msg,
                var_a0_err,
                var_a1_err,
                var_gct_err_lhs,
                var_gct_err_rhs,
                rank as f64,
                k_in,
                k_ggsw,
            ) + 1.0;

            let noise = module
                .glwe_noise(&glwe_out, &pt_out, &sk_prepared, scratch.borrow())
                .std()
                .log2();
            assert!(noise <= max_noise, "noise: {noise} > max_noise: {max_noise}")
        }
    }
}

#[allow(clippy::too_many_arguments)]
pub fn test_glwe_external_product_assign<BE: crate::test_suite::TestBackend>(params: &TestParams, module: &Module<BE>)
where
    Module<BE>: GGSWEncryptSk<BE>
        + GGSWPreparedFactory<BE>
        + VecZnxFillUniform
        + GLWEExternalProduct<BE>
        + GLWEEncryptSk<BE>
        + GLWENoise<BE>
        + VecZnxRotateAssign<BE>
        + GLWESecretPreparedFactory<BE>,
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
            let k_ggsw: usize = k_out + key_base2k * dsize;

            let n: usize = module.n();
            let dnum: usize = k_out.div_ceil(out_base2k * max_dsize);

            let glwe_out_infos = EncryptionLayout::new_from_default_sigma(GLWELayout {
                n: n.into(),
                base2k: out_base2k.into(),
                k: k_out.into(),
                rank: rank.into(),
            })
            .unwrap();

            let ggsw_apply_infos = EncryptionLayout::new_from_default_sigma(GGSWLayout {
                n: n.into(),
                base2k: key_base2k.into(),
                k: k_ggsw.into(),
                dnum: dnum.into(),
                dsize: dsize.into(),
                rank: rank.into(),
            })
            .unwrap();

            let mut ggsw_apply: GGSW<Vec<u8>> = GGSW::alloc_from_infos(&ggsw_apply_infos);
            let mut glwe_out: GLWE<Vec<u8>> = GLWE::alloc_from_infos(&glwe_out_infos);
            let mut pt_ggsw: ScalarZnx<Vec<u8>> = ScalarZnx::alloc(n, 1);
            let mut pt_want: GLWEPlaintext<Vec<u8>> = GLWEPlaintext::alloc_from_infos(&glwe_out_infos);

            let mut source_xs: Source = Source::new([0u8; 32]);
            let mut source_xe: Source = Source::new([0u8; 32]);
            let mut source_xa: Source = Source::new([0u8; 32]);

            // Random input plaintext
            module.vec_znx_fill_uniform(out_base2k, &mut pt_want.data, 0, &mut source_xa);

            pt_want.data.at_mut(0, 0)[1] = 1;

            let k: usize = 1;

            pt_ggsw.raw_mut()[k] = 1; // X^{k}

            let mut scratch: ScratchOwned<BE> = ScratchOwned::alloc(
                (module).ggsw_encrypt_sk_tmp_bytes(&ggsw_apply_infos)
                    | (module).glwe_encrypt_sk_tmp_bytes(&glwe_out_infos)
                    | module.glwe_external_product_tmp_bytes(&glwe_out_infos, &glwe_out_infos, &ggsw_apply_infos),
            );

            let mut sk: GLWESecret<Vec<u8>> = GLWESecret::alloc(n.into(), rank.into());
            sk.fill_ternary_prob(0.5, &mut source_xs);

            let mut sk_prepared: GLWESecretPrepared<DeviceBuf<BE>, BE> = module.glwe_secret_prepared_alloc(rank.into());
            module.glwe_secret_prepare(&mut sk_prepared, &sk);

            module.ggsw_encrypt_sk(
                &mut ggsw_apply,
                &pt_ggsw,
                &sk_prepared,
                &ggsw_apply_infos,
                &mut source_xe,
                &mut source_xa,
                scratch.borrow(),
            );

            module.glwe_encrypt_sk(
                &mut glwe_out,
                &pt_want,
                &sk_prepared,
                &glwe_out_infos,
                &mut source_xe,
                &mut source_xa,
                scratch.borrow(),
            );

            let mut ct_ggsw_prepared: GGSWPrepared<DeviceBuf<BE>, BE> = module.ggsw_prepared_alloc_from_infos(&ggsw_apply);
            module.ggsw_prepare(&mut ct_ggsw_prepared, &ggsw_apply, scratch.borrow());

            module.glwe_external_product_assign(&mut glwe_out, &ct_ggsw_prepared, scratch.borrow());

            module.vec_znx_rotate_assign(k as i64, &mut pt_want.data, 0, scratch.borrow());

            let var_gct_err_lhs: f64 = DEFAULT_SIGMA_XE * DEFAULT_SIGMA_XE;
            let var_gct_err_rhs: f64 = 0f64;

            let var_msg: f64 = 1f64 / n as f64; // X^{k}
            let var_a0_err: f64 = DEFAULT_SIGMA_XE * DEFAULT_SIGMA_XE;
            let var_a1_err: f64 = 1f64 / 12f64;

            let max_noise: f64 = noise_ggsw_product(
                n as f64,
                key_base2k * max_dsize,
                0.5,
                var_msg,
                var_a0_err,
                var_a1_err,
                var_gct_err_lhs,
                var_gct_err_rhs,
                rank as f64,
                k_out,
                k_ggsw,
            ) + 1.0;

            let noise = module
                .glwe_noise(&glwe_out, &pt_want, &sk_prepared, scratch.borrow())
                .std()
                .log2();
            assert!(noise <= max_noise, "noise: {noise} > max_noise: {max_noise}")
        }
    }
}
