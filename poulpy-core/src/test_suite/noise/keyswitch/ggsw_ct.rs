use poulpy_hal::{
    api::{ScratchOwnedAlloc, ScratchOwnedBorrow},
    layouts::{Module, ScalarZnx, ScratchOwned},
    source::Source,
    test_suite::TestParams,
};

use crate::layouts::GLWESecretSampling;
use crate::layouts::LWEInfos;
use crate::layouts::prepared::{GGLWEPreparedToBackendRef, GGLWEToGGSWKeyPreparedToBackendRef};
use crate::{Distribution, ScalarZnxFillDistribution};
use crate::{
    GGLWEToGGSWKeyEncryptSk, GGSWEncryptSk, GGSWKeyswitch, GGSWNoise, GLWESwitchingKeyEncryptSk,
    encryption::DEFAULT_SIGMA_XE,
    layouts::{
        Dnum, GGLWEInfos, GGLWEToGGSWKey, GGLWEToGGSWKeyLayout, GGLWEToGGSWKeyPrepared, GGLWEToGGSWKeyPreparedFactory, GGSW,
        GGSWInfos, GGSWLayout, GGSWViewWithDigits, GLWEInfos, GLWESecret, GLWESecretPreparedFactory, GLWESwitchingKey,
        GLWESwitchingKeyLayout, GLWESwitchingKeyPreparedFactory, ModuleCoreAlloc,
        prepared::{GLWESecretPrepared, GLWESwitchingKeyPrepared},
    },
    noise::GGLWENoiseModel,
};
use poulpy_hal::test_suite::scalar_znx_backend_mut;

#[allow(clippy::too_many_arguments)]
pub fn test_ggsw_keyswitch<BE: crate::test_suite::noise::TestBackend>(params: &TestParams, module: &Module<BE>)
where
    BE::OwnedBuf: poulpy_hal::layouts::HostDataMut,
    for<'a> BE::BufRef<'a>: poulpy_hal::layouts::HostDataRef,
    for<'a> BE::BufMut<'a>: poulpy_hal::layouts::HostDataMut,
    Module<BE>: GGSWEncryptSk<BE>
        + GLWESwitchingKeyEncryptSk<BE>
        + GGLWEToGGSWKeyEncryptSk<BE>
        + GGSWKeyswitch<BE>
        + GLWESecretPreparedFactory<BE>
        + GGLWEToGGSWKeyPreparedFactory<BE>
        + GLWESwitchingKeyPreparedFactory<BE>
        + GGSWNoise<BE>,
    ScratchOwned<BE>: ScratchOwnedAlloc<BE> + ScratchOwnedBorrow<BE>,
{
    let base2k: usize = params.base2k;
    let in_base2k: usize = base2k - 1;
    let key_base2k: usize = base2k;
    let out_base2k: usize = in_base2k; // MUST BE SAME
    let k_in: usize = 4 * in_base2k + 1;
    let max_dsize: usize = k_in.div_ceil(key_base2k);
    // Whole-limb digits, then widths off a limb boundary, which the switch and
    // the row expansion lay out on the digit grid.
    let k_digits: Vec<usize> = (1..max_dsize + 1)
        .map(|dsize| dsize * key_base2k)
        .chain([key_base2k / 2 + 1, key_base2k + 1, 2 * key_base2k - 1])
        .collect();

    for rank in 1_usize..3 {
        for &k_digit in &k_digits {
            let n: usize = module.n();
            let dnum_in: usize = k_in / in_base2k;

            let dsize_in: usize = 1;

            let ggsw_in_infos = GGSWLayout {
                n: n.into(),
                base2k: in_base2k.into(),
                dnum: dnum_in.into(),
                k_aux: (dsize_in * in_base2k + module.log_n()).into(),
                k_digit: crate::layouts::k_digit_of(in_base2k, dsize_in),
                rank: rank.into(),
            };

            let ggsw_out_infos: GGSWLayout = GGSWLayout {
                n: n.into(),
                base2k: out_base2k.into(),
                dnum: dnum_in.into(),
                k_aux: (dsize_in * out_base2k + module.log_n()).into(),
                k_digit: crate::layouts::k_digit_of(out_base2k, dsize_in),
                rank: rank.into(),
            };

            // Decomposes the rows it switches whole, their guard included.
            let dnum_ksk: usize = ggsw_in_infos.k().max(ggsw_out_infos.k()).as_usize().div_ceil(k_digit);

            let tsk_infos = GGLWEToGGSWKeyLayout {
                n: n.into(),
                base2k: key_base2k.into(),
                dnum: dnum_ksk.into(),
                k_aux: (k_digit + module.log_n()).into(),
                k_digit: k_digit.into(),
                rank: rank.into(),
            };

            let ksk_apply_infos = GLWESwitchingKeyLayout {
                n: n.into(),
                base2k: key_base2k.into(),
                dnum: dnum_ksk.into(),
                k_aux: (k_digit + module.log_n()).into(),
                k_digit: k_digit.into(),
                k_offset: crate::layouts::TorusPrecision(0),
                rank_in: rank.into(),
                rank_out: rank.into(),
            };

            let mut ggsw_in: GGSW<BE::OwnedBuf, BE::ZnxWord> = module.ggsw_alloc_from_infos(&ggsw_in_infos);
            let mut ggsw_out: GGSW<BE::OwnedBuf, BE::ZnxWord> = module.ggsw_alloc_from_infos(&ggsw_out_infos);
            let mut tsk: GGLWEToGGSWKey<BE::OwnedBuf, BE::ZnxWord> = module.gglwe_to_ggsw_key_alloc_from_infos(&tsk_infos);
            let mut ksk: GLWESwitchingKey<BE::OwnedBuf, BE::ZnxWord> =
                module.glwe_switching_key_alloc_from_infos(&ksk_apply_infos);
            let mut pt_scalar: ScalarZnx<BE::OwnedBuf, BE::ZnxWord> = module.scalar_znx_alloc(module.n(), 1);

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

            let mut sk_in: GLWESecret<BE::OwnedBuf, BE::ZnxWord> = module.glwe_secret_alloc(rank.into());
            module.glwe_secret_fill_ternary_prob(&mut sk_in, var_xs, &mut source_xs);

            let mut sk_in_prepared: GLWESecretPrepared<BE::OwnedBuf, BE> = module.glwe_secret_prepared_alloc(rank.into());
            module.glwe_secret_prepare(&mut sk_in_prepared, &sk_in);

            let mut sk_out: GLWESecret<BE::OwnedBuf, BE::ZnxWord> = module.glwe_secret_alloc(rank.into());
            module.glwe_secret_fill_ternary_prob(&mut sk_out, var_xs, &mut source_xs);

            let mut sk_out_prepared: GLWESecretPrepared<BE::OwnedBuf, BE> = module.glwe_secret_prepared_alloc(rank.into());
            module.glwe_secret_prepare(&mut sk_out_prepared, &sk_out);

            module.glwe_switching_key_encrypt_sk(
                &mut ksk,
                &sk_in,
                &sk_out,
                &mut source_xe,
                &mut source_xa,
                &mut scratch.arena(),
            );
            module.gglwe_to_ggsw_key_encrypt_sk(
                &mut tsk,
                &sk_out,
                &mut source_xe,
                &mut source_xa,
                &mut crate::test_suite::noise::scratch_host_arena(&mut scratch),
            );

            module.scalar_znx_fill_distribution(
                &mut scalar_znx_backend_mut::<BE>(&mut pt_scalar),
                0,
                Distribution::TernaryFixed(n),
                &mut source_xs,
            );

            module.ggsw_encrypt_sk(
                &mut ggsw_in,
                &pt_scalar,
                &sk_in_prepared,
                &mut source_xe,
                &mut source_xa,
                &mut scratch.borrow(),
            );

            let mut ksk_prepared: GLWESwitchingKeyPrepared<BE::OwnedBuf, BE> =
                module.glwe_switching_key_prepared_alloc_from_infos(&ksk);
            module.glwe_switching_key_prepare(&mut ksk_prepared, &ksk, &mut scratch.borrow());

            let mut tsk_prepared: GGLWEToGGSWKeyPrepared<BE::OwnedBuf, BE> =
                module.gglwe_to_ggsw_key_prepared_alloc_from_infos(&tsk);
            module.gglwe_to_ggsw_key_prepare(&mut tsk_prepared, &tsk, &mut scratch.borrow());

            module.ggsw_keyswitch(
                &mut ggsw_out,
                &ggsw_in,
                &ksk_prepared.to_backend_ref(),
                &tsk_prepared.to_backend_ref(),
                &mut scratch.borrow(),
            );

            let max_noise = |col_j: usize| -> f64 {
                ksk_apply_infos.log2_std_noise_ggsw_keyswitch(
                    &tsk_infos,
                    col_j,
                    &ggsw_in_infos,
                    &ggsw_out_infos,
                    var_xs,
                    var_xs,
                    DEFAULT_SIGMA_XE * DEFAULT_SIGMA_XE,
                    DEFAULT_SIGMA_XE * DEFAULT_SIGMA_XE,
                    0f64,
                ) + 0.5
            };

            for row in 0..ggsw_out.dnum().as_usize() {
                for col in 0..ggsw_out.rank().as_usize() + 1 {
                    let noise = ggsw_out
                        .noise_stats(module, row, col, &pt_scalar.to_ref(), &sk_out_prepared, &mut scratch.borrow())
                        .std()
                        .log2();
                    let max_noise = max_noise(col);
                    println!(
                        "DBG ggsw_ks col={col} have={noise:.2} max={max_noise:.2} slack={:.2}",
                        max_noise - noise
                    );
                    assert!(noise <= max_noise, "noise: {noise} > max_noise: {max_noise}")
                }
            }
        }
    }
}

/// A switching key and a tensor key of a GGSW's own gadget switch it through a
/// view of its first digits, into a GGSW of that many digits: the rows decrypt
/// within the noise model of the view's width.
pub fn test_ggsw_keyswitch_digits<BE: crate::test_suite::noise::TestBackend>(params: &TestParams, module: &Module<BE>)
where
    BE::OwnedBuf: poulpy_hal::layouts::HostDataMut,
    for<'a> BE::BufRef<'a>: poulpy_hal::layouts::HostDataRef,
    for<'a> BE::BufMut<'a>: poulpy_hal::layouts::HostDataMut,
    Module<BE>: GGSWEncryptSk<BE>
        + GLWESwitchingKeyEncryptSk<BE>
        + GGLWEToGGSWKeyEncryptSk<BE>
        + GGSWKeyswitch<BE>
        + GLWESecretPreparedFactory<BE>
        + GLWESwitchingKeyPreparedFactory<BE>
        + GGLWEToGGSWKeyPreparedFactory<BE>
        + GGSWNoise<BE>,
    ScratchOwned<BE>: ScratchOwnedAlloc<BE> + ScratchOwnedBorrow<BE>,
{
    let base2k: usize = params.base2k;
    let n: usize = module.n();
    let dnum: usize = 4;
    let k_aux: usize = base2k + module.log_n();
    let var_xs: f64 = 0.5;
    for rank in 1_usize..3 {
        let ggsw_layout = GGSWLayout {
            n: n.into(),
            base2k: base2k.into(),
            dnum: dnum.into(),
            k_digit: base2k.into(),
            k_aux: k_aux.into(),
            rank: rank.into(),
        };
        let ksk_layout = GLWESwitchingKeyLayout {
            n: n.into(),
            base2k: base2k.into(),
            dnum: dnum.into(),
            k_aux: k_aux.into(),
            k_digit: base2k.into(),
            k_offset: crate::layouts::TorusPrecision(0),
            rank_in: rank.into(),
            rank_out: rank.into(),
        };
        let tsk_layout = GGLWEToGGSWKeyLayout {
            n: n.into(),
            base2k: base2k.into(),
            dnum: dnum.into(),
            k_aux: k_aux.into(),
            k_digit: base2k.into(),
            rank: rank.into(),
        };
        let max_input_k = ksk_layout.gglwe_layout().max_input_k();
        let at = |d: usize| GGSWLayout {
            dnum: d.into(),
            ..ggsw_layout
        };
        let digits: usize = (1..dnum).rev().find(|&d| at(d).k() <= max_input_k).unwrap();
        let view_layout = at(digits);

        let mut scratch: ScratchOwned<BE> = ScratchOwned::alloc(
            module
                .ggsw_encrypt_sk_tmp_bytes(&ggsw_layout)
                .max(module.glwe_switching_key_encrypt_sk_tmp_bytes(&ksk_layout))
                .max(module.gglwe_to_ggsw_key_encrypt_sk_tmp_bytes(&tsk_layout))
                .max(module.ggsw_keyswitch_tmp_bytes(&view_layout, &view_layout, &ksk_layout, &tsk_layout))
                .max(module.ggsw_noise_tmp_bytes(&view_layout)),
        );
        let mut source_xs: Source = Source::new([7u8; 32]);
        let mut source_xe: Source = Source::new([8u8; 32]);
        let mut source_xa: Source = Source::new([9u8; 32]);
        let mut sk_in: GLWESecret<BE::OwnedBuf, BE::ZnxWord> = module.glwe_secret_alloc(rank.into());
        module.glwe_secret_fill_ternary_prob(&mut sk_in, var_xs, &mut source_xs);
        let mut sk_in_prepared: GLWESecretPrepared<BE::OwnedBuf, BE> = module.glwe_secret_prepared_alloc(rank.into());
        module.glwe_secret_prepare(&mut sk_in_prepared, &sk_in);
        let mut sk_out: GLWESecret<BE::OwnedBuf, BE::ZnxWord> = module.glwe_secret_alloc(rank.into());
        module.glwe_secret_fill_ternary_prob(&mut sk_out, var_xs, &mut source_xs);
        let mut sk_out_prepared: GLWESecretPrepared<BE::OwnedBuf, BE> = module.glwe_secret_prepared_alloc(rank.into());
        module.glwe_secret_prepare(&mut sk_out_prepared, &sk_out);

        let mut ksk: GLWESwitchingKey<BE::OwnedBuf, BE::ZnxWord> = module.glwe_switching_key_alloc_from_infos(&ksk_layout);
        module.glwe_switching_key_encrypt_sk(
            &mut ksk,
            &sk_in,
            &sk_out,
            &mut source_xe,
            &mut source_xa,
            &mut scratch.arena(),
        );
        let mut tsk: GGLWEToGGSWKey<BE::OwnedBuf, BE::ZnxWord> = module.gglwe_to_ggsw_key_alloc_from_infos(&tsk_layout);
        module.gglwe_to_ggsw_key_encrypt_sk(
            &mut tsk,
            &sk_out,
            &mut source_xe,
            &mut source_xa,
            &mut crate::test_suite::noise::scratch_host_arena(&mut scratch),
        );
        let mut pt_scalar: ScalarZnx<BE::OwnedBuf, BE::ZnxWord> = module.scalar_znx_alloc(n, 1);
        module.scalar_znx_fill_distribution(
            &mut scalar_znx_backend_mut::<BE>(&mut pt_scalar),
            0,
            Distribution::TernaryFixed(n),
            &mut source_xs,
        );
        let mut ggsw_in: GGSW<BE::OwnedBuf, BE::ZnxWord> = module.ggsw_alloc_from_infos(&ggsw_layout);
        module.ggsw_encrypt_sk(
            &mut ggsw_in,
            &pt_scalar,
            &sk_in_prepared,
            &mut source_xe,
            &mut source_xa,
            &mut scratch.borrow(),
        );
        let mut ksk_prepared: GLWESwitchingKeyPrepared<BE::OwnedBuf, BE> =
            module.glwe_switching_key_prepared_alloc_from_infos(&ksk);
        module.glwe_switching_key_prepare(&mut ksk_prepared, &ksk, &mut scratch.borrow());
        let mut tsk_prepared: GGLWEToGGSWKeyPrepared<BE::OwnedBuf, BE> = module.gglwe_to_ggsw_key_prepared_alloc_from_infos(&tsk);
        module.gglwe_to_ggsw_key_prepare(&mut tsk_prepared, &tsk, &mut scratch.borrow());

        let mut ggsw_out: GGSW<BE::OwnedBuf, BE::ZnxWord> = module.ggsw_alloc_from_infos(&view_layout);
        module.ggsw_keyswitch(
            &mut ggsw_out,
            &ggsw_in.view_with_digits(Dnum(digits as u32)).unwrap(),
            &ksk_prepared.to_backend_ref(),
            &tsk_prepared.to_backend_ref(),
            &mut scratch.borrow(),
        );

        for row in 0..digits {
            for col in 0..rank + 1 {
                let max_noise: f64 = ksk_layout.log2_std_noise_ggsw_keyswitch(
                    &tsk_layout,
                    col,
                    &view_layout,
                    &view_layout,
                    var_xs,
                    var_xs,
                    DEFAULT_SIGMA_XE * DEFAULT_SIGMA_XE,
                    DEFAULT_SIGMA_XE * DEFAULT_SIGMA_XE,
                    0f64,
                ) + 0.5;
                let noise: f64 = module
                    .ggsw_noise(
                        &ggsw_out,
                        row,
                        col,
                        &pt_scalar.to_ref(),
                        &sk_out_prepared,
                        &mut scratch.borrow(),
                    )
                    .std()
                    .log2();
                assert!(
                    noise <= max_noise,
                    "rank={rank} row={row} col={col}: noise: {noise} > max_noise: {max_noise}"
                );
            }
        }
    }
}

#[allow(clippy::too_many_arguments)]
pub fn test_ggsw_keyswitch_assign<BE: crate::test_suite::noise::TestBackend>(params: &TestParams, module: &Module<BE>)
where
    BE::OwnedBuf: poulpy_hal::layouts::HostDataMut,
    for<'a> BE::BufRef<'a>: poulpy_hal::layouts::HostDataRef,
    for<'a> BE::BufMut<'a>: poulpy_hal::layouts::HostDataMut,
    Module<BE>: GGSWEncryptSk<BE>
        + GLWESwitchingKeyEncryptSk<BE>
        + GGLWEToGGSWKeyEncryptSk<BE>
        + GGSWKeyswitch<BE>
        + GLWESecretPreparedFactory<BE>
        + GGLWEToGGSWKeyPreparedFactory<BE>
        + GLWESwitchingKeyPreparedFactory<BE>
        + GGSWNoise<BE>,
    ScratchOwned<BE>: ScratchOwnedAlloc<BE> + ScratchOwnedBorrow<BE>,
{
    let base2k: usize = params.base2k;
    let out_base2k: usize = base2k - 1;
    let key_base2k: usize = base2k;
    let k_out: usize = 4 * out_base2k + 1;
    let max_dsize: usize = k_out.div_ceil(key_base2k);
    // Whole-limb digits, then widths off a limb boundary, which the switch and
    // the row expansion lay out on the digit grid.
    let k_digits: Vec<usize> = (1..max_dsize + 1)
        .map(|dsize| dsize * key_base2k)
        .chain([key_base2k / 2 + 1, key_base2k + 1, 2 * key_base2k - 1])
        .collect();

    for rank in 1_usize..3 {
        for &k_digit in &k_digits {
            let n: usize = module.n();
            let dnum_in: usize = k_out / out_base2k;
            let dsize_in: usize = 1;

            let ggsw_out_infos = GGSWLayout {
                n: n.into(),
                base2k: out_base2k.into(),
                dnum: dnum_in.into(),
                k_aux: (dsize_in * out_base2k + module.log_n()).into(),
                k_digit: crate::layouts::k_digit_of(out_base2k, dsize_in),
                rank: rank.into(),
            };

            // Decomposes the rows it switches whole, their guard included.
            let dnum_ksk: usize = ggsw_out_infos.k().as_usize().div_ceil(k_digit);

            let tsk_infos = GGLWEToGGSWKeyLayout {
                n: n.into(),
                base2k: key_base2k.into(),
                dnum: dnum_ksk.into(),
                k_aux: (k_digit + module.log_n()).into(),
                k_digit: k_digit.into(),
                rank: rank.into(),
            };

            let ksk_apply_infos = GLWESwitchingKeyLayout {
                n: n.into(),
                base2k: key_base2k.into(),
                dnum: dnum_ksk.into(),
                k_aux: (k_digit + module.log_n()).into(),
                k_digit: k_digit.into(),
                k_offset: crate::layouts::TorusPrecision(0),
                rank_in: rank.into(),
                rank_out: rank.into(),
            };

            let mut ggsw_out: GGSW<BE::OwnedBuf, BE::ZnxWord> = module.ggsw_alloc_from_infos(&ggsw_out_infos);
            let mut tsk: GGLWEToGGSWKey<BE::OwnedBuf, BE::ZnxWord> = module.gglwe_to_ggsw_key_alloc_from_infos(&tsk_infos);
            let mut ksk: GLWESwitchingKey<BE::OwnedBuf, BE::ZnxWord> =
                module.glwe_switching_key_alloc_from_infos(&ksk_apply_infos);
            let mut pt_scalar: ScalarZnx<BE::OwnedBuf, BE::ZnxWord> = module.scalar_znx_alloc(module.n(), 1);

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

            let mut sk_in: GLWESecret<BE::OwnedBuf, BE::ZnxWord> = module.glwe_secret_alloc(rank.into());
            module.glwe_secret_fill_ternary_prob(&mut sk_in, var_xs, &mut source_xs);

            let mut sk_in_prepared: GLWESecretPrepared<BE::OwnedBuf, BE> = module.glwe_secret_prepared_alloc(rank.into());
            module.glwe_secret_prepare(&mut sk_in_prepared, &sk_in);

            let mut sk_out: GLWESecret<BE::OwnedBuf, BE::ZnxWord> = module.glwe_secret_alloc(rank.into());
            module.glwe_secret_fill_ternary_prob(&mut sk_out, var_xs, &mut source_xs);

            let mut sk_out_prepared: GLWESecretPrepared<BE::OwnedBuf, BE> = module.glwe_secret_prepared_alloc(rank.into());
            module.glwe_secret_prepare(&mut sk_out_prepared, &sk_out);

            module.glwe_switching_key_encrypt_sk(
                &mut ksk,
                &sk_in,
                &sk_out,
                &mut source_xe,
                &mut source_xa,
                &mut scratch.arena(),
            );
            module.gglwe_to_ggsw_key_encrypt_sk(
                &mut tsk,
                &sk_out,
                &mut source_xe,
                &mut source_xa,
                &mut crate::test_suite::noise::scratch_host_arena(&mut scratch),
            );

            module.scalar_znx_fill_distribution(
                &mut scalar_znx_backend_mut::<BE>(&mut pt_scalar),
                0,
                Distribution::TernaryFixed(n),
                &mut source_xs,
            );

            module.ggsw_encrypt_sk(
                &mut ggsw_out,
                &pt_scalar,
                &sk_in_prepared,
                &mut source_xe,
                &mut source_xa,
                &mut scratch.borrow(),
            );

            let mut ksk_prepared: GLWESwitchingKeyPrepared<BE::OwnedBuf, BE> =
                module.glwe_switching_key_prepared_alloc_from_infos(&ksk);
            module.glwe_switching_key_prepare(&mut ksk_prepared, &ksk, &mut scratch.borrow());

            let mut tsk_prepared: GGLWEToGGSWKeyPrepared<BE::OwnedBuf, BE> =
                module.gglwe_to_ggsw_key_prepared_alloc_from_infos(&tsk);
            module.gglwe_to_ggsw_key_prepare(&mut tsk_prepared, &tsk, &mut scratch.borrow());

            module.ggsw_keyswitch_assign(
                &mut ggsw_out,
                &ksk_prepared.to_backend_ref(),
                &tsk_prepared.to_backend_ref(),
                &mut scratch.borrow(),
            );

            let max_noise = |col_j: usize| -> f64 {
                ksk_apply_infos.log2_std_noise_ggsw_keyswitch(
                    &tsk_infos,
                    col_j,
                    &ggsw_out_infos,
                    &ggsw_out_infos,
                    var_xs,
                    var_xs,
                    DEFAULT_SIGMA_XE * DEFAULT_SIGMA_XE,
                    DEFAULT_SIGMA_XE * DEFAULT_SIGMA_XE,
                    0f64,
                ) + 0.5
            };

            for row in 0..ggsw_out.dnum().as_usize() {
                for col in 0..ggsw_out.rank().as_usize() + 1 {
                    let noise = ggsw_out
                        .noise_stats(module, row, col, &pt_scalar.to_ref(), &sk_out_prepared, &mut scratch.borrow())
                        .std()
                        .log2();
                    let max_noise = max_noise(col);
                    println!(
                        "DBG ggsw_ks col={col} have={noise:.2} max={max_noise:.2} slack={:.2}",
                        max_noise - noise
                    );
                    assert!(noise <= max_noise, "noise: {noise} > max_noise: {max_noise}")
                }
            }
        }
    }
}
