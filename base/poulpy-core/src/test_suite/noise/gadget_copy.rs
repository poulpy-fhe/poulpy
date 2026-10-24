use poulpy_hal::{
    api::{ScratchOwnedAlloc, ScratchOwnedBorrow},
    layouts::{Module, ScalarZnx, ScratchOwned},
    source::Source,
    test_suite::{TestParams, scalar_znx_backend_mut},
};

use crate::{
    Distribution, GGLWECopy, GGLWENoise, GGSWCopy, GGSWEncryptSk, GGSWNoise, GLWECopy, GLWESwitchingKeyEncryptSk,
    ScalarZnxFillDistribution,
    encryption::DEFAULT_SIGMA_XE,
    layouts::{
        GGLWEToBackendMut, GGLWEToBackendRef, GGSW, GGSWLayout, GGSWToBackendMut, GGSWToBackendRef, GLWESecret,
        GLWESecretPreparedFactory, GLWESecretSampling, GLWESwitchingKey, GLWESwitchingKeyLayout, LWEInfos, ModuleCoreAlloc,
        TorusPrecision, prepared::GLWESecretPrepared,
    },
};

/// `log2` of the noise of a fresh encryption at precision `k`, rounded at `res_k`, of a rank `rank` ternary secret
/// of variance 1/2.
fn rounded_log2_std(rank: usize, n: usize, k: usize, res_k: usize) -> f64 {
    let fresh: f64 = DEFAULT_SIGMA_XE * DEFAULT_SIGMA_XE * (-2.0 * k as f64).exp2();
    let rounding: f64 = (1.0 + (rank * n) as f64 * 0.5) * (-2.0 * res_k as f64).exp2() / 12.0;
    (fresh + rounding).sqrt().log2()
}

/// `gglwe_copy` into a key of as many digits gives the key back, noise estimate included, and into a key of fewer
/// digits gives the rows `glwe_copy` gives, which decrypt within the fresh noise at the destination's precision.
pub fn test_gglwe_copy<BE: crate::test_suite::noise::TestBackend>(params: &TestParams, module: &Module<BE>)
where
    BE::OwnedBuf: poulpy_hal::layouts::HostDataMut,
    for<'a> BE::BufRef<'a>: poulpy_hal::layouts::HostDataRef,
    for<'a> BE::BufMut<'a>: poulpy_hal::layouts::HostDataMut,
    Module<BE>: GLWESwitchingKeyEncryptSk<BE> + GGLWECopy<BE> + GLWECopy<BE> + GLWESecretPreparedFactory<BE> + GGLWENoise<BE>,
    ScratchOwned<BE>: ScratchOwnedAlloc<BE> + ScratchOwnedBorrow<BE>,
{
    let base2k: usize = params.base2k;
    let n: usize = module.n();
    let dnum: usize = 4;
    for k_offset in [0, base2k + 5] {
        for rank in 1_usize..3 {
            let layout = GLWESwitchingKeyLayout {
                n: n.into(),
                base2k: base2k.into(),
                dnum: dnum.into(),
                k_aux: (base2k + module.log_n()).into(),
                k_digit: crate::layouts::k_digit_of(base2k, 1_usize),
                k_offset: TorusPrecision(k_offset as u32),
                rank_in: rank.into(),
                rank_out: rank.into(),
            };
            let mut scratch: ScratchOwned<BE> = ScratchOwned::alloc(
                module
                    .glwe_switching_key_encrypt_sk_tmp_bytes(&layout)
                    .max(module.gglwe_copy_tmp_bytes(&layout, &layout))
                    .max(module.glwe_copy_tmp_bytes(&layout, &layout))
                    .max(module.gglwe_noise_tmp_bytes(&layout)),
            );
            let mut source_xs: Source = Source::new([20u8; 32]);
            let mut source_xe: Source = Source::new([21u8; 32]);
            let mut source_xa: Source = Source::new([22u8; 32]);
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

            for digits in [dnum, 2] {
                let res_layout = GLWESwitchingKeyLayout {
                    dnum: digits.into(),
                    ..layout
                };
                let mut res: GLWESwitchingKey<BE::OwnedBuf, BE::ZnxWord> =
                    module.glwe_switching_key_alloc_from_infos(&res_layout);
                module.gglwe_copy(&mut res, &key, &mut scratch.borrow());

                let mut want: GLWESwitchingKey<BE::OwnedBuf, BE::ZnxWord> =
                    module.glwe_switching_key_alloc_from_infos(&res_layout);
                {
                    let a = GGLWEToBackendRef::<BE>::to_backend_ref(&key);
                    let mut want = GGLWEToBackendMut::<BE>::to_backend_mut(&mut want);
                    for row in 0..digits {
                        for col in 0..rank {
                            module.glwe_copy(&mut want.at_view_mut(row, col), &a.at_view(row, col), &mut scratch.borrow());
                        }
                    }
                }
                assert!(
                    res.key.data() == want.key.data(),
                    "k_offset={k_offset} rank={rank} digits={digits}: not the rows glwe_copy gives"
                );

                if digits == dnum {
                    assert!(
                        res.key == key.key,
                        "k_offset={k_offset} rank={rank}: the copy differs from the key"
                    );
                    continue;
                }
                assert!(res.noise().is_none(), "a narrowing copy kept the noise estimate");
                // The rows round at the copy's precision, their masks folded through the secret.
                let max_noise: f64 = rounded_log2_std(rank, n, layout.k().as_usize(), res_layout.k().as_usize()) + 0.5;
                for row in 0..digits {
                    for col in 0..rank {
                        let noise_have: f64 = res
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
                        assert!(
                            noise_have <= max_noise,
                            "k_offset={k_offset} rank={rank} row={row} col={col}: noise_have: {noise_have} > max_noise: {max_noise}"
                        );
                    }
                }
            }
        }
    }
}

/// `ggsw_copy` into a GGSW of as many digits gives the GGSW back, noise estimate included, and into a GGSW of fewer
/// digits gives the rows `glwe_copy` gives, which decrypt within the fresh noise at the destination's precision.
pub fn test_ggsw_copy<BE: crate::test_suite::noise::TestBackend>(params: &TestParams, module: &Module<BE>)
where
    BE::OwnedBuf: poulpy_hal::layouts::HostDataMut,
    for<'a> BE::BufRef<'a>: poulpy_hal::layouts::HostDataRef,
    for<'a> BE::BufMut<'a>: poulpy_hal::layouts::HostDataMut,
    Module<BE>: GGSWEncryptSk<BE> + GGSWCopy<BE> + GLWECopy<BE> + GLWESecretPreparedFactory<BE> + GGSWNoise<BE>,
    ScratchOwned<BE>: ScratchOwnedAlloc<BE> + ScratchOwnedBorrow<BE>,
{
    let base2k: usize = params.base2k;
    let n: usize = module.n();
    let dnum: usize = 4;
    for rank in 1_usize..3 {
        let layout = GGSWLayout {
            n: n.into(),
            base2k: base2k.into(),
            dnum: dnum.into(),
            k_aux: (base2k + module.log_n()).into(),
            k_digit: crate::layouts::k_digit_of(base2k, 1_usize),
            rank: rank.into(),
        };
        let mut scratch: ScratchOwned<BE> = ScratchOwned::alloc(
            module
                .ggsw_encrypt_sk_tmp_bytes(&layout)
                .max(module.ggsw_copy_tmp_bytes(&layout, &layout))
                .max(module.glwe_copy_tmp_bytes(&layout, &layout))
                .max(module.ggsw_noise_tmp_bytes(&layout)),
        );
        let mut source_xs: Source = Source::new([23u8; 32]);
        let mut source_xe: Source = Source::new([24u8; 32]);
        let mut source_xa: Source = Source::new([25u8; 32]);
        let mut sk: GLWESecret<BE::OwnedBuf, BE::ZnxWord> = module.glwe_secret_alloc(rank.into());
        module.glwe_secret_fill_ternary_prob(&mut sk, 0.5, &mut source_xs);
        let mut sk_prepared: GLWESecretPrepared<BE::OwnedBuf, BE> = module.glwe_secret_prepared_alloc(rank.into());
        module.glwe_secret_prepare(&mut sk_prepared, &sk);
        let mut pt_scalar: ScalarZnx<BE::OwnedBuf, BE::ZnxWord> = module.scalar_znx_alloc(n, 1);
        module.scalar_znx_fill_distribution(
            &mut scalar_znx_backend_mut::<BE>(&mut pt_scalar),
            0,
            Distribution::TernaryFixed(n),
            &mut source_xs,
        );

        let mut ggsw: GGSW<BE::OwnedBuf, BE::ZnxWord> = module.ggsw_alloc_from_infos(&layout);
        module.ggsw_encrypt_sk(
            &mut ggsw,
            &pt_scalar,
            &sk_prepared,
            &mut source_xe,
            &mut source_xa,
            &mut scratch.borrow(),
        );

        for digits in [dnum, 2] {
            let res_layout = GGSWLayout {
                dnum: digits.into(),
                ..layout
            };
            let mut res: GGSW<BE::OwnedBuf, BE::ZnxWord> = module.ggsw_alloc_from_infos(&res_layout);
            module.ggsw_copy(&mut res, &ggsw, &mut scratch.borrow());

            let mut want: GGSW<BE::OwnedBuf, BE::ZnxWord> = module.ggsw_alloc_from_infos(&res_layout);
            {
                let a = GGSWToBackendRef::<BE>::to_backend_ref(&ggsw);
                let mut want = GGSWToBackendMut::<BE>::to_backend_mut(&mut want);
                for row in 0..digits {
                    for col in 0..rank + 1 {
                        module.glwe_copy(&mut want.at_view_mut(row, col), &a.at_view(row, col), &mut scratch.borrow());
                    }
                }
            }
            assert!(
                res.data == want.data,
                "rank={rank} digits={digits}: not the rows glwe_copy gives"
            );

            if digits == dnum {
                assert!(res == ggsw, "rank={rank}: the copy differs from the GGSW");
                continue;
            }
            assert!(res.noise().is_none(), "a narrowing copy kept the noise estimate");
            let max_noise: f64 = rounded_log2_std(rank, n, layout.k().as_usize(), res_layout.k().as_usize()) + 0.5;
            for row in 0..digits {
                for col in 0..rank + 1 {
                    let noise_have: f64 = res
                        .noise_stats(module, row, col, &pt_scalar.to_ref(), &sk_prepared, &mut scratch.borrow())
                        .std()
                        .log2();
                    assert!(
                        noise_have <= max_noise,
                        "rank={rank} row={row} col={col}: noise_have: {noise_have} > max_noise: {max_noise}"
                    );
                }
            }
        }
    }
}
