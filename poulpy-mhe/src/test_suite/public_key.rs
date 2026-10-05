//! Collective public key over three parties: a ciphertext encrypted
//! under the finalized key decrypts under the ideal secret.

use poulpy_core::{
    DEFAULT_SIGMA_XE, Distribution, GLWEEncryptPk, GLWEEncryptPkSmudged, GLWEEncryptSk, GLWENoise, GLWEPublicKeyGenerate,
    GetDistributionMut, Noise,
    layouts::{
        GLWE, GLWELayout, GLWEPlaintext, GLWEPublicKey, GLWEPublicKeyCompressedSeedMut, GLWEPublicKeyPrepared,
        GLWEPublicKeyPreparedFactory, GLWESecret, GLWESecretPrepared, GLWESecretPreparedFactory, GLWESecretSampling, LWEInfos,
        ModuleCoreAlloc, Rank, TorusPrecision,
    },
};
use poulpy_hal::{
    AlignedBuf,
    api::{ModuleNew, ScratchOwnedAlloc, ScratchOwnedBorrow, VecZnxAddScalarAssign, VecZnxFillUniformSource},
    layouts::{HostBackend, HostDataMut, HostDataRef, Module, ReaderFrom, ScratchOwned, WriterTo},
    source::Source,
    test_suite::vec_znx_backend_mut,
};

use super::fixtures::{BASE2K, K, PARTIES, RANK, SEEDS, collective_public_key, ideal_secret, party_secrets};
use crate::{api::GLWEPublicKeyMHEProtocol, layouts::MHEModuleAlloc};

pub fn test_glwe_public_key<BE>(module: &Module<BE>)
where
    BE: HostBackend<OwnedBuf = AlignedBuf, ZnxWord = i64>,
    for<'a> BE::BufRef<'a>: HostDataRef,
    for<'a> BE::BufMut<'a>: HostDataMut,
    Module<BE>: MHEModuleAlloc<BE>
        + GLWEPublicKeyMHEProtocol<BE>
        + GLWESecretSampling<BE>
        + GLWESecretPreparedFactory<BE>
        + VecZnxAddScalarAssign<BE>
        + GLWEPublicKeyPreparedFactory<BE>
        + GLWEEncryptPk<BE>
        + GLWENoise<BE>
        + VecZnxFillUniformSource<BE>,
    ScratchOwned<BE>: ScratchOwnedAlloc<BE> + ScratchOwnedBorrow<BE>,
{
    let layout = GLWELayout {
        n: module.n().into(),
        base2k: BASE2K,
        k: K,
        rank: RANK,
    };

    let parties = party_secrets(module);
    let sk_ideal = ideal_secret(module, &parties);
    let pk_prepared = collective_public_key(module, &parties, &layout);
    let mut scratch: ScratchOwned<BE> = ScratchOwned::alloc(
        module
            .glwe_encrypt_pk_tmp_bytes(&layout, &layout)
            .max(module.glwe_noise_tmp_bytes(&layout))
            .max(module.mhe_glwe_public_key_share_gen_tmp_bytes(&layout)),
    );

    // A mismatch in the last entry must be rejected before an earlier entry
    // changes, including its coefficients and the aggregate's provenance.
    let mut acc = module.glwe_public_key_share_alloc_from_infos(&layout);
    let mut share = module.glwe_public_key_share_alloc_from_infos(&layout);
    for (i, dst) in [&mut acc, &mut share].into_iter().enumerate() {
        module.mhe_glwe_public_key_share_gen(
            dst,
            &parties[i].1,
            SEEDS[0],
            &mut Source::new([70 + i as u8; 32]),
            &mut scratch.borrow(),
        );
    }
    share.seed_mut()[RANK.as_usize() - 1][0] ^= 1;
    let unchanged = acc.clone();
    super::fixtures::assert_panics_with("invalid aggregation: seeds differ", || {
        module.mhe_glwe_public_key_share_aggregate(&mut acc, &share);
    });
    assert!(acc == unchanged);
    super::fixtures::assert_collective_metadata(&acc, 1);

    let mut pt: GLWEPlaintext<AlignedBuf, i64> = module.glwe_plaintext_alloc_from_infos(&layout);
    module.vec_znx_fill_uniform_source(
        BASE2K.as_usize(),
        K.as_usize(),
        &mut vec_znx_backend_mut::<BE>(pt.data_mut()),
        0,
        &mut Source::new([30u8; 32]),
    );
    let mut ct: GLWE<AlignedBuf, i64> = module.glwe_alloc_from_infos(&layout);
    module.glwe_encrypt_pk(
        &mut ct,
        &pt,
        &pk_prepared,
        &mut Source::new([31u8; 32]),
        &mut Source::new([32u8; 32]),
        &mut scratch.borrow(),
    );

    super::fixtures::assert_collective_metadata(&ct, PARTIES);

    // Sum_l u_l e_l over rank entries whose errors sum PARTIES errors, plus <e_1, s> with ideal-secret variance PARTIES / 2.
    let n = module.n() as f64;
    let rank = RANK.as_usize() as f64;
    let variance = 2.0 * rank * n * 0.5 * PARTIES as f64 * DEFAULT_SIGMA_XE * DEFAULT_SIGMA_XE;
    super::fixtures::assert_fresh_noise(&ct, variance + DEFAULT_SIGMA_XE.powi(2), K);
    let bound = variance.sqrt().log2() - K.as_usize() as f64 + 1.25_f64.log2();
    let noise: f64 = module.glwe_noise(&ct, &pt, &sk_ideal, &mut scratch.borrow()).std().log2();
    assert!(noise <= bound, "noise {noise} above bound {bound}");
}

/// Actual SK, single-PK and collective-PK producers replace a reused
/// ciphertext's fresh phase-error estimate, including its precision and secret.
pub fn test_ciphertext_encryption_noise_tags<BE>(module: &Module<BE>)
where
    BE: HostBackend<OwnedBuf = AlignedBuf, ZnxWord = i64>,
    for<'a> BE::BufRef<'a>: HostDataRef,
    for<'a> BE::BufMut<'a>: HostDataMut,
    Module<BE>: MHEModuleAlloc<BE>
        + GLWEPublicKeyMHEProtocol<BE>
        + GLWESecretSampling<BE>
        + GLWESecretPreparedFactory<BE>
        + GLWEPublicKeyPreparedFactory<BE>
        + GLWEPublicKeyGenerate<BE>
        + GLWEEncryptSk<BE>
        + GLWEEncryptPk<BE>
        + GLWEEncryptPkSmudged<BE>,
    ScratchOwned<BE>: ScratchOwnedAlloc<BE> + ScratchOwnedBorrow<BE>,
{
    let key_k = TorusPrecision(K.0 + 2 * BASE2K.0);
    let key_layout = GLWELayout {
        n: module.n().into(),
        base2k: BASE2K,
        k: key_k,
        rank: RANK,
    };
    let sigma2 = DEFAULT_SIGMA_XE.powi(2);
    let rank_n = RANK.as_usize() as f64 * module.n() as f64;
    let mut scratch: ScratchOwned<BE> = ScratchOwned::alloc(
        module
            .glwe_encrypt_sk_tmp_bytes(&key_layout)
            .max(module.glwe_encrypt_pk_tmp_bytes(&key_layout, &key_layout))
            .max(module.glwe_encrypt_pk_smudged_tmp_bytes(&key_layout, &key_layout))
            .max(module.glwe_public_key_generate_tmp_bytes(&key_layout))
            .max(module.glwe_public_key_prepare_tmp_bytes(&key_layout))
            .max(module.mhe_glwe_public_key_share_gen_tmp_bytes(&key_layout))
            .max(module.mhe_glwe_public_key_share_finalize_tmp_bytes()),
    );

    for base in [Distribution::TernaryProb(0.5), Distribution::BinaryProb(0.5)] {
        let mut secrets = Vec::new();
        let mut aggregate = module.glwe_public_key_share_alloc_from_infos(&key_layout);
        let mut share = module.glwe_public_key_share_alloc_from_infos(&key_layout);
        for i in 0..PARTIES {
            let mut sk: GLWESecret<AlignedBuf, i64> = module.glwe_secret_alloc(RANK);
            let mut source = Source::new([100 + i as u8; 32]);
            match base {
                Distribution::TernaryProb(p) => module.glwe_secret_fill_ternary_prob(&mut sk, p, &mut source),
                Distribution::BinaryProb(p) => module.glwe_secret_fill_binary_prob(&mut sk, p, &mut source),
                _ => unreachable!(),
            }
            let mut prepared: GLWESecretPrepared<AlignedBuf, BE> = module.glwe_secret_prepared_alloc(RANK);
            module.glwe_secret_prepare(&mut prepared, &sk);
            let dst = if i == 0 { &mut aggregate } else { &mut share };
            module.mhe_glwe_public_key_share_gen(
                dst,
                &prepared,
                SEEDS[0],
                &mut Source::new([110 + i as u8; 32]),
                &mut scratch.borrow(),
            );
            if i > 0 {
                module.mhe_glwe_public_key_share_aggregate(&mut aggregate, &share);
            }
            assert_noise_tag(&aggregate, base, i + 1, (i + 1) as f64 * sigma2, key_k);
            secrets.push(prepared);
        }

        let mut encoded = Vec::new();
        aggregate.write_to(&mut encoded).unwrap();
        let mut decoded = module.glwe_public_key_share_alloc_from_infos(&key_layout);
        decoded.read_from(&mut encoded.as_slice()).unwrap();
        assert!(decoded == aggregate);
        let mut collective: GLWEPublicKey<AlignedBuf, i64> = module.glwe_public_key_alloc_from_infos(&key_layout);
        module.mhe_glwe_public_key_share_finalize(&mut collective, &decoded, &mut scratch.borrow());
        assert_noise_tag(&collective, base, PARTIES, PARTIES as f64 * sigma2, key_k);
        let mut collective_prepared: GLWEPublicKeyPrepared<AlignedBuf, BE> =
            module.glwe_public_key_prepared_alloc_from_infos(&key_layout);
        module.glwe_public_key_prepare(&mut collective_prepared, &collective, &mut scratch.borrow());
        assert_noise_tag(&collective_prepared, base, PARTIES, PARTIES as f64 * sigma2, key_k);

        let mut single: GLWEPublicKey<AlignedBuf, i64> = module.glwe_public_key_alloc_from_infos(&key_layout);
        module.glwe_public_key_generate(
            &mut single,
            &secrets[0],
            &mut Source::new([120; 32]),
            &mut Source::new([121; 32]),
            &mut scratch.borrow(),
        );
        let mut single_prepared: GLWEPublicKeyPrepared<AlignedBuf, BE> =
            module.glwe_public_key_prepared_alloc_from_infos(&key_layout);
        module.glwe_public_key_prepare(&mut single_prepared, &single, &mut scratch.borrow());
        assert_noise_tag(&single_prepared, base, 1, sigma2, key_k);

        for k in [key_k, TorusPrecision(key_k.0 - 1), TorusPrecision(K.0 - BASE2K.0)] {
            let layout = GLWELayout { k, ..key_layout };
            let mut ct: GLWE<AlignedBuf, i64> = module.glwe_alloc_from_infos(&layout);
            let mut pt: GLWEPlaintext<AlignedBuf, i64> = module.glwe_plaintext_alloc_from_infos(&layout);
            pt.encode_vec_i64(&vec![1; module.n()], k);
            let zero_pt: GLWEPlaintext<AlignedBuf, i64> = module.glwe_plaintext_alloc_from_infos(&layout);
            let mut source_xu = Source::new([130; 32]);
            let mut source_xe = Source::new([131; 32]);
            let mut source_xa = Source::new([132; 32]);
            let mut source_smudge = Source::new([133; 32]);
            module.glwe_encrypt_sk(
                &mut ct,
                &pt,
                &secrets[0],
                &mut source_xe,
                &mut source_xa,
                &mut scratch.borrow(),
            );
            assert_noise_tag(&ct, base, 1, sigma2, k);

            for (parties, pk) in [(1, &single_prepared), (PARTIES, &collective_prepared)] {
                let count = parties as f64;
                // E[S^2] = N E[s^2] + N(N-1) E[s]^2. Binary secrets
                // include the cross-party mean term, unlike centered ternary.
                let mean = if matches!(base, Distribution::BinaryProb(_)) {
                    0.5
                } else {
                    0.0
                };
                let secret_second = count * 0.5 + count * (count - 1.0) * mean * mean;
                let key_grid_scale = (2.0 * (k.as_usize() as f64 - key_k.as_usize() as f64)).exp2();
                let inherited = rank_n * 0.5 * count * sigma2 * key_grid_scale;
                let mask_error = rank_n * secret_second * sigma2;
                let phase_fold = 1.0 + rank_n * secret_second;
                let rounding = phase_fold / 4.0;
                let extra_bits = (key_k.0 - k.0) as usize;
                let prefix_amplification = if mean == 0.0 {
                    rank_n * 0.5 * phase_fold
                } else {
                    (rank_n * 0.5_f64.sqrt() * (1.0 + rank_n * secret_second.sqrt())).powi(2)
                };
                let expected = |fresh| {
                    super::fixtures::expected_pk_variance(
                        inherited,
                        fresh,
                        phase_fold,
                        prefix_amplification,
                        BASE2K.as_usize(),
                        k.as_usize(),
                        key_k.as_usize(),
                    )
                };
                let (ordinary_delta, ordinary) = expected(mask_error + sigma2);
                let (smudged_delta, without_body) = expected(mask_error);
                if extra_bits > BASE2K.as_usize() {
                    // The three-bit candidate omits too much of the PK. One
                    // more bit crosses the limb boundary and meets the target
                    // while still omitting two of the five available limbs.
                    assert_eq!(ordinary_delta, 4);
                    assert_eq!(smudged_delta, 4);
                    let work_limbs = (k.as_usize() + ordinary_delta).div_ceil(BASE2K.as_usize());
                    assert_eq!(work_limbs, 3);
                    assert_eq!(key_k.as_usize().div_ceil(BASE2K.as_usize()), 5);
                    assert!(ordinary - rounding <= rounding);
                    let tail = 0.5 / (1.0 - (-(BASE2K.as_usize() as f64)).exp2());
                    let previous_work_k = (k.as_usize() + 3).div_ceil(BASE2K.as_usize()) * BASE2K.as_usize();
                    let previous_tail =
                        prefix_amplification * tail.powi(2) * (-2.0 * (previous_work_k - k.as_usize()) as f64).exp2();
                    assert!((inherited.sqrt() + previous_tail.sqrt()).powi(2) + (mask_error + sigma2) / 64.0 > rounding);
                } else {
                    assert_eq!(ordinary_delta, extra_bits);
                    assert_eq!(smudged_delta, extra_bits);
                    if extra_bits == 1 {
                        // The inherited key error alone already exceeds the
                        // target, so the selector must stop at the key's cap.
                        assert!(inherited > rounding);
                        assert!(ordinary - rounding > rounding);
                    }
                }
                assert!(ordinary > sigma2);
                module.glwe_encrypt_pk(&mut ct, &pt, pk, &mut source_xu, &mut source_xe, &mut scratch.borrow());
                assert_noise_tag(&ct, base, parties, ordinary, k);
                module.glwe_encrypt_pk_at_col(
                    &mut ct,
                    &zero_pt,
                    RANK.as_usize(),
                    pk,
                    &mut source_xu,
                    &mut source_xe,
                    &mut scratch.borrow(),
                );
                assert_noise_tag(&ct, base, parties, ordinary, k);
                module.glwe_encrypt_zero_pk(&mut ct, pk, &mut source_xu, &mut source_xe, &mut scratch.borrow());
                assert_noise_tag(&ct, base, parties, ordinary, k);
                for (flood, variance) in [
                    // Even the default Gaussian is an output-grid flood here,
                    // rather than the ordinary intermediate-grid body error.
                    (Noise::ENCRYPTION, sigma2),
                    (Noise::Uniform { bits: 4 }, 21.25),
                    (
                        Noise::Gaussian {
                            sigma: 128.0,
                            cutoff_factor: 6,
                        },
                        16384.0,
                    ),
                ] {
                    module.glwe_encrypt_pk_smudged(
                        &mut ct,
                        &pt,
                        pk,
                        flood,
                        &mut source_xu,
                        &mut source_xe,
                        &mut source_smudge,
                        &mut scratch.borrow(),
                    );
                    assert_noise_tag(&ct, base, parties, without_body + variance, k);
                }
            }
            // Reusing a collective-PK ciphertext for SK encryption must clear
            // both its amplified estimate and its collective party count.
            module.glwe_encrypt_sk(
                &mut ct,
                &pt,
                &secrets[0],
                &mut source_xe,
                &mut source_xa,
                &mut scratch.borrow(),
            );
            assert_noise_tag(&ct, base, 1, sigma2, k);
        }
    }
}

fn assert_noise_tag(value: &impl LWEInfos, base: Distribution, parties: usize, variance: f64, k: TorusPrecision) {
    let metadata = value.encryption_metadata().expect("fresh encryption must record provenance");
    assert_eq!(metadata.secret_distribution().base(), base);
    assert_eq!(metadata.parties(), parties as u64);
    super::fixtures::assert_fresh_noise(value, variance, k);
    assert!((metadata.fresh_noise().std_dev() - variance.sqrt()).abs() <= variance.sqrt() * 1e-12);
}

/// Finalizing a share without a samplable distribution panics; a fresh share has `NONE`.
pub fn test_glwe_public_key_finalize_dist_none<BE>(module: &Module<BE>)
where
    BE: HostBackend<OwnedBuf = AlignedBuf, ZnxWord = i64>,
    for<'a> BE::BufRef<'a>: HostDataRef,
    for<'a> BE::BufMut<'a>: HostDataMut,
    Module<BE>: MHEModuleAlloc<BE> + GLWEPublicKeyMHEProtocol<BE>,
    ScratchOwned<BE>: ScratchOwnedAlloc<BE> + ScratchOwnedBorrow<BE>,
{
    let layout = GLWELayout {
        n: module.n().into(),
        base2k: BASE2K,
        k: K,
        rank: RANK,
    };
    let share = module.glwe_public_key_share_alloc_from_infos(&layout);
    let mut pk: GLWEPublicKey<AlignedBuf, i64> = module.glwe_public_key_alloc_from_infos(&layout);
    let mut scratch: ScratchOwned<BE> = ScratchOwned::alloc(module.mhe_glwe_public_key_share_finalize_tmp_bytes());
    module.mhe_glwe_public_key_share_finalize(&mut pk, &share, &mut scratch.borrow());
}

/// Entries sharing a seed share their masks, which makes encryption rank 1 in the ephemerals; a fresh share has zero seeds.
pub fn test_glwe_public_key_finalize_shared_seed<BE>(module: &Module<BE>)
where
    BE: HostBackend<OwnedBuf = AlignedBuf, ZnxWord = i64>,
    for<'a> BE::BufRef<'a>: HostDataRef,
    for<'a> BE::BufMut<'a>: HostDataMut,
    Module<BE>: MHEModuleAlloc<BE> + GLWEPublicKeyMHEProtocol<BE>,
    ScratchOwned<BE>: ScratchOwnedAlloc<BE> + ScratchOwnedBorrow<BE>,
{
    let layout = GLWELayout {
        n: module.n().into(),
        base2k: BASE2K,
        k: K,
        rank: RANK,
    };
    let mut share = module.glwe_public_key_share_alloc_from_infos(&layout);
    *share.dist_mut() = Distribution::TernaryProb(0.5);
    let mut pk: GLWEPublicKey<AlignedBuf, i64> = module.glwe_public_key_alloc_from_infos(&layout);
    let mut scratch: ScratchOwned<BE> = ScratchOwned::alloc(module.mhe_glwe_public_key_share_finalize_tmp_bytes());
    module.mhe_glwe_public_key_share_finalize(&mut pk, &share, &mut scratch.borrow());
}

/// Reading a share of another rank fails.
pub fn test_glwe_public_key_share_read_rank_mismatch<BE>(module: &Module<BE>)
where
    BE: HostBackend<OwnedBuf = AlignedBuf, ZnxWord = i64>,
    Module<BE>: MHEModuleAlloc<BE>,
{
    let share = module.glwe_public_key_share_alloc(BASE2K, K, RANK);
    let mut bytes = Vec::new();
    share.write_to(&mut bytes).unwrap();
    let mut res = module.glwe_public_key_share_alloc(BASE2K, K, Rank(1));
    assert!(res.read_from(&mut bytes.as_slice()).is_err());
}

/// Aggregating shares of different ranks panics.
pub fn test_glwe_public_key_aggregate_rank_mismatch<BE>(module: &Module<BE>)
where
    BE: HostBackend<OwnedBuf = AlignedBuf, ZnxWord = i64>,
    Module<BE>: MHEModuleAlloc<BE> + GLWEPublicKeyMHEProtocol<BE>,
{
    let mut a = module.glwe_public_key_share_alloc(BASE2K, K, RANK);
    let b = module.glwe_public_key_share_alloc(BASE2K, K, Rank(1));
    module.mhe_glwe_public_key_share_aggregate(&mut a, &b);
}

/// Aggregating shares generated under secrets of different distributions panics.
pub fn test_glwe_public_key_aggregate_dist_mismatch<BE>(module: &Module<BE>)
where
    BE: HostBackend<OwnedBuf = AlignedBuf, ZnxWord = i64>,
    Module<BE>: MHEModuleAlloc<BE> + GLWEPublicKeyMHEProtocol<BE>,
{
    let layout = GLWELayout {
        n: module.n().into(),
        base2k: BASE2K,
        k: K,
        rank: RANK,
    };
    let mut a = module.glwe_public_key_share_alloc_from_infos(&layout);
    let mut b = module.glwe_public_key_share_alloc_from_infos(&layout);
    *b.dist_mut() = Distribution::TernaryProb(0.5);
    module.mhe_glwe_public_key_share_aggregate(&mut a, &b);
}

/// Sharing under a secret without a samplable distribution panics.
pub fn test_glwe_public_key_gen_secret_none<BE>(module: &Module<BE>)
where
    BE: HostBackend<OwnedBuf = AlignedBuf, ZnxWord = i64>,
    for<'a> BE::BufRef<'a>: HostDataRef,
    for<'a> BE::BufMut<'a>: HostDataMut,
    Module<BE>: MHEModuleAlloc<BE> + GLWEPublicKeyMHEProtocol<BE> + GLWESecretPreparedFactory<BE>,
    ScratchOwned<BE>: ScratchOwnedAlloc<BE> + ScratchOwnedBorrow<BE>,
{
    let layout = GLWELayout {
        n: module.n().into(),
        base2k: BASE2K,
        k: K,
        rank: RANK,
    };

    let sk: GLWESecretPrepared<AlignedBuf, BE> = module.glwe_secret_prepared_alloc(RANK);
    let mut res = module.glwe_public_key_share_alloc_from_infos(&layout);
    let mut scratch: ScratchOwned<BE> = ScratchOwned::alloc(module.mhe_glwe_public_key_share_gen_tmp_bytes(&layout));
    module.mhe_glwe_public_key_share_gen(&mut res, &sk, SEEDS[0], &mut Source::new([10u8; 32]), &mut scratch.borrow());
}

/// Invalid shapes fail at the protocol boundary with an exact static message.
pub fn test_glwe_public_key_gen_shape_guards<BE>(module: &Module<BE>)
where
    BE: HostBackend<OwnedBuf = AlignedBuf, ZnxWord = i64>,
    Module<BE>: ModuleNew<BE> + MHEModuleAlloc<BE> + GLWEPublicKeyMHEProtocol<BE> + GLWESecretPreparedFactory<BE>,
    ScratchOwned<BE>: ScratchOwnedAlloc<BE> + ScratchOwnedBorrow<BE>,
{
    let layout = GLWELayout {
        n: module.n().into(),
        base2k: BASE2K,
        k: K,
        rank: RANK,
    };

    let small_module = Module::<BE>::new((module.n() / 2) as u64);
    let expected = [
        "invalid share: secret degree differs from the share's",
        "invalid share: secret rank differs from the share's",
        "invalid layout: degree differs from the module's",
    ];
    for (case, expected) in expected.into_iter().enumerate() {
        super::fixtures::assert_panics_with(expected, || {
            if case == 2 {
                module.mhe_glwe_public_key_share_gen_tmp_bytes(&GLWELayout {
                    n: (module.n() / 2).into(),
                    ..layout
                });
                return;
            }
            let sk = if case == 0 {
                small_module.glwe_secret_prepared_alloc(RANK)
            } else {
                module.glwe_secret_prepared_alloc(Rank(1))
            };
            let mut res = module.glwe_public_key_share_alloc_from_infos(&layout);
            let mut scratch: ScratchOwned<BE> = ScratchOwned::alloc(module.mhe_glwe_public_key_share_gen_tmp_bytes(&layout));
            module.mhe_glwe_public_key_share_gen(&mut res, &sk, SEEDS[0], &mut Source::new([10u8; 32]), &mut scratch.borrow());
        });
    }
}
