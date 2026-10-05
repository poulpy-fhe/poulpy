//! Collective key switching over three parties: the finalized ciphertext
//! decrypts under the ideal output secret, and its noise carries the parties'
//! smudging noise.

use poulpy_core::{
    DEFAULT_SIGMA_XE, GLWEAdd, GLWEEncryptSk, GLWENoise, GLWENormalize, Noise,
    layouts::{
        Base2K, GLWE, GLWEInfos, GLWELayout, GLWEMask, GLWEPlaintext, GLWEPublicKeyPrepared, GLWEPublicKeyPreparedFactory,
        GLWESecretPrepared, GLWESecretPreparedFactory, GLWESecretSampling, ModuleCoreAlloc, Rank, TorusPrecision,
    },
};
use poulpy_hal::{
    AlignedBuf,
    api::{ScratchOwnedAlloc, ScratchOwnedBorrow, VecZnxAddScalarAssign, VecZnxCopy, VecZnxFillUniformSource},
    layouts::{HostBackend, HostDataMut, HostDataRef, Module, ScratchOwned},
    source::Source,
    test_suite::{vec_znx_backend_mut, vec_znx_backend_ref},
};

use super::fixtures::{
    BASE2K, K, PARTIES, RANK, Secret, assert_panics_with, collective_public_key, ideal_secret, party_secrets, secret_from_seed,
    secret_from_seed_at,
};
use crate::{
    api::{GLWEPrivateKeyswitchMHEProtocol, GLWEPublicKeyMHEProtocol, GLWEPublicKeyswitchMHEProtocol},
    layouts::MHEModuleAlloc,
    layouts::{GLWEPrivateKeyswitchShare, GLWEPublicKeyswitchShare},
};

/// Smudging noise sigma of every party, well above the fresh noise.
const SIGMA_FLOOD: f64 = 1024.0;
const FLOOD: Noise = Noise::Gaussian {
    sigma: 1024.0,
    cutoff: 6,
};

pub fn test_glwe_private_keyswitch<BE>(module: &Module<BE>)
where
    BE: HostBackend<OwnedBuf = AlignedBuf, ZnxWord = i64>,
    for<'a> BE::BufRef<'a>: HostDataRef,
    for<'a> BE::BufMut<'a>: HostDataMut,
    Module<BE>: GLWEPrivateKeyswitchMHEProtocol<BE>
        + GLWESecretSampling<BE>
        + GLWESecretPreparedFactory<BE>
        + GLWEEncryptSk<BE>
        + GLWEAdd<BE>
        + GLWENormalize<BE>
        + GLWENoise<BE>
        + VecZnxAddScalarAssign<BE>
        + VecZnxCopy<BE>
        + VecZnxFillUniformSource<BE>,
    ScratchOwned<BE>: ScratchOwnedAlloc<BE> + ScratchOwnedBorrow<BE>,
{
    let layout = glwe_layout(module);

    let parties_in = input_secrets(module);
    let parties_out = party_secrets(module);
    let sk_out = ideal_secret(module, &parties_out);
    let mut scratch: ScratchOwned<BE> = ScratchOwned::alloc(
        module
            .glwe_encrypt_sk_tmp_bytes(&layout)
            .max(module.mhe_glwe_private_keyswitch_share_gen_tmp_bytes(&layout))
            .max(module.mhe_glwe_private_keyswitch_share_finalize_tmp_bytes())
            .max(module.glwe_normalize_tmp_bytes())
            .max(module.glwe_noise_tmp_bytes(&layout)),
    );
    let (pt, ct) = encrypted_plaintext(module, &ideal_secret(module, &parties_in), &mut scratch);
    let mask = ciphertext_mask(module, &ct);

    let mut acc = module.glwe_private_keyswitch_share_alloc_from_infos(&layout);
    let mut share = module.glwe_private_keyswitch_share_alloc_from_infos(&layout);
    for (i, ((_, sk_in), (_, sk_out_i))) in parties_in.iter().zip(&parties_out).enumerate() {
        let dst = if i == 0 { &mut acc } else { &mut share };
        dst.inner.set_canonical(false);
        let mut source_smudge = Source::new([10 + i as u8; 32]);
        // The inner products with both secrets would be left in the scratch.
        poulpy_core::test_suite::assert_wipes_scratch::<BE>(
            module.mhe_glwe_private_keyswitch_share_gen_tmp_bytes(&layout),
            |scratch| {
                module.mhe_glwe_private_keyswitch_share_gen(dst, &mask, sk_in, sk_out_i, FLOOD, &mut source_smudge, scratch)
            },
        );
        assert!(dst.inner.is_canonical());
        if i > 0 {
            module.mhe_glwe_private_keyswitch_share_aggregate(&mut acc, &share);
        }
    }
    assert!(!acc.inner.is_canonical());

    let mut res: GLWE<AlignedBuf, i64> = module.glwe_alloc_from_infos(&layout);
    module.mhe_glwe_private_keyswitch_share_finalize(&mut res, &ct, &acc, &mut scratch.borrow());
    super::fixtures::assert_collective_metadata(&res, PARTIES);
    assert!(res.is_canonical());
    assert_flooded_noise(module, &res, &pt, &sk_out, 0.0, layout.k, &mut scratch);
}

pub fn test_glwe_public_keyswitch<BE>(module: &Module<BE>)
where
    BE: HostBackend<OwnedBuf = AlignedBuf, ZnxWord = i64>,
    for<'a> BE::BufRef<'a>: HostDataRef,
    for<'a> BE::BufMut<'a>: HostDataMut,
    Module<BE>: MHEModuleAlloc<BE>
        + GLWEPublicKeyswitchMHEProtocol<BE>
        + GLWEPublicKeyMHEProtocol<BE>
        + GLWESecretSampling<BE>
        + GLWESecretPreparedFactory<BE>
        + GLWEPublicKeyPreparedFactory<BE>
        + GLWEEncryptSk<BE>
        + GLWEAdd<BE>
        + GLWENormalize<BE>
        + GLWENoise<BE>
        + VecZnxAddScalarAssign<BE>
        + VecZnxCopy<BE>
        + VecZnxFillUniformSource<BE>,
    ScratchOwned<BE>: ScratchOwnedAlloc<BE> + ScratchOwnedBorrow<BE>,
{
    // Public key switching also supports a different destination rank and precision.
    for (rank_out, k_out) in [(RANK, K), (Rank(1), TorusPrecision(K.0 + BASE2K.0))] {
        let layout = glwe_layout(module);
        let share_layout = GLWELayout {
            rank: rank_out,
            k: k_out,
            ..layout
        };
        // A public key more precise than the share exercises the share scratch query for real.
        let pk_layout = GLWELayout {
            k: TorusPrecision(k_out.0 + BASE2K.0),
            ..share_layout
        };

        let parties_in = input_secrets(module);
        let parties_out: Vec<Secret<BE>> = (0..PARTIES)
            .map(|i| secret_from_seed_at(module, rank_out, [100 + i as u8; 32]))
            .collect();
        let sk_out = ideal_secret(module, &parties_out);
        let mut scratch: ScratchOwned<BE> = ScratchOwned::alloc(
            module
                .glwe_encrypt_sk_tmp_bytes(&layout)
                .max(module.mhe_glwe_public_keyswitch_share_finalize_tmp_bytes())
                .max(module.glwe_normalize_tmp_bytes())
                .max(module.glwe_noise_tmp_bytes(&share_layout)),
        );
        let share_bytes = module.mhe_glwe_public_keyswitch_share_gen_tmp_bytes(&layout, &share_layout, &pk_layout);

        let pk_out = collective_public_key(module, &parties_out, &pk_layout);

        let (pt, ct) = encrypted_plaintext(module, &ideal_secret(module, &parties_in), &mut scratch);
        let mask = ciphertext_mask(module, &ct);

        let mut acc = module.glwe_public_keyswitch_share_alloc_from_infos(&share_layout);
        let mut share = module.glwe_public_keyswitch_share_alloc_from_infos(&share_layout);
        for (i, (_, sk_in)) in parties_in.iter().enumerate() {
            let dst = if i == 0 { &mut acc } else { &mut share };
            dst.inner.set_canonical(false);
            let mut source_xu = Source::new([20 + i as u8; 32]);
            let mut source_xe = Source::new([10 + i as u8; 32]);
            let mut source_smudge = Source::new([30 + i as u8; 32]);
            // The inner product with the input secret would be left in the scratch.
            poulpy_core::test_suite::assert_wipes_scratch::<BE>(share_bytes, |scratch| {
                module.mhe_glwe_public_keyswitch_share_gen(
                    dst,
                    &mask,
                    sk_in,
                    &pk_out,
                    FLOOD,
                    &mut source_xu,
                    &mut source_xe,
                    &mut source_smudge,
                    scratch,
                )
            });
            // The flood is the body's only error: one encryption error per mask column, one flood.
            let mut want_xe = Source::new([10 + i as u8; 32]);
            (0..rank_out.as_usize()).for_each(|_| {
                want_xe.new_seed();
            });
            assert_eq!(source_xe.new_seed(), want_xe.new_seed());
            let mut want_smudge = Source::new([30 + i as u8; 32]);
            want_smudge.new_seed();
            assert_eq!(source_smudge.new_seed(), want_smudge.new_seed());
            assert!(dst.inner.is_canonical());
            if i > 0 {
                module.mhe_glwe_public_keyswitch_share_aggregate(&mut acc, &share);
            }
        }
        assert!(!acc.inner.is_canonical());

        let mut res: GLWE<AlignedBuf, i64> = module.glwe_alloc_from_infos(&share_layout);
        module.mhe_glwe_public_keyswitch_share_finalize(&mut res, &ct, &acc, &mut scratch.borrow());
        super::fixtures::assert_collective_metadata(&res, PARTIES);
        assert!(res.is_canonical());
        // Each party's pk encryption adds 2 * rank * n * 0.5 * PARTIES * sigma^2, as in the pk test.
        let n = module.n() as f64;
        let rank = rank_out.as_usize() as f64;
        let pk_noise = PARTIES as f64 * 2.0 * rank * n * 0.5 * PARTIES as f64 * DEFAULT_SIGMA_XE * DEFAULT_SIGMA_XE;
        assert_flooded_noise(module, &res, &pt, &sk_out, pk_noise, share_layout.k, &mut scratch);
    }
}

/// Finalizing into an output whose radix differs from the ciphertext's panics.
pub fn test_glwe_public_keyswitch_finalize_layout_mismatch<BE>(module: &Module<BE>)
where
    BE: HostBackend<OwnedBuf = AlignedBuf, ZnxWord = i64>,
    Module<BE>: GLWEPublicKeyswitchMHEProtocol<BE>,
    ScratchOwned<BE>: ScratchOwnedAlloc<BE> + ScratchOwnedBorrow<BE>,
{
    let layout = glwe_layout(module);
    let ct_layout = GLWELayout {
        base2k: Base2K(BASE2K.0 + 1),
        ..layout
    };
    let ct: GLWE<AlignedBuf, i64> = module.glwe_alloc_from_infos(&ct_layout);
    let share = GLWEPublicKeyswitchShare {
        inner: module.glwe_alloc_from_infos(&layout),
    };
    let mut res: GLWE<AlignedBuf, i64> = module.glwe_alloc_from_infos(&layout);
    let mut scratch: ScratchOwned<BE> = ScratchOwned::alloc(module.mhe_glwe_public_keyswitch_share_finalize_tmp_bytes());
    module.mhe_glwe_public_keyswitch_share_finalize(&mut res, &ct, &share, &mut scratch.borrow());
}

/// Aggregating key switching shares of different layouts panics.
pub fn test_glwe_private_keyswitch_aggregate_layout_mismatch<BE>(module: &Module<BE>)
where
    BE: HostBackend<OwnedBuf = AlignedBuf, ZnxWord = i64>,
    Module<BE>: MHEModuleAlloc<BE> + GLWEPrivateKeyswitchMHEProtocol<BE>,
{
    let mut a = module.glwe_private_keyswitch_share_alloc(BASE2K, K);
    let b = module.glwe_private_keyswitch_share_alloc(BASE2K, TorusPrecision(K.0 + BASE2K.0));
    module.mhe_glwe_private_keyswitch_share_aggregate(&mut a, &b);
}

/// Sharing under a public key less precise than the share panics.
pub fn test_glwe_public_keyswitch_pk_precision<BE>(module: &Module<BE>)
where
    BE: HostBackend<OwnedBuf = AlignedBuf, ZnxWord = i64>,
    Module<BE>: GLWEPublicKeyswitchMHEProtocol<BE>
        + GLWESecretSampling<BE>
        + GLWESecretPreparedFactory<BE>
        + GLWEPublicKeyPreparedFactory<BE>,
    ScratchOwned<BE>: ScratchOwnedAlloc<BE> + ScratchOwnedBorrow<BE>,
{
    let layout = glwe_layout(module);
    let pk_layout = GLWELayout {
        k: TorusPrecision(K.0 - BASE2K.0),
        ..layout
    };

    let (_, sk_in) = secret_from_seed(module, [150u8; 32]);
    let pk_out: GLWEPublicKeyPrepared<AlignedBuf, BE> = module.glwe_public_key_prepared_alloc_from_infos(&pk_layout);
    let ct: GLWE<AlignedBuf, i64> = module.glwe_alloc_from_infos(&layout);
    let mut res = GLWEPublicKeyswitchShare {
        inner: module.glwe_alloc_from_infos(&layout),
    };
    let mut scratch: ScratchOwned<BE> =
        ScratchOwned::alloc(module.mhe_glwe_public_keyswitch_share_gen_tmp_bytes(&layout, &layout, &pk_layout));
    module.mhe_glwe_public_keyswitch_share_gen(
        &mut res,
        &ct,
        &sk_in,
        &pk_out,
        FLOOD,
        &mut Source::new([20u8; 32]),
        &mut Source::new([10u8; 32]),
        &mut Source::new([30u8; 32]),
        &mut scratch.borrow(),
    );
}

/// Invalid result and secret layouts are rejected with exact static messages.
pub fn test_glwe_private_keyswitch_share_layout_guards<BE>(module: &Module<BE>)
where
    BE: HostBackend<OwnedBuf = AlignedBuf, ZnxWord = i64>,
    Module<BE>: GLWEPrivateKeyswitchMHEProtocol<BE> + GLWESecretPreparedFactory<BE>,
    ScratchOwned<BE>: ScratchOwnedAlloc<BE> + ScratchOwnedBorrow<BE>,
{
    let layout = glwe_layout(module);
    let body = GLWELayout { rank: Rank(0), ..layout };
    let ct: GLWE<AlignedBuf, i64> = module.glwe_alloc_from_infos(&layout);
    let mut scratch: ScratchOwned<BE> = ScratchOwned::alloc(module.mhe_glwe_private_keyswitch_share_gen_tmp_bytes(&layout));
    for (res_layout, rank_in, rank_out, expected) in [
        (layout, RANK, RANK, "invalid share: share rank differs from 0"),
        (
            GLWELayout {
                base2k: Base2K(BASE2K.0 + 1),
                ..body
            },
            RANK,
            RANK,
            "invalid share: share and mask layouts differ",
        ),
        (
            body,
            Rank(1),
            RANK,
            "invalid share: input secret rank differs from the mask's",
        ),
        (
            body,
            RANK,
            Rank(1),
            "invalid share: output secret rank differs from the mask's",
        ),
    ] {
        let mut res = GLWEPrivateKeyswitchShare {
            inner: module.glwe_alloc_from_infos(&res_layout),
        };
        let sk_in: GLWESecretPrepared<AlignedBuf, BE> = module.glwe_secret_prepared_alloc(rank_in);
        let sk_out: GLWESecretPrepared<AlignedBuf, BE> = module.glwe_secret_prepared_alloc(rank_out);
        assert_panics_with(expected, || {
            module.mhe_glwe_private_keyswitch_share_gen(
                &mut res,
                &ct,
                &sk_in,
                &sk_out,
                FLOOD,
                &mut Source::new([10u8; 32]),
                &mut scratch.borrow(),
            );
        });
    }
}

/// Reject invalid input and public-key layouts before core encryption/decryption.
pub fn test_glwe_public_keyswitch_share_layout_guards<BE>(module: &Module<BE>)
where
    BE: HostBackend<OwnedBuf = AlignedBuf, ZnxWord = i64>,
    Module<BE>: GLWEPublicKeyswitchMHEProtocol<BE> + GLWESecretPreparedFactory<BE> + GLWEPublicKeyPreparedFactory<BE>,
    ScratchOwned<BE>: ScratchOwnedAlloc<BE> + ScratchOwnedBorrow<BE>,
{
    let layout = glwe_layout(module);
    let ct: GLWE<AlignedBuf, i64> = module.glwe_alloc_from_infos(&layout);

    let mut scratch: ScratchOwned<BE> =
        ScratchOwned::alloc(module.mhe_glwe_public_keyswitch_share_gen_tmp_bytes(&layout, &layout, &layout));
    for (res_layout, rank_in, pk_layout, expected) in [
        (
            GLWELayout {
                base2k: Base2K(BASE2K.0 + 1),
                ..layout
            },
            RANK,
            layout,
            "invalid share: share and mask layouts differ",
        ),
        (
            layout,
            Rank(1),
            layout,
            "invalid share: input secret rank differs from the mask's",
        ),
        (
            layout,
            RANK,
            GLWELayout { rank: Rank(1), ..layout },
            "invalid share: public key and share layouts differ",
        ),
    ] {
        let mut res = GLWEPublicKeyswitchShare {
            inner: module.glwe_alloc_from_infos(&res_layout),
        };
        let sk_in: GLWESecretPrepared<AlignedBuf, BE> = module.glwe_secret_prepared_alloc(rank_in);
        let pk_out: GLWEPublicKeyPrepared<AlignedBuf, BE> = module.glwe_public_key_prepared_alloc_from_infos(&pk_layout);
        assert_panics_with(expected, || {
            module.mhe_glwe_public_keyswitch_share_gen(
                &mut res,
                &ct,
                &sk_in,
                &pk_out,
                FLOOD,
                &mut Source::new([20u8; 32]),
                &mut Source::new([10u8; 32]),
                &mut Source::new([30u8; 32]),
                &mut scratch.borrow(),
            );
        });
    }
}

/// Finalization rejects mismatched body/output layouts with static messages.
pub fn test_glwe_private_keyswitch_finalize_layout_guards<BE>(module: &Module<BE>)
where
    BE: HostBackend<OwnedBuf = AlignedBuf, ZnxWord = i64>,
    Module<BE>: GLWEPrivateKeyswitchMHEProtocol<BE> + GLWEPublicKeyswitchMHEProtocol<BE>,
    ScratchOwned<BE>: ScratchOwnedAlloc<BE> + ScratchOwnedBorrow<BE>,
{
    let layout = glwe_layout(module);
    let body = GLWELayout { rank: Rank(0), ..layout };
    let wrong_rank = GLWELayout { rank: Rank(1), ..layout };
    let wrong_degree = GLWELayout {
        n: (module.n() / 2).into(),
        ..layout
    };
    let ct: GLWE<AlignedBuf, i64> = module.glwe_alloc_from_infos(&layout);
    let mut scratch: ScratchOwned<BE> = ScratchOwned::alloc(
        module
            .mhe_glwe_private_keyswitch_share_finalize_tmp_bytes()
            .max(module.mhe_glwe_public_keyswitch_share_finalize_tmp_bytes()),
    );
    for (res_layout, share_layout, expected) in [
        (wrong_rank, body, "invalid finalization: ciphertext and output layouts differ"),
        (
            layout,
            wrong_rank,
            "invalid finalization: share layout differs from the ciphertext body",
        ),
    ] {
        let mut res: GLWE<AlignedBuf, i64> = module.glwe_alloc_from_infos(&res_layout);
        let share = GLWEPrivateKeyswitchShare {
            inner: module.glwe_alloc_from_infos(&share_layout),
        };
        assert_panics_with(expected, || {
            module.mhe_glwe_private_keyswitch_share_finalize(&mut res, &ct, &share, &mut scratch.borrow());
        });
    }
    for share_layout in [wrong_rank, wrong_degree] {
        let mut res: GLWE<AlignedBuf, i64> = module.glwe_alloc_from_infos(&layout);
        let share = GLWEPublicKeyswitchShare {
            inner: module.glwe_alloc_from_infos(&share_layout),
        };
        assert_panics_with("invalid finalization: share and output layouts differ", || {
            module.mhe_glwe_public_keyswitch_share_finalize(&mut res, &ct, &share, &mut scratch.borrow());
        });
    }
}

/// The flood is sized against the precision it is sampled at, and a flood too
/// wide for it is rejected before randomness is consumed.
pub fn test_glwe_private_keyswitch_flood_bound_guards<BE>(module: &Module<BE>)
where
    BE: HostBackend<OwnedBuf = AlignedBuf, ZnxWord = i64>,
    Module<BE>: GLWEPrivateKeyswitchMHEProtocol<BE>
        + GLWEPublicKeyswitchMHEProtocol<BE>
        + GLWESecretPreparedFactory<BE>
        + GLWEPublicKeyPreparedFactory<BE>,
    ScratchOwned<BE>: ScratchOwnedAlloc<BE> + ScratchOwnedBorrow<BE>,
{
    let layout = glwe_layout(module);
    let body_layout = GLWELayout {
        rank: Rank(0),
        k: TorusPrecision(K.0 - BASE2K.0),
        ..layout
    };
    let public_layout = GLWELayout {
        k: TorusPrecision(K.0 + BASE2K.0),
        ..layout
    };
    let ct: GLWE<AlignedBuf, i64> = module.glwe_alloc_from_infos(&layout);
    let sk: GLWESecretPrepared<AlignedBuf, BE> = module.glwe_secret_prepared_alloc(RANK);
    let pk: GLWEPublicKeyPrepared<AlignedBuf, BE> = module.glwe_public_key_prepared_alloc_from_infos(&public_layout);

    let mut scratch: ScratchOwned<BE> = ScratchOwned::alloc(
        module
            .mhe_glwe_private_keyswitch_share_gen_tmp_bytes(&layout)
            .max(module.mhe_glwe_public_keyswitch_share_gen_tmp_bytes(&layout, &public_layout, &public_layout)),
    );
    let expected = "invalid noise: Gaussian bound outside the precision";
    // A `k`-bit bound at the sampled precision `k`, which the other layouts would accept.
    let too_wide = |k: TorusPrecision| Noise::Gaussian {
        sigma: 2.0f64.powi((k.as_usize() - 3) as i32),
        cutoff: 6,
    };
    // CKS samples into the narrower result.
    let mut res = GLWEPrivateKeyswitchShare {
        inner: module.glwe_alloc_from_infos(&body_layout),
    };
    let mut source_smudge = Source::new([30u8; 32]);
    assert_panics_with(expected, || {
        module.mhe_glwe_private_keyswitch_share_gen(
            &mut res,
            &ct,
            &sk,
            &sk,
            too_wide(body_layout.k),
            &mut source_smudge,
            &mut scratch.borrow(),
        );
    });
    assert_eq!(source_smudge.next_i64(), Source::new([30u8; 32]).next_i64());
    // PCKS samples into its wider share, not the ciphertext's precision.
    let mut res = GLWEPublicKeyswitchShare {
        inner: module.glwe_alloc_from_infos(&public_layout),
    };
    let mut source_xu = Source::new([20u8; 32]);
    let mut source_xe = Source::new([10u8; 32]);
    let mut source_smudge = Source::new([30u8; 32]);
    assert_panics_with(expected, || {
        module.mhe_glwe_public_keyswitch_share_gen(
            &mut res,
            &ct,
            &sk,
            &pk,
            too_wide(public_layout.k),
            &mut source_xu,
            &mut source_xe,
            &mut source_smudge,
            &mut scratch.borrow(),
        );
    });
    assert_eq!(source_xu.next_i64(), Source::new([20u8; 32]).next_i64());
    assert_eq!(source_xe.next_i64(), Source::new([10u8; 32]).next_i64());
    assert_eq!(source_smudge.next_i64(), Source::new([30u8; 32]).next_i64());
}

fn glwe_layout<BE: poulpy_hal::layouts::Backend>(module: &Module<BE>) -> GLWELayout {
    GLWELayout {
        n: module.n().into(),
        base2k: BASE2K,
        k: K,
        rank: RANK,
    }
}

/// Input secret shares, drawn apart from the output ones.
fn input_secrets<BE>(module: &Module<BE>) -> Vec<Secret<BE>>
where
    BE: HostBackend<OwnedBuf = AlignedBuf, ZnxWord = i64>,
    Module<BE>: GLWESecretSampling<BE> + GLWESecretPreparedFactory<BE>,
{
    (0..PARTIES).map(|i| secret_from_seed(module, [150 + i as u8; 32])).collect()
}

/// A uniform plaintext and its encryption under `sk`.
fn encrypted_plaintext<BE>(
    module: &Module<BE>,
    sk: &GLWESecretPrepared<AlignedBuf, BE>,
    scratch: &mut ScratchOwned<BE>,
) -> (GLWEPlaintext<AlignedBuf, i64>, GLWE<AlignedBuf, i64>)
where
    BE: HostBackend<OwnedBuf = AlignedBuf, ZnxWord = i64>,
    Module<BE>: GLWEEncryptSk<BE> + VecZnxFillUniformSource<BE>,
    ScratchOwned<BE>: ScratchOwnedBorrow<BE>,
{
    let layout = glwe_layout(module);
    let mut pt: GLWEPlaintext<AlignedBuf, i64> = module.glwe_plaintext_alloc_from_infos(&layout);
    module.vec_znx_fill_uniform_source(
        BASE2K.as_usize(),
        K.as_usize(),
        &mut vec_znx_backend_mut::<BE>(pt.data_mut()),
        0,
        &mut Source::new([30u8; 32]),
    );
    let mut ct: GLWE<AlignedBuf, i64> = module.glwe_alloc_from_infos(&layout);
    module.glwe_encrypt_sk(
        &mut ct,
        &pt,
        sk,
        &mut Source::new([31u8; 32]),
        &mut Source::new([32u8; 32]),
        &mut scratch.borrow(),
    );
    (pt, ct)
}

/// The noise of `ct` lies between the parties' flood noise, sampled at
/// `k_flood`, and the sum of the fresh, flood and `other` variances.
fn assert_flooded_noise<BE>(
    module: &Module<BE>,
    ct: &GLWE<AlignedBuf, i64>,
    pt: &GLWEPlaintext<AlignedBuf, i64>,
    sk: &GLWESecretPrepared<AlignedBuf, BE>,
    other: f64,
    k_flood: TorusPrecision,
    scratch: &mut ScratchOwned<BE>,
) where
    BE: HostBackend<OwnedBuf = AlignedBuf, ZnxWord = i64>,
    for<'a> BE::BufRef<'a>: HostDataRef,
    for<'a> BE::BufMut<'a>: HostDataMut,
    Module<BE>: GLWENoise<BE>,
    ScratchOwned<BE>: ScratchOwnedBorrow<BE>,
{
    let (k, k_flood) = (K.as_usize() as f64, k_flood.as_usize() as f64);
    let flood = PARTIES as f64 * SIGMA_FLOOD * SIGMA_FLOOD * (-2.0 * k_flood).exp2();
    let rest = (DEFAULT_SIGMA_XE * DEFAULT_SIGMA_XE + other) * (-2.0 * k).exp2();
    let lower = 0.5 * flood.log2() - 0.5;
    let upper = 0.5 * (flood + rest).log2() + 0.5;
    let noise: f64 = module.glwe_noise(ct, pt, sk, &mut scratch.borrow()).std().log2();
    assert!(noise >= lower && noise <= upper, "noise {noise} outside [{lower}, {upper}]");
}

/// The mask of `ct` alone, as the parties of a key switch receive it.
fn ciphertext_mask<BE>(module: &Module<BE>, ct: &GLWE<AlignedBuf, i64>) -> GLWEMask<AlignedBuf, i64>
where
    BE: HostBackend<OwnedBuf = AlignedBuf, ZnxWord = i64>,
    for<'a> BE::BufRef<'a>: HostDataRef,
    for<'a> BE::BufMut<'a>: HostDataMut,
    Module<BE>: VecZnxCopy<BE>,
{
    let mut mask = module.glwe_mask_alloc_from_infos(ct);
    for j in 0..ct.rank().as_usize() {
        module.vec_znx_copy(
            &mut vec_znx_backend_mut::<BE>(mask.data_mut()),
            j,
            &vec_znx_backend_ref::<BE>(ct.data()),
            j + 1,
        );
    }
    mask
}
