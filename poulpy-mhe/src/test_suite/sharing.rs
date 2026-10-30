//! Encryption to shares and shares to encryption over three parties: the
//! parties' shares sum to the plaintext plus input and flooding noise, and shares encrypted at a
//! larger precision decrypt to their integer sum.

use poulpy_core::{
    DEFAULT_BOUND_XE, DEFAULT_SIGMA_XE, GLWEAdd, GLWEDecrypt, GLWEEncryptSk, GLWENormalize, Noise,
    api::{GLWESecretPreparedFactory, GLWESecretSampling},
    layouts::{GLWE, GLWEAlloc, GLWELayout, GLWEPlaintext, GLWEPlaintextAlloc, Rank, TorusPrecision},
    test_suite::with_scratch,
};
use poulpy_hal::{
    AlignedBuf,
    api::{ScratchOwnedAlloc, ScratchOwnedBorrow, VecZnxAddScalarAssign},
    layouts::{HostBackend, HostDataMut, HostDataRef, Module, ScratchOwned},
    source::Source,
};

use super::fixtures::{
    BASE2K, K, K_OUT, LOG_MESSAGE, PARTIES, SEED_XE, SEEDS, assert_flooded_integers, assert_panics_with,
    assert_short_scratch_panics, bounded_integers, encrypt_integers, glwe_layout_at, ideal_secret, integer_plaintext,
    party_secrets, plaintext_integers, plaintext_state, secret_from_seed,
};
use crate::{
    MHEGLWEShareToEncShareAlloc,
    api::{MHEGLWEEncToShareProtocol, MHEGLWEShareToEncProtocol},
    layouts::MHEGLWEEncToShareShare,
    layouts::MHEGLWEEncToShareShareAlloc,
};

pub fn test_glwe_enc_to_share<BE>(module: &Module<BE>)
where
    BE: HostBackend<OwnedBuf = AlignedBuf, ZnxWord = i64>,
    for<'a> BE::BufRef<'a>: HostDataRef,
    for<'a> BE::BufMut<'a>: HostDataMut,
    Module<BE>: MHEGLWEEncToShareProtocol<BE>
        + GLWESecretSampling<BE>
        + GLWESecretPreparedFactory<BE>
        + GLWEEncryptSk<BE>
        + GLWEAdd<BE>
        + GLWENormalize<BE>
        + VecZnxAddScalarAssign<BE>,
    ScratchOwned<BE>: ScratchOwnedAlloc<BE> + ScratchOwnedBorrow<BE>,
{
    // Include a limb-aligned precision so normalization leaves nonzero carries.
    for k in [K, TorusPrecision(3 * BASE2K.0)] {
        for sigma in [1024.0, 4096.0] {
            let layout = glwe_layout_at(module, k);
            let flood = Noise::Gaussian { sigma };
            let parties = party_secrets(module);
            let mut scratch: ScratchOwned<BE> = ScratchOwned::alloc(module.glwe_encrypt_sk_tmp_bytes(&layout));
            let m = bounded_integers(module.n(), LOG_MESSAGE, [30u8; 32]);
            let ct = encrypt_integers(module, &layout, &m, &ideal_secret(module, &parties), &mut scratch);

            let mut secrets: Vec<GLWEPlaintext<AlignedBuf, i64>> = (0..PARTIES)
                .map(|_| module.glwe_plaintext_alloc_from_infos(&layout))
                .collect();
            let mut acc = module.mhe_glwe_enc_to_share_share_alloc_from_infos(&layout);
            let mut public = module.mhe_glwe_enc_to_share_share_alloc_from_infos(&layout);
            for (i, ((_, sk), secret)) in parties.iter().zip(secrets.iter_mut()).enumerate() {
                let dst = if i == 0 { &mut acc } else { &mut public };
                dst.inner.set_canonical(false);
                let mut source_xm = Source::new([60 + i as u8; 32]);
                let mut source_smudge = Source::new([90 + i as u8; 32]);
                let gen_bytes = module.mhe_glwe_enc_to_share_share_gen_tmp_bytes(&layout);
                if i == 0 {
                    let (public_before, secret_before) = (dst.clone(), plaintext_state(secret));
                    assert_short_scratch_panics::<BE>(gen_bytes, "invalid share: insufficient scratch", |s| {
                        module.mhe_glwe_enc_to_share_share_gen(dst, secret, &ct, sk, flood, &mut source_xm, &mut source_smudge, s)
                    });
                    assert!(*dst == public_before && plaintext_state(secret) == secret_before);
                    assert_eq!(source_xm.next_i64(), Source::new([60u8; 32]).next_i64());
                    assert_eq!(source_smudge.next_i64(), Source::new([90u8; 32]).next_i64());
                    (source_xm, source_smudge) = (Source::new([60u8; 32]), Source::new([90u8; 32]));
                }
                // The inner product with the party's secret would be left in the scratch.
                poulpy_core::test_suite::assert_wipes_scratch::<BE>(gen_bytes, |scratch| {
                    module.mhe_glwe_enc_to_share_share_gen(
                        dst,
                        secret,
                        &ct,
                        sk,
                        flood,
                        &mut source_xm,
                        &mut source_smudge,
                        scratch,
                    )
                });
                assert!(dst.inner.is_canonical());
                if i > 0 {
                    module.mhe_glwe_enc_to_share_share_aggregate(&mut acc, &public);
                }
                super::fixtures::assert_noise_components(&acc, &[(i + 1) as f64 * sigma * sigma]);
            }
            // The masks span the whole torus at the share's precision, top bits and bottom bit.
            for secret in &secrets[1..] {
                let mask = plaintext_integers(secret);
                assert!(mask.iter().any(|x| x.abs() >= 1 << (k.as_usize() - 3)));
                assert!(mask.iter().any(|x| x & 1 == 1));
            }

            let finalize_bytes = module.mhe_glwe_enc_to_share_share_finalize_tmp_bytes();
            let before = plaintext_state(&secrets[0]);
            assert_short_scratch_panics::<BE>(finalize_bytes, "invalid finalization: insufficient scratch", |s| {
                module.mhe_glwe_enc_to_share_share_finalize(&mut secrets[0], &ct, &acc, s)
            });
            assert!(plaintext_state(&secrets[0]) == before);
            // Normalization can leave carries depending on the private share in scratch.
            poulpy_core::test_suite::assert_wipes_scratch::<BE>(finalize_bytes, |scratch| {
                module.mhe_glwe_enc_to_share_share_finalize(&mut secrets[0], &ct, &acc, scratch)
            });
            let mut sum = vec![0i64; module.n()];
            for secret in &secrets {
                for (s, x) in sum.iter_mut().zip(plaintext_integers(secret)) {
                    *s = wrap(*s + x, k.as_usize());
                }
            }
            let bound = (DEFAULT_BOUND_XE + PARTIES as f64 * 6.0 * sigma).ceil() as i64 + PARTIES as i64;
            assert_flooded_integers(&sum, &m, sigma, DEFAULT_SIGMA_XE.powi(2), bound);
        }
    }
}

pub fn test_glwe_share_to_enc<BE>(module: &Module<BE>)
where
    BE: HostBackend<OwnedBuf = AlignedBuf, ZnxWord = i64>,
    for<'a> BE::BufRef<'a>: HostDataRef,
    for<'a> BE::BufMut<'a>: HostDataMut,
    Module<BE>: MHEGLWEShareToEncShareAlloc<BE>
        + MHEGLWEShareToEncProtocol<BE>
        + GLWESecretSampling<BE>
        + GLWESecretPreparedFactory<BE>
        + GLWEDecrypt<BE>
        + VecZnxAddScalarAssign<BE>,
    ScratchOwned<BE>: ScratchOwnedAlloc<BE> + ScratchOwnedBorrow<BE>,
{
    let in_layout = glwe_layout_at(module, K);
    let out_layout = glwe_layout_at(module, K_OUT);
    // Encryption noise reaches the output grid even when raising the plaintext precision.
    {
        let parties = party_secrets(module);
        let mut scratch: ScratchOwned<BE> = ScratchOwned::alloc(module.glwe_decrypt_tmp_bytes(&out_layout));
        let shares: Vec<Vec<i64>> = (0..PARTIES)
            .map(|i| bounded_integers(module.n(), K.as_usize(), [70 + i as u8; 32]))
            .collect();

        let mut acc = module.mhe_glwe_share_to_enc_share_alloc_from_infos(&out_layout);
        let mut share = module.mhe_glwe_share_to_enc_share_alloc_from_infos(&out_layout);
        for (i, ((_, sk), data)) in parties.iter().zip(&shares).enumerate() {
            let dst = if i == 0 { &mut acc } else { &mut share };
            let secret = integer_plaintext(module, &in_layout, data);
            let mut source_xe = Source::new([80 + i as u8; 32]);
            let gen_bytes = module.mhe_glwe_share_to_enc_share_gen_tmp_bytes(&out_layout, &in_layout);
            if i == 0 {
                let before = dst.clone();
                assert_short_scratch_panics::<BE>(gen_bytes, "invalid share: insufficient scratch", |s| {
                    module.mhe_glwe_share_to_enc_share_gen(dst, &secret, sk, SEEDS[0], &mut source_xe, s)
                });
                assert!(*dst == before);
                assert_eq!(source_xe.next_i64(), Source::new([80u8; 32]).next_i64());
                source_xe = Source::new([80u8; 32]);
            }
            // The party's raised share would be left in the scratch.
            poulpy_core::test_suite::assert_wipes_scratch::<BE>(gen_bytes, |scratch| {
                module.mhe_glwe_share_to_enc_share_gen(dst, &secret, sk, SEEDS[0], &mut source_xe, scratch)
            });
            if i > 0 {
                module.mhe_glwe_share_to_enc_share_aggregate(&mut acc, &share);
            }
        }

        let mut ct: GLWE<AlignedBuf, i64> = module.glwe_alloc_from_infos(&out_layout);
        let finalize_bytes = module.mhe_glwe_share_to_enc_share_finalize_tmp_bytes();
        let before = ct.clone();
        assert_short_scratch_panics::<BE>(finalize_bytes, "invalid finalization: insufficient scratch", |s| {
            module.mhe_glwe_share_to_enc_share_finalize(&mut ct, &acc, s)
        });
        assert!(ct == before);
        with_scratch::<BE, _>(finalize_bytes, |s| {
            module.mhe_glwe_share_to_enc_share_finalize(&mut ct, &acc, s)
        });
        super::fixtures::assert_collective_metadata(&ct, PARTIES);
        super::fixtures::assert_fresh_noise(&ct, PARTIES as f64 * DEFAULT_SIGMA_XE.powi(2), K_OUT);
        // The shares add up modulo 1; the output's extra precision only carries the noise.
        let want: Vec<i64> = (0..module.n())
            .map(|j| shares.iter().fold(0, |sum, s| wrap(sum + s[j], K.as_usize())))
            .collect();
        let mut pt: GLWEPlaintext<AlignedBuf, i64> = module.glwe_plaintext_alloc_from_infos(&in_layout);
        module.glwe_decrypt(&ct, &mut pt, &ideal_secret(module, &parties), &mut scratch.borrow());
        for (got, want) in plaintext_integers(&pt).iter().zip(&want) {
            assert!(wrap(got - want, K.as_usize()).abs() <= 1, "decrypted {got}, want {want}");
        }
    }
}

/// A share more precise than the output panics.
pub fn test_mhe_glwe_share_to_enc_share_gen_rejects_precision<BE>(module: &Module<BE>)
where
    BE: HostBackend<OwnedBuf = AlignedBuf, ZnxWord = i64>,
    Module<BE>:
        MHEGLWEShareToEncShareAlloc<BE> + MHEGLWEShareToEncProtocol<BE> + GLWESecretSampling<BE> + GLWESecretPreparedFactory<BE>,
    ScratchOwned<BE>: ScratchOwnedAlloc<BE> + ScratchOwnedBorrow<BE>,
{
    let layout = glwe_layout_at(module, K);
    let secret_layout = glwe_layout_at(module, K_OUT);
    let (_, sk) = secret_from_seed(module, [100u8; 32]);
    let secret: GLWEPlaintext<AlignedBuf, i64> = module.glwe_plaintext_alloc_from_infos(&secret_layout);
    let mut res = module.mhe_glwe_share_to_enc_share_alloc_from_infos(&layout);

    let before = res.clone();
    let mut source_xe = Source::new(SEED_XE);
    assert_panics_with("invalid share: secret share more precise than the output", || {
        with_scratch::<BE, _>(
            module.mhe_glwe_share_to_enc_share_gen_tmp_bytes(&layout, &secret_layout),
            |s| module.mhe_glwe_share_to_enc_share_gen(&mut res, &secret, &sk, SEEDS[0], &mut source_xe, s),
        )
    });
    assert!(res == before);
    assert_eq!(source_xe.next_i64(), Source::new(SEED_XE).next_i64());
}

/// A share less precise than the ciphertext panics.
pub fn test_mhe_glwe_enc_to_share_share_gen_rejects_precision<BE>(module: &Module<BE>)
where
    BE: HostBackend<OwnedBuf = AlignedBuf, ZnxWord = i64>,
    Module<BE>: MHEGLWEEncToShareProtocol<BE> + GLWESecretSampling<BE> + GLWESecretPreparedFactory<BE>,
    ScratchOwned<BE>: ScratchOwnedAlloc<BE> + ScratchOwnedBorrow<BE>,
{
    let ct_layout = glwe_layout_at(module, K);
    let secret_layout = glwe_layout_at(module, K_OUT);
    let (_, sk) = secret_from_seed(module, [100u8; 32]);
    let ct: GLWE<AlignedBuf, i64> = module.glwe_alloc_from_infos(&ct_layout);
    let mut public = MHEGLWEEncToShareShare {
        inner: module.glwe_alloc_from_infos(&GLWELayout {
            rank: Rank(0),
            ..ct_layout
        }),
    };
    let mut secret: GLWEPlaintext<AlignedBuf, i64> = module.glwe_plaintext_alloc_from_infos(&secret_layout);
    let public_before = public.inner.clone();
    let secret_before = plaintext_state(&secret);
    let mut source_mask = Source::new([60u8; 32]);
    let mut source_xe = Source::new(SEED_XE);
    assert_panics_with("invalid share: share and mask layouts differ", || {
        with_scratch::<BE, _>(module.mhe_glwe_enc_to_share_share_gen_tmp_bytes(&ct_layout), |s| {
            module.mhe_glwe_enc_to_share_share_gen(
                &mut public,
                &mut secret,
                &ct,
                &sk,
                Noise::Gaussian { sigma: 1024.0 },
                &mut source_mask,
                &mut source_xe,
                s,
            )
        })
    });
    assert!(public.inner == public_before);
    assert!(plaintext_state(&secret) == secret_before);
    assert_eq!(source_mask.next_i64(), Source::new([60u8; 32]).next_i64());
    assert_eq!(source_xe.next_i64(), Source::new(SEED_XE).next_i64());
}

/// Finalizing a share whose layout differs from the ciphertext's, or that is not of rank zero, panics.
pub fn test_mhe_glwe_enc_to_share_share_finalize_rejects_layout_mismatch<BE>(module: &Module<BE>)
where
    BE: HostBackend<OwnedBuf = AlignedBuf, ZnxWord = i64>,
    Module<BE>: MHEGLWEEncToShareProtocol<BE>,
    ScratchOwned<BE>: ScratchOwnedAlloc<BE> + ScratchOwnedBorrow<BE>,
{
    let ct_layout = glwe_layout_at(module, K);
    let ct: GLWE<AlignedBuf, i64> = module.glwe_alloc_from_infos(&ct_layout);
    for (share_layout, secret_k, expected) in [
        (
            GLWELayout {
                rank: Rank(0),
                ..ct_layout
            },
            K_OUT,
            "invalid finalization: ciphertext and share layouts differ",
        ),
        (ct_layout, K, "invalid finalization: additive shares must have rank zero"),
    ] {
        let share = MHEGLWEEncToShareShare {
            inner: module.glwe_alloc_from_infos(&share_layout),
        };
        let mut secret: GLWEPlaintext<AlignedBuf, i64> =
            module.glwe_plaintext_alloc_from_infos(&glwe_layout_at(module, secret_k));
        let before = plaintext_state(&secret);
        assert_panics_with(expected, || {
            with_scratch::<BE, _>(module.mhe_glwe_enc_to_share_share_finalize_tmp_bytes(), |s| {
                module.mhe_glwe_enc_to_share_share_finalize(&mut secret, &ct, &share, s)
            })
        });
        assert!(plaintext_state(&secret) == before);
    }
}

/// Generating a public share that is not of rank zero panics before any draw.
pub fn test_mhe_glwe_enc_to_share_share_gen_rejects_nonzero_rank<BE>(module: &Module<BE>)
where
    BE: HostBackend<OwnedBuf = AlignedBuf, ZnxWord = i64>,
    Module<BE>: MHEGLWEEncToShareProtocol<BE> + GLWESecretSampling<BE> + GLWESecretPreparedFactory<BE>,
    ScratchOwned<BE>: ScratchOwnedAlloc<BE> + ScratchOwnedBorrow<BE>,
{
    let layout = glwe_layout_at(module, K);
    let (_, sk) = secret_from_seed(module, [100u8; 32]);
    let ct: GLWE<AlignedBuf, i64> = module.glwe_alloc_from_infos(&layout);
    let mut public = MHEGLWEEncToShareShare {
        inner: module.glwe_alloc_from_infos(&layout),
    };
    let mut secret: GLWEPlaintext<AlignedBuf, i64> = module.glwe_plaintext_alloc_from_infos(&layout);
    let (public_before, secret_before) = (public.clone(), plaintext_state(&secret));
    let mut source_xm = Source::new([60u8; 32]);
    let mut source_smudge = Source::new(SEED_XE);
    assert_panics_with("invalid share: additive shares must have rank zero", || {
        with_scratch::<BE, _>(module.mhe_glwe_enc_to_share_share_gen_tmp_bytes(&layout), |s| {
            module.mhe_glwe_enc_to_share_share_gen(
                &mut public,
                &mut secret,
                &ct,
                &sk,
                Noise::Gaussian { sigma: 1024.0 },
                &mut source_xm,
                &mut source_smudge,
                s,
            )
        });
    });
    assert!(public == public_before && plaintext_state(&secret) == secret_before);
    assert_eq!(source_xm.next_i64(), Source::new([60u8; 32]).next_i64());
    assert_eq!(source_smudge.next_i64(), Source::new(SEED_XE).next_i64());
}

/// Generating from an additive share that is not of rank zero panics before any draw.
pub fn test_mhe_glwe_share_to_enc_share_gen_rejects_nonzero_rank<BE>(module: &Module<BE>)
where
    BE: HostBackend<OwnedBuf = AlignedBuf, ZnxWord = i64>,
    Module<BE>:
        MHEGLWEShareToEncShareAlloc<BE> + MHEGLWEShareToEncProtocol<BE> + GLWESecretSampling<BE> + GLWESecretPreparedFactory<BE>,
    ScratchOwned<BE>: ScratchOwnedAlloc<BE> + ScratchOwnedBorrow<BE>,
{
    let layout = glwe_layout_at(module, K);
    let (_, sk) = secret_from_seed(module, [100u8; 32]);
    let share: GLWE<AlignedBuf, i64> = module.glwe_alloc_from_infos(&layout);
    let mut res = module.mhe_glwe_share_to_enc_share_alloc_from_infos(&layout);
    let before = res.clone();
    let mut source_xe = Source::new(SEED_XE);
    assert_panics_with("invalid share: additive share must have rank zero", || {
        with_scratch::<BE, _>(module.mhe_glwe_share_to_enc_share_gen_tmp_bytes(&layout, &layout), |s| {
            module.mhe_glwe_share_to_enc_share_gen(&mut res, &share, &sk, SEEDS[0], &mut source_xe, s)
        });
    });
    assert!(res == before);
    assert_eq!(source_xe.next_i64(), Source::new(SEED_XE).next_i64());
}

/// Invalid smudging descriptors panic before any draw or write.
pub fn test_mhe_glwe_enc_to_share_share_gen_rejects_flood<BE>(module: &Module<BE>)
where
    BE: HostBackend<OwnedBuf = AlignedBuf, ZnxWord = i64>,
    Module<BE>: MHEGLWEEncToShareProtocol<BE> + GLWESecretSampling<BE> + GLWESecretPreparedFactory<BE>,
    ScratchOwned<BE>: ScratchOwnedAlloc<BE> + ScratchOwnedBorrow<BE>,
{
    let layout = glwe_layout_at(module, K);
    let (_, sk) = secret_from_seed(module, [100u8; 32]);
    let ct: GLWE<AlignedBuf, i64> = module.glwe_alloc_from_infos(&layout);
    let mut public = MHEGLWEEncToShareShare {
        inner: module.glwe_alloc_from_infos(&GLWELayout { rank: Rank(0), ..layout }),
    };
    let mut secret: GLWEPlaintext<AlignedBuf, i64> = module.glwe_plaintext_alloc_from_infos(&layout);
    let (public_before, secret_before) = (public.clone(), plaintext_state(&secret));
    let gen_bytes = module.mhe_glwe_enc_to_share_share_gen_tmp_bytes(&layout);
    for (noise, message) in [
        (
            Noise::Gaussian { sigma: 0.5 },
            "invalid noise: Gaussian sigma must be finite and at least 1",
        ),
        (
            Noise::Gaussian {
                sigma: 2.0f64.powi((K.as_usize()) as i32),
            },
            "invalid noise: Gaussian bound outside the precision",
        ),
        (
            Noise::Uniform { bits: K.as_usize() },
            "invalid noise: uniform width outside the precision",
        ),
    ] {
        let mut source_xm = Source::new([60u8; 32]);
        let mut source_smudge = Source::new(SEED_XE);
        assert_panics_with(message, || {
            with_scratch::<BE, _>(gen_bytes, |s| {
                module.mhe_glwe_enc_to_share_share_gen(
                    &mut public,
                    &mut secret,
                    &ct,
                    &sk,
                    noise,
                    &mut source_xm,
                    &mut source_smudge,
                    s,
                )
            });
        });
        assert!(public == public_before && plaintext_state(&secret) == secret_before);
        assert_eq!(source_xm.next_i64(), Source::new([60u8; 32]).next_i64());
        assert_eq!(source_smudge.next_i64(), Source::new(SEED_XE).next_i64());
    }
}

/// An encryption-to-shares generation over a flag-clear mask is rejected before any draw.
pub fn test_mhe_glwe_enc_to_share_share_gen_rejects_flag_clear_mask<BE>(module: &Module<BE>)
where
    BE: HostBackend<OwnedBuf = AlignedBuf, ZnxWord = i64>,
    Module<BE>: MHEGLWEEncToShareProtocol<BE> + GLWESecretSampling<BE> + GLWESecretPreparedFactory<BE>,
    ScratchOwned<BE>: ScratchOwnedAlloc<BE> + ScratchOwnedBorrow<BE>,
{
    let layout = glwe_layout_at(module, K);
    let (_, sk) = secret_from_seed(module, [100u8; 32]);
    let mut ct: GLWE<AlignedBuf, i64> = module.glwe_alloc_from_infos(&layout);
    ct.set_canonical(false);
    let mut public = MHEGLWEEncToShareShare {
        inner: module.glwe_alloc_from_infos(&GLWELayout { rank: Rank(0), ..layout }),
    };
    let mut secret: GLWEPlaintext<AlignedBuf, i64> = module.glwe_plaintext_alloc_from_infos(&layout);
    let (public_before, secret_before) = (public.clone(), plaintext_state(&secret));
    let mut source_xm = Source::new([60u8; 32]);
    let mut source_smudge = Source::new(SEED_XE);
    super::fixtures::assert_panics_with("invalid share: mask not canonical", || {
        with_scratch::<BE, _>(module.mhe_glwe_enc_to_share_share_gen_tmp_bytes(&layout), |s| {
            module.mhe_glwe_enc_to_share_share_gen(
                &mut public,
                &mut secret,
                &ct,
                &sk,
                Noise::Gaussian { sigma: 1024.0 },
                &mut source_xm,
                &mut source_smudge,
                s,
            )
        });
    });
    assert!(public == public_before && plaintext_state(&secret) == secret_before);
    assert_eq!(source_xm.next_i64(), Source::new([60u8; 32]).next_i64());
    assert_eq!(source_smudge.next_i64(), Source::new(SEED_XE).next_i64());
}

/// `x` reduced modulo `2^k` into `[-2^(k-1), 2^(k-1))`.
fn wrap(x: i64, k: usize) -> i64 {
    let shift = 64 - k as u32;
    (x << shift) >> shift
}
