//! Encryption to shares and shares to encryption over three parties: the
//! parties' shares sum to the plaintext plus input and flooding noise, and shares encrypted at a
//! larger precision decrypt to their integer sum.

use poulpy_core::{
    DEFAULT_BOUND_XE, DEFAULT_SIGMA_XE, GLWEAdd, GLWEDecrypt, GLWEEncryptSk, GLWENormalize,
    layouts::{
        GLWE, GLWELayout, GLWEPlaintext, GLWESecretPreparedFactory, GLWESecretSampling, ModuleCoreAlloc, Rank, TorusPrecision,
    },
};
use poulpy_hal::{
    AlignedBuf,
    api::{ScratchOwnedAlloc, ScratchOwnedBorrow, VecZnxAddScalarAssign},
    layouts::{HostBackend, HostDataMut, HostDataRef, Module, ScratchOwned},
    source::Source,
};

use super::fixtures::{
    BASE2K, K, K_OUT, LOG_MESSAGE, PARTIES, SEED_XE, SEEDS, assert_flooded_integers, bounded_integers, encrypt_integers,
    glwe_layout_at, ideal_secret, integer_flood_infos, integer_plaintext, party_secrets, plaintext_integers, secret_from_seed,
};
use crate::{
    api::{GLWEEncToShareMHEProtocol, GLWEShareToEncMHEProtocol},
    layouts::GLWEEncToShareShare,
    layouts::MHEModuleAlloc,
};

pub fn test_glwe_enc_to_share<BE>(module: &Module<BE>)
where
    BE: HostBackend<OwnedBuf = AlignedBuf, ZnxWord = i64>,
    for<'a> BE::BufRef<'a>: HostDataRef,
    for<'a> BE::BufMut<'a>: HostDataMut,
    Module<BE>: GLWEEncToShareMHEProtocol<BE>
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
            let flood = integer_flood_infos(sigma);
            let parties = party_secrets(module);
            let mut scratch: ScratchOwned<BE> = ScratchOwned::alloc(
                module
                    .glwe_encrypt_sk_tmp_bytes(&layout)
                    .max(module.mhe_glwe_enc_to_share_share_gen_tmp_bytes(&layout))
                    .max(module.mhe_glwe_enc_to_share_share_finalize_tmp_bytes()),
            );
            let m = bounded_integers(module.n(), LOG_MESSAGE, [30u8; 32]);
            let ct = encrypt_integers(module, &layout, &m, &ideal_secret(module, &parties), &mut scratch);

            let mut secrets: Vec<GLWEPlaintext<AlignedBuf, i64>> = (0..PARTIES)
                .map(|_| module.glwe_plaintext_alloc_from_infos(&layout))
                .collect();
            let mut acc = module.glwe_enc_to_share_share_alloc_from_infos(&layout);
            let mut public = module.glwe_enc_to_share_share_alloc_from_infos(&layout);
            for (i, ((_, sk), secret)) in parties.iter().zip(secrets.iter_mut()).enumerate() {
                let dst = if i == 0 { &mut acc } else { &mut public };
                dst.inner.set_canonical(false);
                let mut source_xm = Source::new([60 + i as u8; 32]);
                let mut source_smudge = Source::new([90 + i as u8; 32]);
                // The inner product with the party's secret would be left in the scratch.
                poulpy_core::test_suite::assert_wipes_scratch::<BE>(
                    module.mhe_glwe_enc_to_share_share_gen_tmp_bytes(&layout),
                    |scratch| {
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
                    },
                );
                assert!(dst.inner.is_canonical());
                if i > 0 {
                    module.mhe_glwe_enc_to_share_share_aggregate(&mut acc, &public);
                }
            }
            // The masks span the whole torus at the share's precision, top bits and bottom bit.
            for secret in &secrets[1..] {
                let mask = plaintext_integers(secret);
                assert!(mask.iter().any(|x| x.abs() >= 1 << (k.as_usize() - 3)));
                assert!(mask.iter().any(|x| x & 1 == 1));
            }

            // Normalization can leave carries depending on the private share in scratch.
            poulpy_core::test_suite::assert_wipes_scratch::<BE>(
                module.mhe_glwe_enc_to_share_share_finalize_tmp_bytes(),
                |scratch| module.mhe_glwe_enc_to_share_share_finalize(&mut secrets[0], &ct, &acc, scratch),
            );
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
    Module<BE>: MHEModuleAlloc<BE>
        + GLWEShareToEncMHEProtocol<BE>
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
        let mut scratch: ScratchOwned<BE> = ScratchOwned::alloc(
            module
                .mhe_glwe_share_to_enc_share_gen_tmp_bytes(&out_layout, &in_layout)
                .max(module.mhe_glwe_share_to_enc_share_finalize_tmp_bytes())
                .max(module.glwe_decrypt_tmp_bytes(&out_layout)),
        );
        let shares: Vec<Vec<i64>> = (0..PARTIES)
            .map(|i| bounded_integers(module.n(), K.as_usize(), [70 + i as u8; 32]))
            .collect();

        let mut acc = module.glwe_share_to_enc_share_alloc_from_infos(&out_layout);
        let mut share = module.glwe_share_to_enc_share_alloc_from_infos(&out_layout);
        for (i, ((_, sk), data)) in parties.iter().zip(&shares).enumerate() {
            let dst = if i == 0 { &mut acc } else { &mut share };
            let secret = integer_plaintext(module, &in_layout, data);
            let mut source_xe = Source::new([80 + i as u8; 32]);
            // The party's raised share would be left in the scratch.
            poulpy_core::test_suite::assert_wipes_scratch::<BE>(
                module.mhe_glwe_share_to_enc_share_gen_tmp_bytes(&out_layout, &in_layout),
                |scratch| module.mhe_glwe_share_to_enc_share_gen(dst, &secret, sk, SEEDS[0], &mut source_xe, scratch),
            );
            if i > 0 {
                module.mhe_glwe_share_to_enc_share_aggregate(&mut acc, &share);
            }
        }

        let mut ct: GLWE<AlignedBuf, i64> = module.glwe_alloc_from_infos(&out_layout);
        module.mhe_glwe_share_to_enc_share_finalize(&mut ct, &acc, &mut scratch.borrow());
        super::fixtures::assert_collective_metadata(&ct, PARTIES);
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
pub fn test_glwe_share_to_enc_precision<BE>(module: &Module<BE>)
where
    BE: HostBackend<OwnedBuf = AlignedBuf, ZnxWord = i64>,
    Module<BE>: MHEModuleAlloc<BE> + GLWEShareToEncMHEProtocol<BE> + GLWESecretSampling<BE> + GLWESecretPreparedFactory<BE>,
    ScratchOwned<BE>: ScratchOwnedAlloc<BE> + ScratchOwnedBorrow<BE>,
{
    let layout = glwe_layout_at(module, K);
    let secret_layout = glwe_layout_at(module, K_OUT);
    let (_, sk) = secret_from_seed(module, [100u8; 32]);
    let secret: GLWEPlaintext<AlignedBuf, i64> = module.glwe_plaintext_alloc_from_infos(&secret_layout);
    let mut res = module.glwe_share_to_enc_share_alloc_from_infos(&layout);

    let mut scratch: ScratchOwned<BE> =
        ScratchOwned::alloc(module.mhe_glwe_share_to_enc_share_gen_tmp_bytes(&layout, &secret_layout));
    module.mhe_glwe_share_to_enc_share_gen(
        &mut res,
        &secret,
        &sk,
        SEEDS[0],
        &mut Source::new(SEED_XE),
        &mut scratch.borrow(),
    );
}

/// A share less precise than the ciphertext panics.
pub fn test_glwe_enc_to_share_precision<BE>(module: &Module<BE>)
where
    BE: HostBackend<OwnedBuf = AlignedBuf, ZnxWord = i64>,
    Module<BE>: GLWEEncToShareMHEProtocol<BE> + GLWESecretSampling<BE> + GLWESecretPreparedFactory<BE>,
    ScratchOwned<BE>: ScratchOwnedAlloc<BE> + ScratchOwnedBorrow<BE>,
{
    let ct_layout = glwe_layout_at(module, K);
    let secret_layout = glwe_layout_at(module, K_OUT);
    let (_, sk) = secret_from_seed(module, [100u8; 32]);
    let ct: GLWE<AlignedBuf, i64> = module.glwe_alloc_from_infos(&ct_layout);
    let mut public = GLWEEncToShareShare {
        inner: module.glwe_alloc_from_infos(&GLWELayout {
            rank: Rank(0),
            ..ct_layout
        }),
    };
    let mut secret: GLWEPlaintext<AlignedBuf, i64> = module.glwe_plaintext_alloc_from_infos(&secret_layout);
    let mut scratch: ScratchOwned<BE> = ScratchOwned::alloc(module.mhe_glwe_enc_to_share_share_gen_tmp_bytes(&ct_layout));
    module.mhe_glwe_enc_to_share_share_gen(
        &mut public,
        &mut secret,
        &ct,
        &sk,
        integer_flood_infos(1024.0),
        &mut Source::new([60u8; 32]),
        &mut Source::new(SEED_XE),
        &mut scratch.borrow(),
    );
}

/// Finalizing a share whose layout differs from the ciphertext's panics.
pub fn test_glwe_enc_to_share_finalize_layout_mismatch<BE>(module: &Module<BE>)
where
    BE: HostBackend<OwnedBuf = AlignedBuf, ZnxWord = i64>,
    Module<BE>: GLWEEncToShareMHEProtocol<BE>,
    ScratchOwned<BE>: ScratchOwnedAlloc<BE> + ScratchOwnedBorrow<BE>,
{
    let ct_layout = glwe_layout_at(module, K);
    let secret_layout = GLWELayout {
        rank: Rank(0),
        ..ct_layout
    };
    let ct: GLWE<AlignedBuf, i64> = module.glwe_alloc_from_infos(&ct_layout);
    let share = GLWEEncToShareShare {
        inner: module.glwe_alloc_from_infos(&secret_layout),
    };
    let mut secret: GLWEPlaintext<AlignedBuf, i64> = module.glwe_plaintext_alloc_from_infos(&glwe_layout_at(module, K_OUT));
    let mut scratch: ScratchOwned<BE> = ScratchOwned::alloc(module.mhe_glwe_enc_to_share_share_finalize_tmp_bytes());
    module.mhe_glwe_enc_to_share_share_finalize(&mut secret, &ct, &share, &mut scratch.borrow());
}

/// Plaintext shares must be rank zero at both conversion boundaries.
pub fn test_glwe_sharing_layout_guards<BE>(module: &Module<BE>)
where
    BE: HostBackend<OwnedBuf = AlignedBuf, ZnxWord = i64>,
    Module<BE>: MHEModuleAlloc<BE>
        + GLWEEncToShareMHEProtocol<BE>
        + GLWEShareToEncMHEProtocol<BE>
        + GLWESecretSampling<BE>
        + GLWESecretPreparedFactory<BE>,
    ScratchOwned<BE>: ScratchOwnedAlloc<BE> + ScratchOwnedBorrow<BE>,
{
    let layout = glwe_layout_at(module, K);
    let (_, sk) = secret_from_seed(module, [100u8; 32]);
    let ct: GLWE<AlignedBuf, i64> = module.glwe_alloc_from_infos(&layout);
    let mut public = GLWEEncToShareShare {
        inner: module.glwe_alloc_from_infos(&layout),
    };
    let mut secret: GLWEPlaintext<AlignedBuf, i64> = module.glwe_plaintext_alloc_from_infos(&layout);
    let mut scratch: ScratchOwned<BE> = ScratchOwned::alloc(
        module
            .mhe_glwe_enc_to_share_share_gen_tmp_bytes(&layout)
            .max(module.mhe_glwe_share_to_enc_share_gen_tmp_bytes(&layout, &layout))
            .max(module.mhe_glwe_enc_to_share_share_finalize_tmp_bytes()),
    );
    super::fixtures::assert_panics_with("invalid share: additive shares must have rank zero", || {
        module.mhe_glwe_enc_to_share_share_gen(
            &mut public,
            &mut secret,
            &ct,
            &sk,
            integer_flood_infos(1024.0),
            &mut Source::new([60u8; 32]),
            &mut Source::new(SEED_XE),
            &mut scratch.borrow(),
        );
    });
    super::fixtures::assert_panics_with("invalid finalization: additive shares must have rank zero", || {
        module.mhe_glwe_enc_to_share_share_finalize(&mut secret, &ct, &public, &mut scratch.borrow());
    });
    let mut pat = module.glwe_share_to_enc_share_alloc_from_infos(&layout);
    super::fixtures::assert_panics_with("invalid share: additive share must have rank zero", || {
        module.mhe_glwe_share_to_enc_share_gen(&mut pat, &ct, &sk, SEEDS[0], &mut Source::new(SEED_XE), &mut scratch.borrow());
    });
}

/// Reject invalid smudging descriptors before touching the shares.
pub fn test_glwe_enc_to_share_flood_guards<BE>(module: &Module<BE>)
where
    BE: HostBackend<OwnedBuf = AlignedBuf, ZnxWord = i64>,
    Module<BE>: GLWEEncToShareMHEProtocol<BE> + GLWESecretSampling<BE> + GLWESecretPreparedFactory<BE>,
    ScratchOwned<BE>: ScratchOwnedAlloc<BE> + ScratchOwnedBorrow<BE>,
{
    let layout = glwe_layout_at(module, K);
    let (_, sk) = secret_from_seed(module, [100u8; 32]);
    let ct: GLWE<AlignedBuf, i64> = module.glwe_alloc_from_infos(&layout);
    let mut public = GLWEEncToShareShare {
        inner: module.glwe_alloc_from_infos(&GLWELayout { rank: Rank(0), ..layout }),
    };
    let mut secret: GLWEPlaintext<AlignedBuf, i64> = module.glwe_plaintext_alloc_from_infos(&layout);
    let mut scratch: ScratchOwned<BE> = ScratchOwned::alloc(module.mhe_glwe_enc_to_share_share_gen_tmp_bytes(&layout));
    for (noise, message) in [
        (
            poulpy_core::Noise::Gaussian {
                sigma: 1024.0,
                cutoff_factor: 0,
            },
            "invalid noise: Gaussian cutoff factor must be positive",
        ),
        (
            poulpy_core::Noise::Gaussian {
                sigma: 2.0f64.powi((K.as_usize()) as i32),
                cutoff_factor: 6,
            },
            "invalid noise: Gaussian bound outside the precision",
        ),
        (
            poulpy_core::Noise::Uniform { bits: K.as_usize() },
            "invalid noise: uniform width outside the precision",
        ),
    ] {
        let mut source_xm = Source::new([60u8; 32]);
        let mut source_smudge = Source::new(SEED_XE);
        super::fixtures::assert_panics_with(message, || {
            module.mhe_glwe_enc_to_share_share_gen(
                &mut public,
                &mut secret,
                &ct,
                &sk,
                noise,
                &mut source_xm,
                &mut source_smudge,
                &mut scratch.borrow(),
            );
        });
        assert_eq!(source_xm.next_i64(), Source::new([60u8; 32]).next_i64());
        assert_eq!(source_smudge.next_i64(), Source::new(SEED_XE).next_i64());
    }
}

/// `x` reduced modulo `2^k` into `[-2^(k-1), 2^(k-1))`.
fn wrap(x: i64, k: usize) -> i64 {
    let shift = 64 - k as u32;
    (x << shift) >> shift
}
