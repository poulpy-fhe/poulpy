//! Shamir thresholdization: every party shares its secret, every party
//! aggregates the shares it receives, and any active set of at least the
//! threshold combines its shares into additive shares of the secrets' sum.

use poulpy_core::{
    DEFAULT_SIGMA_XE, EncryptionLayout, GLWEAdd, GLWEEncryptSk, GLWENoise, GLWENormalize, SmudgingNoise,
    layouts::{
        Base2K, GLWE, GLWELayout, GLWEPublicKeyPreparedFactory, GLWESecret, GLWESecretPreparedFactory, GLWESecretSampling,
        ModuleCoreAlloc, Rank, TorusPrecision,
    },
};
use poulpy_hal::{
    AlignedBuf,
    api::{ScratchOwnedAlloc, ScratchOwnedBorrow, VecZnxAddScalarAssign},
    layouts::{HostBackend, HostDataMut, HostDataRef, Module, ScratchOwned, ZnxView},
    source::Source,
};

use super::{
    fixtures::{
        BASE2K, K, LOG_MESSAGE, PARTIES, RANK, Secret, bounded_integers, collective_public_key, encrypt_integers, glwe_layout_at,
        ideal_secret, integer_plaintext, secret_from_seed,
    },
    keyswitch::{assert_flooded_noise, flood_infos},
};
use crate::{
    api::{
        GLWEPublicKeyMHEProtocol, GLWEShamirMHEProtocol, GLWEThresholdKeyswitchMHEProtocol,
        GLWEThresholdPublicKeyswitchMHEProtocol, GLWEWideSecretPrepare,
    },
    layouts::{GLWEShamirLayout, GLWEShamirShareOwned, GLWEWideSecret, GLWEWideSecretOwned, MHEModuleAlloc},
};

/// A precision that is not a multiple of the base: the last limb is partial.
const K_SHARE: TorusPrecision = TorusPrecision(73);
const B_SHARE: Base2K = Base2K(13);

pub fn test_glwe_threshold<BE>(module: &Module<BE>)
where
    BE: HostBackend<OwnedBuf = AlignedBuf, ZnxWord = i64>,
    for<'a> BE::BufRef<'a>: HostDataRef,
    for<'a> BE::BufMut<'a>: HostDataMut,
    Module<BE>: MHEModuleAlloc<BE>
        + GLWEShamirMHEProtocol<BE>
        + GLWESecretSampling<BE>
        + GLWESecretPreparedFactory<BE>
        + GLWEAdd<BE>
        + GLWENormalize<BE>,
    ScratchOwned<BE>: ScratchOwnedAlloc<BE> + ScratchOwnedBorrow<BE>,
{
    threshold_combine(module, 5, 3, &[&[1, 2, 3], &[2, 4, 5], &[1, 3, 4, 5]]);
}

pub fn test_glwe_threshold_all_active<BE>(module: &Module<BE>)
where
    BE: HostBackend<OwnedBuf = AlignedBuf, ZnxWord = i64>,
    for<'a> BE::BufRef<'a>: HostDataRef,
    for<'a> BE::BufMut<'a>: HostDataMut,
    Module<BE>: MHEModuleAlloc<BE>
        + GLWEShamirMHEProtocol<BE>
        + GLWESecretSampling<BE>
        + GLWESecretPreparedFactory<BE>
        + GLWEAdd<BE>
        + GLWENormalize<BE>,
    ScratchOwned<BE>: ScratchOwnedAlloc<BE> + ScratchOwnedBorrow<BE>,
{
    threshold_combine(module, 3, 3, &[&[1, 2, 3]]);
}

const ACTIVES: [u32; PARTIES] = [1, 3, 5];

pub fn test_glwe_threshold_decrypt<BE>(module: &Module<BE>)
where
    BE: HostBackend<OwnedBuf = AlignedBuf, ZnxWord = i64>,
    for<'a> BE::BufRef<'a>: HostDataRef,
    for<'a> BE::BufMut<'a>: HostDataMut,
    Module<BE>: MHEModuleAlloc<BE>
        + GLWEShamirMHEProtocol<BE>
        + GLWEWideSecretPrepare<BE>
        + GLWEThresholdKeyswitchMHEProtocol<BE>
        + GLWESecretSampling<BE>
        + GLWESecretPreparedFactory<BE>
        + GLWEEncryptSk<BE>
        + GLWENoise<BE>
        + GLWEAdd<BE>
        + GLWENormalize<BE>
        + VecZnxAddScalarAssign<BE>,
    ScratchOwned<BE>: ScratchOwnedAlloc<BE> + ScratchOwnedBorrow<BE>,
{
    let zero = || {
        let sk: GLWESecret<AlignedBuf, i64> = module.glwe_secret_alloc(RANK);
        let mut prepared = module.glwe_secret_prepared_alloc(RANK);
        module.glwe_secret_prepare(&mut prepared, &sk);
        (sk, prepared)
    };
    threshold_keyswitch(module, (0..PARTIES).map(|_| zero()).collect());
}

pub fn test_glwe_threshold_keyswitch<BE>(module: &Module<BE>)
where
    BE: HostBackend<OwnedBuf = AlignedBuf, ZnxWord = i64>,
    for<'a> BE::BufRef<'a>: HostDataRef,
    for<'a> BE::BufMut<'a>: HostDataMut,
    Module<BE>: MHEModuleAlloc<BE>
        + GLWEShamirMHEProtocol<BE>
        + GLWEWideSecretPrepare<BE>
        + GLWEThresholdKeyswitchMHEProtocol<BE>
        + GLWESecretSampling<BE>
        + GLWESecretPreparedFactory<BE>
        + GLWEEncryptSk<BE>
        + GLWENoise<BE>
        + GLWEAdd<BE>
        + GLWENormalize<BE>
        + VecZnxAddScalarAssign<BE>,
    ScratchOwned<BE>: ScratchOwnedAlloc<BE> + ScratchOwnedBorrow<BE>,
{
    threshold_keyswitch(
        module,
        (0..PARTIES).map(|i| secret_from_seed(module, [120 + i as u8; 32])).collect(),
    );
}

pub fn test_glwe_threshold_public_keyswitch<BE>(module: &Module<BE>)
where
    BE: HostBackend<OwnedBuf = AlignedBuf, ZnxWord = i64>,
    for<'a> BE::BufRef<'a>: HostDataRef,
    for<'a> BE::BufMut<'a>: HostDataMut,
    Module<BE>: MHEModuleAlloc<BE>
        + GLWEShamirMHEProtocol<BE>
        + GLWEWideSecretPrepare<BE>
        + GLWEThresholdPublicKeyswitchMHEProtocol<BE>
        + GLWEPublicKeyMHEProtocol<BE>
        + GLWEPublicKeyPreparedFactory<BE>
        + GLWESecretSampling<BE>
        + GLWESecretPreparedFactory<BE>
        + GLWEEncryptSk<BE>
        + GLWENoise<BE>
        + GLWEAdd<BE>
        + GLWENormalize<BE>
        + VecZnxAddScalarAssign<BE>,
    ScratchOwned<BE>: ScratchOwnedAlloc<BE> + ScratchOwnedBorrow<BE>,
{
    let ct_layout = glwe_layout_at(module, K);
    // A public key more precise than the share exercises the share scratch query for real.
    let pk_layout = GLWELayout {
        k: TorusPrecision(K.0 + BASE2K.0),
        ..ct_layout
    };
    let enc_infos = EncryptionLayout::new_from_default_sigma(ct_layout).unwrap();
    let out: Vec<Secret<BE>> = (0..PARTIES).map(|i| secret_from_seed(module, [120 + i as u8; 32])).collect();
    let pk_out = collective_public_key(module, &out, &pk_layout);
    let (secrets, layout, wide) = combined_shares(module);
    let mut scratch: ScratchOwned<BE> = ScratchOwned::alloc(
        module
            .glwe_encrypt_sk_tmp_bytes(&ct_layout)
            .max(module.mhe_glwe_wide_secret_prepare_tmp_bytes(&layout))
            .max(module.mhe_glwe_threshold_public_keyswitch_share_finalize_tmp_bytes())
            .max(module.glwe_noise_tmp_bytes(&ct_layout)),
    );
    let mut share_scratch: ScratchOwned<BE> =
        ScratchOwned::alloc(module.mhe_glwe_threshold_public_keyswitch_share_gen_tmp_bytes(&ct_layout, &ct_layout, &pk_layout));
    let m = bounded_integers(module.n(), LOG_MESSAGE, [30u8; 32]);
    let ct = encrypt_integers(module, &ct_layout, &m, &ideal_secret(module, &secrets), &mut scratch);

    let flood = flood_infos(ct_layout);
    let mut acc = module.glwe_public_keyswitch_share_alloc_from_infos(&ct_layout);
    let mut share = module.glwe_public_keyswitch_share_alloc_from_infos(&ct_layout);
    for (i, sigma) in wide.iter().enumerate() {
        let mut prepared = module.glwe_wide_secret_prepared_alloc(&layout);
        module.mhe_glwe_wide_secret_prepare(&mut prepared, sigma, &mut scratch.borrow());
        let dst = if i == 0 { &mut acc } else { &mut share };
        dst.inner.set_canonical(false);
        module.mhe_glwe_threshold_public_keyswitch_share_gen(
            dst,
            &ct,
            &prepared,
            &pk_out,
            &flood,
            &enc_infos,
            128,
            &mut Source::new([20 + i as u8; 32]),
            &mut Source::new([10 + i as u8; 32]),
            &mut share_scratch.borrow(),
        );
        assert!(dst.inner.is_canonical());
        if i > 0 {
            module.mhe_glwe_threshold_public_keyswitch_share_aggregate(&mut acc, &share);
        }
    }

    let mut res: GLWE<AlignedBuf, i64> = module.glwe_alloc_from_infos(&ct_layout);
    module.mhe_glwe_threshold_public_keyswitch_share_finalize(&mut res, &ct, &acc, &mut scratch.borrow());
    // Each party's pk encryption adds 2 * rank * n * 0.5 * PARTIES * sigma^2, as in the pk test.
    let (n, rank) = (module.n() as f64, RANK.as_usize() as f64);
    let pk_noise = PARTIES as f64 * 2.0 * rank * n * 0.5 * PARTIES as f64 * DEFAULT_SIGMA_XE * DEFAULT_SIGMA_XE;
    let pt = integer_plaintext(module, &ct_layout, &m);
    assert_flooded_noise(module, &res, &pt, &ideal_secret(module, &out), pk_noise, &mut scratch);
}

/// Sharing with a wide secret less precise than the ciphertext panics.
pub fn test_glwe_threshold_keyswitch_precision<BE>(module: &Module<BE>)
where
    BE: HostBackend<OwnedBuf = AlignedBuf, ZnxWord = i64>,
    Module<BE>:
        MHEModuleAlloc<BE> + GLWEThresholdKeyswitchMHEProtocol<BE> + GLWESecretSampling<BE> + GLWESecretPreparedFactory<BE>,
    ScratchOwned<BE>: ScratchOwnedAlloc<BE> + ScratchOwnedBorrow<BE>,
{
    let ct_layout = glwe_layout_at(module, K);
    let layout = GLWEShamirLayout {
        k: TorusPrecision(K.0 - BASE2K.0),
        ..shamir_layout(module, 3, 3)
    };
    let prepared = module.glwe_wide_secret_prepared_alloc(&layout);
    let (_, sk_out) = secret_from_seed(module, [120u8; 32]);
    let ct: GLWE<AlignedBuf, i64> = module.glwe_alloc_from_infos(&ct_layout);
    let mut res = module.glwe_keyswitch_share_alloc_from_infos(&ct_layout);
    let mut scratch: ScratchOwned<BE> = ScratchOwned::alloc(module.mhe_glwe_threshold_keyswitch_share_gen_tmp_bytes(&ct_layout));
    module.mhe_glwe_threshold_keyswitch_share_gen(
        &mut res,
        &ct,
        &prepared,
        &sk_out,
        &flood_infos(ct_layout),
        128,
        &mut Source::new([10u8; 32]),
        &mut scratch.borrow(),
    );
}

/// Large ciphertext and wide-secret digits must be rejected before products.
pub fn test_glwe_threshold_keyswitch_digit_budget<BE>(module: &Module<BE>)
where
    BE: HostBackend<OwnedBuf = AlignedBuf, ZnxWord = i64>,
    Module<BE>: MHEModuleAlloc<BE> + GLWEThresholdKeyswitchMHEProtocol<BE> + GLWESecretPreparedFactory<BE>,
    ScratchOwned<BE>: ScratchOwnedAlloc<BE> + ScratchOwnedBorrow<BE>,
{
    threshold_keyswitch_rejected(module, Base2K(60), Base2K(60), TorusPrecision(120), 128, RANK, None);
}

/// A zero numerical failure target must not disable the product budget.
pub fn test_glwe_threshold_keyswitch_failure_bits<BE>(module: &Module<BE>)
where
    BE: HostBackend<OwnedBuf = AlignedBuf, ZnxWord = i64>,
    Module<BE>: MHEModuleAlloc<BE> + GLWEThresholdKeyswitchMHEProtocol<BE> + GLWESecretPreparedFactory<BE>,
    ScratchOwned<BE>: ScratchOwnedAlloc<BE> + ScratchOwnedBorrow<BE>,
{
    threshold_keyswitch_rejected(module, BASE2K, BASE2K, K, 0, RANK, None);
}

/// Mismatched output-secret ranks must fail before core decryption.
pub fn test_glwe_threshold_keyswitch_output_rank<BE>(module: &Module<BE>)
where
    BE: HostBackend<OwnedBuf = AlignedBuf, ZnxWord = i64>,
    Module<BE>: MHEModuleAlloc<BE> + GLWEThresholdKeyswitchMHEProtocol<BE> + GLWESecretPreparedFactory<BE>,
    ScratchOwned<BE>: ScratchOwnedAlloc<BE> + ScratchOwnedBorrow<BE>,
{
    threshold_keyswitch_rejected(module, BASE2K, BASE2K, K, 128, Rank(1), None);
}

/// Individually admissible products still need room for their lazy sum.
pub fn test_glwe_threshold_keyswitch_headroom<BE>(module: &Module<BE>)
where
    BE: HostBackend<OwnedBuf = AlignedBuf, ZnxWord = i64>,
    Module<BE>: MHEModuleAlloc<BE> + GLWEThresholdKeyswitchMHEProtocol<BE> + GLWESecretPreparedFactory<BE>,
    ScratchOwned<BE>: ScratchOwnedAlloc<BE> + ScratchOwnedBorrow<BE>,
{
    threshold_keyswitch_rejected(module, Base2K(62), Base2K(12), TorusPrecision(192), 128, RANK, None);
}

/// Flood precision must fit the actual sampling output, not just its descriptor.
pub fn test_glwe_threshold_keyswitch_flood_precision<BE>(module: &Module<BE>)
where
    BE: HostBackend<OwnedBuf = AlignedBuf, ZnxWord = i64>,
    Module<BE>: MHEModuleAlloc<BE> + GLWEThresholdKeyswitchMHEProtocol<BE> + GLWESecretPreparedFactory<BE>,
    ScratchOwned<BE>: ScratchOwnedAlloc<BE> + ScratchOwnedBorrow<BE>,
{
    threshold_keyswitch_rejected(module, BASE2K, BASE2K, K, 128, RANK, Some(K.as_usize() + 1));
}

fn threshold_keyswitch_rejected<BE>(
    module: &Module<BE>,
    base2k: Base2K,
    wide_base2k: Base2K,
    k: TorusPrecision,
    failure_bits: usize,
    output_rank: Rank,
    flood_k: Option<usize>,
) where
    BE: HostBackend<OwnedBuf = AlignedBuf, ZnxWord = i64>,
    Module<BE>: MHEModuleAlloc<BE> + GLWEThresholdKeyswitchMHEProtocol<BE> + GLWESecretPreparedFactory<BE>,
    ScratchOwned<BE>: ScratchOwnedAlloc<BE> + ScratchOwnedBorrow<BE>,
{
    let ct_layout = GLWELayout {
        base2k,
        ..glwe_layout_at(module, k)
    };
    let layout = GLWEShamirLayout {
        base2k: wide_base2k,
        k,
        ..shamir_layout(module, 2, 2)
    };
    let prepared = module.glwe_wide_secret_prepared_alloc(&layout);
    let sk_out = module.glwe_secret_prepared_alloc(output_rank);
    let ct: GLWE<AlignedBuf, i64> = module.glwe_alloc_from_infos(&ct_layout);
    let mut res = module.glwe_keyswitch_share_alloc_from_infos(&ct_layout);
    let mut flood = SmudgingNoise::gaussian(ct_layout.k.as_usize(), 2, 6);
    if let Some(k) = flood_k {
        flood.k = k;
    }
    let mut scratch: ScratchOwned<BE> = ScratchOwned::alloc(module.mhe_glwe_threshold_keyswitch_share_gen_tmp_bytes(&ct_layout));
    module.mhe_glwe_threshold_keyswitch_share_gen(
        &mut res,
        &ct,
        &prepared,
        &sk_out,
        &flood,
        failure_bits,
        &mut Source::new([10u8; 32]),
        &mut scratch.borrow(),
    );
}

/// Public key switching must enforce the same wide-secret product budget.
pub fn test_glwe_threshold_public_keyswitch_digit_budget<BE>(module: &Module<BE>)
where
    BE: HostBackend<OwnedBuf = AlignedBuf, ZnxWord = i64>,
    Module<BE>: MHEModuleAlloc<BE>
        + GLWEThresholdPublicKeyswitchMHEProtocol<BE>
        + GLWESecretPreparedFactory<BE>
        + GLWEPublicKeyPreparedFactory<BE>,
    ScratchOwned<BE>: ScratchOwnedAlloc<BE> + ScratchOwnedBorrow<BE>,
{
    let ct_layout = GLWELayout {
        base2k: Base2K(60),
        ..glwe_layout_at(module, TorusPrecision(120))
    };
    let layout = GLWEShamirLayout {
        base2k: ct_layout.base2k,
        k: ct_layout.k,
        ..shamir_layout(module, 2, 2)
    };
    let prepared = module.glwe_wide_secret_prepared_alloc(&layout);
    let pk_out = module.glwe_public_key_prepared_alloc_from_infos(&ct_layout);
    let ct: GLWE<AlignedBuf, i64> = module.glwe_alloc_from_infos(&ct_layout);
    let mut res = module.glwe_public_keyswitch_share_alloc_from_infos(&ct_layout);
    let enc_infos = EncryptionLayout::new_from_default_sigma(ct_layout).unwrap();
    let mut scratch: ScratchOwned<BE> =
        ScratchOwned::alloc(module.mhe_glwe_threshold_public_keyswitch_share_gen_tmp_bytes(&ct_layout, &ct_layout, &ct_layout));
    module.mhe_glwe_threshold_public_keyswitch_share_gen(
        &mut res,
        &ct,
        &prepared,
        &pk_out,
        &SmudgingNoise::gaussian(ct_layout.k.as_usize(), 2, 6),
        &enc_infos,
        128,
        &mut Source::new([20u8; 32]),
        &mut Source::new([10u8; 32]),
        &mut scratch.borrow(),
    );
}

/// Combining with fewer active parties than the threshold panics.
pub fn test_glwe_threshold_too_few_actives<BE>(module: &Module<BE>)
where
    BE: HostBackend<OwnedBuf = AlignedBuf, ZnxWord = i64>,
    Module<BE>: MHEModuleAlloc<BE> + GLWEShamirMHEProtocol<BE>,
    ScratchOwned<BE>: ScratchOwnedAlloc<BE> + ScratchOwnedBorrow<BE>,
{
    let layout = shamir_layout(module, 3, 3);
    let share = module.glwe_shamir_share_alloc(&layout);
    let mut res = module.glwe_wide_secret_alloc(&layout);
    let mut scratch: ScratchOwned<BE> = ScratchOwned::alloc(module.mhe_glwe_shamir_share_finalize_tmp_bytes());
    module.mhe_glwe_shamir_share_finalize(&mut res, &share, 1, &[1, 2], &mut scratch.borrow());
}

/// Combining for a party outside the active set panics.
pub fn test_glwe_threshold_own_not_active<BE>(module: &Module<BE>)
where
    BE: HostBackend<OwnedBuf = AlignedBuf, ZnxWord = i64>,
    Module<BE>: MHEModuleAlloc<BE> + GLWEShamirMHEProtocol<BE>,
    ScratchOwned<BE>: ScratchOwnedAlloc<BE> + ScratchOwnedBorrow<BE>,
{
    let layout = shamir_layout(module, 3, 3);
    let share = module.glwe_shamir_share_alloc(&layout);
    let mut res = module.glwe_wide_secret_alloc(&layout);
    let mut scratch: ScratchOwned<BE> = ScratchOwned::alloc(module.mhe_glwe_shamir_share_finalize_tmp_bytes());
    module.mhe_glwe_shamir_share_finalize(&mut res, &share, 1, &[2, 3, 4], &mut scratch.borrow());
}

/// Aggregating shares of different thresholds panics.
pub fn test_glwe_threshold_aggregate_mismatch<BE>(module: &Module<BE>)
where
    BE: HostBackend<OwnedBuf = AlignedBuf, ZnxWord = i64>,
    Module<BE>: MHEModuleAlloc<BE> + GLWEShamirMHEProtocol<BE>,
{
    let mut res = module.glwe_shamir_share_alloc(&shamir_layout(module, 3, 3));
    let share = module.glwe_shamir_share_alloc(&shamir_layout(module, 2, 3));
    module.mhe_glwe_shamir_share_aggregate(&mut res, &share);
}

fn shamir_layout<BE: poulpy_hal::layouts::Backend>(module: &Module<BE>, threshold: usize, gr_degree: usize) -> GLWEShamirLayout {
    GLWEShamirLayout {
        n: module.n().into(),
        base2k: B_SHARE,
        k: K_SHARE,
        rank: RANK,
        gr_degree,
        threshold,
    }
}

/// Thresholdizes one secret per party, then checks, for every active set,
/// that the combined additive shares sum to the secrets' sum.
fn threshold_combine<BE>(module: &Module<BE>, parties: usize, threshold: usize, active_sets: &[&[u32]])
where
    BE: HostBackend<OwnedBuf = AlignedBuf, ZnxWord = i64>,
    for<'a> BE::BufRef<'a>: HostDataRef,
    for<'a> BE::BufMut<'a>: HostDataMut,
    Module<BE>: MHEModuleAlloc<BE>
        + GLWEShamirMHEProtocol<BE>
        + GLWESecretSampling<BE>
        + GLWESecretPreparedFactory<BE>
        + GLWEAdd<BE>
        + GLWENormalize<BE>,
    ScratchOwned<BE>: ScratchOwnedAlloc<BE> + ScratchOwnedBorrow<BE>,
{
    let (secrets, layout, shares) = thresholdize(module, parties, threshold);
    let mut scratch: ScratchOwned<BE> = ScratchOwned::alloc(
        module
            .glwe_normalize_tmp_bytes()
            .max(module.mhe_glwe_shamir_share_finalize_tmp_bytes()),
    );
    let n = module.n();
    for &actives in active_sets {
        let mut sum = module.glwe_wide_secret_alloc(&layout);
        let mut part = module.glwe_wide_secret_alloc(&layout);
        for &own in actives {
            module.mhe_glwe_shamir_share_finalize(&mut part, &shares[own as usize - 1], own, actives, &mut scratch.borrow());
            assert_canonical(&part);
            module.glwe_add_assign(&mut sum.inner, &part.inner);
        }
        module.glwe_normalize_assign(&mut sum.inner, &mut scratch.borrow());
        for col in 0..RANK.as_usize() {
            let mut got = vec![0i128; n];
            sum.data()
                .decode_vec_i128(B_SHARE.as_usize(), col, K_SHARE.as_usize(), &mut got);
            for (i, &g) in got.iter().enumerate() {
                let want: i64 = secrets.iter().map(|(sk, _)| sk.data().at(col, 0)[i]).sum();
                assert_eq!(g, want as i128, "combined shares do not sum to the secret");
            }
        }
    }
}

/// One secret per party, thresholdized: every party's aggregated share of
/// the secrets' sum, recipient `i` at index `i - 1`.
fn thresholdize<BE>(
    module: &Module<BE>,
    parties: usize,
    threshold: usize,
) -> (Vec<Secret<BE>>, GLWEShamirLayout, Vec<GLWEShamirShareOwned<BE>>)
where
    BE: HostBackend<OwnedBuf = AlignedBuf, ZnxWord = i64>,
    for<'a> BE::BufRef<'a>: HostDataRef,
    for<'a> BE::BufMut<'a>: HostDataMut,
    Module<BE>: MHEModuleAlloc<BE>
        + GLWEShamirMHEProtocol<BE>
        + GLWESecretSampling<BE>
        + GLWESecretPreparedFactory<BE>
        + GLWEAdd<BE>
        + GLWENormalize<BE>,
    ScratchOwned<BE>: ScratchOwnedAlloc<BE> + ScratchOwnedBorrow<BE>,
{
    // The smallest Galois ring degree with a point per party.
    let gr_degree = (usize::BITS - parties.leading_zeros()) as usize;
    let layout = shamir_layout(module, threshold, gr_degree);
    let mut scratch: ScratchOwned<BE> = ScratchOwned::alloc(
        module
            .mhe_glwe_shamir_polynomial_gen_tmp_bytes(&layout)
            .max(module.mhe_glwe_shamir_share_gen_tmp_bytes(&layout))
            .max(module.mhe_glwe_shamir_share_finalize_tmp_bytes()),
    );
    let secrets: Vec<Secret<BE>> = (0..parties).map(|i| secret_from_seed(module, [200 + i as u8; 32])).collect();

    let polys: Vec<_> = secrets
        .iter()
        .enumerate()
        .map(|(j, (sk, _))| {
            let mut poly = module.glwe_shamir_polynomial_alloc(&layout);
            module.mhe_glwe_shamir_polynomial_gen(&mut poly, sk, &mut Source::new([210 + j as u8; 32]), &mut scratch.borrow());
            poly
        })
        .collect();
    let shares: Vec<GLWEShamirShareOwned<BE>> = (1..=parties as u32)
        .map(|recipient| {
            let mut acc = module.glwe_shamir_share_alloc(&layout);
            let mut share = module.glwe_shamir_share_alloc(&layout);
            for (j, poly) in polys.iter().enumerate() {
                let dst = if j == 0 { &mut acc } else { &mut share };
                module.mhe_glwe_shamir_share_gen(dst, poly, recipient, &mut scratch.borrow());
                if j > 0 {
                    module.mhe_glwe_shamir_share_aggregate(&mut acc, &share);
                }
            }
            acc
        })
        .collect();
    (secrets, layout, shares)
}

/// Five parties, threshold 3: the parties' secrets, the sharing layout and
/// the combined additive shares of [`ACTIVES`].
fn combined_shares<BE>(module: &Module<BE>) -> (Vec<Secret<BE>>, GLWEShamirLayout, Vec<GLWEWideSecretOwned<BE>>)
where
    BE: HostBackend<OwnedBuf = AlignedBuf, ZnxWord = i64>,
    for<'a> BE::BufRef<'a>: HostDataRef,
    for<'a> BE::BufMut<'a>: HostDataMut,
    Module<BE>: MHEModuleAlloc<BE>
        + GLWEShamirMHEProtocol<BE>
        + GLWESecretSampling<BE>
        + GLWESecretPreparedFactory<BE>
        + GLWEAdd<BE>
        + GLWENormalize<BE>,
    ScratchOwned<BE>: ScratchOwnedAlloc<BE> + ScratchOwnedBorrow<BE>,
{
    let (secrets, layout, shares) = thresholdize(module, 5, 3);
    let mut scratch: ScratchOwned<BE> = ScratchOwned::alloc(module.mhe_glwe_shamir_share_finalize_tmp_bytes());
    let wide = ACTIVES
        .iter()
        .map(|&own| {
            let mut sigma = module.glwe_wide_secret_alloc(&layout);
            module.mhe_glwe_shamir_share_finalize(&mut sigma, &shares[own as usize - 1], own, &ACTIVES, &mut scratch.borrow());
            sigma
        })
        .collect();
    (secrets, layout, wide)
}

/// Threshold key switching of a ciphertext of the parties' secrets to the sum
/// of `out`, one output secret per active party: the finalized ciphertext
/// decrypts under that sum within the smudging noise.
fn threshold_keyswitch<BE>(module: &Module<BE>, out: Vec<Secret<BE>>)
where
    BE: HostBackend<OwnedBuf = AlignedBuf, ZnxWord = i64>,
    for<'a> BE::BufRef<'a>: HostDataRef,
    for<'a> BE::BufMut<'a>: HostDataMut,
    Module<BE>: MHEModuleAlloc<BE>
        + GLWEShamirMHEProtocol<BE>
        + GLWEWideSecretPrepare<BE>
        + GLWEThresholdKeyswitchMHEProtocol<BE>
        + GLWESecretSampling<BE>
        + GLWESecretPreparedFactory<BE>
        + GLWEEncryptSk<BE>
        + GLWENoise<BE>
        + GLWEAdd<BE>
        + GLWENormalize<BE>
        + VecZnxAddScalarAssign<BE>,
    ScratchOwned<BE>: ScratchOwnedAlloc<BE> + ScratchOwnedBorrow<BE>,
{
    let ct_layout = glwe_layout_at(module, K);
    let (secrets, layout, wide) = combined_shares(module);
    let mut scratch: ScratchOwned<BE> = ScratchOwned::alloc(
        module
            .glwe_encrypt_sk_tmp_bytes(&ct_layout)
            .max(module.mhe_glwe_wide_secret_prepare_tmp_bytes(&layout))
            .max(module.mhe_glwe_threshold_keyswitch_share_gen_tmp_bytes(&ct_layout))
            .max(module.mhe_glwe_threshold_keyswitch_share_finalize_tmp_bytes())
            .max(module.glwe_noise_tmp_bytes(&ct_layout)),
    );
    let m = bounded_integers(module.n(), LOG_MESSAGE, [30u8; 32]);
    let ct = encrypt_integers(module, &ct_layout, &m, &ideal_secret(module, &secrets), &mut scratch);

    let flood = flood_infos(ct_layout);
    let mut acc = module.glwe_keyswitch_share_alloc_from_infos(&ct_layout);
    let mut share = module.glwe_keyswitch_share_alloc_from_infos(&ct_layout);
    for (i, (sigma, (_, sk_out))) in wide.iter().zip(&out).enumerate() {
        let mut prepared = module.glwe_wide_secret_prepared_alloc(&layout);
        module.mhe_glwe_wide_secret_prepare(&mut prepared, sigma, &mut scratch.borrow());
        let dst = if i == 0 { &mut acc } else { &mut share };
        dst.inner.set_canonical(false);
        let mut source_xe = Source::new([10 + i as u8; 32]);
        module.mhe_glwe_threshold_keyswitch_share_gen(
            dst,
            &ct,
            &prepared,
            sk_out,
            &flood,
            128,
            &mut source_xe,
            &mut scratch.borrow(),
        );
        assert!(dst.inner.is_canonical());
        if i > 0 {
            module.mhe_glwe_threshold_keyswitch_share_aggregate(&mut acc, &share);
        }
    }

    let mut res: GLWE<AlignedBuf, i64> = module.glwe_alloc_from_infos(&ct_layout);
    module.mhe_glwe_threshold_keyswitch_share_finalize(&mut res, &ct, &acc, &mut scratch.borrow());
    let pt = integer_plaintext(module, &ct_layout, &m);
    assert_flooded_noise(module, &res, &pt, &ideal_secret(module, &out), 0.0, &mut scratch);
}

/// The limbs of `secret` are balanced digits and its last limb has no bits
/// below its precision.
fn assert_canonical(secret: &GLWEWideSecret<AlignedBuf, i64>) {
    let (base2k, k) = (secret.base2k().as_usize(), secret.k().as_usize());
    let size = k.div_ceil(base2k);
    let (half, pad) = (1i64 << (base2k - 1), size * base2k - k);
    for col in 0..secret.rank().as_usize() {
        for limb in 0..size {
            for &x in secret.data().at(col, limb) {
                assert!((-half..half).contains(&x), "combined share digit out of range");
                if limb == size - 1 {
                    assert_eq!(x % (1 << pad), 0, "combined share has bits below its precision");
                }
            }
        }
    }
}
