//! Wide smudging through every decryption-type protocol. Integer oracles and
//! replayed private error streams detect lost high bits, low bits, parties,
//! signs, and precision scaling without converting the flood to a float.

use dashu_int::IBig;
use poulpy_core::{
    DEFAULT_BOUND_XE, EncryptionLayout, GLWEDecrypt, GLWEEncryptSk, GLWENormalize, SmudgingNoise, VecZnxAddSmudging,
    layouts::{
        Base2K, GLWE, GLWELayout, GLWEPlaintext, GLWEPublicKeyPreparedFactory, GLWESecretPrepared, GLWESecretPreparedFactory,
        GLWESecretSampling, LWEInfos, ModuleCoreAlloc, TorusPrecision,
    },
};
use poulpy_hal::{
    AlignedBuf,
    api::{ScratchOwnedAlloc, ScratchOwnedBorrow, VecZnxAddScalarAssign},
    layouts::{Backend, HostBackend, HostDataMut, HostDataRef, Module, ScratchOwned, VecZnx, ZnxView, ZnxViewMut},
    source::Source,
    test_suite::vec_znx_backend_mut,
};

use super::fixtures::{BASE2K, PARTIES, RANK, SEEDS, Secret, collective_public_key, ideal_secret, secret_from_seed};
use crate::{
    api::{
        GLWEEncToShareMHEProtocol, GLWEKeyswitchMHEProtocol, GLWEPublicKeyMHEProtocol, GLWEPublicKeyswitchMHEProtocol,
        GLWERefreshMHEProtocol, GLWEShamirMHEProtocol, GLWEThresholdKeyswitchMHEProtocol,
        GLWEThresholdPublicKeyswitchMHEProtocol, GLWEWideSecretPrepare,
    },
    layouts::{GLWEShamirLayout, GLWEWideSecretPreparedOwned, MHEModuleAlloc},
};

const K: TorusPrecision = TorusPrecision(256);
const K_OUT: TorusPrecision = TorusPrecision(281);
const MASK_BITS: usize = 230;
const ACTIVES: [u32; 2] = [1, 3];

fn error_seed(party: usize) -> [u8; 32] {
    [90 + party as u8; 32]
}

/// Reconstruct exactly, including the last limb's precision padding.
fn integers(data: &VecZnx<AlignedBuf, i64>, base2k: usize, k: usize) -> Vec<IBig> {
    let size = k.div_ceil(base2k);
    let padding = size * base2k - k;
    let padding_mask = (IBig::ONE << padding) - IBig::ONE;
    (0..data.n())
        .map(|coefficient| {
            let value = (0..size).fold(IBig::ZERO, |value, limb| {
                (value << base2k) + IBig::from(data.at(0, limb)[coefficient])
            });
            assert_eq!(&value & &padding_mask, IBig::ZERO, "nonzero precision padding");
            value >> padding
        })
        .collect()
}

fn plaintext<BE: Backend<OwnedBuf = AlignedBuf, ZnxWord = i64>>(
    module: &Module<BE>,
    layout: &GLWELayout,
    values: &[IBig],
) -> GLWEPlaintext<AlignedBuf, i64> {
    let mut res = module.glwe_plaintext_alloc_from_infos(layout);
    let base = layout.base2k.as_usize();
    let size = layout.k.as_usize().div_ceil(base);
    let padding = size * base - layout.k.as_usize();
    let half = IBig::ONE << (base - 1);
    for (coefficient, value) in values.iter().enumerate() {
        let mut value = value << padding;
        for limb in (0..size).rev() {
            let carry = (&value + &half) >> base;
            res.data_mut().at_mut(0, limb)[coefficient] = i64::try_from(&value - (&carry << base)).unwrap();
            value = carry;
        }
        assert_eq!(value, IBig::ZERO, "test plaintext exceeds the signed frame");
    }
    assert_eq!(integers(res.data(), base, layout.k.as_usize()), values);
    res
}

/// The returned values are the exact sum of the draws each protocol must add.
fn replay_flood<BE>(module: &Module<BE>, layout: &GLWELayout, flood: SmudgingNoise, parties: usize) -> Vec<IBig>
where
    BE: HostBackend<OwnedBuf = AlignedBuf, ZnxWord = i64>,
    Module<BE>: VecZnxAddSmudging<BE> + GLWENormalize<BE>,
    ScratchOwned<BE>: ScratchOwnedAlloc<BE> + ScratchOwnedBorrow<BE>,
{
    let mut sum = vec![IBig::ZERO; module.n()];
    let mut scratch = ScratchOwned::<BE>::alloc(module.glwe_normalize_tmp_bytes());
    for party in 0..parties {
        let mut sample: GLWEPlaintext<AlignedBuf, i64> = module.glwe_plaintext_alloc_from_infos(layout);
        module.vec_znx_add_smudging(
            layout.base2k.as_usize(),
            &mut vec_znx_backend_mut::<BE>(sample.data_mut()),
            0,
            flood,
            &mut Source::new(error_seed(party)),
        );
        module.glwe_normalize_assign(&mut sample, &mut scratch.borrow());
        let values = integers(sample.data(), layout.base2k.as_usize(), layout.k.as_usize());
        let wide = IBig::ONE << 128;
        assert!(
            values.iter().any(|value| value > &wide || value < &(-&wide)),
            "flood never exceeds 128 bits"
        );
        assert!(
            values.iter().any(|value| (value & IBig::ONE) == IBig::ONE),
            "flood lost its low bit"
        );
        assert!(
            values.iter().any(|value| (value & IBig::ONE) == IBig::ZERO),
            "flood low bit is constant"
        );
        for (sum, value) in sum.iter_mut().zip(values) {
            *sum += value;
        }
    }
    sum
}

fn assert_replayed_noise(actual: &[IBig], message: &[IBig], flood: &[IBig], rounding_bound: i64) {
    assert_eq!(actual.len(), message.len());
    assert_eq!(actual.len(), flood.len());
    let bound = IBig::from(rounding_bound);
    for ((actual, message), flood) in actual.iter().zip(message).zip(flood) {
        let residual = actual - message - flood;
        assert!(
            residual >= -&bound && residual <= bound,
            "wide flood changed before reconstruction"
        );
    }
}

fn decrypt_integers<BE>(
    module: &Module<BE>,
    ct: &GLWE<AlignedBuf, i64>,
    sk: &GLWESecretPrepared<AlignedBuf, BE>,
    scratch: &mut ScratchOwned<BE>,
) -> Vec<IBig>
where
    BE: HostBackend<OwnedBuf = AlignedBuf, ZnxWord = i64>,
    Module<BE>: GLWEDecrypt<BE>,
    ScratchOwned<BE>: ScratchOwnedBorrow<BE>,
{
    let mut pt = module.glwe_plaintext_alloc_from_infos(ct);
    module.glwe_decrypt(ct, &mut pt, sk, &mut scratch.borrow());
    integers(pt.data(), ct.base2k().as_usize(), ct.k().as_usize())
}

/// Thresholdize all three input secrets, but combine only two active parties.
fn threshold_inputs<BE>(module: &Module<BE>, secrets: &[Secret<BE>]) -> Vec<GLWEWideSecretPreparedOwned<BE>>
where
    BE: HostBackend<OwnedBuf = AlignedBuf, ZnxWord = i64>,
    Module<BE>: MHEModuleAlloc<BE> + GLWEShamirMHEProtocol<BE> + GLWEWideSecretPrepare<BE> + GLWESecretPreparedFactory<BE>,
    ScratchOwned<BE>: ScratchOwnedAlloc<BE> + ScratchOwnedBorrow<BE>,
{
    let layout = GLWEShamirLayout {
        n: module.n().into(),
        base2k: Base2K(8),
        k: TorusPrecision(263),
        rank: RANK,
        gr_degree: 2,
        threshold: 2,
    };
    let mut scratch = ScratchOwned::<BE>::alloc(
        module
            .mhe_glwe_shamir_polynomial_gen_tmp_bytes(&layout)
            .max(module.mhe_glwe_shamir_share_gen_tmp_bytes(&layout))
            .max(module.mhe_glwe_shamir_share_finalize_tmp_bytes())
            .max(module.mhe_glwe_wide_secret_prepare_tmp_bytes(&layout)),
    );
    let polys: Vec<_> = secrets
        .iter()
        .enumerate()
        .map(|(party, (sk, _))| {
            let mut poly = module.glwe_shamir_polynomial_alloc(&layout);
            module.mhe_glwe_shamir_polynomial_gen(
                &mut poly,
                sk,
                &mut Source::new([210 + party as u8; 32]),
                &mut scratch.borrow(),
            );
            poly
        })
        .collect();
    ACTIVES
        .iter()
        .map(|&recipient| {
            let mut acc = module.glwe_shamir_share_alloc(&layout);
            let mut part = module.glwe_shamir_share_alloc(&layout);
            for (party, poly) in polys.iter().enumerate() {
                let dst = if party == 0 { &mut acc } else { &mut part };
                module.mhe_glwe_shamir_share_gen(dst, poly, recipient, &mut scratch.borrow());
                if party != 0 {
                    module.mhe_glwe_shamir_share_aggregate(&mut acc, &part);
                }
            }
            let mut combined = module.glwe_wide_secret_alloc(&layout);
            module.mhe_glwe_shamir_share_finalize(&mut combined, &acc, recipient, &ACTIVES, &mut scratch.borrow());
            let mut prepared = module.glwe_wide_secret_prepared_alloc(&layout);
            module.mhe_glwe_wide_secret_prepare(&mut prepared, &combined, &mut scratch.borrow());
            prepared
        })
        .collect()
}

/// Exercises both variants of CKS, their threshold counterparts, E2S and refresh.
/// Tests replay each private flood exactly; only ordinary RLWE/rounding error
/// may remain. These functional parameters are not a production parameter set.
pub fn test_wide_smudging<BE>(module: &Module<BE>, gaussian: bool)
where
    BE: HostBackend<OwnedBuf = AlignedBuf, ZnxWord = i64>,
    for<'a> BE::BufRef<'a>: HostDataRef,
    for<'a> BE::BufMut<'a>: HostDataMut,
    Module<BE>: MHEModuleAlloc<BE>
        + GLWEKeyswitchMHEProtocol<BE>
        + GLWEPublicKeyswitchMHEProtocol<BE>
        + GLWEPublicKeyMHEProtocol<BE>
        + GLWEEncToShareMHEProtocol<BE>
        + GLWERefreshMHEProtocol<BE>
        + GLWEShamirMHEProtocol<BE>
        + GLWEWideSecretPrepare<BE>
        + GLWEThresholdKeyswitchMHEProtocol<BE>
        + GLWEThresholdPublicKeyswitchMHEProtocol<BE>
        + GLWESecretSampling<BE>
        + GLWESecretPreparedFactory<BE>
        + GLWEPublicKeyPreparedFactory<BE>
        + GLWEEncryptSk<BE>
        + GLWEDecrypt<BE>
        + GLWENormalize<BE>
        + VecZnxAddSmudging<BE>
        + VecZnxAddScalarAssign<BE>,
    ScratchOwned<BE>: ScratchOwnedAlloc<BE> + ScratchOwnedBorrow<BE>,
{
    let layout = GLWELayout {
        n: module.n().into(),
        base2k: BASE2K,
        k: K,
        rank: RANK,
    };
    let out_layout = GLWELayout { k: K_OUT, ..layout };
    let flood = if gaussian {
        SmudgingNoise::gaussian(K.as_usize(), 130, 16)
    } else {
        SmudgingNoise::uniform(K.as_usize(), 132)
    };
    let enc_infos = EncryptionLayout::new_from_default_sigma(layout).unwrap();
    let out_enc_infos = EncryptionLayout::new_from_default_sigma(out_layout).unwrap();
    let inputs: Vec<Secret<BE>> = (0..PARTIES).map(|i| secret_from_seed(module, [150 + i as u8; 32])).collect();
    let outputs: Vec<Secret<BE>> = (0..PARTIES).map(|i| secret_from_seed(module, [100 + i as u8; 32])).collect();
    let ideal_input = ideal_secret(module, &inputs);
    let wide = threshold_inputs(module, &inputs);
    let message: Vec<IBig> = (0..module.n())
        .map(|i| {
            let value = (IBig::ONE << 150) + (IBig::from(i) << 70) + IBig::from(i);
            if i % 2 == 0 { value } else { -value }
        })
        .collect();
    let pt = plaintext(module, &layout, &message);
    let mut scratch = ScratchOwned::<BE>::alloc(
        module
            .glwe_encrypt_sk_tmp_bytes(&layout)
            .max(module.glwe_decrypt_tmp_bytes(&out_layout))
            .max(module.mhe_glwe_keyswitch_share_gen_tmp_bytes(&layout))
            .max(module.mhe_glwe_keyswitch_share_finalize_tmp_bytes())
            .max(module.mhe_glwe_public_keyswitch_share_gen_tmp_bytes(&layout, &layout, &layout))
            .max(module.mhe_glwe_public_keyswitch_share_finalize_tmp_bytes())
            .max(module.mhe_glwe_threshold_keyswitch_share_gen_tmp_bytes(&layout))
            .max(module.mhe_glwe_threshold_keyswitch_share_finalize_tmp_bytes())
            .max(module.mhe_glwe_threshold_public_keyswitch_share_gen_tmp_bytes(&layout, &layout, &layout))
            .max(module.mhe_glwe_threshold_public_keyswitch_share_finalize_tmp_bytes())
            .max(module.mhe_glwe_enc_to_share_share_gen_tmp_bytes(&layout))
            .max(module.mhe_glwe_enc_to_share_share_finalize_tmp_bytes())
            .max(module.mhe_glwe_refresh_share_gen_tmp_bytes(&layout, &out_layout))
            .max(module.mhe_glwe_refresh_share_finalize_tmp_bytes(&out_layout))
            .max(module.glwe_normalize_tmp_bytes()),
    );
    let mut ct = module.glwe_alloc_from_infos(&layout);
    module.glwe_encrypt_sk(
        &mut ct,
        &pt,
        &ideal_input,
        &enc_infos,
        &mut Source::new([31; 32]),
        &mut Source::new([32; 32]),
        &mut scratch.borrow(),
    );
    let fresh_bound = DEFAULT_BOUND_XE.ceil() as i64 + 1;
    let ordinary_bound = (PARTIES as i64 + 1) * (fresh_bound + 2);

    // Keep the lattice at k=256: changing its precision or dropping any party's
    // flood changes the replay residual by far more than the ordinary errors.
    for mode in 0..4 {
        let threshold = mode >= 2;
        let public = mode % 2 == 1;
        let count = if threshold { ACTIVES.len() } else { PARTIES };
        let replayed = replay_flood(module, &layout, flood, count);
        let ideal_output = ideal_secret(module, &outputs[..count]);
        let mut res = module.glwe_alloc_from_infos(&layout);
        if public {
            let pk = collective_public_key(module, &outputs[..count], &layout);
            let mut acc = module.glwe_public_keyswitch_share_alloc_from_infos(&layout);
            let mut share = module.glwe_public_keyswitch_share_alloc_from_infos(&layout);
            for party in 0..count {
                let dst = if party == 0 { &mut acc } else { &mut share };
                let mut source_xu = Source::new([20 + party as u8; 32]);
                let mut source_xe = Source::new(error_seed(party));
                if threshold {
                    module.mhe_glwe_threshold_public_keyswitch_share_gen(
                        dst,
                        &ct,
                        &wide[party],
                        &pk,
                        &flood,
                        &enc_infos,
                        128,
                        &mut source_xu,
                        &mut source_xe,
                        &mut scratch.borrow(),
                    );
                    if party != 0 {
                        module.mhe_glwe_threshold_public_keyswitch_share_aggregate(&mut acc, &share);
                    }
                } else {
                    module.mhe_glwe_public_keyswitch_share_gen(
                        dst,
                        &ct,
                        &inputs[party].1,
                        &pk,
                        &flood,
                        &enc_infos,
                        &mut source_xu,
                        &mut source_xe,
                        &mut scratch.borrow(),
                    );
                    if party != 0 {
                        module.mhe_glwe_public_keyswitch_share_aggregate(&mut acc, &share);
                    }
                }
            }
            if threshold {
                module.mhe_glwe_threshold_public_keyswitch_share_finalize(&mut res, &ct, &acc, &mut scratch.borrow());
            } else {
                module.mhe_glwe_public_keyswitch_share_finalize(&mut res, &ct, &acc, &mut scratch.borrow());
            }
        } else {
            let mut acc = module.glwe_keyswitch_share_alloc_from_infos(&layout);
            let mut share = module.glwe_keyswitch_share_alloc_from_infos(&layout);
            for party in 0..count {
                let dst = if party == 0 { &mut acc } else { &mut share };
                let mut source_xe = Source::new(error_seed(party));
                if threshold {
                    module.mhe_glwe_threshold_keyswitch_share_gen(
                        dst,
                        &ct,
                        &wide[party],
                        &outputs[party].1,
                        &flood,
                        128,
                        &mut source_xe,
                        &mut scratch.borrow(),
                    );
                    if party != 0 {
                        module.mhe_glwe_threshold_keyswitch_share_aggregate(&mut acc, &share);
                    }
                } else {
                    module.mhe_glwe_keyswitch_share_gen(
                        dst,
                        &ct,
                        &inputs[party].1,
                        &outputs[party].1,
                        &flood,
                        &mut source_xe,
                        &mut scratch.borrow(),
                    );
                    if party != 0 {
                        module.mhe_glwe_keyswitch_share_aggregate(&mut acc, &share);
                    }
                }
            }
            if threshold {
                module.mhe_glwe_threshold_keyswitch_share_finalize(&mut res, &ct, &acc, &mut scratch.borrow());
            } else {
                module.mhe_glwe_keyswitch_share_finalize(&mut res, &ct, &acc, &mut scratch.borrow());
            }
        }
        let actual = decrypt_integers(module, &res, &ideal_output, &mut scratch);
        // Ternary u has |u_i|<=1, ideal-secret coefficients <=count, and pk
        // error <=count*fresh_bound. Bound both polynomial products explicitly.
        let pk_bound = count as i64 * (((RANK.as_usize() + 1) * module.n() * count) as i64 * fresh_bound + fresh_bound);
        assert_replayed_noise(
            &actual,
            &message,
            &replayed,
            ordinary_bound + if public { pk_bound } else { 0 },
        );
    }

    let replayed = replay_flood(module, &layout, flood, PARTIES);
    let mut public_acc = module.glwe_enc_to_share_share_alloc_from_infos(&layout);
    let mut public_share = module.glwe_enc_to_share_share_alloc_from_infos(&layout);
    let mut masks: Vec<_> = (0..PARTIES)
        .map(|_| module.glwe_plaintext_alloc_from_infos(&layout))
        .collect();
    for party in 0..PARTIES {
        module.mhe_glwe_enc_to_share_share_gen(
            if party == 0 { &mut public_acc } else { &mut public_share },
            &mut masks[party],
            &ct,
            &inputs[party].1,
            MASK_BITS,
            &flood,
            &mut Source::new([60 + party as u8; 32]),
            &mut Source::new(error_seed(party)),
            &mut scratch.borrow(),
        );
        if party != 0 {
            module.mhe_glwe_enc_to_share_share_aggregate(&mut public_acc, &public_share);
        }
    }
    module.mhe_glwe_enc_to_share_share_finalize(&mut masks[0], &ct, &public_acc, &mut scratch.borrow());
    let mut reconstructed = vec![IBig::ZERO; module.n()];
    for mask in &masks {
        for (sum, value) in reconstructed
            .iter_mut()
            .zip(integers(mask.data(), BASE2K.as_usize(), K.as_usize()))
        {
            *sum += value;
        }
    }
    assert_replayed_noise(&reconstructed, &message, &replayed, ordinary_bound);

    let mut acc = module.glwe_refresh_share_alloc_from_infos(&layout, &out_layout);
    let mut share = module.glwe_refresh_share_alloc_from_infos(&layout, &out_layout);
    for (party, (_, sk)) in inputs.iter().enumerate() {
        module.mhe_glwe_refresh_share_gen(
            if party == 0 { &mut acc } else { &mut share },
            &ct,
            sk,
            MASK_BITS,
            SEEDS[1],
            &flood,
            &out_enc_infos,
            &mut Source::new([60 + party as u8; 32]),
            &mut Source::new(error_seed(party)),
            &mut scratch.borrow(),
        );
        if party != 0 {
            module.mhe_glwe_refresh_share_aggregate(&mut acc, &share);
        }
    }
    let mut res = module.glwe_alloc_from_infos(&out_layout);
    module.mhe_glwe_refresh_share_finalize(&mut res, &ct, &acc, &mut scratch.borrow());
    assert_replayed_noise(
        &decrypt_integers(module, &res, &ideal_input, &mut scratch),
        &message,
        &replayed,
        ordinary_bound,
    );
}
