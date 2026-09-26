//! Collective key switching over three parties: the finalized ciphertext
//! decrypts under the ideal output secret, its noise carries the parties'
//! smudging noise, and finalizing before or after normalization gives the
//! same result.

use poulpy_core::{
    DEFAULT_SIGMA_XE, Distribution, EncryptionLayout, GLWEAdd, GLWEEncryptSk, GLWENoise, GLWENormalize, NoiseInfos,
    layouts::{
        Base2K, GLWE, GLWELayout, GLWEPlaintext, GLWEPublicKey, GLWEPublicKeyPrepared, GLWEPublicKeyPreparedFactory,
        GLWESecretPrepared, GLWESecretPreparedFactory, GLWESecretSampling, ModuleCoreAlloc, Rank,
    },
};
use poulpy_hal::{
    AlignedBuf,
    api::{ScratchOwnedAlloc, ScratchOwnedBorrow, VecZnxFillUniformSource},
    layouts::{HostBackend, HostDataMut, HostDataRef, Module, ScratchOwned, WriterTo},
    source::Source,
    test_suite::vec_znx_backend_mut,
};

use super::fixtures::{BASE2K, K, PARTIES, RANK, SEEDS, Secret, ideal_secret, party_secrets, secret_from_seed};
use crate::{
    api::{GLWEKeyswitchShare, GLWEPublicKeyShare, GLWEPublicKeyswitchShare, PatAggregate},
    layouts::MHEModuleAlloc,
};

/// Smudging noise sigma of every party, well above the fresh noise.
const SIGMA_FLOOD: f64 = 1024.0;

pub fn test_glwe_keyswitch<BE>(module: &Module<BE>)
where
    BE: HostBackend<OwnedBuf = AlignedBuf, ZnxWord = i64>,
    for<'a> BE::BufRef<'a>: HostDataRef,
    for<'a> BE::BufMut<'a>: HostDataMut,
    Module<BE>: GLWEKeyswitchShare<BE>
        + GLWESecretSampling<BE>
        + GLWESecretPreparedFactory<BE>
        + GLWEEncryptSk<BE>
        + GLWEAdd<BE>
        + GLWENormalize<BE>
        + GLWENoise<BE>
        + VecZnxFillUniformSource<BE>,
    ScratchOwned<BE>: ScratchOwnedAlloc<BE> + ScratchOwnedBorrow<BE>,
{
    let layout = glwe_layout(module);
    let share_layout = GLWELayout { rank: Rank(0), ..layout };
    let enc_infos = EncryptionLayout::new_from_default_sigma(layout).unwrap();
    let flood = flood_infos(layout);
    let parties_in = input_secrets(module);
    let parties_out = party_secrets(module);
    let sk_out = ideal_secret(module, &parties_out);
    let mut scratch: ScratchOwned<BE> = ScratchOwned::alloc(
        module
            .glwe_encrypt_sk_tmp_bytes(&layout)
            .max(module.glwe_keyswitch_share_tmp_bytes(&layout))
            .max(module.glwe_keyswitch_finalize_tmp_bytes())
            .max(module.glwe_normalize_tmp_bytes())
            .max(module.glwe_noise_tmp_bytes(&layout)),
    );
    let (pt, ct) = encrypted_plaintext(module, &ideal_secret(module, &parties_in), &enc_infos, &mut scratch);

    let mut acc: GLWE<AlignedBuf, i64> = module.glwe_alloc_from_infos(&share_layout);
    let mut share: GLWE<AlignedBuf, i64> = module.glwe_alloc_from_infos(&share_layout);
    for (i, ((_, sk_in), (_, sk_out_i))) in parties_in.iter().zip(&parties_out).enumerate() {
        let dst = if i == 0 { &mut acc } else { &mut share };
        dst.set_canonical(false);
        let mut source_xe = Source::new([10 + i as u8; 32]);
        module.glwe_keyswitch_share(dst, &ct, sk_in, sk_out_i, &flood, &mut source_xe, &mut scratch.borrow());
        assert!(dst.is_canonical());
        if i > 0 {
            module.glwe_add_assign(&mut acc, &share);
        }
    }
    assert!(!acc.is_canonical());

    let mut lazy: GLWE<AlignedBuf, i64> = module.glwe_alloc_from_infos(&layout);
    module.glwe_keyswitch_finalize(&mut lazy, &ct, &acc, &mut scratch.borrow());
    assert!(lazy.is_canonical());
    assert_flooded_noise(module, &lazy, &pt, &sk_out, 0.0, &mut scratch);

    module.glwe_normalize_assign(&mut acc, &mut scratch.borrow());
    let mut eager: GLWE<AlignedBuf, i64> = module.glwe_alloc_from_infos(&layout);
    module.glwe_keyswitch_finalize(&mut eager, &ct, &acc, &mut scratch.borrow());
    assert_eq!(lazy, eager);
    acc.write_to(&mut Vec::new()).unwrap();
}

pub fn test_glwe_public_keyswitch<BE>(module: &Module<BE>)
where
    BE: HostBackend<OwnedBuf = AlignedBuf, ZnxWord = i64>,
    for<'a> BE::BufRef<'a>: HostDataRef,
    for<'a> BE::BufMut<'a>: HostDataMut,
    Module<BE>: MHEModuleAlloc<BE>
        + GLWEPublicKeyswitchShare<BE>
        + GLWEPublicKeyShare<BE>
        + PatAggregate<BE>
        + GLWESecretSampling<BE>
        + GLWESecretPreparedFactory<BE>
        + GLWEPublicKeyPreparedFactory<BE>
        + GLWEEncryptSk<BE>
        + GLWEAdd<BE>
        + GLWENormalize<BE>
        + GLWENoise<BE>
        + VecZnxFillUniformSource<BE>,
    ScratchOwned<BE>: ScratchOwnedAlloc<BE> + ScratchOwnedBorrow<BE>,
{
    let layout = glwe_layout(module);
    let enc_infos = EncryptionLayout::new_from_default_sigma(layout).unwrap();
    let flood = flood_infos(layout);
    let parties_in = input_secrets(module);
    let parties_out = party_secrets(module);
    let sk_out = ideal_secret(module, &parties_out);
    let mut scratch: ScratchOwned<BE> = ScratchOwned::alloc(
        module
            .glwe_encrypt_sk_tmp_bytes(&layout)
            .max(module.glwe_public_key_share_tmp_bytes(&layout))
            .max(module.glwe_public_key_finalize_tmp_bytes())
            .max(module.glwe_public_key_prepare_tmp_bytes(&layout))
            .max(module.glwe_public_keyswitch_share_tmp_bytes(&layout, &layout))
            .max(module.glwe_public_keyswitch_finalize_tmp_bytes())
            .max(module.glwe_normalize_tmp_bytes())
            .max(module.glwe_noise_tmp_bytes(&layout)),
    );

    let mut pk_acc = module.glwe_pat_compressed_alloc_from_infos(&layout);
    let mut pk_share = module.glwe_pat_compressed_alloc_from_infos(&layout);
    for (i, (_, sk)) in parties_out.iter().enumerate() {
        let dst = if i == 0 { &mut pk_acc } else { &mut pk_share };
        let mut source_xe = Source::new([40 + i as u8; 32]);
        module.glwe_public_key_share(dst, sk, SEEDS[0], &enc_infos, &mut source_xe, &mut scratch.borrow());
        if i > 0 {
            module.glwe_pat_compressed_aggregate_assign(&mut pk_acc, &pk_share);
        }
    }
    let mut pk: GLWEPublicKey<AlignedBuf, i64> = module.glwe_public_key_alloc_from_infos(&layout);
    module.glwe_public_key_finalize(&mut pk, &pk_acc, Distribution::TernaryProb(0.5), &mut scratch.borrow());
    let mut pk_out: GLWEPublicKeyPrepared<AlignedBuf, BE> = module.glwe_public_key_prepared_alloc_from_infos(&layout);
    module.glwe_public_key_prepare(&mut pk_out, &pk, &mut scratch.borrow());

    let (pt, ct) = encrypted_plaintext(module, &ideal_secret(module, &parties_in), &enc_infos, &mut scratch);

    let mut acc: GLWE<AlignedBuf, i64> = module.glwe_alloc_from_infos(&layout);
    let mut share: GLWE<AlignedBuf, i64> = module.glwe_alloc_from_infos(&layout);
    for (i, (_, sk_in)) in parties_in.iter().enumerate() {
        let dst = if i == 0 { &mut acc } else { &mut share };
        dst.set_canonical(false);
        let mut source_xu = Source::new([20 + i as u8; 32]);
        let mut source_xe = Source::new([10 + i as u8; 32]);
        module.glwe_public_keyswitch_share(
            dst,
            &ct,
            sk_in,
            &pk_out,
            &flood,
            &enc_infos,
            &mut source_xu,
            &mut source_xe,
            &mut scratch.borrow(),
        );
        assert!(dst.is_canonical());
        if i > 0 {
            module.glwe_add_assign(&mut acc, &share);
        }
    }
    assert!(!acc.is_canonical());

    let mut lazy: GLWE<AlignedBuf, i64> = module.glwe_alloc_from_infos(&layout);
    module.glwe_public_keyswitch_finalize(&mut lazy, &ct, &acc, &mut scratch.borrow());
    assert!(lazy.is_canonical());
    // Each party's pk encryption adds (rank + 1) * n * 0.5 * PARTIES * sigma^2, as in the pk test.
    let n = module.n() as f64;
    let rank = RANK.as_usize() as f64;
    let pk_noise = PARTIES as f64 * (rank + 1.0) * n * 0.5 * PARTIES as f64 * DEFAULT_SIGMA_XE * DEFAULT_SIGMA_XE;
    assert_flooded_noise(module, &lazy, &pt, &sk_out, pk_noise, &mut scratch);

    module.glwe_normalize_assign(&mut acc, &mut scratch.borrow());
    let mut eager: GLWE<AlignedBuf, i64> = module.glwe_alloc_from_infos(&layout);
    module.glwe_public_keyswitch_finalize(&mut eager, &ct, &acc, &mut scratch.borrow());
    assert_eq!(lazy, eager);
    acc.write_to(&mut Vec::new()).unwrap();
}

/// Finalizing into an output whose radix differs from the ciphertext's panics.
pub fn test_glwe_public_keyswitch_finalize_layout_mismatch<BE>(module: &Module<BE>)
where
    BE: HostBackend<OwnedBuf = AlignedBuf, ZnxWord = i64>,
    Module<BE>: GLWEPublicKeyswitchShare<BE>,
    ScratchOwned<BE>: ScratchOwnedAlloc<BE> + ScratchOwnedBorrow<BE>,
{
    let layout = glwe_layout(module);
    let ct_layout = GLWELayout {
        base2k: Base2K(BASE2K.0 + 1),
        ..layout
    };
    let ct: GLWE<AlignedBuf, i64> = module.glwe_alloc_from_infos(&ct_layout);
    let share: GLWE<AlignedBuf, i64> = module.glwe_alloc_from_infos(&layout);
    let mut res: GLWE<AlignedBuf, i64> = module.glwe_alloc_from_infos(&layout);
    let mut scratch: ScratchOwned<BE> = ScratchOwned::alloc(module.glwe_public_keyswitch_finalize_tmp_bytes());
    module.glwe_public_keyswitch_finalize(&mut res, &ct, &share, &mut scratch.borrow());
}

fn glwe_layout<BE: poulpy_hal::layouts::Backend>(module: &Module<BE>) -> GLWELayout {
    GLWELayout {
        n: module.n().into(),
        base2k: BASE2K,
        k: K,
        rank: RANK,
    }
}

fn flood_infos(layout: GLWELayout) -> EncryptionLayout<GLWELayout> {
    EncryptionLayout::new(layout, NoiseInfos::new(K.as_usize(), SIGMA_FLOOD, 6.0 * SIGMA_FLOOD).unwrap()).unwrap()
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
    enc_infos: &EncryptionLayout<GLWELayout>,
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
        enc_infos,
        &mut Source::new([31u8; 32]),
        &mut Source::new([32u8; 32]),
        &mut scratch.borrow(),
    );
    (pt, ct)
}

/// The noise of `ct` lies between the parties' flood noise and the sum of the
/// fresh, flood and `other` variances.
fn assert_flooded_noise<BE>(
    module: &Module<BE>,
    ct: &GLWE<AlignedBuf, i64>,
    pt: &GLWEPlaintext<AlignedBuf, i64>,
    sk: &GLWESecretPrepared<AlignedBuf, BE>,
    other: f64,
    scratch: &mut ScratchOwned<BE>,
) where
    BE: HostBackend<OwnedBuf = AlignedBuf, ZnxWord = i64>,
    for<'a> BE::BufRef<'a>: HostDataRef,
    for<'a> BE::BufMut<'a>: HostDataMut,
    Module<BE>: GLWENoise<BE>,
    ScratchOwned<BE>: ScratchOwnedBorrow<BE>,
{
    let k = K.as_usize() as f64;
    let flood = PARTIES as f64 * SIGMA_FLOOD * SIGMA_FLOOD;
    let lower = 0.5 * flood.log2() - k - 0.5;
    let upper = 0.5 * (DEFAULT_SIGMA_XE * DEFAULT_SIGMA_XE + flood + other).log2() - k + 0.5;
    let noise: f64 = module.glwe_noise(ct, pt, sk, &mut scratch.borrow()).std().log2();
    assert!(noise >= lower && noise <= upper, "noise {noise} outside [{lower}, {upper}]");
}
