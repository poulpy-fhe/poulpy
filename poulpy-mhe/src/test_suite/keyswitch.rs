//! Collective key switching over three parties: the finalized ciphertext
//! decrypts under the ideal output secret, its noise carries the parties'
//! smudging noise, and finalizing before or after normalization gives the
//! same result.

use poulpy_core::{
    DEFAULT_SIGMA_XE, EncryptionLayout, GLWEAdd, GLWEEncryptSk, GLWENoise, GLWENormalize, NoiseInfos,
    layouts::{
        GLWE, GLWELayout, GLWEPlaintext, GLWESecretPrepared, GLWESecretPreparedFactory, GLWESecretSampling, ModuleCoreAlloc, Rank,
    },
};
use poulpy_hal::{
    AlignedBuf,
    api::{ScratchOwnedAlloc, ScratchOwnedBorrow, VecZnxFillUniformSource},
    layouts::{HostBackend, HostDataMut, HostDataRef, Module, ScratchOwned, WriterTo},
    source::Source,
    test_suite::vec_znx_backend_mut,
};

use super::fixtures::{BASE2K, K, PARTIES, RANK, Secret, ideal_secret, party_secrets, secret_from_seed};
use crate::api::GLWEKeyswitchShare;

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
