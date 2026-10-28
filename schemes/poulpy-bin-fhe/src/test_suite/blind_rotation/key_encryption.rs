//! Refusals of the blind-rotation key encryptions, each before any write or draw.
use poulpy_core::{
    api::{GLWESecretPreparedFactory, LWESecretSampling},
    layouts::{GLWESecretPrepared, LWESecret, ModuleCoreAlloc, Rank},
    test_suite::with_scratch,
};
use poulpy_hal::AlignedBuf;
use poulpy_hal::{
    layouts::{Backend, Module},
    source::Source,
};

use crate::{
    api::{BlindRotationKeyCompressedEncryptSk, BlindRotationKeyCompressedFactory, BlindRotationKeyEncryptSk},
    blind_rotation::{BlindRotationKey, BlindRotationKeyLayout, CGGI},
    test_suite::parity::assert_panics_with,
};

fn layout<BE: Backend>(module: &Module<BE>) -> BlindRotationKeyLayout {
    BlindRotationKeyLayout {
        n_glwe: (module.n() as u32).into(),
        n_lwe: 8_usize.into(),
        base2k: 12_usize.into(),
        dnum: 2_usize.into(),
        k_aux: 30_usize.into(),
        rank: 1_usize.into(),
    }
}

/// A refused case: the LWE secret, the GLWE secret and the expected message.
type RefusedSecrets<BE> = (
    LWESecret<<BE as Backend>::OwnedBuf, i64>,
    GLWESecretPrepared<<BE as Backend>::OwnedBuf, BE>,
    &'static str,
);

/// The secrets of each refused case: an LWE secret of another dimension than the key count, then a GLWE secret of
/// another rank than the keys.
fn refused_secrets<BE: Backend<ZnxWord = i64>>(module: &Module<BE>, layout: &BlindRotationKeyLayout) -> [RefusedSecrets<BE>; 2]
where
    Module<BE>: GLWESecretPreparedFactory<BE> + LWESecretSampling<BE> + ModuleCoreAlloc<OwnedBuf = BE::OwnedBuf, ZnxWord = i64>,
{
    let n_lwe = layout.n_lwe.as_usize();
    let mut source_xs = Source::new([1u8; 32]);
    let mut secret = |n: usize| {
        let mut sk = module.lwe_secret_alloc(n.into());
        module.lwe_secret_fill_binary_block(&mut sk, 1, &mut source_xs);
        sk
    };
    [
        (
            secret(n_lwe + 1),
            module.glwe_secret_prepared_alloc(layout.rank),
            "the key count differs from the LWE secret dimension",
        ),
        (
            secret(n_lwe),
            module.glwe_secret_prepared_alloc(Rank(layout.rank.as_u32() + 1)),
            "the GLWE secret rank differs from the key rank",
        ),
    ]
}

fn assert_sources_untouched(source_xe: &mut Source, source_xa: &mut Source) {
    assert!(
        source_xe.new_seed() == Source::new([3u8; 32]).new_seed() && source_xa.new_seed() == Source::new([2u8; 32]).new_seed(),
        "a refused key encryption drew from its sources"
    );
}

pub fn test_blind_rotation_key_encrypt_sk_rejects_secret_shape<BE: Backend<OwnedBuf = AlignedBuf, ZnxWord = i64>>(
    module: &Module<BE>,
) where
    Module<BE>: BlindRotationKeyEncryptSk<CGGI, BE>
        + GLWESecretPreparedFactory<BE>
        + LWESecretSampling<BE>
        + ModuleCoreAlloc<OwnedBuf = BE::OwnedBuf, ZnxWord = i64>,
{
    let layout = layout(module);
    for (sk_lwe, sk_glwe, message) in refused_secrets(module, &layout) {
        let mut res: BlindRotationKey<BE::OwnedBuf, CGGI, i64> = BlindRotationKey::alloc(module, &layout);
        let before = res.clone();
        let mut source_xe = Source::new([3u8; 32]);
        let mut source_xa = Source::new([2u8; 32]);
        assert_panics_with(message, || {
            with_scratch::<BE, _>(0, |s| {
                module.blind_rotation_key_encrypt_sk(&mut res, &sk_lwe, &sk_glwe, &mut source_xe, &mut source_xa, s)
            })
        });
        assert!(res == before, "a refused key encryption changed its destination");
        assert_sources_untouched(&mut source_xe, &mut source_xa);
    }
}

pub fn test_blind_rotation_key_compressed_encrypt_sk_rejects_secret_shape<BE: Backend<OwnedBuf = AlignedBuf, ZnxWord = i64>>(
    module: &Module<BE>,
) where
    Module<BE>: BlindRotationKeyCompressedEncryptSk<CGGI, BE>
        + BlindRotationKeyCompressedFactory<CGGI, BE>
        + GLWESecretPreparedFactory<BE>
        + LWESecretSampling<BE>
        + ModuleCoreAlloc<OwnedBuf = BE::OwnedBuf, ZnxWord = i64>,
{
    let layout = layout(module);
    for (sk_lwe, sk_glwe, message) in refused_secrets(module, &layout) {
        let mut res = module.blind_rotation_key_compressed_alloc_from_infos(&layout);
        let before = res.clone();
        let mut source_xe = Source::new([3u8; 32]);
        let mut source_xa = Source::new([2u8; 32]);
        assert_panics_with(message, || {
            with_scratch::<BE, _>(0, |s| {
                module.blind_rotation_key_compressed_encrypt_sk(&mut res, &sk_lwe, &sk_glwe, [5u8; 32], &mut source_xe, s)
            })
        });
        assert!(res == before, "a refused key encryption changed its destination");
        assert_sources_untouched(&mut source_xe, &mut source_xa);
    }
}
