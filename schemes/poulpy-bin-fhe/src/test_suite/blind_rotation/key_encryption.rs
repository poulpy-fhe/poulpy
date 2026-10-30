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

/// The refusal of each case, kept apart from the secrets that trigger it.
const REFUSALS: [&str; 2] = [
    "the key count differs from the LWE secret dimension",
    "the GLWE secret rank differs from the key rank",
];

/// The secrets of refused case `case`: an LWE secret of another dimension than the key count, then a GLWE secret of
/// another rank than the keys.
fn refused_secrets<BE: Backend<ZnxWord = i64>>(
    module: &Module<BE>,
    layout: &BlindRotationKeyLayout,
    case: usize,
) -> (LWESecret<BE::OwnedBuf, i64>, GLWESecretPrepared<BE::OwnedBuf, BE>)
where
    Module<BE>: GLWESecretPreparedFactory<BE> + LWESecretSampling<BE> + ModuleCoreAlloc<OwnedBuf = BE::OwnedBuf, ZnxWord = i64>,
{
    let n_lwe = layout.n_lwe.as_usize() + usize::from(case == 0);
    let rank = Rank(layout.rank.as_u32() + u32::from(case == 1));
    let mut sk_lwe = module.lwe_secret_alloc(n_lwe.into());
    module.lwe_secret_fill_binary_block(&mut sk_lwe, 1, &mut Source::new([1u8; 32]));
    (sk_lwe, module.glwe_secret_prepared_alloc(rank))
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
    for (case, message) in REFUSALS.into_iter().enumerate() {
        let (sk_lwe, sk_glwe) = refused_secrets(module, &layout, case);
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
    for (case, message) in REFUSALS.into_iter().enumerate() {
        let (sk_lwe, sk_glwe) = refused_secrets(module, &layout, case);
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
