//! Semantic coverage for the known-zero-limb-prefix key switch.

use poulpy_hal::{
    api::{ScratchOwnedAlloc, ScratchOwnedBorrow, VecZnxFillUniformSourceBackend},
    layouts::{DigestU64, Module, ScratchOwned, ZnxViewMut},
    source::Source,
    test_suite::{TestParams, vec_znx_backend_mut},
};

use crate::layouts::GLWESecretSampling;
use crate::{
    EncryptionLayout, GLWEEncryptSk, GLWEKeyswitch, GLWESwitchingKeyEncryptSk,
    layouts::{
        GLWE, GLWELayout, GLWEPlaintext, GLWESecret, GLWESecretPreparedFactory, GLWESwitchingKey, GLWESwitchingKeyLayout,
        GLWESwitchingKeyPreparedFactory, ModuleCoreAlloc,
        prepared::{GLWESecretPrepared, GLWESwitchingKeyPrepared},
    },
};

/// Skipping the forward transform of known-zero limbs must not change the
/// result. The prefix is zeroed in the coefficient domain first, so both paths
/// see the same ciphertext.
pub fn test_glwe_keyswitch_zero_prefix_matches_plain<BE: crate::test_suite::noise::TestBackend>(
    params: &TestParams,
    module: &Module<BE>,
) where
    BE::OwnedBuf: poulpy_hal::layouts::HostDataMut,
    for<'a> BE::BufRef<'a>: poulpy_hal::layouts::HostDataRef,
    for<'a> BE::BufMut<'a>: poulpy_hal::layouts::HostDataMut,
    Module<BE>: VecZnxFillUniformSourceBackend<BE>
        + GLWESwitchingKeyEncryptSk<BE>
        + GLWEEncryptSk<BE>
        + GLWEKeyswitch<BE>
        + GLWESecretPreparedFactory<BE>
        + GLWESwitchingKeyPreparedFactory<BE>,
    ScratchOwned<BE>: ScratchOwnedAlloc<BE> + ScratchOwnedBorrow<BE>,
{
    let base2k: usize = params.base2k;
    let n: usize = module.n();
    let rank: usize = 1;
    let size: usize = 5;
    let k: usize = size * base2k;

    for dsize in 1_usize..4 {
        let dnum: usize = k.div_ceil(base2k * dsize);
        let glwe_infos = EncryptionLayout::new_from_default_sigma(GLWELayout {
            n: n.into(),
            base2k: base2k.into(),
            k: k.into(),
            rank: rank.into(),
        })
        .unwrap();
        let alt_infos = EncryptionLayout::new_from_default_sigma(GLWELayout {
            n: n.into(),
            base2k: (base2k - 1).into(),
            k: k.into(),
            rank: rank.into(),
        })
        .unwrap();
        let ksk_infos = EncryptionLayout::new_from_default_sigma(GLWESwitchingKeyLayout {
            n: n.into(),
            base2k: base2k.into(),
            dnum: dnum.into(),
            k_aux: (dsize * base2k + module.log_n()).into(),
            dsize: dsize.into(),
            rank_in: rank.into(),
            rank_out: rank.into(),
        })
        .unwrap();

        let mut source_xs: Source = Source::new([3u8; 32]);
        let mut source_xe: Source = Source::new([5u8; 32]);
        let mut source_xa: Source = Source::new([7u8; 32]);

        let mut ksk: GLWESwitchingKey<BE::OwnedBuf, BE::ZnxWord> = module.glwe_switching_key_alloc_from_infos(&ksk_infos);
        let mut pt: GLWEPlaintext<BE::OwnedBuf, BE::ZnxWord> = module.glwe_plaintext_alloc_from_infos(&glwe_infos);
        module.vec_znx_fill_uniform_source_backend(base2k, &mut vec_znx_backend_mut::<BE>(&mut pt.data), 0, &mut source_xa);

        let mut scratch: ScratchOwned<BE> = ScratchOwned::alloc(
            module.glwe_switching_key_encrypt_sk_tmp_bytes(&ksk_infos)
                | module.glwe_encrypt_sk_tmp_bytes(&glwe_infos)
                | module.glwe_encrypt_sk_tmp_bytes(&alt_infos)
                | module.glwe_keyswitch_tmp_bytes(&glwe_infos, &glwe_infos, &ksk_infos)
                | module.glwe_keyswitch_tmp_bytes(&alt_infos, &alt_infos, &ksk_infos),
        );

        let mut sk: GLWESecret<BE::OwnedBuf, BE::ZnxWord> = module.glwe_secret_alloc(rank.into());
        module.glwe_secret_fill_ternary_prob(&mut sk, 0.5, &mut source_xs);
        let mut sk_prepared: GLWESecretPrepared<BE::OwnedBuf, BE> = module.glwe_secret_prepared_alloc(rank.into());
        module.glwe_secret_prepare(&mut sk_prepared, &sk);

        module.glwe_switching_key_encrypt_sk(
            &mut ksk,
            &sk,
            &sk,
            &ksk_infos,
            &mut source_xe,
            &mut source_xa,
            &mut scratch.arena(),
        );
        let mut ksk_prepared: GLWESwitchingKeyPrepared<BE::OwnedBuf, BE> =
            module.glwe_switching_key_prepared_alloc_from_infos(&ksk);
        module.glwe_switching_key_prepare(&mut ksk_prepared, &ksk, &mut scratch.borrow());

        let mut base: GLWE<BE::OwnedBuf, BE::ZnxWord> = module.glwe_alloc_from_infos(&glwe_infos);
        module.glwe_encrypt_sk(
            &mut base,
            &pt,
            &sk_prepared,
            &glwe_infos,
            &mut source_xe,
            &mut source_xa,
            &mut scratch.borrow(),
        );

        for zero_limbs in 0..=size {
            let mut plain = base.clone();
            for col in 0..rank + 1 {
                for limb in 0..zero_limbs {
                    plain.data.at_mut(col, limb).fill(Default::default());
                }
            }
            let mut hinted = plain.clone();

            module.glwe_keyswitch_assign(&mut plain, &ksk_prepared, &mut scratch.borrow());
            module.glwe_keyswitch_assign_zero_prefix(&mut hinted, &ksk_prepared, zero_limbs, &mut scratch.borrow());

            assert_eq!(
                plain.data().digest_u64(),
                hinted.data().digest_u64(),
                "zero-prefix keyswitch diverges from the plain path (dsize={dsize}, zero_limbs={zero_limbs})"
            );
        }

        // A radix change re-cuts the limb boundaries, so the hint is only
        // accepted at zero; the conversion branch must still agree there.
        let mut alt_pt: GLWEPlaintext<BE::OwnedBuf, BE::ZnxWord> = module.glwe_plaintext_alloc_from_infos(&alt_infos);
        module.vec_znx_fill_uniform_source_backend(
            base2k - 1,
            &mut vec_znx_backend_mut::<BE>(&mut alt_pt.data),
            0,
            &mut source_xa,
        );
        let mut plain: GLWE<BE::OwnedBuf, BE::ZnxWord> = module.glwe_alloc_from_infos(&alt_infos);
        module.glwe_encrypt_sk(
            &mut plain,
            &alt_pt,
            &sk_prepared,
            &alt_infos,
            &mut source_xe,
            &mut source_xa,
            &mut scratch.borrow(),
        );
        let mut hinted = plain.clone();
        module.glwe_keyswitch_assign(&mut plain, &ksk_prepared, &mut scratch.borrow());
        module.glwe_keyswitch_assign_zero_prefix(&mut hinted, &ksk_prepared, 0, &mut scratch.borrow());
        assert_eq!(
            plain.data().digest_u64(),
            hinted.data().digest_u64(),
            "cross-radix keyswitch diverges from the plain path (dsize={dsize})"
        );
    }
}
