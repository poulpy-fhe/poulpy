//! Phase 3 coverage: the staged keyswitch.
//!
//! Drives `glwe_keyswitch_into_big` + `glwe_finalize_big_into` directly through
//! the public API and checks the pair reproduces `glwe_keyswitch` bit for bit,
//! across mismatched input/key/output `base2k` and every `dsize`.
//!
//! The ergonomic path is implemented as this composition, so a divergence here
//! means one of the two stages grew a behavior the other did not.

use poulpy_hal::{
    api::{ScratchOwnedAlloc, ScratchOwnedBorrow, VecZnxFillUniformSourceBackend},
    layouts::{DigestU64, Module, ScratchOwned},
    source::Source,
    test_suite::{TestParams, vec_znx_backend_mut},
};

use crate::{
    EncryptionLayout, GLWEEncryptSk, GLWEKeyswitch, GLWESwitchingKeyEncryptSk,
    api::{GLWEFinalizeBig, GLWEKeyswitchIntoBig},
    layouts::{
        GLWE, GLWEInfos, GLWELayout, GLWEPlaintext, GLWESecret, GLWESecretPreparedFactory, GLWESwitchingKey,
        GLWESwitchingKeyLayout, GLWESwitchingKeyPreparedFactory, LWEInfos, ModuleCoreAlloc,
        prepared::{GLWESecretPrepared, GLWESwitchingKeyPrepared},
    },
    scratch::ScratchArenaTakeCore,
};

/// `into_big -> finalize` must equal the single-shot `glwe_keyswitch`.
pub fn test_glwe_keyswitch_staged_matches_default<BE: crate::test_suite::TestBackend>(params: &TestParams, module: &Module<BE>)
where
    BE::OwnedBuf: poulpy_hal::layouts::HostDataMut,
    for<'a> BE::BufRef<'a>: poulpy_hal::layouts::HostDataRef,
    for<'a> BE::BufMut<'a>: poulpy_hal::layouts::HostDataMut,
    Module<BE>: VecZnxFillUniformSourceBackend<BE>
        + GLWESwitchingKeyEncryptSk<BE>
        + GLWEEncryptSk<BE>
        + GLWEKeyswitch<BE>
        + GLWEKeyswitchIntoBig<BE>
        + GLWEFinalizeBig<BE>
        + GLWESecretPreparedFactory<BE>
        + GLWESwitchingKeyPreparedFactory<BE>,
    ScratchOwned<BE>: ScratchOwnedAlloc<BE> + ScratchOwnedBorrow<BE>,
{
    let base2k: usize = params.base2k;
    // Deliberately mismatched widths: exercises the base2k-conversion branch of
    // `into_big`, which is where the body fold and the mask product disagree.
    let in_base2k: usize = base2k - 1;
    let key_base2k: usize = base2k;
    let out_base2k: usize = base2k - 2;
    let k_in: usize = 4 * in_base2k + 1;
    let max_dsize: usize = k_in.div_ceil(key_base2k);
    let n: usize = module.n();

    for rank in 1_usize..3 {
        for dsize in 1_usize..max_dsize + 1 {
            let k_ksk: usize = k_in + key_base2k * dsize;
            let dnum: usize = k_in.div_ceil(key_base2k * dsize);

            let glwe_in_infos = EncryptionLayout::new_from_default_sigma(GLWELayout {
                n: n.into(),
                base2k: in_base2k.into(),
                k: k_in.into(),
                rank: rank.into(),
            })
            .unwrap();

            let glwe_out_infos: GLWELayout = GLWELayout {
                n: n.into(),
                base2k: out_base2k.into(),
                k: k_ksk.into(),
                rank: rank.into(),
            };

            let ksk_infos = EncryptionLayout::new_from_default_sigma(GLWESwitchingKeyLayout {
                n: n.into(),
                base2k: key_base2k.into(),
                dnum: dnum.into(),
                k_aux: (dsize * key_base2k + module.log_n()).into(),
                dsize: dsize.into(),
                rank_in: rank.into(),
                rank_out: rank.into(),
            })
            .unwrap();

            let mut ksk: GLWESwitchingKey<BE::OwnedBuf, BE::ZnxWord> = module.glwe_switching_key_alloc_from_infos(&ksk_infos);
            let mut glwe_in: GLWE<BE::OwnedBuf, BE::ZnxWord> = module.glwe_alloc_from_infos(&glwe_in_infos);
            let mut expected: GLWE<BE::OwnedBuf, BE::ZnxWord> = module.glwe_alloc_from_infos(&glwe_out_infos);
            let mut staged: GLWE<BE::OwnedBuf, BE::ZnxWord> = module.glwe_alloc_from_infos(&glwe_out_infos);
            let mut pt_in: GLWEPlaintext<BE::OwnedBuf, BE::ZnxWord> = module.glwe_plaintext_alloc_from_infos(&glwe_in_infos);

            let mut source_xs: Source = Source::new([0u8; 32]);
            let mut source_xe: Source = Source::new([0u8; 32]);
            let mut source_xa: Source = Source::new([0u8; 32]);

            module.vec_znx_fill_uniform_source_backend(
                pt_in.base2k().into(),
                &mut vec_znx_backend_mut::<BE>(&mut pt_in.data),
                0,
                &mut source_xa,
            );

            let big_layout = crate::layouts::glwe_keyswitch_big_layout(&glwe_in_infos, &ksk_infos);
            let staged_bytes =
                <BE as poulpy_hal::layouts::Backend>::bytes_of_vec_znx_big(n, (big_layout.rank() + 1).into(), big_layout.size())
                    + module
                        .glwe_keyswitch_into_big_tmp_bytes(&glwe_out_infos, &glwe_in_infos, &ksk_infos)
                        .max(module.glwe_finalize_big_tmp_bytes());

            let mut scratch: ScratchOwned<BE> = ScratchOwned::alloc(
                module.glwe_switching_key_encrypt_sk_tmp_bytes(&ksk_infos)
                    | module.glwe_encrypt_sk_tmp_bytes(&glwe_in_infos)
                    | module.glwe_keyswitch_tmp_bytes(&glwe_out_infos, &glwe_in_infos, &ksk_infos)
                    | staged_bytes,
            );

            let mut sk: GLWESecret<BE::OwnedBuf, BE::ZnxWord> = module.glwe_secret_alloc(rank.into());
            sk.fill_ternary_prob(0.5, &mut source_xs);
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

            module.glwe_encrypt_sk(
                &mut glwe_in,
                &pt_in,
                &sk_prepared,
                &glwe_in_infos,
                &mut source_xe,
                &mut source_xa,
                &mut scratch.borrow(),
            );

            let mut ksk_prepared: GLWESwitchingKeyPrepared<BE::OwnedBuf, BE> =
                module.glwe_switching_key_prepared_alloc_from_infos(&ksk);
            module.glwe_switching_key_prepare(&mut ksk_prepared, &ksk, &mut scratch.borrow());

            // Reference: the ergonomic single-shot path.
            module.glwe_keyswitch(&mut expected, &glwe_in, &ksk_prepared, &mut scratch.borrow());

            // Staged: stop in the big domain, then finalize into the destination.
            scratch.borrow().scope(|arena| {
                let (mut acc, mut arena_1) = arena.take_glwe_big_scratch(module, &big_layout);

                // The accumulator carries the key's base2k, not the output's.
                assert_eq!(acc.base2k(), ksk_infos.base2k());
                assert_ne!(acc.base2k(), glwe_out_infos.base2k());

                module.glwe_keyswitch_into_big(&mut acc, &glwe_in, &ksk_prepared, &mut arena_1.borrow());
                module.glwe_finalize_big_into(&mut staged, &acc, &mut arena_1.borrow());
            });

            assert_eq!(
                staged.data().digest_u64(),
                expected.data().digest_u64(),
                "staged keyswitch differs from the single-shot path (rank={rank}, dsize={dsize})"
            );
        }
    }
}

/// A keyswitch must not depend on the contents of the scratch it is handed.
///
/// The gadget product writes its destination through a *narrowed* view for every
/// digit past the first, so for `dsize >= 3` the top `dsize - 2` limbs used to be
/// reachable only by the accumulating passes. Callers papered over that by
/// zeroing the buffer first — an obligation that was invisible at the call site
/// and that the automorphism path did not meet. Running the same keyswitch over a
/// zeroed and a poisoned arena pins the buffer as self-initializing.
pub fn test_glwe_keyswitch_ignores_dirty_scratch<BE: crate::test_suite::TestBackend>(params: &TestParams, module: &Module<BE>)
where
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
    let key_base2k: usize = base2k;
    let k_in: usize = 4 * base2k + 1;
    // `dsize >= 3` is what opens the gap; sweep the whole range anyway.
    let max_dsize: usize = k_in.div_ceil(key_base2k);
    let n: usize = module.n();
    let rank: usize = 1;

    for dsize in 1_usize..max_dsize + 1 {
        let k_ksk: usize = k_in + key_base2k * dsize;
        let dnum: usize = k_in.div_ceil(key_base2k * dsize);

        let glwe_in_infos = EncryptionLayout::new_from_default_sigma(GLWELayout {
            n: n.into(),
            base2k: base2k.into(),
            k: k_in.into(),
            rank: rank.into(),
        })
        .unwrap();

        let glwe_out_infos: GLWELayout = GLWELayout {
            n: n.into(),
            base2k: base2k.into(),
            k: k_ksk.into(),
            rank: rank.into(),
        };

        let ksk_infos = EncryptionLayout::new_from_default_sigma(GLWESwitchingKeyLayout {
            n: n.into(),
            base2k: key_base2k.into(),
            dnum: dnum.into(),
            k_aux: (dsize * key_base2k + module.log_n()).into(),
            dsize: dsize.into(),
            rank_in: rank.into(),
            rank_out: rank.into(),
        })
        .unwrap();

        let mut ksk: GLWESwitchingKey<BE::OwnedBuf, BE::ZnxWord> = module.glwe_switching_key_alloc_from_infos(&ksk_infos);
        let mut glwe_in: GLWE<BE::OwnedBuf, BE::ZnxWord> = module.glwe_alloc_from_infos(&glwe_in_infos);
        let mut over_zeroed: GLWE<BE::OwnedBuf, BE::ZnxWord> = module.glwe_alloc_from_infos(&glwe_out_infos);
        let mut over_poisoned: GLWE<BE::OwnedBuf, BE::ZnxWord> = module.glwe_alloc_from_infos(&glwe_out_infos);
        let mut pt_in: GLWEPlaintext<BE::OwnedBuf, BE::ZnxWord> = module.glwe_plaintext_alloc_from_infos(&glwe_in_infos);

        let mut source_xs: Source = Source::new([0u8; 32]);
        let mut source_xe: Source = Source::new([0u8; 32]);
        let mut source_xa: Source = Source::new([0u8; 32]);

        module.vec_znx_fill_uniform_source_backend(
            pt_in.base2k().into(),
            &mut vec_znx_backend_mut::<BE>(&mut pt_in.data),
            0,
            &mut source_xa,
        );

        let mut scratch: ScratchOwned<BE> = ScratchOwned::alloc(
            module.glwe_switching_key_encrypt_sk_tmp_bytes(&ksk_infos)
                | module.glwe_encrypt_sk_tmp_bytes(&glwe_in_infos)
                | module.glwe_keyswitch_tmp_bytes(&glwe_out_infos, &glwe_in_infos, &ksk_infos),
        );

        let mut sk: GLWESecret<BE::OwnedBuf, BE::ZnxWord> = module.glwe_secret_alloc(rank.into());
        sk.fill_ternary_prob(0.5, &mut source_xs);
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

        module.glwe_encrypt_sk(
            &mut glwe_in,
            &pt_in,
            &sk_prepared,
            &glwe_in_infos,
            &mut source_xe,
            &mut source_xa,
            &mut scratch.borrow(),
        );

        let mut ksk_prepared: GLWESwitchingKeyPrepared<BE::OwnedBuf, BE> =
            module.glwe_switching_key_prepared_alloc_from_infos(&ksk);
        module.glwe_switching_key_prepare(&mut ksk_prepared, &ksk, &mut scratch.borrow());

        // 0x00 then 0xFF: as `f64` the poison is NaN and as `i64` it is -1, so any
        // limb read before being written shows up in the digest.
        <BE::OwnedBuf as AsMut<[u8]>>::as_mut(&mut scratch.data).fill(0x00);
        module.glwe_keyswitch(&mut over_zeroed, &glwe_in, &ksk_prepared, &mut scratch.borrow());

        <BE::OwnedBuf as AsMut<[u8]>>::as_mut(&mut scratch.data).fill(0xFF);
        module.glwe_keyswitch(&mut over_poisoned, &glwe_in, &ksk_prepared, &mut scratch.borrow());

        assert_eq!(
            over_zeroed.data().digest_u64(),
            over_poisoned.data().digest_u64(),
            "keyswitch result depends on incoming scratch contents (rank={rank}, dsize={dsize})"
        );
    }
}
