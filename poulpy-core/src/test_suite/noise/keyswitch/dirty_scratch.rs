//! Regression coverage for the multi-digit gadget product.
//!
//! The product writes its destination through a *narrowed* view for every digit
//! past the first. Its single overwriting pass used to run through that same
//! narrowed view, so for `dsize >= 3` the top `dsize - 2` limbs were reachable
//! only by the accumulating passes and summed whatever the arena happened to
//! hold. Callers compensated by pre-zeroing, an obligation invisible at the call
//! site and not met on every path.
//!
//! Both tests run the operation twice over the same arena, once filled with
//! `0x00` and once with `0xFF`, and require bit-identical results.

use poulpy_hal::{
    api::{ScratchOwnedAlloc, ScratchOwnedBorrow, VecZnxFillUniformSourceBackend},
    layouts::{DigestU64, Module, ScalarZnx, ScratchOwned, ZnxViewMut},
    source::Source,
    test_suite::{TestParams, vec_znx_backend_mut},
};

use crate::layouts::GLWESecretSampling;
use crate::{
    EncryptionLayout, GGSWEncryptSk, GLWEEncryptSk, GLWEExternalProduct, GLWEKeyswitch, GLWESwitchingKeyEncryptSk,
    layouts::{
        GGSW, GGSWLayout, GGSWPreparedFactory, GLWE, GLWELayout, GLWEPlaintext, GLWESecret, GLWESecretPreparedFactory,
        GLWESwitchingKey, GLWESwitchingKeyLayout, GLWESwitchingKeyPreparedFactory, LWEInfos, ModuleCoreAlloc,
        prepared::{GGSWPrepared, GLWESecretPrepared, GLWESwitchingKeyPrepared},
    },
};

/// A keyswitch must not depend on the contents of the scratch it is handed.
///
/// The gadget product writes its destination through a *narrowed* view for every
/// digit past the first, so for `dsize >= 3` the top `dsize - 2` limbs used to be
/// reachable only by the accumulating passes. Callers papered over that by
/// zeroing the buffer first — an obligation that was invisible at the call site
/// and that the automorphism path did not meet. Running the same keyswitch over a
/// zeroed and a poisoned arena pins the buffer as self-initializing.
pub fn test_glwe_keyswitch_ignores_dirty_scratch<BE: crate::test_suite::noise::TestBackend>(
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
            pt_in.k().as_usize(),
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

/// The external product must not read whatever the arena happened to hold.
///
/// Its gadget product used to run its single *overwriting* pass through a view
/// narrowed by `dsize - 2` limbs, so for `dsize >= 3` the top limbs of the
/// destination were written only by the later accumulating passes and summed
/// stale scratch. Callers compensated by pre-zeroing `res_dft`, an obligation
/// invisible at the call site. Sibling of
/// `test_glwe_keyswitch_ignores_dirty_scratch`.
pub fn test_glwe_external_product_ignores_dirty_scratch<BE: crate::test_suite::noise::TestBackend>(
    params: &TestParams,
    module: &Module<BE>,
) where
    BE::OwnedBuf: poulpy_hal::layouts::HostDataMut,
    for<'a> BE::BufRef<'a>: poulpy_hal::layouts::HostDataRef,
    for<'a> BE::BufMut<'a>: poulpy_hal::layouts::HostDataMut,
    Module<BE>: GGSWEncryptSk<BE>
        + GGSWPreparedFactory<BE>
        + VecZnxFillUniformSourceBackend<BE>
        + GLWEExternalProduct<BE>
        + GLWEEncryptSk<BE>
        + GLWESecretPreparedFactory<BE>,
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
        let k_ggsw: usize = k_in + key_base2k * dsize;
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
            k: k_ggsw.into(),
            rank: rank.into(),
        };

        let ggsw_infos = EncryptionLayout::new_from_default_sigma(GGSWLayout {
            n: n.into(),
            base2k: key_base2k.into(),
            dnum: dnum.into(),
            k_aux: (dsize * key_base2k + module.log_n()).into(),
            dsize: dsize.into(),
            rank: rank.into(),
        })
        .unwrap();

        let mut ggsw: GGSW<BE::OwnedBuf, BE::ZnxWord> = module.ggsw_alloc_from_infos(&ggsw_infos);
        let mut glwe_in: GLWE<BE::OwnedBuf, BE::ZnxWord> = module.glwe_alloc_from_infos(&glwe_in_infos);
        let mut over_zeroed: GLWE<BE::OwnedBuf, BE::ZnxWord> = module.glwe_alloc_from_infos(&glwe_out_infos);
        let mut over_poisoned: GLWE<BE::OwnedBuf, BE::ZnxWord> = module.glwe_alloc_from_infos(&glwe_out_infos);
        let mut pt_ggsw: ScalarZnx<BE::OwnedBuf, BE::ZnxWord> = module.scalar_znx_alloc(1);
        let mut pt_in: GLWEPlaintext<BE::OwnedBuf, BE::ZnxWord> = module.glwe_plaintext_alloc_from_infos(&glwe_in_infos);

        let mut source_xs: Source = Source::new([0u8; 32]);
        let mut source_xe: Source = Source::new([0u8; 32]);
        let mut source_xa: Source = Source::new([0u8; 32]);

        module.vec_znx_fill_uniform_source_backend(
            base2k,
            pt_in.k().as_usize(),
            &mut vec_znx_backend_mut::<BE>(&mut pt_in.data),
            0,
            &mut source_xa,
        );
        pt_ggsw.raw_mut()[1] = 1; // X^1

        let mut scratch: ScratchOwned<BE> = ScratchOwned::alloc(
            module.ggsw_encrypt_sk_tmp_bytes(&ggsw_infos)
                | module.glwe_encrypt_sk_tmp_bytes(&glwe_in_infos)
                | module.glwe_external_product_tmp_bytes(&glwe_out_infos, &glwe_in_infos, &ggsw_infos),
        );

        let mut sk: GLWESecret<BE::OwnedBuf, BE::ZnxWord> = module.glwe_secret_alloc(rank.into());
        module.glwe_secret_fill_ternary_prob(&mut sk, 0.5, &mut source_xs);
        let mut sk_prepared: GLWESecretPrepared<BE::OwnedBuf, BE> = module.glwe_secret_prepared_alloc(rank.into());
        module.glwe_secret_prepare(&mut sk_prepared, &sk);

        module.ggsw_encrypt_sk(
            &mut ggsw,
            &pt_ggsw,
            &sk_prepared,
            &ggsw_infos,
            &mut source_xe,
            &mut source_xa,
            &mut scratch.borrow(),
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

        let mut ggsw_prepared: GGSWPrepared<BE::OwnedBuf, BE> = module.ggsw_prepared_alloc_from_infos(&ggsw);
        module.ggsw_prepare(&mut ggsw_prepared, &ggsw, &mut scratch.borrow());

        // 0x00 then 0xFF: as `f64` the poison is NaN and as `i64` it is -1, so any
        // limb read before being written shows up in the digest.
        <BE::OwnedBuf as AsMut<[u8]>>::as_mut(&mut scratch.data).fill(0x00);
        module.glwe_external_product(&mut over_zeroed, &glwe_in, &ggsw_prepared, &mut scratch.borrow());

        <BE::OwnedBuf as AsMut<[u8]>>::as_mut(&mut scratch.data).fill(0xFF);
        module.glwe_external_product(&mut over_poisoned, &glwe_in, &ggsw_prepared, &mut scratch.borrow());

        assert_eq!(
            over_zeroed.data().digest_u64(),
            over_poisoned.data().digest_u64(),
            "external product result depends on incoming scratch contents (rank={rank}, dsize={dsize})"
        );
    }
}
