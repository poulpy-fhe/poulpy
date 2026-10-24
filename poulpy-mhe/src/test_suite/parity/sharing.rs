use poulpy_core::{
    GLWEMaskFill, Noise, TransferInto,
    api::{GLWESecretPreparedFactory, GLWESecretSampling},
    layouts::{GLWE, GLWELayout, GLWEPlaintext, GLWESecret, ModuleCoreAlloc, TorusPrecision},
};
use poulpy_hal::{layouts::Module, source::Source};

use super::{
    ParityBackend, ReferenceBackend, Snapshot, fixture_secrets, keyswitch::ciphertext, poison_glwe, poison_glwe_compressed,
    prepare_secrets, snapshot_glwe, snapshot_glwe_compressed, snapshot_source, with_scratch,
};
use crate::{
    api::{GLWEEncToShareMHEProtocol, GLWEShareToEncMHEProtocol},
    layouts::MHEModuleAlloc,
    test_suite::fixtures::{BASE2K, K, K_OUT, PARTIES, RANK, SEEDS, bounded_integers, glwe_layout_at, integer_plaintext},
};

/// Encryption to shares at the fixture precision and at a whole number of
/// limbs: every public share and mask, aggregate, and the finalized share.
pub fn test_glwe_enc_to_share_parity<BR, BT>(module_ref: &Module<BR>, module_test: &Module<BT>)
where
    BR: ReferenceBackend,
    BT: ParityBackend,
    Module<BR>: MHEModuleAlloc<BR>
        + GLWEEncToShareMHEProtocol<BR>
        + GLWESecretSampling<BR>
        + GLWESecretPreparedFactory<BR>
        + GLWEMaskFill<BR>,
    Module<BT>: MHEModuleAlloc<BT> + GLWEEncToShareMHEProtocol<BT> + GLWESecretPreparedFactory<BT>,
{
    let (secrets_ref, secrets_test) = fixture_secrets(module_ref, module_test, RANK, 100);
    for k in [K, TorusPrecision(3 * BASE2K.0)] {
        let layout = glwe_layout_at(module_ref, k);
        let (ct_ref, ct_test) = ciphertext(module_ref, module_test, &layout, 30);
        let want = run_enc_to_share(module_ref, &layout, &ct_ref, &secrets_ref);
        let have = run_enc_to_share(module_test, &layout, &ct_test, &secrets_test);
        assert_eq!(want, have, "glwe enc to share, {layout:?}");
    }
}

fn run_enc_to_share<BE: ParityBackend>(
    module: &Module<BE>,
    layout: &GLWELayout,
    ct: &GLWE<BE::OwnedBuf, i64>,
    secrets: &[GLWESecret<BE::OwnedBuf, i64>],
) -> Vec<Snapshot>
where
    Module<BE>: MHEModuleAlloc<BE> + GLWEEncToShareMHEProtocol<BE> + GLWESecretPreparedFactory<BE>,
{
    let secrets = prepare_secrets(module, secrets);
    let flood = Noise::Gaussian { sigma: 1024.0 };
    let mut out = Vec::new();
    let mut masks: Vec<GLWEPlaintext<BE::OwnedBuf, i64>> =
        (0..PARTIES).map(|_| module.glwe_plaintext_alloc_from_infos(layout)).collect();
    let mut acc = module.glwe_enc_to_share_share_alloc_from_infos(layout);
    let mut public = module.glwe_enc_to_share_share_alloc_from_infos(layout);
    for (i, (sk, mask)) in secrets.iter().zip(masks.iter_mut()).enumerate() {
        let dst = if i == 0 { &mut acc } else { &mut public };
        poison_glwe::<BE, _>(&mut dst.inner);
        poison_glwe::<BE, _>(mask);
        let mut source_xm = Source::new([60 + i as u8; 32]);
        let mut source_smudge = Source::new([90 + i as u8; 32]);
        with_scratch::<BE, _>(module.mhe_glwe_enc_to_share_share_gen_tmp_bytes(ct), |scratch| {
            module.mhe_glwe_enc_to_share_share_gen(dst, mask, ct, sk, flood, &mut source_xm, &mut source_smudge, scratch)
        });
        out.push(snapshot_glwe::<BE, _>("share", &dst.inner));
        out.push(snapshot_glwe::<BE, _>("share mask", mask));
        out.push(snapshot_source("share source_xm", &mut source_xm));
        out.push(snapshot_source("share source_smudge", &mut source_smudge));
        if i > 0 {
            module.mhe_glwe_enc_to_share_share_aggregate(&mut acc, &public);
            out.push(snapshot_glwe::<BE, _>("aggregate", &acc.inner));
        }
    }
    with_scratch::<BE, _>(module.mhe_glwe_enc_to_share_share_finalize_tmp_bytes(), |scratch| {
        module.mhe_glwe_enc_to_share_share_finalize(&mut masks[0], ct, &acc, scratch)
    });
    let finalized = snapshot_glwe::<BE, _>("finalize", &masks[0]);
    finalized.assert_untagged();
    out.push(finalized);
    out.push(snapshot_glwe::<BE, _>("finalize share", &acc.inner));
    out
}

/// Shares to encryption at an output two limbs more precise than the shares:
/// every share, aggregate and the finalized ciphertext.
pub fn test_glwe_share_to_enc_parity<BR, BT>(module_ref: &Module<BR>, module_test: &Module<BT>)
where
    BR: ReferenceBackend,
    BT: ParityBackend,
    Module<BR>: MHEModuleAlloc<BR> + GLWEShareToEncMHEProtocol<BR> + GLWESecretSampling<BR> + GLWESecretPreparedFactory<BR>,
    Module<BT>: MHEModuleAlloc<BT> + GLWEShareToEncMHEProtocol<BT> + GLWESecretPreparedFactory<BT>,
{
    let in_layout = glwe_layout_at(module_ref, K);
    let out_layout = glwe_layout_at(module_ref, K_OUT);
    let (secrets_ref, secrets_test) = fixture_secrets(module_ref, module_test, RANK, 100);
    let (shares_ref, shares_test): (Vec<_>, Vec<_>) = (0..PARTIES)
        .map(|i| {
            let data = bounded_integers(module_ref.n(), K.as_usize(), [70 + i as u8; 32]);
            let share = integer_plaintext(module_ref, &in_layout, &data);
            let mut copy = module_test.glwe_plaintext_alloc_from_infos(&in_layout);
            share.transfer_into(&mut copy);
            (share, copy)
        })
        .unzip();
    let want = run_share_to_enc(module_ref, &in_layout, &out_layout, &shares_ref, &secrets_ref);
    let have = run_share_to_enc(module_test, &in_layout, &out_layout, &shares_test, &secrets_test);
    assert_eq!(want, have, "glwe share to enc");
}

fn run_share_to_enc<BE: ParityBackend>(
    module: &Module<BE>,
    in_layout: &GLWELayout,
    out_layout: &GLWELayout,
    shares: &[GLWEPlaintext<BE::OwnedBuf, i64>],
    secrets: &[GLWESecret<BE::OwnedBuf, i64>],
) -> Vec<Snapshot>
where
    Module<BE>: MHEModuleAlloc<BE> + GLWEShareToEncMHEProtocol<BE> + GLWESecretPreparedFactory<BE>,
{
    let secrets = prepare_secrets(module, secrets);
    let mut out = Vec::new();
    let mut acc = module.glwe_share_to_enc_share_alloc_from_infos(out_layout);
    let mut share = module.glwe_share_to_enc_share_alloc_from_infos(out_layout);
    for (i, (sk, secret)) in secrets.iter().zip(shares).enumerate() {
        let dst = if i == 0 { &mut acc } else { &mut share };
        poison_glwe_compressed::<BE, _>(&mut dst.inner);
        let mut source_xe = Source::new([80 + i as u8; 32]);
        with_scratch::<BE, _>(
            module.mhe_glwe_share_to_enc_share_gen_tmp_bytes(out_layout, in_layout),
            |scratch| module.mhe_glwe_share_to_enc_share_gen(dst, secret, sk, SEEDS[0], &mut source_xe, scratch),
        );
        out.push(snapshot_glwe_compressed::<BE, _>("share", &dst.inner));
        out.push(snapshot_source("share source_xe", &mut source_xe));
        if i > 0 {
            module.mhe_glwe_share_to_enc_share_aggregate(&mut acc, &share);
            out.push(snapshot_glwe_compressed::<BE, _>("aggregate", &acc.inner));
        }
    }
    let mut res = module.glwe_alloc_from_infos(out_layout);
    poison_glwe::<BE, _>(&mut res);
    with_scratch::<BE, _>(module.mhe_glwe_share_to_enc_share_finalize_tmp_bytes(), |scratch| {
        module.mhe_glwe_share_to_enc_share_finalize(&mut res, &acc, scratch)
    });
    out.push(snapshot_glwe::<BE, _>("finalize", &res));
    out.push(snapshot_glwe_compressed::<BE, _>("finalize share", &acc.inner));
    out
}
