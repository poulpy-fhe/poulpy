use poulpy_core::{
    GLWEMaskFill, Noise, TransferInto,
    api::{GLWEPublicKeyPreparedFactory, GLWESecretPreparedFactory, GLWESecretSampling},
    layouts::{GLWE, GLWELayout, GLWEPublicKey, GLWESecret, ModuleCoreAlloc, Rank, TorusPrecision},
    test_suite::with_scratch,
};
use poulpy_hal::{AlignedBuf, layouts::Module, source::Source};

use super::{
    ParityBackend, ReferenceBackend, Snapshot, fixture_secrets, poison_glwe, prepare_secrets, public_key::collective_public_key,
    snapshot_glwe, snapshot_source,
};
use crate::{
    api::{GLWEPrivateKeyswitchMHEProtocol, GLWEPublicKeyMHEProtocol, GLWEPublicKeyswitchMHEProtocol},
    layouts::MHEModuleAlloc,
    test_suite::fixtures::{BASE2K, K, PARTIES, RANK, glwe_layout_at, secret_from_seed_at},
};

const FLOOD: Noise = Noise::Gaussian { sigma: 1024.0 };

/// A canonical uniform ciphertext at `layout`, filled on `module_ref` and transferred.
pub(crate) fn ciphertext<BR, BT>(
    module_ref: &Module<BR>,
    module_test: &Module<BT>,
    layout: &GLWELayout,
    seed: u8,
) -> (GLWE<AlignedBuf, i64>, GLWE<BT::OwnedBuf, i64>)
where
    BR: ReferenceBackend,
    BT: ParityBackend,
    Module<BR>: GLWEMaskFill<BR> + ModuleCoreAlloc<OwnedBuf = AlignedBuf, ZnxWord = i64>,
    Module<BT>: ModuleCoreAlloc<OwnedBuf = BT::OwnedBuf, ZnxWord = i64>,
{
    let mut ct_ref = module_ref.glwe_alloc_from_infos(layout);
    module_ref.fill_glwe_from_source(&mut ct_ref, &mut Source::new([seed; 32]));
    let mut ct_test = module_test.glwe_alloc_from_infos(layout);
    ct_ref.transfer_into(&mut ct_test);
    (ct_ref, ct_test)
}

/// Private key switching at the ciphertext's precision and one limb below,
/// under a Gaussian and a uniform flood: every share, aggregate and the
/// finalized ciphertext.
pub fn test_glwe_private_keyswitch_parity<BR, BT>(module_ref: &Module<BR>, module_test: &Module<BT>)
where
    BR: ReferenceBackend,
    BT: ParityBackend,
    Module<BR>: MHEModuleAlloc<BR>
        + GLWEPrivateKeyswitchMHEProtocol<BR>
        + GLWESecretSampling<BR>
        + GLWESecretPreparedFactory<BR>
        + GLWEMaskFill<BR>,
    Module<BT>: MHEModuleAlloc<BT> + GLWEPrivateKeyswitchMHEProtocol<BT> + GLWESecretPreparedFactory<BT>,
{
    let layout = glwe_layout_at(module_ref, K);
    let (ct_ref, ct_test) = ciphertext(module_ref, module_test, &layout, 30);
    let (in_ref, in_test) = fixture_secrets(module_ref, module_test, RANK, 150);
    let (out_ref, out_test) = fixture_secrets(module_ref, module_test, RANK, 100);
    for share_k in [K, TorusPrecision(K.0 - BASE2K.0)] {
        for flood in [FLOOD, Noise::Uniform { bits: 10 }] {
            let share_layout = GLWELayout { k: share_k, ..layout };
            let want = run_private(module_ref, &share_layout, &ct_ref, &in_ref, &out_ref, flood);
            let have = run_private(module_test, &share_layout, &ct_test, &in_test, &out_test, flood);
            assert_eq!(want, have, "glwe private keyswitch, {share_layout:?}, {flood:?}");
        }
    }
}

fn run_private<BE: ParityBackend>(
    module: &Module<BE>,
    share_layout: &GLWELayout,
    ct: &GLWE<BE::OwnedBuf, i64>,
    secrets_in: &[GLWESecret<BE::OwnedBuf, i64>],
    secrets_out: &[GLWESecret<BE::OwnedBuf, i64>],
    flood: Noise,
) -> Vec<Snapshot>
where
    Module<BE>: MHEModuleAlloc<BE> + GLWEPrivateKeyswitchMHEProtocol<BE> + GLWESecretPreparedFactory<BE>,
{
    let secrets_in = prepare_secrets(module, secrets_in);
    let secrets_out = prepare_secrets(module, secrets_out);
    let mut out = Vec::new();
    let mut acc = module.glwe_private_keyswitch_share_alloc_from_infos(share_layout);
    let mut share = module.glwe_private_keyswitch_share_alloc_from_infos(share_layout);
    for (i, (sk_in, sk_out)) in secrets_in.iter().zip(&secrets_out).enumerate() {
        let dst = if i == 0 { &mut acc } else { &mut share };
        poison_glwe::<BE, _>(&mut dst.inner);
        let mut source_smudge = Source::new([10 + i as u8; 32]);
        with_scratch::<BE, _>(module.mhe_glwe_private_keyswitch_share_gen_tmp_bytes(ct), |scratch| {
            module.mhe_glwe_private_keyswitch_share_gen(dst, ct, sk_in, sk_out, flood, &mut source_smudge, scratch)
        });
        out.push(snapshot_glwe::<BE, _>("share", &dst.inner));
        out.push(snapshot_source("share source_smudge", &mut source_smudge));
        if i > 0 {
            module.mhe_glwe_private_keyswitch_share_aggregate(&mut acc, &share);
            out.push(snapshot_glwe::<BE, _>("aggregate", &acc.inner));
        }
    }
    let mut res = module.glwe_alloc_from_infos(ct);
    poison_glwe::<BE, _>(&mut res);
    with_scratch::<BE, _>(module.mhe_glwe_private_keyswitch_share_finalize_tmp_bytes(), |scratch| {
        module.mhe_glwe_private_keyswitch_share_finalize(&mut res, ct, &acc, scratch)
    });
    let finalized = snapshot_glwe::<BE, _>("finalize", &res);
    finalized.assert_untagged();
    out.push(finalized);
    out.push(snapshot_glwe::<BE, _>("finalize share", &acc.inner));
    out
}

/// Public key switching to the input rank and precision, to rank 1 one limb
/// above, and one limb below at the key's precision: every share, aggregate
/// and the finalized ciphertext.
pub fn test_glwe_public_keyswitch_parity<BR, BT>(module_ref: &Module<BR>, module_test: &Module<BT>)
where
    BR: ReferenceBackend,
    BT: ParityBackend,
    Module<BR>: MHEModuleAlloc<BR>
        + GLWEPublicKeyswitchMHEProtocol<BR>
        + GLWEPublicKeyMHEProtocol<BR>
        + GLWESecretSampling<BR>
        + GLWESecretPreparedFactory<BR>
        + GLWEPublicKeyPreparedFactory<BR>
        + GLWEMaskFill<BR>,
    Module<BT>: MHEModuleAlloc<BT>
        + GLWEPublicKeyswitchMHEProtocol<BT>
        + GLWESecretPreparedFactory<BT>
        + GLWEPublicKeyPreparedFactory<BT>,
{
    let layout = glwe_layout_at(module_ref, K);
    let (ct_ref, ct_test) = ciphertext(module_ref, module_test, &layout, 30);
    let (in_ref, in_test) = fixture_secrets(module_ref, module_test, RANK, 150);
    for (rank_out, k_out, key_gap) in [
        (RANK, K, BASE2K.0),
        (Rank(1), TorusPrecision(K.0 + BASE2K.0), BASE2K.0),
        (RANK, TorusPrecision(K.0 - BASE2K.0), 0),
    ] {
        let share_layout = GLWELayout {
            rank: rank_out,
            k: k_out,
            ..layout
        };
        let pk_layout = GLWELayout {
            k: TorusPrecision(k_out.0 + key_gap),
            ..share_layout
        };
        let secrets_out: Vec<_> = (0..PARTIES)
            .map(|i| secret_from_seed_at(module_ref, rank_out, [100 + i as u8; 32]).0)
            .collect();
        let pk_ref = collective_public_key(module_ref, &secrets_out, &pk_layout);
        let mut pk_test = module_test.glwe_public_key_alloc_from_infos(&pk_layout);
        pk_ref.transfer_into(&mut pk_test);
        let want = run_public(module_ref, &share_layout, &ct_ref, &in_ref, &pk_ref);
        let have = run_public(module_test, &share_layout, &ct_test, &in_test, &pk_test);
        assert_eq!(want, have, "glwe public keyswitch, {share_layout:?}");
    }
}

fn run_public<BE: ParityBackend>(
    module: &Module<BE>,
    share_layout: &GLWELayout,
    ct: &GLWE<BE::OwnedBuf, i64>,
    secrets_in: &[GLWESecret<BE::OwnedBuf, i64>],
    pk: &GLWEPublicKey<BE::OwnedBuf, i64>,
) -> Vec<Snapshot>
where
    Module<BE>: MHEModuleAlloc<BE>
        + GLWEPublicKeyswitchMHEProtocol<BE>
        + GLWESecretPreparedFactory<BE>
        + GLWEPublicKeyPreparedFactory<BE>,
{
    let secrets_in = prepare_secrets(module, secrets_in);
    let mut pk_prepared = module.glwe_public_key_prepared_alloc_from_infos(pk);
    with_scratch::<BE, _>(module.glwe_public_key_prepare_tmp_bytes(pk), |scratch| {
        module.glwe_public_key_prepare(&mut pk_prepared, pk, scratch)
    });
    let mut out = Vec::new();
    let mut acc = module.glwe_public_keyswitch_share_alloc_from_infos(share_layout);
    let mut share = module.glwe_public_keyswitch_share_alloc_from_infos(share_layout);
    for (i, sk_in) in secrets_in.iter().enumerate() {
        let dst = if i == 0 { &mut acc } else { &mut share };
        poison_glwe::<BE, _>(&mut dst.inner);
        let mut source_xu = Source::new([20 + i as u8; 32]);
        let mut source_xe = Source::new([10 + i as u8; 32]);
        let mut source_smudge = Source::new([30 + i as u8; 32]);
        with_scratch::<BE, _>(
            module.mhe_glwe_public_keyswitch_share_gen_tmp_bytes(ct, share_layout, pk),
            |scratch| {
                module.mhe_glwe_public_keyswitch_share_gen(
                    dst,
                    ct,
                    sk_in,
                    &pk_prepared,
                    FLOOD,
                    &mut source_xu,
                    &mut source_xe,
                    &mut source_smudge,
                    scratch,
                )
            },
        );
        out.push(snapshot_glwe::<BE, _>("share", &dst.inner));
        out.push(snapshot_source("share source_xu", &mut source_xu));
        out.push(snapshot_source("share source_xe", &mut source_xe));
        out.push(snapshot_source("share source_smudge", &mut source_smudge));
        if i > 0 {
            module.mhe_glwe_public_keyswitch_share_aggregate(&mut acc, &share);
            out.push(snapshot_glwe::<BE, _>("aggregate", &acc.inner));
        }
    }
    let mut res = module.glwe_alloc_from_infos(share_layout);
    poison_glwe::<BE, _>(&mut res);
    with_scratch::<BE, _>(module.mhe_glwe_public_keyswitch_share_finalize_tmp_bytes(), |scratch| {
        module.mhe_glwe_public_keyswitch_share_finalize(&mut res, ct, &acc, scratch)
    });
    let finalized = snapshot_glwe::<BE, _>("finalize", &res);
    finalized.assert_untagged();
    out.push(finalized);
    out.push(snapshot_glwe::<BE, _>("finalize share", &acc.inner));
    out
}
