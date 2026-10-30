use poulpy_core::{
    api::{GLWESecretPreparedFactory, GLWESecretSampling},
    layouts::{GLWELayout, GLWEPublicKey, GLWEPublicKeyAlloc, GLWEPublicKeyCompressedToBackendMut, GLWESecret, Rank},
    test_suite::with_scratch,
};
use poulpy_hal::{
    AlignedBuf,
    layouts::{Backend, DataViewMut, Module},
    source::Source,
};

use super::{
    ParityBackend, ReferenceBackend, Snapshot, fixture_secrets, poison_public_key, poison_view, prepare_secrets,
    snapshot_public_key, snapshot_public_key_share, snapshot_source,
};
use crate::{
    api::GLWEPublicKeyMHEProtocol,
    layouts::{GLWEPublicKeyShareOwned, MHEModuleAlloc},
    test_suite::fixtures::{K, RANK, SEEDS, glwe_layout_at},
};

/// Collective public key at ranks 2 and 1: every share, aggregate and the
/// finalized key.
pub fn test_glwe_public_key_parity<BR, BT>(module_ref: &Module<BR>, module_test: &Module<BT>)
where
    BR: ReferenceBackend,
    BT: ParityBackend,
    Module<BR>: MHEModuleAlloc<BR> + GLWEPublicKeyMHEProtocol<BR> + GLWESecretSampling<BR> + GLWESecretPreparedFactory<BR>,
    Module<BT>: MHEModuleAlloc<BT> + GLWEPublicKeyMHEProtocol<BT> + GLWESecretPreparedFactory<BT>,
{
    for rank in [RANK, Rank(1)] {
        let layout = GLWELayout {
            rank,
            ..glwe_layout_at(module_ref, K)
        };
        let (secrets_ref, secrets_test) = fixture_secrets(module_ref, module_test, rank, 100);
        let want = run(module_ref, &layout, &secrets_ref);
        let have = run(module_test, &layout, &secrets_test);
        assert_eq!(want, have, "glwe public key, rank {rank:?}");
    }
}

fn run<BE: ParityBackend>(module: &Module<BE>, layout: &GLWELayout, secrets: &[GLWESecret<BE::OwnedBuf, i64>]) -> Vec<Snapshot>
where
    Module<BE>: MHEModuleAlloc<BE> + GLWEPublicKeyMHEProtocol<BE> + GLWESecretPreparedFactory<BE>,
{
    let secrets = prepare_secrets(module, secrets);
    let mut out = Vec::new();
    let mut acc = module.glwe_public_key_share_alloc_from_infos(layout);
    let mut share = module.glwe_public_key_share_alloc_from_infos(layout);
    for (i, sk) in secrets.iter().enumerate() {
        let dst = if i == 0 { &mut acc } else { &mut share };
        poison_public_key_share::<BE>(dst);
        let mut source_xe = Source::new([10 + i as u8; 32]);
        with_scratch::<BE, _>(module.mhe_glwe_public_key_share_gen_tmp_bytes(layout), |scratch| {
            module.mhe_glwe_public_key_share_gen(dst, sk, SEEDS[0], &mut source_xe, scratch)
        });
        out.push(snapshot_public_key_share::<BE>("share", dst));
        out.push(snapshot_source("share source_xe", &mut source_xe));
        if i > 0 {
            module.mhe_glwe_public_key_share_aggregate(&mut acc, &share);
            out.push(snapshot_public_key_share::<BE>("aggregate", &acc));
        }
    }
    let mut pk = module.glwe_public_key_alloc_from_infos(layout);
    poison_public_key::<BE, _>(&mut pk);
    with_scratch::<BE, _>(module.mhe_glwe_public_key_share_finalize_tmp_bytes(), |scratch| {
        module.mhe_glwe_public_key_share_finalize(&mut pk, &acc, scratch)
    });
    out.push(snapshot_public_key::<BE, _>("finalize", &pk));
    out.push(snapshot_public_key_share::<BE>("finalize share", &acc));
    out
}

fn poison_public_key_share<BE: Backend>(share: &mut GLWEPublicKeyShareOwned<BE>) {
    poison_view::<BE>(
        GLWEPublicKeyCompressedToBackendMut::<BE>::to_backend_mut(share)
            .data_mut()
            .data_mut(),
    );
}

/// The collective public key of `secrets` at `layout` on `module_ref`, unprepared.
pub(crate) fn collective_public_key<BR>(
    module_ref: &Module<BR>,
    secrets: &[GLWESecret<AlignedBuf, i64>],
    layout: &GLWELayout,
) -> GLWEPublicKey<AlignedBuf, i64>
where
    BR: ReferenceBackend,
    Module<BR>: MHEModuleAlloc<BR> + GLWEPublicKeyMHEProtocol<BR> + GLWESecretPreparedFactory<BR>,
{
    let secrets = prepare_secrets(module_ref, secrets);
    let mut acc = module_ref.glwe_public_key_share_alloc_from_infos(layout);
    let mut share = module_ref.glwe_public_key_share_alloc_from_infos(layout);
    for (i, sk) in secrets.iter().enumerate() {
        let dst = if i == 0 { &mut acc } else { &mut share };
        let mut source_xe = Source::new([40 + i as u8; 32]);
        with_scratch::<BR, _>(module_ref.mhe_glwe_public_key_share_gen_tmp_bytes(layout), |scratch| {
            module_ref.mhe_glwe_public_key_share_gen(dst, sk, SEEDS[1], &mut source_xe, scratch)
        });
        if i > 0 {
            module_ref.mhe_glwe_public_key_share_aggregate(&mut acc, &share);
        }
    }
    let mut pk = module_ref.glwe_public_key_alloc_from_infos(layout);
    with_scratch::<BR, _>(module_ref.mhe_glwe_public_key_share_finalize_tmp_bytes(), |scratch| {
        module_ref.mhe_glwe_public_key_share_finalize(&mut pk, &acc, scratch)
    });
    pk
}
