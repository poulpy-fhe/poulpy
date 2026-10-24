use poulpy_core::{
    api::{GLWEPublicKeyPreparedFactory, GLWESecretPreparedFactory, GLWESecretSampling},
    layouts::{Dnum, Dsize, GLWEPublicKey, GLWESecret, GLWETensorKeyLayout, LWEInfos, ModuleCoreAlloc, TorusPrecision},
};
use poulpy_hal::{layouts::Module, source::Source};

use super::{
    ParityBackend, ReferenceBackend, Snapshot, fixture_secrets, poison_gglwe, public_key::collective_public_key, snapshot_gglwe,
    snapshot_source, transfer_public_key, with_scratch,
};
use crate::{
    api::{GLWEPublicKeyMHEProtocol, GLWETensorKeyMHEProtocol},
    layouts::MHEModuleAlloc,
    test_suite::{
        fixtures::{BASE2K, DNUM, DSIZE, RANK, glwe_layout_at},
        tensor_key::tensor_key_layout,
    },
};

/// Collective tensor key under a collective public key one limb more precise,
/// at one and two limbs per digit: every share, aggregate and the finalized
/// key.
pub fn test_glwe_tensor_key_parity<BR, BT>(module_ref: &Module<BR>, module_test: &Module<BT>)
where
    BR: ReferenceBackend,
    BT: ParityBackend,
    Module<BR>: MHEModuleAlloc<BR>
        + GLWETensorKeyMHEProtocol<BR>
        + GLWEPublicKeyMHEProtocol<BR>
        + GLWESecretSampling<BR>
        + GLWESecretPreparedFactory<BR>
        + GLWEPublicKeyPreparedFactory<BR>,
    Module<BT>: MHEModuleAlloc<BT> + GLWETensorKeyMHEProtocol<BT> + GLWEPublicKeyPreparedFactory<BT>,
{
    for (dnum, dsize) in [(DNUM, DSIZE), (Dnum(2), Dsize(2))] {
        let layout = tensor_key_layout(module_ref, dnum, dsize);
        let pk_layout = glwe_layout_at(module_ref, TorusPrecision(layout.k().0 + BASE2K.0));
        let (secrets_ref, secrets_test) = fixture_secrets(module_ref, module_test, RANK, 100);
        let pk_ref = collective_public_key(module_ref, &secrets_ref, &pk_layout);
        let mut pk_test = module_test.glwe_public_key_alloc_from_infos(&pk_layout);
        transfer_public_key::<BT>(&pk_ref, &mut pk_test);
        let want = run(module_ref, &layout, &secrets_ref, &pk_ref);
        let have = run(module_test, &layout, &secrets_test, &pk_test);
        assert_eq!(want, have, "glwe tensor key, {layout:?}");
    }
}

fn run<BE: ParityBackend>(
    module: &Module<BE>,
    layout: &GLWETensorKeyLayout,
    secrets: &[GLWESecret<BE::OwnedBuf, i64>],
    pk: &GLWEPublicKey<BE::OwnedBuf, i64>,
) -> Vec<Snapshot>
where
    Module<BE>: MHEModuleAlloc<BE> + GLWETensorKeyMHEProtocol<BE> + GLWEPublicKeyPreparedFactory<BE>,
{
    let mut pk_prepared = module.glwe_public_key_prepared_alloc_from_infos(pk);
    with_scratch::<BE, _>(module.glwe_public_key_prepare_tmp_bytes(pk), |scratch| {
        module.glwe_public_key_prepare(&mut pk_prepared, pk, scratch)
    });
    let mut out = Vec::new();
    let mut acc = module.glwe_tensor_key_share_alloc_from_infos(layout);
    let mut share = module.glwe_tensor_key_share_alloc_from_infos(layout);
    for (i, sk) in secrets.iter().enumerate() {
        let dst = if i == 0 { &mut acc } else { &mut share };
        poison_gglwe::<BE, _>(&mut dst.key);
        let mut source_xu = Source::new([20 + i as u8; 32]);
        let mut source_xe = Source::new([10 + i as u8; 32]);
        with_scratch::<BE, _>(module.mhe_glwe_tensor_key_share_gen_tmp_bytes(layout, pk), |scratch| {
            module.mhe_glwe_tensor_key_share_gen(dst, sk, &pk_prepared, &mut source_xu, &mut source_xe, scratch)
        });
        out.push(snapshot_gglwe::<BE, _>("share", &dst.key));
        out.push(snapshot_source("share source_xu", &mut source_xu));
        out.push(snapshot_source("share source_xe", &mut source_xe));
        if i > 0 {
            module.mhe_glwe_tensor_key_share_aggregate(&mut acc, &share);
            out.push(snapshot_gglwe::<BE, _>("aggregate", &acc.key));
        }
    }
    let mut res = module.glwe_tensor_key_alloc_from_infos(layout);
    poison_gglwe::<BE, _>(&mut res);
    with_scratch::<BE, _>(module.mhe_glwe_tensor_key_share_finalize_tmp_bytes(), |scratch| {
        module.mhe_glwe_tensor_key_share_finalize(&mut res, &acc, scratch)
    });
    out.push(snapshot_gglwe::<BE, _>("finalize", &res));
    out.push(snapshot_gglwe::<BE, _>("finalize share", &acc.key));
    out
}
