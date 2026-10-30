use poulpy_core::{
    TransferInto,
    api::{GLWESecretPreparedFactory, GLWESecretSampling, GLWESwitchingKeyPreparedFactory},
    layouts::{
        Dnum, GGLWELayout, GGSWAlloc, GGSWLayout, GLWESecret, GLWESwitchingKey, GLWESwitchingKeyAlloc, Rank, TorusPrecision,
    },
    test_suite::with_scratch,
};
use poulpy_hal::{
    AlignedBuf,
    api::ScalarZnxAlloc,
    layouts::{Backend, DataView, DataViewMut, Module, ScalarZnx, ZnxViewMut, transfer_buf_into},
    source::Source,
};

use super::{
    ParityBackend, ReferenceBackend, Snapshot, fixture_secrets, poison_gglwe_compressed, poison_ggsw, prepare_secrets,
    snapshot_gglwe_compressed, snapshot_ggsw, snapshot_source,
};
use crate::{
    api::{GGSWMHEProtocol, GLWESwitchingKeyMHEProtocol},
    layouts::{GGSWShareOwned, MHEModuleAlloc},
    test_suite::{
        fixtures::{BASE2K, PARTIES, SEEDS, bounded_integers, ggsw_layout},
        ggsw::ephemeral_key_layout,
    },
};

/// Collective GGSW at the fixture layout, at rank 1 with two-limb digits and
/// with digits one bit past a limb, finalized under an ephemeral key built on
/// the comparison backend: every share, aggregate and the finalized GGSW.
pub fn test_ggsw_parity<BR, BT>(module_ref: &Module<BR>, module_test: &Module<BT>)
where
    BR: ReferenceBackend,
    BT: ParityBackend,
    Module<BR>: MHEModuleAlloc<BR>
        + GGSWMHEProtocol<BR>
        + GLWESwitchingKeyMHEProtocol<BR>
        + GLWESecretSampling<BR>
        + GLWESecretPreparedFactory<BR>
        + GLWESwitchingKeyPreparedFactory<BR>
        + ScalarZnxAlloc<BR>,
    Module<BT>: MHEModuleAlloc<BT>
        + GGSWMHEProtocol<BT>
        + GLWESecretPreparedFactory<BT>
        + GLWESwitchingKeyPreparedFactory<BT>
        + ScalarZnxAlloc<BT>,
{
    let base = ggsw_layout(module_ref);
    let k_digit = BASE2K.0 + 1;
    let layouts = [
        base,
        GGSWLayout {
            dnum: Dnum(2),
            k_digit: TorusPrecision(2 * BASE2K.0),
            k_aux: TorusPrecision(2 * BASE2K.0 + module_ref.log_n() as u32),
            rank: Rank(1),
            ..base
        },
        GGSWLayout {
            dnum: Dnum((base.dnum.as_u32() * base.k_digit.as_u32()).div_ceil(k_digit)),
            k_digit: TorusPrecision(k_digit),
            k_aux: TorusPrecision(k_digit + module_ref.log_n() as u32),
            ..base
        },
    ];
    let n = module_ref.n();
    for layout in layouts {
        let key_layout = ephemeral_key_layout(module_ref, &layout);
        let (secrets_ref, secrets_test) = fixture_secrets(module_ref, module_test, layout.rank, 100);
        let (ephemerals_ref, ephemerals_test) = fixture_secrets(module_ref, module_test, layout.rank, 170);
        let (messages_ref, messages_test): (Vec<_>, Vec<_>) = (0..PARTIES)
            .map(|i| {
                let mut m: ScalarZnx<AlignedBuf, i64> = module_ref.scalar_znx_alloc(n, 1);
                m.at_mut(0, 0).copy_from_slice(&bounded_integers(n, 2, [80 + i as u8; 32]));
                let mut copy = module_test.scalar_znx_alloc(n, 1);
                transfer_buf_into(m.data(), copy.data_mut());
                (m, copy)
            })
            .unzip();
        let key_ref = ephemeral_key(module_ref, &key_layout, &ephemerals_ref, &secrets_ref);
        let mut key_test = module_test.glwe_switching_key_alloc_from_infos(&key_layout);
        key_ref.transfer_into(&mut key_test);
        let want = run(
            module_ref,
            &layout,
            &key_layout,
            &secrets_ref,
            &ephemerals_ref,
            &messages_ref,
            &key_ref,
        );
        let have = run(
            module_test,
            &layout,
            &key_layout,
            &secrets_test,
            &ephemerals_test,
            &messages_test,
            &key_test,
        );
        assert_eq!(want, have, "ggsw, {layout:?}");
    }
}

/// The collective switching key from the sum of `ephemerals` to the sum of `secrets`.
fn ephemeral_key<BR>(
    module_ref: &Module<BR>,
    layout: &GGLWELayout,
    ephemerals: &[GLWESecret<AlignedBuf, i64>],
    secrets: &[GLWESecret<AlignedBuf, i64>],
) -> GLWESwitchingKey<AlignedBuf, i64>
where
    BR: ReferenceBackend,
    Module<BR>: MHEModuleAlloc<BR> + GLWESwitchingKeyMHEProtocol<BR>,
{
    let mut acc = module_ref.glwe_switching_key_share_alloc_from_infos(layout);
    let mut share = module_ref.glwe_switching_key_share_alloc_from_infos(layout);
    for (i, (u, sk)) in ephemerals.iter().zip(secrets).enumerate() {
        let dst = if i == 0 { &mut acc } else { &mut share };
        let mut source_xe = Source::new([60 + i as u8; 32]);
        with_scratch::<BR, _>(module_ref.mhe_glwe_switching_key_share_gen_tmp_bytes(layout), |scratch| {
            module_ref.mhe_glwe_switching_key_share_gen(dst, u, sk, SEEDS[1], &mut source_xe, scratch)
        });
        if i > 0 {
            module_ref.mhe_glwe_switching_key_share_aggregate(&mut acc, &share);
        }
    }
    let mut key = module_ref.glwe_switching_key_alloc_from_infos(layout);
    with_scratch::<BR, _>(module_ref.mhe_glwe_switching_key_share_finalize_tmp_bytes(), |scratch| {
        module_ref.mhe_glwe_switching_key_share_finalize(&mut key, &acc, scratch)
    });
    key
}

fn run<BE: ParityBackend>(
    module: &Module<BE>,
    layout: &GGSWLayout,
    key_layout: &GGLWELayout,
    secrets: &[GLWESecret<BE::OwnedBuf, i64>],
    ephemerals: &[GLWESecret<BE::OwnedBuf, i64>],
    messages: &[ScalarZnx<BE::OwnedBuf, i64>],
    key: &GLWESwitchingKey<BE::OwnedBuf, i64>,
) -> Vec<Snapshot>
where
    Module<BE>: MHEModuleAlloc<BE> + GGSWMHEProtocol<BE> + GLWESecretPreparedFactory<BE> + GLWESwitchingKeyPreparedFactory<BE>,
{
    let secrets = prepare_secrets(module, secrets);
    let ephemerals = prepare_secrets(module, ephemerals);
    let mut key_prepared = module.glwe_switching_key_prepared_alloc_from_infos(key_layout);
    with_scratch::<BE, _>(module.glwe_switching_key_prepare_tmp_bytes(key_layout), |scratch| {
        module.glwe_switching_key_prepare(&mut key_prepared, key, scratch)
    });
    let mut out = Vec::new();
    let mut acc = module.ggsw_share_alloc_from_infos(layout);
    let mut share = module.ggsw_share_alloc_from_infos(layout);
    for (i, ((sk, u), m)) in secrets.iter().zip(&ephemerals).zip(messages).enumerate() {
        let dst = if i == 0 { &mut acc } else { &mut share };
        dst.parts_mut().for_each(poison_gglwe_compressed::<BE, _>);
        let mut source_xe = Source::new([10 + i as u8; 32]);
        with_scratch::<BE, _>(module.mhe_ggsw_share_gen_tmp_bytes(layout), |scratch| {
            module.mhe_ggsw_share_gen(dst, m, sk, u, SEEDS[0], &mut source_xe, scratch)
        });
        out.extend(snapshot_share::<BE>("share", dst));
        out.push(snapshot_source("share source_xe", &mut source_xe));
        if i > 0 {
            module.mhe_ggsw_share_aggregate(&mut acc, &share);
            out.extend(snapshot_share::<BE>("aggregate", &acc));
        }
    }
    let mut res = module.ggsw_alloc_from_infos(layout);
    poison_ggsw::<BE, _>(&mut res);
    with_scratch::<BE, _>(module.mhe_ggsw_share_finalize_tmp_bytes(layout, key_layout), |scratch| {
        module.mhe_ggsw_share_finalize(&mut res, &acc, &key_prepared, scratch)
    });
    out.push(snapshot_ggsw::<BE, _>("finalize", &res));
    out.extend(snapshot_share::<BE>("finalize share", &acc));
    out
}

fn snapshot_share<BE: Backend>(label: &'static str, share: &GGSWShareOwned<BE>) -> Vec<Snapshot> {
    share
        .parts()
        .map(|part| snapshot_gglwe_compressed::<BE, _>(label, part))
        .collect()
}
