use poulpy_core::{
    api::{GLWESecretPreparedFactory, GLWESecretSampling},
    layouts::{
        GGLWEInfos, GGLWELayout, GLWEAutomorphismKeyAlloc, GLWESecret, GLWESwitchingKeyAlloc, GLWESwitchingKeyDegrees, Rank,
        TorusPrecision,
    },
    test_suite::with_scratch,
};
use poulpy_hal::{layouts::Module, source::Source};

use super::{
    Meta, ParityBackend, ReferenceBackend, Snapshot, fixture_secrets, poison_gglwe, poison_gglwe_compressed, snapshot_gglwe,
    snapshot_gglwe_compressed, snapshot_source,
};
use crate::{
    MHEGLWESwitchingKeyShareAlloc,
    api::{MHEGLWEAutomorphismKeyProtocol, MHEGLWESwitchingKeyProtocol},
    layouts::MHEGLWEAutomorphismKeyShareAlloc,
    test_suite::fixtures::{BASE2K, P, RANK, SEEDS, gglwe_layout},
};

/// The fixture layout, digits one bit past a limb, and for a switching key a
/// rank-1 input.
fn key_layouts<BE: poulpy_hal::layouts::Backend>(module: &Module<BE>, rank_change: bool) -> Vec<GGLWELayout> {
    let layout = gglwe_layout(module);
    let k_digit = BASE2K.0 + 1;
    let mut layouts = vec![
        layout,
        GGLWELayout {
            k_digit: TorusPrecision(k_digit),
            k_aux: TorusPrecision(k_digit + module.log_n() as u32),
            ..layout
        },
    ];
    if rank_change {
        layouts.push(GGLWELayout {
            rank_in: Rank(1),
            ..layout
        });
    }
    layouts
}

/// Collective switching key: every share, aggregate and the finalized key.
pub fn test_glwe_switching_key_parity<BR, BT>(module_ref: &Module<BR>, module_test: &Module<BT>)
where
    BR: ReferenceBackend,
    BT: ParityBackend,
    Module<BR>: MHEGLWESwitchingKeyProtocol<BR> + GLWESecretSampling<BR> + GLWESecretPreparedFactory<BR>,
    Module<BT>: MHEGLWESwitchingKeyProtocol<BT>,
{
    for layout in key_layouts(module_ref, true) {
        let (in_ref, in_test) = fixture_secrets(module_ref, module_test, layout.rank_in(), 150);
        let (out_ref, out_test) = fixture_secrets(module_ref, module_test, layout.rank_out(), 100);
        let want = run_switching_key(module_ref, &layout, &in_ref, &out_ref);
        let have = run_switching_key(module_test, &layout, &in_test, &out_test);
        assert_eq!(want, have, "glwe switching key, {layout:?}");
    }
}

fn run_switching_key<BE: ParityBackend>(
    module: &Module<BE>,
    layout: &GGLWELayout,
    secrets_in: &[GLWESecret<BE::OwnedBuf, i64>],
    secrets_out: &[GLWESecret<BE::OwnedBuf, i64>],
) -> Vec<Snapshot>
where
    Module<BE>: MHEGLWESwitchingKeyShareAlloc<BE> + MHEGLWESwitchingKeyProtocol<BE>,
{
    let mut out = Vec::new();
    let mut acc = module.mhe_glwe_switching_key_share_alloc_from_infos(layout);
    let mut share = module.mhe_glwe_switching_key_share_alloc_from_infos(layout);
    for (i, (sk_in, sk_out)) in secrets_in.iter().zip(secrets_out).enumerate() {
        let dst = if i == 0 { &mut acc } else { &mut share };
        poison_gglwe_compressed::<BE, _>(&mut dst.key);
        let mut source_xe = Source::new([10 + i as u8; 32]);
        with_scratch::<BE, _>(module.mhe_glwe_switching_key_share_gen_tmp_bytes(layout), |scratch| {
            module.mhe_glwe_switching_key_share_gen(dst, sk_in, sk_out, SEEDS[0], &mut source_xe, scratch)
        });
        out.push(snapshot_gglwe_compressed::<BE, _>("share", &dst.key).with(Meta::Degrees(dst.input_degree, dst.output_degree)));
        out.push(snapshot_source("share source_xe", &mut source_xe));
        if i > 0 {
            module.mhe_glwe_switching_key_share_aggregate(&mut acc, &share);
            out.push(
                snapshot_gglwe_compressed::<BE, _>("aggregate", &acc.key)
                    .with(Meta::Degrees(acc.input_degree, acc.output_degree)),
            );
        }
    }
    let mut res = module.glwe_switching_key_alloc_from_infos(layout);
    poison_gglwe::<BE, _>(&mut res);
    with_scratch::<BE, _>(module.mhe_glwe_switching_key_share_finalize_tmp_bytes(), |scratch| {
        module.mhe_glwe_switching_key_share_finalize(&mut res, &acc, scratch)
    });
    out.push(snapshot_gglwe::<BE, _>("finalize", &res).with(Meta::Degrees(*res.input_degree(), *res.output_degree())));
    out.push(snapshot_gglwe_compressed::<BE, _>("finalize share", &acc.key));
    out
}

/// Collective automorphism key: every share, aggregate and the finalized key.
pub fn test_glwe_automorphism_key_parity<BR, BT>(module_ref: &Module<BR>, module_test: &Module<BT>)
where
    BR: ReferenceBackend,
    BT: ParityBackend,
    Module<BR>: MHEGLWEAutomorphismKeyProtocol<BR> + GLWESecretSampling<BR> + GLWESecretPreparedFactory<BR>,
    Module<BT>: MHEGLWEAutomorphismKeyProtocol<BT>,
{
    for layout in key_layouts(module_ref, false) {
        let (secrets_ref, secrets_test) = fixture_secrets(module_ref, module_test, RANK, 100);
        let want = run_automorphism_key(module_ref, &layout, &secrets_ref);
        let have = run_automorphism_key(module_test, &layout, &secrets_test);
        assert_eq!(want, have, "glwe automorphism key, {layout:?}");
    }
}

fn run_automorphism_key<BE: ParityBackend>(
    module: &Module<BE>,
    layout: &GGLWELayout,
    secrets: &[GLWESecret<BE::OwnedBuf, i64>],
) -> Vec<Snapshot>
where
    Module<BE>: MHEGLWEAutomorphismKeyShareAlloc<BE> + MHEGLWEAutomorphismKeyProtocol<BE>,
{
    let mut out = Vec::new();
    let mut acc = module.mhe_glwe_automorphism_key_share_alloc_from_infos(layout);
    let mut share = module.mhe_glwe_automorphism_key_share_alloc_from_infos(layout);
    for (i, sk) in secrets.iter().enumerate() {
        let dst = if i == 0 { &mut acc } else { &mut share };
        poison_gglwe_compressed::<BE, _>(&mut dst.key);
        let mut source_xe = Source::new([10 + i as u8; 32]);
        with_scratch::<BE, _>(module.mhe_glwe_automorphism_key_share_gen_tmp_bytes(layout), |scratch| {
            module.mhe_glwe_automorphism_key_share_gen(dst, P, sk, SEEDS[0], &mut source_xe, scratch)
        });
        out.push(snapshot_gglwe_compressed::<BE, _>("share", &dst.key).with(Meta::GaloisElement(dst.p)));
        out.push(snapshot_source("share source_xe", &mut source_xe));
        if i > 0 {
            module.mhe_glwe_automorphism_key_share_aggregate(&mut acc, &share);
            out.push(snapshot_gglwe_compressed::<BE, _>("aggregate", &acc.key).with(Meta::GaloisElement(acc.p)));
        }
    }
    let mut res = module.glwe_automorphism_key_alloc_from_infos(layout);
    poison_gglwe::<BE, _>(&mut res);
    with_scratch::<BE, _>(module.mhe_glwe_automorphism_key_share_finalize_tmp_bytes(), |scratch| {
        module.mhe_glwe_automorphism_key_share_finalize(&mut res, &acc, scratch)
    });
    out.push(snapshot_gglwe::<BE, _>("finalize", &res).with(Meta::GaloisElement(res.p())));
    out.push(snapshot_gglwe_compressed::<BE, _>("finalize share", &acc.key));
    out
}
