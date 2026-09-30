//! Folding of CKKS batches into the ciphertexts a bootstrap refreshes, and back.
use super::{
    helpers::{Snapshot, fixture_ciphertext, snapshot, with_scratch},
    keys::{key_layout, prepared_automorphism_key, prepared_gglwe},
};
use crate::{
    CKKSInfos, CKKSLayout, CKKSMeta, SlotsKind,
    layouts::{CKKSCiphertextOwned, CKKSFoldKeySet, CKKSFoldKeys, CKKSModuleAlloc, RingSwitchKeys},
    oep::CKKSFoldImpl,
    reference::fold::CKKSFoldRing,
    test_suite::CKKSTestParams,
};
use poulpy_core::{
    GLWEMaskFill,
    layouts::{GGLWEPrepared, GGLWEPreparedFactory, GLWEAutomorphismKeyPreparedFactory, GLWELayout, GetAutomorphismKey},
};
use poulpy_hal::{
    api::ModuleNew,
    layouts::{Backend, Module, Standard},
};

type Outcome = (Result<(), String>, Vec<Snapshot>);

/// Conjugation keys of inputs under the bootstrap secret at its degree.
struct Conjugation<H>(H);

impl<B: Backend, H: GetAutomorphismKey<B>> CKKSFoldKeys<B, B> for Conjugation<H> {
    type SwitchingKey = GGLWEPrepared<B::OwnedBuf, B>;
    type ConjugationKeys = H;

    fn ring_switch(&self) -> Option<&RingSwitchKeys<Self::SwitchingKey>> {
        None
    }

    fn conjugation(&self) -> Option<&H> {
        Some(&self.0)
    }
}

fn layout(n: usize, base2k: usize, k: usize, log_delta: usize, slots: SlotsKind) -> CKKSLayout {
    CKKSLayout {
        glwe_layout: GLWELayout {
            n: n.into(),
            base2k: base2k.into(),
            k: k.into(),
            rank: 1usize.into(),
        },
        meta: CKKSMeta {
            log_delta,
            log_sparsity: 0,
            slots,
        },
    }
}

/// Folds `ins` and unfolds fixture refreshed ciphertexts of width `k_refreshed`
/// into outputs like `out`, each within its exact guarded scratch.
fn run_case<B, K>(
    module: &Module<B>,
    input_module: &Module<B>,
    ins: &[CKKSCiphertextOwned<B>],
    out: &CKKSLayout,
    k_refreshed: usize,
    keys: &K,
) -> Vec<Outcome>
where
    B: Backend<ZnxWord = i64> + CKKSFoldImpl,
    Module<B>: CKKSModuleAlloc<B> + GLWEMaskFill<B>,
    Standard: CKKSFoldRing<B, B>,
    K: CKKSFoldKeys<B, B>,
{
    let bytes = B::ckks_fold_tmp_bytes_impl(module, input_module, out, &ins[0], keys);
    let folded_layout = B::ckks_fold_layout_impl(module, input_module, &ins[0], keys);
    let mut folded: Vec<_> = (0..B::ckks_fold_count_impl(module, input_module, ins))
        .map(|i| fixture_ciphertext(module, &folded_layout, 150 + i as u8))
        .collect();
    let result = with_scratch::<B, _>(bytes, |scratch| {
        B::ckks_fold_impl(module, input_module, &mut folded, ins, keys, scratch)
    });
    assert!(result.is_ok(), "fold: {result:?}");
    let mut outcomes = vec![(
        result.map_err(|e| e.to_string()),
        folded.iter().map(snapshot::<B, _>).collect(),
    )];
    let refreshed: Vec<_> = folded
        .iter()
        .enumerate()
        .map(|(i, ct)| {
            let refreshed = CKKSLayout {
                glwe_layout: GLWELayout {
                    k: k_refreshed.into(),
                    ..folded_layout.glwe_layout
                },
                meta: ct.meta(),
            };
            fixture_ciphertext(module, &refreshed, 170 + i as u8)
        })
        .collect();
    let mut outs: Vec<_> = (0..ins.len())
        .map(|i| fixture_ciphertext(input_module, out, 190 + i as u8))
        .collect();
    let result = with_scratch::<B, _>(bytes, |scratch| {
        B::ckks_unfold_impl(module, input_module, &mut outs, &refreshed, ins, keys, scratch)
    });
    assert!(result.is_ok(), "unfold: {result:?}");
    outcomes.push((result.map_err(|e| e.to_string()), outs.iter().map(snapshot::<B, _>).collect()));
    outcomes
}

fn run_fold<B>(params: CKKSTestParams, module: &Module<B>) -> Vec<Outcome>
where
    B: Backend<ZnxWord = i64> + CKKSFoldImpl,
    Module<B>:
        ModuleNew<B> + CKKSModuleAlloc<B> + GLWEMaskFill<B> + GLWEAutomorphismKeyPreparedFactory<B> + GGLWEPreparedFactory<B>,
    Standard: CKKSFoldRing<B, B>,
{
    let (n, b) = (module.n(), params.base2k);
    let (log_delta, k_in, k_refreshed, k_out) = (12, 3 * b, 4 * b, 6 * b);
    let slots = [
        SlotsKind::Complex,
        SlotsKind::Real,
        SlotsKind::Real,
        SlotsKind::Real,
        SlotsKind::Real,
    ];
    let fixtures = |module: &Module<B>| -> Vec<_> {
        slots
            .iter()
            .zip([111u8, 113, 127, 131, 137])
            .map(|(&slots, seed)| fixture_ciphertext(module, &layout(module.n(), b, k_in, log_delta, slots), seed))
            .collect()
    };
    // Inputs under the bootstrap secret at its degree.
    let conjugation = Conjugation(prepared_automorphism_key(module, &key_layout(n, b, k_out, 2, 1, 1), -1, 61));
    let mut outcomes = run_case(
        module,
        module,
        &fixtures(module),
        &layout(n, b, k_out, log_delta, SlotsKind::Complex),
        k_refreshed,
        &conjugation,
    );
    // Inputs of half the degree under their own secret, merged two per bootstrap.
    let half = Module::<B>::new((n / 2) as u64);
    let ring_switch = RingSwitchKeys {
        inbound: prepared_gglwe(module, &key_layout(n, b, k_in, 2, 1, 1), 107),
        outbound: prepared_gglwe(module, &key_layout(n, b, k_refreshed, 1, 1, 1), 109),
    };
    let half_conjugation = prepared_automorphism_key(&half, &key_layout(n / 2, b, k_out, 2, 1, 1), -1, 63);
    let keys = CKKSFoldKeySet {
        ring_switch: &ring_switch,
        conjugation: &half_conjugation,
    };
    outcomes.extend(run_case(
        module,
        &half,
        &fixtures(&half),
        &layout(n / 2, b, k_out, log_delta, SlotsKind::Complex),
        k_refreshed,
        &keys,
    ));
    outcomes
}

/// Compare selected fold implementations on the same fixture inputs and keys:
/// a complex input and real pairs under the bootstrap secret at its degree, and
/// merged from half the degree under their own secret.
pub fn test_fold_parity<BR, BT, F>(params: CKKSTestParams, reference: &Module<BR>, tested: &Module<BT>)
where
    BR: Backend<ZnxWord = i64> + CKKSFoldImpl,
    BT: Backend<ZnxWord = i64> + CKKSFoldImpl,
    Module<BR>: ModuleNew<BR>
        + CKKSModuleAlloc<BR>
        + GLWEMaskFill<BR>
        + GLWEAutomorphismKeyPreparedFactory<BR>
        + GGLWEPreparedFactory<BR>,
    Module<BT>: ModuleNew<BT>
        + CKKSModuleAlloc<BT>
        + GLWEMaskFill<BT>
        + GLWEAutomorphismKeyPreparedFactory<BT>
        + GGLWEPreparedFactory<BT>,
    Standard: CKKSFoldRing<BR, BR> + CKKSFoldRing<BT, BT>,
{
    assert_eq!(reference.n(), tested.n());
    assert_eq!(run_fold(params, reference), run_fold(params, tested), "fold differs");
}
