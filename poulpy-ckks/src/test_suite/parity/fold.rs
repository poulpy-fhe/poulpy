//! Folding of CKKS batches into the ciphertexts a bootstrap refreshes, and back.
use std::collections::HashMap;

use super::{
    helpers::{Snapshot, fixture_ciphertext, snapshot, with_scratch},
    keys::{key_layout, prepared_automorphism_key, prepared_gglwe},
};
use crate::{
    CKKSInfos, CKKSLayout, CKKSMeta, SlotsKind,
    layouts::{CKKSCiphertext, CKKSFoldKeysLayout, CKKSModuleAlloc, CKKSRingCiphertext, RingSwitchKeys},
    oep::{CKKSFoldImpl, CKKSFoldLayoutImpl},
    test_suite::CKKSTestParams,
};
use poulpy_core::{
    GLWEMaskFill,
    layouts::{
        GGLWEInfos, GGLWEPrepared, GGLWEPreparedFactory, GLWEAutomorphismKeyPrepared, GLWEAutomorphismKeyPreparedFactory,
        GLWELayout,
    },
};
use poulpy_hal::layouts::{Backend, Data, Module, Ring, Standard, ZnxWord};

type Outcome = (Result<(), String>, Vec<Snapshot>);
type SwitchKeys<B> = RingSwitchKeys<GGLWEPrepared<<B as Backend>::OwnedBuf, B>>;
type AutomorphismKeys<B> = HashMap<i64, GLWEAutomorphismKeyPrepared<<B as Backend>::OwnedBuf, B>>;

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
/// into outputs of the layout of `out` labeled like `ins`, each within its exact
/// guarded scratch.
fn run_case<B, R>(
    module: &Module<B>,
    ins: &[CKKSRingCiphertext<B, R>],
    out: &CKKSLayout,
    k_refreshed: usize,
    ring_switch: Option<&SwitchKeys<B>>,
    automorphisms: Option<&AutomorphismKeys<B>>,
) -> Vec<Outcome>
where
    B: Backend<ZnxWord = i64> + CKKSFoldLayoutImpl + CKKSFoldImpl<R>,
    R: Ring,
    Module<B>: CKKSModuleAlloc<B> + GLWEMaskFill<B>,
{
    let degree = module.n();
    let keys_layout = CKKSFoldKeysLayout {
        ring_switch: ring_switch.map(RingSwitchKeys::gglwe_layout),
        automorphism: automorphisms
            .and_then(|keys| keys.values().next())
            .map(|key| key.gglwe_layout()),
    };
    let bytes = B::ckks_fold_tmp_bytes_impl(module, out, &ins[0], degree.into(), &keys_layout);
    let folded_layout = B::ckks_fold_layout_impl(module, &ins[0], degree.into(), &keys_layout);
    let mut folded: Vec<_> = (0..B::ckks_fold_count_impl(module, ins, degree.into()))
        .map(|i| fixture_ciphertext(module, &folded_layout, 150 + i as u8))
        .collect();
    let result = with_scratch::<B, _>(bytes, |scratch| {
        B::ckks_fold_impl(module, &mut folded, ins, ring_switch.map(|keys| &keys.inbound), scratch)
    });
    assert!(result.is_ok(), "fold: {result:?}");
    let mut outcomes = vec![(
        result.map_err(|e| e.to_string()),
        folded.iter().map(snapshot::<B, _>).collect(),
    )];
    let mut refreshed: Vec<_> = folded
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
    let mut outs: Vec<_> = ins
        .iter()
        .enumerate()
        .map(|(i, ct)| CKKSCiphertext::from_inner(fixture_ciphertext(module, out, 190 + i as u8).inner, ct.meta()))
        .collect();
    let result = with_scratch::<B, _>(bytes, |scratch| {
        B::ckks_unfold_impl(
            module,
            &mut outs,
            &mut refreshed,
            ring_switch.map(|keys| &keys.outbound),
            automorphisms,
            scratch,
        )
    });
    assert!(result.is_ok(), "unfold: {result:?}");
    outcomes.push((
        result.map_err(|e| e.to_string()),
        outs.into_iter()
            .map(|ct| snapshot::<B, _>(&relabel::<_, _, R, Standard>(ct)))
            .collect(),
    ));
    outcomes
}

/// Views a ciphertext as one of ring `R`: any coefficients are valid.
fn relabel<D: Data, W: ZnxWord, S: Ring, R: Ring>(ct: CKKSCiphertext<D, W, S>) -> CKKSCiphertext<D, W, R> {
    let meta = ct.meta();
    CKKSCiphertext::from_inner(ct.inner, meta)
}

fn run_fold<B>(params: CKKSTestParams, module: &Module<B>) -> Vec<Outcome>
where
    B: Backend<ZnxWord = i64, Ring = Standard> + CKKSFoldLayoutImpl + CKKSFoldImpl<Standard>,
    Module<B>: CKKSModuleAlloc<B> + GLWEMaskFill<B> + GLWEAutomorphismKeyPreparedFactory<B> + GGLWEPreparedFactory<B>,
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
    let fixtures = |degree: usize| -> Vec<_> {
        slots
            .iter()
            .zip([111u8, 113, 127, 131, 137])
            .map(|(&slots, seed)| fixture_ciphertext(module, &layout(degree, b, k_in, log_delta, slots), seed))
            .collect()
    };
    // Inputs under the bootstrap secret at its degree.
    let conjugation = HashMap::from([(
        -1,
        prepared_automorphism_key(module, &key_layout(n, b, k_out, 2, 1, 1), -1, 61),
    )]);
    let mut outcomes = run_case::<B, Standard>(
        module,
        &fixtures(n),
        &layout(n, b, k_out, log_delta, SlotsKind::Complex),
        k_refreshed,
        None,
        Some(&conjugation),
    );
    // The same inputs, sparse: two share each coefficient position of a bootstrap.
    let sparse = |layout: CKKSLayout| CKKSLayout {
        meta: CKKSMeta {
            log_sparsity: 1,
            ..layout.meta
        },
        ..layout
    };
    let sparse_ins: Vec<_> = slots
        .iter()
        .zip([139u8, 149, 151, 157, 163])
        .map(|(&slots, seed)| fixture_ciphertext(module, &sparse(layout(n, b, k_in, log_delta, slots)), seed))
        .collect();
    // Keys for the real inputs, which pair and split with the conjugation key.
    let automorphisms: HashMap<i64, _> = <B as CKKSFoldImpl<Standard>>::ckks_unfold_galois_elements_impl(module, &sparse_ins[1])
        .into_iter()
        .zip(65u8..)
        .map(|(p, seed)| {
            (
                p,
                prepared_automorphism_key(module, &key_layout(n, b, k_out, 2, 1, 1), p, seed),
            )
        })
        .collect();
    outcomes.extend(run_case::<B, Standard>(
        module,
        &sparse_ins,
        &sparse(layout(n, b, k_out, log_delta, SlotsKind::Complex)),
        k_refreshed,
        None,
        Some(&automorphisms),
    ));
    // The same inputs under their own secret, paired at the bootstrap degree, then
    // of half the degree, unpaired and merged two per bootstrap.
    let ring_switch = RingSwitchKeys {
        inbound: prepared_gglwe(module, &key_layout(n, b, k_in, 2, 1, 1), 107),
        outbound: prepared_gglwe(module, &key_layout(n, b, k_refreshed, 1, 1, 1), 109),
    };
    let own_conjugation = HashMap::from([(
        -1,
        prepared_automorphism_key(module, &key_layout(n, b, k_out, 2, 1, 1), -1, 63),
    )]);
    outcomes.extend(run_case::<B, Standard>(
        module,
        &fixtures(n),
        &layout(n, b, k_out, log_delta, SlotsKind::Complex),
        k_refreshed,
        Some(&ring_switch),
        Some(&own_conjugation),
    ));
    outcomes.extend(run_case::<B, Standard>(
        module,
        &fixtures(n / 2),
        &layout(n / 2, b, k_out, log_delta, SlotsKind::Complex),
        k_refreshed,
        Some(&ring_switch),
        None,
    ));
    outcomes
}

/// Compare selected fold implementations on the same fixture inputs and keys:
/// a complex input and real pairs under the bootstrap secret at its degree, and
/// the same inputs under their own secret at that degree and merged from half of it.
pub fn test_fold_parity<BR, BT, F>(params: CKKSTestParams, reference: &Module<BR>, tested: &Module<BT>)
where
    BR: Backend<ZnxWord = i64, Ring = Standard> + CKKSFoldLayoutImpl + CKKSFoldImpl<Standard>,
    BT: Backend<ZnxWord = i64, Ring = Standard> + CKKSFoldLayoutImpl + CKKSFoldImpl<Standard>,
    Module<BR>: CKKSModuleAlloc<BR> + GLWEMaskFill<BR> + GLWEAutomorphismKeyPreparedFactory<BR> + GGLWEPreparedFactory<BR>,
    Module<BT>: CKKSModuleAlloc<BT> + GLWEMaskFill<BT> + GLWEAutomorphismKeyPreparedFactory<BT> + GGLWEPreparedFactory<BT>,
{
    assert_eq!(reference.n(), tested.n());
    assert_eq!(run_fold(params, reference), run_fold(params, tested), "fold differs");
}
