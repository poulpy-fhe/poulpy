//! Folding of CKKS batches into the ciphertexts a bootstrap refreshes, and back.
use std::{cell::Cell, collections::HashMap};

use super::{
    helpers::{Snapshot, fixture_ciphertext, snapshot, with_scratch},
    keys::{key_layout, prepared_automorphism_key, prepared_gglwe},
};
use crate::{
    CKKSInfos, CKKSLayout, CKKSMeta, SetCKKSInfos, SlotsKind,
    layouts::{CKKSCiphertextOwned, CKKSFoldKeysLayout, CKKSModuleAlloc, RingSwitchKeys},
    oep::{CKKSFoldImpl, CKKSFoldLayoutImpl},
    test_suite::CKKSTestParams,
};
use poulpy_core::{
    GLWEMaskFill,
    layouts::{
        GGLWEInfos, GGLWEPrepared, GGLWEPreparedFactory, GLWEAutomorphismKeyPrepared, GLWEAutomorphismKeyPreparedFactory,
        GLWELayout, GetAutomorphismKey, LWEInfos, TorusPrecision, prepared::GLWEAutomorphismKeyPreparedBackendRef,
    },
};
use poulpy_hal::layouts::{Backend, Module, Standard};

type Outcome = [Vec<Snapshot>; 2];
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
fn run_case<B>(
    module: &Module<B>,
    ins: &[CKKSCiphertextOwned<B>],
    out: &CKKSLayout,
    k_refreshed: usize,
    ring_switch: Option<&SwitchKeys<B>>,
    automorphisms: Option<&AutomorphismKeys<B>>,
) -> Outcome
where
    B: Backend<ZnxWord = i64, Ring = Standard> + CKKSFoldLayoutImpl + CKKSFoldImpl<Standard>,
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
    let folded_layout = CKKSLayout {
        glwe_layout: B::ckks_fold_layout_impl(module, &ins[0], degree.into(), &keys_layout),
        meta: ins[0].meta(),
    };
    let mut folded: Vec<_> = (0..B::ckks_fold_count_impl(module, ins, degree.into()))
        .map(|i| fixture_ciphertext(module, &folded_layout, 150 + i as u8))
        .collect();
    with_scratch::<B, _>(bytes, |scratch| {
        B::ckks_fold_impl(module, &mut folded, ins, ring_switch.map(|keys| &keys.inbound), scratch)
    })
    .expect("fold");
    let folded_snapshot = folded.iter().map(snapshot::<B, _>).collect();
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
        .map(|(i, ct)| fixture_ciphertext(module, &CKKSLayout { meta: ct.meta(), ..*out }, 190 + i as u8))
        .collect();
    with_scratch::<B, _>(bytes, |scratch| {
        B::ckks_unfold_impl(
            module,
            &mut outs,
            &mut refreshed,
            ring_switch.map(|keys| &keys.outbound),
            automorphisms,
            scratch,
        )
    })
    .expect("unfold");
    [folded_snapshot, outs.iter().map(snapshot::<B, _>).collect()]
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
    let conjugation = HashMap::from([(
        -1,
        prepared_automorphism_key(module, &key_layout(n, b, k_out, 2, 1, 1), -1, 61),
    )]);
    let mut sparse_layout = layout(n, b, k_in, log_delta, SlotsKind::Real);
    sparse_layout.meta.log_sparsity = 1;
    let sparse_keys = <B as CKKSFoldImpl<Standard>>::ckks_unfold_galois_elements_impl(module, &sparse_layout)
        .into_iter()
        .zip(65u8..)
        .map(|(p, seed)| {
            (
                p,
                prepared_automorphism_key(module, &key_layout(n, b, k_out, 2, 1, 1), p, seed),
            )
        })
        .collect();
    let ring_switch = RingSwitchKeys {
        inbound: prepared_gglwe(module, &key_layout(n, b, k_in, 2, 1, 1), 107),
        outbound: prepared_gglwe(module, &key_layout(n, b, k_refreshed, 1, 1, 1), 109),
    };
    let own_conjugation = HashMap::from([(
        -1,
        prepared_automorphism_key(module, &key_layout(n, b, k_out, 2, 1, 1), -1, 63),
    )]);
    // Dense and sparse inputs under the bootstrap secret, then inputs under
    // another secret at the full degree (paired) and half degree (ring-packed).
    let mut outcomes = Vec::new();
    for (degree, log_sparsity, ring_switch, automorphisms) in [
        (n, 0, None, Some(&conjugation)),
        (n, 1, None, Some(&sparse_keys)),
        (n, 0, Some(&ring_switch), Some(&own_conjugation)),
        (n / 2, 0, Some(&ring_switch), None),
    ] {
        let seeds = if log_sparsity == 0 {
            [111u8, 113, 127, 131, 137]
        } else {
            [139u8, 149, 151, 157, 163]
        };
        let ins: Vec<_> = slots
            .iter()
            .zip(seeds)
            .map(|(&slots, seed)| {
                let mut input = layout(degree, b, k_in, log_delta, slots);
                input.meta.log_sparsity = log_sparsity;
                fixture_ciphertext(module, &input, seed)
            })
            .collect();
        let mut out = layout(degree, b, k_out, log_delta, SlotsKind::Complex);
        out.meta.log_sparsity = log_sparsity;
        outcomes.push(run_case(module, &ins, &out, k_refreshed, ring_switch, automorphisms));
    }
    outcomes
}

/// A transparent ciphertext with three exactly representable dyadic terms. The
/// zero mask keeps switching noise out, while the low terms detect limb copying
/// across radices. Real inputs use a constant polynomial, complex ones also X.
fn transparent<B: Backend<ZnxWord = i64, Ring = Standard>>(
    module: &Module<B>,
    ct_layout: &CKKSLayout,
    value: i64,
) -> CKKSCiphertextOwned<B> {
    let mut ct = module.ckks_ciphertext_alloc_from_infos(ct_layout);
    let (n, b) = (ct.n().as_usize(), ct.base2k().as_usize());
    let mut digits = vec![0; 2 * n * ct.max_size()];
    for (bits, numerator) in [(8usize, value), (32, 1), (52, 1)] {
        let limb = bits.div_ceil(b) - 1;
        digits[2 * limb * n] += numerator << ((limb + 1) * b - bits);
    }
    if ct.slots() == SlotsKind::Complex {
        digits[1] = 1 << (b - 8);
    }
    B::copy_from_host(ct.inner.data_mut().data_mut(), bytemuck::cast_slice(&digits));
    ct.inner.set_canonical(true);
    ct
}

/// Key providers may resolve lazily: the preflight must retain its answer.
struct Once<'a, B: Backend>(&'a AutomorphismKeys<B>, Cell<bool>);

impl<B: Backend> GetAutomorphismKey<B> for Once<'_, B> {
    fn lookup_automorphism_key(
        &self,
        p: i64,
        k: TorusPrecision,
    ) -> poulpy_core::Result<GLWEAutomorphismKeyPreparedBackendRef<'_, B>> {
        if self.1.replace(true) {
            return Err(poulpy_core::CoreError::GGLWEKeyUse {
                op: "fold test",
                detail: "automorphism key already retrieved".into(),
            });
        }
        self.0.get_automorphism_key(p, k)
    }
}

fn check_mixed_radices<B>(module: &Module<B>)
where
    B: Backend<ZnxWord = i64, Ring = Standard> + CKKSFoldLayoutImpl + CKKSFoldImpl<Standard>,
    Module<B>: CKKSModuleAlloc<B> + GLWEMaskFill<B> + GLWEAutomorphismKeyPreparedFactory<B> + GGLWEPreparedFactory<B>,
{
    let n = module.n();
    let ring_switch = RingSwitchKeys {
        inbound: prepared_gglwe(module, &key_layout(n, 15, 60, 2, 1, 1), 201),
        outbound: prepared_gglwe(module, &key_layout(n, 17, 60, 2, 1, 1), 203),
    };
    let automorphisms = HashMap::from([(
        -1,
        prepared_automorphism_key(module, &key_layout(n, 13, 60, 2, 1, 1), -1, 205),
    )]);
    let keys_layout = CKKSFoldKeysLayout {
        ring_switch: Some(ring_switch.gglwe_layout()),
        automorphism: Some(automorphisms[&-1].gglwe_layout()),
    };
    for degree in [n, n / 2] {
        let ins: Vec<_> = [SlotsKind::Complex, SlotsKind::Real, SlotsKind::Real]
            .into_iter()
            .enumerate()
            .map(|(i, slots)| transparent(module, &layout(degree, 12, 60, 12, slots), i as i64 + 1))
            .collect();
        let mut outs: Vec<_> = ins
            .iter()
            .map(|ct| fixture_ciphertext(module, &layout(degree, 19, 80, 12, ct.slots()), 207))
            .collect();
        let folded_layout: GLWELayout = B::ckks_fold_layout_impl(module, &ins[0], n.into(), &keys_layout);
        let mut folded: Vec<_> = (0..B::ckks_fold_count_impl(module, &ins, n.into()))
            .map(|_| module.ckks_ciphertext_alloc_from_glwe_infos(&folded_layout))
            .collect();
        let bytes = B::ckks_fold_tmp_bytes_impl(module, &outs[0], &ins[0], n.into(), &keys_layout);
        let once = Once(&automorphisms, Cell::new(false));
        with_scratch::<B, _>(bytes, |scratch| {
            B::ckks_fold_impl(module, &mut folded, &ins, Some(&ring_switch.inbound), scratch).unwrap();
            B::ckks_unfold_impl(
                module,
                &mut outs,
                &mut folded,
                Some(&ring_switch.outbound),
                Some(&once),
                scratch,
            )
            .unwrap();
        });
        assert_eq!(once.1.get(), degree == n, "only real pairs require the key");
        for (i, out) in outs.iter().enumerate() {
            let paired = degree == n && i > 0;
            let mut expected = transparent(module, &layout(degree, 19, 80, 12, ins[i].slots()), i as i64 + 1);
            expected.set_k((60 - usize::from(paired)).into());
            let mut expected = snapshot::<B, _>(&expected);
            if paired {
                expected.digits.iter_mut().for_each(|digit| *digit *= 2);
            }
            let actual = snapshot::<B, _>(out);
            assert_eq!(
                actual.layout, expected.layout,
                "unfold metadata at degree {degree}, input {i}"
            );
            assert_eq!(actual.digits, expected.digits, "unfold values at degree {degree}, input {i}");
            if !paired {
                assert!(actual.canonical, "singleton unfold must preserve canonical digits");
            }
        }
    }
}

fn check_unfold_errors<B>(module: &Module<B>)
where
    B: Backend<ZnxWord = i64, Ring = Standard> + CKKSFoldLayoutImpl + CKKSFoldImpl<Standard>,
    Module<B>: CKKSModuleAlloc<B> + GLWEMaskFill<B> + GLWEAutomorphismKeyPreparedFactory<B> + GGLWEPreparedFactory<B>,
{
    let n = module.n();
    let outbound = prepared_gglwe(module, &key_layout(n, 17, 60, 2, 1, 1), 211);
    let empty = AutomorphismKeys::<B>::new();
    let short = HashMap::from([(
        -1,
        prepared_automorphism_key(module, &key_layout(n, 13, 20, 1, 1, 1), -1, 213),
    )]);
    let conjugation = HashMap::from([(
        -1,
        prepared_automorphism_key(module, &key_layout(n, 13, 80, 2, 1, 1), -1, 215),
    )]);
    // A missing/short conjugation key would previously be discovered after the
    // singleton output and folded ciphertexts had already changed. Sparse keys
    // are needed before splitting too, and capacity must cover the refreshed k.
    let mixed = &[SlotsKind::Complex, SlotsKind::Real, SlotsKind::Real][..];
    let singleton = &[SlotsKind::Complex][..];
    for (name, slots, k_alloc, k_out, log_sparsity, keys) in [
        ("no keys", mixed, 80, 80usize, 0, None),
        ("missing conjugation key", mixed, 80, 80, 0, Some(&empty)),
        ("short conjugation key", mixed, 80, 80, 0, Some(&short)),
        ("missing sparse key", mixed, 80, 80, 1, Some(&conjugation)),
        ("short allocation", singleton, 20, 20, 0, None),
        ("short output width", singleton, 80, 20, 0, None),
    ] {
        let mut outs: Vec<_> = slots
            .iter()
            .map(|&slots| {
                let mut ct = fixture_ciphertext(module, &layout(n, 19, k_alloc, 12, slots), 217);
                ct.set_k(k_out.into());
                ct.set_log_sparsity(log_sparsity);
                ct
            })
            .collect();
        let mut folded: Vec<_> = (0..B::ckks_fold_count_impl(module, &outs, n.into()))
            .map(|_| fixture_ciphertext(module, &layout(n, 15, 60, 12, SlotsKind::Complex), 219))
            .collect();
        let before_out: Vec<_> = outs.iter().map(snapshot::<B, _>).collect();
        let before_folded: Vec<_> = folded.iter().map(snapshot::<B, _>).collect();
        let keys_layout = CKKSFoldKeysLayout {
            ring_switch: Some(RingSwitchKeys {
                inbound: outbound.gglwe_layout(),
                outbound: outbound.gglwe_layout(),
            }),
            automorphism: keys.and_then(|keys| keys.values().next()).map(GGLWEInfos::gglwe_layout),
        };
        let bytes = B::ckks_fold_tmp_bytes_impl(module, &outs[0], &folded[0], n.into(), &keys_layout);
        let result = with_scratch::<B, _>(bytes, |scratch| {
            B::ckks_unfold_impl(module, &mut outs, &mut folded, Some(&outbound), keys, scratch)
        });
        assert!(result.is_err(), "{name}: invalid unfold should fail");
        assert_eq!(
            before_out,
            outs.iter().map(snapshot::<B, _>).collect::<Vec<_>>(),
            "{name}: failed unfold changed outputs"
        );
        assert_eq!(
            before_folded,
            folded.iter().map(snapshot::<B, _>).collect::<Vec<_>>(),
            "{name}: failed unfold changed folded ciphertexts"
        );
    }
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
    check_mixed_radices(reference);
    check_mixed_radices(tested);
    check_unfold_errors(reference);
    check_unfold_errors(tested);
    assert_eq!(run_fold(params, reference), run_fold(params, tested), "fold differs");
}
