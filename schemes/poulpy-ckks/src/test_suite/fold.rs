//! Fold and unfold round trip, checked on ring coefficients: the fold only moves
//! coefficients, so the folded message has a cleartext model.

use crate::{Scale, ckks_log_delta};
use std::collections::HashMap;

use poulpy_core::layouts::{
    GGLWEInfos, GGLWELayout, GGLWEPrepared, GLWEAutomorphismKeyPrepared, GLWEInfos, GLWELayout, GLWEPlaintextInfos,
    GLWESecretPrepared, GLWEToBackendMut, LWEInfos, SetGLWEPlaintextInfos, SetK,
};
use poulpy_hal::{
    api::ScratchOwnedBorrow,
    layouts::{Backend, HostBytesBackend, HostDataMut, HostDataRef, Module, ScratchOwned, Standard},
    source::Source,
};

use crate::{
    CKKSCompositionError, CKKSInfos, CKKSLayout, GLWEPlaintextMeta, SetCKKSInfos, SlotLayout, SlotsKind,
    api::{CKKSFoldLayoutOps, CKKSFoldOps},
    layouts::{CKKSCiphertextOwned, CKKSFoldKeysLayout, CKKSModuleAlloc, CKKSPlaintextVecHostCodec, RingSwitchKeys},
    test_suite::{
        CKKSTestParams,
        helpers::{
            TestContextBackend, TestContextModule, TestContextSharedModule, TestScalar, alloc_scratch, assert_ckks_error,
            assert_precision, ckks_decrypt_with_prec, ckks_encrypt_coeffs, gen_atk, gen_sk_with_raw,
        },
        parity::{
            helpers::{fixture_ciphertext, snapshot, with_scratch},
            keys::{key_layout, prepared_automorphism_key, prepared_gglwe},
        },
    },
};

type AutomorphismKeys<BE> = HashMap<i64, GLWEAutomorphismKeyPrepared<<BE as Backend>::OwnedBuf, BE>>;

/// Folds a complex input and two real pairs at the input width, decrypts and checks
/// the folded messages, re-encrypts them wider as a bootstrap would, then unfolds and
/// checks the inputs return: under the fold secret at its degree, dense and sparse,
/// and under their own secret at half of it, paired then merged two per folded ciphertext.
pub fn test_fold_unfold<BE, F, E>(params: CKKSTestParams, module: &Module<BE>, host_module: &Module<HostBytesBackend>)
where
    BE: TestContextBackend<Ring = Standard>,
    Module<BE>: TestContextModule<BE> + CKKSFoldLayoutOps<BE> + CKKSFoldOps<BE, Standard>,
    F: TestScalar,
    for<'a> BE::BufRef<'a>: HostDataRef,
    for<'a> BE::BufMut<'a>: HostDataMut,
{
    let n = module.n();
    let half_params = CKKSTestParams {
        n: n / 2,
        hw: params.hw.min(n / 2),
        ..params
    };
    let k_in = params.prec().k().as_usize();
    let k_out = 2 * k_in;
    let mut scratch = alloc_scratch(&params, module);
    let (sk_raw, sk) = gen_sk_with_raw(&params, module, host_module, [31; 32]);
    let (half_sk_raw, half_sk) = gen_sk_with_raw(&half_params, module, host_module, [32; 32]);
    let ring_switch = RingSwitchKeys {
        inbound: params.ksk_layout(k_in),
        outbound: params.ksk_layout(k_out),
    }
    .generate(
        module,
        &half_sk_raw,
        &sk_raw,
        &mut Source::new([33; 32]),
        &mut Source::new([34; 32]),
        &mut scratch.borrow(),
    )
    .unwrap()
    .prepare(module, &mut scratch.borrow())
    .unwrap();
    // Conjugation splits real pairs; two sparse elements exercise both split levels.
    let sparse_real = CKKSLayout {
        meta: GLWEPlaintextMeta {
            slots: SlotsKind::Real,
            log_sparsity: 2,
            ..params.prec_meta
        },
        ..params.prec()
    };
    let automorphisms: AutomorphismKeys<BE> = CKKSFoldOps::<_, Standard>::ckks_unfold_galois_elements(module, &sparse_real)
        .into_iter()
        .map(|p| (p, gen_atk(&params, module, p, &sk_raw, &mut scratch.borrow())))
        .collect();
    let half_automorphisms = HashMap::from([(-1, gen_atk(&half_params, module, -1, &half_sk_raw, &mut scratch.borrow()))]);
    for (input_params, input_sk, ring_switch, automorphisms, log_sparsity) in [
        (&params, &sk, None, Some(&automorphisms), 0),
        (&params, &sk, None, Some(&automorphisms), 2),
        (&half_params, &half_sk, Some(&ring_switch), Some(&half_automorphisms), 0),
    ] {
        let n_in = input_params.n;
        let log_delta = ckks_log_delta(&params.prec_meta);
        let label = format!("degree {n_in}, sparsity {log_sparsity}");
        let mut scratch = alloc_scratch(input_params, module);
        let slots = [
            SlotsKind::Complex,
            SlotsKind::Real,
            SlotsKind::Real,
            SlotsKind::Real,
            SlotsKind::Real,
        ];
        let msgs: Vec<Vec<F>> = slots
            .iter()
            .enumerate()
            .map(|(seed, &slots)| message(n_in, slots, log_sparsity, seed))
            .collect();
        let mut ins: Vec<_> = msgs
            .iter()
            .zip(slots)
            .map(|(msg, slots)| {
                let prec = CKKSLayout {
                    meta: GLWEPlaintextMeta {
                        slots,
                        log_sparsity,
                        ..input_params.prec_meta
                    },
                    ..input_params.prec()
                };
                ckks_encrypt_coeffs(
                    input_params,
                    module,
                    host_module,
                    input_sk,
                    k_in,
                    msg,
                    prec,
                    &mut scratch.borrow(),
                )
            })
            .collect();

        // Real inputs pair at either degree; a folded ciphertext holds
        // `g = n / n_in` positions, with four sparse parts in the same-ring sparse case.
        let units = [(0, None), (1, Some(2)), (3, Some(4))];
        let g = n / n_in;
        let span = g << log_sparsity;
        let degree = n.into();
        let count = module.ckks_fold_count(degree, &ins);
        assert_eq!(count, units.len().div_ceil(span), "folded count, {label}");
        let keys_layout = CKKSFoldKeysLayout {
            ring_switch: ring_switch.map(RingSwitchKeys::gglwe_layout),
            automorphism: automorphisms
                .and_then(|keys| keys.values().next())
                .map(|key| key.gglwe_layout()),
        };
        let output = CKKSLayout {
            glwe_layout: GLWELayout {
                k: k_out.into(),
                ..ins[0].glwe_layout()
            },
            meta: ins[0].meta(),
        };
        let folded_layout = module.ckks_fold_layout(degree, &ins[0], &keys_layout);
        let fold_bytes = module.ckks_fold_tmp_bytes(degree, &output, &ins[0], &keys_layout);
        let mut folded: Vec<_> = (0..count)
            .map(|_| module.ckks_ciphertext_alloc_from_glwe_infos(&folded_layout))
            .collect();
        with_scratch::<BE, _>(fold_bytes, |s| {
            module.ckks_fold(&mut folded, &ins, ring_switch.map(|keys| &keys.inbound), s)
        })
        .unwrap();

        for ct in &folded {
            assert_eq!(ct.noise(), None, "folded provenance, {label}");
        }

        let mut refresh_scratch = alloc_scratch(&params, module);
        let mut refreshed: Vec<_> = folded
            .iter()
            .zip(units.chunks(span))
            .enumerate()
            .map(|(i, (ct, group))| {
                assert_eq!(
                    (ct.meta(), ct.log_sparsity()),
                    (
                        GLWEPlaintextMeta {
                            scale: Scale::Log(log_delta),
                            slots: SlotsKind::Complex,
                            log_sparsity: 0,
                            slot_layout: SlotLayout::Full,
                        },
                        0
                    ),
                    "folded metadata {i}, {label}"
                );
                let got = decrypt_coeffs::<BE, F>(module, &params, ct, &sk, &mut refresh_scratch);
                assert_precision(
                    &format!("folded, {label}"),
                    &got,
                    &folded_message(&msgs, group, g, n),
                    log_delta,
                    n,
                );
                let prec = CKKSLayout {
                    meta: ct.meta(),
                    ..params.prec()
                };
                ckks_encrypt_coeffs(
                    &params,
                    module,
                    host_module,
                    &sk,
                    k_out,
                    &got,
                    prec,
                    &mut refresh_scratch.borrow(),
                )
            })
            .collect();

        let mut outs: Vec<_> = ins
            .iter()
            .map(|ct| {
                let mut out = module.ckks_ciphertext_alloc_from_glwe_infos(&output);
                out.set_meta(ct.meta());
                out
            })
            .collect();
        with_scratch::<BE, _>(fold_bytes, |s| {
            module.ckks_unfold(
                &mut outs,
                &mut refreshed,
                automorphisms,
                ring_switch.map(|keys| &keys.outbound),
                s,
            )
        })
        .unwrap();
        for (i, (out, msg)) in outs.iter().zip(&msgs).enumerate() {
            assert_eq!(
                (out.meta(), out.log_sparsity()),
                (ins[i].meta(), ins[i].log_sparsity()),
                "unfolded {i}, {label}"
            );
            assert_eq!(out.noise(), None, "unfolded provenance {i}, {label}");
            let got = decrypt_coeffs::<BE, F>(module, input_params, out, input_sk, &mut scratch);
            assert_precision(&format!("unfolded {i}, {label}"), &got, msg, log_delta, n_in);
        }

        if ring_switch.is_none() {
            GLWEToBackendMut::<BE>::set_noise(&mut ins[1], None);
            with_scratch::<BE, _>(fold_bytes, |s| {
                module.ckks_fold(&mut folded, &ins, ring_switch.map(|keys| &keys.inbound), s)
            })
            .unwrap();
            for ct in &folded {
                assert_eq!(ct.noise(), None, "mixed folded provenance, {label}");
            }
        }
    }
}

/// Coefficients of a message of degree `n` in `Z[X^(2^log_sparsity)]`; real slots
/// are self-conjugate, `c_(n−i) = −c_i`.
pub(crate) fn message<F: TestScalar>(n: usize, slots: SlotsKind, log_sparsity: usize, seed: usize) -> Vec<F> {
    let mut c: Vec<f64> = (0..n)
        .map(|i| {
            if i.is_multiple_of(1 << log_sparsity) {
                ((i * 37 + seed * 101) % 255) as f64 / 255.0 - 0.5
            } else {
                0.0
            }
        })
        .collect();
    if slots == SlotsKind::Real {
        c[n / 2] = 0.0;
        for i in n / 2 + 1..n {
            c[i] = -c[n - i];
        }
    }
    c.into_iter().map(|x| F::from_f64(x).unwrap()).collect()
}

/// `Σ_j X^j·(re_j + X^(n/2)·im_j)(X^g)` of degree `n`, over the units of `group`.
pub(crate) fn folded_message<F: TestScalar>(msgs: &[Vec<F>], group: &[(usize, Option<usize>)], g: usize, n: usize) -> Vec<F> {
    let mut want = vec![F::zero(); n];
    for (j, &(re, im)) in group.iter().enumerate() {
        for (offset, part) in [(0, Some(re)), (n / 2, im)] {
            for (i, &c) in part.map_or(&[][..], |part| &msgs[part]).iter().enumerate() {
                // Negacyclic: `X^n = −1`.
                let e = j + i * g + offset;
                want[e % n] = want[e % n] + if (e / n).is_multiple_of(2) { c } else { -c };
            }
        }
    }
    want
}

pub(crate) fn decrypt_coeffs<BE, F>(
    module: &Module<BE>,
    params: &CKKSTestParams,
    ct: &CKKSCiphertextOwned<BE>,
    sk: &GLWESecretPrepared<BE::OwnedBuf, BE>,
    scratch: &mut ScratchOwned<BE>,
) -> Vec<F>
where
    BE: TestContextBackend,
    Module<BE>: TestContextSharedModule<BE>,
    F: TestScalar,
{
    let prec = CKKSLayout {
        meta: ct.meta(),
        ..params.prec()
    };
    let pt = ckks_decrypt_with_prec(module, ct, sk, prec, &mut scratch.borrow()).unwrap();
    let mut coeffs = vec![F::zero(); ct.n().as_usize()];
    pt.decode_host_floats(&mut coeffs).unwrap();
    coeffs
}

/// A ciphertext layout of the fold tests: rank 1, scale `2^-log_delta`, dense.
fn fold_layout(n: usize, base2k: usize, k: usize, log_delta: usize, slots: SlotsKind) -> CKKSLayout {
    CKKSLayout {
        glwe_layout: GLWELayout {
            n: n.into(),
            base2k: base2k.into(),
            k: k.into(),
            rank: 1usize.into(),
        },
        meta: GLWEPlaintextMeta {
            slot_layout: poulpy_core::layouts::SlotLayout::Full,
            scale: Scale::Log(log_delta),
            slots,
            log_sparsity: 0,
        },
    }
}

/// A gadget key of degree `n` and radix `base2k` with digits of one limb, whose gadget covers `gadget_limbs` limbs
/// and whose storage holds `stored_limbs`, at least one more.
fn gadget_layout(n: usize, base2k: usize, gadget_limbs: usize, stored_limbs: usize) -> GGLWELayout {
    GGLWELayout {
        n: n.into(),
        base2k: base2k.into(),
        dnum: gadget_limbs.into(),
        k_digit: base2k.into(),
        k_offset: 0usize.into(),
        k_aux: ((stored_limbs - gadget_limbs) * base2k).into(),
        rank_in: 1usize.into(),
        rank_out: 1usize.into(),
        stride: 1,
    }
}

/// Inputs and folded ciphertexts [`CKKSFoldOps::ckks_fold`] refuses, each as `InvalidOperand` of `ckks_fold` for its
/// reason on an empty arena, before any check of the arena, `folded` untouched: an empty batch, folded ciphertexts of
/// two degrees or of rank 2, an input of rank 2, of a scale above its width or claiming more width than it stores,
/// inputs of two layouts, a sparsity reaching the packed degree, inputs wider than the fold degree, inputs of a smaller
/// degree without an inbound key, an inbound key whose `max_input_k` is below the input width, and a folded count other
/// than the batch's.
pub fn test_fold_rejects_invalid_operands<BE, F, E>(
    params: CKKSTestParams,
    module: &Module<BE>,
    _host_module: &Module<HostBytesBackend>,
) where
    BE: TestContextBackend<Ring = Standard>,
    Module<BE>: TestContextModule<BE> + CKKSFoldLayoutOps<BE> + CKKSFoldOps<BE, Standard>,
    F: TestScalar,
{
    let (n, b, log_delta) = (module.n(), params.base2k, 12);
    let k_in = 3 * b;
    let complex =
        |n: usize, k: usize, seed: u8| fixture_ciphertext(module, &fold_layout(n, b, k, log_delta, SlotsKind::Complex), seed);
    let valid_ins = || vec![complex(n, k_in, 1), complex(n, k_in, 2), complex(n, k_in, 3)];
    let valid_folded = || vec![complex(n, k_in, 4), complex(n, k_in, 5), complex(n, k_in, 6)];
    // Inputs of half the degree, under another secret: two merge into each folded ciphertext.
    let half = || vec![complex(n / 2, k_in, 7), complex(n / 2, k_in, 8)];
    // Its storage holds the switched limbs and a digit, only its gadget is short of the input.
    let narrow = prepared_gglwe(module, &gadget_layout(n, b, 2, 4), 10);
    assert!(narrow.gglwe_layout().max_input_k().as_usize() < k_in);
    let no_key: Option<&GGLWEPrepared<BE::OwnedBuf, BE>> = None;

    let rank2 = |n: usize, seed: u8| {
        let mut layout = fold_layout(n, b, k_in, log_delta, SlotsKind::Complex);
        layout.glwe_layout.rank = 2usize.into();
        fixture_ciphertext(module, &layout, seed)
    };
    let with = |edit: &dyn Fn(&mut CKKSCiphertextOwned<BE>)| {
        let mut ins = valid_ins();
        edit(&mut ins[1]);
        ins
    };
    type Case<'a, BE> = (
        &'a str,
        Vec<CKKSCiphertextOwned<BE>>,
        Vec<CKKSCiphertextOwned<BE>>,
        Option<&'a GGLWEPrepared<<BE as Backend>::OwnedBuf, BE>>,
        String,
    );
    let cases: Vec<Case<'_, BE>> = vec![
        (
            "folded of two degrees",
            valid_ins(),
            vec![complex(n, k_in, 11), complex(n / 2, k_in, 12)],
            no_key,
            "a fold needs folded ciphertexts of one degree".into(),
        ),
        (
            "folded of rank 2",
            valid_ins(),
            vec![rank2(n, 13)],
            no_key,
            "folded ciphertexts must have rank 1".into(),
        ),
        (
            "empty batch",
            Vec::new(),
            valid_folded(),
            no_key,
            "a fold needs at least one ciphertext".into(),
        ),
        (
            "input wider than its storage",
            with(&|ct| ct.set_k((ct.max_k().as_usize() + 1).into())),
            valid_folded(),
            no_key,
            format!(
                "fold ciphertext torus width {} exceeds its {k_in}-bit storage capacity",
                k_in + 1
            ),
        ),
        (
            "input of rank 2",
            vec![rank2(n, 14)],
            valid_folded(),
            no_key,
            "folding supports rank-1 ciphertexts only".into(),
        ),
        (
            "scale above the width",
            vec![fixture_ciphertext(
                module,
                &fold_layout(n, b, k_in, log_delta, SlotsKind::Complex),
                15,
            )]
            .into_iter()
            .map(|mut ct| {
                ct.set_meta(GLWEPlaintextMeta {
                    scale: Scale::Log(k_in + 1),
                    ..ct.meta()
                });
                ct
            })
            .collect(),
            valid_folded(),
            no_key,
            "the scale exceeds the width".into(),
        ),
        (
            "inputs of two widths",
            with(&|ct| ct.set_k((k_in - 1).into())),
            valid_folded(),
            no_key,
            "a batch must share its layout, scale and sparsity".into(),
        ),
        (
            "sparsity reaching the packed degree",
            valid_ins()
                .into_iter()
                .map(|mut ct| {
                    ct.set_log_sparsity(n.ilog2() as usize);
                    ct
                })
                .collect(),
            valid_folded(),
            no_key,
            "fold ciphertext sparsity exceeds its packed degree".into(),
        ),
        (
            "inputs wider than the fold degree",
            valid_ins(),
            vec![complex(n / 2, k_in, 16)],
            no_key,
            format!("the packed degree {n} must divide the fold degree {}", n / 2),
        ),
        (
            "smaller inputs without an inbound key",
            half(),
            vec![complex(n, k_in, 17)],
            no_key,
            "merging inputs of a smaller degree needs ring-switch keys".into(),
        ),
        (
            "inbound key narrower than the input",
            half(),
            vec![complex(n, k_in, 18)],
            Some(&narrow),
            "the input is wider than the inbound ring-switch key's max_input_k".into(),
        ),
        (
            "folded count",
            valid_ins(),
            vec![complex(n, k_in, 19), complex(n, k_in, 20)],
            no_key,
            "the batch folds into 3 ciphertexts, got 2".into(),
        ),
    ];
    for (label, ins, mut folded, inbound, reason) in cases {
        let before: Vec<_> = folded.iter().map(snapshot::<BE, _>).collect();
        let err = with_scratch::<BE, _>(0, |s| module.ckks_fold(&mut folded, &ins, inbound, s)).unwrap_err();
        assert_ckks_error(label, &err, CKKSCompositionError::InvalidOperand { op: "ckks_fold", reason });
        assert_eq!(
            folded.iter().map(snapshot::<BE, _>).collect::<Vec<_>>(),
            before,
            "{label}: a refused fold wrote folded"
        );
    }
}

/// Outputs, folded ciphertexts and keys [`CKKSFoldOps::ckks_unfold`] refuses on an empty arena, before any check of
/// the arena, `outs` and `folded` untouched: `MissingAutomorphismKey` of `ckks_unfold` for real pairs without keys,
/// without the conjugation key or with one narrower than the refreshed width, and for sparse outputs without their
/// split key; `InvalidOperand` of `ckks_unfold` for outputs or folded ciphertexts of rank 2, a folded count other than
/// the batch's, a folded scale above its width, a folded ciphertext of no message budget, outputs narrower than the
/// refreshed width, by their width or their allocation, and an outbound key whose `max_input_k` is below the refreshed
/// width.
pub fn test_unfold_rejects_invalid_operands<BE, F, E>(
    params: CKKSTestParams,
    module: &Module<BE>,
    _host_module: &Module<HostBytesBackend>,
) where
    BE: TestContextBackend<Ring = Standard>,
    Module<BE>: TestContextModule<BE> + CKKSFoldLayoutOps<BE> + CKKSFoldOps<BE, Standard>,
    F: TestScalar,
{
    enum Refusal {
        Operand(String),
        MissingKey(i64),
    }
    let (n, b, log_delta) = (module.n(), params.base2k, 12);
    let (k_refreshed, k_out) = (4 * b, 6 * b);
    let ct = |n: usize, k: usize, slots: SlotsKind, seed: u8| {
        fixture_ciphertext(module, &fold_layout(n, b, k, log_delta, slots), seed)
    };
    let outs = |slots: &[SlotsKind]| -> Vec<CKKSCiphertextOwned<BE>> {
        slots
            .iter()
            .zip(30u8..)
            .map(|(&slots, seed)| ct(n, k_out, slots, seed))
            .collect()
    };
    let refreshed = |count: usize| -> Vec<CKKSCiphertextOwned<BE>> {
        (0..count)
            .map(|i| ct(n, k_refreshed, SlotsKind::Complex, 40 + i as u8))
            .collect()
    };
    let rank2 = |k: usize, seed: u8| {
        let mut layout = fold_layout(n, b, k, log_delta, SlotsKind::Complex);
        layout.glwe_layout.rank = 2usize.into();
        fixture_ciphertext(module, &layout, seed)
    };
    let mixed = [SlotsKind::Complex, SlotsKind::Real, SlotsKind::Real];
    let complex = [SlotsKind::Complex];
    let conjugation: AutomorphismKeys<BE> = HashMap::from([(
        -1,
        prepared_automorphism_key(module, &key_layout(n, b, k_out, 1, 1, 1), -1, 50),
    )]);
    let short: AutomorphismKeys<BE> = HashMap::from([(
        -1,
        prepared_automorphism_key(module, &key_layout(n, b, 2 * b, 1, 1, 1), -1, 51),
    )]);
    let empty = AutomorphismKeys::<BE>::new();
    let narrow = prepared_gglwe(module, &gadget_layout(n, b, 3, 5), 52);
    assert!(narrow.gglwe_layout().max_input_k().as_usize() < k_refreshed);
    let sparse_element = {
        let mut layout = fold_layout(n, b, k_out, log_delta, SlotsKind::Complex);
        layout.meta.log_sparsity = 1;
        CKKSFoldOps::<_, Standard>::ckks_unfold_galois_elements(module, &layout)[0]
    };
    let sparse = {
        let mut outs = outs(&complex);
        outs[0].set_log_sparsity(1);
        outs
    };
    let narrow_out = {
        let mut outs = outs(&complex);
        outs[0].set_k((2 * b).into());
        outs
    };
    let scale_above = {
        let mut folded = refreshed(1);
        let meta = GLWEPlaintextMeta {
            scale: Scale::Log(k_refreshed + 1),
            ..folded[0].meta()
        };
        folded[0].set_meta(meta);
        folded
    };
    let no_budget = {
        let mut folded = refreshed(1);
        folded[0].set_k((log_delta + 1).into());
        folded
    };
    let not_covered = format!("unfold output width {} does not cover refreshed width {k_refreshed}", 2 * b);
    type Case<'a, BE> = (
        &'a str,
        Vec<CKKSCiphertextOwned<BE>>,
        Vec<CKKSCiphertextOwned<BE>>,
        Option<&'a AutomorphismKeys<BE>>,
        Option<&'a GGLWEPrepared<<BE as Backend>::OwnedBuf, BE>>,
        Refusal,
    );
    let cases: Vec<Case<'_, BE>> = vec![
        (
            "outputs of rank 2",
            vec![rank2(k_out, 60)],
            refreshed(1),
            None,
            None,
            Refusal::Operand("folding supports rank-1 ciphertexts only".into()),
        ),
        (
            "folded of rank 2",
            outs(&complex),
            vec![rank2(k_refreshed, 61)],
            None,
            None,
            Refusal::Operand("folded ciphertexts must have rank 1".into()),
        ),
        (
            "folded count",
            outs(&[SlotsKind::Complex; 3]),
            refreshed(2),
            None,
            None,
            Refusal::Operand("the batch unfolds from 3 ciphertexts, got 2".into()),
        ),
        (
            "folded scale above its width",
            outs(&complex),
            scale_above,
            None,
            None,
            Refusal::Operand("the folded scale exceeds the width".into()),
        ),
        (
            "folded of no message budget",
            outs(&complex),
            no_budget,
            None,
            None,
            Refusal::Operand("a folded ciphertext has no message budget".into()),
        ),
        (
            "outputs narrower than the refreshed width",
            narrow_out,
            refreshed(1),
            None,
            None,
            Refusal::Operand(not_covered.clone()),
        ),
        (
            "outputs allocated narrower than the refreshed width",
            vec![ct(n, 2 * b, SlotsKind::Complex, 62)],
            refreshed(1),
            None,
            None,
            Refusal::Operand(not_covered),
        ),
        (
            "outbound key narrower than the refreshed width",
            outs(&complex),
            refreshed(1),
            None,
            Some(&narrow),
            Refusal::Operand("the input is wider than the outbound ring-switch key's max_input_k".into()),
        ),
        (
            "pairs without keys",
            outs(&mixed),
            refreshed(2),
            None,
            None,
            Refusal::MissingKey(-1),
        ),
        (
            "pairs without the conjugation key",
            outs(&mixed),
            refreshed(2),
            Some(&empty),
            None,
            Refusal::MissingKey(-1),
        ),
        (
            "pairs with a short conjugation key",
            outs(&mixed),
            refreshed(2),
            Some(&short),
            None,
            Refusal::MissingKey(-1),
        ),
        (
            "sparse outputs without their split key",
            sparse,
            refreshed(1),
            Some(&conjugation),
            None,
            Refusal::MissingKey(sparse_element),
        ),
    ];
    for (label, mut outs, mut folded, automorphisms, outbound, refusal) in cases {
        let before = (
            outs.iter().map(snapshot::<BE, _>).collect::<Vec<_>>(),
            folded.iter().map(snapshot::<BE, _>).collect::<Vec<_>>(),
        );
        let err =
            with_scratch::<BE, _>(0, |s| module.ckks_unfold(&mut outs, &mut folded, automorphisms, outbound, s)).unwrap_err();
        match refusal {
            Refusal::Operand(reason) => {
                assert_ckks_error(
                    label,
                    &err,
                    CKKSCompositionError::InvalidOperand {
                        op: "ckks_unfold",
                        reason,
                    },
                );
            }
            Refusal::MissingKey(element) => assert!(
                matches!(
                    err.composition(),
                    Some(CKKSCompositionError::MissingAutomorphismKey { op: "ckks_unfold", rotation, .. }) if *rotation == element
                ),
                "{label}: unexpected error: {err}"
            ),
        }
        assert_eq!(
            (
                outs.iter().map(snapshot::<BE, _>).collect::<Vec<_>>(),
                folded.iter().map(snapshot::<BE, _>).collect::<Vec<_>>(),
            ),
            before,
            "{label}: a refused unfold wrote outs or folded"
        );
    }
}
