//! Shaped operations of a replicated slot layout: rounding, products,
//! automorphisms and polynomial evaluation with masked constants.
use super::{
    arithmetic::layout,
    helpers::{Snapshot, assert_untagged, fixture_ciphertext, fixture_plaintext, snapshot, with_scratch},
    keys::{key_layout, key_layout_bits, prepared_automorphism_key, prepared_tensor_key},
};
use crate::{
    CKKSLayout, CoeffsMeta, SlotLayout, SlotsKind,
    api::{
        CKKSAddOps, CKKSAllOpsTmpBytes, CKKSConjugateOps, CKKSEncodingHostOps, CKKSEncodingScalar, CKKSModuleInfos, CKKSMulOps,
        CKKSPolynomialEvaluationOps, CKKSRotateOps, CKKSRoundOps, CKKSSubOps,
    },
    layouts::encode_bsgs_replicated,
    oep::{CKKSEncodingImpl, CKKSImpl},
    polynomial::{Basis, Polynomial, SplitStrategy},
    shaping::ShapingUnit,
    test_suite::CKKSTestParams,
};
use poulpy_core::{
    GLWEMaskFill,
    layouts::{GLWEAutomorphismKeyPreparedFactory, GLWELayout, GLWETensorKeyPreparedFactory, LWEInfos},
};
use poulpy_hal::layouts::{Backend, Module};

/// A replicated layout of period 8 holding 2 values, on `2kR = 32` slots.
fn replicated(params: CKKSTestParams, k: usize) -> CKKSLayout {
    let b = params.base2k;
    let mut out = layout(params, 1, k, b - 1, params.n.ilog2() as usize - 5, SlotsKind::Complex);
    out.meta.slot_layout = SlotLayout::Replicated(ShapingUnit::U16_20.layout(1, 8).unwrap());
    out
}

fn replicated_ops<B, F>(params: CKKSTestParams, module: &Module<B>) -> Vec<(&'static str, Snapshot)>
where
    B: Backend<ZnxWord = i64> + CKKSImpl + CKKSEncodingImpl<F>,
    F: CKKSEncodingScalar,
    Module<B>: CKKSAllOpsTmpBytes<B> + GLWETensorKeyPreparedFactory<B> + GLWEAutomorphismKeyPreparedFactory<B> + GLWEMaskFill<B>,
{
    let b = params.base2k;
    let la = replicated(params, 8 * b + 5);
    let lr = replicated(params, 5 * b + 3);
    let a = fixture_ciphertext(module, &la, 71);
    let rhs = fixture_ciphertext(module, &la, 73);
    let before = snapshot::<B, _>(&a);
    let tsk_layout = key_layout(params.n, b, la.k().as_usize(), 1, 1, 1);
    let tsk = prepared_tensor_key(module, &tsk_layout, 113);
    let atk_layout = key_layout(params.n, b, la.k().as_usize(), 1, 1, 1);
    let shift = 8;
    let rot = prepared_automorphism_key(module, &atk_layout, module.ckks_galois_element(shift), 62);
    let conj = prepared_automorphism_key(module, &atk_layout, -1, 63);
    let mut results = Vec::new();

    let mut out = fixture_ciphertext(module, &replicated(params, la.k().as_usize() - b / 2), 99);
    with_scratch::<B, _>(module.ckks_round_tmp_bytes(&out, &a), |s| {
        module.ckks_round_into(&mut out, &a, s)
    })
    .unwrap();
    results.push(("round_into", snapshot::<B, _>(&out)));

    let mut out = fixture_ciphertext(module, &lr, 99);
    with_scratch::<B, _>(module.ckks_mul_tmp_bytes(&out, &a, &rhs, &tsk_layout), |s| {
        module.ckks_mul_into(&mut out, &a, &rhs, &tsk, s)
    })
    .unwrap();
    results.push(("mul_into", snapshot::<B, _>(&out)));

    let mut out = fixture_ciphertext(module, &lr, 99);
    with_scratch::<B, _>(module.ckks_square_tmp_bytes(&out, &a, &tsk_layout), |s| {
        module.ckks_square_into(&mut out, &a, &tsk, s)
    })
    .unwrap();
    results.push(("square_into", snapshot::<B, _>(&out)));

    let mut lp = replicated(params, b + 3);
    lp.glwe_layout = GLWELayout {
        n: 32usize.into(),
        rank: 0usize.into(),
        ..lp.glwe_layout
    };
    lp.meta.log_sparsity = 0;
    let pt = fixture_plaintext(module, &lp, 54);
    let mut out = fixture_ciphertext(module, &lr, 99);
    with_scratch::<B, _>(module.ckks_mul_pt_vec_tmp_bytes(&out, &a, &pt), |s| {
        module.ckks_mul_pt_vec_into(&mut out, &a, &pt, s)
    })
    .unwrap();
    results.push(("mul_pt_vec_into", snapshot::<B, _>(&out)));

    let mut out = fixture_ciphertext(module, &lr, 99);
    with_scratch::<B, _>(module.ckks_rotate_tmp_bytes(&a, &atk_layout), |s| {
        module.ckks_rotate_into(&mut out, &a, shift, &rot, s)
    })
    .unwrap();
    results.push(("rotate_into", snapshot::<B, _>(&out)));

    let mut out = fixture_ciphertext(module, &lr, 99);
    with_scratch::<B, _>(module.ckks_conjugate_tmp_bytes(&a, &atk_layout), |s| {
        module.ckks_conjugate_into(&mut out, &a, &conj, s)
    })
    .unwrap();
    results.push(("conjugate_into", snapshot::<B, _>(&out)));

    let mut source_layout = la;
    source_layout.glwe_layout.k = 130usize.into();
    source_layout.meta.scale = crate::Scale::Log(40);
    source_layout.meta.log_sparsity = 0;
    let shaped_layout = ShapingUnit::U16_40.layout(1, 24).unwrap();
    for slot_layout in [SlotLayout::Full, SlotLayout::Replicated(shaped_layout)] {
        source_layout.meta.slot_layout = slot_layout;
        let source = fixture_ciphertext(module, &source_layout, 91);
        let source_before = snapshot::<B, _>(&source);
        let mut destination_layout = source_layout;
        destination_layout.glwe_layout.k = 110usize.into();
        for conjugate in [false, true] {
            let p = if conjugate { -1 } else { module.ckks_galois_element(8) };
            for key_bits in [110, 180] {
                let key_infos = key_layout_bits(params.n, b, key_bits, 10, 1, 1);
                let key = prepared_automorphism_key(module, &key_infos, p, 93);
                let mut destination = fixture_ciphertext(module, &destination_layout, 95);
                let before = snapshot::<B, _>(&destination);
                let bytes = if conjugate {
                    module.ckks_conjugate_tmp_bytes(&source, &key_infos)
                } else {
                    module.ckks_rotate_tmp_bytes(&source, &key_infos)
                };
                let result = with_scratch::<B, _>(bytes, |scratch| {
                    if conjugate {
                        module.ckks_conjugate_into(&mut destination, &source, &key, scratch)
                    } else {
                        module.ckks_rotate_into(&mut destination, &source, 8, &key, scratch)
                    }
                });
                if slot_layout != SlotLayout::Full && key_bits == 110 {
                    let err = result.unwrap_err();
                    let required = 110 + crate::shaping::ShapingProfile::new(&shaped_layout).shaping_depth(40, shaped_layout.w());
                    let expected_op = if conjugate { "conjugate" } else { "rotate" };
                    assert!(
                        matches!(err.composition(), Some(crate::CKKSCompositionError::MissingAutomorphismKey { op, k, .. }) if *op == expected_op && *k as usize == required)
                    );
                    assert_eq!(snapshot::<B, _>(&destination), before);
                } else {
                    result.unwrap();
                    results.push((
                        if conjugate { "narrow_conjugate" } else { "narrow_rotate" },
                        snapshot::<B, _>(&destination),
                    ));
                }
                assert_eq!(snapshot::<B, _>(&source), source_before);
            }
        }
    }

    for (label, add) in [("add_one_assign", true), ("sub_one_assign", false)] {
        let mut out = fixture_ciphertext(module, &la, 71);
        let bytes = if add {
            module.ckks_add_one_tmp_bytes(out.size())
        } else {
            module.ckks_sub_one_tmp_bytes(out.size())
        };
        with_scratch::<B, _>(bytes, |s| {
            if add {
                module.ckks_add_one_assign(&mut out, s)
            } else {
                module.ckks_sub_one_assign(&mut out, s)
            }
        })
        .unwrap();
        results.push((label, snapshot::<B, _>(&out)));
    }

    let coeff_meta = CoeffsMeta::from_delta_budget(b - 1, 4);
    let polynomial = Polynomial::new(
        Basis::Monomial,
        [0.125, -0.25, 0.0625, 0.125]
            .into_iter()
            .map(|v| F::from_f64(v).unwrap())
            .collect(),
    );
    let SlotLayout::Replicated(rl) = la.meta.slot_layout else {
        unreachable!()
    };
    let encoding = <Module<B> as CKKSEncodingHostOps<B, F>>::ckks_reim_tmp_bytes(module, module.n() / 2);
    let bsgs = with_scratch::<B, _>(encoding, |s| {
        encode_bsgs_replicated(&polynomial, module, b.into(), coeff_meta, SplitStrategy::MinDepth, rl, s)
    })
    .unwrap();
    let pt_layout = CKKSLayout {
        glwe_layout: GLWELayout {
            k: coeff_meta.k,
            rank: 0usize.into(),
            ..la.glwe_layout
        },
        meta: coeff_meta.meta,
    };
    let mut out = fixture_ciphertext(module, &la, 99);
    with_scratch::<B, _>(module.ckks_all_ops_tmp_bytes(&la, &tsk_layout, &pt_layout), |s| {
        module.ckks_eval_poly_real_const_coeffs(&mut out, &a, &bsgs, &tsk, s)
    })
    .unwrap();
    results.push(("eval_poly", snapshot::<B, _>(&out)));

    assert_eq!(before, snapshot::<B, _>(&a));
    results
}

/// Rounding, products, automorphisms and polynomial evaluation of a
/// replicated layout give the same digits and metadata on both backends.
pub fn test_replicated_parity<BR, BT, F>(params: CKKSTestParams, reference: &Module<BR>, tested: &Module<BT>)
where
    BR: Backend<ZnxWord = i64> + CKKSImpl + CKKSEncodingImpl<F>,
    BT: Backend<ZnxWord = i64> + CKKSImpl + CKKSEncodingImpl<F>,
    F: CKKSEncodingScalar,
    Module<BR>:
        CKKSAllOpsTmpBytes<BR> + GLWETensorKeyPreparedFactory<BR> + GLWEAutomorphismKeyPreparedFactory<BR> + GLWEMaskFill<BR>,
    Module<BT>:
        CKKSAllOpsTmpBytes<BT> + GLWETensorKeyPreparedFactory<BT> + GLWEAutomorphismKeyPreparedFactory<BT> + GLWEMaskFill<BT>,
{
    assert_eq!(reference.n(), tested.n());
    let want = replicated_ops::<BR, F>(params, reference);
    assert_untagged("replicated", want.iter().map(|(_, output)| output));
    assert_eq!(want, replicated_ops::<BT, F>(params, tested), "replicated parity differs");
}
