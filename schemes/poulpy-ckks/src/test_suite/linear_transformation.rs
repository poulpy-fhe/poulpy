//! Linear transformation (BSGS) test.
//!
//! Minimal correctness check: `dec(lt(enc(a), B)) ≈ B·a`, where `a` is the
//! encrypted complex slot vector and `B` is a complex matrix supplied as a raw
//! diagonal map. The transform applies the matrix on the *left*
//! (`(B·a)[j] = Σ_k B[j][k]·a[k]`); the expected value is computed by the
//! scheme-agnostic plaintext evaluator [`ComplexDiagonals`] / [`Evaluate`],
//! which the homomorphic engine must match up to CKKS precision.

use crate::api::CKKSEncodingOps;
use poulpy_core::test_suite::keys::TransformKeys;
use std::collections::{HashMap, HashSet};

use poulpy_core::{
    GLWEBytesOf,
    layouts::{
        BackendGLWESecret, Diagonals, Evaluate, GGLWEInfos, GLWEInfos, GLWELayout, GLWEPlaintextInfos, GLWEToBackendRef,
        GetAutomorphismKey, GetTensorKey, LWEInfos, LinearTransformationDiagonal, LinearTransformationGiantStep,
        LinearTransformationPlan, LinearTransformationStrategy, Rank, SetGLWEPlaintextInfos, TorusPrecision, k_digit_of,
        prepared::{
            GLWEAutomorphismKeyPrepared, GLWEAutomorphismKeyPreparedBackendRef, GLWEAutomorphismKeyPreparedToBackendRef,
            GLWESecretPrepared, GLWETensorKeyPreparedBackendRef,
        },
    },
};
use poulpy_hal::{
    api::{CnvPVecAlloc, CnvPVecBytesOf, NegacyclicFFT, NegacyclicFFTNew, ScratchAvailable, ScratchOwnedBorrow},
    layouts::{
        Backend, CyclotomicOrder, DataView, DataViewMut, HostBytesBackend, Module, PrepareHint, ScratchArena, Standard,
        galois_element,
    },
};

use crate::{
    CKKSCompositionError, CKKSCtBounds, CKKSInfos, GLWEPlaintextMeta, Scale, SetCKKSInfos, SlotsKind,
    api::{
        CKKSCopyOps, CKKSLinearTransformationOps, LinearTransformation, LinearTransformationBabySteps,
        LinearTransformationPrepared,
    },
    layouts::{CKKSCiphertextOwned, CKKSModuleAlloc, CKKSPlaintextOwned, ComplexDiagonals},
    test_suite::reference_encoder::ReferenceEncoder,
    test_suite::{
        CKKSTestParams,
        dot_product::TensorKeyAt,
        helpers::{
            QueryLog, TestContextBackend, TestContextModule, TestScalar, alloc_ct, alloc_scratch, assert_ckks_error,
            assert_ckks_internal, assert_decrypt_precision, assert_invalid_operand, assert_mul_ct_output_meta, ckks_encrypt,
            dirty_ct, gen_atk, gen_sk_with_raw, gen_tsk, pt_eval_tmp_bytes, pt_streamed_eval_tmp_bytes, test_vector_1,
            test_vector_2, want_add,
        },
        parity::helpers::{snapshot, with_scratch},
    },
};

/// One complex diagonal of the test matrix at index `i`.
fn diagonal<F: TestScalar>(i: usize, m: usize) -> (Vec<F>, Vec<F>) {
    let re: Vec<F> = (0..m)
        .map(|j| F::from_f64(0.25 * (i as f64 + 1.0) / (1.0 + (j % 8) as f64)).unwrap())
        .collect();
    let im: Vec<F> = (0..m)
        .map(|j| F::from_f64(0.125 * (i as f64 + 1.0) / (1.0 + ((j + 3) % 8) as f64)).unwrap())
        .collect();
    (re, im)
}

/// Builds the complex matrix `B` as a raw diagonal map over the given non-zero
/// diagonal indices (no pre-rotation, that is the encoder's job). Diagonal `i`
/// carries the values of test diagonal `i + shift`, so matrices with one set of
/// indices, hence one BSGS schedule, can differ.
fn complex_diagonals<F: TestScalar>(diag_indices: &[usize], m: usize, shift: usize) -> ComplexDiagonals<F> {
    let mut re = Diagonals::<F>::new(m);
    let mut im = Diagonals::<F>::new(m);
    for &i in diag_indices {
        let (dr, di) = diagonal::<F>(i + shift, m);
        re.set(i as i64, dr);
        im.set(i as i64, di);
    }
    ComplexDiagonals::new(re, im)
}

/// Encodes the diagonal map `b` into a `LinearTransformation<CKKSPlaintext>`
/// under the BSGS schedule for `n1`, via the production helper. `transpose`
/// picks the matrix-vector orientation: `false` → `B·a`, `true` → `a·B`.
fn encode_lt<BE, F>(
    module: &Module<BE>,
    params: &CKKSTestParams,
    b: &ComplexDiagonals<F>,
    n1: usize,
    transpose: bool,
    scratch: &mut ScratchArena<'_, BE>,
) -> LinearTransformation<CKKSPlaintextOwned<BE>>
where
    BE: TestContextBackend<Ring = Standard>,
    Module<BE>: TestContextModule<BE> + CKKSEncodingOps<BE, F>,
    F: TestScalar,
{
    crate::encoding::ckks_encode_linear_transformation_from_diagonals(
        module,
        params.base2k.into(),
        params.prec().into(),
        b,
        LinearTransformationStrategy::Bsgs { giant_step: n1 },
        transpose,
        scratch,
    )
    .unwrap()
}

/// Materializes the prepared (resident-RHS) form of a linear transformation of plaintext or ciphertext diagonals, so
/// the same unified evaluator can be exercised on both the resident (`P = PreparedDiagonal`) and streamed
/// (`P = CKKSPlaintext`) paths.
fn prepare_lt<BE, P>(module: &Module<BE>, lt: &LinearTransformation<P>) -> LinearTransformationPrepared<BE>
where
    BE: TestContextBackend<Ring = Standard>,
    for<'a> <BE as poulpy_hal::layouts::Backend>::BufRef<'a>: poulpy_hal::layouts::HostDataRef,
    for<'a> <BE as poulpy_hal::layouts::Backend>::BufMut<'a>: poulpy_hal::layouts::HostDataMut,
    Module<BE>: TestContextModule<BE> + CKKSLinearTransformationOps<BE> + CnvPVecAlloc<BE>,
    P: GLWEToBackendRef<BE> + CKKSCtBounds,
{
    let first = lt.first_diagonal_plaintext().expect("linear transformation has no diagonals");
    let mut prepared = LinearTransformationPrepared::<BE>::alloc_prepared_from_index(module, &lt.index(), first);
    with_scratch::<BE, _>(module.ckks_prepare_linear_transformation_rhs_tmp_bytes(first), |s| {
        module.ckks_prepare_linear_transformation_rhs(&mut prepared, lt, s)
    })
    .unwrap();
    prepared
}

/// `dec(lt(enc(a), B)) ≈ B·a` for a complex matrix `B` in BSGS form.
pub fn test_linear_transformation<BE, F, E>(params: CKKSTestParams, module: &Module<BE>, host_module: &Module<HostBytesBackend>)
where
    BE: TestContextBackend<Ring = Standard>,
    for<'a> <BE as poulpy_hal::layouts::Backend>::BufRef<'a>: poulpy_hal::layouts::HostDataRef,
    for<'a> <BE as poulpy_hal::layouts::Backend>::BufMut<'a>: poulpy_hal::layouts::HostDataMut,
    Module<BE>: TestContextModule<BE> + CKKSEncodingOps<BE, F> + CKKSLinearTransformationOps<BE> + CnvPVecAlloc<BE>,
    F: TestScalar,
    E: NegacyclicFFT<F> + NegacyclicFFTNew<F>,
    for<'a> ScratchArena<'a, BE>: ScratchAvailable,
{
    let m = params.n / 2;
    let encoder = ReferenceEncoder::<E>::new(m).unwrap();
    let (a_re, a_im) = test_vector_1::<F>(m);
    let (sk_raw, sk) = gen_sk_with_raw(&params, module, host_module, [0u8; 32]);
    // Exercise multi-limb automorphism products and multiple non-identity
    // giant rotations, so the shortened shared accumulator is covered.
    let key_params = CKKSTestParams {
        dsize: params.dsize.max(4),
        ..params
    };
    let mut scratch = alloc_scratch(&key_params, module);

    // Complex matrix B (six diagonals), decomposed with n1 = 2 baby steps.
    // Encode both orientations through the `transpose` flag and check each one.
    let n1 = 2;
    let strategy = LinearTransformationStrategy::Bsgs { giant_step: n1 };
    let b = complex_diagonals::<F>(&[0, 1, 2, 3, 4, 5], m, 0);
    let lt_left = encode_lt(module, &params, &b, n1, false, &mut scratch.borrow());
    let lt_right = encode_lt(module, &params, &b, n1, true, &mut scratch.borrow());

    // Automorphism keys are indexed by Galois element throughout the engine.
    let order = module.cyclotomic_order();
    let mut atks = HashMap::new();
    for p in lt_left
        .galois_elements(order)
        .into_iter()
        .chain(lt_right.galois_elements(order))
    {
        atks.entry(p)
            .or_insert_with(|| gen_atk(&key_params, module, p, &sk_raw, &mut scratch.borrow()));
    }

    let ct = ckks_encrypt(
        &params,
        module,
        host_module,
        &encoder,
        &sk,
        params.k,
        &a_re,
        &a_im,
        &mut scratch.borrow(),
    );

    // The documented bound of `self_into`: the larger of the baby-step and evaluation queries.
    let self_into_bytes = |plan: &LinearTransformationPlan, streamed: bool| {
        let eval = if streamed {
            pt_streamed_eval_tmp_bytes(module, &ct, plan, &key_params.atk_layout())
        } else {
            pt_eval_tmp_bytes(module, &ct, plan, &key_params.atk_layout())
        };
        module
            .ckks_prepare_linear_transformation_baby_steps_tmp_bytes(&ct, plan, &key_params.atk_layout())
            .max(eval)
    };

    // transpose = false: dec(lt(enc(a), B)) ≈ B·a. Resident path (prepared RHS).
    let prepared_left = prepare_lt(module, &lt_left);
    let mut ct_left = alloc_ct(&params, module, params.k);
    with_scratch::<BE, _>(self_into_bytes(&lt_left.index(), false), |s| {
        module.ckks_eval_linear_transformation_self_into(&mut ct_left, &ct, &prepared_left, &atks, s)
    })
    .unwrap();
    let (want_left_re, want_left_im) = b.evaluate((a_re.as_slice(), a_im.as_slice()), strategy);
    assert_decrypt_precision(
        "linear_transformation_B_times_a",
        &params,
        module,
        &encoder,
        &ct_left,
        &sk,
        &want_left_re,
        &want_left_im,
        &mut scratch.borrow(),
    );

    // Streamed (plaintext-RHS) path must match the resident path against the
    // same reference — same unified evaluator, different `P`.
    let mut ct_left_streamed = alloc_ct(&params, module, params.k);
    with_scratch::<BE, _>(self_into_bytes(&lt_left.index(), true), |s| {
        module.ckks_eval_linear_transformation_self_into(&mut ct_left_streamed, &ct, &lt_left, &atks, s)
    })
    .unwrap();
    assert_decrypt_precision(
        "linear_transformation_B_times_a_streamed",
        &params,
        module,
        &encoder,
        &ct_left_streamed,
        &sk,
        &want_left_re,
        &want_left_im,
        &mut scratch.borrow(),
    );

    // transpose = true: dec(lt(enc(a), Bᵀ)) ≈ Bᵀ·a = a·B. Resident path.
    let prepared_right = prepare_lt(module, &lt_right);
    let mut ct_right = alloc_ct(&params, module, params.k);
    with_scratch::<BE, _>(self_into_bytes(&lt_right.index(), false), |s| {
        module.ckks_eval_linear_transformation_self_into(&mut ct_right, &ct, &prepared_right, &atks, s)
    })
    .unwrap();
    let mut b_t = b.clone();
    b_t.transpose();
    let (want_right_re, want_right_im) = b_t.evaluate((a_re.as_slice(), a_im.as_slice()), strategy);
    assert_decrypt_precision(
        "linear_transformation_a_times_B",
        &params,
        module,
        &encoder,
        &ct_right,
        &sk,
        &want_right_re,
        &want_right_im,
        &mut scratch.borrow(),
    );
}

/// The summed transform with shared giant steps, `Σ_t M_t·a_t`, on its exact
/// poisoned budget.
///
/// One term is the single transform bit for bit. Distinct terms decrypt to the
/// sum of the single transforms and carry the single transform's metadata, and their streamed diagonals give the same
/// bits on the streamed query's budget. A mismatched term is rejected, for its own reason, before `dst` is written.
pub fn test_linear_transformation_sum<BE, F, E>(
    params: CKKSTestParams,
    module: &Module<BE>,
    host_module: &Module<HostBytesBackend>,
) where
    BE: TestContextBackend<Ring = Standard>,
    for<'a> <BE as poulpy_hal::layouts::Backend>::BufRef<'a>: poulpy_hal::layouts::HostDataRef,
    for<'a> <BE as poulpy_hal::layouts::Backend>::BufMut<'a>: poulpy_hal::layouts::HostDataMut,
    Module<BE>: TestContextModule<BE> + CKKSEncodingOps<BE, F> + CKKSLinearTransformationOps<BE> + CnvPVecAlloc<BE>,
    F: TestScalar,
    E: NegacyclicFFT<F> + NegacyclicFFTNew<F>,
    for<'a> ScratchArena<'a, BE>: ScratchAvailable,
{
    let m = params.n / 2;
    let encoder = ReferenceEncoder::<E>::new(m).unwrap();
    let (sk_raw, sk) = gen_sk_with_raw(&params, module, host_module, [0u8; 32]);
    let mut scratch = alloc_scratch(&params, module);

    // Three matrices on the same diagonals, hence one schedule, with distinct values.
    let n1 = 2;
    let strategy = LinearTransformationStrategy::Bsgs { giant_step: n1 };
    let bs: Vec<ComplexDiagonals<F>> = (0..3).map(|t| complex_diagonals(&[0, 1, 2, 3, 4, 5], m, t)).collect();
    let lts: Vec<_> = bs
        .iter()
        .map(|b| encode_lt(module, &params, b, n1, false, &mut scratch.borrow()))
        .collect();
    let prepared: Vec<_> = lts.iter().map(|lt| prepare_lt(module, lt)).collect();
    let keys: HashMap<i64, _> = lts[0]
        .galois_elements(module.cyclotomic_order())
        .into_iter()
        .map(|p| (p, gen_atk(&params, module, p, &sk_raw, &mut scratch.borrow())))
        .collect();

    let inputs = [test_vector_1::<F>(m), test_vector_2::<F>(m)];
    let cts: Vec<_> = inputs
        .iter()
        .map(|(re, im)| {
            ckks_encrypt(
                &params,
                module,
                host_module,
                &encoder,
                &sk,
                params.k,
                re,
                im,
                &mut scratch.borrow(),
            )
        })
        .collect();
    let babies: Vec<_> = cts
        .iter()
        .map(|ct| {
            let mut babies = LinearTransformationBabySteps::alloc(module, prepared[0].baby_steps(), ct);
            with_scratch::<BE, _>(
                module.ckks_prepare_linear_transformation_baby_steps_tmp_bytes(ct, &prepared[0].index(), &params.atk_layout()),
                |s| module.ckks_prepare_linear_transformation_baby_steps(&mut babies, ct, &keys, s),
            )
            .unwrap();
            babies
        })
        .collect();
    let sum_bytes = module.ckks_eval_linear_transformation_sum_tmp_bytes(
        &cts[0],
        &crate::api::ckks_plaintext_diagonal_layout(&cts[0]),
        &prepared[0].index(),
        &params.atk_layout(),
        &params.atk_layout(),
    );

    // One term: the single transform, bit for bit.
    let mut single = alloc_ct(&params, module, params.k);
    with_scratch::<BE, _>(
        pt_eval_tmp_bytes(module, &cts[0], &prepared[0].index(), &params.atk_layout()),
        |s| module.ckks_eval_linear_transformation_into(&mut single, &cts[0], &babies[0], &prepared[0], &keys, s),
    )
    .unwrap();
    let single = snapshot::<BE, _>(&single);
    let mut one = alloc_ct(&params, module, params.k);
    with_scratch::<BE, _>(sum_bytes, |scratch| {
        module.ckks_eval_linear_transformation_sum_into(&mut one, &[&cts[0]], &[&babies[0]], &[&prepared[0]], &keys, scratch)
    })
    .unwrap();
    assert_eq!(snapshot::<BE, _>(&one), single, "a one-term sum is not the single transform");

    // Distinct terms, as (source, matrix): the sum of the single transforms.
    let terms = [(0, 0), (1, 1), (0, 2)];
    let term_srcs: Vec<_> = terms.iter().map(|&(s, _)| &cts[s]).collect();
    let term_babies: Vec<_> = terms.iter().map(|&(s, _)| &babies[s]).collect();
    let term_lts: Vec<_> = terms.iter().map(|&(_, l)| &prepared[l]).collect();
    let mut sum = alloc_ct(&params, module, params.k);
    with_scratch::<BE, _>(sum_bytes, |scratch| {
        module.ckks_eval_linear_transformation_sum_into(&mut sum, &term_srcs, &term_babies, &term_lts, &keys, scratch)
    })
    .unwrap();
    let have = snapshot::<BE, _>(&sum);
    assert_eq!(
        (&have.layout, have.canonical, &have.noise),
        (&single.layout, single.canonical, &single.noise),
        "the sum's metadata is not the single transform's"
    );
    let (want_re, want_im) = terms
        .iter()
        .map(|&(s, l)| bs[l].evaluate((inputs[s].0.as_slice(), inputs[s].1.as_slice()), strategy))
        .reduce(|(re, im), (re_t, im_t)| want_add(&re, &im, &re_t, &im_t))
        .unwrap();
    assert_decrypt_precision(
        "linear_transformation_sum",
        &params,
        module,
        &encoder,
        &sum,
        &sk,
        &want_re,
        &want_im,
        &mut scratch.borrow(),
    );

    // The same terms streamed, plaintext diagonals: the resident sum, bit for bit.
    let streamed_lts: Vec<_> = terms.iter().map(|&(_, l)| &lts[l]).collect();
    let mut streamed = alloc_ct(&params, module, params.k);
    with_scratch::<BE, _>(
        module.ckks_eval_linear_transformation_sum_streamed_tmp_bytes(
            &cts[0],
            &crate::api::ckks_plaintext_diagonal_layout(&cts[0]),
            &prepared[0].index(),
            &params.atk_layout(),
            &params.atk_layout(),
        ),
        |scratch| {
            module.ckks_eval_linear_transformation_sum_into(
                &mut streamed,
                &term_srcs,
                &term_babies,
                &streamed_lts,
                &keys,
                scratch,
            )
        },
    )
    .unwrap();
    assert_eq!(snapshot::<BE, _>(&streamed), have, "the streamed sum is not the resident one");

    // A mismatched term: rejected as `InvalidOperand` for `reason`, with `sum` untouched.
    let mut reject = |srcs: &[&CKKSCiphertextOwned<BE>],
                      cache: &[&LinearTransformationBabySteps<BE>],
                      lts: &[&LinearTransformationPrepared<BE>],
                      reason: &str| {
        let before = snapshot::<BE, _>(&sum);
        let err = with_scratch::<BE, _>(0, |scratch| {
            module.ckks_eval_linear_transformation_sum_into(&mut sum, srcs, cache, lts, &keys, scratch)
        })
        .unwrap_err();
        assert_invalid_operand(reason, &err, "ckks_eval_linear_transformation_sum_into", reason);
        assert_eq!(snapshot::<BE, _>(&sum), before, "a rejected sum wrote dst ({reason})");
    };
    let two = [&cts[0], &cts[1]];
    reject(&[], &[], &[], "has no terms");
    reject(&two, &[&babies[0]], &[&prepared[0], &prepared[1]], "differ in length");
    let empty: LinearTransformationPrepared<BE> = LinearTransformation {
        baby_steps: vec![0],
        giant_steps: Vec::new(),
        rotations: Default::default(),
    };
    reject(&[&cts[0]], &[&babies[0]], &[&empty], "has no diagonals");
    let other_schedule = prepare_lt(
        module,
        &encode_lt(module, &params, &bs[1], 2 * n1, false, &mut scratch.borrow()),
    );
    reject(
        &two,
        &[&babies[0], &babies[1]],
        &[&prepared[0], &other_schedule],
        "giant schedule",
    );
    let mut bad = prepare_lt(module, &lts[1]);
    let meta = bad.plaintext_meta();
    let Scale::Log(scale) = meta.scale;
    let shifted = GLWEPlaintextMeta {
        scale: Scale::Log(scale + 1),
        ..meta
    };
    bad.set_plaintext_meta(shifted);
    reject(
        &two,
        &[&babies[0], &babies[1]],
        &[&prepared[0], &bad],
        "diagonal layout, width, scale and slot kind",
    );
    // Only a later diagonal differs, which core's first-diagonal check misses.
    bad.set_plaintext_meta(meta);
    bad.giant_steps.last_mut().unwrap().diagonals[0]
        .plaintext
        .set_plaintext_meta(shifted);
    reject(&two, &[&babies[0], &babies[1]], &[&prepared[0], &bad], "not uniform");
    let narrow = alloc_ct(&params, module, params.k - params.base2k);
    reject(
        &[&cts[0], &narrow],
        &[&babies[0], &babies[1]],
        &[&prepared[0], &prepared[1]],
        "source layout and scale",
    );
    // Core reads the sources through their caches: one built from a narrower source is rejected here, not there.
    let narrow_babies = LinearTransformationBabySteps::alloc(module, prepared[1].baby_steps(), &narrow);
    reject(
        &two,
        &[&babies[0], &narrow_babies],
        &[&prepared[0], &prepared[1]],
        "baby-step cache layout",
    );
    // An empty cache holds no limb to read a layout from.
    let empty_babies = LinearTransformationBabySteps::alloc(module, &[], &cts[1]);
    reject(
        &two,
        &[&babies[0], &empty_babies],
        &[&prepared[0], &prepared[1]],
        "missing prepared baby-step rotation",
    );
    let no_babies = LinearTransformationBabySteps::alloc(module, &[0], &cts[1]);
    reject(
        &two,
        &[&babies[0], &no_babies],
        &[&prepared[0], &prepared[1]],
        "missing prepared baby-step rotation",
    );

    // The single transform refuses the same transform without diagonals and cache without baby steps.
    let mut dst = dirty_ct(module, &cts[1], params.k);
    for (label, cache, lt, reason) in [
        ("no diagonals", &babies[1], &empty, "linear transformation has no diagonals"),
        (
            "missing baby step",
            &no_babies,
            &prepared[1],
            "missing prepared baby-step rotation",
        ),
    ] {
        let before = snapshot::<BE, _>(&dst);
        let err = with_scratch::<BE, _>(0, |s| {
            module.ckks_eval_linear_transformation_into(&mut dst, &cts[1], cache, lt, &keys, s)
        })
        .unwrap_err();
        assert_invalid_operand(label, &err, "ckks_eval_linear_transformation_into", reason);
        assert_eq!(snapshot::<BE, _>(&dst), before, "{label}: a refused transform wrote dst");
    }
}

/// Prepares `lt` into `res` on an empty arena, which must be refused as `InvalidOperand` for `reason` before any write,
/// `res` staying equal to `want`.
fn assert_prepare_refused<BE, P>(
    module: &Module<BE>,
    res: &mut LinearTransformationPrepared<BE>,
    want: &LinearTransformationPrepared<BE>,
    lt: &LinearTransformation<P>,
    reason: &str,
) where
    BE: TestContextBackend<Ring = Standard>,
    Module<BE>: CKKSLinearTransformationOps<BE>,
    P: GLWEToBackendRef<BE> + CKKSCtBounds,
{
    let err = with_scratch::<BE, _>(0, |scratch| module.ckks_prepare_linear_transformation_rhs(res, lt, scratch)).unwrap_err();
    assert!(
        matches!(
            err.composition(),
            Some(CKKSCompositionError::InvalidOperand {
                op: "ckks_prepare_linear_transformation_rhs",
                ..
            })
        ),
        "{reason}: {err:?}"
    );
    assert!(err.to_string().contains(reason), "expected {reason:?}, got: {err}");
    assert_prepared_unchanged(res, want, reason);
}

/// Asserts that a refused preparation left `res` equal to `want`.
fn assert_prepared_unchanged<BE>(res: &LinearTransformationPrepared<BE>, want: &LinearTransformationPrepared<BE>, reason: &str)
where
    BE: TestContextBackend<Ring = Standard>,
{
    let diagonals = |lt: &LinearTransformationPrepared<BE>| {
        lt.giant_steps
            .iter()
            .flat_map(|gs| gs.diagonals.iter().map(move |d| (gs.rot, d.baby)))
            .collect::<Vec<_>>()
    };
    let stashes = |lt: &LinearTransformationPrepared<BE>| {
        lt.giant_steps
            .iter()
            .flat_map(|gs| &gs.diagonals)
            .map(|d| d.plaintext.plaintext_meta())
            .collect::<Vec<_>>()
    };
    assert_eq!(
        (diagonals(res), stashes(res)),
        (diagonals(want), stashes(want)),
        "{reason}: a rejected preparation wrote the destination"
    );
    assert!(
        res.giant_steps
            .iter()
            .flat_map(|gs| &gs.diagonals)
            .zip(want.giant_steps.iter().flat_map(|gs| &gs.diagonals))
            .all(|(have, want)| have.plaintext.cnv() == want.plaintext.cnv()),
        "{reason}: a rejected preparation wrote the destination"
    );
}

/// A transform whose diagonals do not share one scale or slot kind is refused by its preparation as `InvalidOperand`,
/// before any write and any scratch, with the destination unchanged.
pub fn test_prepare_linear_transformation_rhs_rejects_mixed_diagonals<BE, F, E>(
    params: CKKSTestParams,
    module: &Module<BE>,
    _host_module: &Module<HostBytesBackend>,
) where
    BE: TestContextBackend<Ring = Standard>,
    for<'a> <BE as poulpy_hal::layouts::Backend>::BufRef<'a>: poulpy_hal::layouts::HostDataRef,
    for<'a> <BE as poulpy_hal::layouts::Backend>::BufMut<'a>: poulpy_hal::layouts::HostDataMut,
    Module<BE>: TestContextModule<BE> + CKKSEncodingOps<BE, F> + CKKSLinearTransformationOps<BE> + CnvPVecAlloc<BE>,
    F: TestScalar,
{
    let m = params.n / 2;
    let mut scratch = alloc_scratch(&params, module);
    let indices = [0, 1, 2, 3, 4, 5];
    let lt = encode_lt(
        module,
        &params,
        &complex_diagonals::<F>(&indices, m, 0),
        2,
        false,
        &mut scratch.borrow(),
    );
    let want = prepare_lt(module, &lt);
    let mut res = prepare_lt(module, &lt);
    for case in ["scale", "slot kind"] {
        // Other values on the same schedule, so a write would show in the destination.
        let mut mixed = encode_lt(
            module,
            &params,
            &complex_diagonals::<F>(&indices, m, 1),
            2,
            false,
            &mut scratch.borrow(),
        );
        let last = &mut mixed.giant_steps.last_mut().unwrap().diagonals.last_mut().unwrap().plaintext;
        if case == "scale" {
            let mut meta = last.meta();
            meta.scale = Scale::Log(last.log_delta() + 1);
            last.set_meta(meta);
        } else {
            last.set_slots(SlotsKind::Real);
        }
        assert_prepare_refused(module, &mut res, &want, &mixed, "not uniform");
    }
}

/// An arena one byte below the preparation's query is refused as `CKKSError::Internal` before any write, the
/// destination, prepared from other values, unchanged.
pub fn test_prepare_linear_transformation_rhs_rejects_short_scratch<BE, F, E>(
    params: CKKSTestParams,
    module: &Module<BE>,
    _host_module: &Module<HostBytesBackend>,
) where
    BE: TestContextBackend<Ring = Standard>,
    for<'a> <BE as poulpy_hal::layouts::Backend>::BufRef<'a>: poulpy_hal::layouts::HostDataRef,
    for<'a> <BE as poulpy_hal::layouts::Backend>::BufMut<'a>: poulpy_hal::layouts::HostDataMut,
    Module<BE>: TestContextModule<BE> + CKKSEncodingOps<BE, F> + CKKSLinearTransformationOps<BE> + CnvPVecAlloc<BE>,
    F: TestScalar,
{
    let m = params.n / 2;
    let mut scratch = alloc_scratch(&params, module);
    let indices = [0, 1, 2, 3, 4, 5];
    let mut encode = |offset| {
        encode_lt(
            module,
            &params,
            &complex_diagonals::<F>(&indices, m, offset),
            2,
            false,
            &mut scratch.borrow(),
        )
    };
    let (lt, other) = (encode(0), encode(1));
    let want = prepare_lt(module, &other);
    let mut res = prepare_lt(module, &other);
    let bytes = module.ckks_prepare_linear_transformation_rhs_tmp_bytes(lt.first_diagonal_plaintext().unwrap());
    if bytes == 0 {
        eprintln!("test_prepare_linear_transformation_rhs_rejects_short_scratch: the preparation takes no scratch here");
        return;
    }
    let err = with_scratch::<BE, _>(bytes - 1, |s| module.ckks_prepare_linear_transformation_rhs(&mut res, &lt, s)).unwrap_err();
    assert_ckks_internal(
        "one byte short",
        &err,
        "ckks_prepare_linear_transformation_rhs: scratch below its query",
    );
    assert_prepared_unchanged(&res, &want, "one byte short");
}

/// A destination allocated for another schedule or diagonal layout is refused by the preparation as `InvalidOperand`,
/// before any write and any scratch, with the destination unchanged.
pub fn test_prepare_linear_transformation_rhs_rejects_foreign_layout<BE, F, E>(
    params: CKKSTestParams,
    module: &Module<BE>,
    _host_module: &Module<HostBytesBackend>,
) where
    BE: TestContextBackend<Ring = Standard>,
    for<'a> <BE as poulpy_hal::layouts::Backend>::BufRef<'a>: poulpy_hal::layouts::HostDataRef,
    for<'a> <BE as poulpy_hal::layouts::Backend>::BufMut<'a>: poulpy_hal::layouts::HostDataMut,
    Module<BE>: TestContextModule<BE> + CKKSEncodingOps<BE, F> + CKKSLinearTransformationOps<BE> + CnvPVecAlloc<BE>,
    F: TestScalar,
{
    let m = params.n / 2;
    let mut scratch = alloc_scratch(&params, module);
    let indices = [0, 1, 2, 3, 4, 5];
    let lt = encode_lt(
        module,
        &params,
        &complex_diagonals::<F>(&indices, m, 0),
        2,
        false,
        &mut scratch.borrow(),
    );
    let want = prepare_lt(module, &lt);
    let mut res = prepare_lt(module, &lt);
    // Other values, so a write would show in the destination.
    let other = complex_diagonals::<F>(&indices, m, 1);
    let other_schedule = encode_lt(module, &params, &other, 4, false, &mut scratch.borrow());
    assert_prepare_refused(module, &mut res, &want, &other_schedule, "schedule");
    let other_radix = crate::encoding::ckks_encode_linear_transformation_from_diagonals(
        module,
        (params.base2k - 1).into(),
        params.prec().into(),
        &other,
        LinearTransformationStrategy::Bsgs { giant_step: 2 },
        false,
        &mut scratch.borrow(),
    )
    .unwrap();
    assert_prepare_refused(module, &mut res, &want, &other_radix, "diagonal's layout");
}

/// A prepared pair sharing one schedule and diagonal layout is refused by the preparation as `InvalidOperand` before
/// any write and any scratch when it holds no diagonal, or when its baby steps do not start with the identity rotation.
pub fn test_prepare_linear_transformation_rhs_rejects_malformed_schedule<BE, F, E>(
    params: CKKSTestParams,
    module: &Module<BE>,
    _host_module: &Module<HostBytesBackend>,
) where
    BE: TestContextBackend<Ring = Standard>,
    for<'a> <BE as poulpy_hal::layouts::Backend>::BufRef<'a>: poulpy_hal::layouts::HostDataRef,
    for<'a> <BE as poulpy_hal::layouts::Backend>::BufMut<'a>: poulpy_hal::layouts::HostDataMut,
    Module<BE>: TestContextModule<BE> + CKKSEncodingOps<BE, F> + CKKSLinearTransformationOps<BE> + CnvPVecAlloc<BE>,
    F: TestScalar,
{
    const OP: &str = "ckks_prepare_linear_transformation_rhs";
    let refused = |res: &mut LinearTransformationPrepared<BE>,
                   want: &LinearTransformationPrepared<BE>,
                   lt: &LinearTransformation<CKKSPlaintextOwned<BE>>,
                   reason: &str| {
        let err = with_scratch::<BE, _>(0, |s| module.ckks_prepare_linear_transformation_rhs(res, lt, s)).unwrap_err();
        assert_ckks_error(
            reason,
            &err,
            CKKSCompositionError::InvalidOperand {
                op: OP,
                reason: reason.into(),
            },
        );
        assert_prepared_unchanged(res, want, reason);
    };

    let empty_prepared = || LinearTransformationPrepared::<BE> {
        baby_steps: vec![0],
        giant_steps: Vec::new(),
        rotations: Default::default(),
    };
    let empty = LinearTransformation::<CKKSPlaintextOwned<BE>> {
        baby_steps: vec![0],
        giant_steps: Vec::new(),
        rotations: Default::default(),
    };
    let (mut res, want) = (empty_prepared(), empty_prepared());
    refused(&mut res, &want, &empty, "the prepared transformation has no diagonal");

    let m = params.n / 2;
    let mut scratch = alloc_scratch(&params, module);
    let indices = [0, 1, 2, 3, 4, 5];
    let encode = |offset, scratch: &mut ScratchArena<'_, BE>| {
        encode_lt(
            module,
            &params,
            &complex_diagonals::<F>(&indices, m, offset),
            2,
            false,
            scratch,
        )
    };
    let want = prepare_lt(module, &encode(0, &mut scratch.borrow()));
    let mut res = prepare_lt(module, &encode(0, &mut scratch.borrow()));
    // Other values on the same schedule, the baby steps of both reordered alike, so only their first entry is wrong.
    let mut lt = encode(1, &mut scratch.borrow());
    lt.baby_steps.rotate_left(1);
    res.baby_steps.rotate_left(1);
    refused(&mut res, &want, &lt, "the baby steps do not start with the identity rotation");
}

/// Both prepared sides of a transformation, the right side of prepared diagonals and the left baby-step cache, under
/// `PrepareHint::Reuse`, the only hint their allocations take: an allocation holds the bytes the convolution queries
/// give, preparing into one already prepared from other values gives the bits of preparing into a fresh one, and a
/// `0xA5`-poisoned refill gives them again without reallocating. A right side of another schedule or radix is refused
/// with `InvalidOperand`, and a left side missing a key with `MissingAutomorphismKey`, on an empty arena, bytes and
/// infos unchanged.
pub fn test_prepare_linear_transformation_rhs_lifecycle<BE, F, E>(
    params: CKKSTestParams,
    module: &Module<BE>,
    host_module: &Module<HostBytesBackend>,
) where
    BE: TestContextBackend<Ring = Standard>,
    for<'a> <BE as poulpy_hal::layouts::Backend>::BufRef<'a>: poulpy_hal::layouts::HostDataRef,
    for<'a> <BE as poulpy_hal::layouts::Backend>::BufMut<'a>: poulpy_hal::layouts::HostDataMut,
    Module<BE>: TestContextModule<BE> + CKKSEncodingOps<BE, F> + CKKSLinearTransformationOps<BE> + CnvPVecAlloc<BE>,
    F: TestScalar,
    E: NegacyclicFFT<F> + NegacyclicFFTNew<F>,
{
    let m = params.n / 2;
    let mut scratch = alloc_scratch(&params, module);
    let indices = [0, 1, 2, 3, 4, 5];
    let lt = encode_lt(
        module,
        &params,
        &complex_diagonals::<F>(&indices, m, 0),
        2,
        false,
        &mut scratch.borrow(),
    );
    let other = encode_lt(
        module,
        &params,
        &complex_diagonals::<F>(&indices, m, 1),
        2,
        false,
        &mut scratch.borrow(),
    );
    let first = lt.first_diagonal_plaintext().unwrap();
    let rhs_bytes = module.ckks_prepare_linear_transformation_rhs_tmp_bytes(first);

    let buffers = |p: &LinearTransformationPrepared<BE>| -> Vec<Vec<u8>> {
        p.giant_steps
            .iter()
            .flat_map(|gs| &gs.diagonals)
            .map(|d| d.plaintext.cnv().data().to_vec())
            .collect()
    };
    let addresses = |p: &LinearTransformationPrepared<BE>| -> Vec<*const u8> {
        p.giant_steps
            .iter()
            .flat_map(|gs| &gs.diagonals)
            .map(|d| d.plaintext.cnv().data().as_ptr())
            .collect()
    };
    let infos = |p: &LinearTransformationPrepared<BE>| -> (LinearTransformationPlan, Vec<GLWEPlaintextMeta>) {
        let stashes = p
            .giant_steps
            .iter()
            .flat_map(|gs| &gs.diagonals)
            .map(|d| d.plaintext.plaintext_meta())
            .collect();
        (p.index(), stashes)
    };

    let fresh = prepare_lt(module, &lt);
    let mut res = LinearTransformationPrepared::<BE>::alloc_prepared_from_index(module, &lt.index(), first);
    let diagonal_count: usize = lt.index().index.iter().map(Vec::len).sum();
    assert_eq!(
        buffers(&res).iter().map(Vec::len).sum::<usize>(),
        diagonal_count * module.bytes_of_cnv_pvec_right(first.n().as_usize(), 1, first.size(), PrepareHint::Reuse),
        "right side: bytes_of"
    );
    for lt in [&other, &lt] {
        with_scratch::<BE, _>(rhs_bytes, |s| module.ckks_prepare_linear_transformation_rhs(&mut res, lt, s)).unwrap();
    }
    assert!(
        buffers(&res) == buffers(&fresh),
        "right side: prepare-into over other values differs from a fresh preparation"
    );
    assert_eq!(infos(&res), infos(&fresh), "right side: prepare-into infos differ");
    let before = addresses(&res);
    for diagonal in res.giant_steps.iter_mut().flat_map(|gs| &mut gs.diagonals) {
        diagonal.plaintext.cnv_mut().data_mut().as_mut().fill(0xA5);
    }
    with_scratch::<BE, _>(rhs_bytes, |s| module.ckks_prepare_linear_transformation_rhs(&mut res, &lt, s)).unwrap();
    assert_eq!(addresses(&res), before, "right side: the refill reallocated");
    assert!(
        buffers(&res) == buffers(&fresh),
        "right side: the refill differs from prepare"
    );

    let other_schedule = encode_lt(
        module,
        &params,
        &complex_diagonals::<F>(&indices, m, 1),
        4,
        false,
        &mut scratch.borrow(),
    );
    let other_radix = crate::encoding::ckks_encode_linear_transformation_from_diagonals(
        module,
        (params.base2k - 1).into(),
        params.prec().into(),
        &complex_diagonals::<F>(&indices, m, 1),
        LinearTransformationStrategy::Bsgs { giant_step: 2 },
        false,
        &mut scratch.borrow(),
    )
    .unwrap();
    for (foreign, reason) in [
        (
            &other_schedule,
            "the transformation's schedule differs from the prepared one's",
        ),
        (&other_radix, "a diagonal's layout differs from the prepared transformation's"),
    ] {
        let err = with_scratch::<BE, _>(0, |s| module.ckks_prepare_linear_transformation_rhs(&mut res, foreign, s)).unwrap_err();
        assert_ckks_error(
            reason,
            &err,
            CKKSCompositionError::InvalidOperand {
                op: "ckks_prepare_linear_transformation_rhs",
                reason: reason.into(),
            },
        );
        assert!(
            buffers(&res) == buffers(&fresh),
            "{reason}: a refused preparation changed the bytes"
        );
        assert_eq!(
            infos(&res),
            infos(&fresh),
            "{reason}: a refused preparation changed the infos"
        );
    }

    // Left side: the baby-step cache of a ciphertext.
    let encoder = ReferenceEncoder::<E>::new(m).unwrap();
    let (sk_raw, sk) = gen_sk_with_raw(&params, module, host_module, [0u8; 32]);
    let mut keys = HashMap::new();
    for p in lt.galois_elements(module.cyclotomic_order()) {
        keys.entry(p)
            .or_insert_with(|| gen_atk(&params, module, p, &sk_raw, &mut scratch.borrow()));
    }
    let (re1, im1) = test_vector_1::<F>(m);
    let (re2, im2) = test_vector_2::<F>(m);
    let ct = ckks_encrypt(
        &params,
        module,
        host_module,
        &encoder,
        &sk,
        params.k,
        &re1,
        &im1,
        &mut scratch.borrow(),
    );
    let ct_other = ckks_encrypt(
        &params,
        module,
        host_module,
        &encoder,
        &sk,
        params.k,
        &re2,
        &im2,
        &mut scratch.borrow(),
    );
    let baby_bytes = module.ckks_prepare_linear_transformation_baby_steps_tmp_bytes(&ct, &lt.index(), &params.atk_layout());
    let baby_buffers = |b: &LinearTransformationBabySteps<BE>| -> Vec<Vec<u8>> {
        b.baby_steps().map(|rot| b.baby_step(rot).data().to_vec()).collect()
    };
    let baby_addresses = |b: &LinearTransformationBabySteps<BE>| -> Vec<*const u8> {
        b.baby_steps().map(|rot| b.baby_step(rot).data().as_ptr()).collect()
    };
    let mut fresh = LinearTransformationBabySteps::alloc_for(module, &lt, &ct);
    with_scratch::<BE, _>(baby_bytes, |s| {
        module.ckks_prepare_linear_transformation_baby_steps(&mut fresh, &ct, &keys, s)
    })
    .unwrap();
    let mut babies = LinearTransformationBabySteps::alloc_for(module, &lt, &ct);
    assert_eq!(
        baby_buffers(&babies).iter().map(Vec::len).sum::<usize>(),
        babies.len() * module.bytes_of_cnv_pvec_left(ct.n().as_usize(), ct.rank().as_usize() + 1, ct.size(), PrepareHint::Reuse),
        "left side: bytes_of"
    );
    for src in [&ct_other, &ct] {
        with_scratch::<BE, _>(baby_bytes, |s| {
            module.ckks_prepare_linear_transformation_baby_steps(&mut babies, src, &keys, s)
        })
        .unwrap();
    }
    assert!(
        baby_buffers(&babies) == baby_buffers(&fresh),
        "left side: prepare-into over other values differs from a fresh preparation"
    );
    let before = baby_addresses(&babies);
    for (_, mut operand) in babies.baby_steps_mut() {
        operand.data_mut().as_mut().fill(0xA5);
    }
    with_scratch::<BE, _>(baby_bytes, |s| {
        module.ckks_prepare_linear_transformation_baby_steps(&mut babies, &ct, &keys, s)
    })
    .unwrap();
    assert_eq!(baby_addresses(&babies), before, "left side: the refill reallocated");
    assert!(
        baby_buffers(&babies) == baby_buffers(&fresh),
        "left side: the refill differs from prepare"
    );
    let no_keys: HashMap<i64, GLWEAutomorphismKeyPrepared<BE::OwnedBuf, BE>> = HashMap::new();
    let err = with_scratch::<BE, _>(0, |s| {
        module.ckks_prepare_linear_transformation_baby_steps(&mut babies, &ct_other, &no_keys, s)
    })
    .unwrap_err();
    assert!(
        matches!(
            err.composition(),
            Some(CKKSCompositionError::MissingAutomorphismKey {
                op: "ckks_prepare_linear_transformation_baby_steps",
                ..
            })
        ),
        "left side without keys: unexpected error: {err}"
    );
    assert!(
        baby_buffers(&babies) == baby_buffers(&fresh),
        "left side: a refused preparation changed the bytes"
    );
}

/// The stored keys, with the Galois elements listed in `coarse_ps` answered
/// through a coarser `dsize`: one key set carrying two different layouts.
struct MixedDsize<BE: Backend> {
    keys: HashMap<i64, GLWEAutomorphismKeyPrepared<BE::OwnedBuf, BE>>,
    coarse: TorusPrecision,
    coarse_ps: HashSet<i64>,
}

impl<BE: Backend> GetAutomorphismKey<BE> for MixedDsize<BE> {
    fn lookup_automorphism_key(
        &self,
        p: i64,
        k: TorusPrecision,
    ) -> poulpy_core::Result<GLWEAutomorphismKeyPreparedBackendRef<'_, BE>> {
        let key = self.keys.get(&p).ok_or(poulpy_core::CoreError::GGLWEKeyUse {
            op: "get_automorphism_key",
            detail: format!("no automorphism key for p={p}"),
        })?;
        let _ = k;
        if self.coarse_ps.contains(&p) {
            key.with_k_digit(self.coarse)
        } else {
            Ok(key.to_backend_ref())
        }
    }
}

impl<BE: Backend> GetTensorKey<BE> for MixedDsize<BE> {
    fn lookup_tensor_key(&self, k: TorusPrecision) -> poulpy_core::Result<GLWETensorKeyPreparedBackendRef<'_, BE>> {
        self.keys.lookup_tensor_key(k)
    }
}

/// Baby rotations resolve at the source precision, giant rotations at the
/// destination one.
///
/// The two sides of a factor operate on different values: baby steps rotate the
/// input, giant steps rotate the post-product accumulator. The destination is
/// narrower than the input here, so a giant step resolving at `ct.k()` would
/// size its product for a value that no longer exists.
pub fn test_linear_transformation_pins_operation_precisions<BE, F, E>(
    params: CKKSTestParams,
    module: &Module<BE>,
    host_module: &Module<HostBytesBackend>,
) where
    BE: TestContextBackend<Ring = Standard>,
    for<'a> <BE as poulpy_hal::layouts::Backend>::BufRef<'a>: poulpy_hal::layouts::HostDataRef,
    for<'a> <BE as poulpy_hal::layouts::Backend>::BufMut<'a>: poulpy_hal::layouts::HostDataMut,
    Module<BE>: TestContextModule<BE> + CKKSEncodingOps<BE, F> + CKKSLinearTransformationOps<BE> + CnvPVecAlloc<BE>,
    F: TestScalar,
    E: NegacyclicFFT<F> + NegacyclicFFTNew<F>,
    for<'a> ScratchArena<'a, BE>: ScratchAvailable,
{
    let m = params.n / 2;
    let encoder = ReferenceEncoder::<E>::new(m).unwrap();
    let (a_re, a_im) = test_vector_1::<F>(m);
    let (sk_raw, sk) = gen_sk_with_raw(&params, module, host_module, [0u8; 32]);
    let key_params = CKKSTestParams {
        dsize: params.dsize.max(4),
        ..params
    };
    let mut scratch = alloc_scratch(&key_params, module);

    let n1 = 2;
    let b = complex_diagonals::<F>(&[0, 1, 2, 3, 4, 5], m, 0);
    let lt = encode_lt(module, &params, &b, n1, false, &mut scratch.borrow());

    let order = module.cyclotomic_order();
    let mut keys = HashMap::new();
    for p in lt.galois_elements(order) {
        keys.entry(p)
            .or_insert_with(|| gen_atk(&key_params, module, p, &sk_raw, &mut scratch.borrow()));
    }
    let keys = QueryLog::new(keys);

    let ct = ckks_encrypt(
        &params,
        module,
        host_module,
        &encoder,
        &sk,
        params.k,
        &a_re,
        &a_im,
        &mut scratch.borrow(),
    );

    let prepared = prepare_lt(module, &lt);
    let mut res = alloc_ct(&params, module, params.k - params.base2k);
    with_scratch::<BE, _>(
        module
            .ckks_prepare_linear_transformation_baby_steps_tmp_bytes(&ct, &lt.index(), &key_params.atk_layout())
            .max(pt_eval_tmp_bytes(module, &ct, &lt.index(), &key_params.atk_layout())),
        |s| module.ckks_eval_linear_transformation_self_into(&mut res, &ct, &prepared, &keys, s),
    )
    .unwrap();
    let dst_k = res.k();

    let baby_elements: HashSet<i64> = lt
        .baby_steps()
        .iter()
        .copied()
        .filter(|&rot| rot != 0)
        .map(|rot| galois_element(rot, order))
        .collect();
    let giant_elements: HashSet<i64> = lt
        .giant_steps
        .iter()
        .filter(|gs| gs.rot != 0 && !gs.diagonals.is_empty())
        .map(|gs| galois_element(gs.rot, order))
        .collect();

    let seen = keys.take();
    assert!(!seen.is_empty(), "the factor consulted no key");
    for &(p, k) in seen.iter() {
        match (baby_elements.contains(&p), giant_elements.contains(&p)) {
            (true, false) => assert_eq!(k, ct.k(), "baby key p={p} resolved at {k}, not at the source precision"),
            (false, true) => assert_eq!(k, dst_k, "giant key p={p} resolved at {k}, not at the destination precision"),
            (true, true) => assert!(
                k == ct.k() || k == dst_k,
                "shared baby/giant key p={p} resolved at unrelated precision {k}"
            ),
            (false, false) => panic!("unexpected key lookup p={p} at k={k}"),
        }
    }
}

/// A key set carrying two layouts must fit the scratch the public query
/// promises: a caller holding several key layouts can only take the max over
/// the per-layout queries, so no route may cost more than that.
pub fn test_linear_transformation_mixed_key_layouts<BE, F, E>(
    params: CKKSTestParams,
    module: &Module<BE>,
    host_module: &Module<HostBytesBackend>,
) where
    BE: TestContextBackend<Ring = Standard>,
    for<'a> <BE as poulpy_hal::layouts::Backend>::BufRef<'a>: poulpy_hal::layouts::HostDataRef,
    for<'a> <BE as poulpy_hal::layouts::Backend>::BufMut<'a>: poulpy_hal::layouts::HostDataMut,
    Module<BE>: TestContextModule<BE> + CKKSEncodingOps<BE, F> + CKKSLinearTransformationOps<BE> + CnvPVecAlloc<BE>,
    F: TestScalar,
    E: NegacyclicFFT<F> + NegacyclicFFTNew<F>,
    for<'a> ScratchArena<'a, BE>: ScratchAvailable,
{
    let m = params.n / 2;
    let encoder = ReferenceEncoder::<E>::new(m).unwrap();
    let (a_re, a_im) = test_vector_1::<F>(m);
    let (sk_raw, sk) = gen_sk_with_raw(&params, module, host_module, [0u8; 32]);
    // Stored at the smallest `dsize` that still leaves a digit when read at
    // twice that value, so the coarsened twin below is always reachable, with
    // rows enough that this read still covers the ciphertext: it keeps one
    // stored digit as guard.
    let dsize = params.dsize.max(2);
    let k_digit = dsize * params.base2k;
    let key_params = CKKSTestParams {
        dsize,
        k: (2 * params.k.div_ceil(2 * k_digit) + 1) * k_digit,
        ..params
    };
    let mut scratch = alloc_scratch(&key_params, module);

    let n1 = 2;
    let b = complex_diagonals::<F>(&[0, 1, 2, 3, 4, 5], m, 0);
    let lt = encode_lt(module, &params, &b, n1, false, &mut scratch.borrow());

    // Half the rotations answer through twice their stored `dsize`, so baby and
    // giant steps both see a key set that is not of one shape.
    let order = module.cyclotomic_order();
    let mut keys = HashMap::new();
    for p in lt.galois_elements(order) {
        keys.entry(p)
            .or_insert_with(|| gen_atk(&key_params, module, p, &sk_raw, &mut scratch.borrow()));
    }
    let mut elements: Vec<i64> = keys.keys().copied().collect();
    elements.sort_unstable();
    let coarse_ps: HashSet<i64> = elements.iter().copied().step_by(2).collect();
    let stored_infos = key_params.atk_layout();
    let coarse_k_digit = k_digit_of(key_params.base2k, 2 * key_params.dsize);
    let coarse_infos = stored_infos.gglwe_layout().at_k_digit(coarse_k_digit).unwrap();
    let keys = MixedDsize::<BE> {
        keys,
        coarse: coarse_k_digit,
        coarse_ps,
    };

    let ct = ckks_encrypt(
        &params,
        module,
        host_module,
        &encoder,
        &sk,
        params.k,
        &a_re,
        &a_im,
        &mut scratch.borrow(),
    );
    let prepared = prepare_lt(module, &lt);
    let mut res = alloc_ct(&params, module, params.k - params.base2k);

    // One call reading both layouts takes the larger of the baby-step and evaluation queries at each.
    let eval_bytes = module
        .ckks_prepare_linear_transformation_baby_steps_tmp_bytes(&ct, &prepared.index(), &stored_infos)
        .max(module.ckks_prepare_linear_transformation_baby_steps_tmp_bytes(&ct, &prepared.index(), &coarse_infos))
        .max(pt_eval_tmp_bytes(module, &ct, &prepared.index(), &stored_infos))
        .max(pt_eval_tmp_bytes(module, &ct, &prepared.index(), &coarse_infos));
    with_scratch::<BE, _>(eval_bytes, |s| {
        module.ckks_eval_linear_transformation_self_into(&mut res, &ct, &prepared, &keys, s)
    })
    .unwrap();
}

/// The secret side of the tests of ciphertext diagonals: encrypts sources and diagonals, and checks results.
struct Secret<'a, BE: Backend, E> {
    params: &'a CKKSTestParams,
    module: &'a Module<BE>,
    host_module: &'a Module<HostBytesBackend>,
    encoder: ReferenceEncoder<E>,
    sk: GLWESecretPrepared<BE::OwnedBuf, BE>,
    sk_raw: BackendGLWESecret<BE>,
}

impl<'a, BE, E> Secret<'a, BE, E>
where
    BE: TestContextBackend<Ring = Standard>,
    for<'b> <BE as Backend>::BufRef<'b>: poulpy_hal::layouts::HostDataRef,
    for<'b> <BE as Backend>::BufMut<'b>: poulpy_hal::layouts::HostDataMut,
    Module<BE>: TestContextModule<BE>,
{
    fn new<F: TestScalar>(params: &'a CKKSTestParams, module: &'a Module<BE>, host_module: &'a Module<HostBytesBackend>) -> Self
    where
        E: NegacyclicFFTNew<F>,
    {
        let (sk_raw, sk) = gen_sk_with_raw(params, module, host_module, [0u8; 32]);
        Self {
            params,
            module,
            host_module,
            encoder: ReferenceEncoder::new(params.n / 2).unwrap(),
            sk,
            sk_raw,
        }
    }

    /// `(re, im)` encrypted at `k` bits, labelled `log_sparsity`.
    fn encrypt<F: TestScalar>(
        &self,
        re: &[F],
        im: &[F],
        k: usize,
        log_sparsity: usize,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> CKKSCiphertextOwned<BE>
    where
        E: NegacyclicFFT<F>,
    {
        let mut ct = ckks_encrypt(
            self.params,
            self.module,
            self.host_module,
            &self.encoder,
            &self.sk,
            k,
            re,
            im,
            scratch,
        );
        ct.set_log_sparsity(log_sparsity);
        ct
    }

    /// Asserts that `dst`, of `dst_k` bits before the call, is `Σ_t M_t·srcs[t]` for transforms of the ciphertext
    /// diagonals `diagonals[t]`: the rule of `ckks_mul_into` between a source and a diagonal, the slots of every
    /// operand joined, the smallest of their sparsities, and a decryption to `want`.
    #[allow(clippy::too_many_arguments)]
    fn assert_transform<F: TestScalar>(
        &self,
        label: &str,
        dst: &CKKSCiphertextOwned<BE>,
        dst_k: usize,
        srcs: &[&CKKSCiphertextOwned<BE>],
        diagonals: &[&CKKSCiphertextOwned<BE>],
        (want_re, want_im): &(Vec<F>, Vec<F>),
        scratch: &mut ScratchArena<'_, BE>,
    ) where
        E: NegacyclicFFT<F>,
    {
        assert_mul_ct_output_meta(label, dst, dst_k, srcs[0], diagonals[0]);
        let operands = || srcs.iter().chain(diagonals);
        assert_eq!(
            (dst.slots(), dst.log_sparsity()),
            (
                operands().map(|ct| ct.slots()).reduce(SlotsKind::join).unwrap(),
                operands().map(|ct| ct.log_sparsity()).min().unwrap()
            ),
            "{label}: slots and sparsity"
        );
        assert_decrypt_precision(
            label,
            self.params,
            self.module,
            &self.encoder,
            dst,
            &self.sk,
            want_re,
            want_im,
            scratch,
        );
    }
}

/// The baby cache of `src` for `lt`, prepared on its query.
fn prepare_babies<BE, P, H>(
    module: &Module<BE>,
    params: &CKKSTestParams,
    lt: &LinearTransformation<P>,
    src: &CKKSCiphertextOwned<BE>,
    keys: &H,
) -> LinearTransformationBabySteps<BE>
where
    BE: TestContextBackend<Ring = Standard>,
    Module<BE>: TestContextModule<BE> + CKKSLinearTransformationOps<BE> + CnvPVecAlloc<BE>,
    H: GetAutomorphismKey<BE>,
{
    let mut babies = LinearTransformationBabySteps::alloc_for(module, lt, src);
    with_scratch::<BE, _>(
        module.ckks_prepare_linear_transformation_baby_steps_tmp_bytes(src, &lt.index(), &params.atk_layout()),
        |s| module.ckks_prepare_linear_transformation_baby_steps(&mut babies, src, keys, s),
    )
    .unwrap();
    babies
}

/// `dec(lt(enc(a), M)) ≈ B·a` for `M` the encrypted, pre-rotated diagonals of `B`, on its exact poisoned budget, with
/// diagonals narrower or wider than the source and with a tensor key the one-pass product cannot take. The `_assign` and
/// `_self_` forms give the bits of `_into`. Keys without the tensor key at the larger width are refused as
/// `MissingRelinearizationKey` before `dst` is written.
pub fn test_linear_transformation_ct<BE, F, E>(
    params: CKKSTestParams,
    module: &Module<BE>,
    host_module: &Module<HostBytesBackend>,
) where
    BE: TestContextBackend<Ring = Standard>,
    for<'a> <BE as poulpy_hal::layouts::Backend>::BufRef<'a>: poulpy_hal::layouts::HostDataRef,
    for<'a> <BE as poulpy_hal::layouts::Backend>::BufMut<'a>: poulpy_hal::layouts::HostDataMut,
    Module<BE>: TestContextModule<BE> + CKKSLinearTransformationOps<BE> + CnvPVecAlloc<BE>,
    F: TestScalar,
    E: NegacyclicFFT<F> + NegacyclicFFTNew<F>,
{
    const OP: &str = "ckks_eval_linear_transformation_into";
    let m = params.n / 2;
    let (a_re, a_im) = test_vector_1::<F>(m);
    let secret = Secret::<BE, E>::new::<F>(&params, module, host_module);
    let mut scratch = alloc_scratch(&params, module);
    let tsk = gen_tsk(&params, module, &secret.sk_raw, &mut scratch.borrow());
    // At another `base2k`, which the one-pass product cannot take: relinearized as a tensor.
    let tensor_params = CKKSTestParams {
        base2k: params.base2k - 1,
        ..params
    };
    let tensor_tsk = gen_tsk(&tensor_params, module, &secret.sk_raw, &mut scratch.borrow());
    let b = complex_diagonals::<F>(&[0, 1, 2, 3, 4, 5], m, 0);
    let order = module.cyclotomic_order();
    let narrow = params.k - params.base2k;
    let (bsgs, pow2) = (
        LinearTransformationStrategy::Bsgs { giant_step: 2 },
        LinearTransformationStrategy::BsgsPowersOfTwo { giant_step: 2 },
    );

    for (strategy, src_k, diagonal_k, tsk, tsk_layout) in [
        (bsgs, params.k, narrow, &tsk, params.tsk_layout()),
        (pow2, narrow, params.k, &tsk, params.tsk_layout()),
        (bsgs, params.k, narrow, &tensor_tsk, tensor_params.tsk_layout()),
    ] {
        let label = format!("linear_transformation_ct {strategy:?}, key base2k {}", tsk_layout.base2k());
        let src = secret.encrypt(&a_re, &a_im, src_k, 1, &mut scratch.borrow());
        let lt = b.build_transform(strategy, |re, im| {
            secret.encrypt(re, im, diagonal_k, 1, &mut scratch.borrow())
        });
        let atks: HashMap<_, _> = lt
            .galois_elements(order)
            .into_iter()
            .map(|p| (p, gen_atk(&params, module, p, &secret.sk_raw, &mut scratch.borrow())))
            .collect();
        let key_k = TorusPrecision(src_k.max(diagonal_k) as u32);
        let at_key_k = TensorKeyAt(tsk, key_k);
        let keys = TransformKeys {
            automorphism: &atks,
            tensor: &at_key_k,
        };
        let prepared = prepare_lt(module, &lt);
        let babies = prepare_babies(module, &params, &prepared, &src, &atks);
        let first = lt.first_diagonal_plaintext().unwrap();
        let plan = lt.index();
        let bytes = module.ckks_eval_linear_transformation_tmp_bytes(&src, first, &plan, &tsk_layout, &params.atk_layout());
        let baby_bytes = module.ckks_prepare_linear_transformation_baby_steps_tmp_bytes(&src, &plan, &params.atk_layout());

        let mut dst = alloc_ct(&params, module, src_k);
        with_scratch::<BE, _>(bytes, |s| {
            module.ckks_eval_linear_transformation_into(&mut dst, &src, &babies, &prepared, &keys, s)
        })
        .unwrap();
        let want = b.evaluate((a_re.as_slice(), a_im.as_slice()), strategy);
        secret.assert_transform(&label, &dst, src_k, &[&src], &[first], &want, &mut scratch.borrow());
        let have = snapshot::<BE, _>(&dst);

        let mut assigned = src.clone();
        with_scratch::<BE, _>(bytes, |s| {
            module.ckks_eval_linear_transformation_assign(&mut assigned, &babies, &prepared, &keys, s)
        })
        .unwrap();
        let mut self_into = alloc_ct(&params, module, src_k);
        with_scratch::<BE, _>(baby_bytes.max(bytes), |s| {
            module.ckks_eval_linear_transformation_self_into(&mut self_into, &src, &prepared, &keys, s)
        })
        .unwrap();
        // The documented bound of `self_assign`: a working copy of `src` plus the larger of the baby-step and
        // evaluation queries.
        let mut self_assigned = src.clone();
        with_scratch::<BE, _>(
            BE::scratch_aligned(module.glwe_bytes_of_from_infos(&src)) + baby_bytes.max(bytes),
            |s| module.ckks_eval_linear_transformation_self_assign(&mut self_assigned, &prepared, &keys, s),
        )
        .unwrap();
        // The unprepared diagonals, prepared in the call.
        let mut streamed = alloc_ct(&params, module, src_k);
        with_scratch::<BE, _>(
            module.ckks_eval_linear_transformation_streamed_tmp_bytes(&src, first, &plan, &tsk_layout, &params.atk_layout()),
            |s| module.ckks_eval_linear_transformation_into(&mut streamed, &src, &babies, &lt, &keys, s),
        )
        .unwrap();
        // Diagonals allocated wider than their width, as a copy into a larger ciphertext leaves them, read as theirs.
        let wide_lt = LinearTransformation {
            baby_steps: lt.baby_steps.clone(),
            rotations: lt.rotations,
            giant_steps: lt
                .giant_steps
                .iter()
                .map(|gs| LinearTransformationGiantStep {
                    rot: gs.rot,
                    diagonals: gs
                        .diagonals
                        .iter()
                        .map(|d| {
                            let mut wide = alloc_ct(&params, module, diagonal_k + params.base2k);
                            with_scratch::<BE, _>(module.ckks_copy_tmp_bytes(&wide, &d.plaintext), |s| {
                                module.ckks_copy(&mut wide, &d.plaintext, s)
                            })
                            .unwrap();
                            LinearTransformationDiagonal {
                                baby: d.baby,
                                plaintext: wide,
                            }
                        })
                        .collect(),
                })
                .collect(),
        };
        let wide_prepared = prepare_lt(module, &wide_lt);
        let mut wide_resident = alloc_ct(&params, module, src_k);
        with_scratch::<BE, _>(bytes, |s| {
            module.ckks_eval_linear_transformation_into(&mut wide_resident, &src, &babies, &wide_prepared, &keys, s)
        })
        .unwrap();
        let mut wide_streamed = alloc_ct(&params, module, src_k);
        with_scratch::<BE, _>(
            module.ckks_eval_linear_transformation_streamed_tmp_bytes(&src, first, &plan, &tsk_layout, &params.atk_layout()),
            |s| module.ckks_eval_linear_transformation_into(&mut wide_streamed, &src, &babies, &wide_lt, &keys, s),
        )
        .unwrap();
        for (form, out) in [
            ("_assign", &assigned),
            ("_self_into", &self_into),
            ("_self_assign", &self_assigned),
            ("streamed _into", &streamed),
            ("wide diagonals _into", &wide_resident),
            ("wide streamed diagonals _into", &wide_streamed),
        ] {
            assert_eq!(snapshot::<BE, _>(out), have, "{label}: {form} differs from _into");
        }

        for (case, k) in [("no tensor key", 1), ("smaller width", src_k.min(diagonal_k))] {
            let at_k = TensorKeyAt(tsk, TorusPrecision(k as u32));
            let keys = TransformKeys {
                automorphism: &atks,
                tensor: &at_k,
            };
            let err = with_scratch::<BE, _>(0, |s| {
                module.ckks_eval_linear_transformation_into(&mut dst, &src, &babies, &prepared, &keys, s)
            })
            .unwrap_err();
            assert!(
                matches!(
                    err.composition(),
                    Some(CKKSCompositionError::MissingRelinearizationKey { op, k, .. }) if *op == OP && *k == key_k.as_u32()
                ),
                "{label}, {case}: unexpected error: {err}"
            );
            assert_eq!(
                snapshot::<BE, _>(&dst),
                have,
                "{label}, {case}: a refused call wrote into dst"
            );
        }
    }
}

/// The summed transform of ciphertext diagonals, `Σ_t M_t·a_t`, over terms sharing one schedule, on its exact poisoned
/// budget: one term is the single transform bit for bit, two decrypt to the sum of the single transforms.
pub fn test_linear_transformation_sum_ct<BE, F, E>(
    params: CKKSTestParams,
    module: &Module<BE>,
    host_module: &Module<HostBytesBackend>,
) where
    BE: TestContextBackend<Ring = Standard>,
    for<'a> <BE as poulpy_hal::layouts::Backend>::BufRef<'a>: poulpy_hal::layouts::HostDataRef,
    for<'a> <BE as poulpy_hal::layouts::Backend>::BufMut<'a>: poulpy_hal::layouts::HostDataMut,
    Module<BE>: TestContextModule<BE> + CKKSLinearTransformationOps<BE> + CnvPVecAlloc<BE>,
    F: TestScalar,
    E: NegacyclicFFT<F> + NegacyclicFFTNew<F>,
{
    let m = params.n / 2;
    let secret = Secret::<BE, E>::new::<F>(&params, module, host_module);
    let mut scratch = alloc_scratch(&params, module);
    let tsk = gen_tsk(&params, module, &secret.sk_raw, &mut scratch.borrow());

    // Two matrices on the same diagonals, hence one schedule, with distinct values.
    let strategy = LinearTransformationStrategy::Bsgs { giant_step: 2 };
    let bs: Vec<_> = (0..2).map(|t| complex_diagonals::<F>(&[0, 1, 2, 3, 4, 5], m, t)).collect();
    let inputs = [test_vector_1::<F>(m), test_vector_2::<F>(m)];
    let srcs: Vec<_> = inputs
        .iter()
        .map(|(re, im)| secret.encrypt(re, im, params.k, 1, &mut scratch.borrow()))
        .collect();
    let lts: Vec<_> = bs
        .iter()
        .zip([1, 0])
        .map(|(b, log_sparsity)| {
            b.build_transform(strategy, |re, im| {
                secret.encrypt(re, im, params.k, log_sparsity, &mut scratch.borrow())
            })
        })
        .collect();
    let atks: HashMap<_, _> = lts[0]
        .galois_elements(module.cyclotomic_order())
        .into_iter()
        .map(|p| (p, gen_atk(&params, module, p, &secret.sk_raw, &mut scratch.borrow())))
        .collect();
    let keys = TransformKeys {
        automorphism: &atks,
        tensor: &tsk,
    };
    let prepared: Vec<_> = lts.iter().map(|lt| prepare_lt(module, lt)).collect();
    let babies: Vec<_> = srcs
        .iter()
        .map(|src| prepare_babies(module, &params, &prepared[0], src, &atks))
        .collect();
    let diagonals: Vec<_> = lts.iter().map(|lt| lt.first_diagonal_plaintext().unwrap()).collect();
    let plan = lts[0].index();

    let mut single = alloc_ct(&params, module, params.k);
    with_scratch::<BE, _>(
        module.ckks_eval_linear_transformation_tmp_bytes(
            &srcs[0],
            diagonals[0],
            &plan,
            &params.tsk_layout(),
            &params.atk_layout(),
        ),
        |s| module.ckks_eval_linear_transformation_into(&mut single, &srcs[0], &babies[0], &prepared[0], &keys, s),
    )
    .unwrap();
    let sum_bytes = module.ckks_eval_linear_transformation_sum_tmp_bytes(
        &srcs[0],
        diagonals[0],
        &plan,
        &params.tsk_layout(),
        &params.atk_layout(),
    );
    let mut one = alloc_ct(&params, module, params.k);
    with_scratch::<BE, _>(sum_bytes, |s| {
        module.ckks_eval_linear_transformation_sum_into(&mut one, &[&srcs[0]], &[&babies[0]], &[&prepared[0]], &keys, s)
    })
    .unwrap();
    assert_eq!(
        snapshot::<BE, _>(&one),
        snapshot::<BE, _>(&single),
        "a one-term sum is not the single transform"
    );

    let mut sum = alloc_ct(&params, module, params.k);
    with_scratch::<BE, _>(sum_bytes, |s| {
        module.ckks_eval_linear_transformation_sum_into(
            &mut sum,
            &[&srcs[0], &srcs[1]],
            &[&babies[0], &babies[1]],
            &[&prepared[0], &prepared[1]],
            &keys,
            s,
        )
    })
    .unwrap();
    let want = inputs
        .iter()
        .zip(&bs)
        .map(|((re, im), b)| b.evaluate((re.as_slice(), im.as_slice()), strategy))
        .reduce(|(re, im), (re_t, im_t)| want_add(&re, &im, &re_t, &im_t))
        .unwrap();
    secret.assert_transform(
        "linear_transformation_sum_ct",
        &sum,
        params.k,
        &[&srcs[0], &srcs[1]],
        &diagonals,
        &want,
        &mut scratch.borrow(),
    );
}

/// A transform of ciphertext diagonals is refused before any write or key read: a destination of another degree,
/// `base2k` or rank than the diagonals, `base2k` past 62, a baby cache of another width than the source, and the
/// preparation of ciphertext diagonals into a cache allocated for plaintext ones.
pub fn test_linear_transformation_ct_rejects_foreign_layouts<BE, F, E>(
    params: CKKSTestParams,
    module: &Module<BE>,
    _host_module: &Module<HostBytesBackend>,
) where
    BE: TestContextBackend<Ring = Standard>,
    for<'a> <BE as poulpy_hal::layouts::Backend>::BufRef<'a>: poulpy_hal::layouts::HostDataRef,
    for<'a> <BE as poulpy_hal::layouts::Backend>::BufMut<'a>: poulpy_hal::layouts::HostDataMut,
    Module<BE>: TestContextModule<BE> + CKKSLinearTransformationOps<BE> + CnvPVecAlloc<BE>,
    F: TestScalar,
{
    const OP: &str = "ckks_eval_linear_transformation_into";
    let p = &params;
    let lt = complex_diagonals::<F>(&[0, 1, 2, 3, 4, 5], p.n / 2, 0)
        .build_transform(LinearTransformationStrategy::Bsgs { giant_step: 2 }, |_, _| {
            alloc_ct(p, module, p.k)
        });
    let plan = lt.index();
    let prepared =
        LinearTransformationPrepared::<BE>::alloc_prepared_from_index(module, &plan, lt.first_diagonal_plaintext().unwrap());
    let src = alloc_ct(p, module, p.k);
    let babies = LinearTransformationBabySteps::alloc_for(module, &prepared, &src);
    let no_keys: HashMap<i64, GLWEAutomorphismKeyPrepared<BE::OwnedBuf, BE>> = HashMap::new();
    let refused = |dst: &mut CKKSCiphertextOwned<BE>, babies: &_, lt: &LinearTransformationPrepared<BE>| {
        let before = snapshot::<BE, _>(&*dst);
        let err = with_scratch::<BE, _>(0, |s| {
            module.ckks_eval_linear_transformation_into(dst, &src, babies, lt, &no_keys, s)
        })
        .unwrap_err();
        assert_eq!(snapshot::<BE, _>(&*dst), before, "a refused transform wrote dst: {err}");
        err
    };

    for layout in [
        GLWELayout {
            n: (p.n / 2).into(),
            ..p.glwe_layout()
        },
        GLWELayout {
            base2k: (p.base2k - 1).into(),
            ..p.glwe_layout()
        },
        GLWELayout {
            rank: Rank((p.rank % 2 + 1) as u32),
            ..p.glwe_layout()
        },
    ] {
        let err = refused(&mut module.ckks_ciphertext_alloc_from_glwe_infos(&layout), &babies, &prepared);
        assert_ckks_error(
            &format!("{layout:?}"),
            &err,
            CKKSCompositionError::PreparedOperandLayoutMismatch {
                op: OP,
                dst_n: layout.n.as_usize(),
                dst_base2k: layout.base2k.as_usize(),
                dst_rank: layout.rank.as_usize(),
                prep_n: p.n,
                prep_base2k: p.base2k,
                prep_rank: p.rank,
            },
        );
    }

    let narrow = LinearTransformationBabySteps::alloc_for(module, &prepared, &alloc_ct(p, module, p.k - p.base2k));
    let err = refused(&mut alloc_ct(p, module, p.k), &narrow, &prepared);
    assert_invalid_operand(
        "baby cache width",
        &err,
        OP,
        "the baby cache was prepared from a source of another width",
    );

    let wide = GLWELayout {
        base2k: 63_usize.into(),
        k: (2 * 63_usize).into(),
        ..p.glwe_layout()
    };
    let wide_lt = LinearTransformationPrepared::<BE>::alloc_prepared_from_index(module, &plan, &wide);
    let err = refused(&mut module.ckks_ciphertext_alloc_from_glwe_infos(&wide), &babies, &wide_lt);
    assert_invalid_operand("base2k 63", &err, OP, "base2k 63 > 62");

    let proxy = crate::api::ckks_plaintext_diagonal_layout(&src);
    let want = LinearTransformationPrepared::<BE>::alloc_prepared_from_index(module, &plan, &proxy);
    let mut res = LinearTransformationPrepared::<BE>::alloc_prepared_from_index(module, &plan, &proxy);
    assert_prepare_refused(
        module,
        &mut res,
        &want,
        &lt,
        "a diagonal's layout differs from the prepared transformation's",
    );
}
