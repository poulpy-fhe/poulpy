//! Linear transformation (BSGS) test.
//!
//! Minimal correctness check: `dec(lt(enc(a), B)) ≈ B·a`, where `a` is the
//! encrypted complex slot vector and `B` is a complex matrix supplied as a raw
//! diagonal map. The transform applies the matrix on the *left*
//! (`(B·a)[j] = Σ_k B[j][k]·a[k]`); the expected value is computed by the
//! scheme-agnostic plaintext evaluator [`ComplexDiagonals`] / [`Evaluate`],
//! which the homomorphic engine must match up to CKKS precision.

use crate::api::CKKSEncodingOps;
use std::cell::RefCell;
use std::collections::{HashMap, HashSet};

use poulpy_core::layouts::{
    Diagonals, Evaluate, GGLWEInfos, GetAutomorphismKey, LWEInfos, LinearTransformationStrategy, SetGLWEPlaintextInfos,
    TorusPrecision, k_digit_of,
    prepared::{GLWEAutomorphismKeyPrepared, GLWEAutomorphismKeyPreparedBackendRef, GLWEAutomorphismKeyPreparedToBackendRef},
};
use poulpy_hal::{
    api::{CnvPVecAlloc, NegacyclicFFT, NegacyclicFFTNew, ScratchAvailable, ScratchOwnedBorrow},
    layouts::{Backend, CyclotomicOrder, HostBytesBackend, Module, ScratchArena, Standard, galois_element},
};

use crate::{
    CKKSCompositionError, CKKSInfos, Scale, SetCKKSInfos, SlotsKind,
    api::{CKKSLinearTransformationOps, LinearTransformation, LinearTransformationBabySteps, LinearTransformationPrepared},
    layouts::{CKKSCiphertextOwned, CKKSPlaintextOwned, ComplexDiagonals},
    test_suite::reference_encoder::ReferenceEncoder,
    test_suite::{
        CKKSTestParams,
        helpers::{
            TestContextBackend, TestContextModule, TestScalar, alloc_ct, alloc_scratch, assert_decrypt_precision, ckks_encrypt,
            gen_atk, gen_sk_with_raw, test_vector_1, test_vector_2, want_add,
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

/// Materializes the prepared (resident-RHS) form of a plaintext linear
/// transformation, so the same unified evaluator can be exercised on both the
/// resident (`P = PreparedDiagonal`) and streamed (`P = CKKSPlaintext`) paths.
fn prepare_lt<BE>(module: &Module<BE>, lt: &LinearTransformation<CKKSPlaintextOwned<BE>>) -> LinearTransformationPrepared<BE>
where
    BE: TestContextBackend<Ring = Standard>,
    for<'a> <BE as poulpy_hal::layouts::Backend>::BufRef<'a>: poulpy_hal::layouts::HostDataRef,
    for<'a> <BE as poulpy_hal::layouts::Backend>::BufMut<'a>: poulpy_hal::layouts::HostDataMut,
    Module<BE>: TestContextModule<BE> + CKKSLinearTransformationOps<BE> + CnvPVecAlloc<BE>,
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

    // transpose = false: dec(lt(enc(a), B)) ≈ B·a. Resident path (prepared RHS).
    let prepared_left = prepare_lt(module, &lt_left);
    let mut ct_left = alloc_ct(&params, module, params.k);
    with_scratch::<BE, _>(
        module.ckks_eval_linear_transformation_tmp_bytes(&ct, &key_params.atk_layout()),
        |s| module.ckks_eval_linear_transformation_self_into(&mut ct_left, &ct, &prepared_left, &atks, s),
    )
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
    with_scratch::<BE, _>(
        module.ckks_eval_linear_transformation_streamed_tmp_bytes(&ct, &key_params.atk_layout()),
        |s| module.ckks_eval_linear_transformation_self_into(&mut ct_left_streamed, &ct, &lt_left, &atks, s),
    )
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
    with_scratch::<BE, _>(
        module.ckks_eval_linear_transformation_tmp_bytes(&ct, &key_params.atk_layout()),
        |s| module.ckks_eval_linear_transformation_self_into(&mut ct_right, &ct, &prepared_right, &atks, s),
    )
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
/// sum of the single transforms and carry the single transform's metadata. A
/// mismatched term is rejected, for its own reason, before `dst` is written.
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
                module.ckks_prepare_linear_transformation_baby_steps_tmp_bytes(ct, &params.atk_layout()),
                |s| module.ckks_prepare_linear_transformation_baby_steps(&mut babies, ct, &keys, s),
            )
            .unwrap();
            babies
        })
        .collect();
    let sum_bytes = module.ckks_eval_linear_transformation_sum_tmp_bytes(&cts[0], &params.atk_layout());

    // One term: the single transform, bit for bit.
    let mut single = alloc_ct(&params, module, params.k);
    with_scratch::<BE, _>(
        module.ckks_eval_linear_transformation_tmp_bytes(&cts[0], &params.atk_layout()),
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
        (have.layout, have.canonical, have.noise),
        (single.layout, single.canonical, single.noise),
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

    // A mismatched term: rejected for `reason`, with `sum` untouched, as `InvalidOperand` where the sum's own
    // checks reject it.
    let mut reject = |srcs: &[&CKKSCiphertextOwned<BE>],
                      cache: &[&LinearTransformationBabySteps<BE>],
                      lts: &[&LinearTransformationPrepared<BE>],
                      reason: &str,
                      invalid_operand: bool| {
        let before = snapshot::<BE, _>(&sum);
        let err = with_scratch::<BE, _>(sum_bytes, |scratch| {
            module.ckks_eval_linear_transformation_sum_into(&mut sum, srcs, cache, lts, &keys, scratch)
        })
        .unwrap_err();
        assert!(err.to_string().contains(reason), "expected {reason:?}, got: {err}");
        assert_eq!(
            matches!(
                err.composition(),
                Some(CKKSCompositionError::InvalidOperand {
                    op: "ckks_eval_linear_transformation_sum_into",
                    ..
                })
            ),
            invalid_operand,
            "{reason}: {err:?}"
        );
        assert_eq!(snapshot::<BE, _>(&sum), before, "a rejected sum wrote dst ({reason})");
    };
    let two = [&cts[0], &cts[1]];
    reject(&[], &[], &[], "has no terms", true);
    reject(&two, &[&babies[0]], &[&prepared[0], &prepared[1]], "differ in length", true);
    let empty: LinearTransformationPrepared<BE> = LinearTransformation {
        baby_steps: vec![0],
        giant_steps: Vec::new(),
    };
    reject(&[&cts[0]], &[&babies[0]], &[&empty], "has no diagonals", true);
    let other_schedule = prepare_lt(
        module,
        &encode_lt(module, &params, &bs[1], 2 * n1, false, &mut scratch.borrow()),
    );
    reject(
        &two,
        &[&babies[0], &babies[1]],
        &[&prepared[0], &other_schedule],
        "giant schedule",
        true,
    );
    let mut bad = prepare_lt(module, &lts[1]);
    let scale = bad.first_diagonal_plaintext().unwrap().log_scale();
    bad.set_log_scale(scale + 1);
    reject(
        &two,
        &[&babies[0], &babies[1]],
        &[&prepared[0], &bad],
        "diagonal layout, width, scale and slot kind",
        true,
    );
    // Only a later diagonal differs, which core's first-diagonal check misses.
    bad.set_log_scale(scale);
    bad.giant_steps.last_mut().unwrap().diagonals[0]
        .plaintext
        .set_log_scale(scale + 1);
    reject(&two, &[&babies[0], &babies[1]], &[&prepared[0], &bad], "not uniform", true);
    let narrow = alloc_ct(&params, module, params.k - params.base2k);
    reject(
        &[&cts[0], &narrow],
        &[&babies[0], &babies[1]],
        &[&prepared[0], &prepared[1]],
        "source layout and scale",
        true,
    );
    // Core reads the sources through their caches: one built from a narrower source is rejected here, not there.
    let narrow_babies = LinearTransformationBabySteps::alloc(module, prepared[1].baby_steps(), &narrow);
    reject(
        &two,
        &[&babies[0], &narrow_babies],
        &[&prepared[0], &prepared[1]],
        "baby-step cache layout",
        true,
    );
    // An empty cache holds no limb to read a layout from.
    let empty_babies = LinearTransformationBabySteps::alloc(module, &[], &cts[1]);
    reject(
        &two,
        &[&babies[0], &empty_babies],
        &[&prepared[0], &prepared[1]],
        "missing prepared baby-step rotation",
        false,
    );
    let no_babies = LinearTransformationBabySteps::alloc(module, &[0], &cts[1]);
    reject(
        &two,
        &[&babies[0], &no_babies],
        &[&prepared[0], &prepared[1]],
        "missing prepared baby-step rotation",
        false,
    );
}

/// Prepares `lt` into `res` on an empty arena, which must be refused as `InvalidOperand` for `reason` before any write,
/// `res` staying equal to `want`.
fn assert_prepare_refused<BE>(
    module: &Module<BE>,
    res: &mut LinearTransformationPrepared<BE>,
    want: &LinearTransformationPrepared<BE>,
    lt: &LinearTransformation<CKKSPlaintextOwned<BE>>,
    reason: &str,
) where
    BE: TestContextBackend<Ring = Standard>,
    Module<BE>: CKKSLinearTransformationOps<BE>,
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
            .map(|d| (d.plaintext.log_scale(), d.plaintext.real_slots()))
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
pub fn test_prepare_linear_transformation_rejects_mixed_diagonals<BE, F, E>(
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

/// A destination allocated for another schedule or diagonal layout is refused by the preparation as `InvalidOperand`,
/// before any write and any scratch, with the destination unchanged.
pub fn test_prepare_linear_transformation_rejects_foreign_layout<BE, F, E>(
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

/// Key set that records the precision each lookup was made at.
struct QueryLog<K> {
    keys: HashMap<i64, K>,
    seen: RefCell<Vec<(i64, TorusPrecision)>>,
}

impl<BE: Backend, K: GLWEAutomorphismKeyPreparedToBackendRef<BE>> GetAutomorphismKey<BE> for QueryLog<K> {
    fn lookup_automorphism_key(
        &self,
        p: i64,
        k: TorusPrecision,
    ) -> poulpy_core::Result<GLWEAutomorphismKeyPreparedBackendRef<'_, BE>> {
        self.seen.borrow_mut().push((p, k));
        self.keys.get_automorphism_key(p, k)
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
    let keys = QueryLog {
        keys,
        seen: RefCell::new(Vec::new()),
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
    with_scratch::<BE, _>(
        module.ckks_eval_linear_transformation_tmp_bytes(&ct, &key_params.atk_layout()),
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

    let seen = keys.seen.borrow();
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

    // One call reading both layouts takes the larger of their two queries.
    let eval_bytes = module
        .ckks_eval_linear_transformation_tmp_bytes(&ct, &stored_infos)
        .max(module.ckks_eval_linear_transformation_tmp_bytes(&ct, &coarse_infos));
    with_scratch::<BE, _>(eval_bytes, |s| {
        module.ckks_eval_linear_transformation_self_into(&mut res, &ct, &prepared, &keys, s)
    })
    .unwrap();
}
