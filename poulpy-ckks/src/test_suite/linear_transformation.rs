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
use std::collections::{BTreeSet, HashMap};

use poulpy_core::layouts::{
    Diagonals, Dsize, Evaluate, GGLWEInfos, GLWEAutomorphismKeyHelper, GLWEAutomorphismKeyLayoutHelper, LWEInfos,
    LinearTransformationStrategy, TorusPrecision,
};
use poulpy_hal::{
    api::{CnvPVecAlloc, NegacyclicFFT, NegacyclicFFTNew, ScratchAvailable, ScratchOwnedAlloc, ScratchOwnedBorrow},
    layouts::{CyclotomicOrder, HostBytesBackend, Module, ScratchArena, ScratchOwned, galois_element},
};

use crate::{
    api::{CKKSLinearTransformationOps, LinearTransformation, LinearTransformationBabySteps, LinearTransformationPrepared},
    layouts::{CKKSPlaintextOwned, ComplexDiagonals},
    test_suite::reference_encoder::ReferenceEncoder,
    test_suite::{
        CKKSTestParams,
        helpers::{
            TestContextBackend, TestContextModule, TestScalar, alloc_ct, alloc_scratch, assert_decrypt_precision, ckks_encrypt,
            gen_atk, gen_sk_with_raw, test_vector_1,
        },
        polynomial_evaluation::assert_ct_identical,
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
/// diagonal indices (no pre-rotation — that is the encoder's job).
fn complex_diagonals<F: TestScalar>(diag_indices: &[usize], m: usize) -> ComplexDiagonals<F> {
    let mut re = Diagonals::<F>::new(m);
    let mut im = Diagonals::<F>::new(m);
    for &i in diag_indices {
        let (dr, di) = diagonal::<F>(i, m);
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
    BE: TestContextBackend,
    Module<BE>: TestContextModule<BE> + CKKSEncodingOps<BE, F>,
    F: TestScalar,
{
    crate::default::ckks_encode_linear_transformation_from_diagonals(
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
fn prepare_lt<BE>(
    module: &Module<BE>,
    lt: &LinearTransformation<CKKSPlaintextOwned<BE>>,
    scratch: &mut ScratchArena<'_, BE>,
) -> LinearTransformationPrepared<BE>
where
    BE: TestContextBackend,
    for<'a> <BE as poulpy_hal::layouts::Backend>::BufRef<'a>: poulpy_hal::layouts::HostDataRef,
    for<'a> <BE as poulpy_hal::layouts::Backend>::BufMut<'a>: poulpy_hal::layouts::HostDataMut,
    Module<BE>: TestContextModule<BE> + CKKSLinearTransformationOps<BE> + CnvPVecAlloc<BE>,
{
    let first = lt.first_diagonal_plaintext().expect("linear transformation has no diagonals");
    let mut prepared = LinearTransformationPrepared::<BE>::alloc_prepared_from_index(module, &lt.index(), first);
    module.ckks_prepare_linear_transformation_rhs(&mut prepared, lt, scratch);
    prepared
}

/// `dec(lt(enc(a), B)) ≈ B·a` for a complex matrix `B` in BSGS form.
pub fn test_linear_transformation<BE, F, E>(params: CKKSTestParams, module: &Module<BE>, host_module: &Module<HostBytesBackend>)
where
    BE: TestContextBackend,
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
    // giant rotations. Exact backends then cover the shortened shared
    // accumulator, while approximate backends retain the full key width.
    let key_params = CKKSTestParams {
        dsize: params.dsize.max(4),
        ..params
    };
    let mut scratch = alloc_scratch(&key_params, module);

    // Complex matrix B (six diagonals), decomposed with n1 = 2 baby steps.
    // Encode both orientations through the `transpose` flag and check each one.
    let n1 = 2;
    let strategy = LinearTransformationStrategy::Bsgs { giant_step: n1 };
    let b = complex_diagonals::<F>(&[0, 1, 2, 3, 4, 5], m);
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
    let prepared_left = prepare_lt(module, &lt_left, &mut scratch.borrow());
    let mut ct_left = alloc_ct(&params, module, params.k);
    module
        .ckks_eval_linear_transformation_self_into(&mut ct_left, &ct, &prepared_left, &atks, &mut scratch.borrow())
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
    module
        .ckks_eval_linear_transformation_self_into(&mut ct_left_streamed, &ct, &lt_left, &atks, &mut scratch.borrow())
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
    let prepared_right = prepare_lt(module, &lt_right, &mut scratch.borrow());
    let mut ct_right = alloc_ct(&params, module, params.k);
    module
        .ckks_eval_linear_transformation_self_into(&mut ct_right, &ct, &prepared_right, &atks, &mut scratch.borrow())
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

/// Key set that records the precision each lookup was made at.
struct QueryLog<K> {
    keys: HashMap<i64, K>,
    seen: RefCell<Vec<(i64, TorusPrecision)>>,
}

impl<K: GGLWEInfos> GLWEAutomorphismKeyHelper<K> for QueryLog<K> {
    fn get_automorphism_key_for(&self, p: i64, k: TorusPrecision) -> poulpy_core::Result<(&K, Dsize)> {
        self.seen.borrow_mut().push((p, k));
        self.keys.get_automorphism_key_for(p, k)
    }
}

impl<K: GGLWEInfos> GLWEAutomorphismKeyLayoutHelper<K> for QueryLog<K> {
    fn get_automorphism_key_layout_for(&self, p: i64, k: TorusPrecision) -> poulpy_core::Result<(&K, Dsize)> {
        self.seen.borrow_mut().push((p, k));
        self.keys.get_automorphism_key_layout_for(p, k)
    }
}

/// Helper that makes one late giant key disappear while delegating every
/// earlier lookup unchanged.
struct RejectElement<'a, K> {
    inner: &'a QueryLog<K>,
    reject: i64,
}

impl<K: GGLWEInfos> GLWEAutomorphismKeyHelper<K> for RejectElement<'_, K> {
    fn get_automorphism_key_for(&self, p: i64, k: TorusPrecision) -> poulpy_core::Result<(&K, Dsize)> {
        self.inner
            .get_automorphism_key_for(if p == self.reject { i64::MIN } else { p }, k)
    }
}

impl<K: GGLWEInfos> GLWEAutomorphismKeyLayoutHelper<K> for RejectElement<'_, K> {
    fn get_automorphism_key_layout_for(&self, p: i64, k: TorusPrecision) -> poulpy_core::Result<(&K, Dsize)> {
        self.inner
            .get_automorphism_key_layout_for(if p == self.reject { i64::MIN } else { p }, k)
    }
}

/// Helper that returns a key but an unrealizable decomposition for one late
/// giant, exercising covering-bind validation before evaluation writes.
struct InvalidDsize<'a, K> {
    inner: &'a QueryLog<K>,
    invalid: i64,
}

impl<K: GGLWEInfos> GLWEAutomorphismKeyHelper<K> for InvalidDsize<'_, K> {
    fn get_automorphism_key_for(&self, p: i64, k: TorusPrecision) -> poulpy_core::Result<(&K, Dsize)> {
        let (key, dsize) = self.inner.get_automorphism_key_for(p, k)?;
        Ok((key, if p == self.invalid { Dsize(u32::MAX) } else { dsize }))
    }
}

impl<K: GGLWEInfos> GLWEAutomorphismKeyLayoutHelper<K> for InvalidDsize<'_, K> {
    fn get_automorphism_key_layout_for(&self, p: i64, k: TorusPrecision) -> poulpy_core::Result<(&K, Dsize)> {
        let (key, dsize) = self.inner.get_automorphism_key_layout_for(p, k)?;
        Ok((key, if p == self.invalid { Dsize(u32::MAX) } else { dsize }))
    }
}

/// Baby keys resolve at source precision and giant keys at destination
/// precision, in both exact scratch planning and execution. Explicit `_into`
/// honors its caller-selected destination, while `self_assign` plans the
/// factor's natural lower post-product destination before key lookup.
///
/// Giant rotations deliberately use heterogeneous physical decompositions so
/// the lazy accumulator must combine a global widest width with each key's own
/// rotation width instead of forcing every rotation through one shared shape.
pub fn test_linear_transformation_operation_precisions<BE, F, E>(
    params: CKKSTestParams,
    module: &Module<BE>,
    host_module: &Module<HostBytesBackend>,
) where
    BE: TestContextBackend,
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
    let fine_key_params = CKKSTestParams {
        dsize: params.dsize.max(4),
        ..params
    };
    let wide_key_params = CKKSTestParams {
        dsize: params.dsize.max(8),
        ..params
    };
    let mut scratch = alloc_scratch(&fine_key_params, module);

    let n1 = 2;
    let b = complex_diagonals::<F>(&[0, 1, 2, 3, 4, 5], m);
    let lt = encode_lt(module, &params, &b, n1, false, &mut scratch.borrow());

    let order = module.cyclotomic_order();
    let baby_elements = lt
        .baby_steps()
        .iter()
        .copied()
        .filter(|&rot| rot != 0)
        .map(|rot| galois_element(rot, order))
        .collect::<BTreeSet<_>>();
    let giant_elements = lt
        .giant_steps
        .iter()
        .filter(|gs| gs.rot != 0 && !gs.diagonals.is_empty())
        .map(|gs| galois_element(gs.rot, order))
        .collect::<BTreeSet<_>>();
    assert!(giant_elements.len() >= 2, "regression needs two giant rotations");

    let mut keys = HashMap::new();
    for p in lt.galois_elements(order) {
        let params_for_key = giant_elements
            .iter()
            .position(|&giant| giant == p)
            .filter(|index| index % 2 == 1)
            .map_or(&fine_key_params, |_| &wide_key_params);
        keys.entry(p)
            .or_insert_with(|| gen_atk(params_for_key, module, p, &sk_raw, &mut scratch.borrow()));
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

    let prepared = prepare_lt(module, &lt, &mut scratch.borrow());
    let mut res = alloc_ct(&params, module, params.k - params.base2k);
    let (src_k, dst_k) = (ct.k(), res.k());

    let assert_precisions = |seen: &[(i64, TorusPrecision)], giant_k: TorusPrecision| {
        assert!(!seen.is_empty(), "the factor consulted no key");
        for &(p, k) in seen {
            match (baby_elements.contains(&p), giant_elements.contains(&p)) {
                (true, false) => assert_eq!(k, src_k, "baby key p={p} resolved at the wrong precision"),
                (false, true) => assert_eq!(k, giant_k, "giant key p={p} resolved at the wrong precision"),
                (true, true) => assert!(
                    k == src_k || k == giant_k,
                    "shared baby/giant key p={p} resolved at unrelated precision {k}"
                ),
                (false, false) => panic!("unexpected key lookup p={p} at k={k}"),
            }
        }
    };

    let exact_scratch_bytes = module.ckks_eval_linear_transformation_into_tmp_bytes(&res, &ct, &prepared, &keys);
    assert_precisions(&keys.seen.borrow(), dst_k);
    keys.seen.borrow_mut().clear();

    let mut babies = LinearTransformationBabySteps::alloc(module, prepared.baby_steps(), &ct);
    module
        .ckks_prepare_linear_transformation_baby_steps(&mut babies, &ct, &keys, &mut scratch.borrow())
        .unwrap();
    let mut exact_scratch: ScratchOwned<BE> = ScratchOwned::alloc(exact_scratch_bytes);
    module
        .ckks_eval_linear_transformation_into(&mut res, &ct, &babies, &prepared, &keys, &mut exact_scratch.borrow())
        .unwrap();
    assert_precisions(&keys.seen.borrow(), dst_k);
    keys.seen.borrow_mut().clear();

    // The assign API keeps the source-shaped physical scratch allocation, but
    // the factor consumes its diagonal scale before giant rotations. Its exact
    // query and execution must therefore both resolve giant keys at this
    // natural lower precision rather than at `src_k`.
    let factor_log_scale = prepared
        .first_diagonal_plaintext()
        .expect("linear transformation has no diagonals")
        .log_scale();
    let assign_dst_k: TorusPrecision = src_k
        .as_usize()
        .checked_sub(factor_log_scale)
        .expect("test factor scale exceeds source precision")
        .into();
    let assign_scratch_bytes = module.ckks_eval_linear_transformation_tmp_bytes(&ct, &prepared, &keys);
    assert_precisions(&keys.seen.borrow(), assign_dst_k);
    keys.seen.borrow_mut().clear();
    let streamed_assign_scratch_bytes = module.ckks_eval_linear_transformation_streamed_tmp_bytes(&ct, &lt, &keys);
    assert!(streamed_assign_scratch_bytes > 0);
    assert_precisions(&keys.seen.borrow(), assign_dst_k);
    keys.seen.borrow_mut().clear();

    // A cache prepared for a different source precision must fail before the
    // destination's metadata or limbs are touched.
    let mismatched_babies = LinearTransformationBabySteps::alloc(module, prepared.baby_steps(), &res);
    let mut guarded = alloc_ct(&params, module, params.k - params.base2k);
    let untouched = guarded.clone();
    let err = module
        .ckks_eval_linear_transformation_into(&mut guarded, &ct, &mismatched_babies, &prepared, &keys, &mut scratch.borrow())
        .expect_err("a baby cache at the wrong source precision must be rejected");
    assert!(
        err.to_string().contains("baby cache precision"),
        "unexpected cache error: {err}"
    );
    assert_ct_identical::<BE>("mismatched baby cache", &untouched, &guarded);
    keys.seen.borrow_mut().clear();

    let late_giant = *giant_elements.last().expect("regression needs a giant key");
    {
        let helper = RejectElement {
            inner: &keys,
            reject: late_giant,
        };
        let mut guarded = alloc_ct(&params, module, params.k - params.base2k);
        let untouched = guarded.clone();
        module
            .ckks_eval_linear_transformation_into(&mut guarded, &ct, &babies, &prepared, &helper, &mut scratch.borrow())
            .expect_err("late missing giant");
        assert_ct_identical::<BE>("late missing giant", &untouched, &guarded);
        keys.seen.borrow_mut().clear();
    }
    {
        let helper = InvalidDsize {
            inner: &keys,
            invalid: late_giant,
        };
        let mut guarded = alloc_ct(&params, module, params.k - params.base2k);
        let untouched = guarded.clone();
        module
            .ckks_eval_linear_transformation_into(&mut guarded, &ct, &babies, &prepared, &helper, &mut scratch.borrow())
            .expect_err("late invalid giant");
        assert_ct_identical::<BE>("late invalid giant", &untouched, &guarded);
        keys.seen.borrow_mut().clear();
    }

    let mut assigned = ct;
    let mut assign_scratch: ScratchOwned<BE> = ScratchOwned::alloc(assign_scratch_bytes);
    module
        .ckks_eval_linear_transformation_self_assign(&mut assigned, &prepared, &keys, &mut assign_scratch.borrow())
        .unwrap();
    assert_precisions(&keys.seen.borrow(), assign_dst_k);
    assert_eq!(assigned.k(), assign_dst_k, "self_assign retained the pre-factor precision");

    let strategy = LinearTransformationStrategy::Bsgs { giant_step: n1 };
    let (want_re, want_im) = b.evaluate((a_re.as_slice(), a_im.as_slice()), strategy);
    assert_decrypt_precision(
        "linear_transformation_operation_precisions",
        &params,
        module,
        &encoder,
        &res,
        &sk,
        &want_re,
        &want_im,
        &mut scratch.borrow(),
    );
    assert_decrypt_precision(
        "linear_transformation_self_assign_operation_precisions",
        &params,
        module,
        &encoder,
        &assigned,
        &sk,
        &want_re,
        &want_im,
        &mut scratch.borrow(),
    );
}
