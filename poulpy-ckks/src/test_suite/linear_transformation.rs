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
    Diagonals, Dsize, Evaluate, GGLWEInfos, GLWEToBackendRef, GetAutomorphismKey, LWEInfos, LinearTransformationStrategy,
    TorusPrecision,
    prepared::{GLWEAutomorphismKeyPrepared, GLWEAutomorphismKeyPreparedBackendRef, GLWEAutomorphismKeyPreparedToBackendRef},
};
use poulpy_hal::{
    api::{CnvPVecAlloc, NegacyclicFFT, NegacyclicFFTNew, ScratchAvailable, ScratchOwnedBorrow},
    layouts::{
        Backend, CyclotomicOrder, HostBytesBackend, HostDataRef, Module, ScratchArena, ZnxView, ZnxViewMut, galois_element,
    },
};

use crate::{
    api::{CKKSLinearTransformationOps, LinearTransformation, LinearTransformationPrepared},
    layouts::{CKKSPlaintextOwned, ComplexDiagonals},
    test_suite::reference_encoder::ReferenceEncoder,
    test_suite::{
        CKKSTestParams,
        helpers::{
            TestContextBackend, TestContextModule, TestScalar, alloc_ct, alloc_scratch, assert_decrypt_precision, ckks_encrypt,
            gen_atk, gen_sk_with_raw, test_vector_1,
        },
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

fn assert_partial_limb_canonical<BE, C>(ct: &C)
where
    BE: TestContextBackend,
    C: GLWEToBackendRef<BE> + LWEInfos,
    for<'a> BE::BufRef<'a>: HostDataRef,
{
    let base2k = ct.base2k().as_usize();
    let padding = (base2k - ct.k().as_usize() % base2k) % base2k;
    assert_ne!(padding, 0, "test requires a partial output limb");

    let low_mask = (1i64 << padding) - 1;
    let ct_ref = ct.to_backend_ref();
    let bottom_limb = ct.size() - 1;
    for col in 0..ct_ref.data().cols() {
        assert!(
            ct_ref.data().at(col, bottom_limb).iter().all(|value| value & low_mask == 0),
            "linear transformation produced noncanonical padding in column {col}"
        );
        for limb in ct.size()..ct_ref.data().size() {
            assert!(
                ct_ref.data().at(col, limb).iter().all(|&value| value == 0),
                "linear transformation left stale data in column {col}, limb {limb}"
            );
        }
    }
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
    ct_left.data_mut().raw_mut().fill(1);
    module
        .ckks_eval_linear_transformation_self_into(&mut ct_left, &ct, &prepared_left, &atks, &mut scratch.borrow())
        .unwrap();
    assert_partial_limb_canonical::<BE, _>(&ct_left);
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
    assert_partial_limb_canonical::<BE, _>(&ct_left_streamed);
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

/// The stored keys, with the Galois elements listed in `coarse_ps` answered
/// through a coarser `dsize`: one key set carrying two different layouts.
struct MixedDsize<BE: Backend> {
    keys: HashMap<i64, GLWEAutomorphismKeyPrepared<BE::OwnedBuf, BE>>,
    coarse: Dsize,
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
            key.with_dsize(self.coarse)
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
    let key_params = CKKSTestParams {
        dsize: params.dsize.max(4),
        ..params
    };
    let mut scratch = alloc_scratch(&key_params, module);

    let n1 = 2;
    let b = complex_diagonals::<F>(&[0, 1, 2, 3, 4, 5], m);
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

    let prepared = prepare_lt(module, &lt, &mut scratch.borrow());
    let mut res = alloc_ct(&params, module, params.k - params.base2k);
    // The evaluation stamps its own result metadata onto `res`, so the giant
    // precision to compare against is the one it was given, not the one left.
    let dst_k = res.k();
    module
        .ckks_eval_linear_transformation_self_into(&mut res, &ct, &prepared, &keys, &mut scratch.borrow())
        .unwrap();

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
    // Stored at the smallest `dsize` that still leaves a digit when read at
    // twice that value, so the coarsened twin below is always reachable.
    let key_params = CKKSTestParams {
        dsize: params.dsize.max(2),
        ..params
    };
    let mut scratch = alloc_scratch(&key_params, module);

    let n1 = 2;
    let b = complex_diagonals::<F>(&[0, 1, 2, 3, 4, 5], m);
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
    let coarse_dsize = Dsize(2 * key_params.dsize as u32);
    let coarse_infos = stored_infos.gglwe_layout().at_dsize(coarse_dsize).unwrap();
    let keys = MixedDsize::<BE> {
        keys,
        coarse: coarse_dsize,
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
    let prepared = prepare_lt(module, &lt, &mut scratch.borrow());
    let mut res = alloc_ct(&params, module, params.k - params.base2k);

    // Exactly the budget a caller can derive: the larger of the two per-layout
    // queries, nothing rounded up on top.
    let mut eval_scratch = <poulpy_hal::layouts::ScratchOwned<BE> as poulpy_hal::api::ScratchOwnedAlloc<BE>>::alloc(
        module
            .ckks_eval_linear_transformation_tmp_bytes(&ct, &stored_infos)
            .max(module.ckks_eval_linear_transformation_tmp_bytes(&ct, &coarse_infos)),
    );
    module
        .ckks_eval_linear_transformation_self_into(&mut res, &ct, &prepared, &keys, &mut eval_scratch.borrow())
        .unwrap();
}
