//! Linear transformation (BSGS) test.
//!
//! Minimal correctness check: `dec(lt(enc(a), B)) ≈ B·a`, where `a` is the
//! encrypted complex slot vector and `B` is a complex matrix supplied as a raw
//! diagonal map. The transform applies the matrix on the *left*
//! (`(B·a)[j] = Σ_k B[j][k]·a[k]`); the expected value is computed by the
//! scheme-agnostic plaintext evaluator [`ComplexDiagonals`] / [`Evaluate`],
//! which the homomorphic engine must match up to CKKS precision.

use crate::api::CKKSEncodingOps;
use std::collections::HashMap;

use poulpy_core::layouts::{Diagonals, Evaluate, LinearTransformationStrategy};
use poulpy_hal::{
    api::{CnvPVecAlloc, NegacyclicFFT, NegacyclicFFTNew, ScratchAvailable, ScratchOwnedBorrow},
    layouts::{CyclotomicOrder, HostBytesBackend, Module, ScratchArena},
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
    let mut scratch = alloc_scratch(&params, module);

    // Complex matrix B (four diagonals), decomposed with n1 = 2 baby steps.
    // Encode both orientations through the `transpose` flag and check each one.
    let n1 = 2;
    let strategy = LinearTransformationStrategy::Bsgs { giant_step: n1 };
    let b = complex_diagonals::<F>(&[0, 1, 2, 3], m);
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
            .or_insert_with(|| gen_atk(&params, module, p, &sk_raw, &mut scratch.borrow()));
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
