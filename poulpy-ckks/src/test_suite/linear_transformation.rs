//! Linear transformation (BSGS) test.
//!
//! Builds a small linear map `M` from an explicit set of slot diagonals, encodes
//! it in baby-step / giant-step form, evaluates it homomorphically with
//! [`LinearTransformationOps`], and checks the decrypted result against the
//! plaintext diagonal sum `M·v = Σ_i diag_i ⊙ rot(v, i)`.

use std::collections::HashMap;

use poulpy_core::layouts::prepared::GLWEAutomorphismKeyPrepared;
use poulpy_hal::{
    api::{NegacyclicFFT, NegacyclicFFTNew, ScratchAvailable, ScratchOwnedBorrow},
    layouts::{Backend, HostBytesBackend, Module, ScratchArena, TransferFrom},
};

use crate::{
    CKKSCompositionError, CKKSCtBounds, CKKSInfos,
    api::{
        Diagonal, GiantStep, LinearTransformation, LinearTransformationOps, PreparedBabyStepHelper, PreparedLinearTransformation,
    },
    encoding::reim::Encoder,
    layouts::CKKSPlaintext,
    test_suite::{
        CKKSTestParams,
        helpers::{
            TestContextBackend, TestContextModule, TestScalar, alloc_ct, alloc_scratch, assert_decrypt_precision,
            assert_precision, ckks_decrypt_decode, ckks_encrypt, encode_and_upload_pt, gen_atk, gen_sk_with_raw, test_vector_1,
            want_mul, want_rotate,
        },
    },
};

type TestAtk<BE> = GLWEAutomorphismKeyPrepared<<BE as Backend>::OwnedBuf, BE>;

fn prepare_linear_transform<BE, P>(
    module: &Module<BE>,
    lt: &LinearTransformation<P>,
    scratch: &mut ScratchArena<'_, BE>,
) -> PreparedLinearTransformation<BE>
where
    BE: Backend,
    Module<BE>: LinearTransformationOps<BE>,
    P: poulpy_core::layouts::GLWEToBackendRef<BE> + CKKSCtBounds,
{
    let mut prepared = PreparedLinearTransformation::default();
    module.ckks_prepare_linear_transformation(lt, &mut prepared, scratch);
    prepared
}

fn diagonal<F: TestScalar>(i: usize, m: usize) -> (Vec<F>, Vec<F>) {
    let re: Vec<F> = (0..m)
        .map(|j| F::from_f64(0.25 * (i as f64 + 1.0) / (1.0 + (j % 8) as f64)).unwrap())
        .collect();
    let im = vec![F::from_f64(0.0).unwrap(); m];
    (re, im)
}

fn build_transform<BE, F, E>(
    params: CKKSTestParams,
    module: &Module<BE>,
    host_module: &Module<HostBytesBackend>,
    encoder: &Encoder<E>,
    diag_indices: &[usize],
    n1: usize,
) -> (
    LinearTransformation<CKKSPlaintext<BE::OwnedBuf>>,
    Vec<(usize, Vec<F>, Vec<F>)>,
)
where
    BE: TestContextBackend + TransferFrom<HostBytesBackend>,
    Module<HostBytesBackend>: crate::test_suite::helpers::TestContextHostModule,
    F: TestScalar,
    E: NegacyclicFFT<F>,
{
    let m = params.n / 2;
    let baby_steps: Vec<i64> = (0..n1).map(|k| k as i64).collect();
    let n2 = diag_indices.iter().copied().max().map_or(0, |i| (i / n1) + 1);
    let mut giant_steps: Vec<GiantStep<CKKSPlaintext<BE::OwnedBuf>>> = (0..n2)
        .map(|j| GiantStep {
            rot: (n1 * j) as i64,
            diagonals: Vec::new(),
        })
        .collect();
    let mut reference_diags = Vec::with_capacity(diag_indices.len());

    for &i in diag_indices {
        let j = i / n1;
        let k = i % n1;
        let rot_slots = (n1 * j) as i64;
        let (diag_re, diag_im) = diagonal::<F>(i, m);
        let (pre_re, pre_im) = want_rotate(&diag_re, &diag_im, -rot_slots, m);
        let plaintext = encode_and_upload_pt(
            host_module,
            module,
            encoder,
            params.base2k.into(),
            params.prec,
            &pre_re,
            &pre_im,
        );
        giant_steps[j].diagonals.push(Diagonal {
            baby: k as i64,
            plaintext,
        });
        reference_diags.push((i, diag_re, diag_im));
    }

    (LinearTransformation { baby_steps, giant_steps }, reference_diags)
}

fn reference_linear_transform<F: TestScalar>(
    m: usize,
    v_re: &[F],
    v_im: &[F],
    diagonals: &[(usize, Vec<F>, Vec<F>)],
) -> (Vec<F>, Vec<F>) {
    let mut want_re = vec![F::from_f64(0.0).unwrap(); m];
    let mut want_im = vec![F::from_f64(0.0).unwrap(); m];
    for (i, diag_re, diag_im) in diagonals {
        let (rv_re, rv_im) = want_rotate(v_re, v_im, *i as i64, m);
        let (p_re, p_im) = want_mul(diag_re, diag_im, &rv_re, &rv_im);
        for j in 0..m {
            want_re[j] = want_re[j] + p_re[j];
            want_im[j] = want_im[j] + p_im[j];
        }
    }
    (want_re, want_im)
}

/// Evaluates `M·v` where `M` has four real diagonals `{0,1,2,3}`, decomposed with
/// `n1 = 2` baby steps and `n2 = 2` giant steps.
pub fn test_linear_transformation<BE, F, E>(params: CKKSTestParams, module: &Module<BE>, host_module: &Module<HostBytesBackend>)
where
    BE: TestContextBackend,
    Module<BE>: TestContextModule<BE> + LinearTransformationOps<BE>,
    F: TestScalar,
    E: NegacyclicFFT<F> + NegacyclicFFTNew<F>,
    for<'a> ScratchArena<'a, BE>: ScratchAvailable,
{
    let m = params.n / 2;
    let encoder = Encoder::<E>::new(m).unwrap();

    let (v_re, v_im) = test_vector_1::<F>(m);
    let (sk_raw, sk) = gen_sk_with_raw(&params, module, host_module, [0u8; 32]);
    let mut scratch = alloc_scratch(&params, module);

    // --- Define M by four real diagonals over the slots. ---
    let num_diag: usize = 4;
    let n1: usize = 2;
    let n2: usize = num_diag / n1;

    let diag_re: Vec<Vec<F>> = (0..num_diag)
        .map(|i| {
            (0..m)
                .map(|j| F::from_f64(0.25 * (i as f64 + 1.0) / (1.0 + (j % 8) as f64)).unwrap())
                .collect()
        })
        .collect();
    let diag_im: Vec<Vec<F>> = vec![vec![F::from_f64(0.0).unwrap(); m]; num_diag];

    // --- Encode the pre-rotated diagonals ũ_{j,k} = rot(diag_{n1·j+k}, −n1·j). ---
    let baby_steps: Vec<i64> = (0..n1).map(|k| k as i64).collect();
    let mut giant_steps: Vec<GiantStep<_>> = Vec::with_capacity(n2);
    for j in 0..n2 {
        let rot_slots = (n1 * j) as i64;
        let mut diagonals = Vec::with_capacity(n1);
        for k in 0..n1 {
            let i = n1 * j + k;
            let (pre_re, pre_im) = want_rotate(&diag_re[i], &diag_im[i], -rot_slots, m);
            let plaintext = encode_and_upload_pt(
                host_module,
                module,
                &encoder,
                params.base2k.into(),
                params.prec,
                &pre_re,
                &pre_im,
            );
            diagonals.push(Diagonal {
                baby: k as i64,
                plaintext,
            });
        }
        giant_steps.push(GiantStep {
            rot: rot_slots,
            diagonals,
        });
    }
    let lt = LinearTransformation { baby_steps, giant_steps };

    // --- Automorphism keys for every required rotation (keyed by rotation amount). ---
    let mut atks = HashMap::new();
    for r in lt.required_rotations() {
        let atk = gen_atk(&params, module, r, &sk_raw, &mut scratch.borrow());
        atks.insert(r, atk);
    }

    // --- Encrypt v and evaluate M·v. ---
    let ct = ckks_encrypt(
        &params,
        module,
        host_module,
        &encoder,
        &sk,
        params.k,
        &v_re,
        &v_im,
        &mut scratch.borrow(),
    );
    let mut ct_res = alloc_ct(&params, module, params.k);
    module
        .ckks_eval_linear_transformation_into(&mut ct_res, &ct, &lt, &atks, &mut scratch.borrow())
        .unwrap();

    // --- Reference: M·v = Σ_i diag_i ⊙ rot(v, i). ---
    let mut want_re = vec![F::from_f64(0.0).unwrap(); m];
    let mut want_im = vec![F::from_f64(0.0).unwrap(); m];
    for i in 0..num_diag {
        let (rv_re, rv_im) = want_rotate(&v_re, &v_im, i as i64, m);
        let (p_re, p_im) = want_mul(&diag_re[i], &diag_im[i], &rv_re, &rv_im);
        for j in 0..m {
            want_re[j] = want_re[j] + p_re[j];
            want_im[j] = want_im[j] + p_im[j];
        }
    }

    assert_decrypt_precision(
        "linear_transformation",
        &params,
        module,
        &encoder,
        &ct_res,
        &sk,
        &want_re,
        &want_im,
        &mut scratch.borrow(),
    );
}

/// Sparse transform whose first giant step is empty; also exercises `_assign`.
pub fn test_linear_transformation_sparse_assign<BE, F, E>(
    params: CKKSTestParams,
    module: &Module<BE>,
    host_module: &Module<HostBytesBackend>,
) where
    BE: TestContextBackend,
    Module<BE>: TestContextModule<BE> + LinearTransformationOps<BE>,
    F: TestScalar,
    E: NegacyclicFFT<F> + NegacyclicFFTNew<F>,
    for<'a> ScratchArena<'a, BE>: ScratchAvailable,
{
    let m = params.n / 2;
    let encoder = Encoder::<E>::new(m).unwrap();
    let (v_re, v_im) = test_vector_1::<F>(m);
    let (sk_raw, sk) = gen_sk_with_raw(&params, module, host_module, [1u8; 32]);
    let mut scratch = alloc_scratch(&params, module);

    let (lt, reference_diags) = build_transform::<BE, F, E>(params, module, host_module, &encoder, &[3, 5], 3);
    let mut atks = HashMap::new();
    for r in lt.required_rotations() {
        let atk = gen_atk(&params, module, r, &sk_raw, &mut scratch.borrow());
        atks.insert(r, atk);
    }

    let mut ct = ckks_encrypt(
        &params,
        module,
        host_module,
        &encoder,
        &sk,
        params.k,
        &v_re,
        &v_im,
        &mut scratch.borrow(),
    );
    module
        .ckks_eval_linear_transformation_assign(&mut ct, &lt, &atks, &mut scratch.borrow())
        .unwrap();

    let (want_re, want_im) = reference_linear_transform(m, &v_re, &v_im, &reference_diags);
    assert_decrypt_precision(
        "linear_transformation_sparse_assign",
        &params,
        module,
        &encoder,
        &ct,
        &sk,
        &want_re,
        &want_im,
        &mut scratch.borrow(),
    );
}

/// Prepared transform API preserves the unprepared schedule's rotation semantics.
pub fn test_linear_transformation_prepared<BE, F, E>(
    params: CKKSTestParams,
    module: &Module<BE>,
    host_module: &Module<HostBytesBackend>,
) where
    BE: TestContextBackend,
    Module<BE>: TestContextModule<BE> + LinearTransformationOps<BE>,
    F: TestScalar,
    E: NegacyclicFFT<F> + NegacyclicFFTNew<F>,
    for<'a> ScratchArena<'a, BE>: ScratchAvailable,
{
    let m = params.n / 2;
    let encoder = Encoder::<E>::new(m).unwrap();
    let (v_re, v_im) = test_vector_1::<F>(m);
    let (sk_raw, sk) = gen_sk_with_raw(&params, module, host_module, [4u8; 32]);
    let mut scratch = alloc_scratch(&params, module);

    let (lt, reference_diags) = build_transform::<BE, F, E>(params, module, host_module, &encoder, &[0, 2, 5, 7], 3);
    let unprepared_required_rotations = lt.required_rotations();
    let prepared = prepare_linear_transform(module, &lt, &mut scratch.borrow());
    assert_eq!(prepared.required_rotations(), unprepared_required_rotations);
    assert_eq!(prepared.baby_steps, vec![0, 1, 2]);
    assert_eq!(prepared.giant_steps.len(), 3);
    assert_eq!(
        prepared
            .giant_steps
            .iter()
            .map(|gs| gs.baby_step_indexes.clone())
            .collect::<Vec<_>>(),
        vec![vec![0, 2], vec![2], vec![1]]
    );

    let mut atks = HashMap::new();
    for r in prepared.required_rotations() {
        let atk = gen_atk(&params, module, r, &sk_raw, &mut scratch.borrow());
        atks.insert(r, atk);
    }

    let ct = ckks_encrypt(
        &params,
        module,
        host_module,
        &encoder,
        &sk,
        params.k,
        &v_re,
        &v_im,
        &mut scratch.borrow(),
    );
    let mut ct_res = alloc_ct(&params, module, params.k);
    module
        .ckks_eval_prepared_linear_transformation_into(&mut ct_res, &ct, &lt, &prepared, &atks, &mut scratch.borrow())
        .unwrap();

    let (want_re, want_im) = reference_linear_transform(m, &v_re, &v_im, &reference_diags);
    assert_decrypt_precision(
        "linear_transformation_prepared",
        &params,
        module,
        &encoder,
        &ct_res,
        &sk,
        &want_re,
        &want_im,
        &mut scratch.borrow(),
    );
}

/// Borrowed one-shot evaluation matches an explicitly prepared sparse BSGS map.
pub fn test_linear_transformation_one_shot_matches_prepared<BE, F, E>(
    params: CKKSTestParams,
    module: &Module<BE>,
    host_module: &Module<HostBytesBackend>,
) where
    BE: TestContextBackend,
    Module<BE>: TestContextModule<BE> + LinearTransformationOps<BE>,
    F: TestScalar,
    E: NegacyclicFFT<F> + NegacyclicFFTNew<F>,
    for<'a> ScratchArena<'a, BE>: ScratchAvailable,
{
    let m = params.n / 2;
    let encoder = Encoder::<E>::new(m).unwrap();
    let (v_re, v_im) = test_vector_1::<F>(m);
    let (sk_raw, sk) = gen_sk_with_raw(&params, module, host_module, [9u8; 32]);
    let mut scratch = alloc_scratch(&params, module);

    let diag_indices = &[0, 1, 2, 5, 7];
    let (lt_one_shot, reference_diags) = build_transform::<BE, F, E>(params, module, host_module, &encoder, diag_indices, 3);
    let (lt_for_prepare, _) = build_transform::<BE, F, E>(params, module, host_module, &encoder, diag_indices, 3);
    let prepared = prepare_linear_transform(module, &lt_for_prepare, &mut scratch.borrow());

    let mut required_rotations = lt_one_shot.required_rotations();
    required_rotations.extend(prepared.required_rotations());
    required_rotations.sort_unstable();
    required_rotations.dedup();

    let mut atks = HashMap::new();
    for r in required_rotations {
        let atk = gen_atk(&params, module, r, &sk_raw, &mut scratch.borrow());
        atks.insert(r, atk);
    }

    let ct = ckks_encrypt(
        &params,
        module,
        host_module,
        &encoder,
        &sk,
        params.k,
        &v_re,
        &v_im,
        &mut scratch.borrow(),
    );
    let mut ct_one_shot = alloc_ct(&params, module, params.k);
    let mut ct_prepared = alloc_ct(&params, module, params.k);
    module
        .ckks_eval_linear_transformation_into(&mut ct_one_shot, &ct, &lt_one_shot, &atks, &mut scratch.borrow())
        .unwrap();
    module
        .ckks_eval_prepared_linear_transformation_into(
            &mut ct_prepared,
            &ct,
            &lt_for_prepare,
            &prepared,
            &atks,
            &mut scratch.borrow(),
        )
        .unwrap();

    assert_eq!(ct_one_shot.log_delta(), ct_prepared.log_delta());
    assert_eq!(ct_one_shot.log_budget(), ct_prepared.log_budget());

    let (one_shot_re, one_shot_im) = ckks_decrypt_decode(&params, module, &encoder, &ct_one_shot, &sk, &mut scratch.borrow());
    let (prepared_re, prepared_im) = ckks_decrypt_decode(&params, module, &encoder, &ct_prepared, &sk, &mut scratch.borrow());
    assert_precision(
        "linear_transformation_one_shot_matches_prepared re",
        &prepared_re,
        &one_shot_re,
        ct_one_shot.log_delta(),
        params.n,
    );
    assert_precision(
        "linear_transformation_one_shot_matches_prepared im",
        &prepared_im,
        &one_shot_im,
        ct_one_shot.log_delta(),
        params.n,
    );

    let (want_re, want_im) = reference_linear_transform(m, &v_re, &v_im, &reference_diags);
    assert_decrypt_precision(
        "linear_transformation_one_shot_matches_prepared_one_shot",
        &params,
        module,
        &encoder,
        &ct_one_shot,
        &sk,
        &want_re,
        &want_im,
        &mut scratch.borrow(),
    );
    assert_decrypt_precision(
        "linear_transformation_one_shot_matches_prepared_prepared",
        &params,
        module,
        &encoder,
        &ct_prepared,
        &sk,
        &want_re,
        &want_im,
        &mut scratch.borrow(),
    );
}

/// Dense-enough prepared evaluation stresses lazy BIG accumulation over many giant steps.
pub fn test_linear_transformation_prepared_big_accumulator_stress<BE, F, E>(
    params: CKKSTestParams,
    module: &Module<BE>,
    host_module: &Module<HostBytesBackend>,
) where
    BE: TestContextBackend,
    Module<BE>: TestContextModule<BE> + LinearTransformationOps<BE>,
    F: TestScalar,
    E: NegacyclicFFT<F> + NegacyclicFFTNew<F>,
    for<'a> ScratchArena<'a, BE>: ScratchAvailable,
{
    let m = params.n / 2;
    let encoder = Encoder::<E>::new(m).unwrap();
    let (v_re, v_im) = test_vector_1::<F>(m);
    let (sk_raw, sk) = gen_sk_with_raw(&params, module, host_module, [10u8; 32]);
    let mut scratch = alloc_scratch(&params, module);

    let diag_indices = &[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11];
    let (lt_one_shot, reference_diags) = build_transform::<BE, F, E>(params, module, host_module, &encoder, diag_indices, 2);
    let (lt_for_prepare, _) = build_transform::<BE, F, E>(params, module, host_module, &encoder, diag_indices, 2);
    let prepared = prepare_linear_transform(module, &lt_for_prepare, &mut scratch.borrow());

    assert_eq!(prepared.giant_steps.len(), 6);
    assert_eq!(prepared.giant_steps.iter().filter(|gs| gs.rot != 0).count(), 5);

    let mut required_rotations = lt_one_shot.required_rotations();
    required_rotations.extend(prepared.required_rotations());
    required_rotations.sort_unstable();
    required_rotations.dedup();

    let mut atks = HashMap::new();
    for r in required_rotations {
        let atk = gen_atk(&params, module, r, &sk_raw, &mut scratch.borrow());
        atks.insert(r, atk);
    }

    let ct = ckks_encrypt(
        &params,
        module,
        host_module,
        &encoder,
        &sk,
        params.k,
        &v_re,
        &v_im,
        &mut scratch.borrow(),
    );
    let mut ct_one_shot = alloc_ct(&params, module, params.k);
    let mut ct_prepared = alloc_ct(&params, module, params.k);
    module
        .ckks_eval_linear_transformation_into(&mut ct_one_shot, &ct, &lt_one_shot, &atks, &mut scratch.borrow())
        .unwrap();
    module
        .ckks_eval_prepared_linear_transformation_into(
            &mut ct_prepared,
            &ct,
            &lt_for_prepare,
            &prepared,
            &atks,
            &mut scratch.borrow(),
        )
        .unwrap();

    assert_eq!(ct_one_shot.log_delta(), ct_prepared.log_delta());
    assert_eq!(ct_one_shot.log_budget(), ct_prepared.log_budget());

    let (one_shot_re, one_shot_im) = ckks_decrypt_decode(&params, module, &encoder, &ct_one_shot, &sk, &mut scratch.borrow());
    let (prepared_re, prepared_im) = ckks_decrypt_decode(&params, module, &encoder, &ct_prepared, &sk, &mut scratch.borrow());
    assert_precision(
        "linear_transformation_prepared_big_accumulator_stress re",
        &prepared_re,
        &one_shot_re,
        ct_one_shot.log_delta(),
        params.n,
    );
    assert_precision(
        "linear_transformation_prepared_big_accumulator_stress im",
        &prepared_im,
        &one_shot_im,
        ct_one_shot.log_delta(),
        params.n,
    );

    let (want_re, want_im) = reference_linear_transform(m, &v_re, &v_im, &reference_diags);
    assert_decrypt_precision(
        "linear_transformation_prepared_big_accumulator_stress_prepared",
        &params,
        module,
        &encoder,
        &ct_prepared,
        &sk,
        &want_re,
        &want_im,
        &mut scratch.borrow(),
    );
}

/// Prepared identity exercises final lazy normalization without automorphism keys.
pub fn test_linear_transformation_prepared_identity_no_keys<BE, F, E>(
    params: CKKSTestParams,
    module: &Module<BE>,
    host_module: &Module<HostBytesBackend>,
) where
    BE: TestContextBackend,
    Module<BE>: TestContextModule<BE> + LinearTransformationOps<BE>,
    F: TestScalar,
    E: NegacyclicFFT<F> + NegacyclicFFTNew<F>,
    for<'a> ScratchArena<'a, BE>: ScratchAvailable,
{
    let m = params.n / 2;
    let encoder = Encoder::<E>::new(m).unwrap();
    let (v_re, v_im) = test_vector_1::<F>(m);
    let (_sk_raw, sk) = gen_sk_with_raw(&params, module, host_module, [11u8; 32]);
    let mut scratch = alloc_scratch(&params, module);

    let (lt, reference_diags) = build_transform::<BE, F, E>(params, module, host_module, &encoder, &[0], 4);
    let prepared = prepare_linear_transform(module, &lt, &mut scratch.borrow());
    assert!(prepared.required_rotations().is_empty());
    let atks: HashMap<i64, TestAtk<BE>> = HashMap::new();

    let ct = ckks_encrypt(
        &params,
        module,
        host_module,
        &encoder,
        &sk,
        params.k,
        &v_re,
        &v_im,
        &mut scratch.borrow(),
    );
    let mut ct_res = alloc_ct(&params, module, params.k);
    module
        .ckks_eval_prepared_linear_transformation_into(&mut ct_res, &ct, &lt, &prepared, &atks, &mut scratch.borrow())
        .unwrap();

    assert_eq!(ct_res.log_delta(), ct.log_delta());

    let (want_re, want_im) = reference_linear_transform(m, &v_re, &v_im, &reference_diags);
    assert_decrypt_precision(
        "linear_transformation_prepared_identity_no_keys",
        &params,
        module,
        &encoder,
        &ct_res,
        &sk,
        &want_re,
        &want_im,
        &mut scratch.borrow(),
    );
}

/// Direct prepared schedules match a BSGS prepared schedule and plaintext baseline.
pub fn test_linear_transformation_prepared_direct_matches_bsgs<BE, F, E>(
    params: CKKSTestParams,
    module: &Module<BE>,
    host_module: &Module<HostBytesBackend>,
) where
    BE: TestContextBackend,
    Module<BE>: TestContextModule<BE> + LinearTransformationOps<BE>,
    F: TestScalar,
    E: NegacyclicFFT<F> + NegacyclicFFTNew<F>,
    for<'a> ScratchArena<'a, BE>: ScratchAvailable,
{
    let m = params.n / 2;
    let encoder = Encoder::<E>::new(m).unwrap();
    let (v_re, v_im) = test_vector_1::<F>(m);
    let (sk_raw, sk) = gen_sk_with_raw(&params, module, host_module, [12u8; 32]);
    let mut scratch = alloc_scratch(&params, module);

    let diag_indices = &[2, 5];
    let (lt_direct, reference_diags) = build_transform::<BE, F, E>(params, module, host_module, &encoder, diag_indices, 1);
    let (lt_bsgs, _) = build_transform::<BE, F, E>(params, module, host_module, &encoder, diag_indices, 3);
    let prepared_direct = prepare_linear_transform(module, &lt_direct, &mut scratch.borrow());
    let prepared_bsgs = prepare_linear_transform(module, &lt_bsgs, &mut scratch.borrow());

    assert_eq!(prepared_direct.baby_steps, vec![0]);
    assert!(
        prepared_direct
            .giant_steps
            .iter()
            .all(|gs| gs.diagonals.len() == 1 && gs.diagonals.contains_key(&0))
    );

    let mut required_rotations = prepared_direct.required_rotations();
    required_rotations.extend(prepared_bsgs.required_rotations());
    required_rotations.sort_unstable();
    required_rotations.dedup();

    let mut atks = HashMap::new();
    for r in required_rotations {
        let atk = gen_atk(&params, module, r, &sk_raw, &mut scratch.borrow());
        atks.insert(r, atk);
    }

    let ct = ckks_encrypt(
        &params,
        module,
        host_module,
        &encoder,
        &sk,
        params.k,
        &v_re,
        &v_im,
        &mut scratch.borrow(),
    );
    let mut ct_direct = alloc_ct(&params, module, params.k);
    let mut ct_bsgs = alloc_ct(&params, module, params.k);
    module
        .ckks_eval_prepared_linear_transformation_into(
            &mut ct_direct,
            &ct,
            &lt_direct,
            &prepared_direct,
            &atks,
            &mut scratch.borrow(),
        )
        .unwrap();
    module
        .ckks_eval_prepared_linear_transformation_into(&mut ct_bsgs, &ct, &lt_bsgs, &prepared_bsgs, &atks, &mut scratch.borrow())
        .unwrap();

    assert_eq!(ct_direct.log_delta(), ct_bsgs.log_delta());
    assert_eq!(ct_direct.log_budget(), ct_bsgs.log_budget());

    let (direct_re, direct_im) = ckks_decrypt_decode(&params, module, &encoder, &ct_direct, &sk, &mut scratch.borrow());
    let (bsgs_re, bsgs_im) = ckks_decrypt_decode(&params, module, &encoder, &ct_bsgs, &sk, &mut scratch.borrow());
    assert_precision(
        "linear_transformation_prepared_direct_matches_bsgs re",
        &direct_re,
        &bsgs_re,
        ct_direct.log_delta(),
        params.n,
    );
    assert_precision(
        "linear_transformation_prepared_direct_matches_bsgs im",
        &direct_im,
        &bsgs_im,
        ct_direct.log_delta(),
        params.n,
    );

    let (want_re, want_im) = reference_linear_transform(m, &v_re, &v_im, &reference_diags);
    assert_decrypt_precision(
        "linear_transformation_prepared_direct_matches_bsgs",
        &params,
        module,
        &encoder,
        &ct_direct,
        &sk,
        &want_re,
        &want_im,
        &mut scratch.borrow(),
    );
}

/// Prepared assign path matches prepared evaluation semantics.
pub fn test_linear_transformation_prepared_assign<BE, F, E>(
    params: CKKSTestParams,
    module: &Module<BE>,
    host_module: &Module<HostBytesBackend>,
) where
    BE: TestContextBackend,
    Module<BE>: TestContextModule<BE> + LinearTransformationOps<BE>,
    F: TestScalar,
    E: NegacyclicFFT<F> + NegacyclicFFTNew<F>,
    for<'a> ScratchArena<'a, BE>: ScratchAvailable,
{
    let m = params.n / 2;
    let encoder = Encoder::<E>::new(m).unwrap();
    let (v_re, v_im) = test_vector_1::<F>(m);
    let (sk_raw, sk) = gen_sk_with_raw(&params, module, host_module, [5u8; 32]);
    let mut scratch = alloc_scratch(&params, module);

    let (lt, reference_diags) = build_transform::<BE, F, E>(params, module, host_module, &encoder, &[1, 4, 6], 4);
    let prepared = prepare_linear_transform(module, &lt, &mut scratch.borrow());

    let mut atks = HashMap::new();
    for r in prepared.required_rotations() {
        let atk = gen_atk(&params, module, r, &sk_raw, &mut scratch.borrow());
        atks.insert(r, atk);
    }

    let mut ct = ckks_encrypt(
        &params,
        module,
        host_module,
        &encoder,
        &sk,
        params.k,
        &v_re,
        &v_im,
        &mut scratch.borrow(),
    );
    module
        .ckks_eval_prepared_linear_transformation_assign(&mut ct, &lt, &prepared, &atks, &mut scratch.borrow())
        .unwrap();

    let (want_re, want_im) = reference_linear_transform(m, &v_re, &v_im, &reference_diags);
    assert_decrypt_precision(
        "linear_transformation_prepared_assign",
        &params,
        module,
        &encoder,
        &ct,
        &sk,
        &want_re,
        &want_im,
        &mut scratch.borrow(),
    );
}

/// Missing keys are reported before prepared evaluation.
pub fn test_linear_transformation_prepared_missing_key_error<BE, F, E>(
    params: CKKSTestParams,
    module: &Module<BE>,
    host_module: &Module<HostBytesBackend>,
) where
    BE: TestContextBackend,
    Module<BE>: TestContextModule<BE> + LinearTransformationOps<BE>,
    F: TestScalar,
    E: NegacyclicFFT<F> + NegacyclicFFTNew<F>,
    for<'a> ScratchArena<'a, BE>: ScratchAvailable,
{
    let m = params.n / 2;
    let encoder = Encoder::<E>::new(m).unwrap();
    let (v_re, v_im) = test_vector_1::<F>(m);
    let (sk_raw, sk) = gen_sk_with_raw(&params, module, host_module, [6u8; 32]);
    let mut scratch = alloc_scratch(&params, module);

    let (lt, _) = build_transform::<BE, F, E>(params, module, host_module, &encoder, &[2, 5], 3);
    let prepared = prepare_linear_transform(module, &lt, &mut scratch.borrow());
    let mut atks = HashMap::new();
    for r in prepared.required_rotations() {
        let atk = gen_atk(&params, module, r, &sk_raw, &mut scratch.borrow());
        atks.insert(r, atk);
    }
    let missing = *prepared.required_rotations().first().unwrap();
    atks.remove(&missing);

    let ct = ckks_encrypt(
        &params,
        module,
        host_module,
        &encoder,
        &sk,
        params.k,
        &v_re,
        &v_im,
        &mut scratch.borrow(),
    );
    let mut ct_res = alloc_ct(&params, module, params.k);
    let err = module
        .ckks_eval_prepared_linear_transformation_into(&mut ct_res, &ct, &lt, &prepared, &atks, &mut scratch.borrow())
        .unwrap_err();
    let err = err.downcast_ref::<CKKSCompositionError>().unwrap();
    assert_eq!(
        err,
        &CKKSCompositionError::MissingAutomorphismKey {
            op: "linear_transformation",
            rotation: missing,
        }
    );
}

/// Many-prepared API evaluates several transforms over one input.
pub fn test_linear_transformation_many_prepared<BE, F, E>(
    params: CKKSTestParams,
    module: &Module<BE>,
    host_module: &Module<HostBytesBackend>,
) where
    BE: TestContextBackend,
    Module<BE>: TestContextModule<BE> + LinearTransformationOps<BE>,
    F: TestScalar,
    E: NegacyclicFFT<F> + NegacyclicFFTNew<F>,
    for<'a> ScratchArena<'a, BE>: ScratchAvailable,
{
    let m = params.n / 2;
    let encoder = Encoder::<E>::new(m).unwrap();
    let (v_re, v_im) = test_vector_1::<F>(m);
    let (sk_raw, sk) = gen_sk_with_raw(&params, module, host_module, [7u8; 32]);
    let mut scratch = alloc_scratch(&params, module);

    let (lt_a, reference_a) = build_transform::<BE, F, E>(params, module, host_module, &encoder, &[0, 3, 6], 3);
    let (lt_b, reference_b) = build_transform::<BE, F, E>(params, module, host_module, &encoder, &[1, 2, 5], 2);
    let transforms = vec![lt_a, lt_b];
    let prepared_transforms = transforms
        .iter()
        .map(|transform| prepare_linear_transform(module, transform, &mut scratch.borrow()))
        .collect::<Vec<_>>();

    let mut required_rotations = prepared_transforms
        .iter()
        .flat_map(|transform| transform.required_rotations())
        .collect::<Vec<_>>();
    required_rotations.sort_unstable();
    required_rotations.dedup();

    let mut atks = HashMap::new();
    for r in required_rotations {
        let atk = gen_atk(&params, module, r, &sk_raw, &mut scratch.borrow());
        atks.insert(r, atk);
    }

    let ct = ckks_encrypt(
        &params,
        module,
        host_module,
        &encoder,
        &sk,
        params.k,
        &v_re,
        &v_im,
        &mut scratch.borrow(),
    );

    let mut baby_steps = prepared_transforms
        .iter()
        .flat_map(|transform| transform.baby_steps.iter().copied())
        .collect::<Vec<_>>();
    baby_steps.sort_unstable();
    baby_steps.dedup();
    let prepared_babies = module
        .ckks_prepare_baby_rotations(&baby_steps, &ct, &atks, &mut scratch.borrow())
        .unwrap();
    assert_eq!(prepared_babies.baby_steps().collect::<Vec<_>>(), baby_steps);
    let _first_baby = prepared_babies.baby_step(0);

    let mut reused_output = alloc_ct(&params, module, params.k);
    module
        .ckks_eval_prepared_linear_transformation_with_babies_into(
            &mut reused_output,
            &ct,
            &transforms[0],
            &prepared_transforms[0],
            &prepared_babies,
            &atks,
            &mut scratch.borrow(),
        )
        .unwrap();
    let (want_re, want_im) = reference_linear_transform(m, &v_re, &v_im, &reference_a);
    assert_decrypt_precision(
        "linear_transformation_prepared_with_babies",
        &params,
        module,
        &encoder,
        &reused_output,
        &sk,
        &want_re,
        &want_im,
        &mut scratch.borrow(),
    );

    let mut outputs = vec![alloc_ct(&params, module, params.k), alloc_ct(&params, module, params.k)];
    module
        .ckks_eval_many_prepared_linear_transformations_into(
            &mut outputs,
            &ct,
            &transforms,
            &prepared_transforms,
            &atks,
            &mut scratch.borrow(),
        )
        .unwrap();

    let references = [reference_a, reference_b];
    for (i, (ct_res, reference_diags)) in outputs.iter().zip(references.iter()).enumerate() {
        let (want_re, want_im) = reference_linear_transform(m, &v_re, &v_im, reference_diags);
        assert_decrypt_precision(
            &format!("linear_transformation_many_prepared_{i}"),
            &params,
            module,
            &encoder,
            ct_res,
            &sk,
            &want_re,
            &want_im,
            &mut scratch.borrow(),
        );
    }
}

/// Sequential prepared API matches manual step-by-step evaluation.
pub fn test_linear_transformation_sequential_prepared<BE, F, E>(
    params: CKKSTestParams,
    module: &Module<BE>,
    host_module: &Module<HostBytesBackend>,
) where
    BE: TestContextBackend,
    Module<BE>: TestContextModule<BE> + LinearTransformationOps<BE>,
    F: TestScalar,
    E: NegacyclicFFT<F> + NegacyclicFFTNew<F>,
    for<'a> ScratchArena<'a, BE>: ScratchAvailable,
{
    let m = params.n / 2;
    let encoder = Encoder::<E>::new(m).unwrap();
    let (v_re, v_im) = test_vector_1::<F>(m);
    let (sk_raw, sk) = gen_sk_with_raw(&params, module, host_module, [13u8; 32]);
    let mut scratch = alloc_scratch(&params, module);

    let (lt_a, _) = build_transform::<BE, F, E>(params, module, host_module, &encoder, &[0, 2, 5], 3);
    let (lt_b, _) = build_transform::<BE, F, E>(params, module, host_module, &encoder, &[1, 3], 2);
    let transforms = vec![lt_a, lt_b];
    let prepared_transforms = transforms
        .iter()
        .map(|transform| prepare_linear_transform(module, transform, &mut scratch.borrow()))
        .collect::<Vec<_>>();

    let mut required_rotations = prepared_transforms
        .iter()
        .flat_map(|transform| transform.required_rotations())
        .collect::<Vec<_>>();
    required_rotations.sort_unstable();
    required_rotations.dedup();

    let mut atks = HashMap::new();
    for r in required_rotations {
        let atk = gen_atk(&params, module, r, &sk_raw, &mut scratch.borrow());
        atks.insert(r, atk);
    }

    let ct = ckks_encrypt(
        &params,
        module,
        host_module,
        &encoder,
        &sk,
        params.k,
        &v_re,
        &v_im,
        &mut scratch.borrow(),
    );

    let mut ct_seq = alloc_ct(&params, module, params.k);
    module
        .ckks_eval_sequential_prepared_linear_transformations_into(
            &mut ct_seq,
            &ct,
            &transforms,
            &prepared_transforms,
            &atks,
            &mut scratch.borrow(),
        )
        .unwrap();

    let mut ct_manual_step = alloc_ct(&params, module, params.k);
    let mut ct_manual = alloc_ct(&params, module, params.k);
    module
        .ckks_eval_prepared_linear_transformation_into(
            &mut ct_manual_step,
            &ct,
            &transforms[0],
            &prepared_transforms[0],
            &atks,
            &mut scratch.borrow(),
        )
        .unwrap();
    module
        .ckks_eval_prepared_linear_transformation_into(
            &mut ct_manual,
            &ct_manual_step,
            &transforms[1],
            &prepared_transforms[1],
            &atks,
            &mut scratch.borrow(),
        )
        .unwrap();

    assert_eq!(ct_seq.log_delta(), ct_manual.log_delta());
    assert_eq!(ct_seq.log_budget(), ct_manual.log_budget());

    let (seq_re, seq_im) = ckks_decrypt_decode(&params, module, &encoder, &ct_seq, &sk, &mut scratch.borrow());
    let (manual_re, manual_im) = ckks_decrypt_decode(&params, module, &encoder, &ct_manual, &sk, &mut scratch.borrow());
    assert_precision(
        "linear_transformation_sequential_prepared re",
        &seq_re,
        &manual_re,
        ct_seq.log_delta(),
        params.n,
    );
    assert_precision(
        "linear_transformation_sequential_prepared im",
        &seq_im,
        &manual_im,
        ct_seq.log_delta(),
        params.n,
    );
}

/// Identity diagonal requires no automorphism keys.
pub fn test_linear_transformation_identity_no_keys<BE, F, E>(
    params: CKKSTestParams,
    module: &Module<BE>,
    host_module: &Module<HostBytesBackend>,
) where
    BE: TestContextBackend,
    Module<BE>: TestContextModule<BE> + LinearTransformationOps<BE>,
    F: TestScalar,
    E: NegacyclicFFT<F> + NegacyclicFFTNew<F>,
    for<'a> ScratchArena<'a, BE>: ScratchAvailable,
{
    let m = params.n / 2;
    let encoder = Encoder::<E>::new(m).unwrap();
    let (v_re, v_im) = test_vector_1::<F>(m);
    let (_sk_raw, sk) = gen_sk_with_raw(&params, module, host_module, [2u8; 32]);
    let mut scratch = alloc_scratch(&params, module);

    let (lt, reference_diags) = build_transform::<BE, F, E>(params, module, host_module, &encoder, &[0], 4);
    assert!(lt.required_rotations().is_empty());
    let atks: HashMap<i64, TestAtk<BE>> = HashMap::new();

    let ct = ckks_encrypt(
        &params,
        module,
        host_module,
        &encoder,
        &sk,
        params.k,
        &v_re,
        &v_im,
        &mut scratch.borrow(),
    );
    let mut ct_res = alloc_ct(&params, module, params.k);
    module
        .ckks_eval_linear_transformation_into(&mut ct_res, &ct, &lt, &atks, &mut scratch.borrow())
        .unwrap();

    let (want_re, want_im) = reference_linear_transform(m, &v_re, &v_im, &reference_diags);
    assert_decrypt_precision(
        "linear_transformation_identity_no_keys",
        &params,
        module,
        &encoder,
        &ct_res,
        &sk,
        &want_re,
        &want_im,
        &mut scratch.borrow(),
    );
}

/// Missing required automorphism keys are reported before evaluation.
pub fn test_linear_transformation_missing_key_error<BE, F, E>(
    params: CKKSTestParams,
    module: &Module<BE>,
    host_module: &Module<HostBytesBackend>,
) where
    BE: TestContextBackend,
    Module<BE>: TestContextModule<BE> + LinearTransformationOps<BE>,
    F: TestScalar,
    E: NegacyclicFFT<F> + NegacyclicFFTNew<F>,
    for<'a> ScratchArena<'a, BE>: ScratchAvailable,
{
    let m = params.n / 2;
    let encoder = Encoder::<E>::new(m).unwrap();
    let (v_re, v_im) = test_vector_1::<F>(m);
    let (sk_raw, sk) = gen_sk_with_raw(&params, module, host_module, [3u8; 32]);
    let mut scratch = alloc_scratch(&params, module);

    let (lt, _) = build_transform::<BE, F, E>(params, module, host_module, &encoder, &[1, 3], 2);
    let mut atks = HashMap::new();
    for r in lt.required_rotations() {
        let atk = gen_atk(&params, module, r, &sk_raw, &mut scratch.borrow());
        atks.insert(r, atk);
    }
    let missing = *lt.required_rotations().first().unwrap();
    atks.remove(&missing);

    let ct = ckks_encrypt(
        &params,
        module,
        host_module,
        &encoder,
        &sk,
        params.k,
        &v_re,
        &v_im,
        &mut scratch.borrow(),
    );
    let mut ct_res = alloc_ct(&params, module, params.k);
    let err = module
        .ckks_eval_linear_transformation_into(&mut ct_res, &ct, &lt, &atks, &mut scratch.borrow())
        .unwrap_err();
    let err = err.downcast_ref::<CKKSCompositionError>().unwrap();
    assert_eq!(
        err,
        &CKKSCompositionError::MissingAutomorphismKey {
            op: "linear_transformation",
            rotation: missing,
        }
    );
}
