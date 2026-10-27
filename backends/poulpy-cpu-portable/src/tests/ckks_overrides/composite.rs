//! Sums, multiply-accumulate, affine forms and approximations enter the selected
//! constituents, are sized by their queries and match the portable backend.
use super::OverrideBackend;
use super::carry::{
    ADD_ASSIGN_CALLS, ADD_ASSIGN_EXTRA, ADD_CONST_CALLS, ADD_PT_VEC_CALLS, ADD_PT_VEC_EXTRA, SUB_ASSIGN_CALLS, SUB_ASSIGN_EXTRA,
};
use crate::FFT64Portable;
use poulpy_ckks::{
    CKKSCompositionError, CKKSLayout, GLWEPlaintextMeta, Scale, SetCKKSInfos, SlotsKind,
    api::{
        CKKSAddManyOps, CKKSAddOps, CKKSAffineOps, CKKSAllOpsTmpBytes, CKKSApproximationOps, CKKSCopyOps, CKKSMulAddOps,
        CKKSMulOps, CKKSMulSubOps, CKKSSubOps,
    },
    layouts::{CKKSModuleAlloc, CKKSPlaintextOwned, PolynomialApproximation},
    polynomial::BSGSPolynomial,
    test_suite::{
        CKKSTestParams,
        parity::{
            helpers::{Snapshot, fixture_ciphertext, fixture_plaintext, snapshot, with_scratch},
            layout,
        },
    },
};
use poulpy_core::{
    GLWEBytesOf, GLWEMaskFill, GLWENormalize, GLWETensorKeyEncryptSk,
    api::{GLWESecretSampling, GLWETensorKeyPreparedFactory},
    layouts::{GLWETensorKeyLayout, LWEInfos, ModuleCoreAlloc},
};
use poulpy_hal::{
    api::{ScratchOwnedAlloc, ScratchOwnedBorrow},
    layouts::{Backend, Module, ScratchOwned},
    source::Source,
};
use std::cell::Cell;

fn params() -> CKKSTestParams {
    CKKSTestParams {
        n: 64,
        base2k: 16,
        k: 64,
        prec_meta: GLWEPlaintextMeta {
            scale: Scale::Log(16),
            ..Default::default()
        },
        prec_log_budget: 16,
        hw: 8,
        dsize: 1,
        rank: 1,
    }
}

fn source_layout() -> CKKSLayout {
    let b = params().base2k;
    layout(params(), 1, 3 * b + 5, b - 1, 0, SlotsKind::Complex)
}

fn plaintext_layout() -> CKKSLayout {
    let b = params().base2k;
    layout(params(), 0, b + 3, b - 1, 0, SlotsKind::Complex)
}

fn tensor_key_layout() -> GLWETensorKeyLayout {
    GLWETensorKeyLayout {
        n: 64usize.into(),
        base2k: 16usize.into(),
        k_aux: 16usize.into(),
        rank: 1usize.into(),
        dnum: 4usize.into(),
        k_digit: 16usize.into(),
    }
}

/// Counts the calls `run` makes through `calls`.
fn count(calls: &'static std::thread::LocalKey<Cell<usize>>, run: impl FnOnce()) -> usize {
    calls.set(0);
    run();
    calls.get()
}

#[test]
fn add_many_enters_the_selected_additions() {
    let module_ref = Module::<FFT64Portable>::new(64);
    let module_test = Module::<OverrideBackend>::new(64);
    let out_layout = layout(
        params(),
        1,
        2 * params().base2k + 3,
        params().base2k - 1,
        0,
        SlotsKind::Complex,
    );
    let sum = |module_ref: &Module<FFT64Portable>, terms: usize| -> Snapshot {
        let inputs: Vec<_> = (0..terms)
            .map(|i| fixture_ciphertext(module_ref, &source_layout(), 40 + i as u8))
            .collect();
        let mut out = fixture_ciphertext(module_ref, &out_layout, 31);
        let bytes = module_ref.ckks_add_many_tmp_bytes(&out, &inputs[0]);
        with_scratch::<FFT64Portable, _>(bytes, |scratch| {
            module_ref.ckks_add_many(usize::MAX, &mut out, &inputs.iter().collect::<Vec<_>>(), scratch)
        })
        .unwrap();
        snapshot::<FFT64Portable, _>(&out)
    };

    // Several inputs enter the selected additions, one input the selected copy, each on
    // exactly the query, which covers both.
    for terms in [3usize, 1] {
        let inputs: Vec<_> = (0..terms)
            .map(|i| fixture_ciphertext(&module_test, &source_layout(), 40 + i as u8))
            .collect();
        let mut out = fixture_ciphertext(&module_test, &out_layout, 31);
        let bytes = module_test.ckks_add_many_tmp_bytes(&out, &inputs[0]);
        assert_eq!(
            bytes,
            module_test
                .ckks_add_tmp_bytes(out.max_size())
                .max(module_test.glwe_normalize_tmp_bytes())
                .max(module_test.ckks_copy_tmp_bytes(&out, &inputs[0]))
        );
        super::MAX_COPY_CAPACITY.set(0);
        let calls = count(&ADD_ASSIGN_CALLS, || {
            with_scratch::<OverrideBackend, _>(bytes, |scratch| {
                module_test.ckks_add_many(usize::MAX, &mut out, &inputs.iter().collect::<Vec<_>>(), scratch)
            })
            .unwrap();
        });
        assert_eq!(calls, terms.saturating_sub(2));
        assert_eq!(super::MAX_COPY_CAPACITY.get(), if terms == 1 { out.max_size() } else { 0 });
        assert_eq!(snapshot::<OverrideBackend, _>(&out), sum(&module_ref, terms));
    }

    let mut out = fixture_ciphertext(&module_test, &out_layout, 31);
    let bytes = module_test.ckks_add_many_tmp_bytes(&out, &out);
    let unchanged = snapshot::<OverrideBackend, _>(&out);
    let calls = count(&ADD_ASSIGN_CALLS, || {
        let empty: &[&poulpy_ckks::layouts::CKKSCiphertextOwned<OverrideBackend>] = &[];
        let err = with_scratch::<OverrideBackend, _>(bytes, |scratch| {
            module_test.ckks_add_many(usize::MAX, &mut out, empty, scratch)
        })
        .unwrap_err();
        assert_eq!(
            err.composition(),
            Some(&CKKSCompositionError::InvalidOperand {
                op: "ckks_add_many",
                reason: "a must contain at least one ciphertext".into(),
            })
        );
    });
    assert_eq!(calls, 0);
    assert_eq!(snapshot::<OverrideBackend, _>(&out), unchanged);
}

/// Runs `op` on `module` with fixture operands at the query `query` reports, returning the output.
fn run_pt_form<BE, Q, O>(module: &Module<BE>, query: Q, op: O) -> Snapshot
where
    BE: Backend<ZnxWord = i64>,
    Module<BE>: GLWEMaskFill<BE> + CKKSModuleAlloc<BE>,
    Q: FnOnce(
        &poulpy_ckks::layouts::CKKSCiphertextOwned<BE>,
        &poulpy_ckks::layouts::CKKSCiphertextOwned<BE>,
        &CKKSPlaintextOwned<BE>,
    ) -> usize,
    O: FnOnce(
        &mut poulpy_ckks::layouts::CKKSCiphertextOwned<BE>,
        &poulpy_ckks::layouts::CKKSCiphertextOwned<BE>,
        &CKKSPlaintextOwned<BE>,
        &mut poulpy_hal::layouts::ScratchArena<'_, BE>,
    ) -> poulpy_ckks::CKKSResult<()>,
{
    let a = fixture_ciphertext(module, &source_layout(), 41);
    let pt = fixture_plaintext(module, &plaintext_layout(), 43);
    let mut out = fixture_ciphertext(module, &source_layout(), 31);
    let bytes = query(&out, &a, &pt);
    with_scratch::<BE, _>(bytes, |scratch| op(&mut out, &a, &pt, scratch)).unwrap();
    snapshot::<BE, _>(&out)
}

#[test]
fn multiply_accumulate_enters_the_selected_addition_and_subtraction() {
    let module_ref = Module::<FFT64Portable>::new(64);
    let module_test = Module::<OverrideBackend>::new(64);
    let a = fixture_ciphertext(&module_test, &source_layout(), 41);
    let pt = fixture_plaintext(&module_test, &plaintext_layout(), 43);
    let res = fixture_ciphertext(&module_test, &source_layout(), 31);
    let tsk = tensor_key_layout();
    let tmp = module_test.glwe_bytes_of_from_infos(&res);
    assert_eq!(
        module_test.ckks_mul_add_ct_tmp_bytes(&res, &a, &a, &tsk),
        tmp + module_test
            .ckks_mul_tmp_bytes(&res, &a, &a, &tsk)
            .max(module_test.ckks_add_tmp_bytes(res.size()))
    );
    assert_eq!(
        module_test.ckks_mul_sub_ct_tmp_bytes(&res, &a, &a, &tsk),
        tmp + module_test
            .ckks_mul_tmp_bytes(&res, &a, &a, &tsk)
            .max(module_test.ckks_sub_tmp_bytes(res.size()))
    );
    for (query, product, add) in [
        (
            module_test.ckks_mul_add_pt_vec_tmp_bytes(&res, &a, &pt),
            module_test.ckks_mul_pt_vec_tmp_bytes(&res, &a, &pt),
            module_test.ckks_add_tmp_bytes(res.size()),
        ),
        (
            module_test.ckks_mul_add_pt_const_tmp_bytes(&res, &a, &pt),
            module_test.ckks_mul_pt_const_tmp_bytes(&res, &a, &pt),
            module_test.ckks_add_tmp_bytes(res.size()),
        ),
        (
            module_test.ckks_mul_sub_pt_vec_tmp_bytes(&res, &a, &pt),
            module_test.ckks_mul_pt_vec_tmp_bytes(&res, &a, &pt),
            module_test.ckks_sub_tmp_bytes(res.size()),
        ),
        (
            module_test.ckks_mul_sub_pt_const_tmp_bytes(&res, &a, &pt),
            module_test.ckks_mul_pt_const_tmp_bytes(&res, &a, &pt),
            module_test.ckks_sub_tmp_bytes(res.size()),
        ),
    ] {
        assert_eq!(query, tmp + product.max(add));
    }
    assert_eq!(
        module_test.ckks_add_tmp_bytes(res.size()),
        module_ref.ckks_add_tmp_bytes(res.size()) + ADD_ASSIGN_EXTRA
    );
    assert_eq!(
        module_test.ckks_sub_tmp_bytes(res.size()),
        module_ref.ckks_sub_tmp_bytes(res.size()) + SUB_ASSIGN_EXTRA
    );

    macro_rules! probe {
        ($calls:ident, $query:ident, $op:ident $(, $coeff:expr)?) => {{
            let want = run_pt_form(
                &module_ref,
                |r, a, pt| module_ref.$query(r, a, pt),
                |r, a, pt, s| module_ref.$op(r, a, pt $(, $coeff)?, s),
            );
            let mut got = None;
            let calls = count(&$calls, || {
                got = Some(run_pt_form(
                    &module_test,
                    |r, a, pt| module_test.$query(r, a, pt),
                    |r, a, pt, s| module_test.$op(r, a, pt $(, $coeff)?, s),
                ));
            });
            assert_eq!(calls, 1, stringify!($op));
            assert_eq!(got.unwrap(), want, stringify!($op));
        }};
    }
    probe!(ADD_ASSIGN_CALLS, ckks_mul_add_pt_vec_tmp_bytes, ckks_mul_add_pt_vec_into);
    probe!(
        ADD_ASSIGN_CALLS,
        ckks_mul_add_pt_const_tmp_bytes,
        ckks_mul_add_pt_const_into,
        3
    );
    probe!(SUB_ASSIGN_CALLS, ckks_mul_sub_pt_vec_tmp_bytes, ckks_mul_sub_pt_vec_into);
    probe!(
        SUB_ASSIGN_CALLS,
        ckks_mul_sub_pt_const_tmp_bytes,
        ckks_mul_sub_pt_const_into,
        3
    );

    // The ciphertext forms under one tensor key, encrypted on the portable backend and prepared on each.
    let mut sk = module_ref.glwe_secret_alloc_from_infos(&layout(params(), 1, 64, 0, 0, SlotsKind::Complex));
    module_ref.glwe_secret_fill_ternary_hw(&mut sk, params().hw, &mut Source::new([1u8; 32]));
    let mut key = module_ref.glwe_tensor_key_alloc_from_infos(&tsk);
    let mut setup = ScratchOwned::<FFT64Portable>::alloc(
        module_ref
            .glwe_tensor_key_encrypt_sk_tmp_bytes(&tsk)
            .max(module_ref.glwe_tensor_key_prepare_tmp_bytes(&tsk)),
    );
    module_ref.glwe_tensor_key_encrypt_sk(
        &mut key,
        &sk,
        &mut Source::new([2u8; 32]),
        &mut Source::new([3u8; 32]),
        &mut setup.borrow(),
    );
    let mut key_ref = module_ref.alloc_tensor_key_prepared_from_infos(&tsk);
    module_ref.glwe_tensor_key_prepare(&mut key_ref, &key, &mut setup.borrow());
    let mut key_test = module_test.alloc_tensor_key_prepared_from_infos(&tsk);
    let mut setup_test = ScratchOwned::<OverrideBackend>::alloc(module_test.glwe_tensor_key_prepare_tmp_bytes(&tsk));
    module_test.glwe_tensor_key_prepare(&mut key_test, &key, &mut setup_test.borrow());
    for subtract in [false, true] {
        let a_ref = fixture_ciphertext(&module_ref, &source_layout(), 41);
        let mut want = fixture_ciphertext(&module_ref, &source_layout(), 31);
        let bytes = module_ref.ckks_mul_add_ct_tmp_bytes(&want, &a_ref, &a_ref, &tsk);
        with_scratch::<FFT64Portable, _>(bytes, |s| {
            if subtract {
                module_ref.ckks_mul_sub_ct_into(&mut want, &a_ref, &a_ref, &key_ref, s)
            } else {
                module_ref.ckks_mul_add_ct_into(&mut want, &a_ref, &a_ref, &key_ref, s)
            }
        })
        .unwrap();
        let mut got = fixture_ciphertext(&module_test, &source_layout(), 31);
        let calls = if subtract { &SUB_ASSIGN_CALLS } else { &ADD_ASSIGN_CALLS };
        let bytes = if subtract {
            module_test.ckks_mul_sub_ct_tmp_bytes(&got, &a, &a, &tsk)
        } else {
            module_test.ckks_mul_add_ct_tmp_bytes(&got, &a, &a, &tsk)
        };
        let entered = count(calls, || {
            with_scratch::<OverrideBackend, _>(bytes, |s| {
                if subtract {
                    module_test.ckks_mul_sub_ct_into(&mut got, &a, &a, &key_test, s)
                } else {
                    module_test.ckks_mul_add_ct_into(&mut got, &a, &a, &key_test, s)
                }
            })
            .unwrap();
        });
        assert_eq!(entered, 1, "subtract = {subtract}");
        assert_eq!(
            snapshot::<OverrideBackend, _>(&got),
            snapshot::<FFT64Portable, _>(&want),
            "subtract = {subtract}"
        );
    }

    // A product the input's budget cannot pay fails before the accumulation is entered.
    let b = params().base2k;
    let narrow = fixture_ciphertext(&module_test, &layout(params(), 1, b, b - 1, 0, SlotsKind::Complex), 42);
    let mut out = fixture_ciphertext(&module_test, &source_layout(), 31);
    let unchanged = snapshot::<OverrideBackend, _>(&out);
    let bytes = module_test.ckks_mul_add_pt_vec_tmp_bytes(&out, &narrow, &pt);
    let calls = count(&ADD_ASSIGN_CALLS, || {
        let err = with_scratch::<OverrideBackend, _>(bytes, |s| module_test.ckks_mul_add_pt_vec_into(&mut out, &narrow, &pt, s))
            .unwrap_err();
        assert!(
            matches!(
                err.composition(),
                Some(CKKSCompositionError::MultiplicationPrecisionUnderflow {
                    op: "ckks_mul_add_pt_vec_into",
                    ..
                })
            ),
            "{err}"
        );
    });
    assert_eq!(calls, 0);
    assert_eq!(snapshot::<OverrideBackend, _>(&out), unchanged);
}

#[test]
fn affine_forms_enter_the_selected_plaintext_additions() {
    let module_ref = Module::<FFT64Portable>::new(64);
    let module_test = Module::<OverrideBackend>::new(64);
    let a = fixture_ciphertext(&module_test, &source_layout(), 41);
    let pt = fixture_plaintext(&module_test, &plaintext_layout(), 43);
    let res = fixture_ciphertext(&module_test, &source_layout(), 31);
    assert_eq!(
        module_test.ckks_affine_pt_const_tmp_bytes(&res, &a, &pt),
        module_test
            .ckks_mul_pt_const_tmp_bytes(&res, &a, &pt)
            .max(module_test.ckks_add_pt_const_tmp_bytes(res.size()))
    );
    assert_eq!(
        module_test.ckks_affine_pt_vec_tmp_bytes(&res, &a, &pt),
        module_test
            .ckks_mul_pt_vec_tmp_bytes(&res, &a, &pt)
            .max(module_test.ckks_add_pt_vec_tmp_bytes(res.size()))
    );
    assert_eq!(
        module_test.ckks_add_pt_vec_tmp_bytes(res.size()),
        module_ref.ckks_add_pt_vec_tmp_bytes(res.size()) + ADD_PT_VEC_EXTRA
    );

    let affine_const = |r: &Module<FFT64Portable>| {
        run_pt_form(
            r,
            |res, a, pt| r.ckks_affine_pt_const_tmp_bytes(res, a, pt),
            |res, a, pt, s| r.ckks_affine_pt_const_into(res, a, pt, 1, 3, s),
        )
    };
    let mut got = None;
    let calls = count(&ADD_CONST_CALLS, || {
        got = Some(run_pt_form(
            &module_test,
            |res, a, pt| module_test.ckks_affine_pt_const_tmp_bytes(res, a, pt),
            |res, a, pt, s| module_test.ckks_affine_pt_const_into(res, a, pt, 1, 3, s),
        ));
    });
    assert_eq!(calls, 1);
    assert_eq!(got.unwrap(), affine_const(&module_ref));

    let affine_vec = |r: &Module<FFT64Portable>| {
        run_pt_form(
            r,
            |res, a, pt| r.ckks_affine_pt_vec_tmp_bytes(res, a, pt),
            |res, a, pt, s| r.ckks_affine_pt_vec_into(res, a, pt, pt, s),
        )
    };
    let mut got = None;
    let calls = count(&ADD_PT_VEC_CALLS, || {
        got = Some(run_pt_form(
            &module_test,
            |res, a, pt| module_test.ckks_affine_pt_vec_tmp_bytes(res, a, pt),
            |res, a, pt, s| module_test.ckks_affine_pt_vec_into(res, a, pt, pt, s),
        ));
    });
    assert_eq!(calls, 1);
    assert_eq!(got.unwrap(), affine_vec(&module_ref));

    // The in-place forms, from `res` itself.
    for (calls, vec) in [(&ADD_CONST_CALLS, false), (&ADD_PT_VEC_CALLS, true)] {
        let assign = |r: &Module<FFT64Portable>| {
            run_pt_form(
                r,
                |res, _, pt| {
                    if vec {
                        r.ckks_affine_pt_vec_tmp_bytes(res, res, pt)
                    } else {
                        r.ckks_affine_pt_const_tmp_bytes(res, res, pt)
                    }
                },
                |res, _, pt, s| {
                    if vec {
                        r.ckks_affine_pt_vec_assign(res, pt, pt, s)
                    } else {
                        r.ckks_affine_pt_const_assign(res, pt, 1, 3, s)
                    }
                },
            )
        };
        let mut got = None;
        let entered = count(calls, || {
            got = Some(run_pt_form(
                &module_test,
                |res, _, pt| {
                    if vec {
                        module_test.ckks_affine_pt_vec_tmp_bytes(res, res, pt)
                    } else {
                        module_test.ckks_affine_pt_const_tmp_bytes(res, res, pt)
                    }
                },
                |res, _, pt, s| {
                    if vec {
                        module_test.ckks_affine_pt_vec_assign(res, pt, pt, s)
                    } else {
                        module_test.ckks_affine_pt_const_assign(res, pt, 1, 3, s)
                    }
                },
            ));
        });
        assert_eq!(entered, 1, "vec = {vec}");
        assert_eq!(got.unwrap(), assign(&module_ref), "vec = {vec}");
    }
}

/// A linear polynomial behind a plaintext interval map, its plaintexts allocated on `module` and filled with uniform
/// digits from fixed seeds.
fn approximation<BE: Backend<ZnxWord = i64>>(module: &Module<BE>) -> PolynomialApproximation<CKKSPlaintextOwned<BE>>
where
    Module<BE>: CKKSModuleAlloc<BE> + GLWEMaskFill<BE>,
{
    use poulpy_ckks::polynomial::{Basis, Polynomial, SplitStrategy};
    let seed = Cell::new(50u8);
    let coeffs = |count: usize| -> anyhow::Result<CKKSPlaintextOwned<BE>> {
        let mut pt = module.ckks_pt_coeffs_alloc(count, 16usize.into(), 32usize.into());
        pt.set_meta(GLWEPlaintextMeta {
            scale: Scale::Log(8),
            ..Default::default()
        });
        seed.set(seed.get() + 1);
        module.fill_glwe_from_source(&mut pt, &mut Source::new([seed.get(); 32]));
        Ok(pt)
    };
    let poly: BSGSPolynomial<CKKSPlaintextOwned<BE>> = Polynomial::new(Basis::Monomial, vec![0.0f64, 1.0])
        .decompose_bsgs_with(SplitStrategy::MinDepth, |c| coeffs(c.len()))
        .unwrap();
    PolynomialApproximation {
        poly,
        affine: Some(coeffs(2).unwrap()),
        scale_pow2: None,
        coeff_log_delta: 8,
    }
}

/// The approximation input layout: 64 bits at scale 16.
fn approximation_input() -> CKKSLayout {
    layout(params(), 1, 64, 16, 0, SlotsKind::Complex)
}

#[test]
fn approximations_enter_the_selected_map_and_polynomial_evaluation() {
    let module_ref = Module::<FFT64Portable>::new(64);
    let module_test = Module::<OverrideBackend>::new(64);
    let tsk = tensor_key_layout();
    let evaluate = |r: &Module<FFT64Portable>| {
        let approx = approximation(r);
        let src = fixture_ciphertext(r, &approximation_input(), 41);
        let mut res = fixture_ciphertext(r, &approximation_input(), 31);
        let bytes = r.ckks_approximation_tmp_bytes(&res, &src, &approx, &tsk);
        with_scratch::<FFT64Portable, _>(bytes, |s| {
            r.ckks_eval_approximation(&mut res, &src, &approx, &super::NoTensorKey, s)
        })
        .unwrap();
        snapshot::<FFT64Portable, _>(&res)
    };

    let approx = approximation(&module_test);
    let src = fixture_ciphertext(&module_test, &approximation_input(), 41);
    let mut res = fixture_ciphertext(&module_test, &approximation_input(), 31);
    let affine = approx.affine.as_ref().unwrap();
    let coeffs = approx.poly.baby_step(0);
    let eval = module_test
        .ckks_all_ops_tmp_bytes(&res, coeffs, &tsk)
        .max(module_test.ckks_all_ops_tmp_bytes(&src, coeffs, &tsk));
    let bytes = module_test.ckks_approximation_tmp_bytes(&res, &src, &approx, &tsk);
    assert_eq!(
        bytes,
        OverrideBackend::scratch_aligned(module_test.glwe_bytes_of_from_infos(&src))
            + eval.max(module_test.ckks_affine_pt_const_tmp_bytes(&src, &src, affine))
    );
    super::REAL_CALLS.set(0);
    let calls = count(&ADD_CONST_CALLS, || {
        with_scratch::<OverrideBackend, _>(bytes, |s| {
            module_test.ckks_eval_approximation(&mut res, &src, &approx, &super::NoTensorKey, s)
        })
        .unwrap();
    });
    assert_eq!(calls, 1);
    assert_eq!(super::REAL_CALLS.get(), 1);
    assert_eq!(snapshot::<OverrideBackend, _>(&res), evaluate(&module_ref));

    // An input whose budget cannot pay the approximation is refused before any evaluation.
    let short = fixture_ciphertext(&module_test, &layout(params(), 1, 20, 16, 0, SlotsKind::Complex), 42);
    let unchanged = snapshot::<OverrideBackend, _>(&res);
    super::REAL_CALLS.set(0);
    let calls = count(&ADD_CONST_CALLS, || {
        let err = with_scratch::<OverrideBackend, _>(bytes, |s| {
            module_test.ckks_eval_approximation(&mut res, &short, &approx, &super::NoTensorKey, s)
        })
        .unwrap_err();
        assert_eq!(
            err.composition(),
            Some(&CKKSCompositionError::InsufficientHomomorphicCapacity {
                op: "ckks_eval_approximation",
                available_log_budget: 4,
                required_bits: approx.consumed_bits(16),
            })
        );
    });
    assert_eq!(calls, 0);
    assert_eq!(super::REAL_CALLS.get(), 0);
    assert_eq!(snapshot::<OverrideBackend, _>(&res), unchanged);
}
