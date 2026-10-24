//! Sums, multiply-accumulate, affine forms and approximations enter the selected
//! constituents, are sized by their queries and match the portable backend.
use super::OverrideBackend;
use super::carry::{
    ADD_ASSIGN_CALLS, ADD_ASSIGN_EXTRA, ADD_CONST_CALLS, ADD_PT_VEC_CALLS, ADD_PT_VEC_EXTRA, SUB_ASSIGN_CALLS, SUB_ASSIGN_EXTRA,
};
use crate::FFT64Portable;
use poulpy_ckks::{
    CKKSLayout, GLWEPlaintextMeta, Scale, SetCKKSInfos, SlotsKind,
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
    GLWEBytesOf, GLWEMaskFill,
    layouts::{GLWETensorKeyLayout, LWEInfos},
};
use poulpy_hal::layouts::{Backend, Module};
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
    let reference = Module::<FFT64Portable>::new(64);
    let module = Module::<OverrideBackend>::new(64);
    let out_layout = layout(
        params(),
        1,
        2 * params().base2k + 3,
        params().base2k - 1,
        0,
        SlotsKind::Complex,
    );
    let sum = |module_r: &Module<FFT64Portable>, terms: usize| -> Snapshot {
        let inputs: Vec<_> = (0..terms)
            .map(|i| fixture_ciphertext(module_r, &source_layout(), 40 + i as u8))
            .collect();
        let mut out = fixture_ciphertext(module_r, &out_layout, 31);
        let bytes = module_r.ckks_add_many_tmp_bytes(&out, &inputs[0]);
        with_scratch::<FFT64Portable, _>(bytes, |scratch| {
            module_r.ckks_add_many(&mut out, &inputs.iter().collect::<Vec<_>>(), scratch)
        })
        .unwrap();
        snapshot::<FFT64Portable, _>(&out)
    };

    // Several inputs enter the selected additions, one input the selected copy, each on
    // exactly the query, which covers both.
    for terms in [3usize, 1] {
        let inputs: Vec<_> = (0..terms)
            .map(|i| fixture_ciphertext(&module, &source_layout(), 40 + i as u8))
            .collect();
        let mut out = fixture_ciphertext(&module, &out_layout, 31);
        let bytes = module.ckks_add_many_tmp_bytes(&out, &inputs[0]);
        assert_eq!(
            bytes,
            module
                .ckks_add_tmp_bytes(out.max_size())
                .max(module.ckks_copy_tmp_bytes(&out, &inputs[0]))
        );
        super::MAX_COPY_CAPACITY.set(0);
        let calls = count(&ADD_ASSIGN_CALLS, || {
            with_scratch::<OverrideBackend, _>(bytes, |scratch| {
                module.ckks_add_many(&mut out, &inputs.iter().collect::<Vec<_>>(), scratch)
            })
            .unwrap();
        });
        assert_eq!(calls, terms.saturating_sub(2));
        assert_eq!(super::MAX_COPY_CAPACITY.get(), if terms == 1 { out.max_size() } else { 0 });
        assert_eq!(snapshot::<OverrideBackend, _>(&out), sum(&reference, terms));
    }

    let mut out = fixture_ciphertext(&module, &out_layout, 31);
    let bytes = module.ckks_add_many_tmp_bytes(&out, &out);
    let unchanged = snapshot::<OverrideBackend, _>(&out);
    let calls = count(&ADD_ASSIGN_CALLS, || {
        let empty: &[&poulpy_ckks::layouts::CKKSCiphertextOwned<OverrideBackend>] = &[];
        assert!(with_scratch::<OverrideBackend, _>(bytes, |scratch| module.ckks_add_many(&mut out, empty, scratch)).is_err());
    });
    assert_eq!(calls, 0);
    assert_eq!(snapshot::<OverrideBackend, _>(&out), unchanged);
}

/// Runs `op` on `module` with fixture operands at the query `query` reports, returning the output.
fn run_pt_form<B, Q, O>(module: &Module<B>, query: Q, op: O) -> Snapshot
where
    B: Backend<ZnxWord = i64>,
    Module<B>: GLWEMaskFill<B> + CKKSModuleAlloc<B>,
    Q: FnOnce(
        &poulpy_ckks::layouts::CKKSCiphertextOwned<B>,
        &poulpy_ckks::layouts::CKKSCiphertextOwned<B>,
        &CKKSPlaintextOwned<B>,
    ) -> usize,
    O: FnOnce(
        &mut poulpy_ckks::layouts::CKKSCiphertextOwned<B>,
        &poulpy_ckks::layouts::CKKSCiphertextOwned<B>,
        &CKKSPlaintextOwned<B>,
        &mut poulpy_hal::layouts::ScratchArena<'_, B>,
    ) -> poulpy_ckks::CKKSResult<()>,
{
    let a = fixture_ciphertext(module, &source_layout(), 41);
    let pt = fixture_plaintext(module, &plaintext_layout(), 43);
    let mut out = fixture_ciphertext(module, &source_layout(), 31);
    let bytes = query(&out, &a, &pt);
    with_scratch::<B, _>(bytes, |scratch| op(&mut out, &a, &pt, scratch)).unwrap();
    snapshot::<B, _>(&out)
}

#[test]
fn multiply_accumulate_enters_the_selected_addition_and_subtraction() {
    let reference = Module::<FFT64Portable>::new(64);
    let module = Module::<OverrideBackend>::new(64);
    let a = fixture_ciphertext(&module, &source_layout(), 41);
    let pt = fixture_plaintext(&module, &plaintext_layout(), 43);
    let res = fixture_ciphertext(&module, &source_layout(), 31);
    let tsk = tensor_key_layout();
    let tmp = module.glwe_bytes_of_from_infos(&res);
    assert_eq!(
        module.ckks_mul_add_ct_tmp_bytes(&res, &a, &a, &tsk),
        tmp + module
            .ckks_mul_tmp_bytes(&res, &a, &a, &tsk)
            .max(module.ckks_add_tmp_bytes(res.size()))
    );
    assert_eq!(
        module.ckks_mul_sub_ct_tmp_bytes(&res, &a, &a, &tsk),
        tmp + module
            .ckks_mul_tmp_bytes(&res, &a, &a, &tsk)
            .max(module.ckks_sub_tmp_bytes(res.size()))
    );
    for (query, product, add) in [
        (
            module.ckks_mul_add_pt_vec_tmp_bytes(&res, &a, &pt),
            module.ckks_mul_pt_vec_tmp_bytes(&res, &a, &pt),
            module.ckks_add_tmp_bytes(res.size()),
        ),
        (
            module.ckks_mul_add_pt_const_tmp_bytes(&res, &a, &pt),
            module.ckks_mul_pt_const_tmp_bytes(&res, &a, &pt),
            module.ckks_add_tmp_bytes(res.size()),
        ),
        (
            module.ckks_mul_sub_pt_vec_tmp_bytes(&res, &a, &pt),
            module.ckks_mul_pt_vec_tmp_bytes(&res, &a, &pt),
            module.ckks_sub_tmp_bytes(res.size()),
        ),
        (
            module.ckks_mul_sub_pt_const_tmp_bytes(&res, &a, &pt),
            module.ckks_mul_pt_const_tmp_bytes(&res, &a, &pt),
            module.ckks_sub_tmp_bytes(res.size()),
        ),
    ] {
        assert_eq!(query, tmp + product.max(add));
    }
    assert_eq!(
        module.ckks_add_tmp_bytes(res.size()),
        reference.ckks_add_tmp_bytes(res.size()) + ADD_ASSIGN_EXTRA
    );
    assert_eq!(
        module.ckks_sub_tmp_bytes(res.size()),
        reference.ckks_sub_tmp_bytes(res.size()) + SUB_ASSIGN_EXTRA
    );

    macro_rules! probe {
        ($calls:ident, $query:ident, $op:ident $(, $coeff:expr)?) => {{
            let want = run_pt_form(
                &reference,
                |r, a, pt| reference.$query(r, a, pt),
                |r, a, pt, s| reference.$op(r, a, pt $(, $coeff)?, s),
            );
            let mut got = None;
            let calls = count(&$calls, || {
                got = Some(run_pt_form(
                    &module,
                    |r, a, pt| module.$query(r, a, pt),
                    |r, a, pt, s| module.$op(r, a, pt $(, $coeff)?, s),
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

    // A product the input's budget cannot pay fails before the accumulation is entered.
    let b = params().base2k;
    let narrow = fixture_ciphertext(&module, &layout(params(), 1, b, b - 1, 0, SlotsKind::Complex), 42);
    let mut out = fixture_ciphertext(&module, &source_layout(), 31);
    let unchanged = snapshot::<OverrideBackend, _>(&out);
    let bytes = module.ckks_mul_add_pt_vec_tmp_bytes(&out, &narrow, &pt);
    let calls = count(&ADD_ASSIGN_CALLS, || {
        assert!(
            with_scratch::<OverrideBackend, _>(bytes, |s| module.ckks_mul_add_pt_vec_into(&mut out, &narrow, &pt, s)).is_err()
        );
    });
    assert_eq!(calls, 0);
    assert_eq!(snapshot::<OverrideBackend, _>(&out), unchanged);
}

#[test]
fn affine_forms_enter_the_selected_plaintext_additions() {
    let reference = Module::<FFT64Portable>::new(64);
    let module = Module::<OverrideBackend>::new(64);
    let a = fixture_ciphertext(&module, &source_layout(), 41);
    let pt = fixture_plaintext(&module, &plaintext_layout(), 43);
    let res = fixture_ciphertext(&module, &source_layout(), 31);
    assert_eq!(
        module.ckks_affine_pt_const_tmp_bytes(&res, &a, &pt),
        module
            .ckks_mul_pt_const_tmp_bytes(&res, &a, &pt)
            .max(module.ckks_add_pt_const_tmp_bytes(res.size()))
    );
    assert_eq!(
        module.ckks_affine_pt_vec_tmp_bytes(&res, &a, &pt),
        module
            .ckks_mul_pt_vec_tmp_bytes(&res, &a, &pt)
            .max(module.ckks_add_pt_vec_tmp_bytes(res.size()))
    );
    assert_eq!(
        module.ckks_add_pt_vec_tmp_bytes(res.size()),
        reference.ckks_add_pt_vec_tmp_bytes(res.size()) + ADD_PT_VEC_EXTRA
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
            &module,
            |res, a, pt| module.ckks_affine_pt_const_tmp_bytes(res, a, pt),
            |res, a, pt, s| module.ckks_affine_pt_const_into(res, a, pt, 1, 3, s),
        ));
    });
    assert_eq!(calls, 1);
    assert_eq!(got.unwrap(), affine_const(&reference));

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
            &module,
            |res, a, pt| module.ckks_affine_pt_vec_tmp_bytes(res, a, pt),
            |res, a, pt, s| module.ckks_affine_pt_vec_into(res, a, pt, pt, s),
        ));
    });
    assert_eq!(calls, 1);
    assert_eq!(got.unwrap(), affine_vec(&reference));
}

/// A linear polynomial behind a plaintext interval map, its plaintexts allocated on `module`.
fn approximation<B: Backend>(module: &Module<B>) -> PolynomialApproximation<CKKSPlaintextOwned<B>>
where
    Module<B>: CKKSModuleAlloc<B>,
{
    use poulpy_ckks::polynomial::{Basis, Polynomial, SplitStrategy};
    let coeffs = |count: usize| -> anyhow::Result<CKKSPlaintextOwned<B>> {
        let mut pt = module.ckks_pt_coeffs_alloc(count, 16usize.into(), 32usize.into());
        pt.set_meta(GLWEPlaintextMeta {
            scale: Scale::Log(8),
            ..Default::default()
        });
        Ok(pt)
    };
    let poly: BSGSPolynomial<CKKSPlaintextOwned<B>> = Polynomial::new(Basis::Monomial, vec![0.0f64, 1.0])
        .decompose_bsgs_with(SplitStrategy::MinDepth, |c| coeffs(c.len()))
        .unwrap();
    PolynomialApproximation {
        poly,
        affine: Some(coeffs(2).unwrap()),
        scale_pow2: None,
        coeff_log_delta: 8,
    }
}

#[test]
fn approximations_enter_the_selected_map_and_polynomial_evaluation() {
    let reference = Module::<FFT64Portable>::new(64);
    let module = Module::<OverrideBackend>::new(64);
    let tsk = tensor_key_layout();
    let evaluate = |r: &Module<FFT64Portable>| {
        let approx = approximation(r);
        let mut src = r.ckks_ciphertext_alloc(16usize.into(), 64usize.into());
        src.set_meta(GLWEPlaintextMeta {
            scale: Scale::Log(16),
            ..Default::default()
        });
        let mut res = r.ckks_ciphertext_alloc_from_infos(&src);
        let bytes = r.ckks_approximation_tmp_bytes(&res, &src, &tsk, &approx);
        with_scratch::<FFT64Portable, _>(bytes, |s| {
            r.ckks_eval_approximation(&mut res, &src, &approx, &super::NoTensorKey, s)
        })
        .unwrap();
        snapshot::<FFT64Portable, _>(&res)
    };

    let approx = approximation(&module);
    let mut src = module.ckks_ciphertext_alloc(16usize.into(), 64usize.into());
    src.set_meta(GLWEPlaintextMeta {
        scale: Scale::Log(16),
        ..Default::default()
    });
    let mut res = module.ckks_ciphertext_alloc_from_infos(&src);
    let affine = approx.affine.as_ref().unwrap();
    let coeffs = approx.poly.baby_step(0);
    let eval = module
        .ckks_all_ops_tmp_bytes(&res, &tsk, coeffs)
        .max(module.ckks_all_ops_tmp_bytes(&src, &tsk, coeffs));
    let bytes = module.ckks_approximation_tmp_bytes(&res, &src, &tsk, &approx);
    assert_eq!(
        bytes,
        module.glwe_bytes_of_from_infos(&src) + eval.max(module.ckks_affine_pt_const_tmp_bytes(&src, &src, affine))
    );
    super::REAL_CALLS.set(0);
    let calls = count(&ADD_CONST_CALLS, || {
        with_scratch::<OverrideBackend, _>(bytes, |s| {
            module.ckks_eval_approximation(&mut res, &src, &approx, &super::NoTensorKey, s)
        })
        .unwrap();
    });
    assert_eq!(calls, 1);
    assert_eq!(super::REAL_CALLS.get(), 1);
    assert_eq!(snapshot::<OverrideBackend, _>(&res), evaluate(&reference));

    // An input whose budget cannot pay the approximation is refused before any evaluation.
    let mut short = module.ckks_ciphertext_alloc(16usize.into(), 20usize.into());
    short.set_meta(GLWEPlaintextMeta {
        scale: Scale::Log(16),
        ..Default::default()
    });
    let unchanged = snapshot::<OverrideBackend, _>(&res);
    super::REAL_CALLS.set(0);
    let calls = count(&ADD_CONST_CALLS, || {
        assert!(
            with_scratch::<OverrideBackend, _>(bytes, |s| module.ckks_eval_approximation(
                &mut res,
                &short,
                &approx,
                &super::NoTensorKey,
                s
            ))
            .is_err()
        );
    });
    assert_eq!(calls, 0);
    assert_eq!(super::REAL_CALLS.get(), 0);
    assert_eq!(snapshot::<OverrideBackend, _>(&res), unchanged);
}
