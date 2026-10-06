//! Paired polynomial engines, one-shot derived schedules, and EvalMod.
use super::{
    helpers::{HostCiphertext, assert_untagged, fixture_ciphertext, host_ciphertext, with_scratch},
    keys::{key_layout, prepared_tensor_key},
};
use crate::{
    CKKSLayout, CKKSMeta, CoeffsMeta, SlotsKind,
    api::{
        CKKSAllOpsTmpBytes, CKKSComplexPolynomialEvaluationOps, CKKSEncodingHostOps, CKKSEncodingScalar, CKKSEvalModOps,
        CKKSPolynomialEvaluationOps,
    },
    layouts::{
        CKKSCiphertextOwned, CKKSModuleAlloc, CKKSPlaintextOwned,
        eval_mod::{EvalModPlan, EvalModType, compile_eval_mod},
    },
    oep::{CKKSEncodingImpl, CKKSImpl, CKKSPolynomialEvaluationImpl},
    polynomial::{Basis, ComplexBSGSPolynomial, EncodeBSGS, Parity, Polynomial, SplitStrategy},
    power_basis::{PowerBasis, PowerBasisGen},
    test_suite::CKKSTestParams,
};
use poulpy_core::{
    GLWEMaskFill,
    layouts::{GGLWELayout, GLWELayout, GLWETensorKeyPrepared, GLWETensorKeyPreparedFactory, LWEInfos},
};
use poulpy_hal::{
    layouts::{Backend, HostBytesBackend, Module},
    test_suite::upload_vec_znx,
};

fn upload<B: Backend<ZnxWord = i64>>(
    module: &Module<B>,
    pt: &CKKSPlaintextOwned<HostBytesBackend<B::Ring>>,
) -> CKKSPlaintextOwned<B> {
    let mut out = module.ckks_plaintext_alloc_from_infos(pt);
    *out.data_mut() = upload_vec_znx::<B>(pt.data());
    out
}

/// Shape, tensor key and scratch shared by the polynomial runs.
struct Setup<B: Backend> {
    layout: CKKSLayout,
    key: GGLWELayout,
    prepared_key: GLWETensorKeyPrepared<B::OwnedBuf, B>,
    coeff_meta: CoeffsMeta,
    bytes: usize,
    host: Module<HostBytesBackend<B::Ring>>,
}

fn setup<B>(params: CKKSTestParams, module: &Module<B>) -> Setup<B>
where
    B: Backend<ZnxWord = i64>,
    Module<B>: CKKSAllOpsTmpBytes<B> + GLWETensorKeyPreparedFactory<B> + GLWEMaskFill<B>,
{
    let b = params.base2k;
    let layout = CKKSLayout {
        glwe_layout: GLWELayout {
            n: module.n().into(),
            base2k: b.into(),
            k: (12 * b + 1).into(),
            rank: 1usize.into(),
        },
        meta: CKKSMeta {
            log_delta: 10,
            log_sparsity: 1,
            slots: SlotsKind::Real,
        },
    };
    let key = key_layout(module.n(), b, layout.k().as_usize(), 1, 1, 1);
    let prepared_key = prepared_tensor_key(module, &key, 113);
    let coeff_meta = CoeffsMeta::from_delta_budget(10, 10);
    let pt_layout = CKKSLayout {
        glwe_layout: GLWELayout {
            k: coeff_meta.k,
            rank: 0usize.into(),
            ..layout.glwe_layout
        },
        meta: coeff_meta.meta,
    };
    let bytes = module.ckks_all_ops_tmp_bytes(&layout, &key, &pt_layout);
    let host = Module::<HostBytesBackend<B::Ring>>::new(module.n() as u64);
    Setup {
        layout,
        key,
        prepared_key,
        coeff_meta,
        bytes,
        host,
    }
}

fn polynomial<F: CKKSEncodingScalar>(basis: Basis) -> Polynomial<F> {
    Polynomial::new(
        basis,
        [0.125, -0.25, 0.0625, 0.125]
            .into_iter()
            .map(|v| F::from_f64(v).unwrap())
            .collect(),
    )
}

fn folded<F: CKKSEncodingScalar>(basis: Basis, parity: Parity) -> Polynomial<F> {
    let coeffs = match parity {
        Parity::Even => vec![0.125, 0.0, 0.0625, 0.0, 0.03125],
        _ => vec![0.0, 0.125, 0.0, 0.0625, 0.0, 0.03125],
    };
    Polynomial::new_with_parity(basis, coeffs.into_iter().map(|v| F::from_f64(v).unwrap()).collect(), parity)
}

fn real_polynomials<B, F>(params: CKKSTestParams, module: &Module<B>) -> Vec<HostCiphertext>
where
    B: Backend<ZnxWord = i64> + CKKSPolynomialEvaluationImpl,
    F: CKKSEncodingScalar,
    Module<B>: CKKSAllOpsTmpBytes<B> + GLWETensorKeyPreparedFactory<B> + GLWEMaskFill<B>,
{
    let b = params.base2k;
    let Setup {
        layout,
        prepared_key,
        coeff_meta,
        bytes,
        host,
        ..
    } = setup(params, module);
    let mut results = Vec::new();
    for basis in [Basis::Monomial, Basis::Chebyshev] {
        let encoded = polynomial::<F>(basis)
            .encode_bsgs_with(&host, b.into(), coeff_meta, SplitStrategy::MinDepth)
            .unwrap()
            .map_baby_steps_ref(|pt| upload(module, pt));
        let input = fixture_ciphertext(module, &layout, 127);
        let before = host_ciphertext::<B, _>(&input);
        let mut powers = PowerBasis::new(basis, fixture_ciphertext(module, &layout, 127));
        with_scratch::<B, _>(bytes, |scratch| {
            powers.populate(
                encoded.degree(),
                encoded.log_split(),
                encoded.parity(),
                module,
                &prepared_key,
                scratch,
            )
        })
        .unwrap();
        let mut prepared_out = fixture_ciphertext(module, &layout, 131);
        with_scratch::<B, _>(bytes, |scratch| {
            module.ckks_eval_poly_real_const_coeffs_from_power_basis::<_, _, CKKSCiphertextOwned<B>, _, _>(
                &mut prepared_out,
                &encoded,
                &powers,
                &prepared_key,
                scratch,
            )
        })
        .unwrap();
        let mut one_shot = fixture_ciphertext(module, &layout, 131);
        with_scratch::<B, _>(bytes, |scratch| {
            module.ckks_eval_poly_real_const_coeffs(&mut one_shot, &input, &encoded, &prepared_key, scratch)
        })
        .unwrap();
        assert!(
            host_ciphertext::<B, _>(&prepared_out) == host_ciphertext::<B, _>(&one_shot),
            "prepared and one-shot real evaluation differ"
        );
        results.push(host_ciphertext::<B, _>(&one_shot));
        // Exercise x², x²·x, T₂, and T₂·x input folds through production dispatch.
        for parity in [Parity::Even, Parity::Odd] {
            let encoded = folded::<F>(basis, parity)
                .encode_bsgs_folded_with(&host, b.into(), coeff_meta, SplitStrategy::MinDepth)
                .unwrap()
                .map_baby_steps_ref(|pt| upload(module, pt));
            let mut out = fixture_ciphertext(module, &layout, 139);
            with_scratch::<B, _>(bytes, |scratch| {
                module.ckks_eval_poly_real_const_coeffs(&mut out, &input, &encoded, &prepared_key, scratch)
            })
            .unwrap();
            results.push(host_ciphertext::<B, _>(&out));
        }
        assert!(
            before == host_ciphertext::<B, _>(&input),
            "polynomial evaluation result differs"
        );
    }
    results
}

fn complex_polynomials_and_eval_mod<B, F>(params: CKKSTestParams, module: &Module<B>) -> Vec<HostCiphertext>
where
    B: Backend<ZnxWord = i64> + CKKSImpl + CKKSEncodingImpl<F>,
    F: CKKSEncodingScalar,
    Module<B>: CKKSAllOpsTmpBytes<B> + CKKSEvalModOps<B> + GLWETensorKeyPreparedFactory<B> + GLWEMaskFill<B>,
{
    let b = params.base2k;
    let Setup {
        layout,
        key,
        prepared_key,
        coeff_meta,
        bytes,
        host,
    } = setup(params, module);
    let mut results = Vec::new();
    for basis in [Basis::Monomial, Basis::Chebyshev] {
        let poly = polynomial::<F>(basis);
        let encode = || {
            poly.encode_bsgs_with(&host, b.into(), coeff_meta, SplitStrategy::MinDepth)
                .unwrap()
                .map_baby_steps_ref(|pt| upload(module, pt))
        };
        let complex = ComplexBSGSPolynomial {
            re: encode(),
            im: encode(),
        };
        let input = fixture_ciphertext(module, &layout, 127);
        let before = host_ciphertext::<B, _>(&input);
        let mut powers = PowerBasis::new(basis, fixture_ciphertext(module, &layout, 127));
        with_scratch::<B, _>(bytes, |scratch| {
            powers.populate(
                complex.re.degree(),
                complex.re.log_split(),
                complex.re.parity(),
                module,
                &prepared_key,
                scratch,
            )
        })
        .unwrap();
        let mut prepared_out = fixture_ciphertext(module, &layout, 137);
        with_scratch::<B, _>(bytes, |scratch| {
            module.ckks_eval_poly_complex_const_coeffs_from_power_basis::<_, _, CKKSCiphertextOwned<B>, _, _>(
                &mut prepared_out,
                &complex,
                &powers,
                &prepared_key,
                scratch,
            )
        })
        .unwrap();
        let mut one_shot = fixture_ciphertext(module, &layout, 137);
        with_scratch::<B, _>(bytes, |scratch| {
            module.ckks_eval_poly_complex_const_coeffs(&mut one_shot, &input, &complex, &prepared_key, scratch)
        })
        .unwrap();
        assert!(
            host_ciphertext::<B, _>(&prepared_out) == host_ciphertext::<B, _>(&one_shot),
            "prepared and one-shot complex evaluation differ"
        );
        results.push(host_ciphertext::<B, _>(&one_shot));
        for parity in [Parity::Even, Parity::Odd] {
            let poly = folded::<F>(basis, parity);
            let encode_folded = || {
                poly.encode_bsgs_folded_with(&host, b.into(), coeff_meta, SplitStrategy::MinDepth)
                    .unwrap()
                    .map_baby_steps_ref(|pt| upload(module, pt))
            };
            let complex = ComplexBSGSPolynomial {
                re: encode_folded(),
                im: encode_folded(),
            };
            let mut out = fixture_ciphertext(module, &layout, 143);
            with_scratch::<B, _>(bytes, |scratch| {
                module.ckks_eval_poly_complex_const_coeffs(&mut out, &input, &complex, &prepared_key, scratch)
            })
            .unwrap();
            results.push(host_ciphertext::<B, _>(&out));

            // Reject inconsistent real/imaginary input schedules before the
            // input transform can allocate or mutate any ciphertext.
            let incompatible = ComplexBSGSPolynomial {
                re: poly
                    .encode_bsgs_with(&host, b.into(), coeff_meta, SplitStrategy::MinDepth)
                    .unwrap()
                    .map_baby_steps_ref(|pt| upload(module, pt)),
                im: complex.im,
            };
            let untouched = host_ciphertext::<B, _>(&out);
            assert!(
                with_scratch::<B, _>(0, |scratch| module.ckks_eval_poly_complex_const_coeffs(
                    &mut out,
                    &input,
                    &incompatible,
                    &prepared_key,
                    scratch
                ))
                .is_err()
            );
            assert!(
                untouched == host_ciphertext::<B, _>(&out),
                "polynomial evaluation result differs"
            );
        }
        assert!(
            before == host_ciphertext::<B, _>(&input),
            "polynomial evaluation result differs"
        );
    }
    for kind in [
        EvalModType::SinCheby,
        EvalModType::CosCheby,
        EvalModType::CosHK,
        EvalModType::ExpCmplx,
    ] {
        for inverse in [None, Some(3)] {
            if kind == EvalModType::ExpCmplx && inverse.is_some() {
                continue;
            }
            let plan = EvalModPlan {
                eval_mod_type: kind,
                log_msg_ratio: 3,
                f_mod_degree: 7,
                f_mod_interval: 2,
                f_mod_log_interval_reduction: if kind == EvalModType::SinCheby { 0 } else { 1 },
                f_mod_inv_degree: inverse,
                scaling: None,
                split_strategy: SplitStrategy::MinDepth,
                coeffs_meta: coeff_meta,
                f_mod_log_delta: 10,
            };
            let encoding = <Module<B> as CKKSEncodingHostOps<B, F>>::ckks_reim_tmp_bytes(module, module.n() / 2);
            let compiled =
                with_scratch::<B, _>(encoding, |scratch| compile_eval_mod::<B, F>(b.into(), plan, module, scratch)).unwrap();
            let input = fixture_ciphertext(module, &layout, 149);
            let before = host_ciphertext::<B, _>(&input);
            let mut out = fixture_ciphertext(module, &layout, 151);
            let eval_bytes = module.ckks_eval_mod_tmp_bytes(&out, &input, &compiled, &key);
            with_scratch::<B, _>(eval_bytes, |scratch| {
                module.ckks_eval_mod(&mut out, &input, &compiled, &prepared_key, scratch)
            })
            .unwrap();
            assert!(
                before == host_ciphertext::<B, _>(&input),
                "polynomial evaluation result differs"
            );
            results.push(host_ciphertext::<B, _>(&out));
        }
    }
    results
}

/// Checks real prepared-basis and one-shot evaluation and folded inputs.
pub fn test_real_polynomial_parity<BR, BT, F>(params: CKKSTestParams, reference: &Module<BR>, tested: &Module<BT>)
where
    BR: Backend<ZnxWord = i64> + CKKSPolynomialEvaluationImpl,
    BT: Backend<ZnxWord = i64> + CKKSPolynomialEvaluationImpl,
    F: CKKSEncodingScalar,
    Module<BR>: CKKSAllOpsTmpBytes<BR> + GLWETensorKeyPreparedFactory<BR> + GLWEMaskFill<BR>,
    Module<BT>: CKKSAllOpsTmpBytes<BT> + GLWETensorKeyPreparedFactory<BT> + GLWEMaskFill<BT>,
{
    assert_eq!(reference.n(), tested.n());
    let want = real_polynomials::<BR, F>(params, reference);
    assert_untagged("real polynomial", &want);
    assert!(
        want == real_polynomials::<BT, F>(params, tested),
        "real polynomial parity differs"
    );
}

/// Checks complex prepared-basis and one-shot evaluation, folded inputs, and
/// every EvalMod family with its selected scratch query and unchanged input.
pub fn test_polynomial_eval_mod_parity<BR, BT, F>(params: CKKSTestParams, reference: &Module<BR>, tested: &Module<BT>)
where
    BR: Backend<ZnxWord = i64> + CKKSImpl + CKKSEncodingImpl<F>,
    BT: Backend<ZnxWord = i64> + CKKSImpl + CKKSEncodingImpl<F>,
    F: CKKSEncodingScalar,
    Module<BR>: CKKSAllOpsTmpBytes<BR> + CKKSEvalModOps<BR> + GLWETensorKeyPreparedFactory<BR> + GLWEMaskFill<BR>,
    Module<BT>: CKKSAllOpsTmpBytes<BT> + CKKSEvalModOps<BT> + GLWETensorKeyPreparedFactory<BT> + GLWEMaskFill<BT>,
{
    assert_eq!(reference.n(), tested.n());
    let want = complex_polynomials_and_eval_mod::<BR, F>(params, reference);
    assert_untagged("complex polynomial or EvalMod", &want);
    assert!(
        want == complex_polynomials_and_eval_mod::<BT, F>(params, tested),
        "complex polynomial or EvalMod parity differs"
    );
}
