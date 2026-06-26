use poulpy_core::layouts::{
    GGLWEInfos, GLWETensorKeyPrepared, GLWEToBackendMut, GLWEToBackendRef, LWEInfos, prepared::GLWETensorKeyPreparedToBackendRef,
};
use poulpy_hal::{
    api::{NegacyclicFFT, NegacyclicFFTNew, ScratchOwnedAlloc, ScratchOwnedBorrow},
    layouts::{HostBytesBackend, Module, ScratchOwned},
    source::Source,
};

use crate::{
    CKKSCtBounds, CKKSInfos, CKKSLayout, CKKSMeta, SetCKKSInfos,
    api::{CKKSAllOpsTmpBytes, CKKSEvalModOps},
    encoding::reim::Encoder,
    layouts::{
        CKKSCiphertext, CKKSModuleAlloc, CKKSPlaintext,
        eval_mod::{EvalMod, EvalModPlan, EvalModPoly, EvalModType},
    },
    polynomial::SplitStrategy,
    test_suite::CKKSTestParams,
};

use super::helpers::{
    TestContextBackend, TestContextModule, TestScalar, ckks_decrypt_decode, ckks_encrypt_with_prec, ckks_spec, gen_sk_with_raw,
    gen_tsk, precision_stats, upload_pt,
};

fn alloc_scratch_eval_mod<BE, F>(
    params: &super::CKKSTestParams,
    module: &Module<BE>,
    eval_mod: &EvalMod<F, CKKSPlaintext<BE::OwnedBuf>>,
    res_k: usize,
) -> ScratchOwned<BE>
where
    BE: TestContextBackend,
    Module<BE>: TestContextModule<BE>,
    ScratchOwned<BE>: ScratchOwnedAlloc<BE>,
{
    let mut ct = module.ckks_ciphertext_alloc(params.base2k.into(), params.k.into());
    ct.set_meta(params.prec().meta);
    let mut res = module.ckks_ciphertext_alloc(params.base2k.into(), res_k.into());
    res.set_meta(params.prec().meta);
    let pt_prec = ckks_spec(params.n, params.base2k, 8, 10);
    let scratch_size = module
        .ckks_all_ops_tmp_bytes(&res, &params.tsk_layout(), &pt_prec)
        .max(module.ckks_eval_mod_tmp_bytes(&res, &ct, eval_mod, &params.tsk_layout()));
    ScratchOwned::<BE>::alloc(scratch_size)
}

fn upload_params<BE, F>(module: &Module<BE>, host: EvalMod<F, CKKSPlaintext<Vec<u8>>>) -> EvalMod<F, CKKSPlaintext<BE::OwnedBuf>>
where
    BE: TestContextBackend,
    Module<BE>: TestContextModule<BE>,
{
    host.map_plaintexts(|pt| upload_pt(module, pt))
}

#[derive(Clone, Copy, Debug)]
enum Reference {
    /// The retained circuit polynomials — isolates FHE/BSGS fidelity.
    Polynomial,
    /// The ideal trigonometric `x mod 1` — tests approximation + noise end-to-end.
    Ideal,
}

fn two_pi<F: TestScalar>() -> F {
    F::PI() + F::PI()
}

/// Reference evaluation of the `x mod 1` pipeline at the normalized Chebyshev
/// variable `t` (the encrypted value, in `[-1, 1]`).
fn oracle<F, P>(params: &EvalMod<F, P>, lit: &EvalModPlan, t: F, reference: Reference) -> (F, F)
where
    F: TestScalar,
{
    match reference {
        Reference::Polynomial => oracle_polynomial(params, lit, t),
        Reference::Ideal => oracle_ideal(params, lit, t),
    }
}

/// Replays the circuit on its retained polynomials: base polynomial, then the
/// range-extension squarings, then the optional arcsine post-composition. The
/// circuit feeds the encrypted value straight into the bare Chebyshev
/// recurrence, so the polynomial is evaluated directly at `t`.
fn oracle_polynomial<F, P>(params: &EvalMod<F, P>, _lit: &EvalModPlan, t: F) -> (F, F)
where
    F: TestScalar,
{
    let two = F::one() + F::one();
    match &params.f_mod_poly {
        EvalModPoly::Complex(poly) => {
            let (mut re, mut im) = poly.evaluate(t);
            for _ in 0..params.plan.f_mod_log_interval_reduction {
                (re, im) = (re * re - im * im, two * re * im);
            }
            (re, im)
        }
        EvalModPoly::Real(poly) => {
            // The CosCheby phase shift is baked into `poly`, so evaluate at `t` directly.
            let mut p = poly.evaluate(t);
            let s = params.range_extension_scale();
            for i in 0..params.plan.f_mod_log_interval_reduction {
                let dac = F::from_f64(s.powi(1i32 << (i + 1))).unwrap();
                p = two * p * p - dac;
            }
            if let Some(inv) = &params.f_mod_inv_poly {
                p = inv.evaluate(p);
            }
            (p, F::zero())
        }
    }
}

/// The exact target the pipeline approximates. The encrypted value `t` is the
/// normalized Chebyshev variable; the modular coordinate is `x = interval·t =
/// I + m/MessageRatio`. Regardless of the variant, the final
/// amplitude is `(1/2π)·scaling` and the function reduces to `sin(2π·x)`
/// (`exp` returns the complex exponential), post-composed with `asin` when an
/// arcsine inverse is configured.
fn oracle_ideal<F, P>(params: &EvalMod<F, P>, lit: &EvalModPlan, t: F) -> (F, F)
where
    F: TestScalar,
{
    let amp = F::from_f64(params.plan.scaling.unwrap_or(1.0) * std::f64::consts::TAU.recip()).unwrap();
    let x = F::from_usize(lit.f_mod_interval).unwrap() * t;
    let theta = two_pi::<F>() * x;
    if matches!(lit.eval_mod_type, EvalModType::ExpCmplx) {
        return (amp * theta.cos(), amp * theta.sin());
    }
    let mut y = theta.sin();
    if params.f_mod_inv_poly.is_some() {
        y = y.asin();
    }
    (amp * y, F::zero())
}

fn run_eval_mod_case<BE, F, E>(
    params: super::CKKSTestParams,
    module: &Module<BE>,
    host_module: &Module<HostBytesBackend>,
    label: &str,
    lit: EvalModPlan,
    required_log2_prec: f64,
) where
    BE: TestContextBackend,
    Module<BE>: TestContextModule<BE> + CKKSEvalModOps<BE>,
    CKKSCiphertext<BE::OwnedBuf>: GLWEToBackendMut<BE> + GLWEToBackendRef<BE> + CKKSCtBounds + SetCKKSInfos,
    CKKSPlaintext<BE::OwnedBuf>: GLWEToBackendRef<BE> + LWEInfos,
    GLWETensorKeyPrepared<BE::OwnedBuf, BE>: GLWETensorKeyPreparedToBackendRef<BE> + GGLWEInfos,
    F: TestScalar,
    E: NegacyclicFFT<F> + NegacyclicFFTNew<F>,
{
    // Coefficients are encoded at the scale EvalMod runs at (`f_mod_log_delta`).
    let mut lit = lit;
    lit.coeffs_meta = ckks_spec(params.n, params.base2k, lit.f_mod_log_delta, params.base2k);
    let host_params = EvalMod::<F, _>::from_literal(params.base2k.into(), lit, host_module).expect("EvalMod::from_literal");

    // Input message scale, below the plan scale so EvalMod's internal raise to
    // `f_mod_log_delta` is exercised.
    let input_log_delta = 40;
    let dsize = 2;
    let test_params = CKKSTestParams {
        n: params.n,
        base2k: params.base2k,
        // Budget for the evaluation (charged at `f_mod_log_delta`) + head-room,
        // rounded to `dsize·base2k` so the tensor-key gadget layout stays valid.
        k: (lit.consumed_bits() + input_log_delta + 2 * params.base2k).next_multiple_of(dsize * params.base2k),
        hw: 192,
        prec_meta: CKKSMeta {
            log_sparsity: 0,
            log_delta: input_log_delta,
        },
        prec_log_budget: 10,
        dsize,
    };

    let params_be = upload_params(module, host_params);

    let slots = test_params.n / 2;
    let encoder = Encoder::<E>::new(slots).unwrap();

    // Sample the plaintext as I·q + m : I is a
    // random integer multiple in [-(interval-1), interval-1], m a message in
    // [-1, 1], and q = MessageRatio (QDiff = 1 since poulpy's Q = 2^k). The
    // encrypted value is the normalized Chebyshev variable t = (I·q + m)/(q·interval).
    let mr = (1u64 << lit.log_msg_ratio) as f64;
    let interval = lit.f_mod_interval as f64;
    let k = (lit.f_mod_interval - 1) as f64;
    let mut source = Source::new([0u8; 32]);
    let mut x_re_raw: Vec<F> = (0..slots)
        .map(|_| {
            let value = source.next_f64(-k, k).round() * mr + source.next_f64(-1.0, 1.0);
            F::from_f64(value / (mr * interval)).unwrap()
        })
        .collect();
    // Worst-case slot: largest integer multiple plus a half-message.
    x_re_raw[0] = F::from_f64((k * mr + 0.5) / (mr * interval)).unwrap();
    let x_im_raw = vec![F::zero(); x_re_raw.len()];

    let (sk_raw, sk) = gen_sk_with_raw(&test_params, module, host_module, [0u8; 32]);
    // `res` must span the raised scale EvalMod evaluates at (`f_mod_log_delta`),
    // which is wider than the input scale by `f_mod_log_delta - input_log_delta`.
    let res_k = test_params.k + lit.f_mod_log_delta.saturating_sub(input_log_delta);
    let mut scratch = alloc_scratch_eval_mod(&test_params, module, &params_be, res_k);
    let tsk = gen_tsk(&test_params, module, &sk_raw, &mut scratch.borrow());

    let ct_input = ckks_encrypt_with_prec(
        &test_params,
        module,
        host_module,
        &encoder,
        &sk,
        test_params.k,
        &x_re_raw,
        &x_im_raw,
        test_params.prec(),
        &mut scratch.borrow(),
    );

    let (in_ld, in_lb) = (ct_input.log_delta(), ct_input.log_budget());
    let mut res = module.ckks_ciphertext_alloc(test_params.base2k.into(), res_k.into());
    module
        .ckks_eval_mod(&mut res, &ct_input, &params_be, &tsk, &mut scratch.borrow())
        .expect("ckks_eval_mod");

    // Exact externally visible bit-consumption: EvalMod arithmetic is charged at
    // the plan scale. Returning from that raised scale to the input scale
    // preserves the remaining budget, matching `ckks_set_log_delta`.
    assert_eq!(res.log_delta(), in_ld, "{label}: eval_mod should preserve log_delta");
    assert_eq!(
        in_lb - res.log_budget(),
        params_be.consumed_bits(),
        "{label}: eval_mod consumed bits mismatch"
    );

    let (re_out, im_out) = ckks_decrypt_decode::<BE, F, E>(&test_params, module, &encoder, &res, &sk, &mut scratch.borrow());

    // The eval_mod output recovers the message scaled by 1/message_ratio (the
    // amplitude folds in `scaling`, which is 1 here). Scale the output and the
    // references back up by message_ratio so precision is measured on the recovered
    // message at its true magnitude rather than on the down-scaled slot value.
    let mr_f = F::from_f64(mr).unwrap();
    let got_re: Vec<F> = re_out.iter().map(|&v| v * mr_f).collect();

    // Compare the FHE output against both references: the circuit's own
    // polynomials (FHE fidelity) and the ideal x mod 1 (approximation + noise).
    for reference in [Reference::Polynomial, Reference::Ideal] {
        let want: Vec<(F, F)> = x_re_raw.iter().map(|&t| oracle(&params_be, &lit, t, reference)).collect();
        let want_re: Vec<F> = want.iter().map(|&(re, _)| re * mr_f).collect();

        let stats = precision_stats(&got_re, &want_re, test_params.prec().log_delta());

        println!(
            "PREC {label} [{reference:?}]: avg={:.2} min={:.2}",
            stats.avg_log2_prec, stats.min_log2_prec
        );

        assert!(
            stats.avg_log2_prec >= required_log2_prec,
            "{label} [{reference:?}]: avg precision {:.1} bits < {required_log2_prec:.1} (worst_err={}, worst_idx={}, got={}, want={})",
            stats.avg_log2_prec,
            stats.worst_err,
            stats.worst_idx,
            stats.worst_got,
            stats.worst_want
        );

        if matches!(lit.eval_mod_type, EvalModType::ExpCmplx) {
            let want_im: Vec<F> = want.iter().map(|&(_, im)| im * mr_f).collect();
            let got_im: Vec<F> = im_out.iter().map(|&v| v * mr_f).collect();
            let stats_im = precision_stats(&got_im, &want_im, test_params.prec().log_delta());
            assert!(
                stats_im.avg_log2_prec >= required_log2_prec,
                "{label} [{reference:?}] (imag): avg precision {:.1} bits < {required_log2_prec:.1} (worst_err={}, worst_idx={}, got={}, want={})",
                stats_im.avg_log2_prec,
                stats_im.worst_err,
                stats_im.worst_idx,
                stats_im.worst_got,
                stats_im.worst_want
            );
        }
    }
}

pub fn test_eval_mod_sin_continuous_minimal<BE, F, E>(
    params: super::CKKSTestParams,
    module: &Module<BE>,
    host_module: &Module<HostBytesBackend>,
) where
    BE: TestContextBackend,
    Module<BE>: TestContextModule<BE> + CKKSEvalModOps<BE>,
    CKKSCiphertext<BE::OwnedBuf>: GLWEToBackendMut<BE> + GLWEToBackendRef<BE> + CKKSCtBounds + SetCKKSInfos,
    CKKSPlaintext<BE::OwnedBuf>: GLWEToBackendRef<BE> + LWEInfos,
    GLWETensorKeyPrepared<BE::OwnedBuf, BE>: GLWETensorKeyPreparedToBackendRef<BE> + GGLWEInfos,
    F: TestScalar,
    E: NegacyclicFFT<F> + NegacyclicFFTNew<F>,
{
    let lit = EvalModPlan {
        eval_mod_type: EvalModType::SinCheby,
        log_msg_ratio: 8,
        f_mod_degree: 127,
        f_mod_interval: 14,
        f_mod_log_interval_reduction: 0,
        f_mod_inv_degree: None,
        scaling: None,
        split_strategy: SplitStrategy::MinDepth,
        coeffs_meta: CKKSLayout::default(),
        f_mod_log_delta: 60,
    };
    run_eval_mod_case::<BE, F, E>(params, module, host_module, "eval_mod_sin_continuous_minimal", lit, 18.0);
}

pub fn test_eval_mod_sin_continuous_with_arcsine<BE, F, E>(
    params: super::CKKSTestParams,
    module: &Module<BE>,
    host_module: &Module<HostBytesBackend>,
) where
    BE: TestContextBackend,
    Module<BE>: TestContextModule<BE> + CKKSEvalModOps<BE>,
    CKKSCiphertext<BE::OwnedBuf>: GLWEToBackendMut<BE> + GLWEToBackendRef<BE> + CKKSCtBounds + SetCKKSInfos,
    CKKSPlaintext<BE::OwnedBuf>: GLWEToBackendRef<BE> + LWEInfos,
    GLWETensorKeyPrepared<BE::OwnedBuf, BE>: GLWETensorKeyPreparedToBackendRef<BE> + GGLWEInfos,
    F: TestScalar,
    E: NegacyclicFFT<F> + NegacyclicFFTNew<F>,
{
    let lit = EvalModPlan {
        eval_mod_type: EvalModType::SinCheby,
        log_msg_ratio: 8,
        f_mod_degree: 127,
        f_mod_interval: 14,
        f_mod_log_interval_reduction: 0,
        f_mod_inv_degree: Some(3),
        f_mod_log_delta: 60,
        scaling: None,
        split_strategy: SplitStrategy::MinDepth,
        coeffs_meta: CKKSLayout::default(),
    };
    run_eval_mod_case::<BE, F, E>(params, module, host_module, "eval_mod_sin_continuous_arcsine", lit, 18.0);
}

pub fn test_eval_mod_cos_discrete<BE, F, E>(
    params: super::CKKSTestParams,
    module: &Module<BE>,
    host_module: &Module<HostBytesBackend>,
) where
    BE: TestContextBackend,
    Module<BE>: TestContextModule<BE> + CKKSEvalModOps<BE>,
    CKKSCiphertext<BE::OwnedBuf>: GLWEToBackendMut<BE> + GLWEToBackendRef<BE> + CKKSCtBounds + SetCKKSInfos,
    CKKSPlaintext<BE::OwnedBuf>: GLWEToBackendRef<BE> + LWEInfos,
    GLWETensorKeyPrepared<BE::OwnedBuf, BE>: GLWETensorKeyPreparedToBackendRef<BE> + GGLWEInfos,
    F: TestScalar,
    E: NegacyclicFFT<F> + NegacyclicFFTNew<F>,
{
    let lit = EvalModPlan {
        eval_mod_type: EvalModType::CosHK,
        log_msg_ratio: 8,
        f_mod_degree: 30,
        f_mod_interval: 16,
        f_mod_log_interval_reduction: 3,
        f_mod_inv_degree: None,
        f_mod_log_delta: 60,
        scaling: None,
        split_strategy: SplitStrategy::MinDepth,
        coeffs_meta: CKKSLayout::default(),
    };
    run_eval_mod_case::<BE, F, E>(params, module, host_module, "eval_mod_cos_discrete", lit, 18.0);
}

pub fn test_eval_mod_cos_continuous<BE, F, E>(
    params: super::CKKSTestParams,
    module: &Module<BE>,
    host_module: &Module<HostBytesBackend>,
) where
    BE: TestContextBackend,
    Module<BE>: TestContextModule<BE> + CKKSEvalModOps<BE>,
    CKKSCiphertext<BE::OwnedBuf>: GLWEToBackendMut<BE> + GLWEToBackendRef<BE> + CKKSCtBounds + SetCKKSInfos,
    CKKSPlaintext<BE::OwnedBuf>: GLWEToBackendRef<BE> + LWEInfos,
    GLWETensorKeyPrepared<BE::OwnedBuf, BE>: GLWETensorKeyPreparedToBackendRef<BE> + GGLWEInfos,
    F: TestScalar,
    E: NegacyclicFFT<F> + NegacyclicFFTNew<F>,
{
    let lit = EvalModPlan {
        eval_mod_type: EvalModType::CosCheby,
        log_msg_ratio: 4,
        f_mod_degree: 31,
        f_mod_interval: 16,
        f_mod_log_interval_reduction: 3,
        f_mod_inv_degree: None,
        scaling: None,
        split_strategy: SplitStrategy::MinDepth,
        coeffs_meta: CKKSLayout::default(),
        f_mod_log_delta: 60,
    };
    run_eval_mod_case::<BE, F, E>(params, module, host_module, "eval_mod_cos_continuous", lit, 18.0);
}

pub fn test_eval_mod_exp<BE, F, E>(params: super::CKKSTestParams, module: &Module<BE>, host_module: &Module<HostBytesBackend>)
where
    BE: TestContextBackend,
    Module<BE>: TestContextModule<BE> + CKKSEvalModOps<BE>,
    CKKSCiphertext<BE::OwnedBuf>: GLWEToBackendMut<BE> + GLWEToBackendRef<BE> + CKKSCtBounds + SetCKKSInfos,
    CKKSPlaintext<BE::OwnedBuf>: GLWEToBackendRef<BE> + LWEInfos,
    GLWETensorKeyPrepared<BE::OwnedBuf, BE>: GLWETensorKeyPreparedToBackendRef<BE> + GGLWEInfos,
    F: TestScalar,
    E: NegacyclicFFT<F> + NegacyclicFFTNew<F>,
{
    let lit = EvalModPlan {
        eval_mod_type: EvalModType::ExpCmplx,
        log_msg_ratio: 4,
        f_mod_degree: 31,
        f_mod_interval: 16,
        f_mod_log_interval_reduction: 3,
        f_mod_inv_degree: None,
        scaling: None,
        split_strategy: SplitStrategy::MinDepth,
        coeffs_meta: CKKSLayout::default(),
        f_mod_log_delta: 60,
    };
    run_eval_mod_case::<BE, F, E>(params, module, host_module, "eval_mod_exp", lit, 18.0);
}
