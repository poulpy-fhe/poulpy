use poulpy_core::{
    EncryptionLayout,
    layouts::{
        GGLWEInfos, GLWELayout, GLWETensorKeyLayout, GLWETensorKeyPrepared, GLWEToBackendMut, GLWEToBackendRef, LWEInfos, Rank,
        prepared::GLWETensorKeyPreparedToBackendRef,
    },
};
use poulpy_hal::{
    api::{NegacyclicFFT, NegacyclicFFTNew, ScratchOwnedAlloc, ScratchOwnedBorrow},
    layouts::{HostBytesBackend, Module, ScratchOwned},
    source::Source,
};

use crate::{
    CKKSCtBounds, CKKSMeta, SetCKKSInfos,
    api::{CKKSAllOpsTmpBytes, CKKSEvalModOps},
    default::eval_mod::{EvalModParameters, EvalModParametersLiteral, EvalModPoly, EvalModType},
    encoding::reim::Encoder,
    layouts::{CKKSCiphertext, CKKSModuleAlloc, CKKSPlaintext},
    polynomial::SplitStrategy,
};

use super::helpers::{
    TestContextBackend, TestContextModule, TestScalar, ckks_decrypt_decode, ckks_encrypt_with_prec, gen_sk_with_raw, gen_tsk,
    precision_stats, upload_pt,
};

#[derive(Clone, Copy, Debug)]
struct EvalModTestParams {
    pub n: usize,
    pub base2k: usize,
    pub k: usize,
    pub prec: CKKSMeta,
    pub hw: usize,
    pub dsize: usize,
}

impl EvalModTestParams {
    fn glwe_layout(&self) -> EncryptionLayout<GLWELayout> {
        EncryptionLayout::new_from_default_sigma(GLWELayout {
            n: self.n.into(),
            base2k: self.base2k.into(),
            k: self.k.into(),
            rank: Rank(1),
        })
        .unwrap()
    }

    fn tsk_layout(&self) -> EncryptionLayout<GLWETensorKeyLayout> {
        let k = self.k + self.dsize * self.base2k;
        let dnum = k.div_ceil(self.dsize * self.base2k);
        EncryptionLayout::new_from_default_sigma(GLWETensorKeyLayout {
            n: self.n.into(),
            base2k: self.base2k.into(),
            k: k.into(),
            rank: Rank(1),
            dsize: self.dsize.into(),
            dnum: dnum.into(),
        })
        .unwrap()
    }

    fn as_test_params(&self) -> super::CKKSTestParams {
        super::CKKSTestParams {
            n: self.n,
            base2k: self.base2k,
            k: self.k,
            prec: self.prec,
            hw: self.hw,
            dsize: self.dsize,
        }
    }
}

fn eval_mod_params(n: usize, base2k: usize, log_delta: usize, depth: usize) -> EvalModTestParams {
    let log_budget = (depth+1) * log_delta + 10;
    let k = (log_delta + log_budget).next_multiple_of(base2k);
    EvalModTestParams {
        n,
        base2k,
        k,
        prec: CKKSMeta {
            log_delta,
            log_budget,
            log_sparsity: 0,
        },
        hw: 192,
        dsize: 1,
    }
}

fn alloc_scratch_eval_mod<BE>(params: &EvalModTestParams, module: &Module<BE>) -> ScratchOwned<BE>
where
    BE: TestContextBackend,
    Module<BE>: TestContextModule<BE>,
    ScratchOwned<BE>: ScratchOwnedAlloc<BE>,
{
    let mut ct = module.ckks_ciphertext_alloc_from_infos(&params.glwe_layout());
    ct.set_meta(params.prec);
    let pt_prec = CKKSMeta {
        log_delta: 8,
        log_budget: 10,
        log_sparsity: 0,
    };
    let scratch_size = module.ckks_all_ops_tmp_bytes(&ct, &params.tsk_layout(), &pt_prec);
    ScratchOwned::<BE>::alloc(scratch_size)
}

fn upload_params<BE, F>(
    module: &Module<BE>,
    host: EvalModParameters<F, CKKSPlaintext<Vec<u8>>>,
) -> EvalModParameters<F, CKKSPlaintext<BE::OwnedBuf>>
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
fn oracle<F, P>(params: &EvalModParameters<F, P>, lit: &EvalModParametersLiteral, t: F, reference: Reference) -> (F, F)
where
    F: TestScalar,
{
    match reference {
        Reference::Polynomial => oracle_polynomial(params, lit, t),
        Reference::Ideal => oracle_ideal(params, lit, t),
    }
}

/// Replays the circuit on its retained polynomials: base polynomial, then the
/// double-angle squarings, then the optional arcsine post-composition. The
/// circuit feeds the encrypted value straight into the bare Chebyshev
/// recurrence, so the polynomial is evaluated directly at `t`.
fn oracle_polynomial<F, P>(params: &EvalModParameters<F, P>, lit: &EvalModParametersLiteral, t: F) -> (F, F)
where
    F: TestScalar,
{
    let two = F::one() + F::one();
    match &params.eval_mod_poly {
        EvalModPoly::Complex(poly) => {
            let (mut re, mut im) = poly.evaluate(t);
            for _ in 0..params.double_angle {
                (re, im) = (re * re - im * im, two * re * im);
            }
            (re, im)
        }
        EvalModPoly::Real(poly) => {
            let mut v = t;
            if matches!(lit.eval_mod_type, EvalModType::CosContinuous) {
                v = v + F::from_f64(-0.25 / lit.eval_mod_interval as f64).unwrap();
            }
            let mut p = poly.evaluate(v);
            let s = params.double_angle_scale();
            for i in 0..params.double_angle {
                let dac = F::from_f64(s.powi(1i32 << (i + 1))).unwrap();
                p = two * p * p - dac;
            }
            if let Some(inv) = &params.eval_mod_inv_poly {
                p = inv.evaluate(p);
            }
            (p, F::zero())
        }
    }
}

/// The exact target the pipeline approximates. The encrypted value `t` is the
/// normalized Chebyshev variable; the modular coordinate is `x = interval·t =
/// I + m/MessageRatio`. Regardless of the sin/cos/double-angle path, the final
/// amplitude is `(1/2π)·scaling` and the function reduces to `sin(2π·x)`
/// (`exp` returns the complex exponential), post-composed with `asin` when an
/// arcsine inverse is configured.
fn oracle_ideal<F, P>(params: &EvalModParameters<F, P>, lit: &EvalModParametersLiteral, t: F) -> (F, F)
where
    F: TestScalar,
{
    let amp = F::from_f64(params.scaling * std::f64::consts::TAU.recip()).unwrap();
    let x = F::from_usize(lit.eval_mod_interval).unwrap() * t;
    let theta = two_pi::<F>() * x;
    if matches!(lit.eval_mod_type, EvalModType::Exp) {
        return (amp * theta.cos(), amp * theta.sin());
    }
    let mut y = theta.sin();
    if params.eval_mod_inv_poly.is_some() {
        y = y.asin();
    }
    (amp * y, F::zero())
}

fn run_eval_mod_case<BE, F, E>(
    label: &str,
    base2k: usize,
    log_delta: usize,
    lit: EvalModParametersLiteral,
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
    const N: usize = 256;
    let host_module = Module::<HostBytesBackend>::new(N as u64);

    // Build the eval_mod parameters first so the input ciphertext is sized from the
    // exact number of levels the pipeline consumes (`EvalModParameters::depth`)
    // rather than a duplicated depth estimate. `coeff_meta` is independent of the
    // ciphertext modulus `k`, so it can be constructed before `params`.
    let coeff_meta = CKKSMeta {
        log_delta,
        log_budget: base2k,
        log_sparsity: 0,
    };
    let host_params = EvalModParameters::<F, _>::from_literal(coeff_meta, base2k.into(), lit, &host_module)
        .expect("EvalModParameters::from_literal");

    let params = eval_mod_params(N, base2k, log_delta, host_params.depth());
    let module = Module::<BE>::new(params.n as u64);
    let test_params = params.as_test_params();
    let params_be = upload_params(&module, host_params);

    let m = params.n / 2;
    let encoder = Encoder::<E>::new(m).unwrap();

    // Sample the plaintext as I·q + m (Lattigo's `mod1_evaluator_test`): I is a
    // random integer multiple in [-(interval-1), interval-1], m a message in
    // [-1, 1], and q = MessageRatio (QDiff = 1 since poulpy's Q = 2^k). The
    // encrypted value is the normalized Chebyshev variable t = (I·q + m)/(q·interval).
    let mr = (1u64 << lit.log_message_ratio) as f64;
    let interval = lit.eval_mod_interval as f64;
    let k = (lit.eval_mod_interval - 1) as f64;
    let mut source = Source::new([0u8; 32]);
    let mut x_re_raw: Vec<F> = (0..m)
        .map(|_| {
            let value = source.next_f64(-k, k).round() * mr + source.next_f64(-1.0, 1.0);
            F::from_f64(value / (mr * interval)).unwrap()
        })
        .collect();
    // Worst-case slot: largest integer multiple plus a half-message.
    x_re_raw[0] = F::from_f64((k * mr + 0.5) / (mr * interval)).unwrap();
    let x_im_raw = vec![F::zero(); x_re_raw.len()];

    let (sk_raw, sk) = gen_sk_with_raw(&test_params, &module, &host_module, [0u8; 32]);
    let mut scratch = alloc_scratch_eval_mod(&params, &module);
    let tsk = gen_tsk(&test_params, &module, &sk_raw, &mut scratch.borrow());

    let ct_input = ckks_encrypt_with_prec(
        &test_params,
        &module,
        &host_module,
        &encoder,
        &sk,
        params.k,
        &x_re_raw,
        &x_im_raw,
        params.prec,
        &mut scratch.borrow(),
    );

    let mut res = module.ckks_ciphertext_alloc(params.base2k.into(), params.k.into());
    module
        .ckks_eval_mod(&mut res, &ct_input, &params_be, &tsk, &mut scratch.borrow())
        .expect("ckks_eval_mod");

    let (re_out, im_out) = ckks_decrypt_decode::<BE, F, E>(&test_params, &module, &encoder, &res, &sk, &mut scratch.borrow());

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

        let stats = precision_stats(&got_re, &want_re, log_delta);

        println!("PREC {label} [{reference:?}]: avg={:.2} min={:.2}", stats.avg_log2_prec, stats.min_log2_prec);

        assert!(
            stats.avg_log2_prec >= required_log2_prec,
            "{label} [{reference:?}]: avg precision {:.1} bits < {required_log2_prec:.1} (worst_err={}, worst_idx={}, got={}, want={})",
            stats.avg_log2_prec,
            stats.worst_err,
            stats.worst_idx,
            stats.worst_got,
            stats.worst_want
        );

        if matches!(lit.eval_mod_type, EvalModType::Exp) {
            let want_im: Vec<F> = want.iter().map(|&(_, im)| im * mr_f).collect();
            let got_im: Vec<F> = im_out.iter().map(|&v| v * mr_f).collect();
            let stats_im = precision_stats(&got_im, &want_im, log_delta);
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
    _params_unused: super::CKKSTestParams,
    _module_unused: &Module<BE>,
    _host_unused: &Module<HostBytesBackend>,
) where
    BE: TestContextBackend,
    Module<BE>: TestContextModule<BE> + CKKSEvalModOps<BE>,
    CKKSCiphertext<BE::OwnedBuf>: GLWEToBackendMut<BE> + GLWEToBackendRef<BE> + CKKSCtBounds + SetCKKSInfos,
    CKKSPlaintext<BE::OwnedBuf>: GLWEToBackendRef<BE> + LWEInfos,
    GLWETensorKeyPrepared<BE::OwnedBuf, BE>: GLWETensorKeyPreparedToBackendRef<BE> + GGLWEInfos,
    F: TestScalar,
    E: NegacyclicFFT<F> + NegacyclicFFTNew<F>,
{
    let lit = EvalModParametersLiteral {
        eval_mod_type: EvalModType::SinContinuous,
        log_message_ratio: 4,
        eval_mod_degree: 31,
        eval_mod_interval: 4,
        double_angle: 0,
        eval_mod_inv_degree: 0,
        scaling: 1.0,
        split_strategy: SplitStrategy::MinDepth,
    };
    run_eval_mod_case::<BE, F, E>("eval_mod_sin_continuous_minimal", 19, 30, lit, 9.0);
}

pub fn test_eval_mod_sin_continuous_with_arcsine<BE, F, E>(
    _params_unused: super::CKKSTestParams,
    _module_unused: &Module<BE>,
    _host_unused: &Module<HostBytesBackend>,
) where
    BE: TestContextBackend,
    Module<BE>: TestContextModule<BE> + CKKSEvalModOps<BE>,
    CKKSCiphertext<BE::OwnedBuf>: GLWEToBackendMut<BE> + GLWEToBackendRef<BE> + CKKSCtBounds + SetCKKSInfos,
    CKKSPlaintext<BE::OwnedBuf>: GLWEToBackendRef<BE> + LWEInfos,
    GLWETensorKeyPrepared<BE::OwnedBuf, BE>: GLWETensorKeyPreparedToBackendRef<BE> + GGLWEInfos,
    F: TestScalar,
    E: NegacyclicFFT<F> + NegacyclicFFTNew<F>,
{
    let lit = EvalModParametersLiteral {
        eval_mod_type: EvalModType::SinContinuous,
        log_message_ratio: 8,
        eval_mod_degree: 63,
        eval_mod_interval: 8,
        double_angle: 0,
        eval_mod_inv_degree: 7,
        scaling: 1.0,
        split_strategy: SplitStrategy::MinDepth,
    };
    run_eval_mod_case::<BE, F, E>("eval_mod_sin_continuous_arcsine", 19, 30, lit, 9.0);
}

pub fn test_eval_mod_cos_discrete<BE, F, E>(
    _params_unused: super::CKKSTestParams,
    _module_unused: &Module<BE>,
    _host_unused: &Module<HostBytesBackend>,
) where
    BE: TestContextBackend,
    Module<BE>: TestContextModule<BE> + CKKSEvalModOps<BE>,
    CKKSCiphertext<BE::OwnedBuf>: GLWEToBackendMut<BE> + GLWEToBackendRef<BE> + CKKSCtBounds + SetCKKSInfos,
    CKKSPlaintext<BE::OwnedBuf>: GLWEToBackendRef<BE> + LWEInfos,
    GLWETensorKeyPrepared<BE::OwnedBuf, BE>: GLWETensorKeyPreparedToBackendRef<BE> + GGLWEInfos,
    F: TestScalar,
    E: NegacyclicFFT<F> + NegacyclicFFTNew<F>,
{
    let lit = EvalModParametersLiteral {
        eval_mod_type: EvalModType::CosDiscrete,
        log_message_ratio: 8,
        eval_mod_degree: 30,
        eval_mod_interval: 12,
        double_angle: 3,
        eval_mod_inv_degree: 0,
        scaling: 1.0,
        split_strategy: SplitStrategy::MinDepth,
    };
    run_eval_mod_case::<BE, F, E>("eval_mod_cos_discrete", 19, 60, lit, 40.0);
}

pub fn test_eval_mod_cos_continuous<BE, F, E>(
    _params_unused: super::CKKSTestParams,
    _module_unused: &Module<BE>,
    _host_unused: &Module<HostBytesBackend>,
) where
    BE: TestContextBackend,
    Module<BE>: TestContextModule<BE> + CKKSEvalModOps<BE>,
    CKKSCiphertext<BE::OwnedBuf>: GLWEToBackendMut<BE> + GLWEToBackendRef<BE> + CKKSCtBounds + SetCKKSInfos,
    CKKSPlaintext<BE::OwnedBuf>: GLWEToBackendRef<BE> + LWEInfos,
    GLWETensorKeyPrepared<BE::OwnedBuf, BE>: GLWETensorKeyPreparedToBackendRef<BE> + GGLWEInfos,
    F: TestScalar,
    E: NegacyclicFFT<F> + NegacyclicFFTNew<F>,
{
    let lit = EvalModParametersLiteral {
        eval_mod_type: EvalModType::CosContinuous,
        log_message_ratio: 4,
        eval_mod_degree: 31,
        eval_mod_interval: 8,
        double_angle: 3,
        eval_mod_inv_degree: 0,
        scaling: 1.0,
        split_strategy: SplitStrategy::MinDepth,
    };
    run_eval_mod_case::<BE, F, E>("eval_mod_cos_continuous", 19, 30, lit, 16.0);
}

pub fn test_eval_mod_exp<BE, F, E>(
    _params_unused: super::CKKSTestParams,
    _module_unused: &Module<BE>,
    _host_unused: &Module<HostBytesBackend>,
) where
    BE: TestContextBackend,
    Module<BE>: TestContextModule<BE> + CKKSEvalModOps<BE>,
    CKKSCiphertext<BE::OwnedBuf>: GLWEToBackendMut<BE> + GLWEToBackendRef<BE> + CKKSCtBounds + SetCKKSInfos,
    CKKSPlaintext<BE::OwnedBuf>: GLWEToBackendRef<BE> + LWEInfos,
    GLWETensorKeyPrepared<BE::OwnedBuf, BE>: GLWETensorKeyPreparedToBackendRef<BE> + GGLWEInfos,
    F: TestScalar,
    E: NegacyclicFFT<F> + NegacyclicFFTNew<F>,
{
    let lit = EvalModParametersLiteral {
        eval_mod_type: EvalModType::Exp,
        log_message_ratio: 4,
        eval_mod_degree: 31,
        eval_mod_interval: 8,
        double_angle: 3,
        eval_mod_inv_degree: 0,
        scaling: 1.0,
        split_strategy: SplitStrategy::MinDepth,
    };
    run_eval_mod_case::<BE, F, E>("eval_mod_exp", 19, 30, lit, 16.0);
}
