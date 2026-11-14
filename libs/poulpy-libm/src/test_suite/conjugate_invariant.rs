//! Tests for the libm operations on the conjugate-invariant ring, whose `n` slots are real.

use std::collections::HashMap;

use poulpy_core::layouts::{GLWEAutomorphismKeyPrepared, GLWELayout, GLWESecretPrepared, LWEInfos, Rank, SlotLayout};
use poulpy_hal::{
    AlignedBuf,
    api::{ScratchOwnedAlloc, ScratchOwnedBorrow},
    layouts::{ConjugateInvariant, HostBytesBackend, Module, ScratchArena, ScratchOwned},
    source::Source,
};

use poulpy_ckks::{
    CKKSInfos, CKKSLayout, CKKSModuleInfos, GLWEPlaintextMeta, Scale, SetCKKSInfos, SlotsKind,
    api::{CKKSDecryptOps, CKKSEncodingHostOps},
    layouts::{CKKSCiphertextAlloc, CKKSCiphertextOwned, CKKSPlaintext, CKKSPlaintextAlloc},
    polynomial::SplitStrategy,
    test_suite::{
        CKKSTestParams,
        helpers::{
            TestContextBackend, TestContextSharedModule, ckks_encrypt_pt, ckks_spec, gen_atk, gen_sk_with_raw, gen_tsk, upload_pt,
        },
    },
};

use crate::{
    comparison::{CKKSComparisonOps, DEFAULT_SIGN_COMPOSITE_CHEBYSHEV},
    exp::{CKKSExpOps, ExpOptions, ExpPlan},
    reduce::CKKSReductionOps,
    sign::{CKKSSignOps, SignComposite},
    softmax::{CKKSSoftmaxOps, SoftmaxOptions, SoftmaxPlan},
};

use super::helpers::{assert_precision_bits, params_for};

type CI = ConjugateInvariant;
type HostPt = CKKSPlaintext<AlignedBuf, i64, CI>;
type HostComposite = SignComposite<f64, HostPt>;
type Keys<BE> = HashMap<i64, GLWEAutomorphismKeyPrepared<<BE as poulpy_hal::layouts::Backend>::OwnedBuf, BE, CI>>;

/// `params_for` with real slots.
fn ci_params(base: &CKKSTestParams, consumed: usize) -> CKKSTestParams {
    let mut params = params_for(base, consumed);
    params.prec_meta.slots = SlotsKind::Real;
    params
}

/// The sign composite of the standard-ring comparison tests, fitted to `|x| >= 0.1` with degree-15 factors.
fn ci_composite(base: &CKKSTestParams, host_module: &Module<HostBytesBackend>) -> HostComposite {
    let coeff_meta = ckks_spec(base.n, base.base2k, base.prec().log_delta(), base.base2k);
    HostComposite::from_minimax(
        0.1,
        20.0,
        &[15],
        12,
        base.base2k.into(),
        coeff_meta,
        SplitStrategy::MinDepth,
        host_module,
    )
    .expect("SignComposite::from_minimax")
}

/// Keys of the slot reductions on the conjugate-invariant ring, which differ from the standard ring's.
fn ci_rotation_keys<BE>(
    params: &CKKSTestParams,
    module: &Module<BE>,
    sk_raw: &poulpy_core::layouts::BackendGLWESecret<BE, CI>,
    scratch: &mut ScratchOwned<BE>,
) -> Keys<BE>
where
    BE: TestContextBackend,
    Module<BE>: TestContextSharedModule<BE, CI> + CKKSReductionOps<BE, CI>,
{
    let elements = module.ckks_sum_slots_galois_elements(&params.prec());
    assert_eq!(
        elements.len(),
        params.n.ilog2() as usize,
        "one rotation per halving of the n real slots"
    );
    let order = 4 * params.n as i64;
    for (i, &gal) in elements.iter().enumerate() {
        assert_eq!(
            gal,
            poulpy_hal::layouts::galois_element(1 << i, order),
            "rotation {i} at the order 4n"
        );
    }
    elements
        .into_iter()
        .map(|gal| (gal, gen_atk(params, module, gal, sk_raw, &mut scratch.borrow())))
        .collect()
}

fn sample(slots: usize, lo: f64, hi: f64, seed: u8) -> Vec<f64> {
    let mut source = Source::new([seed; 32]);
    (0..slots).map(|_| source.next_f64(lo, hi)).collect()
}

/// The `n` real slots of `x`, encoded with the module's encoder and encrypted at `params.k`.
fn ci_encrypt<BE>(
    params: &CKKSTestParams,
    module: &Module<BE>,
    sk: &GLWESecretPrepared<BE::OwnedBuf, BE, CI>,
    x: &[f64],
    scratch: &mut ScratchArena<'_, BE>,
) -> CKKSCiphertextOwned<BE, CI>
where
    BE: TestContextBackend,
    Module<BE>: TestContextSharedModule<BE, CI> + CKKSEncodingHostOps<BE, f64, CI>,
{
    let mut pt =
        module.ckks_plaintext_alloc_from_slots::<CI>(module.ckks_max_slots::<CI>(), params.base2k.into(), params.prec().k());
    pt.set_meta(params.prec().meta());
    module
        .ckks_encode_slots_host_into(&mut pt, x, &vec![0.0; x.len()], scratch)
        .expect("ckks_encode_slots_host_into");
    ckks_encrypt_pt(params, module, sk, params.k, &pt.to_host_owned::<BE>(), scratch)
}

/// The real slots of `ct`, decrypted at its scale with the budget capped as the standard-ring decoder does.
fn ci_decrypt<BE>(
    params: &CKKSTestParams,
    module: &Module<BE>,
    ct: &CKKSCiphertextOwned<BE, CI>,
    sk: &GLWESecretPrepared<BE::OwnedBuf, BE, CI>,
    scratch: &mut ScratchArena<'_, BE>,
) -> Vec<f64>
where
    BE: TestContextBackend,
    Module<BE>: TestContextSharedModule<BE, CI> + CKKSEncodingHostOps<BE, f64, CI>,
{
    let log_delta = ct.log_delta();
    let log_budget = ct
        .log_budget()
        .min(params.prec().log_budget())
        .min(127usize.saturating_sub(log_delta));
    let prec = CKKSLayout {
        glwe_layout: GLWELayout {
            n: ct.n(),
            base2k: ct.base2k(),
            k: (log_delta + log_budget).into(),
            rank: Rank(1),
        },
        meta: GLWEPlaintextMeta {
            scale: Scale::Log(log_delta),
            slots: SlotsKind::Real,
            log_sparsity: 0,
            slot_layout: SlotLayout::Full,
        },
    };
    let mut pt = module.ckks_plaintext_alloc_from_slots::<CI>(module.ckks_max_slots::<CI>(), ct.base2k(), prec.k());
    pt.set_meta(prec.meta());
    module.ckks_decrypt(&mut pt, ct, sk, scratch).expect("ckks_decrypt");
    let slots = module.ckks_max_slots::<CI>();
    let mut re = vec![0.0; slots];
    let mut im = vec![0.0; slots];
    module
        .ckks_decode_slots_host_into(&pt, &mut re, &mut im, scratch)
        .expect("ckks_decode_slots_host_into");
    re
}

/// `sign` and `step` with full-scale factors and no conjugation: the key source is empty, and the scratch is exactly
/// the query, which counts no cleaning temporary.
pub fn test_sign_step<BE>(base: CKKSTestParams, module: &Module<BE>, host_module: &Module<HostBytesBackend>)
where
    BE: TestContextBackend,
    Module<BE>: TestContextSharedModule<BE, CI> + CKKSEncodingHostOps<BE, f64, CI> + CKKSSignOps<BE, CI>,
{
    for step in [false, true] {
        run_sign_case(base, module, host_module, step);
    }
}

fn run_sign_case<BE>(base: CKKSTestParams, module: &Module<BE>, host_module: &Module<HostBytesBackend>, step: bool)
where
    BE: TestContextBackend,
    Module<BE>: TestContextSharedModule<BE, CI> + CKKSEncodingHostOps<BE, f64, CI> + CKKSSignOps<BE, CI>,
{
    let label = if step { "ci/step_composite" } else { "ci/sign_composite" };
    let log_delta = base.prec().log_delta();
    let coeff_meta = ckks_spec(base.n, base.base2k, log_delta, base.base2k);

    let host_composite = HostComposite::from_default(base.base2k.into(), coeff_meta, SplitStrategy::MinDepth, host_module)
        .expect("SignComposite::from_default");
    let consumed = host_composite.consumed_bits(log_delta);

    let k = (consumed + log_delta + 2 * base.base2k).next_multiple_of(base.dsize * base.base2k);
    let mut params = CKKSTestParams {
        k,
        prec_log_budget: k - log_delta,
        ..base
    };
    params.prec_meta.scale = Scale::Log(log_delta);
    params.prec_meta.slots = SlotsKind::Real;

    let slots = module.ckks_max_slots::<CI>();
    assert_eq!(slots, params.n, "{label}: the conjugate-invariant ring has n real slots");

    // |x| >= 0.05: below the composite's resolution the sign is ill-defined.
    let mut source = Source::new([0x5au8; 32]);
    let mut x: Vec<f64> = (0..slots)
        .map(|_| {
            let mag = source.next_f64(0.05, 1.0);
            if source.next_f64(-1.0, 1.0) < 0.0 { -mag } else { mag }
        })
        .collect();
    x[..4].copy_from_slice(&[1.0, -1.0, 0.05, -0.05]);

    let composite = host_composite.map_plaintexts(|pt| upload_pt(module, pt));
    let (sk_raw, sk) = gen_sk_with_raw::<BE, CI>(&params, module, host_module, [0u8; 32]);

    let mut sizing = module.ckks_ciphertext_alloc::<CI>(params.base2k.into(), params.k.into());
    sizing.set_meta(params.prec().meta);
    let sign_bytes = module.ckks_sign_tmp_bytes(&sizing, &sizing, &coeff_meta, &params.tsk_layout(), &params.atk_layout());
    let mut scratch = ScratchOwned::<BE>::alloc(sign_bytes);
    let tsk = gen_tsk(&params, module, &sk_raw, &mut scratch.borrow());
    let no_keys: HashMap<i64, GLWEAutomorphismKeyPrepared<BE::OwnedBuf, BE, CI>> = HashMap::new();

    // Each evaluation gets a fresh arena of exactly the queried size.
    let eval = |res: &mut CKKSCiphertextOwned<BE, CI>,
                ct: &CKKSCiphertextOwned<BE, CI>,
                composite: &SignComposite<f64, _>,
                step: bool| {
        let mut exact = ScratchOwned::<BE>::alloc(sign_bytes);
        let result = if step {
            module.ckks_step(res, ct, composite, &tsk, &no_keys, &mut exact.borrow())
        } else {
            module.ckks_sign(res, ct, composite, &tsk, &no_keys, &mut exact.borrow())
        };
        result.unwrap_or_else(|e| panic!("{label}: {e:#}"));
    };

    let ct = ci_encrypt(&params, module, &sk, &x, &mut scratch.borrow());
    let lb0 = ct.log_budget();
    let mut res = module.ckks_ciphertext_alloc::<CI>(params.base2k.into(), params.k.into());
    eval(&mut res, &ct, &composite, step);

    let out_log_delta = if step { log_delta + 1 } else { log_delta };
    assert_eq!(res.log_delta(), out_log_delta, "{label}: unexpected log_delta");
    assert_eq!(
        lb0 - res.log_budget(),
        consumed + usize::from(step),
        "{label}: consumed bits mismatch"
    );

    // Each factor is held to the noise it adds, as on the standard ring: replayed as a one-row composite, its output is
    // compared with its polynomial applied in the clear to the decrypted input of that factor.
    let required = log_delta as f64 - 15.0;
    let rows = DEFAULT_SIGN_COMPOSITE_CHEBYSHEV;
    let mut cur = ct;
    for j in 0..rows.len() {
        let last_step = step && j + 1 == rows.len();
        let host_factor = HostComposite::from_coeffs(
            base.base2k.into(),
            &rows[j..=j],
            coeff_meta,
            SplitStrategy::MinDepth,
            host_module,
        )
        .expect("one-row SignComposite");
        let x_in = ci_decrypt(&params, module, &cur, &sk, &mut scratch.borrow());
        let want: Vec<f64> = x_in
            .iter()
            .map(|&v| {
                let s = host_factor.evaluate(v);
                if last_step { (s + 1.0) * 0.5 } else { s }
            })
            .collect();
        let factor = host_factor.map_plaintexts(|pt| upload_pt(module, pt));
        let mut next = module.ckks_ciphertext_alloc::<CI>(params.base2k.into(), params.k.into());
        eval(&mut next, &cur, &factor, last_step);
        let got = ci_decrypt(&params, module, &next, &sk, &mut scratch.borrow());
        assert_precision_bits(&format!("{label}/factor {j}"), &got, &want, required, params.n);
        cur = next;
    }

    // The composite is the composition of its factors: the same bits come out.
    let got = ci_decrypt(&params, module, &res, &sk, &mut scratch.borrow());
    let replayed = ci_decrypt(&params, module, &cur, &sk, &mut scratch.borrow());
    assert!(got == replayed, "{label}: differs from its factors evaluated one at a time");
}

/// `exp` from a plan built for the conjugate-invariant ring.
pub fn test_exp<BE>(base: CKKSTestParams, module: &Module<BE>, host_module: &Module<HostBytesBackend>)
where
    BE: TestContextBackend,
    Module<BE>: TestContextSharedModule<BE, CI> + CKKSEncodingHostOps<BE, f64, CI> + CKKSExpOps<BE, CI>,
{
    let log_delta = base.prec().log_delta();
    let coeff_meta = ckks_spec(base.n, base.base2k, log_delta, base.base2k);
    let options = ExpOptions {
        target_bits: 20.0,
        ..ExpOptions::default()
    };
    let host_plan = ExpPlan::<HostPt>::from_precision(0.0, 3.0, base.base2k.into(), coeff_meta, options, host_module)
        .expect("ExpPlan::from_precision");
    let consumed = host_plan.consumed_bits(log_delta);
    let plan = host_plan.map_plaintexts(|pt| upload_pt(module, pt));
    let params = ci_params(&base, consumed);

    let (sk_raw, sk) = gen_sk_with_raw::<BE, CI>(&params, module, host_module, [0x91u8; 32]);
    let mut sizing = module.ckks_ciphertext_alloc::<CI>(params.base2k.into(), params.k.into());
    sizing.set_meta(params.prec().meta);
    let mut scratch = ScratchOwned::<BE>::alloc(module.ckks_exp_tmp_bytes(&sizing, &sizing, &plan, &params.tsk_layout()));
    let tsk = gen_tsk(&params, module, &sk_raw, &mut scratch.borrow());

    let mut x = sample(params.n, 0.0, 3.0, 0x92);
    x[..2].copy_from_slice(&[0.0, 3.0]);
    let want: Vec<f64> = x.iter().map(|v| v.exp()).collect();
    let input = ci_encrypt(&params, module, &sk, &x, &mut scratch.borrow());
    let mut res = module.ckks_ciphertext_alloc::<CI>(params.base2k.into(), params.k.into());
    module
        .ckks_exp(&mut res, &input, &plan, &tsk, &mut scratch.borrow())
        .expect("ckks_exp");
    assert_eq!(input.log_budget() - res.log_budget(), consumed, "ci/exp: consumed bits");
    let got = ci_decrypt(&params, module, &res, &sk, &mut scratch.borrow());
    assert_precision_bits("ci/exp", &got, &want, options.target_bits, params.n);
}

/// `fmax` and `fmin` with no conjugation key.
pub fn test_max_min<BE>(base: CKKSTestParams, module: &Module<BE>, host_module: &Module<HostBytesBackend>)
where
    BE: TestContextBackend,
    Module<BE>: TestContextSharedModule<BE, CI> + CKKSEncodingHostOps<BE, f64, CI> + CKKSComparisonOps<BE, CI>,
{
    let log_delta = base.prec().log_delta();
    let coeff_meta = ckks_spec(base.n, base.base2k, log_delta, base.base2k);
    let host_composite = ci_composite(&base, host_module);
    let consumed = host_composite.consumed_bits(log_delta);
    let params = ci_params(&base, 2 * consumed + 2 * log_delta + 2);
    let composite = ci_composite(&base, host_module).map_plaintexts(|pt| upload_pt(module, pt));

    let mut a = sample(params.n, -0.4, 0.4, 0x33);
    let mut b = sample(params.n, -0.4, 0.4, 0x34);
    (a[0], b[0], a[1], b[1]) = (0.4, -0.4, -0.4, 0.4);
    let (want_max, want_min): (Vec<f64>, Vec<f64>) = a
        .iter()
        .zip(&b)
        .map(|(&av, &bv)| {
            let s = (host_composite.evaluate(av - bv) + 1.0) * 0.5;
            (av * s + bv * (1.0 - s), av * (1.0 - s) + bv * s)
        })
        .unzip();

    let (sk_raw, sk) = gen_sk_with_raw::<BE, CI>(&params, module, host_module, [0x35u8; 32]);
    let mut sizing = module.ckks_ciphertext_alloc::<CI>(params.base2k.into(), params.k.into());
    sizing.set_meta(params.prec().meta);
    let size = module.ckks_comparison_tmp_bytes(
        &sizing,
        &sizing,
        &sizing,
        &coeff_meta,
        &params.tsk_layout(),
        &params.atk_layout(),
    );
    let mut scratch = ScratchOwned::<BE>::alloc(size);
    let tsk = gen_tsk(&params, module, &sk_raw, &mut scratch.borrow());
    let no_keys: Keys<BE> = HashMap::new();
    let op0 = ci_encrypt(&params, module, &sk, &a, &mut scratch.borrow());
    let op1 = ci_encrypt(&params, module, &sk, &b, &mut scratch.borrow());

    for (max, want, label) in [(true, &want_max, "ci/fmax"), (false, &want_min, "ci/fmin")] {
        let mut res = module.ckks_ciphertext_alloc::<CI>(params.base2k.into(), params.k.into());
        if max {
            module.ckks_fmax(&mut res, &op0, &op1, &composite, &tsk, &no_keys, &mut scratch.borrow())
        } else {
            module.ckks_fmin(&mut res, &op0, &op1, &composite, &tsk, &no_keys, &mut scratch.borrow())
        }
        .unwrap_or_else(|e| panic!("{label}: {e:#}"));
        assert_eq!(
            op0.log_budget() - res.log_budget(),
            consumed + log_delta + 1,
            "{label}: consumed bits"
        );
        let got = ci_decrypt(&params, module, &res, &sk, &mut scratch.borrow());
        assert_precision_bits(label, &got, want, log_delta as f64 - 15.0, params.n);
    }
}

/// `sum_slots`, `fmax_slots` and `fmin_slots` over the `n` real slots, with rotation keys at the order `4n` only.
pub fn test_slot_reductions<BE>(base: CKKSTestParams, module: &Module<BE>, host_module: &Module<HostBytesBackend>)
where
    BE: TestContextBackend,
    Module<BE>: TestContextSharedModule<BE, CI> + CKKSEncodingHostOps<BE, f64, CI> + CKKSReductionOps<BE, CI>,
{
    let log_delta = base.prec().log_delta();
    let coeff_meta = ckks_spec(base.n, base.base2k, log_delta, base.base2k);
    let host_composite = ci_composite(&base, host_module);
    let sign_consumed = host_composite.consumed_bits(log_delta);
    let stages = base.n.ilog2() as usize;
    let params = ci_params(&base, stages * (sign_consumed + log_delta + 2));
    let composite = host_composite.map_plaintexts(|pt| upload_pt(module, pt));

    let (sk_raw, sk) = gen_sk_with_raw::<BE, CI>(&params, module, host_module, [0xe2u8; 32]);
    let mut sizing = module.ckks_ciphertext_alloc::<CI>(params.base2k.into(), params.k.into());
    sizing.set_meta(params.prec().meta);
    let (tsk_layout, atk_layout) = (params.tsk_layout(), params.atk_layout());
    let size = module
        .ckks_sum_slots_tmp_bytes(&sizing, &sizing, &atk_layout)
        .max(module.ckks_fmax_slots_tmp_bytes(&sizing, &sizing, &coeff_meta, &tsk_layout, &atk_layout))
        .max(module.ckks_fmin_slots_tmp_bytes(&sizing, &sizing, &coeff_meta, &tsk_layout, &atk_layout));
    let mut scratch = ScratchOwned::<BE>::alloc(size);
    let tsk = gen_tsk(&params, module, &sk_raw, &mut scratch.borrow());
    let keys = ci_rotation_keys(&params, module, &sk_raw, &mut scratch);

    let x = sample(params.n, -0.01, 0.01, 0xe1);
    let want = vec![x.iter().sum::<f64>(); params.n];
    let input = ci_encrypt(&params, module, &sk, &x, &mut scratch.borrow());
    let mut res = module.ckks_ciphertext_alloc::<CI>(params.base2k.into(), params.k.into());
    module
        .ckks_sum_slots(&mut res, &input, &keys, &mut scratch.borrow())
        .expect("ckks_sum_slots");
    assert_eq!(
        res.log_budget(),
        input.log_budget(),
        "ci/sum_slots: rotations consume no budget"
    );
    let got = ci_decrypt(&params, module, &res, &sk, &mut scratch.borrow());
    assert_precision_bits("ci/sum_slots", &got, &want, log_delta as f64 - 15.0, params.n);

    let mut x = sample(params.n, -0.15, 0.15, 0xe3);
    (x[0], x[1]) = (-0.4, 0.4);
    let input = ci_encrypt(&params, module, &sk, &x, &mut scratch.borrow());
    for (max, label, value) in [(true, "ci/fmax_slots", 0.4), (false, "ci/fmin_slots", -0.4)] {
        let mut res = module.ckks_ciphertext_alloc::<CI>(params.base2k.into(), params.k.into());
        if max {
            module.ckks_fmax_slots(&mut res, &input, &composite, &tsk, &keys, &mut scratch.borrow())
        } else {
            module.ckks_fmin_slots(&mut res, &input, &composite, &tsk, &keys, &mut scratch.borrow())
        }
        .unwrap_or_else(|e| panic!("{label}: {e:#}"));
        assert_eq!(
            input.log_budget() - res.log_budget(),
            stages * (sign_consumed + log_delta + 1),
            "{label}: consumed bits"
        );
        let got = ci_decrypt(&params, module, &res, &sk, &mut scratch.borrow());
        assert_precision_bits(label, &got, &vec![value; params.n], 8.0, params.n);
    }
}

/// `softmax` over the `n` real slots, from a plan that counts them.
pub fn test_softmax<BE>(base: CKKSTestParams, module: &Module<BE>, host_module: &Module<HostBytesBackend>)
where
    BE: TestContextBackend,
    Module<BE>:
        TestContextSharedModule<BE, CI> + CKKSEncodingHostOps<BE, f64, CI> + CKKSReductionOps<BE, CI> + CKKSSoftmaxOps<BE, CI>,
{
    let log_delta = base.prec().log_delta();
    let coeff_meta = ckks_spec(base.n, base.base2k, log_delta, base.base2k);
    let options = SoftmaxOptions::default();
    let host_plan = SoftmaxPlan::<HostPt>::from_precision(-1.0, 1.0, base.base2k.into(), coeff_meta, options, host_module)
        .expect("SoftmaxPlan::from_precision");
    assert_eq!(host_plan.slots(), base.n, "ci/softmax: the plan counts the n real slots");
    let consumed = host_plan.consumed_bits(log_delta);
    let plan = host_plan.map_plaintexts(|pt| upload_pt(module, pt));
    let params = ci_params(&base, consumed);

    let mut x = sample(params.n, -1.0, 1.0, 0xf1);
    (x[0], x[1]) = (-1.0, 1.0);
    let denominator: f64 = x.iter().map(|v| v.exp()).sum();
    let want: Vec<f64> = x.iter().map(|v| v.exp() / denominator).collect();

    let (sk_raw, sk) = gen_sk_with_raw::<BE, CI>(&params, module, host_module, [0xf2u8; 32]);
    let mut sizing = module.ckks_ciphertext_alloc::<CI>(params.base2k.into(), params.k.into());
    sizing.set_meta(params.prec().meta);
    let size = module.ckks_softmax_tmp_bytes(&sizing, &sizing, &plan, &params.tsk_layout(), &params.atk_layout());
    let mut scratch = ScratchOwned::<BE>::alloc(size);
    let tsk = gen_tsk(&params, module, &sk_raw, &mut scratch.borrow());
    let keys = ci_rotation_keys(&params, module, &sk_raw, &mut scratch);
    let input = ci_encrypt(&params, module, &sk, &x, &mut scratch.borrow());
    let mut res = module.ckks_ciphertext_alloc::<CI>(params.base2k.into(), params.k.into());
    module
        .ckks_softmax(&mut res, &input, &plan, &tsk, &keys, &mut scratch.borrow())
        .expect("ckks_softmax");
    assert_eq!(input.log_budget() - res.log_budget(), consumed, "ci/softmax: consumed bits");
    let got = ci_decrypt(&params, module, &res, &sk, &mut scratch.borrow());
    assert_precision_bits("ci/softmax", &got, &want, options.target_bits, params.n);
    assert_precision_bits("ci/softmax/sum", &[got.iter().sum::<f64>()], &[1.0], 15.0, params.n);
}
