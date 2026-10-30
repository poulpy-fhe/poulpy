//! End-to-end CKKS bootstrapping test (backend-generic).
//!
//! Exercises the [`ckks_bootstrap`](crate::api::CKKSBootstrappingOps) orchestrator
//! end to end over the refresh pipeline:
//!
//! ```text
//! ModUp ─► CoeffsToSlots(split) ─► EvalMod(×2) ─► SlotsToCoeffs(split)
//! ```
//!
//! 1. Encrypt slots `z` at the input modulus `q = 2^log_modulus_in` ("level 0").
//! 2. ModUp re-interprets the ciphertext at the wide bootstrap modulus, exposing
//!    the integer wrap-around `I(X)·q` in the coefficients.
//! 3. CoeffsToSlots moves the coefficients `q·I_j + Δ·c_j` into the slots of two
//!    real ciphertexts (real/imag halves).
//! 4. EvalMod removes `q·I_j` from each.
//! 5. SlotsToCoeffs maps the slots back to coefficients — a refreshed `z`.
//!
//! Scale bridge (see [`BootstrappingContextPrepared`](crate::layouts::BootstrappingContextPrepared)): CoeffsToSlots is pre-scaled by
//! `1/K` (`K = f_mod_interval`) into EvalMod's `[-1, 1]` domain; after ModUp the
//! ciphertext is relabeled at the input-modulus scale (free division by the
//! message ratio), restored by a `2^R` scale-up after SlotsToCoeffs.
//!
//! EvalMod set_scale ([`BootstrappingPlan::eval_mod`]'s `f_mod_log_delta`): EvalMod is run at a
//! wider scale than the input — `set_scale(eval_mod) → EvalMod → set_scale(input)`
//! — so its `ct×ct` chain keeps more precision. The recovered average precision
//! measures ~28 bits across the suite configurations; the assertions enforce
//! the `MIN_AVG_LOG2_PREC` regression floor a few bits under that.

use crate::CKKSModuleInfos;
use crate::Scale;
use crate::api::CKKSEncodingOps;
use crate::layouts::CKKSCiphertextOwned;
use crate::layouts::CKKSPlaintextOwned;

use poulpy_core::layouts::{
    GGLWEInfos, GLWEAutomorphismKeyLayout, GLWEInfos, GLWEPlaintextInfos, GLWESecretPreparedToBackendRef, GLWETensorKeyPrepared,
    GLWEToBackendMut, GLWEToBackendRef, LWEInfos, LinearTransformation, SetGLWEPlaintextInfos,
    prepared::GLWETensorKeyPreparedToBackendRef,
};
use poulpy_hal::{
    api::{ReimNew, ScratchOwnedAlloc, ScratchOwnedBorrow},
    layouts::{Backend, HostBytesBackend, HostDataMut, HostDataRef, Module, ScratchArena, ScratchOwned, Standard, ZnxView},
    source::Source,
};

use crate::SlotsKind;
use crate::{
    CKKSCompositionError, CKKSCtBounds, CKKSInfos, CoeffsMeta, GLWEPlaintextMeta, SetCKKSInfos,
    api::{
        CKKSAddOps, CKKSBootstrappingOps, CKKSDFTMatrixOps, CKKSDFTOps, CKKSDecryptOps, CKKSEvalModOps, CKKSPow2Ops, CKKSSubOps,
    },
    layouts::{
        BootstrappingKeys, BootstrappingKeysLayout, BootstrappingLinearKeys, BootstrappingPipeline, BootstrappingPlan,
        BootstrappingTechniques, CKKSModuleAlloc, DFTMatrix, DFTOutputFormat, DFTPlan, DFTType, EncapsulationKeysLayout,
        EvalRoundPlus, SparseSecretEncapsulation,
        eval_mod::{EvalModPlan, EvalModType},
    },
    polynomial::SplitStrategy,
    test_suite::reference_encoder::ReferenceEncoder,
    test_suite::{
        CKKSTestParams,
        helpers::{
            LoggedBootstrappingKeys, TestContextBackend, TestContextModule, TestScalar, assert_canonical_at_k, assert_ckks_error,
            assert_invalid_operand, ckks_encrypt_with_prec, ckks_spec, gen_sk_with_raw, next_test_seed, precision_stats,
            test_vector_1,
        },
        parity::helpers::{fixture_ciphertext, snapshot, with_scratch},
        presets::{bootstrap_setup_tmp_bytes, compile_context},
    },
};

/// Scratch of one DFT stage from `src` to `dst`, under the largest of the rotation key layouts of `keys`.
fn dft_stage_bytes<BE, Dir, Fmt, P, Dst, Src>(
    module: &Module<BE>,
    keys: &BootstrappingKeysLayout,
    dst: &Dst,
    src: &Src,
    dft: &DFTMatrix<BE, Dir, Fmt, LinearTransformation<P>>,
) -> usize
where
    BE: Backend,
    Module<BE>: CKKSDFTOps<BE>,
    P: GLWEInfos,
    Dst: CKKSCtBounds,
    Src: CKKSCtBounds,
{
    keys.rotation_key_layouts()
        .iter()
        .map(|atk| module.ckks_dft_tmp_bytes(dst, src, dft, atk))
        .max()
        .unwrap()
}

/// `log2` of the live complex slot count (ring degree `n = 2·2^LOG_SLOTS`), kept
/// small to bound the depth — and modulus width — of a self-contained test.
const LOG_SLOTS: usize = 10;
const FMOD_INTERVAL: usize = 16;
const LOG_MSG_RATIO: usize = 11;
/// Hamming weight of the ephemeral sparse-encapsulation secret selected by the
/// bootstrapping recipe.
const EPHEMERAL_SECRET_WEIGHT: usize = 32;
/// Regression floor for the recovered average precision, in bits.
///
/// Every suite configuration measures 27.5–28.3 average bits on the reference
/// backend (2026-07-19, release); the floor sits ~4 bits under the weakest
/// measurement to absorb backend FFT and noise variance while still failing on
/// any real precision regression (the previous 5-bit smoke floor would have
/// passed a 28 → 6-bit collapse).
const MIN_AVG_LOG2_PREC: f64 = 24.0;

fn meta(log_delta: usize, log_budget: usize) -> CoeffsMeta {
    CoeffsMeta::from_delta_budget(log_delta, log_budget)
}

/// Shared recipe for the standard, EvalRound+ and S2C-first tests.
/// Only plain C2S-first uses optimal BSGS; S2C-first uses higher-precision S2C
/// coefficients at half scale because that transform runs before ModUp.
fn bootstrap_plan(
    pipeline: BootstrappingPipeline,
    eval_round_plus: bool,
    log_msg_ratio: usize,
    guard_bits: usize,
    first_factor_on_raise: bool,
) -> BootstrappingPlan {
    plan_with_encapsulation(
        pipeline,
        eval_round_plus,
        log_msg_ratio,
        guard_bits,
        first_factor_on_raise,
        true,
    )
}

/// Every rotation key of `plan`, at the layout of `tp`, for an input of `input_k` bits at scale
/// `2^log_delta` bootstrapped at `tp.k`.
fn uniform_rotation_keys(
    plan: &BootstrappingPlan,
    tp: &CKKSTestParams,
    input_k: usize,
    log_delta: usize,
) -> std::collections::BTreeMap<i64, GLWEAutomorphismKeyLayout> {
    let widths = plan.key_widths(tp.log_n(), tp.k, input_k, log_delta);
    widths.rotation_keys.into_keys().map(|p| (p, tp.atk_layout())).collect()
}

/// [`bootstrap_plan`], with sparse-secret encapsulation or without it.
fn plan_with_encapsulation(
    pipeline: BootstrappingPipeline,
    eval_round_plus: bool,
    log_msg_ratio: usize,
    guard_bits: usize,
    first_factor_on_raise: bool,
    encapsulated: bool,
) -> BootstrappingPlan {
    let s2c_first = pipeline == BootstrappingPipeline::S2CFirst;
    let mut coeffs_to_slots = DFTPlan::new(
        DFTType::Encode,
        vec![(2, 4), (2, 4), (3, 4), (3, 4)],
        DFTOutputFormat::SplitRealAndImag,
        meta(if eval_round_plus { 29 } else { 58 }, 2),
    )
    .unwrap();
    let mut slots_to_coeffs = DFTPlan::new(
        DFTType::Decode,
        vec![(3, 4), (3, 4), (2, 4), (2, 4)],
        DFTOutputFormat::SplitRealAndImag,
        meta(if s2c_first { 45 } else { 39 }, 2),
    )
    .unwrap()
    .with_scaling(if s2c_first { 0.5 } else { (log_msg_ratio as f64).exp2() })
    .unwrap();
    if !s2c_first && !eval_round_plus {
        coeffs_to_slots = coeffs_to_slots.with_optimal_bsgs(LOG_SLOTS + 1);
        slots_to_coeffs = slots_to_coeffs.with_optimal_bsgs(LOG_SLOTS + 1);
    }
    if first_factor_on_raise {
        coeffs_to_slots = coeffs_to_slots.with_first_factor_on_raise();
    }
    let plan = BootstrappingPlan::new(
        pipeline,
        BootstrappingTechniques {
            sparse_secret_encapsulation: encapsulated.then_some(SparseSecretEncapsulation {
                hamming_weight: EPHEMERAL_SECRET_WEIGHT,
            }),
            eval_round_plus: eval_round_plus.then(|| EvalRoundPlus {
                coeffs_to_slots_bypass: DFTPlan::new(
                    DFTType::Encode,
                    vec![(1, 1); LOG_SLOTS],
                    DFTOutputFormat::SplitRealAndImag,
                    meta(58, 2),
                )
                .unwrap()
                .with_scaling(1.0)
                .unwrap(),
            }),
        },
        coeffs_to_slots,
        EvalModPlan {
            eval_mod_type: EvalModType::CosHK,
            log_msg_ratio,
            f_mod_degree: 30,
            f_mod_interval: FMOD_INTERVAL,
            f_mod_log_interval_reduction: 3,
            f_mod_inv_degree: None,
            scaling: None,
            split_strategy: SplitStrategy::MinDepth,
            coeffs_meta: meta(48, 4),
            f_mod_log_delta: 60,
        },
        slots_to_coeffs,
    )
    .unwrap();
    if s2c_first {
        plan.with_c2s_guard_bits(guard_bits).unwrap()
    } else {
        plan
    }
}

/// End-to-end bootstrapping with the first CoeffsToSlots factor applied by the
/// raise, in both pipelines and for complex and real-tagged inputs, under
/// sparse-secret encapsulation and without it: the factor costs no modulus,
/// and the precision is that of the ordinary pipeline.
pub fn test_bootstrapping_first_factor_on_raise<BE, F, E>(params: CKKSTestParams)
where
    BE: TestContextBackend<Ring = Standard>,
    Module<BE>: TestContextModule<BE>
        + CKKSEncodingOps<BE, F>
        + CKKSBootstrappingOps<BE>
        + CKKSDFTOps<BE>
        + CKKSEvalModOps<BE>
        + CKKSDFTMatrixOps<BE, F>,
    F: TestScalar,
    E: ReimNew<F>,
    for<'a> <BE as Backend>::BufRef<'a>: HostDataRef,
    for<'a> <BE as Backend>::BufMut<'a>: HostDataMut,
    ScratchOwned<BE>: ScratchOwnedAlloc<BE>,
    CKKSCiphertextOwned<BE>: GLWEToBackendMut<BE> + GLWEToBackendRef<BE> + CKKSCtBounds + SetCKKSInfos,
    CKKSPlaintextOwned<BE>: GLWEToBackendRef<BE> + LWEInfos,
    GLWETensorKeyPrepared<BE::OwnedBuf, BE>: GLWETensorKeyPreparedToBackendRef<BE> + GGLWEInfos,
    BE: crate::oep::CKKSModUpFirstFactorImpl,
{
    let n = 1 << (LOG_SLOTS + 1);
    let m = n / 2;
    let base2k = params.base2k;
    // The scales and message ratios of the ordinary tests of each pipeline, at a
    // raised width on a limb boundary and off one. Without encapsulation the raise
    // runs under the dense secret, here of the ephemeral secret's weight.
    for (pipeline, log_delta, log_msg_ratio, guard_bits, aligned, encapsulated) in [
        (BootstrappingPipeline::C2SFirst, 45, LOG_MSG_RATIO, 0, true, true),
        (BootstrappingPipeline::S2CFirst, 40, 16, 6, true, true),
        (BootstrappingPipeline::C2SFirst, 45, LOG_MSG_RATIO, 0, false, true),
        (BootstrappingPipeline::S2CFirst, 40, 16, 6, false, true),
        (BootstrappingPipeline::C2SFirst, 45, LOG_MSG_RATIO, 0, true, false),
    ] {
        let s2c_first = pipeline == BootstrappingPipeline::S2CFirst;
        let ordinary = plan_with_encapsulation(pipeline, false, log_msg_ratio, guard_bits, false, encapsulated);
        let plan = plan_with_encapsulation(pipeline, false, log_msg_ratio, guard_bits, true, encapsulated);
        assert_eq!(plan.sparse_secret_hamming_weight().is_some(), encapsulated);
        let factor_scale = plan.coeffs_to_slots().coeffs_meta().log_delta();
        assert_eq!(plan.consumed_bits() + factor_scale, ordinary.consumed_bits());

        let log_modulus_in = log_delta + plan.eval_mod().log_msg_ratio;
        let k_in = plan.input_k(log_modulus_in);
        let k_boot = if s2c_first {
            plan.bootstrap_k(3 * log_delta, log_delta)
        } else {
            plan.bootstrap_k(log_modulus_in + 2 * log_delta, log_delta)
        }
        .next_multiple_of(base2k)
            + if aligned { 0 } else { base2k / 2 + 1 };

        let module = Module::<BE>::new(n as u64);
        let host_module = Module::<HostBytesBackend>::new(n as u64);
        let encoder = ReferenceEncoder::<E>::new::<F>(m).unwrap();
        let tp = CKKSTestParams {
            n,
            base2k,
            k: k_boot,
            prec_meta: GLWEPlaintextMeta {
                scale: Scale::Log(log_delta),
                slots: SlotsKind::Complex,
                log_sparsity: 0,
            },
            prec_log_budget: 8,
            hw: if encapsulated { 192 } else { EPHEMERAL_SECRET_WEIGHT },
            dsize: 7,
            rank: 1,
        };
        let first_factor_key = tp.first_factor_key_layout(&plan, k_boot, log_modulus_in);
        let keys_layout = BootstrappingKeysLayout {
            rotation_keys: uniform_rotation_keys(&plan, &tp, k_in, log_delta),
            first_factor_key: Some(first_factor_key),
            conversion_keys: None,
            tensor_key: tp.tsk_layout(),
            // The first-factor keys switch back to the dense secret.
            encapsulation: plan.sparse_secret_hamming_weight().map(|_| EncapsulationKeysLayout {
                dense_to_sparse: tp.ksk_layout(log_modulus_in),
                sparse_to_dense: None,
            }),
        };
        let boot_spec = ckks_spec(n, base2k, log_delta, k_boot - log_delta);
        let in_spec = ckks_spec(n, base2k, log_delta, k_in - log_delta);
        let mut scratch = ScratchOwned::<BE>::alloc(bootstrap_setup_tmp_bytes(&module, &boot_spec, &plan, &keys_layout));
        let ctx = compile_context::<BE, F>(&module, base2k.into(), &plan);
        // The matrix holds the paid factors; the first one stands apart, a rotation per diagonal.
        let first = ctx
            .coeffs_to_slots_first_factor()
            .expect("the first factor is left to the raise");
        assert_eq!(ctx.coeffs_to_slots().num_factors() + 1, plan.coeffs_to_slots().num_factors());
        assert!(first.giant_steps.iter().all(|giant| giant.diagonals.len() == 1));
        let boot_tmp = module.ckks_bootstrap_tmp_bytes(&boot_spec, &in_spec, &ctx, &keys_layout);

        let (sk_raw, sk) = gen_sk_with_raw(&tp, &module, &host_module, next_test_seed(0));
        let (mut src_xs, mut src_xa, mut src_xe) = (
            Source::new(next_test_seed(7)),
            Source::new(next_test_seed(1)),
            Source::new(next_test_seed(2)),
        );
        let key_set = ctx
            .generate_keys(
                &module,
                &sk_raw,
                &keys_layout,
                &mut src_xs,
                &mut src_xe,
                &mut src_xa,
                &mut scratch.borrow(),
            )
            .unwrap();
        assert_eq!(key_set.first_factor_keys.len(), first.giant_steps.len());
        let bsk = key_set.prepare(&module, &mut scratch.borrow());

        // The raise alone runs on exactly the scratch it advertises.
        let (k_small, raised) = if s2c_first {
            (log_modulus_in, plan.raised_window(log_modulus_in))
        } else {
            (k_in, plan.raised_window(log_modulus_in))
        };
        let small_spec = ckks_spec(n, base2k, log_delta, k_small - log_delta);
        let raise_tmp = BE::ckks_mod_up_first_factor_tmp_bytes(
            &module,
            &boot_spec,
            &small_spec,
            first,
            keys_layout.encapsulation.as_ref().map(|encaps| &encaps.dense_to_sparse),
            &first_factor_key,
        );
        assert!(raise_tmp > 0);
        {
            let mut src = module.ckks_ciphertext_alloc(base2k.into(), k_small.into());
            src.set_meta(small_spec.meta);
            let mut dst = module.ckks_ciphertext_alloc(base2k.into(), k_boot.into());
            with_scratch::<BE, _>(raise_tmp, |s| {
                BE::ckks_mod_up_first_factor(&module, &mut dst, &mut src, raised - k_small, first, &bsk, s)
            })
            .unwrap();
        }

        // Keys laid out for the raise refuse a narrower bootstrap before any work.
        let mut narrow = fixture_ciphertext(&module, &ckks_spec(n, base2k, log_delta, k_boot - base2k - log_delta), 23);
        let ct_any = module.ckks_ciphertext_alloc(base2k.into(), k_in.into());
        let before = snapshot::<BE, _>(&narrow);
        let err = with_scratch::<BE, _>(0, |s| module.ckks_bootstrap(&mut narrow, &ct_any, &ctx, &bsk, s)).unwrap_err();
        assert_invalid_operand(
            "a bootstrap narrower than its raise keys",
            &err,
            "ckks_bootstrap",
            "the first-factor keys switch",
        );
        assert_eq!(snapshot::<BE, _>(&narrow), before, "a refused bootstrap wrote res");
        let mut rank2 = in_spec;
        rank2.glwe_layout.rank = 2usize.into();
        let ct_rank2 = fixture_ciphertext(&module, &rank2, 29);
        let err = with_scratch::<BE, _>(0, |s| module.ckks_bootstrap(&mut narrow, &ct_rank2, &ctx, &bsk, s)).unwrap_err();
        assert_ckks_error(
            "a bootstrap of a rank-2 input",
            &err,
            CKKSCompositionError::InvalidOperand {
                op: "ckks_bootstrap",
                reason: "supports rank-1 ciphertexts only, got rank 2 -> 1".into(),
            },
        );
        assert_eq!(snapshot::<BE, _>(&narrow), before, "a refused bootstrap wrote res");

        let (re, im) = test_vector_1::<F>(m);
        let im_zero = vec![F::zero(); m];
        for (want_im, slots) in [(&im, SlotsKind::Complex), (&im_zero, SlotsKind::Real)] {
            let mut ct0 = ckks_encrypt_with_prec(
                &tp,
                &module,
                &host_module,
                &encoder,
                &sk,
                k_in,
                &re,
                want_im,
                in_spec,
                &mut scratch.borrow(),
            );
            ct0.set_slots(slots);
            let mut ct_bs = module.ckks_ciphertext_alloc(base2k.into(), k_boot.into());
            let logged = LoggedBootstrappingKeys::new::<BE>(&bsk);
            with_scratch::<BE, _>(boot_tmp, |s| module.ckks_bootstrap(&mut ct_bs, &ct0, &ctx, &logged, s)).unwrap();
            let widths = plan.key_widths(tp.log_n(), k_boot, ct0.k().as_usize(), ct0.log_delta());
            logged.assert_lookups_at(&widths, "first factor on the raise");
            assert_eq!((ct_bs.log_delta(), ct_bs.slots()), (log_delta, slots));
            let scale_drop = if s2c_first {
                0
            } else {
                plan.eval_mod().f_mod_log_delta - log_delta
            };
            assert_eq!(ct_bs.k().as_usize(), k_boot - plan.post_mod_up_consumed_bits() - scale_drop);
            let (got_re, got_im) = decrypt(&module, &encoder, &ct_bs, &sk, &mut scratch.borrow());
            for (got, want, tag) in [(&got_re, &re, "re"), (&got_im, want_im, "im")] {
                let stats = precision_stats(got, want, log_delta);
                println!(
                    "ckks_bootstrap (first factor on raise, {pipeline:?}, {slots:?}, aligned={aligned}, encapsulated={encapsulated}) ({tag}) avg={:.2} min={:.2} bits",
                    stats.avg_log2_prec, stats.min_log2_prec
                );
                assert!(
                    stats.avg_log2_prec >= MIN_AVG_LOG2_PREC,
                    "first factor on raise ({pipeline:?}, {slots:?}, encapsulated={encapsulated}, {tag}): {:.1} bits < {MIN_AVG_LOG2_PREC}",
                    stats.avg_log2_prec
                );
            }
        }
    }
}

/// End-to-end bootstrapping: encrypt at level 0, refresh, check the slots return.
pub fn test_bootstrapping_standard_e2e<BE, F, E>(params: CKKSTestParams)
where
    BE: TestContextBackend<Ring = Standard>,
    Module<BE>: TestContextModule<BE>
        + CKKSEncodingOps<BE, F>
        + CKKSBootstrappingOps<BE>
        + CKKSDFTOps<BE>
        + CKKSEvalModOps<BE>
        + CKKSDFTMatrixOps<BE, F>,
    F: TestScalar,
    E: ReimNew<F>,
    for<'a> <BE as Backend>::BufRef<'a>: HostDataRef,
    for<'a> <BE as Backend>::BufMut<'a>: HostDataMut,
    ScratchOwned<BE>: ScratchOwnedAlloc<BE>,
    CKKSCiphertextOwned<BE>: GLWEToBackendMut<BE> + GLWEToBackendRef<BE> + CKKSCtBounds + SetCKKSInfos,
    CKKSPlaintextOwned<BE>: GLWEToBackendRef<BE> + LWEInfos,
    GLWETensorKeyPrepared<BE::OwnedBuf, BE>: GLWETensorKeyPreparedToBackendRef<BE> + GGLWEInfos,
{
    let plan = bootstrap_plan(BootstrappingPipeline::C2SFirst, false, LOG_MSG_RATIO, 0, false);

    let n = 1 << (LOG_SLOTS + 1);
    let m = n / 2;
    let base2k = params.base2k;
    let log_delta = 45;
    // Scale of the ciphertext entering the pipeline.
    let log_modulus_in = log_delta + plan.eval_mod().log_msg_ratio;

    let k_boot = plan
        .bootstrap_k(log_modulus_in + 2 * log_delta, log_delta)
        .next_multiple_of(base2k);

    let module = Module::<BE>::new(n as u64);
    let host_module = Module::<HostBytesBackend>::new(n as u64);
    let encoder = ReferenceEncoder::<E>::new::<F>(m).unwrap();

    let tp = CKKSTestParams {
        n,
        base2k,
        k: k_boot,
        prec_meta: GLWEPlaintextMeta {
            scale: Scale::Log(log_delta),
            slots: SlotsKind::Complex,
            log_sparsity: 0,
        },
        prec_log_budget: 8,
        hw: 192,
        dsize: 7,
        rank: 1,
    };

    // One scratch for the whole pipeline (plaintext precision sized for the
    // largest plaintext op, EvalMod).
    // Keys shaped by use, as the presets do: a rotation key used below the
    // raised width only covers the widest ciphertext it switches there.
    let after_c2s = k_boot - plan.coeffs_to_slots().consumed_bits();
    let narrow = |k: usize| CKKSTestParams { k, ..tp };
    let widths = |k_boot: usize| plan.key_widths(tp.log_n(), k_boot, log_modulus_in, log_delta);
    let rotation_keys: std::collections::BTreeMap<_, _> = widths(k_boot)
        .rotation_keys
        .into_iter()
        .map(|(p, k)| (p, narrow(k).atk_layout()))
        .collect();
    let keys_layout = BootstrappingKeysLayout {
        rotation_keys,
        first_factor_key: None,
        conversion_keys: None,
        tensor_key: tp.tsk_layout(),
        encapsulation: plan.sparse_secret_hamming_weight().map(|_| EncapsulationKeysLayout {
            dense_to_sparse: tp.ksk_layout(log_modulus_in),
            // The raise is the sparse-to-dense switch, with rows under the raised ciphertext.
            sparse_to_dense: Some(tp.ksk_offset_layout(k_boot, plan.raised_window(log_modulus_in))),
        }),
    };
    let scratch_size = bootstrap_setup_tmp_bytes(
        &module,
        &ckks_spec(n, base2k, log_delta, k_boot - log_delta),
        &plan,
        &keys_layout,
    );
    let mut scratch = ScratchOwned::<BE>::alloc(scratch_size);

    let ctx = compile_context::<BE, F>(&module, base2k.into(), &plan);

    let (sk_raw, sk) = gen_sk_with_raw(&tp, &module, &host_module, [0u8; 32]);

    let boot_tmp = module.ckks_bootstrap_tmp_bytes(
        &ckks_spec(n, base2k, log_delta, k_boot - log_delta),
        &ckks_spec(n, base2k, log_delta, log_modulus_in - log_delta),
        &ctx,
        &keys_layout,
    );
    let (mut src_xs, mut src_xa, mut src_xe) = (Source::new([7u8; 32]), Source::new([1u8; 32]), Source::new([2u8; 32]));
    // `generate_keys` returns the keys *unprepared* (the serializable / GPU-resident
    // form); `prepare` preprocesses the whole set up front for this CPU path.
    let bsk = ctx
        .generate_keys(
            &module,
            &sk_raw,
            &keys_layout,
            &mut src_xs,
            &mut src_xe,
            &mut src_xa,
            &mut scratch.borrow(),
        )
        .unwrap()
        .prepare(&module, &mut scratch.borrow());

    // The tensor key covers EvalMod's input.
    {
        use crate::layouts::BootstrappingKeys;
        use poulpy_core::layouts::GGLWEInfos;
        assert!(after_c2s <= bsk.tensor_key().gglwe_layout().max_input_k().as_usize());
    }

    // Encrypt z at the input ("level 0") modulus.
    let (re, im) = test_vector_1::<F>(m);

    let ct0 = ckks_encrypt_with_prec(
        &tp,
        &module,
        &host_module,
        &encoder,
        &sk,
        log_modulus_in,
        &re,
        &im,
        ckks_spec(n, base2k, log_delta, log_modulus_in - log_delta),
        &mut scratch.borrow(),
    );

    // A working width narrower than CoeffsToSlots and EvalMod consume fails at entry.
    {
        let mut ct_narrow = fixture_ciphertext(&module, &ckks_spec(n, base2k, log_delta, base2k), 31);
        let before = snapshot::<BE, _>(&ct_narrow);
        let err = with_scratch::<BE, _>(0, |s| module.ckks_bootstrap(&mut ct_narrow, &ct0, &ctx, &bsk, s)).unwrap_err();
        assert_invalid_operand(
            "a bootstrap narrower than its stages",
            &err,
            "ckks_bootstrap",
            "is narrower than the bits its stages consume",
        );
        assert_eq!(snapshot::<BE, _>(&ct_narrow), before, "a refused bootstrap wrote res");
    }

    // The narrowest working width that takes a key past its layout fails at
    // entry, with the key source's refusal as its source.
    {
        use poulpy_core::layouts::GGLWEInfos;
        let covered = |k_boot: usize| {
            widths(k_boot)
                .rotation_keys
                .iter()
                .all(|(p, &k)| k <= bsk.rotation_keys[p].gglwe_layout().max_input_k().as_usize())
        };
        let wide_k = (k_boot..).step_by(base2k).find(|&k| !covered(k)).unwrap();
        let mut ct_wide = fixture_ciphertext(&module, &ckks_spec(n, base2k, log_delta, wide_k - log_delta), 23);
        let before = snapshot::<BE, _>(&ct_wide);
        let err = with_scratch::<BE, _>(0, |s| module.ckks_bootstrap(&mut ct_wide, &ct0, &ctx, &bsk, s)).unwrap_err();
        match err.composition() {
            Some(CKKSCompositionError::MissingAutomorphismKey {
                op: "ckks_bootstrap",
                source,
                ..
            }) => assert!(source.to_string().contains("wider than the key's max_input_k"), "{err}"),
            _ => panic!("expected a key refused at entry, got: {err}"),
        }
        assert_eq!(snapshot::<BE, _>(&ct_wide), before, "a refused bootstrap wrote res");
    }

    // Compare the public orchestrator with the explicit pipeline below.
    let ct_bs = {
        let mut ct_bs = module.ckks_ciphertext_alloc(base2k.into(), k_boot.into());
        let logged = LoggedBootstrappingKeys::new::<BE>(&bsk);
        with_scratch::<BE, _>(boot_tmp, |s| module.ckks_bootstrap(&mut ct_bs, &ct0, &ctx, &logged, s)).unwrap();
        let widths = plan.key_widths(tp.log_n(), k_boot, ct0.k().as_usize(), ct0.log_delta());
        logged.assert_lookups_at(&widths, "C2S-first");
        assert_eq!(ct_bs.log_delta(), log_delta);
        assert_eq!(
            ct_bs.k().as_usize(),
            k_boot - plan.consumed_bits() - (plan.eval_mod().f_mod_log_delta - log_delta)
        );
        let (re_bs, im_bs) = decrypt(&module, &encoder, &ct_bs, &sk, &mut scratch.borrow());
        for (got, want, tag) in [(&re_bs, &re, "re"), (&im_bs, &im, "im")] {
            let s = precision_stats(got, want, log_delta);
            println!(
                "ckks_bootstrap (standard) ({tag}) avg={:.2} min={:.2} bits",
                s.avg_log2_prec, s.min_log2_prec
            );
            assert!(
                s.avg_log2_prec >= MIN_AVG_LOG2_PREC,
                "ckks_bootstrap standard ({tag}): {:.1} bits < {MIN_AVG_LOG2_PREC}",
                s.avg_log2_prec
            );
        }
        ct_bs
    };

    // A real-tagged input must come back real-tagged, whichever pipeline the
    // recipe selects, and its imaginary part must stay zero.
    {
        let im_zero = vec![F::zero(); m];
        let mut ct_real = ckks_encrypt_with_prec(
            &tp,
            &module,
            &host_module,
            &encoder,
            &sk,
            log_modulus_in,
            &re,
            &im_zero,
            ckks_spec(n, base2k, log_delta, log_modulus_in - log_delta),
            &mut scratch.borrow(),
        );
        ct_real.set_slots(SlotsKind::Real);
        let mut ct_bs = module.ckks_ciphertext_alloc(base2k.into(), k_boot.into());
        with_scratch::<BE, _>(boot_tmp, |s| module.ckks_bootstrap(&mut ct_bs, &ct_real, &ctx, &bsk, s)).unwrap();
        assert_eq!(ct_bs.slots(), SlotsKind::Real, "standard output slot kind");
        assert_eq!(ct_bs.log_delta(), log_delta);
        assert_eq!(
            ct_bs.k().as_usize(),
            k_boot - plan.consumed_bits() - (plan.eval_mod().f_mod_log_delta - log_delta)
        );
        let (re_bs, im_bs) = decrypt(&module, &encoder, &ct_bs, &sk, &mut scratch.borrow());
        assert!(precision_stats(&re_bs, &re, log_delta).avg_log2_prec >= MIN_AVG_LOG2_PREC);
        assert!(precision_stats(&im_bs, &im_zero, log_delta).avg_log2_prec >= 5.0);
    }

    // 1) The whole raise step: lift to the plan's message ratio, (encapsulate)
    //    denseToSparse / ModUp / sparseToDense so the integer wrap-around `I(X)·q`
    //    is bounded by the *sparse* secret's Hamming weight, and relabel by the
    //    message ratio: `I(X)·q` becomes the integer part, the message the
    //    residue `Δ·c/q`.
    let mut ct = module.ckks_ciphertext_alloc(base2k.into(), k_boot.into());
    // `ckks_bootstrap_tmp_bytes` is the query of the raise step.
    with_scratch::<BE, _>(boot_tmp, |s| {
        module.ckks_bootstrap_mod_up(&mut ct, &ct0, plan.eval_mod(), &bsk, s)
    })
    .unwrap();

    let mut log_budget_check = k_boot - ct.log_delta();

    assert_eq!(ct.log_budget(), log_budget_check);
    assert_canonical_at_k::<BE>("ModUp", &ct);

    // 2) CoeffsToSlots (split): coefficients → (real, imag) slots.
    let mut ct_real = module.ckks_ciphertext_alloc(base2k.into(), k_boot.into());
    let mut ct_imag = module.ckks_ciphertext_alloc(base2k.into(), k_boot.into());
    let bytes = dft_stage_bytes(&module, &keys_layout, &ct_real, &ct, ctx.coeffs_to_slots());
    with_scratch::<BE, _>(bytes, |s| {
        module.ckks_coeffs_to_slots_split(&mut ct_real, &mut ct_imag, &ct, ctx.coeffs_to_slots(), bsk.rotation_keys(), s)
    })
    .unwrap();

    log_budget_check -= plan.coeffs_to_slots().consumed_bits();

    assert_eq!(ct_real.log_budget(), log_budget_check);
    assert_eq!(ct_imag.log_budget(), log_budget_check);
    for ct in [&ct_real, &ct_imag] {
        assert_canonical_at_k::<BE>("CoeffsToSlots", ct);
    }

    // 3) EvalMod each half. EvalMod raises the ciphertext to its own plan scale
    //    (`f_mod_log_delta`) internally and restores the input scale on the result,
    //    so no manual set_scale is needed here. The results are allocated at
    //    exactly `k_boot`: an allocation's width is the width ct×ct squaring
    //    computes at.
    let mut res_real = module.ckks_ciphertext_alloc(base2k.into(), k_boot.into());
    let mut res_imag = module.ckks_ciphertext_alloc(base2k.into(), k_boot.into());
    with_scratch::<BE, _>(
        module.ckks_eval_mod_tmp_bytes(&res_real, &ct_real, ctx.eval_mod().unwrap(), &bsk.tensor_key),
        |s| module.ckks_eval_mod(&mut res_real, &ct_real, ctx.eval_mod().unwrap(), &bsk.tensor_key, s),
    )
    .unwrap();
    with_scratch::<BE, _>(
        module.ckks_eval_mod_tmp_bytes(&res_imag, &ct_imag, ctx.eval_mod().unwrap(), &bsk.tensor_key),
        |s| module.ckks_eval_mod(&mut res_imag, &ct_imag, ctx.eval_mod().unwrap(), &bsk.tensor_key, s),
    )
    .unwrap();

    log_budget_check -= plan.eval_mod().consumed_bits();

    assert_eq!(res_real.log_budget(), log_budget_check);
    assert_eq!(res_imag.log_budget(), log_budget_check);
    for ct in [&res_real, &res_imag] {
        assert_canonical_at_k::<BE>("EvalMod", ct);
    }

    // 4) SlotsToCoeffs (split), then restore the message ratio EvalMod divided out.
    let mut ct_out = module.ckks_ciphertext_alloc(base2k.into(), k_boot.into());
    let bytes = dft_stage_bytes(&module, &keys_layout, &ct_out, &res_real, ctx.slots_to_coeffs());
    with_scratch::<BE, _>(bytes, |s| {
        module.ckks_slots_to_coeffs_split(
            &mut ct_out,
            &res_real,
            &res_imag,
            ctx.slots_to_coeffs(),
            bsk.rotation_keys(),
            s,
        )
    })
    .unwrap();

    log_budget_check -= plan.slots_to_coeffs().consumed_bits();

    assert_eq!(log_budget_check, k_boot - plan.consumed_bits() - ct_out.log_delta());
    assert_eq!(ct_out.log_budget(), log_budget_check);

    ct_out.set_log_delta(log_delta);
    assert_same_bootstrap::<BE>(&ct_out, &ct_bs);
    let (re_out, im_out) = decrypt(&module, &encoder, &ct_out, &sk, &mut scratch.borrow());

    for (got, want, tag) in [(&re_out, &re, "re"), (&im_out, &im, "im")] {
        let s = precision_stats(got, want, log_delta);
        println!(
            "BOOTSTRAP-PREC ({tag}) avg={:.2} min={:.2} bits",
            s.avg_log2_prec, s.min_log2_prec,
        );
        assert!(
            s.avg_log2_prec >= MIN_AVG_LOG2_PREC,
            "bootstrap_e2e ({tag}): {:.1} bits < {MIN_AVG_LOG2_PREC} (worst got={} want={})",
            s.avg_log2_prec,
            s.worst_got,
            s.worst_want,
        );
    }
}

/// End-to-end **slot-domain EvalRound+** bootstrapping (eprint 2024/1379).
///
/// ```text
/// ModUp ─► CoeffsToSlots(split) ×2 : LP (low-prec) and HP (high-prec)
///       ─► r1 = r0_hp − K·r0_lp + EvalMod(r0_lp) = IDFT(Δ·m)
///       ─► SlotsToCoeffs(split) = m(X)
/// ```
///
/// EvalMod runs on the **low-precision** CoeffsToSlots (`log_delta = 29`); its DFT
/// error `e` cancels in `r0_hp − K·r0_lp + EvalMod(r0_lp)` (the `−e` from `K·r0_lp`
/// and the `+e` from `EvalMod` annihilate), so the message is reconstructed at the
/// **high-precision** CoeffsToSlots' (`log_delta = 58`) precision. Because EvalMod
/// only needs to resolve the large integer part, halving its CoeffsToSlots
/// precision shrinks the bootstrap modulus without hurting the message.
///
/// The HP CoeffsToSlots (the "bypass") runs in the modulus depth the LP+EvalMod
/// path already occupies, so it does not enlarge `k_boot`.
///
/// Scale bridge: the LP C2S folds in `1/K` (EvalMod's `[-1,1]` domain) while EvalMod
/// emits the residue at natural scale, so `r0_lp` is scaled up by `K`; the HP C2S
/// uses natural (`1.0`) scaling, and SlotsToCoeffs the standard `2^log_message_ratio`.
pub fn test_bootstrapping_evalround_e2e<BE, F, E>(params: CKKSTestParams)
where
    BE: TestContextBackend<Ring = Standard>,
    Module<BE>: TestContextModule<BE>
        + CKKSEncodingOps<BE, F>
        + CKKSBootstrappingOps<BE>
        + CKKSDFTOps<BE>
        + CKKSEvalModOps<BE>
        + CKKSDFTMatrixOps<BE, F>,
    F: TestScalar,
    E: ReimNew<F>,
    for<'a> <BE as Backend>::BufRef<'a>: HostDataRef,
    for<'a> <BE as Backend>::BufMut<'a>: HostDataMut,
    ScratchOwned<BE>: ScratchOwnedAlloc<BE>,
    CKKSCiphertextOwned<BE>: GLWEToBackendMut<BE> + GLWEToBackendRef<BE> + CKKSCtBounds + SetCKKSInfos,
    CKKSPlaintextOwned<BE>: GLWEToBackendRef<BE> + LWEInfos,
    GLWETensorKeyPrepared<BE::OwnedBuf, BE>: GLWETensorKeyPreparedToBackendRef<BE> + GGLWEInfos,
{
    let plan = bootstrap_plan(BootstrappingPipeline::C2SFirst, true, LOG_MSG_RATIO, 0, false);

    let n = 1 << (LOG_SLOTS + 1);
    let m = n / 2;
    let base2k = params.base2k;
    let log_delta = 45;
    let log_modulus_in = log_delta + plan.eval_mod().log_msg_ratio;

    let k_boot = plan
        .bootstrap_k(log_modulus_in + 2 * log_delta, log_delta)
        .next_multiple_of(base2k);

    let module = Module::<BE>::new(n as u64);
    let host_module = Module::<HostBytesBackend>::new(n as u64);
    let encoder = ReferenceEncoder::<E>::new::<F>(m).unwrap();

    let tp = CKKSTestParams {
        n,
        base2k,
        k: k_boot,
        prec_meta: GLWEPlaintextMeta {
            scale: Scale::Log(log_delta),
            slots: SlotsKind::Complex,
            log_sparsity: 0,
        },
        prec_log_budget: 8,
        hw: 192,
        dsize: 7,
        rank: 1,
    };

    let keys_layout = BootstrappingKeysLayout {
        rotation_keys: uniform_rotation_keys(&plan, &tp, log_modulus_in, log_delta),
        first_factor_key: None,
        conversion_keys: None,
        tensor_key: tp.tsk_layout(),
        encapsulation: plan.sparse_secret_hamming_weight().map(|_| EncapsulationKeysLayout {
            dense_to_sparse: tp.ksk_layout(log_modulus_in),
            sparse_to_dense: Some(tp.ksk_layout(k_boot)),
        }),
    };
    let scratch_size = bootstrap_setup_tmp_bytes(
        &module,
        &ckks_spec(n, base2k, log_delta, k_boot - log_delta),
        &plan,
        &keys_layout,
    );
    let mut scratch = ScratchOwned::<BE>::alloc(scratch_size);

    let ctx = compile_context::<BE, F>(&module, base2k.into(), &plan);

    let (sk_raw, sk) = gen_sk_with_raw(&tp, &module, &host_module, [0u8; 32]);

    let boot_tmp = module.ckks_bootstrap_tmp_bytes(
        &ckks_spec(n, base2k, log_delta, k_boot - log_delta),
        &ckks_spec(n, base2k, log_delta, log_modulus_in - log_delta),
        &ctx,
        &keys_layout,
    );
    let (mut src_xs, mut src_xa, mut src_xe) = (Source::new([7u8; 32]), Source::new([1u8; 32]), Source::new([2u8; 32]));
    // Generated unprepared (serializable / GPU-resident), then prepared up front.
    let bsk = ctx
        .generate_keys(
            &module,
            &sk_raw,
            &keys_layout,
            &mut src_xs,
            &mut src_xe,
            &mut src_xa,
            &mut scratch.borrow(),
        )
        .unwrap()
        .prepare(&module, &mut scratch.borrow());

    let (re, im) = test_vector_1::<F>(m);

    let ct0 = ckks_encrypt_with_prec(
        &tp,
        &module,
        &host_module,
        &encoder,
        &sk,
        log_modulus_in,
        &re,
        &im,
        ckks_spec(n, base2k, log_delta, log_modulus_in - log_delta),
        &mut scratch.borrow(),
    );

    // Compare the public EvalRound+ orchestrator with the explicit pipeline below.
    let ct_bs = {
        let mut ct_bs = module.ckks_ciphertext_alloc(base2k.into(), k_boot.into());
        let logged = LoggedBootstrappingKeys::new::<BE>(&bsk);
        with_scratch::<BE, _>(boot_tmp, |s| module.ckks_bootstrap(&mut ct_bs, &ct0, &ctx, &logged, s)).unwrap();
        let widths = plan.key_widths(tp.log_n(), k_boot, ct0.k().as_usize(), ct0.log_delta());
        logged.assert_lookups_at(&widths, "EvalRound+, C2S-first");
        assert_eq!(ct_bs.log_delta(), log_delta);
        assert_eq!(
            ct_bs.k().as_usize(),
            k_boot - plan.consumed_bits() - (plan.eval_mod().f_mod_log_delta - log_delta)
        );
        let (re_bs, im_bs) = decrypt(&module, &encoder, &ct_bs, &sk, &mut scratch.borrow());
        for (got, want, tag) in [(&re_bs, &re, "re"), (&im_bs, &im, "im")] {
            let s = precision_stats(got, want, log_delta);
            println!(
                "ckks_bootstrap (evalround) ({tag}) avg={:.2} min={:.2} bits",
                s.avg_log2_prec, s.min_log2_prec
            );
            assert!(
                s.avg_log2_prec >= MIN_AVG_LOG2_PREC,
                "ckks_bootstrap evalround ({tag}): {:.1} bits < {MIN_AVG_LOG2_PREC}",
                s.avg_log2_prec
            );
        }
        ct_bs
    };

    // A real-tagged input must come back real-tagged, whichever pipeline the
    // recipe selects, and its imaginary part must stay zero.
    {
        let im_zero = vec![F::zero(); m];
        let mut ct_real = ckks_encrypt_with_prec(
            &tp,
            &module,
            &host_module,
            &encoder,
            &sk,
            log_modulus_in,
            &re,
            &im_zero,
            ckks_spec(n, base2k, log_delta, log_modulus_in - log_delta),
            &mut scratch.borrow(),
        );
        ct_real.set_slots(SlotsKind::Real);
        let mut ct_bs = module.ckks_ciphertext_alloc(base2k.into(), k_boot.into());
        with_scratch::<BE, _>(boot_tmp, |s| module.ckks_bootstrap(&mut ct_bs, &ct_real, &ctx, &bsk, s)).unwrap();
        assert_eq!(ct_bs.slots(), SlotsKind::Real, "evalround output slot kind");
        assert_eq!(ct_bs.log_delta(), log_delta);
        assert_eq!(
            ct_bs.k().as_usize(),
            k_boot - plan.consumed_bits() - (plan.eval_mod().f_mod_log_delta - log_delta)
        );
        let (re_bs, im_bs) = decrypt(&module, &encoder, &ct_bs, &sk, &mut scratch.borrow());
        assert!(precision_stats(&re_bs, &re, log_delta).avg_log2_prec >= MIN_AVG_LOG2_PREC);
        assert!(precision_stats(&im_bs, &im_zero, log_delta).avg_log2_prec >= 5.0);
    }

    // 1) The whole raise step: lift, (encapsulate) denseToSparse / ModUp /
    //    sparseToDense, relabel by the message ratio.
    let mut ct = module.ckks_ciphertext_alloc(base2k.into(), k_boot.into());
    // `ckks_bootstrap_tmp_bytes` is the query of the raise step.
    with_scratch::<BE, _>(boot_tmp, |s| {
        module.ckks_bootstrap_mod_up(&mut ct, &ct0, plan.eval_mod(), &bsk, s)
    })
    .unwrap();

    // 2) CoeffsToSlots (split): LP (low precision, `1/K` scaling) for the round, and
    //    HP (full precision, natural scaling) for the high-precision `Δm + I·q`.
    let mut r0_lp = module.ckks_ciphertext_alloc(base2k.into(), k_boot.into());
    let mut i0_lp = module.ckks_ciphertext_alloc(base2k.into(), k_boot.into());
    let bytes = dft_stage_bytes(&module, &keys_layout, &r0_lp, &ct, ctx.coeffs_to_slots());
    with_scratch::<BE, _>(bytes, |s| {
        module.ckks_coeffs_to_slots_split(&mut r0_lp, &mut i0_lp, &ct, ctx.coeffs_to_slots(), bsk.rotation_keys(), s)
    })
    .unwrap();
    let mut r0_hp = module.ckks_ciphertext_alloc(base2k.into(), k_boot.into());
    let mut i0_hp = module.ckks_ciphertext_alloc(base2k.into(), k_boot.into());
    let bytes = dft_stage_bytes(&module, &keys_layout, &r0_hp, &ct, ctx.coeffs_to_slots_bypass().unwrap());
    with_scratch::<BE, _>(bytes, |s| {
        module.ckks_coeffs_to_slots_split(
            &mut r0_hp,
            &mut i0_hp,
            &ct,
            ctx.coeffs_to_slots_bypass().unwrap(),
            bsk.rotation_keys(),
            s,
        )
    })
    .unwrap();

    // 3) EvalMod each LP half: the residue `Δm + e` at natural scale.
    let mut res_real = module.ckks_ciphertext_alloc(base2k.into(), k_boot.into());
    let mut res_imag = module.ckks_ciphertext_alloc(base2k.into(), k_boot.into());
    with_scratch::<BE, _>(
        module.ckks_eval_mod_tmp_bytes(&res_real, &r0_lp, ctx.eval_mod().unwrap(), &bsk.tensor_key()),
        |s| module.ckks_eval_mod(&mut res_real, &r0_lp, ctx.eval_mod().unwrap(), bsk.tensor_key(), s),
    )
    .unwrap();
    with_scratch::<BE, _>(
        module.ckks_eval_mod_tmp_bytes(&res_imag, &i0_lp, ctx.eval_mod().unwrap(), &bsk.tensor_key()),
        |s| module.ckks_eval_mod(&mut res_imag, &i0_lp, ctx.eval_mod().unwrap(), bsk.tensor_key(), s),
    )
    .unwrap();

    // 4) r1 = r0_hp − K·r0_lp + EvalMod(r0_lp) = IDFT(Δ·m).
    //    EvalMod emits the residue at natural scale while the LP C2S is at `1/K`, so
    //    `r0_lp` is scaled up by K first. Then
    //    `(Δm+I·q) − (Δm+I·q+e) + (Δm+e) = Δm`: the integer part and the LP error `e`
    //    both cancel, leaving the message at the HP CoeffsToSlots' precision.
    let log2_k = FMOD_INTERVAL.trailing_zeros() as usize;
    with_scratch::<BE, _>(module.ckks_mul_pow2_tmp_bytes(r0_lp.max_size()), |s| {
        module.ckks_mul_pow2_assign(log2_k, &mut r0_lp, s)
    })
    .unwrap();
    with_scratch::<BE, _>(module.ckks_mul_pow2_tmp_bytes(i0_lp.max_size()), |s| {
        module.ckks_mul_pow2_assign(log2_k, &mut i0_lp, s)
    })
    .unwrap();
    with_scratch::<BE, _>(module.ckks_sub_tmp_bytes(r0_hp.max_size()), |s| {
        module.ckks_sub_assign(&mut r0_hp, &r0_lp, s)
    })
    .unwrap();
    with_scratch::<BE, _>(module.ckks_sub_tmp_bytes(i0_hp.max_size()), |s| {
        module.ckks_sub_assign(&mut i0_hp, &i0_lp, s)
    })
    .unwrap();
    with_scratch::<BE, _>(module.ckks_add_tmp_bytes(r0_hp.max_size()), |s| {
        module.ckks_add_assign(&mut r0_hp, &res_real, s)
    })
    .unwrap();
    with_scratch::<BE, _>(module.ckks_add_tmp_bytes(i0_hp.max_size()), |s| {
        module.ckks_add_assign(&mut i0_hp, &res_imag, s)
    })
    .unwrap();

    // 5) SlotsToCoeffs (split): IDFT(Δ·m) slots → refreshed message coefficients.
    let mut ct_out = module.ckks_ciphertext_alloc(base2k.into(), k_boot.into());
    let bytes = dft_stage_bytes(&module, &keys_layout, &ct_out, &r0_hp, ctx.slots_to_coeffs());
    with_scratch::<BE, _>(bytes, |s| {
        module.ckks_slots_to_coeffs_split(&mut ct_out, &r0_hp, &i0_hp, ctx.slots_to_coeffs(), bsk.rotation_keys(), s)
    })
    .unwrap();

    ct_out.set_log_delta(log_delta);
    assert_same_bootstrap::<BE>(&ct_out, &ct_bs);
    let (re_out, im_out) = decrypt(&module, &encoder, &ct_out, &sk, &mut scratch.borrow());

    for (got, want, tag) in [(&re_out, &re, "re"), (&im_out, &im, "im")] {
        let s = precision_stats(got, want, log_delta);
        println!(
            "[evalround] BOOTSTRAP-PREC ({tag}) avg={:.2} min={:.2} bits",
            s.avg_log2_prec, s.min_log2_prec,
        );
        assert!(
            s.avg_log2_prec >= MIN_AVG_LOG2_PREC,
            "bootstrapping_evalround_e2e ({tag}): {:.1} bits < {MIN_AVG_LOG2_PREC} (worst got={} want={})",
            s.avg_log2_prec,
            s.worst_got,
            s.worst_want,
        );
    }
}

/// SlotsToCoeffs-first bootstrapping:
///
/// ```text
/// SlotsToCoeffs(split) ─► ModRaise ─► CoeffsToSlots(split) ─► EvalMod(×2) ─► relabel /2^R
/// ```
pub fn test_bootstrapping_s2c_first_e2e<BE, F, E>(params: CKKSTestParams)
where
    BE: TestContextBackend<Ring = Standard>,
    Module<BE>: TestContextModule<BE>
        + CKKSEncodingOps<BE, F>
        + CKKSBootstrappingOps<BE>
        + CKKSDFTOps<BE>
        + CKKSEvalModOps<BE>
        + CKKSDFTMatrixOps<BE, F>,
    F: TestScalar,
    E: ReimNew<F>,
    for<'a> <BE as Backend>::BufRef<'a>: HostDataRef,
    for<'a> <BE as Backend>::BufMut<'a>: HostDataMut,
    ScratchOwned<BE>: ScratchOwnedAlloc<BE>,
    CKKSCiphertextOwned<BE>: GLWEToBackendMut<BE> + GLWEToBackendRef<BE> + CKKSCtBounds + SetCKKSInfos,
    CKKSPlaintextOwned<BE>: GLWEToBackendRef<BE> + LWEInfos,
    GLWETensorKeyPrepared<BE::OwnedBuf, BE>: GLWETensorKeyPreparedToBackendRef<BE> + GGLWEInfos,
{
    for (log_delta, log_msg_ratio, eval_round_plus, guard_bits, case) in [
        (40, 16, false, 0, "standard"),
        (40, 16, false, 6, "guarded"),
        (40, 16, true, 0, "evalround+"),
        (40, 16, true, 6, "guarded_evalround+"),
        (35, 13, false, 6, "guard_precision"),
    ] {
        let (re, im) = run_s2c_first_case::<BE, F, E>(params.base2k, log_delta, log_msg_ratio, eval_round_plus, guard_bits);
        for (avg, tag) in [(re, "re"), (im, "im")] {
            println!("[s2c_first/{case}] BOOTSTRAP-PREC ({tag}) avg={avg:.2} bits");
            assert!(
                avg >= 24.0,
                "bootstrapping_s2c_first_e2e ({case}/{tag}): {avg:.1} bits < 24.0"
            );
        }
    }
}

fn run_s2c_first_case<BE, F, E>(
    base2k: usize,
    log_delta: usize,
    log_msg_ratio: usize,
    eval_round_plus: bool,
    guard_bits: usize,
) -> (f64, f64)
where
    BE: TestContextBackend<Ring = Standard>,
    Module<BE>: TestContextModule<BE>
        + CKKSEncodingOps<BE, F>
        + CKKSBootstrappingOps<BE>
        + CKKSDFTOps<BE>
        + CKKSEvalModOps<BE>
        + CKKSDFTMatrixOps<BE, F>,
    F: TestScalar,
    E: ReimNew<F>,
    for<'a> <BE as Backend>::BufRef<'a>: HostDataRef,
    for<'a> <BE as Backend>::BufMut<'a>: HostDataMut,
    ScratchOwned<BE>: ScratchOwnedAlloc<BE>,
    CKKSCiphertextOwned<BE>: GLWEToBackendMut<BE> + GLWEToBackendRef<BE> + CKKSCtBounds + SetCKKSInfos,
    CKKSPlaintextOwned<BE>: GLWEToBackendRef<BE> + LWEInfos,
    GLWETensorKeyPrepared<BE::OwnedBuf, BE>: GLWETensorKeyPreparedToBackendRef<BE> + GGLWEInfos,
{
    let plan = bootstrap_plan(
        BootstrappingPipeline::S2CFirst,
        eval_round_plus,
        log_msg_ratio,
        guard_bits,
        false,
    );

    let n = 1 << (LOG_SLOTS + 1);
    let m = n / 2;
    let log_modulus_in = log_delta + plan.eval_mod().log_msg_ratio;
    let k_in = plan.input_k(log_modulus_in);
    let k_boot = plan.bootstrap_k(3 * log_delta, log_delta).next_multiple_of(base2k);

    let module = Module::<BE>::new(n as u64);
    let host_module = Module::<HostBytesBackend>::new(n as u64);
    let encoder = ReferenceEncoder::<E>::new::<F>(m).unwrap();

    let tp = CKKSTestParams {
        n,
        base2k,
        k: k_boot,
        prec_meta: GLWEPlaintextMeta {
            scale: Scale::Log(log_delta),
            slots: SlotsKind::Complex,
            log_sparsity: 0,
        },
        prec_log_budget: 8,
        hw: 192,
        dsize: 7,
        rank: 1,
    };

    let keys_layout = BootstrappingKeysLayout {
        rotation_keys: uniform_rotation_keys(&plan, &tp, k_in, log_delta),
        first_factor_key: None,
        conversion_keys: None,
        tensor_key: tp.tsk_layout(),
        encapsulation: plan.sparse_secret_hamming_weight().map(|_| EncapsulationKeysLayout {
            dense_to_sparse: tp.ksk_layout(log_modulus_in),
            // The raise is the sparse-to-dense switch, with rows under the raised ciphertext.
            sparse_to_dense: Some(tp.ksk_offset_layout(k_boot, plan.raised_window(log_modulus_in))),
        }),
    };
    let scratch_size = bootstrap_setup_tmp_bytes(
        &module,
        &ckks_spec(n, base2k, log_delta, k_boot - log_delta),
        &plan,
        &keys_layout,
    );
    let mut scratch = ScratchOwned::<BE>::alloc(scratch_size);

    let ctx = compile_context::<BE, F>(&module, base2k.into(), &plan);

    let (sk_raw, sk) = gen_sk_with_raw(&tp, &module, &host_module, [0u8; 32]);

    let boot_tmp = module.ckks_bootstrap_tmp_bytes(
        &ckks_spec(n, base2k, log_delta, k_boot - log_delta),
        &ckks_spec(n, base2k, log_delta, k_in - log_delta),
        &ctx,
        &keys_layout,
    );
    let (mut src_xs, mut src_xa, mut src_xe) = (Source::new([7u8; 32]), Source::new([1u8; 32]), Source::new([2u8; 32]));
    let bsk = ctx
        .generate_keys(
            &module,
            &sk_raw,
            &keys_layout,
            &mut src_xs,
            &mut src_xe,
            &mut src_xa,
            &mut scratch.borrow(),
        )
        .unwrap()
        .prepare(&module, &mut scratch.borrow());

    let (re, im) = test_vector_1::<F>(m);
    let ct0 = ckks_encrypt_with_prec(
        &tp,
        &module,
        &host_module,
        &encoder,
        &sk,
        k_in,
        &re,
        &im,
        ckks_spec(n, base2k, log_delta, k_in - log_delta),
        &mut scratch.borrow(),
    );

    let (bs_re, bs_im) = {
        let mut ct_bs = module.ckks_ciphertext_alloc(base2k.into(), k_boot.into());
        let logged = LoggedBootstrappingKeys::new::<BE>(&bsk);
        with_scratch::<BE, _>(boot_tmp, |s| module.ckks_bootstrap(&mut ct_bs, &ct0, &ctx, &logged, s)).unwrap();
        let widths = plan.key_widths(tp.log_n(), k_boot, ct0.k().as_usize(), ct0.log_delta());
        logged.assert_lookups_at(&widths, "S2C-first");
        assert_eq!(ct_bs.k().as_usize(), k_boot - plan.post_mod_up_consumed_bits());
        assert_eq!(ct_bs.log_delta(), log_delta);
        decrypt(&module, &encoder, &ct_bs, &sk, &mut scratch.borrow())
    };

    {
        let im_zero = vec![F::zero(); m];
        let mut ct_real = ckks_encrypt_with_prec(
            &tp,
            &module,
            &host_module,
            &encoder,
            &sk,
            k_in,
            &re,
            &im_zero,
            ckks_spec(n, base2k, log_delta, k_in - log_delta),
            &mut scratch.borrow(),
        );
        // Declaring the slots real selects the single-EvalMod pipeline.
        ct_real.set_slots(SlotsKind::Real);
        let (real_bs_re, real_bs_im) = {
            let mut ct_bs = module.ckks_ciphertext_alloc(base2k.into(), k_boot.into());
            with_scratch::<BE, _>(boot_tmp, |s| module.ckks_bootstrap(&mut ct_bs, &ct_real, &ctx, &bsk, s)).unwrap();
            assert_eq!(ct_bs.slots(), SlotsKind::Real);
            assert_eq!(ct_bs.k().as_usize(), k_boot - plan.post_mod_up_consumed_bits());
            assert_eq!(ct_bs.log_delta(), log_delta);
            decrypt(&module, &encoder, &ct_bs, &sk, &mut scratch.borrow())
        };
        for (got, want) in [(&real_bs_re, &re), (&real_bs_im, &im_zero)] {
            let avg = precision_stats(got, want, log_delta).avg_log2_prec;
            assert!(avg >= 24.0, "real-slot S2C precision: {avg:.1} bits < 24.0");
        }
    }

    let insufficient_k = log_delta + plan.pre_mod_up_consumed_bits() - 1;
    let ct_insufficient = ckks_encrypt_with_prec(
        &tp,
        &module,
        &host_module,
        &encoder,
        &sk,
        insufficient_k,
        &re,
        &im,
        ckks_spec(n, base2k, log_delta, insufficient_k - log_delta),
        &mut scratch.borrow(),
    );
    let mut ct_out = fixture_ciphertext(&module, &ckks_spec(n, base2k, log_delta, k_boot - log_delta), 24);
    // The S2C-first budget refusal comes from its first stage, after the pipeline carved its working ciphertexts.
    let before = snapshot::<BE, _>(&ct_out);
    let err = with_scratch::<BE, _>(boot_tmp, |s| {
        module.ckks_bootstrap(&mut ct_out, &ct_insufficient, &ctx, &bsk, s)
    })
    .unwrap_err();
    assert!(matches!(
        err.composition(),
        Some(CKKSCompositionError::MultiplicationPrecisionUnderflow { .. })
    ));
    assert_eq!(snapshot::<BE, _>(&ct_out), before, "a refused bootstrap wrote res");

    let s_re = precision_stats(&bs_re, &re, log_delta);
    let s_im = precision_stats(&bs_im, &im, log_delta);
    (s_re.avg_log2_prec, s_im.avg_log2_prec)
}

fn decrypt<BE: Backend<ZnxWord = i64> + TestContextBackend, F, E, S>(
    module: &Module<BE>,
    encoder: &ReferenceEncoder<E>,
    ct: &CKKSCiphertextOwned<BE>,
    sk: &S,
    scratch: &mut ScratchArena<'_, BE>,
) -> (Vec<F>, Vec<F>)
where
    F: TestScalar,
    Module<BE>: CKKSDecryptOps<BE>,
    E: ReimNew<F>,
    S: GLWESecretPreparedToBackendRef<BE> + GLWEInfos,
{
    assert_canonical_at_k::<BE>("decrypt", ct);
    // Decrypt, decode, and confirm the slots are recovered. Cap the budget so
    // `log_delta + log_budget <= 127` fits the i128 decode codec (the unused
    // high-order budget is dropped losslessly).
    let prec = meta(ct.log_delta(), ct.log_budget().min(127usize.saturating_sub(ct.log_delta())));
    let mut pt_out = module.ckks_pt_vec_alloc(module.ckks_max_slots(), ct.base2k(), prec.k);
    pt_out.set_meta(prec.meta);
    module.ckks_decrypt(&mut pt_out, ct, sk, &mut scratch.borrow()).unwrap();
    let m = 1 << (ct.log_n() - ct.log_sparsity() - 1);

    let pt_host = pt_out.to_host_owned::<BE>();
    let (mut re_out, mut im_out) = (vec![F::zero(); m], vec![F::zero(); m]);
    encoder.decode_reim(&pt_host, &mut re_out, &mut im_out).unwrap();

    (re_out, im_out)
}

fn assert_same_bootstrap<BE: Backend>(got: &CKKSCiphertextOwned<BE>, want: &CKKSCiphertextOwned<BE>) {
    assert_eq!(got.meta(), want.meta());
    assert_eq!(got.k(), want.k());
    let got = got.to_host_owned::<BE>();
    let want = want.to_host_owned::<BE>();
    for col in 0..got.data().cols() {
        for limb in 0..got.size() {
            assert_eq!(
                got.data().at(col, limb),
                want.data().at(col, limb),
                "bootstrap output col={col} limb={limb}"
            );
        }
    }
}
