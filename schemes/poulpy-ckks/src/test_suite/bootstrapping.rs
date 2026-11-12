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

use poulpy_core::GLWECIKeyEncryptSk;
use poulpy_core::api::{GLWECIKeyPreparedFactory, GLWESecretSampling, GLWESwitchingKeyPreparedFactory};
use poulpy_core::layouts::{
    GGLWEInfos, GLWEAutomorphismKeyLayout, GLWEInfos, GLWEPlaintextInfos, GLWESecretAlloc, GLWESecretPreparedToBackendRef,
    GLWETensorKeyPrepared, GLWEToBackendMut, GLWEToBackendRef, LWEInfos, SetGLWEPlaintextInfos,
    prepared::GLWETensorKeyPreparedToBackendRef,
};
use poulpy_hal::{
    api::{ReimNew, ScratchOwnedAlloc, ScratchOwnedBorrow},
    layouts::{
        Backend, ConjugateInvariant, HostBytesBackend, HostDataMut, HostDataRef, Module, Ring, ScratchArena, ScratchOwned,
        Standard, ZnxView,
    },
    source::Source,
};

use crate::SlotsKind;
use crate::{
    CKKSCompositionError, CKKSCtBounds, CKKSInfos, CoeffsMeta, GLWEPlaintextMeta, SetCKKSInfos, SlotLayout,
    api::{
        CKKSAddOps, CKKSBootstrappingOps, CKKSCopyOps, CKKSDFTMatrixOps, CKKSDFTMatrixPreparedFactory, CKKSDFTOps,
        CKKSDecryptOps, CKKSEvalModOps, CKKSPow2Ops, CKKSSubOps,
    },
    ckks_log_delta,
    layouts::{
        BootstrappingKeysLayout, BootstrappingPipeline, BootstrappingPlan, BootstrappingTechniques, CKKSCiphertextAlloc,
        CKKSDFTFactors, CKKSEvalModInfos, CKKSPlaintextAlloc, DFTOutputFormat, DFTPlan, DFTType, EncapsulationKeysLayout,
        EvalRoundPlus, ModRaisePlan, SparseSecretEncapsulation,
        eval_mod::{EvalModPlan, EvalModType},
    },
    polynomial::SplitStrategy,
    test_suite::reference_encoder::ReferenceEncoder,
    test_suite::{
        CKKSTestParams,
        helpers::{
            LoggedBootstrappingKeys, TestContextBackend, TestContextModule, TestContextSharedModule, TestScalar,
            assert_canonical_at_k, assert_ckks_error, assert_invalid_operand, ckks_encrypt_with_prec, ckks_spec, gen_sk_with_raw,
            next_test_seed, precision_stats, test_vector_1,
        },
        parity::helpers::{fixture_ciphertext, snapshot, with_scratch},
        presets::{bootstrap_setup_tmp_bytes, compile_context},
    },
};

/// Scratch of one DFT stage from `src` to `dst`, under the largest of the rotation key layouts of `keys`.
fn dft_stage_bytes<BE, Dir, Fmt, D, Dst, Src>(
    module: &Module<BE>,
    keys: &BootstrappingKeysLayout,
    dst: &Dst,
    src: &Src,
    dft: &D,
) -> usize
where
    BE: Backend,
    Module<BE>: CKKSDFTOps<BE>,
    D: CKKSDFTFactors<BE, Dir, Fmt>,
    Dst: CKKSCtBounds,
    Src: CKKSCtBounds,
{
    keys.rotation_key_layouts()
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
    BE: TestContextBackend,
    Module<BE>: TestContextSharedModule<BE, ConjugateInvariant> + CKKSEncodingOps<BE, F, ConjugateInvariant>,
    Module<BE>: CKKSDFTMatrixPreparedFactory<BE>
        + GLWECIKeyEncryptSk<BE>
        + GLWECIKeyPreparedFactory<BE>
        + TestContextModule<BE>
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
                slot_layout: SlotLayout::Full,
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
        let key_set = with_scratch::<BE, _>(keys_layout.generate_tmp_bytes(&module), |s| {
            ctx.generate_keys(&module, &sk_raw, &keys_layout, &mut src_xs, &mut src_xe, &mut src_xa, s)
        })
        .unwrap();
        assert_eq!(key_set.first_factor_keys.len(), first.giant_steps.len());
        let mut bsk = with_scratch::<BE, _>(keys_layout.prepare_tmp_bytes(&module), |s| key_set.prepare(&module, s)).unwrap();

        // The raise alone runs on exactly the scratch it advertises.
        let k_small = if s2c_first { log_modulus_in } else { k_in };
        let small_spec = ckks_spec(n, base2k, log_delta, k_small - log_delta);
        let raise_tmp = module.ckks_mod_raise_tmp_bytes(&boot_spec, &small_spec, &ctx, &keys_layout);
        assert!(raise_tmp > 0);
        {
            let mut src = module.ckks_ciphertext_alloc(base2k.into(), k_small.into());
            src.set_meta(small_spec.meta);
            let mut dst = module.ckks_ciphertext_alloc(base2k.into(), k_boot.into());
            with_scratch::<BE, _>(raise_tmp, |s| module.ckks_mod_raise(&mut dst, &mut src, &ctx, &bsk, s)).unwrap();
        }
        {
            let mut src = fixture_ciphertext(&module, &small_spec, 31);
            let mut dst = fixture_ciphertext(&module, &boot_spec, 37);
            assert_mod_raise_refuses(
                &mut dst,
                &mut src,
                "the plan moves the first CoeffsToSlots factor onto the raise",
                |res, a, s| module.ckks_mod_raise(res, a, &plan.mod_raise(), &bsk, s),
            );
            if encapsulated {
                let plain = ModRaisePlan {
                    first_factor_on_raise: false,
                    ..plan.mod_raise()
                };
                assert_mod_raise_refuses(
                    &mut dst,
                    &mut src,
                    "the encapsulated ModUp needs a sparse-to-dense key",
                    |res, a, s| module.ckks_mod_raise(res, a, &plain, &bsk, s),
                );
                // One bit of headroom gives C2S-first a lift, which the refusal of a key under that width precedes.
                let lifted_k = k_small + 1;
                let mut lifted = fixture_ciphertext(&module, &ckks_spec(n, base2k, log_delta, lifted_k - log_delta), 47);
                if let Some(lift) = plan.mod_raise().lift() {
                    assert_eq!(lift.input_scale_up("ckks_mod_raise", lifted_k, log_delta).unwrap(), 1);
                }
                let mut narrow_key = keys_layout.encapsulation.as_ref().unwrap().dense_to_sparse;
                narrow_key.dnum = 1usize.into();
                narrow_key.k_digit = k_small.into();
                assert!(narrow_key.gglwe_layout().max_input_k().as_usize() < lifted_k);
                let dense_to_sparse = module.glwe_switching_key_prepared_alloc_from_infos(&narrow_key);
                let wide_keys = bsk.encapsulation_keys.replace((dense_to_sparse, None));
                assert_mod_raise_refuses(
                    &mut dst,
                    &mut lifted,
                    "the input is wider than an encapsulation key's max_input_k",
                    |res, a, s| module.ckks_mod_raise(res, a, &ctx, &bsk, s),
                );
                bsk.encapsulation_keys = wide_keys;
            }
            let narrow_spec = ckks_spec(n, base2k, log_delta, k_boot - base2k - log_delta);
            assert_mod_raise_refuses(
                &mut fixture_ciphertext(&module, &narrow_spec, 41),
                &mut src,
                "the first-factor keys switch",
                |res, a, s| module.ckks_mod_raise(res, a, ctx.mod_raise(), &bsk, s),
            );
        }
        {
            let mut src = fixture_ciphertext(&module, &small_spec, 49);
            let mut dst = fixture_ciphertext(&module, &boot_spec, 51);
            let before = (snapshot::<BE, _, Standard>(&dst), snapshot::<BE, _, Standard>(&src));
            let err =
                with_scratch::<BE, _>(raise_tmp - 1, |s| module.ckks_mod_raise(&mut dst, &mut src, &ctx, &bsk, s)).unwrap_err();
            assert!(
                err.composition().is_none() && err.to_string().contains("needs a scratch arena"),
                "{err}"
            );
            assert_eq!(
                (snapshot::<BE, _, Standard>(&dst), snapshot::<BE, _, Standard>(&src)),
                before,
                "a refused raise wrote an operand"
            );
        }

        // Keys laid out for the raise refuse a narrower bootstrap before any work.
        let mut narrow = fixture_ciphertext(&module, &ckks_spec(n, base2k, log_delta, k_boot - base2k - log_delta), 23);
        let ct_any = module.ckks_ciphertext_alloc(base2k.into(), k_in.into());
        let before = snapshot::<BE, _, Standard>(&narrow);
        let err = with_scratch::<BE, _>(0, |s| module.ckks_bootstrap(&mut narrow, &ct_any, &ctx, &bsk, s)).unwrap_err();
        assert_invalid_operand(
            "a bootstrap narrower than its raise keys",
            &err,
            "ckks_bootstrap",
            "the first-factor keys switch",
        );
        assert_eq!(snapshot::<BE, _, Standard>(&narrow), before, "a refused bootstrap wrote res");
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
        assert_eq!(snapshot::<BE, _, Standard>(&narrow), before, "a refused bootstrap wrote res");
        let mut half = in_spec;
        half.glwe_layout.n = (n / 2).into();
        let ct_half = fixture_ciphertext(&module, &half, 31);
        let mut res_half = fixture_ciphertext(&module, &ckks_spec(n / 2, base2k, log_delta, k_boot - base2k - log_delta), 37);
        let before_half = snapshot::<BE, _, Standard>(&res_half);
        let err = with_scratch::<BE, _>(0, |s| module.ckks_bootstrap(&mut res_half, &ct_half, &ctx, &bsk, s)).unwrap_err();
        assert_ckks_error(
            "a bootstrap of ciphertexts below the module degree",
            &err,
            CKKSCompositionError::InvalidOperand {
                op: "ckks_bootstrap",
                reason: "the transforms of the context must fill the slots of the input's degree, or of the subring of its gap"
                    .into(),
            },
        );
        assert_eq!(
            snapshot::<BE, _, Standard>(&res_half),
            before_half,
            "a refused bootstrap wrote res"
        );
        let mut other_radix = in_spec;
        other_radix.glwe_layout.base2k = (base2k - 1).into();
        let ct_radix = fixture_ciphertext(&module, &other_radix, 43);
        let err = with_scratch::<BE, _>(0, |s| module.ckks_bootstrap(&mut narrow, &ct_radix, &ctx, &bsk, s)).unwrap_err();
        assert_ckks_error(
            "a bootstrap of two radices",
            &err,
            CKKSCompositionError::InvalidOperand {
                op: "ckks_bootstrap",
                reason: "takes ciphertexts of one base2k".into(),
            },
        );
        assert_eq!(snapshot::<BE, _, Standard>(&narrow), before, "a refused bootstrap wrote res");

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

/// `run`, a call of [`CKKSBootstrappingOps::ckks_bootstrap_eval_mod`], refuses for `reason` on a zero-byte arena,
/// which puts the check before the scratch check, and leaves `res` as it was.
pub(crate) fn assert_eval_mod_refuses<BE, Rg: Ring>(
    res: &mut CKKSCiphertextOwned<BE, Rg>,
    a: &CKKSCiphertextOwned<BE, Rg>,
    reason: &str,
    run: impl FnOnce(
        &mut CKKSCiphertextOwned<BE, Rg>,
        &CKKSCiphertextOwned<BE, Rg>,
        &mut ScratchArena<'_, BE>,
    ) -> crate::CKKSResult<()>,
) where
    BE: Backend<ZnxWord = i64>,
    CKKSCiphertextOwned<BE, Rg>: GLWEToBackendRef<BE, Rg> + CKKSCtBounds,
{
    let before = snapshot::<BE, _, Rg>(res);
    let err = with_scratch::<BE, _>(0, |s| run(res, a, s)).unwrap_err();
    assert_ckks_error(
        reason,
        &err,
        CKKSCompositionError::InvalidOperand {
            op: "ckks_bootstrap_eval_mod",
            reason: reason.into(),
        },
    );
    assert_eq!(snapshot::<BE, _, Rg>(res), before, "a refused step wrote res");
}

/// `run`, a call of [`CKKSBootstrappingOps::ckks_mod_raise`], refuses for `reason` on a zero-byte arena, which puts the
/// check before the scratch check, and leaves `res` and `a` as they were.
pub(crate) fn assert_mod_raise_refuses<BE, Rg: Ring>(
    res: &mut CKKSCiphertextOwned<BE, Rg>,
    a: &mut CKKSCiphertextOwned<BE, Rg>,
    reason: &str,
    run: impl FnOnce(
        &mut CKKSCiphertextOwned<BE, Rg>,
        &mut CKKSCiphertextOwned<BE, Rg>,
        &mut ScratchArena<'_, BE>,
    ) -> crate::CKKSResult<()>,
) where
    BE: Backend<ZnxWord = i64>,
    CKKSCiphertextOwned<BE, Rg>: GLWEToBackendRef<BE, Rg> + CKKSCtBounds,
{
    let before = (snapshot::<BE, _, Rg>(res), snapshot::<BE, _, Rg>(a));
    let err = with_scratch::<BE, _>(0, |s| run(res, a, s)).unwrap_err();
    assert_invalid_operand(reason, &err, "ckks_mod_raise", reason);
    assert_eq!(
        (snapshot::<BE, _, Rg>(res), snapshot::<BE, _, Rg>(a)),
        before,
        "a refused raise wrote an operand"
    );
}

/// [`CKKSBootstrappingOps::ckks_bootstrap_eval_mod`] in a hand composition that stamps nothing gives the snapshot of
/// `ckks_bootstrap`, on every pipeline, on real and complex messages, with and without EvalRound+.
/// [`CKKSBootstrappingOps::ckks_mod_raise`] takes the literal plan, the compiled raise and the context with one result.
pub fn test_ckks_bootstrap_eval_mod<BE>()
where
    BE: TestContextBackend<ZnxWord = i64>,
    Module<BE>: TestContextSharedModule<BE, ConjugateInvariant> + CKKSEncodingOps<BE, f64, ConjugateInvariant>,
    Module<BE>: CKKSDFTMatrixPreparedFactory<BE>
        + GLWECIKeyEncryptSk<BE>
        + GLWECIKeyPreparedFactory<BE>
        + TestContextModule<BE>
        + CKKSEncodingOps<BE, f64>
        + CKKSBootstrappingOps<BE>
        + CKKSDFTOps<BE>
        + CKKSEvalModOps<BE>
        + CKKSDFTMatrixOps<BE, f64>,
    for<'a> <BE as Backend>::BufRef<'a>: HostDataRef,
    for<'a> <BE as Backend>::BufMut<'a>: HostDataMut,
    CKKSCiphertextOwned<BE>: GLWEToBackendMut<BE> + GLWEToBackendRef<BE> + CKKSCtBounds + SetCKKSInfos,
    CKKSPlaintextOwned<BE>: GLWEToBackendRef<BE> + LWEInfos,
    GLWETensorKeyPrepared<BE::OwnedBuf, BE>: GLWETensorKeyPreparedToBackendRef<BE> + GGLWEInfos,
{
    use crate::presets::bootstrapping::{insecure_n11_d35_k100_p17_c2s, insecure_n11_d35_k180_p20_s2c};

    let s2c = insecure_n11_d35_k180_p20_s2c().unwrap();
    for preset in [
        s2c.clone(),
        insecure_n11_d35_k100_p17_c2s().unwrap(),
        s2c.with_eval_round_plus(30, 50).unwrap(),
        insecure_n11_d35_k100_p17_c2s().unwrap().with_eval_round_plus(30, 51).unwrap(),
    ] {
        let module = Module::<BE>::new(preset.n() as u64);
        let boot = preset.bootstrap_layout();
        let keys_layout = preset.keys_layout();
        let mut sk = module.glwe_secret_alloc_from_infos(&boot.glwe_layout);
        module.glwe_secret_fill_ternary_hw(&mut sk, preset.dense_secret_hamming_weight(), &mut Source::new([9; 32]));
        let context = compile_context::<BE, f64>(&module, preset.base2k().into(), preset.plan());
        let (mut xs, mut xa, mut xe) = (Source::new([7; 32]), Source::new([1; 32]), Source::new([2; 32]));
        let key_set = with_scratch::<BE, _>(keys_layout.generate_tmp_bytes(&module), |s| {
            preset
                .plan()
                .generate_keys(&module, &sk, keys_layout, &mut xs, &mut xe, &mut xa, s)
        })
        .unwrap();
        let mut keys = with_scratch::<BE, _>(keys_layout.prepare_tmp_bytes(&module), |s| {
            key_set.transfer_and_prepare(&module, s)
        })
        .unwrap();

        let mut digits = Vec::new();
        for slots in [SlotsKind::Real, SlotsKind::Complex] {
            let mut layout = preset.input_layout();
            layout.meta.slots = slots;
            let input = fixture_ciphertext(&module, &layout, 5);
            let mut whole = fixture_ciphertext(&module, &boot, 67);
            let bytes = module.ckks_bootstrap_tmp_bytes(&boot, &input, &context, keys_layout);
            with_scratch::<BE, _>(bytes, |s| module.ckks_bootstrap(&mut whole, &input, &context, &keys, s)).unwrap();
            digits.push(snapshot::<BE, _, Standard>(&whole).digits);

            let run = context.run(input.meta());
            let mut coeffs = module.ckks_ciphertext_alloc_from_infos(&input);
            let mut raised = module.ckks_ciphertext_alloc_from_glwe_infos(&boot);
            let mut z_hp = module.ckks_ciphertext_alloc_from_glwe_infos(&boot);
            let mut stepped = fixture_ciphertext(&module, &boot, 71);
            let s2c_first = context.pipeline() == BootstrappingPipeline::S2CFirst;
            with_scratch::<BE, _>(bytes, |s| {
                if s2c_first {
                    module.ckks_double_into(&mut coeffs, &input, s)?;
                    module.ckks_slots_to_coeffs_assign(&mut coeffs, &run, &keys, s)?;
                } else {
                    module.ckks_copy(&mut coeffs, &input, s)?;
                }
                module.ckks_mod_raise(&mut raised, &mut coeffs, &run, &keys, s)?;
                if let Some(bypass) = run.bypass() {
                    module.ckks_copy(&mut z_hp, &raised, s)?;
                    module.ckks_coeffs_to_slots_assign(&mut z_hp, &bypass, &keys, s)?;
                }
                module.ckks_coeffs_to_slots_assign(&mut raised, &run, &keys, s)?;
                module.ckks_bootstrap_eval_mod(&mut stepped, &raised, &run, &keys, s)?;
                if run.bypass().is_some() {
                    module.ckks_mul_pow2_assign(1, &mut z_hp, s)?;
                    module.ckks_add_assign(&mut z_hp, &stepped, s)?;
                    module.ckks_copy(&mut stepped, &z_hp, s)?;
                    run.stamp_eval_mod_output("ckks_bootstrap", &mut stepped)?;
                }
                if !s2c_first {
                    module.ckks_slots_to_coeffs_assign(&mut stepped, &run, &keys, s)?;
                }
                Ok::<_, crate::CKKSError>(())
            })
            .unwrap();
            assert_eq!(
                snapshot::<BE, _, Standard>(&stepped),
                snapshot::<BE, _, Standard>(&whole),
                "{:?} {slots:?}",
                context.pipeline()
            );

            // The step runs on exactly the scratch it advertises, and refuses one byte less.
            let mut exact = module.ckks_ciphertext_alloc_from_glwe_infos(&boot);
            let eval_mod_tmp = module.ckks_bootstrap_eval_mod_tmp_bytes(&exact, &raised, &run, keys_layout);
            assert!(eval_mod_tmp > 0);
            with_scratch::<BE, _>(eval_mod_tmp, |s| {
                module.ckks_bootstrap_eval_mod(&mut exact, &raised, &run, &keys, s)
            })
            .unwrap();
            let mut short = fixture_ciphertext(&module, &boot, 73);
            let before = snapshot::<BE, _, Standard>(&short);
            let err = with_scratch::<BE, _>(eval_mod_tmp - 1, |s| {
                module.ckks_bootstrap_eval_mod(&mut short, &raised, &run, &keys, s)
            })
            .unwrap_err();
            assert!(
                err.composition().is_none() && err.to_string().contains("needs a scratch arena"),
                "{err}"
            );
            assert_eq!(snapshot::<BE, _, Standard>(&short), before, "a refused step wrote res");

            let mut rank2 = boot;
            rank2.glwe_layout.rank = 2usize.into();
            let a_rank2 = fixture_ciphertext(&module, &rank2, 29);
            assert_eval_mod_refuses(
                &mut stepped,
                &a_rank2,
                "supports rank-1 ciphertexts only, got rank 2 -> 1",
                |res, a, s| module.ckks_bootstrap_eval_mod(res, a, &run, &keys, s),
            );
            let mut half = boot;
            half.glwe_layout.n = (preset.n() / 2).into();
            let degree = "takes ciphertexts of one degree, which the module serves";
            assert_eval_mod_refuses(&mut fixture_ciphertext(&module, &half, 11), &raised, degree, |res, a, s| {
                module.ckks_bootstrap_eval_mod(res, a, &run, &keys, s)
            });
            let small = Module::<BE>::new(preset.n() as u64 / 2);
            assert_eval_mod_refuses(&mut stepped, &raised, degree, |res, a, s| {
                small.ckks_bootstrap_eval_mod(res, a, &run, &keys, s)
            });
            let mut other_radix = boot;
            other_radix.glwe_layout.base2k = (boot.base2k().as_usize() - 1).into();
            assert_eval_mod_refuses(
                &mut fixture_ciphertext(&module, &other_radix, 13),
                &raised,
                "takes ciphertexts of one base2k",
                |res, a, s| module.ckks_bootstrap_eval_mod(res, a, &run, &keys, s),
            );
            // A tensor key narrower than EvalMod's widest product refuses before the first write.
            let covered = keys.tensor_key.as_ref().unwrap().gglwe_layout().max_input_k().as_usize();
            let mut wide = boot;
            wide.glwe_layout.k = (covered + raised.log_delta() + 1).into();
            wide.meta = raised.meta();
            let wide = fixture_ciphertext(&module, &wide, 17);
            let before = snapshot::<BE, _, Standard>(&stepped);
            let err =
                with_scratch::<BE, _>(0, |s| module.ckks_bootstrap_eval_mod(&mut stepped, &wide, &run, &keys, s)).unwrap_err();
            assert!(
                matches!(
                    err.composition(),
                    Some(CKKSCompositionError::MissingRelinearizationKey {
                        op: "ckks_bootstrap_eval_mod",
                        ..
                    })
                ),
                "{err}"
            );
            assert_eq!(snapshot::<BE, _, Standard>(&stepped), before, "a refused step wrote res");
            // The budget of EvalMod refuses before the route runs.
            let mut starved = boot;
            starved.glwe_layout.k = (raised.log_delta() + 1).into();
            starved.meta = raised.meta();
            let starved = fixture_ciphertext(&module, &starved, 19);
            let before = snapshot::<BE, _, Standard>(&starved);
            let err =
                with_scratch::<BE, _>(0, |s| module.ckks_bootstrap_eval_mod(&mut stepped, &starved, &run, &keys, s)).unwrap_err();
            assert!(
                matches!(
                    err.composition(),
                    Some(CKKSCompositionError::InsufficientHomomorphicCapacity {
                        op: "ckks_bootstrap_eval_mod",
                        ..
                    })
                ),
                "{err}"
            );
            assert_eq!(
                snapshot::<BE, _, Standard>(&starved),
                before,
                "a refused step wrote its input"
            );

            // A plain EvalMod has no message: the label of its input picks the route, and it stamps nothing.
            if slots == SlotsKind::Complex {
                let plain = context.eval_mod().unwrap();
                let mut plain_digits = Vec::new();
                for label in [SlotsKind::Real, SlotsKind::Complex] {
                    raised.set_slots(label);
                    let mut out = module.ckks_ciphertext_alloc_from_glwe_infos(&boot);
                    let bytes = module.ckks_bootstrap_eval_mod_tmp_bytes(&out, &raised, plain, keys_layout);
                    with_scratch::<BE, _>(bytes, |s| module.ckks_bootstrap_eval_mod(&mut out, &raised, plain, &keys, s)).unwrap();
                    assert_eq!(out.log_delta(), raised.log_delta(), "a plain EvalMod stamped its output");
                    plain_digits.push(snapshot::<BE, _, Standard>(&out).digits);
                }
                assert_ne!(
                    plain_digits[0], plain_digits[1],
                    "a plain EvalMod ignored the label of its input"
                );
            }
        }
        // Real slots take one EvalMod on an S2C-first plan without a bypass, and only there.
        let one_eval_mod = context.pipeline() == BootstrappingPipeline::S2CFirst && context.coeffs_to_slots_bypass().is_none();
        assert!(
            (digits[0] != digits[1]) == one_eval_mod,
            "the real route does not run where it should"
        );

        // The input of the raise, on a composition: `input` itself or the output of SlotsToCoeffs.
        let input = fixture_ciphertext(&module, &preset.input_layout(), 5);
        let bytes = module.ckks_bootstrap_tmp_bytes(&boot, &input, &context, keys_layout);
        let raise_input = |seed: u8| {
            let mut coeffs = fixture_ciphertext(&module, &preset.input_layout(), seed);
            with_scratch::<BE, _>(bytes, |s| {
                if context.pipeline() == BootstrappingPipeline::S2CFirst {
                    module.ckks_double_into(&mut coeffs, &input, s)?;
                    module.ckks_dft_matrix_eval_assign(&mut coeffs, context.slots_to_coeffs(), &keys, s)
                } else {
                    module.ckks_copy(&mut coeffs, &input, s)
                }
            })
            .unwrap();
            coeffs
        };
        let mod_raise = preset.plan().mod_raise();
        let (mut a_context, mut a_plan, mut a_prepared) = (raise_input(31), raise_input(37), raise_input(41));
        let mut from_context = fixture_ciphertext(&module, &boot, 43);
        let mut from_plan = fixture_ciphertext(&module, &boot, 47);
        let mut from_prepared = fixture_ciphertext(&module, &boot, 53);
        let raise_tmp = module.ckks_mod_raise_tmp_bytes(&boot, &a_context, &mod_raise, keys_layout);
        assert_eq!(
            raise_tmp,
            module.ckks_mod_raise_tmp_bytes(&boot, &a_context, &context, keys_layout)
        );
        with_scratch::<BE, _>(raise_tmp, |s| {
            module.ckks_mod_raise(&mut from_context, &mut a_context, &context, &keys, s)?;
            module.ckks_mod_raise(&mut from_plan, &mut a_plan, &mod_raise, &keys, s)?;
            module.ckks_mod_raise(&mut from_prepared, &mut a_prepared, context.mod_raise(), &keys, s)
        })
        .unwrap();
        assert_eq!(
            snapshot::<BE, _, Standard>(&from_plan),
            snapshot::<BE, _, Standard>(&from_context)
        );
        assert_eq!(
            snapshot::<BE, _, Standard>(&from_prepared),
            snapshot::<BE, _, Standard>(&from_context)
        );

        let mut a = raise_input(59);
        let mut res = fixture_ciphertext(&module, &boot, 61);
        let mut rank2 = preset.input_layout();
        rank2.glwe_layout.rank = 2usize.into();
        assert_mod_raise_refuses(
            &mut res,
            &mut fixture_ciphertext(&module, &rank2, 29),
            "supports rank-1 ciphertexts only, got rank 2 -> 1",
            |res, a, s| module.ckks_mod_raise(res, a, &context, &keys, s),
        );
        let mut half = boot;
        half.glwe_layout.n = (preset.n() / 2).into();
        let degree = "takes ciphertexts of one degree, which the module serves";
        assert_mod_raise_refuses(&mut fixture_ciphertext(&module, &half, 11), &mut a, degree, |res, a, s| {
            module.ckks_mod_raise(res, a, &context, &keys, s)
        });
        let small = Module::<BE>::new(preset.n() as u64 / 2);
        assert_mod_raise_refuses(&mut res, &mut a, degree, |res, a, s| {
            small.ckks_mod_raise(res, a, &context, &keys, s)
        });
        let mut other_radix = boot;
        other_radix.glwe_layout.base2k = (preset.base2k() - 1).into();
        assert_mod_raise_refuses(
            &mut fixture_ciphertext(&module, &other_radix, 13),
            &mut a,
            "takes ciphertexts of one base2k",
            |res, a, s| module.ckks_mod_raise(res, a, &context, &keys, s),
        );
        let other_encapsulation = ModRaisePlan {
            sparse_secret_hamming_weight: match mod_raise.sparse_secret_hamming_weight() {
                Some(_) => None,
                None => Some(EPHEMERAL_SECRET_WEIGHT),
            },
            ..mod_raise.clone()
        };
        assert_mod_raise_refuses(
            &mut res,
            &mut a,
            "bootstrapping key encapsulation does not match the compiled recipe",
            |res, a, s| module.ckks_mod_raise(res, a, &other_encapsulation, &keys, s),
        );
        let mut narrow = boot;
        narrow.glwe_layout.k = (a.k().as_usize() - 1).into();
        assert_mod_raise_refuses(
            &mut fixture_ciphertext(&module, &narrow, 17),
            &mut a,
            "ModUp must widen, not shrink, the modulus",
            |res, a, s| module.ckks_mod_raise(res, a, &context, &keys, s),
        );
        // One bit of headroom gives a lift, which the refusal precedes.
        if let Some(lift) = mod_raise.lift() {
            let mut wider = preset.input_layout();
            wider.glwe_layout.k = (a.k().as_usize() + 1).into();
            let mut lifted = fixture_ciphertext(&module, &wider, 23);
            assert_eq!(
                lift.input_scale_up("ckks_mod_raise", lifted.k().as_usize(), lifted.log_delta())
                    .unwrap(),
                1
            );
            assert_mod_raise_refuses(
                &mut fixture_ciphertext(&module, &narrow, 17),
                &mut lifted,
                "ModUp must widen, not shrink, the modulus",
                |res, a, s| module.ckks_mod_raise(res, a, &context, &keys, s),
            );
            // An input one bit short of its scale plus the message ratio leaves no room for the first lift.
            let mut short = preset.input_layout();
            short.glwe_layout.k = (ckks_log_delta(&short.meta) + lift.log_msg_ratio - 1).into();
            let reason = "the input holds fewer bits than its scale plus the plan's message ratio";
            assert_mod_raise_refuses(&mut res, &mut fixture_ciphertext(&module, &short, 27), reason, |res, a, s| {
                module.ckks_mod_raise(res, a, &context, &keys, s)
            });
            let short = fixture_ciphertext(&module, &short, 27);
            let mut out = fixture_ciphertext(&module, &boot, 33);
            let before = snapshot::<BE, _, Standard>(&out);
            let err = with_scratch::<BE, _>(0, |s| module.ckks_bootstrap(&mut out, &short, &context, &keys, s)).unwrap_err();
            assert_ckks_error(
                "a bootstrap of an input short of its lift",
                &err,
                CKKSCompositionError::InvalidOperand {
                    op: "ckks_bootstrap",
                    reason: reason.into(),
                },
            );
            assert_eq!(snapshot::<BE, _, Standard>(&out), before, "a refused bootstrap wrote res");
        }

        // The tensor key resolves before the scratch check, on the standard ring too.
        let tensor_key = keys.tensor_key.take();
        let z = fixture_ciphertext(&module, &boot, 19);
        let z_run = context.run(z.meta());
        assert_eval_mod_refuses(&mut res, &z, "the keys hold no tensor key of the plan's ring", |res, a, s| {
            module.ckks_bootstrap_eval_mod(res, a, &z_run, &keys, s)
        });
        keys.tensor_key = tensor_key;
    }
}

/// End-to-end bootstrapping: encrypt at level 0, refresh, check the slots return.
pub fn test_bootstrapping_standard_e2e<BE, F, E>(params: CKKSTestParams)
where
    BE: TestContextBackend,
    Module<BE>: TestContextSharedModule<BE, ConjugateInvariant> + CKKSEncodingOps<BE, F, ConjugateInvariant>,
    Module<BE>: CKKSDFTMatrixPreparedFactory<BE>
        + GLWECIKeyEncryptSk<BE>
        + GLWECIKeyPreparedFactory<BE>
        + TestContextModule<BE>
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
            slot_layout: SlotLayout::Full,
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
    let mut bsk = ctx
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
        .prepare(&module, &mut scratch.borrow())
        .unwrap();

    // The tensor key covers EvalMod's input.
    {
        use poulpy_core::layouts::GGLWEInfos;
        assert!(after_c2s <= bsk.tensor_key.as_ref().unwrap().gglwe_layout().max_input_k().as_usize());
    }

    // Encrypt z at the input ("level 0") modulus.
    let (re, im) = test_vector_1::<F>(m);

    let mut ct0 = ckks_encrypt_with_prec(
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

    // A working width narrower than CoeffsToSlots and EvalMod consume fails at entry, and so does one that holds every
    // stage's width but leaves SlotsToCoeffs, which runs on `res`, short of its budget above the scale of EvalMod.
    {
        let no_budget_for_s2c = plan.post_mod_up_consumed_bits() + plan.eval_mod().f_mod_log_delta - 1 - log_delta;
        for log_budget in [base2k, no_budget_for_s2c] {
            let mut ct_narrow = fixture_ciphertext(&module, &ckks_spec(n, base2k, log_delta, log_budget), 31);
            let before = snapshot::<BE, _, Standard>(&ct_narrow);
            let err = with_scratch::<BE, _>(0, |s| module.ckks_bootstrap(&mut ct_narrow, &ct0, &ctx, &bsk, s)).unwrap_err();
            assert_invalid_operand(
                "a bootstrap narrower than its stages",
                &err,
                "ckks_bootstrap",
                "is narrower than the bits its stages consume",
            );
            assert_eq!(
                snapshot::<BE, _, Standard>(&ct_narrow),
                before,
                "a refused bootstrap wrote res"
            );
        }
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
        let before = snapshot::<BE, _, Standard>(&ct_wide);
        let err = with_scratch::<BE, _>(0, |s| module.ckks_bootstrap(&mut ct_wide, &ct0, &ctx, &bsk, s)).unwrap_err();
        match err.composition() {
            Some(CKKSCompositionError::MissingAutomorphismKey {
                op: "ckks_bootstrap",
                source,
                ..
            }) => assert!(source.to_string().contains("wider than the key's max_input_k"), "{err}"),
            _ => panic!("expected a key refused at entry, got: {err}"),
        }
        assert_eq!(snapshot::<BE, _, Standard>(&ct_wide), before, "a refused bootstrap wrote res");

        let tensor_key = bsk.tensor_key.take();
        let err = with_scratch::<BE, _>(0, |s| module.ckks_bootstrap(&mut ct_wide, &ct0, &ctx, &bsk, s)).unwrap_err();
        assert_invalid_operand("no tensor key", &err, "ckks_bootstrap", "no tensor key of the plan's ring");
        assert_eq!(snapshot::<BE, _, Standard>(&ct_wide), before, "a refused bootstrap wrote res");
        bsk.tensor_key = tensor_key;
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
    let mod_up_tmp = module.ckks_mod_raise_tmp_bytes(&ct, &ct0, &ctx, &keys_layout);
    with_scratch::<BE, _>(mod_up_tmp, |s| module.ckks_mod_raise(&mut ct, &mut ct0, &ctx, &bsk, s)).unwrap();

    let mut log_budget_check = k_boot - ct.log_delta();

    assert_eq!(ct.log_budget(), log_budget_check);
    assert_canonical_at_k::<BE, Standard>("ModUp", &ct);

    // 2) CoeffsToSlots (split): coefficients → (real, imag) slots.
    let mut ct_real = module.ckks_ciphertext_alloc(base2k.into(), k_boot.into());
    let mut ct_imag = module.ckks_ciphertext_alloc(base2k.into(), k_boot.into());
    let bytes = dft_stage_bytes(&module, &keys_layout, &ct_real, &ct, ctx.coeffs_to_slots());
    with_scratch::<BE, _>(bytes, |s| {
        module.ckks_coeffs_to_slots_split(&mut ct_real, &mut ct_imag, &ct, ctx.coeffs_to_slots(), &bsk, s)
    })
    .unwrap();

    log_budget_check -= plan.coeffs_to_slots().consumed_bits();

    assert_eq!(ct_real.log_budget(), log_budget_check);
    assert_eq!(ct_imag.log_budget(), log_budget_check);
    for ct in [&ct_real, &ct_imag] {
        assert_canonical_at_k::<BE, Standard>("CoeffsToSlots", ct);
    }

    // 3) EvalMod each half. EvalMod raises the ciphertext to its own plan scale
    //    (`f_mod_log_delta`) internally and restores the input scale on the result,
    //    so no manual set_scale is needed here. The results are allocated at
    //    exactly `k_boot`: an allocation's width is the width ct×ct squaring
    //    computes at.
    let mut res_real = module.ckks_ciphertext_alloc(base2k.into(), k_boot.into());
    let mut res_imag = module.ckks_ciphertext_alloc(base2k.into(), k_boot.into());
    with_scratch::<BE, _>(
        module.ckks_eval_mod_tmp_bytes(&res_real, &ct_real, ctx.eval_mod().unwrap(), bsk.tensor_key.as_ref().unwrap()),
        |s| {
            module.ckks_eval_mod(
                &mut res_real,
                &mut ct_real.clone(),
                ctx.eval_mod().unwrap(),
                bsk.tensor_key.as_ref().unwrap(),
                s,
            )
        },
    )
    .unwrap();
    with_scratch::<BE, _>(
        module.ckks_eval_mod_tmp_bytes(&res_imag, &ct_imag, ctx.eval_mod().unwrap(), bsk.tensor_key.as_ref().unwrap()),
        |s| {
            module.ckks_eval_mod(
                &mut res_imag,
                &mut ct_imag.clone(),
                ctx.eval_mod().unwrap(),
                bsk.tensor_key.as_ref().unwrap(),
                s,
            )
        },
    )
    .unwrap();

    log_budget_check -= plan.eval_mod().consumed_bits();

    assert_eq!(res_real.log_budget(), log_budget_check);
    assert_eq!(res_imag.log_budget(), log_budget_check);
    for ct in [&res_real, &res_imag] {
        assert_canonical_at_k::<BE, Standard>("EvalMod", ct);
    }

    // 4) SlotsToCoeffs (split), then restore the message ratio EvalMod divided out.
    let mut ct_out = module.ckks_ciphertext_alloc(base2k.into(), k_boot.into());
    let bytes = dft_stage_bytes(&module, &keys_layout, &ct_out, &res_real, ctx.slots_to_coeffs());
    with_scratch::<BE, _>(bytes, |s| {
        module.ckks_slots_to_coeffs_split(&mut ct_out, &res_real, &res_imag, ctx.slots_to_coeffs(), &bsk, s)
    })
    .unwrap();

    log_budget_check -= plan.slots_to_coeffs().consumed_bits();

    assert_eq!(log_budget_check, k_boot - plan.consumed_bits() - ct_out.log_delta());
    assert_eq!(ct_out.log_budget(), log_budget_check);

    ct_out.set_log_delta(log_delta);
    assert_same_bootstrap::<BE, Standard>(&ct_out, &ct_bs);
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
    BE: TestContextBackend,
    Module<BE>: TestContextSharedModule<BE, ConjugateInvariant> + CKKSEncodingOps<BE, F, ConjugateInvariant>,
    Module<BE>: CKKSDFTMatrixPreparedFactory<BE>
        + GLWECIKeyEncryptSk<BE>
        + GLWECIKeyPreparedFactory<BE>
        + TestContextModule<BE>
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
            slot_layout: SlotLayout::Full,
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
        .prepare(&module, &mut scratch.borrow())
        .unwrap();

    let (re, im) = test_vector_1::<F>(m);

    let mut ct0 = ckks_encrypt_with_prec(
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
    let mod_up_tmp = module.ckks_mod_raise_tmp_bytes(&ct, &ct0, &ctx, &keys_layout);
    with_scratch::<BE, _>(mod_up_tmp, |s| module.ckks_mod_raise(&mut ct, &mut ct0, &ctx, &bsk, s)).unwrap();

    // 2) CoeffsToSlots (split): LP (low precision, `1/K` scaling) for the round, and
    //    HP (full precision, natural scaling) for the high-precision `Δm + I·q`.
    let mut r0_lp = module.ckks_ciphertext_alloc(base2k.into(), k_boot.into());
    let mut i0_lp = module.ckks_ciphertext_alloc(base2k.into(), k_boot.into());
    let bytes = dft_stage_bytes(&module, &keys_layout, &r0_lp, &ct, ctx.coeffs_to_slots());
    with_scratch::<BE, _>(bytes, |s| {
        module.ckks_coeffs_to_slots_split(&mut r0_lp, &mut i0_lp, &ct, ctx.coeffs_to_slots(), &bsk, s)
    })
    .unwrap();
    let mut r0_hp = module.ckks_ciphertext_alloc(base2k.into(), k_boot.into());
    let mut i0_hp = module.ckks_ciphertext_alloc(base2k.into(), k_boot.into());
    let bytes = dft_stage_bytes(&module, &keys_layout, &r0_hp, &ct, ctx.coeffs_to_slots_bypass().unwrap());
    with_scratch::<BE, _>(bytes, |s| {
        module.ckks_coeffs_to_slots_split(&mut r0_hp, &mut i0_hp, &ct, ctx.coeffs_to_slots_bypass().unwrap(), &bsk, s)
    })
    .unwrap();

    // 3) EvalMod each LP half: the residue `Δm + e` at natural scale.
    let mut res_real = module.ckks_ciphertext_alloc(base2k.into(), k_boot.into());
    let mut res_imag = module.ckks_ciphertext_alloc(base2k.into(), k_boot.into());
    with_scratch::<BE, _>(
        module.ckks_eval_mod_tmp_bytes(&res_real, &r0_lp, ctx.eval_mod().unwrap(), bsk.tensor_key.as_ref().unwrap()),
        |s| {
            module.ckks_eval_mod(
                &mut res_real,
                &mut r0_lp.clone(),
                ctx.eval_mod().unwrap(),
                bsk.tensor_key.as_ref().unwrap(),
                s,
            )
        },
    )
    .unwrap();
    with_scratch::<BE, _>(
        module.ckks_eval_mod_tmp_bytes(&res_imag, &i0_lp, ctx.eval_mod().unwrap(), bsk.tensor_key.as_ref().unwrap()),
        |s| {
            module.ckks_eval_mod(
                &mut res_imag,
                &mut i0_lp.clone(),
                ctx.eval_mod().unwrap(),
                bsk.tensor_key.as_ref().unwrap(),
                s,
            )
        },
    )
    .unwrap();

    // 4) r1 = r0_hp − K·r0_lp + EvalMod(r0_lp) = IDFT(Δ·m).
    //    EvalMod emits the residue at natural scale while the LP C2S is at `1/K`, so
    //    `r0_lp` is scaled up by K first. Then
    //    `(Δm+I·q) − (Δm+I·q+e) + (Δm+e) = Δm`: the integer part and the LP error `e`
    //    both cancel, leaving the message at the HP CoeffsToSlots' precision.
    let log2_k = FMOD_INTERVAL.trailing_zeros() as usize;
    with_scratch::<BE, _>(
        <Module<BE> as CKKSPow2Ops<BE>>::ckks_mul_pow2_tmp_bytes(&module, r0_lp.max_size()),
        |s| module.ckks_mul_pow2_assign(log2_k, &mut r0_lp, s),
    )
    .unwrap();
    with_scratch::<BE, _>(
        <Module<BE> as CKKSPow2Ops<BE>>::ckks_mul_pow2_tmp_bytes(&module, i0_lp.max_size()),
        |s| module.ckks_mul_pow2_assign(log2_k, &mut i0_lp, s),
    )
    .unwrap();
    with_scratch::<BE, _>(
        <Module<BE> as CKKSSubOps<BE>>::ckks_sub_tmp_bytes(&module, r0_hp.max_size()),
        |s| module.ckks_sub_assign(&mut r0_hp, &r0_lp, s),
    )
    .unwrap();
    with_scratch::<BE, _>(
        <Module<BE> as CKKSSubOps<BE>>::ckks_sub_tmp_bytes(&module, i0_hp.max_size()),
        |s| module.ckks_sub_assign(&mut i0_hp, &i0_lp, s),
    )
    .unwrap();
    with_scratch::<BE, _>(
        <Module<BE> as CKKSAddOps<BE>>::ckks_add_tmp_bytes(&module, r0_hp.max_size()),
        |s| module.ckks_add_assign(&mut r0_hp, &res_real, s),
    )
    .unwrap();
    with_scratch::<BE, _>(
        <Module<BE> as CKKSAddOps<BE>>::ckks_add_tmp_bytes(&module, i0_hp.max_size()),
        |s| module.ckks_add_assign(&mut i0_hp, &res_imag, s),
    )
    .unwrap();

    // 5) SlotsToCoeffs (split): IDFT(Δ·m) slots → refreshed message coefficients.
    let mut ct_out = module.ckks_ciphertext_alloc(base2k.into(), k_boot.into());
    let bytes = dft_stage_bytes(&module, &keys_layout, &ct_out, &r0_hp, ctx.slots_to_coeffs());
    with_scratch::<BE, _>(bytes, |s| {
        module.ckks_slots_to_coeffs_split(&mut ct_out, &r0_hp, &i0_hp, ctx.slots_to_coeffs(), &bsk, s)
    })
    .unwrap();

    ct_out.set_log_delta(log_delta);
    assert_same_bootstrap::<BE, Standard>(&ct_out, &ct_bs);
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
    BE: TestContextBackend,
    Module<BE>: TestContextSharedModule<BE, ConjugateInvariant> + CKKSEncodingOps<BE, F, ConjugateInvariant>,
    Module<BE>: CKKSDFTMatrixPreparedFactory<BE>
        + GLWECIKeyEncryptSk<BE>
        + GLWECIKeyPreparedFactory<BE>
        + TestContextModule<BE>
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
    BE: TestContextBackend,
    Module<BE>: TestContextSharedModule<BE, ConjugateInvariant> + CKKSEncodingOps<BE, F, ConjugateInvariant>,
    Module<BE>: CKKSDFTMatrixPreparedFactory<BE>
        + GLWECIKeyEncryptSk<BE>
        + GLWECIKeyPreparedFactory<BE>
        + TestContextModule<BE>
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
            slot_layout: SlotLayout::Full,
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
        .prepare(&module, &mut scratch.borrow())
        .unwrap();

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
    let before = snapshot::<BE, _, Standard>(&ct_out);
    let err = with_scratch::<BE, _>(boot_tmp, |s| {
        module.ckks_bootstrap(&mut ct_out, &ct_insufficient, &ctx, &bsk, s)
    })
    .unwrap_err();
    assert!(matches!(
        err.composition(),
        Some(CKKSCompositionError::MultiplicationPrecisionUnderflow { .. })
    ));
    assert_eq!(snapshot::<BE, _, Standard>(&ct_out), before, "a refused bootstrap wrote res");

    let s_re = precision_stats(&bs_re, &re, log_delta);
    let s_im = precision_stats(&bs_im, &im, log_delta);
    (s_re.avg_log2_prec, s_im.avg_log2_prec)
}

fn decrypt<BE: Backend<ZnxWord = i64> + TestContextBackend, F, E, S, Rg: Ring>(
    module: &Module<BE>,
    encoder: &ReferenceEncoder<E>,
    ct: &CKKSCiphertextOwned<BE, Rg>,
    sk: &S,
    scratch: &mut ScratchArena<'_, BE>,
) -> (Vec<F>, Vec<F>)
where
    F: TestScalar,
    Module<BE>: CKKSDecryptOps<BE, Rg>,
    E: ReimNew<F>,
    S: GLWESecretPreparedToBackendRef<BE, Rg> + GLWEInfos,
{
    assert_canonical_at_k::<BE, Rg>("decrypt", ct);
    // Decrypt, decode, and confirm the slots are recovered. Cap the budget so
    // `log_delta + log_budget <= 127` fits the i128 decode codec (the unused
    // high-order budget is dropped losslessly).
    let prec = meta(ct.log_delta(), ct.log_budget().min(127usize.saturating_sub(ct.log_delta())));
    let mut pt_out = module.ckks_plaintext_alloc_from_slots(module.ckks_max_slots::<Rg>(), ct.base2k(), prec.k);
    pt_out.set_meta(prec.meta);
    module.ckks_decrypt(&mut pt_out, ct, sk, &mut scratch.borrow()).unwrap();
    let m = 1 << (ct.log_n() - ct.log_sparsity() - 1);

    let pt_host = pt_out.to_host_owned::<BE>();
    let (mut re_out, mut im_out) = (vec![F::zero(); m], vec![F::zero(); m]);
    encoder.decode_reim(&pt_host, &mut re_out, &mut im_out).unwrap();

    (re_out, im_out)
}

fn assert_same_bootstrap<BE: Backend, Rg: Ring>(got: &CKKSCiphertextOwned<BE, Rg>, want: &CKKSCiphertextOwned<BE, Rg>) {
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
