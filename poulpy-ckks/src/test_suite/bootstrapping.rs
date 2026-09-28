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
//! Scale bridge (see [`BootstrappingContext`]): CoeffsToSlots is pre-scaled by
//! `1/K` (`K = f_mod_interval`) into EvalMod's `[-1, 1]` domain; after ModUp the
//! ciphertext is relabeled at the input-modulus scale (free division by the
//! message ratio), restored by a `2^R` scale-up after SlotsToCoeffs.
//!
//! EvalMod set_scale ([`BootstrappingPlan::eval_mod`]'s `f_mod_log_delta`): EvalMod is run at a
//! wider scale than the input — `set_scale(eval_mod) → EvalMod → set_scale(input)`
//! — so its `ct×ct` chain keeps more precision. The recovered average precision
//! measures ~28 bits across the suite configurations; the assertions enforce
//! the `MIN_AVG_LOG2_PREC` regression floor a few bits under that.

use crate::api::CKKSEncodingOps;
use crate::layouts::CKKSCiphertextOwned;
use crate::layouts::CKKSPlaintextOwned;

use poulpy_core::layouts::{
    GGLWEInfos, GLWEInfos, GLWESecretPreparedToBackendRef, GLWETensorKeyPrepared, GLWEToBackendMut, GLWEToBackendRef, LWEInfos,
    prepared::GLWETensorKeyPreparedToBackendRef,
};
use poulpy_hal::{
    api::{NegacyclicFFT, NegacyclicFFTNew, ScratchOwnedAlloc, ScratchOwnedBorrow},
    layouts::{Backend, HostBytesBackend, HostDataMut, HostDataRef, Module, ScratchArena, ScratchOwned, Standard, ZnxView},
    source::Source,
};

use crate::SlotsKind;
use crate::{
    CKKSCompositionError, CKKSCtBounds, CKKSInfos, CKKSMeta, CoeffsMeta, SetCKKSInfos,
    api::{
        CKKSAddOps, CKKSBootstrappingOps, CKKSDFTMatrixOps, CKKSDFTOps, CKKSDecryptOps, CKKSEvalModOps, CKKSPow2Ops, CKKSSubOps,
    },
    layouts::{
        BootstrappingContext, BootstrappingKeys, BootstrappingKeysLayout, BootstrappingPipeline, BootstrappingPlan,
        BootstrappingTechniques, CKKSModuleAlloc, DFTOutputFormat, DFTPlan, DFTType, EncapsulationKeysLayout, EvalRoundPlus,
        SparseSecretEncapsulation,
        eval_mod::{EvalModPlan, EvalModType},
    },
    polynomial::SplitStrategy,
    test_suite::reference_encoder::ReferenceEncoder,
    test_suite::{
        CKKSTestParams,
        helpers::{
            TestContextBackend, TestContextModule, TestScalar, assert_canonical_at_k, ckks_encrypt_with_prec, ckks_spec,
            gen_sk_with_raw, precision_stats, test_vector_1,
        },
        presets::bootstrap_setup_tmp_bytes,
    },
};

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
    let plan = BootstrappingPlan::new(
        pipeline,
        BootstrappingTechniques {
            sparse_secret_encapsulation: Some(SparseSecretEncapsulation {
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

/// End-to-end bootstrapping: encrypt at level 0, refresh, check the slots return.
pub fn test_bootstrapping_standard_e2e<BE, F, E>(params: CKKSTestParams)
where
    BE: TestContextBackend<Ring = Standard>,
    Module<BE>: TestContextModule<BE> + CKKSEncodingOps<BE, F> + CKKSBootstrappingOps<BE> + CKKSDFTMatrixOps<BE, F>,
    F: TestScalar,
    E: NegacyclicFFT<F> + NegacyclicFFTNew<F>,
    for<'a> <BE as Backend>::BufRef<'a>: HostDataRef,
    for<'a> <BE as Backend>::BufMut<'a>: HostDataMut,
    ScratchOwned<BE>: ScratchOwnedAlloc<BE>,
    CKKSCiphertextOwned<BE>: GLWEToBackendMut<BE> + GLWEToBackendRef<BE> + CKKSCtBounds + SetCKKSInfos,
    CKKSPlaintextOwned<BE>: GLWEToBackendRef<BE> + LWEInfos,
    GLWETensorKeyPrepared<BE::OwnedBuf, BE>: GLWETensorKeyPreparedToBackendRef<BE> + GGLWEInfos,
{
    let plan = bootstrap_plan(BootstrappingPipeline::C2SFirst, false, LOG_MSG_RATIO, 0);

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
        prec_meta: CKKSMeta {
            log_sparsity: 0,
            log_delta,
            slots: SlotsKind::Complex,
        },
        prec_log_budget: 8,
        hw: 192,
        dsize: 7,
        rank: 1,
    };

    // One scratch for the whole pipeline (plaintext precision sized for the
    // largest plaintext op, EvalMod).
    let keys_layout = BootstrappingKeysLayout {
        automorphism_key: tp.atk_layout().layout,
        tensor_key: tp.tsk_layout().layout,
        encapsulation: plan.sparse_secret_hamming_weight().map(|_| EncapsulationKeysLayout {
            dense_to_sparse: tp.ksk_layout(log_modulus_in).layout,
            sparse_to_dense: tp.ksk_layout(k_boot).layout,
        }),
    };
    let scratch_size = bootstrap_setup_tmp_bytes(
        &module,
        &ckks_spec(n, base2k, log_delta, k_boot - log_delta),
        &plan,
        &keys_layout,
    );
    let mut scratch = ScratchOwned::<BE>::alloc(scratch_size);

    let ctx = BootstrappingContext::<BE, F>::compile(&module, base2k.into(), &plan, &mut scratch.borrow()).unwrap();

    let (sk_raw, sk) = gen_sk_with_raw(&tp, &module, &host_module, [0u8; 32]);

    // The compiled pipeline adds its live intermediates to the setup scratch.
    {
        let boot_tmp = module.ckks_bootstrap_tmp_bytes(
            &ckks_spec(n, base2k, log_delta, k_boot - log_delta),
            &ckks_spec(n, base2k, log_delta, log_modulus_in - log_delta),
            &ctx,
            &keys_layout,
        );
        if boot_tmp > scratch_size {
            scratch = ScratchOwned::<BE>::alloc(boot_tmp);
        }
    }
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

    // Compare the public orchestrator with the explicit pipeline below.
    let ct_bs = {
        let mut ct_bs = module.ckks_ciphertext_alloc(base2k.into(), k_boot.into());
        module
            .ckks_bootstrap(&mut ct_bs, &ct0, &ctx, &bsk, &mut scratch.borrow())
            .unwrap();
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
        module
            .ckks_bootstrap(&mut ct_bs, &ct_real, &ctx, &bsk, &mut scratch.borrow())
            .unwrap();
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
    module
        .ckks_bootstrap_mod_up(&mut ct, &ct0, plan.eval_mod(), &bsk, &mut scratch.borrow())
        .unwrap();

    let mut log_budget_check = k_boot - ct.log_delta();

    assert_eq!(ct.log_budget(), log_budget_check);
    assert_canonical_at_k::<BE>("ModUp", &ct);

    // 2) CoeffsToSlots (split): coefficients → (real, imag) slots.
    let mut ct_real = module.ckks_ciphertext_alloc(base2k.into(), k_boot.into());
    let mut ct_imag = module.ckks_ciphertext_alloc(base2k.into(), k_boot.into());
    module
        .ckks_coeffs_to_slots_split(
            &mut ct_real,
            &mut ct_imag,
            &ct,
            ctx.coeffs_to_slots(),
            bsk.rotation_keys(),
            &mut scratch.borrow(),
        )
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
    module
        .ckks_eval_mod(
            &mut res_real,
            &ct_real,
            ctx.eval_mod(),
            bsk.tensor_key(),
            &mut scratch.borrow(),
        )
        .unwrap();
    module
        .ckks_eval_mod(
            &mut res_imag,
            &ct_imag,
            ctx.eval_mod(),
            bsk.tensor_key(),
            &mut scratch.borrow(),
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
    module
        .ckks_slots_to_coeffs_split(
            &mut ct_out,
            &res_real,
            &res_imag,
            ctx.slots_to_coeffs(),
            bsk.rotation_keys(),
            &mut scratch.borrow(),
        )
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
    Module<BE>: TestContextModule<BE> + CKKSEncodingOps<BE, F> + CKKSBootstrappingOps<BE> + CKKSDFTMatrixOps<BE, F>,
    F: TestScalar,
    E: NegacyclicFFT<F> + NegacyclicFFTNew<F>,
    for<'a> <BE as Backend>::BufRef<'a>: HostDataRef,
    for<'a> <BE as Backend>::BufMut<'a>: HostDataMut,
    ScratchOwned<BE>: ScratchOwnedAlloc<BE>,
    CKKSCiphertextOwned<BE>: GLWEToBackendMut<BE> + GLWEToBackendRef<BE> + CKKSCtBounds + SetCKKSInfos,
    CKKSPlaintextOwned<BE>: GLWEToBackendRef<BE> + LWEInfos,
    GLWETensorKeyPrepared<BE::OwnedBuf, BE>: GLWETensorKeyPreparedToBackendRef<BE> + GGLWEInfos,
{
    let plan = bootstrap_plan(BootstrappingPipeline::C2SFirst, true, LOG_MSG_RATIO, 0);

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
        prec_meta: CKKSMeta {
            log_sparsity: 0,
            log_delta,
            slots: SlotsKind::Complex,
        },
        prec_log_budget: 8,
        hw: 192,
        dsize: 7,
        rank: 1,
    };

    let keys_layout = BootstrappingKeysLayout {
        automorphism_key: tp.atk_layout().layout,
        tensor_key: tp.tsk_layout().layout,
        encapsulation: plan.sparse_secret_hamming_weight().map(|_| EncapsulationKeysLayout {
            dense_to_sparse: tp.ksk_layout(log_modulus_in).layout,
            sparse_to_dense: tp.ksk_layout(k_boot).layout,
        }),
    };
    let scratch_size = bootstrap_setup_tmp_bytes(
        &module,
        &ckks_spec(n, base2k, log_delta, k_boot - log_delta),
        &plan,
        &keys_layout,
    );
    let mut scratch = ScratchOwned::<BE>::alloc(scratch_size);

    let ctx = BootstrappingContext::<BE, F>::compile(&module, base2k.into(), &plan, &mut scratch.borrow()).unwrap();

    let (sk_raw, sk) = gen_sk_with_raw(&tp, &module, &host_module, [0u8; 32]);

    // The compiled pipeline adds its live intermediates to the setup scratch.
    {
        let boot_tmp = module.ckks_bootstrap_tmp_bytes(
            &ckks_spec(n, base2k, log_delta, k_boot - log_delta),
            &ckks_spec(n, base2k, log_delta, log_modulus_in - log_delta),
            &ctx,
            &keys_layout,
        );
        if boot_tmp > scratch_size {
            scratch = ScratchOwned::<BE>::alloc(boot_tmp);
        }
    }
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
        module
            .ckks_bootstrap(&mut ct_bs, &ct0, &ctx, &bsk, &mut scratch.borrow())
            .unwrap();
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
        module
            .ckks_bootstrap(&mut ct_bs, &ct_real, &ctx, &bsk, &mut scratch.borrow())
            .unwrap();
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
    module
        .ckks_bootstrap_mod_up(&mut ct, &ct0, plan.eval_mod(), &bsk, &mut scratch.borrow())
        .unwrap();

    // 2) CoeffsToSlots (split): LP (low precision, `1/K` scaling) for the round, and
    //    HP (full precision, natural scaling) for the high-precision `Δm + I·q`.
    let mut r0_lp = module.ckks_ciphertext_alloc(base2k.into(), k_boot.into());
    let mut i0_lp = module.ckks_ciphertext_alloc(base2k.into(), k_boot.into());
    module
        .ckks_coeffs_to_slots_split(
            &mut r0_lp,
            &mut i0_lp,
            &ct,
            ctx.coeffs_to_slots(),
            bsk.rotation_keys(),
            &mut scratch.borrow(),
        )
        .unwrap();
    let mut r0_hp = module.ckks_ciphertext_alloc(base2k.into(), k_boot.into());
    let mut i0_hp = module.ckks_ciphertext_alloc(base2k.into(), k_boot.into());
    module
        .ckks_coeffs_to_slots_split(
            &mut r0_hp,
            &mut i0_hp,
            &ct,
            ctx.coeffs_to_slots_bypass().unwrap(),
            bsk.rotation_keys(),
            &mut scratch.borrow(),
        )
        .unwrap();

    // 3) EvalMod each LP half: the residue `Δm + e` at natural scale.
    let mut res_real = module.ckks_ciphertext_alloc(base2k.into(), k_boot.into());
    let mut res_imag = module.ckks_ciphertext_alloc(base2k.into(), k_boot.into());
    module
        .ckks_eval_mod(&mut res_real, &r0_lp, ctx.eval_mod(), bsk.tensor_key(), &mut scratch.borrow())
        .unwrap();
    module
        .ckks_eval_mod(&mut res_imag, &i0_lp, ctx.eval_mod(), bsk.tensor_key(), &mut scratch.borrow())
        .unwrap();

    // 4) r1 = r0_hp − K·r0_lp + EvalMod(r0_lp) = IDFT(Δ·m).
    //    EvalMod emits the residue at natural scale while the LP C2S is at `1/K`, so
    //    `r0_lp` is scaled up by K first. Then
    //    `(Δm+I·q) − (Δm+I·q+e) + (Δm+e) = Δm`: the integer part and the LP error `e`
    //    both cancel, leaving the message at the HP CoeffsToSlots' precision.
    let log2_k = FMOD_INTERVAL.trailing_zeros() as usize;
    module
        .ckks_mul_pow2_assign(&mut r0_lp, log2_k, &mut scratch.borrow())
        .unwrap();
    module
        .ckks_mul_pow2_assign(&mut i0_lp, log2_k, &mut scratch.borrow())
        .unwrap();
    module.ckks_sub_assign(&mut r0_hp, &r0_lp, &mut scratch.borrow()).unwrap();
    module.ckks_sub_assign(&mut i0_hp, &i0_lp, &mut scratch.borrow()).unwrap();
    module.ckks_add_assign(&mut r0_hp, &res_real, &mut scratch.borrow()).unwrap();
    module.ckks_add_assign(&mut i0_hp, &res_imag, &mut scratch.borrow()).unwrap();

    // 5) SlotsToCoeffs (split): IDFT(Δ·m) slots → refreshed message coefficients.
    let mut ct_out = module.ckks_ciphertext_alloc(base2k.into(), k_boot.into());
    module
        .ckks_slots_to_coeffs_split(
            &mut ct_out,
            &r0_hp,
            &i0_hp,
            ctx.slots_to_coeffs(),
            bsk.rotation_keys(),
            &mut scratch.borrow(),
        )
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
    Module<BE>: TestContextModule<BE> + CKKSEncodingOps<BE, F> + CKKSBootstrappingOps<BE> + CKKSDFTMatrixOps<BE, F>,
    F: TestScalar,
    E: NegacyclicFFT<F> + NegacyclicFFTNew<F>,
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
    Module<BE>: TestContextModule<BE> + CKKSEncodingOps<BE, F> + CKKSBootstrappingOps<BE> + CKKSDFTMatrixOps<BE, F>,
    F: TestScalar,
    E: NegacyclicFFT<F> + NegacyclicFFTNew<F>,
    for<'a> <BE as Backend>::BufRef<'a>: HostDataRef,
    for<'a> <BE as Backend>::BufMut<'a>: HostDataMut,
    ScratchOwned<BE>: ScratchOwnedAlloc<BE>,
    CKKSCiphertextOwned<BE>: GLWEToBackendMut<BE> + GLWEToBackendRef<BE> + CKKSCtBounds + SetCKKSInfos,
    CKKSPlaintextOwned<BE>: GLWEToBackendRef<BE> + LWEInfos,
    GLWETensorKeyPrepared<BE::OwnedBuf, BE>: GLWETensorKeyPreparedToBackendRef<BE> + GGLWEInfos,
{
    let plan = bootstrap_plan(BootstrappingPipeline::S2CFirst, eval_round_plus, log_msg_ratio, guard_bits);

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
        prec_meta: CKKSMeta {
            log_sparsity: 0,
            log_delta,
            slots: SlotsKind::Complex,
        },
        prec_log_budget: 8,
        hw: 192,
        dsize: 7,
        rank: 1,
    };

    let keys_layout = BootstrappingKeysLayout {
        automorphism_key: tp.atk_layout().layout,
        tensor_key: tp.tsk_layout().layout,
        encapsulation: plan.sparse_secret_hamming_weight().map(|_| EncapsulationKeysLayout {
            dense_to_sparse: tp.ksk_layout(log_modulus_in).layout,
            sparse_to_dense: tp.ksk_layout(k_boot).layout,
        }),
    };
    let scratch_size = bootstrap_setup_tmp_bytes(
        &module,
        &ckks_spec(n, base2k, log_delta, k_boot - log_delta),
        &plan,
        &keys_layout,
    );
    let mut scratch = ScratchOwned::<BE>::alloc(scratch_size);

    let ctx = BootstrappingContext::<BE, F>::compile(&module, base2k.into(), &plan, &mut scratch.borrow()).unwrap();

    let (sk_raw, sk) = gen_sk_with_raw(&tp, &module, &host_module, [0u8; 32]);

    {
        let boot_tmp = module.ckks_bootstrap_tmp_bytes(
            &ckks_spec(n, base2k, log_delta, k_boot - log_delta),
            &ckks_spec(n, base2k, log_delta, k_in - log_delta),
            &ctx,
            &keys_layout,
        );
        if boot_tmp > scratch_size {
            scratch = ScratchOwned::<BE>::alloc(boot_tmp);
        }
    }
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
        module
            .ckks_bootstrap(&mut ct_bs, &ct0, &ctx, &bsk, &mut scratch.borrow())
            .unwrap();
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
            module
                .ckks_bootstrap(&mut ct_bs, &ct_real, &ctx, &bsk, &mut scratch.borrow())
                .unwrap();
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
    let mut ct_out = module.ckks_ciphertext_alloc(base2k.into(), k_boot.into());
    let err = module
        .ckks_bootstrap(&mut ct_out, &ct_insufficient, &ctx, &bsk, &mut scratch.borrow())
        .unwrap_err();
    assert!(matches!(
        err.composition(),
        Some(CKKSCompositionError::MultiplicationPrecisionUnderflow { .. })
    ));

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
    E: NegacyclicFFT<F> + NegacyclicFFTNew<F>,
    S: GLWESecretPreparedToBackendRef<BE> + GLWEInfos,
{
    assert_canonical_at_k::<BE>("decrypt", ct);
    // Decrypt, decode, and confirm the slots are recovered. Cap the budget so
    // `log_delta + log_budget <= 127` fits the i128 decode codec (the unused
    // high-order budget is dropped losslessly).
    let prec = meta(ct.log_delta(), ct.log_budget().min(127usize.saturating_sub(ct.log_delta())));
    let mut pt_out = module.ckks_pt_vec_alloc(ct.base2k(), prec.k);
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
