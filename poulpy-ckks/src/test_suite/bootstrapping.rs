//! End-to-end CKKS bootstrapping test (backend-generic).
//!
//! The crate ships **no orchestrator**, so this test *is* the reference
//! composition of the refresh pipeline, assembled from the public op surface:
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
//! EvalMod set_scale ([`BootstrappingPlan::eval_mod_meta`]): EvalMod is run at a
//! wider scale than the input — `set_scale(eval_mod) → EvalMod → set_scale(input)`
//! — so its `ct×ct` chain keeps more precision. At `base2k=19` this lifts the
//! recovered precision from ~9 bits to ~14 (a `+20`-bit scale-up); the assertion
//! is a conservative smoke-test floor.
//!
//! RUSTFLAGS="-C target-feature=+avx2,+fma" cargo test -p poulpy-cpu-avx --release --features enable-avx,enable-ckks ntt120_f64::bootstrapping -- --nocapture
//! cargo test -p poulpy-cpu-ref --features enable-ckks --release ntt120_f64::bootstrapping_e2e -- --nocapture

use std::time::Instant;

use poulpy_core::{
    GLWEKeyswitch,
    layouts::{
        GGLWEInfos, GLWEInfos, GLWESecretPreparedToBackendRef, GLWETensorKeyPrepared, GLWEToBackendMut, GLWEToBackendRef,
        LWEInfos, prepared::GLWETensorKeyPreparedToBackendRef,
    },
};
use poulpy_hal::{
    api::{NegacyclicFFT, NegacyclicFFTNew, ScratchOwnedAlloc, ScratchOwnedBorrow},
    layouts::{Backend, HostBytesBackend, HostDataMut, HostDataRef, Module, ScratchArena, ScratchOwned},
    source::Source,
};

use crate::{
    CKKSCtBounds, CKKSInfos, CKKSLayout, CKKSMeta, SetCKKSInfos,
    api::{CKKSAddOps, CKKSAllOpsTmpBytes, CKKSBootstrappingOps, CKKSDecrypt, CKKSEvalModOps, CKKSPow2Ops, CKKSSubOps, DFTOps},
    encoding::reim::Encoder,
    layouts::{
        BootstrappingContext, BootstrappingKeys, BootstrappingKeysLayout, BootstrappingPlan, CKKSCiphertext, CKKSModuleAlloc,
        CKKSPlaintext, CKKSPlaintextVecHostCodec, DFTOutputFormat, DFTPlan, DFTType, EncapsulationKeysLayout,
        eval_mod::{EvalModPlan, EvalModType},
    },
    polynomial::SplitStrategy,
    test_suite::{
        CKKSTestParams,
        helpers::{
            TestContextBackend, TestContextHostModule, TestContextModule, TestScalar, ckks_encrypt_with_prec, ckks_spec,
            gen_sk_with_raw, precision_stats, test_vector_1,
        },
    },
};

/// `log2` of the live complex slot count (ring degree `n = 2·2^LOG_SLOTS`), kept
/// small to bound the depth — and modulus width — of a self-contained test.
const LOG_SLOTS: usize = 10;
const FMOD_INTERVAL: usize = 16;
const LOG_MSG_RATIO: usize = 11;

fn meta(log_delta: usize, log_budget: usize) -> CKKSLayout {
    // n/base2k are placeholders; consumers pass an explicit base2k/n and read k()/meta().
    ckks_spec(0, 0, log_delta, log_budget)
}

/// End-to-end bootstrapping: encrypt at level 0, refresh, check the slots return.
pub fn test_bootstrapping_standard_e2e<BE, F, E>(
    params: CKKSTestParams,
    _module: &Module<BE>,
    _host_module: &Module<HostBytesBackend>,
) where
    BE: TestContextBackend + Backend<OwnedBuf = Vec<u8>>,
    Module<BE>: TestContextModule<BE> + CKKSBootstrappingOps<BE>,
    Module<HostBytesBackend>: TestContextHostModule,
    F: TestScalar,
    E: NegacyclicFFT<F> + NegacyclicFFTNew<F>,
    for<'a> <BE as Backend>::BufRef<'a>: HostDataRef,
    for<'a> <BE as Backend>::BufMut<'a>: HostDataMut,
    ScratchOwned<BE>: ScratchOwnedAlloc<BE>,
    CKKSPlaintext<Vec<u8>>: CKKSPlaintextVecHostCodec<f64> + CKKSPlaintextVecHostCodec<F>,
    CKKSCiphertext<BE::OwnedBuf>: GLWEToBackendMut<BE> + GLWEToBackendRef<BE> + CKKSCtBounds + SetCKKSInfos,
    CKKSPlaintext<BE::OwnedBuf>: GLWEToBackendRef<BE> + LWEInfos,
    GLWETensorKeyPrepared<BE::OwnedBuf, BE>: GLWETensorKeyPreparedToBackendRef<BE> + GGLWEInfos,
{
    let plan = BootstrappingPlan {
        ephemeral_secret_weight: 32,
        coeffs_to_slots: DFTPlan {
            kind: DFTType::Encode,
            factorization_depth: vec![2, 2, 3, 3],
            giant_steps: vec![4, 4, 4, 4],
            format: DFTOutputFormat::SplitRealAndImag,
            scaling: Some(1. / FMOD_INTERVAL as f64),
            bit_reversed: false,
            meta: meta(58, 2),
        },
        coeffs_to_slots_bypass: None,
        eval_mod: EvalModPlan {
            eval_mod_type: EvalModType::CosHK,
            log_msg_ratio: LOG_MSG_RATIO,
            f_mod_degree: 30,
            f_mod_interval: FMOD_INTERVAL,
            f_mod_log_interval_reduction: 3,
            f_mod_inv_degree: None,
            scaling: None,
            split_strategy: SplitStrategy::MinDepth,
            coeffs_meta: meta(48, 4), //~log_message_ratio+log(f_mod_interval)+log_final_prec
            f_mod_log_delta: 60,      // ~ log(f_mod_interval) + log_message_ratio + log_delta_in
        },
        slots_to_coeffs: DFTPlan {
            kind: DFTType::Decode,
            factorization_depth: vec![3, 3, 2, 2],
            giant_steps: vec![4, 4, 4, 4],
            format: DFTOutputFormat::SplitRealAndImag,
            scaling: Some((LOG_MSG_RATIO as f64).exp2()),
            bit_reversed: false,
            meta: meta(39, 2),
        },
    };

    let n = 1 << (LOG_SLOTS + 1);
    let m = n / 2;
    let base2k = params.base2k;
    let log_delta = 45;
    // Scale of the ciphertext entering the pipeline.
    let log_modulus_in = log_delta + plan.eval_mod.log_msg_ratio;

    // Size the bootstrap modulus from the plan: the ciphertext enters at
    // `log_modulus_in`, the pipeline consumes `consumed_bits` of budget (EvalMod
    // charged at the scale it runs — `f_mod_log_delta`, not the message scale, the
    // set-scale round-trip being budget-neutral), plus output head-room.
    let k_boot = (log_modulus_in + plan.consumed_bits() + 2 * log_delta).next_multiple_of(base2k);

    let module = Module::<BE>::new(n as u64);
    let host_module = Module::<HostBytesBackend>::new(n as u64);
    let encoder = Encoder::<E>::new::<F>(m).unwrap();

    let tp = CKKSTestParams {
        n,
        base2k,
        k: k_boot,
        prec_meta: CKKSMeta {
            log_sparsity: 0,
            log_delta,
        },
        prec_log_budget: 8,
        hw: 192,
        dsize: 7,
    };

    println!("n     : {}", n);
    println!("base2k: {}", base2k);
    println!("log_delta: {}", log_delta);
    println!("k_boot: {k_boot}");
    println!("dsize : {}", tp.dsize);
    println!("plan.consummed_bits(): {}", plan.consumed_bits());

    // One scratch for the whole pipeline (plaintext precision sized for the
    // largest plaintext op, EvalMod).
    let scratch_size;
    let mut scratch = {
        let mut c = module.ckks_ciphertext_alloc(base2k.into(), k_boot.into());
        c.set_meta(tp.prec().meta);
        scratch_size = module.ckks_all_ops_with_atk_tmp_bytes(&c, &tp.tsk_layout(), &tp.atk_layout(), &plan.eval_mod.coeffs_meta);
        ScratchOwned::<BE>::alloc(scratch_size)
    };

    let now = Instant::now();
    let ctx =
        BootstrappingContext::<BE, F>::compile(&module, &host_module, &encoder, base2k.into(), &plan, &mut scratch.borrow())
            .unwrap();
    {
        let mut eval_ct = module.ckks_ciphertext_alloc(base2k.into(), k_boot.into());
        eval_ct.set_meta(meta(log_delta, k_boot - log_delta).meta);
        let eval_tmp = module.ckks_eval_mod_tmp_bytes(&eval_ct, &eval_ct, &ctx.eval_mod, &tp.tsk_layout());
        if eval_tmp > scratch_size {
            scratch = ScratchOwned::<BE>::alloc(eval_tmp);
        }
    }
    println!("BootstrappingContext::compile: {:?}", now.elapsed());

    let now = Instant::now();
    let (sk_raw, sk) = gen_sk_with_raw(&tp, &module, &host_module, [0u8; 32]);

    // All evaluation keys via the bootstrapping-context helper: rotations (read off
    // the compiled DFT matrices), conjugation, EvalMod's tensor key, and — when the
    // sparse-secret encapsulation trick is enabled — the `denseToSparse` (input
    // modulus) / `sparseToDense` (bootstrap modulus) key-switching keys.
    let keys_layout = BootstrappingKeysLayout {
        automorphism_key: tp.atk_layout().layout,
        tensor_key: tp.tsk_layout().layout,
        encapsulation: (plan.ephemeral_secret_weight > 0).then(|| EncapsulationKeysLayout {
            ephemeral_secret_weight: plan.ephemeral_secret_weight,
            dense_to_sparse: tp.ksk_layout(log_modulus_in).layout,
            sparse_to_dense: tp.ksk_layout(k_boot).layout,
        }),
    };
    let (mut src_xs, mut src_xa, mut src_xe) = (Source::new([7u8; 32]), Source::new([1u8; 32]), Source::new([2u8; 32]));
    // `generate_keys` returns the keys *unprepared* (the serializable / GPU-resident
    // form); `prepare` preprocesses the whole set up front for this CPU path.
    let bsk = ctx
        .generate_keys(
            &module,
            &host_module,
            &sk_raw,
            &keys_layout,
            &mut src_xs,
            &mut src_xa,
            &mut src_xe,
            &mut scratch.borrow(),
        )
        .unwrap()
        .prepare(&module, &mut scratch.borrow());
    println!("KeyGen: {:?}", now.elapsed());

    // Encrypt z at the input ("level 0") modulus.
    let (re, im) = test_vector_1::<F>(m);

    // Per-step reference: the message's polynomial coefficients. The homomorphic
    // DFT shuttles these between the coefficient and slot domains, so `decode_reim`
    // of a CoeffsToSlots / EvalMod output recovers them in `bitrev` slot order
    // (real half in `ct_real`, imag half in `ct_imag`). Computed once in cleartext
    // by encoding `(re, im)` and reading back the coefficients.
    let (ref_real, ref_imag): (Vec<f64>, Vec<f64>) = {
        let mut pt = module.ckks_pt_vec_alloc(base2k.into(), meta(log_delta, 8).k());
        pt.set_meta(meta(log_delta, 8).meta());
        encoder.encode_reim(&mut pt, &re, &im).unwrap();
        let mut c = vec![F::zero(); n];
        pt.decode_host_floats(&mut c).unwrap();
        let c: Vec<f64> = c.iter().map(|x| x.to_f64().unwrap()).collect();
        let (mut rr, mut ri) = (vec![0f64; m], vec![0f64; m]);
        for j in 0..m {
            let b = bitrev(j, LOG_SLOTS);
            rr[j] = c[b];
            ri[j] = c[m + b];
        }
        (rr, ri)
    };

    let mut ct0 = ckks_encrypt_with_prec(
        &tp,
        &module,
        &host_module,
        &encoder,
        &sk,
        log_modulus_in,
        &re,
        &im,
        meta(log_delta, log_modulus_in - log_delta),
        &mut scratch.borrow(),
    );

    // Cross-check the one-shot orchestrator (the public API) against the explicit
    // pipeline below — run first, on the fresh input, since the manual path mutates
    // `ct0` in place for the encapsulation key-switch.
    {
        let mut ct_bs = module.ckks_ciphertext_alloc(base2k.into(), k_boot.into());
        module
            .ckks_bootstrap(&mut ct_bs, &ct0, &ctx, &bsk, &mut scratch.borrow())
            .unwrap();
        let (re_bs, im_bs) = decrypt(&module, &encoder, &ct_bs, &sk, &mut scratch.borrow());
        for (got, want, tag) in [(&re_bs, &re, "re"), (&im_bs, &im, "im")] {
            let s = precision_stats(got, want, log_delta);
            println!(
                "ckks_bootstrap (standard) ({tag}) avg={:.2} min={:.2} bits",
                s.avg_log2_prec, s.min_log2_prec
            );
            assert!(
                s.avg_log2_prec >= 5.0,
                "ckks_bootstrap standard ({tag}): {:.1} bits < 5.0",
                s.avg_log2_prec
            );
        }
    }

    let now = Instant::now();
    // 1) (encapsulate) denseToSparse, ModUp, sparseToDense — so the integer
    //    wrap-around `I(X)·q` ModUp exposes is bounded by the *sparse* secret's
    //    Hamming weight. Then relabel at the input-modulus scale (free
    //    /message-ratio): `I(X)·q` becomes the integer part, the message the
    //    residue `Δ·c/q`.
    if let Some((dense_to_sparse, _)) = bsk.encapsulation_keys() {
        module.glwe_keyswitch_assign(&mut ct0, dense_to_sparse, dense_to_sparse.max_size(), &mut scratch.borrow());
    }
    println!("denseToSparse: {:?}", now.elapsed());

    let now = Instant::now();
    let mut ct = module.ckks_ciphertext_alloc(base2k.into(), k_boot.into());
    module.ckks_mod_up_into(&mut ct, &ct0, &mut scratch.borrow()).unwrap();
    if let Some((_, sparse_to_dense)) = bsk.encapsulation_keys() {
        module.glwe_keyswitch_assign(&mut ct, sparse_to_dense, sparse_to_dense.max_size(), &mut scratch.borrow());
    }
    ct.set_meta(meta(log_modulus_in, k_boot - log_modulus_in).meta);
    println!("ckks_mod_up_into: {:?}", now.elapsed());

    let mut log_budget_check = k_boot - ct.log_delta();

    assert_eq!(ct.log_budget(), log_budget_check);

    let now = Instant::now();
    // 2) CoeffsToSlots (split): coefficients → (real, imag) slots.
    let mut ct_real = module.ckks_ciphertext_alloc(base2k.into(), k_boot.into());
    let mut ct_imag = module.ckks_ciphertext_alloc(base2k.into(), k_boot.into());
    module
        .ckks_coeffs_to_slots_split(
            &mut ct_real,
            &mut ct_imag,
            &ct,
            &ctx.coeffs_to_slots,
            bsk.rotation_keys(),
            bsk.conjugation_key(),
            &mut scratch.borrow(),
        )
        .unwrap();
    println!("ckks_coeffs_to_slots_split: {:?}", now.elapsed());
    println!("ct: {} {}", ct.k(), ct.size());
    println!("ct_real: {} {}", ct_real.k(), ct_real.size());
    println!("ct_imag: {} {}", ct_imag.k(), ct_imag.size());

    log_budget_check -= plan.coeffs_to_slots.consumed_bits();

    assert_eq!(ct_real.log_budget(), log_budget_check);
    assert_eq!(ct_imag.log_budget(), log_budget_check);

    // C2S accuracy: reference the C2S slot output against the *actual* coefficients
    // of the modup'd `ct` (integer parts `I_j` included), so this isolates C2S from
    // EvalMod. `decode_reim(ct_real)[j] = ct_coeffs[bitrev(j)]` (real half / imag half).
    {
        let ct_coeffs = decrypt_coeffs(&module, &ct, &sk, &mut scratch.borrow());
        let (mut cref_re, mut cref_im) = (vec![0f64; m], vec![0f64; m]);
        for j in 0..m {
            let b = bitrev(j, LOG_SLOTS);
            cref_re[j] = ct_coeffs[b];
            cref_im[j] = ct_coeffs[m + b];
        }
        let (re_c, _) = decrypt(&module, &encoder, &ct_real, &sk, &mut scratch.borrow());
        let (im_c, _) = decrypt(&module, &encoder, &ct_imag, &sk, &mut scratch.borrow());
        let re_c: Vec<f64> = re_c.iter().map(|x| x.to_f64().unwrap()).collect();
        let im_c: Vec<f64> = im_c.iter().map(|x| x.to_f64().unwrap()).collect();
        println!(
            "C2S-PREC   (re) snr={:.2} (im) snr={:.2} bits",
            snr_bits(&re_c, &cref_re),
            snr_bits(&im_c, &cref_im)
        );
    }

    let now = Instant::now();
    // 3) EvalMod each half. EvalMod raises the ciphertext to its own plan scale
    //    (`f_mod_log_delta`) internally and restores the input scale on the result,
    //    so no manual set_scale is needed here. Compact first: the ct×ct squaring
    //    needs compact operands (storage == k).
    let mut res_real = module.ckks_ciphertext_alloc(base2k.into(), k_boot.into());
    let mut res_imag = module.ckks_ciphertext_alloc(base2k.into(), k_boot.into());
    module
        .ckks_eval_mod(
            &mut res_real,
            &ct_real,
            &ctx.eval_mod,
            bsk.tensor_key(),
            &mut scratch.borrow(),
        )
        .unwrap();
    println!("ckks_eval_mod: {:?}", now.elapsed());
    let now = Instant::now();
    module
        .ckks_eval_mod(
            &mut res_imag,
            &ct_imag,
            &ctx.eval_mod,
            bsk.tensor_key(),
            &mut scratch.borrow(),
        )
        .unwrap();
    println!("ckks_eval_mod: {:?}", now.elapsed());

    log_budget_check -= plan.eval_mod.consumed_bits();

    assert_eq!(res_real.log_budget(), log_budget_check);
    assert_eq!(res_imag.log_budget(), log_budget_check);

    // After EvalMod the integer parts are removed, so the slots hold the clean
    // message coefficients: this SNR is a genuine precision of the C2S→EvalMod
    // chain (vs. the cleartext coefficient reference, scale-aligned).
    {
        let (re_e, _) = decrypt(&module, &encoder, &res_real, &sk, &mut scratch.borrow());
        let (im_e, _) = decrypt(&module, &encoder, &res_imag, &sk, &mut scratch.borrow());
        let re_e: Vec<f64> = re_e.iter().map(|x| x.to_f64().unwrap()).collect();
        let im_e: Vec<f64> = im_e.iter().map(|x| x.to_f64().unwrap()).collect();
        println!(
            "EVALMOD-PREC (re) snr={:.2} (im) snr={:.2} bits",
            snr_bits(&re_e, &ref_real),
            snr_bits(&im_e, &ref_imag)
        );
    }

    let now = Instant::now();
    // 4) SlotsToCoeffs (split), then restore the message ratio EvalMod divided out.
    let mut ct_out = module.ckks_ciphertext_alloc(base2k.into(), k_boot.into());
    module
        .ckks_slots_to_coeffs_split(
            &mut ct_out,
            &res_real,
            &res_imag,
            &ctx.slots_to_coeffs,
            bsk.rotation_keys(),
            &mut scratch.borrow(),
        )
        .unwrap();

    println!("ckks_slots_to_coeffs_split: {:?}", now.elapsed());

    log_budget_check -= plan.slots_to_coeffs.consumed_bits();

    assert_eq!(log_budget_check, k_boot - plan.consumed_bits() - ct_out.log_delta());
    assert_eq!(ct_out.log_budget(), log_budget_check);

    let (re_out, im_out) = decrypt(&module, &encoder, &ct_out, &sk, &mut scratch.borrow());

    for (got, want, tag) in [(&re_out, &re, "re"), (&im_out, &im, "im")] {
        let s = precision_stats(got, want, log_delta);
        println!(
            "BOOTSTRAP-PREC ({tag}) avg={:.2} min={:.2} bits",
            s.avg_log2_prec, s.min_log2_prec,
        );
        assert!(
            s.avg_log2_prec >= 5.0,
            "bootstrap_e2e ({tag}): {:.1} bits < 5.0 (worst got={} want={})",
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
pub fn test_bootstrapping_evalround_e2e<BE, F, E>(
    params: CKKSTestParams,
    _module: &Module<BE>,
    _host_module: &Module<HostBytesBackend>,
) where
    BE: TestContextBackend + Backend<OwnedBuf = Vec<u8>>,
    Module<BE>: TestContextModule<BE> + CKKSBootstrappingOps<BE>,
    Module<HostBytesBackend>: TestContextHostModule,
    F: TestScalar,
    E: NegacyclicFFT<F> + NegacyclicFFTNew<F>,
    for<'a> <BE as Backend>::BufRef<'a>: HostDataRef,
    for<'a> <BE as Backend>::BufMut<'a>: HostDataMut,
    ScratchOwned<BE>: ScratchOwnedAlloc<BE>,
    CKKSPlaintext<Vec<u8>>: CKKSPlaintextVecHostCodec<f64> + CKKSPlaintextVecHostCodec<F>,
    CKKSCiphertext<BE::OwnedBuf>: GLWEToBackendMut<BE> + GLWEToBackendRef<BE> + CKKSCtBounds + SetCKKSInfos,
    CKKSPlaintext<BE::OwnedBuf>: GLWEToBackendRef<BE> + LWEInfos,
    GLWETensorKeyPrepared<BE::OwnedBuf, BE>: GLWETensorKeyPreparedToBackendRef<BE> + GGLWEInfos,
{
    // `coeffs_to_slots` here is the LP transform feeding EvalMod: `log_delta = 29`
    // (half precision) — its error `e` cancels in the round, so it does not reach
    // the message and the bootstrap modulus shrinks by `num_factors × (58 − 29)`
    // bits. The HP CoeffsToSlots is compiled separately below.
    let plan = BootstrappingPlan {
        ephemeral_secret_weight: 32,
        coeffs_to_slots: DFTPlan {
            kind: DFTType::Encode,
            factorization_depth: vec![2, 2, 3, 3],
            giant_steps: vec![4, 4, 4, 4],
            format: DFTOutputFormat::SplitRealAndImag,
            scaling: Some(1. / FMOD_INTERVAL as f64),
            bit_reversed: false,
            meta: meta(29, 2),
        },
        coeffs_to_slots_bypass: Some(DFTPlan {
            kind: DFTType::Encode,
            factorization_depth: vec![1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
            giant_steps: vec![1; 10],
            format: DFTOutputFormat::SplitRealAndImag,
            scaling: Some(1.0),
            bit_reversed: false,
            meta: meta(58, 2),
        }),
        eval_mod: EvalModPlan {
            eval_mod_type: EvalModType::CosHK,
            log_msg_ratio: LOG_MSG_RATIO,
            f_mod_degree: 30,
            f_mod_interval: FMOD_INTERVAL,
            f_mod_log_interval_reduction: 3,
            f_mod_inv_degree: None,
            scaling: None,
            split_strategy: SplitStrategy::MinDepth,
            coeffs_meta: meta(48, 4),
            f_mod_log_delta: 60,
        },
        slots_to_coeffs: DFTPlan {
            kind: DFTType::Decode,
            factorization_depth: vec![3, 3, 2, 2],
            giant_steps: vec![4, 4, 4, 4],
            format: DFTOutputFormat::SplitRealAndImag,
            // `r1 = IDFT(Δ·m)` at natural scale (the residue scale EvalMod emits);
            // the standard `2^log_message_ratio` S2C scaling maps it to `m`.
            scaling: Some((LOG_MSG_RATIO as f64).exp2()),
            bit_reversed: false,
            meta: meta(39, 2),
        },
    };

    let n = 1 << (LOG_SLOTS + 1);
    let m = n / 2;
    let base2k = params.base2k;
    let log_delta = 45;
    let log_modulus_in = log_delta + plan.eval_mod.log_msg_ratio;

    let k_boot = (log_modulus_in + plan.consumed_bits() + 2 * log_delta).next_multiple_of(base2k);

    let module = Module::<BE>::new(n as u64);
    let host_module = Module::<HostBytesBackend>::new(n as u64);
    let encoder = Encoder::<E>::new::<F>(m).unwrap();

    let tp = CKKSTestParams {
        n,
        base2k,
        k: k_boot,
        prec_meta: CKKSMeta {
            log_sparsity: 0,
            log_delta,
        },
        prec_log_budget: 8,
        hw: 192,
        dsize: 7,
    };

    println!("[evalround] n={n} base2k={base2k} log_delta={log_delta} k_boot={k_boot}");
    println!("[evalround] plan.consumed_bits()={}", plan.consumed_bits());

    let scratch_size;
    let mut scratch = {
        let mut c = module.ckks_ciphertext_alloc(base2k.into(), k_boot.into());
        c.set_meta(tp.prec().meta);
        scratch_size = module.ckks_all_ops_with_atk_tmp_bytes(&c, &tp.tsk_layout(), &tp.atk_layout(), &plan.eval_mod.coeffs_meta);
        ScratchOwned::<BE>::alloc(scratch_size)
    };

    let ctx =
        BootstrappingContext::<BE, F>::compile(&module, &host_module, &encoder, base2k.into(), &plan, &mut scratch.borrow())
            .unwrap();
    {
        let mut eval_ct = module.ckks_ciphertext_alloc(base2k.into(), k_boot.into());
        eval_ct.set_meta(meta(log_delta, k_boot - log_delta).meta);
        let eval_tmp = module.ckks_eval_mod_tmp_bytes(&eval_ct, &eval_ct, &ctx.eval_mod, &tp.tsk_layout());
        if eval_tmp > scratch_size {
            scratch = ScratchOwned::<BE>::alloc(eval_tmp);
        }
    }

    let (sk_raw, sk) = gen_sk_with_raw(&tp, &module, &host_module, [0u8; 32]);

    // All evaluation keys via the bootstrapping-context helper. The generator reads
    // the rotation Galois elements off the compiled DFT matrices — including the
    // high-precision `coeffs_to_slots_bypass` — so the LP+HP CoeffsToSlots, the
    // conjugation, EvalMod's tensor key, and the encapsulation keys are all covered.
    let keys_layout = BootstrappingKeysLayout {
        automorphism_key: tp.atk_layout().layout,
        tensor_key: tp.tsk_layout().layout,
        encapsulation: (plan.ephemeral_secret_weight > 0).then(|| EncapsulationKeysLayout {
            ephemeral_secret_weight: plan.ephemeral_secret_weight,
            dense_to_sparse: tp.ksk_layout(log_modulus_in).layout,
            sparse_to_dense: tp.ksk_layout(k_boot).layout,
        }),
    };
    let (mut src_xs, mut src_xa, mut src_xe) = (Source::new([7u8; 32]), Source::new([1u8; 32]), Source::new([2u8; 32]));
    // Generated unprepared (serializable / GPU-resident), then prepared up front.
    let bsk = ctx
        .generate_keys(
            &module,
            &host_module,
            &sk_raw,
            &keys_layout,
            &mut src_xs,
            &mut src_xa,
            &mut src_xe,
            &mut scratch.borrow(),
        )
        .unwrap()
        .prepare(&module, &mut scratch.borrow());

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
        meta(log_delta, log_modulus_in - log_delta),
        &mut scratch.borrow(),
    );

    // Cross-check the one-shot EvalRound+ orchestrator (the public API) against the
    // explicit pipeline below — run first, on the fresh input, since the manual path
    // mutates `ct0` in place for the encapsulation key-switch.
    {
        let mut ct_bs = module.ckks_ciphertext_alloc(base2k.into(), k_boot.into());
        module
            .ckks_bootstrap(&mut ct_bs, &ct0, &ctx, &bsk, &mut scratch.borrow())
            .unwrap();
        let (re_bs, im_bs) = decrypt(&module, &encoder, &ct_bs, &sk, &mut scratch.borrow());
        for (got, want, tag) in [(&re_bs, &re, "re"), (&im_bs, &im, "im")] {
            let s = precision_stats(got, want, log_delta);
            println!(
                "ckks_bootstrap (evalround) ({tag}) avg={:.2} min={:.2} bits",
                s.avg_log2_prec, s.min_log2_prec
            );
            assert!(
                s.avg_log2_prec >= 5.0,
                "ckks_bootstrap evalround ({tag}): {:.1} bits < 5.0",
                s.avg_log2_prec
            );
        }
    }

    // 1) (encapsulate) denseToSparse, ModUp, sparseToDense.
    if let Some((dense_to_sparse, _)) = bsk.encapsulation_keys() {
        module.glwe_keyswitch_assign(&mut ct0, dense_to_sparse, dense_to_sparse.max_size(), &mut scratch.borrow());
    }
    let mut ct = module.ckks_ciphertext_alloc(base2k.into(), k_boot.into());
    module.ckks_mod_up_into(&mut ct, &ct0, &mut scratch.borrow()).unwrap();
    if let Some((_, sparse_to_dense)) = bsk.encapsulation_keys() {
        module.glwe_keyswitch_assign(&mut ct, sparse_to_dense, sparse_to_dense.max_size(), &mut scratch.borrow());
    }
    ct.set_meta(meta(log_modulus_in, k_boot - log_modulus_in).meta);

    // 2) CoeffsToSlots (split): LP (low precision, `1/K` scaling) for the round, and
    //    HP (full precision, natural scaling) for the high-precision `Δm + I·q`.
    let now = Instant::now();
    let mut r0_lp = module.ckks_ciphertext_alloc(base2k.into(), k_boot.into());
    let mut i0_lp = module.ckks_ciphertext_alloc(base2k.into(), k_boot.into());
    module
        .ckks_coeffs_to_slots_split(
            &mut r0_lp,
            &mut i0_lp,
            &ct,
            &ctx.coeffs_to_slots,
            bsk.rotation_keys(),
            bsk.conjugation_key(),
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
            &ctx.coeffs_to_slots_bypass.unwrap(),
            bsk.rotation_keys(),
            bsk.conjugation_key(),
            &mut scratch.borrow(),
        )
        .unwrap();
    println!("[evalround] coeffs_to_slots (LP+HP): {:?}", now.elapsed());

    // 3) EvalMod each LP half: the residue `Δm + e` at natural scale.
    let mut res_real = module.ckks_ciphertext_alloc(base2k.into(), k_boot.into());
    let mut res_imag = module.ckks_ciphertext_alloc(base2k.into(), k_boot.into());
    let now = Instant::now();
    module
        .ckks_eval_mod(&mut res_real, &r0_lp, &ctx.eval_mod, bsk.tensor_key(), &mut scratch.borrow())
        .unwrap();
    module
        .ckks_eval_mod(&mut res_imag, &i0_lp, &ctx.eval_mod, bsk.tensor_key(), &mut scratch.borrow())
        .unwrap();
    println!("[evalround] eval_mod x2: {:?}", now.elapsed());

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
    let now = Instant::now();
    let mut ct_out = module.ckks_ciphertext_alloc(base2k.into(), k_boot.into());
    module
        .ckks_slots_to_coeffs_split(
            &mut ct_out,
            &r0_hp,
            &i0_hp,
            &ctx.slots_to_coeffs,
            bsk.rotation_keys(),
            &mut scratch.borrow(),
        )
        .unwrap();
    println!("[evalround] slots_to_coeffs: {:?}", now.elapsed());

    let (re_out, im_out) = decrypt(&module, &encoder, &ct_out, &sk, &mut scratch.borrow());

    for (got, want, tag) in [(&re_out, &re, "re"), (&im_out, &im, "im")] {
        let s = precision_stats(got, want, log_delta);
        println!(
            "[evalround] BOOTSTRAP-PREC ({tag}) avg={:.2} min={:.2} bits",
            s.avg_log2_prec, s.min_log2_prec,
        );
        assert!(
            s.avg_log2_prec >= 5.0,
            "bootstrapping_evalround_e2e ({tag}): {:.1} bits < 5.0 (worst got={} want={})",
            s.avg_log2_prec,
            s.worst_got,
            s.worst_want,
        );
    }
}

fn decrypt<BE: Backend, C, F, E, S>(
    module: &Module<BE>,
    encoder: &Encoder<E>,
    ct: &C,
    sk: &S,
    scratch: &mut ScratchArena<'_, BE>,
) -> (Vec<F>, Vec<F>)
where
    C: GLWEToBackendRef<BE> + CKKSInfos + CKKSCtBounds,
    F: TestScalar,
    Module<BE>: CKKSDecrypt<BE>,
    E: NegacyclicFFT<F> + NegacyclicFFTNew<F>,
    S: GLWESecretPreparedToBackendRef<BE> + GLWEInfos,
{
    // Decrypt, decode, and confirm the slots are recovered. Cap the budget so
    // `log_delta + log_budget <= 127` fits the i128 decode codec (the unused
    // high-order budget is dropped losslessly).
    let prec = meta(ct.log_delta(), ct.log_budget().min(127usize.saturating_sub(ct.log_delta())));
    let mut pt_out = module.ckks_pt_vec_alloc(ct.base2k(), prec.k());
    pt_out.set_meta(prec.meta());
    module.ckks_decrypt(&mut pt_out, ct, sk, &mut scratch.borrow()).unwrap();
    let m = 1 << (ct.log_n() - ct.log_sparsity() - 1);

    let pt_host = pt_out.to_host_owned::<BE>();
    let (mut re_out, mut im_out) = (vec![F::zero(); m], vec![F::zero(); m]);
    encoder.decode_reim(&pt_host, &mut re_out, &mut im_out).unwrap();

    (re_out, im_out)
}

/// Decrypts `ct` and returns its raw polynomial coefficients (length `n`).
fn decrypt_coeffs<BE, C, S>(module: &Module<BE>, ct: &C, sk: &S, scratch: &mut ScratchArena<'_, BE>) -> Vec<f64>
where
    BE: Backend<OwnedBuf = Vec<u8>>,
    C: GLWEToBackendRef<BE> + CKKSInfos + CKKSCtBounds,
    Module<BE>: CKKSDecrypt<BE>,
    S: GLWESecretPreparedToBackendRef<BE> + GLWEInfos,
    CKKSPlaintext<Vec<u8>>: CKKSPlaintextVecHostCodec<f64>,
{
    let prec = meta(ct.log_delta(), ct.log_budget().min(127usize.saturating_sub(ct.log_delta())));
    let mut pt = module.ckks_pt_vec_alloc(ct.base2k(), prec.k());
    pt.set_meta(prec.meta());
    module.ckks_decrypt(&mut pt, ct, sk, &mut scratch.borrow()).unwrap();
    let mut c = vec![0f64; ct.n().as_usize()];
    pt.decode_host_floats(&mut c).unwrap();
    c
}

/// Bit-reversal of `j` over `bits` bits (poulpy's slot-map / coefficient order).
fn bitrev(j: usize, bits: usize) -> usize {
    ((j as u32).reverse_bits() >> (u32::BITS - bits as u32)) as usize
}

/// Scale-invariant signal-to-noise ratio in bits: best-fit a global scale `s`
/// between `got` and `want`, then report `-0.5·log2(||got - s·want||² / ||s·want||²)`.
/// Robust to the per-step scale bookkeeping (`1/K`, message ratio, eval scale),
/// so it measures only how well the recovered *shape* matches the reference.
fn snr_bits(got: &[f64], want: &[f64]) -> f64 {
    let dot_gw: f64 = got.iter().zip(want).map(|(g, w)| g * w).sum();
    let dot_ww: f64 = want.iter().map(|w| w * w).sum();
    let s = if dot_ww > 0.0 { dot_gw / dot_ww } else { 0.0 };
    let err2: f64 = got.iter().zip(want).map(|(g, w)| (g - s * w).powi(2)).sum();
    let sig2: f64 = want.iter().map(|w| (s * w).powi(2)).sum();
    if err2 <= 0.0 || sig2 <= 0.0 {
        return f64::INFINITY;
    }
    -0.5 * (err2 / sig2).log2()
}
