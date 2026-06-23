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

use std::{collections::HashMap, time::Instant};

use poulpy_core::{
    GLWEKeyswitch,
    layouts::{
        GGLWEInfos, GLWETensorKeyPrepared, GLWEToBackendMut, GLWEToBackendRef, LWEInfos,
        prepared::GLWETensorKeyPreparedToBackendRef,
    },
};
use poulpy_hal::{
    api::{NegacyclicFFT, NegacyclicFFTNew, ScratchOwnedAlloc, ScratchOwnedBorrow},
    layouts::{Backend, CyclotomicOrder, HostBytesBackend, HostDataMut, HostDataRef, Module, ScratchOwned},
};

use crate::{
    CKKSCtBounds, CKKSInfos, CKKSLayout, CKKSMeta, SetCKKSInfos,
    api::{CKKSAllOpsTmpBytes, CKKSBootstrappingOps, CKKSDecrypt, CKKSEvalModOps, DFTOps},
    encoding::reim::Encoder,
    layouts::{
        BootstrappingContext, BootstrappingPlan, CKKSCiphertext, CKKSModuleAlloc, CKKSPlaintext, CKKSPlaintextVecHostCodec,
        DFTOutputFormat, DFTPlan, DFTType,
        eval_mod::{EvalModPlan, EvalModType},
    },
    polynomial::SplitStrategy,
    test_suite::{
        CKKSTestParams,
        helpers::{
            TestContextBackend, TestContextHostModule, TestContextModule, TestScalar, ckks_encrypt_with_prec, ckks_spec, gen_atk,
            gen_encapsulation_keys, gen_sk_with_raw, gen_tsk, precision_stats, test_vector_1,
        },
    },
};

/// `log2` of the live complex slot count (ring degree `n = 2·2^LOG_SLOTS`), kept
/// small to bound the depth — and modulus width — of a self-contained test.
const LOG_SLOTS: usize = 13;

fn meta(log_delta: usize, log_budget: usize) -> CKKSLayout {
    // n/base2k are placeholders; consumers pass an explicit base2k/n and read k()/meta().
    ckks_spec(0, 0, log_delta, log_budget)
}

/// End-to-end bootstrapping: encrypt at level 0, refresh, check the slots return.
pub fn test_bootstrapping_e2e<BE, F, E>(params: CKKSTestParams, _module: &Module<BE>, _host_module: &Module<HostBytesBackend>)
where
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
    let f_mod_interval = 16;
    let log_message_ratio = 10usize;

    let plan = BootstrappingPlan {
        ephemeral_secret_weight: 32,
        coeffs_to_slots: DFTPlan {
            kind: DFTType::Encode,
            factorization_depth: vec![4, 4, 5],
            giant_steps: vec![16, 16, 16],
            format: DFTOutputFormat::SplitRealAndImag,
            scaling: Some(1. / f_mod_interval as f64),
            bit_reversed: false,
            meta: meta(58, 10),
        },
        eval_mod: EvalModPlan {
            eval_mod_type: EvalModType::CosHK,
            log_message_ratio,
            f_mod_degree: 30,
            f_mod_interval,
            f_mod_log_interval_reduction: 3,
            f_mod_inv_degree: None,
            scaling: None,
            split_strategy: SplitStrategy::MinDepth,
            coeffs_meta: meta(48, 4), //~log_message_ratio+log(f_mod_interval)+log_final_prec
            f_mod_log_delta: 60,      // ~ log(f_mod_interval) + log_message_ratio + log_delta_in
        },
        slots_to_coeffs: DFTPlan {
            kind: DFTType::Decode,
            factorization_depth: vec![5, 4, 4],
            giant_steps: vec![16, 16, 16],
            format: DFTOutputFormat::SplitRealAndImag,
            scaling: Some((log_message_ratio as f64).exp2()),
            bit_reversed: false,
            meta: meta(39, 10),
        },
    };

    let n = 1 << (LOG_SLOTS + 1);
    let m = n / 2;
    let base2k = params.base2k;
    let log_delta = 45;
    // Scale of the ciphertext entering the pipeline.
    let log_modulus_in = log_delta + plan.eval_mod.log_message_ratio;

    // Size the bootstrap modulus from the plan: the ciphertext enters at
    // `log_modulus_in`, the pipeline consumes `consumed_bits` of budget (EvalMod
    // charged at the scale it runs — `f_mod_log_delta`, not the message scale, the
    // set-scale round-trip being budget-neutral), plus output head-room.
    let k_boot = (log_modulus_in + plan.consumed_bits() + 4 * log_delta).next_multiple_of(base2k);

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
    let tsk = gen_tsk(&tp, &module, &sk_raw, &mut scratch.borrow());

    // Galois keys: both transforms' rotations + the split forward conjugation.
    let order = module.cyclotomic_order();

    let mut gal_els = Vec::new();
    gal_els.extend_from_slice(&ctx.coeffs_to_slots.galois_elements(order));
    gal_els.extend_from_slice(&ctx.slots_to_coeffs.galois_elements(order));
    gal_els.sort_unstable();
    gal_els.dedup();

    println!("gal_els: {}", gal_els.len());

    let mut atks = HashMap::new();
    for el in gal_els {
        atks.entry(el)
            .or_insert_with(|| gen_atk(&tp, &module, el, &sk_raw, &mut scratch.borrow()));
    }
    let conj_key = gen_atk(&tp, &module, -1, &sk_raw, &mut scratch.borrow());

    // Sparse-secret encapsulation keys (https://eprint.iacr.org/2022/024):
    // `denseToSparse` at the input modulus, `sparseToDense` at the bootstrap
    // modulus. `None` when the trick is disabled.
    let encaps = (plan.ephemeral_secret_weight > 0).then(|| {
        gen_encapsulation_keys(
            &tp,
            &module,
            &host_module,
            &sk_raw,
            plan.ephemeral_secret_weight,
            log_modulus_in,
            k_boot,
            &mut scratch.borrow(),
        )
    });
    println!("KeyGen: {:?}", now.elapsed());

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
        meta(log_delta, log_modulus_in - log_delta),
        &mut scratch.borrow(),
    );

    let now = Instant::now();
    // 1) (encapsulate) denseToSparse, ModUp, sparseToDense — so the integer
    //    wrap-around `I(X)·q` ModUp exposes is bounded by the *sparse* secret's
    //    Hamming weight. Then relabel at the input-modulus scale (free
    //    /message-ratio): `I(X)·q` becomes the integer part, the message the
    //    residue `Δ·c/q`.
    if let Some((dense_to_sparse, _)) = &encaps {
        module.glwe_keyswitch_assign(&mut ct0, dense_to_sparse, dense_to_sparse.max_size(), &mut scratch.borrow());
    }
    println!("denseToSparse: {:?}", now.elapsed());

    let now = Instant::now();
    let mut ct = module.ckks_ciphertext_alloc(base2k.into(), k_boot.into());
    module.ckks_mod_up_into(&mut ct, &ct0, &mut scratch.borrow()).unwrap();
    if let Some((_, sparse_to_dense)) = &encaps {
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
            &atks,
            &conj_key,
            &mut scratch.borrow(),
        )
        .unwrap();
    println!("ckks_coeffs_to_slots_split: {:?}", now.elapsed());

    log_budget_check -= plan.coeffs_to_slots.consumed_bits();

    assert_eq!(ct_real.log_budget(), log_budget_check);
    assert_eq!(ct_imag.log_budget(), log_budget_check);

    let now = Instant::now();
    // 3) EvalMod each half. EvalMod raises the ciphertext to its own plan scale
    //    (`f_mod_log_delta`) internally and restores the input scale on the result,
    //    so no manual set_scale is needed here. Compact first: the ct×ct squaring
    //    needs compact operands (storage == k).
    let ct_real = ct_real.compact(&module, &mut scratch.borrow()).unwrap();
    let ct_imag = ct_imag.compact(&module, &mut scratch.borrow()).unwrap();
    let mut res_real = module.ckks_ciphertext_alloc(base2k.into(), k_boot.into());
    let mut res_imag = module.ckks_ciphertext_alloc(base2k.into(), k_boot.into());
    module
        .ckks_eval_mod(&mut res_real, &ct_real, &ctx.eval_mod, &tsk, &mut scratch.borrow())
        .unwrap();
    println!("ckks_eval_mod: {:?}", now.elapsed());
    let now = Instant::now();
    module
        .ckks_eval_mod(&mut res_imag, &ct_imag, &ctx.eval_mod, &tsk, &mut scratch.borrow())
        .unwrap();
    println!("ckks_eval_mod: {:?}", now.elapsed());

    log_budget_check -= plan.eval_mod.consumed_bits();

    assert_eq!(res_real.log_budget(), log_budget_check);
    assert_eq!(res_imag.log_budget(), log_budget_check);

    let now = Instant::now();
    // 4) SlotsToCoeffs (split), then restore the message ratio EvalMod divided out.
    let mut ct_out = module.ckks_ciphertext_alloc(base2k.into(), k_boot.into());
    module
        .ckks_slots_to_coeffs_split(
            &mut ct_out,
            &res_real,
            &res_imag,
            &ctx.slots_to_coeffs,
            &atks,
            &mut scratch.borrow(),
        )
        .unwrap();

    println!("ckks_slots_to_coeffs_split: {:?}", now.elapsed());

    log_budget_check -= plan.slots_to_coeffs.consumed_bits();

    assert_eq!(log_budget_check, k_boot - plan.consumed_bits() - ct_out.log_delta());
    assert_eq!(ct_out.log_budget(), log_budget_check);

    // Decrypt, decode, and confirm the slots are recovered. Cap the budget so
    // `log_delta + log_budget <= 127` fits the i128 decode codec (the unused
    // high-order budget is dropped losslessly).
    let prec = meta(
        ct_out.log_delta(),
        ct_out.log_budget().min(127usize.saturating_sub(ct_out.log_delta())),
    );
    let mut pt_out = module.ckks_pt_vec_alloc(base2k.into(), prec.k());
    pt_out.set_meta(prec.meta());
    module.ckks_decrypt(&mut pt_out, &ct_out, &sk, &mut scratch.borrow()).unwrap();

    let pt_host = pt_out.to_host_owned::<BE>();
    let (mut re_out, mut im_out) = (vec![F::zero(); m], vec![F::zero(); m]);
    encoder.decode_reim(&pt_host, &mut re_out, &mut im_out).unwrap();

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
