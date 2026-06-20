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
//! message ratio), restored by a `2^R` scale-up after SlotsToCoeffs. Recovered
//! precision scales with the parameters (~6 bits at `base2k=19`, ~16 at `52`);
//! the assertion is a conservative smoke-test floor.

use std::collections::HashMap;

use poulpy_core::layouts::{
    GGLWEInfos, GLWETensorKeyPrepared, GLWEToBackendMut, GLWEToBackendRef, LWEInfos, prepared::GLWETensorKeyPreparedToBackendRef,
};
use poulpy_hal::{
    api::{NegacyclicFFT, NegacyclicFFTNew, ScratchOwnedAlloc, ScratchOwnedBorrow},
    layouts::{Backend, CyclotomicOrder, HostBytesBackend, HostDataMut, HostDataRef, Module, ScratchOwned},
};

use crate::{
    CKKSCtBounds, CKKSInfos, CKKSMeta, SetCKKSInfos,
    api::{CKKSAllOpsTmpBytes, CKKSBootstrappingOps, CKKSDecrypt, CKKSEvalModOps, CKKSPow2Ops, DFTOps},
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
            TestContextBackend, TestContextHostModule, TestContextModule, TestScalar, ckks_encrypt_with_prec, gen_atk,
            gen_sk_with_raw, gen_tsk, precision_stats, test_vector_1,
        },
    },
};

/// `log2` of the live complex slot count (ring degree `n = 2·2^LOG_SLOTS`), kept
/// small to bound the depth — and modulus width — of a self-contained test.
const LOG_SLOTS: usize = 9;

fn meta(log_delta: usize, log_budget: usize) -> CKKSMeta {
    CKKSMeta {
        log_sparsity: 0,
        log_delta,
        log_budget,
    }
}

/// End-to-end bootstrapping: encrypt at level 0, refresh, check the slots return.
pub fn test_bootstrapping_e2e<BE, F, E>(params: CKKSTestParams, _module: &Module<BE>, _host_module: &Module<HostBytesBackend>)
where
    BE: TestContextBackend,
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

    // PARAM KNOBS: per-stage circuits + their coefficient metas (the scale each
    // stage is encoded at and consumes). `factorization_depth` must sum to
    // `LOG_SLOTS`.
    let plan = BootstrappingPlan {
        coeffs_to_slots: DFTPlan {
            kind: DFTType::Encode,
            factorization_depth: vec![3, 3, 3],
            giant_steps: vec![16, 16, 16],
            format: DFTOutputFormat::SplitRealAndImag,
            scaling: None,
            bit_reversed: false,
            meta: meta(58, 10),
        },
        eval_mod: EvalModPlan {
            eval_mod_type: EvalModType::CosHK,
            log_message_ratio: 8,
            f_mod_degree: 30,
            f_mod_interval: 12,
            f_mod_log_interval_reduction: 3,
            f_mod_inv_degree: None,
            scaling: None,
            split_strategy: SplitStrategy::MinDepth,
            meta: meta(60, 10),
        },
        slots_to_coeffs: DFTPlan {
            kind: DFTType::Decode,
            factorization_depth: vec![3, 3, 3],
            giant_steps: vec![16, 16, 16],
            format: DFTOutputFormat::SplitRealAndImag,
            scaling: None,
            bit_reversed: false,
            meta: meta(36, 10),
        },
    };

    let base2k = params.base2k;
    let log_delta = params.prec.log_delta;
    let log_modulus_in = log_delta + plan.eval_mod.log_message_ratio;
    let n = 1 << (LOG_SLOTS + 1);
    let m = n / 2;

    // Size the bootstrap modulus straight from the plan (no need to compile
    // first): input modulus + the bits the three stages consume + output
    // head-room.
    let k_boot = (log_modulus_in + plan.consumed_bits() + 4 * log_delta).next_multiple_of(base2k);

    let module = Module::<BE>::new(n as u64);
    let host_module = Module::<HostBytesBackend>::new(n as u64);
    let encoder = Encoder::<E>::new::<F>(m).unwrap();

    let tp = CKKSTestParams {
        n,
        base2k,
        k: k_boot,
        prec: meta(log_delta, k_boot - log_delta),
        hw: params.hw.min(1 << LOG_SLOTS),
        dsize: params.dsize,
    };

    // One scratch for the whole pipeline (plaintext precision sized for the
    // largest plaintext op, EvalMod).
    let mut scratch = {
        let mut c = module.ckks_ciphertext_alloc(base2k.into(), k_boot.into());
        c.set_meta(tp.prec);
        ScratchOwned::<BE>::alloc(module.ckks_all_ops_with_atk_tmp_bytes(
            &c,
            &tp.tsk_layout(),
            &tp.atk_layout(),
            &plan.eval_mod.meta,
        ))
    };

    let ctx =
        BootstrappingContext::<BE, F>::compile(&module, &host_module, &encoder, base2k.into(), &plan, &mut scratch.borrow())
            .unwrap();
    let (sk_raw, sk) = gen_sk_with_raw(&tp, &module, &host_module, [0u8; 32]);
    let tsk = gen_tsk(&tp, &module, &sk_raw, &mut scratch.borrow());

    // Galois keys: both transforms' rotations + the split forward conjugation.
    let order = module.cyclotomic_order();
    let mut atks = HashMap::new();
    for el in ctx
        .coeffs_to_slots
        .galois_elements(order)
        .into_iter()
        .chain(ctx.slots_to_coeffs.galois_elements(order))
    {
        atks.entry(el)
            .or_insert_with(|| gen_atk(&tp, &module, el, &sk_raw, &mut scratch.borrow()));
    }
    let conj_key = gen_atk(&tp, &module, -1, &sk_raw, &mut scratch.borrow());

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
        meta(log_delta, log_modulus_in - log_delta),
        &mut scratch.borrow(),
    );

    // 1) ModUp, then relabel at the input-modulus scale (free /message-ratio):
    //    `I(X)·q` becomes the integer part, the message the residue `Δ·c/q`.
    let mut ct = module.ckks_ciphertext_alloc(base2k.into(), k_boot.into());
    module.ckks_mod_up_into(&mut ct, &ct0, &mut scratch.borrow()).unwrap();
    ct.set_meta(meta(log_modulus_in, k_boot - log_modulus_in));

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

    // 3) EvalMod each half. Compact first: the ct×ct squaring needs compact
    //    operands (storage == effective_k), and the budget dropped in step 2.
    let ct_real = ct_real.compact(&module, &mut scratch.borrow()).unwrap();
    let ct_imag = ct_imag.compact(&module, &mut scratch.borrow()).unwrap();
    let mut res_real = module.ckks_ciphertext_alloc(base2k.into(), ct_real.max_k());
    let mut res_imag = module.ckks_ciphertext_alloc(base2k.into(), ct_imag.max_k());
    module
        .ckks_eval_mod(&mut res_real, &ct_real, &ctx.eval_mod, &tsk, &mut scratch.borrow())
        .unwrap();
    module
        .ckks_eval_mod(&mut res_imag, &ct_imag, &ctx.eval_mod, &tsk, &mut scratch.borrow())
        .unwrap();

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
    module
        .ckks_mul_pow2_assign(&mut ct_out, log_modulus_in - log_delta, &mut scratch.borrow())
        .unwrap();

    // Decrypt, decode, and confirm the slots are recovered. Cap the budget so
    // `log_delta + log_budget <= 127` fits the i128 decode codec (the unused
    // high-order budget is dropped losslessly).
    let prec = meta(
        ct_out.log_delta(),
        ct_out.log_budget().min(127usize.saturating_sub(ct_out.log_delta())),
    );
    let mut pt_out = module.ckks_pt_vec_alloc(base2k.into(), prec);
    module.ckks_decrypt(&mut pt_out, &ct_out, &sk, &mut scratch.borrow()).unwrap();

    let pt_host = pt_out.to_host_owned::<BE>();
    let (mut re_out, mut im_out) = (vec![F::zero(); m], vec![F::zero(); m]);
    encoder.decode_reim(&pt_host, &mut re_out, &mut im_out).unwrap();

    for (got, want, tag) in [(&re_out, &re, "re"), (&im_out, &im, "im")] {
        let s = precision_stats(got, want, log_delta);
        assert!(
            s.avg_log2_prec >= 5.0,
            "bootstrap_e2e ({tag}): {:.1} bits < 5.0 (worst got={} want={})",
            s.avg_log2_prec,
            s.worst_got,
            s.worst_want,
        );
    }
}
