//! PaCo end-to-end tests through the public sequential bootstrap API.
//!
//! Runs [`CKKSPaCoOps::ckks_paco_bootstrap_direct_into`] on a real exhausted
//! ciphertext under the structured key and gates the result against the
//! independent cleartext oracle:
//!
//! - **Budget**: the run consumes exactly [`PaCoPlan::consumed_bits`].
//! - **Metadata**: scale and sparse-packing metadata match the public API
//!   contract.
//! - **Recovery**: the decrypted output matches the oracle's final slots, and
//!   the oracle's recovered coefficients match the true message within the
//!   small-angle bound, closing the end-to-end chain
//!   `decrypt(bootstrap(ct)) ≈ m_{i·N/C}`.
//!
//! The final comparison accounts for η's `q/(4π)` amplification of the
//! computation noise — the PaCo precision structure (paper §7: precision
//! `≈ log_delta − log q` bits).

use crate::api::CKKSEncodingOps;
use std::collections::HashMap;

use poulpy_core::{
    GLWECopy, TransferInto,
    layouts::{GLWESecretPreparedFactory, ModuleCoreAlloc},
};
use poulpy_hal::{
    api::{CnvPVecAlloc, NegacyclicFFT, NegacyclicFFTNew, ScratchOwnedBorrow},
    layouts::{HostBytesBackend, Module},
    source::Source,
};

use crate::{
    CKKSInfos, SetCKKSInfos,
    api::{CKKSLinearTransformationOps, CKKSPaCoOps, PaCoScalar},
    default::paco::ops::PaCoSlotOps,
    encoding::paco::coeff_enc::glwe_column_residues,
    layouts::{CKKSModuleAlloc, PaCoContext, PaCoDFTPlan, PaCoKeysPrepared, PaCoPlan, PaCoSecretSpec, ScratchArenaTakeCKKS},
    test_suite::reference_encoder::ReferenceEncoder,
    test_suite::{
        CKKSTestParams,
        helpers::{
            TestContextBackend, TestContextHostModule, TestContextModule, TestScalar, alloc_scratch, ckks_encrypt,
            ckks_encrypt_coeffs, ckks_spec, gen_atk, gen_tsk,
        },
        paco_ops::assert_slots,
        paco_reference_model::seq_paco_reference,
    },
};

const PACO_H: usize = 4;
const PACO_C: usize = 8;

/// Full seqPaCo on a real exhausted ciphertext, gated against the independent
/// cleartext oracle (suite-scale parameters).
pub fn test_paco_seq_bootstrap<BE, F, E>(params: CKKSTestParams, module: &Module<BE>, host_module: &Module<HostBytesBackend>)
where
    BE: TestContextBackend,
    Module<BE>: TestContextModule<BE>
        + CKKSEncodingOps<BE, F>
        + CKKSLinearTransformationOps<BE>
        + PaCoSlotOps<BE>
        + CKKSPaCoOps<BE, F>
        + CnvPVecAlloc<BE>
        + GLWECopy<BE>,
    Module<HostBytesBackend>: TestContextHostModule,
    F: TestScalar + PaCoScalar,
    E: NegacyclicFFT<F> + NegacyclicFFTNew<F>,
{
    let _ = (module, host_module);
    for slot_order in [
        crate::layouts::PaCoSlotOrder::Natural,
        crate::layouts::PaCoSlotOrder::BitRevLow,
    ] {
        // Suite-scale case (C = 8).
        seq_bootstrap_case::<BE, F, E>(params, PACO_H, PACO_C, vec![2, 3], vec![2, 1], slot_order);
        // Large-C payoff case (C = 32, g1 = 1): the configuration where
        // BitRevLow removes the folded StC′ permutation entirely (first
        // butterfly factor 14 → 3 diagonals, see the lt.rs pins).
        seq_bootstrap_case::<BE, F, E>(params, 2, 32, vec![2, 2, 3], vec![1, 1, 1, 1, 1], slot_order);
    }
}

/// Paper-scale run (`N = 2^15`, `h = 64` — the PaCo I ring/weight): ignored
/// by default, run explicitly with `--ignored`.
pub fn test_paco_paper_scale<BE, F, E>(params: CKKSTestParams, module: &Module<BE>, host_module: &Module<HostBytesBackend>)
where
    BE: TestContextBackend,
    Module<BE>: TestContextModule<BE>
        + CKKSEncodingOps<BE, F>
        + CKKSLinearTransformationOps<BE>
        + PaCoSlotOps<BE>
        + CKKSPaCoOps<BE, F>
        + CnvPVecAlloc<BE>
        + GLWECopy<BE>,
    Module<HostBytesBackend>: TestContextHostModule,
    F: TestScalar + PaCoScalar,
    E: NegacyclicFFT<F> + NegacyclicFFTNew<F>,
{
    let _ = (module, host_module);
    for slot_order in [
        crate::layouts::PaCoSlotOrder::Natural,
        crate::layouts::PaCoSlotOrder::BitRevLow,
    ] {
        seq_bootstrap_case::<BE, F, E>(
            CKKSTestParams { n: 1 << 15, ..params },
            64,
            8,
            vec![2, 3],
            vec![2, 1],
            slot_order,
        );
    }
}

/// Shared body: seqPaCo at the given `(h, C)` on `params.n` with the given
/// chain schedules and slot-order convention, gated against the oracle's
/// final result and the public metadata contract (both of which are
/// convention-independent — the relabel lives strictly between the two DFT
/// chains).
fn seq_bootstrap_case<BE, F, E>(
    params: CKKSTestParams,
    paco_h: usize,
    paco_c: usize,
    c2s_depth: Vec<usize>,
    stc_depth: Vec<usize>,
    slot_order: crate::layouts::PaCoSlotOrder,
) where
    BE: TestContextBackend,
    Module<BE>: TestContextModule<BE>
        + CKKSEncodingOps<BE, F>
        + CKKSLinearTransformationOps<BE>
        + PaCoSlotOps<BE>
        + CKKSPaCoOps<BE, F>
        + CnvPVecAlloc<BE>
        + GLWECopy<BE>,
    Module<HostBytesBackend>: TestContextHostModule,
    F: TestScalar + PaCoScalar,
    E: NegacyclicFFT<F> + NegacyclicFFTNew<F>,
{
    let log_delta = params.prec().log_delta();
    // Base modulus below the working scale (the PaCo precision requirement
    // Δ ≥ q), capped at 52 bits for exact f64 residues.
    let k_in = (log_delta - 2).min(52);
    let log_msg = k_in - 10;

    // Full parameterization: C = 8 → c2s units [L1..L4, ψ] scheduled [2, 3]
    // (ψ fused with the last two layers), stc units [pack, L1, L2] scheduled
    // [2, 1] (pack composed with the first layer). The stc chain runs at a
    // lower scale and carries a dyadic test scaling (compensated by the
    // relabel) to exercise the multi-scale/scaling plumbing.
    let factor_budget = params.prec().log_budget();
    let plan = PaCoPlan::new(params.n.trailing_zeros() as usize, paco_h, paco_c, k_in as u32)
        .and_then(|plan| {
            Ok((
                plan,
                PaCoDFTPlan::new(c2s_depth.clone(), vec![2; c2s_depth.len()], log_delta, factor_budget, 1.0)?,
                PaCoDFTPlan::new(
                    stc_depth.clone(),
                    vec![2; stc_depth.len()],
                    log_delta - 6,
                    factor_budget,
                    2f64.powi(-4),
                )?,
            ))
        })
        .and_then(|(plan, c2s, stc)| plan.with_slot_order(slot_order).with_evaluation(log_delta, 16, c2s, stc))
        .unwrap();
    let p = plan.clone();
    // Headroom = final budget: must hold the recovered |m̃| < 2^log_msg.
    let k_boot = plan.k_boot(params.base2k, log_msg + 8).unwrap();
    let params = CKKSTestParams { k: k_boot, ..params };

    let module = Module::<BE>::new(params.n as u64);
    let host_module = Module::<HostBytesBackend>::new(params.n as u64);
    let encoder_full = ReferenceEncoder::<E>::new::<F>(params.n / 2).unwrap();
    let encoder_block = ReferenceEncoder::<E>::new::<F>(2 * p.c()).unwrap();
    let mut scratch = alloc_scratch(&params, &module);

    // Structured secret + key material.
    let mut source = Source::new([9u8; 32]);
    let spec = PaCoSecretSpec::sample(&p, &mut source).unwrap();
    let glwe_infos = params.glwe_layout();
    let mut sk_host = host_module.glwe_secret_alloc_from_infos(&glwe_infos);
    spec.fill_glwe_secret(&p, &mut sk_host).unwrap();
    let mut sk_raw = module.glwe_secret_alloc_from_infos(&glwe_infos);
    sk_host.transfer_into(&mut sk_raw);
    let mut sk = module.glwe_secret_prepared_alloc_from_infos(&glwe_infos);
    module.glwe_secret_prepare(&mut sk, &sk_raw);

    let mut atks = HashMap::new();
    for p_el in plan.galois_elements() {
        atks.entry(p_el)
            .or_insert_with(|| gen_atk(&params, &module, p_el, &sk_raw, &mut scratch.borrow()));
    }
    let tsk = gen_tsk(&params, &module, &sk_raw, &mut scratch.borrow());

    // bsk_t = Enc(σ_t) at the bootstrap width.
    let bsk: [_; 4] = (0..4)
        .map(|t| {
            let s = spec
                .sigma_slots_with::<F, _>(&p, t, &mut |coeffs, re, im| encoder_block.unpack_reim_coeffs(coeffs, re, im))
                .unwrap();
            let re: Vec<F> = s.iter().map(|x| x.re).collect();
            let im: Vec<F> = s.iter().map(|x| x.im).collect();
            ckks_encrypt(
                &params,
                &module,
                &host_module,
                &encoder_full,
                &sk,
                k_boot,
                &re,
                &im,
                &mut scratch.borrow(),
            )
        })
        .collect::<Vec<_>>()
        .try_into()
        .unwrap_or_else(|_| unreachable!("exactly four PaCo bootstrapping keys"));
    let keys = PaCoKeysPrepared::new(&plan, bsk, atks, tsk, None).unwrap();
    let ctx = PaCoContext::<BE, F>::compile(&module, params.base2k.into(), plan.clone(), &mut scratch.borrow()).unwrap();
    // The bootstrap output is allocated at its maximum final level `k_out`
    // (`k_boot` is the wider seed/working width, not a valid output level).
    let k_out = ctx.max_output_k(&keys).unwrap();

    // Real exhausted input at k_in; message coefficients |m| < 2^log_msg.
    let coeffs: Vec<F> = (0..params.n)
        .map(|i| F::from_f64(0.4 * (((i.wrapping_mul(2654435761) % 1024) as f64) / 512.0 - 1.0)).unwrap())
        .collect();
    let mut ct_in = ckks_encrypt_coeffs(
        &params,
        &module,
        &host_module,
        &sk,
        k_in,
        &coeffs,
        ckks_spec(params.n, params.base2k, log_msg, 10),
        &mut scratch.borrow(),
    );
    // One branch recovers `C` coefficients, so an input at this sparsity is
    // exactly the single-branch seqPaCo schedule this test models.
    ct_in.set_log_sparsity((p.n() / p.c()).trailing_zeros() as usize);

    // Cleartext model from the same residues.
    let host_in = ct_in.to_host_owned::<BE>();
    let ct0 = glwe_column_residues(host_in.data(), 0, k_in, params.base2k).unwrap();
    let ct1 = glwe_column_residues(host_in.data(), 1, k_in, params.base2k).unwrap();
    let oracle = seq_paco_reference(&ct0, &ct1, &spec, &p, &encoder_block, log_msg);

    // The final decode is msg-valued (the output is re-anchored onto the
    // input's scale Δ_in = log_msg). Expected noise is approximately
    // k_in − log_delta + 10 − log_msg bits after folded-η amplification,
    // while the signal is about 2^-2 (|msg| ≤ 0.4). The midpoint keeps a
    // clear margin from structural failures at signal level.
    let final_bound = ((k_in as f64 - log_delta as f64 + 10.0 - log_msg as f64) + (-2.0)) / 2.0;

    // Run through the caller-allocated public direct API (one branch) and
    // compare only the final ciphertext with the independent cleartext model.
    let bsk_budget = k_boot - log_delta;
    let mut out = module.ckks_ciphertext_alloc(params.base2k.into(), k_out);
    module
        .ckks_paco_bootstrap_direct_into::<_, _>(&mut out, &ct_in, &ctx, &keys, &mut scratch.borrow())
        .unwrap();
    assert_slots::<BE, F, E>(
        &format!("paco_seq_bootstrap[final]({paco_h},{paco_c},{slot_order:?})"),
        &module,
        &host_module,
        &encoder_full,
        &out,
        &sk,
        &oracle.final_slots,
        final_bound,
        &mut scratch.borrow(),
    );

    // Leveled output: request an output below `k_out`, deliberately NOT a multiple of
    // `base2k`. The blind rotation produces the phase directly at the lower working
    // width, so the whole circuit runs narrower and must still recover the message.
    if k_out.as_usize() > 2 * params.base2k {
        let k_low = k_out.as_usize() - 5;
        let mut out_low = module.ckks_ciphertext_alloc(params.base2k.into(), k_low.into());
        module
            .ckks_paco_bootstrap_direct_into::<_, _>(&mut out_low, &ct_in, &ctx, &keys, &mut scratch.borrow())
            .unwrap();
        assert_eq!(
            poulpy_core::layouts::LWEInfos::k(&out_low).as_usize(),
            k_low,
            "reduced-level bootstrap must produce exactly the requested output level"
        );
        assert_slots::<BE, F, E>(
            &format!("paco_seq_bootstrap[reduced-level]({paco_h},{paco_c},{slot_order:?})"),
            &module,
            &host_module,
            &encoder_full,
            &out_low,
            &sk,
            &oracle.final_slots,
            final_bound + 7.0,
            &mut scratch.borrow(),
        );
    }

    // The final relabel compensates both eta's q/4 factor and the configured
    // dyadic chain scaling. This re-anchors the *decoded value* onto the
    // exhausted input's message; the output metadata scale remains anchored
    // at the bootstrapping-key scale. These scales happen to coincide for the
    // f64 suites, but differ when the Quad suite uses an 80-bit working scale
    // with the input modulus capped at 52 bits.
    let relabel = p.log_q() as i64 - 2 - log_msg as i64 - plan.extra_scale_log2();
    let expected_output_scale = usize::try_from(plan.log_delta_bsk() as i64 - relabel)
        .expect("validated test plan must produce a non-negative output scale");
    assert_eq!(
        out.log_delta(),
        expected_output_scale,
        "bootstrap output scale must include the validated PaCo relabel"
    );
    assert_eq!(
        out.log_sparsity(),
        (p.n() / p.c()).trailing_zeros() as usize,
        "sequential PaCo must mark the N/C coefficient stride"
    );

    // Budget: exactly the plan's schedule, minus the relabel credit — the
    // ~log(q/Δ_in) headroom bits between the input's message scale and its
    // modulus (adjusted by the chain-scaling compensation), returned when the
    // output is re-anchored onto Δ_in = log_msg.
    assert_eq!(
        bsk_budget as i64 - out.log_budget() as i64,
        plan.consumed_bits() as i64 - relabel,
        "bootstrap must consume the per-chain schedule's budget bits (minus the scale re-anchoring)"
    );

    // End-to-end recovery: the oracle was fed the real residues (and thus the
    // input encryption noise), its final slots match the homomorphic output,
    // and its recovered coefficients must match the original message.
    let mut max_err = 0.0f64;
    for i in 0..p.c() {
        let want = (coeffs[i * p.n() / p.c()].to_f64().unwrap() * (1u64 << log_msg) as f64).round();
        max_err = max_err.max((oracle.m_recovered[i] - want).abs());
    }
    let tol = (1u64 << log_msg) as f64 * 2f64.powi(-9) + 128.0;
    assert!(max_err < tol, "recovered coefficients off by {max_err:.1} (tol {tol:.1})");

    // The input is generic over any backend-readable ciphertext: a
    // scratch-carved view of the same exhausted ciphertext must produce a
    // bit-identical bootstrap output.
    let mut view_arena = alloc_scratch(&params, &module);
    let view_scratch = view_arena.borrow();
    let (mut ct_view, _rest) = view_scratch.take_ckks_ciphertext_like_scratch(&ct_in);
    module.glwe_copy(&mut ct_view, &ct_in);
    let mut out_view = module.ckks_ciphertext_alloc(params.base2k.into(), k_out);
    module
        .ckks_paco_bootstrap_direct_into::<_, _>(&mut out_view, &ct_view, &ctx, &keys, &mut scratch.borrow())
        .unwrap();
    assert_eq!(
        out_view.meta(),
        out.meta(),
        "view-input bootstrap must reproduce the owned-input metadata"
    );
    assert_eq!(
        out_view.data(),
        out.data(),
        "view-input bootstrap must be bit-identical to the owned-input run"
    );
}
