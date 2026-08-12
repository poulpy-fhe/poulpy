//! Parallel PaCo (Algorithm 5) integration tests.
//!
//! - **Full-coefficient recovery**: with `κ = 4` branches the sequential
//!   driver must recover **all** coefficients at indices `j·N/D`
//!   (`D = κ·C`), checked in the coefficient domain against the true
//!   message (this also pins the `glwe_rotate` shift directions: a sign
//!   error would land the classes on the wrong indices).
//! - **Parallel ≡ sequential**: the scoped-thread driver must produce a
//!   **bit-identical** ciphertext (same limbs, same metadata) — parallelism
//!   is pure orchestration.

use crate::api::CKKSEncodingOps;
use poulpy_core::layouts::IntPolyInfos;
use std::collections::HashMap;

use poulpy_core::{
    GLWERotate, ModuleTransfer,
    layouts::{GLWEInfos, GLWELayout, GLWESecretPreparedFactory, LWEInfos, ModuleCoreAlloc, Rank},
};
use poulpy_hal::{
    api::{CnvPVecAlloc, NegacyclicFFT, NegacyclicFFTNew, ScratchOwnedAlloc, ScratchOwnedBorrow},
    layouts::{HostBytesBackend, Module, ScratchOwned, ZnxView},
    source::Source,
};

use crate::SlotsKind;
use crate::{
    CKKSInfos, CKKSMeta, SetCKKSInfos,
    api::{CKKSLinearTransformationOps, CKKSPaCoOps, PaCoScalar},
    default::paco::ops::PaCoSlotOps,
    layouts::{CKKSModuleAlloc, PaCoContext, PaCoDFTPlan, PaCoKeySet, PaCoKeysPrepared, PaCoPlan, PaCoSecretSpec, PaCoWorker},
    test_suite::reference_encoder::ReferenceEncoder,
    test_suite::{
        CKKSTestParams,
        helpers::{
            TestContextBackend, TestContextHostModule, TestContextModule, TestScalar, alloc_scratch, ckks_encrypt,
            ckks_encrypt_coeffs, ckks_spec, gen_atk, gen_tsk,
        },
    },
};

const PACO_H: usize = 4;
const PACO_C: usize = 8;
const KAPPA: usize = 4;

/// Asserts that a rejected public call did not alter either the ciphertext's
/// semantic/layout metadata or its backend bytes.
fn assert_ciphertext_unchanged<BE>(
    before: &crate::layouts::CKKSCiphertextOwned<HostBytesBackend>,
    after: &crate::layouts::CKKSCiphertextOwned<BE>,
) where
    BE: TestContextBackend,
{
    let after = after.to_host_owned::<BE>();
    assert_eq!(before.meta(), after.meta(), "a rejected call changed CKKS metadata");
    assert_eq!(before.n(), after.n(), "a rejected call changed the output degree");
    assert_eq!(before.rank(), after.rank(), "a rejected call changed the output rank");
    assert_eq!(before.base2k(), after.base2k(), "a rejected call changed the output radix");
    assert_eq!(before.k(), after.k(), "a rejected call changed the output torus width");
    assert_eq!(
        before.max_size(),
        after.max_size(),
        "a rejected call changed the output stored limb width"
    );
    assert_eq!(before.data().raw(), after.data().raw(), "a rejected call changed output data");
}

/// Decrypts on `BE` at full precision, downloads, and reconstructs the raw
/// plaintext coefficients host-side as `f64` values at the ciphertext's
/// scale (limb-wise, no float decode codec). No host-readability is assumed
/// of `BE`.
fn decrypt_coeffs_host<BE>(
    module: &Module<BE>,
    ct: &crate::layouts::CKKSCiphertextOwned<BE>,
    sk: &poulpy_core::layouts::prepared::GLWESecretPrepared<BE::OwnedBuf, BE>,
    n: usize,
    scratch: &mut poulpy_hal::layouts::ScratchArena<'_, BE>,
) -> Vec<f64>
where
    BE: TestContextBackend,
    Module<BE>: TestContextModule<BE>,
{
    use crate::test_suite::helpers::ckks_decrypt_with_prec;
    let (log_delta, base2k) = (ct.log_delta(), ct.base2k());
    let prec = crate::CKKSLayout {
        glwe_layout: GLWELayout {
            n: ct.n(),
            base2k,
            k: ct.k(),
            rank: Rank(1),
        },
        meta: CKKSMeta {
            log_sparsity: 0,
            log_delta,
            slots: SlotsKind::Complex,
        },
    };
    let pt = ckks_decrypt_with_prec(module, ct, sk, prec, scratch).unwrap();
    // Digits are aligned to the plaintext's STORAGE width (max_k, a whole
    // number of limbs), not its effective k — mirror the codec convention.
    let max_k = pt.encoded_k().as_usize() as i32;
    let data = pt.data();
    let b2k = base2k.as_usize() as i32;
    (0..n)
        .map(|i| {
            (0..data.size())
                .map(|j| data.at(0, j)[i] as f64 * 2.0f64.powi(max_k - log_delta as i32 - (j as i32 + 1) * b2k))
                .sum()
        })
        .collect()
}

/// Algorithm 5: sequential-loop full-coefficient recovery, then the parallel
/// driver bit-identical to it.
pub fn test_paco_parallel_bootstrap<BE, F, E>(
    params: CKKSTestParams,
    _module: &Module<BE>,
    _host_module: &Module<HostBytesBackend>,
) where
    BE: TestContextBackend,
    Module<BE>: TestContextModule<BE>
        + CKKSEncodingOps<BE, F>
        + CKKSLinearTransformationOps<BE>
        + PaCoSlotOps<BE>
        + CKKSPaCoOps<BE, F>
        + CnvPVecAlloc<BE>
        + GLWERotate<BE>
        + Sync,
    Module<HostBytesBackend>: TestContextHostModule + Sync,
    F: TestScalar + PaCoScalar,
    E: NegacyclicFFT<F> + NegacyclicFFTNew<F> + Sync,
{
    let log_delta = params.prec().log_delta();
    let k_in = (log_delta - 2).min(52);
    let log_msg = k_in - 10;

    let plan = PaCoPlan::new(params.n.trailing_zeros() as usize, PACO_H, PACO_C, k_in as u32)
        .unwrap()
        // BitRevLow here (the sequential suite covers both conventions): the
        // parallel driver is convention-oblivious, and this is the only
        // homomorphic coverage of the Mask-tail × conjugated-chain
        // combination.
        .with_slot_order(crate::layouts::PaCoSlotOrder::BitRevLow)
        .with_evaluation(
            log_delta,
            16,
            // ψ scheduled alone: exercises the operation-lean mask tail (one
            // fused conj-rotate keyswitch + one μ multiplication) end to end;
            // the sequential suite covers the merged conjugation-augmented
            // pair. Same factor count as [3, 2], so the budget is unchanged.
            PaCoDFTPlan::new(vec![4, 1], vec![2, 2], log_delta, 16, 1.0).unwrap(),
            PaCoDFTPlan::uniform(3, 2, 2, log_delta, 16).unwrap(),
        )
        .unwrap();
    let p = plan.clone();
    let k_boot = plan.k_boot(params.base2k, log_msg + 8).unwrap();
    let params = CKKSTestParams { k: k_boot, ..params };
    let d = KAPPA * p.c();
    let stride = p.n() / d;

    let module = Module::<BE>::new(params.n as u64);
    let host_module = Module::<HostBytesBackend>::new(params.n as u64);
    let encoder_full = ReferenceEncoder::<E>::new::<F>(params.n / 2).unwrap();
    let encoder_block = ReferenceEncoder::<E>::new::<F>(2 * p.c()).unwrap();
    let mut scratch = alloc_scratch(&params, &module);

    // Structured secret and key material shared by every branch.
    let mut source = Source::new([11u8; 32]);
    let spec = PaCoSecretSpec::sample(&p, &mut source).unwrap();
    let glwe_infos = params.glwe_layout();
    let mut sk_host = host_module.glwe_secret_alloc_from_infos(&glwe_infos);
    spec.fill_glwe_secret(&p, &mut sk_host).unwrap();
    let sk_raw = module.upload_glwe_secret(&sk_host);
    let mut sk = module.glwe_secret_prepared_alloc_from_infos(&glwe_infos);
    module.glwe_secret_prepare(&mut sk, &sk_raw);

    let mut atks = HashMap::new();
    for p_el in plan.galois_elements() {
        atks.entry(p_el)
            .or_insert_with(|| gen_atk(&params, &module, p_el, &sk_raw, &mut scratch.borrow()));
    }
    let tsk = gen_tsk(&params, &module, &sk_raw, &mut scratch.borrow());

    let bsk = std::array::from_fn(|t| {
        let s = spec
            .sigma_slots_with(&p, t, &mut |coeffs, re, im| encoder_block.unpack_reim_coeffs(coeffs, re, im))
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
    });
    let keys = PaCoKeysPrepared::new(&plan, bsk, atks, tsk, None).unwrap();
    let ctx = PaCoContext::<BE, F>::compile(&module, params.base2k.into(), plan, &mut scratch.borrow()).unwrap();

    // Maximum final output level; the bootstrap output is allocated here (a lower
    // level would run the circuit cheaper and is exercised separately below).
    let k_out = ctx.max_output_k(&keys).unwrap();

    // Exhausted input.
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
    // The branch count is derived from the input: an input at this sparsity has
    // `D = KAPPA*C` live coefficients, one per branch class, which is exactly
    // the set this test verifies below.
    ct_in.set_log_sparsity(stride.trailing_zeros() as usize);

    // Public-entry validation must reject malformed schedules and layouts
    // before touching the caller's output.
    let mut rejected = module.ckks_ciphertext_alloc(ctx.base2k(), k_out);
    let mut too_sparse = ct_in.clone();
    too_sparse.set_log_sparsity((p.n() / p.c()).trailing_zeros() as usize + 1);
    let before = rejected.to_host_owned::<BE>();
    let error = module
        .ckks_paco_bootstrap_direct_into::<_, _>(&mut rejected, &too_sparse, &ctx, &keys, &mut scratch.borrow())
        .expect_err("an input with fewer live coefficients than C must be rejected");
    assert!(
        error.to_string().contains("fewer than the plan's"),
        "unexpected sparsity error: {error:#}"
    );
    assert_ciphertext_unchanged::<BE>(&before, &rejected);

    let required = module
        .ckks_paco_bootstrap_direct_tmp_bytes(&rejected, &ct_in, &ctx, &keys)
        .unwrap();
    assert!(required > 0, "a PaCo branch must require non-zero scratch");
    let mut no_scratch = ScratchOwned::<BE>::alloc(0);
    let before = rejected.to_host_owned::<BE>();
    let error = module
        .ckks_paco_bootstrap_direct_into::<_, _>(&mut rejected, &ct_in, &ctx, &keys, &mut no_scratch.borrow())
        .expect_err("undersized caller scratch must be rejected");
    assert!(error.to_string().contains("scratch bytes"), "unexpected error: {error:#}");
    assert_ciphertext_unchanged::<BE>(&before, &rejected);

    let before = rejected.to_host_owned::<BE>();
    let error = module
        .ckks_paco_bootstrap_into::<_, _>(&mut rejected, &ct_in, &ctx, &keys, &mut scratch.borrow())
        .expect_err("encapsulated mode must require an encapsulation key");
    assert!(
        error.to_string().contains("dense-to-PaCo switching key"),
        "unexpected error: {error:#}"
    );
    assert_ciphertext_unchanged::<BE>(&before, &rejected);
    let error = module
        .ckks_paco_bootstrap_tmp_bytes(&rejected, &ct_in, &ctx, &keys)
        .expect_err("the encapsulated scratch query must require an encapsulation key");
    assert!(
        error.to_string().contains("dense-to-PaCo switching key"),
        "unexpected error: {error:#}"
    );

    let bad_base2k = if params.base2k > 1 {
        params.base2k - 1
    } else {
        params.base2k + 1
    };
    let mut bad_output = module.ckks_ciphertext_alloc(bad_base2k.into(), keys.bootstrapping_keys()[0].k());
    let before = bad_output.to_host_owned::<BE>();
    let error = module
        .ckks_paco_bootstrap_direct_into::<_, _>(&mut bad_output, &ct_in, &ctx, &keys, &mut scratch.borrow())
        .expect_err("an output with the wrong radix must be rejected");
    assert!(error.to_string().contains("output base2k"), "unexpected error: {error:#}");
    assert_ciphertext_unchanged::<BE>(&before, &bad_output);

    // An output level above `k_out` cannot be produced by the circuit. The full
    // bootstrapping-key width (`k_out + consumed_bits`) is such a level, so it is
    // rejected up front.
    let too_high_k = keys.bootstrapping_keys()[0].k();
    assert!(
        too_high_k.as_usize() > k_out.as_usize(),
        "test precondition: bsk width exceeds k_out"
    );
    let mut too_high_output = module.ckks_ciphertext_alloc(ctx.base2k(), too_high_k);
    let before = too_high_output.to_host_owned::<BE>();
    let error = module
        .ckks_paco_bootstrap_direct_into::<_, _>(&mut too_high_output, &ct_in, &ctx, &keys, &mut scratch.borrow())
        .expect_err("an output level above k_out must be rejected");
    assert!(
        error.to_string().contains("exceeds the maximum bootstrap output level"),
        "unexpected error: {error:#}"
    );
    assert_ciphertext_unchanged::<BE>(&before, &too_high_output);

    // Worker modules and scratch are preflighted before any branch starts.
    let mut worker_rejected = module.ckks_ciphertext_alloc(ctx.base2k(), k_out);
    let before = worker_rejected.to_host_owned::<BE>();
    let mut rejected_caller_scratch = alloc_scratch(&params, &module);
    let mut undersized_workers = vec![PaCoWorker::new(
        Module::<BE>::new(params.n as u64),
        ScratchOwned::<BE>::alloc(0),
    )];
    let error = module
        .ckks_paco_bootstrap_parallel_direct_into::<_, _>(
            &mut worker_rejected,
            &ct_in,
            &ctx,
            &keys,
            &mut undersized_workers,
            &mut rejected_caller_scratch.borrow(),
        )
        .expect_err("undersized worker scratch must be reported");
    assert!(error.to_string().contains("worker 1 needs"), "unexpected error: {error:#}");
    assert_ciphertext_unchanged::<BE>(&before, &worker_rejected);

    let before = worker_rejected.to_host_owned::<BE>();
    let mut wrong_degree_workers = vec![PaCoWorker::new(
        Module::<BE>::new((params.n / 2) as u64),
        alloc_scratch(&params, &module),
    )];
    let error = module
        .ckks_paco_bootstrap_parallel_direct_into::<_, _>(
            &mut worker_rejected,
            &ct_in,
            &ctx,
            &keys,
            &mut wrong_degree_workers,
            &mut rejected_caller_scratch.borrow(),
        )
        .expect_err("a worker module for the wrong ring must be rejected");
    assert!(error.to_string().contains("module degree"), "unexpected error: {error:#}");
    assert_ciphertext_unchanged::<BE>(&before, &worker_rejected);

    // ── Sequential driver: all D coefficient classes recovered ────────────
    let mut seq = module.ckks_ciphertext_alloc(ctx.base2k(), k_out);
    let mut exact_scratch = ScratchOwned::<BE>::alloc(required);
    module
        .ckks_paco_bootstrap_direct_into::<_, _>(&mut seq, &ct_in, &ctx, &keys, &mut exact_scratch.borrow())
        .unwrap();
    let expected_sparsity = stride.trailing_zeros() as usize;
    assert_eq!(seq.log_sparsity(), expected_sparsity, "sequential output sparsity");

    // Coefficient-domain gate: coefficient j·N/D decodes to c_j itself (the
    // output is re-anchored onto the input's scale Δ_in = log_msg, so it
    // decodes to the same values the input decoded to); everything else zero.
    let want: Vec<F> = (0..params.n)
        .map(|j| {
            if j % stride == 0 {
                coeffs[j]
            } else {
                F::from_f64(0.0).unwrap()
            }
        })
        .collect();
    let got = decrypt_coeffs_host::<BE>(&module, &seq, &sk, params.n, &mut scratch.borrow());
    let mut max_err = 0.0f64;
    for j in 0..params.n {
        max_err = max_err.max((got[j] - want[j].to_f64().unwrap()).abs());
    }
    {
        let bound = -5.0;
        assert!(
            max_err.log2() < bound,
            "parallel PaCo full recovery: max coefficient error log2={:.1} (bound {bound:.1})",
            max_err.log2()
        );
    }

    // ── Parallel driver: bit-identical to the sequential loop ──────────────
    let mut par = module.ckks_ciphertext_alloc(ctx.base2k(), k_out);
    // Two background workers may complete branches out of order while the
    // caller recombines them in canonical sequential order.
    let mut workers = (0..2)
        .map(|_| PaCoWorker::new(Module::<BE>::new(params.n as u64), alloc_scratch(&params, &module)))
        .collect::<Vec<_>>();
    module
        .ckks_paco_bootstrap_parallel_direct_into::<_, _>(&mut par, &ct_in, &ctx, &keys, &mut workers, &mut scratch.borrow())
        .unwrap();

    assert_eq!(seq.log_delta(), par.log_delta(), "metadata must match");
    assert_eq!(seq.log_budget(), par.log_budget(), "metadata must match");
    assert_eq!(par.log_sparsity(), expected_sparsity, "parallel output sparsity");
    let (seq_host, par_host) = (seq.to_host_owned::<BE>(), par.to_host_owned::<BE>());
    assert_eq!(
        seq_host.data().raw(),
        par_host.data().raw(),
        "parallel output must be bit-identical"
    );

    // Worker modules and arenas remain reusable across calls.
    let mut par_reused = module.ckks_ciphertext_alloc(ctx.base2k(), k_out);
    module
        .ckks_paco_bootstrap_parallel_direct_into::<_, _>(
            &mut par_reused,
            &ct_in,
            &ctx,
            &keys,
            &mut workers,
            &mut scratch.borrow(),
        )
        .unwrap();
    assert_eq!(
        seq_host.data().raw(),
        par_reused.to_host_owned::<BE>().data().raw(),
        "reused PaCo workers must remain deterministic",
    );
}

/// Sparse-secret encapsulation: the application keeps a standard dense
/// ternary key; the bootstrap interior runs under the structured PaCo key.
/// Input is encrypted (and the output verified) under the **dense** key.
pub fn test_paco_encapsulated_bootstrap<BE, F, E>(
    params: CKKSTestParams,
    _module: &Module<BE>,
    _host_module: &Module<HostBytesBackend>,
) where
    BE: TestContextBackend,
    Module<BE>: TestContextModule<BE>
        + CKKSEncodingOps<BE, F>
        + CKKSLinearTransformationOps<BE>
        + PaCoSlotOps<BE>
        + CKKSPaCoOps<BE, F>
        + CnvPVecAlloc<BE>
        + GLWERotate<BE>
        + Sync,
    Module<HostBytesBackend>: TestContextHostModule + Sync,
    F: TestScalar + PaCoScalar,
    E: NegacyclicFFT<F> + NegacyclicFFTNew<F> + Sync,
{
    use crate::test_suite::helpers::gen_sk_with_raw;
    use poulpy_core::{GLWEAutomorphismKeyEncryptSk, GLWESwitchingKeyEncryptSk, GLWETensorKeyEncryptSk};

    let log_delta = params.prec().log_delta();
    let k_in = (log_delta - 2).min(52);
    let log_msg = k_in - 10;
    let kappa = 2usize;

    let plan = PaCoPlan::new(params.n.trailing_zeros() as usize, PACO_H, PACO_C, k_in as u32)
        .unwrap()
        .with_evaluation(
            log_delta,
            16,
            // ψ alone: the encapsulated flow also runs the mask tail.
            PaCoDFTPlan::new(vec![4, 1], vec![2, 2], log_delta, 16, 1.0).unwrap(),
            PaCoDFTPlan::uniform(3, 2, 2, log_delta, 16).unwrap(),
        )
        .unwrap();
    let p = plan.clone();
    let k_boot = plan.k_boot(params.base2k, log_msg + 8).unwrap();
    let params = CKKSTestParams { k: k_boot, ..params };
    let stride = p.n() / (kappa * p.c());

    let module = Module::<BE>::new(params.n as u64);
    let host_module = Module::<HostBytesBackend>::new(params.n as u64);
    let encoder_full = ReferenceEncoder::<E>::new::<F>(params.n / 2).unwrap();
    let encoder_block = ReferenceEncoder::<E>::new::<F>(2 * p.c()).unwrap();
    let mut scratch = alloc_scratch(&params, &module);

    // Application (dense) key and PaCo (structured) key.
    let (sk_dense_raw, sk_dense) = gen_sk_with_raw(&params, &module, &host_module, [21u8; 32]);
    let mut source = Source::new([22u8; 32]);
    let spec = PaCoSecretSpec::sample(&p, &mut source).unwrap();
    let glwe_infos = params.glwe_layout();
    let mut sk_paco_host = host_module.glwe_secret_alloc_from_infos(&glwe_infos);
    spec.fill_glwe_secret(&p, &mut sk_paco_host).unwrap();
    let sk_paco_raw = module.upload_glwe_secret(&sk_paco_host);
    let mut sk_paco = module.glwe_secret_prepared_alloc_from_infos(&glwe_infos);
    module.glwe_secret_prepare(&mut sk_paco, &sk_paco_raw);

    // All evaluation material goes through the backend-agnostic PaCoKeySet:
    // UNPREPARED (data-generic) keys assembled first, then prepared for the
    // backend in one step — the flow a GPU deployment would use.

    let (mut xa, mut xe) = (Source::new([1u8; 32]), Source::new([2u8; 32]));

    let atk_enc = params.atk_layout();
    let mut rotation_keys = HashMap::new();
    for p_el in plan.galois_elements() {
        let mut atk = module.glwe_automorphism_key_alloc_from_infos(&atk_enc);
        module.glwe_automorphism_key_encrypt_sk(
            &mut atk,
            p_el,
            &sk_dense_raw,
            &atk_enc,
            &mut xe,
            &mut xa,
            &mut scratch.borrow(),
        );
        rotation_keys.insert(p_el, atk);
    }
    let tsk_enc = params.tsk_layout();
    let mut tensor_key = module.glwe_tensor_key_alloc_from_infos(&tsk_enc);
    module.glwe_tensor_key_encrypt_sk(
        &mut tensor_key,
        &sk_dense_raw,
        &tsk_enc,
        &mut xe,
        &mut xa,
        &mut scratch.borrow(),
    );

    // The single encapsulation key: dense→PaCo, sized at the SMALL base
    // modulus (the paper's §8.2 security argument — the structured key never
    // acts as a key above q). Everything else (bsk, atks, tsk) is under the
    // dense application key, so no PaCo→dense key exists.
    let d2p_enc = params.ksk_layout(k_in);
    let mut dense_to_paco = module.glwe_switching_key_alloc_from_infos(&d2p_enc);
    module.glwe_switching_key_encrypt_sk(
        &mut dense_to_paco,
        &sk_dense_raw,
        &sk_paco_raw,
        &d2p_enc,
        &mut xe,
        &mut xa,
        &mut scratch.borrow(),
    );

    let bsk = std::array::from_fn(|t| {
        let s = spec
            .sigma_slots_with(&p, t, &mut |coeffs, re, im| encoder_block.unpack_reim_coeffs(coeffs, re, im))
            .unwrap();
        let re: Vec<F> = s.iter().map(|x| x.re).collect();
        let im: Vec<F> = s.iter().map(|x| x.im).collect();
        ckks_encrypt(
            &params,
            &module,
            &host_module,
            &encoder_full,
            &sk_dense,
            k_boot,
            &re,
            &im,
            &mut scratch.borrow(),
        )
    });

    let keyset = PaCoKeySet::new(&plan, bsk, rotation_keys, tensor_key, Some(dense_to_paco)).unwrap();
    let prepared = keyset.into_prepare(&plan, &module, &mut scratch.borrow()).unwrap();
    let ctx = PaCoContext::<BE, F>::compile(&module, params.base2k.into(), plan, &mut scratch.borrow()).unwrap();
    let k_out = ctx.max_output_k(&prepared).unwrap();

    // Exhausted input under the DENSE application key.
    let coeffs: Vec<F> = (0..params.n)
        .map(|i| F::from_f64(0.4 * (((i.wrapping_mul(2654435761) % 1024) as f64) / 512.0 - 1.0)).unwrap())
        .collect();
    let mut ct_in = ckks_encrypt_coeffs(
        &params,
        &module,
        &host_module,
        &sk_dense,
        k_in,
        &coeffs,
        ckks_spec(params.n, params.base2k, log_msg, 10),
        &mut scratch.borrow(),
    );
    // Selects the `kappa`-branch schedule this test models.
    ct_in.set_log_sparsity(stride.trailing_zeros() as usize);

    let mut out = module.ckks_ciphertext_alloc(ctx.base2k(), k_out);
    module
        .ckks_paco_bootstrap_into::<_, _>(&mut out, &ct_in, &ctx, &prepared, &mut scratch.borrow())
        .unwrap();
    let expected_sparsity = stride.trailing_zeros() as usize;
    assert_eq!(out.log_sparsity(), expected_sparsity, "sequential output sparsity");

    // The default parallel driver must be bit-identical to the default
    // sequential driver (same encapsulation, same per-branch limbs).
    let mut par = module.ckks_ciphertext_alloc(ctx.base2k(), k_out);
    let mut workers = vec![PaCoWorker::new(
        Module::<BE>::new(params.n as u64),
        alloc_scratch(&params, &module),
    )];
    module
        .ckks_paco_bootstrap_parallel_into::<_, _>(&mut par, &ct_in, &ctx, &prepared, &mut workers, &mut scratch.borrow())
        .unwrap();
    assert_eq!(out.log_delta(), par.log_delta(), "metadata must match");
    assert_eq!(out.log_budget(), par.log_budget(), "metadata must match");
    assert_eq!(par.log_sparsity(), expected_sparsity, "parallel output sparsity");
    let (out_host, par_host) = (out.to_host_owned::<BE>(), par.to_host_owned::<BE>());
    assert_eq!(
        out_host.data().raw(),
        par_host.data().raw(),
        "default parallel driver must be bit-identical to the default sequential driver"
    );

    // Verify under the DENSE key: all kappa·C coefficient classes recovered,
    // decoding to c_j itself (output re-anchored onto the input's scale).
    let want: Vec<F> = (0..params.n)
        .map(|j| {
            if j % stride == 0 {
                coeffs[j]
            } else {
                F::from_f64(0.0).unwrap()
            }
        })
        .collect();
    let got = decrypt_coeffs_host::<BE>(&module, &out, &sk_dense, params.n, &mut scratch.borrow());
    let mut max_err = 0.0f64;
    for j in 0..params.n {
        max_err = max_err.max((got[j] - want[j].to_f64().unwrap()).abs());
    }
    let bound = -5.0;
    assert!(
        max_err.log2() < bound,
        "encapsulated PaCo recovery: max coefficient error log2={:.1} (bound {bound:.1})",
        max_err.log2()
    );
}
