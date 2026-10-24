//! Tests for direct approximation evaluation and precision tuning.

use poulpy_core::layouts::{
    GGLWEInfos, GLWETensorKeyPrepared, GLWEToBackendMut, GLWEToBackendRef, LWEInfos, SetBSGSMeta,
    prepared::GLWETensorKeyPreparedToBackendRef,
};
use poulpy_hal::{
    api::{NegacyclicFFT, NegacyclicFFTNew, ScratchOwnedAlloc, ScratchOwnedBorrow},
    layouts::{HostBytesBackend, Module, ScratchOwned},
};

use poulpy_ckks::{
    CKKSCtBounds, CKKSInfos, SetCKKSInfos,
    layouts::{CKKSCiphertext, CKKSModuleAlloc, CKKSPlaintext, CKKSPlaintextVecHostCodec},
    polynomial::{Polynomial, SplitStrategy},
    test_suite::reference_encoder::ReferenceEncoder,
    test_suite::{
        CKKSTestParams,
        helpers::{
            TestContextBackend, TestContextModule, TestScalar, ckks_decrypt_decode,
            ckks_encrypt_with_prec, ckks_spec, gen_sk_with_raw, gen_tsk, upload_pt,
        },
    },
};

use crate::{
    approximation::{
        CKKSApproximationOps, Parity, PolynomialApproximation, RemezOptions, degree_for_precision,
        minimax_with,
    },
    trig::{CKKSTrigOps, SinPlan, TrigOptions},
};

use super::helpers::{assert_error, assert_precision_bits, params_for, sample_interval};

pub fn test_approximation<BE, F, E>(
    base: CKKSTestParams,
    module: &Module<BE>,
    host_module: &Module<HostBytesBackend>,
) where
    BE: TestContextBackend,
    Module<BE>: TestContextModule<BE> + CKKSApproximationOps<BE>,
    CKKSCiphertext<BE::OwnedBuf, BE::ZnxWord>:
        GLWEToBackendMut<BE> + GLWEToBackendRef<BE> + CKKSCtBounds + SetCKKSInfos + SetBSGSMeta,
    CKKSPlaintext<BE::OwnedBuf, BE::ZnxWord>:
        GLWEToBackendRef<BE> + LWEInfos + poulpy_core::layouts::IntPolyInfos,
    CKKSPlaintext<Vec<u8>, i64>: CKKSPlaintextVecHostCodec<F>,
    GLWETensorKeyPrepared<BE::OwnedBuf, BE>: GLWETensorKeyPreparedToBackendRef<BE> + GGLWEInfos,
    F: TestScalar,
    E: NegacyclicFFT<F> + NegacyclicFFTNew<F>,
{
    let input_log_delta = base.prec_meta.log_delta;
    let coeff_log_delta = 20;
    let coeff_meta = ckks_spec(base.n, base.base2k, coeff_log_delta, base.base2k);
    let lo = F::from_f64(0.5).unwrap();
    let hi = F::from_f64(1.5).unwrap();
    assert_error(
        minimax_with(
            |x: F| x * x,
            lo,
            hi,
            2,
            Parity::Full,
            RemezOptions {
                max_iters: 0,
                ..RemezOptions::default()
            },
        ),
        "minimax: max_iters must be positive",
    );
    assert_error(
        degree_for_precision(
            |x: F| x * x,
            lo,
            hi,
            Parity::Full,
            0.0,
            15,
            SplitStrategy::MinDepth,
        ),
        "degree_for_precision: target_bits must be positive and finite",
    );
    let poly = Polynomial::chebyshev_interpolate(2, lo, hi, |x| x * x).unwrap();
    let host_plan = PolynomialApproximation::from_polynomial(
        &poly,
        base.base2k.into(),
        coeff_meta,
        SplitStrategy::MinDepth,
        host_module,
    )
    .expect("PolynomialApproximation::from_polynomial");
    assert_eq!(host_plan.interval(), (0.5, 1.5));
    assert_eq!(host_plan.degree(), 2);
    assert!(host_plan.depth() > 0);
    assert_eq!(host_plan.coeff_log_delta, coeff_log_delta);

    let consumed = host_plan.consumed_bits(input_log_delta);
    let plan = host_plan.map_plaintexts(|pt| upload_pt(module, pt));
    let params = params_for(&base, consumed);
    let slots = params.n / 2;
    let encoder = ReferenceEncoder::<E>::new::<F>(slots).unwrap();
    let x = sample_interval::<F>(slots, 0.5, 1.5, 0x61);
    let im = vec![F::zero(); slots];
    let (sk_raw, sk) = gen_sk_with_raw(&params, module, host_module, [0x62; 32]);
    let mut sizing = module.ckks_ciphertext_alloc(params.base2k.into(), params.k.into());
    sizing.set_meta(params.prec().meta);
    let size = module.ckks_approximation_tmp_bytes(&sizing, &sizing, &params.tsk_layout(), &plan);
    let mut scratch = ScratchOwned::<BE>::alloc(size);
    let tsk = gen_tsk(&params, module, &sk_raw, &mut scratch.borrow());
    let input = ckks_encrypt_with_prec(
        &params,
        module,
        host_module,
        &encoder,
        &sk,
        params.k,
        &x,
        &im,
        params.prec(),
        &mut scratch.borrow(),
    );
    let mut res = module.ckks_ciphertext_alloc(params.base2k.into(), params.k.into());
    module
        .ckks_eval_approximation(&mut res, &input, &plan, &tsk, &mut scratch.borrow())
        .expect("ckks_eval_approximation");
    assert_eq!(input.log_budget() - res.log_budget(), consumed);

    let (re_out, _) = ckks_decrypt_decode::<BE, F, E>(
        &params,
        module,
        &encoder,
        &res,
        &sk,
        &mut scratch.borrow(),
    );
    let want: Vec<F> = x.iter().map(|&v| v * v).collect();
    assert_precision_bits("approximation", &re_out, &want, 16.0, params.n);
}

pub fn test_precision_tuning<BE, F, E>(
    base: CKKSTestParams,
    module: &Module<BE>,
    host_module: &Module<HostBytesBackend>,
) where
    BE: TestContextBackend,
    Module<BE>: TestContextModule<BE> + CKKSTrigOps<BE>,
    CKKSCiphertext<BE::OwnedBuf, BE::ZnxWord>:
        GLWEToBackendMut<BE> + GLWEToBackendRef<BE> + CKKSCtBounds + SetCKKSInfos,
    CKKSPlaintext<BE::OwnedBuf, BE::ZnxWord>:
        GLWEToBackendRef<BE> + LWEInfos + poulpy_core::layouts::IntPolyInfos,
    CKKSPlaintext<Vec<u8>, i64>: CKKSPlaintextVecHostCodec<F>,
    GLWETensorKeyPrepared<BE::OwnedBuf, BE>: GLWETensorKeyPreparedToBackendRef<BE> + GGLWEInfos,
    F: TestScalar,
    E: NegacyclicFFT<F> + NegacyclicFFTNew<F>,
{
    let input_log_delta = base.prec_meta.log_delta;
    let low_target = 12.0;
    let high_target = 28.0;
    let low_meta = ckks_spec(base.n, base.base2k, 20, base.base2k);
    let high_meta = ckks_spec(base.n, base.base2k, 52, base.base2k);
    let low_host = SinPlan::from_precision(
        -F::one(),
        F::one(),
        base.base2k.into(),
        low_meta,
        TrigOptions {
            target_bits: low_target,
            max_degree: 31,
            strategy: SplitStrategy::MinDepth,
        },
        host_module,
    )
    .expect("low-precision SinPlan");
    let scaled_host = SinPlan::from_precision(
        -F::one(),
        F::one(),
        base.base2k.into(),
        high_meta,
        TrigOptions {
            target_bits: low_target,
            max_degree: 31,
            strategy: SplitStrategy::MinDepth,
        },
        host_module,
    )
    .expect("high-scale SinPlan");
    let high_host = SinPlan::from_precision(
        -F::one(),
        F::one(),
        base.base2k.into(),
        high_meta,
        TrigOptions {
            target_bits: high_target,
            max_degree: 31,
            strategy: SplitStrategy::MinDepth,
        },
        host_module,
    )
    .expect("high-precision SinPlan");
    let low_consumed = low_host.consumed_bits(input_log_delta);
    let scaled_consumed = scaled_host.consumed_bits(input_log_delta);
    let high_consumed = high_host.consumed_bits(input_log_delta);
    assert_eq!(low_host.degree(), scaled_host.degree());
    if input_log_delta < high_meta.meta.log_delta {
        assert!(
            low_consumed < scaled_consumed,
            "low/scaled costs={low_consumed}/{scaled_consumed}"
        );
    } else {
        assert_eq!(
            low_consumed, scaled_consumed,
            "coefficient scale at or below the ciphertext scale should be off the critical path"
        );
    }
    assert!(scaled_host.degree() < high_host.degree());
    assert!(scaled_consumed <= high_consumed);
    assert!(
        low_consumed < high_consumed,
        "low degree/cost={}/{low_consumed}, high degree/cost={}/{high_consumed}",
        low_host.degree(),
        high_host.degree()
    );
    println!(
        "precision tuning logN={}: low(target={low_target}, coeff_delta=20, degree={}, depth={}, consumed={low_consumed}); scaled(target={low_target}, coeff_delta=52, degree={}, depth={}, consumed={scaled_consumed}); high(target={high_target}, coeff_delta=52, degree={}, depth={}, consumed={high_consumed})",
        base.log_n(),
        low_host.degree(),
        low_host.depth(),
        scaled_host.degree(),
        scaled_host.depth(),
        high_host.degree(),
        high_host.depth(),
    );

    let low = low_host.map_plaintexts(|pt| upload_pt(module, pt));
    let scaled = scaled_host.map_plaintexts(|pt| upload_pt(module, pt));
    let high = high_host.map_plaintexts(|pt| upload_pt(module, pt));
    let params = params_for(&base, high_consumed);
    let slots = params.n / 2;
    let encoder = ReferenceEncoder::<E>::new::<F>(slots).unwrap();
    let x = sample_interval::<F>(slots, -1.0, 1.0, 0x63);
    let im = vec![F::zero(); slots];
    let (sk_raw, sk) = gen_sk_with_raw(&params, module, host_module, [0x64; 32]);
    let mut sizing = module.ckks_ciphertext_alloc(params.base2k.into(), params.k.into());
    sizing.set_meta(params.prec().meta);
    let size = module
        .ckks_sin_tmp_bytes(&sizing, &params.tsk_layout(), &low)
        .max(module.ckks_sin_tmp_bytes(&sizing, &params.tsk_layout(), &scaled))
        .max(module.ckks_sin_tmp_bytes(&sizing, &params.tsk_layout(), &high));
    let mut scratch = ScratchOwned::<BE>::alloc(size);
    let tsk = gen_tsk(&params, module, &sk_raw, &mut scratch.borrow());
    let input = ckks_encrypt_with_prec(
        &params,
        module,
        host_module,
        &encoder,
        &sk,
        params.k,
        &x,
        &im,
        params.prec(),
        &mut scratch.borrow(),
    );
    let want: Vec<F> = x.iter().map(|&v| v.sin()).collect();

    for (label, plan, consumed, target) in [
        ("precision/low", &low, low_consumed, low_target),
        ("precision/scaled", &scaled, scaled_consumed, low_target),
        ("precision/high", &high, high_consumed, high_target),
    ] {
        let mut res = module.ckks_ciphertext_alloc(params.base2k.into(), params.k.into());
        module
            .ckks_sin(&mut res, &input, plan, &tsk, &mut scratch.borrow())
            .unwrap_or_else(|e| panic!("{label}: {e}"));
        assert_eq!(input.log_budget() - res.log_budget(), consumed);
        let (re_out, _) = ckks_decrypt_decode::<BE, F, E>(
            &params,
            module,
            &encoder,
            &res,
            &sk,
            &mut scratch.borrow(),
        );
        let stats =
            poulpy_ckks::test_suite::helpers::precision_stats(&re_out, &want, res.log_delta());
        println!(
            "{label}: avg={:.2}, min={:.2} precision bits",
            stats.avg_log2_prec, stats.min_log2_prec
        );
        assert_precision_bits(label, &re_out, &want, target, params.n);
    }
}
