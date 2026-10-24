//! Tests for exponential functions.

use poulpy_core::layouts::{
    GGLWEInfos, GLWETensorKeyPrepared, GLWEToBackendMut, GLWEToBackendRef, LWEInfos,
    prepared::GLWETensorKeyPreparedToBackendRef,
};
use poulpy_hal::{
    api::{NegacyclicFFT, NegacyclicFFTNew, ScratchOwnedAlloc, ScratchOwnedBorrow},
    layouts::{HostBytesBackend, Module, ScratchOwned},
    source::Source,
};

use poulpy_ckks::{
    CKKSCtBounds, CKKSInfos, SetCKKSInfos,
    layouts::{CKKSCiphertext, CKKSModuleAlloc, CKKSPlaintext, CKKSPlaintextVecHostCodec},
    test_suite::reference_encoder::ReferenceEncoder,
    test_suite::{
        CKKSTestParams,
        helpers::{
            TestContextBackend, TestContextModule, TestScalar, ckks_decrypt_decode,
            ckks_encrypt_with_prec, ckks_spec, gen_sk_with_raw, gen_tsk, upload_pt,
        },
    },
};

use crate::exp::{CKKSExpOps, Exp2Plan, Exp10Plan, ExpOptions, ExpPlan, Expm1Plan};

use super::helpers::{assert_precision_bits, params_for};

pub fn test_exp<BE, F, E>(
    base: CKKSTestParams,
    module: &Module<BE>,
    host_module: &Module<HostBytesBackend>,
) where
    BE: TestContextBackend,
    Module<BE>: TestContextModule<BE> + CKKSExpOps<BE>,
    CKKSCiphertext<BE::OwnedBuf, BE::ZnxWord>:
        GLWEToBackendMut<BE> + GLWEToBackendRef<BE> + CKKSCtBounds + SetCKKSInfos,
    CKKSPlaintext<BE::OwnedBuf, BE::ZnxWord>:
        GLWEToBackendRef<BE> + LWEInfos + poulpy_core::layouts::IntPolyInfos,
    CKKSPlaintext<Vec<u8>, i64>: CKKSPlaintextVecHostCodec<F>,
    GLWETensorKeyPrepared<BE::OwnedBuf, BE>: GLWETensorKeyPreparedToBackendRef<BE> + GGLWEInfos,
    F: TestScalar,
    E: NegacyclicFFT<F> + NegacyclicFFTNew<F>,
{
    let log_delta = base.prec_meta.log_delta;
    let coeff_meta = ckks_spec(base.n, base.base2k, log_delta, base.base2k);
    let options = ExpOptions {
        target_bits: 20.0,
        ..ExpOptions::default()
    };
    let mut host_plans = Vec::new();
    for (label, hi) in [("general", 3.0), ("pow2", 4.0)] {
        let plan = ExpPlan::from_precision::<F>(
            F::zero(),
            F::from_f64(hi).unwrap(),
            base.base2k.into(),
            coeff_meta,
            options,
            host_module,
        )
        .expect("ExpPlan::from_precision");
        host_plans.push((label, hi, plan));
    }
    assert_eq!(host_plans[0].2.approximation.scale_pow2, None);
    assert_eq!(host_plans[1].2.approximation.scale_pow2, Some(-1));
    let max_consumed = host_plans
        .iter()
        .map(|(_, _, plan)| plan.consumed_bits(log_delta))
        .max()
        .unwrap();
    let plans: Vec<_> = host_plans
        .into_iter()
        .map(|(label, hi, plan)| (label, hi, plan.map_plaintexts(|pt| upload_pt(module, pt))))
        .collect();
    let params = params_for(&base, max_consumed);

    let slots = params.n / 2;
    let encoder = ReferenceEncoder::<E>::new::<F>(slots).unwrap();
    let im = vec![F::zero(); slots];

    let (sk_raw, sk) = gen_sk_with_raw(&params, module, host_module, [0u8; 32]);
    let mut sizing = module.ckks_ciphertext_alloc(params.base2k.into(), params.k.into());
    sizing.set_meta(params.prec().meta);
    let size = plans
        .iter()
        .map(|(_, _, plan)| module.ckks_exp_tmp_bytes(&sizing, &params.tsk_layout(), plan))
        .max()
        .unwrap();
    let mut scratch = ScratchOwned::<BE>::alloc(size);
    let tsk = gen_tsk(&params, module, &sk_raw, &mut scratch.borrow());

    for (case, (label, hi, plan)) in plans.iter().enumerate() {
        let mut source = Source::new([0x91u8 + case as u8; 32]);
        let mut x: Vec<F> = (0..slots)
            .map(|_| F::from_f64(source.next_f64(0.0, *hi)).unwrap())
            .collect();
        x[0] = F::zero();
        x[1] = F::from_f64(*hi).unwrap();
        let want: Vec<F> = x.iter().map(|&v| v.exp()).collect();
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

        let input_budget = input.log_budget();
        let mut res = module.ckks_ciphertext_alloc(params.base2k.into(), params.k.into());
        module
            .ckks_exp(&mut res, &input, plan, &tsk, &mut scratch.borrow())
            .expect("ckks_exp");
        assert_eq!(
            res.log_delta(),
            plan.approximation.output_log_delta(log_delta),
            "{label}"
        );
        assert_eq!(
            input_budget - res.log_budget(),
            plan.consumed_bits(log_delta),
            "{label}"
        );

        let (re_out, _) = ckks_decrypt_decode::<BE, F, E>(
            &params,
            module,
            &encoder,
            &res,
            &sk,
            &mut scratch.borrow(),
        );
        assert_precision_bits(
            &format!("exp/{label}"),
            &re_out,
            &want,
            options.target_bits,
            params.n,
        );
    }
}

pub fn test_exp2_expm1<BE, F, E>(
    base: CKKSTestParams,
    module: &Module<BE>,
    host_module: &Module<HostBytesBackend>,
) where
    BE: TestContextBackend,
    Module<BE>: TestContextModule<BE> + CKKSExpOps<BE>,
    CKKSCiphertext<BE::OwnedBuf, BE::ZnxWord>:
        GLWEToBackendMut<BE> + GLWEToBackendRef<BE> + CKKSCtBounds + SetCKKSInfos,
    CKKSPlaintext<BE::OwnedBuf, BE::ZnxWord>:
        GLWEToBackendRef<BE> + LWEInfos + poulpy_core::layouts::IntPolyInfos,
    CKKSPlaintext<Vec<u8>, i64>: CKKSPlaintextVecHostCodec<F>,
    GLWETensorKeyPrepared<BE::OwnedBuf, BE>: GLWETensorKeyPreparedToBackendRef<BE> + GGLWEInfos,
    F: TestScalar,
    E: NegacyclicFFT<F> + NegacyclicFFTNew<F>,
{
    let log_delta = base.prec_meta.log_delta;
    let coeff_meta = ckks_spec(base.n, base.base2k, log_delta, base.base2k);
    let options = ExpOptions {
        target_bits: 20.0,
        ..ExpOptions::default()
    };
    let lo = F::from_f64(-4.0).unwrap();
    let hi = F::from_f64(4.0).unwrap();
    let host_exp2 =
        Exp2Plan::from_precision(lo, hi, base.base2k.into(), coeff_meta, options, host_module)
            .expect("Exp2Plan::from_precision");
    let host_expm1 =
        Expm1Plan::from_precision(lo, hi, base.base2k.into(), coeff_meta, options, host_module)
            .expect("Expm1Plan::from_precision");
    let max_consumed = host_exp2
        .consumed_bits(log_delta)
        .max(host_expm1.consumed_bits(log_delta));
    let exp2 = host_exp2.map_plaintexts(|pt| upload_pt(module, pt));
    let expm1 = host_expm1.map_plaintexts(|pt| upload_pt(module, pt));
    let params = params_for(&base, max_consumed);

    let slots = params.n / 2;
    let encoder = ReferenceEncoder::<E>::new::<F>(slots).unwrap();
    let mut source = Source::new([0xa1u8; 32]);
    let mut x: Vec<F> = (0..slots)
        .map(|_| F::from_f64(source.next_f64(-4.0, 4.0)).unwrap())
        .collect();
    x[0] = lo;
    x[1] = hi;
    x[2] = F::zero();
    x[3] = F::from_f64(1e-4).unwrap();
    let im = vec![F::zero(); slots];

    let (sk_raw, sk) = gen_sk_with_raw(&params, module, host_module, [0u8; 32]);
    let mut sizing = module.ckks_ciphertext_alloc(params.base2k.into(), params.k.into());
    sizing.set_meta(params.prec().meta);
    let size = module
        .ckks_exp2_tmp_bytes(&sizing, &params.tsk_layout(), &exp2)
        .max(module.ckks_expm1_tmp_bytes(&sizing, &params.tsk_layout(), &expm1));
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

    for (op, label) in [(0, "exp2"), (1, "expm1")] {
        let want: Vec<F> = if op == 0 {
            x.iter().map(|&v| v.exp2()).collect()
        } else {
            x.iter().map(|&v| v.exp_m1()).collect()
        };
        let consumed = if op == 0 {
            exp2.consumed_bits(log_delta)
        } else {
            expm1.consumed_bits(log_delta)
        };
        let mut res = module.ckks_ciphertext_alloc(params.base2k.into(), params.k.into());
        if op == 0 {
            module.ckks_exp2(&mut res, &input, &exp2, &tsk, &mut scratch.borrow())
        } else {
            module.ckks_expm1(&mut res, &input, &expm1, &tsk, &mut scratch.borrow())
        }
        .unwrap_or_else(|e| panic!("ckks_{label}: {e}"));
        assert_eq!(input.log_budget() - res.log_budget(), consumed, "{label}");

        let (re_out, _) = ckks_decrypt_decode::<BE, F, E>(
            &params,
            module,
            &encoder,
            &res,
            &sk,
            &mut scratch.borrow(),
        );
        assert_precision_bits(label, &re_out, &want, options.target_bits, params.n);
    }
}

pub fn test_exp10<BE, F, E>(
    base: CKKSTestParams,
    module: &Module<BE>,
    host_module: &Module<HostBytesBackend>,
) where
    BE: TestContextBackend,
    Module<BE>: TestContextModule<BE> + CKKSExpOps<BE>,
    CKKSCiphertext<BE::OwnedBuf, BE::ZnxWord>:
        GLWEToBackendMut<BE> + GLWEToBackendRef<BE> + CKKSCtBounds + SetCKKSInfos,
    CKKSPlaintext<BE::OwnedBuf, BE::ZnxWord>:
        GLWEToBackendRef<BE> + LWEInfos + poulpy_core::layouts::IntPolyInfos,
    CKKSPlaintext<Vec<u8>, i64>: CKKSPlaintextVecHostCodec<F>,
    GLWETensorKeyPrepared<BE::OwnedBuf, BE>: GLWETensorKeyPreparedToBackendRef<BE> + GGLWEInfos,
    F: TestScalar,
    E: NegacyclicFFT<F> + NegacyclicFFTNew<F>,
{
    let log_delta = base.prec_meta.log_delta;
    let coeff_meta = ckks_spec(base.n, base.base2k, log_delta, base.base2k);
    let options = ExpOptions::default();
    let lo = -F::one();
    let hi = F::one();
    let host_plan =
        Exp10Plan::from_precision(lo, hi, base.base2k.into(), coeff_meta, options, host_module)
            .expect("Exp10Plan::from_precision");
    let consumed = host_plan.consumed_bits(log_delta);
    let plan = host_plan.map_plaintexts(|pt| upload_pt(module, pt));
    let params = params_for(&base, consumed);
    let slots = params.n / 2;
    let encoder = ReferenceEncoder::<E>::new::<F>(slots).unwrap();
    let mut source = Source::new([0xa5u8; 32]);
    let mut x: Vec<F> = (0..slots)
        .map(|_| F::from_f64(source.next_f64(-1.0, 1.0)).unwrap())
        .collect();
    x[0] = lo;
    x[1] = F::zero();
    x[2] = hi;
    let im = vec![F::zero(); slots];
    let (sk_raw, sk) = gen_sk_with_raw(&params, module, host_module, [0xa6u8; 32]);
    let mut sizing = module.ckks_ciphertext_alloc(params.base2k.into(), params.k.into());
    sizing.set_meta(params.prec().meta);
    let size = module.ckks_exp10_tmp_bytes(&sizing, &params.tsk_layout(), &plan);
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
        .ckks_exp10(&mut res, &input, &plan, &tsk, &mut scratch.borrow())
        .expect("ckks_exp10");
    assert_eq!(input.log_budget() - res.log_budget(), consumed);
    let (re_out, _) = ckks_decrypt_decode::<BE, F, E>(
        &params,
        module,
        &encoder,
        &res,
        &sk,
        &mut scratch.borrow(),
    );
    let ten = F::from_f64(10.0).unwrap();
    let want: Vec<F> = x.iter().map(|&v| ten.powf(v)).collect();
    assert_precision_bits("exp10", &re_out, &want, options.target_bits, params.n);
}
