//! Tests for hyperbolic functions.

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

use crate::hyperbolic::{
    AcoshPlan, AsinhPlan, AtanhPlan, CKKSHyperbolicOps, CoshPlan, HyperbolicOptions, SinhPlan,
    TanhPlan,
};

use super::helpers::{assert_error, assert_precision_bits, params_for};

pub fn test_hyperbolic_family<BE, F, E>(
    base: CKKSTestParams,
    module: &Module<BE>,
    host_module: &Module<HostBytesBackend>,
) where
    BE: TestContextBackend,
    Module<BE>: TestContextModule<BE> + CKKSHyperbolicOps<BE>,
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
    let options = HyperbolicOptions::default();
    let bound = F::from_f64(2.0).unwrap();
    let host_sinh = SinhPlan::from_precision(
        -bound,
        bound,
        base.base2k.into(),
        coeff_meta,
        options,
        host_module,
    )
    .expect("SinhPlan::from_precision");
    let host_cosh = CoshPlan::from_precision(
        -bound,
        bound,
        base.base2k.into(),
        coeff_meta,
        options,
        host_module,
    )
    .expect("CoshPlan::from_precision");
    let host_tanh = TanhPlan::from_precision(
        -bound,
        bound,
        base.base2k.into(),
        coeff_meta,
        options,
        host_module,
    )
    .expect("TanhPlan::from_precision");
    let max_consumed = host_sinh
        .consumed_bits(log_delta)
        .max(host_cosh.consumed_bits(log_delta))
        .max(host_tanh.consumed_bits(log_delta));
    let sinh = host_sinh.map_plaintexts(|pt| upload_pt(module, pt));
    let cosh = host_cosh.map_plaintexts(|pt| upload_pt(module, pt));
    let tanh = host_tanh.map_plaintexts(|pt| upload_pt(module, pt));
    let params = params_for(&base, max_consumed);

    let slots = params.n / 2;
    let encoder = ReferenceEncoder::<E>::new::<F>(slots).unwrap();
    let im = vec![F::zero(); slots];
    let (sk_raw, sk) = gen_sk_with_raw(&params, module, host_module, [0xd1u8; 32]);
    let mut sizing = module.ckks_ciphertext_alloc(params.base2k.into(), params.k.into());
    sizing.set_meta(params.prec().meta);
    let size = module
        .ckks_sinh_tmp_bytes(&sizing, &params.tsk_layout(), &sinh)
        .max(module.ckks_cosh_tmp_bytes(&sizing, &params.tsk_layout(), &cosh))
        .max(module.ckks_tanh_tmp_bytes(&sizing, &params.tsk_layout(), &tanh));
    let mut scratch = ScratchOwned::<BE>::alloc(size);
    let tsk = gen_tsk(&params, module, &sk_raw, &mut scratch.borrow());

    let mut source = Source::new([0xd2u8; 32]);
    let mut x: Vec<F> = (0..slots)
        .map(|_| F::from_f64(source.next_f64(-2.0, 2.0)).unwrap())
        .collect();
    x[0] = -bound;
    x[1] = F::zero();
    x[2] = bound;
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

    for (op, label) in [(0, "sinh"), (1, "cosh"), (2, "tanh")] {
        let (consumed, want): (usize, Vec<F>) = match op {
            0 => (
                sinh.consumed_bits(log_delta),
                x.iter().map(|&v| v.sinh()).collect(),
            ),
            1 => (
                cosh.consumed_bits(log_delta),
                x.iter().map(|&v| v.cosh()).collect(),
            ),
            _ => (
                tanh.consumed_bits(log_delta),
                x.iter().map(|&v| v.tanh()).collect(),
            ),
        };
        let mut res = module.ckks_ciphertext_alloc(params.base2k.into(), params.k.into());
        match op {
            0 => module.ckks_sinh(&mut res, &input, &sinh, &tsk, &mut scratch.borrow()),
            1 => module.ckks_cosh(&mut res, &input, &cosh, &tsk, &mut scratch.borrow()),
            _ => module.ckks_tanh(&mut res, &input, &tanh, &tsk, &mut scratch.borrow()),
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

pub fn test_inverse_hyperbolic<BE, F, E>(
    base: CKKSTestParams,
    module: &Module<BE>,
    host_module: &Module<HostBytesBackend>,
) where
    BE: TestContextBackend,
    Module<BE>: TestContextModule<BE> + CKKSHyperbolicOps<BE>,
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
    let options = HyperbolicOptions::default();
    let host_asinh = AsinhPlan::from_precision(
        F::from_f64(-2.0).unwrap(),
        F::from_f64(2.0).unwrap(),
        base.base2k.into(),
        coeff_meta,
        options,
        host_module,
    )
    .expect("AsinhPlan::from_precision");
    let host_acosh = AcoshPlan::from_precision(
        F::from_f64(1.25).unwrap(),
        F::from_f64(3.0).unwrap(),
        base.base2k.into(),
        coeff_meta,
        options,
        host_module,
    )
    .expect("AcoshPlan::from_precision");
    let host_atanh = AtanhPlan::from_precision(
        F::from_f64(-0.75).unwrap(),
        F::from_f64(0.75).unwrap(),
        base.base2k.into(),
        coeff_meta,
        options,
        host_module,
    )
    .expect("AtanhPlan::from_precision");
    assert_error(
        AcoshPlan::from_precision(
            F::from_f64(0.9).unwrap(),
            F::from_f64(2.0).unwrap(),
            base.base2k.into(),
            coeff_meta,
            options,
            host_module,
        ),
        "acosh: interval must start at or above 1",
    );
    assert_error(
        AtanhPlan::from_precision(
            -F::one(),
            F::from_f64(0.5).unwrap(),
            base.base2k.into(),
            coeff_meta,
            options,
            host_module,
        ),
        "atanh: interval must lie strictly inside (-1, 1)",
    );
    let max_consumed = host_asinh
        .consumed_bits(log_delta)
        .max(host_acosh.consumed_bits(log_delta))
        .max(host_atanh.consumed_bits(log_delta));
    let asinh = host_asinh.map_plaintexts(|pt| upload_pt(module, pt));
    let acosh = host_acosh.map_plaintexts(|pt| upload_pt(module, pt));
    let atanh = host_atanh.map_plaintexts(|pt| upload_pt(module, pt));
    let params = params_for(&base, max_consumed);
    let slots = params.n / 2;
    let encoder = ReferenceEncoder::<E>::new::<F>(slots).unwrap();
    let im = vec![F::zero(); slots];
    let (sk_raw, sk) = gen_sk_with_raw(&params, module, host_module, [0xd3u8; 32]);
    let mut sizing = module.ckks_ciphertext_alloc(params.base2k.into(), params.k.into());
    sizing.set_meta(params.prec().meta);
    let size = module
        .ckks_asinh_tmp_bytes(&sizing, &params.tsk_layout(), &asinh)
        .max(module.ckks_acosh_tmp_bytes(&sizing, &params.tsk_layout(), &acosh))
        .max(module.ckks_atanh_tmp_bytes(&sizing, &params.tsk_layout(), &atanh));
    let mut scratch = ScratchOwned::<BE>::alloc(size);
    let tsk = gen_tsk(&params, module, &sk_raw, &mut scratch.borrow());
    let mut source = Source::new([0xd4u8; 32]);

    for (op, label, lo, hi) in [
        (0, "asinh", -2.0, 2.0),
        (1, "acosh", 1.25, 3.0),
        (2, "atanh", -0.75, 0.75),
    ] {
        let mut x: Vec<F> = (0..slots)
            .map(|_| F::from_f64(source.next_f64(lo, hi)).unwrap())
            .collect();
        x[0] = F::from_f64(lo).unwrap();
        x[1] = F::from_f64(hi).unwrap();
        let want: Vec<F> = match op {
            0 => x.iter().map(|&v| v.asinh()).collect(),
            1 => x.iter().map(|&v| v.acosh()).collect(),
            _ => x.iter().map(|&v| v.atanh()).collect(),
        };
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
        let mut result_ct = module.ckks_ciphertext_alloc(params.base2k.into(), params.k.into());
        let (consumed, result) = match op {
            0 => (
                asinh.consumed_bits(log_delta),
                module.ckks_asinh(&mut result_ct, &input, &asinh, &tsk, &mut scratch.borrow()),
            ),
            1 => (
                acosh.consumed_bits(log_delta),
                module.ckks_acosh(&mut result_ct, &input, &acosh, &tsk, &mut scratch.borrow()),
            ),
            _ => (
                atanh.consumed_bits(log_delta),
                module.ckks_atanh(&mut result_ct, &input, &atanh, &tsk, &mut scratch.borrow()),
            ),
        };
        result.unwrap_or_else(|e| panic!("ckks_{label}: {e}"));
        assert_eq!(
            input.log_budget() - result_ct.log_budget(),
            consumed,
            "{label}"
        );
        let (re_out, _) = ckks_decrypt_decode::<BE, F, E>(
            &params,
            module,
            &encoder,
            &result_ct,
            &sk,
            &mut scratch.borrow(),
        );
        assert_precision_bits(label, &re_out, &want, options.target_bits, params.n);
    }
}
