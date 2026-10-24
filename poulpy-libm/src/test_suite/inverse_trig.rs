//! Tests for inverse trigonometric functions.

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
            ckks_encrypt_with_prec, ckks_spec, gen_atk, gen_sk_with_raw, gen_tsk, upload_pt,
        },
    },
};

use crate::trig::{
    AcosPlan, AsinPlan, Atan2Options, Atan2Plan, AtanPlan, CKKSAtan2Ops, CKKSInverseTrigOps,
    InverseTrigOptions,
};

use super::helpers::{assert_precision_bits, params_for};

pub fn test_inverse_trig_family<BE, F, E>(
    base: CKKSTestParams,
    module: &Module<BE>,
    host_module: &Module<HostBytesBackend>,
) where
    BE: TestContextBackend,
    Module<BE>: TestContextModule<BE> + CKKSInverseTrigOps<BE>,
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
    let options = InverseTrigOptions::default();
    let bound = F::from_f64(0.9).unwrap();
    let host_atan = AtanPlan::from_precision(
        -bound,
        bound,
        base.base2k.into(),
        coeff_meta,
        options,
        host_module,
    )
    .expect("AtanPlan::from_precision");
    let host_asin = AsinPlan::from_precision(
        -bound,
        bound,
        base.base2k.into(),
        coeff_meta,
        options,
        host_module,
    )
    .expect("AsinPlan::from_precision");
    let host_acos = AcosPlan::from_precision(
        -bound,
        bound,
        base.base2k.into(),
        coeff_meta,
        options,
        host_module,
    )
    .expect("AcosPlan::from_precision");
    let max_consumed = host_atan
        .consumed_bits(log_delta)
        .max(host_asin.consumed_bits(log_delta))
        .max(host_acos.consumed_bits(log_delta));
    let atan = host_atan.map_plaintexts(|pt| upload_pt(module, pt));
    let asin = host_asin.map_plaintexts(|pt| upload_pt(module, pt));
    let acos = host_acos.map_plaintexts(|pt| upload_pt(module, pt));
    let params = params_for(&base, max_consumed);
    let slots = params.n / 2;
    let encoder = ReferenceEncoder::<E>::new::<F>(slots).unwrap();
    let mut source = Source::new([0xb5u8; 32]);
    let mut x: Vec<F> = (0..slots)
        .map(|_| F::from_f64(source.next_f64(-0.9, 0.9)).unwrap())
        .collect();
    x[0] = -bound;
    x[1] = F::zero();
    x[2] = bound;
    let im = vec![F::zero(); slots];
    let (sk_raw, sk) = gen_sk_with_raw(&params, module, host_module, [0xb6u8; 32]);
    let mut sizing = module.ckks_ciphertext_alloc(params.base2k.into(), params.k.into());
    sizing.set_meta(params.prec().meta);
    let size = module
        .ckks_atan_tmp_bytes(&sizing, &params.tsk_layout(), &atan)
        .max(module.ckks_asin_tmp_bytes(&sizing, &params.tsk_layout(), &asin))
        .max(module.ckks_acos_tmp_bytes(&sizing, &params.tsk_layout(), &acos));
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

    for (op, label) in [(0, "atan"), (1, "asin"), (2, "acos")] {
        let (consumed, want): (usize, Vec<F>) = match op {
            0 => (
                atan.consumed_bits(log_delta),
                x.iter().map(|&v| v.atan()).collect(),
            ),
            1 => (
                asin.consumed_bits(log_delta),
                x.iter().map(|&v| v.asin()).collect(),
            ),
            _ => (
                acos.consumed_bits(log_delta),
                x.iter().map(|&v| v.acos()).collect(),
            ),
        };
        let mut res = module.ckks_ciphertext_alloc(params.base2k.into(), params.k.into());
        match op {
            0 => module.ckks_atan(&mut res, &input, &atan, &tsk, &mut scratch.borrow()),
            1 => module.ckks_asin(&mut res, &input, &asin, &tsk, &mut scratch.borrow()),
            _ => module.ckks_acos(&mut res, &input, &acos, &tsk, &mut scratch.borrow()),
        }
        .unwrap_or_else(|e| panic!("ckks_{label}: {e}"));
        assert_eq!(input.log_budget() - res.log_budget(), consumed);

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

pub fn test_atan2<BE, F, E>(
    base: CKKSTestParams,
    module: &Module<BE>,
    host_module: &Module<HostBytesBackend>,
) where
    BE: TestContextBackend,
    Module<BE>: TestContextModule<BE> + CKKSAtan2Ops<BE>,
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
    let options = Atan2Options {
        input_bound: 4.0,
        ..Atan2Options::default()
    };
    let host_plan = Atan2Plan::from_precision(
        F::from_f64(0.9).unwrap(),
        base.base2k.into(),
        coeff_meta,
        options,
        host_module,
    )
    .expect("Atan2Plan::from_precision");
    let consumed = host_plan.consumed_bits(log_delta);
    let plan = host_plan.map_plaintexts(|pt| upload_pt(module, pt));
    let params = params_for(&base, consumed);
    let slots = params.n / 2;
    let encoder = ReferenceEncoder::<E>::new::<F>(slots).unwrap();
    let x: Vec<F> = (0..slots)
        .map(|i| {
            let magnitude = 2.4 + 1.2 * (i as f64 / slots as f64);
            F::from_f64(if i % 2 == 0 { magnitude } else { -magnitude }).unwrap()
        })
        .collect();
    let y: Vec<F> = (0..slots)
        .map(|i| {
            let magnitude = 0.8 + 1.2 * (i as f64 / slots as f64);
            F::from_f64(if (i / 2) % 2 == 0 {
                magnitude
            } else {
                -magnitude
            })
            .unwrap()
        })
        .collect();
    let im = vec![F::zero(); slots];
    let (sk_raw, sk) = gen_sk_with_raw(&params, module, host_module, [0xb7u8; 32]);
    let mut sizing = module.ckks_ciphertext_alloc(params.base2k.into(), params.k.into());
    sizing.set_meta(params.prec().meta);
    let size =
        module.ckks_atan2_tmp_bytes(&sizing, &params.tsk_layout(), &params.atk_layout(), &plan);
    let mut scratch = ScratchOwned::<BE>::alloc(size);
    let tsk = gen_tsk(&params, module, &sk_raw, &mut scratch.borrow());
    let conj_key = gen_atk(&params, module, -1, &sk_raw, &mut scratch.borrow());
    let cx = ckks_encrypt_with_prec(
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
    let cy = ckks_encrypt_with_prec(
        &params,
        module,
        host_module,
        &encoder,
        &sk,
        params.k,
        &y,
        &im,
        params.prec(),
        &mut scratch.borrow(),
    );
    let mut res = module.ckks_ciphertext_alloc(params.base2k.into(), params.k.into());
    module
        .ckks_atan2(
            &mut res,
            &cy,
            &cx,
            &plan,
            &tsk,
            &conj_key,
            &mut scratch.borrow(),
        )
        .expect("ckks_atan2");
    assert_eq!(cx.log_budget() - res.log_budget(), consumed);
    let (re_out, _) = ckks_decrypt_decode::<BE, F, E>(
        &params,
        module,
        &encoder,
        &res,
        &sk,
        &mut scratch.borrow(),
    );
    let want: Vec<F> = y.iter().zip(&x).map(|(&y, &x)| y.atan2(x)).collect();
    assert_precision_bits("atan2", &re_out, &want, options.target_bits, params.n);
}
