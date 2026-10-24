//! Tests for logarithmic functions.

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

use crate::log::{CKKSLogOps, Log1pPlan, Log2Plan, Log10Plan, LogOptions, LogPlan};

use super::helpers::{assert_precision_bits, params_for};

pub fn test_log_family<BE, F, E>(
    base: CKKSTestParams,
    module: &Module<BE>,
    host_module: &Module<HostBytesBackend>,
) where
    BE: TestContextBackend,
    Module<BE>: TestContextModule<BE> + CKKSLogOps<BE>,
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
    let options = LogOptions::default();
    let half = F::from_f64(0.5).unwrap();
    let one = F::one();
    let host_log = LogPlan::from_precision(
        half,
        one,
        base.base2k.into(),
        coeff_meta,
        options,
        host_module,
    )
    .expect("LogPlan::from_precision");
    let host_log2 = Log2Plan::from_precision(
        half,
        one,
        base.base2k.into(),
        coeff_meta,
        options,
        host_module,
    )
    .expect("Log2Plan::from_precision");
    let host_log10 = Log10Plan::from_precision(
        half,
        one,
        base.base2k.into(),
        coeff_meta,
        options,
        host_module,
    )
    .expect("Log10Plan::from_precision");
    let host_log1p = Log1pPlan::from_precision(
        -half,
        half,
        base.base2k.into(),
        coeff_meta,
        options,
        host_module,
    )
    .expect("Log1pPlan::from_precision");
    let max_consumed = host_log
        .consumed_bits(log_delta)
        .max(host_log2.consumed_bits(log_delta))
        .max(host_log10.consumed_bits(log_delta))
        .max(host_log1p.consumed_bits(log_delta));
    let log = host_log.map_plaintexts(|pt| upload_pt(module, pt));
    let log2 = host_log2.map_plaintexts(|pt| upload_pt(module, pt));
    let log10 = host_log10.map_plaintexts(|pt| upload_pt(module, pt));
    let log1p = host_log1p.map_plaintexts(|pt| upload_pt(module, pt));
    let params = params_for(&base, max_consumed);

    let slots = params.n / 2;
    let encoder = ReferenceEncoder::<E>::new::<F>(slots).unwrap();
    let im = vec![F::zero(); slots];
    let (sk_raw, sk) = gen_sk_with_raw(&params, module, host_module, [0u8; 32]);
    let mut sizing = module.ckks_ciphertext_alloc(params.base2k.into(), params.k.into());
    sizing.set_meta(params.prec().meta);
    let size = module
        .ckks_log_tmp_bytes(&sizing, &params.tsk_layout(), &log)
        .max(module.ckks_log2_tmp_bytes(&sizing, &params.tsk_layout(), &log2))
        .max(module.ckks_log10_tmp_bytes(&sizing, &params.tsk_layout(), &log10))
        .max(module.ckks_log1p_tmp_bytes(&sizing, &params.tsk_layout(), &log1p));
    let mut scratch = ScratchOwned::<BE>::alloc(size);
    let tsk = gen_tsk(&params, module, &sk_raw, &mut scratch.borrow());

    for (op, label) in [(0, "log"), (1, "log2"), (2, "log10"), (3, "log1p")] {
        let mut source = Source::new([0xb1u8 + op as u8; 32]);
        let mut x: Vec<F> = if op < 3 {
            (0..slots)
                .map(|_| F::from_f64(source.next_f64(0.5, 1.0)).unwrap())
                .collect()
        } else {
            (0..slots)
                .map(|_| F::from_f64(source.next_f64(-0.5, 0.5)).unwrap())
                .collect()
        };
        x[0] = if op < 3 { half } else { -half };
        x[1] = if op < 3 { one } else { half };
        if op == 3 {
            x[2] = F::zero();
            x[3] = F::from_f64(1e-4).unwrap();
        }
        let want: Vec<F> = match op {
            0 => x.iter().map(|&v| v.ln()).collect(),
            1 => x.iter().map(|&v| v.log2()).collect(),
            2 => x.iter().map(|&v| v.log10()).collect(),
            _ => x.iter().map(|&v| v.ln_1p()).collect(),
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
        let mut res = module.ckks_ciphertext_alloc(params.base2k.into(), params.k.into());
        let consumed = match op {
            0 => {
                module
                    .ckks_log(&mut res, &input, &log, &tsk, &mut scratch.borrow())
                    .expect("ckks_log");
                log.consumed_bits(log_delta)
            }
            1 => {
                module
                    .ckks_log2(&mut res, &input, &log2, &tsk, &mut scratch.borrow())
                    .expect("ckks_log2");
                log2.consumed_bits(log_delta)
            }
            2 => {
                module
                    .ckks_log10(&mut res, &input, &log10, &tsk, &mut scratch.borrow())
                    .expect("ckks_log10");
                log10.consumed_bits(log_delta)
            }
            _ => {
                module
                    .ckks_log1p(&mut res, &input, &log1p, &tsk, &mut scratch.borrow())
                    .expect("ckks_log1p");
                log1p.consumed_bits(log_delta)
            }
        };
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
