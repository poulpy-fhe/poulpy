//! Shared function benchmarks.

use std::hint::black_box;

use criterion::{BatchSize, Criterion, Throughput};
use poulpy_core::layouts::{
    GGLWEInfos, GLWETensorKeyPrepared, GLWEToBackendMut, GLWEToBackendRef, LWEInfos, SetBSGSMeta,
    prepared::GLWETensorKeyPreparedToBackendRef,
};
use poulpy_hal::{
    api::{NegacyclicFFT, NegacyclicFFTNew, ScratchOwnedAlloc, ScratchOwnedBorrow},
    layouts::{HostBytesBackend, Module, ScratchOwned},
};

use poulpy_ckks::{
    CKKSCtBounds, CKKSMeta, SetCKKSInfos,
    layouts::{CKKSCiphertext, CKKSModuleAlloc, CKKSPlaintext, CKKSPlaintextVecHostCodec},
    test_suite::{
        CKKSTestParams, NTT4X30_PARAMS_F64,
        helpers::{
            TestContextBackend, TestContextModule, ckks_encrypt_with_prec, ckks_spec,
            gen_sk_with_raw, gen_tsk, upload_pt,
        },
        reference_encoder::ReferenceEncoder,
    },
};

use poulpy_libm::{
    exp::{CKKSExpOps, ExpOptions, ExpPlan},
    log::{CKKSLogOps, LogOptions, LogPlan},
    pow::{CKKSPowOps, PowOptions, PowPlan},
};

fn params_for(base: &CKKSTestParams, consumed: usize) -> CKKSTestParams {
    let log_delta = base.prec_meta.log_delta;
    let k = (consumed + 2 * log_delta + 2 * base.base2k).next_multiple_of(base.dsize * base.base2k);
    CKKSTestParams {
        n: base.n,
        base2k: base.base2k,
        k,
        prec_meta: CKKSMeta {
            log_sparsity: 0,
            log_delta,
            slots: Default::default(),
        },
        prec_log_budget: k - log_delta,
        hw: base.hw,
        dsize: base.dsize,
        rank: base.rank,
    }
}

pub fn bench_functions<BE, E>(c: &mut Criterion, backend: &str)
where
    BE: TestContextBackend,
    Module<BE>: TestContextModule<BE> + CKKSExpOps<BE> + CKKSLogOps<BE> + CKKSPowOps<BE>,
    CKKSCiphertext<BE::OwnedBuf, BE::ZnxWord>:
        GLWEToBackendMut<BE> + GLWEToBackendRef<BE> + CKKSCtBounds + SetCKKSInfos + SetBSGSMeta,
    CKKSPlaintext<BE::OwnedBuf, BE::ZnxWord>:
        GLWEToBackendRef<BE> + LWEInfos + poulpy_core::layouts::IntPolyInfos,
    CKKSPlaintext<Vec<u8>, i64>: CKKSPlaintextVecHostCodec<f64>,
    GLWETensorKeyPrepared<BE::OwnedBuf, BE>: GLWETensorKeyPreparedToBackendRef<BE> + GGLWEInfos,
    E: NegacyclicFFT<f64> + NegacyclicFFTNew<f64>,
{
    let base = NTT4X30_PARAMS_F64;
    let log_delta = base.prec_meta.log_delta;
    let module = Module::<BE>::new(base.n as u64);
    let host_module = Module::<HostBytesBackend>::new(base.n as u64);
    let coeff_meta = ckks_spec(base.n, base.base2k, log_delta, base.base2k);
    let host_exp = ExpPlan::from_precision(
        0.5f64,
        1.5,
        base.base2k.into(),
        coeff_meta,
        ExpOptions::default(),
        &host_module,
    )
    .expect("ExpPlan::from_precision");
    let host_log = LogPlan::from_precision(
        0.5f64,
        1.5,
        base.base2k.into(),
        coeff_meta,
        LogOptions::default(),
        &host_module,
    )
    .expect("LogPlan::from_precision");
    let host_pow = PowPlan::from_precision(
        0.5f64,
        1.5,
        -1.0,
        1.0,
        base.base2k.into(),
        coeff_meta,
        PowOptions::default(),
        &host_module,
    )
    .expect("PowPlan::from_precision");
    let consumed = host_exp
        .consumed_bits(log_delta)
        .max(host_log.consumed_bits(log_delta))
        .max(host_pow.consumed_bits(log_delta, log_delta));
    let exp = host_exp.map_plaintexts(|pt| upload_pt(&module, pt));
    let log = host_log.map_plaintexts(|pt| upload_pt(&module, pt));
    let pow = host_pow.map_plaintexts(|pt| upload_pt(&module, pt));
    let params = params_for(&base, consumed);
    let slots = params.n / 2;
    let encoder = ReferenceEncoder::<E>::new::<f64>(slots).expect("ReferenceEncoder::new");

    let (sk_raw, sk) = gen_sk_with_raw(&params, &module, &host_module, [0xd1; 32]);
    let mut sizing = module.ckks_ciphertext_alloc(params.base2k.into(), params.k.into());
    sizing.set_meta(params.prec().meta);
    let scratch_bytes = module
        .ckks_exp_tmp_bytes(&sizing, &params.tsk_layout(), &exp)
        .max(module.ckks_log_tmp_bytes(&sizing, &params.tsk_layout(), &log))
        .max(module.ckks_pow_tmp_bytes(&sizing, &params.tsk_layout(), &pow));
    let mut scratch = ScratchOwned::<BE>::alloc(scratch_bytes);
    let tsk: GLWETensorKeyPrepared<_, _> =
        gen_tsk(&params, &module, &sk_raw, &mut scratch.borrow());
    let values: Vec<f64> = (0..slots).map(|i| 0.5 + i as f64 / slots as f64).collect();
    let exponents: Vec<f64> = (0..slots)
        .map(|i| -1.0 + 2.0 * i as f64 / slots as f64)
        .collect();
    let zeros = vec![0.0; slots];
    let input: CKKSCiphertext<_, BE::ZnxWord> = ckks_encrypt_with_prec(
        &params,
        &module,
        &host_module,
        &encoder,
        &sk,
        params.k,
        &values,
        &zeros,
        params.prec(),
        &mut scratch.borrow(),
    );
    let exponent: CKKSCiphertext<_, BE::ZnxWord> = ckks_encrypt_with_prec(
        &params,
        &module,
        &host_module,
        &encoder,
        &sk,
        params.k,
        &exponents,
        &zeros,
        params.prec(),
        &mut scratch.borrow(),
    );

    let mut group = c.benchmark_group(format!("functions/{backend}"));
    group.throughput(Throughput::Elements(slots as u64));
    group.bench_function("exp", |b| {
        b.iter_batched(
            || module.ckks_ciphertext_alloc(params.base2k.into(), params.k.into()),
            |mut res| {
                module
                    .ckks_exp(
                        black_box(&mut res),
                        black_box(&input),
                        black_box(&exp),
                        &tsk,
                        &mut scratch.borrow(),
                    )
                    .expect("ckks_exp");
                black_box(res)
            },
            BatchSize::SmallInput,
        )
    });
    group.bench_function("log", |b| {
        b.iter_batched(
            || module.ckks_ciphertext_alloc(params.base2k.into(), params.k.into()),
            |mut res| {
                module
                    .ckks_log(
                        black_box(&mut res),
                        black_box(&input),
                        black_box(&log),
                        &tsk,
                        &mut scratch.borrow(),
                    )
                    .expect("ckks_log");
                black_box(res)
            },
            BatchSize::SmallInput,
        )
    });
    group.bench_function("pow", |b| {
        b.iter_batched(
            || module.ckks_ciphertext_alloc(params.base2k.into(), params.k.into()),
            |mut res| {
                module
                    .ckks_pow(
                        black_box(&mut res),
                        black_box(&input),
                        black_box(&exponent),
                        black_box(&pow),
                        &tsk,
                        &mut scratch.borrow(),
                    )
                    .expect("ckks_pow");
                black_box(res)
            },
            BatchSize::SmallInput,
        )
    });
    group.finish();
}
