//! Tests for power functions.

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

use crate::pow::{CKKSPowOps, PowOptions, PowPlan};

use super::helpers::{assert_precision_bits, params_for};

pub fn test_powi<BE, F, E>(
    base: CKKSTestParams,
    module: &Module<BE>,
    host_module: &Module<HostBytesBackend>,
) where
    BE: TestContextBackend,
    Module<BE>: TestContextModule<BE> + CKKSPowOps<BE>,
    CKKSCiphertext<BE::OwnedBuf, BE::ZnxWord>:
        GLWEToBackendMut<BE> + GLWEToBackendRef<BE> + CKKSCtBounds + SetCKKSInfos,
    GLWETensorKeyPrepared<BE::OwnedBuf, BE>: GLWETensorKeyPreparedToBackendRef<BE> + GGLWEInfos,
    F: TestScalar,
    E: NegacyclicFFT<F> + NegacyclicFFTNew<F>,
{
    let log_delta = base.prec_meta.log_delta;
    let params = params_for(&base, 6 * log_delta);
    let slots = params.n / 2;
    let encoder = ReferenceEncoder::<E>::new::<F>(slots).unwrap();
    let mut source = Source::new([0xc1u8; 32]);
    let mut x: Vec<F> = (0..slots)
        .map(|_| F::from_f64(source.next_f64(-0.8, 0.8)).unwrap())
        .collect();
    x[0] = F::from_f64(-0.8).unwrap();
    x[1] = F::from_f64(0.8).unwrap();
    x[2] = F::zero();
    let im = vec![F::zero(); slots];

    let (sk_raw, sk) = gen_sk_with_raw(&params, module, host_module, [0u8; 32]);
    let mut sizing = module.ckks_ciphertext_alloc(params.base2k.into(), params.k.into());
    sizing.set_meta(params.prec().meta);
    let size = module.ckks_powi_tmp_bytes(&sizing, &params.tsk_layout());
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

    for exponent in [0u32, 1, 2, 3, 8, 15] {
        let want: Vec<F> = x.iter().map(|&v| v.powi(exponent as i32)).collect();
        let mut res = module.ckks_ciphertext_alloc(params.base2k.into(), params.k.into());
        module
            .ckks_powi(&mut res, &input, exponent, &tsk, &mut scratch.borrow())
            .expect("ckks_powi");
        let multiplications = if exponent <= 1 {
            0
        } else {
            (31 - exponent.leading_zeros()) as usize + exponent.count_ones() as usize - 1
        };
        assert_eq!(
            input.log_budget() - res.log_budget(),
            multiplications * log_delta
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
            &format!("powi/{exponent}"),
            &re_out,
            &want,
            log_delta as f64 - 14.0,
            params.n,
        );
    }
}

pub fn test_pow<BE, F, E>(
    base: CKKSTestParams,
    module: &Module<BE>,
    host_module: &Module<HostBytesBackend>,
) where
    BE: TestContextBackend,
    Module<BE>: TestContextModule<BE> + CKKSPowOps<BE>,
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
    let options = PowOptions::default();
    let base_a = F::from_f64(0.5).unwrap();
    let base_b = F::from_f64(1.5).unwrap();
    let exponent_a = F::from_f64(-2.0).unwrap();
    let exponent_b = F::from_f64(2.0).unwrap();
    let host_plan = PowPlan::from_precision(
        base_a,
        base_b,
        exponent_a,
        exponent_b,
        base.base2k.into(),
        coeff_meta,
        options,
        host_module,
    )
    .expect("PowPlan::from_precision");
    let consumed = host_plan.consumed_bits(log_delta, log_delta);
    let plan = host_plan.map_plaintexts(|pt| upload_pt(module, pt));
    let params = params_for(&base, consumed);
    let slots = params.n / 2;
    let encoder = ReferenceEncoder::<E>::new::<F>(slots).unwrap();
    let mut source = Source::new([0xc3u8; 32]);
    let mut x: Vec<F> = (0..slots)
        .map(|_| F::from_f64(source.next_f64(0.5, 1.5)).unwrap())
        .collect();
    let mut y: Vec<F> = (0..slots)
        .map(|_| F::from_f64(source.next_f64(-2.0, 2.0)).unwrap())
        .collect();
    x[0] = base_a;
    x[1] = base_b;
    y[0] = exponent_a;
    y[1] = exponent_b;
    let want: Vec<F> = x.iter().zip(&y).map(|(&x, &y)| x.powf(y)).collect();
    let im = vec![F::zero(); slots];
    let (sk_raw, sk) = gen_sk_with_raw(&params, module, host_module, [0xc4u8; 32]);
    let mut sizing = module.ckks_ciphertext_alloc(params.base2k.into(), params.k.into());
    sizing.set_meta(params.prec().meta);
    let size = module.ckks_pow_tmp_bytes(&sizing, &params.tsk_layout(), &plan);
    let mut scratch = ScratchOwned::<BE>::alloc(size);
    let tsk = gen_tsk(&params, module, &sk_raw, &mut scratch.borrow());
    let base_ct = ckks_encrypt_with_prec(
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
    let exponent_ct = ckks_encrypt_with_prec(
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
        .ckks_pow(
            &mut res,
            &base_ct,
            &exponent_ct,
            &plan,
            &tsk,
            &mut scratch.borrow(),
        )
        .expect("ckks_pow");
    assert_eq!(base_ct.log_budget() - res.log_budget(), consumed);

    let (re_out, _) = ckks_decrypt_decode::<BE, F, E>(
        &params,
        module,
        &encoder,
        &res,
        &sk,
        &mut scratch.borrow(),
    );
    assert_precision_bits("pow", &re_out, &want, options.target_bits, params.n);
}
