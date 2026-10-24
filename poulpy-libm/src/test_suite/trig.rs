//! Tests for trigonometric functions.

use poulpy_core::layouts::{
    GGLWEInfos, GLWETensorKeyPrepared, GLWEToBackendMut, GLWEToBackendRef, LWEInfos,
    prepared::{GLWESecretPrepared, GLWETensorKeyPreparedToBackendRef},
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

use crate::trig::{CKKSTrigOps, CosPlan, SinPlan, TanPlan, TrigOptions};

use super::helpers::{assert_error, assert_error_starts_with, assert_precision_bits, params_for};

pub fn test_trig_family<BE, F, E>(
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
    let log_delta = base.prec_meta.log_delta;
    let coeff_meta = ckks_spec(base.n, base.base2k, log_delta, base.base2k);
    let options = TrigOptions::default();
    let pi = F::PI();
    assert_error(
        SinPlan::from_precision(
            F::one(),
            -F::one(),
            base.base2k.into(),
            coeff_meta,
            options,
            host_module,
        ),
        "sin: empty interval [a, b]",
    );
    assert_error_starts_with(
        SinPlan::from_precision(
            -F::one(),
            F::one(),
            base.base2k.into(),
            coeff_meta,
            TrigOptions {
                target_bits: 80.0,
                max_degree: 1,
                ..options
            },
            host_module,
        ),
        "sin: degree_for_precision: 80.0 bits not reached by degree 1",
    );
    let host_sin = SinPlan::from_precision(
        -pi,
        pi,
        base.base2k.into(),
        coeff_meta,
        options,
        host_module,
    )
    .expect("SinPlan::from_precision");
    let host_cos = CosPlan::from_precision(
        -pi,
        pi,
        base.base2k.into(),
        coeff_meta,
        options,
        host_module,
    )
    .expect("CosPlan::from_precision");
    let tan_bound = F::from_f64(0.75).unwrap();
    let host_tan = TanPlan::from_precision(
        -tan_bound,
        tan_bound,
        base.base2k.into(),
        coeff_meta,
        options,
        host_module,
    )
    .expect("TanPlan::from_precision");
    let max_consumed = host_sin
        .consumed_bits(log_delta)
        .max(host_cos.consumed_bits(log_delta))
        .max(host_tan.consumed_bits(log_delta));
    let sin = host_sin.map_plaintexts(|pt| upload_pt(module, pt));
    let cos = host_cos.map_plaintexts(|pt| upload_pt(module, pt));
    let tan = host_tan.map_plaintexts(|pt| upload_pt(module, pt));
    let params = params_for(&base, max_consumed);

    let slots = params.n / 2;
    let encoder = ReferenceEncoder::<E>::new::<F>(slots).unwrap();
    let im = vec![F::zero(); slots];
    let (sk_raw, sk) = gen_sk_with_raw(&params, module, host_module, [0xc1u8; 32]);
    let mut sizing = module.ckks_ciphertext_alloc(params.base2k.into(), params.k.into());
    sizing.set_meta(params.prec().meta);
    let size = module
        .ckks_sin_tmp_bytes(&sizing, &params.tsk_layout(), &sin)
        .max(module.ckks_cos_tmp_bytes(&sizing, &params.tsk_layout(), &cos))
        .max(module.ckks_tan_tmp_bytes(&sizing, &params.tsk_layout(), &tan));
    let mut scratch = ScratchOwned::<BE>::alloc(size);
    let tsk = gen_tsk(&params, module, &sk_raw, &mut scratch.borrow());

    let mut source = Source::new([0xc2u8; 32]);
    let mut x: Vec<F> = (0..slots)
        .map(|_| F::from_f64(source.next_f64(-std::f64::consts::PI, std::f64::consts::PI)).unwrap())
        .collect();
    x[0] = -pi;
    x[1] = F::zero();
    x[2] = pi;
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
    let mut sin_res = module.ckks_ciphertext_alloc(params.base2k.into(), params.k.into());
    let mut cos_res = module.ckks_ciphertext_alloc(params.base2k.into(), params.k.into());
    module
        .ckks_sincos(
            &mut sin_res,
            &mut cos_res,
            &input,
            &sin,
            &cos,
            &tsk,
            &mut scratch.borrow(),
        )
        .expect("ckks_sincos");
    check(
        "sin",
        &params,
        module,
        &encoder,
        &sin_res,
        &sk,
        &x.iter().map(|&v| v.sin()).collect::<Vec<_>>(),
        input.log_budget() - sin.consumed_bits(log_delta),
        options.target_bits,
        &mut scratch,
    );
    check(
        "cos",
        &params,
        module,
        &encoder,
        &cos_res,
        &sk,
        &x.iter().map(|&v| v.cos()).collect::<Vec<_>>(),
        input.log_budget() - cos.consumed_bits(log_delta),
        options.target_bits,
        &mut scratch,
    );

    let want_sin: Vec<F> = x.iter().map(|&v| v.sin()).collect();
    let want_cos: Vec<F> = x.iter().map(|&v| v.cos()).collect();
    let mut sin_res = module.ckks_ciphertext_alloc(params.base2k.into(), params.k.into());
    module
        .ckks_sin(&mut sin_res, &input, &sin, &tsk, &mut scratch.borrow())
        .expect("ckks_sin");
    check(
        "sin/direct",
        &params,
        module,
        &encoder,
        &sin_res,
        &sk,
        &want_sin,
        input.log_budget() - sin.consumed_bits(log_delta),
        options.target_bits,
        &mut scratch,
    );
    let mut cos_res = module.ckks_ciphertext_alloc(params.base2k.into(), params.k.into());
    module
        .ckks_cos(&mut cos_res, &input, &cos, &tsk, &mut scratch.borrow())
        .expect("ckks_cos");
    check(
        "cos/direct",
        &params,
        module,
        &encoder,
        &cos_res,
        &sk,
        &want_cos,
        input.log_budget() - cos.consumed_bits(log_delta),
        options.target_bits,
        &mut scratch,
    );

    let mut x: Vec<F> = (0..slots)
        .map(|_| F::from_f64(source.next_f64(-0.75, 0.75)).unwrap())
        .collect();
    x[0] = -tan_bound;
    x[1] = F::zero();
    x[2] = tan_bound;
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
        .ckks_tan(&mut res, &input, &tan, &tsk, &mut scratch.borrow())
        .expect("ckks_tan");
    check(
        "tan",
        &params,
        module,
        &encoder,
        &res,
        &sk,
        &x.iter().map(|&v| v.tan()).collect::<Vec<_>>(),
        input.log_budget() - tan.consumed_bits(log_delta),
        options.target_bits,
        &mut scratch,
    );
}

#[allow(clippy::too_many_arguments)]
fn check<BE, F, E>(
    label: &str,
    params: &CKKSTestParams,
    module: &Module<BE>,
    encoder: &ReferenceEncoder<E>,
    res: &CKKSCiphertext<BE::OwnedBuf, BE::ZnxWord>,
    sk: &GLWESecretPrepared<BE::OwnedBuf, BE>,
    want: &[F],
    want_budget: usize,
    target_bits: f64,
    scratch: &mut ScratchOwned<BE>,
) where
    BE: TestContextBackend,
    Module<BE>: TestContextModule<BE>,
    CKKSCiphertext<BE::OwnedBuf, BE::ZnxWord>: GLWEToBackendRef<BE> + CKKSCtBounds,
    F: TestScalar,
    E: NegacyclicFFT<F> + NegacyclicFFTNew<F>,
{
    assert_eq!(res.log_budget(), want_budget);
    let (re_out, _) =
        ckks_decrypt_decode::<BE, F, E>(params, module, encoder, res, sk, &mut scratch.borrow());
    assert_precision_bits(label, &re_out, want, target_bits, params.n);
}
