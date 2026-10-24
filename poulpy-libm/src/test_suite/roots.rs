//! Tests for root functions.

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

use crate::roots::{CKKSRootOps, CbrtPlan, HypotPlan, RootOptions};

use super::helpers::{assert_precision_bits, params_for};

pub fn test_root_family<BE, F, E>(
    base: CKKSTestParams,
    module: &Module<BE>,
    host_module: &Module<HostBytesBackend>,
) where
    BE: TestContextBackend,
    Module<BE>: TestContextModule<BE> + CKKSRootOps<BE>,
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
    let options = RootOptions::default();
    let host_cbrt = CbrtPlan::from_precision(
        F::from_f64(-1.5).unwrap(),
        F::from_f64(-0.5).unwrap(),
        base.base2k.into(),
        coeff_meta,
        options,
        host_module,
    )
    .expect("CbrtPlan::from_precision");
    let host_hypot = HypotPlan::from_precision(
        F::from_f64(0.5).unwrap(),
        F::one(),
        F::from_f64(0.5).unwrap(),
        F::one(),
        base.base2k.into(),
        coeff_meta,
        options,
        host_module,
    )
    .expect("HypotPlan::from_precision");
    let max_consumed = host_cbrt
        .consumed_bits(log_delta)
        .max(host_hypot.consumed_bits(log_delta));
    let cbrt = host_cbrt.map_plaintexts(|pt| upload_pt(module, pt));
    let hypot = host_hypot.map_plaintexts(|pt| upload_pt(module, pt));
    let params = params_for(&base, max_consumed);
    let slots = params.n / 2;
    let encoder = ReferenceEncoder::<E>::new::<F>(slots).unwrap();
    let mut source = Source::new([0xc5u8; 32]);
    let x: Vec<F> = (0..slots)
        .map(|_| F::from_f64(source.next_f64(-1.5, -0.5)).unwrap())
        .collect();
    let hx: Vec<F> = (0..slots)
        .map(|_| F::from_f64(source.next_f64(0.5, 1.0)).unwrap())
        .collect();
    let hy: Vec<F> = (0..slots)
        .map(|_| F::from_f64(source.next_f64(0.5, 1.0)).unwrap())
        .collect();
    let im = vec![F::zero(); slots];
    let (sk_raw, sk) = gen_sk_with_raw(&params, module, host_module, [0xc6u8; 32]);
    let mut sizing = module.ckks_ciphertext_alloc(params.base2k.into(), params.k.into());
    sizing.set_meta(params.prec().meta);
    let size = module
        .ckks_cbrt_tmp_bytes(&sizing, &params.tsk_layout(), &cbrt)
        .max(module.ckks_hypot_tmp_bytes(&sizing, &params.tsk_layout(), &hypot));
    let mut scratch = ScratchOwned::<BE>::alloc(size);
    let tsk = gen_tsk(&params, module, &sk_raw, &mut scratch.borrow());
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
    let chx = ckks_encrypt_with_prec(
        &params,
        module,
        host_module,
        &encoder,
        &sk,
        params.k,
        &hx,
        &im,
        params.prec(),
        &mut scratch.borrow(),
    );
    let chy = ckks_encrypt_with_prec(
        &params,
        module,
        host_module,
        &encoder,
        &sk,
        params.k,
        &hy,
        &im,
        params.prec(),
        &mut scratch.borrow(),
    );

    let mut res = module.ckks_ciphertext_alloc(params.base2k.into(), params.k.into());
    module
        .ckks_cbrt(&mut res, &cx, &cbrt, &tsk, &mut scratch.borrow())
        .expect("ckks_cbrt");
    assert_eq!(
        cx.log_budget() - res.log_budget(),
        cbrt.consumed_bits(log_delta)
    );
    let (re_out, _) = ckks_decrypt_decode::<BE, F, E>(
        &params,
        module,
        &encoder,
        &res,
        &sk,
        &mut scratch.borrow(),
    );
    let want: Vec<F> = x.iter().map(|&v| v.cbrt()).collect();
    assert_precision_bits("cbrt", &re_out, &want, options.target_bits, params.n);

    let mut res = module.ckks_ciphertext_alloc(params.base2k.into(), params.k.into());
    module
        .ckks_hypot(&mut res, &chx, &chy, &hypot, &tsk, &mut scratch.borrow())
        .expect("ckks_hypot");
    assert_eq!(
        chx.log_budget() - res.log_budget(),
        hypot.consumed_bits(log_delta)
    );
    let (re_out, _) = ckks_decrypt_decode::<BE, F, E>(
        &params,
        module,
        &encoder,
        &res,
        &sk,
        &mut scratch.borrow(),
    );
    let want: Vec<F> = hx
        .iter()
        .zip(&hy)
        .map(|(&x, &y)| (x * x + y * y).sqrt())
        .collect();
    assert_precision_bits("hypot", &re_out, &want, options.target_bits, params.n);
}
