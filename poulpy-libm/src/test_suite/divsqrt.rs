//! Tests for [`CKKSDivSqrtOps`]: `div`, `sqrt`.

use poulpy_core::layouts::{
    GGLWEInfos, GLWETensorKeyPrepared, GLWEToBackendMut, GLWEToBackendRef,
    prepared::GLWETensorKeyPreparedToBackendRef,
};
use poulpy_hal::{
    api::{NegacyclicFFT, NegacyclicFFTNew, ScratchOwnedAlloc, ScratchOwnedBorrow},
    layouts::{HostBytesBackend, Module, ScratchOwned},
};

use poulpy_ckks::{
    CKKSCtBounds, CKKSInfos, SetCKKSInfos,
    layouts::{CKKSCiphertext, CKKSModuleAlloc},
    test_suite::reference_encoder::ReferenceEncoder,
    test_suite::{
        CKKSTestParams,
        helpers::{
            TestContextBackend, TestContextModule, TestScalar, ckks_decrypt_decode,
            ckks_encrypt_with_prec, gen_sk_with_raw, gen_tsk,
        },
    },
};

use crate::iterative::CKKSDivSqrtOps;

use super::helpers::{assert_precision_bits, params_for_with_headroom, sample_uniform};

pub fn test_div<BE, F, E>(
    base: CKKSTestParams,
    module: &Module<BE>,
    host_module: &Module<HostBytesBackend>,
) where
    BE: TestContextBackend,
    Module<BE>: TestContextModule<BE> + CKKSDivSqrtOps<BE>,
    CKKSCiphertext<BE::OwnedBuf, BE::ZnxWord>:
        GLWEToBackendMut<BE> + GLWEToBackendRef<BE> + CKKSCtBounds + SetCKKSInfos,
    GLWETensorKeyPrepared<BE::OwnedBuf, BE>: GLWETensorKeyPreparedToBackendRef<BE> + GGLWEInfos,
    F: TestScalar,
    E: NegacyclicFFT<F> + NegacyclicFFTNew<F>,
{
    let log_delta = base.prec_meta.log_delta;
    let iters = 6usize;
    let params = params_for_with_headroom(&base, (iters + 2) * log_delta, log_delta);
    let slots = params.n / 2;
    let encoder = ReferenceEncoder::<E>::new::<F>(slots).unwrap();

    let a = sample_uniform::<F>(slots, -0.8, 0.8, 0x11);
    // Goldschmidt domain.
    let b = sample_uniform::<F>(slots, 0.5, 1.5, 0x22);
    let im = vec![F::zero(); slots];
    let want: Vec<F> = a.iter().zip(&b).map(|(&av, &bv)| av / bv).collect();

    let (sk_raw, sk) = gen_sk_with_raw(&params, module, host_module, [0u8; 32]);
    let mut sizing = module.ckks_ciphertext_alloc(params.base2k.into(), params.k.into());
    sizing.set_meta(params.prec().meta);
    let size = module.ckks_div_sqrt_tmp_bytes(&sizing, &params.tsk_layout());
    let mut scratch = ScratchOwned::<BE>::alloc(size);
    let tsk = gen_tsk(&params, module, &sk_raw, &mut scratch.borrow());

    let ca = ckks_encrypt_with_prec(
        &params,
        module,
        host_module,
        &encoder,
        &sk,
        params.k,
        &a,
        &im,
        params.prec(),
        &mut scratch.borrow(),
    );
    let cb = ckks_encrypt_with_prec(
        &params,
        module,
        host_module,
        &encoder,
        &sk,
        params.k,
        &b,
        &im,
        params.prec(),
        &mut scratch.borrow(),
    );
    let input_budget = cb.log_budget();
    let mut res = module.ckks_ciphertext_alloc(params.base2k.into(), params.k.into());
    module
        .ckks_div(&mut res, &ca, &cb, iters, &tsk, &mut scratch.borrow())
        .expect("ckks_div");
    assert_eq!(res.log_delta(), log_delta, "div: log_delta preserved");
    assert_eq!(input_budget - res.log_budget(), (iters + 2) * log_delta);

    let (re_out, _) = ckks_decrypt_decode::<BE, F, E>(
        &params,
        module,
        &encoder,
        &res,
        &sk,
        &mut scratch.borrow(),
    );
    assert_precision_bits("div", &re_out, &want, log_delta as f64 - 12.0, params.n);
}

pub fn test_sqrt<BE, F, E>(
    base: CKKSTestParams,
    module: &Module<BE>,
    host_module: &Module<HostBytesBackend>,
) where
    BE: TestContextBackend,
    Module<BE>: TestContextModule<BE> + CKKSDivSqrtOps<BE>,
    CKKSCiphertext<BE::OwnedBuf, BE::ZnxWord>:
        GLWEToBackendMut<BE> + GLWEToBackendRef<BE> + CKKSCtBounds + SetCKKSInfos,
    GLWETensorKeyPrepared<BE::OwnedBuf, BE>: GLWETensorKeyPreparedToBackendRef<BE> + GGLWEInfos,
    F: TestScalar,
    E: NegacyclicFFT<F> + NegacyclicFFTNew<F>,
{
    let log_delta = base.prec_meta.log_delta;
    let r = 5usize;
    let params = params_for_with_headroom(&base, (2 * r + 1) * log_delta, log_delta);
    let slots = params.n / 2;
    let encoder = ReferenceEncoder::<E>::new::<F>(slots).unwrap();

    // x is the initial rsqrt estimate.
    let x = sample_uniform::<F>(slots, 0.9, 1.1, 0x33);
    let x_half: Vec<F> = x.iter().map(|&v| v * F::from_f64(0.5).unwrap()).collect();
    let im = vec![F::zero(); slots];
    let want: Vec<F> = x.iter().map(|&v| v.sqrt()).collect();

    let (sk_raw, sk) = gen_sk_with_raw(&params, module, host_module, [0u8; 32]);
    let mut sizing = module.ckks_ciphertext_alloc(params.base2k.into(), params.k.into());
    sizing.set_meta(params.prec().meta);
    let size = module.ckks_div_sqrt_tmp_bytes(&sizing, &params.tsk_layout());
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
    let in_half = ckks_encrypt_with_prec(
        &params,
        module,
        host_module,
        &encoder,
        &sk,
        params.k,
        &x_half,
        &im,
        params.prec(),
        &mut scratch.borrow(),
    );
    let input_budget = cx.log_budget();
    let mut res = module.ckks_ciphertext_alloc(params.base2k.into(), params.k.into());
    module
        .ckks_sqrt(&mut res, &cx, &in_half, r, &tsk, &mut scratch.borrow())
        .expect("ckks_sqrt");
    assert_eq!(res.log_delta(), log_delta, "sqrt: log_delta preserved");
    assert_eq!(input_budget - res.log_budget(), (2 * r + 1) * log_delta);

    let (re_out, _) = ckks_decrypt_decode::<BE, F, E>(
        &params,
        module,
        &encoder,
        &res,
        &sk,
        &mut scratch.borrow(),
    );
    assert_precision_bits("sqrt", &re_out, &want, log_delta as f64 - 15.0, params.n);
}
