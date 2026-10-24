//! Tests for thin arithmetic functions.

use poulpy_core::layouts::{
    GGLWEInfos, GLWETensorKeyPrepared, GLWEToBackendMut, GLWEToBackendRef,
    prepared::GLWETensorKeyPreparedToBackendRef,
};
use poulpy_hal::{
    api::{NegacyclicFFT, NegacyclicFFTNew, ScratchOwnedAlloc, ScratchOwnedBorrow},
    layouts::{HostBytesBackend, Module, ScratchOwned},
    source::Source,
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

use crate::arithmetic::CKKSArithmeticOps;

use super::helpers::{assert_precision_bits, params_for};

pub fn test_arithmetic<BE, F, E>(
    base: CKKSTestParams,
    module: &Module<BE>,
    host_module: &Module<HostBytesBackend>,
) where
    BE: TestContextBackend,
    Module<BE>: TestContextModule<BE> + CKKSArithmeticOps<BE>,
    CKKSCiphertext<BE::OwnedBuf, BE::ZnxWord>:
        GLWEToBackendMut<BE> + GLWEToBackendRef<BE> + CKKSCtBounds + SetCKKSInfos,
    GLWETensorKeyPrepared<BE::OwnedBuf, BE>: GLWETensorKeyPreparedToBackendRef<BE> + GGLWEInfos,
    F: TestScalar,
    E: NegacyclicFFT<F> + NegacyclicFFTNew<F>,
{
    let log_delta = base.prec_meta.log_delta;
    let params = params_for(&base, log_delta);
    let slots = params.n / 2;
    let encoder = ReferenceEncoder::<E>::new::<F>(slots).unwrap();
    let mut source = Source::new([0x35u8; 32]);
    let a: Vec<F> = (0..slots)
        .map(|_| F::from_f64(source.next_f64(-0.2, 0.2)).unwrap())
        .collect();
    let b: Vec<F> = (0..slots)
        .map(|_| F::from_f64(source.next_f64(-0.5, 0.5)).unwrap())
        .collect();
    let c: Vec<F> = (0..slots)
        .map(|_| F::from_f64(source.next_f64(-0.2, 0.2)).unwrap())
        .collect();
    let im = vec![F::zero(); slots];
    let (sk_raw, sk) = gen_sk_with_raw(&params, module, host_module, [0x36u8; 32]);
    let mut sizing = module.ckks_ciphertext_alloc(params.base2k.into(), params.k.into());
    sizing.set_meta(params.prec().meta);
    let size = module
        .ckks_fma_tmp_bytes(&sizing, &params.tsk_layout())
        .max(module.ckks_scalbn_tmp_bytes());
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
    let cc = ckks_encrypt_with_prec(
        &params,
        module,
        host_module,
        &encoder,
        &sk,
        params.k,
        &c,
        &im,
        params.prec(),
        &mut scratch.borrow(),
    );

    let mut res = module.ckks_ciphertext_alloc(params.base2k.into(), params.k.into());
    module
        .ckks_fma(&mut res, &ca, &cb, &cc, &tsk, &mut scratch.borrow())
        .expect("ckks_fma");
    assert_eq!(ca.log_budget() - res.log_budget(), log_delta);
    let (re_out, _) = ckks_decrypt_decode::<BE, F, E>(
        &params,
        module,
        &encoder,
        &res,
        &sk,
        &mut scratch.borrow(),
    );
    let want: Vec<F> = a
        .iter()
        .zip(&b)
        .zip(&c)
        .map(|((&a, &b), &c)| a * b + c)
        .collect();
    assert_precision_bits("fma", &re_out, &want, log_delta as f64 - 10.0, params.n);

    let mut res = module.ckks_ciphertext_alloc(params.base2k.into(), params.k.into());
    module
        .ckks_scalbn(&mut res, &ca, 3, &mut scratch.borrow())
        .expect("ckks_scalbn");
    assert_eq!(ca.log_budget(), res.log_budget());
    let (re_out, _) = ckks_decrypt_decode::<BE, F, E>(
        &params,
        module,
        &encoder,
        &res,
        &sk,
        &mut scratch.borrow(),
    );
    let eight = F::from_f64(8.0).unwrap();
    let want: Vec<F> = a.iter().map(|&v| v * eight).collect();
    assert_precision_bits("scalbn", &re_out, &want, log_delta as f64 - 8.0, params.n);

    let mut res = module.ckks_ciphertext_alloc(params.base2k.into(), params.k.into());
    module
        .ckks_ldexp(&mut res, &ca, -2, &mut scratch.borrow())
        .expect("ckks_ldexp");
    assert_eq!(ca.log_budget() - res.log_budget(), 2);
    let (re_out, _) = ckks_decrypt_decode::<BE, F, E>(
        &params,
        module,
        &encoder,
        &res,
        &sk,
        &mut scratch.borrow(),
    );
    let four = F::from_f64(4.0).unwrap();
    let want: Vec<F> = a.iter().map(|&v| v / four).collect();
    assert_precision_bits("ldexp", &re_out, &want, log_delta as f64 - 8.0, params.n);
}
