//! Tests for [`CKKSPredicateOps`].

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
    api::CKKSMulOps,
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

use crate::sign::{CKKSPredicateOps, CKKSSignOps};

use super::helpers::{assert_precision_bits, gen_composite, params_for};

pub fn test_fabs<BE, F, E>(
    base: CKKSTestParams,
    module: &Module<BE>,
    host_module: &Module<HostBytesBackend>,
) where
    BE: TestContextBackend,
    Module<BE>: TestContextModule<BE> + CKKSSignOps<BE> + CKKSPredicateOps<BE>,
    CKKSCiphertext<BE::OwnedBuf, BE::ZnxWord>:
        GLWEToBackendMut<BE> + GLWEToBackendRef<BE> + CKKSCtBounds + SetCKKSInfos,
    CKKSPlaintext<BE::OwnedBuf, BE::ZnxWord>:
        GLWEToBackendRef<BE> + LWEInfos + poulpy_core::layouts::IntPolyInfos,
    GLWETensorKeyPrepared<BE::OwnedBuf, BE>: GLWETensorKeyPreparedToBackendRef<BE> + GGLWEInfos,
    F: TestScalar,
    E: NegacyclicFFT<F> + NegacyclicFFTNew<F>,
{
    let log_delta = base.prec_meta.log_delta;
    let coeff_meta = ckks_spec(base.n, base.base2k, log_delta, base.base2k);
    let (composite, host_ref, consumed) = gen_composite::<F, BE>(&base, module, host_module);
    let params = params_for(&base, consumed);

    let slots = params.n / 2;
    let encoder = ReferenceEncoder::<E>::new::<F>(slots).unwrap();
    let mut source = Source::new([0x2au8; 32]);
    let x: Vec<F> = (0..slots)
        .map(|_| {
            let mag = source.next_f64(0.1, 1.0);
            let s = if source.next_f64(-1.0, 1.0) < 0.0 {
                -1.0
            } else {
                1.0
            };
            F::from_f64(s * mag).unwrap()
        })
        .collect();
    let im = vec![F::zero(); slots];

    // Circuit output is x·sign_approx(x); it must match |x| to the composite's precision.
    let want: Vec<F> = x.iter().map(|&v| v * host_ref.evaluate(v)).collect();
    let ideal: Vec<F> = x.iter().map(|&v| v.abs()).collect();
    assert_precision_bits("fabs/reference", &want, &ideal, 16.0, params.n);

    let (sk_raw, sk) = gen_sk_with_raw(&params, module, host_module, [0u8; 32]);
    let mut sizing = module.ckks_ciphertext_alloc(params.base2k.into(), params.k.into());
    sizing.set_meta(params.prec().meta);
    let size = module.ckks_predicate_tmp_bytes(
        &sizing,
        &params.tsk_layout(),
        &params.atk_layout(),
        &coeff_meta,
    );
    let mut scratch = ScratchOwned::<BE>::alloc(size);
    let tsk = gen_tsk(&params, module, &sk_raw, &mut scratch.borrow());
    let conj_key = gen_atk(&params, module, -1, &sk_raw, &mut scratch.borrow());

    let ct = ckks_encrypt_with_prec(
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
        .ckks_fabs(
            &mut res,
            &ct,
            &composite,
            &tsk,
            &conj_key,
            &mut scratch.borrow(),
        )
        .expect("ckks_fabs");

    let (re_out, _) = ckks_decrypt_decode::<BE, F, E>(
        &params,
        module,
        &encoder,
        &res,
        &sk,
        &mut scratch.borrow(),
    );
    assert_precision_bits("fabs", &re_out, &want, log_delta as f64 - 15.0, params.n);
}

pub fn test_fdim_copysign<BE, F, E>(
    base: CKKSTestParams,
    module: &Module<BE>,
    host_module: &Module<HostBytesBackend>,
) where
    BE: TestContextBackend,
    Module<BE>: TestContextModule<BE> + CKKSSignOps<BE> + CKKSPredicateOps<BE>,
    CKKSCiphertext<BE::OwnedBuf, BE::ZnxWord>:
        GLWEToBackendMut<BE> + GLWEToBackendRef<BE> + CKKSCtBounds + SetCKKSInfos,
    CKKSPlaintext<BE::OwnedBuf, BE::ZnxWord>:
        GLWEToBackendRef<BE> + LWEInfos + poulpy_core::layouts::IntPolyInfos,
    GLWETensorKeyPrepared<BE::OwnedBuf, BE>: GLWETensorKeyPreparedToBackendRef<BE> + GGLWEInfos,
    F: TestScalar,
    E: NegacyclicFFT<F> + NegacyclicFFTNew<F>,
{
    let log_delta = base.prec_meta.log_delta;
    let coeff_meta = ckks_spec(base.n, base.base2k, log_delta, base.base2k);
    let (composite, host_ref, consumed) = gen_composite::<F, BE>(&base, module, host_module);
    let params = params_for(&base, consumed + 2 * log_delta + 1);
    let slots = params.n / 2;
    let encoder = ReferenceEncoder::<E>::new::<F>(slots).unwrap();
    let mut source = Source::new([0x71u8; 32]);
    let a: Vec<F> = (0..slots)
        .map(|_| {
            let mag = source.next_f64(0.2, 0.8);
            let sign = if source.next_f64(-1.0, 1.0) < 0.0 {
                -1.0
            } else {
                1.0
            };
            F::from_f64(sign * mag).unwrap()
        })
        .collect();
    let y: Vec<F> = (0..slots)
        .map(|_| {
            let mag = source.next_f64(0.2, 0.8);
            let sign = if source.next_f64(-1.0, 1.0) < 0.0 {
                -1.0
            } else {
                1.0
            };
            F::from_f64(sign * mag).unwrap()
        })
        .collect();
    let zero = vec![F::zero(); slots];
    let half = F::from_f64(0.5).unwrap();
    let want_fdim: Vec<F> = a
        .iter()
        .map(|&v| (v + v * host_ref.evaluate(v)) * half)
        .collect();
    let ideal_fdim: Vec<F> = a.iter().map(|&v| (v + v.abs()) * half).collect();
    let want_copysign: Vec<F> = a
        .iter()
        .zip(&y)
        .map(|(&x, &y)| x * host_ref.evaluate(x) * host_ref.evaluate(y))
        .collect();
    let ideal_copysign: Vec<F> = a
        .iter()
        .zip(&y)
        .map(|(&x, &y)| x.abs() * y / y.abs())
        .collect();
    assert_precision_bits("fdim/reference", &want_fdim, &ideal_fdim, 16.0, params.n);
    assert_precision_bits(
        "copysign/reference",
        &want_copysign,
        &ideal_copysign,
        16.0,
        params.n,
    );

    let (sk_raw, sk) = gen_sk_with_raw(&params, module, host_module, [0x72u8; 32]);
    let mut sizing = module.ckks_ciphertext_alloc(params.base2k.into(), params.k.into());
    sizing.set_meta(params.prec().meta);
    let size = module.ckks_predicate_tmp_bytes(
        &sizing,
        &params.tsk_layout(),
        &params.atk_layout(),
        &coeff_meta,
    );
    let mut scratch = ScratchOwned::<BE>::alloc(size);
    let tsk = gen_tsk(&params, module, &sk_raw, &mut scratch.borrow());
    let conj_key = gen_atk(&params, module, -1, &sk_raw, &mut scratch.borrow());
    let ca = ckks_encrypt_with_prec(
        &params,
        module,
        host_module,
        &encoder,
        &sk,
        params.k,
        &a,
        &zero,
        params.prec(),
        &mut scratch.borrow(),
    );
    let czero = ckks_encrypt_with_prec(
        &params,
        module,
        host_module,
        &encoder,
        &sk,
        params.k,
        &zero,
        &zero,
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
        &zero,
        params.prec(),
        &mut scratch.borrow(),
    );

    let mut res = module.ckks_ciphertext_alloc(params.base2k.into(), params.k.into());
    module
        .ckks_fdim(
            &mut res,
            &ca,
            &czero,
            &composite,
            &tsk,
            &conj_key,
            &mut scratch.borrow(),
        )
        .expect("ckks_fdim");
    assert_eq!(ca.log_budget() - res.log_budget(), consumed + log_delta + 1);
    let (re_out, _) = ckks_decrypt_decode::<BE, F, E>(
        &params,
        module,
        &encoder,
        &res,
        &sk,
        &mut scratch.borrow(),
    );
    assert_precision_bits(
        "fdim",
        &re_out,
        &want_fdim,
        log_delta as f64 - 16.0,
        params.n,
    );

    let mut res = module.ckks_ciphertext_alloc(params.base2k.into(), params.k.into());
    module
        .ckks_copysign(
            &mut res,
            &ca,
            &cy,
            &composite,
            &tsk,
            &conj_key,
            &mut scratch.borrow(),
        )
        .expect("ckks_copysign");
    assert_eq!(ca.log_budget() - res.log_budget(), consumed + 2 * log_delta);
    let (re_out, _) = ckks_decrypt_decode::<BE, F, E>(
        &params,
        module,
        &encoder,
        &res,
        &sk,
        &mut scratch.borrow(),
    );
    assert_precision_bits(
        "copysign",
        &re_out,
        &want_copysign,
        log_delta as f64 - 18.0,
        params.n,
    );
}

pub fn test_compare<BE, F, E>(
    base: CKKSTestParams,
    module: &Module<BE>,
    host_module: &Module<HostBytesBackend>,
) where
    BE: TestContextBackend,
    Module<BE>: TestContextModule<BE> + CKKSSignOps<BE> + CKKSPredicateOps<BE>,
    CKKSCiphertext<BE::OwnedBuf, BE::ZnxWord>:
        GLWEToBackendMut<BE> + GLWEToBackendRef<BE> + CKKSCtBounds + SetCKKSInfos,
    CKKSPlaintext<BE::OwnedBuf, BE::ZnxWord>:
        GLWEToBackendRef<BE> + LWEInfos + poulpy_core::layouts::IntPolyInfos,
    GLWETensorKeyPrepared<BE::OwnedBuf, BE>: GLWETensorKeyPreparedToBackendRef<BE> + GGLWEInfos,
    F: TestScalar,
    E: NegacyclicFFT<F> + NegacyclicFFTNew<F>,
{
    let log_delta = base.prec_meta.log_delta;
    let coeff_meta = ckks_spec(base.n, base.base2k, log_delta, base.base2k);
    let (composite, host_ref, consumed) = gen_composite::<F, BE>(&base, module, host_module);
    let params = params_for(&base, consumed);

    let slots = params.n / 2;
    let encoder = ReferenceEncoder::<E>::new::<F>(slots).unwrap();
    let mut source = Source::new([0x6bu8; 32]);
    let a: Vec<F> = (0..slots)
        .map(|_| F::from_f64(source.next_f64(-0.4, 0.4)).unwrap())
        .collect();
    let b: Vec<F> = (0..slots)
        .map(|_| F::from_f64(source.next_f64(-0.4, 0.4)).unwrap())
        .collect();
    let im = vec![F::zero(); slots];

    let half = F::from_f64(0.5).unwrap();
    let want_cmp: Vec<F> = a
        .iter()
        .zip(&b)
        .map(|(&av, &bv)| host_ref.evaluate(av - bv))
        .collect();
    let want_gt: Vec<F> = a
        .iter()
        .zip(&b)
        .map(|(&av, &bv)| (host_ref.evaluate(av - bv) + F::one()) * half)
        .collect();
    let want_lt: Vec<F> = a
        .iter()
        .zip(&b)
        .map(|(&av, &bv)| (host_ref.evaluate(bv - av) + F::one()) * half)
        .collect();

    let (sk_raw, sk) = gen_sk_with_raw(&params, module, host_module, [0u8; 32]);
    let mut sizing = module.ckks_ciphertext_alloc(params.base2k.into(), params.k.into());
    sizing.set_meta(params.prec().meta);
    let size = module.ckks_predicate_tmp_bytes(
        &sizing,
        &params.tsk_layout(),
        &params.atk_layout(),
        &coeff_meta,
    );
    let mut scratch = ScratchOwned::<BE>::alloc(size);
    let tsk = gen_tsk(&params, module, &sk_raw, &mut scratch.borrow());
    let conj_key = gen_atk(&params, module, -1, &sk_raw, &mut scratch.borrow());

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
    for (op, label, want) in [
        (0, "cmp", &want_cmp),
        (1, "gt", &want_gt),
        (2, "ge", &want_gt),
        (3, "lt", &want_lt),
        (4, "le", &want_lt),
    ] {
        let mut res = module.ckks_ciphertext_alloc(params.base2k.into(), params.k.into());
        match op {
            0 => module.ckks_cmp(
                &mut res,
                &ca,
                &cb,
                &composite,
                &tsk,
                &conj_key,
                &mut scratch.borrow(),
            ),
            1 => module.ckks_gt(
                &mut res,
                &ca,
                &cb,
                &composite,
                &tsk,
                &conj_key,
                &mut scratch.borrow(),
            ),
            2 => module.ckks_ge(
                &mut res,
                &ca,
                &cb,
                &composite,
                &tsk,
                &conj_key,
                &mut scratch.borrow(),
            ),
            3 => module.ckks_lt(
                &mut res,
                &ca,
                &cb,
                &composite,
                &tsk,
                &conj_key,
                &mut scratch.borrow(),
            ),
            _ => module.ckks_le(
                &mut res,
                &ca,
                &cb,
                &composite,
                &tsk,
                &conj_key,
                &mut scratch.borrow(),
            ),
        }
        .unwrap_or_else(|e| panic!("ckks_{label}: {e}"));

        let (re_out, _) = ckks_decrypt_decode::<BE, F, E>(
            &params,
            module,
            &encoder,
            &res,
            &sk,
            &mut scratch.borrow(),
        );
        assert_precision_bits(label, &re_out, want, log_delta as f64 - 15.0, params.n);
    }
}

pub fn test_indicator_eq<BE, F, E>(
    base: CKKSTestParams,
    module: &Module<BE>,
    host_module: &Module<HostBytesBackend>,
) where
    BE: TestContextBackend,
    Module<BE>: TestContextModule<BE> + CKKSSignOps<BE> + CKKSPredicateOps<BE>,
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
    let (composite, host_ref, consumed) = gen_composite::<F, BE>(&base, module, host_module);
    let params = params_for(&base, consumed);

    let slots = params.n / 2;
    let encoder = ReferenceEncoder::<E>::new::<F>(slots).unwrap();
    let mut source = Source::new([0x81u8; 32]);
    let mut a: Vec<F> = (0..slots)
        .map(|_| F::from_f64(source.next_f64(-0.25, 0.25)).unwrap())
        .collect();
    let mut b: Vec<F> = a
        .iter()
        .map(|&v| v + F::from_f64(source.next_f64(-0.45, 0.45)).unwrap())
        .collect();
    a[0] = F::from_f64(0.1).unwrap();
    b[0] = a[0];
    a[1] = F::from_f64(0.1).unwrap();
    b[1] = F::from_f64(0.18).unwrap();
    a[2] = F::from_f64(0.1).unwrap();
    b[2] = F::from_f64(0.4).unwrap();
    let im = vec![F::zero(); slots];

    let lo = F::from_f64(-0.2).unwrap();
    let hi = F::from_f64(0.25).unwrap();
    let epsilon = F::from_f64(0.15).unwrap();
    let half = F::from_f64(0.5).unwrap();
    let step = |v| (host_ref.evaluate(v) + F::one()) * half;
    let want_indicator: Vec<F> = a.iter().map(|&v| step(v - lo) * step(hi - v)).collect();
    let want_eq: Vec<F> = a
        .iter()
        .zip(&b)
        .map(|(&av, &bv)| step(av - bv + epsilon) * step(bv - av + epsilon))
        .collect();

    let mut host_constants =
        host_module.ckks_pt_coeffs_alloc(3, base.base2k.into(), coeff_meta.k());
    host_constants.set_meta(coeff_meta.meta());
    host_constants
        .encode_host_floats(&[lo, hi, epsilon])
        .unwrap();
    let constants = upload_pt(module, &host_constants);

    let (sk_raw, sk) = gen_sk_with_raw(&params, module, host_module, [0u8; 32]);
    let mut sizing = module.ckks_ciphertext_alloc(params.base2k.into(), params.k.into());
    sizing.set_meta(params.prec().meta);
    let size = module.ckks_predicate_tmp_bytes(
        &sizing,
        &params.tsk_layout(),
        &params.atk_layout(),
        &coeff_meta,
    );
    let mut scratch = ScratchOwned::<BE>::alloc(size);
    let tsk = gen_tsk(&params, module, &sk_raw, &mut scratch.borrow());
    let conj_key = gen_atk(&params, module, -1, &sk_raw, &mut scratch.borrow());

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

    for (op, label, want) in [(0, "indicator", &want_indicator), (1, "eq", &want_eq)] {
        let mut res = module.ckks_ciphertext_alloc(params.base2k.into(), params.k.into());
        if op == 0 {
            module.ckks_indicator(
                &mut res,
                &ca,
                &constants,
                0,
                1,
                &composite,
                &tsk,
                &conj_key,
                &mut scratch.borrow(),
            )
        } else {
            module.ckks_eq(
                &mut res,
                &ca,
                &cb,
                &constants,
                2,
                &composite,
                &tsk,
                &conj_key,
                &mut scratch.borrow(),
            )
        }
        .unwrap_or_else(|e| panic!("ckks_{label}: {e}"));

        let (re_out, _) = ckks_decrypt_decode::<BE, F, E>(
            &params,
            module,
            &encoder,
            &res,
            &sk,
            &mut scratch.borrow(),
        );
        assert_precision_bits(label, &re_out, want, log_delta as f64 - 18.0, params.n);
    }
}

pub fn test_select<BE, F, E>(
    base: CKKSTestParams,
    module: &Module<BE>,
    host_module: &Module<HostBytesBackend>,
) where
    BE: TestContextBackend,
    Module<BE>: TestContextModule<BE> + CKKSPredicateOps<BE> + CKKSMulOps<BE>,
    CKKSCiphertext<BE::OwnedBuf, BE::ZnxWord>:
        GLWEToBackendMut<BE> + GLWEToBackendRef<BE> + CKKSCtBounds + SetCKKSInfos,
    GLWETensorKeyPrepared<BE::OwnedBuf, BE>: GLWETensorKeyPreparedToBackendRef<BE> + GGLWEInfos,
    F: TestScalar,
    E: NegacyclicFFT<F> + NegacyclicFFTNew<F>,
{
    let log_delta = base.prec_meta.log_delta;
    // select is a single ct×ct product: a small budget suffices.
    let params = params_for(&base, 2 * base.base2k);

    let slots = params.n / 2;
    let encoder = ReferenceEncoder::<E>::new::<F>(slots).unwrap();
    let mut source = Source::new([0x0eu8; 32]);
    let mask: Vec<F> = (0..slots)
        .map(|i| F::from_f64((i % 2) as f64).unwrap())
        .collect();
    let a: Vec<F> = (0..slots)
        .map(|_| F::from_f64(source.next_f64(-0.5, 0.5)).unwrap())
        .collect();
    let b: Vec<F> = (0..slots)
        .map(|_| F::from_f64(source.next_f64(-0.5, 0.5)).unwrap())
        .collect();
    let im = vec![F::zero(); slots];

    let want: Vec<F> = (0..slots)
        .map(|i| mask[i] * a[i] + (F::one() - mask[i]) * b[i])
        .collect();

    let (sk_raw, sk) = gen_sk_with_raw(&params, module, host_module, [0u8; 32]);
    let mut sizing = module.ckks_ciphertext_alloc(params.base2k.into(), params.k.into());
    sizing.set_meta(params.prec().meta);
    let size = module.ckks_mul_tmp_bytes(&sizing, &sizing, &sizing, &params.tsk_layout())
        + 4 * poulpy_core::layouts::GLWE::<Vec<u8>, BE::ZnxWord>::bytes_of_from_infos(&sizing);
    let mut scratch = ScratchOwned::<BE>::alloc(size);
    let tsk = gen_tsk(&params, module, &sk_raw, &mut scratch.borrow());

    let cm = ckks_encrypt_with_prec(
        &params,
        module,
        host_module,
        &encoder,
        &sk,
        params.k,
        &mask,
        &im,
        params.prec(),
        &mut scratch.borrow(),
    );
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
    let mut res = module.ckks_ciphertext_alloc(params.base2k.into(), params.k.into());
    module
        .ckks_select(&mut res, &cm, &ca, &cb, &tsk, &mut scratch.borrow())
        .expect("ckks_select");

    let (re_out, _) = ckks_decrypt_decode::<BE, F, E>(
        &params,
        module,
        &encoder,
        &res,
        &sk,
        &mut scratch.borrow(),
    );
    assert_precision_bits("select", &re_out, &want, log_delta as f64 - 12.0, params.n);
}
