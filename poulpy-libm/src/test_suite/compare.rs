//! Tests for [`CKKSComparisonOps`] (fmax, fmin) at a single modulus.

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

use crate::sign::{CKKSComparisonOps, CKKSSignOps};

use super::helpers::{assert_precision_bits, gen_composite, params_for};

pub fn test_max_min<BE, F, E>(
    base: CKKSTestParams,
    module: &Module<BE>,
    host_module: &Module<HostBytesBackend>,
) where
    BE: TestContextBackend,
    Module<BE>: TestContextModule<BE> + CKKSSignOps<BE> + CKKSComparisonOps<BE>,
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

    let (composite, host_composite, consumed) = gen_composite::<F, BE>(&base, module, host_module);
    let params = params_for(&base, 2 * consumed + 2 * log_delta + 2);

    let slots = params.n / 2;
    let encoder = ReferenceEncoder::<E>::new::<F>(slots).unwrap();

    let mut source = Source::new([0x33u8; 32]);
    let mut a: Vec<F> = (0..slots)
        .map(|_| F::from_f64(source.next_f64(-0.4, 0.4)).unwrap())
        .collect();
    let mut b: Vec<F> = (0..slots)
        .map(|_| F::from_f64(source.next_f64(-0.4, 0.4)).unwrap())
        .collect();
    (a[0], b[0]) = (F::from_f64(0.4).unwrap(), F::from_f64(-0.4).unwrap());
    (a[1], b[1]) = (F::from_f64(-0.4).unwrap(), F::from_f64(0.4).unwrap());
    let im = vec![F::zero(); slots];

    let half = F::from_f64(0.5).unwrap();
    let (want_max, want_min): (Vec<F>, Vec<F>) = a
        .iter()
        .zip(&b)
        .map(|(&av, &bv)| {
            let s = (host_composite.evaluate(av - bv) + F::one()) * half;
            (av * s + bv * (F::one() - s), av * (F::one() - s) + bv * s)
        })
        .unzip();
    let lo = F::from_f64(-0.2).unwrap();
    let hi = F::from_f64(0.25).unwrap();
    let step = |v| (host_composite.evaluate(v) + F::one()) * half;
    let max_const = |v| v + step(lo - v) * (lo - v);
    let min_const = |v| v - step(v - hi) * (v - hi);
    let want_max_const: Vec<F> = a.iter().copied().map(max_const).collect();
    let want_min_const: Vec<F> = a.iter().copied().map(min_const).collect();
    let want_clamp: Vec<F> = a.iter().copied().map(|v| min_const(max_const(v))).collect();

    let mut host_bounds = host_module.ckks_pt_coeffs_alloc(2, base.base2k.into(), coeff_meta.k());
    host_bounds.set_meta(coeff_meta.meta());
    host_bounds.encode_host_floats(&[lo, hi]).unwrap();
    let bounds = upload_pt(module, &host_bounds);

    let (sk_raw, sk) = gen_sk_with_raw(&params, module, host_module, [0u8; 32]);

    let mut sizing = module.ckks_ciphertext_alloc(params.base2k.into(), params.k.into());
    sizing.set_meta(params.prec().meta);
    let scratch_size = module.ckks_comparison_tmp_bytes(
        &sizing,
        &params.tsk_layout(),
        &params.atk_layout(),
        &coeff_meta,
    );
    let mut scratch = ScratchOwned::<BE>::alloc(scratch_size);
    let tsk = gen_tsk(&params, module, &sk_raw, &mut scratch.borrow());
    let conj_key = gen_atk(&params, module, -1, &sk_raw, &mut scratch.borrow());

    let op0 = ckks_encrypt_with_prec(
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
    let op1 = ckks_encrypt_with_prec(
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

    let input_budget = op0.log_budget();
    let required = log_delta as f64 - 15.0;
    for (op, want, label) in [
        (0, &want_max, "fmax"),
        (1, &want_min, "fmin"),
        (2, &want_max_const, "fmax_const"),
        (3, &want_min_const, "fmin_const"),
        (4, &want_clamp, "clamp"),
    ] {
        let mut res = module.ckks_ciphertext_alloc(params.base2k.into(), params.k.into());
        match op {
            0 => module.ckks_fmax(
                &mut res,
                &op0,
                &op1,
                &composite,
                &tsk,
                &conj_key,
                &mut scratch.borrow(),
            ),
            1 => module.ckks_fmin(
                &mut res,
                &op0,
                &op1,
                &composite,
                &tsk,
                &conj_key,
                &mut scratch.borrow(),
            ),
            2 => module.ckks_fmax_const(
                &mut res,
                &op0,
                &bounds,
                0,
                &composite,
                &tsk,
                &conj_key,
                &mut scratch.borrow(),
            ),
            3 => module.ckks_fmin_const(
                &mut res,
                &op0,
                &bounds,
                1,
                &composite,
                &tsk,
                &conj_key,
                &mut scratch.borrow(),
            ),
            _ => module.ckks_clamp(
                &mut res,
                &op0,
                &bounds,
                0,
                1,
                &composite,
                &tsk,
                &conj_key,
                &mut scratch.borrow(),
            ),
        }
        .unwrap_or_else(|e| panic!("ckks_{label}: {e}"));
        assert_eq!(res.log_delta(), log_delta, "{label}: log_delta preserved");
        let expected = if op == 4 {
            2 * (consumed + log_delta + 1)
        } else {
            consumed + log_delta + 1
        };
        assert_eq!(
            input_budget - res.log_budget(),
            expected,
            "{label}: consumed bits"
        );

        let (re_out, _) = ckks_decrypt_decode::<BE, F, E>(
            &params,
            module,
            &encoder,
            &res,
            &sk,
            &mut scratch.borrow(),
        );
        assert_precision_bits(label, &re_out, want, required, params.n);
    }
}
