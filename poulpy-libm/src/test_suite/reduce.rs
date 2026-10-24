//! Tests for slot reductions.

use std::collections::HashMap;

use poulpy_core::layouts::{
    GGLWEInfos, GLWETensorKeyPrepared, GLWEToBackendMut, GLWEToBackendRef, LWEInfos,
    prepared::GLWETensorKeyPreparedToBackendRef,
};
use poulpy_core::{GLWEAutomorphism, GLWEShift};
use poulpy_hal::{
    api::{NegacyclicFFT, NegacyclicFFTNew, ScratchOwnedAlloc, ScratchOwnedBorrow},
    layouts::{GaloisElement, HostBytesBackend, Module, ScratchOwned},
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
            ckks_encrypt_with_prec, ckks_spec, gen_atk, gen_sk_with_raw, gen_tsk,
        },
    },
};

use crate::reduce::CKKSReductionOps;

use super::helpers::{assert_precision_bits, gen_composite, params_for};

pub fn test_sum_slots<BE, F, E>(
    base: CKKSTestParams,
    module: &Module<BE>,
    host_module: &Module<HostBytesBackend>,
) where
    BE: TestContextBackend,
    Module<BE>: TestContextModule<BE> + GLWEAutomorphism<BE> + GLWEShift<BE> + CKKSReductionOps<BE>,
    CKKSCiphertext<BE::OwnedBuf, BE::ZnxWord>:
        GLWEToBackendMut<BE> + GLWEToBackendRef<BE> + CKKSCtBounds + SetCKKSInfos,
    F: TestScalar,
    E: NegacyclicFFT<F> + NegacyclicFFTNew<F>,
{
    let params = base;
    let slots = params.n / 2;
    let encoder = ReferenceEncoder::<E>::new::<F>(slots).unwrap();
    let mut source = Source::new([0xe1u8; 32]);
    let x: Vec<F> = (0..slots)
        .map(|_| F::from_f64(source.next_f64(-0.01, 0.01)).unwrap())
        .collect();
    let sum = x.iter().copied().fold(F::zero(), |a, b| a + b);
    let want = vec![sum; slots];
    let im = vec![F::zero(); slots];
    let (sk_raw, sk) = gen_sk_with_raw(&params, module, host_module, [0xe2u8; 32]);

    let mut sizing = module.ckks_ciphertext_alloc(params.base2k.into(), params.k.into());
    sizing.set_meta(params.prec().meta);
    let size = module.ckks_sum_slots_tmp_bytes(&sizing, &params.atk_layout());
    let mut scratch = ScratchOwned::<BE>::alloc(size);
    let rotation_keys = rotation_keys(&params, module, &sk_raw, &mut scratch);
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
        .ckks_sum_slots(&mut res, &input, &rotation_keys, &mut scratch.borrow())
        .expect("ckks_sum_slots");
    assert_eq!(res.meta(), input.meta());
    assert_eq!(res.log_budget(), input.log_budget());

    let (re_out, _) = ckks_decrypt_decode::<BE, F, E>(
        &params,
        module,
        &encoder,
        &res,
        &sk,
        &mut scratch.borrow(),
    );
    assert_precision_bits(
        "sum_slots",
        &re_out,
        &want,
        params.prec_meta.log_delta as f64 - 15.0,
        params.n,
    );
}

pub fn test_extrema_slots<BE, F, E>(
    base: CKKSTestParams,
    module: &Module<BE>,
    host_module: &Module<HostBytesBackend>,
) where
    BE: TestContextBackend,
    Module<BE>: TestContextModule<BE> + GLWEAutomorphism<BE> + GLWEShift<BE> + CKKSReductionOps<BE>,
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
    let (composite, _, sign_consumed) = gen_composite::<F, BE>(&base, module, host_module);
    let stages = (base.n / 2).ilog2() as usize;
    let params = params_for(&base, stages * (sign_consumed + log_delta + 2));
    let slots = params.n / 2;
    let encoder = ReferenceEncoder::<E>::new::<F>(slots).unwrap();
    let mut source = Source::new([0xe3u8; 32]);
    let mut x: Vec<F> = (0..slots)
        .map(|_| F::from_f64(source.next_f64(-0.15, 0.15)).unwrap())
        .collect();
    x[0] = F::from_f64(-0.4).unwrap();
    x[1] = F::from_f64(0.4).unwrap();
    let im = vec![F::zero(); slots];
    let (sk_raw, sk) = gen_sk_with_raw(&params, module, host_module, [0xe4u8; 32]);

    let mut sizing = module.ckks_ciphertext_alloc(params.base2k.into(), params.k.into());
    sizing.set_meta(params.prec().meta);
    let size = module.ckks_fmax_slots_tmp_bytes(
        &sizing,
        &params.tsk_layout(),
        &params.atk_layout(),
        &coeff_meta,
    );
    let mut scratch = ScratchOwned::<BE>::alloc(size);
    let tsk = gen_tsk(&params, module, &sk_raw, &mut scratch.borrow());
    let conj_key = gen_atk(&params, module, -1, &sk_raw, &mut scratch.borrow());
    let rotation_keys = rotation_keys(&params, module, &sk_raw, &mut scratch);
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

    for (max, label, value) in [
        (true, "fmax_slots", F::from_f64(0.4).unwrap()),
        (false, "fmin_slots", F::from_f64(-0.4).unwrap()),
    ] {
        let mut res = module.ckks_ciphertext_alloc(params.base2k.into(), params.k.into());
        if max {
            module.ckks_fmax_slots(
                &mut res,
                &input,
                &composite,
                &tsk,
                &rotation_keys,
                &conj_key,
                &mut scratch.borrow(),
            )
        } else {
            module.ckks_fmin_slots(
                &mut res,
                &input,
                &composite,
                &tsk,
                &rotation_keys,
                &conj_key,
                &mut scratch.borrow(),
            )
        }
        .unwrap_or_else(|e| panic!("ckks_{label}: {e}"));
        assert_eq!(res.log_delta(), log_delta, "{label}: log_delta");
        assert_eq!(
            input.log_budget() - res.log_budget(),
            stages * (sign_consumed + log_delta + 1),
            "{label}: consumed bits"
        );

        let want = vec![value; slots];
        let (re_out, _) = ckks_decrypt_decode::<BE, F, E>(
            &params,
            module,
            &encoder,
            &res,
            &sk,
            &mut scratch.borrow(),
        );
        assert_precision_bits(label, &re_out, &want, 8.0, params.n);
    }
}

pub(super) fn rotation_keys<BE>(
    params: &CKKSTestParams,
    module: &Module<BE>,
    sk_raw: &poulpy_core::layouts::BackendGLWESecret<BE>,
    scratch: &mut ScratchOwned<BE>,
) -> HashMap<i64, poulpy_core::layouts::GLWEAutomorphismKeyPrepared<BE::OwnedBuf, BE>>
where
    BE: TestContextBackend,
    Module<BE>: TestContextModule<BE>,
{
    (0..(params.n / 2).ilog2())
        .map(|i| {
            let shift = 1i64 << i;
            let gal = module.galois_element(shift);
            let key = gen_atk(params, module, gal, sk_raw, &mut scratch.borrow());
            (gal, key)
        })
        .collect()
}
