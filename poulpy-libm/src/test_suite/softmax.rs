//! Tests for softmax.

use poulpy_core::layouts::{
    GGLWEInfos, GLWETensorKeyPrepared, GLWEToBackendMut, GLWEToBackendRef, LWEInfos,
    prepared::GLWETensorKeyPreparedToBackendRef,
};
use poulpy_core::{GLWEAutomorphism, GLWEShift};
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

use crate::softmax::{CKKSSoftmaxOps, SoftmaxOptions, SoftmaxPlan};

use super::{
    helpers::{assert_precision_bits, params_for},
    reduce::rotation_keys,
};

pub fn test_softmax<BE, F, E>(
    base: CKKSTestParams,
    module: &Module<BE>,
    host_module: &Module<HostBytesBackend>,
) where
    BE: TestContextBackend,
    Module<BE>: TestContextModule<BE> + GLWEAutomorphism<BE> + GLWEShift<BE> + CKKSSoftmaxOps<BE>,
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
    let options = SoftmaxOptions::default();
    let lo = F::from_f64(-1.0).unwrap();
    let hi = F::from_f64(1.0).unwrap();
    let host_plan =
        SoftmaxPlan::from_precision(lo, hi, base.base2k.into(), coeff_meta, options, host_module)
            .expect("SoftmaxPlan::from_precision");
    let consumed = host_plan.consumed_bits(log_delta);
    let plan = host_plan.map_plaintexts(|pt| upload_pt(module, pt));
    let params = params_for(&base, consumed);
    let slots = params.n / 2;
    let encoder = ReferenceEncoder::<E>::new::<F>(slots).unwrap();
    let mut source = Source::new([0xf1u8; 32]);
    let mut x: Vec<F> = (0..slots)
        .map(|_| F::from_f64(source.next_f64(-1.0, 1.0)).unwrap())
        .collect();
    x[0] = lo;
    x[1] = hi;
    let denominator = x
        .iter()
        .copied()
        .map(|v| v.exp())
        .fold(F::zero(), |a, b| a + b);
    let want: Vec<F> = x.iter().copied().map(|v| v.exp() / denominator).collect();
    let im = vec![F::zero(); slots];
    let (sk_raw, sk) = gen_sk_with_raw(&params, module, host_module, [0xf2u8; 32]);

    let mut sizing = module.ckks_ciphertext_alloc(params.base2k.into(), params.k.into());
    sizing.set_meta(params.prec().meta);
    let size =
        module.ckks_softmax_tmp_bytes(&sizing, &params.tsk_layout(), &params.atk_layout(), &plan);
    let mut scratch = ScratchOwned::<BE>::alloc(size);
    let tsk = gen_tsk(&params, module, &sk_raw, &mut scratch.borrow());
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
        .ckks_softmax(
            &mut res,
            &input,
            &plan,
            &tsk,
            &rotation_keys,
            &mut scratch.borrow(),
        )
        .expect("ckks_softmax");
    assert_eq!(input.log_budget() - res.log_budget(), consumed);

    let (re_out, _) = ckks_decrypt_decode::<BE, F, E>(
        &params,
        module,
        &encoder,
        &res,
        &sk,
        &mut scratch.borrow(),
    );
    assert_precision_bits("softmax", &re_out, &want, options.target_bits, params.n);
    let sum = re_out.iter().copied().fold(F::zero(), |a, b| a + b);
    assert_precision_bits("softmax/sum", &[sum], &[F::one()], 15.0, params.n);
}
