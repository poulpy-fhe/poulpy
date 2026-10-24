//! Tests for error functions.

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

use crate::erf::{CKKSErfOps, ErfOptions, ErfPlan, ErfcPlan};

use super::helpers::{assert_precision_bits, params_for};

pub fn test_erf_family<BE, F, E>(
    base: CKKSTestParams,
    module: &Module<BE>,
    host_module: &Module<HostBytesBackend>,
) where
    BE: TestContextBackend,
    Module<BE>: TestContextModule<BE> + CKKSErfOps<BE>,
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
    let options = ErfOptions::default();
    let bound = F::from_f64(3.0).unwrap();
    let host_erf = ErfPlan::from_precision(
        -bound,
        bound,
        base.base2k.into(),
        coeff_meta,
        options,
        host_module,
    )
    .expect("ErfPlan::from_precision");
    let host_erfc = ErfcPlan::from_precision(
        -bound,
        bound,
        base.base2k.into(),
        coeff_meta,
        options,
        host_module,
    )
    .expect("ErfcPlan::from_precision");
    let max_consumed = host_erf
        .consumed_bits(log_delta)
        .max(host_erfc.consumed_bits(log_delta));
    let erf = host_erf.map_plaintexts(|pt| upload_pt(module, pt));
    let erfc = host_erfc.map_plaintexts(|pt| upload_pt(module, pt));
    let params = params_for(&base, max_consumed);
    let slots = params.n / 2;
    let encoder = ReferenceEncoder::<E>::new::<F>(slots).unwrap();
    let mut source = Source::new([0xa5u8; 32]);
    let mut x: Vec<F> = (0..slots)
        .map(|_| F::from_f64(source.next_f64(-3.0, 3.0)).unwrap())
        .collect();
    x[0] = -bound;
    x[1] = F::zero();
    x[2] = bound;
    let im = vec![F::zero(); slots];
    let (sk_raw, sk) = gen_sk_with_raw(&params, module, host_module, [0xa6u8; 32]);
    let mut sizing = module.ckks_ciphertext_alloc(params.base2k.into(), params.k.into());
    sizing.set_meta(params.prec().meta);
    let size = module
        .ckks_erf_tmp_bytes(&sizing, &params.tsk_layout(), &erf)
        .max(module.ckks_erfc_tmp_bytes(&sizing, &params.tsk_layout(), &erfc));
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

    for (op, label) in [(0, "erf"), (1, "erfc")] {
        let want: Vec<F> = if op == 0 {
            x.iter()
                .map(|&v| F::from_f64(libm::erf(v.to_f64().unwrap())).unwrap())
                .collect()
        } else {
            x.iter()
                .map(|&v| F::from_f64(libm::erfc(v.to_f64().unwrap())).unwrap())
                .collect()
        };
        let consumed = if op == 0 {
            erf.consumed_bits(log_delta)
        } else {
            erfc.consumed_bits(log_delta)
        };
        let mut res = module.ckks_ciphertext_alloc(params.base2k.into(), params.k.into());
        if op == 0 {
            module.ckks_erf(&mut res, &input, &erf, &tsk, &mut scratch.borrow())
        } else {
            module.ckks_erfc(&mut res, &input, &erfc, &tsk, &mut scratch.borrow())
        }
        .unwrap_or_else(|e| panic!("ckks_{label}: {e}"));
        assert_eq!(input.log_budget() - res.log_budget(), consumed);

        let (re_out, _) = ckks_decrypt_decode::<BE, F, E>(
            &params,
            module,
            &encoder,
            &res,
            &sk,
            &mut scratch.borrow(),
        );
        assert_precision_bits(label, &re_out, &want, options.target_bits, params.n);
    }
}
