//! Tests for smooth special functions.

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

use crate::special::{
    CKKSBesselOps, CKKSGammaOps, J0Plan, J1Plan, JnPlan, LgammaPlan, SpecialOptions, TgammaPlan,
    Y0Plan, Y1Plan, YnPlan,
};

use super::helpers::{assert_error, assert_precision_bits, params_for};

pub fn test_special_functions<BE, F, E>(
    base: CKKSTestParams,
    module: &Module<BE>,
    host_module: &Module<HostBytesBackend>,
) where
    BE: TestContextBackend,
    Module<BE>: TestContextModule<BE> + CKKSBesselOps<BE> + CKKSGammaOps<BE>,
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
    let options = SpecialOptions::default();
    let g0 = F::one();
    let g1 = F::from_f64(3.0).unwrap();
    let j0 = F::from_f64(-4.0).unwrap();
    let j1 = F::from_f64(4.0).unwrap();
    let y0 = F::one();
    let y1 = F::from_f64(4.0).unwrap();
    let host_tgamma =
        TgammaPlan::from_precision(g0, g1, base.base2k.into(), coeff_meta, options, host_module)
            .expect("TgammaPlan::from_precision");
    let host_lgamma =
        LgammaPlan::from_precision(g0, g1, base.base2k.into(), coeff_meta, options, host_module)
            .expect("LgammaPlan::from_precision");
    let host_j0 =
        J0Plan::from_precision(j0, j1, base.base2k.into(), coeff_meta, options, host_module)
            .expect("J0Plan::from_precision");
    let host_j1 =
        J1Plan::from_precision(j0, j1, base.base2k.into(), coeff_meta, options, host_module)
            .expect("J1Plan::from_precision");
    let host_jn = JnPlan::from_precision(
        3,
        j0,
        j1,
        base.base2k.into(),
        coeff_meta,
        options,
        host_module,
    )
    .expect("JnPlan::from_precision");
    let host_y0 =
        Y0Plan::from_precision(y0, y1, base.base2k.into(), coeff_meta, options, host_module)
            .expect("Y0Plan::from_precision");
    let host_y1 =
        Y1Plan::from_precision(y0, y1, base.base2k.into(), coeff_meta, options, host_module)
            .expect("Y1Plan::from_precision");
    let host_yn = YnPlan::from_precision(
        2,
        y0,
        y1,
        base.base2k.into(),
        coeff_meta,
        options,
        host_module,
    )
    .expect("YnPlan::from_precision");
    assert_error(
        TgammaPlan::from_precision(
            F::zero(),
            g1,
            base.base2k.into(),
            coeff_meta,
            options,
            host_module,
        ),
        "tgamma: interval must be positive",
    );
    assert_error(
        Y0Plan::from_precision(
            F::zero(),
            y1,
            base.base2k.into(),
            coeff_meta,
            options,
            host_module,
        ),
        "y0: interval must be positive",
    );

    let consumed = [
        host_tgamma.consumed_bits(log_delta),
        host_lgamma.consumed_bits(log_delta),
        host_j0.consumed_bits(log_delta),
        host_j1.consumed_bits(log_delta),
        host_jn.consumed_bits(log_delta),
        host_y0.consumed_bits(log_delta),
        host_y1.consumed_bits(log_delta),
        host_yn.consumed_bits(log_delta),
    ];
    let max_consumed = *consumed.iter().max().unwrap();
    let tgamma = host_tgamma.map_plaintexts(|pt| upload_pt(module, pt));
    let lgamma = host_lgamma.map_plaintexts(|pt| upload_pt(module, pt));
    let jp0 = host_j0.map_plaintexts(|pt| upload_pt(module, pt));
    let jp1 = host_j1.map_plaintexts(|pt| upload_pt(module, pt));
    let jpn = host_jn.map_plaintexts(|pt| upload_pt(module, pt));
    let yp0 = host_y0.map_plaintexts(|pt| upload_pt(module, pt));
    let yp1 = host_y1.map_plaintexts(|pt| upload_pt(module, pt));
    let ypn = host_yn.map_plaintexts(|pt| upload_pt(module, pt));
    let params = params_for(&base, max_consumed);
    let slots = params.n / 2;
    let encoder = ReferenceEncoder::<E>::new::<F>(slots).unwrap();
    let mut source = Source::new([0xe1u8; 32]);
    let gx: Vec<F> = (0..slots)
        .map(|_| F::from_f64(source.next_f64(1.0, 3.0)).unwrap())
        .collect();
    let jx: Vec<F> = (0..slots)
        .map(|_| F::from_f64(source.next_f64(-4.0, 4.0)).unwrap())
        .collect();
    let yx: Vec<F> = (0..slots)
        .map(|_| F::from_f64(source.next_f64(1.0, 4.0)).unwrap())
        .collect();
    let im = vec![F::zero(); slots];
    let (sk_raw, sk) = gen_sk_with_raw(&params, module, host_module, [0xe2u8; 32]);
    let mut sizing = module.ckks_ciphertext_alloc(params.base2k.into(), params.k.into());
    sizing.set_meta(params.prec().meta);
    let size = module
        .ckks_tgamma_tmp_bytes(&sizing, &params.tsk_layout(), &tgamma)
        .max(module.ckks_lgamma_tmp_bytes(&sizing, &params.tsk_layout(), &lgamma))
        .max(module.ckks_j0_tmp_bytes(&sizing, &params.tsk_layout(), &jp0))
        .max(module.ckks_j1_tmp_bytes(&sizing, &params.tsk_layout(), &jp1))
        .max(module.ckks_jn_tmp_bytes(&sizing, &params.tsk_layout(), &jpn))
        .max(module.ckks_y0_tmp_bytes(&sizing, &params.tsk_layout(), &yp0))
        .max(module.ckks_y1_tmp_bytes(&sizing, &params.tsk_layout(), &yp1))
        .max(module.ckks_yn_tmp_bytes(&sizing, &params.tsk_layout(), &ypn));
    let mut scratch = ScratchOwned::<BE>::alloc(size);
    let tsk = gen_tsk(&params, module, &sk_raw, &mut scratch.borrow());
    let cg = ckks_encrypt_with_prec(
        &params,
        module,
        host_module,
        &encoder,
        &sk,
        params.k,
        &gx,
        &im,
        params.prec(),
        &mut scratch.borrow(),
    );
    let cj = ckks_encrypt_with_prec(
        &params,
        module,
        host_module,
        &encoder,
        &sk,
        params.k,
        &jx,
        &im,
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
        &yx,
        &im,
        params.prec(),
        &mut scratch.borrow(),
    );

    for (op, &expected_consumed) in consumed.iter().enumerate() {
        #[allow(clippy::type_complexity)]
        let (label, input, x, want): (
            &str,
            &CKKSCiphertext<BE::OwnedBuf, BE::ZnxWord>,
            &[F],
            Vec<F>,
        ) = match op {
            0 => (
                "tgamma",
                &cg,
                &gx,
                gx.iter()
                    .map(|v| F::from_f64(libm::tgamma(v.to_f64().unwrap())).unwrap())
                    .collect(),
            ),
            1 => (
                "lgamma",
                &cg,
                &gx,
                gx.iter()
                    .map(|v| F::from_f64(libm::lgamma(v.to_f64().unwrap())).unwrap())
                    .collect(),
            ),
            2 => (
                "j0",
                &cj,
                &jx,
                jx.iter()
                    .map(|v| F::from_f64(libm::j0(v.to_f64().unwrap())).unwrap())
                    .collect(),
            ),
            3 => (
                "j1",
                &cj,
                &jx,
                jx.iter()
                    .map(|v| F::from_f64(libm::j1(v.to_f64().unwrap())).unwrap())
                    .collect(),
            ),
            4 => (
                "jn",
                &cj,
                &jx,
                jx.iter()
                    .map(|v| F::from_f64(libm::jn(3, v.to_f64().unwrap())).unwrap())
                    .collect(),
            ),
            5 => (
                "y0",
                &cy,
                &yx,
                yx.iter()
                    .map(|v| F::from_f64(libm::y0(v.to_f64().unwrap())).unwrap())
                    .collect(),
            ),
            6 => (
                "y1",
                &cy,
                &yx,
                yx.iter()
                    .map(|v| F::from_f64(libm::y1(v.to_f64().unwrap())).unwrap())
                    .collect(),
            ),
            _ => (
                "yn",
                &cy,
                &yx,
                yx.iter()
                    .map(|v| F::from_f64(libm::yn(2, v.to_f64().unwrap())).unwrap())
                    .collect(),
            ),
        };
        assert_eq!(x.len(), slots);
        let result = match op {
            0 => module.ckks_tgamma(&mut sizing, input, &tgamma, &tsk, &mut scratch.borrow()),
            1 => module.ckks_lgamma(&mut sizing, input, &lgamma, &tsk, &mut scratch.borrow()),
            2 => module.ckks_j0(&mut sizing, input, &jp0, &tsk, &mut scratch.borrow()),
            3 => module.ckks_j1(&mut sizing, input, &jp1, &tsk, &mut scratch.borrow()),
            4 => module.ckks_jn(&mut sizing, input, &jpn, &tsk, &mut scratch.borrow()),
            5 => module.ckks_y0(&mut sizing, input, &yp0, &tsk, &mut scratch.borrow()),
            6 => module.ckks_y1(&mut sizing, input, &yp1, &tsk, &mut scratch.borrow()),
            _ => module.ckks_yn(&mut sizing, input, &ypn, &tsk, &mut scratch.borrow()),
        };
        result.unwrap_or_else(|e| panic!("ckks_{label}: {e}"));
        assert_eq!(
            input.log_budget() - sizing.log_budget(),
            expected_consumed,
            "{label}"
        );
        let (re_out, _) = ckks_decrypt_decode::<BE, F, E>(
            &params,
            module,
            &encoder,
            &sizing,
            &sk,
            &mut scratch.borrow(),
        );
        assert_precision_bits(label, &re_out, &want, options.target_bits, params.n);
    }
}
