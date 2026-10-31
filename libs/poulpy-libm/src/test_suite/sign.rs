//! Tests for [`CKKSSignOps`] (sign, step) at a single modulus.

use poulpy_core::layouts::{
    GGLWEInfos, GLWETensorKeyPrepared, GLWEToBackendMut, GLWEToBackendRef, LWEInfos, prepared::GLWETensorKeyPreparedToBackendRef,
};
use poulpy_hal::{
    api::{ReimNew, ScratchOwnedAlloc, ScratchOwnedBorrow},
    layouts::{HostBytesBackend, Module, ScratchOwned, Standard},
    source::Source,
};

use poulpy_ckks::{
    CKKSCtBounds, CKKSInfos, GLWEPlaintextMeta, Scale, SetCKKSInfos,
    layouts::{CKKSCiphertext, CKKSCiphertextAlloc, CKKSPlaintext},
    polynomial::SplitStrategy,
    test_suite::{
        CKKSTestParams,
        helpers::{
            TestContextBackend, TestContextModule, TestScalar, ckks_decrypt_decode, ckks_encrypt_with_prec, ckks_spec, gen_atk,
            gen_sk_with_raw, gen_tsk, upload_pt,
        },
        reference_encoder::ReferenceEncoder,
    },
};

use crate::{
    comparison::DEFAULT_SIGN_COMPOSITE_CHEBYSHEV,
    sign::{CKKSSignOps, SignComposite},
};

use super::helpers::{assert_error, assert_precision_bits};

pub fn test_sign_composite<BE, F, E>(base: CKKSTestParams, module: &Module<BE>, host_module: &Module<HostBytesBackend>)
where
    BE: TestContextBackend<Ring = Standard>,
    Module<BE>: TestContextModule<BE> + CKKSSignOps<BE>,
    CKKSCiphertext<BE::OwnedBuf, BE::ZnxWord, BE::Ring>:
        GLWEToBackendMut<BE> + GLWEToBackendRef<BE> + CKKSCtBounds + SetCKKSInfos,
    CKKSPlaintext<BE::OwnedBuf, BE::ZnxWord, BE::Ring>: GLWEToBackendRef<BE> + LWEInfos + poulpy_core::layouts::IntPolyInfos,
    GLWETensorKeyPrepared<BE::OwnedBuf, BE>: GLWETensorKeyPreparedToBackendRef<BE> + GGLWEInfos,
    F: TestScalar,
    E: ReimNew<F>,
{
    let coeff_meta = ckks_spec(base.n, base.base2k, base.prec().log_delta(), base.base2k);
    assert_error(
        SignComposite::<F, _>::from_coeffs(base.base2k.into(), &[], coeff_meta, SplitStrategy::MinDepth, host_module),
        "sign composite: coefficient rows must be non-empty",
    );
    assert_error(
        SignComposite::<F, _>::from_minimax(
            F::zero(),
            15.0,
            &[15],
            12,
            base.base2k.into(),
            coeff_meta,
            SplitStrategy::MinDepth,
            host_module,
        ),
        "sign_composite_coeffs_with_margin: tau must lie in (0, 1)",
    );
    run_case::<BE, F, E>(base, module, host_module, false);
}

pub fn test_step_composite<BE, F, E>(base: CKKSTestParams, module: &Module<BE>, host_module: &Module<HostBytesBackend>)
where
    BE: TestContextBackend<Ring = Standard>,
    Module<BE>: TestContextModule<BE> + CKKSSignOps<BE>,
    CKKSCiphertext<BE::OwnedBuf, BE::ZnxWord, BE::Ring>:
        GLWEToBackendMut<BE> + GLWEToBackendRef<BE> + CKKSCtBounds + SetCKKSInfos,
    CKKSPlaintext<BE::OwnedBuf, BE::ZnxWord, BE::Ring>: GLWEToBackendRef<BE> + LWEInfos + poulpy_core::layouts::IntPolyInfos,
    GLWETensorKeyPrepared<BE::OwnedBuf, BE>: GLWETensorKeyPreparedToBackendRef<BE> + GGLWEInfos,
    F: TestScalar,
    E: ReimNew<F>,
{
    run_case::<BE, F, E>(base, module, host_module, true);
}

/// Sign evaluated with a composite **generated** by the Remez engine
/// ([`SignComposite::from_minimax`]) rather than the tabulated constants.
pub fn test_sign_minimax<BE, F, E>(base: CKKSTestParams, module: &Module<BE>, host_module: &Module<HostBytesBackend>)
where
    BE: TestContextBackend<Ring = Standard>,
    Module<BE>: TestContextModule<BE> + CKKSSignOps<BE>,
    CKKSCiphertext<BE::OwnedBuf, BE::ZnxWord, BE::Ring>:
        GLWEToBackendMut<BE> + GLWEToBackendRef<BE> + CKKSCtBounds + SetCKKSInfos,
    CKKSPlaintext<BE::OwnedBuf, BE::ZnxWord, BE::Ring>: GLWEToBackendRef<BE> + LWEInfos + poulpy_core::layouts::IntPolyInfos,
    GLWETensorKeyPrepared<BE::OwnedBuf, BE>: GLWETensorKeyPreparedToBackendRef<BE> + GGLWEInfos,
    F: TestScalar,
    E: ReimNew<F>,
{
    let log_delta = base.prec().log_delta();
    let coeff_meta = ckks_spec(base.n, base.base2k, log_delta, base.base2k);

    // Generate a composite resolving |x| ≥ tau to ~20 bits with degree-15 factors.
    let tau = 0.1_f64;
    let target_bits = 20.0;
    let host_composite = SignComposite::<F, _>::from_minimax(
        F::from_f64(tau).unwrap(),
        target_bits,
        &[15],
        12,
        base.base2k.into(),
        coeff_meta,
        SplitStrategy::MinDepth,
        host_module,
    )
    .expect("SignComposite::from_minimax");
    let consumed = host_composite.consumed_bits(log_delta);

    let dsize = base.dsize;
    let k = (consumed + log_delta + 2 * base.base2k).next_multiple_of(dsize * base.base2k);
    let params = CKKSTestParams {
        n: base.n,
        base2k: base.base2k,
        k,
        prec_meta: GLWEPlaintextMeta {
            log_sparsity: 0,
            scale: Scale::Log(log_delta),
            slots: Default::default(),
        },
        prec_log_budget: k - log_delta,
        hw: base.hw,
        dsize,
        rank: base.rank,
    };

    let slots = params.n / 2;
    let encoder = ReferenceEncoder::<E>::new::<F>(slots).unwrap();

    let mut source = Source::new([0x7cu8; 32]);
    let x: Vec<F> = (0..slots)
        .map(|_| {
            let mag = source.next_f64(tau, 1.0);
            let s = if source.next_f64(-1.0, 1.0) < 0.0 { -1.0 } else { 1.0 };
            F::from_f64(s * mag).unwrap()
        })
        .collect();
    let im = vec![F::zero(); slots];

    // The generated composite must itself approximate sign on |x| ≥ tau.
    let want: Vec<F> = x.iter().map(|&v| host_composite.evaluate(v)).collect();
    let ideal: Vec<F> = x.iter().map(|&v| if v >= F::zero() { F::one() } else { -F::one() }).collect();
    assert_precision_bits("sign_minimax/reference", &want, &ideal, target_bits - 2.0, params.n);
    let composite = host_composite.map_plaintexts(|pt| upload_pt(module, pt));

    let (sk_raw, sk) = gen_sk_with_raw(&params, module, host_module, [0u8; 32]);
    let mut sizing = module.ckks_ciphertext_alloc(params.base2k.into(), params.k.into());
    sizing.set_meta(params.prec().meta);
    let scratch_size = module.ckks_sign_tmp_bytes(&sizing, &sizing, &params.tsk_layout(), &params.atk_layout(), &coeff_meta);
    let mut scratch = ScratchOwned::<BE>::alloc(scratch_size);
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
        .ckks_sign(&mut res, &ct, &composite, &tsk, &conj_key, &mut scratch.borrow())
        .expect("ckks_sign");

    let (re_out, _) = ckks_decrypt_decode::<BE, F, E>(&params, module, &encoder, &res, &sk, &mut scratch.borrow());
    let required = log_delta as f64 - 15.0;
    assert_precision_bits("sign_minimax", &re_out, &want, required, params.n);
}

fn run_case<BE, F, E>(base: CKKSTestParams, module: &Module<BE>, host_module: &Module<HostBytesBackend>, step: bool)
where
    BE: TestContextBackend<Ring = Standard>,
    Module<BE>: TestContextModule<BE> + CKKSSignOps<BE>,
    CKKSCiphertext<BE::OwnedBuf, BE::ZnxWord, BE::Ring>:
        GLWEToBackendMut<BE> + GLWEToBackendRef<BE> + CKKSCtBounds + SetCKKSInfos,
    CKKSPlaintext<BE::OwnedBuf, BE::ZnxWord, BE::Ring>: GLWEToBackendRef<BE> + LWEInfos + poulpy_core::layouts::IntPolyInfos,
    GLWETensorKeyPrepared<BE::OwnedBuf, BE>: GLWETensorKeyPreparedToBackendRef<BE> + GGLWEInfos,
    F: TestScalar,
    E: ReimNew<F>,
{
    let label = if step { "step_composite" } else { "sign_composite" };
    let log_delta = base.prec().log_delta();
    let coeff_meta = ckks_spec(base.n, base.base2k, log_delta, base.base2k);

    let host_composite =
        SignComposite::<F, _>::from_default(base.base2k.into(), coeff_meta, SplitStrategy::MinDepth, host_module)
            .expect("SignComposite::from_default");
    let consumed = host_composite.consumed_bits(log_delta);

    let dsize = base.dsize;
    let k = (consumed + log_delta + 2 * base.base2k).next_multiple_of(dsize * base.base2k);
    let params = CKKSTestParams {
        n: base.n,
        base2k: base.base2k,
        k,
        prec_meta: GLWEPlaintextMeta {
            log_sparsity: 0,
            scale: Scale::Log(log_delta),
            slots: Default::default(),
        },
        prec_log_budget: k - log_delta,
        hw: base.hw,
        dsize,
        rank: base.rank,
    };

    let slots = params.n / 2;
    let encoder = ReferenceEncoder::<E>::new::<F>(slots).unwrap();

    // |x| >= 0.05: below the composite's resolution the sign is ill-defined.
    let mut source = Source::new([0x59u8; 32]);
    let mut x: Vec<F> = (0..slots)
        .map(|_| {
            let mag = source.next_f64(0.05, 1.0);
            let sign = if source.next_f64(-1.0, 1.0) < 0.0 { -1.0 } else { 1.0 };
            F::from_f64(sign * mag).unwrap()
        })
        .collect();
    x[0] = F::from_f64(1.0).unwrap();
    x[1] = F::from_f64(-1.0).unwrap();
    x[2] = F::from_f64(0.05).unwrap();
    x[3] = F::from_f64(-0.05).unwrap();
    let im = vec![F::zero(); slots];

    let composite = host_composite.map_plaintexts(|pt| upload_pt(module, pt));

    let (sk_raw, sk) = gen_sk_with_raw(&params, module, host_module, [0u8; 32]);

    let mut sizing = module.ckks_ciphertext_alloc(params.base2k.into(), params.k.into());
    sizing.set_meta(params.prec().meta);
    let scratch_size = module.ckks_sign_tmp_bytes(&sizing, &sizing, &params.tsk_layout(), &params.atk_layout(), &coeff_meta);
    let mut scratch = ScratchOwned::<BE>::alloc(scratch_size);
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

    let lb0 = ct.log_budget();
    let mut res = module.ckks_ciphertext_alloc(params.base2k.into(), params.k.into());
    if step {
        module
            .ckks_step(&mut res, &ct, &composite, &tsk, &conj_key, &mut scratch.borrow())
            .expect("ckks_step");
    } else {
        module
            .ckks_sign(&mut res, &ct, &composite, &tsk, &conj_key, &mut scratch.borrow())
            .expect("ckks_sign");
    }

    let out_log_delta = if step { log_delta + 1 } else { log_delta };
    assert_eq!(res.log_delta(), out_log_delta, "{label}: unexpected log_delta");
    assert_eq!(
        lb0 - res.log_budget(),
        consumed + step as usize,
        "{label}: consumed bits mismatch"
    );

    // Near +-1 every early factor amplifies the error about twentyfold, so the
    // composite's end-to-end error depends on the noise draw. Each factor is held
    // to the noise it adds instead: replayed through the same API as a one-row
    // composite, its output is compared with its polynomial applied in the clear
    // to the decrypted input of that factor.
    //
    // Floor: over 16 secret-key and input seeds on NTT4x30Portable at
    // log_delta = 40, the worst factor kept 29.3 bits on average and 25.0 on its
    // worst slot, against `log_delta - 15` = 25 and the helper's worst-slot floor
    // of 17: at least 4.3 and 8 bits of margin.
    let required = log_delta as f64 - 15.0;
    let half = F::from_f64(0.5).unwrap();
    let rows = DEFAULT_SIGN_COMPOSITE_CHEBYSHEV;
    let mut cur = ct;
    for j in 0..rows.len() {
        let last_step = step && j + 1 == rows.len();
        let host_factor = SignComposite::<F, _>::from_coeffs(
            base.base2k.into(),
            &rows[j..=j],
            coeff_meta,
            SplitStrategy::MinDepth,
            host_module,
        )
        .expect("one-row SignComposite");
        let (re_in, _) = ckks_decrypt_decode::<BE, F, E>(&params, module, &encoder, &cur, &sk, &mut scratch.borrow());
        let want: Vec<F> = re_in
            .iter()
            .map(|&v| {
                let s = host_factor.evaluate(v);
                if last_step { (s + F::one()) * half } else { s }
            })
            .collect();
        let factor = host_factor.map_plaintexts(|pt| upload_pt(module, pt));
        let mut next = module.ckks_ciphertext_alloc(params.base2k.into(), params.k.into());
        if last_step {
            module
                .ckks_step(&mut next, &cur, &factor, &tsk, &conj_key, &mut scratch.borrow())
                .expect("ckks_step");
        } else {
            module
                .ckks_sign(&mut next, &cur, &factor, &tsk, &conj_key, &mut scratch.borrow())
                .expect("ckks_sign");
        }
        let (re_out, im_out) = ckks_decrypt_decode::<BE, F, E>(&params, module, &encoder, &next, &sk, &mut scratch.borrow());
        assert_precision_bits(&format!("{label}/factor {j}"), &re_out, &want, required, params.n);
        assert_precision_bits(
            &format!("{label}/factor {j}/imag"),
            &im_out,
            &vec![F::zero(); slots],
            required,
            params.n,
        );
        cur = next;
    }

    // The composite is the composition of its factors: the same bits come out.
    let (re_out, im_out) = ckks_decrypt_decode::<BE, F, E>(&params, module, &encoder, &res, &sk, &mut scratch.borrow());
    let (re_rep, im_rep) = ckks_decrypt_decode::<BE, F, E>(&params, module, &encoder, &cur, &sk, &mut scratch.borrow());
    assert!(
        re_out == re_rep && im_out == im_rep,
        "{label}: differs from its factors evaluated one at a time"
    );
}
