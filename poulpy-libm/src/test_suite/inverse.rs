//! Tests for the iterative reciprocal / rsqrt ops
//! ([`CKKSInverseOps`] and [`CKKSInverseDomainOps`]) at a single modulus.

use poulpy_core::layouts::{
    GGLWEInfos, GLWETensorKeyPrepared, GLWEToBackendMut, GLWEToBackendRef, LWEInfos,
    prepared::GLWETensorKeyPreparedToBackendRef,
};
use poulpy_hal::{
    api::{NegacyclicFFT, NegacyclicFFTNew, ScratchOwnedAlloc, ScratchOwnedBorrow},
    layouts::{HostBytesBackend, Module, ScratchOwned},
};

use poulpy_ckks::{
    CKKSCtBounds, CKKSInfos, CKKSMeta, SetCKKSInfos,
    api::{CKKSAllOpsTmpBytes, CKKSMulOps},
    layouts::{CKKSCiphertext, CKKSModuleAlloc, CKKSPlaintext},
    polynomial::SplitStrategy,
    test_suite::reference_encoder::ReferenceEncoder,
    test_suite::{
        CKKSTestParams,
        helpers::{
            PT_PREC, TestContextBackend, TestContextModule, TestScalar, ckks_decrypt_decode,
            ckks_encrypt_with_prec, ckks_spec, gen_atk, gen_sk_with_raw, gen_tsk, upload_pt,
        },
    },
};

use crate::iterative::{CKKSInverseDomainOps, CKKSInverseOps};
use crate::range::IntervalNorm;
use crate::sign::SignComposite;

use super::helpers::{assert_precision_bits, params_for_with_headroom, sample_interval};

fn alloc_inverse_scratch<BE>(params: &CKKSTestParams, module: &Module<BE>) -> ScratchOwned<BE>
where
    BE: TestContextBackend,
    Module<BE>: TestContextModule<BE> + CKKSInverseOps<BE>,
    ScratchOwned<BE>: ScratchOwnedAlloc<BE>,
{
    let mut ct = module.ckks_ciphertext_alloc(params.base2k.into(), params.k.into());
    ct.set_meta(params.prec().meta);
    let size = module
        .ckks_all_ops_tmp_bytes(&ct, &params.tsk_layout(), &PT_PREC)
        .max(module.ckks_inverse_tmp_bytes(&ct, &params.tsk_layout()));
    ScratchOwned::<BE>::alloc(size)
}

pub fn test_goldschmidt_division<BE, F, E>(
    base: CKKSTestParams,
    module: &Module<BE>,
    host_module: &Module<HostBytesBackend>,
) where
    BE: TestContextBackend,
    Module<BE>: TestContextModule<BE> + CKKSInverseOps<BE> + CKKSInverseDomainOps<BE>,
    CKKSCiphertext<BE::OwnedBuf, BE::ZnxWord>:
        GLWEToBackendMut<BE> + GLWEToBackendRef<BE> + CKKSCtBounds + SetCKKSInfos,
    GLWETensorKeyPrepared<BE::OwnedBuf, BE>: GLWETensorKeyPreparedToBackendRef<BE> + GGLWEInfos,
    F: TestScalar,
    E: NegacyclicFFT<F> + NegacyclicFFTNew<F>,
{
    let log_delta = base.prec_meta.log_delta;
    let iters = 6usize;
    let params = params_for_with_headroom(&base, (iters + 1) * log_delta, log_delta);

    let slots = params.n / 2;
    let encoder = ReferenceEncoder::<E>::new::<F>(slots).unwrap();
    let x = sample_interval::<F>(slots, 0.5, 1.5, 0x51);
    let im = vec![F::zero(); slots];

    let (sk_raw, sk) = gen_sk_with_raw(&params, module, host_module, [0u8; 32]);
    let mut scratch = alloc_inverse_scratch(&params, module);
    let tsk = gen_tsk(&params, module, &sk_raw, &mut scratch.borrow());

    let want: Vec<F> = x.iter().map(|&v| F::one() / v).collect();
    let required = log_delta as f64 - 12.0;
    for (positive_alias, label) in [(false, "goldschmidt"), (true, "inverse_positive_domain")] {
        let mut ct = ckks_encrypt_with_prec(
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
        let input_budget = ct.log_budget();
        if positive_alias {
            module.ckks_inverse_positive_domain(&mut ct, iters, &tsk, &mut scratch.borrow())
        } else {
            module.ckks_goldschmidt_division(&mut ct, iters, &tsk, &mut scratch.borrow())
        }
        .unwrap_or_else(|e| panic!("{label}: {e}"));
        assert_eq!(ct.log_delta(), log_delta);
        assert_eq!(input_budget - ct.log_budget(), (iters + 1) * log_delta);
        let (re_out, _) = ckks_decrypt_decode::<BE, F, E>(
            &params,
            module,
            &encoder,
            &ct,
            &sk,
            &mut scratch.borrow(),
        );
        assert_precision_bits(label, &re_out, &want, required, params.n);
    }
}

pub fn test_rsqrt<BE, F, E>(
    base: CKKSTestParams,
    module: &Module<BE>,
    host_module: &Module<HostBytesBackend>,
) where
    BE: TestContextBackend,
    Module<BE>: TestContextModule<BE> + CKKSInverseOps<BE>,
    CKKSCiphertext<BE::OwnedBuf, BE::ZnxWord>:
        GLWEToBackendMut<BE> + GLWEToBackendRef<BE> + CKKSCtBounds + SetCKKSInfos,
    GLWETensorKeyPrepared<BE::OwnedBuf, BE>: GLWETensorKeyPreparedToBackendRef<BE> + GGLWEInfos,
    F: TestScalar,
    E: NegacyclicFFT<F> + NegacyclicFFTNew<F>,
{
    let log_delta = base.prec_meta.log_delta;
    let r = 5usize; // interval [0.9, 1.1]: x is already a good initial guess.
    let params = params_for_with_headroom(&base, 2 * r * log_delta, log_delta);

    let slots = params.n / 2;
    let encoder = ReferenceEncoder::<E>::new::<F>(slots).unwrap();
    let x = sample_interval::<F>(slots, 0.9, 1.1, 0x1b);
    let x_half: Vec<F> = x.iter().map(|&v| v * F::from_f64(0.5).unwrap()).collect();
    let im = vec![F::zero(); slots];

    let (sk_raw, sk) = gen_sk_with_raw(&params, module, host_module, [0u8; 32]);
    let mut scratch = alloc_inverse_scratch(&params, module);
    let tsk = gen_tsk(&params, module, &sk_raw, &mut scratch.borrow());

    let mut y = ckks_encrypt_with_prec(
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

    let lb0 = y.log_budget();
    module
        .ckks_rsqrt(&mut y, &in_half, r, &tsk, &mut scratch.borrow())
        .expect("ckks_rsqrt");

    assert_eq!(y.log_delta(), log_delta, "rsqrt: log_delta preserved");
    assert_eq!(
        lb0 - y.log_budget(),
        2 * r * log_delta,
        "rsqrt: consumed bits mismatch"
    );

    let (re_out, _) =
        ckks_decrypt_decode::<BE, F, E>(&params, module, &encoder, &y, &sk, &mut scratch.borrow());
    let want: Vec<F> = x.iter().map(|&v| F::one() / v.sqrt()).collect();

    let required = log_delta as f64 - 15.0;
    assert_precision_bits("rsqrt", &re_out, &want, required, params.n);
}

pub fn test_inverse_negative_domain<BE, F, E>(
    base: CKKSTestParams,
    module: &Module<BE>,
    host_module: &Module<HostBytesBackend>,
) where
    BE: TestContextBackend,
    Module<BE>: TestContextModule<BE> + CKKSInverseDomainOps<BE>,
    CKKSCiphertext<BE::OwnedBuf, BE::ZnxWord>:
        GLWEToBackendMut<BE> + GLWEToBackendRef<BE> + CKKSCtBounds + SetCKKSInfos,
    GLWETensorKeyPrepared<BE::OwnedBuf, BE>: GLWETensorKeyPreparedToBackendRef<BE> + GGLWEInfos,
    F: TestScalar,
    E: NegacyclicFFT<F> + NegacyclicFFTNew<F>,
{
    let log_delta = base.prec_meta.log_delta;
    let iters = 6usize;
    let params = params_for_with_headroom(&base, (iters + 1) * log_delta, log_delta);

    let slots = params.n / 2;
    let encoder = ReferenceEncoder::<E>::new::<F>(slots).unwrap();
    let x = sample_interval::<F>(slots, -1.5, -0.5, 0x2d);
    let im = vec![F::zero(); slots];

    let (sk_raw, sk) = gen_sk_with_raw(&params, module, host_module, [0u8; 32]);
    let mut scratch = alloc_inverse_scratch(&params, module);
    let tsk = gen_tsk(&params, module, &sk_raw, &mut scratch.borrow());

    let mut ct = ckks_encrypt_with_prec(
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
    let input_budget = ct.log_budget();
    module
        .ckks_inverse_negative_domain(&mut ct, iters, &tsk, &mut scratch.borrow())
        .expect("ckks_inverse_negative_domain");
    assert_eq!(
        ct.log_delta(),
        log_delta,
        "inverse_negative_domain: log_delta"
    );
    assert_eq!(input_budget - ct.log_budget(), (iters + 1) * log_delta);

    let (re_out, _) =
        ckks_decrypt_decode::<BE, F, E>(&params, module, &encoder, &ct, &sk, &mut scratch.borrow());
    let want: Vec<F> = x.iter().map(|&v| F::one() / v).collect();
    let required = log_delta as f64 - 12.0;
    assert_precision_bits(
        "inverse_negative_domain",
        &re_out,
        &want,
        required,
        params.n,
    );
}

pub fn test_inverse_full_domain<BE, F, E>(
    base: CKKSTestParams,
    module: &Module<BE>,
    host_module: &Module<HostBytesBackend>,
) where
    BE: TestContextBackend,
    Module<BE>: TestContextModule<BE> + CKKSInverseDomainOps<BE>,
    CKKSCiphertext<BE::OwnedBuf, BE::ZnxWord>:
        GLWEToBackendMut<BE> + GLWEToBackendRef<BE> + CKKSCtBounds + SetCKKSInfos,
    CKKSPlaintext<BE::OwnedBuf, BE::ZnxWord>:
        GLWEToBackendRef<BE> + LWEInfos + poulpy_core::layouts::IntPolyInfos,
    GLWETensorKeyPrepared<BE::OwnedBuf, BE>: GLWETensorKeyPreparedToBackendRef<BE> + GGLWEInfos,
    F: TestScalar,
    E: NegacyclicFFT<F> + NegacyclicFFTNew<F>,
{
    let log_delta = base.prec_meta.log_delta;
    let iters = 6usize;
    let coeff_meta = ckks_spec(base.n, base.base2k, log_delta, base.base2k);
    let host_composite = SignComposite::<F, _>::from_default(
        base.base2k.into(),
        coeff_meta,
        SplitStrategy::MinDepth,
        host_module,
    )
    .expect("SignComposite::from_default");
    let consumed = host_composite.consumed_bits(log_delta) + (iters + 3) * log_delta;
    let composite = host_composite.map_plaintexts(|pt| upload_pt(module, pt));

    let dsize = base.dsize;
    let k = (consumed + log_delta + 2 * base.base2k).next_multiple_of(dsize * base.base2k);
    let params = CKKSTestParams {
        n: base.n,
        base2k: base.base2k,
        k,
        prec_meta: CKKSMeta {
            log_sparsity: 0,
            log_delta,
            slots: Default::default(),
        },
        prec_log_budget: k - log_delta,
        hw: base.hw,
        dsize,
        rank: base.rank,
    };

    let slots = params.n / 2;
    let encoder = ReferenceEncoder::<E>::new::<F>(slots).unwrap();

    // |x| ∈ [0.6, 0.9] band, alternating sign: sign needs [−1, 1], and a
    // conditioned band keeps the deep composite stable on the noisier fft64.
    let x: Vec<F> = (0..slots)
        .map(|i| {
            let mag = 0.6 + 0.3 * (i as f64 / slots as f64);
            let s = if i % 2 == 0 { 1.0 } else { -1.0 };
            F::from_f64(s * mag).unwrap()
        })
        .collect();
    let im = vec![F::zero(); slots];

    let (sk_raw, sk) = gen_sk_with_raw(&params, module, host_module, [0u8; 32]);
    let mut sizing = module.ckks_ciphertext_alloc(params.base2k.into(), params.k.into());
    sizing.set_meta(params.prec().meta);
    let scratch_size = module.ckks_inverse_domain_tmp_bytes(
        &sizing,
        &params.tsk_layout(),
        &params.atk_layout(),
        &coeff_meta,
    );
    let mut scratch = ScratchOwned::<BE>::alloc(scratch_size);
    let tsk = gen_tsk(&params, module, &sk_raw, &mut scratch.borrow());
    let conj_key = gen_atk(&params, module, -1, &sk_raw, &mut scratch.borrow());

    let mut ct = ckks_encrypt_with_prec(
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
    let input_budget = ct.log_budget();
    module
        .ckks_inverse_full_domain(
            &mut ct,
            iters,
            &composite,
            &tsk,
            &conj_key,
            &mut scratch.borrow(),
        )
        .expect("ckks_inverse_full_domain");
    assert_eq!(ct.log_delta(), log_delta, "inverse_full_domain: log_delta");
    assert_eq!(input_budget - ct.log_budget(), consumed);

    let (re_out, _) =
        ckks_decrypt_decode::<BE, F, E>(&params, module, &encoder, &ct, &sk, &mut scratch.borrow());
    let want: Vec<F> = x.iter().map(|&v| F::one() / v).collect();
    let required = log_delta as f64 - 15.0;
    assert_precision_bits("inverse_full_domain", &re_out, &want, required, params.n);
}

pub fn test_interval_normalization<BE, F, E>(
    base: CKKSTestParams,
    module: &Module<BE>,
    host_module: &Module<HostBytesBackend>,
) where
    BE: TestContextBackend,
    Module<BE>: TestContextModule<BE> + CKKSInverseOps<BE> + CKKSInverseDomainOps<BE>,
    CKKSCiphertext<BE::OwnedBuf, BE::ZnxWord>:
        GLWEToBackendMut<BE> + GLWEToBackendRef<BE> + CKKSCtBounds + SetCKKSInfos,
    CKKSPlaintext<BE::OwnedBuf, BE::ZnxWord>:
        GLWEToBackendRef<BE> + LWEInfos + poulpy_core::layouts::IntPolyInfos,
    GLWETensorKeyPrepared<BE::OwnedBuf, BE>: GLWETensorKeyPreparedToBackendRef<BE> + GGLWEInfos,
    F: TestScalar,
    E: NegacyclicFFT<F> + NegacyclicFFTNew<F>,
{
    let log_delta = base.prec_meta.log_delta;
    let max = 8.0;
    let iters = 6usize;
    let coeff_meta = ckks_spec(base.n, base.base2k, log_delta, base.base2k);
    super::helpers::assert_error(
        IntervalNorm::from_max::<F>(f64::NAN, base.base2k.into(), coeff_meta, host_module),
        "interval_norm: max must be positive and finite",
    );
    let host_norm = IntervalNorm::from_max::<F>(max, base.base2k.into(), coeff_meta, host_module)
        .expect("IntervalNorm::from_max");
    let norm_consumed = host_norm.consumed_bits(log_delta);
    let consumed = norm_consumed + (iters + 2) * log_delta;
    let norm_params = host_norm.map_plaintexts(|pt| upload_pt(module, pt));

    let params = params_for_with_headroom(&base, consumed, log_delta);
    let slots = params.n / 2;
    let encoder = ReferenceEncoder::<E>::new::<F>(slots).unwrap();
    // Full range [0.5, Max]: values below 1 are kept, above 1 compressed toward 1.
    let x = sample_interval::<F>(slots, 0.5, max, 0x41);
    let im = vec![F::zero(); slots];

    let (sk_raw, sk) = gen_sk_with_raw(&params, module, host_module, [0u8; 32]);
    let mut sizing = module.ckks_ciphertext_alloc(params.base2k.into(), params.k.into());
    sizing.set_meta(params.prec().meta);
    let scratch_size = module.ckks_inverse_domain_tmp_bytes(
        &sizing,
        &params.tsk_layout(),
        &params.atk_layout(),
        &coeff_meta,
    );
    let mut scratch = ScratchOwned::<BE>::alloc(scratch_size);
    let tsk = gen_tsk(&params, module, &sk_raw, &mut scratch.borrow());

    let mut ct = ckks_encrypt_with_prec(
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

    // 1/x = norm · (1 / (x · norm)).
    let mut norm = module.ckks_ciphertext_alloc(params.base2k.into(), params.k.into());
    let input_budget = ct.log_budget();
    module
        .ckks_interval_normalization(
            &mut norm,
            &mut ct,
            &norm_params,
            &tsk,
            &mut scratch.borrow(),
        )
        .expect("ckks_interval_normalization");
    assert_eq!(
        ct.log_delta(),
        log_delta,
        "interval_normalization: log_delta"
    );
    assert_eq!(input_budget - ct.log_budget(), norm_consumed);
    module
        .ckks_goldschmidt_division(&mut ct, iters, &tsk, &mut scratch.borrow())
        .expect("ckks_goldschmidt_division");
    module
        .ckks_mul_assign(&mut ct, &norm, &tsk, &mut scratch.borrow())
        .expect("mul norm");

    let (re_out, _) =
        ckks_decrypt_decode::<BE, F, E>(&params, module, &encoder, &ct, &sk, &mut scratch.borrow());
    let want: Vec<F> = x.iter().map(|&v| F::one() / v).collect();
    let required = log_delta as f64 - 15.0;
    assert_precision_bits("interval_normalization", &re_out, &want, required, params.n);
}
