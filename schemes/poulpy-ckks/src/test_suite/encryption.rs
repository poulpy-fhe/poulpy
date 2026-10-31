//! Encrypt / decrypt round-trip tests.

use super::helpers::{
    TestContextBackend, TestContextModule, TestScalar, alloc_scratch, assert_ckks_error, assert_ct_meta, assert_precision,
    assert_precision_for_log_delta, ckks_decrypt_decode, ckks_decrypt_with_prec, ckks_encrypt, ckks_encrypt_with_prec, ckks_spec,
    download_pt, gen_sk, quantized_slots, test_vector_1,
};
use super::parity::helpers::snapshot;
use super::parity::helpers::with_scratch;
use crate::CKKSModuleInfos;
use crate::Scale;
use crate::{
    CKKSCompositionError, CKKSInfos, CKKSLayout, GLWEPlaintextMeta, SetCKKSInfos,
    api::{CKKSDecryptOps, CKKSEncryptOps},
    layouts::{CKKSCiphertextAlloc, CKKSPlaintextAlloc},
};
use poulpy_core::{
    GLWEDecrypt, GLWENoise, GLWENormalize, GLWESub, GetDistribution,
    api::GLWESecretPreparedFactory,
    layouts::{GLWEInfos, GLWELayout, GLWEPlaintextAlloc, IntPolyInfos, LWEInfos},
};
use poulpy_hal::{
    api::{ModuleNew, ReimNew, ScratchOwnedAlloc, ScratchOwnedBorrow},
    layouts::{Backend, HostBytesBackend, Module, ScratchOwned, Standard},
    source::Source,
};

use crate::SlotsKind;
use crate::{test_suite::CKKSTestParams, test_suite::reference_encoder::ReferenceEncoder};

fn extract_src_prec(params: &CKKSTestParams) -> CKKSLayout {
    if params.base2k == 19 {
        ckks_spec(params.n, params.base2k, 40, 17)
    } else {
        ckks_spec(params.n, params.base2k, 40, 12)
    }
}

fn assert_decrypt_extract_success<BE, F, E>(
    label: &str,
    params: &CKKSTestParams,
    module: &Module<BE>,
    host_module: &Module<HostBytesBackend>,
    encoder: &ReferenceEncoder<E>,
    dst_prec: CKKSLayout,
) where
    BE: TestContextBackend<Ring = Standard>,
    for<'a> <BE as poulpy_hal::layouts::Backend>::BufRef<'a>: poulpy_hal::layouts::HostDataRef,
    for<'a> <BE as poulpy_hal::layouts::Backend>::BufMut<'a>: poulpy_hal::layouts::HostDataMut,
    Module<BE>: TestContextModule<BE>,
    F: TestScalar,
    E: ReimNew<F>,
    Module<BE>: CKKSDecryptOps<BE>,
{
    let src_prec = extract_src_prec(params);
    let sk = gen_sk(params, module, host_module, [0u8; 32]);
    let mut scratch = alloc_scratch(params, module);
    let m = params.n / 2;

    let (re1, im1) = test_vector_1::<F>(m);
    let ct = ckks_encrypt_with_prec(
        params,
        module,
        host_module,
        encoder,
        &sk,
        src_prec.k().as_usize(),
        &re1,
        &im1,
        src_prec,
        &mut scratch.borrow(),
    );
    assert_ct_meta(&format!("{label} src"), &ct, src_prec.log_delta(), src_prec.log_budget());

    let pt = ckks_decrypt_with_prec(module, &ct, &sk, dst_prec, &mut scratch.borrow()).unwrap();
    assert_eq!(pt.meta(), dst_prec.meta, "{label}: decrypt changed destination metadata");

    let mut re_out = vec![F::zero(); m];
    let mut im_out = vec![F::zero(); m];
    encoder.decode_reim(&pt, &mut re_out, &mut im_out).unwrap();

    let (want_prec, assert_log_delta) = if dst_prec.log_delta() > src_prec.log_delta() {
        (src_prec, src_prec.log_delta())
    } else {
        (dst_prec, dst_prec.log_delta())
    };
    let (want_re, want_im) = quantized_slots(host_module, encoder, params.base2k.into(), want_prec, &re1, &im1);
    assert_precision_for_log_delta(&format!("{label} re"), &re_out, &want_re, assert_log_delta, params.n);
    assert_precision_for_log_delta(&format!("{label} im"), &im_out, &want_im, assert_log_delta, params.n);
}

/// Verifies that encrypt → decrypt → decode recovers the original message.
pub fn test_encrypt_decrypt<BE, F, E>(params: CKKSTestParams, module: &Module<BE>, host_module: &Module<HostBytesBackend>)
where
    BE: TestContextBackend<Ring = Standard>,
    for<'a> <BE as poulpy_hal::layouts::Backend>::BufRef<'a>: poulpy_hal::layouts::HostDataRef,
    for<'a> <BE as poulpy_hal::layouts::Backend>::BufMut<'a>: poulpy_hal::layouts::HostDataMut,
    Module<BE>: TestContextModule<BE> + GLWENoise<BE>,
    F: TestScalar,
    E: ReimNew<F>,
{
    let m = params.n / 2;
    let encoder = ReferenceEncoder::<E>::new(m).unwrap();
    let (re1, im1) = test_vector_1::<F>(m);
    let sk = gen_sk(&params, module, host_module, [0u8; 32]);
    let mut scratch = alloc_scratch(&params, module);

    let ct = ckks_encrypt(
        &params,
        module,
        host_module,
        &encoder,
        &sk,
        params.k,
        &re1,
        &im1,
        &mut scratch.borrow(),
    );
    let metadata = ct.noise().expect("CKKS encryption must record provenance");
    assert_eq!(metadata.precision(), ct.k());
    assert_eq!(metadata.parties(), 1);
    assert_eq!(metadata.secret_distribution().base(), *sk.dist());
    assert_eq!(metadata.rank(), ct.rank().as_usize());
    assert_eq!(metadata.body().std_dev(), poulpy_core::DEFAULT_SIGMA_XE);
    assert!(metadata.components()[1..].iter().all(|component| component.variance() == 0.0));
    assert_eq!(
        metadata.phase_noise(ct.n().as_usize()).std_dev(),
        poulpy_core::DEFAULT_SIGMA_XE
    );
    let mut host_pt =
        host_module.ckks_plaintext_alloc_from_slots(host_module.ckks_max_slots(), params.base2k.into(), params.prec().k());
    host_pt.set_meta(params.prec().meta());
    encoder.encode_reim(&mut host_pt, &re1, &im1).unwrap();
    // CKKS aligns the encoded integer message to the ciphertext's torus grid.
    let mut message = vec![0i128; params.n];
    host_pt.inner.decode_vec_i128(&mut message, host_pt.encoded_k());
    let mut pt_want = module.glwe_plaintext_alloc_from_infos(&ct);
    pt_want.encode_vec_i128(&message, ct.k());
    let mut residual = module.glwe_plaintext_alloc_from_infos(&ct);
    let mut integer_error = vec![0i64; params.n];
    let mut noise_scratch = ScratchOwned::<BE>::alloc(module.glwe_noise_tmp_bytes(&ct));
    let mut measured = 0.0;
    let mut modeled = 0.0;
    for trial in 0..16 {
        let fresh;
        let ciphertext = if trial == 0 {
            &ct
        } else {
            fresh = ckks_encrypt(
                &params,
                module,
                host_module,
                &encoder,
                &sk,
                params.k,
                &re1,
                &im1,
                &mut scratch.borrow(),
            );
            &fresh
        };
        let noise = ciphertext.noise().unwrap();
        assert_eq!(noise.precision(), ciphertext.k());
        module.glwe_decrypt(ciphertext, &mut residual, &sk, &mut noise_scratch.borrow());
        module.glwe_sub_assign(&mut residual, &pt_want);
        module.glwe_normalize_assign(&mut residual, &mut noise_scratch.borrow());
        residual.decode_vec_i64(&mut integer_error, ciphertext.k());
        measured += integer_error.iter().map(|&e| (e as f64).powi(2)).sum::<f64>() / params.n as f64;
        modeled += noise.phase_noise(params.n).variance_at(ciphertext.k());
    }
    let ratio = measured / modeled;
    assert!((0.75..=1.25).contains(&ratio), "CKKS empirical/model second moment={ratio}");
    assert_ct_meta(
        "encrypt_decrypt",
        &ct,
        params.prec().log_delta(),
        params.k - params.prec().log_delta(),
    );
    let (re_out, im_out) = ckks_decrypt_decode::<BE, F, E>(&params, module, &encoder, &ct, &sk, &mut scratch.borrow());
    assert_precision_for_log_delta("encrypt_decrypt re", &re_out, &re1, ct.log_delta(), params.n);
    assert_precision_for_log_delta("encrypt_decrypt im", &im_out, &im1, ct.log_delta(), params.n);
}

/// CKKS returns a degree error without changing outputs or consuming randomness.
pub fn test_encrypt_sk_rejects_degree_mismatch<BE>()
where
    BE: Backend<ZnxWord = i64> + crate::oep::CKKSEncryptionImpl,
    Module<BE>: ModuleNew<BE> + GLWESecretPreparedFactory<BE>,
{
    let n = BE::MIN_DEGREE.max(16);
    let module = Module::<BE>::new(n as u64);
    let other = Module::<BE>::new((2 * n) as u64);
    let enc = GLWELayout {
        n: n.into(),
        base2k: 8usize.into(),
        k: 8usize.into(),
        rank: 1usize.into(),
    };
    let mut ct = module.ckks_ciphertext_alloc_from_glwe_infos(&enc);
    let mut pt = module.ckks_plaintext_alloc_from_slots(module.ckks_max_slots(), 8usize.into(), 8usize.into());
    let sk = other.glwe_secret_prepared_alloc(1usize.into());
    let ct_before = snapshot::<BE, _>(&ct);
    let pt_before = snapshot::<BE, _>(&pt);
    let mut xe = Source::new([1; 32]);
    let mut xa = Source::new([2; 32]);

    for call_module in [&module, &other] {
        let expected = |op| CKKSCompositionError::EncryptionDegreeMismatch {
            op,
            module_n: call_module.n(),
            ct_n: n,
            sk_n: 2 * n,
        };
        let err = with_scratch::<BE, _>(0, |s| call_module.ckks_decrypt(&mut pt, &ct, &sk, s)).unwrap_err();
        assert_ckks_error("decrypt_degree_mismatch", &err, expected("ckks_decrypt"));
        assert_eq!(pt_before, snapshot::<BE, _>(&pt));

        let err = with_scratch::<BE, _>(0, |s| call_module.ckks_encrypt_sk(&mut ct, &pt, &sk, &mut xe, &mut xa, s)).unwrap_err();
        assert_ckks_error("encrypt_degree_mismatch", &err, expected("ckks_encrypt_sk"));
        assert_eq!(ct_before, snapshot::<BE, _>(&ct));
    }

    // A plaintext scale above the ciphertext precision fails before encryption.
    let sk = module.glwe_secret_prepared_alloc(1usize.into());
    pt.set_log_delta(9);
    let err = with_scratch::<BE, _>(0, |s| module.ckks_encrypt_sk(&mut ct, &pt, &sk, &mut xe, &mut xa, s)).unwrap_err();
    assert_ckks_error(
        "encrypt_insufficient_budget",
        &err,
        CKKSCompositionError::InsufficientHomomorphicCapacity {
            op: "ckks_encrypt_sk",
            available_log_budget: 8,
            required_bits: 9,
        },
    );
    assert_eq!(ct_before, snapshot::<BE, _>(&ct));
    assert_eq!(xe.new_seed(), Source::new([1; 32]).new_seed());
    assert_eq!(xa.new_seed(), Source::new([2; 32]).new_seed());
}

pub fn test_decrypt_extract_same_meta<BE, F, E>(
    params: CKKSTestParams,
    module: &Module<BE>,
    host_module: &Module<HostBytesBackend>,
) where
    BE: TestContextBackend<Ring = Standard>,
    for<'a> <BE as poulpy_hal::layouts::Backend>::BufRef<'a>: poulpy_hal::layouts::HostDataRef,
    for<'a> <BE as poulpy_hal::layouts::Backend>::BufMut<'a>: poulpy_hal::layouts::HostDataMut,
    Module<BE>: TestContextModule<BE>,
    F: TestScalar,
    E: ReimNew<F>,
    Module<BE>: CKKSDecryptOps<BE>,
{
    let m = params.n / 2;
    let encoder = ReferenceEncoder::<E>::new(m).unwrap();
    assert_decrypt_extract_success(
        "decrypt_extract_same_meta",
        &params,
        module,
        host_module,
        &encoder,
        extract_src_prec(&params),
    );
}

pub fn test_decrypt_extract_truncates_log_budget<BE, F, E>(
    params: CKKSTestParams,
    module: &Module<BE>,
    host_module: &Module<HostBytesBackend>,
) where
    BE: TestContextBackend<Ring = Standard>,
    for<'a> <BE as poulpy_hal::layouts::Backend>::BufRef<'a>: poulpy_hal::layouts::HostDataRef,
    for<'a> <BE as poulpy_hal::layouts::Backend>::BufMut<'a>: poulpy_hal::layouts::HostDataMut,
    Module<BE>: TestContextModule<BE>,
    F: TestScalar,
    E: ReimNew<F>,
    Module<BE>: CKKSDecryptOps<BE>,
{
    let m = params.n / 2;
    let encoder = ReferenceEncoder::<E>::new(m).unwrap();
    let src_prec = extract_src_prec(&params);
    assert_decrypt_extract_success(
        "decrypt_extract_truncates_log_budget",
        &params,
        module,
        host_module,
        &encoder,
        ckks_spec(params.n, params.base2k, src_prec.log_delta(), 0),
    );
}

pub fn test_decrypt_extract_rsh_for_smaller_log_delta<BE, F, E>(
    params: CKKSTestParams,
    module: &Module<BE>,
    host_module: &Module<HostBytesBackend>,
) where
    BE: TestContextBackend<Ring = Standard>,
    for<'a> <BE as poulpy_hal::layouts::Backend>::BufRef<'a>: poulpy_hal::layouts::HostDataRef,
    for<'a> <BE as poulpy_hal::layouts::Backend>::BufMut<'a>: poulpy_hal::layouts::HostDataMut,
    Module<BE>: TestContextModule<BE>,
    F: TestScalar,
    E: ReimNew<F>,
    Module<BE>: CKKSDecryptOps<BE>,
{
    let m = params.n / 2;
    let encoder = ReferenceEncoder::<E>::new(m).unwrap();
    let src_prec = extract_src_prec(&params);
    assert_decrypt_extract_success(
        "decrypt_extract_rsh",
        &params,
        module,
        host_module,
        &encoder,
        ckks_spec(params.n, params.base2k, src_prec.log_delta() - 8, src_prec.log_budget()),
    );
}

pub fn test_decrypt_extract_lsh_for_larger_log_delta<BE, F, E>(
    params: CKKSTestParams,
    module: &Module<BE>,
    host_module: &Module<HostBytesBackend>,
) where
    BE: TestContextBackend<Ring = Standard>,
    for<'a> <BE as poulpy_hal::layouts::Backend>::BufRef<'a>: poulpy_hal::layouts::HostDataRef,
    for<'a> <BE as poulpy_hal::layouts::Backend>::BufMut<'a>: poulpy_hal::layouts::HostDataMut,
    Module<BE>: TestContextModule<BE>,
    F: TestScalar,
    E: ReimNew<F>,
    Module<BE>: CKKSDecryptOps<BE>,
{
    let m = params.n / 2;
    let encoder = ReferenceEncoder::<E>::new(m).unwrap();
    let src_prec = extract_src_prec(&params);
    assert_decrypt_extract_success(
        "decrypt_extract_lsh",
        &params,
        module,
        host_module,
        &encoder,
        ckks_spec(params.n, params.base2k, src_prec.log_delta(), src_prec.log_budget() - 8),
    );
}

pub fn test_decrypt_rejects_output_hom_rem_too_large<BE, F, E>(
    params: CKKSTestParams,
    module: &Module<BE>,
    host_module: &Module<HostBytesBackend>,
) where
    BE: TestContextBackend<Ring = Standard>,
    for<'a> <BE as poulpy_hal::layouts::Backend>::BufRef<'a>: poulpy_hal::layouts::HostDataRef,
    for<'a> <BE as poulpy_hal::layouts::Backend>::BufMut<'a>: poulpy_hal::layouts::HostDataMut,
    Module<BE>: TestContextModule<BE>,
    F: TestScalar,
    E: ReimNew<F>,
{
    let m = params.n / 2;
    let encoder = ReferenceEncoder::<E>::new(m).unwrap();
    let src_prec = extract_src_prec(&params);
    let sk = gen_sk(&params, module, host_module, [0u8; 32]);
    let mut scratch = alloc_scratch(&params, module);

    let (re1, im1) = test_vector_1::<F>(m);
    let ct = ckks_encrypt_with_prec(
        &params,
        module,
        host_module,
        &encoder,
        &sk,
        src_prec.k().as_usize(),
        &re1,
        &im1,
        src_prec,
        &mut scratch.borrow(),
    );
    let mut pt = module.ckks_plaintext_alloc_from_slots(
        module.ckks_max_slots(),
        params.base2k.into(),
        (src_prec.log_delta() + src_prec.log_budget() + 1).into(),
    );
    pt.set_meta(GLWEPlaintextMeta {
        scale: Scale::Log(src_prec.log_delta()),
        slots: SlotsKind::Complex,
        log_sparsity: 0,
    });
    let (pt_before, ct_before) = (snapshot::<BE, _>(&pt), snapshot::<BE, _>(&ct));
    let err = with_scratch::<BE, _>(0, |s| module.ckks_decrypt(&mut pt, &ct, &sk, s)).unwrap_err();
    assert_ckks_error(
        "decrypt_rejects_output_hom_rem_too_large",
        &err,
        CKKSCompositionError::PlaintextAlignmentImpossible {
            op: "ckks_decrypt",
            ct_log_budget: src_prec.log_budget(),
            pt_log_delta: src_prec.log_delta(),
            // Alignment is checked against the output plaintext's effective
            // precision `log_delta + log_budget`, not its physical `max_k`.
            pt_k: src_prec.log_delta() + src_prec.log_budget() + 1,
        },
    );
    assert_eq!(snapshot::<BE, _>(&pt), pt_before, "a refused ckks_decrypt wrote res");
    assert_eq!(snapshot::<BE, _>(&ct), ct_before, "a refused ckks_decrypt changed its source");
}

pub fn test_decrypt_rejects_base2k_mismatch<BE, F, E>(
    params: CKKSTestParams,
    module: &Module<BE>,
    host_module: &Module<HostBytesBackend>,
) where
    BE: TestContextBackend<Ring = Standard>,
    for<'a> <BE as poulpy_hal::layouts::Backend>::BufRef<'a>: poulpy_hal::layouts::HostDataRef,
    for<'a> <BE as poulpy_hal::layouts::Backend>::BufMut<'a>: poulpy_hal::layouts::HostDataMut,
    Module<BE>: TestContextModule<BE>,
    F: TestScalar,
    E: ReimNew<F>,
{
    let m = params.n / 2;
    let encoder = ReferenceEncoder::<E>::new(m).unwrap();
    let src_prec = extract_src_prec(&params);
    let sk = gen_sk(&params, module, host_module, [0u8; 32]);
    let mut scratch = alloc_scratch(&params, module);

    let (re1, im1) = test_vector_1::<F>(m);
    let ct = ckks_encrypt_with_prec(
        &params,
        module,
        host_module,
        &encoder,
        &sk,
        src_prec.k().as_usize(),
        &re1,
        &im1,
        src_prec,
        &mut scratch.borrow(),
    );
    let mismatched_base2k = (params.base2k / 2).into();
    let mut pt = host_module.ckks_plaintext_alloc_from_slots(host_module.ckks_max_slots(), mismatched_base2k, src_prec.k());
    pt.set_meta(src_prec.meta());
    let (pt_before, ct_before) = (snapshot::<BE, _>(&pt), snapshot::<BE, _>(&ct));
    let err = with_scratch::<BE, _>(0, |s| module.ckks_decrypt(&mut pt, &ct, &sk, s)).unwrap_err();
    assert_ckks_error(
        "decrypt_extract_base2k_mismatch",
        &err,
        CKKSCompositionError::PlaintextBase2KMismatch {
            op: "ckks_decrypt",
            ct_base2k: params.base2k,
            pt_base2k: mismatched_base2k.as_usize(),
        },
    );
    assert_eq!(snapshot::<BE, _>(&pt), pt_before, "a refused ckks_decrypt wrote res");
    assert_eq!(snapshot::<BE, _>(&ct), ct_before, "a refused ckks_decrypt changed its source");
}

/// Decrypting a dense ciphertext into a plaintext of half its degree is a `PlaintextDegreeMismatch`; a ciphertext
/// whose slots repeat every `N / 4` decrypts into it, every second coefficient of its polynomial.
pub fn test_decrypt_rejects_degree_mismatch<BE, F, E>(
    params: CKKSTestParams,
    module: &Module<BE>,
    host_module: &Module<HostBytesBackend>,
) where
    BE: TestContextBackend<Ring = Standard>,
    for<'a> <BE as poulpy_hal::layouts::Backend>::BufRef<'a>: poulpy_hal::layouts::HostDataRef,
    for<'a> <BE as poulpy_hal::layouts::Backend>::BufMut<'a>: poulpy_hal::layouts::HostDataMut,
    Module<BE>: TestContextModule<BE>,
    F: TestScalar,
    E: ReimNew<F>,
{
    let m = params.n / 2;
    let encoder = ReferenceEncoder::<E>::new(m).unwrap();
    let src_prec = extract_src_prec(&params);
    let sk = gen_sk(&params, module, host_module, [0u8; 32]);
    let mut scratch = alloc_scratch(&params, module);

    let (re1, im1) = test_vector_1::<F>(m);
    let ct = ckks_encrypt_with_prec(
        &params,
        module,
        host_module,
        &encoder,
        &sk,
        src_prec.k().as_usize(),
        &re1,
        &im1,
        src_prec,
        &mut scratch.borrow(),
    );
    let mut pt = host_module.ckks_plaintext_alloc_from_slots(m / 2, params.base2k.into(), src_prec.k());
    pt.set_meta(src_prec.meta());
    let (pt_before, ct_before) = (snapshot::<BE, _>(&pt), snapshot::<BE, _>(&ct));
    let err = with_scratch::<BE, _>(0, |s| module.ckks_decrypt(&mut pt, &ct, &sk, s)).unwrap_err();
    assert_ckks_error(
        "decrypt_extract_degree_mismatch",
        &err,
        CKKSCompositionError::PlaintextDegreeMismatch {
            op: "ckks_decrypt",
            ct_n: params.n,
            pt_n: params.n / 2,
        },
    );
    assert_eq!(snapshot::<BE, _>(&pt), pt_before, "a refused ckks_decrypt wrote res");
    assert_eq!(snapshot::<BE, _>(&ct), ct_before, "a refused ckks_decrypt changed its source");

    let half = ReferenceEncoder::<E>::new(m / 2).unwrap();
    let (re2, im2) = test_vector_1::<F>(m / 2);
    let mut sparse_prec = src_prec;
    sparse_prec.meta.log_sparsity = 1;
    let ct = ckks_encrypt_with_prec(
        &params,
        module,
        host_module,
        &half,
        &sk,
        src_prec.k().as_usize(),
        &re2,
        &im2,
        sparse_prec,
        &mut scratch.borrow(),
    );
    let mut pt = module.ckks_plaintext_alloc_from_slots(m / 2, params.base2k.into(), src_prec.k());
    pt.set_meta(src_prec.meta());
    with_scratch::<BE, _>(module.ckks_decrypt_tmp_bytes(&pt, &ct), |s| {
        module.ckks_decrypt(&mut pt, &ct, &sk, s)
    })
    .unwrap();
    let (mut re_out, mut im_out) = (vec![F::zero(); m / 2], vec![F::zero(); m / 2]);
    half.decode_reim(&download_pt::<BE>(&pt), &mut re_out, &mut im_out).unwrap();
    assert_precision(
        "decrypt into half the degree re",
        &re_out,
        &re2,
        src_prec.log_delta(),
        params.n,
    );
    assert_precision(
        "decrypt into half the degree im",
        &im_out,
        &im2,
        src_prec.log_delta(),
        params.n,
    );
}

/// `ckks_decrypt` leaves no byte of its query in the arena, into a plaintext of the ciphertext's degree and into one of
/// half its degree.
pub fn test_decrypt_wipes_scratch<BE, F, E>(
    params: crate::test_suite::CKKSTestParams,
    module: &poulpy_hal::layouts::Module<BE>,
    host_module: &poulpy_hal::layouts::Module<poulpy_hal::layouts::HostBytesBackend>,
) where
    BE: crate::test_suite::helpers::TestContextBackend<Ring = poulpy_hal::layouts::Standard>,
    poulpy_hal::layouts::Module<BE>: crate::test_suite::helpers::TestContextModule<BE>,
    F: crate::test_suite::helpers::TestScalar,
    E: ReimNew<F>,
{
    let m = params.n / 2;
    let encoder = ReferenceEncoder::<E>::new(m).unwrap();
    let sk = gen_sk(&params, module, host_module, [0u8; 32]);
    let mut scratch = alloc_scratch(&params, module);
    let (re, im) = test_vector_1::<F>(m);
    let mut ct = ckks_encrypt(
        &params,
        module,
        host_module,
        &encoder,
        &sk,
        params.k,
        &re,
        &im,
        &mut scratch.borrow(),
    );
    // Half the degree reads every second coefficient, which a sparsity of one allows.
    poulpy_core::layouts::SetGLWEPlaintextInfos::set_log_sparsity(&mut ct, 1);
    for slots in [module.ckks_max_slots(), module.ckks_max_slots() / 2] {
        let mut pt = module.ckks_plaintext_alloc_from_slots(slots, params.base2k.into(), ct.k());
        pt.set_meta(ct.meta());
        let bytes = module.ckks_decrypt_tmp_bytes(&pt, &ct);
        poulpy_core::test_suite::assert_wipes_scratch::<BE>(bytes, |s| module.ckks_decrypt(&mut pt, &ct, &sk, s).unwrap());
    }
}

/// A plaintext wider than the ciphertext's budget aligns is `PlaintextAlignmentImpossible` of `ckks_decrypt` before
/// the secret-derived decryption, on an empty arena and on the query, the plaintext unchanged.
pub fn test_decrypt_rejects_alignment_on_empty_arena<BE, F, E>(
    params: crate::test_suite::CKKSTestParams,
    module: &poulpy_hal::layouts::Module<BE>,
    host_module: &poulpy_hal::layouts::Module<poulpy_hal::layouts::HostBytesBackend>,
) where
    BE: crate::test_suite::helpers::TestContextBackend<Ring = poulpy_hal::layouts::Standard>,
    poulpy_hal::layouts::Module<BE>: crate::test_suite::helpers::TestContextModule<BE>,
    F: crate::test_suite::helpers::TestScalar,
    E: ReimNew<F>,
{
    use super::parity::{helpers::fixture_ciphertext, layout};
    let b = params.base2k;
    let sk = gen_sk(&params, module, host_module, [0u8; 32]);
    let ct = fixture_ciphertext(
        module,
        &layout(params, params.rank, 3 * b + 5, b - 1, 0, SlotsKind::Complex),
        41,
    );
    let mut pt = module.ckks_plaintext_alloc_from_slots(module.ckks_max_slots(), b.into(), (3 * b + 6).into());
    pt.set_meta(GLWEPlaintextMeta {
        scale: Scale::Log(b - 1),
        slots: SlotsKind::Complex,
        log_sparsity: 0,
    });
    for bytes in [0, module.ckks_decrypt_tmp_bytes(&pt, &ct)] {
        let before = snapshot::<BE, _>(&pt);
        let err = with_scratch::<BE, _>(bytes, |s| module.ckks_decrypt(&mut pt, &ct, &sk, s)).unwrap_err();
        assert_ckks_error(
            "decrypt: plaintext wider than the budget",
            &err,
            CKKSCompositionError::PlaintextAlignmentImpossible {
                op: "ckks_decrypt",
                ct_log_budget: 2 * b + 6,
                pt_log_delta: b - 1,
                pt_k: 3 * b + 6,
            },
        );
        assert_eq!(snapshot::<BE, _>(&pt), before, "a refused decryption wrote the plaintext");
    }
}

/// A plaintext of another radix is `PlaintextBase2KMismatch` of `ckks_encrypt_sk` before the first draw, on an empty
/// arena and on the query, the ciphertext and both sources unchanged.
pub fn test_encrypt_sk_rejects_plaintext_base2k_mismatch<BE, F, E>(
    params: crate::test_suite::CKKSTestParams,
    module: &poulpy_hal::layouts::Module<BE>,
    host_module: &poulpy_hal::layouts::Module<poulpy_hal::layouts::HostBytesBackend>,
) where
    BE: crate::test_suite::helpers::TestContextBackend<Ring = poulpy_hal::layouts::Standard>,
    poulpy_hal::layouts::Module<BE>: crate::test_suite::helpers::TestContextModule<BE>,
    F: crate::test_suite::helpers::TestScalar,
    E: ReimNew<F>,
{
    use super::parity::{
        helpers::{fixture_ciphertext, fixture_plaintext},
        layout,
    };
    let b = params.base2k;
    let sk = gen_sk(&params, module, host_module, [0u8; 32]);
    let mut ct = fixture_ciphertext(
        module,
        &layout(params, params.rank, 3 * b + 5, b - 1, 0, SlotsKind::Complex),
        41,
    );
    let pt_layout = layout(params, 0, b + 3, b - 1, 0, SlotsKind::Complex);
    let pt = fixture_plaintext(
        module,
        &CKKSLayout {
            glwe_layout: GLWELayout {
                base2k: (b - 1).into(),
                ..pt_layout.glwe_layout
            },
            meta: pt_layout.meta,
        },
        43,
    );
    for bytes in [0, module.ckks_encrypt_sk_tmp_bytes(&ct)] {
        let before = snapshot::<BE, _>(&ct);
        let (mut source_xe, mut source_xa) = (Source::new([5u8; 32]), Source::new([6u8; 32]));
        let err = with_scratch::<BE, _>(bytes, |s| {
            module.ckks_encrypt_sk(&mut ct, &pt, &sk, &mut source_xe, &mut source_xa, s)
        })
        .unwrap_err();
        assert_ckks_error(
            "encrypt_sk: plaintext of another radix",
            &err,
            CKKSCompositionError::PlaintextBase2KMismatch {
                op: "ckks_encrypt_sk",
                ct_base2k: b,
                pt_base2k: b - 1,
            },
        );
        assert_eq!(snapshot::<BE, _>(&ct), before, "a refused encryption wrote the ciphertext");
        assert_eq!(
            source_xe.new_seed(),
            Source::new([5u8; 32]).new_seed(),
            "a refused encryption drew from source_xe"
        );
        assert_eq!(
            source_xa.new_seed(),
            Source::new([6u8; 32]).new_seed(),
            "a refused encryption drew from source_xa"
        );
    }
}
