//! Free-function test helpers for the CKKS test suite.
//!
//! Provides trait aliases ([`TestContextBackend`], [`TestContextModule`],
//! [`TestContextHostModule`]), test-vector generators, "want" functions for
//! expected values, key-generation helpers, encode/upload/download utilities,
//! encrypt/decrypt wrappers, and precision/metadata assertion helpers.
//!
//! Each test function is expected to be self-contained: it takes
//! `(params, module, host_module)`, generates its own keys, encodes test
//! vectors as host-side [`CKKSPlaintext<Vec<u8>>`](CKKSPlaintext), uploads
//! them to the backend, performs the operation, downloads, and asserts
//! correctness.

use std::{f64::consts::TAU, fmt::Debug};

use crate::{
    CKKSCompositionError, CKKSInfos, CKKSMeta, SetCKKSInfos,
    api::{
        CKKSAddManyOps, CKKSAddOps, CKKSAffineOps, CKKSAllOpsTmpBytes, CKKSConjugateOps, CKKSCopyOps, CKKSDotProductOps,
        CKKSImagOps, CKKSMulAddOps, CKKSMulOps, CKKSMulSubOps, CKKSNegOps, CKKSPlaintextVecOps, CKKSPow2Ops, CKKSRescaleOps,
        CKKSRotateOps, CKKSScaleManage, CKKSSubOps,
    },
    encoding::reim::Encoder,
    layouts::{
        CKKSCiphertext, CKKSModuleAlloc, CKKSNormalizationState, CKKSPlaintextVecHostCodec,
        ciphertext::{CKKSMaintainOpsDefault, CKKSOffset},
        plaintext::CKKSPlaintext,
    },
    leveled::api::{CKKSDecrypt, CKKSEncrypt},
};
use poulpy_core::{
    EncryptionLayout, GLWEAutomorphism, GLWEAutomorphismKeyEncryptSk, GLWEDecrypt, GLWENormalize, GLWESub,
    GLWETensorKeyEncryptSk, ModuleTransfer, ScratchArenaTakeCore,
    layouts::{
        BackendGLWESecret, Base2K, GLWEAutomorphismKeyPrepared, GLWEAutomorphismKeyPreparedFactory, GLWESecretPreparedFactory,
        GLWETensorKeyPrepared, GLWETensorKeyPreparedFactory, LWEInfos, ModuleCoreAlloc, prepared::GLWESecretPrepared,
    },
};
use rand_distr::num_traits::{Float, FloatConst, FromPrimitive, ToPrimitive};

use poulpy_hal::{
    api::{ModuleNew, NegacyclicFFT, ScratchOwnedAlloc},
    layouts::{
        Backend, Data, GaloisElement, HostBackend, HostBytesBackend, HostDataMut, HostDataRef, Module, ScratchArena,
        ScratchOwned, TransferFrom,
    },
    source::Source,
};

use super::CKKSTestParams;

// ─── constants ───────────────────────────────────────────────────────────────

/// Default plaintext precision used in all tests.
pub const PT_PREC: CKKSMeta = CKKSMeta {
    log_sparsity: 0,
    log_delta: 8,
    log_budget: 10,
};

/// Fixed real and imaginary constants used in add/sub constant tests.
pub const ADD_SUB_CONST: (f64, f64) = (0.314_159_265_358_979_3, -0.271_828_182_845_904_5);

/// Fixed real and imaginary constants used in multiply-constant tests.
pub const MUL_CONST: (f64, f64) = (0.271_828_182_845_904_5, -0.141_421_356_237_309_5);

// ─── trait aliases ────────────────────────────────────────────────────────────

/// Backend bound for the CKKS test suite.
pub trait TestContextBackend:
    Backend<OwnedBuf = Vec<u8>> + HostBackend + TransferFrom<HostBytesBackend> + Send + Sync + 'static
where
    ScratchOwned<Self>: ScratchOwnedAlloc<Self>,
    for<'a> ScratchArena<'a, Self>: ScratchArenaTakeCore<'a, Self>,
{
}

impl<BE> TestContextBackend for BE
where
    BE: Backend<OwnedBuf = Vec<u8>> + HostBackend + TransferFrom<HostBytesBackend> + Send + Sync + 'static,
    ScratchOwned<BE>: ScratchOwnedAlloc<BE>,
{
}

/// Aggregates all `Module<BE>` capabilities needed by the CKKS test suite.
#[allow(private_bounds)]
pub trait TestContextModule<BE: Backend>:
    ModuleNew<BE>
    + ModuleCoreAlloc<OwnedBuf = BE::OwnedBuf>
    + CKKSModuleAlloc<BE>
    + CKKSMaintainOpsDefault<BE>
    + CKKSAllOpsTmpBytes<BE>
    + CKKSEncrypt<BE>
    + CKKSDecrypt<BE>
    + CKKSAddOps<BE>
    + CKKSSubOps<BE>
    + CKKSMulOps<BE>
    + CKKSNegOps<BE>
    + CKKSCopyOps<BE>
    + CKKSRotateOps<BE>
    + CKKSConjugateOps<BE>
    + CKKSImagOps<BE>
    + CKKSPow2Ops<BE>
    + CKKSRescaleOps<BE>
    + CKKSScaleManage<BE>
    + CKKSPlaintextVecOps<BE>
    + CKKSAddManyOps<BE>
    + CKKSMulAddOps<BE>
    + CKKSMulSubOps<BE>
    + CKKSAffineOps<BE>
    + CKKSDotProductOps<BE>
    + GLWEAutomorphism<BE>
    + GLWEDecrypt<BE>
    + GLWENormalize<BE>
    + GLWESub<BE>
    + GLWESecretPreparedFactory<BE>
    + GLWETensorKeyPreparedFactory<BE>
    + GLWEAutomorphismKeyPreparedFactory<BE>
    + GLWETensorKeyEncryptSk<BE>
    + GLWEAutomorphismKeyEncryptSk<BE>
    + GaloisElement
{
}

impl<BE: Backend, M> TestContextModule<BE> for M where
    M: ModuleNew<BE>
        + ModuleCoreAlloc<OwnedBuf = BE::OwnedBuf>
        + CKKSModuleAlloc<BE>
        + CKKSMaintainOpsDefault<BE>
        + CKKSAllOpsTmpBytes<BE>
        + CKKSEncrypt<BE>
        + CKKSDecrypt<BE>
        + CKKSAddOps<BE>
        + CKKSSubOps<BE>
        + CKKSMulOps<BE>
        + CKKSNegOps<BE>
        + CKKSCopyOps<BE>
        + CKKSRotateOps<BE>
        + CKKSConjugateOps<BE>
        + CKKSImagOps<BE>
        + CKKSPow2Ops<BE>
        + CKKSRescaleOps<BE>
        + CKKSScaleManage<BE>
        + CKKSPlaintextVecOps<BE>
        + CKKSAddManyOps<BE>
        + CKKSMulAddOps<BE>
        + CKKSMulSubOps<BE>
        + CKKSAffineOps<BE>
        + CKKSDotProductOps<BE>
        + GLWEAutomorphism<BE>
        + GLWEDecrypt<BE>
        + GLWENormalize<BE>
        + GLWESub<BE>
        + GLWESecretPreparedFactory<BE>
        + GLWETensorKeyPreparedFactory<BE>
        + GLWEAutomorphismKeyPreparedFactory<BE>
        + GLWETensorKeyEncryptSk<BE>
        + GLWEAutomorphismKeyEncryptSk<BE>
        + GaloisElement
{
}

/// Aggregates all `Module<HostBytesBackend>` capabilities needed by the CKKS
/// test suite.
pub trait TestContextHostModule: ModuleNew<HostBytesBackend> + CKKSModuleAlloc<HostBytesBackend> {}

impl<M: ModuleNew<HostBytesBackend> + CKKSModuleAlloc<HostBytesBackend>> TestContextHostModule for M {}

// ─── scalar + test-vector marker ─────────────────────────────────────────────

pub trait TestScalar: Copy + Float + FloatConst + FromPrimitive + ToPrimitive + std::fmt::Debug + Send + Sync + 'static {}

impl<T> TestScalar for T where T: Copy + Float + FloatConst + FromPrimitive + ToPrimitive + std::fmt::Debug + Send + Sync + 'static
{}

#[derive(Clone, Copy)]
pub enum TestVector {
    First,
    Second,
}

// ─── test-vector generators ──────────────────────────────────────────────────

fn to_scalar<F: TestScalar>(x: f64) -> F {
    F::from_f64(x).expect("f64 → scalar conversion should succeed")
}

/// Generates test vector 1: cos/sin wave at frequency 1/(2m).
///
/// `re[i] = cos(2π(i+0.25)/m)`, `im[i] = sin(2π(i+0.25)/m)`.
pub fn test_vector_1<F: TestScalar>(m: usize) -> (Vec<F>, Vec<F>) {
    let tau = to_scalar::<F>(TAU);
    let quarter = to_scalar::<F>(0.25);
    let m_f = F::from_usize(m).expect("usize → scalar");
    let re = (0..m)
        .map(|i| {
            let i_f = F::from_usize(i).expect("usize → scalar");
            (tau * (i_f + quarter) / m_f).cos()
        })
        .collect();
    let im = (0..m)
        .map(|i| {
            let i_f = F::from_usize(i).expect("usize → scalar");
            (tau * (i_f + quarter) / m_f).sin()
        })
        .collect();
    (re, im)
}

/// Generates test vector 2: cos/sin wave at frequency 5/(4m).
///
/// `re[i] = cos(2π(5i+3)/(2m))`, `im[i] = sin(2π(5i+3)/(2m))`.
pub fn test_vector_2<F: TestScalar>(m: usize) -> (Vec<F>, Vec<F>) {
    let tau = to_scalar::<F>(TAU);
    let five = to_scalar::<F>(5.0);
    let three = to_scalar::<F>(3.0);
    let two = to_scalar::<F>(2.0);
    let m_f = F::from_usize(m).expect("usize → scalar");
    let re = (0..m)
        .map(|i| {
            let i_f = F::from_usize(i).expect("usize → scalar");
            (tau * (five * i_f + three) / (two * m_f)).cos()
        })
        .collect();
    let im = (0..m)
        .map(|i| {
            let i_f = F::from_usize(i).expect("usize → scalar");
            (tau * (five * i_f + three) / (two * m_f)).sin()
        })
        .collect();
    (re, im)
}

// ─── "want" expected-value functions ─────────────────────────────────────────

pub fn want_add<F: Float>(a_re: &[F], a_im: &[F], b_re: &[F], b_im: &[F]) -> (Vec<F>, Vec<F>) {
    let re = a_re.iter().zip(b_re).map(|(a, b)| *a + *b).collect();
    let im = a_im.iter().zip(b_im).map(|(a, b)| *a + *b).collect();
    (re, im)
}

pub fn want_sub<F: Float>(a_re: &[F], a_im: &[F], b_re: &[F], b_im: &[F]) -> (Vec<F>, Vec<F>) {
    let re = a_re.iter().zip(b_re).map(|(a, b)| *a - *b).collect();
    let im = a_im.iter().zip(b_im).map(|(a, b)| *a - *b).collect();
    (re, im)
}

pub fn want_neg<F: Float>(re: &[F], im: &[F]) -> (Vec<F>, Vec<F>) {
    (re.iter().map(|x| -*x).collect(), im.iter().map(|x| -*x).collect())
}

pub fn want_add_const<F: Float>(a_re: &[F], a_im: &[F], c_re: F, c_im: F) -> (Vec<F>, Vec<F>) {
    let re = a_re.iter().map(|x| *x + c_re).collect();
    let im = a_im.iter().map(|x| *x + c_im).collect();
    (re, im)
}

pub fn want_mul_const<F: Float>(a_re: &[F], a_im: &[F], c_re: F, c_im: F) -> (Vec<F>, Vec<F>) {
    let mut re = Vec::with_capacity(a_re.len());
    let mut im = Vec::with_capacity(a_im.len());
    for i in 0..a_re.len() {
        re.push(a_re[i] * c_re - a_im[i] * c_im);
        im.push(a_re[i] * c_im + a_im[i] * c_re);
    }
    (re, im)
}

pub fn want_mul<F: Float>(a_re: &[F], a_im: &[F], b_re: &[F], b_im: &[F]) -> (Vec<F>, Vec<F>) {
    let m = a_re.len();
    let mut re = Vec::with_capacity(m);
    let mut im = Vec::with_capacity(m);
    for i in 0..m {
        re.push(a_re[i] * b_re[i] - a_im[i] * b_im[i]);
        im.push(a_re[i] * b_im[i] + b_re[i] * a_im[i]);
    }
    (re, im)
}

pub fn want_square<F: Float>(re_in: &[F], im_in: &[F]) -> (Vec<F>, Vec<F>) {
    let two = F::from(2.0).unwrap();
    let m = re_in.len();
    let mut re = Vec::with_capacity(m);
    let mut im = Vec::with_capacity(m);
    for i in 0..m {
        re.push(re_in[i] * re_in[i] - im_in[i] * im_in[i]);
        im.push(two * re_in[i] * im_in[i]);
    }
    (re, im)
}

pub fn want_conjugate<F: Float>(re: &[F], im: &[F]) -> (Vec<F>, Vec<F>) {
    (re.to_vec(), im.iter().map(|x| -*x).collect())
}

pub fn want_rotate<F: Float + Copy>(re: &[F], im: &[F], k: i64, m: usize) -> (Vec<F>, Vec<F>) {
    let re_rot = (0..m).map(|j| re[((j as i64 + k).rem_euclid(m as i64)) as usize]).collect();
    let im_rot = (0..m).map(|j| im[((j as i64 + k).rem_euclid(m as i64)) as usize]).collect();
    (re_rot, im_rot)
}

pub fn want_mul_pow2<F: TestScalar>(re: &[F], im: &[F], bits: usize) -> (Vec<F>, Vec<F>) {
    let scale = to_scalar::<F>((1u64 << bits) as f64);
    (
        re.iter().map(|x| *x * scale).collect(),
        im.iter().map(|x| *x * scale).collect(),
    )
}

pub fn want_div_pow2<F: TestScalar>(re: &[F], im: &[F], bits: usize) -> (Vec<F>, Vec<F>) {
    let scale = to_scalar::<F>((1u64 << bits) as f64);
    (
        re.iter().map(|x| *x / scale).collect(),
        im.iter().map(|x| *x / scale).collect(),
    )
}

pub fn want_mul_i<F: Float>(re: &[F], im: &[F]) -> (Vec<F>, Vec<F>) {
    (im.iter().map(|x| -*x).collect(), re.to_vec())
}

pub fn want_div_i<F: Float>(re: &[F], im: &[F]) -> (Vec<F>, Vec<F>) {
    (im.to_vec(), re.iter().map(|x| -*x).collect())
}

pub fn scale_slots<F: TestScalar>(re: &[F], im: &[F], bits: isize) -> (Vec<F>, Vec<F>) {
    let scale = to_scalar::<F>(2.0_f64).powi(bits as i32);
    (
        re.iter().map(|x| *x * scale).collect(),
        im.iter().map(|x| *x * scale).collect(),
    )
}

// ─── constant quantization ────────────────────────────────────────────────────

/// Rounds `(re, im)` to the nearest multiple of `2^-log_delta`.
pub fn quantized_const<F: TestScalar>(re: f64, im: f64, log_delta: usize) -> (F, F) {
    let scale = to_scalar::<F>(2.0_f64).powi(log_delta as i32);
    let re = (to_scalar::<F>(re) * scale).round() / scale;
    let im = (to_scalar::<F>(im) * scale).round() / scale;
    (re, im)
}

// ─── slot quantization ────────────────────────────────────────────────────────

/// Encodes then immediately decodes `(re, im)` to obtain quantized slots.
pub fn quantized_slots<F: TestScalar, E: NegacyclicFFT<F>>(
    host_module: &Module<HostBytesBackend>,
    encoder: &Encoder<E>,
    base2k: Base2K,
    prec: CKKSMeta,
    re: &[F],
    im: &[F],
) -> (Vec<F>, Vec<F>)
where
    Module<HostBytesBackend>: TestContextHostModule,
{
    let mut pt = host_module.ckks_pt_vec_alloc(base2k, prec);
    encoder.encode_reim(&mut pt, re, im).unwrap();
    let m = re.len();
    let mut re_out = vec![F::zero(); m];
    let mut im_out = vec![F::zero(); m];
    encoder.decode_reim(&pt, &mut re_out, &mut im_out).unwrap();
    (re_out, im_out)
}

/// Returns the quantized slots of a test vector scaled to `log_delta`.
pub fn quantized_vector<F: TestScalar, E: NegacyclicFFT<F>>(
    host_module: &Module<HostBytesBackend>,
    encoder: &Encoder<E>,
    params: &CKKSTestParams,
    which: TestVector,
    log_delta: usize,
) -> (Vec<F>, Vec<F>)
where
    Module<HostBytesBackend>: TestContextHostModule,
{
    let m = params.n / 2;
    let (re, im) = match which {
        TestVector::First => test_vector_1::<F>(m),
        TestVector::Second => test_vector_2::<F>(m),
    };
    let scale = to_scalar::<F>(2.0_f64).powi((log_delta as isize - params.prec.log_delta as isize) as i32);
    let re_scaled: Vec<F> = re.iter().map(|x| *x * scale).collect();
    let im_scaled: Vec<F> = im.iter().map(|x| *x * scale).collect();
    quantized_slots(
        host_module,
        encoder,
        params.base2k.into(),
        precision_at(params, log_delta),
        &re_scaled,
        &im_scaled,
    )
}

/// Rounds each element to the nearest multiple of `2^-log_delta`.
pub fn quantize<F: TestScalar>(values: &[F], log_delta: usize) -> Vec<F> {
    let scale = to_scalar::<F>(2.0_f64.powi(log_delta as i32));
    values.iter().map(|x| (*x * scale).round() / scale).collect()
}

// ─── CKKSMeta helpers ─────────────────────────────────────────────────────────

/// Returns a `CKKSMeta` at the given `log_delta` with the standard budget from params.
pub fn precision_at(params: &CKKSTestParams, log_delta: usize) -> CKKSMeta {
    CKKSMeta {
        log_sparsity: 0,
        log_delta,
        log_budget: params.prec.log_budget(),
    }
}

// ─── scratch allocation ───────────────────────────────────────────────────────

/// Allocates scratch large enough for the full CKKS test suite (including ATK ops).
pub fn alloc_scratch<BE>(params: &CKKSTestParams, module: &Module<BE>) -> ScratchOwned<BE>
where
    BE: TestContextBackend,
    Module<BE>: TestContextModule<BE>,
    ScratchOwned<BE>: ScratchOwnedAlloc<BE>,
{
    let mut ct = module.ckks_ciphertext_alloc_from_infos(&params.glwe_layout());
    ct.set_meta(params.prec);
    let tsk_infos = params.tsk_layout();
    let atk_infos = params.atk_layout();
    let scratch_size = module.ckks_all_ops_with_atk_tmp_bytes(&ct, &tsk_infos, &atk_infos, &PT_PREC);
    ScratchOwned::<BE>::alloc(scratch_size)
}

// ─── ciphertext allocation ────────────────────────────────────────────────────

/// Allocates a ciphertext with `k` limbs according to `params`.
pub fn alloc_ct<BE: Backend>(params: &CKKSTestParams, module: &Module<BE>, k: usize) -> CKKSCiphertext<BE::OwnedBuf>
where
    Module<BE>: CKKSModuleAlloc<BE>,
{
    let mut layout = params.glwe_layout();
    layout.layout.k = k.into();
    module.ckks_ciphertext_alloc_from_infos(&layout)
}

// ─── plaintext upload / download ─────────────────────────────────────────────

/// Uploads a host-side plaintext to the backend.
pub fn upload_pt<BE>(module: &Module<BE>, pt: &CKKSPlaintext<Vec<u8>>) -> CKKSPlaintext<BE::OwnedBuf>
where
    BE: Backend + TransferFrom<HostBytesBackend>,
{
    CKKSPlaintext::from_inner(module.upload_glwe_plaintext(&pt.inner), pt.meta())
}

/// Downloads a backend plaintext to the host.
pub fn download_pt<BE: Backend>(pt: &CKKSPlaintext<BE::OwnedBuf>) -> CKKSPlaintext<Vec<u8>> {
    pt.to_host_owned::<BE>()
}

// ─── plaintext encoding helpers ───────────────────────────────────────────────

/// Encodes complex slots into a host plaintext then uploads it to the backend.
pub fn encode_and_upload_pt<BE, F, E>(
    host_module: &Module<HostBytesBackend>,
    module: &Module<BE>,
    encoder: &Encoder<E>,
    base2k: Base2K,
    prec: CKKSMeta,
    re: &[F],
    im: &[F],
) -> CKKSPlaintext<BE::OwnedBuf>
where
    BE: Backend + TransferFrom<HostBytesBackend>,
    Module<HostBytesBackend>: TestContextHostModule,
    F: TestScalar,
    E: NegacyclicFFT<F>,
{
    let mut host_pt = host_module.ckks_pt_vec_alloc(base2k, prec);
    encoder.encode_reim(&mut host_pt, re, im).unwrap();
    upload_pt(module, &host_pt)
}

/// Encodes a packed constant (at most 2 coefficients: re, im) and uploads.
pub fn ckks_pt_cst<BE, F>(
    host_module: &Module<HostBytesBackend>,
    module: &Module<BE>,
    base2k: Base2K,
    prec: CKKSMeta,
    re: Option<f64>,
    im: Option<f64>,
) -> CKKSPlaintext<BE::OwnedBuf>
where
    BE: Backend + TransferFrom<HostBytesBackend>,
    Module<HostBytesBackend>: TestContextHostModule,
    F: TestScalar,
{
    let coeff_count = if im.is_some() { 2 } else { 1 };
    let mut host_pt = host_module.ckks_pt_coeffs_alloc(coeff_count, base2k, prec);
    let mut packed: Vec<F> = vec![F::zero(); coeff_count];
    if let Some(r) = re {
        packed[0] = to_scalar(r);
    }
    if let Some(i) = im {
        packed[1] = to_scalar(i);
    }
    host_pt.encode_host_floats(&packed).unwrap();
    upload_pt(module, &host_pt)
}

/// Encodes a full-degree constant (sets coefficient 0 = re, coefficient m = im) and uploads.
pub fn ckks_pt_cst_full<BE, F>(
    host_module: &Module<HostBytesBackend>,
    module: &Module<BE>,
    base2k: Base2K,
    prec: CKKSMeta,
    m: usize,
    re: Option<f64>,
    im: Option<f64>,
) -> CKKSPlaintext<BE::OwnedBuf>
where
    BE: Backend + TransferFrom<HostBytesBackend>,
    Module<HostBytesBackend>: TestContextHostModule,
    F: TestScalar,
{
    let n = m * 2;
    let mut host_pt = host_module.ckks_pt_vec_alloc(base2k, prec);
    let mut coeffs: Vec<F> = vec![F::zero(); n];
    if let Some(r) = re {
        coeffs[0] = to_scalar(r);
    }
    if let Some(i) = im {
        coeffs[m] = to_scalar(i);
    }
    host_pt.encode_host_floats(&coeffs).unwrap();
    upload_pt(module, &host_pt)
}

/// Encodes and uploads the add/sub test constant.
pub fn add_sub_const_pt<BE, F>(
    host_module: &Module<HostBytesBackend>,
    module: &Module<BE>,
    base2k: Base2K,
) -> CKKSPlaintext<BE::OwnedBuf>
where
    BE: Backend + TransferFrom<HostBytesBackend>,
    Module<HostBytesBackend>: TestContextHostModule,
    F: TestScalar,
{
    ckks_pt_cst::<BE, F>(
        host_module,
        module,
        base2k,
        PT_PREC,
        Some(ADD_SUB_CONST.0),
        Some(ADD_SUB_CONST.1),
    )
}

/// Encodes and uploads the multiply test constant as a full-degree plaintext.
pub fn mul_const_full_pt<BE, F>(
    host_module: &Module<HostBytesBackend>,
    module: &Module<BE>,
    base2k: Base2K,
    m: usize,
) -> CKKSPlaintext<BE::OwnedBuf>
where
    BE: Backend + TransferFrom<HostBytesBackend>,
    Module<HostBytesBackend>: TestContextHostModule,
    F: TestScalar,
{
    ckks_pt_cst_full::<BE, F>(host_module, module, base2k, PT_PREC, m, Some(MUL_CONST.0), Some(MUL_CONST.1))
}

// ─── key generation ───────────────────────────────────────────────────────────

/// Generates and uploads a raw secret key plus a prepared secret key.
///
/// Returns `(sk_raw, sk_prepared)`.  `sk_raw` is needed to generate evaluation
/// keys; `sk_prepared` is used for encrypt/decrypt.
pub fn gen_sk_with_raw<BE>(
    params: &CKKSTestParams,
    module: &Module<BE>,
    host_module: &Module<HostBytesBackend>,
    seed: [u8; 32],
) -> (BackendGLWESecret<BE>, GLWESecretPrepared<BE::OwnedBuf, BE>)
where
    BE: TestContextBackend,
    Module<BE>: TestContextModule<BE>,
    Module<HostBytesBackend>: TestContextHostModule,
{
    let glwe_infos = params.glwe_layout();
    let mut source = Source::new(seed);
    let mut sk_host = host_module.glwe_secret_alloc_from_infos(&glwe_infos);
    sk_host.fill_ternary_hw(params.hw, &mut source);
    let sk_raw = module.upload_glwe_secret(&sk_host);
    let mut sk = module.glwe_secret_prepared_alloc_from_infos(&glwe_infos);
    module.glwe_secret_prepare(&mut sk, &sk_raw);
    (sk_raw, sk)
}

/// Generates a prepared secret key (convenience wrapper around [`gen_sk_with_raw`]).
pub fn gen_sk<BE>(
    params: &CKKSTestParams,
    module: &Module<BE>,
    host_module: &Module<HostBytesBackend>,
    seed: [u8; 32],
) -> GLWESecretPrepared<BE::OwnedBuf, BE>
where
    BE: TestContextBackend,
    Module<BE>: TestContextModule<BE>,
    Module<HostBytesBackend>: TestContextHostModule,
{
    gen_sk_with_raw(params, module, host_module, seed).1
}

/// Generates a prepared tensor key (multiplication relinearisation key).
pub fn gen_tsk<BE>(
    params: &CKKSTestParams,
    module: &Module<BE>,
    sk_raw: &BackendGLWESecret<BE>,
    scratch: &mut ScratchArena<'_, BE>,
) -> GLWETensorKeyPrepared<BE::OwnedBuf, BE>
where
    BE: TestContextBackend,
    Module<BE>: TestContextModule<BE>,
{
    let tsk_infos = params.tsk_layout();
    let mut xa = Source::new([1u8; 32]);
    let mut xe = Source::new([2u8; 32]);
    let mut tsk = module.glwe_tensor_key_alloc_from_infos(&tsk_infos);
    module.glwe_tensor_key_encrypt_sk(&mut tsk, sk_raw, &tsk_infos, &mut xe, &mut xa, scratch);
    let mut tsk_prepared = module.alloc_tensor_key_prepared_from_infos(&tsk_infos);
    module.prepare_tensor_key(&mut tsk_prepared, &tsk, scratch);
    tsk_prepared
}

/// Generates a prepared automorphism key for rotation (`index ≥ 0`) or
/// conjugation (`index == -1`).
pub fn gen_atk<BE>(
    params: &CKKSTestParams,
    module: &Module<BE>,
    galois_element: i64,
    sk_raw: &BackendGLWESecret<BE>,
    scratch: &mut ScratchArena<'_, BE>,
) -> GLWEAutomorphismKeyPrepared<BE::OwnedBuf, BE>
where
    BE: TestContextBackend,
    Module<BE>: TestContextModule<BE>,
{
    let atk_infos = params.atk_layout();
    let mut xa = Source::new([1u8; 32]);
    let mut xe = Source::new([2u8; 32]);
    let mut atk = module.glwe_automorphism_key_alloc_from_infos(&atk_infos);
    module.glwe_automorphism_key_encrypt_sk(&mut atk, galois_element, sk_raw, &atk_infos, &mut xe, &mut xa, scratch);
    let mut atk_prepared = module.glwe_automorphism_key_prepared_alloc_from_infos(&atk_infos);
    module.glwe_automorphism_key_prepare(&mut atk_prepared, &atk, scratch);
    atk_prepared
}

// ─── encrypt / decrypt ────────────────────────────────────────────────────────

/// Encrypts `(re, im)` at the given `k` using `params.prec` as the plaintext
/// precision.
#[allow(clippy::too_many_arguments)]
pub fn ckks_encrypt<BE, F, E>(
    params: &CKKSTestParams,
    module: &Module<BE>,
    host_module: &Module<HostBytesBackend>,
    encoder: &Encoder<E>,
    sk: &GLWESecretPrepared<BE::OwnedBuf, BE>,
    k: usize,
    re: &[F],
    im: &[F],
    scratch: &mut ScratchArena<'_, BE>,
) -> CKKSCiphertext<BE::OwnedBuf>
where
    BE: TestContextBackend,
    Module<BE>: TestContextModule<BE>,
    Module<HostBytesBackend>: TestContextHostModule,
    F: TestScalar,
    E: NegacyclicFFT<F>,
{
    ckks_encrypt_with_prec(params, module, host_module, encoder, sk, k, re, im, params.prec, scratch)
}

/// Encrypts a real **coefficient** vector directly (no slot/FFT encoding), at the
/// given `k` and `prec`. `coeffs` has length `n` (the ring degree) and is placed
/// into the polynomial coefficients. This is the input form for the homomorphic
/// `CoeffsToSlots` test pipeline, which encrypts `bitReverse(vReal)||bitReverse(vImag)`
/// coefficient-wise.
#[allow(clippy::too_many_arguments)]
pub fn ckks_encrypt_coeffs<BE, F>(
    params: &CKKSTestParams,
    module: &Module<BE>,
    host_module: &Module<HostBytesBackend>,
    sk: &GLWESecretPrepared<BE::OwnedBuf, BE>,
    k: usize,
    coeffs: &[F],
    prec: CKKSMeta,
    scratch: &mut ScratchArena<'_, BE>,
) -> CKKSCiphertext<BE::OwnedBuf>
where
    BE: TestContextBackend,
    Module<BE>: TestContextModule<BE>,
    Module<HostBytesBackend>: TestContextHostModule,
    F: TestScalar,
    CKKSPlaintext<Vec<u8>>: CKKSPlaintextVecHostCodec<F>,
{
    let mut host_pt = host_module.ckks_pt_vec_alloc(params.base2k.into(), prec);
    host_pt.encode_host_floats(coeffs).unwrap();
    let pt = upload_pt(module, &host_pt);

    let mut layout = params.glwe_layout().layout;
    layout.k = k.into();
    let enc_infos = EncryptionLayout::new_from_default_sigma(layout).unwrap();

    let mut ct = alloc_ct(params, module, k);
    let mut xa = Source::new([5u8; 32]);
    let mut xe = Source::new([6u8; 32]);
    module
        .ckks_encrypt_sk(&mut ct, &pt, sk, &enc_infos, &mut xa, &mut xe, scratch)
        .unwrap();
    ct
}

/// Encrypts `(re, im)` at the given `k` and explicit `prec`.
#[allow(clippy::too_many_arguments)]
pub fn ckks_encrypt_with_prec<BE, F, E>(
    params: &CKKSTestParams,
    module: &Module<BE>,
    host_module: &Module<HostBytesBackend>,
    encoder: &Encoder<E>,
    sk: &GLWESecretPrepared<BE::OwnedBuf, BE>,
    k: usize,
    re: &[F],
    im: &[F],
    prec: CKKSMeta,
    scratch: &mut ScratchArena<'_, BE>,
) -> CKKSCiphertext<BE::OwnedBuf>
where
    BE: TestContextBackend,
    Module<BE>: TestContextModule<BE>,
    Module<HostBytesBackend>: TestContextHostModule,
    F: TestScalar,
    E: NegacyclicFFT<F>,
{
    let mut host_pt = host_module.ckks_pt_vec_alloc(params.base2k.into(), prec);
    encoder.encode_reim(&mut host_pt, re, im).unwrap();
    let pt = upload_pt(module, &host_pt);

    let mut layout = params.glwe_layout().layout;
    layout.k = k.into();
    let enc_infos = EncryptionLayout::new_from_default_sigma(layout).unwrap();

    let mut ct = alloc_ct(params, module, k);
    let mut xa = Source::new([3u8; 32]);
    let mut xe = Source::new([4u8; 32]);
    module
        .ckks_encrypt_sk(&mut ct, &pt, sk, &enc_infos, &mut xa, &mut xe, scratch)
        .unwrap();
    ct
}

/// Encrypts an already-encoded host plaintext at the given `k`. Use this when the
/// slot encoding is not a plain dense `(re, im)` or coefficient vector — e.g. a
/// sparse / repacked layout the caller built with [`Encoder::encode_reim_sparse`].
pub fn ckks_encrypt_pt<BE>(
    params: &CKKSTestParams,
    module: &Module<BE>,
    sk: &GLWESecretPrepared<BE::OwnedBuf, BE>,
    k: usize,
    host_pt: &CKKSPlaintext<Vec<u8>>,
    scratch: &mut ScratchArena<'_, BE>,
) -> CKKSCiphertext<BE::OwnedBuf>
where
    BE: TestContextBackend,
    Module<BE>: TestContextModule<BE>,
{
    let pt = upload_pt(module, host_pt);

    let mut layout = params.glwe_layout().layout;
    layout.k = k.into();
    let enc_infos = EncryptionLayout::new_from_default_sigma(layout).unwrap();

    let mut ct = alloc_ct(params, module, k);
    let mut xa = Source::new([3u8; 32]);
    let mut xe = Source::new([4u8; 32]);
    module
        .ckks_encrypt_sk(&mut ct, &pt, sk, &enc_infos, &mut xa, &mut xe, scratch)
        .unwrap();
    ct
}

/// Decrypts `ct` with `prec` metadata and returns the host-side plaintext.
pub fn ckks_decrypt_with_prec<BE>(
    module: &Module<BE>,
    ct: &CKKSCiphertext<BE::OwnedBuf>,
    sk: &GLWESecretPrepared<BE::OwnedBuf, BE>,
    prec: CKKSMeta,
    scratch: &mut ScratchArena<'_, BE>,
) -> anyhow::Result<CKKSPlaintext<Vec<u8>>>
where
    BE: TestContextBackend,
    Module<BE>: TestContextModule<BE>,
{
    let mut pt = module.ckks_pt_vec_alloc(ct.base2k(), prec);
    module.ckks_decrypt(&mut pt, ct, sk, scratch)?;
    Ok(download_pt::<BE>(&pt))
}

/// Decrypts and decodes `ct`, returning the slot vectors.
pub fn ckks_decrypt_decode<BE, F, E>(
    params: &CKKSTestParams,
    module: &Module<BE>,
    encoder: &Encoder<E>,
    ct: &CKKSCiphertext<BE::OwnedBuf>,
    sk: &GLWESecretPrepared<BE::OwnedBuf, BE>,
    scratch: &mut ScratchArena<'_, BE>,
) -> (Vec<F>, Vec<F>)
where
    BE: TestContextBackend,
    Module<BE>: TestContextModule<BE>,
    F: TestScalar,
    E: NegacyclicFFT<F>,
{
    // Cap the integer headroom so the centered plaintext fits the i128 decode codec
    // (`log_delta + log_budget <= 127`). This is the decode-time equivalent of
    // rescaling the ciphertext to a smaller modulus before decoding: the decoded
    // values are bounded, so dropping the unused high-order budget is lossless and
    // lets the suite exercise scales (`log_delta`) wider than would otherwise leave
    // room for the full noise budget.
    let prec = CKKSMeta {
        log_sparsity: 0,
        log_delta: ct.log_delta(),
        log_budget: ct
            .log_budget()
            .min(params.prec.log_budget())
            .min(127usize.saturating_sub(ct.log_delta())),
    };
    let pt = ckks_decrypt_with_prec(module, ct, sk, prec, scratch).unwrap();
    ckks_decode_pt(encoder, params.n / 2, &pt)
}

/// Decodes a host-side plaintext to slot vectors.
pub fn ckks_decode_pt<F, E>(encoder: &Encoder<E>, m: usize, pt: &CKKSPlaintext<Vec<u8>>) -> (Vec<F>, Vec<F>)
where
    F: TestScalar,
    E: NegacyclicFFT<F>,
{
    let mut re = vec![F::zero(); m];
    let mut im = vec![F::zero(); m];
    encoder.decode_reim(pt, &mut re, &mut im).unwrap();
    (re, im)
}

// ─── precision assertion helpers ─────────────────────────────────────────────

#[derive(Clone, Copy, Debug)]
pub struct PrecisionStats {
    pub min_log2_prec: f64,
    pub max_log2_prec: f64,
    pub avg_log2_prec: f64,
    pub worst_idx: usize,
    pub worst_got: f64,
    pub worst_want: f64,
    pub worst_err: f64,
}

const PRECISION_GUARD_BITS: f64 = 2.0;

/// Slack (in log2 bits) added to the analytic decryption-noise floor
/// `2^(-effective_k + log2(N))` used by [`assert_decrypt_precision`]. Sized to
/// absorb the extra noise of key-switched operations (conjugate / rotate sit
/// ~2.5 bits above the base floor) while keeping the ceiling tight enough to
/// flag noise regressions and head-room corruption.
const NOISE_GUARD_BITS: f64 = 4.0;

/// Returns the minimum expected average log2 precision for standard-ring CKKS.
pub fn expected_log2_precision(log_delta: usize, degree: usize) -> f64 {
    (log_delta as f64 - degree.ilog2() as f64 - PRECISION_GUARD_BITS).max(0.0)
}

/// Computes per-slot log2 precision statistics.
pub fn precision_stats<F>(got: &[F], want: &[F], log_delta: usize) -> PrecisionStats
where
    F: Float + ToPrimitive + Debug,
{
    assert_eq!(got.len(), want.len(), "precision_stats: vector length mismatch");
    let capped_prec = log_delta as f64;
    let mut min_log2_prec = f64::INFINITY;
    let mut max_log2_prec: f64 = 0.0;
    let mut sum_log2_prec = 0.0;
    let mut worst_idx = 0usize;
    let mut worst_got = 0.0f64;
    let mut worst_want = 0.0f64;
    let mut worst_err = 0.0f64;

    for (idx, (g, w)) in got.iter().zip(want.iter()).enumerate() {
        let err = (*g - *w).abs();
        let err_f64 = err.to_f64().unwrap();
        let prec = if err.is_zero() {
            capped_prec
        } else {
            (-err.log2().to_f64().unwrap()).min(capped_prec)
        };
        if err_f64 > worst_err {
            worst_err = err_f64;
            worst_idx = idx;
            worst_got = g.to_f64().unwrap();
            worst_want = w.to_f64().unwrap();
        }
        min_log2_prec = min_log2_prec.min(prec);
        max_log2_prec = max_log2_prec.max(prec);
        sum_log2_prec += prec;
    }

    PrecisionStats {
        min_log2_prec,
        max_log2_prec,
        avg_log2_prec: sum_log2_prec / got.len() as f64,
        worst_idx,
        worst_got,
        worst_want,
        worst_err,
    }
}

/// Asserts that `got` and `want` meet the expected average log2 precision.
pub fn assert_precision<F>(label: &str, got: &[F], want: &[F], log_delta: usize, degree: usize)
where
    F: Float + ToPrimitive + Debug,
{
    let stats = precision_stats(got, want, log_delta);
    let min_bits = expected_log2_precision(log_delta, degree);
    assert!(
        stats.avg_log2_prec >= min_bits,
        "{label}: avg precision {:.1} bits < {:.1} (log_delta={}, degree={}, min={:.1}, max={:.1}, max_err={}, sample_idx={}, got={}, want={})",
        stats.avg_log2_prec,
        min_bits,
        log_delta,
        degree,
        stats.min_log2_prec,
        stats.max_log2_prec,
        stats.worst_err,
        stats.worst_idx,
        stats.worst_got,
        stats.worst_want
    );
}

/// Asserts precision at a given `log_delta` (without decryption).
pub fn assert_precision_for_log_delta<F>(label: &str, got: &[F], want: &[F], log_delta: usize, degree: usize)
where
    F: Float + ToPrimitive + Debug,
{
    assert_precision(label, got, want, log_delta, degree);
}

/// Decrypts `ct`, decodes, and asserts precision at `ct.log_delta()`.
#[allow(clippy::too_many_arguments)]
pub fn assert_decrypt_precision<BE, F, E>(
    label: &str,
    params: &CKKSTestParams,
    module: &Module<BE>,
    encoder: &Encoder<E>,
    ct: &CKKSCiphertext<BE::OwnedBuf>,
    sk: &GLWESecretPrepared<BE::OwnedBuf, BE>,
    want_re: &[F],
    want_im: &[F],
    scratch: &mut ScratchArena<'_, BE>,
) where
    BE: TestContextBackend,
    Module<BE>: TestContextModule<BE>,
    F: TestScalar,
    E: NegacyclicFFT<F>,
    for<'a> BE::BufRef<'a>: HostDataRef,
    for<'a> BE::BufMut<'a>: HostDataMut,
{
    assert_decrypt_precision_at_log_delta(
        label,
        params,
        module,
        encoder,
        ct,
        sk,
        want_re,
        want_im,
        ct.log_delta(),
        scratch,
    )
}

/// Asserts that `ct` decrypts to `(want_re, want_im)` with two complementary
/// checks, at the caller-provided precision target `log_delta`.
///
/// **Ring domain (valid plaintext).** Encodes the expected message at the
/// ciphertext's full metadata — so the reference spans the same limbs as a
/// full-width decryption — and measures the noise `std` of `decrypt(ct) - want(X)`
/// directly over the polynomial coefficients. Unlike decrypt-then-decode, this
/// does *not* clip the limbs above the plaintext head-room, so any corruption
/// there contributes to the noise instead of being silently discarded. The std
/// is bounded by the analytic floor `2^(-effective_k + log2(N)) * 2^NOISE_GUARD_BITS`,
/// with `effective_k = log_delta + log_budget`. This mainly asserts that the
/// top bits of the plaintext are zero, i.e. that the ciphertext is valid.
///
/// **Canonical embedding (precision).** Decrypts, decodes back to slots, and
/// asserts the per-slot log2 precision matches the analytic expectation at
/// `log_delta` (see [`assert_precision`]). This checks that the precision of the
/// recovered plaintext is what we would expect.
#[allow(clippy::too_many_arguments)]
pub fn assert_decrypt_precision_at_log_delta<BE, F, E>(
    label: &str,
    params: &CKKSTestParams,
    module: &Module<BE>,
    encoder: &Encoder<E>,
    ct: &CKKSCiphertext<BE::OwnedBuf>,
    sk: &GLWESecretPrepared<BE::OwnedBuf, BE>,
    want_re: &[F],
    want_im: &[F],
    log_delta: usize,
    scratch: &mut ScratchArena<'_, BE>,
) where
    BE: TestContextBackend,
    Module<BE>: TestContextModule<BE>,
    F: TestScalar,
    E: NegacyclicFFT<F>,
    for<'a> BE::BufRef<'a>: HostDataRef,
    for<'a> BE::BufMut<'a>: HostDataMut,
{
    // Encode the expected message at the ciphertext's full metadata, so the
    // reference spans the same limbs as a full-width decryption (no head-room
    // clipping — any corruption above the head-room shows up as noise).
    let mut pt_want = module.ckks_pt_vec_alloc(
        ct.base2k(),
        CKKSMeta {
            log_sparsity: ct.log_sparsity(),
            log_delta: ct.log_delta(),
            log_budget: ct.log_budget(),
        },
    );
    encoder.encode_reim(&mut pt_want, want_re, want_im).unwrap();

    // Compact the ciphertext to its `effective_k` limbs first: the decryption is
    // sized like its input, so without this the noise would be measured over the
    // rounded `max_k` storage (including the sub-head-room padding) rather than
    // the semantic `effective_k` head-room the bound targets.
    let ct_compact = ct.compact(module, scratch).unwrap();

    // Decrypt once into the raw full-width plaintext; both checks below extract
    // their own view from it (re-extracting an already-extracted plaintext would
    // shift the message scale twice). `full_pt` carries `ct_compact`'s metadata so
    // the extracts re-precision it exactly as `ckks_decrypt` would.
    let mut full_pt = module.glwe_plaintext_alloc_from_infos(&ct_compact);
    module.glwe_decrypt(&ct_compact, &mut full_pt, sk, scratch);
    let full_pt = CKKSPlaintext::from_inner(full_pt, ct_compact.meta());

    // ── Ring-domain check: the decryption is a valid plaintext. ──────────────
    // Re-precision the decryption at full width, subtract the reference and bound
    // the residual noise `std` directly over the polynomial coefficients. Unlike
    // the decode below, this does *not* clip the limbs above the plaintext
    // head-room, so any corruption there fails here.
    let mut pt_noise = module.ckks_plaintext_alloc_from_infos(&ct_compact);
    module.ckks_extract_pt(&mut pt_noise, &full_pt, scratch).unwrap();
    module.glwe_sub_assign(&mut pt_noise, &pt_want);

    let noise = pt_noise.inner.data().stats(pt_noise.base2k().into(), 0);
    let noise_log2 = if noise.std() == 0.0 {
        f64::NEG_INFINITY
    } else {
        noise.std().log2()
    };

    let effective_k = log_delta + ct_compact.log_budget();
    let log_n = params.n.ilog2() as f64;
    let bound = -(effective_k as f64) + log_n + NOISE_GUARD_BITS;
    assert!(
        noise_log2 <= bound,
        "{label}: ring noise std {noise_log2:.1} bits > bound {bound:.1} \
         (effective_k={effective_k}, log_delta={log_delta}, log_budget={}, log_n={log_n})",
        ct_compact.log_budget(),
    );

    // ── Canonical-embedding check: the decoded slots match at `log_delta`. ───
    // Re-precision the decryption down to a budget the decoder can represent —
    // decoding the full storage width would overflow the decoder's
    // `log_delta + log_budget <= 127` limit.
    let mut pt_decode = module.ckks_pt_vec_alloc(
        ct.base2k(),
        CKKSMeta {
            log_sparsity: ct.log_sparsity(),
            log_delta: ct.log_delta(),
            log_budget: ct.log_budget().min(params.prec.log_budget()),
        },
    );
    module.ckks_extract_pt(&mut pt_decode, &full_pt, scratch).unwrap();
    let pt_host = download_pt::<BE>(&pt_decode);
    let (re_out, im_out) = ckks_decode_pt(encoder, params.n / 2, &pt_host);
    assert_precision(&format!("{label} re"), &re_out, want_re, log_delta, params.n);
    assert_precision(&format!("{label} im"), &im_out, want_im, log_delta, params.n);
}

// ─── metadata assertion helpers ───────────────────────────────────────────────

pub fn assert_ct_meta<D: Data, S: CKKSNormalizationState>(
    label: &str,
    ct: &CKKSCiphertext<D, S>,
    log_delta: usize,
    log_budget: usize,
) {
    assert_eq!(ct.log_delta(), log_delta, "{label}: unexpected log_delta");
    assert_eq!(ct.log_budget(), log_budget, "{label}: unexpected log_budget");
}

pub fn assert_ckks_error(label: &str, err: &anyhow::Error, want: CKKSCompositionError) {
    let got = err.downcast_ref::<CKKSCompositionError>();
    assert_eq!(got, Some(&want), "{label}: unexpected error: {err}");
}

pub fn assert_unary_output_meta<D: Data, S: CKKSNormalizationState>(
    label: &str,
    ct: &CKKSCiphertext<D, S>,
    input: &CKKSCiphertext<impl Data>,
) {
    assert_ct_meta(label, ct, input.log_delta(), input.log_budget() - ct.offset_unary(input));
}

pub fn assert_binary_output_meta<D: Data, S: CKKSNormalizationState>(
    label: &str,
    ct: &CKKSCiphertext<D, S>,
    a: &CKKSCiphertext<impl Data>,
    b: &CKKSCiphertext<impl Data>,
) {
    assert_ct_meta(
        label,
        ct,
        a.log_delta().min(b.log_delta()),
        a.log_budget().min(b.log_budget()) - ct.offset_binary(a, b),
    );
}

pub fn assert_mul_ct_output_meta<D: Data, S: CKKSNormalizationState>(
    label: &str,
    ct: &CKKSCiphertext<D, S>,
    a: &impl CKKSInfos,
    b: &impl CKKSInfos,
) {
    let log_budget = a.log_budget().min(b.log_budget()) - a.log_delta().max(b.log_delta());
    let log_delta = a.log_delta().min(b.log_delta());
    let offset = (log_budget + log_delta).saturating_sub(ct.max_k().as_usize());
    assert_ct_meta(label, ct, log_delta, log_budget - offset);
}

pub fn assert_mul_pt_output_meta<D: Data, S: CKKSNormalizationState>(
    label: &str,
    ct: &CKKSCiphertext<D, S>,
    a: &impl CKKSInfos,
    b: &impl CKKSInfos,
) {
    let log_budget = a.log_budget() - a.log_delta().min(b.log_delta());
    let log_delta = a.log_delta().max(b.log_delta());
    let offset = (log_budget + log_delta).saturating_sub(ct.max_k().as_usize());
    assert_ct_meta(label, ct, log_delta, log_budget - offset);
}
