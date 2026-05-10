//! Test context builders and precision assertion helpers.
//!
//! [`TestContext`] owns the backend module, prepared secret key, optional
//! evaluation keys, and two pairs of test messages.  It provides convenience
//! methods for encrypt, decrypt-and-decode, and scratch allocation.

use std::{collections::HashMap, f64::consts::TAU, fmt::Debug, marker::PhantomData};

use super::CKKSTestParams;
use crate::{
    CKKSCompositionError, CKKSInfos, CKKSMeta, SetCKKSInfos,
    api::CKKSMulAddOps,
    encoding::reim::Encoder,
    layouts::{CKKSCiphertext, CKKSModuleAlloc, CKKSPlaintextVecHostCodec, ciphertext::CKKSOffset, plaintext::CKKSPlaintext},
    leveled::api::{
        CKKSAddOps, CKKSConjugateOps, CKKSCopyOps, CKKSDecrypt, CKKSEncrypt, CKKSImagOps, CKKSMulOps, CKKSNegOps, CKKSPow2Ops,
        CKKSRescaleOps, CKKSRotateOps, CKKSSubOps, PolynomialEvaluation,
    },
    oep::{
        CKKSAddImpl, CKKSConjugateImpl, CKKSCopyImpl, CKKSEncryptionImpl, CKKSImagImpl, CKKSMulImpl, CKKSNegImpl,
        CKKSPlaintextZnxImpl, CKKSPolynomialEvaluationImpl, CKKSPow2Impl, CKKSRescaleImpl, CKKSRotateImpl, CKKSSubImpl,
    },
};
use poulpy_core::{
    EncryptionLayout, GLWEAdd, GLWEAutomorphism, GLWEAutomorphismKeyEncryptSk, GLWECopy, GLWEMulConst, GLWEMulPlain, GLWENegate,
    GLWENormalize, GLWERotate, GLWEShift, GLWESub, GLWETensorKeyEncryptSk, GLWETensoring, ScratchArenaTakeCore,
    layouts::{
        Base2K, Degree, GLWEAutomorphismKeyPrepared, GLWEAutomorphismKeyPreparedFactory, GLWESecretPreparedFactory,
        GLWETensorKeyPrepared, GLWETensorKeyPreparedFactory, LWEInfos, ModuleCoreAlloc, prepared::GLWESecretPrepared,
    },
    oep::{
        AutomorphismImpl, ConversionImpl, DecryptionImpl, EncryptionImpl, GGLWEExternalProductImpl, GGLWEKeyswitchImpl,
        GGSWExternalProductImpl, GGSWKeyswitchImpl, GGSWRotateImpl, GLWEAddImpl, GLWECopyImpl, GLWEExternalProductImpl,
        GLWEKeyswitchImpl, GLWEMulConstImpl, GLWEMulPlainImpl, GLWEMulXpMinusOneImpl, GLWENegateImpl, GLWENormalizeImpl,
        GLWEPackImpl, GLWERotateImpl, GLWEShiftImpl, GLWESubImpl, GLWETensoringImpl, GLWETraceImpl, LWEKeyswitchImpl,
    },
};
use rand_distr::num_traits::{Float, FloatConst, FromPrimitive, ToPrimitive};

use poulpy_hal::{
    api::{
        ModuleN, ModuleNew, NegacyclicFFT, NegacyclicFFTNew, ScratchAvailable, ScratchOwnedAlloc, ScratchOwnedBorrow,
        VecZnxRshAddIntoBackend, VecZnxRshSubBackend,
    },
    layouts::{Backend, Data, GaloisElement, HostBackend, HostBytesBackend, Module, ScratchArena, ScratchOwned},
    oep::{HalConvolutionImpl, HalModuleImpl, HalSvpImpl, HalVecZnxBigImpl, HalVecZnxDftImpl, HalVecZnxImpl, HalVmpImpl},
    source::Source,
};

pub trait TestBackend:
    Backend
    + HostBackend<OwnedBuf = Vec<u8>>
    + GLWEKeyswitchImpl<Self>
    + GLWEAddImpl<Self>
    + GLWESubImpl<Self>
    + GLWENegateImpl<Self>
    + GLWECopyImpl<Self>
    + GGLWEKeyswitchImpl<Self>
    + GGSWKeyswitchImpl<Self>
    + LWEKeyswitchImpl<Self>
    + GLWEExternalProductImpl<Self>
    + GGLWEExternalProductImpl<Self>
    + GGSWExternalProductImpl<Self>
    + GLWETensoringImpl<Self>
    + GLWEMulConstImpl<Self>
    + GLWEMulPlainImpl<Self>
    + GLWERotateImpl<Self>
    + GLWEMulXpMinusOneImpl<Self>
    + GLWEShiftImpl<Self>
    + GLWENormalizeImpl<Self>
    + GLWETraceImpl<Self>
    + GLWEPackImpl<Self>
    + GGSWRotateImpl<Self>
    + EncryptionImpl<Self>
    + DecryptionImpl<Self>
    + ConversionImpl<Self>
    + AutomorphismImpl<Self>
    + HalModuleImpl<Self>
    + HalVecZnxImpl<Self>
    + HalVecZnxBigImpl<Self>
    + HalVecZnxDftImpl<Self>
    + HalSvpImpl<Self>
    + HalVmpImpl<Self>
    + HalConvolutionImpl<Self>
    + CKKSPlaintextZnxImpl<Self>
    + CKKSCopyImpl<Self>
    + CKKSAddImpl<Self>
    + CKKSEncryptionImpl<Self>
    + CKKSSubImpl<Self>
    + CKKSNegImpl<Self>
    + CKKSPow2Impl<Self>
    + CKKSImagImpl<Self>
    + CKKSRescaleImpl<Self>
    + CKKSRotateImpl<Self>
    + CKKSConjugateImpl<Self>
    + CKKSMulImpl<Self>
    + CKKSPolynomialEvaluationImpl<Self>
{
}

impl<T> TestBackend for T where
    T: Backend
        + HostBackend<OwnedBuf = Vec<u8>>
        + GLWEKeyswitchImpl<T>
        + GLWEAddImpl<T>
        + GLWESubImpl<T>
        + GLWENegateImpl<T>
        + GLWECopyImpl<T>
        + GGLWEKeyswitchImpl<T>
        + GGSWKeyswitchImpl<T>
        + LWEKeyswitchImpl<T>
        + GLWEExternalProductImpl<T>
        + GGLWEExternalProductImpl<T>
        + GGSWExternalProductImpl<T>
        + GLWETensoringImpl<T>
        + GLWEMulConstImpl<T>
        + GLWEMulPlainImpl<T>
        + GLWERotateImpl<T>
        + GLWEMulXpMinusOneImpl<T>
        + GLWEShiftImpl<T>
        + GLWENormalizeImpl<T>
        + GLWETraceImpl<T>
        + GLWEPackImpl<T>
        + GGSWRotateImpl<T>
        + EncryptionImpl<T>
        + DecryptionImpl<T>
        + ConversionImpl<T>
        + AutomorphismImpl<T>
        + HalModuleImpl<T>
        + HalVecZnxImpl<T>
        + HalVecZnxBigImpl<T>
        + HalVecZnxDftImpl<T>
        + HalSvpImpl<T>
        + HalVmpImpl<T>
        + HalConvolutionImpl<T>
        + CKKSPlaintextZnxImpl<T>
        + CKKSCopyImpl<T>
        + CKKSAddImpl<T>
        + CKKSEncryptionImpl<T>
        + CKKSSubImpl<T>
        + CKKSNegImpl<T>
        + CKKSPow2Impl<T>
        + CKKSImagImpl<T>
        + CKKSRescaleImpl<T>
        + CKKSRotateImpl<T>
        + CKKSConjugateImpl<T>
        + CKKSMulImpl<T>
        + CKKSPolynomialEvaluationImpl<T>
{
}

pub trait TestContextBackend: TestBackend
where
    Module<Self>: ModuleNew<Self>
        + ModuleN
        + GLWESecretPreparedFactory<Self>
        + GLWETensorKeyEncryptSk<Self>
        + GLWETensorKeyPreparedFactory<Self>
        + GLWEAutomorphismKeyEncryptSk<Self>
        + CKKSEncrypt<Self>
        + CKKSDecrypt<Self>,
    ScratchOwned<Self>: ScratchOwnedAlloc<Self> + ScratchOwnedBorrow<Self>,
    for<'a> ScratchArena<'a, Self>: ScratchArenaTakeCore<'a, Self>,
{
}

impl<T> TestContextBackend for T
where
    T: TestBackend,
    Module<T>: ModuleNew<T>
        + ModuleN
        + GLWESecretPreparedFactory<T>
        + GLWETensorKeyEncryptSk<T>
        + GLWETensorKeyPreparedFactory<T>
        + GLWEAutomorphismKeyEncryptSk<T>
        + CKKSEncrypt<T>
        + CKKSDecrypt<T>,
    ScratchOwned<T>: ScratchOwnedAlloc<T> + ScratchOwnedBorrow<T>,
    for<'a> ScratchArena<'a, T>: ScratchArenaTakeCore<'a, T>,
{
}

pub trait TestCiphertextBackend: TestContextBackend
where
    Module<Self>: ModuleNew<Self>
        + ModuleN
        + GLWESecretPreparedFactory<Self>
        + GLWETensorKeyEncryptSk<Self>
        + GLWETensorKeyPreparedFactory<Self>
        + GLWEAutomorphismKeyEncryptSk<Self>
        + CKKSEncrypt<Self>
        + CKKSDecrypt<Self>,
    ScratchOwned<Self>: ScratchOwnedAlloc<Self> + ScratchOwnedBorrow<Self>,
    for<'a> ScratchArena<'a, Self>: ScratchArenaTakeCore<'a, Self>,
{
}

impl<T> TestCiphertextBackend for T
where
    T: TestContextBackend,
    Module<T>: ModuleNew<T>
        + ModuleN
        + GLWESecretPreparedFactory<T>
        + GLWETensorKeyEncryptSk<T>
        + GLWETensorKeyPreparedFactory<T>
        + GLWEAutomorphismKeyEncryptSk<T>
        + CKKSEncrypt<T>
        + CKKSDecrypt<T>,
    ScratchOwned<T>: ScratchOwnedAlloc<T> + ScratchOwnedBorrow<T>,
    for<'a> ScratchArena<'a, T>: ScratchArenaTakeCore<'a, T>,
{
}

pub trait TestAddBackend: TestCiphertextBackend
where
    Module<Self>: GLWEAdd<Self> + GLWEShift<Self> + VecZnxRshAddIntoBackend<Self>,
    for<'a> ScratchArena<'a, Self>: ScratchAvailable,
{
}

impl<T> TestAddBackend for T
where
    T: TestCiphertextBackend,
    Module<T>: GLWEAdd<T> + GLWEShift<T> + VecZnxRshAddIntoBackend<T>,
    for<'a> ScratchArena<'a, T>: ScratchAvailable,
{
}

pub trait TestSubBackend: TestCiphertextBackend
where
    Module<Self>: GLWESub<Self> + GLWEShift<Self> + VecZnxRshSubBackend<Self>,
    for<'a> ScratchArena<'a, Self>: ScratchAvailable,
{
}

impl<T> TestSubBackend for T
where
    T: TestCiphertextBackend,
    Module<T>: GLWESub<T> + GLWEShift<T> + VecZnxRshSubBackend<T>,
    for<'a> ScratchArena<'a, T>: ScratchAvailable,
{
}

pub trait TestNegBackend: TestCiphertextBackend
where
    Module<Self>: GLWENegate<Self> + GLWEShift<Self>,
    for<'a> ScratchArena<'a, Self>: ScratchAvailable,
{
}

impl<T> TestNegBackend for T
where
    T: TestCiphertextBackend,
    Module<T>: GLWENegate<T> + GLWEShift<T>,
    for<'a> ScratchArena<'a, T>: ScratchAvailable,
{
}

pub trait TestCopyBackend: TestCiphertextBackend
where
    Module<Self>: CKKSCopyOps<Self> + GLWECopy<Self> + GLWEShift<Self>,
    for<'a> ScratchArena<'a, Self>: ScratchAvailable,
{
}

impl<T> TestCopyBackend for T
where
    T: TestCiphertextBackend,
    Module<T>: CKKSCopyOps<T> + GLWECopy<T> + GLWEShift<T>,
    for<'a> ScratchArena<'a, T>: ScratchAvailable,
{
}

pub trait TestRotateBackend: TestCiphertextBackend
where
    Module<Self>: CKKSRotateOps<Self> + CKKSConjugateOps<Self>,
    for<'a> ScratchArena<'a, Self>: ScratchAvailable,
{
}

impl<T> TestRotateBackend for T
where
    T: TestCiphertextBackend,
    Module<T>: CKKSRotateOps<T> + CKKSConjugateOps<T>,
    for<'a> ScratchArena<'a, T>: ScratchAvailable,
{
}

pub trait TestMulBackend: TestCiphertextBackend
where
    Module<Self>: GLWEMulConst<Self> + GLWERotate<Self> + GLWETensoring<Self> + GLWEShift<Self>,
    for<'a> ScratchArena<'a, Self>: ScratchAvailable,
{
}

impl<T> TestMulBackend for T
where
    T: TestCiphertextBackend,
    Module<T>: GLWEMulConst<T> + GLWERotate<T> + GLWETensoring<T> + GLWEShift<T>,
    for<'a> ScratchArena<'a, T>: ScratchAvailable,
{
}

pub trait TestPolynomialEvaluationBackend: TestMulBackend
where
    Module<Self>: PolynomialEvaluation<Self>,
    for<'a> ScratchArena<'a, Self>: ScratchAvailable,
{
}

impl<T> TestPolynomialEvaluationBackend for T
where
    T: TestMulBackend,
    Module<T>: PolynomialEvaluation<T>,
    for<'a> ScratchArena<'a, T>: ScratchAvailable,
{
}

pub trait TestPow2Backend: TestCiphertextBackend
where
    Module<Self>: GLWEShift<Self> + GLWECopy<Self>,
{
}

impl<T> TestPow2Backend for T
where
    T: TestCiphertextBackend,
    Module<T>: GLWEShift<T> + GLWECopy<T>,
{
}

pub trait TestLevelBackend: TestCiphertextBackend
where
    Module<Self>: GLWEShift<Self> + GLWECopy<Self>,
{
}

impl<T> TestLevelBackend for T
where
    T: TestCiphertextBackend,
    Module<T>: GLWEShift<T> + GLWECopy<T>,
{
}

pub trait TestCompositionBackend: TestCiphertextBackend
where
    Module<Self>: GLWEAdd<Self> + GLWEMulPlain<Self> + GLWENormalize<Self> + GLWEShift<Self> + GLWETensoring<Self>,
{
}

impl<T> TestCompositionBackend for T
where
    T: TestCiphertextBackend,
    Module<T>: GLWEAdd<T> + GLWEMulPlain<T> + GLWENormalize<T> + GLWEShift<T> + GLWETensoring<T>,
{
}

#[derive(Clone, Copy)]
pub enum TestVector {
    First,
    Second,
}

pub trait TestScalar: Copy + Float + FloatConst + FromPrimitive + ToPrimitive + std::fmt::Debug + Send + Sync + 'static {}

impl<T> TestScalar for T where T: Copy + Float + FloatConst + FromPrimitive + ToPrimitive + std::fmt::Debug + Send + Sync + 'static
{}

/// Shared test state: module, keys, and two complex test messages.
///
/// Constructed via [`TestContext::new`] (base), [`TestContext::new_with_tsk`]
/// (adds tensor key for multiplication), or [`TestContext::new_with_atk`]
/// (adds automorphism keys for rotation and conjugation).
pub struct TestContext<BE: TestBackend, F: TestScalar, E> {
    pub module: Module<BE>,
    pub host_module: Module<HostBytesBackend>,
    pub encoder: Encoder<E>,
    pub params: CKKSTestParams,
    pub sk: GLWESecretPrepared<BE::OwnedBuf, BE>,
    pub tsk: GLWETensorKeyPrepared<BE::OwnedBuf, BE>,
    pub atks: HashMap<i64, GLWEAutomorphismKeyPrepared<BE::OwnedBuf, BE>>,
    pub scratch_size: usize,
    pub re1: Vec<F>,
    pub im1: Vec<F>,
    pub re2: Vec<F>,
    pub im2: Vec<F>,
    _scalar: PhantomData<F>,
}

impl<BE: TestBackend, F: TestScalar, E> TestContext<BE, F, E> {
    const PT_PREC: CKKSMeta = CKKSMeta {
        log_delta: 8,
        log_budget: 10,
    };

    pub fn m(&self) -> usize {
        self.params.n / 2
    }

    pub fn degree(&self) -> Degree {
        self.params.n.into()
    }

    pub fn base2k(&self) -> Base2K {
        self.params.base2k.into()
    }

    pub fn meta(&self) -> CKKSMeta {
        self.params.prec
    }

    pub fn meta_pt(&self) -> CKKSMeta {
        Self::PT_PREC
    }

    pub fn max_k(&self) -> usize {
        self.params.k
    }

    pub fn precision_at(&self, log_delta: usize) -> CKKSMeta {
        CKKSMeta {
            log_delta,
            log_budget: self.params.prec.log_budget(),
        }
    }

    fn to_scalar(x: f64) -> F {
        F::from_f64(x).expect("f64 -> scalar conversion should succeed")
    }
}

impl<BE: TestContextBackend, F: TestScalar, E> TestContext<BE, F, E>
where
    Module<BE>: CKKSEncrypt<BE>
        + CKKSDecrypt<BE>
        + CKKSAddOps<BE>
        + CKKSConjugateOps<BE>
        + CKKSCopyOps<BE>
        + CKKSSubOps<BE>
        + CKKSNegOps<BE>
        + CKKSPow2Ops<BE>
        + CKKSRescaleOps<BE>
        + CKKSRotateOps<BE>
        + CKKSMulOps<BE>
        + GLWEAutomorphism<BE>
        + GLWEAutomorphismKeyEncryptSk<BE>
        + GLWEAutomorphismKeyPreparedFactory<BE>
        + CKKSMulAddOps<BE>
        + ModuleN
        + GLWEShift<BE>
        + GLWEMulPlain<BE>
        + GLWEMulConst<BE>
        + GLWERotate<BE>
        + GLWETensoring<BE>
        + GLWETensorKeyEncryptSk<BE>
        + GLWETensorKeyPreparedFactory<BE>
        + poulpy_hal::api::VecZnxLshBackend<BE>
        + poulpy_hal::api::VecZnxLshTmpBytes
        + poulpy_hal::api::VecZnxRshBackend<BE>
        + poulpy_hal::api::VecZnxRshAddIntoBackend<BE>
        + poulpy_hal::api::VecZnxRshSubBackend<BE>
        + poulpy_hal::api::VecZnxRshTmpBytes,
{
    /// Creates a base context with a prepared secret key and two test messages.
    pub fn new(params: CKKSTestParams, rotations: &[i64]) -> Self
    where
        E: NegacyclicFFTNew<F> + Send + Sync + 'static,
    {
        let module = Module::<BE>::new(params.n as u64);
        let host_module = Module::<HostBytesBackend>::new(params.n as u64);
        let m = module.n() / 2;
        let glwe_infos = params.glwe_layout();
        let tsk_infos = params.tsk_layout();
        let atk_infos = params.atk_layout();

        let mut xa = Source::new([1u8; 32]);
        let mut xe = Source::new([2u8; 32]);

        let mut source_xs = Source::new([0u8; 32]);
        let mut sk_raw = module.glwe_secret_alloc_from_infos(&glwe_infos);
        sk_raw.fill_ternary_hw(params.hw, &mut source_xs);
        let mut sk = module.glwe_secret_prepared_alloc_from_infos(&glwe_infos);
        module.glwe_secret_prepare(&mut sk, &sk_raw);

        let mut ct_infos = host_module.ckks_ciphertext_alloc_from_infos(&params.glwe_layout());
        ct_infos.set_meta(params.prec);
        let pt_prec = Self::PT_PREC;
        let scratch_size = module
            .ckks_encrypt_sk_tmp_bytes(&ct_infos)
            .max(module.ckks_decrypt_tmp_bytes(&ct_infos))
            .max(module.ckks_add_tmp_bytes())
            .max(module.ckks_add_pt_vec_tmp_bytes())
            .max(module.ckks_add_pt_const_tmp_bytes())
            .max(module.ckks_copy_tmp_bytes())
            .max(module.ckks_sub_tmp_bytes())
            .max(module.ckks_sub_pt_vec_tmp_bytes())
            .max(module.ckks_sub_pt_const_tmp_bytes())
            .max(module.ckks_neg_tmp_bytes())
            .max(module.ckks_mul_pow2_tmp_bytes())
            .max(module.ckks_div_pow2_tmp_bytes())
            .max(module.ckks_mul_i_tmp_bytes())
            .max(module.ckks_div_i_tmp_bytes())
            .max(module.ckks_rescale_tmp_bytes())
            .max(module.ckks_align_tmp_bytes())
            .max(module.ckks_mul_tmp_bytes(&ct_infos, &tsk_infos))
            .max(module.ckks_square_tmp_bytes(&ct_infos, &tsk_infos))
            .max(module.ckks_mul_pt_vec_tmp_bytes(&ct_infos, &ct_infos, &pt_prec))
            .max(module.ckks_mul_pt_const_tmp_bytes(&ct_infos, &ct_infos, &pt_prec))
            .max(module.prepare_tensor_key_tmp_bytes(&tsk_infos))
            .max(module.glwe_tensor_key_encrypt_sk_tmp_bytes(&tsk_infos))
            .max(module.ckks_rotate_tmp_bytes(&ct_infos, &atk_infos))
            .max(module.ckks_conjugate_tmp_bytes(&ct_infos, &atk_infos))
            .max(module.glwe_automorphism_key_encrypt_sk_tmp_bytes(&atk_infos))
            .max(module.glwe_automorphism_key_prepare_tmp_bytes(&atk_infos));

        let mut scratch = ScratchOwned::<BE>::alloc(scratch_size);

        let mut tsk = module.glwe_tensor_key_alloc_from_infos(&tsk_infos);
        {
            let mut scratch_local = scratch.borrow();
            module.glwe_tensor_key_encrypt_sk(&mut tsk, &sk_raw, &tsk_infos, &mut xa, &mut xe, &mut scratch_local);
        }
        let mut tsk_prepared = module.alloc_tensor_key_prepared_from_infos(&tsk_infos);
        {
            let mut scratch_local = scratch.borrow();
            module.prepare_tensor_key(&mut tsk_prepared, &tsk, &mut scratch_local);
        }

        // Store keys by the public index used by operations/tests:
        // rotation shift `k` for slot rotations, and `-1` for conjugation.
        let mut automorphism_indices: Vec<i64> = rotations.to_vec();
        automorphism_indices.push(-1);
        automorphism_indices.sort();
        automorphism_indices.dedup();

        let mut atks = HashMap::new();
        for &index in &automorphism_indices {
            let mut atk = module.glwe_automorphism_key_alloc_from_infos(&atk_infos);
            let galois_element = if index == -1 { -1 } else { module.galois_element(index) };
            {
                let mut scratch_local = scratch.borrow();
                module.glwe_automorphism_key_encrypt_sk(
                    &mut atk,
                    galois_element,
                    &sk_raw,
                    &atk_infos,
                    &mut xa,
                    &mut xe,
                    &mut scratch_local,
                );
            }
            let mut atk_prepared = module.glwe_automorphism_key_prepared_alloc_from_infos(&atk_infos);
            {
                let mut scratch_local = scratch.borrow();
                module.glwe_automorphism_key_prepare(&mut atk_prepared, &atk, &mut scratch_local);
            }
            atks.insert(index, atk_prepared);
        }

        let tau = Self::to_scalar(TAU);
        let quarter = Self::to_scalar(0.25);
        let five = Self::to_scalar(5.0);
        let three = Self::to_scalar(3.0);
        let two = Self::to_scalar(2.0);
        let m_f = F::from_usize(m).expect("usize -> scalar conversion should succeed");

        let re1_hp: Vec<F> = (0..m)
            .map(|i| {
                let i_f = F::from_usize(i).expect("usize -> scalar conversion should succeed");
                (tau * (i_f + quarter) / m_f).cos()
            })
            .collect();
        let im1_hp: Vec<F> = (0..m)
            .map(|i| {
                let i_f = F::from_usize(i).expect("usize -> scalar conversion should succeed");
                (tau * (i_f + quarter) / m_f).sin()
            })
            .collect();
        let re2_hp: Vec<F> = (0..m)
            .map(|i| {
                let i_f = F::from_usize(i).expect("usize -> scalar conversion should succeed");
                (tau * (five * i_f + three) / (two * m_f)).cos()
            })
            .collect();
        let im2_hp: Vec<F> = (0..m)
            .map(|i| {
                let i_f = F::from_usize(i).expect("usize -> scalar conversion should succeed");
                (tau * (five * i_f + three) / (two * m_f)).sin()
            })
            .collect();

        let encoder = Encoder::new::<F>(m).unwrap();
        Self {
            module,
            host_module,
            encoder,
            params,
            sk,
            tsk: tsk_prepared,
            atks,
            scratch_size,
            re1: re1_hp,
            im1: im1_hp,
            re2: re2_hp,
            im2: im2_hp,
            _scalar: PhantomData,
        }
    }
}

impl<BE: TestBackend, F: TestScalar, E: NegacyclicFFT<F>> TestContext<BE, F, E> {
    const ADD_SUB_CONST_RE: f64 = 0.314_159_265_358_979_3;
    const ADD_SUB_CONST_IM: f64 = -0.271_828_182_845_904_5;
    const MUL_CONST_RE: f64 = 0.271_828_182_845_904_5;
    const MUL_CONST_IM: f64 = -0.141_421_356_237_309_5;

    pub fn test_vector(&self, which: TestVector) -> (&[F], &[F]) {
        match which {
            TestVector::First => (&self.re1, &self.im1),
            TestVector::Second => (&self.re2, &self.im2),
        }
    }

    pub fn sample_add_sub_const(&self) -> (f64, f64) {
        (Self::ADD_SUB_CONST_RE, Self::ADD_SUB_CONST_IM)
    }

    pub fn sample_mul_const(&self) -> (f64, f64) {
        (Self::MUL_CONST_RE, Self::MUL_CONST_IM)
    }

    pub fn quantized_const(&self, re: f64, im: f64, log_delta: usize) -> (F, F) {
        let scale = Self::to_scalar(2.0).powi(log_delta as i32);
        let re = (Self::to_scalar(re) * scale).round() / scale;
        let im = (Self::to_scalar(im) * scale).round() / scale;
        (re, im)
    }

    pub fn add_sub_const(&self) -> (F, F) {
        let (re, im) = self.sample_add_sub_const();
        self.quantized_const_pt(re, im)
    }

    pub fn mul_const_pt(&self) -> (F, F) {
        let (re, im) = self.sample_mul_const();
        self.quantized_const_pt(re, im)
    }

    pub fn quantized_const_pt(&self, re: f64, im: f64) -> (F, F) {
        self.quantized_const(re, im, self.meta_pt().log_delta)
    }

    pub fn cst(&self, re: Option<f64>, im: Option<f64>, prec: CKKSMeta) -> CKKSPlaintext<Vec<u8>> {
        let coeff_count = if im.is_some() { 2 } else { 1 };
        let mut pt = self.host_module.ckks_pt_coeffs_alloc(coeff_count, self.base2k(), prec);
        let mut packed_coeffs = vec![F::zero(); coeff_count];
        if let Some(re) = re {
            packed_coeffs[0] = Self::to_scalar(re);
        }
        if let Some(im) = im {
            packed_coeffs[1] = Self::to_scalar(im);
        }
        pt.encode_host_floats(&packed_coeffs).unwrap();
        pt
    }

    pub fn add_sub_const_pt(&self) -> CKKSPlaintext<Vec<u8>> {
        let (re, im) = self.sample_add_sub_const();
        self.cst(Some(re), Some(im), self.meta_pt())
    }

    pub fn const_full(&self, re: Option<f64>, im: Option<f64>, prec: CKKSMeta) -> CKKSPlaintext<Vec<u8>> {
        let mut pt = self.host_module.ckks_pt_vec_alloc(self.base2k(), prec);
        let mut coeffs = vec![F::zero(); self.degree().as_usize()];
        if let Some(re) = re {
            coeffs[0] = Self::to_scalar(re);
        }
        if let Some(im) = im {
            coeffs[self.m()] = Self::to_scalar(im);
        }
        pt.encode_host_floats(&coeffs).unwrap();
        pt
    }

    pub fn mul_const_full_pt(&self) -> CKKSPlaintext<Vec<u8>> {
        let (re, im) = self.sample_mul_const();
        self.const_full(Some(re), Some(im), self.meta_pt())
    }

    pub fn tsk(&self) -> &GLWETensorKeyPrepared<BE::OwnedBuf, BE> {
        &self.tsk
    }

    pub fn atks(&self) -> &HashMap<i64, GLWEAutomorphismKeyPrepared<BE::OwnedBuf, BE>> {
        &self.atks
    }

    pub fn atk(&self, index: i64) -> &GLWEAutomorphismKeyPrepared<BE::OwnedBuf, BE> {
        self.atks()
            .get(&index)
            .unwrap_or_else(|| panic!("missing automorphism key for index {index}"))
    }

    /// Encodes and encrypts complex slot values into a fresh ciphertext.
    pub fn encrypt<'a>(&self, k: usize, re: &[F], im: &[F], scratch: &mut ScratchArena<'a, BE>) -> CKKSCiphertext<Vec<u8>>
    where
        Module<BE>: CKKSEncrypt<BE>,
        ScratchArena<'a, BE>: ScratchArenaTakeCore<'a, BE>,
        BE: 'a,
    {
        self.encrypt_with_prec(k, re, im, self.meta(), scratch)
    }

    pub fn encrypt_with_prec<'a>(
        &self,
        k: usize,
        re: &[F],
        im: &[F],
        prec: CKKSMeta,
        scratch: &mut ScratchArena<'a, BE>,
    ) -> CKKSCiphertext<Vec<u8>>
    where
        Module<BE>: CKKSEncrypt<BE>,
        ScratchArena<'a, BE>: ScratchArenaTakeCore<'a, BE>,
        BE: 'a,
    {
        let mut pt = self.host_module.ckks_pt_vec_alloc(self.base2k(), prec);
        self.encoder.encode_reim(&mut pt, re, im).unwrap();

        let mut ct = self.alloc_ct(k);
        let mut xa = Source::new([3u8; 32]);
        let mut xe = Source::new([4u8; 32]);

        let mut layout = self.params.glwe_layout().layout;
        layout.k = k.into();
        let enc_infos = EncryptionLayout::new_from_default_sigma(layout).unwrap();

        self.module
            .ckks_encrypt_sk(&mut ct, &pt, &self.sk, &enc_infos, &mut xa, &mut xe, scratch)
            .unwrap();
        ct
    }

    /// Decrypts and decodes a ciphertext back to complex slot values.
    pub fn decrypt_decode(&self, ct: &CKKSCiphertext<Vec<u8>>, scratch: &mut ScratchArena<'_, BE>) -> (Vec<F>, Vec<F>)
    where
        Module<BE>: CKKSDecrypt<BE>,
        for<'a> ScratchArena<'a, BE>: ScratchArenaTakeCore<'a, BE>,
    {
        let prec = CKKSMeta {
            log_delta: ct.log_delta(),
            log_budget: ct.log_budget().min(self.params.prec.log_budget()),
        };
        let pt = self.decrypt_with_prec(ct, prec, scratch).unwrap();

        self.decode_pt(&pt)
    }

    /// Decrypts `ct` into a plaintext buffer allocated with the caller-provided
    /// metadata.
    ///
    /// Inputs:
    /// - `ct`: ciphertext to decrypt
    /// - `prec`: destination plaintext metadata used for extraction
    /// - `scratch`: temporary workspace
    ///
    /// Output:
    /// - returns the extracted CKKS plaintext on success
    ///
    /// Behavior:
    /// - this exercises the real `ckks_decrypt` API rather than the legacy
    ///   test helper projection used by [`Self::decrypt_decode`]
    ///
    /// Errors:
    /// - propagates any `ckks_decrypt` error, including base2k mismatch and
    ///   plaintext alignment failures
    pub fn decrypt_with_prec(
        &self,
        ct: &CKKSCiphertext<Vec<u8>>,
        prec: CKKSMeta,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> anyhow::Result<CKKSPlaintext<Vec<u8>>>
    where
        Module<BE>: CKKSDecrypt<BE>,
        for<'a> ScratchArena<'a, BE>: ScratchArenaTakeCore<'a, BE>,
    {
        let mut pt = self.host_module.ckks_pt_vec_alloc(ct.base2k(), prec);
        self.module.ckks_decrypt(&mut pt, ct, &self.sk, scratch)?;
        Ok(pt)
    }

    /// Decodes a CKKS ZNX plaintext into complex slot vectors.
    ///
    /// Inputs:
    /// - `pt`: plaintext in torus/ZNX form
    ///
    /// Output:
    /// - `(re, im)` slot vectors decoded through the current test encoder
    ///
    /// Behavior:
    /// - decodes the host-side ZNX plaintext back into slot-domain real and
    ///   imaginary vectors
    ///
    /// Errors:
    /// - this helper unwraps internal conversion/decoder results and therefore
    ///   panics instead of returning an error in tests
    pub fn decode_pt(&self, pt: &CKKSPlaintext<Vec<u8>>) -> (Vec<F>, Vec<F>) {
        let m = self.params.n / 2;
        let mut re = vec![F::zero(); m];
        let mut im = vec![F::zero(); m];
        self.encoder.decode_reim(pt, &mut re, &mut im).unwrap();

        (re, im)
    }

    /// Allocates enough scratch for encrypt + decrypt.
    pub fn alloc_scratch(&self) -> ScratchOwned<BE>
    where
        ScratchOwned<BE>: ScratchOwnedAlloc<BE>,
    {
        ScratchOwned::<BE>::alloc(self.scratch_size)
    }

    /// Returns element-wise (re1+re2, im1+im2).
    pub fn want_add(&self) -> (Vec<F>, Vec<F>) {
        self.want_add_from(&self.re1, &self.im1, &self.re2, &self.im2)
    }

    pub fn want_add_from(&self, a_re: &[F], a_im: &[F], b_re: &[F], b_im: &[F]) -> (Vec<F>, Vec<F>) {
        let re = a_re.iter().zip(b_re.iter()).map(|(a, b)| *a + *b).collect();
        let im = a_im.iter().zip(b_im.iter()).map(|(a, b)| *a + *b).collect();
        (re, im)
    }

    pub fn want_sub(&self) -> (Vec<F>, Vec<F>) {
        self.want_sub_from(&self.re1, &self.im1, &self.re2, &self.im2)
    }

    pub fn want_sub_from(&self, a_re: &[F], a_im: &[F], b_re: &[F], b_im: &[F]) -> (Vec<F>, Vec<F>) {
        let re = a_re.iter().zip(b_re.iter()).map(|(a, b)| *a - *b).collect();
        let im = a_im.iter().zip(b_im.iter()).map(|(a, b)| *a - *b).collect();
        (re, im)
    }

    pub fn want_neg(&self) -> (Vec<F>, Vec<F>) {
        let re = self.re1.iter().copied().map(|x| -x).collect();
        let im = self.im1.iter().copied().map(|x| -x).collect();
        (re, im)
    }

    pub fn want_mul_pow2(&self, bits: usize) -> (Vec<F>, Vec<F>) {
        let scale = Self::to_scalar((1u64 << bits) as f64);
        let re = self.re1.iter().copied().map(|x| x * scale).collect();
        let im = self.im1.iter().copied().map(|x| x * scale).collect();
        (re, im)
    }

    pub fn want_div_pow2(&self, bits: usize) -> (Vec<F>, Vec<F>) {
        let scale = Self::to_scalar((1u64 << bits) as f64);
        let re = self.re1.iter().copied().map(|x| x / scale).collect();
        let im = self.im1.iter().copied().map(|x| x / scale).collect();
        (re, im)
    }

    pub fn want_mul_i(&self) -> (Vec<F>, Vec<F>) {
        let re = self.im1.iter().copied().map(|x| -x).collect();
        let im = self.re1.clone();
        (re, im)
    }

    pub fn want_div_i(&self) -> (Vec<F>, Vec<F>) {
        let re = self.im1.clone();
        let im = self.re1.iter().copied().map(|x| -x).collect();
        (re, im)
    }

    pub fn scale_slots(&self, re: &[F], im: &[F], bits: isize) -> (Vec<F>, Vec<F>) {
        let scale = Self::to_scalar(2.0).powi(bits as i32);
        let re = re.iter().copied().map(|x| x * scale).collect();
        let im = im.iter().copied().map(|x| x * scale).collect();
        (re, im)
    }

    pub fn want_conjugate(&self) -> (Vec<F>, Vec<F>) {
        let re = self.re1.clone();
        let im = self.im1.iter().copied().map(|x| -x).collect();
        (re, im)
    }

    pub fn want_rotate(&self, k: i64) -> (Vec<F>, Vec<F>) {
        let m = self.params.n / 2;
        let re = (0..m)
            .map(|j| self.re1[((j as i64 + k).rem_euclid(m as i64)) as usize])
            .collect();
        let im = (0..m)
            .map(|j| self.im1[((j as i64 + k).rem_euclid(m as i64)) as usize])
            .collect();
        (re, im)
    }

    pub fn want_square(&self) -> (Vec<F>, Vec<F>) {
        self.want_square_from(&self.re1, &self.im1)
    }

    pub fn want_square_from(&self, re_in: &[F], im_in: &[F]) -> (Vec<F>, Vec<F>) {
        let m = self.params.n / 2;
        let mut re = Vec::with_capacity(m);
        let mut im = Vec::with_capacity(m);

        for i in 0..m {
            let re1 = re_in[i];
            let im1 = im_in[i];

            re.push(re1 * re1 - im1 * im1);
            im.push(Self::to_scalar(2.0) * re1 * im1);
        }
        (re, im)
    }

    pub fn want_mul(&self) -> (Vec<F>, Vec<F>) {
        self.want_mul_from(&self.re1, &self.im1, &self.re2, &self.im2)
    }

    pub fn want_mul_from(&self, a_re: &[F], a_im: &[F], b_re: &[F], b_im: &[F]) -> (Vec<F>, Vec<F>) {
        let m = self.params.n / 2;
        let mut re = Vec::with_capacity(m);
        let mut im = Vec::with_capacity(m);

        for i in 0..m {
            let re1 = a_re[i];
            let im1 = a_im[i];
            let re2 = b_re[i];
            let im2 = b_im[i];

            re.push(re1 * re2 - im1 * im2);
            im.push(re1 * im2 + re2 * im1);
        }
        (re, im)
    }

    pub fn want_add_const_from(&self, a_re: &[F], a_im: &[F], c_re: F, c_im: F) -> (Vec<F>, Vec<F>) {
        let re = a_re.iter().copied().map(|x| x + c_re).collect();
        let im = a_im.iter().copied().map(|x| x + c_im).collect();
        (re, im)
    }

    pub fn want_mul_const_from(&self, a_re: &[F], a_im: &[F], c_re: F, c_im: F) -> (Vec<F>, Vec<F>) {
        let m = self.params.n / 2;
        let mut re = Vec::with_capacity(m);
        let mut im = Vec::with_capacity(m);

        for i in 0..m {
            re.push(a_re[i] * c_re - a_im[i] * c_im);
            im.push(a_re[i] * c_im + a_im[i] * c_re);
        }

        (re, im)
    }

    /// Allocates a ciphertext with one fewer limb than the default (k − base2k).
    pub fn alloc_ct(&self, k: usize) -> CKKSCiphertext<Vec<u8>> {
        let mut layout = self.params.glwe_layout();
        layout.layout.k = k.into();
        self.host_module.ckks_ciphertext_alloc_from_infos(&layout)
    }

    /// Returns a representative ciphertext infos object carrying both GLWE
    /// layout information and CKKS metadata for tmp-bytes estimation helpers.
    pub fn ct_infos(&self) -> CKKSCiphertext<Vec<u8>> {
        let mut ct = self.alloc_ct(self.max_k());
        ct.set_meta(self.meta());
        ct
    }

    /// Encodes (re2, im2) into a ZNX plaintext (IFFT + quantise).
    pub fn encode_pt(&self, re: &[F], im: &[F]) -> CKKSPlaintext<Vec<u8>> {
        self.encode_pt_with_prec(re, im, self.meta_pt())
    }

    pub fn encode_pt_with_prec(&self, re: &[F], im: &[F], prec: CKKSMeta) -> CKKSPlaintext<Vec<u8>> {
        let mut pt = self.host_module.ckks_pt_vec_alloc(self.base2k(), prec);
        self.encoder.encode_reim(&mut pt, re, im).unwrap();
        pt
    }

    pub fn quantized_slots(&self, re: &[F], im: &[F], prec: CKKSMeta) -> (Vec<F>, Vec<F>) {
        let pt = self.encode_pt_with_prec(re, im, prec);
        let m = self.params.n / 2;
        let mut re_out = vec![F::zero(); m];
        let mut im_out = vec![F::zero(); m];
        self.encoder.decode_reim(&pt, &mut re_out, &mut im_out).unwrap();
        (re_out, im_out)
    }

    pub fn quantized_pt_slots(&self, re: &[F], im: &[F]) -> (Vec<F>, Vec<F>) {
        self.quantized_slots(re, im, self.meta_pt())
    }

    pub fn quantized_vector(&self, which: TestVector, log_delta: usize) -> (Vec<F>, Vec<F>) {
        let (re, im) = self.test_vector(which);
        let scale = ((log_delta as isize) - (self.meta().log_delta as isize)) as i32;
        let factor = Self::to_scalar(2.0).powi(scale);
        let re_scaled = re.iter().copied().map(|x| x * factor).collect::<Vec<_>>();
        let im_scaled = im.iter().copied().map(|x| x * factor).collect::<Vec<_>>();
        self.quantized_slots(&re_scaled, &im_scaled, self.precision_at(log_delta))
    }

    pub fn pt_vector(&self, which: TestVector) -> (Vec<F>, Vec<F>) {
        let (re, im) = self.test_vector(which);
        self.quantized_pt_slots(re, im)
    }

    /// Returns the minimum expected average log2 precision for a standard-ring
    /// CKKS value encoded at `log_delta`.
    ///
    /// This follows the crate's dynamic precision heuristic:
    /// `log_delta - log2(ring_degree) - 2`, clamped at zero.
    pub fn expected_log2_precision(&self, log_delta: usize) -> f64 {
        expected_log2_precision(log_delta, self.degree().as_usize())
    }

    /// Asserts that `got` and `want` meet the expected average precision for
    /// the provided `log_delta`.
    pub fn assert_precision_for_log_delta(&self, label: &str, got: &[F], want: &[F], log_delta: usize) {
        assert_precision(label, got, want, log_delta, self.degree().as_usize());
    }

    /// Decrypts `ct`, decodes, and asserts both channels meet the expected
    /// average precision for the caller-provided `log_delta`.
    pub fn assert_decrypt_precision_at_log_delta(
        &self,
        label: &str,
        ct: &CKKSCiphertext<Vec<u8>>,
        want_re: &[F],
        want_im: &[F],
        log_delta: usize,
        scratch: &mut ScratchArena<'_, BE>,
    ) where
        Module<BE>: CKKSDecrypt<BE>,
        for<'a> ScratchArena<'a, BE>: ScratchArenaTakeCore<'a, BE>,
    {
        let (re_out, im_out) = self.decrypt_decode(ct, scratch);
        self.assert_precision_for_log_delta(&format!("{label} re"), &re_out, want_re, log_delta);
        self.assert_precision_for_log_delta(&format!("{label} im"), &im_out, want_im, log_delta);
    }

    /// Decrypts `ct`, decodes, and asserts both channels meet the expected
    /// average precision for `ct.log_delta()`.
    pub fn assert_decrypt_precision(
        &self,
        label: &str,
        ct: &CKKSCiphertext<Vec<u8>>,
        want_re: &[F],
        want_im: &[F],
        scratch: &mut ScratchArena<'_, BE>,
    ) where
        Module<BE>: CKKSDecrypt<BE>,
        for<'a> ScratchArena<'a, BE>: ScratchArenaTakeCore<'a, BE>,
    {
        self.assert_decrypt_precision_at_log_delta(label, ct, want_re, want_im, ct.log_delta(), scratch);
    }
}

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

/// Additional safety margin applied on top of the
/// `log_delta - log2(ring_degree)` bound.
const PRECISION_GUARD_BITS: f64 = 2.0;

/// Returns the minimum expected average log2 precision for standard-ring CKKS
/// at the given ring degree and scaling precision.
pub fn expected_log2_precision(log_delta: usize, degree: usize) -> f64 {
    (log_delta as f64 - degree.ilog2() as f64 - PRECISION_GUARD_BITS).max(0.0)
}

/// Computes per-slot log2 precision statistics.
///
/// Precision is evaluated slot-by-slot as `-log2(abs(err))`, with exact
/// matches capped at `log_delta`.
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

/// Asserts that `got` and `want` meet the expected average log2 precision for
/// a standard-ring CKKS value encoded at `log_delta`.
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

pub fn assert_ct_meta(label: &str, ct: &CKKSCiphertext<impl Data>, log_delta: usize, log_budget: usize) {
    assert_eq!(ct.log_delta(), log_delta, "{label}: unexpected log_delta");
    assert_eq!(ct.log_budget(), log_budget, "{label}: unexpected log_budget");
}

pub fn assert_ckks_error(label: &str, err: &anyhow::Error, want: CKKSCompositionError) {
    let got = err.downcast_ref::<CKKSCompositionError>();
    assert_eq!(got, Some(&want), "{label}: unexpected error: {err}");
}

pub fn assert_unary_output_meta(label: &str, ct: &CKKSCiphertext<impl Data>, input: &CKKSCiphertext<impl Data>) {
    assert_ct_meta(label, ct, input.log_delta(), input.log_budget() - ct.offset_unary(input));
}

pub fn assert_binary_output_meta(
    label: &str,
    ct: &CKKSCiphertext<impl Data>,
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

pub fn assert_mul_ct_output_meta(label: &str, ct: &CKKSCiphertext<impl Data>, a: &impl CKKSInfos, b: &impl CKKSInfos) {
    let log_budget = a.log_budget().min(b.log_budget()) - a.log_delta().max(b.log_delta());
    let log_delta = a.log_delta().min(b.log_delta());
    let offset = (log_budget + log_delta).saturating_sub(ct.max_k().as_usize());
    assert_ct_meta(label, ct, log_delta, log_budget - offset);
}

pub fn assert_mul_pt_output_meta(label: &str, ct: &CKKSCiphertext<impl Data>, a: &impl CKKSInfos, b: &impl CKKSInfos) {
    let log_budget = a.log_budget() - a.log_delta().min(b.log_delta());
    let log_delta = a.log_delta().max(b.log_delta());
    let offset = (log_budget + log_delta).saturating_sub(ct.max_k().as_usize());
    assert_ct_meta(label, ct, log_delta, log_budget - offset);
}
