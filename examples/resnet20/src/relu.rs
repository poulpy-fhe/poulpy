use anyhow::Result;
use poulpy_ckks::{
    CKKSCtBounds, CoeffsMeta, SetCKKSInfos,
    api::BSGSPolynomialInfos,
    api::{CKKSAddOps, CKKSAllOpsTmpBytes, CKKSEncodingHostOps, CKKSMulOps, CKKSPolynomialEvaluationOps},
    layouts::{CKKSCiphertextOwned, CKKSPlaintextAlloc, CKKSPlaintextOwned, ScratchArenaTakeCKKSCiphertext},
    polynomial::{BSGSPolynomial, Basis, Polynomial, SplitStrategy},
};
use poulpy_core::{
    GLWEBytesOf,
    layouts::{Base2K, GGLWEInfos, GLWEToBackendMut, GLWEToBackendRef, GetTensorKey, SetBSGSMeta, SetK},
};
use poulpy_hal::layouts::{Backend, Module, ScratchArena};

use crate::model::SIGN_STAGES;

/// `x + x * sign(x)`, twice the approximate ReLU of `x` in `[-1, 1]`, with the composite sign of
/// [`SIGN_STAGES`](crate::model::SIGN_STAGES): two BSGS evaluations and a product, slot by slot, on any ring.
pub struct Relu<BE: Backend> {
    stages: [BSGSPolynomial<CKKSPlaintextOwned<BE>>; 2],
}

impl<BE: Backend> Relu<BE> {
    /// Encodes the sign stages at radix `base2k` and scale `2^log_delta`, the scale of the activations: a product takes
    /// the smaller scale of its operands and spends the larger, so a level costs `log_delta` bits and leaves that scale.
    pub fn new(module: &Module<BE>, base2k: Base2K, log_delta: usize, scratch: &mut ScratchArena<'_, BE>) -> Result<Self>
    where
        Module<BE>: CKKSPlaintextAlloc<BE> + CKKSEncodingHostOps<BE, f64>,
    {
        // The coefficients of a stage stay below 2 in magnitude: a sign bit and an integer bit above the scale.
        let meta = CoeffsMeta::from_delta_budget(log_delta, 2);
        let mut encode = |coeffs: &[f64]| {
            Polynomial::new(Basis::Chebyshev, coeffs.to_vec()).decompose_bsgs_with(SplitStrategy::MinDepth, |baby| {
                let mut pt = module.ckks_plaintext_alloc(baby.len().into(), base2k, meta.k);
                pt.set_meta(meta.meta);
                module.ckks_encode_coeffs_host_into(&mut pt, baby, scratch)?;
                Ok(pt)
            })
        };
        Ok(Self {
            stages: [encode(&SIGN_STAGES[0])?, encode(&SIGN_STAGES[1])?],
        })
    }

    pub fn apply_tmp_bytes<A, T>(&self, module: &Module<BE>, a: &A, tsk: &T) -> usize
    where
        Module<BE>: GLWEBytesOf<BE> + CKKSAllOpsTmpBytes<BE>,
        A: CKKSCtBounds,
        T: GGLWEInfos,
    {
        2 * module.glwe_bytes_of_from_infos(a)
            + module.ckks_all_ops_tmp_bytes(a, BSGSPolynomialInfos::<BE>::baby_step(&self.stages[0], 0), tsk)
    }

    /// `res = a + a * sign(a)`.
    pub fn apply<R, A, H>(
        &self,
        module: &Module<BE>,
        res: &mut R,
        a: &A,
        tsk: &H,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        Module<BE>: CKKSPolynomialEvaluationOps<BE> + CKKSMulOps<BE> + CKKSAddOps<BE>,
        R: GLWEToBackendMut<BE> + CKKSCtBounds + SetCKKSInfos + SetBSGSMeta,
        A: GLWEToBackendRef<BE> + CKKSCtBounds,
        H: GetTensorKey<BE>,
        CKKSCiphertextOwned<BE>: GLWEToBackendMut<BE> + GLWEToBackendRef<BE> + CKKSCtBounds + SetCKKSInfos,
    {
        let (mut first, scratch_1) = scratch.borrow().take_ckks_ciphertext_scratch(a, a.meta());
        let (mut sign, mut scratch_2) = scratch_1.take_ckks_ciphertext_scratch(a, a.meta());
        module.ckks_eval_poly_real_const_coeffs(&mut first, a, &self.stages[0], tsk, &mut scratch_2)?;
        module.ckks_eval_poly_real_const_coeffs(&mut sign, &first, &self.stages[1], tsk, &mut scratch_2)?;
        first.set_k(a.k());
        module.ckks_mul_into(&mut first, a, &sign, tsk, &mut scratch_2)?;
        module.ckks_add_into(res, a, &first, &mut scratch_2)?;
        Ok(())
    }
}
