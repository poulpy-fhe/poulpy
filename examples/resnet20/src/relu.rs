use anyhow::Result;
use poulpy_ckks::{
    CKKSCtBounds, CoeffsMeta, SetCKKSInfos, SlotsKind,
    api::{
        BSGSPolynomialInfos, CKKSAddOps, CKKSAllOpsTmpBytes, CKKSComplexPolynomialEvaluationOps, CKKSConjugateOps,
        CKKSEncodingHostOps,
    },
    layouts::{CKKSCiphertextOwned, CKKSPlaintextAlloc, CKKSPlaintextOwned, ScratchArenaTakeCKKSCiphertext},
    polynomial::{Basis, ComplexBSGSPolynomial, Parity, Polynomial, SplitStrategy},
};
use poulpy_core::{
    GLWEBytesOf,
    layouts::{
        Base2K, GGLWEInfos, GLWEToBackendMut, GLWEToBackendRef, GetAutomorphismKey, GetTensorKey, IntPolyInfos, SetBSGSMeta, SetK,
    },
};
use poulpy_hal::layouts::{Backend, Module, ScratchArena};

use crate::model::RELU_TRIG;

/// The ReLU of [`relu`](crate::model::relu) on the unit circle the raise leaves in the slots: `z(a) + conj(z(a))` for
/// [`RELU_TRIG`], a BSGS evaluation in the powers of `a` and a conjugation, slot by slot.
pub struct Relu<BE: Backend> {
    z: ComplexBSGSPolynomial<CKKSPlaintextOwned<BE>>,
}

impl<BE: Backend> Relu<BE> {
    /// Encodes the polynomial at radix `base2k` and scale `2^log_delta`, the scale of the activations.
    pub fn new(module: &Module<BE>, base2k: Base2K, log_delta: usize, scratch: &mut ScratchArena<'_, BE>) -> Result<Self>
    where
        Module<BE>: CKKSPlaintextAlloc<BE> + CKKSEncodingHostOps<BE, f64>,
    {
        let meta = CoeffsMeta::from_delta_budget(log_delta, 2);
        let mut encode = |coeffs: Vec<f64>| {
            Polynomial::new_with_parity(Basis::Monomial, coeffs, Parity::Full).decompose_bsgs_with(
                SplitStrategy::MinDepth,
                |baby| {
                    let mut pt = module.ckks_plaintext_alloc(baby.len().into(), base2k, meta.k);
                    pt.set_meta(meta.meta);
                    module.ckks_encode_coeffs_host_into(&mut pt, baby, scratch)?;
                    Ok(pt)
                },
            )
        };
        Ok(Self {
            z: ComplexBSGSPolynomial {
                re: encode(RELU_TRIG.iter().map(|c| c.0).collect())?,
                im: encode(RELU_TRIG.iter().map(|c| c.1).collect())?,
            },
        })
    }

    pub fn apply_tmp_bytes<A, T, K>(&self, module: &Module<BE>, a: &A, tsk: &T, atk: &K) -> usize
    where
        Module<BE>: GLWEBytesOf<BE> + CKKSAllOpsTmpBytes<BE>,
        A: CKKSCtBounds,
        T: GGLWEInfos,
        K: GGLWEInfos,
    {
        module.glwe_bytes_of_from_infos(a)
            + module.ckks_all_ops_with_atk_tmp_bytes(a, BSGSPolynomialInfos::<BE>::baby_step(&self.z.re, 0), tsk, atk)
    }

    /// `res = z(a) + conj(z(a)) + offset`, in real slots. The sum vanishes where `a` is 1, on the slots the mask of the
    /// raise left out: `offset` holds the ReLU of zero on the others.
    #[allow(clippy::too_many_arguments)]
    pub fn apply<R, A, P, T, K>(
        &self,
        module: &Module<BE>,
        res: &mut R,
        a: &A,
        offset: &P,
        tsk: &T,
        keys: &K,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        Module<BE>: CKKSComplexPolynomialEvaluationOps<BE> + CKKSConjugateOps<BE> + CKKSAddOps<BE>,
        R: GLWEToBackendMut<BE> + GLWEToBackendRef<BE> + CKKSCtBounds + SetCKKSInfos + SetBSGSMeta + SetK,
        A: GLWEToBackendRef<BE> + CKKSCtBounds,
        P: GLWEToBackendRef<BE> + CKKSCtBounds + IntPolyInfos,
        T: GetTensorKey<BE>,
        K: GetAutomorphismKey<BE>,
        CKKSCiphertextOwned<BE>: GLWEToBackendMut<BE> + GLWEToBackendRef<BE> + CKKSCtBounds + SetCKKSInfos,
    {
        let (mut conjugate, mut scratch) = scratch.borrow().take_ckks_ciphertext_scratch(a, a.meta());
        res.set_k(a.k());
        module.ckks_eval_poly_complex_const_coeffs(res, a, &self.z, tsk, &mut scratch)?;
        module.ckks_conjugate_into(&mut conjugate, res, keys, &mut scratch)?;
        module.ckks_add_assign(res, &conjugate, &mut scratch)?;
        res.set_slots(SlotsKind::Real);
        module.ckks_add_pt_vec_assign(res, offset, &mut scratch)?;
        Ok(())
    }
}
