use anyhow::Result;
use poulpy_ckks::{
    CKKSInfos, CoeffsMeta, SetCKKSInfos, SetGLWEPlaintextInfos, SlotsKind,
    api::{
        BSGSPolynomialInfos, CKKSAddOps, CKKSAllOpsTmpBytes, CKKSComplexPolynomialEvaluationOps, CKKSConjugateOps,
        CKKSEncodingHostOps,
    },
    layouts::{CKKSPlaintextAlloc, ScratchArenaTakeCKKSCiphertext},
    polynomial::{Basis, ComplexBSGSPolynomial, Parity, Polynomial, SplitStrategy},
};
use poulpy_core::{
    GLWEBytesOf,
    layouts::{Base2K, GGLWEInfos, GetAutomorphismKey, GetTensorKey, LWEInfos, SetK},
};
use poulpy_hal::layouts::{Module, ScratchArena};

use crate::{Ct, Pt, ServerBackend, model::RELU_TRIG};

/// Bits the encoded coefficients of the polynomial hold above the scale.
const LOG_COEFFS_RANGE: usize = 2;

/// The ReLU of [`relu`](crate::model::relu) on the unit circle the raise leaves in the slots: `z(a) + conj(z(a))` for
/// [`RELU_TRIG`], a BSGS evaluation in the powers of `a` and a conjugation, slot by slot.
pub(crate) struct Relu {
    z: ComplexBSGSPolynomial<Pt>,
}

impl Relu {
    /// Encodes the polynomial at radix `base2k` and scale `2^log_delta`, the scale of the activations.
    pub(crate) fn new(
        module: &Module<ServerBackend>,
        base2k: Base2K,
        log_delta: usize,
        scratch: &mut ScratchArena<'_, ServerBackend>,
    ) -> Result<Self> {
        let meta = CoeffsMeta::from_delta_budget(log_delta, LOG_COEFFS_RANGE);
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

    /// Scratch of [`Self::apply`] on `a`, with a tensor key of layout `tsk` and rotation keys of layout `atk`.
    pub(crate) fn apply_tmp_bytes<T, K>(&self, module: &Module<ServerBackend>, a: &Ct, tsk: &T, atk: &K) -> usize
    where
        T: GGLWEInfos,
        K: GGLWEInfos,
    {
        module.glwe_bytes_of_from_infos(a)
            + module.ckks_all_ops_with_atk_tmp_bytes(a, BSGSPolynomialInfos::<ServerBackend>::baby_step(&self.z.re, 0), tsk, atk)
    }

    /// `res = z(a) + conj(z(a)) + offset`, in real slots. The sum vanishes where `a` is 1, on the slots the mask of the
    /// raise left out: `offset` holds the ReLU of zero on the others.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn apply<T, K>(
        &self,
        module: &Module<ServerBackend>,
        res: &mut Ct,
        a: &Ct,
        offset: &Pt,
        tsk: &T,
        atk: &K,
        scratch: &mut ScratchArena<'_, ServerBackend>,
    ) -> Result<()>
    where
        T: GetTensorKey<ServerBackend>,
        K: GetAutomorphismKey<ServerBackend>,
    {
        let (mut conjugate, mut scratch) = scratch.borrow().take_ckks_ciphertext_scratch(a, a.meta());
        res.set_k(a.k());
        module.ckks_eval_poly_complex_const_coeffs(res, a, &self.z, tsk, &mut scratch)?;
        module.ckks_conjugate_into(&mut conjugate, res, atk, &mut scratch)?;
        module.ckks_add_assign(res, &conjugate, &mut scratch)?;
        res.set_slots(SlotsKind::Real);
        module.ckks_add_pt_vec_assign(res, offset, &mut scratch)?;
        Ok(())
    }
}
