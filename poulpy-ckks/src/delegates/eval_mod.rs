use crate::CKKSResult as Result;
use poulpy_core::layouts::IntPolyInfos;
use poulpy_core::layouts::{
    BSGSMeta, Base2K, Degree, GGLWEInfos, GLWEInfos, GLWETensorKeyPrepared, GLWEToBackendMut, GLWEToBackendRef, LWEInfos, Rank,
    SetBSGSMeta, TorusPrecision,
};
use poulpy_hal::api::{CnvPVecBytesOf, Convolution, VecZnxBigNormalizeTmpBytes, VecZnxRshTmpBytes};
use poulpy_hal::layouts::{Backend, Module, ScratchArena};

use crate::SlotsKind;
use crate::{
    CKKSCtBounds, CKKSInfos, CKKSMeta, SetCKKSInfos,
    api::{CKKSAddOps, CKKSCopyOps, CKKSEvalModOps, CKKSMulOps, CKKSSubOps, PolynomialInputTransform},
    layouts::eval_mod::{EvalMod, EvalModBsgs},
    oep::CKKSEvalModImpl,
};

#[derive(Clone, Copy)]
struct EvalModWorkCtInfos {
    n: Degree,
    base2k: Base2K,
    rank: Rank,
    max_size: usize,
    k: TorusPrecision,
    meta: CKKSMeta,
}

impl LWEInfos for EvalModWorkCtInfos {
    fn base2k(&self) -> Base2K {
        self.base2k
    }

    fn n(&self) -> Degree {
        self.n
    }

    fn max_size(&self) -> usize {
        self.max_size
    }

    fn k(&self) -> TorusPrecision {
        self.k
    }
}

impl GLWEInfos for EvalModWorkCtInfos {
    fn rank(&self) -> Rank {
        self.rank
    }
}

impl CKKSInfos for EvalModWorkCtInfos {
    fn meta(&self) -> CKKSMeta {
        self.meta
    }
}

impl<BE: Backend + CKKSEvalModImpl<BE>> CKKSEvalModOps<BE> for Module<BE>
where
    Module<BE>: CKKSAddOps<BE>
        + CKKSSubOps<BE>
        + CKKSMulOps<BE>
        + CKKSCopyOps<BE>
        + CnvPVecBytesOf
        + Convolution<BE>
        + VecZnxRshTmpBytes
        + VecZnxBigNormalizeTmpBytes,
{
    fn ckks_eval_mod_tmp_bytes<R, C, P, F, T>(&self, res: &R, ct: &C, params: &EvalMod<F, P>, tsk: &T) -> usize
    where
        R: CKKSCtBounds,
        C: CKKSCtBounds,
        P: CKKSCtBounds,
        T: GGLWEInfos,
    {
        let work_k = ct.k().as_usize().max(ct.log_budget() + params.plan.f_mod_log_delta);
        let work = EvalModWorkCtInfos {
            n: ct.n(),
            base2k: ct.base2k(),
            rank: ct.rank(),
            max_size: work_k.div_ceil(ct.base2k().as_usize()).max(1),
            // Total torus width = budget carried into eval_mod + the plan scale.
            k: (ct.log_budget() + params.plan.f_mod_log_delta).into(),
            meta: CKKSMeta {
                log_sparsity: ct.log_sparsity(),
                log_delta: params.plan.f_mod_log_delta,
                slots: SlotsKind::Complex,
            },
        };

        let cols: usize = (work.rank() + 1).into();
        let compact_work = BE::bytes_of_vec_znx(work.n().into(), cols, work.max_size());
        // The giant step hoists the prepared `X^{gsp}` right operand, kept alive
        // across the baby-step pairs that share it.
        let hoisted_right = self.bytes_of_cnv_pvec_right(cols, work.max_size());
        let bsgs_giant = self
            .ckks_mul_tmp_bytes(&work, &work, &work, tsk)
            .max(self.ckks_add_tmp_bytes())
            + 3 * compact_work
            + hoisted_right;
        let square_scope = (self.ckks_square_tmp_bytes(&work, &work, tsk) + compact_work).max(
            // Scratch is a physical working-set budget: size the square-scope
            // copy off `res`'s allocated capacity, the upper bound on the limbs
            // any runtime re-expansion can expose.
            self.ckks_square_tmp_bytes(res, res, tsk)
                + BE::bytes_of_vec_znx(res.n().into(), (res.rank() + 1).into(), res.max_size()),
        );
        // Identity base polynomials transfer an owned, relabelled input directly
        // into the power basis. Scratch only needs a full working ciphertext for
        // a transformed base or for the optional inverse composition.
        let needs_work_copy = params.f_mod_inv_bsgs.is_some()
            || match &params.f_mod_bsgs {
                EvalModBsgs::Real(poly) => poly.input_transform() != PolynomialInputTransform::Identity,
                EvalModBsgs::Complex(poly) => {
                    poly.re.input_transform() != PolynomialInputTransform::Identity
                        || poly.im.input_transform() != PolynomialInputTransform::Identity
                }
            };
        // Working set of the fused baby-step linear combination.
        let pt_size_max = {
            let poly_max =
                |poly: &crate::polynomial::BSGSPolynomial<P>| poly.baby_steps().iter().map(|c| c.size()).max().unwrap_or(0);
            let base = match &params.f_mod_bsgs {
                EvalModBsgs::Real(poly) => poly_max(poly),
                EvalModBsgs::Complex(poly) => poly_max(&poly.re).max(poly_max(&poly.im)),
            };
            base.max(params.f_mod_inv_bsgs.as_ref().map_or(0, poly_max))
        };
        let fused_baby =
            crate::default::polynomial_evaluation::eval_baby_linear_combination_tmp_bytes(self, work.max_size(), pt_size_max);

        usize::from(needs_work_copy) * compact_work
            + self
                .ckks_copy_tmp_bytes()
                .max(self.ckks_add_pt_const_tmp_bytes())
                .max(self.ckks_sub_pt_const_tmp_bytes())
                .max(bsgs_giant)
                .max(square_scope)
                .max(fused_baby)
    }

    fn ckks_eval_mod<R, C, P, F>(
        &self,
        res: &mut R,
        ct: &C,
        params: &EvalMod<F, P>,
        tsk: &GLWETensorKeyPrepared<BE::OwnedBuf, BE>,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        R: GLWEToBackendMut<BE> + GLWEToBackendRef<BE> + CKKSCtBounds + SetCKKSInfos + SetBSGSMeta,
        C: GLWEToBackendRef<BE> + CKKSCtBounds,
        P: GLWEToBackendRef<BE> + IntPolyInfos + CKKSCtBounds + BSGSMeta,
    {
        BE::ckks_eval_mod_impl::<R, C, P, F>(self, res, ct, params, tsk, scratch)
    }
}
