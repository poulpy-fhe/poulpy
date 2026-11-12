use poulpy_core::layouts::{
    Base2K, Degree, GGLWEInfos, GLWEInfos, GLWEPlaintextInfos, GLWEPlaintextMeta, LWEInfos, Rank, Scale, TorusPrecision,
};
use poulpy_core::oep::GLWEMulConstImpl;
use poulpy_core::{GLWEBytesOf, GLWECopy, GLWENormalize};
use poulpy_hal::api::{CnvOps, CnvPVecBytesOf, VecZnxBigNormalizeTmpBytes, VecZnxIdftApplyTmpBytes, VecZnxRshTmpBytes};
use poulpy_hal::layouts::{Backend, Module, Ring};

use crate::{
    CKKSCtBounds, CKKSInfos,
    api::{CKKSAddOps, CKKSCopyOps, CKKSMulAddOps, CKKSMulOps, CKKSPow2Ops, CKKSRoundOps, CKKSSubOps, PolynomialInputTransform},
    layouts::eval_mod::{EvalMod, EvalModBsgs},
    power_basis::populated_powers,
};
use crate::{SlotLayout, SlotsKind};

#[derive(Clone, Copy)]
struct EvalModWorkCtInfos {
    n: Degree,
    base2k: Base2K,
    rank: Rank,
    max_size: usize,
    k: TorusPrecision,
    meta: GLWEPlaintextMeta,
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

impl GLWEPlaintextInfos for EvalModWorkCtInfos {
    fn plaintext_meta(&self) -> Option<GLWEPlaintextMeta> {
        Some(self.meta)
    }
}

impl CKKSInfos for EvalModWorkCtInfos {}

pub fn ckks_eval_mod_tmp_bytes_reference<BE: Backend + GLWEMulConstImpl<Rg>, R, C, P, F, T, Rg: Ring>(
    module: &Module<BE>,
    res: &R,
    ct: &C,
    params: &EvalMod<F, P>,
    tsk: &T,
) -> usize
where
    Module<BE>: CKKSAddOps<BE, Rg>
        + CKKSSubOps<BE, Rg>
        + CKKSMulOps<BE, Rg>
        + crate::api::CKKSPow2Ops<BE, Rg>
        + GLWENormalize<BE, Rg>
        + CKKSCopyOps<BE, Rg>
        + GLWECopy<BE, Rg>
        + CnvPVecBytesOf
        + CnvOps<BE, Rg>
        + VecZnxRshTmpBytes
        + VecZnxBigNormalizeTmpBytes
        + VecZnxIdftApplyTmpBytes
        + GLWEBytesOf<BE>
        + CKKSRoundOps<BE, Rg>
        + CKKSMulAddOps<BE, Rg>,
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
        meta: GLWEPlaintextMeta {
            scale: Scale::Log(params.plan.f_mod_log_delta),
            slots: SlotsKind::Complex,
            log_sparsity: ct.log_sparsity(),
            slot_layout: ct.slot_layout(),
        },
    };

    let cols: usize = (work.rank() + 1).into();
    let compact_work = BE::vec_znx_bytes_of(work.n().into(), cols, work.max_size());
    let bsgs_giant = module
        .ckks_mul_tmp_bytes(&work, &work, &work, tsk)
        .max(module.ckks_add_tmp_bytes(work.max_size()))
        + 3 * compact_work;
    let square_scope = (module.ckks_square_tmp_bytes(&work, &work, tsk) + compact_work).max(
        // Scratch is a physical working-set budget: size the square-scope
        // copy off `res`'s allocated capacity, the upper bound on the limbs
        // any runtime re-expansion can expose.
        module.ckks_square_tmp_bytes(res, res, tsk)
            + BE::vec_znx_bytes_of(res.n().into(), (res.rank() + 1).into(), res.max_size()),
    );
    // Scratch holds the powers of the base polynomial, its transformed input, the input when its allocation is too
    // narrow, and a working ciphertext for the optional inverse composition.
    let powers = match &params.f_mod_bsgs {
        EvalModBsgs::Real(poly) => {
            populated_powers(poly.degree(), poly.log_split(), poly.parity(), poly.basis())
                + usize::from(poly.input_transform() != PolynomialInputTransform::Identity)
        }
        EvalModBsgs::Complex(poly)
            if poly.re.input_transform() == PolynomialInputTransform::Identity
                && poly.im.input_transform() == PolynomialInputTransform::Identity =>
        {
            populated_powers(poly.re.degree(), poly.re.log_split(), poly.re.parity(), poly.re.basis())
        }
        _ => 0,
    };
    // A replicated input that drops bits to the plan scale is rounded into a carved copy.
    let input_copy = ct.log_budget() + params.plan.f_mod_log_delta > ct.max_k().as_usize()
        || matches!(ct.slot_layout(), SlotLayout::Replicated(_)) && params.plan.f_mod_log_delta < ct.log_delta();
    let carved =
        (powers + usize::from(input_copy) + usize::from(params.f_mod_inv_bsgs.is_some())) * BE::scratch_aligned(compact_work);
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
        crate::oep::derived::polynomial_evaluation::eval_baby_linear_combination_tmp_bytes(module, work.max_size(), pt_size_max);

    // A replicated layout holds one working-width ciphertext while it takes
    // masked products, squares its last step or rounds between scales.
    let replicated = match ct.slot_layout() {
        SlotLayout::Replicated(_) => {
            let coeffs = match &params.f_mod_bsgs {
                EvalModBsgs::Real(poly) => poly.baby_step(0),
                EvalModBsgs::Complex(poly) => poly.re.baby_step(0),
            };
            let masked_products = module
                .ckks_mul_add_pt_vec_tmp_bytes(&work, &work, coeffs)
                .max(module.ckks_mul_pt_vec_tmp_bytes(&work, &work, coeffs));
            compact_work
                + masked_products
                    .max(module.ckks_square_tmp_bytes(&work, &work, tsk))
                    .max(module.ckks_mul_pow2_tmp_bytes(work.max_size()))
                    .max(module.ckks_copy_tmp_bytes(&work, res))
                    .max(module.ckks_round_tmp_bytes(res, &work))
                    .max(module.ckks_round_tmp_bytes(&work, ct))
                    .max(
                        crate::reference::polynomial_evaluation::eval_shaped_baby_linear_combination_tmp_bytes(
                            module,
                            &work,
                            pt_size_max,
                        ),
                    )
        }
        SlotLayout::Full => 0,
    };

    let steps = module
        .ckks_copy_tmp_bytes(&work, &work)
        // Polynomial evaluation finishes by copying into the caller's
        // destination, whose allocated capacity can exceed the work layout.
        .max(module.ckks_copy_tmp_bytes(res, &work))
        .max(module.ckks_copy_tmp_bytes(&work, res))
        .max(module.ckks_copy_tmp_bytes(&work, ct))
        .max(module.glwe_copy_tmp_bytes(&work, ct))
        .max(module.ckks_add_pt_const_tmp_bytes(work.max_size()))
        .max(module.ckks_sub_pt_const_tmp_bytes(work.max_size()))
        .max(module.ckks_sub_one_tmp_bytes(work.max_size()))
        .max(bsgs_giant)
        .max(square_scope)
        .max(crate::power_basis::ckks_power_product_tmp_bytes(module, &work, tsk))
        .max(fused_baby);
    carved + steps.max(replicated)
}
