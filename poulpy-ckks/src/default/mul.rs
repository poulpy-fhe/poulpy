use crate::CKKSResult as Result;
use poulpy_core::layouts::IntPolyInfos;
use poulpy_core::{
    GLWECopy, GLWEMulConst, GLWEMulPlain, GLWERotate, GLWETensoring, GiantStepTensorBounds, ScratchArenaTakeCore,
    glwe_prepare_right,
    layouts::{
        GGLWEInfos, GLWEInfos, GLWELayout, GLWEPlaintextLayout, GLWETensorViewMut, GLWEToBackendMut, GLWEToBackendRef, LWEInfos,
        ModuleCoreAlloc, TorusPrecision,
        prepared::{GGLWEPreparedToBackendRef, GLWETensorKeyPreparedToBackendRef},
    },
};
use poulpy_hal::{
    api::{CnvPVecAlloc, Convolution, VecZnxCopyBackend},
    layouts::{Backend, ScratchArena},
};

use crate::SlotsKind;
use crate::{
    CKKSCtBounds, CKKSInfos, CKKSMeta, SetCKKSInfos,
    api::{CKKSAddOps, CKKSMulOps},
    checked_log_budget_sub, checked_mul_ct_log_budget, checked_mul_pt_log_budget, ckks_ensure,
    layouts::{CKKSPreparedRight, ScratchArenaTakeCKKS},
};
use poulpy_core::GLWEBytesOf;

/// One planned term of an ordered ct×plaintext-constant batch.
///
/// Plain data: a backend may lower it to launch parameters, but it holds no
/// operand view and outlives nothing. Entry `i` describes term `i`, in order.
#[doc(hidden)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct CKKSMulAddPtConstPlan {
    /// Coefficient of the constant container this term multiplies.
    pub coeff: usize,
    /// Full ct×pt convolution offset of the product.
    pub cnv_offset: usize,
    /// Metadata stamped on the product before the add.
    pub product_meta: CKKSMeta,
    /// Budget of the product before the add.
    pub product_log_budget: usize,
    /// Left shift applied to the accumulator to align it with the product.
    pub dst_shift: usize,
    /// Left shift applied to the product to align it with the accumulator.
    pub product_shift: usize,
    /// Accumulator metadata after this term's add.
    pub dst_meta: CKKSMeta,
    /// Accumulator budget after this term's add.
    pub dst_log_budget: usize,
}

/// Plans an ordered ct×plaintext-constant batch against a **virtual**
/// destination whose metadata evolves term by term, exactly as execution will.
///
/// Each term's product is carved at the accumulator's *current* torus width
/// `log_delta + log_budget`, which shrinks as budget is spent, so a term that is
/// valid at the initial width can still fail later. Planning the whole slice
/// here is what makes the batch validate before it mutates anything.
pub fn ckks_mul_add_pt_consts_plan<Dst, A, P>(dst: &Dst, terms: &[(&A, usize)], coeffs: &P) -> Result<Vec<CKKSMulAddPtConstPlan>>
where
    Dst: CKKSCtBounds,
    A: CKKSCtBounds,
    P: IntPolyInfos + CKKSCtBounds,
{
    let n_coeffs: usize = coeffs.n().into();
    let (mut meta, mut log_budget) = (dst.meta(), dst.log_budget());
    let mut plans: Vec<CKKSMulAddPtConstPlan> = Vec::with_capacity(terms.len());

    for &(a, coeff) in terms {
        ckks_ensure!(
            coeff < n_coeffs,
            "ckks_mul_add_pt_consts_into: coefficient index {coeff} >= {n_coeffs}"
        );
        // The product is carved like the accumulator, so its width is the
        // accumulator's width *at this point in the batch*.
        let (product_log_budget, product_log_delta, cnv_offset) = mul_pt_params_raw(
            meta.log_delta + log_budget,
            a.log_delta(),
            a.log_budget(),
            coeffs.log_delta(),
            coeffs.log_budget(),
            coeffs.encoded_k().as_usize(),
        )?;
        // A scalar-constant multiply preserves the operand's sparsity and is real.
        let product_meta = CKKSMeta {
            log_delta: product_log_delta,
            log_sparsity: a.log_sparsity(),
            slots: a.slots(),
        };
        plans.push(CKKSMulAddPtConstPlan {
            coeff,
            cnv_offset,
            product_meta,
            product_log_budget,
            dst_shift: log_budget.saturating_sub(product_log_budget),
            product_shift: product_log_budget.saturating_sub(log_budget),
            dst_meta: CKKSMeta {
                log_delta: meta.log_delta.min(product_log_delta),
                log_sparsity: meta.log_sparsity.min(product_meta.log_sparsity),
                slots: meta.slots.join(product_meta.slots),
            },
            dst_log_budget: log_budget.min(product_log_budget),
        });
        let last = plans.last().expect("just pushed");
        (meta, log_budget) = (last.dst_meta, last.dst_log_budget);
    }
    Ok(plans)
}

/// Ordered batch of `dst += a·coeffs[idx]`, the shared body behind
/// [`CKKSMulDefault::ckks_mul_add_pt_consts_into_default`] and the
/// [`CKKSMulImpl`](crate::oep::CKKSMulImpl) hook.
///
/// Exactly the ordered composition of `ckks_mul_add_pt_const_into` over
/// `terms`: each term keeps its own convolution offset, rounding, budget
/// alignment and carry normalization, and `dst`'s metadata evolves term by term.
/// `plans` is [`ckks_mul_add_pt_consts_plan`]'s output for the same slice, in
/// the same order; producing it is what validates the batch, so `dst` is never
/// touched on a planning failure. The per-term product buffer is scoped, so
/// scratch is that of one scalar operation regardless of the term count.
pub fn ckks_mul_add_pt_consts_into_ordered<BE, M, Dst, A, P>(
    module: &M,
    dst: &mut Dst,
    terms: &[(&A, usize)],
    plans: &[CKKSMulAddPtConstPlan],
    coeffs: &P,
    scratch: &mut ScratchArena<'_, BE>,
) -> Result<()>
where
    BE: Backend,
    M: CKKSMulOps<BE> + CKKSAddOps<BE>,
    Dst: GLWEToBackendMut<BE> + CKKSCtBounds + SetCKKSInfos,
    A: GLWEToBackendRef<BE> + CKKSCtBounds,
    P: GLWEToBackendRef<BE> + IntPolyInfos + CKKSCtBounds,
{
    ckks_ensure!(
        plans.len() == terms.len(),
        "ckks_mul_add_pt_consts_into: {} plans for {} terms",
        plans.len(),
        terms.len()
    );

    for (&(a, coeff), plan) in terms.iter().zip(plans) {
        debug_assert_eq!(coeff, plan.coeff, "plan entry describes a different term");
        scratch.scope(|scratch_local| {
            let (mut tmp, mut scratch_local) = scratch_local.take_ckks_ciphertext_like_scratch(dst);
            module.ckks_mul_pt_const_into(&mut tmp, a, coeffs, coeff, &mut scratch_local)?;
            debug_assert_eq!((tmp.meta(), tmp.log_budget()), (plan.product_meta, plan.product_log_budget));
            module.ckks_add_assign(dst, &tmp, &mut scratch_local)
        })?;
        debug_assert_eq!((dst.meta(), dst.log_budget()), (plan.dst_meta, plan.dst_log_budget));
    }
    Ok(())
}

pub trait CKKSMulDefault<BE: Backend> {
    fn ckks_mul_tmp_bytes_default<R, A, B, T>(&self, res: &R, a: &A, b: &B, tsk: &T) -> usize
    where
        Self: GLWEBytesOf<BE>,
        R: GLWEInfos,
        A: GLWEInfos,
        B: GLWEInfos,
        T: GGLWEInfos,
        Self: GLWETensoring<BE>,
    {
        // The op carves its tensor intermediate at the operands' effective
        // width `max(a.k, b.k)` (`ckks_mul_into`) or `max(dst.k, a.k)`
        // (`ckks_mul_assign`), matching the `cnv_offset` rule in
        // `mul_ct_params_raw` (which is already expressed on effective `k`).
        // Sizing must cover the widest of the three — a destination narrower
        // than its operands is a supported call.
        let tensor_layout = GLWELayout {
            n: res.n(),
            base2k: res.base2k(),
            k: TorusPrecision(res.k().max(a.k()).max(b.k()).as_u32()),
            rank: res.rank(),
        };

        let lvl_0 = self.glwe_tensor_bytes_of_from_infos(&tensor_layout);
        let lvl_1 = self
            .glwe_tensor_apply_tmp_bytes(&tensor_layout, a, b)
            .max(self.glwe_tensor_relinearize_tmp_bytes(res, &tensor_layout, tsk));

        lvl_0 + lvl_1
    }

    fn ckks_mul_into_default<Dst, A, B, T>(
        &self,
        dst: &mut Dst,
        a: &A,
        b: &B,
        tsk: &T,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        Self: GLWETensoring<BE> + GLWECopy<BE> + ModuleCoreAlloc<OwnedBuf = BE::OwnedBuf, ZnxWord = BE::ZnxWord>,
        Dst: GLWEToBackendMut<BE> + CKKSInfos + SetCKKSInfos + GLWEInfos,
        A: GLWEToBackendRef<BE> + CKKSInfos + GLWEInfos,
        B: GLWEToBackendRef<BE> + CKKSInfos + GLWEInfos,
        T: GLWETensorKeyPreparedToBackendRef<BE> + GGLWEPreparedToBackendRef<BE> + GGLWEInfos,
    {
        let (res_log_budget, res_log_delta, cnv_offset) = get_mul_ct_params(dst, a, b)?;

        tensor_mul_core(
            self,
            dst,
            tsk,
            a.k().max(b.k()),
            MulStamp {
                log_budget: res_log_budget,
                log_delta: res_log_delta,
                // The product of values sparse at `s` and `t` is sparse at `min(s, t)`.
                log_sparsity: Some(a.log_sparsity().min(b.log_sparsity())),
                slots: Some(a.slots().join(b.slots())),
            },
            StampOrder::BeforeApply,
            scratch,
            |tmp, _dst, s| self.glwe_tensor_apply(cnv_offset, tmp, a, b, s),
        )
    }

    fn ckks_mul_assign_default<Dst, A, T>(&self, dst: &mut Dst, a: &A, tsk: &T, scratch: &mut ScratchArena<'_, BE>) -> Result<()>
    where
        Self: GLWETensoring<BE> + GLWECopy<BE> + ModuleCoreAlloc<OwnedBuf = BE::OwnedBuf, ZnxWord = BE::ZnxWord>,
        Dst: GLWEToBackendMut<BE> + GLWEToBackendRef<BE> + CKKSInfos + SetCKKSInfos + GLWEInfos,
        A: GLWEToBackendRef<BE> + CKKSInfos + GLWEInfos,
        T: GLWETensorKeyPreparedToBackendRef<BE> + GGLWEPreparedToBackendRef<BE> + GGLWEInfos,
    {
        let (res_log_budget, res_log_delta, cnv_offset) = get_mul_ct_params(dst, dst, a)?;

        tensor_mul_core(
            self,
            dst,
            tsk,
            dst.k().max(a.k()),
            MulStamp {
                log_budget: res_log_budget,
                log_delta: res_log_delta,
                // The product of values sparse at `s` and `t` is sparse at `min(s, t)`.
                log_sparsity: Some(dst.log_sparsity().min(a.log_sparsity())),
                slots: Some(dst.slots().join(a.slots())),
            },
            StampOrder::AfterApply,
            scratch,
            |tmp, dst_ref, s| self.glwe_tensor_apply(cnv_offset, tmp, dst_ref, a, s),
        )
    }

    fn ckks_prepare_right_default<A>(&self, a: &A, scratch: &mut ScratchArena<'_, BE>) -> Result<CKKSPreparedRight<BE>>
    where
        Self: Convolution<BE> + CnvPVecAlloc<BE> + Sized,
        A: GLWEToBackendRef<BE> + CKKSInfos + GLWEInfos,
    {
        // Hoist `a` once into a backend-resident right operand. `glwe_prepare_right`
        // reads only the top `k` limbs, so the operand is sized to that
        // effective limb count.
        let cols = a.rank().as_usize() + 1;
        let k: usize = a.k().into();
        let size = k.div_ceil(a.base2k().as_usize());
        let mut prep = self.cnv_pvec_right_alloc(cols, size);
        glwe_prepare_right(self, &mut prep, a, k, scratch);
        Ok(CKKSPreparedRight {
            prep,
            size,
            log_delta: a.log_delta(),
            k,
            log_sparsity: a.log_sparsity(),
            slots: a.slots(),
            layout: GLWELayout {
                n: a.n(),
                base2k: a.base2k(),
                k: a.k(),
                rank: a.rank(),
            },
        })
    }

    fn ckks_mul_prepared_assign_default<Dst, T>(
        &self,
        dst: &mut Dst,
        prepared: &CKKSPreparedRight<BE>,
        tsk: &T,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        Self: GLWETensoring<BE> + GiantStepTensorBounds<BE>,
        Dst: GLWEToBackendMut<BE> + GLWEToBackendRef<BE> + CKKSInfos + SetCKKSInfos + GLWEInfos,
        T: GLWETensorKeyPreparedToBackendRef<BE> + GGLWEPreparedToBackendRef<BE> + GGLWEInfos,
    {
        // Prepared operands are long-lived cached objects: reject one built
        // under a different ring degree, radix, or rank before touching `dst`.
        if prepared.layout.n != dst.n() || prepared.layout.base2k != dst.base2k() || prepared.layout.rank != dst.rank() {
            return Err(crate::CKKSCompositionError::PreparedOperandLayoutMismatch {
                op: "mul_prepared",
                dst_n: dst.n().as_usize(),
                dst_base2k: dst.base2k().as_usize(),
                dst_rank: dst.rank().as_usize(),
                prep_n: prepared.layout.n.as_usize(),
                prep_base2k: prepared.layout.base2k.as_usize(),
                prep_rank: prepared.layout.rank.as_usize(),
            }
            .into());
        }
        let (res_log_budget, res_log_delta, cnv_offset) = mul_ct_params_raw(
            dst.k().as_usize(),
            dst.log_delta(),
            dst.k().into(),
            prepared.log_delta,
            prepared.k,
        )?;

        // Size the intermediate from the right operand's `k` rather than
        // its full `max_k`: the tensor product only consumes the top `k`
        // limbs (via the prepared operand).
        tensor_mul_core(
            self,
            dst,
            tsk,
            dst.k().max(TorusPrecision(prepared.k as u32)),
            MulStamp {
                log_budget: res_log_budget,
                log_delta: res_log_delta,
                // The product of values sparse at `s` and `t` is sparse at `min(s, t)`.
                log_sparsity: Some(dst.log_sparsity().min(prepared.log_sparsity)),
                slots: Some(dst.slots().join(prepared.slots)),
            },
            StampOrder::AfterApply,
            scratch,
            |tmp, dst_ref, s| self.glwe_tensor_apply_prepared_right(cnv_offset, tmp, dst_ref, &prepared.prep, prepared.size, s),
        )
    }

    fn ckks_square_tmp_bytes_default<R, A, T>(&self, res: &R, a: &A, tsk: &T) -> usize
    where
        Self: GLWEBytesOf<BE>,
        R: GLWEInfos,
        A: GLWEInfos,
        T: GGLWEInfos,
        Self: GLWETensoring<BE>,
    {
        // Mirror of `ckks_mul_tmp_bytes_default`: the op's tensor intermediate is
        // carved at the operand's effective width, which may exceed the
        // destination's.
        let tensor_layout = GLWELayout {
            n: res.n(),
            base2k: res.base2k(),
            k: TorusPrecision(res.k().max(a.k()).as_u32()),
            rank: res.rank(),
        };

        let lvl_0 = self.glwe_tensor_bytes_of_from_infos(&tensor_layout);
        let lvl_1 = self
            .glwe_tensor_square_apply_tmp_bytes(&tensor_layout, a)
            .max(self.glwe_tensor_relinearize_tmp_bytes(res, &tensor_layout, tsk));

        lvl_0 + lvl_1
    }

    fn ckks_square_into_default<Dst, A, T>(&self, dst: &mut Dst, a: &A, tsk: &T, scratch: &mut ScratchArena<'_, BE>) -> Result<()>
    where
        Self: GLWETensoring<BE> + GLWECopy<BE> + ModuleCoreAlloc<OwnedBuf = BE::OwnedBuf, ZnxWord = BE::ZnxWord>,
        Dst: GLWEToBackendMut<BE> + CKKSInfos + SetCKKSInfos + GLWEInfos,
        A: GLWEToBackendRef<BE> + CKKSInfos + GLWEInfos,
        T: GLWETensorKeyPreparedToBackendRef<BE> + GGLWEPreparedToBackendRef<BE> + GGLWEInfos,
    {
        let (res_log_budget, res_log_delta, cnv_offset) = get_mul_ct_params(dst, a, a)?;

        tensor_mul_core(
            self,
            dst,
            tsk,
            a.k(),
            MulStamp {
                log_budget: res_log_budget,
                log_delta: res_log_delta,
                log_sparsity: Some(a.log_sparsity()),
                slots: Some(a.slots()),
            },
            StampOrder::BeforeApply,
            scratch,
            |tmp, _dst, s| self.glwe_tensor_square_apply(cnv_offset, tmp, a, s),
        )
    }

    fn ckks_square_assign_default<Dst, T>(&self, dst: &mut Dst, tsk: &T, scratch: &mut ScratchArena<'_, BE>) -> Result<()>
    where
        Self: GLWETensoring<BE> + GLWECopy<BE> + ModuleCoreAlloc<OwnedBuf = BE::OwnedBuf, ZnxWord = BE::ZnxWord>,
        Dst: GLWEToBackendMut<BE> + GLWEToBackendRef<BE> + CKKSInfos + SetCKKSInfos + GLWEInfos,
        T: GLWETensorKeyPreparedToBackendRef<BE> + GGLWEPreparedToBackendRef<BE> + GGLWEInfos,
    {
        let (res_log_budget, res_log_delta, cnv_offset) = get_mul_ct_params(dst, dst, dst)?;

        tensor_mul_core(
            self,
            dst,
            tsk,
            dst.k(),
            MulStamp {
                log_budget: res_log_budget,
                log_delta: res_log_delta,
                // `min(dst, dst)` is the identity, so squaring leaves both
                // the sparsity and the slot kind as-is.
                log_sparsity: None,
                slots: None,
            },
            StampOrder::AfterApply,
            scratch,
            |tmp, dst_ref, s| self.glwe_tensor_square_apply(cnv_offset, tmp, dst_ref, s),
        )
    }

    fn ckks_mul_pt_vec_tmp_bytes_default<R, A>(&self, res: &R, a: &A, b_k: TorusPrecision) -> usize
    where
        R: GLWEInfos,
        A: GLWEInfos,
        Self: GLWEMulPlain<BE>,
    {
        let b_infos = GLWEPlaintextLayout {
            n: res.n(),
            base2k: res.base2k(),
            k: b_k,
        };
        self.glwe_mul_plain_tmp_bytes(res, a, &b_infos)
    }

    fn ckks_mul_pt_const_tmp_bytes_default<R, A>(&self, res: &R, a: &A, b_k: TorusPrecision) -> usize
    where
        Self: GLWEBytesOf<BE>,
        R: GLWEInfos,
        A: GLWEInfos,
        Self: GLWEMulConst<BE> + GLWERotate<BE>,
    {
        let b_infos = GLWEPlaintextLayout {
            n: res.n(),
            base2k: res.base2k(),
            k: b_k,
        };
        self.glwe_bytes_of_from_infos(res)
            + self
                .glwe_mul_const_tmp_bytes(res, a, &b_infos)
                .max(self.glwe_rotate_tmp_bytes())
    }

    fn ckks_mul_pt_vec_into_default<Dst, A, P>(
        &self,
        dst: &mut Dst,
        a: &A,
        pt: &P,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        P: GLWEToBackendRef<BE> + LWEInfos + IntPolyInfos + GLWEInfos + CKKSInfos,
        Self: GLWECopy<BE>
            + GLWEMulPlain<BE>
            + ModuleCoreAlloc<OwnedBuf = BE::OwnedBuf, ZnxWord = BE::ZnxWord>
            + VecZnxCopyBackend<BE>,
        Dst: GLWEToBackendMut<BE> + CKKSInfos + SetCKKSInfos + GLWEInfos,
        A: GLWEToBackendRef<BE> + CKKSInfos + GLWEInfos,
    {
        let (res_log_budget, res_log_delta, cnv_offset) = get_mul_pt_params(dst, a, pt)?;
        // Set the result metadata first: `dst.size()` is meta-derived and a fresh
        // `dst` carries zero meta, so `glwe_mul_plain` (which writes `res.size()`
        // limbs) would otherwise produce an empty output.
        dst.set_log_budget(res_log_budget);
        dst.set_log_delta(res_log_delta);
        // The product of values sparse at `s` and `t` is sparse at `min(s, t)`.
        dst.set_log_sparsity(a.log_sparsity().min(pt.log_sparsity()));
        dst.set_slots(a.slots().join(pt.slots()));
        self.glwe_mul_plain(cnv_offset, dst, a, pt, scratch);
        Ok(())
    }

    fn ckks_mul_pt_vec_assign_default<Dst, P>(&self, dst: &mut Dst, pt: &P, scratch: &mut ScratchArena<'_, BE>) -> Result<()>
    where
        P: GLWEToBackendRef<BE> + LWEInfos + IntPolyInfos + GLWEInfos + CKKSInfos,
        Self: GLWECopy<BE>
            + GLWEMulPlain<BE>
            + ModuleCoreAlloc<OwnedBuf = BE::OwnedBuf, ZnxWord = BE::ZnxWord>
            + VecZnxCopyBackend<BE>,
        Dst: GLWEToBackendMut<BE> + CKKSInfos + SetCKKSInfos + GLWEInfos,
    {
        let (res_log_budget, res_log_delta, cnv_offset) = get_mul_pt_params(dst, dst, pt)?;
        self.glwe_mul_plain_assign(cnv_offset, dst, pt, scratch);
        dst.set_log_budget(res_log_budget);
        dst.set_log_delta(res_log_delta);
        // The product of values sparse at `s` and `t` is sparse at `min(s, t)`.
        dst.set_log_sparsity(dst.log_sparsity().min(pt.log_sparsity()));
        dst.set_slots(dst.slots().join(pt.slots()));
        Ok(())
    }

    fn ckks_mul_pt_const_into_default<Dst, A, P>(
        &self,
        dst: &mut Dst,
        a: &A,
        pt: &P,
        pt_coeff: usize,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        P: GLWEToBackendRef<BE> + LWEInfos + IntPolyInfos + GLWEInfos + CKKSInfos,
        Self: GLWEMulConst<BE>,
        Dst: GLWEToBackendMut<BE> + CKKSInfos + SetCKKSInfos + GLWEInfos,
        A: GLWEToBackendRef<BE> + CKKSInfos + GLWEInfos,
    {
        let (res_log_budget, res_log_delta, cnv_offset) = get_mul_pt_params(dst, a, pt)?;
        // Set the result metadata first: `dst.size()` is meta-derived and a fresh
        // `dst` carries zero meta, so `glwe_mul_const` (which writes `res.size()`
        // limbs) would otherwise produce an empty output.
        dst.set_log_budget(res_log_budget);
        dst.set_log_delta(res_log_delta);
        // A scalar-constant multiply preserves the operand's sparsity pattern.
        dst.set_log_sparsity(a.log_sparsity());
        // A scalar-constant multiply is always real.
        dst.set_slots(a.slots());
        self.glwe_mul_const(cnv_offset, dst, a, pt, pt_coeff, scratch);

        Ok(())
    }

    fn ckks_mul_pt_const_assign_default<Dst, P>(
        &self,
        dst: &mut Dst,
        cnst: &P,
        cnst_coeff: usize,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        P: GLWEToBackendRef<BE> + LWEInfos + IntPolyInfos + GLWEInfos + CKKSInfos,
        Self: GLWEMulConst<BE>,
        Dst: GLWEToBackendMut<BE> + GLWEToBackendRef<BE> + CKKSInfos + SetCKKSInfos + GLWEInfos,
    {
        let (res_log_budget, res_log_delta, cnv_offset) = get_mul_pt_params(dst, dst, cnst)?;

        self.glwe_mul_const_assign(cnv_offset, dst, cnst, cnst_coeff, scratch);

        dst.set_log_budget(res_log_budget);
        dst.set_log_delta(res_log_delta);
        // A real scalar multiplier leaves the slot kind unchanged.
        Ok(())
    }
    /// Ordered batch of `dst += a·coeffs[idx]`; see
    /// [`ckks_mul_add_pt_consts_into_ordered`]. Override this to fuse the terms
    /// while keeping their ordered numerical semantics.
    #[allow(clippy::too_many_arguments)]
    fn ckks_mul_add_pt_consts_into_default<Dst, A, P>(
        &self,
        dst: &mut Dst,
        terms: &[(&A, usize)],
        plans: &[CKKSMulAddPtConstPlan],
        coeffs: &P,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        Self: CKKSMulOps<BE> + CKKSAddOps<BE> + Sized,
        Dst: GLWEToBackendMut<BE> + CKKSCtBounds + SetCKKSInfos,
        A: GLWEToBackendRef<BE> + CKKSCtBounds,
        P: GLWEToBackendRef<BE> + IntPolyInfos + CKKSCtBounds,
    {
        ckks_mul_add_pt_consts_into_ordered(self, dst, terms, plans, coeffs, scratch)
    }
}

/// Result metadata stamped on `dst` by [`tensor_mul_core`]: the checked budget
/// and scale, plus the variant's sparsity rule (`None` leaves `dst`'s sparsity
/// unchanged — squaring in place, where `min(dst, dst)` is the identity).
struct MulStamp {
    log_budget: usize,
    log_delta: usize,
    log_sparsity: Option<usize>,
    slots: Option<SlotsKind>,
}

/// When [`tensor_mul_core`] stamps the result metadata relative to the
/// variant's apply step.
///
/// `_into` variants stamp **before**: `dst.size()` is meta-derived and a
/// freshly-allocated `dst` carries zero meta (hence `size() == 0`), which
/// would leave the relinearized output empty. `_assign` variants stamp
/// **after**: `dst` is also an operand read by `apply`, so its metadata must
/// stay untouched until the tensoring has consumed it.
enum StampOrder {
    BeforeApply,
    AfterApply,
}

/// Shared body of the five tensor-multiplication variants (`mul_into`,
/// `mul_assign`, `mul_prepared_assign`, `square_into`, `square_assign`):
/// stamp (per `order`), carve the tensor intermediate at `tensor_k`, run the
/// variant's `apply` into it, relinearize into `dst`, stamp the metadata.
///
/// `apply` receives the carved tensor, a shared reborrow of `dst` (used by the
/// `_assign` variants, ignored by `_into`), and the nested scratch arena.
#[allow(clippy::too_many_arguments)]
fn tensor_mul_core<BE, M, Dst, T>(
    module: &M,
    dst: &mut Dst,
    tsk: &T,
    tensor_k: TorusPrecision,
    stamp: MulStamp,
    order: StampOrder,
    scratch: &mut ScratchArena<'_, BE>,
    apply: impl for<'t> FnOnce(&mut GLWETensorViewMut<'t, BE>, &Dst, &mut ScratchArena<'t, BE>),
) -> Result<()>
where
    BE: Backend,
    M: GLWETensoring<BE> + ?Sized,
    Dst: GLWEToBackendMut<BE> + CKKSInfos + SetCKKSInfos + GLWEInfos,
    T: GLWETensorKeyPreparedToBackendRef<BE> + GGLWEPreparedToBackendRef<BE> + GGLWEInfos,
{
    let do_stamp = |dst: &mut Dst| {
        dst.set_log_budget(stamp.log_budget);
        dst.set_log_delta(stamp.log_delta);
        if let Some(log_sparsity) = stamp.log_sparsity {
            dst.set_log_sparsity(log_sparsity);
        }
        if let Some(slots) = stamp.slots {
            dst.set_slots(slots);
        }
    };
    if matches!(order, StampOrder::BeforeApply) {
        do_stamp(dst);
    }

    let tensor_layout = GLWELayout {
        n: dst.n(),
        base2k: dst.base2k(),
        k: tensor_k,
        rank: dst.rank(),
    };
    let scratch_local = scratch.borrow();
    let (mut tmp, mut scratch_local) = scratch_local.take_glwe_tensor_scratch(&tensor_layout);
    apply(&mut tmp, &*dst, &mut scratch_local);
    module.glwe_tensor_relinearize(dst, &tmp, tsk, &mut scratch_local);

    if matches!(order, StampOrder::AfterApply) {
        do_stamp(dst);
    }
    Ok(())
}

fn get_mul_ct_params<R, A, B>(res: &R, a: &A, b: &B) -> Result<(usize, usize, usize)>
where
    R: CKKSInfos,
    A: CKKSInfos,
    B: CKKSInfos,
{
    mul_ct_params_raw(res.k().as_usize(), a.log_delta(), a.k().into(), b.log_delta(), b.k().into())
}

/// Shared `(log_budget, log_delta, cnv_offset)` rule for ct × ct multiplication,
/// expressed on raw values so the BSGS driver computes bit-identical parameters.
#[allow(clippy::too_many_arguments)]
pub(crate) fn mul_ct_params_raw(
    res_max_k: usize,
    a_log_delta: usize,
    a_k: usize,
    b_log_delta: usize,
    b_k: usize,
) -> Result<(usize, usize, usize)> {
    let a_log_budget = a_k.saturating_sub(a_log_delta);
    let b_log_budget = b_k.saturating_sub(b_log_delta);

    let res_log_budget = checked_mul_ct_log_budget("mul", a_log_budget, b_log_budget, a_log_delta, b_log_delta)?;
    let res_log_delta = a_log_delta.min(b_log_delta);

    let res_offset = (res_log_budget + res_log_delta).saturating_sub(res_max_k);
    // Addition/subtraction align to the shared, lower effective precision
    // (`ckks_offset_binary` uses `min`). Multiplication is different: the
    // bivariate convolution must traverse every live input limb, so the
    // convolution offset starts after the wider operand span and then skips any
    // extra limbs that cannot fit in `res`. This matches the already-rescaled
    // multiplication rule documented by `CKKSMulOps` and the bivariate Torus
    // analysis cited in the README/ePrint 2023/771.
    let cnv_offset = a_k.max(b_k) + res_offset;

    Ok((
        checked_log_budget_sub("mul", res_log_budget, res_offset)?,
        res_log_delta,
        cnv_offset,
    ))
}

pub fn get_mul_pt_params<R, A, B>(res: &R, a: &A, b: &B) -> Result<(usize, usize, usize)>
where
    R: CKKSInfos,
    A: CKKSInfos,
    B: CKKSInfos + IntPolyInfos,
{
    mul_pt_params_raw(
        res.k().as_usize(),
        a.log_delta(),
        a.log_budget(),
        b.log_delta(),
        b.log_budget(),
        b.encoded_k().as_usize(),
    )
}

/// Shared `(log_budget, log_delta, cnv_offset)` rule for ct × pt multiplication,
/// expressed on raw values so the BSGS driver computes bit-identical parameters.
pub fn mul_pt_params_raw(
    res_max_k: usize,
    a_log_delta: usize,
    a_log_budget: usize,
    b_log_delta: usize,
    b_log_budget: usize,
    b_max_k: usize,
) -> Result<(usize, usize, usize)> {
    let res_log_budget = checked_mul_pt_log_budget("mul", a_log_budget, b_log_budget, a_log_delta, b_log_delta)?;
    let res_log_delta = a_log_delta;
    let res_offset = (res_log_budget + res_log_delta).saturating_sub(res_max_k);
    let cnv_offset = b_max_k + res_offset;

    Ok((
        checked_log_budget_sub("mul", res_log_budget, res_offset)?,
        res_log_delta,
        cnv_offset,
    ))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::SlotsKind;
    use poulpy_core::layouts::{Base2K, Degree, Rank};

    const BASE2K: usize = 52;

    /// Layout-only operand: `ckks_mul_add_pt_consts_plan` reads metadata, never
    /// data, so the planner is testable without a backend.
    #[derive(Clone, Copy)]
    struct Spec {
        k: usize,
        log_delta: usize,
        encoded_k: usize,
        n: usize,
    }

    impl LWEInfos for Spec {
        fn base2k(&self) -> Base2K {
            Base2K(BASE2K as u32)
        }
        fn n(&self) -> Degree {
            Degree(self.n as u32)
        }
        fn max_size(&self) -> usize {
            self.k.div_ceil(BASE2K)
        }
        fn k(&self) -> TorusPrecision {
            TorusPrecision(self.k as u32)
        }
    }
    impl GLWEInfos for Spec {
        fn rank(&self) -> Rank {
            Rank(1)
        }
    }
    impl CKKSInfos for Spec {
        fn meta(&self) -> CKKSMeta {
            CKKSMeta {
                log_delta: self.log_delta,
                log_sparsity: 0,
                slots: SlotsKind::Real,
            }
        }
    }
    impl IntPolyInfos for Spec {
        fn encoded_k(&self) -> TorusPrecision {
            TorusPrecision(self.encoded_k as u32)
        }
    }

    fn ct(log_delta: usize, log_budget: usize) -> Spec {
        Spec {
            k: log_delta + log_budget,
            log_delta,
            encoded_k: 0,
            n: 256,
        }
    }

    fn coeffs(log_delta: usize, log_budget: usize, encoded_k: usize, n: usize) -> Spec {
        Spec {
            k: log_delta + log_budget,
            log_delta,
            encoded_k,
            n,
        }
    }

    /// The accumulator's torus width shrinks as budget is spent, and each term's
    /// product is carved at that *current* width. A term valid against the
    /// initial width can therefore fail later, so planning must walk the
    /// evolving destination.
    #[test]
    fn plan_walks_the_narrowing_destination() {
        let dst = ct(40, 100);
        let pt = coeffs(8, 60, 52, 4);
        // Term 1 leaves the accumulator at log_delta 40, log_budget 42: k = 82.
        let a1 = ct(40, 50);
        // `a2.log_delta()` fits under the initial width 140 but not under 82.
        let a2 = ct(90, 60);

        let alone = ckks_mul_add_pt_consts_plan(&dst, &[(&a2, 0)], &pt).expect("valid against the initial width");
        assert_eq!(alone.len(), 1);

        let first = ckks_mul_add_pt_consts_plan(&dst, &[(&a1, 0)], &pt).expect("first term plans");
        assert_eq!(first[0].dst_log_budget, 42);
        assert_eq!(first[0].dst_meta.log_delta, 40);

        assert!(
            ckks_mul_add_pt_consts_plan(&dst, &[(&a1, 0), (&a2, 1)], &pt).is_err(),
            "a term that only fails at the narrowed width must be rejected while planning"
        );
    }

    /// Every term is planned, and the accumulator state threaded between them is
    /// the one the ordered execution produces.
    #[test]
    fn plan_threads_the_accumulator_state() {
        let dst = ct(40, 100);
        let pt = coeffs(8, 60, 52, 4);
        let terms = [(&ct(40, 90), 0usize), (&ct(40, 70), 1), (&ct(40, 60), 2)];
        let refs: Vec<(&Spec, usize)> = terms.iter().map(|&(a, i)| (a, i)).collect();
        let plans = ckks_mul_add_pt_consts_plan(&dst, &refs, &pt).expect("plans");

        assert_eq!(plans.len(), 3);
        let mut log_budget = dst.log_budget();
        for (i, plan) in plans.iter().enumerate() {
            assert_eq!(plan.coeff, i);
            // `res_log_budget = a.log_budget - coeffs.log_delta`.
            assert_eq!(plan.product_log_budget, refs[i].0.log_budget() - pt.log_delta());
            assert_eq!(plan.dst_shift, log_budget.saturating_sub(plan.product_log_budget));
            assert_eq!(plan.product_shift, plan.product_log_budget.saturating_sub(log_budget));
            assert_eq!(plan.dst_log_budget, log_budget.min(plan.product_log_budget));
            log_budget = plan.dst_log_budget;
        }
        assert!(
            plans[2].dst_log_budget < plans[0].dst_log_budget,
            "the accumulator must narrow across the batch"
        );
    }

    /// The planned convolution offsets of a degree-three baby step at
    /// `base2k = 52`: 120, 62 and 52 bits.
    ///
    /// `eval_baby_step` seeds the accumulator from the highest power, the
    /// lowest-budget operand, so every product has at least the accumulator's
    /// budget and the accumulator does not narrow: the width evolution is
    /// exercised by [`plan_walks_the_narrowing_destination`] instead, which is
    /// the shape an arbitrary caller of the public op can produce.
    #[test]
    fn plan_offsets_of_a_degree_three_baby_step() {
        let pt = coeffs(8, 60, BASE2K, 4);
        // Chosen so the three ct×pt products land on 120, 62 and 52 bits at
        // base2k = 52: the offsets of a degree-three baby step.
        let dst = ct(40, 30);
        let terms = [(&ct(40, 106), 1usize), (&ct(40, 48), 2), (&ct(40, 38), 3)];
        let refs: Vec<(&Spec, usize)> = terms.iter().map(|&(a, i)| (a, i)).collect();
        let plans = ckks_mul_add_pt_consts_plan(&dst, &refs, &pt).expect("plans");
        assert_eq!(plans.iter().map(|p| p.cnv_offset).collect::<Vec<_>>(), vec![120, 62, 52]);
        assert_eq!(
            plans[2].dst_log_budget,
            dst.log_budget(),
            "a highest-power-seeded accumulator must not narrow across the baby step"
        );
        // The last term shifts the accumulator into line with its product.
        assert_eq!(plans[2].dst_shift, 0);
        assert_eq!(plans[2].product_shift, 0);
    }
}
