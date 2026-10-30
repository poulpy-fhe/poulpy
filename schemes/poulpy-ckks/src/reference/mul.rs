use crate::CKKSResult as Result;
use poulpy_core::layouts::GetTensorKey;
use poulpy_core::layouts::IntPolyInfos;
use poulpy_core::{
    GLWECopy, GLWEMulConst, GLWEMulPlain, GLWENormalize, GLWETensoring, ScratchArenaTakeCore, glwe_prepare_left,
    glwe_prepare_left_tmp_bytes, glwe_prepare_right, glwe_prepare_right_tmp_bytes,
    layouts::{
        GGLWEInfos, GLWEInfos, GLWELayout, GLWEPlaintextLayout, GLWETensorViewMut, GLWEToBackendMut, GLWEToBackendRef, LWEInfos,
        ModuleCoreAlloc, TorusPrecision, prepared::GLWETensorKeyPreparedBackendRef,
    },
};
use poulpy_hal::{
    api::{CnvOps, VecZnxCopy},
    layouts::{Backend, ScratchArena},
};

use crate::{CKKSCompositionError, SlotsKind};
use crate::{
    CKKSInfos, GLWEPlaintextInfos, SetCKKSInfos, ckks_mul_pt_const_checks, ckks_mul_pt_vec_checks, get_mul_ct_params,
    layouts::{CKKSPreparedLeft, CKKSPreparedRight},
    mul_ct_params_raw,
};
use poulpy_core::GLWEBytesOf;

pub trait CKKSMulReference<BE: Backend> {
    fn ckks_mul_tmp_bytes_reference<R, A, B, T>(&self, res: &R, a: &A, b: &B, tsk: &T) -> usize
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
        // (`ckks_mul_assign`), which bounds the product's stamped width.
        // `res.k` is its pre-call metadata and does not size this intermediate.
        let tensor_layout = GLWELayout {
            n: res.n(),
            base2k: res.base2k(),
            k: a.k().max(b.k()),
            rank: res.rank(),
        };
        if tensor_layout.k().as_u32() == 0 {
            return 0;
        }

        let lvl_0 = self.glwe_tensor_bytes_of_from_infos(&tensor_layout);
        let lvl_1 = self
            .glwe_tensor_apply_tmp_bytes(&tensor_layout, a, b)
            .max(self.glwe_tensor_apply_prepared_right_tmp_bytes(&tensor_layout, a, a.size(), b.size()))
            .max(self.glwe_tensor_relinearize_tmp_bytes(res, &tensor_layout, tsk));
        // The one-pass path. The prepared variant has the same shape.
        let size = tensor_layout.k().as_usize().div_ceil(res.base2k().as_usize());
        let one_pass = self.glwe_mul_relinearize_tmp_bytes(res, size, size, tensor_layout.k(), tsk);

        (lvl_0 + lvl_1).max(one_pass)
    }

    fn ckks_mul_into_reference<Dst, A, B, T>(
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
        T: GetTensorKey<BE>,
    {
        let k = a.k().max(b.k());
        tsk.get_tensor_key(k)
            .map_err(|source| CKKSCompositionError::MissingRelinearizationKey {
                op: "ckks_mul_into",
                k: k.into(),
                source,
            })?;

        let (res_log_budget, res_log_delta, cnv_offset) = get_mul_ct_params("ckks_mul_into", dst, a, b)?;

        let one_pass = mul_one_pass(dst, &[a.base2k(), b.base2k()], a.k().max(b.k()), cnv_offset, tsk).then_some(
            |dst: &mut Dst, res_k, s: &mut ScratchArena<'_, BE>| self.glwe_mul_relinearize(cnv_offset, dst, res_k, a, b, tsk, s),
        );
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
            one_pass,
            |tmp, _dst, s| self.glwe_tensor_apply(cnv_offset, tmp, a, b, s),
        )
    }

    fn ckks_mul_assign_reference<Dst, A, T>(
        &self,
        dst: &mut Dst,
        a: &A,
        tsk: &T,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        Self: GLWETensoring<BE> + GLWECopy<BE> + ModuleCoreAlloc<OwnedBuf = BE::OwnedBuf, ZnxWord = BE::ZnxWord>,
        Dst: GLWEToBackendMut<BE> + GLWEToBackendRef<BE> + CKKSInfos + SetCKKSInfos + GLWEInfos,
        A: GLWEToBackendRef<BE> + CKKSInfos + GLWEInfos,
        T: GetTensorKey<BE>,
    {
        let k = dst.k().max(a.k());
        tsk.get_tensor_key(k)
            .map_err(|source| CKKSCompositionError::MissingRelinearizationKey {
                op: "ckks_mul_assign",
                k: k.into(),
                source,
            })?;

        let (res_log_budget, res_log_delta, cnv_offset) = get_mul_ct_params("ckks_mul_assign", dst, dst, a)?;

        let one_pass = mul_one_pass(dst, &[a.base2k()], dst.k().max(a.k()), cnv_offset, tsk).then_some(
            |dst: &mut Dst, res_k, s: &mut ScratchArena<'_, BE>| {
                self.glwe_mul_relinearize_assign(cnv_offset, dst, res_k, a, tsk, s)
            },
        );
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
            one_pass,
            |tmp, dst_ref, s| self.glwe_tensor_apply(cnv_offset, tmp, dst_ref, a, s),
        )
    }

    /// What `glwe_prepare_left` and `glwe_prepare_right` take to prepare `a` on its own limbs.
    fn ckks_prepare_tmp_bytes_reference<A>(&self, a: &A) -> usize
    where
        Self: GLWEBytesOf<BE> + GLWENormalize<BE> + CnvOps<BE> + Sized,
        A: GLWEInfos,
    {
        glwe_prepare_left_tmp_bytes(self, a.size(), a).max(glwe_prepare_right_tmp_bytes(self, a.size(), a))
    }

    /// `glwe_prepare_left` of `a`, normalized first if its canonical flag is clear, on its `ceil(k / base2k)` limbs,
    /// into `res`, which then reports `a`'s metadata. `InvalidOperand` for an `a` of another layout than `res`'s,
    /// before any write.
    fn ckks_prepare_left_into_reference<A>(
        &self,
        res: &mut CKKSPreparedLeft<BE>,
        a: &A,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        Self: CnvOps<BE> + GLWENormalize<BE> + Sized,
        A: GLWEToBackendRef<BE> + CKKSInfos + GLWEInfos,
    {
        check_source_layout("ckks_prepare_left_into", res, a)?;
        glwe_prepare_left(self, &mut res.prep, a, a.k().as_usize(), scratch);
        res.meta = a.plaintext_meta();
        Ok(())
    }

    /// As [`Self::ckks_prepare_left_into_reference`], by `glwe_prepare_right` into a right operand.
    fn ckks_prepare_right_into_reference<A>(
        &self,
        res: &mut CKKSPreparedRight<BE>,
        a: &A,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        Self: CnvOps<BE> + GLWENormalize<BE> + Sized,
        A: GLWEToBackendRef<BE> + CKKSInfos + GLWEInfos,
    {
        check_source_layout("ckks_prepare_right_into", res, a)?;
        glwe_prepare_right(self, &mut res.prep, a, a.k().as_usize(), scratch);
        res.meta = a.plaintext_meta();
        Ok(())
    }

    fn ckks_mul_prepared_assign_reference<Dst, T>(
        &self,
        dst: &mut Dst,
        prepared: &CKKSPreparedRight<BE>,
        tsk: &T,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        Self: GLWETensoring<BE>,
        Dst: GLWEToBackendMut<BE> + GLWEToBackendRef<BE> + CKKSInfos + SetCKKSInfos + GLWEInfos,
        T: GetTensorKey<BE>,
    {
        let k = dst.k().max(prepared.k());
        tsk.get_tensor_key(k)
            .map_err(|source| CKKSCompositionError::MissingRelinearizationKey {
                op: "ckks_mul_prepared_assign",
                k: k.into(),
                source,
            })?;

        let (res_log_budget, res_log_delta, cnv_offset, tensor_k) = get_mul_prepared_params(&*dst, prepared)?;

        let one_pass = mul_one_pass(dst, &[prepared.base2k()], tensor_k, cnv_offset, tsk).then_some(
            |dst: &mut Dst, res_k, s: &mut ScratchArena<'_, BE>| {
                self.glwe_mul_prepared_relinearize_assign(
                    cnv_offset,
                    dst,
                    res_k,
                    &prepared.prep,
                    prepared.size,
                    prepared.k(),
                    tsk,
                    s,
                )
            },
        );
        // Size the intermediate from the right operand's `k` rather than
        // its full `max_k`: the tensor product only consumes the top `k`
        // limbs (via the prepared operand).
        tensor_mul_core(
            self,
            dst,
            tsk,
            tensor_k,
            MulStamp {
                log_budget: res_log_budget,
                log_delta: res_log_delta,
                // The product of values sparse at `s` and `t` is sparse at `min(s, t)`.
                log_sparsity: Some(dst.log_sparsity().min(prepared.log_sparsity())),
                slots: Some(dst.slots().join(prepared.slots())),
            },
            StampOrder::AfterApply,
            scratch,
            one_pass,
            |tmp, dst_ref, s| self.glwe_tensor_apply_prepared_right(cnv_offset, tmp, dst_ref, &prepared.prep, prepared.size, s),
        )
    }

    fn ckks_square_tmp_bytes_reference<R, A, T>(&self, res: &R, a: &A, tsk: &T) -> usize
    where
        Self: GLWEBytesOf<BE>,
        R: GLWEInfos,
        A: GLWEInfos,
        T: GGLWEInfos,
        Self: GLWETensoring<BE>,
    {
        // Mirror of `ckks_mul_tmp_bytes_reference`: the op's tensor intermediate is
        // carved at the operand's effective width, which may exceed the
        // destination's.
        let tensor_layout = GLWELayout {
            n: res.n(),
            base2k: res.base2k(),
            k: a.k(),
            rank: res.rank(),
        };
        if tensor_layout.k().as_u32() == 0 {
            return 0;
        }

        let lvl_0 = self.glwe_tensor_bytes_of_from_infos(&tensor_layout);
        let lvl_1 = self
            .glwe_tensor_square_apply_tmp_bytes(&tensor_layout, a)
            .max(self.glwe_tensor_relinearize_tmp_bytes(res, &tensor_layout, tsk));
        // The one-pass path.
        let a_size = a.k().as_usize().div_ceil(a.base2k().as_usize());
        let one_pass = self.glwe_mul_relinearize_tmp_bytes(res, a_size, a_size, a.k(), tsk);

        (lvl_0 + lvl_1).max(one_pass)
    }

    fn ckks_square_into_reference<Dst, A, T>(
        &self,
        dst: &mut Dst,
        a: &A,
        tsk: &T,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        Self: GLWETensoring<BE> + GLWECopy<BE> + ModuleCoreAlloc<OwnedBuf = BE::OwnedBuf, ZnxWord = BE::ZnxWord>,
        Dst: GLWEToBackendMut<BE> + CKKSInfos + SetCKKSInfos + GLWEInfos,
        A: GLWEToBackendRef<BE> + CKKSInfos + GLWEInfos,
        T: GetTensorKey<BE>,
    {
        let k = a.k();
        tsk.get_tensor_key(k)
            .map_err(|source| CKKSCompositionError::MissingRelinearizationKey {
                op: "ckks_square_into",
                k: k.into(),
                source,
            })?;

        let (res_log_budget, res_log_delta, cnv_offset) = get_mul_ct_params("ckks_square_into", dst, a, a)?;

        let one_pass = mul_one_pass(dst, &[a.base2k()], a.k(), cnv_offset, tsk).then_some(
            |dst: &mut Dst, res_k, s: &mut ScratchArena<'_, BE>| self.glwe_square_relinearize(cnv_offset, dst, res_k, a, tsk, s),
        );
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
            one_pass,
            |tmp, _dst, s| self.glwe_tensor_square_apply(cnv_offset, tmp, a, s),
        )
    }

    fn ckks_square_assign_reference<Dst, T>(&self, dst: &mut Dst, tsk: &T, scratch: &mut ScratchArena<'_, BE>) -> Result<()>
    where
        Self: GLWETensoring<BE> + GLWECopy<BE> + ModuleCoreAlloc<OwnedBuf = BE::OwnedBuf, ZnxWord = BE::ZnxWord>,
        Dst: GLWEToBackendMut<BE> + GLWEToBackendRef<BE> + CKKSInfos + SetCKKSInfos + GLWEInfos,
        T: GetTensorKey<BE>,
    {
        let k = dst.k();
        tsk.get_tensor_key(k)
            .map_err(|source| CKKSCompositionError::MissingRelinearizationKey {
                op: "ckks_square_assign",
                k: k.into(),
                source,
            })?;

        let (res_log_budget, res_log_delta, cnv_offset) = get_mul_ct_params("ckks_square_assign", dst, dst, dst)?;

        let one_pass =
            mul_one_pass(dst, &[], dst.k(), cnv_offset, tsk).then_some(|dst: &mut Dst, res_k, s: &mut ScratchArena<'_, BE>| {
                self.glwe_square_relinearize_assign(cnv_offset, dst, res_k, tsk, s)
            });
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
            one_pass,
            |tmp, dst_ref, s| self.glwe_tensor_square_apply(cnv_offset, tmp, dst_ref, s),
        )
    }

    fn ckks_mul_pt_vec_tmp_bytes_reference<R, A, P>(&self, res: &R, a: &A, b: &P) -> usize
    where
        Self: GLWEBytesOf<BE>,
        R: GLWEInfos,
        A: GLWEInfos,
        P: CKKSInfos,
        Self: GLWEMulPlain<BE>,
    {
        let b_infos = GLWEPlaintextLayout {
            n: res.n(),
            base2k: res.base2k(),
            k: b.k(),
        };
        self.glwe_bytes_of_from_infos(a) + self.glwe_mul_plain_tmp_bytes(res, a, &b_infos)
    }

    fn ckks_mul_pt_const_tmp_bytes_reference<R, A, P>(&self, res: &R, a: &A, b: &P) -> usize
    where
        Self: GLWEBytesOf<BE>,
        R: GLWEInfos,
        A: GLWEInfos,
        P: CKKSInfos,
        Self: GLWEMulConst<BE>,
    {
        let b_infos = GLWEPlaintextLayout {
            n: res.n(),
            base2k: res.base2k(),
            k: b.k(),
        };
        self.glwe_bytes_of_from_infos(res) + self.glwe_mul_const_tmp_bytes(res, a, &b_infos)
    }

    fn ckks_mul_pt_vec_into_reference<Dst, A, P>(
        &self,
        dst: &mut Dst,
        a: &A,
        pt: &P,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        P: GLWEToBackendRef<BE> + LWEInfos + IntPolyInfos + GLWEInfos + CKKSInfos,
        Self: GLWECopy<BE> + GLWEMulPlain<BE> + ModuleCoreAlloc<OwnedBuf = BE::OwnedBuf, ZnxWord = BE::ZnxWord> + VecZnxCopy<BE>,
        Dst: GLWEToBackendMut<BE> + CKKSInfos + SetCKKSInfos + GLWEInfos,
        A: GLWEToBackendRef<BE> + CKKSInfos + GLWEInfos,
    {
        let (res_log_budget, res_log_delta, cnv_offset) =
            ckks_mul_pt_vec_checks::<BE, _, _, _>("ckks_mul_pt_vec_into", dst, a, pt)?;
        // Set the result metadata first: `dst.size()` is meta-derived and a fresh
        // `dst` carries zero meta, so `glwe_mul_plain` (which writes `res.size()`
        // limbs) would otherwise produce an empty output.
        dst.set_log_budget(res_log_budget);
        dst.set_log_delta(res_log_delta);
        // The product of values sparse at `s` and `t` is sparse at `min(s, t)`;
        // a plaintext is dense at its own degree.
        dst.set_log_sparsity(a.log_sparsity().min(a.log_n() - pt.log_n()));
        dst.set_slots(a.slots().join(pt.slots()));
        self.glwe_mul_plain(cnv_offset, dst, a, pt, scratch);
        Ok(())
    }

    fn ckks_mul_pt_vec_assign_reference<Dst, P>(&self, dst: &mut Dst, pt: &P, scratch: &mut ScratchArena<'_, BE>) -> Result<()>
    where
        P: GLWEToBackendRef<BE> + LWEInfos + IntPolyInfos + GLWEInfos + CKKSInfos,
        Self: GLWECopy<BE> + GLWEMulPlain<BE> + ModuleCoreAlloc<OwnedBuf = BE::OwnedBuf, ZnxWord = BE::ZnxWord> + VecZnxCopy<BE>,
        Dst: GLWEToBackendMut<BE> + CKKSInfos + SetCKKSInfos + GLWEInfos,
    {
        let (res_log_budget, res_log_delta, cnv_offset) =
            ckks_mul_pt_vec_checks::<BE, _, _, _>("ckks_mul_pt_vec_assign", dst, dst, pt)?;
        let log_sparsity = dst.log_sparsity().min(dst.log_n() - pt.log_n());
        let slots = dst.slots().join(pt.slots());
        scratch.scope(|scratch_local| {
            let (mut input, mut op_scratch) = scratch_local.take_glwe_scratch(&*dst);
            self.glwe_copy(&mut input, &*dst, &mut op_scratch);
            dst.set_log_budget(res_log_budget);
            dst.set_log_delta(res_log_delta);
            dst.set_log_sparsity(log_sparsity);
            dst.set_slots(slots);
            self.glwe_mul_plain(cnv_offset, dst, &input, pt, &mut op_scratch);
        });
        Ok(())
    }

    fn ckks_mul_pt_const_into_reference<Dst, A, P>(
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
        let (res_log_budget, res_log_delta, cnv_offset) =
            ckks_mul_pt_const_checks("ckks_mul_pt_const_into", dst, a, pt, pt_coeff)?;
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

    fn ckks_mul_pt_const_assign_reference<Dst, P>(
        &self,
        dst: &mut Dst,
        cnst: &P,
        cnst_coeff: usize,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        P: GLWEToBackendRef<BE> + LWEInfos + IntPolyInfos + GLWEInfos + CKKSInfos,
        Self: GLWECopy<BE> + GLWEMulConst<BE> + ModuleCoreAlloc<OwnedBuf = BE::OwnedBuf, ZnxWord = BE::ZnxWord> + VecZnxCopy<BE>,
        Dst: GLWEToBackendMut<BE> + GLWEToBackendRef<BE> + CKKSInfos + SetCKKSInfos + GLWEInfos,
    {
        let (res_log_budget, res_log_delta, cnv_offset) =
            ckks_mul_pt_const_checks("ckks_mul_pt_const_assign", dst, dst, cnst, cnst_coeff)?;
        scratch.scope(|scratch_local| {
            let (mut input, mut op_scratch) = scratch_local.take_glwe_scratch(&*dst);
            self.glwe_copy(&mut input, &*dst, &mut op_scratch);
            dst.set_log_budget(res_log_budget);
            dst.set_log_delta(res_log_delta);
            self.glwe_mul_const(cnv_offset, dst, &input, cnst, cnst_coeff, &mut op_scratch);
        });
        Ok(())
    }
}

/// Result metadata stamped on `dst` by [`tensor_mul_core`]: the checked budget
/// and scale, plus the variant's sparsity rule (`None` leaves `dst`'s sparsity
/// unchanged — squaring in place, where `min(dst, dst)` is the identity).
pub(crate) struct MulStamp {
    pub(crate) log_budget: usize,
    pub(crate) log_delta: usize,
    pub(crate) log_sparsity: Option<usize>,
    pub(crate) slots: Option<SlotsKind>,
}

/// When [`tensor_mul_core`] stamps the result metadata relative to the
/// variant's apply step.
///
/// `_into` variants stamp **before**: `dst.size()` is meta-derived and a
/// freshly-allocated `dst` carries zero meta (hence `size() == 0`), which
/// would leave the relinearized output empty. `_assign` variants stamp
/// **after**: `dst` is also an operand read by `apply`, so its metadata must
/// stay untouched until the tensoring has consumed it.
pub(crate) enum StampOrder {
    BeforeApply,
    AfterApply,
}

/// Shared body of the five tensor-multiplication variants (`mul_into`,
/// `mul_assign`, `mul_prepared_assign`, `square_into`, `square_assign`) and of
/// the dot products that relinearize once:
/// stamp (per `order`), carve the tensor intermediate at `tensor_k`, run the
/// variant's `apply` into it, stamp (per `order`), then relinearize into `dst`.
///
/// `apply` receives the carved tensor, a shared reborrow of `dst` (used by the
/// `_assign` variants, ignored by `_into`), and the nested scratch arena.
///
/// `one_pass`, when given, replaces the tensor intermediate: it writes the
/// relinearized product into `dst` at the stamped width, reading `dst` first when
/// `dst` is an operand, so `dst` is stamped after it whatever `order` says.
#[allow(clippy::too_many_arguments)]
pub(crate) fn tensor_mul_core<BE, M, Dst, T>(
    module: &M,
    dst: &mut Dst,
    tsk: &T,
    tensor_k: TorusPrecision,
    stamp: MulStamp,
    order: StampOrder,
    scratch: &mut ScratchArena<'_, BE>,
    one_pass: Option<impl FnOnce(&mut Dst, TorusPrecision, &mut ScratchArena<'_, BE>)>,
    apply: impl for<'t> FnOnce(&mut GLWETensorViewMut<'t, BE>, &Dst, &mut ScratchArena<'t, BE>),
) -> Result<()>
where
    BE: Backend,
    M: GLWETensoring<BE> + ?Sized,
    Dst: GLWEToBackendMut<BE> + CKKSInfos + SetCKKSInfos + GLWEInfos,
    T: GetTensorKey<BE>,
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
    if let Some(one_pass) = one_pass {
        one_pass(dst, TorusPrecision((stamp.log_budget + stamp.log_delta) as u32), scratch);
        do_stamp(dst);
        // Stamping lowers `k`, which clears the flag of a product canonical at its new width.
        dst.set_canonical(true);
        return Ok(());
    }
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
    if matches!(order, StampOrder::AfterApply) {
        do_stamp(dst);
    }
    module.glwe_tensor_relinearize(dst, &tmp, tsk, &mut scratch_local);
    Ok(())
}

/// The tensor key of a product of width `k`, `MissingRelinearizationKey` for `op` where `tsk` holds none.
pub(crate) fn tensor_key_at<'k, BE, T>(
    op: &'static str,
    tsk: &'k T,
    k: TorusPrecision,
) -> Result<GLWETensorKeyPreparedBackendRef<'k, BE>>
where
    BE: Backend,
    T: GetTensorKey<BE>,
{
    tsk.get_tensor_key(k)
        .map_err(|source| CKKSCompositionError::MissingRelinearizationKey { op, k: k.into(), source }.into())
}

/// Whether a product of width `tensor_k` takes the one-pass product and relinearization.
///
/// The one-pass path needs rank 1, one `base2k` across `dst`, the operands other than `dst` and the
/// tensor key, and `cnv_offset >= base2k`. Other products go through the tensor intermediate.
pub(crate) fn mul_one_pass<BE, Dst, T>(
    dst: &Dst,
    operands_base2k: &[poulpy_core::layouts::Base2K],
    tensor_k: TorusPrecision,
    cnv_offset: usize,
    tsk: &T,
) -> bool
where
    BE: Backend,
    Dst: GLWEInfos,
    T: GetTensorKey<BE>,
{
    dst.rank().as_usize() == 1
        && cnv_offset >= dst.base2k().as_usize()
        && operands_base2k.iter().all(|&base2k| base2k == dst.base2k())
        && tsk.get_tensor_key(tensor_k).is_ok_and(|key| key.base2k() == dst.base2k())
}

/// Prepared operands are long-lived cached objects: reject one built under a
/// different ring degree, radix or rank before touching `dst`.
pub(crate) fn check_prepared_layout<D: GLWEInfos, P: GLWEInfos>(op: &'static str, dst: &D, prepared: &P) -> Result<()> {
    if (prepared.n(), prepared.base2k(), prepared.rank()) != (dst.n(), dst.base2k(), dst.rank()) {
        return Err(crate::CKKSCompositionError::PreparedOperandLayoutMismatch {
            op,
            dst_n: dst.n().as_usize(),
            dst_base2k: dst.base2k().as_usize(),
            dst_rank: dst.rank().as_usize(),
            prep_n: prepared.n().as_usize(),
            prep_base2k: prepared.base2k().as_usize(),
            prep_rank: prepared.rank().as_usize(),
        }
        .into());
    }
    Ok(())
}

/// Rejects a source of another layout than the prepared operand `res` was allocated for, before any write.
fn check_source_layout(op: &'static str, res: &impl GLWEInfos, a: &impl GLWEInfos) -> Result<()> {
    if res.glwe_layout() != a.glwe_layout() {
        return Err(crate::CKKSCompositionError::InvalidOperand {
            op,
            reason: "the source's layout differs from the prepared operand's".into(),
        }
        .into());
    }
    Ok(())
}

/// Full scalar-equivalent validation and parameters for a prepared multiply.
///
/// Batch delegates call this for every lane before resolving any bound or
/// writing any destination, so a late insufficient-budget error cannot leave
/// an earlier lane modified.
pub(crate) fn get_mul_prepared_params<BE: Backend, D: GLWEInfos + CKKSInfos>(
    dst: &D,
    prepared: &CKKSPreparedRight<BE>,
) -> Result<(usize, usize, usize, TorusPrecision)> {
    const OP: &str = "ckks_mul_prepared_assign";
    check_prepared_layout(OP, dst, prepared)?;
    let tensor_k = dst.k().max(prepared.k());
    let (log_budget, log_delta, cnv_offset) = mul_ct_params_raw(
        OP,
        dst.k().as_usize(),
        dst.log_delta(),
        dst.k().into(),
        prepared.log_delta(),
        prepared.k().into(),
    )?;
    Ok((log_budget, log_delta, cnv_offset, tensor_k))
}
impl<BE: Backend> CKKSMulReference<BE> for poulpy_hal::layouts::Module<BE> {}
