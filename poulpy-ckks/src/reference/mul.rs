use crate::CKKSResult as Result;
use poulpy_core::layouts::GetTensorKey;
use poulpy_core::layouts::IntPolyInfos;
use poulpy_core::{
    GLWECopy, GLWEMulConst, GLWEMulPlain, GLWENormalize, GLWETensoring, ScratchArenaTakeCore, glwe_prepare_left,
    glwe_prepare_left_tmp_bytes, glwe_prepare_right, glwe_prepare_right_tmp_bytes,
    layouts::{
        GGLWEInfos, GLWEInfos, GLWELayout, GLWEPlaintextLayout, GLWETensorViewMut, GLWEToBackendMut, GLWEToBackendRef, LWEInfos,
        ModuleCoreAlloc, TorusPrecision,
    },
};
use poulpy_hal::{
    api::{CnvPVecAlloc, CnvPVecBytesOf, Convolution, VecZnxCopy},
    layouts::{Backend, PrepareHint, ScratchArena},
};

use crate::{
    CKKSCompositionError, CKKSCtBounds, CKKSLayout, SlotLayout, SlotsKind,
    api::CKKSRoundOps,
    layouts::{CKKSCiphertextViewMut, ScratchArenaTakeCKKS},
    shaping::{MAX_PASS_BITS, ShapingProfile},
};
use crate::{
    CKKSInfos, GLWEPlaintextInfos, SetCKKSInfos, checked_log_budget_sub, checked_mul_ct_log_budget, checked_mul_pt_log_budget,
    ensure_plaintext_degree_embeds,
    layouts::{CKKSPreparedLeft, CKKSPreparedRight},
};
use poulpy_core::GLWEBytesOf;

pub trait CKKSMulReference<BE: Backend> {
    fn ckks_mul_tmp_bytes_reference<R, A, B, T>(&self, res: &R, a: &A, b: &B, tsk: &T) -> usize
    where
        Self: GLWEBytesOf<BE> + CKKSRoundOps<BE>,
        R: GLWEInfos,
        A: CKKSCtBounds,
        B: CKKSCtBounds,
        T: GGLWEInfos,
        Self: GLWETensoring<BE>,
    {
        // The op carves its tensor intermediate at the operands' effective
        // width `max(a.k, b.k)` (`ckks_mul_into`) or `max(dst.k, a.k)`
        // (`ckks_mul_assign`), which bounds the product's stamped width.
        // `res.k` is its pre-call metadata and does not size this intermediate.
        if a.k().max(b.k()).as_u32() == 0 {
            return 0;
        }
        let plain = |res: &GLWELayout, tensor_k: TorusPrecision| {
            let tensor_layout = GLWELayout { k: tensor_k, ..*res };
            let lvl_0 = self.glwe_tensor_bytes_of_from_infos(&tensor_layout);
            let lvl_1 = self
                .glwe_tensor_apply_tmp_bytes(&tensor_layout, a, b)
                .max(self.glwe_tensor_apply_prepared_right_tmp_bytes(&tensor_layout, a, a.size(), b.size()))
                .max(self.glwe_tensor_relinearize_tmp_bytes(res, &tensor_layout, tsk));
            // The one-pass path. The prepared variant has the same shape.
            let size = tensor_k.as_usize().div_ceil(res.base2k().as_usize());
            let one_pass = self.glwe_mul_relinearize_tmp_bytes(res, size, size, tensor_k, tsk);
            (lvl_0 + lvl_1).max(one_pass)
        };
        with_shaped_bytes(self, res, a.k().max(b.k()), is_replicated(a) || is_replicated(b), plain)
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
        Self:
            GLWETensoring<BE> + GLWECopy<BE> + ModuleCoreAlloc<OwnedBuf = BE::OwnedBuf, ZnxWord = BE::ZnxWord> + CKKSRoundOps<BE>,
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

        let slot_layout = product_layout("ckks_mul_into", &[a.slot_layout(), b.slot_layout()])?;
        let (res_log_budget, res_log_delta, cnv_offset) = get_mul_ct_params(dst, a, b)?;
        let stamp = MulStamp {
            log_budget: res_log_budget,
            log_delta: res_log_delta,
            // The product of values sparse at `s` and `t` is sparse at `min(s, t)`.
            log_sparsity: Some(a.log_sparsity().min(b.log_sparsity())),
            slots: Some(a.slots().join(b.slots())),
            slot_layout: Some(slot_layout),
        };
        match product_depth(slot_layout, res_log_delta) {
            None => mul_into_at(self, dst, a, b, tsk, cnv_offset, stamp, scratch),
            Some(lambda) => {
                let (mut wide, mut scratch) = take_wide(scratch, &*dst, lambda);
                mul_into_at(self, &mut wide, a, b, tsk, cnv_offset, stamp.widen(lambda), &mut scratch)?;
                round_into(self, dst, &wide, &stamp, &mut scratch)
            }
        }
    }

    fn ckks_mul_assign_reference<Dst, A, T>(
        &self,
        dst: &mut Dst,
        a: &A,
        tsk: &T,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        Self:
            GLWETensoring<BE> + GLWECopy<BE> + ModuleCoreAlloc<OwnedBuf = BE::OwnedBuf, ZnxWord = BE::ZnxWord> + CKKSRoundOps<BE>,
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

        let slot_layout = product_layout("ckks_mul_assign", &[dst.slot_layout(), a.slot_layout()])?;
        let (res_log_budget, res_log_delta, cnv_offset) = get_mul_ct_params(dst, dst, a)?;
        if let Some(lambda) = product_depth(slot_layout, res_log_delta) {
            let stamp = MulStamp {
                log_budget: res_log_budget,
                log_delta: res_log_delta,
                log_sparsity: Some(dst.log_sparsity().min(a.log_sparsity())),
                slots: Some(dst.slots().join(a.slots())),
                slot_layout: Some(slot_layout),
            };
            let (mut wide, mut scratch) = take_wide(scratch, &*dst, lambda);
            mul_into_at(self, &mut wide, &*dst, a, tsk, cnv_offset, stamp.widen(lambda), &mut scratch)?;
            return round_into(self, dst, &wide, &stamp, &mut scratch);
        }

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
                slot_layout: Some(slot_layout),
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
        Self: GLWEBytesOf<BE> + GLWENormalize<BE> + Convolution<BE> + Sized,
        A: GLWEInfos,
    {
        glwe_prepare_left_tmp_bytes(self, a.size(), a).max(glwe_prepare_right_tmp_bytes(self, a.size(), a))
    }

    /// The bytes of a left operand of `infos` under `hint`: its convolution operand on `ceil(k / base2k)` limbs and
    /// `rank + 1` columns.
    fn ckks_prepared_left_bytes_of_from_infos_reference<A>(&self, infos: &A, hint: PrepareHint) -> usize
    where
        Self: CnvPVecBytesOf,
        A: GLWEInfos,
    {
        self.bytes_of_cnv_pvec_left(infos.n().as_usize(), infos.rank().as_usize() + 1, infos.size(), hint)
    }

    /// As [`Self::ckks_prepared_left_bytes_of_from_infos_reference`], for a right operand.
    fn ckks_prepared_right_bytes_of_from_infos_reference<A>(&self, infos: &A, hint: PrepareHint) -> usize
    where
        Self: CnvPVecBytesOf,
        A: GLWEInfos,
    {
        self.bytes_of_cnv_pvec_right(infos.n().as_usize(), infos.rank().as_usize() + 1, infos.size(), hint)
    }

    /// A left operand of `infos` under `hint`, of unspecified contents, reporting `infos` and no metadata.
    fn ckks_prepared_left_alloc_from_infos_reference<A>(&self, infos: &A, hint: PrepareHint) -> CKKSPreparedLeft<BE>
    where
        Self: CnvPVecAlloc<BE>,
        A: GLWEInfos,
    {
        let size = infos.size();
        CKKSPreparedLeft {
            prep: self.cnv_pvec_left_alloc(infos.n().as_usize(), infos.rank().as_usize() + 1, size, hint),
            size,
            layout: infos.glwe_layout(),
            meta: None,
        }
    }

    /// As [`Self::ckks_prepared_left_alloc_from_infos_reference`], for a right operand.
    fn ckks_prepared_right_alloc_from_infos_reference<A>(&self, infos: &A, hint: PrepareHint) -> CKKSPreparedRight<BE>
    where
        Self: CnvPVecAlloc<BE>,
        A: GLWEInfos,
    {
        let size = infos.size();
        CKKSPreparedRight {
            prep: self.cnv_pvec_right_alloc(infos.n().as_usize(), infos.rank().as_usize() + 1, size, hint),
            size,
            layout: infos.glwe_layout(),
            meta: None,
        }
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
        Self: Convolution<BE> + GLWENormalize<BE> + Sized,
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
        Self: Convolution<BE> + GLWENormalize<BE> + Sized,
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
        Self: GLWETensoring<BE> + CKKSRoundOps<BE>,
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

        let slot_layout = product_layout("ckks_mul_prepared_assign", &[dst.slot_layout(), prepared.slot_layout()])?;
        let (res_log_budget, res_log_delta, cnv_offset, tensor_k) = get_mul_prepared_params(&*dst, prepared)?;
        if let Some(lambda) = product_depth(slot_layout, res_log_delta) {
            let stamp = MulStamp {
                log_budget: res_log_budget,
                log_delta: res_log_delta,
                log_sparsity: Some(dst.log_sparsity().min(prepared.log_sparsity())),
                slots: Some(dst.slots().join(prepared.slots())),
                slot_layout: Some(slot_layout),
            };
            let (mut wide, mut scratch) = take_wide(scratch, &*dst, lambda);
            mul_prepared_into_at(
                self,
                &mut wide,
                &*dst,
                prepared,
                tsk,
                cnv_offset,
                tensor_k,
                stamp.widen(lambda),
                &mut scratch,
            )?;
            return round_into(self, dst, &wide, &stamp, &mut scratch);
        }

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
                slot_layout: Some(slot_layout),
            },
            StampOrder::AfterApply,
            scratch,
            one_pass,
            |tmp, dst_ref, s| self.glwe_tensor_apply_prepared_right(cnv_offset, tmp, dst_ref, &prepared.prep, prepared.size, s),
        )
    }

    fn ckks_square_tmp_bytes_reference<R, A, T>(&self, res: &R, a: &A, tsk: &T) -> usize
    where
        Self: GLWEBytesOf<BE> + CKKSRoundOps<BE>,
        R: GLWEInfos,
        A: CKKSCtBounds,
        T: GGLWEInfos,
        Self: GLWETensoring<BE>,
    {
        // Mirror of `ckks_mul_tmp_bytes_reference`: the op's tensor intermediate is
        // carved at the operand's effective width, which may exceed the
        // destination's.
        if a.k().as_u32() == 0 {
            return 0;
        }
        let plain = |res: &GLWELayout, tensor_k: TorusPrecision| {
            let tensor_layout = GLWELayout { k: tensor_k, ..*res };
            let lvl_0 = self.glwe_tensor_bytes_of_from_infos(&tensor_layout);
            let lvl_1 = self
                .glwe_tensor_square_apply_tmp_bytes(&tensor_layout, a)
                .max(self.glwe_tensor_relinearize_tmp_bytes(res, &tensor_layout, tsk));
            // The one-pass path.
            let a_size = a.k().as_usize().div_ceil(a.base2k().as_usize());
            let one_pass = self.glwe_mul_relinearize_tmp_bytes(res, a_size, a_size, tensor_k, tsk);
            (lvl_0 + lvl_1).max(one_pass)
        };
        with_shaped_bytes(self, res, a.k(), is_replicated(a), plain)
    }

    fn ckks_square_into_reference<Dst, A, T>(
        &self,
        dst: &mut Dst,
        a: &A,
        tsk: &T,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        Self:
            GLWETensoring<BE> + GLWECopy<BE> + ModuleCoreAlloc<OwnedBuf = BE::OwnedBuf, ZnxWord = BE::ZnxWord> + CKKSRoundOps<BE>,
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

        let (res_log_budget, res_log_delta, cnv_offset) = get_mul_ct_params(dst, a, a)?;
        let stamp = MulStamp {
            log_budget: res_log_budget,
            log_delta: res_log_delta,
            log_sparsity: Some(a.log_sparsity()),
            slots: Some(a.slots()),
            slot_layout: Some(a.slot_layout()),
        };
        match product_depth(a.slot_layout(), res_log_delta) {
            None => square_into_at(self, dst, a, tsk, cnv_offset, stamp, scratch),
            Some(lambda) => {
                let (mut wide, mut scratch) = take_wide(scratch, &*dst, lambda);
                square_into_at(self, &mut wide, a, tsk, cnv_offset, stamp.widen(lambda), &mut scratch)?;
                round_into(self, dst, &wide, &stamp, &mut scratch)
            }
        }
    }

    fn ckks_square_assign_reference<Dst, T>(&self, dst: &mut Dst, tsk: &T, scratch: &mut ScratchArena<'_, BE>) -> Result<()>
    where
        Self:
            GLWETensoring<BE> + GLWECopy<BE> + ModuleCoreAlloc<OwnedBuf = BE::OwnedBuf, ZnxWord = BE::ZnxWord> + CKKSRoundOps<BE>,
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

        let (res_log_budget, res_log_delta, cnv_offset) = get_mul_ct_params(dst, dst, dst)?;
        if let Some(lambda) = product_depth(dst.slot_layout(), res_log_delta) {
            let stamp = MulStamp {
                log_budget: res_log_budget,
                log_delta: res_log_delta,
                log_sparsity: Some(dst.log_sparsity()),
                slots: Some(dst.slots()),
                slot_layout: Some(dst.slot_layout()),
            };
            let (mut wide, mut scratch) = take_wide(scratch, &*dst, lambda);
            square_into_at(self, &mut wide, &*dst, tsk, cnv_offset, stamp.widen(lambda), &mut scratch)?;
            return round_into(self, dst, &wide, &stamp, &mut scratch);
        }

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
                slot_layout: None,
            },
            StampOrder::AfterApply,
            scratch,
            one_pass,
            |tmp, dst_ref, s| self.glwe_tensor_square_apply(cnv_offset, tmp, dst_ref, s),
        )
    }

    fn ckks_mul_pt_vec_tmp_bytes_reference<R, A>(&self, res: &R, a: &A, b_k: TorusPrecision) -> usize
    where
        Self: GLWEBytesOf<BE> + CKKSRoundOps<BE>,
        R: GLWEInfos,
        A: CKKSCtBounds,
        Self: GLWEMulPlain<BE>,
    {
        let plain = |res: &GLWELayout, _: TorusPrecision| {
            let b_infos = GLWEPlaintextLayout {
                n: res.n(),
                base2k: res.base2k(),
                k: b_k,
            };
            self.glwe_bytes_of_from_infos(a) + self.glwe_mul_plain_tmp_bytes(res, a, &b_infos)
        };
        with_shaped_bytes(self, res, a.k(), is_replicated(a), plain)
    }

    fn ckks_mul_pt_const_tmp_bytes_reference<R, A>(&self, res: &R, a: &A, b_k: TorusPrecision) -> usize
    where
        Self: GLWEBytesOf<BE> + CKKSRoundOps<BE>,
        R: GLWEInfos,
        A: CKKSCtBounds,
        Self: GLWEMulConst<BE>,
    {
        let plain = |res: &GLWELayout, _: TorusPrecision| {
            let b_infos = GLWEPlaintextLayout {
                n: res.n(),
                base2k: res.base2k(),
                k: b_k,
            };
            self.glwe_bytes_of_from_infos(res) + self.glwe_mul_const_tmp_bytes(res, a, &b_infos)
        };
        with_shaped_bytes(self, res, a.k(), is_replicated(a), plain)
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
        Self: GLWECopy<BE>
            + GLWEMulPlain<BE>
            + ModuleCoreAlloc<OwnedBuf = BE::OwnedBuf, ZnxWord = BE::ZnxWord>
            + VecZnxCopy<BE>
            + CKKSRoundOps<BE>,
        Dst: GLWEToBackendMut<BE> + CKKSInfos + SetCKKSInfos + GLWEInfos,
        A: GLWEToBackendRef<BE> + CKKSInfos + GLWEInfos,
    {
        let slot_layout = product_layout("ckks_mul_pt_vec", &[a.slot_layout(), pt.slot_layout()])?;
        let (res_log_budget, res_log_delta, cnv_offset) = get_mul_pt_params(dst, a, pt)?;
        ensure_plaintext_degree_embeds("ckks_mul_pt_vec", a.n().as_usize(), pt.n().as_usize(), BE::MIN_DEGREE)?;
        let stamp = MulStamp {
            log_budget: res_log_budget,
            log_delta: res_log_delta,
            // The product of values sparse at `s` and `t` is sparse at `min(s, t)`;
            // a plaintext is dense at its own degree.
            log_sparsity: Some(a.log_sparsity().min(a.log_n() - pt.log_n())),
            slots: Some(a.slots().join(pt.slots())),
            slot_layout: Some(slot_layout),
        };
        match product_depth(slot_layout, pt.log_delta()) {
            None => {
                mul_pt_vec_into_at(self, dst, a, pt, cnv_offset, &stamp, scratch);
                Ok(())
            }
            Some(lambda) => {
                let (mut wide, mut scratch) = take_wide(scratch, &*dst, lambda);
                mul_pt_vec_into_at(self, &mut wide, a, pt, cnv_offset, &stamp.widen(lambda), &mut scratch);
                round_into(self, dst, &wide, &stamp, &mut scratch)
            }
        }
    }

    fn ckks_mul_pt_vec_assign_reference<Dst, P>(&self, dst: &mut Dst, pt: &P, scratch: &mut ScratchArena<'_, BE>) -> Result<()>
    where
        P: GLWEToBackendRef<BE> + LWEInfos + IntPolyInfos + GLWEInfos + CKKSInfos,
        Self: GLWECopy<BE>
            + GLWEMulPlain<BE>
            + ModuleCoreAlloc<OwnedBuf = BE::OwnedBuf, ZnxWord = BE::ZnxWord>
            + VecZnxCopy<BE>
            + CKKSRoundOps<BE>,
        Dst: GLWEToBackendMut<BE> + CKKSInfos + SetCKKSInfos + GLWEInfos,
    {
        let (res_log_budget, res_log_delta, cnv_offset) = get_mul_pt_params(dst, dst, pt)?;
        ensure_plaintext_degree_embeds(
            "ckks_mul_pt_vec_assign",
            dst.n().as_usize(),
            pt.n().as_usize(),
            BE::MIN_DEGREE,
        )?;
        let slot_layout = product_layout("ckks_mul_pt_vec_assign", &[dst.slot_layout(), pt.slot_layout()])?;
        let log_sparsity = dst.log_sparsity().min(dst.log_n() - pt.log_n());
        let slots = dst.slots().join(pt.slots());
        if let Some(lambda) = product_depth(slot_layout, pt.log_delta()) {
            let stamp = MulStamp {
                log_budget: res_log_budget,
                log_delta: res_log_delta,
                log_sparsity: Some(log_sparsity),
                slots: Some(slots),
                slot_layout: Some(slot_layout),
            };
            let (mut wide, mut scratch) = take_wide(scratch, &*dst, lambda);
            mul_pt_vec_into_at(self, &mut wide, &*dst, pt, cnv_offset, &stamp.widen(lambda), &mut scratch);
            return round_into(self, dst, &wide, &stamp, &mut scratch);
        }
        scratch.scope(|scratch_local| {
            let (mut input, mut op_scratch) = scratch_local.take_glwe_scratch(&*dst);
            self.glwe_copy(&mut input, &*dst, &mut op_scratch);
            dst.set_log_budget(res_log_budget);
            dst.set_log_delta(res_log_delta);
            dst.set_log_sparsity(log_sparsity);
            dst.set_slots(slots);
            dst.set_slot_layout(slot_layout);
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
        Self: GLWEMulConst<BE> + CKKSRoundOps<BE>,
        Dst: GLWEToBackendMut<BE> + CKKSInfos + SetCKKSInfos + GLWEInfos,
        A: GLWEToBackendRef<BE> + CKKSInfos + GLWEInfos,
    {
        let (res_log_budget, res_log_delta, cnv_offset) = get_mul_pt_params(dst, a, pt)?;
        // A scalar-constant multiply preserves the operand's sparsity pattern,
        // slot kind and layout.
        let stamp = MulStamp {
            log_budget: res_log_budget,
            log_delta: res_log_delta,
            log_sparsity: Some(a.log_sparsity()),
            slots: Some(a.slots()),
            slot_layout: Some(a.slot_layout()),
        };
        match product_depth(a.slot_layout(), pt.log_delta()) {
            None => {
                mul_pt_const_into_at(self, dst, a, pt, pt_coeff, cnv_offset, &stamp, scratch);
                Ok(())
            }
            Some(lambda) => {
                let (mut wide, mut scratch) = take_wide(scratch, &*dst, lambda);
                mul_pt_const_into_at(
                    self,
                    &mut wide,
                    a,
                    pt,
                    pt_coeff,
                    cnv_offset,
                    &stamp.widen(lambda),
                    &mut scratch,
                );
                round_into(self, dst, &wide, &stamp, &mut scratch)
            }
        }
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
        Self: GLWECopy<BE>
            + GLWEMulConst<BE>
            + ModuleCoreAlloc<OwnedBuf = BE::OwnedBuf, ZnxWord = BE::ZnxWord>
            + VecZnxCopy<BE>
            + CKKSRoundOps<BE>,
        Dst: GLWEToBackendMut<BE> + GLWEToBackendRef<BE> + CKKSInfos + SetCKKSInfos + GLWEInfos,
    {
        let (res_log_budget, res_log_delta, cnv_offset) = get_mul_pt_params(dst, dst, cnst)?;
        if let Some(lambda) = product_depth(dst.slot_layout(), cnst.log_delta()) {
            let stamp = MulStamp {
                log_budget: res_log_budget,
                log_delta: res_log_delta,
                log_sparsity: Some(dst.log_sparsity()),
                slots: Some(dst.slots()),
                slot_layout: Some(dst.slot_layout()),
            };
            let (mut wide, mut scratch) = take_wide(scratch, &*dst, lambda);
            mul_pt_const_into_at(
                self,
                &mut wide,
                &*dst,
                cnst,
                cnst_coeff,
                cnv_offset,
                &stamp.widen(lambda),
                &mut scratch,
            );
            return round_into(self, dst, &wide, &stamp, &mut scratch);
        }
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
#[derive(Clone, Copy)]
pub(crate) struct MulStamp {
    pub(crate) log_budget: usize,
    pub(crate) log_delta: usize,
    pub(crate) log_sparsity: Option<usize>,
    pub(crate) slots: Option<SlotsKind>,
    pub(crate) slot_layout: Option<SlotLayout>,
}

impl MulStamp {
    /// The stamp of a result computed `lambda` bits under the scale.
    fn widen(&self, lambda: usize) -> Self {
        Self {
            log_delta: self.log_delta + lambda,
            ..*self
        }
    }

    fn set_on<D: SetCKKSInfos>(&self, dst: &mut D) {
        dst.set_log_budget(self.log_budget);
        dst.set_log_delta(self.log_delta);
        if let Some(log_sparsity) = self.log_sparsity {
            dst.set_log_sparsity(log_sparsity);
        }
        if let Some(slots) = self.slots {
            dst.set_slots(slots);
        }
        if let Some(slot_layout) = self.slot_layout {
            dst.set_slot_layout(slot_layout);
        }
    }
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
    let do_stamp = |dst: &mut Dst| stamp.set_on(dst);
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

    // A shaped product reads bits under the operands' width.
    let tensor_layout = GLWELayout {
        n: dst.n(),
        base2k: dst.base2k(),
        k: tensor_k.max(((stamp.log_budget + stamp.log_delta) as u32).into()),
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
    check_prepared_layout("mul_prepared", dst, prepared)?;
    let tensor_k = dst.k().max(prepared.k());
    let (log_budget, log_delta, cnv_offset) = mul_ct_params_raw(
        dst.k().as_usize(),
        dst.log_delta(),
        dst.k().into(),
        prepared.log_delta(),
        prepared.k().into(),
    )?;
    Ok((log_budget, log_delta, cnv_offset, tensor_k))
}
pub(crate) fn get_mul_ct_params<R, A, B>(res: &R, a: &A, b: &B) -> Result<(usize, usize, usize)>
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
    // An operand holds its message at `2^-log_budget` on the torus and the
    // convolution returns `a * b * 2^cnv_offset`, so lifting the product from
    // `2^-(a_log_budget + b_log_budget)` to the stamped budget
    // `min(log_budget) - max(log_delta) - res_offset` takes
    // `max(log_budget) + max(log_delta) + res_offset`. That is
    // `max(a_k, b_k) + res_offset` only when the operand with the larger budget
    // also has the larger or equal scale.
    let cnv_offset = a_log_budget.max(b_log_budget) + a_log_delta.max(b_log_delta) + res_offset;

    Ok((
        checked_log_budget_sub("mul", res_log_budget, res_offset)?,
        res_log_delta,
        cnv_offset,
    ))
}

pub(crate) fn get_mul_pt_params<R, A, B>(res: &R, a: &A, b: &B) -> Result<(usize, usize, usize)>
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
pub(crate) fn mul_pt_params_raw(
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

impl<BE: Backend> CKKSMulReference<BE> for poulpy_hal::layouts::Module<BE> {}

/// The common slot layout of a product's operands.
pub(crate) fn product_layout(op: &'static str, layouts: &[SlotLayout]) -> Result<SlotLayout> {
    let first = layouts[0];
    if layouts.iter().any(|&layout| layout != first) {
        return Err(CKKSCompositionError::InvalidOperand {
            op,
            reason: "the operands' slot layouts differ".into(),
        }
        .into());
    }
    Ok(first)
}

/// Shaping depth of a product dividing by `2^log_delta` in `slot_layout`, `None` when it rounds plainly.
pub(crate) fn product_depth(slot_layout: SlotLayout, log_delta: usize) -> Option<usize> {
    match slot_layout {
        SlotLayout::Replicated(layout) => {
            Some(ShapingProfile::new(&layout).shaping_depth(log_delta, layout.w())).filter(|&lambda| lambda > 0)
        }
        SlotLayout::Full => None,
    }
}

/// A ciphertext `lambda` bits wider than `dst`, for a product rounded into `dst`.
pub(crate) fn take_wide<'a, BE: Backend + 'a, D: GLWEInfos + CKKSInfos>(
    scratch: &'a mut ScratchArena<'_, BE>,
    dst: &D,
    lambda: usize,
) -> (CKKSCiphertextViewMut<'a, BE>, ScratchArena<'a, BE>) {
    let mut layout = dst.glwe_layout();
    layout.k = (layout.k.as_usize() + lambda).into();
    scratch.borrow().take_ckks_ciphertext_scratch(&layout, dst.meta())
}

/// Rounds a product computed under the scale into `dst` at the width `stamp` labels.
pub(crate) fn round_into<BE, M, Dst>(
    module: &M,
    dst: &mut Dst,
    wide: &CKKSCiphertextViewMut<'_, BE>,
    stamp: &MulStamp,
    scratch: &mut ScratchArena<'_, BE>,
) -> Result<()>
where
    BE: Backend,
    M: CKKSRoundOps<BE> + ?Sized,
    Dst: GLWEToBackendMut<BE> + CKKSInfos + SetCKKSInfos + GLWEInfos,
{
    dst.set_k((stamp.log_budget + stamp.log_delta).into());
    module.ckks_round_into(dst, wide, scratch)
}

#[allow(clippy::too_many_arguments)]
fn mul_into_at<BE, M, Out, A, B, T>(
    module: &M,
    out: &mut Out,
    a: &A,
    b: &B,
    tsk: &T,
    cnv_offset: usize,
    stamp: MulStamp,
    scratch: &mut ScratchArena<'_, BE>,
) -> Result<()>
where
    BE: Backend,
    M: GLWETensoring<BE> + ?Sized,
    Out: GLWEToBackendMut<BE> + CKKSInfos + SetCKKSInfos + GLWEInfos,
    A: GLWEToBackendRef<BE> + CKKSInfos + GLWEInfos,
    B: GLWEToBackendRef<BE> + CKKSInfos + GLWEInfos,
    T: GetTensorKey<BE>,
{
    let tensor_k = a.k().max(b.k());
    let one_pass = mul_one_pass(out, &[a.base2k(), b.base2k()], tensor_k, cnv_offset, tsk).then_some(
        |out: &mut Out, res_k, s: &mut ScratchArena<'_, BE>| module.glwe_mul_relinearize(cnv_offset, out, res_k, a, b, tsk, s),
    );
    tensor_mul_core(
        module,
        out,
        tsk,
        tensor_k,
        stamp,
        StampOrder::BeforeApply,
        scratch,
        one_pass,
        |tmp, _out, s| module.glwe_tensor_apply(cnv_offset, tmp, a, b, s),
    )
}

#[allow(clippy::too_many_arguments)]
fn square_into_at<BE, M, Out, A, T>(
    module: &M,
    out: &mut Out,
    a: &A,
    tsk: &T,
    cnv_offset: usize,
    stamp: MulStamp,
    scratch: &mut ScratchArena<'_, BE>,
) -> Result<()>
where
    BE: Backend,
    M: GLWETensoring<BE> + ?Sized,
    Out: GLWEToBackendMut<BE> + CKKSInfos + SetCKKSInfos + GLWEInfos,
    A: GLWEToBackendRef<BE> + CKKSInfos + GLWEInfos,
    T: GetTensorKey<BE>,
{
    let one_pass = mul_one_pass(out, &[a.base2k()], a.k(), cnv_offset, tsk).then_some(
        |out: &mut Out, res_k, s: &mut ScratchArena<'_, BE>| module.glwe_square_relinearize(cnv_offset, out, res_k, a, tsk, s),
    );
    tensor_mul_core(
        module,
        out,
        tsk,
        a.k(),
        stamp,
        StampOrder::BeforeApply,
        scratch,
        one_pass,
        |tmp, _out, s| module.glwe_tensor_square_apply(cnv_offset, tmp, a, s),
    )
}

#[allow(clippy::too_many_arguments)]
fn mul_prepared_into_at<BE, M, Out, L, T>(
    module: &M,
    out: &mut Out,
    left: &L,
    prepared: &CKKSPreparedRight<BE>,
    tsk: &T,
    cnv_offset: usize,
    tensor_k: TorusPrecision,
    stamp: MulStamp,
    scratch: &mut ScratchArena<'_, BE>,
) -> Result<()>
where
    BE: Backend,
    M: GLWETensoring<BE> + ?Sized,
    Out: GLWEToBackendMut<BE> + CKKSInfos + SetCKKSInfos + GLWEInfos,
    L: GLWEToBackendRef<BE> + CKKSInfos + GLWEInfos,
    T: GetTensorKey<BE>,
{
    let one_pass = mul_one_pass(out, &[left.base2k(), prepared.base2k()], tensor_k, cnv_offset, tsk).then_some(
        |out: &mut Out, res_k, s: &mut ScratchArena<'_, BE>| {
            module.glwe_mul_prepared_relinearize(
                cnv_offset,
                out,
                res_k,
                left,
                &prepared.prep,
                prepared.size,
                prepared.k(),
                tsk,
                s,
            )
        },
    );
    tensor_mul_core(
        module,
        out,
        tsk,
        tensor_k,
        stamp,
        StampOrder::BeforeApply,
        scratch,
        one_pass,
        |tmp, _out, s| module.glwe_tensor_apply_prepared_right(cnv_offset, tmp, left, &prepared.prep, prepared.size, s),
    )
}

fn mul_pt_vec_into_at<BE, M, Out, A, P>(
    module: &M,
    out: &mut Out,
    a: &A,
    pt: &P,
    cnv_offset: usize,
    stamp: &MulStamp,
    scratch: &mut ScratchArena<'_, BE>,
) where
    BE: Backend,
    M: GLWEMulPlain<BE> + ?Sized,
    Out: GLWEToBackendMut<BE> + CKKSInfos + SetCKKSInfos + GLWEInfos,
    A: GLWEToBackendRef<BE> + CKKSInfos + GLWEInfos,
    P: GLWEToBackendRef<BE> + LWEInfos + IntPolyInfos + GLWEInfos + CKKSInfos,
{
    // Stamp first: `out.size()` is meta-derived and a fresh `out` carries zero
    // meta, so `glwe_mul_plain` (which writes `res.size()` limbs) would
    // otherwise produce an empty output.
    stamp.set_on(out);
    module.glwe_mul_plain(cnv_offset, out, a, pt, scratch);
}

#[allow(clippy::too_many_arguments)]
fn mul_pt_const_into_at<BE, M, Out, A, P>(
    module: &M,
    out: &mut Out,
    a: &A,
    pt: &P,
    pt_coeff: usize,
    cnv_offset: usize,
    stamp: &MulStamp,
    scratch: &mut ScratchArena<'_, BE>,
) where
    BE: Backend,
    M: GLWEMulConst<BE> + ?Sized,
    Out: GLWEToBackendMut<BE> + CKKSInfos + SetCKKSInfos + GLWEInfos,
    A: GLWEToBackendRef<BE> + CKKSInfos + GLWEInfos,
    P: GLWEToBackendRef<BE> + LWEInfos + IntPolyInfos + GLWEInfos + CKKSInfos,
{
    stamp.set_on(out);
    module.glwe_mul_const(cnv_offset, out, a, pt, pt_coeff, scratch);
}

pub(crate) fn is_replicated(infos: &impl CKKSInfos) -> bool {
    matches!(infos.slot_layout(), SlotLayout::Replicated(_))
}

/// Scratch of `plain(res, tensor_k)`, and when `shaped`, of the same op computed
/// `MAX_PASS_BITS` under the scale into a wider ciphertext and rounded into `res`.
fn with_shaped_bytes<BE, M>(
    module: &M,
    res: &impl GLWEInfos,
    tensor_k: TorusPrecision,
    shaped: bool,
    plain: impl Fn(&GLWELayout, TorusPrecision) -> usize,
) -> usize
where
    BE: Backend,
    M: GLWEBytesOf<BE> + CKKSRoundOps<BE> + ?Sized,
{
    let layout = res.glwe_layout();
    if !shaped {
        return plain(&layout, tensor_k);
    }
    let widen = |k: TorusPrecision| TorusPrecision::from(k.as_usize() + MAX_PASS_BITS);
    let wide = GLWELayout {
        k: widen(layout.k),
        ..layout
    };
    let ckks = |glwe_layout| CKKSLayout {
        glwe_layout,
        meta: Default::default(),
    };
    let round = module.ckks_round_tmp_bytes(&ckks(layout), &ckks(wide));
    plain(&layout, tensor_k).max(module.glwe_bytes_of_from_infos(&wide) + plain(&wide, widen(tensor_k)).max(round))
}
