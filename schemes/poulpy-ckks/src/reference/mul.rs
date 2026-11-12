use crate::CKKSResult as Result;
use poulpy_core::layouts::GetTensorKey;
use poulpy_core::layouts::IntPolyInfos;
use poulpy_core::{
    GLWECopy, GLWEMulConst, GLWEMulPlain, GLWENormalize, GLWETensoring, ScratchArenaTakeGLWE, ScratchArenaTakeGLWETensor,
    glwe_left_operand_prepare, glwe_left_operand_prepare_tmp_bytes, glwe_right_operand_prepare,
    glwe_right_operand_prepare_tmp_bytes,
    layouts::{
        GGLWEInfos, GLWEInfos, GLWELayout, GLWEPlaintextLayout, GLWETensorViewMut, GLWEToBackendMut, GLWEToBackendRef, LWEInfos,
        ModuleCoreAlloc, TorusPrecision, prepared::GLWETensorKeyPreparedBackendRef,
    },
};
use poulpy_hal::{
    api::{CnvOps, ModuleN, VecZnxCopy},
    layouts::{Backend, Ring, ScratchArena, Standard},
};

use crate::{
    CKKSCompositionError, CKKSCtBounds, CKKSLayout, SlotLayout, SlotsKind,
    api::CKKSRoundOps,
    layouts::{CKKSCiphertextViewMut, ScratchArenaTakeCKKSCiphertext},
    shaping::MAX_PASS_BITS,
};
use crate::{
    CKKSInfos, GLWEPlaintextInfos, SetCKKSInfos, ckks_mul_pt_const_checks, ckks_mul_pt_vec_checks, get_mul_ct_params,
    layouts::{CKKSLeftOperandPrepared, CKKSRightOperandPrepared},
    mul_ct_params_raw,
};
use poulpy_core::{GLWEBytesOf, GLWETensorBytesOf};

pub trait CKKSMulReference<BE: Backend, Rg: Ring = Standard>: ModuleN {
    fn ckks_mul_tmp_bytes_reference<R, A, B, T>(&self, res: &R, a: &A, b: &B, tsk: &T) -> usize
    where
        Self: GLWEBytesOf<BE> + GLWETensorBytesOf<BE> + CKKSRoundOps<BE, Rg>,
        R: CKKSCtBounds,
        A: CKKSCtBounds,
        B: CKKSCtBounds,
        T: GGLWEInfos,
        Self: GLWETensoring<BE, Rg>,
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
        Self: GLWETensoring<BE, Rg>
            + GLWECopy<BE, Rg>
            + ModuleCoreAlloc<OwnedBuf = BE::OwnedBuf, ZnxWord = BE::ZnxWord>
            + CKKSRoundOps<BE, Rg>,
        Dst: GLWEToBackendMut<BE, Rg> + CKKSInfos + SetCKKSInfos + GLWEInfos,
        A: GLWEToBackendRef<BE, Rg> + CKKSInfos + GLWEInfos,
        B: GLWEToBackendRef<BE, Rg> + CKKSInfos + GLWEInfos,
        T: GetTensorKey<BE, Rg>,
    {
        crate::ensure_operand_degrees("ckks_mul_into", self, [dst.n(), a.n(), b.n()])?;
        let k = a.k().max(b.k());
        tsk.get_tensor_key(k)
            .map_err(|source| CKKSCompositionError::MissingRelinearizationKey {
                op: "ckks_mul_into",
                k: k.into(),
                source,
            })?;

        let slot_layout = crate::layouts::validation::product_layout("ckks_mul_into", &[a.slot_layout(), b.slot_layout()])?;
        let (res_log_budget, res_log_delta, cnv_offset) = get_mul_ct_params("ckks_mul_into", dst, a, b)?;
        let stamp = MulStamp {
            log_budget: res_log_budget,
            log_delta: res_log_delta,
            // The product of values sparse at `s` and `t` is sparse at `min(s, t)`.
            log_sparsity: Some(a.log_sparsity().min(b.log_sparsity())),
            slots: Some(a.slots().join(b.slots())),
            slot_layout: Some(slot_layout),
        };
        match crate::shaping::product_depth(slot_layout, res_log_delta) {
            None => mul_into_at(self, dst, a, b, tsk, cnv_offset, stamp, scratch),
            Some(lambda) => {
                let (mut wide, mut scratch) = take_wide(scratch, &*dst, dst.k().as_usize() + lambda);
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
        Self: GLWETensoring<BE, Rg>
            + GLWECopy<BE, Rg>
            + ModuleCoreAlloc<OwnedBuf = BE::OwnedBuf, ZnxWord = BE::ZnxWord>
            + CKKSRoundOps<BE, Rg>,
        Dst: GLWEToBackendMut<BE, Rg> + GLWEToBackendRef<BE, Rg> + CKKSInfos + SetCKKSInfos + GLWEInfos,
        A: GLWEToBackendRef<BE, Rg> + CKKSInfos + GLWEInfos,
        T: GetTensorKey<BE, Rg>,
    {
        crate::ensure_operand_degrees("ckks_mul_assign", self, [dst.n(), a.n()])?;
        let k = dst.k().max(a.k());
        tsk.get_tensor_key(k)
            .map_err(|source| CKKSCompositionError::MissingRelinearizationKey {
                op: "ckks_mul_assign",
                k: k.into(),
                source,
            })?;

        let slot_layout = crate::layouts::validation::product_layout("ckks_mul_assign", &[dst.slot_layout(), a.slot_layout()])?;
        let (res_log_budget, res_log_delta, cnv_offset) = get_mul_ct_params("ckks_mul_assign", dst, dst, a)?;
        if let Some(lambda) = crate::shaping::product_depth(slot_layout, res_log_delta) {
            let stamp = MulStamp {
                log_budget: res_log_budget,
                log_delta: res_log_delta,
                log_sparsity: Some(dst.log_sparsity().min(a.log_sparsity())),
                slots: Some(dst.slots().join(a.slots())),
                slot_layout: Some(slot_layout),
            };
            let (mut wide, mut scratch) = take_wide(scratch, &*dst, dst.k().as_usize() + lambda);
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

    /// What `glwe_left_operand_prepare` takes to prepare `a` on its own limbs.
    fn ckks_left_operand_prepare_tmp_bytes_reference<A>(&self, a: &A) -> usize
    where
        Self: GLWEBytesOf<BE> + GLWENormalize<BE, Rg> + CnvOps<BE, Rg> + Sized,
        A: GLWEInfos,
    {
        glwe_left_operand_prepare_tmp_bytes(self, a.size(), a)
    }

    /// What `glwe_right_operand_prepare` takes to prepare `a` on its own limbs.
    fn ckks_right_operand_prepare_tmp_bytes_reference<A>(&self, a: &A) -> usize
    where
        Self: GLWEBytesOf<BE> + GLWENormalize<BE, Rg> + CnvOps<BE, Rg> + Sized,
        A: GLWEInfos,
    {
        glwe_right_operand_prepare_tmp_bytes(self, a.size(), a)
    }

    /// `glwe_left_operand_prepare` of `a`, normalized first if its canonical flag is clear, on its `ceil(k / base2k)` limbs,
    /// into `res`, which then reports `a`'s metadata. `InvalidOperand` for an `a` of another layout than `res`'s,
    /// before any write.
    fn ckks_left_operand_prepare_reference<A>(
        &self,
        res: &mut CKKSLeftOperandPrepared<BE, Rg>,
        a: &A,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        Self: CnvOps<BE, Rg> + GLWENormalize<BE, Rg> + Sized,
        A: GLWEToBackendRef<BE, Rg> + CKKSInfos + GLWEInfos,
    {
        crate::ensure_operand_degrees("ckks_left_operand_prepare", self, [res.n(), a.n()])?;
        check_source_layout("ckks_left_operand_prepare", res, a)?;
        glwe_left_operand_prepare(self, &mut res.prep, a, a.k().as_usize(), scratch);
        res.meta = a.plaintext_meta();
        Ok(())
    }

    /// As [`Self::ckks_left_operand_prepare_reference`], by `glwe_right_operand_prepare` into a right operand.
    fn ckks_right_operand_prepare_reference<A>(
        &self,
        res: &mut CKKSRightOperandPrepared<BE, Rg>,
        a: &A,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        Self: CnvOps<BE, Rg> + GLWENormalize<BE, Rg> + Sized,
        A: GLWEToBackendRef<BE, Rg> + CKKSInfos + GLWEInfos,
    {
        crate::ensure_operand_degrees("ckks_right_operand_prepare", self, [res.n(), a.n()])?;
        check_source_layout("ckks_right_operand_prepare", res, a)?;
        glwe_right_operand_prepare(self, &mut res.prep, a, a.k().as_usize(), scratch);
        res.meta = a.plaintext_meta();
        Ok(())
    }

    fn ckks_mul_prepared_assign_reference<Dst, T>(
        &self,
        dst: &mut Dst,
        prepared: &CKKSRightOperandPrepared<BE, Rg>,
        tsk: &T,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        Self: GLWETensoring<BE, Rg> + CKKSRoundOps<BE, Rg>,
        Dst: GLWEToBackendMut<BE, Rg> + GLWEToBackendRef<BE, Rg> + CKKSInfos + SetCKKSInfos + GLWEInfos,
        T: GetTensorKey<BE, Rg>,
    {
        crate::ensure_operand_degrees("ckks_mul_prepared_assign", self, [dst.n(), prepared.n()])?;
        let k = dst.k().max(prepared.k());
        tsk.get_tensor_key(k)
            .map_err(|source| CKKSCompositionError::MissingRelinearizationKey {
                op: "ckks_mul_prepared_assign",
                k: k.into(),
                source,
            })?;

        let slot_layout =
            crate::layouts::validation::product_layout("ckks_mul_prepared_assign", &[dst.slot_layout(), prepared.slot_layout()])?;
        let (res_log_budget, res_log_delta, cnv_offset, tensor_k) = get_mul_prepared_params(&*dst, prepared)?;
        if let Some(lambda) = crate::shaping::product_depth(slot_layout, res_log_delta) {
            let stamp = MulStamp {
                log_budget: res_log_budget,
                log_delta: res_log_delta,
                log_sparsity: Some(dst.log_sparsity().min(prepared.log_sparsity())),
                slots: Some(dst.slots().join(prepared.slots())),
                slot_layout: Some(slot_layout),
            };
            let (mut wide, mut scratch) = take_wide(scratch, &*dst, dst.k().as_usize() + lambda);
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
        Self: GLWEBytesOf<BE> + GLWETensorBytesOf<BE> + CKKSRoundOps<BE, Rg>,
        R: CKKSCtBounds,
        A: CKKSCtBounds,
        T: GGLWEInfos,
        Self: GLWETensoring<BE, Rg>,
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
        Self: GLWETensoring<BE, Rg>
            + GLWECopy<BE, Rg>
            + ModuleCoreAlloc<OwnedBuf = BE::OwnedBuf, ZnxWord = BE::ZnxWord>
            + CKKSRoundOps<BE, Rg>,
        Dst: GLWEToBackendMut<BE, Rg> + CKKSInfos + SetCKKSInfos + GLWEInfos,
        A: GLWEToBackendRef<BE, Rg> + CKKSInfos + GLWEInfos,
        T: GetTensorKey<BE, Rg>,
    {
        crate::ensure_operand_degrees("ckks_square_into", self, [dst.n(), a.n()])?;
        let k = a.k();
        tsk.get_tensor_key(k)
            .map_err(|source| CKKSCompositionError::MissingRelinearizationKey {
                op: "ckks_square_into",
                k: k.into(),
                source,
            })?;

        let (res_log_budget, res_log_delta, cnv_offset) = get_mul_ct_params("ckks_square_into", dst, a, a)?;
        let stamp = MulStamp {
            log_budget: res_log_budget,
            log_delta: res_log_delta,
            log_sparsity: Some(a.log_sparsity()),
            slots: Some(a.slots()),
            slot_layout: Some(a.slot_layout()),
        };
        match crate::shaping::product_depth(a.slot_layout(), res_log_delta) {
            None => square_into_at(self, dst, a, tsk, cnv_offset, stamp, scratch),
            Some(lambda) => {
                let (mut wide, mut scratch) = take_wide(scratch, &*dst, dst.k().as_usize() + lambda);
                square_into_at(self, &mut wide, a, tsk, cnv_offset, stamp.widen(lambda), &mut scratch)?;
                round_into(self, dst, &wide, &stamp, &mut scratch)
            }
        }
    }

    fn ckks_square_assign_reference<Dst, T>(&self, dst: &mut Dst, tsk: &T, scratch: &mut ScratchArena<'_, BE>) -> Result<()>
    where
        Self: GLWETensoring<BE, Rg>
            + GLWECopy<BE, Rg>
            + ModuleCoreAlloc<OwnedBuf = BE::OwnedBuf, ZnxWord = BE::ZnxWord>
            + CKKSRoundOps<BE, Rg>,
        Dst: GLWEToBackendMut<BE, Rg> + GLWEToBackendRef<BE, Rg> + CKKSInfos + SetCKKSInfos + GLWEInfos,
        T: GetTensorKey<BE, Rg>,
    {
        crate::ensure_operand_degrees("ckks_square_assign", self, [dst.n()])?;
        let k = dst.k();
        tsk.get_tensor_key(k)
            .map_err(|source| CKKSCompositionError::MissingRelinearizationKey {
                op: "ckks_square_assign",
                k: k.into(),
                source,
            })?;

        let (res_log_budget, res_log_delta, cnv_offset) = get_mul_ct_params("ckks_square_assign", dst, dst, dst)?;
        if let Some(lambda) = crate::shaping::product_depth(dst.slot_layout(), res_log_delta) {
            let stamp = MulStamp {
                log_budget: res_log_budget,
                log_delta: res_log_delta,
                log_sparsity: Some(dst.log_sparsity()),
                slots: Some(dst.slots()),
                slot_layout: Some(dst.slot_layout()),
            };
            let (mut wide, mut scratch) = take_wide(scratch, &*dst, dst.k().as_usize() + lambda);
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

    fn ckks_mul_pt_vec_tmp_bytes_reference<R, A, P>(&self, res: &R, a: &A, b: &P) -> usize
    where
        Self: GLWEBytesOf<BE> + CKKSRoundOps<BE, Rg>,
        R: CKKSCtBounds,
        A: CKKSCtBounds,
        P: CKKSInfos,
        Self: GLWEMulPlain<BE, Rg>,
    {
        let plain = |res: &GLWELayout, _: TorusPrecision| {
            let b_infos = GLWEPlaintextLayout {
                n: res.n(),
                base2k: res.base2k(),
                k: b.k(),
            };
            self.glwe_bytes_of_from_infos(a) + self.glwe_mul_plain_tmp_bytes(res, a, &b_infos)
        };
        with_shaped_bytes(self, res, a.k(), is_replicated(a), plain)
    }

    fn ckks_mul_pt_const_tmp_bytes_reference<R, A, P>(&self, res: &R, a: &A, b: &P) -> usize
    where
        Self: GLWEBytesOf<BE> + CKKSRoundOps<BE, Rg>,
        R: CKKSCtBounds,
        A: CKKSCtBounds,
        P: CKKSInfos,
        Self: GLWEMulConst<BE, Rg>,
    {
        let plain = |res: &GLWELayout, _: TorusPrecision| {
            let b_infos = GLWEPlaintextLayout {
                n: res.n(),
                base2k: res.base2k(),
                k: b.k(),
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
        P: GLWEToBackendRef<BE, Rg> + LWEInfos + IntPolyInfos + GLWEInfos + CKKSInfos,
        Self: GLWECopy<BE, Rg>
            + GLWEMulPlain<BE, Rg>
            + ModuleCoreAlloc<OwnedBuf = BE::OwnedBuf, ZnxWord = BE::ZnxWord>
            + VecZnxCopy<BE, Rg>
            + CKKSRoundOps<BE, Rg>,
        Dst: GLWEToBackendMut<BE, Rg> + CKKSInfos + SetCKKSInfos + GLWEInfos,
        A: GLWEToBackendRef<BE, Rg> + CKKSInfos + GLWEInfos,
    {
        crate::ensure_operand_degrees("ckks_mul_pt_vec_into", self, [dst.n(), a.n()])?;
        let slot_layout =
            crate::layouts::validation::product_layout("ckks_mul_pt_vec_into", &[a.slot_layout(), pt.slot_layout()])?;
        let (res_log_budget, res_log_delta, cnv_offset) =
            ckks_mul_pt_vec_checks::<BE, _, _, _>("ckks_mul_pt_vec_into", dst, a, pt)?;
        let stamp = MulStamp {
            log_budget: res_log_budget,
            log_delta: res_log_delta,
            // The product of values sparse at `s` and `t` is sparse at `min(s, t)`;
            // a plaintext is dense at its own degree.
            log_sparsity: Some(a.log_sparsity().min(a.log_n() - pt.log_n())),
            slots: Some(a.slots().join(pt.slots())),
            slot_layout: Some(slot_layout),
        };
        match crate::shaping::product_depth(slot_layout, pt.log_delta()) {
            None => {
                mul_pt_vec_into_at(self, dst, a, pt, cnv_offset, &stamp, scratch);
                Ok(())
            }
            Some(lambda) => {
                let (mut wide, mut scratch) = take_wide(scratch, &*dst, dst.k().as_usize() + lambda);
                mul_pt_vec_into_at(self, &mut wide, a, pt, cnv_offset, &stamp.widen(lambda), &mut scratch);
                round_into(self, dst, &wide, &stamp, &mut scratch)
            }
        }
    }

    fn ckks_mul_pt_vec_assign_reference<Dst, P>(&self, dst: &mut Dst, pt: &P, scratch: &mut ScratchArena<'_, BE>) -> Result<()>
    where
        P: GLWEToBackendRef<BE, Rg> + LWEInfos + IntPolyInfos + GLWEInfos + CKKSInfos,
        Self: GLWECopy<BE, Rg>
            + GLWEMulPlain<BE, Rg>
            + ModuleCoreAlloc<OwnedBuf = BE::OwnedBuf, ZnxWord = BE::ZnxWord>
            + VecZnxCopy<BE, Rg>
            + CKKSRoundOps<BE, Rg>,
        Dst: GLWEToBackendMut<BE, Rg> + CKKSInfos + SetCKKSInfos + GLWEInfos,
    {
        crate::ensure_operand_degrees("ckks_mul_pt_vec_assign", self, [dst.n()])?;
        let (res_log_budget, res_log_delta, cnv_offset) =
            ckks_mul_pt_vec_checks::<BE, _, _, _>("ckks_mul_pt_vec_assign", dst, dst, pt)?;
        let slot_layout =
            crate::layouts::validation::product_layout("ckks_mul_pt_vec_assign", &[dst.slot_layout(), pt.slot_layout()])?;
        let log_sparsity = dst.log_sparsity().min(dst.log_n() - pt.log_n());
        let slots = dst.slots().join(pt.slots());
        if let Some(lambda) = crate::shaping::product_depth(slot_layout, pt.log_delta()) {
            let stamp = MulStamp {
                log_budget: res_log_budget,
                log_delta: res_log_delta,
                log_sparsity: Some(log_sparsity),
                slots: Some(slots),
                slot_layout: Some(slot_layout),
            };
            let (mut wide, mut scratch) = take_wide(scratch, &*dst, dst.k().as_usize() + lambda);
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
        P: GLWEToBackendRef<BE, Rg> + LWEInfos + IntPolyInfos + GLWEInfos + CKKSInfos,
        Self: GLWEMulConst<BE, Rg> + CKKSRoundOps<BE, Rg>,
        Dst: GLWEToBackendMut<BE, Rg> + CKKSInfos + SetCKKSInfos + GLWEInfos,
        A: GLWEToBackendRef<BE, Rg> + CKKSInfos + GLWEInfos,
    {
        crate::ensure_operand_degrees("ckks_mul_pt_const_into", self, [dst.n(), a.n()])?;
        let (res_log_budget, res_log_delta, cnv_offset) =
            ckks_mul_pt_const_checks("ckks_mul_pt_const_into", dst, a, pt, pt_coeff)?;
        // A scalar-constant multiply preserves the operand's sparsity pattern,
        // slot kind and layout.
        let stamp = MulStamp {
            log_budget: res_log_budget,
            log_delta: res_log_delta,
            log_sparsity: Some(a.log_sparsity()),
            slots: Some(a.slots()),
            slot_layout: Some(a.slot_layout()),
        };
        match crate::shaping::product_depth(a.slot_layout(), pt.log_delta()) {
            None => {
                mul_pt_const_into_at(self, dst, a, pt, pt_coeff, cnv_offset, &stamp, scratch);
                Ok(())
            }
            Some(lambda) => {
                let (mut wide, mut scratch) = take_wide(scratch, &*dst, dst.k().as_usize() + lambda);
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
        P: GLWEToBackendRef<BE, Rg> + LWEInfos + IntPolyInfos + GLWEInfos + CKKSInfos,
        Self: GLWECopy<BE, Rg>
            + GLWEMulConst<BE, Rg>
            + ModuleCoreAlloc<OwnedBuf = BE::OwnedBuf, ZnxWord = BE::ZnxWord>
            + VecZnxCopy<BE, Rg>
            + CKKSRoundOps<BE, Rg>,
        Dst: GLWEToBackendMut<BE, Rg> + GLWEToBackendRef<BE, Rg> + CKKSInfos + SetCKKSInfos + GLWEInfos,
    {
        crate::ensure_operand_degrees("ckks_mul_pt_const_assign", self, [dst.n()])?;
        let (res_log_budget, res_log_delta, cnv_offset) =
            ckks_mul_pt_const_checks("ckks_mul_pt_const_assign", dst, dst, cnst, cnst_coeff)?;
        if let Some(lambda) = crate::shaping::product_depth(dst.slot_layout(), cnst.log_delta()) {
            let stamp = MulStamp {
                log_budget: res_log_budget,
                log_delta: res_log_delta,
                log_sparsity: Some(dst.log_sparsity()),
                slots: Some(dst.slots()),
                slot_layout: Some(dst.slot_layout()),
            };
            let (mut wide, mut scratch) = take_wide(scratch, &*dst, dst.k().as_usize() + lambda);
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
pub(crate) fn tensor_mul_core<BE, M, Dst, T, Rg: Ring>(
    module: &M,
    dst: &mut Dst,
    tsk: &T,
    tensor_k: TorusPrecision,
    stamp: MulStamp,
    order: StampOrder,
    scratch: &mut ScratchArena<'_, BE>,
    one_pass: Option<impl FnOnce(&mut Dst, TorusPrecision, &mut ScratchArena<'_, BE>)>,
    apply: impl for<'t> FnOnce(&mut GLWETensorViewMut<'t, BE, Rg>, &Dst, &mut ScratchArena<'t, BE>),
) -> Result<()>
where
    BE: Backend,
    M: GLWETensoring<BE, Rg> + ?Sized,
    Dst: GLWEToBackendMut<BE, Rg> + CKKSInfos + SetCKKSInfos + GLWEInfos,
    T: GetTensorKey<BE, Rg>,
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

/// The tensor key of a product of width `k`, `MissingRelinearizationKey` for `op` where `tsk` holds none.
pub(crate) fn tensor_key_at<'k, BE, T, Rg: Ring>(
    op: &'static str,
    tsk: &'k T,
    k: TorusPrecision,
) -> Result<GLWETensorKeyPreparedBackendRef<'k, BE, Rg>>
where
    BE: Backend,
    T: GetTensorKey<BE, Rg>,
{
    tsk.get_tensor_key(k)
        .map_err(|source| CKKSCompositionError::MissingRelinearizationKey { op, k: k.into(), source }.into())
}

/// Whether a product of width `tensor_k` takes the one-pass product and relinearization.
///
/// The one-pass path needs rank 1, one `base2k` across `dst`, the operands other than `dst` and the
/// tensor key, and `cnv_offset >= base2k`. Other products go through the tensor intermediate.
pub(crate) fn mul_one_pass<BE, Dst, T, Rg: Ring>(
    dst: &Dst,
    operands_base2k: &[poulpy_core::layouts::Base2K],
    tensor_k: TorusPrecision,
    cnv_offset: usize,
    tsk: &T,
) -> bool
where
    BE: Backend,
    Dst: GLWEInfos,
    T: GetTensorKey<BE, Rg>,
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
pub(crate) fn get_mul_prepared_params<BE: Backend, D: GLWEInfos + CKKSInfos, Rg: Ring>(
    dst: &D,
    prepared: &CKKSRightOperandPrepared<BE, Rg>,
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
impl<BE: Backend, Rg: Ring> CKKSMulReference<BE, Rg> for poulpy_hal::layouts::Module<BE> {}

/// A ciphertext shaped as `dst` at width `k`, for a result rounded into `dst`.
pub(crate) fn take_wide<'a, BE: Backend + 'a, D: GLWEInfos + CKKSInfos, Rg: Ring>(
    scratch: &'a mut ScratchArena<'_, BE>,
    dst: &D,
    k: usize,
) -> (CKKSCiphertextViewMut<'a, BE, Rg>, ScratchArena<'a, BE>) {
    let layout = GLWELayout {
        k: k.into(),
        ..dst.glwe_layout()
    };
    scratch.borrow().take_ckks_ciphertext_scratch(&layout, dst.meta())
}

/// `a²` computed `lambda` bits under its scale into `wide`, without rounding:
/// the square of a destination `lambda` bits narrower than `wide`, labeled
/// `lambda` bits finer. Refusals carry the caller's `op`.
pub(crate) fn square_wide_into<BE, M, Out, A, T, Rg: Ring>(
    op: &'static str,
    module: &M,
    wide: &mut Out,
    a: &A,
    tsk: &T,
    lambda: usize,
    scratch: &mut ScratchArena<'_, BE>,
) -> Result<()>
where
    BE: Backend,
    M: GLWETensoring<BE, Rg> + ?Sized,
    Out: GLWEToBackendMut<BE, Rg> + CKKSInfos + SetCKKSInfos + GLWEInfos,
    A: GLWEToBackendRef<BE, Rg> + CKKSInfos + GLWEInfos,
    T: GetTensorKey<BE, Rg>,
{
    let k = a.k();
    tsk.get_tensor_key(k)
        .map_err(|source| CKKSCompositionError::MissingRelinearizationKey { op, k: k.into(), source })?;
    let max_k = wide.k().as_usize().saturating_sub(lambda);
    let (log_budget, log_delta, cnv_offset) = mul_ct_params_raw(op, max_k, a.log_delta(), k.into(), a.log_delta(), k.into())?;
    let stamp = MulStamp {
        log_budget,
        log_delta,
        log_sparsity: Some(a.log_sparsity()),
        slots: Some(a.slots()),
        slot_layout: Some(a.slot_layout()),
    };
    square_into_at(module, wide, a, tsk, cnv_offset, stamp.widen(lambda), scratch)
}

/// Rounds a product computed under the scale into `dst` at the width `stamp` labels.
pub(crate) fn round_into<BE, M, Dst, Rg: Ring>(
    module: &M,
    dst: &mut Dst,
    wide: &CKKSCiphertextViewMut<'_, BE, Rg>,
    stamp: &MulStamp,
    scratch: &mut ScratchArena<'_, BE>,
) -> Result<()>
where
    BE: Backend,
    M: CKKSRoundOps<BE, Rg> + ?Sized,
    Dst: GLWEToBackendMut<BE, Rg> + CKKSInfos + SetCKKSInfos + GLWEInfos,
{
    dst.set_k((stamp.log_budget + stamp.log_delta).into());
    module.ckks_round(dst, wide, scratch)
}

#[allow(clippy::too_many_arguments)]
fn mul_into_at<BE, M, Out, A, B, T, Rg: Ring>(
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
    M: GLWETensoring<BE, Rg> + ?Sized,
    Out: GLWEToBackendMut<BE, Rg> + CKKSInfos + SetCKKSInfos + GLWEInfos,
    A: GLWEToBackendRef<BE, Rg> + CKKSInfos + GLWEInfos,
    B: GLWEToBackendRef<BE, Rg> + CKKSInfos + GLWEInfos,
    T: GetTensorKey<BE, Rg>,
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
fn square_into_at<BE, M, Out, A, T, Rg: Ring>(
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
    M: GLWETensoring<BE, Rg> + ?Sized,
    Out: GLWEToBackendMut<BE, Rg> + CKKSInfos + SetCKKSInfos + GLWEInfos,
    A: GLWEToBackendRef<BE, Rg> + CKKSInfos + GLWEInfos,
    T: GetTensorKey<BE, Rg>,
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
fn mul_prepared_into_at<BE, M, Out, L, T, Rg: Ring>(
    module: &M,
    out: &mut Out,
    left: &L,
    prepared: &CKKSRightOperandPrepared<BE, Rg>,
    tsk: &T,
    cnv_offset: usize,
    tensor_k: TorusPrecision,
    stamp: MulStamp,
    scratch: &mut ScratchArena<'_, BE>,
) -> Result<()>
where
    BE: Backend,
    M: GLWETensoring<BE, Rg> + ?Sized,
    Out: GLWEToBackendMut<BE, Rg> + CKKSInfos + SetCKKSInfos + GLWEInfos,
    L: GLWEToBackendRef<BE, Rg> + CKKSInfos + GLWEInfos,
    T: GetTensorKey<BE, Rg>,
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

fn mul_pt_vec_into_at<BE, M, Out, A, P, Rg: Ring>(
    module: &M,
    out: &mut Out,
    a: &A,
    pt: &P,
    cnv_offset: usize,
    stamp: &MulStamp,
    scratch: &mut ScratchArena<'_, BE>,
) where
    BE: Backend,
    M: GLWEMulPlain<BE, Rg> + ?Sized,
    Out: GLWEToBackendMut<BE, Rg> + CKKSInfos + SetCKKSInfos + GLWEInfos,
    A: GLWEToBackendRef<BE, Rg> + CKKSInfos + GLWEInfos,
    P: GLWEToBackendRef<BE, Rg> + LWEInfos + IntPolyInfos + GLWEInfos + CKKSInfos,
{
    // Stamp first: `out.size()` is meta-derived and a fresh `out` carries zero
    // meta, so `glwe_mul_plain` (which writes `res.size()` limbs) would
    // otherwise produce an empty output.
    stamp.set_on(out);
    module.glwe_mul_plain(cnv_offset, out, a, pt, scratch);
}

#[allow(clippy::too_many_arguments)]
fn mul_pt_const_into_at<BE, M, Out, A, P, Rg: Ring>(
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
    M: GLWEMulConst<BE, Rg> + ?Sized,
    Out: GLWEToBackendMut<BE, Rg> + CKKSInfos + SetCKKSInfos + GLWEInfos,
    A: GLWEToBackendRef<BE, Rg> + CKKSInfos + GLWEInfos,
    P: GLWEToBackendRef<BE, Rg> + LWEInfos + IntPolyInfos + GLWEInfos + CKKSInfos,
{
    stamp.set_on(out);
    module.glwe_mul_const(cnv_offset, out, a, pt, pt_coeff, scratch);
}

pub(crate) fn is_replicated(infos: &impl CKKSInfos) -> bool {
    matches!(infos.slot_layout(), SlotLayout::Replicated(_))
}

/// Scratch of `plain(res, tensor_k)`, and when `shaped`, of the same op computed
/// `MAX_PASS_BITS` under the scale into a wider ciphertext and rounded into `res`.
pub(crate) fn with_shaped_bytes<BE, M, Rg: Ring>(
    module: &M,
    res: &impl GLWEInfos,
    tensor_k: TorusPrecision,
    shaped: bool,
    plain: impl Fn(&GLWELayout, TorusPrecision) -> usize,
) -> usize
where
    BE: Backend,
    M: CKKSRoundOps<BE, Rg> + ?Sized,
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
    let wide_bytes = BE::scratch_aligned(BE::vec_znx_bytes_of(wide.n.as_usize(), wide.rank.as_usize() + 1, wide.size()));
    plain(&layout, tensor_k).max(wide_bytes + plain(&wide, widen(tensor_k)).max(round))
}
