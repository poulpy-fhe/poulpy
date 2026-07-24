use crate::CKKSResult as Result;
use poulpy_core::{
    GLWECopy, GLWEMulConst, GLWEMulPlain, GLWERotate, GLWETensoring, GiantStepTensorBounds, ScratchArenaTakeCore,
    glwe_prepare_right, glwe_tensor_apply_prepared_right,
    layouts::{
        Compact, GGLWEInfos, GLWE, GLWEInfos, GLWELayout, GLWEPlaintextLayout, GLWETensor, GLWETensorViewMut, GLWEToBackendMut,
        GLWEToBackendRef, LWEInfos, ModuleCoreAlloc, TorusPrecision, prepared::GLWETensorKeyPreparedToBackendRef,
    },
};
use poulpy_hal::{
    api::{CnvPVecAlloc, Convolution, VecZnxCopyBackend},
    layouts::{Backend, ScratchArena},
};

use crate::{
    CKKSInfos, SetCKKSInfos, checked_log_budget_sub, checked_mul_ct_log_budget, checked_mul_pt_log_budget,
    layouts::CKKSPreparedRight,
};

pub trait CKKSMulDefault<BE: Backend> {
    fn ckks_mul_tmp_bytes_default<R, A, B, T>(&self, res: &R, a: &A, b: &B, tsk: &T) -> usize
    where
        R: GLWEInfos,
        A: GLWEInfos,
        B: GLWEInfos,
        T: GGLWEInfos,
        Self: GLWETensoring<BE>,
    {
        // The op carves its tensor intermediate at `max(a.max_k, b.max_k)`
        // (`ckks_mul_into`) or `max(dst.max_k, a.max_k)` (`ckks_mul_assign`), so
        // sizing must cover the widest of the three — a destination narrower than
        // its operands is a supported call, and the tensor-apply scratch also
        // scales with the real operand limb counts.
        let tensor_layout = GLWELayout {
            n: res.n(),
            base2k: res.base2k(),
            k: TorusPrecision(res.max_k().max(a.max_k()).max(b.max_k()).as_u32()),
            rank: res.rank(),
        };

        let lvl_0 = GLWETensor::bytes_of_from_infos(&tensor_layout);
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
        Self: GLWETensoring<BE> + GLWECopy<BE> + ModuleCoreAlloc<OwnedBuf = BE::OwnedBuf>,
        Dst: GLWEToBackendMut<BE> + CKKSInfos + SetCKKSInfos + GLWEInfos + Compact,
        A: GLWEToBackendRef<BE> + CKKSInfos + GLWEInfos,
        B: GLWEToBackendRef<BE> + CKKSInfos + GLWEInfos,
        T: GLWETensorKeyPreparedToBackendRef<BE> + GGLWEInfos,
    {
        let (res_log_budget, res_log_delta, cnv_offset) = get_mul_ct_params(dst, a, b)?;

        tensor_mul_core(
            self,
            dst,
            tsk,
            a.max_k().max(b.max_k()),
            MulStamp {
                log_budget: res_log_budget,
                log_delta: res_log_delta,
                // The product of values sparse at `s` and `t` is sparse at `min(s, t)`.
                log_sparsity: Some(a.log_sparsity().min(b.log_sparsity())),
            },
            StampOrder::BeforeApply,
            scratch,
            |tmp, _dst, s| self.glwe_tensor_apply(cnv_offset, tmp, a, b, s),
        )
    }

    fn ckks_mul_assign_default<Dst, A, T>(&self, dst: &mut Dst, a: &A, tsk: &T, scratch: &mut ScratchArena<'_, BE>) -> Result<()>
    where
        Self: GLWETensoring<BE> + GLWECopy<BE> + ModuleCoreAlloc<OwnedBuf = BE::OwnedBuf>,
        Dst: GLWEToBackendMut<BE> + GLWEToBackendRef<BE> + CKKSInfos + SetCKKSInfos + GLWEInfos + Compact,
        A: GLWEToBackendRef<BE> + CKKSInfos + GLWEInfos,
        T: GLWETensorKeyPreparedToBackendRef<BE> + GGLWEInfos,
    {
        let (res_log_budget, res_log_delta, cnv_offset) = get_mul_ct_params(dst, dst, a)?;

        tensor_mul_core(
            self,
            dst,
            tsk,
            dst.max_k().max(a.max_k()),
            MulStamp {
                log_budget: res_log_budget,
                log_delta: res_log_delta,
                // The product of values sparse at `s` and `t` is sparse at `min(s, t)`.
                log_sparsity: Some(dst.log_sparsity().min(a.log_sparsity())),
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
        Dst: GLWEToBackendMut<BE> + GLWEToBackendRef<BE> + CKKSInfos + SetCKKSInfos + GLWEInfos + Compact,
        T: GLWETensorKeyPreparedToBackendRef<BE> + GGLWEInfos,
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
            dst.max_k().max(TorusPrecision(prepared.k as u32)),
            MulStamp {
                log_budget: res_log_budget,
                log_delta: res_log_delta,
                // The product of values sparse at `s` and `t` is sparse at `min(s, t)`.
                log_sparsity: Some(dst.log_sparsity().min(prepared.log_sparsity)),
            },
            StampOrder::AfterApply,
            scratch,
            |tmp, dst_ref, s| glwe_tensor_apply_prepared_right(self, cnv_offset, tmp, dst_ref, &prepared.prep, prepared.size, s),
        )
    }

    fn ckks_square_tmp_bytes_default<R, A, T>(&self, res: &R, a: &A, tsk: &T) -> usize
    where
        R: GLWEInfos,
        A: GLWEInfos,
        T: GGLWEInfos,
        Self: GLWETensoring<BE>,
    {
        // Mirror of `ckks_mul_tmp_bytes_default`: the op's tensor intermediate is
        // carved at the operand's width, which may exceed the destination's.
        let tensor_layout = GLWELayout {
            n: res.n(),
            base2k: res.base2k(),
            k: TorusPrecision(res.max_k().max(a.max_k()).as_u32()),
            rank: res.rank(),
        };

        let lvl_0 = GLWETensor::bytes_of_from_infos(&tensor_layout);
        let lvl_1 = self
            .glwe_tensor_square_apply_tmp_bytes(&tensor_layout, a)
            .max(self.glwe_tensor_relinearize_tmp_bytes(res, &tensor_layout, tsk));

        lvl_0 + lvl_1
    }

    fn ckks_square_into_default<Dst, A, T>(&self, dst: &mut Dst, a: &A, tsk: &T, scratch: &mut ScratchArena<'_, BE>) -> Result<()>
    where
        Self: GLWETensoring<BE> + GLWECopy<BE> + ModuleCoreAlloc<OwnedBuf = BE::OwnedBuf>,
        Dst: GLWEToBackendMut<BE> + CKKSInfos + SetCKKSInfos + GLWEInfos + Compact,
        A: GLWEToBackendRef<BE> + CKKSInfos + GLWEInfos,
        T: GLWETensorKeyPreparedToBackendRef<BE> + GGLWEInfos,
    {
        let (res_log_budget, res_log_delta, cnv_offset) = get_mul_ct_params(dst, a, a)?;

        tensor_mul_core(
            self,
            dst,
            tsk,
            a.max_k(),
            MulStamp {
                log_budget: res_log_budget,
                log_delta: res_log_delta,
                log_sparsity: Some(a.log_sparsity()),
            },
            StampOrder::BeforeApply,
            scratch,
            |tmp, _dst, s| self.glwe_tensor_square_apply(cnv_offset, tmp, a, s),
        )
    }

    fn ckks_square_assign_default<Dst, T>(&self, dst: &mut Dst, tsk: &T, scratch: &mut ScratchArena<'_, BE>) -> Result<()>
    where
        Self: GLWETensoring<BE> + GLWECopy<BE> + ModuleCoreAlloc<OwnedBuf = BE::OwnedBuf>,
        Dst: GLWEToBackendMut<BE> + GLWEToBackendRef<BE> + CKKSInfos + SetCKKSInfos + GLWEInfos + Compact,
        T: GLWETensorKeyPreparedToBackendRef<BE> + GGLWEInfos,
    {
        let (res_log_budget, res_log_delta, cnv_offset) = get_mul_ct_params(dst, dst, dst)?;

        tensor_mul_core(
            self,
            dst,
            tsk,
            dst.max_k(),
            MulStamp {
                log_budget: res_log_budget,
                log_delta: res_log_delta,
                // `min(dst, dst)` is the identity — squaring leaves sparsity as-is.
                log_sparsity: None,
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
        R: GLWEInfos,
        A: GLWEInfos,
        Self: GLWEMulConst<BE> + GLWERotate<BE>,
    {
        let b_infos = GLWEPlaintextLayout {
            n: res.n(),
            base2k: res.base2k(),
            k: b_k,
        };
        GLWE::<Vec<u8>>::bytes_of_from_infos(res)
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
        P: GLWEToBackendRef<BE> + LWEInfos + GLWEInfos + CKKSInfos,
        Self: GLWECopy<BE> + GLWEMulPlain<BE> + ModuleCoreAlloc<OwnedBuf = BE::OwnedBuf> + VecZnxCopyBackend<BE>,
        Dst: GLWEToBackendMut<BE> + CKKSInfos + SetCKKSInfos + GLWEInfos + Compact,
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
        // `set_log_*` updates the effective torus width but deliberately does
        // not grow a buffer compacted by an earlier operation. Re-expand the
        // complete active backend view before writing: the core copy then
        // zeroes every limb above the natural result, including carry-guard
        // limbs that `compact()` may retain. This makes caller-owned
        // destinations safely reusable across different result widths.
        dst.set_size(dst.max_size());
        self.glwe_mul_plain(cnv_offset, dst, a, pt, scratch);
        dst.compact();
        Ok(())
    }

    fn ckks_mul_pt_vec_assign_default<Dst, P>(&self, dst: &mut Dst, pt: &P, scratch: &mut ScratchArena<'_, BE>) -> Result<()>
    where
        P: GLWEToBackendRef<BE> + LWEInfos + GLWEInfos + CKKSInfos,
        Self: GLWECopy<BE> + GLWEMulPlain<BE> + ModuleCoreAlloc<OwnedBuf = BE::OwnedBuf> + VecZnxCopyBackend<BE>,
        Dst: GLWEToBackendMut<BE> + CKKSInfos + SetCKKSInfos + GLWEInfos + Compact,
    {
        let (res_log_budget, res_log_delta, cnv_offset) = get_mul_pt_params(dst, dst, pt)?;
        self.glwe_mul_plain_assign(cnv_offset, dst, pt, scratch);
        dst.set_log_budget(res_log_budget);
        dst.set_log_delta(res_log_delta);
        // The product of values sparse at `s` and `t` is sparse at `min(s, t)`.
        dst.set_log_sparsity(dst.log_sparsity().min(pt.log_sparsity()));
        dst.compact();
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
        P: GLWEToBackendRef<BE> + LWEInfos + GLWEInfos + CKKSInfos,
        Self: GLWEMulConst<BE>,
        Dst: GLWEToBackendMut<BE> + CKKSInfos + SetCKKSInfos + GLWEInfos + Compact,
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
        self.glwe_mul_const(cnv_offset, dst, a, pt, pt_coeff, scratch);

        dst.compact();
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
        P: GLWEToBackendRef<BE> + LWEInfos + GLWEInfos + CKKSInfos,
        Self: GLWEMulConst<BE>,
        Dst: GLWEToBackendMut<BE> + GLWEToBackendRef<BE> + CKKSInfos + SetCKKSInfos + GLWEInfos + Compact,
    {
        let (res_log_budget, res_log_delta, cnv_offset) = get_mul_pt_params(dst, dst, cnst)?;

        self.glwe_mul_const_assign(cnv_offset, dst, cnst, cnst_coeff, scratch);

        dst.set_log_budget(res_log_budget);
        dst.set_log_delta(res_log_delta);
        dst.compact();
        Ok(())
    }
}

/// Result metadata stamped on `dst` by [`tensor_mul_core`]: the checked budget
/// and scale, plus the variant's sparsity rule (`None` leaves `dst`'s sparsity
/// unchanged — squaring in place, where `min(dst, dst)` is the identity).
struct MulStamp {
    log_budget: usize,
    log_delta: usize,
    log_sparsity: Option<usize>,
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
/// variant's `apply` into it, relinearize into `dst`, stamp, compact.
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
    Dst: GLWEToBackendMut<BE> + CKKSInfos + SetCKKSInfos + GLWEInfos + Compact,
    T: GLWETensorKeyPreparedToBackendRef<BE> + GGLWEInfos,
{
    let do_stamp = |dst: &mut Dst| {
        dst.set_log_budget(stamp.log_budget);
        dst.set_log_delta(stamp.log_delta);
        if let Some(log_sparsity) = stamp.log_sparsity {
            dst.set_log_sparsity(log_sparsity);
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
    dst.compact();
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

pub(crate) fn get_mul_pt_params<R, A, B>(res: &R, a: &A, b: &B) -> Result<(usize, usize, usize)>
where
    R: CKKSInfos,
    A: CKKSInfos,
    B: CKKSInfos,
{
    mul_pt_params_raw(
        res.k().as_usize(),
        a.log_delta(),
        a.log_budget(),
        b.log_delta(),
        b.log_budget(),
        b.max_k().as_usize(),
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
