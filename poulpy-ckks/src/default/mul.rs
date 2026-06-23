use anyhow::Result;
use poulpy_core::{
    GLWECopy, GLWEMulConst, GLWEMulPlain, GLWERotate, GLWETensoring, GiantStepTensorBounds, ScratchArenaTakeCore,
    glwe_prepare_right, glwe_tensor_apply_prepared_right,
    layouts::{
        Compact, GGLWEInfos, GLWE, GLWEInfos, GLWELayout, GLWEPlaintextLayout, GLWETensor, GLWEToBackendMut, GLWEToBackendRef,
        LWEInfos, ModuleCoreAlloc, TorusPrecision, prepared::GLWETensorKeyPreparedToBackendRef,
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
    fn ckks_mul_tmp_bytes_default<R, T>(&self, res: &R, tsk: &T) -> usize
    where
        R: GLWEInfos,
        T: GGLWEInfos,
        Self: GLWETensoring<BE>,
    {
        let glwe_layout = GLWELayout {
            n: res.n(),
            base2k: res.base2k(),
            k: TorusPrecision(res.max_k().as_u32()),
            rank: res.rank(),
        };

        let lvl_0 = GLWETensor::bytes_of_from_infos(&glwe_layout);
        let lvl_1 = self
            .glwe_tensor_apply_tmp_bytes(&glwe_layout, res, res)
            .max(self.glwe_tensor_relinearize_tmp_bytes(res, &glwe_layout, tsk));

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

        // Set the result metadata before the tensoring: `dst.size()` is derived
        // from `k() = log_budget + log_delta`, and `glwe_tensor_relinearize` uses
        // `res.size()` to size the result. A freshly-allocated `dst` carries zero
        // meta (hence `size() == 0`), which would leave the output empty.
        dst.set_log_budget(res_log_budget);
        dst.set_log_delta(res_log_delta);

        let tensor_layout = GLWELayout {
            n: dst.n(),
            base2k: dst.base2k(),
            k: a.max_k().max(b.max_k()),
            rank: dst.rank(),
        };
        let scratch_local = scratch.borrow();
        let (mut tmp, mut scratch_local) = scratch_local.take_glwe_tensor_scratch(&tensor_layout);
        self.glwe_tensor_apply(cnv_offset, &mut tmp, a, b, &mut scratch_local);
        self.glwe_tensor_relinearize(dst, &tmp, tsk, tmp.max_size() + tsk.dsize().as_usize(), &mut scratch_local);

        dst.compact();
        Ok(())
    }

    fn ckks_mul_assign_default<Dst, A, T>(&self, dst: &mut Dst, a: &A, tsk: &T, scratch: &mut ScratchArena<'_, BE>) -> Result<()>
    where
        Self: GLWETensoring<BE> + GLWECopy<BE> + ModuleCoreAlloc<OwnedBuf = BE::OwnedBuf>,
        Dst: GLWEToBackendMut<BE> + GLWEToBackendRef<BE> + CKKSInfos + SetCKKSInfos + GLWEInfos + Compact,
        A: GLWEToBackendRef<BE> + CKKSInfos + GLWEInfos,
        T: GLWETensorKeyPreparedToBackendRef<BE> + GGLWEInfos,
    {
        let (res_log_budget, res_log_delta, cnv_offset) = get_mul_ct_params(dst, dst, a)?;

        let tensor_layout = GLWELayout {
            n: dst.n(),
            base2k: dst.base2k(),
            k: dst.max_k().max(a.max_k()),
            rank: dst.rank(),
        };
        let scratch_local = scratch.borrow();
        let (mut tmp, mut scratch_local) = scratch_local.take_glwe_tensor_scratch(&tensor_layout);
        self.glwe_tensor_apply(cnv_offset, &mut tmp, &*dst, a, &mut scratch_local);
        self.glwe_tensor_relinearize(dst, &tmp, tsk, tmp.max_size() + tsk.dsize().as_usize(), &mut scratch_local);

        dst.set_log_budget(res_log_budget);
        dst.set_log_delta(res_log_delta);
        dst.compact();
        Ok(())
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
            log_budget: a.log_budget(),
            k,
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
        let (res_log_budget, res_log_delta, cnv_offset) = mul_ct_params_raw(
            dst.max_k().as_usize(),
            dst.log_delta(),
            dst.log_budget(),
            dst.k().into(),
            prepared.log_delta,
            prepared.log_budget,
            prepared.k,
        )?;

        // Size the intermediate from the right operand's `k` rather than
        // its full `max_k`: the tensor product only consumes the top `k`
        // limbs (via the prepared operand).
        let tensor_layout = GLWELayout {
            n: dst.n(),
            base2k: dst.base2k(),
            k: dst.max_k().max(TorusPrecision(prepared.k as u32)),
            rank: dst.rank(),
        };
        let scratch_local = scratch.borrow();
        let (mut tmp, mut scratch_local) = scratch_local.take_glwe_tensor_scratch(&tensor_layout);
        let dst_k = dst.k();
        glwe_tensor_apply_prepared_right(
            self,
            cnv_offset,
            &mut tmp,
            &*dst,
            dst_k.into(),
            &prepared.prep,
            prepared.size,
            &mut scratch_local,
        );
        self.glwe_tensor_relinearize(dst, &tmp, tsk, tmp.max_size() + tsk.dsize().as_usize(), &mut scratch_local);

        dst.set_log_budget(res_log_budget);
        dst.set_log_delta(res_log_delta);
        dst.compact();
        Ok(())
    }

    fn ckks_square_tmp_bytes_default<R, T>(&self, res: &R, tsk: &T) -> usize
    where
        R: GLWEInfos,
        T: GGLWEInfos,
        Self: GLWETensoring<BE>,
    {
        let glwe_layout = GLWELayout {
            n: res.n(),
            base2k: res.base2k(),
            k: TorusPrecision(res.max_k().as_u32()),
            rank: res.rank(),
        };

        let lvl_0 = GLWETensor::bytes_of_from_infos(&glwe_layout);
        let lvl_1 = self
            .glwe_tensor_square_apply_tmp_bytes(&glwe_layout, res)
            .max(self.glwe_tensor_relinearize_tmp_bytes(res, &glwe_layout, tsk));

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

        // Set the result metadata before the tensoring: `dst.size()` is derived
        // from `k() = log_budget + log_delta`, and `glwe_tensor_relinearize` uses
        // `res.size()` to size the result. A freshly-allocated `dst` carries zero
        // meta (hence `size() == 0`), which would leave the output empty.
        dst.set_log_budget(res_log_budget);
        dst.set_log_delta(res_log_delta);

        let tensor_layout = GLWELayout {
            n: dst.n(),
            base2k: dst.base2k(),
            k: a.max_k(),
            rank: dst.rank(),
        };
        let scratch_local = scratch.borrow();
        let (mut tmp, mut scratch_local) = scratch_local.take_glwe_tensor_scratch(&tensor_layout);
        self.glwe_tensor_square_apply(cnv_offset, &mut tmp, a, &mut scratch_local);
        self.glwe_tensor_relinearize(dst, &tmp, tsk, tmp.max_size() + tsk.dsize().as_usize(), &mut scratch_local);

        dst.compact();
        Ok(())
    }

    fn ckks_square_assign_default<Dst, T>(&self, dst: &mut Dst, tsk: &T, scratch: &mut ScratchArena<'_, BE>) -> Result<()>
    where
        Self: GLWETensoring<BE> + GLWECopy<BE> + ModuleCoreAlloc<OwnedBuf = BE::OwnedBuf>,
        Dst: GLWEToBackendMut<BE> + GLWEToBackendRef<BE> + CKKSInfos + SetCKKSInfos + GLWEInfos + Compact,
        T: GLWETensorKeyPreparedToBackendRef<BE> + GGLWEInfos,
    {
        let (res_log_budget, res_log_delta, cnv_offset) = get_mul_ct_params(dst, dst, dst)?;

        let tensor_layout = GLWELayout {
            n: dst.n(),
            base2k: dst.base2k(),
            k: dst.max_k(),
            rank: dst.rank(),
        };
        let scratch_local = scratch.borrow();
        let (mut tmp, mut scratch_local) = scratch_local.take_glwe_tensor_scratch(&tensor_layout);
        self.glwe_tensor_square_apply(cnv_offset, &mut tmp, &*dst, &mut scratch_local);
        self.glwe_tensor_relinearize(dst, &tmp, tsk, tmp.max_size() + tsk.dsize().as_usize(), &mut scratch_local);

        dst.set_log_budget(res_log_budget);
        dst.set_log_delta(res_log_delta);
        dst.compact();
        Ok(())
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
        self.glwe_mul_plain(cnv_offset, dst, a, a.k().into(), pt, pt.max_k().as_usize(), scratch);
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
        let dst_k = dst.k();
        self.glwe_mul_plain_assign(cnv_offset, dst, dst_k.into(), pt, pt.max_k().as_usize(), scratch);
        dst.set_log_budget(res_log_budget);
        dst.set_log_delta(res_log_delta);
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

fn get_mul_ct_params<R, A, B>(res: &R, a: &A, b: &B) -> Result<(usize, usize, usize)>
where
    R: LWEInfos + CKKSInfos,
    A: LWEInfos + CKKSInfos,
    B: LWEInfos + CKKSInfos,
{
    mul_ct_params_raw(
        res.max_k().as_usize(),
        a.log_delta(),
        a.log_budget(),
        a.k().into(),
        b.log_delta(),
        b.log_budget(),
        b.k().into(),
    )
}

/// Shared `(log_budget, log_delta, cnv_offset)` rule for ct × ct multiplication,
/// expressed on raw values so the BSGS driver computes bit-identical parameters.
#[allow(clippy::too_many_arguments)]
pub(crate) fn mul_ct_params_raw(
    res_max_k: usize,
    a_log_delta: usize,
    a_log_budget: usize,
    a_k: usize,
    b_log_delta: usize,
    b_log_budget: usize,
    b_k: usize,
) -> Result<(usize, usize, usize)> {
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
    R: LWEInfos + CKKSInfos,
    A: LWEInfos + CKKSInfos,
    B: LWEInfos + CKKSInfos,
{
    mul_pt_params_raw(
        res.max_k().as_usize(),
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
