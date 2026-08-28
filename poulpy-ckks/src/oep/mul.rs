use crate::CKKSResult as Result;
use crate::api::{CKKSAddOps, CKKSMulIntoItem, CKKSMulOps, CKKSPreparedMulAssignItem, CKKSSquareAssignItem, CKKSSquareIntoItem};
use crate::default::mul::{
    CKKSMulAddPtConstPlan, CKKSMulDefault, ckks_mul_add_pt_consts_into_ordered, ckks_mul_into_batch_ordered,
    ckks_mul_into_batch_tmp_bytes_ordered, ckks_mul_prepared_assign_batch_ordered,
    ckks_mul_prepared_assign_batch_tmp_bytes_ordered, ckks_square_assign_batch_ordered,
    ckks_square_assign_batch_tmp_bytes_ordered, ckks_square_into_batch_ordered, ckks_square_into_batch_tmp_bytes_ordered,
};
use poulpy_core::layouts::IntPolyInfos;

use poulpy_core::{
    GLWEAdd, GLWECopy, GLWEMulConst, GLWEMulPlain, GLWERotate, GLWETensoring, GiantStepTensorBounds,
    layouts::{
        GGLWEActiveUse, GGLWEInfos, GLWEInfos, LWEInfos, ModuleCoreAlloc, TorusPrecision,
        prepared::{GGLWEPreparedToBackendRef, GLWETensorKeyPreparedBound, GLWETensorKeyPreparedToBackendRef},
    },
};
use poulpy_hal::{
    api::{CnvPVecAlloc, VecZnxCopyBackend},
    layouts::{Backend, Module, ScratchArena},
};

use crate::{
    CKKSCtBounds, CKKSInfos, GLWEToBackendMut, GLWEToBackendRef, SetCKKSInfos,
    layouts::{CKKSPreparedRight, CKKSPreparedRightInfos},
};

/// # Safety
///
/// Implementations must satisfy the contracts of all trait methods, including
/// any HAL-level invariants (alignment, layout, scratch sizing) implied by the
/// associated method signatures.
pub unsafe trait CKKSMulImpl<BE: Backend>: Backend {
    fn ckks_mul_tmp_bytes_impl<R: GLWEInfos, A: GLWEInfos, B: GLWEInfos, T: GGLWEInfos>(
        module: &Module<BE>,
        res: &R,
        a: &A,
        b: &B,
        tsk: &T,
    ) -> usize;
    fn ckks_square_tmp_bytes_impl<R: GLWEInfos, A: GLWEInfos, T: GGLWEInfos>(
        module: &Module<BE>,
        res: &R,
        a: &A,
        tsk: &T,
    ) -> usize;
    fn ckks_mul_pt_vec_tmp_bytes_impl<R: GLWEInfos, A: GLWEInfos>(
        module: &Module<BE>,
        res: &R,
        a: &A,
        b_k: TorusPrecision,
    ) -> usize;
    fn ckks_mul_pt_const_tmp_bytes_impl<R: GLWEInfos, A: GLWEInfos>(
        module: &Module<BE>,
        res: &R,
        a: &A,
        b_k: TorusPrecision,
    ) -> usize;
    fn ckks_mul_into_impl<Dst, A, B, T>(
        module: &Module<BE>,
        dst: &mut Dst,
        a: &A,
        b: &B,
        tsk: &T,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        Dst: GLWEToBackendMut<BE> + CKKSInfos + SetCKKSInfos + GLWEInfos,
        A: GLWEToBackendRef<BE> + CKKSInfos + GLWEInfos,
        B: GLWEToBackendRef<BE> + CKKSInfos + GLWEInfos,
        T: GLWETensorKeyPreparedToBackendRef<BE> + GGLWEPreparedToBackendRef<BE> + GGLWEInfos;
    fn ckks_mul_assign_impl<Dst, A, T>(
        module: &Module<BE>,
        dst: &mut Dst,
        a: &A,
        tsk: &T,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        Dst: GLWEToBackendMut<BE> + GLWEToBackendRef<BE> + CKKSInfos + SetCKKSInfos + GLWEInfos,
        A: GLWEToBackendRef<BE> + CKKSInfos + GLWEInfos,
        T: GLWETensorKeyPreparedToBackendRef<BE> + GGLWEPreparedToBackendRef<BE> + GGLWEInfos;
    fn ckks_prepare_right_impl<A>(
        module: &Module<BE>,
        a: &A,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<CKKSPreparedRight<BE>>
    where
        A: GLWEToBackendRef<BE> + CKKSInfos + GLWEInfos;
    fn ckks_mul_prepared_assign_impl<Dst, T>(
        module: &Module<BE>,
        dst: &mut Dst,
        prepared: &CKKSPreparedRight<BE>,
        tsk: &T,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        Dst: GLWEToBackendMut<BE> + GLWEToBackendRef<BE> + CKKSInfos + SetCKKSInfos + GLWEInfos,
        T: GLWETensorKeyPreparedToBackendRef<BE> + GGLWEPreparedToBackendRef<BE> + GGLWEInfos;
    fn ckks_square_into_impl<Dst, A, T>(
        module: &Module<BE>,
        dst: &mut Dst,
        a: &A,
        tsk: &T,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        Dst: GLWEToBackendMut<BE> + CKKSInfos + SetCKKSInfos + GLWEInfos,
        A: GLWEToBackendRef<BE> + CKKSInfos + GLWEInfos,
        T: GLWETensorKeyPreparedToBackendRef<BE> + GGLWEPreparedToBackendRef<BE> + GGLWEInfos;
    fn ckks_square_assign_impl<Dst, T>(
        module: &Module<BE>,
        dst: &mut Dst,
        tsk: &T,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        Dst: GLWEToBackendMut<BE> + GLWEToBackendRef<BE> + CKKSInfos + SetCKKSInfos + GLWEInfos,
        T: GLWETensorKeyPreparedToBackendRef<BE> + GGLWEPreparedToBackendRef<BE> + GGLWEInfos;
    fn ckks_mul_pt_vec_into_impl<Dst, A, P>(
        module: &Module<BE>,
        dst: &mut Dst,
        a: &A,
        pt: &P,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        Dst: GLWEToBackendMut<BE> + CKKSInfos + SetCKKSInfos + GLWEInfos,
        A: GLWEToBackendRef<BE> + CKKSInfos + GLWEInfos,
        P: GLWEToBackendRef<BE> + LWEInfos + IntPolyInfos + CKKSCtBounds;
    fn ckks_mul_pt_vec_assign_impl<Dst, P>(
        module: &Module<BE>,
        dst: &mut Dst,
        pt: &P,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        Dst: GLWEToBackendMut<BE> + CKKSInfos + SetCKKSInfos + GLWEInfos,
        P: GLWEToBackendRef<BE> + LWEInfos + IntPolyInfos + CKKSCtBounds;
    fn ckks_mul_pt_const_into_impl<Dst, A, P>(
        module: &Module<BE>,
        dst: &mut Dst,
        a: &A,
        pt: &P,
        pt_coeff: usize,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        Dst: GLWEToBackendMut<BE> + CKKSInfos + SetCKKSInfos + GLWEInfos,
        A: GLWEToBackendRef<BE> + CKKSInfos + GLWEInfos,
        P: GLWEToBackendRef<BE> + LWEInfos + IntPolyInfos + CKKSCtBounds;
    fn ckks_mul_pt_const_assign_impl<Dst, P>(
        module: &Module<BE>,
        dst: &mut Dst,
        pt: &P,
        pt_coeff: usize,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        Dst: GLWEToBackendMut<BE> + GLWEToBackendRef<BE> + CKKSInfos + SetCKKSInfos + GLWEInfos,
        P: GLWEToBackendRef<BE> + LWEInfos + IntPolyInfos + CKKSCtBounds;

    /// Ordered batch of `dst += a·coeffs[idx]` over `terms`.
    ///
    /// Provided, so an existing explicit `CKKSMulImpl` keeps compiling: the
    /// default is the ordered scalar composition
    /// ([`ckks_mul_add_pt_consts_into_ordered`]). An override may fuse the terms
    /// but must reproduce each one's convolution offset, rounding, budget
    /// alignment, carry normalization and metadata step, in order.
    #[allow(clippy::too_many_arguments)]
    fn ckks_mul_add_pt_consts_into_impl<Dst, A, P>(
        module: &Module<BE>,
        dst: &mut Dst,
        terms: &[(&A, usize)],
        plans: &[CKKSMulAddPtConstPlan],
        coeffs: &P,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        Module<BE>: CKKSMulOps<BE> + CKKSAddOps<BE>,
        Dst: GLWEToBackendMut<BE> + CKKSCtBounds + SetCKKSInfos,
        A: GLWEToBackendRef<BE> + CKKSCtBounds,
        P: GLWEToBackendRef<BE> + IntPolyInfos + CKKSCtBounds,
    {
        ckks_mul_add_pt_consts_into_ordered(module, dst, terms, plans, coeffs, scratch)
    }

    /// Dependency-frontier batch of `ckks_mul_into`.
    ///
    /// The `i`th prepared bound is the authoritative key/use pairing for the
    /// `i`th item. The default plans every item, then issues one core tensor
    /// batch. An override may fuse compatible lanes or split incompatible ones,
    /// but must preserve that alignment and validate the entire frontier before
    /// its first destination write. It must also reproduce each item's
    /// parameters, tensor layout and stamp.
    fn ckks_mul_into_batch_impl<Dst, A, B>(
        module: &Module<BE>,
        items: &mut [CKKSMulIntoItem<&mut Dst, &A, &B>],
        bounds: &[GLWETensorKeyPreparedBound<'_, BE>],
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        Module<BE>: GLWETensoring<BE>,
        Dst: GLWEToBackendMut<BE> + CKKSInfos + SetCKKSInfos + GLWEInfos,
        A: GLWEToBackendRef<BE> + CKKSInfos + GLWEInfos,
        B: GLWEToBackendRef<BE> + CKKSInfos + GLWEInfos,
    {
        ckks_mul_into_batch_ordered(module, items, bounds, scratch)
    }

    fn ckks_mul_into_batch_tmp_bytes_impl<Dst, A, B>(
        module: &Module<BE>,
        items: &[CKKSMulIntoItem<&Dst, &A, &B>],
        uses: &[GGLWEActiveUse],
    ) -> usize
    where
        Module<BE>: GLWETensoring<BE>,
        Dst: GLWEInfos,
        A: GLWEInfos,
        B: GLWEInfos,
    {
        ckks_mul_into_batch_tmp_bytes_ordered(module, items, uses)
    }

    /// Dependency-frontier batch of `ckks_square_into`. Provided.
    fn ckks_square_into_batch_impl<Dst, A>(
        module: &Module<BE>,
        items: &mut [CKKSSquareIntoItem<&mut Dst, &A>],
        bounds: &[GLWETensorKeyPreparedBound<'_, BE>],
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        Module<BE>: GLWETensoring<BE>,
        Dst: GLWEToBackendMut<BE> + CKKSInfos + SetCKKSInfos + GLWEInfos,
        A: GLWEToBackendRef<BE> + CKKSInfos + GLWEInfos,
    {
        ckks_square_into_batch_ordered(module, items, bounds, scratch)
    }

    fn ckks_square_into_batch_tmp_bytes_impl<Dst, A>(
        module: &Module<BE>,
        items: &[CKKSSquareIntoItem<&Dst, &A>],
        uses: &[GGLWEActiveUse],
    ) -> usize
    where
        Module<BE>: GLWETensoring<BE>,
        Dst: GLWEInfos,
        A: GLWEInfos,
    {
        ckks_square_into_batch_tmp_bytes_ordered(module, items, uses)
    }

    /// Dependency-frontier batch of `ckks_square_assign`. Provided.
    fn ckks_square_assign_batch_impl<Dst>(
        module: &Module<BE>,
        items: &mut [CKKSSquareAssignItem<&mut Dst>],
        bounds: &[GLWETensorKeyPreparedBound<'_, BE>],
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        Module<BE>: GLWETensoring<BE>,
        Dst: GLWEToBackendMut<BE> + GLWEToBackendRef<BE> + CKKSInfos + SetCKKSInfos + GLWEInfos,
    {
        ckks_square_assign_batch_ordered(module, items, bounds, scratch)
    }

    fn ckks_square_assign_batch_tmp_bytes_impl<Dst>(
        module: &Module<BE>,
        items: &[CKKSSquareAssignItem<&Dst>],
        uses: &[GGLWEActiveUse],
    ) -> usize
    where
        Module<BE>: GLWETensoring<BE>,
        Dst: GLWEInfos,
    {
        ckks_square_assign_batch_tmp_bytes_ordered(module, items, uses)
    }

    /// Dependency-frontier batch of `ckks_mul_prepared_assign`. Provided.
    fn ckks_mul_prepared_assign_batch_impl<Dst>(
        module: &Module<BE>,
        items: &mut [CKKSPreparedMulAssignItem<&mut Dst, &CKKSPreparedRight<BE>>],
        bounds: &[GLWETensorKeyPreparedBound<'_, BE>],
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        Module<BE>: GLWETensoring<BE>,
        Dst: GLWEToBackendMut<BE> + GLWEToBackendRef<BE> + CKKSInfos + SetCKKSInfos + GLWEInfos,
    {
        ckks_mul_prepared_assign_batch_ordered(module, items, bounds, scratch)
    }

    fn ckks_mul_prepared_assign_batch_tmp_bytes_impl<Dst, PR>(
        module: &Module<BE>,
        items: &[CKKSPreparedMulAssignItem<&Dst, &PR>],
        uses: &[GGLWEActiveUse],
    ) -> usize
    where
        Module<BE>: GLWETensoring<BE>,
        Dst: GLWEInfos,
        PR: CKKSPreparedRightInfos,
    {
        ckks_mul_prepared_assign_batch_tmp_bytes_ordered(module, items, uses)
    }
}

unsafe impl<BE: Backend> CKKSMulImpl<BE> for BE
where
    BE: poulpy_hal::oep::HalVecZnxImpl<BE>,
    Module<BE>: crate::default::mul::CKKSMulDefault<BE>
        + GLWEAdd<BE>
        + GLWECopy<BE>
        + GLWEMulConst<BE>
        + GLWEMulPlain<BE>
        + GLWERotate<BE>
        + GLWETensoring<BE>
        + GiantStepTensorBounds<BE>
        + CnvPVecAlloc<BE>
        + ModuleCoreAlloc<OwnedBuf = BE::OwnedBuf, ZnxWord = BE::ZnxWord>
        + VecZnxCopyBackend<BE>,
{
    fn ckks_mul_tmp_bytes_impl<R: GLWEInfos, A: GLWEInfos, B: GLWEInfos, T: GGLWEInfos>(
        module: &Module<BE>,
        res: &R,
        a: &A,
        b: &B,
        tsk: &T,
    ) -> usize {
        module.ckks_mul_tmp_bytes_default(res, a, b, tsk)
    }

    fn ckks_square_tmp_bytes_impl<R: GLWEInfos, A: GLWEInfos, T: GGLWEInfos>(
        module: &Module<BE>,
        res: &R,
        a: &A,
        tsk: &T,
    ) -> usize {
        module.ckks_square_tmp_bytes_default(res, a, tsk)
    }

    fn ckks_mul_pt_vec_tmp_bytes_impl<R: GLWEInfos, A: GLWEInfos>(
        module: &Module<BE>,
        res: &R,
        a: &A,
        b_k: TorusPrecision,
    ) -> usize {
        module.ckks_mul_pt_vec_tmp_bytes_default(res, a, b_k)
    }

    fn ckks_mul_pt_const_tmp_bytes_impl<R: GLWEInfos, A: GLWEInfos>(
        module: &Module<BE>,
        res: &R,
        a: &A,
        b_k: TorusPrecision,
    ) -> usize {
        module.ckks_mul_pt_const_tmp_bytes_default(res, a, b_k)
    }

    fn ckks_mul_into_impl<Dst, A, B, T>(
        module: &Module<BE>,
        dst: &mut Dst,
        a: &A,
        b: &B,
        tsk: &T,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        Dst: GLWEToBackendMut<BE> + CKKSInfos + SetCKKSInfos + GLWEInfos,
        A: GLWEToBackendRef<BE> + CKKSInfos + GLWEInfos,
        B: GLWEToBackendRef<BE> + CKKSInfos + GLWEInfos,
        T: GLWETensorKeyPreparedToBackendRef<BE> + GGLWEPreparedToBackendRef<BE> + GGLWEInfos,
    {
        module.ckks_mul_into_default(dst, a, b, tsk, scratch)
    }

    fn ckks_mul_assign_impl<Dst, A, T>(
        module: &Module<BE>,
        dst: &mut Dst,
        a: &A,
        tsk: &T,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        Dst: GLWEToBackendMut<BE> + GLWEToBackendRef<BE> + CKKSInfos + SetCKKSInfos + GLWEInfos,
        A: GLWEToBackendRef<BE> + CKKSInfos + GLWEInfos,
        T: GLWETensorKeyPreparedToBackendRef<BE> + GGLWEPreparedToBackendRef<BE> + GGLWEInfos,
    {
        module.ckks_mul_assign_default(dst, a, tsk, scratch)
    }

    fn ckks_prepare_right_impl<A>(module: &Module<BE>, a: &A, scratch: &mut ScratchArena<'_, BE>) -> Result<CKKSPreparedRight<BE>>
    where
        A: GLWEToBackendRef<BE> + CKKSInfos + GLWEInfos,
    {
        module.ckks_prepare_right_default(a, scratch)
    }

    fn ckks_mul_prepared_assign_impl<Dst, T>(
        module: &Module<BE>,
        dst: &mut Dst,
        prepared: &CKKSPreparedRight<BE>,
        tsk: &T,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        Dst: GLWEToBackendMut<BE> + GLWEToBackendRef<BE> + CKKSInfos + SetCKKSInfos + GLWEInfos,
        T: GLWETensorKeyPreparedToBackendRef<BE> + GGLWEPreparedToBackendRef<BE> + GGLWEInfos,
    {
        module.ckks_mul_prepared_assign_default(dst, prepared, tsk, scratch)
    }

    fn ckks_square_into_impl<Dst, A, T>(
        module: &Module<BE>,
        dst: &mut Dst,
        a: &A,
        tsk: &T,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        Dst: GLWEToBackendMut<BE> + CKKSInfos + SetCKKSInfos + GLWEInfos,
        A: GLWEToBackendRef<BE> + CKKSInfos + GLWEInfos,
        T: GLWETensorKeyPreparedToBackendRef<BE> + GGLWEPreparedToBackendRef<BE> + GGLWEInfos,
    {
        module.ckks_square_into_default(dst, a, tsk, scratch)
    }

    fn ckks_square_assign_impl<Dst, T>(
        module: &Module<BE>,
        dst: &mut Dst,
        tsk: &T,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        Dst: GLWEToBackendMut<BE> + GLWEToBackendRef<BE> + CKKSInfos + SetCKKSInfos + GLWEInfos,
        T: GLWETensorKeyPreparedToBackendRef<BE> + GGLWEPreparedToBackendRef<BE> + GGLWEInfos,
    {
        module.ckks_square_assign_default(dst, tsk, scratch)
    }

    fn ckks_mul_pt_vec_into_impl<Dst, A, P>(
        module: &Module<BE>,
        dst: &mut Dst,
        a: &A,
        pt: &P,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        Dst: GLWEToBackendMut<BE> + CKKSInfos + SetCKKSInfos + GLWEInfos,
        A: GLWEToBackendRef<BE> + CKKSInfos + GLWEInfos,
        P: GLWEToBackendRef<BE> + LWEInfos + IntPolyInfos + CKKSCtBounds,
    {
        module.ckks_mul_pt_vec_into_default(dst, a, pt, scratch)
    }

    fn ckks_mul_pt_vec_assign_impl<Dst, P>(
        module: &Module<BE>,
        dst: &mut Dst,
        pt: &P,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        Dst: GLWEToBackendMut<BE> + CKKSInfos + SetCKKSInfos + GLWEInfos,
        P: GLWEToBackendRef<BE> + LWEInfos + IntPolyInfos + CKKSCtBounds,
    {
        module.ckks_mul_pt_vec_assign_default(dst, pt, scratch)
    }

    fn ckks_mul_pt_const_into_impl<Dst, A, P>(
        module: &Module<BE>,
        dst: &mut Dst,
        a: &A,
        pt: &P,
        pt_coeff: usize,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        Dst: GLWEToBackendMut<BE> + CKKSInfos + SetCKKSInfos + GLWEInfos,
        A: GLWEToBackendRef<BE> + CKKSInfos + GLWEInfos,
        P: GLWEToBackendRef<BE> + LWEInfos + IntPolyInfos + CKKSCtBounds,
    {
        module.ckks_mul_pt_const_into_default(dst, a, pt, pt_coeff, scratch)
    }

    fn ckks_mul_pt_const_assign_impl<Dst, P>(
        module: &Module<BE>,
        dst: &mut Dst,
        pt: &P,
        pt_coeff: usize,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        Dst: GLWEToBackendMut<BE> + GLWEToBackendRef<BE> + CKKSInfos + SetCKKSInfos + GLWEInfos,
        P: GLWEToBackendRef<BE> + LWEInfos + IntPolyInfos + CKKSCtBounds,
    {
        module.ckks_mul_pt_const_assign_default(dst, pt, pt_coeff, scratch)
    }

    #[allow(clippy::too_many_arguments)]
    fn ckks_mul_add_pt_consts_into_impl<Dst, A, P>(
        module: &Module<BE>,
        dst: &mut Dst,
        terms: &[(&A, usize)],
        plans: &[CKKSMulAddPtConstPlan],
        coeffs: &P,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        Module<BE>: CKKSMulOps<BE> + CKKSAddOps<BE>,
        Dst: GLWEToBackendMut<BE> + CKKSCtBounds + SetCKKSInfos,
        A: GLWEToBackendRef<BE> + CKKSCtBounds,
        P: GLWEToBackendRef<BE> + IntPolyInfos + CKKSCtBounds,
    {
        module.ckks_mul_add_pt_consts_into_default(dst, terms, plans, coeffs, scratch)
    }

    fn ckks_mul_into_batch_impl<Dst, A, B>(
        module: &Module<BE>,
        items: &mut [CKKSMulIntoItem<&mut Dst, &A, &B>],
        bounds: &[GLWETensorKeyPreparedBound<'_, BE>],
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        Module<BE>: GLWETensoring<BE>,
        Dst: GLWEToBackendMut<BE> + CKKSInfos + SetCKKSInfos + GLWEInfos,
        A: GLWEToBackendRef<BE> + CKKSInfos + GLWEInfos,
        B: GLWEToBackendRef<BE> + CKKSInfos + GLWEInfos,
    {
        module.ckks_mul_into_batch_default(items, bounds, scratch)
    }

    fn ckks_mul_into_batch_tmp_bytes_impl<Dst, A, B>(
        module: &Module<BE>,
        items: &[CKKSMulIntoItem<&Dst, &A, &B>],
        uses: &[GGLWEActiveUse],
    ) -> usize
    where
        Module<BE>: GLWETensoring<BE>,
        Dst: GLWEInfos,
        A: GLWEInfos,
        B: GLWEInfos,
    {
        module.ckks_mul_into_batch_tmp_bytes_default(items, uses)
    }

    fn ckks_square_into_batch_impl<Dst, A>(
        module: &Module<BE>,
        items: &mut [CKKSSquareIntoItem<&mut Dst, &A>],
        bounds: &[GLWETensorKeyPreparedBound<'_, BE>],
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        Module<BE>: GLWETensoring<BE>,
        Dst: GLWEToBackendMut<BE> + CKKSInfos + SetCKKSInfos + GLWEInfos,
        A: GLWEToBackendRef<BE> + CKKSInfos + GLWEInfos,
    {
        module.ckks_square_into_batch_default(items, bounds, scratch)
    }

    fn ckks_square_into_batch_tmp_bytes_impl<Dst, A>(
        module: &Module<BE>,
        items: &[CKKSSquareIntoItem<&Dst, &A>],
        uses: &[GGLWEActiveUse],
    ) -> usize
    where
        Module<BE>: GLWETensoring<BE>,
        Dst: GLWEInfos,
        A: GLWEInfos,
    {
        module.ckks_square_into_batch_tmp_bytes_default(items, uses)
    }

    fn ckks_square_assign_batch_impl<Dst>(
        module: &Module<BE>,
        items: &mut [CKKSSquareAssignItem<&mut Dst>],
        bounds: &[GLWETensorKeyPreparedBound<'_, BE>],
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        Module<BE>: GLWETensoring<BE>,
        Dst: GLWEToBackendMut<BE> + GLWEToBackendRef<BE> + CKKSInfos + SetCKKSInfos + GLWEInfos,
    {
        module.ckks_square_assign_batch_default(items, bounds, scratch)
    }

    fn ckks_square_assign_batch_tmp_bytes_impl<Dst>(
        module: &Module<BE>,
        items: &[CKKSSquareAssignItem<&Dst>],
        uses: &[GGLWEActiveUse],
    ) -> usize
    where
        Module<BE>: GLWETensoring<BE>,
        Dst: GLWEInfos,
    {
        module.ckks_square_assign_batch_tmp_bytes_default(items, uses)
    }

    fn ckks_mul_prepared_assign_batch_impl<Dst>(
        module: &Module<BE>,
        items: &mut [CKKSPreparedMulAssignItem<&mut Dst, &CKKSPreparedRight<BE>>],
        bounds: &[GLWETensorKeyPreparedBound<'_, BE>],
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        Module<BE>: GLWETensoring<BE>,
        Dst: GLWEToBackendMut<BE> + GLWEToBackendRef<BE> + CKKSInfos + SetCKKSInfos + GLWEInfos,
    {
        module.ckks_mul_prepared_assign_batch_default(items, bounds, scratch)
    }

    fn ckks_mul_prepared_assign_batch_tmp_bytes_impl<Dst, PR>(
        module: &Module<BE>,
        items: &[CKKSPreparedMulAssignItem<&Dst, &PR>],
        uses: &[GGLWEActiveUse],
    ) -> usize
    where
        Module<BE>: GLWETensoring<BE>,
        Dst: GLWEInfos,
        PR: CKKSPreparedRightInfos,
    {
        module.ckks_mul_prepared_assign_batch_tmp_bytes_default(items, uses)
    }
}

#[macro_export]
macro_rules! impl_ckks_mul_defaults {
    ($be:ty) => {
        impl $crate::default::mul::CKKSMulDefault<$be> for ::poulpy_hal::layouts::Module<$be> {}
    };
}
pub use crate::impl_ckks_mul_defaults;
