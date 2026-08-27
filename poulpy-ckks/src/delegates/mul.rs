use crate::CKKSResult as Result;
use poulpy_core::layouts::IntPolyInfos;
use poulpy_core::{
    GLWEAdd, GLWECopy, GLWEMulConst, GLWEMulPlain, GLWERotate, GLWETensoring,
    layouts::{
        GGLWEInfos, GLWERelinearizationKeyHelper, GLWERelinearizationKeyLayoutHelper, GLWEToBackendMut, GLWEToBackendRef,
        LWEInfos, ModuleCoreAlloc, TorusPrecision, WithEffectiveDsize,
        prepared::{GGLWEPreparedToBackendRef, GLWETensorKeyPreparedToBackendRef},
    },
};
use poulpy_hal::{
    api::{ModuleN, VecZnxCopyBackend},
    layouts::{Backend, Module, ScratchArena},
};

use crate::api::{CKKSMulIntoItem, CKKSMulOps, CKKSPreparedMulAssignItem, CKKSSquareAssignItem, CKKSSquareIntoItem};

use crate::{
    CKKSCompositionError, CKKSCtBounds, CKKSInfos, SetCKKSInfos,
    layouts::{CKKSPreparedRight, CKKSPreparedRightInfos},
    oep::CKKSMulImpl,
};

/// A batch shares one effective decomposition, resolved at the widest item, so
/// a frontier issued as one call cannot straddle two of them.
fn resolve<'a, H>(keys: &'a H, k: TorusPrecision, op: &'static str) -> Result<(&'a H::Key, poulpy_core::layouts::Dsize)>
where
    H: GLWERelinearizationKeyHelper,
{
    keys.get_relinearization_key_for(k)
        .map_err(|_| CKKSCompositionError::MissingRelinearizationKey { op, k: k.into() }.into())
}

impl<BE: Backend + CKKSMulImpl<BE>> CKKSMulOps<BE> for Module<BE>
where
    Module<BE>: GLWEAdd<BE>
        + GLWECopy<BE>
        + GLWEMulConst<BE>
        + GLWEMulPlain<BE>
        + GLWERotate<BE>
        + GLWETensoring<BE>
        + ModuleN
        + ModuleCoreAlloc<OwnedBuf = BE::OwnedBuf, ZnxWord = BE::ZnxWord>
        + VecZnxCopyBackend<BE>,
{
    fn ckks_mul_tmp_bytes<R, A, B, H>(&self, res: &R, a: &A, b: &B, tsk: &H) -> usize
    where
        R: CKKSCtBounds,
        A: CKKSCtBounds,
        B: CKKSCtBounds,
        H: GLWERelinearizationKeyLayoutHelper,
    {
        let (tsk, dsize) = tsk
            .get_relinearization_key_layout_for(a.k().max(b.k()))
            .unwrap_or_else(|e| panic!("{e}"));
        BE::ckks_mul_tmp_bytes_impl(self, res, a, b, &tsk.with_dsize(dsize))
    }

    fn ckks_square_tmp_bytes<R, A, H>(&self, res: &R, a: &A, tsk: &H) -> usize
    where
        R: CKKSCtBounds,
        A: CKKSCtBounds,
        H: GLWERelinearizationKeyLayoutHelper,
    {
        let (tsk, dsize) = tsk
            .get_relinearization_key_layout_for(a.k())
            .unwrap_or_else(|e| panic!("{e}"));
        BE::ckks_square_tmp_bytes_impl(self, res, a, &tsk.with_dsize(dsize))
    }

    fn ckks_mul_pt_vec_tmp_bytes<R, A, P>(&self, res: &R, a: &A, b: &P) -> usize
    where
        R: CKKSCtBounds,
        A: CKKSCtBounds,
        P: CKKSInfos,
    {
        BE::ckks_mul_pt_vec_tmp_bytes_impl(self, res, a, b.k())
    }

    fn ckks_mul_pt_const_tmp_bytes<R, A, P>(&self, res: &R, a: &A, b: &P) -> usize
    where
        R: CKKSCtBounds,
        A: CKKSCtBounds,
        P: CKKSInfos,
    {
        BE::ckks_mul_pt_const_tmp_bytes_impl(self, res, a, b.k())
    }

    fn ckks_mul_into<Dst, A, B, H>(&self, dst: &mut Dst, a: &A, b: &B, tsk: &H, scratch: &mut ScratchArena<'_, BE>) -> Result<()>
    where
        Dst: GLWEToBackendMut<BE> + CKKSCtBounds + SetCKKSInfos,
        A: GLWEToBackendRef<BE> + CKKSCtBounds,
        B: GLWEToBackendRef<BE> + CKKSCtBounds,
        H: GLWERelinearizationKeyHelper,
        H::Key: GGLWEInfos + GLWETensorKeyPreparedToBackendRef<BE> + GGLWEPreparedToBackendRef<BE>,
    {
        let (tsk, dsize) =
            tsk.get_relinearization_key_for(a.k().max(b.k()))
                .map_err(|_| CKKSCompositionError::MissingRelinearizationKey {
                    op: "ckks_mul_into",
                    k: a.k().max(b.k()).into(),
                })?;
        BE::ckks_mul_into_impl(self, dst, a, b, &tsk.with_dsize(dsize), scratch)
    }

    fn ckks_mul_assign<Dst, A, H>(&self, dst: &mut Dst, a: &A, tsk: &H, scratch: &mut ScratchArena<'_, BE>) -> Result<()>
    where
        Dst: GLWEToBackendMut<BE> + GLWEToBackendRef<BE> + CKKSCtBounds + SetCKKSInfos,
        A: GLWEToBackendRef<BE> + CKKSCtBounds,
        H: GLWERelinearizationKeyHelper,
        H::Key: GGLWEInfos + GLWETensorKeyPreparedToBackendRef<BE> + GGLWEPreparedToBackendRef<BE>,
    {
        let (tsk, dsize) =
            tsk.get_relinearization_key_for(dst.k().max(a.k()))
                .map_err(|_| CKKSCompositionError::MissingRelinearizationKey {
                    op: "ckks_mul_assign",
                    k: dst.k().max(a.k()).into(),
                })?;
        BE::ckks_mul_assign_impl(self, dst, a, &tsk.with_dsize(dsize), scratch)
    }

    fn ckks_prepare_right<A>(&self, a: &A, scratch: &mut ScratchArena<'_, BE>) -> Result<CKKSPreparedRight<BE>>
    where
        A: GLWEToBackendRef<BE> + CKKSCtBounds,
    {
        BE::ckks_prepare_right_impl(self, a, scratch)
    }

    fn ckks_mul_prepared_assign<Dst, H>(
        &self,
        dst: &mut Dst,
        prepared: &CKKSPreparedRight<BE>,
        tsk: &H,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        Dst: GLWEToBackendMut<BE> + GLWEToBackendRef<BE> + CKKSCtBounds + SetCKKSInfos,
        H: GLWERelinearizationKeyHelper,
        H::Key: GGLWEInfos + GLWETensorKeyPreparedToBackendRef<BE> + GGLWEPreparedToBackendRef<BE>,
    {
        let (tsk, dsize) = tsk
            .get_relinearization_key_for(dst.k().max(TorusPrecision(prepared.k as u32)))
            .map_err(|_| CKKSCompositionError::MissingRelinearizationKey {
                op: "ckks_mul_prepared_assign",
                k: dst.k().max(TorusPrecision(prepared.k as u32)).into(),
            })?;
        BE::ckks_mul_prepared_assign_impl(self, dst, prepared, &tsk.with_dsize(dsize), scratch)
    }

    fn ckks_square_into<Dst, A, H>(&self, dst: &mut Dst, a: &A, tsk: &H, scratch: &mut ScratchArena<'_, BE>) -> Result<()>
    where
        Dst: GLWEToBackendMut<BE> + CKKSCtBounds + SetCKKSInfos,
        A: GLWEToBackendRef<BE> + CKKSCtBounds,
        H: GLWERelinearizationKeyHelper,
        H::Key: GGLWEInfos + GLWETensorKeyPreparedToBackendRef<BE> + GGLWEPreparedToBackendRef<BE>,
    {
        let (tsk, dsize) =
            tsk.get_relinearization_key_for(a.k())
                .map_err(|_| CKKSCompositionError::MissingRelinearizationKey {
                    op: "ckks_square_into",
                    k: a.k().into(),
                })?;
        BE::ckks_square_into_impl(self, dst, a, &tsk.with_dsize(dsize), scratch)
    }

    fn ckks_square_assign<Dst, H>(&self, dst: &mut Dst, tsk: &H, scratch: &mut ScratchArena<'_, BE>) -> Result<()>
    where
        Dst: GLWEToBackendMut<BE> + GLWEToBackendRef<BE> + CKKSCtBounds + SetCKKSInfos,
        H: GLWERelinearizationKeyHelper,
        H::Key: GGLWEInfos + GLWETensorKeyPreparedToBackendRef<BE> + GGLWEPreparedToBackendRef<BE>,
    {
        let (tsk, dsize) =
            tsk.get_relinearization_key_for(dst.k())
                .map_err(|_| CKKSCompositionError::MissingRelinearizationKey {
                    op: "ckks_square_assign",
                    k: dst.k().into(),
                })?;
        BE::ckks_square_assign_impl(self, dst, &tsk.with_dsize(dsize), scratch)
    }

    fn ckks_mul_pt_vec_into<Dst, A, P>(&self, dst: &mut Dst, a: &A, pt: &P, scratch: &mut ScratchArena<'_, BE>) -> Result<()>
    where
        Dst: GLWEToBackendMut<BE> + CKKSCtBounds + SetCKKSInfos,
        A: GLWEToBackendRef<BE> + CKKSCtBounds,
        P: GLWEToBackendRef<BE> + IntPolyInfos + CKKSCtBounds,
    {
        BE::ckks_mul_pt_vec_into_impl(self, dst, a, pt, scratch)
    }

    fn ckks_mul_pt_vec_assign<Dst, P>(&self, dst: &mut Dst, pt: &P, scratch: &mut ScratchArena<'_, BE>) -> Result<()>
    where
        Dst: GLWEToBackendMut<BE> + GLWEToBackendRef<BE> + CKKSCtBounds + SetCKKSInfos,
        P: GLWEToBackendRef<BE> + IntPolyInfos + CKKSCtBounds,
    {
        BE::ckks_mul_pt_vec_assign_impl(self, dst, pt, scratch)
    }

    fn ckks_mul_pt_const_into<Dst, A, P>(
        &self,
        dst: &mut Dst,
        a: &A,
        pt: &P,
        pt_coeff: usize,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        Dst: GLWEToBackendMut<BE> + CKKSCtBounds + SetCKKSInfos,
        A: GLWEToBackendRef<BE> + CKKSCtBounds,
        P: GLWEToBackendRef<BE> + IntPolyInfos + CKKSCtBounds,
    {
        BE::ckks_mul_pt_const_into_impl(self, dst, a, pt, pt_coeff, scratch)
    }

    fn ckks_mul_pt_const_assign<Dst, P>(
        &self,
        dst: &mut Dst,
        pt: &P,
        pt_coeff: usize,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        Dst: GLWEToBackendMut<BE> + GLWEToBackendRef<BE> + CKKSCtBounds + SetCKKSInfos,
        P: GLWEToBackendRef<BE> + IntPolyInfos + CKKSCtBounds,
    {
        BE::ckks_mul_pt_const_assign_impl(self, dst, pt, pt_coeff, scratch)
    }
    fn ckks_mul_into_batch_tmp_bytes<Dst, A, B, H>(&self, items: &[CKKSMulIntoItem<&Dst, &A, &B>], tsk: &H) -> usize
    where
        Dst: CKKSCtBounds,
        A: CKKSCtBounds,
        B: CKKSCtBounds,
        H: GLWERelinearizationKeyLayoutHelper,
    {
        let k = items.iter().fold(TorusPrecision(0), |k, i| k.max(i.a.k().max(i.b.k())));
        let (tsk, dsize) = tsk.get_relinearization_key_layout_for(k).unwrap_or_else(|e| panic!("{e}"));
        BE::ckks_mul_into_batch_tmp_bytes_impl(self, items, &tsk.with_dsize(dsize))
    }

    fn ckks_mul_into_batch<Dst, A, B, H>(
        &self,
        items: &mut [CKKSMulIntoItem<&mut Dst, &A, &B>],
        tsk: &H,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        Dst: GLWEToBackendMut<BE> + CKKSCtBounds + SetCKKSInfos,
        A: GLWEToBackendRef<BE> + CKKSCtBounds,
        B: GLWEToBackendRef<BE> + CKKSCtBounds,
        H: GLWERelinearizationKeyHelper,
        H::Key: GGLWEInfos + GLWETensorKeyPreparedToBackendRef<BE> + GGLWEPreparedToBackendRef<BE>,
    {
        let k = items.iter().fold(TorusPrecision(0), |k, i| k.max(i.a.k().max(i.b.k())));
        let (tsk, dsize) = resolve(tsk, k, "ckks_mul_into_batch")?;
        BE::ckks_mul_into_batch_impl(self, items, &tsk.with_dsize(dsize), scratch)
    }

    fn ckks_square_into_batch_tmp_bytes<Dst, A, H>(&self, items: &[CKKSSquareIntoItem<&Dst, &A>], tsk: &H) -> usize
    where
        Dst: CKKSCtBounds,
        A: CKKSCtBounds,
        H: GLWERelinearizationKeyLayoutHelper,
    {
        let k = items.iter().fold(TorusPrecision(0), |k, i| k.max(i.a.k()));
        let (tsk, dsize) = tsk.get_relinearization_key_layout_for(k).unwrap_or_else(|e| panic!("{e}"));
        BE::ckks_square_into_batch_tmp_bytes_impl(self, items, &tsk.with_dsize(dsize))
    }

    fn ckks_square_into_batch<Dst, A, H>(
        &self,
        items: &mut [CKKSSquareIntoItem<&mut Dst, &A>],
        tsk: &H,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        Dst: GLWEToBackendMut<BE> + CKKSCtBounds + SetCKKSInfos,
        A: GLWEToBackendRef<BE> + CKKSCtBounds,
        H: GLWERelinearizationKeyHelper,
        H::Key: GGLWEInfos + GLWETensorKeyPreparedToBackendRef<BE> + GGLWEPreparedToBackendRef<BE>,
    {
        let k = items.iter().fold(TorusPrecision(0), |k, i| k.max(i.a.k()));
        let (tsk, dsize) = resolve(tsk, k, "ckks_square_into_batch")?;
        BE::ckks_square_into_batch_impl(self, items, &tsk.with_dsize(dsize), scratch)
    }

    fn ckks_square_assign_batch_tmp_bytes<Dst, H>(&self, items: &[CKKSSquareAssignItem<&Dst>], tsk: &H) -> usize
    where
        Dst: CKKSCtBounds,
        H: GLWERelinearizationKeyLayoutHelper,
    {
        let k = items.iter().fold(TorusPrecision(0), |k, i| k.max(i.dst.k()));
        let (tsk, dsize) = tsk.get_relinearization_key_layout_for(k).unwrap_or_else(|e| panic!("{e}"));
        BE::ckks_square_assign_batch_tmp_bytes_impl(self, items, &tsk.with_dsize(dsize))
    }

    fn ckks_square_assign_batch<Dst, H>(
        &self,
        items: &mut [CKKSSquareAssignItem<&mut Dst>],
        tsk: &H,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        Dst: GLWEToBackendMut<BE> + GLWEToBackendRef<BE> + CKKSCtBounds + SetCKKSInfos,
        H: GLWERelinearizationKeyHelper,
        H::Key: GGLWEInfos + GLWETensorKeyPreparedToBackendRef<BE> + GGLWEPreparedToBackendRef<BE>,
    {
        let k = items.iter().fold(TorusPrecision(0), |k, i| k.max(i.dst.k()));
        let (tsk, dsize) = resolve(tsk, k, "ckks_square_assign_batch")?;
        BE::ckks_square_assign_batch_impl(self, items, &tsk.with_dsize(dsize), scratch)
    }

    fn ckks_mul_prepared_assign_batch_tmp_bytes<Dst, PR, H>(
        &self,
        items: &[CKKSPreparedMulAssignItem<&Dst, &PR>],
        tsk: &H,
    ) -> usize
    where
        Dst: CKKSCtBounds,
        PR: CKKSPreparedRightInfos,
        H: GLWERelinearizationKeyLayoutHelper,
    {
        let k = items.iter().fold(TorusPrecision(0), |k, i| {
            k.max(i.dst.k().max(TorusPrecision(i.prepared.prepared_k() as u32)))
        });
        let (tsk, dsize) = tsk.get_relinearization_key_layout_for(k).unwrap_or_else(|e| panic!("{e}"));
        BE::ckks_mul_prepared_assign_batch_tmp_bytes_impl(self, items, &tsk.with_dsize(dsize))
    }

    fn ckks_mul_prepared_assign_batch<Dst, H>(
        &self,
        items: &mut [CKKSPreparedMulAssignItem<&mut Dst, &CKKSPreparedRight<BE>>],
        tsk: &H,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        Dst: GLWEToBackendMut<BE> + GLWEToBackendRef<BE> + CKKSCtBounds + SetCKKSInfos,
        H: GLWERelinearizationKeyHelper,
        H::Key: GGLWEInfos + GLWETensorKeyPreparedToBackendRef<BE> + GGLWEPreparedToBackendRef<BE>,
    {
        let k = items.iter().fold(TorusPrecision(0), |k, i| {
            k.max(i.dst.k().max(TorusPrecision(i.prepared.k as u32)))
        });
        let (tsk, dsize) = resolve(tsk, k, "ckks_mul_prepared_assign_batch")?;
        BE::ckks_mul_prepared_assign_batch_impl(self, items, &tsk.with_dsize(dsize), scratch)
    }
}
