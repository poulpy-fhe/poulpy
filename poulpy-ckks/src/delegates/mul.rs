use crate::CKKSResult as Result;
use poulpy_core::layouts::IntPolyInfos;
use poulpy_core::{
    GLWEAdd, GLWECopy, GLWEMulConst, GLWEMulPlain, GLWERotate, GLWETensoring,
    layouts::{
        GGLWEBind, GGLWEInfos, GGLWEUse, GLWERelinearizationKeyHelper, GLWERelinearizationKeyLayoutHelper, GLWEToBackendMut,
        GLWEToBackendRef, LWEInfos, ModuleCoreAlloc, TorusPrecision, WithEffectiveDsize,
        prepared::{GGLWEPreparedToBackendRef, GLWETensorKeyPreparedBound, GLWETensorKeyPreparedToBackendRef},
    },
};
use poulpy_hal::{
    api::{ModuleN, VecZnxCopyBackend},
    layouts::{Backend, Module, ScratchArena},
};

use crate::api::{CKKSMulIntoItem, CKKSMulOps, CKKSPreparedMulAssignItem, CKKSSquareAssignItem, CKKSSquareIntoItem};

use crate::default::mul::{get_mul_ct_params, get_mul_prepared_params};
use crate::{
    CKKSCompositionError, CKKSCtBounds, CKKSInfos, SetCKKSInfos, ckks_ensure,
    layouts::{CKKSPreparedRight, CKKSPreparedRightInfos},
    oep::CKKSMulImpl,
};

fn resolve<'a, H>(keys: &'a H, k: TorusPrecision, op: &'static str) -> Result<(&'a H::Key, poulpy_core::layouts::Dsize)>
where
    H: GLWERelinearizationKeyHelper,
{
    keys.get_relinearization_key_for(k)
        .map_err(|_| CKKSCompositionError::MissingRelinearizationKey { op, k: k.into() }.into())
}

fn core_error(error: poulpy_core::CoreError) -> crate::CKKSError {
    crate::CKKSError::Internal(anyhow::Error::new(error))
}

/// The precision each ciphertext-ciphertext operation resolves its key at.
///
/// One definition per operation, used by the scratch query and by the execution
/// alike, so the two cannot drift apart. The assign forms read their
/// destination as an operand, which is why it enters here and not only there.
fn mul_k<A: CKKSCtBounds, B: CKKSCtBounds>(a: &A, b: &B) -> TorusPrecision {
    a.k().max(b.k())
}

fn square_k<A: CKKSCtBounds>(a: &A) -> TorusPrecision {
    a.k()
}

fn prepared_mul_k_checked<D: CKKSCtBounds>(dst: &D, prepared_k: usize) -> Result<TorusPrecision> {
    let prepared_k = u32::try_from(prepared_k)
        .map_err(|_| crate::CKKSError::Internal(anyhow::anyhow!("prepared precision {prepared_k} exceeds u32")))?;
    Ok(dst.k().max(TorusPrecision(prepared_k)))
}

fn prepared_mul_k<D: CKKSCtBounds>(dst: &D, prepared_k: usize) -> TorusPrecision {
    prepared_mul_k_checked(dst, prepared_k).unwrap_or_else(|e| panic!("{e}"))
}

/// Plans every item of a batch before any of them is dispatched.
///
/// A batch is documented to validate the whole slice before it writes anything.
/// Splitting it into per-key groups would otherwise weaken that to per group: a
/// malformed item in the second group would be caught only after the first had
/// already written its destinations.
fn plan_all<T, F>(items: &[T], plan: F) -> Result<()>
where
    F: Fn(&T) -> Result<()>,
{
    items.iter().try_for_each(plan)
}

fn ensure_batch_scratch<BE: Backend>(scratch: &ScratchArena<'_, BE>, required: usize, op: &str) -> Result<()> {
    ckks_ensure!(
        scratch.available() >= required,
        "scratch.available(): {} < {op}: {required}",
        scratch.available()
    );
    Ok(())
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
        let k = mul_k(a, b);
        let (tsk, dsize) = tsk.get_relinearization_key_layout_for(k).unwrap_or_else(|e| panic!("{e}"));
        let tsk = tsk.with_dsize(dsize);
        tsk.bind_covering_for(k).unwrap_or_else(|e| panic!("{e}"));
        BE::ckks_mul_tmp_bytes_impl(self, res, a, b, &tsk)
    }

    fn ckks_square_tmp_bytes<R, A, H>(&self, res: &R, a: &A, tsk: &H) -> usize
    where
        R: CKKSCtBounds,
        A: CKKSCtBounds,
        H: GLWERelinearizationKeyLayoutHelper,
    {
        let k = square_k(a);
        let (tsk, dsize) = tsk.get_relinearization_key_layout_for(k).unwrap_or_else(|e| panic!("{e}"));
        let tsk = tsk.with_dsize(dsize);
        tsk.bind_covering_for(k).unwrap_or_else(|e| panic!("{e}"));
        BE::ckks_square_tmp_bytes_impl(self, res, a, &tsk)
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
        let k = mul_k(a, b);
        let (tsk, dsize) = tsk
            .get_relinearization_key_for(k)
            .map_err(|_| CKKSCompositionError::MissingRelinearizationKey {
                op: "ckks_mul_into",
                k: k.into(),
            })?;
        let tsk = tsk.with_dsize(dsize);
        tsk.bind_covering_for(k).map_err(core_error)?;
        BE::ckks_mul_into_impl(self, dst, a, b, &tsk, scratch)
    }

    fn ckks_mul_assign<Dst, A, H>(&self, dst: &mut Dst, a: &A, tsk: &H, scratch: &mut ScratchArena<'_, BE>) -> Result<()>
    where
        Dst: GLWEToBackendMut<BE> + GLWEToBackendRef<BE> + CKKSCtBounds + SetCKKSInfos,
        A: GLWEToBackendRef<BE> + CKKSCtBounds,
        H: GLWERelinearizationKeyHelper,
        H::Key: GGLWEInfos + GLWETensorKeyPreparedToBackendRef<BE> + GGLWEPreparedToBackendRef<BE>,
    {
        let k = mul_k(dst, a);
        let (tsk, dsize) = tsk
            .get_relinearization_key_for(k)
            .map_err(|_| CKKSCompositionError::MissingRelinearizationKey {
                op: "ckks_mul_assign",
                k: k.into(),
            })?;
        let tsk = tsk.with_dsize(dsize);
        tsk.bind_covering_for(k).map_err(core_error)?;
        BE::ckks_mul_assign_impl(self, dst, a, &tsk, scratch)
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
        let k = prepared_mul_k_checked(dst, prepared.k)?;
        let (tsk, dsize) = tsk
            .get_relinearization_key_for(k)
            .map_err(|_| CKKSCompositionError::MissingRelinearizationKey {
                op: "ckks_mul_prepared_assign",
                k: k.into(),
            })?;
        let tsk = tsk.with_dsize(dsize);
        tsk.bind_covering_for(k).map_err(core_error)?;
        BE::ckks_mul_prepared_assign_impl(self, dst, prepared, &tsk, scratch)
    }

    fn ckks_square_into<Dst, A, H>(&self, dst: &mut Dst, a: &A, tsk: &H, scratch: &mut ScratchArena<'_, BE>) -> Result<()>
    where
        Dst: GLWEToBackendMut<BE> + CKKSCtBounds + SetCKKSInfos,
        A: GLWEToBackendRef<BE> + CKKSCtBounds,
        H: GLWERelinearizationKeyHelper,
        H::Key: GGLWEInfos + GLWETensorKeyPreparedToBackendRef<BE> + GGLWEPreparedToBackendRef<BE>,
    {
        let k = square_k(a);
        let (tsk, dsize) = tsk
            .get_relinearization_key_for(k)
            .map_err(|_| CKKSCompositionError::MissingRelinearizationKey {
                op: "ckks_square_into",
                k: k.into(),
            })?;
        let tsk = tsk.with_dsize(dsize);
        tsk.bind_covering_for(k).map_err(core_error)?;
        BE::ckks_square_into_impl(self, dst, a, &tsk, scratch)
    }

    fn ckks_square_assign<Dst, H>(&self, dst: &mut Dst, tsk: &H, scratch: &mut ScratchArena<'_, BE>) -> Result<()>
    where
        Dst: GLWEToBackendMut<BE> + GLWEToBackendRef<BE> + CKKSCtBounds + SetCKKSInfos,
        H: GLWERelinearizationKeyHelper,
        H::Key: GGLWEInfos + GLWETensorKeyPreparedToBackendRef<BE> + GGLWEPreparedToBackendRef<BE>,
    {
        let k = square_k(dst);
        let (tsk, dsize) = tsk
            .get_relinearization_key_for(k)
            .map_err(|_| CKKSCompositionError::MissingRelinearizationKey {
                op: "ckks_square_assign",
                k: k.into(),
            })?;
        let tsk = tsk.with_dsize(dsize);
        tsk.bind_covering_for(k).map_err(core_error)?;
        BE::ckks_square_assign_impl(self, dst, &tsk, scratch)
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
        let mut active_items = Vec::with_capacity(items.len());
        let mut uses = Vec::with_capacity(items.len());
        let mut best = 0;
        for item in items {
            let k = mul_k(item.a, item.b);
            let (layout, dsize) = tsk.get_relinearization_key_layout_for(k).unwrap_or_else(|e| panic!("{e}"));
            let layout = layout.with_dsize(dsize);
            match layout.bind_covering_for(k).unwrap_or_else(|e| panic!("{e}")) {
                GGLWEUse::Active(use_) => {
                    active_items.push(CKKSMulIntoItem {
                        dst: item.dst,
                        a: item.a,
                        b: item.b,
                    });
                    uses.push(use_);
                }
                GGLWEUse::Empty => {
                    best = best.max(BE::ckks_mul_tmp_bytes_impl(self, item.dst, item.a, item.b, &layout));
                }
            }
        }
        if !active_items.is_empty() {
            best = best.max(BE::ckks_mul_into_batch_tmp_bytes_impl(self, &active_items, &uses));
        }
        best
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
        const OP: &str = "ckks_mul_into_batch";
        plan_all(items, |item| get_mul_ct_params(&*item.dst, item.a, item.b).map(|_| ()))?;

        let mut active_indices = Vec::with_capacity(items.len());
        let mut bounds = Vec::with_capacity(items.len());
        let mut empty = Vec::new();
        for (index, item) in items.iter().enumerate() {
            let k = mul_k(item.a, item.b);
            let (key, dsize) = resolve(tsk, k, OP)?;
            match key.with_dsize(dsize).bind_covering_for(k).map_err(core_error)? {
                GGLWEUse::Active(use_) => {
                    let bound = GLWETensorKeyPreparedBound::new(GLWETensorKeyPreparedToBackendRef::to_backend_ref(key), use_)
                        .map_err(core_error)?;
                    active_indices.push(index);
                    bounds.push(bound);
                }
                GGLWEUse::Empty => empty.push((index, key, dsize)),
            }
        }

        let uses: Vec<_> = bounds.iter().map(|bound| *bound.use_()).collect();
        let active_query: Vec<_> = active_indices
            .iter()
            .map(|&index| {
                let item = &items[index];
                CKKSMulIntoItem {
                    dst: &*item.dst,
                    a: item.a,
                    b: item.b,
                }
            })
            .collect();
        let mut required = if active_query.is_empty() {
            0
        } else {
            BE::ckks_mul_into_batch_tmp_bytes_impl(self, &active_query, &uses)
        };
        for &(index, key, dsize) in &empty {
            let item = &items[index];
            required = required.max(BE::ckks_mul_tmp_bytes_impl(
                self,
                &*item.dst,
                item.a,
                item.b,
                &key.with_dsize(dsize),
            ));
        }
        ensure_batch_scratch(scratch, required, OP)?;
        drop(active_query);

        if !active_indices.is_empty() {
            let mut pos = 0;
            let mut active_items = Vec::with_capacity(active_indices.len());
            for (index, item) in items.iter_mut().enumerate() {
                if active_indices.get(pos).copied() == Some(index) {
                    active_items.push(CKKSMulIntoItem {
                        dst: &mut *item.dst,
                        a: item.a,
                        b: item.b,
                    });
                    pos += 1;
                }
            }
            BE::ckks_mul_into_batch_impl(self, &mut active_items, &bounds, scratch)?;
        }
        for (index, key, dsize) in empty {
            let item = &mut items[index];
            BE::ckks_mul_into_impl(self, item.dst, item.a, item.b, &key.with_dsize(dsize), scratch)?;
        }
        Ok(())
    }

    fn ckks_square_into_batch_tmp_bytes<Dst, A, H>(&self, items: &[CKKSSquareIntoItem<&Dst, &A>], tsk: &H) -> usize
    where
        Dst: CKKSCtBounds,
        A: CKKSCtBounds,
        H: GLWERelinearizationKeyLayoutHelper,
    {
        let mut active_items = Vec::with_capacity(items.len());
        let mut uses = Vec::with_capacity(items.len());
        let mut best = 0;
        for item in items {
            let k = square_k(item.a);
            let (layout, dsize) = tsk.get_relinearization_key_layout_for(k).unwrap_or_else(|e| panic!("{e}"));
            let layout = layout.with_dsize(dsize);
            match layout.bind_covering_for(k).unwrap_or_else(|e| panic!("{e}")) {
                GGLWEUse::Active(use_) => {
                    active_items.push(CKKSSquareIntoItem {
                        dst: item.dst,
                        a: item.a,
                    });
                    uses.push(use_);
                }
                GGLWEUse::Empty => {
                    best = best.max(BE::ckks_square_tmp_bytes_impl(self, item.dst, item.a, &layout));
                }
            }
        }
        if !active_items.is_empty() {
            best = best.max(BE::ckks_square_into_batch_tmp_bytes_impl(self, &active_items, &uses));
        }
        best
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
        const OP: &str = "ckks_square_into_batch";
        plan_all(items, |item| get_mul_ct_params(&*item.dst, item.a, item.a).map(|_| ()))?;

        let mut active_indices = Vec::with_capacity(items.len());
        let mut bounds = Vec::with_capacity(items.len());
        let mut empty = Vec::new();
        for (index, item) in items.iter().enumerate() {
            let k = square_k(item.a);
            let (key, dsize) = resolve(tsk, k, OP)?;
            match key.with_dsize(dsize).bind_covering_for(k).map_err(core_error)? {
                GGLWEUse::Active(use_) => {
                    let bound = GLWETensorKeyPreparedBound::new(GLWETensorKeyPreparedToBackendRef::to_backend_ref(key), use_)
                        .map_err(core_error)?;
                    active_indices.push(index);
                    bounds.push(bound);
                }
                GGLWEUse::Empty => empty.push((index, key, dsize)),
            }
        }

        let uses: Vec<_> = bounds.iter().map(|bound| *bound.use_()).collect();
        let active_query: Vec<_> = active_indices
            .iter()
            .map(|&index| {
                let item = &items[index];
                CKKSSquareIntoItem {
                    dst: &*item.dst,
                    a: item.a,
                }
            })
            .collect();
        let mut required = if active_query.is_empty() {
            0
        } else {
            BE::ckks_square_into_batch_tmp_bytes_impl(self, &active_query, &uses)
        };
        for &(index, key, dsize) in &empty {
            let item = &items[index];
            required = required.max(BE::ckks_square_tmp_bytes_impl(
                self,
                &*item.dst,
                item.a,
                &key.with_dsize(dsize),
            ));
        }
        ensure_batch_scratch(scratch, required, OP)?;
        drop(active_query);

        if !active_indices.is_empty() {
            let mut pos = 0;
            let mut active_items = Vec::with_capacity(active_indices.len());
            for (index, item) in items.iter_mut().enumerate() {
                if active_indices.get(pos).copied() == Some(index) {
                    active_items.push(CKKSSquareIntoItem {
                        dst: &mut *item.dst,
                        a: item.a,
                    });
                    pos += 1;
                }
            }
            BE::ckks_square_into_batch_impl(self, &mut active_items, &bounds, scratch)?;
        }
        for (index, key, dsize) in empty {
            let item = &mut items[index];
            BE::ckks_square_into_impl(self, item.dst, item.a, &key.with_dsize(dsize), scratch)?;
        }
        Ok(())
    }

    fn ckks_square_assign_batch_tmp_bytes<Dst, H>(&self, items: &[CKKSSquareAssignItem<&Dst>], tsk: &H) -> usize
    where
        Dst: CKKSCtBounds,
        H: GLWERelinearizationKeyLayoutHelper,
    {
        let mut active_items = Vec::with_capacity(items.len());
        let mut uses = Vec::with_capacity(items.len());
        let mut best = 0;
        for item in items {
            let k = square_k(item.dst);
            let (layout, dsize) = tsk.get_relinearization_key_layout_for(k).unwrap_or_else(|e| panic!("{e}"));
            let layout = layout.with_dsize(dsize);
            match layout.bind_covering_for(k).unwrap_or_else(|e| panic!("{e}")) {
                GGLWEUse::Active(use_) => {
                    active_items.push(CKKSSquareAssignItem { dst: item.dst });
                    uses.push(use_);
                }
                GGLWEUse::Empty => {
                    best = best.max(BE::ckks_square_tmp_bytes_impl(self, item.dst, item.dst, &layout));
                }
            }
        }
        if !active_items.is_empty() {
            best = best.max(BE::ckks_square_assign_batch_tmp_bytes_impl(self, &active_items, &uses));
        }
        best
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
        const OP: &str = "ckks_square_assign_batch";
        plan_all(items, |item| {
            let dst = &*item.dst;
            get_mul_ct_params(dst, dst, dst).map(|_| ())
        })?;

        let mut active_indices = Vec::with_capacity(items.len());
        let mut bounds = Vec::with_capacity(items.len());
        let mut empty = Vec::new();
        for (index, item) in items.iter().enumerate() {
            let k = square_k(item.dst);
            let (key, dsize) = resolve(tsk, k, OP)?;
            match key.with_dsize(dsize).bind_covering_for(k).map_err(core_error)? {
                GGLWEUse::Active(use_) => {
                    let bound = GLWETensorKeyPreparedBound::new(GLWETensorKeyPreparedToBackendRef::to_backend_ref(key), use_)
                        .map_err(core_error)?;
                    active_indices.push(index);
                    bounds.push(bound);
                }
                GGLWEUse::Empty => empty.push((index, key, dsize)),
            }
        }

        let uses: Vec<_> = bounds.iter().map(|bound| *bound.use_()).collect();
        let active_query: Vec<_> = active_indices
            .iter()
            .map(|&index| CKKSSquareAssignItem { dst: &*items[index].dst })
            .collect();
        let mut required = if active_query.is_empty() {
            0
        } else {
            BE::ckks_square_assign_batch_tmp_bytes_impl(self, &active_query, &uses)
        };
        for &(index, key, dsize) in &empty {
            let item = &items[index];
            required = required.max(BE::ckks_square_tmp_bytes_impl(
                self,
                &*item.dst,
                &*item.dst,
                &key.with_dsize(dsize),
            ));
        }
        ensure_batch_scratch(scratch, required, OP)?;
        drop(active_query);

        if !active_indices.is_empty() {
            let mut pos = 0;
            let mut active_items = Vec::with_capacity(active_indices.len());
            for (index, item) in items.iter_mut().enumerate() {
                if active_indices.get(pos).copied() == Some(index) {
                    active_items.push(CKKSSquareAssignItem { dst: &mut *item.dst });
                    pos += 1;
                }
            }
            BE::ckks_square_assign_batch_impl(self, &mut active_items, &bounds, scratch)?;
        }
        for (index, key, dsize) in empty {
            let item = &mut items[index];
            BE::ckks_square_assign_impl(self, item.dst, &key.with_dsize(dsize), scratch)?;
        }
        Ok(())
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
        let mut active_items = Vec::with_capacity(items.len());
        let mut uses = Vec::with_capacity(items.len());
        let mut best = 0;
        for item in items {
            let k = prepared_mul_k(item.dst, item.prepared.prepared_k());
            let (layout, dsize) = tsk.get_relinearization_key_layout_for(k).unwrap_or_else(|e| panic!("{e}"));
            let layout = layout.with_dsize(dsize);
            match layout.bind_covering_for(k).unwrap_or_else(|e| panic!("{e}")) {
                GGLWEUse::Active(use_) => {
                    active_items.push(CKKSPreparedMulAssignItem {
                        dst: item.dst,
                        prepared: item.prepared,
                    });
                    uses.push(use_);
                }
                GGLWEUse::Empty => {
                    let prepared_k =
                        u32::try_from(item.prepared.prepared_k()).unwrap_or_else(|_| panic!("prepared precision exceeds u32"));
                    let right = poulpy_core::layouts::GLWELayout {
                        n: item.dst.n(),
                        base2k: item.dst.base2k(),
                        k: TorusPrecision(prepared_k),
                        rank: item.dst.rank(),
                    };
                    best = best.max(BE::ckks_mul_tmp_bytes_impl(self, item.dst, item.dst, &right, &layout));
                }
            }
        }
        if !active_items.is_empty() {
            best = best.max(BE::ckks_mul_prepared_assign_batch_tmp_bytes_impl(self, &active_items, &uses));
        }
        best
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
        const OP: &str = "ckks_mul_prepared_assign_batch";
        let planned_ks: Vec<TorusPrecision> = items
            .iter()
            .map(|item| get_mul_prepared_params(&*item.dst, item.prepared).map(|(_, _, _, k)| k))
            .collect::<Result<_>>()?;

        let mut active_indices = Vec::with_capacity(items.len());
        let mut bounds = Vec::with_capacity(items.len());
        let mut empty = Vec::new();
        for (index, k) in planned_ks.iter().copied().enumerate() {
            let (key, dsize) = resolve(tsk, k, OP)?;
            match key.with_dsize(dsize).bind_covering_for(k).map_err(core_error)? {
                GGLWEUse::Active(use_) => {
                    let bound = GLWETensorKeyPreparedBound::new(GLWETensorKeyPreparedToBackendRef::to_backend_ref(key), use_)
                        .map_err(core_error)?;
                    active_indices.push(index);
                    bounds.push(bound);
                }
                GGLWEUse::Empty => empty.push((index, key, dsize)),
            }
        }

        let uses: Vec<_> = bounds.iter().map(|bound| *bound.use_()).collect();
        let active_query: Vec<_> = active_indices
            .iter()
            .map(|&index| {
                let item = &items[index];
                CKKSPreparedMulAssignItem {
                    dst: &*item.dst,
                    prepared: item.prepared,
                }
            })
            .collect();
        let mut required = if active_query.is_empty() {
            0
        } else {
            BE::ckks_mul_prepared_assign_batch_tmp_bytes_impl(self, &active_query, &uses)
        };
        for &(index, key, dsize) in &empty {
            let item = &items[index];
            let prepared_k = u32::try_from(item.prepared.k).expect("prepared precision was checked during batch planning");
            let right = poulpy_core::layouts::GLWELayout {
                n: item.dst.n(),
                base2k: item.dst.base2k(),
                k: TorusPrecision(prepared_k),
                rank: item.dst.rank(),
            };
            required = required.max(BE::ckks_mul_tmp_bytes_impl(
                self,
                &*item.dst,
                &*item.dst,
                &right,
                &key.with_dsize(dsize),
            ));
        }
        ensure_batch_scratch(scratch, required, OP)?;
        drop(active_query);

        if !active_indices.is_empty() {
            let mut pos = 0;
            let mut active_items = Vec::with_capacity(active_indices.len());
            for (index, item) in items.iter_mut().enumerate() {
                if active_indices.get(pos).copied() == Some(index) {
                    active_items.push(CKKSPreparedMulAssignItem {
                        dst: &mut *item.dst,
                        prepared: item.prepared,
                    });
                    pos += 1;
                }
            }
            BE::ckks_mul_prepared_assign_batch_impl(self, &mut active_items, &bounds, scratch)?;
        }
        for (index, key, dsize) in empty {
            let item = &mut items[index];
            BE::ckks_mul_prepared_assign_impl(self, item.dst, item.prepared, &key.with_dsize(dsize), scratch)?;
        }
        Ok(())
    }
}
