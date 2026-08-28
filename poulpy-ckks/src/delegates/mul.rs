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

fn resolve<'a, H>(keys: &'a H, k: TorusPrecision, op: &'static str) -> Result<(&'a H::Key, poulpy_core::layouts::Dsize)>
where
    H: GLWERelinearizationKeyHelper,
{
    keys.get_relinearization_key_for(k)
        .map_err(|_| CKKSCompositionError::MissingRelinearizationKey { op, k: k.into() }.into())
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

fn prepared_mul_k<D: CKKSCtBounds>(dst: &D, prepared_k: usize) -> TorusPrecision {
    dst.k().max(TorusPrecision(prepared_k as u32))
}

/// What one lane of a batch resolves to, as a comparable value.
///
/// A batch is a set of independent items, each at its own precision, so two
/// lanes need not resolve to the same key or the same decomposition. Lanes are
/// grouped by this id and one call is issued per group: a batch never straddles
/// two physical keys or two decompositions, and no lane is dispatched under a
/// key resolved for a different precision.
///
/// The shape half is a value, not a pointer, so the scratch query and the
/// execution partition the same way even though one holds layouts and the other
/// holds prepared keys. Execution refines the partition by key identity, since
/// two distinct keys of the same shape must not share a call; a query group is
/// therefore a union of execution groups, and the reference batch scratch is the
/// maximum over its items, so answering on the coarser partition is an upper
/// bound on every finer one.
type Shape = (u32, u32, u32, u32, u32, u32, u32);

fn shape_of<K: GGLWEInfos>(key: &K, dsize: poulpy_core::layouts::Dsize) -> Shape {
    (
        key.n().as_u32(),
        key.base2k().as_u32(),
        key.dnum().as_u32(),
        key.k_aux().as_u32(),
        key.rank_in().as_u32(),
        key.rank_out().as_u32(),
        dsize.as_u32(),
    )
}

/// Execution lane: the resolved shape, refined by which key it actually is.
type Lane = (Shape, usize);

fn lane_of<H>(keys: &H, k: TorusPrecision, op: &'static str) -> Result<Lane>
where
    H: GLWERelinearizationKeyHelper,
    H::Key: GGLWEInfos,
{
    let (key, dsize) = resolve(keys, k, op)?;
    Ok((shape_of(key, dsize), key as *const H::Key as usize))
}

fn lane_layout_of<H>(keys: &H, k: TorusPrecision) -> Shape
where
    H: GLWERelinearizationKeyLayoutHelper,
{
    let (layout, dsize) = keys.get_relinearization_key_layout_for(k).unwrap_or_else(|e| panic!("{e}"));
    shape_of(layout, dsize)
}

/// Runs of consecutive equal lanes, after sorting the items by lane.
///
/// Items in a frontier are independent, so reordering them is not observable in
/// the results; it is what lets each group be handed over as one contiguous
/// slice.
fn group_by_lane<T, L: Ord + Copy, F>(items: &mut [T], mut lane: F) -> Vec<(L, usize, usize)>
where
    F: FnMut(&T) -> L,
{
    items.sort_by_key(&mut lane);
    let mut runs: Vec<(L, usize, usize)> = Vec::new();
    for (i, item) in items.iter().enumerate() {
        let id = lane(item);
        match runs.last_mut() {
            Some((last, _, end)) if *last == id => *end = i + 1,
            _ => runs.push((id, i, i + 1)),
        }
    }
    runs
}

/// The precision a run is bound at: the first lane's.
///
/// Every lane of a run resolves to the same key and decomposition, so any of
/// them selects it; the first is the one already known to resolve, which the
/// widest is not (it is a maximum over the run, not necessarily any lane's own
/// precision, and may fall outside the policy table).
fn run_k<T, F: Fn(&T) -> TorusPrecision>(run: &[T], k_of: F) -> TorusPrecision {
    k_of(run.first().expect("a run holds at least one lane"))
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
            .get_relinearization_key_layout_for(mul_k(a, b))
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
            .get_relinearization_key_layout_for(square_k(a))
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
            tsk.get_relinearization_key_for(mul_k(a, b))
                .map_err(|_| CKKSCompositionError::MissingRelinearizationKey {
                    op: "ckks_mul_into",
                    k: mul_k(a, b).into(),
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
            tsk.get_relinearization_key_for(mul_k(dst, a))
                .map_err(|_| CKKSCompositionError::MissingRelinearizationKey {
                    op: "ckks_mul_assign",
                    k: mul_k(dst, a).into(),
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
            .get_relinearization_key_for(prepared_mul_k(dst, prepared.k))
            .map_err(|_| CKKSCompositionError::MissingRelinearizationKey {
                op: "ckks_mul_prepared_assign",
                k: prepared_mul_k(dst, prepared.k).into(),
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
            tsk.get_relinearization_key_for(square_k(a))
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
            tsk.get_relinearization_key_for(square_k(dst))
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
        let k_of = |i: &CKKSMulIntoItem<&Dst, &A, &B>| mul_k(i.a, i.b);
        let mut lanes: Vec<(Shape, &CKKSMulIntoItem<&Dst, &A, &B>)> =
            items.iter().map(|i| (lane_layout_of(tsk, k_of(i)), i)).collect();
        let mut best: usize = 0;
        for run in group_by_lane(&mut lanes, |(lane, _)| *lane)
            .into_iter()
            .map(|(_, from, to)| &lanes[from..to])
        {
            let group: Vec<CKKSMulIntoItem<&Dst, &A, &B>> = run
                .iter()
                .map(|(_, i)| CKKSMulIntoItem {
                    dst: i.dst,
                    a: i.a,
                    b: i.b,
                })
                .collect();
            let k = run_k(run, |(_, i)| k_of(i));
            let (tsk, dsize) = tsk.get_relinearization_key_layout_for(k).unwrap_or_else(|e| panic!("{e}"));
            best = best.max(BE::ckks_mul_into_batch_tmp_bytes_impl(self, &group, &tsk.with_dsize(dsize)));
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
        let k_of = |i: &CKKSMulIntoItem<&mut Dst, &A, &B>| mul_k(i.a, i.b);
        for i in items.iter() {
            lane_of(tsk, k_of(i), OP)?;
        }
        for (_, from, to) in group_by_lane(items, |i| lane_of(tsk, k_of(i), OP).expect("every lane resolved above")) {
            let run = &mut items[from..to];
            let (key, dsize) = resolve(tsk, run_k(run, k_of), OP)?;
            BE::ckks_mul_into_batch_impl(self, run, &key.with_dsize(dsize), scratch)?;
        }
        Ok(())
    }

    fn ckks_square_into_batch_tmp_bytes<Dst, A, H>(&self, items: &[CKKSSquareIntoItem<&Dst, &A>], tsk: &H) -> usize
    where
        Dst: CKKSCtBounds,
        A: CKKSCtBounds,
        H: GLWERelinearizationKeyLayoutHelper,
    {
        let k_of = |i: &CKKSSquareIntoItem<&Dst, &A>| square_k(i.a);
        let mut lanes: Vec<(Shape, &CKKSSquareIntoItem<&Dst, &A>)> =
            items.iter().map(|i| (lane_layout_of(tsk, k_of(i)), i)).collect();
        let mut best: usize = 0;
        for run in group_by_lane(&mut lanes, |(lane, _)| *lane)
            .into_iter()
            .map(|(_, from, to)| &lanes[from..to])
        {
            let group: Vec<CKKSSquareIntoItem<&Dst, &A>> =
                run.iter().map(|(_, i)| CKKSSquareIntoItem { dst: i.dst, a: i.a }).collect();
            let k = run_k(run, |(_, i)| k_of(i));
            let (tsk, dsize) = tsk.get_relinearization_key_layout_for(k).unwrap_or_else(|e| panic!("{e}"));
            best = best.max(BE::ckks_square_into_batch_tmp_bytes_impl(
                self,
                &group,
                &tsk.with_dsize(dsize),
            ));
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
        let k_of = |i: &CKKSSquareIntoItem<&mut Dst, &A>| square_k(i.a);
        for i in items.iter() {
            lane_of(tsk, k_of(i), OP)?;
        }
        for (_, from, to) in group_by_lane(items, |i| lane_of(tsk, k_of(i), OP).expect("every lane resolved above")) {
            let run = &mut items[from..to];
            let (key, dsize) = resolve(tsk, run_k(run, k_of), OP)?;
            BE::ckks_square_into_batch_impl(self, run, &key.with_dsize(dsize), scratch)?;
        }
        Ok(())
    }

    fn ckks_square_assign_batch_tmp_bytes<Dst, H>(&self, items: &[CKKSSquareAssignItem<&Dst>], tsk: &H) -> usize
    where
        Dst: CKKSCtBounds,
        H: GLWERelinearizationKeyLayoutHelper,
    {
        let k_of = |i: &CKKSSquareAssignItem<&Dst>| square_k(i.dst);
        let mut lanes: Vec<(Shape, &CKKSSquareAssignItem<&Dst>)> =
            items.iter().map(|i| (lane_layout_of(tsk, k_of(i)), i)).collect();
        let mut best: usize = 0;
        for run in group_by_lane(&mut lanes, |(lane, _)| *lane)
            .into_iter()
            .map(|(_, from, to)| &lanes[from..to])
        {
            let group: Vec<CKKSSquareAssignItem<&Dst>> = run.iter().map(|(_, i)| CKKSSquareAssignItem { dst: i.dst }).collect();
            let k = run_k(run, |(_, i)| k_of(i));
            let (tsk, dsize) = tsk.get_relinearization_key_layout_for(k).unwrap_or_else(|e| panic!("{e}"));
            best = best.max(BE::ckks_square_assign_batch_tmp_bytes_impl(
                self,
                &group,
                &tsk.with_dsize(dsize),
            ));
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
        let k_of = |i: &CKKSSquareAssignItem<&mut Dst>| square_k(i.dst);
        for i in items.iter() {
            lane_of(tsk, k_of(i), OP)?;
        }
        for (_, from, to) in group_by_lane(items, |i| lane_of(tsk, k_of(i), OP).expect("every lane resolved above")) {
            let run = &mut items[from..to];
            let (key, dsize) = resolve(tsk, run_k(run, k_of), OP)?;
            BE::ckks_square_assign_batch_impl(self, run, &key.with_dsize(dsize), scratch)?;
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
        let k_of = |i: &CKKSPreparedMulAssignItem<&Dst, &PR>| prepared_mul_k(i.dst, i.prepared.prepared_k());
        let mut lanes: Vec<(Shape, &CKKSPreparedMulAssignItem<&Dst, &PR>)> =
            items.iter().map(|i| (lane_layout_of(tsk, k_of(i)), i)).collect();
        let mut best: usize = 0;
        for run in group_by_lane(&mut lanes, |(lane, _)| *lane)
            .into_iter()
            .map(|(_, from, to)| &lanes[from..to])
        {
            let group: Vec<CKKSPreparedMulAssignItem<&Dst, &PR>> = run
                .iter()
                .map(|(_, i)| CKKSPreparedMulAssignItem {
                    dst: i.dst,
                    prepared: i.prepared,
                })
                .collect();
            let k = run_k(run, |(_, i)| k_of(i));
            let (tsk, dsize) = tsk.get_relinearization_key_layout_for(k).unwrap_or_else(|e| panic!("{e}"));
            best = best.max(BE::ckks_mul_prepared_assign_batch_tmp_bytes_impl(
                self,
                &group,
                &tsk.with_dsize(dsize),
            ));
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
        let k_of = |i: &CKKSPreparedMulAssignItem<&mut Dst, &CKKSPreparedRight<BE>>| prepared_mul_k(i.dst, i.prepared.k);
        for i in items.iter() {
            lane_of(tsk, k_of(i), OP)?;
        }
        for (_, from, to) in group_by_lane(items, |i| lane_of(tsk, k_of(i), OP).expect("every lane resolved above")) {
            let run = &mut items[from..to];
            let (key, dsize) = resolve(tsk, run_k(run, k_of), OP)?;
            BE::ckks_mul_prepared_assign_batch_impl(self, run, &key.with_dsize(dsize), scratch)?;
        }
        Ok(())
    }
}
