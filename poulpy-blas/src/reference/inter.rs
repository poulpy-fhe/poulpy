use poulpy_ckks::{
    CKKSResult as Result,
    api::{CKKSAddOps, CKKSCopyOps, CKKSModuleInfos, CKKSRotateOps},
    layouts::ScratchArenaTakeCKKS,
};
use poulpy_core::{
    GLWEBytesOf,
    layouts::{GGLWEInfos, GLWEToBackendMut, GetAutomorphismKey, LWEInfos, TorusPrecision},
};
use poulpy_hal::layouts::{Backend, Module, ScratchArena};

use super::{
    KeyLayouts, assert_scratch, ensure_grid, ensure_keys, ensure_layouts, ensure_shape, max_lazy_terms, reject, run_tasks,
    tasks_tmp_bytes,
};
use crate::layouts::{BlasGrid, BlasGridInfos, BlasGridMut, grid_batch, sum_rotations};

/// Checks a [`crate::api::BlasMatInterOps::blas_rotate`] call from metadata alone, before any implementation runs, and
/// returns the batch `res` records, `a`'s: `a` and `res` are grids at `a`'s `log_d` whose blocks tile their shape and
/// share one layout whose label allows their batch record ([`BlasGrid::blocks`], [`BlasGridInfos::mat_slots`]), `res`
/// has `a`'s shape and its blocks `a`'s degree, radix and rank, and `atk` holds the rotation key at `a`'s width.
///
/// # Errors
///
/// `InvalidOperand` for a grid, shape or layout it rejects, then `MissingAutomorphismKey`. It checks no budget:
/// poulpy-ckks rejects those inside the implementation, before any write.
pub fn validate_blas_rotate<BE: Backend>(
    module: &Module<BE>,
    res: &impl BlasGrid<BE>,
    a: &impl BlasGrid<BE>,
    r: i64,
    atk: &impl GetAutomorphismKey<BE>,
) -> Result<usize> {
    let op = "blas_rotate";
    ensure_same_grids(op, module, res, a)?;
    ensure_keys(op, module, [r << (2 * a.log_d())], atk, a.block_layout().k())?;
    Ok(grid_batch::<BE::Ring>(a))
}

/// Checks a [`crate::api::BlasMatInterOps::blas_sum`] call as [`validate_blas_rotate`] does, the rotation keys of the
/// batch at the narrower of `a`'s and `res`'s widths, and that the `log2(mat_slots) + 1` terms the sum adds up fit
/// `res`'s digits (`2^(63 - base2k)`), and returns the batch `res` records, `a`'s.
///
/// # Errors
///
/// `InvalidOperand` for a grid, shape, layout or term count it rejects, then `MissingAutomorphismKey`. It checks no
/// budget: poulpy-ckks rejects those inside the implementation, before any write.
pub fn validate_blas_sum<BE: Backend>(
    module: &Module<BE>,
    res: &impl BlasGrid<BE>,
    a: &impl BlasGrid<BE>,
    atk: &impl GetAutomorphismKey<BE>,
) -> Result<usize> {
    let op = "blas_sum";
    ensure_same_grids(op, module, res, a)?;
    let batch = grid_batch::<BE::Ring>(a);
    let (terms, base2k) = (batch.ilog2() as usize + 1, res.block_layout().base2k().as_usize());
    if terms > max_lazy_terms(base2k) {
        return Err(reject(op, format!("{terms} lazy terms risk i64 overflow at base2k={base2k}")));
    }
    ensure_keys(op, module, sum_rotations(a.log_d(), batch), atk, sum_key_width(res, a))?;
    Ok(batch)
}

/// `a` and `res` are grids at `a`'s `log_d`, `res` of `a`'s shape and its blocks of the module's degree and `a`'s radix
/// and rank.
fn ensure_same_grids<BE: Backend>(
    op: &'static str,
    module: &Module<BE>,
    res: &impl BlasGrid<BE>,
    a: &impl BlasGrid<BE>,
) -> Result<()> {
    ensure_grid::<BE::Ring>(op, a.log_d(), a, a.blocks())?;
    ensure_grid::<BE::Ring>(op, a.log_d(), res, res.blocks())?;
    ensure_shape(op, res, a.rows(), a.cols())?;
    ensure_layouts(op, module.n(), res, &[a])
}

/// The width `blas_sum` rotates at: `ckks_copy` leaves `res` at the narrower of `a` and `res`.
fn sum_key_width(res: &impl BlasGridInfos, a: &impl BlasGridInfos) -> TorusPrecision {
    a.block_layout().k().min(res.block_layout().k())
}

/// Reference compositions of [`crate::api::BlasMatInterOps`], on inputs [`validate_blas_rotate`] and
/// [`validate_blas_sum`] accepted: they check nothing but their scratch, and return poulpy-ckks's budget errors before
/// any write. They split the blocks over up to four workers of `BE`'s task executor, each on its own scratch slice, and
/// return the error of the lowest failing block, as a serial loop would.
pub trait BlasMatInterReference<BE: Backend> {
    /// The rotation of one block, on each worker's slice.
    fn blas_rotate_tmp_bytes_reference<R, A, K>(&self, res: &R, a: &A, atk: &K) -> usize
    where
        Self: CKKSRotateOps<BE>,
        R: BlasGridInfos,
        A: BlasGridInfos,
        K: GGLWEInfos,
    {
        let block = self
            .ckks_rotate_tmp_bytes(&res.block_layout(), atk)
            .max(self.ckks_rotate_tmp_bytes(&a.block_layout(), atk));
        tasks_tmp_bytes::<BE>(a.rows() * a.cols(), block)
    }

    /// `ckks_rotate_into` by `r d^2` slots on every block, the blocks in parallel.
    fn blas_rotate_reference<R, A, K>(
        &self,
        res: &mut R,
        a: &A,
        r: i64,
        atk: &K,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        Self: CKKSRotateOps<BE> + CKKSModuleInfos + Sync,
        R: BlasGridMut<BE>,
        A: BlasGrid<BE>,
        K: GetAutomorphismKey<BE> + Sync,
    {
        let op = "blas_rotate";
        let rotation = r << (2 * a.log_d());
        let used = KeyLayouts::new(atk);
        let _ = ensure_keys(op, self, [rotation], &used, a.block_layout().k());
        let bytes = used.max_bytes(|key| self.blas_rotate_tmp_bytes_reference(&*res, a, key));
        assert_scratch(op, bytes, scratch);
        let mut blocks: Vec<_> = res.blocks_mut().iter_mut().zip(a.blocks()).collect();
        run_tasks::<BE::TaskExecutor, _, _>(&mut blocks, scratch, |(dst, src), scratch| {
            self.ckks_rotate_into(*dst, *src, rotation, atk, scratch)
        })
    }

    /// The loop over one block carves, rotates and adds on the layout of a block of `res`, on each worker's slice.
    fn blas_sum_tmp_bytes_reference<R, A, K>(&self, res: &R, a: &A, atk: &K) -> usize
    where
        Self: CKKSRotateOps<BE> + CKKSAddOps<BE> + CKKSCopyOps<BE> + GLWEBytesOf<BE>,
        R: BlasGridInfos,
        A: BlasGridInfos,
        K: GGLWEInfos,
    {
        let (tasks, res, a) = (a.rows() * a.cols(), res.block_layout(), a.block_layout());
        let block = BE::scratch_aligned(self.glwe_bytes_of_from_infos(&res))
            + self
                .ckks_rotate_tmp_bytes(&res, atk)
                .max(self.ckks_add_tmp_bytes(res.size()))
                .max(self.ckks_copy_tmp_bytes(&res, &a));
        tasks_tmp_bytes::<BE>(tasks, block)
    }

    /// Per block, `res = a`, block after block, then `res += rot(res, 2^i d^2)` for every `i < log2(mat_slots)` in
    /// order, the blocks in parallel.
    fn blas_sum_reference<R, A, K>(&self, res: &mut R, a: &A, atk: &K, scratch: &mut ScratchArena<'_, BE>) -> Result<()>
    where
        Self: CKKSRotateOps<BE> + CKKSAddOps<BE> + CKKSCopyOps<BE> + CKKSModuleInfos + GLWEBytesOf<BE> + Sync,
        R: BlasGridMut<BE>,
        A: BlasGrid<BE>,
        K: GetAutomorphismKey<BE> + Sync,
    {
        let op = "blas_sum";
        let rotations: Vec<i64> = sum_rotations(a.log_d(), grid_batch::<BE::Ring>(a)).collect();
        let used = KeyLayouts::new(atk);
        let _ = ensure_keys(op, self, rotations.iter().copied(), &used, sum_key_width(&*res, a));
        let bytes = used.max_bytes(|key| self.blas_sum_tmp_bytes_reference(&*res, a, key));
        assert_scratch(op, bytes, scratch);
        for (dst, src) in res.blocks_mut().iter_mut().zip(a.blocks()) {
            self.ckks_copy(dst, src, scratch)?;
            dst.set_noise(None);
        }
        run_tasks::<BE::TaskExecutor, _, _>(res.blocks_mut(), scratch, |dst, scratch| {
            for &rotation in &rotations {
                scratch.scope(|s| -> Result<()> {
                    let (mut rotated, mut s) = s.take_ckks_ciphertext_like_scratch(&*dst);
                    self.ckks_rotate_into(&mut rotated, &*dst, rotation, atk, &mut s)?;
                    self.ckks_add_assign(dst, &rotated, &mut s)
                })?;
            }
            Ok(())
        })
    }
}

impl<BE: Backend> BlasMatInterReference<BE> for Module<BE> {}
