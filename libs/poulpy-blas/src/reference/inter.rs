use poulpy_ckks::{
    CKKSResult as Result,
    api::{CKKSInnerSumOps, CKKSRotateOps},
};
use poulpy_core::layouts::{GGLWEInfos, GetAutomorphismKey};
use poulpy_hal::layouts::{Backend, Module, ScratchArena};

use crate::layouts::{
    BlasGrid, BlasGridInfos, BlasGridMut, grid_batch,
    validate::{KeyLayouts, assert_scratch, ensure_rotation, rotate_uses, sum_uses, validate_blas_rotate, validate_blas_sum},
};

/// Reference compositions of [`crate::api::BlasMatInterOps`]. Each runs its validator,
/// [`crate::layouts::validate_blas_rotate`] or [`crate::layouts::validate_blas_sum`], before any write, so its errors
/// leave `res` and its batch record untouched, then asserts its scratch and returns poulpy-ckks's budget errors before
/// any write; on `Ok`, `res` records the batch the validator returns, `a`'s. They run block after block on one scratch,
/// and the first error stops them.
pub trait BlasMatInterReference<BE: Backend> {
    /// The rotation of one block.
    fn blas_rotate_tmp_bytes_reference<R, A, K>(&self, res: &R, a: &A, keys: &K) -> usize
    where
        Self: CKKSRotateOps<BE>,
        R: BlasGridInfos,
        A: BlasGridInfos,
        K: GGLWEInfos,
    {
        self.ckks_rotate_tmp_bytes(&res.block_layout(), keys)
            .max(self.ckks_rotate_tmp_bytes(&a.block_layout(), keys))
    }

    /// `ckks_rotate_into` by `r d^2` slots on every block.
    fn blas_rotate_reference<R, A, K>(
        &self,
        r: i64,
        res: &mut R,
        a: &A,
        keys: &K,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        Self: CKKSRotateOps<BE>,
        R: BlasGridMut<BE>,
        A: BlasGrid<BE>,
        K: GetAutomorphismKey<BE> + Sync;

    /// The inner sum of one block into a block of `res`.
    fn blas_sum_tmp_bytes_reference<R, A, K>(&self, res: &R, a: &A, keys: &K) -> usize
    where
        Self: CKKSInnerSumOps<BE>,
        R: BlasGridInfos,
        A: BlasGridInfos,
        K: GGLWEInfos,
    {
        self.ckks_inner_sum_tmp_bytes(&res.block_layout(), &a.block_layout(), keys)
    }

    /// Per block, poulpy-ckks's `ckks_inner_sum` over `mat_slots` batches of `d^2` slots, a sum rather than a mean:
    /// `res = a`, then `res += rot(res, 2^i d^2)` for every `i < log2(mat_slots)` in order.
    fn blas_sum_reference<R, A, K>(&self, res: &mut R, a: &A, keys: &K, scratch: &mut ScratchArena<'_, BE>) -> Result<()>
    where
        Self: CKKSInnerSumOps<BE>,
        R: BlasGridMut<BE>,
        A: BlasGrid<BE>,
        K: GetAutomorphismKey<BE> + Sync;
}

impl<BE: Backend> BlasMatInterReference<BE> for Module<BE> {
    fn blas_rotate_reference<R, A, K>(
        &self,
        r: i64,
        res: &mut R,
        a: &A,
        keys: &K,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        Self: CKKSRotateOps<BE>,
        R: BlasGridMut<BE>,
        A: BlasGrid<BE>,
        K: GetAutomorphismKey<BE> + Sync,
    {
        let op = "blas_rotate";
        let batch = validate_blas_rotate(self, r, &*res, a, keys)?;
        let rotation = r << (2 * a.log_d());
        let used = KeyLayouts::new(keys);
        let _ = rotate_uses(r, &*res, a, &mut |u| ensure_rotation(op, self, &used, u));
        let bytes = used.max_bytes(|key| self.blas_rotate_tmp_bytes_reference(&*res, a, key));
        assert_scratch(op, bytes, scratch);
        for (dst, src) in res.blocks_mut().iter_mut().zip(a.blocks()) {
            self.ckks_rotate_into(rotation, dst, src, keys, scratch)?;
        }
        res.set_mat_slots(batch);
        Ok(())
    }

    fn blas_sum_reference<R, A, K>(&self, res: &mut R, a: &A, keys: &K, scratch: &mut ScratchArena<'_, BE>) -> Result<()>
    where
        Self: CKKSInnerSumOps<BE>,
        R: BlasGridMut<BE>,
        A: BlasGrid<BE>,
        K: GetAutomorphismKey<BE> + Sync,
    {
        let op = "blas_sum";
        let batch = validate_blas_sum(self, &*res, a, keys)?;
        let (log_batch, log_count) = (2 * a.log_d(), grid_batch::<BE::Ring>(a).ilog2() as usize);
        let used = KeyLayouts::new(keys);
        let _ = sum_uses::<BE::Ring>(&*res, a, &mut |u| ensure_rotation(op, self, &used, u));
        let bytes = used.max_bytes(|key| self.blas_sum_tmp_bytes_reference(&*res, a, key));
        assert_scratch(op, bytes, scratch);
        for (dst, src) in res.blocks_mut().iter_mut().zip(a.blocks()) {
            self.ckks_inner_sum(log_batch, log_count, false, dst, src, keys, scratch)?;
        }
        res.set_mat_slots(batch);
        Ok(())
    }
}
