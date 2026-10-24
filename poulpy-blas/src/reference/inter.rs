use poulpy_ckks::{
    CKKSResult as Result,
    api::{CKKSAddOps, CKKSCopyOps, CKKSRotateOps},
    layouts::ScratchArenaTakeCKKS,
};
use poulpy_core::{
    GLWEBytesOf,
    layouts::{GGLWEInfos, GLWEToBackendMut, GetAutomorphismKey, LWEInfos},
};
use poulpy_hal::layouts::{Backend, Module, ScratchArena};

use crate::layouts::{
    BlasGrid, BlasGridInfos, BlasGridMut,
    context::sum_rotations,
    grid_batch,
    validate::{KeyLayouts, assert_scratch, ensure_rotation, rotate_uses, sum_uses, validate_blas_rotate, validate_blas_sum},
};

/// Reference compositions of [`crate::api::BlasMatInterOps`]. Each runs its validator,
/// [`crate::layouts::validate_blas_rotate`] or [`crate::layouts::validate_blas_sum`], before any write, so its errors
/// leave `res` and its batch record untouched, then asserts its scratch and returns poulpy-ckks's budget errors before
/// any write; on `Ok`, `res` records the batch the validator returns, `a`'s. They run block after block on one scratch,
/// and the first error stops them.
pub trait BlasMatInterReference<BE: Backend> {
    /// The rotation of one block.
    fn blas_rotate_tmp_bytes_reference<R, A, K>(&self, res: &R, a: &A, atk: &K) -> usize
    where
        Self: CKKSRotateOps<BE>,
        R: BlasGridInfos,
        A: BlasGridInfos,
        K: GGLWEInfos,
    {
        self.ckks_rotate_tmp_bytes(&res.block_layout(), atk)
            .max(self.ckks_rotate_tmp_bytes(&a.block_layout(), atk))
    }

    /// `ckks_rotate_into` by `r d^2` slots on every block.
    fn blas_rotate_reference<R, A, K>(
        &self,
        res: &mut R,
        a: &A,
        r: i64,
        atk: &K,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        Self: CKKSRotateOps<BE>,
        R: BlasGridMut<BE>,
        A: BlasGrid<BE>,
        K: GetAutomorphismKey<BE> + Sync;

    /// The loop over one block carves, rotates and adds on the layout of a block of `res`.
    fn blas_sum_tmp_bytes_reference<R, A, K>(&self, res: &R, a: &A, atk: &K) -> usize
    where
        Self: CKKSRotateOps<BE> + CKKSAddOps<BE> + CKKSCopyOps<BE> + GLWEBytesOf<BE>,
        R: BlasGridInfos,
        A: BlasGridInfos,
        K: GGLWEInfos,
    {
        let (res, a) = (res.block_layout(), a.block_layout());
        BE::scratch_aligned(self.glwe_bytes_of_from_infos(&res))
            + self
                .ckks_rotate_tmp_bytes(&res, atk)
                .max(self.ckks_add_tmp_bytes(res.size()))
                .max(self.ckks_copy_tmp_bytes(&res, &a))
    }

    /// Per block, `res = a`, then `res += rot(res, 2^i d^2)` for every `i < log2(mat_slots)` in order.
    fn blas_sum_reference<R, A, K>(&self, res: &mut R, a: &A, atk: &K, scratch: &mut ScratchArena<'_, BE>) -> Result<()>
    where
        Self: CKKSRotateOps<BE> + CKKSAddOps<BE> + CKKSCopyOps<BE> + GLWEBytesOf<BE>,
        R: BlasGridMut<BE>,
        A: BlasGrid<BE>,
        K: GetAutomorphismKey<BE> + Sync;
}

impl<BE: Backend> BlasMatInterReference<BE> for Module<BE> {
    fn blas_rotate_reference<R, A, K>(
        &self,
        res: &mut R,
        a: &A,
        r: i64,
        atk: &K,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        Self: CKKSRotateOps<BE>,
        R: BlasGridMut<BE>,
        A: BlasGrid<BE>,
        K: GetAutomorphismKey<BE> + Sync,
    {
        let op = "blas_rotate";
        let batch = validate_blas_rotate(self, &*res, a, r, atk)?;
        let rotation = r << (2 * a.log_d());
        let used = KeyLayouts::new(atk);
        let _ = rotate_uses(&*res, a, r, &mut |u| ensure_rotation(op, self, &used, u));
        let bytes = used.max_bytes(|key| self.blas_rotate_tmp_bytes_reference(&*res, a, key));
        assert_scratch(op, bytes, scratch);
        for (dst, src) in res.blocks_mut().iter_mut().zip(a.blocks()) {
            self.ckks_rotate_into(rotation, dst, src, atk, scratch)?;
        }
        res.set_mat_slots(batch);
        Ok(())
    }

    fn blas_sum_reference<R, A, K>(&self, res: &mut R, a: &A, atk: &K, scratch: &mut ScratchArena<'_, BE>) -> Result<()>
    where
        Self: CKKSRotateOps<BE> + CKKSAddOps<BE> + CKKSCopyOps<BE> + GLWEBytesOf<BE>,
        R: BlasGridMut<BE>,
        A: BlasGrid<BE>,
        K: GetAutomorphismKey<BE> + Sync,
    {
        let op = "blas_sum";
        let batch = validate_blas_sum(self, &*res, a, atk)?;
        let rotations: Vec<i64> = sum_rotations(a.log_d(), grid_batch::<BE::Ring>(a)).collect();
        let used = KeyLayouts::new(atk);
        let _ = sum_uses::<BE::Ring>(&*res, a, &mut |u| ensure_rotation(op, self, &used, u));
        let bytes = used.max_bytes(|key| self.blas_sum_tmp_bytes_reference(&*res, a, key));
        assert_scratch(op, bytes, scratch);
        for (dst, src) in res.blocks_mut().iter_mut().zip(a.blocks()) {
            self.ckks_copy(dst, src, scratch)?;
            dst.set_noise(None);
            for &rotation in &rotations {
                scratch.scope(|s| -> Result<()> {
                    let (mut rotated, mut s) = s.take_ckks_ciphertext_like_scratch(&*dst);
                    self.ckks_rotate_into(rotation, &mut rotated, &*dst, atk, &mut s)?;
                    self.ckks_add_assign(dst, &rotated, &mut s)
                })?;
            }
        }
        res.set_mat_slots(batch);
        Ok(())
    }
}
