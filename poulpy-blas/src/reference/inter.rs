use poulpy_ckks::{
    CKKSResult as Result,
    api::{CKKSAddOps, CKKSCopyOps, CKKSModuleInfos, CKKSRotateOps},
    layouts::ScratchArenaTakeCKKS,
};
use poulpy_core::{
    GLWEBytesOf,
    layouts::{GGLWEInfos, GLWEToBackendMut, GetAutomorphismKey, LWEInfos},
};
use poulpy_hal::layouts::{Backend, Module, ScratchArena};

use super::{KeyLayouts, assert_scratch, ensure_grid, ensure_keys, ensure_shape};
use crate::layouts::{BlasGrid, BlasGridInfos, BlasGridMut, batch_of, sum_rotations};

/// Reference compositions of [`crate::api::BlasMatInterOps`].
pub trait BlasMatInterReference<BE: Backend> {
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
        Self: CKKSRotateOps<BE> + CKKSModuleInfos,
        R: BlasGridMut<BE>,
        A: BlasGrid<BE>,
        K: GetAutomorphismKey<BE> + Sync,
    {
        let op = "blas_rotate";
        ensure_grid::<BE::Ring>(op, a.log_d(), a, a.blocks())?;
        ensure_grid::<BE::Ring>(op, a.log_d(), &*res, res.blocks())?;
        ensure_shape(op, res, a.rows(), a.cols())?;
        let rotation = r << (2 * a.log_d());
        let used = KeyLayouts::new(atk);
        ensure_keys(op, self, [rotation], &used, a.block_layout().k())?;
        let bytes = used.max_bytes(|key| self.blas_rotate_tmp_bytes_reference(&*res, a, key));
        assert_scratch(op, bytes, scratch);
        for (dst, src) in res.blocks_mut().iter_mut().zip(a.blocks()) {
            self.ckks_rotate_into(dst, src, rotation, atk, scratch)?;
        }
        Ok(())
    }

    /// The loop carves, rotates and adds on the layout of a block of `res`.
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

    /// Per block, `res = a`, then `res += rot(res, 2^i d^2)` for every `i < log2(mat_slots)`.
    fn blas_sum_reference<R, A, K>(&self, res: &mut R, a: &A, atk: &K, scratch: &mut ScratchArena<'_, BE>) -> Result<()>
    where
        Self: CKKSRotateOps<BE> + CKKSAddOps<BE> + CKKSCopyOps<BE> + CKKSModuleInfos + GLWEBytesOf<BE>,
        R: BlasGridMut<BE>,
        A: BlasGrid<BE>,
        K: GetAutomorphismKey<BE> + Sync,
    {
        let op = "blas_sum";
        ensure_grid::<BE::Ring>(op, a.log_d(), a, a.blocks())?;
        ensure_grid::<BE::Ring>(op, a.log_d(), &*res, res.blocks())?;
        ensure_shape(op, res, a.rows(), a.cols())?;
        let rotations: Vec<i64> = sum_rotations(a.log_d(), batch_of::<BE::Ring>(a)).collect();
        // Keys are checked at the width the loop rotates at: `ckks_copy` leaves `res` at the narrower of `a` and `res`.
        let k = a.block_layout().k().min(res.block_layout().k());
        let used = KeyLayouts::new(atk);
        ensure_keys(op, self, rotations.iter().copied(), &used, k)?;
        let bytes = used.max_bytes(|key| self.blas_sum_tmp_bytes_reference(&*res, a, key));
        assert_scratch(op, bytes, scratch);
        for (dst, src) in res.blocks_mut().iter_mut().zip(a.blocks()) {
            self.ckks_copy(dst, src, scratch)?;
            dst.set_noise(None);
            for &rotation in &rotations {
                scratch.scope(|s| -> Result<()> {
                    let (mut rotated, mut s) = s.take_ckks_ciphertext_like_scratch(&*dst);
                    self.ckks_rotate_into(&mut rotated, &*dst, rotation, atk, &mut s)?;
                    self.ckks_add_assign(dst, &rotated, &mut s)
                })?;
            }
        }
        Ok(())
    }
}

impl<BE: Backend> BlasMatInterReference<BE> for Module<BE> {}
