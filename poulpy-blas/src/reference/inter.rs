use poulpy_ckks::{
    CKKSResult as Result,
    api::{CKKSAddOps, CKKSCopyOps, CKKSModuleInfos, CKKSRotateOps},
    layouts::{CKKSCiphertext, ScratchArenaTakeCKKS},
};
use poulpy_core::{
    GLWEBytesOf,
    layouts::{GGLWEInfos, GLWEToBackendMut, GLWEToBackendRef, GetAutomorphismKey, LWEInfos},
};
use poulpy_hal::layouts::{Backend, Data, Module, ScratchArena};

use super::{KeyLayouts, assert_scratch, ensure_grid, ensure_keys, ensure_shape};
use crate::layouts::{BlasCiphertext, BlasGridInfos, sum_rotations};

/// Reference compositions of [`crate::api::BlasMatInterOps`].
pub trait BlasMatInterReference<BE: Backend> {
    fn blas_rotate_tmp_bytes_reference<R, A, K>(&self, res: &R, a: &A, key: &K) -> usize
    where
        Self: CKKSRotateOps<BE>,
        R: BlasGridInfos,
        A: BlasGridInfos,
        K: GGLWEInfos,
    {
        self.ckks_rotate_tmp_bytes(&res.block_layout(), key)
            .max(self.ckks_rotate_tmp_bytes(&a.block_layout(), key))
    }

    /// `ckks_rotate_into` by `r d^2` slots on every block.
    fn blas_rotate_reference<DR, DA, K>(
        &self,
        res: &mut BlasCiphertext<DR, BE::ZnxWord, BE::Ring>,
        a: &BlasCiphertext<DA, BE::ZnxWord, BE::Ring>,
        r: i64,
        keys: &K,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        Self: CKKSRotateOps<BE> + CKKSModuleInfos,
        DR: Data,
        DA: Data,
        CKKSCiphertext<DR, BE::ZnxWord, BE::Ring>: GLWEToBackendMut<BE>,
        CKKSCiphertext<DA, BE::ZnxWord, BE::Ring>: GLWEToBackendRef<BE>,
        K: GetAutomorphismKey<BE>,
    {
        let op = "blas_rotate";
        ensure_grid(op, a.log_d(), a)?;
        ensure_grid(op, a.log_d(), res)?;
        ensure_shape(op, res, a.rows(), a.cols())?;
        let rotation = r << (2 * a.log_d());
        let used = KeyLayouts::new(keys);
        ensure_keys(op, self, [rotation], &used, a.block_layout().k())?;
        let bytes = used.max_bytes(|key| self.blas_rotate_tmp_bytes_reference(&*res, a, key));
        assert_scratch(op, bytes, scratch);
        for (dst, src) in res.blocks_mut().iter_mut().zip(a.blocks()) {
            self.ckks_rotate_into(dst, src, rotation, keys, scratch)?;
        }
        Ok(())
    }

    /// The loop carves, rotates and adds on the layout of a block of `res`.
    fn blas_sum_tmp_bytes_reference<R, A, K>(&self, res: &R, a: &A, key: &K) -> usize
    where
        Self: CKKSRotateOps<BE> + CKKSAddOps<BE> + CKKSCopyOps<BE> + GLWEBytesOf<BE>,
        R: BlasGridInfos,
        A: BlasGridInfos,
        K: GGLWEInfos,
    {
        let (res, a) = (res.block_layout(), a.block_layout());
        BE::scratch_aligned(self.glwe_bytes_of_from_infos(&res))
            + self
                .ckks_rotate_tmp_bytes(&res, key)
                .max(self.ckks_add_tmp_bytes(res.size()))
                .max(self.ckks_copy_tmp_bytes(&res, &a))
    }

    /// Per block, `res = a`, then `res += rot(res, 2^i d^2)` for every `i < log2(mat_slots)`.
    fn blas_sum_reference<DR, DA, K>(
        &self,
        res: &mut BlasCiphertext<DR, BE::ZnxWord, BE::Ring>,
        a: &BlasCiphertext<DA, BE::ZnxWord, BE::Ring>,
        keys: &K,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        Self: CKKSRotateOps<BE> + CKKSAddOps<BE> + CKKSCopyOps<BE> + CKKSModuleInfos + GLWEBytesOf<BE>,
        DR: Data,
        DA: Data,
        CKKSCiphertext<DR, BE::ZnxWord, BE::Ring>: GLWEToBackendMut<BE>,
        CKKSCiphertext<DA, BE::ZnxWord, BE::Ring>: GLWEToBackendRef<BE>,
        K: GetAutomorphismKey<BE>,
    {
        let op = "blas_sum";
        ensure_grid(op, a.log_d(), a)?;
        ensure_grid(op, a.log_d(), res)?;
        ensure_shape(op, res, a.rows(), a.cols())?;
        let rotations: Vec<i64> = sum_rotations(a.log_d(), a.mat_slots).collect();
        // Keys are checked at the width the loop rotates at: `ckks_copy` leaves `res` at the narrower of `a` and `res`.
        let k = a.block_layout().k().min(res.block_layout().k());
        let used = KeyLayouts::new(keys);
        ensure_keys(op, self, rotations.iter().copied(), &used, k)?;
        let bytes = used.max_bytes(|key| self.blas_sum_tmp_bytes_reference(&*res, a, key));
        assert_scratch(op, bytes, scratch);
        for (dst, src) in res.blocks_mut().iter_mut().zip(a.blocks()) {
            self.ckks_copy(dst, src, scratch)?;
            dst.set_noise(None);
            for &rotation in &rotations {
                scratch.scope(|s| -> Result<()> {
                    let (mut rotated, mut s) = s.take_ckks_ciphertext_like_scratch(&*dst);
                    self.ckks_rotate_into(&mut rotated, &*dst, rotation, keys, &mut s)?;
                    self.ckks_add_assign(dst, &rotated, &mut s)
                })?;
            }
        }
        Ok(())
    }
}

impl<BE: Backend> BlasMatInterReference<BE> for Module<BE> {}
