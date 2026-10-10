use poulpy_ckks::{CKKSResult as Result, layouts::CKKSCiphertext};
use poulpy_core::layouts::{GGLWEInfos, GLWEToBackendMut, GLWEToBackendRef, GetAutomorphismKey};
use poulpy_hal::layouts::{Backend, Data, ScratchArena};

use crate::layouts::{BlasCiphertext, BlasContext, BlasGridInfos};

/// The same map applied to every matrix of the batch, across the blocks of a grid.
pub trait BlasMatIntraOps<BE: Backend> {
    /// Scratch bytes [`Self::blas_transpose`] needs into `res` from `a` with keys of layout `key`.
    fn blas_transpose_tmp_bytes<R, A, K>(&self, res: &R, a: &A, key: &K) -> usize
    where
        R: BlasGridInfos,
        A: BlasGridInfos,
        K: GGLWEInfos;

    /// `res` receives the transpose of every matrix of `a`.
    fn blas_transpose<DR, DA, K>(
        &self,
        res: &mut BlasCiphertext<DR, BE::ZnxWord, BE::Ring>,
        a: &BlasCiphertext<DA, BE::ZnxWord, BE::Ring>,
        ctx: &BlasContext<BE>,
        keys: &K,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        DR: Data,
        DA: Data,
        CKKSCiphertext<DR, BE::ZnxWord, BE::Ring>: GLWEToBackendMut<BE>,
        CKKSCiphertext<DA, BE::ZnxWord, BE::Ring>: GLWEToBackendRef<BE>,
        K: GetAutomorphismKey<BE>;

    /// Scratch bytes [`Self::blas_rotate_rows`] needs into `res` from `a` with keys of layout `key`.
    fn blas_rotate_rows_tmp_bytes<R, A, K>(&self, res: &R, a: &A, key: &K) -> usize
    where
        R: BlasGridInfos,
        A: BlasGridInfos,
        K: GGLWEInfos;

    /// `res` receives every matrix of `a` with row `r` replaced by row `r + k` modulo the matrix height.
    fn blas_rotate_rows<DR, DA, K>(
        &self,
        res: &mut BlasCiphertext<DR, BE::ZnxWord, BE::Ring>,
        a: &BlasCiphertext<DA, BE::ZnxWord, BE::Ring>,
        k: usize,
        ctx: &BlasContext<BE>,
        keys: &K,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        DR: Data,
        DA: Data,
        CKKSCiphertext<DR, BE::ZnxWord, BE::Ring>: GLWEToBackendMut<BE>,
        CKKSCiphertext<DA, BE::ZnxWord, BE::Ring>: GLWEToBackendRef<BE>,
        K: GetAutomorphismKey<BE>;

    /// Scratch bytes [`Self::blas_rotate_cols`] needs into `res` from `a` with keys of layout `key`.
    fn blas_rotate_cols_tmp_bytes<R, A, K>(&self, res: &R, a: &A, key: &K) -> usize
    where
        R: BlasGridInfos,
        A: BlasGridInfos,
        K: GGLWEInfos;

    /// `res` receives every matrix of `a` with column `c` replaced by column `c + k` modulo the matrix width.
    fn blas_rotate_cols<DR, DA, K>(
        &self,
        res: &mut BlasCiphertext<DR, BE::ZnxWord, BE::Ring>,
        a: &BlasCiphertext<DA, BE::ZnxWord, BE::Ring>,
        k: usize,
        ctx: &BlasContext<BE>,
        keys: &K,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        DR: Data,
        DA: Data,
        CKKSCiphertext<DR, BE::ZnxWord, BE::Ring>: GLWEToBackendMut<BE>,
        CKKSCiphertext<DA, BE::ZnxWord, BE::Ring>: GLWEToBackendRef<BE>,
        K: GetAutomorphismKey<BE>;
}
