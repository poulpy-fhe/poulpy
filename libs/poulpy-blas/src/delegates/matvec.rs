use poulpy_ckks::CKKSResult as Result;
use poulpy_ckks::layouts::CKKSRightOperandPrepared;
use poulpy_core::layouts::{GGLWEInfos, GLWEInfos, GetAutomorphismKey, GetTensorKey};
use poulpy_hal::layouts::{Backend, Module, ScratchArena};

use crate::{
    api::BlasMatVecOps,
    layouts::{
        BlasDiagonalCiphertext, BlasDiagonalCiphertextInfos, BlasDiagonalCiphertextMap, BlasGrid, BlasGridInfos, BlasGridLine,
        BlasGridMut, BlasMatVecMaps, BlasVectorInfos, BlasVectorMut, BlasVectorRef,
    },
    oep::BlasMatVecImpl,
};

impl<BE: Backend + BlasMatVecImpl> BlasMatVecOps<BE> for Module<BE> {
    fn blas_matvec_pt_tmp_bytes<R, W, A, K>(&self, res: &R, w: &W, a: &A, keys: &K) -> usize
    where
        R: BlasVectorInfos,
        W: BlasMatVecMaps<BE>,
        A: BlasVectorInfos,
        K: GGLWEInfos,
    {
        BE::blas_matvec_pt_tmp_bytes(self, res, w, a, keys)
    }

    fn blas_matvec_pt<R, W, A, K>(&self, res: &mut R, w: &W, a: &A, keys: &K, scratch: &mut ScratchArena<'_, BE>) -> Result<()>
    where
        R: BlasVectorMut<BE>,
        W: BlasMatVecMaps<BE>,
        A: BlasVectorRef<BE>,
        K: GetAutomorphismKey<BE> + GetTensorKey<BE> + Sync,
    {
        BE::blas_matvec_pt(self, res, w, a, keys, scratch)
    }

    fn blas_grid_to_vector_tmp_bytes<R, W, A, K>(&self, res: &R, w: &W, a: &A, keys: &K) -> usize
    where
        R: BlasVectorInfos,
        W: BlasMatVecMaps<BE>,
        A: BlasGridInfos,
        K: GGLWEInfos,
    {
        BE::blas_grid_to_vector_tmp_bytes(self, res, w, a, keys)
    }

    fn blas_grid_to_vector<R, W, A, K>(
        &self,
        res: &mut R,
        w: &W,
        a: &A,
        line: BlasGridLine,
        keys: &K,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        R: BlasVectorMut<BE>,
        W: BlasMatVecMaps<BE>,
        A: BlasGrid<BE>,
        K: GetAutomorphismKey<BE> + GetTensorKey<BE> + Sync,
    {
        BE::blas_grid_to_vector(self, res, w, a, line, keys, scratch)
    }

    fn blas_vector_to_grid_tmp_bytes<R, W, A, K>(&self, res: &R, w: &W, a: &A, keys: &K) -> usize
    where
        R: BlasGridInfos,
        W: BlasMatVecMaps<BE>,
        A: BlasVectorInfos,
        K: GGLWEInfos,
    {
        BE::blas_vector_to_grid_tmp_bytes(self, res, w, a, keys)
    }

    fn blas_vector_to_grid<R, W, A, K>(
        &self,
        res: &mut R,
        line: BlasGridLine,
        w: &W,
        a: &A,
        keys: &K,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        R: BlasGridMut<BE>,
        W: BlasMatVecMaps<BE>,
        A: BlasVectorRef<BE>,
        K: GetAutomorphismKey<BE> + GetTensorKey<BE> + Sync,
    {
        BE::blas_vector_to_grid(self, res, line, w, a, keys, scratch)
    }

    fn blas_matvec_diagonal_tmp_bytes<R, M, A, T, K>(&self, res: &R, m: &M, a: &A, tsk: &T, keys: &K) -> usize
    where
        R: BlasVectorInfos,
        M: BlasDiagonalCiphertextMap<BE>,
        A: BlasVectorInfos,
        T: GGLWEInfos,
        K: GGLWEInfos,
    {
        BE::blas_matvec_diagonal_tmp_bytes(self, res, m, a, tsk, keys)
    }

    fn blas_matvec_diagonal<R, M, A, K>(
        &self,
        res: &mut R,
        m: &M,
        a: &A,
        keys: &K,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        R: BlasVectorMut<BE>,
        M: BlasDiagonalCiphertextMap<BE>,
        A: BlasVectorRef<BE>,
        K: GetAutomorphismKey<BE> + GetTensorKey<BE> + Sync,
    {
        BE::blas_matvec_diagonal(self, res, m, a, keys, scratch)
    }

    fn blas_append_row_diagonal_tmp_bytes<M, V, P, K>(&self, m: &M, v: &V, mask: &P, keys: &K) -> usize
    where
        M: BlasDiagonalCiphertextInfos,
        V: BlasVectorInfos,
        P: GLWEInfos,
        K: GGLWEInfos,
    {
        BE::blas_append_row_diagonal_tmp_bytes(self, m, v, mask, keys)
    }

    fn blas_append_row_diagonal<V, K>(
        &self,
        m: &mut BlasDiagonalCiphertext<BE>,
        row: i64,
        v: &V,
        mask: &CKKSRightOperandPrepared<BE>,
        keys: &K,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        V: BlasVectorRef<BE>,
        K: GetAutomorphismKey<BE> + Sync,
    {
        BE::blas_append_row_diagonal(self, m, row, v, mask, keys, scratch)
    }

    fn blas_append_col_diagonal_tmp_bytes<M, V, P, K>(&self, m: &M, v: &V, mask: &P, keys: &K) -> usize
    where
        M: BlasDiagonalCiphertextInfos,
        V: BlasVectorInfos,
        P: GLWEInfos,
        K: GGLWEInfos,
    {
        BE::blas_append_col_diagonal_tmp_bytes(self, m, v, mask, keys)
    }

    fn blas_append_col_diagonal<V, K>(
        &self,
        m: &mut BlasDiagonalCiphertext<BE>,
        col: i64,
        v: &V,
        mask: &CKKSRightOperandPrepared<BE>,
        keys: &K,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        V: BlasVectorRef<BE>,
        K: GetAutomorphismKey<BE> + Sync,
    {
        BE::blas_append_col_diagonal(self, m, col, v, mask, keys, scratch)
    }
}
