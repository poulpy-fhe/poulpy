use poulpy_ckks::CKKSResult as Result;
use poulpy_core::layouts::{GGLWEInfos, GetAutomorphismKey};
use poulpy_hal::layouts::{Backend, Module, ScratchArena};

use crate::{
    api::BlasMatInterOps,
    layouts::{BlasGrid, BlasGridInfos, BlasGridMut},
    oep::BlasMatInterImpl,
};

impl<BE: Backend + BlasMatInterImpl> BlasMatInterOps<BE> for Module<BE> {
    fn blas_rotate_tmp_bytes<R, A, K>(&self, res: &R, a: &A, atk: &K) -> usize
    where
        R: BlasGridInfos,
        A: BlasGridInfos,
        K: GGLWEInfos,
    {
        BE::blas_rotate_tmp_bytes(self, res, a, atk)
    }

    fn blas_rotate<R, A, K>(&self, res: &mut R, a: &A, r: i64, atk: &K, scratch: &mut ScratchArena<'_, BE>) -> Result<()>
    where
        R: BlasGridMut<BE>,
        A: BlasGrid<BE>,
        K: GetAutomorphismKey<BE> + Sync,
    {
        BE::blas_rotate(self, res, a, r, atk, scratch)
    }

    fn blas_sum_tmp_bytes<R, A, K>(&self, res: &R, a: &A, atk: &K) -> usize
    where
        R: BlasGridInfos,
        A: BlasGridInfos,
        K: GGLWEInfos,
    {
        BE::blas_sum_tmp_bytes(self, res, a, atk)
    }

    fn blas_sum<R, A, K>(&self, res: &mut R, a: &A, atk: &K, scratch: &mut ScratchArena<'_, BE>) -> Result<()>
    where
        R: BlasGridMut<BE>,
        A: BlasGrid<BE>,
        K: GetAutomorphismKey<BE> + Sync,
    {
        BE::blas_sum(self, res, a, atk, scratch)
    }
}
