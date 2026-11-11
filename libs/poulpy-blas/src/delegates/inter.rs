use poulpy_ckks::CKKSResult as Result;
use poulpy_core::layouts::{GGLWEInfos, GetAutomorphismKey};
use poulpy_hal::layouts::{Backend, Module, Ring, ScratchArena};

use crate::{
    api::BlasMatInterOps,
    layouts::{BlasGrid, BlasGridInfos, BlasGridMut},
    oep::BlasMatInterImpl,
};

impl<BE: Backend + BlasMatInterImpl<Rg>, Rg: Ring> BlasMatInterOps<BE, Rg> for Module<BE> {
    fn blas_rotate_tmp_bytes<R, A, K>(&self, res: &R, a: &A, keys: &K) -> usize
    where
        R: BlasGridInfos,
        A: BlasGridInfos,
        K: GGLWEInfos,
    {
        BE::blas_rotate_tmp_bytes(self, res, a, keys)
    }

    fn blas_rotate<R, A, K>(&self, r: i64, res: &mut R, a: &A, keys: &K, scratch: &mut ScratchArena<'_, BE>) -> Result<()>
    where
        R: BlasGridMut<BE, Rg>,
        A: BlasGrid<BE, Rg>,
        K: GetAutomorphismKey<BE, Rg> + Sync,
    {
        BE::blas_rotate(self, r, res, a, keys, scratch)
    }

    fn blas_sum_tmp_bytes<R, A, K>(&self, res: &R, a: &A, keys: &K) -> usize
    where
        R: BlasGridInfos,
        A: BlasGridInfos,
        K: GGLWEInfos,
    {
        BE::blas_sum_tmp_bytes(self, res, a, keys)
    }

    fn blas_sum<R, A, K>(&self, res: &mut R, a: &A, keys: &K, scratch: &mut ScratchArena<'_, BE>) -> Result<()>
    where
        R: BlasGridMut<BE, Rg>,
        A: BlasGrid<BE, Rg>,
        K: GetAutomorphismKey<BE, Rg> + Sync,
    {
        BE::blas_sum(self, res, a, keys, scratch)
    }
}
