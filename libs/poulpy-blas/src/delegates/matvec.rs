use poulpy_ckks::CKKSResult as Result;
use poulpy_core::layouts::{GGLWEInfos, GetAutomorphismKey, GetTensorKey};
use poulpy_hal::layouts::{Backend, Module, ScratchArena};

use crate::{
    api::BlasMatVecOps,
    layouts::{BlasMatVecMaps, BlasVectorInfos, BlasVectorMut, BlasVectorRef},
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
}
