use crate::CKKSResult as Result;
use poulpy_core::layouts::{Degree, GGLWEInfos, GLWEToBackendMut, GLWEToBackendRef, GetAutomorphismKey};
use poulpy_hal::layouts::{Backend, Module, ScratchArena};

use crate::{CKKSCtBounds, SetCKKSInfos, api::CKKSInnerSumOps, oep::CKKSInnerSumImpl};

impl<BE: Backend + CKKSInnerSumImpl> CKKSInnerSumOps<BE> for Module<BE> {
    fn ckks_inner_sum_galois_elements(&self, n: Degree, log_batch: usize, log_count: usize) -> Vec<i64> {
        BE::ckks_inner_sum_galois_elements(self, n, log_batch, log_count)
    }

    fn ckks_replicate_galois_elements(&self, n: Degree, log_batch: usize, log_count: usize) -> Vec<i64> {
        BE::ckks_replicate_galois_elements(self, n, log_batch, log_count)
    }

    fn ckks_inner_sum_tmp_bytes<R, A, K>(&self, res: &R, a: &A, key: &K) -> usize
    where
        R: CKKSCtBounds,
        A: CKKSCtBounds,
        K: GGLWEInfos,
    {
        BE::ckks_inner_sum_tmp_bytes(self, res, a, key)
    }

    fn ckks_inner_sum<R, A, H>(
        &self,
        res: &mut R,
        a: &A,
        log_batch: usize,
        log_count: usize,
        normalize: bool,
        keys: &H,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        R: GLWEToBackendMut<BE> + CKKSCtBounds + SetCKKSInfos,
        A: GLWEToBackendRef<BE> + CKKSCtBounds,
        H: GetAutomorphismKey<BE>,
    {
        BE::ckks_inner_sum(self, res, a, log_batch, log_count, normalize, keys, scratch)
    }

    fn ckks_inner_sum_assign<R, H>(
        &self,
        res: &mut R,
        log_batch: usize,
        log_count: usize,
        normalize: bool,
        keys: &H,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        R: GLWEToBackendMut<BE> + CKKSCtBounds + SetCKKSInfos,
        H: GetAutomorphismKey<BE>,
    {
        BE::ckks_inner_sum_assign(self, res, log_batch, log_count, normalize, keys, scratch)
    }

    fn ckks_replicate<R, A, H>(
        &self,
        res: &mut R,
        a: &A,
        log_batch: usize,
        log_count: usize,
        keys: &H,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        R: GLWEToBackendMut<BE> + CKKSCtBounds + SetCKKSInfos,
        A: GLWEToBackendRef<BE> + CKKSCtBounds,
        H: GetAutomorphismKey<BE>,
    {
        BE::ckks_replicate(self, res, a, log_batch, log_count, keys, scratch)
    }

    fn ckks_replicate_assign<R, H>(
        &self,
        res: &mut R,
        log_batch: usize,
        log_count: usize,
        keys: &H,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        R: GLWEToBackendMut<BE> + CKKSCtBounds + SetCKKSInfos,
        H: GetAutomorphismKey<BE>,
    {
        BE::ckks_replicate_assign(self, res, log_batch, log_count, keys, scratch)
    }
}
