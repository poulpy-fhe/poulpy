use crate::CKKSResult as Result;
use poulpy_core::layouts::{GGLWEInfos, GLWEToBackendMut, GLWEToBackendRef, GetAutomorphismKey};
use poulpy_hal::layouts::{Backend, Module, ScratchArena};

use crate::{CKKSCtBounds, SetCKKSInfos, api::CKKSInnerSumOps, oep::CKKSInnerSumImpl};

impl<BE: Backend + CKKSInnerSumImpl> CKKSInnerSumOps<BE> for Module<BE> {
    fn ckks_inner_sum_into_tmp_bytes<R, A, K>(&self, res: &R, a: &A, key: &K) -> usize
    where
        R: CKKSCtBounds,
        A: CKKSCtBounds,
        K: GGLWEInfos,
    {
        BE::ckks_inner_sum_into_tmp_bytes_impl(self, res, a, key)
    }

    fn ckks_inner_sum_assign_tmp_bytes<R, K>(&self, res: &R, key: &K) -> usize
    where
        R: CKKSCtBounds,
        K: GGLWEInfos,
    {
        BE::ckks_inner_sum_assign_tmp_bytes_impl(self, res, key)
    }

    fn ckks_inner_sum_into<Dst, Src, H>(
        &self,
        dst: &mut Dst,
        src: &Src,
        log_batch: usize,
        log_count: usize,
        normalize: bool,
        keys: &H,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        Dst: GLWEToBackendMut<BE> + CKKSCtBounds + SetCKKSInfos,
        Src: GLWEToBackendRef<BE> + CKKSCtBounds,
        H: GetAutomorphismKey<BE>,
    {
        BE::ckks_inner_sum_into_impl(self, dst, src, log_batch, log_count, normalize, keys, scratch)
    }

    fn ckks_inner_sum_assign<Dst, H>(
        &self,
        dst: &mut Dst,
        log_batch: usize,
        log_count: usize,
        normalize: bool,
        keys: &H,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        Dst: GLWEToBackendMut<BE> + CKKSCtBounds + SetCKKSInfos,
        H: GetAutomorphismKey<BE>,
    {
        BE::ckks_inner_sum_assign_impl(self, dst, log_batch, log_count, normalize, keys, scratch)
    }

    fn ckks_replicate_into_tmp_bytes<R, A, K>(&self, res: &R, a: &A, key: &K) -> usize
    where
        R: CKKSCtBounds,
        A: CKKSCtBounds,
        K: GGLWEInfos,
    {
        BE::ckks_replicate_into_tmp_bytes_impl(self, res, a, key)
    }

    fn ckks_replicate_assign_tmp_bytes<R, K>(&self, res: &R, key: &K) -> usize
    where
        R: CKKSCtBounds,
        K: GGLWEInfos,
    {
        BE::ckks_replicate_assign_tmp_bytes_impl(self, res, key)
    }

    fn ckks_replicate_into<Dst, Src, H>(
        &self,
        dst: &mut Dst,
        src: &Src,
        log_batch: usize,
        log_count: usize,
        keys: &H,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        Dst: GLWEToBackendMut<BE> + CKKSCtBounds + SetCKKSInfos,
        Src: GLWEToBackendRef<BE> + CKKSCtBounds,
        H: GetAutomorphismKey<BE>,
    {
        BE::ckks_replicate_into_impl(self, dst, src, log_batch, log_count, keys, scratch)
    }

    fn ckks_replicate_assign<Dst, H>(
        &self,
        dst: &mut Dst,
        log_batch: usize,
        log_count: usize,
        keys: &H,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        Dst: GLWEToBackendMut<BE> + CKKSCtBounds + SetCKKSInfos,
        H: GetAutomorphismKey<BE>,
    {
        BE::ckks_replicate_assign_impl(self, dst, log_batch, log_count, keys, scratch)
    }
}
