use anyhow::Result;
use poulpy_core::{
    GLWECopy, GLWEShift, ScratchArenaTakeCore,
    layouts::{GLWEToBackendMut, GLWEToBackendRef},
};
use poulpy_hal::layouts::{Backend, Module, ScratchArena};

use crate::{CKKSCtBounds, SetCKKSInfos, oep::CKKSPow2Impl};

use crate::api::CKKSPow2Ops;

impl<BE: Backend + CKKSPow2Impl<BE>> CKKSPow2Ops<BE> for Module<BE>
where
    Module<BE>: GLWECopy<BE> + GLWEShift<BE>,
    for<'a> ScratchArena<'a, BE>: ScratchArenaTakeCore<'a, BE>,
{
    fn ckks_mul_pow2_tmp_bytes(&self) -> usize {
        BE::ckks_mul_pow2_tmp_bytes(self)
    }

    fn ckks_mul_pow2_into<Dst, Src>(
        &self,
        dst: &mut Dst,
        src: &Src,
        bits: usize,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        Dst: GLWEToBackendMut<BE> + CKKSCtBounds + SetCKKSInfos,
        Src: GLWEToBackendRef<BE> + CKKSCtBounds,
    {
        BE::ckks_mul_pow2_into(self, dst, src, bits, scratch)
    }

    fn ckks_mul_pow2_assign<Dst>(&self, dst: &mut Dst, bits: usize, scratch: &mut ScratchArena<'_, BE>) -> Result<()>
    where
        Dst: GLWEToBackendMut<BE> + CKKSCtBounds + SetCKKSInfos,
    {
        BE::ckks_mul_pow2_assign(self, dst, bits, scratch)
    }

    fn ckks_div_pow2_tmp_bytes(&self) -> usize {
        BE::ckks_div_pow2_tmp_bytes(self)
    }

    fn ckks_div_pow2_into<Dst, Src>(
        &self,
        dst: &mut Dst,
        src: &Src,
        bits: usize,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        Dst: GLWEToBackendMut<BE> + CKKSCtBounds + SetCKKSInfos,
        Src: GLWEToBackendRef<BE> + CKKSCtBounds,
    {
        BE::ckks_div_pow2_into(self, dst, src, bits, scratch)
    }

    fn ckks_div_pow2_assign<Dst>(&self, dst: &mut Dst, bits: usize) -> Result<()>
    where
        Dst: GLWEToBackendMut<BE> + CKKSCtBounds + SetCKKSInfos,
    {
        BE::ckks_div_pow2_assign(self, dst, bits)
    }
}
