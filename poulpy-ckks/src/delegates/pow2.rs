use crate::CKKSResult as Result;
use poulpy_core::layouts::{GLWEToBackendMut, GLWEToBackendRef};
use poulpy_hal::layouts::{Backend, Module, ScratchArena};

use crate::{CKKSCtBounds, SetCKKSInfos, oep::CKKSPow2Impl};

use crate::api::CKKSPow2Ops;

impl<BE: Backend + CKKSPow2Impl> CKKSPow2Ops<BE> for Module<BE> {
    fn ckks_mul_pow2_tmp_bytes(&self, res_size: usize) -> usize {
        BE::ckks_mul_pow2_tmp_bytes_impl(self, res_size)
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
        crate::api::CKKSModuleInfos::ckks_ring(self).check_ciphertext("ckks_mul_pow2_into", dst)?;
        crate::api::CKKSModuleInfos::ckks_ring(self).check_ciphertext("ckks_mul_pow2_into", src)?;
        BE::ckks_mul_pow2_into_impl(self, dst, src, bits, scratch)
    }

    fn ckks_mul_pow2_assign<Dst>(&self, dst: &mut Dst, bits: usize, scratch: &mut ScratchArena<'_, BE>) -> Result<()>
    where
        Dst: GLWEToBackendMut<BE> + CKKSCtBounds + SetCKKSInfos,
    {
        crate::api::CKKSModuleInfos::ckks_ring(self).check_ciphertext("ckks_mul_pow2_assign", dst)?;
        BE::ckks_mul_pow2_assign_impl(self, dst, bits, scratch)
    }

    fn ckks_div_pow2_tmp_bytes(&self, res_size: usize) -> usize {
        BE::ckks_div_pow2_tmp_bytes_impl(self, res_size)
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
        crate::api::CKKSModuleInfos::ckks_ring(self).check_ciphertext("ckks_div_pow2_into", dst)?;
        crate::api::CKKSModuleInfos::ckks_ring(self).check_ciphertext("ckks_div_pow2_into", src)?;
        BE::ckks_div_pow2_into_impl(self, dst, src, bits, scratch)
    }

    fn ckks_div_pow2_assign<Dst>(&self, dst: &mut Dst, bits: usize) -> Result<()>
    where
        Dst: GLWEToBackendMut<BE> + CKKSCtBounds + SetCKKSInfos,
    {
        crate::api::CKKSModuleInfos::ckks_ring(self).check_ciphertext("ckks_div_pow2_assign", dst)?;
        BE::ckks_div_pow2_assign_impl(self, dst, bits)
    }
}
