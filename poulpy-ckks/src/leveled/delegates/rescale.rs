use anyhow::Result;
use poulpy_core::{GLWEShift, ScratchTakeCore};
use poulpy_hal::layouts::{Backend, DataMut, DataRef, Module, Scratch};

use crate::{
    layouts::CKKSCiphertext,
    leveled::{api::CKKSRescaleOps, default::CKKSRescaleOpsDefault},
};

impl<BE: Backend> CKKSRescaleOps<BE> for Module<BE>
where
    Module<BE>: CKKSRescaleOpsDefault<BE>,
{
    fn ckks_rescale_tmp_bytes(&self) -> usize
    where
        Self: GLWEShift<BE>,
    {
        self.ckks_rescale_tmp_bytes_default()
    }

    fn ckks_rescale_assign(&self, ct: &mut CKKSCiphertext<impl DataMut>, k: usize, scratch: &mut Scratch<BE>) -> Result<()>
    where
        Self: GLWEShift<BE>,
        Scratch<BE>: ScratchTakeCore<BE>,
    {
        self.ckks_rescale_assign_default(ct, k, scratch)
    }

    fn ckks_rescale_into(
        &self,
        dst: &mut CKKSCiphertext<impl DataMut>,
        k: usize,
        src: &CKKSCiphertext<impl DataRef>,
        scratch: &mut Scratch<BE>,
    ) -> Result<()>
    where
        Self: GLWEShift<BE>,
        Scratch<BE>: ScratchTakeCore<BE>,
    {
        self.ckks_rescale_into_default(dst, k, src, scratch)
    }

    fn ckks_align_assign(
        &self,
        a: &mut CKKSCiphertext<impl DataMut>,
        b: &mut CKKSCiphertext<impl DataMut>,
        scratch: &mut Scratch<BE>,
    ) -> Result<()>
    where
        Self: GLWEShift<BE>,
        Scratch<BE>: ScratchTakeCore<BE>,
    {
        self.ckks_align_assign_default(a, b, scratch)
    }

    fn ckks_align_tmp_bytes(&self) -> usize
    where
        Self: GLWEShift<BE>,
    {
        self.ckks_align_tmp_bytes_default()
    }
}
