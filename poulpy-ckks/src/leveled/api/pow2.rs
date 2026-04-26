use anyhow::Result;
use poulpy_core::{GLWECopy, GLWEShift, ScratchTakeCore};
use poulpy_hal::layouts::{Backend, DataMut, DataRef, Scratch};

use crate::{layouts::CKKSCiphertext, oep::CKKSImpl};

pub trait CKKSPow2Ops<BE: Backend + CKKSImpl<BE>> {
    fn ckks_mul_pow2_tmp_bytes(&self) -> usize
    where
        Self: GLWEShift<BE>;

    fn ckks_mul_pow2_into(
        &self,
        dst: &mut CKKSCiphertext<impl DataMut>,
        src: &CKKSCiphertext<impl DataRef>,
        bits: usize,
        scratch: &mut Scratch<BE>,
    ) -> Result<()>
    where
        Self: GLWEShift<BE>,
        Scratch<BE>: ScratchTakeCore<BE>;

    fn ckks_mul_pow2_assign(&self, dst: &mut CKKSCiphertext<impl DataMut>, bits: usize, scratch: &mut Scratch<BE>) -> Result<()>
    where
        Self: GLWEShift<BE>,
        Scratch<BE>: ScratchTakeCore<BE>;

    fn ckks_div_pow2_tmp_bytes(&self) -> usize
    where
        Self: GLWEShift<BE>;

    fn ckks_div_pow2_into(
        &self,
        dst: &mut CKKSCiphertext<impl DataMut>,
        src: &CKKSCiphertext<impl DataRef>,
        bits: usize,
        scratch: &mut Scratch<BE>,
    ) -> Result<()>
    where
        Self: GLWEShift<BE> + GLWECopy,
        Scratch<BE>: ScratchTakeCore<BE>;

    fn ckks_div_pow2_assign(&self, dst: &mut CKKSCiphertext<impl DataMut>, bits: usize) -> Result<()>;
}
