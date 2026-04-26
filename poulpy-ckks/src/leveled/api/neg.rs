use anyhow::Result;
use poulpy_core::{GLWENegate, GLWEShift, ScratchTakeCore};
use poulpy_hal::{
    api::ScratchAvailable,
    layouts::{Backend, DataMut, DataRef, Scratch},
};

use crate::{layouts::CKKSCiphertext, oep::CKKSImpl};

pub trait CKKSNegOps<BE: Backend + CKKSImpl<BE>> {
    fn ckks_neg_tmp_bytes(&self) -> usize
    where
        Self: GLWEShift<BE>;

    fn ckks_neg_into(
        &self,
        dst: &mut CKKSCiphertext<impl DataMut>,
        src: &CKKSCiphertext<impl DataRef>,
        scratch: &mut Scratch<BE>,
    ) -> Result<()>
    where
        Self: GLWENegate + GLWEShift<BE>,
        Scratch<BE>: ScratchAvailable + ScratchTakeCore<BE>;

    fn ckks_neg_assign(&self, dst: &mut CKKSCiphertext<impl DataMut>) -> Result<()>
    where
        Self: GLWENegate;
}
