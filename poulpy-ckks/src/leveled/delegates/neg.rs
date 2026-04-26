use anyhow::Result;
use poulpy_core::{GLWENegate, GLWEShift, ScratchTakeCore};
use poulpy_hal::{
    api::ScratchAvailable,
    layouts::{Backend, DataMut, DataRef, Module, Scratch},
};

use crate::{layouts::CKKSCiphertext, oep::CKKSImpl};

use crate::leveled::{api::CKKSNegOps, oep::CKKSNegOep};

impl<BE: Backend + CKKSImpl<BE>> CKKSNegOps<BE> for Module<BE> {
    fn ckks_neg_tmp_bytes(&self) -> usize
    where
        Self: GLWEShift<BE>,
    {
        CKKSNegOep::ckks_neg_tmp_bytes(self)
    }

    fn ckks_neg_into(
        &self,
        dst: &mut CKKSCiphertext<impl DataMut>,
        src: &CKKSCiphertext<impl DataRef>,
        scratch: &mut Scratch<BE>,
    ) -> Result<()>
    where
        Self: GLWENegate + GLWEShift<BE>,
        Scratch<BE>: ScratchAvailable + ScratchTakeCore<BE>,
    {
        CKKSNegOep::ckks_neg_into(self, dst, src, scratch)
    }

    fn ckks_neg_assign(&self, dst: &mut CKKSCiphertext<impl DataMut>) -> Result<()>
    where
        Self: GLWENegate,
    {
        CKKSNegOep::ckks_neg_assign(self, dst)
    }
}
