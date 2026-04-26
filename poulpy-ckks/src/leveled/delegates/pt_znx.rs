use anyhow::Result;
use poulpy_core::{ScratchTakeCore, layouts::GLWEPlaintext};
use poulpy_hal::{
    api::{VecZnxLsh, VecZnxLshTmpBytes, VecZnxRsh, VecZnxRshTmpBytes},
    layouts::{Backend, DataMut, DataRef, Module, Scratch},
};

use crate::{CKKSInfos, layouts::CKKSPlaintextVecZnx, oep::CKKSImpl};

use crate::leveled::{api::CKKSPlaintextZnxOps, oep::CKKSPlaintextZnxOep};

impl<BE: Backend + CKKSImpl<BE>> CKKSPlaintextZnxOps<BE> for Module<BE> {
    fn ckks_extract_pt_znx_tmp_bytes(&self) -> usize
    where
        Self: VecZnxLshTmpBytes + VecZnxRshTmpBytes,
    {
        CKKSPlaintextZnxOep::ckks_extract_pt_znx_tmp_bytes(self)
    }

    fn ckks_extract_pt_znx<S>(
        &self,
        dst: &mut CKKSPlaintextVecZnx<impl DataMut>,
        src: &GLWEPlaintext<impl DataRef>,
        src_meta: &S,
        scratch: &mut Scratch<BE>,
    ) -> Result<()>
    where
        S: CKKSInfos,
        Scratch<BE>: ScratchTakeCore<BE>,
        Self: VecZnxLsh<BE> + VecZnxRsh<BE>,
    {
        CKKSPlaintextZnxOep::ckks_extract_pt_znx(self, dst, src, src_meta, scratch)
    }
}
