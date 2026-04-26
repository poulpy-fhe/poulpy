use anyhow::Result;
use poulpy_core::{ScratchTakeCore, layouts::GLWEPlaintext};
use poulpy_hal::{
    api::{VecZnxLsh, VecZnxLshTmpBytes, VecZnxRsh, VecZnxRshTmpBytes},
    layouts::{Backend, DataMut, DataRef, Scratch},
};

use crate::{CKKSInfos, layouts::CKKSPlaintextVecZnx, oep::CKKSImpl};

pub trait CKKSPlaintextZnxOps<BE: Backend + CKKSImpl<BE>> {
    fn ckks_extract_pt_znx_tmp_bytes(&self) -> usize
    where
        Self: VecZnxLshTmpBytes + VecZnxRshTmpBytes;

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
        Self: VecZnxLsh<BE> + VecZnxRsh<BE>;
}
