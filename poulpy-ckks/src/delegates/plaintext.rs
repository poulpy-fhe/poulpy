use anyhow::Result;
use poulpy_core::{
    ScratchArenaTakeCore,
    layouts::{GLWEInfos, GLWEToBackendRef, LWEInfos},
};
use poulpy_hal::{
    api::{VecZnxLshBackend, VecZnxLshTmpBytes, VecZnxRshBackend, VecZnxRshTmpBytes},
    layouts::{Backend, Module, ScratchArena},
};

use crate::GLWEToBackendMut;

use crate::{CKKSInfos, SetCKKSInfos, oep::CKKSPlaintextZnxImpl};

use crate::api::CKKSPlaintextVecOps;

impl<BE: Backend + CKKSPlaintextZnxImpl<BE>> CKKSPlaintextVecOps<BE> for Module<BE>
where
    BE: poulpy_hal::oep::HalVecZnxImpl<BE>,
    Module<BE>: VecZnxLshBackend<BE> + VecZnxLshTmpBytes + VecZnxRshBackend<BE> + VecZnxRshTmpBytes,
    for<'a> ScratchArena<'a, BE>: ScratchArenaTakeCore<'a, BE>,
{
    fn ckks_extract_pt_tmp_bytes(&self) -> usize {
        BE::ckks_extract_pt_tmp_bytes(self)
    }

    fn ckks_extract_pt<D, S>(&self, dst: &mut D, src: &S, scratch: &mut ScratchArena<'_, BE>) -> Result<()>
    where
        D: GLWEToBackendMut<BE> + GLWEInfos + CKKSInfos + SetCKKSInfos + LWEInfos,
        S: GLWEToBackendRef<BE> + GLWEInfos + LWEInfos + CKKSInfos,
    {
        BE::ckks_extract_pt(self, dst, src, scratch)
    }
}
