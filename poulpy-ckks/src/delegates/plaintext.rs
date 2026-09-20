use crate::CKKSResult as Result;
use poulpy_core::layouts::IntPolyInfos;
use poulpy_core::layouts::{GLWEInfos, GLWEToBackendRef};
use poulpy_hal::{
    api::{VecZnxLsh, VecZnxLshTmpBytes, VecZnxRsh, VecZnxRshTmpBytes},
    layouts::{Backend, Module, ScratchArena},
};

use crate::GLWEToBackendMut;

use crate::{CKKSInfos, SetCKKSInfos, oep::CKKSPlaintextZnxImpl};

use crate::api::CKKSPlaintextVecOps;

impl<BE: Backend + CKKSPlaintextZnxImpl> CKKSPlaintextVecOps<BE> for Module<BE>
where
    BE: poulpy_hal::oep::HalVecZnxImpl,
    Module<BE>: VecZnxLsh<BE> + VecZnxLshTmpBytes + VecZnxRsh<BE> + VecZnxRshTmpBytes,
{
    fn ckks_extract_pt_tmp_bytes(&self, res_size: usize) -> usize {
        BE::ckks_extract_pt_tmp_bytes_impl(self, res_size)
    }

    fn ckks_extract_pt<D, S>(&self, dst: &mut D, src: &S, scratch: &mut ScratchArena<'_, BE>) -> Result<()>
    where
        D: GLWEToBackendMut<BE> + GLWEInfos + CKKSInfos + SetCKKSInfos + IntPolyInfos,
        S: GLWEToBackendRef<BE> + GLWEInfos + CKKSInfos,
    {
        crate::api::CKKSModuleInfos::ckks_ring(self).check_plaintext("ckks_extract_pt", dst)?;
        crate::api::CKKSModuleInfos::ckks_ring(self).check_ciphertext("ckks_extract_pt", src)?;
        BE::ckks_extract_pt_impl(self, dst, src, scratch)
    }
}
