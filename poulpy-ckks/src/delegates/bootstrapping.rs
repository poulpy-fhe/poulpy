use anyhow::Result;
use poulpy_core::{
    GLWECopy, GLWEShift,
    layouts::{GLWEToBackendMut, GLWEToBackendRef},
};
use poulpy_hal::layouts::{Backend, Module, ScratchArena};

use crate::{
    CKKSCtBounds, SetCKKSInfos,
    api::{CKKSBootstrappingOps, CKKSEvalModOps, DFTOps},
    oep::CKKSBootstrappingImpl,
};

impl<BE: Backend + CKKSBootstrappingImpl<BE>> CKKSBootstrappingOps<BE> for Module<BE>
where
    Module<BE>: GLWECopy<BE> + GLWEShift<BE> + DFTOps<BE> + CKKSEvalModOps<BE>,
{
    fn ckks_mod_up_tmp_bytes(&self) -> usize {
        BE::ckks_mod_up_tmp_bytes(self)
    }

    fn ckks_mod_up_into<Dst, Src>(&self, dst: &mut Dst, src: &Src, scratch: &mut ScratchArena<'_, BE>) -> Result<()>
    where
        Dst: GLWEToBackendMut<BE> + CKKSCtBounds + SetCKKSInfos,
        Src: GLWEToBackendRef<BE> + CKKSCtBounds,
    {
        BE::ckks_mod_up_into(self, dst, src, scratch)
    }
}
