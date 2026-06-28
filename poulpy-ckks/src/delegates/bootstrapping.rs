use anyhow::Result;
use poulpy_core::{
    GLWECopy, GLWEShift,
    layouts::{GLWETensorKeyPrepared, GLWEToBackendMut, GLWEToBackendRef},
};
use poulpy_hal::layouts::{Backend, HostBytesBackend, Module, ScratchArena, TransferFrom};

use crate::{
    CKKSCtBounds, SetCKKSInfos,
    api::{CKKSBootstrappingOps, CKKSEvalModOps, DFTOps},
    layouts::{BootstrappingContext, BootstrappingKeys, CKKSCiphertext},
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

    fn ckks_bootstrap<F, K>(
        &self,
        ct_out: &mut CKKSCiphertext<BE::OwnedBuf>,
        ct_in: &CKKSCiphertext<BE::OwnedBuf>,
        ctx: &BootstrappingContext<BE, F>,
        keys: &K,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        BE: TransferFrom<HostBytesBackend>,
        K: BootstrappingKeys<BE, TensorKey = GLWETensorKeyPrepared<BE::OwnedBuf, BE>>,
    {
        BE::ckks_bootstrap::<F, K>(self, ct_out, ct_in, ctx, keys, scratch)
    }
}
