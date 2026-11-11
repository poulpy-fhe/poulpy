use crate::CKKSResult as Result;
use poulpy_core::layouts::{GLWEToBackendMut, GLWEToBackendRef};
use poulpy_hal::layouts::{Backend, Module, Ring, ScratchArena};

use crate::{CKKSCtBounds, SetCKKSInfos, api::CKKSRoundOps, oep::CKKSRoundImpl};

impl<BE: Backend + CKKSRoundImpl<Rg>, Rg: Ring> CKKSRoundOps<BE, Rg> for Module<BE> {
    fn ckks_round_tmp_bytes<R, A>(&self, res: &R, a: &A) -> usize
    where
        R: CKKSCtBounds,
        A: CKKSCtBounds,
    {
        BE::ckks_round_tmp_bytes(self, res, a)
    }

    fn ckks_round<R, A>(&self, res: &mut R, a: &A, scratch: &mut ScratchArena<'_, BE>) -> Result<()>
    where
        R: GLWEToBackendMut<BE, Rg> + CKKSCtBounds + SetCKKSInfos,
        A: GLWEToBackendRef<BE, Rg> + CKKSCtBounds,
    {
        BE::ckks_round(self, res, a, scratch)
    }
}
