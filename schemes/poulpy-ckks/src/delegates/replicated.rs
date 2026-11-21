use crate::CKKSResult as Result;
use poulpy_core::layouts::{GGLWEInfos, GLWEToBackendMut, GLWEToBackendRef, GetAutomorphismKey, GetTensorKey};
use poulpy_hal::layouts::{Backend, Module, Ring, ScratchArena};

use crate::{CKKSCtBounds, SetCKKSInfos, api::CKKSReplicatedOps, layouts::ReplicatedConversion, oep::CKKSReplicatedImpl};

impl<BE: Backend + CKKSReplicatedImpl<Rg>, Rg: Ring> CKKSReplicatedOps<BE, Rg> for Module<BE> {
    fn ckks_replicated_conversion_tmp_bytes<R, A, K>(
        &self,
        res: &R,
        a: &A,
        key_infos: &K,
        ctx: &ReplicatedConversion<BE, Rg>,
    ) -> usize
    where
        R: CKKSCtBounds,
        A: CKKSCtBounds,
        K: GGLWEInfos,
    {
        <BE as CKKSReplicatedImpl<Rg>>::ckks_replicated_conversion_tmp_bytes(self, res, a, key_infos, ctx)
    }

    fn ckks_to_replicated<R, A, H>(
        &self,
        res: &mut R,
        a: &A,
        ctx: &ReplicatedConversion<BE, Rg>,
        keys: &H,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        R: GLWEToBackendMut<BE, Rg> + CKKSCtBounds + SetCKKSInfos,
        A: GLWEToBackendRef<BE, Rg> + CKKSCtBounds,
        H: GetAutomorphismKey<BE, Rg> + GetTensorKey<BE, Rg>,
    {
        <BE as CKKSReplicatedImpl<Rg>>::ckks_to_replicated(self, res, a, ctx, keys, scratch)
    }

    fn ckks_from_replicated<R, A, H>(
        &self,
        res: &mut R,
        a: &A,
        ctx: &ReplicatedConversion<BE, Rg>,
        keys: &H,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        R: GLWEToBackendMut<BE, Rg> + CKKSCtBounds + SetCKKSInfos,
        A: GLWEToBackendRef<BE, Rg> + CKKSCtBounds,
        H: GetAutomorphismKey<BE, Rg> + GetTensorKey<BE, Rg>,
    {
        <BE as CKKSReplicatedImpl<Rg>>::ckks_from_replicated(self, res, a, ctx, keys, scratch)
    }
}
