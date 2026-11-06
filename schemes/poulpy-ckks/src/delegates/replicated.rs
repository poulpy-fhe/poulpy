use crate::CKKSResult as Result;
use poulpy_core::layouts::{
    GGLWEInfos, GGLWEPreparedToBackendRef, GLWECIEmbedKeyPrepared, GLWECITraceKeyPrepared, GLWEToBackendMut, GLWEToBackendRef,
    GetAutomorphismKey, GetTensorKey,
};
use poulpy_hal::layouts::{Backend, Data, Module, ScratchArena, Standard};

use crate::{
    CKKSCtBounds, SetCKKSInfos,
    api::{CKKSReplicatedBootstrapOps, CKKSReplicatedCIBootstrapOps, CKKSReplicatedOps},
    layouts::{
        BootstrappingKeys, ReplicatedBootstrapWorkspace, ReplicatedBootstrapping, ReplicatedConversion,
        ReplicatedTransformWorkspace,
    },
    oep::{CKKSReplicatedBootstrapImpl, CKKSReplicatedCIBootstrapImpl, CKKSReplicatedImpl},
};

impl<BE: Backend + CKKSReplicatedImpl> CKKSReplicatedOps<BE> for Module<BE> {
    fn ckks_replicated_conversion_tmp_bytes<C, K>(&self, ct_infos: &C, key_infos: &K, conv: &ReplicatedConversion<BE>) -> usize
    where
        C: CKKSCtBounds,
        K: GGLWEInfos,
    {
        BE::ckks_replicated_conversion_tmp_bytes(self, ct_infos, key_infos, conv)
    }

    fn ckks_to_replicated<Dst, Src, H>(
        &self,
        dst: &mut Dst,
        src: &Src,
        conv: &ReplicatedConversion<BE>,
        workspace: &mut ReplicatedTransformWorkspace<BE>,
        keys: &H,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        Dst: GLWEToBackendMut<BE> + CKKSCtBounds + SetCKKSInfos,
        Src: GLWEToBackendRef<BE> + CKKSCtBounds,
        H: GetAutomorphismKey<BE> + GetTensorKey<BE>,
    {
        BE::ckks_to_replicated(self, dst, src, conv, workspace, keys, scratch)
    }

    fn ckks_from_replicated<Dst, Src, H>(
        &self,
        dst: &mut Dst,
        src: &Src,
        conv: &ReplicatedConversion<BE>,
        workspace: &mut ReplicatedTransformWorkspace<BE>,
        keys: &H,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        Dst: GLWEToBackendMut<BE> + CKKSCtBounds + SetCKKSInfos,
        Src: GLWEToBackendRef<BE> + CKKSCtBounds,
        H: GetAutomorphismKey<BE> + GetTensorKey<BE>,
    {
        BE::ckks_from_replicated(self, dst, src, conv, workspace, keys, scratch)
    }
}

impl<BE: Backend<Ring = Standard> + CKKSReplicatedBootstrapImpl> CKKSReplicatedBootstrapOps<BE> for Module<BE> {
    fn ckks_replicated_bootstrap_tmp_bytes<C, I, F, A, T, S>(
        &self,
        ct_out: &C,
        ct_in: &I,
        ctx: &ReplicatedBootstrapping<BE, F>,
        atk_infos: &A,
        tsk_infos: &T,
        encapsulation_infos: Option<(&S, &S)>,
    ) -> usize
    where
        C: CKKSCtBounds,
        I: CKKSCtBounds,
        A: GGLWEInfos,
        T: GGLWEInfos,
        S: GGLWEInfos,
    {
        BE::ckks_replicated_bootstrap_tmp_bytes(self, ct_out, ct_in, ctx, atk_infos, tsk_infos, encapsulation_infos)
    }

    fn ckks_replicated_bootstrap<Dst, Src, F, K>(
        &self,
        dst: &mut Dst,
        src: &Src,
        ctx: &ReplicatedBootstrapping<BE, F>,
        workspace: &mut ReplicatedBootstrapWorkspace<BE>,
        keys: &K,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        Dst: GLWEToBackendMut<BE> + GLWEToBackendRef<BE> + CKKSCtBounds + SetCKKSInfos,
        Src: GLWEToBackendRef<BE> + CKKSCtBounds,
        K: BootstrappingKeys<BE>,
    {
        BE::ckks_replicated_bootstrap(self, dst, src, ctx, workspace, keys, scratch)
    }
}

impl<BE: Backend + CKKSReplicatedCIBootstrapImpl> CKKSReplicatedCIBootstrapOps<BE> for Module<BE> {
    type CI = BE::CI;

    fn ckks_replicated_ci_bootstrap_tmp_bytes<C, I, F, A, T, S, K>(
        &self,
        ci: &Module<BE::CI>,
        ct_out: &C,
        ct_in: &I,
        ctx: &ReplicatedBootstrapping<BE, F>,
        atk_infos: &A,
        tsk_infos: &T,
        encapsulation_infos: Option<(&S, &S)>,
        conversion_infos: &K,
    ) -> (usize, usize)
    where
        C: CKKSCtBounds,
        I: CKKSCtBounds,
        A: GGLWEInfos,
        T: GGLWEInfos,
        S: GGLWEInfos,
        K: GGLWEInfos,
    {
        BE::ckks_replicated_ci_bootstrap_tmp_bytes(
            self,
            ci,
            ct_out,
            ct_in,
            ctx,
            atk_infos,
            tsk_infos,
            encapsulation_infos,
            conversion_infos,
        )
    }

    fn ckks_replicated_ci_bootstrap<Dst, Src, F, K, E, T>(
        &self,
        ci: &Module<BE::CI>,
        dst: &mut Dst,
        src: &Src,
        ctx: &ReplicatedBootstrapping<BE, F>,
        workspace: &mut ReplicatedBootstrapWorkspace<BE>,
        keys: &K,
        conversion_keys: (&GLWECIEmbedKeyPrepared<E, BE>, &GLWECITraceKeyPrepared<T, BE>),
        scratch: &mut ScratchArena<'_, BE>,
        scratch_ci: &mut ScratchArena<'_, BE::CI>,
    ) -> Result<()>
    where
        Dst: GLWEToBackendMut<BE::CI> + CKKSCtBounds + SetCKKSInfos,
        Src: GLWEToBackendRef<BE::CI> + CKKSCtBounds,
        K: BootstrappingKeys<BE>,
        E: Data,
        T: Data,
        GLWECIEmbedKeyPrepared<E, BE>: GGLWEPreparedToBackendRef<BE>,
        GLWECITraceKeyPrepared<T, BE>: GGLWEPreparedToBackendRef<BE>,
    {
        BE::ckks_replicated_ci_bootstrap(self, ci, dst, src, ctx, workspace, keys, conversion_keys, scratch, scratch_ci)
    }
}
