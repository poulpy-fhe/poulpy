use crate::CKKSResult as Result;
use poulpy_core::layouts::{
    GGLWEInfos, GGLWEPreparedToBackendRef, GLWECIEmbedKeyPrepared, GLWECITraceKeyPrepared, GLWEToBackendMut, GLWEToBackendRef,
    GetAutomorphismKey,
};
use poulpy_hal::layouts::{Backend, ConjugateInvariant, Data, Module, Ring, ScratchArena};

use crate::{
    CKKSCtBounds, SetCKKSInfos,
    api::{CKKSReplicatedBootstrapOps, CKKSReplicatedCIBootstrapOps, CKKSReplicatedOps},
    layouts::{
        BootstrappingKeys, ReplicatedBootstrapWorkspace, ReplicatedBootstrapping, ReplicatedConversion,
        ReplicatedTransformWorkspace,
    },
    oep::{CKKSReplicatedBootstrapImpl, CKKSReplicatedCIBootstrapImpl, CKKSReplicatedImpl},
};

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
        workspace: &mut ReplicatedTransformWorkspace<BE, Rg>,
        keys: &H,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        R: GLWEToBackendMut<BE, Rg> + CKKSCtBounds + SetCKKSInfos,
        A: GLWEToBackendRef<BE, Rg> + CKKSCtBounds,
        H: GetAutomorphismKey<BE, Rg>,
    {
        <BE as CKKSReplicatedImpl<Rg>>::ckks_to_replicated(self, res, a, ctx, workspace, keys, scratch)
    }

    fn ckks_from_replicated<R, A, H>(
        &self,
        res: &mut R,
        a: &A,
        ctx: &ReplicatedConversion<BE, Rg>,
        workspace: &mut ReplicatedTransformWorkspace<BE, Rg>,
        keys: &H,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        R: GLWEToBackendMut<BE, Rg> + CKKSCtBounds + SetCKKSInfos,
        A: GLWEToBackendRef<BE, Rg> + CKKSCtBounds,
        H: GetAutomorphismKey<BE, Rg>,
    {
        <BE as CKKSReplicatedImpl<Rg>>::ckks_from_replicated(self, res, a, ctx, workspace, keys, scratch)
    }
}

impl<BE: Backend + CKKSReplicatedBootstrapImpl> CKKSReplicatedBootstrapOps<BE> for Module<BE> {
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
    fn ckks_replicated_ci_bootstrap_tmp_bytes<C, I, F, A, T, S, K>(
        &self,
        ct_out: &C,
        ct_in: &I,
        ctx: &ReplicatedBootstrapping<BE, F>,
        atk_infos: &A,
        tsk_infos: &T,
        encapsulation_infos: Option<(&S, &S)>,
        conversion_infos: &K,
    ) -> usize
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
        dst: &mut Dst,
        src: &Src,
        ctx: &ReplicatedBootstrapping<BE, F>,
        workspace: &mut ReplicatedBootstrapWorkspace<BE>,
        keys: &K,
        conversion_keys: (&GLWECIEmbedKeyPrepared<E, BE>, &GLWECITraceKeyPrepared<T, BE>),
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        Dst: GLWEToBackendMut<BE, ConjugateInvariant> + CKKSCtBounds + SetCKKSInfos,
        Src: GLWEToBackendRef<BE, ConjugateInvariant> + CKKSCtBounds,
        K: BootstrappingKeys<BE>,
        E: Data,
        T: Data,
        GLWECIEmbedKeyPrepared<E, BE>: GGLWEPreparedToBackendRef<BE>,
        GLWECITraceKeyPrepared<T, BE>: GGLWEPreparedToBackendRef<BE>,
    {
        BE::ckks_replicated_ci_bootstrap(self, dst, src, ctx, workspace, keys, conversion_keys, scratch)
    }
}
