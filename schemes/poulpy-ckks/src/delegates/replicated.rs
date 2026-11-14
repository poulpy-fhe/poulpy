use crate::CKKSResult as Result;
use poulpy_core::layouts::{
    GGLWEInfos, GGLWEPreparedToBackendRef, GLWECIEmbedKeyPrepared, GLWECITraceKeyPrepared, GLWEToBackendMut, GLWEToBackendRef,
    GetAutomorphismKey, GetTensorKey,
};
use poulpy_hal::layouts::{Backend, ConjugateInvariant, Data, Module, Ring, ScratchArena};

use crate::{
    CKKSCtBounds, SetCKKSInfos,
    api::{CKKSReplicatedBootstrapOps, CKKSReplicatedCIBootstrapOps, CKKSReplicatedOps},
    layouts::{
        BootstrappingKeys, BootstrappingKeysLayout, ReplicatedBootstrapWorkspace, ReplicatedBootstrapping, ReplicatedConversion,
        ReplicatedLut, ReplicatedTransformWorkspace,
    },
    oep::{CKKSReplicatedBootstrapImpl, CKKSReplicatedCIBootstrapImpl, CKKSReplicatedImpl},
};

impl<BE: Backend + CKKSReplicatedImpl<Rg>, Rg: Ring> CKKSReplicatedOps<BE, Rg> for Module<BE> {
    fn ckks_replicated_conversion_tmp_bytes<C, K>(
        &self,
        ct_infos: &C,
        key_infos: &K,
        conv: &ReplicatedConversion<BE, Rg>,
    ) -> usize
    where
        C: CKKSCtBounds,
        K: GGLWEInfos,
    {
        <BE as CKKSReplicatedImpl<Rg>>::ckks_replicated_conversion_tmp_bytes(self, ct_infos, key_infos, conv)
    }

    fn ckks_to_replicated<Dst, Src, H>(
        &self,
        dst: &mut Dst,
        src: &Src,
        conv: &ReplicatedConversion<BE, Rg>,
        workspace: &mut ReplicatedTransformWorkspace<BE, Rg>,
        keys: &H,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        Dst: GLWEToBackendMut<BE, Rg> + CKKSCtBounds + SetCKKSInfos,
        Src: GLWEToBackendRef<BE, Rg> + CKKSCtBounds,
        H: GetAutomorphismKey<BE, Rg> + GetTensorKey<BE, Rg>,
    {
        <BE as CKKSReplicatedImpl<Rg>>::ckks_to_replicated(self, dst, src, conv, workspace, keys, scratch)
    }

    fn ckks_from_replicated<Dst, Src, H>(
        &self,
        dst: &mut Dst,
        src: &Src,
        conv: &ReplicatedConversion<BE, Rg>,
        workspace: &mut ReplicatedTransformWorkspace<BE, Rg>,
        keys: &H,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        Dst: GLWEToBackendMut<BE, Rg> + CKKSCtBounds + SetCKKSInfos,
        Src: GLWEToBackendRef<BE, Rg> + CKKSCtBounds,
        H: GetAutomorphismKey<BE, Rg> + GetTensorKey<BE, Rg>,
    {
        <BE as CKKSReplicatedImpl<Rg>>::ckks_from_replicated(self, dst, src, conv, workspace, keys, scratch)
    }
}

impl<BE: Backend + CKKSReplicatedBootstrapImpl> CKKSReplicatedBootstrapOps<BE> for Module<BE> {
    fn ckks_replicated_bootstrap_tmp_bytes<C, I, F>(
        &self,
        ct_out: &C,
        ct_in: &I,
        ctx: &ReplicatedBootstrapping<BE, F>,
        keys_layout: &BootstrappingKeysLayout,
    ) -> usize
    where
        C: CKKSCtBounds,
        I: CKKSCtBounds,
    {
        BE::ckks_replicated_bootstrap_tmp_bytes(self, ct_out, ct_in, ctx, keys_layout)
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

    fn ckks_replicated_functional_bootstrap_tmp_bytes<C, I, F>(
        &self,
        ct_out: &C,
        ct_in: &I,
        ctx: &ReplicatedBootstrapping<BE, F>,
        luts: &[ReplicatedLut<BE>],
        keys_layout: &BootstrappingKeysLayout,
    ) -> usize
    where
        C: CKKSCtBounds,
        I: CKKSCtBounds,
    {
        BE::ckks_replicated_functional_bootstrap_tmp_bytes(self, ct_out, ct_in, ctx, luts, keys_layout)
    }

    fn ckks_replicated_functional_bootstrap<Dst, Src, F, K>(
        &self,
        dst: &mut [Dst],
        src: &Src,
        ctx: &ReplicatedBootstrapping<BE, F>,
        luts: &[ReplicatedLut<BE>],
        workspace: &mut ReplicatedBootstrapWorkspace<BE>,
        keys: &K,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        Dst: GLWEToBackendMut<BE> + GLWEToBackendRef<BE> + CKKSCtBounds + SetCKKSInfos,
        Src: GLWEToBackendRef<BE> + CKKSCtBounds,
        K: BootstrappingKeys<BE>,
    {
        BE::ckks_replicated_functional_bootstrap(self, dst, src, ctx, luts, workspace, keys, scratch)
    }
}

impl<BE: Backend + CKKSReplicatedCIBootstrapImpl> CKKSReplicatedCIBootstrapOps<BE> for Module<BE> {
    fn ckks_replicated_ci_bootstrap_tmp_bytes<C, I, F, K>(
        &self,
        ct_out: &C,
        ct_in: &I,
        ctx: &ReplicatedBootstrapping<BE, F>,
        keys_layout: &BootstrappingKeysLayout,
        conversion_infos: &K,
    ) -> usize
    where
        C: CKKSCtBounds,
        I: CKKSCtBounds,
        K: GGLWEInfos,
    {
        BE::ckks_replicated_ci_bootstrap_tmp_bytes(self, ct_out, ct_in, ctx, keys_layout, conversion_infos)
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
