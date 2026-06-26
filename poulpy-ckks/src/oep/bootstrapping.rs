use crate::default::bootstrapping::CKKSBootstrappingOpsDefault;

use anyhow::Result;
use poulpy_core::{
    GLWECopy, GLWEKeyswitch, GLWEShift,
    layouts::{
        BSGSMeta, Compact, GGLWEInfos, GLWEInfos, GLWETensorKeyPrepared, LWEInfos, SetBSGSMeta,
        prepared::GLWETensorKeyPreparedToBackendRef,
    },
};
use poulpy_hal::layouts::{Backend, HostBytesBackend, Module, ScratchArena, TransferFrom};

use crate::{
    CKKSCtBounds, CKKSInfos, GLWEToBackendMut, GLWEToBackendRef, SetCKKSInfos,
    api::{CKKSAddOps, CKKSCopyOps, CKKSEvalModOps, CKKSPow2Ops, CKKSSubOps, DFTOps},
    layouts::{BootstrappingContext, BootstrappingKeys, CKKSCiphertext, CKKSModuleAlloc},
};

/// Backend override hook for [`CKKSBootstrappingOps`](crate::api::CKKSBootstrappingOps).
///
/// The blanket impl below forwards to the backend-generic reference in
/// [`CKKSBootstrappingOpsDefault`]; a backend may instead provide a specialized
/// implementation by implementing this trait directly.
///
/// # Safety
///
/// Implementations must satisfy the contracts of all trait methods, including
/// any HAL-level invariants (alignment, layout, scratch sizing) implied by the
/// associated method signatures.
pub unsafe trait CKKSBootstrappingImpl<BE: Backend>: Backend {
    fn ckks_mod_up_tmp_bytes(module: &Module<BE>) -> usize;

    fn ckks_mod_up_into<Dst, Src>(
        module: &Module<BE>,
        dst: &mut Dst,
        src: &Src,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        Dst: GLWEToBackendMut<BE> + LWEInfos + CKKSInfos + SetCKKSInfos,
        Src: GLWEToBackendRef<BE> + GLWEInfos + LWEInfos + CKKSInfos;

    /// See [`CKKSBootstrappingOps::ckks_bootstrap`](crate::api::CKKSBootstrappingOps::ckks_bootstrap).
    fn ckks_bootstrap<F, K>(
        module: &Module<BE>,
        ct_out: &mut CKKSCiphertext<BE::OwnedBuf>,
        ct_in: &CKKSCiphertext<BE::OwnedBuf>,
        ctx: &BootstrappingContext<BE, F>,
        keys: &K,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        BE: TransferFrom<HostBytesBackend>,
        K: BootstrappingKeys<BE, TensorKey = GLWETensorKeyPrepared<BE::OwnedBuf, BE>>;
}

unsafe impl<BE: Backend> CKKSBootstrappingImpl<BE> for BE
where
    Module<BE>: CKKSBootstrappingOpsDefault<BE>
        + GLWECopy<BE>
        + GLWEShift<BE>
        + CKKSModuleAlloc<BE>
        + GLWEKeyswitch<BE>
        + CKKSCopyOps<BE>
        + CKKSPow2Ops<BE>
        + CKKSAddOps<BE>
        + CKKSSubOps<BE>
        + DFTOps<BE>
        + CKKSEvalModOps<BE>,
    CKKSCiphertext<BE::OwnedBuf>:
        GLWEToBackendMut<BE> + GLWEToBackendRef<BE> + CKKSCtBounds + SetCKKSInfos + Compact + SetBSGSMeta + BSGSMeta,
    GLWETensorKeyPrepared<BE::OwnedBuf, BE>: GLWETensorKeyPreparedToBackendRef<BE> + GGLWEInfos,
{
    fn ckks_mod_up_tmp_bytes(module: &Module<BE>) -> usize {
        module.ckks_mod_up_tmp_bytes_default()
    }

    fn ckks_mod_up_into<Dst, Src>(module: &Module<BE>, dst: &mut Dst, src: &Src, scratch: &mut ScratchArena<'_, BE>) -> Result<()>
    where
        Dst: GLWEToBackendMut<BE> + LWEInfos + CKKSInfos + SetCKKSInfos,
        Src: GLWEToBackendRef<BE> + GLWEInfos + LWEInfos + CKKSInfos,
    {
        module.ckks_mod_up_into_default(dst, src, scratch)
    }

    fn ckks_bootstrap<F, K>(
        module: &Module<BE>,
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
        module.ckks_bootstrap_default(ct_out, ct_in, ctx, keys, scratch)
    }
}

#[macro_export]
macro_rules! impl_ckks_bootstrapping_defaults {
    ($be:ty) => {
        impl $crate::default::bootstrapping::CKKSBootstrappingOpsDefault<$be> for ::poulpy_hal::layouts::Module<$be> {}
    };
}
pub use crate::impl_ckks_bootstrapping_defaults;
