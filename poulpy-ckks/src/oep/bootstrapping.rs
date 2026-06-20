use crate::default::bootstrapping::CKKSBootstrappingOpsDefault;

use anyhow::Result;
use poulpy_core::{
    GLWECopy, GLWEShift,
    layouts::{GLWEInfos, LWEInfos},
};
use poulpy_hal::layouts::{Backend, Module, ScratchArena};

use crate::{CKKSInfos, GLWEToBackendMut, GLWEToBackendRef, SetCKKSInfos};

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
}

unsafe impl<BE: Backend> CKKSBootstrappingImpl<BE> for BE
where
    Module<BE>: CKKSBootstrappingOpsDefault<BE> + GLWECopy<BE> + GLWEShift<BE>,
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
}

#[macro_export]
macro_rules! impl_ckks_bootstrapping_defaults {
    ($be:ty) => {
        impl $crate::default::bootstrapping::CKKSBootstrappingOpsDefault<$be> for ::poulpy_hal::layouts::Module<$be> {}
    };
}
pub use crate::impl_ckks_bootstrapping_defaults;
