use crate::CKKSResult as Result;
use crate::default::rotate::CKKSRotateDefault;

use poulpy_core::{
    GLWEAutomorphism, GLWEShift,
    layouts::{GGLWEInfos, GLWEInfos, GLWEToBackendMut, GLWEToBackendRef, GetGaloisElement, prepared::GGLWEPreparedToBackendRef},
};
use poulpy_hal::{
    layouts::{Backend, Module, ScratchArena},
    oep::HalVecZnxImpl,
};

use crate::{CKKSCtBounds, SetCKKSInfos};

/// # Safety
///
/// Implementations must satisfy the contracts of all trait methods, including
/// any HAL-level invariants (alignment, layout, scratch sizing) implied by the
/// associated method signatures.
pub unsafe trait CKKSRotateImpl<BE: Backend>: Backend {
    fn ckks_rotate_tmp_bytes_impl<C: GLWEInfos, K: GGLWEInfos>(module: &Module<BE>, ct_infos: &C, key_infos: &K) -> usize;

    fn ckks_rotate_into_impl<Dst, Src, K>(
        module: &Module<BE>,
        dst: &mut Dst,
        src: &Src,
        key: &K,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        Dst: GLWEToBackendMut<BE> + GLWEInfos + CKKSCtBounds + SetCKKSInfos,
        Src: GLWEToBackendRef<BE> + GLWEInfos + CKKSCtBounds,
        K: GGLWEPreparedToBackendRef<BE> + GetGaloisElement + GGLWEInfos;

    fn ckks_rotate_assign_impl<Dst, K>(
        module: &Module<BE>,
        dst: &mut Dst,
        key: &K,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        Dst: GLWEToBackendMut<BE> + GLWEInfos + CKKSCtBounds + SetCKKSInfos,
        K: GGLWEPreparedToBackendRef<BE> + GetGaloisElement + GGLWEInfos;
}

unsafe impl<BE: Backend> CKKSRotateImpl<BE> for BE
where
    BE: Backend + HalVecZnxImpl<BE>,
    Module<BE>: CKKSRotateDefault<BE> + GLWEAutomorphism<BE> + GLWEShift<BE>,
{
    fn ckks_rotate_tmp_bytes_impl<C: GLWEInfos, K: GGLWEInfos>(module: &Module<BE>, ct_infos: &C, key_infos: &K) -> usize {
        module.ckks_rotate_tmp_bytes_default(ct_infos, key_infos)
    }

    fn ckks_rotate_into_impl<Dst, Src, K>(
        module: &Module<BE>,
        dst: &mut Dst,
        src: &Src,
        key: &K,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        Dst: GLWEToBackendMut<BE> + GLWEInfos + CKKSCtBounds + SetCKKSInfos,
        Src: GLWEToBackendRef<BE> + GLWEInfos + CKKSCtBounds,
        K: GGLWEPreparedToBackendRef<BE> + GetGaloisElement + GGLWEInfos,
    {
        module.ckks_rotate_into_default(dst, src, key, scratch)
    }

    fn ckks_rotate_assign_impl<Dst, K>(
        module: &Module<BE>,
        dst: &mut Dst,
        key: &K,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        Dst: GLWEToBackendMut<BE> + GLWEInfos + CKKSCtBounds + SetCKKSInfos,
        K: GGLWEPreparedToBackendRef<BE> + GetGaloisElement + GGLWEInfos,
    {
        module.ckks_rotate_assign_default(dst, key, scratch)
    }
}

#[macro_export]
macro_rules! impl_ckks_rotate_defaults {
    ($be:ty) => {
        impl $crate::default::rotate::CKKSRotateDefault<$be> for ::poulpy_hal::layouts::Module<$be> {}
    };
}
pub use crate::impl_ckks_rotate_defaults;
