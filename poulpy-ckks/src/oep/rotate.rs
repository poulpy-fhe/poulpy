use crate::default::rotate::CKKSRotateDefault;

use anyhow::Result;
use poulpy_core::{
    GLWEAutomorphism, GLWEShift, ScratchArenaTakeCore,
    layouts::{
        GGLWEInfos, GLWEInfos, GLWEToBackendMut, GLWEToBackendRef, GetGaloisElement, LWEInfos,
        prepared::GGLWEPreparedToBackendRef,
    },
};
use poulpy_hal::{
    layouts::{Backend, Module, ScratchArena},
    oep::HalVecZnxImpl,
};

use crate::{CKKSInfos, SetCKKSInfos};

/// # Safety
///
/// Implementations must satisfy the contracts of all trait methods, including
/// any HAL-level invariants (alignment, layout, scratch sizing) implied by the
/// associated method signatures.
pub unsafe trait CKKSRotateImpl<BE: Backend>: Backend {
    fn ckks_rotate_tmp_bytes<C: GLWEInfos, K: GGLWEInfos>(module: &Module<BE>, ct_infos: &C, key_infos: &K) -> usize;

    fn ckks_rotate_into<'s, Dst, Src, K>(
        module: &Module<BE>,
        dst: &mut Dst,
        src: &Src,
        key: &K,
        scratch: &mut ScratchArena<'s, BE>,
    ) -> Result<()>
    where
        Dst: GLWEToBackendMut<BE> + GLWEInfos + LWEInfos + CKKSInfos + SetCKKSInfos,
        Src: GLWEToBackendRef<BE> + GLWEInfos + LWEInfos + CKKSInfos,
        K: GetGaloisElement + GGLWEPreparedToBackendRef<BE> + GGLWEInfos,
        BE: 's;

    fn ckks_rotate_assign<'s, Dst, K>(
        module: &Module<BE>,
        dst: &mut Dst,
        key: &K,
        scratch: &mut ScratchArena<'s, BE>,
    ) -> Result<()>
    where
        Dst: GLWEToBackendMut<BE> + GLWEInfos + LWEInfos + CKKSInfos + SetCKKSInfos,
        K: GetGaloisElement + GGLWEPreparedToBackendRef<BE> + GGLWEInfos,
        BE: 's;
}

#[allow(private_bounds)]
unsafe impl<BE: Backend> CKKSRotateImpl<BE> for BE
where
    BE: HalVecZnxImpl<BE>,
    Module<BE>: CKKSRotateDefault<BE> + GLWEAutomorphism<BE> + GLWEShift<BE>,
    for<'a> ScratchArena<'a, BE>: ScratchArenaTakeCore<'a, BE>,
{
    fn ckks_rotate_tmp_bytes<C: GLWEInfos, K: GGLWEInfos>(module: &Module<BE>, ct_infos: &C, key_infos: &K) -> usize {
        module.ckks_rotate_tmp_bytes_default(ct_infos, key_infos)
    }

    fn ckks_rotate_into<'s, Dst, Src, K>(
        module: &Module<BE>,
        dst: &mut Dst,
        src: &Src,
        key: &K,
        scratch: &mut ScratchArena<'s, BE>,
    ) -> Result<()>
    where
        Dst: GLWEToBackendMut<BE> + GLWEInfos + LWEInfos + CKKSInfos + SetCKKSInfos,
        Src: GLWEToBackendRef<BE> + GLWEInfos + LWEInfos + CKKSInfos,
        K: GetGaloisElement + GGLWEPreparedToBackendRef<BE> + GGLWEInfos,
        BE: 's,
    {
        module.ckks_rotate_into_default(dst, src, key, scratch)
    }

    fn ckks_rotate_assign<'s, Dst, K>(
        module: &Module<BE>,
        dst: &mut Dst,
        key: &K,
        scratch: &mut ScratchArena<'s, BE>,
    ) -> Result<()>
    where
        Dst: GLWEToBackendMut<BE> + GLWEInfos + LWEInfos + CKKSInfos + SetCKKSInfos,
        K: GetGaloisElement + GGLWEPreparedToBackendRef<BE> + GGLWEInfos,
        BE: 's,
    {
        module.ckks_rotate_assign_default(dst, key, scratch)
    }
}

#[macro_export]
macro_rules! impl_ckks_rotate_default {
    ($be:ty) => {
        impl $crate::default::rotate::CKKSRotateDefault<$be> for ::poulpy_hal::layouts::Module<$be> {}
    };
}
pub use crate::impl_ckks_rotate_default;
