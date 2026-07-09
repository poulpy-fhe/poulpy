use crate::CKKSResult as Result;
use crate::default::conjugate::CKKSConjugateDefault;

use poulpy_core::{
    GLWEAutomorphism,
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
pub unsafe trait CKKSConjugateImpl<BE: Backend>: Backend {
    fn ckks_conjugate_tmp_bytes_impl<C: GLWEInfos, K: GGLWEInfos>(module: &Module<BE>, ct_infos: &C, key_infos: &K) -> usize;

    fn ckks_conjugate_into_impl<Dst, Src, K>(
        module: &Module<BE>,
        dst: &mut Dst,
        src: &Src,
        key: &K,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        Dst: GLWEToBackendMut<BE> + GLWEInfos + CKKSCtBounds + SetCKKSInfos,
        Src: GLWEToBackendRef<BE> + GLWEInfos + CKKSCtBounds,
        K: GetGaloisElement + GGLWEPreparedToBackendRef<BE> + GGLWEInfos;

    fn ckks_conjugate_assign_impl<Dst, K>(
        module: &Module<BE>,
        dst: &mut Dst,
        key: &K,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        Dst: GLWEToBackendMut<BE> + CKKSCtBounds + SetCKKSInfos,
        K: GetGaloisElement + GGLWEPreparedToBackendRef<BE> + GGLWEInfos;
}

unsafe impl<BE: Backend> CKKSConjugateImpl<BE> for BE
where
    BE: HalVecZnxImpl<BE>,
    Module<BE>: CKKSConjugateDefault<BE> + GLWEAutomorphism<BE>,
{
    fn ckks_conjugate_tmp_bytes_impl<C: GLWEInfos, K: GGLWEInfos>(module: &Module<BE>, ct_infos: &C, key_infos: &K) -> usize {
        module.ckks_conjugate_tmp_bytes_default(ct_infos, key_infos)
    }

    fn ckks_conjugate_into_impl<Dst, Src, K>(
        module: &Module<BE>,
        dst: &mut Dst,
        src: &Src,
        key: &K,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        Dst: GLWEToBackendMut<BE> + GLWEInfos + CKKSCtBounds + SetCKKSInfos,
        Src: GLWEToBackendRef<BE> + GLWEInfos + CKKSCtBounds,
        K: GetGaloisElement + GGLWEPreparedToBackendRef<BE> + GGLWEInfos,
    {
        module.ckks_conjugate_into_default(dst, src, key, scratch)
    }

    fn ckks_conjugate_assign_impl<Dst, K>(
        module: &Module<BE>,
        dst: &mut Dst,
        key: &K,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        Dst: GLWEToBackendMut<BE> + CKKSCtBounds + SetCKKSInfos,
        K: GetGaloisElement + GGLWEPreparedToBackendRef<BE> + GGLWEInfos,
    {
        module.ckks_conjugate_assign_default(dst, key, scratch)
    }
}

#[macro_export]
macro_rules! impl_ckks_conjugate_defaults {
    ($be:ty) => {
        impl $crate::default::conjugate::CKKSConjugateDefault<$be> for ::poulpy_hal::layouts::Module<$be> {}
    };
}
pub use crate::impl_ckks_conjugate_defaults;
