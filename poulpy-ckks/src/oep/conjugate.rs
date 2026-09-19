use crate::CKKSResult as Result;
use crate::reference::conjugate::CKKSConjugateReference;

use poulpy_core::{
    GLWEAutomorphism,
    layouts::{GGLWEInfos, GLWEInfos, GLWEToBackendMut, GLWEToBackendRef, prepared::GLWEAutomorphismKeyPreparedBackendRef},
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
pub unsafe trait CKKSConjugateImpl: Backend {
    fn ckks_conjugate_tmp_bytes_impl<C: GLWEInfos, K: GGLWEInfos>(module: &Module<Self>, ct_infos: &C, key_infos: &K) -> usize;

    fn ckks_conjugate_into_impl<Dst, Src>(
        module: &Module<Self>,
        dst: &mut Dst,
        src: &Src,
        key: &GLWEAutomorphismKeyPreparedBackendRef<'_, Self>,
        scratch: &mut ScratchArena<'_, Self>,
    ) -> Result<()>
    where
        Dst: GLWEToBackendMut<Self> + GLWEInfos + CKKSCtBounds + SetCKKSInfos,
        Src: GLWEToBackendRef<Self> + GLWEInfos + CKKSCtBounds;

    fn ckks_conjugate_assign_impl<Dst>(
        module: &Module<Self>,
        dst: &mut Dst,
        key: &GLWEAutomorphismKeyPreparedBackendRef<'_, Self>,
        scratch: &mut ScratchArena<'_, Self>,
    ) -> Result<()>
    where
        Dst: GLWEToBackendMut<Self> + CKKSCtBounds + SetCKKSInfos;
}

unsafe impl<BE: Backend> CKKSConjugateImpl for BE
where
    BE: Backend + HalVecZnxImpl,
    Module<BE>: CKKSConjugateReference<BE> + GLWEAutomorphism<BE> + poulpy_core::GLWEShift<BE>,
{
    fn ckks_conjugate_tmp_bytes_impl<C: GLWEInfos, K: GGLWEInfos>(module: &Module<BE>, ct_infos: &C, key_infos: &K) -> usize {
        module.ckks_conjugate_tmp_bytes_reference(ct_infos, key_infos)
    }

    fn ckks_conjugate_into_impl<Dst, Src>(
        module: &Module<BE>,
        dst: &mut Dst,
        src: &Src,
        key: &GLWEAutomorphismKeyPreparedBackendRef<'_, BE>,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        Dst: GLWEToBackendMut<BE> + GLWEInfos + CKKSCtBounds + SetCKKSInfos,
        Src: GLWEToBackendRef<BE> + GLWEInfos + CKKSCtBounds,
    {
        module.ckks_conjugate_into_reference(dst, src, key, scratch)
    }

    fn ckks_conjugate_assign_impl<Dst>(
        module: &Module<BE>,
        dst: &mut Dst,
        key: &GLWEAutomorphismKeyPreparedBackendRef<'_, BE>,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        Dst: GLWEToBackendMut<BE> + CKKSCtBounds + SetCKKSInfos,
    {
        module.ckks_conjugate_assign_reference(dst, key, scratch)
    }
}

#[macro_export]
macro_rules! impl_ckks_conjugate_reference {
    ($be:ty) => {
        impl $crate::reference::conjugate::CKKSConjugateReference<$be> for ::poulpy_hal::layouts::Module<$be> {}
    };
}
pub use crate::impl_ckks_conjugate_reference;
