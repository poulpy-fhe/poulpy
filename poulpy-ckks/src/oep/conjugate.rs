use crate::CKKSResult as Result;
use crate::default::conjugate::CKKSConjugateDefault;

use poulpy_core::{
    GLWEAutomorphism,
    layouts::{GGLWEInfos, GLWEInfos, GLWEToBackendMut, GLWEToBackendRef, prepared::GLWEAutomorphismKeyPreparedBackendRef},
};
use poulpy_hal::{
    layouts::{Backend, CoeffNormalized, Module, ScratchArena},
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

    fn ckks_conjugate_into_impl<Dst, Src>(
        module: &Module<BE>,
        dst: &mut Dst,
        src: &Src,
        key: &GLWEAutomorphismKeyPreparedBackendRef<'_, BE>,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        Dst: GLWEToBackendMut<BE, State = CoeffNormalized> + GLWEInfos + CKKSCtBounds + SetCKKSInfos,
        Src: GLWEToBackendRef<BE, State = CoeffNormalized> + GLWEInfos + CKKSCtBounds;

    fn ckks_conjugate_assign_impl<Dst>(
        module: &Module<BE>,
        dst: &mut Dst,
        key: &GLWEAutomorphismKeyPreparedBackendRef<'_, BE>,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        Dst: GLWEToBackendMut<BE, State = CoeffNormalized> + CKKSCtBounds + SetCKKSInfos;
}

unsafe impl<BE: Backend> CKKSConjugateImpl<BE> for BE
where
    BE: Backend + HalVecZnxImpl<BE>,
    Module<BE>: CKKSConjugateDefault<BE> + GLWEAutomorphism<BE> + poulpy_core::GLWEShift<BE>,
{
    fn ckks_conjugate_tmp_bytes_impl<C: GLWEInfos, K: GGLWEInfos>(module: &Module<BE>, ct_infos: &C, key_infos: &K) -> usize {
        module.ckks_conjugate_tmp_bytes_default(ct_infos, key_infos)
    }

    fn ckks_conjugate_into_impl<Dst, Src>(
        module: &Module<BE>,
        dst: &mut Dst,
        src: &Src,
        key: &GLWEAutomorphismKeyPreparedBackendRef<'_, BE>,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        Dst: GLWEToBackendMut<BE, State = CoeffNormalized> + GLWEInfos + CKKSCtBounds + SetCKKSInfos,
        Src: GLWEToBackendRef<BE, State = CoeffNormalized> + GLWEInfos + CKKSCtBounds,
    {
        module.ckks_conjugate_into_default(dst, src, key, scratch)
    }

    fn ckks_conjugate_assign_impl<Dst>(
        module: &Module<BE>,
        dst: &mut Dst,
        key: &GLWEAutomorphismKeyPreparedBackendRef<'_, BE>,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        Dst: GLWEToBackendMut<BE, State = CoeffNormalized> + CKKSCtBounds + SetCKKSInfos,
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
