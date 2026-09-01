use crate::CKKSResult as Result;
use crate::default::neg::CKKSNegDefault;
use poulpy_hal::layouts::Normalized;

use poulpy_core::{GLWENegate, GLWEShift, layouts::GLWEInfos};
use poulpy_hal::layouts::{Backend, FitsIn, Module, ScratchArena};

use crate::{CKKSCtBounds, GLWEToBackendMut, GLWEToBackendRef, SetCKKSInfos};

/// # Safety
///
/// Implementations must satisfy the contracts of all trait methods, including
/// any HAL-level invariants (alignment, layout, scratch sizing) implied by the
/// associated method signatures.
pub unsafe trait CKKSNegImpl<BE: Backend>: Backend {
    fn ckks_neg_tmp_bytes_impl(module: &Module<BE>) -> usize;

    fn ckks_neg_into_impl<Dst, Src>(
        module: &Module<BE>,
        dst: &mut Dst,
        src: &Src,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        Dst: GLWEToBackendMut<BE, State = Normalized> + CKKSCtBounds + SetCKKSInfos,
        Src: GLWEToBackendRef<BE, State = Normalized> + GLWEInfos + CKKSCtBounds,
        <Src as GLWEToBackendRef<BE>>::State: FitsIn<<Dst as GLWEToBackendRef<BE>>::State>;

    fn ckks_neg_assign_impl<Dst>(module: &Module<BE>, dst: &mut Dst) -> Result<()>
    where
        Dst: GLWEToBackendMut<BE, State = Normalized> + CKKSCtBounds + SetCKKSInfos;
}

unsafe impl<BE: Backend> CKKSNegImpl<BE> for BE
where
    BE: poulpy_hal::oep::HalVecZnxImpl<BE>,
    Module<BE>: crate::default::neg::CKKSNegDefault<BE> + GLWENegate<BE> + GLWEShift<BE>,
{
    fn ckks_neg_tmp_bytes_impl(module: &Module<BE>) -> usize {
        module.ckks_neg_tmp_bytes_default()
    }

    fn ckks_neg_into_impl<Dst, Src>(
        module: &Module<BE>,
        dst: &mut Dst,
        src: &Src,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        Dst: GLWEToBackendMut<BE, State = Normalized> + CKKSCtBounds + SetCKKSInfos,
        Src: GLWEToBackendRef<BE, State = Normalized> + GLWEInfos + CKKSCtBounds,
        <Src as GLWEToBackendRef<BE>>::State: FitsIn<<Dst as GLWEToBackendRef<BE>>::State>,
    {
        module.ckks_neg_into_default(dst, src, scratch)
    }

    fn ckks_neg_assign_impl<Dst>(module: &Module<BE>, dst: &mut Dst) -> Result<()>
    where
        Dst: GLWEToBackendMut<BE, State = Normalized> + CKKSCtBounds + SetCKKSInfos,
    {
        module.ckks_neg_assign_default(dst)
    }
}

#[macro_export]
macro_rules! impl_ckks_neg_defaults {
    ($be:ty) => {
        impl $crate::default::neg::CKKSNegDefault<$be> for ::poulpy_hal::layouts::Module<$be> {}
    };
}
pub use crate::impl_ckks_neg_defaults;
