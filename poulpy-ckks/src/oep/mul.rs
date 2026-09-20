use crate::CKKSResult as Result;
use crate::reference::mul::CKKSMulReference;
use poulpy_core::layouts::GetTensorKey;
use poulpy_core::layouts::IntPolyInfos;

use poulpy_core::{
    GLWEAdd, GLWECopy, GLWEMulConst, GLWEMulPlain, GLWERotate, GLWETensoring, GiantStepTensorBounds,
    layouts::{GGLWEInfos, GLWEInfos, LWEInfos, ModuleCoreAlloc, TorusPrecision},
};
use poulpy_hal::{
    api::{CnvPVecAlloc, VecZnxCopy},
    layouts::{Backend, Module, ScratchArena},
};

use crate::{CKKSCtBounds, CKKSInfos, GLWEToBackendMut, GLWEToBackendRef, SetCKKSInfos, layouts::CKKSPreparedRight};

/// # Safety
///
/// Implementations must satisfy the contracts of all trait methods, including
/// any HAL-level invariants (alignment, layout, scratch sizing) implied by the
/// associated method signatures.
pub unsafe trait CKKSMulImpl: Backend {
    fn ckks_mul_tmp_bytes_impl<R: GLWEInfos, A: GLWEInfos, B: GLWEInfos, T: GGLWEInfos>(
        module: &Module<Self>,
        res: &R,
        a: &A,
        b: &B,
        tsk: &T,
    ) -> usize;
    fn ckks_square_tmp_bytes_impl<R: GLWEInfos, A: GLWEInfos, T: GGLWEInfos>(
        module: &Module<Self>,
        res: &R,
        a: &A,
        tsk: &T,
    ) -> usize;
    fn ckks_mul_pt_vec_tmp_bytes_impl<R: GLWEInfos, A: GLWEInfos>(
        module: &Module<Self>,
        res: &R,
        a: &A,
        b_k: TorusPrecision,
    ) -> usize;
    fn ckks_mul_pt_const_tmp_bytes_impl<R: GLWEInfos, A: GLWEInfos>(
        module: &Module<Self>,
        res: &R,
        a: &A,
        b_k: TorusPrecision,
    ) -> usize;
    fn ckks_mul_into_impl<Dst, A, B, T>(
        module: &Module<Self>,
        dst: &mut Dst,
        a: &A,
        b: &B,
        tsk: &crate::layouts::CKKSKey<T>,
        scratch: &mut ScratchArena<'_, Self>,
    ) -> Result<()>
    where
        Dst: GLWEToBackendMut<Self> + CKKSInfos + SetCKKSInfos + GLWEInfos,
        A: GLWEToBackendRef<Self> + CKKSInfos + GLWEInfos,
        B: GLWEToBackendRef<Self> + CKKSInfos + GLWEInfos,
        T: GetTensorKey<Self>;
    fn ckks_mul_assign_impl<Dst, A, T>(
        module: &Module<Self>,
        dst: &mut Dst,
        a: &A,
        tsk: &crate::layouts::CKKSKey<T>,
        scratch: &mut ScratchArena<'_, Self>,
    ) -> Result<()>
    where
        Dst: GLWEToBackendMut<Self> + GLWEToBackendRef<Self> + CKKSInfos + SetCKKSInfos + GLWEInfos,
        A: GLWEToBackendRef<Self> + CKKSInfos + GLWEInfos,
        T: GetTensorKey<Self>;
    fn ckks_prepare_right_impl<A>(
        module: &Module<Self>,
        a: &A,
        scratch: &mut ScratchArena<'_, Self>,
    ) -> Result<CKKSPreparedRight<Self>>
    where
        A: GLWEToBackendRef<Self> + CKKSInfos + GLWEInfos;
    fn ckks_mul_prepared_assign_impl<Dst, T>(
        module: &Module<Self>,
        dst: &mut Dst,
        prepared: &CKKSPreparedRight<Self>,
        tsk: &crate::layouts::CKKSKey<T>,
        scratch: &mut ScratchArena<'_, Self>,
    ) -> Result<()>
    where
        Dst: GLWEToBackendMut<Self> + GLWEToBackendRef<Self> + CKKSInfos + SetCKKSInfos + GLWEInfos,
        T: GetTensorKey<Self>;
    fn ckks_square_into_impl<Dst, A, T>(
        module: &Module<Self>,
        dst: &mut Dst,
        a: &A,
        tsk: &crate::layouts::CKKSKey<T>,
        scratch: &mut ScratchArena<'_, Self>,
    ) -> Result<()>
    where
        Dst: GLWEToBackendMut<Self> + CKKSInfos + SetCKKSInfos + GLWEInfos,
        A: GLWEToBackendRef<Self> + CKKSInfos + GLWEInfos,
        T: GetTensorKey<Self>;
    fn ckks_square_assign_impl<Dst, T>(
        module: &Module<Self>,
        dst: &mut Dst,
        tsk: &crate::layouts::CKKSKey<T>,
        scratch: &mut ScratchArena<'_, Self>,
    ) -> Result<()>
    where
        Dst: GLWEToBackendMut<Self> + GLWEToBackendRef<Self> + CKKSInfos + SetCKKSInfos + GLWEInfos,
        T: GetTensorKey<Self>;
    fn ckks_mul_pt_vec_into_impl<Dst, A, P>(
        module: &Module<Self>,
        dst: &mut Dst,
        a: &A,
        pt: &P,
        scratch: &mut ScratchArena<'_, Self>,
    ) -> Result<()>
    where
        Dst: GLWEToBackendMut<Self> + CKKSInfos + SetCKKSInfos + GLWEInfos,
        A: GLWEToBackendRef<Self> + CKKSInfos + GLWEInfos,
        P: GLWEToBackendRef<Self> + LWEInfos + IntPolyInfos + CKKSCtBounds;
    fn ckks_mul_pt_vec_assign_impl<Dst, P>(
        module: &Module<Self>,
        dst: &mut Dst,
        pt: &P,
        scratch: &mut ScratchArena<'_, Self>,
    ) -> Result<()>
    where
        Dst: GLWEToBackendMut<Self> + CKKSInfos + SetCKKSInfos + GLWEInfos,
        P: GLWEToBackendRef<Self> + LWEInfos + IntPolyInfos + CKKSCtBounds;
    fn ckks_mul_pt_const_into_impl<Dst, A, P>(
        module: &Module<Self>,
        dst: &mut Dst,
        a: &A,
        pt: &P,
        pt_coeff: usize,
        scratch: &mut ScratchArena<'_, Self>,
    ) -> Result<()>
    where
        Dst: GLWEToBackendMut<Self> + CKKSInfos + SetCKKSInfos + GLWEInfos,
        A: GLWEToBackendRef<Self> + CKKSInfos + GLWEInfos,
        P: GLWEToBackendRef<Self> + LWEInfos + IntPolyInfos + CKKSCtBounds;
    fn ckks_mul_pt_const_assign_impl<Dst, P>(
        module: &Module<Self>,
        dst: &mut Dst,
        pt: &P,
        pt_coeff: usize,
        scratch: &mut ScratchArena<'_, Self>,
    ) -> Result<()>
    where
        Dst: GLWEToBackendMut<Self> + GLWEToBackendRef<Self> + CKKSInfos + SetCKKSInfos + GLWEInfos,
        P: GLWEToBackendRef<Self> + LWEInfos + IntPolyInfos + CKKSCtBounds;
}

unsafe impl<BE: Backend> CKKSMulImpl for BE
where
    BE: poulpy_hal::oep::HalVecZnxImpl,
    Module<BE>: crate::reference::mul::CKKSMulReference<BE>
        + GLWEAdd<BE>
        + GLWECopy<BE>
        + GLWEMulConst<BE>
        + GLWEMulPlain<BE>
        + GLWERotate<BE>
        + GLWETensoring<BE>
        + GiantStepTensorBounds<BE>
        + CnvPVecAlloc<BE>
        + ModuleCoreAlloc<OwnedBuf = BE::OwnedBuf, ZnxWord = BE::ZnxWord>
        + VecZnxCopy<BE>,
{
    fn ckks_mul_tmp_bytes_impl<R: GLWEInfos, A: GLWEInfos, B: GLWEInfos, T: GGLWEInfos>(
        module: &Module<BE>,
        res: &R,
        a: &A,
        b: &B,
        tsk: &T,
    ) -> usize {
        module.ckks_mul_tmp_bytes_reference(res, a, b, tsk)
    }

    fn ckks_square_tmp_bytes_impl<R: GLWEInfos, A: GLWEInfos, T: GGLWEInfos>(
        module: &Module<BE>,
        res: &R,
        a: &A,
        tsk: &T,
    ) -> usize {
        module.ckks_square_tmp_bytes_reference(res, a, tsk)
    }

    fn ckks_mul_pt_vec_tmp_bytes_impl<R: GLWEInfos, A: GLWEInfos>(
        module: &Module<BE>,
        res: &R,
        a: &A,
        b_k: TorusPrecision,
    ) -> usize {
        module.ckks_mul_pt_vec_tmp_bytes_reference(res, a, b_k)
    }

    fn ckks_mul_pt_const_tmp_bytes_impl<R: GLWEInfos, A: GLWEInfos>(
        module: &Module<BE>,
        res: &R,
        a: &A,
        b_k: TorusPrecision,
    ) -> usize {
        module.ckks_mul_pt_const_tmp_bytes_reference(res, a, b_k)
    }

    fn ckks_mul_into_impl<Dst, A, B, T>(
        module: &Module<BE>,
        dst: &mut Dst,
        a: &A,
        b: &B,
        tsk: &crate::layouts::CKKSKey<T>,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        Dst: GLWEToBackendMut<BE> + CKKSInfos + SetCKKSInfos + GLWEInfos,
        A: GLWEToBackendRef<BE> + CKKSInfos + GLWEInfos,
        B: GLWEToBackendRef<BE> + CKKSInfos + GLWEInfos,
        T: GetTensorKey<BE>,
    {
        module.ckks_mul_into_reference(dst, a, b, tsk, scratch)
    }

    fn ckks_mul_assign_impl<Dst, A, T>(
        module: &Module<BE>,
        dst: &mut Dst,
        a: &A,
        tsk: &crate::layouts::CKKSKey<T>,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        Dst: GLWEToBackendMut<BE> + GLWEToBackendRef<BE> + CKKSInfos + SetCKKSInfos + GLWEInfos,
        A: GLWEToBackendRef<BE> + CKKSInfos + GLWEInfos,
        T: GetTensorKey<BE>,
    {
        module.ckks_mul_assign_reference(dst, a, tsk, scratch)
    }

    fn ckks_prepare_right_impl<A>(module: &Module<BE>, a: &A, scratch: &mut ScratchArena<'_, BE>) -> Result<CKKSPreparedRight<BE>>
    where
        A: GLWEToBackendRef<BE> + CKKSInfos + GLWEInfos,
    {
        module.ckks_prepare_right_reference(a, scratch)
    }

    fn ckks_mul_prepared_assign_impl<Dst, T>(
        module: &Module<BE>,
        dst: &mut Dst,
        prepared: &CKKSPreparedRight<BE>,
        tsk: &crate::layouts::CKKSKey<T>,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        Dst: GLWEToBackendMut<BE> + GLWEToBackendRef<BE> + CKKSInfos + SetCKKSInfos + GLWEInfos,
        T: GetTensorKey<BE>,
    {
        module.ckks_mul_prepared_assign_reference(dst, prepared, tsk, scratch)
    }

    fn ckks_square_into_impl<Dst, A, T>(
        module: &Module<BE>,
        dst: &mut Dst,
        a: &A,
        tsk: &crate::layouts::CKKSKey<T>,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        Dst: GLWEToBackendMut<BE> + CKKSInfos + SetCKKSInfos + GLWEInfos,
        A: GLWEToBackendRef<BE> + CKKSInfos + GLWEInfos,
        T: GetTensorKey<BE>,
    {
        module.ckks_square_into_reference(dst, a, tsk, scratch)
    }

    fn ckks_square_assign_impl<Dst, T>(
        module: &Module<BE>,
        dst: &mut Dst,
        tsk: &crate::layouts::CKKSKey<T>,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        Dst: GLWEToBackendMut<BE> + GLWEToBackendRef<BE> + CKKSInfos + SetCKKSInfos + GLWEInfos,
        T: GetTensorKey<BE>,
    {
        module.ckks_square_assign_reference(dst, tsk, scratch)
    }

    fn ckks_mul_pt_vec_into_impl<Dst, A, P>(
        module: &Module<BE>,
        dst: &mut Dst,
        a: &A,
        pt: &P,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        Dst: GLWEToBackendMut<BE> + CKKSInfos + SetCKKSInfos + GLWEInfos,
        A: GLWEToBackendRef<BE> + CKKSInfos + GLWEInfos,
        P: GLWEToBackendRef<BE> + LWEInfos + IntPolyInfos + CKKSCtBounds,
    {
        module.ckks_mul_pt_vec_into_reference(dst, a, pt, scratch)
    }

    fn ckks_mul_pt_vec_assign_impl<Dst, P>(
        module: &Module<BE>,
        dst: &mut Dst,
        pt: &P,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        Dst: GLWEToBackendMut<BE> + CKKSInfos + SetCKKSInfos + GLWEInfos,
        P: GLWEToBackendRef<BE> + LWEInfos + IntPolyInfos + CKKSCtBounds,
    {
        module.ckks_mul_pt_vec_assign_reference(dst, pt, scratch)
    }

    fn ckks_mul_pt_const_into_impl<Dst, A, P>(
        module: &Module<BE>,
        dst: &mut Dst,
        a: &A,
        pt: &P,
        pt_coeff: usize,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        Dst: GLWEToBackendMut<BE> + CKKSInfos + SetCKKSInfos + GLWEInfos,
        A: GLWEToBackendRef<BE> + CKKSInfos + GLWEInfos,
        P: GLWEToBackendRef<BE> + LWEInfos + IntPolyInfos + CKKSCtBounds,
    {
        module.ckks_mul_pt_const_into_reference(dst, a, pt, pt_coeff, scratch)
    }

    fn ckks_mul_pt_const_assign_impl<Dst, P>(
        module: &Module<BE>,
        dst: &mut Dst,
        pt: &P,
        pt_coeff: usize,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        Dst: GLWEToBackendMut<BE> + GLWEToBackendRef<BE> + CKKSInfos + SetCKKSInfos + GLWEInfos,
        P: GLWEToBackendRef<BE> + LWEInfos + IntPolyInfos + CKKSCtBounds,
    {
        module.ckks_mul_pt_const_assign_reference(dst, pt, pt_coeff, scratch)
    }
}

#[macro_export]
macro_rules! impl_ckks_mul_reference {
    ($be:ty) => {
        impl $crate::reference::mul::CKKSMulReference<$be> for ::poulpy_hal::layouts::Module<$be> {}
    };
}
pub use crate::impl_ckks_mul_reference;
