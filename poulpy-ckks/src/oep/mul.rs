use crate::CKKSResult as Result;
use poulpy_core::layouts::GetTensorKey;
use poulpy_core::layouts::IntPolyInfos;

use poulpy_core::layouts::{GGLWEInfos, GLWEInfos, LWEInfos, TorusPrecision};
use poulpy_hal::layouts::{Backend, Module, PrepareHint, ScratchArena};

use crate::{
    CKKSCtBounds, CKKSInfos, GLWEToBackendMut, GLWEToBackendRef, SetCKKSInfos,
    layouts::{CKKSPreparedLeft, CKKSPreparedRight},
};

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
        tsk: &T,
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
        tsk: &T,
        scratch: &mut ScratchArena<'_, Self>,
    ) -> Result<()>
    where
        Dst: GLWEToBackendMut<Self> + GLWEToBackendRef<Self> + CKKSInfos + SetCKKSInfos + GLWEInfos,
        A: GLWEToBackendRef<Self> + CKKSInfos + GLWEInfos,
        T: GetTensorKey<Self>;
    /// Scratch bytes of [`Self::ckks_prepare_left_impl`] and [`Self::ckks_prepare_right_impl`] for an operand shaped
    /// like `a`, which [`Self::ckks_mul_tmp_bytes_impl`] of a product reading such an operand covers, as the API
    /// promises.
    fn ckks_prepare_tmp_bytes_impl<A: GLWEInfos>(module: &Module<Self>, a: &A) -> usize;
    /// Prepares a left operand. Whatever `hint` selects, the products must give the reference's bits.
    fn ckks_prepare_left_impl<A>(
        module: &Module<Self>,
        a: &A,
        hint: PrepareHint,
        scratch: &mut ScratchArena<'_, Self>,
    ) -> Result<CKKSPreparedLeft<Self>>
    where
        A: GLWEToBackendRef<Self> + CKKSInfos + GLWEInfos;
    /// Prepares a right operand. Whatever `hint` selects, the products must give the reference's bits.
    fn ckks_prepare_right_impl<A>(
        module: &Module<Self>,
        a: &A,
        hint: PrepareHint,
        scratch: &mut ScratchArena<'_, Self>,
    ) -> Result<CKKSPreparedRight<Self>>
    where
        A: GLWEToBackendRef<Self> + CKKSInfos + GLWEInfos;
    fn ckks_mul_prepared_assign_impl<Dst, T>(
        module: &Module<Self>,
        dst: &mut Dst,
        prepared: &CKKSPreparedRight<Self>,
        tsk: &T,
        scratch: &mut ScratchArena<'_, Self>,
    ) -> Result<()>
    where
        Dst: GLWEToBackendMut<Self> + GLWEToBackendRef<Self> + CKKSInfos + SetCKKSInfos + GLWEInfos,
        T: GetTensorKey<Self>;
    fn ckks_square_into_impl<Dst, A, T>(
        module: &Module<Self>,
        dst: &mut Dst,
        a: &A,
        tsk: &T,
        scratch: &mut ScratchArena<'_, Self>,
    ) -> Result<()>
    where
        Dst: GLWEToBackendMut<Self> + CKKSInfos + SetCKKSInfos + GLWEInfos,
        A: GLWEToBackendRef<Self> + CKKSInfos + GLWEInfos,
        T: GetTensorKey<Self>;
    fn ckks_square_assign_impl<Dst, T>(
        module: &Module<Self>,
        dst: &mut Dst,
        tsk: &T,
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

/// Implements this contract with the callable reference algorithms.
#[macro_export]
macro_rules! impl_ckks_mul_reference {
    ($be:ty) => {
        unsafe impl $crate::oep::CKKSMulImpl for $be {
            fn ckks_mul_tmp_bytes_impl<
                R: ::poulpy_core::layouts::GLWEInfos,
                A: ::poulpy_core::layouts::GLWEInfos,
                B: ::poulpy_core::layouts::GLWEInfos,
                T: ::poulpy_core::layouts::GGLWEInfos,
            >(
                module: &::poulpy_hal::layouts::Module<Self>,
                res: &R,
                a: &A,
                b: &B,
                tsk: &T,
            ) -> usize {
                $crate::reference::mul::CKKSMulReference::ckks_mul_tmp_bytes_reference(module, res, a, b, tsk)
            }

            fn ckks_square_tmp_bytes_impl<
                R: ::poulpy_core::layouts::GLWEInfos,
                A: ::poulpy_core::layouts::GLWEInfos,
                T: ::poulpy_core::layouts::GGLWEInfos,
            >(
                module: &::poulpy_hal::layouts::Module<Self>,
                res: &R,
                a: &A,
                tsk: &T,
            ) -> usize {
                $crate::reference::mul::CKKSMulReference::ckks_square_tmp_bytes_reference(module, res, a, tsk)
            }

            fn ckks_mul_pt_vec_tmp_bytes_impl<R: ::poulpy_core::layouts::GLWEInfos, A: ::poulpy_core::layouts::GLWEInfos>(
                module: &::poulpy_hal::layouts::Module<Self>,
                res: &R,
                a: &A,
                b_k: ::poulpy_core::layouts::TorusPrecision,
            ) -> usize {
                $crate::reference::mul::CKKSMulReference::ckks_mul_pt_vec_tmp_bytes_reference(module, res, a, b_k)
            }

            fn ckks_mul_pt_const_tmp_bytes_impl<R: ::poulpy_core::layouts::GLWEInfos, A: ::poulpy_core::layouts::GLWEInfos>(
                module: &::poulpy_hal::layouts::Module<Self>,
                res: &R,
                a: &A,
                b_k: ::poulpy_core::layouts::TorusPrecision,
            ) -> usize {
                $crate::reference::mul::CKKSMulReference::ckks_mul_pt_const_tmp_bytes_reference(module, res, a, b_k)
            }

            fn ckks_mul_into_impl<Dst, A, B, T>(
                module: &::poulpy_hal::layouts::Module<Self>,
                dst: &mut Dst,
                a: &A,
                b: &B,
                tsk: &T,
                scratch: &mut ::poulpy_hal::layouts::ScratchArena<'_, Self>,
            ) -> $crate::CKKSResult<()>
            where
                Dst: ::poulpy_core::layouts::GLWEToBackendMut<Self>
                    + $crate::CKKSInfos
                    + $crate::SetCKKSInfos
                    + ::poulpy_core::layouts::GLWEInfos,
                A: ::poulpy_core::layouts::GLWEToBackendRef<Self> + $crate::CKKSInfos + ::poulpy_core::layouts::GLWEInfos,
                B: ::poulpy_core::layouts::GLWEToBackendRef<Self> + $crate::CKKSInfos + ::poulpy_core::layouts::GLWEInfos,
                T: ::poulpy_core::layouts::GetTensorKey<Self>,
            {
                $crate::reference::mul::CKKSMulReference::ckks_mul_into_reference(module, dst, a, b, tsk, scratch)
            }

            fn ckks_mul_assign_impl<Dst, A, T>(
                module: &::poulpy_hal::layouts::Module<Self>,
                dst: &mut Dst,
                a: &A,
                tsk: &T,
                scratch: &mut ::poulpy_hal::layouts::ScratchArena<'_, Self>,
            ) -> $crate::CKKSResult<()>
            where
                Dst: ::poulpy_core::layouts::GLWEToBackendMut<Self>
                    + ::poulpy_core::layouts::GLWEToBackendRef<Self>
                    + $crate::CKKSInfos
                    + $crate::SetCKKSInfos
                    + ::poulpy_core::layouts::GLWEInfos,
                A: ::poulpy_core::layouts::GLWEToBackendRef<Self> + $crate::CKKSInfos + ::poulpy_core::layouts::GLWEInfos,
                T: ::poulpy_core::layouts::GetTensorKey<Self>,
            {
                $crate::reference::mul::CKKSMulReference::ckks_mul_assign_reference(module, dst, a, tsk, scratch)
            }

            fn ckks_prepare_tmp_bytes_impl<A: ::poulpy_core::layouts::GLWEInfos>(
                module: &::poulpy_hal::layouts::Module<Self>,
                a: &A,
            ) -> usize {
                $crate::reference::mul::CKKSMulReference::ckks_prepare_tmp_bytes_reference(module, a)
            }

            fn ckks_prepare_left_impl<A>(
                module: &::poulpy_hal::layouts::Module<Self>,
                a: &A,
                hint: ::poulpy_hal::layouts::PrepareHint,
                scratch: &mut ::poulpy_hal::layouts::ScratchArena<'_, Self>,
            ) -> $crate::CKKSResult<$crate::layouts::CKKSPreparedLeft<Self>>
            where
                A: ::poulpy_core::layouts::GLWEToBackendRef<Self> + $crate::CKKSInfos + ::poulpy_core::layouts::GLWEInfos,
            {
                $crate::reference::mul::CKKSMulReference::ckks_prepare_left_reference(module, a, hint, scratch)
            }

            fn ckks_prepare_right_impl<A>(
                module: &::poulpy_hal::layouts::Module<Self>,
                a: &A,
                hint: ::poulpy_hal::layouts::PrepareHint,
                scratch: &mut ::poulpy_hal::layouts::ScratchArena<'_, Self>,
            ) -> $crate::CKKSResult<$crate::layouts::CKKSPreparedRight<Self>>
            where
                A: ::poulpy_core::layouts::GLWEToBackendRef<Self> + $crate::CKKSInfos + ::poulpy_core::layouts::GLWEInfos,
            {
                $crate::reference::mul::CKKSMulReference::ckks_prepare_right_reference(module, a, hint, scratch)
            }

            fn ckks_mul_prepared_assign_impl<Dst, T>(
                module: &::poulpy_hal::layouts::Module<Self>,
                dst: &mut Dst,
                prepared: &$crate::layouts::CKKSPreparedRight<Self>,
                tsk: &T,
                scratch: &mut ::poulpy_hal::layouts::ScratchArena<'_, Self>,
            ) -> $crate::CKKSResult<()>
            where
                Dst: ::poulpy_core::layouts::GLWEToBackendMut<Self>
                    + ::poulpy_core::layouts::GLWEToBackendRef<Self>
                    + $crate::CKKSInfos
                    + $crate::SetCKKSInfos
                    + ::poulpy_core::layouts::GLWEInfos,
                T: ::poulpy_core::layouts::GetTensorKey<Self>,
            {
                $crate::reference::mul::CKKSMulReference::ckks_mul_prepared_assign_reference(module, dst, prepared, tsk, scratch)
            }

            fn ckks_square_into_impl<Dst, A, T>(
                module: &::poulpy_hal::layouts::Module<Self>,
                dst: &mut Dst,
                a: &A,
                tsk: &T,
                scratch: &mut ::poulpy_hal::layouts::ScratchArena<'_, Self>,
            ) -> $crate::CKKSResult<()>
            where
                Dst: ::poulpy_core::layouts::GLWEToBackendMut<Self>
                    + $crate::CKKSInfos
                    + $crate::SetCKKSInfos
                    + ::poulpy_core::layouts::GLWEInfos,
                A: ::poulpy_core::layouts::GLWEToBackendRef<Self> + $crate::CKKSInfos + ::poulpy_core::layouts::GLWEInfos,
                T: ::poulpy_core::layouts::GetTensorKey<Self>,
            {
                $crate::reference::mul::CKKSMulReference::ckks_square_into_reference(module, dst, a, tsk, scratch)
            }

            fn ckks_square_assign_impl<Dst, T>(
                module: &::poulpy_hal::layouts::Module<Self>,
                dst: &mut Dst,
                tsk: &T,
                scratch: &mut ::poulpy_hal::layouts::ScratchArena<'_, Self>,
            ) -> $crate::CKKSResult<()>
            where
                Dst: ::poulpy_core::layouts::GLWEToBackendMut<Self>
                    + ::poulpy_core::layouts::GLWEToBackendRef<Self>
                    + $crate::CKKSInfos
                    + $crate::SetCKKSInfos
                    + ::poulpy_core::layouts::GLWEInfos,
                T: ::poulpy_core::layouts::GetTensorKey<Self>,
            {
                $crate::reference::mul::CKKSMulReference::ckks_square_assign_reference(module, dst, tsk, scratch)
            }

            fn ckks_mul_pt_vec_into_impl<Dst, A, P>(
                module: &::poulpy_hal::layouts::Module<Self>,
                dst: &mut Dst,
                a: &A,
                pt: &P,
                scratch: &mut ::poulpy_hal::layouts::ScratchArena<'_, Self>,
            ) -> $crate::CKKSResult<()>
            where
                Dst: ::poulpy_core::layouts::GLWEToBackendMut<Self>
                    + $crate::CKKSInfos
                    + $crate::SetCKKSInfos
                    + ::poulpy_core::layouts::GLWEInfos,
                A: ::poulpy_core::layouts::GLWEToBackendRef<Self> + $crate::CKKSInfos + ::poulpy_core::layouts::GLWEInfos,
                P: ::poulpy_core::layouts::GLWEToBackendRef<Self>
                    + ::poulpy_core::layouts::LWEInfos
                    + ::poulpy_core::layouts::IntPolyInfos
                    + $crate::CKKSCtBounds,
            {
                $crate::reference::mul::CKKSMulReference::ckks_mul_pt_vec_into_reference(module, dst, a, pt, scratch)
            }

            fn ckks_mul_pt_vec_assign_impl<Dst, P>(
                module: &::poulpy_hal::layouts::Module<Self>,
                dst: &mut Dst,
                pt: &P,
                scratch: &mut ::poulpy_hal::layouts::ScratchArena<'_, Self>,
            ) -> $crate::CKKSResult<()>
            where
                Dst: ::poulpy_core::layouts::GLWEToBackendMut<Self>
                    + $crate::CKKSInfos
                    + $crate::SetCKKSInfos
                    + ::poulpy_core::layouts::GLWEInfos,
                P: ::poulpy_core::layouts::GLWEToBackendRef<Self>
                    + ::poulpy_core::layouts::LWEInfos
                    + ::poulpy_core::layouts::IntPolyInfos
                    + $crate::CKKSCtBounds,
            {
                $crate::reference::mul::CKKSMulReference::ckks_mul_pt_vec_assign_reference(module, dst, pt, scratch)
            }

            fn ckks_mul_pt_const_into_impl<Dst, A, P>(
                module: &::poulpy_hal::layouts::Module<Self>,
                dst: &mut Dst,
                a: &A,
                pt: &P,
                pt_coeff: usize,
                scratch: &mut ::poulpy_hal::layouts::ScratchArena<'_, Self>,
            ) -> $crate::CKKSResult<()>
            where
                Dst: ::poulpy_core::layouts::GLWEToBackendMut<Self>
                    + $crate::CKKSInfos
                    + $crate::SetCKKSInfos
                    + ::poulpy_core::layouts::GLWEInfos,
                A: ::poulpy_core::layouts::GLWEToBackendRef<Self> + $crate::CKKSInfos + ::poulpy_core::layouts::GLWEInfos,
                P: ::poulpy_core::layouts::GLWEToBackendRef<Self>
                    + ::poulpy_core::layouts::LWEInfos
                    + ::poulpy_core::layouts::IntPolyInfos
                    + $crate::CKKSCtBounds,
            {
                $crate::reference::mul::CKKSMulReference::ckks_mul_pt_const_into_reference(module, dst, a, pt, pt_coeff, scratch)
            }

            fn ckks_mul_pt_const_assign_impl<Dst, P>(
                module: &::poulpy_hal::layouts::Module<Self>,
                dst: &mut Dst,
                pt: &P,
                pt_coeff: usize,
                scratch: &mut ::poulpy_hal::layouts::ScratchArena<'_, Self>,
            ) -> $crate::CKKSResult<()>
            where
                Dst: ::poulpy_core::layouts::GLWEToBackendMut<Self>
                    + ::poulpy_core::layouts::GLWEToBackendRef<Self>
                    + $crate::CKKSInfos
                    + $crate::SetCKKSInfos
                    + ::poulpy_core::layouts::GLWEInfos,
                P: ::poulpy_core::layouts::GLWEToBackendRef<Self>
                    + ::poulpy_core::layouts::LWEInfos
                    + ::poulpy_core::layouts::IntPolyInfos
                    + $crate::CKKSCtBounds,
            {
                $crate::reference::mul::CKKSMulReference::ckks_mul_pt_const_assign_reference(module, dst, pt, pt_coeff, scratch)
            }
        }
    };
}
pub use crate::impl_ckks_mul_reference;
