use crate::CKKSResult as Result;
use poulpy_core::layouts::GetTensorKey;
use poulpy_core::layouts::IntPolyInfos;

use poulpy_core::layouts::{GGLWEInfos, GLWEInfos, LWEInfos};
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
    fn ckks_mul_tmp_bytes<R: GLWEInfos, A: GLWEInfos, B: GLWEInfos, T: GGLWEInfos>(
        module: &Module<Self>,
        res: &R,
        a: &A,
        b: &B,
        tsk: &T,
    ) -> usize;
    fn ckks_square_tmp_bytes<R: GLWEInfos, A: GLWEInfos, T: GGLWEInfos>(module: &Module<Self>, res: &R, a: &A, tsk: &T) -> usize;
    fn ckks_mul_pt_vec_tmp_bytes<R, A, P>(module: &Module<Self>, res: &R, a: &A, b: &P) -> usize
    where
        R: CKKSCtBounds,
        A: CKKSCtBounds,
        P: CKKSInfos;
    fn ckks_mul_pt_const_tmp_bytes<R, A, P>(module: &Module<Self>, res: &R, a: &A, b: &P) -> usize
    where
        R: CKKSCtBounds,
        A: CKKSCtBounds,
        P: CKKSInfos;
    fn ckks_mul_into<Dst, A, B, T>(
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
    fn ckks_mul_assign<Dst, A, T>(
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
    /// Scratch bytes of [`Self::ckks_prepare_left`], [`Self::ckks_prepare_right`] and their `_into`
    /// forms for an operand shaped like `a`, which [`Self::ckks_mul_tmp_bytes`] of a product reading such an
    /// operand covers, as the API promises.
    fn ckks_prepare_tmp_bytes<A: GLWEInfos>(module: &Module<Self>, a: &A) -> usize;
    /// The bytes [`Self::ckks_prepared_left_alloc_from_infos`] allocates for `infos` under `hint`.
    fn ckks_prepared_left_bytes_of_from_infos<A: GLWEInfos>(module: &Module<Self>, infos: &A, hint: PrepareHint) -> usize;
    /// The bytes [`Self::ckks_prepared_right_alloc_from_infos`] allocates for `infos` under `hint`.
    fn ckks_prepared_right_bytes_of_from_infos<A: GLWEInfos>(module: &Module<Self>, infos: &A, hint: PrepareHint) -> usize;
    /// A left operand of a ciphertext of `infos`, in the representation `hint` selects, which
    /// [`Self::ckks_prepare_left_into`] fills; its infos report `infos` and no metadata until then.
    fn ckks_prepared_left_alloc_from_infos<A: GLWEInfos>(
        module: &Module<Self>,
        infos: &A,
        hint: PrepareHint,
    ) -> CKKSPreparedLeft<Self>;
    /// As [`Self::ckks_prepared_left_alloc_from_infos`], for a right operand.
    fn ckks_prepared_right_alloc_from_infos<A: GLWEInfos>(
        module: &Module<Self>,
        infos: &A,
        hint: PrepareHint,
    ) -> CKKSPreparedRight<Self>;
    /// Prepares `a` into `res`, with no allocation: `res` then reports `a`'s layout and metadata, and the products
    /// that read it give the reference's bits. An `a` of another layout than `res` was allocated for is
    /// `InvalidOperand`, before any write to `res`.
    fn ckks_prepare_left_into<A>(
        module: &Module<Self>,
        res: &mut CKKSPreparedLeft<Self>,
        a: &A,
        scratch: &mut ScratchArena<'_, Self>,
    ) -> Result<()>
    where
        A: GLWEToBackendRef<Self> + CKKSInfos + GLWEInfos;
    /// As [`Self::ckks_prepare_left_into`], into a right operand.
    fn ckks_prepare_right_into<A>(
        module: &Module<Self>,
        res: &mut CKKSPreparedRight<Self>,
        a: &A,
        scratch: &mut ScratchArena<'_, Self>,
    ) -> Result<()>
    where
        A: GLWEToBackendRef<Self> + CKKSInfos + GLWEInfos;
    /// Prepares a left operand: [`Self::ckks_prepare_left_into`] into
    /// [`Self::ckks_prepared_left_alloc_from_infos`] of `a` under `hint`.
    fn ckks_prepare_left<A>(
        module: &Module<Self>,
        a: &A,
        hint: PrepareHint,
        scratch: &mut ScratchArena<'_, Self>,
    ) -> Result<CKKSPreparedLeft<Self>>
    where
        A: GLWEToBackendRef<Self> + CKKSInfos + GLWEInfos,
    {
        super::derived::mul::ckks_prepare_left(module, a, hint, scratch)
    }
    /// Prepares a right operand: [`Self::ckks_prepare_right_into`] into
    /// [`Self::ckks_prepared_right_alloc_from_infos`] of `a` under `hint`.
    fn ckks_prepare_right<A>(
        module: &Module<Self>,
        a: &A,
        hint: PrepareHint,
        scratch: &mut ScratchArena<'_, Self>,
    ) -> Result<CKKSPreparedRight<Self>>
    where
        A: GLWEToBackendRef<Self> + CKKSInfos + GLWEInfos,
    {
        super::derived::mul::ckks_prepare_right(module, a, hint, scratch)
    }
    fn ckks_mul_prepared_assign<Dst, T>(
        module: &Module<Self>,
        dst: &mut Dst,
        prepared: &CKKSPreparedRight<Self>,
        tsk: &T,
        scratch: &mut ScratchArena<'_, Self>,
    ) -> Result<()>
    where
        Dst: GLWEToBackendMut<Self> + GLWEToBackendRef<Self> + CKKSInfos + SetCKKSInfos + GLWEInfos,
        T: GetTensorKey<Self>;
    fn ckks_square_into<Dst, A, T>(
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
    fn ckks_square_assign<Dst, T>(
        module: &Module<Self>,
        dst: &mut Dst,
        tsk: &T,
        scratch: &mut ScratchArena<'_, Self>,
    ) -> Result<()>
    where
        Dst: GLWEToBackendMut<Self> + GLWEToBackendRef<Self> + CKKSInfos + SetCKKSInfos + GLWEInfos,
        T: GetTensorKey<Self>;
    fn ckks_mul_pt_vec_into<Dst, A, P>(
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
    fn ckks_mul_pt_vec_assign<Dst, P>(
        module: &Module<Self>,
        dst: &mut Dst,
        pt: &P,
        scratch: &mut ScratchArena<'_, Self>,
    ) -> Result<()>
    where
        Dst: GLWEToBackendMut<Self> + CKKSInfos + SetCKKSInfos + GLWEInfos,
        P: GLWEToBackendRef<Self> + LWEInfos + IntPolyInfos + CKKSCtBounds;
    fn ckks_mul_pt_const_into<Dst, A, P>(
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
    fn ckks_mul_pt_const_assign<Dst, P>(
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

/// Selects the reference implementation of `CKKSMulImpl` for `$be`.
#[macro_export]
macro_rules! impl_ckks_mul_reference_full {
    ($be:ty) => {
        unsafe impl $crate::oep::CKKSMulImpl for $be {
            fn ckks_mul_tmp_bytes<
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

            fn ckks_square_tmp_bytes<
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

            fn ckks_mul_pt_vec_tmp_bytes<R, A, P>(module: &::poulpy_hal::layouts::Module<Self>, res: &R, a: &A, b: &P) -> usize
            where
                R: $crate::CKKSCtBounds,
                A: $crate::CKKSCtBounds,
                P: $crate::CKKSInfos,
            {
                $crate::reference::mul::CKKSMulReference::ckks_mul_pt_vec_tmp_bytes_reference(module, res, a, b)
            }

            fn ckks_mul_pt_const_tmp_bytes<R, A, P>(module: &::poulpy_hal::layouts::Module<Self>, res: &R, a: &A, b: &P) -> usize
            where
                R: $crate::CKKSCtBounds,
                A: $crate::CKKSCtBounds,
                P: $crate::CKKSInfos,
            {
                $crate::reference::mul::CKKSMulReference::ckks_mul_pt_const_tmp_bytes_reference(module, res, a, b)
            }

            fn ckks_mul_into<Dst, A, B, T>(
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

            fn ckks_mul_assign<Dst, A, T>(
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

            fn ckks_prepare_tmp_bytes<A: ::poulpy_core::layouts::GLWEInfos>(
                module: &::poulpy_hal::layouts::Module<Self>,
                a: &A,
            ) -> usize {
                $crate::reference::mul::CKKSMulReference::ckks_prepare_tmp_bytes_reference(module, a)
            }

            fn ckks_prepared_left_bytes_of_from_infos<A: ::poulpy_core::layouts::GLWEInfos>(
                module: &::poulpy_hal::layouts::Module<Self>,
                infos: &A,
                hint: ::poulpy_hal::layouts::PrepareHint,
            ) -> usize {
                $crate::reference::mul::CKKSMulReference::ckks_prepared_left_bytes_of_from_infos_reference(module, infos, hint)
            }

            fn ckks_prepared_right_bytes_of_from_infos<A: ::poulpy_core::layouts::GLWEInfos>(
                module: &::poulpy_hal::layouts::Module<Self>,
                infos: &A,
                hint: ::poulpy_hal::layouts::PrepareHint,
            ) -> usize {
                $crate::reference::mul::CKKSMulReference::ckks_prepared_right_bytes_of_from_infos_reference(module, infos, hint)
            }

            fn ckks_prepared_left_alloc_from_infos<A: ::poulpy_core::layouts::GLWEInfos>(
                module: &::poulpy_hal::layouts::Module<Self>,
                infos: &A,
                hint: ::poulpy_hal::layouts::PrepareHint,
            ) -> $crate::layouts::CKKSPreparedLeft<Self> {
                $crate::reference::mul::CKKSMulReference::ckks_prepared_left_alloc_from_infos_reference(module, infos, hint)
            }

            fn ckks_prepared_right_alloc_from_infos<A: ::poulpy_core::layouts::GLWEInfos>(
                module: &::poulpy_hal::layouts::Module<Self>,
                infos: &A,
                hint: ::poulpy_hal::layouts::PrepareHint,
            ) -> $crate::layouts::CKKSPreparedRight<Self> {
                $crate::reference::mul::CKKSMulReference::ckks_prepared_right_alloc_from_infos_reference(module, infos, hint)
            }

            fn ckks_prepare_left_into<A>(
                module: &::poulpy_hal::layouts::Module<Self>,
                res: &mut $crate::layouts::CKKSPreparedLeft<Self>,
                a: &A,
                scratch: &mut ::poulpy_hal::layouts::ScratchArena<'_, Self>,
            ) -> $crate::CKKSResult<()>
            where
                A: ::poulpy_core::layouts::GLWEToBackendRef<Self> + $crate::CKKSInfos + ::poulpy_core::layouts::GLWEInfos,
            {
                $crate::reference::mul::CKKSMulReference::ckks_prepare_left_into_reference(module, res, a, scratch)
            }

            fn ckks_prepare_right_into<A>(
                module: &::poulpy_hal::layouts::Module<Self>,
                res: &mut $crate::layouts::CKKSPreparedRight<Self>,
                a: &A,
                scratch: &mut ::poulpy_hal::layouts::ScratchArena<'_, Self>,
            ) -> $crate::CKKSResult<()>
            where
                A: ::poulpy_core::layouts::GLWEToBackendRef<Self> + $crate::CKKSInfos + ::poulpy_core::layouts::GLWEInfos,
            {
                $crate::reference::mul::CKKSMulReference::ckks_prepare_right_into_reference(module, res, a, scratch)
            }

            fn ckks_mul_prepared_assign<Dst, T>(
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

            fn ckks_square_into<Dst, A, T>(
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

            fn ckks_square_assign<Dst, T>(
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

            fn ckks_mul_pt_vec_into<Dst, A, P>(
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

            fn ckks_mul_pt_vec_assign<Dst, P>(
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

            fn ckks_mul_pt_const_into<Dst, A, P>(
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

            fn ckks_mul_pt_const_assign<Dst, P>(
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
pub use crate::impl_ckks_mul_reference_full;
