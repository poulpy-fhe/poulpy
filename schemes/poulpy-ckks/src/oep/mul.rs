use crate::CKKSResult as Result;
use poulpy_core::layouts::GetTensorKey;
use poulpy_core::layouts::IntPolyInfos;

use poulpy_core::layouts::{GGLWEInfos, GLWEInfos, LWEInfos};
use poulpy_hal::layouts::{Backend, Module, PrepareHint, ScratchArena};

use crate::{
    CKKSCtBounds, CKKSInfos, GLWEToBackendMut, GLWEToBackendRef, SetCKKSInfos,
    layouts::{CKKSLeftOperandPrepared, CKKSRightOperandPrepared},
};

/// Backend contract of [`CKKSMulOps`](crate::api::CKKSMulOps).
///
/// # Safety
/// Implementations reproduce the bodies their methods document bit for bit, report for each
/// `_tmp_bytes` query a bound that covers the body, and refuse as the reference does. The memory,
/// layout, synchrony and aliasing rules of `poulpy-hal/docs/backend_safety_contract.md` apply.
pub unsafe trait CKKSMulImpl: Backend {
    /// Scratch bound of [`Self::ckks_mul_into`], [`Self::ckks_mul_assign`] and [`Self::ckks_mul_prepared_assign`]; may
    /// report less than the reference, never less than the body takes. The operands are passed, a prepared right
    /// operand through its infos.
    fn ckks_mul_tmp_bytes<R: GLWEInfos, A: GLWEInfos, B: GLWEInfos, T: GGLWEInfos>(
        module: &Module<Self>,
        res: &R,
        a: &A,
        b: &B,
        tsk: &T,
    ) -> usize;

    /// Scratch bound of [`Self::ckks_square_into`] and [`Self::ckks_square_assign`]; may report less than the
    /// reference, never less than the body takes.
    fn ckks_square_tmp_bytes<R: GLWEInfos, A: GLWEInfos, T: GGLWEInfos>(module: &Module<Self>, res: &R, a: &A, tsk: &T) -> usize;

    /// Scratch bound of [`Self::ckks_mul_pt_vec_into`] and [`Self::ckks_mul_pt_vec_assign`]; may report less than the
    /// reference, never less than the body takes.
    fn ckks_mul_pt_vec_tmp_bytes<R, A, P>(module: &Module<Self>, res: &R, a: &A, b: &P) -> usize
    where
        R: CKKSCtBounds,
        A: CKKSCtBounds,
        P: CKKSInfos;

    /// Scratch bound of [`Self::ckks_mul_pt_const_into`] and [`Self::ckks_mul_pt_const_assign`]; may report less than
    /// the reference, never less than the body takes.
    fn ckks_mul_pt_const_tmp_bytes<R, A, P>(module: &Module<Self>, res: &R, a: &A, b: &P) -> usize
    where
        R: CKKSCtBounds,
        A: CKKSCtBounds,
        P: CKKSInfos;

    /// Reproduces [`ckks_mul_into_reference`](crate::reference::mul::CKKSMulReference::ckks_mul_into_reference) bit for
    /// bit, within [`Self::ckks_mul_tmp_bytes`]; free in kernels and scratch use.
    fn ckks_mul_into<Dst, A, B, H>(
        module: &Module<Self>,
        dst: &mut Dst,
        a: &A,
        b: &B,
        tsk: &H,
        scratch: &mut ScratchArena<'_, Self>,
    ) -> Result<()>
    where
        Dst: GLWEToBackendMut<Self> + CKKSInfos + SetCKKSInfos + GLWEInfos,
        A: GLWEToBackendRef<Self> + CKKSInfos + GLWEInfos,
        B: GLWEToBackendRef<Self> + CKKSInfos + GLWEInfos,
        H: GetTensorKey<Self>;

    /// Reproduces [`ckks_mul_assign_reference`](crate::reference::mul::CKKSMulReference::ckks_mul_assign_reference) bit
    /// for bit, within [`Self::ckks_mul_tmp_bytes`]; free in kernels and scratch use.
    fn ckks_mul_assign<Dst, A, H>(
        module: &Module<Self>,
        dst: &mut Dst,
        a: &A,
        tsk: &H,
        scratch: &mut ScratchArena<'_, Self>,
    ) -> Result<()>
    where
        Dst: GLWEToBackendMut<Self> + GLWEToBackendRef<Self> + CKKSInfos + SetCKKSInfos + GLWEInfos,
        A: GLWEToBackendRef<Self> + CKKSInfos + GLWEInfos,
        H: GetTensorKey<Self>;

    /// Scratch bound of [`Self::ckks_left_operand_prepare`] and [`Self::ckks_left_operand_prepare_alloc`] for an operand
    /// shaped like `a`; may report less than the reference, never less than the body takes.
    /// [`Self::ckks_mul_tmp_bytes`] of a product reading such an operand covers it, as the API promises.
    fn ckks_left_operand_prepare_tmp_bytes<A: GLWEInfos>(module: &Module<Self>, a: &A) -> usize;

    /// Scratch bound of [`Self::ckks_right_operand_prepare`] and [`Self::ckks_right_operand_prepare_alloc`] for an
    /// operand shaped like `a`; may report less than the reference, never less than the body takes.
    /// [`Self::ckks_mul_tmp_bytes`] of a product reading such an operand covers it, as the API promises.
    fn ckks_right_operand_prepare_tmp_bytes<A: GLWEInfos>(module: &Module<Self>, a: &A) -> usize;

    /// Reproduces
    /// [`ckks_left_operand_prepare_reference`](crate::reference::mul::CKKSMulReference::ckks_left_operand_prepare_reference)
    /// bit for bit, within [`Self::ckks_left_operand_prepare_tmp_bytes`]; free in kernels and scratch use. Prepares `a`
    /// into `res`, with no allocation: `res` then reports `a`'s layout and metadata, and the products that read it give
    /// the reference's bits. An `a` of another layout than `res` was allocated for is `InvalidOperand`, before any write
    /// to `res`.
    fn ckks_left_operand_prepare<A>(
        module: &Module<Self>,
        res: &mut CKKSLeftOperandPrepared<Self>,
        a: &A,
        scratch: &mut ScratchArena<'_, Self>,
    ) -> Result<()>
    where
        A: GLWEToBackendRef<Self> + CKKSInfos + GLWEInfos;

    /// Reproduces
    /// [`ckks_right_operand_prepare_reference`](crate::reference::mul::CKKSMulReference::ckks_right_operand_prepare_reference)
    /// bit for bit, within [`Self::ckks_right_operand_prepare_tmp_bytes`]; free in kernels and scratch use. As
    /// [`Self::ckks_left_operand_prepare`], into a right operand.
    fn ckks_right_operand_prepare<A>(
        module: &Module<Self>,
        res: &mut CKKSRightOperandPrepared<Self>,
        a: &A,
        scratch: &mut ScratchArena<'_, Self>,
    ) -> Result<()>
    where
        A: GLWEToBackendRef<Self> + CKKSInfos + GLWEInfos;

    /// Default body: `ckks_left_operand_prepare_alloc_derived`, a left operand allocated from `a` under `hint` with
    /// [`ckks_left_operand_prepared_alloc_from_infos`](crate::api::CKKSLeftOperandPreparedFactory::ckks_left_operand_prepared_alloc_from_infos),
    /// then [`Self::ckks_left_operand_prepare`] into it.
    fn ckks_left_operand_prepare_alloc<A>(
        module: &Module<Self>,
        a: &A,
        hint: PrepareHint,
        scratch: &mut ScratchArena<'_, Self>,
    ) -> Result<CKKSLeftOperandPrepared<Self>>
    where
        A: GLWEToBackendRef<Self> + CKKSInfos + GLWEInfos,
    {
        super::derived::mul::ckks_left_operand_prepare_alloc_derived(module, a, hint, scratch)
    }

    /// Default body: `ckks_right_operand_prepare_alloc_derived`, a right operand allocated from `a` under `hint` with
    /// [`ckks_right_operand_prepared_alloc_from_infos`](crate::api::CKKSRightOperandPreparedFactory::ckks_right_operand_prepared_alloc_from_infos),
    /// then [`Self::ckks_right_operand_prepare`] into it.
    fn ckks_right_operand_prepare_alloc<A>(
        module: &Module<Self>,
        a: &A,
        hint: PrepareHint,
        scratch: &mut ScratchArena<'_, Self>,
    ) -> Result<CKKSRightOperandPrepared<Self>>
    where
        A: GLWEToBackendRef<Self> + CKKSInfos + GLWEInfos,
    {
        super::derived::mul::ckks_right_operand_prepare_alloc_derived(module, a, hint, scratch)
    }

    /// Reproduces
    /// [`ckks_mul_prepared_assign_reference`](crate::reference::mul::CKKSMulReference::ckks_mul_prepared_assign_reference)
    /// bit for bit, within [`Self::ckks_mul_tmp_bytes`]; free in kernels and scratch use.
    fn ckks_mul_prepared_assign<Dst, H>(
        module: &Module<Self>,
        dst: &mut Dst,
        prepared: &CKKSRightOperandPrepared<Self>,
        tsk: &H,
        scratch: &mut ScratchArena<'_, Self>,
    ) -> Result<()>
    where
        Dst: GLWEToBackendMut<Self> + GLWEToBackendRef<Self> + CKKSInfos + SetCKKSInfos + GLWEInfos,
        H: GetTensorKey<Self>;

    /// Reproduces [`ckks_square_into_reference`](crate::reference::mul::CKKSMulReference::ckks_square_into_reference)
    /// bit for bit, within [`Self::ckks_square_tmp_bytes`]; free in kernels and scratch use.
    fn ckks_square_into<Dst, A, H>(
        module: &Module<Self>,
        dst: &mut Dst,
        a: &A,
        tsk: &H,
        scratch: &mut ScratchArena<'_, Self>,
    ) -> Result<()>
    where
        Dst: GLWEToBackendMut<Self> + CKKSInfos + SetCKKSInfos + GLWEInfos,
        A: GLWEToBackendRef<Self> + CKKSInfos + GLWEInfos,
        H: GetTensorKey<Self>;

    /// Reproduces
    /// [`ckks_square_assign_reference`](crate::reference::mul::CKKSMulReference::ckks_square_assign_reference) bit for
    /// bit, within [`Self::ckks_square_tmp_bytes`]; free in kernels and scratch use.
    fn ckks_square_assign<Dst, H>(
        module: &Module<Self>,
        dst: &mut Dst,
        tsk: &H,
        scratch: &mut ScratchArena<'_, Self>,
    ) -> Result<()>
    where
        Dst: GLWEToBackendMut<Self> + GLWEToBackendRef<Self> + CKKSInfos + SetCKKSInfos + GLWEInfos,
        H: GetTensorKey<Self>;

    /// Reproduces
    /// [`ckks_mul_pt_vec_into_reference`](crate::reference::mul::CKKSMulReference::ckks_mul_pt_vec_into_reference) bit
    /// for bit, within [`Self::ckks_mul_pt_vec_tmp_bytes`]; free in kernels and scratch use.
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

    /// Reproduces
    /// [`ckks_mul_pt_vec_assign_reference`](crate::reference::mul::CKKSMulReference::ckks_mul_pt_vec_assign_reference)
    /// bit for bit, within [`Self::ckks_mul_pt_vec_tmp_bytes`]; free in kernels and scratch use.
    fn ckks_mul_pt_vec_assign<Dst, P>(
        module: &Module<Self>,
        dst: &mut Dst,
        pt: &P,
        scratch: &mut ScratchArena<'_, Self>,
    ) -> Result<()>
    where
        Dst: GLWEToBackendMut<Self> + CKKSInfos + SetCKKSInfos + GLWEInfos,
        P: GLWEToBackendRef<Self> + LWEInfos + IntPolyInfos + CKKSCtBounds;

    /// Reproduces
    /// [`ckks_mul_pt_const_into_reference`](crate::reference::mul::CKKSMulReference::ckks_mul_pt_const_into_reference)
    /// bit for bit, within [`Self::ckks_mul_pt_const_tmp_bytes`]; free in kernels and scratch use.
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

    /// Reproduces
    /// [`ckks_mul_pt_const_assign_reference`](crate::reference::mul::CKKSMulReference::ckks_mul_pt_const_assign_reference)
    /// bit for bit, within [`Self::ckks_mul_pt_const_tmp_bytes`]; free in kernels and scratch use.
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

            fn ckks_mul_into<Dst, A, B, H>(
                module: &::poulpy_hal::layouts::Module<Self>,
                dst: &mut Dst,
                a: &A,
                b: &B,
                tsk: &H,
                scratch: &mut ::poulpy_hal::layouts::ScratchArena<'_, Self>,
            ) -> $crate::CKKSResult<()>
            where
                Dst: ::poulpy_core::layouts::GLWEToBackendMut<Self>
                    + $crate::CKKSInfos
                    + $crate::SetCKKSInfos
                    + ::poulpy_core::layouts::GLWEInfos,
                A: ::poulpy_core::layouts::GLWEToBackendRef<Self> + $crate::CKKSInfos + ::poulpy_core::layouts::GLWEInfos,
                B: ::poulpy_core::layouts::GLWEToBackendRef<Self> + $crate::CKKSInfos + ::poulpy_core::layouts::GLWEInfos,
                H: ::poulpy_core::layouts::GetTensorKey<Self>,
            {
                $crate::reference::mul::CKKSMulReference::ckks_mul_into_reference(module, dst, a, b, tsk, scratch)
            }

            fn ckks_mul_assign<Dst, A, H>(
                module: &::poulpy_hal::layouts::Module<Self>,
                dst: &mut Dst,
                a: &A,
                tsk: &H,
                scratch: &mut ::poulpy_hal::layouts::ScratchArena<'_, Self>,
            ) -> $crate::CKKSResult<()>
            where
                Dst: ::poulpy_core::layouts::GLWEToBackendMut<Self>
                    + ::poulpy_core::layouts::GLWEToBackendRef<Self>
                    + $crate::CKKSInfos
                    + $crate::SetCKKSInfos
                    + ::poulpy_core::layouts::GLWEInfos,
                A: ::poulpy_core::layouts::GLWEToBackendRef<Self> + $crate::CKKSInfos + ::poulpy_core::layouts::GLWEInfos,
                H: ::poulpy_core::layouts::GetTensorKey<Self>,
            {
                $crate::reference::mul::CKKSMulReference::ckks_mul_assign_reference(module, dst, a, tsk, scratch)
            }

            fn ckks_left_operand_prepare_tmp_bytes<A: ::poulpy_core::layouts::GLWEInfos>(
                module: &::poulpy_hal::layouts::Module<Self>,
                a: &A,
            ) -> usize {
                $crate::reference::mul::CKKSMulReference::ckks_left_operand_prepare_tmp_bytes_reference(module, a)
            }

            fn ckks_right_operand_prepare_tmp_bytes<A: ::poulpy_core::layouts::GLWEInfos>(
                module: &::poulpy_hal::layouts::Module<Self>,
                a: &A,
            ) -> usize {
                $crate::reference::mul::CKKSMulReference::ckks_right_operand_prepare_tmp_bytes_reference(module, a)
            }

            fn ckks_left_operand_prepare<A>(
                module: &::poulpy_hal::layouts::Module<Self>,
                res: &mut $crate::layouts::CKKSLeftOperandPrepared<Self>,
                a: &A,
                scratch: &mut ::poulpy_hal::layouts::ScratchArena<'_, Self>,
            ) -> $crate::CKKSResult<()>
            where
                A: ::poulpy_core::layouts::GLWEToBackendRef<Self> + $crate::CKKSInfos + ::poulpy_core::layouts::GLWEInfos,
            {
                $crate::reference::mul::CKKSMulReference::ckks_left_operand_prepare_reference(module, res, a, scratch)
            }

            fn ckks_right_operand_prepare<A>(
                module: &::poulpy_hal::layouts::Module<Self>,
                res: &mut $crate::layouts::CKKSRightOperandPrepared<Self>,
                a: &A,
                scratch: &mut ::poulpy_hal::layouts::ScratchArena<'_, Self>,
            ) -> $crate::CKKSResult<()>
            where
                A: ::poulpy_core::layouts::GLWEToBackendRef<Self> + $crate::CKKSInfos + ::poulpy_core::layouts::GLWEInfos,
            {
                $crate::reference::mul::CKKSMulReference::ckks_right_operand_prepare_reference(module, res, a, scratch)
            }

            fn ckks_mul_prepared_assign<Dst, H>(
                module: &::poulpy_hal::layouts::Module<Self>,
                dst: &mut Dst,
                prepared: &$crate::layouts::CKKSRightOperandPrepared<Self>,
                tsk: &H,
                scratch: &mut ::poulpy_hal::layouts::ScratchArena<'_, Self>,
            ) -> $crate::CKKSResult<()>
            where
                Dst: ::poulpy_core::layouts::GLWEToBackendMut<Self>
                    + ::poulpy_core::layouts::GLWEToBackendRef<Self>
                    + $crate::CKKSInfos
                    + $crate::SetCKKSInfos
                    + ::poulpy_core::layouts::GLWEInfos,
                H: ::poulpy_core::layouts::GetTensorKey<Self>,
            {
                $crate::reference::mul::CKKSMulReference::ckks_mul_prepared_assign_reference(module, dst, prepared, tsk, scratch)
            }

            fn ckks_square_into<Dst, A, H>(
                module: &::poulpy_hal::layouts::Module<Self>,
                dst: &mut Dst,
                a: &A,
                tsk: &H,
                scratch: &mut ::poulpy_hal::layouts::ScratchArena<'_, Self>,
            ) -> $crate::CKKSResult<()>
            where
                Dst: ::poulpy_core::layouts::GLWEToBackendMut<Self>
                    + $crate::CKKSInfos
                    + $crate::SetCKKSInfos
                    + ::poulpy_core::layouts::GLWEInfos,
                A: ::poulpy_core::layouts::GLWEToBackendRef<Self> + $crate::CKKSInfos + ::poulpy_core::layouts::GLWEInfos,
                H: ::poulpy_core::layouts::GetTensorKey<Self>,
            {
                $crate::reference::mul::CKKSMulReference::ckks_square_into_reference(module, dst, a, tsk, scratch)
            }

            fn ckks_square_assign<Dst, H>(
                module: &::poulpy_hal::layouts::Module<Self>,
                dst: &mut Dst,
                tsk: &H,
                scratch: &mut ::poulpy_hal::layouts::ScratchArena<'_, Self>,
            ) -> $crate::CKKSResult<()>
            where
                Dst: ::poulpy_core::layouts::GLWEToBackendMut<Self>
                    + ::poulpy_core::layouts::GLWEToBackendRef<Self>
                    + $crate::CKKSInfos
                    + $crate::SetCKKSInfos
                    + ::poulpy_core::layouts::GLWEInfos,
                H: ::poulpy_core::layouts::GetTensorKey<Self>,
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
