//! Backend contracts for homomorphic DFT evaluation and matrix generation.
//!
//! Primitive preparation/evaluation use callable algorithms in
//! [`crate::reference::dft`]. Format wrappers are private derived defaults that
//! re-enter the selected generic DFT and CKKS arithmetic operations.

#![allow(clippy::too_many_arguments)]

use poulpy_core::layouts::IntPolyInfos;

use crate::CKKSResult as Result;
use poulpy_core::{
    layouts::{Base2K, GLWEToBackendMut, GLWEToBackendRef, GetAutomorphismKey, LinearTransformation},
    reference::linear_transformation::DiagonalProd,
};
use poulpy_hal::layouts::{Backend, Module, ScratchArena};

use crate::{
    CKKSCtBounds, SetCKKSInfos,
    api::{CKKSEncodingScalar, LtDiagonalScale},
    layouts::{DFTMatrix, DFTMatrixPrepared, DFTPlan, Decode, DftDirection, DftFormat, Encode, Repack, Split, Standard},
};

/// Backend hook for the homomorphic-DFT family.
///
/// # Safety
/// Implementors must preserve the semantics, scratch requirements, aliasing
/// guarantees, and backend bit-parity contract expected by end-to-end pipelines.
pub unsafe trait DFTImpl:
    Backend
    + super::CKKSCopyImpl
    + super::CKKSConjugateImpl
    + super::CKKSAddImpl
    + super::CKKSSubImpl
    + super::CKKSImagImpl
    + super::CKKSRotateImpl
{
    fn ckks_prepare_dft_matrix_impl<Dir, Fmt, P>(
        module: &Module<Self>,
        dft: &DFTMatrix<Self, Dir, Fmt, LinearTransformation<P>>,
        scratch: &mut ScratchArena<'_, Self>,
    ) -> DFTMatrixPrepared<Self, Dir, Fmt>
    where
        P: GLWEToBackendRef<Self> + IntPolyInfos + CKKSCtBounds + DiagonalProd<Self>;

    fn ckks_dft_evaluate_assign_impl<Dir, Fmt, P, Dst, H>(
        module: &Module<Self>,
        ct: &mut Dst,
        dft: &DFTMatrix<Self, Dir, Fmt, LinearTransformation<P>>,
        keys: &H,
        scratch: &mut ScratchArena<'_, Self>,
    ) -> Result<()>
    where
        P: DiagonalProd<Self> + LtDiagonalScale + IntPolyInfos,
        Dst: GLWEToBackendMut<Self> + GLWEToBackendRef<Self> + CKKSCtBounds + SetCKKSInfos,
        H: GetAutomorphismKey<Self>;

    fn ckks_coeffs_to_slots_impl<P, Dst, H>(
        module: &Module<Self>,
        ct: &mut Dst,
        dft: &DFTMatrix<Self, Encode, Standard, LinearTransformation<P>>,
        keys: &H,
        scratch: &mut ScratchArena<'_, Self>,
    ) -> Result<()>
    where
        P: DiagonalProd<Self> + LtDiagonalScale + IntPolyInfos,
        Dst: GLWEToBackendMut<Self> + GLWEToBackendRef<Self> + CKKSCtBounds + SetCKKSInfos,
        H: GetAutomorphismKey<Self>,
    {
        crate::oep::derived::dft::ckks_coeffs_to_slots_assign(module, ct, dft, keys, scratch)
    }

    fn ckks_slots_to_coeffs_impl<P, Dst, H>(
        module: &Module<Self>,
        ct: &mut Dst,
        dft: &DFTMatrix<Self, Decode, Standard, LinearTransformation<P>>,
        keys: &H,
        scratch: &mut ScratchArena<'_, Self>,
    ) -> Result<()>
    where
        P: DiagonalProd<Self> + LtDiagonalScale + IntPolyInfos,
        Dst: GLWEToBackendMut<Self> + GLWEToBackendRef<Self> + CKKSCtBounds + SetCKKSInfos,
        H: GetAutomorphismKey<Self>,
    {
        crate::oep::derived::dft::ckks_slots_to_coeffs_assign(module, ct, dft, keys, scratch)
    }

    fn ckks_coeffs_to_slots_split_impl<P, Dst, Src, H>(
        module: &Module<Self>,
        ct_real: &mut Dst,
        ct_imag: &mut Dst,
        ct_in: &Src,
        dft: &DFTMatrix<Self, Encode, Split, LinearTransformation<P>>,
        keys: &H,
        scratch: &mut ScratchArena<'_, Self>,
    ) -> Result<()>
    where
        P: DiagonalProd<Self> + LtDiagonalScale + IntPolyInfos,
        Dst: GLWEToBackendMut<Self> + GLWEToBackendRef<Self> + CKKSCtBounds + SetCKKSInfos,
        Src: GLWEToBackendRef<Self> + CKKSCtBounds,
        H: GetAutomorphismKey<Self>,
    {
        crate::oep::derived::dft::ckks_coeffs_to_slots_split(module, ct_real, ct_imag, ct_in, dft, keys, scratch)
    }

    fn ckks_slots_to_coeffs_split_impl<P, Dst, Src, H>(
        module: &Module<Self>,
        op_out: &mut Dst,
        ct_real: &Src,
        ct_imag: &Src,
        dft: &DFTMatrix<Self, Decode, Split, LinearTransformation<P>>,
        keys: &H,
        scratch: &mut ScratchArena<'_, Self>,
    ) -> Result<()>
    where
        P: DiagonalProd<Self> + LtDiagonalScale + IntPolyInfos,
        Dst: GLWEToBackendMut<Self> + GLWEToBackendRef<Self> + CKKSCtBounds + SetCKKSInfos,
        Src: GLWEToBackendRef<Self> + CKKSCtBounds,
        H: GetAutomorphismKey<Self>,
    {
        crate::oep::derived::dft::ckks_slots_to_coeffs_split(module, op_out, ct_real, ct_imag, dft, keys, scratch)
    }

    fn ckks_coeffs_to_slots_repack_impl<P, Dst, Src, H>(
        module: &Module<Self>,
        ct_out: &mut Dst,
        ct_in: &Src,
        dft: &DFTMatrix<Self, Encode, Repack, LinearTransformation<P>>,
        keys: &H,
        scratch: &mut ScratchArena<'_, Self>,
    ) -> Result<()>
    where
        P: DiagonalProd<Self> + LtDiagonalScale + IntPolyInfos,
        Dst: GLWEToBackendMut<Self> + GLWEToBackendRef<Self> + CKKSCtBounds + SetCKKSInfos,
        Src: GLWEToBackendRef<Self> + CKKSCtBounds,
        H: GetAutomorphismKey<Self>,
    {
        crate::oep::derived::dft::ckks_coeffs_to_slots_repack(module, ct_out, ct_in, dft, keys, scratch)
    }

    fn ckks_slots_to_coeffs_repack_impl<P, Dst, Src, H>(
        module: &Module<Self>,
        op_out: &mut Dst,
        ct_in: &Src,
        dft: &DFTMatrix<Self, Decode, Repack, LinearTransformation<P>>,
        keys: &H,
        scratch: &mut ScratchArena<'_, Self>,
    ) -> Result<()>
    where
        P: DiagonalProd<Self> + LtDiagonalScale + IntPolyInfos,
        Dst: GLWEToBackendMut<Self> + GLWEToBackendRef<Self> + CKKSCtBounds + SetCKKSInfos,
        Src: GLWEToBackendRef<Self> + CKKSCtBounds,
        H: GetAutomorphismKey<Self>,
    {
        crate::oep::derived::dft::ckks_slots_to_coeffs_repack(module, op_out, ct_in, dft, keys, scratch)
    }
}

/// Backend hook for homomorphic-DFT matrix generation at scalar precision `F`.
///
/// Separate from [`DFTImpl`] so scalar precision belongs to the contract.
/// Reference matrix generation carries its own scalar and encoding bounds;
/// custom implementations only need the method's backend-resident signature.
///
/// # Safety
/// Implementors must preserve the semantics, scratch requirements, aliasing
/// guarantees, and backend bit-parity contract expected by end-to-end pipelines.
pub unsafe trait DFTMatrixImpl<F: CKKSEncodingScalar>: Backend {
    fn ckks_new_dft_matrix_impl<Dir, Fmt>(
        module: &Module<Self>,
        base2k: Base2K,
        literal: &DFTPlan,
        scratch: &mut ScratchArena<'_, Self>,
    ) -> Result<DFTMatrix<Self, Dir, Fmt>>
    where
        Dir: DftDirection,
        Fmt: DftFormat;
}

/// Wires matrix generation and primitive DFT evaluation to their reference algorithms.
#[macro_export]
macro_rules! impl_ckks_dft_reference {
    ($be:ty) => {
        unsafe impl $crate::oep::DFTImpl for $be {
            fn ckks_prepare_dft_matrix_impl<Dir, Fmt, P>(
                module: &::poulpy_hal::layouts::Module<Self>,
                dft: &$crate::layouts::DFTMatrix<Self, Dir, Fmt, ::poulpy_core::layouts::LinearTransformation<P>>,
                scratch: &mut ::poulpy_hal::layouts::ScratchArena<'_, Self>,
            ) -> $crate::layouts::DFTMatrixPrepared<Self, Dir, Fmt>
            where
                P: ::poulpy_core::layouts::GLWEToBackendRef<Self>
                    + ::poulpy_core::layouts::IntPolyInfos
                    + $crate::CKKSCtBounds
                    + ::poulpy_core::reference::linear_transformation::DiagonalProd<Self>,
            {
                $crate::reference::dft::ckks_prepare_dft_matrix::<Dir, Fmt, Self, P>(module, dft, scratch)
            }
            fn ckks_dft_evaluate_assign_impl<Dir, Fmt, P, Dst, H>(
                module: &::poulpy_hal::layouts::Module<Self>,
                ct: &mut Dst,
                dft: &$crate::layouts::DFTMatrix<Self, Dir, Fmt, ::poulpy_core::layouts::LinearTransformation<P>>,
                keys: &H,
                scratch: &mut ::poulpy_hal::layouts::ScratchArena<'_, Self>,
            ) -> $crate::CKKSResult<()>
            where
                P: ::poulpy_core::reference::linear_transformation::DiagonalProd<Self>
                    + $crate::api::LtDiagonalScale
                    + ::poulpy_core::layouts::IntPolyInfos,
                Dst: ::poulpy_core::layouts::GLWEToBackendMut<Self>
                    + ::poulpy_core::layouts::GLWEToBackendRef<Self>
                    + $crate::CKKSCtBounds
                    + $crate::SetCKKSInfos,
                H: ::poulpy_core::layouts::GetAutomorphismKey<Self>,
            {
                $crate::reference::dft::ckks_dft_evaluate_assign(module, ct, dft, keys, scratch)
            }
        }
        unsafe impl<F: $crate::api::CKKSEncodingScalar + $crate::reference::dft::DftScalar> $crate::oep::DFTMatrixImpl<F> for $be
        where
            $be: $crate::oep::CKKSEncodingImpl<F>,
        {
            fn ckks_new_dft_matrix_impl<Dir, Fmt>(
                module: &::poulpy_hal::layouts::Module<Self>,
                base2k: ::poulpy_core::layouts::Base2K,
                literal: &$crate::layouts::DFTPlan,
                scratch: &mut ::poulpy_hal::layouts::ScratchArena<'_, Self>,
            ) -> $crate::CKKSResult<$crate::layouts::DFTMatrix<Self, Dir, Fmt>>
            where
                Dir: $crate::layouts::DftDirection,
                Fmt: $crate::layouts::DftFormat,
            {
                $crate::reference::dft::ckks_new_dft_matrix::<Dir, Fmt, Self, F>(module, base2k, literal, scratch)
            }
        }
    };
}
pub use crate::impl_ckks_dft_reference;
