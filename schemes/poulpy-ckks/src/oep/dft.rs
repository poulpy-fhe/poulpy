//! Backend contracts for homomorphic DFT evaluation and matrix generation.
//!
//! Primitive preparation/evaluation use callable algorithms in
//! [`crate::reference::dft`]. Format wrappers re-enter the selected generic DFT and CKKS arithmetic operations.

#![allow(clippy::too_many_arguments)]

use poulpy_core::layouts::IntPolyInfos;

use crate::CKKSResult as Result;
use poulpy_core::{
    DiagonalProd,
    layouts::{Base2K, GLWEToBackendMut, GLWEToBackendRef, GetAutomorphismKey, LinearTransformation},
};
use poulpy_hal::layouts::{Backend, Module, ScratchArena};

use crate::{
    CKKSCtBounds, SetCKKSInfos,
    api::{CKKSEncodingScalar, LtDiagonalMeta},
    layouts::{DFTMatrix, DFTMatrixPrepared, DFTPlan, Decode, DftDirection, DftFormat, Encode, Repack, Split, Standard},
};

/// Backend hook for the homomorphic-DFT family.
///
/// # Safety
/// Implementations reproduce the bodies their methods document bit for bit, report for each
/// `_tmp_bytes` query a bound that covers the body, and refuse as the reference does. The memory,
/// layout, synchrony and aliasing rules of `poulpy-hal/docs/backend_safety_contract.md` apply.
pub unsafe trait DFTImpl:
    Backend
    + super::CKKSCopyImpl
    + super::CKKSConjugateImpl
    + super::CKKSAddImpl
    + super::CKKSSubImpl
    + super::CKKSImagImpl
    + super::CKKSRotateImpl
{
    /// Scratch bound of [`Self::ckks_prepare_dft_matrix`] and [`Self::ckks_prepare_dft_matrix_alloc`]; may report less
    /// than the reference, never less than the body takes.
    fn ckks_prepare_dft_matrix_tmp_bytes<Dir, Fmt, P>(
        module: &Module<Self>,
        dft: &DFTMatrix<Self, Dir, Fmt, LinearTransformation<P>>,
    ) -> usize
    where
        P: poulpy_core::layouts::LWEInfos;

    /// Scratch bound of [`Self::ckks_dft_evaluate_assign`], [`Self::ckks_coeffs_to_slots`],
    /// [`Self::ckks_slots_to_coeffs`], [`Self::ckks_coeffs_to_slots_split`], [`Self::ckks_slots_to_coeffs_split`],
    /// [`Self::ckks_coeffs_to_slots_repack`] and [`Self::ckks_slots_to_coeffs_repack`]; may report less than the
    /// reference, never less than the body takes.
    fn ckks_dft_tmp_bytes<Dir, Fmt, P, Dst, Src, K>(
        module: &Module<Self>,
        dst: &Dst,
        src: &Src,
        dft: &DFTMatrix<Self, Dir, Fmt, LinearTransformation<P>>,
        key: &K,
    ) -> usize
    where
        P: poulpy_core::layouts::GLWEInfos,
        Dst: CKKSCtBounds,
        Src: CKKSCtBounds,
        K: poulpy_core::layouts::GGLWEInfos;

    /// Reproduces [`ckks_prepare_dft_matrix`](crate::reference::dft::ckks_prepare_dft_matrix) bit for bit, within
    /// [`Self::ckks_prepare_dft_matrix_tmp_bytes`]; free in kernels, prepared form and scratch use. A `res` allocated
    /// for another matrix layout, or a factor whose diagonals do not share one scale, width and slot kind, is
    /// `InvalidOperand`, before any write.
    fn ckks_prepare_dft_matrix<Dir, Fmt, P>(
        module: &Module<Self>,
        res: &mut DFTMatrixPrepared<Self, Dir, Fmt>,
        dft: &DFTMatrix<Self, Dir, Fmt, LinearTransformation<P>>,
        scratch: &mut ScratchArena<'_, Self>,
    ) -> Result<()>
    where
        P: GLWEToBackendRef<Self> + IntPolyInfos + CKKSCtBounds + DiagonalProd<Self>;

    /// Default body: `ckks_prepare_dft_matrix_alloc_derived`, a prepared matrix allocated from `dft` with
    /// [`ckks_dft_matrix_prepared_alloc_from_infos`](crate::layouts::CKKSModuleAlloc::ckks_dft_matrix_prepared_alloc_from_infos),
    /// then [`Self::ckks_prepare_dft_matrix`] into it.
    fn ckks_prepare_dft_matrix_alloc<Dir, Fmt, P>(
        module: &Module<Self>,
        dft: &DFTMatrix<Self, Dir, Fmt, LinearTransformation<P>>,
        scratch: &mut ScratchArena<'_, Self>,
    ) -> Result<DFTMatrixPrepared<Self, Dir, Fmt>>
    where
        P: GLWEToBackendRef<Self> + IntPolyInfos + CKKSCtBounds + DiagonalProd<Self>,
    {
        crate::oep::derived::dft::ckks_prepare_dft_matrix_alloc_derived(module, dft, scratch)
    }

    /// Reproduces [`ckks_dft_evaluate_assign`](crate::reference::dft::ckks_dft_evaluate_assign) bit for bit, within
    /// [`Self::ckks_dft_tmp_bytes`]; free in kernels and scratch use.
    fn ckks_dft_evaluate_assign<Dir, Fmt, P, Dst, H>(
        module: &Module<Self>,
        ct: &mut Dst,
        dft: &DFTMatrix<Self, Dir, Fmt, LinearTransformation<P>>,
        keys: &H,
        scratch: &mut ScratchArena<'_, Self>,
    ) -> Result<()>
    where
        P: DiagonalProd<Self> + LtDiagonalMeta + IntPolyInfos,
        Dst: GLWEToBackendMut<Self> + GLWEToBackendRef<Self> + CKKSCtBounds + SetCKKSInfos,
        H: GetAutomorphismKey<Self>;

    /// Default body: `ckks_coeffs_to_slots_assign_derived`, the selected `ckks_dft_evaluate_assign` of the encode
    /// matrix.
    fn ckks_coeffs_to_slots<P, Dst, H>(
        module: &Module<Self>,
        ct: &mut Dst,
        dft: &DFTMatrix<Self, Encode, Standard, LinearTransformation<P>>,
        keys: &H,
        scratch: &mut ScratchArena<'_, Self>,
    ) -> Result<()>
    where
        P: DiagonalProd<Self> + LtDiagonalMeta + IntPolyInfos,
        Dst: GLWEToBackendMut<Self> + GLWEToBackendRef<Self> + CKKSCtBounds + SetCKKSInfos,
        H: GetAutomorphismKey<Self>,
    {
        crate::oep::derived::dft::ckks_coeffs_to_slots_assign_derived(module, ct, dft, keys, scratch)
    }

    /// Default body: `ckks_slots_to_coeffs_assign_derived`, the selected `ckks_dft_evaluate_assign` of the decode
    /// matrix.
    fn ckks_slots_to_coeffs<P, Dst, H>(
        module: &Module<Self>,
        ct: &mut Dst,
        dft: &DFTMatrix<Self, Decode, Standard, LinearTransformation<P>>,
        keys: &H,
        scratch: &mut ScratchArena<'_, Self>,
    ) -> Result<()>
    where
        P: DiagonalProd<Self> + LtDiagonalMeta + IntPolyInfos,
        Dst: GLWEToBackendMut<Self> + GLWEToBackendRef<Self> + CKKSCtBounds + SetCKKSInfos,
        H: GetAutomorphismKey<Self>,
    {
        crate::oep::derived::dft::ckks_slots_to_coeffs_assign_derived(module, ct, dft, keys, scratch)
    }

    /// Default body: `ckks_coeffs_to_slots_split_derived`, `ckks_copy` of `ct_in` into `ct_real`,
    /// `ckks_dft_evaluate_assign` on it, `ckks_conjugate_into` of the result into `ct_imag`, `ckks_sub_into` of the two
    /// into an allocated temporary, `ckks_add_assign` of `ct_imag` into `ct_real`, `ckks_div_i_into` of the temporary
    /// into `ct_imag`, then both slot kinds set to real.
    fn ckks_coeffs_to_slots_split<P, Dst, Src, H>(
        module: &Module<Self>,
        ct_real: &mut Dst,
        ct_imag: &mut Dst,
        ct_in: &Src,
        dft: &DFTMatrix<Self, Encode, Split, LinearTransformation<P>>,
        keys: &H,
        scratch: &mut ScratchArena<'_, Self>,
    ) -> Result<()>
    where
        P: DiagonalProd<Self> + LtDiagonalMeta + IntPolyInfos,
        Dst: GLWEToBackendMut<Self> + GLWEToBackendRef<Self> + CKKSCtBounds + SetCKKSInfos,
        Src: GLWEToBackendRef<Self> + CKKSCtBounds,
        H: GetAutomorphismKey<Self>,
    {
        crate::oep::derived::dft::ckks_coeffs_to_slots_split_derived(module, ct_real, ct_imag, ct_in, dft, keys, scratch)
    }

    /// Default body: `ckks_slots_to_coeffs_split_derived`, `ckks_mul_i_into` of `ct_imag` into `op_out`,
    /// `ckks_add_assign` of `ct_real`, then `ckks_dft_evaluate_assign` on `op_out`.
    fn ckks_slots_to_coeffs_split<P, Dst, Src, H>(
        module: &Module<Self>,
        op_out: &mut Dst,
        ct_real: &Src,
        ct_imag: &Src,
        dft: &DFTMatrix<Self, Decode, Split, LinearTransformation<P>>,
        keys: &H,
        scratch: &mut ScratchArena<'_, Self>,
    ) -> Result<()>
    where
        P: DiagonalProd<Self> + LtDiagonalMeta + IntPolyInfos,
        Dst: GLWEToBackendMut<Self> + GLWEToBackendRef<Self> + CKKSCtBounds + SetCKKSInfos,
        Src: GLWEToBackendRef<Self> + CKKSCtBounds,
        H: GetAutomorphismKey<Self>,
    {
        crate::oep::derived::dft::ckks_slots_to_coeffs_split_derived(module, op_out, ct_real, ct_imag, dft, keys, scratch)
    }

    /// Default body: `ckks_coeffs_to_slots_repack_derived`, `ckks_copy` of `ct_in` into `ct_out`,
    /// `ckks_dft_evaluate_assign` on it, `ckks_conjugate_into` of the result into an allocated temporary,
    /// `ckks_sub_into` of the two into a second allocated temporary, `ckks_div_i_assign` on that one,
    /// `ckks_add_assign` of the conjugate into `ct_out`, `ckks_rotate_assign` of the second temporary by the slot
    /// count, `ckks_add_assign` of it into `ct_out`, then the input's `log_sparsity` less one (saturating) and real
    /// slots on `ct_out`.
    fn ckks_coeffs_to_slots_repack<P, Dst, Src, H>(
        module: &Module<Self>,
        ct_out: &mut Dst,
        ct_in: &Src,
        dft: &DFTMatrix<Self, Encode, Repack, LinearTransformation<P>>,
        keys: &H,
        scratch: &mut ScratchArena<'_, Self>,
    ) -> Result<()>
    where
        P: DiagonalProd<Self> + LtDiagonalMeta + IntPolyInfos,
        Dst: GLWEToBackendMut<Self> + GLWEToBackendRef<Self> + CKKSCtBounds + SetCKKSInfos,
        Src: GLWEToBackendRef<Self> + CKKSCtBounds,
        H: GetAutomorphismKey<Self>,
    {
        crate::oep::derived::dft::ckks_coeffs_to_slots_repack_derived(module, ct_out, ct_in, dft, keys, scratch)
    }

    /// Default body: `ckks_slots_to_coeffs_repack_derived`, `ckks_copy` of `ct_in` into `op_out`,
    /// `ckks_dft_evaluate_assign` on it, then the input's `log_sparsity` plus one on `op_out`.
    fn ckks_slots_to_coeffs_repack<P, Dst, Src, H>(
        module: &Module<Self>,
        op_out: &mut Dst,
        ct_in: &Src,
        dft: &DFTMatrix<Self, Decode, Repack, LinearTransformation<P>>,
        keys: &H,
        scratch: &mut ScratchArena<'_, Self>,
    ) -> Result<()>
    where
        P: DiagonalProd<Self> + LtDiagonalMeta + IntPolyInfos,
        Dst: GLWEToBackendMut<Self> + GLWEToBackendRef<Self> + CKKSCtBounds + SetCKKSInfos,
        Src: GLWEToBackendRef<Self> + CKKSCtBounds,
        H: GetAutomorphismKey<Self>,
    {
        crate::oep::derived::dft::ckks_slots_to_coeffs_repack_derived(module, op_out, ct_in, dft, keys, scratch)
    }
}

/// Backend hook for homomorphic-DFT matrix generation at scalar precision `F`.
///
/// Separate from [`DFTImpl`] so scalar precision belongs to the contract.
/// Reference matrix generation carries its own scalar and encoding bounds;
/// custom implementations only need the method's backend-resident signature.
///
/// # Safety
/// Implementations reproduce the bodies their methods document bit for bit, report for each
/// `_tmp_bytes` query a bound that covers the body, and refuse as the reference does. The memory,
/// layout, synchrony and aliasing rules of `poulpy-hal/docs/backend_safety_contract.md` apply.
pub unsafe trait DFTMatrixImpl<F: CKKSEncodingScalar>: Backend<Ring = poulpy_hal::layouts::Standard> {
    /// Scratch bound of [`Self::ckks_new_dft_matrix`]; may report less than the reference, never less than the body takes.
    /// The reference reports the largest slot encoding of one factor's diagonals.
    fn ckks_new_dft_matrix_tmp_bytes<Dir, Fmt>(module: &Module<Self>, literal: &DFTPlan) -> Result<usize>
    where
        Dir: DftDirection,
        Fmt: DftFormat;

    /// Reproduces [`ckks_new_dft_matrix`](crate::reference::dft::ckks_new_dft_matrix) bit for bit; free in kernels.
    fn ckks_new_dft_matrix<Dir, Fmt>(
        module: &Module<Self>,
        base2k: Base2K,
        literal: &DFTPlan,
        scratch: &mut ScratchArena<'_, Self>,
    ) -> Result<DFTMatrix<Self, Dir, Fmt>>
    where
        Dir: DftDirection,
        Fmt: DftFormat;
}

/// Selects the reference DFT-matrix preparation and evaluation.
#[macro_export]
macro_rules! impl_dft_reference_full {
    ($be:ty) => {
        unsafe impl $crate::oep::DFTImpl for $be {
            fn ckks_prepare_dft_matrix_tmp_bytes<Dir, Fmt, P>(
                module: &::poulpy_hal::layouts::Module<Self>,
                dft: &$crate::layouts::DFTMatrix<Self, Dir, Fmt, ::poulpy_core::layouts::LinearTransformation<P>>,
            ) -> usize
            where
                P: poulpy_core::layouts::LWEInfos,
            {
                $crate::reference::dft::ckks_prepare_dft_matrix_tmp_bytes(module, dft)
            }

            fn ckks_dft_tmp_bytes<Dir, Fmt, P, Dst, Src, K>(
                module: &::poulpy_hal::layouts::Module<Self>,
                dst: &Dst,
                src: &Src,
                dft: &$crate::layouts::DFTMatrix<Self, Dir, Fmt, ::poulpy_core::layouts::LinearTransformation<P>>,
                key: &K,
            ) -> usize
            where
                P: poulpy_core::layouts::GLWEInfos,
                Dst: $crate::CKKSCtBounds,
                Src: $crate::CKKSCtBounds,
                K: poulpy_core::layouts::GGLWEInfos,
            {
                $crate::reference::dft::ckks_dft_tmp_bytes(module, dst, src, dft, key)
            }
            fn ckks_prepare_dft_matrix<Dir, Fmt, P>(
                module: &::poulpy_hal::layouts::Module<Self>,
                res: &mut $crate::layouts::DFTMatrixPrepared<Self, Dir, Fmt>,
                dft: &$crate::layouts::DFTMatrix<Self, Dir, Fmt, ::poulpy_core::layouts::LinearTransformation<P>>,
                scratch: &mut ::poulpy_hal::layouts::ScratchArena<'_, Self>,
            ) -> $crate::CKKSResult<()>
            where
                P: ::poulpy_core::layouts::GLWEToBackendRef<Self>
                    + ::poulpy_core::layouts::IntPolyInfos
                    + $crate::CKKSCtBounds
                    + ::poulpy_core::DiagonalProd<Self>,
            {
                $crate::reference::dft::ckks_prepare_dft_matrix::<Dir, Fmt, Self, P>(module, res, dft, scratch)
            }
            fn ckks_dft_evaluate_assign<Dir, Fmt, P, Dst, H>(
                module: &::poulpy_hal::layouts::Module<Self>,
                ct: &mut Dst,
                dft: &$crate::layouts::DFTMatrix<Self, Dir, Fmt, ::poulpy_core::layouts::LinearTransformation<P>>,
                keys: &H,
                scratch: &mut ::poulpy_hal::layouts::ScratchArena<'_, Self>,
            ) -> $crate::CKKSResult<()>
            where
                P: ::poulpy_core::DiagonalProd<Self> + $crate::api::LtDiagonalMeta + ::poulpy_core::layouts::IntPolyInfos,
                Dst: ::poulpy_core::layouts::GLWEToBackendMut<Self>
                    + ::poulpy_core::layouts::GLWEToBackendRef<Self>
                    + $crate::CKKSCtBounds
                    + $crate::SetCKKSInfos,
                H: ::poulpy_core::layouts::GetAutomorphismKey<Self>,
            {
                $crate::reference::dft::ckks_dft_evaluate_assign(module, ct, dft, keys, scratch)
            }
        }
    };
}

/// Selects the reference DFT-matrix generation for every encoding scalar of the backend.
#[macro_export]
macro_rules! impl_dft_matrix_reference_full {
    ($be:ty) => {
        unsafe impl<F: $crate::api::CKKSEncodingScalar + $crate::layouts::DftScalar> $crate::oep::DFTMatrixImpl<F> for $be
        where
            $be: $crate::oep::CKKSEncodingImpl<F>,
        {
            fn ckks_new_dft_matrix_tmp_bytes<Dir, Fmt>(
                module: &::poulpy_hal::layouts::Module<Self>,
                literal: &$crate::layouts::DFTPlan,
            ) -> $crate::CKKSResult<usize>
            where
                Dir: $crate::layouts::DftDirection,
                Fmt: $crate::layouts::DftFormat,
            {
                $crate::reference::dft::ckks_new_dft_matrix_tmp_bytes::<Dir, Fmt, Self, F>(module, literal)
            }

            fn ckks_new_dft_matrix<Dir, Fmt>(
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
pub use crate::impl_dft_matrix_reference_full;
pub use crate::impl_dft_reference_full;
