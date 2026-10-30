use poulpy_ckks::{CKKSResult, api::CKKSEncodingScalar, oep::CKKSEncodingImpl};
use poulpy_core::layouts::DiagonalArithmetic;
use poulpy_hal::layouts::{Backend, Module, ScratchArena};

use crate::layouts::{
    BlasContext, BlasContextInfos, BlasContextPrepared, BlasMatVecOperand, BlasMatVecOperandInfos, BlasMatVecOperandPrepared,
    BlasPtOperand, BlasPtOperandInfos, BlasPtOperandPrepared, BlasPtSide,
};

/// Extension point of [`crate::api::BlasEncodingOps`], [`crate::api::BlasPtOperandPreparedFactory`],
/// [`crate::api::BlasMatVecOperandPreparedFactory`] and [`crate::api::BlasContextPreparedFactory`]: the encodings of
/// public operands and contexts and their preparations, all required. The encoding scalar `F` is a method parameter bounded by `Self: CKKSEncodingImpl<F>`.
///
/// # Safety
/// Every method reproduces its `_reference` counterpart in
/// [`BlasEncodingReference`](crate::reference::BlasEncodingReference) bit for bit, every diagonal and its metadata,
/// within the scratch the matching `_tmp_bytes` reports, and refuses what the reference refuses. Every
/// implementation overwrites every allocated byte of `res` without reallocating it, so a refill equals a fresh build.
/// The memory, layout, synchrony and aliasing rules of `poulpy-hal/docs/backend_safety_contract.md` apply.
pub unsafe trait BlasEncodingImpl: Backend {
    /// Scratch bound of [`Self::blas_pt_operand_encode`]; may report less than
    /// [`BlasEncodingReference::blas_pt_operand_encode_tmp_bytes_reference`](crate::reference::BlasEncodingReference::blas_pt_operand_encode_tmp_bytes_reference),
    /// never less than the body takes.
    fn blas_pt_operand_encode_tmp_bytes<F, A>(module: &Module<Self>, infos: &A) -> usize
    where
        Self: CKKSEncodingImpl<F>,
        F: CKKSEncodingScalar + DiagonalArithmetic,
        A: BlasPtOperandInfos;

    /// Reproduces
    /// [`BlasEncodingReference::blas_pt_operand_encode_reference`](crate::reference::BlasEncodingReference::blas_pt_operand_encode_reference)
    /// bit for bit, within [`Self::blas_pt_operand_encode_tmp_bytes`].
    ///
    /// **Fixed:** each map's host diagonals, built from `entry` by [`crate::layouts::maps`], its schedule, the plan's,
    /// the blocks sharing the first block's, and its encoding by poulpy-ckks's `ckks_encode_linear_transformation_into`
    /// at the operand's `coeffs_meta`, map after map in the order of
    /// [`BlasPtOperand::maps`](crate::layouts::BlasPtOperand::maps), the first error stopping it.
    /// **Free:** its kernels and its scratch use.
    ///
    /// It panics on a `res` allocated for another module's slots or ring, then on a scratch short of its query, before
    /// any write, and returns poulpy-ckks's errors, after which `res` is unspecified.
    fn blas_pt_operand_encode<F, S>(
        module: &Module<Self>,
        res: &mut BlasPtOperand<Self, S>,
        entry: impl Fn(usize, usize, usize) -> F,
        scratch: &mut ScratchArena<'_, Self>,
    ) -> CKKSResult<()>
    where
        Self: CKKSEncodingImpl<F>,
        F: CKKSEncodingScalar + DiagonalArithmetic,
        S: BlasPtSide;

    /// Scratch bound of [`Self::blas_matvec_operand_encode`]; may report less than
    /// [`BlasEncodingReference::blas_matvec_operand_encode_tmp_bytes_reference`](crate::reference::BlasEncodingReference::blas_matvec_operand_encode_tmp_bytes_reference),
    /// never less than the body takes.
    fn blas_matvec_operand_encode_tmp_bytes<F, A>(module: &Module<Self>, infos: &A) -> usize
    where
        Self: CKKSEncodingImpl<F>,
        F: CKKSEncodingScalar + DiagonalArithmetic,
        A: BlasMatVecOperandInfos;

    /// Reproduces
    /// [`BlasEncodingReference::blas_matvec_operand_encode_reference`](crate::reference::BlasEncodingReference::blas_matvec_operand_encode_reference)
    /// bit for bit, within [`Self::blas_matvec_operand_encode_tmp_bytes`].
    ///
    /// **Fixed:** each block's host diagonals, from `entry` by the slot map
    /// [`BlasEncodingOps::blas_matvec_operand_encode`](crate::api::BlasEncodingOps::blas_matvec_operand_encode) states,
    /// its schedule, and its encoding by poulpy-ckks's `ckks_encode_linear_transformation_into` at the operand's
    /// `coeffs_meta`, block after block, output-major, the first error stopping it. **Free:** its kernels and its
    /// scratch use.
    ///
    /// It panics on a `res` allocated for another module's slots, then on a scratch short of its query, before any
    /// write, and returns poulpy-ckks's errors, after which `res` is unspecified.
    fn blas_matvec_operand_encode<F>(
        module: &Module<Self>,
        res: &mut BlasMatVecOperand<Self>,
        entry: impl Fn(usize, usize) -> F,
        scratch: &mut ScratchArena<'_, Self>,
    ) -> CKKSResult<()>
    where
        Self: CKKSEncodingImpl<F>,
        F: CKKSEncodingScalar + DiagonalArithmetic;

    /// Scratch bound of [`Self::blas_context_encode`]; may report less than
    /// [`BlasEncodingReference::blas_context_encode_tmp_bytes_reference`](crate::reference::BlasEncodingReference::blas_context_encode_tmp_bytes_reference),
    /// never less than the body takes.
    fn blas_context_encode_tmp_bytes<F, A>(module: &Module<Self>, infos: &A) -> usize
    where
        Self: CKKSEncodingImpl<F>,
        F: CKKSEncodingScalar + DiagonalArithmetic,
        A: BlasContextInfos;

    /// Reproduces
    /// [`BlasEncodingReference::blas_context_encode_reference`](crate::reference::BlasEncodingReference::blas_context_encode_reference)
    /// bit for bit, within [`Self::blas_context_encode_tmp_bytes`].
    ///
    /// **Fixed:** each map's host diagonals ([`crate::layouts::maps`]) and schedule, as the plan gives them, and its
    /// encoding by poulpy-ckks's `ckks_encode_linear_transformation_into` at the context's `coeffs_meta`, map after map
    /// in the order of [`BlasContext::maps`], the first error stopping it. **Free:** its kernels and its scratch
    /// use.
    ///
    /// It panics on a `res` allocated for another module's slots or ring, then on a scratch short of its query, before
    /// any write, and returns poulpy-ckks's errors, after which `res` is unspecified.
    fn blas_context_encode<F>(
        module: &Module<Self>,
        res: &mut BlasContext<Self>,
        scratch: &mut ScratchArena<'_, Self>,
    ) -> CKKSResult<()>
    where
        Self: CKKSEncodingImpl<F>,
        F: CKKSEncodingScalar + DiagonalArithmetic;

    /// Scratch bound of [`Self::blas_pt_operand_prepare`]; may report less than
    /// [`BlasEncodingReference::blas_pt_operand_prepare_tmp_bytes_reference`](crate::reference::BlasEncodingReference::blas_pt_operand_prepare_tmp_bytes_reference),
    /// never less than the body takes.
    fn blas_pt_operand_prepare_tmp_bytes<A: BlasPtOperandInfos>(module: &Module<Self>, infos: &A) -> usize;

    /// Reproduces
    /// [`BlasEncodingReference::blas_pt_operand_prepare_reference`](crate::reference::BlasEncodingReference::blas_pt_operand_prepare_reference)
    /// bit for bit, within [`Self::blas_pt_operand_prepare_tmp_bytes`].
    ///
    /// **Fixed:** poulpy-ckks's `ckks_prepare_linear_transformation_rhs` of every map, and under `JklsStored` and
    /// `JklsGalois` its `ckks_prepare_right_into` of each term's one plaintext. **Free:** the prepared form, kernels,
    /// scratch use and the order of the maps.
    ///
    /// It panics before any write on `res` and `other` of different layouts, on a map whose diagonals do not share one
    /// scale, width and slot kind (`"<op>: a map's diagonals do not share one scale, width and slot kind"`) and on a
    /// scratch short of its query, in that order, and with poulpy-ckks's refusal on any other refusal of a map's
    /// preparation.
    fn blas_pt_operand_prepare<S: BlasPtSide>(
        module: &Module<Self>,
        res: &mut BlasPtOperandPrepared<Self, S>,
        other: &BlasPtOperand<Self, S>,
        scratch: &mut ScratchArena<'_, Self>,
    );

    /// Scratch bound of [`Self::blas_matvec_operand_prepare`]; may report less than
    /// [`BlasEncodingReference::blas_matvec_operand_prepare_tmp_bytes_reference`](crate::reference::BlasEncodingReference::blas_matvec_operand_prepare_tmp_bytes_reference),
    /// never less than the body takes.
    fn blas_matvec_operand_prepare_tmp_bytes<A: BlasMatVecOperandInfos>(module: &Module<Self>, infos: &A) -> usize;

    /// Reproduces
    /// [`BlasEncodingReference::blas_matvec_operand_prepare_reference`](crate::reference::BlasEncodingReference::blas_matvec_operand_prepare_reference)
    /// bit for bit, within [`Self::blas_matvec_operand_prepare_tmp_bytes`].
    ///
    /// **Fixed:** poulpy-ckks's `ckks_prepare_linear_transformation_rhs` of every map. **Free:** the prepared form,
    /// kernels, scratch use and the order of the maps.
    ///
    /// It panics as [`Self::blas_context_prepare`] does.
    fn blas_matvec_operand_prepare(
        module: &Module<Self>,
        res: &mut BlasMatVecOperandPrepared<Self>,
        other: &BlasMatVecOperand<Self>,
        scratch: &mut ScratchArena<'_, Self>,
    );

    /// Scratch bound of [`Self::blas_context_prepare`]; may report less than
    /// [`BlasEncodingReference::blas_context_prepare_tmp_bytes_reference`](crate::reference::BlasEncodingReference::blas_context_prepare_tmp_bytes_reference),
    /// never less than the body takes.
    fn blas_context_prepare_tmp_bytes<A: BlasContextInfos>(module: &Module<Self>, infos: &A) -> usize;

    /// Reproduces
    /// [`BlasEncodingReference::blas_context_prepare_reference`](crate::reference::BlasEncodingReference::blas_context_prepare_reference)
    /// bit for bit, within [`Self::blas_context_prepare_tmp_bytes`].
    ///
    /// **Fixed:** poulpy-ckks's `ckks_prepare_linear_transformation_rhs` of every map. **Free:** the prepared form,
    /// kernels, scratch use and the order of the maps.
    ///
    /// It panics before any write on `res` and `other` of different layouts, on a map whose diagonals do not share one
    /// scale, width and slot kind (`"<op>: a map's diagonals do not share one scale, width and slot kind"`) and on a
    /// scratch short of its query, in that order, and with poulpy-ckks's refusal on any other refusal of a map's
    /// preparation.
    fn blas_context_prepare(
        module: &Module<Self>,
        res: &mut BlasContextPrepared<Self>,
        other: &BlasContext<Self>,
        scratch: &mut ScratchArena<'_, Self>,
    );
}

/// Selects [`BlasEncodingReference`](crate::reference::BlasEncodingReference) for `$be`.
#[macro_export]
macro_rules! impl_blas_encoding_reference_full {
    ($be:ty) => {
        unsafe impl $crate::oep::BlasEncodingImpl for $be {
            fn blas_pt_operand_encode_tmp_bytes<F, A>(module: &::poulpy_hal::layouts::Module<Self>, infos: &A) -> usize
            where
                Self: ::poulpy_ckks::oep::CKKSEncodingImpl<F>,
                F: ::poulpy_ckks::api::CKKSEncodingScalar + ::poulpy_core::layouts::DiagonalArithmetic,
                A: $crate::layouts::BlasPtOperandInfos,
            {
                $crate::reference::BlasEncodingReference::blas_pt_operand_encode_tmp_bytes_reference::<F, A>(module, infos)
            }

            fn blas_pt_operand_encode<F, S>(
                module: &::poulpy_hal::layouts::Module<Self>,
                res: &mut $crate::layouts::BlasPtOperand<Self, S>,
                entry: impl Fn(usize, usize, usize) -> F,
                scratch: &mut ::poulpy_hal::layouts::ScratchArena<'_, Self>,
            ) -> ::poulpy_ckks::CKKSResult<()>
            where
                Self: ::poulpy_ckks::oep::CKKSEncodingImpl<F>,
                F: ::poulpy_ckks::api::CKKSEncodingScalar + ::poulpy_core::layouts::DiagonalArithmetic,
                S: $crate::layouts::BlasPtSide,
            {
                $crate::reference::BlasEncodingReference::blas_pt_operand_encode_reference(module, res, entry, scratch)
            }

            fn blas_matvec_operand_encode_tmp_bytes<F, A>(module: &::poulpy_hal::layouts::Module<Self>, infos: &A) -> usize
            where
                Self: ::poulpy_ckks::oep::CKKSEncodingImpl<F>,
                F: ::poulpy_ckks::api::CKKSEncodingScalar + ::poulpy_core::layouts::DiagonalArithmetic,
                A: $crate::layouts::BlasMatVecOperandInfos,
            {
                $crate::reference::BlasEncodingReference::blas_matvec_operand_encode_tmp_bytes_reference::<F, A>(module, infos)
            }

            fn blas_matvec_operand_encode<F>(
                module: &::poulpy_hal::layouts::Module<Self>,
                res: &mut $crate::layouts::BlasMatVecOperand<Self>,
                entry: impl Fn(usize, usize) -> F,
                scratch: &mut ::poulpy_hal::layouts::ScratchArena<'_, Self>,
            ) -> ::poulpy_ckks::CKKSResult<()>
            where
                Self: ::poulpy_ckks::oep::CKKSEncodingImpl<F>,
                F: ::poulpy_ckks::api::CKKSEncodingScalar + ::poulpy_core::layouts::DiagonalArithmetic,
            {
                $crate::reference::BlasEncodingReference::blas_matvec_operand_encode_reference(module, res, entry, scratch)
            }

            fn blas_context_encode_tmp_bytes<F, A>(module: &::poulpy_hal::layouts::Module<Self>, infos: &A) -> usize
            where
                Self: ::poulpy_ckks::oep::CKKSEncodingImpl<F>,
                F: ::poulpy_ckks::api::CKKSEncodingScalar + ::poulpy_core::layouts::DiagonalArithmetic,
                A: $crate::layouts::BlasContextInfos,
            {
                $crate::reference::BlasEncodingReference::blas_context_encode_tmp_bytes_reference::<F, A>(module, infos)
            }

            fn blas_context_encode<F>(
                module: &::poulpy_hal::layouts::Module<Self>,
                res: &mut $crate::layouts::BlasContext<Self>,
                scratch: &mut ::poulpy_hal::layouts::ScratchArena<'_, Self>,
            ) -> ::poulpy_ckks::CKKSResult<()>
            where
                Self: ::poulpy_ckks::oep::CKKSEncodingImpl<F>,
                F: ::poulpy_ckks::api::CKKSEncodingScalar + ::poulpy_core::layouts::DiagonalArithmetic,
            {
                $crate::reference::BlasEncodingReference::blas_context_encode_reference::<F>(module, res, scratch)
            }

            fn blas_pt_operand_prepare_tmp_bytes<A: $crate::layouts::BlasPtOperandInfos>(
                module: &::poulpy_hal::layouts::Module<Self>,
                infos: &A,
            ) -> usize {
                $crate::reference::BlasEncodingReference::blas_pt_operand_prepare_tmp_bytes_reference(module, infos)
            }

            fn blas_pt_operand_prepare<S: $crate::layouts::BlasPtSide>(
                module: &::poulpy_hal::layouts::Module<Self>,
                res: &mut $crate::layouts::BlasPtOperandPrepared<Self, S>,
                other: &$crate::layouts::BlasPtOperand<Self, S>,
                scratch: &mut ::poulpy_hal::layouts::ScratchArena<'_, Self>,
            ) {
                $crate::reference::BlasEncodingReference::blas_pt_operand_prepare_reference(module, res, other, scratch)
            }

            fn blas_matvec_operand_prepare_tmp_bytes<A: $crate::layouts::BlasMatVecOperandInfos>(
                module: &::poulpy_hal::layouts::Module<Self>,
                infos: &A,
            ) -> usize {
                $crate::reference::BlasEncodingReference::blas_matvec_operand_prepare_tmp_bytes_reference(module, infos)
            }

            fn blas_matvec_operand_prepare(
                module: &::poulpy_hal::layouts::Module<Self>,
                res: &mut $crate::layouts::BlasMatVecOperandPrepared<Self>,
                other: &$crate::layouts::BlasMatVecOperand<Self>,
                scratch: &mut ::poulpy_hal::layouts::ScratchArena<'_, Self>,
            ) {
                $crate::reference::BlasEncodingReference::blas_matvec_operand_prepare_reference(module, res, other, scratch)
            }

            fn blas_context_prepare_tmp_bytes<A: $crate::layouts::BlasContextInfos>(
                module: &::poulpy_hal::layouts::Module<Self>,
                infos: &A,
            ) -> usize {
                $crate::reference::BlasEncodingReference::blas_context_prepare_tmp_bytes_reference(module, infos)
            }

            fn blas_context_prepare(
                module: &::poulpy_hal::layouts::Module<Self>,
                res: &mut $crate::layouts::BlasContextPrepared<Self>,
                other: &$crate::layouts::BlasContext<Self>,
                scratch: &mut ::poulpy_hal::layouts::ScratchArena<'_, Self>,
            ) {
                $crate::reference::BlasEncodingReference::blas_context_prepare_reference(module, res, other, scratch)
            }
        }
    };
}
