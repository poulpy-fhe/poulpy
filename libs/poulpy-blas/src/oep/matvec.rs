use poulpy_ckks::CKKSResult as Result;
use poulpy_core::layouts::{GGLWEInfos, GetAutomorphismKey, GetTensorKey};
use poulpy_hal::layouts::{Backend, Module, Ring, ScratchArena, Standard};

use poulpy_ckks::layouts::CKKSRightOperandPrepared;
use poulpy_core::layouts::GLWEInfos;

use crate::layouts::{
    BlasDiagonalCiphertext, BlasDiagonalCiphertextInfos, BlasDiagonalCiphertextMap, BlasGrid, BlasGridInfos, BlasGridLine,
    BlasGridMut, BlasMatVecMaps, BlasVectorInfos, BlasVectorMut, BlasVectorRef,
};

/// Extension point of [`crate::api::BlasMatVecOps`]: the products of a vector, all required.
///
/// # Safety
/// Every method reproduces its `_reference` counterpart in [`BlasMatVecReference`](crate::reference::BlasMatVecReference)
/// bit for bit, digits and CKKS metadata, within the scratch the matching `_tmp_bytes` reports; with less, it panics
/// before any write to `res` when it looks up a key, as the reference does. Anything the doc of a method does not list
/// as free changes bits.
///
/// A method first runs its validator, named on its doc, before any write, and returns its errors with `res` and the
/// operands untouched. On inputs it accepted, it returns `Ok`, or poulpy-ckks's budget error before any write to `res`.
/// Any other `Err` is a backend fault (a validator gap or a device failure), and `res` is then unspecified.
///
/// The memory, layout, synchrony and aliasing rules of `poulpy-hal/docs/backend_safety_contract.md` apply.
pub unsafe trait BlasMatVecImpl<Rg: Ring = Standard>: Backend {
    /// Scratch bound of [`Self::blas_matvec_pt`]; may report less than the reference, never less than the body takes.
    fn blas_matvec_pt_tmp_bytes<R, W, A, K>(module: &Module<Self>, res: &R, w: &W, a: &A, keys: &K) -> usize
    where
        R: BlasVectorInfos,
        W: BlasMatVecMaps<Self, Rg>,
        A: BlasVectorInfos,
        K: GGLWEInfos;

    /// Reproduces
    /// [`BlasMatVecReference::blas_matvec_pt_reference`](crate::reference::BlasMatVecReference::blas_matvec_pt_reference)
    /// bit for bit, within [`Self::blas_matvec_pt_tmp_bytes`]; validator: [`crate::layouts::validate_blas_matvec_pt`].
    ///
    /// **Fixed:**
    ///
    /// - output ciphertext `res_o` is poulpy-ckks's `ckks_linear_transformation_eval_sum_into` of its `K` terms
    ///   `(a_i, w_oi)` in ascending `i`, by the operand's resident or streamed diagonals. Each giant step sums the
    ///   products of every term in the transform domain, in that order, before its one rotation; an FFT backend rounds
    ///   that sum, so the order is part of the bits. Each ciphertext of `res` is then stamped with the label
    ///   `log2(S / output_period)`, for `S` the operand's slots;
    /// - the key widths: every map's baby steps at `a.k`, its giant steps at the width it leaves into `res`;
    /// - the errors and their order: the validator's, then poulpy-ckks's budget errors, before any write to `res`.
    ///
    /// **Free:** kernels, scratch use, the order of the output ciphertexts, and the concurrency between them.
    fn blas_matvec_pt<R, W, A, K>(
        module: &Module<Self>,
        res: &mut R,
        w: &W,
        a: &A,
        keys: &K,
        scratch: &mut ScratchArena<'_, Self>,
    ) -> Result<()>
    where
        R: BlasVectorMut<Self, Rg>,
        W: BlasMatVecMaps<Self, Rg>,
        A: BlasVectorRef<Self, Rg>,
        K: GetAutomorphismKey<Self, Rg> + GetTensorKey<Self, Rg> + Sync;

    /// Scratch bound of [`Self::blas_grid_to_vector`]; may report less than the reference, never less than the body
    /// takes.
    fn blas_grid_to_vector_tmp_bytes<R, W, A, K>(module: &Module<Self>, res: &R, w: &W, a: &A, keys: &K) -> usize
    where
        R: BlasVectorInfos,
        W: BlasMatVecMaps<Self, Rg>,
        A: BlasGridInfos,
        K: GGLWEInfos;

    /// Reproduces
    /// [`BlasMatVecReference::blas_grid_to_vector_reference`](crate::reference::BlasMatVecReference::blas_grid_to_vector_reference)
    /// bit for bit, within [`Self::blas_grid_to_vector_tmp_bytes`]; validator:
    /// [`crate::layouts::validate_blas_grid_to_vector`].
    ///
    /// **Fixed:** as [`Self::blas_matvec_pt`], the input ciphertexts being the blocks holding `line`, in entry order.
    ///
    /// **Free:** as [`Self::blas_matvec_pt`].
    fn blas_grid_to_vector<R, W, A, K>(
        module: &Module<Self>,
        res: &mut R,
        w: &W,
        a: &A,
        line: BlasGridLine,
        keys: &K,
        scratch: &mut ScratchArena<'_, Self>,
    ) -> Result<()>
    where
        R: BlasVectorMut<Self, Rg>,
        W: BlasMatVecMaps<Self, Rg>,
        A: BlasGrid<Self, Rg>,
        K: GetAutomorphismKey<Self, Rg> + GetTensorKey<Self, Rg> + Sync;

    /// Scratch bound of [`Self::blas_vector_to_grid`]; may report less than the reference, never less than the body
    /// takes.
    fn blas_vector_to_grid_tmp_bytes<R, W, A, K>(module: &Module<Self>, res: &R, w: &W, a: &A, keys: &K) -> usize
    where
        R: BlasGridInfos,
        W: BlasMatVecMaps<Self, Rg>,
        A: BlasVectorInfos,
        K: GGLWEInfos;

    /// Reproduces
    /// [`BlasMatVecReference::blas_vector_to_grid_reference`](crate::reference::BlasMatVecReference::blas_vector_to_grid_reference)
    /// bit for bit, within [`Self::blas_vector_to_grid_tmp_bytes`]; validator:
    /// [`crate::layouts::validate_blas_vector_to_grid`].
    ///
    /// **Fixed:** as [`Self::blas_matvec_pt`], the output ciphertexts being the blocks of `res`, in entry order, then
    /// the batch `output_period / d^2` recorded on `res` after `Ok`.
    ///
    /// **Free:** as [`Self::blas_matvec_pt`].
    fn blas_vector_to_grid<R, W, A, K>(
        module: &Module<Self>,
        res: &mut R,
        line: BlasGridLine,
        w: &W,
        a: &A,
        keys: &K,
        scratch: &mut ScratchArena<'_, Self>,
    ) -> Result<()>
    where
        R: BlasGridMut<Self, Rg>,
        W: BlasMatVecMaps<Self, Rg>,
        A: BlasVectorRef<Self, Rg>,
        K: GetAutomorphismKey<Self, Rg> + GetTensorKey<Self, Rg> + Sync;

    /// Scratch bound of [`Self::blas_matvec_diagonal`]; may report less than the reference, never less than the body takes.
    fn blas_matvec_diagonal_tmp_bytes<R, M, A, T, K>(module: &Module<Self>, res: &R, m: &M, a: &A, tsk: &T, keys: &K) -> usize
    where
        R: BlasVectorInfos,
        M: BlasDiagonalCiphertextMap<Self, Rg>,
        A: BlasVectorInfos,
        T: GGLWEInfos,
        K: GGLWEInfos;

    /// Reproduces
    /// [`BlasMatVecReference::blas_matvec_diagonal_reference`](crate::reference::BlasMatVecReference::blas_matvec_diagonal_reference) bit
    /// for bit, within [`Self::blas_matvec_diagonal_tmp_bytes`]; validator: [`crate::layouts::validate_blas_matvec_diagonal`].
    ///
    /// **Fixed:** poulpy-ckks's `ckks_linear_transformation_eval_into` of `m`'s ciphertext diagonals from `a`, resident or
    /// prepared at each use as `m`'s form says,
    /// its baby steps prepared as `m`'s schedule reaches them (one key per rotation, or doubling passes), its key widths,
    /// and the errors and their order: the validator's, then poulpy-ckks's, before any write to `res`.
    ///
    /// **Free:** kernels and scratch use.
    fn blas_matvec_diagonal<R, M, A, K>(
        module: &Module<Self>,
        res: &mut R,
        m: &M,
        a: &A,
        keys: &K,
        scratch: &mut ScratchArena<'_, Self>,
    ) -> Result<()>
    where
        R: BlasVectorMut<Self, Rg>,
        M: BlasDiagonalCiphertextMap<Self, Rg>,
        A: BlasVectorRef<Self, Rg>,
        K: GetAutomorphismKey<Self, Rg> + GetTensorKey<Self, Rg> + Sync;

    /// Scratch bound of [`Self::blas_append_row_diagonal`]; may report less than the reference, never less than the body takes.
    fn blas_append_row_diagonal_tmp_bytes<M, V, P, K>(module: &Module<Self>, m: &M, v: &V, mask: &P, keys: &K) -> usize
    where
        M: BlasDiagonalCiphertextInfos,
        V: BlasVectorInfos,
        P: GLWEInfos,
        K: GGLWEInfos;

    /// Reproduces
    /// [`BlasMatVecReference::blas_append_row_diagonal_reference`](crate::reference::BlasMatVecReference::blas_append_row_diagonal_reference)
    /// bit for bit, within [`Self::blas_append_row_diagonal_tmp_bytes`]; validator: [`crate::layouts::validate_blas_append_row_diagonal`].
    /// It writes `m` through [`BlasDiagonalCiphertext::diagonals_mut`].
    ///
    /// **Fixed:** the masked accumulates of the reference, each poulpy-ckks's `ckks_dot_product_prepared_pt_vec_add` of the
    /// mask read through its shift's Galois element and a baby rotation of `v`; the key widths and the errors and their order: the validator's, then poulpy-ckks's, before any write to `m`.
    ///
    /// **Free:** kernels, scratch use, how the rotations of `v` are made and prepared, and the order of the entries.
    fn blas_append_row_diagonal<V, K>(
        module: &Module<Self>,
        m: &mut BlasDiagonalCiphertext<Self, Rg>,
        row: i64,
        v: &V,
        mask: &CKKSRightOperandPrepared<Self, Rg>,
        keys: &K,
        scratch: &mut ScratchArena<'_, Self>,
    ) -> Result<()>
    where
        V: BlasVectorRef<Self, Rg>,
        K: GetAutomorphismKey<Self, Rg> + Sync;

    /// Scratch bound of [`Self::blas_append_col_diagonal`]; may report less than the reference, never less than the body takes.
    fn blas_append_col_diagonal_tmp_bytes<M, V, P, K>(module: &Module<Self>, m: &M, v: &V, mask: &P, keys: &K) -> usize
    where
        M: BlasDiagonalCiphertextInfos,
        V: BlasVectorInfos,
        P: GLWEInfos,
        K: GGLWEInfos;

    /// As [`Self::blas_append_row_diagonal`], for
    /// [`BlasMatVecReference::blas_append_col_diagonal_reference`](crate::reference::BlasMatVecReference::blas_append_col_diagonal_reference)
    /// within [`Self::blas_append_col_diagonal_tmp_bytes`]; validator: [`crate::layouts::validate_blas_append_col_diagonal`]. Each entry
    /// reads a giant rotation of `v` and the mask through the Galois element of `b - col`.
    fn blas_append_col_diagonal<V, K>(
        module: &Module<Self>,
        m: &mut BlasDiagonalCiphertext<Self, Rg>,
        col: i64,
        v: &V,
        mask: &CKKSRightOperandPrepared<Self, Rg>,
        keys: &K,
        scratch: &mut ScratchArena<'_, Self>,
    ) -> Result<()>
    where
        V: BlasVectorRef<Self, Rg>,
        K: GetAutomorphismKey<Self, Rg> + Sync;
}

/// Selects [`BlasMatVecReference`](crate::reference::BlasMatVecReference) for `$be` on both rings, or on `$ring` alone
/// through `($be, $ring)`.
#[macro_export]
macro_rules! impl_blas_mat_vec_reference_full {
    ($be:ty) => {
        $crate::impl_blas_mat_vec_reference_full!($be, ::poulpy_hal::layouts::Standard);
        $crate::impl_blas_mat_vec_reference_full!($be, ::poulpy_hal::layouts::ConjugateInvariant);
    };
    ($be:ty, $ring:ty) => {
        unsafe impl $crate::oep::BlasMatVecImpl<$ring> for $be {
            fn blas_matvec_pt_tmp_bytes<R, W, A, K>(
                module: &::poulpy_hal::layouts::Module<Self>,
                res: &R,
                w: &W,
                a: &A,
                keys: &K,
            ) -> usize
            where
                R: $crate::layouts::BlasVectorInfos,
                W: $crate::layouts::BlasMatVecMaps<Self, $ring>,
                A: $crate::layouts::BlasVectorInfos,
                K: ::poulpy_core::layouts::GGLWEInfos,
            {
                $crate::reference::BlasMatVecReference::<Self, $ring>::blas_matvec_pt_tmp_bytes_reference(module, res, w, a, keys)
            }

            fn blas_matvec_pt<R, W, A, K>(
                module: &::poulpy_hal::layouts::Module<Self>,
                res: &mut R,
                w: &W,
                a: &A,
                keys: &K,
                scratch: &mut ::poulpy_hal::layouts::ScratchArena<'_, Self>,
            ) -> ::poulpy_ckks::CKKSResult<()>
            where
                R: $crate::layouts::BlasVectorMut<Self, $ring>,
                W: $crate::layouts::BlasMatVecMaps<Self, $ring>,
                A: $crate::layouts::BlasVectorRef<Self, $ring>,
                K: ::poulpy_core::layouts::GetAutomorphismKey<Self, $ring>
                    + ::poulpy_core::layouts::GetTensorKey<Self, $ring>
                    + Sync,
            {
                $crate::reference::BlasMatVecReference::<Self, $ring>::blas_matvec_pt_reference(module, res, w, a, keys, scratch)
            }

            fn blas_grid_to_vector_tmp_bytes<R, W, A, K>(
                module: &::poulpy_hal::layouts::Module<Self>,
                res: &R,
                w: &W,
                a: &A,
                keys: &K,
            ) -> usize
            where
                R: $crate::layouts::BlasVectorInfos,
                W: $crate::layouts::BlasMatVecMaps<Self, $ring>,
                A: $crate::layouts::BlasGridInfos,
                K: ::poulpy_core::layouts::GGLWEInfos,
            {
                $crate::reference::BlasMatVecReference::<Self, $ring>::blas_grid_to_vector_tmp_bytes_reference(
                    module, res, w, a, keys,
                )
            }

            fn blas_grid_to_vector<R, W, A, K>(
                module: &::poulpy_hal::layouts::Module<Self>,
                res: &mut R,
                w: &W,
                a: &A,
                line: $crate::layouts::BlasGridLine,
                keys: &K,
                scratch: &mut ::poulpy_hal::layouts::ScratchArena<'_, Self>,
            ) -> ::poulpy_ckks::CKKSResult<()>
            where
                R: $crate::layouts::BlasVectorMut<Self, $ring>,
                W: $crate::layouts::BlasMatVecMaps<Self, $ring>,
                A: $crate::layouts::BlasGrid<Self, $ring>,
                K: ::poulpy_core::layouts::GetAutomorphismKey<Self, $ring>
                    + ::poulpy_core::layouts::GetTensorKey<Self, $ring>
                    + Sync,
            {
                $crate::reference::BlasMatVecReference::<Self, $ring>::blas_grid_to_vector_reference(
                    module, res, w, a, line, keys, scratch,
                )
            }

            fn blas_vector_to_grid_tmp_bytes<R, W, A, K>(
                module: &::poulpy_hal::layouts::Module<Self>,
                res: &R,
                w: &W,
                a: &A,
                keys: &K,
            ) -> usize
            where
                R: $crate::layouts::BlasGridInfos,
                W: $crate::layouts::BlasMatVecMaps<Self, $ring>,
                A: $crate::layouts::BlasVectorInfos,
                K: ::poulpy_core::layouts::GGLWEInfos,
            {
                $crate::reference::BlasMatVecReference::<Self, $ring>::blas_vector_to_grid_tmp_bytes_reference(
                    module, res, w, a, keys,
                )
            }

            fn blas_vector_to_grid<R, W, A, K>(
                module: &::poulpy_hal::layouts::Module<Self>,
                res: &mut R,
                line: $crate::layouts::BlasGridLine,
                w: &W,
                a: &A,
                keys: &K,
                scratch: &mut ::poulpy_hal::layouts::ScratchArena<'_, Self>,
            ) -> ::poulpy_ckks::CKKSResult<()>
            where
                R: $crate::layouts::BlasGridMut<Self, $ring>,
                W: $crate::layouts::BlasMatVecMaps<Self, $ring>,
                A: $crate::layouts::BlasVectorRef<Self, $ring>,
                K: ::poulpy_core::layouts::GetAutomorphismKey<Self, $ring>
                    + ::poulpy_core::layouts::GetTensorKey<Self, $ring>
                    + Sync,
            {
                $crate::reference::BlasMatVecReference::<Self, $ring>::blas_vector_to_grid_reference(
                    module, res, line, w, a, keys, scratch,
                )
            }

            fn blas_matvec_diagonal_tmp_bytes<R, M, A, T, K>(
                module: &::poulpy_hal::layouts::Module<Self>,
                res: &R,
                m: &M,
                a: &A,
                tsk: &T,
                keys: &K,
            ) -> usize
            where
                R: $crate::layouts::BlasVectorInfos,
                M: $crate::layouts::BlasDiagonalCiphertextMap<Self, $ring>,
                A: $crate::layouts::BlasVectorInfos,
                T: ::poulpy_core::layouts::GGLWEInfos,
                K: ::poulpy_core::layouts::GGLWEInfos,
            {
                $crate::reference::BlasMatVecReference::<Self, $ring>::blas_matvec_diagonal_tmp_bytes_reference(
                    module, res, m, a, tsk, keys,
                )
            }

            fn blas_matvec_diagonal<R, M, A, K>(
                module: &::poulpy_hal::layouts::Module<Self>,
                res: &mut R,
                m: &M,
                a: &A,
                keys: &K,
                scratch: &mut ::poulpy_hal::layouts::ScratchArena<'_, Self>,
            ) -> ::poulpy_ckks::CKKSResult<()>
            where
                R: $crate::layouts::BlasVectorMut<Self, $ring>,
                M: $crate::layouts::BlasDiagonalCiphertextMap<Self, $ring>,
                A: $crate::layouts::BlasVectorRef<Self, $ring>,
                K: ::poulpy_core::layouts::GetAutomorphismKey<Self, $ring>
                    + ::poulpy_core::layouts::GetTensorKey<Self, $ring>
                    + Sync,
            {
                $crate::reference::BlasMatVecReference::<Self, $ring>::blas_matvec_diagonal_reference(
                    module, res, m, a, keys, scratch,
                )
            }

            fn blas_append_row_diagonal_tmp_bytes<M, V, P, K>(
                module: &::poulpy_hal::layouts::Module<Self>,
                m: &M,
                v: &V,
                mask: &P,
                keys: &K,
            ) -> usize
            where
                M: $crate::layouts::BlasDiagonalCiphertextInfos,
                V: $crate::layouts::BlasVectorInfos,
                P: ::poulpy_core::layouts::GLWEInfos,
                K: ::poulpy_core::layouts::GGLWEInfos,
            {
                $crate::reference::BlasMatVecReference::<Self, $ring>::blas_append_row_diagonal_tmp_bytes_reference(
                    module, m, v, mask, keys,
                )
            }

            fn blas_append_row_diagonal<V, K>(
                module: &::poulpy_hal::layouts::Module<Self>,
                m: &mut $crate::layouts::BlasDiagonalCiphertext<Self, $ring>,
                row: i64,
                v: &V,
                mask: &::poulpy_ckks::layouts::CKKSRightOperandPrepared<Self, $ring>,
                keys: &K,
                scratch: &mut ::poulpy_hal::layouts::ScratchArena<'_, Self>,
            ) -> ::poulpy_ckks::CKKSResult<()>
            where
                V: $crate::layouts::BlasVectorRef<Self, $ring>,
                K: ::poulpy_core::layouts::GetAutomorphismKey<Self, $ring> + Sync,
            {
                $crate::reference::BlasMatVecReference::<Self, $ring>::blas_append_row_diagonal_reference(
                    module, m, row, v, mask, keys, scratch,
                )
            }

            fn blas_append_col_diagonal_tmp_bytes<M, V, P, K>(
                module: &::poulpy_hal::layouts::Module<Self>,
                m: &M,
                v: &V,
                mask: &P,
                keys: &K,
            ) -> usize
            where
                M: $crate::layouts::BlasDiagonalCiphertextInfos,
                V: $crate::layouts::BlasVectorInfos,
                P: ::poulpy_core::layouts::GLWEInfos,
                K: ::poulpy_core::layouts::GGLWEInfos,
            {
                $crate::reference::BlasMatVecReference::<Self, $ring>::blas_append_col_diagonal_tmp_bytes_reference(
                    module, m, v, mask, keys,
                )
            }

            fn blas_append_col_diagonal<V, K>(
                module: &::poulpy_hal::layouts::Module<Self>,
                m: &mut $crate::layouts::BlasDiagonalCiphertext<Self, $ring>,
                col: i64,
                v: &V,
                mask: &::poulpy_ckks::layouts::CKKSRightOperandPrepared<Self, $ring>,
                keys: &K,
                scratch: &mut ::poulpy_hal::layouts::ScratchArena<'_, Self>,
            ) -> ::poulpy_ckks::CKKSResult<()>
            where
                V: $crate::layouts::BlasVectorRef<Self, $ring>,
                K: ::poulpy_core::layouts::GetAutomorphismKey<Self, $ring> + Sync,
            {
                $crate::reference::BlasMatVecReference::<Self, $ring>::blas_append_col_diagonal_reference(
                    module, m, col, v, mask, keys, scratch,
                )
            }
        }
    };
}
