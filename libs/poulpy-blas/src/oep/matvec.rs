use poulpy_ckks::CKKSResult as Result;
use poulpy_core::layouts::{GGLWEInfos, GetAutomorphismKey, GetTensorKey};
use poulpy_hal::layouts::{Backend, Module, ScratchArena};

use poulpy_ckks::layouts::CKKSPreparedRight;
use poulpy_core::layouts::GLWEInfos;

use crate::layouts::{
    BlasDiagonalCiphertext, BlasDiagonalCiphertextInfos, BlasDiagonalCiphertextMap, BlasMatVecMaps, BlasVectorInfos,
    BlasVectorMut, BlasVectorRef,
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
pub unsafe trait BlasMatVecImpl: Backend {
    /// Scratch bound of [`Self::blas_matvec_pt`]; may report less than the reference, never less than the body takes.
    fn blas_matvec_pt_tmp_bytes<R, W, A, K>(module: &Module<Self>, res: &R, w: &W, a: &A, keys: &K) -> usize
    where
        R: BlasVectorInfos,
        W: BlasMatVecMaps<Self>,
        A: BlasVectorInfos,
        K: GGLWEInfos;

    /// Reproduces
    /// [`BlasMatVecReference::blas_matvec_pt_reference`](crate::reference::BlasMatVecReference::blas_matvec_pt_reference)
    /// bit for bit, within [`Self::blas_matvec_pt_tmp_bytes`]; validator: [`crate::layouts::validate_blas_matvec_pt`].
    ///
    /// **Fixed:**
    ///
    /// - output ciphertext `res_o` is poulpy-ckks's `ckks_eval_linear_transformation_sum_into` of its `K` terms
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
        R: BlasVectorMut<Self>,
        W: BlasMatVecMaps<Self>,
        A: BlasVectorRef<Self>,
        K: GetAutomorphismKey<Self> + GetTensorKey<Self> + Sync;

    /// Scratch bound of [`Self::blas_matvec_diagonal`]; may report less than the reference, never less than the body takes.
    fn blas_matvec_diagonal_tmp_bytes<R, M, A, T, K>(module: &Module<Self>, res: &R, m: &M, a: &A, tsk: &T, keys: &K) -> usize
    where
        R: BlasVectorInfos,
        M: BlasDiagonalCiphertextMap<Self>,
        A: BlasVectorInfos,
        T: GGLWEInfos,
        K: GGLWEInfos;

    /// Reproduces
    /// [`BlasMatVecReference::blas_matvec_diagonal_reference`](crate::reference::BlasMatVecReference::blas_matvec_diagonal_reference) bit
    /// for bit, within [`Self::blas_matvec_diagonal_tmp_bytes`]; validator: [`crate::layouts::validate_blas_matvec_diagonal`].
    ///
    /// **Fixed:** poulpy-ckks's `ckks_eval_linear_transformation_into` of `m`'s ciphertext diagonals from `a`, resident or
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
        R: BlasVectorMut<Self>,
        M: BlasDiagonalCiphertextMap<Self>,
        A: BlasVectorRef<Self>,
        K: GetAutomorphismKey<Self> + GetTensorKey<Self> + Sync;

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
    /// **Fixed:** the masked accumulates of the reference, each poulpy-ckks's `ckks_dot_product_prepared_pt_add` of the
    /// mask read through its shift's Galois element and a baby rotation of `v`; the key widths and the errors and their order: the validator's, then poulpy-ckks's, before any write to `m`.
    ///
    /// **Free:** kernels, scratch use, how the rotations of `v` are made and prepared, and the order of the entries.
    fn blas_append_row_diagonal<V, K>(
        module: &Module<Self>,
        m: &mut BlasDiagonalCiphertext<Self>,
        row: i64,
        v: &V,
        mask: &CKKSPreparedRight<Self>,
        keys: &K,
        scratch: &mut ScratchArena<'_, Self>,
    ) -> Result<()>
    where
        V: BlasVectorRef<Self>,
        K: GetAutomorphismKey<Self> + Sync;

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
        m: &mut BlasDiagonalCiphertext<Self>,
        col: i64,
        v: &V,
        mask: &CKKSPreparedRight<Self>,
        keys: &K,
        scratch: &mut ScratchArena<'_, Self>,
    ) -> Result<()>
    where
        V: BlasVectorRef<Self>,
        K: GetAutomorphismKey<Self> + Sync;
}

/// Selects [`BlasMatVecReference`](crate::reference::BlasMatVecReference) for `$be`.
#[macro_export]
macro_rules! impl_blas_mat_vec_reference_full {
    ($be:ty) => {
        unsafe impl $crate::oep::BlasMatVecImpl for $be {
            fn blas_matvec_pt_tmp_bytes<R, W, A, K>(
                module: &::poulpy_hal::layouts::Module<Self>,
                res: &R,
                w: &W,
                a: &A,
                keys: &K,
            ) -> usize
            where
                R: $crate::layouts::BlasVectorInfos,
                W: $crate::layouts::BlasMatVecMaps<Self>,
                A: $crate::layouts::BlasVectorInfos,
                K: ::poulpy_core::layouts::GGLWEInfos,
            {
                $crate::reference::BlasMatVecReference::blas_matvec_pt_tmp_bytes_reference(module, res, w, a, keys)
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
                R: $crate::layouts::BlasVectorMut<Self>,
                W: $crate::layouts::BlasMatVecMaps<Self>,
                A: $crate::layouts::BlasVectorRef<Self>,
                K: ::poulpy_core::layouts::GetAutomorphismKey<Self> + ::poulpy_core::layouts::GetTensorKey<Self> + Sync,
            {
                $crate::reference::BlasMatVecReference::blas_matvec_pt_reference(module, res, w, a, keys, scratch)
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
                M: $crate::layouts::BlasDiagonalCiphertextMap<Self>,
                A: $crate::layouts::BlasVectorInfos,
                T: ::poulpy_core::layouts::GGLWEInfos,
                K: ::poulpy_core::layouts::GGLWEInfos,
            {
                $crate::reference::BlasMatVecReference::blas_matvec_diagonal_tmp_bytes_reference(module, res, m, a, tsk, keys)
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
                R: $crate::layouts::BlasVectorMut<Self>,
                M: $crate::layouts::BlasDiagonalCiphertextMap<Self>,
                A: $crate::layouts::BlasVectorRef<Self>,
                K: ::poulpy_core::layouts::GetAutomorphismKey<Self> + ::poulpy_core::layouts::GetTensorKey<Self> + Sync,
            {
                $crate::reference::BlasMatVecReference::blas_matvec_diagonal_reference(module, res, m, a, keys, scratch)
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
                $crate::reference::BlasMatVecReference::blas_append_row_diagonal_tmp_bytes_reference(module, m, v, mask, keys)
            }

            fn blas_append_row_diagonal<V, K>(
                module: &::poulpy_hal::layouts::Module<Self>,
                m: &mut $crate::layouts::BlasDiagonalCiphertext<Self>,
                row: i64,
                v: &V,
                mask: &::poulpy_ckks::layouts::CKKSPreparedRight<Self>,
                keys: &K,
                scratch: &mut ::poulpy_hal::layouts::ScratchArena<'_, Self>,
            ) -> ::poulpy_ckks::CKKSResult<()>
            where
                V: $crate::layouts::BlasVectorRef<Self>,
                K: ::poulpy_core::layouts::GetAutomorphismKey<Self> + Sync,
            {
                $crate::reference::BlasMatVecReference::blas_append_row_diagonal_reference(module, m, row, v, mask, keys, scratch)
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
                $crate::reference::BlasMatVecReference::blas_append_col_diagonal_tmp_bytes_reference(module, m, v, mask, keys)
            }

            fn blas_append_col_diagonal<V, K>(
                module: &::poulpy_hal::layouts::Module<Self>,
                m: &mut $crate::layouts::BlasDiagonalCiphertext<Self>,
                col: i64,
                v: &V,
                mask: &::poulpy_ckks::layouts::CKKSPreparedRight<Self>,
                keys: &K,
                scratch: &mut ::poulpy_hal::layouts::ScratchArena<'_, Self>,
            ) -> ::poulpy_ckks::CKKSResult<()>
            where
                V: $crate::layouts::BlasVectorRef<Self>,
                K: ::poulpy_core::layouts::GetAutomorphismKey<Self> + Sync,
            {
                $crate::reference::BlasMatVecReference::blas_append_col_diagonal_reference(module, m, col, v, mask, keys, scratch)
            }
        }
    };
}
