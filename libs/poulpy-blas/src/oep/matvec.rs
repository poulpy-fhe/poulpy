use poulpy_ckks::CKKSResult as Result;
use poulpy_core::layouts::{GGLWEInfos, GetAutomorphismKey, GetTensorKey};
use poulpy_hal::layouts::{Backend, Module, ScratchArena};

use crate::layouts::{BlasMatVecMaps, BlasVectorInfos, BlasVectorMut, BlasVectorRef};

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
        }
    };
}
