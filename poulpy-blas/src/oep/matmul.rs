use poulpy_ckks::CKKSResult as Result;
use poulpy_core::layouts::{GGLWEInfos, GetAutomorphismKey, GetTensorKey};
use poulpy_hal::layouts::{Backend, Module, ScratchArena};

use super::derived;
use crate::{
    layouts::{BlasContext, BlasGrid, BlasGridInfos, BlasGridMut, BlasPtOperand},
    reference::Left,
};

/// Extension point of [`crate::api::BlasMatMulOps`]: the prepared operands, the preparations, the prepared product and
/// the products by public matrices, on the right and on the left, are required, and `blas_matmul` is provided over the
/// preparations and the prepared product.
///
/// # Safety
///
/// Every required method reproduces its `_reference` counterpart in [`crate::reference::BlasMatMulReference`] bit for
/// bit, digits and CKKS metadata, within the scratch the matching `_tmp_bytes_impl` reports; with less, it panics
/// before any write to `res` when it looks up a key, as the reference does.
///
/// **Prepared operands.** [`Self::MatMulLeft`] and [`Self::MatMulRight`] may hold any representation, made only by the
/// preparations. Through [`BlasGridInfos`], an operand prepared from `a` under `ctx` reports `a`'s dimensions, the
/// layout [`crate::reference::blas_matmul_term_layout`] gives and `Some` of `a`'s batch,
/// [`grid_batch::<Self::Ring>(a)`](crate::layouts::grid_batch), which reads the batch of an input without a record from
/// its label: the `_prepared` validator, the batch the API records and the provided `blas_matmul` query read nothing
/// else. The prepared product of two operands equals the reference's on the reference's operands of the same sources.
/// The preparations look up rotation keys only, the prepared product the tensor key only.
///
/// **Fixed** (anything not listed as free changes bits):
/// - the `d` terms of a source block `a_il`, all at the width `w = a.k - 2p` for the maps' scale `p`: term `0` is sigma
///   of `a_il` evaluated into a block of `a`'s layout, which leaves it at `a.k - p`, then narrowed exactly to `w` by
///   poulpy-ckks's `ckks_copy`; term `k` is phi^k of that sigma into a block of width `w`, through the `Direct` shift
///   maps the plan records. On the right, tau and psi^k;
/// - one JKLS kernel per block of `res`, [`crate::oep::BlasJklsImpl::blas_jkls_block_impl`], over its `K d` pairs of
///   terms `(a_il[k], b_lj[k])`; the kernel's `# Safety` fixes its circuit;
/// - the key widths: sigma's (tau's) baby steps at `a.k` and giant steps at `a.k - p`, the shifts' giant steps at `w`,
///   the tensor key at the wider of both sides' `w`; a public operand's baby steps at `a.k` and giant steps at
///   `min(res.k, a.k - p_B)` for its scale `p_B`;
/// - the products by public matrices share their giant steps across the inner index: block `res_ij` is poulpy-ckks's
///   `ckks_eval_linear_transformation_sum_into` of its `K` terms in ascending `l`, `(a_il, b_lj)` on the right and
///   `(a_lj, b_il)` on the left, then stamped with `a`'s label. Each giant step sums the products of every term in the
///   transform domain, in that order, before its one rotation; an FFT backend rounds that sum, so the order is part of
///   the bits.
///
/// **Free (bit-identical):** the order and split of the blocks over workers; materializing or streaming the terms;
/// caching them or their convolution forms; sharing baby-step caches; batching key switches. Hoisting the shifts
/// changes bits and the key widths, so it would arrive as a plan variant.
///
/// The API runs [`crate::reference::validate_blas_matmul`], [`crate::reference::validate_blas_matmul_prepare_left`],
/// [`crate::reference::validate_blas_matmul_prepare_right`], [`crate::reference::validate_blas_matmul_prepared`],
/// [`crate::reference::validate_blas_matmul_pt`] or [`crate::reference::validate_blas_matmul_pt_left`] first, so a
/// method runs only on inputs its validator accepted, and must return `Ok` on them, or poulpy-ckks's budget error before
/// any write to `res`. Any other `Err` is a backend fault (a validator gap or a device failure), and `res` is then
/// unspecified.
pub unsafe trait BlasMatMulImpl: Backend {
    /// The prepared left operand of [`Self::blas_matmul_prepared_impl`].
    type MatMulLeft: BlasGridInfos + Send + Sync;

    /// The prepared right operand of [`Self::blas_matmul_prepared_impl`].
    type MatMulRight: BlasGridInfos + Send + Sync;

    fn blas_matmul_prepare_left_tmp_bytes_impl<A, K>(module: &Module<Self>, a: &A, ctx: &BlasContext<Self>, atk: &K) -> usize
    where
        A: BlasGridInfos,
        K: GGLWEInfos;

    fn blas_matmul_prepare_left_impl<A, K>(
        module: &Module<Self>,
        a: &A,
        ctx: &BlasContext<Self>,
        atk: &K,
        scratch: &mut ScratchArena<'_, Self>,
    ) -> Result<Self::MatMulLeft>
    where
        A: BlasGrid<Self>,
        K: GetAutomorphismKey<Self> + Sync;

    fn blas_matmul_prepare_right_tmp_bytes_impl<B, K>(module: &Module<Self>, b: &B, ctx: &BlasContext<Self>, atk: &K) -> usize
    where
        B: BlasGridInfos,
        K: GGLWEInfos;

    fn blas_matmul_prepare_right_impl<B, K>(
        module: &Module<Self>,
        b: &B,
        ctx: &BlasContext<Self>,
        atk: &K,
        scratch: &mut ScratchArena<'_, Self>,
    ) -> Result<Self::MatMulRight>
    where
        B: BlasGrid<Self>,
        K: GetAutomorphismKey<Self> + Sync;

    /// Scratch of [`Self::blas_matmul_prepared_impl`]; `a` and `b` are the operands or the layouts
    /// [`crate::reference::blas_matmul_term_layout`] gives.
    fn blas_matmul_prepared_tmp_bytes_impl<R, A, B, T>(module: &Module<Self>, res: &R, a: &A, b: &B, tsk: &T) -> usize
    where
        R: BlasGridInfos,
        A: BlasGridInfos,
        B: BlasGridInfos,
        T: GGLWEInfos;

    fn blas_matmul_prepared_impl<R, T>(
        module: &Module<Self>,
        res: &mut R,
        a: &Self::MatMulLeft,
        b: &Self::MatMulRight,
        tsk: &T,
        scratch: &mut ScratchArena<'_, Self>,
    ) -> Result<()>
    where
        R: BlasGridMut<Self>,
        T: GetTensorKey<Self> + Sync;

    fn blas_matmul_pt_tmp_bytes_impl<R, A, K>(module: &Module<Self>, res: &R, a: &A, b: &BlasPtOperand<Self>, atk: &K) -> usize
    where
        R: BlasGridInfos,
        A: BlasGridInfos,
        K: GGLWEInfos;

    fn blas_matmul_pt_impl<R, A, K>(
        module: &Module<Self>,
        res: &mut R,
        a: &A,
        b: &BlasPtOperand<Self>,
        atk: &K,
        scratch: &mut ScratchArena<'_, Self>,
    ) -> Result<()>
    where
        R: BlasGridMut<Self>,
        A: BlasGrid<Self>,
        K: GetAutomorphismKey<Self> + Sync;

    fn blas_matmul_pt_left_tmp_bytes_impl<R, A, K>(
        module: &Module<Self>,
        res: &R,
        b: &BlasPtOperand<Self, Left>,
        a: &A,
        atk: &K,
    ) -> usize
    where
        R: BlasGridInfos,
        A: BlasGridInfos,
        K: GGLWEInfos;

    /// `res = b a`, the product by public matrices on the left.
    fn blas_matmul_pt_left_impl<R, A, K>(
        module: &Module<Self>,
        res: &mut R,
        b: &BlasPtOperand<Self, Left>,
        a: &A,
        atk: &K,
        scratch: &mut ScratchArena<'_, Self>,
    ) -> Result<()>
    where
        R: BlasGridMut<Self>,
        A: BlasGrid<Self>,
        K: GetAutomorphismKey<Self> + Sync;

    /// The larger of both preparations' queries and the prepared product's on their term layouts.
    fn blas_matmul_tmp_bytes_impl<R, A, B, T, K>(
        module: &Module<Self>,
        res: &R,
        a: &A,
        b: &B,
        ctx: &BlasContext<Self>,
        tsk: &T,
        atk: &K,
    ) -> usize
    where
        R: BlasGridInfos,
        A: BlasGridInfos,
        B: BlasGridInfos,
        T: GGLWEInfos,
        K: GGLWEInfos,
    {
        derived::blas_matmul_tmp_bytes(module, res, a, b, ctx, tsk, atk)
    }

    /// [`Self::blas_matmul_prepare_left_impl`] on `a`, [`Self::blas_matmul_prepare_right_impl`] on `b`, then
    /// [`Self::blas_matmul_prepared_impl`] into `res`.
    #[allow(clippy::too_many_arguments)]
    fn blas_matmul_impl<R, A, B, T, K>(
        module: &Module<Self>,
        res: &mut R,
        a: &A,
        b: &B,
        ctx: &BlasContext<Self>,
        tsk: &T,
        atk: &K,
        scratch: &mut ScratchArena<'_, Self>,
    ) -> Result<()>
    where
        R: BlasGridMut<Self>,
        A: BlasGrid<Self>,
        B: BlasGrid<Self>,
        T: GetTensorKey<Self> + Sync,
        K: GetAutomorphismKey<Self> + Sync,
    {
        derived::blas_matmul(module, res, a, b, ctx, tsk, atk, scratch)
    }
}

/// Selects [`crate::reference::BlasMatMulReference`] for `$be`, with [`crate::reference::BlasMatMulOperand`] as both
/// prepared operands, under the provided `blas_matmul`. The reference prepares through `$be`'s map hook and multiplies
/// through its JKLS kernel, so `$be` also implements [`crate::oep::BlasMatIntraImpl`] and [`crate::oep::BlasJklsImpl`].
#[macro_export]
macro_rules! impl_blas_matmul_reference {
    ($be:ty) => {
        unsafe impl $crate::oep::BlasMatMulImpl for $be {
            type MatMulLeft = $crate::reference::BlasMatMulOperand<Self, $crate::reference::Left>;
            type MatMulRight = $crate::reference::BlasMatMulOperand<Self, $crate::reference::Right>;

            fn blas_matmul_prepare_left_tmp_bytes_impl<A, K>(
                module: &::poulpy_hal::layouts::Module<Self>,
                a: &A,
                ctx: &$crate::layouts::BlasContext<Self>,
                atk: &K,
            ) -> usize
            where
                A: $crate::layouts::BlasGridInfos,
                K: ::poulpy_core::layouts::GGLWEInfos,
            {
                $crate::reference::BlasMatMulReference::blas_matmul_prepare_left_tmp_bytes_reference(module, a, ctx, atk)
            }

            fn blas_matmul_prepare_left_impl<A, K>(
                module: &::poulpy_hal::layouts::Module<Self>,
                a: &A,
                ctx: &$crate::layouts::BlasContext<Self>,
                atk: &K,
                scratch: &mut ::poulpy_hal::layouts::ScratchArena<'_, Self>,
            ) -> ::poulpy_ckks::CKKSResult<Self::MatMulLeft>
            where
                A: $crate::layouts::BlasGrid<Self>,
                K: ::poulpy_core::layouts::GetAutomorphismKey<Self> + Sync,
            {
                $crate::reference::BlasMatMulReference::blas_matmul_prepare_left_reference(module, a, ctx, atk, scratch)
            }

            fn blas_matmul_prepare_right_tmp_bytes_impl<B, K>(
                module: &::poulpy_hal::layouts::Module<Self>,
                b: &B,
                ctx: &$crate::layouts::BlasContext<Self>,
                atk: &K,
            ) -> usize
            where
                B: $crate::layouts::BlasGridInfos,
                K: ::poulpy_core::layouts::GGLWEInfos,
            {
                $crate::reference::BlasMatMulReference::blas_matmul_prepare_right_tmp_bytes_reference(module, b, ctx, atk)
            }

            fn blas_matmul_prepare_right_impl<B, K>(
                module: &::poulpy_hal::layouts::Module<Self>,
                b: &B,
                ctx: &$crate::layouts::BlasContext<Self>,
                atk: &K,
                scratch: &mut ::poulpy_hal::layouts::ScratchArena<'_, Self>,
            ) -> ::poulpy_ckks::CKKSResult<Self::MatMulRight>
            where
                B: $crate::layouts::BlasGrid<Self>,
                K: ::poulpy_core::layouts::GetAutomorphismKey<Self> + Sync,
            {
                $crate::reference::BlasMatMulReference::blas_matmul_prepare_right_reference(module, b, ctx, atk, scratch)
            }

            fn blas_matmul_prepared_tmp_bytes_impl<R, A, B, T>(
                module: &::poulpy_hal::layouts::Module<Self>,
                res: &R,
                a: &A,
                b: &B,
                tsk: &T,
            ) -> usize
            where
                R: $crate::layouts::BlasGridInfos,
                A: $crate::layouts::BlasGridInfos,
                B: $crate::layouts::BlasGridInfos,
                T: ::poulpy_core::layouts::GGLWEInfos,
            {
                $crate::reference::BlasMatMulReference::blas_matmul_prepared_tmp_bytes_reference(module, res, a, b, tsk)
            }

            fn blas_matmul_prepared_impl<R, T>(
                module: &::poulpy_hal::layouts::Module<Self>,
                res: &mut R,
                a: &Self::MatMulLeft,
                b: &Self::MatMulRight,
                tsk: &T,
                scratch: &mut ::poulpy_hal::layouts::ScratchArena<'_, Self>,
            ) -> ::poulpy_ckks::CKKSResult<()>
            where
                R: $crate::layouts::BlasGridMut<Self>,
                T: ::poulpy_core::layouts::GetTensorKey<Self> + Sync,
            {
                $crate::reference::BlasMatMulReference::blas_matmul_prepared_reference(module, res, a, b, tsk, scratch)
            }

            fn blas_matmul_pt_tmp_bytes_impl<R, A, K>(
                module: &::poulpy_hal::layouts::Module<Self>,
                res: &R,
                a: &A,
                b: &$crate::layouts::BlasPtOperand<Self>,
                atk: &K,
            ) -> usize
            where
                R: $crate::layouts::BlasGridInfos,
                A: $crate::layouts::BlasGridInfos,
                K: ::poulpy_core::layouts::GGLWEInfos,
            {
                $crate::reference::BlasMatMulReference::blas_matmul_pt_tmp_bytes_reference(module, res, a, b, atk)
            }

            fn blas_matmul_pt_impl<R, A, K>(
                module: &::poulpy_hal::layouts::Module<Self>,
                res: &mut R,
                a: &A,
                b: &$crate::layouts::BlasPtOperand<Self>,
                atk: &K,
                scratch: &mut ::poulpy_hal::layouts::ScratchArena<'_, Self>,
            ) -> ::poulpy_ckks::CKKSResult<()>
            where
                R: $crate::layouts::BlasGridMut<Self>,
                A: $crate::layouts::BlasGrid<Self>,
                K: ::poulpy_core::layouts::GetAutomorphismKey<Self> + Sync,
            {
                $crate::reference::BlasMatMulReference::blas_matmul_pt_reference(module, res, a, b, atk, scratch)
            }

            fn blas_matmul_pt_left_tmp_bytes_impl<R, A, K>(
                module: &::poulpy_hal::layouts::Module<Self>,
                res: &R,
                b: &$crate::layouts::BlasPtOperand<Self, $crate::reference::Left>,
                a: &A,
                atk: &K,
            ) -> usize
            where
                R: $crate::layouts::BlasGridInfos,
                A: $crate::layouts::BlasGridInfos,
                K: ::poulpy_core::layouts::GGLWEInfos,
            {
                $crate::reference::BlasMatMulReference::blas_matmul_pt_left_tmp_bytes_reference(module, res, b, a, atk)
            }

            fn blas_matmul_pt_left_impl<R, A, K>(
                module: &::poulpy_hal::layouts::Module<Self>,
                res: &mut R,
                b: &$crate::layouts::BlasPtOperand<Self, $crate::reference::Left>,
                a: &A,
                atk: &K,
                scratch: &mut ::poulpy_hal::layouts::ScratchArena<'_, Self>,
            ) -> ::poulpy_ckks::CKKSResult<()>
            where
                R: $crate::layouts::BlasGridMut<Self>,
                A: $crate::layouts::BlasGrid<Self>,
                K: ::poulpy_core::layouts::GetAutomorphismKey<Self> + Sync,
            {
                $crate::reference::BlasMatMulReference::blas_matmul_pt_left_reference(module, res, b, a, atk, scratch)
            }
        }
    };
}
