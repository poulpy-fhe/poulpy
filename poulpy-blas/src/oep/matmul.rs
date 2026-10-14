use poulpy_ckks::CKKSResult as Result;
use poulpy_core::layouts::{GGLWEInfos, GetAutomorphismKey, GetTensorKey};
use poulpy_hal::layouts::{Backend, Module, PrepareHint, ScratchArena};

use super::derived;
use crate::{
    layouts::{BlasContext, BlasGrid, BlasGridInfos, BlasGridMut, BlasMatMulOperandInfos, BlasPtOperand},
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
/// preparations, under a [`PrepareHint`]: hot, `OneShot`, for one product, or cold, `Reuse`, for many. Through
/// [`BlasMatMulOperandInfos`], an operand prepared from `a` reports the hint it was given, `a`'s block layout as its
/// source, `a`'s dimensions, `Some` of `a`'s batch, [`grid_batch::<Self::Ring>(a)`](crate::layouts::grid_batch), which
/// reads the batch of an input without a record from its label, and as its block layout `a`'s when hot, the layout
/// [`crate::reference::blas_matmul_term_layout`] gives under the preparation's `ctx` when cold:
/// [`crate::reference::blas_matmul_operand_layout`] gives these infos before preparing. The `_prepared` validator, the
/// batch the API records and the provided `blas_matmul` query read nothing else. The terms of a cold operand are those
/// of the preparation's `ctx`; those of a hot operand are those of the `ctx` of the product it enters, whatever the
/// context it was prepared under, at whatever scale: the product evaluates its maps, which spend their scale from the
/// operand's budget, and stamps by the ct x ct rule. So a hot operand holds its source, or terms it recomputes when the
/// product's `ctx` is not the one they came from; the reference's is a verbatim copy of the source. The prepared
/// product of two operands, of any hints, equals the reference's on the reference's operands of the same sources and
/// hints under the same contexts, and so `blas_matmul` of the sources when every side is hot or prepared under the
/// product's `ctx`. A cold preparation looks up the rotation keys of its terms and a hot one none, after its validator
/// checked them under either hint; the prepared product looks up those of each hot side's terms under its `ctx`, and
/// the tensor key.
///
/// **Fixed** (anything not listed as free changes bits):
/// - the `d` terms of a source block `a_il`, all at the width `w = a.k - 2p` for the maps' scale `p`: term `0` is sigma
///   of `a_il` evaluated into a block of `a`'s layout, which leaves it at `a.k - p`, then narrowed exactly to `w` by
///   poulpy-ckks's `ckks_copy`; term `k` is phi^k of that sigma into a block of width `w`, through the `Direct` shift
///   maps the plan records. On the right, tau and psi^k. Each term is converted once, by poulpy-ckks's
///   `ckks_prepare_left` (`ckks_prepare_right`), on its `ceil(w / base2k)` limbs;
/// - one JKLS kernel per block of `res`, [`crate::oep::BlasJklsImpl::blas_jkls_block_impl`], over its `K d` pairs of
///   terms `(a_il[k], b_lj[k])`, `l`-major: ascending `l`, then ascending `k` within each `l`, the order the one-pass
///   kernel sums them in, part of the bits on FFT; the kernel's `# Safety` fixes its circuit;
/// - the bits, whatever the hints: the prepared product equals the reference's on the reference's operands of the same
///   sources and hints under the same contexts, so an operand prepared under the product's `ctx` gives the same
///   products under either hint;
/// - the key widths: sigma's (tau's) baby steps at `a.k` and giant steps at `a.k - p`, the shifts' giant steps at `w`,
///   the tensor key at the wider of both sides' `w`; a public operand's baby steps at `a.k` and giant steps at
///   `min(res.k, a.k - p_B)` for its scale `p_B`;
/// - the errors and their order: the validator's, then poulpy-ckks's budget errors, before any write to `res`: the left
///   terms', then the right terms', then the first block's ct x ct rule. A hot side's come from the product, so a hot
///   preparation of an input short of its budget returns `Ok`;
/// - the products by public matrices share their giant steps across the inner index: block `res_ij` is poulpy-ckks's
///   `ckks_eval_linear_transformation_sum_into` of its `K` terms in ascending `l`, `(a_il, b_lj)` on the right and
///   `(a_lj, b_il)` on the left, then stamped with `a`'s label. Each giant step sums the products of every term in the
///   transform domain, in that order, before its one rotation; an FFT backend rounds that sum, so the order is part of
///   the bits.
///
/// **Free (bit-identical):** the representation of an operand, under either hint: a hot operand may hold its source
/// and evaluate its terms inside the product, or prepare them eagerly, as long as it reports `OneShot` and its source,
/// returns its budget error from the product and gives the terms of the product's `ctx`; the order of the blocks;
/// which side the product holds and which it streams; materializing or streaming the terms; caching them or their
/// convolution forms; sharing baby-step caches; batching key switches. Hoisting the shifts changes bits and the key
/// widths, so it would arrive as a plan variant.
///
/// The API runs [`crate::reference::validate_blas_matmul`], [`crate::reference::validate_blas_matmul_prepare_left`],
/// [`crate::reference::validate_blas_matmul_prepare_right`], [`crate::reference::validate_blas_matmul_prepared`],
/// [`crate::reference::validate_blas_matmul_pt`] or [`crate::reference::validate_blas_matmul_pt_left`] first, so a
/// method runs only on inputs its validator accepted, and must return `Ok` on them, or poulpy-ckks's budget error before
/// any write to `res`. Any other `Err` is a backend fault (a validator gap or a device failure), and `res` is then
/// unspecified.
pub unsafe trait BlasMatMulImpl: Backend {
    /// The prepared left operand of [`Self::blas_matmul_prepared_impl`], hot or cold.
    type MatMulLeft: BlasMatMulOperandInfos + Send + Sync;

    /// The prepared right operand of [`Self::blas_matmul_prepared_impl`], hot or cold.
    type MatMulRight: BlasMatMulOperandInfos + Send + Sync;

    fn blas_matmul_prepare_left_tmp_bytes_impl<A, K>(
        module: &Module<Self>,
        a: &A,
        ctx: &BlasContext<Self>,
        atk: &K,
        hint: PrepareHint,
    ) -> usize
    where
        A: BlasGridInfos,
        K: GGLWEInfos;

    fn blas_matmul_prepare_left_impl<A, K>(
        module: &Module<Self>,
        a: &A,
        ctx: &BlasContext<Self>,
        atk: &K,
        hint: PrepareHint,
        scratch: &mut ScratchArena<'_, Self>,
    ) -> Result<Self::MatMulLeft>
    where
        A: BlasGrid<Self>,
        K: GetAutomorphismKey<Self> + Sync;

    fn blas_matmul_prepare_right_tmp_bytes_impl<B, K>(
        module: &Module<Self>,
        b: &B,
        ctx: &BlasContext<Self>,
        atk: &K,
        hint: PrepareHint,
    ) -> usize
    where
        B: BlasGridInfos,
        K: GGLWEInfos;

    fn blas_matmul_prepare_right_impl<B, K>(
        module: &Module<Self>,
        b: &B,
        ctx: &BlasContext<Self>,
        atk: &K,
        hint: PrepareHint,
        scratch: &mut ScratchArena<'_, Self>,
    ) -> Result<Self::MatMulRight>
    where
        B: BlasGrid<Self>,
        K: GetAutomorphismKey<Self> + Sync;

    /// Scratch of [`Self::blas_matmul_prepared_impl`] under `ctx`; `a` and `b` are the operands or the layouts
    /// [`crate::reference::blas_matmul_operand_layout`] gives.
    fn blas_matmul_prepared_tmp_bytes_impl<R, A, B, T, K>(
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
        A: BlasMatMulOperandInfos,
        B: BlasMatMulOperandInfos,
        T: GGLWEInfos,
        K: GGLWEInfos;

    #[allow(clippy::too_many_arguments)]
    fn blas_matmul_prepared_impl<R, T, K>(
        module: &Module<Self>,
        res: &mut R,
        a: &Self::MatMulLeft,
        b: &Self::MatMulRight,
        ctx: &BlasContext<Self>,
        tsk: &T,
        atk: &K,
        scratch: &mut ScratchArena<'_, Self>,
    ) -> Result<()>
    where
        R: BlasGridMut<Self>,
        T: GetTensorKey<Self> + Sync,
        K: GetAutomorphismKey<Self> + Sync;

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

    /// The larger of both hot preparations' queries and the prepared product's on the layouts they give.
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

    /// [`Self::blas_matmul_prepare_left_impl`] on `a` and [`Self::blas_matmul_prepare_right_impl`] on `b`, both hot,
    /// `OneShot`, then [`Self::blas_matmul_prepared_impl`] into `res` under `ctx`.
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
/// prepared operands, under the provided `blas_matmul`. The reference prepares cold operands, and the terms of hot
/// ones inside the product, through `$be`'s map hook, converts them through its poulpy-ckks preparations and multiplies
/// them through its JKLS kernel, so `$be` also implements [`crate::oep::BlasMatIntraImpl`] and
/// [`crate::oep::BlasJklsImpl`].
#[macro_export]
macro_rules! impl_blas_matmul_reference {
    ($be:ty) => {
        unsafe impl $crate::oep::BlasMatMulImpl for $be {
            type MatMulLeft = $crate::reference::BlasMatMulOperand<Self, ::poulpy_ckks::layouts::CKKSPreparedLeft<Self>>;
            type MatMulRight = $crate::reference::BlasMatMulOperand<Self, ::poulpy_ckks::layouts::CKKSPreparedRight<Self>>;

            fn blas_matmul_prepare_left_tmp_bytes_impl<A, K>(
                module: &::poulpy_hal::layouts::Module<Self>,
                a: &A,
                ctx: &$crate::layouts::BlasContext<Self>,
                atk: &K,
                hint: ::poulpy_hal::layouts::PrepareHint,
            ) -> usize
            where
                A: $crate::layouts::BlasGridInfos,
                K: ::poulpy_core::layouts::GGLWEInfos,
            {
                $crate::reference::BlasMatMulReference::blas_matmul_prepare_left_tmp_bytes_reference(module, a, ctx, atk, hint)
            }

            fn blas_matmul_prepare_left_impl<A, K>(
                module: &::poulpy_hal::layouts::Module<Self>,
                a: &A,
                ctx: &$crate::layouts::BlasContext<Self>,
                atk: &K,
                hint: ::poulpy_hal::layouts::PrepareHint,
                scratch: &mut ::poulpy_hal::layouts::ScratchArena<'_, Self>,
            ) -> ::poulpy_ckks::CKKSResult<Self::MatMulLeft>
            where
                A: $crate::layouts::BlasGrid<Self>,
                K: ::poulpy_core::layouts::GetAutomorphismKey<Self> + Sync,
            {
                $crate::reference::BlasMatMulReference::blas_matmul_prepare_left_reference(module, a, ctx, atk, hint, scratch)
            }

            fn blas_matmul_prepare_right_tmp_bytes_impl<B, K>(
                module: &::poulpy_hal::layouts::Module<Self>,
                b: &B,
                ctx: &$crate::layouts::BlasContext<Self>,
                atk: &K,
                hint: ::poulpy_hal::layouts::PrepareHint,
            ) -> usize
            where
                B: $crate::layouts::BlasGridInfos,
                K: ::poulpy_core::layouts::GGLWEInfos,
            {
                $crate::reference::BlasMatMulReference::blas_matmul_prepare_right_tmp_bytes_reference(module, b, ctx, atk, hint)
            }

            fn blas_matmul_prepare_right_impl<B, K>(
                module: &::poulpy_hal::layouts::Module<Self>,
                b: &B,
                ctx: &$crate::layouts::BlasContext<Self>,
                atk: &K,
                hint: ::poulpy_hal::layouts::PrepareHint,
                scratch: &mut ::poulpy_hal::layouts::ScratchArena<'_, Self>,
            ) -> ::poulpy_ckks::CKKSResult<Self::MatMulRight>
            where
                B: $crate::layouts::BlasGrid<Self>,
                K: ::poulpy_core::layouts::GetAutomorphismKey<Self> + Sync,
            {
                $crate::reference::BlasMatMulReference::blas_matmul_prepare_right_reference(module, b, ctx, atk, hint, scratch)
            }

            fn blas_matmul_prepared_tmp_bytes_impl<R, A, B, T, K>(
                module: &::poulpy_hal::layouts::Module<Self>,
                res: &R,
                a: &A,
                b: &B,
                ctx: &$crate::layouts::BlasContext<Self>,
                tsk: &T,
                atk: &K,
            ) -> usize
            where
                R: $crate::layouts::BlasGridInfos,
                A: $crate::layouts::BlasMatMulOperandInfos,
                B: $crate::layouts::BlasMatMulOperandInfos,
                T: ::poulpy_core::layouts::GGLWEInfos,
                K: ::poulpy_core::layouts::GGLWEInfos,
            {
                $crate::reference::BlasMatMulReference::blas_matmul_prepared_tmp_bytes_reference(module, res, a, b, ctx, tsk, atk)
            }

            fn blas_matmul_prepared_impl<R, T, K>(
                module: &::poulpy_hal::layouts::Module<Self>,
                res: &mut R,
                a: &Self::MatMulLeft,
                b: &Self::MatMulRight,
                ctx: &$crate::layouts::BlasContext<Self>,
                tsk: &T,
                atk: &K,
                scratch: &mut ::poulpy_hal::layouts::ScratchArena<'_, Self>,
            ) -> ::poulpy_ckks::CKKSResult<()>
            where
                R: $crate::layouts::BlasGridMut<Self>,
                T: ::poulpy_core::layouts::GetTensorKey<Self> + Sync,
                K: ::poulpy_core::layouts::GetAutomorphismKey<Self> + Sync,
            {
                $crate::reference::BlasMatMulReference::blas_matmul_prepared_reference(module, res, a, b, ctx, tsk, atk, scratch)
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
