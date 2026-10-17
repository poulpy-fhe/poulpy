use poulpy_ckks::{CKKSResult as Result, oep::CKKSMulImpl};
use poulpy_core::layouts::{GGLWEInfos, GetAutomorphismKey, GetTensorKey};
use poulpy_hal::layouts::{Backend, Module, PrepareHint, ScratchArena};

use super::derived;
use crate::layouts::{
    BlasContext, BlasGrid, BlasGridInfos, BlasGridMut, BlasMatMulLeft, BlasMatMulOperandInfos, BlasMatMulRight, BlasPtOperand,
    Left,
};

/// Extension point of [`crate::api::BlasMatMulOps`]: the preparations, the prepared product and the products by public
/// matrices, on the right and on the left, are required, and `blas_matmul` is provided over the preparations and the
/// prepared product.
///
/// # Safety
///
/// Every required method reproduces its `_reference` counterpart in
/// [`BlasMatMulReference`](crate::reference::BlasMatMulReference) bit for bit, digits and CKKS metadata, within the
/// scratch the matching `_tmp_bytes_impl` reports; with less, it panics before any write to `res` when it looks up a
/// key, as the reference does. The memory, layout and aliasing rules of `poulpy-hal/docs/backend_safety_contract.md`
/// apply.
///
/// **Prepared operands.** A backend does not choose their type: [`BlasMatMulLeft`] and [`BlasMatMulRight`] are
/// [`BlasMatMulOperand`](crate::layouts::BlasMatMulOperand)s of the backend's poulpy-ckks ciphertexts and prepared
/// terms, which [`BlasMatMulOperandPreparedFactory`](crate::layouts::BlasMatMulOperandPreparedFactory) allocates. A
/// preparation fills one through its `blocks_mut` or `terms_mut`, with no allocation of its storage, from a source
/// whose infos have its shape (the validator checks the hint, the dimensions and the GLWE layouts, not the metadata),
/// under its hint, and returns poulpy-ckks's budget error before any write to it, which a cold preparation evaluates
/// through heap working blocks. It then records with `set_prepared` the infos
/// [`blas_matmul_left_operand_layout`](crate::layouts::blas_matmul_left_operand_layout)
/// ([`blas_matmul_right_operand_layout`](crate::layouts::blas_matmul_right_operand_layout)) gives for its source under
/// the preparation's `ctx`, and the source's batch, [`grid_batch::<Self::Ring>(a)`](crate::layouts::grid_batch). The
/// `_prepared` validator, the batch the prepared product records and the provided `blas_matmul` query read nothing
/// else. The terms of a cold operand are those of the preparation's `ctx`; those of a hot operand are those of the
/// `ctx` of the product it enters, whatever the context it was prepared under, at whatever scale: the product evaluates
/// its maps, which spend their scale from the operand's budget, and stamps by the ct x ct rule. So a hot operand holds
/// a copy of its source's live limbs. A cold preparation looks up the rotation keys of its terms and a hot one none,
/// after its validator checked them under either hint; the prepared product looks up those of each hot side's terms
/// under its `ctx`, and the tensor key.
///
/// **Fixed** (anything not listed as free changes bits):
/// - the `d` terms of a source block `a_il`, all at the width `w = a.k - 2p` for the maps' scale `p`: term `0` is sigma
///   of `a_il` evaluated into a block of `a`'s layout, which leaves it at `a.k - p`, then narrowed exactly to `w` by
///   poulpy-ckks's `ckks_copy`; term `k` is phi^k of that sigma into a block of width `w`, through the `Direct` shift
///   maps the plan records. On the right, tau and psi^k, or, for a replicated `b` ([`crate::layouts::replicated`], read
///   from its label), [`BlasContext::tau_rep`] and [`BlasContext::rotate_rows_rep`], all at the width `b.k - p`. Each
///   term is converted once, by poulpy-ckks's `ckks_prepare_left_into` (`ckks_prepare_right_into`) in a preparation
///   and `ckks_prepare_left` (`ckks_prepare_right`) in a product, on its `ceil(w / base2k)` limbs;
/// - one JKLS kernel per block of `res`, [`crate::oep::BlasJklsImpl::blas_jkls_block_impl`], over its `K d` pairs of
///   terms `(a_il[k], b_lj[k])`, `l`-major: ascending `l`, then ascending `k` within each `l`, the order the one-pass
///   kernel sums them in, part of the bits on FFT; the kernel's `# Safety` fixes its circuit;
/// - the bits, whatever the hints: the prepared product equals the reference's on the reference's operands of the same
///   sources and hints under the same contexts, so an operand prepared under the product's `ctx` gives the same
///   products under either hint;
/// - the key widths of the Keys table of `poulpy-blas/docs/blas-contracts.md`;
/// - the maps of a left product by public matrices, read once per call from the input's label: when `a` is replicated
///   ([`crate::layouts::replicated`]) and `b` holds them ([`crate::layouts::BlasPtOperand::block_replicated`]), the
///   blocks' maps are those, `d` diagonals scheduled at `left_mul`'s giant step; otherwise
///   [`crate::layouts::BlasPtOperand::block`]'s, `2d - 1`. The two choices differ in bits. The validator's key checks,
///   [`blas_matmul_pt_left_max_base2k`](crate::reference::blas_matmul_pt_left_max_base2k) and the key widths follow the
///   same choice; the scratch query does not read the block maps;
/// - the errors and their order: the validator's, then poulpy-ckks's budget errors, before any write to `res`: the left
///   terms', then the right terms', then the first block's ct x ct rule. A hot side's come from the product, so a hot
///   preparation of an input short of its budget returns `Ok`;
/// - the products by public matrices share their giant steps across the inner index: block `res_ij` is poulpy-ckks's
///   `ckks_eval_linear_transformation_sum_into` of its `K` terms in ascending `l`, `(s_il, b_lj)` on the right and
///   `(s_lj, b_il)` on the left, where `s` is `a` under `Diagonal`, and under `Jkls` the blocks of `a` through the
///   operand's first map, one map-hook call ([`crate::oep::BlasMatIntraImpl::blas_map_blocks_impl`]) per block row of
///   `a` on the right and per block column on the left, into heap blocks of `a`'s layout at the width that map leaves.
///   Each giant step sums the products of every term in the transform domain, in that order, before its one rotation;
///   an FFT backend rounds that sum, so the order is part of the bits. Each block of `res` is then stamped with the
///   label `min(a.log_sparsity, log2(S / (count d^2)))`, for `S` the slots of a block and `count` the operand's
///   ([`crate::layouts::BlasPtOperandInfos::count`]).
///
/// **Free (bit-identical):** the order of the blocks; which side the product holds and which it streams; materializing
/// or streaming the terms; caching them or their convolution forms; sharing baby-step caches; batching key switches.
/// Hoisting the shifts changes bits and the key widths, so it would arrive as a plan variant.
///
/// A method first runs its validator, [`crate::layouts::validate_blas_matmul`],
/// [`crate::layouts::validate_blas_matmul_prepare_left`], [`crate::layouts::validate_blas_matmul_prepare_right`],
/// [`crate::layouts::validate_blas_matmul_prepared`], [`crate::layouts::validate_blas_matmul_pt_right`] or
/// [`crate::layouts::validate_blas_matmul_pt_left`], before any write, and returns its errors with `res`, its batch
/// record and the operands untouched. On inputs it accepted, it returns `Ok`, after a product stamps `res` with the
/// batch the validator returns ([`crate::layouts::BlasGridMut::set_mat_slots`]) and a preparation records its source's
/// batch on its operand, or poulpy-ckks's budget error before any write to `res`. Any other `Err` is a backend fault (a
/// validator gap or a device failure), and `res` is then unspecified.
pub unsafe trait BlasMatMulImpl: Backend + CKKSMulImpl {
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

    /// Fills `res`, allocated from the infos its validator checks, from `a` under `res`'s hint.
    fn blas_matmul_prepare_left_impl<A, K>(
        module: &Module<Self>,
        res: &mut BlasMatMulLeft<Self>,
        a: &A,
        ctx: &BlasContext<Self>,
        atk: &K,
        scratch: &mut ScratchArena<'_, Self>,
    ) -> Result<()>
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

    /// Fills `res`, allocated from the infos its validator checks, from `b` under `res`'s hint.
    fn blas_matmul_prepare_right_impl<B, K>(
        module: &Module<Self>,
        res: &mut BlasMatMulRight<Self>,
        b: &B,
        ctx: &BlasContext<Self>,
        atk: &K,
        scratch: &mut ScratchArena<'_, Self>,
    ) -> Result<()>
    where
        B: BlasGrid<Self>,
        K: GetAutomorphismKey<Self> + Sync;

    /// Scratch of [`Self::blas_matmul_prepared_impl`] under `ctx`; `a` and `b` are the operands or the layouts
    /// [`blas_matmul_left_operand_layout`](crate::layouts::blas_matmul_left_operand_layout) and
    /// [`blas_matmul_right_operand_layout`](crate::layouts::blas_matmul_right_operand_layout) give.
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
        a: &BlasMatMulLeft<Self>,
        b: &BlasMatMulRight<Self>,
        ctx: &BlasContext<Self>,
        tsk: &T,
        atk: &K,
        scratch: &mut ScratchArena<'_, Self>,
    ) -> Result<()>
    where
        R: BlasGridMut<Self>,
        T: GetTensorKey<Self> + Sync,
        K: GetAutomorphismKey<Self> + Sync;

    fn blas_matmul_pt_right_tmp_bytes_impl<R, A, K>(
        module: &Module<Self>,
        res: &R,
        a: &A,
        b: &BlasPtOperand<Self>,
        atk: &K,
    ) -> usize
    where
        R: BlasGridInfos,
        A: BlasGridInfos,
        K: GGLWEInfos;

    fn blas_matmul_pt_right_impl<R, A, K>(
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
        derived::blas_matmul_tmp_bytes_derived(module, res, a, b, ctx, tsk, atk)
    }

    /// [`Self::blas_matmul_prepare_left_impl`] on `a` and [`Self::blas_matmul_prepare_right_impl`] on `b`, into hot,
    /// `OneShot`, operands allocated from the infos
    /// [`blas_matmul_left_operand_layout`](crate::layouts::blas_matmul_left_operand_layout) and
    /// [`blas_matmul_right_operand_layout`](crate::layouts::blas_matmul_right_operand_layout) give, then
    /// [`Self::blas_matmul_prepared_impl`] into `res` under `ctx`. [`crate::impl_blas_matmul_reference!`] overrides it with
    /// [`BlasMatMulReference::blas_matmul_reference`](crate::reference::BlasMatMulReference::blas_matmul_reference), the
    /// same product on the borrowed grids, without the copies.
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
        derived::blas_matmul_derived(module, res, a, b, ctx, tsk, atk, scratch)
    }
}

/// Selects [`BlasMatMulReference`](crate::reference::BlasMatMulReference) for `$be`, and its `blas_matmul`, which
/// streams the borrowed grids without the hot preparations' copies, under the provided query. The reference prepares
/// cold operands, and the terms of hot ones inside the product, through `$be`'s map hook, converts them through its
/// poulpy-ckks preparations and multiplies them through its JKLS kernel, so `$be` also implements
/// [`crate::oep::BlasMatIntraImpl`] and [`crate::oep::BlasJklsImpl`].
#[macro_export]
macro_rules! impl_blas_matmul_reference {
    ($be:ty) => {
        unsafe impl $crate::oep::BlasMatMulImpl for $be {
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
                res: &mut $crate::layouts::BlasMatMulLeft<Self>,
                a: &A,
                ctx: &$crate::layouts::BlasContext<Self>,
                atk: &K,
                scratch: &mut ::poulpy_hal::layouts::ScratchArena<'_, Self>,
            ) -> ::poulpy_ckks::CKKSResult<()>
            where
                A: $crate::layouts::BlasGrid<Self>,
                K: ::poulpy_core::layouts::GetAutomorphismKey<Self> + Sync,
            {
                $crate::reference::BlasMatMulReference::blas_matmul_prepare_left_reference(module, res, a, ctx, atk, scratch)
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
                res: &mut $crate::layouts::BlasMatMulRight<Self>,
                b: &B,
                ctx: &$crate::layouts::BlasContext<Self>,
                atk: &K,
                scratch: &mut ::poulpy_hal::layouts::ScratchArena<'_, Self>,
            ) -> ::poulpy_ckks::CKKSResult<()>
            where
                B: $crate::layouts::BlasGrid<Self>,
                K: ::poulpy_core::layouts::GetAutomorphismKey<Self> + Sync,
            {
                $crate::reference::BlasMatMulReference::blas_matmul_prepare_right_reference(module, res, b, ctx, atk, scratch)
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
                a: &$crate::layouts::BlasMatMulLeft<Self>,
                b: &$crate::layouts::BlasMatMulRight<Self>,
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

            fn blas_matmul_impl<R, A, B, T, K>(
                module: &::poulpy_hal::layouts::Module<Self>,
                res: &mut R,
                a: &A,
                b: &B,
                ctx: &$crate::layouts::BlasContext<Self>,
                tsk: &T,
                atk: &K,
                scratch: &mut ::poulpy_hal::layouts::ScratchArena<'_, Self>,
            ) -> ::poulpy_ckks::CKKSResult<()>
            where
                R: $crate::layouts::BlasGridMut<Self>,
                A: $crate::layouts::BlasGrid<Self>,
                B: $crate::layouts::BlasGrid<Self>,
                T: ::poulpy_core::layouts::GetTensorKey<Self> + Sync,
                K: ::poulpy_core::layouts::GetAutomorphismKey<Self> + Sync,
            {
                $crate::reference::BlasMatMulReference::blas_matmul_reference(module, res, a, b, ctx, tsk, atk, scratch)
            }

            fn blas_matmul_pt_right_tmp_bytes_impl<R, A, K>(
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
                $crate::reference::BlasMatMulReference::blas_matmul_pt_right_tmp_bytes_reference(module, res, a, b, atk)
            }

            fn blas_matmul_pt_right_impl<R, A, K>(
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
                $crate::reference::BlasMatMulReference::blas_matmul_pt_right_reference(module, res, a, b, atk, scratch)
            }

            fn blas_matmul_pt_left_tmp_bytes_impl<R, A, K>(
                module: &::poulpy_hal::layouts::Module<Self>,
                res: &R,
                b: &$crate::layouts::BlasPtOperand<Self, $crate::layouts::Left>,
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
                b: &$crate::layouts::BlasPtOperand<Self, $crate::layouts::Left>,
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
