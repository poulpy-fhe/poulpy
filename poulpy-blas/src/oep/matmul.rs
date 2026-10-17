use poulpy_ckks::CKKSResult as Result;
use poulpy_core::layouts::{GGLWEInfos, GetAutomorphismKey, GetTensorKey};
use poulpy_hal::layouts::{Backend, Module, PrepareHint, ScratchArena};

use super::derived;
use crate::layouts::{
    BlasContext, BlasGrid, BlasGridInfos, BlasGridMut, BlasMatMulLeft, BlasMatMulOperandInfos, BlasMatMulRight, BlasPtOperand,
    Left,
};

/// Extension point of [`crate::api::BlasMatMulOps`]: the prepared operands, the preparations, the prepared product and
/// the products by public matrices, on the right and on the left, are required, and `blas_matmul` is provided over the
/// preparations and the prepared product.
///
/// # Safety
///
/// Every required method reproduces its `_reference` counterpart in `reference::BlasMatMulReference` bit for
/// bit, digits and CKKS metadata, within the scratch the matching `_tmp_bytes_impl` reports; with less, it panics
/// before any write to `res` when it looks up a key, as the reference does.
///
/// **Prepared operands.** A backend does not choose their type: [`BlasMatMulLeft`] and [`BlasMatMulRight`] are
/// [`crate::layouts::BlasMatMulOperand`]s of poulpy-ckks ciphertexts and prepared terms, whose storage is the
/// backend's. That storage is crate-private, so a backend outside poulpy-blas forwards the allocations, the
/// preparations and the prepared product to `BlasMatMulReference`. The allocations,
/// [`Self::blas_matmul_left_alloc_from_infos_impl`] and [`Self::blas_matmul_right_alloc_from_infos_impl`], make one of
/// [`BlasMatMulOperandInfos`] holding all its storage, which reports those infos and the batch `0`; the
/// `_bytes_of_from_infos_impl` queries give the bytes it holds; the preparations fill it, with no allocation of its
/// storage, from a source whose infos have its shape (the validator checks the hint, the dimensions and the GLWE
/// layouts, not the metadata), under the [`PrepareHint`] it was allocated under: hot, `OneShot`, for one product, or
/// cold, `Reuse`, for many, and return poulpy-ckks's budget error before any write to it, which a cold preparation
/// evaluates through heap working blocks. An operand reports the infos of its latest preparation: prepared from `a`, it
/// reports its hint, `a`'s block layout as its source, `a`'s dimensions, `Some` of `a`'s batch,
/// [`grid_batch::<Self::Ring>(a)`](crate::layouts::grid_batch), which reads the batch of an input without a record from
/// its label, and as its block layout `a`'s when hot, the layout `reference::blas_matmul_left_term_layout`
/// (`reference::blas_matmul_right_term_layout` on the right) gives under the preparation's `ctx` when cold:
/// `reference::blas_matmul_left_operand_layout` and `reference::blas_matmul_right_operand_layout` give
/// these infos before preparing. The `_prepared` validator, the batch the prepared product records and the provided
/// `blas_matmul` query read nothing else. The terms of a cold operand are those of the preparation's `ctx`; those of a
/// hot operand are those of the `ctx` of the product it enters, whatever the context it was prepared under, at whatever
/// scale: the product evaluates its maps, which spend their scale from the operand's budget, and stamps by the ct x ct
/// rule. So a hot operand holds a copy of its source's live limbs. The prepared product of two operands, of any hints,
/// equals the reference's on the reference's operands of the same sources and hints under the same contexts, and so
/// `blas_matmul` of the sources when every side is hot or prepared under the product's `ctx`. A cold preparation looks
/// up the rotation keys of its terms and a hot one none, after its validator checked them under either hint; the
/// prepared product looks up those of each hot side's terms under its `ctx`, and the tensor key.
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
/// - the key widths: sigma's (tau's) baby steps at `a.k` and giant steps at `a.k - p`, the shifts' giant steps at their
///   side's term width, the tensor key at the wider of both sides' term widths; a public operand's maps, for its scale
///   `p_B`: under `BlasPtAlgorithm::Diagonal` its blocks' baby steps at `a.k` and giant steps at
///   `min(res.k, a.k - p_B)`; under `Jkls` its first map's ([`crate::layouts::BlasPtOperand::first`]) baby steps at
///   `a.k` and giant steps at `a.k - p_B`, then its blocks' baby steps at `a.k - p_B` and giant steps at
///   `min(res.k, a.k - 2 p_B)`;
/// - the maps of a left product by public matrices, read once per call from the input's label: when `a` is replicated
///   ([`crate::layouts::replicated`]) and `b` holds them ([`crate::layouts::BlasPtOperand::block_replicated`]), the
///   blocks' maps are those, `d` diagonals scheduled at `left_mul`'s giant step; otherwise
///   [`crate::layouts::BlasPtOperand::block`]'s, `2d - 1`. The two choices differ in bits. The validator's key checks,
///   `reference::blas_matmul_pt_left_max_base2k` and the key widths above follow the same choice; the scratch
///   query does not read the block maps;
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
/// batch on its operand, or poulpy-ckks's budget error before any write to `res`. Any other `Err` is a backend fault
/// (a validator gap or a device failure), and `res` is then unspecified.
pub unsafe trait BlasMatMulImpl: Backend {
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

    /// Bytes a left operand of `infos` holds, as [`Self::blas_matmul_left_alloc_from_infos_impl`] allocates it.
    fn blas_matmul_left_bytes_of_from_infos_impl<A>(module: &Module<Self>, infos: &A) -> usize
    where
        A: BlasMatMulOperandInfos;

    /// A left operand of `infos`, which [`Self::blas_matmul_prepare_left_impl`] fills.
    fn blas_matmul_left_alloc_from_infos_impl<A>(module: &Module<Self>, infos: &A) -> BlasMatMulLeft<Self>
    where
        A: BlasMatMulOperandInfos;

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

    /// Bytes a right operand of `infos` holds, as [`Self::blas_matmul_right_alloc_from_infos_impl`] allocates it.
    fn blas_matmul_right_bytes_of_from_infos_impl<A>(module: &Module<Self>, infos: &A) -> usize
    where
        A: BlasMatMulOperandInfos;

    /// A right operand of `infos`, which [`Self::blas_matmul_prepare_right_impl`] fills.
    fn blas_matmul_right_alloc_from_infos_impl<A>(module: &Module<Self>, infos: &A) -> BlasMatMulRight<Self>
    where
        A: BlasMatMulOperandInfos;

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
    /// `reference::blas_matmul_left_operand_layout` and `reference::blas_matmul_right_operand_layout`
    /// give.
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
        derived::blas_matmul_tmp_bytes(module, res, a, b, ctx, tsk, atk)
    }

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
