use poulpy_ckks::{
    CKKSCtBounds, CKKSLayout, CKKSResult as Result, SetCKKSInfos,
    api::{
        CKKSCopyOps, CKKSDotProductOps, CKKSLinearTransformationOps, CKKSModuleInfos, CKKSMulOps, LinearTransformationBabySteps,
        LinearTransformationPrepared,
    },
    layouts::{CKKSCiphertextOwned, CKKSModuleAlloc, CKKSPreparedLeft, CKKSPreparedRight},
};
use poulpy_core::{
    GLWEMulPreparedPlainSum,
    layouts::{GGLWEInfos, GGLWELayout, GLWEToBackendMut, GLWEToBackendRef, GetAutomorphismKey, GetTensorKey, LWEInfos},
};
use poulpy_hal::{
    api::{CnvPVecAlloc, ModuleN},
    layouts::{Backend, Module, PrepareHint, ScratchArena},
};

use super::lt_tmp_bytes;
use crate::{
    layouts::{
        BlasContext, BlasGrid, BlasGridInfos, BlasGridMut, BlasMatMulOperand, BlasMatMulOperandInfos, BlasMatMulOperandLayout,
        BlasPlan, BlasPtAlgorithm, BlasPtOperand, BlasPtOperandInfos, Left,
        ciphertext::layout_of,
        encoding::mapped_block,
        grid_batch,
        matmul_operand::{
            left_operand_layout, left_term_layout, product_terms, right_maps, right_operand_layout, right_term_layout,
        },
        planning::pt_label,
        pt_operand::masked_rotations,
        replicated,
        validate::{
            KeyLayouts, assert_scratch, context_masks, ensure_rotation, ensure_tensor_key, left_takes_replicated,
            prepared_tensor, prepared_uses, pt_left_uses, pt_right_uses, term_uses, validate_blas_matmul,
            validate_blas_matmul_prepare_left, validate_blas_matmul_prepare_right, validate_blas_matmul_prepared,
            validate_blas_matmul_pt_left, validate_blas_matmul_pt_right,
        },
    },
    oep::{BlasJklsImpl, BlasMatIntraImpl},
};

/// Fails as [`ensure_rotation`] unless `keys` holds every key of [`term_uses`].
fn ensure_term_keys<'c, BE: Backend + 'c>(
    op: &'static str,
    module: &(impl CKKSModuleInfos + ?Sized),
    a: &impl BlasGridInfos,
    first: &LinearTransformationPrepared<BE>,
    shift: impl Fn(usize) -> &'c LinearTransformationPrepared<BE>,
    keys: &impl GetAutomorphismKey<BE>,
) -> Result<()> {
    term_uses(a, first, shift, &mut |u| ensure_rotation(op, module, keys, u))
}

/// Reference compositions of [`crate::api::BlasMatMulOps`]. Each operation runs its validator, the matching
/// `validate_*` of [`crate::layouts`], before any write, so its errors leave `res`, its batch record and the operands
/// untouched, then asserts its scratch and returns poulpy-ckks's budget errors before any write to `res`; on `Ok`, a
/// product stamps `res` with the batch the validator returns, and a preparation records its source's batch on its
/// operand. A hot preparation copies its input; a cold one makes its terms through the map hook of [`BlasMatIntraImpl`],
/// block row after block row, each converted once by poulpy-ckks's preparations. The prepared product, and
/// `blas_matmul` on the borrowed grids, make a hot side's terms the same way under their context, a block row of `a` or
/// a block column of `b` at a time, holding, of two hot sides, the smaller one's terms and streaming the other's, and
/// read the terms one output block at a time through the JKLS kernel of [`BlasJklsImpl`]. The products by public
/// matrices evaluate one poulpy-ckks summed linear transformation per output block, under JKLS of the results of the
/// operand's first map, made through the map hook a block row (column) of `a` at a time; under `JklsStored` and
/// `JklsGalois`, one poulpy-ckks summed product by prepared plaintexts per output block, of terms made as a hot side's.
/// They run block after block on one scratch, and the first error stops them.
pub trait BlasMatMulReference<BE: Backend> {
    /// Under `OneShot`, the copy of a block of `a` into a block of its layout. Under `Reuse`, the largest of the map
    /// hook's queries through sigma, from a block row of `a` into blocks of its layout at the width sigma leaves, and
    /// through every shift, from those into blocks of the terms' layout, of the copy of one into a block of the terms'
    /// layout, and of the conversion of a term.
    fn blas_matmul_prepare_left_tmp_bytes_reference<A, K>(
        &self,
        a: &A,
        ctx: &BlasContext<BE>,
        keys: &K,
        hint: PrepareHint,
    ) -> usize
    where
        BE: BlasMatIntraImpl,
        Self: CKKSMulOps<BE> + CKKSCopyOps<BE>,
        A: BlasGridInfos,
        K: GGLWEInfos;

    /// Fills `res` from `a` under `res`'s hint, with no allocation of `res`'s storage. Under `OneShot`, `ckks_copy` of
    /// every block of `a` into the block of its layout `res` holds, its live limbs, metadata, canonical flag and noise
    /// tag. Under `Reuse`, the terms of every block of `a`, row-major, block row after block row: one hook call
    /// evaluates `S = sigma(a_il)` for every block of the row into a heap block of `a`'s layout at the width sigma
    /// leaves; one hook call evaluates `phi^k(S)` for every `S` and every `k` in `1..d` into heap blocks of the terms'
    /// layout, one baby-step cache per `S`; then `ckks_copy` narrows each `S` into a block of the terms' layout, its
    /// term `0`, exactly: it charges the shift to the budget. Each term is converted once by `ckks_prepare_left_into`
    /// into its place in `res`, and has the layout
    /// [`blas_matmul_left_term_layout`](crate::layouts::blas_matmul_left_term_layout) gives. `res` then reports the
    /// infos [`blas_matmul_left_operand_layout`](crate::layouts::blas_matmul_left_operand_layout) gives for `a` under
    /// its hint and `a`'s batch, [`grid_batch`].
    fn blas_matmul_prepare_left_reference<A, K>(
        &self,
        res: &mut BlasMatMulOperand<BE, CKKSPreparedLeft<BE>>,
        a: &A,
        ctx: &BlasContext<BE>,
        keys: &K,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        BE: BlasMatIntraImpl,
        Self: CKKSMulOps<BE> + CKKSCopyOps<BE>,
        A: BlasGrid<BE>,
        K: GetAutomorphismKey<BE> + Sync;

    /// As [`Self::blas_matmul_prepare_left_tmp_bytes_reference`], with the right maps
    /// [`blas_matmul_right_term_layout`](crate::layouts::blas_matmul_right_term_layout) reads.
    fn blas_matmul_prepare_right_tmp_bytes_reference<B, K>(
        &self,
        b: &B,
        ctx: &BlasContext<BE>,
        keys: &K,
        hint: PrepareHint,
    ) -> usize
    where
        BE: BlasMatIntraImpl,
        Self: CKKSMulOps<BE> + CKKSCopyOps<BE>,
        B: BlasGridInfos,
        K: GGLWEInfos;

    /// As [`Self::blas_matmul_prepare_left_reference`], with the right maps
    /// [`blas_matmul_right_term_layout`](crate::layouts::blas_matmul_right_term_layout) reads, converted by
    /// `ckks_prepare_right_into`. For a replicated `b`, whose shifts spend no scale, the copy keeps `S`'s width.
    fn blas_matmul_prepare_right_reference<B, K>(
        &self,
        res: &mut BlasMatMulOperand<BE, CKKSPreparedRight<BE>>,
        b: &B,
        ctx: &BlasContext<BE>,
        keys: &K,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        BE: BlasMatIntraImpl,
        Self: CKKSMulOps<BE> + CKKSCopyOps<BE>,
        B: BlasGrid<BE>,
        K: GetAutomorphismKey<BE> + Sync;

    /// The largest of the JKLS kernel's query for a block of `res` and the `K d` pairs of terms of the operands `a` and
    /// `b`, or of the layouts [`blas_matmul_left_operand_layout`](crate::layouts::blas_matmul_left_operand_layout) and
    /// [`blas_matmul_right_operand_layout`](crate::layouts::blas_matmul_right_operand_layout) give, a hot side's terms at
    /// the layout [`blas_matmul_left_term_layout`](crate::layouts::blas_matmul_left_term_layout)
    /// ([`blas_matmul_right_term_layout`](crate::layouts::blas_matmul_right_term_layout)) gives under `ctx`, and for each
    /// hot side, the query of the terms of one of its units, `K` blocks of its source: a block row of `a`, a block column
    /// of `b`.
    #[allow(clippy::too_many_arguments)]
    fn blas_matmul_prepared_tmp_bytes_reference<R, A, B, T, K>(
        &self,
        res: &R,
        a: &A,
        b: &B,
        ctx: &BlasContext<BE>,
        tsk: &T,
        keys: &K,
    ) -> usize
    where
        BE: BlasJklsImpl + BlasMatIntraImpl,
        Self: CKKSMulOps<BE> + CKKSCopyOps<BE>,
        R: BlasGridInfos,
        A: BlasMatMulOperandInfos,
        B: BlasMatMulOperandInfos,
        T: GGLWEInfos,
        K: GGLWEInfos;

    /// For every block `(i, j)` of `res`, one JKLS kernel, [`BlasJklsImpl::blas_jkls_block`], over the `K d` pairs
    /// `(a_il[k], b_lj[k])`, `l`-major: ascending `l`, then ascending `k` within each `l`. A cold side brings its terms.
    /// A hot side's are made under `ctx` as the cold preparation of its source makes them, but converted under
    /// `OneShot`, one unit at a time: a block row of `a` or a block column of `b`, `K` blocks, through two hook calls,
    /// sigma (tau) over the unit, then every shift of every result. Each unit is made once and every term converted
    /// once:
    /// - one hot side streams: each of its units serves its row (column) of `res`, then is dropped, the other side's
    ///   terms read in place;
    /// - two hot sides: one is held and the other streams. For `res` of `m x n` blocks and terms of `La` and `Lb` limbs,
    ///   `a` is held when `(m - 1) La <= (n - 1) Lb`, the fewer limbs for the held terms plus one unit of the other
    ///   side; `b` otherwise, after `a`'s first unit.
    ///
    /// So beside `res` and the operands the heap holds the held terms, if any, in the backend's transform-domain form,
    /// and one unit with its sigma (tau) results; the blocks of `res` are written row by row, or column by column when
    /// `b` streams. poulpy-ckks's budget errors come before any write to `res`: the left terms', then the right terms',
    /// then the first block's ct x ct rule.
    #[allow(clippy::too_many_arguments)]
    fn blas_matmul_prepared_reference<R, T, K>(
        &self,
        res: &mut R,
        a: &BlasMatMulOperand<BE, CKKSPreparedLeft<BE>>,
        b: &BlasMatMulOperand<BE, CKKSPreparedRight<BE>>,
        ctx: &BlasContext<BE>,
        tsk: &T,
        keys: &K,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        BE: BlasJklsImpl + BlasMatIntraImpl,
        Self: CKKSMulOps<BE> + CKKSCopyOps<BE>,
        R: BlasGridMut<BE>,
        T: GetTensorKey<BE> + Sync,
        K: GetAutomorphismKey<BE> + Sync;

    /// The larger of both reference hot preparations' queries and the reference prepared product's on the layouts they
    /// give.
    fn blas_matmul_tmp_bytes_reference<R, A, B, T, K>(
        &self,
        res: &R,
        a: &A,
        b: &B,
        ctx: &BlasContext<BE>,
        tsk: &T,
        keys: &K,
    ) -> usize
    where
        BE: BlasJklsImpl + BlasMatIntraImpl,
        Self: CKKSMulOps<BE> + CKKSCopyOps<BE>,
        R: BlasGridInfos,
        A: BlasGridInfos,
        B: BlasGridInfos,
        T: GGLWEInfos,
        K: GGLWEInfos;

    /// `res = a b`: [`Self::blas_matmul_prepared_reference`] of two hot operands of `a` and `b`, bit for bit, reading
    /// the blocks of `a` and `b` in place, without the copies of the hot preparations. It asserts
    /// [`Self::blas_matmul_tmp_bytes_reference`] over the keys it looks up.
    #[allow(clippy::too_many_arguments)]
    fn blas_matmul_reference<R, A, B, T, K>(
        &self,
        res: &mut R,
        a: &A,
        b: &B,
        ctx: &BlasContext<BE>,
        tsk: &T,
        keys: &K,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        BE: BlasJklsImpl + BlasMatIntraImpl,
        Self: CKKSMulOps<BE> + CKKSCopyOps<BE>,
        R: BlasGridMut<BE>,
        A: BlasGrid<BE>,
        B: BlasGrid<BE>,
        T: GetTensorKey<BE> + Sync,
        K: GetAutomorphismKey<BE> + Sync;

    /// The baby steps of one block and one summed linear transformation from blocks of `a`'s layout into a block of
    /// `res`'s; under JKLS, from blocks of the layout `b`'s first map leaves, and the map hook's query of that map from
    /// a block row of `a` into such blocks. Under `JklsStored` and `JklsGalois`, in place of the baby steps and the
    /// summed linear transformation, the map hook's query of `ctx`'s masks, phi^k, from those blocks into blocks of the
    /// terms' layout, the copy of one into a block of the terms' layout, the conversion of a term, and poulpy-ckks's
    /// `ckks_dot_product_prepared_pt` query of a block of `res` by terms of that layout and `b`'s plaintexts. `ctx` is
    /// read only then.
    fn blas_matmul_pt_right_tmp_bytes_reference<R, A, K>(
        &self,
        res: &R,
        a: &A,
        b: &BlasPtOperand<BE>,
        ctx: &BlasContext<BE>,
        keys: &K,
    ) -> usize
    where
        BE: BlasMatIntraImpl + BlasJklsImpl,
        Self: CKKSLinearTransformationOps<BE>
            + CKKSMulOps<BE>
            + CKKSCopyOps<BE>
            + CKKSDotProductOps<BE>
            + GLWEMulPreparedPlainSum<BE>,
        R: BlasGridInfos,
        A: BlasGridInfos,
        K: GGLWEInfos;

    /// For every block row `i` of `res`, in order: under JKLS, one map-hook call first evaluates `b`'s first map,
    /// sigma, of every block `a_il` into a heap block `s_il` of the layout it leaves, labelled as `a_il`; then one
    /// baby-step cache per block `a_il` (`s_il` under JKLS), `l < K`, covering the baby steps of every block map of
    /// `b`, then every block `res_ij` is poulpy-ckks's summed linear transformation of the `K` terms `(a_il, b_lj)`
    /// (`(s_il, b_lj)`), in ascending `l`, which shares the giant steps across them, stamped with the label
    /// `min(a.log_sparsity, log2(S / (count d^2)))`, for `S` the slots of a block and `count` that of `b`.
    ///
    /// Under `JklsStored` and `JklsGalois`, per block row of `a`, a unit: one masked-unit hook,
    /// [`BlasJklsImpl::blas_jkls_pt_unit`], through sigma and `ctx`'s phi^k, `1 <= k < d`, by `b`'s prepared plaintexts
    /// of the blocks `(l, j)`: under `JklsStored` term `k`'s own through `1`, under `JklsGalois` the block's one through
    /// the Galois element of the rotation by `k d`; every block it writes then stamped with the label above. The Scratch
    /// section of `docs/blas-contracts.md` bounds its heap.
    fn blas_matmul_pt_right_reference<R, A, K>(
        &self,
        res: &mut R,
        a: &A,
        b: &BlasPtOperand<BE>,
        ctx: &BlasContext<BE>,
        keys: &K,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        BE: BlasMatIntraImpl + BlasJklsImpl,
        Self: CKKSLinearTransformationOps<BE>
            + CKKSMulOps<BE>
            + CKKSCopyOps<BE>
            + CKKSDotProductOps<BE>
            + GLWEMulPreparedPlainSum<BE>
            + ModuleN
            + CnvPVecAlloc<BE>
            + Sized,
        R: BlasGridMut<BE>,
        A: BlasGrid<BE>,
        K: GetAutomorphismKey<BE> + Sync;

    /// As [`Self::blas_matmul_pt_right_tmp_bytes_reference`], under JKLS from a block column of `a`, and under
    /// `JklsStored` and `JklsGalois` through `ctx`'s column halves of psi^k, `rows_left(k)` then `rows_right(k)`.
    ///
    /// # Panics
    ///
    /// Under those paths, for a `ctx` without those masks, whose plan selects neither.
    fn blas_matmul_pt_left_tmp_bytes_reference<R, A, K>(
        &self,
        res: &R,
        b: &BlasPtOperand<BE, Left>,
        a: &A,
        ctx: &BlasContext<BE>,
        keys: &K,
    ) -> usize
    where
        BE: BlasMatIntraImpl + BlasJklsImpl,
        Self: CKKSLinearTransformationOps<BE>
            + CKKSMulOps<BE>
            + CKKSCopyOps<BE>
            + CKKSDotProductOps<BE>
            + GLWEMulPreparedPlainSum<BE>,
        R: BlasGridInfos,
        A: BlasGridInfos,
        K: GGLWEInfos;

    /// `res = b a`. For every block column `j` of `res`, in order: under JKLS, one map-hook call first evaluates `b`'s
    /// first map, tau, of every block `a_lj` into a heap block `s_lj` of the layout it leaves, labelled as `a_lj`; then
    /// one baby-step cache per block `a_lj` (`s_lj` under JKLS), `l < K`, covering the baby steps of every block map of
    /// `b`, then every block `res_ij` is poulpy-ckks's summed linear transformation of the `K` terms `(a_lj, b_il)`
    /// (`(s_lj, b_il)`), in ascending `l`, which shares the giant steps across them, stamped with the label of
    /// [`Self::blas_matmul_pt_right_reference`].
    ///
    /// Under `JklsStored` and `JklsGalois`, as there, per block column of `a`, one masked-unit hook through tau then
    /// `ctx`'s `rows_left(k)` then `rows_right(k)` for `1 <= k < d`, by `b`'s prepared plaintexts of the blocks `(i, l)`,
    /// under `JklsGalois` through the Galois elements of the rotations by `0`, then `k` and `k - d`.
    fn blas_matmul_pt_left_reference<R, A, K>(
        &self,
        res: &mut R,
        b: &BlasPtOperand<BE, Left>,
        a: &A,
        ctx: &BlasContext<BE>,
        keys: &K,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        BE: BlasMatIntraImpl + BlasJklsImpl,
        Self: CKKSLinearTransformationOps<BE>
            + CKKSMulOps<BE>
            + CKKSCopyOps<BE>
            + CKKSDotProductOps<BE>
            + GLWEMulPreparedPlainSum<BE>
            + ModuleN
            + CnvPVecAlloc<BE>
            + Sized,
        R: BlasGridMut<BE>,
        A: BlasGrid<BE>,
        K: GetAutomorphismKey<BE> + Sync;
}

impl<BE: Backend> BlasMatMulReference<BE> for Module<BE> {
    fn blas_matmul_prepare_left_tmp_bytes_reference<A, K>(
        &self,
        a: &A,
        ctx: &BlasContext<BE>,
        keys: &K,
        hint: PrepareHint,
    ) -> usize
    where
        BE: BlasMatIntraImpl,
        Self: CKKSMulOps<BE> + CKKSCopyOps<BE>,
        A: BlasGridInfos,
        K: GGLWEInfos,
    {
        prepare_tmp_bytes(self, a, ctx.sigma(), |k| ctx.rotate_cols(k), keys, hint)
    }

    fn blas_matmul_prepare_left_reference<A, K>(
        &self,
        res: &mut BlasMatMulOperand<BE, CKKSPreparedLeft<BE>>,
        a: &A,
        ctx: &BlasContext<BE>,
        keys: &K,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        BE: BlasMatIntraImpl,
        Self: CKKSMulOps<BE> + CKKSCopyOps<BE>,
        A: BlasGrid<BE>,
        K: GetAutomorphismKey<BE> + Sync,
    {
        validate_blas_matmul_prepare_left(self, &*res, a, ctx, keys)?;
        let convert = |res: &mut CKKSPreparedLeft<BE>, term: &CKKSCiphertextOwned<BE>, s: &mut ScratchArena<'_, BE>| {
            self.ckks_prepare_left_into(res, term, s)
        };
        let (first, shift) = (ctx.sigma(), |k| ctx.rotate_cols(k));
        let layout = left_operand_layout(a, &ctx.maps, res.hint());
        prepare(
            self,
            "blas_matmul_prepare_left",
            res,
            layout,
            a,
            first,
            shift,
            keys,
            convert,
            scratch,
        )
    }

    fn blas_matmul_prepare_right_tmp_bytes_reference<B, K>(
        &self,
        b: &B,
        ctx: &BlasContext<BE>,
        keys: &K,
        hint: PrepareHint,
    ) -> usize
    where
        BE: BlasMatIntraImpl,
        Self: CKKSMulOps<BE> + CKKSCopyOps<BE>,
        B: BlasGridInfos,
        K: GGLWEInfos,
    {
        let (first, shift) = right_maps(&ctx.maps, replicated::<BE::Ring>(b));
        prepare_tmp_bytes(self, b, first, shift, keys, hint)
    }

    fn blas_matmul_prepare_right_reference<B, K>(
        &self,
        res: &mut BlasMatMulOperand<BE, CKKSPreparedRight<BE>>,
        b: &B,
        ctx: &BlasContext<BE>,
        keys: &K,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        BE: BlasMatIntraImpl,
        Self: CKKSMulOps<BE> + CKKSCopyOps<BE>,
        B: BlasGrid<BE>,
        K: GetAutomorphismKey<BE> + Sync,
    {
        validate_blas_matmul_prepare_right(self, &*res, b, ctx, keys)?;
        let convert = |res: &mut CKKSPreparedRight<BE>, term: &CKKSCiphertextOwned<BE>, s: &mut ScratchArena<'_, BE>| {
            self.ckks_prepare_right_into(res, term, s)
        };
        let (first, shift) = right_maps(&ctx.maps, replicated::<BE::Ring>(b));
        let layout = right_operand_layout(b, &ctx.maps, res.hint());
        prepare(
            self,
            "blas_matmul_prepare_right",
            res,
            layout,
            b,
            first,
            shift,
            keys,
            convert,
            scratch,
        )
    }

    fn blas_matmul_prepared_tmp_bytes_reference<R, A, B, T, K>(
        &self,
        res: &R,
        a: &A,
        b: &B,
        ctx: &BlasContext<BE>,
        tsk: &T,
        keys: &K,
    ) -> usize
    where
        BE: BlasJklsImpl + BlasMatIntraImpl,
        Self: CKKSMulOps<BE> + CKKSCopyOps<BE>,
        R: BlasGridInfos,
        A: BlasMatMulOperandInfos,
        B: BlasMatMulOperandInfos,
        T: GGLWEInfos,
        K: GGLWEInfos,
    {
        kernel_tmp_bytes(self, res, a, b, ctx, tsk).max(hot_tmp_bytes(self, a, b, ctx, keys))
    }

    fn blas_matmul_prepared_reference<R, T, K>(
        &self,
        res: &mut R,
        a: &BlasMatMulOperand<BE, CKKSPreparedLeft<BE>>,
        b: &BlasMatMulOperand<BE, CKKSPreparedRight<BE>>,
        ctx: &BlasContext<BE>,
        tsk: &T,
        keys: &K,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        BE: BlasJklsImpl + BlasMatIntraImpl,
        Self: CKKSMulOps<BE> + CKKSCopyOps<BE>,
        R: BlasGridMut<BE>,
        T: GetTensorKey<BE> + Sync,
        K: GetAutomorphismKey<BE> + Sync,
    {
        let op = "blas_matmul_prepared";
        let batch = validate_blas_matmul_prepared(self, &*res, a, b, ctx, tsk, keys)?;
        assert_scratch(op, product_scratch(self, op, &*res, a, b, ctx, tsk, keys), scratch);
        let b_rep = replicated::<BE::Ring>(b);
        product(
            self,
            res,
            a.side(),
            b.side(),
            a.cols(),
            a.log_d(),
            ctx,
            b_rep,
            tsk,
            keys,
            scratch,
        )?;
        res.set_mat_slots(batch);
        Ok(())
    }

    fn blas_matmul_tmp_bytes_reference<R, A, B, T, K>(
        &self,
        res: &R,
        a: &A,
        b: &B,
        ctx: &BlasContext<BE>,
        tsk: &T,
        keys: &K,
    ) -> usize
    where
        BE: BlasJklsImpl + BlasMatIntraImpl,
        Self: CKKSMulOps<BE> + CKKSCopyOps<BE>,
        R: BlasGridInfos,
        A: BlasGridInfos,
        B: BlasGridInfos,
        T: GGLWEInfos,
        K: GGLWEInfos,
    {
        let hot = PrepareHint::OneShot;
        let (left, right) = (
            left_operand_layout(a, &ctx.maps, hot),
            right_operand_layout(b, &ctx.maps, hot),
        );
        self.blas_matmul_prepare_left_tmp_bytes_reference(a, ctx, keys, hot)
            .max(self.blas_matmul_prepare_right_tmp_bytes_reference(b, ctx, keys, hot))
            .max(self.blas_matmul_prepared_tmp_bytes_reference(res, &left, &right, ctx, tsk, keys))
    }

    fn blas_matmul_reference<R, A, B, T, K>(
        &self,
        res: &mut R,
        a: &A,
        b: &B,
        ctx: &BlasContext<BE>,
        tsk: &T,
        keys: &K,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        BE: BlasJklsImpl + BlasMatIntraImpl,
        Self: CKKSMulOps<BE> + CKKSCopyOps<BE>,
        R: BlasGridMut<BE>,
        A: BlasGrid<BE>,
        B: BlasGrid<BE>,
        T: GetTensorKey<BE> + Sync,
        K: GetAutomorphismKey<BE> + Sync,
    {
        let op = "blas_matmul";
        let batch = validate_blas_matmul(self, &*res, a, b, ctx, tsk, keys)?;
        let hot = PrepareHint::OneShot;
        let (left, right) = (
            left_operand_layout(a, &ctx.maps, hot),
            right_operand_layout(b, &ctx.maps, hot),
        );
        let bytes = product_scratch(self, op, &*res, &left, &right, ctx, tsk, keys)
            .max(copy_tmp_bytes(self, a))
            .max(copy_tmp_bytes(self, b));
        assert_scratch(op, bytes, scratch);
        let (a_side, b_side) = (Side::Hot(a.blocks()), Side::Hot(b.blocks()));
        let b_rep = replicated::<BE::Ring>(b);
        product(self, res, a_side, b_side, a.cols(), a.log_d(), ctx, b_rep, tsk, keys, scratch)?;
        res.set_mat_slots(batch);
        Ok(())
    }

    fn blas_matmul_pt_right_tmp_bytes_reference<R, A, K>(
        &self,
        res: &R,
        a: &A,
        b: &BlasPtOperand<BE>,
        ctx: &BlasContext<BE>,
        keys: &K,
    ) -> usize
    where
        BE: BlasMatIntraImpl + BlasJklsImpl,
        Self: CKKSLinearTransformationOps<BE>
            + CKKSMulOps<BE>
            + CKKSCopyOps<BE>
            + CKKSDotProductOps<BE>
            + GLWEMulPreparedPlainSum<BE>,
        R: BlasGridInfos,
        A: BlasGridInfos,
        K: GGLWEInfos,
    {
        let masks = context_masks(b.plan(), &ctx.maps, false);
        let masked = b.plan().pt_algorithm().masked().then(|| (masks.as_slice(), &b.plaintexts[0]));
        pt_tmp_bytes(self, res, a, b.first(), masked, a.cols(), res.cols(), keys)
    }

    fn blas_matmul_pt_right_reference<R, A, K>(
        &self,
        res: &mut R,
        a: &A,
        b: &BlasPtOperand<BE>,
        ctx: &BlasContext<BE>,
        keys: &K,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        BE: BlasMatIntraImpl + BlasJklsImpl,
        Self: CKKSLinearTransformationOps<BE>
            + CKKSMulOps<BE>
            + CKKSCopyOps<BE>
            + CKKSDotProductOps<BE>
            + GLWEMulPreparedPlainSum<BE>
            + ModuleN
            + CnvPVecAlloc<BE>
            + Sized,
        R: BlasGridMut<BE>,
        A: BlasGrid<BE>,
        K: GetAutomorphismKey<BE> + Sync,
    {
        let op = "blas_matmul_pt_right";
        let batch = validate_blas_matmul_pt_right(self, &*res, a, b, ctx, keys)?;
        let (inner, n) = (a.cols(), b.cols());
        let masks = context_masks(b.plan(), &ctx.maps, false);
        let used = KeyLayouts::new(keys);
        let _ = pt_right_uses(&*res, a, &b.maps, &masks, &mut |u| ensure_rotation(op, self, &used, u));
        let bytes = used.max_bytes(|key| self.blas_matmul_pt_right_tmp_bytes_reference(&*res, a, b, ctx, key));
        assert_scratch(op, bytes, scratch);
        let (blocks, label) = (a.blocks(), pt_label(b.slots(), a, b.count()));
        let rows: Vec<Vec<_>> = res.blocks_mut().chunks_mut(n).map(|row| row.iter_mut().collect()).collect();
        let source = |i: usize, l: usize| &blocks[i * inner + l];
        let map = |_: usize, j: usize, l: usize| b.maps.at(l, j);
        let block_maps = || -> Vec<_> { (0..inner).flat_map(|l| (0..n).map(move |j| b.maps.at(l, j))).collect() };
        match b.first() {
            Some(first) if b.plan().pt_algorithm().masked() => {
                let galois = masked_galois(self, b.plan(), false);
                let plaintexts = |j: usize, l: usize| b.plaintexts_at(l, j);
                masked_units(
                    self, rows, source, inner, first, &masks, plaintexts, &galois, label, keys, scratch,
                )?;
            }
            None => {
                let mut caches = baby_caches(self, &block_maps(), &blocks[0], inner);
                shared_giants(self, &mut caches, rows, source, map, label, keys, scratch)?;
            }
            Some(first) => {
                let mut staged = alloc_blocks(self, &mapped_block(layout_of(&blocks[0]), first), inner);
                let mut caches = baby_caches(self, &block_maps(), &staged[0], inner);
                for (i, row) in rows.into_iter().enumerate() {
                    let unit: Vec<_> = blocks[i * inner..(i + 1) * inner].iter().collect();
                    BE::blas_map_blocks(self, &mut staged, &unit, &[first], keys, scratch)?;
                    shared_giants(self, &mut caches, vec![row], |_, l| &staged[l], map, label, keys, scratch)?;
                }
            }
        }
        res.set_mat_slots(batch);
        Ok(())
    }

    fn blas_matmul_pt_left_tmp_bytes_reference<R, A, K>(
        &self,
        res: &R,
        b: &BlasPtOperand<BE, Left>,
        a: &A,
        ctx: &BlasContext<BE>,
        keys: &K,
    ) -> usize
    where
        BE: BlasMatIntraImpl + BlasJklsImpl,
        Self: CKKSLinearTransformationOps<BE>
            + CKKSMulOps<BE>
            + CKKSCopyOps<BE>
            + CKKSDotProductOps<BE>
            + GLWEMulPreparedPlainSum<BE>,
        R: BlasGridInfos,
        A: BlasGridInfos,
        K: GGLWEInfos,
    {
        let masks = context_masks(b.plan(), &ctx.maps, true);
        let masked = b.plan().pt_algorithm().masked().then(|| (masks.as_slice(), &b.plaintexts[0]));
        pt_tmp_bytes(self, res, a, b.first(), masked, a.rows(), res.rows(), keys)
    }

    fn blas_matmul_pt_left_reference<R, A, K>(
        &self,
        res: &mut R,
        b: &BlasPtOperand<BE, Left>,
        a: &A,
        ctx: &BlasContext<BE>,
        keys: &K,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        BE: BlasMatIntraImpl + BlasJklsImpl,
        Self: CKKSLinearTransformationOps<BE>
            + CKKSMulOps<BE>
            + CKKSCopyOps<BE>
            + CKKSDotProductOps<BE>
            + GLWEMulPreparedPlainSum<BE>
            + ModuleN
            + CnvPVecAlloc<BE>
            + Sized,
        R: BlasGridMut<BE>,
        A: BlasGrid<BE>,
        K: GetAutomorphismKey<BE> + Sync,
    {
        let op = "blas_matmul_pt_left";
        let batch = validate_blas_matmul_pt_left(self, &*res, b, a, ctx, keys)?;
        let (inner, n) = (a.rows(), a.cols());
        let rep = left_takes_replicated(&b.maps, a);
        let masks = context_masks(b.plan(), &ctx.maps, true);
        let used = KeyLayouts::new(keys);
        let _ = pt_left_uses(&*res, &b.maps, &masks, a, &mut |u| ensure_rotation(op, self, &used, u));
        let bytes = used.max_bytes(|key| self.blas_matmul_pt_left_tmp_bytes_reference(&*res, b, a, ctx, key));
        assert_scratch(op, bytes, scratch);
        let mut columns: Vec<Vec<_>> = (0..n).map(|_| Vec::new()).collect();
        for (t, block) in res.blocks_mut().iter_mut().enumerate() {
            columns[t % n].push(block);
        }
        let (blocks, label) = (a.blocks(), pt_label(b.slots(), a, b.count()));
        let source = |j: usize, l: usize| &blocks[l * n + j];
        let map = |_: usize, i: usize, l: usize| b.maps.at_for(i, l, rep);
        let block_maps = || -> Vec<_> {
            (0..b.rows())
                .flat_map(|i| (0..inner).map(move |l| b.maps.at_for(i, l, rep)))
                .collect()
        };
        match b.first() {
            Some(first) if b.plan().pt_algorithm().masked() => {
                let galois = masked_galois(self, b.plan(), true);
                let plaintexts = |i: usize, l: usize| b.plaintexts_at(i, l);
                masked_units(
                    self, columns, source, inner, first, &masks, plaintexts, &galois, label, keys, scratch,
                )?;
            }
            None => {
                let mut caches = baby_caches(self, &block_maps(), &blocks[0], inner);
                shared_giants(self, &mut caches, columns, source, map, label, keys, scratch)?;
            }
            Some(first) => {
                let mut staged = alloc_blocks(self, &mapped_block(layout_of(&blocks[0]), first), inner);
                let mut caches = baby_caches(self, &block_maps(), &staged[0], inner);
                for (j, column) in columns.into_iter().enumerate() {
                    let unit: Vec<_> = (0..inner).map(|l| &blocks[l * n + j]).collect();
                    BE::blas_map_blocks(self, &mut staged, &unit, &[first], keys, scratch)?;
                    shared_giants(self, &mut caches, vec![column], |_, l| &staged[l], map, label, keys, scratch)?;
                }
            }
        }
        res.set_mat_slots(batch);
        Ok(())
    }
}

impl<BE: Backend, P> BlasMatMulOperand<BE, P> {
    /// The side the operand is in a product: its source blocks when hot, its terms when cold.
    fn side(&self) -> Side<'_, CKKSCiphertextOwned<BE>, P> {
        match self.layout.hint {
            PrepareHint::OneShot => Side::Hot(&self.blocks),
            PrepareHint::Reuse => Side::Cold(&self.terms),
        }
    }
}

/// One side of a product: a hot side's source blocks, row-major, whose terms the product makes under its context, or a
/// cold side's terms, block `(i, j)`'s term `k` at `(i * cols + j) * d + k`.
enum Side<'a, S, P> {
    Hot(&'a [S]),
    Cold(&'a [P]),
}

/// The prepared product's query on the layouts of the keys it looks up in `tsk` and `keys`, for `res` and the operands,
/// or their layouts, `a` and `b` under `ctx`: the tensor key's for the kernel, a hot side's rotation keys for its terms.
/// The keys passed `op`'s validator, so the look-ups only record their layouts; a call that looks up none asks for `0`.
#[allow(clippy::too_many_arguments)]
fn product_scratch<BE, T, K>(
    module: &Module<BE>,
    op: &'static str,
    res: &impl BlasGridInfos,
    a: &impl BlasMatMulOperandInfos,
    b: &impl BlasMatMulOperandInfos,
    ctx: &BlasContext<BE>,
    tsk: &T,
    keys: &K,
) -> usize
where
    BE: BlasJklsImpl + BlasMatIntraImpl,
    Module<BE>: CKKSMulOps<BE> + CKKSCopyOps<BE>,
    T: GetTensorKey<BE>,
    K: GetAutomorphismKey<BE>,
{
    let (tsks, atks) = (KeyLayouts::new(tsk), KeyLayouts::new(keys));
    let (k, rank) = prepared_tensor(a, b, &ctx.maps);
    let _ = ensure_tensor_key(op, &tsks, k, rank);
    let _ = prepared_uses(a, b, &ctx.maps, &mut |u| ensure_rotation(op, module, &atks, u));
    tsks.max_bytes(|tsk| kernel_tmp_bytes(module, res, a, b, ctx, tsk))
        .max(atks.max_bytes(|keys| hot_tmp_bytes(module, a, b, ctx, keys)))
}

/// `res = a b` under `ctx`, `res` of `m x n` blocks, for the left side `a` of `m x K` blocks and the right side `b` of
/// `K x n`, `K = inner`, as [`BlasMatMulReference::blas_matmul_prepared_reference`] states: one JKLS kernel per block
/// `(i, j)` of `res` over the `K d` pairs `(a_il[k], b_lj[k])`, `l`-major, a hot side's terms made one unit at a time,
/// a block row of `a` or a block column of `b`, by [`unit_terms`] under `OneShot`, a hot `b`'s through [`right_maps`]
/// of `b_rep`, whether `b` is replicated. Each unit is made once, and `a`'s first before any of `b`'s, so the budget
/// errors come in the order the contract fixes, before any write to `res`: of two hot sides, the held side's units come
/// first, after `a`'s first when `b` is held.
#[allow(clippy::too_many_arguments)]
fn product<BE, R, SA, SB, T, K>(
    module: &Module<BE>,
    res: &mut R,
    a: Side<'_, SA, CKKSPreparedLeft<BE>>,
    b: Side<'_, SB, CKKSPreparedRight<BE>>,
    inner: usize,
    log_d: usize,
    ctx: &BlasContext<BE>,
    b_rep: bool,
    tsk: &T,
    keys: &K,
    scratch: &mut ScratchArena<'_, BE>,
) -> Result<()>
where
    BE: BlasJklsImpl + BlasMatIntraImpl,
    Module<BE>: CKKSMulOps<BE> + CKKSCopyOps<BE>,
    R: BlasGridMut<BE>,
    SA: GLWEToBackendRef<BE> + CKKSCtBounds + Sync,
    SB: GLWEToBackendRef<BE> + CKKSCtBounds + Sync,
    T: GetTensorKey<BE> + Sync,
    K: GetAutomorphismKey<BE> + Sync,
{
    let (m, n, d, hot) = (res.rows(), res.cols(), 1usize << log_d, PrepareHint::OneShot);
    let (b_first, b_shift) = right_maps(&ctx.maps, b_rep);
    // A cold side reads no map of `ctx`, which may then be at another dimension.
    let a_shifts: Vec<_> = match a {
        Side::Hot(_) => (1..d).map(|k| ctx.rotate_cols(k)).collect(),
        Side::Cold(_) => Vec::new(),
    };
    let b_shifts: Vec<_> = match b {
        Side::Hot(_) => (1..d).map(b_shift).collect(),
        Side::Cold(_) => Vec::new(),
    };
    // The terms of block row `i` of a hot `a`, and of block column `j` of a hot `b`, `l`-major.
    let a_row = |blocks: &[SA], i: usize, scratch: &mut ScratchArena<'_, BE>| {
        let sources: Vec<_> = blocks[i * inner..(i + 1) * inner].iter().collect();
        let convert = |term: &CKKSCiphertextOwned<BE>, s: &mut ScratchArena<'_, BE>| module.ckks_prepare_left(term, hot, s);
        new_unit_terms(module, &sources, ctx.sigma(), &a_shifts, keys, convert, scratch)
    };
    let b_col = |blocks: &[SB], j: usize, scratch: &mut ScratchArena<'_, BE>| {
        let sources: Vec<_> = (0..inner).map(|l| &blocks[l * n + j]).collect();
        let convert = |term: &CKKSCiphertextOwned<BE>, s: &mut ScratchArena<'_, BE>| module.ckks_prepare_right(term, hot, s);
        new_unit_terms(module, &sources, b_first, &b_shifts, keys, convert, scratch)
    };
    let mut jkls =
        |i: usize, j: usize, lhs: &[&CKKSPreparedLeft<BE>], rhs: &[&CKKSPreparedRight<BE>], s: &mut ScratchArena<'_, BE>| {
            BE::blas_jkls_block(module, &mut res.blocks_mut()[i * n + j], lhs, rhs, tsk, s)
        };
    match (a, b) {
        (Side::Cold(at), Side::Cold(bt)) => {
            for i in 0..m {
                let lhs = cold_row(at, i, inner, d);
                for j in 0..n {
                    jkls(i, j, &lhs, &cold_col(bt, j, inner, n, d), scratch)?;
                }
            }
        }
        (Side::Hot(ab), Side::Cold(bt)) => {
            for i in 0..m {
                let row = a_row(ab, i, scratch)?;
                let lhs: Vec<_> = row.iter().collect();
                for j in 0..n {
                    jkls(i, j, &lhs, &cold_col(bt, j, inner, n, d), scratch)?;
                }
            }
        }
        (Side::Cold(at), Side::Hot(bb)) => {
            for j in 0..n {
                let col = b_col(bb, j, scratch)?;
                let rhs: Vec<_> = col.iter().collect();
                for i in 0..m {
                    jkls(i, j, &cold_row(at, i, inner, d), &rhs, scratch)?;
                }
            }
        }
        (Side::Hot(ab), Side::Hot(bb)) => {
            // Held terms stay in the transform domain: as coefficients, each streamed unit would transform them again.
            let limbs = |block: CKKSLayout, first, shift| mapped_block(mapped_block(block, first), shift).size();
            let la = limbs(layout_of(&ab[0]), ctx.sigma(), ctx.rotate_cols(1));
            let lb = limbs(layout_of(&bb[0]), b_first, b_shift(1));
            if (m - 1) * la <= (n - 1) * lb {
                let held = (0..m).map(|i| a_row(ab, i, scratch)).collect::<Result<Vec<_>>>()?;
                let lhs: Vec<Vec<_>> = held.iter().map(|row| row.iter().collect()).collect();
                for j in 0..n {
                    let col = b_col(bb, j, scratch)?;
                    let rhs: Vec<_> = col.iter().collect();
                    for (i, lhs) in lhs.iter().enumerate() {
                        jkls(i, j, lhs, &rhs, scratch)?;
                    }
                }
            } else {
                let mut first = Some(a_row(ab, 0, scratch)?);
                let held = (0..n).map(|j| b_col(bb, j, scratch)).collect::<Result<Vec<_>>>()?;
                let rhs: Vec<Vec<_>> = held.iter().map(|col| col.iter().collect()).collect();
                for i in 0..m {
                    let row = match first.take() {
                        Some(row) => row,
                        None => a_row(ab, i, scratch)?,
                    };
                    let lhs: Vec<_> = row.iter().collect();
                    for (j, rhs) in rhs.iter().enumerate() {
                        jkls(i, j, &lhs, rhs, scratch)?;
                    }
                }
            }
        }
    }
    Ok(())
}

/// The terms of block row `i` of a cold left side of `inner` block columns and `d` terms per block, `l`-major.
fn cold_row<P>(terms: &[P], i: usize, inner: usize, d: usize) -> Vec<&P> {
    terms[i * inner * d..(i + 1) * inner * d].iter().collect()
}

/// The terms of block column `j` of a cold right side of `inner` block rows, `n` block columns and `d` terms per block,
/// `l`-major.
fn cold_col<P>(terms: &[P], j: usize, inner: usize, n: usize, d: usize) -> Vec<&P> {
    (0..inner)
        .flat_map(|l| &terms[(l * n + j) * d..(l * n + j + 1) * d])
        .collect()
}

/// The JKLS kernel's query for a block of `res` and the `K d` pairs of the terms of `a` and `b` in a product under `ctx`.
fn kernel_tmp_bytes<BE: BlasJklsImpl>(
    module: &Module<BE>,
    res: &impl BlasGridInfos,
    a: &impl BlasMatMulOperandInfos,
    b: &impl BlasMatMulOperandInfos,
    ctx: &BlasContext<BE>,
    tsk: &impl GGLWEInfos,
) -> usize {
    let (res, lhs, rhs) = (
        res.block_layout(),
        product_terms(a, || left_term_layout(a, &ctx.maps)),
        product_terms(b, || right_term_layout(b, &ctx.maps)),
    );
    // A store may answer at the product's width with a narrower key, whose relinearization can take more scratch.
    let tsk = tsk.gglwe_layout();
    let at_width = GGLWELayout {
        dnum: GGLWELayout::dnum_for_input(lhs.k().max(rhs.k()), tsk.k_digit).min(tsk.dnum),
        ..tsk
    };
    let terms = a.cols() << a.log_d();
    BE::blas_jkls_block_tmp_bytes(module, terms, &res, &lhs, &rhs, &tsk)
        .max(BE::blas_jkls_block_tmp_bytes(module, terms, &res, &lhs, &rhs, &at_width))
}

/// The scratch of the terms of the hot sides of `a` and `b` in a product under `ctx`, a unit at a time, `K` blocks each:
/// a block row of `a`, a block column of `b`; `0` without a hot side.
fn hot_tmp_bytes<BE>(
    module: &Module<BE>,
    a: &impl BlasMatMulOperandInfos,
    b: &impl BlasMatMulOperandInfos,
    ctx: &BlasContext<BE>,
    keys: &impl GGLWEInfos,
) -> usize
where
    BE: BlasMatIntraImpl,
    Module<BE>: CKKSMulOps<BE> + CKKSCopyOps<BE>,
{
    let left = match a.hint() {
        PrepareHint::OneShot => {
            let shifts: Vec<_> = (1..1usize << a.log_d()).map(|k| ctx.rotate_cols(k)).collect();
            unit_tmp_bytes(module, a, ctx.sigma(), &shifts, keys, a.cols())
        }
        PrepareHint::Reuse => 0,
    };
    let right = match b.hint() {
        PrepareHint::OneShot => {
            let (first, shift) = right_maps(&ctx.maps, replicated::<BE::Ring>(b));
            let shifts: Vec<_> = (1..1usize << b.log_d()).map(shift).collect();
            unit_tmp_bytes(module, b, first, &shifts, keys, b.rows())
        }
        PrepareHint::Reuse => 0,
    };
    left.max(right)
}

/// Scratch of a hot preparation of `a`: the copy of a block of `a` into a block of its layout.
fn copy_tmp_bytes<BE: Backend>(module: &Module<BE>, a: &impl BlasGridInfos) -> usize
where
    Module<BE>: CKKSCopyOps<BE>,
{
    let block = a.block_layout();
    module.ckks_copy_tmp_bytes(&block, &block)
}

/// Scratch of a preparation of `a` under `hint` with `first` (sigma or tau) and `shift(k)` (phi^k or psi^k): the copy
/// of a block of `a` into a block of its layout when hot, the query of one block row's terms when cold.
fn prepare_tmp_bytes<'c, BE, K>(
    module: &Module<BE>,
    a: &impl BlasGridInfos,
    first: &LinearTransformationPrepared<BE>,
    shift: impl Fn(usize) -> &'c LinearTransformationPrepared<BE>,
    keys: &K,
    hint: PrepareHint,
) -> usize
where
    BE: BlasMatIntraImpl + 'c,
    Module<BE>: CKKSMulOps<BE> + CKKSCopyOps<BE>,
    K: GGLWEInfos,
{
    match hint {
        PrepareHint::OneShot => copy_tmp_bytes(module, a),
        PrepareHint::Reuse => {
            let shifts: Vec<_> = (1..1usize << a.log_d()).map(shift).collect();
            unit_tmp_bytes(module, a, first, &shifts, keys, a.cols())
        }
    }
}

/// Scratch of the terms of `sources` blocks of `a`'s layout through `first` and every map of `shifts` ([`unit_terms`]),
/// each converted once: the largest of the map hook's queries through `first`, from those blocks into blocks of the
/// layout it leaves, and through `shifts`, from those into blocks of the terms' layout, of the copy of one into a block
/// of the terms' layout, and of the conversion of a term.
fn unit_tmp_bytes<BE, K>(
    module: &Module<BE>,
    a: &impl BlasGridInfos,
    first: &LinearTransformationPrepared<BE>,
    shifts: &[&LinearTransformationPrepared<BE>],
    keys: &K,
    sources: usize,
) -> usize
where
    BE: BlasMatIntraImpl,
    Module<BE>: CKKSMulOps<BE> + CKKSCopyOps<BE>,
    K: GGLWEInfos,
{
    let block = a.block_layout();
    let s = mapped_block(block, first);
    let term = mapped_block(s, shifts[0]);
    BE::blas_map_blocks_tmp_bytes(module, sources, &s, &block, &[first], keys)
        .max(BE::blas_map_blocks_tmp_bytes(module, sources, &term, &s, shifts, keys))
        .max(module.ckks_copy_tmp_bytes(&term, &s))
        .max(module.ckks_prepare_tmp_bytes(&term))
}

/// Fills `res` from `a` under its hint: hot, `ckks_copy` of every block of `a` into `res`'s; cold, the terms of every
/// block, block row after block row ([`unit_terms`]), each converted once by `convert` into its place in `res`. `res`
/// then reports `layout`, the infos of this preparation, and `a`'s batch.
#[allow(clippy::too_many_arguments)]
fn prepare<'c, BE, A, K, P>(
    module: &Module<BE>,
    op: &'static str,
    res: &mut BlasMatMulOperand<BE, P>,
    layout: BlasMatMulOperandLayout,
    a: &A,
    first: &LinearTransformationPrepared<BE>,
    shift: impl Fn(usize) -> &'c LinearTransformationPrepared<BE>,
    keys: &K,
    convert: impl Fn(&mut P, &CKKSCiphertextOwned<BE>, &mut ScratchArena<'_, BE>) -> Result<()>,
    scratch: &mut ScratchArena<'_, BE>,
) -> Result<()>
where
    BE: BlasMatIntraImpl + 'c,
    Module<BE>: CKKSMulOps<BE> + CKKSCopyOps<BE>,
    A: BlasGrid<BE>,
    K: GetAutomorphismKey<BE> + Sync,
{
    match res.layout.hint {
        PrepareHint::OneShot => {
            assert_scratch(op, copy_tmp_bytes(module, a), scratch);
            for (block, source) in res.blocks.iter_mut().zip(a.blocks()) {
                module.ckks_copy(block, source, scratch)?;
            }
        }
        PrepareHint::Reuse => {
            let used = KeyLayouts::new(keys);
            let _ = ensure_term_keys(op, module, a, first, &shift, &used);
            let bytes = used.max_bytes(|key| prepare_tmp_bytes(module, a, first, &shift, key, PrepareHint::Reuse));
            assert_scratch(op, bytes, scratch);
            let (row_terms, shifts) = (
                a.cols() << a.log_d(),
                (1..1usize << a.log_d()).map(&shift).collect::<Vec<_>>(),
            );
            for (row, out) in a.blocks().chunks(a.cols()).zip(res.terms.chunks_mut(row_terms)) {
                let sources: Vec<_> = row.iter().collect();
                let emit = |t: usize, term: &CKKSCiphertextOwned<BE>, s: &mut ScratchArena<'_, BE>| convert(&mut out[t], term, s);
                unit_terms(module, &sources, first, &shifts, keys, emit, scratch)?;
            }
        }
    }
    res.layout = layout;
    res.mat_slots = grid_batch::<BE::Ring>(a);
    Ok(())
}

/// `count` ciphertexts of layout `layout`.
fn alloc_blocks<BE: BlasMatIntraImpl>(module: &Module<BE>, layout: &CKKSLayout, count: usize) -> Vec<CKKSCiphertextOwned<BE>> {
    (0..count).map(|_| module.ckks_ciphertext_alloc_from_infos(layout)).collect()
}

/// The terms of `sources`, blocks of one layout, through the map hook: `S = first(s)` for every source into a heap block
/// at the width `first` leaves, then every map of `shifts` of every `S` into heap blocks of the terms' layout, then each
/// `S` narrowed by `ckks_copy` into a block of the terms' layout, its term `0`, and dropped. Each term is passed once to
/// `emit` with its index, term `k` of source `t` at `t * (shifts.len() + 1) + k`, then dropped.
fn unit_terms<BE, S, K>(
    module: &Module<BE>,
    sources: &[&S],
    first: &LinearTransformationPrepared<BE>,
    shifts: &[&LinearTransformationPrepared<BE>],
    keys: &K,
    mut emit: impl FnMut(usize, &CKKSCiphertextOwned<BE>, &mut ScratchArena<'_, BE>) -> Result<()>,
    scratch: &mut ScratchArena<'_, BE>,
) -> Result<()>
where
    BE: BlasMatIntraImpl,
    Module<BE>: CKKSCopyOps<BE>,
    S: GLWEToBackendRef<BE> + CKKSCtBounds + Sync,
    K: GetAutomorphismKey<BE> + Sync,
{
    let (block, per) = (layout_of(sources[0]), shifts.len() + 1);
    let staged = mapped_block(block, first);
    let term = mapped_block(staged, shifts[0]);
    let mut sigma = alloc_blocks(module, &staged, sources.len());
    BE::blas_map_blocks(module, &mut sigma, sources, &[first], keys, scratch)?;
    let results: Vec<_> = sigma.iter().collect();
    let mut shifted = alloc_blocks(module, &term, sigma.len() * shifts.len());
    BE::blas_map_blocks(module, &mut shifted, &results, shifts, keys, scratch)?;
    let mut shifted = shifted.into_iter();
    for (t, s) in sigma.into_iter().enumerate() {
        let mut narrowed = module.ckks_ciphertext_alloc_from_infos(&term);
        module.ckks_copy(&mut narrowed, &s, scratch)?;
        drop(s);
        emit(t * per, &narrowed, scratch)?;
        for (k, shifted) in shifted.by_ref().take(shifts.len()).enumerate() {
            emit(t * per + k + 1, &shifted, scratch)?;
        }
    }
    Ok(())
}

/// [`unit_terms`] of `sources`, each term converted once by `convert` into a new operand, in their order.
fn new_unit_terms<BE, S, K, P>(
    module: &Module<BE>,
    sources: &[&S],
    first: &LinearTransformationPrepared<BE>,
    shifts: &[&LinearTransformationPrepared<BE>],
    keys: &K,
    convert: impl Fn(&CKKSCiphertextOwned<BE>, &mut ScratchArena<'_, BE>) -> Result<P>,
    scratch: &mut ScratchArena<'_, BE>,
) -> Result<Vec<P>>
where
    BE: BlasMatIntraImpl,
    Module<BE>: CKKSCopyOps<BE>,
    S: GLWEToBackendRef<BE> + CKKSCtBounds + Sync,
    K: GetAutomorphismKey<BE> + Sync,
{
    let mut terms = Vec::with_capacity(sources.len() * (shifts.len() + 1));
    let emit = |_: usize, term: &CKKSCiphertextOwned<BE>, s: &mut ScratchArena<'_, BE>| {
        terms.push(convert(term, s)?);
        Ok(())
    };
    unit_terms(module, sources, first, shifts, keys, emit, scratch)?;
    Ok(terms)
}

/// Scratch of a product by public matrices of blocks of layout `a` into blocks of layout `res`, `sources` blocks of `a`
/// per unit: the baby steps of one block and one summed linear transformation, from the blocks `first` leaves, and
/// under JKLS the map hook's query of `first` over a unit; under the masked paths, `masked` holding the context's masks
/// and one of the operand's prepared plaintexts, [`BlasJklsImpl::blas_jkls_pt_unit`]'s query into the `outputs` blocks
/// a unit writes. The working blocks live on the heap.
#[allow(clippy::too_many_arguments)]
fn pt_tmp_bytes<BE: BlasMatIntraImpl + BlasJklsImpl>(
    module: &Module<BE>,
    res: &impl BlasGridInfos,
    a: &impl BlasGridInfos,
    first: Option<&LinearTransformationPrepared<BE>>,
    masked: Option<(&[&LinearTransformationPrepared<BE>], &CKKSPreparedRight<BE>)>,
    sources: usize,
    outputs: usize,
    keys: &impl GGLWEInfos,
) -> usize
where
    Module<BE>: CKKSLinearTransformationOps<BE>,
{
    let (res, block) = (res.block_layout(), a.block_layout());
    let Some(first) = first else {
        return lt_tmp_bytes(module, &res, &block, keys, true);
    };
    let Some((masks, plaintext)) = masked else {
        let staged = mapped_block(block, first);
        return BE::blas_map_blocks_tmp_bytes(module, sources, &staged, &block, &[first], keys)
            .max(lt_tmp_bytes(module, &res, &staged, keys, true));
    };
    BE::blas_jkls_pt_unit_tmp_bytes(module, outputs, &res, &block, first, masks, plaintext, keys)
}

/// `count` baby-step caches for sources of `src`'s layout, each covering the baby steps of every map of `maps`.
fn baby_caches<BE, M, S>(
    module: &M,
    maps: &[&LinearTransformationPrepared<BE>],
    src: &S,
    count: usize,
) -> Vec<LinearTransformationBabySteps<BE>>
where
    BE: Backend,
    M: CKKSLinearTransformationOps<BE> + ModuleN + CnvPVecAlloc<BE>,
    S: GLWEToBackendRef<BE> + CKKSCtBounds,
{
    let rotations: Vec<i64> = maps.iter().flat_map(|lt| lt.baby_steps().iter().copied()).collect();
    (0..count)
        .map(|_| LinearTransformationBabySteps::alloc(module, &rotations, src))
        .collect()
}

/// The products by public matrices, which share their giant steps across their terms, one per cache of `caches`
/// ([`baby_caches`]): group `g` of `groups` holds the blocks `o` of `res` it writes, block `o` receiving poulpy-ckks's
/// summed linear transformation of the terms `(source(g, l), map(g, o, l))` in ascending `l`, then `label`, the
/// product's [`pt_label`]. The caches are filled from a group's sources and serve all its blocks. The groups run in
/// order, refilling the same caches, and the first error stops them.
#[allow(clippy::too_many_arguments)]
fn shared_giants<'s, 'm, BE, M, D, S, K>(
    module: &M,
    caches: &mut [LinearTransformationBabySteps<BE>],
    groups: Vec<Vec<&mut D>>,
    source: impl Fn(usize, usize) -> &'s S,
    map: impl Fn(usize, usize, usize) -> &'m LinearTransformationPrepared<BE>,
    label: usize,
    keys: &K,
    scratch: &mut ScratchArena<'_, BE>,
) -> Result<()>
where
    BE: Backend + 's + 'm,
    M: CKKSLinearTransformationOps<BE>,
    D: GLWEToBackendMut<BE> + CKKSCtBounds + SetCKKSInfos,
    S: GLWEToBackendRef<BE> + CKKSCtBounds + 's,
    K: GetAutomorphismKey<BE>,
{
    let inner = caches.len();
    let (mut srcs, mut lts) = (Vec::with_capacity(inner), Vec::with_capacity(inner));
    for (g, blocks) in groups.into_iter().enumerate() {
        for (l, cache) in caches.iter_mut().enumerate() {
            module.ckks_prepare_linear_transformation_baby_steps(cache, source(g, l), keys, scratch)?;
        }
        let babies: Vec<_> = caches.iter().collect();
        srcs.clear();
        srcs.extend((0..inner).map(|l| source(g, l)));
        for (o, dst) in blocks.into_iter().enumerate() {
            lts.clear();
            lts.extend((0..inner).map(|l| map(g, o, l)));
            module.ckks_eval_linear_transformation_sum_into(&mut *dst, &srcs, &babies, &lts, keys, scratch)?;
            dst.set_log_sparsity(label);
        }
    }
    Ok(())
}

/// The products by public matrices under `JklsStored` and `JklsGalois`: one [`BlasJklsImpl::blas_jkls_pt_unit`] per
/// group `g` of `groups`, the blocks `o` of `res` it writes, from its `inner` sources `source(g, l)` by the prepared
/// plaintexts `plaintexts(o, l)` read through `galois`, then `label` on each block. The groups run in order, and the
/// first error stops them.
#[allow(clippy::too_many_arguments)]
fn masked_units<'s, 'p, BE, D, S, K>(
    module: &Module<BE>,
    groups: Vec<Vec<&mut D>>,
    source: impl Fn(usize, usize) -> &'s S,
    inner: usize,
    first: &LinearTransformationPrepared<BE>,
    masks: &[&LinearTransformationPrepared<BE>],
    plaintexts: impl Fn(usize, usize) -> &'p [CKKSPreparedRight<BE>],
    galois: &[i64],
    label: usize,
    keys: &K,
    scratch: &mut ScratchArena<'_, BE>,
) -> Result<()>
where
    BE: BlasJklsImpl + 's + 'p,
    D: GLWEToBackendMut<BE> + CKKSCtBounds + SetCKKSInfos + Send,
    S: GLWEToBackendRef<BE> + CKKSCtBounds + Sync + 's,
    K: GetAutomorphismKey<BE> + Sync,
{
    let (mut sources, mut pts) = (Vec::with_capacity(inner), Vec::new());
    for (g, mut blocks) in groups.into_iter().enumerate() {
        sources.clear();
        sources.extend((0..inner).map(|l| source(g, l)));
        pts.clear();
        pts.extend(
            (0..blocks.len())
                .flat_map(|o| (0..inner).map(move |l| (o, l)))
                .map(|(o, l)| plaintexts(o, l)),
        );
        BE::blas_jkls_pt_unit(module, &mut blocks, &sources, first, masks, &pts, galois, keys, scratch)?;
        for dst in blocks {
            dst.set_log_sparsity(label);
        }
    }
    Ok(())
}

/// The Galois element of every term of a source block, in the term order of [`masked_rotations`]: `1` under
/// `JklsStored`, whose plaintexts are stored rotated; under `JklsGalois`, that of the term's rotation, which reads the
/// block's one plaintext.
fn masked_galois<BE: Backend>(module: &Module<BE>, plan: BlasPlan, left: bool) -> Vec<i64> {
    let rotations = masked_rotations(plan.log_d(), left);
    match plan.pt_algorithm() {
        BlasPtAlgorithm::JklsGalois => rotations.iter().map(|&r| module.ckks_galois_element(r)).collect(),
        _ => vec![1; rotations.len()],
    }
}
