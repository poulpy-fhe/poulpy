use poulpy_ckks::CKKSResult as Result;
use poulpy_core::layouts::{GGLWEInfos, GetAutomorphismKey, GetTensorKey};
use poulpy_hal::layouts::{Backend, PrepareHint, ScratchArena};

use crate::layouts::{
    BlasContext, BlasGrid, BlasGridInfos, BlasGridMut, BlasMatMulLeft, BlasMatMulOperandInfos, BlasMatMulRight, BlasPtOperand,
    Left,
};

/// Matrix products of every matrix of the batch, across the blocks of a grid. A product records the larger batch of its
/// operands ([`BlasGridInfos::mat_slots`]); a prepared operand and the products by public matrices record their
/// input's.
///
/// In the `# Metadata` sections, `x.k` is the width of the blocks of `x`, for `res` the width they have before the
/// call and for a prepared operand the width it reports, its source's when hot and its terms' when cold; `p` is the
/// scale of the maps of `ctx`, that of the `coeffs_meta` it was encoded at, and `p_B` that of the public operand `b`; an
/// input's batch is its record, or the batch its blocks' label proves when it has none
/// ([`grid_batch`](crate::layouts::grid_batch)). The maps, of `ctx` and of public operands, have `Real` slots. The
/// products of two encrypted grids follow poulpy-ckks's ct x ct rule ([`CKKSMulOps`](poulpy_ckks::api::CKKSMulOps)) on
/// their terms, for grids `a` and `b` through the maps of `ctx`: `w_a = a.k - 2p`, and `w_b = b.k - p` when `b` is
/// replicated ([`crate::layouts::replicated`]), `b.k - 2p` otherwise. On a replicated `b`, every matrix slot holding
/// the same matrix, psi^k is a rotation by `k d` at no scale. If `b.log_delta <= a.log_delta`, a replicated right
/// factor gains a level only when fed `p` narrower than the left factor; at equal widths its terms and the tensor key
/// are `p` wider for the same result. With a larger right scale, its terms widen the result by
/// `min(b.log_delta - a.log_delta, p)`, up to `res.k`. At equal widths and `b.log_delta <= a.log_delta`, narrowing a
/// replicated right factor by `p` first gives the same result with narrower terms and tensor key.
///
/// A prepared operand is hot or cold, after the [`PrepareHint`] it was prepared under. A hot one, `OneShot`, prepared
/// for one product, is a relabel of its source: the product evaluates its terms under its own `ctx`. A cold one,
/// `Reuse`, prepared for many, holds every term under its preparation's `ctx`, each converted once to the transform
/// domain. Under the same context, the hint selects where the work happens and the memory held, never the bits.
pub trait BlasMatMulOps<BE: Backend> {
    /// Scratch bytes [`Self::blas_matmul`] needs into `res` from `a` and `b` with the maps of `ctx`, a tensor key of
    /// layout `tsk` and rotation keys of layout `atk`. The right factor's label selects its terms, so a layout given in
    /// its place carries its label.
    ///
    /// # Panics
    ///
    /// May panic on layouts [`validate_blas_matmul`](crate::layouts::validate_blas_matmul) rejects: blocks of a degree
    /// other than the module's, factors of different radices, a factor at another `log_d` than `ctx`'s, or
    /// `rows * cols` overflowing `usize`.
    fn blas_matmul_tmp_bytes<R, A, B, T, K>(&self, res: &R, a: &A, b: &B, ctx: &BlasContext<BE>, tsk: &T, atk: &K) -> usize
    where
        R: BlasGridInfos,
        A: BlasGridInfos,
        B: BlasGridInfos,
        T: GGLWEInfos,
        K: GGLWEInfos;

    /// `res` receives the product of every matrix of `a` with the matching matrix of `b`: `a` and `b` prepared hot, then
    /// their prepared product under `ctx`, with the rotation keys `atk` and the tensor key `tsk`, bit for bit.
    ///
    /// Batches of different sizes broadcast: with `m_a` and `m_b` matrices in `a` and `b`, matrix slot `s` of `res` holds
    /// `A_(s mod m_a) B_(s mod m_b)`, so `res` holds the larger batch, at the smaller `log_sparsity` of the two.
    ///
    /// Each block of `res` is one JKLS kernel over its `K d` pairs of terms, `K = a.cols()`, relinearized once. On rank
    /// 1, with one `base2k` across the blocks and the tensor key and an offset of at least `base2k` in poulpy-ckks's ct x
    /// ct rule, which `max(w_a, w_b) >= base2k` guarantees, the kernel adds the products of every pair in the transform
    /// domain before one inverse transform, so choose `base2k` for the `2 L K d` limb products it adds in an output
    /// limb, up to `2 L` per pair, `L` the limbs of the narrower terms, `min(w_a, w_b)` over `base2k` rounded up, as for
    /// poulpy-core's `glwe_mul_relinearize_sum`, and for the maps that make the terms, each the count
    /// [`Self::blas_matmul_pt_right`] documents for one term:
    /// `blas_matmul_max_base2k` returns the largest radix `Module::max_base2k`
    /// allows for every one of them, counting limbs at each radix. An FFT backend adds the kernel's products in floating
    /// point, in the order of the pairs `(a_il[k], b_lj[k])`, `l`-major: ascending `l`, then ascending `k` within each
    /// `l`. That order is then part of the bits.
    ///
    /// The reference reads `a` and `b` in place and makes every term once, converted once to the transform domain. For
    /// `a` of `m x K` blocks and `b` of `K x n`, it holds the terms of the side that, with one unit of the other, takes
    /// fewer limbs (`a` when `m <= n` at one term width), and makes those of the other side a unit at a time, a block
    /// row of `a` or a block column of `b`, each serving its row (column) of `res`. So beside `a`, `b` and `res` its heap
    /// holds the held side's `K d` terms per block row (column), in the backend's transform-domain form, and one unit of
    /// the other with its sigma (tau) results, whatever the extent of the streamed side.
    ///
    /// # Metadata
    ///
    /// Every block of `res` is one JKLS product of the terms of `a` and `b`, as [`Self::blas_matmul_prepared`] makes it:
    ///
    /// ```text
    /// log_delta    = min(a.log_delta, b.log_delta)
    /// k            = min(res.k, min(w_a - a.log_delta, w_b - b.log_delta) - max(a.log_delta, b.log_delta) + log_delta)
    /// log_sparsity = min(a.log_sparsity, b.log_sparsity)
    /// slots        = a.slots.join(b.slots).join(Real)
    /// canonical    = true
    /// noise        = None
    /// mat_slots    = max(a's batch, b's batch)
    /// ```
    ///
    /// For scales equal to `delta`, `k = min(res.k, min(w_a, w_b) - delta)`.
    ///
    /// # Errors
    ///
    /// - The implementation runs [`validate_blas_matmul`](crate::layouts::validate_blas_matmul) before any write, so on
    ///   its errors `res`, its batch record and the factors are untouched: `InvalidOperand` for a grid, shape, layout,
    ///   context or radix above 62 it rejects, then, key by key, `MissingAutomorphismKey` for a key of the preparations
    ///   `atk` lacks, at the widths [`Self::blas_matmul_prepare_left`] and [`Self::blas_matmul_prepare_right`] state,
    ///   or `MissingRelinearizationKey` for the tensor key `tsk` lacks at `max(w_a, w_b)`, or `InvalidOperand` for a
    ///   key of another rank than the factors' blocks.
    /// - poulpy-ckks's budget errors come from inside the backend before any write to `res`, and leave it untouched
    ///   too: the terms' `MultiplicationPrecisionUnderflow` when the budget of `a` or `b`, its `k - log_delta`, is
    ///   below what its terms spend, `2p`, or `p` for a replicated `b`, `a`'s first, then the kernel's,
    ///   `MultiplicationPrecisionUnderflow` when `min(w_a - a.log_delta, w_b - b.log_delta)` is below
    ///   `max(a.log_delta, b.log_delta)` and `InsufficientHomomorphicCapacity` when `res.k` is below
    ///   `min(a.log_delta, b.log_delta)`.
    /// - Any other `Err` is a backend fault (a validator gap or a device failure), and `res` is then unspecified.
    #[allow(clippy::too_many_arguments)]
    fn blas_matmul<R, A, B, T, K>(
        &self,
        res: &mut R,
        a: &A,
        b: &B,
        ctx: &BlasContext<BE>,
        tsk: &T,
        atk: &K,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        R: BlasGridMut<BE>,
        A: BlasGrid<BE>,
        B: BlasGrid<BE>,
        T: GetTensorKey<BE> + Sync,
        K: GetAutomorphismKey<BE> + Sync;

    /// Scratch bytes [`Self::blas_matmul_prepare_left`] needs from `a` under `hint`, with the maps of `ctx` and rotation
    /// keys of layout `atk`: the copies of a hot preparation; the maps, copies and conversions of a cold one.
    ///
    /// # Panics
    ///
    /// May panic on layouts [`validate_blas_matmul_prepare_left`](crate::layouts::validate_blas_matmul_prepare_left)
    /// rejects: blocks of a degree other than the module's, a grid at another `log_d` than `ctx`'s, or `rows * cols`
    /// overflowing `usize`.
    fn blas_matmul_prepare_left_tmp_bytes<A, K>(&self, a: &A, ctx: &BlasContext<BE>, atk: &K, hint: PrepareHint) -> usize
    where
        A: BlasGridInfos,
        K: GGLWEInfos;

    /// Bytes a left operand of `infos` holds, as [`Self::blas_matmul_left_alloc_from_infos`] allocates it.
    fn blas_matmul_left_bytes_of_from_infos<A>(&self, infos: &A) -> usize
    where
        A: BlasMatMulOperandInfos;

    /// A left operand of `infos`, holding all its storage, which [`Self::blas_matmul_prepare_left`] fills: the infos
    /// `blas_matmul_left_operand_layout` gives for a grid, a context
    /// and a hint. Until then it reports `infos` and the batch `0`, which the prepared product rejects.
    ///
    /// # Panics
    ///
    /// May panic on infos of no block, or whose bytes overflow `usize`.
    fn blas_matmul_left_alloc_from_infos<A>(&self, infos: &A) -> BlasMatMulLeft<BE>
    where
        A: BlasMatMulOperandInfos;

    /// Fills `res`, a left operand of [`Self::blas_matmul_prepared`] allocated by
    /// [`Self::blas_matmul_left_alloc_from_infos`] from the infos
    /// `blas_matmul_left_operand_layout` gives for `a`, `ctx` and a
    /// hint, or for a grid of `a`'s shape (dimensions, degree, radix, width and rank), whatever its metadata, from `a`
    /// under that hint, with no allocation of its storage:
    /// - hot, `OneShot`, for one product: a relabel of `a`, which the reference copies into its blocks by poulpy-ckks's
    ///   `ckks_copy`, through no map, key or transform: the live limbs, metadata, canonical flag and noise tag of a
    ///   canonical block, a non-canonical one normalized. The product evaluates its terms under its own context,
    ///   whatever `ctx` is here;
    /// - cold, `Reuse`, for many: every term the product multiplies, sigma of every block of `a` and its shifts under
    ///   `ctx`, each converted once to the transform domain by poulpy-ckks's `ckks_prepare_left_into`, a block row at a
    ///   time through heap working blocks.
    ///
    /// When `ctx` is the product's, the products give the same bits under either hint, which selects where the work
    /// happens and the memory held: for `a` of `m x K` blocks, a hot operand holds `m K` blocks of `a`'s layout, a cold
    /// one `m K d` terms in the backend's transform-domain form. The operand owns its storage, so `a` may change or go
    /// once it returns, and preparing into it again refills it.
    ///
    /// # Metadata
    ///
    /// The operand's [`BlasMatMulOperandInfos`] report the infos of its latest preparation: its hint, `a`'s block
    /// layout as its source, `a`'s `log_d`, `rows`, `cols` and batch, and as its block layout `a`'s when hot and that
    /// of every term when cold, as
    /// `blas_matmul_left_operand_layout` gives them before
    /// preparing:
    ///
    /// ```text
    ///                hot               cold
    /// k            = a.k               a.k - 2p
    /// log_delta    = a.log_delta       a.log_delta
    /// log_sparsity = a.log_sparsity    a.log_sparsity
    /// slots        = a.slots           a.slots.join(Real)
    /// mat_slots    = Some(a's batch)   Some(a's batch)
    /// ```
    ///
    /// Nothing else is specified: the backend holds the operand in its own representation, without a canonical flag or a
    /// noise tag to read.
    ///
    /// # Errors
    ///
    /// - The implementation runs
    ///   [`validate_blas_matmul_prepare_left`](crate::layouts::validate_blas_matmul_prepare_left) before any write,
    ///   whatever the hint: `InvalidOperand` for a grid, layout or context it rejects or for `res` of another shape,
    ///   then, key by key, `MissingAutomorphismKey` for a key `atk` lacks: sigma's baby steps at `a.k` and giant steps
    ///   at `a.k - p`, each phi^k's baby steps at `a.k - p` and giant steps at `a.k - 2p`; or `InvalidOperand` for one
    ///   of another rank than `a`'s blocks.
    /// - poulpy-ckks's budget error comes from inside a cold preparation: `MultiplicationPrecisionUnderflow` when `a`'s
    ///   budget, `a.k - a.log_delta`, is below `2p`. A hot preparation checks no budget: the product returns that error.
    /// - Any other `Err` is a backend fault (a validator gap or a device failure).
    ///
    /// A preparation writes only `res`, so no error changes `a`, and the validator's and poulpy-ckks's errors leave
    /// `res` as it was.
    fn blas_matmul_prepare_left<A, K>(
        &self,
        res: &mut BlasMatMulLeft<BE>,
        a: &A,
        ctx: &BlasContext<BE>,
        atk: &K,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        A: BlasGrid<BE>,
        K: GetAutomorphismKey<BE> + Sync;

    /// Scratch bytes [`Self::blas_matmul_prepare_right`] needs from `b` under `hint`, with the maps of `ctx` and
    /// rotation keys of layout `atk`. The right factor's label selects its terms, so a layout given in its place carries
    /// its label.
    ///
    /// # Panics
    ///
    /// As [`Self::blas_matmul_prepare_left_tmp_bytes`].
    fn blas_matmul_prepare_right_tmp_bytes<B, K>(&self, b: &B, ctx: &BlasContext<BE>, atk: &K, hint: PrepareHint) -> usize
    where
        B: BlasGridInfos,
        K: GGLWEInfos;

    /// Bytes a right operand of `infos` holds, as [`Self::blas_matmul_right_alloc_from_infos`] allocates it.
    fn blas_matmul_right_bytes_of_from_infos<A>(&self, infos: &A) -> usize
    where
        A: BlasMatMulOperandInfos;

    /// A right operand of `infos`, holding all its storage, which [`Self::blas_matmul_prepare_right`] fills: the infos
    /// `blas_matmul_right_operand_layout` gives for a grid, a
    /// context and a hint. Until then it reports `infos` and the batch `0`, which the prepared product rejects.
    ///
    /// # Panics
    ///
    /// As [`Self::blas_matmul_left_alloc_from_infos`].
    fn blas_matmul_right_alloc_from_infos<A>(&self, infos: &A) -> BlasMatMulRight<BE>
    where
        A: BlasMatMulOperandInfos;

    /// Fills `res`, a right operand allocated by [`Self::blas_matmul_right_alloc_from_infos`] from the infos
    /// `blas_matmul_right_operand_layout` gives for `b`, `ctx` and
    /// a hint, from `b` under that hint, as [`Self::blas_matmul_prepare_left`]:
    /// hot, a relabel of `b`; cold, tau of every block and its shifts, each converted once by poulpy-ckks's
    /// `ckks_prepare_right_into`. On a replicated `b` the maps are [`BlasContext::tau_rep`] and
    /// [`BlasContext::rotate_rows_rep`], whose psi^k spend no scale.
    ///
    /// # Metadata
    ///
    /// As [`Self::blas_matmul_prepare_left`], from `b`, as
    /// `blas_matmul_right_operand_layout` gives them: a cold
    /// operand's terms sit at `w_b`, `b.k - p` for a replicated `b`.
    ///
    /// # Errors
    ///
    /// As [`Self::blas_matmul_prepare_left`], with tau and psi^k, from
    /// [`validate_blas_matmul_prepare_right`](crate::layouts::validate_blas_matmul_prepare_right). On a replicated `b`,
    /// psi^k's giant steps sit at `b.k - p`, and a cold preparation's budget error comes when `b`'s budget is below `p`.
    fn blas_matmul_prepare_right<B, K>(
        &self,
        res: &mut BlasMatMulRight<BE>,
        b: &B,
        ctx: &BlasContext<BE>,
        atk: &K,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        B: BlasGrid<BE>,
        K: GetAutomorphismKey<BE> + Sync;

    /// Scratch bytes [`Self::blas_matmul_prepared`] needs into `res` under `ctx`, with a tensor key of layout `tsk` and
    /// rotation keys of layout `atk`, from the prepared operands `a` and `b`, or, before preparing them, from the
    /// layouts `blas_matmul_left_operand_layout` and
    /// `blas_matmul_right_operand_layout` give: the JKLS kernel's
    /// on the terms, and for each hot side the evaluation and conversion of its terms. The right factor's label selects
    /// its terms, so a layout given in its place carries its label.
    ///
    /// # Panics
    ///
    /// May panic on layouts [`validate_blas_matmul_prepared`](crate::layouts::validate_blas_matmul_prepared) rejects:
    /// blocks of a degree other than the module's, operands of different radices, or a hot operand at another `log_d`
    /// than `ctx`'s.
    fn blas_matmul_prepared_tmp_bytes<R, A, B, T, K>(
        &self,
        res: &R,
        a: &A,
        b: &B,
        ctx: &BlasContext<BE>,
        tsk: &T,
        atk: &K,
    ) -> usize
    where
        R: BlasGridInfos,
        A: BlasMatMulOperandInfos,
        B: BlasMatMulOperandInfos,
        T: GGLWEInfos,
        K: GGLWEInfos;

    /// `res` receives the product of the grids `a` and `b` were prepared from, under any hints, with the tensor key
    /// `tsk`; batches of different sizes broadcast as in [`Self::blas_matmul`]. A cold operand brings its terms. The
    /// product evaluates those of a hot one under `ctx`, with the rotation keys `atk`, each converted once, whatever the
    /// context it was prepared under: the maps spend their scale `p` from the operand's budget, so its terms sit at
    /// `w_a` or `w_b`, and the product stamps by the ct x ct rule, at the smaller scale. With every side hot or prepared
    /// under `ctx`, `res` is [`Self::blas_matmul`] of the sources, bit for bit. A cold side takes no rotation key, so of
    /// two cold operands `ctx` and `atk` read nothing. `a` and `b` are unchanged: a cold operand serves any number of
    /// products, and a hot one too, its terms evaluated anew each time.
    ///
    /// Each block of `res` is one JKLS kernel over its `K d` pairs of terms, `K = a.cols()`. On rank 1, with one `base2k`
    /// across the blocks and the tensor key and an offset of at least `base2k` in poulpy-ckks's ct x ct rule, which
    /// `max(w_a, w_b) >= base2k` guarantees, choose `base2k` as [`Self::blas_matmul`] states, `L` the limbs of the
    /// narrower of the terms' widths `w_a` and `w_b`: the kernel adds the products of every pair in the transform
    /// domain before one inverse transform, in the order [`Self::blas_matmul`] states.
    /// `blas_matmul_max_base2k` takes the source grids of the operands, not a
    /// cold operand's term layout. The reference makes a hot side's terms a block row of `a` or a block column of `b` at
    /// a time, each once, and keeps a row (column) only while it serves: beside a cold side, one at a time; of two hot
    /// sides, it holds one side's, as [`Self::blas_matmul`] does. Of two cold operands it makes no term.
    ///
    /// # Metadata
    ///
    /// With `a` and `b` read through their [`BlasMatMulOperandInfos`] and `w_x` the width of the terms of `x`, as above
    /// for a hot `x` and `x.k` for a cold one, every block of `res` is one JKLS kernel over its `K d` pairs of terms,
    /// poulpy-ckks's `ckks_dot_product_prepared` in the reference, so the ct x ct rule on the terms:
    ///
    /// ```text
    /// log_delta    = min(a.log_delta, b.log_delta)
    /// k            = min(res.k, min(w_a - a.log_delta, w_b - b.log_delta) - max(a.log_delta, b.log_delta) + log_delta)
    /// log_sparsity = min(a.log_sparsity, b.log_sparsity)
    /// slots        = a.slots.join(b.slots).join(Real)
    /// canonical    = true
    /// noise        = None
    /// mat_slots    = max(a.mat_slots, b.mat_slots)
    /// ```
    ///
    /// # Errors
    ///
    /// - The implementation runs [`validate_blas_matmul_prepared`](crate::layouts::validate_blas_matmul_prepared)
    ///   before any write, so on its errors `res`, its batch record and the operands are untouched: `InvalidOperand`
    ///   for an operand without a batch record, a grid, shape, layout or radix above 62 it rejects, or a hot operand at
    ///   another `log_d` than `ctx`'s or under a context encoded for another slot count or radix; then, key by key,
    ///   `MissingAutomorphismKey` for a key of a hot side's terms `atk` lacks, at the widths
    ///   [`Self::blas_matmul_prepare_left`] and [`Self::blas_matmul_prepare_right`] state from that side's `k`, or
    ///   `MissingRelinearizationKey` for the tensor key `tsk` lacks at `max(w_a, w_b)`, or `InvalidOperand` for a key
    ///   of another rank than the operands'.
    /// - poulpy-ckks's budget errors come from inside the backend before any write, and leave them untouched too: a hot
    ///   side's `MultiplicationPrecisionUnderflow` when its budget, `k - log_delta`, is below what its terms spend, `2p`,
    ///   or `p` for a replicated `b`, `a`'s first, then the kernel's, `MultiplicationPrecisionUnderflow` when
    ///   `min(w_a - a.log_delta, w_b - b.log_delta)` is below `max(a.log_delta, b.log_delta)`, and
    ///   `InsufficientHomomorphicCapacity` when `res.k` is below `log_delta`.
    /// - Any other `Err` is a backend fault (a validator gap or a device failure), and `res` is then unspecified.
    #[allow(clippy::too_many_arguments)]
    fn blas_matmul_prepared<R, T, K>(
        &self,
        res: &mut R,
        a: &BlasMatMulLeft<BE>,
        b: &BlasMatMulRight<BE>,
        ctx: &BlasContext<BE>,
        tsk: &T,
        atk: &K,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        R: BlasGridMut<BE>,
        T: GetTensorKey<BE> + Sync,
        K: GetAutomorphismKey<BE> + Sync;

    /// Scratch bytes [`Self::blas_matmul_pt_right`] needs into `res` from `a` through `b` with rotation keys of layout `atk`.
    ///
    /// # Panics
    ///
    /// May panic on layouts [`validate_blas_matmul_pt_right`](crate::layouts::validate_blas_matmul_pt_right) rejects:
    /// blocks of a degree other than the module's.
    fn blas_matmul_pt_right_tmp_bytes<R, A, K>(&self, res: &R, a: &A, b: &BlasPtOperand<BE>, atk: &K) -> usize
    where
        R: BlasGridInfos,
        A: BlasGridInfos,
        K: GGLWEInfos;

    /// `res` receives the product of every matrix of `a` with the public matrix `b` holds for its matrix slot.
    ///
    /// Under `Jkls` (the operand's plan), every block of `a` first goes through `b.first()`, once, a block row at a time.
    /// Each block of `res` is one linear transformation over its `K = a.cols()` terms, which share their giant steps: a
    /// block costs the giant-step key switches of one of `b`'s maps, whatever `K`. Each giant step adds the products of
    /// every term with each of its diagonals in the transform domain before its rotation, so choose `base2k` for the
    /// `K g min(L, P)` limb products it adds in an output limb: `g` the most diagonals a giant step of `b`'s block maps
    /// groups, `L` the limbs of their input, `a.k` over `base2k` rounded up (`a.k - p_B` under `Jkls`), and `P` those of a
    /// diagonal, the width `b` was encoded at over `base2k` rounded up. Under `Jkls` the first map adds its own
    /// `g min(L, P)`, with its own `g` and `P`, for one term and `L` the limbs of `a.k`.
    /// `blas_matmul_pt_right_max_base2k` returns the largest radix
    /// `Module::max_base2k` allows for the larger count, counting `L` and `P` at each radix, as for poulpy-core's own
    /// summed linear transformation.
    ///
    /// No circuit privacy: as for every evaluation in this workspace (`SECURITY.md`), `res` may reveal `b` to the holder
    /// of the secret key. Re-randomize `res`, or flood its noise, before returning it when `b` must stay private.
    ///
    /// Several public operands applied to one input share their baby steps only within one operand, so concatenate them
    /// into one operand: the columns of the weights on the right, the rows on the left.
    ///
    /// # Metadata
    ///
    /// Block `(i, j)` of `res` is poulpy-ckks's `ckks_eval_linear_transformation_sum_into` of the `K` terms
    /// `(a_il, b_lj)`, or under `Jkls` `(s_il, b_lj)` for `s_il` the block `a_il` through `b.first()`, labelled, as by
    /// poulpy-ckks's plaintext product, with the smaller of `a`'s label and the largest label the `b.count()` distinct
    /// matrices allow, for `S` the slots of a block:
    ///
    /// ```text
    /// k            = min(res.k, a.k - p_B)       under Diagonal
    ///                min(res.k, a.k - 2 p_B)     under Jkls
    /// log_delta    = a.log_delta
    /// log_sparsity = min(a.log_sparsity, log2(S / (b.count() d^2)))
    /// slots        = a.slots.join(Real)
    /// canonical    = true
    /// noise        = None
    /// mat_slots    = a's batch
    /// ```
    ///
    /// So a result holding several distinct matrices never claims one ([`crate::layouts::replicated`]), even from an
    /// `a` relabelled to one matrix over a larger batch record.
    ///
    /// # Errors
    ///
    /// - The implementation runs [`validate_blas_matmul_pt_right`](crate::layouts::validate_blas_matmul_pt_right)
    ///   before any write, so on its errors `res`, its batch record and `a` are untouched: `InvalidOperand` for a grid,
    ///   shape, layout, operand or count it rejects, then, key by key, `MissingAutomorphismKey` for a key of `b`'s maps
    ///   `atk` lacks: under `Diagonal` their baby steps at `a.k` and their giant steps at the result's `k`; under
    ///   `Jkls` the first map's baby steps at `a.k` and giant steps at `a.k - p_B`, then the blocks' baby steps at
    ///   `a.k - p_B` and giant steps at the result's `k`; or `InvalidOperand` for a key of another rank than `a`'s
    ///   blocks.
    /// - poulpy-ckks's budget errors come from inside the backend before any write, and leave them untouched too:
    ///   `MultiplicationPrecisionUnderflow` when `a`'s budget, `a.k - a.log_delta`, is below `p_B` under `Diagonal`;
    ///   under `Jkls` when it is below `p_B` (the first map), then when it is below `2 p_B` (the blocks' maps); and
    ///   `InsufficientHomomorphicCapacity` when `res.k` is below `a.log_delta`.
    /// - Any other `Err` is a backend fault (a validator gap or a device failure), and `res` is then unspecified.
    fn blas_matmul_pt_right<R, A, K>(
        &self,
        res: &mut R,
        a: &A,
        b: &BlasPtOperand<BE>,
        atk: &K,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        R: BlasGridMut<BE>,
        A: BlasGrid<BE>,
        K: GetAutomorphismKey<BE> + Sync;

    /// Scratch bytes [`Self::blas_matmul_pt_left`] needs into `res` through `b` from `a` with rotation keys of layout
    /// `atk`.
    ///
    /// # Panics
    ///
    /// As [`Self::blas_matmul_pt_right_tmp_bytes`], on layouts
    /// [`validate_blas_matmul_pt_left`](crate::layouts::validate_blas_matmul_pt_left) rejects.
    fn blas_matmul_pt_left_tmp_bytes<R, A, K>(&self, res: &R, b: &BlasPtOperand<BE, Left>, a: &A, atk: &K) -> usize
    where
        R: BlasGridInfos,
        A: BlasGridInfos,
        K: GGLWEInfos;

    /// `res = B A`: `res` receives the product of the public matrix `b` holds for each matrix slot with the matrix of `a`
    /// there. Under `Diagonal` it consumes `b`'s scale alone, under `Jkls` twice, where the same product through
    /// transposes, `transpose(transpose(a) B^T)` by [`Self::blas_matmul_pt_right`], consumes the context's scale twice
    /// more.
    ///
    /// Each block of `res` is one linear transformation over its `K = a.rows()` terms, which share their giant steps: a
    /// block costs the giant-step key switches of one of `b`'s maps, whatever `K`. Each giant step adds the products of
    /// every term with each of its diagonals in the transform domain before its rotation, so choose `base2k` for the
    /// count [`Self::blas_matmul_pt_right`] documents:
    /// `blas_matmul_pt_left_max_base2k` returns the largest radix
    /// `Module::max_base2k` allows for it.
    ///
    /// No circuit privacy: as for every evaluation in this workspace (`SECURITY.md`), `res` may reveal `b` to the holder
    /// of the secret key. Re-randomize `res`, or flood its noise, before returning it when `b` must stay private.
    ///
    /// **Replicated input.** When `a` is replicated ([`crate::layouts::replicated`], read from its label) and `b` holds
    /// them ([`BlasPtOperand::block_replicated`]: `Diagonal`, one public matrix, more than one matrix slot), the blocks'
    /// maps are those `d`-diagonal maps, scheduled at `left_mul`'s giant step so they rotate within the same key set,
    /// in place of the `2d - 1` of [`BlasPtOperand::block`]: about 22 key switches per block instead of 30 at `d = 128`.
    /// The choice is made once per call from `a`'s label, by the validator, the call and
    /// `blas_matmul_pt_left_max_base2k` alike, and changes the bits
    /// against the `2d - 1` maps. Metadata and keys are unchanged, and the scratch query does not read the maps.
    ///
    /// # Metadata
    ///
    /// As [`Self::blas_matmul_pt_right`]: block `(i, j)` of `res` is the summed linear transformation of the `K` terms
    /// `(a_lj, b_il)`, or under `Jkls` `(s_lj, b_il)` for `s_lj` the block `a_lj` through `b.first()` (once, a block
    /// column at a time), labelled as there, and the table there holds.
    ///
    /// # Errors
    ///
    /// As [`Self::blas_matmul_pt_right`], from [`validate_blas_matmul_pt_left`](crate::layouts::validate_blas_matmul_pt_left).
    fn blas_matmul_pt_left<R, A, K>(
        &self,
        res: &mut R,
        b: &BlasPtOperand<BE, Left>,
        a: &A,
        atk: &K,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        R: BlasGridMut<BE>,
        A: BlasGrid<BE>,
        K: GetAutomorphismKey<BE> + Sync;
}
