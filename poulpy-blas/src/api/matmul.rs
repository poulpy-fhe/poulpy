use poulpy_ckks::CKKSResult as Result;
use poulpy_core::layouts::{GGLWEInfos, GetAutomorphismKey, GetTensorKey};
use poulpy_hal::layouts::ScratchArena;

use crate::{
    layouts::{BlasContext, BlasGrid, BlasGridInfos, BlasGridMut, BlasMatMulLeft, BlasMatMulRight, BlasPtOperand},
    oep::BlasMatMulImpl,
    reference::Left,
};

/// Matrix products of every matrix of the batch, across the blocks of a grid. A product records the larger batch of its
/// operands ([`BlasGridInfos::mat_slots`]); a prepared operand and the products by public matrices record their
/// input's.
///
/// In the `# Metadata` sections, `x.k` is the width of the blocks of `x`, for `res` the width they have before the
/// call and for a prepared operand that of its terms; `p` is the scale of the maps of `ctx`, that of the `coeffs_meta`
/// it was encoded at, and `p_B` that of the public operand `b`; an input's batch is its record, or the batch its
/// blocks' label proves when it has none ([`grid_batch`](crate::layouts::grid_batch)). The maps, of `ctx` and of
/// public operands, have `Real` slots. The products of two encrypted grids follow poulpy-ckks's ct x ct rule
/// ([`CKKSMulOps`](poulpy_ckks::api::CKKSMulOps)) on the terms of their preparations, which sit at `w_a = a.k - 2p`
/// and `w_b = b.k - 2p`.
pub trait BlasMatMulOps<BE: BlasMatMulImpl> {
    /// Scratch bytes [`Self::blas_matmul`] needs into `res` from `a` and `b` with the maps of `ctx`, a tensor key of
    /// layout `tsk` and rotation keys of layout `atk`.
    ///
    /// # Panics
    ///
    /// May panic on layouts [`validate_blas_matmul`](crate::reference::validate_blas_matmul) rejects: blocks of a degree
    /// other than the module's, factors of different radices, a factor at another `log_d` than `ctx`'s, or
    /// `rows * cols` overflowing `usize`.
    fn blas_matmul_tmp_bytes<R, A, B, T, K>(&self, res: &R, a: &A, b: &B, ctx: &BlasContext<BE>, tsk: &T, atk: &K) -> usize
    where
        R: BlasGridInfos,
        A: BlasGridInfos,
        B: BlasGridInfos,
        T: GGLWEInfos,
        K: GGLWEInfos;

    /// `res` receives the product of every matrix of `a` with the matching matrix of `b`: `a` and `b` prepared with the
    /// rotation keys `atk`, then their prepared product with the tensor key `tsk`, bit for bit.
    ///
    /// Batches of different sizes broadcast: with `m_a` and `m_b` matrices in `a` and `b`, matrix slot `s` of `res` holds
    /// `A_(s mod m_a) B_(s mod m_b)`, so `res` holds the larger batch, at the smaller `log_sparsity` of the two.
    ///
    /// Each block of `res` is one JKLS kernel over its `K d` pairs of terms, `K = a.cols()`, relinearized once. On rank
    /// 1, with one `base2k` across the blocks and the tensor key and an offset of at least `base2k` in poulpy-ckks's ct x
    /// ct rule, which `max(w_a, w_b) >= base2k` guarantees, the kernel adds the products of every pair in the transform
    /// domain before one inverse transform, so choose `base2k` for `K d` times the products of one pair, up to `2 L` per
    /// output limb, `L` the limbs of the narrower terms, `min(w_a, w_b)` over `base2k` rounded up: with
    /// `Module::max_base2k(n, 2 L K d, ..)`, as for poulpy-core's `glwe_mul_relinearize_sum`. An FFT backend adds them in
    /// floating point, in the order of the pairs `(a_il[k], b_lj[k])`, `l`-major: ascending `l`, then ascending `k`
    /// within each `l`. That order is then part of the bits.
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
    /// For scales equal to `delta`, `k = min(res.k, min(a.k, b.k) - 2p - delta)`.
    ///
    /// # Errors
    ///
    /// - [`validate_blas_matmul`](crate::reference::validate_blas_matmul) runs before the backend, so on its errors
    ///   `res`, its batch record and the factors are untouched: `InvalidOperand` for a grid, shape, layout, context or
    ///   radix above 62 it rejects, then, key by key, `MissingAutomorphismKey` for a key of the preparations `atk` lacks,
    ///   at the widths [`Self::blas_matmul_prepare_left`] states, or `MissingRelinearizationKey` for the tensor key `tsk`
    ///   lacks at `max(w_a, w_b)`, or `InvalidOperand` for a key of another rank than the factors' blocks.
    /// - poulpy-ckks's budget errors come from inside the backend before any write to `res`, and leave it untouched
    ///   too: the preparations' `MultiplicationPrecisionUnderflow` when the budget of `a` or `b`, its `k - log_delta`,
    ///   is below `2p`, then the product's, `MultiplicationPrecisionUnderflow` when
    ///   `min(w_a - a.log_delta, w_b - b.log_delta)` is below `max(a.log_delta, b.log_delta)` and
    ///   `InsufficientHomomorphicCapacity` when `res.k` is below `min(a.log_delta, b.log_delta)`.
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

    /// Scratch bytes [`Self::blas_matmul_prepare_left`] needs from `a` with the maps of `ctx` and rotation keys of layout
    /// `atk`.
    ///
    /// # Panics
    ///
    /// May panic on layouts [`validate_blas_matmul_prepare_left`](crate::reference::validate_blas_matmul_prepare_left)
    /// rejects: blocks of a degree other than the module's, a grid at another `log_d` than `ctx`'s, or `rows * cols`
    /// overflowing `usize`.
    fn blas_matmul_prepare_left_tmp_bytes<A, K>(&self, a: &A, ctx: &BlasContext<BE>, atk: &K) -> usize
    where
        A: BlasGridInfos,
        K: GGLWEInfos;

    /// The left operand of [`Self::blas_matmul_prepared`] from `a`, with every rotation of the product: sigma of every
    /// block and its shifts. It records `a`'s batch, and its [`BlasGridInfos`] report `a`'s dimensions and the layout
    /// [`blas_matmul_term_layout`](crate::reference::blas_matmul_term_layout) gives.
    ///
    /// # Metadata
    ///
    /// The operand's [`BlasGridInfos`] report `a`'s `log_d`, `rows` and `cols`, and the layout of every term:
    ///
    /// ```text
    /// k            = a.k - 2p
    /// log_delta    = a.log_delta
    /// log_sparsity = a.log_sparsity
    /// slots        = a.slots.join(Real)
    /// mat_slots    = Some(a's batch)
    /// ```
    ///
    /// Nothing else is specified: the backend holds the terms in its own representation, without a canonical flag or a
    /// noise tag to read.
    ///
    /// # Errors
    ///
    /// - [`validate_blas_matmul_prepare_left`](crate::reference::validate_blas_matmul_prepare_left) runs before the
    ///   backend: `InvalidOperand` for a grid, layout or context it rejects, then, key by key, `MissingAutomorphismKey`
    ///   for a key `atk` lacks: sigma's baby steps at `a.k` and giant steps at `a.k - p`, each phi^k's baby steps at
    ///   `a.k - p` and giant steps at `a.k - 2p`; or `InvalidOperand` for one of another rank than `a`'s blocks.
    /// - poulpy-ckks's budget error comes from inside the backend: `MultiplicationPrecisionUnderflow` when `a`'s budget,
    ///   `a.k - a.log_delta`, is below `2p`.
    /// - Any other `Err` is a backend fault (a validator gap or a device failure).
    ///
    /// A preparation writes only the operand it returns, so no error changes `a`.
    fn blas_matmul_prepare_left<A, K>(
        &self,
        a: &A,
        ctx: &BlasContext<BE>,
        atk: &K,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<BlasMatMulLeft<BE>>
    where
        A: BlasGrid<BE>,
        K: GetAutomorphismKey<BE> + Sync;

    /// Scratch bytes [`Self::blas_matmul_prepare_right`] needs from `b` with the maps of `ctx` and rotation keys of
    /// layout `atk`.
    ///
    /// # Panics
    ///
    /// As [`Self::blas_matmul_prepare_left_tmp_bytes`].
    fn blas_matmul_prepare_right_tmp_bytes<B, K>(&self, b: &B, ctx: &BlasContext<BE>, atk: &K) -> usize
    where
        B: BlasGridInfos,
        K: GGLWEInfos;

    /// The right operand of [`Self::blas_matmul_prepared`] from `b`: tau of every block and its shifts, as
    /// [`Self::blas_matmul_prepare_left`].
    ///
    /// # Metadata
    ///
    /// As [`Self::blas_matmul_prepare_left`], from `b`.
    ///
    /// # Errors
    ///
    /// As [`Self::blas_matmul_prepare_left`], with tau and psi^k, from
    /// [`validate_blas_matmul_prepare_right`](crate::reference::validate_blas_matmul_prepare_right).
    fn blas_matmul_prepare_right<B, K>(
        &self,
        b: &B,
        ctx: &BlasContext<BE>,
        atk: &K,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<BlasMatMulRight<BE>>
    where
        B: BlasGrid<BE>,
        K: GetAutomorphismKey<BE> + Sync;

    /// Scratch bytes [`Self::blas_matmul_prepared`] needs into `res` with a tensor key of layout `tsk`, from the prepared
    /// operands `a` and `b`, or, before preparing them, from the layouts
    /// [`blas_matmul_term_layout`](crate::reference::blas_matmul_term_layout) gives.
    ///
    /// # Panics
    ///
    /// May panic on layouts [`validate_blas_matmul_prepared`](crate::reference::validate_blas_matmul_prepared) rejects:
    /// blocks of a degree other than the module's, or operands of different radices.
    fn blas_matmul_prepared_tmp_bytes<R, A, B, T>(&self, res: &R, a: &A, b: &B, tsk: &T) -> usize
    where
        R: BlasGridInfos,
        A: BlasGridInfos,
        B: BlasGridInfos,
        T: GGLWEInfos;

    /// `res` receives the product of the grids `a` and `b` were prepared from, as [`Self::blas_matmul`] gives it, with the
    /// tensor key alone; batches of different sizes broadcast as there. `a` and `b` are unchanged, so they serve any
    /// number of products.
    ///
    /// Each block of `res` is one JKLS kernel over its `K d` pairs of terms, `K = a.cols()`. On rank 1, with one `base2k`
    /// across the blocks and the tensor key and an offset of at least `base2k` in poulpy-ckks's ct x ct rule, which
    /// `max(a.k, b.k) >= base2k` guarantees, choose `base2k` as [`Self::blas_matmul`] states, `L` the limbs of the
    /// narrower of the terms' widths `a.k` and `b.k`: the kernel adds the products of every pair in the transform domain
    /// before one inverse transform, in the order [`Self::blas_matmul`] states.
    ///
    /// # Metadata
    ///
    /// With `a` and `b` read through their [`BlasGridInfos`], the layout of their terms, every block of `res` is one JKLS
    /// kernel over its `K d` pairs of terms, poulpy-ckks's `ckks_dot_product_ct` in the reference, so the ct x ct rule on
    /// the terms:
    ///
    /// ```text
    /// log_delta    = min(a.log_delta, b.log_delta)
    /// k            = min(res.k, min(a.k - a.log_delta, b.k - b.log_delta) - max(a.log_delta, b.log_delta) + log_delta)
    /// log_sparsity = min(a.log_sparsity, b.log_sparsity)
    /// slots        = a.slots.join(b.slots)
    /// canonical    = true
    /// noise        = None
    /// mat_slots    = max(a.mat_slots, b.mat_slots)
    /// ```
    ///
    /// # Errors
    ///
    /// - [`validate_blas_matmul_prepared`](crate::reference::validate_blas_matmul_prepared) runs before the backend, so
    ///   on its errors `res`, its batch record and the operands are untouched: `InvalidOperand` for an operand without a
    ///   batch record, or a grid, shape, layout or radix above 62 it rejects, then `MissingRelinearizationKey` for the
    ///   tensor key `tsk` lacks at `max(a.k, b.k)`, or `InvalidOperand` when it is of another rank than the operands'
    ///   terms.
    /// - poulpy-ckks's budget errors come from inside the backend before any write, and leave them untouched too:
    ///   `MultiplicationPrecisionUnderflow` when `min(a.k - a.log_delta, b.k - b.log_delta)` is below
    ///   `max(a.log_delta, b.log_delta)`, and `InsufficientHomomorphicCapacity` when `res.k` is below `log_delta`.
    /// - Any other `Err` is a backend fault (a validator gap or a device failure), and `res` is then unspecified.
    fn blas_matmul_prepared<R, T>(
        &self,
        res: &mut R,
        a: &BlasMatMulLeft<BE>,
        b: &BlasMatMulRight<BE>,
        tsk: &T,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        R: BlasGridMut<BE>,
        T: GetTensorKey<BE> + Sync;

    /// Scratch bytes [`Self::blas_matmul_pt`] needs into `res` from `a` through `b` with rotation keys of layout `atk`.
    ///
    /// # Panics
    ///
    /// May panic on layouts [`validate_blas_matmul_pt`](crate::reference::validate_blas_matmul_pt) rejects: blocks of a
    /// degree other than the module's.
    fn blas_matmul_pt_tmp_bytes<R, A, K>(&self, res: &R, a: &A, b: &BlasPtOperand<BE>, atk: &K) -> usize
    where
        R: BlasGridInfos,
        A: BlasGridInfos,
        K: GGLWEInfos;

    /// `res` receives the product of every matrix of `a` with the public matrix `b` holds for its matrix slot.
    ///
    /// Each block of `res` is one linear transformation over its `K = a.cols()` terms, which share their giant steps: a
    /// block costs the giant-step key switches of one of `b`'s maps, whatever `K`. Each giant step adds `K` times its
    /// diagonals' products in the transform domain before its rotation, so choose `base2k` for that many products,
    /// with `Module::max_base2k(n, K x the diagonals of a giant step, ..)`, as for poulpy-core's own linear
    /// transformation.
    ///
    /// No circuit privacy: as for every evaluation in this workspace (`SECURITY.md`), `res` may reveal `b` to the holder
    /// of the secret key. Re-randomize `res`, or flood its noise, before returning it when `b` must stay private.
    ///
    /// # Metadata
    ///
    /// Block `(i, j)` of `res` is poulpy-ckks's `ckks_eval_linear_transformation_sum_into` of the `K` terms
    /// `(a_il, b_lj)`, labelled as `a`:
    ///
    /// ```text
    /// k            = min(res.k, a.k - p_B)
    /// log_delta    = a.log_delta
    /// log_sparsity = a.log_sparsity
    /// slots        = a.slots.join(Real)
    /// canonical    = true
    /// noise        = None
    /// mat_slots    = a's batch
    /// ```
    ///
    /// # Errors
    ///
    /// - [`validate_blas_matmul_pt`](crate::reference::validate_blas_matmul_pt) runs before the backend, so on its
    ///   errors `res`, its batch record and `a` are untouched: `InvalidOperand` for a grid, shape, layout, operand or
    ///   count it rejects, then, key by key, `MissingAutomorphismKey` for a key of `b`'s maps `atk` lacks, their baby
    ///   steps at `a.k` and their giant steps at the result's `k`, or `InvalidOperand` for one of another rank than `a`'s
    ///   blocks.
    /// - poulpy-ckks's budget errors come from inside the backend before any write, and leave them untouched too:
    ///   `MultiplicationPrecisionUnderflow` when `a`'s budget, `a.k - a.log_delta`, is below `p_B`, and
    ///   `InsufficientHomomorphicCapacity` when `res.k` is below `a.log_delta`.
    /// - Any other `Err` is a backend fault (a validator gap or a device failure), and `res` is then unspecified.
    fn blas_matmul_pt<R, A, K>(
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
    /// As [`Self::blas_matmul_pt_tmp_bytes`], on layouts
    /// [`validate_blas_matmul_pt_left`](crate::reference::validate_blas_matmul_pt_left) rejects.
    fn blas_matmul_pt_left_tmp_bytes<R, A, K>(&self, res: &R, b: &BlasPtOperand<BE, Left>, a: &A, atk: &K) -> usize
    where
        R: BlasGridInfos,
        A: BlasGridInfos,
        K: GGLWEInfos;

    /// `res = B A`: `res` receives the product of the public matrix `b` holds for each matrix slot with the matrix of `a`
    /// there. It consumes `b`'s scale alone, where the same product through transposes, `transpose(transpose(a) B^T)` by
    /// [`Self::blas_matmul_pt`], consumes the context's scale twice more.
    ///
    /// Each block of `res` is one linear transformation over its `K = a.rows()` terms, which share their giant steps: a
    /// block costs the giant-step key switches of one of `b`'s maps, whatever `K`. Each giant step adds `K` times its
    /// diagonals' products in the transform domain before its rotation, so choose `base2k` for that many products,
    /// with `Module::max_base2k(n, K x the diagonals of a giant step, ..)`, as for poulpy-core's own linear
    /// transformation.
    ///
    /// No circuit privacy: as for every evaluation in this workspace (`SECURITY.md`), `res` may reveal `b` to the holder
    /// of the secret key. Re-randomize `res`, or flood its noise, before returning it when `b` must stay private.
    ///
    /// # Metadata
    ///
    /// As [`Self::blas_matmul_pt`]: block `(i, j)` of `res` is the summed linear transformation of the `K` terms
    /// `(a_lj, b_il)`, labelled as `a`, and the table there holds.
    ///
    /// # Errors
    ///
    /// As [`Self::blas_matmul_pt`], from [`validate_blas_matmul_pt_left`](crate::reference::validate_blas_matmul_pt_left).
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
