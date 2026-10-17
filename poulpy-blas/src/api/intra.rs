use poulpy_ckks::CKKSResult as Result;
use poulpy_core::layouts::{GGLWEInfos, GetAutomorphismKey};
use poulpy_hal::layouts::{Backend, ScratchArena};

use crate::layouts::{BlasContext, BlasGrid, BlasGridInfos, BlasGridMut};

/// The same map applied to every matrix of the batch, across the blocks of a grid. `res` records the batch of `a`
/// ([`BlasGridInfos::mat_slots`]).
///
/// In the `# Metadata` sections, `x.k` is the width of the blocks of `x`, for `res` the width they have before the
/// call, `p` is the scale of the maps of `ctx`, that of the `coeffs_meta` it was encoded at, and `a`'s batch is its
/// record, or the batch its blocks' label proves when it has none ([`grid_batch`](crate::layouts::grid_batch)). A map
/// of `ctx` is evaluated by poulpy-ckks's `ckks_eval_linear_transformation_into`, then labelled as its source; its
/// slots are `Real`. A copy is poulpy-ckks's `ckks_copy`: it keeps the canonical flag of `a` when `res.k >= a.k` and
/// `res` has room for every limb of a non-canonical `a`, and otherwise normalizes, which sets it.
pub trait BlasMatIntraOps<BE: Backend> {
    /// Scratch bytes [`Self::blas_transpose`] needs into `res` from `a` with the maps of `ctx` and rotation keys of layout
    /// `atk`.
    ///
    /// # Panics
    ///
    /// May panic on layouts [`validate_blas_transpose`](crate::layouts::validate_blas_transpose) rejects: blocks of a
    /// degree other than the module's, or `rows * cols` overflowing `usize`.
    fn blas_transpose_tmp_bytes<R, A, K>(&self, res: &R, a: &A, ctx: &BlasContext<BE>, atk: &K) -> usize
    where
        R: BlasGridInfos,
        A: BlasGridInfos,
        K: GGLWEInfos;

    /// `res` receives the transpose of every matrix of `a`.
    ///
    /// # Metadata
    ///
    /// Block `(j, i)` of `res` is the transpose map of `ctx` evaluated from block `(i, j)` of `a`:
    ///
    /// ```text
    /// k            = min(res.k, a.k - p)
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
    /// - The implementation runs [`validate_blas_transpose`](crate::layouts::validate_blas_transpose) before any write,
    ///   so on its errors `res`, its batch record and `a` are untouched: `InvalidOperand` for a grid, shape, layout or
    ///   context it rejects, then, key by key, `MissingAutomorphismKey` for a key of the map `atk` lacks, its baby
    ///   steps at `a.k` and its giant steps at the result's `k`, or `InvalidOperand` for one of another rank than `a`'s
    ///   blocks.
    /// - poulpy-ckks's budget errors come from inside the backend before any write, and leave them untouched too:
    ///   `MultiplicationPrecisionUnderflow` when `a`'s budget, `a.k - a.log_delta`, is below `p`, and
    ///   `InsufficientHomomorphicCapacity` when `res.k` is below `a.log_delta`.
    /// - Any other `Err` is a backend fault (a validator gap or a device failure), and `res` is then unspecified.
    fn blas_transpose<R, A, K>(
        &self,
        res: &mut R,
        a: &A,
        ctx: &BlasContext<BE>,
        atk: &K,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        R: BlasGridMut<BE>,
        A: BlasGrid<BE>,
        K: GetAutomorphismKey<BE> + Sync;

    /// Scratch bytes [`Self::blas_rotate_rows`] needs into `res` from `a` with the maps of `ctx` and rotation keys of layout
    /// `atk`.
    ///
    /// # Panics
    ///
    /// As [`Self::blas_transpose_tmp_bytes`], on layouts
    /// [`validate_blas_rotate_rows`](crate::layouts::validate_blas_rotate_rows) rejects.
    fn blas_rotate_rows_tmp_bytes<R, A, K>(&self, res: &R, a: &A, ctx: &BlasContext<BE>, atk: &K) -> usize
    where
        R: BlasGridInfos,
        A: BlasGridInfos,
        K: GGLWEInfos;

    /// `res` receives every matrix of `a` with row `r` replaced by row `r + k` modulo the matrix height.
    ///
    /// # Metadata
    ///
    /// With `k mod (a.rows() d) = q d + k'` and block rows counted modulo `a.rows()`, block `(i, j)` of `res` is: for
    /// `k' = 0`, a copy of block `(i + q, j)` of `a`; on a grid one block high, psi^k' of `ctx` evaluated from block
    /// `(i, j)`; otherwise, poulpy-ckks's `ckks_add_into` of the two half maps of psi^k', the first evaluated from block
    /// `(i + q, j)` and the second from block `(i + q + 1, j)`, each into a block of `res`'s layout:
    ///
    /// ```text
    ///                k' = 0             one block high        otherwise
    /// k            = min(res.k, a.k)    min(res.k, a.k - p)   min(res.k, a.k - p)
    /// log_delta    = a.log_delta        a.log_delta           a.log_delta
    /// log_sparsity = a.log_sparsity     a.log_sparsity        a.log_sparsity
    /// slots        = a.slots            a.slots.join(Real)    a.slots.join(Real)
    /// canonical    = the copy's flag    true                  false
    /// noise        = None               None                  None
    /// mat_slots    = a's batch          a's batch             a's batch
    /// ```
    ///
    /// # Errors
    ///
    /// - The implementation runs [`validate_blas_rotate_rows`](crate::layouts::validate_blas_rotate_rows) before any
    ///   write, so on its errors `res`, its batch record and `a` are untouched: `InvalidOperand` for a grid, shape,
    ///   layout or context it rejects, then, key by key, `MissingAutomorphismKey` for a key of the maps the shift
    ///   evaluates `atk` lacks, at the widths [`Self::blas_transpose`] looks its keys up at, or `InvalidOperand` for
    ///   one of another rank than `a`'s blocks. A copy takes no key.
    /// - poulpy-ckks's budget errors come from inside the backend before any write, and leave them untouched too: those
    ///   of [`Self::blas_transpose`], or for `k' = 0` the copy's `InsufficientHomomorphicCapacity` when `res.k` is below
    ///   `a.log_delta`.
    /// - Any other `Err` is a backend fault (a validator gap or a device failure), and `res` is then unspecified.
    fn blas_rotate_rows<R, A, K>(
        &self,
        res: &mut R,
        a: &A,
        k: usize,
        ctx: &BlasContext<BE>,
        atk: &K,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        R: BlasGridMut<BE>,
        A: BlasGrid<BE>,
        K: GetAutomorphismKey<BE> + Sync;

    /// Scratch bytes [`Self::blas_rotate_cols`] needs into `res` from `a` with the maps of `ctx` and rotation keys of layout
    /// `atk`.
    ///
    /// # Panics
    ///
    /// As [`Self::blas_transpose_tmp_bytes`], on layouts
    /// [`validate_blas_rotate_cols`](crate::layouts::validate_blas_rotate_cols) rejects.
    fn blas_rotate_cols_tmp_bytes<R, A, K>(&self, res: &R, a: &A, ctx: &BlasContext<BE>, atk: &K) -> usize
    where
        R: BlasGridInfos,
        A: BlasGridInfos,
        K: GGLWEInfos;

    /// `res` receives every matrix of `a` with column `c` replaced by column `c + k` modulo the matrix width.
    ///
    /// # Metadata
    ///
    /// As [`Self::blas_rotate_rows`] along the block columns, with phi^k' and its two half maps: with
    /// `k mod (a.cols() d) = q d + k'`, block `(i, j)` of `res` is a copy of block `(i, j + q)` for `k' = 0`, phi^k' of
    /// block `(i, j)` on a grid one block wide, and otherwise the sum of the first half map of block `(i, j + q)` and the
    /// second of block `(i, j + q + 1)`. Its table holds, a grid one block wide in place of one block high.
    ///
    /// # Errors
    ///
    /// As [`Self::blas_rotate_rows`], from [`validate_blas_rotate_cols`](crate::layouts::validate_blas_rotate_cols).
    fn blas_rotate_cols<R, A, K>(
        &self,
        res: &mut R,
        a: &A,
        k: usize,
        ctx: &BlasContext<BE>,
        atk: &K,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        R: BlasGridMut<BE>,
        A: BlasGrid<BE>,
        K: GetAutomorphismKey<BE> + Sync;
}
