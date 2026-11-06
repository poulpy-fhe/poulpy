use poulpy_ckks::CKKSResult as Result;
use poulpy_core::layouts::{GGLWEInfos, GetAutomorphismKey, GetTensorKey};
use poulpy_hal::layouts::{Backend, ScratchArena};

use crate::layouts::{BlasContextPrepared, BlasGrid, BlasGridInfos, BlasGridMut};

/// The same map applied to every matrix of the batch, across the blocks of a grid. `res` records the batch of `a`
/// ([`BlasGridInfos::mat_slots`]).
///
/// In the `# Metadata` sections, `x.k` is the width of the blocks of `x`, for `res` the width they have before the
/// call, `p` is the scale of the maps of `ctx`, that of the `coeffs_meta` it was encoded at, and `a`'s batch is its
/// record, or the batch its blocks' label proves when it has none ([`grid_batch`](crate::layouts::grid_batch)). A map
/// of `ctx` is evaluated by poulpy-ckks's `ckks_linear_transformation_eval_into`, then labelled as its source; its
/// slots are `Real`. A copy is poulpy-ckks's `ckks_copy`: it keeps the canonical flag of `a` when `res.k >= a.k` and
/// `res` has room for every limb of a non-canonical `a`, and otherwise normalizes, which sets it.
pub trait BlasMatIntraOps<BE: Backend> {
    /// Scratch bytes [`Self::blas_transpose`] needs into `res` from `a` with the maps of `ctx` and rotation keys of layout
    /// `keys`.
    ///
    /// # Panics
    ///
    /// May panic on layouts [`validate_blas_transpose`](crate::layouts::validate_blas_transpose) rejects: blocks of a
    /// degree other than the module's, or `rows * cols` overflowing `usize`.
    fn blas_transpose_tmp_bytes<R, A, K>(&self, res: &R, a: &A, ctx: &BlasContextPrepared<BE>, keys: &K) -> usize
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
    ///   context it rejects, then, key by key, `MissingAutomorphismKey` for a key of the map `keys` lacks, its baby
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
        ctx: &BlasContextPrepared<BE>,
        keys: &K,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        R: BlasGridMut<BE>,
        A: BlasGrid<BE>,
        K: GetAutomorphismKey<BE> + GetTensorKey<BE> + Sync;

    /// Scratch bytes [`Self::blas_rotate_rows`] needs into `res` from `a` with the maps of `ctx` and rotation keys of
    /// layout `keys`.
    ///
    /// # Panics
    ///
    /// As [`Self::blas_transpose_tmp_bytes`], on layouts
    /// [`validate_blas_rotate_rows`](crate::layouts::validate_blas_rotate_rows) rejects.
    fn blas_rotate_rows_tmp_bytes<R, A, K>(&self, res: &R, a: &A, ctx: &BlasContextPrepared<BE>, keys: &K) -> usize
    where
        R: BlasGridInfos,
        A: BlasGridInfos,
        K: GGLWEInfos;

    /// `res` receives every matrix of `a` with row `r` replaced by row `r + k` modulo the matrix height.
    ///
    /// # Metadata
    ///
    /// With `k mod (a.rows() d) = q d + k'` and block rows counted modulo `a.rows()`, block `(i, j)` of `res` is: for
    /// `k' = 0`, a copy of block `(i + q, j)` of `a`; on a grid one block high, poulpy-ckks's `ckks_rotate_into` of block
    /// `(i, j)` by `k' d` slots when `a` is replicated ([`replicated`](crate::layouts::replicated)), where the rotation
    /// wraps inside a block, and psi^k' of `ctx` evaluated from it otherwise; on a taller grid, poulpy-ckks's
    /// `ckks_add_into` of the two half maps of psi^k', the first evaluated from block `(i + q, j)` and the second from
    /// block `(i + q + 1, j)`, each into a block of `res`'s layout:
    ///
    /// ```text
    ///                k' = 0             one block high,      one block high        otherwise
    ///                                   replicated
    /// k            = min(res.k, a.k)    min(res.k, a.k)      min(res.k, a.k - p)   min(res.k, a.k - p)
    /// log_delta    = a.log_delta        a.log_delta          a.log_delta           a.log_delta
    /// log_sparsity = a.log_sparsity     a.log_sparsity       a.log_sparsity        a.log_sparsity
    /// slots        = a.slots            a.slots              a.slots.join(Real)    a.slots.join(Real)
    /// canonical    = the copy's flag    true                 true                  false
    /// noise        = None               None                 None                  None
    /// mat_slots    = a's batch          a's batch            a's batch             a's batch
    /// ```
    ///
    /// # Errors
    ///
    /// - The implementation runs [`validate_blas_rotate_rows`](crate::layouts::validate_blas_rotate_rows) before any
    ///   write, so on its errors `res`, its batch record and `a` are untouched: `InvalidOperand` for a grid, shape,
    ///   layout or context it rejects, then, key by key, `MissingAutomorphismKey` for a key of the maps the shift
    ///   evaluates `keys` lacks, at the widths [`Self::blas_transpose`] looks its keys up at, or for the key of the
    ///   rotation at `min(res.k, a.k)`, or `InvalidOperand` for one of another rank than `a`'s blocks. A copy takes no key.
    /// - poulpy-ckks's budget errors come from inside the backend before any write, and leave them untouched too: those
    ///   of [`Self::blas_transpose`], or for `k' = 0` the copy's and for the rotation its
    ///   `InsufficientHomomorphicCapacity` when `res.k` is below `a.log_delta`.
    /// - Any other `Err` is a backend fault (a validator gap or a device failure), and `res` is then unspecified.
    fn blas_rotate_rows<R, A, K>(
        &self,
        k: usize,
        res: &mut R,
        a: &A,
        ctx: &BlasContextPrepared<BE>,
        keys: &K,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        R: BlasGridMut<BE>,
        A: BlasGrid<BE>,
        K: GetAutomorphismKey<BE> + GetTensorKey<BE> + Sync;

    /// Scratch bytes [`Self::blas_rotate_cols`] needs into `res` from `a` with the maps of `ctx` and rotation keys of
    /// layout `keys`.
    ///
    /// # Panics
    ///
    /// As [`Self::blas_transpose_tmp_bytes`], on layouts
    /// [`validate_blas_rotate_cols`](crate::layouts::validate_blas_rotate_cols) rejects.
    fn blas_rotate_cols_tmp_bytes<R, A, K>(&self, res: &R, a: &A, ctx: &BlasContextPrepared<BE>, keys: &K) -> usize
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
    /// block `(i, j)` on a grid one block wide, replicated or not, since a rotation by `k'` slots crosses into the next
    /// row, and otherwise the sum of the first half map of block `(i, j + q)` and the second of block `(i, j + q + 1)`.
    /// The table of [`Self::blas_rotate_rows`] holds without its replicated column, with "one block wide" in place of
    /// "one block high".
    ///
    /// # Errors
    ///
    /// As [`Self::blas_rotate_rows`], from [`validate_blas_rotate_cols`](crate::layouts::validate_blas_rotate_cols).
    fn blas_rotate_cols<R, A, K>(
        &self,
        k: usize,
        res: &mut R,
        a: &A,
        ctx: &BlasContextPrepared<BE>,
        keys: &K,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        R: BlasGridMut<BE>,
        A: BlasGrid<BE>,
        K: GetAutomorphismKey<BE> + GetTensorKey<BE> + Sync;

    /// Scratch bytes [`Self::blas_sum_rows`] needs into `res` from `a` with the maps of `ctx` and rotation keys of
    /// layout `keys`.
    ///
    /// # Panics
    ///
    /// As [`Self::blas_transpose_tmp_bytes`], on layouts [`validate_blas_sum_rows`](crate::layouts::validate_blas_sum_rows)
    /// rejects.
    fn blas_sum_rows_tmp_bytes<R, A, K>(&self, res: &R, a: &A, ctx: &BlasContextPrepared<BE>, keys: &K) -> usize
    where
        R: BlasGridInfos,
        A: BlasGridInfos,
        K: GGLWEInfos;

    /// Every entry of every matrix of `a` receives its column's sum over the matrix rows: `res[r][c] = sum_r' a[r'][c]`.
    ///
    /// # Metadata
    ///
    /// For each block column `j`, a block `u` of `a`'s layout is poulpy-ckks's `ckks_add_many` of the blocks of the
    /// column. Its inner sum over `d` batches of `d` slots, a sum rather than a mean, adds its own rotation by `2^i d` slots for
    /// each `i < log2(d)` in turn and leaves the column sums in row 0, and on a replicated `a`
    /// ([`replicated`](crate::layouts::replicated)), whose rotations by multiples of `d` wrap inside a block, in every
    /// row: block `(0, j)` of `res` is then `ckks_inner_sum` of `u`, at `res`'s width. Otherwise `u` takes its
    /// `ckks_inner_sum_assign`, and block `(0, j)` of `res` is the projection on row 0 of `ctx`
    /// ([`BlasContextPrepared::first_row`]) evaluated from `u`, then its `ckks_replicate_assign` over `d` batches of `d` slots,
    /// which adds its own rotation by `-2^i d` slots for each `i < log2(d)` from the last down. The other blocks of the
    /// column are copies of block `(0, j)`:
    ///
    /// ```text
    ///                replicated                otherwise
    /// k            = min(res.k, a.k)           min(res.k, a.k - p)
    /// log_delta    = a.log_delta               a.log_delta
    /// log_sparsity = a.log_sparsity            a.log_sparsity
    /// slots        = a.slots                   a.slots.join(Real)
    /// canonical    = true                      true
    /// noise        = None                      None
    /// mat_slots    = a's batch                 a's batch
    /// ```
    ///
    /// The sum of the blocks normalizes in place at the bound the Lazy sums section of
    /// `libs/poulpy-blas/docs/blas-contracts.md` states. A flag-clear block of `a` holds at most two canonical digits per
    /// coefficient, the sum of two canonical values: a caller obligation, unchecked.
    ///
    /// # Errors
    ///
    /// - The implementation runs [`validate_blas_sum_rows`](crate::layouts::validate_blas_sum_rows) before any write, so
    ///   on its errors `res`, its batch record and `a` are untouched: `InvalidOperand` for a grid, shape, layout or
    ///   context it rejects, then, key by key, `MissingAutomorphismKey` for a rotation key `keys` lacks, the tree's at
    ///   `a.k`, or at `min(res.k, a.k)` on a replicated `a`, and the broadcast's at the result's `k`, or
    ///   `InvalidOperand` for one of another rank than `a`'s blocks.
    /// - poulpy-ckks's budget errors come from inside the backend before any write, and leave them untouched too: on a
    ///   replicated `a`, the inner sum's `InsufficientHomomorphicCapacity` when `res.k` is below `a.log_delta`; otherwise
    ///   those of [`Self::blas_transpose`].
    /// - Any other `Err` is a backend fault (a validator gap or a device failure), and `res` is then unspecified.
    fn blas_sum_rows<R, A, K>(
        &self,
        res: &mut R,
        a: &A,
        ctx: &BlasContextPrepared<BE>,
        keys: &K,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        R: BlasGridMut<BE>,
        A: BlasGrid<BE>,
        K: GetAutomorphismKey<BE> + GetTensorKey<BE> + Sync;

    /// Scratch bytes [`Self::blas_sum_cols`] needs into `res` from `a` with the maps of `ctx` and rotation keys of
    /// layout `keys`.
    ///
    /// # Panics
    ///
    /// As [`Self::blas_transpose_tmp_bytes`], on layouts [`validate_blas_sum_cols`](crate::layouts::validate_blas_sum_cols)
    /// rejects.
    fn blas_sum_cols_tmp_bytes<R, A, K>(&self, res: &R, a: &A, ctx: &BlasContextPrepared<BE>, keys: &K) -> usize
    where
        R: BlasGridInfos,
        A: BlasGridInfos,
        K: GGLWEInfos;

    /// Every entry of every matrix of `a` receives its row's sum over the matrix columns: `res[r][c] = sum_c' a[r][c']`.
    ///
    /// # Metadata
    ///
    /// As [`Self::blas_sum_rows`] along the block rows, with the inner sum and the replicate over `d` batches of one
    /// slot, rotations by `2^i` and `-2^i` slots, and the projection on column 0 ([`BlasContextPrepared::first_col`]) between
    /// them, on every grid: a rotation by `2^i` slots crosses into the next row, replicated or not, and the broadcast
    /// never does, since `2^(i+1) <= d`. For each block row `i`, `u` sums the blocks of the row, block `(i, 0)` of `res`
    /// receives the projection and the broadcast, and the other blocks of the row are copies of it. The column
    /// "otherwise" of the table of [`Self::blas_sum_rows`] holds, and so does its caller obligation on flag-clear blocks.
    ///
    /// # Errors
    ///
    /// As [`Self::blas_sum_rows`] off a replicated `a`, from [`validate_blas_sum_cols`](crate::layouts::validate_blas_sum_cols).
    fn blas_sum_cols<R, A, K>(
        &self,
        res: &mut R,
        a: &A,
        ctx: &BlasContextPrepared<BE>,
        keys: &K,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        R: BlasGridMut<BE>,
        A: BlasGrid<BE>,
        K: GetAutomorphismKey<BE> + GetTensorKey<BE> + Sync;
}
