use poulpy_ckks::CKKSResult as Result;
use poulpy_core::layouts::{GGLWEInfos, GetAutomorphismKey};
use poulpy_hal::layouts::{Backend, ScratchArena};

use crate::layouts::{BlasGrid, BlasGridInfos, BlasGridMut};

/// Operations moving whole matrices between matrix slots, on every block of a grid. `res` records the batch of `a`
/// ([`BlasGridInfos::mat_slots`]), the matrices these operations move and sum over.
///
/// In the `# Metadata` sections, `x.k` is the width of the blocks of `x`, for `res` the width they have before the
/// call, and `a`'s batch is its record, or the batch its blocks' label proves when it has none
/// ([`grid_batch`](crate::layouts::grid_batch)). A copy is poulpy-ckks's `ckks_copy`: it keeps the canonical flag of
/// `a` when `res.k >= a.k` and `res` has room for every limb of a non-canonical `a`, and otherwise normalizes, which
/// sets it.
pub trait BlasMatInterOps<BE: Backend> {
    /// Scratch bytes [`Self::blas_rotate`] needs into `res` from `a` with rotation keys of layout `atk`.
    ///
    /// # Panics
    ///
    /// May panic on layouts [`validate_blas_rotate`](crate::layouts::validate_blas_rotate) rejects: blocks of a degree
    /// other than the module's.
    fn blas_rotate_tmp_bytes<R, A, K>(&self, res: &R, a: &A, atk: &K) -> usize
    where
        R: BlasGridInfos,
        A: BlasGridInfos,
        K: GGLWEInfos;

    /// `res` receives `a` with matrix slot `s` holding matrix slot `(s + r) mod mat_slots` of `a`.
    ///
    /// # Metadata
    ///
    /// Every block of `res` is poulpy-ckks's `ckks_rotate_into` of the matching block of `a` by `r d^2` slots, which is
    /// a copy when `r d^2` is a multiple of the slot count:
    ///
    /// ```text
    /// k            = min(res.k, a.k)
    /// log_delta    = a.log_delta
    /// log_sparsity = a.log_sparsity
    /// slots        = a.slots
    /// canonical    = true, or the copy's flag
    /// noise        = None
    /// mat_slots    = a's batch
    /// ```
    ///
    /// # Errors
    ///
    /// - The implementation runs [`validate_blas_rotate`](crate::layouts::validate_blas_rotate) before any write, so on
    ///   its errors `res`, its batch record and `a` are untouched: `InvalidOperand` for a grid, shape or layout it
    ///   rejects, then `MissingAutomorphismKey` for the rotation key `atk` lacks at `a.k`, or `InvalidOperand` when
    ///   that key is of another rank than `a`'s blocks.
    /// - poulpy-ckks's budget error, `InsufficientHomomorphicCapacity` when `res.k` is below `a.log_delta`, comes from
    ///   inside the backend before any write, and leaves them untouched too.
    /// - Any other `Err` is a backend fault (a validator gap or a device failure), and `res` is then unspecified.
    fn blas_rotate<R, A, K>(&self, res: &mut R, a: &A, r: i64, atk: &K, scratch: &mut ScratchArena<'_, BE>) -> Result<()>
    where
        R: BlasGridMut<BE>,
        A: BlasGrid<BE>,
        K: GetAutomorphismKey<BE> + Sync;

    /// Scratch bytes [`Self::blas_sum`] needs into `res` from `a` with rotation keys of layout `atk`.
    ///
    /// # Panics
    ///
    /// As [`Self::blas_rotate_tmp_bytes`], on layouts [`validate_blas_sum`](crate::layouts::validate_blas_sum) rejects.
    fn blas_sum_tmp_bytes<R, A, K>(&self, res: &R, a: &A, atk: &K) -> usize
    where
        R: BlasGridInfos,
        A: BlasGridInfos,
        K: GGLWEInfos;

    /// Every matrix slot of `res` receives the sum of all matrix slots of `a`.
    ///
    /// # Metadata
    ///
    /// Every block of `res` starts as a copy of the matching block of `a`, then adds its own rotation by `2^i d^2` slots
    /// for each `i < log2(mat_slots)` in turn, by poulpy-ckks's `ckks_rotate_into` and `ckks_add_assign`:
    ///
    /// ```text
    /// k            = min(res.k, a.k)
    /// log_delta    = a.log_delta
    /// log_sparsity = a.log_sparsity
    /// slots        = a.slots
    /// canonical    = false, or the copy's flag on a batch of one matrix
    /// noise        = None
    /// mat_slots    = a's batch
    /// ```
    ///
    /// # Errors
    ///
    /// - The implementation runs [`validate_blas_sum`](crate::layouts::validate_blas_sum) before any write, so on its
    ///   errors `res`, its batch record and `a` are untouched: `InvalidOperand` for a grid, shape or layout it rejects;
    ///   then, key by key, `MissingAutomorphismKey` for a rotation key `atk` lacks at `min(res.k, a.k)`, or
    ///   `InvalidOperand` for one of another rank than `a`'s blocks.
    /// - poulpy-ckks's budget error, `InsufficientHomomorphicCapacity` from the copy when `res.k` is below `a.log_delta`,
    ///   comes from inside the backend before any write, and leaves them untouched too.
    /// - Any other `Err` is a backend fault (a validator gap or a device failure), and `res` is then unspecified.
    fn blas_sum<R, A, K>(&self, res: &mut R, a: &A, atk: &K, scratch: &mut ScratchArena<'_, BE>) -> Result<()>
    where
        R: BlasGridMut<BE>,
        A: BlasGrid<BE>,
        K: GetAutomorphismKey<BE> + Sync;
}
