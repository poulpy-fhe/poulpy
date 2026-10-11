use poulpy_ckks::CKKSResult as Result;
use poulpy_core::layouts::{GGLWEInfos, GetAutomorphismKey};
use poulpy_hal::layouts::{Backend, ScratchArena};

use crate::layouts::{BlasGrid, BlasGridInfos, BlasGridMut};

/// Operations moving whole matrices between matrix slots, on every block of a grid. `res` records the batch of `a`
/// ([`BlasGridInfos::mat_slots`]), the matrices these operations move and sum over.
pub trait BlasMatInterOps<BE: Backend> {
    /// Scratch bytes [`Self::blas_rotate`] needs into `res` from `a` with rotation keys of layout `atk`.
    fn blas_rotate_tmp_bytes<R, A, K>(&self, res: &R, a: &A, atk: &K) -> usize
    where
        R: BlasGridInfos,
        A: BlasGridInfos,
        K: GGLWEInfos;

    /// `res` receives `a` with matrix slot `s` holding matrix slot `(s + r) mod mat_slots` of `a`.
    ///
    /// # Errors
    ///
    /// The errors of [`validate_blas_rotate`](crate::reference::validate_blas_rotate), raised before the backend runs,
    /// so `res` and the inputs are untouched: `InvalidOperand` for a grid, shape or layout it rejects,
    /// `MissingAutomorphismKey` for a key `atk` lacks at its width. Then poulpy-ckks's budget errors, such as
    /// `InsufficientHomomorphicCapacity` when `res` is narrower than `a`'s scale, raised before any write.
    fn blas_rotate<R, A, K>(&self, res: &mut R, a: &A, r: i64, atk: &K, scratch: &mut ScratchArena<'_, BE>) -> Result<()>
    where
        R: BlasGridMut<BE>,
        A: BlasGrid<BE>,
        K: GetAutomorphismKey<BE> + Sync;

    /// Scratch bytes [`Self::blas_sum`] needs into `res` from `a` with rotation keys of layout `atk`.
    fn blas_sum_tmp_bytes<R, A, K>(&self, res: &R, a: &A, atk: &K) -> usize
    where
        R: BlasGridInfos,
        A: BlasGridInfos,
        K: GGLWEInfos;

    /// Every matrix slot of `res` receives the sum of all matrix slots of `a`.
    ///
    /// # Errors
    ///
    /// The errors of [`validate_blas_sum`](crate::reference::validate_blas_sum), raised before the backend runs, so
    /// `res` and the inputs are untouched: `InvalidOperand` for a grid, shape, layout or term count it rejects,
    /// `MissingAutomorphismKey` for a key `atk` lacks at its width. Then poulpy-ckks's budget errors, such as
    /// `InsufficientHomomorphicCapacity` when `res` is narrower than `a`'s scale, raised before any write.
    fn blas_sum<R, A, K>(&self, res: &mut R, a: &A, atk: &K, scratch: &mut ScratchArena<'_, BE>) -> Result<()>
    where
        R: BlasGridMut<BE>,
        A: BlasGrid<BE>,
        K: GetAutomorphismKey<BE> + Sync;
}
