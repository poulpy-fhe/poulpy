use poulpy_ckks::CKKSResult as Result;
use poulpy_core::layouts::{GGLWEInfos, GetAutomorphismKey};
use poulpy_hal::layouts::{Backend, ScratchArena};

use crate::layouts::{BlasContext, BlasGrid, BlasGridInfos, BlasGridMut};

/// The same map applied to every matrix of the batch, across the blocks of a grid. `res` records the batch of `a`
/// ([`BlasGridInfos::mat_slots`]).
pub trait BlasMatIntraOps<BE: Backend> {
    /// Scratch bytes [`Self::blas_transpose`] needs into `res` from `a` with the maps of `ctx` and rotation keys of layout
    /// `atk`.
    fn blas_transpose_tmp_bytes<R, A, K>(&self, res: &R, a: &A, ctx: &BlasContext<BE>, atk: &K) -> usize
    where
        R: BlasGridInfos,
        A: BlasGridInfos,
        K: GGLWEInfos;

    /// `res` receives the transpose of every matrix of `a`.
    ///
    /// # Errors
    ///
    /// The errors of [`validate_blas_transpose`](crate::reference::validate_blas_transpose), raised before the backend
    /// runs, so `res` and the inputs are untouched: `InvalidOperand` for a grid, shape, layout or context it rejects,
    /// `MissingAutomorphismKey` for a key `atk` lacks at its width. Then poulpy-ckks's budget errors, such as
    /// `InsufficientHomomorphicCapacity` when `res` is narrower than `a`'s scale, raised before any write.
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
    fn blas_rotate_rows_tmp_bytes<R, A, K>(&self, res: &R, a: &A, ctx: &BlasContext<BE>, atk: &K) -> usize
    where
        R: BlasGridInfos,
        A: BlasGridInfos,
        K: GGLWEInfos;

    /// `res` receives every matrix of `a` with row `r` replaced by row `r + k` modulo the matrix height.
    ///
    /// # Errors
    ///
    /// The errors of [`validate_blas_rotate_rows`](crate::reference::validate_blas_rotate_rows), raised before the
    /// backend runs, so `res` and the inputs are untouched: `InvalidOperand` for a grid, shape, layout or context it
    /// rejects, `MissingAutomorphismKey` for a key `atk` lacks at its width. Then poulpy-ckks's budget errors, such as
    /// `InsufficientHomomorphicCapacity` when `res` is narrower than `a`'s scale, raised before any write.
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
    fn blas_rotate_cols_tmp_bytes<R, A, K>(&self, res: &R, a: &A, ctx: &BlasContext<BE>, atk: &K) -> usize
    where
        R: BlasGridInfos,
        A: BlasGridInfos,
        K: GGLWEInfos;

    /// `res` receives every matrix of `a` with column `c` replaced by column `c + k` modulo the matrix width.
    ///
    /// # Errors
    ///
    /// The errors of [`validate_blas_rotate_cols`](crate::reference::validate_blas_rotate_cols), raised before the
    /// backend runs, so `res` and the inputs are untouched: `InvalidOperand` for a grid, shape, layout or context it
    /// rejects, `MissingAutomorphismKey` for a key `atk` lacks at its width. Then poulpy-ckks's budget errors, such as
    /// `InsufficientHomomorphicCapacity` when `res` is narrower than `a`'s scale, raised before any write.
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
