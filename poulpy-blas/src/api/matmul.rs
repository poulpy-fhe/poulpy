use poulpy_ckks::CKKSResult as Result;
use poulpy_core::layouts::{GGLWEInfos, GetAutomorphismKey, GetTensorKey};
use poulpy_hal::layouts::ScratchArena;

use crate::{
    layouts::{BlasContext, BlasGrid, BlasGridInfos, BlasGridMut, BlasMatMulLeft, BlasMatMulRight, BlasPtOperand},
    oep::BlasMatMulImpl,
};

/// Matrix products of every matrix of the batch, across the blocks of a grid. A product records the larger batch of its
/// operands ([`BlasGridInfos::mat_slots`]); a prepared operand and the product by public matrices record their input's.
pub trait BlasMatMulOps<BE: BlasMatMulImpl> {
    /// Scratch bytes [`Self::blas_matmul`] needs into `res` from `a` and `b` with the maps of `ctx`, a tensor key of
    /// layout `tsk` and rotation keys of layout `atk`.
    ///
    /// # Panics
    ///
    /// May panic on layouts [`validate_blas_matmul`](crate::reference::validate_blas_matmul) rejects: blocks of a degree
    /// other than the module's, or factors of different radices.
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
    /// # Errors
    ///
    /// The errors of [`validate_blas_matmul`](crate::reference::validate_blas_matmul), raised before the backend runs, so
    /// `res` and the inputs are untouched: `InvalidOperand` for a grid, shape, layout, context or radix above 62 it
    /// rejects, `MissingAutomorphismKey` for a key `atk` lacks at its width, `MissingRelinearizationKey` for one `tsk`
    /// lacks at the operands' term width. Then poulpy-ckks's budget errors, such as `InsufficientHomomorphicCapacity`,
    /// before any write to `res`.
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
    /// May panic on blocks of a degree other than the module's, which the operation rejects.
    fn blas_matmul_prepare_left_tmp_bytes<A, K>(&self, a: &A, ctx: &BlasContext<BE>, atk: &K) -> usize
    where
        A: BlasGridInfos,
        K: GGLWEInfos;

    /// The left operand of [`Self::blas_matmul_prepared`] from `a`, with every rotation of the product: sigma of every
    /// block and its shifts. It records `a`'s batch, and its [`BlasGridInfos`] report `a`'s dimensions and the layout
    /// [`blas_matmul_term_layout`](crate::reference::blas_matmul_term_layout) gives.
    ///
    /// # Errors
    ///
    /// The errors of [`validate_blas_matmul_prepare_left`](crate::reference::validate_blas_matmul_prepare_left), raised
    /// before the backend runs: `InvalidOperand` for a grid, layout or context it rejects, `MissingAutomorphismKey` for a
    /// key `atk` lacks at its width. Then poulpy-ckks's budget errors, such as `InsufficientHomomorphicCapacity` when
    /// `a` is narrower than twice the maps' scale.
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
    /// # Errors
    ///
    /// As [`Self::blas_matmul_prepare_left`], from
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
    /// # Errors
    ///
    /// The errors of [`validate_blas_matmul_prepared`](crate::reference::validate_blas_matmul_prepared), raised before
    /// the backend runs, so `res` and the operands are untouched: `InvalidOperand` for an operand without a batch record,
    /// or a grid, shape, layout or radix above 62 it rejects, `MissingRelinearizationKey` for a tensor key `tsk` lacks at
    /// the operands' term width. Then poulpy-ckks's budget errors, such as `InsufficientHomomorphicCapacity`, before any
    /// write to `res`.
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
    /// May panic on blocks of a degree other than the module's, which the operation rejects.
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
    /// # Errors
    ///
    /// The errors of [`validate_blas_matmul_pt`](crate::reference::validate_blas_matmul_pt), raised before the backend
    /// runs, so `res` and the inputs are untouched: `InvalidOperand` for a grid, shape, layout, operand or count it
    /// rejects, `MissingAutomorphismKey` for a key `atk` lacks at its width. Then poulpy-ckks's budget errors, such as
    /// `InsufficientHomomorphicCapacity` when `res` is narrower than `a`'s scale, raised before any write.
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
}
