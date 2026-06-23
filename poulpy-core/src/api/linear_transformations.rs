//! Safe, user-facing traits for the GLWE linear transformation (BSGS).
//!
//! Dispatch follows the same `api -> oep -> delegates <- default` pattern as the
//! other operation families: this module defines the abstract trait, the backend
//! extension point lives in [`crate::oep::LinearTransformationImpl`], the blanket
//! wiring is in the (private) `delegates` module, and the reference algorithms
//! are in [`crate::default::linear_transformation`].

#![allow(clippy::too_many_arguments)]

use poulpy_hal::layouts::{Backend, ScratchArena};

use crate::layouts::{
    GGLWEInfos, GLWEAutomorphismKeyHelper, GLWEInfos, GLWEToBackendMut, GLWEToBackendRef, GetGaloisElement, LWEInfos,
    LinearTransformation,
    prepared::{GGLWEPreparedToBackendRef, LinearTransformationBabySteps, PreparedDiagonal},
};

/// GLWE-level setup and evaluation of a resident (prepared) linear
/// transformation (`LinearTransformation<PreparedDiagonal<…>>`).
///
/// The API is split into three phases (allocate / populate / evaluate); see
/// [`LinearTransformation::alloc_prepared`] and
/// [`LinearTransformationBabySteps::alloc`] for the allocation half.
pub trait GLWELinearTransformations<BE: Backend> {
    /// Scratch bytes required by [`Self::glwe_eval_linear_transformation_into`].
    fn glwe_eval_linear_transformation_tmp_bytes<R, A, B, K>(&self, res: &R, a: &A, pt: &B, key: &K) -> usize
    where
        R: GLWEInfos,
        A: GLWEInfos,
        B: GLWEInfos,
        K: GGLWEInfos;

    /// Scratch bytes required by [`Self::glwe_eval_linear_transformation_into`] when
    /// the RHS is *streamed* (an unprepared plaintext-diagonal `P`): the streamed
    /// inner product additionally holds one resident `CnvPVecR` diagonal slot.
    fn glwe_eval_linear_transformation_unprepared_rhs_tmp_bytes<R, A, B, K>(&self, res: &R, a: &A, pt: &B, key: &K) -> usize
    where
        R: GLWEInfos,
        A: GLWEInfos,
        B: GLWEInfos,
        K: GGLWEInfos;

    /// Scratch bytes required by [`Self::glwe_prepare_linear_transformation_baby_steps`].
    fn glwe_prepare_linear_transformation_baby_steps_tmp_bytes<A, K>(&self, a: &A, key: &K) -> usize
    where
        A: GLWEInfos,
        K: GGLWEInfos;

    /// Scratch bytes required by [`Self::glwe_prepare_linear_transformation_rhs`].
    fn glwe_prepare_linear_transformation_rhs_tmp_bytes<P>(&self, pt_infos: &P) -> usize
    where
        P: LWEInfos;

    /// Encodes every diagonal of `lt` into the matching pre-allocated `CnvPVecR`
    /// slot of `prepared`.
    ///
    /// `prepared` must have been sized via
    /// [`LinearTransformation::alloc_prepared`] for the same BSGS schedule as
    /// `lt`. Performs zero `CnvPVecR` allocations.
    fn glwe_prepare_linear_transformation_rhs<P>(
        &self,
        prepared: &mut LinearTransformation<PreparedDiagonal<BE::OwnedBuf, BE>>,
        lt: &LinearTransformation<P>,
        scratch: &mut ScratchArena<'_, BE>,
    ) where
        P: GLWEToBackendRef<BE> + GLWEInfos;

    /// Fills a pre-allocated baby-step cache with the rotated, prepared versions
    /// of `a`.
    ///
    /// `cache` must have been sized via [`LinearTransformationBabySteps::alloc`].
    /// Performs zero `CnvPVecL` allocations.
    fn glwe_prepare_linear_transformation_baby_steps<A, H, K>(
        &self,
        cache: &mut LinearTransformationBabySteps<BE>,
        a: &A,
        a_k: usize,
        keys: &H,
        key_size: usize,
        scratch: &mut ScratchArena<'_, BE>,
    ) where
        A: GLWEToBackendRef<BE> + GLWEInfos,
        K: GetGaloisElement + GGLWEPreparedToBackendRef<BE> + GGLWEInfos,
        H: GLWEAutomorphismKeyHelper<K, BE>;

    /// Computes `res = M(a)` from the prepared left cache `lhs` (the input baby
    /// rotations) and the matrix `rhs`, generic over the diagonal representation
    /// `P` (see [`DiagonalProd`](crate::default::linear_transformation::DiagonalProd)):
    ///
    /// - `P = PreparedDiagonal` (resident): the diagonals are already in the
    ///   convolution domain, so each giant step is one fused accumulation.
    /// - a plaintext diagonal (streamed): each diagonal is prepared on the fly
    ///   into scratch instead of from a materialized
    ///   `LinearTransformation<PreparedDiagonal>` — lower peak memory, higher
    ///   compute, for memory-bound backends (e.g. GPU).
    ///
    /// Both paths share this single evaluator and give the same result. `lhs` may
    /// carry exactly `rhs.baby_steps` or a superset (e.g. the union of baby
    /// rotations needed by several transforms).
    fn glwe_eval_linear_transformation_into<R, P, H, K>(
        &self,
        cnv_offset: usize,
        res: &mut R,
        lhs: &LinearTransformationBabySteps<BE>,
        rhs: &LinearTransformation<P>,
        keys: &H,
        key_size: usize,
        scratch: &mut ScratchArena<'_, BE>,
    ) where
        R: GLWEToBackendMut<BE> + GLWEInfos,
        P: crate::default::linear_transformation::DiagonalProd<BE>,
        K: GetGaloisElement + GGLWEPreparedToBackendRef<BE> + GGLWEInfos,
        H: GLWEAutomorphismKeyHelper<K, BE>;
}
