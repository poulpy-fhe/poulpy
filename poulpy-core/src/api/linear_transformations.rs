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

    /// Scratch bytes required by [`Self::glwe_eval_linear_transformation_unprepared_rhs_into`].
    fn glwe_eval_linear_transformation_unprepared_rhs_tmp_bytes<R, A, B, K>(&self, res: &R, a: &A, pt: &B, key: &K) -> usize
    where
        R: GLWEInfos,
        A: GLWEInfos,
        B: GLWEInfos,
        K: GGLWEInfos;

    /// Scratch bytes required by [`Self::glwe_prepare_linear_transformation_lhs`].
    fn glwe_prepare_linear_transformation_lhs_tmp_bytes<A, K>(&self, a: &A, key: &K) -> usize
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
    fn glwe_prepare_linear_transformation_lhs<A, H, K>(
        &self,
        cache: &mut LinearTransformationBabySteps<BE>,
        a: &A,
        a_effective_k: usize,
        keys: &H,
        key_size: usize,
        scratch: &mut ScratchArena<'_, BE>,
    ) where
        A: GLWEToBackendRef<BE> + GLWEInfos,
        K: GetGaloisElement + GGLWEPreparedToBackendRef<BE> + GGLWEInfos,
        H: GLWEAutomorphismKeyHelper<K, BE>;

    /// Computes `res = M(a)` from the prepared left (`lhs`, the input baby
    /// rotations) and prepared right (`rhs`, the matrix diagonals) caches.
    ///
    /// `lhs` may carry exactly `rhs.baby_steps` or a superset (e.g. the union
    /// of baby rotations needed by several transforms).
    fn glwe_eval_linear_transformation_into<R, H, K>(
        &self,
        cnv_offset: usize,
        res: &mut R,
        lhs: &LinearTransformationBabySteps<BE>,
        rhs: &LinearTransformation<PreparedDiagonal<BE::OwnedBuf, BE>>,
        keys: &H,
        key_size: usize,
        scratch: &mut ScratchArena<'_, BE>,
    ) where
        R: GLWEToBackendMut<BE> + GLWEInfos,
        K: GetGaloisElement + GGLWEPreparedToBackendRef<BE> + GGLWEInfos,
        H: GLWEAutomorphismKeyHelper<K, BE>;

    /// Computes `res = M(a)` from the prepared left cache `lhs` and the
    /// *unprepared* matrix `rhs`, preparing each diagonal on the fly into scratch
    /// rather than from a materialized `LinearTransformation<PreparedDiagonal>`.
    ///
    /// Same result as [`Self::glwe_eval_linear_transformation_into`] with lower
    /// peak memory and higher compute — for memory-bound backends (e.g. GPU).
    fn glwe_eval_linear_transformation_unprepared_rhs_into<R, P, H, K>(
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
        P: GLWEToBackendRef<BE> + GLWEInfos + crate::default::linear_transformation::DiagonalProd<BE>,
        K: GetGaloisElement + GGLWEPreparedToBackendRef<BE> + GGLWEInfos,
        H: GLWEAutomorphismKeyHelper<K, BE>;
}
