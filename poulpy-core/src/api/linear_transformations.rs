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
    GGLWEInfos, GLWEAutomorphismKeyHelper, GLWEInfos, LinearTransformation, GLWEToBackendMut, GLWEToBackendRef, GetGaloisElement,
    LWEInfos,
    prepared::{GGLWEPreparedToBackendRef, LinearTransformationLhsPrepared, LinearTransformationRhsPrepared},
};

/// GLWE-level setup and evaluation of a [`LinearTransformationRhsPrepared`].
///
/// The API is split into three phases (allocate / populate / evaluate); see
/// [`LinearTransformationRhsPrepared::alloc`] and
/// [`LinearTransformationLhsPrepared::alloc`] for the allocation half.
pub trait GLWELinearTransformations<BE: Backend> {
    /// Scratch bytes required by [`Self::glwe_eval_linear_transformation_into`].
    fn glwe_eval_linear_transformation_tmp_bytes<R, A, B, K>(&self, res: &R, a: &A, pt: &B, key: &K) -> usize
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
    /// [`LinearTransformationRhsPrepared::alloc`] for the same BSGS schedule as
    /// `lt`. Performs zero `CnvPVecR` allocations.
    fn glwe_prepare_linear_transformation_rhs<P>(
        &self,
        prepared: &mut LinearTransformationRhsPrepared<BE>,
        lt: &LinearTransformation<P>,
        scratch: &mut ScratchArena<'_, BE>,
    ) where
        P: GLWEToBackendRef<BE> + GLWEInfos;

    /// Fills a pre-allocated baby-step cache with the rotated, prepared versions
    /// of `a`.
    ///
    /// `cache` must have been sized via [`LinearTransformationLhsPrepared::alloc`].
    /// Performs zero `CnvPVecL` allocations.
    fn glwe_prepare_linear_transformation_lhs<A, H, K>(
        &self,
        cache: &mut LinearTransformationLhsPrepared<BE>,
        a: &A,
        a_effective_k: usize,
        key_size: usize,
        keys: &H,
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
        res: &mut R,
        lhs: &LinearTransformationLhsPrepared<BE>,
        rhs: &LinearTransformationRhsPrepared<BE>,
        cnv_offset: usize,
        key_size: usize,
        keys: &H,
        scratch: &mut ScratchArena<'_, BE>,
    ) where
        R: GLWEToBackendMut<BE> + GLWEInfos,
        K: GetGaloisElement + GGLWEPreparedToBackendRef<BE> + GGLWEInfos,
        H: GLWEAutomorphismKeyHelper<K, BE>;
}
