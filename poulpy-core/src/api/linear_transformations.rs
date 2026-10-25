//! Safe, user-facing traits for the GLWE linear transformation (BSGS).
//!
//! Dispatch follows the same `api -> oep -> delegates <- reference` pattern as the
//! other operation families: this module defines the abstract trait, the backend
//! extension point lives in [`crate::oep::LinearTransformationImpl`], the blanket
//! wiring is in the (private) `delegates` module, and the reference algorithms
//! are in [`crate::reference::linear_transformation`].

#![allow(clippy::too_many_arguments)]

use poulpy_hal::layouts::{Backend, ScratchArena};

use crate::layouts::{
    GGLWEInfos, GLWEInfos, GLWEToBackendMut, GLWEToBackendRef, GetAutomorphismKey, LWEInfos, LinearTransformation,
    LinearTransformationPlan,
    prepared::{LinearTransformationBabySteps, LinearTransformationPrepared, PreparedDiagonal},
};

/// GLWE-level setup and evaluation of a resident (prepared) linear
/// transformation (`LinearTransformation<PreparedDiagonal<…>>`).
///
/// The API is split into three phases (allocate / populate / evaluate); see
/// [`LinearTransformation::alloc_prepared`] and
/// [`LinearTransformationBabySteps::alloc`] for the allocation half.
pub trait GLWELinearTransformations<BE: Backend> {
    /// Scratch bytes required by [`Self::glwe_eval_linear_transformation_into`] for a transform
    /// with the schedule `plan` ([`LinearTransformation::index`]), which gives the giant-step
    /// tree of [`BsgsRotations::PowersOfTwo`](crate::layouts::BsgsRotations::PowersOfTwo); every
    /// plan with a key per rotation gives the same bytes. `key` is the widest key the transform
    /// switches with.
    fn glwe_eval_linear_transformation_tmp_bytes<R, A, B, K>(
        &self,
        res: &R,
        a: &A,
        pt: &B,
        plan: &LinearTransformationPlan,
        key: &K,
    ) -> usize
    where
        R: GLWEInfos,
        A: GLWEInfos,
        B: GLWEInfos,
        K: GGLWEInfos;

    /// Scratch bytes required by [`Self::glwe_eval_linear_transformation_into`] when
    /// the RHS is *streamed* (an unprepared plaintext-diagonal `P`): the streamed
    /// inner product additionally holds one resident `CnvPVecR` diagonal slot. `plan`
    /// as in [`Self::glwe_eval_linear_transformation_tmp_bytes`].
    fn glwe_eval_linear_transformation_unprepared_rhs_tmp_bytes<R, A, B, K>(
        &self,
        res: &R,
        a: &A,
        pt: &B,
        plan: &LinearTransformationPlan,
        key: &K,
    ) -> usize
    where
        R: GLWEInfos,
        A: GLWEInfos,
        B: GLWEInfos,
        K: GGLWEInfos;

    /// Scratch bytes required by [`Self::glwe_eval_linear_transformation_sum_into`]:
    /// [`Self::glwe_eval_linear_transformation_tmp_bytes`] plus one giant-step product buffer,
    /// `plan` the schedule of the first term.
    fn glwe_eval_linear_transformation_sum_tmp_bytes<R, A, B, K>(
        &self,
        res: &R,
        a: &A,
        pt: &B,
        plan: &LinearTransformationPlan,
        key: &K,
    ) -> usize
    where
        R: GLWEInfos,
        A: GLWEInfos,
        B: GLWEInfos,
        K: GGLWEInfos;

    /// Scratch bytes required by [`Self::glwe_prepare_linear_transformation_baby_steps`] for a
    /// cache of the baby steps of `plan`, which gives the doubling of
    /// [`BsgsRotations::PowersOfTwo`](crate::layouts::BsgsRotations::PowersOfTwo); every plan with
    /// a key per rotation gives the same bytes.
    fn glwe_prepare_linear_transformation_baby_steps_tmp_bytes<A, K>(
        &self,
        a: &A,
        plan: &LinearTransformationPlan,
        key: &K,
    ) -> usize
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
    /// `lt`. Performs zero `CnvPVecR` allocations. The proxy must also carry the
    /// diagonals' degree, which may be below the ring degree;
    /// `lt.first_diagonal_plaintext()` is the proxy every caller should pass.
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
    fn glwe_prepare_linear_transformation_baby_steps<A, H>(
        &self,
        cache: &mut LinearTransformationBabySteps<BE>,
        a: &A,
        keys: &H,
        scratch: &mut ScratchArena<'_, BE>,
    ) where
        A: GLWEToBackendRef<BE> + GLWEInfos,
        H: GetAutomorphismKey<BE>;

    /// Computes `res = M(a)` from the prepared left cache `lhs` (the input baby
    /// rotations) and the matrix `rhs`, generic over the diagonal representation
    /// `P` (see [`DiagonalProd`](crate::layouts::DiagonalProd)):
    ///
    /// - `P = PreparedDiagonal` (resident): the diagonals are already in the
    ///   convolution domain, so each giant step is one fused accumulation.
    /// - a plaintext diagonal (streamed): each diagonal is prepared on the fly
    ///   into scratch instead of from a materialized
    ///   `LinearTransformation<PreparedDiagonal>`, lower peak memory, higher
    ///   compute, for implementations constrained by memory bandwidth.
    ///
    /// Both paths share this single evaluator and give the same result. `lhs` may
    /// carry exactly `rhs.baby_steps` or a superset (e.g. the union of baby
    /// rotations needed by several transforms).
    ///
    /// # Panics
    ///
    /// Before writing `res`, if `rhs` has no diagonal or a giant step
    /// without one, if a giant step's first diagonal is not as wide as the
    /// transform's first diagonal, or if `lhs` lacks a baby rotation that `rhs`
    /// reads. The other diagonals are not checked: every diagonal of a
    /// transform shares one width, scale and degree, as the encoder produces
    /// them, and preparing asserts the degree.
    fn glwe_eval_linear_transformation_into<R, P, H>(
        &self,
        cnv_offset: usize,
        res: &mut R,
        lhs: &LinearTransformationBabySteps<BE>,
        rhs: &LinearTransformation<P>,
        keys: &H,
        scratch: &mut ScratchArena<'_, BE>,
    ) where
        R: GLWEToBackendMut<BE> + GLWEInfos,
        P: crate::layouts::DiagonalProd<BE>,
        H: GetAutomorphismKey<BE>;

    /// Computes `res = Σ_t M_t(a_t)`, where `babies[t]` is the baby cache of
    /// `a_t` and `lts[t]` the resident transform `M_t`, all sharing one giant
    /// schedule.
    ///
    /// Each giant step's products are summed over the terms in the DFT domain
    /// and rotated once, so the sum costs the giant key switches of one
    /// transform, and `cnv_offset` applies to every term. One term is
    /// [`Self::glwe_eval_linear_transformation_into`], bit for bit. A giant
    /// step's products over every term accumulate in one unnormalized buffer:
    /// pick `base2k` for that many products (`Module::max_base2k`).
    ///
    /// # Panics
    ///
    /// Before writing `res`, if `babies` is empty or `lts` is not as
    /// long, if a transform has no diagonal or a giant step without one, or if
    /// a cache lacks a baby rotation that its transform reads, and unless every term shares the first one's giant schedule, source layout,
    /// and first diagonal's layout, width and scale, and every giant step's
    /// first diagonal has that width. The other diagonals are not checked:
    /// every diagonal of a transform shares one width, scale and degree, as the
    /// encoder produces them, and preparing asserts the degree.
    fn glwe_eval_linear_transformation_sum_into<R, H>(
        &self,
        cnv_offset: usize,
        res: &mut R,
        babies: &[&LinearTransformationBabySteps<BE>],
        lts: &[&LinearTransformationPrepared<BE>],
        keys: &H,
        scratch: &mut ScratchArena<'_, BE>,
    ) where
        R: GLWEToBackendMut<BE> + GLWEInfos,
        H: GetAutomorphismKey<BE>;
}
