//! CKKS-facing API for the homomorphic linear transformation (a matrix–vector
//! product over the slots), evaluated with the baby-step / giant-step (BSGS)
//! decomposition.
//!
//! The transformation itself is the scheme-agnostic GLWE-level engine in
//! [`poulpy_core`] (`LinearTransformation` / `GLWELinearTransformations`); this layer
//! re-exports the data type under CKKS names and adds the scale-aware entry point.
//! All `log_delta` / `log_budget` accounting lives here — the core engine only
//! receives base2k-level alignment integers. See
//! [`docs/lt_bsgs.md`](https://github.com/poulpy-fhe/poulpy/blob/main/docs/lt_bsgs.md).
//!
//! # Typical flow
//!
//! ```ignore
//! // setup, once per transform / per input shape
//! let mut prepared = PreparedLinearTransformationRhs::alloc(module, &layout, &pt_proxy);
//! module.ckks_prepare_linear_transformation_rhs(&mut prepared, &lt, &mut scratch);
//! let mut babies = PreparedLinearTransformationLhs::alloc(module, prepared.baby_steps(), &ct);
//!
//! // per evaluation
//! module.ckks_prepare_linear_transformation_lhs(&mut babies, &ct, &atks, &mut scratch)?;
//! module.ckks_eval_prepared_linear_transformation_into(&mut dst, &ct, &prepared, &babies, &atks, &mut scratch)?;
//! ```

use anyhow::Result;
use poulpy_core::layouts::{
    GGLWEInfos, GGLWEPreparedToBackendRef, GLWEAutomorphismKeyHelper, GLWEToBackendMut, GLWEToBackendRef, GetGaloisElement,
    LWEInfos, prepared::GLWEAutomorphismKeyPreparedToBackendRef,
};
use poulpy_hal::layouts::{Backend, ScratchArena};

use crate::{CKKSCtBounds, SetCKKSInfos};

pub use poulpy_core::{
    LinearTransformation as LinearTransformation, LinearTransformationDiagonal as Diagonal,
    LinearTransformationGiantStep as GiantStep, LinearTransformationPlan as LinearTransformationIndex,
    LinearTransformationLhsPrepared as PreparedLinearTransformationLhs,
    LinearTransformationRhsPrepared as PreparedLinearTransformationRhs,
    LinearTransformationRhsGiantStepPrepared as PreparedGiantStep, LinearTransformationLayout, LinearTransformationStrategy,
    optimal_bsgs_giant_step,
};

/// Homomorphic evaluation of a [`LinearTransformation`] on a CKKS ciphertext.
///
/// The API is shaped around three phases:
/// 1. **Allocate** the prepared caches up-front:
///    [`PreparedLinearTransformationRhs::alloc`] for the right side,
///    [`PreparedLinearTransformationLhs::alloc`] for the left side.
/// 2. **Populate** them whenever the underlying data changes:
///    [`Self::ckks_prepare_linear_transformation_rhs`] /
///    [`Self::ckks_prepare_linear_transformation_lhs`].
/// 3. **Evaluate** with both caches:
///    [`Self::ckks_eval_prepared_linear_transformation_into`] (or the `_assign`
///    or `_many_` variants).
///
/// A one-shot convenience entry point,
/// [`Self::ckks_eval_linear_transformation_into`], allocates and populates both
/// caches internally for code paths that only evaluate a transform once.
pub trait LinearTransformationOps<BE: Backend> {
    // ----- tmp_bytes -----

    /// Scratch bytes required by [`Self::ckks_prepare_linear_transformation_rhs`].
    fn ckks_prepare_linear_transformation_rhs_tmp_bytes<P>(&self, pt_infos: &P) -> usize
    where
        P: LWEInfos;

    /// Scratch bytes required by [`Self::ckks_prepare_linear_transformation_lhs`].
    fn ckks_prepare_linear_transformation_lhs_tmp_bytes<C, K>(&self, ct: &C, key: &K) -> usize
    where
        C: CKKSCtBounds,
        K: GGLWEInfos;

    /// Scratch bytes required by the prepared-eval entry points.
    fn ckks_eval_linear_transformation_tmp_bytes<C, K>(&self, ct: &C, key: &K) -> usize
    where
        C: CKKSCtBounds,
        K: GGLWEInfos;

    // ----- populate -----

    /// Encodes every diagonal of `lt` into the matching pre-allocated slot
    /// of `prepared`.
    ///
    /// `prepared` must have been sized via
    /// [`PreparedLinearTransformationRhs::alloc`] for the same BSGS schedule as
    /// `lt`. Performs zero `CnvPVecR` allocations.
    fn ckks_prepare_linear_transformation_rhs<P>(
        &self,
        prepared: &mut PreparedLinearTransformationRhs<BE>,
        lt: &LinearTransformation<P>,
        scratch: &mut ScratchArena<'_, BE>,
    ) where
        P: GLWEToBackendRef<BE> + CKKSCtBounds;

    /// Fills `babies` with the prepared baby-step rotations of `src`.
    ///
    /// `babies` must have been sized via
    /// [`PreparedLinearTransformationLhs::alloc`] for the rotations the caller wants
    /// populated. Performs zero `CnvPVecL` allocations.
    fn ckks_prepare_linear_transformation_lhs<Src, H, K>(
        &self,
        babies: &mut PreparedLinearTransformationLhs<BE>,
        src: &Src,
        keys: &H,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        Src: GLWEToBackendRef<BE> + CKKSCtBounds,
        K: GLWEAutomorphismKeyPreparedToBackendRef<BE> + GGLWEPreparedToBackendRef<BE> + GetGaloisElement + GGLWEInfos,
        H: GLWEAutomorphismKeyHelper<K, BE>;

    // ----- eval -----

    /// Computes `dst = M · src` using the prepared right and left caches.
    ///
    /// `keys` must contain an automorphism key for every non-zero giant
    /// rotation of `prepared`. `babies` must cover at least
    /// `prepared.baby_steps`; supersets are allowed (e.g. when sharing a
    /// cache across several transforms).
    fn ckks_eval_prepared_linear_transformation_into<Dst, Src, H, K>(
        &self,
        dst: &mut Dst,
        src: &Src,
        prepared: &PreparedLinearTransformationRhs<BE>,
        babies: &PreparedLinearTransformationLhs<BE>,
        keys: &H,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        Dst: GLWEToBackendMut<BE> + CKKSCtBounds + SetCKKSInfos,
        Src: GLWEToBackendRef<BE> + CKKSCtBounds,
        K: GLWEAutomorphismKeyPreparedToBackendRef<BE> + GGLWEPreparedToBackendRef<BE> + GetGaloisElement + GGLWEInfos,
        H: GLWEAutomorphismKeyHelper<K, BE>;

    /// Computes `dst = M · dst` using the prepared right and left caches.
    fn ckks_eval_prepared_linear_transformation_assign<Dst, H, K>(
        &self,
        dst: &mut Dst,
        prepared: &PreparedLinearTransformationRhs<BE>,
        babies: &PreparedLinearTransformationLhs<BE>,
        keys: &H,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        Dst: GLWEToBackendMut<BE> + GLWEToBackendRef<BE> + CKKSCtBounds + SetCKKSInfos,
        K: GLWEAutomorphismKeyPreparedToBackendRef<BE> + GGLWEPreparedToBackendRef<BE> + GetGaloisElement + GGLWEInfos,
        H: GLWEAutomorphismKeyHelper<K, BE>;

    // ----- one-shot convenience -----

    /// Computes `dst = M · src` from a raw [`LinearTransformation`], allocating
    /// and populating both caches internally.
    ///
    /// Use the prepared entry points for repeated evaluation. This form is for
    /// one-off calls where the alloc cost is acceptable.
    fn ckks_eval_linear_transformation_into<Dst, Src, P, H, K>(
        &self,
        dst: &mut Dst,
        src: &Src,
        lt: &LinearTransformation<P>,
        keys: &H,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        Dst: GLWEToBackendMut<BE> + CKKSCtBounds + SetCKKSInfos,
        Src: GLWEToBackendRef<BE> + CKKSCtBounds,
        P: GLWEToBackendRef<BE> + CKKSCtBounds,
        K: GLWEAutomorphismKeyPreparedToBackendRef<BE> + GGLWEPreparedToBackendRef<BE> + GetGaloisElement + GGLWEInfos,
        H: GLWEAutomorphismKeyHelper<K, BE>;

    /// One-shot `dst = M · dst`, allocating and populating both caches internally.
    fn ckks_eval_linear_transformation_assign<Dst, P, H, K>(
        &self,
        dst: &mut Dst,
        lt: &LinearTransformation<P>,
        keys: &H,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        Dst: GLWEToBackendMut<BE> + GLWEToBackendRef<BE> + CKKSCtBounds + SetCKKSInfos,
        P: GLWEToBackendRef<BE> + CKKSCtBounds,
        K: GLWEAutomorphismKeyPreparedToBackendRef<BE> + GGLWEPreparedToBackendRef<BE> + GetGaloisElement + GGLWEInfos,
        H: GLWEAutomorphismKeyHelper<K, BE>;
}
