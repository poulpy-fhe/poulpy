//! CKKS-facing API for the homomorphic linear transformation (a matrix–vector
//! product over the slots), evaluated with the baby-step / giant-step (BSGS)
//! decomposition.
//!
//! The transformation itself is the scheme-agnostic GLWE-level engine in
//! [`poulpy_core`] (`LinearTransformation` / `GLWELinearTransformations`); this layer
//! re-exports the data type under CKKS names and adds the scale-aware entry point.
//! All `log_delta` / `log_budget` accounting lives here — the core engine only
//! receives base2k-level alignment integers. See
//! [`docs/linear_transformation.md`](https://github.com/poulpy-fhe/poulpy/blob/main/docs/linear_transformation.md).
//!
//! # Prepared vs streamed — one evaluator
//!
//! Evaluation is generic over the diagonal representation `P`: a prepared
//! [`PreparedDiagonal`] (resident, fastest for repeated use) or a plaintext
//! [`CKKSPlaintext`](crate::layouts::CKKSPlaintext) diagonal (streamed, lower peak memory). Both go through the
//! same [`Self::ckks_eval_linear_transformation_into`] — the `P = PreparedDiagonal`
//! case is the resident path and a plaintext `P` is the streamed path. There is no
//! separate "prepared" vs "unprepared" method; you pick the path by which `P` you
//! pass.
//!
//! # Typical flow
//!
//! ```ignore
//! // setup, once per transform / per input shape
//! let mut prepared = LinearTransformation::alloc_prepared(module, &layout, &pt_proxy);
//! module.ckks_prepare_linear_transformation_rhs(&mut prepared, &lt, &mut scratch);
//! let mut babies = LinearTransformationBabySteps::alloc(module, prepared.baby_steps(), &ct);
//!
//! // per evaluation
//! module.ckks_prepare_linear_transformation_baby_steps(&mut babies, &ct, &atks, &mut scratch)?;
//! module.ckks_eval_linear_transformation_into(&mut dst, &ct, &babies, &prepared, &atks, &mut scratch)?;
//! ```

use crate::CKKSAtkBounds;
use crate::CKKSResult as Result;
use poulpy_core::layouts::IntPolyInfos;
use poulpy_core::{
    default::linear_transformation::DiagonalProd,
    layouts::{
        GGLWEInfos, GLWEAutomorphismKeyHelper, GLWEAutomorphismKeyLayoutHelper, GLWEToBackendMut, GLWEToBackendRef, LWEInfos,
    },
};
use poulpy_hal::layouts::{Backend, ScratchArena};

use crate::{CKKSCtBounds, SetCKKSInfos};

pub use poulpy_core::{
    LinearTransformation, LinearTransformationBabySteps, LinearTransformationDiagonal as Diagonal,
    LinearTransformationGiantStep as GiantStep, LinearTransformationLayout, LinearTransformationPlan,
    LinearTransformationPrepared, LinearTransformationStrategy, layouts::prepared::PreparedDiagonal, optimal_bsgs_giant_step,
};

/// The CKKS encoding scale (`log_delta`) of a linear-transformation diagonal.
///
/// The CKKS scale / key-size bookkeeping reads it (together with the
/// scheme-agnostic [`LWEInfos::k`]) off the transform's first diagonal,
/// uniformly across the resident and streamed representations. Keeping it here —
/// rather than on `poulpy-core`'s `DiagonalProd` engine trait — is deliberate:
/// the core engine is scheme-agnostic (a scheme encoding values mod `P` has no
/// `log_delta`), so it treats a prepared diagonal's scale as an *opaque* integer
/// (stashed via [`PreparedDiagonal::set_log_scale`] during preparation) and
/// carries no scale concept of its own.
///
/// Implemented for the two diagonal representations: [`CKKSPlaintext`](crate::layouts::CKKSPlaintext) (streamed)
/// via its `log_delta`, and the core [`PreparedDiagonal`] (resident) via that
/// stashed scale.
pub trait LtDiagonalScale {
    /// `log2` of the diagonal plaintext's scaling factor.
    fn lt_log_scale(&self) -> usize;
}

/// Homomorphic evaluation of a [`LinearTransformation`] on a CKKS ciphertext.
///
/// The API is shaped around three phases:
/// 1. **Allocate** the prepared caches up-front:
///    [`LinearTransformation::alloc_prepared`] for the right side,
///    [`LinearTransformationBabySteps::alloc`] for the left side.
/// 2. **Populate** them whenever the underlying data changes:
///    [`Self::ckks_prepare_linear_transformation_rhs`] /
///    [`Self::ckks_prepare_linear_transformation_baby_steps`].
/// 3. **Evaluate** with [`Self::ckks_eval_linear_transformation_into`] (or the
///    `_assign` / `_self_` variants), generic over the diagonal representation.
///
/// The `_self_` variants allocate and prepare the baby cache internally for code
/// paths that evaluate a transform once.
pub trait CKKSLinearTransformationOps<BE: Backend> {
    // ----- tmp_bytes -----

    /// Scratch bytes required by [`Self::ckks_prepare_linear_transformation_rhs`].
    fn ckks_prepare_linear_transformation_rhs_tmp_bytes<P>(&self, pt_infos: &P) -> usize
    where
        P: LWEInfos;

    /// Scratch bytes required by [`Self::ckks_prepare_linear_transformation_baby_steps`].
    fn ckks_prepare_linear_transformation_baby_steps_tmp_bytes<C, K>(&self, ct: &C, key: &K) -> usize
    where
        C: CKKSCtBounds,
        K: GGLWEInfos;

    /// Scratch bytes required to evaluate with a **resident** RHS (`P =
    /// PreparedDiagonal`).
    fn ckks_eval_linear_transformation_tmp_bytes<C, K>(&self, ct: &C, key: &K) -> usize
    where
        C: CKKSCtBounds,
        K: GGLWEInfos;

    /// Scratch bytes required to evaluate with a **streamed** RHS (a plaintext
    /// diagonal `P`): the streamed inner product additionally holds one resident
    /// `CnvPVecR` diagonal slot, so this is larger than the resident budget.
    fn ckks_eval_linear_transformation_streamed_tmp_bytes<C, K>(&self, ct: &C, key: &K) -> usize
    where
        C: CKKSCtBounds,
        K: GGLWEInfos;

    /// Scratch bytes required by a whole chained transform (homomorphic DFT):
    /// the widest per-factor budget plus the one ciphertext the chain
    /// ping-pongs through across factors. Covers a streamed as well as a
    /// resident RHS, so a custom chain evaluator can size cross-factor
    /// workspace from it.
    fn ckks_dft_evaluate_tmp_bytes<C, K>(&self, ct: &C, key: &K) -> usize
    where
        C: CKKSCtBounds,
        K: GGLWEInfos;

    // ----- populate -----

    /// Encodes every diagonal of `lt` into the matching pre-allocated slot
    /// of `prepared`.
    ///
    /// `prepared` must have been sized via
    /// [`LinearTransformation::alloc_prepared`] for the same BSGS schedule as
    /// `lt`. Performs zero `CnvPVecR` allocations.
    fn ckks_prepare_linear_transformation_rhs<P>(
        &self,
        prepared: &mut LinearTransformationPrepared<BE>,
        lt: &LinearTransformation<P>,
        scratch: &mut ScratchArena<'_, BE>,
    ) where
        P: GLWEToBackendRef<BE> + IntPolyInfos + CKKSCtBounds + DiagonalProd<BE>;

    /// Fills `babies` with the prepared baby-step rotations of `src`.
    ///
    /// `babies` must have been sized via
    /// [`LinearTransformationBabySteps::alloc`] for the rotations the caller wants
    /// populated. Performs zero `CnvPVecL` allocations.
    fn ckks_prepare_linear_transformation_baby_steps<Src, H, K>(
        &self,
        babies: &mut LinearTransformationBabySteps<BE>,
        src: &Src,
        keys: &H,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        Src: GLWEToBackendRef<BE> + CKKSCtBounds,
        K: CKKSAtkBounds<BE>,
        H: GLWEAutomorphismKeyHelper<K> + GLWEAutomorphismKeyLayoutHelper<K>;

    // ----- eval (caller-supplied baby cache) -----

    /// Computes `dst = M · src` from the matrix `lt` and a caller-supplied,
    /// already-prepared baby cache, generic over the diagonal representation `P`:
    ///
    /// - `P = PreparedDiagonal` (a [`LinearTransformationPrepared`]): resident,
    ///   the diagonals are already convolution-domain.
    /// - `P = CKKSPlaintext`: streamed, each diagonal is prepared on the fly (the
    ///   matrix RHS is never fully materialized) — lower peak memory.
    ///
    /// `keys` must contain an automorphism key for every non-zero giant rotation
    /// of `lt`. `babies` must cover at least `lt`'s baby rotations; supersets are
    /// allowed (e.g. when sharing a cache across several transforms).
    fn ckks_eval_linear_transformation_into<Dst, Src, P, H, K>(
        &self,
        dst: &mut Dst,
        src: &Src,
        babies: &LinearTransformationBabySteps<BE>,
        lt: &LinearTransformation<P>,
        keys: &H,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        Dst: GLWEToBackendMut<BE> + CKKSCtBounds + SetCKKSInfos,
        Src: GLWEToBackendRef<BE> + CKKSCtBounds,
        P: DiagonalProd<BE> + LtDiagonalScale + IntPolyInfos,
        K: CKKSAtkBounds<BE>,
        H: GLWEAutomorphismKeyHelper<K> + GLWEAutomorphismKeyLayoutHelper<K>;

    /// In-place `dst = M · dst` with a caller-supplied baby cache (see
    /// [`Self::ckks_eval_linear_transformation_into`]).
    fn ckks_eval_linear_transformation_assign<Dst, P, H, K>(
        &self,
        dst: &mut Dst,
        babies: &LinearTransformationBabySteps<BE>,
        lt: &LinearTransformation<P>,
        keys: &H,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        Dst: GLWEToBackendMut<BE> + GLWEToBackendRef<BE> + CKKSCtBounds + SetCKKSInfos,
        P: DiagonalProd<BE> + LtDiagonalScale + IntPolyInfos,
        K: CKKSAtkBounds<BE>,
        H: GLWEAutomorphismKeyHelper<K> + GLWEAutomorphismKeyLayoutHelper<K>;

    // ----- eval (self-allocated baby cache) -----

    /// Computes `dst = M · src`, allocating and preparing the baby cache
    /// internally.
    ///
    /// Convenience over [`Self::ckks_eval_linear_transformation_into`] for one-off
    /// evaluations; for repeated evaluation that shares a baby cache, use that
    /// method. A plaintext `lt` takes the streamed path; a prepared `lt` the
    /// resident path.
    fn ckks_eval_linear_transformation_self_into<Dst, Src, P, H, K>(
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
        P: DiagonalProd<BE> + LtDiagonalScale + IntPolyInfos,
        K: CKKSAtkBounds<BE>,
        H: GLWEAutomorphismKeyHelper<K> + GLWEAutomorphismKeyLayoutHelper<K>;

    /// In-place `dst = M · dst`, self-allocating the baby cache (see
    /// [`Self::ckks_eval_linear_transformation_self_into`]).
    fn ckks_eval_linear_transformation_self_assign<Dst, P, H, K>(
        &self,
        dst: &mut Dst,
        lt: &LinearTransformation<P>,
        keys: &H,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        Dst: GLWEToBackendMut<BE> + GLWEToBackendRef<BE> + CKKSCtBounds + SetCKKSInfos,
        P: DiagonalProd<BE> + LtDiagonalScale + IntPolyInfos,
        K: CKKSAtkBounds<BE>,
        H: GLWEAutomorphismKeyHelper<K> + GLWEAutomorphismKeyLayoutHelper<K>;
}
