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
//! let mut prepared = LinearTransformation::alloc_prepared(module, &layout, lt.first_diagonal_plaintext().unwrap());
//! module.ckks_prepare_linear_transformation_rhs(&mut prepared, &lt, &mut scratch)?;
//! let mut babies = LinearTransformationBabySteps::alloc(module, prepared.baby_steps(), &ct);
//!
//! // per evaluation
//! module.ckks_prepare_linear_transformation_baby_steps(&mut babies, &ct, &atks, &mut scratch)?;
//! module.ckks_eval_linear_transformation_into(&mut dst, &ct, &babies, &prepared, &atks, &mut scratch)?;
//! ```

use crate::CKKSResult as Result;
use poulpy_core::layouts::IntPolyInfos;
use poulpy_core::{
    DiagonalProd,
    layouts::{GGLWEInfos, GLWEToBackendMut, GLWEToBackendRef, GetAutomorphismKey, LWEInfos, TorusPrecision},
};
use poulpy_hal::layouts::{Backend, ScratchArena};

use crate::{CKKSCtBounds, CKKSLayout, SetCKKSInfos, SlotsKind};

pub use poulpy_core::{
    LinearTransformation, LinearTransformationBabySteps, LinearTransformationDiagonal as Diagonal,
    LinearTransformationGiantStep as GiantStep, LinearTransformationLayout, LinearTransformationPlan,
    LinearTransformationPrepared, LinearTransformationStrategy, layouts::prepared::PreparedDiagonal, optimal_bsgs_giant_step,
};

/// The CKKS metadata of a linear-transformation diagonal: its encoding scale
/// (`log_delta`) and slot kind.
///
/// The CKKS scale / key-size bookkeeping and the result's slot kind read both
/// (together with the scheme-agnostic [`LWEInfos::k`]) off the transform's first
/// diagonal, which every diagonal shares, uniformly across the resident and
/// streamed representations. Keeping it here, rather than on `poulpy-core`'s
/// `DiagonalProd` engine trait, is deliberate: the core engine is scheme-agnostic
/// (a scheme encoding values mod `P` has neither), so it treats a prepared
/// diagonal's scale and real-slot claim as *opaque* values (stashed via
/// [`PreparedDiagonal::set_log_scale`] and [`PreparedDiagonal::set_real_slots`]
/// during preparation).
///
/// Implemented for the two diagonal representations: [`CKKSPlaintext`](crate::layouts::CKKSPlaintext) (streamed)
/// via its metadata, and the core [`PreparedDiagonal`] (resident) via those stashes.
pub trait LtDiagonalMeta {
    /// `log2` of the diagonal plaintext's scaling factor.
    fn lt_log_scale(&self) -> usize;

    /// Slot kind of the diagonal plaintext.
    fn lt_slots(&self) -> SlotsKind;

    /// Sparsity the diagonal plaintext imposes on its product with a ciphertext of degree `2^log_n`: `log2` of the
    /// ratio of that degree to the one the diagonal is dense at.
    fn lt_log_sparsity(&self, log_n: usize) -> usize;
}

/// Torus width of the result of [`CKKSLinearTransformationOps::ckks_eval_linear_transformation_into`] evaluating `lt`
/// from a ciphertext of width `src_k` into one of width `dst_k`, when it evaluates: the diagonals' scale comes off the
/// source, capped at the destination. Baby-step keys are looked up at `src_k`, giant-step keys at this width.
pub fn ckks_linear_transformation_output_k<P: LtDiagonalMeta>(
    lt: &LinearTransformation<P>,
    src_k: TorusPrecision,
    dst_k: TorusPrecision,
) -> TorusPrecision {
    let scale = lt.first_diagonal_plaintext().map_or(0, LtDiagonalMeta::lt_log_scale);
    src_k.as_usize().saturating_sub(scale).min(dst_k.as_usize()).into()
}

/// Layout of that result from a ciphertext of layout `src`: `src` at [`ckks_linear_transformation_output_k`], its
/// slots joined with the diagonals', sparse where both are.
pub fn ckks_linear_transformation_output_layout<P: LtDiagonalMeta>(
    lt: &LinearTransformation<P>,
    mut src: CKKSLayout,
    dst_k: TorusPrecision,
) -> CKKSLayout {
    src.glwe_layout.k = ckks_linear_transformation_output_k(lt, src.k(), dst_k);
    if let Some(diagonal) = lt.first_diagonal_plaintext() {
        src.meta.slots = src.meta.slots.join(diagonal.lt_slots());
        src.meta.log_sparsity = src.meta.log_sparsity.min(diagonal.lt_log_sparsity(src.log_n()));
    }
    src
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

    /// Scratch bytes required by [`Self::ckks_eval_linear_transformation_sum_into`].
    fn ckks_eval_linear_transformation_sum_tmp_bytes<C, K>(&self, ct: &C, key: &K) -> usize
    where
        C: CKKSCtBounds,
        K: GGLWEInfos;

    // ----- populate -----

    /// Encodes every diagonal of `lt` into the matching pre-allocated slot
    /// of `prepared`.
    ///
    /// `prepared` must have been sized via
    /// [`LinearTransformation::alloc_prepared`] for the same BSGS schedule as
    /// `lt`. Performs zero `CnvPVecR` allocations. The proxy must also carry the
    /// diagonals' degree, which may be below the ring degree;
    /// `lt.first_diagonal_plaintext()` is the proxy every caller should pass.
    ///
    /// # Errors
    ///
    /// `InvalidOperand` when the diagonals of `lt` do not share one scale, width and slot kind, before any write to
    /// `prepared`.
    fn ckks_prepare_linear_transformation_rhs<P>(
        &self,
        prepared: &mut LinearTransformationPrepared<BE>,
        lt: &LinearTransformation<P>,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        P: GLWEToBackendRef<BE> + IntPolyInfos + CKKSCtBounds + DiagonalProd<BE>;

    /// Fills `babies` with the prepared baby-step rotations of `src`.
    ///
    /// `babies` must have been sized via
    /// [`LinearTransformationBabySteps::alloc`] for the rotations the caller wants
    /// populated. Performs zero `CnvPVecL` allocations.
    fn ckks_prepare_linear_transformation_baby_steps<Src, H>(
        &self,
        babies: &mut LinearTransformationBabySteps<BE>,
        src: &Src,
        keys: &H,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        Src: GLWEToBackendRef<BE> + CKKSCtBounds,
        H: GetAutomorphismKey<BE>;

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
    fn ckks_eval_linear_transformation_into<Dst, Src, P, H>(
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
        P: DiagonalProd<BE> + LtDiagonalMeta + IntPolyInfos,
        H: GetAutomorphismKey<BE>;

    /// In-place `dst = M · dst` with a caller-supplied baby cache (see
    /// [`Self::ckks_eval_linear_transformation_into`]).
    fn ckks_eval_linear_transformation_assign<Dst, P, H>(
        &self,
        dst: &mut Dst,
        babies: &LinearTransformationBabySteps<BE>,
        lt: &LinearTransformation<P>,
        keys: &H,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        Dst: GLWEToBackendMut<BE> + GLWEToBackendRef<BE> + CKKSCtBounds + SetCKKSInfos,
        P: DiagonalProd<BE> + LtDiagonalMeta + IntPolyInfos,
        H: GetAutomorphismKey<BE>;

    /// Computes `dst = Σ_t M_t · src_t` over resident transforms that share one
    /// giant schedule, at the giant key switches of one transform. Term `t` is
    /// `(srcs[t], babies[t], lts[t])`, as for [`Self::ckks_eval_linear_transformation_into`];
    /// one term is that method, bit for bit. The terms are added in slice order;
    /// see [`GLWELinearTransformations::glwe_eval_linear_transformation_sum_into`](poulpy_core::GLWELinearTransformations::glwe_eval_linear_transformation_sum_into)
    /// for why the order matters and how to pick `base2k`.
    ///
    /// `dst` gets the single transform's metadata; its slot kind joins every term's.
    ///
    /// Returns `InvalidOperand`, before writing `dst`, if the slices differ in
    /// length or are empty, or if a term does not share the first one's giant
    /// schedule, diagonal layout, width, scale and slot kind, source layout and
    /// scale, or baby-cache layout. Also returns an error before writing `dst`
    /// if a term fails the checks of [`Self::ckks_eval_linear_transformation_into`].
    fn ckks_eval_linear_transformation_sum_into<Dst, Src, H>(
        &self,
        dst: &mut Dst,
        srcs: &[&Src],
        babies: &[&LinearTransformationBabySteps<BE>],
        lts: &[&LinearTransformationPrepared<BE>],
        keys: &H,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        Dst: GLWEToBackendMut<BE> + CKKSCtBounds + SetCKKSInfos,
        Src: GLWEToBackendRef<BE> + CKKSCtBounds,
        H: GetAutomorphismKey<BE>;

    // ----- eval (self-allocated baby cache) -----

    /// Computes `dst = M · src`, allocating and preparing the baby cache
    /// internally.
    ///
    /// Convenience over [`Self::ckks_eval_linear_transformation_into`] for one-off
    /// evaluations; for repeated evaluation that shares a baby cache, use that
    /// method. A plaintext `lt` takes the streamed path; a prepared `lt` the
    /// resident path.
    fn ckks_eval_linear_transformation_self_into<Dst, Src, P, H>(
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
        P: DiagonalProd<BE> + LtDiagonalMeta + IntPolyInfos,
        H: GetAutomorphismKey<BE>;

    /// In-place `dst = M · dst`, self-allocating the baby cache (see
    /// [`Self::ckks_eval_linear_transformation_self_into`]).
    fn ckks_eval_linear_transformation_self_assign<Dst, P, H>(
        &self,
        dst: &mut Dst,
        lt: &LinearTransformation<P>,
        keys: &H,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        Dst: GLWEToBackendMut<BE> + GLWEToBackendRef<BE> + CKKSCtBounds + SetCKKSInfos,
        P: DiagonalProd<BE> + LtDiagonalMeta + IntPolyInfos,
        H: GetAutomorphismKey<BE>;
}
