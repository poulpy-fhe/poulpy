//! CKKS-facing API for the homomorphic linear transformation (a matrix–vector
//! product over the slots), evaluated with the baby-step / giant-step (BSGS)
//! decomposition.
//!
//! The transformation itself is the scheme-agnostic GLWE-level engine in
//! [`poulpy_core`] (`LinearTransformation` / `GLWELinearTransformations`); this layer
//! re-exports the data type under CKKS names and adds the scale-aware entry point.
//! All `log_delta` / `log_budget` accounting lives here — the core engine only
//! receives base2k-level alignment integers. See
//! [`docs/guides/linear_transformation.md`](https://github.com/poulpy-fhe/poulpy/blob/main/docs/guides/linear_transformation.md).
//!
//! # Prepared vs streamed, one evaluator
//!
//! Evaluation is generic over the diagonal representation `P`: a prepared
//! [`PreparedDiagonal`] (resident, fastest for repeated use) or an unprepared diagonal (streamed, lower peak memory),
//! a [`CKKSPlaintext`](crate::layouts::CKKSPlaintext) or a [`CKKSCiphertext`](crate::layouts::CKKSCiphertext) of the
//! source's rank. All go through the same
//! [`CKKSLinearTransformationOps::ckks_eval_linear_transformation_into`]: the `P = PreparedDiagonal` case is the
//! resident path, an unprepared `P` the streamed path, which prepares each diagonal in the call. The rank of the
//! diagonals picks the product: plaintext diagonals are multiplied as plaintexts, ciphertext diagonals tensored and
//! relinearized. There is no separate "prepared" vs "unprepared" method; you pick the path by which `P` you pass.
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
use poulpy_core::{
    DiagonalProd,
    layouts::{
        GGLWEInfos, GLWEInfos, GLWELayout, GLWEToBackendMut, GLWEToBackendRef, GetAutomorphismKey, GetTensorKey, LWEInfos, Rank,
        TorusPrecision,
    },
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
/// diagonal's plaintext meta as *opaque* (stashed via
/// [`PreparedDiagonal::set_plaintext_meta`] during preparation).
///
/// Implemented for the diagonal representations: [`CKKSPlaintext`](crate::layouts::CKKSPlaintext) and
/// [`CKKSCiphertext`](crate::layouts::CKKSCiphertext) (streamed) via their metadata, and the core [`PreparedDiagonal`]
/// (resident) via those stashes.
pub trait LtDiagonalMeta {
    /// `log2` of the diagonal plaintext's scaling factor.
    fn lt_log_scale(&self) -> usize;

    /// Slot kind of the diagonal plaintext.
    fn lt_slots(&self) -> SlotsKind;

    /// Sparsity the diagonal plaintext imposes on its product with a ciphertext of degree `2^log_n`: `log2` of the
    /// ratio of that degree to the one the diagonal is dense at.
    fn lt_log_sparsity(&self, log_n: usize) -> usize;

    /// The width of the diagonal when it is a ciphertext, multiplied as one; `None` for a plaintext.
    fn lt_ciphertext_k(&self) -> Option<TorusPrecision> {
        None
    }
}

/// Torus width of the result of [`CKKSLinearTransformationOps::ckks_eval_linear_transformation_into`] evaluating `lt`
/// of plaintext diagonals from a ciphertext of width `src_k` into one of width `dst_k`, when it evaluates: the
/// diagonals' scale comes off the source, capped at the destination. Baby-step keys are looked up at `src_k`, giant-step
/// keys at this width. Diagonals of the source's rank need the source's scale too:
/// [`ckks_linear_transformation_output_layout`] gives the width of either.
pub fn ckks_linear_transformation_output_k<P: LtDiagonalMeta>(
    lt: &LinearTransformation<P>,
    src_k: TorusPrecision,
    dst_k: TorusPrecision,
) -> TorusPrecision {
    let scale = lt.first_diagonal_plaintext().map_or(0, LtDiagonalMeta::lt_log_scale);
    src_k.as_usize().saturating_sub(scale).min(dst_k.as_usize()).into()
}

/// Layout of a plaintext diagonal of a transform on a ciphertext of layout `ct`, at its degree, `base2k` and storage
/// width: the diagonal the scratch queries take for a transform of plaintext diagonals.
pub fn ckks_plaintext_diagonal_layout<C: GLWEInfos>(ct: &C) -> GLWELayout {
    GLWELayout {
        n: ct.n(),
        base2k: ct.base2k(),
        k: ct.max_k(),
        rank: Rank(0),
    }
}

/// Layout of that result from a ciphertext of layout `src`, its slots joined with the diagonals', sparse where it and
/// every diagonal are: for plaintext diagonals `src` at [`ckks_linear_transformation_output_k`], for diagonals of the source's rank the
/// width and scale of the rule of [`CKKSMulOps::ckks_mul_into`](crate::api::CKKSMulOps::ckks_mul_into) between `src`
/// and a diagonal, a width of 0 when that rule leaves no result.
pub fn ckks_linear_transformation_output_layout<P: LtDiagonalMeta>(
    lt: &LinearTransformation<P>,
    mut src: CKKSLayout,
    dst_k: TorusPrecision,
) -> CKKSLayout {
    let src_k = src.k();
    src.glwe_layout.k = ckks_linear_transformation_output_k(lt, src_k, dst_k);
    if let Some(diagonal) = lt.first_diagonal_plaintext() {
        if let Some(diagonal_k) = diagonal.lt_ciphertext_k() {
            let (log_budget, log_delta, _) = crate::mul_ct_params_raw(
                "ckks_linear_transformation_output_layout",
                dst_k.as_usize(),
                crate::CKKSInfos::log_delta(&src),
                src_k.into(),
                diagonal.lt_log_scale(),
                diagonal_k.into(),
            )
            .unwrap_or_default();
            src.meta.scale = crate::Scale::Log(log_delta);
            src.glwe_layout.k = (log_budget + log_delta).into();
        }
        src.meta.slots = src.meta.slots.join(diagonal.lt_slots());
        let log_n = src.log_n();
        // Streamed ciphertext diagonals each carry their own sparsity label.
        for d in lt.giant_steps.iter().flat_map(|gs| &gs.diagonals) {
            src.meta.log_sparsity = src.meta.log_sparsity.min(d.plaintext.lt_log_sparsity(log_n));
        }
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
        P: GLWEInfos;

    /// Scratch bytes required by [`Self::ckks_prepare_linear_transformation_baby_steps`] for a
    /// cache of the baby steps of `plan` (see
    /// [`GLWELinearTransformations::glwe_prepare_linear_transformation_baby_steps_tmp_bytes`](poulpy_core::GLWELinearTransformations::glwe_prepare_linear_transformation_baby_steps_tmp_bytes)).
    fn ckks_prepare_linear_transformation_baby_steps_tmp_bytes<C, K>(
        &self,
        ct: &C,
        plan: &LinearTransformationPlan,
        key: &K,
    ) -> usize
    where
        C: CKKSCtBounds,
        K: GGLWEInfos;

    /// Scratch bytes required to evaluate with a **resident** RHS (`P =
    /// PreparedDiagonal`) a transform with the schedule `plan` and diagonals of the layout `diagonal`, plaintexts or
    /// ciphertexts by its rank ([`ckks_plaintext_diagonal_layout`] for plaintexts), with the tensor key `tsk` for
    /// ciphertexts (see
    /// [`GLWELinearTransformations::glwe_eval_linear_transformation_tmp_bytes`](poulpy_core::GLWELinearTransformations::glwe_eval_linear_transformation_tmp_bytes)).
    fn ckks_eval_linear_transformation_tmp_bytes<C, D, K, T>(
        &self,
        ct: &C,
        diagonal: &D,
        plan: &LinearTransformationPlan,
        tsk: &T,
        key: &K,
    ) -> usize
    where
        C: CKKSCtBounds,
        D: GLWEInfos,
        K: GGLWEInfos,
        T: GGLWEInfos;

    /// Scratch bytes required to evaluate with a **streamed** RHS (an unprepared diagonal `P`): the streamed inner
    /// product additionally holds one resident `CnvPVecR` diagonal slot, so this is larger than the resident budget.
    /// `diagonal`, `plan` and `tsk` as in [`Self::ckks_eval_linear_transformation_tmp_bytes`].
    fn ckks_eval_linear_transformation_streamed_tmp_bytes<C, D, K, T>(
        &self,
        ct: &C,
        diagonal: &D,
        plan: &LinearTransformationPlan,
        tsk: &T,
        key: &K,
    ) -> usize
    where
        C: CKKSCtBounds,
        D: GLWEInfos,
        K: GGLWEInfos,
        T: GGLWEInfos;

    /// Scratch bytes required by [`Self::ckks_eval_linear_transformation_sum_into`], `plan` the
    /// schedule of the first term.
    fn ckks_eval_linear_transformation_sum_tmp_bytes<C, D, K, T>(
        &self,
        ct: &C,
        diagonal: &D,
        plan: &LinearTransformationPlan,
        tsk: &T,
        key: &K,
    ) -> usize
    where
        C: CKKSCtBounds,
        D: GLWEInfos,
        K: GGLWEInfos,
        T: GGLWEInfos;

    // ----- populate -----

    /// Encodes every diagonal of `lt` into the matching pre-allocated slot
    /// of `prepared`.
    ///
    /// `prepared` must have been sized via
    /// [`LinearTransformation::alloc_prepared`] for the same BSGS schedule as
    /// `lt`. Performs zero `CnvPVecR` allocations. The proxy must also carry the
    /// diagonals' degree, which may be below the ring degree, and rank;
    /// `lt.first_diagonal_plaintext()` is the proxy every caller should pass. The diagonals are CKKS plaintexts, or
    /// ciphertexts of the source's rank, which the transform then multiplies as ciphertexts and whose smallest
    /// sparsity it records.
    ///
    /// # Errors
    ///
    /// `InvalidOperand`, before any write to `prepared`, when `prepared` was allocated for another schedule or
    /// diagonal degree, radix, rank or width than `lt`'s, or when the diagonals of `lt` do not share one scale, width
    /// and slot kind.
    fn ckks_prepare_linear_transformation_rhs<P>(
        &self,
        prepared: &mut LinearTransformationPrepared<BE>,
        lt: &LinearTransformation<P>,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        P: GLWEToBackendRef<BE> + CKKSCtBounds;

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
    /// - `P = CKKSPlaintext` or `P = CKKSCiphertext`: streamed, each diagonal is prepared on the fly (the
    ///   matrix RHS is never fully materialized), lower peak memory.
    ///
    /// `keys` must contain an automorphism key for every non-zero giant rotation
    /// of `lt`. `babies` must cover at least `lt`'s baby rotations; supersets are
    /// allowed (e.g. when sharing a cache across several transforms).
    ///
    /// Diagonals of the rank of `src`, ciphertexts prepared by [`Self::ckks_prepare_linear_transformation_rhs`] or
    /// streamed, are multiplied as ciphertexts: each giant step's products are summed and relinearized once, as by
    /// [`CKKSDotProductOps::ckks_dot_product_prepared`](crate::api::CKKSDotProductOps::ckks_dot_product_prepared), with
    /// the tensor key of `keys` at the larger of the widths of `src` and of the diagonals.
    ///
    /// The output metadata is [`ckks_linear_transformation_output_layout`].
    ///
    /// # Errors
    ///
    /// Each before any write to `dst`:
    /// - `InvalidOperand` when the diagonals of `lt` do not share one scale, width and slot kind, or when `src` is of
    ///   a degree above the module's;
    /// - for ciphertext diagonals, in this order, `PreparedOperandLayoutMismatch` for a diagonal of another degree,
    ///   `base2k` or rank than `dst`, `InvalidOperand` past `base2k` 62 or for `babies` prepared from a source of another
    ///   width than `src`, and `MissingRelinearizationKey` without the tensor key;
    /// - `MultiplicationPrecisionUnderflow` or `InsufficientHomomorphicCapacity` when the budget rule leaves no result
    ///   at `dst`'s width;
    /// - `MissingAutomorphismKey` for a non-zero giant rotation of `lt` with no key at the output width;
    /// - `CKKSError::Internal` when `lt` has no diagonals or `babies` lacks one of `lt`'s baby rotations.
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
        P: DiagonalProd<BE> + LtDiagonalMeta,
        H: GetAutomorphismKey<BE> + GetTensorKey<BE>;

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
        P: DiagonalProd<BE> + LtDiagonalMeta,
        H: GetAutomorphismKey<BE> + GetTensorKey<BE>;

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
    /// scale, or baby-cache layout and width. Also returns an error before writing `dst`
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
        H: GetAutomorphismKey<BE> + GetTensorKey<BE>;

    // ----- eval (self-allocated baby cache) -----

    /// Computes `dst = M · src`, allocating and preparing the baby cache
    /// internally.
    ///
    /// Convenience over [`Self::ckks_eval_linear_transformation_into`] for one-off
    /// evaluations; for repeated evaluation that shares a baby cache, use that
    /// method. An unprepared `lt` takes the streamed path; a prepared `lt` the
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
        P: DiagonalProd<BE> + LtDiagonalMeta,
        H: GetAutomorphismKey<BE> + GetTensorKey<BE>;

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
        P: DiagonalProd<BE> + LtDiagonalMeta,
        H: GetAutomorphismKey<BE> + GetTensorKey<BE>;
}
