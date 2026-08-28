//! CKKS wrapper for the GLWE-level linear transformation.
//!
//! Computes the scale-derived convolution parameters (`a_k`,
//! `cnv_offset`) and the result `log_delta` / `log_budget`, delegates the actual
//! evaluation to the scheme-agnostic core engine
//! [`GLWELinearTransformations`](trait@poulpy_core::GLWELinearTransformations), and stamps the
//! CKKS metadata onto the result. See `docs/linear_transformation.md`.

use crate::CKKSAtkBounds;
use crate::SlotsKind;
use crate::{CKKSResult as Result, ckks_ensure};
use poulpy_core::default::operations::cnv_offset_to_limb_offset;
use poulpy_core::layouts::IntPolyInfos;
use poulpy_core::{
    GLWECopy, GLWELinearTransformations, LinearTransformationBabySteps, LinearTransformationGiantStep,
    LinearTransformationPrepared,
    default::linear_transformation::{DiagonalProd, glwe_accumulate_streamed_baby_steps_dft},
    layouts::{
        Base2K, Degree, GGLWEBind, GGLWEInfos, GGLWEUse, GLWEAutomorphismKeyHelper, GLWEAutomorphismKeyLayoutHelper, GLWEInfos,
        GLWEToBackendMut, GLWEToBackendRef, GetGaloisElement, LWEInfos, Rank, TorusPrecision, WithEffectiveDsize,
        prepared::{GGLWEPreparedBound, GGLWEPreparedToBackendRef, GLWEAutomorphismKeyPreparedToBackendRef, PreparedDiagonal},
    },
};
use poulpy_hal::{
    api::{CnvPVecBytesOf, Convolution, ModuleN},
    layouts::{Backend, CyclotomicOrder, Data, Module, ScratchArena, VecZnxDftBackendMut, ZnxWord, galois_element},
};

use crate::{
    CKKSCompositionError, CKKSCtBounds, CKKSInfos, CKKSMeta, SetCKKSInfos,
    api::{CKKSCopyOps, CKKSLinearTransformationOps, LinearTransformation, LtDiagonalScale},
    default::mul::mul_pt_params_raw,
    layouts::{CKKSModuleAlloc, CKKSPlaintext, ScratchArenaTakeCKKS},
};
use poulpy_core::GLWEBytesOf;

/// Per-giant streamed PROD for CKKS plaintext diagonals.
///
/// The scheme-side half of [`DiagonalProd`]: where the resident path (core's
/// [`PreparedDiagonal`]) fuses
/// already-prepared diagonals, the streamed path prepares each
/// [`CKKSPlaintext`] diagonal on the fly. Implementing it here (per concrete
/// plaintext type) is what lets the resident and streamed transforms share the
/// single `LinearTransformation<P>` container without overlapping impls.
impl<BE: Backend, D: Data> DiagonalProd<BE> for CKKSPlaintext<D, BE::ZnxWord>
where
    CKKSPlaintext<D, BE::ZnxWord>: GLWEToBackendRef<BE>,
{
    fn accumulate_giant_prod<M>(
        module: &M,
        cnv_offset_hi: usize,
        prod_dft: &mut VecZnxDftBackendMut<'_, BE>,
        lhs: &LinearTransformationBabySteps<BE>,
        gs: &LinearTransformationGiantStep<Self>,
        scratch: &mut ScratchArena<'_, BE>,
    ) where
        M: CnvPVecBytesOf + Convolution<BE> + ModuleN,
    {
        glwe_accumulate_streamed_baby_steps_dft(module, cnv_offset_hi, prod_dft, lhs, gs, scratch);
    }
}

/// Streamed-diagonal scale: a [`CKKSPlaintext`] carries its scale as `log_delta`.
impl<D: Data, W: ZnxWord> LtDiagonalScale for CKKSPlaintext<D, W> {
    fn lt_log_scale(&self) -> usize {
        self.log_delta()
    }
}

/// Resident-diagonal scale: a core [`PreparedDiagonal`] carries the (opaque to the
/// core engine) scale the CKKS prepare step stashed on it via `set_log_scale`.
impl<D: Data, BE: Backend> LtDiagonalScale for PreparedDiagonal<D, BE> {
    fn lt_log_scale(&self) -> usize {
        self.log_scale()
    }
}

impl<BE: Backend> CKKSLinearTransformationOps<BE> for Module<BE>
where
    Module<BE>: GLWELinearTransformations<BE> + GLWECopy<BE> + CKKSCopyOps<BE> + CKKSModuleAlloc<BE> + CyclotomicOrder,
{
    // ---------- tmp_bytes ----------

    fn ckks_prepare_linear_transformation_rhs_tmp_bytes<P>(&self, pt_infos: &P) -> usize
    where
        P: LWEInfos,
    {
        self.glwe_prepare_linear_transformation_rhs_tmp_bytes(pt_infos)
    }

    fn ckks_prepare_linear_transformation_baby_steps_tmp_bytes<C, H, K>(&self, ct: &C, rotations: &[i64], keys: &H) -> usize
    where
        C: CKKSCtBounds,
        K: GGLWEInfos,
        H: GLWEAutomorphismKeyLayoutHelper<K>,
    {
        self.glwe_prepare_linear_transformation_baby_steps_tmp_bytes(ct, rotations, keys)
    }

    fn ckks_eval_linear_transformation_into_tmp_bytes<Dst, Src, P, H, K>(
        &self,
        dst: &Dst,
        src: &Src,
        lt: &LinearTransformation<P>,
        keys: &H,
    ) -> usize
    where
        Dst: CKKSCtBounds,
        Src: CKKSCtBounds,
        P: LWEInfos,
        K: GGLWEInfos,
        H: GLWEAutomorphismKeyLayoutHelper<K>,
    {
        self.glwe_eval_linear_transformation_tmp_bytes(dst, src, lt, keys)
    }

    fn ckks_eval_linear_transformation_streamed_into_tmp_bytes<Dst, Src, P, H, K>(
        &self,
        dst: &Dst,
        src: &Src,
        lt: &LinearTransformation<P>,
        keys: &H,
    ) -> usize
    where
        Dst: CKKSCtBounds,
        Src: CKKSCtBounds,
        P: LWEInfos,
        K: GGLWEInfos,
        H: GLWEAutomorphismKeyLayoutHelper<K>,
    {
        self.glwe_eval_linear_transformation_unprepared_rhs_tmp_bytes(dst, src, lt, keys)
    }

    fn ckks_eval_linear_transformation_tmp_bytes<C, P, H, K>(&self, ct: &C, lt: &LinearTransformation<P>, keys: &H) -> usize
    where
        C: CKKSCtBounds,
        P: LtDiagonalScale + IntPolyInfos,
        K: GGLWEInfos,
        H: GLWEAutomorphismKeyLayoutHelper<K>,
    {
        // The extra buffer is the natural post-factor destination the `_assign`
        // wrappers carve; the query proxy carries both its lower logical precision
        // and correspondingly narrower physical size.
        let target = plan_linear_transformation_assign_target(ct, lt)
            .unwrap_or_else(|e| panic!("ckks linear-transformation assign scratch query: {e}"));
        let dst = LinearTransformationAssignDst::new(ct, target);
        self.ckks_eval_linear_transformation_into_tmp_bytes(&dst, ct, lt, keys) + self.glwe_bytes_of_from_infos(&dst)
    }

    fn ckks_eval_linear_transformation_streamed_tmp_bytes<C, P, H, K>(
        &self,
        ct: &C,
        lt: &LinearTransformation<P>,
        keys: &H,
    ) -> usize
    where
        C: CKKSCtBounds,
        P: LtDiagonalScale + IntPolyInfos,
        K: GGLWEInfos,
        H: GLWEAutomorphismKeyLayoutHelper<K>,
    {
        let target = plan_linear_transformation_assign_target(ct, lt)
            .unwrap_or_else(|e| panic!("ckks streamed linear-transformation assign scratch query: {e}"));
        let dst = LinearTransformationAssignDst::new(ct, target);
        self.ckks_eval_linear_transformation_streamed_into_tmp_bytes(&dst, ct, lt, keys) + self.glwe_bytes_of_from_infos(&dst)
    }

    fn ckks_dft_evaluate_tmp_bytes<C, K>(&self, ct: &C, key: &K) -> usize
    where
        C: CKKSCtBounds,
        K: GGLWEInfos,
    {
        // The whole-chain arena, asked before the factors are known: `ct` doubles
        // as the plaintext-operand proxy and `key` stands in for every rotation,
        // so this is the bound form. A caller holding the factors sizes each one
        // exactly through `ckks_eval_linear_transformation_streamed_tmp_bytes`.
        //
        // The chain carves one ciphertext up front and ping-pongs the running
        // value through it, so the per-factor budgets nest inside it rather than
        // each carving their own.
        self.glwe_bytes_of_from_infos(ct) + self.glwe_eval_linear_transformation_unprepared_rhs_bound_tmp_bytes(ct, ct, ct, key)
    }

    // ---------- populate ----------

    fn ckks_prepare_linear_transformation_rhs<P>(
        &self,
        prepared: &mut LinearTransformationPrepared<BE>,
        lt: &LinearTransformation<P>,
        scratch: &mut ScratchArena<'_, BE>,
    ) where
        P: GLWEToBackendRef<BE> + IntPolyInfos + CKKSCtBounds + DiagonalProd<BE>,
    {
        // Stash the plaintext scale exponent while filling the diagonals so eval
        // no longer needs `lt` for `cnv_offset` math. Contract: the diagonals
        // must share one scale/width (the crate's compilers always produce
        // uniform diagonals; the unprepared eval path rejects heterogeneous
        // hand-built inputs via `ensure_uniform_diagonal_scale` — this
        // infallible prepare stashes the first diagonal's scale for all).
        if let Some(first_pt) = lt.first_diagonal_plaintext() {
            prepared.set_log_scale(first_pt.log_delta());
        }
        self.glwe_prepare_linear_transformation_rhs(prepared, lt, scratch);
    }

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
        H: GLWEAutomorphismKeyHelper<K> + GLWEAutomorphismKeyLayoutHelper<K>,
    {
        let cyclotomic_order = self.cyclotomic_order();
        let src_k = src.k();
        for rotation in babies.baby_steps().filter(|&rotation| rotation != 0) {
            if keys
                .get_automorphism_key_layout_for(galois_element(rotation, cyclotomic_order), src_k)
                .is_err()
            {
                return Err(CKKSCompositionError::MissingAutomorphismKey {
                    op: "linear_transformation",
                    rotation,
                    k: src_k.into(),
                }
                .into());
            }
        }
        self.glwe_prepare_linear_transformation_baby_steps(babies, src, keys, scratch);
        Ok(())
    }

    // ---------- eval (caller-supplied baby cache) ----------

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
        H: GLWEAutomorphismKeyHelper<K> + GLWEAutomorphismKeyLayoutHelper<K>,
    {
        let params = LinearTransformationEvalParams::new(dst.k().as_usize(), src, babies, lt)?;
        preflight_linear_transformation_eval::<BE, _, _, _, _>(src, babies, lt, keys, self.cyclotomic_order(), dst.k())?;
        self.glwe_eval_linear_transformation_into(params.cnv_offset, dst, babies, lt, keys, scratch);
        dst.set_log_budget(params.log_budget);
        dst.set_log_delta(params.log_delta);
        // Diagonals are complex in general, so a transformed value leaves the
        // reals unless the caller can prove otherwise.
        dst.set_slots(SlotsKind::Complex);
        Ok(())
    }

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
        H: GLWEAutomorphismKeyHelper<K> + GLWEAutomorphismKeyLayoutHelper<K>,
    {
        let target = plan_linear_transformation_assign_target(dst, lt)?;
        // The natural post-factor destination is carved from scratch (accounted
        // for by `ckks_eval_linear_transformation_tmp_bytes`), not heap-allocated.
        scratch.scope(|scratch_local| {
            let target_layout = LinearTransformationAssignDst::new(dst, target);
            let (mut tmp, mut scratch_local) = scratch_local.take_ckks_ciphertext_scratch(&target_layout, target.meta);
            self.ckks_eval_linear_transformation_into(&mut tmp, dst, babies, lt, keys, &mut scratch_local)?;
            // `ckks_copy` moves both the limbs and the CKKS metadata the eval consumed
            // into `dst` (a plain `glwe_copy` would leave the budget/scale stale).
            self.ckks_copy(dst, &tmp, &mut scratch_local)
        })
    }

    // ---------- eval (self-allocated baby cache) ----------

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
        H: GLWEAutomorphismKeyHelper<K> + GLWEAutomorphismKeyLayoutHelper<K>,
    {
        // Only the (small) input baby cache is materialized here; with a plaintext
        // `lt` the matrix RHS itself is streamed inside the eval.
        let mut babies = LinearTransformationBabySteps::alloc(self, lt.baby_steps(), src);
        self.ckks_prepare_linear_transformation_baby_steps(&mut babies, src, keys, scratch)?;
        self.ckks_eval_linear_transformation_into(dst, src, &babies, lt, keys, scratch)
    }

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
        H: GLWEAutomorphismKeyHelper<K> + GLWEAutomorphismKeyLayoutHelper<K>,
    {
        let target = plan_linear_transformation_assign_target(dst, lt)?;
        // The natural post-factor destination is carved from scratch (accounted
        // for by `ckks_eval_linear_transformation_tmp_bytes`), not heap-allocated.
        scratch.scope(|scratch_local| {
            let target_layout = LinearTransformationAssignDst::new(dst, target);
            let (mut tmp, mut scratch_local) = scratch_local.take_ckks_ciphertext_scratch(&target_layout, target.meta);
            self.ckks_eval_linear_transformation_self_into(&mut tmp, dst, lt, keys, &mut scratch_local)?;
            // `ckks_copy` moves both the limbs and the CKKS metadata the eval consumed
            // into `dst` (a plain `glwe_copy` would leave the budget/scale stale).
            self.ckks_copy(dst, &tmp, &mut scratch_local)
        })
    }
}

/// Natural destination metadata for an in-place linear-transformation factor.
///
/// Its logical precision describes the post-plaintext-product value before
/// giant rotations resolve their keys. Assign wrappers also use that precision
/// to carve a narrower physical destination; ping-pong callers can stamp the
/// same target onto an already-allocated wider buffer.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct LinearTransformationAssignTarget {
    pub(crate) k: TorusPrecision,
    pub(crate) meta: CKKSMeta,
}

impl LinearTransformationAssignTarget {
    pub(crate) fn stamp<Dst: SetCKKSInfos>(self, dst: &mut Dst) {
        dst.set_meta(self.meta);
        dst.set_k(self.k);
    }
}

/// Plans the uncapped, natural post-factor destination shared by the assign
/// wrappers, their exact scratch queries, and DFT ping-pong evaluation.
pub(crate) fn plan_linear_transformation_assign_target<Src, P>(
    src: &Src,
    lt: &LinearTransformation<P>,
) -> Result<LinearTransformationAssignTarget>
where
    Src: CKKSCtBounds,
    P: LtDiagonalScale + IntPolyInfos,
{
    // Using the source width as the destination cap introduces no extra
    // rounding: a plaintext product only spends `pt_log_scale` from the source
    // budget. The returned budget therefore defines the factor's natural precision.
    let (_, log_budget, log_delta, _) = linear_transformation_mul_params(src.k().as_usize(), src, lt)?;
    let k = log_budget
        .checked_add(log_delta)
        .ok_or_else(|| anyhow::anyhow!("linear-transformation destination precision overflows usize"))?;
    let k = u32::try_from(k).map_err(|_| anyhow::anyhow!("linear-transformation destination precision {k} exceeds u32"))?;
    let mut meta = src.meta();
    meta.log_delta = log_delta;
    meta.slots = SlotsKind::Complex;
    Ok(LinearTransformationAssignTarget {
        k: TorusPrecision(k),
        meta,
    })
}

/// Read-only natural-destination layout used identically by assign scratch
/// query and execution.
struct LinearTransformationAssignDst<'a, Src> {
    src: &'a Src,
    target: LinearTransformationAssignTarget,
}

impl<'a, Src> LinearTransformationAssignDst<'a, Src> {
    fn new(src: &'a Src, target: LinearTransformationAssignTarget) -> Self {
        Self { src, target }
    }
}

impl<Src: CKKSCtBounds> LWEInfos for LinearTransformationAssignDst<'_, Src> {
    fn n(&self) -> Degree {
        self.src.n()
    }

    fn base2k(&self) -> Base2K {
        self.src.base2k()
    }

    fn max_size(&self) -> usize {
        self.target.k.as_usize().div_ceil(self.src.base2k().as_usize())
    }

    fn k(&self) -> TorusPrecision {
        self.target.k
    }
}

impl<Src: CKKSCtBounds> GLWEInfos for LinearTransformationAssignDst<'_, Src> {
    fn rank(&self) -> Rank {
        self.src.rank()
    }
}

impl<Src: CKKSCtBounds> CKKSInfos for LinearTransformationAssignDst<'_, Src> {
    fn meta(&self) -> CKKSMeta {
        self.target.meta
    }
}

/// Validated evaluation parameters of one linear transformation.
///
/// Everything the BSGS driver derives from the (uniform) diagonal scale before
/// it touches a limb: the convolution alignment (`cnv_offset` and its limb
/// split), the PROD product width, and the result metadata. Built once by
/// [`Self::new`], which rejects a transform with no diagonals or with
/// non-uniform diagonal scales, so an external chain evaluator can plan a
/// factor without re-deriving this correctness-sensitive arithmetic.
#[derive(Clone, Copy, Debug)]
pub struct LinearTransformationEvalParams {
    /// Convolution alignment between the input and diagonal scales.
    pub cnv_offset: usize,
    /// Limb part of `cnv_offset` in the diagonals' `base2k`.
    pub cnv_offset_hi: usize,
    /// Sub-limb part of `cnv_offset` in the diagonals' `base2k`.
    pub cnv_offset_lo: i64,
    /// Limb count of the per-giant PROD output.
    pub prod_size: usize,
    /// `log_budget` of the result.
    pub log_budget: usize,
    /// `log_delta` of the result (the input's).
    pub log_delta: usize,
}

impl LinearTransformationEvalParams {
    /// Derives the parameters for evaluating `lt` on an input described by
    /// `src` and prepared into `babies`, writing into a destination of width
    /// `dst_k`.
    pub fn new<BE, Src, P>(
        dst_k: usize,
        src: &Src,
        babies: &LinearTransformationBabySteps<BE>,
        lt: &LinearTransformation<P>,
    ) -> Result<Self>
    where
        BE: Backend,
        Src: CKKSCtBounds,
        P: LtDiagonalScale + IntPolyInfos + LWEInfos,
    {
        let (first, log_budget, log_delta, cnv_offset) = linear_transformation_mul_params(dst_k, src, lt)?;
        let (cnv_offset_hi, cnv_offset_lo) = cnv_offset_to_limb_offset(cnv_offset, first.base2k().as_usize());
        let baby_size = babies.size();
        let diagonal_size = first.size();
        let prod_size = baby_size
            .checked_add(diagonal_size)
            .and_then(|width| width.checked_sub(cnv_offset_hi))
            .ok_or_else(|| {
                anyhow::anyhow!(
                    "linear transformation PROD width underflows: baby size {baby_size} + diagonal size {diagonal_size} - cnv_offset_hi {cnv_offset_hi}"
                )
            })?;
        Ok(Self {
            cnv_offset,
            cnv_offset_hi,
            cnv_offset_lo,
            prod_size,
            log_budget,
            log_delta,
        })
    }
}

/// Validates the factor's diagonal metadata and applies the single ct×plaintext
/// arithmetic rule used by explicit destinations and natural assign targets.
fn linear_transformation_mul_params<'a, Src, P>(
    dst_k: usize,
    src: &Src,
    lt: &'a LinearTransformation<P>,
) -> Result<(&'a P, usize, usize, usize)>
where
    Src: CKKSCtBounds,
    P: LtDiagonalScale + IntPolyInfos,
{
    let first = lt
        .first_diagonal_plaintext()
        .ok_or_else(|| anyhow::anyhow!("linear transformation has no diagonals"))?;
    // The diagonal scale (`lt_log_scale`) and its effective torus width `k` are
    // read off the first diagonal. The convolution offset must match the width
    // the diagonal data was masked/positioned at in `cnv_prepare_right` (its
    // effective `k`), which can be below the rounded physical `max_k`.
    let (pt_log_scale, pt_max_k) = (first.lt_log_scale(), first.encoded_k().as_usize());
    ensure_uniform_diagonal_scale(lt, pt_log_scale, pt_max_k)?;
    // ct × (plaintext diagonal): the ct × pt convolution rule, with the diagonal
    // described by just its scale (`pt_log_scale` → rhs `log_delta`) and storage
    // width (`pt_max_k` → rhs `max_k`). Its `log_budget` is dead in this math
    // (`checked_mul_pt_log_budget` reads the rhs budget only for diagnostics), so 0.
    let (log_budget, log_delta, cnv_offset) =
        mul_pt_params_raw(dst_k, src.log_delta(), src.log_budget(), pt_log_scale, 0, pt_max_k)?;
    Ok((first, log_budget, log_delta, cnv_offset))
}

/// Verifies that all automorphism keys required by `lt` are present (keyed by
/// Galois element) and that `babies` covers every baby rotation `lt` needs.
pub(crate) fn preflight_linear_transformation_eval<BE: Backend, Src, P, H, K>(
    src: &Src,
    babies: &LinearTransformationBabySteps<BE>,
    lt: &LinearTransformation<P>,
    keys: &H,
    cyclotomic_order: i64,
    giant_k: TorusPrecision,
) -> Result<()>
where
    Src: CKKSCtBounds,
    K: CKKSAtkBounds<BE>,
    H: GLWEAutomorphismKeyHelper<K> + GLWEAutomorphismKeyLayoutHelper<K>,
{
    ckks_ensure!(
        babies.k() == src.k(),
        "linear-transformation baby cache precision {} does not match source precision {}",
        babies.k(),
        src.k(),
    );
    for rotation in lt.baby_steps().iter().copied() {
        ckks_ensure!(
            babies.contains_baby_step(rotation),
            "missing prepared baby-step rotation {rotation}"
        );
    }
    for rotation in lt
        .giant_steps
        .iter()
        .filter(|gs| !gs.diagonals.is_empty())
        .map(|gs| gs.rot)
        .filter(|&r| r != 0)
    {
        let gal_el = galois_element(rotation, cyclotomic_order);
        let (key, effective_dsize) =
            keys.get_automorphism_key_for(gal_el, giant_k)
                .map_err(|_| CKKSCompositionError::MissingAutomorphismKey {
                    op: "linear_transformation",
                    rotation,
                    k: giant_k.into(),
                })?;
        ckks_ensure!(
            key.p() == gal_el,
            "linear-transformation helper returned Galois element {} for requested giant rotation {rotation} (element {gal_el})",
            key.p(),
        );
        let automorphism_view = GLWEAutomorphismKeyPreparedToBackendRef::<BE>::to_backend_ref(key);
        ckks_ensure!(
            automorphism_view.p() == gal_el,
            "linear-transformation automorphism-key backend view returned Galois element {} for requested giant rotation {rotation} (element {gal_el})",
            automorphism_view.p(),
        );
        let selected = key.with_dsize(effective_dsize);
        let use_ = selected
            .bind_covering_for(giant_k)
            .map_err(|e| anyhow::anyhow!("linear-transformation giant rotation {rotation}: {e}"))?;
        if let GGLWEUse::Active(active) = use_ {
            let prepared = GGLWEPreparedToBackendRef::<BE>::to_backend_ref(key);
            GGLWEPreparedBound::new(prepared, active)
                .map_err(|e| anyhow::anyhow!("linear-transformation giant rotation {rotation}: {e}"))?;
        }
    }
    Ok(())
}

/// Verifies every diagonal shares the first diagonal's scale and storage width.
///
/// The evaluation derives one `cnv_offset` (and the result metadata) from the
/// first diagonal alone; a hand-built transform with heterogeneous diagonal
/// scales would silently mis-scale every other diagonal's contribution. The
/// crate's own compilers (`ckks_encode_linear_transformation_from_diagonals`,
/// the DFT/PaCo factor encoders) always produce uniform diagonals, so this
/// only rejects malformed hand-assembled inputs.
fn ensure_uniform_diagonal_scale<P>(lt: &LinearTransformation<P>, log_scale: usize, max_k: usize) -> Result<()>
where
    P: LtDiagonalScale + IntPolyInfos + LWEInfos,
{
    for gs in &lt.giant_steps {
        for diag in &gs.diagonals {
            let pt = &diag.plaintext;
            ckks_ensure!(
                pt.lt_log_scale() == log_scale && pt.encoded_k().as_usize() == max_k,
                "linear transformation diagonals are not scale-uniform: diagonal (giant rot {}, baby {}) has (log_scale {}, max_k {}) but the first diagonal — which cnv_offset and the result metadata are derived from — has ({log_scale}, {max_k})",
                gs.rot,
                diag.baby,
                pt.lt_log_scale(),
                pt.encoded_k().as_usize(),
            );
        }
    }
    Ok(())
}
