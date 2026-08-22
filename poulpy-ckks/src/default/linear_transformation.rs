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
    layouts::{GGLWEInfos, GLWEAutomorphismKeyHelper, GLWEToBackendMut, GLWEToBackendRef, LWEInfos, prepared::PreparedDiagonal},
};
use poulpy_hal::{
    api::{CnvPVecBytesOf, Convolution, ModuleN},
    layouts::{Backend, CyclotomicOrder, Data, Module, ScratchArena, VecZnxDftBackendMut, ZnxWord, galois_element},
};

use crate::{
    CKKSCompositionError, CKKSCtBounds, CKKSInfos, SetCKKSInfos,
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

    fn ckks_prepare_linear_transformation_baby_steps_tmp_bytes<C, K>(&self, ct: &C, key: &K) -> usize
    where
        C: CKKSCtBounds,
        K: GGLWEInfos,
    {
        self.glwe_prepare_linear_transformation_baby_steps_tmp_bytes(ct, key)
    }

    fn ckks_eval_linear_transformation_tmp_bytes<C, K>(&self, ct: &C, key: &K) -> usize
    where
        C: CKKSCtBounds,
        K: GGLWEInfos,
    {
        // `ct` doubles as the plaintext-operand proxy: it bounds the convolution
        // sizes from above, so the result is a safe upper bound. The extra
        // ct-sized buffer is the dst-shaped working copy the `_assign` wrappers
        // carve from scratch (an upper bound for the `_into` paths, which skip it).
        self.glwe_eval_linear_transformation_tmp_bytes(ct, ct, ct, key) + self.glwe_bytes_of_from_infos(ct)
    }

    fn ckks_eval_linear_transformation_streamed_tmp_bytes<C, K>(&self, ct: &C, key: &K) -> usize
    where
        C: CKKSCtBounds,
        K: GGLWEInfos,
    {
        // `ct` doubles as the plaintext-operand proxy (upper bound on diagonal
        // shape). The extra ct-sized buffer covers the `_assign` wrappers'
        // scratch-carved working copy, as above.
        self.glwe_eval_linear_transformation_unprepared_rhs_tmp_bytes(ct, ct, ct, key) + self.glwe_bytes_of_from_infos(ct)
    }

    fn ckks_dft_evaluate_tmp_bytes<C, K>(&self, ct: &C, key: &K) -> usize
    where
        C: CKKSCtBounds,
        K: GGLWEInfos,
    {
        // The chain carves one ciphertext up front and ping-pongs the running
        // value through it, so the per-factor budgets (baby prep, then the
        // widest eval) nest inside it rather than each carving their own.
        self.glwe_bytes_of_from_infos(ct)
            + self
                .ckks_prepare_linear_transformation_baby_steps_tmp_bytes(ct, key)
                .max(self.glwe_eval_linear_transformation_unprepared_rhs_tmp_bytes(ct, ct, ct, key))
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
        H: GLWEAutomorphismKeyHelper<K, BE>,
    {
        let cyclotomic_order = self.cyclotomic_order();
        for rotation in babies.baby_steps().filter(|&rotation| rotation != 0) {
            if keys
                .get_automorphism_key(galois_element(rotation, cyclotomic_order))
                .is_none()
            {
                return Err(CKKSCompositionError::MissingAutomorphismKey {
                    op: "linear_transformation",
                    rotation,
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
        H: GLWEAutomorphismKeyHelper<K, BE>,
    {
        check_required_keys(lt, babies, keys, self.cyclotomic_order())?;

        let params = LinearTransformationEvalParams::new(dst.k().as_usize(), src, babies, lt)?;
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
        H: GLWEAutomorphismKeyHelper<K, BE>,
    {
        // The dst-shaped working copy is carved from scratch (accounted for by
        // `ckks_eval_linear_transformation_tmp_bytes`), not heap-allocated.
        scratch.scope(|scratch_local| {
            let (mut tmp, mut scratch_local) = scratch_local.take_ckks_ciphertext_like_scratch(dst);
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
        H: GLWEAutomorphismKeyHelper<K, BE>,
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
        H: GLWEAutomorphismKeyHelper<K, BE>,
    {
        // The dst-shaped working copy is carved from scratch (accounted for by
        // `ckks_eval_linear_transformation_tmp_bytes`), not heap-allocated.
        scratch.scope(|scratch_local| {
            let (mut tmp, mut scratch_local) = scratch_local.take_ckks_ciphertext_like_scratch(dst);
            self.ckks_eval_linear_transformation_self_into(&mut tmp, dst, lt, keys, &mut scratch_local)?;
            // `ckks_copy` moves both the limbs and the CKKS metadata the eval consumed
            // into `dst` (a plain `glwe_copy` would leave the budget/scale stale).
            self.ckks_copy(dst, &tmp, &mut scratch_local)
        })
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
        let (cnv_offset_hi, cnv_offset_lo) = cnv_offset_to_limb_offset(cnv_offset, first.base2k().as_usize());
        Ok(Self {
            cnv_offset,
            cnv_offset_hi,
            cnv_offset_lo,
            prod_size: babies.size() + first.size() - cnv_offset_hi,
            log_budget,
            log_delta,
        })
    }
}

/// Verifies that all automorphism keys required by `lt` are present (keyed by
/// Galois element) and that `babies` covers every baby rotation `lt` needs.
fn check_required_keys<BE: Backend, P, H, K>(
    lt: &LinearTransformation<P>,
    babies: &LinearTransformationBabySteps<BE>,
    keys: &H,
    cyclotomic_order: i64,
) -> Result<()>
where
    K: CKKSAtkBounds<BE>,
    H: GLWEAutomorphismKeyHelper<K, BE>,
{
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
        if keys.get_automorphism_key(gal_el).is_none() {
            return Err(CKKSCompositionError::MissingAutomorphismKey {
                op: "linear_transformation",
                rotation,
            }
            .into());
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
