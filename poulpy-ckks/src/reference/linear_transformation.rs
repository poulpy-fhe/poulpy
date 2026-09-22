//! CKKS wrapper for the GLWE-level linear transformation.
//!
//! Computes the plaintext width, convolution offset, and final result metadata,
//! then delegates the actual evaluation to the scheme-agnostic core engine
//! [`GLWELinearTransformations`](trait@poulpy_core::GLWELinearTransformations). See
//! `docs/linear_transformation.md`.

use crate::SlotsKind;
use crate::{CKKSResult as Result, ckks_ensure};
use poulpy_core::layouts::IntPolyInfos;
use poulpy_core::{
    GLWECopy, GLWELinearTransformations, LinearTransformationBabySteps, LinearTransformationGiantStep,
    LinearTransformationPrepared,
    layouts::{
        GGLWEInfos, GLWEToBackendMut, GLWEToBackendRef, GetAutomorphismKey, LWEInfos, TorusPrecision, prepared::PreparedDiagonal,
    },
    reference::linear_transformation::{DiagonalProd, glwe_accumulate_streamed_baby_steps_dft},
};

use poulpy_hal::{
    api::{CnvPVecBytesOf, Convolution, ModuleN},
    layouts::{Backend, CyclotomicOrder, Data, Module, ScratchArena, VecZnxDftBackendMut, ZnxWord, galois_element},
};

use crate::api::CKKSModuleInfos;
use crate::{
    CKKSCompositionError, CKKSCtBounds, CKKSInfos, SetCKKSInfos,
    api::{CKKSCopyOps, CKKSLinearTransformationOps, LinearTransformation, LtDiagonalScale},
    layouts::{CKKSModuleAlloc, CKKSPlaintext, ScratchArenaTakeCKKS},
    reference::mul::mul_pt_params_raw,
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
        lhs: &poulpy_core::LinearTransformationBabySteps<BE>,
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

    fn lt_check_ring(&self, op: &'static str, ring: crate::CKKSRing) -> Result<()> {
        ring.check_plaintext(op, self)
    }
}

/// Resident-diagonal scale: a core [`PreparedDiagonal`] carries the (opaque to the
/// core engine) scale the CKKS prepare step stashed on it via `set_log_scale`.
impl<D: Data, BE: Backend> LtDiagonalScale for PreparedDiagonal<D, BE> {
    fn lt_log_scale(&self) -> usize {
        self.log_scale()
    }

    fn lt_check_ring(&self, op: &'static str, ring: crate::CKKSRing) -> Result<()> {
        let kind = if BE::CYCLOTOMIC_ORDER_FACTOR == 4 {
            crate::CKKSRingKind::ConjugateInvariant
        } else {
            crate::CKKSRingKind::Standard
        };
        let n = self.n().as_usize();
        if kind != ring.kind || n == 0 || !ring.n.as_usize().is_multiple_of(n) {
            return Err(CKKSCompositionError::RingMismatch {
                op,
                expected: ring,
                actual: crate::CKKSRing { kind, n: self.n() },
            }
            .into());
        }
        Ok(())
    }
}

pub(crate) fn check_linear_transformation<P: LtDiagonalScale>(
    op: &'static str,
    ring: crate::CKKSRing,
    lt: &LinearTransformation<P>,
) -> Result<()> {
    for step in &lt.giant_steps {
        for diagonal in &step.diagonals {
            diagonal.plaintext.lt_check_ring(op, ring)?;
        }
    }
    Ok(())
}

fn check_baby_steps<BE: Backend>(
    op: &'static str,
    ring: crate::CKKSRing,
    babies: &LinearTransformationBabySteps<BE>,
) -> Result<()> {
    for rotation in babies.baby_steps() {
        ring.check(
            op,
            crate::CKKSRing {
                kind: ring.kind,
                n: babies.baby_step(rotation).n().into(),
            },
        )?;
    }
    Ok(())
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

    /// The ciphertext stands in for the diagonal, so the budget is an upper bound for a compact diagonal.
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

    /// The ciphertext stands in for the diagonal, so the budget is an upper bound for a compact diagonal.
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

    // ---------- populate ----------

    fn ckks_prepare_linear_transformation_rhs<P>(
        &self,
        prepared: &mut LinearTransformationPrepared<BE>,
        lt: &LinearTransformation<P>,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        P: GLWEToBackendRef<BE> + IntPolyInfos + CKKSCtBounds + DiagonalProd<BE>,
    {
        for step in &lt.giant_steps {
            for diagonal in &step.diagonals {
                self.ckks_ring()
                    .check_plaintext("ckks_prepare_linear_transformation_rhs", &diagonal.plaintext)?;
            }
        }
        check_linear_transformation("ckks_prepare_linear_transformation_rhs", self.ckks_ring(), prepared)?;
        if let Some(first_pt) = lt.first_diagonal_plaintext() {
            for step in &lt.giant_steps {
                for diagonal in &step.diagonals {
                    ckks_ensure!(
                        diagonal.plaintext.log_delta() == first_pt.log_delta()
                            && diagonal.plaintext.encoded_k() == first_pt.encoded_k(),
                        "linear transformation diagonals must share scale and width"
                    );
                }
            }
            prepared.set_log_scale(first_pt.log_delta());
        }
        self.glwe_prepare_linear_transformation_rhs(prepared, lt, scratch);
        Ok(())
    }

    fn ckks_prepare_linear_transformation_baby_steps<Src, H>(
        &self,
        babies: &mut LinearTransformationBabySteps<BE>,
        src: &Src,
        keys: &H,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        Src: GLWEToBackendRef<BE> + CKKSCtBounds,
        H: GetAutomorphismKey<BE>,
    {
        self.ckks_ring()
            .check_ciphertext("ckks_prepare_linear_transformation_baby_steps", src)?;
        check_baby_steps("ckks_prepare_linear_transformation_baby_steps", self.ckks_ring(), babies)?;
        let cyclotomic_order = self.cyclotomic_order();
        let src_k = src.k();
        for rotation in babies.baby_steps().filter(|&rotation| rotation != 0) {
            if keys
                .get_automorphism_key(galois_element(rotation, cyclotomic_order), src_k)
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
        P: DiagonalProd<BE> + LtDiagonalScale + IntPolyInfos,
        H: GetAutomorphismKey<BE>,
    {
        self.ckks_ring()
            .check_ciphertext("ckks_eval_linear_transformation_into", src)?;
        self.ckks_ring()
            .check_ciphertext("ckks_eval_linear_transformation_into", dst)?;
        check_linear_transformation("ckks_eval_linear_transformation_into", self.ckks_ring(), lt)?;
        check_baby_steps("ckks_eval_linear_transformation_into", self.ckks_ring(), babies)?;
        let first = lt
            .first_diagonal_plaintext()
            .ok_or_else(|| anyhow::anyhow!("linear transformation has no diagonals"))?;
        // The diagonal scale (`lt_log_scale`) and its effective torus width `k` are
        // read off the first diagonal. The convolution offset must match the width
        // the diagonal data was positioned at in `cnv_prepare_right` (its
        // effective `k`), which can be below the rounded physical `max_k`.
        let (pt_log_scale, pt_max_k) = (first.lt_log_scale(), first.encoded_k().as_usize());
        ensure_uniform_diagonal_scale(lt, pt_log_scale, pt_max_k)?;
        // ct × (plaintext diagonal): the ct × pt convolution rule, with the diagonal
        // described by just its scale (`pt_log_scale` → rhs `log_delta`) and storage
        // width (`pt_max_k` → rhs `max_k`). Its `log_budget` is dead in this math
        // (`checked_mul_pt_log_budget` reads the rhs budget only for diagnostics), so 0.
        let (res_log_budget, res_log_delta, cnv_offset) = mul_pt_params_raw(
            dst.k().as_usize(),
            src.log_delta(),
            src.log_budget(),
            pt_log_scale,
            0,
            pt_max_k,
        )?;
        let res_k: TorusPrecision = (res_log_budget + res_log_delta).into();
        check_required_keys(lt, babies, keys, self.cyclotomic_order(), res_k)?;
        dst.set_log_budget(res_log_budget);
        dst.set_log_delta(res_log_delta);
        self.glwe_eval_linear_transformation_into(cnv_offset, dst, babies, lt, keys, scratch);
        dst.set_slots(if self.ckks_is_conjugate_invariant() {
            SlotsKind::Real
        } else {
            SlotsKind::Complex
        });
        Ok(())
    }

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
        P: DiagonalProd<BE> + LtDiagonalScale + IntPolyInfos,
        H: GetAutomorphismKey<BE>,
    {
        self.ckks_ring()
            .check_ciphertext("ckks_eval_linear_transformation_assign", dst)?;
        check_linear_transformation("ckks_eval_linear_transformation_assign", self.ckks_ring(), lt)?;
        check_baby_steps("ckks_eval_linear_transformation_assign", self.ckks_ring(), babies)?;
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
        P: DiagonalProd<BE> + LtDiagonalScale + IntPolyInfos,
        H: GetAutomorphismKey<BE>,
    {
        self.ckks_ring()
            .check_ciphertext("ckks_eval_linear_transformation_self_into", src)?;
        self.ckks_ring()
            .check_ciphertext("ckks_eval_linear_transformation_self_into", dst)?;
        check_linear_transformation("ckks_eval_linear_transformation_self_into", self.ckks_ring(), lt)?;
        // Only the (small) input baby cache is materialized here; with a plaintext
        // `lt` the matrix RHS itself is streamed inside the eval.
        let mut babies = LinearTransformationBabySteps::alloc(self, lt.baby_steps(), src);
        self.ckks_prepare_linear_transformation_baby_steps(&mut babies, src, keys, scratch)?;
        self.ckks_eval_linear_transformation_into(dst, src, &babies, lt, keys, scratch)
    }

    fn ckks_eval_linear_transformation_self_assign<Dst, P, H>(
        &self,
        dst: &mut Dst,
        lt: &LinearTransformation<P>,
        keys: &H,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        Dst: GLWEToBackendMut<BE> + GLWEToBackendRef<BE> + CKKSCtBounds + SetCKKSInfos,
        P: DiagonalProd<BE> + LtDiagonalScale + IntPolyInfos,
        H: GetAutomorphismKey<BE>,
    {
        self.ckks_ring()
            .check_ciphertext("ckks_eval_linear_transformation_self_assign", dst)?;
        check_linear_transformation("ckks_eval_linear_transformation_self_assign", self.ckks_ring(), lt)?;
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

/// Verifies that all automorphism keys required by `lt` are present (keyed by
/// Galois element) and that `babies` covers every baby rotation `lt` needs.
fn check_required_keys<BE: Backend, P, H>(
    lt: &LinearTransformation<P>,
    babies: &LinearTransformationBabySteps<BE>,
    keys: &H,
    cyclotomic_order: i64,
    giant_k: TorusPrecision,
) -> Result<()>
where
    H: GetAutomorphismKey<BE>,
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
        keys.get_automorphism_key(gal_el, giant_k)
            .map_err(|_| CKKSCompositionError::MissingAutomorphismKey {
                op: "linear_transformation",
                rotation,
                k: giant_k.into(),
            })?;
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
