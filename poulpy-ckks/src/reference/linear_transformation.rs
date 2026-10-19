//! CKKS wrapper for the GLWE-level linear transformation.
//!
//! Computes the plaintext width, convolution offset, and final result metadata,
//! then delegates the actual evaluation to the scheme-agnostic core engine
//! [`GLWELinearTransformations`](trait@poulpy_core::GLWELinearTransformations). See
//! `docs/linear_transformation.md`.

use crate::SlotsKind;
use crate::{CKKSError, CKKSResult as Result, ckks_ensure};
use poulpy_core::layouts::GLWEPlaintextInfos;
use poulpy_core::layouts::IntPolyInfos;
use poulpy_core::{
    GLWELinearTransformations, LinearTransformationBabySteps, LinearTransformationGiantStep, LinearTransformationPrepared,
    layouts::{
        GGLWEInfos, GLWEInfos, GLWEToBackendMut, GLWEToBackendRef, GetAutomorphismKey, LWEInfos, TorusPrecision,
        prepared::PreparedDiagonal,
    },
    reference::linear_transformation::{DiagonalProd, glwe_accumulate_streamed_baby_steps_dft},
};
use poulpy_hal::{
    api::{CnvPVecBytesOf, Convolution, ModuleN},
    layouts::{Backend, CyclotomicOrder, Data, Module, Ring, ScratchArena, VecZnxDftBackendMut, ZnxWord, galois_element},
};

use crate::{
    CKKSCompositionError, CKKSCtBounds, CKKSInfos, SetCKKSInfos,
    api::{LinearTransformation, LtDiagonalMeta},
    layouts::CKKSPlaintext,
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
impl<BE: Backend, D: Data> DiagonalProd<BE> for CKKSPlaintext<D, BE::ZnxWord, BE::Ring>
where
    CKKSPlaintext<D, BE::ZnxWord, BE::Ring>: GLWEToBackendRef<BE>,
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

/// Streamed-diagonal metadata: a [`CKKSPlaintext`] carries its scale as `log_delta`.
impl<D: Data, W: ZnxWord, R: Ring> LtDiagonalMeta for CKKSPlaintext<D, W, R> {
    fn lt_log_scale(&self) -> usize {
        self.log_delta()
    }

    fn lt_slots(&self) -> SlotsKind {
        self.slots()
    }
}

/// Resident-diagonal metadata: a core [`PreparedDiagonal`] carries the (opaque to
/// the core engine) scale and real-slot claim the CKKS prepare step stashed on it.
impl<D: Data, BE: Backend> LtDiagonalMeta for PreparedDiagonal<D, BE> {
    fn lt_log_scale(&self) -> usize {
        self.log_scale()
    }

    fn lt_slots(&self) -> SlotsKind {
        if self.real_slots() {
            SlotsKind::Real
        } else {
            SlotsKind::Complex
        }
    }
}

/// The reference algorithms of [`CKKSLinearTransformationImpl`](crate::oep::CKKSLinearTransformationImpl), which
/// `impl_ckks_linear_transformation_reference!` forwards to and an override can call for the methods it keeps.
pub trait CKKSLinearTransformationReference<BE: Backend> {
    fn ckks_prepare_linear_transformation_rhs_tmp_bytes_reference<P>(&self, pt_infos: &P) -> usize
    where
        Self: GLWELinearTransformations<BE>,
        P: LWEInfos,
    {
        self.glwe_prepare_linear_transformation_rhs_tmp_bytes(pt_infos)
    }

    fn ckks_prepare_linear_transformation_baby_steps_tmp_bytes_reference<C, K>(&self, ct: &C, key: &K) -> usize
    where
        Self: GLWELinearTransformations<BE>,
        C: CKKSCtBounds,
        K: GGLWEInfos,
    {
        self.glwe_prepare_linear_transformation_baby_steps_tmp_bytes(ct, key)
    }

    /// The ciphertext stands in for the diagonal, so the budget is an upper bound for a compact diagonal.
    fn ckks_eval_linear_transformation_tmp_bytes_reference<C, K>(&self, ct: &C, key: &K) -> usize
    where
        Self: GLWELinearTransformations<BE> + GLWEBytesOf<BE>,
        C: CKKSCtBounds,
        K: GGLWEInfos,
    {
        // `ct` doubles as the plaintext-operand proxy: it bounds the convolution
        // sizes from above, so the result is a safe upper bound. The extra
        // ct-sized buffer is the dst-shaped working copy the `_assign` wrappers
        // carve from scratch (an upper bound for the `_into` paths, which skip it).
        self.glwe_eval_linear_transformation_tmp_bytes(ct, ct, ct, key) + BE::scratch_aligned(self.glwe_bytes_of_from_infos(ct))
    }

    /// The ciphertext stands in for the diagonal, so the budget is an upper bound for a compact diagonal.
    fn ckks_eval_linear_transformation_streamed_tmp_bytes_reference<C, K>(&self, ct: &C, key: &K) -> usize
    where
        Self: GLWELinearTransformations<BE> + GLWEBytesOf<BE>,
        C: CKKSCtBounds,
        K: GGLWEInfos,
    {
        // `ct` doubles as the plaintext-operand proxy (upper bound on diagonal
        // shape). The extra ct-sized buffer covers the `_assign` wrappers'
        // scratch-carved working copy, as above.
        self.glwe_eval_linear_transformation_unprepared_rhs_tmp_bytes(ct, ct, ct, key)
            + BE::scratch_aligned(self.glwe_bytes_of_from_infos(ct))
    }

    /// The ciphertext stands in for the diagonal, so the budget is an upper bound for a compact diagonal.
    fn ckks_eval_linear_transformation_sum_tmp_bytes_reference<C, K>(&self, ct: &C, key: &K) -> usize
    where
        Self: GLWELinearTransformations<BE>,
        C: CKKSCtBounds,
        K: GGLWEInfos,
    {
        // No `_assign` wrapper, so no dst-shaped working copy.
        self.glwe_eval_linear_transformation_sum_tmp_bytes(ct, ct, ct, key)
    }

    fn ckks_prepare_linear_transformation_rhs_reference<P>(
        &self,
        prepared: &mut LinearTransformationPrepared<BE>,
        lt: &LinearTransformation<P>,
        scratch: &mut ScratchArena<'_, BE>,
    ) where
        Self: GLWELinearTransformations<BE>,
        P: GLWEToBackendRef<BE> + IntPolyInfos + CKKSCtBounds + DiagonalProd<BE>,
    {
        // Stash the plaintext scale exponent and real-slot claim while filling the
        // diagonals so eval no longer needs `lt` for `cnv_offset` math and the
        // result's slot kind. Contract: the diagonals must share one scale, width
        // and slot kind (the crate's compilers always produce uniform diagonals;
        // the unprepared eval path rejects heterogeneous hand-built inputs via
        // `ensure_uniform_diagonals`, so this infallible prepare stashes the first
        // diagonal's for all; debug builds assert the contract).
        if let Some(first_pt) = lt.first_diagonal_plaintext() {
            debug_assert!(
                lt.giant_steps.iter().flat_map(|gs| &gs.diagonals).all(|diag| {
                    let pt = &diag.plaintext;
                    pt.log_delta() == first_pt.log_delta()
                        && pt.encoded_k() == first_pt.encoded_k()
                        && pt.slots() == first_pt.slots()
                }),
                "linear transformation diagonals must share one scale, width and slot kind"
            );
            prepared.set_log_scale(first_pt.log_delta());
            prepared.set_real_slots(first_pt.slots().is_real());
        }
        self.glwe_prepare_linear_transformation_rhs(prepared, lt, scratch);
    }

    fn ckks_prepare_linear_transformation_baby_steps_reference<Src, H>(
        &self,
        babies: &mut LinearTransformationBabySteps<BE>,
        src: &Src,
        keys: &H,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        Self: GLWELinearTransformations<BE> + CyclotomicOrder,
        Src: GLWEToBackendRef<BE> + CKKSCtBounds,
        H: GetAutomorphismKey<BE>,
    {
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

    fn ckks_eval_linear_transformation_into_reference<Dst, Src, P, H>(
        &self,
        dst: &mut Dst,
        src: &Src,
        babies: &LinearTransformationBabySteps<BE>,
        lt: &LinearTransformation<P>,
        keys: &H,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        Self: GLWELinearTransformations<BE> + CyclotomicOrder,
        Dst: GLWEToBackendMut<BE> + CKKSCtBounds + SetCKKSInfos,
        Src: GLWEToBackendRef<BE> + CKKSCtBounds,
        P: DiagonalProd<BE> + LtDiagonalMeta + IntPolyInfos,
        H: GetAutomorphismKey<BE>,
    {
        let first = lt
            .first_diagonal_plaintext()
            .ok_or_else(|| anyhow::anyhow!("linear transformation has no diagonals"))?;
        // The diagonal scale (`lt_log_scale`), its effective torus width `k` and the
        // slot kind are read off the first diagonal. The convolution offset must
        // match the width the diagonal data was positioned at in `cnv_prepare_right`
        // (its effective `k`), which can be below the rounded physical `max_k`.
        let (pt_log_scale, pt_max_k, pt_slots) = (first.lt_log_scale(), first.encoded_k().as_usize(), first.lt_slots());
        ensure_uniform_diagonals(lt, pt_log_scale, pt_max_k, pt_slots)?;
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
        debug_assert_eq!(res_k, crate::api::ckks_linear_transformation_output_k(lt, src.k(), dst.k()));
        check_required_keys(lt, babies, keys, self.cyclotomic_order(), res_k)?;
        // `Σ diag ⊙ rot(src)` has real slots when `src` and the diagonals do.
        let slots = src.slots().join(pt_slots);
        dst.set_log_budget(res_log_budget);
        dst.set_log_delta(res_log_delta);
        self.glwe_eval_linear_transformation_into(cnv_offset, dst, babies, lt, keys, scratch);
        dst.set_slots(slots);
        Ok(())
    }

    fn ckks_eval_linear_transformation_sum_into_reference<Dst, Src, H>(
        &self,
        dst: &mut Dst,
        srcs: &[&Src],
        babies: &[&LinearTransformationBabySteps<BE>],
        lts: &[&LinearTransformationPrepared<BE>],
        keys: &H,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        Self: GLWELinearTransformations<BE> + CyclotomicOrder,
        Dst: GLWEToBackendMut<BE> + CKKSCtBounds + SetCKKSInfos,
        Src: GLWEToBackendRef<BE> + CKKSCtBounds,
        H: GetAutomorphismKey<BE>,
    {
        if babies.len() != srcs.len() || lts.len() != srcs.len() {
            return Err(invalid_sum("srcs, babies and lts differ in length".into()));
        }
        let &src = srcs
            .first()
            .ok_or_else(|| invalid_sum("linear transformation sum has no terms".into()))?;
        let (cache_0, lt) = (babies[0], lts[0]);
        let first = lt
            .first_diagonal_plaintext()
            .ok_or_else(|| invalid_sum("linear transformation has no diagonals".into()))?;
        let (pt_log_scale, pt_max_k, pt_slots) = (first.lt_log_scale(), first.encoded_k().as_usize(), first.lt_slots());
        // Core sums each giant step's products over the terms before one rotation,
        // at one `cnv_offset`, so the terms must agree on everything those read.
        let diagonal = |lt: &LinearTransformationPrepared<BE>| {
            lt.first_diagonal_plaintext()
                .map(|d| (d.glwe_layout(), d.max_size(), d.lt_log_scale(), d.lt_slots()))
        };
        let source = |src: &Src| (src.glwe_layout(), src.log_delta());
        // Core reads each source through its baby cache. A cache without the rotations its transform reads may hold
        // none, so its layout is read only once those are checked.
        let cache = |babies: &LinearTransformationBabySteps<BE>| (babies.size(), babies.cols());
        check_babies(lt, cache_0)?;
        for (t, ((&src_t, &babies_t), &lt_t)) in srcs.iter().zip(babies).zip(lts).enumerate().skip(1) {
            let mismatch = |what: &str| {
                Err(invalid_sum(format!(
                    "summed linear transformation {t} does not share the first one's {what}"
                )))
            };
            if !lt_t
                .giant_steps
                .iter()
                .map(|gs| gs.rot)
                .eq(lt.giant_steps.iter().map(|gs| gs.rot))
            {
                return mismatch("giant schedule");
            }
            if diagonal(lt_t) != diagonal(lt) {
                return mismatch("diagonal layout, width, scale and slot kind");
            }
            if source(src_t) != source(src) {
                return mismatch("source layout and scale");
            }
            check_babies(lt_t, babies_t)?;
            if cache(babies_t) != cache(cache_0) {
                return mismatch("baby-step cache layout");
            }
        }
        for &lt_t in lts {
            ensure_uniform_diagonals(lt_t, pt_log_scale, pt_max_k, pt_slots)?;
        }
        // The terms share the source and diagonal metadata, so the first term's
        // ct × pt rule holds for all of them (see `ckks_eval_linear_transformation_into`).
        let (res_log_budget, res_log_delta, cnv_offset) = mul_pt_params_raw(
            dst.k().as_usize(),
            src.log_delta(),
            src.log_budget(),
            pt_log_scale,
            0,
            pt_max_k,
        )?;
        let res_k: TorusPrecision = (res_log_budget + res_log_delta).into();
        for (&babies_t, &lt_t) in babies.iter().zip(lts) {
            check_required_keys(lt_t, babies_t, keys, self.cyclotomic_order(), res_k)?;
        }
        // The sum has real slots when every source and the diagonals do.
        let slots = srcs.iter().fold(pt_slots, |slots, src_t| slots.join(src_t.slots()));
        dst.set_log_budget(res_log_budget);
        dst.set_log_delta(res_log_delta);
        self.glwe_eval_linear_transformation_sum_into(cnv_offset, dst, babies, lts, keys, scratch);
        dst.set_slots(slots);
        Ok(())
    }
}

impl<BE: Backend> CKKSLinearTransformationReference<BE> for Module<BE> {}

fn invalid_sum(reason: String) -> CKKSError {
    CKKSCompositionError::InvalidOperand {
        op: "ckks_eval_linear_transformation_sum_into",
        reason,
    }
    .into()
}

/// Verifies that `babies` covers every baby rotation `lt` needs.
fn check_babies<BE: Backend, P>(lt: &LinearTransformation<P>, babies: &LinearTransformationBabySteps<BE>) -> Result<()> {
    for rotation in lt.baby_steps().iter().copied() {
        ckks_ensure!(
            babies.contains_baby_step(rotation),
            "missing prepared baby-step rotation {rotation}"
        );
    }
    Ok(())
}

/// Verifies that `babies` covers every baby rotation `lt` needs and that all
/// automorphism keys required by `lt` are present (keyed by Galois element).
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
    check_babies(lt, babies)?;
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

/// Verifies every diagonal shares the first diagonal's scale, storage width and slot kind.
///
/// The evaluation derives one `cnv_offset` (and the result metadata) from the
/// first diagonal alone; a hand-built transform with heterogeneous diagonal
/// scales would silently mis-scale every other diagonal's contribution, and
/// heterogeneous slot kinds would mislabel the result. The crate's own compilers
/// (`ckks_encode_linear_transformation_from_diagonals`, the DFT/PaCo factor
/// encoders) always produce uniform diagonals, so this only rejects malformed
/// hand-assembled inputs.
fn ensure_uniform_diagonals<P>(lt: &LinearTransformation<P>, log_scale: usize, max_k: usize, slots: SlotsKind) -> Result<()>
where
    P: LtDiagonalMeta + IntPolyInfos + LWEInfos,
{
    for gs in &lt.giant_steps {
        for diag in &gs.diagonals {
            let pt = &diag.plaintext;
            ckks_ensure!(
                pt.lt_log_scale() == log_scale && pt.encoded_k().as_usize() == max_k && pt.lt_slots() == slots,
                "linear transformation diagonals are not uniform: diagonal (giant rot {}, baby {}) has (log_scale {}, max_k {}, {:?}) but the first diagonal, which cnv_offset and the result metadata are derived from, has ({log_scale}, {max_k}, {slots:?})",
                gs.rot,
                diag.baby,
                pt.lt_log_scale(),
                pt.encoded_k().as_usize(),
                pt.lt_slots(),
            );
        }
    }
    Ok(())
}
