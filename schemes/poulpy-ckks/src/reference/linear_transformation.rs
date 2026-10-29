//! CKKS wrapper for the GLWE-level linear transformation.
//!
//! Computes the plaintext width, convolution offset, and final result metadata,
//! then delegates the actual evaluation to the scheme-agnostic core engine
//! [`GLWELinearTransformations`](trait@poulpy_core::GLWELinearTransformations). See
//! `docs/guides/linear_transformation.md`.

use crate::{CKKSError, CKKSResult as Result, ckks_ensure, ckks_ensure_operand, operand_cyclotomic_order};
use crate::{GLWEPlaintextMeta, Scale, SlotsKind};
use poulpy_core::layouts::GLWEPlaintextInfos;
use poulpy_core::layouts::LinearTransformationPlan;
use poulpy_core::{
    DiagonalProd, GLWELinearTransformations, LinearTransformationBabySteps, LinearTransformationPrepared,
    layouts::{
        BsgsRotations, DiagonalOperand, GGLWEInfos, GLWEInfos, GLWEToBackendMut, GLWEToBackendRef, GetAutomorphismKey,
        GetTensorKey, LWEInfos, TorusPrecision, prepared::PreparedDiagonal,
    },
};
use poulpy_hal::layouts::{Backend, CyclotomicOrder, Data, Module, Ring, ScratchArena, ZnxWord, galois_element};

use crate::{
    CKKSCompositionError, CKKSCtBounds, CKKSInfos, CKKSLayout, SetCKKSInfos,
    api::{CKKSCopyOps, LinearTransformation, LtDiagonalMeta, ckks_linear_transformation_output_layout},
    layouts::{
        CKKSCiphertext, CKKSPlaintext,
        validation::{ensure_prepared_transformation_layout, ensure_uniform_diagonals},
    },
    mul_ct_params_raw, mul_pt_params_raw,
    reference::{
        dot_product::check_lazy_base2k,
        mul::{check_prepared_layout, tensor_key_at},
    },
};
use poulpy_core::GLWEBytesOf;

/// A streamed plaintext diagonal: where the resident path (core's [`PreparedDiagonal`]) fuses already-prepared
/// diagonals, a [`CKKSPlaintext`] diagonal is a rank-0 GLWE view the product prepares on the fly.
impl<BE: Backend, D: Data> DiagonalProd<BE> for CKKSPlaintext<D, BE::ZnxWord, BE::Ring>
where
    CKKSPlaintext<D, BE::ZnxWord, BE::Ring>: GLWEToBackendRef<BE>,
{
    fn operand(&self) -> DiagonalOperand<'_, BE> {
        DiagonalOperand::Unprepared(GLWEToBackendRef::to_backend_ref(self))
    }
}

/// A streamed ciphertext diagonal: a GLWE view of the source's rank the product prepares on the fly and multiplies as
/// a ciphertext.
impl<BE: Backend, D: Data> DiagonalProd<BE> for CKKSCiphertext<D, BE::ZnxWord, BE::Ring>
where
    CKKSCiphertext<D, BE::ZnxWord, BE::Ring>: GLWEToBackendRef<BE>,
{
    fn operand(&self) -> DiagonalOperand<'_, BE> {
        DiagonalOperand::Unprepared(GLWEToBackendRef::to_backend_ref(self))
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

    fn lt_log_sparsity(&self, log_n: usize) -> usize {
        log_n.saturating_sub(self.log_n())
    }
}

/// Streamed ciphertext-diagonal metadata: a [`CKKSCiphertext`] carries its scale as `log_delta` and its sparsity as a
/// label.
impl<D: Data, W: ZnxWord, R: Ring> LtDiagonalMeta for CKKSCiphertext<D, W, R> {
    fn lt_log_scale(&self) -> usize {
        self.log_delta()
    }

    fn lt_slots(&self) -> SlotsKind {
        self.slots()
    }

    fn lt_log_sparsity(&self, _log_n: usize) -> usize {
        self.log_sparsity()
    }

    fn lt_ciphertext_k(&self) -> Option<TorusPrecision> {
        Some(self.k())
    }
}

/// Resident-diagonal metadata: a core [`PreparedDiagonal`] carries the plaintext meta the CKKS prepare step stashed
/// on it.
impl<D: Data, BE: Backend> LtDiagonalMeta for PreparedDiagonal<D, BE> {
    fn lt_log_scale(&self) -> usize {
        let Scale::Log(log_scale) = self.plaintext_meta().scale;
        log_scale
    }

    fn lt_slots(&self) -> SlotsKind {
        self.plaintext_meta().slots
    }

    /// Its degree's sparsity, or the sparsity its ciphertexts claimed.
    fn lt_log_sparsity(&self, log_n: usize) -> usize {
        log_n.saturating_sub(self.log_n()).max(self.plaintext_meta().log_sparsity)
    }

    fn lt_ciphertext_k(&self) -> Option<TorusPrecision> {
        (self.rank().as_usize() > 0).then(|| self.k())
    }
}

/// The reference algorithms of [`CKKSLinearTransformationImpl`](crate::oep::CKKSLinearTransformationImpl), which
/// `impl_ckks_linear_transformation_reference_full!` forwards to and an override can call for the methods it keeps.
pub trait CKKSLinearTransformationReference<BE: Backend> {
    fn ckks_prepare_linear_transformation_rhs_tmp_bytes_reference<P>(&self, pt_infos: &P) -> usize
    where
        Self: GLWELinearTransformations<BE>,
        P: GLWEInfos,
    {
        self.glwe_prepare_linear_transformation_rhs_tmp_bytes(pt_infos)
    }

    fn ckks_prepare_linear_transformation_baby_steps_tmp_bytes_reference<C, K>(
        &self,
        ct: &C,
        plan: &LinearTransformationPlan,
        key: &K,
    ) -> usize
    where
        Self: GLWELinearTransformations<BE>,
        C: CKKSCtBounds,
        K: GGLWEInfos,
    {
        self.glwe_prepare_linear_transformation_baby_steps_tmp_bytes(ct, plan, key)
    }

    fn ckks_eval_linear_transformation_tmp_bytes_reference<C, D, K, T>(
        &self,
        ct: &C,
        diagonal: &D,
        plan: &LinearTransformationPlan,
        tsk: &T,
        key: &K,
    ) -> usize
    where
        Self: GLWELinearTransformations<BE> + GLWEBytesOf<BE> + CKKSCopyOps<BE>,
        C: CKKSCtBounds,
        D: GLWEInfos,
        K: GGLWEInfos,
        T: GGLWEInfos,
    {
        // `ct` stands for the destination and the source. The extra ct-sized buffer is the dst-shaped working copy the
        // `_assign` wrappers carve from scratch and copy back with the selected `ckks_copy` (an upper bound for the
        // `_into` paths, which skip both).
        self.glwe_eval_linear_transformation_tmp_bytes(ct, ct, diagonal, plan, tsk, key)
            .max(self.ckks_copy_tmp_bytes(ct, ct))
            + BE::scratch_aligned(self.glwe_bytes_of_from_infos(ct))
    }

    /// Sized at the module degree, an upper bound for a diagonal below the ring degree.
    fn ckks_eval_linear_transformation_streamed_tmp_bytes_reference<C, D, K, T>(
        &self,
        ct: &C,
        diagonal: &D,
        plan: &LinearTransformationPlan,
        tsk: &T,
        key: &K,
    ) -> usize
    where
        Self: GLWELinearTransformations<BE> + GLWEBytesOf<BE> + CKKSCopyOps<BE>,
        C: CKKSCtBounds,
        D: GLWEInfos,
        K: GGLWEInfos,
        T: GGLWEInfos,
    {
        // The extra ct-sized buffer and the copy cover the `_assign` wrappers' scratch-carved working copy, as above.
        self.glwe_eval_linear_transformation_unprepared_rhs_tmp_bytes(ct, ct, diagonal, plan, tsk, key)
            .max(self.ckks_copy_tmp_bytes(ct, ct))
            + BE::scratch_aligned(self.glwe_bytes_of_from_infos(ct))
    }

    /// The streamed sum's query, `ct` standing for the destination and every source.
    fn ckks_eval_linear_transformation_sum_streamed_tmp_bytes_reference<C, D, K, T>(
        &self,
        ct: &C,
        diagonal: &D,
        plan: &LinearTransformationPlan,
        tsk: &T,
        key: &K,
    ) -> usize
    where
        Self: GLWELinearTransformations<BE>,
        C: CKKSCtBounds,
        D: GLWEInfos,
        K: GGLWEInfos,
        T: GGLWEInfos,
    {
        self.glwe_eval_linear_transformation_sum_unprepared_rhs_tmp_bytes(ct, ct, diagonal, plan, tsk, key)
    }

    fn ckks_eval_linear_transformation_sum_tmp_bytes_reference<C, D, K, T>(
        &self,
        ct: &C,
        diagonal: &D,
        plan: &LinearTransformationPlan,
        tsk: &T,
        key: &K,
    ) -> usize
    where
        Self: GLWELinearTransformations<BE>,
        C: CKKSCtBounds,
        D: GLWEInfos,
        K: GGLWEInfos,
        T: GGLWEInfos,
    {
        // No `_assign` wrapper, so no dst-shaped working copy.
        self.glwe_eval_linear_transformation_sum_tmp_bytes(ct, ct, diagonal, plan, tsk, key)
    }

    /// `InvalidOperand`, before any write, for a `prepared` allocated for another layout than `lt`'s or diagonals of
    /// `lt` not sharing one scale, width and slot kind.
    fn ckks_prepare_linear_transformation_rhs_reference<P>(
        &self,
        prepared: &mut LinearTransformationPrepared<BE>,
        lt: &LinearTransformation<P>,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        Self: GLWELinearTransformations<BE>,
        P: GLWEToBackendRef<BE> + CKKSCtBounds,
    {
        const OP: &str = "ckks_prepare_linear_transformation_rhs";
        ensure_prepared_transformation_layout(OP, prepared, lt)?;
        ensure_uniform_diagonals(OP, lt, |pt| (pt.log_delta(), pt.max_k().as_usize(), pt.slots()))?;
        let first_pt = lt.first_diagonal_plaintext();
        if let Some(first_pt) = first_pt {
            ckks_ensure!(
                scratch.available() >= self.ckks_prepare_linear_transformation_rhs_tmp_bytes_reference(first_pt),
                "{OP}: scratch below its query"
            );
        }
        self.glwe_prepare_linear_transformation_rhs(prepared, lt, scratch);
        // The first diagonal's scale and slot kind are stashed for all, so eval needs no `lt` for its `cnv_offset` and
        // slot kind. A plaintext is sparse by its degree; ciphertexts by their label, the products sparse where all are.
        if let Some(first_pt) = first_pt {
            let log_sparsity = lt
                .giant_steps
                .iter()
                .flat_map(|gs| &gs.diagonals)
                .filter(|d| d.plaintext.rank().as_usize() > 0)
                .map(|d| d.plaintext.log_sparsity())
                .min()
                .unwrap_or(0);
            prepared.set_plaintext_meta(GLWEPlaintextMeta {
                scale: Scale::Log(first_pt.log_delta()),
                slots: first_pt.slots(),
                log_sparsity,
            });
        }
        Ok(())
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
        let cyclotomic_order =
            operand_cyclotomic_order("ckks_prepare_linear_transformation_baby_steps", self, src.n().as_usize())?;
        let src_k = src.k();
        for rotation in babies.rotations().baby_key_rotations(babies.baby_steps()) {
            if let Err(source) = keys.get_automorphism_key(galois_element(rotation, cyclotomic_order), src_k) {
                return Err(CKKSCompositionError::MissingAutomorphismKey {
                    op: "ckks_prepare_linear_transformation_baby_steps",
                    rotation,
                    k: src_k.into(),
                    source,
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
        P: DiagonalProd<BE> + LtDiagonalMeta,
        H: GetAutomorphismKey<BE> + GetTensorKey<BE>,
    {
        const OP: &str = "ckks_eval_linear_transformation_into";
        let first = lt.first_diagonal_plaintext().ok_or_else(|| -> CKKSError {
            CKKSCompositionError::InvalidOperand {
                op: OP,
                reason: "linear transformation has no diagonals".into(),
            }
            .into()
        })?;
        ensure_uniform_diagonals(OP, lt, lt_diagonal_meta)?;
        let (res_log_budget, res_log_delta, cnv_offset) = lt_rule(OP, &*dst, src, babies, first, keys)?;
        let res_k: TorusPrecision = (res_log_budget + res_log_delta).into();
        let src_layout = CKKSLayout {
            glwe_layout: src.glwe_layout(),
            meta: src.meta(),
        };
        let out = ckks_linear_transformation_output_layout(lt, src_layout, dst.k());
        debug_assert_eq!(res_k, out.k());
        let cyclotomic_order = operand_cyclotomic_order(OP, self, src.n().as_usize())?;
        check_required_keys(OP, lt, babies, keys, cyclotomic_order, res_k)?;
        dst.set_log_budget(res_log_budget);
        dst.set_log_delta(res_log_delta);
        dst.set_log_sparsity(out.meta.log_sparsity);
        self.glwe_eval_linear_transformation_into(cnv_offset, dst, babies, lt, keys, scratch);
        dst.set_slots(out.meta.slots);
        Ok(())
    }

    fn ckks_eval_linear_transformation_sum_into_reference<Dst, Src, P, H>(
        &self,
        dst: &mut Dst,
        srcs: &[&Src],
        babies: &[&LinearTransformationBabySteps<BE>],
        lts: &[&LinearTransformation<P>],
        keys: &H,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        Self: GLWELinearTransformations<BE> + CyclotomicOrder,
        Dst: GLWEToBackendMut<BE> + CKKSCtBounds + SetCKKSInfos,
        Src: GLWEToBackendRef<BE> + CKKSCtBounds,
        P: DiagonalProd<BE> + LtDiagonalMeta,
        H: GetAutomorphismKey<BE> + GetTensorKey<BE>,
    {
        const OP: &str = "ckks_eval_linear_transformation_sum_into";
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
        // Core sums each giant step's products over the terms before one rotation,
        // at one `cnv_offset`, so the terms must agree on everything those read.
        let diagonal = |lt: &LinearTransformation<P>| {
            lt.first_diagonal_plaintext()
                .map(|d| (d.glwe_layout(), d.max_size(), d.lt_log_scale(), d.lt_slots()))
        };
        let source = |src: &Src| (src.glwe_layout(), src.log_delta());
        // Core reads each source through its baby cache. A cache without the rotations its transform reads may hold
        // none, so its layout is read only once those are checked.
        let cache = |babies: &LinearTransformationBabySteps<BE>| (babies.size(), babies.cols(), babies.k());
        check_babies(OP, lt, cache_0)?;
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
            check_babies(OP, lt_t, babies_t)?;
            if cache(babies_t) != cache(cache_0) {
                return mismatch("baby-step cache layout");
            }
        }
        for &lt_t in lts {
            ensure_uniform_diagonals(OP, lt_t, lt_diagonal_meta)?;
        }
        // The terms share the source and diagonal metadata, so the first term's
        // rule holds for all of them (see `ckks_eval_linear_transformation_into`).
        let (res_log_budget, res_log_delta, cnv_offset) = lt_rule(OP, &*dst, src, cache_0, first, keys)?;
        let res_k: TorusPrecision = (res_log_budget + res_log_delta).into();
        let cyclotomic_order = operand_cyclotomic_order(OP, self, dst.n().as_usize())?;
        for (&babies_t, &lt_t) in babies.iter().zip(lts) {
            check_required_keys(OP, lt_t, babies_t, keys, cyclotomic_order, res_k)?;
        }
        // The sum has real slots when every source and the diagonals do, and is sparse where all of them are.
        let slots = srcs.iter().fold(first.lt_slots(), |slots, src_t| slots.join(src_t.slots()));
        let log_sparsity = srcs.iter().zip(lts).fold(usize::MAX, |s, (src_t, lt_t)| {
            let diagonal_t = lt_t.first_diagonal_plaintext().expect("checked non-empty");
            s.min(src_t.log_sparsity()).min(diagonal_t.lt_log_sparsity(src.log_n()))
        });
        dst.set_log_budget(res_log_budget);
        dst.set_log_delta(res_log_delta);
        dst.set_log_sparsity(log_sparsity);
        self.glwe_eval_linear_transformation_sum_into(cnv_offset, dst, babies, lts, keys, scratch);
        dst.set_slots(slots);
        Ok(())
    }
}

impl<BE: Backend> CKKSLinearTransformationReference<BE> for Module<BE> {}

/// `(log_budget, log_delta, cnv_offset)` of a transform of the diagonals `first` shares on `src` into `dst`, before any write.
fn lt_rule<BE, Dst, Src, P, H>(
    op: &'static str,
    dst: &Dst,
    src: &Src,
    babies: &LinearTransformationBabySteps<BE>,
    first: &P,
    keys: &H,
) -> Result<(usize, usize, usize)>
where
    BE: Backend,
    Dst: CKKSCtBounds,
    Src: CKKSCtBounds,
    P: LtDiagonalMeta + GLWEInfos,
    H: GetTensorKey<BE>,
{
    if first.rank().as_usize() == 0 {
        return mul_pt_params_raw(
            op,
            dst.k().as_usize(),
            src.log_delta(),
            src.log_budget(),
            first.lt_log_scale(),
            0,
            first.max_k().as_usize(),
        );
    }
    check_prepared_layout(op, dst, first)?;
    check_lazy_base2k(op, dst)?;
    // The product looks the tensor key up at the cache's width.
    ckks_ensure_operand!(
        op,
        babies.k() == src.k(),
        "the baby cache was prepared from a source of another width"
    );
    tensor_key_at(op, keys, src.k().max(first.k()))?;
    mul_ct_params_raw(
        op,
        dst.k().as_usize(),
        src.log_delta(),
        src.k().into(),
        first.lt_log_scale(),
        first.k().into(),
    )
}

/// The scale, width and slot kind a diagonal's evaluation reads: a ciphertext's width, a plaintext's storage width.
fn lt_diagonal_meta<P: LtDiagonalMeta + LWEInfos>(pt: &P) -> (usize, usize, SlotsKind) {
    let k = pt.lt_ciphertext_k().unwrap_or_else(|| pt.max_k());
    (pt.lt_log_scale(), k.as_usize(), pt.lt_slots())
}

fn invalid_sum(reason: String) -> CKKSError {
    CKKSCompositionError::InvalidOperand {
        op: "ckks_eval_linear_transformation_sum_into",
        reason,
    }
    .into()
}

/// `InvalidOperand` for `op` unless `babies` covers every baby rotation `lt` needs.
fn check_babies<BE: Backend, P>(
    op: &'static str,
    lt: &LinearTransformation<P>,
    babies: &LinearTransformationBabySteps<BE>,
) -> Result<()> {
    for rotation in lt.baby_steps().iter().copied() {
        ckks_ensure_operand!(
            op,
            babies.contains_baby_step(rotation),
            "missing prepared baby-step rotation {rotation}"
        );
    }
    Ok(())
}

/// Verifies that `babies` covers every baby rotation `lt` needs and that all
/// automorphism keys required by `lt` are present (keyed by Galois element).
fn check_required_keys<BE: Backend, P, H>(
    op: &'static str,
    lt: &LinearTransformation<P>,
    babies: &LinearTransformationBabySteps<BE>,
    keys: &H,
    cyclotomic_order: i64,
    giant_k: TorusPrecision,
) -> Result<()>
where
    H: GetAutomorphismKey<BE>,
{
    check_babies(op, lt, babies)?;
    let giants = lt.giant_steps.iter().filter(|gs| !gs.diagonals.is_empty()).map(|gs| gs.rot);
    let rotations: Vec<i64> = match lt.rotations {
        BsgsRotations::PerRotation => giants.filter(|&r| r != 0).collect(),
        rotations => rotations.giant_key_rotations(giants),
    };
    for rotation in rotations {
        let gal_el = galois_element(rotation, cyclotomic_order);
        keys.get_automorphism_key(gal_el, giant_k)
            .map_err(|source| CKKSCompositionError::MissingAutomorphismKey {
                op,
                rotation,
                k: giant_k.into(),
                source,
            })?;
    }
    Ok(())
}
