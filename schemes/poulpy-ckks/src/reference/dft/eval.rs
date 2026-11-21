//! Homomorphic (I)DFT generation and evaluation.
//!
//! Builds the prepared factor operands of a [`DFTMatrix`] from a
//! [`DFTPlan`] (encode each generated [`crate::layouts::ComplexDiagonals`] factor into a
//! CKKS linear transformation, then prepare its right operand), and evaluates the
//! transform by chaining one prepared linear transformation per factor.
//!
//! Scale accounting (see the homomorphic-DFT scale accounting in
//! [`docs/guides/bootstrapping.md`](https://github.com/poulpy-fhe/poulpy/blob/main/docs/guides/bootstrapping.md)). poulpy's torus plaintext-multiply
//! already realigns the result to the **input** `log_delta` via its `cnv_offset` —
//! i.e. the rescale is implicit in the linear-transform eval. So the homomorphic
//! (I)DFT is simply *one prepared linear transformation per factor, chained*, with
//! **no** explicit rescale between them. Each factor consumes `factor_meta.log_delta`
//! bits of `log_budget`; the whole transform consumes [`DFTMatrix::consumed_bits`],
//! which the input ciphertext must have available.
//!
//! `Standard` format goes through `coeffs_to_slots` / `slots_to_coeffs`;
//! `SplitRealAndImag` (real/imag returned in two ciphertexts) goes through
//! `coeffs_to_slots_split` / `slots_to_coeffs_split`; the sparse `RepackImagAsReal`
//! packing (imag packed into the right half) through `coeffs_to_slots_repack` /
//! `slots_to_coeffs_repack`.

use crate::{CKKSResult as Result, ckks_ensure, ckks_ensure_operand};
use poulpy_core::layouts::IntPolyInfos;
use poulpy_core::{
    GLWEBytesOf, GLWEDiagonalProd, GLWELinearTransformationBabyStepsBytesOf, GLWELinearTransformationOps, GLWETrace,
    ScratchArenaTakeGLWELinearTransformationBabySteps,
    layouts::{
        Base2K, GGLWEInfos, GLWEInfos, GLWELayout, GLWEToBackendMut, GLWEToBackendRef, GetAutomorphismKey, LWEInfos,
        LinearTransformation, LinearTransformationPlan,
    },
};
use poulpy_hal::{
    api::CnvPVecAlloc,
    layouts::{Backend, Module, ScratchArena, Standard},
};

use crate::reference::linear_transformation::PlaintextDiagonalKeys;
use crate::{
    CKKSCompositionError, CKKSCtBounds, SetCKKSInfos, SlotLayout,
    api::{
        CKKSAddOps, CKKSConjugateOps, CKKSCopyOps, CKKSEncodingHostOps, CKKSEncodingOps, CKKSEncodingScalar, CKKSImagOps,
        CKKSLinearTransformationOps, CKKSRotateOps, CKKSSubOps, LtDiagonalMeta,
    },
    layouts::{
        CKKSDFTFactors, CKKSPlaintextOwned, ComplexDiagonals, DFTMatrix, DFTMatrixFactors, DFTMatrixPrepared, DFTPlan, DFTType,
        DftDirection, DftFormat, DftScalar,
        dft_matrices::ckks_dft_matrices_gen,
        replicated::factor_coeffs_meta,
        validation::{
            check_replicated_chain, ensure_chain_input, ensure_prepared_transformation_layout, ensure_scratch,
            ensure_uniform_diagonals,
        },
    },
    oep::CKKSEncodingImpl,
};

/// One unprepared factor: the diagonals of a single DFT factor matrix encoded
/// as a CKKS linear transformation (plaintext diagonals).
type DftFactorLt<BE> = LinearTransformation<CKKSPlaintextOwned<BE>>;

fn checked_dft_log_slots(literal: &DFTPlan, log_n: usize) -> Result<usize> {
    // Shape validity (including a non-overflowing layer sum) is guaranteed by
    // `DFTPlan::new`; only the plan-vs-ring clause remains to check here.
    let log_slots = match literal.slot_layout() {
        SlotLayout::Full => literal.log_slots(),
        SlotLayout::Replicated(layout) => layout.log_k() + layout.log_r(),
    };
    if log_slots >= log_n {
        return Err(CKKSCompositionError::InvalidPlan {
            plan: "DFTPlan",
            reason: format!(
                "DFT has 2^{log_slots} slots, but a degree-2^{log_n} CKKS ring supports at most 2^{} slots",
                log_n.saturating_sub(1)
            ),
        }
        .into());
    }
    Ok(log_slots)
}

/// Resolves the plan against the ring degree, so the stored `format` is canonical and `plan.is_sparse()` is exact, and
/// generates the diagonals of each factor. The per-factor scale is carried by `literal.coeffs_meta`.
fn dft_plan_diagonals<F: DftScalar>(literal: &DFTPlan, log_n: usize) -> Result<(DFTPlan, Vec<ComplexDiagonals<F>>)> {
    literal.check_slot_layout()?;
    checked_dft_log_slots(literal, log_n)?;
    let plan = literal.resolved_at(log_n);
    let factors = ckks_dft_matrices_gen::<F>(&plan, log_n);
    Ok((plan, factors))
}

/// Encodes each factor's generated diagonals into an (unprepared) CKKS linear
/// transformation (the plaintext factors of an unprepared [`DFTMatrix`]);
/// [`ckks_dft_matrix_prepare_reference`] later prepares each factor into a `CnvPVec` of the
/// resident [`DFTMatrixPrepared`].
#[allow(clippy::too_many_arguments)]
fn dft_factor_lts<BE, F>(
    module: &Module<BE>,
    base2k: Base2K,
    literal: &DFTPlan,
    scratch: &mut ScratchArena<'_, BE>,
) -> Result<(DFTPlan, Vec<DftFactorLt<BE>>)>
where
    BE: Backend + CKKSEncodingImpl<F>,
    Module<BE>: CnvPVecAlloc<BE> + CKKSLinearTransformationOps<BE> + CKKSEncodingOps<BE, F>,
    F: DftScalar + CKKSEncodingScalar,
{
    let (plan, factors_cd) = dft_plan_diagonals::<F>(literal, module.log_n())?;

    // Each factor's BSGS width is taken from the plan's schedule; the library
    // applies no implicit optimum. `giant_step == 1` is the direct schedule.
    let last = factors_cd.len().saturating_sub(1);
    let lts = factors_cd
        .iter()
        .zip(plan.factor_encodings(module.log_n()))
        .enumerate()
        .map(|(i, (cd, (strategy, log_delta)))| {
            let coeffs_meta = match plan.slot_layout() {
                SlotLayout::Full => plan.coeffs_meta,
                // The last `Encode` factor is quantized with the shaping of the layout it writes.
                SlotLayout::Replicated(layout) => {
                    let shaped = plan.kind == DFTType::Encode && i == last;
                    factor_coeffs_meta(cd, log_delta, shaped.then_some(layout))
                }
            };
            crate::encoding::ckks_linear_transformation_encode_from_diagonals::<BE, F, Standard>(
                module,
                base2k,
                coeffs_meta,
                cd,
                strategy,
                false,
                scratch,
            )
        })
        .collect::<core::result::Result<Vec<_>, ::anyhow::Error>>()
        .map_err(crate::CKKSError::from)?;

    Ok((plan, lts))
}

/// Prepares an unprepared (host plaintext) [`DFTMatrix`] into `res`, its resident
/// convolution-domain form [`DFTMatrixPrepared`], with no allocation.
///
/// Each factor's plaintext diagonals are prepared into the matching `CnvPVec` right
/// operand of `res`, so repeated evaluation no longer materializes them per factor,
/// and `res` takes `dft`'s resolved plan. A `res` of another factor count, factor
/// schedule or diagonal degree, radix or width, or a factor whose diagonals do not
/// share one scale, width and slot kind, is `InvalidOperand`, and an arena below
/// [`ckks_dft_matrix_prepare_tmp_bytes_reference`] is `CKKSError::Internal`, both before any write.
/// Build the input via [`ckks_dft_matrix_new_reference`].
pub fn ckks_dft_matrix_prepare_reference<Dir, Fmt, BE, P>(
    module: &Module<BE>,
    res: &mut DFTMatrixPrepared<BE, Dir, Fmt>,
    dft: &DFTMatrix<BE, Dir, Fmt, LinearTransformation<P>>,
    scratch: &mut ScratchArena<'_, BE>,
) -> Result<()>
where
    BE: Backend,
    Module<BE>: CKKSLinearTransformationOps<BE>,
    P: GLWEToBackendRef<BE> + IntPolyInfos + CKKSCtBounds + GLWEDiagonalProd<BE>,
{
    const OP: &str = "ckks_dft_matrix_prepare";
    crate::ensure_operand_degrees(
        OP,
        module,
        dft.factor_operands()
            .iter()
            .filter_map(|lt| lt.first_diagonal_plaintext().map(|pt| pt.n())),
    )?;
    if res.factor_operands().len() != dft.factor_operands().len() {
        return Err(CKKSCompositionError::InvalidOperand {
            op: OP,
            reason: "the matrix's factor count differs from the prepared matrix's".into(),
        }
        .into());
    }
    for (prepared, lt) in res.factor_operands().iter().zip(dft.factor_operands()) {
        ensure_prepared_transformation_layout(OP, prepared, lt)?;
        ensure_uniform_diagonals(OP, lt, |pt| (pt.log_delta(), pt.encoded_k().as_usize(), pt.slots()))?;
    }
    ckks_ensure!(
        scratch.available() >= ckks_dft_matrix_prepare_tmp_bytes_reference(module, dft),
        "{OP}: scratch below its query"
    );
    for (prepared, lt) in res.inner.factors.iter_mut().zip(dft.factor_operands()) {
        module.ckks_linear_transformation_rhs_prepare(prepared, lt, scratch)?;
    }
    res.inner.plan = dft.plan().clone();
    Ok(())
}

/// Builds the backend-owned, unprepared homomorphic (I)DFT for direction `Dir` and
/// output format `Fmt`, described by `literal`.
///
/// The `Dir`/`Fmt` type parameters are authoritative: they set the transform
/// direction and the requested output format (`literal.kind` / `literal.format`
/// are ignored, like `factor_log_delta`). Each factor matrix is encoded into a
/// CKKS linear transformation whose diagonals are kept as plaintexts (the default
/// [`DFTMatrix`]); evaluating it materializes each diagonal on the fly per factor.
/// Prepare it into the resident form with [`ckks_dft_matrix_prepare_reference`].
///
/// Returns an error when the plan is invalid, a generated diagonal cannot be
/// encoded at the requested metadata, or `Fmt = Repack` is requested with
/// dense parameters (`log_slots ≥ log_n − 1`), in which case the repack
/// collapses to `Split` and the caller must build with `Split` instead. The
/// latter is the single runtime format resolution; every downstream eval is
/// type-checked.
#[allow(clippy::too_many_arguments)]
pub fn ckks_dft_matrix_new_reference<Dir, Fmt, BE, F>(
    module: &Module<BE>,
    base2k: Base2K,
    literal: &DFTPlan,
    scratch: &mut ScratchArena<'_, BE>,
) -> Result<DFTMatrix<BE, Dir, Fmt>>
where
    Dir: DftDirection,
    Fmt: DftFormat,
    BE: Backend + CKKSEncodingImpl<F>,
    Module<BE>: CnvPVecAlloc<BE> + CKKSLinearTransformationOps<BE> + CKKSEncodingOps<BE, F>,
    F: DftScalar + CKKSEncodingScalar,
{
    // The markers are the source of truth for direction and requested format.
    let mut literal = literal.clone();
    literal.kind = Dir::KIND;
    literal.format = Fmt::FORMAT;

    let (plan, lts) = dft_factor_lts::<BE, F>(module, base2k, &literal, scratch)?;

    // The dense-`RepackImagAsReal` → `Split` canonicalization is the only runtime
    // format resolution; if it changed the requested format, the request was
    // unrepresentable for these parameters.
    if plan.format != Fmt::FORMAT {
        return Err(CKKSCompositionError::DftMatrixMismatch {
            op: "ckks_dft_matrix_new",
            expected: "sparse Repack (RepackImagAsReal)",
            got: "dense parameters — build with the `Split` format instead".to_string(),
        }
        .into());
    }

    Ok(DFTMatrix::from_factors(DFTMatrixFactors::new(plan, lts)))
}

/// Scratch bytes of [`ckks_dft_matrix_new_reference`] for `literal`: the factors are encoded one after the other on the
/// same scratch, so the largest slot encoding of one factor's diagonals, at that factor's period.
///
/// # Errors
///
/// The plan-versus-ring error of [`ckks_dft_matrix_new_reference`] for a plan with more slots than the ring
/// supports.
pub fn ckks_dft_matrix_new_tmp_bytes_reference<Dir, Fmt, BE, F>(module: &Module<BE>, literal: &DFTPlan) -> Result<usize>
where
    Dir: DftDirection,
    Fmt: DftFormat,
    BE: Backend,
    Module<BE>: CKKSEncodingHostOps<BE, F>,
    F: DftScalar + CKKSEncodingScalar,
{
    let mut literal = literal.clone();
    literal.kind = Dir::KIND;
    literal.format = Fmt::FORMAT;
    let (_, factors) = dft_plan_diagonals::<F>(&literal, module.log_n())?;
    Ok(factors
        .iter()
        .map(|factor| crate::encoding::ckks_linear_transformation_encode_tmp_bytes::<BE, _, F, Standard>(module, factor.period()))
        .max()
        .unwrap_or(0))
}

impl<BE: Backend, Dir, Fmt: DftFormat, P> DFTMatrix<BE, Dir, Fmt, LinearTransformation<P>> {
    /// The distinct Galois elements whose automorphism keys evaluating this
    /// transform requires (the union over all factors, plus the `slots` repack
    /// rotation for the sparse path).
    pub fn galois_elements(&self, cyclotomic_order: i64) -> Vec<i64> {
        let mut set: std::collections::BTreeSet<i64> = std::collections::BTreeSet::new();
        for f in self.factor_operands() {
            set.extend(f.galois_elements(cyclotomic_order));
        }
        if self.is_sparse() {
            set.insert(poulpy_hal::layouts::galois_element(
                1i64 << self.plan().log_slots(),
                cyclotomic_order,
            ));
        }
        // Defensive: drop the identity element if present.
        set.remove(&poulpy_hal::layouts::galois_element(0, cyclotomic_order));
        set.into_iter().collect()
    }
}

/// Evaluates the homomorphic (I)DFT `dft` in place on `ct`.
///
/// Chains one linear transformation per factor — generic over the diagonal
/// representation `P` (resident `PreparedDiagonal` or streamed plaintext),
/// dispatched by the single unified
/// [`ckks_linear_transformation_eval_assign`](CKKSLinearTransformationOps::ckks_linear_transformation_eval_assign)
/// — preparing the baby rotations of the running ciphertext each time. No
/// explicit rescale is needed (the plaintext-multiply realigns to the input
/// `log_delta`, see the module docs). The input `ct.log_budget()` must be at
/// least `dft.consumed_bits()`.
///
/// ## Progressive compaction
///
/// Each factor consumes `factor_log_delta` bits of budget, so after a factor the
/// running ciphertext's `k` (and therefore the work of the next
/// factor's baby-step keyswitches, which scale with the operand limb count) is
/// smaller than its storage. After every factor `ct` is therefore compacted in
/// place.
///
/// ## Replicated layout
///
/// The chain of a plan on a replicated layout ([`DFTPlan::with_slot_layout`]) starts with its normalized trace, which
/// leaves the sparsity of its period. A layout past the slots of `ct`'s degree, less budget than the factors consume,
/// a key of the trace or of a factor missing at its width and the arena are checked before it writes `ct`. A `Decode`
/// on a replicated layout reads `ct` in that layout and leaves it on the full layout, every other chain reads the full
/// layout.
pub fn ckks_dft_matrix_eval_assign_reference<BE, Dir, Fmt, D, Dst, H>(
    module: &Module<BE>,
    ct: &mut Dst,
    dft: &D,
    keys: &H,
    scratch: &mut ScratchArena<'_, BE>,
) -> Result<()>
where
    BE: Backend,
    D: CKKSDFTFactors<BE, Dir, Fmt>,
    Module<BE>: CKKSLinearTransformationOps<BE> + GLWELinearTransformationOps<BE> + CKKSCopyOps<BE> + GLWETrace<BE>,
    Dst: GLWEToBackendMut<BE> + GLWEToBackendRef<BE> + CKKSCtBounds + SetCKKSInfos,
    H: GetAutomorphismKey<BE>,
{
    const OP: &str = "ckks_dft_matrix_eval_assign";
    ensure_chain_input(OP, dft.dft_plan(), crate::GLWEPlaintextInfos::slot_layout(ct))?;
    crate::ensure_operand_degrees(OP, module, [ct.n()])?;
    ckks_ensure_operand!(
        OP,
        dft.factor_plans().len() == dft.factors().count(),
        "takes one schedule per factor"
    );
    let log_n = ct.n().as_usize().ilog2() as usize;
    if let Some(levels) = dft.dft_plan().trace_levels(log_n) {
        let shape = (ct.n().as_usize(), ct.k().as_usize(), ct.log_budget());
        let bytes = check_replicated_chain(OP, dft, shape, keys, |key| {
            eval_tmp_bytes(module, std::slice::from_ref(&*ct), dft, key)
        })?;
        ensure_scratch(OP, scratch, bytes)?;
        let sparsity = log_n - levels.start;
        ct.set_slot_layout(SlotLayout::Full);
        module.glwe_trace_assign(ct, levels, false, true, keys, scratch);
        ct.set_log_sparsity(sparsity);
    }
    // One factor at a time, in place on `ct`; compact `ct` after each so the next
    // factor's baby-step keyswitches operate on fewer limbs as the budget shrinks.
    for (factor, plan) in dft.factors().zip(dft.factor_plans()) {
        eval_factor(module, ct, factor, plan, keys, scratch)?;
    }
    ct.set_noise(None);
    Ok(())
}

/// Evaluates a single homomorphic-DFT factor in place on `running`: carves from
/// scratch and prepares the baby rotations of the current operand, then applies the
/// unified linear-transformation eval. `P` only decides how the factor's RHS is
/// materialized inside the eval (resident vs streamed).
fn eval_factor<BE, P, Dst, H>(
    module: &Module<BE>,
    running: &mut Dst,
    factor: &LinearTransformation<P>,
    plan: &LinearTransformationPlan,
    keys: &H,
    scratch: &mut ScratchArena<'_, BE>,
) -> Result<()>
where
    BE: Backend,
    P: GLWEDiagonalProd<BE> + LtDiagonalMeta,
    Module<BE>: CKKSLinearTransformationOps<BE>,
    Dst: GLWEToBackendMut<BE> + GLWEToBackendRef<BE> + CKKSCtBounds + SetCKKSInfos,
    H: GetAutomorphismKey<BE>,
{
    ckks_ensure!(
        scratch.available() >= module.glwe_linear_transformation_baby_steps_bytes_of(plan, &*running),
        "ckks_dft_matrix_eval_assign: scratch below its baby-step cache"
    );
    let (mut babies, mut scratch) = scratch
        .borrow()
        .take_glwe_linear_transformation_baby_steps_scratch(module, plan, &*running);
    module.ckks_linear_transformation_baby_steps_prepare(&mut babies, running, keys, &mut scratch)?;
    module.ckks_linear_transformation_eval_assign(running, &babies, factor, &PlaintextDiagonalKeys(keys), &mut scratch)?;
    Ok(())
}

/// Workspace for preparing the largest factor through the selected Core operation.
pub fn ckks_dft_matrix_prepare_tmp_bytes_reference<BE, Dir, Fmt, P>(
    module: &Module<BE>,
    dft: &DFTMatrix<BE, Dir, Fmt, LinearTransformation<P>>,
) -> usize
where
    BE: Backend,
    P: GLWEInfos,
    Module<BE>: CKKSLinearTransformationOps<BE>,
{
    dft.factor_operands()
        .iter()
        .flat_map(|factor| &factor.giant_steps)
        .flat_map(|giant| &giant.diagonals)
        .map(|diagonal| module.ckks_linear_transformation_rhs_prepare_tmp_bytes(&diagonal.plaintext))
        .max()
        .unwrap_or(0)
}

/// Workspace for the reference DFT chain and all format compositions.
/// Covers both the prepared and the streamed evaluation of every factor.
///
/// # Panics
///
/// When `dft` holds another number of schedules than of factors, which the evaluation refuses.
pub fn ckks_dft_tmp_bytes_reference<BE, Dir, Fmt, D, Dst, Src, K>(
    module: &Module<BE>,
    dst: &Dst,
    src: &Src,
    dft: &D,
    key: &K,
) -> usize
where
    BE: Backend,
    D: CKKSDFTFactors<BE, Dir, Fmt>,
    Dst: CKKSCtBounds,
    Src: CKKSCtBounds,
    K: GGLWEInfos,
    Module<BE>: GLWELinearTransformationOps<BE>
        + GLWEBytesOf<BE>
        + GLWELinearTransformationBabyStepsBytesOf<BE>
        + CKKSCopyOps<BE>
        + CKKSConjugateOps<BE>
        + CKKSRotateOps<BE>
        + CKKSAddOps<BE>
        + CKKSSubOps<BE>
        + CKKSImagOps<BE>
        + GLWETrace<BE>,
{
    let envelope = |ct: &dyn CKKSCtBounds| crate::CKKSLayout {
        glwe_layout: GLWELayout {
            n: ct.n(),
            base2k: ct.base2k(),
            k: ct.max_k(),
            rank: ct.rank(),
        },
        meta: ct.meta(),
    };
    assert!(
        dft.factor_plans().len() == dft.factors().count(),
        "ckks_dft_tmp_bytes: one schedule per factor"
    );
    let layouts = [envelope(dst), envelope(src)];
    let mut bytes = module.ckks_copy_tmp_bytes(dst, src);
    for ct in &layouts {
        bytes = bytes
            .max(module.ckks_copy_tmp_bytes(ct, ct))
            .max(module.ckks_conjugate_tmp_bytes(ct, key))
            .max(module.ckks_rotate_tmp_bytes(ct, key))
            .max(module.ckks_add_tmp_bytes(ct.max_size()))
            .max(module.ckks_sub_tmp_bytes(ct.max_size()))
            .max(module.ckks_mul_i_tmp_bytes(ct.max_size()))
            .max(module.ckks_div_i_tmp_bytes(ct.max_size()));
        // The split and repack formats carve at most two temporaries shaped like their destination.
        let temporaries = 2 * BE::scratch_aligned(module.glwe_bytes_of_from_infos(ct));
        bytes = bytes.max(
            temporaries
                + module
                    .ckks_conjugate_tmp_bytes(ct, key)
                    .max(module.ckks_rotate_tmp_bytes(ct, key))
                    .max(module.ckks_add_tmp_bytes(ct.max_size()))
                    .max(module.ckks_sub_tmp_bytes(ct.max_size()))
                    .max(module.ckks_div_i_tmp_bytes(ct.max_size())),
        );
    }
    bytes.max(eval_tmp_bytes(module, &layouts, dft, key))
}

/// Scratch of the evaluation of `dft` in place on a ciphertext of each of `layouts`: the trace of a replicated layout,
/// and each factor's baby cache beside its preparation, and beside a working ciphertext and its evaluation.
fn eval_tmp_bytes<BE, Dir, Fmt, D>(module: &Module<BE>, layouts: &[impl CKKSCtBounds], dft: &D, key: &impl GGLWEInfos) -> usize
where
    BE: Backend,
    D: CKKSDFTFactors<BE, Dir, Fmt>,
    Module<BE>: GLWELinearTransformationOps<BE> + CKKSCopyOps<BE> + GLWETrace<BE>,
{
    let mut bytes = 0;
    for ct in layouts {
        if dft.dft_plan().trace_levels(ct.n().as_usize().ilog2() as usize).is_some() {
            bytes = bytes.max(module.glwe_trace_assign_tmp_bytes(ct, key));
        }
        // Each factor's own schedule, so that a factor on power-of-two keys counts its
        // doubling and its giant-step tree, beside its baby cache.
        for (factor, plan) in dft.factors().zip(dft.factor_plans()) {
            let babies = module.glwe_linear_transformation_baby_steps_bytes_of(plan, ct);
            bytes = bytes.max(babies + module.glwe_linear_transformation_baby_steps_prepare_tmp_bytes(ct, plan, key));
            for pt in factor
                .giant_steps
                .iter()
                .flat_map(|giant| &giant.diagonals)
                .map(|diagonal| &diagonal.plaintext)
            {
                // Prepared and streamed evaluation are selected independently, so
                // neither query is assumed to cover the other.
                let factor_work = module
                    .glwe_linear_transformation_eval_unprepared_rhs_tmp_bytes(ct, ct, pt, plan, key, key)
                    .max(module.glwe_linear_transformation_eval_tmp_bytes(ct, ct, pt, plan, key, key))
                    .max(module.ckks_copy_tmp_bytes(ct, ct));
                bytes = bytes.max(babies + BE::scratch_aligned(module.glwe_bytes_of_from_infos(ct)) + factor_work);
            }
        }
    }
    bytes
}

/// Reference bodies of the required [`CKKSDFTImpl`](crate::oep::CKKSDFTImpl) methods, which
/// [`impl_ckks_dft_reference_full!`](crate::impl_ckks_dft_reference_full) forwards to.
pub trait CKKSDFTReference<BE: Backend> {
    fn ckks_dft_matrix_prepare_tmp_bytes_reference<Dir, Fmt, P>(
        &self,
        dft: &DFTMatrix<BE, Dir, Fmt, LinearTransformation<P>>,
    ) -> usize
    where
        P: GLWEInfos,
        Module<BE>: CKKSLinearTransformationOps<BE>;

    fn ckks_dft_tmp_bytes_reference<Dir, Fmt, D, Dst, Src, K>(&self, dst: &Dst, src: &Src, dft: &D, key: &K) -> usize
    where
        D: CKKSDFTFactors<BE, Dir, Fmt>,
        Dst: CKKSCtBounds,
        Src: CKKSCtBounds,
        K: GGLWEInfos,
        Module<BE>: GLWELinearTransformationOps<BE>
            + GLWEBytesOf<BE>
            + GLWELinearTransformationBabyStepsBytesOf<BE>
            + CKKSCopyOps<BE>
            + CKKSConjugateOps<BE>
            + CKKSRotateOps<BE>
            + CKKSAddOps<BE>
            + CKKSSubOps<BE>
            + CKKSImagOps<BE>
            + GLWETrace<BE>;

    fn ckks_dft_matrix_prepare_reference<Dir, Fmt, P>(
        &self,
        res: &mut DFTMatrixPrepared<BE, Dir, Fmt>,
        dft: &DFTMatrix<BE, Dir, Fmt, LinearTransformation<P>>,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        Module<BE>: CKKSLinearTransformationOps<BE>,
        P: GLWEToBackendRef<BE> + IntPolyInfos + CKKSCtBounds + GLWEDiagonalProd<BE>;

    fn ckks_dft_matrix_eval_assign_reference<Dir, Fmt, D, Dst, H>(
        &self,
        ct: &mut Dst,
        dft: &D,
        keys: &H,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        D: CKKSDFTFactors<BE, Dir, Fmt>,
        Module<BE>: CKKSLinearTransformationOps<BE> + GLWELinearTransformationOps<BE> + CKKSCopyOps<BE> + GLWETrace<BE>,
        Dst: GLWEToBackendMut<BE> + GLWEToBackendRef<BE> + CKKSCtBounds + SetCKKSInfos,
        H: GetAutomorphismKey<BE>;
}

impl<BE: Backend> CKKSDFTReference<BE> for Module<BE> {
    fn ckks_dft_matrix_prepare_tmp_bytes_reference<Dir, Fmt, P>(
        &self,
        dft: &DFTMatrix<BE, Dir, Fmt, LinearTransformation<P>>,
    ) -> usize
    where
        P: GLWEInfos,
        Module<BE>: CKKSLinearTransformationOps<BE>,
    {
        ckks_dft_matrix_prepare_tmp_bytes_reference(self, dft)
    }

    fn ckks_dft_tmp_bytes_reference<Dir, Fmt, D, Dst, Src, K>(&self, dst: &Dst, src: &Src, dft: &D, key: &K) -> usize
    where
        D: CKKSDFTFactors<BE, Dir, Fmt>,
        Dst: CKKSCtBounds,
        Src: CKKSCtBounds,
        K: GGLWEInfos,
        Module<BE>: GLWELinearTransformationOps<BE>
            + GLWEBytesOf<BE>
            + GLWELinearTransformationBabyStepsBytesOf<BE>
            + CKKSCopyOps<BE>
            + CKKSConjugateOps<BE>
            + CKKSRotateOps<BE>
            + CKKSAddOps<BE>
            + CKKSSubOps<BE>
            + CKKSImagOps<BE>
            + GLWETrace<BE>,
    {
        ckks_dft_tmp_bytes_reference(self, dst, src, dft, key)
    }

    fn ckks_dft_matrix_prepare_reference<Dir, Fmt, P>(
        &self,
        res: &mut DFTMatrixPrepared<BE, Dir, Fmt>,
        dft: &DFTMatrix<BE, Dir, Fmt, LinearTransformation<P>>,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        Module<BE>: CKKSLinearTransformationOps<BE>,
        P: GLWEToBackendRef<BE> + IntPolyInfos + CKKSCtBounds + GLWEDiagonalProd<BE>,
    {
        ckks_dft_matrix_prepare_reference(self, res, dft, scratch)
    }

    fn ckks_dft_matrix_eval_assign_reference<Dir, Fmt, D, Dst, H>(
        &self,
        ct: &mut Dst,
        dft: &D,
        keys: &H,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        D: CKKSDFTFactors<BE, Dir, Fmt>,
        Module<BE>: CKKSLinearTransformationOps<BE> + GLWELinearTransformationOps<BE> + CKKSCopyOps<BE> + GLWETrace<BE>,
        Dst: GLWEToBackendMut<BE> + GLWEToBackendRef<BE> + CKKSCtBounds + SetCKKSInfos,
        H: GetAutomorphismKey<BE>,
    {
        ckks_dft_matrix_eval_assign_reference(self, ct, dft, keys, scratch)
    }
}

/// Reference bodies of the required [`CKKSDFTMatrixImpl<F>`](crate::oep::CKKSDFTMatrixImpl) methods, which
/// [`impl_ckks_dft_matrix_reference_full!`](crate::impl_ckks_dft_matrix_reference_full) forwards to.
pub trait CKKSDFTMatrixReference<BE: Backend, F: DftScalar + CKKSEncodingScalar> {
    fn ckks_dft_matrix_new_tmp_bytes_reference<Dir, Fmt>(&self, literal: &DFTPlan) -> Result<usize>
    where
        Dir: DftDirection,
        Fmt: DftFormat,
        Module<BE>: CKKSEncodingHostOps<BE, F>;

    fn ckks_dft_matrix_new_reference<Dir, Fmt>(
        &self,
        base2k: Base2K,
        literal: &DFTPlan,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<DFTMatrix<BE, Dir, Fmt>>
    where
        Dir: DftDirection,
        Fmt: DftFormat,
        BE: CKKSEncodingImpl<F>,
        Module<BE>: CnvPVecAlloc<BE> + CKKSLinearTransformationOps<BE> + CKKSEncodingOps<BE, F>;
}

impl<BE: Backend, F: DftScalar + CKKSEncodingScalar> CKKSDFTMatrixReference<BE, F> for Module<BE> {
    fn ckks_dft_matrix_new_tmp_bytes_reference<Dir, Fmt>(&self, literal: &DFTPlan) -> Result<usize>
    where
        Dir: DftDirection,
        Fmt: DftFormat,
        Module<BE>: CKKSEncodingHostOps<BE, F>,
    {
        ckks_dft_matrix_new_tmp_bytes_reference::<Dir, Fmt, BE, F>(self, literal)
    }

    fn ckks_dft_matrix_new_reference<Dir, Fmt>(
        &self,
        base2k: Base2K,
        literal: &DFTPlan,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<DFTMatrix<BE, Dir, Fmt>>
    where
        Dir: DftDirection,
        Fmt: DftFormat,
        BE: CKKSEncodingImpl<F>,
        Module<BE>: CnvPVecAlloc<BE> + CKKSLinearTransformationOps<BE> + CKKSEncodingOps<BE, F>,
    {
        ckks_dft_matrix_new_reference::<Dir, Fmt, BE, F>(self, base2k, literal, scratch)
    }
}

#[cfg(test)]
mod validation_tests {
    use crate::layouts::{DFTOutputFormat, DFTType};
    use crate::shaping::ShapingUnit;

    use super::*;

    fn plan(factorization_depth: Vec<usize>) -> anyhow::Result<DFTPlan> {
        DFTPlan::new(
            DFTType::Encode,
            factorization_depth.into_iter().map(|d| (d, 1)).collect::<Vec<_>>(),
            DFTOutputFormat::Standard,
            crate::CoeffsMeta::from_delta_budget(0, 0),
        )
    }

    #[test]
    fn malformed_dft_dimensions_are_rejected() {
        // An overflowing layer sum never yields a plan: `DFTPlan::new` rejects it.
        assert!(plan(vec![usize::MAX, 1]).is_err());

        // A valid plan can still be too wide for a given ring.
        let error =
            checked_dft_log_slots(&plan(vec![4]).unwrap(), 4).expect_err("a DFT cannot expose more slots than its CKKS ring");
        assert_eq!(
            error.composition(),
            Some(&CKKSCompositionError::InvalidPlan {
                plan: "DFTPlan",
                reason: "DFT has 2^4 slots, but a degree-2^4 CKKS ring supports at most 2^3 slots".into(),
            }),
            "unexpected error: {error:#}"
        );
    }

    /// The compilation refuses each setting a replicated layout cannot run, and its own settings on the full layout.
    #[test]
    fn replicated_settings_are_refused() {
        let dft = |kind, format, depths: &[usize]| {
            let schedule: Vec<_> = depths.iter().map(|&depth| (depth, 1)).collect();
            DFTPlan::new(kind, schedule, format, crate::CoeffsMeta::from_delta_budget(30, 0)).unwrap()
        };
        let (split, repack) = (DFTOutputFormat::SplitRealAndImag, DFTOutputFormat::RepackImagAsReal);
        let on = |layout, plan: DFTPlan| plan.with_slot_layout(SlotLayout::Replicated(layout));
        let (eight, four) = (ShapingUnit::U16_40.layout(3).unwrap(), ShapingUnit::U16_40.layout(2).unwrap());
        let decode = || on(eight, dft(DFTType::Decode, split, &[3])).with_placement_factors(1);
        let full = "a replicated layout takes no repack, no first factor on the raise, no bit reversal and a key per rotation";
        let replicated = "placement factors and a last-factor scale take a replicated layout";
        let points = "a transform on 2^2 points takes a replicated layout of as many values, or Encode of twice as many, got 2^3";
        let merged = "a placement merged into the transform takes a schedule of one factor";
        let (factors, period) = (
            "the placement takes at most log2(k) = 3 factors",
            "a merged placement reads a period of at most the 2^2 points",
        );
        let overflow = "the scales of the factors overflow the consumed-bit count";
        let ring = "DFT has 2^6 slots, but a degree-2^6 CKKS ring supports at most 2^5 slots";
        let plain = || dft(DFTType::Encode, split, &[3]);
        for (plan, log_n, reason) in [
            (plain().with_placement_factors(1), 12, replicated),
            (plain().with_last_factor_log_delta(30), 12, replicated),
            (on(eight, dft(DFTType::Decode, split, &[2])), 12, points),
            (on(eight, dft(DFTType::Decode, repack, &[3])), 12, full),
            (decode().with_first_factor_on_raise(), 12, full),
            (decode().with_bit_reversed(true), 12, full),
            (decode().with_power_of_two_keys(true), 12, full),
            (decode().with_placement_factors(4), 12, factors),
            (on(eight, dft(DFTType::Encode, split, &[1, 2])), 12, merged),
            (on(four, dft(DFTType::Decode, split, &[2])), 12, period),
            (decode().with_last_factor_log_delta(usize::MAX), 12, overflow),
            (decode(), 6, ring),
        ] {
            let error = dft_plan_diagonals::<f64>(&plan, log_n).expect_err("a refused plan");
            let want = CKKSCompositionError::InvalidPlan {
                plan: "DFTPlan",
                reason: reason.into(),
            };
            assert_eq!(error.composition(), Some(&want), "unexpected error: {error:#}");
        }
    }
}
