//! The compiled form of a plan: the resident matrices and the encoded EvalMod, reused across bootstraps.

use std::sync::OnceLock;

use anyhow::Result;
use poulpy_core::{
    GLWEDiagonalProd,
    layouts::{Base2K, GLWEToBackendRef, LinearTransformation, LinearTransformationPlan},
};
use poulpy_hal::layouts::{Backend, Module, ScratchArena};

use crate::{
    CKKSCtBounds,
    api::{
        CKKSDFTMatrixOps, CKKSDFTMatrixPreparedFactory, CKKSDFTOps, CKKSEncodingHostOps, CKKSEncodingOps, CKKSEncodingScalar,
        CKKSRightOperandPreparedFactory,
    },
    layouts::{
        CKKSCITwin, CKKSDFTFactors, CKKSPlaintextOwned, CKKSRightOperandPrepared, DFTMatrix, DFTMatrixPrepared, DFTPlan, Decode,
        Encode, EvalMod, EvalModPlan, ModRaisePrepared, Split, Standard, eval_mod::ckks_eval_mod_compile,
        mod_raise::first_factor_prepare,
    },
};

use super::{
    BootstrappingPipeline, BootstrappingPlan, EvalModCIPrepared, EvalModRing,
    eval_mod_ci::{CIPre, EvalModCI},
    key_widths::{RotationKeyUse, StageWidths},
};

/// Compiled [`BootstrappingPlan`]: the resident DFT matrices and encoded EvalMod
/// the bootstrapping pipeline evaluates.
///
/// Built once by [`Self::compile`] and reused across bootstraps. The
/// `SplitRealAndImag` format is used for both transforms so the real and
/// imaginary coefficient halves can be reduced independently by EvalMod.
pub struct BootstrappingContextPrepared<BE: Backend, F> {
    /// The plan the context was compiled from, at the degree it was compiled for: the recipe
    /// of its keys.
    pub(crate) plan: BootstrappingPlan,
    c2s_guard_bits: usize,
    functional_message_modulus: Option<usize>,
    /// Prepared CoeffsToSlots matrix (homomorphic encoding).
    ///
    /// The plan constructor scales this transform for its EvalMod variant.
    coeffs_to_slots: DFTMatrixPrepared<BE, Encode, Split>,

    /// The modulus raise and, when the plan leaves it there ([`DFTPlan::with_first_factor_on_raise`]), the first
    /// CoeffsToSlots factor: one giant step per diagonal, the diagonal rotated back by its own index. Each diagonal is
    /// prepared once for the products on a ciphertext of a few limbs.
    pub(crate) mod_raise: ModRaisePrepared<BE>,

    /// Prepared bypass CoeffsToSlots matrix
    coeffs_to_slots_bypass: Option<DFTMatrixPrepared<BE, Encode, Split>>,

    /// Prepared SlotsToCoeffs matrix (homomorphic decoding). S2C-first plans
    /// cancel the initial real/imaginary split and normalize functional messages.
    slots_to_coeffs: DFTMatrixPrepared<BE, Decode, Split>,

    /// EvalMod (`x mod 1`): its plan, and its encoding when it runs on the
    /// standard ring.
    eval_mod_plan: EvalModPlan,
    eval_mod: Option<EvalMod<F, CKKSPlaintextOwned<BE>>>,

    /// The ring EvalMod runs on, and its plaintexts on the conjugate-invariant ring.
    eval_mod_ring: EvalModRing,
    eval_mod_ci: Option<EvalModCIPrepared<BE, F>>,

    /// Selected ModUp/EvalMod pipeline.
    pipeline: BootstrappingPipeline,

    /// Ephemeral sparse-secret weight required by the recipe, if enabled.
    sparse_secret_hamming_weight: Option<usize>,

    /// The degree of the first bootstrap and the uses of the rotation keys there, read once for every later entry check.
    rotation_key_uses: OnceLock<(usize, Vec<RotationKeyUse>)>,
}

// A context holds `Split` matrices, evaluated unsplit: see `ckks_coeffs_to_slots_assign` for the scale.
impl<BE: Backend, F> CKKSDFTFactors<BE, Encode, Standard> for BootstrappingContextPrepared<BE, F> {
    type Diagonal = <DFTMatrixPrepared<BE, Encode, Split> as CKKSDFTFactors<BE, Encode, Split>>::Diagonal;

    fn dft_plan(&self) -> &DFTPlan {
        self.coeffs_to_slots.dft_plan()
    }

    fn factors<'a>(&'a self) -> impl Iterator<Item = &'a LinearTransformation<Self::Diagonal>>
    where
        BE: 'a,
        Encode: 'a,
        Standard: 'a,
    {
        self.coeffs_to_slots.factors()
    }

    fn factor_plans(&self) -> &[LinearTransformationPlan] {
        self.coeffs_to_slots.factor_plans()
    }
}

impl<BE: Backend, F> BootstrappingContextPrepared<BE, F> {
    /// The uses of the rotation keys at degree `2^log_n`, `None` at another degree than the first asked for.
    pub(crate) fn rotation_key_uses(&self, log_n: usize) -> Option<&[RotationKeyUse]> {
        let (degree, uses) = self
            .rotation_key_uses
            .get_or_init(|| (log_n, self.stage_widths().rotation_key_uses(log_n)));
        (*degree == log_n).then_some(uses.as_slice())
    }

    /// The stages that set the widths at which a bootstrap uses its keys.
    pub(crate) fn stage_widths(&self) -> StageWidths<'_> {
        StageWidths {
            pipeline: self.pipeline,
            c2s_guard_bits: self.c2s_guard_bits,
            eval_mod_ring: self.eval_mod_ring,
            coeffs_to_slots: self.coeffs_to_slots.plan(),
            coeffs_to_slots_bypass: self.coeffs_to_slots_bypass.as_ref().map(|bypass| bypass.plan()),
            slots_to_coeffs: self.slots_to_coeffs.plan(),
            eval_mod: &self.eval_mod_plan,
        }
    }

    pub(crate) fn functional_message_modulus(&self) -> Option<usize> {
        self.functional_message_modulus
    }

    /// Extra scale bits used during CoeffsToSlots.
    pub fn c2s_guard_bits(&self) -> usize {
        self.c2s_guard_bits
    }

    /// Selected ModUp/EvalMod pipeline.
    pub fn pipeline(&self) -> BootstrappingPipeline {
        self.pipeline
    }

    /// Prepared CoeffsToSlots matrix.
    pub fn coeffs_to_slots(&self) -> &DFTMatrixPrepared<BE, Encode, Split> {
        &self.coeffs_to_slots
    }

    /// The modulus raise of this context, the step's operand with the first factor it may carry.
    pub fn mod_raise(&self) -> &ModRaisePrepared<BE> {
        &self.mod_raise
    }

    /// First CoeffsToSlots factor, when the modulus raise applies it
    /// ([`DFTPlan::with_first_factor_on_raise`]); [`Self::coeffs_to_slots`]
    /// then holds the remaining factors.
    pub fn coeffs_to_slots_first_factor(&self) -> Option<&LinearTransformation<CKKSRightOperandPrepared<BE>>> {
        self.mod_raise.first_factor.as_ref()
    }

    /// Prepared EvalRound+ bypass matrix, when enabled.
    pub fn coeffs_to_slots_bypass(&self) -> Option<&DFTMatrixPrepared<BE, Encode, Split>> {
        self.coeffs_to_slots_bypass.as_ref()
    }

    /// Prepared SlotsToCoeffs matrix.
    pub fn slots_to_coeffs(&self) -> &DFTMatrixPrepared<BE, Decode, Split> {
        &self.slots_to_coeffs
    }

    /// Encoded, backend-resident EvalMod circuit, when EvalMod runs on the
    /// standard ring; [`Self::eval_mod_ci`] holds it on the conjugate-invariant ring.
    pub fn eval_mod(&self) -> Option<&EvalMod<F, CKKSPlaintextOwned<BE>>> {
        self.eval_mod.as_ref()
    }

    /// The EvalMod plan, on either ring.
    pub fn eval_mod_plan(&self) -> &EvalModPlan {
        &self.eval_mod_plan
    }

    /// The ring EvalMod runs on.
    pub fn eval_mod_ring(&self) -> EvalModRing {
        self.eval_mod_ring
    }

    /// EvalMod on the conjugate-invariant ring and its conversions, for a plan on that ring.
    pub fn eval_mod_ci(&self) -> Option<&EvalModCIPrepared<BE, F>> {
        self.eval_mod_ci.as_ref()
    }

    /// Ephemeral sparse-secret weight required by the compiled recipe.
    pub fn sparse_secret_hamming_weight(&self) -> Option<usize> {
        self.sparse_secret_hamming_weight
    }
}

/// A bootstrapping context before preparation, the unprepared stage of a [`BootstrappingContextPrepared`]: the
/// generated DFT matrices and the encoded EvalMod, before the matrices are prepared.
///
/// Built by [`BootstrappingContextPrepared::compile_unprepared`]. Preparation is a
/// separate step so that its scratch can be sized with
/// [`Self::prepare_tmp_bytes`], which follows the backend's selected
/// preparation budget.
pub struct BootstrappingContext<BE: Backend, F> {
    plan: BootstrappingPlan,
    c2s_guard_bits: usize,
    functional_message_modulus: Option<usize>,
    coeffs_to_slots: DFTMatrix<BE, Encode, Split>,
    coeffs_to_slots_first_factor: Option<LinearTransformation<CKKSPlaintextOwned<BE>>>,
    coeffs_to_slots_bypass: Option<DFTMatrix<BE, Encode, Split>>,
    slots_to_coeffs: DFTMatrix<BE, Decode, Split>,
    eval_mod_plan: EvalModPlan,
    eval_mod: Option<EvalMod<F, CKKSPlaintextOwned<BE>>>,
    eval_mod_ring: EvalModRing,
    eval_mod_ci: Option<EvalModCI<BE, F>>,
    pipeline: BootstrappingPipeline,
    sparse_secret_hamming_weight: Option<usize>,
}

/// The former name of [`BootstrappingContext`].
#[deprecated(note = "renamed BootstrappingContext; the prepared context is BootstrappingContextPrepared")]
pub type BootstrappingContextUnprepared<BE, F> = BootstrappingContext<BE, F>;

impl<BE: Backend, F> BootstrappingContext<BE, F> {
    /// Scratch required by [`Self::prepare`]: the largest selected preparation
    /// budget among the matrices and the first factor on the raise.
    pub fn prepare_tmp_bytes(&self, module: &Module<BE>) -> usize
    where
        Module<BE>: CKKSDFTOps<BE> + CKKSDFTMatrixPreparedFactory<BE> + CKKSRightOperandPreparedFactory<BE>,
        CKKSPlaintextOwned<BE>: CKKSCtBounds,
    {
        let mut bytes = module
            .ckks_dft_matrix_prepare_tmp_bytes(&self.coeffs_to_slots)
            .max(module.ckks_dft_matrix_prepare_tmp_bytes(&self.slots_to_coeffs));
        if let Some(bypass) = &self.coeffs_to_slots_bypass {
            bytes = bytes.max(module.ckks_dft_matrix_prepare_tmp_bytes(bypass));
        }
        if let Some(CIPre::Folded(folded)) = self.eval_mod_ci.as_ref().map(|ci| &ci.pre) {
            bytes = bytes.max(module.ckks_dft_matrix_prepare_tmp_bytes(folded));
        }
        if let Some(diagonal) = self
            .coeffs_to_slots_first_factor
            .as_ref()
            .and_then(|f| f.first_diagonal_plaintext())
        {
            bytes = bytes.max(module.ckks_right_operand_prepare_tmp_bytes(diagonal));
        }
        bytes
    }

    /// Prepares the matrices and the diagonals of a first factor on the raise into the resident form the pipeline
    /// evaluates. Each unprepared operand is released as soon as it has been prepared.
    ///
    /// # Errors
    ///
    /// As [`CKKSDFTMatrixPreparedFactory::ckks_dft_matrix_prepare`].
    pub fn prepare(self, module: &Module<BE>, scratch: &mut ScratchArena<'_, BE>) -> Result<BootstrappingContextPrepared<BE, F>>
    where
        Module<BE>: CKKSDFTOps<BE> + CKKSDFTMatrixPreparedFactory<BE> + CKKSRightOperandPreparedFactory<BE>,
        CKKSPlaintextOwned<BE>: GLWEToBackendRef<BE> + CKKSCtBounds + GLWEDiagonalProd<BE>,
    {
        let Self {
            plan,
            c2s_guard_bits,
            functional_message_modulus,
            coeffs_to_slots,
            coeffs_to_slots_first_factor,
            coeffs_to_slots_bypass,
            slots_to_coeffs,
            eval_mod_plan,
            eval_mod,
            eval_mod_ring,
            eval_mod_ci,
            pipeline,
            sparse_secret_hamming_weight,
        } = self;
        let coeffs_to_slots = {
            let unprepared = coeffs_to_slots;
            module.ckks_dft_matrix_prepare_alloc(&unprepared, scratch)?
        };
        let slots_to_coeffs = {
            let unprepared = slots_to_coeffs;
            module.ckks_dft_matrix_prepare_alloc(&unprepared, scratch)?
        };
        let coeffs_to_slots_bypass = coeffs_to_slots_bypass
            .map(|unprepared| module.ckks_dft_matrix_prepare_alloc(&unprepared, scratch))
            .transpose()?;
        let eval_mod_ci = eval_mod_ci.map(|ci| ci.prepare(module, scratch)).transpose()?;
        let coeffs_to_slots_first_factor = coeffs_to_slots_first_factor
            .map(|factor| first_factor_prepare(module, factor, scratch))
            .transpose()?;
        let mod_raise = ModRaisePrepared {
            plan: plan.mod_raise(),
            first_factor: coeffs_to_slots_first_factor,
        };
        Ok(BootstrappingContextPrepared {
            plan,
            c2s_guard_bits,
            functional_message_modulus,
            coeffs_to_slots,
            mod_raise,
            coeffs_to_slots_bypass,
            slots_to_coeffs,
            eval_mod_plan,
            eval_mod,
            eval_mod_ring,
            eval_mod_ci,
            pipeline,
            sparse_secret_hamming_weight,
            rotation_key_uses: OnceLock::new(),
        })
    }
}

impl<BE: CKKSCITwin, F> BootstrappingContextPrepared<BE, F>
where
    F: CKKSEncodingScalar,
{
    /// Compiles `plan` directly into backend-resident matrices and EvalMod.
    ///
    /// Each stage carries its own coefficient metadata (`DFTPlan::meta` /
    /// `EvalModPlan::meta`), including the CoeffsToSlots scaling resolved by
    /// [`BootstrappingPlan::new`].
    ///
    /// With EvalMod on the conjugate-invariant ring, `ci` is `Some`, of degree `N`: EvalMod
    /// is encoded on it, and `u` and `v` at degree `2N`, which `module` serves on top of the
    /// bootstrap's degree `N`. The DFT matrices are generated as at degree `N`. `ci` is
    /// `None` with EvalMod on the standard ring.
    ///
    /// `scratch` serves both generation and preparation, so it must cover
    /// [`Self::compile_unprepared_tmp_bytes`] and
    /// [`BootstrappingContext::prepare_tmp_bytes`]. The latter is only
    /// known once the matrices exist: use [`Self::compile_unprepared`] followed by
    /// [`BootstrappingContext::prepare`] to size preparation exactly.
    ///
    /// # Errors
    ///
    /// [`InvalidOperand`](crate::CKKSCompositionError::InvalidOperand) for a `ci` that does not
    /// match the plan's ring, or for a `module` under degree `2N` on the conjugate-invariant
    /// ring, before any work.
    pub fn compile(
        module: &Module<BE>,
        ci: Option<&Module<BE::CI>>,
        base2k: Base2K,
        plan: &BootstrappingPlan,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<Self>
    where
        Module<BE>: CKKSDFTMatrixPreparedFactory<BE>
            + CKKSDFTOps<BE>
            + CKKSDFTMatrixOps<BE, F>
            + CKKSEncodingOps<BE, F>
            + CKKSRightOperandPreparedFactory<BE>,
        Module<BE::CI>: CKKSEncodingOps<BE::CI, F>,
        CKKSPlaintextOwned<BE>: GLWEToBackendRef<BE> + CKKSCtBounds + GLWEDiagonalProd<BE>,
    {
        Self::compile_unprepared(module, ci, base2k, plan, scratch)?.prepare(module, scratch)
    }

    /// Scratch of [`Self::compile_unprepared`] for `plan`. On the standard ring, the largest of
    /// the queries of the stages, which run one after the other on the same scratch,
    /// [`CKKSDFTMatrixOps::ckks_dft_matrix_new_tmp_bytes`] of each matrix and the slot encoding
    /// EvalMod stages its coefficients through. On the conjugate-invariant ring, of bootstrap
    /// degree `N = ci.n()`, the larger of that query, with the DFT formats resolved at degree
    /// `N`, and the encoding of `u` and `v`, of degree `2N`, plus the encoding of the
    /// coefficients of a plaintext of degree `N` on `ci`.
    ///
    /// # Errors
    ///
    /// As [`Self::compile`], and the plan-versus-ring error of [`CKKSDFTMatrixOps::ckks_dft_matrix_new`].
    pub fn compile_unprepared_tmp_bytes(
        module: &Module<BE>,
        ci: Option<&Module<BE::CI>>,
        plan: &BootstrappingPlan,
    ) -> Result<usize>
    where
        Module<BE>: CKKSDFTMatrixOps<BE, F> + CKKSEncodingOps<BE, F>,
        Module<BE::CI>: CKKSEncodingOps<BE::CI, F>,
    {
        plan.check_ci_module("compile_unprepared_tmp_bytes", module, ci)?;
        let Some(ci) = ci else {
            return Self::stages_tmp_bytes(module, plan);
        };
        let n = ci.n();
        let plan = plan.with_dft_formats_resolved_at(ci.log_n());
        let host = <Module<BE> as CKKSEncodingHostOps<BE, F>>::ckks_encoding_host_tmp_bytes(module, n);
        let host_ci = <Module<BE::CI> as CKKSEncodingHostOps<BE::CI, F>>::ckks_encoding_host_tmp_bytes(ci, n / 2);
        Ok(Self::stages_tmp_bytes(module, &plan)?.max(host) + host_ci)
    }

    /// [`Self::compile_unprepared_tmp_bytes`] of the stages on `module`.
    fn stages_tmp_bytes(module: &Module<BE>, plan: &BootstrappingPlan) -> Result<usize>
    where
        Module<BE>: CKKSDFTMatrixOps<BE, F> + CKKSEncodingOps<BE, F>,
    {
        let mut bytes = module
            .ckks_dft_matrix_new_tmp_bytes::<Encode, Split>(&plan.coeffs_to_slots)?
            .max(module.ckks_dft_matrix_new_tmp_bytes::<Decode, Split>(&plan.slots_to_coeffs)?);
        if let Some(bypass) = plan.coeffs_to_slots_bypass() {
            bytes = bytes.max(module.ckks_dft_matrix_new_tmp_bytes::<Encode, Split>(bypass)?);
        }
        if plan.eval_mod_ring() == EvalModRing::Standard {
            bytes = bytes.max(<Module<BE> as CKKSEncodingHostOps<BE, F>>::ckks_encoding_host_tmp_bytes(
                module,
                module.n() / 2,
            ));
        }
        Ok(bytes)
    }

    /// First stage of [`Self::compile`]: generates the DFT matrices and encodes
    /// EvalMod, and the conversions on the conjugate-invariant ring, without preparing the
    /// matrices.
    ///
    /// # Errors
    ///
    /// As [`Self::compile`].
    pub fn compile_unprepared(
        module: &Module<BE>,
        ci: Option<&Module<BE::CI>>,
        base2k: Base2K,
        plan: &BootstrappingPlan,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<BootstrappingContext<BE, F>>
    where
        Module<BE>: CKKSDFTMatrixOps<BE, F> + CKKSEncodingOps<BE, F>,
        Module<BE::CI>: CKKSEncodingOps<BE::CI, F>,
    {
        plan.check_ci_module("compile_unprepared", module, ci)?;
        let (Some(ci), EvalModRing::ConjugateInvariant { pre, post }) = (ci, plan.eval_mod_ring()) else {
            return Self::compile_stages(module, base2k, plan, scratch);
        };
        // `compile_stages` resolves the DFT formats at the degree of `module`, at least
        // `2N`, where every slot of degree `N` would be a sparse plan's: resolve them at
        // the bootstrap's degree first.
        let plan = plan.with_dft_formats_resolved_at(ci.log_n());
        let mut context = Self::compile_stages(module, base2k, &plan, scratch)?;
        context.eval_mod_ci = Some(EvalModCI::compile(
            module,
            ci,
            base2k,
            plan.eval_mod,
            [pre, post],
            &mut context.coeffs_to_slots,
            scratch,
        )?);
        Ok(context)
    }

    /// The stages on `module`: the DFT matrices, and EvalMod on the standard ring.
    fn compile_stages(
        module: &Module<BE>,
        base2k: Base2K,
        plan: &BootstrappingPlan,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<BootstrappingContext<BE, F>>
    where
        Module<BE>: CKKSDFTMatrixOps<BE, F> + CKKSEncodingOps<BE, F>,
    {
        let mut coeffs_to_slots = module.ckks_dft_matrix_new::<Encode, Split>(base2k, &plan.coeffs_to_slots, scratch)?;
        // A first factor left to the modulus raise is evaluated from its plaintexts,
        // not by the matrix.
        let coeffs_to_slots_first_factor = plan
            .coeffs_to_slots
            .first_factor_on_raise()
            .then(|| coeffs_to_slots.inner.factors.remove(0));
        let slots_to_coeffs = module.ckks_dft_matrix_new::<Decode, Split>(base2k, &plan.slots_to_coeffs, scratch)?;
        let eval_mod = match plan.eval_mod_ring() {
            EvalModRing::Standard => Some(ckks_eval_mod_compile::<BE, F>(base2k, plan.eval_mod, module, scratch)?),
            EvalModRing::ConjugateInvariant { .. } => None,
        };
        let coeffs_to_slots_bypass = match plan.coeffs_to_slots_bypass() {
            Some(bypass) => Some(module.ckks_dft_matrix_new::<Encode, Split>(base2k, bypass, scratch)?),
            None => None,
        };

        Ok(BootstrappingContext {
            plan: plan.clone(),
            c2s_guard_bits: plan.c2s_guard_bits(),
            functional_message_modulus: plan.functional_message_modulus,
            coeffs_to_slots,
            coeffs_to_slots_first_factor,
            coeffs_to_slots_bypass,
            slots_to_coeffs,
            eval_mod_plan: plan.eval_mod,
            eval_mod,
            eval_mod_ring: plan.eval_mod_ring(),
            eval_mod_ci: None,
            pipeline: plan.pipeline(),
            sparse_secret_hamming_weight: plan.sparse_secret_hamming_weight(),
        })
    }
}
