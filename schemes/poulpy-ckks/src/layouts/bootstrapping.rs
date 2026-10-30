//! Bootstrapping parameterization.
//!
//! [`BootstrappingPlan`] bundles the parameters of the three homomorphic
//! sub-circuits that make up CKKS bootstrapping — CoeffsToSlots (an `Encode`
//! [`DFTPlan`]), EvalMod (an [`EvalModPlan`]) and SlotsToCoeffs (a `Decode`
//! [`DFTPlan`]).
//!
//! The plan also selects the ModUp/EvalMod pipeline and its optional techniques.
//! Torus widths (`base2k`, the encoding scale `log_delta`, and the input/raised
//! moduli) remain per-ciphertext metadata supplied at call time — ModUp reads
//! them from its source and destination ciphertexts — and the message ratio is
//! already part of the [`EvalModPlan`]. See
//! [`CKKSBootstrappingOps`](crate::api::CKKSBootstrappingOps) for the ModUp
//! semantics (a digit shift in the base-`2^base2k` torus model, not an RNS
//! prime-basis extension).
//!
//! [`CKKSBootstrappingOps::ckks_bootstrap`](crate::api::CKKSBootstrappingOps)
//! composes the selected pipeline from a compiled [`BootstrappingContextPrepared`] and
//! a prepared key set. The plan remains a plain parameter bundle, so callers
//! can also compose the stages manually from the respective op traits.
//!
//! [`BootstrappingContextPrepared`] is the *compiled* form of a [`BootstrappingPlan`]:
//! the prepared, backend-resident homomorphic DFT matrices and the encoded,
//! uploaded EvalMod, built once and reused across bootstraps.

use std::collections::BTreeMap;

use crate::layouts::{CKKSPlaintext, CKKSPlaintextOwned};
use anyhow::{Result, ensure};
use poulpy_core::{
    GLWEDiagonalProd, LinearTransformationPrepared,
    layouts::{Base2K, Diagonals, GLWEToBackendRef, LWEInfos, LinearTransformation},
};
use poulpy_hal::layouts::{Backend, ConjugateInvariant, Module, ScratchArena};

use crate::{
    CKKSCtBounds, CKKSInfos, GLWEPlaintextMeta, Scale, SetCKKSInfos, SlotsKind,
    api::{CKKSDFTMatrixOps, CKKSDFTOps, CKKSEncodingHostOps, CKKSEncodingOps, CKKSEncodingScalar},
    layouts::{
        CKKSModuleAlloc, ComplexDiagonals, DFTMatrix, DFTMatrixPrepared, DFTPlan, DFTType, Decode, DftScalar, Encode, EncodedLut,
        EvalMod, EvalModPlan, Split,
        eval_mod::{EvalModType, compile_eval_mod},
    },
};

/// Stage ordering for the ModUp/EvalMod bootstrapping family.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BootstrappingPipeline {
    /// CoeffsToSlots before EvalMod and SlotsToCoeffs.
    C2SFirst,
    /// SlotsToCoeffs below ModUp, then CoeffsToSlots and EvalMod above it.
    S2CFirst,
}

/// The ring EvalMod runs on, after an S2C-first CoeffsToSlots.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum EvalModRing {
    /// The ring of the bootstrap, once per half of the slots.
    #[default]
    Standard,
    /// The conjugate-invariant ring of the same degree, once for all `N`
    /// coefficients, reached through the standard ring of degree `2N`: by `u`
    /// before EvalMod, `pre`, and `v` after it, `post`.
    ConjugateInvariant { pre: CIConversion, post: CIConversion },
}

/// How a conversion of [`EvalModRing::ConjugateInvariant`] reaches the ciphertext.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CIConversion {
    /// A product by its plaintext, at scale `2^log_delta`, which consumes `log_delta` bits.
    Product { log_delta: usize },
    /// Folded into the adjacent DFT factor, which then runs twice, on the even and
    /// the odd part of the plaintext: no product and no budget. Only `u`, into
    /// the last CoeffsToSlots factor.
    Folded,
}

impl CIConversion {
    /// Budget the conversion consumes.
    pub fn consumed_bits(&self) -> usize {
        match *self {
            Self::Product { log_delta } => log_delta,
            Self::Folded => 0,
        }
    }
}

impl EvalModRing {
    /// Budget the products of the conversions consume.
    pub fn consumed_bits(&self) -> usize {
        match *self {
            Self::Standard => 0,
            Self::ConjugateInvariant { pre, post } => pre.consumed_bits() + post.consumed_bits(),
        }
    }

    fn pre_consumed_bits(&self) -> usize {
        match *self {
            Self::Standard => 0,
            Self::ConjugateInvariant { pre, .. } => pre.consumed_bits(),
        }
    }
}

/// Sparse-secret encapsulation parameters for ModUp.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SparseSecretEncapsulation {
    /// Sparse-secret ModRaise encapsulation (<https://eprint.iacr.org/2022/024>);
    /// this is the ephemeral ternary secret weight. Use `None` in
    /// [`BootstrappingTechniques`] to disable the technique.
    pub hamming_weight: usize,
}

/// EvalRound+ parameters (<https://eprint.iacr.org/2024/1379>).
#[derive(Clone, Debug)]
pub struct EvalRoundPlus {
    /// High-precision CoeffsToSlots transform bypassing EvalMod.
    pub coeffs_to_slots_bypass: DFTPlan,
}

/// Optional techniques composed into a ModUp/EvalMod recipe.
#[derive(Clone, Debug, Default)]
pub struct BootstrappingTechniques {
    /// Sparse-secret key switches around ModUp.
    pub sparse_secret_encapsulation: Option<SparseSecretEncapsulation>,
    /// EvalRound+ high-precision bypass.
    pub eval_round_plus: Option<EvalRoundPlus>,
}

/// End-to-end parameterization of CKKS bootstrapping.
///
/// Built with [`Self::new`], which validates the stage roles once — a
/// `BootstrappingPlan` that exists always has an `Encode` CoeffsToSlots and a
/// `Decode` SlotsToCoeffs, so key provisioning ([`Self::galois_elements`])
/// cannot silently derive rotations for the wrong transform direction.
///
/// See the [module docs](self) for the overall role.
#[derive(Clone, Debug)]
pub struct BootstrappingPlan {
    pipeline: BootstrappingPipeline,
    c2s_guard_bits: usize,
    functional_message_modulus: Option<usize>,
    eval_mod_ring: EvalModRing,

    techniques: BootstrappingTechniques,

    /// CoeffsToSlots: homomorphic encoding ([`DFTType::Encode`]).
    pub(crate) coeffs_to_slots: DFTPlan,
    /// EvalMod: approximate `x mod 1`. EvalMod runs at its own
    /// ([`EvalModPlan::f_mod_log_delta`]) scale — `ckks_eval_mod` sets the
    /// ciphertext to it on entry and restores the input scale on exit (a pure,
    /// budget-neutral reinterpretation), so it can keep more `ct×ct` precision.
    pub(crate) eval_mod: EvalModPlan,

    /// SlotsToCoeffs: homomorphic decoding ([`DFTType::Decode`]).
    pub(crate) slots_to_coeffs: DFTPlan,
}

impl BootstrappingPlan {
    /// The stages that set the widths at which a bootstrap uses its keys.
    fn stage_widths(&self) -> StageWidths<'_> {
        StageWidths {
            pipeline: self.pipeline,
            c2s_guard_bits: self.c2s_guard_bits,
            eval_mod_ring: self.eval_mod_ring,
            coeffs_to_slots: &self.coeffs_to_slots,
            coeffs_to_slots_bypass: self.coeffs_to_slots_bypass(),
            slots_to_coeffs: &self.slots_to_coeffs,
            eval_mod: &self.eval_mod,
        }
    }

    /// The widths at which a bootstrap of this plan uses its keys, at degree
    /// `2^log_n`, working at `bootstrap_k` bits on an input of `input_k` bits at
    /// scale `2^input_log_delta`.
    ///
    /// The presets lay their keys out from them, and
    /// [`ckks_bootstrap`](crate::api::CKKSBootstrappingOps::ckks_bootstrap) checks
    /// every rotation and tensor key against them before it starts.
    ///
    /// # Panics
    ///
    /// When `bootstrap_k` or `input_k` is narrower than the bits the stages running
    /// on it consume.
    pub fn key_widths(&self, log_n: usize, bootstrap_k: usize, input_k: usize, input_log_delta: usize) -> BootstrappingKeyWidths {
        self.stage_widths()
            .key_widths(log_n, bootstrap_k, input_k, input_log_delta)
            .expect("a bootstrap width is narrower than the bits its stages consume")
    }

    /// Validated constructor for a complete bootstrapping recipe.
    ///
    /// `coeffs_to_slots` must be a
    /// [`DFTType::Encode`] plan and `slots_to_coeffs` a [`DFTType::Decode`]
    /// plan (errors [`InvalidPlan`](crate::CKKSCompositionError::InvalidPlan)
    /// otherwise) — a swapped direction would make [`Self::galois_elements`]
    /// derive rotation keys for a transform that
    /// [`BootstrappingContextPrepared::compile`] never builds. The constructor also
    /// rejects recipe techniques that the selected pipeline cannot evaluate.
    pub fn new(
        pipeline: BootstrappingPipeline,
        techniques: BootstrappingTechniques,
        coeffs_to_slots: DFTPlan,
        eval_mod: EvalModPlan,
        slots_to_coeffs: DFTPlan,
    ) -> Result<Self> {
        let invalid = |reason: String| -> anyhow::Error {
            crate::CKKSCompositionError::InvalidPlan {
                plan: "BootstrappingPlan",
                reason,
            }
            .into()
        };
        if coeffs_to_slots.kind() != DFTType::Encode {
            return Err(invalid("coeffs_to_slots must be a DFTType::Encode plan".to_string()));
        }
        if slots_to_coeffs.kind() != DFTType::Decode {
            return Err(invalid("slots_to_coeffs must be a DFTType::Decode plan".to_string()));
        }
        if coeffs_to_slots.first_factor_on_raise() && techniques.eval_round_plus.is_some() {
            return Err(invalid(
                "a first CoeffsToSlots factor on the raise cannot be combined with EvalRound+".to_string(),
            ));
        }
        if slots_to_coeffs.first_factor_on_raise() {
            return Err(invalid(
                "only CoeffsToSlots follows the raise: slots_to_coeffs cannot leave its first factor to it".to_string(),
            ));
        }
        eval_mod.validate()?;
        let coeffs_to_slots = coeffs_to_slots.with_scaling(eval_mod.input_scaling()?)?;
        if let Some(sse) = techniques.sparse_secret_encapsulation
            && sse.hamming_weight == 0
        {
            return Err(invalid(
                "sparse-secret encapsulation hamming_weight must be nonzero".to_string(),
            ));
        }
        if let Some(eval_round) = &techniques.eval_round_plus {
            if eval_round.coeffs_to_slots_bypass.kind() != DFTType::Encode {
                return Err(invalid(
                    "EvalRound+ coeffs_to_slots_bypass must be a DFTType::Encode plan".to_string(),
                ));
            }
            if !eval_mod.f_mod_interval.is_power_of_two() {
                return Err(invalid(format!(
                    "EvalRound+ requires a power-of-two f_mod_interval, got {}",
                    eval_mod.f_mod_interval
                )));
            }
        }
        Ok(Self {
            pipeline,
            c2s_guard_bits: 0,
            functional_message_modulus: None,
            eval_mod_ring: EvalModRing::Standard,
            techniques,
            coeffs_to_slots,
            eval_mod,
            slots_to_coeffs,
        })
    }

    /// Configures S2C-first functional bootstrapping of integer messages modulo the LUT length.
    /// The initial transform includes the normalization to the LUT's unit-circle phase.
    pub fn with_functional_bootstrap<P>(mut self, lut: &EncodedLut<P>) -> Result<Self> {
        ensure!(
            self.pipeline == BootstrappingPipeline::S2CFirst,
            "functional bootstrapping requires an S2C-first plan"
        );
        ensure!(
            self.eval_mod_ring == EvalModRing::Standard,
            "functional bootstrapping runs EvalMod on the standard ring"
        );
        let q = (1usize << lut.log_msg_ratio()) as f64;
        self.slots_to_coeffs = self.slots_to_coeffs.with_scaling(q / (2.0 * lut.message_modulus() as f64))?;
        self.functional_message_modulus = Some(lut.message_modulus());
        Ok(self)
    }

    /// The selected ModUp/EvalMod bootstrapping pipeline.
    pub fn pipeline(&self) -> BootstrappingPipeline {
        self.pipeline
    }

    /// Adds scale bits at ModUp to reduce CoeffsToSlots rounding error.
    /// Charged to the raised width and removed at the ordinary bootstrap output.
    pub fn with_c2s_guard_bits(mut self, bits: usize) -> Result<Self> {
        ensure!(
            self.pipeline == BootstrappingPipeline::S2CFirst,
            "C2S guard bits require an S2C-first plan"
        );
        ensure!(
            (self.post_mod_up_consumed_bits() - self.c2s_guard_bits)
                .checked_add(bits)
                .is_some(),
            "C2S guard bits overflow the post-ModUp budget"
        );
        self.c2s_guard_bits = bits;
        Ok(self)
    }

    /// Extra internal scale bits used between ModUp and CoeffsToSlots.
    pub fn c2s_guard_bits(&self) -> usize {
        self.c2s_guard_bits
    }

    /// Selects the ring EvalMod runs on. The conjugate-invariant ring takes an
    /// S2C-first plan with a real EvalMod, without a LUT; `u` folds
    /// into a CoeffsToSlots factor the matrix evaluates, `v` does not fold.
    pub fn with_eval_mod_ring(mut self, ring: EvalModRing) -> Result<Self> {
        ensure!(
            ring == EvalModRing::Standard
                || (self.pipeline == BootstrappingPipeline::S2CFirst
                    && self.functional_message_modulus.is_none()
                    && self.eval_mod.eval_mod_type != EvalModType::ExpCmplx),
            "EvalMod on the conjugate-invariant ring takes an S2C-first plan with a real EvalMod, without a LUT"
        );
        if let EvalModRing::ConjugateInvariant { pre, post } = ring {
            ensure!(post != CIConversion::Folded, "the product by v after EvalMod does not fold");
            ensure!(
                pre != CIConversion::Folded || self.coeffs_to_slots.num_evaluated_factors() > 0,
                "folding u takes a CoeffsToSlots factor the matrix evaluates"
            );
        }
        self.eval_mod_ring = ring;
        Ok(self)
    }

    /// The ring EvalMod runs on.
    pub fn eval_mod_ring(&self) -> EvalModRing {
        self.eval_mod_ring
    }

    /// Optional techniques applied by the recipe.
    pub fn techniques(&self) -> &BootstrappingTechniques {
        &self.techniques
    }

    /// CoeffsToSlots: homomorphic encoding ([`DFTType::Encode`]).
    pub fn coeffs_to_slots(&self) -> &DFTPlan {
        &self.coeffs_to_slots
    }

    /// The optional high-precision CoeffsToSlots bypass (EvalRound+).
    pub fn coeffs_to_slots_bypass(&self) -> Option<&DFTPlan> {
        self.techniques
            .eval_round_plus
            .as_ref()
            .map(|eval_round| &eval_round.coeffs_to_slots_bypass)
    }

    /// Ephemeral sparse-secret weight required by the recipe, if enabled.
    pub fn sparse_secret_hamming_weight(&self) -> Option<usize> {
        self.techniques.sparse_secret_encapsulation.map(|sse| sse.hamming_weight)
    }

    /// EvalMod: approximate `x mod 1`.
    pub fn eval_mod(&self) -> &EvalModPlan {
        &self.eval_mod
    }

    /// SlotsToCoeffs: homomorphic decoding ([`DFTType::Decode`]).
    pub fn slots_to_coeffs(&self) -> &DFTPlan {
        &self.slots_to_coeffs
    }

    /// The plan with its DFT formats resolved at degree `2^log_n` ([`DFTPlan::resolved_at`]).
    fn with_dft_formats_resolved_at(&self, log_n: usize) -> Self {
        let mut plan = self.clone();
        plan.coeffs_to_slots = plan.coeffs_to_slots.resolved_at(log_n);
        plan.slots_to_coeffs = plan.slots_to_coeffs.resolved_at(log_n);
        if let Some(eval_round) = plan.techniques.eval_round_plus.as_mut() {
            eval_round.coeffs_to_slots_bypass = eval_round.coeffs_to_slots_bypass.resolved_at(log_n);
        }
        plan
    }

    /// Budget consumed before ModUp.
    pub fn pre_mod_up_consumed_bits(&self) -> usize {
        match self.pipeline {
            BootstrappingPipeline::C2SFirst => 0,
            BootstrappingPipeline::S2CFirst => self.slots_to_coeffs.consumed_bits(),
        }
    }

    /// Budget consumed after ModUp.
    ///
    /// EvalRound+ evaluates its bypass in parallel with the low-precision
    /// CoeffsToSlots + EvalMod branch, so the wider of the two branch costs is
    /// charged before any trailing SlotsToCoeffs. The conversions to the ring
    /// EvalMod runs on belong to that branch.
    /// S2C-first also charges the CoeffsToSlots guard bits.
    pub fn post_mod_up_consumed_bits(&self) -> usize {
        let c2s_eval_mod =
            self.coeffs_to_slots.consumed_bits() + self.eval_mod.consumed_bits() + self.eval_mod_ring.consumed_bits();
        let eval_round = self
            .coeffs_to_slots_bypass()
            .map_or(c2s_eval_mod, |bypass| c2s_eval_mod.max(bypass.consumed_bits()));
        match self.pipeline {
            BootstrappingPipeline::C2SFirst => eval_round + self.slots_to_coeffs.consumed_bits(),
            BootstrappingPipeline::S2CFirst => eval_round + self.c2s_guard_bits,
        }
    }

    /// Input width for a given modulus at ModUp.
    pub fn input_k(&self, log_modulus: usize) -> usize {
        log_modulus + self.pre_mod_up_consumed_bits()
    }

    /// Bits the raised ciphertext fills at the bottom of the raised width, for
    /// a modulus of `log_modulus` bits at ModUp: what a sparse-to-dense key
    /// with a row offset of `bootstrap_k - raised_window` switches
    /// ([`EncapsulationKeysLayout::sparse_to_dense`](crate::layouts::EncapsulationKeysLayout::sparse_to_dense)).
    /// C2S-first lifts every input to the EvalMod scale, so its window does not
    /// depend on the input; S2C-first raises its modulus and the C2S guard bits,
    /// and a narrower input is shifted into the window.
    pub fn raised_window(&self, log_modulus: usize) -> usize {
        match self.pipeline {
            BootstrappingPipeline::C2SFirst => self.eval_mod.f_mod_log_delta,
            BootstrappingPipeline::S2CFirst => log_modulus + self.c2s_guard_bits,
        }
    }

    /// Bootstrap width for an output at the input scale `log_delta`.
    /// C2S-first includes the width removed when restoring that scale.
    pub fn bootstrap_k(&self, output_k: usize, log_delta: usize) -> usize {
        let scale_drop = match self.pipeline {
            BootstrappingPipeline::C2SFirst => self
                .eval_mod
                .f_mod_log_delta
                .checked_sub(log_delta)
                .expect("C2S-first input scale exceeds EvalMod scale"),
            BootstrappingPipeline::S2CFirst => 0,
        };
        output_k + self.post_mod_up_consumed_bits() + scale_drop
    }

    /// Raised width required by an S2C-first functional bootstrap.
    ///
    /// `log_delta` is the input ciphertext scale; the LUT message ratio is
    /// folded in to obtain the post-S2C scale used by the LUT evaluation.
    pub fn functional_bootstrap_k<P>(&self, output_k: usize, log_delta: usize, lut: &EncodedLut<P>) -> Result<usize>
    where
        P: CKKSInfos,
    {
        ensure!(
            self.pipeline == BootstrappingPipeline::S2CFirst,
            "functional bootstrapping requires an S2C-first plan"
        );
        let eval_mod = if lut.requires_eval_mod() {
            self.eval_mod.consumed_bits()
        } else {
            0
        };
        ensure!(
            self.functional_message_modulus.is_none_or(|p| p == lut.message_modulus()),
            "functional bootstrap plan and LUT must have the same message modulus"
        );
        let lut_log_delta = log_delta + lut.log_msg_ratio();
        Ok(output_k + self.c2s_guard_bits + self.coeffs_to_slots.consumed_bits() + eval_mod + lut.consumed_bits(lut_log_delta))
    }

    /// Total `log_budget` bits the pipeline consumes: the two DFT stages plus
    /// EvalMod, plus any CoeffsToSlots guard bits. EvalMod is charged at its
    /// own `f_mod_log_delta` scale; its set-scale round-trip is budget-neutral.
    pub fn consumed_bits(&self) -> usize {
        self.pre_mod_up_consumed_bits() + self.post_mod_up_consumed_bits()
    }

    /// Distinct Galois elements the pipeline's rotation keys must cover: the union
    /// of the CoeffsToSlots and SlotsToCoeffs transforms
    /// ([`DFTPlan::galois_elements`]), of the bypass, and, with `u` folded into the
    /// last CoeffsToSlots factor ([`CIConversion::Folded`]), of the two folded factors,
    /// for a ring of degree `2^log_n` and the given `cyclotomic_order`. The split
    /// forward transform's conjugation key is separate (generate it from Galois
    /// element `−1`) and not included here.
    pub fn galois_elements(&self, log_n: usize, cyclotomic_order: i64) -> Vec<i64> {
        let mut set: std::collections::BTreeSet<i64> = std::collections::BTreeSet::new();
        set.extend(self.coeffs_to_slots.galois_elements(log_n, cyclotomic_order));
        if let Some(bypass) = self.coeffs_to_slots_bypass() {
            set.extend(bypass.galois_elements(log_n, cyclotomic_order));
        }
        set.extend(self.slots_to_coeffs.galois_elements(log_n, cyclotomic_order));
        // The folded factors are the last CoeffsToSlots factor times diagonals of the full
        // slots, as `compile_ci` builds them.
        if let EvalModRing::ConjugateInvariant {
            pre: CIConversion::Folded,
            ..
        } = self.eval_mod_ring
        {
            set.extend(self.coeffs_to_slots.resolved_at(log_n).last_factor_galois_elements_at(
                log_n,
                1 << (log_n - 1),
                cyclotomic_order,
            ));
        }
        set.into_iter().collect()
    }
}

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

    /// First CoeffsToSlots factor, when the plan leaves it to the modulus
    /// raise ([`DFTPlan::with_first_factor_on_raise`]): one giant step per
    /// diagonal, the diagonal rotated back by its own index. Kept unprepared,
    /// its products run on a ciphertext of a few limbs.
    coeffs_to_slots_first_factor: Option<LinearTransformation<CKKSPlaintextOwned<BE>>>,

    /// Prepared bypass CoeffsToSlots matrix
    coeffs_to_slots_bypass: Option<DFTMatrixPrepared<BE, Encode, Split>>,

    /// Prepared SlotsToCoeffs matrix (homomorphic decoding). S2C-first plans
    /// cancel the initial real/imaginary split and normalize functional messages.
    slots_to_coeffs: DFTMatrixPrepared<BE, Decode, Split>,

    /// EvalMod (`x mod 1`): its plan, and its encoding when it runs on the
    /// standard ring.
    eval_mod_plan: EvalModPlan,
    eval_mod: Option<EvalMod<F, CKKSPlaintextOwned<BE>>>,

    /// The ring EvalMod runs on, and its plaintexts on the conjugate-invariant
    /// ring ([`Self::compile_ci`]).
    eval_mod_ring: EvalModRing,
    eval_mod_ci: Option<EvalModCIPrepared<BE, F>>,

    /// Selected ModUp/EvalMod pipeline.
    pipeline: BootstrappingPipeline,

    /// Ephemeral sparse-secret weight required by the recipe, if enabled.
    sparse_secret_hamming_weight: Option<usize>,
}

/// EvalMod on the conjugate-invariant ring of degree `N` and the plaintexts of
/// degree `2N` of the conversions there and back
/// ([`EvalModRing::ConjugateInvariant`]).
pub struct EvalModCIPrepared<BE: Backend, F> {
    /// EvalMod in plaintexts of the CI ring.
    pub(crate) eval_mod: EvalMod<F, CKKSPlaintext<BE::OwnedBuf, BE::ZnxWord, ConjugateInvariant>>,
    /// `u`, `1` on slot `j` and `−i` on slot `j + N/2`: the trace of `u·z(X²)`
    /// holds the real parts of `z_j` and `−i·z_j`, `c_j` and `c_{j+N/2}`.
    pub(crate) pre: CIPrePrepared<BE>,
    /// `v`, `2` on slot `j` and `2i` on slot `j + N/2`, which pairs them back and restores the
    /// factor 2 of the conjugate pair.
    pub(crate) post: CKKSPlaintextOwned<BE>,
}

impl<BE: Backend, F> EvalModCIPrepared<BE, F> {
    /// EvalMod in plaintexts of the CI ring.
    pub fn eval_mod(&self) -> &EvalMod<F, CKKSPlaintext<BE::OwnedBuf, BE::ZnxWord, ConjugateInvariant>> {
        &self.eval_mod
    }

    /// `u`, the conversion before EvalMod.
    pub fn pre(&self) -> &CIPrePrepared<BE> {
        &self.pre
    }

    /// `v`, the conversion after EvalMod.
    pub fn post(&self) -> &CKKSPlaintextOwned<BE> {
        &self.post
    }
}

/// `u` of [`EvalModCIPrepared`], as its [`CIConversion`] applies it.
pub struct CIPrePrepared<BE: Backend>(pub(crate) CIPreForm<BE>);

pub(crate) enum CIPreForm<BE: Backend> {
    Product(CKKSPlaintextOwned<BE>),
    Folded {
        even: LinearTransformationPrepared<BE>,
        odd: LinearTransformationPrepared<BE>,
    },
}

impl<BE: Backend> CIPrePrepared<BE> {
    /// `u` as a plaintext of degree `2N`, for [`CIConversion::Product`].
    pub fn product(&self) -> Option<&CKKSPlaintextOwned<BE>> {
        match &self.0 {
            CIPreForm::Product(u) => Some(u),
            CIPreForm::Folded { .. } => None,
        }
    }

    /// For [`CIConversion::Folded`], the last CoeffsToSlots factor `M` times the
    /// slots of the even and odd parts of `u = u_e(X²) + X·u_o(X²)`, in that order:
    /// with `a = u_e·M(w)` and `b = u_o·M(w)` of degree `N`, `u·z(X²) = a(X²) + X·b(X²)`.
    /// The CoeffsToSlots matrix of the context then holds the other factors.
    pub fn folded(&self) -> Option<[&LinearTransformationPrepared<BE>; 2]> {
        match &self.0 {
            CIPreForm::Product(_) => None,
            CIPreForm::Folded { even, odd } => Some([even, odd]),
        }
    }
}

/// [`EvalModCIPrepared`] before its folded factors are prepared.
struct EvalModCI<BE: Backend, F> {
    eval_mod: EvalMod<F, CKKSPlaintext<BE::OwnedBuf, BE::ZnxWord, ConjugateInvariant>>,
    pre: CIPre<BE>,
    post: CKKSPlaintextOwned<BE>,
}

/// [`CIPrePrepared`] before its folded factors are prepared.
enum CIPre<BE: Backend> {
    Product(CKKSPlaintextOwned<BE>),
    /// The even and odd factors, in a matrix of the CoeffsToSlots plan.
    Folded(DFTMatrix<BE, Encode, Split>),
}

/// The widths, in bits, at which a bootstrap uses its keys
/// ([`BootstrappingPlan::key_widths`]): a key covers its width when its
/// `max_input_k` is at least that width.
///
/// A rotation key's width is the input width of the widest transform factor
/// using it, which bounds the ciphertexts it switches: a key used only for giant
/// steps switches the factor's product, one factor scale narrower. The tensor
/// key's width is that of the widest product EvalMod relinearizes.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct BootstrappingKeyWidths {
    /// Every rotation key of the plan, by Galois element, the conjugation `-1`
    /// included with EvalMod on the standard ring.
    pub rotation_keys: BTreeMap<i64, usize>,
    /// The EvalMod relinearization (tensor) key.
    pub tensor_key: usize,
    /// The trace and embedding keys, with EvalMod on the conjugate-invariant ring.
    pub conversion_keys: Option<(usize, usize)>,
}

/// The stages that set the widths at which a bootstrap uses its keys, given the
/// width it runs at and its input.
pub(crate) struct StageWidths<'a> {
    pipeline: BootstrappingPipeline,
    c2s_guard_bits: usize,
    eval_mod_ring: EvalModRing,
    coeffs_to_slots: &'a DFTPlan,
    coeffs_to_slots_bypass: Option<&'a DFTPlan>,
    slots_to_coeffs: &'a DFTPlan,
    eval_mod: &'a EvalModPlan,
}

impl StageWidths<'_> {
    /// [`BootstrappingPlan::key_widths`], or `None` when `bootstrap_k` or `input_k`
    /// is narrower than the bits the stages running on it consume.
    pub(crate) fn key_widths(
        &self,
        log_n: usize,
        bootstrap_k: usize,
        input_k: usize,
        input_log_delta: usize,
    ) -> Option<BootstrappingKeyWidths> {
        Some(BootstrappingKeyWidths {
            rotation_keys: self.rotation_key_inputs(log_n, bootstrap_k, input_k)?,
            tensor_key: self.eval_mod_relinearization_k(bootstrap_k, input_log_delta),
            conversion_keys: self.conversions(bootstrap_k),
        })
    }

    /// Width, in bits, of each rotation key, by Galois element: the input width of
    /// the widest factor using it (see [`BootstrappingKeyWidths`]).
    ///
    /// Factor `i` of a transform runs on a ciphertext `i` factor scales below the
    /// width the transform starts at: the raised width `bootstrap_k` for
    /// CoeffsToSlots and its EvalRound+ bypass, and for SlotsToCoeffs the input
    /// width `input_k` (S2C-first) or what EvalMod, and the bypass, leave
    /// (C2S-first). The folded factors of [`CIConversion::Folded`] run where the
    /// last CoeffsToSlots factor would. The conjugation follows each split
    /// CoeffsToSlots, which on the conjugate-invariant ring does not split.
    fn rotation_key_inputs(&self, log_n: usize, bootstrap_k: usize, input_k: usize) -> Option<BTreeMap<i64, usize>> {
        let cyclotomic_order = 1i64 << (log_n + 1);
        let after_c2s = bootstrap_k.checked_sub(self.coeffs_to_slots.consumed_bits())?;
        let after_bypass = match self.coeffs_to_slots_bypass {
            Some(bypass) => Some(bootstrap_k.checked_sub(bypass.consumed_bits())?),
            None => None,
        };
        let s2c_k = match self.pipeline {
            BootstrappingPipeline::S2CFirst => input_k,
            BootstrappingPipeline::C2SFirst => {
                let after_eval_mod = after_c2s.checked_sub(self.eval_mod.consumed_bits())?;
                after_bypass.map_or(after_eval_mod, |k| k.min(after_eval_mod))
            }
        };
        // Every factor below then runs above the bits its transform consumes.
        s2c_k.checked_sub(self.slots_to_coeffs.consumed_bits())?;

        let mut inputs: BTreeMap<i64, usize> = BTreeMap::new();
        let mut widen = |p: i64, k: usize| {
            let widest = inputs.entry(p).or_insert(k);
            *widest = (*widest).max(k);
        };
        let bypass = self.coeffs_to_slots_bypass.map(|bypass| (bypass, bootstrap_k));
        for (dft, start) in [(self.coeffs_to_slots, bootstrap_k), (self.slots_to_coeffs, s2c_k)]
            .into_iter()
            .chain(bypass)
        {
            let log_delta = dft.coeffs_meta().log_delta();
            let factors = dft.galois_elements_per_factor(log_n, cyclotomic_order);
            // The repack rotation of the sparse path belongs to no factor.
            for p in dft.galois_elements(log_n, cyclotomic_order) {
                if !factors.iter().any(|elements| elements.contains(&p)) {
                    widen(p, start);
                }
            }
            // A first factor applied by the raise has no element here and costs no width.
            let free = usize::from(dft.first_factor_on_raise());
            for (factor, elements) in factors.iter().enumerate() {
                for &p in elements {
                    widen(p, start - (factor - free) * log_delta);
                }
            }
        }
        if let EvalModRing::ConjugateInvariant {
            pre: CIConversion::Folded,
            ..
        } = self.eval_mod_ring
        {
            let c2s = self.coeffs_to_slots.resolved_at(log_n);
            let last = c2s.num_factors() - 1;
            let last_k = bootstrap_k - (last - usize::from(c2s.first_factor_on_raise())) * c2s.coeffs_meta().log_delta();
            for p in c2s.last_factor_galois_elements_at(log_n, 1 << (log_n - 1), cyclotomic_order) {
                widen(p, last_k);
            }
        }
        if self.eval_mod_ring == EvalModRing::Standard {
            widen(-1, after_c2s);
            if let Some(k) = after_bypass {
                widen(-1, k);
            }
        }
        Some(inputs)
    }

    /// Widths of the ciphertexts the trace and the embedding switch on the way
    /// to the conjugate-invariant ring and back, if EvalMod runs there.
    pub(crate) fn conversions(&self, bootstrap_k: usize) -> Option<(usize, usize)> {
        let traced = self.eval_mod_k(bootstrap_k);
        (self.eval_mod_ring != EvalModRing::Standard).then(|| (traced, traced.saturating_sub(self.eval_mod.consumed_bits())))
    }

    /// Width of EvalMod's input: what CoeffsToSlots and the product by `u` leave.
    fn eval_mod_k(&self, bootstrap_k: usize) -> usize {
        bootstrap_k.saturating_sub(self.coeffs_to_slots.consumed_bits() + self.eval_mod_ring.pre_consumed_bits())
    }

    /// `(k_small, k_small + scale_up)` of the ciphertext ModUp raises for a
    /// bootstrap input of `input_k` bits: C2S-first lifts it to EvalMod's scale
    /// at its own width (an input wider than that scale fails at the lift),
    /// S2C-first raises it once SlotsToCoeffs is done, with the C2S guard bits.
    pub(crate) fn raise(&self, input_k: usize) -> (usize, usize) {
        match self.pipeline {
            BootstrappingPipeline::C2SFirst => (input_k, self.eval_mod.f_mod_log_delta.max(input_k)),
            BootstrappingPipeline::S2CFirst => {
                let k_small = input_k.saturating_sub(self.slots_to_coeffs.consumed_bits());
                (k_small, k_small + self.c2s_guard_bits)
            }
        }
    }

    /// Widest ciphertext EvalMod relinearizes, in bits, for an input of scale
    /// `2^input_log_delta`. EvalMod reads the low-precision CoeffsToSlots alone,
    /// with or without the EvalRound+ bypass.
    ///
    /// EvalMod relabels its input at its own scale and keeps its budget, so its
    /// widest product is at that budget plus `f_mod_log_delta`. The budget is what
    /// CoeffsToSlots and the product by `u` leave of the raised width above the scale the raise labels:
    /// EvalMod's own scale (C2S-first), or the input scale, the message ratio and
    /// the guard bits (S2C-first).
    fn eval_mod_relinearization_k(&self, bootstrap_k: usize, input_log_delta: usize) -> usize {
        let f_mod_log_delta = self.eval_mod.f_mod_log_delta;
        let raised_log_delta = match self.pipeline {
            BootstrappingPipeline::C2SFirst => f_mod_log_delta,
            BootstrappingPipeline::S2CFirst => input_log_delta + self.eval_mod.log_msg_ratio + self.c2s_guard_bits,
        };
        // A width too narrow for EvalMod fails at its key lookup or in EvalMod, not here.
        (self.eval_mod_k(bootstrap_k) + f_mod_log_delta).saturating_sub(raised_log_delta)
    }
}

impl<BE: Backend, F> BootstrappingContextPrepared<BE, F> {
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

    /// First CoeffsToSlots factor, when the modulus raise applies it
    /// ([`DFTPlan::with_first_factor_on_raise`]); [`Self::coeffs_to_slots`]
    /// then holds the remaining factors.
    pub fn coeffs_to_slots_first_factor(&self) -> Option<&LinearTransformation<CKKSPlaintextOwned<BE>>> {
        self.coeffs_to_slots_first_factor.as_ref()
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

    /// EvalMod on the standard ring, which `op` runs: a context of [`Self::compile_ci`] has none.
    pub(crate) fn standard_eval_mod(&self, op: &'static str) -> crate::CKKSResult<&EvalMod<F, CKKSPlaintextOwned<BE>>> {
        self.eval_mod.as_ref().ok_or_else(|| {
            crate::CKKSCompositionError::InvalidOperand {
                op,
                reason: "the context runs EvalMod on the conjugate-invariant ring".into(),
            }
            .into()
        })
    }

    /// The ring EvalMod runs on.
    pub fn eval_mod_ring(&self) -> EvalModRing {
        self.eval_mod_ring
    }

    /// EvalMod on the conjugate-invariant ring and its conversions, if compiled by [`Self::compile_ci`].
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
    /// budget among the matrices.
    pub fn prepare_tmp_bytes(&self, module: &Module<BE>) -> usize
    where
        Module<BE>: CKKSDFTOps<BE>,
        CKKSPlaintextOwned<BE>: LWEInfos,
    {
        let mut bytes = module
            .ckks_prepare_dft_matrix_tmp_bytes(&self.coeffs_to_slots)
            .max(module.ckks_prepare_dft_matrix_tmp_bytes(&self.slots_to_coeffs));
        if let Some(bypass) = &self.coeffs_to_slots_bypass {
            bytes = bytes.max(module.ckks_prepare_dft_matrix_tmp_bytes(bypass));
        }
        if let Some(CIPre::Folded(folded)) = self.eval_mod_ci.as_ref().map(|ci| &ci.pre) {
            bytes = bytes.max(module.ckks_prepare_dft_matrix_tmp_bytes(folded));
        }
        bytes
    }

    /// Prepares the matrices into the resident form the pipeline evaluates.
    /// Each unprepared matrix is released as soon as it has been prepared.
    ///
    /// # Errors
    ///
    /// As [`CKKSDFTOps::ckks_prepare_dft_matrix`].
    pub fn prepare(self, module: &Module<BE>, scratch: &mut ScratchArena<'_, BE>) -> Result<BootstrappingContextPrepared<BE, F>>
    where
        Module<BE>: CKKSDFTOps<BE>,
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
            module.ckks_prepare_dft_matrix_alloc(&unprepared, scratch)?
        };
        let slots_to_coeffs = {
            let unprepared = slots_to_coeffs;
            module.ckks_prepare_dft_matrix_alloc(&unprepared, scratch)?
        };
        let coeffs_to_slots_bypass = coeffs_to_slots_bypass
            .map(|unprepared| module.ckks_prepare_dft_matrix_alloc(&unprepared, scratch))
            .transpose()?;
        let eval_mod_ci = match eval_mod_ci {
            Some(ci) => Some(EvalModCIPrepared {
                eval_mod: ci.eval_mod,
                pre: match ci.pre {
                    CIPre::Product(pre) => CIPrePrepared(CIPreForm::Product(pre)),
                    CIPre::Folded(folded) => {
                        let [even, odd]: [_; 2] = module
                            .ckks_prepare_dft_matrix_alloc(&folded, scratch)?
                            .inner
                            .factors
                            .try_into()
                            .unwrap_or_else(|_| unreachable!("an even and an odd factor"));
                        CIPrePrepared(CIPreForm::Folded { even, odd })
                    }
                },
                post: ci.post,
            }),
            None => None,
        };
        Ok(BootstrappingContextPrepared {
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
        })
    }
}

impl<BE: Backend, F> BootstrappingContextPrepared<BE, F>
where
    F: CKKSEncodingScalar,
{
    /// Compiles `plan` directly into backend-resident matrices and EvalMod.
    ///
    /// Each stage carries its own coefficient metadata (`DFTPlan::meta` /
    /// `EvalModPlan::meta`), including the CoeffsToSlots scaling resolved by
    /// [`BootstrappingPlan::new`].
    ///
    /// `scratch` serves both generation and preparation, so it must cover
    /// [`Self::compile_unprepared_tmp_bytes`] and
    /// [`BootstrappingContext::prepare_tmp_bytes`]. The latter is only
    /// known once the matrices exist: use [`Self::compile_unprepared`] followed by
    /// [`BootstrappingContext::prepare`] to size preparation exactly.
    pub fn compile(
        module: &Module<BE>,
        base2k: Base2K,
        plan: &BootstrappingPlan,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<Self>
    where
        Module<BE>: CKKSDFTOps<BE> + CKKSDFTMatrixOps<BE, F> + CKKSModuleAlloc<BE> + CKKSEncodingOps<BE, F>,
        CKKSPlaintextOwned<BE>: GLWEToBackendRef<BE> + CKKSCtBounds + GLWEDiagonalProd<BE>,
    {
        Self::compile_unprepared(module, base2k, plan, scratch)?.prepare(module, scratch)
    }

    /// [`Self::compile`] for EvalMod on the conjugate-invariant ring: also encodes
    /// EvalMod on `ci`, of degree `N`, and `u` and `v` at degree `2N`, which
    /// `module` serves on top of the bootstrap's degree `N`. The DFT matrices are
    /// generated as at degree `N`.
    ///
    /// `scratch` serves generation and preparation, as for [`Self::compile`]:
    /// [`Self::compile_ci_unprepared_tmp_bytes`] and
    /// [`BootstrappingContext::prepare_tmp_bytes`] size the two stages of
    /// [`Self::compile_ci_unprepared`] and [`BootstrappingContext::prepare`].
    pub fn compile_ci<BCI>(
        module: &Module<BE>,
        ci: &Module<BCI>,
        base2k: Base2K,
        plan: &BootstrappingPlan,
        scratch: &mut ScratchArena<'_, BE>,
        scratch_ci: &mut ScratchArena<'_, BCI>,
    ) -> Result<Self>
    where
        BCI: Backend<Ring = ConjugateInvariant, OwnedBuf = BE::OwnedBuf, ZnxWord = BE::ZnxWord>,
        F: DftScalar,
        Module<BE>: CKKSDFTOps<BE> + CKKSDFTMatrixOps<BE, F> + CKKSModuleAlloc<BE> + CKKSEncodingOps<BE, F>,
        Module<BCI>: CKKSModuleAlloc<BCI> + CKKSEncodingOps<BCI, F>,
        CKKSPlaintextOwned<BE>: GLWEToBackendRef<BE> + CKKSCtBounds + GLWEDiagonalProd<BE>,
    {
        Self::compile_ci_unprepared(module, ci, base2k, plan, scratch, scratch_ci)?.prepare(module, scratch)
    }

    /// Scratch of [`Self::compile_ci_unprepared`] for `plan` and a bootstrap of degree `N = ci.n()`:
    /// `(scratch, scratch_ci)`. On `module`, the larger of [`Self::compile_unprepared_tmp_bytes`] of `plan` with its
    /// DFT formats resolved at degree `N` and the encoding of `u` and `v`, of degree `2N`; on `ci`, the encoding of the
    /// coefficients of a plaintext of degree `N`.
    ///
    /// # Errors
    ///
    /// As [`Self::compile_unprepared_tmp_bytes`].
    pub fn compile_ci_unprepared_tmp_bytes<BCI>(
        module: &Module<BE>,
        ci: &Module<BCI>,
        plan: &BootstrappingPlan,
    ) -> Result<(usize, usize)>
    where
        BCI: Backend,
        Module<BE>: CKKSDFTMatrixOps<BE, F> + CKKSEncodingOps<BE, F>,
        Module<BCI>: CKKSEncodingOps<BCI, F>,
    {
        let n = ci.n();
        let plan = plan.with_dft_formats_resolved_at(ci.log_n());
        Ok((
            Self::compile_unprepared_tmp_bytes(module, &plan)?
                .max(<Module<BE> as CKKSEncodingHostOps<BE, F>>::ckks_reim_tmp_bytes(module, n)),
            <Module<BCI> as CKKSEncodingHostOps<BCI, F>>::ckks_reim_tmp_bytes(ci, n / 2),
        ))
    }

    /// First stage of [`Self::compile_ci`]: generates the DFT matrices and encodes
    /// EvalMod and the conversions, without preparing the matrices.
    pub fn compile_ci_unprepared<BCI>(
        module: &Module<BE>,
        ci: &Module<BCI>,
        base2k: Base2K,
        plan: &BootstrappingPlan,
        scratch: &mut ScratchArena<'_, BE>,
        scratch_ci: &mut ScratchArena<'_, BCI>,
    ) -> Result<BootstrappingContext<BE, F>>
    where
        BCI: Backend<Ring = ConjugateInvariant, OwnedBuf = BE::OwnedBuf, ZnxWord = BE::ZnxWord>,
        Module<BE>: CKKSDFTMatrixOps<BE, F> + CKKSModuleAlloc<BE> + CKKSEncodingOps<BE, F>,
        Module<BCI>: CKKSModuleAlloc<BCI> + CKKSEncodingOps<BCI, F>,
    {
        let EvalModRing::ConjugateInvariant { pre, post } = plan.eval_mod_ring else {
            anyhow::bail!("compile_ci takes a plan with EvalMod on the conjugate-invariant ring");
        };
        ensure!(
            module.n() >= 2 * ci.n(),
            "compile_ci takes a module of degree at least 2N and a CI module of degree N"
        );
        // `compile_unprepared` resolves the DFT formats at the degree of `module`, at least
        // `2N`, where every slot of degree `N` would be a sparse plan's: resolve them at
        // the bootstrap's degree first.
        let plan = plan.with_dft_formats_resolved_at(ci.log_n());
        let mut context = Self::compile_unprepared(module, base2k, &plan, scratch)?;
        // Slot `j` of degree `N` is slot `j` and slot `j + N/2` of degree `2N`.
        let half = ci.n() / 2;
        let mut pair = |modulus: f64, second: f64, log_delta: usize| {
            let (mut re, mut im) = (vec![F::zero(); 2 * half], vec![F::zero(); 2 * half]);
            re[..half].fill(F::from_f64(modulus).unwrap());
            im[half..].fill(F::from_f64(modulus * second).unwrap());
            // Values of modulus one take two bits above the scale, of modulus two three.
            let headroom = 2 + usize::from(modulus > 1.0);
            let mut pt = module.ckks_pt_vec_alloc(2 * half, base2k, (log_delta + headroom).into());
            pt.set_meta(GLWEPlaintextMeta {
                scale: Scale::Log(log_delta),
                slots: SlotsKind::Complex,
                log_sparsity: 0,
            });
            module.ckks_encode_reim_into(&mut pt, &re, &im, scratch).map(|_| pt)
        };
        // `v`, a product since the plan refuses to fold it, also carries the factor 2
        // of the conjugate pair, which the even coefficients of degree `2N` keep once
        // back at degree `N`.
        let post = pair(2.0, 1.0, post.consumed_bits())?;
        let pre = match pre {
            CIConversion::Product { log_delta } => CIPre::Product(pair(1.0, -1.0, log_delta)?),
            CIConversion::Folded => {
                // `u` is `1` at the roots `ω_j = ζ_{4N}^{5^j}` of slots `j` and `−i` at
                // `−ω_j`, slots `j + N/2`: `u_e = (1 − i)/2` and `u_o = (1 + i)/(2ω_j)`
                // on slot `j` of degree `N`, at the root `ω_j²`.
                let order = 4 * ci.n() as u64;
                let one_half = F::from_f64(0.5).unwrap();
                let slot_scales = |re: Vec<F>, im: Vec<F>| {
                    let mut scales = ComplexDiagonals::new(Diagonals::new(half), Diagonals::new(half));
                    scales.re.set(0, re);
                    scales.im.set(0, im);
                    scales
                };
                let (mut odd_re, mut odd_im) = (Vec::with_capacity(half), Vec::with_capacity(half));
                let mut pow5 = 1u64;
                for _ in 0..half {
                    // (1 + i)/2 times the conjugate root `1/ω_j`.
                    let (cos, sin) = F::ckks_root_of_unity(order - pow5, order.trailing_zeros());
                    odd_re.push((cos - sin) * one_half);
                    odd_im.push((cos + sin) * one_half);
                    pow5 = pow5 * 5 % order;
                }
                let scales = [slot_scales(vec![one_half], vec![-one_half]), slot_scales(odd_re, odd_im)];
                let dft = &mut context.coeffs_to_slots.inner;
                dft.factors.pop();
                CIPre::Folded(crate::layouts::dft::ckks_scaled_last_factors::<BE, F>(
                    module,
                    base2k,
                    &dft.plan,
                    ci.log_n(),
                    &scales,
                    scratch,
                )?)
            }
        };
        context.eval_mod_ci = Some(EvalModCI {
            eval_mod: compile_eval_mod(base2k, plan.eval_mod, ci, scratch_ci)?,
            pre,
            post,
        });
        Ok(context)
    }

    /// Scratch of [`Self::compile_unprepared`] for `plan`: the DFT matrices and EvalMod are built one after the other on
    /// the same scratch, so the largest of their queries, [`CKKSDFTMatrixOps::ckks_new_dft_matrix_tmp_bytes`] of each
    /// matrix and the slot encoding EvalMod stages its coefficients through.
    ///
    /// # Errors
    ///
    /// The plan-versus-ring error of [`CKKSDFTMatrixOps::ckks_new_dft_matrix`].
    pub fn compile_unprepared_tmp_bytes(module: &Module<BE>, plan: &BootstrappingPlan) -> Result<usize>
    where
        Module<BE>: CKKSDFTMatrixOps<BE, F> + CKKSEncodingOps<BE, F>,
    {
        let mut bytes = module
            .ckks_new_dft_matrix_tmp_bytes::<Encode, Split>(&plan.coeffs_to_slots)?
            .max(module.ckks_new_dft_matrix_tmp_bytes::<Decode, Split>(&plan.slots_to_coeffs)?);
        if let Some(bypass) = plan.coeffs_to_slots_bypass() {
            bytes = bytes.max(module.ckks_new_dft_matrix_tmp_bytes::<Encode, Split>(bypass)?);
        }
        if plan.eval_mod_ring == EvalModRing::Standard {
            bytes = bytes.max(<Module<BE> as CKKSEncodingHostOps<BE, F>>::ckks_reim_tmp_bytes(
                module,
                module.n() / 2,
            ));
        }
        Ok(bytes)
    }

    /// First stage of [`Self::compile`]: generates the DFT matrices and encodes
    /// EvalMod, without preparing the matrices.
    pub fn compile_unprepared(
        module: &Module<BE>,
        base2k: Base2K,
        plan: &BootstrappingPlan,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<BootstrappingContext<BE, F>>
    where
        Module<BE>: CKKSDFTMatrixOps<BE, F> + CKKSModuleAlloc<BE> + CKKSEncodingOps<BE, F>,
    {
        let mut coeffs_to_slots = module.ckks_new_dft_matrix::<Encode, Split>(base2k, &plan.coeffs_to_slots, scratch)?;
        // A first factor left to the modulus raise is evaluated from its plaintexts,
        // not by the matrix.
        let coeffs_to_slots_first_factor = plan
            .coeffs_to_slots
            .first_factor_on_raise()
            .then(|| coeffs_to_slots.inner.factors.remove(0));
        let slots_to_coeffs = module.ckks_new_dft_matrix::<Decode, Split>(base2k, &plan.slots_to_coeffs, scratch)?;
        // EvalMod on the conjugate-invariant ring is encoded by `compile_ci`, on its module.
        let eval_mod = match plan.eval_mod_ring {
            EvalModRing::Standard => Some(compile_eval_mod::<BE, F>(base2k, plan.eval_mod, module, scratch)?),
            EvalModRing::ConjugateInvariant { .. } => None,
        };
        let coeffs_to_slots_bypass = match plan.coeffs_to_slots_bypass() {
            Some(bypass) => Some(module.ckks_new_dft_matrix::<Encode, Split>(base2k, bypass, scratch)?),
            None => None,
        };

        Ok(BootstrappingContext {
            plan: plan.clone(),
            c2s_guard_bits: plan.c2s_guard_bits,
            functional_message_modulus: plan.functional_message_modulus,
            coeffs_to_slots,
            coeffs_to_slots_first_factor,
            coeffs_to_slots_bypass,
            slots_to_coeffs,
            eval_mod_plan: plan.eval_mod,
            eval_mod,
            eval_mod_ring: plan.eval_mod_ring,
            eval_mod_ci: None,
            pipeline: plan.pipeline,
            sparse_secret_hamming_weight: plan.sparse_secret_hamming_weight(),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        CoeffsMeta,
        layouts::{DFTOutputFormat, eval_mod::EvalModType},
        polynomial::SplitStrategy,
    };

    fn dft(kind: DFTType) -> DFTPlan {
        DFTPlan::new(
            kind,
            vec![(1, 1)],
            DFTOutputFormat::SplitRealAndImag,
            CoeffsMeta::from_delta_budget(8, 2),
        )
        .unwrap()
    }

    #[test]
    fn functional_bootstrapping_normalization_is_in_s2c() {
        let module = Module::<poulpy_hal::layouts::HostBytesBackend>::new(16);
        let lut = |p: usize| {
            EncodedLut::general(
                &module,
                &vec![1.0f64; p],
                32usize.into(),
                CoeffsMeta::from_delta_budget(55, 32),
                SplitStrategy::MinDepth,
            )
            .unwrap()
        };
        let lut5 = lut(5);
        let lut7 = lut(7);
        let base = plan(BootstrappingPipeline::S2CFirst, BootstrappingTechniques::default(), 16).unwrap();
        let configured = base.clone().with_functional_bootstrap(&lut5).unwrap();
        assert_eq!(configured.slots_to_coeffs().scaling(), Some(4.0 / 5.0));
        assert_eq!(configured.pre_mod_up_consumed_bits(), base.pre_mod_up_consumed_bits());
        assert_eq!(configured.post_mod_up_consumed_bits(), base.post_mod_up_consumed_bits());
        assert!(configured.functional_bootstrap_k(100, 40, &lut7).is_err());
        let configured = configured.with_functional_bootstrap(&lut5).unwrap();
        assert_eq!(configured.slots_to_coeffs().scaling(), Some(4.0 / 5.0));
        let configured = configured.with_functional_bootstrap(&lut7).unwrap();
        assert_eq!(configured.functional_message_modulus, Some(7));
        assert_eq!(configured.slots_to_coeffs().scaling(), Some(4.0 / 7.0));
        assert!(
            plan(BootstrappingPipeline::C2SFirst, BootstrappingTechniques::default(), 16)
                .unwrap()
                .with_functional_bootstrap(&lut5)
                .is_err()
        );
    }

    #[test]
    fn c2s_guard_bits_preserve_input_width_and_charge_raised_width() {
        let input_log_delta = 8;
        let plain = plan(BootstrappingPipeline::S2CFirst, BootstrappingTechniques::default(), 16).unwrap();
        let guarded = plain.clone().with_c2s_guard_bits(6).unwrap();
        assert_eq!(guarded.input_k(48), plain.input_k(48));
        assert_eq!(
            guarded.bootstrap_k(720, input_log_delta),
            plain.bootstrap_k(720, input_log_delta) + 6
        );
        assert_eq!(guarded.consumed_bits(), plain.consumed_bits() + 6);
        assert_eq!(
            guarded.with_c2s_guard_bits(3).unwrap().bootstrap_k(720, input_log_delta),
            plain.bootstrap_k(720, input_log_delta) + 3
        );
        assert!(plain.with_c2s_guard_bits(usize::MAX).is_err());
        assert!(
            plan(BootstrappingPipeline::C2SFirst, BootstrappingTechniques::default(), 16)
                .unwrap()
                .with_c2s_guard_bits(6)
                .is_err()
        );
    }

    fn eval_mod(f_mod_interval: usize) -> EvalModPlan {
        EvalModPlan {
            eval_mod_type: EvalModType::CosHK,
            log_msg_ratio: 2,
            f_mod_degree: 2 * (f_mod_interval - 1),
            f_mod_interval,
            f_mod_log_interval_reduction: 1,
            f_mod_inv_degree: None,
            scaling: None,
            split_strategy: SplitStrategy::MinDepth,
            coeffs_meta: CoeffsMeta::from_delta_budget(8, 2),
            f_mod_log_delta: 10,
        }
    }

    fn plan(
        pipeline: BootstrappingPipeline,
        techniques: BootstrappingTechniques,
        f_mod_interval: usize,
    ) -> Result<BootstrappingPlan> {
        BootstrappingPlan::new(
            pipeline,
            techniques,
            dft(DFTType::Encode),
            eval_mod(f_mod_interval),
            dft(DFTType::Decode),
        )
    }

    #[test]
    fn conjugate_invariant_ring_refuses_what_does_not_fold() {
        let ring = |pre, post| EvalModRing::ConjugateInvariant { pre, post };
        let s2c = plan(BootstrappingPipeline::S2CFirst, BootstrappingTechniques::default(), 16).unwrap();
        let err = s2c
            .clone()
            .with_eval_mod_ring(ring(CIConversion::Product { log_delta: 46 }, CIConversion::Folded))
            .unwrap_err();
        assert!(
            err.to_string().contains("the product by v after EvalMod does not fold"),
            "{err}"
        );
        // A CoeffsToSlots of one factor, left to the raise, leaves `u` no factor to fold into.
        let raised_only = BootstrappingPlan::new(
            BootstrappingPipeline::S2CFirst,
            BootstrappingTechniques::default(),
            dft(DFTType::Encode).with_first_factor_on_raise(),
            eval_mod(16),
            dft(DFTType::Decode),
        )
        .unwrap();
        let folded = ring(CIConversion::Folded, CIConversion::Product { log_delta: 38 });
        let err = raised_only.with_eval_mod_ring(folded).unwrap_err();
        assert!(
            err.to_string()
                .contains("folding u takes a CoeffsToSlots factor the matrix evaluates"),
            "{err}"
        );
        assert!(s2c.with_eval_mod_ring(folded).is_ok());
    }

    #[test]
    fn recipe_records_sparse_secret_weight() {
        let plan = plan(
            BootstrappingPipeline::C2SFirst,
            BootstrappingTechniques {
                sparse_secret_encapsulation: Some(SparseSecretEncapsulation { hamming_weight: 32 }),
                eval_round_plus: None,
            },
            16,
        )
        .unwrap();

        assert_eq!(plan.sparse_secret_hamming_weight(), Some(32));
    }

    #[test]
    fn recipe_accepts_s2c_first_pipeline() {
        let plan = plan(BootstrappingPipeline::S2CFirst, BootstrappingTechniques::default(), 16).unwrap();
        assert_eq!(plan.pipeline(), BootstrappingPipeline::S2CFirst);
    }

    #[test]
    fn recipe_owns_eval_mod_input_scaling() {
        let c2s = dft(DFTType::Encode).with_scaling(7.0).unwrap();
        let plan = BootstrappingPlan::new(
            BootstrappingPipeline::C2SFirst,
            BootstrappingTechniques::default(),
            c2s,
            eval_mod(16),
            dft(DFTType::Decode),
        )
        .unwrap();
        assert_eq!(plan.coeffs_to_slots().scaling(), Some(1.0 / 16.0));

        let mut centered = eval_mod(16);
        centered.eval_mod_type = EvalModType::CosHKEven;
        let plan = BootstrappingPlan::new(
            BootstrappingPipeline::S2CFirst,
            BootstrappingTechniques::default(),
            dft(DFTType::Encode),
            centered,
            dft(DFTType::Decode),
        )
        .unwrap();
        assert_eq!(plan.coeffs_to_slots().scaling(), Some(1.0 / 16.0));
    }

    #[test]
    fn recipe_accounts_for_pipeline_order() {
        let input_log_delta = 8;
        let output_k = 30;
        let c2s = plan(BootstrappingPipeline::C2SFirst, BootstrappingTechniques::default(), 16).unwrap();
        assert_eq!(c2s.pre_mod_up_consumed_bits(), 0);
        assert_eq!(c2s.post_mod_up_consumed_bits(), c2s.consumed_bits());
        assert_eq!(c2s.input_k(20), 20);
        let scale_restoration_bits = c2s.eval_mod().f_mod_log_delta - input_log_delta;
        assert_eq!(
            c2s.bootstrap_k(output_k, input_log_delta),
            output_k + c2s.consumed_bits() + scale_restoration_bits
        );

        let s2c = plan(BootstrappingPipeline::S2CFirst, BootstrappingTechniques::default(), 16).unwrap();
        assert_eq!(s2c.pre_mod_up_consumed_bits(), s2c.slots_to_coeffs().consumed_bits());
        assert_eq!(
            s2c.post_mod_up_consumed_bits(),
            s2c.coeffs_to_slots().consumed_bits() + s2c.eval_mod().consumed_bits()
        );
        assert_eq!(s2c.input_k(20), 20 + s2c.slots_to_coeffs().consumed_bits());
        assert_eq!(
            s2c.bootstrap_k(output_k, input_log_delta),
            output_k + s2c.post_mod_up_consumed_bits()
        );
    }

    #[test]
    fn recipe_accepts_eval_round_plus_with_s2c_first() {
        let plan = plan(
            BootstrappingPipeline::S2CFirst,
            BootstrappingTechniques {
                sparse_secret_encapsulation: None,
                eval_round_plus: Some(EvalRoundPlus {
                    coeffs_to_slots_bypass: dft(DFTType::Encode),
                }),
            },
            16,
        )
        .unwrap();
        assert!(plan.coeffs_to_slots_bypass().is_some());
    }

    #[test]
    fn recipe_accounts_for_eval_round_plus_bypass() {
        let bypass = DFTPlan::new(
            DFTType::Encode,
            vec![(1, 1); 10],
            DFTOutputFormat::SplitRealAndImag,
            CoeffsMeta::from_delta_budget(8, 2),
        )
        .unwrap();
        let bypass_cost = bypass.consumed_bits();

        for pipeline in [BootstrappingPipeline::C2SFirst, BootstrappingPipeline::S2CFirst] {
            let plan = plan(
                pipeline,
                BootstrappingTechniques {
                    sparse_secret_encapsulation: None,
                    eval_round_plus: Some(EvalRoundPlus {
                        coeffs_to_slots_bypass: bypass.clone(),
                    }),
                },
                16,
            )
            .unwrap();

            let trailing_s2c = match pipeline {
                BootstrappingPipeline::C2SFirst => plan.slots_to_coeffs().consumed_bits(),
                BootstrappingPipeline::S2CFirst => 0,
            };
            assert!(bypass_cost > plan.coeffs_to_slots().consumed_bits() + plan.eval_mod().consumed_bits());
            assert_eq!(plan.post_mod_up_consumed_bits(), bypass_cost + trailing_s2c);
        }
    }

    #[test]
    fn recipe_rejects_zero_sparse_secret_weight() {
        let err = plan(
            BootstrappingPipeline::C2SFirst,
            BootstrappingTechniques {
                sparse_secret_encapsulation: Some(SparseSecretEncapsulation { hamming_weight: 0 }),
                eval_round_plus: None,
            },
            16,
        )
        .unwrap_err();
        assert!(err.to_string().contains("hamming_weight"));
    }

    #[test]
    fn eval_round_plus_requires_power_of_two_interval() {
        let err = plan(
            BootstrappingPipeline::C2SFirst,
            BootstrappingTechniques {
                sparse_secret_encapsulation: None,
                eval_round_plus: Some(EvalRoundPlus {
                    coeffs_to_slots_bypass: dft(DFTType::Encode),
                }),
            },
            3,
        )
        .unwrap_err();
        assert!(err.to_string().contains("power-of-two"));
    }
}
