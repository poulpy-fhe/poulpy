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
//! composes the selected pipeline from a compiled [`BootstrappingContext`] and
//! a prepared key set. The plan remains a plain parameter bundle, so callers
//! can also compose the stages manually from the respective op traits.
//!
//! [`BootstrappingContext`] is the *compiled* form of a [`BootstrappingPlan`]:
//! the prepared, backend-resident homomorphic DFT matrices and the encoded,
//! uploaded EvalMod, built once and reused across bootstraps.

use std::collections::BTreeMap;

use crate::layouts::CKKSPlaintextOwned;
use anyhow::{Result, ensure};
use poulpy_core::{
    layouts::{Base2K, GLWEToBackendRef, LWEInfos, LinearTransformation},
    reference::linear_transformation::DiagonalProd,
};
use poulpy_hal::layouts::{Backend, Module, ScratchArena};

use crate::{
    CKKSCtBounds, CKKSInfos,
    api::{CKKSDFTMatrixOps, CKKSDFTOps, CKKSEncodingOps, CKKSEncodingScalar},
    layouts::{
        CKKSModuleAlloc, DFTMatrix, DFTMatrixPrepared, DFTPlan, DFTType, Decode, Encode, EncodedLut, EvalMod, EvalModPlan, Split,
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
    pub(crate) functional_message_modulus: Option<usize>,
    /// The ring EvalMod runs on ([`Self::with_eval_mod_on_ci_ring`]).
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
    pub(crate) fn key_widths(&self) -> KeyWidths<'_> {
        KeyWidths {
            pipeline: self.pipeline,
            c2s_guard_bits: self.c2s_guard_bits,
            eval_mod_ring: self.eval_mod_ring,
            coeffs_to_slots: &self.coeffs_to_slots,
            slots_to_coeffs: &self.slots_to_coeffs,
            eval_mod: &self.eval_mod,
        }
    }

    /// Validated constructor for a complete bootstrapping recipe.
    ///
    /// `coeffs_to_slots` must be a
    /// [`DFTType::Encode`] plan and `slots_to_coeffs` a [`DFTType::Decode`]
    /// plan (errors [`InvalidPlan`](crate::CKKSCompositionError::InvalidPlan)
    /// otherwise) — a swapped direction would make [`Self::galois_elements`]
    /// derive rotation keys for a transform that
    /// [`BootstrappingContext::compile`] never builds. The constructor also
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
            "functional bootstrapping runs EvalMod on the ring of the bootstrap"
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

    /// Budget consumed before ModUp.
    pub fn pre_mod_up_consumed_bits(&self) -> usize {
        match self.pipeline {
            BootstrappingPipeline::C2SFirst => 0,
            BootstrappingPipeline::S2CFirst => self.slots_to_coeffs.consumed_bits(),
        }
    }

    /// Runs EvalMod once on the conjugate-invariant ring, through the conversions
    /// of `conversion` ([`crate::layouts::bootstrapping_ci`]). The plan is
    /// S2C-first, with a real EvalMod family and neither EvalRound+ nor a LUT.
    pub fn with_eval_mod_on_ci_ring(mut self, conversion: EvalModCIPlan) -> Result<Self> {
        ensure!(
            self.pipeline == BootstrappingPipeline::S2CFirst,
            "EvalMod on the conjugate-invariant ring requires an S2C-first plan"
        );
        ensure!(
            self.coeffs_to_slots_bypass().is_none(),
            "EvalMod on the conjugate-invariant ring does not take the EvalRound+ bypass"
        );
        ensure!(
            self.functional_message_modulus.is_none(),
            "EvalMod on the conjugate-invariant ring does not run functional bootstrapping"
        );
        ensure!(
            self.eval_mod.eval_mod_type != EvalModType::ExpCmplx,
            "EvalMod on the conjugate-invariant ring needs a real EvalMod family"
        );
        ensure!(
            conversion.pre_log_delta > 0 && conversion.post_log_delta > 0,
            "the conversion scales must be positive"
        );
        self.eval_mod_ring = EvalModRing::ConjugateInvariant(conversion);
        Ok(self)
    }

    /// The ring EvalMod runs on.
    pub fn eval_mod_ring(&self) -> EvalModRing {
        self.eval_mod_ring
    }

    /// Budget consumed after ModUp.
    ///
    /// EvalRound+ evaluates its bypass in parallel with the low-precision
    /// CoeffsToSlots + EvalMod branch, so the wider of the two branch costs is
    /// charged before any trailing SlotsToCoeffs.
    /// S2C-first also charges the CoeffsToSlots guard bits and any conversions.
    pub fn post_mod_up_consumed_bits(&self) -> usize {
        let c2s_eval_mod = self.coeffs_to_slots.consumed_bits() + self.eval_mod.consumed_bits();
        let eval_round = self
            .coeffs_to_slots_bypass()
            .map_or(c2s_eval_mod, |bypass| c2s_eval_mod.max(bypass.consumed_bits()));
        match self.pipeline {
            BootstrappingPipeline::C2SFirst => eval_round + self.slots_to_coeffs.consumed_bits(),
            BootstrappingPipeline::S2CFirst => eval_round + self.c2s_guard_bits + self.eval_mod_ring.consumed_bits(),
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
    /// ([`DFTPlan::galois_elements`]), for a ring of degree `2^log_n` and the given
    /// `cyclotomic_order`. The split forward transform's conjugation key is
    /// separate (generate it from Galois element `−1`) and not included here.
    pub fn galois_elements(&self, log_n: usize, cyclotomic_order: i64) -> Vec<i64> {
        let mut set: std::collections::BTreeSet<i64> = std::collections::BTreeSet::new();
        set.extend(self.coeffs_to_slots.galois_elements(log_n, cyclotomic_order));
        if let Some(bypass) = self.coeffs_to_slots_bypass() {
            set.extend(bypass.galois_elements(log_n, cyclotomic_order));
        }
        set.extend(self.slots_to_coeffs.galois_elements(log_n, cyclotomic_order));
        set.into_iter().collect()
    }
}

/// The ring EvalMod runs on, after CoeffsToSlots.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum EvalModRing {
    /// The ring of the bootstrap, once per half of the slots.
    Standard,
    /// The conjugate-invariant ring of the same degree, once, between two
    /// conversions ([`crate::layouts::bootstrapping_ci`]).
    ConjugateInvariant(EvalModCIPlan),
}

impl EvalModRing {
    /// Budget the conversions consume.
    pub fn consumed_bits(&self) -> usize {
        match self {
            Self::Standard => 0,
            Self::ConjugateInvariant(conversion) => conversion.pre_log_delta + conversion.post_log_delta,
        }
    }
}

/// Scales, in bits, of the plaintext products of the conversions before and
/// after EvalMod on the conjugate-invariant ring, each spent from the budget.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct EvalModCIPlan {
    /// Scale of `u`, which pairs each slot of degree `N` into two real CI slots.
    pub pre_log_delta: usize,
    /// Scale of `v`, which pairs the reduced CI slots back.
    pub post_log_delta: usize,
}

/// Compiled [`BootstrappingPlan`]: the resident DFT matrices and encoded EvalMod
/// the bootstrapping pipeline evaluates.
///
/// Built once by [`Self::compile`] and reused across bootstraps. The
/// `SplitRealAndImag` format is used for both transforms so the real and
/// imaginary coefficient halves can be reduced independently by EvalMod.
///
/// `BEM` is the backend EvalMod runs on: `BE`, or the CI backend for
/// [`EvalModRing::ConjugateInvariant`] ([`Self::compile_ci`](crate::layouts::bootstrapping_ci)).
pub struct BootstrappingContext<BE: Backend, F, BEM: Backend = BE> {
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

    /// Encoded, backend-resident EvalMod (`x mod 1`), on the backend it runs on.
    eval_mod: EvalMod<F, CKKSPlaintextOwned<BEM>>,

    /// The conversion plaintexts `u` and `v` of degree `2N`, on the CI ring only.
    pub(crate) conversions: Option<(CKKSPlaintextOwned<BE>, CKKSPlaintextOwned<BE>)>,

    /// The ring EvalMod runs on.
    eval_mod_ring: EvalModRing,

    /// Selected ModUp/EvalMod pipeline.
    pipeline: BootstrappingPipeline,

    /// Ephemeral sparse-secret weight required by the recipe, if enabled.
    sparse_secret_hamming_weight: Option<usize>,
}

/// The widths, in bits, at which a bootstrap uses its keys, given the width it
/// runs at and its input.
///
/// The presets size their keys from them, and
/// [`ckks_bootstrap`](crate::api::CKKSBootstrappingOps::ckks_bootstrap) checks
/// every key against them before it starts. Not for a recipe with the
/// EvalRound+ bypass, which runs at widths of its own.
pub(crate) struct KeyWidths<'a> {
    pipeline: BootstrappingPipeline,
    c2s_guard_bits: usize,
    eval_mod_ring: EvalModRing,
    coeffs_to_slots: &'a DFTPlan,
    slots_to_coeffs: &'a DFTPlan,
    eval_mod: &'a EvalModPlan,
}

impl KeyWidths<'_> {
    /// Widest ciphertext, in bits, each rotation key switches, by Galois element.
    ///
    /// Factor `i` of a transform runs on a ciphertext `i` factor scales below the
    /// width the transform starts at: the raised width `bootstrap_k` for
    /// CoeffsToSlots, and for SlotsToCoeffs the input width `input_k`
    /// (S2C-first) or what EvalMod leaves (C2S-first). The conjugation follows
    /// CoeffsToSlots.
    pub(crate) fn rotation_key_inputs(&self, log_n: usize, bootstrap_k: usize, input_k: usize) -> BTreeMap<i64, usize> {
        let cyclotomic_order = 1i64 << (log_n + 1);
        let after_c2s = bootstrap_k - self.coeffs_to_slots.consumed_bits();
        let s2c_k = match self.pipeline {
            BootstrappingPipeline::S2CFirst => input_k,
            BootstrappingPipeline::C2SFirst => after_c2s - self.eval_mod.consumed_bits(),
        };

        let mut inputs: BTreeMap<i64, usize> = BTreeMap::new();
        let mut widen = |p: i64, k: usize| {
            let widest = inputs.entry(p).or_insert(k);
            *widest = (*widest).max(k);
        };
        for (dft, start) in [(self.coeffs_to_slots, bootstrap_k), (self.slots_to_coeffs, s2c_k)] {
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
        // The split CoeffsToSlots conjugates; on the conjugate-invariant ring it is not split.
        if self.eval_mod_ring == EvalModRing::Standard {
            widen(-1, after_c2s);
        }
        inputs
    }

    /// Widths of the traced and of the embedded ciphertext of a bootstrap at
    /// `bootstrap_k` bits, when EvalMod runs on the conjugate-invariant ring.
    pub(crate) fn conversions(&self, bootstrap_k: usize) -> Option<(usize, usize)> {
        let EvalModRing::ConjugateInvariant(conversion) = self.eval_mod_ring else {
            return None;
        };
        // CoeffsToSlots and the product by `u` leave the traced width; the C2S guard
        // bits are only dropped from the scale at the output.
        let traced = bootstrap_k - self.coeffs_to_slots.consumed_bits() - conversion.pre_log_delta;
        Some((traced, traced - self.eval_mod.consumed_bits()))
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

    /// Widest ciphertext EvalMod relinearizes, in bits, for an EvalMod input of
    /// `eval_mod_k` bits and a bootstrap input of scale `2^input_log_delta`.
    ///
    /// EvalMod relabels its input at its own scale and keeps its budget, so its
    /// widest product is at that budget plus `f_mod_log_delta`. The budget is what
    /// the input holds above the scale the raise labels: EvalMod's own scale
    /// (C2S-first), or the input scale, the message ratio and the guard bits
    /// (S2C-first).
    pub(crate) fn eval_mod_relinearization_k(&self, eval_mod_k: usize, input_log_delta: usize) -> usize {
        let f_mod_log_delta = self.eval_mod.f_mod_log_delta;
        let raised_log_delta = match self.pipeline {
            BootstrappingPipeline::C2SFirst => f_mod_log_delta,
            BootstrappingPipeline::S2CFirst => input_log_delta + self.eval_mod.log_msg_ratio + self.c2s_guard_bits,
        };
        eval_mod_k - raised_log_delta + f_mod_log_delta
    }
}

impl<BE: Backend, F, BEM: Backend> BootstrappingContext<BE, F, BEM> {
    /// The stages that set the widths at which a bootstrap uses its keys.
    pub(crate) fn key_widths(&self) -> KeyWidths<'_> {
        KeyWidths {
            pipeline: self.pipeline,
            c2s_guard_bits: self.c2s_guard_bits,
            eval_mod_ring: self.eval_mod_ring,
            coeffs_to_slots: self.coeffs_to_slots.plan(),
            slots_to_coeffs: self.slots_to_coeffs.plan(),
            eval_mod: &self.eval_mod.plan,
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

    /// Encoded, backend-resident EvalMod circuit.
    pub fn eval_mod(&self) -> &EvalMod<F, CKKSPlaintextOwned<BEM>> {
        &self.eval_mod
    }

    /// The ring EvalMod runs on.
    pub fn eval_mod_ring(&self) -> EvalModRing {
        self.eval_mod_ring
    }

    /// The conversion plaintexts `u` and `v` of degree `2N`, on the CI ring only.
    pub fn conversions(&self) -> Option<(&CKKSPlaintextOwned<BE>, &CKKSPlaintextOwned<BE>)> {
        self.conversions.as_ref().map(|(pre, post)| (pre, post))
    }

    /// Ephemeral sparse-secret weight required by the compiled recipe.
    pub fn sparse_secret_hamming_weight(&self) -> Option<usize> {
        self.sparse_secret_hamming_weight
    }
}

/// Unprepared stage of a [`BootstrappingContext`]: the generated DFT matrices and
/// the encoded EvalMod, before the matrices are prepared.
///
/// Built by [`BootstrappingContext::compile_unprepared`]. Preparation is a
/// separate step so that its scratch can be sized with
/// [`Self::prepare_tmp_bytes`], which follows the backend's selected
/// preparation budget.
pub struct BootstrappingContextUnprepared<BE: Backend, F, BEM: Backend = BE> {
    c2s_guard_bits: usize,
    functional_message_modulus: Option<usize>,
    coeffs_to_slots: DFTMatrix<BE, Encode, Split>,
    coeffs_to_slots_first_factor: Option<LinearTransformation<CKKSPlaintextOwned<BE>>>,
    coeffs_to_slots_bypass: Option<DFTMatrix<BE, Encode, Split>>,
    slots_to_coeffs: DFTMatrix<BE, Decode, Split>,
    pub(crate) eval_mod: EvalMod<F, CKKSPlaintextOwned<BEM>>,
    pub(crate) conversions: Option<(CKKSPlaintextOwned<BE>, CKKSPlaintextOwned<BE>)>,
    eval_mod_ring: EvalModRing,
    pipeline: BootstrappingPipeline,
    sparse_secret_hamming_weight: Option<usize>,
}

impl<BE: Backend, F, BEM: Backend> BootstrappingContextUnprepared<BE, F, BEM> {
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
        bytes
    }

    /// Prepares the matrices into the resident form the pipeline evaluates.
    /// Each unprepared matrix is released as soon as it has been prepared.
    pub fn prepare(self, module: &Module<BE>, scratch: &mut ScratchArena<'_, BE>) -> BootstrappingContext<BE, F, BEM>
    where
        Module<BE>: CKKSDFTOps<BE>,
        CKKSPlaintextOwned<BE>: GLWEToBackendRef<BE> + CKKSCtBounds + DiagonalProd<BE>,
    {
        let Self {
            c2s_guard_bits,
            functional_message_modulus,
            coeffs_to_slots,
            coeffs_to_slots_first_factor,
            coeffs_to_slots_bypass,
            slots_to_coeffs,
            eval_mod,
            conversions,
            eval_mod_ring,
            pipeline,
            sparse_secret_hamming_weight,
        } = self;
        let coeffs_to_slots = {
            let unprepared = coeffs_to_slots;
            module.ckks_prepare_dft_matrix(&unprepared, scratch)
        };
        let slots_to_coeffs = {
            let unprepared = slots_to_coeffs;
            module.ckks_prepare_dft_matrix(&unprepared, scratch)
        };
        let coeffs_to_slots_bypass =
            coeffs_to_slots_bypass.map(|unprepared| module.ckks_prepare_dft_matrix(&unprepared, scratch));
        BootstrappingContext {
            c2s_guard_bits,
            functional_message_modulus,
            coeffs_to_slots,
            coeffs_to_slots_first_factor,
            coeffs_to_slots_bypass,
            slots_to_coeffs,
            eval_mod,
            conversions,
            eval_mod_ring,
            pipeline,
            sparse_secret_hamming_weight,
        }
    }
}

impl<BE: Backend, F> BootstrappingContext<BE, F>
where
    F: CKKSEncodingScalar,
{
    /// Compiles `plan` directly into backend-resident matrices and EvalMod.
    ///
    /// Each stage carries its own coefficient metadata (`DFTPlan::meta` /
    /// `EvalModPlan::meta`), including the CoeffsToSlots scaling resolved by
    /// [`BootstrappingPlan::new`].
    ///
    /// `scratch` serves both generation and preparation, so it must also cover
    /// [`BootstrappingContextUnprepared::prepare_tmp_bytes`]. That value is only
    /// known once the matrices exist: use [`Self::compile_unprepared`] followed by
    /// [`BootstrappingContextUnprepared::prepare`] to size preparation exactly.
    pub fn compile(
        module: &Module<BE>,
        base2k: Base2K,
        plan: &BootstrappingPlan,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<Self>
    where
        Module<BE>: CKKSDFTOps<BE> + CKKSDFTMatrixOps<BE, F> + CKKSModuleAlloc<BE> + CKKSEncodingOps<BE, F>,
        CKKSPlaintextOwned<BE>: GLWEToBackendRef<BE> + CKKSCtBounds + DiagonalProd<BE>,
    {
        Ok(Self::compile_unprepared(module, base2k, plan, scratch)?.prepare(module, scratch))
    }

    /// First stage of [`Self::compile`]: generates the DFT matrices and encodes
    /// EvalMod, without preparing the matrices; a plan with EvalMod on the CI ring
    /// takes [`Self::compile_unprepared_ci`](crate::layouts::bootstrapping_ci).
    pub fn compile_unprepared(
        module: &Module<BE>,
        base2k: Base2K,
        plan: &BootstrappingPlan,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<BootstrappingContextUnprepared<BE, F>>
    where
        Module<BE>: CKKSDFTMatrixOps<BE, F> + CKKSModuleAlloc<BE> + CKKSEncodingOps<BE, F>,
    {
        ensure!(
            plan.eval_mod_ring == EvalModRing::Standard,
            "a plan with EvalMod on the conjugate-invariant ring is compiled with its modules"
        );
        let eval_mod = compile_eval_mod::<BE, F>(base2k, plan.eval_mod, module, scratch)?;
        BootstrappingContextUnprepared::with_eval_mod(module, base2k, plan, eval_mod, None, scratch)
    }
}

impl<BE: Backend, F, BEM: Backend> BootstrappingContextUnprepared<BE, F, BEM>
where
    F: CKKSEncodingScalar,
{
    /// The transforms of `plan` around an EvalMod already encoded, and the
    /// conversion plaintexts when it runs on the conjugate-invariant ring.
    pub(crate) fn with_eval_mod(
        module: &Module<BE>,
        base2k: Base2K,
        plan: &BootstrappingPlan,
        eval_mod: EvalMod<F, CKKSPlaintextOwned<BEM>>,
        conversions: Option<(CKKSPlaintextOwned<BE>, CKKSPlaintextOwned<BE>)>,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<Self>
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
        let coeffs_to_slots_bypass = match plan.coeffs_to_slots_bypass() {
            Some(bypass) => Some(module.ckks_new_dft_matrix::<Encode, Split>(base2k, bypass, scratch)?),
            None => None,
        };

        Ok(Self {
            c2s_guard_bits: plan.c2s_guard_bits,
            functional_message_modulus: plan.functional_message_modulus,
            coeffs_to_slots,
            coeffs_to_slots_first_factor,
            coeffs_to_slots_bypass,
            slots_to_coeffs,
            eval_mod,
            conversions,
            eval_mod_ring: plan.eval_mod_ring,
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
