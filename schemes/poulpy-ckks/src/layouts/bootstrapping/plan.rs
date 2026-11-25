//! What a bootstrap runs: the stages, their techniques and what they consume.

use anyhow::{Result, ensure};

use poulpy_core::{Distribution, security::SecretSpec};
use poulpy_hal::layouts::galois_element;

use crate::{
    CKKSCompositionError, CKKSInfos, ReplicatedLayout, SlotLayout,
    layouts::{BootstrappingSecrets, DFTPlan, DFTType, EncodedLut, EvalModPlan, ModRaisePlan, eval_mod::EvalModType},
};

use super::{BootstrappingKeyWidths, CIConversion, EvalModRing, key_widths::StageWidths};

/// Stage ordering of a bootstrap.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BootstrappingPipeline {
    /// The raise, CoeffsToSlots, EvalMod and SlotsToCoeffs. The last transform restores the message ratio and labels
    /// the output, whose scale and width return to the input's.
    C2SFirst,
    /// SlotsToCoeffs below the raise, at scaling `1/2` on a doubled input, on a replicated layout at scaling 1 on a copy
    /// lifted by the S2C guard and working bits, then the raise, CoeffsToSlots and EvalMod, whose output is relabeled at
    /// the input's scale.
    S2CFirst,
}

/// Sparse-secret encapsulation of the raise (<https://eprint.iacr.org/2022/024>): ModUp runs under an ephemeral sparse
/// secret, which bounds the integer part it adds, between a dense-to-sparse and a sparse-to-dense switch. The keys of
/// the bootstrap carry both, as [`encapsulation_keys`](crate::layouts::BootstrappingLinearKeys::encapsulation_keys).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SparseSecretEncapsulation {
    /// Weight of the ephemeral ternary secret.
    pub hamming_weight: usize,
}

/// EvalRound+ (<https://eprint.iacr.org/2024/1379>): a high-precision CoeffsToSlots bypasses EvalMod, and cancels the
/// error of the low-precision transform that feeds it. Takes a power-of-two `f_mod_interval`, and excludes the functional
/// bootstrap, whose LUT step runs no bypass, and [`EvalModType::ExpCmplx`], which holds on one EvalMod.
///
/// [`ckks_bootstrap_eval_mod`](crate::api::CKKSBootstrappingOps::ckks_bootstrap_eval_mod) then returns `EvalMod(x) -
/// K·x`, on every route, with its working label and the guard bits still on, and the composer finishes the step: it
/// runs [`bypass`](crate::layouts::BootstrappingRun::bypass) on a copy of the raised ciphertext taken before CoeffsToSlots, doubles the copy,
/// adds the output of the step, and only then calls [`stamp_eval_mod_output`](crate::layouts::CKKSEvalModInfos::stamp_eval_mod_output).
/// Removing the guard bits earlier lowers `k` and changes the bits of the conjugate-invariant route.
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

fn invalid_plan(reason: &str) -> anyhow::Error {
    crate::CKKSCompositionError::InvalidPlan {
        plan: "BootstrappingPlan",
        reason: reason.to_string(),
    }
    .into()
}

/// End-to-end parameterization of CKKS bootstrapping. Every choice of how a bootstrap runs is a parameter of this plan,
/// and the steps of [`CKKSBootstrappingOps`](crate::api::CKKSBootstrappingOps) follow it.
///
/// Built with [`Self::new`], which validates the stage roles once, so that key provisioning ([`Self::galois_elements`])
/// cannot derive rotations for the wrong transform direction.
///
/// # Parameters
///
/// - `pipeline` ([`BootstrappingPipeline`]): the order of the stages and which one labels the output.
/// - `coeffs_to_slots`, `slots_to_coeffs` ([`DFTPlan`]): the two transforms. [`Self::new`] folds EvalMod's input
///   scaling into CoeffsToSlots, so the caller scales nothing.
/// - `eval_mod` ([`EvalModPlan`]): the `x mod 1` evaluation. It runs at its own scale, a budget-neutral relabel, so it
///   can keep more `ct×ct` precision.
/// - `eval_mod_ring` ([`EvalModRing`], [`Self::with_eval_mod_ring`]): the ring of EvalMod, which takes an S2C-first
///   plan with a real EvalMod and no LUT off the standard ring.
/// - `c2s_guard_bits` ([`Self::with_c2s_guard_bits`], S2C-first): scale bits added at the raise to reduce CoeffsToSlots
///   rounding, charged to the raised width and removed at the output, the working bits included.
/// - `slot_layout` ([`Self::with_slot_layout`]): the layout of the messages, full by default, and on a
///   [replicated layout](SlotLayout::Replicated) the degree of their ciphertexts.
/// - `working_bits` ([`Self::with_working_bits`], replicated): bits under the scale at which SlotsToCoeffs works, which
///   the raise reads as part of its guard bits, and each part of EvalMod's input is rounded.
/// - `s2c_guard_bits` ([`Self::with_s2c_guard_bits`], replicated): scale bits SlotsToCoeffs adds to its input, charged
///   to the raised width and removed at the output.
/// - `techniques` ([`BootstrappingTechniques`]): [`SparseSecretEncapsulation`] around the raise and [`EvalRoundPlus`]
///   around EvalMod. A first CoeffsToSlots factor on the raise excludes EvalRound+.
/// - `functional_message_modulus` ([`Self::with_functional_bootstrap`], S2C-first): turns the bootstrap into a LUT
///   evaluation on integer messages, on the standard ring, without EvalRound+.
/// - `log_gap` ([`Self::with_log_gap`], S2C-first): plaintexts in a subring, whose transforms fill fewer slots and whose
///   raise projects on the subring, on the standard ring, with the first CoeffsToSlots factor on the raise and a sparse
///   secret.
///
/// The raise follows [`Self::mod_raise`].
///
/// # EvalMod
///
/// [`ckks_bootstrap_eval_mod`](crate::api::CKKSBootstrappingOps::ckks_bootstrap_eval_mod) takes its route from the plan
/// and the slots of the message:
///
/// | Plan and message | Route |
/// |---|---|
/// | replicated layout, [packed](EvalModPlan::packed) EvalMod | the real part, one EvalMod on both batches, the second rotated onto the first |
/// | standard ring, real slots, S2C-first, no EvalRound+ | the real part, one EvalMod |
/// | any other plan on the standard ring | the split, one EvalMod per half, on a parallel executor at once, the recombination |
/// | conjugate-invariant ring | the conversion, one EvalMod for all coefficients, the conversion back |
///
/// Real slots read the real part alone because an S2C-first raise leaves only integer multiples of the modulus in the
/// imaginary part, and EvalRound+ needs both halves. A [`BootstrappingRun`] gives the step the slots of the message,
/// and a plain [`EvalMod`] reads those of `a`, has no pipeline and stamps nothing.
///
/// EvalMod spends `consumed = eval_mod.consumed_bits()` of the budget of `a`, and the conversions of the
/// conjugate-invariant ring the scales of their products. The output of a run is:
///
/// - S2C-first: `res.k() = a.k() - c2s_guard_bits - consumed` at the scale, slot kind and sparsity of the message;
/// - on a replicated layout: each part of `a` rounded with shaping `working_bits` under its scale before EvalMod, and
///   the output rounded with shaping to the scale of the message, `res.k() = a.k() - c2s_guard_bits - s2c_guard_bits -
///   consumed`;
/// - C2S-first: `log_delta_out = a.log_delta()`, `log_budget_out = a.log_budget() - consumed`, until SlotsToCoeffs
///   labels it;
/// - under [`EvalRoundPlus`], and on a plain [`EvalMod`]: the same, with the guard bits still on and the working label,
///   slot kind and sparsity being those of the route on a plain [`EvalMod`].
///
/// The split route leaves the canonical flag clear: its recombination is a lazy sum.
///
/// [`EvalModType::ExpCmplx`] holds on the real part alone: the split would add the exponentials of the two halves.
/// [`Self::new`] refuses it on a C2S-first plan or with EvalRound+, and the step on a complex message. It leaves
/// `exp(2πi·m/2^log_msg_ratio)`, which a run labels complex.
/// [`ckks_bootstrap`](crate::api::CKKSBootstrappingOps::ckks_bootstrap) refuses the plan, which refreshes nothing.
///
/// # LUTs
///
/// [`ckks_eval_lut`](crate::api::CKKSBootstrappingOps::ckks_eval_lut) runs no bypass, refuses a plan with one, and
/// removes the guard bits itself, before the exponential. Real slots of an S2C-first plan take the real part of `a` and
/// skip the imaginary half, and any other input is split, each half evaluated and the halves recombined. General LUTs
/// of a batch share the power basis of each half.
///
/// Each LUT spends `bits`, EvalMod's when it needs it and its own series', of the budget of `a`. A run gives an output
/// `k_out = a.k() - c2s_guard_bits - bits`, `log_delta_out = log_delta + log_msg_ratio` from the `log_delta` of the
/// message, and the slot kind and sparsity of the message. A plain [`EvalMod`] reads the slots of `a` and leaves the
/// outputs what the evaluation gives them. The split route leaves the canonical flag clear.
///
/// [`BootstrappingRun`]: crate::layouts::BootstrappingRun
/// [`EvalMod`]: crate::layouts::EvalMod
#[derive(Clone, Debug)]
pub struct BootstrappingPlan {
    pipeline: BootstrappingPipeline,
    c2s_guard_bits: usize,
    log_gap: usize,
    pub(super) functional_message_modulus: Option<usize>,
    eval_mod_ring: EvalModRing,

    techniques: BootstrappingTechniques,

    /// CoeffsToSlots: homomorphic encoding ([`DFTType::Encode`]).
    pub(crate) coeffs_to_slots: DFTPlan,
    /// EvalMod: approximate `x mod 1`.
    pub(crate) eval_mod: EvalModPlan,

    /// SlotsToCoeffs: homomorphic decoding ([`DFTType::Decode`]).
    pub(crate) slots_to_coeffs: DFTPlan,

    /// `log2` of the degree a replicated layout states, `None` on the full layout, whose transforms and gap fill it.
    log_n: Option<usize>,
    working_bits: usize,
    s2c_guard_bits: usize,
}

impl BootstrappingPlan {
    /// The stages that set the widths at which a bootstrap uses its keys.
    pub(crate) fn stage_widths(&self) -> StageWidths<'_> {
        StageWidths {
            pipeline: self.pipeline,
            functional: self.functional_message_modulus.is_some(),
            c2s_guard_bits: self.c2s_guard_bits,
            working_bits: self.working_bits,
            s2c_guard_bits: self.s2c_guard_bits,
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
    /// The presets lay their keys out from them, a key generated outside their policy takes them for its layout, and
    /// [`ckks_bootstrap`](crate::api::CKKSBootstrappingOps::ckks_bootstrap) checks every rotation and tensor key against
    /// them before it starts.
    ///
    /// # Panics
    ///
    /// When `bootstrap_k` or `input_k` is narrower than the bits the stages running
    /// on it consume, on S2C-first when `bootstrap_k` is below the scale the raise labels on `input_k` plus the bits
    /// CoeffsToSlots and EvalMod consume, or when `input_k` lifted by the S2C guard and working bits overflows a `usize`.
    pub fn key_widths(&self, log_n: usize, bootstrap_k: usize, input_k: usize, input_log_delta: usize) -> BootstrappingKeyWidths {
        self.stage_widths()
            .key_widths(log_n, bootstrap_k, input_k, input_log_delta)
            .expect("a bootstrap width is narrower than the bits its stages consume or overflows a usize")
    }

    /// The `max_input_k` the tensor key's layout reaches at least, for a bootstrap working at `bootstrap_k` bits on an
    /// input at scale `2^input_log_delta`: the widest lookup of [`key_widths`](Self::key_widths) on the full layout,
    /// and on a replicated one EvalMod's product, wider by the bits its shaping depth exceeds EvalMod's scale
    /// `f_mod_log_delta`, which [`ckks_bootstrap`](crate::api::CKKSBootstrappingOps::ckks_bootstrap) checks at entry.
    /// A tensor key generated outside the presets' policy takes it for its layout.
    pub fn tensor_key_max_input_k(&self, bootstrap_k: usize, input_log_delta: usize) -> usize {
        self.stage_widths().eval_mod_product_k(bootstrap_k, input_log_delta)
    }

    /// Validated constructor for a complete bootstrapping recipe.
    ///
    /// `coeffs_to_slots` must be a
    /// [`DFTType::Encode`] plan and `slots_to_coeffs` a [`DFTType::Decode`]
    /// plan (errors [`InvalidPlan`](crate::CKKSCompositionError::InvalidPlan)
    /// otherwise): a swapped direction would make [`Self::galois_elements`]
    /// derive rotation keys for a transform that
    /// [`compile`](crate::layouts::BootstrappingContextPrepared::compile) never builds. The constructor also
    /// rejects recipe techniques that the selected pipeline cannot evaluate, [`EvalModType::ExpCmplx`] off the route of
    /// one EvalMod (see [EvalMod](Self#evalmod)), and an `eval_mod` or a transform off the full layout, which
    /// [`Self::with_slot_layout`] sets. A [packed](EvalModPlan::packed) EvalMod and the transform settings
    /// [`DFTPlan::with_slot_layout`] lists wait for it, which refuses them on the full layout, as the compilation of a plan
    /// it never set does; until then, the planning queries ([`Self::consumed_bits`], [`Self::key_widths`]) count them as
    /// set.
    pub fn new(
        pipeline: BootstrappingPipeline,
        techniques: BootstrappingTechniques,
        coeffs_to_slots: DFTPlan,
        eval_mod: EvalModPlan,
        slots_to_coeffs: DFTPlan,
    ) -> Result<Self> {
        let invalid = |reason: String| -> anyhow::Error {
            CKKSCompositionError::InvalidPlan {
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
        if eval_mod.eval_mod_type == EvalModType::ExpCmplx
            && (pipeline != BootstrappingPipeline::S2CFirst || techniques.eval_round_plus.is_some())
        {
            return Err(invalid(
                "the complex exponential takes an S2C-first plan without EvalRound+".to_string(),
            ));
        }
        let bypass = techniques
            .eval_round_plus
            .as_ref()
            .map(|eval_round| &eval_round.coeffs_to_slots_bypass);
        let transforms = [&coeffs_to_slots, &slots_to_coeffs].into_iter().chain(bypass);
        if eval_mod.slot_layout() != SlotLayout::Full || transforms.map(DFTPlan::slot_layout).any(|l| l != SlotLayout::Full) {
            return Err(invalid(
                "eval_mod and the transforms run on the full layout: with_slot_layout sets another".to_string(),
            ));
        }
        // The packing takes the replicated layout `with_slot_layout` validates it on.
        EvalModPlan {
            packed: false,
            ..eval_mod
        }
        .validate()?;
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
            log_gap: 0,
            functional_message_modulus: None,
            eval_mod_ring: EvalModRing::Standard,
            techniques,
            coeffs_to_slots,
            eval_mod,
            slots_to_coeffs,
            log_n: None,
            working_bits: 0,
            s2c_guard_bits: 0,
        })
    }

    /// Runs the stages on `slot_layout`, for ciphertexts of degree `2^log_n`: SlotsToCoeffs on `slot_layout`, and
    /// CoeffsToSlots and EvalMod on the layout CoeffsToSlots writes, of the values of `slot_layout` on two batches when
    /// EvalMod is [packed](EvalModPlan::packed) ([`DFTPlan::with_slot_layout`], [`EvalModPlan::with_slot_layout`]). The
    /// full layout takes the degree its transforms and gap fill.
    ///
    /// A [replicated layout](SlotLayout::Replicated) of `k` values of period `R <= k`, the small-batch bootstrapping of
    /// [ePrint 2026/2513](https://eprint.iacr.org/2026/2513), takes an S2C-first plan with a real EvalMod on the standard
    /// ring, without LUT, gap or EvalRound+, a degree whose slots hold `kR` values, twice as many packed, and transforms
    /// on its `k` values with the settings [`DFTPlan::with_slot_layout`] lists, each starting with the normalized trace
    /// onto the period its first factor reads. SlotsToCoeffs runs at scaling 1, which this sets, on a copy of the input, not doubled,
    /// lifted by [`Self::with_s2c_guard_bits`] and [`Self::with_working_bits`], the raise reads the working bits and lifts
    /// the message by the rest of `c2s_guard_bits`, and EvalMod runs as [its section](Self#evalmod) says.
    ///
    /// # Errors
    ///
    /// [`InvalidPlan`](crate::CKKSCompositionError::InvalidPlan) for a setting or a degree a replicated layout refuses,
    /// a transform or an EvalMod plan invalid on it, and on the full layout working or S2C guard bits, another degree
    /// than its transforms fill, a packed EvalMod or a transform setting of a replicated layout. The full layout is also
    /// refused on a plan already on a replicated layout, whose SlotsToCoeffs scaling this set to 1.
    pub fn with_slot_layout(mut self, slot_layout: SlotLayout, log_n: usize) -> Result<Self> {
        if slot_layout == SlotLayout::Full && self.slots_to_coeffs.slot_layout() != SlotLayout::Full {
            return Err(invalid_plan("a plan on a replicated layout cannot return to the full layout"));
        }
        let batches = match slot_layout {
            SlotLayout::Replicated(layout) if self.eval_mod.packed => {
                let log_k = layout.log_k() as u8 + 1;
                let packed = ReplicatedLayout::new(log_k, layout.log_r() as u8, layout.shaping());
                SlotLayout::Replicated(packed.map_err(|err| invalid_plan(&format!("a packed EvalMod: {err}")))?)
            }
            _ => slot_layout,
        };
        self.slots_to_coeffs = self.slots_to_coeffs.with_slot_layout(slot_layout);
        if slot_layout != SlotLayout::Full {
            self.slots_to_coeffs = self.slots_to_coeffs.with_scaling(1.0)?;
        }
        self.coeffs_to_slots = self.coeffs_to_slots.with_slot_layout(batches);
        self.eval_mod = self.eval_mod.with_slot_layout(batches);
        self.log_n = (slot_layout != SlotLayout::Full).then_some(log_n);
        self.check_layout()?;
        self.check_log_gap(log_n)?;
        self.slots_to_coeffs.check_slot_layout()?;
        self.coeffs_to_slots.check_slot_layout()?;
        self.eval_mod.validate()?;
        Ok(self)
    }

    /// Works SlotsToCoeffs of a replicated layout `bits` under the scale, which the raise reads as part of its guard
    /// bits ([`Self::with_c2s_guard_bits`]), and rounds each part of EvalMod's input with shaping `bits` under its
    /// scale.
    ///
    /// # Errors
    ///
    /// [`InvalidPlan`](crate::CKKSCompositionError::InvalidPlan) on the full layout, for more bits than the guard bits
    /// of the raise, and for more bits, with the S2C guard bits, than SlotsToCoeffs consumes.
    pub fn with_working_bits(mut self, bits: usize) -> Result<Self> {
        self.working_bits = bits;
        self.check_layout()?;
        Ok(self)
    }

    /// Lifts the input of SlotsToCoeffs on a replicated layout by `bits` scale bits, charged to the raised width and
    /// removed at the output: the raise reads its input `bits` above the scale of the message.
    ///
    /// # Errors
    ///
    /// [`InvalidPlan`](crate::CKKSCompositionError::InvalidPlan) on the full layout, and for more bits, with the working
    /// bits, than SlotsToCoeffs consumes.
    pub fn with_s2c_guard_bits(mut self, bits: usize) -> Result<Self> {
        self.s2c_guard_bits = bits;
        self.check_layout()?;
        Ok(self)
    }

    /// What [`Self::with_slot_layout`] refuses of the plan on its layout, and working or S2C guard bits on the full one.
    fn check_layout(&self) -> Result<()> {
        let (SlotLayout::Replicated(layout), Some(log_n)) = (self.slots_to_coeffs.slot_layout(), self.log_n) else {
            if self.working_bits > 0 || self.s2c_guard_bits > 0 {
                return Err(invalid_plan("working bits and S2C guard bits take a replicated layout"));
            }
            return Ok(());
        };
        let refused = [
            (self.pipeline != BootstrappingPipeline::S2CFirst, "C2S-first pipeline"),
            (self.eval_mod.eval_mod_type == EvalModType::ExpCmplx, "complex exponential"),
            (self.eval_mod_ring != EvalModRing::Standard, "EvalMod off the standard ring"),
            (self.functional_message_modulus.is_some(), "LUT"),
            (self.log_gap > 0, "gap"),
            (self.techniques.eval_round_plus.is_some(), "EvalRound+ bypass"),
        ];
        if let Some((_, setting)) = refused.into_iter().find(|&(refused, _)| refused) {
            return Err(invalid_plan(&format!("a replicated layout takes no {setting}")));
        }
        let Some(degree) = u32::try_from(log_n).ok().and_then(|log_n| 1usize.checked_shl(log_n)) else {
            return Err(invalid_plan(&format!("the degree 2^{log_n} overflows a usize")));
        };
        let (k, r, slots) = (1usize << layout.log_k(), 1usize << layout.log_r(), degree / 2);
        let reason = if k * r > slots {
            format!("{k} values of period {r} need more than the {slots} slots of degree 2^{log_n}")
        } else if self.eval_mod.packed && 2 * k * r > slots {
            format!("a packed EvalMod needs its two batches of {k} values of period {r} to fit the {slots} slots")
        } else if r > k {
            format!("bootstrapping needs the period {r} to divide the batch size {k}")
        } else if self.working_bits > self.c2s_guard_bits {
            "the working bits exceed the guard bits of the raise".to_string()
        } else if self
            .s2c_guard_bits
            .checked_add(self.working_bits)
            .is_none_or(|bits| bits > self.slots_to_coeffs.consumed_bits())
        {
            "the S2C guard and working bits exceed the bits SlotsToCoeffs consumes".to_string()
        } else {
            return Ok(());
        };
        Err(invalid_plan(&reason))
    }

    /// Working bits of a replicated layout, see [`Self::with_working_bits`].
    pub(crate) fn working_bits(&self) -> usize {
        self.working_bits
    }

    /// Guard bits SlotsToCoeffs adds on a replicated layout, see [`Self::with_s2c_guard_bits`].
    pub(crate) fn s2c_guard_bits(&self) -> usize {
        self.s2c_guard_bits
    }

    /// Configures S2C-first functional bootstrapping of integer messages modulo the LUT length.
    /// The initial transform includes the normalization to the LUT's unit-circle phase.
    ///
    /// # Errors
    ///
    /// [`InvalidPlan`](crate::CKKSCompositionError::InvalidPlan) on a plan with [`EvalRoundPlus`], whose bypass the
    /// LUT step does not run, with a gap, or on a replicated layout.
    pub fn with_functional_bootstrap<P>(mut self, lut: &EncodedLut<P>) -> Result<Self> {
        ensure!(
            self.pipeline == BootstrappingPipeline::S2CFirst,
            "functional bootstrapping requires an S2C-first plan"
        );
        ensure!(
            self.eval_mod_ring == EvalModRing::Standard,
            "functional bootstrapping runs EvalMod on the standard ring"
        );
        if self.techniques.eval_round_plus.is_some() {
            return Err(invalid_plan("the functional bootstrap does not run the EvalRound+ bypass"));
        }
        if self.log_gap > 0 {
            return Err(invalid_plan("functional bootstrapping takes a plan without a gap"));
        }
        let q = (1usize << lut.log_msg_ratio()) as f64;
        self.slots_to_coeffs = self.slots_to_coeffs.with_scaling(q / (2.0 * lut.message_modulus() as f64))?;
        self.functional_message_modulus = Some(lut.message_modulus());
        self.check_layout()?;
        Ok(self)
    }

    /// The selected ModUp/EvalMod bootstrapping pipeline.
    pub fn pipeline(&self) -> BootstrappingPipeline {
        self.pipeline
    }

    /// Plaintexts in `Z[X^(2^log_gap)]`: both transforms act on `N / 2^(log_gap + 1)` slots, and the sparse secret lies in
    /// that subring, so the raise projects the ciphertext on the subring in place of a trace. A plan on fewer slots than
    /// the ring needs its gap: nothing else keeps the overflow of the raise in the subring. Generating the keys checks the
    /// transforms against `N`, and a bootstrap its input's degree. Compiling the context checks only that they fit the
    /// degree of the module, which may exceed `N`.
    ///
    /// # Errors
    ///
    /// [`InvalidPlan`](crate::CKKSCompositionError::InvalidPlan) for a nonzero gap on a plan that is not S2C-first with
    /// EvalMod on the standard ring, the first CoeffsToSlots factor on the raise and a sparse secret, a functional plan,
    /// or a replicated layout.
    pub fn with_log_gap(mut self, log_gap: usize) -> Result<Self> {
        if log_gap > 0 && self.functional_message_modulus.is_some() {
            return Err(invalid_plan("a gap takes a plan without a LUT"));
        }
        if log_gap > 0
            && !(self.pipeline == BootstrappingPipeline::S2CFirst
                && self.eval_mod_ring == EvalModRing::Standard
                && self.coeffs_to_slots.first_factor_on_raise()
                && self.sparse_secret_hamming_weight().is_some())
        {
            return Err(invalid_plan(
                "a gap takes an S2C-first plan with EvalMod on the standard ring, the first factor on the raise and a sparse secret",
            ));
        }
        self.log_gap = log_gap;
        self.check_layout()?;
        Ok(self)
    }

    /// `log2` of the degree whose subring of the plan's gap the slots of `dft` fill, the one a replicated layout states.
    fn filled_log_n(&self, dft: &DFTPlan) -> Option<usize> {
        self.log_n
            .or_else(|| dft.log_slots().checked_add(1).and_then(|s| s.checked_add(self.log_gap)))
    }

    /// Refuses transforms that do not fill `N / 2^(log_gap + 1)` slots at the degree `N = 2^log_n` of the keys, and
    /// another degree than a replicated layout states.
    pub(crate) fn check_log_gap(&self, log_n: usize) -> Result<()> {
        let fills = |dft: &DFTPlan| self.filled_log_n(dft) == Some(log_n);
        if !(fills(&self.coeffs_to_slots) && fills(&self.slots_to_coeffs)) {
            return Err(invalid_plan(match self.log_n {
                Some(_) => "a replicated layout takes the degree it states",
                None => "the transforms must fill the slots of the subring of the gap",
            }));
        }
        Ok(())
    }

    /// Refuses transforms past `2^(log_n - log_gap - 1)` slots on a module of degree `2^log_n`, which may exceed `N`,
    /// and returns `log2` of the degree of the ciphertexts, [`Self::filled_log_n`] of the transforms.
    pub(crate) fn check_log_gap_fits(&self, log_n: usize) -> Result<usize> {
        let filled = |dft: &DFTPlan| self.filled_log_n(dft).filter(|&filled| filled <= log_n);
        match (filled(&self.coeffs_to_slots), filled(&self.slots_to_coeffs), self.log_n) {
            (Some(_), Some(filled), _) => Ok(filled),
            (.., Some(stated)) => Err(invalid_plan(&format!(
                "a replicated layout takes the degree it states, 2^{stated}, above the module's 2^{log_n}"
            ))),
            _ => Err(invalid_plan("the slots of the subring of the gap do not fit the module")),
        }
    }

    /// `log2` of the gap of the plaintexts, see [`Self::with_log_gap`].
    pub fn log_gap(&self) -> usize {
        self.log_gap
    }

    /// The secrets of the keys under the dense secret `dense`, as
    /// [`BootstrappingKeysLayout`](crate::layouts::BootstrappingKeysLayout) validates them: `dense`, and the ephemeral
    /// sparse secret of the plan's weight on the subring of its gap, of distribution [`Distribution::NONE`], which
    /// [`RlweParams`](poulpy_core::security::RlweParams) does not bound, without sparse-secret encapsulation.
    pub fn secrets(&self, dense: SecretSpec) -> BootstrappingSecrets {
        let sparse = SecretSpec::new(
            dense.n >> self.log_gap,
            self.sparse_secret_hamming_weight()
                .map_or(Distribution::NONE, Distribution::TernaryFixed),
        );
        BootstrappingSecrets { dense, sparse }
    }

    /// Adds scale bits at ModUp to reduce CoeffsToSlots rounding error.
    /// Charged to the raised width and removed at the ordinary bootstrap output. On a replicated layout they hold the
    /// working bits ([`Self::with_working_bits`]), which the raise reads from SlotsToCoeffs, and fewer are refused with
    /// [`InvalidPlan`](crate::CKKSCompositionError::InvalidPlan).
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
        self.check_layout()?;
        Ok(self)
    }

    /// Extra internal scale bits used between ModUp and CoeffsToSlots.
    pub fn c2s_guard_bits(&self) -> usize {
        self.c2s_guard_bits
    }

    /// What the modulus raise of this plan fixes.
    pub fn mod_raise(&self) -> ModRaisePlan {
        ModRaisePlan {
            lift: (self.pipeline == BootstrappingPipeline::C2SFirst).then_some(self.eval_mod),
            c2s_guard_bits: self.c2s_guard_bits - self.working_bits,
            sparse_secret_hamming_weight: self.sparse_secret_hamming_weight(),
            first_factor_on_raise: self.coeffs_to_slots.first_factor_on_raise(),
            log_gap: self.log_gap,
        }
    }

    /// Selects the ring EvalMod runs on. The conjugate-invariant ring takes an
    /// S2C-first plan with a real EvalMod, without a LUT, on the full layout; `u` folds
    /// into a CoeffsToSlots factor the matrix evaluates, `v` does not fold.
    pub fn with_eval_mod_ring(mut self, ring: EvalModRing) -> Result<Self> {
        if self.log_gap > 0 && ring != EvalModRing::Standard {
            return Err(invalid_plan("a gap takes EvalMod on the standard ring"));
        }
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
        self.check_layout()?;
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
    pub(super) fn with_dft_formats_resolved_at(&self, log_n: usize) -> Self {
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

    /// Input width for a given modulus at ModUp: the bits consumed before ModUp above it, less the guard and working
    /// bits SlotsToCoeffs adds.
    pub fn input_k(&self, log_modulus: usize) -> usize {
        log_modulus + (self.pre_mod_up_consumed_bits() - self.s2c_guard_bits - self.working_bits)
    }

    /// Bits the raised ciphertext fills at the bottom of the raised width, for
    /// a modulus of `log_modulus` bits at ModUp: what a sparse-to-dense key
    /// with a row offset of `bootstrap_k - raised_window` switches
    /// ([`EncapsulationKeysLayout::sparse_to_dense`](crate::layouts::EncapsulationKeysLayout::sparse_to_dense)).
    /// C2S-first lifts every input to the EvalMod scale, so its window does not
    /// depend on the input; S2C-first raises its modulus and the C2S guard bits
    /// it does not hold as working bits, and a narrower input is shifted into the window.
    pub fn raised_window(&self, log_modulus: usize) -> usize {
        match self.pipeline {
            BootstrappingPipeline::C2SFirst => self.eval_mod.f_mod_log_delta,
            BootstrappingPipeline::S2CFirst => log_modulus + self.c2s_guard_bits - self.working_bits,
        }
    }

    /// Bootstrap width for an output at the input scale `log_delta`.
    /// C2S-first includes the width removed when restoring that scale, and S2C-first the guard bits SlotsToCoeffs adds.
    pub fn bootstrap_k(&self, output_k: usize, log_delta: usize) -> usize {
        let scale_drop = match self.pipeline {
            BootstrappingPipeline::C2SFirst => self
                .eval_mod
                .f_mod_log_delta
                .checked_sub(log_delta)
                .expect("C2S-first input scale exceeds EvalMod scale"),
            BootstrappingPipeline::S2CFirst => self.s2c_guard_bits,
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
    /// The S2C guard bits are not counted: the output removes them as a scale
    /// drop, which [`Self::bootstrap_k`] adds to the raised width.
    pub fn consumed_bits(&self) -> usize {
        self.pre_mod_up_consumed_bits() + self.post_mod_up_consumed_bits()
    }

    /// Distinct Galois elements the pipeline's rotation keys must cover: the union
    /// of the CoeffsToSlots and SlotsToCoeffs transforms
    /// ([`DFTPlan::galois_elements`]), of the bypass, and, with `u` folded into the
    /// last CoeffsToSlots factor ([`CIConversion::Folded`]), of the two folded factors,
    /// for a ring of degree `2^log_n` and the given `cyclotomic_order`. The split
    /// forward transform's conjugation key is separate (generate it from Galois
    /// element `−1`) and not included here. A packed EvalMod adds the rotation of its second batch
    /// ([`EvalModPlan::packed`]), and on a replicated layout the transforms include their traces.
    pub fn galois_elements(&self, log_n: usize, cyclotomic_order: i64) -> Vec<i64> {
        let mut set: std::collections::BTreeSet<i64> = std::collections::BTreeSet::new();
        set.extend(
            self.eval_mod
                .batch_rotation()
                .map(|batch| galois_element(batch, cyclotomic_order)),
        );
        set.extend(self.coeffs_to_slots.galois_elements(log_n, cyclotomic_order));
        if let Some(bypass) = self.coeffs_to_slots_bypass() {
            set.extend(bypass.galois_elements(log_n, cyclotomic_order));
        }
        set.extend(self.slots_to_coeffs.galois_elements(log_n, cyclotomic_order));
        // The folded factors are the last CoeffsToSlots factor times diagonals of the full
        // slots, as `compile` builds them.
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        CKKSError, CoeffsMeta,
        layouts::{DFTOutputFormat, eval_mod::EvalModType},
        polynomial::SplitStrategy,
        presets::bootstrapping::paper_eval_mod_spec,
        shaping::ShapingUnit,
    };
    use poulpy_hal::layouts::Module;

    fn dft(kind: DFTType) -> DFTPlan {
        DFTPlan::new(
            kind,
            vec![(1, 1)],
            DFTOutputFormat::SplitRealAndImag,
            CoeffsMeta::from_delta_budget(8, 2),
        )
        .unwrap()
    }

    fn refusal(err: anyhow::Error) -> String {
        match err.downcast_ref::<crate::CKKSCompositionError>() {
            Some(crate::CKKSCompositionError::InvalidPlan { plan, reason }) => {
                assert_eq!(*plan, "BootstrappingPlan");
                reason.clone()
            }
            _ => panic!("unexpected error: {err}"),
        }
    }

    #[test]
    fn a_gap_takes_a_plan_of_its_subring() {
        let eval_mod = EvalModPlan {
            eval_mod_type: EvalModType::CosHK,
            log_msg_ratio: 4,
            f_mod_degree: 8,
            f_mod_interval: 4,
            f_mod_log_interval_reduction: 1,
            f_mod_inv_degree: None,
            scaling: None,
            split_strategy: SplitStrategy::MinDepth,
            coeffs_meta: CoeffsMeta::from_delta_budget(16, 2),
            f_mod_log_delta: 24,
            packed: false,
        };
        let new = |sparse: bool, on_raise: bool| {
            let encode = dft(DFTType::Encode);
            BootstrappingPlan::new(
                BootstrappingPipeline::S2CFirst,
                BootstrappingTechniques {
                    sparse_secret_encapsulation: sparse.then_some(SparseSecretEncapsulation { hamming_weight: 8 }),
                    eval_round_plus: None,
                },
                if on_raise {
                    encode.with_first_factor_on_raise()
                } else {
                    encode
                },
                eval_mod,
                dft(DFTType::Decode),
            )
            .unwrap()
        };

        assert!(refusal(new(false, true).with_log_gap(1).unwrap_err()).contains("a gap takes an S2C-first plan"));
        assert!(refusal(new(true, false).with_log_gap(1).unwrap_err()).contains("a gap takes an S2C-first plan"));
        let c2s_first = BootstrappingPlan::new(
            BootstrappingPipeline::C2SFirst,
            BootstrappingTechniques {
                sparse_secret_encapsulation: Some(SparseSecretEncapsulation { hamming_weight: 8 }),
                eval_round_plus: None,
            },
            dft(DFTType::Encode).with_first_factor_on_raise(),
            eval_mod,
            dft(DFTType::Decode),
        )
        .unwrap();
        assert!(refusal(c2s_first.with_log_gap(1).unwrap_err()).contains("a gap takes an S2C-first plan"));
        let plan = new(true, true).with_log_gap(1).unwrap();
        let ci = EvalModRing::ConjugateInvariant {
            pre: CIConversion::Folded,
            post: CIConversion::Product { log_delta: 8 },
        };
        assert!(refusal(plan.clone().with_eval_mod_ring(ci).unwrap_err()).contains("a gap takes EvalMod"));
        let on_ci = EvalModRing::ConjugateInvariant {
            pre: CIConversion::Product { log_delta: 8 },
            post: CIConversion::Product { log_delta: 8 },
        };
        let on_ci = new(true, true).with_eval_mod_ring(on_ci).unwrap();
        assert!(refusal(on_ci.with_log_gap(1).unwrap_err()).contains("a gap takes an S2C-first plan"));
        assert_eq!(plan.mod_raise().log_gap, 1);
        // One slot layer: the gap of 2 fills degree 8 and no other, and fits a module of that degree or above.
        plan.check_log_gap(3).unwrap();
        for log_n in [2, 4, 16] {
            assert!(refusal(plan.check_log_gap(log_n).unwrap_err()).contains("must fill the slots of the subring"));
        }
        for log_n in [3, 4, 16] {
            plan.check_log_gap_fits(log_n).unwrap();
        }
        assert!(refusal(plan.check_log_gap_fits(2).unwrap_err()).contains("do not fit the module"));
        let module = Module::<poulpy_hal::layouts::HostBytesBackend>::new(16);
        let lut = EncodedLut::general(
            &module,
            &[1.0f64; 5],
            32usize.into(),
            CoeffsMeta::from_delta_budget(55, 32),
            SplitStrategy::MinDepth,
        )
        .unwrap();
        assert!(refusal(plan.clone().with_functional_bootstrap(&lut).unwrap_err()).contains("without a gap"));
        let functional = new(true, true).with_functional_bootstrap(&lut).unwrap();
        assert!(refusal(functional.with_log_gap(1).unwrap_err()).contains("a gap takes a plan without a LUT"));
        // A plan on fewer slots than the ring needs its gap.
        assert!(refusal(new(true, true).check_log_gap(3).unwrap_err()).contains("must fill the slots of the subring"));
    }

    /// The constructor takes stage plans on the full layout only, and a replicated layout set on the plan takes the
    /// settings of the paper and refuses, each with its plan and reason, a setting of the full layout, a degree or a packing its
    /// values do not fit, a period above the batch size, working or S2C guard bits it cannot hold and a transform invalid
    /// on it; the full layout refuses working bits, another degree than its transforms fill, a packed EvalMod and a
    /// transform setting of a replicated layout.
    #[test]
    fn a_replicated_layout_refuses_what_it_does_not_run() {
        use BootstrappingPipeline::{C2SFirst, S2CFirst};
        let layout = ShapingUnit::U16_40.layout(3).unwrap();
        let replicated = SlotLayout::Replicated(layout);
        let unpacked = paper_eval_mod_spec(&layout).plan(13);
        let packed = unpacked.with_packed(true);
        let sparse = BootstrappingTechniques {
            sparse_secret_encapsulation: Some(SparseSecretEncapsulation { hamming_weight: 32 }),
            eval_round_plus: None,
        };
        let with_bypass = |coeffs_to_slots_bypass| BootstrappingTechniques {
            eval_round_plus: Some(EvalRoundPlus { coeffs_to_slots_bypass }),
            ..sparse.clone()
        };
        let on_k = |kind, log_delta| {
            let meta = CoeffsMeta::from_delta_budget(log_delta, 0);
            DFTPlan::new(kind, vec![(3, 1)], DFTOutputFormat::SplitRealAndImag, meta).unwrap()
        };
        let (encode, decode) = (|| on_k(DFTType::Encode, 34), || on_k(DFTType::Decode, 26));
        let new = |pipeline, techniques: &BootstrappingTechniques, c2s: DFTPlan, eval_mod, s2c: DFTPlan| {
            BootstrappingPlan::new(pipeline, techniques.clone(), c2s, eval_mod, s2c).unwrap()
        };
        let s2c_first = |eval_mod| new(S2CFirst, &sparse, encode(), eval_mod, decode());
        let off = |dft: DFTPlan| dft.with_slot_layout(replicated);
        for (techniques, c2s, eval_mod, s2c) in [
            (sparse.clone(), off(encode()), packed, decode()),
            (sparse.clone(), encode(), packed.with_slot_layout(replicated), decode()),
            (sparse.clone(), encode(), packed, off(decode())),
            (with_bypass(off(dft(DFTType::Encode))), encode(), unpacked, decode()),
        ] {
            let err = BootstrappingPlan::new(S2CFirst, techniques, c2s, eval_mod, s2c).unwrap_err();
            let reason = "eval_mod and the transforms run on the full layout: with_slot_layout sets another";
            assert_eq!(refusal(err), reason);
        }
        let layouts = |plan: &BootstrappingPlan| {
            let stages = [plan.coeffs_to_slots().slot_layout(), plan.eval_mod().slot_layout()];
            (plan.slots_to_coeffs().slot_layout(), stages)
        };
        let plan = s2c_first(packed)
            .with_slot_layout(replicated, 8)
            .and_then(|plan| plan.with_c2s_guard_bits(14))
            .and_then(|plan| plan.with_working_bits(8))
            .and_then(|plan| plan.with_s2c_guard_bits(10))
            .unwrap();
        let batches = SlotLayout::Replicated(ShapingUnit::U16_40.layout(4).unwrap());
        assert_eq!(layouts(&plan), (replicated, [batches; 2]));
        let module = Module::<poulpy_hal::layouts::HostBytesBackend>::new(16);
        let lut = EncodedLut::general(
            &module,
            &[1.0f64; 5],
            32usize.into(),
            CoeffsMeta::from_delta_budget(55, 32),
            SplitStrategy::MinDepth,
        )
        .unwrap();
        let ci = EvalModRing::ConjugateInvariant {
            pre: CIConversion::Product { log_delta: 8 },
            post: CIConversion::Product { log_delta: 8 },
        };
        let exp = EvalModPlan {
            eval_mod_type: EvalModType::ExpCmplx,
            ..unpacked
        };
        let gapped = new(S2CFirst, &sparse, encode().with_first_factor_on_raise(), unpacked, decode())
            .with_log_gap(1)
            .unwrap();
        let ungathered = SlotLayout::Replicated(ShapingUnit::U16_40.layout(2).unwrap());
        let (bits, refused) = (
            "the working bits exceed the guard bits of the raise",
            "a replicated layout takes no",
        );
        let full_bits = "working bits and S2C guard bits take a replicated layout";
        for (plan, reason) in [
            (
                new(C2SFirst, &sparse, encode(), packed, decode()).with_slot_layout(replicated, 8),
                format!("{refused} C2S-first pipeline"),
            ),
            (
                s2c_first(exp).with_slot_layout(replicated, 8),
                format!("{refused} complex exponential"),
            ),
            (
                plan.clone().with_eval_mod_ring(ci),
                format!("{refused} EvalMod off the standard ring"),
            ),
            (plan.clone().with_functional_bootstrap(&lut), format!("{refused} LUT")),
            (gapped.with_slot_layout(replicated, 8), format!("{refused} gap")),
            (
                new(S2CFirst, &with_bypass(dft(DFTType::Encode)), encode(), packed, decode()).with_slot_layout(replicated, 8),
                format!("{refused} EvalRound+ bypass"),
            ),
            (
                s2c_first(packed).with_slot_layout(replicated, 6),
                "8 values of period 8 need more than the 32 slots of degree 2^6".into(),
            ),
            (
                s2c_first(packed).with_slot_layout(replicated, 7),
                "a packed EvalMod needs its two batches of 8 values of period 8 to fit the 64 slots".into(),
            ),
            (
                s2c_first(packed).with_slot_layout(ungathered, 8),
                "bootstrapping needs the period 8 to divide the batch size 4".into(),
            ),
            (
                s2c_first(packed).with_slot_layout(SlotLayout::Replicated(ShapingUnit::U4_15.layout(61).unwrap()), 8),
                "a packed EvalMod: invalid replicated layout: 2kR must fit in a usize".into(),
            ),
            (plan.clone().with_working_bits(15), bits.into()),
            (plan.clone().with_c2s_guard_bits(7), bits.into()),
            (
                plan.clone().with_s2c_guard_bits(27),
                "the S2C guard and working bits exceed the bits SlotsToCoeffs consumes".into(),
            ),
            (s2c_first(unpacked).with_working_bits(1), full_bits.into()),
            (
                plan.clone().with_slot_layout(SlotLayout::Full, 4),
                "a plan on a replicated layout cannot return to the full layout".into(),
            ),
            (
                s2c_first(unpacked).with_slot_layout(SlotLayout::Full, 5),
                "the transforms must fill the slots of the subring of the gap".into(),
            ),
        ] {
            assert_eq!(refusal(plan.unwrap_err()), reason);
        }
        s2c_first(unpacked).with_slot_layout(SlotLayout::Full, 4).unwrap();
        let placed = |p| new(S2CFirst, &sparse, encode(), packed, decode().with_placement_factors(p));
        for (plan, inner, reason) in [
            (
                placed(4).with_slot_layout(replicated, 8),
                "DFTPlan",
                "the placement takes at most log2(k) = 3 factors",
            ),
            (
                placed(1).with_slot_layout(SlotLayout::Full, 4),
                "DFTPlan",
                "placement factors and a last-factor scale take a replicated layout",
            ),
            (
                s2c_first(packed).with_slot_layout(SlotLayout::Full, 4),
                "EvalModPlan",
                "a packed EvalMod takes a replicated layout",
            ),
        ] {
            let (err, reason) = (plan.unwrap_err(), reason.to_string());
            let got = err
                .downcast_ref::<CKKSError>()
                .map_or(err.downcast_ref(), CKKSError::composition);
            assert_eq!(got, Some(&CKKSCompositionError::InvalidPlan { plan: inner, reason }));
        }
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
            packed: false,
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
