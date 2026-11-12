//! What a bootstrap runs: the stages, their techniques and what they consume.

use anyhow::{Result, ensure};

use crate::{
    CKKSCompositionError, CKKSInfos,
    layouts::{DFTPlan, DFTType, EncodedLut, EvalModPlan, ModRaisePlan, eval_mod::EvalModType},
};

use super::{BootstrappingKeyWidths, CIConversion, EvalModRing, key_widths::StageWidths};

/// Stage ordering of a bootstrap.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BootstrappingPipeline {
    /// The raise, CoeffsToSlots, EvalMod and SlotsToCoeffs. The last transform restores the message ratio and labels
    /// the output, whose scale and width return to the input's.
    C2SFirst,
    /// SlotsToCoeffs below the raise, at scaling `1/2` on a doubled input, then the raise, CoeffsToSlots and EvalMod,
    /// whose output is relabeled at the input's scale.
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
/// bootstrap, whose LUT step runs no bypass.
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
///   rounding, charged to the raised width and removed at the output.
/// - `techniques` ([`BootstrappingTechniques`]): [`SparseSecretEncapsulation`] around the raise and [`EvalRoundPlus`]
///   around EvalMod. A first CoeffsToSlots factor on the raise excludes EvalRound+.
/// - `functional_message_modulus` ([`Self::with_functional_bootstrap`], S2C-first): turns the bootstrap into a LUT
///   evaluation on integer messages, on the standard ring, without EvalRound+.
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
/// - C2S-first: `log_delta_out = a.log_delta()`, `log_budget_out = a.log_budget() - consumed`, until SlotsToCoeffs
///   labels it;
/// - under [`EvalRoundPlus`], and on a plain [`EvalMod`]: the same, with the guard bits still on and the working label,
///   slot kind and sparsity being those of the route on a plain [`EvalMod`].
///
/// The split route leaves the canonical flag clear: its recombination is a lazy sum.
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
    pub(super) functional_message_modulus: Option<usize>,
    eval_mod_ring: EvalModRing,

    techniques: BootstrappingTechniques,

    /// CoeffsToSlots: homomorphic encoding ([`DFTType::Encode`]).
    pub(crate) coeffs_to_slots: DFTPlan,
    /// EvalMod: approximate `x mod 1`.
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
    /// otherwise): a swapped direction would make [`Self::galois_elements`]
    /// derive rotation keys for a transform that
    /// [`compile`](crate::layouts::BootstrappingContextPrepared::compile) never builds. The constructor also
    /// rejects recipe techniques that the selected pipeline cannot evaluate.
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
    ///
    /// # Errors
    ///
    /// [`InvalidPlan`](crate::CKKSCompositionError::InvalidPlan) on a plan with [`EvalRoundPlus`], whose bypass the
    /// LUT step does not run.
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
            return Err(CKKSCompositionError::InvalidPlan {
                plan: "BootstrappingPlan",
                reason: "the functional bootstrap does not run the EvalRound+ bypass".into(),
            }
            .into());
        }
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

    /// What the modulus raise of this plan fixes.
    pub fn mod_raise(&self) -> ModRaisePlan {
        ModRaisePlan {
            lift: (self.pipeline == BootstrappingPipeline::C2SFirst).then_some(self.eval_mod),
            c2s_guard_bits: self.c2s_guard_bits,
            sparse_secret_hamming_weight: self.sparse_secret_hamming_weight(),
            first_factor_on_raise: self.coeffs_to_slots.first_factor_on_raise(),
        }
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
        CoeffsMeta,
        layouts::{DFTOutputFormat, eval_mod::EvalModType},
        polynomial::SplitStrategy,
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
