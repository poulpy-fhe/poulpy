//! The widths at which a bootstrap uses its keys, from the stages that set them.

use std::collections::BTreeMap;

use crate::layouts::{DFTPlan, EvalModPlan};

use super::{BootstrappingPipeline, CIConversion, EvalModRing};

/// The widths, in bits, at which a bootstrap uses its keys
/// ([`key_widths`](crate::layouts::BootstrappingPlan::key_widths)): a key covers its width when its
/// `max_input_k` is at least that width.
///
/// A rotation key's width is that of the widest ciphertext it switches: a baby
/// rotation switches the input of its factor, and a giant rotation the factor's
/// product, one factor scale narrower. The tensor key's width is that of the
/// widest product EvalMod relinearizes.
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

/// The offsets of a rotation key: how far below the start of each stage its widest use sits, in bits, `None` for a stage
/// that does not use it. A plan holds no torus width, so the offsets are the plan's and [`StageStarts`] the ciphertext's.
#[derive(Clone, Copy, Debug)]
pub(crate) struct RotationKeyOffsets {
    pub(crate) p: i64,
    /// CoeffsToSlots, its bypass and the factors `u` folds into, below the raised width.
    raised: Option<usize>,
    slots_to_coeffs: Option<usize>,
    /// The conjugation of the split, on what CoeffsToSlots leaves.
    after_c2s: Option<usize>,
}

/// The widths the stages of a bootstrap start at.
pub(crate) struct StageStarts {
    raised: usize,
    slots_to_coeffs: usize,
    after_c2s: usize,
}

impl StageStarts {
    /// The width of the widest ciphertext `key` switches.
    pub(crate) fn width(&self, key: &RotationKeyOffsets) -> usize {
        [
            (self.raised, key.raised),
            (self.slots_to_coeffs, key.slots_to_coeffs),
            (self.after_c2s, key.after_c2s),
        ]
        .into_iter()
        .filter_map(|(start, below)| Some(start - below?))
        .max()
        .unwrap_or(0)
    }
}

/// The stages that set the widths at which a bootstrap uses its keys, given the
/// width it runs at and its input.
pub(crate) struct StageWidths<'a> {
    pub(super) pipeline: BootstrappingPipeline,
    pub(super) c2s_guard_bits: usize,
    pub(super) eval_mod_ring: EvalModRing,
    pub(super) coeffs_to_slots: &'a DFTPlan,
    pub(super) coeffs_to_slots_bypass: Option<&'a DFTPlan>,
    pub(super) slots_to_coeffs: &'a DFTPlan,
    pub(super) eval_mod: &'a EvalModPlan,
}

impl StageWidths<'_> {
    /// [`key_widths`](crate::layouts::BootstrappingPlan::key_widths), or `None` when `bootstrap_k` or `input_k`
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

    /// Width, in bits, of each rotation key, by Galois element: [`RotationKeyOffsets`] at the starts of
    /// [`Self::stage_starts`].
    fn rotation_key_inputs(&self, log_n: usize, bootstrap_k: usize, input_k: usize) -> Option<BTreeMap<i64, usize>> {
        let starts = self.stage_starts(bootstrap_k, input_k)?;
        Some(
            self.rotation_key_offsets(log_n)
                .iter()
                .map(|key| (key.p, starts.width(key)))
                .collect(),
        )
    }

    /// The widths the stages start at, or `None` when `bootstrap_k` or `input_k` is narrower than the bits they consume.
    /// C2S-first runs SlotsToCoeffs on what EvalMod, and the bypass, leave at EvalMod's scale, so its budget above that
    /// scale holds the bits SlotsToCoeffs consumes.
    pub(crate) fn stage_starts(&self, bootstrap_k: usize, input_k: usize) -> Option<StageStarts> {
        let after_c2s = bootstrap_k.checked_sub(self.coeffs_to_slots.consumed_bits())?;
        let (slots_to_coeffs, scale) = match self.pipeline {
            BootstrappingPipeline::S2CFirst => (input_k, 0),
            BootstrappingPipeline::C2SFirst => {
                let mut start = after_c2s.checked_sub(self.eval_mod.consumed_bits())?;
                if let Some(bypass) = self.coeffs_to_slots_bypass {
                    start = start.min(bootstrap_k.checked_sub(bypass.consumed_bits())?);
                }
                (start, self.eval_mod.f_mod_log_delta)
            }
        };
        // Every factor below then runs above the bits its transform consumes.
        slots_to_coeffs.checked_sub(self.slots_to_coeffs.consumed_bits() + scale)?;
        Some(StageStarts {
            raised: bootstrap_k,
            slots_to_coeffs,
            after_c2s,
        })
    }

    /// The offsets of every rotation key at degree `2^log_n`, sorted by Galois element. Factor `i` of a transform runs `i`
    /// factor scales below the width the transform starts at: the raised width for CoeffsToSlots and its bypass, and for
    /// SlotsToCoeffs the input width (S2C-first) or what EvalMod, and the bypass, leave (C2S-first). The folded factors of
    /// [`CIConversion::Folded`] run where the last CoeffsToSlots factor would. The conjugation follows CoeffsToSlots, in
    /// EvalMod's split or real part, which the conjugate-invariant ring does not take; the bypass, evaluated unsplit,
    /// conjugates nothing.
    pub(crate) fn rotation_key_offsets(&self, log_n: usize) -> Vec<RotationKeyOffsets> {
        let cyclotomic_order = 1i64 << (log_n + 1);
        let mut keys: BTreeMap<i64, RotationKeyOffsets> = BTreeMap::new();
        let mut charge = |stage: fn(&mut RotationKeyOffsets) -> &mut Option<usize>, offsets: BTreeMap<i64, usize>| {
            for (p, below) in offsets {
                let key = keys.entry(p).or_insert(RotationKeyOffsets {
                    p,
                    raised: None,
                    slots_to_coeffs: None,
                    after_c2s: None,
                });
                let offset = stage(key);
                *offset = Some(offset.map_or(below, |seen| seen.min(below)));
            }
        };
        charge(
            |key| &mut key.raised,
            self.coeffs_to_slots.key_offsets(log_n, cyclotomic_order),
        );
        charge(
            |key| &mut key.slots_to_coeffs,
            self.slots_to_coeffs.key_offsets(log_n, cyclotomic_order),
        );
        if let Some(bypass) = self.coeffs_to_slots_bypass {
            charge(|key| &mut key.raised, bypass.key_offsets(log_n, cyclotomic_order));
        }
        if let EvalModRing::ConjugateInvariant {
            pre: CIConversion::Folded,
            ..
        } = self.eval_mod_ring
        {
            let c2s = self.coeffs_to_slots.resolved_at(log_n);
            charge(
                |key| &mut key.raised,
                c2s.last_factor_key_offsets_at(log_n, 1 << (log_n - 1), cyclotomic_order),
            );
        }
        if self.eval_mod_ring == EvalModRing::Standard {
            charge(|key| &mut key.after_c2s, BTreeMap::from([(-1, 0)]));
        }
        keys.into_values().collect()
    }

    /// Widths of the ciphertexts the trace and the embedding switch on the way
    /// to the conjugate-invariant ring and back, if EvalMod runs there.
    pub(crate) fn conversions(&self, bootstrap_k: usize) -> Option<(usize, usize)> {
        (self.eval_mod_ring != EvalModRing::Standard).then(|| self.eval_mod.conversion_widths(self.eval_mod_k(bootstrap_k)))
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
    pub(crate) fn eval_mod_relinearization_k(&self, bootstrap_k: usize, input_log_delta: usize) -> usize {
        let f_mod_log_delta = self.eval_mod.f_mod_log_delta;
        let raised_log_delta = match self.pipeline {
            BootstrappingPipeline::C2SFirst => f_mod_log_delta,
            BootstrappingPipeline::S2CFirst => input_log_delta + self.eval_mod.log_msg_ratio + self.c2s_guard_bits,
        };
        // A width too narrow for EvalMod fails at its key lookup or in EvalMod, not here.
        self.eval_mod
            .relinearization_k(self.eval_mod_k(bootstrap_k).saturating_sub(raised_log_delta))
    }
}
