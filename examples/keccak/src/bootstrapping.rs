//! The binary bootstrapping: integers `m` at scale `2^LOG_DELTA` in, `m mod 2` at scale `2^(LOG_DELTA + 1)` out.
//!
//! It is S2C-first, with sparse-secret encapsulation around the raise, and a binary table, a cosine with `K = 16` and
//! 3 double angles as in the presets, in place of EvalMod. The README walks through the steps.

use anyhow::{Result, ensure};
use poulpy_ckks::{
    CKKSLayout, CoeffsMeta, GLWEPlaintextMeta, Scale, SlotLayout, SlotsKind,
    layouts::{
        BootstrappingKeysLayout, BootstrappingPipeline, BootstrappingPlan, BootstrappingTechniques, CKKSPlaintextOwned,
        DFTOutputFormat, DFTPlan, DFTType, EncodedLut, EvalModPlan, EvalModType, SparseSecretEncapsulation,
    },
    polynomial::SplitStrategy,
    presets::key_policy::BootstrappingKeyPolicy,
};
use poulpy_core::{
    layouts::{Degree, GLWELayout, Rank},
    security::{RlweParams, ValidateSecurity},
};
use poulpy_hal::layouts::{HostBytesBackend, Module};

use crate::{
    layout::SLOTS,
    params::{BASE2K, DENSE_SECRET, LOG_DELTA, LOG_MSG_RATIO, LOG_N, SECURITY, SPARSE_HAMMING_WEIGHT, SPARSE_SECRET},
};

/// SlotsToCoeffs on the 14 slot layers: the layers each factor merges, in evaluation order, and the scale of its
/// diagonals.
const S2C_LAYERS: [usize; 3] = [4, 5, 5];
const S2C_LOG_DELTA: usize = 30;
/// CoeffsToSlots, the same layers in the other order.
const C2S_LAYERS: [usize; 3] = [5, 5, 4];
const C2S_LOG_DELTA: usize = 40;
/// Bits the raise keeps below the message, so the error of CoeffsToSlots stays under the scale.
const C2S_GUARD_BITS: usize = 6;

/// `K`, the interval of the cosine.
const LUT_INTERVAL: usize = 16;
/// The cosine polynomial and its double angles, and the scale of its coefficients.
const LUT_DEGREE: usize = 31;
const LUT_DOUBLE_ANGLES: usize = 3;
const LUT_COEFFS_LOG_DELTA: usize = 50;

/// Scale the raise labels the table's input with: the message, its ratio and the guard bits of CoeffsToSlots. The plan
/// sizes the tensor key from it, as the width the table's products reach.
const RAISED_LOG_DELTA: usize = LOG_DELTA + LOG_MSG_RATIO + C2S_GUARD_BITS;

/// The plan, the table and the widths of the bootstrapping, and the layouts of its keys.
pub(crate) struct Bootstrapping {
    pub(crate) plan: BootstrappingPlan,
    /// The table, encoded on the host: the server moves it to its backend.
    pub(crate) lut: EncodedLut<CKKSPlaintextOwned<HostBytesBackend>>,
    pub(crate) keys: BootstrappingKeysLayout,
    /// Width of the input: the 31 bits of the raise and what SlotsToCoeffs takes above them.
    pub(crate) input_k: usize,
    /// Width of the output, at scale `2^OUTPUT_LOG_DELTA`.
    pub(crate) output_k: usize,
    /// Width of the raised ciphertext.
    pub(crate) raised_k: usize,
}

impl Bootstrapping {
    /// The bootstrapping whose output keeps `round_bits` bits above its input, the budget of a round.
    pub(crate) fn new(round_bits: usize) -> Result<Self> {
        let slots_to_coeffs = dft_plan(DFTType::Decode, &S2C_LAYERS, CoeffsMeta::from_delta_budget(S2C_LOG_DELTA, 2))?;
        let coeffs_to_slots = dft_plan(DFTType::Encode, &C2S_LAYERS, CoeffsMeta::from_delta_budget(C2S_LOG_DELTA, 3))?;
        ensure!(
            slots_to_coeffs.log_slots() == SLOTS.ilog2() as usize && coeffs_to_slots.log_slots() == SLOTS.ilog2() as usize,
            "the transforms cover the {SLOTS} slots"
        );

        let lut = EncodedLut::binary(
            &Module::<HostBytesBackend>::new(1 << LOG_N),
            0.0,
            1.0,
            LUT_DEGREE,
            LUT_INTERVAL,
            LUT_DOUBLE_ANGLES,
            BASE2K.into(),
            CoeffsMeta::from_delta_budget(LUT_COEFFS_LOG_DELTA, 4),
            SplitStrategy::MinDepth,
        )?;

        // The table replaces EvalMod, but the plan still reads three of its fields: the interval, for the `1/K` of
        // CoeffsToSlots, the message ratio, and the scale, which sizes the tensor key the table's products use. The
        // others describe the table, which no EvalMod evaluates.
        let eval_mod = EvalModPlan {
            eval_mod_type: EvalModType::CosCheby,
            log_msg_ratio: LOG_MSG_RATIO,
            f_mod_degree: LUT_DEGREE,
            f_mod_interval: LUT_INTERVAL,
            f_mod_log_interval_reduction: LUT_DOUBLE_ANGLES,
            f_mod_inv_degree: None,
            scaling: None,
            split_strategy: SplitStrategy::MinDepth,
            coeffs_meta: CoeffsMeta::from_delta_budget(LUT_COEFFS_LOG_DELTA, 4),
            f_mod_log_delta: RAISED_LOG_DELTA,
        };
        let plan = BootstrappingPlan::new(
            BootstrappingPipeline::S2CFirst,
            BootstrappingTechniques {
                sparse_secret_encapsulation: Some(SparseSecretEncapsulation {
                    hamming_weight: SPARSE_HAMMING_WEIGHT,
                }),
                eval_round_plus: None,
            },
            coeffs_to_slots,
            eval_mod,
            slots_to_coeffs,
        )?
        .with_c2s_guard_bits(C2S_GUARD_BITS)?
        // Scales SlotsToCoeffs so the table reads `m/2` on its unit circle.
        .with_functional_bootstrap(&lut)?;

        let log_modulus = LOG_DELTA + LOG_MSG_RATIO;
        let input_k = plan.input_k(log_modulus);
        let output_k = input_k + round_bits;
        let raised_k = plan.functional_bootstrap_k(output_k, LOG_DELTA, &lut)?;
        SECURITY.check_log_q(DENSE_SECRET, "raised ciphertext", raised_k)?;
        // Each key as wide as the widest ciphertext it switches, with the fewest digits under its secret's bound.
        let keys = BootstrappingKeyPolicy {
            log_n: LOG_N,
            base2k: BASE2K,
            rank: 1,
            dense_log_modulus: SECURITY.max_log_q(DENSE_SECRET)?,
            sparse_log_modulus: SECURITY.max_log_q(SPARSE_SECRET)?,
            max_dsize: None,
            dense_to_sparse_dsize: 1,
        }
        .keys_layout(&plan, raised_k, input_k, LOG_DELTA)?;
        keys.validate_security_128(&plan.secrets(DENSE_SECRET), &RlweParams)?;
        Ok(Self {
            plan,
            lut,
            keys,
            input_k,
            output_k,
            raised_k,
        })
    }

    /// A ciphertext of `k` bits, at the scale of the bootstrapping's input.
    fn layout(&self, k: usize) -> CKKSLayout {
        CKKSLayout {
            glwe_layout: GLWELayout {
                n: Degree(1 << LOG_N),
                base2k: BASE2K.into(),
                k: k.into(),
                rank: Rank(1),
            },
            meta: GLWEPlaintextMeta {
                scale: Scale::Log(LOG_DELTA),
                slots: SlotsKind::Real,
                log_sparsity: 0,
                slot_layout: SlotLayout::Full,
            },
        }
    }

    /// Layouts of the input and of the raised ciphertext.
    pub(crate) fn input_layout(&self) -> CKKSLayout {
        self.layout(self.input_k)
    }

    pub(crate) fn raised_layout(&self) -> CKKSLayout {
        self.layout(self.raised_k)
    }
}

/// A transform whose factors merge `layers` slot layers each, every factor evaluated baby-step giant-step with the
/// giant step that needs the fewest rotations of its diagonals.
fn dft_plan(kind: DFTType, layers: &[usize], meta: CoeffsMeta) -> Result<DFTPlan> {
    let factors: Vec<_> = layers.iter().map(|&layer| (layer, 1)).collect();
    Ok(DFTPlan::new(kind, factors, DFTOutputFormat::SplitRealAndImag, meta)?.with_optimal_bsgs(LOG_N))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn widths_and_keys_are_certified() {
        let boot = crate::params::Params::new().unwrap().bootstrapping;
        assert_eq!((boot.input_k, boot.output_k, boot.raised_k), (125, 203, 624));
        let secrets = boot.plan.secrets(DENSE_SECRET);
        assert_eq!(boot.keys.validate_security_128(&secrets, &RlweParams), Ok(()));
        // The bounds the keys are shaped under, at rlwe-params v0.2.0.
        assert_eq!(
            (SECURITY.max_log_q(DENSE_SECRET), SECURITY.max_log_q(SPARSE_SECRET)),
            (Ok(851), Ok(162))
        );
    }
}
