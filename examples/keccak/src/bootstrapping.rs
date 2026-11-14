//! The binary bootstrapping: integers `m` at scale `2^LOG_DELTA` in, `m mod 2` at scale `2^(LOG_DELTA + 1)` out.
//!
//! It is S2C-first, with sparse-secret encapsulation around the raise, and a binary table, a cosine with `K = 16` and
//! 3 double angles as in the presets, in place of EvalMod. The README walks through the steps.

use anyhow::{Result, ensure};
use poulpy_ckks::{
    CKKSLayout, CoeffsMeta, GLWEPlaintextMeta, Scale, SlotLayout, SlotsKind,
    layouts::{
        BootstrappingKeysLayout, BootstrappingPipeline, BootstrappingPlan, BootstrappingTechniques, CKKSPlaintextOwned,
        DFTOutputFormat, DFTPlan, DFTType, EncapsulationKeysLayout, EncodedLut, EvalModPlan, EvalModType,
        SparseSecretEncapsulation,
    },
    polynomial::SplitStrategy,
};
use poulpy_core::layouts::{Degree, GGLWEInfos, GGLWELayout, GLWELayout, Rank, TorusPrecision, optimal_bsgs_giant_step};
use poulpy_hal::layouts::{HostBytesBackend, Module};

use crate::{
    layout::SLOTS,
    params::{BASE2K, DENSE_SECRET, LOG_DELTA, LOG_MSG_RATIO, LOG_N, SPARSE_SECRET, key_layout},
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

/// The plan, the table and the widths of the bootstrapping, and the layouts of its keys.
pub struct Bootstrapping {
    pub plan: BootstrappingPlan,
    /// The table, encoded on the host: the server moves it to its backend.
    pub lut: EncodedLut<CKKSPlaintextOwned<HostBytesBackend>>,
    pub keys: BootstrappingKeysLayout,
    /// Width of the input: the 31 bits of the raise and what SlotsToCoeffs takes above them.
    pub input_k: usize,
    /// Width of the output, at scale `2^OUTPUT_LOG_DELTA`.
    pub output_k: usize,
    /// Width of the raised ciphertext.
    pub raised_k: usize,
}

impl Bootstrapping {
    /// The bootstrapping whose output keeps `round_bits` bits above its input, the budget of a round.
    pub fn new(round_bits: usize) -> Result<Self> {
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
        // CoeffsToSlots, the message ratio, and the scale, which sizes the tensor key the table's products use.
        let eval_mod = EvalModPlan {
            eval_mod_type: EvalModType::CosCheby,
            log_msg_ratio: LOG_MSG_RATIO,
            f_mod_degree: LUT_DEGREE,
            f_mod_interval: LUT_INTERVAL,
            f_mod_log_interval_reduction: LUT_DOUBLE_ANGLES,
            f_mod_inv_degree: None,
            scaling: None,
            split_strategy: SplitStrategy::MinDepth,
            coeffs_meta: CoeffsMeta::from_delta_budget(42, 4),
            f_mod_log_delta: 58,
        };
        let plan = BootstrappingPlan::new(
            BootstrappingPipeline::S2CFirst,
            BootstrappingTechniques {
                sparse_secret_encapsulation: Some(SparseSecretEncapsulation {
                    hamming_weight: SPARSE_SECRET.hamming_weight,
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
        ensure!(
            raised_k <= DENSE_SECRET.max_log_modulus,
            "the raised ciphertext of {raised_k} bits is above the dense secret's bound of {}",
            DENSE_SECRET.max_log_modulus
        );
        let keys = keys_layout(&plan, input_k, raised_k)?;
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
    pub fn input_layout(&self) -> CKKSLayout {
        self.layout(self.input_k)
    }

    pub fn raised_layout(&self) -> CKKSLayout {
        self.layout(self.raised_k)
    }
}

/// A transform whose factors merge `layers` slot layers each, every factor evaluated baby-step giant-step with the
/// giant step that needs the fewest rotations of its diagonals.
fn dft_plan(kind: DFTType, layers: &[usize], meta: CoeffsMeta) -> Result<DFTPlan> {
    let plan = |giant_steps: &[usize]| {
        DFTPlan::new(
            kind,
            layers.iter().copied().zip(giant_steps.iter().copied()).collect::<Vec<_>>(),
            DFTOutputFormat::SplitRealAndImag,
            meta,
        )
    };
    let giant_steps: Vec<usize> = plan(&vec![1; layers.len()])?
        .diagonal_indexes(LOG_N)
        .into_iter()
        .map(|indexes| optimal_bsgs_giant_step(indexes, SLOTS))
        .collect();
    plan(&giant_steps)
}

/// The keys of the bootstrapping:
///
/// - a rotation key per Galois element of the transforms, and the tensor key of the table, each as
///   wide as the widest ciphertext it switches ([`BootstrappingPlan::key_widths`]),
/// - the switch to the sparse secret, on the 31 bits of the raise, under the sparse secret's bound,
/// - the switch back, which is the raise itself: its rows sit at the bottom of the raised width, under the few bits
///   the raised ciphertext fills there, so it is narrow although its modulus reaches the raised width.
fn keys_layout(plan: &BootstrappingPlan, input_k: usize, raised_k: usize) -> Result<BootstrappingKeysLayout> {
    let widths = plan.key_widths(LOG_N, raised_k, input_k, LOG_DELTA);
    let rotation_keys = widths
        .rotation_keys
        .iter()
        .map(|(&p, &k)| Ok((p, key_layout(k, DENSE_SECRET.max_log_modulus)?.into())))
        .collect::<Result<_>>()?;
    let tensor_key = key_layout(widths.tensor_key, DENSE_SECRET.max_log_modulus)?.into();

    let dense_to_sparse = key_layout(LOG_DELTA + LOG_MSG_RATIO, SPARSE_SECRET.max_log_modulus)?;
    let window = plan.raised_window(LOG_DELTA + LOG_MSG_RATIO);
    let digit = window.next_multiple_of(BASE2K);
    let k_offset = raised_k - digit;
    let sparse_to_dense = GGLWELayout {
        k_offset: TorusPrecision(k_offset as u32),
        ..key_layout(window, DENSE_SECRET.max_log_modulus - k_offset)?
    };
    ensure!(
        sparse_to_dense.gglwe_layout().gadget_k().as_usize() == digit,
        "the sparse-to-dense key covers its {window}-bit window with one digit"
    );

    Ok(BootstrappingKeysLayout {
        rotation_keys,
        tensor_key,
        encapsulation: Some(EncapsulationKeysLayout {
            dense_to_sparse: dense_to_sparse.into(),
            sparse_to_dense: Some(sparse_to_dense.into()),
        }),
        first_factor_key: None,
        conversion_keys: None,
    })
}

#[cfg(test)]
mod tests {
    use poulpy_core::layouts::LWEInfos;

    use super::*;

    #[test]
    fn widths_and_keys_fit_the_bounds() {
        let boot = crate::params::Params::new().unwrap().bootstrapping;
        assert_eq!((boot.input_k, boot.output_k, boot.raised_k), (125, 203, 624));
        for key in boot.keys.rotation_key_layouts() {
            assert!(key.k().as_usize() <= DENSE_SECRET.max_log_modulus);
        }
    }
}
