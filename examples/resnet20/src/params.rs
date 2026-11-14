//! The bootstrap of the network, `n16_d35_k685_p19_s2c_ffr_ci` refitted to degree `2^15` with real slots, and where
//! each activation sits in the coefficients.

use anyhow::Result;
use poulpy_ckks::{
    GLWEPlaintextMeta, Scale, SlotLayout, SlotsKind,
    layouts::EvalModRing,
    presets::bootstrapping::{BootstrappingPreset, PresetScales, n16_d35_k685_p19_s2c_ffr_ci},
};

use crate::{layout::CoeffLayout, model::ROW_STRIDE};

/// The degree of the ring.
const LOG_N: usize = 15;
/// The largest radix `max_backend_base2k` allows at a failure of `2^-128`.
const BASE2K: usize = 53;
/// The transforms, on 14 layers.
const C2S_SCHEDULE: &[(usize, usize)] = &[(3, 1), (3, 1024), (3, 128), (3, 16), (2, 2)];
const S2C_SCHEDULE: &[(usize, usize)] = &[(4, 4), (4, 64), (6, 2048)];
/// As many factors as [`S2C_SCHEDULE`] on the 15 layers of the preset's degree: the input width counts one scale per
/// factor, and the preset must be narrow before it moves to degree `2^15`.
const S2C_SCHEDULE_N16: &[(usize, usize)] = &[(4, 4), (5, 128), (6, 4096)];
/// The classes of coefficients the last SlotsToCoeffs factor tells apart.
#[cfg(test)]
pub(crate) const MERGED_COSETS: usize = 1 << S2C_SCHEDULE[S2C_SCHEDULE.len() - 1].0;

/// Scale of the last SlotsToCoeffs factor, with the kernels folded in: the bits it spends.
const LOG_DELTA_W: usize = 20;

/// Levels of the ReLU: its polynomial of degree 15.
const RELU_LEVELS: usize = 4;

/// The logits reach `2^5`, past the ratio the layers run at, and leave at this many bits above it.
const LOG_LOGIT_RANGE: usize = 5;

/// The narrowest scales that keep the accuracy of the cleartext polynomial network, tuned on the wall clock.
const SCALES: PresetScales = PresetScales {
    log_delta: 28,
    log_msg_ratio: 1,
    c2s_guard_bits: 2,
    c2s_log_delta: 26,
    s2c_log_delta: 16,
    eval_mod_log_delta: 31,
    eval_mod_coeffs_log_delta: 30,
};

/// What SlotsToCoeffs takes above the raise's input: the bits its last factor spends past a plain one, and the range
/// of the logits.
const S2C_HEADROOM: usize = LOG_DELTA_W - SCALES.s2c_log_delta + LOG_LOGIT_RANGE;

/// The integer part of a raised coefficient, a sum of 33 uniforms, reaches 14 with probability `2^-69`.
const EVAL_MOD_INTERVAL: usize = 14;
const EVAL_MOD_DEGREE: usize = 15;
/// One level fewer leaves 3 bits on the unit circle.
const EVAL_MOD_SQUARINGS: usize = 4;

/// A narrower key outruns the extra row up to this cap on a gadget digit, in limbs.
const MAX_DIGIT_LIMBS: usize = 8;

/// The bootstrap, the widths around the convolutions and the layouts of the activations.
#[derive(Clone)]
pub struct Params {
    preset: BootstrappingPreset,
    layouts: Layouts,
}

impl Params {
    pub fn new() -> Result<Self> {
        let preset = n16_d35_k685_p19_s2c_ffr_ci()?
            .with_slots_to_coeffs(S2C_SCHEDULE_N16)?
            .with_eval_mod_ring(EvalModRing::Standard)?
            .with_eval_mod_exponential(EVAL_MOD_INTERVAL, EVAL_MOD_DEGREE, EVAL_MOD_SQUARINGS)?
            .with_scales(SCALES)?
            .with_input_headroom(S2C_HEADROOM)?;
        // What a layer spends past the bootstrap: the ReLU's levels and the factors of SlotsToCoeffs, the last one
        // holding the convolution.
        let preset = preset.with_output_k(preset.input_k() + RELU_LEVELS * preset.log_delta())?;
        let preset = preset
            .with_dsizes(MAX_DIGIT_LIMBS, preset.dense_to_sparse_dsize())?
            .with_log_n(LOG_N, C2S_SCHEDULE, S2C_SCHEDULE)?
            .with_base2k(BASE2K)?;
        let layouts = Layouts::new(preset.n())?;
        Ok(Self { preset, layouts })
    }

    pub fn preset(&self) -> &BootstrappingPreset {
        &self.preset
    }

    pub(crate) fn layouts(&self) -> &Layouts {
        &self.layouts
    }

    pub(crate) fn log_delta_w(&self) -> usize {
        LOG_DELTA_W
    }

    /// Width of the SlotsToCoeffs input.
    pub(crate) fn in_k(&self) -> usize {
        self.preset.input_k()
    }

    /// Width of a layer's output, the raise input.
    pub(crate) fn level0_k(&self) -> usize {
        self.preset.log_modulus()
    }

    /// Width of the logits.
    pub(crate) fn logits_k(&self) -> usize {
        self.level0_k() + LOG_LOGIT_RANGE
    }

    /// The label of an activation, in the real slots or, between SlotsToCoeffs and the raise, in the coefficients.
    pub(crate) fn activation_meta(&self) -> GLWEPlaintextMeta {
        GLWEPlaintextMeta {
            scale: Scale::Log(self.preset.log_delta()),
            slots: SlotsKind::Real,
            log_sparsity: 0,
            slot_layout: SlotLayout::Full,
        }
    }
}

/// Where each activation of the network sits in the coefficients, a channel per class of the coefficients a
/// convolution tells apart: a stage's step is its channel count, and the logits' the stride of the head times stage
/// 3's.
#[derive(Clone)]
pub(crate) struct Layouts {
    /// The image, three channels of the first stage.
    pub(crate) image: CoeffLayout,
    /// The stages: 16 channels of 32 x 32 on every coefficient of the lower half, 32 of 16 x 16, 64 of 8 x 8.
    pub(crate) stages: [CoeffLayout; 3],
    /// The ten logits, at the coefficients 0 to 9.
    pub(crate) logits: CoeffLayout,
}

impl Layouts {
    pub(crate) fn new(n: usize) -> Result<Self> {
        let grid = |side, step, channels: usize| CoeffLayout::new(n, side, side, ROW_STRIDE, step, (0..channels).collect());
        Ok(Self {
            image: grid(32, 16, 3)?,
            stages: [grid(32, 16, 16)?, grid(16, 32, 32)?, grid(8, 64, 64)?],
            logits: grid(1, 512, 10)?,
        })
    }
}
