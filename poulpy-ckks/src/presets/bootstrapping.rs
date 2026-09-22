//! Default parameter sets for ModUp/EvalMod bootstrapping.
//!
//! Presets bundle the circuit plan with its ciphertext widths, secret weights,
//! and physical evaluation-key layouts. This keeps modulus accounting and key
//! bounds attached to the recipe that requires them.
//!
//! ## Naming
//!
//! A preset is named by what it offers, one token per axis, in the order an
//! application chooses them: `n{log_n}_d{log_delta}_k{output_k}_p{log2_precision}_{circuit}`.
//!
//! - `n`: ring-degree exponent, which with the secret weights fixes the modulus bounds;
//! - `d`: input scale exponent the caller must arrive at;
//! - `k`: output width in bits, at the input scale. The usable budget is
//!   `output_k - input_k`.
//!   S2C-first also reserves the pre-ModUp transform consumption in `input_k`;
//! - `p`: guaranteed output precision in bits (see [`BootstrappingPreset::log2_precision`]);
//! - `circuit`: `c2s` (C2S-first) or `s2c` (S2C-first), extended with a suffix for further techniques.
//!
//! `n16_d35_k600_p19_c2s` is thus the C2S-first preset at `N = 2^16` for inputs at
//! scale `2^35`, producing 600-bit ciphertexts at scale `2^35` with at least 19 bits of precision.
//! CI presets use the `ci_` prefix; `n` then denotes the CI degree and real-slot count.

use anyhow::{Context, Result, ensure};
use poulpy_core::layouts::{
    Base2K, Degree, Dnum, Dsize, GGLWEInfos, GGLWELayout, GLWEAutomorphismKeyLayout, GLWELayout, GLWESwitchingKeyLayout,
    GLWETensorKeyLayout, LWEInfos, Rank, TorusPrecision,
};

use crate::layouts::CIBootstrappingKeysLayout;
use crate::layouts::CKKSRingKind;
use crate::{
    CKKSLayout, CKKSMeta, CoeffsMeta, SlotsKind,
    layouts::{
        BootstrappingKeysLayout, BootstrappingPipeline, BootstrappingPlan, BootstrappingTechniques, DFTOutputFormat, DFTPlan,
        DFTType, EncapsulationKeysLayout, EvalModPlan, EvalModType, SparseSecretEncapsulation,
    },
    polynomial::SplitStrategy,
};

const C2S_SCHEDULE: [(usize, usize); 4] = [(4, 8192), (4, 512), (4, 32), (3, 4)];
const S2C_SCHEDULE: [(usize, usize); 4] = [(3, 4), (4, 32), (4, 512), (4, 8192)];

#[derive(Clone, Copy, Debug)]
struct EvalModSpec {
    eval_mod_type: EvalModType,
    degree: usize,
    interval: usize,
    log_interval_reduction: usize,
    inverse_degree: Option<usize>,
    scaling: Option<f64>,
    split_strategy: SplitStrategy,
    coeffs_log_delta: usize,
    coeffs_log_budget: usize,
    log_delta: usize,
}

#[derive(Clone, Copy, Debug)]
struct PresetSpec {
    name: &'static str,
    log_n: usize,
    base2k: usize,
    rank: usize,
    log_delta: usize,
    output_k: usize,
    log2_precision: usize,
    dense_secret_hamming_weight: usize,
    sparse_secret_hamming_weight: usize,
    max_dense_modulus: usize,
    max_sparse_modulus: usize,
    key_dsize: usize,
    dense_to_sparse_dsize: usize,
    pipeline: BootstrappingPipeline,
    log_msg_ratio: usize,
    c2s_schedule: &'static [(usize, usize)],
    c2s_guard_bits: usize,
    c2s_log_delta: usize,
    c2s_log_budget: usize,
    s2c_schedule: &'static [(usize, usize)],
    s2c_log_delta: usize,
    s2c_log_budget: usize,
    eval_mod: EvalModSpec,
}

/// A complete CKKS bootstrapping parameter set.
///
/// The input and output layouts share the same scale. The net budget is the
/// output log-budget minus the input log-budget.
/// The output cannot be drained further: for an S2C-first preset the input
/// width includes the SlotsToCoeffs evaluated before ModUp, so a larger tail of
/// the output is reserved than for a C2S-first preset.
/// The bootstrap allocation is wider than the logical output because it
/// also carries the post-ModUp circuit.
#[derive(Clone, Debug)]
pub struct BootstrappingPreset<K = BootstrappingKeysLayout> {
    spec: PresetSpec,
    n: usize,
    plan: BootstrappingPlan,
    keys_layout: K,
    ring_kind: CKKSRingKind,
    input_k: usize,
    output_k: usize,
    bootstrap_k: usize,
}

impl<K> BootstrappingPreset<K> {
    /// Stable descriptive name of the preset.
    pub fn name(&self) -> &'static str {
        self.spec.name
    }

    /// Ring-degree exponent (`N = 2^log_n`).
    pub fn log_n(&self) -> usize {
        self.n.ilog2() as usize
    }

    /// Ring degree.
    pub fn n(&self) -> usize {
        self.n
    }

    /// Limb radix exponent.
    pub fn base2k(&self) -> usize {
        self.spec.base2k
    }

    /// Input ciphertext scale exponent.
    pub fn log_delta(&self) -> usize {
        self.spec.log_delta
    }

    /// EvalMod message-ratio exponent.
    pub fn log_msg_ratio(&self) -> usize {
        self.spec.log_msg_ratio
    }

    /// Modulus immediately around ModUp, after any pre-ModUp stage.
    pub fn log_modulus(&self) -> usize {
        self.spec.log_delta + self.spec.log_msg_ratio
    }

    /// Advertised output precision in bits: the minimum slot-wise precision
    /// measured by the preset conformance tests at the nominal shape with
    /// `f64` DFT matrices, rounded down.
    pub fn log2_precision(&self) -> usize {
        self.spec.log2_precision
    }

    /// Hamming weight of each dense secret.
    pub fn dense_secret_hamming_weight(&self) -> usize {
        self.spec.dense_secret_hamming_weight
    }

    /// Hamming weight of the ephemeral sparse ModUp secret.
    pub fn sparse_secret_hamming_weight(&self) -> usize {
        self.spec.sparse_secret_hamming_weight
    }

    /// Configured modulus bound under the bootstrap circuit's dense secret.
    pub fn max_dense_modulus(&self) -> usize {
        self.spec.max_dense_modulus
    }

    /// Configured modulus bound for objects under the ephemeral sparse secret.
    pub fn max_sparse_modulus(&self) -> usize {
        self.spec.max_sparse_modulus
    }

    /// Gadget digit size (in limbs) of the automorphism, tensor, and sparse-to-dense keys.
    pub fn key_dsize(&self) -> usize {
        self.spec.key_dsize
    }

    /// Gadget digit size (in limbs) of the dense-to-sparse key.
    pub fn dense_to_sparse_dsize(&self) -> usize {
        self.spec.dense_to_sparse_dsize
    }

    /// Width required by an input ciphertext, i.e. the width the application
    /// must stop consuming at. For an S2C-first preset this includes the
    /// SlotsToCoeffs bits consumed below ModUp.
    pub fn input_k(&self) -> usize {
        self.input_k
    }

    /// Width after bootstrapping, at the input scale.
    pub fn output_k(&self) -> usize {
        self.output_k
    }

    /// Physical width at ModUp and required bootstrap allocation width.
    pub fn bootstrap_k(&self) -> usize {
        self.bootstrap_k
    }

    /// Validated circuit plan.
    pub fn plan(&self) -> &BootstrappingPlan {
        &self.plan
    }

    /// Evaluation-key layouts sized for this preset.
    pub fn keys_layout(&self) -> &K {
        &self.keys_layout
    }

    /// Layout of a ciphertext accepted by the preset.
    pub fn input_layout(&self) -> CKKSLayout {
        self.ciphertext_layout(self.input_k)
    }

    /// Layout to allocate for the bootstrap destination.
    ///
    /// Evaluation narrows its logical width to [`Self::output_k`], while the
    /// backing allocation retains this bootstrap capacity.
    pub fn bootstrap_layout(&self) -> CKKSLayout {
        self.ciphertext_layout(self.bootstrap_k)
    }

    /// Logical layout produced by the bootstrap, at the input scale.
    pub fn output_layout(&self) -> CKKSLayout {
        self.ciphertext_layout(self.output_k)
    }

    fn ciphertext_layout(&self, k: usize) -> CKKSLayout {
        CKKSLayout {
            ring_kind: self.ring_kind,
            glwe_layout: GLWELayout {
                n: Degree(self.n as u32),
                base2k: Base2K(self.spec.base2k as u32),
                k: TorusPrecision(k as u32),
                rank: Rank(self.spec.rank as u32),
            },
            meta: CKKSMeta {
                log_delta: self.spec.log_delta,
                log_sparsity: 0,
                slots: match self.ring_kind {
                    CKKSRingKind::Standard => SlotsKind::Complex,
                    CKKSRingKind::ConjugateInvariant => SlotsKind::Real,
                },
            },
        }
    }
}

impl BootstrappingPreset {
    /// Re-derives the preset at limb radix `base2k`.
    ///
    /// The plan and every bit width are unchanged; the ciphertext and key
    /// layouts are rebuilt and re-validated against the modulus bounds, so a
    /// radix whose key shapes overflow them is rejected.
    pub fn with_base2k(&self, base2k: usize) -> Result<Self> {
        build(PresetSpec { base2k, ..self.spec }, CKKSRingKind::Standard)
    }

    /// Re-derives the preset with gadget digit sizes `key_dsize` for the
    /// high-modulus keys and `dense_to_sparse_dsize` for the dense-to-sparse
    /// key, re-validating the key moduli against the bounds.
    pub fn with_dsizes(&self, key_dsize: usize, dense_to_sparse_dsize: usize) -> Result<Self> {
        build(
            PresetSpec {
                key_dsize,
                dense_to_sparse_dsize,
                ..self.spec
            },
            CKKSRingKind::Standard,
        )
    }
}

/// C2S-first full-slot preset at `N = 2^16` for inputs at scale `2^35`,
/// producing 600-bit ciphertexts at scale `2^35` with at least 19 bits of precision.
///
/// Uses an optimized Han–Ki EvalMod. The input and raised widths are 40 and
/// 1427 bits. The output has 560 usable bits (16 levels) before reaching the
/// 40-bit input width; the bootstrap restores scale `2^35` automatically.
pub fn n16_d35_k600_p19_c2s() -> Result<BootstrappingPreset> {
    build(
        PresetSpec {
            name: "n16_d35_k600_p19_c2s",
            log_n: 16,
            base2k: 52,
            rank: 1,
            log_delta: 35,
            output_k: 600,
            log2_precision: 19,
            dense_secret_hamming_weight: 1024,
            sparse_secret_hamming_weight: 32,
            max_dense_modulus: 1714,
            max_sparse_modulus: 120,
            key_dsize: 4,
            dense_to_sparse_dsize: 1,
            pipeline: BootstrappingPipeline::C2SFirst,
            log_msg_ratio: 5,
            c2s_schedule: &C2S_SCHEDULE,
            c2s_guard_bits: 0,
            c2s_log_delta: 50,
            c2s_log_budget: 2,
            s2c_schedule: &S2C_SCHEDULE,
            s2c_log_delta: 35,
            s2c_log_budget: 2,
            eval_mod: optimized_han_ki(),
        },
        CKKSRingKind::Standard,
    )
}

/// S2C-first full-slot preset at `N = 2^16` for inputs at scale `2^35`,
/// producing 720-bit ciphertexts with at least 19 bits of precision.
///
/// Uses an optimized Han–Ki EvalMod. The initial S2C is evaluated below ModUp,
/// so the input width is 160 bits (the 48-bit ModUp modulus plus 112 bits of
/// SlotsToCoeffs) and the raised width 1382 bits, including six C2S guard bits.
/// The application must hand the ciphertext back at 160 bits: 560 bits (16 rescales at the input
/// scale) are usable, the same budget as the C2S-first preset despite the larger `k`.
pub fn n16_d35_k720_p19_s2c() -> Result<BootstrappingPreset> {
    build(s2c_spec(), CKKSRingKind::Standard)
}

fn s2c_spec() -> PresetSpec {
    PresetSpec {
        name: "n16_d35_k720_p19_s2c",
        log_n: 16,
        base2k: 52,
        rank: 1,
        log_delta: 35,
        output_k: 720,
        log2_precision: 19,
        dense_secret_hamming_weight: 1024,
        sparse_secret_hamming_weight: 32,
        max_dense_modulus: 1714,
        max_sparse_modulus: 120,
        key_dsize: 4,
        dense_to_sparse_dsize: 1,
        pipeline: BootstrappingPipeline::S2CFirst,
        log_msg_ratio: 13,
        c2s_schedule: &C2S_SCHEDULE,
        c2s_guard_bits: 6,
        c2s_log_delta: 48,
        c2s_log_budget: 3,
        s2c_schedule: &S2C_SCHEDULE,
        s2c_log_delta: 28,
        s2c_log_budget: 2,
        eval_mod: optimized_han_ki(),
    }
}

/// C2S-first full-slot preset at `N = 2^15` for inputs at scale `2^35`,
/// producing 180-bit ciphertexts with at least 18 bits of precision.
///
/// The input and raised widths are 40 and 780 bits. The output has 140 usable
/// bits (four levels) at the input scale. Total key moduli are bounded by
/// 854 bits for the weight-1024 dense secret and 164 bits for the weight-32 sparse secret.
pub fn n15_d35_k180_p18_c2s() -> Result<BootstrappingPreset> {
    build(
        PresetSpec {
            name: "n15_d35_k180_p18_c2s",
            log_n: 15,
            base2k: 52,
            rank: 1,
            log_delta: 35,
            output_k: 180,
            log2_precision: 18,
            dense_secret_hamming_weight: 1024,
            sparse_secret_hamming_weight: 32,
            max_dense_modulus: 854,
            max_sparse_modulus: 164,
            key_dsize: 1,
            dense_to_sparse_dsize: 1,
            pipeline: BootstrappingPipeline::C2SFirst,
            log_msg_ratio: 5,
            c2s_schedule: &[(7, 2048), (7, 16)],
            c2s_guard_bits: 0,
            c2s_log_delta: 49,
            c2s_log_budget: 2,
            s2c_schedule: &[(7, 16), (7, 2048)],
            s2c_log_delta: 30,
            s2c_log_budget: 2,
            eval_mod: EvalModSpec {
                log_delta: 53,
                ..optimized_han_ki()
            },
        },
        CKKSRingKind::Standard,
    )
}

/// Every standard-ring preset, in a stable order.
pub fn all() -> Result<Vec<BootstrappingPreset>> {
    const PRESETS: &[fn() -> Result<BootstrappingPreset>] = &[n16_d35_k600_p19_c2s, n16_d35_k720_p19_s2c, n15_d35_k180_p18_c2s];
    PRESETS.iter().map(|build| build()).collect()
}

/// Parameters for single or paired CI bootstrapping through a standard ring of twice the degree.
pub type CIBootstrappingPreset = BootstrappingPreset<CIBootstrappingKeysLayout>;

impl CIBootstrappingPreset {
    /// Degree of the standard module used to compile the context and prepare keys.
    pub fn standard_n(&self) -> usize {
        2 * self.n()
    }

    /// Rebuilds all layouts at another radix and validates their modulus bounds.
    pub fn with_base2k(&self, base2k: usize) -> Result<Self> {
        build_ci(
            PresetSpec { base2k, ..self.spec },
            self.keys_layout.standard_to_ci.dsize.as_usize(),
        )
    }

    /// Rebuilds the high-modulus and inbound, dense-to-sparse, and return key digits.
    pub fn with_dsizes(&self, key_dsize: usize, dense_to_sparse_dsize: usize, standard_to_ci_dsize: usize) -> Result<Self> {
        build_ci(
            PresetSpec {
                key_dsize,
                dense_to_sparse_dsize,
                ..self.spec
            },
            standard_to_ci_dsize,
        )
    }
}

/// S2C-first CI preset for `2^15` real slots, scale `2^35`, and 19-bit precision.
/// Single and pair evaluation share the same plan and keys, with 128-bit security bounds.
pub fn ci_n15_d35_k720_p19_s2c() -> Result<CIBootstrappingPreset> {
    ci_s2c(15)
}

/// S2C-first CI preset for `2^16` real slots, scale `2^35`, and 19-bit precision.
/// Single and pair evaluation share the same plan and keys, with 128-bit security bounds.
pub fn ci_n16_d35_k720_p19_s2c() -> Result<CIBootstrappingPreset> {
    ci_s2c(16)
}

/// Both CI presets, in increasing slot-count order.
pub fn all_ci() -> Result<Vec<CIBootstrappingPreset>> {
    Ok(vec![ci_n15_d35_k720_p19_s2c()?, ci_n16_d35_k720_p19_s2c()?])
}

fn ci_s2c(log_n: usize) -> Result<CIBootstrappingPreset> {
    let mut spec = s2c_spec();
    spec.name = if log_n == 15 {
        "ci_n15_d35_k720_p19_s2c"
    } else {
        "ci_n16_d35_k720_p19_s2c"
    };
    spec.log_n = log_n + 1;
    if log_n == 16 {
        // A linear fit in N to the logN=12..16, h=1024 bounds gives 3424 bits.
        // Apply a 20% margin and round down to 100 bits.
        spec.max_dense_modulus = 2700;
        spec.key_dsize = 14;
        // The sparse key's logN guard gains one bit at the doubled degree.
        spec.max_sparse_modulus += 1;
        spec.c2s_schedule = &[(4, 16384), (4, 1024), (4, 64), (4, 4)];
        spec.s2c_schedule = &[(5, 8), (5, 256), (6, 8192)];
        spec.s2c_log_delta = 37;
        spec.c2s_log_delta = 50;
        spec.log_msg_ratio = 14;
    }
    build_ci(spec, 1)
}

fn build_ci(spec: PresetSpec, standard_to_ci_dsize: usize) -> Result<CIBootstrappingPreset> {
    let preset = build(spec, CKKSRingKind::ConjugateInvariant)?;
    let standard_n = 2 * preset.n();
    ensure!(standard_to_ci_dsize > 0, "CI return key dsize must be nonzero");
    let max_ci_modulus = match preset.spec.log_n {
        16 => 854,
        17 => 1714,
        _ => anyhow::bail!("unsupported CI preset degree"),
    };
    let spec = &preset.spec;
    let switching_key = |input_k, dsize| {
        let (dnum, k_aux) = key_shape(spec, input_k, dsize);
        GLWESwitchingKeyLayout {
            n: Degree(standard_n as u32),
            base2k: Base2K(spec.base2k as u32),
            dnum,
            k_aux,
            rank_in: Rank(1),
            rank_out: Rank(1),
            dsize: Dsize(dsize as u32),
        }
    };
    let keys_layout = CIBootstrappingKeysLayout {
        bootstrap_keys: *preset.keys_layout(),
        ci_to_standard: switching_key(preset.input_k(), spec.key_dsize),
        standard_to_ci: switching_key(preset.output_k() + spec.c2s_guard_bits + 1, standard_to_ci_dsize),
    };
    validate_key("CI-to-standard", &keys_layout.ci_to_standard, spec.max_dense_modulus)?;
    validate_key("standard-to-CI", &keys_layout.standard_to_ci, max_ci_modulus)?;
    Ok(BootstrappingPreset {
        spec: preset.spec,
        n: preset.n,
        ring_kind: CKKSRingKind::ConjugateInvariant,
        plan: preset.plan,
        keys_layout,
        input_k: preset.input_k,
        output_k: preset.output_k,
        bootstrap_k: preset.bootstrap_k,
    })
}

const fn optimized_han_ki() -> EvalModSpec {
    EvalModSpec {
        eval_mod_type: EvalModType::CosHKEven,
        degree: 30,
        interval: 16,
        log_interval_reduction: 3,
        inverse_degree: None,
        scaling: None,
        split_strategy: SplitStrategy::MinDepth,
        coeffs_log_delta: 42,
        coeffs_log_budget: 4,
        log_delta: 58,
    }
}

fn build(spec: PresetSpec, ring_kind: CKKSRingKind) -> Result<BootstrappingPreset> {
    ensure!(spec.base2k > 0, "bootstrapping preset base2k must be nonzero");
    ensure!(spec.rank == 1, "bootstrapping presets currently require rank 1");
    ensure!(spec.key_dsize > 0, "bootstrapping preset key dsize must be nonzero");
    ensure!(
        spec.dense_to_sparse_dsize > 0,
        "bootstrapping preset dense-to-sparse dsize must be nonzero"
    );
    let log_n = u32::try_from(spec.log_n).context("bootstrapping preset log_n does not fit u32")?;
    let n = 1usize
        .checked_shl(log_n)
        .context("bootstrapping preset ring degree overflow")?;
    ensure!(n <= u32::MAX as usize, "bootstrapping preset ring degree does not fit u32");
    let slots_to_coeffs = DFTPlan::new(
        DFTType::Decode,
        spec.s2c_schedule.to_vec(),
        DFTOutputFormat::SplitRealAndImag,
        CoeffsMeta::from_delta_budget(spec.s2c_log_delta, spec.s2c_log_budget),
    )?
    .with_scaling(match spec.pipeline {
        BootstrappingPipeline::C2SFirst => (spec.log_msg_ratio as f64).exp2(),
        BootstrappingPipeline::S2CFirst => 0.5,
    })?;
    let coeffs_to_slots = DFTPlan::new(
        DFTType::Encode,
        spec.c2s_schedule.to_vec(),
        DFTOutputFormat::SplitRealAndImag,
        CoeffsMeta::from_delta_budget(spec.c2s_log_delta, spec.c2s_log_budget),
    )?;
    let plan = BootstrappingPlan::new(
        spec.pipeline,
        BootstrappingTechniques {
            sparse_secret_encapsulation: Some(SparseSecretEncapsulation {
                hamming_weight: spec.sparse_secret_hamming_weight,
            }),
            eval_round_plus: None,
        },
        coeffs_to_slots,
        EvalModPlan {
            eval_mod_type: spec.eval_mod.eval_mod_type,
            log_msg_ratio: spec.log_msg_ratio,
            f_mod_degree: spec.eval_mod.degree,
            f_mod_interval: spec.eval_mod.interval,
            f_mod_log_interval_reduction: spec.eval_mod.log_interval_reduction,
            f_mod_inv_degree: spec.eval_mod.inverse_degree,
            scaling: spec.eval_mod.scaling,
            split_strategy: spec.eval_mod.split_strategy,
            coeffs_meta: CoeffsMeta::from_delta_budget(spec.eval_mod.coeffs_log_delta, spec.eval_mod.coeffs_log_budget),
            f_mod_log_delta: spec.eval_mod.log_delta,
        },
        slots_to_coeffs,
    )?;
    let plan = if spec.c2s_guard_bits == 0 {
        plan
    } else {
        plan.with_c2s_guard_bits(spec.c2s_guard_bits)?
    };
    let log_slots = spec
        .log_n
        .checked_sub(1)
        .context("bootstrapping preset log_n must be positive")?;
    ensure!(
        plan.coeffs_to_slots().log_slots() == log_slots && plan.slots_to_coeffs().log_slots() == log_slots,
        "bootstrapping preset DFT schedules must cover {log_slots} slot layers"
    );

    let log_modulus = spec
        .log_delta
        .checked_add(spec.log_msg_ratio)
        .context("bootstrapping preset input modulus overflow")?;
    let input_k = plan.input_k(log_modulus);
    let output_k = spec.output_k;
    ensure!(
        output_k >= input_k,
        "bootstrapping preset output width {output_k} is below its input width {input_k}"
    );
    let bootstrap_k = plan.bootstrap_k(
        output_k + usize::from(ring_kind == CKKSRingKind::ConjugateInvariant),
        spec.log_delta,
    );
    let keys_layout = keys_layout(&spec, n, bootstrap_k, log_modulus);

    validate_modulus_bounds(&spec, bootstrap_k, &keys_layout)?;
    Ok(BootstrappingPreset {
        spec,
        n: if ring_kind == CKKSRingKind::ConjugateInvariant {
            n / 2
        } else {
            n
        },
        plan,
        keys_layout,
        ring_kind,
        input_k,
        output_k,
        bootstrap_k,
    })
}

fn keys_layout(spec: &PresetSpec, n: usize, bootstrap_k: usize, log_modulus: usize) -> BootstrappingKeysLayout {
    let (dnum, k_aux) = key_shape(spec, bootstrap_k, spec.key_dsize);
    let (dense_to_sparse_dnum, dense_to_sparse_k_aux) = key_shape(spec, log_modulus, spec.dense_to_sparse_dsize);
    let n = Degree(n as u32);
    let base2k = Base2K(spec.base2k as u32);
    let rank = Rank(spec.rank as u32);
    let dsize = Dsize(spec.key_dsize as u32);
    let dense_to_sparse_dsize = Dsize(spec.dense_to_sparse_dsize as u32);
    let high_modulus_switch = GLWESwitchingKeyLayout {
        n,
        base2k,
        dnum,
        k_aux,
        rank_in: rank,
        rank_out: rank,
        dsize,
    };

    BootstrappingKeysLayout {
        automorphism_key: GLWEAutomorphismKeyLayout {
            n,
            base2k,
            dnum,
            k_aux,
            rank,
            dsize,
        },
        tensor_key: GLWETensorKeyLayout {
            n,
            base2k,
            dnum,
            k_aux,
            rank,
            dsize,
        },
        encapsulation: Some(EncapsulationKeysLayout {
            dense_to_sparse: GLWESwitchingKeyLayout {
                n,
                base2k,
                dnum: dense_to_sparse_dnum,
                k_aux: dense_to_sparse_k_aux,
                rank_in: rank,
                rank_out: rank,
                dsize: dense_to_sparse_dsize,
            },
            sparse_to_dense: high_modulus_switch,
        }),
    }
}

/// Gadget shape of a key covering an input of `input_k` bits: the digit count
/// from [`GGLWELayout::dnum_for_input`] and the preset's guard convention
/// `dsize * base2k + log_n`.
fn key_shape(spec: &PresetSpec, input_k: usize, dsize: usize) -> (Dnum, TorusPrecision) {
    let dnum = GGLWELayout::dnum_for_input(
        Base2K(spec.base2k as u32),
        TorusPrecision(input_k as u32),
        Dsize(dsize as u32),
    );
    (dnum, TorusPrecision((dsize * spec.base2k + spec.log_n) as u32))
}

/// Gadget precision of `key` (see [`GGLWELayout::gadget_k`]).
fn gadget_k<K: GGLWEInfos>(key: &K) -> usize {
    key.gglwe_layout().gadget_k().as_usize()
}

fn validate_modulus_bounds(spec: &PresetSpec, bootstrap_k: usize, keys: &BootstrappingKeysLayout) -> Result<()> {
    ensure!(
        bootstrap_k <= spec.max_dense_modulus,
        "bootstrap modulus {bootstrap_k} exceeds dense-secret limit {}",
        spec.max_dense_modulus
    );
    validate_key("automorphism", &keys.automorphism_key, spec.max_dense_modulus)?;
    validate_key("tensor", &keys.tensor_key, spec.max_dense_modulus)?;
    let encapsulation = keys
        .encapsulation
        .as_ref()
        .context("bootstrapping preset is missing encapsulation keys")?;
    validate_key("dense-to-sparse", &encapsulation.dense_to_sparse, spec.max_sparse_modulus)?;
    validate_key("sparse-to-dense", &encapsulation.sparse_to_dense, spec.max_dense_modulus)
}

fn validate_key<K: GGLWEInfos + LWEInfos>(name: &str, key: &K, limit: usize) -> Result<()> {
    let rounded_k = gadget_k(key);
    ensure!(rounded_k <= limit, "{name} key gadget modulus {rounded_k} exceeds {limit}");
    ensure!(
        key.k_aux().as_usize() <= limit,
        "{name} key auxiliary modulus {} exceeds {limit}",
        key.k_aux()
    );
    ensure!(key.k().as_usize() <= limit, "{name} key modulus {} exceeds {limit}", key.k());
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::CKKSInfos;

    #[test]
    fn n15_d35_k180_p18_c2s_is_composable_and_within_bounds() {
        let preset = n15_d35_k180_p18_c2s().unwrap();

        assert_eq!(preset.n(), 1 << 15);
        assert_eq!(preset.plan().pipeline(), BootstrappingPipeline::C2SFirst);
        assert_eq!(preset.plan().eval_mod().eval_mod_type, EvalModType::CosHKEven);
        assert_eq!(preset.plan().eval_mod().consumed_bits(), 424);
        assert_eq!(preset.plan().coeffs_to_slots().consumed_bits(), 98);
        assert_eq!(preset.plan().slots_to_coeffs().consumed_bits(), 60);
        assert_eq!((preset.input_k(), preset.output_k(), preset.bootstrap_k()), (40, 180, 780));
        assert_eq!(preset.log2_precision(), 18);
        assert_eq!(preset.output_layout().log_delta(), 35);
        assert_eq!(preset.input_layout().meta(), preset.output_layout().meta());
        assert_eq!(
            preset.output_layout().log_budget() - preset.input_layout().log_budget(),
            4 * preset.log_delta()
        );
        assert_eq!(
            (preset.dense_secret_hamming_weight(), preset.max_dense_modulus()),
            (1024, 854)
        );
        assert_eq!(
            (preset.sparse_secret_hamming_weight(), preset.max_sparse_modulus()),
            (32, 164)
        );

        let keys = preset.keys_layout();
        let encapsulation = keys.encapsulation.as_ref().unwrap();
        assert_eq!(keys.automorphism_key.dsize.as_usize(), 1);
        assert_eq!(gadget_k(&keys.automorphism_key), 780);
        assert_eq!(keys.automorphism_key.k_aux.as_usize(), 67);
        assert_eq!(keys.automorphism_key.k().as_usize(), 847);
        assert_eq!(keys.tensor_key.k().as_usize(), 847);
        assert_eq!(encapsulation.sparse_to_dense.k().as_usize(), 847);
        assert_eq!(encapsulation.dense_to_sparse.k().as_usize(), 119);
        assert!(preset.with_dsizes(2, 1).is_err());
        assert!(preset.with_dsizes(1, 2).is_err());
        assert!(preset.with_base2k(19).unwrap().with_dsizes(3, 1).is_err());
    }

    #[test]
    fn n16_d35_k600_p19_c2s_is_composable_and_within_bounds() {
        let preset = n16_d35_k600_p19_c2s().unwrap();

        assert_eq!(preset.plan().pipeline(), BootstrappingPipeline::C2SFirst);
        assert_eq!(preset.plan().eval_mod().eval_mod_type, EvalModType::CosHKEven);
        assert_eq!(preset.plan().eval_mod().consumed_bits(), 464);
        assert_eq!(preset.plan().coeffs_to_slots().consumed_bits(), 200);
        assert_eq!(preset.plan().slots_to_coeffs().consumed_bits(), 140);
        assert_eq!((preset.input_k(), preset.output_k(), preset.bootstrap_k()), (40, 600, 1427));
        assert_eq!(preset.log2_precision(), 19);
        assert_eq!(preset.output_layout().log_delta(), 35);
        assert_eq!(preset.input_layout().meta(), preset.output_layout().meta());
        // 16 rescales at the input scale before the next bootstrap.
        assert_eq!(
            preset.output_layout().log_budget() - preset.input_layout().log_budget(),
            16 * preset.log_delta()
        );

        assert_layouts_within_bounds(&preset);
    }

    #[test]
    fn n16_d35_k720_p19_s2c_is_composable_and_within_bounds() {
        let preset = n16_d35_k720_p19_s2c().unwrap();

        assert_eq!(preset.plan().pipeline(), BootstrappingPipeline::S2CFirst);
        assert_eq!(preset.plan().eval_mod().eval_mod_type, EvalModType::CosHKEven);
        assert_eq!(preset.plan().eval_mod().consumed_bits(), 464);
        assert_eq!(preset.plan().coeffs_to_slots().consumed_bits(), 192);
        assert_eq!(preset.plan().c2s_guard_bits(), 6);
        assert_eq!(preset.plan().slots_to_coeffs().consumed_bits(), 112);
        assert_eq!((preset.input_k(), preset.output_k(), preset.bootstrap_k()), (160, 720, 1382));
        assert_eq!(preset.log2_precision(), 19);
        assert_eq!(
            preset.output_layout().log_budget() - preset.input_layout().log_budget(),
            16 * preset.log_delta()
        );

        assert_layouts_within_bounds(&preset);
    }

    fn assert_layouts_within_bounds(preset: &BootstrappingPreset) {
        let keys = preset.keys_layout();
        let encapsulation = keys.encapsulation.as_ref().unwrap();

        assert_eq!(keys.automorphism_key.dsize.as_usize(), 4);
        assert_eq!(gadget_k(&keys.automorphism_key), 1456);
        assert_eq!(keys.automorphism_key.k_aux.as_usize(), 224);
        assert_eq!(keys.automorphism_key.k().as_usize(), 1680);
        assert_eq!(keys.tensor_key.k().as_usize(), 1680);
        assert_eq!(encapsulation.sparse_to_dense.k().as_usize(), 1680);
        assert!(keys.automorphism_key.k().as_usize() <= preset.max_dense_modulus());

        assert_eq!(encapsulation.dense_to_sparse.dsize.as_usize(), 1);
        assert_eq!(gadget_k(&encapsulation.dense_to_sparse), 52);
        assert_eq!(encapsulation.dense_to_sparse.k_aux.as_usize(), 68);
        assert_eq!(encapsulation.dense_to_sparse.k().as_usize(), 120);
        assert!(encapsulation.dense_to_sparse.k().as_usize() <= preset.max_sparse_modulus());
    }

    #[test]
    fn all_lists_every_preset_once() {
        let names: Vec<&str> = all().unwrap().iter().map(|p| p.name()).collect();
        assert_eq!(
            names,
            ["n16_d35_k600_p19_c2s", "n16_d35_k720_p19_s2c", "n15_d35_k180_p18_c2s"]
        );
    }

    #[test]
    fn ci_presets_cover_both_slot_counts_and_secret_bounds() {
        let presets = all_ci().unwrap();
        assert_eq!(
            presets.iter().map(|p| p.name()).collect::<Vec<_>>(),
            ["ci_n15_d35_k720_p19_s2c", "ci_n16_d35_k720_p19_s2c"]
        );
        for (preset, log_n, ci_limit) in [(presets[0].clone(), 15, 854), (presets[1].clone(), 16, 1714)] {
            let widths = (160, 720, if log_n == 15 { 1383 } else { 1391 });
            assert_eq!(preset.log_n(), log_n);
            assert_eq!(preset.n(), 1 << log_n);
            assert_eq!(preset.standard_n(), 2 * preset.n());
            assert_eq!((preset.input_k(), preset.output_k(), preset.bootstrap_k()), widths);
            assert_eq!(preset.log2_precision(), 19);
            assert_eq!(preset.plan().pipeline(), BootstrappingPipeline::S2CFirst);
            assert_eq!(preset.plan().eval_mod().consumed_bits(), 464);
            assert_eq!(preset.plan().coeffs_to_slots().log_slots(), log_n);
            assert_eq!(preset.plan().slots_to_coeffs().log_slots(), log_n);
            assert_eq!(
                (preset.plan().pre_mod_up_consumed_bits(), preset.log_modulus()),
                if log_n == 15 { (112, 48) } else { (111, 49) }
            );
            assert_eq!(preset.output_k() - preset.input_k(), 16 * preset.log_delta());
            assert!(preset.keys_layout().standard_to_ci.k().as_usize() <= ci_limit);
            for layout in [preset.input_layout(), preset.output_layout(), preset.bootstrap_layout()] {
                assert_eq!(layout.ring_kind, CKKSRingKind::ConjugateInvariant);
                assert_eq!(layout.glwe_layout.n.as_usize(), preset.n());
                assert_eq!(layout.meta.slots, SlotsKind::Real);
                assert_eq!(layout.meta.log_sparsity, 0);
                assert_eq!(layout.meta.log_delta, 35);
            }
            assert_ci_key_bounds(&preset);
            let fft = preset.with_base2k(19).unwrap().with_dsizes(7, 1, 1).unwrap();
            assert_eq!(fft.base2k(), 19);
            assert_eq!((fft.input_k(), fft.output_k(), fft.bootstrap_k()), widths);
            assert_ci_key_bounds(&fft);
            assert!(preset.with_base2k(0).is_err());
            assert!(preset.with_dsizes(0, 1, 1).is_err());
            assert!(preset.with_dsizes(3, 0, 1).is_err());
            assert!(preset.with_dsizes(3, 1, 0).is_err());
            assert!(preset.with_dsizes(64, 1, 1).is_err());
            assert!(preset.with_dsizes(3, 4, 1).is_err());
        }
        assert!(presets[0].with_dsizes(4, 1, 2).is_ok());
        assert!(presets[0].with_dsizes(4, 1, 3).is_err());
        assert!(presets[1].with_dsizes(4, 1, 3).is_ok());
    }

    fn assert_ci_key_bounds(preset: &CIBootstrappingPreset) {
        let keys = preset.keys_layout();
        let enc = keys.bootstrap_keys.encapsulation.as_ref().unwrap();
        for (key, input_k, limit) in [
            (
                keys.ci_to_standard.gglwe_layout(),
                preset.input_k(),
                preset.max_dense_modulus(),
            ),
            (
                keys.standard_to_ci.gglwe_layout(),
                preset.output_k() + preset.plan().c2s_guard_bits() + 1,
                if preset.log_n() == 15 { 854 } else { 1714 },
            ),
            (
                keys.bootstrap_keys.automorphism_key.gglwe_layout(),
                preset.bootstrap_k(),
                preset.max_dense_modulus(),
            ),
            (
                keys.bootstrap_keys.tensor_key.gglwe_layout(),
                preset.bootstrap_k(),
                preset.max_dense_modulus(),
            ),
            (
                enc.sparse_to_dense.gglwe_layout(),
                preset.bootstrap_k(),
                preset.max_dense_modulus(),
            ),
            (
                enc.dense_to_sparse.gglwe_layout(),
                preset.log_modulus(),
                preset.max_sparse_modulus(),
            ),
        ] {
            assert_eq!(key.n.as_usize(), preset.standard_n());
            assert_eq!(key.base2k.as_usize(), preset.base2k());
            assert!(key.gadget_k().as_usize() >= input_k);
            assert_eq!(
                key.k_aux.as_usize(),
                key.dsize.as_usize() * preset.base2k() + preset.log_n() + 1
            );
            assert!(key.k().as_usize() <= limit);
        }
    }

    #[test]
    fn rederived_key_shape_keeps_widths_and_revalidates() {
        for (preset, fft_dsize) in [
            (n16_d35_k600_p19_c2s().unwrap(), 7),
            (n16_d35_k720_p19_s2c().unwrap(), 7),
            (n15_d35_k180_p18_c2s().unwrap(), 2),
        ] {
            let widths = (preset.input_k(), preset.output_k(), preset.bootstrap_k());

            let fft = preset.with_base2k(19).unwrap().with_dsizes(fft_dsize, 1).unwrap();
            assert_eq!((fft.input_k(), fft.output_k(), fft.bootstrap_k()), widths);
            assert_eq!(
                (fft.base2k(), fft.key_dsize(), fft.dense_to_sparse_dsize()),
                (19, fft_dsize, 1)
            );
            assert_eq!(fft.bootstrap_layout().glwe_layout.base2k.as_usize(), 19);
            let keys = fft.keys_layout();
            assert_eq!(
                keys.automorphism_key.dnum.as_usize(),
                preset.bootstrap_k().div_ceil(fft_dsize * 19)
            );
            assert_eq!(keys.automorphism_key.k_aux.as_usize(), fft_dsize * 19 + preset.log_n());
            assert!(keys.automorphism_key.k().as_usize() <= fft.max_dense_modulus());

            let small_key = &keys.encapsulation.as_ref().unwrap().dense_to_sparse;
            assert_eq!(
                small_key.k().as_usize(),
                preset.log_modulus().div_ceil(19) * 19 + 19 + preset.log_n()
            );
            assert!(small_key.k().as_usize() <= fft.max_sparse_modulus());

            // Oversized digits are rejected for both secret bounds.
            assert!(preset.with_dsizes(10, 1).is_err());
            assert!(preset.with_dsizes(4, 2).is_err());
            assert!(preset.with_dsizes(4, 3).is_err());
        }
    }
}
