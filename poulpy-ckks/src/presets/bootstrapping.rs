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
//! - `n`: ring-degree exponent, which with the secret weights fixes the modulus bounds
//!   (see [Security](#security));
//! - `d`: input scale exponent the caller must arrive at;
//! - `k`: output width in bits, at the input scale. The usable budget is
//!   `output_k - input_k`.
//!   S2C-first also reserves the pre-ModUp transform consumption in `input_k`;
//! - `p`: guaranteed output precision in bits (see [`BootstrappingPreset::log2_precision`]);
//! - `circuit`: `c2s` (C2S-first) or `s2c` (S2C-first), extended with a suffix for further techniques.
//!
//! `n16_d35_k636_p20_c2s` is thus the C2S-first preset at `N = 2^16` for inputs at
//! scale `2^35`, producing 636-bit ciphertexts at scale `2^35` with at least 20 bits of precision.
//!
//! ## Security
//!
//! Each secret of a preset comes with the largest modulus `q = 2^k`, in bits, that
//! any object under it may use for 128 bits of security: a ternary secret of that
//! Hamming weight, an error of deviation 3.2, at the preset's ring degree. The
//! values are estimates, not proofs, read from the 128-bit table of
//! [sparse-key-estimate](https://github.com/jdumezy/sparse-key-estimate/blob/aa2c40d4eee4551e6fba06c55d7432f6918e1597/Precomputed-Tables/128bits_security.md):
//!
//! | `N` | dense secret, weight 1024 | sparse secret, weight 32 |
//! | --- | ---: | ---: |
//! | `2^16` | 1707 | 345 |
//! | `2^15` | 851 | 162 |
//!
//! The tool estimates the primal hybrid attack with a meet-in-the-middle search
//! over rotations of the secret ([ePrint 2026/279](https://eprint.iacr.org/2026/279)),
//! and reproduces Table 1 of [ePrint 2022/024](https://eprint.iacr.org/2022/024) at
//! that paper's sparse-key moduli: 161.8 bits of security at 120 bits for
//! `N = 2^16` (168.5 in the paper), 140.6 at 111 bits for `N = 2^15` (140.9). The
//! 121 and 111 bits of 2022/024 are the sizes of its sparse-secret key, not bounds.
//! At the sparse bounds, the dual and dual-hybrid attacks of the
//! [lattice estimator](https://github.com/malb/lattice-estimator) cost above
//! 2^790, so the primal hybrid sets them.
//!
//! Every object under a secret counts, the keys included: their modulus is the
//! widest in a preset, and a key's error sits at its last bit, so the bound
//! applies to its precision `k`. Building a preset checks every key layout against
//! the bounds of its secrets.

use std::collections::BTreeMap;

use anyhow::{Context, Result, ensure};
use poulpy_core::layouts::{Base2K, Degree, GGLWEInfos, GGLWELayout, GLWELayout, LWEInfos, Rank, TorusPrecision, k_digit_of};

use crate::numerics::CKKSFloat;
use crate::{
    CKKSLayout, CoeffsMeta, GLWEPlaintextMeta, Scale, SlotsKind,
    layouts::{
        BootstrappingKeysLayout, BootstrappingPipeline, BootstrappingPlan, BootstrappingTechniques, DFTOutputFormat, DFTPlan,
        DFTType, EncapsulationKeysLayout, EvalModPlan, EvalModType, SparseSecretEncapsulation,
    },
    polynomial::SplitStrategy,
};

/// Three factors of five layers for the logN16 presets, with the giant steps that
/// need the fewest key switches.
const C2S_SCHEDULE: [(usize, usize); 3] = [(5, 4096), (5, 256), (5, 8)];
/// Three factors of four, five and six layers for the logN16 presets.
const S2C_SCHEDULE: [(usize, usize); 3] = [(4, 4), (5, 128), (6, 4096)];

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

/// A secret of a preset: its Hamming weight and the largest modulus, in bits, an
/// object under it may use (see [Security](self#security)).
#[derive(Clone, Copy, Debug)]
struct SecretBound {
    hamming_weight: usize,
    max_log_modulus: usize,
}

/// The dense secret of the logN16 presets.
const N16_DENSE_SECRET: SecretBound = SecretBound {
    hamming_weight: 1024,
    max_log_modulus: 1707,
};
/// The ephemeral sparse secret of the logN16 presets.
const N16_SPARSE_SECRET: SecretBound = SecretBound {
    hamming_weight: 32,
    max_log_modulus: 345,
};
/// The dense secret of the logN15 preset.
const N15_DENSE_SECRET: SecretBound = SecretBound {
    hamming_weight: 1024,
    max_log_modulus: 851,
};
/// The ephemeral sparse secret of the logN15 preset.
const N15_SPARSE_SECRET: SecretBound = SecretBound {
    hamming_weight: 32,
    max_log_modulus: 162,
};

#[derive(Clone, Copy, Debug)]
struct PresetSpec {
    name: &'static str,
    log_n: usize,
    base2k: usize,
    rank: usize,
    log_delta: usize,
    output_k: usize,
    log2_precision: usize,
    dense_secret: SecretBound,
    sparse_secret: SecretBound,
    /// Digit of the high-modulus keys, all alike; `None` gives each key the
    /// widest digit its use admits.
    key_dsize: Option<usize>,
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
pub struct BootstrappingPreset {
    spec: PresetSpec,
    n: usize,
    plan: BootstrappingPlan,
    keys_layout: BootstrappingKeysLayout,
    input_k: usize,
    output_k: usize,
    bootstrap_k: usize,
}

impl BootstrappingPreset {
    /// Stable descriptive name of the preset.
    pub fn name(&self) -> &'static str {
        self.spec.name
    }

    /// Ring-degree exponent (`N = 2^log_n`).
    pub fn log_n(&self) -> usize {
        self.spec.log_n
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
    /// measured on the reference vector at the nominal shape with `f64` DFT
    /// matrices, rounded down. Pinned by
    /// `test_suite::presets::bootstrapping_presets_meet_precision`.
    pub fn log2_precision(&self) -> usize {
        self.spec.log2_precision
    }

    /// Hamming weight of the dense application secret.
    pub fn dense_secret_hamming_weight(&self) -> usize {
        self.spec.dense_secret.hamming_weight
    }

    /// Hamming weight of the ephemeral sparse ModUp secret.
    pub fn sparse_secret_hamming_weight(&self) -> usize {
        self.spec.sparse_secret.hamming_weight
    }

    /// Modulus bound for objects under the dense secret (see [Security](self#security)).
    pub fn max_dense_modulus(&self) -> usize {
        self.spec.dense_secret.max_log_modulus
    }

    /// Modulus bound for objects under the ephemeral sparse secret (see [Security](self#security)).
    pub fn max_sparse_modulus(&self) -> usize {
        self.spec.sparse_secret.max_log_modulus
    }

    /// Gadget digit size (in limbs) of the keys used at the raised width: the
    /// sparse-to-dense key and the rotation keys of the first CoeffsToSlots
    /// factor. Unless [`Self::with_dsizes`] made every key alike, the keys used
    /// on narrower ciphertexts have wider digits (see [`Self::keys_layout`]).
    pub fn key_dsize(&self) -> usize {
        self.keys_layout.automorphism_key.dsize().as_usize()
    }

    /// Gadget digit size (in limbs) of the dense-to-sparse key.
    pub fn dense_to_sparse_dsize(&self) -> usize {
        self.spec.dense_to_sparse_dsize
    }

    /// Re-derives the preset at limb radix `base2k`.
    ///
    /// The plan and every bit width are unchanged; the ciphertext and key
    /// layouts are rebuilt and re-validated against the modulus bounds, so a
    /// radix whose key shapes overflow them is rejected.
    pub fn with_base2k(&self, base2k: usize) -> Result<Self> {
        build(PresetSpec { base2k, ..self.spec })
    }

    /// Re-derives the preset with one gadget digit size, `key_dsize`, for every
    /// high-modulus key, and `dense_to_sparse_dsize` for the dense-to-sparse
    /// key. Fails for a digit too wide for the modulus bound of its secret.
    pub fn with_dsizes(&self, key_dsize: usize, dense_to_sparse_dsize: usize) -> Result<Self> {
        build(PresetSpec {
            key_dsize: Some(key_dsize),
            dense_to_sparse_dsize,
            ..self.spec
        })
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
    pub fn keys_layout(&self) -> &BootstrappingKeysLayout {
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
            glwe_layout: GLWELayout {
                n: Degree(self.n as u32),
                base2k: Base2K(self.spec.base2k as u32),
                k: TorusPrecision(k as u32),
                rank: Rank(self.spec.rank as u32),
            },
            meta: GLWEPlaintextMeta {
                scale: Scale::Log(self.spec.log_delta),
                slots: SlotsKind::Complex,
                log_sparsity: 0,
            },
        }
    }
}

/// C2S-first full-slot preset at `N = 2^16` for inputs at scale `2^35`,
/// producing 636-bit ciphertexts at scale `2^35` with at least 20 bits of precision.
///
/// Uses an optimized Han–Ki EvalMod and three factors per transform, as the
/// S2C-first preset does. The input and raised widths are 41 and 1381 bits. The
/// output has 595 usable bits (17 levels) before reaching the 41-bit input
/// width; the bootstrap restores scale `2^35` automatically.
pub fn n16_d35_k636_p20_c2s() -> Result<BootstrappingPreset> {
    build(PresetSpec {
        name: "n16_d35_k636_p20_c2s",
        log_n: 16,
        base2k: 52,
        rank: 1,
        log_delta: 35,
        output_k: 636,
        log2_precision: 20,
        dense_secret: N16_DENSE_SECRET,
        sparse_secret: N16_SPARSE_SECRET,
        key_dsize: None,
        dense_to_sparse_dsize: 1,
        pipeline: BootstrappingPipeline::C2SFirst,
        log_msg_ratio: 6,
        c2s_schedule: &C2S_SCHEDULE,
        c2s_guard_bits: 0,
        c2s_log_delta: 51,
        c2s_log_budget: 2,
        s2c_schedule: &S2C_SCHEDULE,
        s2c_log_delta: 35,
        s2c_log_budget: 2,
        eval_mod: optimized_han_ki(),
    })
}

/// S2C-first full-slot preset at `N = 2^16` for inputs at scale `2^35`,
/// producing 733-bit ciphertexts with at least 20 bits of precision.
///
/// Uses an optimized Han–Ki EvalMod and three factors per transform. The initial
/// S2C is evaluated below ModUp, so the input width is 138 bits (the 48-bit ModUp
/// modulus plus 90 bits of SlotsToCoeffs) and the raised width 1353 bits,
/// including six C2S guard bits. The application must hand the ciphertext back at
/// 138 bits: 595 bits (17 rescales at the input scale) are usable, one more than
/// the C2S-first preset despite the larger `k`.
pub fn n16_d35_k733_p20_s2c() -> Result<BootstrappingPreset> {
    build(PresetSpec {
        name: "n16_d35_k733_p20_s2c",
        log_n: 16,
        base2k: 52,
        rank: 1,
        log_delta: 35,
        output_k: 733,
        log2_precision: 20,
        dense_secret: N16_DENSE_SECRET,
        sparse_secret: N16_SPARSE_SECRET,
        key_dsize: None,
        dense_to_sparse_dsize: 1,
        pipeline: BootstrappingPipeline::S2CFirst,
        log_msg_ratio: 13,
        c2s_schedule: &C2S_SCHEDULE,
        c2s_guard_bits: 6,
        c2s_log_delta: 50,
        c2s_log_budget: 3,
        s2c_schedule: &S2C_SCHEDULE,
        s2c_log_delta: 30,
        s2c_log_budget: 2,
        eval_mod: optimized_han_ki(),
    })
}

/// C2S-first full-slot preset at `N = 2^15` for inputs at scale `2^35`,
/// producing 180-bit ciphertexts with at least 18 bits of precision.
///
/// The input and raised widths are 40 and 780 bits. The output has 140 usable
/// bits (four levels) at the input scale. Total key moduli are bounded by
/// 851 bits for the weight-1024 dense secret and 162 bits for the weight-32 sparse secret.
pub fn n15_d35_k180_p18_c2s() -> Result<BootstrappingPreset> {
    build(PresetSpec {
        name: "n15_d35_k180_p18_c2s",
        log_n: 15,
        base2k: 52,
        rank: 1,
        log_delta: 35,
        output_k: 180,
        log2_precision: 18,
        dense_secret: N15_DENSE_SECRET,
        sparse_secret: N15_SPARSE_SECRET,
        key_dsize: None,
        dense_to_sparse_dsize: 1,
        pipeline: BootstrappingPipeline::C2SFirst,
        log_msg_ratio: 5,
        c2s_schedule: &[(7, 2048), (7, 16)],
        c2s_guard_bits: 0,
        c2s_log_delta: 50,
        c2s_log_budget: 2,
        s2c_schedule: &[(7, 16), (7, 2048)],
        s2c_log_delta: 29,
        s2c_log_budget: 2,
        eval_mod: EvalModSpec {
            log_delta: 53,
            ..optimized_han_ki()
        },
    })
}

/// Insecure S2C-first preset at `N = 2^11` for the test suite: the reference
/// S2C-first spec on a ring small enough for a non-ignored test, so the key
/// shapes the presets derive (narrow conjugation and SlotsToCoeffs keys) run
/// end to end. Its secret bounds are placeholders, not security limits; the
/// dense one leaves the 800-bit raised width the margin of a few digits.
#[cfg(feature = "test-utils")]
pub(crate) fn insecure_n11_d35_k180_p20_s2c() -> Result<BootstrappingPreset> {
    const TEST_SECRET: SecretBound = SecretBound {
        hamming_weight: 192,
        max_log_modulus: 1024,
    };
    build(PresetSpec {
        name: "insecure_n11_d35_k180_p20_s2c",
        log_n: 11,
        base2k: 52,
        rank: 1,
        log_delta: 35,
        output_k: 180,
        log2_precision: 20,
        dense_secret: TEST_SECRET,
        sparse_secret: SecretBound {
            hamming_weight: 32,
            ..TEST_SECRET
        },
        key_dsize: None,
        dense_to_sparse_dsize: 1,
        pipeline: BootstrappingPipeline::S2CFirst,
        log_msg_ratio: 13,
        c2s_schedule: &[(4, 4), (3, 4), (3, 4)],
        c2s_guard_bits: 6,
        c2s_log_delta: 50,
        c2s_log_budget: 3,
        s2c_schedule: &[(2, 4), (4, 4), (4, 4)],
        s2c_log_delta: 30,
        s2c_log_budget: 2,
        eval_mod: optimized_han_ki(),
    })
}

/// Every preset, in a stable order.
pub fn all() -> Result<Vec<BootstrappingPreset>> {
    const PRESETS: &[fn() -> Result<BootstrappingPreset>] = &[n16_d35_k636_p20_c2s, n16_d35_k733_p20_s2c, n15_d35_k180_p18_c2s];
    PRESETS.iter().map(|build| build()).collect()
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

fn build(spec: PresetSpec) -> Result<BootstrappingPreset> {
    ensure!(spec.base2k > 0, "bootstrapping preset base2k must be nonzero");
    ensure!(spec.rank == 1, "bootstrapping presets currently require rank 1");
    ensure!(spec.key_dsize != Some(0), "bootstrapping preset key dsize must be nonzero");
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
        BootstrappingPipeline::C2SFirst => (spec.log_msg_ratio as f64).ckks_exp2(),
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
                hamming_weight: spec.sparse_secret.hamming_weight,
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
    let bootstrap_k = plan.bootstrap_k(output_k, spec.log_delta);
    let keys_layout = keys_layout(&spec, &plan, bootstrap_k, input_k)?;

    validate_modulus_bounds(&spec, bootstrap_k, &keys_layout)?;
    Ok(BootstrappingPreset {
        spec,
        n,
        plan,
        keys_layout,
        input_k,
        output_k,
        bootstrap_k,
    })
}

fn keys_layout(
    spec: &PresetSpec,
    plan: &BootstrappingPlan,
    bootstrap_k: usize,
    input_k: usize,
) -> Result<BootstrappingKeysLayout> {
    let (dense, sparse) = (spec.dense_secret.max_log_modulus, spec.sparse_secret.max_log_modulus);
    let wide = key_layout(
        spec,
        "high-modulus",
        bootstrap_k,
        spec.key_dsize.unwrap_or(widest_dsize(spec, bootstrap_k, dense)),
        dense,
    )?;
    let dense_to_sparse = key_layout(
        spec,
        "dense-to-sparse",
        spec.log_delta + spec.log_msg_ratio,
        spec.dense_to_sparse_dsize,
        sparse,
    )?;

    // With no digit imposed, a key only used on narrower ciphertexts takes the
    // widest digit those admit.
    let mut rotation_keys = BTreeMap::new();
    let mut tensor_key = wide;
    // EvalRound+ runs its bypass at widths of its own, which the stage widths
    // below do not model: its keys keep the shape of the raised width.
    if spec.key_dsize.is_none() && plan.coeffs_to_slots_bypass().is_none() {
        for (p, k) in rotation_key_inputs(spec, plan, bootstrap_k, input_k) {
            let layout = key_layout(spec, "rotation", k, widest_dsize(spec, k, dense), dense)?;
            if layout != wide {
                rotation_keys.insert(p, layout.into());
            }
        }
        let k = eval_mod_relinearization_k(spec, plan, bootstrap_k);
        tensor_key = key_layout(spec, "tensor", k, widest_dsize(spec, k, dense), dense)?;
    }

    Ok(BootstrappingKeysLayout {
        automorphism_key: wide.into(),
        rotation_keys,
        tensor_key: tensor_key.into(),
        encapsulation: Some(EncapsulationKeysLayout {
            dense_to_sparse: dense_to_sparse.into(),
            sparse_to_dense: wide.into(),
        }),
    })
}

/// Widest ciphertext, in bits, each rotation key switches in a bootstrap.
///
/// Factor `i` of a transform runs on a ciphertext `i` factor scales below the
/// width the transform starts at: the raised width for CoeffsToSlots, and for
/// SlotsToCoeffs the input width (S2C-first) or what EvalMod leaves
/// (C2S-first). The conjugation follows CoeffsToSlots. Not for a plan with the
/// EvalRound+ bypass.
fn rotation_key_inputs(spec: &PresetSpec, plan: &BootstrappingPlan, bootstrap_k: usize, input_k: usize) -> BTreeMap<i64, usize> {
    let cyclotomic_order = 1i64 << (spec.log_n + 1);
    let after_c2s = bootstrap_k - plan.coeffs_to_slots().consumed_bits();
    let s2c_k = match plan.pipeline() {
        BootstrappingPipeline::S2CFirst => input_k,
        BootstrappingPipeline::C2SFirst => after_c2s - plan.eval_mod().consumed_bits(),
    };

    let mut inputs: BTreeMap<i64, usize> = BTreeMap::new();
    let mut widen = |p: i64, k: usize| {
        let widest = inputs.entry(p).or_insert(k);
        *widest = (*widest).max(k);
    };
    let transforms = [(plan.coeffs_to_slots(), bootstrap_k), (plan.slots_to_coeffs(), s2c_k)];
    for (dft, start) in transforms {
        let log_delta = dft.coeffs_meta().log_delta();
        let factors = dft.galois_elements_per_factor(spec.log_n, cyclotomic_order);
        // The repack rotation of the sparse path belongs to no factor.
        for p in dft.galois_elements(spec.log_n, cyclotomic_order) {
            if !factors.iter().any(|elements| elements.contains(&p)) {
                widen(p, start);
            }
        }
        for (factor, elements) in factors.iter().enumerate() {
            for &p in elements {
                widen(p, start - factor * log_delta);
            }
        }
    }
    widen(-1, after_c2s);
    inputs
}

/// Widest ciphertext EvalMod relinearizes, in bits.
///
/// EvalMod relabels its input at its own scale and keeps its budget, so its
/// widest product is at that budget plus `f_mod_log_delta`. The budget is what
/// CoeffsToSlots leaves of the raised width above the scale the raise labels:
/// EvalMod's own scale (C2S-first), or the ModUp modulus and the guard bits
/// (S2C-first).
fn eval_mod_relinearization_k(spec: &PresetSpec, plan: &BootstrappingPlan, bootstrap_k: usize) -> usize {
    let f_mod_log_delta = plan.eval_mod().f_mod_log_delta;
    let raised_log_delta = match plan.pipeline() {
        BootstrappingPipeline::C2SFirst => f_mod_log_delta,
        BootstrappingPipeline::S2CFirst => spec.log_delta + spec.log_msg_ratio + plan.c2s_guard_bits(),
    };
    bootstrap_k - plan.coeffs_to_slots().consumed_bits() - raised_log_delta + f_mod_log_delta
}

/// Bits a key keeps between its modulus and the input plus one digit.
///
/// At that point the key noise is `sigma * sqrt(n * digits / 12)` units of the
/// input's last bit (see [`poulpy_core::layouts::key_k`]): about `2^9` at
/// `N = 2^16`, `2^7` with this margin. A bootstrapped ciphertext shows it only
/// past its own error, `log_delta - log2_precision` bits above its last bit:
/// 15 or more in the presets, which leaves 8. Measured on the S2C-first preset,
/// the precision starts to drop about 10 bits below this margin. A preset with a
/// narrower gap between scale and precision needs more.
const KEY_NOISE_MARGIN: usize = 2;

/// Widest digit, in limbs, a key under a modulus bound of `limit` bits can
/// switch an input of `input_k` bits with (see [`KEY_NOISE_MARGIN`]).
fn widest_dsize(spec: &PresetSpec, input_k: usize, limit: usize) -> usize {
    let limbs = input_k.div_ceil(spec.base2k).max(1);
    (limit.saturating_sub(input_k + KEY_NOISE_MARGIN) / spec.base2k).clamp(1, limbs)
}

/// Layout of a rank-preserving `dsize`-limb key covering an input of `input_k`
/// bits under a modulus bound of `limit` bits: the digit count from
/// [`GGLWELayout::dnum_for_input`], and behind it a digit and `log_n` bits of
/// guard, or what the bound leaves when that is less. The latter is enough
/// while the input does not fill the last digit, by [`KEY_NOISE_MARGIN`].
fn key_layout(spec: &PresetSpec, name: &str, input_k: usize, dsize: usize, limit: usize) -> Result<GGLWELayout> {
    let base2k = Base2K(spec.base2k as u32);
    let dnum = GGLWELayout::dnum_for_input(TorusPrecision(input_k as u32), k_digit_of(base2k, dsize));
    let digit = dsize * spec.base2k;
    let gadget = dnum.as_usize() * digit;
    let k_aux = (digit + spec.log_n).min(limit.saturating_sub(gadget));
    ensure!(
        gadget + k_aux >= input_k + digit + KEY_NOISE_MARGIN,
        "{name} key: a digit of {dsize} limbs on a {input_k}-bit input needs a modulus of {} bits, above {limit}",
        input_k + digit + KEY_NOISE_MARGIN
    );
    Ok(GGLWELayout {
        n: Degree(1 << spec.log_n),
        base2k,
        dnum,
        k_aux: TorusPrecision(k_aux as u32),
        rank_in: Rank(spec.rank as u32),
        rank_out: Rank(spec.rank as u32),
        k_digit: k_digit_of(base2k, dsize),
        stride: 1,
    })
}

/// Gadget precision of `key` (see [`GGLWELayout::gadget_k`]).
fn gadget_k<K: GGLWEInfos>(key: &K) -> usize {
    key.gglwe_layout().gadget_k().as_usize()
}

fn validate_modulus_bounds(spec: &PresetSpec, bootstrap_k: usize, keys: &BootstrappingKeysLayout) -> Result<()> {
    let (dense, sparse) = (spec.dense_secret.max_log_modulus, spec.sparse_secret.max_log_modulus);
    ensure!(
        bootstrap_k <= dense,
        "bootstrap modulus {bootstrap_k} exceeds dense-secret limit {dense}"
    );
    for key in keys.rotation_key_layouts() {
        validate_key("automorphism", &key, dense)?;
    }
    validate_key("tensor", &keys.tensor_key, dense)?;
    let encapsulation = keys
        .encapsulation
        .as_ref()
        .context("bootstrapping preset is missing encapsulation keys")?;
    validate_key("dense-to-sparse", &encapsulation.dense_to_sparse, sparse)?;
    validate_key("sparse-to-dense", &encapsulation.sparse_to_dense, dense)
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
        assert_eq!(preset.plan().coeffs_to_slots().consumed_bits(), 100);
        assert_eq!(preset.plan().slots_to_coeffs().consumed_bits(), 58);
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
            (1024, 851)
        );
        assert_eq!(
            (preset.sparse_secret_hamming_weight(), preset.max_sparse_modulus()),
            (32, 162)
        );

        let keys = preset.keys_layout();
        let encapsulation = keys.encapsulation.as_ref().unwrap();
        assert_eq!(keys.automorphism_key.dsize().as_usize(), 1);
        assert_eq!(gadget_k(&keys.automorphism_key), 780);
        assert_eq!(keys.automorphism_key.k_aux.as_usize(), 67);
        assert_eq!(keys.automorphism_key.k().as_usize(), 847);
        // EvalMod starts 100 bits below the raised width: room for a 3-limb digit.
        assert_eq!(key_shapes([&keys.tensor_key]), [(3, 5)]);
        assert_eq!(encapsulation.sparse_to_dense.k().as_usize(), 847);
        assert_eq!(encapsulation.dense_to_sparse.k().as_usize(), 119);
        assert!(preset.with_dsizes(2, 1).is_err());
        // The dense-to-sparse input leaves its last digit mostly empty, so two limbs fit.
        assert!(preset.with_dsizes(1, 2).is_ok());
        assert!(preset.with_dsizes(1, 3).is_err());
        assert!(preset.with_base2k(18).unwrap().with_dsizes(4, 1).is_err());
    }

    #[test]
    fn n16_d35_k636_p20_c2s_is_composable_and_within_bounds() {
        let preset = n16_d35_k636_p20_c2s().unwrap();

        assert_eq!(preset.plan().pipeline(), BootstrappingPipeline::C2SFirst);
        assert_eq!(preset.plan().eval_mod().eval_mod_type, EvalModType::CosHKEven);
        assert_eq!(preset.plan().eval_mod().consumed_bits(), 464);
        assert_eq!(preset.plan().coeffs_to_slots().consumed_bits(), 153);
        assert_eq!(preset.plan().slots_to_coeffs().consumed_bits(), 105);
        assert_eq!((preset.input_k(), preset.output_k(), preset.bootstrap_k()), (41, 636, 1381));
        assert_eq!(preset.log2_precision(), 20);
        assert_eq!(preset.output_layout().log_delta(), 35);
        assert_eq!(preset.input_layout().meta(), preset.output_layout().meta());
        // 17 rescales at the input scale before the next bootstrap.
        assert_eq!(
            preset.output_layout().log_budget() - preset.input_layout().log_budget(),
            17 * preset.log_delta()
        );

        assert_layouts_within_bounds(&preset, 6);
        // The three CoeffsToSlots factors run at 1381, 1330 and 1279 bits, the
        // conjugation below them, and SlotsToCoeffs after EvalMod.
        let keys = preset.keys_layout();
        assert_eq!(
            key_shapes(&keys.rotation_key_layouts()),
            [(6, 5), (7, 4), (8, 4), (9, 3), (15, 1), (14, 1)]
        );
        assert_eq!(key_shapes([&keys.tensor_key]), [(9, 3)]);
    }

    #[test]
    fn n16_d35_k733_p20_s2c_is_composable_and_within_bounds() {
        let preset = n16_d35_k733_p20_s2c().unwrap();

        assert_eq!(preset.plan().pipeline(), BootstrappingPipeline::S2CFirst);
        assert_eq!(preset.plan().eval_mod().eval_mod_type, EvalModType::CosHKEven);
        assert_eq!(preset.plan().eval_mod().consumed_bits(), 464);
        assert_eq!(preset.plan().coeffs_to_slots().consumed_bits(), 150);
        assert_eq!(preset.plan().c2s_guard_bits(), 6);
        assert_eq!(preset.plan().slots_to_coeffs().consumed_bits(), 90);
        assert_eq!((preset.input_k(), preset.output_k(), preset.bootstrap_k()), (138, 733, 1353));
        assert_eq!(preset.log2_precision(), 20);
        assert_eq!(
            preset.output_layout().log_budget() - preset.input_layout().log_budget(),
            17 * preset.log_delta()
        );

        assert_layouts_within_bounds(&preset, 6);
        // The three CoeffsToSlots factors run at 1353, 1303 and 1253 bits, the
        // conjugation and the first EvalMod level below them; SlotsToCoeffs runs
        // on the 138-bit input.
        let keys = preset.keys_layout();
        assert_eq!(
            key_shapes(&keys.rotation_key_layouts()),
            [(6, 5), (7, 4), (8, 4), (9, 3), (3, 1), (2, 1)]
        );
        assert_eq!(key_shapes([&keys.tensor_key]), [(9, 3)]);
        assert_eq!(keys.rotation_key(-1).gglwe_layout().max_input_k().as_usize(), 1239);
        assert!(keys.tensor_key.gglwe_layout().max_input_k().as_usize() >= 1207);

        // One digit for every key leaves a single shape.
        let uniform = preset.with_dsizes(6, 1).unwrap();
        assert!(uniform.keys_layout().rotation_keys.is_empty());
        assert_eq!(uniform.keys_layout().tensor_key.dnum.as_usize(), 5);
    }

    /// `(dsize, dnum)` of each key, widest input first.
    fn key_shapes<'a, K: GGLWEInfos + LWEInfos + 'a>(keys: impl IntoIterator<Item = &'a K>) -> Vec<(usize, usize)> {
        let mut keys: Vec<&K> = keys.into_iter().collect();
        keys.sort_by_key(|key| core::cmp::Reverse(key.gglwe_layout().max_input_k()));
        keys.iter()
            .map(|key| (key.dsize().as_usize(), key.dnum().as_usize()))
            .collect()
    }

    /// Both presets fill the dense bound: their digit is the widest whose last
    /// one the raised input leaves room behind.
    fn assert_layouts_within_bounds(preset: &BootstrappingPreset, dsize: usize) {
        let keys = preset.keys_layout();
        let encapsulation = keys.encapsulation.as_ref().unwrap();

        assert_eq!((preset.max_dense_modulus(), preset.max_sparse_modulus()), (1707, 345));
        assert_eq!(keys.automorphism_key.dsize().as_usize(), dsize);
        assert_eq!(gadget_k(&keys.automorphism_key), 1560);
        assert_eq!(keys.automorphism_key.k_aux.as_usize(), 147);
        assert_eq!(keys.automorphism_key.k().as_usize(), 1707);
        assert_eq!(encapsulation.sparse_to_dense.k().as_usize(), 1707);
        assert!(preset.with_dsizes(dsize + 1, 1).is_err());

        assert_eq!(encapsulation.dense_to_sparse.dsize().as_usize(), 1);
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
            ["n16_d35_k636_p20_c2s", "n16_d35_k733_p20_s2c", "n15_d35_k180_p18_c2s"]
        );
    }

    #[test]
    fn rederived_key_shape_keeps_widths_and_revalidates() {
        for (preset, fft_dsize) in [
            (n16_d35_k636_p20_c2s().unwrap(), 7),
            (n16_d35_k733_p20_s2c().unwrap(), 7),
            (n15_d35_k180_p18_c2s().unwrap(), 3),
        ] {
            let widths = (preset.input_k(), preset.output_k(), preset.bootstrap_k());

            let fft = preset.with_base2k(18).unwrap().with_dsizes(fft_dsize, 1).unwrap();
            assert_eq!((fft.input_k(), fft.output_k(), fft.bootstrap_k()), widths);
            assert_eq!(
                (fft.base2k(), fft.key_dsize(), fft.dense_to_sparse_dsize()),
                (18, fft_dsize, 1)
            );
            assert_eq!(fft.bootstrap_layout().glwe_layout.base2k.as_usize(), 18);
            let keys = fft.keys_layout();
            assert_eq!(
                keys.automorphism_key.dnum.as_usize(),
                preset.bootstrap_k().div_ceil(fft_dsize * 18)
            );
            // A digit and `log_n` bits of guard, or what the bound leaves.
            let gadget = keys.automorphism_key.dnum.as_usize() * fft_dsize * 18;
            assert_eq!(
                keys.automorphism_key.k_aux.as_usize(),
                (fft_dsize * 18 + preset.log_n()).min(fft.max_dense_modulus() - gadget)
            );
            assert!(keys.automorphism_key.k().as_usize() <= fft.max_dense_modulus());

            let small_key = &keys.encapsulation.as_ref().unwrap().dense_to_sparse;
            assert_eq!(
                small_key.k().as_usize(),
                preset.log_modulus().div_ceil(18) * 18 + 18 + preset.log_n()
            );
            assert!(small_key.k().as_usize() <= fft.max_sparse_modulus());

            // Oversized digits are rejected for both secret bounds.
            assert!(preset.with_dsizes(10, 1).is_err());
            assert!(preset.with_dsizes(1, 7).is_err());
        }
    }
}
