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
//! - `circuit`: `c2s` (C2S-first) or `s2c` (S2C-first), followed by a suffix per further
//!   technique: `ffr` for the first CoeffsToSlots factor on the raise, `ci` for EvalMod
//!   on the conjugate-invariant ring, as [`n16_d35_k685_p19_s2c_ffr_ci`] combines them.
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
use poulpy_core::layouts::{
    Base2K, Degree, GGLWEInfos, GGLWELayout, GLWELayout, GLWESwitchingKeyLayout, LWEInfos, LinearTransformationLayout,
    LinearTransformationStrategy, Rank, TorusPrecision,
};
use poulpy_hal::layouts::{MaxBase2k, Module};

use crate::numerics::CKKSFloat;
use crate::{
    CKKSLayout, CoeffsMeta, GLWEPlaintextMeta, Scale, SlotsKind,
    layouts::{
        BootstrappingKeysLayout, BootstrappingPipeline, BootstrappingPlan, BootstrappingTechniques, CIConversion,
        DFTOutputFormat, DFTPlan, DFTType, EncapsulationKeysLayout, EvalModPlan, EvalModRing, EvalModType, EvalRoundPlus,
        SparseSecretEncapsulation,
    },
    polynomial::SplitStrategy,
    reference::first_factor::{first_factor_diagonal_growth, first_factor_small_bits, first_factor_work},
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
    /// The first CoeffsToSlots factor is applied by the raise, for no modulus.
    c2s_first_factor_on_raise: bool,
    eval_mod_ring: EvalModRing,
    c2s_guard_bits: usize,
    c2s_log_delta: usize,
    c2s_log_budget: usize,
    /// Scale of the high-precision CoeffsToSlots of EvalRound+, with the schedule
    /// of `c2s_schedule`, or `None` without EvalRound+.
    c2s_bypass_log_delta: Option<usize>,
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

    /// Limbs per gadget digit, `ceil(k_digit / base2k)`, of the keys used at
    /// the raised width: the sparse-to-dense key and the rotation keys of the
    /// first CoeffsToSlots factor. Unless [`Self::with_dsizes`] made every key
    /// alike, the keys used on narrower ciphertexts have wider digits (see
    /// [`Self::keys_layout`]).
    pub fn key_dsize(&self) -> usize {
        self.keys_layout.automorphism_key.dsize().as_usize()
    }

    /// Gadget digit width, in bits, of the keys used at the raised width.
    pub fn key_k_digit(&self) -> usize {
        self.keys_layout.automorphism_key.k_digit.as_usize()
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

    /// Re-derives the preset with the first CoeffsToSlots factor applied by the
    /// modulus raise ([`DFTPlan::with_first_factor_on_raise`]) and the factor
    /// schedule `c2s_schedule` (`(layers, giant step)` pairs, in evaluation
    /// order; the first giant step is not used). That factor consumes no
    /// modulus, so the raised width drops by one CoeffsToSlots scale, and the
    /// sparse-to-dense switch is replaced by one key per diagonal of the factor,
    /// `2^layers` of them for the first factor of a full transform, each with its
    /// rows under the few limbs of a diagonal's term.
    pub fn with_first_factor_on_raise(&self, c2s_schedule: &'static [(usize, usize)]) -> Result<Self> {
        build(PresetSpec {
            c2s_schedule,
            c2s_first_factor_on_raise: true,
            ..self.spec
        })
    }

    /// Re-derives the preset with EvalRound+: CoeffsToSlots at scale
    /// `2^c2s_log_delta` feeds EvalMod, and a high-precision CoeffsToSlots of the
    /// same schedule at `2^bypass_log_delta`, which shares its rotation keys,
    /// gives the output its precision.
    pub fn with_eval_round_plus(&self, c2s_log_delta: usize, bypass_log_delta: usize) -> Result<Self> {
        build(PresetSpec {
            c2s_log_delta,
            c2s_bypass_log_delta: Some(bypass_log_delta),
            ..self.spec
        })
    }

    /// Re-derives the preset with EvalMod on `eval_mod_ring`: the conversions to
    /// the conjugate-invariant ring widen the raised width by their scales.
    pub fn with_eval_mod_ring(&self, eval_mod_ring: EvalModRing) -> Result<Self> {
        build(PresetSpec {
            eval_mod_ring,
            ..self.spec
        })
    }

    /// Re-derives the preset with one gadget digit of `key_dsize` whole limbs for
    /// every high-modulus key, and of `dense_to_sparse_dsize` limbs for the
    /// dense-to-sparse key. Fails for a digit too wide for the modulus bound of
    /// its secret.
    pub fn with_dsizes(&self, key_dsize: usize, dense_to_sparse_dsize: usize) -> Result<Self> {
        build(PresetSpec {
            key_dsize: Some(key_dsize),
            dense_to_sparse_dsize,
            ..self.spec
        })
    }

    /// Largest limb radix at which every accumulation of the bootstrap stays
    /// within the extended precision of backend `BE`, and of `BCI` for EvalMod on
    /// the conjugate-invariant ring, with a failure probability of at most
    /// `2^-failure_bits` over the whole bootstrap, by the model of
    /// [`Module::max_base2k`](poulpy_hal::layouts::Module::max_base2k); zero when
    /// no radix fits.
    ///
    /// The widest accumulations are the giant-step passes of the transforms: an
    /// output limb sums, without normalizing, the product of every diagonal of a
    /// factor with a baby rotation, over the diagonal's limbs, and the key switch
    /// of every giant step, over the operand limbs. A first factor on the raise
    /// sums one key switch per diagonal over fewer limbs, which that count covers,
    /// a folded `u` runs the last CoeffsToSlots pass twice, and EvalRound+ adds the
    /// pass of its cancellation.
    /// An output limb of each of EvalMod's ciphertext products, as many as
    /// [`EvalModPlan::max_ciphertext_products`] bounds, sums a product per limb,
    /// one of them a square. On the CI ring EvalMod runs on `BCI`, and the two
    /// conversions of degree `2N` sum a key-switch product per operand limb on `BE`.
    /// The failure budget is split over the outputs of every pass, and the radix
    /// ignores the security bounds, which [`Self::with_backend_base2k`] applies.
    pub fn max_backend_base2k<BE: MaxBase2k, BCI: MaxBase2k>(&self, failure_bits: usize) -> usize {
        let n = self.n;
        let ci = self.plan.eval_mod_ring() != EvalModRing::Standard;
        for base2k in (1..=62).rev() {
            let limbs = self.bootstrap_k.div_ceil(base2k);
            let (products, transform_passes) = self.transform_accumulations(base2k);
            let passes = transform_passes + self.plan.eval_mod().max_ciphertext_products() + if ci { 2 } else { 0 };
            // Each accumulation's output polynomials: two columns, every limb.
            let failure_bits = failure_bits + (passes * 2 * limbs).next_power_of_two().ilog2() as usize;
            let transforms = Module::<BE>::max_base2k(n, products, failure_bits, false);
            let eval_mod = if ci {
                Module::<BCI>::max_base2k_squares(n, limbs, 1, failure_bits).min(Module::<BE>::max_base2k(
                    2 * n,
                    limbs,
                    failure_bits,
                    false,
                ))
            } else {
                Module::<BE>::max_base2k_squares(n, limbs, 1, failure_bits)
            };
            if transforms.min(eval_mod) >= base2k {
                return base2k;
            }
        }
        0
    }

    /// Re-derives the preset at the largest radix [`Self::max_backend_base2k`]
    /// allows on backends `BE` and `BCI` and the modulus bounds admit (see
    /// [`Self::with_base2k`]).
    pub fn with_backend_base2k<BE: MaxBase2k, BCI: MaxBase2k>(&self, failure_bits: usize) -> Result<Self> {
        let max = self.max_backend_base2k::<BE, BCI>(failure_bits);
        (1..=max)
            .rev()
            .find_map(|base2k| self.with_base2k(base2k).ok())
            .with_context(|| format!("no radix up to {max} fits the modulus bounds of {}", self.name()))
    }

    /// Worst count of degree-`N` products an output limb of a transform pass
    /// accumulates at radix `base2k`, and the number of such passes.
    fn transform_accumulations(&self, base2k: usize) -> (usize, usize) {
        let log_n = self.spec.log_n;
        let operand = self.bootstrap_k.div_ceil(base2k);
        let mut transforms = vec![self.plan.coeffs_to_slots(), self.plan.slots_to_coeffs()];
        transforms.extend(self.plan.coeffs_to_slots_bypass());
        let (mut products, mut passes) = (1, 0);
        for dft in transforms {
            let meta = dft.coeffs_meta();
            let diagonal_limbs = (meta.log_delta() + meta.log_budget()).div_ceil(base2k);
            for (indexes, giant_step) in dft.diagonal_indexes(log_n).into_iter().zip(dft.schedule().giant_steps()) {
                let index = LinearTransformationLayout {
                    indexes,
                    slots: 1 << log_n,
                    strategy: LinearTransformationStrategy::Bsgs { giant_step },
                }
                .index();
                let diagonals: usize = index.index.iter().map(Vec::len).sum();
                let giants = index.index.iter().filter(|babies| !babies.is_empty()).count();
                products = products.max(diagonals * diagonal_limbs + giants * operand);
                passes += 1;
            }
        }
        // EvalRound+ subtracts `K·x` from EvalMod's output: one more pass, of two terms.
        if self.plan.coeffs_to_slots_bypass().is_some() {
            passes += 1;
        }
        // A folded `u` runs the last CoeffsToSlots factor twice, on the same diagonals.
        if let EvalModRing::ConjugateInvariant {
            pre: CIConversion::Folded,
            ..
        } = self.plan.eval_mod_ring()
        {
            passes += 1;
        }
        (products, passes)
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
        c2s_first_factor_on_raise: false,
        eval_mod_ring: EvalModRing::Standard,
        c2s_guard_bits: 0,
        c2s_log_delta: 51,
        c2s_log_budget: 2,
        c2s_bypass_log_delta: None,
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
        c2s_first_factor_on_raise: false,
        eval_mod_ring: EvalModRing::Standard,
        c2s_guard_bits: 6,
        c2s_log_delta: 50,
        c2s_log_budget: 3,
        c2s_bypass_log_delta: None,
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
        c2s_first_factor_on_raise: false,
        eval_mod_ring: EvalModRing::Standard,
        c2s_guard_bits: 0,
        c2s_log_delta: 50,
        c2s_log_budget: 2,
        c2s_bypass_log_delta: None,
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
        c2s_first_factor_on_raise: false,
        eval_mod_ring: EvalModRing::Standard,
        c2s_guard_bits: 6,
        c2s_log_delta: 50,
        c2s_log_budget: 3,
        c2s_bypass_log_delta: None,
        s2c_schedule: &[(2, 4), (4, 4), (4, 4)],
        s2c_log_delta: 30,
        s2c_log_budget: 2,
        eval_mod: optimized_han_ki(),
    })
}

/// [`n16_d35_k733_p20_s2c`] tuned for speed: 685-bit ciphertexts with at least
/// 19 bits of precision (19.4 measured, the minimum over the slots), 15 usable levels. The first CoeffsToSlots factor is
/// applied by the raise, three factors of four layers follow, EvalMod runs once
/// on the conjugate-invariant ring ([`EvalModRing::ConjugateInvariant`], 84 bits
/// of conversions), and SlotsToCoeffs has four factors. The raised width is 1383 bits.
pub fn n16_d35_k685_p19_s2c_ffr_ci() -> Result<BootstrappingPreset> {
    build(PresetSpec {
        name: "n16_d35_k685_p19_s2c_ffr_ci",
        output_k: 685,
        log2_precision: 19,
        c2s_schedule: &[(3, 1), (4, 1024), (4, 64), (4, 4)],
        c2s_first_factor_on_raise: true,
        // `u` carries the overflow of the raise, `v` only the message.
        eval_mod_ring: EvalModRing::ConjugateInvariant {
            pre: CIConversion::Product { log_delta: 46 },
            post: CIConversion::Product { log_delta: 38 },
        },
        c2s_log_delta: 48,
        s2c_schedule: &[(3, 4), (4, 32), (4, 512), (4, 8192)],
        s2c_log_delta: 28,
        ..n16_d35_k733_p20_s2c()?.spec
    })
}

/// Every preset, in a stable order.
pub fn all() -> Result<Vec<BootstrappingPreset>> {
    const PRESETS: &[fn() -> Result<BootstrappingPreset>] = &[
        n16_d35_k636_p20_c2s,
        n16_d35_k733_p20_s2c,
        n15_d35_k180_p18_c2s,
        n16_d35_k685_p19_s2c_ffr_ci,
    ];
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
    let coeffs_to_slots = if spec.c2s_first_factor_on_raise {
        coeffs_to_slots.with_first_factor_on_raise()
    } else {
        coeffs_to_slots
    };
    let plan = BootstrappingPlan::new(
        spec.pipeline,
        BootstrappingTechniques {
            sparse_secret_encapsulation: Some(SparseSecretEncapsulation {
                hamming_weight: spec.sparse_secret.hamming_weight,
            }),
            eval_round_plus: match spec.c2s_bypass_log_delta {
                Some(log_delta) => Some(EvalRoundPlus {
                    coeffs_to_slots_bypass: DFTPlan::new(
                        DFTType::Encode,
                        spec.c2s_schedule.to_vec(),
                        DFTOutputFormat::SplitRealAndImag,
                        CoeffsMeta::from_delta_budget(log_delta, spec.c2s_log_budget),
                    )?
                    .with_scaling(1.0)?,
                }),
                None => None,
            },
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
    let plan = plan.with_eval_mod_ring(spec.eval_mod_ring)?;
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
        spec.key_dsize
            .map_or_else(|| widest_k_digit(spec, bootstrap_k, dense), |dsize| dsize * spec.base2k),
        dense,
    )?;
    let dense_to_sparse = key_layout(
        spec,
        "dense-to-sparse",
        spec.log_delta + spec.log_msg_ratio,
        spec.dense_to_sparse_dsize * spec.base2k,
        sparse,
    )?;
    let sparse_to_dense = sparse_to_dense_layout(
        spec,
        plan.raised_window(spec.log_delta + spec.log_msg_ratio),
        bootstrap_k,
        dense,
    )?;

    // With no digit imposed, a key only used on narrower ciphertexts takes the
    // widest digit those admit.
    let mut rotation_keys = BTreeMap::new();
    let mut tensor_key = wide;
    // EvalRound+ runs its bypass at widths of its own, which the stage widths
    // below do not model: its keys keep the shape of the raised width.
    if spec.key_dsize.is_none() && plan.coeffs_to_slots_bypass().is_none() {
        let widths = plan.key_widths();
        for (p, k) in widths.rotation_key_inputs(spec.log_n, bootstrap_k, input_k) {
            let layout = key_layout(spec, "rotation", k, widest_k_digit(spec, k, dense), dense)?;
            if layout != wide {
                rotation_keys.insert(p, layout.into());
            }
        }
        let k = widths.eval_mod_relinearization_k(bootstrap_k, spec.log_delta);
        tensor_key = key_layout(spec, "tensor", k, widest_k_digit(spec, k, dense), dense)?;
    }
    // The conversion keys are of degree 2N, under secrets of degree-N subrings
    // (`s(X²)`, the embedded CI secret), so the bounds of degree N apply.
    let wide_spec = PresetSpec {
        log_n: spec.log_n + 1,
        ..*spec
    };
    let conversion_key = |k| key_layout(&wide_spec, "conversion", k, widest_k_digit(&wide_spec, k, dense), dense);

    Ok(BootstrappingKeysLayout {
        automorphism_key: wide.into(),
        rotation_keys,
        tensor_key: tensor_key.into(),
        encapsulation: Some(EncapsulationKeysLayout {
            dense_to_sparse: dense_to_sparse.into(),
            sparse_to_dense: (!spec.c2s_first_factor_on_raise).then(|| sparse_to_dense.into()),
        }),
        first_factor_key: spec
            .c2s_first_factor_on_raise
            .then(|| first_factor_key_layout(spec, plan, bootstrap_k, dense))
            .transpose()?,
        conversion_keys: match plan.key_widths().conversions(bootstrap_k) {
            Some((traced, embedded)) => Some((conversion_key(traced)?.into(), conversion_key(embedded)?.into())),
            None => None,
        },
    })
}

/// Layout of the sparse-to-dense key, whose switch is the raise: its rows sit
/// under the raised ciphertext, `window` bits at the bottom of the raised
/// width ([`BootstrappingPlan::raised_window`]), and end exactly at that width,
/// so the key's modulus is the one a key covering the whole width would have.
/// The digits are whole limbs, the fewest rows first; the bits their gadget
/// adds over the window are a shift of the input inside it.
fn sparse_to_dense_layout(spec: &PresetSpec, window: usize, bootstrap_k: usize, limit: usize) -> Result<GGLWELayout> {
    let shape = |k_digit: usize| {
        let gadget = window.div_ceil(k_digit) * k_digit;
        bootstrap_k.checked_sub(gadget).map(|k_offset| (gadget, k_offset))
    };
    let fits = |k_digit: usize| {
        shape(k_digit).is_some_and(|(gadget, k_offset)| key_fits(spec, gadget, k_digit, limit.saturating_sub(k_offset)))
    };
    let k_digit = spec.key_dsize.map_or_else(
        || {
            (1..=window.div_ceil(spec.base2k))
                .map(|limbs| limbs * spec.base2k)
                .filter(|&k_digit| fits(k_digit))
                .min_by_key(|&k_digit| (window.div_ceil(k_digit), k_digit))
                .unwrap_or(spec.base2k)
        },
        |dsize| dsize * spec.base2k,
    );
    let (gadget, k_offset) = shape(k_digit).context("sparse-to-dense key: its digits overshoot the raised width")?;
    Ok(GGLWELayout {
        k_offset: TorusPrecision(k_offset as u32),
        ..key_layout(spec, "sparse-to-dense", gadget, k_digit, limit.saturating_sub(k_offset))?
    })
}

/// Layout of the keys of a first CoeffsToSlots factor applied by the raise:
/// each switches a diagonal's term, the raised ciphertext with the guard of
/// the working width ([`first_factor_work`]) below it and the growth of a
/// product above it ([`first_factor_small_bits`]), so its rows sit under that
/// many whole limbs and end at the working width. The digits are those of the
/// key of fewest rows for that input, under the modulus bound less the offset.
fn first_factor_key_layout(
    spec: &PresetSpec,
    plan: &BootstrappingPlan,
    bootstrap_k: usize,
    limit: usize,
) -> Result<GLWESwitchingKeyLayout> {
    let (work, _) = first_factor_work(spec.base2k, bootstrap_k);
    let raised = plan.raised_window(spec.log_delta + spec.log_msg_ratio);
    let growth = first_factor_diagonal_growth(spec.base2k, &plan.coeffs_to_slots().coeffs_meta());
    let bits = first_factor_small_bits(spec.base2k, spec.log_n, bootstrap_k, raised, growth);
    let small = bits.div_ceil(spec.base2k);
    ensure!(
        small < work,
        "first-factor key: a term of {small} limbs does not fit the {work} limbs the raise works at"
    );
    let k_offset = (work - small) * spec.base2k;
    let below = limit.saturating_sub(k_offset);
    let input_k = small * spec.base2k;
    let k_digit = spec
        .key_dsize
        .map_or_else(|| widest_k_digit(spec, input_k, below), |dsize| dsize * spec.base2k);
    Ok(GGLWELayout {
        k_offset: TorusPrecision(k_offset as u32),
        ..key_layout(spec, "first-factor", input_k, k_digit, below)?
    }
    .into())
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

/// Guard behind the digits of a key: a digit and `log_n` bits, or what the
/// bound leaves when that is less.
fn key_guard(spec: &PresetSpec, k_digit: usize, gadget: usize, limit: usize) -> usize {
    (k_digit + spec.log_n).min(limit.saturating_sub(gadget))
}

/// Whether a key of `k_digit`-bit digits switching an input of `input_k` bits
/// fits a modulus bound of `limit` bits: a guard of at least one digit (see
/// [`poulpy_core::layouts::key_k`]) and [`KEY_NOISE_MARGIN`] past the input
/// and one digit.
fn key_fits(spec: &PresetSpec, input_k: usize, k_digit: usize, limit: usize) -> bool {
    let gadget = input_k.div_ceil(k_digit) * k_digit;
    let guard = key_guard(spec, k_digit, gadget, limit);
    guard >= k_digit && gadget + guard >= input_k + k_digit + KEY_NOISE_MARGIN
}

/// Digit width, in bits, of the key of fewest rows that switches an input of
/// `input_k` bits under a modulus bound of `limit` bits, and at that row count
/// the narrowest, so the fewest operand limbs. Digits are whole limbs: a digit
/// off a limb boundary costs the products the digit layout of their input.
fn widest_k_digit(spec: &PresetSpec, input_k: usize, limit: usize) -> usize {
    let input_k = input_k.max(1);
    (1..=input_k.div_ceil(spec.base2k))
        .map(|limbs| limbs * spec.base2k)
        .filter(|&k_digit| key_fits(spec, input_k, k_digit, limit))
        .min_by_key(|&k_digit| (input_k.div_ceil(k_digit), k_digit))
        .unwrap_or(spec.base2k)
}

/// Layout of a rank-preserving key of `k_digit`-bit digits covering an input
/// of `input_k` bits under a modulus bound of `limit` bits: the digit count
/// from [`GGLWELayout::dnum_for_input`], and behind it the guard of
/// [`key_guard`], at least one digit.
fn key_layout(spec: &PresetSpec, name: &str, input_k: usize, k_digit: usize, limit: usize) -> Result<GGLWELayout> {
    let base2k = Base2K(spec.base2k as u32);
    let dnum = GGLWELayout::dnum_for_input(TorusPrecision(input_k as u32), TorusPrecision(k_digit as u32));
    let gadget = dnum.as_usize() * k_digit;
    let k_aux = key_guard(spec, k_digit, gadget, limit);
    ensure!(
        key_fits(spec, input_k, k_digit, limit),
        "{name} key: a digit of {k_digit} bits on a {input_k}-bit input needs a modulus of {} bits, above {limit}",
        (input_k + k_digit + KEY_NOISE_MARGIN).max(gadget + k_digit)
    );
    Ok(GGLWELayout {
        n: Degree(1 << spec.log_n),
        base2k,
        dnum,
        k_aux: TorusPrecision(k_aux as u32),
        rank_in: Rank(spec.rank as u32),
        rank_out: Rank(spec.rank as u32),
        k_digit: TorusPrecision(k_digit as u32),
        k_offset: TorusPrecision(0),
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
    if let Some(key) = &keys.first_factor_key {
        validate_key("first-factor", key, dense)?;
    }
    if let Some((trace, embed)) = &keys.conversion_keys {
        validate_key("trace", trace, dense)?;
        validate_key("embed", embed, dense)?;
    }
    let encapsulation = keys
        .encapsulation
        .as_ref()
        .context("bootstrapping preset is missing encapsulation keys")?;
    validate_key("dense-to-sparse", &encapsulation.dense_to_sparse, sparse)?;
    if let Some(key) = &encapsulation.sparse_to_dense {
        validate_key("sparse-to-dense", key, dense)?;
    }
    Ok(())
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
    use std::collections::BTreeSet;

    use super::*;
    use crate::CKKSInfos;
    use crate::reference::first_factor::FIRST_FACTOR_GUARD_BITS;

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
        // EvalMod starts 100 bits below the raised width: room for a 2-limb digit.
        assert_eq!(key_shapes([&keys.tensor_key]), [(104, 7)]);
        // The raise switches to the dense secret: two rows under the raised ciphertext, ending at the
        // raised width, at the modulus a key covering it would have.
        assert_eq!(sparse_to_dense_shape(&preset), (52, 2, 676, 847));
        assert_eq!(encapsulation.dense_to_sparse.k().as_usize(), 119);
        // A wider digit leaves less than a digit of guard under either bound.
        assert!(preset.with_dsizes(2, 1).is_err());
        assert!(preset.with_dsizes(1, 2).is_err());
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

        assert_layouts_within_bounds(&preset, (208, 7, 224, 1680));
        assert_eq!(sparse_to_dense_shape(&preset), (104, 1, 1277, 1501));
        // The three CoeffsToSlots factors run at 1381, 1330 and 1279 bits, the
        // conjugation below them, and SlotsToCoeffs after EvalMod.
        let keys = preset.keys_layout();
        assert_eq!(
            key_shapes(&keys.rotation_key_layouts()),
            [(208, 7), (260, 5), (416, 3), (780, 1), (728, 1)]
        );
        assert_eq!(key_shapes([&keys.tensor_key]), [(416, 3)]);
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

        assert_layouts_within_bounds(&preset, (208, 7, 224, 1680));
        assert_eq!(sparse_to_dense_shape(&preset), (104, 1, 1249, 1473));
        // The three CoeffsToSlots factors run at 1353, 1303 and 1253 bits, the
        // conjugation and the first EvalMod level below them; SlotsToCoeffs runs
        // on the 138-bit input.
        let keys = preset.keys_layout();
        assert_eq!(
            key_shapes(&keys.rotation_key_layouts()),
            [(208, 7), (260, 5), (416, 3), (156, 1), (104, 1)]
        );
        assert_eq!(key_shapes([&keys.tensor_key]), [(416, 3)]);
        assert_eq!(keys.rotation_key(-1).gglwe_layout().max_input_k().as_usize(), 1248);
        assert!(keys.tensor_key.gglwe_layout().max_input_k().as_usize() >= 1207);

        // One digit for every key leaves a single shape.
        let uniform = preset.with_dsizes(4, 1).unwrap();
        assert!(uniform.keys_layout().rotation_keys.is_empty());
        assert_eq!(uniform.keys_layout().tensor_key.dnum.as_usize(), 7);
    }

    #[test]
    fn n16_d35_k685_p19_s2c_ffr_ci_is_composable_and_within_bounds() {
        let preset = n16_d35_k685_p19_s2c_ffr_ci().unwrap();

        assert_eq!(preset.plan().pipeline(), BootstrappingPipeline::S2CFirst);
        assert_eq!(preset.plan().eval_mod().eval_mod_type, EvalModType::CosHKEven);
        assert_eq!(preset.plan().eval_mod().consumed_bits(), 464);
        assert_eq!(preset.plan().eval_mod_ring().consumed_bits(), 84);
        assert_eq!(preset.plan().coeffs_to_slots().consumed_bits(), 144);
        assert_eq!(preset.plan().c2s_guard_bits(), 6);
        assert_eq!(preset.plan().slots_to_coeffs().consumed_bits(), 112);
        assert_eq!((preset.input_k(), preset.output_k(), preset.bootstrap_k()), (160, 685, 1383));
        assert_eq!(preset.log2_precision(), 19);
        assert_eq!(
            preset.output_layout().log_budget() - preset.input_layout().log_budget(),
            15 * preset.log_delta()
        );

        // The keys of the raised width, under the dense bound.
        assert_eq!((preset.max_dense_modulus(), preset.max_sparse_modulus()), (1707, 345));
        let keys = preset.keys_layout();
        let wide = &keys.automorphism_key;
        assert_eq!(
            (
                wide.k_digit.as_usize(),
                wide.dnum.as_usize(),
                wide.k_aux.as_usize(),
                wide.k().as_usize()
            ),
            (208, 7, 224, 1680)
        );
        assert!(wide.k_aux >= wide.k_digit && wide.k().as_usize() <= preset.max_dense_modulus());
        assert_eq!(
            key_shapes(&keys.rotation_key_layouts()),
            [(208, 7), (260, 5), (156, 1), (104, 1)]
        );
        assert_eq!(key_shapes([&keys.tensor_key]), [(416, 3)]);
        assert!(keys.tensor_key.gglwe_layout().max_input_k().as_usize() >= 1248);
        // The first factor's keys: one row each, below the raised width.
        let first = keys.first_factor_key.unwrap();
        assert_eq!(
            (first.k_digit.as_usize(), first.dnum.as_usize(), first.k_offset.as_usize()),
            (104, 1, 1300)
        );
        // The trace key switches the raised CoeffsToSlots output, the embed key the EvalMod output.
        let (trace, embed) = keys.conversion_keys.unwrap();
        assert_eq!((key_shapes([&trace]), key_shapes([&embed])), (vec![(416, 3)], vec![(780, 1)]));
        assert_eq!(
            (
                trace.gglwe_layout().max_input_k().as_usize(),
                embed.gglwe_layout().max_input_k().as_usize()
            ),
            (1248, 780)
        );
    }

    /// `(k_digit, dnum, k_offset, k)` of the sparse-to-dense key, whose rows end at the raised width.
    fn sparse_to_dense_shape(preset: &BootstrappingPreset) -> (usize, usize, usize, usize) {
        let key = preset.keys_layout().encapsulation.unwrap().sparse_to_dense.unwrap();
        let gadget = gadget_k(&key);
        assert_eq!(key.k_offset.as_usize() + gadget, preset.bootstrap_k());
        assert!(gadget >= preset.plan().raised_window(preset.spec.log_delta + preset.spec.log_msg_ratio));
        (
            key.k_digit.as_usize(),
            key.dnum.as_usize(),
            key.k_offset.as_usize(),
            key.k().as_usize(),
        )
    }

    /// `(k_digit, dnum)` of each key, widest input first.
    fn key_shapes<'a, K: GGLWEInfos + LWEInfos + 'a>(keys: impl IntoIterator<Item = &'a K>) -> Vec<(usize, usize)> {
        let mut keys: Vec<&K> = keys.into_iter().collect();
        keys.sort_by_key(|key| core::cmp::Reverse(key.gglwe_layout().max_input_k()));
        keys.iter()
            .map(|key| (key.k_digit().as_usize(), key.dnum().as_usize()))
            .collect()
    }

    /// `(k_digit, dnum, k_aux, k)` of the keys used at the raised width, each
    /// guarded by at least a digit under the dense bound; a uniform digit of
    /// five limbs leaves less.
    fn assert_layouts_within_bounds(preset: &BootstrappingPreset, wide: (usize, usize, usize, usize)) {
        let keys = preset.keys_layout();
        let encapsulation = keys.encapsulation.as_ref().unwrap();

        assert_eq!((preset.max_dense_modulus(), preset.max_sparse_modulus()), (1707, 345));
        let key = &keys.automorphism_key;
        assert_eq!(
            (
                key.k_digit.as_usize(),
                key.dnum.as_usize(),
                key.k_aux.as_usize(),
                key.k().as_usize()
            ),
            wide
        );
        assert!(key.k_aux >= key.k_digit && key.k().as_usize() <= preset.max_dense_modulus());
        // One row under the raised ciphertext, below the modulus of the keys covering the raised width.
        assert!(encapsulation.sparse_to_dense.unwrap().k() <= key.k());
        assert!(preset.with_dsizes(5, 1).is_err());
        assert!(preset.with_dsizes(4, 1).is_ok());

        assert_eq!(encapsulation.dense_to_sparse.dsize().as_usize(), 1);
        assert_eq!(gadget_k(&encapsulation.dense_to_sparse), 52);
        assert_eq!(encapsulation.dense_to_sparse.k_aux.as_usize(), 68);
        assert_eq!(encapsulation.dense_to_sparse.k().as_usize(), 120);
        assert!(encapsulation.dense_to_sparse.k().as_usize() <= preset.max_sparse_modulus());
    }

    #[test]
    fn first_factor_on_raise_saves_one_factor_of_width() {
        const SCHEDULE: [(usize, usize); 3] = [(3, 1), (6, 512), (6, 8)];
        // Three factors of CoeffsToSlots, of which the raise takes the first.
        let base = n16_d35_k636_p20_c2s().unwrap();
        let preset = base.with_first_factor_on_raise(&SCHEDULE).unwrap();

        // The factor costs no modulus; the widths the application sees do not move.
        assert_eq!(preset.bootstrap_k() + 51, base.bootstrap_k());
        assert_eq!((preset.input_k(), preset.output_k()), (base.input_k(), base.output_k()));
        let c2s = preset.plan().coeffs_to_slots();
        assert!(c2s.first_factor_on_raise());
        assert_eq!(c2s.consumed_bits(), 2 * 51);
        assert_eq!(c2s.num_evaluated_factors(), 2);

        // One key per diagonal, its rows under a term of three limbs and ending at the
        // working width of the raise, the raised width and its guard.
        let keys = preset.keys_layout();
        let key = keys.first_factor_key.unwrap();
        let work = (preset.bootstrap_k() + FIRST_FACTOR_GUARD_BITS).div_ceil(52) * 52;
        assert_eq!(key.k_offset.as_usize() + 3 * 52, work);
        assert_eq!((key.k_digit.as_usize(), key.dnum.as_usize()), (156, 1));
        assert_eq!(key.k_offset.as_usize() + gadget_k(&key), work);
        assert!(key.k().as_usize() <= preset.max_dense_modulus());
        assert!(base.keys_layout().first_factor_key.is_none());
        // The first-factor keys switch to the dense secret: no sparse-to-dense key.
        assert!(keys.encapsulation.unwrap().sparse_to_dense.is_none());
        assert!(base.keys_layout().encapsulation.unwrap().sparse_to_dense.is_some());

        // The first paid factor runs at the raised width, where the first factor ran before.
        assert!(keys.automorphism_key.gglwe_layout().max_input_k().as_usize() >= preset.bootstrap_k());
        assert!(preset.with_dsizes(4, 1).unwrap().keys_layout().first_factor_key.is_some());
    }

    /// Bytes of the coefficients of a key of layout `key`, 8 per coefficient.
    fn key_bytes<K: GGLWEInfos + LWEInfos>(key: &K) -> usize {
        key.n().as_usize() * key.dnum().as_usize() * key.rank_in().as_usize() * (key.rank_out().as_usize() + 1) * key.size() * 8
    }

    /// Bytes of the coefficients of the key set a preset generates, by kind:
    /// rotation keys, tensor key, encapsulation keys, first-factor keys and the
    /// trace and embed keys of EvalMod on the CI ring.
    fn key_set_bytes(preset: &BootstrappingPreset) -> [usize; 5] {
        let (keys, plan, log_n) = (preset.keys_layout(), preset.plan(), preset.log_n());
        let order = 2i64 << log_n;
        let mut elements: BTreeSet<i64> = plan.coeffs_to_slots().galois_elements(log_n, order).into_iter().collect();
        elements.extend(plan.slots_to_coeffs().galois_elements(log_n, order));
        if plan.eval_mod_ring() == EvalModRing::Standard {
            elements.insert(-1);
        }
        let rotations = elements.iter().map(|&p| key_bytes(&keys.rotation_key(p))).sum();
        let encapsulation = keys.encapsulation.as_ref().map_or(0, |encaps| {
            key_bytes(&encaps.dense_to_sparse) + encaps.sparse_to_dense.as_ref().map_or(0, key_bytes)
        });
        let first_factor = keys.first_factor_key.as_ref().map_or(0, |key| {
            plan.coeffs_to_slots().diagonal_indexes(log_n)[0].len() * key_bytes(key)
        });
        let conversion = keys
            .conversion_keys
            .map_or(0, |(trace, embed)| key_bytes(&trace) + key_bytes(&embed));
        [
            rotations,
            key_bytes(&keys.tensor_key),
            encapsulation,
            first_factor,
            conversion,
        ]
    }

    #[test]
    fn first_factor_on_raise_trades_key_size_for_width() {
        let mib = |bytes: usize| {
            assert!(bytes.is_multiple_of(1 << 20));
            bytes >> 20
        };
        let base = n16_d35_k636_p20_c2s().unwrap();
        let preset = base.with_first_factor_on_raise(&[(3, 1), (6, 512), (6, 8)]).unwrap();
        assert_eq!(preset.bootstrap_k() + 51, base.bootstrap_k());
        assert_eq!((preset.input_k(), preset.output_k()), (base.input_k(), base.output_k()));
        // 51 rotation keys, the tensor key and the encapsulation keys: 8310 MiB.
        assert_eq!(key_set_bytes(&base).map(mib), [8179, 99, 32, 0, 0]);
        // The two remaining factors of six layers take 60 rotation keys, and the eight
        // diagonals of the first one a key of 30 MiB each in place of the
        // sparse-to-dense key of 29: 9386 MiB.
        assert_eq!(key_set_bytes(&preset).map(mib), [9044, 99, 3, 240, 0]);
    }

    #[test]
    fn n16_d35_k685_p19_s2c_ffr_ci_runs_eval_mod_on_the_ci_ring() {
        let preset = n16_d35_k685_p19_s2c_ffr_ci().unwrap();
        assert_eq!((preset.input_k(), preset.output_k(), preset.bootstrap_k()), (160, 685, 1383));
        assert_eq!(preset.plan().eval_mod_ring().consumed_bits(), 84);
        // The tensor key is of the CI ring of degree N, the conversion keys of degree 2N.
        let keys = preset.keys_layout();
        let (trace, embed) = keys.conversion_keys.unwrap();
        assert_eq!(
            (keys.tensor_key.n.as_usize(), trace.n.as_usize(), embed.n.as_usize()),
            (1 << 16, 1 << 17, 1 << 17)
        );
    }

    #[test]
    fn folding_u_narrows_n16_d35_k685_p19_s2c_ffr_ci() {
        let preset = n16_d35_k685_p19_s2c_ffr_ci().unwrap();
        let folded = preset
            .with_eval_mod_ring(EvalModRing::ConjugateInvariant {
                pre: CIConversion::Folded,
                post: CIConversion::Product { log_delta: 38 },
            })
            .unwrap();
        // The product by `u` no longer takes its 46 bits.
        assert_eq!((folded.input_k(), folded.output_k(), folded.bootstrap_k()), (160, 685, 1337));
        assert_eq!(folded.plan().eval_mod_ring().consumed_bits(), 38);
        // The rotation keys follow the narrower raised width: 5533 MiB against 6845.
        assert_eq!(key_set_bytes(&folded).map(|bytes| bytes >> 20), [4923, 99, 3, 248, 260]);
    }

    #[test]
    fn n16_d35_k685_p19_s2c_ffr_ci_has_a_smaller_key_set() {
        let mib = |bytes: usize| {
            assert!(bytes.is_multiple_of(1 << 20));
            bytes >> 20
        };
        // Rotation keys with conjugation, the tensor key and the encapsulation keys: 8022 MiB.
        assert_eq!(key_set_bytes(&n16_d35_k733_p20_s2c().unwrap()).map(mib), [7891, 99, 32, 0, 0]);
        // 43 rotation keys and no conjugation, the CI tensor key, the dense-to-sparse key,
        // eight first-factor keys and the trace and embed keys of degree 2N: 6845 MiB.
        let preset = n16_d35_k685_p19_s2c_ffr_ci().unwrap();
        assert_eq!(key_set_bytes(&preset).map(mib), [6243, 99, 3, 240, 260]);
        // With the first factor evaluated by the matrix, every paid factor runs 48 bits
        // wider and the sparse-to-dense key comes back: 8001 MiB.
        let without = build(PresetSpec {
            c2s_schedule: &[(3, 8), (4, 1024), (4, 64), (4, 4)],
            c2s_first_factor_on_raise: false,
            ..preset.spec
        })
        .unwrap();
        assert_eq!(without.bootstrap_k(), 1431);
        assert_eq!(key_set_bytes(&without).map(mib), [7609, 99, 33, 0, 260]);
    }

    #[test]
    fn all_lists_every_preset_once() {
        let names: Vec<&str> = all().unwrap().iter().map(|p| p.name()).collect();
        assert_eq!(
            names,
            [
                "n16_d35_k636_p20_c2s",
                "n16_d35_k733_p20_s2c",
                "n15_d35_k180_p18_c2s",
                "n16_d35_k685_p19_s2c_ffr_ci"
            ]
        );
    }

    #[test]
    fn rederived_key_shape_keeps_widths_and_revalidates() {
        for (preset, fft_dsize) in [
            (n16_d35_k636_p20_c2s().unwrap(), 7),
            (n16_d35_k733_p20_s2c().unwrap(), 7),
            (n15_d35_k180_p18_c2s().unwrap(), 2),
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
