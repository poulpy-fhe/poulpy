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
//! Every preset uses a ternary dense secret of Hamming weight 1024 and an
//! ephemeral ternary sparse secret of weight 32 for the raise, and is certified
//! for 128 bits of security ([`Security::Bits128`]). The bound of each secret, the
//! largest modulus any object under it may use, comes from the rlwe-params table
//! vendored in poulpy-core (see [`poulpy_core::security`]), at the degree of the
//! ring the secret lives in. At rlwe-params v0.2.0:
//!
//! | `N` | dense secret, weight 1024 | sparse secret, weight 32 |
//! | --- | ---: | ---: |
//! | `2^16` | 1707 | 342 |
//! | `2^15` | 851 | 162 |
//!
//! The bounds shape the keys: each key takes the digit of fewest rows whose
//! modulus, auxiliary guard included, fits the bound of its secret (see
//! [`BootstrappingKeyPolicy`]). Building a preset then checks the raised
//! ciphertext and every key against them
//! ([`BootstrappingPlan::check_key_security`]), and its plan carries the level, so
//! generating the keys checks them again under the dense secret actually given.
//! The values are estimates against the attacks of the estimators rlwe-params
//! runs, not proofs.
//!
//! Choosing a certified preset is the opt-in: nothing else in the library checks
//! a level. Generating a preset's keys under a secret its level has no bound for,
//! or one whose bound its keys exceed, is refused: a ternary secret of weight 192
//! at `N = 2^16` is bounded at 1475 bits, below the 1680-bit keys of the logN16
//! presets. A caller who wants those keys anyway clears the level with
//! [`BootstrappingPlan::without_security`], and certifies nothing.
//!
//! The keys of a fold ([`CKKSFoldKeysLayout`](crate::layouts::CKKSFoldKeysLayout)), whose
//! outbound ring-switch and automorphism keys sit under the inputs' secret at their degree,
//! belong to no preset: they are certified only through
//! [`CKKSFoldKeysLayout::check_security`](crate::layouts::CKKSFoldKeysLayout::check_security).

use anyhow::{Context, Result, ensure};
use poulpy_core::{
    Distribution,
    layouts::{
        Base2K, BsgsRotations, Degree, GGLWEInfos, GLWEAutomorphismKeyLayout, GLWELayout, LinearTransformationLayout, Rank,
        TorusPrecision,
    },
    security::{SecretSpec, Security},
};
use poulpy_hal::layouts::{ConjugateInvariant, MaxBase2k, Module, Standard};

use crate::numerics::CKKSFloat;
use crate::{
    CKKSCompositionError, CKKSLayout, CoeffsMeta, GLWEPlaintextMeta, Scale, SlotLayout, SlotsKind,
    layouts::{
        BootstrappingKeysLayout, BootstrappingPipeline, BootstrappingPlan, BootstrappingTechniques, CIConversion,
        DFTOutputFormat, DFTPlan, DFTType, EvalModPlan, EvalModRing, EvalModType, EvalRoundPlus, SparseSecretEncapsulation,
    },
    polynomial::SplitStrategy,
    presets::key_policy::BootstrappingKeyPolicy,
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

/// Hamming weight of the dense secret of the presets.
const DENSE_HAMMING_WEIGHT: usize = 1024;
/// Hamming weight of the ephemeral sparse secret of the presets.
const SPARSE_HAMMING_WEIGHT: usize = 32;

/// Where the modulus bounds that shape the keys of a preset come from.
#[derive(Clone, Copy, Debug)]
enum ModulusBounds {
    /// The bounds of a level for the preset's secrets, which building the preset certifies
    /// (see [Security](self#security)).
    Certified(Security),
    /// Placeholder bounds of an insecure test preset, in bits, at every degree: not security limits.
    #[cfg(feature = "test-utils")]
    Placeholder { dense: usize, sparse: usize },
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
    /// Re-derived with other widths or precision than were measured: no name, no advertised precision.
    derived: bool,
    dense_hamming_weight: usize,
    sparse_hamming_weight: usize,
    bounds: ModulusBounds,
    /// Widest digit, in limbs, of the high-modulus keys, each of which takes the
    /// digit of fewest rows its use admits up to it; `None` leaves them uncapped.
    max_dsize: Option<usize>,
    dense_to_sparse_dsize: usize,
    pipeline: BootstrappingPipeline,
    log_msg_ratio: usize,
    c2s_schedule: &'static [(usize, usize)],
    /// The first CoeffsToSlots factor is applied by the raise, for no modulus.
    c2s_first_factor_on_raise: bool,
    /// The transforms switch with the keys of power-of-two rotations only.
    power_of_two_keys: bool,
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
    /// Bits a circuit between SlotsToCoeffs and the raise spends, on top of the input width.
    input_headroom: usize,
    /// `log2` of the gap of the plaintexts, see [`BootstrappingPlan::with_log_gap`].
    log_gap: usize,
    eval_mod: EvalModSpec,
}

impl PresetSpec {
    /// The level the preset is certified at, `None` for an insecure test preset.
    fn security(&self) -> Option<Security> {
        match self.bounds {
            ModulusBounds::Certified(security) => Some(security),
            #[cfg(feature = "test-utils")]
            ModulusBounds::Placeholder { .. } => None,
        }
    }

    /// The dense secret, in the ring of degree `n`.
    fn dense_secret(&self, n: usize) -> SecretSpec {
        SecretSpec::new(n, Distribution::TernaryFixed(self.dense_hamming_weight))
    }

    /// The sparse secret, in the subring of the gap of the ring of degree `n`.
    fn sparse_secret(&self, n: usize) -> SecretSpec {
        SecretSpec::new(n >> self.log_gap, Distribution::TernaryFixed(self.sparse_hamming_weight))
    }

    /// Bounds, in bits, of the dense and sparse secrets in the ring of degree `n`.
    fn log_modulus_bounds(&self, n: usize) -> poulpy_core::Result<(usize, usize)> {
        match self.bounds {
            ModulusBounds::Certified(security) => Ok((
                security.max_log_q(self.dense_secret(n))?,
                security.max_log_q(self.sparse_secret(n))?,
            )),
            #[cfg(feature = "test-utils")]
            ModulusBounds::Placeholder { dense, sparse } => Ok((dense, sparse)),
        }
    }

    fn key_policy(&self, dense_log_modulus: usize, sparse_log_modulus: usize) -> BootstrappingKeyPolicy {
        BootstrappingKeyPolicy {
            log_n: self.log_n,
            base2k: self.base2k,
            rank: self.rank,
            dense_log_modulus,
            sparse_log_modulus,
            max_dsize: self.max_dsize,
            dense_to_sparse_dsize: self.dense_to_sparse_dsize,
        }
    }
}

/// The scales a preset spends its modulus on, each in bits. Every width and key of the preset follows from them, so a
/// circuit that needs less precision than a preset offers can spend less at each stage.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct PresetScales {
    /// Scale of the ciphertexts.
    pub log_delta: usize,
    /// Bits between the message and the modulus at the raise, which EvalMod's approximation rests on.
    pub log_msg_ratio: usize,
    /// Extra bits CoeffsToSlots carries for EvalMod.
    pub c2s_guard_bits: usize,
    /// Scale of a CoeffsToSlots factor.
    pub c2s_log_delta: usize,
    /// Scale of a SlotsToCoeffs factor.
    pub s2c_log_delta: usize,
    /// Scale of a level of EvalMod.
    pub eval_mod_log_delta: usize,
    /// Scale of EvalMod's polynomial coefficients.
    pub eval_mod_coeffs_log_delta: usize,
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
    dense_log_modulus: usize,
    sparse_log_modulus: usize,
}

impl BootstrappingPreset {
    /// Stable descriptive name of the preset, `None` once re-derived with other widths or precision: a name states the
    /// output width and the precision measured.
    pub fn name(&self) -> Option<&'static str> {
        (!self.spec.derived).then_some(self.spec.name)
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
    /// `test_suite::presets::bootstrapping_presets_meet_precision`. `None` once re-derived with other widths or
    /// precision, which the caller measures.
    pub fn log2_precision(&self) -> Option<usize> {
        (!self.spec.derived).then_some(self.spec.log2_precision)
    }

    /// Hamming weight of the dense application secret.
    pub fn dense_secret_hamming_weight(&self) -> usize {
        self.spec.dense_hamming_weight
    }

    /// Hamming weight of the ephemeral sparse ModUp secret.
    pub fn sparse_secret_hamming_weight(&self) -> usize {
        self.spec.sparse_hamming_weight
    }

    /// The level the preset is certified at, `None` for an insecure test preset (see [Security](self#security)).
    pub fn security(&self) -> Option<Security> {
        self.spec.security()
    }

    /// Modulus bound, in bits, the keys under the dense secret are shaped under: the bound of the dense secret at
    /// [`Self::security`] (see [Security](self#security)).
    pub fn max_dense_modulus(&self) -> usize {
        self.dense_log_modulus
    }

    /// Modulus bound, in bits, the keys under the ephemeral sparse secret are shaped under: the bound of the sparse
    /// secret at [`Self::security`], on the subring of the gap (see [Security](self#security)).
    pub fn max_sparse_modulus(&self) -> usize {
        self.sparse_log_modulus
    }

    /// Limbs per gadget digit, `ceil(k_digit / base2k)`, of the rotation key
    /// used on the widest ciphertext. Each high-modulus key takes the digit of
    /// fewest rows its use admits, up to the cap of [`Self::with_dsizes`], so
    /// the keys used on narrower ciphertexts can have wider digits (see
    /// [`Self::keys_layout`]).
    pub fn key_dsize(&self) -> usize {
        self.widest_rotation_key().dsize().as_usize()
    }

    /// Gadget digit width, in bits, of the rotation key used on the widest ciphertext.
    pub fn key_k_digit(&self) -> usize {
        self.widest_rotation_key().k_digit.as_usize()
    }

    /// The rotation key that covers the widest input.
    fn widest_rotation_key(&self) -> &GLWEAutomorphismKeyLayout {
        self.keys_layout
            .rotation_keys
            .values()
            .max_by_key(|key| key.gglwe_layout().max_input_k())
            .expect("a bootstrapping preset has rotation keys")
    }

    /// How this preset shapes its keys: a caller adjusts it and derives other layouts for the plan with
    /// [`BootstrappingKeyPolicy::keys_layout`].
    pub fn key_policy(&self) -> BootstrappingKeyPolicy {
        self.spec.key_policy(self.dense_log_modulus, self.sparse_log_modulus)
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
        build("with_base2k", PresetSpec { base2k, ..self.spec })
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
        build(
            "with_first_factor_on_raise",
            PresetSpec {
                c2s_schedule,
                c2s_first_factor_on_raise: true,
                ..self.spec
            },
        )
    }

    /// Re-derives the preset with EvalRound+: CoeffsToSlots at scale
    /// `2^c2s_log_delta` feeds EvalMod, and a high-precision CoeffsToSlots of the
    /// same schedule at `2^bypass_log_delta`, which shares its rotation keys,
    /// gives the output its precision.
    pub fn with_eval_round_plus(&self, c2s_log_delta: usize, bypass_log_delta: usize) -> Result<Self> {
        build(
            "with_eval_round_plus",
            PresetSpec {
                c2s_log_delta,
                c2s_bypass_log_delta: Some(bypass_log_delta),
                ..self.spec
            },
        )
    }

    /// Re-derives the preset with the transforms evaluated with the keys of
    /// power-of-two rotations only ([`DFTPlan::with_power_of_two_keys`]): the
    /// same key switches, half the CoeffsToSlots and SlotsToCoeffs keys, each
    /// switching several ciphertexts at once, and a product through more switches.
    pub fn with_power_of_two_keys(&self) -> Result<Self> {
        build(
            "with_power_of_two_keys",
            PresetSpec {
                power_of_two_keys: true,
                ..self.spec
            },
        )
    }

    /// Re-derives the preset with EvalMod on `eval_mod_ring`: the conversions to
    /// the conjugate-invariant ring widen the raised width by their scales.
    pub fn with_eval_mod_ring(&self, eval_mod_ring: EvalModRing) -> Result<Self> {
        build(
            "with_eval_mod_ring",
            PresetSpec {
                eval_mod_ring,
                ..self.spec
            },
        )
    }

    /// The scales of the preset.
    pub fn scales(&self) -> PresetScales {
        PresetScales {
            log_delta: self.spec.log_delta,
            log_msg_ratio: self.spec.log_msg_ratio,
            c2s_guard_bits: self.spec.c2s_guard_bits,
            c2s_log_delta: self.spec.c2s_log_delta,
            s2c_log_delta: self.spec.s2c_log_delta,
            eval_mod_log_delta: self.spec.eval_mod.log_delta,
            eval_mod_coeffs_log_delta: self.spec.eval_mod.coeffs_log_delta,
        }
    }

    /// Re-derives the preset with the scales `scales`, without a name or an advertised precision: a caller that lowers
    /// a scale measures what it gets.
    pub fn with_scales(&self, scales: PresetScales) -> Result<Self> {
        build(
            "with_scales",
            PresetSpec {
                derived: true,
                log_delta: scales.log_delta,
                log_msg_ratio: scales.log_msg_ratio,
                c2s_guard_bits: scales.c2s_guard_bits,
                c2s_log_delta: scales.c2s_log_delta,
                s2c_log_delta: scales.s2c_log_delta,
                eval_mod: EvalModSpec {
                    log_delta: scales.eval_mod_log_delta,
                    coeffs_log_delta: scales.eval_mod_coeffs_log_delta,
                    ..self.spec.eval_mod
                },
                ..self.spec
            },
        )
    }

    /// Re-derives the preset with EvalMod's polynomial: the Han–Ki fit of degree `degree` on `interval` integers on
    /// each side of zero, followed by `log_interval_reduction` squarings.
    ///
    /// `interval` must bound the integer the raise adds: a coefficient past it leaves EvalMod's range and comes out
    /// wrong. That integer has a standard deviation of about `sqrt((h + 1) / 12)` for a sparse secret of Hamming weight
    /// `h`, and the bound is the caller's to size to the coefficients it raises.
    pub fn with_eval_mod_polynomial(&self, interval: usize, degree: usize, log_interval_reduction: usize) -> Result<Self> {
        build(
            "with_eval_mod_polynomial",
            PresetSpec {
                derived: true,
                eval_mod: EvalModSpec {
                    interval,
                    degree,
                    log_interval_reduction,
                    ..self.spec.eval_mod
                },
                ..self.spec
            },
        )
    }

    /// Re-derives the preset to leave `output_k` bits, for a circuit that spends fewer than the preset offers: the
    /// raised width and every key follow.
    pub fn with_output_k(&self, output_k: usize) -> Result<Self> {
        build(
            "with_output_k",
            PresetSpec {
                output_k,
                derived: true,
                ..self.spec
            },
        )
    }

    /// Re-derives the preset with the SlotsToCoeffs schedule `s2c_schedule` (`(layers, giant step)` pairs, in evaluation
    /// order): each factor spends one SlotsToCoeffs scale from the input width.
    pub fn with_slots_to_coeffs(&self, s2c_schedule: &'static [(usize, usize)]) -> Result<Self> {
        build(
            "with_slots_to_coeffs",
            PresetSpec {
                s2c_schedule,
                derived: true,
                ..self.spec
            },
        )
    }

    /// Re-derives the preset with an input `bits` wider, for a circuit between SlotsToCoeffs and the raise that spends
    /// them: SlotsToCoeffs and the keys of its rotations take the wider input, and hand that circuit `bits` above the
    /// raise's input.
    ///
    /// Fails on a C2S-first preset, which has no such place.
    pub fn with_input_headroom(&self, bits: usize) -> Result<Self> {
        ensure!(
            self.spec.pipeline == BootstrappingPipeline::S2CFirst,
            "only an S2C-first bootstrap runs a circuit between SlotsToCoeffs and the raise"
        );
        build(
            "with_input_headroom",
            PresetSpec {
                input_headroom: bits,
                derived: true,
                ..self.spec
            },
        )
    }

    /// Re-derives the preset for plaintexts in `Z[X^(2^log_gap)]`, on the schedules of that many fewer layers: see
    /// [`BootstrappingPlan::with_log_gap`]. A key under the sparse secret of the subring splits into `2^log_gap` RLWE
    /// samples of the subring's degree, so its modulus is bounded at that degree.
    ///
    /// A gap moves the first CoeffsToSlots factor onto the raise, which requires the preset's sparse encapsulation.
    ///
    /// Fails when the preset's level has no bound for the sparse secret at that degree, or a key does not fit it.
    pub fn with_log_gap(
        &self,
        log_gap: usize,
        c2s_schedule: &'static [(usize, usize)],
        s2c_schedule: &'static [(usize, usize)],
    ) -> Result<Self> {
        build(
            "with_log_gap",
            PresetSpec {
                log_gap,
                derived: true,
                c2s_schedule,
                c2s_first_factor_on_raise: true,
                s2c_schedule,
                ..self.spec
            },
        )
    }

    /// Re-derives the preset with the gadget digits of the high-modulus keys
    /// capped at `max_dsize` whole limbs, each key taking the digit of fewest rows
    /// its use admits under the cap, and a digit of `dense_to_sparse_dsize` limbs
    /// for the dense-to-sparse key. Fails when some key has no digit under the cap
    /// that fits the modulus bound of its secret.
    pub fn with_dsizes(&self, max_dsize: usize, dense_to_sparse_dsize: usize) -> Result<Self> {
        build(
            "with_dsizes",
            PresetSpec {
                max_dsize: Some(max_dsize),
                dense_to_sparse_dsize,
                ..self.spec
            },
        )
    }

    /// Largest limb radix at which every accumulation of the bootstrap stays
    /// within the extended precision of backend `BE`, on the conjugate-invariant ring
    /// for EvalMod there, with a failure probability of at most
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
    /// one of them a square. On the CI ring EvalMod runs on that ring, and the two
    /// conversions of degree `2N` sum a key-switch product per operand limb on the standard ring.
    /// The failure budget is split over the outputs of every pass, and the radix
    /// ignores the security bounds, which [`Self::with_backend_base2k`] applies.
    pub fn max_backend_base2k<BE: MaxBase2k>(&self, failure_bits: usize) -> usize {
        let n = self.n;
        let ci = self.plan.eval_mod_ring() != EvalModRing::Standard;
        for base2k in (1..=62).rev() {
            let limbs = self.bootstrap_k.div_ceil(base2k);
            let (products, transform_passes) = self.transform_accumulations(base2k);
            let passes = transform_passes + self.plan.eval_mod().max_ciphertext_products() + if ci { 2 } else { 0 };
            // Each accumulation's output polynomials: two columns, every limb.
            let failure_bits = failure_bits + (passes * 2 * limbs).next_power_of_two().ilog2() as usize;
            let transforms = Module::<BE>::max_base2k::<Standard>(n, products, failure_bits, false);
            let eval_mod =
                if ci {
                    Module::<BE>::max_base2k_squares::<ConjugateInvariant>(n, limbs, 1, failure_bits)
                        .min(Module::<BE>::max_base2k::<Standard>(2 * n, limbs, failure_bits, false))
                } else {
                    Module::<BE>::max_base2k_squares::<Standard>(n, limbs, 1, failure_bits)
                };
            if transforms.min(eval_mod) >= base2k {
                return base2k;
            }
        }
        0
    }

    /// Re-derives the preset at the largest radix [`Self::max_backend_base2k`]
    /// allows on backend `BE` and the modulus bounds admit (see
    /// [`Self::with_base2k`]).
    pub fn with_backend_base2k<BE: MaxBase2k>(&self, failure_bits: usize) -> Result<Self> {
        let max = self.max_backend_base2k::<BE>(failure_bits);
        (1..=max)
            .rev()
            .find_map(|base2k| self.with_base2k(base2k).ok())
            .with_context(|| format!("no radix up to {max} fits the modulus bounds of {}", self.spec.name))
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
                // With the plan's own schedule: the tree of power-of-two keys switches more.
                let index = LinearTransformationLayout {
                    indexes,
                    slots: 1 << log_n,
                    strategy: dft.factor_strategy(giant_step),
                }
                .index();
                let diagonals: usize = index.index.iter().map(Vec::len).sum();
                let giants = index.index.iter().filter(|babies| !babies.is_empty()).count();
                let switches = match index.rotations {
                    BsgsRotations::PerRotation => giants,
                    BsgsRotations::PowersOfTwo { .. } => index.rotations.giant_key_switches(
                        index
                            .giant_steps
                            .iter()
                            .zip(&index.index)
                            .filter(|(_, babies)| !babies.is_empty())
                            .map(|(&rot, _)| rot),
                    ),
                };
                products = products.max(diagonals * diagonal_limbs + switches * operand);
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
                slot_layout: SlotLayout::Full,
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
    build(
        "n16_d35_k636_p20_c2s",
        PresetSpec {
            name: "n16_d35_k636_p20_c2s",
            log_n: 16,
            base2k: 52,
            rank: 1,
            log_delta: 35,
            output_k: 636,
            log2_precision: 20,
            derived: false,
            dense_hamming_weight: DENSE_HAMMING_WEIGHT,
            sparse_hamming_weight: SPARSE_HAMMING_WEIGHT,
            bounds: ModulusBounds::Certified(Security::Bits128),
            max_dsize: None,
            dense_to_sparse_dsize: 1,
            pipeline: BootstrappingPipeline::C2SFirst,
            log_msg_ratio: 6,
            c2s_schedule: &C2S_SCHEDULE,
            c2s_first_factor_on_raise: false,
            power_of_two_keys: false,
            eval_mod_ring: EvalModRing::Standard,
            c2s_guard_bits: 0,
            c2s_log_delta: 51,
            c2s_log_budget: 2,
            c2s_bypass_log_delta: None,
            s2c_schedule: &S2C_SCHEDULE,
            s2c_log_delta: 35,
            s2c_log_budget: 2,
            input_headroom: 0,
            log_gap: 0,
            eval_mod: optimized_han_ki(),
        },
    )
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
    build(
        "n16_d35_k733_p20_s2c",
        PresetSpec {
            name: "n16_d35_k733_p20_s2c",
            log_n: 16,
            base2k: 52,
            rank: 1,
            log_delta: 35,
            output_k: 733,
            log2_precision: 20,
            derived: false,
            dense_hamming_weight: DENSE_HAMMING_WEIGHT,
            sparse_hamming_weight: SPARSE_HAMMING_WEIGHT,
            bounds: ModulusBounds::Certified(Security::Bits128),
            max_dsize: None,
            dense_to_sparse_dsize: 1,
            pipeline: BootstrappingPipeline::S2CFirst,
            log_msg_ratio: 13,
            c2s_schedule: &C2S_SCHEDULE,
            c2s_first_factor_on_raise: false,
            power_of_two_keys: false,
            eval_mod_ring: EvalModRing::Standard,
            c2s_guard_bits: 6,
            c2s_log_delta: 50,
            c2s_log_budget: 3,
            c2s_bypass_log_delta: None,
            s2c_schedule: &S2C_SCHEDULE,
            s2c_log_delta: 30,
            s2c_log_budget: 2,
            input_headroom: 0,
            log_gap: 0,
            eval_mod: optimized_han_ki(),
        },
    )
}

/// C2S-first full-slot preset at `N = 2^15` for inputs at scale `2^35`,
/// producing 180-bit ciphertexts with at least 18 bits of precision.
///
/// The input and raised widths are 40 and 780 bits. The output has 140 usable
/// bits (four levels) at the input scale. Total key moduli are bounded by
/// 851 bits for the weight-1024 dense secret and 162 bits for the weight-32 sparse secret.
pub fn n15_d35_k180_p18_c2s() -> Result<BootstrappingPreset> {
    build(
        "n15_d35_k180_p18_c2s",
        PresetSpec {
            name: "n15_d35_k180_p18_c2s",
            log_n: 15,
            base2k: 52,
            rank: 1,
            log_delta: 35,
            output_k: 180,
            log2_precision: 18,
            derived: false,
            dense_hamming_weight: DENSE_HAMMING_WEIGHT,
            sparse_hamming_weight: SPARSE_HAMMING_WEIGHT,
            bounds: ModulusBounds::Certified(Security::Bits128),
            max_dsize: None,
            dense_to_sparse_dsize: 1,
            pipeline: BootstrappingPipeline::C2SFirst,
            log_msg_ratio: 5,
            c2s_schedule: &[(7, 2048), (7, 16)],
            c2s_first_factor_on_raise: false,
            power_of_two_keys: false,
            eval_mod_ring: EvalModRing::Standard,
            c2s_guard_bits: 0,
            c2s_log_delta: 50,
            c2s_log_budget: 2,
            c2s_bypass_log_delta: None,
            s2c_schedule: &[(7, 16), (7, 2048)],
            s2c_log_delta: 29,
            s2c_log_budget: 2,
            input_headroom: 0,
            log_gap: 0,
            eval_mod: EvalModSpec {
                log_delta: 53,
                ..optimized_han_ki()
            },
        },
    )
}

/// Insecure S2C-first preset at `N = 2^11` for the test suite: the reference
/// S2C-first spec on a ring small enough for a non-ignored test, so the key
/// shapes the presets derive (narrow conjugation and SlotsToCoeffs keys) run
/// end to end. Its secret bounds are placeholders, not security limits; the
/// dense one leaves the 800-bit raised width the margin of a few digits.
#[cfg(feature = "test-utils")]
pub(crate) fn insecure_n11_d35_k180_p20_s2c() -> Result<BootstrappingPreset> {
    build(
        "insecure_n11_d35_k180_p20_s2c",
        PresetSpec {
            name: "insecure_n11_d35_k180_p20_s2c",
            log_n: 11,
            base2k: 52,
            rank: 1,
            log_delta: 35,
            output_k: 180,
            log2_precision: 20,
            derived: false,
            dense_hamming_weight: 192,
            sparse_hamming_weight: SPARSE_HAMMING_WEIGHT,
            bounds: ModulusBounds::Placeholder {
                dense: 1024,
                sparse: 1024,
            },
            max_dsize: None,
            dense_to_sparse_dsize: 1,
            pipeline: BootstrappingPipeline::S2CFirst,
            log_msg_ratio: 13,
            c2s_schedule: &[(4, 4), (3, 4), (3, 4)],
            c2s_first_factor_on_raise: false,
            power_of_two_keys: false,
            eval_mod_ring: EvalModRing::Standard,
            c2s_guard_bits: 6,
            c2s_log_delta: 50,
            c2s_log_budget: 3,
            c2s_bypass_log_delta: None,
            s2c_schedule: &[(2, 4), (4, 4), (4, 4)],
            s2c_log_delta: 30,
            s2c_log_budget: 2,
            input_headroom: 0,
            log_gap: 0,
            eval_mod: optimized_han_ki(),
        },
    )
}

/// Insecure C2S-first preset at `N = 2^11` for the test suite: [`n16_d35_k636_p20_c2s`]
/// on the ring and with the transform factors of [`insecure_n11_d35_k180_p20_s2c`],
/// so the key shapes of a C2S-first plan, SlotsToCoeffs keys used below EvalMod
/// among them, run end to end. Its secret bounds are placeholders, not security limits.
#[cfg(feature = "test-utils")]
pub(crate) fn insecure_n11_d35_k100_p17_c2s() -> Result<BootstrappingPreset> {
    let s2c = insecure_n11_d35_k180_p20_s2c()?.spec;
    build(
        "insecure_n11_d35_k100_p17_c2s",
        PresetSpec {
            name: "insecure_n11_d35_k100_p17_c2s",
            log_n: s2c.log_n,
            output_k: 100,
            log2_precision: 17,
            derived: false,
            dense_hamming_weight: s2c.dense_hamming_weight,
            sparse_hamming_weight: s2c.sparse_hamming_weight,
            bounds: s2c.bounds,
            c2s_schedule: s2c.c2s_schedule,
            s2c_schedule: s2c.s2c_schedule,
            ..n16_d35_k636_p20_c2s()?.spec
        },
    )
}

/// [`n16_d35_k733_p20_s2c`] tuned for speed: 685-bit ciphertexts with at least
/// 19 bits of precision (19.6 measured, the minimum over the slots), 15 usable levels. The first CoeffsToSlots factor is
/// applied by the raise, four factors of three layers follow, EvalMod runs once
/// on the conjugate-invariant ring ([`EvalModRing::ConjugateInvariant`], 84 bits
/// of conversions), and SlotsToCoeffs has four factors. The raised width is 1431 bits.
pub fn n16_d35_k685_p19_s2c_ffr_ci() -> Result<BootstrappingPreset> {
    build(
        "n16_d35_k685_p19_s2c_ffr_ci",
        PresetSpec {
            name: "n16_d35_k685_p19_s2c_ffr_ci",
            output_k: 685,
            log2_precision: 19,
            derived: false,
            c2s_schedule: &[(3, 1), (3, 2048), (3, 256), (3, 32), (3, 4)],
            c2s_first_factor_on_raise: true,
            power_of_two_keys: false,
            // `u` carries the overflow of the raise, `v` only the message.
            eval_mod_ring: EvalModRing::ConjugateInvariant {
                pre: CIConversion::Product { log_delta: 46 },
                post: CIConversion::Product { log_delta: 38 },
            },
            c2s_log_delta: 48,
            s2c_schedule: &[(3, 4), (4, 32), (4, 512), (4, 8192)],
            s2c_log_delta: 28,
            ..n16_d35_k733_p20_s2c()?.spec
        },
    )
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

/// Builds the preset of `spec`, its security refusals naming `op`, the public method that
/// constructs or re-derives it.
fn build(op: &'static str, spec: PresetSpec) -> Result<BootstrappingPreset> {
    ensure!(spec.base2k > 0, "bootstrapping preset base2k must be nonzero");
    ensure!(spec.rank == 1, "bootstrapping presets currently require rank 1");
    ensure!(
        spec.max_dsize != Some(0),
        "bootstrapping preset key dsize cap must be nonzero"
    );
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
    })?
    .with_power_of_two_keys(spec.power_of_two_keys);
    let coeffs_to_slots = DFTPlan::new(
        DFTType::Encode,
        spec.c2s_schedule.to_vec(),
        DFTOutputFormat::SplitRealAndImag,
        CoeffsMeta::from_delta_budget(spec.c2s_log_delta, spec.c2s_log_budget),
    )?
    .with_power_of_two_keys(spec.power_of_two_keys);
    let coeffs_to_slots = if spec.c2s_first_factor_on_raise {
        coeffs_to_slots.with_first_factor_on_raise()
    } else {
        coeffs_to_slots
    };
    let plan = BootstrappingPlan::new(
        spec.pipeline,
        BootstrappingTechniques {
            sparse_secret_encapsulation: Some(SparseSecretEncapsulation {
                hamming_weight: spec.sparse_hamming_weight,
            }),
            eval_round_plus: match spec.c2s_bypass_log_delta {
                Some(log_delta) => Some(EvalRoundPlus {
                    coeffs_to_slots_bypass: DFTPlan::new(
                        DFTType::Encode,
                        spec.c2s_schedule.to_vec(),
                        DFTOutputFormat::SplitRealAndImag,
                        CoeffsMeta::from_delta_budget(log_delta, spec.c2s_log_budget),
                    )?
                    .with_scaling(1.0)?
                    .with_power_of_two_keys(spec.power_of_two_keys),
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
    let plan = plan.with_eval_mod_ring(spec.eval_mod_ring)?.with_log_gap(spec.log_gap)?;
    let log_slots = spec
        .log_n
        .checked_sub(1 + spec.log_gap)
        .context("bootstrapping preset log_n must exceed its gap")?;
    ensure!(
        plan.coeffs_to_slots().log_slots() == log_slots && plan.slots_to_coeffs().log_slots() == log_slots,
        "bootstrapping preset DFT schedules must cover {log_slots} slot layers"
    );

    let log_modulus = spec
        .log_delta
        .checked_add(spec.log_msg_ratio)
        .context("bootstrapping preset input modulus overflow")?;
    let input_k = plan.input_k(log_modulus) + spec.input_headroom;
    let output_k = spec.output_k;
    ensure!(
        output_k >= input_k,
        "bootstrapping preset output width {output_k} is below its input width {input_k}"
    );
    let bootstrap_k = plan.bootstrap_k(output_k, spec.log_delta);
    let refused = |source| CKKSCompositionError::SecurityLevelNotMet { op, source };
    let (dense_log_modulus, sparse_log_modulus) = spec.log_modulus_bounds(n).map_err(refused)?;
    let keys_layout =
        spec.key_policy(dense_log_modulus, sparse_log_modulus)
            .keys_layout(&plan, bootstrap_k, input_k, spec.log_delta)?;

    let plan = match spec.security() {
        Some(security) => {
            let dense = spec.dense_secret(n);
            security
                .check_log_q(dense, "raised ciphertext", bootstrap_k)
                .map_err(refused)?;
            plan.check_key_security_for(op, security, dense, &keys_layout)?;
            plan.with_security(security)
        }
        None => plan,
    };
    Ok(BootstrappingPreset {
        spec,
        n,
        plan,
        keys_layout,
        input_k,
        output_k,
        bootstrap_k,
        dense_log_modulus,
        sparse_log_modulus,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        CKKSCompositionError,
        layouts::{CKKSFoldKeysLayout, KeySecret, RingSwitchKeys},
    };
    use poulpy_core::{
        CoreError,
        layouts::{Dnum, GGLWELayout, LWEInfos},
    };

    /// Gadget precision of `key` (see [`poulpy_core::layouts::GGLWELayout::gadget_k`]).
    fn gadget_k<K: GGLWEInfos>(key: &K) -> usize {
        key.gglwe_layout().gadget_k().as_usize()
    }
    use crate::CKKSInfos;
    use crate::presets::first_factor::FIRST_FACTOR_GUARD_BITS;

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
        assert_eq!(preset.log2_precision(), Some(18));
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
        let wide = preset.widest_rotation_key();
        assert_eq!(wide.dsize().as_usize(), 1);
        assert_eq!(gadget_k(wide), 780);
        assert_eq!(wide.k_aux.as_usize(), 67);
        assert_eq!(wide.k().as_usize(), 847);
        // EvalMod starts 100 bits below the raised width: room for a 2-limb digit.
        assert_eq!(key_shapes([&keys.tensor_key]), [(104, 7)]);
        // The raise switches to the dense secret: two rows under the raised ciphertext, ending at the
        // raised width, at the modulus a key covering it would have.
        assert_eq!(sparse_to_dense_shape(&preset), (52, 2, 676, 847));
        assert_eq!(encapsulation.dense_to_sparse.k().as_usize(), 119);
        // Under a cap of one limb every high-modulus key takes one, and a wider
        // dense-to-sparse digit leaves less than a digit of guard under the sparse bound.
        let capped = preset.with_dsizes(1, 1).unwrap();
        assert!(
            capped
                .keys_layout()
                .rotation_key_layouts()
                .all(|key| key.dsize().as_usize() == 1)
        );
        assert!(preset.with_dsizes(1, 2).is_err());
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
        assert_eq!(preset.log2_precision(), Some(20));
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
            key_shapes(keys.rotation_key_layouts()),
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
        assert_eq!(preset.log2_precision(), Some(20));
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
            key_shapes(keys.rotation_key_layouts()),
            [(208, 7), (260, 5), (416, 3), (156, 1), (104, 1)]
        );
        assert_eq!(key_shapes([&keys.tensor_key]), [(416, 3)]);
        assert_eq!(keys.rotation_keys[&-1].gglwe_layout().max_input_k().as_usize(), 1248);
        assert!(keys.tensor_key.gglwe_layout().max_input_k().as_usize() >= 1207);

        // A cap of four limbs keeps the keys of the raised width and narrows the digits of the others.
        let capped = preset.with_dsizes(4, 1).unwrap();
        assert_eq!(
            key_shapes(capped.keys_layout().rotation_key_layouts()),
            [(208, 7), (208, 6), (156, 1), (104, 1)]
        );
        assert_eq!(key_shapes([&capped.keys_layout().tensor_key]), [(208, 6)]);
    }

    #[test]
    fn n16_d35_k685_p19_s2c_ffr_ci_is_composable_and_within_bounds() {
        let preset = n16_d35_k685_p19_s2c_ffr_ci().unwrap();

        assert_eq!(preset.plan().pipeline(), BootstrappingPipeline::S2CFirst);
        assert_eq!(preset.plan().eval_mod().eval_mod_type, EvalModType::CosHKEven);
        assert_eq!(preset.plan().eval_mod().consumed_bits(), 464);
        assert_eq!(preset.plan().eval_mod_ring().consumed_bits(), 84);
        assert_eq!(preset.plan().coeffs_to_slots().consumed_bits(), 192);
        assert_eq!(preset.plan().c2s_guard_bits(), 6);
        assert_eq!(preset.plan().slots_to_coeffs().consumed_bits(), 112);
        assert_eq!((preset.input_k(), preset.output_k(), preset.bootstrap_k()), (160, 685, 1431));
        assert_eq!(preset.log2_precision(), Some(19));
        assert_eq!(
            preset.output_layout().log_budget() - preset.input_layout().log_budget(),
            15 * preset.log_delta()
        );

        // The keys of the raised width, under the dense bound.
        assert_eq!((preset.max_dense_modulus(), preset.max_sparse_modulus()), (1707, 342));
        let keys = preset.keys_layout();
        let wide = preset.widest_rotation_key();
        assert_eq!(
            (
                wide.k_digit.as_usize(),
                wide.dnum.as_usize(),
                wide.k_aux.as_usize(),
                wide.k().as_usize()
            ),
            (208, 7, 224, 1680)
        );
        assert!(wide.k_aux >= wide.k_digit);
        assert_eq!(
            key_shapes(keys.rotation_key_layouts()),
            [(208, 7), (260, 5), (416, 3), (104, 1), (52, 1)]
        );
        assert_eq!(key_shapes([&keys.tensor_key]), [(416, 3)]);
        assert!(keys.tensor_key.gglwe_layout().max_input_k().as_usize() >= 1248);
        // The first factor's keys: one row each, below the raised width.
        let first = keys.first_factor_key.unwrap();
        assert_eq!(
            (first.k_digit.as_usize(), first.dnum.as_usize(), first.k_offset.as_usize()),
            (104, 1, 1352)
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
    /// guarded by at least a digit under the dense bound, which a cap of their
    /// own digit keeps.
    fn assert_layouts_within_bounds(preset: &BootstrappingPreset, wide: (usize, usize, usize, usize)) {
        let keys = preset.keys_layout();
        let encapsulation = keys.encapsulation.as_ref().unwrap();

        assert_eq!((preset.max_dense_modulus(), preset.max_sparse_modulus()), (1707, 342));
        let key = preset.widest_rotation_key();
        assert_eq!(
            (
                key.k_digit.as_usize(),
                key.dnum.as_usize(),
                key.k_aux.as_usize(),
                key.k().as_usize()
            ),
            wide
        );
        assert!(key.k_aux >= key.k_digit);
        // One row under the raised ciphertext, below the modulus of the keys covering the raised width.
        assert!(encapsulation.sparse_to_dense.unwrap().k() <= key.k());
        let capped = preset.with_dsizes(key.dsize().as_usize(), 1).unwrap();
        assert_eq!(capped.widest_rotation_key(), key);
        assert!(
            capped
                .keys_layout()
                .rotation_key_layouts()
                .all(|other| other.dsize() <= key.dsize())
        );

        assert_eq!(encapsulation.dense_to_sparse.dsize().as_usize(), 1);
        assert_eq!(gadget_k(&encapsulation.dense_to_sparse), 52);
        assert_eq!(encapsulation.dense_to_sparse.k_aux.as_usize(), 68);
        assert_eq!(encapsulation.dense_to_sparse.k().as_usize(), 120);
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
        assert!(base.keys_layout().first_factor_key.is_none());
        // The first-factor keys switch to the dense secret: no sparse-to-dense key.
        assert!(keys.encapsulation.unwrap().sparse_to_dense.is_none());
        assert!(base.keys_layout().encapsulation.unwrap().sparse_to_dense.is_some());

        // The first paid factor runs at the raised width, where the first factor ran before.
        assert!(preset.widest_rotation_key().gglwe_layout().max_input_k().as_usize() >= preset.bootstrap_k());
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
        let rotations = keys.rotation_keys.values().map(key_bytes).sum();
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
        // 51 rotation keys, the tensor key and the encapsulation keys: 7380 MiB.
        assert_eq!(key_set_bytes(&base).map(mib), [7249, 99, 32, 0, 0]);
        // The two remaining factors of six layers take 60 rotation keys, and the eight
        // diagonals of the first one a key of 30 MiB each in place of the
        // sparse-to-dense key of 29: 7388 MiB.
        assert_eq!(key_set_bytes(&preset).map(mib), [7046, 99, 3, 240, 0]);
    }

    #[test]
    fn a_gap_below_the_sparse_bound_of_its_keys_is_refused() {
        const C2S: &[(usize, usize)] = &[(3, 1), (3, 512), (3, 64), (2, 8), (2, 2)];
        const S2C: &[(usize, usize)] = &[(3, 4), (4, 32), (6, 1024)];
        // The weight-32 secret of degree 2^14 is bounded at 77 bits, below a dense-to-sparse key of one digit.
        // The shaping refuses it before the certification sees a key.
        let Err(err) = n16_d35_k733_p20_s2c().unwrap().with_log_gap(2, C2S, S2C) else {
            panic!("the dense-to-sparse key fits a 77-bit bound");
        };
        assert_eq!(
            err.to_string(),
            "dense-to-sparse key: a digit of 52 bits on a 48-bit input needs a modulus of 104 bits, above 77"
        );
    }

    #[test]
    fn a_gap_to_a_degree_without_bounds_is_refused_by_the_certification() {
        const C2S: &[(usize, usize)] = &[(3, 1), (3, 64), (2, 8), (2, 2)];
        const S2C: &[(usize, usize)] = &[(3, 4), (3, 32), (4, 512)];
        // The table has no row at degree 2^11, where a gap of 5 puts the sparse secret.
        let Err(err) = n16_d35_k733_p20_s2c().unwrap().with_log_gap(5, C2S, S2C) else {
            panic!("a sparse secret of degree 2^11 is certified");
        };
        let (op, source) = security_refusal(&err);
        assert_eq!(op, "with_log_gap");
        assert!(
            matches!(
                source,
                CoreError::SecurityUnsupported {
                    security: Security::Bits128,
                    secret: SecretSpec {
                        n: 2048,
                        dist: Distribution::TernaryFixed(32),
                    },
                    reason: "the table has no bound for this secret at this degree",
                }
            ),
            "{source:?}"
        );
    }

    #[test]
    fn fold_keys_are_checked_under_the_secret_of_their_samples() {
        // Inputs of degree 2^14 folded into the logN16 C2S-first preset: the outbound key covers the
        // 636-bit output under the input secret, bounded at 424 bits at that degree.
        let preset = n16_d35_k636_p20_c2s().unwrap();
        let key = |n: u32, dnum: u32| GGLWELayout {
            n: Degree(n),
            base2k: Base2K(52),
            dnum: Dnum(dnum),
            k_aux: TorusPrecision(52 + 16),
            rank_in: Rank(1),
            rank_out: Rank(1),
            k_digit: TorusPrecision(52),
            k_offset: TorusPrecision(0),
            stride: 1,
        };
        let input = SecretSpec::new(1 << 14, Distribution::TernaryFixed(1024));
        let bootstrap = SecretSpec::new(preset.n(), Distribution::TernaryFixed(1024));
        let mut fold = CKKSFoldKeysLayout {
            ring_switch: Some(RingSwitchKeys {
                inbound: key(1 << 16, 2),
                outbound: key(1 << 16, 13),
            }),
            automorphism: Some(key(1 << 14, 2)),
        };
        let err = fold
            .check_security(Security::Bits128, input, bootstrap)
            .expect_err("an outbound key past the input secret's bound is certified");
        assert_eq!(
            security_refusal(&err),
            (
                "check_security",
                &CoreError::SecurityBoundExceeded {
                    security: Security::Bits128,
                    secret: input,
                    what: "outbound ring-switch key".into(),
                    log_q: 13 * 52 + 68,
                    max_log_q: 424,
                }
            )
        );
        // Under that bound, every key is certified.
        fold.ring_switch.as_mut().unwrap().outbound = key(1 << 16, 6);
        fold.check_security(Security::Bits128, input, bootstrap).unwrap();
    }

    /// The `op` and source of the [`CKKSCompositionError::SecurityLevelNotMet`] `err` holds.
    fn security_refusal(err: &anyhow::Error) -> (&'static str, &CoreError) {
        match err.downcast_ref::<CKKSCompositionError>() {
            Some(CKKSCompositionError::SecurityLevelNotMet { op, source }) => (op, source),
            _ => panic!("expected a security refusal, got {err:#}"),
        }
    }

    #[test]
    fn presets_certify_every_key_and_the_plan_carries_the_level() {
        for preset in all().unwrap() {
            assert_eq!(preset.security(), Some(Security::Bits128));
            assert_eq!(preset.plan().security(), Some(Security::Bits128));
            let dense = SecretSpec::new(preset.n(), Distribution::TernaryFixed(1024));
            let keys = preset.keys_layout();
            preset.plan().check_key_security(Security::Bits128, dense, keys).unwrap();
            // Every key is listed: the rotation keys and one tensor key, with the encapsulation of every preset.
            let listed = keys.keys_by_secret();
            assert_eq!(
                listed.iter().filter(|(name, ..)| *name == "rotation").count(),
                keys.rotation_key_layouts().count()
            );
            assert!(
                listed
                    .iter()
                    .any(|&(name, secret, _)| name == "dense-to-sparse" && secret == KeySecret::Sparse)
            );
        }
    }

    #[test]
    fn a_key_one_bit_over_its_bound_or_an_unbounded_secret_is_refused() {
        let preset = n15_d35_k180_p18_c2s().unwrap();
        let dense = SecretSpec::new(preset.n(), Distribution::TernaryFixed(1024));
        let mut keys = preset.keys_layout().clone();
        // Widen the widest rotation key's guard to one bit past the dense bound.
        let wide = *preset.widest_rotation_key();
        let over = preset.max_dense_modulus() + 1 - wide.k().as_usize();
        for key in keys.rotation_keys.values_mut().filter(|key| **key == wide) {
            key.k_aux = TorusPrecision(wide.k_aux.0 + over as u32);
        }
        let err = preset
            .plan()
            .check_key_security(Security::Bits128, dense, &keys)
            .expect_err("a key past the bound is certified");
        let (op, source) = security_refusal(&err);
        assert_eq!(op, "check_key_security");
        assert_eq!(
            source,
            &CoreError::SecurityBoundExceeded {
                security: Security::Bits128,
                secret: dense,
                what: "rotation key".into(),
                log_q: 852,
                max_log_q: 851,
            }
        );
        // A dense secret the table has no cell for, such as a ternary one of density 1/2, certifies nothing.
        let half = SecretSpec::new(preset.n(), Distribution::TernaryProb(0.5));
        let err = preset
            .plan()
            .check_key_security(Security::Bits128, half, preset.keys_layout())
            .expect_err("a secret without a bound is certified");
        assert_eq!(
            security_refusal(&err),
            (
                "check_key_security",
                &CoreError::SecurityUnsupported {
                    security: Security::Bits128,
                    secret: half,
                    reason: "the table has no secret of this distribution",
                }
            )
        );
        // Without the level, the plan checks nothing.
        assert_eq!(preset.plan().clone().without_security().security(), None);
    }

    #[test]
    fn a_circuit_trims_n16_d35_k685_p19_s2c_ffr_ci_to_what_it_spends() {
        let preset = n16_d35_k685_p19_s2c_ffr_ci()
            .unwrap()
            .with_slots_to_coeffs(&S2C_SCHEDULE)
            .unwrap()
            .with_input_headroom(20)
            .unwrap()
            .with_output_k(467)
            .unwrap();
        // Three factors of 28 bits below the input, 20 for the circuit, 48 at the raise.
        assert_eq!((preset.input_k(), preset.output_k(), preset.bootstrap_k()), (152, 467, 1213));
        // Every rotation of SlotsToCoeffs alone reads one digit at its width.
        let keys = preset.keys_layout();
        assert!(keys.rotation_keys.values().all(|key| key.dnum.as_usize() <= 3));
        // The giant steps of CoeffsToSlots, one factor scale below their baby steps, fit two digits.
        assert_eq!(keys.rotation_keys.values().filter(|key| key.dnum.as_usize() == 2).count(), 3);
        let Err(err) = n16_d35_k636_p20_c2s().unwrap().with_input_headroom(20) else {
            panic!("a C2S-first preset takes no headroom");
        };
        assert_eq!(
            err.to_string(),
            "only an S2C-first bootstrap runs a circuit between SlotsToCoeffs and the raise"
        );
        // Four factors of 34 bits for four of 48, then one squaring, one level of 58 bits, fewer.
        let scales = PresetScales {
            c2s_log_delta: 34,
            ..preset.scales()
        };
        let narrower = preset.with_scales(scales).unwrap();
        assert_eq!(narrower.bootstrap_k(), 1213 - 4 * 14);
        let shallower = narrower.with_eval_mod_polynomial(14, 26, 2).unwrap();
        assert_eq!(shallower.bootstrap_k(), 1213 - 4 * 14 - 58);
    }

    #[test]
    fn n16_d35_k685_p19_s2c_ffr_ci_runs_eval_mod_on_the_ci_ring() {
        let preset = n16_d35_k685_p19_s2c_ffr_ci().unwrap();
        assert_eq!((preset.input_k(), preset.output_k(), preset.bootstrap_k()), (160, 685, 1431));
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
        assert_eq!((folded.input_k(), folded.output_k(), folded.bootstrap_k()), (160, 685, 1385));
        assert_eq!(folded.plan().eval_mod_ring().consumed_bits(), 38);
        // The rotation keys follow the narrower raised width: 4551 MiB against 5351.
        assert_eq!(key_set_bytes(&folded).map(|bytes| bytes >> 20), [3949, 99, 3, 240, 260]);
    }

    #[test]
    fn power_of_two_keys_halve_the_rotation_keys_of_n16_d35_k685_p19_s2c_ffr_ci() {
        let preset = n16_d35_k685_p19_s2c_ffr_ci().unwrap();
        let powers = preset.with_power_of_two_keys().unwrap();
        let count = |p: &BootstrappingPreset| p.plan().galois_elements(p.log_n(), 2 * p.n() as i64).len();
        // Fewer keys per paid factor: 21 rotation keys against 35, 3149 MiB against 4741,
        // every width unchanged.
        assert_eq!((count(&preset), count(&powers)), (35, 21));
        assert_eq!(key_set_bytes(&powers).map(|bytes| bytes >> 20), [3149, 99, 3, 248, 260]);
        assert_eq!((powers.input_k(), powers.output_k(), powers.bootstrap_k()), (160, 685, 1431));
    }

    #[test]
    fn n16_d35_k685_p19_s2c_ffr_ci_has_a_smaller_key_set() {
        let mib = |bytes: usize| {
            assert!(bytes.is_multiple_of(1 << 20));
            bytes >> 20
        };
        // Rotation keys with conjugation, the tensor key and the encapsulation keys: 7092 MiB.
        assert_eq!(key_set_bytes(&n16_d35_k733_p20_s2c().unwrap()).map(mib), [6961, 99, 32, 0, 0]);
        // 35 rotation keys and no conjugation, the CI tensor key, the dense-to-sparse key,
        // eight first-factor keys and the trace and embed keys of degree 2N: 5351 MiB.
        let preset = n16_d35_k685_p19_s2c_ffr_ci().unwrap();
        assert_eq!(key_set_bytes(&preset).map(mib), [4741, 99, 3, 248, 260]);
        // With the first factor evaluated by the matrix, every paid factor runs 48 bits
        // wider and the sparse-to-dense key comes back: 6506 MiB.
        let without = build(
            "test",
            PresetSpec {
                c2s_schedule: &[(3, 8), (3, 2048), (3, 256), (3, 32), (3, 4)],
                c2s_first_factor_on_raise: false,
                power_of_two_keys: false,
                ..preset.spec
            },
        )
        .unwrap();
        assert_eq!(without.bootstrap_k(), 1479);
        assert_eq!(key_set_bytes(&without).map(mib), [6113, 99, 34, 0, 260]);
    }

    #[test]
    fn all_lists_every_preset_once() {
        let names: Vec<&str> = all().unwrap().iter().map(|p| p.name().unwrap()).collect();
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
            let (keys, wide) = (fft.keys_layout(), fft.widest_rotation_key());
            assert_eq!(wide.dnum.as_usize(), preset.bootstrap_k().div_ceil(fft_dsize * 18));
            // A digit and `log_n` bits of guard, or what the bound leaves.
            let gadget = wide.dnum.as_usize() * fft_dsize * 18;
            assert_eq!(
                wide.k_aux.as_usize(),
                (fft_dsize * 18 + preset.log_n()).min(fft.max_dense_modulus() - gadget)
            );

            let small_key = &keys.encapsulation.as_ref().unwrap().dense_to_sparse;
            assert_eq!(
                small_key.k().as_usize(),
                preset.log_modulus().div_ceil(18) * 18 + 18 + preset.log_n()
            );

            // A cap at the widest digit changes no key, and an oversized dense-to-sparse digit is rejected.
            let widest = preset
                .keys_layout()
                .rotation_key_layouts()
                .map(|key| key.dsize().as_usize())
                .max();
            let uncapped = preset.with_dsizes(widest.unwrap(), 1).unwrap();
            assert!(
                uncapped
                    .keys_layout()
                    .rotation_key_layouts()
                    .eq(preset.keys_layout().rotation_key_layouts())
            );
            assert!(preset.with_dsizes(1, 7).is_err());
        }
    }
}
