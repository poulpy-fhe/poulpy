//! PaCo bootstrapping plans: level schedules, budget sizing, and key-material
//! enumeration.
//!
//! A [`PaCoPlan`] carries the instance dimensions (`N`, `h`, `C`, `q`) and
//! the full evaluation parameterization, mirroring the standard bootstrap's
//! [`DFTPlan`](crate::layouts::DFTPlan): one [`PaCoDFTPlan`] per factor chain
//! (evaluation-order factorization schedule, per-factor BSGS giant steps,
//! per-chain plaintext scale and matrix scaling), plus the β/bsk scale
//! [`PaCoPlan::log_delta_bsk`] anchoring the phase stages. From it derive the
//! budget the pipeline consumes ([`PaCoPlan::consumed_bits`]), a canonical
//! bootstrap width ([`PaCoPlan::k_boot`]), and the exact Galois-element set
//! the evaluation needs ([`PaCoPlan::galois_elements`]).
//!
//! The factorization schedules cover **unit lists** that include the fused
//! pseudo-layers as first-class, schedulable units:
//!
//! ```text
//! c2s units (evaluation order):  [L_1, …, L_{log 2C}, ψ]
//! stc units (evaluation order):  [pack, L_1, …, L_{log C − 1}]
//! ```
//!
//! `L_i` are the radix-2 butterfly layers; `ψ` is the ψ-pairing + μ mask
//! pseudo-unit (always in the last c2s group: merged with butterfly layers it
//! becomes the conjugation-augmented `(A, B)` pair, spending a second
//! diagonal matrix at that factor to erase the μ level; alone — last entry
//! `1` — it is evaluated as the paper's operation-lean unfused tail, one
//! fused conj-rotate keyswitch plus one μ multiplication); `pack` is the
//! η-routing + pair-packing pseudo-unit (in the first stc group, composed on
//! its input side — alone, it is a standalone first factor). This makes the
//! speed↔depth balance of every level an explicit user choice instead of an
//! implicit greedy grouping.

use std::collections::BTreeSet;

use anyhow::{Context, Result, ensure};
use poulpy_core::layouts::{LinearTransformationLayout, LinearTransformationStrategy};
use poulpy_hal::layouts::{galois_element, galois_elements_from_rotations};

use crate::default::paco::{
    lt::{PaCoPsiTail, paco_psi_c2s_factors, paco_stc_factors},
    ops::{conj_rotate_galois_element, fold_rotations},
};
use crate::layouts::{ComplexDiagonals, dft::FactorSchedule};

/// One factor chain's full parameterization (mirrors
/// [`DFTPlan`](crate::layouts::DFTPlan)).
#[derive(Clone, Debug)]
pub struct PaCoDFTPlan {
    /// Factorization schedule and per-factor BSGS widths over the chain's
    /// unit list, in **evaluation order** (see the module docs for the unit
    /// lists). Shape invariants are owned by [`FactorSchedule`] — the same
    /// validator [`DFTPlan`](crate::layouts::DFTPlan) uses. For a giant-step
    /// heuristic, feed the factor's diagonal indexes to
    /// [`optimal_bsgs_giant_step`](poulpy_core::layouts::optimal_bsgs_giant_step).
    schedule: FactorSchedule,
    /// Plaintext encode scale of this chain's factors; each of the chain's
    /// levels consumes this many budget bits.
    log_delta: usize,
    /// Remaining plaintext budget carried by every encoded factor. Together
    /// with `log_delta`, this determines the torus width required to compile
    /// the factor plaintexts.
    log_budget: usize,
    /// Multiplicative constant the chain's matrix product carries (mirrors
    /// `DFTPlan::scaling`), distributed uniformly across the factors (the
    /// ψ-pair counts as one factor: `A` and `B` both get the last group's
    /// share). Structural constants (η's `1/π` on the stc chain) are applied
    /// on top, not replaced. See [`PaCoPlan::extra_scale_log2`] for the
    /// output-constant accounting.
    scaling: f64,
}

impl PaCoDFTPlan {
    /// Creates a factor-chain plan.
    ///
    /// The schedule must contain at least one non-empty factor, its BSGS
    /// widths must be non-zero and parallel to the schedule, the plaintext
    /// scale and budget must fit in a `usize`, and `scaling` must be positive
    /// and finite. The PaCo-specific unit count is checked when the chain is
    /// attached to a [`PaCoPlan`] with [`PaCoPlan::with_evaluation`].
    pub fn new(
        factorization_depth: Vec<usize>,
        giant_steps: Vec<usize>,
        log_delta: usize,
        log_budget: usize,
        scaling: f64,
    ) -> Result<Self> {
        let plan = Self {
            schedule: FactorSchedule::from_parts(&factorization_depth, &giant_steps, "PaCo DFT plan")?,
            log_delta,
            log_budget,
            scaling,
        };
        plan.check_shape("PaCo DFT plan")?;
        Ok(plan)
    }

    /// A balanced uniform schedule: `⌈units/g⌉` groups whose sizes differ by
    /// at most one (the remainder is spread across the leading groups, never
    /// dumped into a trailing group), all with the same `giant_step`.
    pub fn uniform(units: usize, g: usize, giant_step: usize, log_delta: usize, log_budget: usize) -> Result<Self> {
        ensure!(units > 0, "a PaCo factor chain must contain at least one unit");
        ensure!(g > 0, "a PaCo uniform factor-group width must be non-zero");
        let g = g.clamp(1, units);
        let groups = units.div_ceil(g);
        let (base, rem) = (units / groups, units % groups);
        let factorization_depth: Vec<usize> = (0..groups).map(|i| base + usize::from(i < rem)).collect();
        let giant_steps = vec![giant_step; factorization_depth.len()];
        Self::new(factorization_depth, giant_steps, log_delta, log_budget, 1.0)
    }

    /// The factorization schedule, one [`FactorStep`](crate::layouts::dft::FactorStep)
    /// per factor in evaluation order.
    pub fn schedule(&self) -> &FactorSchedule {
        &self.schedule
    }

    /// Per-factor merged-layer counts in evaluation order (collected view of
    /// [`Self::schedule`]).
    pub fn factorization_depth(&self) -> Vec<usize> {
        self.schedule.depths()
    }

    /// Per-factor BSGS giant-step widths in evaluation order (collected view
    /// of [`Self::schedule`]).
    pub fn giant_steps(&self) -> Vec<usize> {
        self.schedule.giant_steps()
    }

    /// Plaintext scale used to encode each factor.
    pub fn log_delta(&self) -> usize {
        self.log_delta
    }

    /// Plaintext budget retained by each encoded factor.
    pub fn log_budget(&self) -> usize {
        self.log_budget
    }

    /// Multiplicative scaling carried by the full factor chain.
    pub fn scaling(&self) -> f64 {
        self.scaling
    }

    fn check_shape(&self, name: &'static str) -> Result<usize> {
        // Schedule shape invariants are single-sourced in [`FactorSchedule`];
        // only the PaCo-specific width/scaling clauses live here.
        let sum = self.schedule.check(name)?;
        let width = self
            .log_delta
            .checked_add(self.log_budget)
            .with_context(|| format!("{name}: log_delta + log_budget overflows usize"))?;
        ensure!(
            u32::try_from(width).is_ok(),
            "{name}: plaintext width {width} exceeds the layout type's u32 range",
        );
        ensure!(
            self.scaling.is_finite() && self.scaling > 0.0,
            "{name}: scaling must be positive and finite, got {}",
            self.scaling,
        );
        Ok(sum)
    }

    /// Number of factor matrices (= levels) in the chain.
    pub fn num_factors(&self) -> usize {
        self.schedule.num_factors()
    }

    /// Shape validation: non-empty, no zero-unit factor, the schedule sums to
    /// `units`, and the giant steps line up with the schedule.
    pub(crate) fn check(&self, name: &'static str, units: usize) -> Result<()> {
        let sum = self.check_shape(name)?;
        ensure!(
            sum == units,
            "{name}: factorization_depth {:?} must sum to the unit count {units}",
            self.schedule.depths(),
        );
        Ok(())
    }
}

/// The mid-pipeline slot-order convention (partial-C2S output → StC′ input).
///
/// `Natural`: every slot between the two DFT chains holds its plain-formula
/// value (`slot v·2C + i = b′_{λ_v,i}`); the StC′ input-side bit-reversal is
/// folded into the first StC′ factor, growing it toward the dense ceiling
/// `C/2` — cheap at small `C`, up to `C/2 − 3^{g}` extra diagonals at large
/// `C` with light grouping.
///
/// `BitRevLow`: the mid-pipeline is relabeled by the tiled permutation `P`
/// reversing the low `log(C/2)` bits of the slot index (the paper's extended
/// bit-reversal `Π^{(C/2)}`, conjugated into the factor chains). The StC′
/// chain consumes the `P`-ordered data natively with **no folded
/// permutation**; the input-side `P` telescopes into the cleartext packing
/// gather. Sound because every op between the relabeling points commutes with
/// `P` exactly (fold strides ≥ 2C, ψ-pairing offset `C`, μ/η strata — all at
/// bit `log(C/2)` or above; see `docs/spec/paco_dft_convention.md`).
///
/// The convention changes the BSGS diagonal offsets and hence the Galois key
/// set — callers with persisted keys must keep it stable per key bundle.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum PaCoSlotOrder {
    /// Natural mid-pipeline order; bit-reversal folded into the first StC′
    /// factor (the historical convention, and the default).
    #[default]
    Natural,
    /// Mid-pipeline relabeled by the low-`log(C/2)`-bit reversal; no folded
    /// StC′ permutation.
    BitRevLow,
}

/// The full parameterization of one PaCo bootstrap: the instance dimensions
/// (all powers of two — one sequential run bootstraps the `C` coefficients at
/// indices `i·N/C`; the parallel variant covers the rest with
/// monomial-shifted runs) plus the evaluation parameterization.
#[derive(Clone, Debug)]
pub struct PaCoPlan {
    /// `log2(N)`, the ring degree exponent.
    log_n: usize,
    /// Secret-key Hamming weight `h` (power of two, `4·h·C ≤ N`).
    h: usize,
    /// Number of coefficients bootstrapped per sequential run (`C ≥ 2`).
    c: usize,
    /// `log2(q)` of the input modulus `q = 2^log_q` (the base modulus the
    /// exhausted ciphertext lives at). Bounded by 63 because coefficient
    /// extraction uses `u64`; context compilation applies the tighter mantissa
    /// bound of the selected scalar (`52` for exact `f64` residues).
    log_q: u32,
    /// Scale of the β_t plaintexts. The bsk ciphertexts must be encrypted at
    /// this scale (validated by `PaCoContext::compile`): the
    /// accumulator inherits it and each of the `log h` `Pr` ct×ct steps
    /// consumes it — the EvalMod anchor. Requires
    /// `log_delta_bsk ≥ log q + margin` (the PaCo precision requirement).
    log_delta_bsk: usize,
    /// Torus headroom retained by the input-dependent β plaintexts. This is
    /// deliberately independent of the CoeffsToSlots factor budget.
    log_beta_budget: usize,
    /// The partial-CoeffToSlot chain, over units `[L_1, …, L_{log 2C}, ψ]`.
    c2s: PaCoDFTPlan,
    /// The SlotToCoeff′ chain, over units `[pack, L_1, …, L_{log C − 1}]`.
    stc: PaCoDFTPlan,
    /// Constructor-validated budget consumption cached to keep its infallible
    /// accessor free of assertions.
    consumed_bits: usize,
    /// Constructor-validated dyadic scale correction.
    extra_scale_log2: i64,
    /// Mid-pipeline slot-order convention (see [`PaCoSlotOrder`]).
    slot_order: PaCoSlotOrder,
    /// Memoized [`Self::galois_elements`] result (factor generation is
    /// expensive; the plan is immutable once built). Builder methods that
    /// change Galois-relevant parameters reset it.
    galois_cache: std::sync::OnceLock<Vec<i64>>,
}

impl PaCoPlan {
    /// Validates the dimensions and returns a plan with **neutral** chains
    /// (one unit per factor, giant step 1, `log_delta` 0, scaling 1) —
    /// sufficient for cleartext packing, the oracle, and structured-secret
    /// construction. Before compiling a context or constructing a key bundle,
    /// attach `log_delta_bsk`, `c2s`, and `stc` with
    /// [`Self::with_evaluation`].
    pub fn new(log_n: usize, h: usize, c: usize, log_q: u32) -> Result<Self> {
        Self::check_dimensions(log_n, h, c, log_q)?;
        let log_c = c.trailing_zeros() as usize;
        let c2s_units = log_c.checked_add(2).context("c2s unit count overflows usize")?;
        let mut plan = Self {
            log_n,
            h,
            c,
            log_q,
            log_delta_bsk: 0,
            log_beta_budget: 0,
            c2s: PaCoDFTPlan::uniform(c2s_units, 1, 1, 0, 0)?,
            stc: PaCoDFTPlan::uniform(log_c, 1, 1, 0, 0)?,
            consumed_bits: 0,
            extra_scale_log2: 0,
            slot_order: PaCoSlotOrder::default(),
            galois_cache: std::sync::OnceLock::new(),
        };
        plan.refresh_derived()?;
        Ok(plan)
    }

    /// Attaches the homomorphic evaluation parameters to this dimension plan.
    ///
    /// In addition to validating each chain's intrinsic shape, this checks
    /// that its schedule covers the exact PaCo unit list, that all level-budget
    /// products and sums are representable, and that the combined user scaling
    /// `s_c2s^h · s_stc` is a representable power of two.
    pub fn with_evaluation(
        mut self,
        log_delta_bsk: usize,
        log_beta_budget: usize,
        c2s: PaCoDFTPlan,
        stc: PaCoDFTPlan,
    ) -> Result<Self> {
        // Any evaluation-parameter change alters the factor chains, hence the
        // Galois set: drop the memoized copy.
        self.galois_cache = std::sync::OnceLock::new();
        self.log_delta_bsk = log_delta_bsk;
        self.log_beta_budget = log_beta_budget;
        self.c2s = c2s;
        self.stc = stc;
        self.refresh_derived()?;
        Ok(self)
    }

    fn check_dimensions(log_n: usize, h: usize, c: usize, log_q: u32) -> Result<usize> {
        ensure!(h.is_power_of_two(), "h must be a power of two, got {h}");
        ensure!(c.is_power_of_two(), "c must be a power of two, got {c}");
        ensure!(c >= 2, "c must be at least 2, got {c}");
        let shift = u32::try_from(log_n).context("log_n does not fit a shift count")?;
        let n = 1usize
            .checked_shl(shift)
            .with_context(|| format!("ring degree 2^{log_n} does not fit usize"))?;
        ensure!(
            u32::try_from(n).is_ok(),
            "ring degree N={n} exceeds the layout type's u32 range",
        );
        let active = 4usize
            .checked_mul(h)
            .and_then(|x| x.checked_mul(c))
            .context("4*h*c overflows usize")?;
        ensure!(active <= n, "4*h*c = {active} exceeds N = {n}");
        ensure!(0 < log_q && log_q <= 63, "log_q must be in (0, 63], got {log_q}");
        Ok(n)
    }

    /// Ring-degree exponent `log2(N)`.
    pub fn log_n(&self) -> usize {
        self.log_n
    }

    /// Secret-key Hamming weight.
    pub fn h(&self) -> usize {
        self.h
    }

    /// Number of coefficients bootstrapped by one sequential run.
    pub fn c(&self) -> usize {
        self.c
    }

    /// Input-modulus exponent `log2(q)`.
    pub fn log_q(&self) -> u32 {
        self.log_q
    }

    /// Scale of the β plaintexts and bootstrapping-key ciphertexts.
    pub fn log_delta_bsk(&self) -> usize {
        self.log_delta_bsk
    }

    /// Retained torus budget of the input-dependent β plaintexts.
    pub fn log_beta_budget(&self) -> usize {
        self.log_beta_budget
    }

    /// Mid-pipeline slot-order convention (see [`PaCoSlotOrder`]).
    pub fn slot_order(&self) -> PaCoSlotOrder {
        self.slot_order
    }

    /// Selects the mid-pipeline slot-order convention. Affects factor
    /// generation, the cleartext packing gather, and (through the diagonal
    /// offsets) the Galois key set — keep it stable per key bundle. Purely a
    /// layout relabel: budget, levels, and the recovered coefficients are
    /// identical across conventions.
    pub fn with_slot_order(mut self, slot_order: PaCoSlotOrder) -> Self {
        // The convention changes the diagonal offsets, hence the Galois set.
        self.galois_cache = std::sync::OnceLock::new();
        self.slot_order = slot_order;
        self
    }

    /// Partial CoeffsToSlots factor-chain plan.
    pub fn c2s(&self) -> &PaCoDFTPlan {
        &self.c2s
    }

    /// SlotsToCoeffs′ factor-chain plan.
    pub fn stc(&self) -> &PaCoDFTPlan {
        &self.stc
    }

    /// Ring degree `N`.
    pub fn n(&self) -> usize {
        1 << self.log_n
    }

    /// `B = N/(4h)`: the range of the secret shift indices `u_v`.
    pub fn b(&self) -> usize {
        self.n() / (4 * self.h)
    }

    /// `k = B/C = N/(4hC)`: chunks per packed vector.
    pub fn k(&self) -> usize {
        self.b() / self.c
    }

    /// `n = 2hC`: the number of slots actively used per chunk.
    pub fn slots(&self) -> usize {
        2 * self.h * self.c
    }

    /// `N/2`: the full slot count of the ring.
    pub fn half_n(&self) -> usize {
        self.n() / 2
    }

    /// `log2(C)`.
    pub fn log_c(&self) -> usize {
        self.c.trailing_zeros() as usize
    }

    /// The input modulus `q = 2^log_q`.
    pub fn q(&self) -> u64 {
        1u64 << self.log_q
    }

    /// Bitmask reducing a `u64` modulo `q`.
    pub fn q_mask(&self) -> u64 {
        self.q() - 1
    }

    /// Unit count of the c2s chain: `log 2C` butterfly layers + the ψ unit.
    pub fn c2s_units(&self) -> usize {
        self.log_c() + 2
    }

    /// Unit count of the stc chain: the pack unit + `log C − 1` butterfly
    /// layers.
    pub fn stc_units(&self) -> usize {
        self.log_c()
    }

    /// Revalidates the dimensions, chain shapes, checked budget arithmetic,
    /// and dyadic user scaling guaranteed by the constructors.
    pub fn check(&self) -> Result<()> {
        let (consumed_bits, extra_scale_log2) = self.validate_and_derive()?;
        ensure!(
            self.consumed_bits == consumed_bits && self.extra_scale_log2 == extra_scale_log2,
            "PaCo plan derived accounting is inconsistent with its evaluation parameters",
        );
        Ok(())
    }

    /// Validates the complete homomorphic-evaluation parameterization.
    ///
    /// [`Self::new`] intentionally creates a dimension-only plan for
    /// cleartext packing and structured-secret construction. Contexts and key
    /// bundles must use a plan returned by [`Self::with_evaluation`]; this
    /// additional check rejects the neutral zero-scale schedule before any
    /// key material is accepted or any factor compilation starts.
    pub(crate) fn check_evaluation(&self) -> Result<()> {
        self.check()?;
        ensure!(
            self.log_delta_bsk > self.log_q as usize,
            "PaCo bootstrapping-key scale {} must exceed input modulus exponent {}",
            self.log_delta_bsk,
            self.log_q,
        );
        ensure!(self.log_beta_budget > 0, "PaCo beta plaintext budget must be non-zero");
        for (name, dft) in [("CoeffsToSlots", &self.c2s), ("SlotsToCoeffs", &self.stc)] {
            ensure!(dft.log_delta > 0, "PaCo {name} factor scale must be non-zero");
            ensure!(dft.log_budget > 0, "PaCo {name} factor budget must be non-zero");
        }
        Ok(())
    }

    fn validate_and_derive(&self) -> Result<(usize, i64)> {
        Self::check_dimensions(self.log_n, self.h, self.c, self.log_q)?;
        self.c2s.check("c2s", self.c2s_units())?;
        self.stc.check("stc", self.stc_units())?;
        ensure!(
            u32::try_from(self.log_delta_bsk).is_ok(),
            "PaCo bootstrapping-key scale {} exceeds the layout type's u32 range",
            self.log_delta_bsk,
        );
        let beta_width = self
            .log_delta_bsk
            .checked_add(self.log_beta_budget)
            .context("PaCo β plaintext width overflows usize")?;
        ensure!(
            u32::try_from(beta_width).is_ok(),
            "PaCo β plaintext width {beta_width} exceeds the layout type's u32 range",
        );
        Ok((self.checked_consumed_bits()?, self.checked_extra_scale_log2()?))
    }

    fn refresh_derived(&mut self) -> Result<()> {
        let (consumed_bits, extra_scale_log2) = self.validate_and_derive()?;
        self.consumed_bits = consumed_bits;
        self.extra_scale_log2 = extra_scale_log2;
        Ok(())
    }

    fn checked_extra_scale_log2(&self) -> Result<i64> {
        let log2 = (self.h as f64).mul_add(self.c2s.scaling.log2(), self.stc.scaling.log2());
        ensure!(log2.is_finite(), "the total user scaling exponent is not finite");
        let rounded = log2.round();
        let tolerance = 32.0 * f64::EPSILON * log2.abs().max(1.0);
        ensure!(
            (log2 - rounded).abs() <= tolerance,
            "the total user scaling s_c2s^h*s_stc = 2^{log2} must be a power of two \
             to floating-point precision (error {}, tolerance {tolerance}); the relabel compensates it as pure metadata",
            (log2 - rounded).abs(),
        );
        // `i64::MAX as f64` rounds up to 2^63, so the positive bound must be
        // exclusive. The negative endpoint -2^63 is exactly representable.
        const I64_UPPER_EXCLUSIVE: f64 = 9_223_372_036_854_775_808.0;
        ensure!(
            rounded >= i64::MIN as f64 && rounded < I64_UPPER_EXCLUSIVE,
            "the total user scaling exponent {rounded} does not fit i64",
        );
        Ok(rounded as i64)
    }

    /// `log2` of the constant the user scalings contribute to the pipeline
    /// output: `s_c2s^h · s_stc`. The c2s scaling is raised to the `h`-th
    /// power because `Pr_{n→2C}` multiplies `h` block positions of the same
    /// vector — constants can thus be folded into the phase domain at
    /// `h`-th-root cost. The final scale relabel compensates by exactly this
    /// amount, so the total must be a power of two (checked by
    /// [`Self::check`]) and the decoded output is scaling-invariant.
    pub fn extra_scale_log2(&self) -> i64 {
        self.extra_scale_log2
    }

    /// The generated SlotToCoeff′ factors, in application order.
    pub(crate) fn stc_factors(&self) -> Vec<ComplexDiagonals<f64>> {
        paco_stc_factors(self)
    }

    /// Number of levels: blind rotation (1) + one per c2s factor (the last
    /// being the fused `A·w + B·conj(w)` pair) + product (`log h`) + one per
    /// stc factor.
    pub fn levels(&self) -> usize {
        1 + self.c2s.num_factors() + self.h.trailing_zeros() as usize + self.stc.num_factors()
    }

    /// Total budget bits the bootstrap consumes, per the crate's metadata
    /// rules (pt×ct consumes the plaintext's scale; ct×ct consumes the
    /// accumulator's scale, inherited from the bsk):
    /// `log_delta_bsk·(1 + log h) + Σ_c2s log_delta_c2s + Σ_stc log_delta_stc`.
    pub fn consumed_bits(&self) -> usize {
        self.consumed_bits
    }

    /// Widest plaintext torus width used by β, CoeffsToSlots, or
    /// SlotsToCoeffs. Constructor validation guarantees the additions fit.
    pub fn max_plaintext_width(&self) -> usize {
        (self.log_delta_bsk + self.log_beta_budget)
            .max(self.c2s.log_delta + self.c2s.log_budget)
            .max(self.stc.log_delta + self.stc.log_budget)
    }

    fn checked_consumed_bits(&self) -> Result<usize> {
        let product_levels = 1usize
            .checked_add(self.h.trailing_zeros() as usize)
            .context("PaCo product level count overflows usize")?;
        let bsk = self
            .log_delta_bsk
            .checked_mul(product_levels)
            .context("PaCo bootstrapping-key budget consumption overflows usize")?;
        let c2s = self
            .c2s
            .num_factors()
            .checked_mul(self.c2s.log_delta)
            .context("PaCo c2s budget consumption overflows usize")?;
        let stc = self
            .stc
            .num_factors()
            .checked_mul(self.stc.log_delta)
            .context("PaCo stc budget consumption overflows usize")?;
        let bits = bsk
            .checked_add(c2s)
            .and_then(|bits| bits.checked_add(stc))
            .context("total PaCo budget consumption overflows usize")?;
        ensure!(
            u32::try_from(bits).is_ok(),
            "total PaCo budget consumption {bits} exceeds the layout type's u32 range",
        );
        Ok(bits)
    }

    /// Canonical bootstrap torus width: the consumed bits, the accumulator
    /// scale, and `headroom` spare bits, rounded up to a limb multiple.
    ///
    /// `headroom` becomes the **final** budget of the output ciphertext
    /// (before the relabel credit) and must cover the recovered message
    /// magnitude at the output scale
    /// `log_delta_bsk − (log q − 2 − Δ_in) − extra_scale_log2()` (see
    /// [`CKKSPaCoOps::ckks_paco_bootstrap_direct_into`](crate::api::CKKSPaCoOps::ckks_paco_bootstrap_direct_into)).
    pub fn k_boot(&self, base2k: usize, headroom: usize) -> Result<usize> {
        self.check_evaluation()
            .context("cannot size a bootstrap from a dimension-only PaCo plan")?;
        ensure!((1..=63).contains(&base2k), "base2k must be in [1, 63], got {base2k}");
        let bits = self
            .checked_consumed_bits()?
            .checked_add(self.log_delta_bsk)
            .and_then(|bits| bits.checked_add(headroom))
            .context("PaCo bootstrap width overflows usize")?;
        let remainder = bits % base2k;
        let rounded = if remainder == 0 {
            bits
        } else {
            bits.checked_add(base2k - remainder)
                .context("rounded PaCo bootstrap width overflows usize")?
        };
        ensure!(
            u32::try_from(rounded).is_ok(),
            "rounded PaCo bootstrap width {rounded} exceeds the layout type's u32 range",
        );
        Ok(rounded)
    }

    /// The slot rotations of the trace/product folds of Algorithm 4, in
    /// pipeline order: `Tr_{N/2→n}` and `Pr_{n→2C}` (the paper's tail traces
    /// are folded into the SlotToCoeff factors, whose rotations
    /// [`Self::galois_elements`] collects from the factor indexes).
    pub fn fold_rotations(&self) -> Vec<i64> {
        let mut rots = fold_rotations(self.half_n(), self.slots());
        rots.extend(fold_rotations(self.slots(), 2 * self.c));
        rots
    }

    /// The total StC′-chain diagonal count under each slot-order convention,
    /// as `(natural, bit_rev_low)` — the per-bootstrap plaintext-multiply
    /// cost of the folded permutation, for callers choosing a convention.
    ///
    /// `Natural` folds the input-side bit-reversal into the first StC′
    /// butterfly factor (up to the dense ceiling `C/2`); `BitRevLow` uses the
    /// raw factors. The two counts coincide when the chain is fully merged or
    /// `C ≤ 4` (where the permutation is the identity). The C2S side is
    /// second-order: conjugation preserves its factor counts except sparse
    /// merges spanning the `log(C/2)` boundary (measured +2 diagonals, see
    /// the `conjugate_by_low_bitrev_counts` pins); schedules can avoid such
    /// merges by splitting groups at the boundary. Note the convention also
    /// changes the Galois key *values* (and possibly count) — compare
    /// [`Self::galois_elements`] on both variants before persisting keys.
    pub fn stc_fold_cost(&self) -> (usize, usize) {
        let count = |slot_order: PaCoSlotOrder| -> usize {
            self.clone()
                .with_slot_order(slot_order)
                .stc_factors()
                .iter()
                .map(|f| f.indexes().len())
                .sum()
        };
        (count(PaCoSlotOrder::Natural), count(PaCoSlotOrder::BitRevLow))
    }

    /// Every Galois element the bootstrap needs an automorphism key for: the
    /// fold rotations, plain conjugation (`−1`), the ψ tail (the BSGS
    /// elements of the conjugation-augmented pair, or the fused conj-rotate
    /// element when ψ stands alone), and the BSGS elements of both factor
    /// chains at their per-factor giant steps.
    /// Memoized: deriving the set requires generating the full factor chains,
    /// and key-bundle construction/preparation each consult it — the immutable
    /// plan computes it once and serves clones afterwards (the builder methods
    /// reset the memo when they change anything Galois-relevant).
    pub fn galois_elements(&self) -> Vec<i64> {
        self.galois_cache
            .get_or_init(|| {
                let cyclotomic_order = 2 * self.n() as i64;
                let (coeffs_to_slots, psi_tail) = paco_psi_c2s_factors::<f64>(self);
                let slots_to_coeffs = self.stc_factors();
                self.galois_elements_from_factors(&coeffs_to_slots, &psi_tail, &slots_to_coeffs, cyclotomic_order)
            })
            .clone()
    }

    /// Enumerates keys from factor matrices that were already generated for
    /// context compilation, avoiding a second factorization solely to recover
    /// diagonal indexes.
    pub(crate) fn galois_elements_from_factors<T>(
        &self,
        coeffs_to_slots: &[ComplexDiagonals<T>],
        psi_tail: &PaCoPsiTail<T>,
        slots_to_coeffs: &[ComplexDiagonals<T>],
        cyclotomic_order: i64,
    ) -> Vec<i64> {
        let mut set: BTreeSet<i64> = galois_elements_from_rotations(self.fold_rotations(), cyclotomic_order)
            .into_iter()
            .collect();
        set.insert(-1);
        if matches!(psi_tail, PaCoPsiTail::Mask(_)) {
            set.insert(conj_rotate_galois_element(self.c as i64, cyclotomic_order));
        }
        let mut extend = |factor: &ComplexDiagonals<T>, giant_step: usize| {
            let layout = LinearTransformationLayout {
                indexes: factor.indexes(),
                slots: factor.slots(),
                strategy: LinearTransformationStrategy::Bsgs { giant_step },
            };
            set.extend(layout.galois_elements(cyclotomic_order));
        };
        for (factor, step) in coeffs_to_slots.iter().zip(&self.c2s.schedule.steps) {
            extend(factor, step.giant_step);
        }
        if let PaCoPsiTail::Pair(pair) = psi_tail {
            let gs_last = self.c2s.schedule.steps.last().expect("checked non-empty").giant_step;
            for factor in pair {
                extend(factor, gs_last);
            }
        }
        for (factor, step) in slots_to_coeffs.iter().zip(&self.stc.schedule.steps) {
            extend(factor, step.giant_step);
        }
        set.remove(&galois_element(0, cyclotomic_order));
        set.into_iter().collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Measured pin of the automorphism-key delta between conventions: the
    /// BSGS rotation indices follow the factor diagonal offsets, so the sets
    /// differ in *values* while staying comparable in size — and at the
    /// large-C payoff configuration (`C = 32, g1 = 1`) `BitRevLow` needs
    /// strictly fewer keys (the folded first StC′ factor's dense baby-step
    /// spread disappears).
    #[test]
    fn galois_element_counts_by_convention() {
        let with = |so: PaCoSlotOrder| {
            PaCoPlan::new(10, 2, 32, 29)
                .unwrap()
                .with_slot_order(so)
                .with_evaluation(
                    0,
                    0,
                    PaCoDFTPlan::new(vec![2, 2, 3], vec![2, 2, 2], 0, 0, 1.0).unwrap(),
                    PaCoDFTPlan::new(vec![1; 5], vec![2; 5], 0, 0, 1.0).unwrap(),
                )
                .unwrap()
                .galois_elements()
                .len()
        };
        let (nat, brl) = (with(PaCoSlotOrder::Natural), with(PaCoSlotOrder::BitRevLow));
        assert_eq!((nat, brl), (22, 20), "measured automorphism-key counts");
    }

    /// `stc_fold_cost` matches the per-factor pins of the lt.rs suite: at the
    /// `C = 32, g1 = 1` payoff configuration the fold costs 26 vs 15 total
    /// StC′ diagonals; a fully merged chain pays nothing either way.
    #[test]
    fn stc_fold_cost_pins() {
        let with_stc = |depth: Vec<usize>| {
            let gs = vec![2; depth.len()];
            PaCoPlan::new(10, 2, 32, 29)
                .unwrap()
                .with_evaluation(
                    0,
                    0,
                    PaCoDFTPlan::new(vec![2, 2, 3], vec![2, 2, 2], 0, 0, 1.0).unwrap(),
                    PaCoDFTPlan::new(depth, gs, 0, 0, 1.0).unwrap(),
                )
                .unwrap()
        };
        assert_eq!(with_stc(vec![1; 5]).stc_fold_cost(), (26, 15));
        assert_eq!(with_stc(vec![5]).stc_fold_cost().0, with_stc(vec![5]).stc_fold_cost().1);
    }

    /// The slot-order convention defaults to `Natural`, round-trips through
    /// the builder-style setter, and survives `with_evaluation`.
    #[test]
    fn slot_order_round_trip() {
        let p = PaCoPlan::new(9, 8, 4, 29).expect("validated test parameters");
        assert_eq!(p.slot_order(), PaCoSlotOrder::Natural);
        let p = p.with_slot_order(PaCoSlotOrder::BitRevLow);
        assert_eq!(p.slot_order(), PaCoSlotOrder::BitRevLow);
        let p = p
            .clone()
            .with_evaluation(p.log_delta_bsk(), p.log_beta_budget(), p.c2s().clone(), p.stc().clone())
            .expect("re-attaching identical evaluation parameters");
        assert_eq!(
            p.slot_order(),
            PaCoSlotOrder::BitRevLow,
            "with_evaluation must preserve the convention"
        );
    }

    #[test]
    fn derived_quantities() {
        // N = 512, h = 8 -> B = 16; C = 4 -> k = 4, n = 64 = N/(2k).
        let p = PaCoPlan::new(9, 8, 4, 29).expect("validated test parameters");
        assert_eq!(p.n(), 512);
        assert_eq!(p.b(), 16);
        assert_eq!(p.k(), 4);
        assert_eq!(p.slots(), 64);
        assert_eq!(p.slots() * p.k(), p.half_n(), "n*k covers all N/2 slots");
        assert_eq!(p.log_c(), 2);
    }

    #[test]
    fn rejects_oversized_hc() {
        let err = PaCoPlan::new(9, 8, 32, 29).expect_err("oversized h*C must be rejected");
        assert!(err.to_string().contains("4*h*c"));
    }

    fn plan_with_scaling(c2s_scaling: f64, stc_scaling: f64) -> Result<PaCoPlan> {
        // h = 4, C = 8: c2s units = 5 (4 layers + ψ), stc units = 3 (pack + 2).
        PaCoPlan::new(8, 4, 8, 28)?.with_evaluation(
            30,
            16,
            PaCoDFTPlan::new(vec![2, 3], vec![2, 2], 30, 16, c2s_scaling)?,
            PaCoDFTPlan::new(vec![2, 1], vec![2, 2], 24, 16, stc_scaling)?,
        )
    }

    fn plan() -> PaCoPlan {
        plan_with_scaling(1.0, 1.0).unwrap()
    }

    #[test]
    fn dimension_only_plan_is_not_evaluation_ready() {
        let p = PaCoPlan::new(8, 4, 8, 28).unwrap();
        p.check().unwrap();
        let error = p
            .check_evaluation()
            .expect_err("a dimension-only PaCo plan must not be accepted for evaluation");
        assert!(error.to_string().contains("scale"), "unexpected error: {error:#}");
        assert!(
            p.k_boot(19, 10).is_err(),
            "dimension-only plans cannot size bootstrap storage"
        );
    }

    #[test]
    fn level_schedule_matches_paper() {
        // 1 (blind rot) + 2 (c2s, last level the fused ψ/μ pair) + 2 (log h)
        // + 2 (stc, first factor carrying η + pair-packing) = 7 — two levels
        // below the paper's schedule (no μ level, no η level).
        let p = plan();
        p.check().unwrap();
        assert_eq!(p.levels(), 7);
        // 30·(1+2) + 2·30 + 2·24 = 90 + 60 + 48.
        assert_eq!(p.consumed_bits(), 198);
        assert_eq!(p.k_boot(19, 10).unwrap(), (198usize + 30 + 10).next_multiple_of(19));
    }

    #[test]
    fn uniform_schedule_is_balanced() {
        // 5 units at g = 4: ⌈5/4⌉ = 2 groups of sizes {3, 2} — never [4, 1].
        let d = PaCoDFTPlan::uniform(5, 4, 2, 30, 16).unwrap();
        assert_eq!(d.factorization_depth(), [3, 2]);
        assert_eq!(PaCoDFTPlan::uniform(4, 2, 2, 30, 16).unwrap().factorization_depth(), [2, 2],);
        assert_eq!(PaCoDFTPlan::uniform(3, 100, 2, 30, 16).unwrap().factorization_depth(), [3],);
        assert_eq!(PaCoDFTPlan::uniform(1, 1, 2, 30, 16).unwrap().factorization_depth(), [1],);
    }

    #[test]
    fn scaling_accounting() {
        // s_c2s = 2^{1/h} with h = 4 and s_stc = 2^{-5}: total = 2^{1-5}.
        let p = plan_with_scaling(2f64.powf(0.25), 2f64.powi(-5)).unwrap();
        p.check().unwrap();
        assert_eq!(p.extra_scale_log2(), -4);
    }

    #[test]
    fn non_dyadic_scaling_rejected() {
        let err = plan_with_scaling(1.0, 3.0).unwrap_err();
        assert!(err.to_string().contains("power of two"));
    }

    #[test]
    fn invalid_chain_parameters_are_rejected() {
        assert!(PaCoDFTPlan::new(Vec::new(), Vec::new(), 30, 16, 1.0).is_err());
        assert!(PaCoDFTPlan::new(vec![1], vec![0], 30, 16, 1.0).is_err());
        assert!(PaCoDFTPlan::new(vec![1], vec![1], usize::MAX, 1, 1.0).is_err());
        assert!(PaCoDFTPlan::new(vec![1], vec![1], 30, 16, 0.0).is_err());
        assert!(PaCoDFTPlan::new(vec![1], vec![1], 30, 16, f64::INFINITY).is_err());
        assert!(PaCoDFTPlan::new(vec![usize::MAX, 1], vec![1, 1], 30, 16, 1.0).is_err());
    }

    #[test]
    fn invalid_dimensions_and_widths_are_rejected() {
        assert!(PaCoPlan::new(usize::BITS as usize, 1, 2, 28).is_err());
        assert!(PaCoPlan::new(8, 3, 2, 28).is_err());
        assert!(PaCoPlan::new(8, 4, 3, 28).is_err());
        assert!(PaCoPlan::new(8, 4, 32, 28).is_err());
        assert!(PaCoPlan::new(8, 4, 8, 0).is_err());
        assert!(plan().k_boot(0, 10).is_err());
        assert!(plan().k_boot(64, 10).is_err());
        assert!(PaCoPlan::new(8, 4, 8, 63).is_ok());
        assert!(
            PaCoPlan::new(8, 4, 8, 28)
                .unwrap()
                .with_evaluation(30, usize::MAX, plan().c2s().clone(), plan().stc().clone())
                .is_err()
        );
    }

    #[test]
    fn galois_elements_cover_conjugations() {
        let p = plan();
        let els = p.galois_elements();
        assert!(els.contains(&-1), "plain conjugation (fused ψ/μ pair + imag extraction)");
        // Every element distinct and the identity absent.
        assert!(!els.contains(&1));
    }

    #[test]
    fn galois_elements_cover_the_mask_tail() {
        // ψ alone (last entry 1) emits the mask tail: the fused conj-rotate
        // element must be enumerated instead of the pair's BSGS elements.
        let p = PaCoPlan::new(8, 4, 8, 28)
            .unwrap()
            .with_evaluation(
                30,
                16,
                PaCoDFTPlan::new(vec![4, 1], vec![2, 2], 30, 16, 1.0).unwrap(),
                PaCoDFTPlan::new(vec![2, 1], vec![2, 2], 24, 16, 1.0).unwrap(),
            )
            .unwrap();
        let els = p.galois_elements();
        let order = 2 * p.n() as i64;
        assert!(
            els.contains(&conj_rotate_galois_element(p.c() as i64, order)),
            "fused conj-rotate element for the ψ mask tail"
        );
        assert!(els.contains(&-1), "plain conjugation (imag extraction)");
    }
}
