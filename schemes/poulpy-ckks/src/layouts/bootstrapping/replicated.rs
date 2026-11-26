//! The replicated route of a bootstrap: a [replicated slot layout](crate::SlotLayout::Replicated) of `k` values,
//! `R | k`, refreshed S2C-first, the small-batch scheme of [ePrint 2026/2513](https://eprint.iacr.org/2026/2513) (see
//! `docs/guides/replicated.md`).

use std::collections::BTreeMap;

use anyhow::{Context, Result, ensure};
use poulpy_core::{
    GLWEAutomorphismKeyEncryptSk, GLWECIKeyEncryptSk, GLWESecretSampling, GLWESwitchingKeyEncryptSk, GLWETensorKeyEncryptSk,
    GLWETrace,
    layouts::{
        BackendGLWESecret, Base2K, Degree, LWEInfos, LinearTransformation, ModuleCoreAlloc, galois_at, galois_of_rotation,
    },
};
use poulpy_hal::{
    api::{VecZnxAutomorphism, VecZnxSwitchRing},
    layouts::{Backend, ConjugateInvariant, CyclotomicOrder, HostDataMut, Module, ScratchArena, Standard, galois_element},
    source::Source,
};

use crate::{
    CKKSModuleInfos, CoeffsMeta, ReplicatedLayout,
    api::{CKKSEncodingHostOps, CKKSEncodingScalar},
    layouts::{
        BootstrappingKeySet, BootstrappingKeyWidths, BootstrappingKeysLayout, CKKSDFTFactors, CKKSPlaintextOwned,
        ComplexDiagonals, DFTMatrix, DFTMatrixFactors, DFTPlan, DFTType, Decode, Encode, EvalMod, EvalModPlan, EvalModRing,
        MaskedCoeffs, Split,
        bootstrapping_keys::KeyRecipe,
        dft::DFTOutputFormat,
        dft_matrices::ckks_dft_matrices_gen_blockwise,
        eval_mod::{EvalModType, ckks_eval_mod_replicated_compile, ckks_eval_mod_replicated_packed_compile},
        replicated::{
            ReplicatedEncodeModule, ReplicatedFactor, ReplicatedTransform, check_period_at, gather_factor, period_factor,
            placement_compress, placement_spread, set_complex, split_depth,
        },
    },
    polynomial::SplitStrategy,
};

use super::{BootstrappingContextPrepared, BootstrappingPlan};

/// How the replicated route evaluates `SlotsToCoeffs` and `CoeffsToSlots`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ReplicatedTransforms {
    /// Each map is one transform of `k` diagonals: `2·sqrt(k)` key switches
    /// and `k` diagonals over the `kR` slots of the period, one level each.
    Dense,
    /// Each map is the `k`-point DFT of the subring of degree `2k` in
    /// `s2c_factors` and `c2s_factors` factors of the DFT generator, and the
    /// move between the data slots and consecutive slots of `m` values in
    /// `placement_factors` factors of about `m^(1/placement_factors)` 0/1 mask
    /// diagonals. Every `SlotsToCoeffs` factor consumes
    /// `s2c_log_delta_diag` bits. `CoeffsToSlots` applies the DFT on the slots
    /// of period `k`, then the move onto the data slots, whose last factor
    /// alone is quantized with shaping and consumes `c2s_log_delta_diag` bits;
    /// the others consume `c2s_inner_log_delta_diag` bits, enough for their
    /// unshaped quantization error to stay under the output precision. The DFT
    /// factors leave the subring's coefficients in bit-reversed order, which
    /// the two maps undo together.
    Factored {
        s2c_factors: usize,
        c2s_factors: usize,
        placement_factors: usize,
        c2s_inner_log_delta_diag: usize,
    },
}

/// The replicated route of a [`BootstrappingPlan`]: the refresh of a replicated layout of `k` values, `R | k`,
/// S2C-first, set by [`BootstrappingPlan::replicated`].
///
/// 1. SlotsToCoeffs brings an input at `Δ_b` or at the leveled scale to `Δ_b` at the input width, averages the
///    `N/(2kR)` copies of the layout's period by a normalized trace, gathers the data slots onto the coefficients of
///    the subring of degree `2k`, `extra_bits` under the scale, and rounds plainly to the message ratio of EvalMod.
/// 2. The raise, with sparse-secret encapsulation when the plan has it, lifts the message by `mod_up_extra_bits`
///    plus `extra_bits`, the working bits of CoeffsToSlots.
/// 3. CoeffsToSlots removes the lifted multiples off the subring by the normalized trace onto period `k`, and writes
///    the coefficients onto the data slots, packed on two batches when `packed`.
/// 4. EvalMod splits the real and imaginary parts, each rounded with shaping, evaluates them, side by side when
///    packed, recombines them, and rounds with shaping to the leveled scale.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ReplicatedRoute {
    /// `log2` of the degree of the ciphertexts the route bootstraps, at most the module's.
    pub log_n: usize,
    pub layout: ReplicatedLayout,
    /// Scale of the bootstrapped message, `Δ_b`.
    pub log_delta: usize,
    /// Scale of the leveled ciphertexts: of the output, and of an input whose levels are spent.
    pub level_log_delta: usize,
    /// Scale of the `SlotsToCoeffs` diagonals, the bits each factor consumes.
    pub s2c_log_delta_diag: usize,
    /// Scale of the last `CoeffsToSlots` factor, quantized with shaping.
    pub c2s_log_delta_diag: usize,
    /// Bits the raise lifts the message by, removed by the last rounding.
    pub mod_up_extra_bits: usize,
    /// Working bits of the linear maps, under the scale.
    pub extra_bits: usize,
    /// Evaluates EvalMod once on the real and imaginary parts side by side. The two batches need `4kR` slots to fit
    /// twice the slots of the ring, which the compilation checks.
    pub packed: bool,
    /// The evaluation of `SlotsToCoeffs` and `CoeffsToSlots`.
    pub transforms: ReplicatedTransforms,
}

impl ReplicatedRoute {
    /// The parameters of ePrint 2026/2513 for `layout` at degree `2^log_n`: inputs at `2^30` with a ratio of `2^13`
    /// ([`Self::paper_eval_mod`]), outputs at `2^20`, and the packed EvalMod.
    pub fn paper(layout: ReplicatedLayout, log_n: usize) -> Self {
        Self {
            log_n,
            layout,
            log_delta: 30,
            level_log_delta: 20,
            s2c_log_delta_diag: 26,
            c2s_log_delta_diag: 34,
            mod_up_extra_bits: 6,
            extra_bits: 8,
            packed: true,
            transforms: ReplicatedTransforms::Dense,
        }
    }

    /// The EvalMod of [`Self::paper`] for `layout`: the even cosine of degree 30 on `K = 16` with 3 double angles at
    /// the scale of [`Self::eval_mod_log_delta`], constants at `2^26` and a message ratio of `2^13`.
    pub fn paper_eval_mod(layout: &ReplicatedLayout) -> EvalModPlan {
        EvalModPlan {
            eval_mod_type: EvalModType::CosHKEven,
            log_msg_ratio: 13,
            f_mod_degree: 30,
            f_mod_interval: 16,
            f_mod_log_interval_reduction: 3,
            f_mod_inv_degree: None,
            scaling: None,
            split_strategy: SplitStrategy::MinDepth,
            coeffs_meta: CoeffsMeta::from_delta_budget(26, 4),
            f_mod_log_delta: Self::eval_mod_log_delta(layout),
        }
    }

    /// EvalMod scale of [`Self::paper_eval_mod`] for `layout`: `2^28`, or `ceil(guard_bits) + 15` bits for a unit
    /// whose guards grow by more than 13 bits (`U4_15`, `U8_40`). Entering EvalMod rounds with shaping at this scale,
    /// which leaves noise of about `2^(guard_bits + log2(sqrt(N)))` units on the guard slots; the polynomial and the
    /// double angles square the guards, which must stay well below one.
    pub fn eval_mod_log_delta(layout: &ReplicatedLayout) -> usize {
        let guard_bits = crate::shaping::ShapingProfile::<Standard>::new(layout).guard_bits();
        28.max(guard_bits.ceil() as usize + 15)
    }

    /// The bits of budget `SlotsToCoeffs` consumes: its diagonal bits per factor.
    pub fn s2c_consumed_bits(&self) -> usize {
        match self.transforms {
            ReplicatedTransforms::Dense => self.s2c_log_delta_diag,
            ReplicatedTransforms::Factored {
                s2c_factors,
                placement_factors,
                ..
            } => (placement_factors + s2c_factors) * self.s2c_log_delta_diag,
        }
    }

    /// The bits of budget `CoeffsToSlots` consumes: the shaped last factor's diagonal bits and, factored, the inner
    /// factors'.
    pub fn c2s_consumed_bits(&self) -> usize {
        match self.transforms {
            ReplicatedTransforms::Dense => self.c2s_log_delta_diag,
            ReplicatedTransforms::Factored {
                c2s_factors,
                placement_factors,
                c2s_inner_log_delta_diag,
                ..
            } => (c2s_factors + placement_factors - 1) * c2s_inner_log_delta_diag + self.c2s_log_delta_diag,
        }
    }

    /// The bits the raise lifts the message by: `mod_up_extra_bits` and the working bits of `CoeffsToSlots`.
    pub fn raise_bits(&self) -> usize {
        self.mod_up_extra_bits + self.extra_bits
    }

    /// The subring DFTs of the two maps, which the DFT generator turns into their factors: `SlotsToCoeffs` decodes
    /// and `CoeffsToSlots` encodes `k` points, the latter scaled by `1/(2K)` for an EvalMod interval `K`.
    pub(crate) fn dft_plans(&self, eval_mod: &EvalModPlan) -> Result<(DFTPlan, DFTPlan)> {
        let log_k = self.layout.log_k();
        let (s2c_factors, c2s_factors, c2s_inner) = match self.transforms {
            ReplicatedTransforms::Dense => (1, 1, self.c2s_log_delta_diag),
            ReplicatedTransforms::Factored {
                s2c_factors,
                c2s_factors,
                placement_factors,
                c2s_inner_log_delta_diag,
            } => {
                ensure!(
                    [s2c_factors, c2s_factors, placement_factors]
                        .iter()
                        .all(|f| (1..=log_k).contains(f)),
                    "factored transforms take between one and log2(k) = {log_k} factors for each DFT and placement"
                );
                (s2c_factors, c2s_factors, c2s_inner_log_delta_diag)
            }
        };
        let schedule = |factors: usize| {
            let depths = split_depth(log_k, factors);
            let giant_steps = vec![1; depths.len()];
            crate::layouts::dft::FactorSchedule::from_parts(&depths, &giant_steps, "BootstrappingPlan")
        };
        let slots_to_coeffs = DFTPlan::new(
            DFTType::Decode,
            schedule(s2c_factors)?,
            DFTOutputFormat::SplitRealAndImag,
            CoeffsMeta::from_delta_budget(self.s2c_log_delta_diag, 0),
        )?;
        let coeffs_to_slots = DFTPlan::new(
            DFTType::Encode,
            schedule(c2s_factors)?,
            DFTOutputFormat::SplitRealAndImag,
            CoeffsMeta::from_delta_budget(c2s_inner, 0),
        )?
        .with_scaling(eval_mod.input_scaling()? / 2.0)?;
        Ok((coeffs_to_slots, slots_to_coeffs))
    }
}

impl BootstrappingPlan {
    /// The replicated route of this plan, `None` on the full layout.
    pub fn replicated_route(&self) -> Option<&ReplicatedRoute> {
        self.replicated.as_ref()
    }

    /// Width of the input of a replicated plan at `Δ_b`: the message ratio and the bits `SlotsToCoeffs` consumes
    /// above it.
    pub fn replicated_input_k(&self) -> Option<usize> {
        Some(self.replicated_raise_input_k()? + self.replicated.as_ref()?.s2c_consumed_bits())
    }

    /// Width of the input of the raise of a replicated plan at `Δ_b`: the message ratio.
    pub(crate) fn replicated_raise_input_k(&self) -> Option<usize> {
        Some(self.replicated.as_ref()?.log_delta + self.eval_mod.log_msg_ratio)
    }

    /// Width of the widest product EvalMod relinearizes on the replicated route for an output of width `k_out`: EvalMod
    /// reads the parts of what `CoeffsToSlots` leaves, rounded off its working bits at the raised scale, and relabels
    /// them at its own scale.
    pub fn replicated_tensor_key_k(&self, k_out: usize) -> Option<usize> {
        let route = self.replicated.as_ref()?;
        let after_c2s = self.replicated_raised_k(k_out)?.checked_sub(route.c2s_consumed_bits())?;
        let raised_log_delta = route.log_delta + self.eval_mod.log_msg_ratio + route.mod_up_extra_bits;
        Some(
            self.eval_mod
                .relinearization_k(after_c2s.checked_sub(route.extra_bits + raised_log_delta)?),
        )
    }

    /// Width the raise of a replicated plan lands at for an output of width `k_out`: `CoeffsToSlots`, EvalMod, the
    /// rescale from `Δ_b` lifted by `mod_up_extra_bits` to the leveled scale, and the working bits of `CoeffsToSlots`.
    /// Every key the top trace, `CoeffsToSlots` and the conjugation switch covers it.
    pub fn replicated_raised_k(&self, k_out: usize) -> Option<usize> {
        let route = self.replicated.as_ref()?;
        Some(
            k_out + route.c2s_consumed_bits() + self.eval_mod.consumed_bits() + route.log_delta + route.raise_bits()
                - route.level_log_delta,
        )
    }
}

/// The step a [`ReplicatedStage`] serves.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum ReplicatedStep {
    SlotsToCoeffs,
    CoeffsToSlots,
}

/// What a step of a replicated run reads of the route: the trace, the working bits and the layouts around the factor
/// chain of the transform the run holds.
#[derive(Clone, Copy, Debug)]
pub struct ReplicatedStage<'a> {
    pub(crate) step: ReplicatedStep,
    pub(crate) route: &'a ReplicatedRoute,
    /// The message ratio of EvalMod, the budget `SlotsToCoeffs` leaves.
    pub(crate) log_msg_ratio: usize,
    /// The first level of the trace of the step.
    pub(crate) trace_skip: usize,
    /// The sparsity the trace of `CoeffsToSlots` leaves.
    pub(crate) top_sparsity: usize,
    /// The layout of the output of `CoeffsToSlots`: the route's, or two batches side by side when packed.
    pub(crate) packed: Option<ReplicatedLayout>,
    /// The bits each factor of the step's transform consumes, in evaluation order.
    pub(crate) consumed: &'a [usize],
}

impl ReplicatedStage<'_> {
    /// The input width of the run at `Δ_b`, as [`BootstrappingPlan::replicated_input_k`].
    pub(crate) fn input_k(&self) -> usize {
        self.route.log_delta + self.log_msg_ratio + self.route.s2c_consumed_bits()
    }

    /// The layout `CoeffsToSlots` writes.
    pub(crate) fn c2s_layout(&self) -> ReplicatedLayout {
        self.packed.unwrap_or(self.route.layout)
    }
}

/// The compiled replicated route of a context: EvalMod with masked constants and the shape of the traces, beside the
/// factor chains of the transforms, which the context holds as its DFT matrices.
pub(crate) struct ReplicatedStages<BE: Backend, F> {
    pub(crate) route: ReplicatedRoute,
    /// The first levels of the traces onto periods `kR` and `k`.
    pub(crate) bottom_skip: usize,
    pub(crate) top_skip: usize,
    /// Sparsity of the subring of period `k`.
    pub(crate) top_sparsity: usize,
    /// The layout of the two batches of a packed EvalMod.
    pub(crate) packed: Option<ReplicatedLayout>,
    /// The bits each factor of `SlotsToCoeffs` and `CoeffsToSlots` consumes, in evaluation order.
    pub(crate) s2c_consumed: Vec<usize>,
    pub(crate) c2s_consumed: Vec<usize>,
    pub(crate) eval_mod: EvalMod<F, MaskedCoeffs<CKKSPlaintextOwned<BE>>>,
}

impl<BE: Backend, F> ReplicatedStages<BE, F> {
    pub(crate) fn s2c_stage(&self, log_msg_ratio: usize) -> ReplicatedStage<'_> {
        ReplicatedStage {
            step: ReplicatedStep::SlotsToCoeffs,
            route: &self.route,
            log_msg_ratio,
            trace_skip: self.bottom_skip,
            top_sparsity: self.top_sparsity,
            packed: self.packed,
            consumed: &self.s2c_consumed,
        }
    }

    pub(crate) fn c2s_stage(&self, log_msg_ratio: usize) -> ReplicatedStage<'_> {
        ReplicatedStage {
            step: ReplicatedStep::CoeffsToSlots,
            route: &self.route,
            log_msg_ratio,
            trace_skip: self.top_skip,
            top_sparsity: self.top_sparsity,
            packed: self.packed,
            consumed: &self.c2s_consumed,
        }
    }
}

/// The layout of the two batches EvalMod runs on side by side: `None` unpacked, and a refusal for a packed route whose
/// two batches do not fit twice the `max_slots` slots of the ring.
fn packed_layout(route: &ReplicatedRoute, max_slots: usize) -> Result<Option<ReplicatedLayout>> {
    if !route.packed {
        return Ok(None);
    }
    let layout = route.layout;
    let (k, r) = (1usize << layout.log_k(), 1usize << layout.log_r());
    ensure!(
        4 * k * r <= 2 * max_slots,
        "a packed route needs its two batches of {k} values of period {r} to fit the {max_slots} slots"
    );
    Ok(Some(ReplicatedLayout::new(
        layout.log_k() as u8 + 1,
        layout.log_r() as u8,
        layout.shaping(),
    )?))
}

/// Scratch of [`compile`]: the host encoding of the EvalMod constants and of the diagonals of either map, each on at
/// most the slots of the module.
pub(crate) fn compile_tmp_bytes<BE, F>(module: &Module<BE>, route: &ReplicatedRoute) -> Result<usize>
where
    BE: Backend,
    F: CKKSEncodingScalar,
    Module<BE>: ReplicatedEncodeModule<BE, F>,
{
    let n = Degree(1 << route.log_n);
    check_period_at::<BE, Standard>(module, n, &route.layout)?;
    packed_layout(route, module.ckks_slots_of_degree::<Standard>(n.as_usize()))?;
    Ok(module.ckks_encoding_host_tmp_bytes(module.ckks_max_slots::<Standard>()))
}

/// The unprepared matrices of the two maps of `route` and its compiled stages, on `module`.
///
/// The maps take their DFT factors from the generator, with the schedules of the plan's subring DFTs: dense, one factor
/// merged with the placement onto the data slots; factored, the placement in its own factors.
#[allow(clippy::type_complexity)]
pub(crate) fn compile<BE, F>(
    module: &Module<BE>,
    base2k: Base2K,
    plan: &BootstrappingPlan,
    route: &ReplicatedRoute,
    scratch: &mut ScratchArena<'_, BE>,
) -> Result<(
    DFTMatrix<BE, Encode, Split>,
    DFTMatrix<BE, Decode, Split>,
    ReplicatedStages<BE, F>,
)>
where
    BE: Backend,
    F: CKKSEncodingScalar,
    Module<BE>: ReplicatedEncodeModule<BE, F>,
{
    let layout = route.layout;
    let n = Degree(1 << route.log_n);
    check_period_at::<BE, Standard>(module, n, &layout)?;
    let (k, r) = (1usize << layout.log_k(), 1usize << layout.log_r());
    ensure!(r <= k, "bootstrapping needs the period {r} to divide the batch size {k}");
    let half = module.ckks_slots_of_degree::<Standard>(n.as_usize());
    let packed = packed_layout(route, half)?;
    let target = packed.unwrap_or(layout);
    let eval_mod_plan = plan.eval_mod;
    let (s2c, c2s) = match route.transforms {
        ReplicatedTransforms::Dense => dense_factors::<BE, F>(module, plan, route, packed)?,
        ReplicatedTransforms::Factored { .. } => factored_factors::<F>(plan, route, packed)?,
    };
    let consumed = |factors: &[ReplicatedFactor<F>]| factors.iter().map(|f| f.log_delta_diag).collect::<Vec<_>>();
    let (s2c_consumed, c2s_consumed) = (consumed(&s2c), consumed(&c2s));
    ensure!(
        s2c_consumed.iter().sum::<usize>() == route.s2c_consumed_bits()
            && c2s_consumed.iter().sum::<usize>() == route.c2s_consumed_bits(),
        "the maps consume other bits than the route accounts for"
    );
    let encode = |target: ReplicatedLayout, factors: Vec<ReplicatedFactor<F>>, scratch: &mut ScratchArena<'_, BE>| {
        factors
            .into_iter()
            .map(|factor| ReplicatedTransform::<BE>::encode_factor_unprepared(module, target, base2k, factor, scratch))
            .collect::<Result<Vec<LinearTransformation<CKKSPlaintextOwned<BE>>>>>()
    };
    let slots_to_coeffs = DFTMatrix::from_factors(DFTMatrixFactors::new(
        plan.slots_to_coeffs.clone(),
        encode(layout, s2c, scratch)?,
    ));
    let coeffs_to_slots = DFTMatrix::from_factors(DFTMatrixFactors::new(
        plan.coeffs_to_slots.clone(),
        encode(target, c2s, scratch)?,
    ));
    let eval_mod = match packed {
        Some(double) => {
            ckks_eval_mod_replicated_packed_compile::<BE, F, Standard>(base2k, eval_mod_plan, double, module, scratch)?
        }
        None => ckks_eval_mod_replicated_compile::<BE, F, Standard>(base2k, eval_mod_plan, layout, module, scratch)?,
    };
    let stages = ReplicatedStages {
        route: *route,
        bottom_skip: (k * r).trailing_zeros() as usize + 1,
        top_skip: k.trailing_zeros() as usize + 1,
        top_sparsity: (half / k).trailing_zeros() as usize,
        packed,
        s2c_consumed,
        c2s_consumed,
        eval_mod,
    };
    Ok((coeffs_to_slots, slots_to_coeffs, stages))
}

/// The `k x k` matrix of the one-factor DFT of `dft` on `k` points from the generator: `M[c][j] = D_(j - c)[c]` for the
/// diagonals `D` of the factor.
pub(crate) fn one_factor_matrix<F: CKKSEncodingScalar>(kind: DFTType, log_k: usize, dft: &DFTPlan) -> Result<Vec<Vec<(F, F)>>> {
    let k = 1usize << log_k;
    let factor = ckks_dft_matrices_gen_blockwise::<F>(
        kind,
        log_k,
        log_k,
        &dft.schedule().depths(),
        dft.scaling().unwrap_or(1.0),
        dft.bit_reversed(),
    )
    .pop()
    .context("the generator gives a factor")?;
    let entry = |diagonals: &poulpy_core::layouts::Diagonals<F>, o: usize, c: usize| {
        diagonals
            .get(galois_of_rotation(o as i64))
            .map_or(F::zero(), |values| values[c % values.len()])
    };
    Ok((0..k)
        .map(|c| {
            (0..k)
                .map(|j| {
                    let o = (j + k - c) % k;
                    (entry(&factor.re, o, c), entry(&factor.im, o, c))
                })
                .collect()
        })
        .collect())
}

/// The dense maps, each one factor merging the subring DFT of the plan with the placement: `SlotsToCoeffs` gathers onto
/// `V·BR`, the generator's `Decode` of the subring of degree `2k`, which writes the data values as the coefficients of
/// that subring, bit-reversed; `CoeffsToSlots` is `BR·V^H/(2Kk)`, the generator's `Encode` scaled by `1/(2K)`, onto the
/// data slots, `[A; −iA]` on two batches when packed, `K` the EvalMod interval. The two bit reversals cancel.
#[allow(clippy::type_complexity)]
fn dense_factors<BE, F>(
    module: &Module<BE>,
    plan: &BootstrappingPlan,
    route: &ReplicatedRoute,
    packed: Option<ReplicatedLayout>,
) -> Result<(Vec<ReplicatedFactor<F>>, Vec<ReplicatedFactor<F>>)>
where
    BE: Backend,
    F: CKKSEncodingScalar,
    Module<BE>: CKKSModuleInfos,
{
    let layout = route.layout;
    let (log_k, k, r) = (layout.log_k(), 1usize << layout.log_k(), 1usize << layout.log_r());
    let decode = one_factor_matrix::<F>(DFTType::Decode, log_k, &plan.slots_to_coeffs)?;
    let encode = one_factor_matrix::<F>(DFTType::Encode, log_k, &plan.coeffs_to_slots)?;
    let s2c = gather_factor::<BE, F, _, Standard>(module, layout, route.s2c_log_delta_diag, false, |c, j| decode[c][j])?;
    let a = |t: usize, c: usize| encode[t][c];
    let c2s = match packed {
        // `−i·A` on the second batch.
        Some(double) => period_factor::<BE, F, _, Standard>(module, double, k, route.c2s_log_delta_diag, true, |s, c| {
            match (s % r, s / r) {
                (0, t) if t < k => a(t, c),
                (0, t) => {
                    let (re, im) = a(t - k, c);
                    (im, -re)
                }
                _ => (F::zero(), F::zero()),
            }
        })?,
        None => period_factor::<BE, F, _, Standard>(module, layout, k, route.c2s_log_delta_diag, true, |s, c| {
            if s.is_multiple_of(r) {
                a(s / r, c)
            } else {
                (F::zero(), F::zero())
            }
        })?,
    };
    Ok((vec![s2c], vec![c2s]))
}

/// The factored maps of [`ReplicatedTransforms::Factored`]:
///
/// - `SlotsToCoeffs` moves the data slots onto the first `k` slots of each period, weighted `R` for the normalized
///   trace onto period `k` that completes the map after the raise, then applies `V·BR`, the generator's `Decode` of
///   the plan's subring DFT, on them;
/// - `CoeffsToSlots` applies `BR·V^H/(2Kk)`, the generator's `Encode` of the plan's subring DFT, on the slots of
///   period `k`, then moves them onto the data slots of the `m` values of the target layout (`m = 2k` when packed, the
///   second batch times `-i`); its last factor is quantized with the shaping of the target.
#[allow(clippy::type_complexity)]
fn factored_factors<F>(
    plan: &BootstrappingPlan,
    route: &ReplicatedRoute,
    packed: Option<ReplicatedLayout>,
) -> Result<(Vec<ReplicatedFactor<F>>, Vec<ReplicatedFactor<F>>)>
where
    F: CKKSEncodingScalar,
{
    let ReplicatedTransforms::Factored {
        placement_factors,
        c2s_inner_log_delta_diag,
        ..
    } = route.transforms
    else {
        anyhow::bail!("the route's transforms are not factored");
    };
    let layout = route.layout;
    let (log_k, log_r) = (layout.log_k(), layout.log_r());
    let (k, r) = (1usize << log_k, 1usize << log_r);
    let unshaped = |log_delta_diag| {
        move |diagonals| ReplicatedFactor {
            diagonals,
            log_delta_diag,
            shaped: false,
            fitted_giant_step: true,
        }
    };
    let weight = F::from_usize(r).context("the period is a small integer")?;
    let (s2c_dft, c2s_dft) = (&plan.slots_to_coeffs, &plan.coeffs_to_slots);
    let s2c: Vec<ReplicatedFactor<F>> = placement_compress::<F>(k, r, placement_factors, weight)
        .into_iter()
        .chain(ckks_dft_matrices_gen_blockwise::<F>(
            DFTType::Decode,
            log_k,
            log_k + log_r,
            &s2c_dft.schedule().depths(),
            s2c_dft.scaling().unwrap_or(1.0),
            s2c_dft.bit_reversed(),
        ))
        .map(unshaped(route.s2c_log_delta_diag))
        .collect();

    let m = if packed.is_some() { 2 * k } else { k };
    let mut c2s: Vec<ReplicatedFactor<F>> = ckks_dft_matrices_gen_blockwise::<F>(
        DFTType::Encode,
        log_k,
        log_k,
        &c2s_dft.schedule().depths(),
        c2s_dft.scaling().unwrap_or(1.0),
        c2s_dft.bit_reversed(),
    )
    .into_iter()
    .chain(placement_spread::<F>(m, r, placement_factors))
    .map(unshaped(c2s_inner_log_delta_diag))
    .collect();
    let last = c2s.last_mut().context("CoeffsToSlots has factors")?;
    if m > k {
        // `-i` on the second batch.
        times_minus_i_from(&mut last.diagonals, r * k);
    }
    last.log_delta_diag = route.c2s_log_delta_diag;
    last.shaped = true;
    Ok((s2c, c2s))
}

/// Multiplies the outputs of slots `from..` of `diagonals` by `-i`: `(re, im) -> (im, -re)`.
fn times_minus_i_from<F: CKKSEncodingScalar>(diagonals: &mut ComplexDiagonals<F>, from: usize) {
    let n = diagonals.slots();
    let full = |values: Option<&Vec<F>>| -> Vec<F> {
        values.map_or_else(|| vec![F::zero(); n], |v| (0..n).map(|s| v[s % v.len()]).collect())
    };
    for o in diagonals.galois() {
        let (mut re, mut im) = (full(diagonals.re.get(o)), full(diagonals.im.get(o)));
        for s in from..n {
            (re[s], im[s]) = (im[s], -re[s]);
        }
        set_complex(diagonals, o, (re, im));
    }
}

impl<BE: Backend, F> BootstrappingContextPrepared<BE, F> {
    /// Galois elements of the automorphism keys of [`Self::replicated_key_widths`], conjugation (`-1`) among them.
    /// Empty on the full layout.
    pub fn replicated_galois_elements(&self, module: &Module<BE>) -> Vec<i64>
    where
        Module<BE>: GLWETrace<BE> + CyclotomicOrder,
    {
        let mut elements: Vec<i64> = self
            .replicated_key_uses(module, [0; 4])
            .unwrap_or_default()
            .into_iter()
            .map(|(p, _)| p)
            .collect();
        elements.sort_unstable();
        elements.dedup();
        elements
    }

    /// The automorphism keys the replicated route looks up, as `(galois element, k)`: the bottom trace and
    /// `SlotsToCoeffs` at `wide_in`, the top trace and `CoeffsToSlots` at `raised`, each factor's baby rotations at its
    /// input's width and its giant rotations at its output's, the conjugation at `after_c2s`, and with the packed
    /// EvalMod the rotation that adds the second batch at `after_eval_mod`. `None` on the full layout.
    fn replicated_key_uses(
        &self,
        module: &Module<BE>,
        [wide_in, raised, after_c2s, after_eval_mod]: [usize; 4],
    ) -> Option<Vec<(i64, usize)>>
    where
        Module<BE>: GLWETrace<BE> + CyclotomicOrder,
    {
        let stages = self.replicated()?;
        let n = self.bootstrap_degree();
        let log_n = n.ilog2() as usize;
        let order = crate::operand_cyclotomic_order::<_, Standard>("replicated_key_uses", module, n).ok()?;
        let rotation = |r: i64| galois_element(r.rem_euclid(order / 4), order);
        let trace = |skip: usize, k: usize| {
            module
                .glwe_trace_galois_elements(n.into(), skip..log_n, false)
                .into_iter()
                .map(move |p| (p, k))
        };
        let chain = |plans: &[poulpy_core::layouts::LinearTransformationPlan], consumed: &[usize], k: usize| {
            crate::layouts::validation::chain_key_lookups(plans, consumed, k)
                .into_iter()
                .map(|(galois, k)| (galois_at(galois, order), k))
                .collect::<Vec<_>>()
        };
        let packed = stages.packed.map(|double| {
            let k = 1i64 << double.log_k();
            (rotation((k / 2) << double.log_r()), after_eval_mod)
        });
        Some(
            trace(stages.bottom_skip, wide_in)
                .chain(chain(self.slots_to_coeffs().factor_plans(), &stages.s2c_consumed, wide_in))
                .chain(trace(stages.top_skip, raised))
                .chain(chain(self.coeffs_to_slots().factor_plans(), &stages.c2s_consumed, raised))
                .chain([(-1, after_c2s)])
                .chain(packed)
                .collect(),
        )
    }

    /// Generates the keys of the replicated route under `sk`, by the generator of the full layout: the automorphism
    /// keys of [`Self::replicated_galois_elements`], the tensor key, and with the plan's sparse secret the keys to and
    /// from an ephemeral secret of its weight, sampled from `source_xs` into the arena, which is wiped before returning.
    /// `layout` gives one automorphism key per Galois element, and `scratch` covers
    /// [`BootstrappingKeysLayout::generate_tmp_bytes`].
    ///
    /// # Errors
    ///
    /// `InvalidOperand` on the full layout, and as [`Self::generate_keys`], for a layout of other Galois elements, an
    /// encapsulation the plan does not have, or a first-factor or conversion key.
    #[allow(clippy::too_many_arguments)]
    pub fn generate_replicated_keys(
        &self,
        module: &Module<BE>,
        sk: &BackendGLWESecret<BE>,
        layout: &BootstrappingKeysLayout,
        source_xs: &mut Source,
        source_xe: &mut Source,
        source_xa: &mut Source,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<BootstrappingKeySet<BE::OwnedBuf, BE::ZnxWord>>
    where
        BE::OwnedBuf: HostDataMut,
        Module<BE>: ModuleCoreAlloc<OwnedBuf = BE::OwnedBuf, ZnxWord = BE::ZnxWord>
            + CyclotomicOrder
            + GLWETrace<BE>
            + GLWEAutomorphismKeyEncryptSk<BE>
            + GLWETensorKeyEncryptSk<BE>
            + GLWESwitchingKeyEncryptSk<BE>
            + GLWESecretSampling<BE>
            + VecZnxAutomorphism<BE>
            + VecZnxSwitchRing<BE>
            + GLWECIKeyEncryptSk<BE>,
        Module<BE>: GLWETensorKeyEncryptSk<BE, ConjugateInvariant> + GLWESecretSampling<BE, ConjugateInvariant>,
    {
        ensure!(
            self.replicated().is_some(),
            crate::CKKSCompositionError::InvalidOperand {
                op: "generate_replicated_keys",
                reason: "the context has no replicated route".into(),
            }
        );
        let order = crate::operand_cyclotomic_order::<_, Standard>("generate_replicated_keys", module, sk.n().as_usize())?;
        KeyRecipe::new(
            self.replicated_galois_elements(module).into_iter().collect(),
            Vec::new(),
            self.plan.sparse_secret_hamming_weight(),
            EvalModRing::Standard,
            order,
            0,
        )
        .generate_keys(module, sk, layout, source_xs, source_xe, source_xa, scratch)
    }

    /// The widths at which a bootstrapping of the replicated route into an output of `k_out` bits uses its keys, the
    /// one model the entry check, the key layouts and the tests read: each Galois element at its widest use, and the
    /// tensor key at the widest product EvalMod relinearizes. `None` on the full layout.
    pub fn replicated_key_widths(&self, module: &Module<BE>, k_out: usize) -> Option<BootstrappingKeyWidths>
    where
        Module<BE>: GLWETrace<BE> + CyclotomicOrder,
    {
        let (plan, route) = (&self.plan, &self.replicated()?.route);
        let wide_in = plan.replicated_input_k()? + route.extra_bits;
        let raised = plan.replicated_raised_k(k_out)?;
        let after_c2s = raised.checked_sub(route.c2s_consumed_bits())?;
        let after_eval_mod = after_c2s.checked_sub(route.extra_bits + plan.eval_mod.consumed_bits())?;
        let mut rotation_keys: BTreeMap<i64, usize> = BTreeMap::new();
        for (p, k) in self.replicated_key_uses(module, [wide_in, raised, after_c2s, after_eval_mod])? {
            let width = rotation_keys.entry(p).or_default();
            *width = k.max(*width);
        }
        Some(BootstrappingKeyWidths {
            rotation_keys,
            tensor_key: plan.replicated_tensor_key_k(k_out)?,
            conversion_keys: None,
        })
    }
}
