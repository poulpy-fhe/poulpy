//! Default parameter sets for the bootstrapping of a replicated slot layout,
//! with the circuit of [`ReplicatedRoute::paper`] in
//! `SlotsToCoeffs`-first order, its transforms dense or factored
//! ([`ReplicatedTransforms`]).
//!
//! ## Naming
//!
//! As for [`super::bootstrapping`]: `n{log_n}_d{log_delta}_k{output_k}_p{log2_precision}_s2c`.
//!
//! - `n`: ring-degree exponent, which with the secret weights fixes the modulus bounds;
//! - `d`: leveled scale exponent of the inputs and the outputs;
//! - `k`: output width in bits, the levels at `2^d` above the bootstrapping
//!   input width [`BootstrappingPlan::replicated_input_k`];
//! - `p`: guaranteed output precision in bits on the data slots;
//! - `_factored`: factored transforms.
//!
//! The batch size `k`, a power of two with `R | k`, and the shaping unit are the
//! caller's. The input and output widths do not depend on them. The raised
//! modulus and the keys depend on the unit through the EvalMod scale
//! ([`ReplicatedRoute::eval_mod_log_delta`]), which is wider for a unit
//! whose guards grow by more than 13 bits. A preset refuses a unit whose
//! suppression is less than 8 bits above its precision. Factored transforms
//! take a batch of at least `2^f` values for `f` factors per stage. EvalMod runs on
//! the real and imaginary parts side by side when they fit the ring (`2kR <= N/2`).
//!
//! The secrets and their modulus bounds are those of the
//! [Security section of the bootstrapping presets](super::bootstrapping#security), which covers `N = 2^14` to `2^16`.

use anyhow::{Context, Result, ensure};
use poulpy_core::{
    Distribution,
    layouts::{Base2K, Degree, GLWELayout, Rank, TorusPrecision},
    security::{RlweParams, SecretSpec, SecurityTable},
};

use crate::{
    CKKSLayout, GLWEPlaintextMeta, ReplicatedLayout, Scale, SlotLayout, SlotsKind,
    layouts::{
        BootstrappingKeyWidths, BootstrappingKeysLayout, BootstrappingPlan, BootstrappingTechniques, ReplicatedRoute,
        ReplicatedTransforms, SparseSecretEncapsulation,
    },
    presets::key_policy::BootstrappingKeyPolicy,
    shaping::ShapingUnit,
};

use super::bootstrapping::{DENSE_HAMMING_WEIGHT, SPARSE_HAMMING_WEIGHT};

#[derive(Clone, Copy, Debug)]
struct PresetSpec {
    name: &'static str,
    log_n: usize,
    base2k: usize,
    levels: usize,
    log2_precision: usize,
    /// Hamming weights of the secrets, whose modulus bounds are those of [`RlweParams`] at 128 bits, as in the
    /// [Security section of the bootstrapping presets](super::bootstrapping#security).
    dense_hamming_weight: usize,
    sparse_hamming_weight: usize,
    transforms: ReplicatedTransforms,
    s2c_log_delta_diag: usize,
}

/// A bootstrapping parameter set for a replicated slot layout: the circuit, the widths, the secret weights and, for each
/// unit, the key layouts, shaped by the [`BootstrappingKeyPolicy`] from the widths the compiled bootstrapping uses them
/// at.
#[derive(Clone, Debug)]
pub struct ReplicatedBootstrappingPreset {
    spec: PresetSpec,
}

impl ReplicatedBootstrappingPreset {
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
        1 << self.spec.log_n
    }

    /// Limb radix exponent.
    pub fn base2k(&self) -> usize {
        self.spec.base2k
    }

    /// Levels at the leveled scale between two bootstrappings.
    pub fn levels(&self) -> usize {
        self.spec.levels
    }

    /// Guaranteed precision of the data slots in bits.
    pub fn log2_precision(&self) -> usize {
        self.spec.log2_precision
    }

    /// Hamming weight of the dense application secret.
    pub fn dense_secret_hamming_weight(&self) -> usize {
        self.spec.dense_hamming_weight
    }

    /// Hamming weight of the ephemeral sparse ModUp secret.
    pub fn sparse_secret_hamming_weight(&self) -> usize {
        self.spec.sparse_hamming_weight
    }

    /// The circuit for a ciphertext in `layout`. Refuses a unit whose
    /// suppression is less than 8 bits above [`Self::log2_precision`]: the
    /// units of `R = 2` and those targeting 20 bits.
    pub fn plan(&self, layout: ReplicatedLayout) -> Result<BootstrappingPlan> {
        check_unit(&self.spec, &layout)?;
        plan_for(&self.spec, layout)
    }

    /// Width the application hands a ciphertext back at, at the leveled scale,
    /// for every unit.
    pub fn input_k(&self) -> usize {
        let route = reference_route(&self.spec);
        input_k(&self.spec, &route) - (route.log_delta - route.level_log_delta)
    }

    /// Width after bootstrapping, at the leveled scale, for every unit.
    pub fn output_k(&self) -> usize {
        let route = reference_route(&self.spec);
        input_k(&self.spec, &route) + self.spec.levels * route.level_log_delta
    }

    /// Evaluation-key layouts for this preset and `layout`, each key covering its width in `widths`, the widths at which
    /// the bootstrapping of `layout` into [`Self::output_k`] bits uses its keys
    /// ([`BootstrappingContextPrepared::replicated_key_widths`](crate::layouts::BootstrappingContextPrepared::replicated_key_widths)),
    /// shaped by [`BootstrappingKeyPolicy`] under the preset's modulus bounds, with the encapsulation keys of the
    /// ephemeral secret. Refuses the units [`Self::plan`] refuses.
    pub fn keys_layout(&self, layout: ReplicatedLayout, widths: &BootstrappingKeyWidths) -> Result<BootstrappingKeysLayout> {
        key_layouts(&self.spec, &self.plan(layout)?, widths)
    }

    /// Layout of the bootstrap destination, in `layout`.
    pub fn output_layout(&self, layout: ReplicatedLayout) -> CKKSLayout {
        CKKSLayout {
            glwe_layout: GLWELayout {
                n: Degree(self.n() as u32),
                base2k: Base2K(self.spec.base2k as u32),
                k: TorusPrecision(self.output_k() as u32),
                rank: Rank(1),
            },
            meta: GLWEPlaintextMeta {
                scale: Scale::Log(reference_route(&self.spec).level_log_delta),
                slots: SlotsKind::Complex,
                log_sparsity: 0,
                slot_layout: SlotLayout::Replicated(layout),
            },
        }
    }

    /// Re-derives the preset at limb radix `base2k`, with its key layouts.
    pub fn with_base2k(&self, base2k: usize) -> Result<Self> {
        build(PresetSpec { base2k, ..self.spec })
    }

    /// The evaluation of `SlotsToCoeffs` and `CoeffsToSlots`.
    pub fn transforms(&self) -> ReplicatedTransforms {
        self.spec.transforms
    }

    /// Re-derives the preset with `transforms`, with its widths and key layouts.
    pub fn with_transforms(&self, transforms: ReplicatedTransforms) -> Result<Self> {
        build(PresetSpec { transforms, ..self.spec })
    }
}

/// `N = 2^14`: 2 levels at `2^20`, outputs of 109 bits, at least 21 bits of
/// precision.
pub fn n14_d20_k109_p21_s2c() -> Result<ReplicatedBootstrappingPreset> {
    build(PresetSpec {
        name: "n14_d20_k109_p21_s2c",
        log_n: 14,
        base2k: 52,
        levels: 2,
        log2_precision: 21,
        dense_hamming_weight: DENSE_HAMMING_WEIGHT,
        sparse_hamming_weight: SPARSE_HAMMING_WEIGHT,
        transforms: ReplicatedTransforms::Dense,
        s2c_log_delta_diag: 26,
    })
}

/// `N = 2^15`: 16 levels at `2^20`, outputs of 389 bits, at least 20 bits of
/// precision.
pub fn n15_d20_k389_p20_s2c() -> Result<ReplicatedBootstrappingPreset> {
    build(PresetSpec {
        name: "n15_d20_k389_p20_s2c",
        log_n: 15,
        base2k: 52,
        levels: 16,
        log2_precision: 20,
        dense_hamming_weight: DENSE_HAMMING_WEIGHT,
        sparse_hamming_weight: SPARSE_HAMMING_WEIGHT,
        transforms: ReplicatedTransforms::Dense,
        s2c_log_delta_diag: 26,
    })
}

/// `N = 2^16`: 30 levels at `2^20`, outputs of 763 bits, at least 19 bits of
/// precision. Each map is factored into two DFT factors and two placement
/// factors, the `SlotsToCoeffs` diagonals at `2^30` and the inner
/// `CoeffsToSlots` ones at `2^54`, for batches of 4 to `2^10` values with
/// `R = 16`.
pub fn n16_d20_k763_p19_s2c_factored() -> Result<ReplicatedBootstrappingPreset> {
    build(PresetSpec {
        name: "n16_d20_k763_p19_s2c_factored",
        log_n: 16,
        base2k: 52,
        levels: 30,
        log2_precision: 19,
        dense_hamming_weight: DENSE_HAMMING_WEIGHT,
        sparse_hamming_weight: SPARSE_HAMMING_WEIGHT,
        transforms: ReplicatedTransforms::Factored {
            s2c_factors: 2,
            c2s_factors: 2,
            placement_factors: 2,
            c2s_inner_log_delta_diag: 54,
        },
        s2c_log_delta_diag: 30,
    })
}

/// Every replicated bootstrapping preset.
pub fn all() -> Result<Vec<ReplicatedBootstrappingPreset>> {
    Ok(vec![
        n14_d20_k109_p21_s2c()?,
        n15_d20_k389_p20_s2c()?,
        n16_d20_k763_p19_s2c_factored()?,
    ])
}

/// The input and output widths do not depend on the layout.
fn reference_route(spec: &PresetSpec) -> ReplicatedRoute {
    route_for(spec, ShapingUnit::U32_45.layout(4).expect("a valid unit"))
}

/// The route of ePrint 2026/2513 with the preset's transforms and `SlotsToCoeffs` scale, packed when the two batches
/// fit the ring.
fn route_for(spec: &PresetSpec, layout: ReplicatedLayout) -> ReplicatedRoute {
    let (k, r) = (1usize << layout.log_k(), 1usize << layout.log_r());
    ReplicatedRoute {
        transforms: spec.transforms,
        s2c_log_delta_diag: spec.s2c_log_delta_diag,
        packed: 4 * k * r <= 1 << spec.log_n,
        ..ReplicatedRoute::paper(layout, spec.log_n)
    }
}

/// The plan of [`route_for`], its EvalMod [`ReplicatedRoute::paper_eval_mod`], with the preset's sparse secret.
fn plan_for(spec: &PresetSpec, layout: ReplicatedLayout) -> Result<BootstrappingPlan> {
    BootstrappingPlan::replicated(
        route_for(spec, layout),
        ReplicatedRoute::paper_eval_mod(&layout),
        BootstrappingTechniques {
            sparse_secret_encapsulation: Some(SparseSecretEncapsulation {
                hamming_weight: spec.sparse_hamming_weight,
            }),
            eval_round_plus: None,
        },
    )
}

/// The input width of `route` at `Δ_b`, which no unit changes.
fn input_k(spec: &PresetSpec, route: &ReplicatedRoute) -> usize {
    let _ = spec;
    route.log_delta + ReplicatedRoute::paper_eval_mod(&route.layout).log_msg_ratio + route.s2c_consumed_bits()
}

fn output_k(spec: &PresetSpec, plan: &BootstrappingPlan) -> usize {
    let route = plan.replicated_route().expect("a replicated plan");
    plan.replicated_input_k().unwrap_or_default() + spec.levels * route.level_log_delta
}

/// Bits by which a unit's suppression must exceed a preset's precision. The
/// bootstrapping falls short of the suppression by up to 5 bits: units of about
/// 20 bits measure 16 to 19 bits on the logN14 preset.
const SUPPRESSION_MARGIN_BITS: usize = 8;

/// A unit suppresses its rounding errors on the data slots by
/// [`ShapingProfile::suppression_bits`](crate::shaping::ShapingProfile::suppression_bits),
/// which bounds the precision of the bootstrapping.
fn check_unit(spec: &PresetSpec, layout: &ReplicatedLayout) -> Result<()> {
    let suppression = crate::shaping::ShapingProfile::<crate::Standard>::new(layout).suppression_bits();
    let needed = spec.log2_precision + SUPPRESSION_MARGIN_BITS;
    ensure!(
        suppression >= needed as f64,
        "{} guarantees {} bits, which needs a unit suppressing {needed} bits, but it suppresses {suppression:.2}",
        spec.name,
        spec.log2_precision
    );
    Ok(())
}

/// The policy that shapes the keys of the preset: whole-limb digits of the fewest rows under its modulus bounds, the key
/// to the ephemeral secret of one-limb digits.
fn key_policy(spec: &PresetSpec) -> Result<BootstrappingKeyPolicy> {
    let (dense_log_modulus, sparse_log_modulus) = log_modulus_bounds(spec)?;
    Ok(BootstrappingKeyPolicy {
        log_n: spec.log_n,
        base2k: spec.base2k,
        rank: 1,
        dense_log_modulus,
        sparse_log_modulus,
        max_dsize: None,
        dense_to_sparse_dsize: 1,
    })
}

/// Bounds, in bits, of the dense and sparse secrets of the preset, both in its ring: those of [`RlweParams`] at 128
/// bits.
fn log_modulus_bounds(spec: &PresetSpec) -> Result<(usize, usize)> {
    let n = 1 << spec.log_n;
    let bound = |weight| {
        RlweParams
            .max_log_q(128, SecretSpec::new(n, Distribution::TernaryFixed(weight)))
            .with_context(|| format!("no modulus bound for {}", spec.name))
    };
    Ok((bound(spec.dense_hamming_weight)?, bound(spec.sparse_hamming_weight)?))
}

/// The key layouts of `plan`, its keys used at `widths`: the key to the ephemeral secret covers the input of the
/// raise, the message ratio above `Δ_b`, and the key back its lifted window, its rows ending at the raised width.
fn key_layouts(spec: &PresetSpec, plan: &BootstrappingPlan, widths: &BootstrappingKeyWidths) -> Result<BootstrappingKeysLayout> {
    let route = plan.replicated_route().context("a replicated plan")?;
    let raise_input_k = route.log_delta + plan.eval_mod().log_msg_ratio;
    let window = raise_input_k + route.raise_bits();
    let raised = plan
        .replicated_raised_k(output_k(spec, plan))
        .context("the output holds the route")?;
    key_policy(spec)?.keys_layout_at(widths, Some((raise_input_k, Some((window, raised)))))
}

/// The widest key of each kind of `plan`: the raised width for the rotation keys, the widest EvalMod product for the
/// tensor key. Every narrower use fits where these do.
fn widest_key_widths(spec: &PresetSpec, plan: &BootstrappingPlan) -> Result<BootstrappingKeyWidths> {
    let k_out = output_k(spec, plan);
    Ok(BootstrappingKeyWidths {
        rotation_keys: [(-1, plan.replicated_raised_k(k_out).context("the output holds the route")?)].into(),
        tensor_key: plan.replicated_tensor_key_k(k_out).context("the output holds the route")?,
        conversion_keys: None,
    })
}

/// Checks that the keys of every unit the preset accepts fit its modulus bounds.
fn build(spec: PresetSpec) -> Result<ReplicatedBootstrappingPreset> {
    ensure!(spec.levels > 0, "a preset restores at least one level");
    for unit in ShapingUnit::ALL {
        let layout = unit.layout(unit.log_r as u8)?;
        if check_unit(&spec, &layout).is_ok() {
            let plan = plan_for(&spec, layout)?;
            key_layouts(&spec, &plan, &widest_key_widths(&spec, &plan)?)?;
        }
    }
    Ok(ReplicatedBootstrappingPreset { spec })
}

#[cfg(test)]
mod tests {
    use poulpy_core::layouts::{GGLWEInfos, LWEInfos};

    use super::*;

    fn layout(unit: ShapingUnit) -> ReplicatedLayout {
        unit.layout(6).unwrap()
    }

    /// The widest keys of `preset` for `unit`.
    fn widest(preset: &ReplicatedBootstrappingPreset, unit: ShapingUnit) -> BootstrappingKeyWidths {
        widest_key_widths(&preset.spec, &plan_for(&preset.spec, layout(unit)).unwrap()).unwrap()
    }

    /// The gadgets of the widest rotation key, of the tensor key, of the key to the ephemeral secret and of the key
    /// back, as `(dnum, k_digit, k_aux)`, for a unit at the default EvalMod scale and for `U8_40`, whose EvalMod runs
    /// two bits wider: the digits of fewest rows under the modulus bounds, off a limb boundary where no whole limb fits.
    #[test]
    fn preset_key_gadgets() {
        let gadget = |key: &dyn GGLWEInfos| {
            let layout = key.gglwe_layout();
            (layout.dnum.0, layout.k_digit.0, layout.k_aux.0)
        };
        for (preset, unit, rotation, tensor, ephemeral, return_key) in [
            (
                n14_d20_k109_p21_s2c().unwrap(),
                ShapingUnit::U32_45,
                (14, 28, 32),
                (7, 52, 60),
                (2, 22, 33),
                (2, 29, 33),
            ),
            (
                n14_d20_k109_p21_s2c().unwrap(),
                ShapingUnit::U8_40,
                (34, 12, 16),
                (7, 52, 60),
                (2, 22, 33),
                (4, 15, 17),
            ),
            (
                n15_d20_k389_p20_s2c().unwrap(),
                ShapingUnit::U32_45,
                (7, 104, 119),
                (3, 208, 223),
                (1, 52, 67),
                (1, 104, 119),
            ),
            (
                n15_d20_k389_p20_s2c().unwrap(),
                ShapingUnit::U8_40,
                (7, 104, 119),
                (7, 104, 119),
                (1, 52, 67),
                (1, 104, 119),
            ),
            (
                n16_d20_k763_p19_s2c_factored().unwrap(),
                ShapingUnit::U32_45,
                (3, 416, 432),
                (2, 520, 536),
                (1, 52, 68),
                (1, 104, 120),
            ),
            (
                n16_d20_k763_p19_s2c_factored().unwrap(),
                ShapingUnit::U8_40,
                (3, 416, 432),
                (2, 520, 536),
                (1, 52, 68),
                (1, 104, 120),
            ),
        ] {
            let layout = preset.keys_layout(layout(unit), &widest(&preset, unit)).unwrap();
            let encapsulation = layout.encapsulation.unwrap();
            let name = preset.name();
            assert_eq!(gadget(&layout.rotation_keys[&-1]), rotation, "{name} {unit:?}");
            assert_eq!(gadget(&layout.tensor_key), tensor, "{name} {unit:?}");
            assert_eq!(gadget(&encapsulation.dense_to_sparse), ephemeral, "{name} {unit:?}");
            assert_eq!(gadget(&encapsulation.sparse_to_dense.unwrap()), return_key, "{name} {unit:?}");
        }
    }

    /// The bootstrapping input, output and raised widths, the raised width of
    /// `U8_40` 16 bits wider.
    #[test]
    fn preset_widths() {
        for (preset, widths, u8_40_raised) in [
            (n14_d20_k109_p21_s2c().unwrap(), (69, 109, 391), 407),
            (n15_d20_k389_p20_s2c().unwrap(), (69, 389, 671), 687),
            (n16_d20_k763_p19_s2c_factored().unwrap(), (163, 763, 1207), 1223),
        ] {
            let plan = preset.plan(layout(ShapingUnit::U32_45)).unwrap();
            let raised = |plan: &BootstrappingPlan| plan.replicated_raised_k(preset.output_k()).unwrap();
            assert_eq!(
                (plan.replicated_input_k().unwrap(), preset.output_k(), raised(&plan)),
                widths,
                "{}",
                preset.name()
            );
            let plan = preset.plan(layout(ShapingUnit::U8_40)).unwrap();
            assert_eq!(plan.replicated_input_k().unwrap(), widths.0, "{}", preset.name());
            assert_eq!(raised(&plan), u8_40_raised, "{}", preset.name());
        }
    }

    /// Every preset takes the units of at least 30 bits of suppression, and
    /// refuses those of `R = 2` and those targeting 20 bits.
    #[test]
    fn presets_refuse_units_below_their_precision() {
        for preset in all().unwrap() {
            for unit in ShapingUnit::ALL {
                let accepted = preset.plan(layout(unit)).is_ok();
                let strong = unit.log_r > 1 && unit.target_bits >= 30;
                assert_eq!(accepted, strong, "{} with {unit:?}", preset.name());
                assert_eq!(
                    preset.plan(layout(unit)).is_ok() && preset.keys_layout(layout(unit), &widest(&preset, unit)).is_ok(),
                    accepted
                );
            }
        }
    }

    #[test]
    fn offset_keys_cover_the_raise_within_modulus_bounds() {
        for (preset, unit) in all()
            .unwrap()
            .into_iter()
            .flat_map(|preset| [(preset.clone(), ShapingUnit::U32_45), (preset, ShapingUnit::U8_40)])
        {
            for base2k in [18, 19, 28, 52, 54] {
                let preset = preset.with_base2k(base2k).unwrap();
                let plan = preset.plan(layout(unit)).unwrap();
                let keys = preset.keys_layout(layout(unit), &widest(&preset, unit)).unwrap();
                let encapsulation = keys.encapsulation.unwrap();
                let return_key = encapsulation.sparse_to_dense.unwrap();
                let raised = plan.replicated_raised_k(preset.output_k()).unwrap();
                let route = *plan.replicated_route().unwrap();
                let k_small = route.log_delta + plan.eval_mod().log_msg_ratio;
                assert!(return_key.k_offset.0 > 0);
                assert_eq!(
                    return_key.k_offset.as_usize() + return_key.gglwe_layout().gadget_k().as_usize(),
                    raised
                );
                let (dense, sparse) = log_modulus_bounds(&preset.spec).unwrap();
                assert!(return_key.k().as_usize() <= dense);
                assert!(keys.rotation_keys[&-1].k().as_usize() <= dense);
                assert!(keys.tensor_key.k().as_usize() <= dense);
                assert!(encapsulation.dense_to_sparse.k().as_usize() <= sparse);
                crate::layouts::validation::check_offset_raise(
                    "replicated_bootstrapping_preset",
                    raised,
                    k_small,
                    route.raise_bits(),
                    &return_key,
                )
                .unwrap();
                assert!(
                    crate::layouts::validation::check_offset_raise(
                        "replicated_bootstrapping_preset",
                        raised + 1,
                        k_small,
                        route.raise_bits(),
                        &return_key,
                    )
                    .is_err()
                );
            }
        }
    }
}
