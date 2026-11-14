//! Default parameter sets for the bootstrapping of a replicated slot layout,
//! with the circuit of [`ReplicatedBootstrapPlan::paper`] in
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
//!   input width [`ReplicatedBootstrapPlan::k_in`];
//! - `p`: guaranteed output precision in bits on the data slots;
//! - `_factored`: factored transforms.
//!
//! The batch size `k`, a power of two with `R | k`, and the shaping unit are the
//! caller's. The input and output widths do not depend on them. The raised
//! modulus and the keys depend on the unit through the EvalMod scale
//! ([`ReplicatedBootstrapPlan::eval_mod_log_delta`]), which is wider for a unit
//! whose guards grow by more than 13 bits. A preset refuses a unit whose
//! suppression is less than 8 bits above its precision. Factored transforms
//! take a batch of at least `2^f` values for `f` factors per stage, whose real
//! and imaginary parts fit the ring side by side (`2kR <= N/2`).

use anyhow::{Result, ensure};
use poulpy_core::layouts::{Base2K, Degree, GLWELayout, Rank, TorusPrecision};

use crate::{
    CKKSLayout, GLWEPlaintextMeta, ReplicatedLayout, Scale, SlotLayout, SlotsKind,
    layouts::{BootstrappingKeyWidths, BootstrappingKeysLayout, ReplicatedBootstrapPlan, ReplicatedTransforms},
    presets::key_policy::BootstrappingKeyPolicy,
    shaping::ShapingUnit,
};

#[derive(Clone, Copy, Debug)]
struct PresetSpec {
    name: &'static str,
    log_n: usize,
    base2k: usize,
    levels: usize,
    log2_precision: usize,
    dense_secret_hamming_weight: usize,
    sparse_secret_hamming_weight: usize,
    max_dense_modulus: usize,
    max_sparse_modulus: usize,
    transforms: ReplicatedTransforms,
    s2c_log_delta_diag: usize,
}

/// A bootstrapping parameter set for a replicated slot layout: the circuit, the
/// widths, the secret weights and, for each unit, the key layouts, shaped by the
/// [`BootstrappingKeyPolicy`] from the widths the compiled bootstrapping uses
/// them at ([`ReplicatedBootstrapping::key_widths`](crate::layouts::ReplicatedBootstrapping::key_widths)).
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
        self.spec.dense_secret_hamming_weight
    }

    /// Hamming weight of the ephemeral sparse ModUp secret.
    pub fn sparse_secret_hamming_weight(&self) -> usize {
        self.spec.sparse_secret_hamming_weight
    }

    /// The circuit for a ciphertext in `layout`. Refuses a unit whose
    /// suppression is less than 8 bits above [`Self::log2_precision`]: the
    /// units of `R = 2` and those targeting 20 bits.
    pub fn plan(&self, layout: ReplicatedLayout) -> Result<ReplicatedBootstrapPlan> {
        check_unit(&self.spec, &layout)?;
        Ok(plan_for(&self.spec, layout))
    }

    /// Width the application hands a ciphertext back at, at the leveled scale,
    /// for every unit.
    pub fn input_k(&self) -> usize {
        let plan = reference_plan(&self.spec);
        plan.k_in() - (plan.log_delta - plan.level_log_delta)
    }

    /// Width after bootstrapping, at the leveled scale, for every unit.
    pub fn output_k(&self) -> usize {
        output_k(&reference_plan(&self.spec), self.spec.levels)
    }

    /// Evaluation-key layouts for this preset and `layout`, each key covering its
    /// width in `widths`, the widths at which the bootstrapping of `layout` into
    /// [`Self::output_k`] bits uses its keys
    /// ([`ReplicatedBootstrapping::key_widths`](crate::layouts::ReplicatedBootstrapping::key_widths)), with
    /// the digit of fewest rows within the modulus bound of the secret, and the
    /// encapsulation keys of the ephemeral secret. Refuses the units
    /// [`Self::plan`] refuses.
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
                scale: Scale::Log(reference_plan(&self.spec).level_log_delta),
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
        dense_secret_hamming_weight: 1024,
        sparse_secret_hamming_weight: 32,
        max_dense_modulus: 424,
        max_sparse_modulus: 77,
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
        dense_secret_hamming_weight: 1024,
        sparse_secret_hamming_weight: 32,
        max_dense_modulus: 851,
        max_sparse_modulus: 162,
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
        dense_secret_hamming_weight: 1024,
        sparse_secret_hamming_weight: 32,
        max_dense_modulus: 1707,
        max_sparse_modulus: 345,
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
fn reference_plan(spec: &PresetSpec) -> ReplicatedBootstrapPlan {
    plan_for(spec, ShapingUnit::U32_45.layout(4).expect("a valid unit"))
}

/// The circuit of ePrint 2026/2513 with the preset's transforms and `SlotsToCoeffs` scale.
fn plan_for(spec: &PresetSpec, layout: ReplicatedLayout) -> ReplicatedBootstrapPlan {
    ReplicatedBootstrapPlan {
        transforms: spec.transforms,
        s2c_log_delta_diag: spec.s2c_log_delta_diag,
        ..ReplicatedBootstrapPlan::paper(layout)
    }
}

fn output_k(plan: &ReplicatedBootstrapPlan, levels: usize) -> usize {
    plan.k_in() + levels * plan.level_log_delta
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

/// The policy the keys take their shape under: the modulus bounds of the secret
/// and of the ephemeral secret, digits uncapped, and one-limb digits for the
/// key to the ephemeral secret.
fn key_policy(spec: &PresetSpec) -> BootstrappingKeyPolicy {
    BootstrappingKeyPolicy {
        log_n: spec.log_n,
        base2k: spec.base2k,
        rank: 1,
        dense_log_modulus: spec.max_dense_modulus,
        sparse_log_modulus: spec.max_sparse_modulus,
        max_dsize: None,
        dense_to_sparse_dsize: 1,
    }
}

/// The key layouts of `plan`, its keys used at `widths`: the key to the
/// ephemeral secret covers the ModUp input, the message ratio above the
/// scale, and the key back its lifted window, its rows ending at the raised
/// width.
fn key_layouts(
    spec: &PresetSpec,
    plan: &ReplicatedBootstrapPlan,
    widths: &BootstrappingKeyWidths,
) -> Result<BootstrappingKeysLayout> {
    let raise_input_k = plan.log_delta + plan.eval_mod.log_msg_ratio;
    let raised = plan.k_boot(output_k(plan, spec.levels));
    key_policy(spec).keys_layout_at(
        widths,
        Some((raise_input_k, Some((raise_input_k + plan.mod_up_extra_bits, raised)))),
        None,
    )
}

/// The widest key each kind of key of `plan` covers: the raised width and the
/// working bits for the rotation keys, the widest EvalMod product for the
/// tensor key. Every narrower use fits where these do.
fn widest_key_widths(spec: &PresetSpec, plan: &ReplicatedBootstrapPlan) -> BootstrappingKeyWidths {
    let k_out = output_k(plan, spec.levels);
    BootstrappingKeyWidths {
        rotation_keys: [(-1, plan.key_k(k_out))].into(),
        tensor_key: plan.tensor_key_k(k_out),
        conversion_keys: None,
    }
}

/// Checks that the widest keys of every unit the preset accepts fit its modulus bounds.
fn build(spec: PresetSpec) -> Result<ReplicatedBootstrappingPreset> {
    ensure!(spec.levels > 0, "a preset restores at least one level");
    for unit in ShapingUnit::ALL {
        let layout = unit.layout(unit.log_r as u8)?;
        if check_unit(&spec, &layout).is_ok() {
            let plan = plan_for(&spec, layout);
            key_layouts(&spec, &plan, &widest_key_widths(&spec, &plan))?;
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

    /// The widest keys of the preset for `unit`, as the compiled bootstrapping reports them for the widest uses.
    fn widest(preset: &ReplicatedBootstrappingPreset, unit: ShapingUnit) -> BootstrappingKeyWidths {
        widest_key_widths(&preset.spec, &plan_for(&preset.spec, layout(unit)))
    }

    /// The gadgets of the widest rotation key, of the tensor key, of the key
    /// to the ephemeral secret and of the key back, as `(dnum, k_digit,
    /// k_aux)`, for a unit at the default EvalMod scale and for `U8_40`, whose
    /// EvalMod runs two bits wider: whole-limb digits of the fewest rows under
    /// the modulus bounds.
    #[test]
    fn preset_key_gadgets() {
        let gadget = |key: &dyn GGLWEInfos| {
            let layout = key.gglwe_layout();
            (layout.dnum.0, layout.k_digit.0, layout.k_aux.0)
        };
        let presets = [
            (n14_d20_k109_p21_s2c().unwrap(), ShapingUnit::U32_45),
            (n14_d20_k109_p21_s2c().unwrap(), ShapingUnit::U8_40),
            (n15_d20_k389_p20_s2c().unwrap(), ShapingUnit::U32_45),
            (n15_d20_k389_p20_s2c().unwrap(), ShapingUnit::U8_40),
            (n16_d20_k763_p19_s2c_factored().unwrap(), ShapingUnit::U32_45),
            (n16_d20_k763_p19_s2c_factored().unwrap(), ShapingUnit::U8_40),
        ];
        let got: Vec<_> = presets
            .iter()
            .map(|(preset, unit)| {
                let keys = preset.keys_layout(layout(*unit), &widest(preset, *unit)).unwrap();
                let encapsulation = keys.encapsulation.unwrap();
                (
                    preset.name(),
                    gadget(&keys.rotation_keys[&-1]),
                    gadget(&keys.tensor_key),
                    gadget(&encapsulation.dense_to_sparse),
                    gadget(&encapsulation.sparse_to_dense.unwrap()),
                )
            })
            .collect();
        let n14 = "n14_d20_k109_p21_s2c";
        let n15 = "n15_d20_k389_p20_s2c";
        let n16 = "n16_d20_k763_p19_s2c_factored";
        assert_eq!(
            got,
            [
                // Under 424 bits no whole-limb digit fits keys of 391 bits or more.
                (n14, (14, 28, 32), (7, 52, 60), (2, 22, 33), (2, 25, 39)),
                (n14, (34, 12, 16), (7, 52, 60), (2, 22, 33), (3, 17, 25)),
                (n15, (7, 104, 119), (3, 208, 223), (1, 52, 67), (1, 52, 67)),
                (n15, (7, 104, 119), (7, 104, 119), (1, 52, 67), (1, 52, 67)),
                (n16, (3, 416, 432), (2, 520, 536), (1, 52, 68), (1, 52, 68)),
                (n16, (3, 416, 432), (2, 520, 536), (1, 52, 68), (1, 52, 68)),
            ]
        );
    }

    /// The bootstrapping input, output and raised widths, the raised width of
    /// `U8_40` 16 bits wider.
    #[test]
    fn preset_widths() {
        for (preset, widths, u8_40_raised) in [
            (n14_d20_k109_p21_s2c().unwrap(), (69, 109, 383), 399),
            (n15_d20_k389_p20_s2c().unwrap(), (69, 389, 663), 679),
            (n16_d20_k763_p19_s2c_factored().unwrap(), (163, 763, 1199), 1215),
        ] {
            let plan = preset.plan(layout(ShapingUnit::U32_45)).unwrap();
            assert_eq!(
                (plan.k_in(), preset.output_k(), plan.k_boot(preset.output_k())),
                widths,
                "{}",
                preset.name()
            );
            let plan = preset.plan(layout(ShapingUnit::U8_40)).unwrap();
            assert_eq!(plan.k_in(), widths.0, "{}", preset.name());
            assert_eq!(plan.k_boot(preset.output_k()), u8_40_raised, "{}", preset.name());
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
                assert_eq!(preset.keys_layout(layout(unit), &widest(&preset, unit)).is_ok(), accepted);
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
                let raised = plan.k_boot(preset.output_k());
                assert!(return_key.k_offset.0 > 0);
                assert_eq!(
                    return_key.k_offset.as_usize() + return_key.gglwe_layout().gadget_k().as_usize(),
                    raised
                );
                assert!(return_key.k().as_usize() <= preset.spec.max_dense_modulus);
                assert!(keys.rotation_keys[&-1].k().as_usize() <= preset.spec.max_dense_modulus);
                assert!(keys.tensor_key.k().as_usize() <= preset.spec.max_dense_modulus);
                assert!(encapsulation.dense_to_sparse.k().as_usize() <= preset.spec.max_sparse_modulus);
                crate::layouts::validation::check_offset_raise(
                    "replicated_bootstrapping_preset",
                    raised,
                    plan.log_delta + plan.eval_mod.log_msg_ratio,
                    plan.mod_up_extra_bits,
                    &return_key,
                )
                .unwrap();
                assert!(
                    crate::layouts::validation::check_offset_raise(
                        "replicated_bootstrapping_preset",
                        raised + 1,
                        plan.log_delta + plan.eval_mod.log_msg_ratio,
                        plan.mod_up_extra_bits,
                        &return_key,
                    )
                    .is_err()
                );
            }
        }
    }
}
