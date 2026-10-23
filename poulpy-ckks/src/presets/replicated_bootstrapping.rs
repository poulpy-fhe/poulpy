//! Default parameter sets for the bootstrapping of a replicated slot layout,
//! with the circuit of [`ReplicatedBootstrapPlan::paper`] in
//! `SlotsToCoeffs`-first order.
//!
//! ## Naming
//!
//! As for [`super::bootstrapping`]: `n{log_n}_d{log_delta}_k{output_k}_p{log2_precision}_s2c`.
//!
//! - `n`: ring-degree exponent, which with the secret weights fixes the modulus bounds;
//! - `d`: leveled scale exponent of the inputs and the outputs;
//! - `k`: output width in bits, the levels at `2^d` above the bootstrapping
//!   input width [`ReplicatedBootstrapPlan::k_in`];
//! - `p`: guaranteed output precision in bits on the data slots.
//!
//! The batch size `k`, a power of two with `R | k`, and the shaping unit are the
//! caller's: the moduli do not depend on them.

use anyhow::{Result, ensure};
use poulpy_core::layouts::{Base2K, Degree, GLWELayout, Rank, TorusPrecision};

use crate::{
    CKKSLayout, GLWEPlaintextMeta, ReplicatedLayout, Scale, SlotLayout, SlotsKind,
    layouts::{BootstrappingKeysLayout, ReplicatedBootstrapPlan},
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
}

/// A bootstrapping parameter set for a replicated slot layout: the circuit, the
/// widths, the secret weights and the key layouts, whose gadgets have the fewest
/// digits within the modulus bounds
/// ([`ReplicatedBootstrapPlan::secure_key_layouts`]).
#[derive(Clone, Debug)]
pub struct ReplicatedBootstrappingPreset {
    spec: PresetSpec,
    keys_layout: BootstrappingKeysLayout,
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

    /// The circuit for a ciphertext in `layout`.
    pub fn plan(&self, layout: ReplicatedLayout) -> ReplicatedBootstrapPlan {
        ReplicatedBootstrapPlan::paper(layout)
    }

    /// Width the application hands a ciphertext back at, at the leveled scale.
    pub fn input_k(&self) -> usize {
        let plan = reference_plan();
        plan.k_in() - (plan.log_delta - plan.level_log_delta)
    }

    /// Width after bootstrapping, at the leveled scale.
    pub fn output_k(&self) -> usize {
        output_k(&reference_plan(), self.spec.levels)
    }

    /// Evaluation-key layouts sized for this preset.
    pub fn keys_layout(&self) -> &BootstrappingKeysLayout {
        &self.keys_layout
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
                scale: Scale::Log(reference_plan().level_log_delta),
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
    })
}

/// Every replicated bootstrapping preset.
pub fn all() -> Result<Vec<ReplicatedBootstrappingPreset>> {
    Ok(vec![n14_d20_k109_p21_s2c()?, n15_d20_k389_p20_s2c()?])
}

/// The circuit's widths do not depend on the layout.
fn reference_plan() -> ReplicatedBootstrapPlan {
    ReplicatedBootstrapPlan::paper(ShapingUnit::U32_45.layout(4, 24).expect("a valid unit"))
}

fn output_k(plan: &ReplicatedBootstrapPlan, levels: usize) -> usize {
    plan.k_in() + levels * plan.level_log_delta
}

fn build(spec: PresetSpec) -> Result<ReplicatedBootstrappingPreset> {
    ensure!(spec.levels > 0, "a preset restores at least one level");
    let plan = reference_plan();
    let keys_layout = plan.secure_key_layouts(
        Degree(1 << spec.log_n),
        Base2K(spec.base2k as u32),
        output_k(&plan, spec.levels),
        spec.max_dense_modulus,
        spec.max_sparse_modulus,
    )?;
    Ok(ReplicatedBootstrappingPreset { spec, keys_layout })
}

#[cfg(test)]
mod tests {
    use poulpy_core::layouts::{GGLWEInfos, LWEInfos};

    use super::*;

    /// The gadgets of the keys under the secret and of the key to the
    /// ephemeral secret, as `(dnum, k_digit, k_aux)`.
    #[test]
    fn preset_key_gadgets() {
        let gadget = |key: &dyn GGLWEInfos| {
            let layout = key.gglwe_layout();
            (layout.dnum.0, layout.k_digit.0, layout.k_aux.0)
        };
        for (preset, keys, ephemeral, return_key) in [
            (n14_d20_k109_p21_s2c().unwrap(), (14, 28, 28), (2, 22, 22), (2, 25, 25)),
            (n15_d20_k389_p20_s2c().unwrap(), (4, 168, 168), (1, 43, 43), (1, 49, 49)),
        ] {
            let layout = preset.keys_layout();
            let encapsulation = layout.encapsulation.unwrap();
            assert_eq!(gadget(&layout.automorphism_key), keys, "{}", preset.name());
            assert_eq!(gadget(&layout.tensor_key), keys, "{}", preset.name());
            assert_eq!(gadget(&encapsulation.dense_to_sparse), ephemeral, "{}", preset.name());
            assert_eq!(
                gadget(&encapsulation.sparse_to_dense.unwrap()),
                return_key,
                "{}",
                preset.name()
            );
        }
    }

    #[test]
    fn offset_keys_cover_the_raise_within_modulus_bounds() {
        for preset in all().unwrap() {
            for base2k in [18, 19, 28, 52, 54] {
                let preset = preset.with_base2k(base2k).unwrap();
                let plan = reference_plan();
                let keys = preset.keys_layout();
                let encapsulation = keys.encapsulation.unwrap();
                let return_key = encapsulation.sparse_to_dense.unwrap();
                let raised = plan.k_boot(preset.output_k());
                assert!(return_key.k_offset.0 > 0);
                assert_eq!(
                    return_key.k_offset.as_usize() + return_key.gglwe_layout().gadget_k().as_usize(),
                    raised
                );
                assert!(return_key.k().as_usize() <= preset.spec.max_dense_modulus);
                assert!(keys.automorphism_key.k().as_usize() <= preset.spec.max_dense_modulus);
                assert!(keys.tensor_key.k().as_usize() <= preset.spec.max_dense_modulus);
                assert!(encapsulation.dense_to_sparse.k().as_usize() <= preset.spec.max_sparse_modulus);
                crate::reference::bootstrapping::check_offset_raise(
                    raised,
                    plan.log_delta + plan.eval_mod.log_msg_ratio,
                    plan.mod_up_extra_bits,
                    &return_key,
                )
                .unwrap();
                assert!(
                    crate::reference::bootstrapping::check_offset_raise(
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
