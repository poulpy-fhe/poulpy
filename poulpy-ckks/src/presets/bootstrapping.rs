//! Default parameter sets for ModUp/EvalMod bootstrapping.
//!
//! Presets bundle the circuit plan with its ciphertext widths, secret weights,
//! and physical evaluation-key layouts. This keeps modulus accounting and key
//! bounds attached to the recipe that requires them.

use anyhow::{Context, Result, ensure};
use poulpy_core::layouts::{
    Base2K, Degree, Dnum, Dsize, GGLWEInfos, GLWEAutomorphismKeyLayout, GLWELayout, GLWESwitchingKeyLayout, GLWETensorKeyLayout,
    LWEInfos, Rank, TorusPrecision,
};

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
    restored_levels: usize,
    dense_secret_hamming_weight: usize,
    sparse_secret_hamming_weight: usize,
    max_dense_modulus: usize,
    max_sparse_modulus: usize,
    key_dsize: usize,
    dense_to_sparse_dsize: usize,
    pipeline: BootstrappingPipeline,
    log_msg_ratio: usize,
    c2s_schedule: &'static [(usize, usize)],
    c2s_log_delta: usize,
    c2s_log_budget: usize,
    s2c_schedule: &'static [(usize, usize)],
    s2c_log_delta: usize,
    s2c_log_budget: usize,
    eval_mod: EvalModSpec,
}

/// A complete CKKS bootstrapping parameter set.
///
/// The input and output widths are composable: consuming
/// [`restored_levels`](Self::restored_levels) levels from the output leaves
/// exactly [`input_k`](Self::input_k) bits, enough to invoke the same preset
/// again. The bootstrap allocation is wider than the logical output because it
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

    /// Ciphertext scale exponent.
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

    /// Number of usable multiplication/rescale levels before the next bootstrap.
    pub fn restored_levels(&self) -> usize {
        self.spec.restored_levels
    }

    /// Hamming weight of the dense application secret.
    pub fn dense_secret_hamming_weight(&self) -> usize {
        self.spec.dense_secret_hamming_weight
    }

    /// Hamming weight of the ephemeral sparse ModUp secret.
    pub fn sparse_secret_hamming_weight(&self) -> usize {
        self.spec.sparse_secret_hamming_weight
    }

    /// Configured modulus bound for objects under the dense secret.
    pub fn max_dense_modulus(&self) -> usize {
        self.spec.max_dense_modulus
    }

    /// Configured modulus bound for objects under the ephemeral sparse secret.
    pub fn max_sparse_modulus(&self) -> usize {
        self.spec.max_sparse_modulus
    }

    /// Width required by an input ciphertext.
    pub fn input_k(&self) -> usize {
        self.input_k
    }

    /// Logical width after bootstrapping.
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

    /// Logical layout produced by the bootstrap.
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
            meta: CKKSMeta {
                log_delta: self.spec.log_delta,
                log_sparsity: 0,
                slots: SlotsKind::Complex,
            },
        }
    }
}

/// Full-slot presets for `N = 2^16`.
pub mod log_n16 {
    use super::*;

    /// C2S-first preset restoring 16 net levels at `log_delta = 35`.
    ///
    /// Uses an optimized Han–Ki EvalMod. The input, output, and raised widths
    /// are respectively 40, 600, and 1404 bits.
    pub fn c2s_16_levels() -> Result<BootstrappingPreset> {
        build(PresetSpec {
            name: "c2s_16_levels",
            log_n: 16,
            base2k: 52,
            rank: 1,
            log_delta: 35,
            restored_levels: 16,
            dense_secret_hamming_weight: 1024,
            sparse_secret_hamming_weight: 32,
            max_dense_modulus: 1714,
            max_sparse_modulus: 349,
            key_dsize: 4,
            dense_to_sparse_dsize: 3,
            pipeline: BootstrappingPipeline::C2SFirst,
            log_msg_ratio: 5,
            c2s_schedule: &C2S_SCHEDULE,
            c2s_log_delta: 50,
            c2s_log_budget: 2,
            s2c_schedule: &S2C_SCHEDULE,
            s2c_log_delta: 35,
            s2c_log_budget: 2,
            eval_mod: optimized_han_ki(),
        })
    }

    /// S2C-first preset restoring 16 net levels at `log_delta = 35`.
    ///
    /// Uses an optimized Han–Ki EvalMod. The initial S2C is evaluated below
    /// ModUp; the input, output, and raised widths are respectively 158, 718,
    /// and 1358 bits.
    pub fn s2c_16_levels() -> Result<BootstrappingPreset> {
        build(PresetSpec {
            name: "s2c_16_levels",
            log_n: 16,
            base2k: 52,
            rank: 1,
            log_delta: 35,
            restored_levels: 16,
            dense_secret_hamming_weight: 1024,
            sparse_secret_hamming_weight: 32,
            max_dense_modulus: 1714,
            max_sparse_modulus: 349,
            key_dsize: 4,
            dense_to_sparse_dsize: 3,
            pipeline: BootstrappingPipeline::S2CFirst,
            log_msg_ratio: 11,
            c2s_schedule: &C2S_SCHEDULE,
            c2s_log_delta: 44,
            c2s_log_budget: 3,
            s2c_schedule: &S2C_SCHEDULE,
            s2c_log_delta: 28,
            s2c_log_budget: 2,
            eval_mod: optimized_han_ki(),
        })
    }
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
    let restored_bits = spec
        .restored_levels
        .checked_mul(spec.log_delta)
        .context("bootstrapping preset restored width overflow")?;
    let output_k = input_k
        .checked_add(restored_bits)
        .context("bootstrapping preset output width overflow")?;
    let bootstrap_k = plan.bootstrap_k(output_k);
    let keys_layout = keys_layout(&spec, n, bootstrap_k, log_modulus);

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

fn key_shape(spec: &PresetSpec, input_k: usize, dsize: usize) -> (Dnum, TorusPrecision) {
    let digit_width = dsize * spec.base2k;
    let dnum = input_k.div_ceil(digit_width);
    (Dnum(dnum as u32), TorusPrecision((digit_width + spec.log_n) as u32))
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
    let rounded_k = key.dnum().as_usize() * key.dsize().as_usize() * key.base2k().as_usize();
    ensure!(rounded_k <= limit, "{name} key rounded modulus {rounded_k} exceeds {limit}");
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

    #[test]
    fn log_n16_c2s_16_levels_is_composable_and_within_bounds() {
        let preset = log_n16::c2s_16_levels().unwrap();

        assert_eq!(preset.plan().pipeline(), BootstrappingPipeline::C2SFirst);
        assert_eq!(preset.plan().eval_mod().eval_mod_type, EvalModType::CosHKEven);
        assert_eq!(preset.plan().eval_mod().consumed_bits(), 464);
        assert_eq!(preset.plan().coeffs_to_slots().consumed_bits(), 200);
        assert_eq!(preset.plan().slots_to_coeffs().consumed_bits(), 140);
        assert_eq!((preset.input_k(), preset.output_k(), preset.bootstrap_k()), (40, 600, 1404));
        assert_eq!(
            preset.output_k() - preset.restored_levels() * preset.log_delta(),
            preset.input_k()
        );

        assert_layouts_within_bounds(&preset);
    }

    #[test]
    fn log_n16_s2c_16_levels_is_composable_and_within_bounds() {
        let preset = log_n16::s2c_16_levels().unwrap();

        assert_eq!(preset.plan().pipeline(), BootstrappingPipeline::S2CFirst);
        assert_eq!(preset.plan().eval_mod().eval_mod_type, EvalModType::CosHKEven);
        assert_eq!(preset.plan().eval_mod().consumed_bits(), 464);
        assert_eq!(preset.plan().coeffs_to_slots().consumed_bits(), 176);
        assert_eq!(preset.plan().slots_to_coeffs().consumed_bits(), 112);
        assert_eq!((preset.input_k(), preset.output_k(), preset.bootstrap_k()), (158, 718, 1358));
        assert_eq!(
            preset.output_k() - preset.restored_levels() * preset.log_delta(),
            preset.input_k()
        );

        assert_layouts_within_bounds(&preset);
    }

    fn assert_layouts_within_bounds(preset: &BootstrappingPreset) {
        let keys = preset.keys_layout();
        let encapsulation = keys.encapsulation.as_ref().unwrap();

        assert_eq!(keys.automorphism_key.dsize.as_usize(), 4);
        assert_eq!(rounded_k(&keys.automorphism_key), 1456);
        assert_eq!(keys.automorphism_key.k_aux.as_usize(), 224);
        assert_eq!(keys.automorphism_key.k().as_usize(), 1680);
        assert_eq!(keys.tensor_key.k().as_usize(), 1680);
        assert_eq!(encapsulation.sparse_to_dense.k().as_usize(), 1680);
        assert!(keys.automorphism_key.k().as_usize() <= preset.max_dense_modulus());

        assert_eq!(encapsulation.dense_to_sparse.dsize.as_usize(), 3);
        assert_eq!(rounded_k(&encapsulation.dense_to_sparse), 156);
        assert_eq!(encapsulation.dense_to_sparse.k_aux.as_usize(), 172);
        assert_eq!(encapsulation.dense_to_sparse.k().as_usize(), 328);
        assert!(encapsulation.dense_to_sparse.k().as_usize() <= preset.max_sparse_modulus());
    }

    fn rounded_k<K: GGLWEInfos + LWEInfos>(key: &K) -> usize {
        key.dnum().as_usize() * key.dsize().as_usize() * key.base2k().as_usize()
    }
}
