//! Bootstrapping parameterization.
//!
//! [`BootstrappingPlan`] bundles the parameters of the three homomorphic
//! sub-circuits that make up CKKS bootstrapping — CoeffsToSlots (an `Encode`
//! [`DFTPlan`]), EvalMod (an [`EvalModPlan`]) and SlotsToCoeffs (a `Decode`
//! [`DFTPlan`]).
//!
//! The plan also selects the ModUp/EvalMod pipeline and its optional techniques.
//! Torus widths (`base2k`, the encoding scale `log_delta`, and the input/raised
//! moduli) remain per-ciphertext metadata supplied at call time — ModUp reads
//! them from its source and destination ciphertexts — and the message ratio is
//! already part of the [`EvalModPlan`]. See
//! [`CKKSBootstrappingOps`](crate::api::CKKSBootstrappingOps) for the ModUp
//! semantics (a digit shift in the base-`2^base2k` torus model, not an RNS
//! prime-basis extension).
//!
//! [`CKKSBootstrappingOps::ckks_bootstrap`](crate::api::CKKSBootstrappingOps)
//! composes the selected pipeline from a compiled [`BootstrappingContext`] and
//! a prepared key set. The plan remains a plain parameter bundle, so callers
//! can also compose the stages manually from the respective op traits.
//!
//! [`BootstrappingContext`] is the *compiled* form of a [`BootstrappingPlan`]:
//! the prepared, backend-resident homomorphic DFT matrices and the encoded,
//! uploaded EvalMod, built once and reused across bootstraps.

use crate::layouts::CKKSPlaintextOwned;
use anyhow::{Result, ensure};
use poulpy_core::{
    default::linear_transformation::DiagonalProd,
    layouts::{Base2K, GLWEToBackendRef},
};
use poulpy_hal::layouts::{Backend, Module, ScratchArena};

use crate::{
    CKKSCtBounds, CKKSInfos,
    api::{CKKSDFTMatrixOps, CKKSDFTOps, CKKSEncodingOps, CKKSEncodingScalar},
    layouts::{
        CKKSModuleAlloc, DFTMatrix, DFTMatrixPrepared, DFTPlan, DFTType, Decode, Encode, EncodedLut, EvalMod, EvalModPlan, Split,
        eval_mod::compile_eval_mod,
    },
};

/// Stage ordering for the ModUp/EvalMod bootstrapping family.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BootstrappingPipeline {
    /// CoeffsToSlots before EvalMod and SlotsToCoeffs.
    C2SFirst,
    /// SlotsToCoeffs below ModUp, then CoeffsToSlots and EvalMod above it.
    S2CFirst,
}

/// Sparse-secret encapsulation parameters for ModUp.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SparseSecretEncapsulation {
    /// Sparse-secret ModRaise encapsulation (<https://eprint.iacr.org/2022/024>);
    /// this is the ephemeral ternary secret weight. Use `None` in
    /// [`BootstrappingTechniques`] to disable the technique.
    pub hamming_weight: usize,
}

/// EvalRound+ parameters (<https://eprint.iacr.org/2024/1379>).
#[derive(Clone, Debug)]
pub struct EvalRoundPlus {
    /// High-precision CoeffsToSlots transform bypassing EvalMod.
    pub coeffs_to_slots_bypass: DFTPlan,
}

/// Optional techniques composed into a ModUp/EvalMod recipe.
#[derive(Clone, Debug, Default)]
pub struct BootstrappingTechniques {
    /// Sparse-secret key switches around ModUp.
    pub sparse_secret_encapsulation: Option<SparseSecretEncapsulation>,
    /// EvalRound+ high-precision bypass.
    pub eval_round_plus: Option<EvalRoundPlus>,
}

/// End-to-end parameterization of CKKS bootstrapping.
///
/// Built with [`Self::new`], which validates the stage roles once — a
/// `BootstrappingPlan` that exists always has an `Encode` CoeffsToSlots and a
/// `Decode` SlotsToCoeffs, so key provisioning ([`Self::galois_elements`])
/// cannot silently derive rotations for the wrong transform direction.
///
/// See the [module docs](self) for the overall role.
#[derive(Clone, Debug)]
pub struct BootstrappingPlan {
    pipeline: BootstrappingPipeline,

    techniques: BootstrappingTechniques,

    /// CoeffsToSlots: homomorphic encoding ([`DFTType::Encode`]).
    pub(crate) coeffs_to_slots: DFTPlan,
    /// EvalMod: approximate `x mod 1`. EvalMod runs at its own
    /// ([`EvalModPlan::f_mod_log_delta`]) scale — `ckks_eval_mod` sets the
    /// ciphertext to it on entry and restores the input scale on exit (a pure,
    /// budget-neutral reinterpretation), so it can keep more `ct×ct` precision.
    pub(crate) eval_mod: EvalModPlan,

    /// SlotsToCoeffs: homomorphic decoding ([`DFTType::Decode`]).
    pub(crate) slots_to_coeffs: DFTPlan,
}

impl BootstrappingPlan {
    /// Validated constructor for a complete bootstrapping recipe.
    ///
    /// `coeffs_to_slots` must be a
    /// [`DFTType::Encode`] plan and `slots_to_coeffs` a [`DFTType::Decode`]
    /// plan (errors [`InvalidPlan`](crate::CKKSCompositionError::InvalidPlan)
    /// otherwise) — a swapped direction would make [`Self::galois_elements`]
    /// derive rotation keys for a transform that
    /// [`BootstrappingContext::compile`] never builds. The constructor also
    /// rejects recipe techniques that the selected pipeline cannot evaluate.
    pub fn new(
        pipeline: BootstrappingPipeline,
        techniques: BootstrappingTechniques,
        coeffs_to_slots: DFTPlan,
        eval_mod: EvalModPlan,
        slots_to_coeffs: DFTPlan,
    ) -> Result<Self> {
        let invalid = |reason: String| -> anyhow::Error {
            crate::CKKSCompositionError::InvalidPlan {
                plan: "BootstrappingPlan",
                reason,
            }
            .into()
        };
        if coeffs_to_slots.kind() != DFTType::Encode {
            return Err(invalid("coeffs_to_slots must be a DFTType::Encode plan".to_string()));
        }
        if slots_to_coeffs.kind() != DFTType::Decode {
            return Err(invalid("slots_to_coeffs must be a DFTType::Decode plan".to_string()));
        }
        if let Some(sse) = techniques.sparse_secret_encapsulation
            && sse.hamming_weight == 0
        {
            return Err(invalid(
                "sparse-secret encapsulation hamming_weight must be nonzero".to_string(),
            ));
        }
        if let Some(eval_round) = &techniques.eval_round_plus {
            if eval_round.coeffs_to_slots_bypass.kind() != DFTType::Encode {
                return Err(invalid(
                    "EvalRound+ coeffs_to_slots_bypass must be a DFTType::Encode plan".to_string(),
                ));
            }
            if !eval_mod.f_mod_interval.is_power_of_two() {
                return Err(invalid(format!(
                    "EvalRound+ requires a power-of-two f_mod_interval, got {}",
                    eval_mod.f_mod_interval
                )));
            }
        }
        Ok(Self {
            pipeline,
            techniques,
            coeffs_to_slots,
            eval_mod,
            slots_to_coeffs,
        })
    }

    /// The selected ModUp/EvalMod bootstrapping pipeline.
    pub fn pipeline(&self) -> BootstrappingPipeline {
        self.pipeline
    }

    /// Optional techniques applied by the recipe.
    pub fn techniques(&self) -> &BootstrappingTechniques {
        &self.techniques
    }

    /// CoeffsToSlots: homomorphic encoding ([`DFTType::Encode`]).
    pub fn coeffs_to_slots(&self) -> &DFTPlan {
        &self.coeffs_to_slots
    }

    /// The optional high-precision CoeffsToSlots bypass (EvalRound+).
    pub fn coeffs_to_slots_bypass(&self) -> Option<&DFTPlan> {
        self.techniques
            .eval_round_plus
            .as_ref()
            .map(|eval_round| &eval_round.coeffs_to_slots_bypass)
    }

    /// Ephemeral sparse-secret weight required by the recipe, if enabled.
    pub fn sparse_secret_hamming_weight(&self) -> Option<usize> {
        self.techniques.sparse_secret_encapsulation.map(|sse| sse.hamming_weight)
    }

    /// EvalMod: approximate `x mod 1`.
    pub fn eval_mod(&self) -> &EvalModPlan {
        &self.eval_mod
    }

    /// SlotsToCoeffs: homomorphic decoding ([`DFTType::Decode`]).
    pub fn slots_to_coeffs(&self) -> &DFTPlan {
        &self.slots_to_coeffs
    }

    /// Budget consumed before ModUp.
    pub fn pre_mod_up_consumed_bits(&self) -> usize {
        match self.pipeline {
            BootstrappingPipeline::C2SFirst => 0,
            BootstrappingPipeline::S2CFirst => self.slots_to_coeffs.consumed_bits(),
        }
    }

    /// Budget consumed after ModUp.
    ///
    /// EvalRound+ evaluates its bypass in parallel with the low-precision
    /// CoeffsToSlots + EvalMod branch, so the wider of the two branch costs is
    /// charged before any trailing SlotsToCoeffs.
    pub fn post_mod_up_consumed_bits(&self) -> usize {
        let c2s_eval_mod = self.coeffs_to_slots.consumed_bits() + self.eval_mod.consumed_bits();
        let eval_round = self
            .coeffs_to_slots_bypass()
            .map_or(c2s_eval_mod, |bypass| c2s_eval_mod.max(bypass.consumed_bits()));
        match self.pipeline {
            BootstrappingPipeline::C2SFirst => eval_round + self.slots_to_coeffs.consumed_bits(),
            BootstrappingPipeline::S2CFirst => eval_round,
        }
    }

    /// Input width for a given modulus at ModUp.
    pub fn input_k(&self, log_modulus: usize) -> usize {
        log_modulus + self.pre_mod_up_consumed_bits()
    }

    /// Bootstrap width for a desired output width.
    pub fn bootstrap_k(&self, output_k: usize) -> usize {
        output_k + self.post_mod_up_consumed_bits()
    }

    /// Raised width required by an S2C-first functional bootstrap.
    ///
    /// `log_delta` is the input ciphertext scale; the LUT message ratio is
    /// folded in to obtain the post-S2C scale used by the LUT evaluation.
    pub fn functional_bootstrap_k<P>(&self, output_k: usize, log_delta: usize, lut: &EncodedLut<P>) -> Result<usize>
    where
        P: CKKSInfos,
    {
        ensure!(
            self.pipeline == BootstrappingPipeline::S2CFirst,
            "functional bootstrapping requires an S2C-first plan"
        );
        let eval_mod = if lut.requires_eval_mod() {
            self.eval_mod.consumed_bits()
        } else {
            0
        };
        let lut_log_delta = log_delta + lut.log_msg_ratio();
        Ok(output_k + self.coeffs_to_slots.consumed_bits() + eval_mod + lut.consumed_bits(lut_log_delta))
    }

    /// Total `log_budget` bits the pipeline consumes: the two DFT stages plus
    /// EvalMod (charged at its own `f_mod_log_delta` scale; the surrounding
    /// set-scale round-trip is budget-neutral).
    pub fn consumed_bits(&self) -> usize {
        self.pre_mod_up_consumed_bits() + self.post_mod_up_consumed_bits()
    }

    /// Distinct Galois elements the pipeline's rotation keys must cover: the union
    /// of the CoeffsToSlots and SlotsToCoeffs transforms
    /// ([`DFTPlan::galois_elements`]), for a ring of degree `2^log_n` and the given
    /// `cyclotomic_order`. The split forward transform's conjugation key is
    /// separate (generate it from Galois element `−1`) and not included here.
    pub fn galois_elements(&self, log_n: usize, cyclotomic_order: i64) -> Vec<i64> {
        let mut set: std::collections::BTreeSet<i64> = std::collections::BTreeSet::new();
        set.extend(self.coeffs_to_slots.galois_elements(log_n, cyclotomic_order));
        if let Some(bypass) = self.coeffs_to_slots_bypass() {
            set.extend(bypass.galois_elements(log_n, cyclotomic_order));
        }
        set.extend(self.slots_to_coeffs.galois_elements(log_n, cyclotomic_order));
        set.into_iter().collect()
    }
}

/// Compiled [`BootstrappingPlan`]: the resident DFT matrices and encoded EvalMod
/// the bootstrapping pipeline evaluates.
///
/// Built once by [`Self::compile`] and reused across bootstraps. The
/// `SplitRealAndImag` format is used for both transforms so the real and
/// imaginary coefficient halves can be reduced independently by EvalMod.
pub struct BootstrappingContext<BE: Backend, F> {
    /// Prepared CoeffsToSlots matrix (homomorphic encoding).
    ///
    /// The `1/K` amplitude bridge into EvalMod's domain is **not** applied here:
    /// the caller owns it, by folding it into the plan (e.g. building the
    /// CoeffsToSlots stage with
    /// [`with_scaling`](DFTPlan::with_scaling)`(1.0 / f_mod_interval as f64)`)
    /// before [`Self::compile`]. `compile` performs no implicit scaling.
    coeffs_to_slots: DFTMatrixPrepared<BE, Encode, Split>,

    /// Prepared bypass CoeffsToSlots matrix
    coeffs_to_slots_bypass: Option<DFTMatrixPrepared<BE, Encode, Split>>,

    /// Prepared SlotsToCoeffs matrix (homomorphic decoding). S2C-first plans
    /// use scaling `1/2` to cancel the initial real/imaginary split.
    slots_to_coeffs: DFTMatrixPrepared<BE, Decode, Split>,

    /// Encoded, backend-resident EvalMod (`x mod 1`).
    eval_mod: EvalMod<F, CKKSPlaintextOwned<BE>>,

    /// Selected ModUp/EvalMod pipeline.
    pipeline: BootstrappingPipeline,

    /// Ephemeral sparse-secret weight required by the recipe, if enabled.
    sparse_secret_hamming_weight: Option<usize>,
}

impl<BE: Backend, F> BootstrappingContext<BE, F> {
    /// Selected ModUp/EvalMod pipeline.
    pub fn pipeline(&self) -> BootstrappingPipeline {
        self.pipeline
    }

    /// Prepared CoeffsToSlots matrix.
    pub fn coeffs_to_slots(&self) -> &DFTMatrixPrepared<BE, Encode, Split> {
        &self.coeffs_to_slots
    }

    /// Prepared EvalRound+ bypass matrix, when enabled.
    pub fn coeffs_to_slots_bypass(&self) -> Option<&DFTMatrixPrepared<BE, Encode, Split>> {
        self.coeffs_to_slots_bypass.as_ref()
    }

    /// Prepared SlotsToCoeffs matrix.
    pub fn slots_to_coeffs(&self) -> &DFTMatrixPrepared<BE, Decode, Split> {
        &self.slots_to_coeffs
    }

    /// Encoded, backend-resident EvalMod circuit.
    pub fn eval_mod(&self) -> &EvalMod<F, CKKSPlaintextOwned<BE>> {
        &self.eval_mod
    }

    /// Ephemeral sparse-secret weight required by the compiled recipe.
    pub fn sparse_secret_hamming_weight(&self) -> Option<usize> {
        self.sparse_secret_hamming_weight
    }
}

impl<BE: Backend, F> BootstrappingContext<BE, F>
where
    F: CKKSEncodingScalar,
{
    /// Compiles `plan` directly into backend-resident matrices and EvalMod.
    ///
    /// Each stage carries its own coefficient metadata (`DFTPlan::meta` /
    /// `EvalModPlan::meta`), read directly off the plan, including any `scaling`
    /// the caller folded in (the `1/f_mod_interval` amplitude bridge — see the
    /// [`Self::coeffs_to_slots`] docs); no implicit scaling is applied here.
    pub fn compile(
        module: &Module<BE>,
        base2k: Base2K,
        plan: &BootstrappingPlan,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<Self>
    where
        Module<BE>: CKKSDFTOps<BE> + CKKSDFTMatrixOps<BE, F> + CKKSModuleAlloc<BE> + CKKSEncodingOps<BE, F>,
        CKKSPlaintextOwned<BE>: GLWEToBackendRef<BE> + CKKSCtBounds + DiagonalProd<BE>,
    {
        let c2s_lt: DFTMatrix<BE, Encode, Split> =
            module.ckks_new_dft_matrix::<Encode, Split>(base2k, &plan.coeffs_to_slots, scratch)?;
        let coeffs_to_slots = module.ckks_prepare_dft_matrix(&c2s_lt, scratch);

        let s2c_lt: DFTMatrix<BE, Decode, Split> =
            module.ckks_new_dft_matrix::<Decode, Split>(base2k, &plan.slots_to_coeffs, scratch)?;
        let slots_to_coeffs = module.ckks_prepare_dft_matrix(&s2c_lt, scratch);

        let eval_mod = compile_eval_mod::<BE, F>(base2k, plan.eval_mod, module, scratch)?;

        let coeffs_to_slots_bypass = if let Some(bypass) = plan.coeffs_to_slots_bypass() {
            let c2s_lt: DFTMatrix<BE, Encode, Split> = module.ckks_new_dft_matrix::<Encode, Split>(base2k, bypass, scratch)?;

            Some(module.ckks_prepare_dft_matrix(&c2s_lt, scratch))
        } else {
            None
        };

        Ok(Self {
            coeffs_to_slots,
            coeffs_to_slots_bypass,
            slots_to_coeffs,
            eval_mod,
            pipeline: plan.pipeline,
            sparse_secret_hamming_weight: plan.sparse_secret_hamming_weight(),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        CoeffsMeta,
        layouts::{DFTOutputFormat, eval_mod::EvalModType},
        polynomial::SplitStrategy,
    };

    fn dft(kind: DFTType) -> DFTPlan {
        DFTPlan::new(
            kind,
            vec![(1, 1)],
            DFTOutputFormat::SplitRealAndImag,
            CoeffsMeta::from_delta_budget(8, 2),
        )
        .unwrap()
    }

    fn eval_mod(f_mod_interval: usize) -> EvalModPlan {
        EvalModPlan {
            eval_mod_type: EvalModType::CosHK,
            log_msg_ratio: 2,
            f_mod_degree: 3,
            f_mod_interval,
            f_mod_log_interval_reduction: 1,
            f_mod_inv_degree: None,
            scaling: None,
            split_strategy: SplitStrategy::MinDepth,
            coeffs_meta: CoeffsMeta::from_delta_budget(8, 2),
            f_mod_log_delta: 10,
        }
    }

    fn plan(
        pipeline: BootstrappingPipeline,
        techniques: BootstrappingTechniques,
        f_mod_interval: usize,
    ) -> Result<BootstrappingPlan> {
        BootstrappingPlan::new(
            pipeline,
            techniques,
            dft(DFTType::Encode),
            eval_mod(f_mod_interval),
            dft(DFTType::Decode),
        )
    }

    #[test]
    fn recipe_records_sparse_secret_weight() {
        let plan = plan(
            BootstrappingPipeline::C2SFirst,
            BootstrappingTechniques {
                sparse_secret_encapsulation: Some(SparseSecretEncapsulation { hamming_weight: 32 }),
                eval_round_plus: None,
            },
            16,
        )
        .unwrap();

        assert_eq!(plan.sparse_secret_hamming_weight(), Some(32));
    }

    #[test]
    fn recipe_accepts_s2c_first_pipeline() {
        let plan = plan(BootstrappingPipeline::S2CFirst, BootstrappingTechniques::default(), 16).unwrap();
        assert_eq!(plan.pipeline(), BootstrappingPipeline::S2CFirst);
    }

    #[test]
    fn recipe_accounts_for_pipeline_order() {
        let c2s = plan(BootstrappingPipeline::C2SFirst, BootstrappingTechniques::default(), 16).unwrap();
        assert_eq!(c2s.pre_mod_up_consumed_bits(), 0);
        assert_eq!(c2s.post_mod_up_consumed_bits(), c2s.consumed_bits());
        assert_eq!(c2s.input_k(20), 20);
        assert_eq!(c2s.bootstrap_k(30), 30 + c2s.consumed_bits());

        let s2c = plan(BootstrappingPipeline::S2CFirst, BootstrappingTechniques::default(), 16).unwrap();
        assert_eq!(s2c.pre_mod_up_consumed_bits(), s2c.slots_to_coeffs().consumed_bits());
        assert_eq!(
            s2c.post_mod_up_consumed_bits(),
            s2c.coeffs_to_slots().consumed_bits() + s2c.eval_mod().consumed_bits()
        );
        assert_eq!(s2c.input_k(20), 20 + s2c.slots_to_coeffs().consumed_bits());
        assert_eq!(s2c.bootstrap_k(30), 30 + s2c.post_mod_up_consumed_bits());
    }

    #[test]
    fn recipe_accepts_eval_round_plus_with_s2c_first() {
        let plan = plan(
            BootstrappingPipeline::S2CFirst,
            BootstrappingTechniques {
                sparse_secret_encapsulation: None,
                eval_round_plus: Some(EvalRoundPlus {
                    coeffs_to_slots_bypass: dft(DFTType::Encode),
                }),
            },
            16,
        )
        .unwrap();
        assert!(plan.coeffs_to_slots_bypass().is_some());
    }

    #[test]
    fn recipe_accounts_for_eval_round_plus_bypass() {
        let bypass = DFTPlan::new(
            DFTType::Encode,
            vec![(1, 1); 10],
            DFTOutputFormat::SplitRealAndImag,
            CoeffsMeta::from_delta_budget(8, 2),
        )
        .unwrap();
        let bypass_cost = bypass.consumed_bits();

        for pipeline in [BootstrappingPipeline::C2SFirst, BootstrappingPipeline::S2CFirst] {
            let plan = plan(
                pipeline,
                BootstrappingTechniques {
                    sparse_secret_encapsulation: None,
                    eval_round_plus: Some(EvalRoundPlus {
                        coeffs_to_slots_bypass: bypass.clone(),
                    }),
                },
                16,
            )
            .unwrap();

            let trailing_s2c = match pipeline {
                BootstrappingPipeline::C2SFirst => plan.slots_to_coeffs().consumed_bits(),
                BootstrappingPipeline::S2CFirst => 0,
            };
            assert!(bypass_cost > plan.coeffs_to_slots().consumed_bits() + plan.eval_mod().consumed_bits());
            assert_eq!(plan.post_mod_up_consumed_bits(), bypass_cost + trailing_s2c);
        }
    }

    #[test]
    fn recipe_rejects_zero_sparse_secret_weight() {
        let err = plan(
            BootstrappingPipeline::C2SFirst,
            BootstrappingTechniques {
                sparse_secret_encapsulation: Some(SparseSecretEncapsulation { hamming_weight: 0 }),
                eval_round_plus: None,
            },
            16,
        )
        .unwrap_err();
        assert!(err.to_string().contains("hamming_weight"));
    }

    #[test]
    fn eval_round_plus_requires_power_of_two_interval() {
        let err = plan(
            BootstrappingPipeline::C2SFirst,
            BootstrappingTechniques {
                sparse_secret_encapsulation: None,
                eval_round_plus: Some(EvalRoundPlus {
                    coeffs_to_slots_bypass: dft(DFTType::Encode),
                }),
            },
            3,
        )
        .unwrap_err();
        assert!(err.to_string().contains("power-of-two"));
    }
}
