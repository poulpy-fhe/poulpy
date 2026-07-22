//! Bootstrapping parameterization.
//!
//! [`BootstrappingPlan`] bundles the parameters of the three homomorphic
//! sub-circuits that make up CKKS bootstrapping — CoeffsToSlots (an `Encode`
//! [`DFTPlan`]), EvalMod (an [`EvalModPlan`]) and SlotsToCoeffs (a `Decode`
//! [`DFTPlan`]).
//!
//! Only the per-stage circuit descriptions live here. The torus widths
//! (`base2k`, the encoding scale `log_delta`, and the input/raised moduli) are
//! per-ciphertext metadata supplied at call time — ModUp reads them from its
//! source and destination ciphertexts — and the message ratio is already part of
//! the [`EvalModPlan`]. See
//! [`CKKSBootstrappingOps`](crate::api::CKKSBootstrappingOps) for the ModUp
//! semantics (a digit shift in the base-`2^base2k` torus model, not an RNS
//! prime-basis extension).
//!
//! [`CKKSBootstrappingOps::ckks_bootstrap`](crate::api::CKKSBootstrappingOps)
//! is the orchestrator consuming these types: it composes `ModUp →
//! CoeffsToSlots → EvalMod → SlotsToCoeffs` from a compiled
//! [`BootstrappingContext`] and a prepared key set. The plan remains a plain
//! parameter bundle, so callers can also compose the stages manually from the
//! respective op traits.
//!
//! [`BootstrappingContext`] is the *compiled* form of a [`BootstrappingPlan`]:
//! the prepared, backend-resident homomorphic DFT matrices and the encoded,
//! uploaded EvalMod, built once and reused across bootstraps.

use anyhow::Result;
use poulpy_core::{
    default::linear_transformation::DiagonalProd,
    layouts::{Base2K, GLWEToBackendRef},
};
use poulpy_hal::layouts::{Backend, Module, ScratchArena};

use crate::{
    CKKSCtBounds,
    api::{CKKSDFTMatrixOps, CKKSDFTOps, CKKSEncodingOps, CKKSEncodingScalar},
    layouts::{
        CKKSModuleAlloc, CKKSPlaintext, DFTMatrix, DFTMatrixPrepared, DFTPlan, DFTType, Decode, Encode, EvalMod, EvalModPlan,
        Split, eval_mod::compile_eval_mod,
    },
};

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
    /// CoeffsToSlots: homomorphic encoding ([`DFTType::Encode`]).
    pub(crate) coeffs_to_slots: DFTPlan,

    /// CoeffsToSlots high precision for bypass.
    pub(crate) coeffs_to_slots_bypass: Option<DFTPlan>,

    /// EvalMod: approximate `x mod 1`. EvalMod runs at its own
    /// ([`EvalModPlan::f_mod_log_delta`]) scale — `ckks_eval_mod` sets the
    /// ciphertext to it on entry and restores the input scale on exit (a pure,
    /// budget-neutral reinterpretation), so it can keep more `ct×ct` precision.
    pub(crate) eval_mod: EvalModPlan,

    /// SlotsToCoeffs: homomorphic decoding ([`DFTType::Decode`]).
    pub(crate) slots_to_coeffs: DFTPlan,
}

impl BootstrappingPlan {
    /// Validated constructor: `coeffs_to_slots` must be a
    /// [`DFTType::Encode`] plan and `slots_to_coeffs` a [`DFTType::Decode`]
    /// plan (errors [`InvalidPlan`](crate::CKKSCompositionError::InvalidPlan)
    /// otherwise) — a swapped direction would make [`Self::galois_elements`]
    /// derive rotation keys for a transform that
    /// [`BootstrappingContext::compile`] (whose `Dir` markers are authoritative)
    /// never builds. Add the optional high-precision CoeffsToSlots with
    /// [`Self::with_coeffs_to_slots_bypass`].
    pub fn new(coeffs_to_slots: DFTPlan, eval_mod: EvalModPlan, slots_to_coeffs: DFTPlan) -> Result<Self> {
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
        Ok(Self {
            coeffs_to_slots,
            coeffs_to_slots_bypass: None,
            eval_mod,
            slots_to_coeffs,
        })
    }

    /// Sets the high-precision CoeffsToSlots bypass (the EvalRound+ pipeline).
    ///
    /// Errors if `bypass` is not a [`DFTType::Encode`] plan, or if the plan's
    /// `f_mod_interval` is not a power of two: the EvalRound+ pipeline computes
    /// `K·r0` as a power-of-two shift (`f_mod_interval.trailing_zeros()`),
    /// which is silently wrong for any other `K` — rejected here, at plan
    /// construction, rather than per bootstrap.
    pub fn with_coeffs_to_slots_bypass(mut self, bypass: DFTPlan) -> Result<Self> {
        let invalid = |reason: String| -> anyhow::Error {
            crate::CKKSCompositionError::InvalidPlan {
                plan: "BootstrappingPlan",
                reason,
            }
            .into()
        };
        if bypass.kind() != DFTType::Encode {
            return Err(invalid("coeffs_to_slots_bypass must be a DFTType::Encode plan".to_string()));
        }
        if !self.eval_mod.f_mod_interval.is_power_of_two() {
            return Err(invalid(format!(
                "coeffs_to_slots_bypass (EvalRound+) requires a power-of-two f_mod_interval, got {}",
                self.eval_mod.f_mod_interval
            )));
        }
        self.coeffs_to_slots_bypass = Some(bypass);
        Ok(self)
    }

    /// CoeffsToSlots: homomorphic encoding ([`DFTType::Encode`]).
    pub fn coeffs_to_slots(&self) -> &DFTPlan {
        &self.coeffs_to_slots
    }

    /// The optional high-precision CoeffsToSlots bypass (EvalRound+).
    pub fn coeffs_to_slots_bypass(&self) -> Option<&DFTPlan> {
        self.coeffs_to_slots_bypass.as_ref()
    }

    /// EvalMod: approximate `x mod 1`.
    pub fn eval_mod(&self) -> &EvalModPlan {
        &self.eval_mod
    }

    /// SlotsToCoeffs: homomorphic decoding ([`DFTType::Decode`]).
    pub fn slots_to_coeffs(&self) -> &DFTPlan {
        &self.slots_to_coeffs
    }

    /// Total `log_budget` bits the pipeline consumes: the two DFT stages plus
    /// EvalMod (charged at its own `f_mod_log_delta` scale; the surrounding
    /// set-scale round-trip is budget-neutral).
    pub fn consumed_bits(&self) -> usize {
        self.coeffs_to_slots.consumed_bits() + self.eval_mod.consumed_bits() + self.slots_to_coeffs.consumed_bits()
    }

    /// Distinct Galois elements the pipeline's rotation keys must cover: the union
    /// of the CoeffsToSlots and SlotsToCoeffs transforms
    /// ([`DFTPlan::galois_elements`]), for a ring of degree `2^log_n` and the given
    /// `cyclotomic_order`. The split forward transform's conjugation key is
    /// separate (generate it from Galois element `−1`) and not included here.
    pub fn galois_elements(&self, log_n: usize, cyclotomic_order: i64) -> Vec<i64> {
        let mut set: std::collections::BTreeSet<i64> = std::collections::BTreeSet::new();
        set.extend(self.coeffs_to_slots.galois_elements(log_n, cyclotomic_order));
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
    pub coeffs_to_slots: DFTMatrixPrepared<BE, Encode, Split>,

    /// Prepared bypass CoeffsToSlots matrix
    pub coeffs_to_slots_bypass: Option<DFTMatrixPrepared<BE, Encode, Split>>,

    /// Prepared SlotsToCoeffs matrix (homomorphic decoding).
    pub slots_to_coeffs: DFTMatrixPrepared<BE, Decode, Split>,

    /// Encoded, backend-resident EvalMod (`x mod 1`).
    pub eval_mod: EvalMod<F, CKKSPlaintext<BE::OwnedBuf>>,
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
        CKKSPlaintext<BE::OwnedBuf>: GLWEToBackendRef<BE> + CKKSCtBounds + DiagonalProd<BE>,
    {
        let c2s_lt: DFTMatrix<BE, Encode, Split> =
            module.ckks_new_dft_matrix::<Encode, Split>(base2k, &plan.coeffs_to_slots, scratch)?;
        let coeffs_to_slots = module.ckks_prepare_dft_matrix(&c2s_lt, scratch);

        let s2c_lt: DFTMatrix<BE, Decode, Split> =
            module.ckks_new_dft_matrix::<Decode, Split>(base2k, &plan.slots_to_coeffs, scratch)?;
        let slots_to_coeffs = module.ckks_prepare_dft_matrix(&s2c_lt, scratch);

        let eval_mod = compile_eval_mod::<BE, F>(base2k, plan.eval_mod, module, scratch)?;

        let coeffs_to_slots_bypass = if let Some(bypass) = &plan.coeffs_to_slots_bypass {
            // A bypass plan implies a power-of-two `f_mod_interval`; enforced by
            // `BootstrappingPlan::with_coeffs_to_slots_bypass` at construction.
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
        })
    }
}
