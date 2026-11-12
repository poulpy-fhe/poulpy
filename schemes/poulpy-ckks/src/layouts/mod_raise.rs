//! The modulus raise of a bootstrap as a step of its own: what the plan fixes and the first factor it may carry.

use anyhow::Result;
use poulpy_core::layouts::{LinearTransformation, LinearTransformationDiagonal, LinearTransformationGiantStep};
use poulpy_hal::layouts::{Backend, Module, PrepareHint, Ring, ScratchArena, Standard};

use crate::{
    CKKSResult,
    api::CKKSRightOperandPreparedFactory,
    layouts::{BootstrappingContextPrepared, CKKSPlaintextOwned, CKKSRightOperandPrepared, EvalModPlan},
};

/// What the modulus raise of a [`BootstrappingPlan`](crate::layouts::BootstrappingPlan) fixes, built by
/// [`BootstrappingPlan::mod_raise`](crate::layouts::BootstrappingPlan::mod_raise). The raise is a digit shift in the
/// base-`2^base2k` torus, not an RNS basis extension: it leaves `I(X)·q + Δ·m` in `res`, which keeps its `k`, so
/// `log_budget_out = res.k() - log_delta_out`, and takes the slot kind and sparsity of `a`.
///
/// The order is the first lift, the dense-to-sparse switch, the projection on the subring of the gap, ModUp with the
/// second lift or the guard bits in its shift, the first CoeffsToSlots factor, then the sparse-to-dense switch. Without
/// a lift or encapsulation, `a` is only read.
#[derive(Clone, Debug, PartialEq)]
pub struct ModRaisePlan {
    pub(crate) lift: Option<EvalModPlan>,
    pub(crate) c2s_guard_bits: usize,
    pub(crate) sparse_secret_hamming_weight: Option<usize>,
    pub(crate) first_factor_on_raise: bool,
    pub(crate) log_gap: usize,
}

impl ModRaisePlan {
    /// C2S-first: EvalMod's plan, whose two lifts make the message the residue of the raise and `I(X)·q` its integer
    /// part. The first multiplies `a` in place, up to the plan's message ratio, and the second is fused into the shift
    /// of ModUp, so `log_delta_out = a.log_delta() + lifts + log_msg_ratio`. `None` on S2C-first.
    pub fn lift(&self) -> Option<&EvalModPlan> {
        self.lift.as_ref()
    }

    /// S2C-first: bits fused into the shift of ModUp, so `log_delta_out = a.k() + c2s_guard_bits`.
    pub fn c2s_guard_bits(&self) -> usize {
        self.c2s_guard_bits
    }

    /// Weight of the sparse secret of
    /// [`SparseSecretEncapsulation`](crate::layouts::SparseSecretEncapsulation): `a` is switched to it in place before
    /// ModUp, and `res` back to the dense secret after it.
    pub fn sparse_secret_hamming_weight(&self) -> Option<usize> {
        self.sparse_secret_hamming_weight
    }

    /// Whether the first CoeffsToSlots factor runs on the raise, while the ciphertext is small, from a
    /// [`ModRaisePrepared`]. It consumes no modulus, and its switches replace the sparse-to-dense one.
    pub fn first_factor_on_raise(&self) -> bool {
        self.first_factor_on_raise
    }

    /// `log2` of the gap of the plaintexts, see
    /// [`BootstrappingPlan::with_log_gap`](crate::layouts::BootstrappingPlan::with_log_gap).
    pub fn log_gap(&self) -> usize {
        self.log_gap
    }

    /// The bits the raise of an input of `k` bits at scale `log_delta` lifts it by in place, and the bits fused into
    /// the shift of ModUp, or the refusal of `op` when the input cannot reach the lifts.
    pub(crate) fn scale_ups(&self, op: &'static str, k: usize, log_delta: usize) -> CKKSResult<(usize, usize)> {
        match &self.lift {
            Some(lift) => {
                let lift_bits = lift.input_scale_up(op, k, log_delta)?;
                Ok((lift_bits, lift.raised_scale_up(op, log_delta + lift_bits)?))
            }
            None => Ok((0, self.c2s_guard_bits)),
        }
    }
}

/// A [`ModRaisePlan`] with the first CoeffsToSlots factor its raise applies, if the plan leaves it there.
pub struct ModRaisePrepared<BE: Backend, Rg: Ring = Standard> {
    pub(crate) plan: ModRaisePlan,
    pub(crate) first_factor: Option<LinearTransformation<CKKSRightOperandPrepared<BE, Rg>>>,
}

/// Read access to the plan of a modulus raise and to its first factor.
pub trait CKKSModRaiseInfos<BE: Backend> {
    /// What the raise fixes: the lifts or the guard bits, the encapsulation and where the first CoeffsToSlots factor
    /// runs.
    fn mod_raise_plan(&self) -> &ModRaisePlan;

    /// The prepared first CoeffsToSlots factor the raise applies, `None` when the plan leaves it to CoeffsToSlots. A
    /// plan with [`ModRaisePlan::first_factor_on_raise`] takes one.
    fn first_factor(&self) -> Option<&LinearTransformation<CKKSRightOperandPrepared<BE>>>;

    /// `log2` of the gap of the plaintexts, see
    /// [`BootstrappingPlan::with_log_gap`](crate::layouts::BootstrappingPlan::with_log_gap).
    fn log_gap(&self) -> usize {
        self.mod_raise_plan().log_gap()
    }
}

impl<BE: Backend> CKKSModRaiseInfos<BE> for ModRaisePlan {
    fn mod_raise_plan(&self) -> &ModRaisePlan {
        self
    }

    fn first_factor(&self) -> Option<&LinearTransformation<CKKSRightOperandPrepared<BE>>> {
        None
    }
}

impl<BE: Backend> CKKSModRaiseInfos<BE> for ModRaisePrepared<BE> {
    fn mod_raise_plan(&self) -> &ModRaisePlan {
        &self.plan
    }

    fn first_factor(&self) -> Option<&LinearTransformation<CKKSRightOperandPrepared<BE>>> {
        self.first_factor.as_ref()
    }
}

impl<BE: Backend, F> CKKSModRaiseInfos<BE> for BootstrappingContextPrepared<BE, F> {
    fn mod_raise_plan(&self) -> &ModRaisePlan {
        self.mod_raise.mod_raise_plan()
    }

    fn first_factor(&self) -> Option<&LinearTransformation<CKKSRightOperandPrepared<BE>>> {
        self.mod_raise.first_factor()
    }
}

/// Prepares each diagonal of a first CoeffsToSlots factor for the products on a ciphertext of a few limbs.
pub(crate) fn first_factor_prepare<BE: Backend, Rg: Ring>(
    module: &Module<BE>,
    factor: LinearTransformation<CKKSPlaintextOwned<BE, Rg>>,
    scratch: &mut ScratchArena<'_, BE>,
) -> Result<LinearTransformation<CKKSRightOperandPrepared<BE, Rg>>>
where
    Module<BE>: CKKSRightOperandPreparedFactory<BE, Rg>,
{
    let mut giant_steps = Vec::with_capacity(factor.giant_steps.len());
    for giant in factor.giant_steps {
        let mut diagonals = Vec::with_capacity(giant.diagonals.len());
        for diagonal in giant.diagonals {
            diagonals.push(LinearTransformationDiagonal {
                baby: diagonal.baby,
                plaintext: module.ckks_right_operand_prepare_alloc(&diagonal.plaintext, PrepareHint::Reuse, scratch)?,
            });
        }
        giant_steps.push(LinearTransformationGiantStep {
            rot: giant.rot,
            diagonals,
        });
    }
    Ok(LinearTransformation {
        baby_steps: factor.baby_steps,
        giant_steps,
        rotations: factor.rotations,
    })
}
