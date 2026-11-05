//! The modulus raise of a bootstrap as a step of its own: what the plan fixes and the first factor it may carry.

use poulpy_core::layouts::LinearTransformation;
use poulpy_hal::layouts::Backend;

use crate::layouts::{BootstrappingContextPrepared, CKKSRightOperandPrepared, EvalModPlan};

/// What the modulus raise of a [`BootstrappingPlan`](crate::layouts::BootstrappingPlan) fixes. The raise is a digit
/// shift in the base-`2^base2k` torus, not an RNS basis extension: it leaves `I(X)·q + Δ·m` in `res`, which keeps its
/// `k`, so `log_budget_out = res.k() - log_delta_out`, and takes the slot kind and sparsity of `a`.
///
/// The order is the first lift, the dense-to-sparse switch, ModUp with the second lift or the guard bits in its shift,
/// the first CoeffsToSlots factor, then the sparse-to-dense switch. Without a lift or encapsulation, `a` is only read.
#[derive(Clone, Debug, PartialEq)]
pub struct ModRaisePlan {
    /// C2S-first: EvalMod's plan, whose two lifts make the message the residue of the raise and `I(X)·q` its integer
    /// part. The first multiplies `a` in place, up to the plan's message ratio, and the second is fused into the shift
    /// of ModUp, so `log_delta_out = a.log_delta() + lifts + log_msg_ratio`. `None` on S2C-first.
    pub lift: Option<EvalModPlan>,
    /// S2C-first: bits fused into the shift of ModUp, so `log_delta_out = a.k() + c2s_guard_bits`.
    pub c2s_guard_bits: usize,
    /// Weight of the sparse secret of
    /// [`SparseSecretEncapsulation`](crate::layouts::SparseSecretEncapsulation): `a` is switched to it in place before
    /// ModUp, and `res` back to the dense secret after it.
    pub sparse_secret_hamming_weight: Option<usize>,
    /// The first CoeffsToSlots factor runs on the raise, while the ciphertext is small, from a [`ModRaisePrepared`]. It
    /// consumes no modulus, and its switches replace the sparse-to-dense one.
    pub first_factor_on_raise: bool,
}

/// A [`ModRaisePlan`] with the first CoeffsToSlots factor its raise applies, if the plan leaves it there.
pub struct ModRaisePrepared<BE: Backend> {
    pub(crate) plan: ModRaisePlan,
    pub(crate) first_factor: Option<LinearTransformation<CKKSRightOperandPrepared<BE>>>,
}

/// Read access to the plan of a modulus raise and to its first factor.
pub trait CKKSModRaiseInfos<BE: Backend> {
    fn mod_raise_plan(&self) -> &ModRaisePlan;
    fn first_factor(&self) -> Option<&LinearTransformation<CKKSRightOperandPrepared<BE>>>;
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
