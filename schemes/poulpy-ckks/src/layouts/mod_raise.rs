//! The modulus raise of a bootstrap as a step of its own: what the plan fixes and the first factor it may carry.

use poulpy_core::layouts::LinearTransformation;
use poulpy_hal::layouts::Backend;

use crate::layouts::{BootstrappingContextPrepared, CKKSRightOperandPrepared, EvalModPlan};

/// What the modulus raise of a [`BootstrappingPlan`](crate::layouts::BootstrappingPlan) fixes.
#[derive(Clone, Debug, PartialEq)]
pub struct ModRaisePlan {
    /// EvalMod's lifts, C2S-first; `None` on S2C-first, whose raise takes `c2s_guard_bits`.
    pub lift: Option<EvalModPlan>,
    pub c2s_guard_bits: usize,
    pub sparse_secret_hamming_weight: Option<usize>,
    /// The plan moves the first CoeffsToSlots factor onto the raise, which then takes it from a [`ModRaisePrepared`].
    pub first_factor_on_raise: bool,
    /// `log2` of the gap of the plaintexts, see
    /// [`BootstrappingPlan::with_log_gap`](crate::layouts::BootstrappingPlan::with_log_gap).
    pub log_gap: usize,
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

    /// `log2` of the gap of the plaintexts, see
    /// [`BootstrappingPlan::with_log_gap`](crate::layouts::BootstrappingPlan::with_log_gap).
    fn log_gap(&self) -> usize {
        self.mod_raise_plan().log_gap
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
