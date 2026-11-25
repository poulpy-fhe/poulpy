//! Bootstrapping parameterization: [`BootstrappingPlan`] and its compiled form,
//! [`BootstrappingContextPrepared`], built once and reused across bootstraps.
//!
//! Torus widths (`base2k`, `log_delta`, the input and raised moduli) are not part of a plan: they stay per-ciphertext
//! metadata, read at call time.

use poulpy_core::layouts::{GLWELayout, Rank};

use crate::{CKKSCtBounds, CKKSLayout, GLWEPlaintextMeta};

mod context;
mod eval_mod_ci;
mod key_widths;
mod masks;
mod plan;
mod run;

#[allow(deprecated)]
pub use context::BootstrappingContextUnprepared;
pub use context::{BootstrappingContext, BootstrappingContextPrepared};
pub(crate) use eval_mod_ci::CIPreForm;
pub use eval_mod_ci::{CIConversion, CIPrePrepared, EvalModCIPrepared, EvalModRing};
pub use key_widths::BootstrappingKeyWidths;
pub use masks::{CoeffsMaskPrepared, MaskedFactors, MergedS2CFactorPrepared};
pub use plan::{BootstrappingPipeline, BootstrappingPlan, BootstrappingTechniques, EvalRoundPlus, SparseSecretEncapsulation};
pub use run::BootstrappingRun;

/// The working ciphertext of a bootstrap into `ct_out` and its input, at the radix of the input.
pub(crate) fn bootstrap_layouts<C1, C2>(ct_out: &C1, ct_in: &C2) -> (CKKSLayout, CKKSLayout)
where
    C1: CKKSCtBounds,
    C2: CKKSCtBounds,
{
    let base2k = ct_in.base2k();
    let boot_layout = CKKSLayout {
        glwe_layout: GLWELayout {
            n: ct_out.n(),
            base2k,
            k: ct_out.k(),
            rank: Rank(1),
        },
        // `log_delta = 0` maximizes `log_budget`, which upper-bounds the
        // EvalMod working width.
        meta: GLWEPlaintextMeta::default(),
    };
    let in_layout = CKKSLayout {
        glwe_layout: GLWELayout {
            n: ct_in.n(),
            base2k,
            k: ct_in.k(),
            rank: Rank(1),
        },
        meta: GLWEPlaintextMeta::default(),
    };
    (boot_layout, in_layout)
}
